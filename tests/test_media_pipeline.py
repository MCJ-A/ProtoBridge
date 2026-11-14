from __future__ import annotations

import base64
from http.server import ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import threading
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ignait.config import AppConfig
from ignait.diarization.speaker_id import OnlineSpeakerTracker
from ignait.gui import CONTROLLER, StitchHTTPHandler
from ignait.media.session import (
    compute_auto_paced_speed,
    sanitize_media_text,
    update_ewma_latency,
)
from ignait.media.speaker_bank import (
    FEMALE_PRESET_POOL,
    MALE_PRESET_POOL,
    MediaSpeakerVoiceBank,
)
from ignait.media.stream_buffer import InMemoryAudioRingBuffer
from ignait.media.ws_protocol import (
    OPCODE_BINARY,
    OPCODE_CLOSE,
    OPCODE_CONTINUATION,
    OPCODE_PING,
    OPCODE_PONG,
    OPCODE_TEXT,
    WebSocketConnection,
    compute_ws_accept_key,
    pack_media_binary_packet,
    unpack_media_binary_packet,
)
from ignait.paths import get_models_dir


def _build_masked_client_frame(
    opcode: int, payload: bytes, fin: bool = True, mask: bytes = b"\x37\xfa\x21\x3d"
) -> bytes:
    b1 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    length = len(payload)
    if length <= 125:
        hdr = bytes((b1, 0x80 | length))
    elif length <= 0xFFFF:
        hdr = struct.pack("!BBH", b1, 0x80 | 126, length)
    else:
        hdr = struct.pack("!BBQ", b1, 0x80 | 127, length)
    masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    return hdr + mask + masked


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("Socket cerrado antes de tiempo")
        out.extend(chunk)
    return bytes(out)


def _read_server_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    b1, b2 = _recv_exact(sock, 2)
    opcode = b1 & 0x0F
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", _recv_exact(sock, 2))
    elif length == 127:
        (length,) = struct.unpack("!Q", _recv_exact(sock, 8))
    payload = _recv_exact(sock, length) if length > 0 else b""
    return opcode, payload


class TestMediaPipelineComponents(unittest.TestCase):
    def test_ws_accept_key_and_binary_packet_roundtrip(self) -> None:
        self.assertEqual(
            compute_ws_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
        )
        meta = {"type": "tts_chunk", "speaker_label": "Compañero #2", "sample_rate": 24000}
        pcm = (np.arange(128, dtype=np.int16) - 64).astype("<i2").tobytes()
        pkt = pack_media_binary_packet(meta, pcm)
        got_meta, got_pcm = unpack_media_binary_packet(pkt)
        self.assertEqual(got_meta, meta)
        self.assertEqual(got_pcm, pcm)

    def test_ws_connection_masked_fragmented_ping_and_close(self) -> None:
        srv_sock, cli_sock = socket.socketpair()
        srv_sock.settimeout(3.0)
        cli_sock.settimeout(3.0)
        rfile = srv_sock.makefile("rb")
        wfile = srv_sock.makefile("wb")
        ws = WebSocketConnection(rfile, wfile)
        try:
            # 1) PING del cliente -> servidor responde PONG automático y luego entrega el mensaje de texto fragmentado
            cli_sock.sendall(_build_masked_client_frame(OPCODE_PING, b"hb"))
            cli_sock.sendall(_build_masked_client_frame(OPCODE_TEXT, b'{"hello":', fin=False))
            cli_sock.sendall(
                _build_masked_client_frame(OPCODE_CONTINUATION, b'"world"}', fin=True)
            )
            msg = ws.read_message()
            self.assertEqual(msg, (OPCODE_TEXT, b'{"hello":"world"}'))

            op_pong, pl_pong = _read_server_ws_frame(cli_sock)
            self.assertEqual(op_pong, OPCODE_PONG)
            self.assertEqual(pl_pong, b"hb")

            # 2) Envío binario desde servidor
            ws.send_media_packet({"type": "tts_chunk", "seq": 7}, b"\x01\x02\x03\x04")
            op_bin, pl_bin = _read_server_ws_frame(cli_sock)
            self.assertEqual(op_bin, OPCODE_BINARY)
            hdr, pcm = unpack_media_binary_packet(pl_bin)
            self.assertEqual(hdr["seq"], 7)
            self.assertEqual(pcm, b"\x01\x02\x03\x04")

            # 3) CLOSE limpio desde cliente
            cli_sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
            self.assertIsNone(ws.read_message())
            self.assertTrue(ws.closed)
        finally:
            ws.close()
            for obj in (rfile, wfile, srv_sock, cli_sock):
                try:
                    obj.close()
                except Exception:
                    pass

    def test_in_memory_audio_ring_buffer_wraparound_and_preroll(self) -> None:
        rb = InMemoryAudioRingBuffer(capacity_seconds=0.1, sample_rate=1000)  # 512 min capacity
        self.assertGreaterEqual(rb.capacity, 512)
        seq = np.arange(700, dtype=np.float32)
        rb.write(seq[:400])
        rb.write(seq[400:])
        self.assertEqual(rb.available_to_read, rb.capacity)
        win = rb.slice_window(650, 700, preroll_samples=50)
        self.assertTrue(np.allclose(win, seq[600:700]))
        out = np.empty((100,), dtype=np.float32)
        n = rb.read_into(out)
        self.assertEqual(n, 100)
        rb.clear()
        self.assertEqual(rb.available_to_read, 0)

    def test_media_speaker_voice_bank_v1_v2_and_resets(self) -> None:
        bank = MediaSpeakerVoiceBank(session_prefix="test")
        # Audio silencioso no debe contar hacia clonado
        silence = np.zeros((16000 * 3,), dtype=np.float32)
        k0, preset0, ver0, ref0 = bank.ingest_segment(0, silence, pitch_hz=120.0)
        self.assertEqual(k0, "test:spk_0")
        self.assertEqual(ver0, 0)
        self.assertIsNone(ref0)

        # Clip limpio de 1.5s aún no dispara v1 (<2.2s)
        t1 = np.arange(int(16000 * 1.5), dtype=np.float32) / 16000.0
        tone = (np.sin(2.0 * np.pi * 210.0 * t1) * 0.15).astype(np.float32)
        _, _, ver1, ref1 = bank.ingest_segment(0, tone, pitch_hz=210.0)
        self.assertEqual(ver1, 0)
        self.assertIsNone(ref1)
        self.assertNotEqual(preset0, "invalid")

        # Segundo clip limpio de 1.2s -> total 2.7s >= 2.2s => v1
        tone2 = tone[: int(16000 * 1.2)]
        k2, _, ver2, ref2 = bank.ingest_segment(0, tone2, pitch_hz=210.0)
        self.assertEqual(k2, "test:spk_0")
        self.assertEqual(ver2, 1)
        self.assertIsNotNone(ref2)
        self.assertGreaterEqual(len(ref2), int(16000 * 2.2))

        bank.mark_clone_ready(0)
        self.assertTrue(bank.is_clone_ready(0))

        # Refinamiento hasta superar 5.2s => v2 una sola vez
        big = np.concatenate([tone, tone])
        _, _, ver3, ref3 = bank.ingest_segment(0, big, pitch_hz=210.0)
        self.assertEqual(ver3, 2)
        self.assertIsNotNone(ref3)

        _, _, ver4, _ = bank.ingest_segment(0, tone, pitch_hz=210.0)
        self.assertEqual(ver4, 0)

        bank.reset()
        self.assertFalse(bank.is_clone_ready(0))

    def test_online_speaker_tracker_extended_pool(self) -> None:
        tracker = OnlineSpeakerTracker(
            models_dir=get_models_dir(),
            similarity_threshold=0.9999,  # fuerza asignar un hablante nuevo por embedding distinto
            max_speakers=6,
            limit_to_pool=False,
        )
        t = np.arange(16000, dtype=np.float32) / 16000.0
        profiles = []
        for idx, freq in enumerate((110.0, 145.0, 185.0, 225.0, 265.0, 310.0)):
            sig = (0.15 * np.sin(2.0 * np.pi * freq * t) + 0.05 * np.sin(2.0 * np.pi * freq * 2.0 * t)).astype(np.float32)
            with patch.object(
                tracker,
                "_extract_embedding",
                side_effect=lambda _arr, i=idx: np.eye(1, 512, k=i * 20, dtype=np.float32)[0],
            ):
                prof = tracker.identify_speaker(sig)
                profiles.append(prof)
        self.assertEqual(len({p.speaker_id for p in profiles}), 6)
        self.assertEqual(profiles[4].label, "Companero #5")
        self.assertEqual(profiles[5].label, "Companero #6")

    def test_media_pipeline_ws_http_and_xtts_fallback(self) -> None:
        manifest_path = Path(__file__).resolve().parent.parent / "browser_extension" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        war = manifest.get("web_accessible_resources", [])
        self.assertTrue(any("pcm_worklet.js" in entry.get("resources", []) for entry in war))
        cs_entries = manifest.get("content_scripts", [])
        self.assertTrue(
            any(
                "yt_page_bridge.js" in entry.get("js", []) and entry.get("world") == "MAIN"
                for entry in cs_entries
            )
        )

        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        sock: socket.socket | None = None
        try:
            # 1) /api/media/ping expone metadata XTTS/WS/AutoSync
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
            conn.request(
                "GET",
                "/api/media/ping",
                headers={"Host": f"127.0.0.1:{port}", "Origin": ext_origin},
            )
            resp = conn.getresponse()
            ping_data = json.loads(resp.read().decode("utf-8"))
            conn.close()
            self.assertEqual(resp.status, 200)
            self.assertEqual(ping_data["ws_path"], "/ws/media")
            self.assertIn("xtts_available", ping_data)
            self.assertIn("xtts_running", ping_data)
            self.assertIn("media_auto_sync", ping_data)

            # 2) Rechazo CSRF de WebSocket /ws/media con Origin externo
            bad_sock = socket.create_connection(("127.0.0.1", port), timeout=3.0)
            bad_sock.sendall(
                (
                    f"GET /ws/media HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    f"Origin: https://evil.com\r\n"
                    f"Upgrade: websocket\r\n"
                    f"Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                    f"Sec-WebSocket-Version: 13\r\n\r\n"
                ).encode("ascii")
            )
            bad_resp = bad_sock.recv(1024).decode("latin-1", "replace")
            bad_sock.close()
            self.assertIn("403", bad_resp)

            # 3) Sesión /ws/media válida con XTTS streaming mockeado + fallback a Kokoro
            fake_stt = MagicMock()
            fake_stt.transcribe_any.return_value = ("Hold position.", "en", 0.99)
            fake_mt = MagicMock()
            fake_mt.translate.side_effect = ["Mantengan posición.", "Segunda frase de respaldo."]
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.2, 24000)

            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = True
            fake_xtts.synthesize_stream.side_effect = [
                iter(
                    [
                        (np.ones((1200,), dtype=np.float32) * 0.15, 24000),
                        (np.ones((1200,), dtype=np.float32) * 0.12, 24000),
                    ]
                ),
                RuntimeError("Simulated XTTS timeout"),
            ]

            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
            ):
                sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                sock.sendall(
                    (
                        f"GET /ws/media HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Origin: {ext_origin}\r\n"
                        f"Upgrade: websocket\r\n"
                        f"Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        f"Sec-WebSocket-Version: 13\r\n\r\n"
                    ).encode("ascii")
                )
                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    buf.extend(sock.recv(1024))
                hdr_raw = bytes(buf).decode("latin-1")
                self.assertIn("101 Switching Protocols", hdr_raw)
                self.assertIn("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", hdr_raw)

                # Leer hello inicial
                op_hello, pl_hello = _read_server_ws_frame(sock)
                self.assertEqual(op_hello, OPCODE_TEXT)
                hello = json.loads(pl_hello.decode("utf-8"))
                self.assertEqual(hello["type"], "hello")
                self.assertTrue(hello["xtts_available"])
                self.assertIn("recommended_delay_ms", hello)

                # Configurar XTTS y enviar 1a frase por subtítulo -> 2 chunks XTTS + is_final + sync_telemetry
                cfg_msg = json.dumps(
                    {
                        "type": "configure",
                        "source_lang": "en",
                        "target_lang": "es",
                        "voice": "auto",
                        "tts_backend": "xtts_v2",
                    }
                ).encode("utf-8")
                sock.sendall(_build_masked_client_frame(OPCODE_TEXT, cfg_msg))

                cap1 = json.dumps(
                    {"type": "translate_caption", "text": "Hold position.", "play_at_sec": 12.5}
                ).encode("utf-8")
                sock.sendall(_build_masked_client_frame(OPCODE_TEXT, cap1))

                # Recoger 3 paquetes binarios (chunk 0, chunk 1, final) + sync_telemetry
                packets_1: list[tuple[dict[str, Any], bytes]] = []
                for _ in range(3):
                    op, pl = _read_server_ws_frame(sock)
                    self.assertEqual(op, OPCODE_BINARY)
                    packets_1.append(unpack_media_binary_packet(pl))

                self.assertEqual(packets_1[0][0]["tts_engine"], "xtts_v2")
                self.assertEqual(packets_1[0][0]["play_at_sec"], 12.5)
                self.assertFalse(packets_1[0][0]["is_final"])
                self.assertGreater(len(packets_1[0][1]), 0)
                self.assertIn("recommended_delay_ms", packets_1[0][0])
                self.assertIn("effective_speed", packets_1[0][0])
                self.assertTrue(packets_1[2][0]["is_final"])

                op_tel1, pl_tel1 = _read_server_ws_frame(sock)
                self.assertEqual(op_tel1, OPCODE_TEXT)
                tel1 = json.loads(pl_tel1.decode("utf-8"))
                self.assertEqual(tel1["type"], "sync_telemetry")
                self.assertGreaterEqual(tel1["recommended_delay_ms"], 750)

                # Enviar 2a frase -> XTTS falla y hace fallback automático a Kokoro
                cap2 = json.dumps(
                    {"type": "translate_caption", "text": "Fallback line now."}
                ).encode("utf-8")
                sock.sendall(_build_masked_client_frame(OPCODE_TEXT, cap2))

                op_fb, pl_fb = _read_server_ws_frame(sock)
                self.assertEqual(op_fb, OPCODE_BINARY)
                meta_fb, pcm_fb = unpack_media_binary_packet(pl_fb)
                self.assertEqual(meta_fb["tts_engine"], "kokoro")
                self.assertGreater(len(pcm_fb), 0)
                op_end, pl_end = _read_server_ws_frame(sock)
                self.assertEqual(op_end, OPCODE_BINARY)
                self.assertTrue(unpack_media_binary_packet(pl_end)[0]["is_final"])
                op_tel2, pl_tel2 = _read_server_ws_frame(sock)
                self.assertEqual(op_tel2, OPCODE_TEXT)
                self.assertEqual(json.loads(pl_tel2.decode("utf-8"))["type"], "sync_telemetry")

                sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
                sock.close()
        finally:
            CONTROLLER.config = orig_cfg
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_auto_sync_pacing_and_ewma_helpers(self) -> None:
        # Dentro de presupuesto: mantiene velocidad base
        self.assertAlmostEqual(compute_auto_paced_speed("Hola mundo.", 3.0, 1.05), 1.05, places=3)
        # Frase extensa en ventana corta: acelera con tope 1.32x
        boosted = compute_auto_paced_speed(
            "Esta es una frase bastante larga que necesita compresion automatica para caber en la escena.",
            1.4,
            1.0,
        )
        self.assertGreater(boosted, 1.15)
        self.assertLessEqual(boosted, 1.32)
        # Ventana ausente o muy corta: respeta base
        self.assertAlmostEqual(compute_auto_paced_speed("Texto largo cualquiera.", None, 1.1), 1.1, places=3)

        # EWMA asimétrico: sube rápido, baja lento y acota rango [750, 4500] ms
        ewma1, rec1 = update_ewma_latency(None, 1200)
        self.assertEqual(ewma1, 1200.0)
        self.assertEqual(rec1, int(round(1200.0 * 1.25 + 280.0)))

        ewma_up, _ = update_ewma_latency(1000.0, 2000)
        self.assertAlmostEqual(ewma_up, 1450.0, places=2)  # alpha=0.45
        ewma_down, _ = update_ewma_latency(2000.0, 1000)
        self.assertAlmostEqual(ewma_down, 1840.0, places=2)  # alpha=0.16

        _, min_rec = update_ewma_latency(None, 100)
        _, max_rec = update_ewma_latency(None, 5000)
        self.assertEqual(min_rec, 750)
        self.assertEqual(max_rec, 4500)

    def test_media_ws_auto_sync_clock_auto_pacing_and_stale_drop(self) -> None:
        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        sock: socket.socket | None = None
        try:
            fake_stt = MagicMock()
            fake_mt = MagicMock()
            long_translation = (
                "Esta es una traduccion deliberadamente bastante extensa para forzar compresion automatica de ritmo."
            )
            fake_mt.translate.return_value = long_translation
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.2, 24000)
            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = False

            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
            ):
                sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                sock.sendall(
                    (
                        f"GET /ws/media HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Origin: {ext_origin}\r\n"
                        f"Upgrade: websocket\r\n"
                        f"Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        f"Sec-WebSocket-Version: 13\r\n\r\n"
                    ).encode("ascii")
                )
                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    buf.extend(sock.recv(1024))

                op_hello, pl_hello = _read_server_ws_frame(sock)
                self.assertEqual(op_hello, OPCODE_TEXT)
                self.assertEqual(json.loads(pl_hello.decode("utf-8"))["type"], "hello")

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "configure",
                                "source_lang": "en",
                                "target_lang": "es",
                                "voice": "ef_dora",
                                "speed": 1.0,
                                "tts_backend": "kokoro",
                                "auto_sync": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                # Adelantar reloj de video a t=100.0s antes de encolar subtítulos
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {"type": "media_clock", "current_time_sec": 100.0, "delay_sec": 0.0}
                        ).encode("utf-8"),
                    )
                )
                # Subtítulo vencido (play_at=50.0s + dur=2.0s + 1.5s < 100.0s) -> debe descartarse sin MT/TTS
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 10,
                                "text": "Old already played subtitle sentence.",
                                "play_at_sec": 50.0,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                # Subtítulo futuro dentro de horizonte -> debe sintetizarse con auto-pacing (>1.05x)
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 11,
                                "text": "Fresh upcoming line.",
                                "play_at_sec": 104.0,
                                "duration_sec": 1.5,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )

                op_chunk, pl_chunk = _read_server_ws_frame(sock)
                self.assertEqual(op_chunk, OPCODE_BINARY)
                meta_chunk, _ = unpack_media_binary_packet(pl_chunk)
                self.assertEqual(meta_chunk["seq"], 11)
                self.assertTrue(meta_chunk["is_prefetched"])
                self.assertEqual(meta_chunk["play_at_sec"], 104.0)
                self.assertGreater(meta_chunk["effective_speed"], 1.05)
                self.assertEqual(fake_mt.translate.call_count, 1)

                op_end, pl_end = _read_server_ws_frame(sock)
                self.assertEqual(op_end, OPCODE_BINARY)
                self.assertTrue(unpack_media_binary_packet(pl_end)[0]["is_final"])

                op_tel, pl_tel = _read_server_ws_frame(sock)
                self.assertEqual(op_tel, OPCODE_TEXT)
                tel = json.loads(pl_tel.decode("utf-8"))
                self.assertEqual(tel["type"], "sync_telemetry")
                self.assertEqual(tel["seq"], 11)

                sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
                sock.close()
        finally:
            CONTROLLER.config = orig_cfg
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_speaker_gender_slot_no_female_overflow_and_bank_refresh(self) -> None:
        # 5 hablantes masculinos distintos -> ninguno debe caer al slot femenino ef_dora
        tracker = OnlineSpeakerTracker(
            models_dir=get_models_dir(),
            similarity_threshold=0.9999,
            max_speakers=6,
            limit_to_pool=False,
        )
        t = np.arange(16000, dtype=np.float32) / 16000.0
        profs = []
        for idx, freq in enumerate((108.0, 118.0, 126.0, 134.0, 142.0)):
            sig = (0.18 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
            with patch.object(
                tracker,
                "_extract_embedding",
                side_effect=lambda _arr, i=idx: np.eye(1, 512, k=i * 17, dtype=np.float32)[0],
            ):
                profs.append(tracker.identify_speaker(sig))
        self.assertTrue(all(p.voice_name != "ef_dora" for p in profs))

        # Si el 1er segmento corto dio pico agudo (ef_dora), el suavizado F0 masculino lo corrige
        tracker.reset_squad()
        fixed_emb = np.eye(1, 512, k=3, dtype=np.float32)[0]
        high_sig = (0.18 * np.sin(2.0 * np.pi * 205.0 * t)).astype(np.float32)
        male_sig = (0.18 * np.sin(2.0 * np.pi * 112.0 * t)).astype(np.float32)
        with patch.object(tracker, "_extract_embedding", return_value=fixed_emb):
            p_init = tracker.identify_speaker(high_sig)
            self.assertEqual(p_init.voice_name, "ef_dora")
            for _ in range(5):
                p_cur = tracker.identify_speaker(male_sig)
        self.assertLess(p_cur.pitch_hz, 168.0)
        self.assertNotEqual(p_cur.voice_name, "ef_dora")

        # MediaSpeakerVoiceBank refresca preset pre-clonado y acepta audio moderado (-40 dBFS)
        bank = MediaSpeakerVoiceBank(session_prefix="gender")
        quiet_male = (0.01 * np.sin(2.0 * np.pi * 115.0 * np.arange(int(16000 * 2.4), dtype=np.float32) / 16000.0)).astype(np.float32)
        _, p_fem, v0, _ = bank.ingest_segment(0, quiet_male[:16000], pitch_hz=195.0)
        self.assertIn(p_fem, FEMALE_PRESET_POOL)
        self.assertEqual(v0, 0)
        _, p_male, v1, ref1 = bank.ingest_segment(0, quiet_male, pitch_hz=115.0)
        self.assertIn(p_male, MALE_PRESET_POOL)
        self.assertEqual(v1, 1)
        self.assertIsNotNone(ref1)

    def test_media_ws_voice_probe_only_clones_and_genders_prefetched_captions(self) -> None:
        import time as _time

        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        sock: socket.socket | None = None
        try:
            fake_stt = MagicMock()
            fake_mt = MagicMock()
            fake_mt.translate.return_value = "Buenos días a todos."
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.2, 24000)

            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = True
            fake_xtts.register_speaker_async.side_effect = (
                lambda key, pcm, on_done=None: on_done(True) if on_done else None
            )
            fake_xtts.synthesize_stream.return_value = iter(
                [(np.ones((1200,), dtype=np.float32) * 0.15, 24000)]
            )

            t_sig = np.arange(int(16000 * 2.5), dtype=np.float32) / 16000.0
            male_seg = (
                0.20 * np.sin(2.0 * np.pi * 115.0 * t_sig)
                + 0.15 * np.sin(2.0 * np.pi * 230.0 * t_sig)
            ).astype(np.float32)

            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
                patch(
                    "ignait.vad.silero.SileroVADSegmenter.process_available",
                    side_effect=[[male_seg], [], [], []],
                ),
            ):
                sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                sock.sendall(
                    (
                        f"GET /ws/media HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Origin: {ext_origin}\r\n"
                        f"Upgrade: websocket\r\n"
                        f"Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        f"Sec-WebSocket-Version: 13\r\n\r\n"
                    ).encode("ascii")
                )
                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    buf.extend(sock.recv(1024))

                op_h, _ = _read_server_ws_frame(sock)
                self.assertEqual(op_h, OPCODE_TEXT)

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "configure",
                                "source_lang": "en",
                                "target_lang": "es",
                                "voice": "auto",
                                "tts_backend": "xtts_v2",
                                "diarization": True,
                                "voice_probe_only": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                # Enviar audio en modo solo-probe -> debe registrar voz masculina sin llamar STT/MT
                pcm_probe = (np.ones((512,), dtype=np.float32) * 0.1 * 32767.0).astype("<i2").tobytes()
                sock.sendall(_build_masked_client_frame(OPCODE_BINARY, pcm_probe))

                deadline = _time.monotonic() + 3.0
                while _time.monotonic() < deadline:
                    if CONTROLLER._media_service.speaker_bank.is_clone_ready(0):
                        break
                    _time.sleep(0.02)

                self.assertTrue(CONTROLLER._media_service.speaker_bank.is_clone_ready(0))
                self.assertEqual(fake_stt.transcribe_any.call_count, 0)

                # Ahora enviar subtítulo prefetched -> debe heredar speaker_key clonado y preset masculino
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 1,
                                "text": "Good morning everyone.",
                                "play_at_sec": 10.0,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )

                op_c, pl_c = _read_server_ws_frame(sock)
                self.assertEqual(op_c, OPCODE_BINARY)
                meta_c, _ = unpack_media_binary_packet(pl_c)
                self.assertEqual(meta_c["tts_engine"], "xtts_v2")
                self.assertEqual(meta_c["speaker_id"], 0)
                op_e, _ = _read_server_ws_frame(sock)
                self.assertEqual(op_e, OPCODE_BINARY)
                op_t, _ = _read_server_ws_frame(sock)
                self.assertEqual(op_t, OPCODE_TEXT)

                kwargs = fake_xtts.synthesize_stream.call_args.kwargs
                self.assertEqual(kwargs["speaker_key"], "media:spk_0")
                self.assertIn(kwargs["fallback_speaker"], MALE_PRESET_POOL)
                self.assertEqual(fake_stt.transcribe_any.call_count, 0)

                sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
                sock.close()
        finally:
            CONTROLLER.config = orig_cfg
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_media_ws_seek_backward_resets_clock_dedup_and_playback_gen(self) -> None:
        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        entered_stale_mt = threading.Event()
        release_stale_mt = threading.Event()
        try:
            fake_stt = MagicMock()
            fake_mt = MagicMock()

            def _translate(txt: str, src_lang: str = "en", tgt_lang: str = "es") -> str:
                if "blocking" in txt.lower():
                    entered_stale_mt.set()
                    release_stale_mt.wait(timeout=3.0)
                return f"TR:{txt}"

            fake_mt.translate.side_effect = _translate
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.15, 24000)
            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = False

            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
            ):
                sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                sock.sendall(
                    (
                        f"GET /ws/media HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Origin: {ext_origin}\r\n"
                        f"Upgrade: websocket\r\n"
                        f"Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        f"Sec-WebSocket-Version: 13\r\n\r\n"
                    ).encode("ascii")
                )
                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    buf.extend(sock.recv(1024))

                op_h, _ = _read_server_ws_frame(sock)
                self.assertEqual(op_h, OPCODE_TEXT)

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "configure",
                                "source_lang": "en",
                                "target_lang": "es",
                                "voice": "ef_dora",
                                "speed": 1.0,
                                "tts_backend": "kokoro",
                                "auto_sync": True,
                                "playback_gen": 0,
                            }
                        ).encode("utf-8"),
                    )
                )
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps({"type": "media_clock", "current_time_sec": 420.0, "delay_sec": 0.0}).encode(
                            "utf-8"
                        ),
                    )
                )
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 1001,
                                "playback_gen": 0,
                                "text": "Repeated caption line.",
                                "play_at_sec": 421.0,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )

                op1, pl1 = _read_server_ws_frame(sock)
                self.assertEqual(op1, OPCODE_BINARY)
                meta1, _ = unpack_media_binary_packet(pl1)
                self.assertEqual(meta1["seq"], 1001)
                self.assertEqual(meta1["playback_gen"], 0)
                _read_server_ws_frame(sock)  # is_final
                _read_server_ws_frame(sock)  # sync_telemetry

                # Encolar tarea bloqueada en MT antes de saltar atrás a t=20s
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 1002,
                                "playback_gen": 0,
                                "text": "Pre-seek blocking sentence.",
                                "play_at_sec": 423.0,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                self.assertTrue(entered_stale_mt.wait(timeout=3.0))

                # Salto atrás a t=20.0s con playback_gen=1
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "reset_playback",
                                "playback_gen": 1,
                                "current_time_sec": 20.0,
                            }
                        ).encode("utf-8"),
                    )
                )
                release_stale_mt.set()

                # Petición rezagada con playback_gen viejo (0) debe ignorarse
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 1003,
                                "playback_gen": 0,
                                "text": "Late old-gen caption.",
                                "play_at_sec": 424.0,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                # Mismo texto ya escuchado anteriormente re-enviado en t=20s (debe sintetizarse, no dedup ni drop por reloj=420s)
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 1004,
                                "playback_gen": 1,
                                "text": "Repeated caption line.",
                                "play_at_sec": 20.5,
                                "duration_sec": 2.0,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )

                op2, pl2 = _read_server_ws_frame(sock)
                self.assertEqual(op2, OPCODE_BINARY)
                meta2, _ = unpack_media_binary_packet(pl2)
                self.assertEqual(meta2["seq"], 1004)
                self.assertEqual(meta2["playback_gen"], 1)
                self.assertEqual(meta2["source_kind"], "caption")

                op2_end, pl2_end = _read_server_ws_frame(sock)
                self.assertEqual(op2_end, OPCODE_BINARY)
                meta2_end, _ = unpack_media_binary_packet(pl2_end)
                self.assertTrue(meta2_end["is_final"])
                self.assertEqual(meta2_end["playback_gen"], 1)

                op2_tel, pl2_tel = _read_server_ws_frame(sock)
                self.assertEqual(op2_tel, OPCODE_TEXT)
                tel2 = json.loads(pl2_tel.decode("utf-8"))
                self.assertEqual(tel2["seq"], 1004)
                self.assertEqual(tel2["playback_gen"], 1)

                # Solo deben haberse sintetizado seq 1001 y seq 1004 (1002 abortada antes de TTS, 1003 descartada)
                self.assertEqual(fake_tts.synthesize.call_count, 2)

                sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
                sock.close()
        finally:
            release_stale_mt.set()
            CONTROLLER.config = orig_cfg
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_extension_seek_guards_in_node(self) -> None:
        node_exe = shutil.which("node")
        if not node_exe:
            self.skipTest("node no disponible en PATH para prueba de extensión")

        ext_dir = Path(__file__).resolve().parents[1] / "browser_extension"
        content_js = ext_dir / "content.js"
        bg_js = ext_dir / "background.js"

        for js_file in (content_js, bg_js, ext_dir / "popup.js"):
            proc = subprocess.run([node_exe, "--check", str(js_file)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        harness = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const sentCmds = [];
const scheduledStarts = [];
const delayUpdates = [];
let portListeners = [];
let videoListeners = {};
let msgListeners = [];
let winListeners = {};

function makeEl() {
  return {
    style: {},
    dataset: {},
    innerHTML: "",
    textContent: "",
    appendChild() {},
    addEventListener() {},
    removeEventListener() {},
    remove() {},
    contains() { return false; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { width: 800, height: 450 }; },
  };
}

const fakeVideo = {
  paused: false,
  ended: false,
  seeking: false,
  currentTime: 420.0,
  textTracks: [],
  getBoundingClientRect() { return { width: 1280, height: 720 }; },
  addEventListener(ev, fn) {
    if (!videoListeners[ev]) videoListeners[ev] = [];
    videoListeners[ev].push(fn);
  },
  emit(ev) {
    for (const fn of (videoListeners[ev] || [])) fn();
  },
};

class FakeBufferSource {
  constructor() {
    this.buffer = null;
    this.playbackRate = { value: 1.0 };
    this.onended = null;
  }
  connect() {}
  stop() {}
  start(at) {
    scheduledStarts.push({ at, rate: this.playbackRate.value });
  }
}

class FakeAudioContext {
  constructor() {
    this.currentTime = 100.0;
    this.state = "running";
    this.destination = {};
    this.audioWorklet = { addModule: () => Promise.reject(new Error("no worklet in test")) };
  }
  resume() { return Promise.resolve(); }
  createMediaElementSource() { return { connect() {} }; }
  createDelay() {
    return {
      delayTime: {
        value: 0,
        setTargetAtTime(v) {
          this.value = v;
          delayUpdates.push(v);
        },
      },
      connect() {},
    };
  }
  createGain() { return { gain: { value: 1, setTargetAtTime() {}, cancelScheduledValues() {} }, connect() {} }; }
  createScriptProcessor() { return { connect() {}, onaudioprocess: null }; }
  createBuffer(_ch, len, rate) {
    const arr = new Float32Array(len);
    return { duration: len / rate, getChannelData() { return arr; } };
  }
  createBufferSource() { return new FakeBufferSource(); }
}

const fakePort = {
  onMessage: { addListener(fn) { portListeners.push(fn); } },
  onDisconnect: { addListener() {} },
  postMessage(m) { sentCmds.push(m); },
  disconnect() {},
  emit(m) { for (const fn of portListeners) fn(m); },
};

const sandbox = {
  console,
  Math,
  Date,
  Int16Array,
  Float32Array,
  Uint8Array,
  ArrayBuffer,
  Map,
  Set,
  WeakMap,
  btoa: (s) => Buffer.from(s, "binary").toString("base64"),
  atob: (s) => Buffer.from(s, "base64").toString("binary"),
  setInterval: () => 0,
  clearInterval: () => {},
  setTimeout: () => 0,
  clearTimeout: () => {},
  MutationObserver: class { observe() {} },
  AudioContext: FakeAudioContext,
  AudioWorkletNode: class {},
  window: {
    location: { hostname: "www.youtube.com", origin: "https://www.youtube.com" },
    AudioContext: FakeAudioContext,
    addEventListener(ev, fn) {
      if (!winListeners[ev]) winListeners[ev] = [];
      winListeners[ev].push(fn);
    },
    postMessage() {},
  },
  document: {
    documentElement: makeEl(),
    body: makeEl(),
    fullscreenElement: null,
    querySelector() { return null; },
    querySelectorAll(sel) { return sel === "video" ? [fakeVideo] : []; },
    createElement: makeEl,
    addEventListener() {},
  },
  chrome: {
    runtime: {
      getURL: (p) => "chrome-extension://id/" + p,
      connect: () => fakePort,
      onMessage: { addListener(fn) { msgListeners.push(fn); } },
    },
    storage: {
      local: {
        get(defaults, cb) {
          cb({ ...defaults, enabled: true, autoSync: true, useCaptions: true });
        },
      },
      onChanged: { addListener() {} },
    },
  },
};
sandbox.window.window = sandbox.window;

const code = fs.readFileSync(process.argv[1], "utf8");
vm.runInNewContext(code, sandbox);

// Conectar WS + entregar subtítulos prefetched alrededor de t=20s y t=420s
fakePort.emit({ type: "ws_open" });
for (const fn of winListeners["message"] || []) {
  fn({
    source: sandbox.window,
    origin: "https://www.youtube.com",
    data: {
      source: "pb-yt-bridge",
      type: "PB_TIMEDTEXT_JSON3",
      videoId: "vid1",
      lang: "en",
      events: [
        { tStartMs: 20500, dDurationMs: 2000, segs: [{ utf8: "Early scene sentence." }] },
        { tStartMs: 421000, dDurationMs: 2000, segs: [{ utf8: "Late scene sentence." }] },
      ],
    },
  });
}

// Simular salto atrás de 420s -> 20s
const pcmBytes = new Uint8Array(new Int16Array(4800).fill(1000).buffer);
const pcmB64 = Buffer.from(pcmBytes).toString("base64");

fakeVideo.currentTime = 20.0;
fakeVideo.seeking = true;
fakeVideo.emit("seeking");

const resets = sentCmds.filter((c) => c.type === "ws_cmd" && c.cmd && c.cmd.type === "reset_playback");
assert(resets.length >= 1, "seeking debe enviar reset_playback");
const seekReset = resets[resets.length - 1].cmd;
assert.strictEqual(seekReset.playback_gen, 1);
assert.strictEqual(seekReset.current_time_sec, 20.0);

fakeVideo.seeking = false;
fakeVideo.emit("seeked");

// 1) Chunk rezagado de playback_gen viejo (0) debe descartarse
fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 1000001,
    playback_gen: 0,
    source_kind: "caption",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    play_at_sec: 421.0,
    duration_sec: 2.0,
    is_prefetched: true,
    translated_text: "Viejo",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 0, "chunk pre-seek con gen viejo no debe programarse");

// 2) Chunk huérfano chunk_index=1 sin chunk_index=0 (incluso sin playback_gen) debe descartarse
fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 99,
    source_kind: "audio",
    chunk_index: 1,
    is_final: false,
    sample_rate: 24000,
    translated_text: "Huerfano",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 0, "chunk_index>0 huérfano no debe reproducirse");

// 3) Chunk futuro venenoso (play_at_sec=421s cuando currentTime=20s) incluso con gen=1 debe descartarse sin envenenar reloj
fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 1000002,
    playback_gen: 1,
    source_kind: "caption",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    play_at_sec: 421.0,
    duration_sec: 2.0,
    is_prefetched: true,
    translated_text: "Veneno futuro",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 0, "chunk a +401s debe descartarse");

// 4) Audio chunk capturado antes de salto atrás (capture_media_sec=420 cuando currentTime=20s) debe descartarse
fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 50,
    playback_gen: 1,
    source_kind: "audio",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    capture_media_sec: 420.0,
    capture_delay_sec: 2.2,
    is_prefetched: false,
    translated_text: "Audio viejo",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 0, "audio pre-seek backward no debe programarse");

// 5) Chunk válido post-salto (t=20.5s, playback_gen=1) debe sonar INMEDIATO (no +400s)
fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 1000003,
    playback_gen: 1,
    source_kind: "caption",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    play_at_sec: 20.5,
    duration_sec: 2.0,
    is_prefetched: true,
    translated_text: "Oracion temprana.",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 1, "chunk post-seek válido debe programarse");
assert(
  scheduledStarts[0].at >= 100.0 && scheduledStarts[0].at < 102.0,
  "startAt debe estar en tiempo real (~100.5s), recibido: " + scheduledStarts[0].at
);

// 6) rx_mute_state desde Ignait Studio silencia doblaje y devuelve delay a 0s; al quitar RX mute reanuda
fakePort.emit({
  type: "ws_json",
  data: { type: "rx_mute_state", rx_muted: true },
});
assert.strictEqual(
  delayUpdates[delayUpdates.length - 1],
  0,
  "mute RX debe devolver delayNode a 0s"
);
let statusMuted = null;
msgListeners[0]({ type: "PB_GET_STATUS" }, null, (r) => {
  statusMuted = r;
});
assert.strictEqual(statusMuted.rxMuted, true);
assert.strictEqual(statusMuted.effectiveDelayMs, 0);

fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 1000004,
    playback_gen: 2,
    source_kind: "caption",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    play_at_sec: 22.0,
    duration_sec: 2.0,
    is_prefetched: true,
    translated_text: "No debe sonar en RX mute.",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 1, "chunk recibido con RX mute no debe programarse");

fakePort.emit({
  type: "ws_json",
  data: { type: "rx_mute_state", rx_muted: false },
});
let statusUnmuted = null;
msgListeners[0]({ type: "PB_GET_STATUS" }, null, (r) => {
  statusUnmuted = r;
});
assert.strictEqual(statusUnmuted.rxMuted, false);

fakePort.emit({
  type: "ws_tts_chunk",
  meta: {
    type: "tts_chunk",
    seq: 1000005,
    playback_gen: 2,
    source_kind: "caption",
    chunk_index: 0,
    is_final: false,
    sample_rate: 24000,
    play_at_sec: 24.0,
    duration_sec: 2.0,
    is_prefetched: true,
    translated_text: "Reanudado tras quitar RX mute.",
  },
  pcmB64,
});
assert.strictEqual(scheduledStarts.length, 2, "chunk válido tras quitar RX mute debe reproducirse");
"""
        proc = subprocess.run(
            [node_exe, "-e", harness, str(content_js)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    def test_channel_mute_sync_controller_media_and_http_endpoints(self) -> None:
        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        sock: socket.socket | None = None
        try:
            with patch.object(AppConfig, "save"):
                CONTROLLER.config.media_mode_enabled = True
                # 1) Solo Escuchar (mute_tx=True, mute_rx=False) -> MediaLookaheadBuffer activo
                state_rx = CONTROLLER.set_channel_mutes(mute_tx=True, mute_rx=False)
                self.assertEqual(state_rx, {"mute_tx": True, "mute_rx": False})
                self.assertTrue(CONTROLLER._hotkeys.is_tx_muted)
                self.assertFalse(CONTROLLER._hotkeys.is_rx_muted)
                self.assertFalse(CONTROLLER._media_service.is_rx_muted)
                self.assertTrue(CONTROLLER._media_buffer.enabled)

                # 2) Solo Hablar (mute_tx=False, mute_rx=True) -> MediaLookaheadBuffer deshabilitado
                state_tx = CONTROLLER.set_channel_mutes(mute_tx=False, mute_rx=True)
                self.assertEqual(state_tx, {"mute_tx": False, "mute_rx": True})
                self.assertFalse(CONTROLLER._hotkeys.is_tx_muted)
                self.assertTrue(CONTROLLER._hotkeys.is_rx_muted)
                self.assertTrue(CONTROLLER._media_service.is_rx_muted)
                self.assertFalse(CONTROLLER._media_buffer.enabled)

                # 3) HTTP translate con RX mute debe cortar inmediatamente sin llamar STT/TTS
                pcm_b64 = base64.b64encode((np.ones((3200,), dtype="<i2") * 800)).decode("ascii")
                fake_stt = MagicMock()
                fake_mt = MagicMock()
                fake_tts = MagicMock()
                with patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)):
                    res_muted = CONTROLLER._media_service.translate_http_payload(
                        {"pcm_s16le_b64": pcm_b64, "sample_rate": 16000}
                    )
                self.assertTrue(res_muted.get("ok"))
                self.assertTrue(res_muted.get("empty"))
                self.assertTrue(res_muted.get("rx_muted"))
                fake_stt.transcribe_any.assert_not_called()
                fake_tts.synthesize.assert_not_called()

                # 4) GET /api/media/ping reporta mute_tx / mute_rx
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "GET",
                    "/api/media/ping",
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": ext_origin,
                        "X-Ignait-Client": "BrowserExtension",
                    },
                )
                resp = conn.getresponse()
                ping = json.loads(resp.read().decode("utf-8"))
                conn.close()
                self.assertEqual(resp.status, 200)
                self.assertFalse(ping.get("mute_tx"))
                self.assertTrue(ping.get("mute_rx"))

                # 5) POST /api/media/channels desde extensión cambia canales y /api/config sigue bloqueado (403)
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/media/channels",
                    body=json.dumps({"mute_tx": True, "mute_rx": False}).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": ext_origin,
                        "X-Ignait-Client": "BrowserExtension",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                ch_body = json.loads(resp.read().decode("utf-8"))
                conn.close()
                self.assertEqual(resp.status, 200)
                self.assertTrue(ch_body.get("mute_tx"))
                self.assertFalse(ch_body.get("mute_rx"))
                self.assertTrue(CONTROLLER._hotkeys.is_tx_muted)
                self.assertFalse(CONTROLLER._media_service.is_rx_muted)

                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/config",
                    body=json.dumps({"mute_tx": False}).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": ext_origin,
                        "X-Ignait-Client": "BrowserExtension",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                _ = resp.read()
                conn.close()
                self.assertEqual(resp.status, 403)

                # 6) POST /api/media/channels rechaza Origin externo -> 403
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/media/channels",
                    body=json.dumps({"mute_tx": False}).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": "https://evil.com",
                        "X-Ignait-Client": "BrowserExtension",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                _ = resp.read()
                conn.close()
                self.assertEqual(resp.status, 403)

                # 7) POST /api/config desde WebUI actualiza hotkey + media service
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/config",
                    body=json.dumps({"mute_tx": False, "mute_rx": True, "silent": True}).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": f"http://127.0.0.1:{port}",
                        "X-Ignait-Client": "WebUI",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                cfg_body = json.loads(resp.read().decode("utf-8"))
                conn.close()
                self.assertEqual(resp.status, 200)
                self.assertFalse(cfg_body.get("mute_tx"))
                self.assertTrue(cfg_body.get("mute_rx"))
                self.assertFalse(CONTROLLER._hotkeys.is_tx_muted)
                self.assertTrue(CONTROLLER._hotkeys.is_rx_muted)
                self.assertTrue(CONTROLLER._media_service.is_rx_muted)
        finally:
            CONTROLLER.config = orig_cfg
            CONTROLLER.set_channel_mutes(orig_cfg.mute_tx, orig_cfg.mute_rx)
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_media_ws_rx_mute_state_drops_and_resumes(self) -> None:
        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        sock: socket.socket | None = None
        try:
            fake_stt = MagicMock()
            fake_stt.transcribe_any.return_value = ("Enemy spotted", "en", 0.99)
            fake_mt = MagicMock()
            fake_mt.translate.side_effect = lambda txt, src_lang="en", tgt_lang="es": f"ES:{txt}"
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.2, 24000)
            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = False

            CONTROLLER.set_channel_mutes(mute_tx=False, mute_rx=True)

            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
            ):
                sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                sock.sendall(
                    (
                        f"GET /ws/media HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Origin: {ext_origin}\r\n"
                        f"Upgrade: websocket\r\n"
                        f"Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                        f"Sec-WebSocket-Version: 13\r\n\r\n"
                    ).encode("ascii")
                )
                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    buf.extend(sock.recv(1024))

                # 1) hello refleja mute_rx inicial = True
                op_h, pl_h = _read_server_ws_frame(sock)
                self.assertEqual(op_h, OPCODE_TEXT)
                hello = json.loads(pl_h.decode("utf-8"))
                self.assertTrue(hello.get("rx_muted"))

                # 2) Desmutear RX ("Solo Escuchar") empuja rx_mute_state=False en tiempo real
                CONTROLLER.set_channel_mutes(mute_tx=True, mute_rx=False)
                op_m1, pl_m1 = _read_server_ws_frame(sock)
                self.assertEqual(op_m1, OPCODE_TEXT)
                self.assertEqual(
                    json.loads(pl_m1.decode("utf-8")),
                    {"type": "rx_mute_state", "rx_muted": False},
                )

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "configure",
                                "source_lang": "en",
                                "target_lang": "es",
                                "voice": "ef_dora",
                                "speed": 1.0,
                                "tts_backend": "kokoro",
                                "auto_sync": True,
                                "playback_gen": 0,
                            }
                        ).encode("utf-8"),
                    )
                )
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 2001,
                                "playback_gen": 0,
                                "text": "Move forward",
                                "play_at_sec": 10.0,
                                "duration_sec": 1.5,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                op_c1, pl_c1 = _read_server_ws_frame(sock)
                self.assertEqual(
                    op_c1,
                    OPCODE_BINARY,
                    msg=f"Se esperaba OPCODE_BINARY para seq=2001, llegó: {pl_c1!r}",
                )
                meta_c1, _ = unpack_media_binary_packet(pl_c1)
                self.assertEqual(meta_c1["seq"], 2001)
                _read_server_ws_frame(sock)  # is_final
                _read_server_ws_frame(sock)  # sync_telemetry
                self.assertEqual(fake_tts.synthesize.call_count, 1)

                # 3) Silenciar RX ("Solo Hablar"): empuja rx_mute_state=True y descarta captions/audio nuevos
                CONTROLLER.set_channel_mutes(mute_tx=False, mute_rx=True)
                op_m2, pl_m2 = _read_server_ws_frame(sock)
                self.assertEqual(op_m2, OPCODE_TEXT)
                self.assertEqual(
                    json.loads(pl_m2.decode("utf-8")),
                    {"type": "rx_mute_state", "rx_muted": True},
                )

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 2002,
                                "playback_gen": 0,
                                "text": "Must be ignored while muted",
                                "play_at_sec": 12.0,
                                "duration_sec": 1.5,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                pcm_bytes = (np.ones((6400,), dtype="<i2") * 900).tobytes()
                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_BINARY,
                        pack_media_binary_packet({"seq": 55, "sample_rate": 16000}, pcm_bytes),
                    )
                )

                # 4) Reactivar RX ("Bidireccional"): recibe rx_mute_state=False y procesa seq 2003
                CONTROLLER.set_channel_mutes(mute_tx=False, mute_rx=False)
                op_m3, pl_m3 = _read_server_ws_frame(sock)
                self.assertEqual(op_m3, OPCODE_TEXT)
                self.assertEqual(
                    json.loads(pl_m3.decode("utf-8")),
                    {"type": "rx_mute_state", "rx_muted": False},
                )

                sock.sendall(
                    _build_masked_client_frame(
                        OPCODE_TEXT,
                        json.dumps(
                            {
                                "type": "translate_caption",
                                "seq": 2003,
                                "playback_gen": 1,
                                "text": "Back online",
                                "play_at_sec": 14.0,
                                "duration_sec": 1.5,
                                "is_prefetched": True,
                            }
                        ).encode("utf-8"),
                    )
                )
                op_c3, pl_c3 = _read_server_ws_frame(sock)
                self.assertEqual(op_c3, OPCODE_BINARY)
                meta_c3, _ = unpack_media_binary_packet(pl_c3)
                self.assertEqual(meta_c3["seq"], 2003)
                _read_server_ws_frame(sock)  # is_final
                _read_server_ws_frame(sock)  # sync_telemetry
                self.assertEqual(fake_tts.synthesize.call_count, 2)

                sock.sendall(_build_masked_client_frame(OPCODE_CLOSE, b"\x03\xe8"))
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            CONTROLLER.config = orig_cfg
            CONTROLLER.set_channel_mutes(orig_cfg.mute_tx, orig_cfg.mute_rx)
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_single_speaker_hysteresis_blocks_false_splits_and_allows_real_multi_speaker(
        self,
    ) -> None:
        # Verificar limpieza anti-alucinación para texto / subtítulos
        self.assertEqual(sanitize_media_text("[Music] ♪ ♪"), "")
        self.assertEqual(sanitize_media_text("(Applause) Hello everyone!"), "Hello everyone!")
        self.assertEqual(sanitize_media_text("no no no no wait please"), "no wait please")

        def _norm(v: NDArray[np.float32]) -> NDArray[np.float32]:
            return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)

        tracker = OnlineSpeakerTracker(
            models_dir=get_models_dir(),
            max_speakers=8,
            similarity_threshold=0.50,
            limit_to_pool=False,
            continuity_bonus=0.06,
            require_new_confirmation=True,
        )
        t_long = np.arange(int(16000 * 1.25), dtype=np.float32) / 16000.0
        t_short = np.arange(int(16000 * 0.50), dtype=np.float32) / 16000.0
        male_long = (0.16 * np.sin(2.0 * np.pi * 118.0 * t_long)).astype(np.float32)
        male_short = (0.16 * np.sin(2.0 * np.pi * 124.0 * t_short)).astype(np.float32)
        fem_long = (0.16 * np.sin(2.0 * np.pi * 212.0 * t_long)).astype(np.float32)

        e0 = _norm(np.eye(1, 512, k=0, dtype=np.float32)[0])
        # Vector con cos=0.47 respecto a e0 -> rescate por histéresis de continuidad (+0.06)
        e0_mild = _norm(0.47 * e0 + np.sqrt(1.0 - 0.47**2) * np.eye(1, 512, k=1, dtype=np.float32)[0])
        # Ruido único borderline cos=0.42 mismo género -> pendiente sin dividir
        e0_glitch = _norm(0.42 * e0 + np.sqrt(1.0 - 0.42**2) * np.eye(1, 512, k=2, dtype=np.float32)[0])
        # Hablante distinto real (ortogonal) -> split inmediato
        e1_fem = _norm(np.eye(1, 512, k=10, dtype=np.float32)[0])
        e2_male = _norm(np.eye(1, 512, k=20, dtype=np.float32)[0])
        # Dos turnos seguidos de hablante borderline (cos~0.43 respecto a e0) -> se promueve en el 2º turno
        e3_cand = _norm(0.43 * e0 + np.sqrt(1.0 - 0.43**2) * np.eye(1, 512, k=30, dtype=np.float32)[0])

        seq = [
            (e0, male_long, True, 0),
            (e0_mild, male_long, False, 0),
            (e0_glitch, male_long, False, 0),
            (e2_male, male_short, False, 0),  # < 0.90s nunca crea un nuevo hablante
            (e0, male_long, True, 0),
            (e1_fem, fem_long, True, 1),
            (e2_male, male_long, True, 2),
            (e0, male_long, True, 0),
            (e3_cand, male_long, False, 0),  # 1er turno borderline pendiente
            (e3_cand, male_long, True, 3),   # 2do turno consecutivo confirma hablante #4 (id=3)
        ]

        bank = MediaSpeakerVoiceBank(session_prefix="hyst")
        for emb_vec, audio_clip, exp_conf, exp_id in seq:
            with patch.object(tracker, "_extract_embedding", return_value=emb_vec):
                prof = tracker.identify_speaker(audio_clip)
            self.assertEqual(prof.speaker_id, exp_id)
            self.assertEqual(tracker.last_was_confident, exp_conf)
            _, _, v, _ = bank.ingest_segment(
                prof.speaker_id, audio_clip, prof.pitch_hz, confidence_ok=tracker.last_was_confident
            )
            if not exp_conf:
                self.assertEqual(v, 0)

    def test_media_single_speaker_mode_clones_and_syncs_v1_cleanly(self) -> None:
        from concurrent.futures import Future

        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        orig_cfg = AppConfig(**vars(CONTROLLER.config))
        try:
            fake_stt = MagicMock()
            fake_stt.transcribe_any.return_value = ("Hello single speaker.", "en", 0.99)
            fake_mt = MagicMock()
            fake_mt.translate.return_value = "Hola hablante único."
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.2, 24000)

            fake_xtts = MagicMock()
            fake_xtts.is_available.return_value = True
            fake_xtts.is_running.return_value = True

            def _sync_future(key: str, pcm: Any, on_done: Any = None) -> Future[bool]:
                fut: Future[bool] = Future()
                if on_done:
                    on_done(True)
                fut.set_result(True)
                return fut

            fake_xtts.register_speaker_async.side_effect = _sync_future
            fake_xtts.synthesize_stream.return_value = iter(
                [(np.ones((1200,), dtype=np.float32) * 0.15, 24000)]
            )

            t_sig = np.arange(int(16000 * 2.4), dtype=np.float32) / 16000.0
            male_seg = (0.18 * np.sin(2.0 * np.pi * 116.0 * t_sig)).astype(np.float32)
            b64_pcm = base64.b64encode((male_seg * 32767.0).astype("<i2").tobytes()).decode("ascii")

            CONTROLLER._media_service.reset_speakers()
            with (
                patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)),
                patch.object(CONTROLLER._media_service, "xtts_client", fake_xtts),
            ):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                conn.request(
                    "POST",
                    "/api/media/translate",
                    body=json.dumps(
                        {
                            "pcm_s16le_b64": b64_pcm,
                            "sample_rate": 16000,
                            "source_lang": "en",
                            "target_lang": "es",
                            "voice": "auto",
                            "tts_backend": "xtts_v2",
                            "diarization": False,  # Modo 1 sola voz: debe fijar spk_0 y clonar
                        }
                    ).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": ext_origin,
                        "X-Ignait-Client": "BrowserExtension",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                body = json.loads(resp.read().decode("utf-8"))
                conn.close()

            self.assertEqual(resp.status, 200)
            self.assertTrue(body["ok"])
            self.assertEqual(body["speaker_id"], 0)
            self.assertEqual(body["tts_engine"], "xtts_v2")
            # Gracias al sync de v1, la 1ra frase ya recibe speaker_key clonado
            kwargs = fake_xtts.synthesize_stream.call_args.kwargs
            self.assertEqual(kwargs["speaker_key"], "media:spk_0")
        finally:
            CONTROLLER.config = orig_cfg
            CONTROLLER._media_service.reset_speakers()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
