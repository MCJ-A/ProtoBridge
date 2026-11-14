from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import ctypes
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ignait.audio.devices import AudioEndpoint
from ignait.audio.ring_buffer import SPSCRingBuffer
from ignait.audio.streams import (
    PRECALCULATED_CHIRPS,
    TARGET_INTERNAL_RATE,
    AsyncOutputWorker,
    WASAPIInputStream,
    WASAPIOutputPlayer,
)
from ignait.gui import CONTROLLER, StitchHTTPHandler
from ignait.system.os_priority import mmcss_pro_audio
from ignait.tts.kokoro_engine import CONTRACTION_PATTERNS


def _dummy_endpoint(index: int = 1, rate: int = 16000, channels: int = 1) -> AudioEndpoint:
    return AudioEndpoint(
        index=index,
        name="MockWASAPIDevice",
        max_input_channels=channels,
        max_output_channels=channels,
        default_sample_rate=rate,
        is_loopback=False,
        is_default_input=False,
        is_default_output=False,
    )


class TestGoldMasterSystems(unittest.TestCase):
    """Pruebas unitarias Production-Ready Gold Master (Win32 MMCSS, DSP, WASAPI, HTTP CSRF)."""

    def test_mmcss_pro_audio_valid_and_exception_safe_revert(self) -> None:
        mock_avrt = MagicMock()
        mock_avrt.AvSetMmThreadCharacteristicsW.return_value = 0xBEEF

        with patch.object(ctypes, "windll", create=True) as mock_windll:
            mock_windll.Avrt = mock_avrt
            with mmcss_pro_audio() as registered:
                self.assertTrue(registered)

            mock_avrt.AvSetMmThreadCharacteristicsW.assert_called_once()
            self.assertEqual(mock_avrt.AvSetMmThreadCharacteristicsW.call_args[0][0], "Pro Audio")
            mock_avrt.AvRevertMmThreadCharacteristics.assert_called_once_with(0xBEEF)

            # Verificar liberación determinista ante excepciones
            mock_avrt.AvRevertMmThreadCharacteristics.reset_mock()
            with self.assertRaises(RuntimeError):
                with mmcss_pro_audio() as registered_err:
                    self.assertTrue(registered_err)
                    raise RuntimeError("boom")
            mock_avrt.AvRevertMmThreadCharacteristics.assert_called_once_with(0xBEEF)

    def test_mmcss_pro_audio_null_handle_skips_revert(self) -> None:
        mock_avrt = MagicMock()
        mock_avrt.AvSetMmThreadCharacteristicsW.return_value = 0

        with patch.object(ctypes, "windll", create=True) as mock_windll:
            mock_windll.Avrt = mock_avrt
            with mmcss_pro_audio() as registered:
                self.assertFalse(registered)
            mock_avrt.AvRevertMmThreadCharacteristics.assert_not_called()

    def test_async_output_worker_uses_mmcss_pro_audio(self) -> None:
        entered = threading.Event()
        exited = threading.Event()

        @contextmanager
        def fake_mmcss() -> Iterator[bool]:
            entered.set()
            try:
                yield True
            finally:
                exited.set()

        with patch("ignait.audio.streams.mmcss_pro_audio", side_effect=fake_mmcss):
            worker = AsyncOutputWorker(player=None)
            self.assertTrue(entered.wait(timeout=1.0))
            worker.close()
            self.assertTrue(exited.wait(timeout=1.0))

    def test_precalculated_chirps_immutable_and_zero_malloc_hot_path(self) -> None:
        self.assertIsInstance(PRECALCULATED_CHIRPS, MappingProxyType)
        self.assertIn(0.35, PRECALCULATED_CHIRPS)
        chirp_arr = PRECALCULATED_CHIRPS[0.35]
        self.assertFalse(chirp_arr.flags.writeable)

        mock_pa = MagicMock()
        mock_stream = MagicMock()
        mock_pa.open.return_value = mock_stream
        ep = _dummy_endpoint(rate=48000, channels=2)
        player = WASAPIOutputPlayer(pa=mock_pa, endpoint=ep)
        try:
            with patch("ignait.audio.streams.soxr.resample") as mock_resample:
                ok = player.play_precomputed_chirp(0.35)
                self.assertTrue(ok)
                mock_resample.assert_not_called()
                mock_stream.write.assert_called_once()
                written = mock_stream.write.call_args[0][0]
                self.assertIs(written, player._chirp_bytes_by_vol[0.35])
        finally:
            player.close()

    def test_async_output_worker_play_chirp_and_in_flight_flush(self) -> None:
        mock_pa = MagicMock()
        mock_stream = MagicMock()
        started_writing = threading.Event()
        release_write = threading.Event()
        written_blocks: list[bytes] = []

        def slow_write(data: bytes) -> None:
            written_blocks.append(data)
            started_writing.set()
            release_write.wait(timeout=1.0)

        mock_stream.write.side_effect = slow_write
        mock_pa.open.return_value = mock_stream
        ep = _dummy_endpoint(rate=16000, channels=1)
        player = WASAPIOutputPlayer(pa=mock_pa, endpoint=ep)
        worker = AsyncOutputWorker(player=player)
        try:
            # 100 ms de audio no nulo -> múltiples bloques de 20 ms
            pcm = np.ones(1600, dtype=np.float32) * 0.5
            worker.enqueue(pcm, sample_rate=16000)
            self.assertTrue(started_writing.wait(timeout=1.0))

            # Barge-in mientras bloque 0 está escribiéndose
            worker.flush()
            release_write.set()

            # Esperar a que termine el bloque fade-out y la cola quede vacía
            worker._queue.join()
            self.assertEqual(worker.pending_items, 0)
            # Solo debió escribir bloque 0 + bloque fade-out (no los 5 bloques)
            self.assertEqual(len(written_blocks), 2)
            fade_tail = np.frombuffer(written_blocks[1], dtype=np.float32)
            self.assertAlmostEqual(float(fade_tail[-1]), 0.0, places=6)

            # Verificar play_chirp encolando y despachando bytes precalculados
            written_blocks.clear()
            worker.play_chirp(0.35)
            worker._queue.join()
            self.assertEqual(len(written_blocks), 1)
            self.assertIs(written_blocks[0], player._chirp_bytes_by_vol[0.35])
        finally:
            release_write.set()
            worker.close()

    def test_barge_in_fade_out_eliminates_dc_offset_and_cancels_cleanly(self) -> None:
        mock_pa = MagicMock()
        mock_stream = MagicMock()
        written_chunks: list[bytes] = []
        mock_stream.write.side_effect = lambda b: written_chunks.append(b)
        mock_pa.open.return_value = mock_stream

        ep = _dummy_endpoint(rate=16000, channels=1)
        player = WASAPIOutputPlayer(pa=mock_pa, endpoint=ep)
        try:
            # Onda sinusoidal constante de 100 ms (5 bloques de 20 ms @ 16 kHz = 1600 muestras)
            t = np.linspace(0.0, 0.100, 1600, endpoint=False, dtype=np.float32)
            tone = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
            # Evitar coincidencia con cero en el cruce de bloque 1 -> offset DC intencional
            tone = np.clip(tone + np.float32(0.4), -1.0, 1.0).astype(np.float32)

            call_idx = 0

            def cancel_on_second_block() -> bool:
                nonlocal call_idx
                call_idx += 1
                # Llamada 1: pre-chequeo inicial -> False
                # Llamada 2: bloque 0 (0..320) -> False
                # Llamada 3: bloque 1 (320..640) -> True (interrupción barge-in)
                return call_idx >= 3

            completed = player.play_pcm(
                tone,
                source_rate=TARGET_INTERNAL_RATE,
                cancel_check=cancel_on_second_block,
            )
            self.assertFalse(completed)
            # Bloque 0 (20 ms = 320 muestras) + rampa fade-out (~9 ms = 144 muestras)
            self.assertEqual(len(written_chunks), 2)

            first_block = np.frombuffer(written_chunks[0], dtype=np.float32)
            fade_block = np.frombuffer(written_chunks[1], dtype=np.float32)
            self.assertEqual(first_block.shape[0], 320)
            self.assertEqual(fade_block.shape[0], 144)
            # El fade-out debe aterrizar exactamente en 0.0 para eliminar clics/DC offset
            self.assertAlmostEqual(float(fade_block[-1]), 0.0, places=6)
        finally:
            player.close()

    def test_wasapi_output_device_invalidated_retries_5x_exponential_backoff(self) -> None:
        mock_pa = MagicMock()
        corrupted_stream = MagicMock()
        corrupted_stream.write.side_effect = OSError("AUDCLNT_E_DEVICE_INVALIDATED")

        # Primera apertura tiene éxito; las 5 reaperturas fallan con OSError(-9999)
        mock_pa.open.side_effect = [
            corrupted_stream,
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
        ]

        ep = _dummy_endpoint(rate=16000, channels=1)
        player = WASAPIOutputPlayer(pa=mock_pa, endpoint=ep)
        try:
            pcm = np.ones(160, dtype=np.float32) * 0.2
            with patch("ignait.audio.streams.time.sleep") as mock_sleep:
                with self.assertRaises(OSError):
                    player.play_pcm(pcm, source_rate=16000)

                delays = [call[0][0] for call in mock_sleep.call_args_list]
                self.assertEqual(delays, [0.05, 0.1, 0.2, 0.4, 0.8])

            self.assertEqual(mock_pa.open.call_count, 6)  # 1 inicial + 5 reintentos
            corrupted_stream.stop_stream.assert_called()
            corrupted_stream.close.assert_called()
            mock_pa.terminate.assert_not_called()
        finally:
            player.close()

    def test_wasapi_input_device_invalidated_recovery_exponential_backoff(self) -> None:
        mock_pa = MagicMock()
        corrupted_in = MagicMock()
        corrupted_in.is_active.side_effect = OSError(-9996, "AUDCLNT_E_DEVICE_INVALIDATED")
        recovered_in = MagicMock()
        recovered_in.is_active.return_value = True

        # Apertura inicial ok -> falla 2 reintentos -> recupera en el 3er reintento
        mock_pa.open.side_effect = [
            corrupted_in,
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            OSError(-9999, "AUDCLNT_E_DEVICE_INVALIDATED"),
            recovered_in,
        ]

        rb = SPSCRingBuffer(name="test_rb_rec_in", capacity_seconds=0.1, sample_rate=16000, create=True)
        ep = _dummy_endpoint(rate=16000, channels=1)
        inp = WASAPIInputStream(pa=mock_pa, endpoint=ep, ring_buffer=rb)
        try:
            with inp._lock:
                inp._stream = inp._open_stream()

            with patch("ignait.audio.streams.time.sleep") as mock_sleep:
                recovered = inp.check_and_recover()
                self.assertTrue(recovered)
                delays = [call[0][0] for call in mock_sleep.call_args_list]
                self.assertEqual(delays, [0.05, 0.1, 0.2])

            corrupted_in.stop_stream.assert_called()
            corrupted_in.close.assert_called()
            recovered_in.start_stream.assert_called_once()
            mock_pa.terminate.assert_not_called()
        finally:
            inp.stop()
            rb.close()

    def test_contraction_patterns_precompiled(self) -> None:
        self.assertGreater(len(CONTRACTION_PATTERNS), 0)
        for pattern, repl in CONTRACTION_PATTERNS:
            self.assertTrue(hasattr(pattern, "sub"))
            self.assertIsInstance(repl, str)

    def test_http_csrf_host_origin_and_custom_header_guards(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. GET /api/state con Host válido -> 200
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/api/state?since=0", headers={"Host": f"127.0.0.1:{port}"})
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 200)
            conn.close()

            # 2. DNS Rebinding / Host no autorizado -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/api/state", headers={"Host": f"evil.com:{port}"})
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 3. Prefix bypass Host -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/api/state", headers={"Host": f"127.0.0.1.attacker.com:{port}"})
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 4. Origin no autorizado -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "GET",
                "/api/state",
                headers={"Host": f"localhost:{port}", "Origin": "https://attacker.com"},
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 5. Sec-Fetch-Site: cross-site -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "GET",
                "/api/state",
                headers={"Host": f"127.0.0.1:{port}", "Sec-Fetch-Site": "cross-site"},
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 6. POST sin X-Ignait-Client: WebUI -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "POST",
                "/api/config",
                body=b'{"silent": true}',
                headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 7. POST con X-Ignait-Client: WebUI y JSON malformado -> 400
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "POST",
                "/api/config",
                body=b"{invalid json",
                headers={
                    "Host": f"127.0.0.1:{port}",
                    "X-Ignait-Client": "WebUI",
                    "Content-Type": "application/json",
                },
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 400)
            conn.close()

            # 8. POST válido autorizado -> 200 (evitando sobrescribir config.json real)
            with patch.object(CONTROLLER.config, "save") as mock_save:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/config",
                    body=b'{"silent": true}',
                    headers={
                        "Host": f"localhost:{port}",
                        "Origin": f"http://localhost:{port}",
                        "Sec-Fetch-Site": "same-origin",
                        "X-Ignait-Client": "WebUI",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                _ = resp.read()
                self.assertEqual(resp.status, 200)
                mock_save.assert_called_once()
                conn.close()

            # 9. POST /api/action reset_squad -> 200 y llama a CONTROLLER.reset_squad
            with patch.object(CONTROLLER, "reset_squad") as mock_reset:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/action",
                    body=b'{"action": "reset_squad"}',
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "X-Ignait-Client": "WebUI",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                _ = resp.read()
                self.assertEqual(resp.status, 200)
                mock_reset.assert_called_once()
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)


class TestSquadAndCrossVadGoldMaster(unittest.TestCase):
    def test_stream_state_transitions(self) -> None:
        from ignait.audio.streams import (
            StreamState,
            WASAPIInputStream,
            WASAPIOutputPlayer,
        )

        ep_out = _dummy_endpoint(index=1, rate=48000, channels=2)
        fake_pa = MagicMock()
        good_stream = MagicMock()
        reopened_stream = MagicMock()
        fake_pa.open.return_value = good_stream
        player = WASAPIOutputPlayer(pa=fake_pa, endpoint=ep_out)
        self.assertEqual(player.state, StreamState.RUNNING)

        good_stream.write.side_effect = OSError(-9999, "Unanticipated host error")
        fake_pa.open.side_effect = [OSError(-9999, "Unanticipated host error"), reopened_stream]
        with patch("ignait.audio.streams.time.sleep"):
            player._write_bytes_with_recovery(b"\x00" * 16)
        self.assertEqual(player.state, StreamState.RUNNING)
        reopened_stream.write.assert_called_once()

        reopened_stream.write.side_effect = OSError(-9996, "Invalid device")
        fake_pa.open.side_effect = OSError(-9996, "Invalid device")
        with patch("ignait.audio.streams.time.sleep"):
            with self.assertRaises(OSError):
                player._write_bytes_with_recovery(b"\x00" * 16)
        self.assertEqual(player.state, StreamState.FAULTED)

        ep_in = _dummy_endpoint(index=2, rate=48000, channels=1)
        rb = SPSCRingBuffer(name="test_state_in", capacity_seconds=0.5, sample_rate=16000, create=True)
        try:
            fake_pa_in = MagicMock()
            fake_in_stream = MagicMock()
            fake_in_stream.is_active.return_value = True
            fake_pa_in.open.return_value = fake_in_stream
            in_stream = WASAPIInputStream(pa=fake_pa_in, endpoint=ep_in, ring_buffer=rb)
            self.assertEqual(in_stream.state, StreamState.RUNNING)
            in_stream.start()
            self.assertEqual(in_stream.state, StreamState.RUNNING)

            fake_in_stream.is_active.side_effect = OSError(-2004287484, "AUDCLNT_E_DEVICE_INVALIDATED")
            fake_pa_in.open.side_effect = OSError(-2004287484, "AUDCLNT_E_DEVICE_INVALIDATED")
            with patch("ignait.audio.streams.time.sleep"):
                with self.assertRaises(OSError):
                    in_stream.check_and_recover()
            self.assertEqual(in_stream.state, StreamState.FAULTED)
            in_stream.stop()
        finally:
            rb.close()

    def test_hotkey_reset_squad_edge_trigger(self) -> None:
        from ignait.control.hotkeys import GlobalHotkeyListener

        triggered: list[bool] = []
        listener = GlobalHotkeyListener(
            hotkey_reset_squad="ctrl+shift+r",
            on_reset_squad=lambda: triggered.append(True),
        )
        pressed_set: set[int] = set()
        listener._is_combo_down = lambda vks: bool(vks) and vks == listener.hotkey_reset_vks and bool(pressed_set)  # type: ignore[method-assign]

        listener._poll_once()
        self.assertEqual(triggered, [])

        pressed_set.add(1)
        listener._poll_once()
        listener._poll_once()  # held down: must edge-trigger once
        self.assertEqual(len(triggered), 1)

        pressed_set.clear()
        listener._poll_once()
        pressed_set.add(1)
        listener._poll_once()
        self.assertEqual(len(triggered), 2)

    def test_speaker_tracker_reset_squad(self) -> None:
        import tempfile
        from ignait.diarization.speaker_id import OnlineSpeakerTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ignait.diarization.speaker_id._ensure_speaker_model", return_value=Path(tmpdir) / "spk.onnx"), \
                 patch("ignait.diarization.speaker_id.ort.InferenceSession"):
                tracker = OnlineSpeakerTracker(models_dir=Path(tmpdir), max_speakers=4)
                tracker._centroids.append(np.ones((256,), dtype=np.float32))
                tracker._assigned_slots.append(0)
                tracker._pitches.append(120.0)
                tracker._last_speaker_idx = 1

                tracker.reset_squad()
                self.assertEqual(tracker._centroids, [])
                self.assertEqual(tracker._assigned_slots, [])
                self.assertEqual(tracker._pitches, [])
                self.assertEqual(tracker._last_speaker_idx, 0)

    def test_cross_vad_gating_15db_and_hold(self) -> None:
        import tempfile
        from ignait.vad.silero import SileroVADSegmenter, _CROSS_VAD_GAIN_15DB

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ignait.vad.silero.ensure_silero_model", return_value=Path(tmpdir) / "vad.onnx"), \
                 patch("ignait.vad.silero.ort.InferenceSession"):
                vad = SileroVADSegmenter(
                    models_dir=Path(tmpdir),
                    speech_threshold=0.50,
                    min_silence_ms=32,
                    min_speech_ms=32,
                    min_rms_dbfs=-35.0,
                )

                # Feed a weak-to-moderate signal (~ -28 dBFS: passes -35 dBFS normally, blocked under +15 dB gate = -20 dBFS)
                amp = float(10.0 ** (-28.0 / 20.0))
                rb = SPSCRingBuffer(name="test_cross_vad", capacity_seconds=1.0, sample_rate=16000, create=True)
                try:
                    # 1) Gated: prob=0.70 (< 0.75 eff_speech_thresh) and rms < -20 dBFS -> suppressed
                    vad.notify_output_audio(hold_ms=200.0)
                    self.assertTrue(vad.is_cross_vad_gated)
                    self.assertGreater(_CROSS_VAD_GAIN_15DB, 5.5)

                    probs = iter([0.70, 0.10])
                    vad._infer_frame = lambda frame: next(probs)  # type: ignore[method-assign]
                    rb.write(np.full((1024,), amp, dtype=np.float32))
                    segs_gated = vad.process_available(rb)
                    self.assertEqual(segs_gated, [])

                    # 2) Ungated: same prob=0.70 and amp=-28 dBFS -> accepted
                    vad._gate_until_ts = 0.0
                    self.assertFalse(vad.is_cross_vad_gated)
                    probs2 = iter([0.70, 0.10])
                    vad._infer_frame = lambda frame: next(probs2)  # type: ignore[method-assign]
                    rb.write(np.full((1024,), amp, dtype=np.float32))
                    segs_open = vad.process_available(rb)
                    self.assertEqual(len(segs_open), 1)
                finally:
                    rb.close()

    def test_tactical_glossary_config_and_update(self) -> None:
        import json
        import tempfile
        from ignait.config import AppConfig
        from ignait.mt.translator import TacticalTranslatorEngine, apply_glossary

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(
                json.dumps({"tactical_glossary": {"helldrop": "Despliegue Infernal"}}),
                encoding="utf-8",
            )
            cfg = AppConfig.load(cfg_path)
            self.assertEqual(cfg.custom_glossary.get("helldrop"), "Despliegue Infernal")
            self.assertEqual(cfg.hotkey_reset_squad, "ctrl+shift+r")

            engine = TacticalTranslatorEngine.__new__(TacticalTranslatorEngine)
            engine.update_glossary(cfg.custom_glossary)
            self.assertEqual(apply_glossary("helldrop", engine._active_glossary), "Despliegue Infernal")

    def test_russian_tts_routing(self) -> None:
        from ignait.tts.kokoro_engine import KokoroTTSEngine

        engine = KokoroTTSEngine.__new__(KokoroTTSEngine)
        engine.models_dir = Path("models")
        engine.default_voice = "my_cloned"
        engine.user_f0 = 128.0
        fake_ru = MagicMock()
        fake_ru.synthesize.return_value = (np.zeros((2205,), dtype=np.float32), 22050)
        engine._ru_tts = fake_ru

        pcm, sr = engine.synthesize("Привет", voice="am_adam", lang="ru", speed=1.1)
        self.assertEqual(sr, 22050)
        self.assertEqual(pcm.dtype, np.float32)
        fake_ru.synthesize.assert_called_once_with("Привет", speed=1.1)

    def test_media_lookahead_buffer_delay_and_smart_ducking(self) -> None:
        from ignait.audio.streams import MediaLookaheadBuffer

        buf = MediaLookaheadBuffer(
            sample_rate=16000,
            max_delay_seconds=2.0,
            delay_ms=100.0,  # 1600 samples
            duck_db=-14.0,
            normal_gain=0.85,
            enabled=True,
        )
        # Before delay_samples are written, output is held back (zeros)
        buf.write(np.ones((800,), dtype=np.float32))
        self.assertEqual(buf.available_to_read, 0)
        pre = buf.read_mixed_16k(512, ducking=False)
        self.assertTrue(np.allclose(pre, 0.0))

        # Cross delay threshold (800 + 1000 = 1800 >= 1600)
        buf.write(np.full((1000,), 0.5, dtype=np.float32))
        self.assertGreaterEqual(buf.available_to_read, 1600)

        normal_chunk = buf.read_mixed_16k(512, ducking=False)
        self.assertEqual(normal_chunk.shape, (512,))
        self.assertAlmostEqual(float(normal_chunk[0]), 0.85, places=4)

        ducked_chunk = buf.read_mixed_16k(512, ducking=True)
        expected_duck = float(10.0 ** (-14.0 / 20.0))
        # Samples 512..799 are 1.0 (ramp starts at 0.85); sample 1023 is 0.5 (ends at 0.5 * duck_gain)
        self.assertAlmostEqual(float(ducked_chunk[0]), 0.85, places=3)
        self.assertAlmostEqual(float(ducked_chunk[-1]), 0.5 * expected_duck, places=3)

    def test_async_output_worker_media_mode_mixes_ducked_background(self) -> None:
        import time
        from ignait.audio.streams import AsyncOutputWorker, MediaLookaheadBuffer

        media_buf = MediaLookaheadBuffer(
            sample_rate=16000,
            max_delay_seconds=2.0,
            delay_ms=0.0,
            duck_db=-14.0,
            normal_gain=0.85,
            enabled=True,
        )
        # Write constant 1.0 background into media buffer
        media_buf.write(np.ones((4096,), dtype=np.float32))

        played_pcms: list[np.ndarray] = []

        class _FakePlayer:
            def play_pcm(self, pcm, source_rate, pan_left=1.0, pan_right=1.0, cancel_check=None):
                played_pcms.append(pcm.copy())
                return True

            def close(self) -> None:
                pass

        worker = AsyncOutputWorker(player=_FakePlayer(), media_buffer=media_buf)  # type: ignore[arg-type]
        try:
            tts_pcm = np.full((512,), 0.4, dtype=np.float32)
            worker.enqueue(tts_pcm, sample_rate=16000)
            time.sleep(0.15)
            # Also allow idle passthrough chunk to run at least once
            media_buf.write(np.ones((1024,), dtype=np.float32))
            time.sleep(0.12)
        finally:
            worker.close()

        self.assertGreaterEqual(len(played_pcms), 1)
        mixed_tts = played_pcms[0]
        # TTS (0.4) + ducked bg at tail (~0.20) > pure TTS (0.4)
        self.assertGreater(float(mixed_tts[-1]), 0.55)
        self.assertLessEqual(float(np.max(np.abs(mixed_tts))), 1.0)

    def test_chrome_extension_media_endpoints_and_csrf_guards(self) -> None:
        import base64
        import json

        manifest_path = Path(__file__).resolve().parent.parent / "browser_extension" / "manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_data.get("manifest_version"), 3)

        ext_origin = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
        server = ThreadingHTTPServer(("127.0.0.1", 0), StitchHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. GET /api/media/ping desde extensión Chrome -> 200
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "GET",
                "/api/media/ping",
                headers={
                    "Host": f"127.0.0.1:{port}",
                    "Origin": ext_origin,
                    "Sec-Fetch-Site": "cross-site",
                    "X-Ignait-Client": "BrowserExtension",
                },
            )
            resp = conn.getresponse()
            ping_body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertTrue(ping_body.get("ok"))
            self.assertEqual(ping_body.get("app"), "Ignait")
            self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), ext_origin)
            conn.close()

            # 2. Least-privilege: BrowserExtension NO puede llamar /api/action -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "POST",
                "/api/action",
                body=b'{"action": "reset_squad"}',
                headers={
                    "Host": f"127.0.0.1:{port}",
                    "Origin": ext_origin,
                    "X-Ignait-Client": "BrowserExtension",
                    "Content-Type": "application/json",
                },
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 3. Origin web externo en /api/media/translate -> 403
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request(
                "POST",
                "/api/media/translate",
                body=b'{"text": "hello"}',
                headers={
                    "Host": f"127.0.0.1:{port}",
                    "Origin": "https://evil.com",
                    "X-Ignait-Client": "BrowserExtension",
                    "Content-Type": "application/json",
                },
            )
            resp = conn.getresponse()
            _ = resp.read()
            self.assertEqual(resp.status, 403)
            conn.close()

            # 4. POST /api/media/translate con audio PCM mockeado -> devuelve WAV b64 válido
            fake_stt = MagicMock()
            fake_stt.transcribe_any.return_value = ("Hold the objective.", "en", 0.99)
            fake_mt = MagicMock()
            fake_mt.translate.return_value = "Mantengan el objetivo."
            fake_tts = MagicMock()
            fake_tts.synthesize.return_value = (np.ones((2400,), dtype=np.float32) * 0.25, 24000)

            pcm_16k = (np.ones((3200,), dtype=np.float32) * 0.2 * 32767.0).astype("<i2")
            pcm_b64 = base64.b64encode(pcm_16k.tobytes()).decode("ascii")

            with patch.object(CONTROLLER, "_ensure_ai_engines", return_value=(fake_stt, fake_mt, fake_tts)):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request(
                    "POST",
                    "/api/media/translate",
                    body=json.dumps(
                        {
                            "pcm_s16le_b64": pcm_b64,
                            "sample_rate": 16000,
                            "source_lang": "auto",
                            "target_lang": "es",
                            "voice": "em_alex",
                            "speed": 1.12,
                        }
                    ).encode("utf-8"),
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Origin": ext_origin,
                        "Sec-Fetch-Site": "cross-site",
                        "X-Ignait-Client": "BrowserExtension",
                        "Content-Type": "application/json",
                    },
                )
                resp = conn.getresponse()
                body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(resp.status, 200)
                self.assertTrue(body.get("ok"))
                self.assertFalse(body.get("empty"))
                self.assertEqual(body.get("src_text"), "Hold the objective.")
                self.assertEqual(body.get("translated_text"), "Mantengan el objetivo.")
                self.assertEqual(body.get("sample_rate"), 24000)
                wav_bytes = base64.b64decode(body.get("wav_base64", ""))
                self.assertTrue(wav_bytes.startswith(b"RIFF"))
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
