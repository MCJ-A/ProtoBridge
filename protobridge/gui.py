from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import asdict
import gc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
if sys.stdout is None: sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None: sys.stderr = open(os.devnull, 'w', encoding='utf-8')
import numpy as np
import huggingface_hub.utils as _h

_h.disable_progress_bars()
if sys.__stdout__ is None:
    sys.__stdout__ = sys.stdout
if sys.__stderr__ is None:
    sys.__stderr__ = sys.stderr
import threading
import time
from urllib.parse import parse_qs, urlparse
import urllib.request
import zipfile
import pyaudiowpatch as pyaudio
import soxr

from protobridge.audio.devices import AudioEndpoint, WASAPIDeviceManager
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.audio.streams import (
    AsyncOutputWorker,
    MediaLookaheadBuffer,
    TACTICAL_CHIRP_PCM,
    WASAPIInputStream,
)
from protobridge.config import AppConfig
from protobridge.control.hotkeys import GlobalHotkeyListener
from protobridge.diarization.speaker_id import OnlineSpeakerTracker, WESPEAKER_ONNX_URL
from protobridge.media.session import MediaPipelineService, encode_wav_s16le_b64
from protobridge.media.worker_client import XttsWorkerClient
from protobridge.media.ws_protocol import WebSocketConnection, compute_ws_accept_key
from protobridge.mt.translator import NLLB_CT2_REPO, NLLB_TOKENIZER_REPO, TacticalTranslatorEngine
from protobridge.paths import (
    get_browser_extension_dir,
    get_models_dir,
    get_ui_html_path,
    get_xtts_models_dir,
)
from protobridge.stt.whisper_engine import WhisperTranslatorEngine
from protobridge.system.os_priority import set_high_process_priority
from protobridge.tts.kokoro_engine import KokoroTTSEngine, ensure_kokoro_assets
from protobridge.vad.silero import SileroVADSegmenter, ensure_silero_model

from huggingface_hub import hf_hub_download, snapshot_download


def _can_open_input(pa: pyaudio.PyAudio, ep: AudioEndpoint) -> bool:
    """Verifica si un endpoint de entrada o loopback WASAPI responde sin lanzar [Errno -9999]."""
    ch = max(1, ep.max_input_channels if ep.max_input_channels > 0 else ep.max_output_channels)
    try:
        st = pa.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=ep.default_sample_rate,
            input=True,
            input_device_index=ep.index,
            frames_per_buffer=512,
            start=False,
        )
        st.close()
        return True
    except Exception:
        return False


def _can_open_output(pa: pyaudio.PyAudio, ep: AudioEndpoint) -> bool:
    """Verifica si un endpoint de salida WASAPI responde sin lanzar [Errno -9996] ni [Errno -9999]."""
    ch = max(1, ep.max_output_channels)
    try:
        st = pa.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=ep.default_sample_rate,
            output=True,
            output_device_index=ep.index,
            frames_per_buffer=512,
            start=False,
        )
        st.close()
        return True
    except Exception:
        return False


def _encode_wav_s16le_b64(pcm_f32: np.ndarray, sample_rate: int) -> str:
    return encode_wav_s16le_b64(pcm_f32, sample_rate)


class StitchBackendController:
    def __init__(self) -> None:
        self.models_dir = get_models_dir()
        self.config = AppConfig.load()
        self.spatial_3d: bool = True

        self.running: bool = False
        self.loading: bool = False
        self.mic_db: float = -100.0
        self.chat_db: float = -100.0
        self.last_latency_ms: int = 0
        self.radar: dict[str, str] = {
            "left": "Inactivo",
            "center": "Inactivo",
            "right": "Inactivo",
        }

        self._log_lock = threading.Lock()
        self._logs: list[dict[str, object]] = []
        self._log_counter: int = 0

        self._ai_lock = threading.RLock()
        self._shared_stt: WhisperTranslatorEngine | None = None
        self._shared_mt: TacticalTranslatorEngine | None = None
        self._shared_tts: KokoroTTSEngine | None = None
        self._shared_whisper_size: str = ""

        self._xtts_client = XttsWorkerClient(
            models_dir=get_xtts_models_dir(),
            on_log=self.add_log,
        )
        self._media_service = MediaPipelineService(
            models_dir=self.models_dir,
            ai_lock=self._ai_lock,
            ensure_engines_fn=lambda w=None: self._ensure_ai_engines(w),
            get_config_fn=lambda: self.config,
            xtts_client=self._xtts_client,
            on_log=self.add_log,
            on_latency_ms=self._set_latency_ms,
        )

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._speaker_tracker: OnlineSpeakerTracker | None = None
        self._hp_out: AsyncOutputWorker | None = None
        self._media_buffer = MediaLookaheadBuffer(
            enabled=self.config.media_mode_enabled,
            delay_ms=float(self.config.media_delay_ms),
            duck_db=self.config.media_ducking_db,
        )

        self._endpoints: list[AudioEndpoint] = []
        self.devices_payload: dict[str, list[str]] = {"inputs": [], "outputs": [], "loopbacks": []}
        self._scan_verified_devices()

        self._hotkeys = GlobalHotkeyListener(
            hotkey_mute_tx=self.config.hotkey_mute_tx,
            hotkey_mute_rx=self.config.hotkey_mute_rx,
            hotkey_ptt=self.config.hotkey_ptt,
            hotkey_reset_squad=self.config.hotkey_reset_squad,
            ptt_enabled=self.config.ptt_enabled,
            initial_mute_tx=self.config.mute_tx,
            initial_mute_rx=self.config.mute_rx,
            on_tx_mute_changed=self._on_tx_mute_hotkey,
            on_rx_mute_changed=self._on_rx_mute_hotkey,
            on_ptt_changed=self._on_ptt_hotkey,
            on_reset_squad=self.reset_squad,
        )
        self._hotkeys.start()
        self._media_service.set_rx_muted(bool(self.config.mute_rx))

    def _on_tx_mute_hotkey(self, muted: bool) -> None:
        self.config.mute_tx = bool(muted)
        self.add_log(f"[CANAL TX]: {'SILENCIADO' if muted else 'ACTIVO'}", "sys")

    def _on_rx_mute_hotkey(self, muted: bool) -> None:
        self.config.mute_rx = bool(muted)
        self.sync_media_mode()
        self._media_service.set_rx_muted(bool(muted))
        self.add_log(f"[CANAL RX / MEDIA]: {'SILENCIADO' if muted else 'ACTIVO'}", "sys")

    def set_channel_mutes(
        self, mute_tx: bool | None = None, mute_rx: bool | None = None
    ) -> dict[str, bool]:
        if mute_tx is not None:
            tx_val = bool(mute_tx)
            self.config.mute_tx = tx_val
            self._hotkeys.set_mute_tx(tx_val)
        if mute_rx is not None:
            rx_val = bool(mute_rx)
            self.config.mute_rx = rx_val
            self._hotkeys.set_mute_rx(rx_val)
            self._media_service.set_rx_muted(rx_val)
        self.sync_media_mode()
        return {
            "mute_tx": bool(self.config.mute_tx or self._hotkeys.is_tx_muted),
            "mute_rx": bool(self.config.mute_rx or self._hotkeys.is_rx_muted),
        }

    def _on_ptt_hotkey(self, active: bool) -> None:
        if active and self.config.barge_in_enabled and self._hp_out is not None:
            self._hp_out.flush()

    def reset_squad(self) -> None:
        if self._speaker_tracker is not None:
            self._speaker_tracker.reset_squad()
        self.radar = {
            "left": "Escuchando [-45°]",
            "center": "Listo [0°]",
            "right": "Escuchando [+45°]",
        }
        self.add_log("[ESCUADRÓN] Centroides de voz y radar 3D reiniciados.", "sys")

    def add_log(self, msg: str, tag: str = "sys") -> None:
        with self._log_lock:
            self._log_counter += 1
            self._logs.append({"id": self._log_counter, "msg": msg, "tag": tag})
            if len(self._logs) > 250:
                self._logs = self._logs[-250:]

    def get_logs_since(self, since_id: int) -> list[dict[str, object]]:
        with self._log_lock:
            return [item for item in self._logs if int(item["id"]) > since_id]

    def _scan_verified_devices(self) -> None:
        mgr = WASAPIDeviceManager()
        pa = pyaudio.PyAudio()
        try:
            all_eps = mgr.list_wasapi_endpoints()
            topo = mgr.auto_detect_topology()
            self._endpoints = all_eps

            inputs = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.max_input_channels > 0 and not ep.is_loopback and _can_open_input(pa, ep)
            ]
            outputs = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.max_output_channels > 0 and not ep.is_loopback and _can_open_output(pa, ep)
            ]
            loopbacks = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.is_loopback and _can_open_input(pa, ep)
            ]

            self.devices_payload = {
                "inputs": inputs,
                "outputs": outputs,
                "loopbacks": loopbacks + inputs,
            }

            valid_in_set = set(inputs)
            if self.config.mic_device_name not in valid_in_set:
                if topo.physical_mic:
                    self.config.mic_device_name = f"[{topo.physical_mic.index}] {topo.physical_mic.name}"
                elif inputs:
                    self.config.mic_device_name = inputs[0]

            valid_out_set = set(outputs)
            if self.config.headphones_device_name not in valid_out_set:
                if topo.physical_headphones:
                    self.config.headphones_device_name = (
                        f"[{topo.physical_headphones.index}] {topo.physical_headphones.name}"
                    )
                elif outputs:
                    self.config.headphones_device_name = outputs[0]

            if self.config.cable_a_device_name not in valid_out_set:
                if topo.virtual_mic_sink_cable_a:
                    self.config.cable_a_device_name = (
                        f"[{topo.virtual_mic_sink_cable_a.index}] {topo.virtual_mic_sink_cable_a.name}"
                    )
                elif outputs:
                    preferred_cable = next((o for o in outputs if "cable" in o.lower()), outputs[0])
                    self.config.cable_a_device_name = preferred_cable

            # Elegir un Loopback verificado que no falle con -9999 (ej. Steam Streaming Speakers o ROG CARNYX)
            valid_lb_Set = set(loopbacks)
            if self.config.team_loopback_device_name not in valid_lb_Set:
                if topo.teammate_voip_source_cable_b and f"[{topo.teammate_voip_source_cable_b.index}] {topo.teammate_voip_source_cable_b.name}" in valid_lb_Set:
                    self.config.team_loopback_device_name = f"[{topo.teammate_voip_source_cable_b.index}] {topo.teammate_voip_source_cable_b.name}"
                else:
                    preferred_lb = next(
                        (
                            lb
                            for lb in loopbacks
                            if "steam streaming speakers" in lb.lower()
                            or "oculus virtual" in lb.lower()
                            or "carnyx" in lb.lower()
                            or "cable" in lb.lower()
                        ),
                        loopbacks[0] if loopbacks else (inputs[0] if inputs else ""),
                    )
                    self.config.team_loopback_device_name = preferred_lb

            self.config.save()
        finally:
            pa.terminate()
            mgr.close()

    def _resolve_endpoint(self, label: str) -> AudioEndpoint:
        if label.startswith("["):
            idx = int(label.split("]")[0][1:])
            for ep in self._endpoints:
                if ep.index == idx:
                    return ep
        for ep in self._endpoints:
            if ep.name.lower() in label.lower():
                return ep
        raise RuntimeError(f"No se encontró el dispositivo: {label}")

    def get_models_status(self) -> list[dict[str, object]]:
        has_vbcable = any("cable" in o.lower() for o in self.devices_payload["outputs"])
        has_silero = (self.models_dir / "silero_vad.onnx").exists()
        has_wespeaker = (self.models_dir / "wespeaker_campp.onnx").exists()
        has_kokoro = (self.models_dir / "kokoro-v1.0.int8.onnx").exists() and (
            self.models_dir / "voices-v1.0.bin"
        ).exists()
        has_nllb = (self.models_dir / "nllb_200_ct2_int8" / "model.bin").exists()
        xtts_dir = get_xtts_models_dir()
        has_xtts = (
            (xtts_dir / "model.pth").exists()
            and (xtts_dir / "config.json").exists()
            and self._xtts_client.is_available()
        )

        return [
            {
                "title": f"faster-whisper-{self.config.whisper_model_size}-int8_float16 (STT GPU)",
                "subtitle": "GPU NVIDIA RTX 5060 • 99 Idiomas • Detección Automática",
                "ready": True,
            },
            {
                "title": "nllb-200-distilled-600M-int8 (Traducción Táctica)",
                "subtitle": "CTranslate2 CPU • Traducción Directa entre 200 Idiomas",
                "ready": has_nllb,
            },
            {
                "title": "kokoro-82m-v1.0-int8 (Síntesis Multi-Voz 3D)",
                "subtitle": "ONNX Runtime • Voces EN / ES / FR / PT / IT + Voice Blending",
                "ready": has_kokoro,
            },
            {
                "title": "XTTS-v2 Worker 3.12.8 (Doblaje Cinematográfico)",
                "subtitle": "Subproceso Aislado CUDA FP16 • Clonación Multi-Voz en Streaming",
                "ready": has_xtts,
            },
            {
                "title": "wespeaker-voxceleb-cam++ (Identificación de Hablantes)",
                "subtitle": "Diarización en Paralelo (0 ms latencia añadida) + Detector F0",
                "ready": has_wespeaker,
            },
            {
                "title": "silero-vad-v5 + VB-Audio Virtual Cable",
                "subtitle": "Detección de Voz 32ms + Enrutamiento Kernel WASAPI",
                "ready": has_silero and has_vbcable,
            },
        ]

    def toggle_engine(self) -> None:
        if not self.running and not self.loading:
            self._stop_event.clear()
            self.loading = True
            self._worker_thread = threading.Thread(target=self._engine_loop, daemon=True)
            self._worker_thread.start()
        else:
            self._stop_event.set()

    def sync_media_mode(self) -> None:
        rx_muted = bool(self.config.mute_rx or self._hotkeys.is_rx_muted)
        self._media_buffer.configure(
            enabled=bool(self.config.media_mode_enabled and not rx_muted),
            delay_ms=float(self.config.media_delay_ms),
            duck_db=self.config.media_ducking_db,
        )

    def _set_latency_ms(self, val: int) -> None:
        self.last_latency_ms = int(val)

    def reset_media_speakers(self) -> None:
        self._media_service.reset_speakers()
        self.add_log("[CHROME MEDIA] Voces y perfiles de hablantes reiniciados.", "sys")

    def prewarm_xtts_async(self) -> None:
        def _warm() -> None:
            try:
                if not self._xtts_client.is_available():
                    self.add_log(
                        "[XTTS WORKER] Entorno .venv-media-tts o pesos XTTS-v2 no encontrados.",
                        "err",
                    )
                    return
                port = self._xtts_client.ensure_started()
                self.add_log(f"[XTTS WORKER] Motor precalentado y activo en puerto {port}.", "rx")
            except Exception as exc:
                self.add_log(f"[ERROR XTTS WORKER]: {exc}", "err")

        threading.Thread(target=_warm, daemon=True, name="xtts_prewarm").start()

    def unload_xtts(self) -> None:
        self._xtts_client.stop()
        self.add_log("[XTTS WORKER] Subproceso GPU detenido y VRAM liberada.", "sys")

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self._xtts_client.stop()

    def _ensure_ai_engines(
        self, whisper_model_size: str | None = None
    ) -> tuple[WhisperTranslatorEngine, TacticalTranslatorEngine, KokoroTTSEngine]:
        target_whisper = (
            (whisper_model_size or self.config.whisper_model_size or "base").strip() or "base"
        )
        with self._ai_lock:
            if self._shared_stt is None or self._shared_whisper_size != target_whisper:
                if self._shared_stt is not None:
                    self._shared_stt = None
                    gc.collect()
                try:
                    self._shared_stt = WhisperTranslatorEngine(model_size=target_whisper)
                    self._shared_whisper_size = target_whisper
                except Exception as exc:
                    if target_whisper != "small":
                        self.add_log(
                            f"[STT FALLBACK] Error cargando Whisper '{target_whisper}' ({exc}); usando 'small'.",
                            "sys",
                        )
                        gc.collect()
                        self._shared_stt = WhisperTranslatorEngine(model_size="small")
                        self._shared_whisper_size = "small"
                    else:
                        raise
            if self._shared_mt is None:
                self._shared_mt = TacticalTranslatorEngine(
                    models_dir=self.models_dir,
                    custom_glossary=self.config.custom_glossary,
                )
            else:
                self._shared_mt.update_glossary(self.config.custom_glossary)
            if self._shared_tts is None:
                self._shared_tts = KokoroTTSEngine(
                    models_dir=self.models_dir,
                    default_voice=self.config.my_voice_profile,
                )
            return self._shared_stt, self._shared_mt, self._shared_tts

    def translate_media_payload(self, data: dict[str, object]) -> dict[str, object]:
        return self._media_service.translate_http_payload(data)

    def _engine_loop(self) -> None:
        if sys.platform == "win32":
            ctypes.windll.ole32.CoInitialize(None)

        pa: pyaudio.PyAudio | None = None
        mic_stream: WASAPIInputStream | None = None
        team_stream: WASAPIInputStream | None = None
        cable_out: AsyncOutputWorker | None = None
        hp_out: AsyncOutputWorker | None = None
        rb_mic: SPSCRingBuffer | None = None
        rb_team: SPSCRingBuffer | None = None
        executor = ThreadPoolExecutor(max_workers=2)

        try:
            mic_ep = self._resolve_endpoint(self.config.mic_device_name)
            hp_ep = self._resolve_endpoint(self.config.headphones_device_name)
            cable_ep = self._resolve_endpoint(self.config.cable_a_device_name)
            team_ep = self._resolve_endpoint(self.config.team_loopback_device_name)

            self.add_log(
                f"[INICIANDO] Cargando Faster-Whisper ({self.config.whisper_model_size}) en RTX 5060 + NLLB-200 + Kokoro...",
                "sys",
            )

            vad_mic = SileroVADSegmenter(
                models_dir=self.models_dir, min_silence_ms=self.config.vad_silence_ms
            )
            vad_team = SileroVADSegmenter(
                models_dir=self.models_dir, min_silence_ms=self.config.vad_silence_ms
            )
            speaker_tracker = OnlineSpeakerTracker(models_dir=self.models_dir, max_speakers=4)
            self._speaker_tracker = speaker_tracker
            stt, mt, tts = self._ensure_ai_engines()

            rb_mic = SPSCRingBuffer(
                name="protobridge_stitch_mic", capacity_seconds=30.0, sample_rate=16000, create=True
            )
            rb_team = SPSCRingBuffer(
                name="protobridge_stitch_team", capacity_seconds=30.0, sample_rate=16000, create=True
            )

            self.sync_media_mode()
            pa = pyaudio.PyAudio()
            mic_stream = WASAPIInputStream(pa=pa, endpoint=mic_ep, ring_buffer=rb_mic)
            mic_stream.start()

            try:
                team_stream = WASAPIInputStream(
                    pa=pa,
                    endpoint=team_ep,
                    ring_buffer=rb_team,
                    media_buffer=self._media_buffer,
                )
                team_stream.start()
            except Exception as team_err:
                self.add_log(
                    f"[AVISO] El bus '{team_ep.name}' está inactivo ({team_err}). Conmutando a bus Loopback de respaldo...",
                    "sys",
                )
                team_stream = None
                for candidate_str in self.devices_payload["loopbacks"]:
                    cand_ep = self._resolve_endpoint(candidate_str)
                    if cand_ep.index == team_ep.index:
                        continue
                    try:
                        team_stream = WASAPIInputStream(
                            pa=pa,
                            endpoint=cand_ep,
                            ring_buffer=rb_team,
                            media_buffer=self._media_buffer,
                        )
                        team_stream.start()
                        self.add_log(
                            f"[OK] Bus de compañeros conectado en: {cand_ep.name}", "rx"
                        )
                        break
                    except Exception:
                        team_stream = None

            try:
                cable_out = AsyncOutputWorker(pa=pa, endpoint=cable_ep)
            except Exception as cable_err:
                self.add_log(
                    f"[AVISO] Falló el bus CABLE '{cable_ep.name}' ({cable_err}). Conmutando a salida de respaldo...",
                    "sys",
                )
                cable_out = None
                for cand_str in self.devices_payload["outputs"]:
                    cand_ep = self._resolve_endpoint(cand_str)
                    if cand_ep.index == cable_ep.index:
                        continue
                    try:
                        cable_out = AsyncOutputWorker(pa=pa, endpoint=cand_ep)
                        cable_ep = cand_ep
                        self.config.cable_a_device_name = f"[{cable_ep.index}] {cable_ep.name}"
                        self.config.save()
                        self.add_log(f"[OK] Bus CABLE reasignado con éxito a: {cable_ep.name}", "rx")
                        break
                    except Exception:
                        cable_out = None
                if cable_out is None:
                    raise RuntimeError(f"No se pudo inicializar ningún bus de salida para el juego: {cable_err}")

            try:
                hp_out = AsyncOutputWorker(
                    pa=pa, endpoint=hp_ep, media_buffer=self._media_buffer
                )
            except Exception as hp_err:
                self.add_log(
                    f"[AVISO] Falló endpoint auriculares '{hp_ep.name}' ({hp_err}). Conmutando...",
                    "sys",
                )
                hp_out = None
                for cand_str in self.devices_payload["outputs"]:
                    cand_ep = self._resolve_endpoint(cand_str)
                    if cand_ep.index == hp_ep.index:
                        continue
                    try:
                        hp_out = AsyncOutputWorker(
                            pa=pa, endpoint=cand_ep, media_buffer=self._media_buffer
                        )
                        hp_ep = cand_ep
                        self.config.headphones_device_name = f"[{hp_ep.index}] {hp_ep.name}"
                        self.config.save()
                        self.add_log(f"[OK] Auriculares reasignados a: {hp_ep.name}", "rx")
                        break
                    except Exception:
                        hp_out = None
                if hp_out is None:
                    raise RuntimeError(f"No se pudo inicializar la salida de auriculares: {hp_err}")

            self._hp_out = hp_out
            self.loading = False
            self.running = True
            self.radar = {
                "left": "Escuchando [-45°]",
                "center": "Listo [0°]",
                "right": "Escuchando [+45°]",
            }
            self.add_log(
                f"[ONLINE] Motor WASAPI Full-Duplex activo. Mic: {mic_ep.name} -> {cable_ep.name}",
                "rx",
            )

            prev_tx_allowed = False
            prev_rx_allowed = False
            while not self._stop_event.is_set():
                self.mic_db = mic_stream.last_rms_dbfs if mic_stream else -100.0
                self.chat_db = team_stream.last_rms_dbfs if team_stream else -100.0

                # FLUJO 1: TU MICRÓFONO -> JUEGO
                tx_allowed = not self.config.mute_tx and not self._hotkeys.is_tx_muted and (
                    not self.config.ptt_enabled or self._hotkeys.is_ptt_active
                )
                if tx_allowed:
                    mic_segs = vad_mic.process_available(rb_mic)
                    if mic_segs and self.config.barge_in_enabled and hp_out is not None:
                        hp_out.flush()

                    for seg in mic_segs:
                        if self.config.mute_tx or self._hotkeys.is_tx_muted:
                            break
                        t0 = time.perf_counter()
                        my_src = self.config.my_source_lang
                        my_tgt = self.config.my_target_lang

                        with self._ai_lock:
                            if my_tgt == "en":
                                out_text = stt.translate_es_to_en(seg, source_lang=my_src)
                                det_lang = my_src
                            else:
                                raw_txt, det_lang, _ = stt.transcribe_any(seg, source_lang=my_src)
                                out_text = (
                                    mt.translate(raw_txt, src_lang=det_lang, tgt_lang=my_tgt)
                                    if raw_txt
                                    else ""
                                )

                            if not out_text:
                                continue

                            tts_pcm, tts_rate = tts.synthesize(
                                out_text,
                                voice=self.config.my_voice_profile,
                                lang=my_tgt,
                                speed=self.config.tts_speed_in,
                                ref_audio_16k=seg,
                            )
                        if self.config.mute_tx or self._hotkeys.is_tx_muted:
                            break
                        dt_ms = int((time.perf_counter() - t0) * 1000.0)
                        self.last_latency_ms = dt_ms

                        self.add_log(
                            f"[TX TÚ ({det_lang.upper()} -> {my_tgt.upper()}) | {dt_ms}ms]: \"{out_text}\"",
                            "tx",
                        )
                        if cable_out is not None:
                            cable_out.enqueue(tts_pcm, sample_rate=tts_rate)
                        if self.config.tactical_chirp_enabled and hp_out is not None:
                            hp_out.play_chirp(volume=0.35)
                            vad_mic.notify_output_audio(150.0)
                        if self.config.monitor_own_voice and hp_out is not None:
                            hp_out.enqueue(tts_pcm, sample_rate=tts_rate, pan_left=0.5, pan_right=0.5)
                            vad_mic.notify_output_audio(
                                (len(tts_pcm) / max(1, tts_rate)) * 1000.0 + 150.0
                            )
                else:
                    if prev_tx_allowed:
                        vad_mic.reset_state()
                        if cable_out is not None:
                            cable_out.flush()
                    _ = rb_mic.read_all()

                # FLUJO 2: COMPAÑEROS -> TUS AURICULARES (3D + AUTO-IDIOMA)
                rx_allowed = not self.config.mute_rx and not self._hotkeys.is_rx_muted
                if team_stream is not None and rx_allowed:
                    team_segs = vad_team.process_available(rb_team)
                    for seg in team_segs:
                        if self.config.mute_rx or self._hotkeys.is_rx_muted:
                            break
                        t0 = time.perf_counter()
                        tm_src = self.config.teammate_source_lang
                        tm_tgt = self.config.teammate_target_lang

                        fut_spk = executor.submit(speaker_tracker.identify_speaker, seg)
                        with self._ai_lock:
                            src_text, det_lang, _ = stt.transcribe_any(seg, source_lang=tm_src)
                        spk = fut_spk.result()

                        if not src_text or self.config.mute_rx or self._hotkeys.is_rx_muted:
                            continue

                        spd = self.config.tts_speed_in
                        if hp_out.pending_items >= 1:
                            spd = min(1.30, spd + 0.10)

                        with self._ai_lock:
                            translated_text = mt.translate(
                                src_text, src_lang=det_lang, tgt_lang=tm_tgt
                            )
                            if not translated_text:
                                continue

                            tts_pcm, tts_rate = tts.synthesize(
                                translated_text,
                                voice=spk.voice_name,
                                lang=tm_tgt,
                                speed=spd,
                            )
                        if self.config.mute_rx or self._hotkeys.is_rx_muted:
                            break
                        dt_ms = int((time.perf_counter() - t0) * 1000.0)
                        self.last_latency_ms = dt_ms

                        tag_info = f"{spk.voice_name} ({det_lang.upper()}->{tm_tgt.upper()})"
                        if spk.pos_label == "IZQUIERDA":
                            self.radar["left"] = tag_info
                        elif spk.pos_label == "CENTRO":
                            self.radar["center"] = tag_info
                        else:
                            self.radar["right"] = tag_info

                        self.add_log(
                            f"[RX {spk.label} ({spk.pos_label}) | {det_lang.upper()}->{tm_tgt.upper()} | {dt_ms}ms]: \"{src_text}\" => \"{translated_text}\"",
                            "rx",
                        )
                        pan_l = spk.pan_left if self.spatial_3d else 0.7
                        pan_r = spk.pan_right if self.spatial_3d else 0.7
                        hp_out.enqueue(
                            tts_pcm, sample_rate=tts_rate, pan_left=pan_l, pan_right=pan_r
                        )
                        vad_mic.notify_output_audio(
                            (len(tts_pcm) / max(1, tts_rate)) * 1000.0 + 150.0
                        )
                else:
                    if prev_rx_allowed:
                        vad_team.reset_state()
                        if hp_out is not None:
                            hp_out.flush()
                    if rb_team is not None:
                        _ = rb_team.read_all()

                if (
                    self.config.media_mode_enabled
                    and rx_allowed
                    and self._media_buffer.available_to_read > 0
                ):
                    vad_mic.notify_output_audio(250.0)

                prev_tx_allowed = tx_allowed
                prev_rx_allowed = rx_allowed

                time.sleep(0.015)

        except Exception as exc:
            self.add_log(f"[ERROR MOTOR]: {exc}", "err")
        finally:
            executor.shutdown(wait=False)
            if mic_stream is not None:
                mic_stream.stop()
            if team_stream is not None:
                team_stream.stop()
            if cable_out is not None:
                cable_out.close()
            if hp_out is not None:
                hp_out.close()
            if pa is not None:
                pa.terminate()
            if rb_mic is not None:
                rb_mic.close()
            if rb_team is not None:
                rb_team.close()

            self._hp_out = None
            self._speaker_tracker = None
            self.loading = False
            self.running = False
            self.mic_db = -100.0
            self.chat_db = -100.0
            self.radar = {"left": "Inactivo", "center": "Inactivo", "right": "Inactivo"}
            self.add_log("[SISTEMA] Motor detenido.", "sys")
            if sys.platform == "win32":
                ctypes.windll.ole32.CoUninitialize()

    def clone_my_voice_async(self) -> None:
        def _cl() -> None:
            if sys.platform == "win32":
                ctypes.windll.ole32.CoInitialize(None)
            rb = None
            pa = None
            st = None
            try:
                self.add_log("[CLONADOR STEALTH] Grabando 5 segundos de tu voz... Habla ahora con naturalidad por tu micrófono.", "sys")
                mic_ep = self._resolve_endpoint(self.config.mic_device_name)
                rb = SPSCRingBuffer(name="pb_clone_tmp", capacity_seconds=8.0, sample_rate=16000, create=True)
                pa = pyaudio.PyAudio()
                st = WASAPIInputStream(pa=pa, endpoint=mic_ep, ring_buffer=rb)
                st.start()
                for _ in range(50):
                    self.mic_db = st.last_rms_dbfs
                    time.sleep(0.1)
                st.stop()
                st = None
                pa.terminate()
                pa = None
                self.mic_db = -100.0

                buf = np.empty((rb.available_to_read,), dtype=np.float32)
                rb.read_into(buf)
                rb.close()
                rb = None

                if len(buf) < 1600:
                    raise RuntimeError("No se detectó suficiente audio en la grabación. Verifica la ganancia del micrófono.")

                f0 = OnlineSpeakerTracker._estimate_pitch_hz(buf)
                self.models_dir.mkdir(parents=True, exist_ok=True)
                np.savez(self.models_dir / "my_voice_clone.npz", f0=np.float32(f0))
                self.config.my_voice_profile = "my_cloned"
                self.config.save()
                self.add_log(f"[CLONADOR STEALTH] Voz clonada con éxito. Tu registro natural F0 = {f0:.1f} Hz ha sido activado.", "rx")
            except Exception as e:
                self.add_log(f"[ERROR CLONADOR]: {e}", "err")
            finally:
                if st is not None:
                    try:
                        st.stop()
                    except Exception:
                        pass
                if pa is not None:
                    try:
                        pa.terminate()
                    except Exception:
                        pass
                if rb is not None:
                    try:
                        rb.close()
                    except Exception:
                        pass
                if sys.platform == "win32":
                    ctypes.windll.ole32.CoUninitialize()

        threading.Thread(target=_cl, daemon=True).start()

    def download_all_models_async(self) -> None:
        def _dl() -> None:
            try:
                self.add_log("[INSTALADOR] Verificando Silero VAD v5...", "sys")
                ensure_silero_model(self.models_dir)

                ws_path = self.models_dir / "wespeaker_campp.onnx"
                if not ws_path.exists():
                    self.add_log("[INSTALADOR] Descargando WeSpeaker CAM++ ONNX...", "sys")
                    urllib.request.urlretrieve(WESPEAKER_ONNX_URL, ws_path)

                self.add_log("[INSTALADOR] Verificando Kokoro-82M INT8 y Voces...", "sys")
                ensure_kokoro_assets(self.models_dir)

                self.add_log("[INSTALADOR] Verificando Traductor 200 Idiomas NLLB-200...", "sys")
                ct2_dir = self.models_dir / "nllb_200_ct2_int8"
                snapshot_download(
                    repo_id=NLLB_CT2_REPO, local_dir=str(ct2_dir), local_dir_use_symlinks=False
                )
                tok_file = ct2_dir / "tokenizer.json"
                if not tok_file.exists():
                    hf_hub_download(
                        repo_id=NLLB_TOKENIZER_REPO,
                        filename="tokenizer.json",
                        local_dir=str(ct2_dir),
                        local_dir_use_symlinks=False,
                    )

                self.add_log(
                    f"[INSTALADOR] Verificando Faster-Whisper ({self.config.whisper_model_size})...",
                    "sys",
                )
                WhisperTranslatorEngine(model_size=self.config.whisper_model_size)

                xtts_dir = get_xtts_models_dir()
                if not ((xtts_dir / "model.pth").exists() and (xtts_dir / "config.json").exists()):
                    self.add_log("[INSTALADOR] Descargando XTTS-v2 Cinematográfico...", "sys")
                    xtts_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id="tts-hub/XTTS-v2",
                        local_dir=str(xtts_dir),
                        local_dir_use_symlinks=False,
                    )

                self.add_log(
                    "[INSTALADOR] Todos los modelos e idiomas están listos y verificados.", "rx"
                )
            except Exception as exc:
                self.add_log(f"[ERROR INSTALADOR]: {exc}", "err")

        threading.Thread(target=_dl, daemon=True).start()


CONTROLLER = StitchBackendController()
HTML_FILE = get_ui_html_path()

_ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})


def _is_allowed_host_header(raw_host: str | None) -> bool:
    if not raw_host:
        return False
    cleaned = raw_host.strip()
    if not cleaned or any(ch.isspace() for ch in cleaned):
        return False
    try:
        parsed = urlparse(f"http://{cleaned}")
        _ = parsed.port
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    if parsed.netloc.lower() != cleaned.lower():
        return False
    return (parsed.hostname or "").lower() in _ALLOWED_LOCAL_HOSTS


_MAX_POST_BODY_BYTES = 4 * 1024 * 1024
_EXTENSION_MEDIA_PATHS: frozenset[str] = frozenset(
    {"/api/media/ping", "/api/media/translate", "/api/media/channels", "/ws/media"}
)


def _is_allowed_origin_header(raw_origin: str) -> bool:
    cleaned = raw_origin.strip()
    if not cleaned or cleaned.lower() == "null":
        return False
    try:
        parsed = urlparse(cleaned)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    scheme = parsed.scheme.lower()
    if scheme in ("chrome-extension", "moz-extension"):
        ext_id = parsed.netloc
        return bool(ext_id) and all(ch.isalnum() or ch == "-" for ch in ext_id)
    if scheme not in ("http", "https"):
        return False
    return (parsed.hostname or "").lower() in _ALLOWED_LOCAL_HOSTS


class StitchHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _is_request_authorized(self) -> bool:
        if not _is_allowed_host_header(self.headers.get("Host")):
            return False

        origin = self.headers.get("Origin")
        if origin is not None and not _is_allowed_origin_header(origin):
            return False

        client_hdr = (self.headers.get("X-ProtoBridge-Client") or "").strip()
        is_ext = (client_hdr == "BrowserExtension") or (
            origin is not None
            and _is_allowed_origin_header(origin)
            and origin.strip().lower().startswith(("chrome-extension://", "moz-extension://"))
        )

        sec_fetch_site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if sec_fetch_site == "cross-site" and not is_ext:
            return False

        parsed_path = urlparse(self.path).path
        if is_ext and parsed_path not in _EXTENSION_MEDIA_PATHS:
            return False

        if getattr(self, "command", "").upper() == "POST":
            if parsed_path in _EXTENSION_MEDIA_PATHS:
                if client_hdr not in ("WebUI", "BrowserExtension"):
                    return False
            elif client_hdr != "WebUI":
                return False

        return True

    def _drain_small_body(self, max_bytes: int = 65536) -> None:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError:
            return
        if 0 < length <= max_bytes:
            self.rfile.read(length)

    def _send_cors_headers_if_allowed(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and _is_allowed_origin_header(origin):
            self.send_header("Access-Control-Allow-Origin", origin.strip())
            self.send_header("Vary", "Origin")

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers_if_allowed()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if not self._is_request_authorized():
            self.send_error(403, "Forbidden")
            return
        self.send_response(204)
        self._send_cors_headers_if_allowed()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-ProtoBridge-Client")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._is_request_authorized():
            self.send_error(403, "Forbidden")
            return

        parsed = urlparse(self.path)
        if parsed.path == "/":
            html_bytes = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if parsed.path == "/ws/media":
            upgrade_hdr = (self.headers.get("Upgrade") or "").strip().lower()
            ws_key = (self.headers.get("Sec-WebSocket-Key") or "").strip()
            if upgrade_hdr != "websocket" or not ws_key:
                self.send_error(426, "Upgrade Required")
                return
            accept_key = compute_ws_accept_key(ws_key)
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key)
            self.end_headers()
            self.close_connection = True
            ws = WebSocketConnection(self.rfile, self.wfile)
            try:
                CONTROLLER._media_service.serve_websocket(ws)
            finally:
                ws.close()
            return

        if parsed.path == "/api/media/ping":
            self._send_json(
                {
                    "ok": True,
                    "app": "ProtoBridge",
                    "running": CONTROLLER.running,
                    "source_lang": CONTROLLER.config.teammate_source_lang,
                    "target_lang": CONTROLLER.config.teammate_target_lang,
                    "voice": CONTROLLER.config.my_voice_profile,
                    "ducking_db": CONTROLLER.config.media_ducking_db,
                    "media_auto_sync": CONTROLLER.config.media_auto_sync,
                    "media_delay_ms": CONTROLLER.config.media_delay_ms,
                    "speed": CONTROLLER.config.tts_speed_in,
                    "tts_backend": CONTROLLER.config.media_tts_backend,
                    "diarization": CONTROLLER.config.media_diarization_enabled,
                    "media_whisper_model_size": CONTROLLER.config.media_whisper_model_size,
                    "ws_path": "/ws/media",
                    "mute_tx": bool(CONTROLLER.config.mute_tx or CONTROLLER._hotkeys.is_tx_muted),
                    "mute_rx": bool(CONTROLLER.config.mute_rx or CONTROLLER._hotkeys.is_rx_muted),
                    "xtts_available": CONTROLLER._xtts_client.is_available(),
                    "xtts_running": CONTROLLER._xtts_client.is_running(),
                }
            )
            return

        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            try:
                since = int(qs.get("since", ["0"])[0])
            except ValueError:
                self.send_error(400, "Bad Request")
                return
            self._send_json(
                {
                    "running": CONTROLLER.running,
                    "loading": CONTROLLER.loading,
                    "mic_db": CONTROLLER.mic_db,
                    "chat_db": CONTROLLER.chat_db,
                    "last_latency_ms": CONTROLLER.last_latency_ms,
                    "radar": CONTROLLER.radar,
                    "logs": CONTROLLER.get_logs_since(since),
                    "devices": CONTROLLER.devices_payload,
                    "config": asdict(CONTROLLER.config),
                    "models_status": CONTROLLER.get_models_status(),
                    "xtts_available": CONTROLLER._xtts_client.is_available(),
                    "xtts_running": CONTROLLER._xtts_client.is_running(),
                }
            )
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if not self._is_request_authorized():
            self._drain_small_body()
            self.send_error(403, "Forbidden")
            return

        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self.send_error(400, "Bad Request")
            return
        if length < 0 or length > _MAX_POST_BODY_BYTES:
            self.send_error(400, "Bad Request")
            return

        try:
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(raw) if raw.strip() else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, "Bad Request")
            return

        if not isinstance(data, dict):
            self.send_error(400, "Bad Request")
            return

        parsed_path = urlparse(self.path).path

        if parsed_path == "/api/media/translate":
            try:
                res = CONTROLLER.translate_media_payload(data)
            except Exception as exc:
                CONTROLLER.add_log(f"[ERROR CHROME MEDIA]: {exc}", "err")
                res = {"ok": False, "error": str(exc)}
            self._send_json(res)
            return

        if parsed_path == "/api/media/channels":
            mute_tx_arg = bool(data["mute_tx"]) if "mute_tx" in data else None
            mute_rx_arg = bool(data["mute_rx"]) if "mute_rx" in data else None
            state = CONTROLLER.set_channel_mutes(mute_tx=mute_tx_arg, mute_rx=mute_rx_arg)
            CONTROLLER.config.save()
            self._send_json({"ok": True, **state})
            return

        if parsed_path == "/api/toggle":
            CONTROLLER.toggle_engine()
            self._send_json({"ok": True})
            return

        if parsed_path == "/api/config":
            for key, val in data.items():
                if key in AppConfig.__dataclass_fields__ and key not in ("mute_tx", "mute_rx"):
                    setattr(CONTROLLER.config, key, val)
            if "spatial_3d" in data:
                CONTROLLER.spatial_3d = bool(data["spatial_3d"])
            state = CONTROLLER.set_channel_mutes(
                mute_tx=bool(data["mute_tx"]) if "mute_tx" in data else None,
                mute_rx=bool(data["mute_rx"]) if "mute_rx" in data else None,
            )
            CONTROLLER.config.save()
            if not data.get("silent", False):
                CONTROLLER.add_log("[SISTEMA] Configuración guardada en config.json", "sys")
            self._send_json({"ok": True, **state})
            return

        if parsed_path == "/api/action":
            act = data.get("action", "")
            if act == "open_github":
                os.startfile("https://github.com/MCJ-A")
            elif act == "open_linkedin":
                os.startfile("https://www.linkedin.com/in/jhon-cordova-martinez")
            elif act == "open_mmsys":
                subprocess.Popen(["control", "mmsys.cpl"])
            elif act == "clone_voice":
                CONTROLLER.clone_my_voice_async()
            elif act == "download_models":
                CONTROLLER.download_all_models_async()
            elif act == "reset_squad":
                CONTROLLER.reset_squad()
            elif act == "reset_media_speakers":
                CONTROLLER.reset_media_speakers()
            elif act == "prewarm_xtts":
                CONTROLLER.prewarm_xtts_async()
            elif act == "unload_xtts":
                CONTROLLER.unload_xtts()
            elif act == "open_chrome_extension_folder":
                ext_dir = get_browser_extension_dir()
                ext_dir.mkdir(parents=True, exist_ok=True)
                os.startfile(str(ext_dir))
                CONTROLLER.add_log(
                    f"[CHROME BRIDGE] Carpeta abierta: {ext_dir}. En Chrome abre chrome://extensions -> Modo de desarrollador -> Cargar descomprimida.",
                    "sys",
                )
            elif act == "install_vbcable":
                vb_url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
                tmp_zip = Path(os.environ.get("TEMP", ".")) / "VBCABLE.zip"
                tmp_dir = Path(os.environ.get("TEMP", ".")) / "VBCABLE"
                urllib.request.urlretrieve(vb_url, tmp_zip)
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    zf.extractall(tmp_dir)
                os.startfile(str(tmp_dir / "VBCABLE_Setup_x64.exe"), "runas")
            self._send_json({"ok": True})
            return

        self.send_error(404)


def launch_stitch_window(url: str) -> None:
    edge_paths = [
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Microsoft/Edge/Application/msedge.exe",
    ]
    for edge in edge_paths:
        if edge.exists():
            subprocess.Popen(
                [str(edge), f"--app={url}", "--window-size=1220,820"]
            )
            return
    os.startfile(url)


def _build_tray_icon_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (7, 11, 17, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((4, 4, 60, 60), radius=12, outline=(0, 229, 255, 255), width=4)
    draw.polygon([(18, 44), (32, 18), (46, 44)], outline=(245, 158, 11, 255), width=4)
    draw.line((23, 36, 41, 36), fill=(0, 229, 255, 255), width=4)
    return img


def main() -> None:
    set_high_process_priority()
    port = 8765
    server = None
    for p in range(8770, 8810):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), StitchHTTPHandler)
            port = p
            break
        except OSError:
            continue

    if server is None:
        server = ThreadingHTTPServer(("127.0.0.1", port), StitchHTTPHandler)

    url = f"http://127.0.0.1:{port}"
    print(f" [ProtoBridge Studio - Stitch Edition] Servidor activo en {url}")

    server_thread: threading.Thread | None = None
    tray_icon = None
    try:
        import pystray
        import webview

        server_thread = threading.Thread(
            target=server.serve_forever, daemon=True, name="pb_http_server"
        )
        server_thread.start()
        window = webview.create_window(
            "ProtoBridge Studio",
            url,
            width=1280,
            height=840,
            min_size=(960, 640),
            background_color="#070B11",
        )

        def _show_window(_icon=None, _item=None) -> None:
            window.show()
            window.restore()

        def _quit_app(icon=None, _item=None) -> None:
            if icon is not None:
                icon.stop()
            window.destroy()

        def _on_minimized() -> None:
            window.hide()

        window.events.minimized += _on_minimized
        tray_icon = pystray.Icon(
            "ProtoBridge",
            _build_tray_icon_image(),
            "ProtoBridge Studio",
            menu=pystray.Menu(
                pystray.MenuItem("Abrir ProtoBridge", _show_window, default=True),
                pystray.MenuItem("Salir", _quit_app),
            ),
        )
        tray_icon.run_detached()
        webview.start()
    except Exception:
        launch_stitch_window(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        if tray_icon is not None:
            try:
                tray_icon.stop()
            except Exception:
                pass
        CONTROLLER.stop()
        CONTROLLER._hotkeys.stop()
        if server_thread is not None and server_thread.is_alive():
            server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()


