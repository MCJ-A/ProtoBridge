from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import dataclass
import io
import json
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable
import wave
import numpy as np
from numpy.typing import NDArray
import soxr

from ignait.config import AppConfig
from ignait.diarization.speaker_id import OnlineSpeakerTracker, SpeakerProfile
from ignait.media.speaker_bank import FEMALE_PRESET_POOL, MediaSpeakerVoiceBank
from ignait.media.stream_buffer import InMemoryAudioRingBuffer
from ignait.media.worker_client import XttsWorkerClient
from ignait.media.ws_protocol import OPCODE_BINARY, OPCODE_TEXT, WebSocketConnection
from ignait.vad.silero import SileroVADSegmenter


def encode_wav_s16le_b64(pcm_f32: NDArray[np.float32], sample_rate: int) -> str:
    """Codifica PCM float32 mono a WAV PCM16 little-endian en Base64 para Web Audio API."""
    clipped = np.clip(np.asarray(pcm_f32, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_i16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def float32_to_s16le_bytes(pcm_f32: NDArray[np.float32]) -> bytes:
    """Convierte audio PCM float32 [-1, 1] a bytes PCM int16 little-endian."""
    clipped = np.clip(np.asarray(pcm_f32, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


_MEDIA_NOISE_TAG_RE = re.compile(
    r"\[[^\]]*\]|\((?:music|música|applause|aplausos|laughter|risas|cheering|sighs?|suspiros?|silence|silencio|noise|ruido|inaudible|sfx)[^)]*\)|[♪♫♬]+",
    re.IGNORECASE,
)
_MEDIA_STUTTER_RE = re.compile(r"\b(\w+)(?:[\s,]+\1){3,}\b", re.IGNORECASE)


def sanitize_media_text(text: str) -> str:
    """Elimina anotaciones no habladas y bucles repetitivos antes de MT/TTS para evitar alucinaciones."""
    clean = _MEDIA_NOISE_TAG_RE.sub(" ", text or "")
    clean = re.sub(r"https?://\S+|www\.\S+", " ", clean)
    clean = _MEDIA_STUTTER_RE.sub(r"\1", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < 2 or not re.search(r"[0-9A-Za-zÀ-ÿА-Яа-я一-龯ぁ-ゔァ-ヴー가-힣]", clean):
        return ""
    return clean


def compute_auto_paced_speed(
    translated_text: str, target_duration_sec: float | None, base_speed: float
) -> float:
    """Ajusta velocidad TTS cuando la traducción supera la ventana de escena para evitar deriva."""
    clamped_base = float(np.clip(base_speed, 0.75, 1.45))
    if target_duration_sec is None or target_duration_sec < 0.65:
        return clamped_base
    clean = translated_text.strip()
    if not clean:
        return clamped_base
    word_count = len(clean.split())
    est_at_base_sec = max(len(clean) / (15.5 * clamped_base), word_count / (2.75 * clamped_base))
    budget_sec = float(target_duration_sec) + 0.35
    if est_at_base_sec <= budget_sec:
        return clamped_base
    ratio = est_at_base_sec / max(budget_sec, 0.4)
    return float(np.clip(clamped_base * ratio, clamped_base, 1.32))


def update_ewma_latency(prev_ewma_ms: float | None, sample_ms: int) -> tuple[float, int]:
    """Estimador EWMA asimétrico de latencia -> devuelve `(ewma_ms, recommended_delay_ms)`."""
    clamped = float(max(80, min(8000, int(sample_ms))))
    if prev_ewma_ms is None:
        ewma = clamped
    else:
        alpha = 0.45 if clamped > prev_ewma_ms else 0.16
        ewma = (1.0 - alpha) * prev_ewma_ms + alpha * clamped
    rec_delay = int(np.clip(round(ewma * 1.25 + 280.0), 750, 4500))
    return ewma, rec_delay


@dataclass
class _MediaTask:
    generation: int
    seq: int
    source_lang: str
    target_lang: str
    voice: str
    speed: float
    tts_backend: str
    diarization: bool
    auto_sync: bool = True
    pcm_16k: NDArray[np.float32] | None = None
    text: str = ""
    play_at_sec: float | None = None
    duration_sec: float | None = None
    capture_media_sec: float | None = None
    capture_delay_sec: float = 0.0
    is_prefetched: bool = False
    speaker_hint: int | None = None
    playback_gen: int = 0
    source_kind: str = "audio"


class MediaPipelineService:
    """Orquestador compartido para Media Mode (HTTP REST y WebSocket binario con Silero + CAM++ + XTTS-v2)."""

    def __init__(
        self,
        models_dir: Path,
        ai_lock: threading.RLock,
        ensure_engines_fn: Callable[[str | None], tuple[Any, Any, Any]],
        get_config_fn: Callable[[], AppConfig],
        xtts_client: XttsWorkerClient | None,
        on_log: Callable[[str, str], None],
        on_latency_ms: Callable[[int], None],
    ) -> None:
        self.models_dir = models_dir
        self._ai_lock = ai_lock
        self._ensure_engines = ensure_engines_fn
        self._get_config = get_config_fn
        self.xtts_client = xtts_client
        self._log = on_log
        self._on_latency_ms = on_latency_ms

        self._tracker_lock = threading.Lock()
        self._speaker_tracker: OnlineSpeakerTracker | None = None
        self.speaker_bank = MediaSpeakerVoiceBank(session_prefix="media")
        self._http_speaker_lock = threading.Lock()
        self._last_http_speaker: tuple[SpeakerProfile, str, str] | None = None
        self._recent_lock = threading.Lock()
        self._recent_keys: deque[str] = deque(maxlen=64)
        self._recent_set: set[str] = set()
        self._rx_muted_override: bool = False
        self._ws_sessions_lock = threading.Lock()
        self._ws_mute_hooks: dict[int, Callable[[bool], None]] = {}

    @property
    def is_rx_muted(self) -> bool:
        return bool(self._rx_muted_override or getattr(self._get_config(), "mute_rx", False))

    def set_rx_muted(self, muted: bool) -> None:
        self._rx_muted_override = bool(muted)
        effective = self.is_rx_muted
        with self._ws_sessions_lock:
            hooks = list(self._ws_mute_hooks.values())
        for hook in hooks:
            try:
                hook(effective)
            except Exception:
                pass

    def _get_speaker_tracker(self, max_speakers: int) -> OnlineSpeakerTracker:
        target_max = max(1, max_speakers)
        with self._tracker_lock:
            if self._speaker_tracker is None:
                self._speaker_tracker = OnlineSpeakerTracker(
                    models_dir=self.models_dir,
                    max_speakers=target_max,
                    similarity_threshold=0.50,
                    limit_to_pool=False,
                    continuity_bonus=0.06,
                    require_new_confirmation=True,
                )
            else:
                self._speaker_tracker.max_speakers = target_max
            return self._speaker_tracker

    def reset_speakers(self) -> None:
        with self._tracker_lock:
            if self._speaker_tracker is not None:
                self._speaker_tracker.reset_squad()
        with self._http_speaker_lock:
            self._last_http_speaker = None
        self.speaker_bank.reset()
        if self.xtts_client is not None:
            self.xtts_client.reset_speakers()
        with self._recent_lock:
            self._recent_keys.clear()
            self._recent_set.clear()

    def _resolve_whisper_size(self, cfg: AppConfig) -> str | None:
        pref = (cfg.media_whisper_model_size or "").strip()
        if not pref or pref == "follow":
            return None
        return pref

    def _should_use_xtts(self, requested_voice: str, backend_pref: str) -> bool:
        if self.xtts_client is None or not self.xtts_client.is_available():
            return False
        pref = (backend_pref or "auto").strip().lower()
        if pref == "kokoro":
            return False
        voice_clean = (requested_voice or "auto").strip().lower()
        if voice_clean not in ("", "auto", "xtts", "cinematic"):
            return False
        return pref in ("xtts_v2", "auto")

    def _identify_speaker_and_register(
        self,
        pcm_16k: NDArray[np.float32],
        max_speakers: int,
        *,
        wait_v1_sync: bool = False,
    ) -> tuple[SpeakerProfile, str, str]:
        tracker = self._get_speaker_tracker(max_speakers)
        spk = tracker.identify_speaker(pcm_16k)
        conf_ok = tracker.last_was_confident
        spk_key, preset_fallback, clone_ver, ref_pcm = self.speaker_bank.ingest_segment(
            spk.speaker_id,
            pcm_16k,
            spk.pitch_hz,
            confidence_ok=conf_ok,
        )
        if clone_ver > 0 and ref_pcm is not None and self.xtts_client is not None:
            spk_id = spk.speaker_id
            fut = self.xtts_client.register_speaker_async(
                spk_key,
                ref_pcm,
                on_done=lambda ok, sid=spk_id: self.speaker_bank.mark_clone_ready(sid) if ok else None,
            )
            if (
                wait_v1_sync
                and clone_ver == 1
                and isinstance(fut, Future)
                and self.xtts_client.is_running()
            ):
                try:
                    fut.result(timeout=1.2)
                except Exception:
                    pass
        return spk, spk_key, preset_fallback

    def _iter_tts_chunks(
        self,
        translated_text: str,
        tgt_lang: str,
        speed: float,
        requested_voice: str,
        backend_pref: str,
        spk: SpeakerProfile | None,
        spk_key: str | None,
        preset_fallback: str | None,
        kokoro_tts: Any,
    ) -> Iterator[tuple[NDArray[np.float32], int, str]]:
        use_xtts = self._should_use_xtts(requested_voice, backend_pref)
        if use_xtts and self.xtts_client is not None:
            clone_ready = bool(spk is not None and self.speaker_bank.is_clone_ready(spk.speaker_id))
            yielded_any = False
            try:
                for pcm_chunk, sr in self.xtts_client.synthesize_stream(
                    text=translated_text,
                    language=tgt_lang,
                    speaker_key=spk_key if clone_ready else None,
                    fallback_speaker=preset_fallback,
                    speed=speed,
                ):
                    if pcm_chunk.size > 0:
                        yielded_any = True
                        yield pcm_chunk, sr, "xtts_v2"
            except Exception as exc:
                self._log(f"[MEDIA XTTS FALLBACK->KOKORO]: {exc}", "sys")
            if yielded_any:
                return

        kokoro_voice = (
            spk.voice_name
            if spk is not None and requested_voice in ("", "auto", "xtts", "cinematic")
            else requested_voice
        )
        if kokoro_voice in ("", "auto", "xtts", "cinematic"):
            if preset_fallback in FEMALE_PRESET_POOL:
                kokoro_voice = "ef_dora" if tgt_lang.lower().startswith("es") else "bf_emma"
            else:
                kokoro_voice = "em_alex" if tgt_lang.lower().startswith("es") else "am_adam"
        with self._ai_lock:
            pcm_k, rate_k = kokoro_tts.synthesize(
                translated_text, voice=kokoro_voice, lang=tgt_lang, speed=speed
            )
        if pcm_k.size > 0:
            yield np.asarray(pcm_k, dtype=np.float32), int(rate_k), "kokoro"

    def _clear_recent_captions(self) -> None:
        with self._recent_lock:
            self._recent_keys.clear()
            self._recent_set.clear()

    def _remember_caption(self, src_lang: str, tgt_lang: str, text: str) -> bool:
        key = f"{src_lang}->{tgt_lang}:{text.strip().lower()}"
        with self._recent_lock:
            if key in self._recent_set:
                return False
            if len(self._recent_keys) == self._recent_keys.maxlen:
                oldest = self._recent_keys.popleft()
                self._recent_set.discard(oldest)
            self._recent_keys.append(key)
            self._recent_set.add(key)
        return True

    def translate_http_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.is_rx_muted:
            return {"ok": True, "empty": True, "rx_muted": True}
        t0 = time.perf_counter()
        cfg = self._get_config()
        src_lang = str(data.get("source_lang") or cfg.teammate_source_lang or "auto")
        tgt_lang = str(data.get("target_lang") or cfg.teammate_target_lang or "es")
        voice = str(data.get("voice") or cfg.my_voice_profile or "am_adam")
        backend_pref = str(data.get("tts_backend") or cfg.media_tts_backend or "auto")
        diarization = bool(data.get("diarization", cfg.media_diarization_enabled))
        try:
            raw_speed = float(data.get("speed") or cfg.tts_speed_in or 1.10)
        except (TypeError, ValueError):
            raw_speed = float(cfg.tts_speed_in or 1.10)
        speed = float(np.clip(raw_speed, 0.6, 1.8))

        raw_text = str(data.get("text") or "").strip()
        pcm = np.empty((0,), dtype=np.float32)

        if not raw_text:
            b64_audio = str(data.get("pcm_s16le_b64") or "")
            if not b64_audio:
                return {"ok": True, "empty": True}
            try:
                raw_bytes = base64.b64decode(b64_audio, validate=True)
            except (ValueError, TypeError):
                return {"ok": False, "error": "Invalid base64 PCM audio"}
            aligned_len = len(raw_bytes) - (len(raw_bytes) % 2)
            if aligned_len <= 0:
                return {"ok": True, "empty": True}
            pcm = (
                np.frombuffer(raw_bytes[:aligned_len], dtype="<i2").astype(np.float32) / 32768.0
            )
            try:
                in_rate = int(data.get("sample_rate") or 16000)
            except (TypeError, ValueError):
                in_rate = 16000
            if in_rate <= 0:
                in_rate = 16000
            if in_rate != 16000 and len(pcm) > 0:
                pcm = soxr.resample(pcm, in_rate, 16000, quality="HQ").astype(np.float32)
            if len(pcm) < 1600:
                return {"ok": True, "empty": True}
            rms_dbfs = 20.0 * float(np.log10(float(np.sqrt(np.mean(np.square(pcm)))) + 1e-9))
            if rms_dbfs < -46.0:
                return {"ok": True, "empty": True}

        whisper_pref = self._resolve_whisper_size(cfg) if self._should_use_xtts(voice, backend_pref) else None
        with self._ai_lock:
            stt, mt, tts = self._ensure_engines(whisper_pref)

        auto_voice = voice.strip().lower() in ("", "auto", "xtts", "cinematic")
        eff_max_spk = cfg.media_max_speakers if diarization else 1
        spk: SpeakerProfile | None = None
        spk_key: str | None = None
        preset_fallback: str | None = None
        if pcm.size >= 1600 and auto_voice:
            try:
                spk, spk_key, preset_fallback = self._identify_speaker_and_register(
                    pcm,
                    eff_max_spk,
                    wait_v1_sync=self._should_use_xtts(voice, backend_pref),
                )
                with self._http_speaker_lock:
                    self._last_http_speaker = (spk, spk_key, preset_fallback)
            except Exception as exc:
                self._log(f"[MEDIA DIARIZATION AVISO]: {exc}", "sys")
        elif auto_voice:
            with self._http_speaker_lock:
                if self._last_http_speaker is not None:
                    spk, spk_key, preset_fallback = self._last_http_speaker

        with self._ai_lock:
            if raw_text:
                src_text = sanitize_media_text(raw_text)
                det_lang = "en" if src_lang == "auto" else src_lang
            else:
                src_text_raw, det_lang, _ = stt.transcribe_any(pcm, source_lang=src_lang)
                src_text = sanitize_media_text(src_text_raw)

            if not src_text or not src_text.strip():
                return {"ok": True, "empty": True}
            if det_lang.lower() == tgt_lang.lower():
                return {
                    "ok": True,
                    "empty": True,
                    "same_lang": True,
                    "src_text": src_text,
                    "detected_lang": det_lang,
                }
            translated_text = mt.translate(src_text, src_lang=det_lang, tgt_lang=tgt_lang)

        if not translated_text or not translated_text.strip():
            return {"ok": True, "empty": True}

        pcm_chunks: list[NDArray[np.float32]] = []
        tts_rate = 24000
        engine_used = "kokoro"
        for ch, rate, eng in self._iter_tts_chunks(
            translated_text=translated_text,
            tgt_lang=tgt_lang,
            speed=speed,
            requested_voice=voice,
            backend_pref=backend_pref,
            spk=spk,
            spk_key=spk_key,
            preset_fallback=preset_fallback,
            kokoro_tts=tts,
        ):
            pcm_chunks.append(ch)
            tts_rate = rate
            engine_used = eng

        if not pcm_chunks:
            return {"ok": True, "empty": True}

        full_pcm = np.concatenate(pcm_chunks).astype(np.float32) if len(pcm_chunks) > 1 else pcm_chunks[0]
        wav_b64 = encode_wav_s16le_b64(full_pcm, tts_rate)
        dt_ms = int((time.perf_counter() - t0) * 1000.0)
        self._on_latency_ms(dt_ms)

        spk_label = spk.label if spk is not None else "Locutor"
        spk_id = spk.speaker_id if spk is not None else 0
        self._log(
            f"[CHROME MEDIA {engine_used.upper()} • {spk_label} ({det_lang.upper()}->{tgt_lang.upper()}) | {dt_ms}ms]: \"{src_text}\" => \"{translated_text}\"",
            "rx",
        )
        return {
            "ok": True,
            "empty": False,
            "src_text": src_text,
            "translated_text": translated_text,
            "detected_lang": det_lang,
            "target_lang": tgt_lang,
            "speaker_id": spk_id,
            "speaker_label": spk_label,
            "tts_engine": engine_used,
            "latency_ms": dt_ms,
            "sample_rate": int(tts_rate),
            "wav_base64": wav_b64,
        }

    def serve_websocket(self, ws: WebSocketConnection) -> None:
        """Atiende una sesión WebSocket binaria con Silero VAD v5 + diarización + TTS streaming."""
        cfg = self._get_config()
        session_cfg: dict[str, Any] = {
            "source_lang": cfg.teammate_source_lang or "auto",
            "target_lang": cfg.teammate_target_lang or "es",
            "voice": "auto",
            "speed": float(cfg.tts_speed_in or 1.10),
            "tts_backend": cfg.media_tts_backend or "xtts_v2",
            "diarization": bool(cfg.media_diarization_enabled),
            "auto_sync": bool(cfg.media_auto_sync),
            "voice_probe_only": False,
        }
        ring = InMemoryAudioRingBuffer(capacity_seconds=18.0, sample_rate=16000)
        vad = SileroVADSegmenter(
            models_dir=self.models_dir,
            speech_threshold=cfg.vad_threshold,
            min_silence_ms=max(260, int(cfg.vad_silence_ms)),
            min_speech_ms=340,
            max_speech_seconds=4.5,
            min_rms_dbfs=-44.0,
        )
        job_queue: queue.Queue[_MediaTask | None] = queue.Queue(maxsize=10)
        probe_queue: queue.Queue[tuple[int, NDArray[np.float32]] | None] = queue.Queue(maxsize=6)
        state_lock = threading.Lock()
        generation = 0
        client_playback_gen = 0
        speaker_generation = 0
        latest_speaker: tuple[SpeakerProfile, str, str] | None = None
        seq_counter = 0
        ewma_latency_ms: float | None = None
        recommended_delay_ms = int(cfg.media_delay_ms or 2500)
        latest_media_sec: float | None = None
        latest_capture_delay_sec = 0.0

        def _parse_opt_int(raw: Any) -> int | None:
            if raw is None or isinstance(raw, bool):
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def _parse_opt_finite_float(raw: Any) -> float | None:
            if raw is None or isinstance(raw, bool):
                return None
            try:
                val = float(raw)
                return val if bool(np.isfinite(val)) else None
            except (TypeError, ValueError):
                return None

        def _clear_queue() -> None:
            while True:
                try:
                    job_queue.get_nowait()
                except queue.Empty:
                    break

        def _clear_probe_queue() -> None:
            while True:
                try:
                    probe_queue.get_nowait()
                except queue.Empty:
                    break

        def _reset_session_playback(cmd: dict[str, Any], reset_spk: bool = False) -> None:
            nonlocal generation, client_playback_gen, speaker_generation, latest_speaker, latest_media_sec
            req_pgen = _parse_opt_int(cmd.get("playback_gen"))
            req_time = _parse_opt_finite_float(cmd.get("current_time_sec"))
            with state_lock:
                generation += 1
                client_playback_gen = (
                    max(client_playback_gen + 1, req_pgen)
                    if req_pgen is not None
                    else client_playback_gen + 1
                )
                latest_media_sec = req_time
                _clear_queue()
                if reset_spk:
                    speaker_generation += 1
                    latest_speaker = None
                    _clear_probe_queue()
                ring.clear()
                vad.reset_state()
            self._clear_recent_captions()
            if reset_spk:
                self.reset_speakers()

        def _probe_loop() -> None:
            nonlocal latest_speaker
            while not ws.closed:
                item = probe_queue.get()
                if item is None:
                    break
                spk_gen, seg_pcm = item
                with state_lock:
                    if spk_gen != speaker_generation:
                        continue
                    probe_diar = bool(session_cfg["diarization"])
                if seg_pcm.shape[0] < 8000:
                    continue
                live_cfg = self._get_config()
                probe_max = live_cfg.media_max_speakers if probe_diar else 1
                try:
                    identified = self._identify_speaker_and_register(
                        seg_pcm, probe_max
                    )
                    probe_confident = self._get_speaker_tracker(probe_max).last_was_confident
                except Exception as exc:
                    self._log(f"[MEDIA PROBE AVISO]: {exc}", "sys")
                    continue
                with state_lock:
                    if spk_gen == speaker_generation and (
                        latest_speaker is None or probe_confident
                    ):
                        latest_speaker = identified

        def _worker_loop() -> None:
            nonlocal ewma_latency_ms, recommended_delay_ms, latest_speaker
            while not ws.closed:
                task = job_queue.get()
                if task is None:
                    break
                with state_lock:
                    cur_gen = generation
                    cur_pgen = client_playback_gen
                    cur_media_sec = latest_media_sec
                    cur_latest_spk = latest_speaker
                if task.generation != cur_gen or task.playback_gen != cur_pgen or self.is_rx_muted:
                    continue
                if (
                    task.is_prefetched
                    and task.play_at_sec is not None
                    and cur_media_sec is not None
                    and cur_media_sec > task.play_at_sec + max(1.2, task.duration_sec or 0.0) + 1.5
                ):
                    continue

                t0 = time.perf_counter()
                live_cfg = self._get_config()
                whisper_pref = (
                    self._resolve_whisper_size(live_cfg)
                    if (task.pcm_16k is not None and self._should_use_xtts(task.voice, task.tts_backend))
                    else None
                )
                try:
                    with self._ai_lock:
                        stt, mt, tts = self._ensure_engines(whisper_pref)
                except Exception as exc:
                    ws.send_json({"type": "error", "error": str(exc)})
                    continue

                auto_voice = task.voice.strip().lower() in ("", "auto", "xtts", "cinematic")
                eff_max_spk = live_cfg.media_max_speakers if task.diarization else 1
                spk: SpeakerProfile | None = None
                spk_key: str | None = None
                preset_fallback: str | None = None
                if task.pcm_16k is not None and auto_voice:
                    try:
                        with state_lock:
                            pre_spk_gen = speaker_generation
                        spk, spk_key, preset_fallback = self._identify_speaker_and_register(
                            task.pcm_16k,
                            eff_max_spk,
                            wait_v1_sync=self._should_use_xtts(task.voice, task.tts_backend),
                        )
                        with state_lock:
                            if pre_spk_gen == speaker_generation:
                                latest_speaker = (spk, spk_key, preset_fallback)
                    except Exception:
                        spk = None
                elif (
                    task.pcm_16k is None
                    and auto_voice
                    and cur_latest_spk is not None
                ):
                    spk, spk_key, preset_fallback = cur_latest_spk

                with self._ai_lock:
                    if task.text:
                        src_text = sanitize_media_text(task.text)
                        det_lang = "en" if task.source_lang == "auto" else task.source_lang
                    elif task.pcm_16k is not None:
                        src_text_raw, det_lang, _ = stt.transcribe_any(
                            task.pcm_16k, source_lang=task.source_lang
                        )
                        src_text = sanitize_media_text(src_text_raw)
                    else:
                        continue

                    if not src_text or not src_text.strip():
                        continue
                    if det_lang.lower() == task.target_lang.lower():
                        ws.send_json(
                            {
                                "type": "same_lang",
                                "seq": task.seq,
                                "src_text": src_text,
                                "detected_lang": det_lang,
                            }
                        )
                        continue
                    translated_text = mt.translate(
                        src_text, src_lang=det_lang, tgt_lang=task.target_lang
                    )

                if not translated_text or not translated_text.strip():
                    continue

                with state_lock:
                    if (
                        task.generation != generation
                        or task.playback_gen != client_playback_gen
                        or self.is_rx_muted
                    ):
                        continue

                effective_speed = (
                    compute_auto_paced_speed(translated_text, task.duration_sec, task.speed)
                    if task.auto_sync
                    else task.speed
                )

                spk_id = spk.speaker_id if spk is not None else (task.speaker_hint or 0)
                spk_label = spk.label if spk is not None else "Locutor"
                first_chunk_ms: int | None = None
                chunk_idx = 0
                for pcm_chunk, sr, eng in self._iter_tts_chunks(
                    translated_text=translated_text,
                    tgt_lang=task.target_lang,
                    speed=effective_speed,
                    requested_voice=task.voice,
                    backend_pref=task.tts_backend,
                    spk=spk,
                    spk_key=spk_key,
                    preset_fallback=preset_fallback,
                    kokoro_tts=tts,
                ):
                    with state_lock:
                        if (
                            task.generation != generation
                            or task.playback_gen != client_playback_gen
                            or self.is_rx_muted
                            or ws.closed
                        ):
                            break
                    if first_chunk_ms is None:
                        first_chunk_ms = int((time.perf_counter() - t0) * 1000.0)
                        self._on_latency_ms(first_chunk_ms)
                        ewma_latency_ms, recommended_delay_ms = update_ewma_latency(
                            ewma_latency_ms, first_chunk_ms
                        )
                    pcm_bytes = float32_to_s16le_bytes(pcm_chunk)
                    ws.send_media_packet(
                        {
                            "type": "tts_chunk",
                            "seq": task.seq,
                            "playback_gen": task.playback_gen,
                            "source_kind": task.source_kind,
                            "chunk_index": chunk_idx,
                            "is_final": False,
                            "sample_rate": int(sr),
                            "src_text": src_text,
                            "translated_text": translated_text,
                            "detected_lang": det_lang,
                            "target_lang": task.target_lang,
                            "speaker_id": spk_id,
                            "speaker_label": spk_label,
                            "tts_engine": eng,
                            "latency_ms": first_chunk_ms,
                            "ewma_latency_ms": round(ewma_latency_ms or float(first_chunk_ms), 1),
                            "recommended_delay_ms": recommended_delay_ms,
                            "effective_speed": round(effective_speed, 3),
                            "play_at_sec": task.play_at_sec,
                            "duration_sec": task.duration_sec,
                            "capture_media_sec": task.capture_media_sec,
                            "capture_delay_sec": task.capture_delay_sec,
                            "is_prefetched": task.is_prefetched,
                        },
                        pcm_bytes,
                    )
                    chunk_idx += 1

                if chunk_idx > 0:
                    total_ms = int((time.perf_counter() - t0) * 1000.0)
                    ws.send_media_packet(
                        {
                            "type": "tts_chunk",
                            "seq": task.seq,
                            "playback_gen": task.playback_gen,
                            "source_kind": task.source_kind,
                            "chunk_index": chunk_idx,
                            "is_final": True,
                            "sample_rate": 24000,
                            "latency_ms": first_chunk_ms or total_ms,
                            "ewma_latency_ms": round(ewma_latency_ms or float(total_ms), 1),
                            "recommended_delay_ms": recommended_delay_ms,
                            "speaker_id": spk_id,
                            "speaker_label": spk_label,
                            "play_at_sec": task.play_at_sec,
                            "is_prefetched": task.is_prefetched,
                        },
                        b"",
                    )
                    ws.send_json(
                        {
                            "type": "sync_telemetry",
                            "seq": task.seq,
                            "playback_gen": task.playback_gen,
                            "latency_ms": first_chunk_ms or total_ms,
                            "ewma_latency_ms": round(ewma_latency_ms or float(total_ms), 1),
                            "recommended_delay_ms": recommended_delay_ms,
                        }
                    )
                    self._log(
                        f"[WS MEDIA • {spk_label} ({det_lang.upper()}->{task.target_lang.upper()}) | {first_chunk_ms or total_ms}ms]: \"{src_text}\" => \"{translated_text}\"",
                        "rx",
                    )

        def _on_rx_mute_change(muted: bool) -> None:
            nonlocal last_notified_rx_muted
            next_muted = bool(muted)
            with state_lock:
                if next_muted == last_notified_rx_muted:
                    return
                last_notified_rx_muted = next_muted
            if next_muted:
                _reset_session_playback({}, reset_spk=False)
            if not ws.closed:
                try:
                    ws.send_json({"type": "rx_mute_state", "rx_muted": next_muted})
                except OSError:
                    pass

        last_notified_rx_muted = self.is_rx_muted

        probe_worker = threading.Thread(
            target=_probe_loop, daemon=True, name="pb_media_speaker_probe"
        )
        probe_worker.start()
        worker = threading.Thread(target=_worker_loop, daemon=True, name="pb_media_ws_worker")
        worker.start()

        ws.send_json(
            {
                "type": "hello",
                "xtts_available": bool(self.xtts_client and self.xtts_client.is_available()),
                "vad": "silero_v5",
                "diarization": "wespeaker_campp",
                "recommended_delay_ms": recommended_delay_ms,
                "rx_muted": last_notified_rx_muted,
            }
        )
        with self._ws_sessions_lock:
            self._ws_mute_hooks[id(ws)] = _on_rx_mute_change

        try:
            while not ws.closed:
                msg = ws.read_message()
                if msg is None:
                    break
                opcode, payload = msg

                if opcode == OPCODE_BINARY:
                    if self.is_rx_muted:
                        continue
                    aligned = len(payload) - (len(payload) % 2)
                    if aligned <= 0:
                        continue
                    pcm_f32 = (
                        np.frombuffer(payload[:aligned], dtype="<i2").astype(np.float32) / 32768.0
                    )
                    ring.write(pcm_f32)
                    segments = vad.process_available(ring)
                    is_probe_only = bool(session_cfg["voice_probe_only"])
                    is_diar_on = bool(session_cfg["diarization"])
                    probe_auto_voice = str(session_cfg["voice"]).strip().lower() in (
                        "",
                        "auto",
                        "xtts",
                        "cinematic",
                    )
                    if is_probe_only:
                        if not probe_auto_voice:
                            continue
                        with state_lock:
                            cur_spk_gen = speaker_generation
                        for seg in segments:
                            while probe_queue.full():
                                try:
                                    probe_queue.get_nowait()
                                except queue.Empty:
                                    break
                            probe_queue.put_nowait((cur_spk_gen, seg))
                        continue

                    for seg in segments:
                        seg_dur_s = seg.shape[0] / 16000.0
                        with state_lock:
                            seq_counter += 1
                            cap_sec = latest_media_sec
                            cap_delay = latest_capture_delay_sec
                            task = _MediaTask(
                                generation=generation,
                                seq=seq_counter,
                                source_lang=str(session_cfg["source_lang"]),
                                target_lang=str(session_cfg["target_lang"]),
                                voice=str(session_cfg["voice"]),
                                speed=float(session_cfg["speed"]),
                                tts_backend=str(session_cfg["tts_backend"]),
                                diarization=is_diar_on,
                                auto_sync=bool(session_cfg["auto_sync"]),
                                pcm_16k=seg,
                                duration_sec=round(seg_dur_s, 3),
                                capture_media_sec=cap_sec,
                                capture_delay_sec=cap_delay,
                                playback_gen=client_playback_gen,
                                source_kind="audio",
                            )
                        while job_queue.full():
                            try:
                                job_queue.get_nowait()
                            except queue.Empty:
                                break
                        job_queue.put_nowait(task)
                    continue

                if opcode == OPCODE_TEXT:
                    try:
                        cmd = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(cmd, dict):
                        continue
                    cmd_type = str(cmd.get("type") or "")

                    if cmd_type == "media_clock":
                        raw_t = cmd.get("current_time_sec")
                        raw_d = cmd.get("delay_sec")
                        with state_lock:
                            if raw_t is not None:
                                try:
                                    latest_media_sec = float(raw_t)
                                except (TypeError, ValueError):
                                    pass
                            if raw_d is not None:
                                try:
                                    latest_capture_delay_sec = max(0.0, float(raw_d))
                                except (TypeError, ValueError):
                                    pass
                    elif cmd_type == "configure":
                        for key in ("source_lang", "target_lang", "voice", "tts_backend"):
                            if key in cmd and cmd[key] is not None:
                                session_cfg[key] = str(cmd[key])
                        if "speed" in cmd and cmd["speed"] is not None:
                            try:
                                session_cfg["speed"] = float(np.clip(float(cmd["speed"]), 0.6, 1.8))
                            except (TypeError, ValueError):
                                pass
                        if "diarization" in cmd:
                            session_cfg["diarization"] = bool(cmd["diarization"])
                        if "auto_sync" in cmd:
                            session_cfg["auto_sync"] = bool(cmd["auto_sync"])
                        if "voice_probe_only" in cmd:
                            session_cfg["voice_probe_only"] = bool(cmd["voice_probe_only"])
                        cfg_pgen = _parse_opt_int(cmd.get("playback_gen"))
                        if cfg_pgen is not None:
                            with state_lock:
                                if cfg_pgen > client_playback_gen:
                                    client_playback_gen = cfg_pgen
                        if cmd.get("reset_speakers"):
                            with state_lock:
                                speaker_generation += 1
                                latest_speaker = None
                                _clear_probe_queue()
                            self.reset_speakers()
                    elif cmd_type == "reset_playback":
                        _reset_session_playback(cmd, reset_spk=False)
                    elif cmd_type == "reset_speakers":
                        _reset_session_playback(cmd, reset_spk=True)
                    elif cmd_type == "translate_caption":
                        if self.is_rx_muted:
                            continue
                        txt = str(cmd.get("text") or "").strip()
                        if len(txt) < 3:
                            continue
                        req_pgen = _parse_opt_int(cmd.get("playback_gen"))
                        with state_lock:
                            if req_pgen is not None and req_pgen < client_playback_gen:
                                continue
                            if req_pgen is not None and req_pgen > client_playback_gen:
                                client_playback_gen = req_pgen
                            task_pgen = req_pgen if req_pgen is not None else client_playback_gen
                        src_l = str(cmd.get("source_lang") or session_cfg["source_lang"])
                        tgt_l = str(cmd.get("target_lang") or session_cfg["target_lang"])
                        if not self._remember_caption(src_l, tgt_l, txt):
                            continue
                        raw_play_at = cmd.get("play_at_sec")
                        raw_dur = cmd.get("duration_sec")
                        play_at_val = _parse_opt_finite_float(raw_play_at)
                        dur_val = _parse_opt_finite_float(raw_dur)
                        with state_lock:
                            if task_pgen < client_playback_gen:
                                continue
                            seq_counter += 1
                            cur_seq = int(cmd.get("seq") or seq_counter)
                            task = _MediaTask(
                                generation=generation,
                                seq=cur_seq,
                                source_lang=src_l,
                                target_lang=tgt_l,
                                voice=str(cmd.get("voice") or session_cfg["voice"]),
                                speed=float(cmd.get("speed") or session_cfg["speed"]),
                                tts_backend=str(cmd.get("tts_backend") or session_cfg["tts_backend"]),
                                diarization=bool(session_cfg["diarization"]),
                                auto_sync=bool(session_cfg["auto_sync"]),
                                text=txt,
                                play_at_sec=play_at_val,
                                duration_sec=dur_val,
                                is_prefetched=bool(cmd.get("is_prefetched", play_at_val is not None)),
                                playback_gen=task_pgen,
                                source_kind="caption",
                            )
                        while job_queue.full():
                            try:
                                job_queue.get_nowait()
                            except queue.Empty:
                                break
                        job_queue.put_nowait(task)
        finally:
            with self._ws_sessions_lock:
                self._ws_mute_hooks.pop(id(ws), None)
            try:
                probe_queue.put_nowait(None)
            except queue.Full:
                _clear_probe_queue()
                probe_queue.put_nowait(None)
            try:
                job_queue.put_nowait(None)
            except queue.Full:
                _clear_queue()
                job_queue.put_nowait(None)
            probe_worker.join(timeout=2.0)
            worker.join(timeout=2.0)
