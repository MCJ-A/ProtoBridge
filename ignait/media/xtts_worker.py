from __future__ import annotations

import sys

# Garantiza que stdout permanezca limpio para el handshake JSON de 1 línea hacia el proceso padre.
_HANDSHAKE_OUT = sys.stdout
sys.stdout = sys.stderr

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import io
import json
import os
from pathlib import Path
import re
import struct
import threading
import time
from typing import Any
from urllib.parse import urlparse
import wave

import numpy as np
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

SAMPLE_RATE_24K = 24000
_SUPPORTED_XTTS_LANGS: frozenset[str] = frozenset(
    {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "zh-cn",
        "hu",
        "ko",
        "ja",
        "hi",
    }
)


def _normalize_xtts_lang(lang: str) -> str:
    clean = (lang or "es").strip().lower().replace("_", "-")
    if clean in _SUPPORTED_XTTS_LANGS:
        return clean
    base = clean.split("-")[0]
    if base == "zh":
        return "zh-cn"
    if base in _SUPPORTED_XTTS_LANGS:
        return base
    return "es"


def _prepare_reference_audio(pcm_f32: np.ndarray, sample_rate: int) -> np.ndarray:
    """Recorta silencios extremos y normaliza el volumen RMS de voz antes de extraer latentes XTTS."""
    arr = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr

    # Eliminar DC offset antes de medir energía
    arr = arr - float(np.mean(arr))
    hop = max(160, int(0.025 * sample_rate))  # 25 ms
    n_frames = arr.size // hop
    if n_frames >= 16:
        frames = arr[: n_frames * hop].reshape(n_frames, hop)
        rms_per_frame = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        p90 = float(np.percentile(rms_per_frame, 90))
        gate = max(0.008, p90 * 0.14)
        active = rms_per_frame >= gate
        # Dilatación de ±1 trama para conservar ataques consonánticos
        padded_active = active.copy()
        if n_frames > 1:
            padded_active[1:] |= active[:-1]
            padded_active[:-1] |= active[1:]
        if int(np.sum(padded_active)) * hop >= int(0.8 * sample_rate):
            arr = frames[padded_active].reshape(-1)

    rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
    if rms > 1e-4:
        gain = float(np.clip(0.09 / rms, 0.35, 10.0))
        arr = arr * gain

    peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    if peak > 0.85:
        arr = arr * (0.85 / peak)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


_NOISE_TAG_RE = re.compile(
    r"\[[^\]]*\]|\((?:music|música|applause|aplausos|laughter|risas|cheering|sighs?|suspiros?|silence|silencio|noise|ruido|inaudible|sfx)[^)]*\)|\*[^*]*\*",
    re.IGNORECASE,
)


def _sanitize_xtts_text(text: str) -> str:
    """Limpia etiquetas no habladas, bucles de repetición y cierra puntuación para evitar alucinaciones."""
    clean = _NOISE_TAG_RE.sub(" ", text or "")
    clean = re.sub(r"[♪♫♬✨🔥❤️😂👍]+", " ", clean)
    clean = re.sub(r"https?://\S+|www\.\S+", " ", clean)
    clean = clean.replace("...", ",").replace("…", ",")
    clean = re.sub(r"\b(\w+)(?:[\s,]+\1){2,}\b", r"\1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" \t\r\n,;:-–—\"'")
    if not clean or not re.search(r"[0-9A-Za-zÀ-ÿА-Яа-я一-龯ぁ-ゔァ-ヴー가-힣]", clean):
        return ""
    if clean.isupper() and len(clean) > 4:
        clean = clean[0] + clean[1:].lower()
    if clean[-1] not in ".!?。！？":
        clean += "."
    return clean


def _encode_wav_s16le(pcm_f32: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(pcm_f32, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


class XttsEngineService:
    """Servicio XTTS-v2 en GPU con caché de latentes por hablante y streaming por chunks."""

    def __init__(self, models_dir: Path, device: str = "cuda", use_fp16: bool = True) -> None:
        self.models_dir = models_dir
        self.xtts_dir = models_dir if (models_dir / "config.json").exists() else (models_dir / "xtts_v2")
        if not (self.xtts_dir / "model.pth").exists():
            raise FileNotFoundError(f"No se encontró model.pth XTTS-v2 en {self.xtts_dir}")

        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.use_fp16 = bool(use_fp16 and self.use_cuda)

        self._model_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._speaker_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._builtin_speakers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        cfg = XttsConfig()
        cfg.load_json(str(self.xtts_dir / "config.json"))
        self.model = Xtts(cfg)
        self.model.load_checkpoint(
            cfg,
            checkpoint_dir=str(self.xtts_dir),
            use_deepspeed=False,
            vocab_path=str(self.xtts_dir / "vocab.json"),
        )
        self.model.to(self.device)
        self.model.eval()

        spk_file = self.xtts_dir / "speakers_xtts.pth"
        if spk_file.exists():
            raw_spks = torch.load(str(spk_file), map_location="cpu", weights_only=True)
            for name, entry in raw_spks.items():
                if isinstance(entry, dict) and "gpt_cond_latent" in entry and "speaker_embedding" in entry:
                    self._builtin_speakers[str(name)] = (
                        entry["gpt_cond_latent"].float().cpu(),
                        entry["speaker_embedding"].float().cpu(),
                    )

        self._warmup()

    def _warmup(self) -> None:
        if not self._builtin_speakers:
            return
        first_name = "Claribel Dervla" if "Claribel Dervla" in self._builtin_speakers else next(iter(self._builtin_speakers))
        g_cpu, s_cpu = self._builtin_speakers[first_name]
        with self._model_lock, torch.inference_mode():
            # Precalentar también los extractores de clonado para eliminar el retardo en el 1er hablante
            dummy_ref = torch.zeros((1, int(22050 * 1.2)), dtype=torch.float32, device=self.device)
            _ = self.model.get_gpt_cond_latents(dummy_ref, 22050, length=6, chunk_length=6)
            _ = self.model.get_speaker_embedding(dummy_ref, 22050)

            g = g_cpu.to(self.device)
            s = s_cpu.to(self.device)
            if self.use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    for _ in self.model.inference_stream(
                        "Listo.", "es", g, s, stream_chunk_size=16, enable_text_splitting=False
                    ):
                        pass
            else:
                for _ in self.model.inference_stream(
                    "Listo.", "es", g, s, stream_chunk_size=16, enable_text_splitting=False
                ):
                    pass

    @property
    def builtin_names(self) -> list[str]:
        return sorted(self._builtin_speakers.keys())

    def register_speaker_pcm(self, speaker_key: str, pcm_f32: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
        arr = _prepare_reference_audio(pcm_f32, int(sample_rate))
        if arr.size < max(1600, int(0.4 * sample_rate)):
            raise ValueError("Clip de referencia demasiado corto (<0.4s).")

        wav_t = torch.from_numpy(arr.copy()).unsqueeze(0)
        if sample_rate != 22050:
            wav_t = torchaudio.functional.resample(wav_t, int(sample_rate), 22050)
        wav_t = torch.clamp(wav_t[:, : 22050 * 12], -1.0, 1.0)

        with self._model_lock, torch.inference_mode():
            wav_dev = wav_t.to(self.device)
            gpt_cond = self.model.get_gpt_cond_latents(wav_dev, 22050, length=12, chunk_length=6)
            spk_emb = self.model.get_speaker_embedding(wav_dev, 22050)
            lat_pair = (gpt_cond.detach().float().cpu(), spk_emb.detach().float().cpu())

        with self._cache_lock:
            self._speaker_cache[speaker_key] = lat_pair

        return {"ok": True, "speaker_key": speaker_key, "duration_s": round(arr.size / float(sample_rate), 2)}

    def reset_speakers(self) -> None:
        with self._cache_lock:
            self._speaker_cache.clear()

    def _resolve_latents(self, speaker_key: str | None, fallback_speaker: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        with self._cache_lock:
            if speaker_key and speaker_key in self._speaker_cache:
                return self._speaker_cache[speaker_key]

        for candidate in (fallback_speaker, "Andrew Chipper", "Luis Moray", "Claribel Dervla"):
            if candidate and candidate in self._builtin_speakers:
                return self._builtin_speakers[candidate]

        if self._builtin_speakers:
            return next(iter(self._builtin_speakers.values()))
        raise RuntimeError("No hay latentes ni voces base cargadas para XTTS-v2.")

    def iter_synthesis_chunks(
        self,
        text: str,
        language: str = "es",
        speaker_key: str | None = None,
        fallback_speaker: str | None = None,
        speed: float = 1.05,
        chunk_size: int = 18,
    ):
        clean_text = _sanitize_xtts_text(text)
        if not clean_text:
            return
        norm_lang = _normalize_xtts_lang(language)
        eff_speed = float(np.clip(speed, 0.70, 1.45))
        eff_chunk = max(8, min(40, int(chunk_size)))
        g_cpu, s_cpu = self._resolve_latents(speaker_key, fallback_speaker)

        with self._model_lock, torch.inference_mode():
            g = g_cpu.to(self.device)
            s = s_cpu.to(self.device)
            ctx_mgr = torch.amp.autocast("cuda", dtype=torch.float16) if self.use_fp16 else _nullcontext()
            with ctx_mgr:
                stream = self.model.inference_stream(
                    clean_text,
                    norm_lang,
                    g,
                    s,
                    stream_chunk_size=eff_chunk,
                    temperature=0.40,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=30,
                    top_p=0.74,
                    do_sample=True,
                    speed=eff_speed,
                    enable_text_splitting=True,
                )
                for chunk in stream:
                    arr = chunk.detach().float().cpu().numpy().reshape(-1)
                    if arr.size > 0:
                        yield np.clip(arr, -1.0, 1.0).astype(np.float32)

    def vram_info_mb(self) -> dict[str, float]:
        if not self.use_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
        }

    def release_cuda_cache(self) -> None:
        if self.use_cuda:
            with self._model_lock:
                torch.cuda.empty_cache()


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def _make_handler(
    service: XttsEngineService,
    token: str,
    shutdown_event: threading.Event,
) -> type[BaseHTTPRequestHandler]:
    class _WorkerHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _authorized(self) -> bool:
            if not token:
                return True
            given = self.headers.get("X-Worker-Token") or ""
            return hmac.compare_digest(given, token)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_body(self, max_bytes: int = 16 * 1024 * 1024) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return b""
            if length <= 0 or length > max_bytes:
                return b""
            return self.rfile.read(length)

        def do_GET(self) -> None:
            if not self._authorized():
                self._send_json(403, {"ok": False, "error": "Forbidden"})
                return
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(
                    {
                        "ok": True,
                        "device": service.device,
                        "fp16": service.use_fp16,
                        **service.vram_info_mb(),
                    }
                )
                return
            self._send_json(404, {"ok": False, "error": "Not Found"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._read_body(65536)
                self._send_json(403, {"ok": False, "error": "Forbidden"})
                return
            path = urlparse(self.path).path

            if path == "/shutdown":
                self._send_json(200, {"ok": True})
                shutdown_event.set()
                return

            if path == "/reset_speakers":
                service.reset_speakers()
                self._send_json(200, {"ok": True})
                return

            if path == "/unload":
                service.release_cuda_cache()
                self._send_json(200, {"ok": True, **service.vram_info_mb()})
                return

            raw = self._read_body()
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"ok": False, "error": "Invalid JSON"})
                return

            if path == "/register_speaker":
                spk_key = str(data.get("speaker_key") or "").strip()
                b64_pcm = str(data.get("pcm_s16le_b64") or "")
                sr = int(data.get("sample_rate") or 16000)
                if not spk_key or not b64_pcm:
                    self._send_json(400, {"ok": False, "error": "Missing speaker_key or pcm_s16le_b64"})
                    return
                try:
                    raw_pcm = base64.b64decode(b64_pcm, validate=True)
                    aligned = len(raw_pcm) - (len(raw_pcm) % 2)
                    pcm_f32 = np.frombuffer(raw_pcm[:aligned], dtype="<i2").astype(np.float32) / 32768.0
                    res = service.register_speaker_pcm(spk_key, pcm_f32, sample_rate=sr)
                    self._send_json(200, res)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return

            if path == "/synthesize":
                try:
                    chunks = list(
                        service.iter_synthesis_chunks(
                            text=str(data.get("text") or ""),
                            language=str(data.get("language") or "es"),
                            speaker_key=data.get("speaker_key"),
                            fallback_speaker=data.get("fallback_speaker"),
                            speed=float(data.get("speed") or 1.05),
                            chunk_size=int(data.get("chunk_size") or 18),
                        )
                    )
                    full_pcm = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
                    wav_bytes = _encode_wav_s16le(full_pcm, SAMPLE_RATE_24K)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(wav_bytes)))
                    self.end_headers()
                    self.wfile.write(wav_bytes)
                except torch.cuda.OutOfMemoryError as exc:
                    service.release_cuda_cache()
                    self._send_json(503, {"ok": False, "error": f"CUDA OOM: {exc}"})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return

            if path == "/synthesize_stream":
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for pcm_chunk in service.iter_synthesis_chunks(
                        text=str(data.get("text") or ""),
                        language=str(data.get("language") or "es"),
                        speaker_key=data.get("speaker_key"),
                        fallback_speaker=data.get("fallback_speaker"),
                        speed=float(data.get("speed") or 1.05),
                        chunk_size=int(data.get("chunk_size") or 18),
                    ):
                        pcm_i16 = (np.clip(pcm_chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                        frame = struct.pack("<II", SAMPLE_RATE_24K, len(pcm_i16)) + pcm_i16
                        self.wfile.write(frame)
                        self.wfile.flush()
                    self.wfile.write(struct.pack("<II", SAMPLE_RATE_24K, 0))
                    self.wfile.flush()
                    self.close_connection = True
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                except torch.cuda.OutOfMemoryError:
                    service.release_cuda_cache()
                    self.close_connection = True
                except Exception:
                    self.close_connection = True
                return

            self._send_json(404, {"ok": False, "error": "Not Found"})

    return _WorkerHTTPHandler


def _start_parent_stdin_watchdog(shutdown_event: threading.Event) -> None:
    def _watch() -> None:
        try:
            while not shutdown_event.is_set():
                ch = sys.stdin.read(1)
                if not ch:
                    break
        except Exception:
            pass
        shutdown_event.set()

    threading.Thread(target=_watch, daemon=True, name="xtts_parent_watchdog").start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignait Isolated XTTS-v2 GPU Worker")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("PB_WORKER_TOKEN", "").strip()
    shutdown_event = threading.Event()
    _start_parent_stdin_watchdog(shutdown_event)

    service = XttsEngineService(
        models_dir=args.models_dir,
        device=args.device,
        use_fp16=not args.no_fp16,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(service, token, shutdown_event))
    server.daemon_threads = True
    port = int(server.server_address[1])

    handshake = {
        "ready": True,
        "port": port,
        "device": service.device,
        "fp16": service.use_fp16,
        "speakers_count": len(service.builtin_names),
        **service.vram_info_mb(),
    }
    _HANDSHAKE_OUT.write(json.dumps(handshake, ensure_ascii=False) + "\n")
    _HANDSHAKE_OUT.flush()

    srv_thread = threading.Thread(target=server.serve_forever, daemon=True, name="xtts_http_srv")
    srv_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
