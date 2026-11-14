from __future__ import annotations

import base64
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import secrets
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

from ignait.paths import get_media_tts_python, get_xtts_models_dir, get_xtts_worker_script


def _read_exact_http(resp: http.client.HTTPResponse, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = resp.read(size - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


class XttsWorkerClient:
    """Gestor e IPC seguro del subproceso aislado Python 3.12.8 XTTS-v2 en GPU."""

    def __init__(
        self,
        models_dir: Path | None = None,
        worker_python: Path | None = None,
        worker_script: Path | None = None,
        startup_timeout_s: float = 120.0,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        self.models_dir = models_dir or get_xtts_models_dir()
        self.worker_python = worker_python
        self.worker_script = worker_script or get_xtts_worker_script()
        self.startup_timeout_s = float(startup_timeout_s)
        self._on_log = on_log

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._token: str = ""
        self._last_info: dict[str, Any] = {}
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._reg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xtts_spk_reg")

    def _log(self, msg: str, tag: str = "sys") -> None:
        if self._on_log is not None:
            self._on_log(msg, tag)

    def resolve_python(self) -> Path | None:
        if self.worker_python is not None:
            return self.worker_python if self.worker_python.exists() else None
        return get_media_tts_python()

    def is_available(self) -> bool:
        py = self.resolve_python()
        xtts_dir = (
            self.models_dir
            if (self.models_dir / "config.json").exists()
            else (self.models_dir / "xtts_v2")
        )
        return bool(
            py is not None
            and self.worker_script.exists()
            and (xtts_dir / "model.pth").exists()
        )

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None and self._port is not None

    def last_info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_info)

    def _drain_stderr(self, pipe: Any) -> None:
        try:
            for line in iter(pipe.readline, ""):
                cleaned = line.rstrip()
                if cleaned:
                    self._stderr_tail.append(cleaned)
        except Exception:
            pass

    def ensure_started(self) -> int:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._port is not None:
                return self._port

            self._stop_locked()
            py = self.resolve_python()
            if py is None:
                raise RuntimeError("No se encontró el entorno aislado .venv-media-tts (Python 3.12.8).")
            if not self.worker_script.exists():
                raise FileNotFoundError(f"No existe el worker XTTS: {self.worker_script}")
            xtts_dir = (
                self.models_dir
                if (self.models_dir / "config.json").exists()
                else (self.models_dir / "xtts_v2")
            )
            if not (xtts_dir / "model.pth").exists():
                raise FileNotFoundError(f"Faltan pesos XTTS-v2 en {xtts_dir}")

            self._token = secrets.token_urlsafe(24)
            env = os.environ.copy()
            env["PB_WORKER_TOKEN"] = self._token
            env["PYTHONUNBUFFERED"] = "1"

            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self._log("[XTTS WORKER] Iniciando subproceso aislado Python 3.12.8 CUDA...", "sys")
            proc = subprocess.Popen(
                [str(py), str(self.worker_script), "--models-dir", str(xtts_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )
            self._proc = proc
            self._stderr_tail.clear()
            if proc.stderr is not None:
                threading.Thread(
                    target=self._drain_stderr,
                    args=(proc.stderr,),
                    daemon=True,
                    name="xtts_stderr_drain",
                ).start()

            handshake_result: list[dict[str, Any]] = []
            handshake_err: list[Exception] = []

            def _read_handshake() -> None:
                try:
                    if proc.stdout is None:
                        raise RuntimeError("stdout del worker XTTS no disponible.")
                    raw_line = proc.stdout.readline()
                    if not raw_line:
                        raise RuntimeError("Worker XTTS cerró antes de enviar handshake.")
                    parsed = json.loads(raw_line.strip())
                    if not isinstance(parsed, dict) or not parsed.get("ready"):
                        raise RuntimeError(f"Handshake XTTS inválido: {raw_line[:200]}")
                    handshake_result.append(parsed)
                except Exception as exc:
                    handshake_err.append(exc)

            reader = threading.Thread(target=_read_handshake, daemon=True, name="xtts_handshake")
            reader.start()
            reader.join(timeout=self.startup_timeout_s)

            if reader.is_alive() or handshake_err or not handshake_result:
                tail = " | ".join(list(self._stderr_tail)[-5:])
                self._stop_locked()
                reason = str(handshake_err[0]) if handshake_err else f"Timeout ({self.startup_timeout_s}s)"
                raise RuntimeError(f"No arrancó worker XTTS-v2: {reason}. Log: {tail}")

            info = handshake_result[0]
            self._port = int(info["port"])
            self._last_info = info
            self._log(
                f"[XTTS WORKER] Listo en puerto {self._port} ({info.get('device')}, fp16={info.get('fp16')}, VRAM {info.get('reserved_mb', 0)} MB).",
                "rx",
            )
            return self._port

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        hdrs = {
            "Content-Type": "application/json",
            "X-Worker-Token": self._token,
        }
        if extra:
            hdrs.update(extra)
        return hdrs

    def _stop_locked(self) -> None:
        proc = self._proc
        port = self._port
        self._proc = None
        self._port = None

        if port is not None and self._token:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
                conn.request("POST", "/shutdown", body=b"{}", headers=self._headers())
                _ = conn.getresponse()
                conn.close()
            except Exception:
                pass

        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
        self._reg_executor.shutdown(wait=False)

    def reset_speakers(self) -> None:
        with self._lock:
            if not self.is_running():
                return
            port = self._port
        if port is None:
            return
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
            conn.request("POST", "/reset_speakers", body=b"{}", headers=self._headers())
            resp = conn.getresponse()
            _ = resp.read()
            conn.close()
        except Exception:
            pass

    def register_speaker(self, speaker_key: str, pcm_16k: NDArray[np.float32]) -> bool:
        port = self.ensure_started()
        clipped = np.clip(np.asarray(pcm_16k, dtype=np.float32).reshape(-1), -1.0, 1.0)
        pcm_i16 = (clipped * 32767.0).astype("<i2")
        b64_pcm = base64.b64encode(pcm_i16.tobytes()).decode("ascii")
        payload = json.dumps(
            {"speaker_key": speaker_key, "pcm_s16le_b64": b64_pcm, "sample_rate": 16000}
        ).encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30.0)
        try:
            conn.request("POST", "/register_speaker", body=payload, headers=self._headers())
            resp = conn.getresponse()
            _ = resp.read()
            return resp.status == 200
        finally:
            conn.close()

    def register_speaker_async(
        self,
        speaker_key: str,
        pcm_16k: NDArray[np.float32],
        on_done: Callable[[bool], None] | None = None,
    ) -> Future[bool] | None:
        if not self.is_available():
            return None
        copy_pcm = np.asarray(pcm_16k, dtype=np.float32).copy()

        def _job() -> bool:
            ok = False
            try:
                ok = self.register_speaker(speaker_key, copy_pcm)
            except Exception as exc:
                self._log(f"[XTTS CLONADO AVISO] {speaker_key}: {exc}", "sys")
                ok = False
            if on_done is not None:
                try:
                    on_done(ok)
                except Exception:
                    pass
            return ok

        try:
            return self._reg_executor.submit(_job)
        except RuntimeError:
            return None

    def synthesize_stream(
        self,
        text: str,
        language: str = "es",
        speaker_key: str | None = None,
        fallback_speaker: str | None = None,
        speed: float = 1.05,
        chunk_size: int = 18,
    ) -> Iterator[tuple[NDArray[np.float32], int]]:
        port = self.ensure_started()
        payload = json.dumps(
            {
                "text": text,
                "language": language,
                "speaker_key": speaker_key,
                "fallback_speaker": fallback_speaker,
                "speed": float(speed),
                "chunk_size": int(chunk_size),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30.0)
        try:
            conn.request("POST", "/synthesize_stream", body=payload, headers=self._headers())
            resp = conn.getresponse()
            if resp.status != 200:
                err_txt = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Worker XTTS HTTP {resp.status}: {err_txt}")

            while True:
                hdr = _read_exact_http(resp, 8)
                if len(hdr) < 8:
                    break
                sample_rate, byte_len = struct.unpack("<II", hdr)
                if byte_len == 0:
                    break
                pcm_bytes = _read_exact_http(resp, byte_len)
                aligned = len(pcm_bytes) - (len(pcm_bytes) % 2)
                if aligned <= 0:
                    continue
                pcm_f32 = (
                    np.frombuffer(pcm_bytes[:aligned], dtype="<i2").astype(np.float32) / 32768.0
                )
                yield pcm_f32, int(sample_rate)
        finally:
            conn.close()

    def synthesize_full(
        self,
        text: str,
        language: str = "es",
        speaker_key: str | None = None,
        fallback_speaker: str | None = None,
        speed: float = 1.05,
    ) -> tuple[NDArray[np.float32], int]:
        chunks: list[NDArray[np.float32]] = []
        out_rate = 24000
        for chunk, rate in self.synthesize_stream(
            text=text,
            language=language,
            speaker_key=speaker_key,
            fallback_speaker=fallback_speaker,
            speed=speed,
        ):
            out_rate = rate
            chunks.append(chunk)
        if not chunks:
            return np.zeros((0,), dtype=np.float32), out_rate
        return np.concatenate(chunks).astype(np.float32), out_rate
