from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import logging
import queue
import threading
import time
from types import MappingProxyType
import numpy as np
from numpy.typing import NDArray
import pyaudiowpatch as pyaudio
import soxr

from protobridge.audio.devices import AudioEndpoint
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.system.os_priority import mmcss_pro_audio

logger = logging.getLogger(__name__)

TARGET_INTERNAL_RATE: int = 16000
VAD_FRAME_SAMPLES: int = 512

_CHIRP_PRESET_VOLUMES: tuple[float, ...] = (
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.75,
    0.8,
    1.0,
)
_CHIRP_DEFAULT_PAN_LEFT: float = 0.6
_CHIRP_DEFAULT_PAN_RIGHT: float = 0.6

_WASAPI_RECOVER_MAX_RETRIES: int = 5
_WASAPI_RECOVER_BASE_DELAY: float = 0.05  # 50ms, 100ms, 200ms, 400ms, 800ms
_RECOVERABLE_PORTAUDIO_ERRNOS: frozenset[int] = frozenset(
    {-9999, -9996, -9988, -9986, -9985, -2004287484}
)
_RECOVERABLE_WASAPI_TOKENS: tuple[str, ...] = (
    "AUDCLNT_E_DEVICE_INVALIDATED",
    "0X88890004",
    "88890004",
    "-9999",
    "-9996",
    "UNANTICIPATED HOST ERROR",
    "INVALID DEVICE",
    "DEVICE UNAVAILABLE",
)


class StreamState(str, Enum):
    """Estados operativos del enlace WASAPI para observabilidad y auto-reconexión."""

    RUNNING = "RUNNING"
    FAULTED = "FAULTED"
    RECONNECTING = "RECONNECTING"


@dataclass(frozen=True)
class PlaybackItem:
    pcm: NDArray[np.float32]
    sample_rate: int
    pan_left: float = 1.0
    pan_right: float = 1.0
    generation: int = 0
    chirp_volume: float | None = None


def generate_tactical_chirp(
    sample_rate: int = 16000,
    duration_seconds: float = 0.020,
    start_freq: float = 1200.0,
    end_freq: float = 2400.0,
) -> NDArray[np.float32]:
    """Genera un tono chirp táctico acústico de confirmación (20 ms @ 16 kHz = 320 muestras).

    Aplica modulación lineal de frecuencia y ventana de Hanning para prevenir clics espectrales.
    Garantiza forma fija (320,), tipo float32 y valores acotados estrictamente en [-1.0, 1.0].
    """
    num_samples = int(sample_rate * duration_seconds)
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.float32)

    t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False, dtype=np.float32)
    chirp_rate = (end_freq - start_freq) / (2.0 * duration_seconds)
    phase = 2.0 * np.pi * (start_freq * t + chirp_rate * (t**2))
    raw_tone = np.sin(phase).astype(np.float32)

    window = np.hanning(num_samples).astype(np.float32)
    windowed = raw_tone * window

    max_val = float(np.max(np.abs(windowed)))
    if max_val > 1e-6:
        normalized = windowed / max_val
    else:
        normalized = windowed

    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _build_precalculated_chirps(
    base_chirp: NDArray[np.float32],
) -> MappingProxyType[float, NDArray[np.float32]]:
    """Precalcula variantes inmutables por volumen del chirp táctico al cargar el módulo."""
    table: dict[float, NDArray[np.float32]] = {}
    for vol in _CHIRP_PRESET_VOLUMES:
        arr = (base_chirp * np.float32(vol)).astype(np.float32)
        arr.setflags(write=False)
        table[vol] = arr
    return MappingProxyType(table)


# Constante precalculada en memoria para latencia de despacho 0 ms
TACTICAL_CHIRP_PCM: NDArray[np.float32] = generate_tactical_chirp()
TACTICAL_CHIRP_PCM.setflags(write=False)
PRECALCULATED_CHIRPS: MappingProxyType[float, NDArray[np.float32]] = _build_precalculated_chirps(
    TACTICAL_CHIRP_PCM
)


def _nearest_chirp_key(volume: float) -> float:
    if volume in PRECALCULATED_CHIRPS:
        return volume
    return min(PRECALCULATED_CHIRPS, key=lambda k: abs(k - volume))


def _is_wasapi_disconnection_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError):
        return True
    err_no = getattr(exc, "errno", None)
    if isinstance(err_no, int) and err_no in _RECOVERABLE_PORTAUDIO_ERRNOS:
        return True
    msg_upper = str(exc).upper()
    return any(tok in msg_upper for tok in _RECOVERABLE_WASAPI_TOKENS)


def _safe_close_stream(stream: pyaudio.Stream | None) -> None:
    """Detiene y cierra un stream PortAudio sin propagar excepciones ni cerrar PyAudio."""
    if stream is None:
        return
    try:
        stream.stop_stream()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass


def _reopen_stream_with_backoff(
    open_fn: Callable[[], pyaudio.Stream],
    logger_ctx: str,
    should_abort: Callable[[], bool] | None = None,
    max_retries: int = _WASAPI_RECOVER_MAX_RETRIES,
    base_delay: float = _WASAPI_RECOVER_BASE_DELAY,
) -> pyaudio.Stream:
    """Reintenta reabrir un stream WASAPI con backoff exponencial hasta `max_retries` veces."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if should_abort is not None and should_abort():
            raise RuntimeError(f"[{logger_ctx}] Recuperación abortada por detención del stream.")

        delay = base_delay * (2**attempt)
        logger.warning(
            "[%s] Stream WASAPI invalidado. Reintento %d/%d tras %.3fs...",
            logger_ctx,
            attempt + 1,
            max_retries,
            delay,
        )
        time.sleep(delay)

        if should_abort is not None and should_abort():
            raise RuntimeError(f"[{logger_ctx}] Recuperación abortada durante backoff.")

        try:
            return open_fn()
        except Exception as exc:
            if not _is_wasapi_disconnection_error(exc):
                raise
            last_exc = exc
            logger.warning(
                "[%s] Fallo al reabrir stream en intento %d/%d: %s",
                logger_ctx,
                attempt + 1,
                max_retries,
                exc,
            )

    if last_exc is None:
        raise RuntimeError(f"[{logger_ctx}] No se pudo reabrir el stream WASAPI.")
    raise last_exc


class MediaLookaheadBuffer:
    """Buffer circular pre-asignado de retardo lookahead con Smart Ducking para Modo Media."""

    def __init__(
        self,
        sample_rate: int = TARGET_INTERNAL_RATE,
        max_delay_seconds: float = 6.0,
        delay_ms: float = 2500.0,
        duck_db: float = -14.0,
        normal_gain: float = 0.85,
        enabled: bool = False,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.capacity: int = max(sample_rate, int(sample_rate * max_delay_seconds))
        self._buf: NDArray[np.float32] = np.zeros((self.capacity,), dtype=np.float32)
        self._lock = threading.Lock()
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._primed: bool = False
        self.normal_gain: float = float(normal_gain)
        self._current_gain: float = float(normal_gain)
        self.enabled: bool = False
        self.delay_samples: int = 0
        self.duck_gain: float = float(10.0 ** (duck_db / 20.0))
        self.configure(enabled=enabled, delay_ms=delay_ms, duck_db=duck_db)

    def configure(
        self,
        enabled: bool,
        delay_ms: float = 2500.0,
        duck_db: float = -14.0,
    ) -> None:
        max_ms = (self.capacity / self.sample_rate) * 1000.0 - 100.0
        clamped_ms = float(np.clip(delay_ms, 0.0, max(0.0, max_ms)))
        new_delay_samples = int((clamped_ms / 1000.0) * self.sample_rate)
        new_duck_gain = float(10.0 ** (float(np.clip(duck_db, -40.0, 0.0)) / 20.0))
        with self._lock:
            was_enabled = self.enabled
            self.enabled = bool(enabled)
            self.delay_samples = new_delay_samples
            self.duck_gain = new_duck_gain
            if self.enabled and (not was_enabled):
                self._buf.fill(0.0)
                self._write_pos = 0
                self._read_pos = 0
                self._primed = self.delay_samples == 0
                self._current_gain = self.normal_gain

    @property
    def available_to_read(self) -> int:
        with self._lock:
            if not self.enabled or not self._primed:
                return 0
            return max(0, self._write_pos - self._read_pos)

    def write(self, samples_16k: NDArray[np.float32]) -> None:
        if not self.enabled:
            return
        n = int(samples_16k.shape[0])
        if n <= 0:
            return
        with self._lock:
            if not self.enabled:
                return
            if n >= self.capacity:
                samples_16k = samples_16k[-self.capacity :]
                n = self.capacity
            w_idx = self._write_pos % self.capacity
            first = min(n, self.capacity - w_idx)
            self._buf[w_idx : w_idx + first] = samples_16k[:first]
            rem = n - first
            if rem > 0:
                self._buf[:rem] = samples_16k[first:]
            self._write_pos += n
            if not self._primed and (self._write_pos - self._read_pos) >= self.delay_samples:
                self._primed = True
            max_lag = max(
                self.delay_samples + self.sample_rate, self.capacity - self.sample_rate // 2
            )
            if (self._write_pos - self._read_pos) > max_lag:
                self._read_pos = self._write_pos - self.delay_samples

    def read_mixed_16k(self, num_samples: int, ducking: bool = False) -> NDArray[np.float32]:
        if num_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        out = np.zeros((num_samples,), dtype=np.float32)
        with self._lock:
            if not self.enabled or not self._primed:
                return out
            avail = self._write_pos - self._read_pos
            if avail <= 0:
                return out
            take = min(num_samples, avail)
            r_idx = self._read_pos % self.capacity
            first = min(take, self.capacity - r_idx)
            out[:first] = self._buf[r_idx : r_idx + first]
            rem = take - first
            if rem > 0:
                out[first:take] = self._buf[:rem]
            self._read_pos += take

            target_gain = self.duck_gain if ducking else self.normal_gain
            if abs(self._current_gain - target_gain) > 1e-4 and take > 1:
                ramp = np.linspace(self._current_gain, target_gain, take, dtype=np.float32)
                out[:take] *= ramp
            else:
                out[:take] *= np.float32(target_gain)
            self._current_gain = target_gain
        return out


class WASAPIInputStream:
    def __init__(
        self,
        pa: pyaudio.PyAudio,
        endpoint: AudioEndpoint,
        ring_buffer: SPSCRingBuffer,
        media_buffer: MediaLookaheadBuffer | None = None,
    ) -> None:
        self._pa = pa
        self._endpoint = endpoint
        self._ring_buffer = ring_buffer
        self.media_buffer: MediaLookaheadBuffer | None = media_buffer

        self._hw_rate: int = endpoint.default_sample_rate
        self._hw_channels: int = max(
            1,
            endpoint.max_input_channels
            if endpoint.max_input_channels > 0
            else endpoint.max_output_channels,
        )
        self._frames_per_buffer: int = max(128, int(self._hw_rate * 0.016))

        self._resampler: soxr.ResampleStream | None = self._create_resampler()

        self.last_rms_dbfs: float = -100.0
        self.state: StreamState = StreamState.RUNNING
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stream: pyaudio.Stream | None = None
        self._monitor_thread: threading.Thread | None = None

    def _create_resampler(self) -> soxr.ResampleStream | None:
        if self._hw_rate == TARGET_INTERNAL_RATE:
            return None
        return soxr.ResampleStream(
            self._hw_rate,
            TARGET_INTERNAL_RATE,
            num_channels=1,
            dtype="float32",
            quality="HQ",
        )

    def _open_stream(self) -> pyaudio.Stream:
        stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._hw_channels,
            rate=self._hw_rate,
            input=True,
            input_device_index=self._endpoint.index,
            frames_per_buffer=self._frames_per_buffer,
            stream_callback=self._callback,
        )
        try:
            stream.start_stream()
        except Exception:
            _safe_close_stream(stream)
            raise
        return stream

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status_flags: int,
    ) -> tuple[None, int]:
        if in_data is None:
            return (None, pyaudio.paContinue)

        try:
            pcm = np.frombuffer(in_data, dtype=np.float32)
            if self._hw_channels > 1:
                pcm = pcm.reshape(-1, self._hw_channels).mean(axis=1)

            rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12))
            self.last_rms_dbfs = 20.0 * float(np.log10(rms))

            if self._resampler is not None:
                pcm_16k: NDArray[np.float32] = self._resampler.resample_chunk(pcm, last=False)
            else:
                pcm_16k = pcm

            if pcm_16k.shape[0] > 0:
                self._ring_buffer.write(pcm_16k)
                if self.media_buffer is not None and self.media_buffer.enabled:
                    self.media_buffer.write(pcm_16k)
        except Exception as exc:
            logger.error("Error procesando frame de entrada WASAPI: %s", exc)
            return (None, pyaudio.paAbort)

        return (None, pyaudio.paContinue)

    def check_and_recover(self) -> bool:
        """Verifica el estado del stream de entrada WASAPI y lo reabre ante invalidación."""
        if self._stop_event.is_set():
            return False

        with self._lock:
            if self._stop_event.is_set():
                return False

            needs_recovery = False
            if self._stream is None:
                needs_recovery = True
            else:
                try:
                    if not self._stream.is_active():
                        needs_recovery = True
                except Exception as exc:
                    if not _is_wasapi_disconnection_error(exc):
                        raise
                    needs_recovery = True

            if not needs_recovery:
                return False

            self.state = StreamState.FAULTED
            _safe_close_stream(self._stream)
            self._stream = None
            self._resampler = self._create_resampler()
            self.state = StreamState.RECONNECTING
            try:
                self._stream = _reopen_stream_with_backoff(
                    self._open_stream,
                    logger_ctx=f"WASAPIInputStream:{self._endpoint.name}",
                    should_abort=self._stop_event.is_set,
                )
            except Exception:
                self.state = StreamState.FAULTED
                raise
            self.state = StreamState.RUNNING
            return True

    def _monitor_loop(self) -> None:
        with mmcss_pro_audio():
            while not self._stop_event.wait(0.25):
                try:
                    self.check_and_recover()
                except Exception as exc:
                    logger.error(
                        "[WASAPIInputStream:%s] Recuperación fallida tras reintentos: %s",
                        self._endpoint.name,
                        exc,
                    )
                    break

    def start(self) -> None:
        self._stop_event.clear()
        with self._lock:
            self._stream = self._open_stream()
            self.state = StreamState.RUNNING
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True, name="pb_wasapi_in_mon"
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            if threading.current_thread() is not self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
        with self._lock:
            _safe_close_stream(self._stream)
            self._stream = None


class WASAPIOutputPlayer:
    """Reproductor WASAPI de baja latencia con bloques interrumpibles y recuperación automática."""

    def __init__(self, pa: pyaudio.PyAudio, endpoint: AudioEndpoint) -> None:
        self._pa = pa
        self._endpoint = endpoint
        self._hw_rate: int = endpoint.default_sample_rate
        self._hw_channels: int = max(1, endpoint.max_output_channels)
        self._lock = threading.Lock()
        self._closed: bool = False
        self.state: StreamState = StreamState.RUNNING

        # Tamaño de bloque corto (~20 ms) y rampa cosenoidal anti-clic (~9 ms)
        self._block_frames: int = max(1, int(round(self._hw_rate * 0.020)))
        self._fade_frames: int = max(2, int(round(self._hw_rate * 0.009)))
        t_fade = np.linspace(0.0, 1.0, self._fade_frames, dtype=np.float32)
        self._fade_out_curve: NDArray[np.float32] = (
            0.5 * (1.0 + np.cos(np.pi * t_fade))
        ).astype(np.float32)

        self._chirp_bytes_by_vol: dict[float, bytes] = {}
        self._chirp_bytes_by_id: dict[tuple[int, float, float], bytes] = {}
        self._init_chirp_buffers()

        self._stream: pyaudio.Stream | None = self._open_stream()

    def _open_stream(self) -> pyaudio.Stream:
        return self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._hw_channels,
            rate=self._hw_rate,
            output=True,
            output_device_index=self._endpoint.index,
            frames_per_buffer=max(256, int(self._hw_rate * 0.02)),
        )

    def _render_hw_matrix(
        self,
        samples_mono: NDArray[np.float32],
        source_rate: int,
        pan_left: float,
        pan_right: float,
    ) -> NDArray[np.float32]:
        if source_rate != self._hw_rate:
            resampled: NDArray[np.float32] = soxr.resample(
                samples_mono.astype(np.float32),
                source_rate,
                self._hw_rate,
                quality="HQ",
            )
        else:
            resampled = samples_mono.astype(np.float32)

        resampled = np.clip(resampled, -1.0, 1.0)

        if self._hw_channels == 1:
            return resampled.reshape(-1, 1)

        if self._hw_channels == 2:
            return np.column_stack(
                [resampled * np.float32(pan_left), resampled * np.float32(pan_right)]
            ).astype(np.float32)

        tiled = np.zeros((resampled.shape[0], self._hw_channels), dtype=np.float32)
        tiled[:, 0] = resampled * np.float32(pan_left)
        tiled[:, 1] = resampled * np.float32(pan_right)
        return tiled

    def _init_chirp_buffers(self) -> None:
        """Pre-renderiza en bytes finales todos los chirps al rate y canales nativos del dispositivo."""
        for vol, chirp_pcm in PRECALCULATED_CHIRPS.items():
            mat_default = self._render_hw_matrix(
                chirp_pcm,
                source_rate=TARGET_INTERNAL_RATE,
                pan_left=_CHIRP_DEFAULT_PAN_LEFT,
                pan_right=_CHIRP_DEFAULT_PAN_RIGHT,
            )
            bytes_default = mat_default.tobytes()
            self._chirp_bytes_by_vol[vol] = bytes_default
            self._chirp_bytes_by_id[(id(chirp_pcm), _CHIRP_DEFAULT_PAN_LEFT, _CHIRP_DEFAULT_PAN_RIGHT)] = (
                bytes_default
            )

            mat_unity = self._render_hw_matrix(
                chirp_pcm,
                source_rate=TARGET_INTERNAL_RATE,
                pan_left=1.0,
                pan_right=1.0,
            )
            self._chirp_bytes_by_id[(id(chirp_pcm), 1.0, 1.0)] = mat_unity.tobytes()

    def _write_bytes_with_recovery(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                return
            if self._stream is None:
                self.state = StreamState.RECONNECTING
                try:
                    self._stream = _reopen_stream_with_backoff(
                        self._open_stream,
                        logger_ctx=f"WASAPIOutputPlayer:{self._endpoint.name}",
                        should_abort=lambda: self._closed,
                    )
                except Exception:
                    self.state = StreamState.FAULTED
                    raise
                self.state = StreamState.RUNNING
            try:
                self._stream.write(data)
            except Exception as exc:
                if not _is_wasapi_disconnection_error(exc) or self._closed:
                    raise
                self.state = StreamState.FAULTED
                _safe_close_stream(self._stream)
                self._stream = None
                self.state = StreamState.RECONNECTING
                try:
                    self._stream = _reopen_stream_with_backoff(
                        self._open_stream,
                        logger_ctx=f"WASAPIOutputPlayer:{self._endpoint.name}",
                        should_abort=lambda: self._closed,
                    )
                except Exception:
                    self.state = StreamState.FAULTED
                    raise
                self.state = StreamState.RUNNING
                self._stream.write(data)

    def play_precomputed_chirp(
        self,
        volume: float = 0.5,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        """Emite un chirp pre-renderizado en bytes con cero asignaciones en el hot loop."""
        if self._closed or (cancel_check is not None and cancel_check()):
            return False
        key = _nearest_chirp_key(volume)
        pre_bytes = self._chirp_bytes_by_vol[key]
        self._write_bytes_with_recovery(pre_bytes)
        return True

    def _apply_fade_out(self, segment: NDArray[np.float32]) -> NDArray[np.float32]:
        rem = segment.shape[0]
        if rem == 0:
            return segment
        if rem == self._fade_frames:
            curve = self._fade_out_curve[:, np.newaxis]
        elif rem == 1:
            curve = np.zeros((1, 1), dtype=np.float32)
        else:
            t_short = np.linspace(0.0, 1.0, rem, dtype=np.float32)
            curve = (0.5 * (1.0 + np.cos(np.pi * t_short))).astype(np.float32)[:, np.newaxis]
        return (segment * curve).astype(np.float32)

    def play_pcm(
        self,
        samples_mono: NDArray[np.float32],
        source_rate: int,
        pan_left: float = 1.0,
        pan_right: float = 1.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> bool:
        if samples_mono.shape[0] == 0:
            return True
        if self._closed or (cancel_check is not None and cancel_check()):
            return False

        # Ruta cero-malloc si coincide con chirp pre-renderizado
        if source_rate == TARGET_INTERNAL_RATE:
            pre_bytes = self._chirp_bytes_by_id.get((id(samples_mono), pan_left, pan_right))
            if pre_bytes is not None:
                self._write_bytes_with_recovery(pre_bytes)
                return True

        out_matrix = self._render_hw_matrix(samples_mono, source_rate, pan_left, pan_right)
        total_frames = out_matrix.shape[0]

        for pos in range(0, total_frames, self._block_frames):
            if self._closed or (cancel_check is not None and cancel_check()):
                if pos > 0:
                    fade_end = min(total_frames, pos + self._fade_frames)
                    faded = self._apply_fade_out(out_matrix[pos:fade_end])
                    if faded.shape[0] > 0:
                        self._write_bytes_with_recovery(faded.tobytes())
                return False

            block_end = min(total_frames, pos + self._block_frames)
            self._write_bytes_with_recovery(out_matrix[pos:block_end].tobytes())

        return True

    def close(self) -> None:
        self._closed = True
        with self._lock:
            _safe_close_stream(self._stream)
            self._stream = None


class AsyncOutputWorker:
    """Hilo dedicado de reproducción WASAPI no bloqueante con MMCSS, barge-in y chirp zero-malloc."""

    def __init__(
        self,
        pa: pyaudio.PyAudio | None = None,
        endpoint: AudioEndpoint | None = None,
        player: WASAPIOutputPlayer | None = None,
        media_buffer: MediaLookaheadBuffer | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._generation: int = 0
        self.media_buffer: MediaLookaheadBuffer | None = media_buffer
        if player is not None:
            self._player: WASAPIOutputPlayer | None = player
        elif pa is not None and endpoint is not None:
            self._player = WASAPIOutputPlayer(pa=pa, endpoint=endpoint)
        else:
            self._player = None

        self._queue: queue.Queue[PlaybackItem | None] = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pb_async_out")
        self._thread.start()

    @property
    def pending_items(self) -> int:
        return self._queue.qsize()

    def enqueue(
        self,
        pcm: NDArray[np.float32],
        sample_rate: int,
        pan_left: float = 1.0,
        pan_right: float = 1.0,
        chirp_volume: float | None = None,
    ) -> None:
        if self._stop_event.is_set():
            return
        with self._lock:
            item = PlaybackItem(
                pcm=pcm,
                sample_rate=sample_rate,
                pan_left=pan_left,
                pan_right=pan_right,
                generation=self._generation,
                chirp_volume=chirp_volume,
            )
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    discarded = self._queue.get_nowait()
                    if discarded is not None:
                        self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    pass

    def play_chirp(self, volume: float = 0.5) -> None:
        """Encola el chirp acústico precalculado sin asignaciones dinámicas de NumPy."""
        key = _nearest_chirp_key(volume)
        chirp_pcm = PRECALCULATED_CHIRPS[key]
        self.enqueue(
            chirp_pcm,
            sample_rate=TARGET_INTERNAL_RATE,
            pan_left=_CHIRP_DEFAULT_PAN_LEFT,
            pan_right=_CHIRP_DEFAULT_PAN_RIGHT,
            chirp_volume=key,
        )

    def flush(self) -> int:
        """Vacía atómicamente la cola FIFO e invalida la generación en curso para Barge-In.

        Retorna el número de elementos descartados.
        """
        discarded_count = 0
        with self._lock:
            self._generation += 1
            while True:
                try:
                    item = self._queue.get_nowait()
                    if item is not None:
                        discarded_count += 1
                    self._queue.task_done()
                except queue.Empty:
                    break
        return discarded_count

    def _mix_ducked_background(
        self,
        pcm: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        if self.media_buffer is None or not self.media_buffer.enabled or pcm.shape[0] <= 0:
            return pcm
        target_len = int(pcm.shape[0])
        n_16k = max(1, int(round(target_len * TARGET_INTERNAL_RATE / max(1, sample_rate))))
        bg_16k = self.media_buffer.read_mixed_16k(n_16k, ducking=True)
        if not np.any(bg_16k):
            return pcm
        if sample_rate != TARGET_INTERNAL_RATE:
            bg_matched = soxr.resample(bg_16k, TARGET_INTERNAL_RATE, sample_rate, quality="HQ").astype(
                np.float32
            )
        else:
            bg_matched = bg_16k
        if bg_matched.shape[0] < target_len:
            padded = np.zeros((target_len,), dtype=np.float32)
            padded[: bg_matched.shape[0]] = bg_matched
            bg_matched = padded
        elif bg_matched.shape[0] > target_len:
            bg_matched = bg_matched[:target_len]
        return np.clip(pcm + bg_matched, -1.0, 1.0).astype(np.float32)

    def _run(self) -> None:
        with mmcss_pro_audio():
            while not self._stop_event.is_set():
                media_active = self.media_buffer is not None and self.media_buffer.enabled
                timeout = 0.020 if media_active else 0.1
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    if (
                        media_active
                        and self._player is not None
                        and self.media_buffer is not None
                        and self.media_buffer.available_to_read >= VAD_FRAME_SAMPLES
                    ):
                        bg_chunk = self.media_buffer.read_mixed_16k(VAD_FRAME_SAMPLES, ducking=False)
                        if np.any(bg_chunk):
                            try:
                                self._player.play_pcm(
                                    bg_chunk,
                                    source_rate=TARGET_INTERNAL_RATE,
                                    pan_left=0.85,
                                    pan_right=0.85,
                                    cancel_check=lambda: (
                                        not self._queue.empty() or self._stop_event.is_set()
                                    ),
                                )
                            except Exception as exc:
                                logger.error("Error en passthrough de Modo Media: %s", exc)
                    continue

                if item is None:
                    self._queue.task_done()
                    break

                try:
                    if self._player is not None:
                        item_gen = item.generation

                        def _is_cancelled(gen: int = item_gen) -> bool:
                            return self._generation != gen or self._stop_event.is_set()

                        if item.chirp_volume is not None and hasattr(
                            self._player, "play_precomputed_chirp"
                        ):
                            self._player.play_precomputed_chirp(
                                item.chirp_volume, cancel_check=_is_cancelled
                            )
                        else:
                            pcm_to_play = self._mix_ducked_background(item.pcm, item.sample_rate)
                            self._player.play_pcm(
                                pcm_to_play,
                                source_rate=item.sample_rate,
                                pan_left=item.pan_left,
                                pan_right=item.pan_right,
                                cancel_check=_is_cancelled,
                            )
                except Exception as exc:
                    logger.error("Error en reproducción AsyncOutputWorker: %s", exc)
                finally:
                    self._queue.task_done()

    def close(self) -> None:
        self._stop_event.set()
        self.flush()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)
        if self._player is not None:
            self._player.close()
            self._player = None
