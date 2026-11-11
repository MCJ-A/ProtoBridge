from __future__ import annotations

import threading
import numpy as np
from numpy.typing import NDArray
import pyaudiowpatch as pyaudio
import soxr

from protobridge.audio.devices import AudioEndpoint
from protobridge.audio.ring_buffer import SPSCRingBuffer

TARGET_INTERNAL_RATE: int = 16000
VAD_FRAME_SAMPLES: int = 512


class WASAPIInputStream:
    def __init__(
        self,
        pa: pyaudio.PyAudio,
        endpoint: AudioEndpoint,
        ring_buffer: SPSCRingBuffer,
    ) -> None:
        self._pa = pa
        self._endpoint = endpoint
        self._ring_buffer = ring_buffer

        self._hw_rate: int = endpoint.default_sample_rate
        self._hw_channels: int = max(
            1,
            endpoint.max_input_channels
            if endpoint.max_input_channels > 0
            else endpoint.max_output_channels,
        )
        self._frames_per_buffer: int = max(128, int(self._hw_rate * 0.016))

        self._resampler: soxr.ResampleStream | None = None
        if self._hw_rate != TARGET_INTERNAL_RATE:
            self._resampler = soxr.ResampleStream(
                self._hw_rate,
                TARGET_INTERNAL_RATE,
                num_channels=1,
                dtype="float32",
                quality="HQ",
            )

        self.last_rms_dbfs: float = -100.0
        self._stream: pyaudio.Stream | None = None

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status_flags: int,
    ) -> tuple[None, int]:
        if in_data is None:
            return (None, pyaudio.paContinue)

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

        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._hw_channels,
            rate=self._hw_rate,
            input=True,
            input_device_index=self._endpoint.index,
            frames_per_buffer=self._frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None


class WASAPIOutputPlayer:
    """
    Reproductor WASAPI de baja latencia que remuestrea audio mono/estéreo al sample rate
    nativo del dispositivo destino (ej. 48000 Hz en VB-Cable o 96000 Hz en ROG Theta).
    """

    def __init__(self, pa: pyaudio.PyAudio, endpoint: AudioEndpoint) -> None:
        self._pa = pa
        self._endpoint = endpoint
        self._hw_rate: int = endpoint.default_sample_rate
        self._hw_channels: int = max(1, endpoint.max_output_channels)
        self._lock = threading.Lock()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._hw_channels,
            rate=self._hw_rate,
            output=True,
            output_device_index=self._endpoint.index,
            frames_per_buffer=max(256, int(self._hw_rate * 0.02)),
        )

    def play_pcm(
        self,
        samples_mono: NDArray[np.float32],
        source_rate: int,
        pan_left: float = 1.0,
        pan_right: float = 1.0,
    ) -> None:
        if samples_mono.shape[0] == 0:
            return

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
            out_matrix = resampled
        elif self._hw_channels == 2:
            out_matrix = np.column_stack(
                [resampled * np.float32(pan_left), resampled * np.float32(pan_right)]
            ).ravel()
        else:
            tiled = np.zeros((resampled.shape[0], self._hw_channels), dtype=np.float32)
            tiled[:, 0] = resampled * np.float32(pan_left)
            tiled[:, 1] = resampled * np.float32(pan_right)
            out_matrix = tiled.ravel()

        with self._lock:
            self._stream.write(out_matrix.tobytes())

    def close(self) -> None:
        with self._lock:
            self._stream.stop_stream()
            self._stream.close()
