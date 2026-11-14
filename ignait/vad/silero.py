from __future__ import annotations

from pathlib import Path
import time
import urllib.request
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

from ignait.audio.ring_buffer import SPSCRingBuffer

SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)
FRAME_SAMPLES = 512  # 32 ms @ 16 kHz
CONTEXT_SAMPLES = 64
PREROLL_SAMPLES = 4800  # 300 ms @ 16 kHz
_CROSS_VAD_GAIN_15DB: float = 10.0 ** (15.0 / 20.0)


def ensure_silero_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "silero_vad.onnx"
    if not onnx_path.exists():
        print(" [Descargando] Modelo Silero VAD v5 ONNX (~1.8 MB)...")
        urllib.request.urlretrieve(SILERO_ONNX_URL, onnx_path)
    return onnx_path


class SileroVADSegmenter:
    """
    Segmentador de voz en tiempo real sobre ONNX Runtime (CPU).
    Extrae segmentos completos de habla incluyendo 300 ms de Pre-Roll.
    """

    def __init__(
        self,
        models_dir: Path,
        speech_threshold: float = 0.52,
        min_silence_ms: int = 240,
        min_speech_ms: int = 280,
        max_speech_seconds: float = 4.5,
        min_rms_dbfs: float = -48.0,
    ) -> None:
        model_path = ensure_silero_model(models_dir)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._use_v5_state = "state" in self._input_names

        self.speech_threshold = speech_threshold
        self.base_speech_threshold = speech_threshold
        self.silence_threshold = max(0.25, speech_threshold - 0.15)
        self.min_silence_frames = max(1, int(min_silence_ms / 32))
        self.min_speech_samples = int((min_speech_ms / 1000.0) * 16000)
        self.max_speech_samples = int(max_speech_seconds * 16000)
        self.min_rms_linear = float(10.0 ** (min_rms_dbfs / 20.0))
        self.base_min_rms_linear = self.min_rms_linear
        self._gate_until_ts: float = 0.0

        self.reset_state()

    def notify_output_audio(self, hold_ms: float = 150.0) -> None:
        """
        Activa o extiende el Cross-VAD Gating (+15 dB / umbral elevado) durante la emisión
        de audio local y un margen posterior (por defecto 150 ms) para evitar auto-traducción.
        """
        target = time.perf_counter() + max(0.0, hold_ms) / 1000.0
        if target > self._gate_until_ts:
            self._gate_until_ts = target

    @property
    def is_cross_vad_gated(self) -> bool:
        return time.perf_counter() < self._gate_until_ts

    def reset_state(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

        self.is_speaking: bool = False
        self._speech_start_cursor: int = 0
        self._silence_frames_count: int = 0
        self.last_prob: float = 0.0

    def _infer_frame(self, frame_512: NDArray[np.float32]) -> float:
        row = frame_512.reshape(1, FRAME_SAMPLES).astype(np.float32)
        if self._use_v5_state:
            x = np.concatenate([self._context, row], axis=1)
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            self._context = x[:, -CONTEXT_SAMPLES:]
            return float(np.squeeze(out))
        else:
            out, self._h, self._c = self._session.run(
                None, {"input": row, "sr": self._sr, "h": self._h, "c": self._c}
            )
            return float(np.squeeze(out))

    def process_available(self, ring_buffer: SPSCRingBuffer) -> list[NDArray[np.float32]]:
        """
        Drena todas las tramas de 32 ms disponibles en el RingBuffer y devuelve una lista
        de segmentos de voz completos listos para enviar a Whisper.
        """
        completed_segments: list[NDArray[np.float32]] = []
        frame_buf = np.empty((FRAME_SAMPLES,), dtype=np.float32)

        while ring_buffer.available_to_read >= FRAME_SAMPLES:
            frame_start_cursor = ring_buffer.read_cursor
            ring_buffer.read_into(frame_buf)
            frame_end_cursor = ring_buffer.read_cursor

            prob = self._infer_frame(frame_buf)
            self.last_prob = prob

            gated = time.perf_counter() < self._gate_until_ts
            eff_speech_thresh = (
                min(0.92, self.base_speech_threshold + 0.25)
                if gated
                else self.base_speech_threshold
            )
            eff_min_rms = (
                self.base_min_rms_linear * _CROSS_VAD_GAIN_15DB
                if gated
                else self.base_min_rms_linear
            )

            if not self.is_speaking:
                if prob >= eff_speech_thresh:
                    self.is_speaking = True
                    self._speech_start_cursor = frame_start_cursor
                    self._silence_frames_count = 0
            else:
                current_duration = frame_end_cursor - self._speech_start_cursor
                if prob < self.silence_threshold:
                    self._silence_frames_count += 1
                else:
                    self._silence_frames_count = 0

                should_cut = (
                    self._silence_frames_count >= self.min_silence_frames
                    or current_duration >= self.max_speech_samples
                )
                if should_cut:
                    if current_duration >= self.min_speech_samples:
                        segment = ring_buffer.slice_window(
                            self._speech_start_cursor,
                            frame_end_cursor,
                            preroll_samples=PREROLL_SAMPLES,
                        )
                        rms = float(np.sqrt(np.mean(np.square(segment)) + 1e-12))
                        if rms >= eff_min_rms:
                            completed_segments.append(segment)

                    self.is_speaking = False
                    self._silence_frames_count = 0

        return completed_segments
