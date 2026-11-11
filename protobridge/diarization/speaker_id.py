from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

WESPEAKER_ONNX_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "wespeaker_en_voxceleb_CAM++.onnx"
)


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: int
    label: str
    voice_name: str
    pan_left: float
    pan_right: float
    pos_label: str
    pitch_hz: float
    similarity: float


# Pool de 4 identidades vocales en español con posicionamiento espacial 3D en auriculares
SPANISH_VOICE_POOL: tuple[tuple[str, str, float, float, str], ...] = (
    ("Companero #1", "em_alex", 0.90, 0.25, "IZQUIERDA"),
    ("Companero #2", "em_santa", 0.70, 0.70, "CENTRO"),
    ("Companero #3", "ef_dora", 0.25, 0.90, "DERECHA"),
    ("Companero #4", "em_tactical", 0.45, 0.85, "CENTRO-DER"),
)


def _ensure_speaker_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "wespeaker_campp.onnx"
    if not onnx_path.exists():
        print(" [Descargando] Modelo de Identificación de Hablantes WeSpeaker CAM++ (~28 MB)...")
        urllib.request.urlretrieve(WESPEAKER_ONNX_URL, onnx_path)
    return onnx_path


def _create_mel_filterbank(
    sample_rate: int = 16000, n_fft: int = 512, n_mels: int = 80
) -> NDArray[np.float32]:
    f_min = 20.0
    f_max = float(sample_rate) / 2.0
    mel_min = 1127.0 * np.log(1.0 + f_min / 700.0)
    mel_max = 1127.0 * np.log(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (np.exp(mel_points / 1127.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / max(1, (f_m_plus - f_m))
    return fbank


class OnlineSpeakerTracker:
    """
    Identificador de hablantes en tiempo real (~12 ms en CPU).
    Extrae embeddings vocales WeSpeaker CAM++ ONNX + frecuencia fundamental (F0)
    y agrupa hasta 4 compañeros online mediante similitud coseno.
    """

    def __init__(
        self,
        models_dir: Path,
        max_speakers: int = 4,
        similarity_threshold: float = 0.56,
    ) -> None:
        model_path = _ensure_speaker_model(models_dir)
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._mel_basis: NDArray[np.float32] = _create_mel_filterbank(16000, 512, 80)
        self._window: NDArray[np.float32] = np.hamming(400).astype(np.float32)

        self.max_speakers: int = min(max_speakers, len(SPANISH_VOICE_POOL))
        self.similarity_threshold: float = similarity_threshold

        self._centroids: list[NDArray[np.float32]] = []
        self._assigned_slots: list[int] = []
        self._pitches: list[float] = []
        self._last_speaker_idx: int = 0

    def _compute_fbank_80(self, audio_16k: NDArray[np.float32]) -> NDArray[np.float32]:
        # Pre-emphasis
        emphasized = np.append(audio_16k[0], audio_16k[1:] - 0.97 * audio_16k[:-1]).astype(
            np.float32
        )
        frame_len = 400  # 25 ms
        frame_step = 160  # 10 ms
        if emphasized.shape[0] < frame_len:
            emphasized = np.pad(emphasized, (0, frame_len - emphasized.shape[0]))

        num_frames = 1 + (emphasized.shape[0] - frame_len) // frame_step
        strides = (emphasized.strides[0] * frame_step, emphasized.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            emphasized, shape=(num_frames, frame_len), strides=strides
        ).copy()
        frames *= self._window

        mag_frames = np.abs(np.fft.rfft(frames, n=512, axis=1))
        pow_frames = (1.0 / 512.0) * np.square(mag_frames)
        mel_energy = np.dot(pow_frames, self._mel_basis.T)
        log_mel = np.log(np.maximum(mel_energy, 1e-10)).astype(np.float32)
        # CMN (Cepstral Mean Normalization)
        log_mel -= np.mean(log_mel, axis=0, keepdims=True)
        return log_mel

    @staticmethod
    def _estimate_pitch_hz(audio_16k: NDArray[np.float32]) -> float:
        """Estimación rápida de F0 por autocorrelación en tramas sonoras (< 1.5 ms)."""
        frame_size = 1024
        step = 512
        if audio_16k.shape[0] < frame_size:
            return 135.0

        min_lag = int(16000 / 320)  # 320 Hz max
        max_lag = int(16000 / 75)   # 75 Hz min
        pitches: list[float] = []

        for start in range(0, audio_16k.shape[0] - frame_size, step):
            chunk = audio_16k[start : start + frame_size]
            if float(np.sqrt(np.mean(np.square(chunk)))) < 0.02:
                continue
            chunk = chunk - np.mean(chunk)
            corr = np.correlate(chunk, chunk, mode="full")[frame_size - 1 :]
            if corr[0] <= 1e-9:
                continue
            corr_norm = corr / corr[0]
            search = corr_norm[min_lag:max_lag]
            if search.shape[0] > 0 and float(np.max(search)) > 0.32:
                best_lag = min_lag + int(np.argmax(search))
                pitches.append(16000.0 / float(best_lag))

        return float(np.median(pitches)) if pitches else 135.0

    def _extract_embedding(self, audio_16k: NDArray[np.float32]) -> NDArray[np.float32]:
        fbank = self._compute_fbank_80(audio_16k)
        batch_input = np.expand_dims(fbank, axis=0).astype(np.float32)
        outputs = self._session.run(None, {self._input_name: batch_input})
        emb = np.squeeze(outputs[0]).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-9)

    def _pick_voice_slot(self, pitch_hz: float) -> int:
        used = set(self._assigned_slots)
        # Si el tono es agudo/femenino (>= 175 Hz), priorizar slot 2 (ef_dora)
        if pitch_hz >= 175.0 and 2 not in used:
            return 2
        for candidate in (0, 1, 3, 2):
            if candidate not in used:
                return candidate
        return len(self._assigned_slots) % len(SPANISH_VOICE_POOL)

    def identify_speaker(self, audio_16k: NDArray[np.float32]) -> SpeakerProfile:
        pitch_hz = self._estimate_pitch_hz(audio_16k)
        emb = self._extract_embedding(audio_16k)
        duration_s = audio_16k.shape[0] / 16000.0

        # Si el clip es muy corto (< 0.55s) y ya hay un hablante reciente, suavizar umbral
        effective_thresh = (
            self.similarity_threshold - 0.08 if duration_s < 0.65 else self.similarity_threshold
        )

        if not self._centroids:
            slot = self._pick_voice_slot(pitch_hz)
            self._centroids.append(emb)
            self._assigned_slots.append(slot)
            self._pitches.append(pitch_hz)
            self._last_speaker_idx = 0
            label, voice, pan_l, pan_r, pos_lbl = SPANISH_VOICE_POOL[slot]
            return SpeakerProfile(0, label, voice, pan_l, pan_r, pos_lbl, pitch_hz, 1.0)

        sims = [float(np.dot(emb, c)) for c in self._centroids]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= effective_thresh or len(self._centroids) >= self.max_speakers:
            if duration_s >= 0.70:
                updated = 0.82 * self._centroids[best_idx] + 0.18 * emb
                self._centroids[best_idx] = updated / max(float(np.linalg.norm(updated)), 1e-9)
                self._pitches[best_idx] = 0.8 * self._pitches[best_idx] + 0.2 * pitch_hz

            self._last_speaker_idx = best_idx
            slot = self._assigned_slots[best_idx]
            label, voice, pan_l, pan_r, pos_lbl = SPANISH_VOICE_POOL[slot]
            return SpeakerProfile(
                best_idx, label, voice, pan_l, pan_r, pos_lbl, self._pitches[best_idx], best_sim
            )

        # Registrar nuevo compañero detectado
        new_idx = len(self._centroids)
        slot = self._pick_voice_slot(pitch_hz)
        self._centroids.append(emb)
        self._assigned_slots.append(slot)
        self._pitches.append(pitch_hz)
        self._last_speaker_idx = new_idx
        label, voice, pan_l, pan_r, pos_lbl = SPANISH_VOICE_POOL[slot]
        return SpeakerProfile(new_idx, label, voice, pan_l, pan_r, pos_lbl, pitch_hz, best_sim)
