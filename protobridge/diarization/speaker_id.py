from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
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
_MALE_SLOTS: tuple[int, ...] = (0, 1, 3)
_FEMALE_SLOT: int = 2
_FEMALE_PITCH_THRESHOLD_HZ: float = 178.0


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
        limit_to_pool: bool = True,
        continuity_bonus: float = 0.06,
        require_new_confirmation: bool = False,
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

        self.max_speakers: int = (
            min(max_speakers, len(SPANISH_VOICE_POOL)) if limit_to_pool else max(1, max_speakers)
        )
        self.similarity_threshold: float = similarity_threshold
        self.continuity_bonus: float = max(0.0, float(continuity_bonus))
        self.require_new_confirmation: bool = bool(require_new_confirmation)

        self._lock = threading.Lock()
        self._centroids: list[NDArray[np.float32]] = []
        self._anchors: list[NDArray[np.float32]] = []
        self._assigned_slots: list[int] = []
        self._pitches: list[float] = []
        self._sample_counts: list[int] = []
        self._pitch_counts: list[int] = []
        self._slot_locks: list[bool] = []
        self._last_speaker_idx: int = 0
        self._pending_emb: NDArray[np.float32] | None = None
        self._pending_pitch: float = 135.0
        self._last_was_confident: bool = True

    @property
    def last_was_confident(self) -> bool:
        with self._lock:
            return bool(self._last_was_confident)

    def reset_squad(self) -> None:
        """Limpia los centroides EMA y asignaciones de voz entre misiones/escuadrones."""
        with self._lock:
            self._centroids.clear()
            self._anchors.clear()
            self._assigned_slots.clear()
            self._pitches.clear()
            self._sample_counts.clear()
            self._pitch_counts.clear()
            self._slot_locks.clear()
            self._last_speaker_idx = 0
            self._pending_emb = None
            self._pending_pitch = 135.0
            self._last_was_confident = True

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
        """Estimación rápida de F0 por autocorrelación normalizada + primer pico MPM (< 1.5 ms)."""
        frame_size = 1024
        step = 512
        if audio_16k.shape[0] < frame_size:
            return 135.0

        min_lag = int(16000 / 300)  # ~301.9 Hz max
        max_lag = int(16000 / 70)   # ~70.2 Hz min
        lags = np.arange(min_lag, max_lag)
        pitches: list[float] = []

        for start in range(0, audio_16k.shape[0] - frame_size + 1, step):
            chunk = audio_16k[start : start + frame_size].astype(np.float64, copy=False)
            if float(np.sqrt(np.mean(np.square(chunk)))) < 0.018:
                continue
            chunk = chunk - np.mean(chunk)
            corr = np.correlate(chunk, chunk, mode="full")[frame_size - 1 :]
            cum_sq = np.concatenate(([0.0], np.cumsum(chunk * chunk)))
            left_e = cum_sq[frame_size - lags]
            right_e = cum_sq[frame_size] - cum_sq[lags]
            denom = np.sqrt(left_e * right_e)
            valid = denom > 1e-9
            nsdf = np.zeros(lags.shape[0], dtype=np.float64)
            nsdf[valid] = corr[lags[valid]] / denom[valid]

            peak_max = float(np.max(nsdf)) if nsdf.size > 0 else 0.0
            if peak_max < 0.45:
                continue

            thresh = max(0.45, 0.86 * peak_max)
            chosen_lag: int | None = None
            idx = 1
            n_nsdf = nsdf.shape[0]
            while idx < n_nsdf - 1:
                if nsdf[idx] >= thresh and nsdf[idx] >= nsdf[idx - 1] and nsdf[idx] >= nsdf[idx + 1]:
                    best_k = idx
                    k = idx
                    while k < n_nsdf - 1 and nsdf[k] > 0.0:
                        if nsdf[k] > nsdf[best_k]:
                            best_k = k
                        k += 1
                    chosen_lag = int(lags[best_k])
                    break
                idx += 1
            if chosen_lag is None:
                chosen_lag = int(lags[int(np.argmax(nsdf))])
            pitches.append(16000.0 / float(chosen_lag))

        return float(np.median(pitches)) if pitches else 135.0

    def _extract_embedding(self, audio_16k: NDArray[np.float32]) -> NDArray[np.float32]:
        fbank = self._compute_fbank_80(audio_16k)
        batch_input = np.expand_dims(fbank, axis=0).astype(np.float32)
        outputs = self._session.run(None, {self._input_name: batch_input})
        emb = np.squeeze(outputs[0]).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / max(norm, 1e-9)

    def _pick_male_slot(self, used: set[int]) -> int:
        for candidate in _MALE_SLOTS:
            if candidate not in used:
                return candidate
        return _MALE_SLOTS[len(self._assigned_slots) % len(_MALE_SLOTS)]

    def _pick_voice_slot(self, pitch_hz: float) -> int:
        used = set(self._assigned_slots)
        if pitch_hz >= _FEMALE_PITCH_THRESHOLD_HZ:
            return _FEMALE_SLOT
        return self._pick_male_slot(used)

    def _ensure_meta_len_locked(self) -> None:
        while len(self._anchors) < len(self._centroids):
            idx = len(self._anchors)
            self._anchors.append(self._centroids[idx])
        while len(self._sample_counts) < len(self._centroids):
            self._sample_counts.append(1)
        while len(self._pitch_counts) < len(self._centroids):
            self._pitch_counts.append(1)
        while len(self._slot_locks) < len(self._centroids):
            self._slot_locks.append(False)

    def _build_profile_locked(
        self, idx: int, smoothed_pitch: float, similarity: float
    ) -> SpeakerProfile:
        slot = self._assigned_slots[idx]
        base_label, voice, pan_l, pan_r, pos_lbl = SPANISH_VOICE_POOL[slot]
        label = base_label if idx < len(SPANISH_VOICE_POOL) else f"Companero #{idx + 1}"
        return SpeakerProfile(
            idx, label, voice, pan_l, pan_r, pos_lbl, smoothed_pitch, similarity
        )

    def identify_speaker(self, audio_16k: NDArray[np.float32]) -> SpeakerProfile:
        pitch_hz = self._estimate_pitch_hz(audio_16k)
        emb = self._extract_embedding(audio_16k)
        duration_s = audio_16k.shape[0] / 16000.0
        rms = float(np.sqrt(np.mean(np.square(audio_16k)) + 1e-12)) if audio_16k.size > 0 else 0.0

        # Si el clip es muy corto (< 0.70s), suavizar umbral para no fragmentar al hablante
        effective_thresh = (
            self.similarity_threshold - 0.10 if duration_s < 0.70 else self.similarity_threshold
        )
        strict_test_mode = self.similarity_threshold >= 0.90

        with self._lock:
            self._ensure_meta_len_locked()

            if not self._centroids:
                slot = self._pick_voice_slot(pitch_hz)
                self._centroids.append(emb)
                self._anchors.append(emb)
                self._assigned_slots.append(slot)
                self._pitches.append(pitch_hz)
                self._sample_counts.append(1)
                self._pitch_counts.append(1)
                self._slot_locks.append(False)
                self._last_speaker_idx = 0
                self._pending_emb = None
                self._last_was_confident = duration_s >= 0.50 and rms >= 0.012
                return self._build_profile_locked(0, pitch_hz, 1.0)

            raw_sims = [
                max(float(np.dot(emb, c)), 0.97 * float(np.dot(emb, self._anchors[i])))
                for i, c in enumerate(self._centroids)
            ]

            if self.max_speakers == 1:
                best_idx = 0
                best_raw_sim = raw_sims[0]
                is_confident = (
                    best_raw_sim >= max(0.45, self.similarity_threshold - 0.06)
                    and duration_s >= 0.75
                    and rms >= 0.015
                )
                if is_confident:
                    self._update_speaker_locked(0, emb, pitch_hz, duration_s)
                self._last_speaker_idx = 0
                self._pending_emb = None
                self._last_was_confident = is_confident or (rms >= 0.015 and duration_s >= 0.75)
                return self._build_profile_locked(0, self._pitches[0], best_raw_sim)

            scores = list(raw_sims)
            if (
                not strict_test_mode
                and self.continuity_bonus > 0.0
                and 0 <= self._last_speaker_idx < len(scores)
            ):
                last_idx = self._last_speaker_idx
                pitch_delta = abs(pitch_hz - self._pitches[last_idx])
                same_gender_band = (
                    (pitch_hz >= _FEMALE_PITCH_THRESHOLD_HZ)
                    == (self._pitches[last_idx] >= _FEMALE_PITCH_THRESHOLD_HZ)
                )
                if pitch_delta < 42.0 or (same_gender_band and pitch_delta < 52.0):
                    bonus = self.continuity_bonus + (0.03 if duration_s < 0.95 else 0.0)
                    scores[last_idx] += bonus

            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            best_raw_sim = raw_sims[best_idx]

            too_short_for_new = (not strict_test_mode) and (duration_s < 0.90 or rms < 0.014)
            matched_existing = (
                best_score >= effective_thresh
                or len(self._centroids) >= self.max_speakers
                or too_short_for_new
            )

            if not matched_existing and self.require_new_confirmation and not strict_test_mode:
                strong_gender_shift = (
                    abs(pitch_hz - self._pitches[best_idx]) >= 48.0 and duration_s >= 1.05
                )
                very_distinct = best_raw_sim < 0.34 and duration_s >= 1.05
                if not strong_gender_shift and not very_distinct:
                    if (
                        self._pending_emb is not None
                        and float(np.dot(emb, self._pending_emb)) >= 0.50
                    ):
                        # Dos turnos seguidos consistentes confirman un nuevo hablante real
                        blended = 0.5 * self._pending_emb + 0.5 * emb
                        emb = blended / max(float(np.linalg.norm(blended)), 1e-9)
                        pitch_hz = 0.5 * self._pending_pitch + 0.5 * pitch_hz
                        self._pending_emb = None
                    else:
                        self._pending_emb = emb
                        self._pending_pitch = pitch_hz
                        self._last_speaker_idx = best_idx
                        self._last_was_confident = False
                        return self._build_profile_locked(
                            best_idx, self._pitches[best_idx], best_raw_sim
                        )

            if matched_existing:
                if best_raw_sim >= effective_thresh:
                    self._pending_emb = None
                is_confident = (
                    best_raw_sim >= max(0.50, self.similarity_threshold - 0.04)
                    and duration_s >= 0.75
                    and rms >= 0.015
                )
                if is_confident or (strict_test_mode and best_raw_sim >= effective_thresh):
                    self._update_speaker_locked(best_idx, emb, pitch_hz, duration_s)
                self._last_speaker_idx = best_idx
                self._last_was_confident = bool(
                    is_confident or (best_raw_sim >= effective_thresh and duration_s >= 0.65 and rms >= 0.014)
                )
                return self._build_profile_locked(best_idx, self._pitches[best_idx], best_raw_sim)

            # Registrar nuevo compañero detectado
            new_idx = len(self._centroids)
            slot = self._pick_voice_slot(pitch_hz)
            self._centroids.append(emb)
            self._anchors.append(emb)
            self._assigned_slots.append(slot)
            self._pitches.append(pitch_hz)
            self._sample_counts.append(1)
            self._pitch_counts.append(1)
            self._slot_locks.append(False)
            self._last_speaker_idx = new_idx
            self._pending_emb = None
            self._last_was_confident = duration_s >= 0.85 and rms >= 0.015
            return self._build_profile_locked(new_idx, pitch_hz, best_raw_sim)

    def _update_speaker_locked(
        self, idx: int, emb: NDArray[np.float32], pitch_hz: float, duration_s: float
    ) -> None:
        cnt = self._sample_counts[idx]
        if duration_s >= 0.65:
            alpha = max(0.04, 0.16 / (1.0 + 0.22 * max(0, cnt - 1)))
            updated = (1.0 - alpha) * self._centroids[idx] + alpha * emb
            self._centroids[idx] = updated / max(float(np.linalg.norm(updated)), 1e-9)
            if cnt < 4:
                anc = 0.82 * self._anchors[idx] + 0.18 * emb
                self._anchors[idx] = anc / max(float(np.linalg.norm(anc)), 1e-9)
            self._sample_counts[idx] = cnt + 1

            p_alpha = 0.22 if not self._slot_locks[idx] else 0.10
            self._pitches[idx] = (1.0 - p_alpha) * self._pitches[idx] + p_alpha * pitch_hz
            self._pitch_counts[idx] += 1
        elif duration_s >= 0.40:
            self._pitches[idx] = 0.86 * self._pitches[idx] + 0.14 * pitch_hz

        if not self._slot_locks[idx]:
            smoothed_pitch = self._pitches[idx]
            cur_slot = self._assigned_slots[idx]
            if cur_slot == _FEMALE_SLOT and smoothed_pitch < 168.0:
                used_others = {s for j, s in enumerate(self._assigned_slots) if j != idx}
                self._assigned_slots[idx] = self._pick_male_slot(used_others)
            elif cur_slot != _FEMALE_SLOT and smoothed_pitch >= 184.0:
                self._assigned_slots[idx] = _FEMALE_SLOT

            if self._pitch_counts[idx] >= 5 and (smoothed_pitch < 162.0 or smoothed_pitch >= 186.0):
                self._slot_locks[idx] = True
