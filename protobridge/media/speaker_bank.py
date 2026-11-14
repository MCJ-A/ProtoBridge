from __future__ import annotations

from dataclasses import dataclass, field
import threading
import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE_16K = 16000
MAX_REF_SAMPLES = SAMPLE_RATE_16K * 10  # 10.0 s máximo para latentes XTTS promediados
V1_CLONE_MIN_SAMPLES = int(SAMPLE_RATE_16K * 2.2)  # primer clonado rápido tras 2.2 s limpios
V2_CLONE_MIN_SAMPLES = int(SAMPLE_RATE_16K * 5.2)  # refinamiento único tras 5.2 s limpios

# Voces precalculadas de speakers_xtts.pth verificadas localmente como respaldo antes de clonar
FEMALE_PRESET_POOL: tuple[str, ...] = (
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
)
MALE_PRESET_POOL: tuple[str, ...] = (
    "Andrew Chipper",
    "Craig Gutsy",
    "Damien Black",
    "Aaron Dreschner",
)


@dataclass
class SpeakerClipState:
    speaker_id: int
    preset_voice: str
    chunks: list[NDArray[np.float32]] = field(default_factory=list)
    total_samples: int = 0
    last_submitted_version: int = 0
    clone_ready: bool = False


class MediaSpeakerVoiceBank:
    """Acumula audio limpio por hablante para clonación zero-shot y latentes en XTTS-v2."""

    def __init__(self, session_prefix: str = "media") -> None:
        self._lock = threading.Lock()
        self._session_prefix = session_prefix
        self._speakers: dict[int, SpeakerClipState] = {}

    def speaker_key(self, speaker_id: int) -> str:
        return f"{self._session_prefix}:spk_{speaker_id}"

    def reset(self) -> None:
        with self._lock:
            self._speakers.clear()

    def _pick_preset(self, speaker_id: int, pitch_hz: float) -> str:
        if pitch_hz >= 178.0:
            return FEMALE_PRESET_POOL[speaker_id % len(FEMALE_PRESET_POOL)]
        return MALE_PRESET_POOL[speaker_id % len(MALE_PRESET_POOL)]

    def ingest_segment(
        self,
        speaker_id: int,
        audio_16k: NDArray[np.float32],
        pitch_hz: float,
        *,
        confidence_ok: bool = True,
    ) -> tuple[str, str, int, NDArray[np.float32] | None]:
        """
        Registra un segmento hablado limpio para speaker_id.

        Devuelve `(speaker_key, preset_fallback, clone_version_to_register, ref_pcm_16k_or_None)`.
        """
        pcm = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12)) if pcm.size > 0 else 0.0
        peak = float(np.max(np.abs(pcm))) if pcm.size > 0 else 0.0
        active_ratio = (
            float(np.mean(np.abs(pcm) > max(0.003, rms * 0.25))) if pcm.size > 0 else 0.0
        )

        with self._lock:
            state = self._speakers.get(speaker_id)
            if state is None:
                state = SpeakerClipState(
                    speaker_id=speaker_id,
                    preset_voice=self._pick_preset(speaker_id, pitch_hz),
                )
                self._speakers[speaker_id] = state
            elif confidence_ok and not state.clone_ready and state.last_submitted_version < 1:
                state.preset_voice = self._pick_preset(speaker_id, pitch_hz)

            # Solo acumula audio si el segmento pertenece con alta confianza al hablante y no excede v2
            if (
                confidence_ok
                and state.last_submitted_version < 2
                and pcm.size >= 3200
                and 0.006 <= rms <= 0.45
                and peak <= 0.99
                and active_ratio >= 0.18
            ):
                state.chunks.append(pcm.copy())
                state.total_samples += int(pcm.shape[0])
                while state.total_samples > MAX_REF_SAMPLES and len(state.chunks) > 1:
                    dropped = state.chunks.pop(0)
                    state.total_samples -= int(dropped.shape[0])

            target_version = 0
            if state.last_submitted_version < 1 and state.total_samples >= V1_CLONE_MIN_SAMPLES:
                target_version = 1
            elif state.last_submitted_version < 2 and state.total_samples >= V2_CLONE_MIN_SAMPLES:
                target_version = 2

            ref_pcm: NDArray[np.float32] | None = None
            if target_version > 0:
                state.last_submitted_version = target_version
                ref_pcm = np.concatenate(state.chunks).astype(np.float32)

            return (
                self.speaker_key(speaker_id),
                state.preset_voice,
                target_version,
                ref_pcm,
            )

    def mark_clone_ready(self, speaker_id: int) -> None:
        with self._lock:
            state = self._speakers.get(speaker_id)
            if state is not None:
                state.clone_ready = True

    def is_clone_ready(self, speaker_id: int) -> bool:
        with self._lock:
            state = self._speakers.get(speaker_id)
            return bool(state.clone_ready) if state is not None else False
