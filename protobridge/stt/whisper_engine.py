from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import numpy as np
from numpy.typing import NDArray


def _register_cuda_dll_directories() -> None:
    if sys.platform != "win32":
        return
    site_packages = [Path(p) for p in sys.path if "site-packages" in p.lower()]
    for sp in site_packages:
        nvidia_dir = sp / "nvidia"
        if nvidia_dir.exists():
            for bin_dir in nvidia_dir.glob("*/bin"):
                if bin_dir.is_dir():
                    try:
                        os.add_dll_directory(str(bin_dir.resolve()))
                        os.environ["PATH"] = str(bin_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
                    except OSError:
                        pass


_register_cuda_dll_directories()

from faster_whisper import WhisperModel  # noqa: E402

GAMING_PROMPT_EN = (
    "Tactical multiplayer gaming voice chat. "
    "Terms: Helldiver, Stratagem, Bile Titan, Charger, Hulks, Stalker, Eagle 500kg, "
    "Orbital Strike, Reinforce, Extraction, Resupply, Heavy armor, Flank, Reloading, Watch out."
)

HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"thank(s|\s+you)\s+for\s+watching", re.IGNORECASE),
    re.compile(r"subtitles\s+by|amara\.org|transcribed\s+by|subt[íi]tulos\s+por", re.IGNORECASE),
    re.compile(r"please\s+subscribe|like\s+and\s+subscribe|suscr[íi]bete", re.IGNORECASE),
    re.compile(r"^\s*(you|thank you\.?|bye\.?|gracias\.?)\s*$", re.IGNORECASE),
    re.compile(r"^[\W_♪♫♬]+$"),
)

_NON_SPEECH_TAG_RE = re.compile(
    r"\[[^\]]*\]|\((?:music|música|applause|aplausos|laughter|risas|cheering|sighs?|suspiros?|silence|silencio|noise|ruido|inaudible|sfx)[^)]*\)|[♪♫♬]+",
    re.IGNORECASE,
)
_REPEAT_LOOP_RE = re.compile(r"\b(\w+)(?:[\s,]+\1){3,}\b", re.IGNORECASE)


class WhisperTranslatorEngine:
    def __init__(self, model_size: str = "small") -> None:
        self.model_size: str = model_size
        self.device: str = "cuda"
        self.compute_type: str = "int8_float16"

        try:
            self._model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="int8_float16",
                cpu_threads=4,
            )
            dummy = np.zeros((8000,), dtype=np.float32)
            list(self._model.transcribe(dummy, beam_size=1, language="es", task="translate")[0])
        except Exception:
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=6)

    @staticmethod
    def _clean_transcript(text: str) -> str:
        cleaned = _NON_SPEECH_TAG_RE.sub(" ", text or "")
        cleaned = _REPEAT_LOOP_RE.sub(r"\1", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        cleaned = text.strip()
        if len(cleaned) < 2:
            return True
        if not re.search(r"[0-9A-Za-zÀ-ÿА-Яа-я一-龯ぁ-ゔァ-ヴー가-힣]", cleaned):
            return True
        for pattern in HALLUCINATION_PATTERNS:
            if pattern.search(cleaned):
                return True
        return False

    def translate_es_to_en(self, audio_16k: NDArray[np.float32], source_lang: str = "es") -> str:
        lang_arg = None if source_lang == "auto" else source_lang
        segments, _ = self._model.transcribe(
            audio_16k,
            language=lang_arg,
            task="translate",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-0.85,
            compression_ratio_threshold=2.2,
            initial_prompt=GAMING_PROMPT_EN,
        )
        text = self._clean_transcript(" ".join(seg.text.strip() for seg in segments))
        return "" if self._is_hallucination(text) else text

    def transcribe_any(
        self, audio_16k: NDArray[np.float32], source_lang: str = "auto"
    ) -> tuple[str, str, float]:
        """
        Transcribe en el idioma indicado o detecta automáticamente el idioma hablado.
        Devuelve (texto_transcrito, codigo_idioma_detectado, probabilidad).
        """
        lang_arg = None if source_lang == "auto" else source_lang
        segments, info = self._model.transcribe(
            audio_16k,
            language=lang_arg,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-0.85,
            compression_ratio_threshold=2.2,
            initial_prompt=GAMING_PROMPT_EN if lang_arg == "en" else None,
        )
        text = self._clean_transcript(" ".join(seg.text.strip() for seg in segments))
        if self._is_hallucination(text):
            return "", info.language or "en", 0.0
        return text, (info.language or "en"), float(info.language_probability)

    def transcribe_en(self, audio_16k: NDArray[np.float32]) -> str:
        text, _, _ = self.transcribe_any(audio_16k, source_lang="en")
        return text
