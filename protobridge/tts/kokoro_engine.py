from __future__ import annotations

from pathlib import Path
import urllib.request
import numpy as np
from numpy.typing import NDArray
from kokoro_onnx import Kokoro

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

ISO_TO_KOKORO_LANG: dict[str, tuple[str, str]] = {
    "en": ("en-us", "am_adam"),
    "es": ("es", "em_alex"),
    "fr": ("fr-fr", "ff_siwis"),
    "it": ("it", "im_nicola"),
    "pt": ("pt-br", "pm_alex"),
}


def ensure_kokoro_assets(models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "kokoro-v1.0.int8.onnx"
    voices_path = models_dir / "voices-v1.0.bin"

    if not model_path.exists():
        urllib.request.urlretrieve(KOKORO_MODEL_URL, model_path)
    if not voices_path.exists():
        urllib.request.urlretrieve(KOKORO_VOICES_URL, voices_path)

    return model_path, voices_path


class KokoroTTSEngine:
    def __init__(self, models_dir: Path, default_voice: str = "am_adam") -> None:
        model_path, voices_path = ensure_kokoro_assets(models_dir)
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self.default_voice = default_voice

        style_alex = self._kokoro.get_voice_style("em_alex")
        style_george = self._kokoro.get_voice_style("bm_george")
        self._tactical_blend: NDArray[np.float32] = (
            0.60 * style_alex + 0.40 * style_george
        ).astype(np.float32)

        self._kokoro.create("Ready.", voice=self.default_voice, speed=1.1, lang="en-us")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        lang: str = "en-us",
        speed: float = 1.12,
    ) -> tuple[NDArray[np.float32], int]:
        norm_lang, fallback_voice = ISO_TO_KOKORO_LANG.get(lang, (lang, self.default_voice))
        if norm_lang not in ("en-us", "en-gb", "es", "fr-fr", "it", "pt-br"):
            norm_lang = "es"
        selected = voice or fallback_voice
        voice_arg: str | NDArray[np.float32] = (
            self._tactical_blend if selected == "em_tactical" else selected
        )
        samples, sample_rate = self._kokoro.create(
            text, voice=voice_arg, speed=speed, lang=norm_lang
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
