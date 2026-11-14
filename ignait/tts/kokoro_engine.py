from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import threading
import unicodedata
import urllib.request
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort
import soxr
from kokoro_onnx import Kokoro

from ignait.diarization.speaker_id import OnlineSpeakerTracker

logger = logging.getLogger(__name__)

PIPER_RU_REPO = "rhasspy/piper-voices"
PIPER_RU_ONNX = "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx"
PIPER_RU_JSON = "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json"

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

CONTRACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bI am\b", re.IGNORECASE), "I'm"),
    (re.compile(r"\bdo not\b", re.IGNORECASE), "don't"),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "they're"),
    (re.compile(r"\bwe are\b", re.IGNORECASE), "we're"),
    (re.compile(r"\bit is\b", re.IGNORECASE), "it's"),
)


def ensure_kokoro_assets(models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "kokoro-v1.0.int8.onnx"
    voices_path = models_dir / "voices-v1.0.bin"

    if not model_path.exists():
        urllib.request.urlretrieve(KOKORO_MODEL_URL, model_path)
    if not voices_path.exists():
        urllib.request.urlretrieve(KOKORO_VOICES_URL, voices_path)

    return model_path, voices_path


def _ensure_piper_ru_assets(models_dir: Path) -> tuple[Path, Path]:
    piper_dir = models_dir / "piper"
    onnx_path = piper_dir / PIPER_RU_ONNX
    json_path = piper_dir / PIPER_RU_JSON
    if not onnx_path.exists() or not json_path.exists():
        from huggingface_hub import hf_hub_download

        piper_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(PIPER_RU_REPO, PIPER_RU_ONNX, local_dir=str(piper_dir))
        hf_hub_download(PIPER_RU_REPO, PIPER_RU_JSON, local_dir=str(piper_dir))
    return onnx_path, json_path


class _PiperRussianTTS:
    """Sintetizador TTS local para ruso (VITS/Piper ONNX + espeak-ng, 100% CPU offline)."""

    _espeak_init_done: bool = False
    _espeak_fallback_lock = threading.Lock()

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._id_map: dict[str, list[int]] = {}
        self._sample_rate: int = 22050
        self._noise_scale: float = 0.667
        self._length_scale: float = 1.0
        self._noise_w: float = 0.8

        onnx_path = models_dir / "piper" / PIPER_RU_ONNX
        json_path = models_dir / "piper" / PIPER_RU_JSON
        if onnx_path.exists() and json_path.exists():
            self._load_locked(onnx_path, json_path)

    def _get_espeak_lock(self) -> threading.Lock:
        try:
            import kokoro_onnx.tokenizer as ktok

            lock = getattr(ktok, "_espeak_lock", None)
            if isinstance(lock, type(self._espeak_fallback_lock)):
                return lock
        except ImportError:
            pass
        return self._espeak_fallback_lock

    def _load_locked(self, onnx_path: Path, json_path: Path) -> None:
        cfg = json.loads(json_path.read_text(encoding="utf-8"))
        self._id_map = dict(cfg.get("phoneme_id_map", {}))
        audio_cfg = cfg.get("audio", {})
        self._sample_rate = int(audio_cfg.get("sample_rate", 22050))
        infer_cfg = cfg.get("inference", {})
        self._noise_scale = float(infer_cfg.get("noise_scale", 0.667))
        self._length_scale = float(infer_cfg.get("length_scale", 1.0))
        self._noise_w = float(infer_cfg.get("noise_w", 0.8))

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            onnx_path, json_path = _ensure_piper_ru_assets(self._models_dir)
            self._load_locked(onnx_path, json_path)

    def _phonemize_ru(self, text: str) -> str:
        import espeakng_loader
        from phonemizer import phonemize
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        with self._get_espeak_lock():
            if not _PiperRussianTTS._espeak_init_done:
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
                _PiperRussianTTS._espeak_init_done = True
            return str(
                phonemize(
                    text,
                    language="ru",
                    backend="espeak",
                    preserve_punctuation=True,
                    with_stress=True,
                )
            )

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[NDArray[np.float32], int]:
        self._ensure_loaded()
        assert self._session is not None

        ph = unicodedata.normalize("NFD", self._phonemize_ru(text))
        ids: list[int] = [1, 0]
        for ch in ph:
            mapped = self._id_map.get(ch)
            if mapped:
                ids.extend(mapped)
                ids.append(0)
        ids.append(2)

        eff_length_scale = self._length_scale / float(np.clip(speed, 0.5, 2.0))
        inputs = {
            "input": np.asarray([ids], dtype=np.int64),
            "input_lengths": np.asarray([len(ids)], dtype=np.int64),
            "scales": np.asarray(
                [self._noise_scale, eff_length_scale, self._noise_w], dtype=np.float32
            ),
        }
        audio = self._session.run(None, inputs)[0].squeeze()
        return np.asarray(audio, dtype=np.float32), self._sample_rate


class KokoroTTSEngine:
    def __init__(self, models_dir: Path, default_voice: str = "my_cloned") -> None:
        model_path, voices_path = ensure_kokoro_assets(models_dir)
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self._ru_tts = _PiperRussianTTS(models_dir)
        self.default_voice = default_voice
        self.models_dir = models_dir
        self.user_f0: float = 122.0

        clone_file = models_dir / "my_voice_clone.npz"
        if clone_file.exists():
            try:
                loaded = np.load(clone_file)
                self.user_f0 = float(loaded["f0"])
            except (KeyError, ValueError, OSError) as exc:
                logger.warning("No se pudo cargar el perfil de clonación previo: %s", exc)

        style_alex = self._kokoro.get_voice_style("em_alex")
        style_george = self._kokoro.get_voice_style("bm_george")
        self._tactical_blend: NDArray[np.float32] = (
            0.60 * style_alex + 0.40 * style_george
        ).astype(np.float32)

        self._kokoro.create("Ready.", voice="am_adam", speed=1.1, lang="en-us")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        lang: str = "en-us",
        speed: float = 1.12,
        ref_audio_16k: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], int]:
        norm_lang, fallback_voice = ISO_TO_KOKORO_LANG.get(lang, (lang, "am_adam"))
        if norm_lang not in ("en-us", "en-gb", "es", "fr-fr", "it", "pt-br"):
            norm_lang = "es"

        if norm_lang.startswith("en"):
            for pattern, repl in CONTRACTION_PATTERNS:
                text = pattern.sub(repl, text)

        selected = voice or self.default_voice

        clone_file = self.models_dir / "my_voice_clone.npz"
        if clone_file.exists():
            try:
                loaded = np.load(clone_file)
                self.user_f0 = float(loaded["f0"])
            except (KeyError, ValueError, OSError) as exc:
                logger.debug("Error leyendo f0 de clone_file: %s", exc)

        if ref_audio_16k is not None and len(ref_audio_16k) > 1600:
            pitch = OnlineSpeakerTracker._estimate_pitch_hz(ref_audio_16k)
            self.user_f0 = 0.75 * self.user_f0 + 0.25 * pitch

        if lang.lower().startswith("ru"):
            if selected == "my_cloned":
                ratio = float(np.clip(self.user_f0 / 128.0, 0.88, 1.14))
                adjusted_speed = float(np.clip(speed / ratio, 0.75, 1.6))
                raw_pcm, sample_rate = self._ru_tts.synthesize(text, speed=adjusted_speed)
                resampled = soxr.resample(
                    raw_pcm, int(sample_rate * ratio), int(sample_rate), quality="HQ"
                )
                saturated = (np.tanh(1.15 * resampled) * 0.92).astype(np.float32)
                return saturated, int(sample_rate)
            return self._ru_tts.synthesize(text, speed=speed)

        if selected == "my_cloned":
            weight = float(np.clip((135.0 - self.user_f0) / 45.0, 0.15, 0.75))
            style_michael = self._kokoro.get_voice_style("am_michael")
            style_adam = self._kokoro.get_voice_style("am_adam")
            style_george = self._kokoro.get_voice_style("bm_george")

            voice_blend = (
                weight * style_michael
                + (0.85 - weight) * style_adam
                + 0.15 * style_george
            ).astype(np.float32)

            ratio = float(np.clip(self.user_f0 / 128.0, 0.88, 1.14))
            adjusted_speed = float(np.clip(speed / ratio, 0.75, 1.6))
            samples, sample_rate = self._kokoro.create(
                text, voice=voice_blend, speed=adjusted_speed, lang=norm_lang
            )
            raw_pcm = np.asarray(samples, dtype=np.float32)
            resampled = soxr.resample(raw_pcm, int(sample_rate * ratio), int(sample_rate), quality="HQ")
            saturated = (np.tanh(1.15 * resampled) * 0.92).astype(np.float32)
            return saturated, int(sample_rate)

        if selected == "em_tactical":
            voice_arg: str | NDArray[np.float32] = self._tactical_blend
        elif selected in self._kokoro.get_voices():
            voice_arg = selected
        else:
            voice_arg = fallback_voice

        samples, sample_rate = self._kokoro.create(
            text, voice=voice_arg, speed=speed, lang=norm_lang
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
