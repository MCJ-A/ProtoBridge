from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from ignait.paths import get_config_path

CONFIG_PATH = get_config_path()

DEFAULT_GAMING_GLOSSARY: dict[str, str] = {
    "rush": "rusheo",
    "push": "empuje",
    "flank": "flanqueo",
    "rotate": "rotación",
    "fall back": "retroceder",
    "retreat": "retirada",
    "low hp": "poca vida",
    "one shot": "a un tiro",
    "lit": "tocado",
    "cracked": "sin escudo",
    "shield broken": "escudo roto",
    "downed": "derribado",
    "knocked": "noqueado",
    "headshot": "tiro a la cabeza",
    "reload": "recargando",
    "reloading": "recargando",
    "cover me": "cúbreme",
    "hold": "mantengan posición",
    "peek": "asomarse",
    "clear": "despejado",
    "gg": "buen juego",
    "clutch": "clutch",
    "defuse": "desactivar",
    "plant": "plantar",
    "camp": "campear",
    "sniper": "francotirador",
}


@dataclass
class AppConfig:
    mic_device_name: str = ""
    headphones_device_name: str = ""
    cable_a_device_name: str = ""
    team_loopback_device_name: str = ""
    my_source_lang: str = "es"
    my_target_lang: str = "en"
    teammate_source_lang: str = "auto"
    teammate_target_lang: str = "es"
    whisper_model_size: str = "small"
    my_voice_profile: str = "am_adam"
    tts_speed_out: float = 1.10
    tts_speed_in: float = 1.14
    vad_threshold: float = 0.52
    vad_silence_ms: int = 240
    monitor_own_voice: bool = False
    mute_tx: bool = False
    mute_rx: bool = False

    # Controles tácticos y hotkeys globales
    hotkey_mute_tx: str = "ctrl+shift+m"
    hotkey_mute_rx: str = "ctrl+shift+n"
    hotkey_ptt: str = "caps_lock"
    hotkey_reset_squad: str = "ctrl+shift+r"
    ptt_enabled: bool = False

    # Retroalimentación acústica y gestión de colas
    tactical_chirp_enabled: bool = True
    barge_in_enabled: bool = True

    # Modo Media / Navegación (Lookahead Buffer + Smart Ducking + Worker Aislado XTTS-v2)
    media_mode_enabled: bool = False
    media_auto_sync: bool = True
    media_delay_ms: int = 2500
    media_ducking_db: float = -14.0
    media_tts_backend: str = "xtts_v2"  # "xtts_v2" (worker GPU 3.12) | "kokoro" (CPU) | "auto"
    media_whisper_model_size: str = "large-v3-turbo"  # "small" | "medium" | "large-v3-turbo" | "follow"
    media_diarization_enabled: bool = True
    media_max_speakers: int = 8

    # Glosario táctico personalizable desacoplado
    custom_glossary: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_GAMING_GLOSSARY)
    )

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> AppConfig:
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            if "tactical_glossary" in data and "custom_glossary" not in filtered:
                tactical = data["tactical_glossary"]
                if isinstance(tactical, dict):
                    filtered["custom_glossary"] = tactical
            return cls(**filtered)
        except Exception:
            return cls()
