from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")


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
            return cls(**filtered)
        except Exception:
            return cls()
