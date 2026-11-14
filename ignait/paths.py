from __future__ import annotations

import os
from pathlib import Path
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_app_dir() -> Path:
    """Directorio base persistente (junto al .exe en builds PyInstaller, o CWD en desarrollo)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def get_models_dir() -> Path:
    return get_app_dir() / "models"


def get_xtts_models_dir() -> Path:
    """Directorio de pesos locales XTTS-v2 para el worker aislado de Media Mode."""
    return get_models_dir() / "xtts_v2"


def get_xtts_worker_script() -> Path:
    """Resuelve la ruta física de xtts_worker.py en desarrollo o empaquetado PyInstaller."""
    meipass = getattr(sys, "_MEIPASS", None)
    if is_frozen() and meipass:
        candidate = Path(meipass) / "ignait" / "media" / "xtts_worker.py"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent / "media" / "xtts_worker.py"


def get_media_tts_python() -> Path | None:
    """Resuelve el intérprete Python 3.12.8 aislado (.venv-media-tts) para Media TTS GPU."""
    env_override = os.environ.get("IGNAIT_MEDIA_PYTHON", "").strip()
    if env_override:
        p = Path(env_override)
        if p.exists():
            return p

    candidates: list[Path] = []
    app_dir = get_app_dir()
    repo_root = Path(__file__).resolve().parent.parent
    bases = [app_dir, app_dir.parent, app_dir.parent.parent, repo_root]
    rel_bins = (
        Path(".venv-media-tts") / "Scripts" / "python.exe",
        Path(".venv-media-tts") / "bin" / "python",
    )
    for base in bases:
        for rel in rel_bins:
            cand = base / rel
            if cand not in candidates:
                candidates.append(cand)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def get_config_path() -> Path:
    return get_app_dir() / "config.json"


def get_ui_html_path() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if is_frozen() and meipass:
        candidate = Path(meipass) / "ignait" / "ui_stitch.html"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent / "ui_stitch.html"


def get_app_icon_path() -> Path:
    """Devuelve la ruta del icono oficial ignait.ico (PyInstaller o desarrollo)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if is_frozen() and meipass:
        candidate = Path(meipass) / "ignait" / "assets" / "ignait.ico"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent / "assets" / "ignait.ico"


def get_browser_extension_dir() -> Path:
    """Devuelve la carpeta descomprimida de la extensión Chrome/Edge de Ignait."""
    if is_frozen():
        ext_next = get_app_dir() / "browser_extension"
        if ext_next.exists():
            return ext_next
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cand = Path(meipass) / "browser_extension"
            if cand.exists():
                return cand
    return Path(__file__).resolve().parent.parent / "browser_extension"
