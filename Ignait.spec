# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas = [
    ("ignait/ui_stitch.html", "ignait"),
    ("ignait/media/xtts_worker.py", "ignait/media"),
    ("browser_extension", "browser_extension"),
]
binaries = []
hiddenimports = [
    "ignait.paths",
    "pyaudiowpatch",
    "soxr",
    "webview",
    "clr_loader",
    "pythonnet",
    "pystray",
    "PIL",
]

for pkg in (
    "espeakng_loader",
    "kokoro_onnx",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "phonemizer",
    "segments",
    "csvw",
    "language_tags",
    "webview",
    "pystray",
    "PIL",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

for cuda_pkg in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        binaries += collect_dynamic_libs(cuda_pkg)
    except Exception:
        pass

a = Analysis(
    ["ignait/gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Ignait",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Ignait",
)
