# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files


datas = []
datas += collect_data_files("matplotlib")
conda_root = Path(sys.prefix)
tcl_dir = conda_root / "Library" / "lib" / "tcl8.6"
tk_dir = conda_root / "Library" / "lib" / "tk8.6"
if tcl_dir.exists():
    datas.append((str(tcl_dir), "tcl/tcl8.6"))
if tk_dir.exists():
    datas.append((str(tk_dir), "tcl/tk8.6"))

binaries = []
tkinter_pyd = conda_root / "DLLs" / "_tkinter.pyd"
tcl_dll = conda_root / "Library" / "bin" / "tcl86t.dll"
tk_dll = conda_root / "Library" / "bin" / "tk86t.dll"
for binary in (tkinter_pyd, tcl_dll, tk_dll):
    if binary.exists():
        binaries.append((str(binary), "."))

hiddenimports = [
    "matplotlib.backends.backend_tkagg",
    "_tkinter",
    "scipy.signal",
    "scipy.io.matlab",
    "scipy.special",
]

excludes = [
    "dask",
    "IPython",
    "jupyter",
    "numba",
    "pandas",
    "sklearn",
    "tensorflow",
    "torch",
]


a = Analysis(
    ["run_sleep_score_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["tools/pyinstaller_tk_runtime.py"],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OnlineSleepScore",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OnlineSleepScore",
)
