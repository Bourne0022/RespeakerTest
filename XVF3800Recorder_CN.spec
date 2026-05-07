# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = []
binaries += collect_dynamic_libs('libusb_package')

datas = []

python_base = Path(sys.executable).resolve().parent
python_dlls = python_base / 'DLLs'
python_tcl = python_base / 'tcl'

# PyInstaller's tkinter hook can mis-detect this Windows Python install as
# broken. Include Tk explicitly so the one-file GUI can start on the target PC.
binaries += [
    (str(python_dlls / '_tkinter.pyd'), '.'),
    (str(python_dlls / 'tcl86t.dll'), '.'),
    (str(python_dlls / 'tk86t.dll'), '.'),
]
datas += [
    (str(python_tcl / 'tcl8.6'), 'tcl/tcl8.6'),
    (str(python_tcl / 'tk8.6'), 'tcl/tk8.6'),
    (str(python_tcl / 'tcl8'), 'tcl/tcl8'),
]


a = Analysis(
    ['xvf3800_recorder_gui_cn.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        '_tkinter',
        'sounddevice',
        'usb.core',
        'usb.util',
        'usb.backend.libusb1',
        'libusb_package',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_tkinter_local.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='XVF3800Recorder_CN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
