"""Runtime Tcl/Tk paths for the packaged XVF3800 Tkinter GUI."""

import os
import sys
from pathlib import Path

base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
tcl_root = base / "tcl"

os.environ.setdefault("TCL_LIBRARY", str(tcl_root / "tcl8.6"))
os.environ.setdefault("TK_LIBRARY", str(tcl_root / "tk8.6"))
