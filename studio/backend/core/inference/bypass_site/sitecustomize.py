"""Load path remapping without sandbox network guards."""

import runpy
from pathlib import Path


_SHIM = Path(__file__).resolve().parents[1] / "sandbox_site" / "sitecustomize.py"
runpy.run_path(str(_SHIM))
