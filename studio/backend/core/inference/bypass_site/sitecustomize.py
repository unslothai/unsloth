# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load path remapping without the sandbox network / spawn guards.

``run_path`` runs the sandbox shim with ``__name__ != "sitecustomize"``, the
condition ``_install_import_guard`` checks, so the remap runs and guards stay off.
"""

import runpy
from pathlib import Path


_SHIM = Path(__file__).resolve().parents[1] / "sandbox_site" / "sitecustomize.py"
runpy.run_path(str(_SHIM))
