# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The kwarg-spacing pass must leave a file's mode alone.

It rewrites through a temp file and os.replace, and mkstemp creates that temp
file 0600. scripts/run_ruff_format.py is the pre-commit hook's own entry point
and is processed by the hook it runs, so once it carried kwarg spacing every
run stripped its executable bit, pre-commit.ci committed the flip, and the next
run failed before doing anything with "not executable".
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "enforce_kwargs_spacing.py"


def _rewrite(tmp_path: Path, mode: int) -> Path:
    target = tmp_path / "tool.py"
    target.write_text("def f(a=1):\n    return g(b=2)\n", encoding = "utf-8")
    os.chmod(target, mode)
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(target)], capture_output = True, text = True, timeout = 120
    )
    assert proc.returncode == 0, proc.stderr
    assert "g(b = 2)" in target.read_text(encoding = "utf-8"), "the pass did not rewrite the file"
    return target


def test_an_executable_script_keeps_its_bit_through_the_rewrite(tmp_path):
    target = _rewrite(tmp_path, 0o755)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_a_plain_file_keeps_its_group_and_other_read_bits(tmp_path):
    target = _rewrite(tmp_path, 0o644)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
