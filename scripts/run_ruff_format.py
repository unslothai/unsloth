#!/usr/bin/env python3
"""Run a pre-pass (normalize def-signature magic commas + collapse short
multi-line asserts), then `ruff format`, then the kwarg-spacing / import /
string-merge post-pass."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    files = [arg for arg in argv if Path(arg).exists()]
    if not files:
        return 0

    spacing_script = HERE / "enforce_kwargs_spacing.py"

    # Pre-ruff: normalize def-signature magic commas and strip the magic comma
    # from short multi-line asserts so ruff wraps/joins accordingly.
    pre_cmd = [sys.executable, str(spacing_script), "--pre", *files]
    pre_proc = subprocess.run(pre_cmd)
    if pre_proc.returncode != 0:
        return pre_proc.returncode

    ruff_cmd = [sys.executable, "-m", "ruff", "format", *files]
    ruff_proc = subprocess.run(ruff_cmd)
    if ruff_proc.returncode != 0:
        return ruff_proc.returncode

    spacing_cmd = [sys.executable, str(spacing_script), *files]
    spacing_proc = subprocess.run(spacing_cmd)
    return spacing_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
