#!/usr/bin/env python3
"""Run `ruff format` followed by kwarg spacing enforcement."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    files = [arg for arg in argv if Path(arg).exists()]
    if not files:
        return 0

    ruff_cmd = [sys.executable, "-m", "ruff", "format", *files]
    ruff_proc = subprocess.run(ruff_cmd)
    if ruff_proc.returncode != 0:
        return ruff_proc.returncode

    spacing_script = HERE / "enforce_kwargs_spacing.py"
    spacing_cmd = [sys.executable, str(spacing_script), *files]
    spacing_proc = subprocess.run(spacing_cmd)
    return spacing_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
