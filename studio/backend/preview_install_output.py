#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Preview installation-style terminal output (no pip, no Node, no setup.sh).

Run from repo root:

  python studio/backend/preview_install_output.py
  python studio/backend/preview_install_output.py --color on | less -R
  python studio/backend/preview_install_output.py --color off

Edits to install_python_stack.py (_progress, _step, colors) show up after
re-running this script — no full studio setup required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_STUDIO_DIR = _BACKEND.parent


def _apply_color_mode(mode: str) -> None:
    if mode == "off":
        os.environ["NO_COLOR"] = "1"
        os.environ.pop("FORCE_COLOR", None)
    elif mode == "on":
        os.environ.pop("NO_COLOR", None)
        os.environ["FORCE_COLOR"] = "1"
    else:
        os.environ.pop("FORCE_COLOR", None)
        os.environ.pop("NO_COLOR", None)


def _print_setup_sh_style_samples(ips) -> None:
    """Approximate setup.sh output using the same column format."""
    rule = "\u2500" * 52
    print()
    print(f"  {ips._title('🦥 Unsloth Studio Setup')}")
    print(f"  {ips._dim(rule)}")
    ips._step("node", "v20.0.0 | npm 11.0.0")
    ips._step("frontend", "built")
    ips._step("python", "3.12.0 (3.11.x \u2013 3.13.x)")


def _progress_demo(ips) -> None:
    """Drive the real _progress() + success line from install_python_stack."""
    labels = [
        "pip upgrade",
        "base packages",
        "unsloth extras",
        "extra codecs",
        "dependency overrides",
    ]
    if not ips.IS_WINDOWS:
        labels.append("triton kernels")
    labels.extend(
        [
            "studio deps",
            "data designer deps",
            "data designer",
            "local plugin",
            "finalizing",
        ]
    )

    ips._STEP = 0
    ips._TOTAL = len(labels)

    delay = 0.06
    for lb in labels:
        ips._progress(lb)
        time.sleep(delay)
    ips._step(ips._LABEL, "installed")


def _print_remaining_steps(ips) -> None:
    """Approximate the remaining setup.sh steps after Python deps."""
    ips._step("transformers", "5.x pre-installed")
    ips._step("llama.cpp", "built")
    ips._step("llama-quantize", "built")
    rule = "\u2500" * 52
    print(f"  {ips._dim(rule)}")
    print(f"  {ips._title('Unsloth Studio Installed')}")
    ips._step("launch", "unsloth studio -H 0.0.0.0 -p 8888")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Preview install-time terminal styling (no installs).",
    )
    parser.add_argument(
        "--color",
        choices = ("auto", "on", "off"),
        default = "auto",
        help = "on = FORCE_COLOR; off = NO_COLOR",
    )
    parser.add_argument(
        "--python-only",
        action = "store_true",
        help = "Only run the pip progress demo.",
    )
    parser.add_argument(
        "--setup-only",
        action = "store_true",
        help = "Only print setup.sh-style sample lines.",
    )
    args = parser.parse_args()
    _apply_color_mode(args.color)

    if str(_STUDIO_DIR) not in sys.path:
        sys.path.insert(0, str(_STUDIO_DIR))

    import install_python_stack as ips  # noqa: E402

    if args.setup_only:
        _print_setup_sh_style_samples(ips)
        return

    if not args.python_only:
        _print_setup_sh_style_samples(ips)

    _progress_demo(ips)

    if not args.python_only:
        _print_remaining_steps(ips)

    print(ips._dim("  No packages were installed \u2014 safe to run repeatedly."))
    print()


if __name__ == "__main__":
    main()
