# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup installs flash-attn but not causal-conv1d or mamba-ssm: those (~550 MB) install when a
model that needs them is trained or loaded (core/training/worker.py, utils/ssm_runtime.py), and
`unsloth install-kernels` still provisions them ahead of time for offline images."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips  # noqa: E402

SOURCE = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")
TREE = ast.parse(SOURCE)


def _called_names() -> set[str]:
    return {
        node.func.id
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_setup_has_no_ssm_kernel_step():
    assert not hasattr(ips, "_ensure_ssm_kernels")
    assert "install_kernel" not in _called_names()
    assert '_progress("mamba kernels")' not in SOURCE
    assert "causal-conv1d + mamba-ssm" not in SOURCE


def test_flash_attn_is_still_installed_and_resynced_at_all_three_sites():
    calls = [m.start() for m in re.finditer(r"\n\s+_ensure_flash_attn\(\)\n", SOURCE)]
    assert len(calls) == 3
    for start in calls:
        following = SOURCE[start : start + 200]
        assert "ssm" not in following.lower(), following
    assert "base_total += 1  # flash-attn, Linux and Windows" in SOURCE


def test_the_kernel_cli_still_provisions_both_ahead_of_time():
    from utils import kernel_install
    assert {"causal_conv1d", "mamba_ssm"} <= set(kernel_install.KERNELS)
    assert callable(kernel_install.install_kernel)
