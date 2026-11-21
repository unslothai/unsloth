# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fresh interpreter per case: torch latches the variable process-wide at the first SDPA dispatch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
VAR = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def _value_after_import(module: str, preset: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != VAR}
    if preset is not None:
        env[VAR] = preset
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        f"import {module}\n"
        f"print('VALUE=' + str(os.environ.get({VAR!r})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("VALUE=")][-1]
    return line.split("=", 1)[1]


@pytest.mark.parametrize("module", ["core.inference", "core.inference.chat_eos"])
def test_inference_import_sets_the_optin(module):
    assert _value_after_import(module, None) == "1"


@pytest.mark.parametrize("preset", ["0", "1", "false"])
def test_user_value_is_kept(preset):
    assert _value_after_import("core.inference", preset) == preset


def test_import_does_not_pull_torch():
    env = {k: v for k, v in os.environ.items() if k != VAR}
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "import core.inference\n"
        "print('TORCH=' + str('torch' in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "TORCH=False" in out.stdout
