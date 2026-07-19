# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh must map UNSLOTH_LLAMA_CPP_BACKEND=cpu to install_llama_prebuilt.py's
--cpu-fallback so users can force the CPU-only prebuilt on GPU hosts (#7213). The
match is case-insensitive, matching setup.ps1's PowerShell -eq. Runs the real
block extracted from setup.sh so the test tracks the shipped logic.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_SETUP_SH = Path(__file__).resolve().parents[2] / "setup.sh"


def _backend_case_block() -> str:
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(r'case "\$\{UNSLOTH_LLAMA_CPP_BACKEND.*?esac', text, re.DOTALL)
    assert m, "UNSLOTH_LLAMA_CPP_BACKEND case block not found in setup.sh"
    return m.group(0)


def _run(value: str | None) -> list[str]:
    assign = "" if value is None else f"export UNSLOTH_LLAMA_CPP_BACKEND={value}\n"
    script = (
        f'_PREBUILT_CMD=()\n{assign}{_backend_case_block()}\nprintf "%s\\n" "${{_PREBUILT_CMD[@]}}"'
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, check = True)
    return out.stdout.split()


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash unavailable")
@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu"])
def test_backend_cpu_appends_flag(value):
    assert "--cpu-fallback" in _run(value)


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash unavailable")
@pytest.mark.parametrize("value", [None, "auto", "vulkan"])
def test_backend_non_cpu_no_flag(value):
    assert "--cpu-fallback" not in _run(value)
