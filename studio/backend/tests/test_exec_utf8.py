# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_python_exec must round-trip non-ASCII output end to end.

Model-written code routinely contains non-ASCII (arrows, CJK, emoji). The temp
script and the child's stdout pipe both have to be UTF-8 or it crashes/garbles
on Windows, whose default codec is cp1252. Mirrors the report in
unslothai/unsloth#6489. The child is ``python`` with PYTHONIOENCODING=utf-8, so
it emits UTF-8 on every OS; this proves the round-trip on a UTF-8 host and
guards against a regression to the OS default codec.
"""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _python_exec

# Arrow, em-dash, accent, CJK, check mark, astral-plane emoji -- none encodable
# in cp1252, so the OS default codec would raise on write or read.
_UNICODE = "café — 数字 → ✓ 😀"


@pytest.mark.parametrize("disable_sandbox", [False, True])
def test_python_exec_round_trips_non_ascii(disable_sandbox):
    out = _python_exec(f"print({_UNICODE!r})", disable_sandbox = disable_sandbox)
    assert _UNICODE in out, repr(out)
