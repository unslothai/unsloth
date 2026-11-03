# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Non-ASCII must round-trip, since Windows defaults to cp1252 and would crash or garble it."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _python_exec

# none of these are encodable in cp1252
_UNICODE = "café — 数字 → ✓ 😀"


@pytest.mark.parametrize("disable_sandbox", [False, True])
def test_python_exec_round_trips_non_ascii(disable_sandbox):
    out = _python_exec(f"print({_UNICODE!r})", disable_sandbox = disable_sandbox)
    assert _UNICODE in out, repr(out)
