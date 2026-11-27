# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A call that ran with no room for its output still reports how it ended (#19352)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools as tools_module  # noqa: E402

_LONG = "\n".join(map(str, range(1, 401)))


@pytest.fixture
def no_room():
    ctx = tools_module._REQUEST_CONTEXT_TOKENS.set(8192)
    room = tools_module._REQUEST_RESULT_BUDGET.set(0)
    yield
    tools_module._REQUEST_RESULT_BUDGET.reset(room)
    tools_module._REQUEST_CONTEXT_TOKENS.reset(ctx)


def test_a_failed_run_keeps_its_exit_code(no_room, tmp_path):
    failed = tools_module._truncate(f"Exit code 3:\n{_LONG}", workdir = str(tmp_path))
    succeeded = tools_module._truncate(_LONG, workdir = str(tmp_path))

    assert "no context room left" in failed and "no context room left" in succeeded
    assert failed.startswith("Exit code 3:\n")
    assert not succeeded.startswith("Exit code")


def test_a_negative_exit_code_is_kept(no_room, tmp_path):
    assert tools_module._truncate(f"Exit code -9:\n{_LONG}", workdir = str(tmp_path)).startswith(
        "Exit code -9:\n("
    )
