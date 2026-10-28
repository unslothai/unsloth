# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A revoked plan that kept every layer on the card re-arms the page-lock it skipped."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.offload_planner import Plan  # noqa: E402
from test_offload_planner_launch import _flag, _launch_crash_then_ok  # noqa: E402


def _locks(cmd):
    return "--mlock" in cmd or _flag(cmd, "--load-mode") == "mmap+mlock"


@pytest.mark.parametrize("load_mode", [False, True], ids = ["mlock", "load_mode"])
def test_a_revoked_all_on_gpu_plan_relaunches_with_the_page_lock(tmp_path, monkeypatch, load_mode):
    """The knob-only plan launches -ngl -1 --fit off with every layer on the card, so the gate
    skipped the page-lock. It crashes, and the --fit on relaunch may leave layers in host RAM:
    without the lock those pages are evictable and the bookkeeping says the skip was deliberate."""
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (True, False))
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": load_mode}
    )
    assert _flag(cmds[0], "--fit") == "off" and not _locks(cmds[0]), cmds[0]
    assert _flag(cmds[-1], "--fit") == "on", cmds[-1]
    assert _locks(cmds[-1]), cmds[-1]
    assert backend._memory_mlock_applicable is True


def test_a_revoked_plan_that_spilled_is_not_locked_twice(tmp_path, monkeypatch):
    """A plan with -ot already put weights in host RAM, so the first launch carried the lock."""
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (True, False))
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    cmds, _backend = _launch_crash_then_ok(tmp_path, monkeypatch, plan)
    assert _locks(cmds[0]), cmds[0]
    assert cmds[-1].count("--mlock") + cmds[-1].count("--load-mode") <= 1, cmds[-1]
