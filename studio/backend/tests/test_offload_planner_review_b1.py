# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A plan the planner declines launches the flag-off argv: the spill micro-batch raise and the
MoE cache admission read the same resident expert share either way."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.offload_planner import Plan  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402
    _cache_launch,
    _discrete_linux_host,  # noqa: F401
    _has_moe_cache,
    _moe_cache_host,  # noqa: F401
    _ubatch_values,
)

_MASKED = ("--port", "--slot-save-path", "--api-key-file", "-m")


def _masked(cmd):
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        out.append(tok)
        if tok in _MASKED:
            out.append("<masked>")
            skip = True
    return out


def _compute_buffer_per_ubatch(monkeypatch, bytes_per_token):
    """A compute buffer that grows with the micro-batch, so the raise costs resident experts."""
    monkeypatch.setattr(
        LlamaCppBackend,
        "_estimate_compute_buffer_bytes",
        lambda self, n_ubatch = None, n_parallel = 1, per_device_tensor = False, **_kw: (
            int(n_ubatch or 512) * bytes_per_token
        ),
    )


def _declined(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_planned_tensor_spill",
        lambda self, inputs, **_kw: Plan(reason = "declined for the test"),
    )


@pytest.mark.parametrize(
    "free_mib, ubatch_mib_per_token, cache, ubatch",
    [
        # ~53% of 15 GiB of experts resident: inside the cache band, the raise is free.
        (20_000, 0, True, ["2048"]),
        # 1 MiB per micro-batch token: the raise would push >10% of the resident experts off.
        (10_000, 1, False, []),
    ],
    ids = ["cache_in_band", "raise_withheld"],
)
def test_a_declined_plan_launches_the_flag_off_argv(
    tmp_path, _moe_cache_host, monkeypatch, free_mib, ubatch_mib_per_token, cache, ubatch
):
    """The resident-share estimate abstained whenever UNSLOTH_SMART_OFFLOAD was set, though both
    of its readers run before the planner and a declined plan launches --fit on like flag off.
    The unknown share dropped --moe-cache-mib auto (qwen_16 on an L4: decode -22..-25%) and kept
    a --ubatch-size 2048 flag off withholds (gemma_12 / qwen_8 on a T4: decode -2..-8%)."""
    if ubatch_mib_per_token:
        _compute_buffer_per_ubatch(monkeypatch, ubatch_mib_per_token * 1024 * 1024)
    _declined(monkeypatch)
    cmds = {}
    for flag in ("off", "on"):
        if flag == "on":
            monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
        else:
            monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
        arm = tmp_path / flag
        arm.mkdir()
        _backend_, cmds[flag] = _cache_launch(arm, memory = [(0, free_mib, 32_000)], ram_gib = 256)
    for flag, cmd in cmds.items():
        assert cmd[cmd.index("--fit") + 1] == "on", (flag, cmd)
        assert _has_moe_cache(cmd) is cache, (flag, cmd)
        assert _ubatch_values(cmd) == ubatch, (flag, cmd)
    assert _masked(cmds["on"]) == _masked(cmds["off"])
