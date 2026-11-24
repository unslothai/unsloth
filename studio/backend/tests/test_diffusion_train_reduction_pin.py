# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the training regional compile pins inductor's per-process reduction picks (R0_BLOCK and the
near-tied reduction configs), so same-seed runs in fresh processes run the same compiled forward."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch._inductor.config  # noqa: E402

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402
from core.training import diffusion_block_swap as block_swap  # noqa: E402
from core.training import diffusion_dit_trainer as dit  # noqa: E402
from core.training.diffusion_train_common import DiffusionLoraConfig  # noqa: E402

_INDUCTOR = torch._inductor.config
_KILL_SWITCH = "UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK"
_FILTER = "test_configs.force_filter_reduction_configs"
_HAS_FILTER = hasattr(_INDUCTOR.test_configs, "force_filter_reduction_configs")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    before = _INDUCTOR.dynamic_scale_rblock
    monkeypatch.delenv(_KILL_SWITCH, raising = False)
    monkeypatch.setattr(dit, "_should_compile", lambda *a, **k: True)
    monkeypatch.setattr(dit, "_compile_toolchain_missing", lambda: None)
    cc._reset_for_tests()
    yield
    cc._reset_for_tests()
    _INDUCTOR.dynamic_scale_rblock = before


class _Transformer:
    def __init__(self):
        self.kwargs = None

    def compile_repeated_blocks(self, **kwargs):
        self.kwargs = kwargs


def _compile(block_swap_on = False):
    t = _Transformer()
    events = []
    cfg = DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o")
    ok = dit._maybe_compile_transformer(
        t, cfg, False, "cuda", events.append, base_precision = "bf16", block_swap = block_swap_on
    )
    assert ok is True
    assert not [e for e in events if e.get("type") == "warning"]
    return t.kwargs


def test_training_compile_pins_dynamic_scale_rblock_off():
    _INDUCTOR.dynamic_scale_rblock = True
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is False


@pytest.mark.skipif(not _HAS_FILTER, reason = "torch has no inductor reduction-config filter")
def test_training_compile_carries_the_reduction_filter_per_compile():
    _INDUCTOR.test_configs.force_filter_reduction_configs = False
    kwargs = _compile()
    assert kwargs["options"] == {_FILTER: True}
    assert kwargs["dynamic"] is True and kwargs["fullgraph"] is True
    # per compile, never the process knob
    assert _INDUCTOR.test_configs.force_filter_reduction_configs is False


@pytest.mark.skipif(not _HAS_FILTER, reason = "torch has no inductor reduction-config filter")
def test_block_swap_compile_carries_the_filter(monkeypatch):
    seen = {}

    def fake(transformer, **kwargs):
        seen.update(kwargs)
        return 1

    monkeypatch.setattr(dit, "compile_swapped_blocks", fake)
    _compile(block_swap_on = True)
    assert seen["options"] == {_FILTER: True}


def test_compile_swapped_blocks_passes_options(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn

    class Block:
        def forward(self):
            return None

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(block_swap, "swap_block_lists", lambda t: [("blocks", [Block(), Block()])])
    assert block_swap.compile_swapped_blocks(object(), fullgraph = True, options = {_FILTER: True}) == 2
    assert all(c["options"] == {_FILTER: True} for c in calls)


def test_kill_switch_keeps_inductor_picks(monkeypatch):
    monkeypatch.setenv(_KILL_SWITCH, "1")
    _INDUCTOR.dynamic_scale_rblock = True
    kwargs = _compile()
    assert _INDUCTOR.dynamic_scale_rblock is True
    assert "options" not in kwargs


def test_perf_flag_restore_undoes_the_rblock_pin():
    _INDUCTOR.dynamic_scale_rblock = True
    snap = ds.snapshot_backend_flags()
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is False
    ds.restore_backend_flags(snap)
    assert _INDUCTOR.dynamic_scale_rblock is True
