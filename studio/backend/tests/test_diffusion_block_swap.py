# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for the opt-in DiT block swap: request parsing, gating, planning and the compile wiring."""

from __future__ import annotations

import types

import pytest
import torch

import core.training.diffusion_block_swap as bs
import core.training.diffusion_dit_trainer as dit
from core.training.diffusion_train_common import DiffusionLoraConfig, _config_from_dict


def _cfg(**kw):
    return types.SimpleNamespace(**kw)


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, None),
        ("off", None),
        ("", None),
        ("0", None),
        (0, None),
        ("auto", "auto"),
        ("AUTO", "auto"),
        (12, 12),
        ("7", 7),
    ],
)
def test_request_parsing(raw, want, monkeypatch):
    monkeypatch.delenv(bs.BLOCK_SWAP_ENV, raising = False)
    assert bs.block_swap_request(_cfg(block_swap = raw)) == want


def test_request_env_and_bad_value(monkeypatch):
    monkeypatch.setenv(bs.BLOCK_SWAP_ENV, "auto")
    assert bs.block_swap_request(_cfg(block_swap = "off")) == "auto"
    monkeypatch.delenv(bs.BLOCK_SWAP_ENV)
    with pytest.raises(ValueError):
        bs.block_swap_request(_cfg(block_swap = "lots"))


def test_default_config_is_off(monkeypatch):
    monkeypatch.delenv(bs.BLOCK_SWAP_ENV, raising = False)
    cfg = DiffusionLoraConfig(base_model = "x", data_dir = "d", output_dir = "o")
    assert cfg.block_swap == "off"
    assert bs.block_swap_request(cfg) is None
    assert (
        _config_from_dict(
            {"base_model": "x", "data_dir": "d", "output_dir": "o", "block_swap": "auto"}
        ).block_swap
        == "auto"
    )


def test_gating():
    assert bs.block_swap_error(None, "nf4", "cpu", False) is None
    assert "CUDA" in bs.block_swap_error("auto", "bf16", "cpu", True)
    for mode in ("nf4", "int8", "fp8", "mxfp8"):
        assert "bf16" in bs.block_swap_error("auto", mode, "cuda", True)
    assert "gradient_checkpointing" in bs.block_swap_error(4, "bf16", "cuda", False)
    assert bs.block_swap_error(4, "bf16", "cuda", True) is None
    assert "integrated" in bs.block_swap_error(4, "bf16", "cuda", True, integrated = True)


def test_plan_explicit_count_spreads_over_both_lists():
    plan = bs.plan_swap([[340] * 19, [141] * 38], 21)
    assert sum(len(p) for p in plan) == 21
    assert len(plan[0]) == 7 and len(plan[1]) == 14
    # Spread placement ends at each list's last block, never the first.
    assert plan[0][-1] == 18 and plan[1][-1] == 37 and 0 not in plan[0]
    assert all(len(set(p)) == len(p) for p in plan)


def test_plan_auto_covers_need_net_of_pool():
    sizes = [[340] * 19, [141] * 38]
    assert bs.plan_swap(sizes, "auto", need_bytes = 0) == [[], []]
    for need in (100, 1500, 4000):
        plan = bs.plan_swap(sizes, "auto", need_bytes = need, depth = 2)
        freed = sum(
            sum(s[i] for i in p) - max(s[i] for i in p) * min(3, len(p))
            for s, p in zip(sizes, plan)
            if p
        )
        assert freed >= need
    with pytest.raises(ValueError):
        bs.plan_swap(sizes, "auto", need_bytes = 10**9)


def test_reserve_grows_with_tokens():
    small = bs.activation_reserve_bytes(24 << 30, 1536, 3072, 57)
    big = bs.activation_reserve_bytes(24 << 30, 4608, 3072, 57)
    assert small >= 3 << 30 and big > small


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


class _Model(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(3)])
        self.single = torch.nn.ModuleList([_Block() for _ in range(2)])
        self.other = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


def test_swap_block_lists_finds_every_repeated_list():
    names = [n for n, _ in bs.swap_block_lists(_Model())]
    assert names == ["blocks", "single"]


def test_compile_swapped_blocks_compiles_forward_not_call(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    m = _Model()
    assert bs.compile_swapped_blocks(m, fullgraph = True) == 5
    assert all("forward" in b.__dict__ for b in m.blocks) and calls[0]["fullgraph"] is True


def test_maybe_compile_routes_swap_to_forward_compile(monkeypatch):
    monkeypatch.setattr(dit, "_should_compile", lambda *a, **k: True)
    monkeypatch.setattr(dit, "_compile_toolchain_missing", lambda: None)
    used = []
    monkeypatch.setattr(
        dit, "compile_swapped_blocks", lambda t, fullgraph, dynamic: used.append(fullgraph) or 3
    )
    m = _Model()
    m.compile_repeated_blocks = lambda **kw: used.append("regional")
    cfg = _cfg(compile_transformer = "on")
    assert dit._maybe_compile_transformer(m, cfg, False, "cuda", None, "bf16", 100, block_swap = True)
    assert used == [True]
    assert dit._maybe_compile_transformer(m, cfg, False, "cuda", None, "bf16", 100)
    assert used == [True, "regional"]


def test_block_swap_tokens():
    assert dit._block_swap_tokens(None, _cfg(resolution = 512)) == 32 * 32 + 512


def _auto_cfg(**kw):
    base = dict(base_precision = "auto", gradient_checkpointing = True)
    base.update(kw)
    return _cfg(**base)


def test_auto_upgrades_only_the_int8_band(monkeypatch):
    import importlib.util

    monkeypatch.delenv(dit.AUTO_BLOCK_SWAP_ENV, raising = False)
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(dit, "_device_is_integrated", lambda d: False)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a: object() if n == "unsloth_zoo.block_swap" else real(n, *a),
    )
    assert dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda")
    # nf4 (the base does not fit), bf16 (it fits) and explicit precisions are unchanged.
    for picked in ("nf4", "bf16", "fp8"):
        assert not dit._auto_block_swap_wanted(_auto_cfg(), picked, "cuda")
    assert not dit._auto_block_swap_wanted(_auto_cfg(base_precision = "int8"), "int8", "cuda")
    assert not dit._auto_block_swap_wanted(_auto_cfg(gradient_checkpointing = False), "int8", "cuda")
    assert not dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cpu")
    monkeypatch.setenv(dit.AUTO_BLOCK_SWAP_ENV, "0")
    assert not dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda")
    monkeypatch.delenv(dit.AUTO_BLOCK_SWAP_ENV)
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: True)
    assert not dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda")
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(dit, "_device_is_integrated", lambda d: True)
    assert not dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda")


def test_auto_keeps_int8_without_zoo_block_swap(monkeypatch):
    import importlib.util

    monkeypatch.delenv(dit.AUTO_BLOCK_SWAP_ENV, raising = False)
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(dit, "_device_is_integrated", lambda d: False)

    def missing(n, *a):
        raise ImportError(n)

    monkeypatch.setattr(importlib.util, "find_spec", missing)
    assert not dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda")


@pytest.mark.parametrize("avail_gib, wanted", [(64, True), (16, False), (None, False)])
def test_auto_swap_needs_host_ram_for_the_dense_base(monkeypatch, avail_gib, wanted):
    # The swapped base loads dense on the host (the int8 path it replaces streams): a 12 GB base needs
    # 12 * 1.3 + 4 GiB available, and an unreadable reading keeps int8.
    import types

    import core.inference.diffusion_prequant_cache as pc

    import importlib.util

    monkeypatch.delenv(dit.AUTO_BLOCK_SWAP_ENV, raising = False)
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(dit, "_device_is_integrated", lambda d: False)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a: object() if n == "unsloth_zoo.block_swap" else real(n, *a),
    )
    monkeypatch.setattr(
        pc, "_host_memory_bytes", lambda: (None, None if avail_gib is None else avail_gib << 30)
    )
    monkeypatch.setattr(dit, "_dense_bf16_gb", lambda spec, base: 12.0)
    spec = types.SimpleNamespace(dense_bf16_gb = 12.0)
    assert dit._auto_block_swap_wanted(_auto_cfg(), "int8", "cuda", spec) is wanted
