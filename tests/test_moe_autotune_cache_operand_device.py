# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE kernel configs are looked up, refit and keyed for the operand's card, not the current one.

On a mixed host (current cuda:0 an A100, the MoE layer's tensors on an L4 at cuda:1; accelerate moves
inputs without switching the current device) a disk entry was refit against the A100's 166,912 B and
kept bf16 dW (128, 256, 256, 8 warps, 3 stages), which needs 132,096 B on sm_89 (Triton cross-compile)
over the L4's 101,376 B. The card is emulated per device index: no GPU needed."""

import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

CARDS = {0: ((8, 0), 166912), 1: ((8, 9), 101376)}  # A100 current, L4 operand
STALE = {
    "config_fwd": dict(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 128, num_warps = 8, num_stages = 3),
    "config_bwd_dx": dict(BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 256, num_warps = 8, num_stages = 4),
    "config_bwd_dw": dict(BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 256, BLOCK_SIZE_K = 256, num_warps = 8, num_stages = 3),
}


@pytest.fixture
def ac(monkeypatch):
    for name, path in (
        ("unsloth", ROOT / "unsloth"),
        ("unsloth.kernels", ROOT / "unsloth" / "kernels"),
        ("unsloth.kernels.moe", ROOT / "unsloth" / "kernels" / "moe"),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    module.clear_cache()
    yield module
    module.clear_cache()
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


@pytest.fixture
def mixed_host(monkeypatch):
    state = {"current": 0}

    @contextlib.contextmanager
    def device(d):
        index = d.index if isinstance(d, torch.device) else d
        previous = state["current"]
        if index is not None and index >= 0:
            state["current"] = index
        try:
            yield
        finally:
            state["current"] = previous

    def capability(d = None):
        index = state["current"] if d is None else (d.index if isinstance(d, torch.device) else d)
        return CARDS[index][0]

    def properties(index):
        cap, smem = CARDS[index]
        return types.SimpleNamespace(
            shared_memory_per_block_optin = smem, major = cap[0], minor = cap[1]
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    return state


def _lookup(ac, monkeypatch, **kwargs):
    seen = []

    def load(key):
        seen.append(key)
        return {k: dict(v) for k, v in STALE.items()}

    monkeypatch.setattr(ac, "load_cached_config", load)
    configs = ac.get_or_autotune_moe_kernels(
        num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8,
        dtype = torch.bfloat16, **kwargs,
    )
    return configs, seen


def test_disk_entry_is_refit_for_the_operand_card(ac, monkeypatch, mixed_host):
    configs, seen = _lookup(ac, monkeypatch, device = torch.device("cuda", 1))
    for kind, c in zip(("fwd", "dX", "dW"), configs):
        assert ac._estimate_smem(kind, c, 2, 2, (8, 9)) <= 101376, (kind, c)
    # Keyed by the L4's capability, so an A100 entry is never reused for it.
    assert seen == [ac._get_cache_key(128, 2048, 768, 8, torch.bfloat16, (8, 9), 8192)]
    assert mixed_host["current"] == 0  # the current device is restored


def test_without_a_device_the_current_card_is_used(ac, monkeypatch, mixed_host):
    configs, seen = _lookup(ac, monkeypatch)
    assert seen == [ac._get_cache_key(128, 2048, 768, 8, torch.bfloat16, (8, 0), 8192)]
    dw = configs[2]
    assert (dw.BLOCK_SIZE_M, dw.BLOCK_SIZE_N, dw.BLOCK_SIZE_K, dw.num_stages) == (128, 256, 256, 3)


def test_each_card_gets_its_own_in_memory_entry(ac, monkeypatch, mixed_host):
    l4, _ = _lookup(ac, monkeypatch, device = torch.device("cuda", 1))
    a100, _ = _lookup(ac, monkeypatch, device = torch.device("cuda", 0))
    assert ac._estimate_smem("dW", l4[2], 2, 2, (8, 9)) <= 101376
    assert (a100[2].BLOCK_SIZE_M, a100[2].BLOCK_SIZE_N, a100[2].BLOCK_SIZE_K) == (128, 256, 256)


def test_heuristic_configs_fit_the_operand_card(ac, monkeypatch, mixed_host):
    monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", "1")
    configs = ac.get_or_autotune_moe_kernels(
        num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8,
        dtype = torch.float32, weight_dtype = torch.bfloat16, device = torch.device("cuda", 1),
    )
    for kind, c in zip(("fwd", "dX", "dW"), configs):
        assert ac._estimate_smem(kind, c, 4, 2, (8, 9)) <= 101376, (kind, c)
