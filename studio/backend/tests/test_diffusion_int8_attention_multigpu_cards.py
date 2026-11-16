# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention on every card of a multi-GPU load: a CFG-parallel replica or a layer split runs the strided
processors on the selection's second card, where the op runs the kernel only once that card passed its own check."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_multigpu as mg


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in ("_VERDICTS", "_VERIFIED", "_BROKEN", "_CUDA_FAILED"):
        monkeypatch.setattr(I, name, type(getattr(I, name))())
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    yield


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_cudnn"


class _Attention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _two_cards(monkeypatch, secondaries = (1,)):
    """Two fake sm_89 cards and a selection of the load's card 0 plus ``secondaries``; ``_check`` records each card."""
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(
        torch.cuda, "device", lambda *a, **k: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(
        mg,
        "current_selection",
        lambda: mg.MultiGpuSelection(primary = 0, secondaries = tuple(secondaries)),
    )
    checked = []

    def check(
        device,
        dtype,
        heads,
        head_dim,
        incumbent,
        timed,
        impl = "cuda",
    ):
        checked.append((torch.device(device).index, timed))
        I._VERIFIED.add((torch.device(device).index, int(head_dim), str(dtype), impl))
        return {"ok": True, "ms": {}, "min_tokens": 0, "impl": impl}

    monkeypatch.setattr(I, "_check", check)
    return checked


def _qkv(index: int):
    t = types.SimpleNamespace(
        device = torch.device("cuda", index), shape = (1, 4096, 2, 64), dtype = torch.bfloat16
    )
    return t, t, t


def test_strided_install_checks_every_card_of_the_selection(monkeypatch):
    checked = _two_cards(monkeypatch)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, 64)])
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 1
    assert {c for c, _ in checked} == {0, 1}
    # the replica / second-card blocks run the kernel, not native attention
    assert I._verified(*_qkv(0)) and I._verified(*_qkv(1))
    # the race stays the load card's: the second card only takes the accuracy check
    assert all(not timed for c, timed in checked if c == 1)


def test_strided_install_without_a_selection_checks_only_the_load_card(monkeypatch):
    checked = _two_cards(monkeypatch)
    monkeypatch.setattr(mg, "current_selection", lambda: None)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, 64)])
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 1
    assert {c for c, _ in checked} == {0}
    assert not I._verified(*_qkv(1))


def test_strided_install_skips_secondaries_when_the_load_card_fails(monkeypatch):
    checked = _two_cards(monkeypatch)
    monkeypatch.setattr(
        I,
        "_check",
        lambda device, *a, **k: checked.append((torch.device(device).index, None))
        or {"ok": False, "ms": {}, "min_tokens": None, "impl": "cuda"},
    )
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, 64)])
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 0
    assert {c for c, _ in checked} == {0}


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason = "needs two CUDA cards",
)
def test_real_two_cards_strided_install_verifies_card_1(monkeypatch):
    if I.implementation(0, 64) is None or I.implementation(1, 64) is None:
        pytest.skip("no INT8 attention kernel config for these cards")
    monkeypatch.setattr(
        mg,
        "current_selection",
        lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,)),
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, 64)])
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = dtype, force = True) == 1
    q, k, v = I.probe_inputs("cuda:1", dtype, 2048, 2, 64)
    assert I._verified(q, k, v)
    before = I.STATS["int8"]
    with torch.cuda.device(1):
        out = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
    assert I.STATS["int8"] == before + 1 and out.device == q.device
    assert bool(torch.isfinite(out).all())


class _Router:
    """What an armed CFG-parallel router leaves on its denoiser (``_unsloth_cfg_router``)."""

    disabled_reason = None
    secondary = torch.device("cuda", 1)


def test_strided_install_outside_a_load_checks_the_cfg_replica_card(monkeypatch):
    """The deferred speed profile re-installs attention at the 3rd image, outside the load's selection scope: the card
    an armed CFG-parallel replica runs on still gets its check, so the replica does not fall back to native."""
    checked = _two_cards(monkeypatch)
    monkeypatch.setattr(mg, "current_selection", lambda: None)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, 64)])
    dit._unsloth_cfg_router = _Router()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 1
    assert {c for c, _ in checked} == {0, 1}
    assert I._verified(*_qkv(1))
    # a disabled router runs no replica: nothing to check there
    checked.clear()
    I._VERDICTS.clear()
    dit._unsloth_cfg_router.disabled_reason = "uninstalled"
    I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True)
    assert {c for c, _ in checked} == {0}


def test_dispatch_route_outside_a_load_keeps_the_cfg_replica_card(monkeypatch):
    pytest.importorskip("diffusers.models.attention_dispatch")
    from diffusers import WanTransformer3DModel

    from core.inference import diffusion_int8_attention_dispatch as D

    _two_cards(monkeypatch)
    monkeypatch.setattr(mg, "current_selection", lambda: None)
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: False)
    asked = []

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        asked.append(torch.device(device).index)
        return {"impl": "cuda", "ok": True, "min_tokens": 0}

    monkeypatch.setattr(I, "verdict", verdict)
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    torch.manual_seed(0)
    dit = WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 64,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 32,
        ffn_dim = 64,
        num_layers = 1,
        rope_max_seq_len = 32,
    ).to(torch.bfloat16)
    dit.set_attention_backend("native")
    dit._unsloth_cfg_router = _Router()
    pipe = types.SimpleNamespace(transformer = dit)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    assert D.install(pipe, target) == 2
    assert set(asked) == {0, 1}
    routes = [m.processor._attention_backend for m in D.routable_modules(dit)]
    assert all(set(r.cards) == {0, 1} for r in routes)
    D.remove(pipe)
