# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The nearest-family fallback makes newer cards (sm90, sm100, ...) INT8 attention candidates, also as the secondary
card of a multi-GPU load. A candidate card must never serve the kernel on less evidence than a listed one: every
card probes before any processor is enabled (a failed probe leaves nothing half-installed), every card that serves
the kernel under Auto has its own race, the fallback is offered only for an arch the installed NVRTC can target,
and a listed card reads its device facts once. CPU only."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as C
from core.inference import diffusion_multigpu as mg

CAPS = {0: (8, 9), 1: (9, 0)}  # an Ada load card and a newly eligible H100 secondary


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in ("_VERDICTS", "_VERIFIED", "_BROKEN", "_CUDA_FAILED"):
        monkeypatch.setattr(I, name, type(getattr(I, name))())
    if hasattr(I, "_CARD_FLOORS"):
        monkeypatch.setattr(I, "_CARD_FLOORS", {})
    if hasattr(I, "reset_device_facts"):
        I.reset_device_facts()
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    yield
    if hasattr(I, "reset_device_facts"):
        I.reset_device_facts()


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_cudnn"


class _Attention(torch.nn.Module):
    def __init__(
        self,
        heads: int = 2,
        head_dim: int = 64,
    ):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _dit(n: int = 2) -> torch.nn.Module:
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention() for _ in range(n)])
    return dit


def _mixed_cards(monkeypatch):
    """cuda:0 sm89 + cuda:1 sm90 in one selection; the real implementation() / fallback decide what serves."""
    from core.inference import rocm_bf16

    def cap(
        device = None,
        *a,
        **k,
    ):
        index = (
            0
            if device is None
            else (device if isinstance(device, int) else torch.device(device).index or 0)
        )
        return CAPS[int(index)]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", cap)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(shared_memory_per_block_optin = 227 * 1024),
    )
    monkeypatch.setattr(
        torch.cuda, "device", lambda *a, **k: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(C, "nvrtc_arch_supported", lambda capability: True, raising = False)
    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    )
    assert I.implementation(1, 64) == "cuda"  # the fallback makes the H100 a candidate


def _enabled(dit) -> list:
    return [bool(m.processor._unsloth_int8) for m in dit.blocks]


def _fake_qkv(index: int, tokens: int = 4096):
    t = torch.empty(1, tokens, 2, 64, dtype = torch.bfloat16)
    shim = types.SimpleNamespace(
        is_cuda = True,
        device = torch.device("cuda", index),
        dim = t.dim,
        shape = t.shape,
        dtype = t.dtype,
        requires_grad = False,
    )
    return shim, shim, shim


# ------------------------------------------------------------------------------ 1. a failed probe leaves nothing on
@pytest.mark.parametrize("auto_floor", [True, False])
def test_secondary_probe_oom_leaves_no_processor_on_int8(monkeypatch, auto_floor):
    """H3 video installs its four-step 16-bit exact start only after a nonzero install; an OOM in the secondary's
    probe must not leave the primary's processors on INT8 from step 0 while reporting 0."""
    _mixed_cards(monkeypatch)

    def probe(device, dtype, heads, head_dim, *a, **k):
        if torch.device(device).index == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (secondary probe)")
        I._VERIFIED.add((0, int(head_dim), str(dtype), "cuda"))
        return {"ok": True, "ms": {}, "min_tokens": 0, "impl": "cuda"}

    monkeypatch.setattr(I, "verdict", probe)
    floor = (
        (lambda dev, dt, heads, hd, fn, name: (probe(dev, dt, heads, hd), 1024))
        if auto_floor
        else None
    )
    dit = _dit()
    n = I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor)
    assert n == 0
    assert _enabled(dit) == [False, False], "INT8 left enabled after a failed install"


# -------------------------------------------------------------------------- 2. every serving card has its own race
def _accepting_verdict(device, dtype, heads, hd, *a, **k):
    """The accuracy check passes on every card (the race is the floor's)."""
    I._VERIFIED.add((torch.device(device).index, int(hd), str(dtype), "cuda"))
    return {"ok": True, "ms": {}, "min_tokens": 0, "impl": "cuda"}


def test_auto_races_the_secondary_and_keeps_it_native_when_it_loses(monkeypatch):
    _mixed_cards(monkeypatch)
    raced = []

    def floor(dev, dt, heads, hd, fn, name):
        card = torch.device(dev).index
        raced.append(card)
        I._VERIFIED.add((card, int(hd), str(dt), "cuda"))
        found = {"ok": True, "ms": {}, "impl": "cuda", "min_tokens": 0 if card == 0 else None}
        return found, (1024 if card == 0 else None)  # the H100 loses its race

    monkeypatch.setattr(I, "verdict", _accepting_verdict)
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor) == 2
    assert sorted(set(raced)) == [0, 1], "the secondary never ran its own race"
    assert I.unsupported_reason(*_fake_qkv(0)) is None
    assert (
        I.unsupported_reason(*_fake_qkv(1)) is not None
    ), "INT8 served on a card that lost its race"


def test_auto_secondary_keeps_its_own_floor(monkeypatch):
    _mixed_cards(monkeypatch)

    def floor(dev, dt, heads, hd, fn, name):
        card = torch.device(dev).index
        I._VERIFIED.add((card, int(hd), str(dt), "cuda"))
        return {"ok": True, "ms": {}, "impl": "cuda"}, (1024 if card == 0 else 8192)

    monkeypatch.setattr(I, "verdict", _accepting_verdict)
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor) == 2
    assert I.unsupported_reason(*_fake_qkv(1, 4096)) is not None  # under the H100's own floor
    assert I.unsupported_reason(*_fake_qkv(1, 8192)) is None
    assert I.unsupported_reason(*_fake_qkv(0, 4096)) is None


def test_forced_install_keeps_the_accuracy_only_secondary_check(monkeypatch):
    """On (forced): no race anywhere, the op serves every card whose accuracy check passed, as before."""
    _mixed_cards(monkeypatch)
    timed = []

    def probe(
        device,
        dtype,
        heads,
        hd,
        fn = None,
        name = "native",
        timed_ = None,
        **k,
    ):
        timed.append((torch.device(device).index, k.get("timed")))
        I._VERIFIED.add((torch.device(device).index, int(hd), str(dtype), "cuda"))
        return {"ok": True, "ms": {}, "min_tokens": 0, "impl": "cuda"}

    monkeypatch.setattr(I, "verdict", probe)
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 2
    assert all(not t for card, t in timed if card == 1)
    assert I.unsupported_reason(*_fake_qkv(1)) is None


# --------------------------------------------------------------------------------------- 3. the NVRTC toolchain gate
_NVRTC_12_6 = {75, 80, 86, 87, 89, 90}
_NVRTC_13_0 = {75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121}


@pytest.mark.parametrize(
    "archs,cap,served",
    [
        (_NVRTC_12_6, (9, 0), True),
        (_NVRTC_12_6, (10, 0), False),
        (_NVRTC_12_6, (12, 1), False),
        (_NVRTC_13_0, (10, 0), True),
        (_NVRTC_13_0, (12, 1), True),
        (_NVRTC_13_0, (13, 0), False),
        # a listed card is never gated: it ran on every toolchain before the fallback existed
        (set(), (8, 9), True),
        (set(), (12, 0), True),
    ],
)
def test_fallback_only_for_an_arch_nvrtc_can_target(monkeypatch, archs, cap, served):
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(shared_memory_per_block_optin = 227 * 1024),
    )
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(C, "_nvrtc_supported_archs", lambda: frozenset(archs), raising = False)
    if hasattr(I, "reset_device_facts"):
        I.reset_device_facts()
    assert C.nvrtc_arch_supported(cap) == (cap[0] * 10 + cap[1] in archs)
    assert (I.device_config(0, 128, "cuda") is not None) == served
    assert (I.implementation(0, 128) == "cuda") == served


def test_unreadable_nvrtc_blocks_only_the_fallback(monkeypatch):
    monkeypatch.setattr(C, "_nvrtc_supported_archs", lambda: None, raising = False)
    assert C.nvrtc_arch_supported((9, 0)) is False
    assert I.nearest_kernel_config((8, 9), 128)[1] == I.KERNEL_CONFIGS[(8, 9)][128]


# -------------------------------------------------------------------------- 4. GPU tests see the fallback cards
def test_cuda_config_helper_follows_the_fallback(monkeypatch):
    import importlib.util
    from pathlib import Path

    from core.inference import rocm_bf16

    spec = importlib.util.spec_from_file_location(
        "_int8_attention_suite", Path(__file__).with_name("test_diffusion_int8_attention.py")
    )
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(shared_memory_per_block_optin = 227 * 1024),
    )
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(C, "nvrtc_arch_supported", lambda capability: True, raising = False)
    assert T._cuda_config(128) == I.KERNEL_CONFIGS[(8, 9)][128]


# ------------------------------------------------------------------------------ 5. device facts read once per card
def test_a_listed_card_reads_its_device_facts_once(monkeypatch):
    from core.inference import rocm_bf16

    reads = {"cap": 0, "props": 0}

    def cap(*a, **k):
        reads["cap"] += 1
        return (8, 9)

    def props(*a, **k):
        reads["props"] += 1
        return types.SimpleNamespace(shared_memory_per_block_optin = 99 * 1024)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", cap)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    if hasattr(I, "reset_device_facts"):
        I.reset_device_facts()
    for _ in range(5):
        assert I.implementation(0, 128) == "cuda"
        assert I.device_config(0, 128) == I.KERNEL_CONFIGS[(8, 9)][128]
    assert reads == {"cap": 1, "props": 0}, reads


# ------------------------------------------------------- 6. a secondary's race loss gates serving, never validation
def _losing_secondary_install(monkeypatch, incumbent = None):
    """Install with the H100 secondary losing its own race; returns the DiT."""
    _mixed_cards(monkeypatch)

    def floor(dev, dt, heads, hd, fn, name):
        card = torch.device(dev).index
        I._VERIFIED.add((card, int(hd), str(dt), "cuda"))
        return {"ok": True, "ms": {}, "impl": "cuda"}, (1024 if card == 0 else None)

    monkeypatch.setattr(I, "verdict", _accepting_verdict)
    if incumbent is not None:
        monkeypatch.setattr(I, "_sdpa_incumbent", lambda backend: ("_native_cudnn", incumbent))
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor) == 2
    return dit


def test_a_secondary_race_loss_never_fails_that_cards_kernel_check(monkeypatch):
    """The probe and the races call int8_attention: a serving floor there would raise inside the accuracy check,
    and verdict() would record the card's kernel as failed for the process."""
    _losing_secondary_install(monkeypatch)
    monkeypatch.setattr(I, "_run", lambda *a, **k: "kernel ran")
    monkeypatch.setattr(I, "_keep_nonfinite", lambda q, k, v, out: out)  # shim operands
    assert (
        I.unsupported_reason(*_fake_qkv(1)) is not None
    )  # serving still declines on the losing card
    assert I.int8_attention(*_fake_qkv(1)) == "kernel ran"
    assert (1, 64) not in I._CUDA_FAILED
    assert I.implementation(1, 64) == "cuda"


# ----------------------------------------------- 7. a declining secondary keeps its pinned 16-bit incumbent
def test_a_losing_secondary_runs_its_pinned_incumbent_not_unpinned_sdpa(monkeypatch):
    calls = []

    def pinned(q, k, v):
        calls.append("pinned")
        return torch.zeros(1, 4096, 2, 64, dtype = torch.bfloat16)

    _losing_secondary_install(monkeypatch, incumbent = pinned)
    monkeypatch.setattr(
        I, "native_attention", lambda *a, **k: calls.append("sdpa") or torch.zeros(1, 4096, 2, 64)
    )
    out = I.int8_attention_impl(*_fake_qkv(1))
    assert calls == ["pinned"], calls
    assert out.dtype == torch.bfloat16


def test_h3_exact_start_keeps_its_pinned_kernel_when_the_card_declines(monkeypatch):
    from core.inference import video_minimax_h3_attn as h3

    _losing_secondary_install(monkeypatch)
    ran = []
    monkeypatch.setattr(h3, "_capturing_on", lambda device: False)
    monkeypatch.setattr(
        h3,
        "_exact_attention",
        lambda q, k, v, kernel: ran.append(kernel)
        or torch.zeros(1, 4096, 2, 64, dtype = torch.bfloat16),
    )
    monkeypatch.setattr(
        I, "native_attention", lambda *a, **k: ran.append("sdpa") or torch.zeros(1, 4096, 2, 64)
    )
    before = dict(h3.EXACT_START_STATS)
    h3.int8_attention_exact_start_impl(*_fake_qkv(1), torch.zeros(1, dtype = torch.int32), 1)
    assert ran == [1], ran  # the pinned kernel after the boundary, not unrestricted SDPA
    assert h3.EXACT_START_STATS["int8"] == before["int8"], "a declined call counted as an INT8 call"


# ------------------------------------------------- 8. the native fallback keeps V's dtype whatever autocast says
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_fallback_dtype_matches_the_fake_under_autocast(dtype):
    q, k, v = (torch.randn(1, 16, 2, 64).to(dtype) for _ in range(3))
    fake = torch.empty(0, dtype = dtype).new_empty(q.shape)  # the op's fake: v.new_empty(q.shape)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        eager = I.int8_attention_impl(q, k, v)
    assert eager.dtype == fake.dtype == dtype


# ------------------------------------------------ 9. non-finite inputs stay non-finite on a fallback card
@pytest.mark.parametrize(
    "operand,where,spread",
    [("k", (0, 3, 1, 5), "head"), ("v", (0, 7, 0, 9), "channel"), ("q", (0, 2, 1, 0), "row")],
)
def test_a_fallback_card_propagates_non_finite_inputs(monkeypatch, operand, where, spread):
    """torch's attention, which this card ran before the fallback, returns non-finite where an input is non-finite;
    the kernel's quantisers would drop the NaN. Finite rows stay exactly the kernel's."""
    _mixed_cards(monkeypatch)
    qkv = {n: torch.randn(1, 16, 2, 64, dtype = torch.bfloat16) for n in "qkv"}
    qkv[operand][where] = float("nan")
    kernel_out = torch.ones(1, 16, 2, 64, dtype = torch.bfloat16)
    monkeypatch.setattr(
        I, "_call_route", lambda q, k, v, floors = True: (None, 1, "cuda", {"bn": 64})
    )
    monkeypatch.setattr(I, "_verified_at", lambda *a: True)
    monkeypatch.setattr(I, "_cuda_attention", lambda *a, **k: kernel_out.clone())
    out = I.int8_attention_impl(qkv["q"], qkv["k"], qkv["v"])
    bad = torch.isnan(out)
    b, s, h, d = where
    expect = torch.zeros_like(bad)
    if spread == "head":
        expect[b, :, h, :] = True
    elif spread == "channel":
        expect[b, :, h, d] = True
    else:
        expect[b, s, h, :] = True
    assert torch.equal(bad, expect)
    assert torch.equal(out[~bad], kernel_out[~bad])


def test_a_listed_card_output_is_untouched_by_the_non_finite_guard(monkeypatch):
    _mixed_cards(monkeypatch)
    q, k, v = (torch.randn(1, 16, 2, 64, dtype = torch.bfloat16) for _ in range(3))
    k[0, 3, 1, 5] = float("nan")
    kernel_out = torch.ones(1, 16, 2, 64, dtype = torch.bfloat16)
    monkeypatch.setattr(
        I, "_call_route", lambda q, k, v, floors = True: (None, 0, "cuda", {"bn": 64})
    )
    monkeypatch.setattr(I, "_verified_at", lambda *a: True)
    monkeypatch.setattr(I, "_cuda_attention", lambda *a, **k: kernel_out.clone())
    assert torch.equal(I.int8_attention_impl(q, k, v), kernel_out)  # sm89 is listed: as before


# ---------------------------------------------------- 10. the two-card GPU tests see newly eligible secondaries
def test_two_card_gpu_tests_look_up_the_served_config():
    import ast
    from pathlib import Path

    src = Path(__file__).with_name("test_diffusion_int8_attention.py").read_text()
    tree = ast.parse(src)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "needs_two_cards" for t in node.targets
        ):
            segments.append(ast.get_source_segment(src, node))
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "test_each_card_loads_its_own_kernels"
        ):
            segments.append(ast.get_source_segment(src, node))
    assert len(segments) == 2
    for seg in segments:
        assert "device_config(" in seg and "kernel_config(" not in seg, seg[:200]
