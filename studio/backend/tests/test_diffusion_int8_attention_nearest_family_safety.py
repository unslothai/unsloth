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
    _unsloth_int8_cards: dict = {}
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


def _serves(
    proc,
    card: int,
    tokens: int = 4096,
) -> bool:
    """Whether the processor routes a call of ``tokens`` tokens on ``card`` to the kernel (its own floors)."""
    query = types.SimpleNamespace(shape = (1, tokens, 2, 64), device = torch.device("cuda", card))
    return I.takes_int8(proc, query)


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


# ---------------------------------------------------------- 1. a secondary probe out of memory declines that card only
@pytest.mark.parametrize("auto_floor", [True, False])
def test_secondary_probe_oom_declines_only_that_card(monkeypatch, auto_floor):
    """An OOM in the secondary's probe keeps that card on the incumbent; the load card's passing route installs (a
    nonzero count, so H3 video also installs its exact start)."""
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
    assert n == 2
    assert _enabled(dit) == [True, True]
    for m in dit.blocks:
        assert _serves(m.processor, 0) and not _serves(m.processor, 1)


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
    for m in dit.blocks:
        assert _serves(m.processor, 0)
        assert not _serves(m.processor, 1), "INT8 served on a card that lost its race"


def test_auto_secondary_keeps_its_own_floor(monkeypatch):
    _mixed_cards(monkeypatch)

    def floor(dev, dt, heads, hd, fn, name):
        card = torch.device(dev).index
        I._VERIFIED.add((card, int(hd), str(dt), "cuda"))
        return {"ok": True, "ms": {}, "impl": "cuda"}, (1024 if card == 0 else 8192)

    monkeypatch.setattr(I, "verdict", _accepting_verdict)
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor) == 2
    proc = dit.blocks[0].processor
    assert not _serves(proc, 1, 4096)  # under the H100's own floor
    assert _serves(proc, 1, 8192)
    assert _serves(proc, 0, 4096)


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
    assert _serves(dit.blocks[0].processor, 1, 1)


def test_forced_install_keeps_a_secondary_that_ran_out_of_memory_on_its_incumbent(monkeypatch):
    """On (forced): a secondary whose probe verified the kernel and then ran out of memory (the long check) declines;
    the op's verification alone must not serve it."""
    _mixed_cards(monkeypatch)

    def probe(device, dtype, heads, hd, *a, **k):
        card = torch.device(device).index
        I._VERIFIED.add((card, int(hd), str(dtype), "cuda"))
        if card == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (secondary long check)")
        return {"ok": True, "ms": {}, "min_tokens": 0, "impl": "cuda"}

    monkeypatch.setattr(I, "verdict", probe)
    dit = _dit()
    assert I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, force = True) == 2
    for m in dit.blocks:
        assert _serves(m.processor, 0, 1)
        assert not _serves(m.processor, 1), "INT8 served on a forced secondary that declined"


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
    dit = _losing_secondary_install(monkeypatch)
    monkeypatch.setattr(I, "_run", lambda *a, **k: "kernel ran")
    monkeypatch.setattr(I, "_fallback_card", lambda *a: False)  # shim operands
    assert not _serves(dit.blocks[0].processor, 1)  # serving still declines on the losing card
    assert I.unsupported_reason(*_fake_qkv(1)) is None  # the kernel check itself is never gated
    assert I.int8_attention(*_fake_qkv(1)) == "kernel ran"
    assert (1, 64) not in I._CUDA_FAILED
    assert I.implementation(1, 64) == "cuda"


# ----------------------------------------------- 7. a declining secondary keeps its pinned 16-bit incumbent
def test_a_losing_secondary_keeps_its_processors_own_16_bit_path(monkeypatch):
    """A card that lost its race never reaches the op: the processor's own check routes the call to its 16-bit path
    (its pinned backend, the incumbent the card raced against), never torch's unpinned attention inside the op."""
    dit = _losing_secondary_install(monkeypatch)
    for m in dit.blocks:
        assert not _serves(m.processor, 1, 1 << 20)
        assert _serves(m.processor, 0)


def test_both_strided_processors_route_through_their_own_card_floors():
    """FLUX.3's and H3's strided processors decide INT8 per call with the processor's own floors on the call's card."""
    import ast
    from pathlib import Path

    root = Path(I.__file__).parent
    for name, marker in (
        ("diffusion_flux3_attn.py", "card_floor"),
        ("video_minimax_h3_attn.py", "takes_int8"),
    ):
        src = (root / name).read_text(encoding = "utf-8")
        calls = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == marker
        ]
        assert calls, f"{name} does not route INT8 through {marker}"


def test_h3_exact_start_keeps_its_pinned_kernel_when_the_card_declines(monkeypatch):
    from core.inference import video_minimax_h3_attn as h3

    _losing_secondary_install(monkeypatch)
    I._BROKEN.add((1, 64))  # the kernel failed on this card: the op declines the call
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


# ------------------------------------------ 9. a fallback card gives SDPA's output wherever an input is non-finite
def _fallback_impl(
    monkeypatch,
    card: int = 1,
    capturing: bool = False,
):
    """int8_attention_impl on CPU operands as if ``card`` (1: the H100, served by the fallback) took the call; the
    kernel returns ones (finite whatever the inputs, as its quantisers make them)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _mixed_cards(monkeypatch)
    kernel_out = torch.ones(1, 16, 2, 64)
    monkeypatch.setattr(I, "_call_route", lambda q, k, v: (None, card, "cuda", {"bn": 64}))
    monkeypatch.setattr(I, "_verified_at", lambda *a: True)
    monkeypatch.setattr(I, "_cuda_attention", lambda *a, **k: kernel_out.clone())
    monkeypatch.setattr(I, "_capturing_on", lambda device: capturing, raising = False)

    def run(q, k, v):
        with sdpa_kernel(SDPBackend.MATH):
            got = I.int8_attention_impl(q, k, v)
            want = I.native_attention(q, k, v)
        return got, want, kernel_out

    return run


def _qkv_with(case: str):
    torch.manual_seed(0)
    q, k, v = (
        torch.rand(1, 16, 2, 64) + 0.1 for _ in range(3)
    )  # q > 0: a -inf key component is a -inf score
    if case == "-inf k component":
        k[0, 3, 1, 5] = -float("inf")
    elif case == "nan k":
        k[0, 3, 1, 5] = float("nan")
    elif case == "nan v":
        v[0, 7, 0, 9] = float("nan")
    elif case == "nan q":
        q[0, 2, 1, 0] = float("nan")
    elif case == "+inf k":
        k[0, 3, 1, 5] = float("inf")
    elif case == "inf v":
        v[0, 7, 0, 9] = float("inf")
    elif case == "all -inf row":
        k[0, :, 1, 5] = -float("inf")  # every score of head 1 is -inf: SDPA's rows there are NaN
    return q, k, v


@pytest.mark.parametrize(
    "case",
    ["-inf k component", "nan k", "nan v", "nan q", "+inf k", "inf v", "all -inf row"],
)
def test_a_fallback_card_matches_math_sdpa_on_non_finite_inputs(monkeypatch, case):
    """Where any input is non-finite, a fallback card returns torch's (math) SDPA output exactly: non-finite where
    SDPA is, finite where SDPA masks a -inf score. The kernel's quantisers cannot give that."""
    got, want, _ = _fallback_impl(monkeypatch)(*_qkv_with(case))
    assert torch.equal(torch.isfinite(got), torch.isfinite(want)), case
    assert torch.equal(torch.nan_to_num(got), torch.nan_to_num(want)), case
    if case == "-inf k component":
        assert torch.isfinite(got).all()


def test_a_fallback_card_keeps_the_kernel_on_finite_inputs(monkeypatch):
    got, _, kernel_out = _fallback_impl(monkeypatch)(*_qkv_with("finite"))
    assert torch.equal(got, kernel_out)


def test_a_fallback_card_runs_sdpa_while_a_graph_records(monkeypatch):
    """The finite check reads a flag on the host, which a CUDA graph cannot record: a recorded call runs SDPA."""
    got, want, _ = _fallback_impl(monkeypatch, capturing = True)(*_qkv_with("finite"))
    assert torch.equal(got, want)


def test_the_unguarded_entry_runs_sdpa_while_a_graph_records_on_a_fallback_card(monkeypatch):
    """``int8_attention`` on a fallback card never reads the finite check on the host while a CUDA graph records."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _mixed_cards(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(I, "unsupported_reason", lambda q, k, v: None)
    monkeypatch.setattr(I, "implementation", lambda index, hd: "cuda")
    monkeypatch.setattr(I, "_fallback_card", lambda index, hd, impl: True)
    monkeypatch.setattr(I, "_capturing_on", lambda device: True, raising = False)

    def host_read(*a):
        raise AssertionError("host read of the finite check while a graph records")

    monkeypatch.setattr(I, "_nonfinite_inputs", host_read)
    monkeypatch.setattr(I, "_run", lambda *a, **k: torch.ones(1, 16, 2, 64))
    q, k, v = _qkv_with("finite")
    with sdpa_kernel(SDPBackend.MATH):
        assert torch.equal(I.int8_attention(q, k, v), I.native_attention(q, k, v))


def test_a_listed_card_output_is_untouched_by_the_non_finite_check(monkeypatch):
    got, _, kernel_out = _fallback_impl(monkeypatch, card = 0)(*_qkv_with("nan k"))
    assert torch.equal(got, kernel_out)  # sm89 is listed: as before


# ---------------------------------------------------- 10. the two-card GPU tests see newly eligible secondaries
def test_two_card_gpu_tests_look_up_the_served_config():
    import ast
    from pathlib import Path

    src = Path(__file__).with_name("test_diffusion_int8_attention.py").read_text(encoding = "utf-8")
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


# ------------------------------- 11. a secondary's race result belongs to its processor (route), not to a geometry
def _named_install(
    monkeypatch,
    names: list,
    oom_on_secondary: bool = False,
):
    """One DiT whose processors pin backends named in ``names``: on cuda:1 a "winner" incumbent loses to the kernel
    and a "loser" one beats it (same heads, head dim and dtype)."""

    def floor(dev, dt, heads, hd, fn, name):
        card = torch.device(dev).index
        I._VERIFIED.add((card, int(hd), str(dt), "cuda"))
        if card == 1 and oom_on_secondary:
            raise torch.OutOfMemoryError("synthetic OOM in the secondary's timed race")
        return {"ok": True, "ms": {}, "impl": "cuda"}, (
            1024 if card == 0 or name == "winner" else None
        )

    monkeypatch.setattr(I, "verdict", _accepting_verdict)
    monkeypatch.setattr(I, "_sdpa_incumbent", lambda backend: (backend, lambda *a: None))
    dit = _dit(len(names))
    for m, name in zip(dit.blocks, names):
        m.processor._attention_backend = name
    count = I.install_int8_attention(dit, device = "cuda:0", dtype = torch.bfloat16, floor = floor)
    return dit, count


def test_processors_with_opposite_secondary_results_stay_independent(monkeypatch):
    _mixed_cards(monkeypatch)
    first, _ = _named_install(monkeypatch, ["loser", "winner"])
    second, _ = _named_install(monkeypatch, ["winner", "loser"])
    for dit in (first, second):
        for m in dit.blocks:
            assert _serves(m.processor, 1) == (m.processor._attention_backend == "winner")
    # another load's plan reset never re-enables a card whose race lost on a still-installed processor
    I.forget_plans()
    for dit in (first, second):
        for m in dit.blocks:
            assert _serves(m.processor, 1) == (m.processor._attention_backend == "winner")


def test_a_secondary_timed_race_oom_keeps_the_primary_installed(monkeypatch):
    """Two listed cards (sm89 load card, sm80 secondary): an OOM only in the secondary's own timed race declines that
    card; the load card's route installs, as it did before secondary races existed."""
    monkeypatch.setitem(CAPS, 1, (8, 0))
    _mixed_cards(monkeypatch)
    dit, count = _named_install(monkeypatch, ["winner"], oom_on_secondary = True)
    assert count == 1
    proc = dit.blocks[0].processor
    assert proc._unsloth_int8 and _serves(proc, 0) and not _serves(proc, 1)


# ------------------------------------------- 12. the sink race times what a fallback card serves (its input check)
@pytest.mark.parametrize("fallback", [True, False])
def test_the_sink_race_times_the_serving_check_on_a_fallback_card(monkeypatch, fallback):
    checks = []
    monkeypatch.setattr(I, "_raw_stream", lambda index: None)
    monkeypatch.setattr(I, "_launch_quantisers", lambda *a, **k: None)
    monkeypatch.setattr(I, "_launch_attention", lambda *a, **k: None)
    monkeypatch.setattr(I, "_fallback_card", lambda *a: fallback)
    monkeypatch.setattr(
        I, "_nonfinite_inputs", lambda q, k, v: checks.append(1) or False, raising = False
    )
    timed = {}

    def median(fns, rounds = 5):
        for name, fn in fns.items():
            before = len(checks)
            fn()
            timed[name] = len(checks) - before
        return {name: 1.0 for name in fns}

    monkeypatch.setattr(I, "_median_ms", median)
    q, k, v = (torch.rand(1, 16, 2, 64) for _ in range(3))
    plan = {"bytes": 16, "out_shape": (1, 16, 2, 64), "shape": (1, 2, 16, 2)}
    I._sink_race(plan, q, k, v, None, 1)
    expect = 1 if fallback else 0
    assert timed == {"incumbent": 0, "sinks": expect, "plain": expect}, timed
