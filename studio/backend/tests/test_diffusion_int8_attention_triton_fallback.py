# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's INT8 attention on NVIDIA cards where the CUDA kernel cannot be built or loaded: the Triton kernel serves
the same calls (with its own accuracy check), native attention where Triton or a Triton config is missing.

CPU: NVRTC, the module load and the first launch fail on a faked card; ``UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL``,
the kill switch, and that a CUDA verdict never stands in for the Triton path. NVIDIA GPU (sm80+ with Triton): the
forced Triton path against fp32 attention through the registered op."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as K
from core.inference import diffusion_int8_attention_triton as T


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    monkeypatch.setattr(I, "_VERDICTS", {})
    monkeypatch.setattr(I, "_VERIFIED", set())
    monkeypatch.setattr(I, "_CUDA_FAILED", {})
    monkeypatch.setattr(I, "_BROKEN", set())
    monkeypatch.setattr(I, "_LOADED", {})
    monkeypatch.setattr(K, "_FUNCTIONS", {})
    stats = dict(I.STATS)
    yield
    I.STATS.clear()
    I.STATS.update(stats)


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_flash"


class _Attention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _fake_dit(head_dims = (128, 64, 96)):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, d) for d in head_dims])
    return dit


def _fake_nvidia(
    monkeypatch,
    capability = (8, 9),
    triton = True,
):
    """A CPU stand-in for an NVIDIA card: probes and kernels run on CPU tensors. Returns the Triton kernel's calls
    (the fake computes fp32 attention, so its accuracy check passes)."""
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "triton_available", lambda: triton)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "unsupported_reason", lambda q, k, v: None)
    probe = I.probe_inputs
    monkeypatch.setattr(I, "probe_inputs", lambda device, *a, **k: probe("cpu", *a, **k))
    calls = []

    def triton_kernel(
        q,
        k,
        v,
        config = None,
        pv = None,
        scale = None,
    ):
        calls.append((int(q.shape[-1]), dict(config or {})))
        return I.reference_attention(q, k, v).to(v.dtype)

    monkeypatch.setattr(T, "int8_attention", triton_kernel)
    return calls


def _no_nvrtc(monkeypatch):
    """No NVRTC anywhere (a Windows install without the CUDA runtime wheels): every candidate fails to load."""
    monkeypatch.setattr(K, "_NVRTC", None)
    monkeypatch.setattr(K, "_nvrtc_candidates", lambda: ["nvrtc64_missing_0.dll"])


def _module_load_fails(monkeypatch):
    def load_module(*a, **k):
        raise RuntimeError("cuModuleLoadData failed with CUDA error 218")

    monkeypatch.setattr(K, "load_module", load_module)


def _first_launch_fails(monkeypatch):
    monkeypatch.setattr(I, "quantise", lambda *a, **k: None)

    def attend(p):
        raise RuntimeError("cuLaunchKernel(unsloth_i8attn) failed with CUDA error 1")

    monkeypatch.setattr(I, "attend", attend)


FAILURES = {
    "nvrtc": _no_nvrtc,
    "module_load": _module_load_fails,
    "first_launch": _first_launch_fails,
}


@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (12, 0)])
def test_a_cuda_kernel_that_cannot_load_falls_back_to_triton(monkeypatch, failure, capability):
    calls = _fake_nvidia(monkeypatch, capability = capability)
    FAILURES[failure](monkeypatch)
    assert I.implementation(0, 128) == "cuda"
    found = I.verdict("cuda", torch.float32, 4, 128, timed = False)
    assert found["ok"] and found["impl"] == "triton" and found["cos"] >= I.MIN_COSINE
    assert I.implementation(0, 128) == "triton" and (0, 128) in I._CUDA_FAILED
    assert {
        "nvrtc": "NVRTC not found",
        "module_load": "cuModuleLoadData",
        "first_launch": "cuLaunchKernel",
    }[failure] in I._CUDA_FAILED[(0, 128)]
    assert calls == [(128, I.TRITON_KERNEL_CONFIGS[capability][128])] * 2
    assert I._VERIFIED == {(0, 128, "torch.float32", "triton")}
    cuda_key = (0, "torch.float32", 4, 128, "native", False, "cuda")
    assert I._VERDICTS[cuda_key]["failed"] and not I._VERDICTS[cuda_key]["ok"]
    assert I.verdict("cuda", torch.float32, 4, 128, timed = False) is found and len(calls) == 2


def test_the_fallback_engages_on_install_and_serves_the_op(monkeypatch):
    calls = _fake_nvidia(monkeypatch)
    _no_nvrtc(monkeypatch)
    monkeypatch.setenv(
        I.INT8_ATTENTION_ENV, "1"
    )  # unraced: the race is test_a_requested_stand_in_still_has_to_win_its_race
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.float32, force = True) == 2
    assert [m.processor._unsloth_int8 for m in dit.blocks] == [True, True, False]
    assert [getattr(m.processor, "_unsloth_int8_impl", None) for m in dit.blocks[:2]] == [
        "triton"
    ] * 2
    st = I.status(dit)
    assert st["impl"] == ["triton"] and st["modules"] == 2
    assert any("NVRTC not found" in why for why in st["cuda_failed"].values())
    assert all(v["impl"] in ("cuda", "triton") for v in st["verdicts"].values())
    q, k, v = I.probe_inputs("cpu", torch.float32, 40, 2, 128)
    n, before = len(calls), dict(I.STATS)
    out = I.int8_attention_impl(q, k, v)
    assert len(calls) == n + 1 and torch.allclose(out, I.reference_attention(q, k, v))
    assert I.STATS["int8"] == before["int8"] + 1 and I.STATS["triton"] == before["triton"] + 1


@pytest.mark.parametrize("failure", list(FAILURES))
def test_without_triton_a_failed_cuda_kernel_keeps_native(monkeypatch, failure):
    calls = _fake_nvidia(monkeypatch, triton = False)
    FAILURES[failure](monkeypatch)
    found = I.verdict("cuda", torch.float32, 4, 128, timed = False)
    assert not found["ok"] and found["impl"] is None and I.implementation(0, 128) is None
    assert not calls and not I._VERIFIED
    assert (
        I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.float32, force = True) == 0
    )
    q, k, v = I.probe_inputs("cpu", torch.float32, 40, 2, 128)
    before = dict(I.STATS)
    assert torch.equal(I.int8_attention_impl(q, k, v), I.native_attention(q, k, v))
    assert I.STATS["native"] == before["native"] + 1 and I.STATS["int8"] == before["int8"]


def test_a_card_without_a_triton_config_keeps_native(monkeypatch):
    """sm75 has a CUDA config but no Triton one (no int8 tensor-core dot in Triton there): a CUDA failure keeps
    native attention."""
    assert (7, 5) in I.KERNEL_CONFIGS and (7, 5) not in I.TRITON_KERNEL_CONFIGS
    calls = _fake_nvidia(monkeypatch, capability = (7, 5))
    _no_nvrtc(monkeypatch)
    found = I.verdict("cuda", torch.float16, 4, 128, timed = False)
    assert not found["ok"] and found["impl"] is None and "NVRTC not found" in found["error"]
    assert not calls and I.device_config(0, 128) is None


def test_impl_cuda_never_runs_triton(monkeypatch):
    calls = _fake_nvidia(monkeypatch)
    monkeypatch.setenv(I.IMPL_ENV, "cuda")
    _no_nvrtc(monkeypatch)
    found = I.verdict("cuda", torch.float32, 4, 128, timed = False)
    assert not found["ok"] and I.implementation(0, 128) is None and (0, 128) in I._CUDA_FAILED
    assert (
        I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.float32, force = True) == 0
    )
    assert not calls and not I._VERIFIED
    # also with a CUDA kernel that works, and on a card with no CUDA config but a Triton one
    monkeypatch.setattr(I, "_CUDA_FAILED", {})
    assert I.implementation(0, 128) == "cuda"
    monkeypatch.setitem(I.TRITON_KERNEL_CONFIGS, (9, 0), I.TRITON_KERNEL_CONFIGS[(8, 0)])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert I.implementation(0, 128) is None
    monkeypatch.setenv(I.IMPL_ENV, "auto")
    assert I.implementation(0, 128) == "triton"


def test_impl_triton_skips_the_cuda_kernel(monkeypatch):
    calls = _fake_nvidia(monkeypatch)

    def nvrtc():
        raise AssertionError("the CUDA kernel was compiled")

    monkeypatch.setattr(K, "nvrtc", nvrtc)
    for raw in ("triton", " TRITON "):
        monkeypatch.setenv(I.IMPL_ENV, raw)
        assert I.impl_setting() == "triton" and I.implementation(0, 128) == "triton"
    found = I.verdict("cuda", torch.float32, 4, 64, timed = False)
    assert found["ok"] and found["impl"] == "triton" and not I._CUDA_FAILED
    assert calls[-1] == (64, I.TRITON_KERNEL_CONFIGS[(8, 9)][64])
    for raw in ("", "auto", "bogus"):
        monkeypatch.setenv(I.IMPL_ENV, raw)
        assert I.impl_setting() == "auto"


def test_the_kill_switch_still_wins(monkeypatch):
    calls = _fake_nvidia(monkeypatch)
    _no_nvrtc(monkeypatch)
    checked = []
    inner = I.verdict
    monkeypatch.setattr(I, "verdict", lambda *a, **k: checked.append(a) or inner(*a, **k))
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    for impl in ("auto", "triton"):
        monkeypatch.setenv(I.IMPL_ENV, impl)
        dit = _fake_dit()
        assert I.install_int8_attention(dit, device = "cuda", dtype = torch.float32, force = True) == 0
        target = types.SimpleNamespace(device = "cuda", dtype = torch.float32)
        assert (
            I.install_requested(types.SimpleNamespace(transformer = dit), "unsloth_int8", target) == 0
        )
    assert not checked and not calls and I.int8_attention_count(dit) == 0


def test_a_cuda_verdict_never_stands_in_for_triton(monkeypatch):
    calls = _fake_nvidia(monkeypatch)
    checks = []
    inner = I._check
    monkeypatch.setattr(I, "_check", lambda *a, **k: checks.append(a[-1]) or inner(*a, **k))
    cuda_ok = {"ok": True, "ms": {}, "min_tokens": 0, "impl": "cuda", "cos": 1.0}
    I._VERDICTS[(0, "torch.bfloat16", 4, 128, "native", False, "cuda")] = cuda_ok
    I._VERIFIED.add((0, 128, "torch.bfloat16", "cuda"))
    assert I.verdict("cuda", torch.bfloat16, 4, 128, timed = False) is cuda_ok and not checks
    q = types.SimpleNamespace(
        device = torch.device("cuda:0"), shape = (1, 8, 2, 128), dtype = torch.bfloat16
    )
    assert I._verified(q, q, q)
    # the Triton path is not verified by it: the op keeps native until Triton's own check passes
    monkeypatch.setenv(I.IMPL_ENV, "triton")
    assert not I._verified(q, q, q)
    found = I.verdict("cuda", torch.bfloat16, 4, 128, timed = False)
    assert (
        found is not cuda_ok
        and found["impl"] == "triton"
        and checks == ["triton"]
        and len(calls) == 2
    )
    assert I._verified(q, q, q)
    # and the other way round: a Triton pass does not verify the CUDA path
    monkeypatch.setattr(I, "_VERIFIED", {(0, 128, "torch.bfloat16", "triton")})
    monkeypatch.setenv(I.IMPL_ENV, "auto")
    assert not I._verified(q, q, q)
    # the same after a CUDA failure in this process: the CUDA pass no longer counts
    monkeypatch.setattr(I, "_VERIFIED", {(0, 128, "torch.bfloat16", "cuda")})
    I._CUDA_FAILED[(0, 128)] = "NVRTC not found"
    assert not I._verified(q, q, q)


def test_the_triton_race_times_the_triton_kernel(monkeypatch):
    """Timed (not forced) verdicts race the implementation that will serve: the int8 side is the Triton kernel."""
    calls = _fake_nvidia(monkeypatch)
    _no_nvrtc(monkeypatch)
    timed = []

    def median_ms(fns, rounds = 5):
        fns["int8"]()
        timed.append(len(calls))
        return {"incumbent": 1.0, "int8": 0.5}

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "RACE_TOKENS", (64, 32))
    found = I.verdict("cuda", torch.float32, 2, 128, timed = True)
    assert found["ok"] and found["impl"] == "triton" and found["min_tokens"] == 0
    assert timed == [3, 4] and set(found["ms"]) == {64, 32}
    I.verdict("cuda", torch.float32, 2, 128, timed = False)
    names = [k for k in I.status()["verdicts"] if "impl=triton" in k]
    assert len(names) == 2 and sum(" timed " in k for k in names) == 1


def test_triton_configs_cover_the_nvidia_cards():
    assert set(I.TRITON_KERNEL_CONFIGS) == {(8, 0), (8, 6), (8, 9), (12, 0)}
    assert I.TRITON_KERNEL_CONFIGS[(8, 6)] == I.TRITON_KERNEL_CONFIGS[(8, 0)]
    for table in I.TRITON_KERNEL_CONFIGS.values():
        assert set(table) == set(I.HEAD_DIMS)
        for head_dim, c in table.items():
            assert c["pv"] in T.PV_MODES and c.get("kscale", "key") in ("key", "tile")
            # the padded key count is a multiple of QUANT_ROWS and of bn; bm divides the padded query count
            assert int(c["bn"]) in (16, 32, 64, 128) and int(c["bm"]) in (16, 32, 64, 128)
            assert int(c["warps"]) in (1, 2, 4, 8, 16) and int(c["stages"]) >= 1


def test_the_attention_kernel_compiles_through_the_8bit_prefetch_fix(monkeypatch):
    """Triton >= 3.8 spills 8-bit dots on sm_8x unless its prefetch pass is skipped for them: the Triton attention
    kernel installs that fix before it compiles and carries it in its cache key."""
    pytest.importorskip("triton")
    from core.inference import diffusion_triton_prefetch as P

    installed = []
    monkeypatch.setattr(P, "fix_triton_sm8x_8bit_prefetch", lambda: installed.append(1) or True)
    monkeypatch.setattr(T, "_PREFETCH_FIX_ON", None)
    monkeypatch.setattr(torch.version, "hip", None)
    T.kernels()
    assert installed == [1] and T.prefetch_fix() is True and installed == [1]
    assert "PREFETCH_FIX" in T.kernels().attn.arg_names
    monkeypatch.setattr(T, "_PREFETCH_FIX_ON", None)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert T.prefetch_fix() is False and installed == [1]


def _triton_card() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) or T.triton is None:
        return False
    return I.triton_kernel_config(torch.cuda.get_device_capability(), 128) is not None


@pytest.mark.skipif(
    not _triton_card(), reason = "needs an NVIDIA card with a Triton INT8 attention config"
)
@pytest.mark.parametrize("head_dim", I.HEAD_DIMS)
def test_forced_triton_path_on_the_card(monkeypatch, head_dim):
    monkeypatch.setenv(I.IMPL_ENV, "triton")
    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 1500, 4, head_dim)
    I.int8_attention_op()
    before = dict(I.STATS)
    found = I.verdict("cuda", torch.bfloat16, 4, head_dim, timed = False)
    assert found["ok"] and found["impl"] == "triton", found
    assert found["cos"] >= I.MIN_COSINE and found["sharp_cos"] >= I.MIN_SHARP_COSINE
    got = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
    assert I.STATS["triton"] == before["triton"] + 1 and I.STATS["int8"] == before["int8"] + 1
    acc = I.accuracy(got, I.reference_attention(q, k, v))
    assert acc["cos"] > 0.999 and acc["rel_l1"] < I.MAX_REL_L1, acc
    # strided views (a fused QKV projection) and a key count off the tile size
    qkv = torch.randn(1, 1037, 3, 4, head_dim, device = "cuda", dtype = torch.bfloat16)
    q, k, v = qkv.unbind(2)
    got = torch.ops.unsloth_studio.int8_attention_nhd(q, k, v)
    assert I.accuracy(got, I.reference_attention(q, k, v))["cos"] > 0.999
    qg = q.detach().clone().requires_grad_(True)
    assert I.unsupported_reason(qg, k, v) == "needs gradients"


@pytest.mark.parametrize("wins", [True, False])
def test_a_requested_stand_in_still_has_to_win_its_race(monkeypatch, wins):
    """Asked for by name (force) on a card whose CUDA kernel cannot load: the Triton stand-in is timed against the
    incumbent and engages only where it is faster; the env switch 1 and IMPL=triton take it unraced."""
    calls = _fake_nvidia(monkeypatch)
    _no_nvrtc(monkeypatch)
    raced = []

    def median_ms(fns, rounds = 5):
        fns["int8"]()
        raced.append(1)
        return {"incumbent": 1.0, "int8": 0.5 if wins else 1.5}

    monkeypatch.setattr(I, "_median_ms", median_ms)
    monkeypatch.setattr(I, "RACE_TOKENS", (64, 32))
    target = types.SimpleNamespace(device = "cuda", dtype = torch.float32)
    dit = _fake_dit()
    n = I.install_requested(types.SimpleNamespace(transformer = dit), "unsloth_int8", target)
    assert raced and n == (2 if wins else 0)
    assert [m.processor._unsloth_int8 for m in dit.blocks[:2]] == [wins, wins]
    timed = [k for k in I.status()["verdicts"] if " timed " in k]
    assert timed and all("impl=triton" in k for k in timed)
    for env, value in ((I.INT8_ATTENTION_ENV, "1"), (I.IMPL_ENV, "triton")):
        raced.clear()
        monkeypatch.setenv(env, value)
        dit = _fake_dit()
        assert (
            I.install_requested(types.SimpleNamespace(transformer = dit), "unsloth_int8", target) == 2
        )
        assert not raced and [m.processor._unsloth_int8_min_tokens for m in dit.blocks[:2]] == [
            0,
            0,
        ]
        monkeypatch.delenv(env)
    assert calls


def test_a_requested_cuda_kernel_is_not_raced(monkeypatch):
    """The CUDA kernel itself keeps its forced semantics: asked for by name it engages on accuracy alone."""
    _fake_nvidia(monkeypatch)
    monkeypatch.setattr(I, "quantise", lambda *a, **k: None)
    monkeypatch.setattr(I, "attend", lambda p: None)
    monkeypatch.setattr(
        I, "_check", lambda *a, **k: {"ok": True, "ms": {}, "min_tokens": None, "impl": a[-1]}
    )
    target = types.SimpleNamespace(device = "cuda", dtype = torch.float32)
    dit = _fake_dit()
    assert I.install_requested(types.SimpleNamespace(transformer = dit), "unsloth_int8", target) == 2
    assert not [k for k in I.status()["verdicts"] if " timed " in k]
    assert {m.processor._unsloth_int8_impl for m in dit.blocks[:2]} == {"cuda"}
