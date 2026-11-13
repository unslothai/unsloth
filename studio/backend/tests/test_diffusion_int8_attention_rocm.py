# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's INT8 attention on ROCm: the Triton form of the kernel (``diffusion_int8_attention_triton``).

CPU: routing by gfx arch (only measured arches engage; NVIDIA keeps its capability table), the dispatch to the Triton
kernel, and the kernel's numerics under the Triton interpreter. ROCm GPU: the kernel against fp64 attention (also
with sharp scores), the quantisers' rounding, engagement through the registered op and the kill switch."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from core.inference import diffusion_int8_attention as I

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _rocm_config(head_dim: int = 128):
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return None
    try:
        return I.device_config(torch.cuda.current_device(), head_dim)
    except Exception:  # noqa: BLE001
        return None


def _triton_ok() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_rocm_kernel = pytest.mark.skipif(
    _rocm_config() is None or not _triton_ok(),
    reason = "needs a ROCm card with an INT8 attention config and Triton",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    stats = dict(I.STATS)
    yield
    I.STATS.update(stats)


def _fake_rocm(
    monkeypatch,
    arch = "gfx1151:sramecc-:xnack-",
    found = None,
):
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (11, 5))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a, **k: types.SimpleNamespace(gcnArchName = arch)
    )
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: True)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    seen = []

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        seen.append((heads, head_dim, incumbent_name, timed))
        return found(head_dim) if callable(found) else found

    monkeypatch.setattr(I, "verdict", verdict)
    return seen


class _Proc:
    _unsloth_int8 = False


class _Attention(torch.nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.processor = _Proc()


def _fake_dit(head_dims = (128, 128, 64, 96)):
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([_Attention(2, d) for d in head_dims])
    return dit


# ---------------------------------------------------------------- routing (CPU)


def test_rocm_config_is_keyed_by_measured_gfx_arch(monkeypatch):
    _fake_rocm(monkeypatch)
    assert I.device_config(0, 128) == I.ROCM_KERNEL_CONFIGS["gfx1151"][128]
    assert I.device_config(0, 64) == I.ROCM_KERNEL_CONFIGS["gfx1151"][64]
    assert I.device_config(0, 96) is None
    # unmeasured AMD cards keep native attention, whatever capability torch reports for them
    for arch in ("gfx1100", "gfx1150", "gfx1201", "gfx942:sramecc+:xnack-", "gfx950", ""):
        _fake_rocm(monkeypatch, arch = arch)
        assert I.device_config(0, 128) is None, arch
    # the CUDA table is never consulted on ROCm and the ROCm table never on CUDA
    assert (11, 5) not in I.KERNEL_CONFIGS
    for config in I.ROCM_KERNEL_CONFIGS.values():
        assert set(config) <= set(I.HEAD_DIMS)
        assert all(
            c["pv"] in ("int8", "half", "half_strided", "half_t", "int8_t")
            and c.get("kscale", "key") in ("key", "tile")
            for c in config.values()
        )


def test_nvidia_keeps_its_capability_table(monkeypatch):
    from core.inference import rocm_bf16
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    for capability in [*I.KERNEL_CONFIGS, (9, 0), (7, 5), (11, 5)]:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, c = capability, **k: c)
        for head_dim in (64, 128, 96):
            assert I.device_config(0, head_dim) == I.kernel_config(capability, head_dim)


def test_install_on_rocm_engages_only_the_measured_arch(monkeypatch):
    seen = _fake_rocm(monkeypatch, found = {"ok": True, "min_tokens": 0})
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16, force = True) == 3
    assert [m.processor._unsloth_int8 for m in dit.blocks] == [True, True, True, False]
    assert {s[1] for s in seen} == {128, 64}
    _fake_rocm(monkeypatch, arch = "gfx1100", found = {"ok": True, "min_tokens": 0})
    assert (
        I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.bfloat16, force = True) == 0
    )
    # a failed on-card check and the kill switch both keep native attention
    _fake_rocm(monkeypatch, found = {"ok": False, "min_tokens": None})
    assert (
        I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.bfloat16, force = True) == 0
    )
    _fake_rocm(monkeypatch, found = {"ok": True, "min_tokens": 0})
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert (
        I.install_int8_attention(_fake_dit(), device = "cuda", dtype = torch.bfloat16, force = True) == 0
    )


def test_install_on_rocm_follows_the_race(monkeypatch):
    seen = _fake_rocm(
        monkeypatch,
        found = lambda head_dim: {"ok": True, "min_tokens": 2560 if head_dim == 128 else None},
    )
    dit = _fake_dit()
    assert I.install_int8_attention(dit, device = "cuda", dtype = torch.bfloat16) == 2
    assert [m.processor._unsloth_int8 for m in dit.blocks] == [True, True, False, False]
    assert all(s[3] for s in seen)


def test_rocm_call_runs_the_triton_kernel(monkeypatch):
    from core.inference import diffusion_int8_attention_triton as T

    _fake_rocm(monkeypatch)
    calls = []

    def fake(
        q,
        k,
        v,
        config = None,
        pv = None,
        scale = None,
    ):
        calls.append((tuple(q.shape), dict(config), pv, scale))
        return v.clone()

    monkeypatch.setattr(T, "int8_attention", fake)
    monkeypatch.setattr(I, "unsupported_reason", lambda q, k, v: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    q = torch.randn(1, 8, 2, 128, dtype = torch.bfloat16)
    out = I.int8_attention(q, q, q)
    assert out.shape == q.shape
    cfg = I.ROCM_KERNEL_CONFIGS["gfx1151"][128]
    assert calls == [((1, 8, 2, 128), cfg, None, None)]  # P.V mode and key scale ride in the config
    # NVIDIA never reaches the Triton module
    from core.inference import rocm_bf16

    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "attend", lambda p: "cuda")
    monkeypatch.setattr(I, "quantise", lambda *a, **k: None)
    assert I.int8_attention(q, q, q, config = {"bn": 64}) == "cuda" and len(calls) == 1


def test_triton_module_imports_without_triton():
    code = textwrap.dedent(
        """
        import builtins, sys
        real = builtins.__import__
        def guard(name, *a, **k):
            if name == "triton" or name.startswith("triton."):
                raise ImportError("no triton")
            return real(name, *a, **k)
        builtins.__import__ = guard
        from core.inference import diffusion_int8_attention_triton as T
        try:
            T.kernels()
        except RuntimeError as e:
            print("OK", e)
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd = BACKEND_DIR, capture_output = True, text = True, timeout = 120
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]


# ---------------------------------------------------------------- numerics under the Triton interpreter (CPU)

_INTERPRET = textwrap.dedent(
    """
    import json, sys, torch, torch.nn.functional as F
    from core.inference import diffusion_int8_attention as I
    from core.inference import diffusion_int8_attention_triton as T
    out = {}
    for tokens, D in ((131, 64), (200, 128)):
        for gain in (1.0, 5.0):
            q, k, v = I.probe_inputs("cpu", torch.float16, tokens, 2, D, seed = tokens)
            q, k = q * gain, k * gain
            ref = F.scaled_dot_product_attention(*(t.double().permute(0, 2, 1, 3) for t in (q, k, v))).permute(0, 2, 1, 3)
            for pv in ("int8", "half", "half_strided", "half_t", "int8_t"):
                for cfg in ({"bm": 64, "bn": 64, "warps": 4, "stages": 1, "kscale": "key"},
                            {"bm": 32, "bn": 32, "warps": 2, "stages": 1, "kscale": "tile", "long_keys": 150,
                             "long": {"bm": 64}, "acc_in_dot": True}):
                    got = T.int8_attention(q, k, v, cfg, pv = pv).double()
                    key = f"{tokens}/{D}/{gain}/{pv}/{cfg['bm']}x{cfg['bn']}/{cfg['kscale']}"
                    out[key] = {"cos": float(F.cosine_similarity(got.flatten(), ref.flatten(), 0)),
                                "rel_l1": float((got - ref).abs().mean() / ref.abs().mean()),
                                "finite": bool(torch.isfinite(got).all())}
    print("RESULT " + json.dumps(out))
    """
)


def _interpreter_ok() -> bool:
    try:
        import triton
        return tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 7)
    except Exception:  # noqa: BLE001
        return False


# Triton 3.6's interpreter fails on this kernel (InterpreterError on a 0-d array); 3.7 runs it. The card tests below
# cover the compiled kernel wherever a ROCm card is present.
@pytest.mark.skipif(not _interpreter_ok(), reason = "needs the Triton >= 3.7 interpreter")
def test_triton_kernel_numerics_interpreted():
    """The kernel's algorithm (quantisers, online softmax, INT8 and 16-bit P.V, ragged key tiles) vs fp64 attention,
    run by the Triton interpreter on CPU (fp16: the interpreter has no bf16 dot)."""
    env = dict(os.environ, TRITON_INTERPRET = "1")
    r = subprocess.run(
        [sys.executable, "-c", _INTERPRET],
        cwd = BACKEND_DIR,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, (r.stdout[-2000:], r.stderr[-3000:])
    res = json.loads(line[0][7:])
    assert len(res) == 40
    for key, m in res.items():
        gain = float(key.split("/")[2])
        assert m["finite"], key
        assert m["cos"] >= (I.MIN_COSINE if gain == 1.0 else I.MIN_SHARP_COSINE), (key, m)
        if gain == 1.0:
            assert m["rel_l1"] <= I.MAX_REL_L1 / 2, (key, m)


# ---------------------------------------------------------------- the real card (ROCm GPU)


def _rotate_ref(x):
    D = x.shape[-1]
    cols = torch.arange(D, device = x.device)
    flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
    x = x * (1.0 - 2.0 * flip.to(x.dtype))
    for _ in range(4):
        y = x.reshape(*x.shape[:-1], D // 2, 2)
        z = torch.stack((y[..., 0] + y[..., 1], y[..., 0] - y[..., 1]), dim = -1)
        x = z.reshape(*x.shape[:-1], D // 16, 8, 2).transpose(-1, -2).reshape(*x.shape[:-1], D)
    return x


def _fp64(q, k, v):
    return F.scaled_dot_product_attention(
        *(t.double().permute(0, 2, 1, 3) for t in (q, k, v))
    ).permute(0, 2, 1, 3)


def _cos(a, b):
    return float(F.cosine_similarity(a.double().flatten(), b.double().flatten(), 0))


@needs_rocm_kernel
@pytest.mark.parametrize("tokens", [77, 1023, 1024, 1324])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kernel_matches_fp64_on_the_card(tokens, head_dim):
    if _rocm_config(head_dim) is None:
        pytest.skip("no config for this head dim")
    for gain, floor in ((1.0, I.MIN_COSINE), (5.0, I.MIN_SHARP_COSINE), (8.0, 0.98)):
        q, k, v = I.probe_inputs("cuda", torch.bfloat16, tokens, 3, head_dim, seed = tokens)
        q, k = q * gain, k * gain
        got = I.int8_attention(q, k, v)
        assert got.shape == q.shape and got.dtype == v.dtype
        assert bool(torch.isfinite(got).all()), gain
        ref = _fp64(q, k, v)
        assert _cos(got, ref) >= floor, (gain, _cos(got, ref))
        if gain == 1.0:
            assert float((got.double() - ref).abs().mean() / ref.abs().mean()) <= I.MAX_REL_L1


@needs_rocm_kernel
def test_strided_views_batch_and_cross_attention():
    """[B, S, H, D] views cut from a fused qkv projection, batch 2, and Sq != Sk all match fp64."""
    B, S, H, D = 2, 700, 4, 128
    qkv = torch.randn(B, S, 3, H, D, device = "cuda", dtype = torch.bfloat16) * 1.6
    q, k, v = qkv.unbind(2)
    assert not q.is_contiguous()
    got = I.int8_attention(q, k, v)
    assert _cos(got, _fp64(q, k, v)) >= I.MIN_COSINE
    got = I.int8_attention(q[:, :300], k, v)
    assert _cos(got, _fp64(q[:, :300], k, v)) >= I.MIN_COSINE


@needs_rocm_kernel
def test_quantisers_round_to_nearest_on_the_card():
    """The rounding constant must survive compilation: a backend folding (x + c) - c to x would truncate instead."""
    from core.inference import diffusion_int8_attention_triton as T

    q, k, v = I.probe_inputs("cuda", torch.bfloat16, 1000, 2, 128, seed = 3)
    cfg = _rocm_config(128)
    p = T.quantise(q, k, v, cfg, pv = "int8")
    # rebuild the Q quantiser in torch: rotation, per-row amax, round half even
    x = _rotate_ref(q[0].float().permute(1, 0, 2))  # [H, S, D]
    amax = x.abs().amax(-1, keepdim = True).clamp_min(1e-20)
    ref = torch.round(x * (127.0 / amax)).to(torch.int8)
    got = p["qi"][:, :1000]
    diff = (got.int() - ref.int()).abs()
    assert int(diff.max()) <= 1 and float((diff > 0).float().mean()) < 1e-3


@needs_rocm_kernel
def test_op_engages_on_the_card_and_kill_switch(monkeypatch):
    dev = torch.device("cuda", torch.cuda.current_device())
    found = I.verdict(dev, torch.bfloat16, 4, 128, None, "native", timed = False)
    assert found["ok"], found
    op = I.int8_attention_op()
    q, k, v = I.probe_inputs(dev, torch.bfloat16, 1024, 4, 128, seed = 9)
    before = I.STATS["int8"]
    out = op(q, k, v)
    assert I.STATS["int8"] == before + 1
    assert _cos(out, _fp64(q, k, v)) >= I.MIN_COSINE
    dit = _fake_dit()
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    assert I.install_int8_attention(dit, device = str(dev), dtype = torch.bfloat16, force = True) == 0
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    assert I.install_int8_attention(dit, device = str(dev), dtype = torch.bfloat16) == 3


def test_long_sequences_take_the_long_config():
    from core.inference import diffusion_int8_attention_triton as T

    cfg = I.ROCM_KERNEL_CONFIGS["gfx1151"][128]
    assert T.select_config(cfg, int(cfg["long_keys"]) - 1) == cfg
    longer = T.select_config(cfg, int(cfg["long_keys"]))
    assert longer == {**cfg, **cfg["long"]} and longer["bm"] > cfg["bm"]
    plain = {"bm": 64, "bn": 32, "warps": 4, "stages": 1}
    assert T.select_config(plain, 10**6) == plain
    assert T._launch_options(plain) == {"num_warps": 4, "num_stages": 1}
    assert T._launch_options(dict(plain, waves_per_eu = 2))["waves_per_eu"] == 2
