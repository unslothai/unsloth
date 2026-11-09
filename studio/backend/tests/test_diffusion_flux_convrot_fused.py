# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 / FLUX.3 int8 ConvRot producer fusion (diffusion_flux_convrot_fused). The patched helpers and forwards must
compute what the stock diffusers ones do: here the int8 pieces (the shared quant and the producer-fused down
projection) are replaced by the plain modules, so a forced-on patch has to match stock bit for bit on CPU. The GPU
kernel itself is checked against the stock path by its own probe and the GPU parity run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture()
def env(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    f2 = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    f3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    from core.inference import diffusion_flux_convrot_fused as fused
    from core.inference import diffusion_int8_gemm as gemm

    fused.uninstall()
    monkeypatch.delenv(fused.FLUX_CONVROT_FUSED_ENV, raising = False)
    yield torch, f2, f3, fused, gemm
    fused.uninstall()


def _force_on(monkeypatch, torch, fused, gemm, calls):
    """Every Linear counts as a swapped rotated int8 Linear; the int8 pieces run the plain modules."""
    import torch.nn.functional as F

    monkeypatch.setattr(fused, "_rotated", lambda lin: True)

    def shared(x, linears):
        calls["shared"] += 1
        return [m(x) for m in linears]

    def producer(down, prefix, gate, value):
        calls["producer"] += 1
        h = F.silu(gate) * value
        if prefix is not None:
            h = torch.cat([prefix.flatten(2, 3), h], dim = -1)
        return down(h)

    monkeypatch.setattr(gemm, "shared_rotq_linears", shared)
    monkeypatch.setattr(gemm, "producer_linear", producer)


def _rope(torch, seq, dim):
    g = torch.Generator().manual_seed(1)
    ang = torch.rand(seq, dim, generator = g) * 3
    return ang.cos(), ang.sin()


@pytest.fixture(params = ["eager", "compiled"])
def mode(request):
    return request.param


def _run_stock_vs_patched(
    torch,
    fused,
    build,
    call,
    mode = "eager",
):
    """Stock, patched (eager, or traced by dynamo: the per-call gates must hold inside the regional compile) and
    restored outputs of one block."""
    torch.manual_seed(0)
    block = build().eval()
    with torch.no_grad():
        ref = call(block)
        assert fused._install_patches()  # noqa: SLF001 - the install minus the int8 gates
        if mode == "compiled":
            torch._dynamo.reset()
            compiled = torch.compile(block, backend = "eager", fullgraph = True)
            out = call(compiled)
            torch._dynamo.reset()
        else:
            out = call(block)
    fused.uninstall()
    with torch.no_grad():
        back = call(block)
    return ref, out, back


def _same(a, b):
    if isinstance(a, (tuple, list)):
        return all(_same(x, y) for x, y in zip(a, b)) and len(a) == len(b)
    return a.shape == b.shape and bool((a == b).all())


def test_flux2_double_block_matches_stock(env, monkeypatch, mode):
    torch, f2, _f3, fused, gemm = env
    calls = {"shared": 0, "producer": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    dim, heads, hd = 64, 2, 32
    hs = torch.randn(2, 12, dim)
    enc = torch.randn(2, 5, dim)
    mod_i, mod_t = torch.randn(2, 6 * dim) * 0.1, torch.randn(2, 6 * dim) * 0.1
    rope = _rope(torch, 17, hd)
    ref, out, back = _run_stock_vs_patched(
        torch,
        fused,
        lambda: f2.Flux2TransformerBlock(dim, heads, hd),
        lambda b: b(hs, enc, mod_i, mod_t, image_rotary_emb = rope),
        mode,
    )
    assert _same(ref, out) and _same(ref, back)
    assert calls["shared"] == 2  # image Q/K/V and text Q/K/V, one quant each
    assert calls["producer"] == 2  # ff and ff_context down projections


def test_flux2_single_block_matches_stock(env, monkeypatch, mode):
    torch, f2, _f3, fused, gemm = env
    calls = {"shared": 0, "producer": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    dim, heads, hd = 64, 2, 32
    hs = torch.randn(2, 12, dim)
    enc = torch.randn(2, 5, dim)
    mod = torch.randn(2, 3 * dim) * 0.1
    rope = _rope(torch, 17, hd)
    ref, out, back = _run_stock_vs_patched(
        torch,
        fused,
        lambda: f2.Flux2SingleTransformerBlock(dim, heads, hd),
        lambda b: b(hs, enc, mod, image_rotary_emb = rope, split_hidden_states = True),
        mode,
    )
    assert _same(ref, out) and _same(ref, back)
    assert calls["producer"] == 1  # [attention | SwiGLU] into to_out, read from the producers


@pytest.mark.parametrize("kind", ["mode", "single"])
def test_flux3_blocks_match_stock(env, monkeypatch, kind, mode):
    torch, _f2, f3, fused, gemm = env
    calls = {"shared": 0, "producer": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    dim, heads, hd = 64, 2, 32
    hs = torch.randn(2, 12, dim)
    enc = torch.randn(2, 5, dim)
    mod = torch.randn(2, 1, 3 * dim) * 0.1
    cmod = torch.randn(2, 1, 3 * dim) * 0.1
    if kind == "mode":
        rope = _rope(torch, 12, hd)
        build = lambda: f3.Flux3ModeTransformerBlock(dim, heads, hd)  # noqa: E731
        call = lambda b: b(hs, mod, rope)  # noqa: E731
    else:
        rope = _rope(torch, 17, hd)
        build = lambda: f3.Flux3SingleTransformerBlock(dim, heads, hd)  # noqa: E731
        call = lambda b: b(hs, enc, mod, cmod, rope)  # noqa: E731
    ref, out, back = _run_stock_vs_patched(torch, fused, build, call, mode)
    assert _same(ref, out) and _same(ref, back)
    assert calls["shared"] == 1  # Q, K, V and the MLP input: one quant
    assert calls["producer"] == 1


def test_gates_off_runs_stock(env, monkeypatch):
    """Unswapped Linears (CPU, plain bf16, LoRA...): every patch takes the stock path, so nothing int8 is called."""
    torch, f2, f3, fused, gemm = env

    def boom(*a, **k):
        raise AssertionError("int8 path reached without swapped Linears")

    monkeypatch.setattr(gemm, "shared_rotq_linears", boom)
    monkeypatch.setattr(gemm, "producer_linear", boom)
    dim, heads, hd = 64, 2, 32
    torch.manual_seed(0)
    single = f2.Flux2SingleTransformerBlock(dim, heads, hd).eval()
    mode = f3.Flux3ModeTransformerBlock(dim, heads, hd).eval()
    hs = torch.randn(1, 8, dim)
    smod, mmod, rope = (
        torch.randn(1, 3 * dim) * 0.1,
        torch.randn(1, 1, 3 * dim) * 0.1,
        _rope(torch, 8, hd),
    )
    with torch.no_grad():
        a, b = single(hs, None, smod), mode(hs, mmod, rope)
        assert all(fused._install_patches().values())  # noqa: SLF001
        assert torch.equal(single(hs, None, smod), a)
        assert torch.equal(mode(hs, mmod, rope), b)


def test_drifted_body_keeps_stock(env, monkeypatch):
    torch, f2, _f3, fused, _gemm = env
    stock = f2.Flux2ParallelSelfAttnProcessor.__call__
    monkeypatch.setitem(
        fused._NEEDLES,
        "f2_parallel",
        fused._NEEDLES["f2_parallel"] + ("a line diffusers never had",),
    )
    done = fused._install_patches()  # noqa: SLF001
    assert done.get("f2_parallel") is False
    assert f2.Flux2ParallelSelfAttnProcessor.__call__ is stock


def test_uninstall_restores_everything(env):
    torch, f2, f3, fused, _gemm = env
    before = {
        "p2": f2._get_projections,
        "p3": f3._get_projections,
        "ff2": f2.Flux2FeedForward.forward,
        "ff3": f3.Flux3FeedForward.forward,
        "par": f2.Flux2ParallelSelfAttnProcessor.__call__,
        "mode": f3.Flux3ModeTransformerBlock.forward,
        "single": f3.Flux3SingleTransformerBlock.forward,
    }
    done = fused._install_patches()  # noqa: SLF001
    assert all(done.values()), done
    assert fused.is_installed()
    assert fused._install_patches() == done  # noqa: SLF001 - idempotent
    fused.uninstall()
    after = {
        "p2": f2._get_projections,
        "p3": f3._get_projections,
        "ff2": f2.Flux2FeedForward.forward,
        "ff3": f3.Flux3FeedForward.forward,
        "par": f2.Flux2ParallelSelfAttnProcessor.__call__,
        "mode": f3.Flux3ModeTransformerBlock.forward,
        "single": f3.Flux3SingleTransformerBlock.forward,
    }
    assert after == before
    assert not fused.is_installed()


def test_kill_switch(env, monkeypatch):
    torch, f2, _f3, fused, _gemm = env
    monkeypatch.setenv(fused.FLUX_CONVROT_FUSED_ENV, "0")

    class Flux2Transformer2DModel:
        pass

    assert fused.install(Flux2Transformer2DModel())["patches"] == {}
    assert not fused.is_installed()


def test_install_needs_rotated_linears(env):
    torch, _f2, _f3, fused, _gemm = env

    class Flux2Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

    out = fused.install(Flux2Transformer2DModel())
    assert out["rotated"] == 0 and out["patches"] == {} and not fused.is_installed()


# ------------------------------------------------------------------------------------------------ rotq_h16x kernel
def test_prefix_layout_reads_any_attention_output_stride():
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    heads_major = torch.zeros(2, 24, 41, 128).transpose(
        1, 2
    )  # SDPA's [B, H, S, D] storage seen as [B, S, H, D]
    assert cq._prefix_layout(heads_major) == (2, 41, 3072, 24 * 41 * 128, 128, 41 * 128, 128)
    seq_major = torch.zeros(1, 41, 24, 128)
    assert cq._prefix_layout(seq_major) == (1, 41, 3072, 41 * 3072, 3072, 128, 128)
    flat = torch.zeros(41, 3072)
    assert cq._prefix_layout(flat) == (1, 41, 3072, 0, 3072, 256, 256)
    assert cq._prefix_layout(torch.zeros(41, 3072).t()) is None  # last stride must be 1
    assert (
        cq._prefix_layout(torch.zeros(1, 41, 24, 96)) is None
    )  # head width the kernel does not index
    assert not cq.producer_layout_ok(
        None, torch.zeros(8, 256, dtype = torch.bfloat16), torch.zeros(8, 256, dtype = torch.bfloat16)
    )


def test_launch_shape_tables():
    pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    assert cq._h16x_pad((8, 0), 48, True) and not cq._h16x_pad((8, 0), 36, False)
    assert not cq._h16x_pad((8, 0), 16, True)  # a power of two is already one exact pass
    assert not cq._h16x_pad((8, 9), 48, True) and not cq._h16x_pad(None, 48, True)
    assert (
        cq._h16x_warps((8, 0), True, False, 16) == 8
        and cq._h16x_warps((8, 0), False, False, 16) == 4
    )
    assert cq._h16x_warps((8, 0), True, True, 64) == 16
    assert cq._h16x_warps((12, 0), True, False, 16) == cq._H16_WARPS_TWO_PASS


def _cuda_ready():
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0):
            return False
        if getattr(torch.version, "hip", None):
            return False
        import torchao  # noqa: F401
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_cuda = pytest.mark.skipif(
    not _cuda_ready(), reason = "needs NVIDIA sm80+ CUDA, Triton and torchao"
)


@pytest.fixture
def producer_armed(monkeypatch):
    import torch

    from core.inference import diffusion_convrot_quant as cq

    monkeypatch.delenv(cq.INT8_ROTQUANT_PRODUCER_ENV, raising = False)
    index = torch.cuda.current_device()
    cq.rotquant_device_config(index)
    if not cq._H16_DEVICE.get(index):
        cq.unlisted_device_ok(index)
    if not cq._arm_producer(index):
        pytest.skip("rotq_h16 / rotq_h16x probes refused this device")
    return cq


def _producer(torch, m, pre, f, seed):
    from core.inference import diffusion_convrot_quant as cq
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return cq._producer_probe_inputs(
        m, pre, f, g, torch.device("cuda", torch.cuda.current_device())
    )


@needs_cuda
@pytest.mark.parametrize(
    "m, pre, f",
    [
        (300, None, 256 * 36),  # FLUX MLP down projection (two-pass, looped)
        (77, None, 256 * 16),  # one pass
        (2 * 300, (2, 24, 128), 256 * 36),  # FLUX.2 single-block output projection, batch 2
        (129, (1, 4, 128), 256 * 12),
    ],
)
@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_producer_kernel_is_bit_identical_to_the_current_path(producer_armed, m, pre, f, kind):
    """rotq_h16x == the bf16 producer output (ATen SiLU product, concat) fed to the rotq op the current path runs."""
    import torch

    cq = producer_armed
    prefix, gate, value = _producer(torch, m, pre, f, m + f)
    q, s = cq.convrot_producer_act_quant(prefix, gate, value, 256, kind == "v2")
    rq, rs = cq._rotq_run(
        cq.producer_reference(prefix, gate, value).contiguous(), 256, kind == "v2"
    )
    assert torch.equal(q, rq) and torch.equal(s, rs)
    # every launch shape gives the same codes: the rotation and the rounding are per group / per row
    for kw in ({"pad": True}, {"pad": False, "warps": 8}, {"pad": False, "tile_rows": 32}):
        q2, s2 = cq._h16x_launch(prefix, gate, value, kind, **kw)
        assert torch.equal(q, q2) and torch.equal(s, s2), kw


@needs_cuda
def test_producer_declines_what_it_cannot_read(producer_armed):
    import torch

    cq = producer_armed
    prefix, gate, value = _producer(torch, 64, (1, 2, 128), 512, 1)
    assert cq.convrot_producer_act_quant(prefix, gate, value, 128, True) is None  # other group
    assert (
        cq.convrot_producer_act_quant(prefix[:, :32], gate, value, 256, True) is None
    )  # rows disagree
    assert (
        cq.convrot_producer_act_quant(None, gate[:, :200], value[:, :200], 256, True) is None
    )  # off-grid width
    assert cq.convrot_producer_act_quant(None, gate.float(), value.float(), 256, True) is None


@needs_cuda
def test_producer_kill_switch(producer_armed, monkeypatch):
    import torch

    cq = producer_armed
    index = torch.cuda.current_device()
    monkeypatch.setenv(cq.INT8_ROTQUANT_PRODUCER_ENV, "0")
    assert cq._arm_producer(index) is False
    prefix, gate, value = _producer(torch, 64, None, 512, 2)
    assert cq.convrot_producer_act_quant(prefix, gate, value, 256, True) is None
    monkeypatch.delenv(cq.INT8_ROTQUANT_PRODUCER_ENV)
    assert cq._arm_producer(index) is True  # the probe verdict was kept: no second probe
    assert cq.convrot_producer_act_quant(prefix, gate, value, 256, True) is not None


def test_swapped_record_is_seen_inside_a_compiled_block():
    """The shared / producer paths read ``_swapped_rec`` inside the regional compile. Dynamo resolves
    ``forward.__func__ is _linear_forward`` to False, so the traced check compares the instance forward with the bound
    method the swap stored: a swapped Linear is seen, a forward replaced after the swap is not."""
    import types

    torch = pytest.importorskip("torch")
    from core.inference import diffusion_int8_gemm as g8

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            return x + 1 if g8._swapped_rec(self.proj) is not None else x - 1

    def swapped():
        block = Block()
        lin = block.proj
        lin.__dict__[g8._REC] = ("v2", 256, True, lin.weight, True)
        lin.__dict__[g8._MARK] = g8._NO_PREV
        lin.forward = types.MethodType(g8._linear_forward, lin)
        lin.__dict__[g8._FWD] = lin.forward
        return block

    x = torch.zeros(1)
    torch._dynamo.reset()
    assert torch.compile(swapped(), backend = "eager", fullgraph = True)(x).item() == 1
    replaced = swapped()
    replaced.proj.forward = types.MethodType(lambda self, t: t, replaced.proj)
    torch._dynamo.reset()
    assert torch.compile(replaced, backend = "eager", fullgraph = True)(x).item() == -1
    unmarked = swapped()
    del unmarked.proj.__dict__[g8._FWD]
    torch._dynamo.reset()
    assert torch.compile(unmarked, backend = "eager", fullgraph = True)(x).item() == -1


def test_probe_verdicts_round_trip_through_the_cache(monkeypatch, tmp_path):
    """A passing device's probe verdicts persist per card + toolchain fingerprint; a later process adopts them without
    probing, and a changed fingerprint (driver, torch, switches, these sources) is a miss."""
    pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    path = tmp_path / cq._VERDICT_FILE
    fp = {"torch": "x", "gpu_uuid": "GPU-1"}
    monkeypatch.setattr(cq, "_verdict_slot", lambda index: (path, f"k{index}", dict(fp)))
    for name in (
        "_ROTQ_DEVICE",
        "_ROTQ_DEVICE_K",
        "_H16_DEVICE",
        "_SWIGLU_DEVICE",
        "_PRODUCER_DEVICE",
    ):
        monkeypatch.setattr(cq, name, {})
    monkeypatch.setattr(cq, "_rotq_op", lambda: "rotq-op")
    monkeypatch.setattr(cq, "_swiglu_op", lambda: "swiglu-op")
    assert cq._verdict_load(7) is None
    cq._verdict_store(
        7,
        {
            "cfg": [128, 32, 8, 3],
            "rules": [[256, 8192, [64, 32, 4, 3]]],
            "h16": "ieee",
            "swiglu": True,
        },
    )
    cq._verdict_store(7, {"producer": True})
    v = cq._verdict_load(7)
    assert v == {
        "cfg": [128, 32, 8, 3],
        "rules": [[256, 8192, [64, 32, 4, 3]]],
        "h16": "ieee",
        "swiglu": True,
        "producer": True,
    }
    assert cq._adopt_verdict(7, v) == (128, 32, 8, 3)
    assert cq._ROTQ_DEVICE_K[7] == ((256, 8192, (64, 32, 4, 3)),)
    assert (
        cq._H16_DEVICE[7] == "ieee"
        and cq._SWIGLU_DEVICE[7] is True
        and cq._PRODUCER_DEVICE[7] is True
    )
    fp["torch"] = "y"  # another toolchain: a miss, never a stale verdict
    assert cq._verdict_load(7) is None
    path.write_text("{torn")
    assert cq._verdict_load(7) is None
