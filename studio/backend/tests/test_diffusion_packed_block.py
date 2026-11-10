# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""nvfp4 / mxfp8 layers kept packed when no native runtime takes them: same weights as the load-time widen."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_comfy_block as cb  # noqa: E402
import core.inference.diffusion_comfy_quant as cq  # noqa: E402
import core.inference.diffusion_packed_block as pb  # noqa: E402
from .test_diffusion_comfy_block import DIM, _Tiny, _load, block_file  # noqa: E402,F401

LINEARS = [f"blocks.{b}.{n}" for b in range(2) for n in ("to_q", "to_k", "to_v", "to_out")]


_HELPERS = textwrap.dedent(
    """
    import torch


    def _bits(t):
        return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


    def _exhaustive_nvfp4():
        # every code byte against every e4m3 scale byte (NaN 0x7f / 0xff included), 16 blocks per row
        rows = 256
        codes = (torch.arange(128, dtype = torch.int64).view(1, -1) * 2 + torch.arange(rows).view(-1, 1)) % 256
        scale = (torch.arange(rows).view(-1, 1) * 16 + torch.arange(16)) % 256
        return codes.to(torch.uint8).contiguous(), scale.to(torch.uint8).contiguous()
    """
)
exec(_HELPERS)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise, NaN positions included (payloads compared too)."""
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_packed_load_widens_to_the_dense_loads_exact_weights(block_file, fmt, dtype):
    path, _tensors, _expected = block_file(fmt)
    dense = _load(path, dtype = dtype)
    packed = _load(path, dtype = dtype, fp8_storage = True)
    info = packed._unsloth_comfy_quant
    assert info[f"{fmt}_packed"] == 8 and info["dequantized"] == 0
    assert pb.packed_block_layers(packed) == 8
    state = dense.state_dict()
    x = torch.randn(3, 5, DIM, generator = torch.Generator().manual_seed(1)).to(dtype)
    for name in LINEARS:
        layer = packed.get_submodule(name)
        assert pb.is_packed_block_linear(layer), name
        assert not any(b.is_floating_point() for n, b in layer.named_buffers() if n != "bias")
        assert _same(layer.weight, state[name + ".weight"]), name
        ref = dense.get_submodule(name)
        assert _same(layer(x), ref(x)), name
    # packed bytes: nvfp4 codes are half a byte per weight, mxfp8 one
    q = packed.blocks[0].to_q
    assert q.weight_q.numel() == DIM * DIM // (2 if fmt == "nvfp4" else 1)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_the_kill_switch_and_a_widening_pick_keep_todays_dense_load(block_file, fmt, monkeypatch):
    path, _tensors, expected = block_file(fmt)
    for kwargs, env in (({}, None), ({"fp8_storage": True}, "0")):
        if env is None:
            monkeypatch.delenv(pb.PACKED_BLOCK_ENV, raising = False)
        else:
            monkeypatch.setenv(pb.PACKED_BLOCK_ENV, env)
        model = _load(path, **kwargs)
        info = model._unsloth_comfy_quant
        assert info["dequantized"] == 4 and f"{fmt}_packed" not in info
        assert pb.packed_block_layers(model) == 0
        state = model.state_dict()
        for key, want in expected.items():
            assert torch.equal(state[key], want), key


def test_module_casts_and_moves_never_touch_the_packed_buffers(block_file):
    path, _tensors, _expected = block_file("nvfp4")
    model = _load(path, dtype = torch.bfloat16, fp8_storage = True)
    layer = model.blocks[1].to_out
    before = {n: b.clone() for n, b in layer.named_buffers() if n != "bias"}
    weight = layer.weight.clone()
    model.to(torch.float16)
    for name, buf in layer.named_buffers():
        if name != "bias":
            assert buf.dtype == before[name].dtype and torch.equal(buf, before[name]), name
    assert layer.compute_dtype == torch.float16  # the widen target follows the cast
    model.to(torch.bfloat16)
    assert _same(layer.weight, weight)  # and back: the load's exact weight again


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("cast", ["to", "method"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_a_cast_packed_model_forwards_like_a_dense_load_in_that_dtype(block_file, fmt, cast, dtype):
    path, _tensors, _expected = block_file(fmt)
    packed = _load(path, dtype = torch.bfloat16, fp8_storage = True)
    if cast == "to":
        packed.to(dtype)
    else:
        packed.half() if dtype == torch.float16 else packed.float()
    dense = _load(path, dtype = dtype)
    state = dense.state_dict()
    x = torch.randn(3, 5, DIM, generator = torch.Generator().manual_seed(2)).to(dtype)
    for name in LINEARS:
        layer = packed.get_submodule(name)
        assert pb.is_packed_block_linear(layer), name
        assert layer.weight_q.dtype == torch.uint8 and layer.weight_sf.dtype == torch.uint8
        assert _same(layer.weight, state[name + ".weight"]), name
        out = layer(x)
        assert out.dtype == dtype, name
        # the bias went through the cast (bf16 then the new dtype), as a dense model's would
        want = torch.nn.functional.linear(x, state[name + ".weight"], layer.bias)
        assert _same(out, want), name


def _parts(
    fmt,
    rows,
    seed,
    tensor_scale = 0.5,
    smoothing = None,
):
    gen = torch.Generator().manual_seed(seed)
    cols = 64
    block = cb.BLOCK_SIZES[fmt]
    if fmt == "nvfp4":
        codes = torch.randint(0, 256, (rows, cols // 2), dtype = torch.uint8, generator = gen)
        scale = torch.randint(0, 0x7F, (rows, cols // block), dtype = torch.uint8, generator = gen)
        scale = scale.view(torch.float8_e4m3fn)
    else:
        codes = torch.randint(0, 0x7F, (rows, cols), dtype = torch.uint8, generator = gen)
        codes = codes | (torch.randint(0, 2, codes.shape, dtype = torch.uint8, generator = gen) << 7)
        codes = codes.view(torch.float8_e4m3fn)
        scale = torch.randint(100, 150, (rows, cols // block), dtype = torch.uint8, generator = gen)
    extra = {"tensor_scale": tensor_scale, "pre_quant_scale": smoothing}
    return codes, scale, extra


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows", [1, 17, 1023, 1024, 1025])
def test_row_chunks_splits_and_smoothing_widen_bit_identically(fmt, rows, monkeypatch):
    smoothing = torch.rand(64, generator = torch.Generator().manual_seed(7)) + 0.5
    parts = [
        _parts(fmt, rows, 1, tensor_scale = 0.25 if fmt == "nvfp4" else None, smoothing = smoothing),
        _parts(fmt, 3, 2, tensor_scale = None, smoothing = smoothing),
        _parts(fmt, 5, 3, tensor_scale = 3.0 if fmt == "nvfp4" else None, smoothing = smoothing),
    ]
    want = cq._dequant_parts(fmt, parts, torch.bfloat16)
    codes, scale, row_scale, pre_quant = pb.packed_block_args(parts)
    layer = pb.packed_block_linear_class()(
        fmt,
        64,
        int(codes.shape[0]),
        codes = codes,
        scale = scale,
        row_scale = row_scale,
        pre_quant = pre_quant,
        compute_dtype = torch.bfloat16,
    )
    monkeypatch.setattr(pb, "_HOST_CHUNK_BYTES", 4096)
    assert _same(layer.weight, want)
    monkeypatch.setattr(pb, "_HOST_CHUNK_BYTES", 1 << 30)
    assert _same(layer.weight, want)


def test_segments_with_different_smoothing_stay_dense():
    a = _parts("nvfp4", 4, 1, smoothing = torch.ones(64))
    b = _parts("nvfp4", 4, 2, smoothing = torch.full((64,), 2.0))
    c = _parts("nvfp4", 4, 3, smoothing = None)
    assert pb.packed_block_args([a, b]) is None
    assert pb.packed_block_args([a, c]) is None
    assert pb.packed_block_args([a, a]) is not None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_exhaustive_codes_and_scales_through_the_torch_widen(dtype):
    codes, scale = _exhaustive_nvfp4()
    row_scale = torch.full((codes.shape[0],), 2.0**-9, dtype = torch.float32)
    pre_quant = torch.linspace(0.25, 4.0, 2 * codes.shape[1], dtype = torch.float32)
    want = cb.dequant_block(
        "nvfp4",
        codes,
        scale.view(torch.float8_e4m3fn),
        tensor_scale = row_scale,
        pre_quant_scale = pre_quant,
        dtype = dtype,
    )
    got = pb.widen_packed(
        "nvfp4", codes, scale, row_scale.view(torch.int32), pre_quant.view(torch.int32), dtype
    )
    assert _same(got, want)


_INTERPRETED = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, {root!r})
    import core.inference.diffusion_comfy_block as cb
    import core.inference.diffusion_packed_block as pb
    exec({helpers!r})
    codes, scale = _exhaustive_nvfp4()
    rows = codes.shape[0]
    bad = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for rs, pq in ((None, None), (torch.full((rows,), 0.75), None),
                       (torch.full((rows,), 2.0 ** -9), torch.linspace(0.25, 4.0, 2 * codes.shape[1]))):
            want = cb.dequant_block("nvfp4", codes, scale.view(torch.float8_e4m3fn),
                                    tensor_scale = rs, pre_quant_scale = pq, dtype = dtype)
            got = pb._triton_widen(codes, scale, None if rs is None else rs.view(torch.int32),
                                   None if pq is None else pq.view(torch.int32), dtype)
            assert got is not None, pb._TRITON.get("error")
            nan = torch.isnan(want.float())
            if not torch.equal(nan, torch.isnan(got.float())):
                bad.append((str(dtype), "nan positions"))
            elif not torch.equal(_bits(got)[~nan], _bits(want)[~nan]):
                bad.append((str(dtype), "values"))
    print("BAD", bad)
    """
)


def test_the_triton_widen_is_bit_identical_under_the_interpreter():
    pytest.importorskip("triton")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERPRETED.format(root = root, helpers = _HELPERS)],
        cwd = root,
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert "BAD []" in proc.stdout, proc.stdout + proc.stderr[-2000:]


def test_a_compiled_forward_has_one_graph_and_the_same_output(block_file):
    path, _tensors, _expected = block_file("nvfp4")
    model = _load(path, dtype = torch.float32, fp8_storage = True)
    layer = model.blocks[0].to_q
    x = torch.randn(4, DIM, generator = torch.Generator().manual_seed(3))
    want = layer(x)
    torch._dynamo.reset()
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(layer, backend = counter, fullgraph = True)
    assert _same(compiled(x), want)
    y = torch.randn(4, DIM, generator = torch.Generator().manual_seed(4))
    assert _same(compiled(y), layer(y))
    assert counter.frame_count == 1


def test_fp8_storage_prices_block_layers_packed(block_file, monkeypatch):
    path, _tensors, _expected = block_file("nvfp4")
    scan = cq.scan_comfy_quant(path)
    kw = dict(keep_int8 = False, keep_fp8 = False)
    dense = cq.comfy_resident_mib(path, scan, **kw)
    packed = cq.comfy_resident_mib(path, scan, fp8_stored = True, **kw)
    kept = cq.comfy_resident_mib(path, scan, keep_nvfp4 = True, **kw)
    assert packed == kept < dense
    monkeypatch.setenv(pb.PACKED_BLOCK_ENV, "0")
    assert cq.comfy_resident_mib(path, scan, fp8_stored = True, **kw) == dense


def test_packed_layers_render_under_no_grad_and_refuse_a_late_lora(block_file):
    import types

    from core.inference.diffusion import _lora_load_block_reason

    path, _tensors, _expected = block_file("nvfp4")
    model = _load(path, fp8_storage = True)
    pipe = types.SimpleNamespace(transformer = model)
    assert cq.pipe_fp8_stored_layers(pipe) == 8
    assert pb.PACKED_BLOCK_ENV in (_lora_load_block_reason(pipe) or "")


def test_devices_without_fp8_casts_widen_nvfp4_by_table_and_keep_mxfp8_dense(monkeypatch):
    import types

    mps = types.SimpleNamespace(device = "mps", torch_device = "mps")
    assert pb.packed_block_runs_on("nvfp4", mps) and not pb.packed_block_runs_on("mxfp8", mps)
    assert pb.packed_block_runs_on(
        "mxfp8", types.SimpleNamespace(device = "cuda", torch_device = "cuda:1")
    )
    # the table decode of the e4m3 scales gives the same weight as the cast
    codes, scale = _exhaustive_nvfp4()
    want = pb.widen_packed("nvfp4", codes, scale, None, None, torch.bfloat16)
    monkeypatch.setattr(pb, "fp8_casts", lambda device: False)
    assert _same(pb.widen_packed("nvfp4", codes, scale, None, None, torch.bfloat16), want)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a CUDA / ROCm GPU to run the Triton widen"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_triton_widen_on_the_gpu_matches_the_load_time_widen(dtype):
    codes, scale = _exhaustive_nvfp4()
    rows = codes.shape[0]
    for rs, pq in (
        (None, None),
        (torch.full((rows,), 0.75), None),
        (torch.full((rows,), 2.0**-9), torch.linspace(0.25, 4.0, 2 * codes.shape[1])),
    ):
        want = cb.dequant_block(
            "nvfp4",
            codes,
            scale.view(torch.float8_e4m3fn),
            tensor_scale = rs,
            pre_quant_scale = pq,
            dtype = dtype,
        )
        got = pb._triton_widen(
            codes.cuda(),
            scale.cuda(),
            None if rs is None else rs.cuda().view(torch.int32),
            None if pq is None else pq.cuda().view(torch.int32),
            dtype,
        )
        assert got is not None, pb._TRITON.get("error")
        nan = torch.isnan(want.float())
        got = got.cpu()
        assert torch.equal(torch.isnan(got.float()), nan)
        assert torch.equal(_bits(got)[~nan], _bits(want)[~nan])


def _packed_projections(module, names = ("to_q", "to_k", "to_v")):
    cls = pb.packed_block_linear_class()
    for name, mod in list(module.named_modules()):
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in names:
            rows, cols = mod.weight.shape
            layer = cls(
                "mxfp8",
                cols,
                rows,
                codes = torch.ones(rows, cols).to(torch.float8_e4m3fn),
                scale = torch.full((rows, cols // 32), 127, dtype = torch.uint8),
                bias = None if mod.bias is None else mod.bias.detach(),
                compute_dtype = torch.bfloat16,
            )
            parent, _, leaf = name.rpartition(".")
            setattr(module.get_submodule(parent) if parent else module, leaf, layer)
    return module


def test_qkv_fusion_skips_packed_projections():
    import types

    from core.inference import diffusion_speed

    calls = []

    class _Dit(torch.nn.Module):
        def __init__(self, packed):
            super().__init__()
            self.to_q = torch.nn.Linear(64, 64).to(torch.bfloat16)
            if packed:
                _packed_projections(self)

        def fuse_qkv_projections(self):
            calls.append(self)

    packed = types.SimpleNamespace(transformer = _Dit(packed = True))
    assert diffusion_speed._fuse_qkv(packed, None) is False and not calls
    dense = types.SimpleNamespace(transformer = _Dit(packed = False))
    assert diffusion_speed._fuse_qkv(dense, None) is True and len(calls) == 1


def test_qkv_fusion_of_a_packed_flux_builds_no_dense_to_qkv():
    import types

    diffusers = pytest.importorskip("diffusers")
    from core.inference import diffusion_speed

    dit = diffusers.FluxTransformer2DModel(
        in_channels = 8,
        num_layers = 1,
        num_single_layers = 0,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        pooled_projection_dim = 16,
        axes_dims_rope = (4, 6, 6),
    ).to(torch.bfloat16)
    _packed_projections(dit)
    before = sum(t.numel() * t.element_size() for t in (*dit.parameters(), *dit.buffers()))
    pipe = types.SimpleNamespace(transformer = dit, unet = None, vae = None)
    assert diffusion_speed._fuse_qkv(pipe, None) is False
    after = sum(t.numel() * t.element_size() for t in (*dit.parameters(), *dit.buffers()))
    assert after == before
    assert not [n for n, _m in dit.named_modules() if n.endswith(("to_qkv", "to_added_qkv"))]
