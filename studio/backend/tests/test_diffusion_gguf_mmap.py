# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")
mlu = pytest.importorskip("diffusers.models.model_loading_utils")

from core.inference import diffusion_gguf_mmap as gm  # noqa: E402

T = gguf.GGMLQuantizationType
needs_posix = pytest.mark.skipif(os.name == "nt", reason = "the mapped loader is off on Windows")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(gm.GGUF_MMAP_ENV, raising = False)
    gm.uninstall_gguf_mmap()
    gm.MAPPED_LOADS.clear()
    yield
    gm.uninstall_gguf_mmap()
    gm.MAPPED_LOADS.clear()


def _k_blocks(rng, qtype, rows, cols):
    # gguf-py cannot quantize K-quants
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, (rows, cols // block, size), dtype = np.uint8)
    scale = np.frombuffer(np.float16(0.01).tobytes(), dtype = np.uint8)
    if qtype == T.Q4_K:
        raw[..., 0:2] = scale
        raw[..., 2:4] = scale
    else:
        raw[..., size - 2 : size] = scale
    return raw.reshape(rows, -1)


def _write(path, seed = 0):
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(str(path), "flux")
    dense = rng.standard_normal((64, 512)).astype(np.float32)
    raw = gguf.quants.quantize(dense, T.Q8_0)
    writer.add_tensor("blocks.0.attn.to_q.weight", raw, raw_shape = raw.shape, raw_dtype = T.Q8_0)
    for name, qtype in (("blocks.0.ff.w1.weight", T.Q4_K), ("blocks.1.ff.w2.weight", T.Q6_K)):
        writer.add_tensor(name, _k_blocks(rng, qtype, 64, 512), raw_dtype = qtype)
    writer.add_tensor("norm.weight", rng.standard_normal(512).astype(np.float32))
    writer.add_tensor("proj.bias", rng.standard_normal(64).astype(np.float16))
    bf16 = gguf.quants.quantize(rng.standard_normal((8, 512)).astype(np.float32), T.BF16)
    writer.add_tensor("time_embed.weight", bf16, raw_shape = bf16.shape, raw_dtype = T.BF16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _as_mapped(value):
    if getattr(value, "quant_type", None) == T.BF16:
        return value.as_subclass(torch.Tensor).view(torch.bfloat16)
    return value


def _assert_same(
    stock,
    mapped,
    bf16_as_mapped = True,
):
    assert list(stock) == list(mapped)
    for key in stock:
        a, b = stock[key], mapped[key]
        if bf16_as_mapped:
            a = _as_mapped(a)
        assert type(a) is type(b), key
        assert getattr(a, "quant_type", None) == getattr(b, "quant_type", None), key
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert torch.equal(a.as_subclass(torch.Tensor), b.as_subclass(torch.Tensor)), key


def _mapping_of(tensor):
    ptr = tensor.data_ptr()
    with open("/proc/self/maps") as fh:
        for line in fh:
            parts = line.split()
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            if lo <= ptr < hi:
                return parts[5] if len(parts) > 5 else ""
    return None


@needs_posix
def test_mapped_load_equals_the_stock_copy(tmp_path):
    path = _write(tmp_path / "m.gguf")
    stock = mlu.load_gguf_checkpoint(str(path))
    assert gm.install_gguf_mmap() is True
    assert mlu.load_gguf_checkpoint is gm._mapped_load_gguf_checkpoint
    mapped = mlu.load_gguf_checkpoint(str(path))
    _assert_same(stock, mapped)
    assert [p for p, _n in gm.MAPPED_LOADS] == [str(path)]
    assert gm.MAPPED_LOADS[0][1] == sum(t.numel() * t.element_size() for t in mapped.values())
    # from_single_file reads through load_state_dict
    via_state_dict = mlu.load_state_dict(str(path))
    _assert_same(stock, via_state_dict)
    assert len(gm.MAPPED_LOADS) == 2


@needs_posix
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "reads /proc/self/maps")
def test_mapped_tensors_view_the_file_not_an_anonymous_copy(tmp_path):
    path = _write(tmp_path / "m.gguf")
    stock = mlu.load_gguf_checkpoint(str(path))
    gm.install_gguf_mmap()
    mapped = mlu.load_gguf_checkpoint(str(path))
    real = os.path.realpath(path)
    for key, value in mapped.items():
        assert _mapping_of(value) == real, key
    for key, value in stock.items():
        assert _mapping_of(value) != real, key


@needs_posix
def test_writes_stay_private_and_the_file_is_unchanged(tmp_path):
    path = _write(tmp_path / "m.gguf")
    before = path.read_bytes()
    gm.install_gguf_mmap()
    mapped = mlu.load_gguf_checkpoint(str(path))
    for value in mapped.values():
        value.as_subclass(torch.Tensor).zero_()
    del mapped
    assert path.read_bytes() == before
    gm.uninstall_gguf_mmap()
    again = mlu.load_gguf_checkpoint(str(path))
    assert any(bool(v.as_subclass(torch.Tensor).any()) for v in again.values())


@needs_posix
def test_dequantised_forward_is_bit_identical(tmp_path):
    path = _write(tmp_path / "m.gguf")
    stock = mlu.load_gguf_checkpoint(str(path))
    gm.install_gguf_mmap()
    mapped = mlu.load_gguf_checkpoint(str(path))
    x = torch.randn(3, 512, generator = torch.Generator().manual_seed(1))
    for key in ("blocks.0.attn.to_q.weight", "blocks.0.ff.w1.weight", "blocks.1.ff.w2.weight"):
        outs = []
        for state in (stock, mapped):
            lin = gguf_utils.GGUFLinear(512, 64, bias = False, compute_dtype = torch.float32)
            lin.weight = state[key]
            outs.append(lin.forward_native(x))
        assert torch.equal(outs[0], outs[1]), key


@needs_posix
def test_unsupported_types_are_refused_like_the_stock_loader(tmp_path):
    path = tmp_path / "iq.gguf"
    writer = gguf.GGUFWriter(str(path), "flux")
    raw = np.zeros((4, 66), dtype = np.uint8)  # IQ2_XXS: not in diffusers' table
    writer.add_tensor("w", raw, raw_shape = raw.shape, raw_dtype = T.IQ2_XXS)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if T.IQ2_XXS in gguf_utils.SUPPORTED_GGUF_QUANT_TYPES:
        pytest.skip("this diffusers supports IQ2_XXS")
    with pytest.raises(ValueError, match = "unsupported") as stock_err:
        mlu.load_gguf_checkpoint(str(path))
    gm.install_gguf_mmap()
    with pytest.raises(ValueError, match = "unsupported") as mapped_err:
        mlu.load_gguf_checkpoint(str(path))
    assert str(stock_err.value).splitlines()[0] == str(mapped_err.value).splitlines()[0]


@needs_posix
def test_kill_switch_idempotence_and_uninstall(tmp_path, monkeypatch):
    stock = mlu.load_gguf_checkpoint
    monkeypatch.setenv(gm.GGUF_MMAP_ENV, "0")
    assert gm.install_gguf_mmap() is False
    assert mlu.load_gguf_checkpoint is stock
    monkeypatch.delenv(gm.GGUF_MMAP_ENV)
    assert gm.install_gguf_mmap() is True
    assert gm.install_gguf_mmap() is True
    path = _write(tmp_path / "m.gguf")
    monkeypatch.setenv(gm.GGUF_MMAP_ENV, "0")
    _assert_same(stock(str(path)), mlu.load_gguf_checkpoint(str(path)), bf16_as_mapped = False)
    assert gm.MAPPED_LOADS == []
    monkeypatch.delenv(gm.GGUF_MMAP_ENV)
    gm.uninstall_gguf_mmap()
    gm.uninstall_gguf_mmap()
    assert mlu.load_gguf_checkpoint is stock


def test_a_changed_upstream_loader_is_left_alone(monkeypatch):
    def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors = False):  # no whole-file copy line
        return {}

    monkeypatch.setattr(mlu, "load_gguf_checkpoint", load_gguf_checkpoint)
    assert gm.install_gguf_mmap() is False
    assert mlu.load_gguf_checkpoint is load_gguf_checkpoint


def test_off_on_windows(monkeypatch):
    monkeypatch.setattr(gm.os, "name", "nt")
    assert gm.gguf_mmap_enabled() is False
    assert gm.install_gguf_mmap() is False


def _write_bf16_mix(path):
    # F32, F16, BF16 (a 1-D norm as in Qwen-Image-2.1's txt_in.text_norm, a 2-D linear) and Q8_0
    gen = torch.Generator().manual_seed(3)
    norm = torch.randn(512, generator = gen).to(torch.bfloat16)
    norm[:6] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), 1e-40, -3.0e38]).to(
        torch.bfloat16
    )
    linear = torch.randn(64, 512, generator = gen).to(torch.bfloat16)
    writer = gguf.GGUFWriter(str(path), "flux")
    for name, value in (("txt_in.text_norm.weight", norm), ("img_in.weight", linear)):
        raw = value.view(torch.uint8).numpy()
        writer.add_tensor(name, raw, raw_shape = raw.shape, raw_dtype = T.BF16)
    q8 = gguf.quants.quantize(
        np.random.default_rng(0).standard_normal((64, 512)).astype(np.float32), T.Q8_0
    )
    writer.add_tensor("blocks.0.attn.to_q.weight", q8, raw_shape = q8.shape, raw_dtype = T.Q8_0)
    writer.add_tensor("norm.weight", np.arange(512, dtype = np.float32))
    writer.add_tensor("proj.bias", np.arange(64, dtype = np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, {"txt_in.text_norm.weight": norm, "img_in.weight": linear}


@needs_posix
def test_bf16_tensors_load_as_plain_bfloat16_views(tmp_path):
    path, expected = _write_bf16_mix(tmp_path / "bf16.gguf")
    stock = mlu.load_gguf_checkpoint(str(path))
    assert gm.install_gguf_mmap() is True
    mapped = mlu.load_gguf_checkpoint(str(path))
    for name, want in expected.items():
        got = mapped[name]
        assert type(got) is torch.Tensor, name
        assert got.dtype == torch.bfloat16 and got.shape == want.shape, name
        assert torch.equal(
            got.view(torch.int16), want.view(torch.int16)
        ), name  # bit-exact, incl. -0.0 / inf
        raw = stock[name].as_subclass(torch.Tensor)
        assert torch.equal(got.view(torch.uint8), raw), name
    if sys.platform.startswith("linux"):
        real = os.path.realpath(path)
        assert all(_mapping_of(mapped[n]) == real for n in expected)
    # A non-linear BF16 weight now multiplies as [D], not as [2D] raw bytes
    hidden = torch.ones(2, 512, dtype = torch.bfloat16)
    assert (hidden * mapped["txt_in.text_norm.weight"]).shape == (2, 512)
    for name in ("blocks.0.attn.to_q.weight", "norm.weight", "proj.bias"):
        a, b = stock[name], mapped[name]
        assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape, name
        assert getattr(a, "quant_type", None) == getattr(b, "quant_type", None), name
        assert torch.equal(a.as_subclass(torch.Tensor), b.as_subclass(torch.Tensor)), name
    assert gm.MAPPED_LOADS[-1][1] == sum(t.numel() * t.element_size() for t in mapped.values())


@needs_posix
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_bf16_norm_gguf_loads_and_runs_through_from_single_file(tmp_path, dtype):
    # diffusers#15406's crash, without a model download: a BF16 txt_norm reached RMSNorm as [2 * D] raw bytes
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "QwenImageTransformer2DModel", None)
    if cls is None:
        pytest.skip("this diffusers has no QwenImageTransformer2DModel")
    torch.manual_seed(0)
    model = cls(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = 1,
        attention_head_dim = 32,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        axes_dims_rope = (8, 12, 12),
    )
    model.save_config(str(tmp_path))
    path = tmp_path / "q.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen_image")
    bf16 = {}
    for key, value in model.state_dict().items():
        if key in (
            "txt_norm.weight",
            "img_in.weight",
        ):  # a norm and a linear, as in Qwen-Image-2.1 Turbo
            bf16[key] = torch.randn_like(value).to(torch.bfloat16)
            raw = bf16[key].view(torch.uint8).numpy()
            writer.add_tensor(key, raw, raw_shape = raw.shape, raw_dtype = T.BF16)
        elif value.ndim == 2 and value.shape[1] % 32 == 0:
            raw = gguf.quants.quantize(value.float().numpy(), T.Q8_0)
            writer.add_tensor(key, raw, raw_shape = raw.shape, raw_dtype = T.Q8_0)
        else:
            writer.add_tensor(key, value.float().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert gm.install_gguf_mmap() is True
    loaded = cls.from_single_file(
        str(path),
        config = str(tmp_path),
        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype),
        torch_dtype = dtype,
    )
    assert [p for p, _n in gm.MAPPED_LOADS] == [str(path)]
    # BF16 arrives at torch_dtype like F16 does (a bf16-less device asks for fp16), values = bf16 upcast
    for key, want in bf16.items():
        got = loaded.get_parameter(key)
        assert type(got) is torch.nn.Parameter and got.dtype == dtype, key
        assert torch.equal(got, want.to(dtype)), key
    assert type(loaded.transformer_blocks[0].attn.to_q).__name__ == "GGUFLinear"
    out = loaded(
        hidden_states = torch.randn(1, 16, 16, dtype = dtype),
        encoder_hidden_states = torch.randn(1, 5, 32, dtype = dtype),
        encoder_hidden_states_mask = torch.ones(1, 5),
        timestep = torch.tensor([0.5]),
        img_shapes = [(1, 4, 4)],
        return_dict = False,
    )[0]
    assert out.shape == (1, 16, 16) and bool(torch.isfinite(out).all())
