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


def _assert_same(stock, mapped):
    assert list(stock) == list(mapped)
    for key in stock:
        a, b = stock[key], mapped[key]
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
    _assert_same(stock(str(path)), mlu.load_gguf_checkpoint(str(path)))
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
