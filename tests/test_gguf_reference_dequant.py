# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""Bit-exact checks of unsloth.kernels.gguf_reference against gguf-py's numpy dequantizers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REF_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "kernels" / "gguf_reference.py"


def _load_reference():
    # Load the file directly so the test does not need the full unsloth import (GPU checks).
    spec = importlib.util.spec_from_file_location("_unsloth_gguf_reference", _REF_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ref = _load_reference()

# Byte offsets of fp16 scale fields inside one block.
_F16_FIELDS = {
    "Q4_0": (0,),
    "Q4_1": (0, 2),
    "Q5_0": (0,),
    "Q5_1": (0, 2),
    "Q8_0": (0,),
    "Q2_K": (80, 82),
    "Q3_K": (108,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q6_K": (208,),
    "IQ2_XXS": (0,),
    "IQ2_XS": (0,),
    "IQ3_XXS": (0,),
    "IQ1_S": (0,),
    "IQ4_NL": (0,),
    "IQ3_S": (0,),
    "IQ2_S": (0,),
    "IQ4_XS": (0,),
    "TQ1_0": (52,),
    "TQ2_0": (64,),
    "Q1_0": (0,),
    "Q2_0": (0,),
}


def make_blocks(
    name,
    n_rows,
    blocks_per_row,
    rng,
    sanitize = True,
):
    """Random packed bytes (n_rows, row_bytes) for ggml type `name`, scales made finite if asked."""
    t = ref.GGML_TYPE_IDS[name]
    block, size = ref.GGML_BLOCK[t]
    n = n_rows * blocks_per_row
    raw = rng.integers(0, 256, size = (n, size), dtype = np.uint8)
    if sanitize:
        if name in _F16_FIELDS:
            for off in _F16_FIELDS[name]:
                vals = (rng.uniform(0.01, 2.0, n) * rng.choice([-1, 1], n)).astype(np.float16)
                raw[:, off : off + 2] = vals.view(np.uint8).reshape(n, 2)
        elif name == "IQ1_M":
            # fp16 scale is split over the top nibble of each of the 4 u16 scale words.
            bits = rng.uniform(0.01, 2.0, n).astype(np.float16).view(np.uint16).astype(np.uint16)
            words = raw[:, 48:56].copy().view(np.uint16).reshape(n, 4) & np.uint16(0x0FFF)
            for i in range(4):
                words[:, i] |= ((bits >> np.uint16(4 * i)) & np.uint16(0xF)) << np.uint16(12)
            raw[:, 48:56] = words.view(np.uint8).reshape(n, 8)
        elif name in ("MXFP4", "MXFP8"):
            raw[:, 0] = rng.integers(110, 140, n, dtype = np.uint8)
            if name == "MXFP8":
                q = raw[:, 1:]
                raw[:, 1:] = np.where((q & 0x7F) == 0x7F, q & 0xF0, q)
        elif name == "NVFP4":
            raw[:, :4] = rng.integers(0x08, 0x70, (n, 4), dtype = np.uint8)
        elif name == "F32":
            raw = rng.standard_normal(n).astype(np.float32).view(np.uint8).reshape(n, 4)
        elif name == "F16":
            raw = rng.standard_normal(n).astype(np.float16).view(np.uint8).reshape(n, 2)
        elif name == "BF16":
            f = rng.standard_normal(n).astype(np.float32).view(np.uint32) >> np.uint32(16)
            raw = f.astype(np.uint16).view(np.uint8).reshape(n, 2)
        elif name == "F8_E4M3":
            raw = np.where((raw & 0x7F) == 0x7F, raw & 0xF0, raw).astype(np.uint8)
    return raw.reshape(n_rows, blocks_per_row * size), (n_rows, blocks_per_row * block)


def bitexact_mismatch(expected, got):
    """Return (n_mismatch, max_abs_diff); NaNs equal NaNs, every other value compared by bits."""
    expected = torch.as_tensor(np.ascontiguousarray(expected)).to(torch.float32)
    got = got.to(torch.float32).cpu()
    assert expected.shape == got.shape, (expected.shape, got.shape)
    en, gn = torch.isnan(expected), torch.isnan(got)
    bad = (en != gn) | (~en & (expected.view(torch.int32) != got.view(torch.int32)))
    both = ~en & ~gn
    diff = (expected[both] - got[both]).abs()
    max_diff = float(diff.max()) if diff.numel() else 0.0
    return int(bad.sum()), max_diff


def _gguf_quants():
    try:
        import gguf
        import gguf.quants
    except Exception:
        return None
    return gguf


_GGUF = _gguf_quants()


def _oracle_supports(name):
    if _GGUF is None:
        return False
    qt = getattr(_GGUF.GGMLQuantizationType, name, None)
    if qt is None or int(qt) != ref.GGML_TYPE_IDS[name]:
        return False
    return name in ("F32", "F16") or qt in _GGUF.quants._type_traits


def _oracle(name, raw):
    qt = _GGUF.GGMLQuantizationType[name]
    with np.errstate(all = "ignore"):
        return _GGUF.quants.dequantize(raw, qt)


@pytest.mark.parametrize("sanitize", [True, False], ids = ["finite_scales", "raw_bytes"])
@pytest.mark.parametrize("name", list(ref.GGML_TYPE_IDS))
def test_bitexact_vs_gguf(name, sanitize):
    if _GGUF is None:
        pytest.skip("gguf package not installed")
    if not _oracle_supports(name):
        pytest.skip(
            f"installed gguf package has no dequantizer for {name} (id {ref.GGML_TYPE_IDS[name]})"
        )
    rng = np.random.default_rng(1234 + ref.GGML_TYPE_IDS[name] + (0 if sanitize else 7))
    block = ref.GGML_BLOCK[ref.GGML_TYPE_IDS[name]][0]
    blocks_per_row = 2 if block > 1 else 64
    for n_rows in (3, 16):
        raw, shape = make_blocks(name, n_rows, blocks_per_row, rng, sanitize)
        expected = _oracle(name, raw.copy())
        got = ref.dequantize_gguf(torch.from_numpy(raw), ref.GGML_TYPE_IDS[name], shape)
        assert got.dtype == torch.float32 and tuple(got.shape) == shape
        n_bad, max_diff = bitexact_mismatch(expected, got)
        assert n_bad == 0, f"{name}: {n_bad} mismatches, max abs diff {max_diff}"
        if sanitize:
            assert not torch.isnan(got).any()


@pytest.mark.parametrize("name", list(ref.GGML_TYPE_IDS))
def test_bf16_output_matches_cast(name):
    rng = np.random.default_rng(99)
    block = ref.GGML_BLOCK[ref.GGML_TYPE_IDS[name]][0]
    raw, shape = make_blocks(name, 3, 2 if block > 1 else 64, rng)
    packed = torch.from_numpy(raw)
    t = ref.GGML_TYPE_IDS[name]
    f32 = ref.dequantize_gguf(packed, t, shape)
    bf16 = ref.dequantize_gguf(packed, t, shape, dtype = torch.bfloat16)
    assert bf16.dtype == torch.bfloat16
    expected = f32.to(torch.bfloat16)
    assert torch.equal(torch.isnan(bf16), torch.isnan(expected))
    assert torch.equal(bf16.nan_to_num(), expected.nan_to_num())


@pytest.mark.parametrize("name", list(ref.GGML_TYPE_IDS))
def test_3d_expert_layout(name):
    t = ref.GGML_TYPE_IDS[name]
    block = ref.GGML_BLOCK[t][0]
    rng = np.random.default_rng(5)
    E, N, bpr = 2, 3, 2 if block > 1 else 64
    raw, (rows, cols) = make_blocks(name, E * N, bpr, rng)
    flat = ref.dequantize_gguf(torch.from_numpy(raw), t, (rows, cols))
    packed3 = torch.from_numpy(raw).reshape(E, N, ref.row_bytes(t, cols))
    out = ref.dequantize_gguf(packed3, t, (E, N, cols))
    assert tuple(out.shape) == (E, N, cols)
    assert torch.equal(out.reshape(rows, cols).nan_to_num(), flat.nan_to_num())


def test_tables_match_ids_and_sizes():
    assert ref.GGML_TYPE_NAMES == {v: k for k, v in ref.GGML_TYPE_IDS.items()}
    assert set(ref.GGML_BLOCK) == set(ref.GGML_TYPE_NAMES)
    for t in ref.GGML_TYPE_NAMES:
        assert ref.is_supported(t)
    if _GGUF is not None:
        for name, t in ref.GGML_TYPE_IDS.items():
            qt = getattr(_GGUF.GGMLQuantizationType, name, None)
            if qt is not None and int(qt) == t:
                assert tuple(_GGUF.GGML_QUANT_SIZES[qt]) == ref.GGML_BLOCK[t], name


def test_row_bytes():
    ids = ref.GGML_TYPE_IDS
    assert ref.row_bytes(ids["F32"], 7) == 28
    assert ref.row_bytes(ids["BF16"], 7) == 14
    assert ref.row_bytes(ids["Q4_0"], 64) == 36
    assert ref.row_bytes(ids["Q4_K"], 512) == 288
    assert ref.row_bytes(ids["Q6_K"], 256) == 210
    assert ref.row_bytes(ids["IQ1_M"], 256) == 56
    assert ref.row_bytes(ids["NVFP4"], 128) == 72
    assert ref.row_bytes(ids["MXFP8"], 32) == 33
    assert ref.row_bytes(ids["F8_E4M3"], 5) == 5
    with pytest.raises(ValueError, match = "Q4_K"):
        ref.row_bytes(ids["Q4_K"], 100)
    with pytest.raises(ValueError, match = "Q8_1"):
        ref.row_bytes(9, 32)


def test_unknown_type_error():
    assert not ref.is_supported(9)
    with pytest.raises(ValueError, match = r"Q8_K \(id 15\).*supported: .*Q4_K"):
        ref.dequantize_gguf(torch.zeros(292, dtype = torch.uint8), 15, (256,))
    with pytest.raises(ValueError, match = r"id 1000"):
        ref.dequantize_gguf(torch.zeros(4, dtype = torch.uint8), 1000, (4,))


def test_wrong_byte_count():
    with pytest.raises(ValueError, match = "bytes"):
        ref.dequantize_gguf(torch.zeros(17, dtype = torch.uint8), ref.GGML_TYPE_IDS["Q4_0"], (32,))


def test_chunked_path_matches(monkeypatch):
    rng = np.random.default_rng(3)
    raw, shape = make_blocks("Q4_K", 5, 3, rng)
    t = ref.GGML_TYPE_IDS["Q4_K"]
    full = ref.dequantize_gguf(torch.from_numpy(raw), t, shape)
    monkeypatch.setattr(ref, "_CHUNK_ELEMS", 256 * 4)
    chunked = ref.dequantize_gguf(torch.from_numpy(raw), t, shape, dtype = torch.float32)
    assert torch.equal(full, chunked)
