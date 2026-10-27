# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF -> MLX affine converter for the opt-in MLX image engine: numpy + gguf only, runs on Linux."""

import json
import struct
import threading

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf import quants  # noqa: E402

from core.inference import mlx_diffusion_convert as conv  # noqa: E402


def _rng(seed = 0):
    return np.random.default_rng(seed)


def _blocks(
    kind,
    rows,
    cols,
    seed = 0,
):
    x = (_rng(seed).standard_normal((rows, cols)) * 0.05).astype(np.float32)
    return quants.quantize(x, kind)


@pytest.mark.parametrize("kind", [T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0])
@pytest.mark.parametrize("cols", [64, 96])
def test_affine_matches_gguf_dequantize_exactly(kind, cols):
    raw = _blocks(kind, 6, cols, seed = cols)
    reference = quants.dequantize(raw, kind).astype(np.float32)
    w, s, b, bits = conv.affine_from_ggml(kind.name, raw, 6, cols)
    assert bits == {T.Q8_0: 8, T.Q5_0: 5, T.Q5_1: 5}.get(kind, 4)
    assert w.dtype == np.uint32 and w.shape == (6, cols * bits // 32)
    assert s.shape == b.shape == (6, cols // 32)
    got = conv.dequantize_affine(w, s, b, bits)
    assert np.array_equal(got, reference)
    # fp16 scales are lossless: d, m, -8d and -128d are all fp16 values
    s16 = s.astype(np.float16).astype(np.float32)
    b16 = b.astype(np.float16).astype(np.float32)
    assert np.array_equal(s16, s) and np.array_equal(b16, b)
    assert np.array_equal(conv.dequantize_affine(w, s16, b16, bits), reference)


@pytest.mark.parametrize("kind", [T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0])
def test_affine_bf16_scales_within_rounding_bound(kind):
    raw = _blocks(kind, 8, 128, seed = 3)
    reference = quants.dequantize(raw, kind).astype(np.float32)
    w, s, b, bits = conv.affine_from_ggml(kind.name, raw, 8, 128)
    sb, bb = conv.round_to_bf16(s), conv.round_to_bf16(b)
    got = conv.dequantize_affine(w, sb, bb, bits)
    q = conv._unpack(w, bits).astype(np.float32).reshape(8, 4, 32)
    bound = (q * np.abs(sb - s)[..., None] + np.abs(bb - b)[..., None]).reshape(8, 128)
    assert np.all(np.abs(got - reference) <= bound * 1.0001 + 1e-12)
    # and bf16 keeps ~8 bits: never worse than 2^-8 relative on the scale
    assert np.all(np.abs(sb - s) <= np.abs(s) * 2.0**-8)


def test_affine_rejects_16_wide_kquants_and_bad_sizes():
    for kind, size in (("Q2_K", 84), ("Q3_K", 110)):
        with pytest.raises(conv.UnsupportedGGUFError):
            conv.affine_from_ggml(kind, np.zeros(size, np.uint8), 1, 256)
    with pytest.raises(ValueError):
        conv.affine_from_ggml("Q4_0", np.zeros(17, np.uint8), 1, 32)
    with pytest.raises(ValueError):
        conv.affine_from_ggml("Q4_K", np.zeros(143, np.uint8), 1, 256)


# Field offsets of the 256-wide super-blocks (ggml-common.h block_q4_K / block_q5_K / block_q6_K).
_KQUANT_FP16_FIELDS = {"Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}


def _kquant_raw(
    kind,
    rows,
    cols,
    seed = 0,
):
    """Random K-quant blocks with finite, realistic fp16 super-block scales (gguf-py cannot quantize these)."""
    size = conv._BLOCK_BYTES[kind]
    rng = _rng(seed)
    raw = rng.integers(0, 256, size = (rows * cols // 256, size), dtype = np.uint8)
    for offset in _KQUANT_FP16_FIELDS[kind]:
        d = (rng.uniform(1e-4, 3e-3, size = raw.shape[0])).astype(np.float16)
        raw[:, offset : offset + 2] = d.view(np.uint8).reshape(-1, 2)
    return raw


@pytest.mark.parametrize("kind", ["Q4_K", "Q5_K"])
def test_kquant_codes_and_scales_reproduce_gguf_dequantize_exactly(kind):
    raw = _kquant_raw(kind, 4, 512, seed = 5)
    reference = quants.dequantize(raw, getattr(T, kind)).astype(np.float32).reshape(4, 512)
    w, s, b, bits = conv.affine_from_ggml(kind, raw, 4, 512)
    assert bits == (4 if kind == "Q4_K" else 5)
    assert w.shape == (4, 512 * bits // 32) and s.shape == b.shape == (4, 16)
    # Same fp32 products as gguf-py: exact before the 16-bit scale store.
    assert np.array_equal(conv.dequantize_affine(w, s, b, bits), reference)
    # The fp16 store rounds d * sc only: error bounded by the rounding of each group's scale and bias.
    s16 = s.astype(np.float16).astype(np.float32)
    b16 = b.astype(np.float16).astype(np.float32)
    q = conv._unpack(w, bits).astype(np.float32).reshape(4, 16, 32)
    bound = (q * np.abs(s16 - s)[..., None] + np.abs(b16 - b)[..., None]).reshape(4, 512)
    got = conv.dequantize_affine(w, s16, b16, bits)
    assert np.all(np.abs(got - reference) <= bound * 1.0001 + 1e-12)
    assert np.max(np.abs(got - reference)) <= 2.0**-10 * np.max(np.abs(reference))


def test_q6_k_requantizes_to_8_bits_within_a_quarter_of_its_own_step():
    raw = _kquant_raw("Q6_K", 4, 512, seed = 9)
    reference = quants.dequantize(raw, T.Q6_K).astype(np.float32).reshape(4, 512)
    w, s, b, bits = conv.affine_from_ggml("Q6_K", raw, 4, 512)
    assert bits == 8
    got = conv.dequantize_affine(w, s, b, bits)
    # Q6_K's own step per 16 values is |d * scale|; the 8-bit group-32 step over the same values is <= 64 / 255 of
    # the larger one, so the rounding error stays within half of that.
    blocks = raw.reshape(-1, 210)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
    sc = blocks[:, 192:208].view(np.int8).astype(np.float32)
    step = np.abs(d * sc).reshape(4, 32)
    group_step = np.maximum(step[:, 0::2], step[:, 1::2])
    err = np.abs(got - reference).reshape(4, 16, 32).max(axis = -1)
    assert np.all(err <= group_step * 0.25 + 1e-7)


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
def test_pack_matches_mlx_layout_bit_for_bit(bits):
    mx = pytest.importorskip("mlx.core")
    w = mx.array(_rng(bits).standard_normal((8, 256)).astype(np.float32))
    packed, scales, biases = mx.quantize(w, group_size = 32, bits = bits)
    deq = np.array(mx.dequantize(packed, scales, biases, group_size = 32, bits = bits))
    q = np.rint(
        (deq.reshape(8, 8, 32) - np.array(biases)[..., None]) / np.array(scales)[..., None]
    ).astype(np.int64)
    assert np.array_equal(conv._pack(q.reshape(8, 256), bits), np.array(packed))
    assert np.array_equal(conv._unpack(np.array(packed), bits), q.reshape(8, 256))


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_dense_roundtrip(bits):
    w = (_rng(1).standard_normal((5, 64)) * 0.1).astype(np.float32)
    packed, s, b = conv.quantize_dense_affine(w, bits)
    assert packed.shape == (5, 64 * bits // 32) and s.shape == b.shape == (5, 2)
    assert np.array_equal(conv.round_to_bf16(s), s) and np.array_equal(conv.round_to_bf16(b), b)
    got = conv.dequantize_affine(packed, s, b, bits)
    err = np.abs(got - w).reshape(5, 2, 32)
    # half a step from rounding, plus the top edge the bf16 rounding of scale may shave off
    slack = (s * 2.0**-8 * ((1 << bits) - 1) + np.abs(b) * 2.0**-8)[..., None]
    assert np.all(err <= s[..., None] / 2 + slack + 1e-7)


def test_quantize_dense_constant_group():
    w = np.full((2, 32), 0.25, np.float32)
    w[1] = -3.0
    packed, s, b = conv.quantize_dense_affine(w, 4)
    assert np.all(s == 0)
    assert np.array_equal(conv.dequantize_affine(packed, s, b, 4), w)


@pytest.mark.parametrize(
    "src, dst",
    [
        ("final_layer.adaLN_modulation.1.weight", "all_final_layer.2-1.adaLN_modulation.0.weight"),
        ("final_layer.linear.weight", "all_final_layer.2-1.linear.weight"),
        ("x_embedder.weight", "all_x_embedder.2-1.weight"),
        ("t_embedder.mlp.0.weight", "t_embedder.linear1.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.linear2.bias"),
        ("layers.3.attention.out.weight", "layers.3.attention.to_out.0.weight"),
        ("layers.3.attention.q_norm.weight", "layers.3.attention.norm_q.weight"),
        ("layers.3.attention.k_norm.weight", "layers.3.attention.norm_k.weight"),
        ("layers.3.attention.qkv.weight", "layers.3.attention.qkv.weight"),
        ("layers.3.feed_forward.w1.weight", "layers.3.feed_forward.w1.weight"),
    ],
)
def test_zimage_rename(src, dst):
    assert conv.zimage_rename(src) == dst


@pytest.mark.parametrize(
    "name, ok",
    [
        ("z-image-turbo-Q4_0.gguf", True),
        ("z-image-turbo-Q4_1.gguf", True),
        ("z-image-turbo-Q8_0.gguf", True),
        ("z-image-turbo-BF16.gguf", True),
        ("z-image-turbo-F16.gguf", True),
        ("z_image_turbo-Q4_K_M.gguf", True),
        ("z-image-turbo-Q4_K_S.gguf", True),
        ("z-image-turbo-Q5_K_M.gguf", True),
        ("z-image-turbo-Q5_K_S.gguf", True),
        ("z-image-turbo-Q6_K.gguf", True),
        ("z-image-turbo-Q5_0.gguf", True),
        ("z-image-turbo-Q5_1.gguf", True),
        ("z-image-turbo-UD-Q4_K_XL.gguf", False),
        ("z-image-turbo-Q3_K_M.gguf", False),
        ("z-image-turbo-Q2_K.gguf", False),
        ("z-image-turbo-IQ4_XS.gguf", False),
        ("z-image-turbo-MXFP4.gguf", False),
        ("z-image-turbo.gguf", False),
        (None, False),
    ],
)
def test_quant_convertible(name, ok):
    verdict, reason = conv.quant_convertible(name)
    assert verdict is ok
    assert (reason is None) is ok
    if not ok:
        assert "MLX" in reason


def test_quant_convertible_names_the_kquant():
    _ok, reason = conv.quant_convertible("z_image_turbo-Q3_K_M.gguf")
    assert "Q3_K_M" in reason and "not convertible" in reason


# ── End to end on a tiny synthetic GGUF ─────────────────────────────────────────────────────────


def _read_safetensors(path):
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    body = data[8 + n :]
    meta = header.pop("__metadata__")
    np_dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16}
    tensors = {}
    end = 0
    for name, info in sorted(header.items(), key = lambda kv: kv[1]["data_offsets"][0]):
        a, b = info["data_offsets"]
        assert a == end, "offsets must be contiguous"
        end = b
        tensors[name] = (
            info["dtype"],
            np.frombuffer(body[a:b], dtype = np_dtype[info["dtype"]]).reshape(info["shape"]),
        )
    assert end == len(body)
    return meta, tensors


def _bf16(bits16):
    return conv._bf16_bits_to_f32(bits16)


def _write_gguf(path, extra = None):
    rng = _rng(7)
    sources = {}
    w = gguf.GGUFWriter(str(path), "z-image")

    def add_q(name, kind, rows, cols):
        x = (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)
        raw = quants.quantize(x, kind)
        sources[name] = (kind, raw)
        w.add_tensor(name, raw, raw_dtype = kind)

    add_q("layers.0.attention.qkv.weight", T.Q4_0, 3 * 64, 64)
    add_q("layers.0.feed_forward.w1.weight", T.Q8_0, 96, 64)
    add_q("layers.0.attention.out.weight", T.Q4_1, 64, 64)
    dense = (rng.standard_normal((32, 64)) * 0.05).astype(np.float32)
    sources["context_refiner.0.feed_forward.w2.weight"] = (T.BF16, dense)
    w.add_tensor(
        "context_refiner.0.feed_forward.w2.weight",
        quants.quantize(dense, T.BF16),
        raw_dtype = T.BF16,
    )
    norm = rng.standard_normal(64).astype(np.float32)
    sources["layers.0.attention.q_norm.weight"] = (T.F32, norm)
    w.add_tensor("layers.0.attention.q_norm.weight", norm)
    pad = rng.standard_normal(64).astype(np.float32)
    sources["cap_pad_token"] = (T.F32, pad)
    w.add_tensor("cap_pad_token", pad)
    if extra:
        extra(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return sources


def test_convert_end_to_end(tmp_path):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    sources = _write_gguf(src)
    seen = []
    summary = conv.convert_gguf_to_mflux(
        src, tmp_path / "out", progress = lambda d, t: seen.append((d, t))
    )
    assert summary["counts"] == {"Q4_0": 1, "Q8_0": 1, "Q4_1": 1, "BF16": 1, "F32": 2}
    assert seen[-1] == (6, 6)
    out = tmp_path / "out" / "transformer" / "0.safetensors"
    meta, tensors = _read_safetensors(out)
    assert meta["quantization_level"] == "4"
    assert meta["mflux_version"] and meta["source_gguf"] == src.name
    assert meta["unsloth_converter"] == conv.CONVERTER_VERSION
    names = set(tensors)
    for p in ("to_q", "to_k", "to_v"):
        for suffix in ("weight", "scales", "biases"):
            assert f"layers.0.attention.{p}.{suffix}" in names
    assert not any(".qkv." in n for n in names)
    assert "layers.0.attention.to_out.0.scales" in names
    assert "layers.0.attention.norm_q.weight" in names
    kind, raw = sources["layers.0.attention.qkv.weight"]
    ref = quants.dequantize(raw, kind).astype(np.float32)
    for i, p in enumerate(("to_q", "to_k", "to_v")):
        dt, packed = tensors[f"layers.0.attention.{p}.weight"]
        assert dt == "U32" and packed.shape == (64, 8)
        s = _bf16(tensors[f"layers.0.attention.{p}.scales"][1])
        b = _bf16(tensors[f"layers.0.attention.{p}.biases"][1])
        got = conv.dequantize_affine(packed, s, b, 4)
        np.testing.assert_allclose(got, ref[i * 64 : (i + 1) * 64], atol = 2e-3, rtol = 0)
    # Q8_0 packs 4 per word
    assert tensors["layers.0.feed_forward.w1.weight"][1].shape == (96, 16)
    # dense BF16 stays bit-identical; F32 is rounded to bf16; pad token is (1, -1)
    dt, bits16 = tensors["context_refiner.0.feed_forward.w2.weight"]
    assert dt == "BF16" and bits16.shape == (32, 64)
    np.testing.assert_array_equal(
        _bf16(bits16), conv.round_to_bf16(sources["context_refiner.0.feed_forward.w2.weight"][1])
    )
    assert tensors["cap_pad_token"][1].shape == (1, 64)
    np.testing.assert_array_equal(
        _bf16(tensors["layers.0.attention.norm_q.weight"][1]),
        conv.round_to_bf16(sources["layers.0.attention.q_norm.weight"][1]),
    )
    marker = json.loads((out.parent / conv.MARKER_FILENAME).read_text())
    assert marker["output_bytes"] == out.stat().st_size and marker["dense_bits"] == 0
    assert conv.conversion_complete(tmp_path / "out")
    assert conv.conversion_complete(tmp_path / "out", source_path = src)
    assert not list(out.parent.glob(".*tmp"))


def test_convert_f16_scales_are_exact(tmp_path):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    sources = _write_gguf(src)
    conv.convert_gguf_to_mflux(src, tmp_path / "out", scale_dtype = "f16")
    _meta, tensors = _read_safetensors(tmp_path / "out" / "transformer" / "0.safetensors")
    kind, raw = sources["layers.0.attention.out.weight"]
    dt, packed = tensors["layers.0.attention.to_out.0.weight"]
    s = tensors["layers.0.attention.to_out.0.scales"]
    b = tensors["layers.0.attention.to_out.0.biases"]
    assert s[0] == b[0] == "F16"
    got = conv.dequantize_affine(packed, s[1].astype(np.float32), b[1].astype(np.float32), 4)
    assert np.array_equal(got, quants.dequantize(raw, kind).astype(np.float32))


def test_convert_dense_bits_quantizes_dense_linears(tmp_path):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    sources = _write_gguf(src)
    summary = conv.convert_gguf_to_mflux(src, tmp_path / "out", dense_bits = 4)
    assert summary["dense_bits"] == 4
    _meta, tensors = _read_safetensors(tmp_path / "out" / "transformer" / "0.safetensors")
    base = "context_refiner.0.feed_forward.w2"
    assert tensors[base + ".weight"][0] == "U32"
    got = conv.dequantize_affine(
        tensors[base + ".weight"][1],
        _bf16(tensors[base + ".scales"][1]),
        _bf16(tensors[base + ".biases"][1]),
        4,
    )
    dense = sources[base + ".weight"][1]
    assert np.max(np.abs(got - dense)) < 0.05
    # norms (1-D) and pad tokens stay dense
    assert tensors["layers.0.attention.norm_q.weight"][0] == "BF16"
    assert tensors["cap_pad_token"][0] == "BF16"


def test_convert_refuses_kquant_before_writing(tmp_path):
    src = tmp_path / "z-image-turbo-Q3_K_M.gguf"

    def extra(w):
        raw = _rng(2).integers(0, 255, size = (4, 110), dtype = np.uint8)
        w.add_tensor("layers.0.feed_forward.w2.weight", raw, raw_dtype = T.Q3_K)

    _write_gguf(src, extra = extra)
    with pytest.raises(conv.UnsupportedGGUFError, match = "Q3_K") as err:
        conv.convert_gguf_to_mflux(src, tmp_path / "out")
    supported = str(err.value).split("(supported: ", 1)[1].rstrip(")").split(", ")
    assert set(supported) == conv.CONVERTIBLE_TYPES
    assert not (tmp_path / "out" / "transformer" / "0.safetensors").exists()
    assert not conv.conversion_complete(tmp_path / "out")


def test_convert_cancel_leaves_no_file(tmp_path):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    _write_gguf(src)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match = "cancelled"):
        conv.convert_gguf_to_mflux(src, tmp_path / "out", cancel_event = cancel)
    target = tmp_path / "out" / "transformer"
    assert not (target / "0.safetensors").exists()
    assert not list(target.glob(".*"))


def test_conversion_complete_detects_truncation_and_source_change(tmp_path):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    _write_gguf(src)
    conv.convert_gguf_to_mflux(src, tmp_path / "out")
    out = tmp_path / "out" / "transformer" / "0.safetensors"
    with open(out, "ab") as fh:
        fh.write(b"x")
    assert not conv.conversion_complete(tmp_path / "out")
    conv.convert_gguf_to_mflux(src, tmp_path / "out")
    assert conv.conversion_complete(tmp_path / "out", source_path = src)
    with open(src, "ab") as fh:
        fh.write(b"\0" * 32)
    assert not conv.conversion_complete(tmp_path / "out", source_path = src)


@pytest.mark.parametrize("bits", [0, 8, 4])
def test_planned_output_bytes_is_the_converted_size(tmp_path, bits):
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    _write_gguf(src)
    planned = conv.planned_output_bytes(src)
    summary = conv.convert_gguf_to_mflux(src, tmp_path / f"out{bits}", dense_bits = bits)
    assert planned[bits] == summary["bytes"]
    assert planned[0] > planned[8] > planned[4]


def test_planned_output_bytes_refuses_what_conversion_refuses(tmp_path):
    src = tmp_path / "z-image-turbo-Q3_K_M.gguf"

    def extra(w):
        raw = _rng(2).integers(0, 255, size = (4, 110), dtype = np.uint8)
        w.add_tensor("layers.0.feed_forward.w2.weight", raw, raw_dtype = T.Q3_K)

    _write_gguf(src, extra = extra)
    with pytest.raises(conv.UnsupportedGGUFError, match = "Q3_K"):
        conv.planned_output_bytes(src)


def test_mflux_loads_a_mixed_4_and_8_bit_conversion_at_each_layers_bits(tmp_path):
    """The 8-bit dense rung beside 4-bit GGUF blocks: mflux must rebuild each layer at its stored bits."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    applier = pytest.importorskip("mflux.models.common.weights.loading.weight_applier")
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    _write_gguf(src)
    conv.convert_gguf_to_mflux(src, tmp_path / "out", dense_bits = 8)
    loaded = mx.load(str(tmp_path / "out" / "transformer" / "0.safetensors"))

    class Ref(nn.Module):
        def __init__(self, rows, cols):
            super().__init__()
            self.lin = nn.Linear(cols, rows, bias = False)

    layers = {
        "layers.0.attention.to_out.0": (64, 64, 4),
        "layers.0.feed_forward.w1": (96, 64, 8),
        "context_refiner.0.feed_forward.w2": (32, 64, 8),
    }
    x = mx.array(_rng(11).standard_normal((3, 64)).astype(np.float32))
    for path, (rows, cols, bits) in layers.items():
        module = nn.Linear(cols, rows, bias = False)
        layer_weights = {k: loaded[f"{path}.{k}"] for k in ("weight", "scales", "biases")}
        predicate = applier.WeightApplier._stored_layer_predicate({"m": layer_weights}, None)
        holder = Ref(rows, cols)
        holder.lin = module
        nn.quantize(holder, class_predicate = lambda p, m: predicate("m", m), bits = 4, group_size = 64)
        assert holder.lin.bits == bits and holder.lin.group_size == 32
        holder.lin.update(layer_weights)
        got = np.array(holder.lin(x).astype(mx.float32))
        ref_w = conv.dequantize_affine(
            np.array(layer_weights["weight"]),
            np.array(layer_weights["scales"].astype(mx.float32)),
            np.array(layer_weights["biases"].astype(mx.float32)),
            bits,
        )
        np.testing.assert_allclose(got, np.array(x) @ ref_w.T, rtol = 2e-2, atol = 2e-2)


def test_cache_dir_for(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    d = conv.cache_dir_for(
        "unsloth/Z-Image-Turbo-GGUF", "abc123", "z-image-turbo-Q4_0.gguf", dense_bits = 4
    )
    assert d.parent.name == "unsloth--Z-Image-Turbo-GGUF@abc123"
    assert d.name == "z-image-turbo-Q4_0-d4-bf16"
    assert d.parent.parent.name == "mlx_diffusion"
    assert str(d).startswith(str(tmp_path.resolve())) or str(d).startswith(str(tmp_path))
    local = conv.cache_dir_for("../x", None, "sub/../m.gguf", dense_bits = 0)
    assert ".." not in local.parts and local.parent.name.endswith("@local")


def test_mlx_loads_the_output_when_available(tmp_path):
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "z-image-turbo-Q4_0.gguf"
    sources = _write_gguf(src)
    conv.convert_gguf_to_mflux(src, tmp_path / "out")
    loaded, meta = mx.load(
        str(tmp_path / "out" / "transformer" / "0.safetensors"), return_metadata = True
    )
    assert meta["quantization_level"] == "4"
    name = "layers.0.attention.to_out.0"
    got = mx.dequantize(
        loaded[name + ".weight"],
        loaded[name + ".scales"],
        loaded[name + ".biases"],
        group_size = 32,
        bits = 4,
    ).astype(mx.float32)
    kind, raw = sources["layers.0.attention.out.weight"]
    np.testing.assert_allclose(np.array(got), quants.dequantize(raw, kind), atol = 2e-3, rtol = 0)


def test_a_q4_k_m_mixture_converts_and_mlx_reads_every_layer(tmp_path):
    """Q4_K + Q5_K + Q6_K + Q5_0 beside the Q4_0 / Q8_0 / BF16 tensors, read back with MLX's own dequantize."""
    mx = pytest.importorskip("mlx.core")
    src = tmp_path / "z-image-turbo-Q4_K_M.gguf"
    extra_sources = {}

    def extra(w):
        for name, kind, seed in (
            ("layers.1.feed_forward.w1.weight", "Q4_K", 1),
            ("layers.1.feed_forward.w2.weight", "Q5_K", 2),
            ("layers.1.feed_forward.w3.weight", "Q6_K", 3),
        ):
            raw = _kquant_raw(kind, 4, 256, seed = seed)
            extra_sources[name] = (kind, raw)
            w.add_tensor(name, raw.reshape(4, -1), raw_dtype = getattr(T, kind))
        x = (_rng(4).standard_normal((4, 64)) * 0.05).astype(np.float32)
        raw = quants.quantize(x, T.Q5_0)
        extra_sources["layers.1.attention.out.weight"] = ("Q5_0", raw)
        w.add_tensor("layers.1.attention.out.weight", raw, raw_dtype = T.Q5_0)

    _write_gguf(src, extra = extra)
    summary = conv.convert_gguf_to_mflux(src, tmp_path / "out")
    assert {"Q4_K", "Q5_K", "Q6_K", "Q5_0"} <= set(summary["counts"])
    loaded, meta = mx.load(
        str(tmp_path / "out" / "transformer" / "0.safetensors"), return_metadata = True
    )
    assert meta["quantization_level"] not in (None, "None")
    renamed = {
        "layers.1.feed_forward.w1.weight": ("layers.1.feed_forward.w1", 4),
        "layers.1.feed_forward.w2.weight": ("layers.1.feed_forward.w2", 5),
        "layers.1.feed_forward.w3.weight": ("layers.1.feed_forward.w3", 8),
        "layers.1.attention.out.weight": ("layers.1.attention.to_out.0", 5),
    }
    for name, (kind, raw) in extra_sources.items():
        base, bits = renamed[name]
        got = mx.dequantize(
            loaded[base + ".weight"],
            loaded[base + ".scales"],
            loaded[base + ".biases"],
            group_size = 32,
            bits = bits,
        ).astype(mx.float32)
        reference = (
            quants.dequantize(raw, getattr(T, kind)).astype(np.float32).reshape(np.array(got).shape)
        )
        # bf16 scales and biases, and MLX dequantizes into bf16: a few bf16 ulps of the largest weight (a layout
        # error would be off by the weight itself).
        tol = 2.0**-6 * np.max(np.abs(reference))
        np.testing.assert_allclose(np.array(got), reference, atol = tol, rtol = 0)
