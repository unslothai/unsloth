# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Packed W4A8 / W6A8 (grouped-scale 4 / 6-bit codes, fp8 steps, channel scale, ConvRot): codec, writer, loader."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
nn = torch.nn

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
import core.inference.diffusion_grouped_int8 as gi  # noqa: E402
from core.inference.diffusion_convrot import build_convrot_hadamard  # noqa: E402

from .test_diffusion_comfy_quant import DIM, GROUP, _conf, _convert, _save, _sfm, _Tiny  # noqa: E402

# A 16-level table of the kind 4-bit files carry (sorted, inside [-1, 1]).
BOOK = torch.tensor(
    [
        -0.98,
        -0.79,
        -0.64,
        -0.5,
        -0.38,
        -0.26,
        -0.16,
        -0.05,
        0.05,
        0.16,
        0.27,
        0.38,
        0.5,
        0.64,
        0.79,
        0.98,
    ]
)


def _reference_int8(qdata, s_rel, codebook, group, bits):
    """Straight from the format description, one column at a time (independent of the vectorised decode)."""
    raw = qdata.view(torch.uint8).to(torch.int64)
    rows, groups = s_rel.shape
    k = groups * group
    out = torch.empty(rows, k, dtype = torch.int8)
    steps = s_rel.view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    for c in range(k):
        byte = raw[:, c // 2]
        code = (byte >> 4) & 0xF if c % 2 else byte & 0xF
        if bits == 6:
            code = code | (((raw[:, k // 2 + c // 4] >> (2 * (c % 4))) & 0x3) << 4)
        level = (
            codebook[code]
            if codebook is not None
            else code.to(torch.float32) - float(1 << (bits - 1))
        )
        out[:, c] = (
            (level.to(torch.float32) * steps[:, c // group]).round().clamp(-127, 127).to(torch.int8)
        )
    return out


@pytest.mark.parametrize("bits", [4, 6])
def test_pack_unpack_round_trip(bits):
    torch.manual_seed(0)
    codes = torch.randint(0, 1 << bits, (5, 64), dtype = torch.int32)
    packed = gi.pack_codes(codes, bits)
    assert packed.dtype == torch.int8 and tuple(packed.shape) == (5, 64 * bits // 8)
    assert torch.equal(gi.unpack_codes(packed, 64, bits), codes)


def test_bit_order_is_low_nibble_first_then_the_two_bit_plane():
    # column 0 = code 0x2A (low nibble 0xA, top bits 2), column 1 = 0x15, columns 2..31 = 0
    codes = torch.zeros(1, 32, dtype = torch.int32)
    codes[0, 0], codes[0, 1], codes[0, 5] = 0x2A, 0x15, 0x3F
    packed = gi.pack_codes(codes, 6).view(torch.uint8)[0]
    assert packed[0] == 0x5A  # col 1 low nibble (5) high, col 0 low nibble (A) low
    assert packed[2] == 0xF0  # col 5 is the high nibble of byte 2
    assert packed[16] == 0x2 | (0x1 << 2)  # top bits of cols 0..3 at bits 0, 2, 4, 6
    assert packed[17] == 0x3 << 2  # col 5 = byte 17 (5 // 4), bit 2 * (5 % 4)


@pytest.mark.parametrize(
    "bits, group, book", [(6, 16, None), (6, 32, None), (4, 16, BOOK), (4, 16, None)]
)
def test_decode_matches_the_format_reference_bit_for_bit(bits, group, book):
    torch.manual_seed(1)
    rows, k = 7, 512
    codes = torch.randint(0, 1 << bits, (rows, k), dtype = torch.int32)
    raw_steps = torch.randint(
        0, 0x7F, (rows, k // group), dtype = torch.uint8
    )  # every finite e4m3, subnormals too
    raw_steps[0, 0] = 0
    packed = gi.pack_codes(codes, bits)
    got = gi.decode_int8_torch(packed, raw_steps.view(torch.float8_e4m3fn), book, group, bits)
    assert torch.equal(got, _reference_int8(packed, raw_steps, book, group, bits))
    assert torch.equal(got, gi.decode_int8_torch(packed, raw_steps, book, group, bits))


def test_decode_rounds_ties_to_even():
    # level 1 (code 33) times steps 0.5 / 1.5 / 2.5: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
    steps = torch.tensor([[0.5, 1.5, 2.5, 1.0]]).to(torch.float8_e4m3fn)
    codes = torch.full((1, 64), 33, dtype = torch.int32)
    got = gi.decode_int8_torch(gi.pack_codes(codes, 6), steps, None, 16, 6)
    assert got[0, ::16].tolist() == [0, 2, 2, 1]


@pytest.mark.parametrize(
    "fmt, group, book", [(gi.W6A8, 16, None), (gi.W6A8, 32, None), (gi.ASYM_W4A8, 16, BOOK)]
)
def test_writer_is_round_to_nearest_on_the_decoded_grid(fmt, group, book):
    torch.manual_seed(2)
    weight = torch.randn(24, 512)
    weight[3] *= 40.0  # an outlier row: its own channel scale
    t = gi.quantize_grouped(weight, fmt, group_size = group, codebook = book)
    bits = gi.GROUPED_BITS[fmt]
    assert t[""].dtype == torch.int8 and tuple(t[""].shape) == (24, 512 * bits // 8)
    assert t[".weight_s_rel"].dtype == torch.float8_e4m3fn and tuple(t[".weight_s_rel"].shape) == (
        24,
        512 // group,
    )
    assert t[".weight_s_channel"].dtype == torch.float32 and tuple(
        t[".weight_s_channel"].shape
    ) == (24,)
    q8 = gi.decode_int8_torch(t[""], t[".weight_s_rel"], t.get(".weight_codebook"), group, bits)
    # every code is the nearest decoded level to the rotated weight / channel scale
    h = build_convrot_hadamard(GROUP)
    target = (weight.reshape(24, -1, GROUP) @ h).reshape(24, 512) / t[".weight_s_channel"][:, None]
    steps = t[".weight_s_rel"].float().repeat_interleave(group, dim = 1)
    table = gi.level_table(bits, t.get(".weight_codebook"))
    if bits == 6:
        table = table[1:]  # the writer is symmetric: code 0 (-32) is never written
    levels = (table.view(1, 1, -1) * steps.unsqueeze(-1)).round().clamp(-127, 127)
    best = (levels - target.unsqueeze(-1)).abs().min(dim = -1).values
    assert torch.allclose((q8.float() - target).abs(), best, atol = 1e-4)
    w = gi.dequant_grouped(
        t[""],
        t[".weight_s_rel"],
        t[".weight_s_channel"],
        t.get(".weight_codebook"),
        group,
        bits,
        GROUP,
        torch.float32,
    )
    rel = float((w - weight).norm() / weight.norm())
    assert rel < (0.04 if bits == 6 else 0.12), rel
    if bits == 6:
        codes = gi.unpack_codes(t[""], 512, 6)
        assert int(codes.min()) >= 1 and int(codes.max()) <= 63  # symmetric -31..31, code 0 unused


def test_dequant_equals_the_int8_convrot_dequant_of_the_decoded_grid():
    torch.manual_seed(3)
    weight = torch.randn(8, 512)
    t = gi.quantize_grouped(weight, gi.W6A8)
    q8 = gi.decode_int8_torch(t[""], t[".weight_s_rel"], None, 16, 6)
    a = gi.dequant_grouped(
        t[""], t[".weight_s_rel"], t[".weight_s_channel"], None, 16, 6, GROUP, torch.float32
    )
    b = cq._dequant(q8, t[".weight_s_channel"], GROUP, torch.float32)
    assert torch.equal(a, b)


def _grouped_file(
    tmp_path,
    fmt,
    *,
    group = 16,
    book = None,
    meta_table = False,
    name = "g.safetensors",
    **extra,
):
    torch.manual_seed(4)
    weight = torch.randn(16, 512)
    t = gi.quantize_grouped(weight, fmt, group_size = group, codebook = book)
    conf = gi.grouped_declaration(fmt, group)
    tensors = {f"a{k or '.weight'}": v for k, v in t.items()}
    tensors.update(extra)
    meta = None
    if meta_table:
        meta = {cq.QUANT_METADATA_KEY: json.dumps({"layers": {"a": conf}})}
    else:
        tensors["a.comfy_quant"] = gi.declaration_tensor(conf)
    return _save(tmp_path / name, tensors, meta), weight, t


@pytest.mark.parametrize(
    "fmt, group, book, table",
    [
        (gi.W6A8, 32, None, False),
        (gi.ASYM_W4A8, 16, BOOK, True),
        (gi.ASYM_W4A8, 16, None, False),
    ],
)
def test_scan_accepts_grouped_layers_as_int8_convrot(tmp_path, fmt, group, book, table):
    path, _w, _t = _grouped_file(tmp_path, fmt, group = group, book = book, meta_table = table)
    scan = cq.refuse_comfy_quant(path)
    layer = scan.layers["a"]
    assert (layer.format, layer.convrot, layer.group) == ("int8_tensorwise", True, 256)
    assert (layer.packed_bits, layer.group_size, layer.packed_format) == (
        gi.GROUPED_BITS[fmt],
        group,
        fmt,
    )
    assert scan.counts() == {fmt: 1}
    assert cq.comfy_prequant_scheme(scan) is None  # never mistaken for a hosted int8 prequant


def test_scan_fills_the_format_defaults(tmp_path):
    # {"format": "asym_w4a8_int8"} alone (as some files declare it): group 16, ConvRot group 256
    path, _w, t = _grouped_file(tmp_path, gi.ASYM_W4A8, book = BOOK)
    raw = safetensors_torch.load_file(path)
    raw["a.comfy_quant"] = _conf(format = gi.ASYM_W4A8)
    path = _save(tmp_path / "bare.safetensors", raw)
    layer = cq.refuse_comfy_quant(path).layers["a"]
    assert (layer.group_size, layer.group, layer.packed_bits) == (16, 256, 4)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda t: t.update({"a.weight_s_rel": t["a.weight_s_rel"][:, :-1].contiguous()}),
            "weight_s_rel of shape",
        ),
        (lambda t: t.pop("a.weight_s_channel"), "weight_s_channel"),
        (
            lambda t: t.update({"a.weight_scale": torch.ones(16)}),
            "weight_scale, which this format does not use",
        ),
        (lambda t: t.update({"a.weight": t["a.weight"][:, :-12].contiguous()}), "in_features 496"),
        (lambda t: t.update({"a.weight_codebook": torch.zeros(16)}), "codebook"),
    ],
)
def test_scan_refuses_a_malformed_grouped_layer(tmp_path, mutate, match):
    path, _w, _t = _grouped_file(tmp_path, gi.W6A8)
    raw = safetensors_torch.load_file(path)
    mutate(raw)
    path = _save(tmp_path / "bad.safetensors", raw)
    with pytest.raises(ValueError, match = match):
        cq.refuse_comfy_quant(path)


@pytest.fixture
def grouped_file(tmp_path, monkeypatch):
    """``_Tiny`` as a W6A8 file (fused qkv, as single files store it) plus the dense model it came from."""
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    torch.manual_seed(5)
    dense = _Tiny()
    tensors = {}
    for b, block in enumerate(dense.blocks):
        qkv = torch.cat([block.to_q.weight, block.to_k.weight, block.to_v.weight]).detach()
        layers = {
            f"blocks.{b}.qkv": (
                qkv,
                torch.cat([block.to_q.bias, block.to_k.bias, block.to_v.bias]),
            ),
            f"blocks.{b}.out": (block.to_out.weight.detach(), block.to_out.bias),
            f"blocks.{b}.adaLN_modulation.0": (
                block.adaLN_modulation[0].weight.detach(),
                block.adaLN_modulation[0].bias,
            ),
        }
        for name, (weight, bias) in layers.items():
            for suffix, value in gi.quantize_grouped(weight, gi.W6A8).items():
                tensors[f"{name}{suffix or '.weight'}"] = value
            tensors[f"{name}.bias"] = bias.detach().clone()
            tensors[f"{name}.comfy_quant"] = gi.declaration_tensor(gi.grouped_declaration(gi.W6A8))
    tensors["norm.weight"] = dense.norm.weight.detach().clone()
    tensors["norm.bias"] = dense.norm.bias.detach().clone()
    return _save(tmp_path / "tiny_w6a8.safetensors", tensors), dense, tensors


def _load(path, **kwargs):
    kwargs.setdefault("int8_backend", None)
    return cq.load_comfy_quant_transformer(
        _Tiny,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.float32, "config": "base/repo", "subfolder": "transformer"},
        family = "z-image",
        **kwargs,
    )


def _expected_dense(
    tensors,
    name,
    rows = None,
):
    w = gi.dequant_grouped(
        tensors[f"{name}.weight"],
        tensors[f"{name}.weight_s_rel"],
        tensors[f"{name}.weight_s_channel"],
        None,
        16,
        6,
        GROUP,
        torch.float32,
    )
    return w if rows is None else w[rows]


def test_dequant_fallback_loads_the_exact_reference_weights(grouped_file):
    path, dense, tensors = grouped_file
    model = _load(path, grouped_packed = False)
    info = model._unsloth_comfy_quant
    assert info["dequantized"] == 6 and info["packed_formats"] == [gi.W6A8]
    for b, block in enumerate(model.blocks):
        qkv = _expected_dense(tensors, f"blocks.{b}.qkv")
        assert torch.equal(block.to_q.weight, qkv[:DIM])
        assert torch.equal(block.to_k.weight, qkv[DIM : 2 * DIM])
        assert torch.equal(block.to_v.weight, qkv[2 * DIM :])
        assert torch.equal(block.to_out.weight, _expected_dense(tensors, f"blocks.{b}.out"))
        assert torch.equal(block.to_q.bias, dense.blocks[b].to_q.bias)
        rel = (block.to_out.weight - dense.blocks[b].to_out.weight).norm() / dense.blocks[
            b
        ].to_out.weight.norm()
        assert rel < 0.04


def test_packed_keep_holds_the_file_bytes_and_matches_the_dense_layer(grouped_file):
    path, dense, tensors = grouped_file
    model = _load(path, int8_backend = "native", grouped_packed = True)
    info = model._unsloth_comfy_quant
    # q / k / v / out of both blocks stay packed; the z-image int8 filter leaves adaLN dense, as for int8 files
    assert info["grouped_packed"] == 8 and info["dequantized"] == 2
    block = model.blocks[1]
    assert gi.is_grouped_linear(block.to_k)
    qkv = tensors["blocks.1.qkv.weight"]
    assert torch.equal(
        block.to_k.weight_packed, qkv[DIM : 2 * DIM]
    )  # the file's own bytes, row-split
    assert torch.equal(
        block.to_k.weight_s_rel,
        tensors["blocks.1.qkv.weight_s_rel"][DIM : 2 * DIM].view(torch.uint8),
    )
    assert torch.equal(block.to_k.weight, _expected_dense(tensors, "blocks.1.qkv")[DIM : 2 * DIM])
    x = torch.randn(3, DIM)
    want = torch.nn.functional.linear(
        x, _expected_dense(tensors, "blocks.1.qkv")[DIM : 2 * DIM], block.to_k.bias
    )
    assert torch.allclose(block.to_k(x), want, atol = 1e-4, rtol = 1e-4)  # CPU: the dense widen route
    assert gi.grouped_packed_layers(model) == info["grouped_packed"]


def test_packed_keep_survives_a_module_wide_dtype_cast(grouped_file):
    path, _dense, tensors = grouped_file
    model = _load(path, int8_backend = "native", grouped_packed = True)
    layer = model.blocks[0].to_out
    before = layer.weight_s_channel.clone()
    model.to(torch.bfloat16)
    assert layer.weight_s_channel.dtype == torch.int32 and torch.equal(
        layer.weight_s_channel, before
    )
    assert layer.compute_dtype == torch.bfloat16 and layer.rot_h.dtype == torch.bfloat16
    out = layer(torch.randn(2, DIM, dtype = torch.bfloat16))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


def test_packed_wanted_rule(monkeypatch):
    cuda = type("T", (), {"device": "cuda"})()
    cpu = type("T", (), {"device": "cpu"})()
    monkeypatch.delenv(gi.GROUPED_PACKED_ENV, raising = False)
    assert (
        gi.grouped_packed_wanted(cuda, "torchao") is True
    )  # video (the default caller) keeps packed codes
    assert gi.grouped_packed_wanted(cuda, None) is False  # no int8 runtime: decode at load
    assert gi.grouped_packed_wanted(cpu, "native") is False
    assert gi.grouped_packed_wanted(cuda, "torchao", lora = True) is False
    monkeypatch.setenv(gi.GROUPED_PACKED_ENV, "0")
    assert gi.grouped_packed_wanted(cuda, "torchao") is False
    monkeypatch.setenv(gi.GROUPED_PACKED_ENV, "1")
    assert gi.grouped_packed_wanted(cpu, None) is True
    assert gi.grouped_packed_wanted(cpu, None, lora = True) is False


def test_resident_estimate_prices_packed_int8_and_dense(tmp_path):
    path, _w, _t = _grouped_file(tmp_path, gi.W6A8, group = 32)
    path = str(path)
    packed = cq.comfy_resident_mib(path, keep_int8 = True, keep_fp8 = False, grouped_packed = True)
    unpacked = cq.comfy_resident_mib(path, keep_int8 = True, keep_fp8 = False)
    dense = cq.comfy_resident_mib(path, keep_int8 = False, keep_fp8 = False, compute_bytes = 4)
    assert (
        packed == unpacked == dense == 1
    )  # all under a MiB; the byte-level split is checked below
    scan = cq.scan_comfy_quant(path)
    assert scan.layers["a"].packed_bits == 6


def test_text_encoder_loader_decodes_grouped_layers(tmp_path):
    transformers = pytest.importorskip("transformers")
    import core.inference.diffusion_comfy_components as C

    cfg = transformers.Qwen3Config(
        vocab_size = 64,
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 128,
    )
    torch.manual_seed(6)
    ref = transformers.Qwen3Model(cfg).eval()
    out = {}
    for k, v in ref.state_dict().items():
        key = "model." + k
        if k.endswith("mlp.down_proj.weight"):  # K = 512
            stem = key[: -len(".weight")]
            t = gi.quantize_grouped(v, gi.ASYM_W4A8, codebook = BOOK)
            out.update({f"{stem}{s or '.weight'}": x for s, x in t.items()})
            out[f"{stem}.comfy_quant"] = gi.declaration_tensor(gi.grouped_declaration(gi.ASYM_W4A8))
            expected = gi.dequant_grouped(
                t[""],
                t[".weight_s_rel"],
                t[".weight_s_channel"],
                t[".weight_codebook"],
                16,
                4,
                GROUP,
                torch.float32,
            )
        else:
            out[key] = v
    path = _save(tmp_path / "qwen_3_w4a8.safetensors", out)
    enc = C.build_text_encoder(
        path, encoder_cls = transformers.Qwen3Model, config = cfg, dtype = torch.float32
    )
    down = enc.layers[0].mlp.down_proj
    assert getattr(enc, "_unsloth_te_prequant_scheme", None) in ("int8", None)
    x = torch.randn(3, 512)
    want = torch.nn.functional.linear(x, expected)
    got = down(x)
    assert torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim = 0) > 0.9999
    assert C.resident_bytes(path) > 0


def test_forced_packed_keep_without_an_int8_runtime(grouped_file):
    path, _dense, tensors = grouped_file
    model = _load(path, int8_backend = None, grouped_packed = True)
    assert model._unsloth_comfy_quant["grouped_packed"] == 8
    layer = model.blocks[0].to_out
    assert gi.is_grouped_linear(layer)
    x = torch.randn(2, DIM)
    want = torch.nn.functional.linear(x, _expected_dense(tensors, "blocks.0.out"), layer.bias)
    assert torch.allclose(layer(x), want, atol = 1e-4, rtol = 1e-4)
    assert gi.grouped_call_counts()["dense"] > 0


def test_exporter_keeps_source_metadata_and_writes_a_readable_file(tmp_path):
    src = {
        "blocks.0.q.weight": torch.randn(32, 512),
        "blocks.0.k.weight": torch.randn(32, 512),
        "blocks.0.k.bias": torch.randn(32),
        "head.weight": torch.randn(8, 100),  # K not a multiple of the rotation: stays as stored
    }
    meta = {"config": json.dumps({"transformer": {"pdd_steps": 6}})}
    path = _save(tmp_path / "src.safetensors", src, meta)
    out = str(tmp_path / "out.safetensors")
    res = gi.export_grouped_file(
        path,
        out,
        gi.ASYM_W4A8,
        layers = lambda n: True,
        codebook = BOOK,
        record = {"scheme": "asym_w4a8_int8"},
    )
    assert res["quantized"] == 2 and res["kept"] == ["head.weight"]
    from safetensors import safe_open

    with safe_open(out, framework = "pt") as handle:
        got_meta = handle.metadata()
    assert (
        got_meta["config"] == meta["config"]
        and json.loads(got_meta[cq.CONVERSION_RECORD_KEY])["scheme"]
    )
    scan = cq.refuse_comfy_quant(out)
    assert scan.counts() == {gi.ASYM_W4A8: 2}
    raw = safetensors_torch.load_file(out)
    assert torch.equal(raw["blocks.0.k.bias"], src["blocks.0.k.bias"])
    assert torch.equal(raw["head.weight"], src["head.weight"])


@pytest.mark.parametrize(
    "fmt, group", [(gi.ASYM_W4A8, 12), (gi.ASYM_W4A8, 2), (gi.W6A8, 8), (gi.W6A8, True)]
)
def test_writer_refuses_groups_the_reader_refuses(tmp_path, fmt, group):
    with pytest.raises(ValueError, match = "group_size"):
        gi.quantize_grouped(torch.randn(4, 512), fmt, group_size = group)
    path = _save(tmp_path / "s.safetensors", {"a.weight": torch.randn(4, 512)})
    with pytest.raises(ValueError, match = "group_size"):
        gi.export_grouped_file(
            path, str(tmp_path / "o.safetensors"), fmt, layers = ["a"], group_size = group
        )
    with pytest.raises(ValueError, match = "16 levels"):
        gi.quantize_grouped(torch.randn(4, 512), gi.ASYM_W4A8, codebook = torch.zeros(8))


def test_forced_packed_is_priced_packed_without_an_int8_runtime(tmp_path):
    torch.manual_seed(7)
    t = gi.quantize_grouped(torch.randn(2048, 2048), gi.W6A8)
    tensors = {f"a{k or '.weight'}": v for k, v in t.items()}
    tensors["a.comfy_quant"] = gi.declaration_tensor(gi.grouped_declaration(gi.W6A8))
    path = _save(tmp_path / "big.safetensors", tensors)
    kw = dict(keep_int8 = False, keep_fp8 = False, compute_bytes = 4)
    packed = cq.comfy_resident_mib(path, grouped_packed = True, **kw)
    dense = cq.comfy_resident_mib(path, **kw)
    int8 = cq.comfy_resident_mib(path, keep_int8 = True, keep_fp8 = False)
    assert packed == 4  # 3 MiB of codes + 256 KiB of steps + the row scales, rounded up
    assert int8 == 5 and dense == 17


def test_a_dequantized_layer_the_family_keeps_fp32_widens_to_fp32(grouped_file):
    # MiniMax-H3: a packed head the int8 filter skips (96 rows) dequantizes, and the family keeps it float32
    path, _dense, tensors = grouped_file
    model = cq.load_comfy_quant_transformer(
        _Tiny,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo", "subfolder": "transformer"},
        int8_backend = None,
        family = "z-image",
        grouped_packed = False,
        keep_dtype = lambda key: torch.float32 if key.startswith("blocks.0.out.") else None,
    )
    out = model.blocks[0].to_out
    assert out.weight.dtype == torch.float32 and out.bias.dtype == torch.float32
    assert torch.equal(out.weight, _expected_dense(tensors, "blocks.0.out"))
    assert model.blocks[1].to_out.weight.dtype == torch.bfloat16


def test_image_loads_decode_at_load_unless_memory_tight(monkeypatch):
    cuda = type("T", (), {"device": "cuda"})()
    cpu = type("T", (), {"device": "cpu"})()
    monkeypatch.delenv(gi.GROUPED_PACKED_ENV, raising = False)
    choice = gi.grouped_packed_choice
    monkeypatch.setattr(gi, "_tuned_arch", lambda device: True)
    assert choice(cuda, "torchao", image = True) == (False, "image_resident")
    assert choice(cuda, "native", image = True, tight = True) == (True, "memory_tight")
    assert choice(cuda, "torchao") == (True, "video")
    assert choice(cuda, None, image = True, tight = True) == (False, "no_int8_runtime")
    assert choice(cpu, "native", image = True, tight = True) == (False, "not_cuda")
    assert choice(cuda, "native", image = True, tight = True, lora = True) == (False, "lora")
    monkeypatch.setenv(gi.GROUPED_PACKED_ENV, "1")  # forced packed: images too
    assert choice(cuda, "torchao", image = True) == (True, "env_on")
    monkeypatch.setenv(gi.GROUPED_PACKED_ENV, "0")  # kill switch: video decodes too
    assert choice(cuda, "torchao") == (False, "env_off")
    assert choice(cuda, "native", image = True, tight = True) == (False, "env_off")


def test_status_reports_which_form_ran(grouped_file):
    path, _dense, _tensors = grouped_file
    packed = _load(path, int8_backend = "native", grouped_packed = True, grouped_reason = "memory_tight")
    form = gi.grouped_form_status(packed)
    assert form == {"form": "packed", "formats": [gi.W6A8], "layers": 8, "reason": "memory_tight"}
    assert packed._unsloth_comfy_quant["grouped_form"] == "packed"
    assert gi.grouped_form_note(packed) == "; W6A8 kept packed (8 layers, memory_tight)"
    decoded = _load(
        path, int8_backend = "native", grouped_packed = False, grouped_reason = "image_resident"
    )
    form = gi.grouped_form_status(decoded)
    assert form["form"] == "decoded_at_load" and form["reason"] == "image_resident"
    assert gi.grouped_packed_layers(decoded) == 0
    assert "decoded to int8 at load" in gi.grouped_form_note(decoded)
    dense = _load(path, int8_backend = None)  # no int8 runtime: the loader's own rule, bf16 weights
    form = gi.grouped_form_status(dense)
    assert form["form"] == "dequantized" and form["reason"] == "not_cuda"
    assert (
        gi.grouped_form_status(torch.nn.Linear(2, 2)) is None and gi.grouped_form_note(None) == ""
    )


def test_race_keeps_the_incumbent_inside_the_margin():
    win = gi.race_winner
    assert win({"int8": 10.0, "dense": 12.0}) == "int8"
    assert win({"int8": 10.0, "dense": 9.0}) == "dense"  # the incumbents race as before
    assert (
        win({"int8": 10.0, "dense": 12.0, "fused": 9.9}) == "int8"
    )  # 1% faster: inside the margin
    assert win({"int8": 10.0, "dense": 12.0, "fused": 9.7}) == "fused"  # 3% faster
    assert win({"int8": 10.0, "dense": 12.0, "fused": 9.7, "int8r": 9.0}) == "int8r"
    assert win({"fused": 3.0}) == "fused"  # an incumbent that could not run loses


def _cpu_layer(bits = 6):
    torch.manual_seed(0)
    k, n, g = 512, 64, 16
    codes = torch.randint(0, 1 << bits, (n, k), dtype = torch.int32)
    s_rel = torch.full((n, k // g), 0.25).to(torch.float8_e4m3fn)
    return gi.grouped_linear_class()(
        k,
        n,
        qdata = gi.pack_codes(codes, bits),
        s_rel = s_rel,
        s_channel = torch.rand(n) * 0.01,
        bits = bits,
        group_size = g,
        rotation = 256,
    )


def test_forward_dispatches_the_raced_route(monkeypatch):
    layer = _cpu_layer()
    calls = []
    for name in ("_int8", "_int8r", "_fused", "_dense"):
        monkeypatch.setattr(layer, name, lambda x, name = name: calls.append(name) or x[..., :64])
    monkeypatch.setattr(layer, "_int8_ok", lambda x: True)
    monkeypatch.setattr(layer, "_placed", lambda x: None)
    monkeypatch.setattr(layer, "_fused_covers", lambda x: True)
    x = torch.randn(32, 512, dtype = torch.bfloat16)
    layer._tuned, layer._rotq, layer._fused_ok = True, True, True
    for route, want in (
        ("fused", "_fused"),
        ("int8r", "_int8r"),
        ("int8", "_int8"),
        ("dense", "_dense"),
    ):
        calls.clear()
        monkeypatch.setattr(layer, "_race", lambda x, route = route: route)
        layer(x)
        assert calls == [want], route
    # a fused verdict whose bucket holds a row count no tile covers runs the int8 route, and is counted as such
    monkeypatch.setattr(layer, "_fused_covers", lambda x: False)
    before = gi.grouped_call_counts()
    calls.clear()
    monkeypatch.setattr(layer, "_race", lambda x: "fused")
    layer(x)
    assert calls == ["_int8"]
    after = gi.grouped_call_counts()
    assert after["int8"] - before["int8"] == 1 and after["fused"] == before["fused"]
    monkeypatch.setattr(layer, "_fused_covers", lambda x: True)
    # a route whose card decision went away (kill switch / failed probe) falls back to the int8 route
    layer._fused_ok = False
    calls.clear()
    monkeypatch.setattr(layer, "_race", lambda x: "fused")
    layer(x)
    assert calls == ["_int8"]
    layer._rotq = False
    calls.clear()
    monkeypatch.setattr(layer, "_race", lambda x: "int8r")
    layer(x)
    assert calls == ["_int8"]


def test_race_candidates_follow_the_card_decisions(monkeypatch):
    layer = _cpu_layer()
    monkeypatch.setattr(layer, "_fused_covers", lambda x: True)
    x = torch.randn(32, 512, dtype = torch.bfloat16)
    names = lambda: [r for r, _ in layer._routes(x)]  # noqa: E731
    layer._tuned, layer._rotq, layer._fused_ok = True, True, True
    assert names() == ["int8", "dense", "fused"]
    layer._tuned = False  # sm_8x on Triton >= 3.8: the rotation kernel feeds torch._int_mm
    assert names() == ["int8", "dense", "fused", "int8r"]
    layer._rotq = False  # no fused act quant: neither added route
    assert names() == ["int8", "dense"]
    layer._tuned, layer._rotq, layer._fused_ok = True, True, False
    assert names() == ["int8", "dense"]
    assert [r for r, _ in layer._routes(x.half())] == [
        "int8",
        "dense",
    ]  # added routes are bf16 only


def test_card_decisions_are_cached_and_honour_the_kill_switches(monkeypatch):
    import core.inference.diffusion_convrot_quant as cr

    gi._ROTQ.clear()
    probes = []
    monkeypatch.setattr(cr, "rotquant_device_config", lambda i: probes.append(i) or (128, 32, 8, 3))
    monkeypatch.delenv(cr.INT8_ROTQUANT_ENV, raising = False)
    assert gi._rotq_on(0) is True and gi._rotq_on(0) is True and probes == [0]
    monkeypatch.setenv(cr.INT8_ROTQUANT_ENV, "0")
    assert gi._rotq_on(1) is False and probes == [0]
    monkeypatch.setattr(cr, "rotquant_device_config", lambda i: None)
    monkeypatch.delenv(cr.INT8_ROTQUANT_ENV, raising = False)
    assert gi._rotq_on(2) is False
    gi._ROTQ.clear()
    gi._FUSED.clear()
    import sys
    import types

    fake = types.ModuleType("core.inference.diffusion_grouped_int8_gemm")
    fake.fused_config = lambda i: None if i else (64, 128, 128)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_grouped_int8_gemm", fake)
    assert gi._fused_on(0) is True and gi._fused_on(3) is False
    gi._FUSED.clear()


def test_load_time_card_decision_arms_every_route(monkeypatch):
    # the loader decides at load (a compiled forward never probes): all three, not only the tuned GEMM
    monkeypatch.setattr(gi, "_tuned_on", lambda i: True)
    monkeypatch.setattr(gi, "_rotq_on", lambda i: True)
    monkeypatch.setattr(gi, "_fused_on", lambda i: False)
    layer = _cpu_layer()
    assert (layer._tuned, layer._rotq, layer._fused_ok) == (None, None, None)
    gi.decide_on_card(layer, 0)
    assert (layer._tuned, layer._rotq, layer._fused_ok) == (True, True, False)
    monkeypatch.setattr(gi, "_rotq_on", lambda i: False)
    gi.decide_on_card(layer, 0)  # made once: a later call keeps the decision
    assert layer._rotq is True


def test_fused_route_only_where_a_tile_covers_the_rows(monkeypatch):
    layer = _cpu_layer()  # 6-bit, K = 512
    layer._tuned, layer._rotq, layer._fused_ok = True, True, True
    rule = (64, (64, 64, 128, 8, 4, 3, 2, 0), (6,))
    monkeypatch.setitem(gi._FUSED, None, (rule,))  # a CPU tensor's device index
    small = torch.randn(64, 512, dtype = torch.bfloat16)
    big = torch.randn(65, 512, dtype = torch.bfloat16)
    assert layer._fused_covers(small) and "fused" in [r for r, _ in layer._routes(small)]
    assert not layer._fused_covers(big) and "fused" not in [r for r, _ in layer._routes(big)]
    monkeypatch.setitem(
        gi._FUSED, None, ((64, rule[1], (4,)),)
    )  # a 4-bit-only rule skips a 6-bit layer
    assert not layer._fused_covers(small)
    monkeypatch.setitem(gi._FUSED, None, None)
    assert not layer._fused_covers(small)


def test_compiled_default_route_without_a_race(monkeypatch):
    layer = _cpu_layer()
    calls = []
    for name in ("_int8", "_int8r", "_fused", "_dense"):
        monkeypatch.setattr(layer, name, lambda x, name = name: calls.append(name) or x[..., :64])
    monkeypatch.setattr(layer, "_int8_ok", lambda x: True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x = torch.randn(32, 512, dtype = torch.bfloat16)
    layer._route = None
    for tuned, rotq, want in (
        (False, True, "_int8r"),
        (True, True, "_int8"),
        (False, False, "_int8"),
    ):
        layer._tuned, layer._rotq, layer._fused_ok = tuned, rotq, True
        calls.clear()
        layer(x)
        assert calls == [want], (tuned, rotq)
    layer._tuned, layer._rotq, layer._route = False, True, "fused"  # a fused verdict is eager-only
    calls.clear()
    layer(x)
    assert calls == ["_int8"]


def test_int8_epilogue_kernel_declines_what_it_does_not_cover():
    from core.inference.diffusion_grouped_int8_triton import int8_epilogue_triton

    acc = torch.zeros(32, 16, dtype = torch.int32)
    xs, ws = torch.ones(32), torch.ones(16)
    assert int8_epilogue_triton(acc, xs, ws, None) is None  # CPU: the eager reference runs
    assert int8_epilogue_triton(acc.float(), xs, ws, None) is None


def test_memory_tight_images_stay_packed_only_on_a_tuned_int8_arch(monkeypatch):
    import core.inference.diffusion_int8_gemm as g

    cuda = type("T", (), {"device": "cuda:0"})()
    monkeypatch.delenv(gi.GROUPED_PACKED_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap[0])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cap = [(8, 9)]
    monkeypatch.setattr(
        g, "_triton38_sm8x", lambda c: c[0] == 8
    )  # L4 on Triton >= 3.8: no tuned tile
    assert gi.grouped_packed_choice(cuda, "native", image = True, tight = True) == (
        False,
        "memory_tight_untuned_arch",
    )
    assert gi.grouped_packed_choice(cuda, "native") == (True, "video")  # video keeps packed codes
    cap[0] = (12, 0)  # RTX PRO 6000: tuned tile
    assert gi.grouped_packed_choice(cuda, "native", image = True, tight = True) == (
        True,
        "memory_tight",
    )
    cap[0] = (10, 0)  # B200: stock cuBLAS, no tuned tile
    assert gi.grouped_packed_choice(cuda, "native", image = True, tight = True)[0] is False
    monkeypatch.setenv(gi.GROUPED_PACKED_ENV, "1")  # forced packed still wins
    assert gi.grouped_packed_choice(cuda, "native", image = True, tight = True) == (True, "env_on")
