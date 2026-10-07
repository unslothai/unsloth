# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ComfyUI ``nvfp4`` / ``mxfp8`` single files (``diffusion_comfy_block.py`` + the loader).

Synthetic files are written in ComfyUI's on-disk layout by an encoder here that shares no code with the
module under test: nvfp4 codes packed EVEN column in the HIGH nibble, block scales placed by the cuBLAS
128x4 tile formula (offset computed per element, not by a reshape). The reference dequant reads every
value back through that same formula, so a layout mistake in either direction fails bit-exactly.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
nn = torch.nn

import core.inference.diffusion_comfy_block as cb  # noqa: E402
import core.inference.diffusion_comfy_quant as cq  # noqa: E402

DIM = 512
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _conf(**conf) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


def _save(
    path,
    tensors,
    metadata = None,
) -> str:
    safetensors_torch.save_file(
        {k: v.contiguous().clone() for k, v in tensors.items()}, str(path), metadata = metadata
    )
    return str(path)


def _ceil(x: int, m: int) -> int:
    return -(-x // m) * m


def _tile_offsets(rows: int, blocks: int) -> torch.Tensor:
    """Flat offset of scale (r, b) in the cuBLAS 1D block-scaling-factor layout, element by element."""
    r = torch.arange(rows).view(-1, 1)
    b = torch.arange(blocks).view(1, -1)
    tiles_per_row = _ceil(blocks, 4) // 4
    return (
        (r // 128) * tiles_per_row * 512
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (b % 4)
    )


def _tile(plain_u8: torch.Tensor) -> torch.Tensor:
    rows, blocks = plain_u8.shape
    flat = torch.zeros(_ceil(rows, 128) * _ceil(blocks, 4), dtype = torch.uint8)
    flat[_tile_offsets(rows, blocks).reshape(-1)] = plain_u8.reshape(-1)
    return flat.reshape(_ceil(rows, 128), _ceil(blocks, 4))


def _untile(tiled: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    return tiled.reshape(-1)[_tile_offsets(rows, blocks)]


def nvfp4_encode(w: torch.Tensor) -> dict:
    """ComfyUI nvfp4 layout of ``w`` (round-to-nearest e2m1; exactness of the encoder is not under test)."""
    rows, cols = w.shape
    w = w.float()
    tensor_scale = w.abs().amax() / (448.0 * 6.0)
    blocks = w.reshape(rows, cols // 16, 16)
    block_scale = (blocks.abs().amax(-1) / 6.0 / tensor_scale).to(torch.float8_e4m3fn)
    step = block_scale.float() * tensor_scale
    x = blocks / torch.where(step == 0, torch.ones_like(step), step).unsqueeze(-1)
    nibble = (x.unsqueeze(-1) - E2M1[:8].abs().mul(x.sign().unsqueeze(-1))).abs().argmin(-1)
    nibble = torch.where(x < 0, nibble + 8, nibble).reshape(rows, cols).to(torch.uint8)
    packed = (nibble[:, 0::2] << 4) | nibble[:, 1::2]  # even column -> high nibble
    return {
        "weight": packed,
        "weight_scale": _tile(block_scale.view(torch.uint8)).view(torch.float8_e4m3fn),
        "weight_scale_2": tensor_scale.reshape(()),
        "input_scale": torch.tensor(0.01),
    }


def nvfp4_reference(t: dict) -> torch.Tensor:
    q = t["weight"]
    rows, cols = q.shape[0], q.shape[1] * 2
    j = torch.arange(cols)
    byte = q[:, j // 2].to(torch.int64)
    nibble = torch.where(j % 2 == 0, byte >> 4, byte & 15)
    scales = (
        _untile(t["weight_scale"].view(torch.uint8), rows, cols // 16)
        .view(torch.float8_e4m3fn)
        .float()
    )
    step = scales * t["weight_scale_2"].float()
    return E2M1[nibble] * step[:, j // 16]


def mxfp8_encode(w: torch.Tensor) -> dict:
    rows, cols = w.shape
    blocks = w.float().reshape(rows, cols // 32, 32)
    exp = torch.ceil(torch.log2((blocks.abs().amax(-1) / 448.0).clamp(min = 2.0**-127)))
    codes = (
        (blocks / torch.exp2(exp).unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(rows, cols)
    )
    return {
        "weight": codes,
        "weight_scale": _tile((exp + 127).to(torch.uint8)).view(torch.float8_e8m0fnu),
    }


def mxfp8_reference(t: dict) -> torch.Tensor:
    q = t["weight"]
    rows, cols = q.shape
    exp = _untile(t["weight_scale"].view(torch.uint8), rows, cols // 32).float() - 127
    return q.float() * torch.exp2(exp)[:, torch.arange(cols) // 32]


ENCODERS = {"nvfp4": (nvfp4_encode, nvfp4_reference), "mxfp8": (mxfp8_encode, mxfp8_reference)}


@pytest.mark.parametrize("rows,blocks", [(128, 4), (200, 6), (3, 1), (384, 120)])
def test_tiling_matches_the_cublas_formula_and_round_trips(rows, blocks):
    plain = torch.randint(0, 255, (rows, blocks), dtype = torch.uint8)
    tiled = cb.tile_scales(plain)
    assert torch.equal(tiled, _tile(plain).reshape(-1))
    assert torch.equal(cb.untile_scales(tiled, rows, blocks), plain)
    assert cb.tiled_scale_numel(rows, blocks) == tiled.numel()


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("rows", [256, 200])
def test_dequant_is_bit_exact_against_the_reference(fmt, rows):
    torch.manual_seed(0)
    encode, reference = ENCODERS[fmt]
    t = encode(torch.randn(rows, 96) * 0.05)
    codes, scale = cb.decode_layer(fmt, t["weight"], t["weight_scale"])
    got = cb.dequant_block(
        fmt, codes, scale, tensor_scale = t.get("weight_scale_2"), dtype = torch.float32
    )
    assert torch.equal(got, reference(t))


def test_nvfp4_codes_are_repacked_low_nibble_first():
    t = nvfp4_encode(torch.randn(128, 64))
    codes, _scale = cb.decode_layer("nvfp4", t["weight"], t["weight_scale"])
    assert torch.equal(codes & 0x0F, t["weight"] >> 4)
    assert torch.equal(codes >> 4, t["weight"] & 0x0F)


def test_pre_quant_scale_is_folded_into_the_columns():
    t = nvfp4_encode(torch.randn(128, 64))
    codes, scale = cb.decode_layer("nvfp4", t["weight"], t["weight_scale"])
    smooth = torch.rand(64) + 0.5
    got = cb.dequant_block(
        "nvfp4",
        codes,
        scale,
        tensor_scale = t["weight_scale_2"],
        pre_quant_scale = smooth,
        dtype = torch.float32,
    )
    assert torch.equal(got, nvfp4_reference(t) * smooth)


def _layer_file(
    tmp_path,
    fmt,
    rows = 128,
    cols = 64,
    drop = (),
    extra = None,
    conf = None,
    name = "m",
):
    encode, _ref = ENCODERS[fmt]
    t = encode(torch.randn(rows, cols))
    tensors = {f"a.{k}": v for k, v in t.items() if k not in drop}
    tensors.update(extra or {})
    tensors["a.comfy_quant"] = _conf(**(conf or {"format": fmt}))
    return _save(tmp_path / f"{name}.safetensors", tensors)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_block_formats_are_detected(tmp_path, fmt):
    scan = cq.refuse_comfy_quant(_layer_file(tmp_path, fmt))
    assert scan.counts() == {fmt: 1}
    assert not scan.problems


def test_block_formats_in_the_header_table_are_detected(tmp_path):
    t = nvfp4_encode(torch.randn(128, 64))
    meta = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": {"a": {"format": "nvfp4"}}}
        )
    }
    path = _save(tmp_path / "h.safetensors", {f"a.{k}": v for k, v in t.items()}, metadata = meta)
    assert cq.refuse_comfy_quant(path).counts() == {"nvfp4": 1}


@pytest.mark.parametrize(
    "fmt,kwargs,why",
    [
        ("nvfp4", {"drop": ("weight_scale_2",)}, "weight_scale_2"),
        ("nvfp4", {"drop": ("weight_scale",)}, "weight_scale is missing"),
        ("nvfp4", {"extra": {"a.weight_codebook": torch.zeros(4)}}, "weight_codebook"),
        ("nvfp4", {"extra": {"a.pre_quant_scale": torch.ones(3)}}, "pre_quant_scale"),
        ("nvfp4", {"conf": {"format": "nvfp4", "convrot": True}}, "ConvRot on nvfp4"),
        ("mxfp8", {"extra": {"a.weight_scale_2": torch.ones(())}}, "weight_scale_2"),
    ],
)
def test_malformed_block_layers_are_refused(tmp_path, fmt, kwargs, why):
    with pytest.raises(ValueError, match = why):
        cq.refuse_comfy_quant(_layer_file(tmp_path, fmt, **kwargs))


def test_untiled_block_scales_are_refused(tmp_path):
    t = nvfp4_encode(torch.randn(256, 64))
    t["weight_scale"] = t["weight_scale"][:200]  # a plain [N, K/16] matrix is not ComfyUI's layout
    path = _save(
        tmp_path / "u.safetensors",
        {**{f"a.{k}": v for k, v in t.items()}, "a.comfy_quant": _conf(format = "nvfp4")},
    )
    with pytest.raises(ValueError, match = "tiled layout"):
        cq.refuse_comfy_quant(path)


@pytest.mark.parametrize("fmt", ["convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "brand_new"])
def test_remaining_formats_keep_the_refusal(tmp_path, fmt):
    path = _save(
        tmp_path / "r.safetensors",
        {
            "a.weight": torch.zeros(8, 8, dtype = torch.uint8),
            "a.weight_scale": torch.ones(8, 1),
            "a.comfy_quant": _conf(format = fmt),
        },
    )
    with pytest.raises(ValueError, match = rf"1 layer\(s\) in ComfyUI format '{fmt}'"):
        cq.refuse_comfy_quant(path)


def test_resident_size_prices_kept_and_dequantized_nvfp4(tmp_path):
    path = _layer_file(tmp_path, "nvfp4", rows = 2048, cols = 2048)
    kept = cq.comfy_resident_mib(path, keep_int8 = False, keep_fp8 = False, keep_nvfp4 = True)
    dense = cq.comfy_resident_mib(path, keep_int8 = False, keep_fp8 = False)
    assert dense == 8 + 1  # 2048 x 2048 bf16, plus the scales rounded up
    assert kept == 3  # 2 MiB of codes + 256 KiB of block scales


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)


class _Tiny(nn.Module):
    _keep_in_fp32_modules = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(
        self,
        dim: int = DIM,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(blocks))
        self.norm = nn.LayerNorm(dim)

    @classmethod
    def load_config(cls, repo, **_kwargs):
        return {"dim": DIM, "blocks": 2}

    @staticmethod
    def _get_signature_keys(_cls):
        return {"dim", "blocks"}, set()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _convert(checkpoint = None, **_kwargs):
    """What the diffusers converters do to a ComfyUI DiT: split the fused qkv by rows, rename the rest."""
    out = {}
    for key, value in checkpoint.items():
        if ".qkv." in key:
            stem, leaf = key.rsplit(".qkv.", 1)
            for part, chunk in zip(("to_q", "to_k", "to_v"), value.chunk(3, dim = 0)):
                out[f"{stem}.{part}.{leaf}"] = chunk
        else:
            out[key.replace(".out.", ".to_out.")] = value
    return out


@pytest.fixture
def block_file(tmp_path, monkeypatch):
    def make(fmt, drop_input_scale = False):
        monkeypatch.setattr(
            cq,
            "_mapping",
            lambda cls: (
                _convert,
                types.SimpleNamespace(
                    _should_convert_state_dict_to_diffusers = lambda a, b: set(a) != set(b),
                    _get_mapping_function_kwargs = lambda fn, **kw: {},
                ),
            ),
        )
        torch.manual_seed(0)
        dense = _Tiny()
        encode, reference = ENCODERS[fmt]
        tensors, expected = {}, {}
        for b, block in enumerate(dense.blocks):
            layers = {
                f"blocks.{b}.qkv": torch.cat(
                    [block.to_q.weight, block.to_k.weight, block.to_v.weight]
                ),
                f"blocks.{b}.out": block.to_out.weight,
            }
            for name, weight in layers.items():
                t = encode(weight.detach())
                if drop_input_scale:
                    t.pop("input_scale", None)
                tensors.update({f"{name}.{k}": v for k, v in t.items()})
                tensors[f"{name}.comfy_quant"] = _conf(format = fmt)
                ref = reference(t)
                if name.endswith(".qkv"):
                    for part, chunk in zip(("to_q", "to_k", "to_v"), ref.chunk(3)):
                        expected[f"blocks.{b}.{part}.weight"] = chunk
                else:
                    expected[f"blocks.{b}.to_out.weight"] = ref
                bias = getattr(block, "to_out" if name.endswith("out") else "to_q").bias
                tensors[f"{name}.bias"] = (
                    torch.cat([block.to_q.bias, block.to_k.bias, block.to_v.bias])
                    if name.endswith(".qkv")
                    else bias
                ).detach()
        tensors["norm.weight"] = dense.norm.weight.detach()
        tensors["norm.bias"] = dense.norm.bias.detach()
        return _save(tmp_path / f"tiny_{fmt}.safetensors", tensors), tensors, expected

    return make


def _load(
    path,
    dtype = torch.float32,
    **kwargs,
):
    return cq.load_comfy_quant_transformer(
        _Tiny,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": dtype, "config": "base/repo", "subfolder": "transformer"},
        int8_backend = None,
        family = "z-image",
        **kwargs,
    )


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_dequant_load_is_bit_exact_through_the_row_split(block_file, fmt):
    path, tensors, expected = block_file(fmt)
    model = _load(path)
    info = model._unsloth_comfy_quant
    assert info["dequantized"] == 4 and info[fmt] == 0
    state = model.state_dict()
    for key, want in expected.items():
        assert torch.equal(state[key], want), key
    assert torch.equal(state["blocks.0.to_k.bias"], tensors["blocks.0.qkv.bias"][DIM : 2 * DIM])


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_runtime_linears_keep_codes_and_tiled_scales_unchanged(block_file, fmt):
    if fmt == "nvfp4":
        pytest.importorskip("flashinfer")
    path, tensors, _expected = block_file(fmt)
    model = _load(path, dtype = torch.bfloat16, **{f"{fmt}_backend": "kept"})
    assert model._unsloth_comfy_quant[fmt] == 8 and model._unsloth_comfy_quant["dequantized"] == 0
    out = model.blocks[1].to_out
    file_codes = tensors["blocks.1.out.weight"]
    file_scale = tensors["blocks.1.out.weight_scale"].view(torch.uint8).reshape(-1)
    if fmt == "nvfp4":
        assert type(out).__name__ == "NVFP4FlashInferLinear"
        assert torch.equal(out.wq, cb.swap_nibbles(file_codes))
        assert torch.equal(out.w_sf.reshape(-1), file_scale)
        assert out.w_scale.item() == tensors["blocks.1.out.weight_scale_2"].item()
        assert out.a_gsf.item() == pytest.approx(1.0 / tensors["blocks.1.out.input_scale"].item())
        q = model.blocks[
            0
        ].to_q  # a row split of the fused qkv: scales re-tiled from the plain matrix
        plain = cb.untile_scales(
            tensors["blocks.0.qkv.weight_scale"].view(torch.uint8), 3 * DIM, DIM // 16
        )
        assert torch.equal(cb.untile_scales(q.w_sf, DIM, DIM // 16), plain[:DIM])
    else:
        assert type(out).__name__ == "ComfyMXFP8Linear"
        assert torch.equal(out.weight_q.view(torch.uint8), file_codes.view(torch.uint8))
        assert torch.equal(out.weight_sf, file_scale)
    assert torch.equal(out.bias.float(), tensors["blocks.1.out.bias"].bfloat16().float())


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def test_nvfp4_without_an_input_scale_scales_activations_per_call(block_file):
    pytest.importorskip("flashinfer")
    path, _tensors, _expected = block_file("nvfp4", drop_input_scale = True)
    model = _load(path, dtype = torch.bfloat16, nvfp4_backend = "kept")
    assert model._unsloth_comfy_quant["nvfp4"] == 8
    assert {type(m).__name__ for m in model.modules() if hasattr(m, "wq")} == {
        "ComfyNVFP4DynamicLinear"
    }
    from core.inference.diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear
    from core.inference.diffusion_nvfp4_protect import protect_controller, protect_layers

    layers = [m for _n, m in protect_layers(model)]
    assert len(layers) == 8 and all(is_nvfp4_flashinfer_linear(m) for m in layers)
    # its own step controller, not the process-wide one
    assert (
        all(m.protect is layers[0].protect for m in layers)
        and layers[0].protect is not protect_controller()
    )


@pytest.mark.skipif(not _blackwell(), reason = "needs a Blackwell GPU")
@pytest.mark.parametrize("fmt,dynamic", [("nvfp4", False), ("nvfp4", True), ("mxfp8", False)])
def test_runtime_forward_matches_the_dequantized_product(block_file, fmt, dynamic, monkeypatch):
    if fmt == "nvfp4":
        pytest.importorskip("flashinfer")
        monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    path, _tensors, expected = block_file(fmt, drop_input_scale = dynamic)
    model = _load(path, dtype = torch.bfloat16, **{f"{fmt}_backend": "kept"}).cuda()
    x = torch.randn(64, DIM, device = "cuda", dtype = torch.bfloat16)
    ref = (
        x.float() @ expected["blocks.1.to_out.weight"].cuda().T
        + model.blocks[1].to_out.bias.float()
    )
    got = model.blocks[1].to_out(x).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < (0.15 if fmt == "nvfp4" else 0.05), rel  # activation quantization error only


def _target(device):
    return types.SimpleNamespace(torch_device = device, dtype = torch.bfloat16)


def test_nvfp4_on_cpu_or_mps_dequantizes(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    for device in ("cpu", "mps"):
        backend, reason = cb.comfy_block_backend("nvfp4", _target(device), "z-image")
        assert backend is None and "not CUDA" in reason


def test_nvfp4_follows_studios_nvfp4_switch(monkeypatch):
    import core.inference.diffusion_nvfp4_ops as ops

    monkeypatch.setattr(ops, "_resolve_backend", lambda device = None: ("flashinfer", "preflight ok"))
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    backend, reason = cb.comfy_block_backend("nvfp4", _target("cuda:0"), "z-image")
    assert backend is None and "UNSLOTH_NVFP4_DIFFUSION=1" in reason
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    assert cb.comfy_block_backend("nvfp4", _target("cuda:0"), "z-image")[0] == "flashinfer"
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMFY_NVFP4", "0")
    assert cb.comfy_block_backend("nvfp4", _target("cuda:0"), "z-image")[0] is None


def test_nvfp4_without_flashinfer_or_on_a_denied_family_dequantizes(monkeypatch):
    import core.inference.diffusion_nvfp4_ops as ops

    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.setattr(
        ops,
        "_resolve_backend",
        lambda device = None: ("torchao", "sm_90 is not in the flashinfer NVFP4 set"),
    )
    backend, reason = cb.comfy_block_backend("nvfp4", _target("cuda:0"), "z-image")
    assert backend is None and "sm_90" in reason
    monkeypatch.setattr(ops, "_resolve_backend", lambda device = None: ("flashinfer", "ok"))
    assert cb.comfy_block_backend("nvfp4", _target("cuda:0"), "qwen-image")[0] is None


def test_mxfp8_needs_blackwell_bf16_and_an_allowed_family(monkeypatch):
    monkeypatch.setattr(cb, "mxfp8_runtime_reason", lambda target: None)
    assert (
        cb.comfy_block_backend("mxfp8", _target("cuda:0"), "krea-2", dtype = torch.bfloat16)[0]
        == "scaled_mm"
    )
    assert (
        cb.comfy_block_backend("mxfp8", _target("cuda:0"), "krea-2", dtype = torch.float16)[0] is None
    )
    assert cb.comfy_block_backend("mxfp8", _target("cuda:0"), "qwen-image")[0] is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMFY_MXFP8", "0")
    assert cb.comfy_block_backend("mxfp8", _target("cuda:0"), "krea-2")[0] is None


def test_mxfp8_probe_refuses_pre_blackwell_and_non_cuda(monkeypatch):
    assert "not CUDA" in cb.mxfp8_runtime_reason(_target("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert "sm_90" in cb.mxfp8_runtime_reason(_target("cuda:0"))


def test_backends_are_logged_per_format_present(caplog):
    import logging

    scan = types.SimpleNamespace(counts = lambda: {"nvfp4": 3})
    logger = logging.getLogger("comfy_block_test")
    with caplog.at_level(logging.INFO, logger = "comfy_block_test"):
        out = cb.comfy_block_backends(scan, _target("cpu"), "z-image", logger = logger)
    assert out == {"nvfp4_backend": None, "mxfp8_backend": None}
    assert "3 nvfp4 layer(s) are dequantized" in caplog.text and "mxfp8" not in caplog.text
