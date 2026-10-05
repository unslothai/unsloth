# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for loading ComfyUI-format quantized single files (``diffusion_comfy_quant.py``).

Detection and refusal read only the safetensors header, so they run on tiny synthetic files. The
loader runs on CPU against a two-block stand-in transformer whose converter does what the real
diffusers converters do to a ComfyUI checkpoint: split a fused qkv by rows and rename the rest. That
is enough to prove the codes and scales land on the right diffusers weight unchanged, that the
ConvRot rotation is installed on exactly those, and that what cannot be mapped is refused.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
nn = torch.nn

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
from core.inference.diffusion_convrot import build_convrot_hadamard, is_rotated_linear  # noqa: E402

DIM = 512  # the int8 runtime's minimum Linear width
GROUP = 256


def _conf(**conf) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


def _int8(weight: torch.Tensor, group: int = 0) -> tuple:
    """ComfyUI's int8_tensorwise layout: optional ConvRot, then symmetric per-row int8."""
    w = weight.float()
    if group:
        h = build_convrot_hadamard(group)
        w = (w.reshape(w.shape[0], -1, group) @ h.T).reshape(w.shape)
    scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-8) / 127.0
    return torch.round(w / scale).clamp(-127, 127).to(torch.int8), scale


def _save(path, tensors, metadata = None) -> str:
    tensors = {k: v.contiguous().clone() for k, v in tensors.items()}
    safetensors_torch.save_file(tensors, str(path), metadata = metadata)
    return str(path)


# ------------------------------------------------------------------------------------- detection
def test_plain_and_unscaled_fp8_files_are_not_comfy_quant(tmp_path):
    plain = _save(tmp_path / "a.safetensors", {"x.weight": torch.zeros(4, 4, dtype = torch.bfloat16)})
    fp8 = _save(tmp_path / "b.safetensors", {"x.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn)})
    assert cq.scan_comfy_quant(plain) is None
    assert cq.scan_comfy_quant(fp8) is None
    assert cq.scan_comfy_quant(str(tmp_path / "missing.safetensors")) is None
    assert cq.scan_comfy_quant(None) is None


def test_per_layer_int8_convrot_is_detected(tmp_path):
    q, s = _int8(torch.randn(8, GROUP), GROUP)
    path = _save(
        tmp_path / "m.safetensors",
        {
            "a.weight": q,
            "a.weight_scale": s,
            "a.comfy_quant": _conf(format = "int8_tensorwise", convrot = True, convrot_groupsize = GROUP),
            "b.weight": q,
            "b.weight_scale": s,
            "b.comfy_quant": _conf(format = "int8_tensorwise", params = {"convrot": True}),
            "c.weight": torch.zeros(8, 8),
        },
    )
    scan = cq.scan_comfy_quant(path)
    assert scan is not None and not scan.problems
    assert scan.counts() == {"int8_tensorwise+convrot": 2}
    assert scan.layers["b"].group == 256  # ComfyUI's default group, read through "params"
    assert cq.comfy_quant_error(scan) is None


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "brand_new"])
def test_unsupported_formats_are_refused_by_name(tmp_path, fmt):
    path = _save(
        tmp_path / "m.safetensors",
        {
            "a.weight": torch.zeros(8, 8, dtype = torch.uint8),
            "a.weight_scale": torch.ones(8, 1),
            "a.comfy_quant": _conf(format = fmt),
        },
    )
    with pytest.raises(ValueError, match = fmt):
        cq.refuse_comfy_quant(path)


def test_header_quantization_metadata_is_read_and_refused(tmp_path):
    """The Comfy-Org nvfp4 files declare formats only in the header, with no per-layer tensors."""
    meta = {cq.QUANT_METADATA_KEY: json.dumps({"layers": {"a": {"format": "nvfp4"}}})}
    path = _save(
        tmp_path / "m.safetensors",
        {
            "a.weight": torch.zeros(8, 4, dtype = torch.uint8),
            "a.weight_scale": torch.zeros(8, 1, dtype = torch.uint8),
            "a.weight_scale_2": torch.ones(()),
            "a.input_scale": torch.ones(()),
        },
        meta,
    )
    scan = cq.scan_comfy_quant(path)
    assert scan is not None and not scan.layers
    with pytest.raises(ValueError, match = "'nvfp4'"):
        cq.refuse_comfy_quant(path)


def test_header_fp8_table_and_legacy_scaled_fp8_are_supported(tmp_path):
    w8 = torch.randn(8, 8).to(torch.float8_e4m3fn)
    meta = {cq.QUANT_METADATA_KEY: json.dumps({"layers": {"a": {"format": "float8_e4m3fn"}}})}
    table = _save(
        tmp_path / "t.safetensors",
        {"a.weight": w8, "a.weight_scale": torch.tensor(0.5), "a.input_scale": torch.tensor(1.0)},
        meta,
    )
    legacy = _save(
        tmp_path / "l.safetensors",
        {
            "scaled_fp8": torch.zeros(2, dtype = torch.float8_e4m3fn),
            "a.weight": w8,
            "a.scale_weight": torch.tensor([0.5]),
            "a.scale_input": torch.tensor([1.0]),
            "a.bias": torch.zeros(8),
        },
    )
    for path in (table, legacy):
        scan = cq.refuse_comfy_quant(path)
        assert scan.counts() == {"float8_e4m3fn": 1}
    assert cq.scan_comfy_quant(legacy).legacy_scaled_fp8


def test_undeclared_scales(tmp_path):
    """An fp8 weight with a plain weight_scale has one meaning; scaled int8 with no format does not."""
    fp8 = _save(
        tmp_path / "f.safetensors",
        {"a.weight": torch.zeros(8, 8).to(torch.float8_e4m3fn), "a.weight_scale": torch.tensor(2.0)},
    )
    assert cq.refuse_comfy_quant(fp8).counts() == {"float8_e4m3fn": 1}
    int8 = _save(
        tmp_path / "i.safetensors",
        {"a.weight": torch.zeros(8, 8, dtype = torch.int8), "a.weight_scale": torch.ones(8, 1)},
    )
    with pytest.raises(ValueError, match = "no quantization format declared"):
        cq.refuse_comfy_quant(int8)


@pytest.mark.parametrize(
    "tensors, why",
    [
        ({"a.weight": torch.zeros(8, GROUP, dtype = torch.int8), "a.weight_scale": torch.ones(8, 4)}, "block scales"),
        ({"a.weight": torch.zeros(8, GROUP, dtype = torch.int8)}, "weight_scale is missing"),
        ({"a.weight": torch.zeros(8, GROUP, dtype = torch.uint8), "a.weight_scale": torch.ones(8, 1)}, "stored as U8"),
        ({"a.weight_scale": torch.ones(8, 1)}, "no a.weight"),
        ({"a.weight": torch.zeros(2, 8, GROUP, dtype = torch.int8), "a.weight_scale": torch.ones(2, 8, 1)}, "2-D"),
    ],
)
def test_malformed_int8_layers_are_refused(tmp_path, tensors, why):
    tensors = dict(tensors, **{"a.comfy_quant": _conf(format = "int8_tensorwise")})
    with pytest.raises(ValueError, match = why):
        cq.refuse_comfy_quant(_save(tmp_path / "m.safetensors", tensors))


def test_bad_convrot_group_is_refused(tmp_path):
    path = _save(
        tmp_path / "m.safetensors",
        {
            "a.weight": torch.zeros(8, 384, dtype = torch.int8),
            "a.weight_scale": torch.ones(8, 1),
            "a.comfy_quant": _conf(format = "int8_tensorwise", convrot = True, convrot_groupsize = 128),
        },
    )
    with pytest.raises(ValueError, match = "power of 4"):
        cq.refuse_comfy_quant(path)


# ------------------------------------------------------------------------------------- loading
class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.adaLN_modulation = nn.Sequential(nn.Linear(dim, 2 * dim))


class _Tiny(nn.Module):
    """Stand-in diffusers transformer: diffusers names, a ComfyUI checkpoint with a fused qkv."""

    _keep_in_fp32_modules = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, dim: int = DIM, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(blocks))
        self.norm = nn.LayerNorm(dim)

    @classmethod
    def load_config(cls, repo, **_kwargs):
        assert repo == "base/repo"
        return {"dim": DIM, "blocks": 2}

    @staticmethod
    def _get_signature_keys(_cls):
        return {"dim", "blocks"}, set()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _convert(checkpoint = None, config = None, mix_columns = False, **_kwargs):
    out = {}
    for key, value in checkpoint.items():
        if key.endswith(".qkv.weight") or key.endswith(".qkv.bias"):
            stem, leaf = key.rsplit(".qkv.", 1)
            chunks = list(value.chunk(3, dim = 0))
            if mix_columns and leaf == "weight":
                # q's left half spliced onto k's right half: rows of two layers in one weight
                half = chunks[0].shape[1] // 2
                chunks[0] = torch.cat([chunks[0][:, :half], chunks[1][:, half:]], dim = 1)
            for part, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                out[f"{stem}.{part}.{leaf}"] = chunk
        else:
            out[key.replace(".out.", ".to_out.")] = value
    return out


def _sfm():
    return types.SimpleNamespace(
        _should_convert_state_dict_to_diffusers = lambda model_sd, ckpt: set(model_sd) != set(ckpt),
        _get_mapping_function_kwargs = lambda fn, **kw: {},
    )


@pytest.fixture
def comfy_file(tmp_path, monkeypatch):
    """A ComfyUI int8_convrot file of ``_Tiny`` plus the dense weights it was quantized from."""
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    torch.manual_seed(0)
    dense = _Tiny()
    tensors = {}
    for b, block in enumerate(dense.blocks):
        qkv = torch.cat([block.to_q.weight, block.to_k.weight, block.to_v.weight]).detach()
        layers = {
            f"blocks.{b}.qkv": (qkv, torch.cat([block.to_q.bias, block.to_k.bias, block.to_v.bias])),
            f"blocks.{b}.out": (block.to_out.weight.detach(), block.to_out.bias),
            f"blocks.{b}.adaLN_modulation.0": (
                block.adaLN_modulation[0].weight.detach(),
                block.adaLN_modulation[0].bias,
            ),
        }
        for name, (weight, bias) in layers.items():
            q, s = _int8(weight, GROUP)
            tensors[f"{name}.weight"] = q
            tensors[f"{name}.weight_scale"] = s
            tensors[f"{name}.bias"] = bias.detach().clone()
            tensors[f"{name}.comfy_quant"] = _conf(
                format = "int8_tensorwise", convrot = True, convrot_groupsize = GROUP
            )
    tensors["norm.weight"] = dense.norm.weight.detach().clone()
    tensors["norm.bias"] = dense.norm.bias.detach().clone()
    path = _save(tmp_path / "tiny_int8_convrot.safetensors", tensors)
    return path, dense, tensors


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


def _cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim = 0).item()


def test_dequant_path_reproduces_the_dense_weights(comfy_file):
    path, dense, _ = comfy_file
    model = _load(path)
    assert model._unsloth_comfy_quant == {"backend": None, "int8": 0, "convrot": 0, "dequantized": 6}
    assert not any(is_rotated_linear(m) for m in model.modules())
    ref = dense.state_dict()
    for key, value in model.state_dict().items():
        assert value.dtype == torch.float32
        if not ref[key].any():
            assert not value.any(), key
            continue
        # int8 rounding only: the rotation is undone, so every weight lines up with its source
        assert _cos(value, ref[key]) > 0.9999, key
    assert not model.training


def test_int8_runtime_keeps_codes_and_scales_unchanged(comfy_file):
    pytest.importorskip("torchao")
    path, dense, tensors = comfy_file
    model = _load(path, int8_backend = "torchao")
    if not model._unsloth_comfy_quant["int8"]:
        pytest.skip("this torchao has no Int8Tensor")
    # 2 blocks x (q, k, v, out) as int8; adaLN is a modulation Linear Studio keeps dense
    assert model._unsloth_comfy_quant == {"backend": "torchao", "int8": 8, "convrot": 8, "dequantized": 2}
    assert model._unsloth_runtime_quant == "int8"
    for b, block in enumerate(model.blocks):
        q, s = tensors[f"blocks.{b}.qkv.weight"], tensors[f"blocks.{b}.qkv.weight_scale"]
        for i, part in enumerate(("to_q", "to_k", "to_v")):
            linear = getattr(block, part)
            assert is_rotated_linear(linear) and linear.convrot_groupsize == GROUP
            assert torch.equal(linear.weight.qdata, q[i * DIM : (i + 1) * DIM])
            assert torch.equal(linear.weight.scale.reshape(-1, 1), s[i * DIM : (i + 1) * DIM])
        assert torch.equal(block.to_out.weight.qdata, tensors[f"blocks.{b}.out.weight"])
        adaln = block.adaLN_modulation[0]
        assert type(adaln.weight.data) is torch.Tensor and not is_rotated_linear(adaln)
        assert _cos(adaln.weight, dense.blocks[b].adaLN_modulation[0].weight) > 0.9999


def test_native_backend_keeps_codes_and_computes_the_dense_product(comfy_file):
    """Under offload the codes go to the torchao-free twin (plain buffers the hooks can move)."""
    from core.inference.diffusion_native_quant import is_native_linear

    path, dense, tensors = comfy_file
    model = _load(path, int8_backend = "native")
    assert model._unsloth_comfy_quant == {"backend": "native", "int8": 8, "convrot": 8, "dequantized": 2}
    block = model.blocks[1]
    linear = block.to_k
    assert is_native_linear(linear) and linear.rot_group == GROUP
    q, s = tensors["blocks.1.qkv.weight"], tensors["blocks.1.qkv.weight_scale"]
    assert torch.equal(linear.weight_q, q[DIM : 2 * DIM])
    assert torch.equal(linear.weight_scale.view(torch.float32), s[DIM : 2 * DIM, 0])
    assert torch.equal(linear.bias, dense.blocks[1].to_k.bias)
    # weight-only on CPU: rotate(x) @ dequant(W_rot).T is x @ W.T up to the int8 rounding
    x = torch.randn(4, DIM)
    want = torch.nn.functional.linear(x, dense.blocks[1].to_k.weight, dense.blocks[1].to_k.bias)
    assert _cos(linear(x), want) > 0.9999
    assert type(block.adaLN_modulation[0]) is nn.Linear


def test_fp8_layers_dequantize_with_their_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    torch.manual_seed(1)
    ref = _Tiny()
    tensors = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    layer = "blocks.0.to_out"
    w = tensors.pop(f"{layer}.weight")
    scale = w.abs().max() / 448.0
    codes = (w / scale).to(torch.float8_e4m3fn)
    tensors.update({f"{layer}.weight": codes, f"{layer}.scale_weight": scale.reshape(1)})
    tensors["scaled_fp8"] = torch.zeros(2, dtype = torch.float8_e4m3fn)
    path = _save(tmp_path / "legacy_fp8_scaled.safetensors", tensors)
    model = _load(path)
    assert torch.equal(model.blocks[0].to_out.weight, codes.float() * scale)
    assert model._unsloth_comfy_quant["dequantized"] == 1


def test_a_converter_that_mixes_columns_is_refused(comfy_file, monkeypatch):
    pytest.importorskip("torchao")
    path, _, _ = comfy_file
    monkeypatch.setattr(
        cq, "_mapping", lambda cls: (lambda **kw: _convert(mix_columns = True, **kw), _sfm())
    )
    with pytest.raises(ValueError, match = "mixed columns"):
        _load(path, int8_backend = "torchao")


def test_missing_weights_after_conversion_are_refused(comfy_file, monkeypatch):
    path, _, _ = comfy_file
    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (lambda **kw: {k: v for k, v in _convert(**kw).items() if "norm" not in k}, _sfm()),
    )
    with pytest.raises(ValueError, match = "missing 2 weight"):
        _load(path)


def test_the_loader_refuses_what_the_scan_refuses(tmp_path):
    path = _save(
        tmp_path / "m.safetensors",
        {
            "a.weight": torch.zeros(8, 8, dtype = torch.uint8),
            "a.weight_scale": torch.ones(8, 1),
            "a.comfy_quant": _conf(format = "mxfp8"),
        },
    )
    scan = cq.scan_comfy_quant(path)
    with pytest.raises(ValueError, match = "mxfp8"):
        cq.load_comfy_quant_transformer(_Tiny, path, scan, {"config": "base/repo"}, int8_backend = None)
