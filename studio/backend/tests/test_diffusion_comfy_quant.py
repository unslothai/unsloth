# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI quantised single-file loader; ConvRot must be installed on exactly the recorded layers."""

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


def _save(
    path,
    tensors,
    metadata = None,
) -> str:
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


@pytest.mark.parametrize("fmt", ["convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "brand_new"])
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
    """Comfy-Org files declare formats only in the header, with no per-layer tensors."""
    meta = {cq.QUANT_METADATA_KEY: json.dumps({"layers": {"a": {"format": "convrot_w4a4"}}})}
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
    with pytest.raises(ValueError, match = "'convrot_w4a4'"):
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
        {
            "a.weight": torch.zeros(8, 8).to(torch.float8_e4m3fn),
            "a.weight_scale": torch.tensor(2.0),
        },
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
        (
            {
                "a.weight": torch.zeros(8, GROUP, dtype = torch.int8),
                "a.weight_scale": torch.ones(8, 4),
            },
            "block scales",
        ),
        ({"a.weight": torch.zeros(8, GROUP, dtype = torch.int8)}, "weight_scale is missing"),
        (
            {
                "a.weight": torch.zeros(8, GROUP, dtype = torch.uint8),
                "a.weight_scale": torch.ones(8, 1),
            },
            "stored as U8",
        ),
        ({"a.weight_scale": torch.ones(8, 1)}, "no a.weight"),
        (
            {
                "a.weight": torch.zeros(2, 8, GROUP, dtype = torch.int8),
                "a.weight_scale": torch.ones(2, 8, 1),
            },
            "2-D",
        ),
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
        assert repo == "base/repo"
        return {"dim": DIM, "blocks": 2}

    @staticmethod
    def _get_signature_keys(_cls):
        return {"dim", "blocks"}, set()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _convert(
    checkpoint = None,
    config = None,
    mix_columns = False,
    **_kwargs,
):
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
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim = 0
    ).item()


def test_dequant_path_reproduces_the_dense_weights(comfy_file):
    path, dense, _ = comfy_file
    model = _load(path)
    assert model._unsloth_comfy_quant == {
        "backend": None,
        "int8": 0,
        "convrot": 0,
        "fp8_backend": None,
        "fp8": 0,
        "dequantized": 6,
    }
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
    assert model._unsloth_comfy_quant == {
        "backend": "torchao",
        "int8": 8,
        "convrot": 8,
        "fp8_backend": None,
        "fp8": 0,
        "dequantized": 2,
    }
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
    assert model._unsloth_comfy_quant == {
        "backend": "native",
        "int8": 8,
        "convrot": 8,
        "fp8_backend": None,
        "fp8": 0,
        "dequantized": 2,
    }
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


def test_fp16_keeps_fp32_modules_by_their_converted_names(comfy_file):
    # Wan: ComfyUI's time_embedding.0 becomes diffusers' time_embedder only after conversion
    path, dense, _ = comfy_file

    class _KeepOut(_Tiny):
        _keep_in_fp32_modules = ["to_out"]

    model = cq.load_comfy_quant_transformer(
        _KeepOut,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.float16, "config": "base/repo", "subfolder": "transformer"},
        int8_backend = None,
        family = "z-image",
    )
    for b, block in enumerate(model.blocks):
        assert block.to_out.weight.dtype == torch.float32
        assert block.to_out.bias.dtype == torch.float32
        assert torch.equal(block.to_out.bias, dense.blocks[b].to_out.bias)
        assert _cos(block.to_out.weight, dense.blocks[b].to_out.weight) > 0.9999
        assert block.to_q.weight.dtype == torch.float16
        assert block.adaLN_modulation[0].bias.dtype == torch.float16
    assert model.norm.weight.dtype == torch.float16


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
        lambda cls: (
            lambda **kw: {k: v for k, v in _convert(**kw).items() if "norm" not in k},
            _sfm(),
        ),
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
        cq.load_comfy_quant_transformer(
            _Tiny, path, scan, {"config": "base/repo"}, int8_backend = None
        )


# ------------------------------------------------------------------------------------- fp8 runtime
def _fp8_file(
    tmp_path,
    *,
    fmt = "float8_e4m3fn",
    per_row = False,
    record = None,
):
    """A ComfyUI fp8 file of ``_Tiny`` (fused qkv, per-tensor or per-row ``weight_scale``) and its dense source."""
    torch.manual_seed(2)
    dense = _Tiny()
    tensors = {}
    fp8 = getattr(torch, fmt)
    top = torch.finfo(fp8).max
    for b, block in enumerate(dense.blocks):
        layers = {
            f"blocks.{b}.qkv": (
                torch.cat([block.to_q.weight, block.to_k.weight, block.to_v.weight]).detach(),
                torch.cat([block.to_q.bias, block.to_k.bias, block.to_v.bias]),
            ),
            f"blocks.{b}.out": (block.to_out.weight.detach(), block.to_out.bias),
            f"blocks.{b}.adaLN_modulation.0": (
                block.adaLN_modulation[0].weight.detach(),
                block.adaLN_modulation[0].bias,
            ),
        }
        for name, (weight, bias) in layers.items():
            w = weight.float()
            scale = (w.abs().amax(dim = 1, keepdim = True) if per_row else w.abs().max()) / top
            tensors[f"{name}.weight"] = (w / scale).to(fp8)
            tensors[f"{name}.weight_scale"] = scale.float()
            tensors[f"{name}.input_scale"] = torch.tensor(1.0)
            tensors[f"{name}.bias"] = bias.detach().clone()
            tensors[f"{name}.comfy_quant"] = _conf(format = fmt, full_precision_matrix_mult = True)
    tensors["norm.weight"] = dense.norm.weight.detach().clone()
    tensors["norm.bias"] = dense.norm.bias.detach().clone()
    meta = {cq.CONVERSION_RECORD_KEY: json.dumps(record)} if record else None
    return _save(tmp_path / f"tiny_{fmt}.safetensors", tensors, metadata = meta), dense, tensors


def _load_bf16(path, **kwargs):
    kwargs.setdefault("int8_backend", None)
    return cq.load_comfy_quant_transformer(
        _Tiny,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo", "subfolder": "transformer"},
        family = "z-image",
        **kwargs,
    )


@pytest.mark.parametrize("per_row", [False, True])
def test_fp8_runtime_keeps_codes_and_repeats_a_tensor_scale_per_row(tmp_path, monkeypatch, per_row):
    pytest.importorskip("torchao")
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    path, _dense, tensors = _fp8_file(tmp_path, per_row = per_row)
    model = _load_bf16(path, fp8_backend = "torchao")
    assert model._unsloth_comfy_quant == {
        "backend": None,
        "int8": 0,
        "convrot": 0,
        "fp8_backend": "torchao",
        "fp8": 10,
        "dequantized": 0,
    }
    assert model._unsloth_runtime_quant == "fp8"
    for b, block in enumerate(model.blocks):
        codes = tensors[f"blocks.{b}.qkv.weight"]
        scale = tensors[f"blocks.{b}.qkv.weight_scale"].reshape(-1, 1).expand(3 * DIM, 1)
        for i, leaf in enumerate(("to_q", "to_k", "to_v")):
            weight = getattr(block, leaf).weight
            assert type(weight).__name__ == "Float8Tensor"
            assert torch.equal(
                weight.qdata.view(torch.uint8), codes[i * DIM : (i + 1) * DIM].view(torch.uint8)
            )
            assert torch.equal(weight.scale.float().reshape(-1, 1), scale[i * DIM : (i + 1) * DIM])
            # the same per-row dynamic-activation layout as Studio's own FP8 checkpoints
            assert weight.act_quant_kwargs.granularity.__class__.__name__ == "PerRow"
        assert type(block.adaLN_modulation[0].weight).__name__ == "Float8Tensor"
    assert not any(is_rotated_linear(m) for m in model.modules())


def test_fp8_native_backend_keeps_codes_and_computes_the_dense_product(tmp_path, monkeypatch):
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    path, dense, tensors = _fp8_file(tmp_path)
    model = _load_bf16(path, fp8_backend = "native")
    assert model._unsloth_comfy_quant["fp8_backend"] == "native"
    assert model._unsloth_comfy_quant["fp8"] == 10
    linear = model.blocks[1].to_k
    assert type(linear).__name__ == "NativeWeightOnlyLinear" and linear.scheme == "fp8"
    codes = tensors["blocks.1.qkv.weight"][DIM : 2 * DIM]
    assert torch.equal(linear.weight_q, codes.view(torch.uint8))
    x = torch.randn(4, DIM, dtype = torch.bfloat16)
    want = torch.nn.functional.linear(
        x.float(), dense.blocks[1].to_k.weight, dense.blocks[1].to_k.bias
    )
    assert _cos(linear(x), want) > 0.999


def test_fp8_without_a_runtime_dequantizes(tmp_path, monkeypatch):
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    path, _dense, tensors = _fp8_file(tmp_path)
    model = _load_bf16(path, fp8_backend = None)
    assert model._unsloth_comfy_quant["fp8"] == 0
    assert model._unsloth_comfy_quant["dequantized"] == 6
    w = tensors["blocks.0.out.weight"].float() * tensors["blocks.0.out.weight_scale"]
    assert torch.equal(model.blocks[0].to_out.weight, w.to(torch.bfloat16))


def test_e5m2_and_non_bf16_pipelines_dequantize(tmp_path, monkeypatch):
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    path, _dense, _tensors = _fp8_file(tmp_path, fmt = "float8_e5m2")
    assert _load_bf16(path, fp8_backend = "torchao")._unsloth_comfy_quant["fp8"] == 0
    path, _dense, _tensors = _fp8_file(tmp_path)
    model = _load(path, fp8_backend = "torchao")  # float32 pipeline
    assert model._unsloth_comfy_quant["fp8"] == 0
    assert model._unsloth_comfy_quant["dequantized"] == 6


@pytest.mark.parametrize(
    "native, supported, offload, dtype, env, want",
    [
        (None, "fp8", False, "bfloat16", None, "torchao"),
        (None, "fp8", True, "bfloat16", None, None),  # offloaded torchao fp8: dequantize, as before
        (None, None, False, "bfloat16", None, None),  # no fp8 GEMM (older GPU, CPU, MPS)
        ("fp8", None, False, "bfloat16", None, "native"),  # ROCm / stubbed torchao
        (None, "fp8", False, "float16", None, None),
        (None, "fp8", False, "bfloat16", "0", None),  # kill switch
    ],
)
def test_fp8_backend_follows_studios_own_fp8_rule(
    monkeypatch, native, supported, offload, dtype, env, want
):
    import core.inference.diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "native_quant_scheme", lambda *a, **k: native)
    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda *a, **k: supported)
    if env is None:
        monkeypatch.delenv(cq.COMFY_FP8_ENV, raising = False)
    else:
        monkeypatch.setenv(cq.COMFY_FP8_ENV, env)
    target = types.SimpleNamespace(device = "cuda", dtype = getattr(torch, dtype))
    assert cq.comfy_fp8_backend(target, "z-image", offload = offload) == want


# ------------------------------------------------------------------------------------- hosted prequant route
def test_comfy_prequant_scheme_is_read_from_the_header(tmp_path, comfy_file):
    path, _, _ = comfy_file
    assert cq.comfy_prequant_scheme(cq.scan_comfy_quant(path)) == "int8"
    fp8_path, _, _ = _fp8_file(tmp_path)
    assert cq.comfy_prequant_scheme(cq.scan_comfy_quant(fp8_path)) == "fp8"
    from core.inference.diffusion_prequant import local_prequant_scheme

    assert local_prequant_scheme(fp8_path) == "fp8"
    assert local_prequant_scheme(path) == "int8"


def test_comfy_prequant_refuses_another_scheme_or_base(tmp_path, monkeypatch):
    monkeypatch.setattr(cq, "_mapping", lambda cls: (_convert, _sfm()))
    path, _, _ = _fp8_file(tmp_path, record = {"base_model_id": "org/other-model", "scheme": "fp8"})
    kw = dict(base = "base/repo", family = "z-image", dtype = torch.bfloat16)
    with pytest.raises(ValueError, match = "not a int8 checkpoint"):
        cq.load_comfy_prequant(_Tiny, path, scheme = "int8", **kw)
    with pytest.raises(ValueError, match = "converted from org/other-model"):
        cq.load_comfy_prequant(_Tiny, path, scheme = "fp8", **kw)
    with pytest.raises(ValueError, match = "needs a bfloat16 pipeline"):
        cq.load_comfy_prequant(
            _Tiny, path, scheme = "fp8", **dict(kw, dtype = torch.float32, base = "org/other-model")
        )


def test_hosted_loader_rebuilds_a_comfy_file_into_studios_int8_weights(
    comfy_file, monkeypatch, tmp_path
):
    """``load_prequantized_transformer`` on a ComfyUI file: Int8Tensor weights under ConvRot, the loaded file
    recorded, the same placement tail as Studio's own checkpoints."""
    pytest.importorskip("torchao")
    import core.inference.diffusion_prequant as pq

    path, _dense, tensors = comfy_file
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    source = pq.PrequantSource(kind = "path", location = path)
    model = pq.load_prequantized_transformer(
        _Tiny,
        "base/repo",
        source,
        device = "cpu",
        dtype = torch.float32,
        scheme = "int8",
        family = "z-image",
    )
    assert model is not None, pq.last_prequant_failure()
    assert model._unsloth_prequant_path == path
    assert model._unsloth_runtime_quant == "int8"
    weight = model.blocks[0].to_out.weight
    assert type(getattr(weight, "data", weight)).__name__ == "Int8Tensor"
    assert torch.equal(weight.qdata, tensors["blocks.0.out.weight"])
    assert is_rotated_linear(model.blocks[0].to_out)
    # Studio's int8 filter keeps the modulation projection dense, as its own INT8 checkpoints do
    assert type(model.blocks[0].adaLN_modulation[0].weight).__name__ in ("Parameter", "Tensor")
    # asked for the wrong scheme: refused, with the reason recorded, so the caller quantizes dense instead
    assert (
        pq.load_prequantized_transformer(
            _Tiny,
            "base/repo",
            source,
            device = "cpu",
            dtype = torch.float32,
            scheme = "fp8",
            family = "z-image",
        )
        is None
    )
    assert "not a fp8 checkpoint" in (pq.last_prequant_failure() or "")


def test_resident_size_is_priced_from_the_header_not_the_name(tmp_path, comfy_file):
    """Kept int8 / fp8 layers cost their stored bytes, dequantized ones twice that; the file name is not read."""
    int8_path, _, tensors = comfy_file
    scan = cq.scan_comfy_quant(int8_path)
    quant = sum(tensors[f"{n}.weight"].numel() for n in scan.layers)
    plain = sum(
        v.numel()
        for k, v in tensors.items()
        if v.is_floating_point() and k.endswith((".bias", "norm.weight", "norm.bias"))
    )
    kept = cq.comfy_resident_mib(int8_path, scan, keep_int8 = True, keep_fp8 = True)
    dequant = cq.comfy_resident_mib(int8_path, scan, keep_int8 = False, keep_fp8 = True)
    assert dequant - kept == pytest.approx(quant / 2**20, abs = 1)
    assert kept >= (quant + 2 * plain) / 2**20
    fp8_path, _, _ = _fp8_file(tmp_path)
    fp8_scan = cq.scan_comfy_quant(fp8_path)
    assert cq.comfy_resident_mib(
        fp8_path, fp8_scan, keep_int8 = False, keep_fp8 = True
    ) < cq.comfy_resident_mib(fp8_path, fp8_scan, keep_int8 = False, keep_fp8 = False)
    assert (
        cq.comfy_resident_mib(str(tmp_path / "missing.safetensors"), keep_int8 = True, keep_fp8 = True)
        is None
    )


def test_the_row_map_is_traced_on_narrow_tags_and_falls_back_to_full_width(comfy_file, monkeypatch):
    """The fast pass hands the converter 4-column tags; a converter that needs the real width (here: one that
    reshapes by it) gets a second, full-width pass and the same result; one that drops columns is refused."""
    pytest.importorskip("torchao")
    path, _, tensors = comfy_file
    widths = []

    def _spy(
        checkpoint = None,
        config = None,
        **kw,
    ):
        widths.append(checkpoint["blocks.0.qkv.weight"].shape[1])
        return _convert(checkpoint = checkpoint, config = config, **kw)

    monkeypatch.setattr(cq, "_mapping", lambda cls: (_spy, _sfm()))
    fast = _load(path, int8_backend = "torchao")
    assert widths == [cq._NARROW_TAG_COLUMNS]

    def _needs_width(
        checkpoint = None,
        config = None,
        **kw,
    ):
        w = checkpoint["blocks.0.qkv.weight"]
        checkpoint["blocks.0.qkv.weight"] = w.reshape(-1, DIM)  # only valid at the real width
        return _convert(checkpoint = checkpoint, config = config, **kw)

    monkeypatch.setattr(cq, "_mapping", lambda cls: (_needs_width, _sfm()))
    full = _load(path, int8_backend = "torchao")
    for a, b in zip(fast.state_dict().values(), full.state_dict().values()):
        a, b = getattr(a, "qdata", a), getattr(b, "qdata", b)
        assert torch.equal(a, b)

    def _drops_columns(
        checkpoint = None,
        config = None,
        **kw,
    ):
        out = _convert(checkpoint = checkpoint, config = config, **kw)
        out["blocks.0.to_out.weight"] = out["blocks.0.to_out.weight"].chunk(2, dim = 1)[0]
        return out

    monkeypatch.setattr(cq, "_mapping", lambda cls: (_drops_columns, _sfm()))
    with pytest.raises(ValueError, match = "column count"):
        _load(path, int8_backend = "torchao")


def test_the_hosted_route_registers_studios_single_file_converters(monkeypatch):
    """Qwen-Image-2.1's converter is Studio's own, registered by the single-file branch right before its call. The
    hosted prequant route never passes through there, so the row map must register it itself (it used to fail with
    "has no single-file converter" and fall back to quantizing the dense weights)."""
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "QwenImage21Transformer2DModel", None)
    if cls is None:
        pytest.skip("this diffusers has no QwenImage21Transformer2DModel")
    from diffusers.loaders import single_file_model as sfm

    monkeypatch.setattr(
        sfm,
        "SINGLE_FILE_LOADABLE_CLASSES",
        {k: v for k, v in sfm.SINGLE_FILE_LOADABLE_CLASSES.items() if k != cls.__name__},
    )
    mapping_fn, _ = cq._mapping(cls)
    assert callable(mapping_fn)


def test_a_local_comfy_override_is_usable_only_for_an_image_family(tmp_path, monkeypatch):
    """A local ComfyUI-format override reads its scheme from the header for an image family; a video family's loader
    cannot read the layout, so planning must not count on it there."""
    import core.inference.diffusion_prequant as pq
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

    path, _, _ = _fp8_file(tmp_path)
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda *a, **k: True)
    image = detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert pq.usable_prequant_source(image, "fp8", path_override = path) is not None
    assert pq.usable_prequant_source(image, "int8", path_override = path) is None
    video = next(f for f in VIDEO_FAMILIES if any(s == "fp8" for s, _r in (f.prequant_repos or ())))
    assert pq.usable_prequant_source(video, "fp8", path_override = path) is None


def test_only_torchao_comfy_loads_ask_for_compile(comfy_file):
    path, _, _ = comfy_file
    assert not cq.comfy_torchao_quantized(_load(path))
    assert not cq.comfy_torchao_quantized(_load(path, int8_backend = "native"))
    model = nn.Module()
    model._unsloth_comfy_quant = {"backend": "torchao", "int8": 8}
    assert cq.comfy_torchao_quantized(model)
    assert not cq.comfy_torchao_quantized(nn.Linear(2, 2))


def test_buffers_built_in_init_follow_the_compute_dtype(comfy_file):
    # Wan's rope tables are float64 non-persistent buffers; from_single_file casts them with model.to(dtype)
    path, _, _ = comfy_file

    class _Rope(_Tiny):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.register_buffer("freqs", torch.ones(4, dtype = torch.float64), persistent = False)

    model = cq.load_comfy_quant_transformer(
        _Rope,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo", "subfolder": "transformer"},
        int8_backend = "native",
        family = "z-image",
    )
    assert model.freqs.dtype == torch.bfloat16
    assert model.blocks[0].to_q.weight_q.dtype == torch.int8
