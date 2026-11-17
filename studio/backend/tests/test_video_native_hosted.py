# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An offloaded LTX-2.5 int8 load reads the hosted ConvRot int8 file as native Linears instead of quantising the
dense shards: same layer set, rotation and activation path as the runtime native build, codes and scales from the
file, verified against its recorded fingerprint, and refused (dense build) whenever the file is not that build."""

from __future__ import annotations

import gc
import json
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")
if not hasattr(diffusers, "LTX2VideoTransformer3DModel"):
    pytest.skip("diffusers without LTX-2", allow_module_level = True)

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_native_hosted as NH  # noqa: E402
from core.inference import video as V  # noqa: E402
from core.inference.diffusion_convrot import (  # noqa: E402
    CONVROT_KIND,
    ROTATION_FQNS_KEY,
    ROTATION_GROUP_KEY,
    ROTATION_KEY,
    quantize_convrot_int8_weight,
)
from core.inference.diffusion_native_cache import native_spec  # noqa: E402
from core.inference.diffusion_native_quant import (  # noqa: E402
    apply_native_weight_quant,
    is_native_linear,
)
from core.inference.diffusion_prequant import (  # noqa: E402
    CANONICAL_FINGERPRINT_ALGO,
    HostedInt8Parts,
    canonical_weight_digest,
    canonical_weight_fingerprint,
)
from core.inference.diffusion_transformer_quant import make_filter_fn  # noqa: E402
from core.inference.prequant_safetensors import UNSLOTH_ROOT_KEYS_KEY, UNSLOTH_ROOT_PREFIX  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

FAMILY = "ltx-2.5"
CLS = diffusers.LTX2VideoTransformer3DModel


def _config():
    # 2 heads x 256 = 512 wide: the video Linears pass the 512 filter and g256 ConvRot divides them.
    return dict(
        in_channels = 32,
        out_channels = 32,
        num_attention_heads = 2,
        attention_head_dim = 256,
        cross_attention_dim = 512,
        caption_channels = 512,
        audio_in_channels = 16,
        audio_out_channels = 16,
        audio_num_attention_heads = 2,
        audio_attention_head_dim = 32,
        audio_cross_attention_dim = 64,
        num_layers = 1,
    )


def _spec():
    return native_spec("int8", FAMILY, act_int8 = True)


def _dense():
    torch.manual_seed(0)
    return CLS(**_config()).to(torch.bfloat16).eval()


def _filter(spec, require_bf16 = False):
    return make_filter_fn(
        int(spec["min_features"]), tuple(spec["exclude_name_tokens"]), require_bf16 = require_bf16
    )


def _write_hosted(
    root: Path,
    *,
    mutate = None,
    meta_patch = None,
) -> tuple[Path, Path, dict]:
    """A tiny hosted-format int8 ConvRot file (torchao Int8Tensor layout) plus the dense config dir."""
    from safetensors.torch import save_file

    spec = _spec()
    model = _dense()
    cfg_dir = root / "base"
    model.save_config(str(cfg_dir / "transformer"))
    group = int(spec["rot_group"])
    keep = _filter(spec, require_bf16 = True)
    quantised = sorted(n for n, m in model.named_modules() if n and keep(m, n))
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    rotated = []
    parts = {}
    for fqn in quantised:
        weight = state.pop(fqn + ".weight")
        if weight.shape[1] % group == 0:
            codes, scale = quantize_convrot_int8_weight(weight, group)
            rotated.append(fqn)
        else:
            w = weight.float()
            scale = w.abs().amax(dim = 1, keepdim = True) / 127.0
            codes = (w / scale).round().clamp(-128, 127).to(torch.int8)
        zero = torch.zeros(codes.shape[0], 1, dtype = torch.int8)
        state[fqn + "._weight_qdata"] = codes.contiguous()
        state[fqn + "._weight_scale"] = scale.float().contiguous()
        state[fqn + "._weight_zero_point"] = zero
        parts[fqn + ".weight"] = HostedInt8Parts(codes, scale.float(), zero)
    fingerprint = canonical_weight_fingerprint(parts)
    meta = {
        "base_model_id": "Lightricks/LTX-2.5-Diffusers",
        "family": FAMILY,
        "scheme": "int8",
        "min_features": spec["min_features"],
        "exclude_name_tokens": [t for t in spec["exclude_name_tokens"] if t != "lora_"],
        "torch_dtype": "bfloat16",
        "quant_backend": "torchao",
        "transformer_class": "LTX2VideoTransformer3DModel",
        "component": "transformer",
        "fingerprint": {
            "algo": CANONICAL_FINGERPRINT_ALGO,
            "count": fingerprint["count"],
            "canonical_modules": fingerprint["canonical_modules"],
        },
        ROTATION_KEY: CONVROT_KIND,
        ROTATION_GROUP_KEY: group,
        ROTATION_FQNS_KEY: rotated,
    }
    if meta_patch:
        meta.update(meta_patch)
    if mutate:
        mutate(state)
    # The hosted layout: root-level tensors under a prefix (unflatten splits keys on dots).
    roots = [k for k in state if "." not in k]
    assert roots, "the tiny model should have a root-level tensor (scale_shift_table)"
    stored = {(f"{UNSLOTH_ROOT_PREFIX}{k}" if k in roots else k): v for k, v in state.items()}
    path = root / "LTX-2.5-INT8-ConvRot.safetensors"
    save_file(
        stored,
        str(path),
        metadata = {"unsloth_metadata": json.dumps(meta), UNSLOTH_ROOT_KEYS_KEY: json.dumps(roots)},
    )
    return path, cfg_dir, {"quantised": quantised, "rotated": rotated, "state": state}


def _load(
    path,
    cfg_dir,
    spec = None,
    family = FAMILY,
):
    return NH.load_native_from_hosted(
        str(path), CLS, str(cfg_dir), spec = spec or _spec(), family = family, logger = None
    )


def test_reads_the_hosted_codes_into_the_runtime_native_layout(tmp_path):
    path, cfg_dir, built = _write_hosted(tmp_path)
    model = _load(path, cfg_dir)
    assert model is not None
    native = {n: m for n, m in model.named_modules() if is_native_linear(m)}
    assert sorted(native) == built["quantised"]
    for fqn, layer in native.items():
        assert layer.act_int8 and layer.scheme == "int8"
        assert layer.rot_group == (256 if fqn in built["rotated"] else 0)
        assert torch.equal(layer.weight_q, built["state"][fqn + "._weight_qdata"])
        assert torch.equal(
            layer.weight_scale.view(torch.float32),
            built["state"][fqn + "._weight_scale"].reshape(-1),
        )
    # The runtime native build of the dense model swaps exactly these layers, the same way.
    runtime = _dense()
    apply_native_weight_quant(
        runtime, "int8", filter_fn = _filter(_spec(), require_bf16 = True), act_int8 = True
    )
    expect = {n: m for n, m in runtime.named_modules() if is_native_linear(m)}
    assert sorted(expect) == sorted(native)
    for fqn, layer in expect.items():
        assert (layer.rot_group, layer.act_int8, layer.in_features, layer.out_features) == (
            native[fqn].rot_group,
            native[fqn].act_int8,
            native[fqn].in_features,
            native[fqn].out_features,
        )
    # Every dense (non-quantised) tensor is the dense model's, bit for bit, in bf16.
    dense = runtime.state_dict()
    for key, value in model.state_dict().items():
        if key.endswith((".weight_q", ".weight_scale")):
            continue
        assert value.dtype == dense[key].dtype and torch.equal(value, dense[key]), key
    assert model._unsloth_runtime_quant == "int8"


@pytest.mark.parametrize(
    "case",
    [
        "rotation_group",
        "family",
        "unrotated_kind",
        "excludes",
        "missing_layer",
        "static_act",
        "corrupt_codes",
    ],
)
def test_refuses_a_file_that_is_not_the_native_build(tmp_path, case):
    first = None

    def drop_layer(state):
        nonlocal first
        first = sorted(k for k in state if k.endswith("._weight_qdata"))[0][
            : -len("._weight_qdata")
        ]
        for suffix in ("._weight_qdata", "._weight_scale", "._weight_zero_point"):
            state.pop(first + suffix)

    def static_act(state):
        key = sorted(k for k in state if k.endswith("._weight_qdata"))[0]
        state[key[: -len("._weight_qdata")] + "._weight_act_quant_scale"] = torch.ones(1)

    def corrupt(state):
        key = sorted(k for k in state if k.endswith("._weight_qdata"))[0]
        state[key] = state[key].clone()
        state[key][0, 0] = -state[key][0, 0] if state[key][0, 0] != 0 else 1

    kwargs = {
        "rotation_group": {"meta_patch": {ROTATION_GROUP_KEY: 64}},
        "family": {"meta_patch": {"family": "ltx-2"}},
        "unrotated_kind": {"meta_patch": {ROTATION_KEY: None}},
        "excludes": {"meta_patch": {"exclude_name_tokens": ["norm"]}},
        "missing_layer": {"mutate": drop_layer},
        "static_act": {"mutate": static_act},
        "corrupt_codes": {"mutate": corrupt},
    }[case]
    path, cfg_dir, _ = _write_hosted(tmp_path, **kwargs)
    assert _load(path, cfg_dir) is None


def test_refuses_a_weight_only_or_unrotated_native_build(tmp_path):
    path, cfg_dir, _ = _write_hosted(tmp_path)
    assert _load(path, cfg_dir, spec = native_spec("int8", FAMILY, act_int8 = False)) is None
    assert _load(path, cfg_dir, spec = native_spec("fp8", FAMILY, act_int8 = False)) is None


def test_hosted_parts_fingerprint_like_the_torchao_int8_tensor():
    torchao_q = pytest.importorskip("torchao.quantization")
    if not hasattr(torchao_q, "Int8Tensor"):
        pytest.skip("torchao without Int8Tensor")
    from torchao.quantization.granularity import PerRow

    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype = torch.bfloat16)
    tensor = torchao_q.Int8Tensor.from_hp(weight, PerRow())
    parts = HostedInt8Parts(tensor.qdata, tensor.scale, getattr(tensor, "zero_point", None))
    assert canonical_weight_digest(tensor) == canonical_weight_digest(parts) is not None


def _target():
    return types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)


def test_only_an_opted_in_family_offload_int8_takes_the_hosted_file(monkeypatch):
    monkeypatch.setattr(V, "native_quant_scheme", lambda *a, **k: "int8")
    monkeypatch.setattr(V, "native_int8_act", lambda target: True)
    ltx25 = detect_video_family("Lightricks/LTX-2.5-Diffusers")
    assert ltx25 is not None and ltx25.name == FAMILY
    assert V._video_native_hosted_scheme(ltx25, "int8", _target()) == "int8"
    assert V._video_native_hosted_scheme(ltx25, "fp8", _target()) is None
    cpu = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, ordinal = None)
    assert V._video_native_hosted_scheme(ltx25, "int8", cpu) is None
    wan = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert wan is not None and V._video_native_hosted_scheme(wan, "int8", _target()) is None
    monkeypatch.setattr(V, "native_int8_act", lambda target: False)
    assert V._video_native_hosted_scheme(ltx25, "int8", _target()) is None
    monkeypatch.setattr(V, "native_int8_act", lambda target: True)
    monkeypatch.setenv(NH.NATIVE_HOSTED_ENV, "0")
    assert V._video_native_hosted_scheme(ltx25, "int8", _target()) is None


def _file_mapped(path) -> bool:
    real = os.path.realpath(str(path))
    with open("/proc/self/maps") as maps:
        return any(line.rstrip("\n").endswith(real) for line in maps)


def test_offload_repoint_releases_the_hosted_file(tmp_path):
    # Group offload re-points each weight at its pinned copy; a view's base would keep the file mapped.
    path, cfg_dir, _built = _write_hosted(tmp_path)
    model = _load(path, cfg_dir)
    assert model is not None
    tensors = [*model.parameters(), *model.buffers()]
    assert all(t._base is None for t in tensors)
    if not os.path.exists("/proc/self/maps"):
        return
    assert _file_mapped(path)
    for t in tensors:
        t.data = t.data.clone()
    del t, tensors
    gc.collect()
    assert not _file_mapped(path)
