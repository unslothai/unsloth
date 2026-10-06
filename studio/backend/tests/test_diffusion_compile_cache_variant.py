# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compile-cache key tells quantized artifacts of one scheme apart.

Hosted INT8 and INT8-ConvRot both engage ``transformer_quant="int8"``, but ConvRot Linears trace an extra rotation,
so a bundle warmed by one variant and loaded for the other gave a warm start that ran ~2x slower per step instead of
a miss. The key now carries a structural descriptor of the denoiser (``graph_variant``) for quantized loads; dense
and GGUF keys are unchanged. CPU only: tiny Linears, torchao's real Int8Tensor where installed.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_compile_cache as cc  # noqa: E402
from core.inference.diffusion_convrot import (  # noqa: E402
    apply_activation_rotation,
    rotation_metadata,
)

_FP_KW = dict(
    family = "z-image",
    dtype = "torch.bfloat16",
    attention_backend = "_native_cudnn",
    compile_kwargs = {"fullgraph": True, "dynamic": True, "mode": "default"},
)


def _model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 256, bias = False), nn.LayerNorm(256), nn.Linear(256, 64))
    m._repeated_blocks = ["Linear"]
    return m.to(torch.bfloat16)


def _int8(act = True):
    tq = pytest.importorskip("torchao.quantization")
    m = _model()
    cfg = (
        tq.Int8DynamicActivationInt8WeightConfig(version = 2)
        if act
        else tq.Int8WeightOnlyConfig(version = 2)
    )
    tq.quantize_(m, cfg)
    return m


def _convrot(m):
    apply_activation_rotation(m, rotation_metadata(256, ["0"]))
    return m


def _key(transformer, quant = "int8"):
    return cc.cache_key(
        cc.environment_fingerprint(),
        cc.model_fingerprint(transformer = transformer, quant = quant, **_FP_KW),
    )


@pytest.fixture(autouse = True)
def _clean_switches(monkeypatch):
    for name in cc._GRAPH_SWITCHES:
        monkeypatch.delenv(name, raising = False)


def test_identical_config_keys_the_same_across_builds():
    a, b = _int8(), _int8()
    assert _key(a) == _key(b)
    # Survives a JSON round trip (the manifest's exact-match guard compares the parsed dict).
    fp = cc.model_fingerprint(transformer = a, quant = "int8", **_FP_KW)
    assert json.loads(json.dumps(fp, sort_keys = True, default = str)) == fp


def test_plain_and_convrot_int8_key_apart():
    plain = _int8()
    rotated = _convrot(_int8())
    assert _key(plain) != _key(rotated)
    variant = cc.graph_variant(rotated)
    assert variant["rotation"] == {"group": 256, "kind": "convrot_hadamard_v1", "linears": 1}
    assert any(sig.startswith("ConvRotLinear/") for sig in variant["weights"])
    assert "rotation" not in cc.graph_variant(plain)


def test_w8a8_and_weight_only_key_apart():
    assert _key(_int8(act = True)) != _key(_int8(act = False))


def test_runtime_quantise_and_prequant_scale_dtype_key_apart():
    # A v2 hosted checkpoint stores fp32 scales; a runtime quantise or a v1 file carries bf16 ones. Different
    # epilogue graph, so different key.
    a, b = _int8(), _int8()
    w = b[0].weight
    if type(w).__name__ != "Int8Tensor":
        pytest.skip("this torchao predates Int8Tensor")
    other = torch.float32 if w.scale.dtype == torch.bfloat16 else torch.bfloat16
    w.scale = w.scale.to(other)
    if b[0].weight.scale.dtype != other:
        pytest.skip("this torchao does not take a scale reassignment")
    assert _key(a) != _key(b)


def test_a_native_twin_keys_apart_from_torchao():
    # A ComfyUI file served by native layers instead of torchao: different module class, different key.
    a = _int8()
    b = _int8()
    native_cls = type("NativeInt8Linear", (nn.Linear,), {})
    b[0].__class__ = native_cls
    assert _key(a) != _key(b)


def test_an_instance_forward_swap_keys_apart():
    a, b = _int8(), _int8()

    def fused_forward(self, x):  # stands in for a fused-kernel swap installed at load
        return x

    b[0].forward = types.MethodType(fused_forward, b[0])
    assert _key(a) != _key(b)


@pytest.mark.parametrize("name,default", sorted(cc._GRAPH_SWITCHES.items()))
@pytest.mark.parametrize("quant", ["int8", None])
def test_graph_switches_key_apart_only_off_their_default(monkeypatch, name, default, quant):
    m = _int8() if quant else _model()
    k0 = _key(m, quant)
    for same in ("", "auto") + ((default,) if default in ("on", "off") else ()):
        monkeypatch.setenv(name, same)
        assert _key(m, quant) == k0, same
    flipped = {"on": "0", "off": "1", "auto": "0", "raw": "torchao"}[default]
    monkeypatch.setenv(name, flipped)
    assert _key(m, quant) != k0
    assert name in cc.model_fingerprint(transformer = m, quant = quant, **_FP_KW)["switches"]


def test_a_default_environment_adds_no_switches():
    assert cc.graph_switches() == {}
    assert "switches" not in cc.model_fingerprint(transformer = _model(), quant = None, **_FP_KW)


def test_a_comfy_single_file_with_no_engaged_scheme_still_keys_its_layout():
    # ComfyUI single-file loads engage no scheme (quant reads none), so the tree itself has to key them apart.
    dense = _model()
    comfy = _int8()
    fp_dense = cc.model_fingerprint(transformer = dense, quant = None, **_FP_KW)
    fp_comfy = cc.model_fingerprint(transformer = comfy, quant = None, **_FP_KW)
    assert "variant" not in fp_dense
    assert "variant" in fp_comfy
    assert _key(dense, None) != _key(comfy, None)
    assert _key(comfy, None) != _key(_convrot(_int8()), None)


def test_an_offload_hook_forward_alone_keeps_a_dense_key():
    m = _model()
    k0 = _key(m, None)
    m[0].forward = types.MethodType(lambda self, x: x, m[0])
    assert "variant" not in cc.model_fingerprint(transformer = m, quant = None, **_FP_KW)
    assert _key(m, None) == k0


def test_dense_and_gguf_keys_are_unchanged():
    # No variant on a dense or GGUF load, so their existing bundles keep hitting.
    m = _model()
    for quant in (None, "gguf"):
        fp = cc.model_fingerprint(transformer = m, quant = quant, **_FP_KW)
        assert "variant" not in fp


def test_unreadable_tree_never_raises():
    variant = cc.graph_variant(types.SimpleNamespace())
    assert variant["weights"].startswith("unavailable")


def test_a_bundle_saved_for_one_variant_is_a_miss_for_the_other(monkeypatch, tmp_path):
    store: dict = {}
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"artifacts", object()), raising = False
    )

    def fake_load(data):
        store["loaded"] = data
        return object()

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", fake_load, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    monkeypatch.setenv(cc._ENV_MODE, "1")
    rotated = cc.begin(transformer = _convrot(_int8()), quant = "int8", **_FP_KW)
    assert rotated is not None and not rotated.hit
    assert cc.save(rotated) is True
    cc.restore(rotated)
    plain = cc.begin(transformer = _int8(), quant = "int8", **_FP_KW)
    assert plain is not None and not plain.hit and plain.key != rotated.key
    assert "loaded" not in store
    cc.restore(plain)
    again = cc.begin(transformer = _convrot(_int8()), quant = "int8", **_FP_KW)
    assert again is not None and again.hit and again.key == rotated.key
    cc.restore(again)
