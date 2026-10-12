# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Qwen3.5 fused layer / GatedDeltaNet patches must fall back to the stock forward,
with the caller's kwargs, wherever they do not apply.

transformers >= 5.15 passes extra kwargs (`use_cache`, packing lengths) to
`Qwen3_5GatedDeltaNet.forward`, calls a module-level fla kernel instead of
`self.chunk_gated_delta_rule`, and drops `Qwen3_5DecoderLayer.layer_type`; the patched
forward raised `TypeError: unexpected keyword argument 'use_cache'` on every forward there.
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

_MODELS = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    pkg = types.ModuleType("_u35")
    pkg.__path__ = [_MODELS]
    sys.modules.setdefault("_u35", pkg)
    mods = []
    for name in ("_gated_delta_net", "_qwen3_5_layer"):
        spec = importlib.util.spec_from_file_location(
            f"_u35.{name}", os.path.join(_MODELS, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods


def _config():
    return Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 2,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        layer_types = ["linear_attention", "full_attention"],
    )


def _restore(*classes):
    for cls in classes:
        orig = cls.__dict__.get("_unsloth_original_forward")
        if orig is not None:
            cls.forward = orig
            del cls._unsloth_original_forward


def test_gdn_forward_matches_stock_with_extra_kwargs():
    gdn_mod, layer_mod = _load()
    torch.manual_seed(0)
    gdn = modeling.Qwen3_5GatedDeltaNet(_config(), layer_idx = 0).eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        ref = gdn(x)
    try:
        gdn_mod.patch_gated_delta_net_fast_forward(torch.nn.Sequential(gdn))
        engages = callable(getattr(gdn, "chunk_gated_delta_rule", None))
        if engages and gdn._unsloth_gdn_fast_ok and not torch.cuda.is_available():
            pytest.skip("fast path engages on this transformers; its compiled region needs a GPU")
        with torch.no_grad():
            out = gdn(x, use_cache = False)
    finally:
        _restore(type(gdn))
    torch.testing.assert_close(out, ref)


def test_fast_path_off_without_chunk_kernel_attribute():
    gdn_mod, _ = _load()
    gdn = modeling.Qwen3_5GatedDeltaNet(_config(), layer_idx = 0)
    if callable(getattr(gdn, "chunk_gated_delta_rule", None)):
        gdn.chunk_gated_delta_rule = None
    assert gdn_mod._fast_path_applicable(gdn) is False


def test_packed_kwargs_take_the_stock_path():
    gdn_mod, _ = _load()
    assert gdn_mod._is_packed({"cu_seq_lens_q": torch.tensor([0, 4, 8])})
    assert gdn_mod._is_packed({"seq_idx": torch.zeros(1, 8, dtype = torch.int32)})
    assert not gdn_mod._is_packed({"use_cache": False, "cu_seq_lens_q": None})


def test_decoder_layer_falls_back_when_not_applicable():
    gdn_mod, layer_mod = _load()
    torch.manual_seed(0)
    cfg = _config()
    layer = modeling.Qwen3_5DecoderLayer(cfg, layer_idx = 0).eval()
    rotary = modeling.Qwen3_5TextRotaryEmbedding(cfg)
    x = torch.randn(1, 8, 32)
    pos = torch.arange(8)[None, None].expand(3, 1, 8)
    pe = rotary(x, pos)
    with torch.no_grad():
        ref = layer(x, position_embeddings = pe, position_ids = pos)
    ref = ref[0] if isinstance(ref, tuple) else ref
    try:
        layer_mod.patch_qwen3_5_decoder_layers(torch.nn.Sequential(layer))
        gdn_mod.patch_gated_delta_net_fast_forward(torch.nn.Sequential(layer))
        engages = getattr(layer, "layer_type", None) is not None or callable(
            getattr(layer.linear_attn, "chunk_gated_delta_rule", None)
        )
        if engages and not torch.cuda.is_available():
            pytest.skip("fused path engages on this transformers; its compiled regions need a GPU")
        with torch.no_grad():
            out = layer(x, position_embeddings = pe, position_ids = pos, use_cache = False)
    finally:
        _restore(type(layer), type(layer.linear_attn))
    out = out[0] if isinstance(out, tuple) else out
    torch.testing.assert_close(out, ref)
