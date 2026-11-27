# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 LoRA key layouts: every trainer's spelling of one adapter applies the same deltas."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Ideogram4Pipeline"):
    pytest.skip("this diffusers has no Ideogram 4 pipeline", allow_module_level = True)

from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_lora_files as lf  # noqa: E402

FAMILY = "ideogram-4"
LAYERS, HIDDEN, FFN, RANK, ALPHA = 2, 16, 24, 4, 2.0
NATIVE = ("attention.qkv", "attention.o", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
SHAPES = {
    "attention.qkv": (3 * HIDDEN, HIDDEN),
    "attention.o": (HIDDEN, HIDDEN),
    "feed_forward.w1": (FFN, HIDDEN),
    "feed_forward.w2": (HIDDEN, FFN),
    "feed_forward.w3": (FFN, HIDDEN),
}


def _pipe():
    torch.manual_seed(0)
    transformer = diffusers.Ideogram4Transformer2DModel(
        in_channels = 4,
        num_layers = LAYERS,
        attention_head_dim = 8,
        num_attention_heads = 2,
        intermediate_size = FFN,
        adaln_dim = 8,
        llm_features_dim = 8,
        mrope_section = (2, 1, 1),
    )
    return diffusers.Ideogram4Pipeline(
        scheduler = None,
        vae = None,
        text_encoder = None,
        tokenizer = None,
        transformer = transformer,
        unconditional_transformer = None,
    )


def _adapter():
    """One adapter in the checkpoint's own terms: ``(A, B)`` per fused module, its delta ``ALPHA / RANK * B @ A``."""
    g = torch.Generator().manual_seed(7)
    pairs = {}
    for i in range(LAYERS):
        for module in NATIVE:
            out_f, in_f = SHAPES[module]
            pairs[f"layers.{i}.{module}"] = (
                torch.randn(RANK, in_f, generator = g),
                torch.randn(out_f, RANK, generator = g),
            )
    return pairs


def _expected(pairs):
    """Delta per diffusers module: the fused ``qkv`` delta's thirds are ``to_q`` / ``to_k`` / ``to_v``."""
    out = {}
    for stem, (a, b) in pairs.items():
        layer, _, module = (
            stem.partition(".attention.") if ".attention." in stem else (stem, "", "")
        )
        delta = (ALPHA / RANK) * b @ a
        if module == "qkv":
            for name, part in zip(("to_q", "to_k", "to_v"), delta.chunk(3, dim = 0)):
                out[f"{layer}.attention.{name}"] = part
        elif module == "o":
            out[f"{layer}.attention.to_out.0"] = delta
        else:
            out[stem] = delta
    return out


def _folded(pairs, prefix):
    """ai-toolkit / PEFT style: ``lora_A`` / ``lora_B`` with the alpha scale already in ``lora_B``."""
    sd = {}
    for stem, (a, b) in pairs.items():
        sd[f"{prefix}{stem}.lora_A.weight"] = a
        sd[f"{prefix}{stem}.lora_B.weight"] = b * (ALPHA / RANK)
    return sd


def _kohya(pairs):
    """musubi-tuner's networks.lora_ideogram4 save: ``lora_unet_`` + the path with underscores, ``.alpha`` per module."""
    sd = {}
    for stem, (a, b) in pairs.items():
        name = "lora_unet_" + stem.replace(".", "_")
        sd[f"{name}.lora_down.weight"] = a
        sd[f"{name}.lora_up.weight"] = b
        sd[f"{name}.alpha"] = torch.tensor(ALPHA)
    return sd


def _diffusers_split(pairs):
    """diffusers' own module names (split q / k / v, ``to_out.0``) under ``transformer.``, as its training script saves."""
    sd = {}
    for stem, (a, b) in pairs.items():
        layer, _, module = (
            stem.partition(".attention.") if ".attention." in stem else (stem, "", "")
        )
        b = b * (ALPHA / RANK)
        if module == "qkv":
            for name, part in zip(("to_q", "to_k", "to_v"), b.chunk(3, dim = 0)):
                sd[f"transformer.{layer}.attention.{name}.lora_A.weight"] = a.clone()
                sd[f"transformer.{layer}.attention.{name}.lora_B.weight"] = part.contiguous()
        elif module == "o":
            sd[f"transformer.{layer}.attention.to_out.0.lora_A.weight"] = a
            sd[f"transformer.{layer}.attention.to_out.0.lora_B.weight"] = b
        else:
            sd[f"transformer.{stem}.lora_A.weight"] = a
            sd[f"transformer.{stem}.lora_B.weight"] = b
    return sd


LAYOUTS = {
    "ai_toolkit": lambda p: _folded(p, "diffusion_model."),
    "transformer_fused": lambda p: _folded(p, "transformer."),
    "musubi_kohya": _kohya,
    "diffusers_split": _diffusers_split,
}


def _deltas(pipe, name):
    out = {}
    for module_name, module in pipe.transformer.named_modules():
        if hasattr(module, "get_delta_weight") and name in getattr(module, "lora_A", {}):
            out[module_name] = module.get_delta_weight(name)
    return out


def _load(
    tmp_path,
    sd,
    metadata = None,
):
    path = tmp_path / "adapter.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path), metadata = metadata)
    pipe = _pipe()
    lf.load_adapter(pipe, str(path), "a", FAMILY)
    return pipe


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_every_layout_applies_the_same_deltas(tmp_path, layout):
    pairs = _adapter()
    expected = _expected(pairs)
    got = _deltas(_load(tmp_path, LAYOUTS[layout](pairs)), "a")
    assert sorted(got) == sorted(expected)
    for module_name, delta in expected.items():
        torch.testing.assert_close(got[module_name], delta, rtol = 1e-5, atol = 1e-5)


def test_diffusers_names_load_by_path(tmp_path):
    # Its feed_forward / adaln_modulation names are the checkpoint's too: without a fused qkv / o nothing is renamed.
    assert lf._ideogram4_renames(list(_diffusers_split(_adapter()))) == {}
    assert lf._ideogram4_renames(list(_folded(_adapter(), "diffusion_model."))) == {}


def test_a_transformer_prefixed_fused_file_keeps_its_metadata_strength(tmp_path):
    # A PEFT save carries lora_alpha in diffusers' header, which a renamed dict would otherwise drop.
    pairs = _adapter()
    sd = {}
    for stem, (a, b) in pairs.items():
        sd[f"transformer.{stem}.lora_A.weight"] = a
        sd[f"transformer.{stem}.lora_B.weight"] = b
    header = {
        "lora_adapter_metadata": json.dumps(
            {"transformer.r": RANK, "transformer.lora_alpha": ALPHA}
        )
    }
    got = _deltas(_load(tmp_path, sd, header), "a")
    for module_name, delta in _expected(pairs).items():
        torch.testing.assert_close(got[module_name], delta, rtol = 1e-5, atol = 1e-5)


def test_a_file_loading_into_no_layer_is_refused(tmp_path):
    # diffusers keeps only ``transformer.`` keys and drops the rest without an error, so this loaded nothing.
    sd = {
        "lora_unet_final_layer_linear.lora_down.weight": torch.randn(RANK, HIDDEN),
        "lora_unet_final_layer_linear.lora_up.weight": torch.randn(4, RANK),
    }
    with pytest.raises(ValueError, match = "none of its LoRA keys"):
        _load(tmp_path, sd)
