# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Anima kohya stems: splits whose rename names no registered module are ruled out before the per-split probe,
and the resolved path is the same."""

import sys
import types

from torch import nn

from core.inference import diffusion_lora_files as lf


def _install_converter(monkeypatch):
    def convert(state):
        return {
            "transformer." + key[len("diffusion_model.") :]: value for key, value in state.items()
        }

    module = types.ModuleType("diffusers.loaders.lora_conversion_utils")
    module._convert_non_diffusers_anima_lora_to_diffusers = convert
    for name in ("diffusers", "diffusers.loaders"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "diffusers.loaders.lora_conversion_utils", module)


def _pipe():
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = nn.Linear(2, 2)
    block.mlp = nn.Module()
    block.mlp.layer1 = nn.Linear(2, 2)
    root = nn.Module()
    root.blocks = nn.ModuleList([block])
    return types.SimpleNamespace(transformer = root)


def test_only_reachable_splits_are_probed(monkeypatch):
    _install_converter(monkeypatch)
    probed = []
    original = lf._lokr_targets

    def spy(pipe, path, family):
        probed.append(path)
        return original(pipe, path, family)

    monkeypatch.setattr(lf, "_lokr_targets", spy)
    pipe = _pipe()
    assert (
        lf._resolve_kohya_stem(pipe, "blocks_0_self_attn_q_proj", "anima")
        == "blocks.0.self_attn.q_proj"
    )
    assert lf._resolve_kohya_stem(pipe, "blocks_0_mlp_layer1", "anima") == "blocks.0.mlp.layer1"
    assert probed == ["blocks.0.self_attn.q_proj", "blocks.0.mlp.layer1"]


def test_a_stem_naming_no_layer_still_raises(monkeypatch):
    _install_converter(monkeypatch)
    try:
        lf._resolve_kohya_stem(_pipe(), "blocks_0_cross_attn_q_proj", "anima")
    except ValueError as err:
        assert "names no single layer" in str(err)
    else:
        raise AssertionError("expected ValueError")
