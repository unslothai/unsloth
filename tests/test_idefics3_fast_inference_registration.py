# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
from pathlib import Path

VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


def _vllm_supported_vlm():
    for node in ast.parse(VISION_PATH.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VLLM_SUPPORTED_VLM" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("VLLM_SUPPORTED_VLM not found")


def test_idefics3_passes_the_fast_inference_gate():
    # granite-docling-258M reports model_types ["idefics3", "idefics3_vision"]
    supported = _vllm_supported_vlm()
    assert any(arch in supported for arch in ("idefics3", "idefics3_vision"))
    assert len(supported) == len(set(supported))


def _probe(monkeypatch, layer_config):
    import unsloth_zoo.empty_model as empty_model
    from unsloth.models.vision import _zoo_supports_idefics3_fast_inference

    monkeypatch.setattr(empty_model, "get_model_layer_config", layer_config)
    return _zoo_supports_idefics3_fast_inference()


def test_zoo_with_idefics3_templates_is_accepted(monkeypatch):
    config = {"standard_layers": {"model.text_model.layers.{kk}.self_attn.q_proj"}}
    assert _probe(monkeypatch, lambda: config) is True


def test_older_zoo_is_refused_instead_of_crashing_in_conversion(monkeypatch):
    config = {"standard_layers": {"model.layers.{kk}.self_attn.q_proj"}}
    assert _probe(monkeypatch, lambda: config) is False


def test_broken_zoo_probe_is_refused(monkeypatch):
    def boom():
        raise KeyError("standard_layers")

    assert _probe(monkeypatch, boom) is False
