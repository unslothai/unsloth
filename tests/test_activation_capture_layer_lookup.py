# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capture hooks the text decoder's layers, never a vision or audio tower's, whatever the nesting."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "unsloth" / "activation_capture.py"
_spec = importlib.util.spec_from_file_location("activation_capture_lookup_under_test", _SRC)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


class _Text(torch.nn.Module):
    pass


class _Vision(torch.nn.Module):
    pass


def _stack(cls, n):
    return torch.nn.ModuleList([cls() for _ in range(n)])


def _node(**children):
    m = torch.nn.Module()
    for name, child in children.items():
        setattr(m, name, child)
    return m


def _model(inner):
    m = _node(model = inner)
    m.config = SimpleNamespace(hidden_size = 8, intermediate_size = 16, _name_or_path = "tiny")
    return m


LAYOUTS = {
    # Idefics3 / SmolVLM: text under text_model, which the attribute unwrap never visited.
    "text_model": _node(
        vision_model = _node(encoder = _node(layers = _stack(_Vision, 4))),
        text_model = _node(layers = _stack(_Text, 2)),
    ),
    # Vision tower declared first, deeper vision stack with more layers than the text one.
    "vision_first": _node(
        vision_tower = _node(layers = _stack(_Vision, 6)), language_model = _node(layers = _stack(_Text, 2))
    ),
    # An audio tower alongside the decoder.
    "audio": _node(audio_tower = _node(layers = _stack(_Vision, 3)), layers = _stack(_Text, 2)),
    # A plain causal LM.
    "plain": _node(layers = _stack(_Text, 2)),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_text_decoder_layers_are_hooked(tmp_path, layout):
    cap = ac.ActivationCapture(
        _model(LAYOUTS[layout]),
        ac.ActivationCaptureConfig(output_dir = str(tmp_path), capture_lora_norms = False),
    )
    assert {type(layer) for layer in cap._layers} == {_Text}


def test_a_model_without_a_text_stack_is_refused(tmp_path):
    only_vision = _model(_node(vision_tower = _node(layers = _stack(_Vision, 2))))
    with pytest.raises(ValueError, match = "could not locate"):
        ac.ActivationCapture(only_vision, ac.ActivationCaptureConfig(output_dir = str(tmp_path)))
