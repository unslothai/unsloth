from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path


def _load_csm_generation_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "unsloth"
        / "models"
        / "_csm_generation.py"
    )
    spec = importlib.util.spec_from_file_location("_csm_generation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_csm_depth_decoder_generation_signature_accepts_backbone_state():
    module = _load_csm_generation_module()

    class DepthDecoder:
        def __init__(self):
            self.forwarded_kwargs = None

        def forward(self, *args, **kwargs):
            return None

        def prepare_inputs_for_generation(
            self,
            input_ids,
            next_sequence_length = None,
            past_key_values = None,
            attention_mask = None,
            inputs_embeds = None,
            **kwargs,
        ):
            self.forwarded_kwargs = kwargs
            return input_ids, kwargs

    class Model:
        def __init__(self):
            self.depth_decoder = DepthDecoder()

    model = Model()
    module.patch_csm_depth_decoder_generate(model)

    prepare_parameters = inspect.signature(
        model.depth_decoder.prepare_inputs_for_generation
    ).parameters
    forward_parameters = inspect.signature(model.depth_decoder.forward).parameters
    model_args = set(prepare_parameters)
    if "kwargs" in model_args or "model_kwargs" in model_args:
        model_args |= set(forward_parameters)

    assert "backbone_last_hidden_state" in model_args

    input_ids = object()
    assert model.depth_decoder.prepare_inputs_for_generation(
        input_ids,
        backbone_last_hidden_state = "hidden-state",
        vision_specific_arg = "passed-through",
        extra = "kept",
    ) == (
        input_ids,
        {
            "backbone_last_hidden_state": "hidden-state",
            "vision_specific_arg": "passed-through",
            "extra": "kept",
        },
    )
    assert model.depth_decoder.forwarded_kwargs == {
        "backbone_last_hidden_state": "hidden-state",
        "vision_specific_arg": "passed-through",
        "extra": "kept",
    }


def test_csm_depth_decoder_generation_signature_inspection_failure_is_logged(caplog):
    module = _load_csm_generation_module()

    class DepthDecoder:
        prepare_inputs_for_generation = object()

    class Model:
        def __init__(self):
            self.depth_decoder = DepthDecoder()

    caplog.set_level(logging.DEBUG, logger = "_csm_generation")

    model = Model()
    module.patch_csm_depth_decoder_generate(model)

    assert not getattr(model.depth_decoder, "_unsloth_csm_generation_patched", False)
    assert "Could not inspect CSM depth decoder generation signature." in caplog.text
