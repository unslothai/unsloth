# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Dia generation keeps its own audio pad token instead of EOS (#2560)."""

import ast
import inspect
import os
from contextlib import nullcontext
from importlib.metadata import version as installed_version
from pathlib import Path
from types import SimpleNamespace

from packaging.version import Version


VISION_PATH = Path(__file__).parents[1] / "unsloth" / "models" / "vision.py"


def _load_function(name, namespace):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body = [function], type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace[name]


class _FakeTensor:
    shape = (1, 3)

    def to(self, dtype):
        return self


def _captured_generate_kwargs(config, **generate_kwargs):
    # A FlashAttention text config takes the delegate branch, which hands kwargs straight to _old_generate.
    config.architectures = ["DiaForConditionalGeneration"]
    config._attn_implementation = "flash_attention_2"
    namespace = {
        "torch": SimpleNamespace(
            Tensor = _FakeTensor,
            bfloat16 = "bfloat16",
            float16 = "float16",
            inference_mode = nullcontext,
            autocast = lambda **kwargs: nullcontext(),
        ),
        "os": os,
        "inspect": inspect,
        "FastBaseModel": SimpleNamespace(for_inference = lambda model: None),
        "dtype_from_config": lambda config: "bfloat16",
        "_get_dtype": lambda dtype: dtype,
        "_unsloth_generate_accepts_kwarg": lambda model, name: False,
        "NUM_LOGITS_TO_KEEP": {"DiaForConditionalGeneration": None},
        "DEVICE_TYPE_TORCH": "cuda",
        "Version": Version,
        "transformers_version": installed_version("transformers"),
        "_uses_flash_attention_for_generation": lambda config: True,
        "_clear_generation_caches": lambda model: None,
        "_is_text_seq2seq_config": lambda config: False,
    }
    fast_generate = _load_function("unsloth_base_fast_generate", namespace)
    captured = {}

    class Model:
        def forward(self, input_ids = None):
            return input_ids

        def _old_generate(self, *args, **kwargs):
            captured.update(kwargs)

    Model.config = config
    fast_generate(Model(), input_ids = _FakeTensor(), **generate_kwargs)
    return captured


def _dia_config():
    # Dia landed in transformers 4.53, below Unsloth's floor.
    from transformers import DiaConfig

    # Top-level ids as nari-labs/Dia-1.6B-0626's config.json sets them; default DiaConfig() leaves them None.
    return DiaConfig(pad_token_id = 1025, eos_token_id = 1024)


def test_dia_defaults_to_its_audio_pad_token():
    assert _captured_generate_kwargs(_dia_config())["pad_token_id"] == 1025


def test_dia_explicit_pad_token_id_wins():
    assert _captured_generate_kwargs(_dia_config(), pad_token_id = 7)["pad_token_id"] == 7


def test_other_models_still_default_to_eos():
    config = SimpleNamespace(model_type = "csm", eos_token_id = [2, 3], pad_token_id = 0)
    assert _captured_generate_kwargs(config)["pad_token_id"] == 2
