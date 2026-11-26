# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""generate passes output_router_logits = False when the config has it on (TRL < 1.18's SFTTrainer sets it for
MoE models), so a hybrid model's forward does not build the aux loss from generate's attention_mask dict."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
VISION_PATH = ROOT / "unsloth" / "models" / "vision.py"


def _load(*names):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(body = body, type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace


NS = _load("_router_logits_off_for_generate", "_unsloth_generate_accepts_kwarg")
off = NS["_router_logits_off_for_generate"]


class _Config:
    def __init__(
        self,
        flag,
        nested = True,
    ):
        self.text = SimpleNamespace(output_router_logits = flag) if nested else None
        if not nested:
            self.output_router_logits = flag

    def get_text_config(self):
        return self.text if self.text is not None else self


class _MoE:
    def __init__(self, config):
        self.config = config

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {}

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        output_router_logits = None,
        **kwargs,
    ):
        return None


class _NoRouterKwarg(_MoE):
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask = None,
    ):
        return {}

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
    ):
        return None


def test_flag_on_turns_router_logits_off_for_generate():
    kwargs = {"max_new_tokens": 8}
    off(_MoE(_Config(True)), kwargs)
    assert kwargs == {"max_new_tokens": 8, "output_router_logits": False}


def test_flag_on_in_a_flat_config():
    kwargs = {}
    off(_MoE(_Config(True, nested = False)), kwargs)
    assert kwargs == {"output_router_logits": False}


def test_flag_off_leaves_kwargs_untouched():
    kwargs = {"max_new_tokens": 8}
    off(_MoE(_Config(False)), kwargs)
    assert kwargs == {"max_new_tokens": 8}


def test_explicit_request_is_kept():
    kwargs = {"output_router_logits": True}
    off(_MoE(_Config(True)), kwargs)
    assert kwargs == {"output_router_logits": True}


def test_model_without_the_kwarg_is_left_alone():
    kwargs = {}
    off(_NoRouterKwarg(_Config(True)), kwargs)
    assert kwargs == {}


def test_fast_generate_applies_it_after_for_inference():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_base_fast_generate"
    )
    calls = [
        ast.unparse(s) for s in fn.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
    ]
    i = calls.index("FastBaseModel.for_inference(self)")
    assert calls[i + 1] == "_router_logits_off_for_generate(self, kwargs)"
