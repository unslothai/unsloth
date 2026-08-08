# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`save_lora` must be attached with or without a vLLM engine.

`patch_peft_fast_inference` set it only inside `if vllm_engine is not None`,
but unsloth_zoo's `save_lora` is `save_pretrained` over the lora_A/lora_B keys
and never touches the engine. So `LFM2.5_(1.2B)-GRPO`, which loads with
`fast_inference = False` and saves at the end, got `AttributeError:
'Lfm2ForCausalLM' object has no attribute 'save_lora'`, naming neither vLLM nor
the flag that caused it. `load_lora` stays gated: it copies into vLLM's own
adapter buffers.

Source-level, because importing the module pulls the whole model stack.
"""

import ast
import pathlib

import pytest

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _patch_function():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "patch_peft_fast_inference":
            return node
    pytest.fail("patch_peft_fast_inference has moved or been renamed")


def _assigned_attributes(scope):
    """Every `model.<name> = ...` target inside `scope`."""
    found = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == "model":
                    found.add(target.attr)
    return found


def _engine_guard(function):
    """The `if vllm_engine is not None:` block."""
    for node in function.body:
        if isinstance(node, ast.If) and "vllm_engine" in ast.unparse(node.test):
            return node
    pytest.fail("the vllm_engine guard has moved or been renamed")


def test_save_lora_is_not_behind_the_engine_guard():
    function = _patch_function()
    guard = _engine_guard(function)
    assert "save_lora" not in _assigned_attributes(guard), (
        "save_lora is set only when a vLLM engine exists, so fast_inference=False "
        "leaves the model without it"
    )


def test_save_lora_is_still_set_somewhere_in_the_function():
    function = _patch_function()
    assert "save_lora" in _assigned_attributes(function), "save_lora is no longer set at all"


def test_load_lora_stays_behind_the_engine_guard():
    """It writes into vLLM's adapter tensors, so it needs one."""
    function = _patch_function()
    guard = _engine_guard(function)
    assert "load_lora" in _assigned_attributes(guard)
    outside = _assigned_attributes(function) - _assigned_attributes(guard)
    assert "load_lora" not in outside


def test_fast_generate_stays_behind_the_engine_guard():
    """The other engine-only attributes must not have been loosened too."""
    function = _patch_function()
    guard = _assigned_attributes(_engine_guard(function))
    for name in ("vllm_engine", "fast_generate", "fast_generate_batches"):
        assert name in guard, f"{name} escaped the engine guard"


def test_an_existing_save_lora_is_not_replaced():
    """Set only when absent, so a model that already carries one keeps it."""
    function = _patch_function()
    source = ast.unparse(function)
    assert 'hasattr(model, "save_lora")' in source or "hasattr(model, 'save_lora')" in source


def test_a_missing_zoo_helper_does_not_break_loading():
    """Older unsloth_zoo has no `save_lora`; that must not break loading."""
    function = _patch_function()
    handlers = [node for node in ast.walk(function) if isinstance(node, ast.Try)]
    assert handlers, "the save_lora import is unguarded, so an older zoo raises on load"
    guarded = any("save_lora" in ast.unparse(node) for node in handlers)
    assert guarded, "the guarded import does not cover save_lora"
