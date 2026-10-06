# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"
TREE = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))


def _module_constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _is_sparse_moe_config():
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_is_sparse_moe_config":
            namespace = {}
            exec(
                compile(ast.Module(body = [node], type_ignores = []), str(VISION_PATH), "exec"),
                namespace,
            )
            return namespace["_is_sparse_moe_config"]
    raise AssertionError("_is_sparse_moe_config not found")


GEMMA4_MOE = SimpleNamespace(text_config = SimpleNamespace(num_experts = 128, enable_moe_block = True))
GEMMA4_DENSE = SimpleNamespace(
    text_config = SimpleNamespace(num_experts = None, enable_moe_block = False)
)
QWEN3_5_MOE = SimpleNamespace(text_config = SimpleNamespace(num_experts = 256))
QWEN3_5_DENSE = SimpleNamespace(text_config = SimpleNamespace(num_experts = None))


def test_moe_model_types_pass_the_allowlist():
    supported = _module_constant("VLLM_SUPPORTED_VLM")
    for arch in ("qwen3_5_moe", "gemma4", "gemma4_text"):
        assert arch in supported
    assert len(supported) == len(set(supported))


def test_sparse_moe_is_read_off_the_text_config():
    is_moe = _is_sparse_moe_config()
    assert is_moe(GEMMA4_MOE) is True
    assert is_moe(QWEN3_5_MOE) is True
    assert is_moe(GEMMA4_DENSE) is False
    assert is_moe(QWEN3_5_DENSE) is False


def test_a_config_without_text_config_is_read_directly():
    is_moe = _is_sparse_moe_config()
    assert is_moe(SimpleNamespace(num_experts = 128)) is True
    assert is_moe(SimpleNamespace(enable_moe_block = True)) is True
    assert is_moe(SimpleNamespace()) is False


def test_dense_gemma4_is_refused_at_the_gate():
    """Dense Gemma-4 shares the gemma4 model type, so the allowlist alone would admit it."""
    assert set(_module_constant("VLLM_MOE_ONLY_VLM")) == {"gemma4", "gemma4_text"}
    gate, parents = _gate("not the dense ones yet")
    # text_only = True sets is_vlm_config False, so the gate must not sit under it.
    assert not any("is_vlm_config" in test for test in parents)
    only_moe = {"VLLM_MOE_ONLY_VLM": _module_constant("VLLM_MOE_ONLY_VLM"), "fast_inference": True}
    assert _evaluate(gate, model_types = ["gemma4"], auto_config = GEMMA4_DENSE, **only_moe)
    assert _evaluate(
        gate,
        model_types = ["gemma4_text", "gemma4"],
        auto_config = GEMMA4_DENSE.text_config,
        **only_moe,
    )
    assert not _evaluate(gate, model_types = ["gemma4"], auto_config = GEMMA4_MOE, **only_moe)
    assert not _evaluate(
        gate, model_types = ["gemma4_text"], auto_config = GEMMA4_MOE.text_config, **only_moe
    )
    assert not _evaluate(gate, model_types = ["qwen3_5"], auto_config = QWEN3_5_DENSE, **only_moe)
    assert not _evaluate(
        gate,
        **{**only_moe, "fast_inference": False},
        model_types = ["gemma4"],
        auto_config = GEMMA4_DENSE,
    )


def test_the_4bit_refusal_uses_the_same_moe_test():
    source = VISION_PATH.read_text(encoding = "utf-8")
    assert "and _is_sparse_moe_config(model_config)" in source


def _gate(message, condition = ""):
    """The `if` raising `message` (test mentions `condition`) plus its enclosing `if` tests."""

    def walk(node, parents):
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.If)
                and condition in ast.unparse(child.test)
                and any(
                    isinstance(stmt, ast.Raise) and message in ast.unparse(stmt)
                    for stmt in child.body
                )
            ):
                return child, parents
            found = walk(
                child, parents + [ast.unparse(child.test)] if isinstance(child, ast.If) else parents
            )
            if found:
                return found
        return None

    found = walk(TREE, [])
    assert found, message
    return found


def _evaluate(node, **names):
    namespace = {
        "VLLM_ZOO_MOE_VLM": _module_constant("VLLM_ZOO_MOE_VLM"),
        "_is_sparse_moe_config": _is_sparse_moe_config(),
        **names,
    }
    return eval(compile(ast.Expression(node.test), str(VISION_PATH), "eval"), namespace)


def test_the_zoo_gate_covers_text_only_moe_loads():
    """text_only = True sets is_vlm_config False, so the zoo gate must not sit under it."""
    gate, parents = _gate("needs a newer unsloth_zoo", "_zoo_supports_moe_fast_inference")
    assert not any("is_vlm_config" in test for test in parents)
    old_zoo = lambda: False
    for model_types, config in (
        (["qwen3_5_moe"], QWEN3_5_MOE.text_config),
        (["gemma4_text", "gemma4"], GEMMA4_MOE.text_config),
        (["qwen3_5_moe"], QWEN3_5_MOE),
    ):
        refused = _evaluate(
            gate,
            fast_inference = True,
            model_types = model_types,
            auto_config = config,
            _zoo_supports_moe_fast_inference = old_zoo,
        )
        assert refused, model_types
    assert not _evaluate(
        gate,
        fast_inference = True,
        model_types = ["gemma4_text"],
        auto_config = GEMMA4_DENSE,
        _zoo_supports_moe_fast_inference = old_zoo,
    )
    assert not _evaluate(
        gate,
        fast_inference = True,
        model_types = ["gemma4"],
        auto_config = GEMMA4_MOE,
        _zoo_supports_moe_fast_inference = lambda: True,
    )
    assert not _evaluate(
        gate,
        fast_inference = False,
        model_types = ["gemma4"],
        auto_config = GEMMA4_MOE,
        _zoo_supports_moe_fast_inference = old_zoo,
    )


def test_bnb_moe_loads_are_refused_without_load_in_4bit():
    gate, _ = _gate("does not support bitsandbytes weights")

    def quant_type(method):
        return lambda config: method

    def refused(
        load_in_4bit,
        model_name,
        method,
        config = GEMMA4_MOE,
        model_types = ("gemma4",),
        load_in_8bit = False,
    ):
        return bool(
            _evaluate(
                gate,
                load_in_4bit = load_in_4bit,
                load_in_8bit = load_in_8bit,
                model_name = model_name,
                get_quant_type = quant_type(method),
                model_config = config,
                model_types = list(model_types),
            )
        )

    assert refused(True, "google/gemma-4-26B-A4B-it", None)
    assert refused(False, "unsloth/gemma-4-26B-A4B-it-bnb-4bit", None)
    assert refused(False, "someone/gemma-4-26b-moe-4bit", "bitsandbytes")
    assert refused(False, "Qwen/Qwen3.6-35B-A3B", "bitsandbytes", QWEN3_5_MOE, ("qwen3_5_moe",))
    # load_vllm has no 8-bit mode, so the experts would silently load in 16-bit.
    assert refused(
        False, "Qwen/Qwen3.6-35B-A3B", None, QWEN3_5_MOE, ("qwen3_5_moe",), load_in_8bit = True
    )
    assert not refused(False, "google/gemma-4-26B-A4B-it", None)
    assert not refused(False, "Qwen/Qwen3.6-35B-A3B-FP8", "fp8", QWEN3_5_MOE, ("qwen3_5_moe",))
    assert not refused(False, "unsloth/gemma-4-E2B-it-bnb-4bit", "bitsandbytes", GEMMA4_DENSE)
