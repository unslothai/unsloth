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


# Shaped like the real config.json files: model_type gemma4 / qwen3_5_moe at the top, the
# expert count on text_config.
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
    # The text-only decoder path hands over the text config itself.
    is_moe = _is_sparse_moe_config()
    assert is_moe(SimpleNamespace(num_experts = 128)) is True
    assert is_moe(SimpleNamespace(enable_moe_block = True)) is True
    assert is_moe(SimpleNamespace()) is False


def test_dense_gemma4_is_refused_at_the_gate():
    """Dense Gemma-4 shares the gemma4 model type, so the allowlist alone would admit it. It is
    not served yet: vLLM aborts the whole process profiling its audio encoder (`Cannot call
    numel() on tensor with symbolic sizes/strides`), and in 4-bit it reaches a bitsandbytes
    loader vLLM >= 0.28 moved out of tree. Pin that the gate refuses it before loading."""
    assert set(_module_constant("VLLM_MOE_ONLY_VLM")) == {"gemma4", "gemma4_text"}
    source = VISION_PATH.read_text(encoding = "utf-8")
    gate = "if any(arch in VLLM_MOE_ONLY_VLM for arch in model_types) and not _is_sparse_moe_config(auto_config):"
    assert gate in source
    allowlist = source.index("if not any(arch in VLLM_SUPPORTED_VLM for arch in model_types):")
    assert allowlist < source.index(gate) < source.index("llm = load_vllm(**load_vllm_kwargs)")


def test_the_4bit_refusal_uses_the_same_moe_test():
    source = VISION_PATH.read_text(encoding = "utf-8")
    assert "and _is_sparse_moe_config(model_config)" in source
