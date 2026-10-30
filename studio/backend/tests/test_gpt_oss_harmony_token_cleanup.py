# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""gpt-oss cleanup drops only harmony protocol tokens; a literal <|im_end|> in an answer is kept."""

from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "core/inference/inference.py"


def _cleanup():
    """The module's `import re`, regex and _clean_generated_text, without importing torch/unsloth."""
    tree = ast.parse(_SRC.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "re" for alias in node.names)
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_HARMONY_PROTOCOL_TOKEN_RE" for t in node.targets
        ):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_clean_generated_text":
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_SRC), "exec"), namespace)

    class _GptOss:
        def _is_gpt_oss_model(self):
            return True

    return lambda text: namespace["_clean_generated_text"](_GptOss(), text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is 4.<|return|>", "The answer is 4."),
        ("<|start|>assistant<|channel|>final<|message|>Hi<|end|>", "assistantfinalHi"),
        ("Done<|call|>", "Done"),
        ("x<|endoftext|>", "x"),
    ],
)
def test_harmony_protocol_tokens_are_still_removed(text, expected):
    assert _cleanup()(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "parts = text.split('<|im_end|>')",
        "Llama 3 ends a turn with <|eot_id|> and calls tools after <|python_tag|>.",
        "ChatML opens a turn with <|im_start|>user",
    ],
)
def test_other_models_markers_in_an_answer_are_kept(text):
    assert _cleanup()(text) == text
