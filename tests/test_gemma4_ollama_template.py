# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Gemma-4 Ollama template (E2B / E4B and every Gemma-4 template without the empty thought primer).

History assistant turns must render as <|turn>model, with no blank line between turns and no
leading newline, so Ollama matches the HF chat template with add_generation_prompt = True.
The module is loaded via ast so the test runs without a GPU or Go.
"""

import ast
import os

MAPPERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "ollama_template_mappers.py",
)
_TREE = ast.parse(open(MAPPERS_PATH, encoding = "utf-8").read())


def _load():
    namespace = {"OLLAMA_TEMPLATES": {}}
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "gemma4_ollama"
            or (
                isinstance(target, ast.Subscript)
                and getattr(target.value, "id", None) == "OLLAMA_TEMPLATES"
                and getattr(target.slice, "value", None) in ("gemma-4", "gemma4")
            )
            for target in node.targets
        ):
            exec(compile(ast.Module([node], []), MAPPERS_PATH, "exec"), namespace)
    return namespace


NS = _load()

EXPECTED_TEMPLATE = (
    "{{- range $i, $_ := .Messages }}\n"
    "{{- $last := eq (len (slice $.Messages $i)) 1 }}\n"
    '{{- if eq .Role "assistant" }}<|turn>model\n'
    "{{ .Content }}{{ if not $last }}<turn|>\n"
    "{{ end }}\n"
    "{{- else }}<|turn>{{ .Role }}\n"
    "{{ .Content }}<turn|>\n"
    "{{ if $last }}<|turn>model\n"
    "{{ end }}\n"
    "{{- end }}\n"
    "{{- end }}"
)


def _template_body(modelfile):
    start = modelfile.index('TEMPLATE """') + len('TEMPLATE """')
    return modelfile[start : modelfile.index('"""', start)]


def test_registered_under_both_names():
    assert NS["OLLAMA_TEMPLATES"]["gemma-4"] is NS["gemma4_ollama"]
    assert NS["OLLAMA_TEMPLATES"]["gemma4"] is NS["gemma4_ollama"]


def test_template_body_is_exact():
    assert _template_body(NS["gemma4_ollama"]) == EXPECTED_TEMPLATE


def test_history_assistant_turns_use_the_model_role():
    ollama = NS["gemma4_ollama"]
    assert '{{- if eq .Role "assistant" }}<|turn>model\n{{ .Content }}' in ollama
    assert "<|turn>assistant" not in ollama


def test_no_blank_line_or_leading_newline_between_turns():
    body = _template_body(NS["gemma4_ollama"])
    # Every literal newline is either trimmed by a {{- action or is part of a turn marker line.
    assert body.startswith("{{-")
    assert "<turn|>\n\n" not in body
    assert "\n<|turn>" not in body
    # The generation prompt carries no thought channel for this template.
    assert "<|channel>" not in body
