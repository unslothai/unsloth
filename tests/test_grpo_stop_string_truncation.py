# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A stop-string completion below max_completion_length has no EOS but is not truncated (#15382)."""

import ast
import importlib.util
import re

import pytest

EOS, PAD = 2, 0


def _trl_method_source(name):
    spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    with open(spec.origin, "r", encoding = "utf-8") as fh:
        source = fh.read()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    pytest.skip(f"this trl has no GRPOTrainer.{name}")


def _is_truncated(name, transform, completions, max_completion_length):
    source = transform(name, _trl_method_source(name))
    if "eos_and_pad" not in source:
        pytest.skip("this trl does not detect truncation from the last token")
    match = re.search(r"is_truncated = torch\.tensor\((\[[^\n]*for ids in (\w+)\])", source)
    assert match is not None
    trainer = type("Trainer", (), {"max_completion_length": max_completion_length})()
    namespace = {"self": trainer, "eos_and_pad": [EOS, PAD], match.group(2): completions}
    return eval(match.group(1), namespace)


@pytest.mark.parametrize("name", ["_generate", "_generate_and_score_completions"])
def test_only_length_exhausted_completions_without_eos_are_truncated(name):
    from unsloth.models import rl_replacements

    transform = {
        "_generate": rl_replacements.grpo_trainer__generate,
        "_generate_and_score_completions": rl_replacements.grpo_trainer__generate_and_score_completions,
    }[name]
    completions = [
        [5] * 85,  # stopped on a stop string well below the limit
        [5] * 128,
        [5] * 127 + [EOS],
        [5] * 127 + [PAD],
        [5] * 40 + [EOS],
    ]
    assert _is_truncated(name, transform, completions, 128) == [False, True, False, False, False]
    assert _is_truncated(name, transform, completions, None) == [True, True, False, False, False]
