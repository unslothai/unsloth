# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A stop-string completion below max_completion_length has no EOS but is not truncated (#15382),
while one cut by the vLLM context (prompt + completion reaching max_model_len) still is."""

import ast
import importlib.util
import re
import types

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


def _transform(name):
    from unsloth.models import rl_replacements
    return {
        "_generate": rl_replacements.grpo_trainer__generate,
        "_generate_and_score_completions": rl_replacements.grpo_trainer__generate_and_score_completions,
    }[name]


def _trainer(
    max_completion_length,
    max_model_len = None,
    via_vllm_generation = False,
):
    trainer = types.SimpleNamespace(max_completion_length = max_completion_length, use_vllm = False)
    if max_model_len is not None:
        model_config = types.SimpleNamespace(max_model_len = max_model_len)
        llm = types.SimpleNamespace(llm_engine = types.SimpleNamespace(model_config = model_config))
        trainer.use_vllm = True
        if via_vllm_generation:  # TRL 0.28+ keeps the engine on VLLMGeneration
            trainer.vllm_generation = types.SimpleNamespace(llm = llm)
        else:
            trainer.llm = llm
    return trainer


def _is_truncated(
    name,
    completions,
    trainer,
    prompts = None,
):
    from unsloth.models import rl_replacements

    source = _transform(name)(name, _trl_method_source(name))
    if "eos_and_pad" not in source:
        pytest.skip("this trl does not detect truncation from the last token")
    match = re.search(r"is_truncated = torch\.tensor\(([^\n]*), device=device\)", source)
    assert match is not None
    if prompts is None:
        prompts = [[7] * 4 for _ in completions]
    namespace = {
        "self": trainer,
        "eos_and_pad": [EOS, PAD],
        "completion_ids": completions,
        "completion_ids_list": completions,
        "prompt_ids": prompts,
        "prompt_ids_list": prompts,
    }
    helper = getattr(rl_replacements, "_unsloth_grpo_is_truncated", None)
    if helper is not None:
        namespace["_unsloth_grpo_is_truncated"] = helper
    return list(eval(match.group(1), namespace))


NAMES = ["_generate", "_generate_and_score_completions"]


@pytest.mark.parametrize("name", NAMES)
def test_only_length_exhausted_completions_without_eos_are_truncated(name):
    completions = [
        [5] * 85,  # stopped on a stop string well below the limit
        [5] * 128,
        [5] * 127 + [EOS],
        [5] * 127 + [PAD],
        [5] * 40 + [EOS],
    ]
    assert _is_truncated(name, completions, _trainer(128)) == [False, True, False, False, False]
    assert _is_truncated(name, completions, _trainer(None)) == [True, True, False, False, False]


@pytest.mark.parametrize("via_vllm_generation", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_completion_cut_by_vllm_context_is_truncated(name, via_vllm_generation):
    # max_model_len 512 (= max_seq_length of the fast_inference engine), max_completion_length 128.
    completions = [
        [5] * 32,  # 480-token prompt: vLLM stops at 512 total tokens, FINISHED_LENGTH_CAPPED
        [5] * 10,  # 480-token prompt, stop string after 10 tokens: 490 < 512, not truncated
        [5] * 10,  # short prompt, stop string
        [5] * 31 + [EOS],  # EOS exactly at the context limit is a normal stop
        [5] * 128,  # completion budget exhausted
    ]
    prompts = [[7] * 480, [7] * 480, [7] * 20, [7] * 480, [7] * 20]
    trainer = _trainer(128, max_model_len = 512, via_vllm_generation = via_vllm_generation)
    assert _is_truncated(name, completions, trainer, prompts) == [True, False, False, False, True]
    # Without a local engine (HF generate, vLLM server mode) only the completion budget is known.
    assert _is_truncated(name, completions, _trainer(128), prompts) == [
        False,
        False,
        False,
        False,
        True,
    ]
