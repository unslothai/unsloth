# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Patched Online DPO _forward on a tiny model that drops its mask in training like Unsloth's; CPU only."""

import ast
import importlib.util
import re
import textwrap
import types
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
NAMES = ("_ONLINE_DPO_MODEL_CALL", "_ONLINE_DPO_LOGITS_SLICE", "online_dpo_trainer__forward")

# TRL 0.22.2 spells the call with attention_mask= and has no vision_inputs.
TRL_0_22_FORWARD = """
    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        logits = output.logits[:, start_idx:-1]
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs
"""


def _patcher():
    tree = ast.parse(SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in NAMES)
        or (isinstance(node, ast.Assign) and node.targets[0].id in NAMES)
    ]
    assert len(wanted) == len(NAMES)
    namespace = {"re": re, "_warn_once": lambda *a: pytest.fail(f"anchor missed: {a}")}
    exec(compile(ast.Module(wanted, []), str(SOURCE), "exec"), namespace)
    return namespace["online_dpo_trainer__forward"]


def _compile(source):
    namespace = {"torch": torch}
    exec(textwrap.dedent(source), namespace)
    return namespace["_forward"]


def _trl_sources():
    # Read TRL's file, not the live class: an imported unsloth has already swapped in its patched trainer.
    sources = {"trl-0.22.2": TRL_0_22_FORWARD}
    spec = importlib.util.find_spec("trl")
    if spec is None or not spec.submodule_search_locations:
        return sources
    root = Path(spec.submodule_search_locations[0])
    for relative in (
        "trainer/online_dpo_trainer.py",
        "experimental/online_dpo/online_dpo_trainer.py",
    ):
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name == "_forward":
                sources["installed"] = "    " + ast.get_source_segment(text, node)
                return sources
    return sources


class _DropsMaskInTraining(torch.nn.Module):
    """Stands in for Unsloth's forward: training ignores the 2D mask (flash / causal only)."""

    def __init__(self):
        super().__init__()
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size = 97,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            attn_implementation = "eager",
        )
        self.inner = LlamaForCausalLM(config).float()
        self.seen = []

    def forward(
        self,
        input_ids,
        attention_mask = None,
        **kwargs,
    ):
        self.seen.append(attention_mask.clone())
        return self.inner(input_ids, attention_mask = None if self.training else attention_mask)


def _batch():
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(3, 97, (n,), generator = g) for n in (6, 2, 4)]
    completions = [torch.randint(3, 97, (n,), generator = g) for n in (3, 5, 1)]
    P, C = max(map(len, prompts)), max(map(len, completions))
    prompt_ids = torch.zeros(3, P, dtype = torch.long)
    prompt_mask = torch.zeros(3, P, dtype = torch.long)
    completion_ids = torch.zeros(3, C, dtype = torch.long)
    completion_mask = torch.zeros(3, C, dtype = torch.long)
    for r, (p, c) in enumerate(zip(prompts, completions)):
        prompt_ids[r, P - len(p) :], prompt_mask[r, P - len(p) :] = p, 1
        completion_ids[r, : len(c)], completion_mask[r, : len(c)] = c, 1
    return prompts, completions, prompt_ids, prompt_mask, completion_ids, completion_mask


def _reference(model, prompts, completions):
    rows = []
    for p, c in zip(prompts, completions):
        logits = model.inner(torch.cat((p, c))[None]).logits[0, len(p) - 1 : -1]
        rows.append(logits.log_softmax(-1).gather(-1, c[:, None]).squeeze(-1))
    return rows


@pytest.mark.parametrize("version", sorted(_trl_sources()))
def test_patched_forward_matches_unpadded_rows(version):
    source = _trl_sources()[version]
    patched_source = _patcher()("_forward", source)
    assert "_unsloth_left_pad" in patched_source
    assert _patcher()("_forward", patched_source) == patched_source
    original, patched = _compile(source), _compile(patched_source)
    trainer = types.SimpleNamespace(max_length = 64)
    model = _DropsMaskInTraining().train()
    prompts, completions, *tensors = _batch()
    kwargs = {"vision_inputs": None} if "vision_inputs" in source else {}

    with torch.no_grad():
        reference = _reference(model, prompts, completions)
        got = patched(trainer, model, *tensors, **kwargs)
        stock = original(trainer, model, *tensors, **kwargs)

    for r, ref in enumerate(reference):
        torch.testing.assert_close(got[r, : len(ref)], ref, atol = 1e-5, rtol = 1e-5)
    # Patched call first: no pad before a real token. Stock TRL second: left-padded.
    assert not bool((model.seen[0][:, 1:] > model.seen[0][:, :-1]).any())
    assert bool((model.seen[1][:, 1:] > model.seen[1][:, :-1]).any())
    # Stock TRL on the same model is wrong, so the check can fail.
    assert not torch.allclose(stock[1, : len(reference[1])], reference[1], atol = 1e-3)


def test_patched_forward_unchanged_when_mask_is_honoured():
    source = _trl_sources()["trl-0.22.2"]
    original, patched = _compile(source), _compile(_patcher()("_forward", source))
    trainer = types.SimpleNamespace(max_length = 64)
    model = _DropsMaskInTraining().eval()
    _, completions, *tensors = _batch()
    with torch.no_grad():
        want, got = original(trainer, model, *tensors), patched(trainer, model, *tensors)
    completion_mask = tensors[-1].bool()
    torch.testing.assert_close(got[completion_mask], want[completion_mask], atol = 1e-5, rtol = 1e-5)


def test_vision_rows_keep_trl_layout():
    source = _trl_sources().get("installed")
    if source is None or "vision_inputs" not in source:
        pytest.skip("installed TRL has no vision-aware Online DPO _forward")
    assert "if not vision_inputs:" in _patcher()("_forward", source)
