# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`--shard-load` on a checkpoint whose lm_head is tied to the embedding.

A tied head is not saved under its own name: the checkpoint stores only the embedding key. The
loader filtered the shards by exact parameter name, so it looked for `lm_head.weight`, found
nothing, and the meta check refused the load with `unmaterialised tensors remain`. That is most
models, and `--shard-load` is the one path where refetching the weights is not an option.

These tests save a real checkpoint and read it back through the real loader rather than
stubbing safetensors, because the bug lives in the agreement between what is written and what
is looked for.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "spark_pipeline", REPO / "studio" / "spark_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saved(transformers, tmp_path, *, tied: bool) -> str:
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
        tie_word_embeddings = tied,
    )
    out = str(tmp_path / ("tied" if tied else "untied"))
    transformers.LlamaForCausalLM(config).save_pretrained(out, safe_serialization = True)
    return out


def _keys(directory: str) -> set:
    from safetensors import safe_open

    found = set()
    for shard in glob.glob(os.path.join(directory, "*.safetensors")):
        with safe_open(shard, framework = "pt") as handle:
            found |= set(handle.keys())
    return found


def test_a_tied_head_loads_from_the_embedding_it_is_tied_to(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pipeline = _pipeline_module()

    saved = _saved(transformers, tmp_path, tied = True)
    # The premise: the head really is absent from the checkpoint.
    assert "lm_head.weight" not in _keys(saved)
    assert "model.embed_tokens.weight" in _keys(saved)

    # The last stage of a two-way split: no embedding, keeps the head.
    model, _, _ = pipeline.build_stage_model(
        saved,
        rank = 1,
        world = 2,
        device = "cpu",
        shard_load = True,
        dtype = torch.float32,
        log = lambda *a, **k: None,
    )
    assert not model.lm_head.weight.is_meta

    from safetensors import safe_open

    for shard in glob.glob(os.path.join(saved, "*.safetensors")):
        with safe_open(shard, framework = "pt") as handle:
            if "model.embed_tokens.weight" in handle.keys():
                want = handle.get_tensor("model.embed_tokens.weight")
    assert torch.equal(model.lm_head.weight, want.to(torch.float32))


def test_a_stage_holding_both_keeps_them_one_tensor(tmp_path) -> None:
    """Two copies would let the optimizer update halves of a weight the model shares."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pipeline = _pipeline_module()

    saved = _saved(transformers, tmp_path, tied = True)
    model, _, _ = pipeline.build_stage_model(
        saved,
        rank = 0,
        world = 1,
        device = "cpu",
        shard_load = True,
        dtype = torch.float32,
        log = lambda *a, **k: None,
    )
    embed = model.get_input_embeddings().weight
    assert model.lm_head.weight.data_ptr() == embed.data_ptr()


def test_an_untied_checkpoint_is_unaffected(tmp_path) -> None:
    """The head is in the shards under its own name, and nothing here should change that."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pipeline = _pipeline_module()

    saved = _saved(transformers, tmp_path, tied = False)
    assert "lm_head.weight" in _keys(saved)

    model, _, _ = pipeline.build_stage_model(
        saved,
        rank = 1,
        world = 2,
        device = "cpu",
        shard_load = True,
        dtype = torch.float32,
        log = lambda *a, **k: None,
    )
    # The last stage has no embedding at all, so the head must have come from its own key.
    assert not model.lm_head.weight.is_meta
    assert isinstance(model.get_input_embeddings(), torch.nn.Identity)


# A tied checkpoint split across ranks under --full-finetune becomes two parameters on two
# optimizers fed by disjoint gradients. Measured through the real loader: the two ranks load
# equal values as separate objects, and three independent AdamW steps drive them 6.00e-02
# apart, on a model whose architecture requires them to be one tensor.
@pytest.mark.parametrize(
    "schedule,world,expected",
    [
        ("1f1b", 2, True),
        ("gpipe", 2, True),
        ("interleaved", 2, True),
        ("zbv", 2, False),  # a V layout puts the first and last stage on one rank
        ("dualpipev", 2, False),
        ("1f1b", 1, False),  # one rank is one parameter
    ],
)
def test_a_tied_checkpoint_is_refused_only_where_it_would_actually_split(
    schedule: str, world: int, expected: bool
) -> None:
    pipeline = _pipeline_module()
    plan = pipeline.torch_pp_plan(schedule, world, microbatches = 4, virtual_stages = 2, n_layers = 8)
    problem = pipeline.tied_split_problem(
        tied = True, full_finetune = True, world = world, stage_to_rank = plan["stage_to_rank"]
    )
    assert bool(problem) is expected, f"{schedule} world={world}: {problem}"
    if problem:
        assert "tie" in problem and "zbv" in problem


def test_lora_and_untied_checkpoints_are_not_refused() -> None:
    """LoRA freezes the base weights, so tied weights cannot drift; an untied checkpoint has
    nothing to keep equal in the first place."""
    pipeline = _pipeline_module()
    mapping = pipeline.stage_to_rank_map(2, 2, "loop")
    assert pipeline.tied_split_problem(True, False, 2, mapping) is None
    assert pipeline.tied_split_problem(False, True, 2, mapping) is None


def test_the_legacy_backend_is_covered_without_a_plan() -> None:
    """`--pp-backend legacy` builds no stage_to_rank map, and embedding and head still land on
    the first and last rank."""
    pipeline = _pipeline_module()
    assert pipeline.tied_split_problem(True, True, 2, None)
    assert pipeline.tied_split_problem(True, True, 1, None) is None
