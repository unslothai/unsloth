# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""A FastSentenceTransformer LoRA model sharded by native FSDP2 (UNSLOTH_FSDP2=1) must save like one
process: every rank calls save_pretrained, rank 0 writes the gathered weights, and the folder equals,
byte for byte, a single-process save of the same weights."""

from __future__ import annotations

import os
import queue as queue_module
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _st_fsdp2_save_worker as worker  # noqa: E402

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")


def _failed(results):
    """A rank that raised (in backward, in the save) can leave its peer blocked in a collective:
    once one has, wait only briefly for the other."""
    return any(
        out.get("error") or any(isinstance(c, dict) and c.get("error") for c in out.values())
        for out in results.values()
    )


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tiny_bert(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    BertModel(
        BertConfig(
            vocab_size = 64,
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            max_position_embeddings = 40,
            hidden_dropout_prob = 0.0,
            attention_probs_dropout_prob = 0.0,
        ),
        add_pooling_layer = False,
    ).save_pretrained(path)
    vocab = {f"w{i}": i for i in range(2, 64)}
    vocab.update({"[PAD]": 0, "[UNK]": 1})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer, pad_token = "[PAD]", unk_token = "[UNK]", model_max_length = 32
    ).save_pretrained(path)
    return str(path)


def _run_two_ranks(target, tmp_path):
    import torch.multiprocessing as mp

    checkpoint = _tiny_bert(tmp_path / "bert")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = target, args = (r, port, checkpoint, str(tmp_path), queue))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        while len(results) < len(procs):
            try:
                rank, out = queue.get(timeout = 60 if _failed(results) else 600)
            except queue_module.Empty:
                break
            results[rank] = out
    finally:
        for p in procs:
            p.join(5 if len(results) < len(procs) else 120)
            if p.is_alive():
                p.kill()
    assert results, "no rank reported"
    for rank, out in sorted(results.items()):
        assert out["error"] is None, (rank, out["error"])
    assert sorted(results) == [0, 1]
    return results


def test_fsdp2_sentence_transformer_save_equals_one_process(tmp_path):
    results = _run_two_ranks(worker.run_rank, tmp_path)
    for rank in (0, 1):
        assert results[rank]["n_sharded"] > 0
        # The gather put the shards back afterwards.
        assert results[rank]["still_sharded"] == results[rank]["n_sharded"]
    assert results[1]["returned_none"]
    reference, fsdp2 = results[0]["reference"], results[0]["fsdp2"]
    assert any(name.endswith("adapter_model.safetensors") for name in reference)
    assert sorted(fsdp2) == sorted(reference)
    for name in reference:
        assert fsdp2[name] == reference[name], name
    # Trainer.save_model (checkpoints, SentenceTransformerTrainer._save -> save_pretrained) writes the same.
    assert results[0]["trainer"] == fsdp2


def test_fsdp2_sentence_transformer_save_honours_card_option_and_training_mode(tmp_path):
    results = _run_two_ranks(worker.run_rank_save_options, tmp_path)
    for rank in (0, 1):
        out = results[rank]
        assert out["n_sharded"] > 0
        assert out["still_sharded"] == out["n_sharded"]
        assert out["no_card_runs"] == 0, (rank, out)
        assert out["no_card_eval_modules"] == 0, (rank, out)
        assert out["card_runs"] == 1, (rank, out)
        assert out["card_eval_modules"] == 0, (rank, out)
    assert results[0]["no_card_has_st_card"] is False
