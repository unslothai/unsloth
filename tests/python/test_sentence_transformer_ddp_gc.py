# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""FastSentenceTransformer LoRA with gradient checkpointing must train under DDP (#4680); one GPU keeps reentrant checkpointing."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _st_ddp_gc_worker as worker  # noqa: E402

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")

_DIST_ENV = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _tiny_checkpoint(path, arch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, MPNetConfig, MPNetModel, PreTrainedTokenizerFast

    common = dict(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        max_position_embeddings = 40,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )
    torch.manual_seed(0)
    if arch == "mpnet":
        MPNetModel(MPNetConfig(**common), add_pooling_layer = False).save_pretrained(path)
    else:
        BertModel(BertConfig(**common), add_pooling_layer = False).save_pretrained(path)
    vocab = {f"w{i}": i for i in range(2, 64)}
    vocab.update({"[PAD]": 0, "[UNK]": 1})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer, pad_token = "[PAD]", unk_token = "[UNK]", model_max_length = 32
    ).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("arch", ["mpnet", "bert"])
def test_lora_with_gradient_checkpointing_trains_under_ddp(tmp_path, arch):
    import torch.multiprocessing as mp

    checkpoint = _tiny_checkpoint(tmp_path / arch, arch)
    # spawn: fork is unsafe once autograd has run in this process.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = worker.run_rank, args = (rank, port, checkpoint, queue))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    try:
        # A rank that raised can leave the other blocked in an all-reduce, so bound the wait.
        results = [queue.get(timeout = 900) for _ in procs]
    finally:
        for p in procs:
            p.join(120)
            if p.is_alive():
                p.kill()
    for p in procs:
        assert p.exitcode == 0
    for rank, error, trained in results:
        assert error is None, f"rank {rank}: {error}"
        assert trained > 0


@pytest.mark.parametrize("world_size, reentrant", [(None, True), ("2", False)])
def test_mpnet_checkpointing_is_reentrant_only_on_one_gpu(monkeypatch, world_size, reentrant):
    import unsloth  # noqa: F401
    import transformers
    from packaging.version import Version
    from transformers.models.mpnet import modeling_mpnet
    from unsloth.models.sentence_transformer import FastSentenceTransformer

    for name in _DIST_ENV:
        monkeypatch.delenv(name, raising = False)
    if world_size is not None:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    for cls, attr in (
        (modeling_mpnet.MPNetModel, "supports_gradient_checkpointing"),
        (modeling_mpnet.MPNetModel, "_set_gradient_checkpointing"),
        (modeling_mpnet.MPNetEncoder, "forward"),
    ):
        if attr in vars(cls):
            monkeypatch.setattr(cls, attr, vars(cls)[attr])
    if Version(transformers.__version__).major < 5:
        FastSentenceTransformer._patch_mpnet_v4()
    else:
        FastSentenceTransformer._patch_mpnet_v5()
    seen = []
    original = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        seen.append(kwargs.get("use_reentrant"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    config = modeling_mpnet.MPNetConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        max_position_embeddings = 40,
    )
    model = modeling_mpnet.MPNetModel(config, add_pooling_layer = False)
    model.encoder.gradient_checkpointing = True
    model.train()
    ids = torch.randint(2, 64, (2, 6))
    model.get_input_embeddings().weight.requires_grad_(True)
    model(input_ids = ids, attention_mask = torch.ones_like(ids)).last_hidden_state.sum().backward()
    assert seen == [reentrant, reentrant]
