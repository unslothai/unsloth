# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_sentence_transformer_fsdp2_save.py. No unsloth import at module level: a spawned rank must set up the GPU-free harness first."""

from __future__ import annotations

import os
import runpy


def _files(path):
    out = {}
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                out[os.path.relpath(full, path)] = f.read()
    return out


def run_rank(rank, port, checkpoint, out_dir, queue):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_WORLD_SIZE = "2",
        ACCELERATE_USE_CPU = "true",
        UNSLOTH_FSDP2 = "1",
    )
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    )
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    import unsloth  # noqa: F401
    from unsloth import FastSentenceTransformer
    from unsloth.models.fsdp2_native import FSDP2Config, shard_model

    dist.init_process_group("gloo", rank = rank, world_size = 2)
    result = {}
    try:
        torch.manual_seed(0)
        transformer = Transformer(checkpoint)
        model = SentenceTransformer(
            modules = [transformer, Pooling(transformer.get_word_embedding_dimension())],
            device = "cpu",
        )
        # The fast encoder route from_pretrained takes for BERT-family models.
        model._unsloth_fast_encoder = True
        model._compile_mode = None
        model = FastSentenceTransformer.get_peft_model(
            model,
            r = 4,
            lora_alpha = 8,
            target_modules = ["query", "key", "value", "dense"],
            use_gradient_checkpointing = False,
        )
        # Non-zero, rank-independent adapters so the saved bytes are not just the init.
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(torch.randn(p.shape, generator = g) * 0.1)
        reference = os.path.join(out_dir, "one_process")
        if rank == 0:
            # Not sharded yet: the plain single-process save. ST's first save sets the tokenizer's
            # truncation (its model card tokenizes), so compare against a second one, as the FSDP2 one is.
            model.save_pretrained(os.path.join(out_dir, "warm"))
            model.save_pretrained(reference)
        dist.barrier()
        shard_model(model, FSDP2Config(device = "cpu", reshard_after_forward = True))
        n_sharded = sum(type(p).__name__ == "DTensor" for p in model.parameters())
        # A training-style forward / backward first (no optimizer step: the weights stay the reference's).
        ids = torch.randint(2, 64, (4, 8), generator = torch.Generator().manual_seed(rank))
        emb = model({"input_ids": ids, "attention_mask": torch.ones_like(ids)})[
            "sentence_embedding"
        ]
        emb.float().pow(2).mean().backward()
        # The model card's usage snippet (a forward) must be computed again, collectively, by this save.
        model.model_card_data.similarities = None
        error = None
        returned = "unset"
        try:
            returned = model.save_pretrained(os.path.join(out_dir, "fsdp2"))
        except Exception as e:
            error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        result = dict(
            error = error,
            returned_none = returned is None,
            n_sharded = n_sharded,
            still_sharded = sum(type(p).__name__ == "DTensor" for p in model.parameters()),
        )
        if error is None:
            # The native FSDP2 Trainer.save_model hook, with a trainer whose _save is
            # SentenceTransformerTrainer's: model.save_pretrained(output_dir).
            import types

            import transformers
            from unsloth.models.fsdp2_native import install_native_fsdp2_hooks

            install_native_fsdp2_hooks()
            trainer = types.SimpleNamespace(
                model = model,
                args = types.SimpleNamespace(should_save = rank == 0, push_to_hub = False),
                is_fsdp_enabled = False,
            )
            trainer._save = lambda output_dir = None, state_dict = None: trainer.model.save_pretrained(
                output_dir
            )
            model.model_card_data.similarities = None
            transformers.Trainer.save_model(trainer, os.path.join(out_dir, "trainer"))
            dist.barrier()
            if rank == 0:
                result["reference"] = _files(reference)
                result["fsdp2"] = _files(os.path.join(out_dir, "fsdp2"))
                result["trainer"] = _files(os.path.join(out_dir, "trainer"))
        queue.put((rank, result))
    except BaseException:
        import traceback
        queue.put((rank, {"error": traceback.format_exc()}))
        raise
    if result.get("error") is None:
        dist.destroy_process_group()
