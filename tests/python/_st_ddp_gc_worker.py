# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_sentence_transformer_ddp_gc.py. No unsloth import at module level: a spawned rank must set up the GPU-free harness first."""

from __future__ import annotations

import os
import runpy


def run_rank(rank, port, checkpoint, queue):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = "2"
    )
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    )
    import torch
    import torch.distributed as dist
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    import unsloth  # noqa: F401
    from unsloth import FastSentenceTransformer

    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        torch.manual_seed(0)
        transformer = Transformer(checkpoint)
        model = SentenceTransformer(
            modules = [transformer, Pooling(transformer.get_word_embedding_dimension())],
            device = "cpu",
        )
        # The fast encoder route from_pretrained takes for BERT-family models with gradient checkpointing on.
        model._unsloth_fast_encoder = True
        model._compile_mode = None
        model = FastSentenceTransformer.get_peft_model(
            model,
            r = 4,
            lora_alpha = 8,
            target_modules = ["q", "k", "v", "o", "query", "key", "value", "dense"],
            use_gradient_checkpointing = True,
        )
        model.train()
        ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters = True)
        g = torch.Generator().manual_seed(10 + rank)
        error = None
        try:
            for _ in range(2):
                ddp.zero_grad(set_to_none = True)
                # MultipleNegativesRankingLoss: the query and the passage go through the model separately.
                emb = []
                for _ in range(2):
                    ids = torch.randint(2, 64, (4, 8), generator = g)
                    emb.append(
                        ddp({"input_ids": ids, "attention_mask": torch.ones_like(ids)})[
                            "sentence_embedding"
                        ]
                    )
                scores = emb[0] @ emb[1].T
                torch.nn.functional.cross_entropy(scores, torch.arange(4)).backward()
        except RuntimeError as e:
            error = str(e).splitlines()[0]
        trained = sum(
            p.grad is not None
            for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad
        )
        queue.put((rank, error, trained))
    finally:
        dist.destroy_process_group()
