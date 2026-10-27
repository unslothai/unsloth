# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_block_compile_ddp_no_split.py. Loads _block_compile.py by file path: importing
the unsloth package needs an accelerator."""

from __future__ import annotations

import importlib.util
import os
import sys

_BLOCK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "unsloth",
    "models",
    "_block_compile.py",
)


def _load_block_compile():
    spec = importlib.util.spec_from_file_location("_ddp_block_compile", _BLOCK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_rank(
    rank,
    port,
    queue,
    recompute_miss = False,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        CUDA_VISIBLE_DEVICES = "",
    )
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        os.environ.pop(key, None)
    try:
        import torch
        import torch.distributed as dist

        torch.set_num_threads(1)
        import transformers
        from torch._dynamo.backends import distributed as dynamo_distributed

        dist.init_process_group("gloo", rank = rank, world_size = 2)
        block = _load_block_compile()
        splits = []
        original_compile_fn = dynamo_distributed.DDPOptimizer.compile_fn

        def counting_compile_fn(self, gm, example_inputs, **kwargs):
            # A decoder block's graph reads the layer's own weights (L_self_modules_...).
            if any(n.op == "placeholder" and n.name.startswith("l_self_") for n in gm.graph.nodes):
                splits.append(len(gm.graph.nodes))
            return original_compile_fn(self, gm, example_inputs, **kwargs)

        dynamo_distributed.DDPOptimizer.compile_fn = counting_compile_fn

        torch.manual_seed(0)
        # Layers above DDP's 1 MB first bucket, so DDPOptimizer would split each block graph.
        config = transformers.LlamaConfig(
            vocab_size = 128,
            hidden_size = 256,
            intermediate_size = 1024,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
        )
        model = transformers.LlamaForCausalLM(config)
        # What Unsloth arms for VLMs under DDP (vision.py): non-reentrant checkpointing.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        model.train()
        phase = [0]
        if recompute_miss:
            # A guard that differs between the forward and the checkpoint recompute (same branch, so
            # numerics are unchanged): the recompute misses the forward's cache entry and compiles
            # the block again, outside DDP.forward. This is the DeepSeek-OCR condition.
            def hook(module, args):
                if phase[0] >= 0:
                    return None
                return (args[0] * 2,)

            for layer in model.model.layers:
                layer.mlp.register_forward_pre_hook(hook)
        is_available = torch.cuda.is_available
        torch.cuda.is_available = (
            lambda: True
        )  # patch_block_compile only installs on an accelerator
        try:
            wrapped = block.patch_block_compile(model)
        finally:
            torch.cuda.is_available = is_available
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters = True, bucket_cap_mb = 1
        )
        losses = []
        for step in range(2):
            ids = torch.randint(
                1, 128, (2, 16), generator = torch.Generator().manual_seed(10 * rank + step)
            )
            phase[0] = 0
            out = ddp(input_ids = ids, labels = ids)
            phase[0] = 1
            out.loss.backward()
            losses.append(float(out.loss))
        grad_sum = float(sum(p.grad.double().abs().sum() for p in model.parameters()))
        compiled = [
            bool(layer.__dict__.get("_unsloth_block_checked", 0)) for layer in model.model.layers
        ]
        queue.put(
            (
                rank,
                dict(
                    wrapped = wrapped,
                    splits = splits,
                    losses = losses,
                    grad_sum = grad_sum,
                    compiled = compiled,
                    optimize_ddp = bool(torch._dynamo.config.optimize_ddp),
                ),
            )
        )
        dist.destroy_process_group()
    except BaseException:
        import traceback
        queue.put((rank, {"error": traceback.format_exc()}))
        raise
