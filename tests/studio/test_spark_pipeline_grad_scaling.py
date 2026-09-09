# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pipeline gradients must not depend on which torch is installed.

`scale_grads` was added to `torch.distributed.pipelining` in 2.7, together with
`PipelineStage.scale_grads`; 2.6 has neither and performs no scaling of its own. Feature
detecting the keyword and moving on therefore did not fall back to equivalent behaviour, it
dropped the scaling silently while the log still reported it as on. With a mean-reduced loss
over M microbatches every gradient came out M times too large, on the floor of the supported
torch range. Measured at torch 2.6.0, M=4: gradient norms 4.0000x the reference before, 1.0000x
after; at torch 2.11.0 both are 1.0000x.

The reference is a plain whole-batch backward of the same loss, which is what the pipeline is
supposed to reproduce, so this test states the requirement rather than the mechanism and holds
on every version.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import socket
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MICROBATCHES = 4


def _pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "spark_pipeline", REPO / "studio" / "spark_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pipeline_gradients_match_a_plain_backward_of_the_same_loss() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch.distributed.pipelining")
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    pipeline = _pipeline_module()
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    base = transformers.LlamaForCausalLM(config)
    base.config._attn_implementation = "eager"
    ids = torch.randint(0, 64, (8, 6))
    labels = ids.clone()

    # The requirement: one backward of the mean loss over the whole batch.
    reference = copy.deepcopy(base)
    reference.train()
    pipeline.pp_loss_fn(reference(ids).logits, labels).backward()
    want = {name: p.grad.clone() for name, p in reference.named_parameters() if p.grad is not None}

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(_free_port()),
        RANK = "0",
        WORLD_SIZE = "1",
        LOCAL_RANK = "0",
    )
    dist.init_process_group("gloo", rank = 0, world_size = 1)
    try:
        plan = pipeline.torch_pp_plan("gpipe", 1, MICROBATCHES, 2, config.num_hidden_layers)
        model = copy.deepcopy(base)
        model.train()
        schedule, mods, step_kw, scale_grads = pipeline.build_torch_schedule(
            model,
            plan,
            pipeline.plan_for_rank(plan, 0),
            microbatches = MICROBATCHES,
            device = "cpu",
            grad_checkpoint = False,
            log = lambda *a, **k: None,
        )
        for mod in mods:
            for p in mod.parameters():
                p.grad = None
        schedule.step(ids, target = labels, losses = [], **step_kw)
        scale_grads()

        checked = 0
        for mod in mods:
            for name, p in mod.named_parameters():
                if p.grad is None:
                    continue
                match = [w for key, w in want.items() if key.endswith(name)]
                if len(match) != 1:
                    continue
                ratio = (p.grad.norm() / match[0].norm()).item()
                assert ratio == pytest.approx(1.0, abs = 1e-3), (
                    f"{name}: pipeline gradient is {ratio:.4f}x the reference; "
                    f"a factor of {MICROBATCHES} here means the microbatch scaling was skipped"
                )
                checked += 1
        assert checked, "no gradients were compared, so this proved nothing"
    finally:
        dist.destroy_process_group()
