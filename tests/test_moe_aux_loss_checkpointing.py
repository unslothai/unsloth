# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE aux loss keeps its gradient under "unsloth" GC: reentrant GC's no_grad pass dropped it, so non-reentrant is used."""

import pytest

from real_accelerator import (
    has_real_accelerator,
)  # tests/_shared, on sys.path via tests/conftest.py

torch = pytest.importorskip("torch")
# The CPU shards run gpu-marked tests too; the recorded real-accelerator answer is the gate the CUDA spoof cannot flip.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not has_real_accelerator(), reason = "loads a model with offloaded gradient checkpointing"
    ),
]

MODELS = [
    "trl-internal-testing/tiny-GptOssForCausalLM",
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
]


def _grads(repo, gc):
    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(
        repo, max_seq_length = 512, dtype = torch.bfloat16, load_in_4bit = False
    )
    text = model.config.get_text_config()
    text.router_aux_loss_coef = 0.9
    model = FastModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        random_state = 0,
        use_gradient_checkpointing = gc,
    )
    for module in model.modules():
        if hasattr(module, "router_aux_loss_coef"):
            module.router_aux_loss_coef = 0.9
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_((torch.randn(p.shape, generator = generator) * 0.02).to(p.dtype))
    calls = [0]
    for module in model.modules():
        func = module.__dict__.get("_gradient_checkpointing_func")
        if func is not None and getattr(module, "gradient_checkpointing", False):

            def counted(
                *args,
                _func = func,
                **kwargs,
            ):
                calls[0] += 1
                return _func(*args, **kwargs)

            module._gradient_checkpointing_func = counted
    model.train()
    ids = torch.randint(10, 1000, (2, 48), generator = torch.Generator().manual_seed(2)).cuda()
    out = model(input_ids = ids, labels = ids, output_router_logits = True)
    assert out.aux_loss is not None and float(out.aux_loss) > 0
    out.loss.backward()
    norm = torch.sqrt(
        sum(
            p.grad.float().pow(2).sum()
            for n, p in model.named_parameters()
            if "lora_" in n and p.grad is not None
        )
    )
    return float(norm), calls[0]


@pytest.mark.parametrize("repo", MODELS)
def test_router_aux_loss_gradient_survives_unsloth_checkpointing(repo):
    plain, plain_calls = _grads(repo, False)
    checkpointed, calls = _grads(repo, "unsloth")
    assert plain_calls == 0 and calls > 0, (plain_calls, calls)
    assert abs(checkpointed - plain) / plain < 2e-2, (checkpointed, plain)
