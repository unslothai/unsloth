# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Backward through a forward run in eval mode or after for_inference (#895)."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")


@pytest.fixture(scope = "module")
def lora_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        random_state = 3407,
    )
    torch.manual_seed(0)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.02)
    ids = torch.randint(0, model.config.vocab_size, (1, 12), device = model.device)
    return FastLanguageModel, model, ids


def _loss_and_grads(model, ids, **kwargs):
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids, **kwargs).loss
    loss.backward()
    grads = [p.grad.float().clone() for p in model.parameters() if p.grad is not None]
    return loss.item(), grads


@pytest.mark.parametrize("mode", ["eval_use_cache", "for_inference"])
def test_backward_matches_training_mode(lora_model, mode):
    fast, model, ids = lora_model
    fast.for_training(model)
    ref_loss, ref_grads = _loss_and_grads(model, ids)
    try:
        if mode == "eval_use_cache":
            model.eval()
            loss, grads = _loss_and_grads(model, ids, use_cache = True)
        else:
            fast.for_inference(model)
            loss, grads = _loss_and_grads(model, ids)
    finally:
        fast.for_training(model)
    assert loss == pytest.approx(ref_loss)
    assert len(grads) == len(ref_grads) > 0
    for got, ref in zip(grads, ref_grads):
        torch.testing.assert_close(got, ref)


def test_inference_mode_forward_unchanged(lora_model):
    fast, model, ids = lora_model
    fast.for_training(model)
    with torch.no_grad():
        ref = model(input_ids = ids).logits
    fast.for_inference(model)
    try:
        with torch.inference_mode():
            got = model(input_ids = ids, use_cache = True).logits
    finally:
        fast.for_training(model)
    torch.testing.assert_close(got.float(), ref.float())
