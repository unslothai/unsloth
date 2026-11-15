# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A causal decoder embedder on FlashAttention 2 trains on one packed row of real tokens: same
embeddings and gradients as the padded batch, left or right padded."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")


@pytest.mark.parametrize("side", ["left", "right"])
def test_qwen3_embedder_packs_like_padded(monkeypatch, side):
    pytest.importorskip("flash_attn")
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 64
    )
    model = FastSentenceTransformer.get_peft_model(
        model,
        r = 8,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing = False,
        random_state = 3407,
    )
    transformer = model[0]
    assert transformer._unsloth_packing_installed
    transformer.tokenizer.padding_side = side
    texts = [
        "hi",
        "a much longer sentence with quite a few more tokens in it",
        "middle length text",
    ]
    features = {k: v.cuda() if torch.is_tensor(v) else v for k, v in model.tokenize(texts).items()}
    assert not bool(features["attention_mask"].all())
    model.train()

    def embed():
        model.zero_grad(set_to_none = True)
        with torch.autocast("cuda", dtype = torch.bfloat16):
            out = model(dict(features))["sentence_embedding"].float()
        out.square().sum().backward()
        grads = [p.grad.float().clone() for p in model.parameters() if p.grad is not None]
        return out.detach(), grads

    packed, packed_grads = embed()
    assert transformer._unsloth_packed_calls == 1
    monkeypatch.setenv("UNSLOTH_ST_PACKING", "0")
    padded, padded_grads = embed()
    assert transformer._unsloth_packed_calls == 1
    cosine = torch.nn.functional.cosine_similarity(packed, padded)
    assert cosine.min() > 0.999, cosine
    assert len(packed_grads) == len(padded_grads) > 0
    for a, b in zip(packed_grads, padded_grads):
        assert torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim = 0) > 0.99
