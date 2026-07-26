# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fast, GPU-gated real-inference smoke.

GitHub-hosted CI runners have no GPU, so this AUTO-SKIPS there; the full picker
-> load -> chat flow is covered on CPU by tests/studio/playwright_model_config.py
and studio-ui-smoke.yml. This test adds a quick real-generation check for local
dev and self-hosted GPU runners: it loads the smallest model (gemma-3-270m-it)
on the GPU and does a single short greedy generation, asserting a non-empty
reply. Kept deliberately short (a handful of new tokens) so it is a confidence
check, not a benchmark. Select/deselect it by name, e.g. `-k gpu_generation`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Smallest instruct model in the CI fixture family; ~270M params loads and
# generates a few tokens in seconds on any GPU.
MODEL_ID = "unsloth/gemma-3-270m-it"
# A handful of forced real tokens: enough to prove GPU decode produced content,
# short enough to stay a few seconds.
MIN_NEW_TOKENS = 4
MAX_NEW_TOKENS = 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a CUDA GPU")
def test_gpu_generation_smoke():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - env without transformers
        pytest.skip(f"transformers unavailable: {exc}")

    # Gemma is numerically unstable in fp16 (it emits only <pad>); use bf16 where
    # supported, else fp32. The model is tiny, so fp32 is still fast.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype = dtype).to("cuda")
    except Exception as exc:  # offline / gated / download failure is not a code defect
        pytest.skip(f"could not fetch/load {MODEL_ID}: {exc}")

    model.eval()
    messages = [{"role": "user", "content": "Say hello in one word."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, return_dict = True, return_tensors = "pt"
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens = MIN_NEW_TOKENS,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,
        )

    # The model produced new tokens on the GPU (the real inference proof)...
    assert output.shape[1] > prompt_len, "no tokens were generated on the GPU"
    # ...and they decode to non-empty text (min_new_tokens forces real content).
    reply = tokenizer.decode(output[0][prompt_len:], skip_special_tokens = True)
    assert reply.strip(), "expected a non-empty GPU generation"
