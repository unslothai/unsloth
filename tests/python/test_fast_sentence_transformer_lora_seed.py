# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastSentenceTransformer.get_peft_model seeds the adapter init with random_state on the fast
encoder path (BERT, ModernBERT, ...), as FastModel.get_peft_model does. It was ignored there, so the
LoRA matrices depended on whatever had used the RNG before, and under torchrun every rank drew its own
adapters (native FSDP2 then sharded a mix of them)."""

import pytest
from real_accelerator import has_real_accelerator

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

# The loader reads device properties even with device_map = "cpu".
pytestmark = pytest.mark.skipif(
    not has_real_accelerator(), reason = "the loader needs an accelerator"
)

ENCODER = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def _adapters(random_state, burn):
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        ENCODER, max_seq_length = 32, dtype = torch.float32, device_map = "cpu"
    )
    assert getattr(model, "_unsloth_fast_encoder", False)  # the path under test
    torch.rand(burn)  # a different RNG history before the adapters are made
    model = FastSentenceTransformer.get_peft_model(
        model, r = 4, lora_alpha = 8, target_modules = ["query", "value"], random_state = random_state
    )
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_A" in n}


def test_random_state_fixes_the_adapters():
    a = _adapters(7, burn = 1)
    b = _adapters(7, burn = 1000)
    assert a and a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_a_different_random_state_gives_different_adapters():
    a = _adapters(7, burn = 1)
    c = _adapters(8, burn = 1)
    assert any(not torch.equal(a[n], c[n]) for n in a)
