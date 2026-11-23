# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor logits_to_keep selects rows like transformers' slice_logits_to_keep."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")


@pytest.fixture(
    scope = "module",
    params = [
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "hf-internal-testing/tiny-random-MistralForCausalLM",
    ],
)
def model(request):
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        request.param, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16
    )
    model.eval()
    return model


def test_tensor_logits_to_keep(model):
    torch.manual_seed(0)
    ids = torch.randint(5, model.config.vocab_size, (2, 10), device = model.device)
    with torch.no_grad():
        full = model(input_ids = ids, use_cache = False).logits
        mask = torch.zeros(2, 10, dtype = torch.bool, device = model.device)
        mask[0, [2, 5]] = True
        mask[1, 9] = True
        out = model(input_ids = ids, use_cache = False, logits_to_keep = mask).logits
        assert out.shape == (3, full.shape[-1])
        torch.testing.assert_close(out, full[mask])

        positions = torch.tensor([3, 7], device = model.device)
        out = model(input_ids = ids, use_cache = False, logits_to_keep = positions).logits
        torch.testing.assert_close(out, full[:, positions])

        rows = torch.tensor([[1, 4], [0, 9]], device = model.device)
        out = model(input_ids = ids, use_cache = False, logits_to_keep = rows).logits
        torch.testing.assert_close(out, torch.take_along_dim(full, rows[..., None], dim = 1))

        out = model(input_ids = ids, use_cache = False, logits_to_keep = 1).logits
        torch.testing.assert_close(out, full[:, -1:])
