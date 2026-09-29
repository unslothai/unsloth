# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""GPU check: generate() continuing from a user KV cache of the conversation history (issue #497)
must emit the same greedy tokens as re-encoding the whole conversation."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU")

MODELS = [
    "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "trl-internal-testing/tiny-Gemma2ForCausalLM",
]
HISTORY = (
    "user: My name is Zorblat and my favourite colour is teal.\nassistant: Nice to meet you!\n"
)
OTHER = "user: I live near the mountains and I own two small cats.\nassistant: How lovely!\n" * 3
QUESTION = "user: What is my name?\nassistant:"
GEN = dict(max_new_tokens = 8, min_new_tokens = 8, do_sample = False)


@pytest.fixture(scope = "module", params = MODELS)
def model_and_tokenizer(request):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        request.param, max_seq_length = 512, load_in_4bit = False, dtype = torch.bfloat16
    )
    FastLanguageModel.for_inference(model)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


def _ids(tokenizer, text):
    return tokenizer(text, return_tensors = "pt", add_special_tokens = False).to("cuda")


def _cache(model, ids):
    with torch.no_grad():
        return model(**ids, use_cache = True).past_key_values


def _as_tuple(cache):
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    return tuple((k, v) for k, v in cache)


def _generate(model, full, **kwargs):
    out = model.generate(**full, **GEN, output_logits = True, return_dict_in_generate = True, **kwargs)
    return out.sequences[0, full.input_ids.shape[1] :], out.logits[0][0].float()


def test_generate_from_history_cache_matches_full_prompt(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    history, full = _ids(tokenizer, HISTORY), _ids(tokenizer, HISTORY + QUESTION)
    n = history.input_ids.shape[1]
    assert torch.equal(full.input_ids[:, :n], history.input_ids)

    expected_tokens, expected_logits = _generate(model, full)
    tolerance = 0.02 * expected_logits.abs().max()
    cache = _cache(model, history)
    runs = [
        _generate(model, full, past_key_values = past) for past in (cache, _as_tuple(cache), cache)
    ]
    tokens, logits = runs[0]
    # Chunked vs full prefill differs only by bf16 rounding; tiny random models have 1-ulp logit ties.
    assert (logits - expected_logits).abs().max() <= tolerance
    assert tokens[0] == expected_tokens[0]
    # Cache object, its tuple form, and a reuse of the same object (not mutated) agree exactly.
    for other_tokens, other_logits in runs[1:]:
        assert torch.equal(other_tokens, tokens) and torch.equal(other_logits, logits)

    # Negative control: a cache of different text moves the logits far past that tolerance.
    other = _ids(tokenizer, OTHER)
    wrong = {k: v[:, :n] for k, v in other.items()}
    assert wrong["input_ids"].shape[1] == n
    _, wrong_logits = _generate(model, full, past_key_values = _cache(model, wrong))
    assert (wrong_logits - expected_logits).abs().max() > 5 * tolerance
