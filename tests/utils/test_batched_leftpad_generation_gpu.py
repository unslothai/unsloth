"""End-to-end GPU guard for batched left-padded generation (issues #1066, #3699).

Greedy generation in a left-padded batch must match solo batch-size-1
generation for the first PREFIX_TOKENS tokens (the bug makes padded rows
diverge into garbage immediately; a full-length match would be flaky due to
benign batch-numerics tie-flips deep in the sequence) and must not be
gibberish. Skipped without CUDA. Run: `python -m pytest
tests/utils/test_batched_leftpad_generation_gpu.py -v`.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not cuda_available, reason = "requires a CUDA GPU")

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
PREFIX_TOKENS = 16

PROMPTS = [
    "Give me a short introduction to large language model.",
    "Here is an experiment log: "
    + " ".join(f"run {i} completed with stable throughput and no anomalies;" for i in range(1, 41))
    + " In one sentence, what is the overall conclusion?",
]


@pytest.fixture(scope = "module")
def model_and_tokenizer():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize = False,
        add_generation_prompt = True,
    )


def _generate(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors = "pt", padding = True, add_special_tokens = False).to(
        "cuda"
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,
            temperature = None,
            top_p = None,
            top_k = None,
            use_cache = True,
            pad_token_id = tokenizer.pad_token_id,
        )
    suffixes = out[:, inputs["input_ids"].shape[1] :]
    return [row.tolist() for row in suffixes]


def _looks_gibberish(text):
    if not text.strip():
        return True
    exclam = text.count("!") / max(len(text), 1)
    nonascii = sum(1 for c in text if ord(c) > 0x2FFF) / max(len(text), 1)
    return exclam > 0.3 or nonascii > 0.5


def test_batched_leftpad_matches_solo_generation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    texts = [_chat(tokenizer, p) for p in PROMPTS]
    solo = [_generate(model, tokenizer, [t])[0] for t in texts]
    batched = _generate(model, tokenizer, texts)

    for i, prompt in enumerate(PROMPTS):
        solo_text = tokenizer.decode(solo[i], skip_special_tokens = True)
        batch_text = tokenizer.decode(batched[i], skip_special_tokens = True)
        assert batched[i][:PREFIX_TOKENS] == solo[i][:PREFIX_TOKENS], (
            f"prompt {i} ({prompt[:30]!r}...) diverged from solo generation "
            f"within the first {PREFIX_TOKENS} tokens inside a left-padded "
            "batch; batched left-padded generation is broken again "
            f"(issues #1066, #3699).\n"
            f"solo   : {solo_text!r}\nbatched: {batch_text!r}"
        )
        assert not _looks_gibberish(batch_text), (
            f"prompt {i} produced gibberish in a left-padded batch "
            f"(issues #1066, #3699): {batch_text!r}"
        )
