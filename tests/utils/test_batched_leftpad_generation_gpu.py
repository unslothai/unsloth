"""End-to-end GPU guard for batched left-padded generation (issues #1066, #3699).

Greedy generation in a left-padded batch must match solo batch-size-1
generation for the first PREFIX_TOKENS tokens and must not be gibberish. The
bug makes padded rows diverge into garbage immediately, so an earlier
divergence is only accepted where solo greedy decoding was a near tie
(within TIE_TOLERANCE): batch-size-dependent kernels legitimately flip those
on every device (#7638). Skipped without a GPU. Run: `python -m pytest
tests/utils/test_batched_leftpad_generation_gpu.py -v`.
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(
    not (cuda_available or xpu_available), reason = "requires a CUDA or XPU GPU"
)

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 32
PREFIX_TOKENS = 16
# Batch-size-dependent kernels move bf16 logits by ~0.3; garbage loses by several logits.
TIE_TOLERANCE = 0.5

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


def _generate(
    model,
    tokenizer,
    texts,
    return_scores = False,
):
    inputs = tokenizer(texts, return_tensors = "pt", padding = True, add_special_tokens = False).to(
        device
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
            output_scores = return_scores,
            return_dict_in_generate = True,
        )
    suffixes = out.sequences[:, inputs["input_ids"].shape[1] :]
    tokens = [row.tolist() for row in suffixes]
    if return_scores:
        # Processed scores: what greedy argmax actually picked from.
        return tokens, torch.stack(out.scores, 1).float().cpu()
    return tokens


def _looks_gibberish(text):
    if not text.strip():
        return True
    exclam = text.count("!") / max(len(text), 1)
    nonascii = sum(1 for c in text if ord(c) > 0x2FFF) / max(len(text), 1)
    return exclam > 0.3 or nonascii > 0.5


def test_batched_leftpad_matches_solo_generation(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    texts = [_chat(tokenizer, p) for p in PROMPTS]
    solo = [_generate(model, tokenizer, [t], return_scores = True) for t in texts]
    batched = _generate(model, tokenizer, texts)

    for i, prompt in enumerate(PROMPTS):
        solo_tokens, solo_scores = solo[i][0][0], solo[i][1][0]
        solo_text = tokenizer.decode(solo_tokens, skip_special_tokens = True)
        batch_text = tokenizer.decode(batched[i], skip_special_tokens = True)
        step = next(
            (k for k, (a, b) in enumerate(zip(solo_tokens[:PREFIX_TOKENS], batched[i])) if a != b),
            None,
        )
        margin = (
            0.0
            if step is None
            else (solo_scores[step, solo_tokens[step]] - solo_scores[step, batched[i][step]]).item()
        )
        assert margin <= TIE_TOLERANCE, (
            f"prompt {i} ({prompt[:30]!r}...) diverged from solo generation "
            f"at token {step} by a {margin:.3f} logit margin inside a left-padded "
            "batch; batched left-padded generation is broken again "
            f"(issues #1066, #3699).\n"
            f"solo   : {solo_text!r}\nbatched: {batch_text!r}"
        )
        assert not _looks_gibberish(batch_text), (
            f"prompt {i} produced gibberish in a left-padded batch "
            f"(issues #1066, #3699): {batch_text!r}"
        )
