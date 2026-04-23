"""Greedy parity: FastLanguageModel(fast_inference=True) vs HF naive on Gemma 4.

Covers all four Gemma 4 variants (E2B / E4B dense, 31B dense, 26B-A4B MoE) via
``--model``. Runs HF FIRST, frees the GPU, then runs vLLM — loading vLLM first
in the same process leaves global state (allocator, compile cache, patched
functions) that subtly perturbs a later plain HF run, producing false
divergences. HF-first ordering gives bitwise matches.

Requires vLLM nightly (>= 2026-04-17 for `vllm#39291` Gemma 4 LoRA) plus the
`unsloth-zoo#603` vLLM Gemma 4 patches.

Example:
    CUDA_VISIBLE_DEVICES=0 python -u tests/gemma4_fast_inference_parity.py \\
        --model unsloth/gemma-4-26b-a4b-it --max_new_tokens 32
"""
import argparse
import gc
import json
import os

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")

# Import unsloth at module scope so its patches apply to the HF run as well —
# users always import unsloth before touching transformers in practice.
import unsloth  # noqa: F401, E402
import torch  # noqa: E402


def _render(tok, raw_prompts):
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in raw_prompts
    ]


def run_hf_first(model_name, raw_prompts, max_new_tokens):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    prompts = _render(tok, raw_prompts)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0",
    )
    model.eval()
    ids, texts = [], []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None,
            )
        new = gen[0, enc.input_ids.shape[1]:]
        ids.append(new.tolist())
        texts.append(tok.decode(new, skip_special_tokens=True))

    del model, proc
    gc.collect()
    torch.cuda.empty_cache()
    return ids, texts, prompts


def run_vllm_second(model_name, prompts, max_new_tokens):
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=1024, dtype=torch.bfloat16,
        load_in_4bit=False, fast_inference=True, max_batch_size=8,
        gpu_memory_utilization=0.6, max_lora_rank=16,
    )
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outs = model.fast_generate(prompts, sampling_params=sp, use_tqdm=False)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    return ids, texts


DEFAULT_PROMPTS = [
    "In one sentence, what is Paris?",
    "What is 23 + 19? Answer in one word.",
    "Continue this phrase: The quick brown fox jumps over",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-4-26b-a4b-it")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    print(f"=== HF first ({args.model}) ===")
    hf_ids, hf_texts, prompts = run_hf_first(
        args.model, DEFAULT_PROMPTS, args.max_new_tokens,
    )
    for i, t in enumerate(hf_texts):
        print(f"[hf]   P{i}: {t[:120]!r}")

    print("\n=== vLLM second ===")
    fast_ids, fast_texts = run_vllm_second(
        args.model, prompts, args.max_new_tokens,
    )
    for i, t in enumerate(fast_texts):
        print(f"[fast] P{i}: {t[:120]!r}")

    print("\n=== Match ===")
    total, matched, rows = 0, 0, []
    for i, (a, b) in enumerate(zip(fast_ids, hf_ids)):
        n = min(len(a), len(b))
        m = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
        total += n
        matched += m
        print(f"[P{i}] {m}/{n}")
        rows.append({
            "prompt_id": i, "match": m, "total": n,
            "fast": fast_texts[i], "hf": hf_texts[i],
        })
    print(f"\nTotal {matched}/{total}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "model": args.model, "rows": rows,
                "matched": matched, "total": total,
            }, f, indent=2)

    # Exit non-zero on any divergence so CI can gate on bitwise parity.
    if matched != total:
        raise SystemExit(f"Divergence: {matched}/{total}")


if __name__ == "__main__":
    main()
