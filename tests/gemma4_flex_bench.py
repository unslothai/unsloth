"""Throughput bench for Unsloth flex fast-inference on Gemma 4.

Runs ``FastLanguageModel(fast_inference=True)`` through the flex engine
(``UNSLOTH_FAST_INFERENCE=1``) and measures tok/s at a list of batch sizes.
Captures CUDA graphs at each bucket; the bench reports the post-warmup pass.

Example:
    CUDA_VISIBLE_DEVICES=0 UNSLOTH_FAST_INFERENCE=1 \\
        UNSLOTH_MOE_BACKEND=grouped_mm \\
        python -u tests/gemma4_flex_bench.py \\
        --model unsloth/gemma-4-26b-a4b-it --batch_sizes 1 4 8 16 \\
        --max_new_tokens 64 --json_out async_task_outputs/flex_bench_26b.json

Pair with tests/gemma4_fast_inference_parity.py for HF-naive parity and
with tests/gemma4_fast_bench.py (vLLM nightly path) for cross-engine
throughput comparison.
"""
import argparse
import json
import os
import time

os.environ.setdefault("UNSLOTH_FAST_INFERENCE", "1")
os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import unsloth  # noqa: F401, E402
import torch  # noqa: E402


SHORT_PROMPT = "In one sentence, what is Paris?"
LONG_POOL = [
    "In one sentence, what is Paris?",
    "What is 23 + 19? Answer in one word.",
    "Continue this phrase: The quick brown fox jumps over",
    "Name three primary colors.",
    "Who wrote the play Hamlet?",
    "What is the capital of Japan?",
    "What does the acronym 'NASA' stand for?",
    "Explain the water cycle in one sentence.",
    "Give a two-word summary of the French Revolution.",
    "What is the tallest mountain on Earth?",
    "Define 'entropy' in one sentence.",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
    "Name one prime number larger than 10.",
    "What color do you get when you mix blue and yellow?",
    "Give one example of an amphibian.",
]


def _render(tok, prompts):
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]


class _SP:
    def __init__(self, n):
        self.max_tokens = n
        self.temperature = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    t_load = time.perf_counter()
    model, tok_raw = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16, load_in_4bit=False, fast_inference=True,
        max_batch_size=max(args.batch_sizes),
        gpu_memory_utilization=0.6,
    )
    t_load = time.perf_counter() - t_load
    tok = tok_raw.tokenizer if hasattr(tok_raw, "tokenizer") else tok_raw
    print(f"[flex] load {t_load:.1f}s")

    sp = _SP(args.max_new_tokens)
    results = []

    for bs in args.batch_sizes:
        prompts = _render(tok, LONG_POOL[:bs])
        # Warmup: primes the cudagraph bucket and walker compile.
        _ = model.fast_generate(prompts, sampling_params=sp, use_tqdm=False)
        t0 = time.perf_counter()
        outs = model.fast_generate(prompts, sampling_params=sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        tok_s = total / dt if dt > 0 else 0.0
        results.append({
            "batch_size": bs, "gen_s": round(dt, 3),
            "total_tokens": total, "tok_s": round(tok_s, 1),
        })
        print(f"[flex bs={bs}] {total} tok in {dt:.2f}s -> {tok_s:.1f} tok/s")

    # Parity sanity: run SHORT_PROMPT at bs=1, save ids.
    p = _render(tok, [SHORT_PROMPT])
    outs = model.fast_generate(p, sampling_params=sp, use_tqdm=False)
    parity_ids = list(outs[0].outputs[0].token_ids)
    parity_text = outs[0].outputs[0].text

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    summary = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "load_s": round(t_load, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "batches": results,
        "parity_ids": parity_ids,
        "parity_text": parity_text,
    }
    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[flex] wrote {args.json_out}")


if __name__ == "__main__":
    main()
