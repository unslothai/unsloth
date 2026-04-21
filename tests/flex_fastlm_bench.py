"""Batched steady-state throughput bench through ``FastLanguageModel`` +
``UNSLOTH_FAST_INFERENCE=1``. First ``generate`` call primes CUDA graphs;
subsequent calls report steady state. Compare against April CLI-only
numbers for the same workload."""

import os, sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--dtype", choices = ["bf16", "fp16"], default = "bf16")
    p.add_argument("--n_prompts", type = int, default = 8)
    p.add_argument("--max_new_tokens", type = int, default = 64)
    p.add_argument("--max_batch_size", type = int, default = 16)
    p.add_argument("--max_seq_length", type = int, default = 1024)
    p.add_argument("--n_rounds", type = int, default = 3)
    args = p.parse_args()

    os.environ.setdefault("UNSLOTH_FAST_INFERENCE", "1")
    import unsloth
    from unsloth import FastLanguageModel

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    model, tok = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = dtype,
        load_in_4bit = False,
        fast_inference = True,
        max_batch_size = args.max_batch_size,
    )
    print(f"[bench] model={args.model} dtype={args.dtype}")
    prompts = [
        f"In one sentence, a fact about {t} is"
        for t in [
            "the moon",
            "gravity",
            "the ocean",
            "the sun",
            "honey",
            "rain",
            "trees",
            "mountains",
        ][: args.n_prompts]
    ]

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    # Warmup round — captures CUDA graphs.
    print("[bench] warmup (CUDA graph capture)...")
    t0 = time.perf_counter()
    _ = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)
    print(f"[bench] warmup wall: {time.perf_counter() - t0:.2f}s")

    walls = []
    tok_counts = []
    for r in range(args.n_rounds):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        outs = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        walls.append(dt)
        tok_counts.append(n_tok)
        print(f"[bench] round {r}: {n_tok} toks in {dt:.2f}s -> {n_tok/dt:.1f} tok/s")

    if walls:
        wall_med = sorted(walls)[len(walls) // 2]
        tok_med = tok_counts[len(walls) // 2]
        print(
            f"[bench] median: {tok_med} toks in {wall_med:.2f}s "
            f"=> {tok_med / wall_med:.1f} tok/s"
        )


if __name__ == "__main__":
    main()
