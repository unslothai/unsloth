# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Decode throughput bench: FlexMoEInference vs HF generate on Qwen3 MoE.

Apples-to-apples decode on the same prompt set + new-token budget,
same LoRA rank, same precision. Mirrors PR #5123's
``tests/flex_fastlm_bench.py`` shape.

Usage:
    CUDA_VISIBLE_DEVICES=0 UNSLOTH_FAST_INFERENCE=1 python -u \
        tests/flex_moe_bench.py --backend flex --load_in_4bit

    CUDA_VISIBLE_DEVICES=0 python -u \
        tests/flex_moe_bench.py --backend hf   --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices = ["flex", "hf", "hf_naive"], default = "flex")
    p.add_argument("--model", default = "unsloth/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--dtype", choices = ["bf16", "fp16"], default = "bf16")
    p.add_argument("--load_in_4bit", action = "store_true")
    p.add_argument("--n_prompts", type = int, default = 8)
    p.add_argument("--max_new_tokens", type = int, default = 64)
    p.add_argument("--max_seq_length", type = int, default = 1024)
    p.add_argument("--warmup_rounds", type = int, default = 1)
    p.add_argument("--timed_rounds", type = int, default = 2)
    p.add_argument("--out_dir", default = "async_task_outputs/qwen3_moe_grpo_bench")
    args = p.parse_args()

    import torch

    if args.backend == "flex":
        os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()

    if args.backend == "hf_naive":
        # Pure transformers path: NO ``import unsloth`` so none of
        # Unsloth's Qwen3 MoE attention / MLP patches run. This is the
        # fair naive reference to compare flex fast-inference against.
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        quant_cfg = None
        if args.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None or tokenizer.pad_token == "<|PAD_TOKEN|>":
            tokenizer.pad_token = "<|vision_pad|>"
        tokenizer.padding_side = "left"
        attn_impl = os.environ.get("HF_ATTN_IMPL", "eager")
        print(f"[bench] HF attn_implementation={attn_impl}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            quantization_config=quant_cfg,
            device_map="cuda",
            attn_implementation=attn_impl,
        )
        model.eval()
        print(f"[bench] pure transformers (no unsloth patches)")
    else:
        import unsloth
        print(f"[bench] unsloth={unsloth.__file__}")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model,
            max_seq_length = args.max_seq_length,
            dtype = dtype,
            load_in_4bit = args.load_in_4bit,
            fast_inference = args.backend == "flex",
        )

    t_load = time.perf_counter() - t_load0
    peak_load = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[bench] loaded in {t_load:.1f}s peak {peak_load:.1f} GB")

    prompts = [f"The quick brown fox jumps over fence {i}, then"
               for i in range(args.n_prompts)]

    if args.backend == "flex":
        class _SP:
            max_tokens = args.max_new_tokens
            temperature = 0.0

        # Warmup
        for _ in range(args.warmup_rounds):
            _ = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)

        # Timed
        wall_times = []
        tok_counts = []
        for _ in range(args.timed_rounds):
            t0 = time.perf_counter()
            outs = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)
            wall_times.append(time.perf_counter() - t0)
            tok_counts.append(
                sum(len(o.outputs[0].token_ids) for o in outs)
            )
    else:
        # HF generate (shared for "hf" unsloth-patched and "hf_naive" pure).
        inputs = tokenizer(prompts, return_tensors = "pt", padding = True).to("cuda")
        gen_kwargs = dict(
            max_new_tokens = args.max_new_tokens,
            do_sample = False,
            temperature = 1.0,
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # Warmup
        for _ in range(args.warmup_rounds):
            _ = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

        wall_times = []
        tok_counts = []
        for _ in range(args.timed_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()
            wall_times.append(time.perf_counter() - t0)
            n_new = (out.shape[1] - inputs["input_ids"].shape[1]) * out.shape[0]
            tok_counts.append(n_new)

    peak_gen = torch.cuda.max_memory_reserved() / 1024**3
    median_wall = sorted(wall_times)[len(wall_times) // 2]
    median_tok = tok_counts[len(wall_times) // 2]
    tok_per_s = median_tok / median_wall if median_wall > 0 else 0.0

    print(f"[bench] wall: {wall_times}")
    print(f"[bench] tok counts: {tok_counts}")
    print(f"[bench] median wall: {median_wall:.2f}s  median tok/s: {tok_per_s:.1f}")
    print(f"[bench] peak VRAM after gen: {peak_gen:.1f} GB")

    precision = "4bit" if args.load_in_4bit else args.dtype
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    summary = {
        "phase": "bench_decode",
        "backend": args.backend,
        "model": args.model,
        "precision": precision,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "wall_times_s": wall_times,
        "tok_counts": tok_counts,
        "median_wall_s": round(median_wall, 3),
        "median_tok_s": round(tok_per_s, 1),
        "peak_vram_load_gb": round(peak_load, 2),
        "peak_vram_after_gen_gb": round(peak_gen, 2),
        "t_load_s": round(t_load, 1),
    }
    with open(out_dir / f"bench_decode_{args.backend}_{precision}.json", "w") as f:
        json.dump(summary, f, indent = 2)
    print(f"[bench] wrote {out_dir / f'bench_decode_{args.backend}_{precision}.json'}")


if __name__ == "__main__":
    main()
