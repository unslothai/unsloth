# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""vLLM direct throughput comparison for Qwen3 MoE.

Uses ``vllm.LLM`` directly (no unsloth) to bench the same workload
as ``tests/flex_moe_bench.py`` so the flex+compile_walker numbers
have an apples-to-apples vLLM baseline.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u \\
        tests/flex_moe_vllm_bench.py --load_in_4bit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# DeepGEMM isn't installed in this env; vLLM's FP8 warmup crashes the
# engine-core subprocess without it. Disable before importing vllm.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _gpu_mem_used_gb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="use the -bnb-4bit checkpoint variant")
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--warmup_rounds", type=int, default=1)
    p.add_argument("--timed_rounds", type=int, default=2)
    p.add_argument("--out_dir", default="async_task_outputs/qwen3_moe_grpo_bench_v2")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enable_lora", action="store_true",
                   help="enable vLLM LoRA serving; requires --lora_path")
    p.add_argument("--lora_path", default=None,
                   help="path to a LoRA adapter dir to load per-request")
    p.add_argument("--max_lora_rank", type=int, default=16)
    p.add_argument("--tag", default=None,
                   help="extra label for the output json filename")
    args = p.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    import vllm
    print(f"[vllm] version: {vllm.__version__}")

    dtype_str = "bfloat16" if args.dtype == "bf16" else "float16"

    # For 4bit, tell vLLM to apply bnb quantization on-the-fly to the
    # bf16 checkpoint (there's no pre-quantized unsloth 4bit variant for
    # this model on HF). For FP8, point at the FP8 variant directly.
    model_id = args.model
    quantization = None
    if args.load_in_4bit:
        quantization = "bitsandbytes"
        print(f"[vllm] 4bit: model_id={model_id} quantization=bitsandbytes (on-the-fly)")
    elif os.environ.get("USE_FP8") == "1":
        model_id = args.model + "-FP8"
        quantization = "fp8"
        print(f"[vllm] fp8: model_id={model_id}")

    llm_kwargs = dict(
        model=model_id,
        dtype=dtype_str,
        quantization=quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(args.n_prompts, 8),
        enforce_eager=False,  # default — use cuda graphs
        trust_remote_code=False,
    )
    if args.enable_lora:
        llm_kwargs.update(
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            max_loras=1,
        )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    t_load = time.perf_counter() - t0
    # vLLM uses its own allocator; torch's reserved bytes miss it.
    # Query nvidia-smi instead (visible GPU only — i.e. whatever
    # CUDA_VISIBLE_DEVICES exposes as device 0).
    peak_load = _gpu_mem_used_gb()
    print(f"[vllm] loaded in {t_load:.1f}s peak {peak_load:.1f} GB")

    prompts = [f"The quick brown fox jumps over fence {i}, then"
               for i in range(args.n_prompts)]

    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    gen_kwargs = {}
    if args.enable_lora and args.lora_path is not None:
        from vllm.lora.request import LoRARequest
        gen_kwargs["lora_request"] = LoRARequest(
            lora_name="flex_lora",
            lora_int_id=1,
            lora_path=args.lora_path,
        )
        print(f"[vllm] LoRA enabled: path={args.lora_path} rank<={args.max_lora_rank}")

    # Warmup.
    for _ in range(args.warmup_rounds):
        _ = llm.generate(prompts, sampling_params=sp, use_tqdm=False, **gen_kwargs)

    # Timed.
    wall = []
    tok_counts = []
    for _ in range(args.timed_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False, **gen_kwargs)
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        tok_counts.append(sum(len(o.outputs[0].token_ids) for o in outs))

    peak_gen = _gpu_mem_used_gb()
    med_wall = sorted(wall)[len(wall) // 2]
    med_tok = tok_counts[len(wall) // 2]
    tps = med_tok / med_wall if med_wall > 0 else 0.0

    sample_text = outs[0].outputs[0].text if outs else ""

    print(f"[vllm] wall: {wall}")
    print(f"[vllm] tok counts: {tok_counts}")
    print(f"[vllm] median wall: {med_wall:.3f}s  median tok/s: {tps:.1f}")
    print(f"[vllm] peak VRAM after gen: {peak_gen:.1f} GB")
    print(f"[vllm] sample completion[0]: {sample_text[:200]!r}")

    precision = "4bit" if args.load_in_4bit else args.dtype
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "bench_decode",
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "model": model_id,
        "precision": precision,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "wall_times_s": wall,
        "tok_counts": tok_counts,
        "median_wall_s": round(med_wall, 3),
        "median_tok_s": round(tps, 1),
        "peak_vram_load_gb": round(peak_load, 2),
        "peak_vram_after_gen_gb": round(peak_gen, 2),
        "t_load_s": round(t_load, 1),
        "sample_completion": sample_text[:500],
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"bench_decode_vllm_{precision}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[vllm] wrote {out_path}")


if __name__ == "__main__":
    main()
