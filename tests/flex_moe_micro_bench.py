# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Tight, single-load decode throughput probe for Qwen3 MoE.

Loads the model once, captures CUDA graphs, then sweeps batch sizes.
Avoids the 30s cold-load tax of the full bench so optimization
iterations can run in under a minute per config.

Usage:
    CUDA_VISIBLE_DEVICES=5 UNSLOTH_FAST_INFERENCE=1 \\
        UNSLOTH_MOE_BACKEND=grouped_mm python -u \\
        tests/flex_moe_micro_bench.py --load_in_4bit --bs 1,4,8,16,32
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
    p.add_argument("--model", default="unsloth/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--bs", default="1,4,8,16,32",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--max_batch_size", type=int, default=32)
    p.add_argument("--warmup_rounds", type=int, default=1)
    p.add_argument("--timed_rounds", type=int, default=2)
    p.add_argument("--tag", default="baseline",
                   help="label for this config in the output JSON")
    p.add_argument("--compile_mode", choices=["off", "walker", "walker_fullgraph"],
                   default="off",
                   help="wrap call_moe_model_with_flex_kwargs in torch.compile")
    p.add_argument("--compile_opts", choices=["stock", "unsloth_O3", "inference_freeze"],
                   default="stock",
                   help="which inductor / dynamo options profile to apply before compile")
    p.add_argument("--explain", action="store_true",
                   help="run torch._dynamo.explain on the walker first to list breaks")
    p.add_argument("--out_dir", default="async_task_outputs/qwen3_moe_grpo_bench_v2")
    args = p.parse_args()

    bs_list = [int(x) for x in args.bs.split(",") if x.strip()]
    os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
    import torch

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from unsloth.inference import flex_moe as _flex_moe_mod

    # Apply inductor / dynamo config BEFORE wrapping with torch.compile.
    if args.compile_opts == "unsloth_O3":
        # Aggressive autotune + coord descent + aggressive_fusion. Matches
        # unsloth_zoo.patching_utils.patch_torch_compile(O3=True).
        import torch._inductor.config as _ic
        import torch._dynamo.config as _dc
        _ic.max_autotune = True
        _ic.max_autotune_pointwise = True
        _ic.coordinate_descent_tuning = True
        _ic.aggressive_fusion = True
        _ic.cuda.use_fast_math = True
        _dc.cache_size_limit = 1024
        _dc.recompile_limit = 1024
        _dc.capture_scalar_outputs = True
        _dc.capture_dynamic_output_shape_ops = True
        print("[micro] inductor/dynamo options: unsloth_O3")
    elif args.compile_opts == "inference_freeze":
        # Inference-friendly: constant-fold weights via freezing=True.
        # Only safe when the model weights won't be updated after compile
        # (true here — we capture graphs post-load and never refresh
        # during bench).
        import torch._inductor.config as _ic
        import torch._dynamo.config as _dc
        _ic.freezing = True
        _ic.max_autotune = True
        _ic.coordinate_descent_tuning = True
        _ic.cuda.use_fast_math = True
        _dc.cache_size_limit = 1024
        _dc.capture_scalar_outputs = True
        print("[micro] inductor/dynamo options: inference_freeze")

    # Apply torch.compile to the decode walker BEFORE the engine is built
    # / graphs are captured, so the compiled kernels get recorded into the
    # CUDA graph.
    if args.compile_mode != "off":
        fullgraph = args.compile_mode == "walker_fullgraph"
        orig_walker = _flex_moe_mod.call_moe_model_with_flex_kwargs
        compile_kwargs = dict(fullgraph=fullgraph, dynamic=False)
        tmode = os.environ.get("FLEX_COMPILE_MODE", "")
        if tmode:
            compile_kwargs["mode"] = tmode
        compiled = torch.compile(orig_walker, **compile_kwargs)
        _flex_moe_mod.call_moe_model_with_flex_kwargs = compiled
        print(f"[micro] wrapped call_moe_model_with_flex_kwargs with "
              f"torch.compile(fullgraph={fullgraph}, mode={tmode or 'default'})")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_batch_size=args.max_batch_size,
    )
    print(f"[micro] loaded in {time.perf_counter() - t0:.1f}s")

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    results = []
    for bs in bs_list:
        prompts = [f"The quick brown fox jumps over fence {i}, then"
                   for i in range(bs)]

        # Warmup (first call captures the graphs for all buckets).
        for _ in range(args.warmup_rounds):
            _ = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)

        # Timed.
        wall = []
        n_tok = []
        for _ in range(args.timed_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
            torch.cuda.synchronize()
            wall.append(time.perf_counter() - t0)
            n_tok.append(sum(len(o.outputs[0].token_ids) for o in outs))

        med_wall = sorted(wall)[len(wall) // 2]
        med_tok = n_tok[len(wall) // 2]
        tps = med_tok / med_wall if med_wall > 0 else 0.0
        # Sanity: print the first completion so we can eyeball for
        # gibberish. A compile bug or bad capture shows up here first.
        sample_text = outs[0].outputs[0].text if outs else ""
        sample_preview = sample_text.replace("\n", "\\n")[:120]
        print(f"[micro] bs={bs:>3}  tok={med_tok:>5}  "
              f"wall={med_wall:.3f}s  tok/s={tps:.1f}")
        print(f"[micro] bs={bs:>3}  completion[0]: {sample_preview!r}")
        results.append({
            "bs": bs,
            "max_new_tokens": args.max_new_tokens,
            "median_wall_s": round(med_wall, 3),
            "median_tok": med_tok,
            "tok_per_s": round(tps, 1),
            "wall_times_s": wall,
            "sample_completion": sample_text[:400],
        })

    peak = torch.cuda.max_memory_reserved() / 1024**3
    precision = "4bit" if args.load_in_4bit else args.dtype
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"micro_bench_{args.tag}_{precision}.json"
    with open(out_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "precision": precision,
            "peak_vram_gb": round(peak, 2),
            "results": results,
        }, f, indent=2)
    print(f"[micro] peak VRAM: {peak:.1f} GB")
    print(f"[micro] wrote {out_path}")


if __name__ == "__main__":
    main()
