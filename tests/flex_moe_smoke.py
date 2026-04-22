# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Smoke-test ``UNSLOTH_FAST_INFERENCE=1`` on a Qwen3 MoE model.

Mirrors ``tests/flex_fastlm_smoke.py`` but targets the new
``FlexMoEInference`` path added for ``Qwen3MoeForCausalLM``.

Invoked as:
    CUDA_VISIBLE_DEVICES=0 UNSLOTH_FAST_INFERENCE=1 python -u \
        tests/flex_moe_smoke.py \
        --model unsloth/Qwen3-30B-A3B-Instruct-2507 \
        --load_in_4bit

Writes a small JSON summary to
``async_task_outputs/qwen3_moe_grpo_bench/smoke_A_{precision}.json``.
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
    p.add_argument(
        "--model", default = "unsloth/Qwen3-30B-A3B-Instruct-2507"
    )
    p.add_argument("--dtype", choices = ["bf16", "fp16"], default = "bf16")
    p.add_argument("--load_in_4bit", action = "store_true")
    p.add_argument("--with_lora", action = "store_true")
    p.add_argument("--max_new_tokens", type = int, default = 32)
    p.add_argument("--max_seq_length", type = int, default = 1024)
    p.add_argument("--prompt", default = "The quick brown fox jumps over")
    p.add_argument("--out_dir", default = "async_task_outputs/qwen3_moe_grpo_bench")
    args = p.parse_args()

    import torch

    os.environ.setdefault("UNSLOTH_FAST_INFERENCE", "1")
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
    print(f"[smoke] UNSLOTH_FAST_INFERENCE={os.environ.get('UNSLOTH_FAST_INFERENCE')}")
    print(f"[smoke] UNSLOTH_MOE_BACKEND={os.environ.get('UNSLOTH_MOE_BACKEND')}")

    import unsloth

    print(f"[smoke] unsloth={unsloth.__file__}")
    from unsloth import FastLanguageModel

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = dtype,
        load_in_4bit = args.load_in_4bit,
        fast_inference = True,
    )
    t_load = time.perf_counter() - t0
    peak_after_load = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[smoke] loaded model in {t_load:.1f}s; peak VRAM after load: {peak_after_load:.2f} GB")
    print(f"[smoke] hasattr(model, 'vllm_engine'): {hasattr(model, 'vllm_engine')}")
    print(f"[smoke] vllm_engine type: {type(model.vllm_engine).__name__}")
    arch = getattr(model.vllm_engine, "arch", "?")
    impl = type(model.vllm_engine._impl).__name__
    print(f"[smoke] FlexEngine.arch={arch}  impl={impl}")

    if args.with_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "gate_up_proj",
            ],
            lora_alpha = 32,
            lora_dropout = 0.0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        print(f"[smoke] PEFT model type: {type(model).__name__}")

    prompts = [args.prompt]

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    # First call includes prefill + any lazy engine bring-up; measure separately.
    t_first0 = time.perf_counter()
    outputs = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)
    t_first = time.perf_counter() - t_first0
    out = outputs[0]
    n_tok = len(out.outputs[0].token_ids)

    # Warm steady-state: run again and measure.
    t_warm0 = time.perf_counter()
    outputs2 = model.fast_generate(prompts, sampling_params = _SP(), use_tqdm = False)
    t_warm = time.perf_counter() - t_warm0
    n_tok_warm = len(outputs2[0].outputs[0].token_ids)

    peak_after_gen = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"[smoke] first call: generated {n_tok} tokens in {t_first:.2f}s "
        f"({n_tok / t_first:.1f} tok/s)"
    )
    print(
        f"[smoke] warm call: generated {n_tok_warm} tokens in {t_warm:.2f}s "
        f"({n_tok_warm / t_warm:.1f} tok/s)"
    )
    print(f"[smoke] peak VRAM after gen: {peak_after_gen:.2f} GB")
    print(f"[smoke] prompt: {args.prompt!r}")
    print(f"[smoke] completion: {out.outputs[0].text!r}")

    precision = "4bit" if args.load_in_4bit else args.dtype
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    summary = {
        "phase": "smoke_A",
        "model": args.model,
        "precision": precision,
        "dtype": str(dtype),
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "with_lora": args.with_lora,
        "t_load_s": round(t_load, 2),
        "peak_vram_after_load_gb": round(peak_after_load, 2),
        "peak_vram_after_gen_gb": round(peak_after_gen, 2),
        "first_call_s": round(t_first, 2),
        "first_call_tok_s": round(n_tok / t_first, 1),
        "warm_call_s": round(t_warm, 2),
        "warm_call_tok_s": round(n_tok_warm / t_warm, 1),
        "arch": arch,
        "impl": impl,
        "prompt": args.prompt,
        "completion": out.outputs[0].text,
    }
    with open(out_dir / f"smoke_A_{precision}.json", "w") as f:
        json.dump(summary, f, indent = 2)
    print(f"[smoke] wrote {out_dir / f'smoke_A_{precision}.json'}")


if __name__ == "__main__":
    main()
