"""Smoke-test the ``UNSLOTH_FAST_INFERENCE=1`` path through
``FastLanguageModel.from_pretrained``.

Invoked as:
    CUDA_VISIBLE_DEVICES=2 UNSLOTH_FAST_INFERENCE=1 python tests/flex_fastlm_smoke.py \
        --model unsloth/Qwen3-4B-Base --dtype bf16 --no-lora

Prints tokens/s + the first generated string.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the local fork importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3-4B-Base")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--with_lora", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--prompt", default="The quick brown fox jumps over")
    args = p.parse_args()

    import torch

    os.environ.setdefault("UNSLOTH_FAST_INFERENCE", "1")
    print(f"[smoke] UNSLOTH_FAST_INFERENCE={os.environ.get('UNSLOTH_FAST_INFERENCE')}")

    import unsloth
    print(f"[smoke] unsloth={unsloth.__file__}")
    from unsloth import FastLanguageModel

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
    )
    t_load = time.perf_counter() - t0
    print(f"[smoke] loaded model in {t_load:.1f}s; dtype={model.dtype}")
    print(f"[smoke] hasattr(model, 'vllm_engine'): {hasattr(model, 'vllm_engine')}")
    print(f"[smoke] vllm_engine type: {type(model.vllm_engine).__name__}")

    if args.with_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        print(f"[smoke] PEFT model type: {type(model).__name__}")
        print(f"[smoke] model.vllm_engine bound to PEFT: "
              f"{hasattr(model, 'vllm_engine')}")

    from unsloth.inference.vllm_shim import LoRARequest
    prompts = [args.prompt]
    # Minimal SamplingParams stand-in
    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0
    t1 = time.perf_counter()
    outputs = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    dt = time.perf_counter() - t1
    out = outputs[0]
    n_tok = len(out.outputs[0].token_ids)
    print(f"[smoke] generated {n_tok} tokens in {dt:.2f}s "
          f"({n_tok / dt:.1f} tok/s)")
    print(f"[smoke] prompt: {args.prompt!r}")
    print(f"[smoke] completion: {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
