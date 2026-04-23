# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Token parity for FlexGemma4Inference / FlexGemma4MoEInference vs pure HF.

Usage::
    CUDA_VISIBLE_DEVICES=2 UNSLOTH_FAST_INFERENCE=1 python -u \\
        tests/flex_gemma4_parity.py --backend flex --model unsloth/gemma-4-31B-it

    CUDA_VISIBLE_DEVICES=3 python -u tests/flex_gemma4_parity.py \\
        --backend hf --model unsloth/gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CHAT_PROMPTS = [
    "In one sentence, what is Paris?",
    "What is 23 + 19? Answer in one word.",
    "Continue: The quick brown fox jumps over",
]


def _run_flex(args, dtype, *, capture: bool, lora_path=None):
    import torch
    os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    import unsloth  # noqa
    from unsloth import FastLanguageModel

    if not capture:
        # Force the eager decode path across arches.
        try:
            from unsloth.inference.flex_gemma4 import FlexGemma4Inference
            FlexGemma4Inference.capture_decode_cudagraph = lambda self: None
        except Exception:
            pass
        try:
            from unsloth.inference.flex_gemma4_moe import FlexGemma4MoEInference
            FlexGemma4MoEInference.capture_decode_cudagraph = lambda self: None
        except Exception:
            pass

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        fast_inference=True,
        max_batch_size=4,
        gpu_memory_utilization=0.6,
    )

    if lora_path is not None:
        model.load_adapter(lora_path, adapter_name="default")
        print(f"[parity-flex] LoRA attached from {lora_path}")

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in CHAT_PROMPTS[: args.num_prompts]
    ]

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    _ = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    outs = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    return token_ids, texts, tok


def _run_hf(args, dtype, *, lora_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map="cuda",
            attn_implementation="eager",
        )
    except Exception:
        # Multimodal Gemma 4 (ConditionalGeneration) — load the top-level class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=dtype, device_map="cuda",
            attn_implementation="eager",
        )
    model.eval()

    if lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print(f"[parity-hf] LoRA attached from {lora_path}")

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in CHAT_PROMPTS[: args.num_prompts]
    ]
    inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    token_ids = []
    texts = []
    for row in out:
        ids = row[prompt_len:].tolist()
        while ids and ids[-1] in (eos, pad):
            ids.pop()
        token_ids.append(ids)
        texts.append(tok.decode(ids, skip_special_tokens=False))
    return token_ids, texts, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gemma-4-31B-it")
    p.add_argument("--backend", choices=["flex", "flex_eager", "hf"], required=True)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_prompts", type=int, default=3)
    p.add_argument("--lora_path", default=None)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--out_dir", default="async_task_outputs/gemma4_moe_bench")
    args = p.parse_args()

    import torch
    dtype = torch.bfloat16

    if args.backend == "flex":
        token_ids, texts, _ = _run_flex(args, dtype, capture=True, lora_path=args.lora_path)
    elif args.backend == "flex_eager":
        token_ids, texts, _ = _run_flex(args, dtype, capture=False, lora_path=args.lora_path)
    else:
        token_ids, texts, _ = _run_hf(args, dtype, lora_path=args.lora_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    suffix = "_lora" if args.lora_path else ""
    out_path = out_dir / f"parity_{model_tag}_{args.backend}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "model": args.model,
                "prompts": CHAT_PROMPTS[: args.num_prompts],
                "token_ids": token_ids,
                "texts": texts,
            },
            f,
            indent=2,
        )
    print(f"[parity-{args.backend}] wrote {out_path}")
    for i, (pp, tt) in enumerate(zip(CHAT_PROMPTS[: args.num_prompts], texts)):
        print(f"[parity-{args.backend}] prompt {i}: {pp!r}")
        print(f"[parity-{args.backend}] completion {i}: {tt!r}")


if __name__ == "__main__":
    main()
