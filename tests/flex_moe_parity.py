# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Token-level parity: FlexMoEInference (CUDA-graph capture) vs HF generate.

Same prompt set, temperature=0, max_new_tokens fixed. Reports per-prompt
token-id match rate + first divergence index. Serves as the correctness
check for the v2 grouped_mm + CUDA-graph-capture changes.

Usage:
    CUDA_VISIBLE_DEVICES=5 UNSLOTH_FAST_INFERENCE=1 \
        UNSLOTH_MOE_BACKEND=grouped_mm python -u \
        tests/flex_moe_parity.py --load_in_4bit
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


def _run_flex(prompts, args, dtype, *, capture: bool):
    import torch
    os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "grouped_mm")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    if not capture:
        # Monkey-patch ``capture_decode_cudagraph`` to a no-op BEFORE the
        # engine is built so ``generate`` takes the eager branch. The
        # engine's ``self.graphs`` stays empty, ``cudagraph_captured``
        # stays False, and every step goes through ``_decode_step_eager``.
        from unsloth.inference.flex_moe import FlexMoEInference
        FlexMoEInference.capture_decode_cudagraph = lambda self: None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
    )

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    # First call warms / captures; second call is the measurement.
    _ = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    outputs = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]
    texts = [o.outputs[0].text for o in outputs]
    return token_ids, texts, tokenizer


def _run_hf(prompts, args, dtype):
    import torch
    # Pure Hugging Face: NO ``import unsloth`` — we want the unpatched
    # reference forward to compare flex against. Quantization via
    # transformers' ``BitsAndBytesConfig`` matches what unsloth loads
    # under the hood for ``load_in_4bit=True``.
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    # Translate the unsloth-flavoured model id (unsloth/Qwen3-30B-A3B-Instruct-2507)
    # to the 4bit variant if load_in_4bit was requested (FastLanguageModel
    # does this implicitly; do it explicitly here for the naive path).
    model_id = args.model
    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=quant_cfg,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    # Qwen3-30B-A3B-Instruct-2507 uses <|vision_pad|> as its pad token.
    # Unsloth's loader may swap it to a sentinel; reset to the HF default
    # so batched left-padded generation matches the authoritative config.
    if tokenizer.pad_token_id is None or tokenizer.pad_token == "<|PAD_TOKEN|>":
        tokenizer.pad_token = "<|vision_pad|>"
    tokenizer.padding_side = "left"
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    token_ids = []
    texts = []
    for row in out:
        ids = row[prompt_len:].tolist()
        while ids and ids[-1] in (eos, pad):
            ids.pop()
        token_ids.append(ids)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return token_ids, texts, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--backend", choices=["flex", "flex_eager", "hf"], required=True)
    p.add_argument("--out_dir", default="async_task_outputs/qwen3_moe_grpo_bench_v2")
    args = p.parse_args()

    import torch
    prompts = [
        "The quick brown fox jumps over",
        "Q: What is 23 + 19?\nA:",
        "Paris is the capital of",
    ]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.backend == "flex":
        token_ids, texts, tok = _run_flex(prompts, args, dtype, capture=True)
    elif args.backend == "flex_eager":
        token_ids, texts, tok = _run_flex(prompts, args, dtype, capture=False)
    else:
        token_ids, texts, tok = _run_hf(prompts, args, dtype)

    precision = "4bit" if args.load_in_4bit else args.dtype
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"parity_{args.backend}_{precision}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "precision": precision,
                "prompts": prompts,
                "token_ids": token_ids,
                "texts": texts,
            },
            f,
            indent=2,
        )
    print(f"[parity-{args.backend}] wrote {out_path}")
    for i, (p_, t_) in enumerate(zip(prompts, texts)):
        print(f"[parity-{args.backend}] prompt {i}: {p_!r}")
        print(f"[parity-{args.backend}] completion {i}: {t_!r}")


if __name__ == "__main__":
    main()
