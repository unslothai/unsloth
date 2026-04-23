# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Token parity for FlexGptOssInference vs pure HF generate on gpt-oss-20b.

Usage::
    # Flex side (cudagraph off in Phase 1).
    CUDA_VISIBLE_DEVICES=2 UNSLOTH_FAST_INFERENCE=1 python -u \\
        tests/flex_gpt_oss_parity.py --backend flex

    # HF reference.
    CUDA_VISIBLE_DEVICES=3 python -u tests/flex_gpt_oss_parity.py \\
        --backend hf
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


def _run_flex(args, dtype):
    import torch
    os.environ["UNSLOTH_FAST_INFERENCE"] = "1"
    import unsloth  # noqa
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        fast_inference=True,
        max_batch_size=4,
        gpu_memory_utilization=0.6,
    )

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in CHAT_PROMPTS
    ]

    class _SP:
        max_tokens = args.max_new_tokens
        temperature = 0.0

    # First call primes; second is the measurement.
    _ = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    outs = model.fast_generate(prompts, sampling_params=_SP(), use_tqdm=False)
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    return token_ids, texts, tok


def _run_hf(args, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in CHAT_PROMPTS
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
    p.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    p.add_argument("--backend", choices=["flex", "hf"], required=True)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--out_dir", default="async_task_outputs/qwen3_moe_grpo_bench_v2")
    args = p.parse_args()

    import torch
    dtype = torch.bfloat16

    if args.backend == "flex":
        token_ids, texts, _ = _run_flex(args, dtype)
    else:
        token_ids, texts, _ = _run_hf(args, dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"parity_gptoss_{args.backend}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "backend": args.backend,
                "model": args.model,
                "prompts": CHAT_PROMPTS,
                "token_ids": token_ids,
                "texts": texts,
            },
            f,
            indent=2,
        )
    print(f"[parity-{args.backend}] wrote {out_path}")
    for i, (pp, tt) in enumerate(zip(CHAT_PROMPTS, texts)):
        print(f"[parity-{args.backend}] prompt {i}: {pp!r}")
        print(f"[parity-{args.backend}] completion {i}: {tt!r}")


if __name__ == "__main__":
    main()
