# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""
Smoke test for the unsloth-blackwell image.

What this checks (in order, fail-fast):
  1. torch sees the GPU and the arch list contains sm_100 + sm_120.
  2. The runtime device's compute capability is supported.
  3. xformers / bitsandbytes / triton import without ImportError.
  4. unsloth imports and exposes FastLanguageModel.
  5. A 5-step LoRA train on a tiny model actually runs forward + backward.

Run inside the container:
    docker run --rm --gpus all unsloth-blackwell:latest python /workspace/smoke_test.py

Skip step 5 (faster, no model download):
    docker run --rm --gpus all unsloth-blackwell:latest python /workspace/smoke_test.py --skip-train
"""

from __future__ import annotations

import argparse
import sys


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush = True)


def check_torch() -> tuple[int, int]:
    banner("torch + arch list")
    import torch

    # Raw C++ accessor works even without CUDA (partial smoke test on no-GPU host).
    arches = torch._C._cuda_getArchFlags().split()
    print(f"torch       {torch.__version__}")
    print(f"cuda build  {torch.version.cuda}")
    print(f"arches      {arches}")
    assert "sm_100" in arches, f"sm_100 missing: {arches}"
    assert "sm_120" in arches, f"sm_120 missing: {arches}"

    assert torch.cuda.is_available(), "CUDA not visible -- did you pass --gpus all?"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device 0    {name}  sm_{cap[0]}{cap[1]}")
    # cu128 wheels ship SASS down to sm_75 (Turing); match the entrypoint floor so
    # a Turing-only runner doesn't false-fail (Turing falls back to fp16).
    if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
        sys.exit(f"FAIL: pre-Turing GPU {name} is not supported by this image")
    if cap[0] < 8:
        print(f"NOTE: {name} is Turing (sm_{cap[0]}{cap[1]}) -- bf16 unavailable, fp16 fallback.")
    return cap


def check_imports() -> None:
    banner("dep imports")
    import triton

    print(f"triton      {triton.__version__}")
    # Import order matters: unsloth before transformers/trl/peft (so its patches
    # land) and before unsloth_zoo (which needs the UNSLOTH_IS_PRESENT marker).
    import unsloth

    print(f"unsloth     {unsloth.__version__}")
    import unsloth_zoo

    print(f"unsloth_zoo {unsloth_zoo.__version__}")
    # xformers has no aarch64 cu128 wheel; arm64 omits it. Best-effort so one
    # script covers both arches.
    try:
        import xformers
        print(f"xformers    {xformers.__version__}")
    except ImportError:
        print("xformers    (missing -- expected on arm64 [huggingface] extras)")
    import bitsandbytes as bnb

    print(f"bnb         {bnb.__version__}")
    import transformers

    print(f"transformers {transformers.__version__}")
    import trl

    print(f"trl         {trl.__version__}")
    import peft

    print(f"peft        {peft.__version__}")


def check_unsloth_import() -> None:
    banner("unsloth FastLanguageModel reachable")
    # Already imported in check_imports(); this re-import is a no-op.
    import unsloth
    from unsloth import FastLanguageModel

    print(f"unsloth     {unsloth.__version__}")
    print(f"FastLanguageModel  {FastLanguageModel}")


def check_tiny_train(cap: tuple[int, int]) -> None:
    banner("tiny LoRA train (5 steps)")
    import os

    # Unsloth must be imported first.
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    # Small, public, no-gate. ~125M params.
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    print(f"loading     {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
    )

    prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: 2 + 2 = ?\nA:",
        "Q: Name a primary color.\nA:",
        "Q: Hello, who are you?\nA:",
    ] * 2
    enc = tokenizer(prompts, return_tensors = "pt", padding = True, truncation = True, max_length = 64)
    enc = {k: v.cuda() for k, v in enc.items()}
    labels = enc["input_ids"].clone()

    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-4)
    for step in range(5):
        out = model(**enc, labels = labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none = True)
        print(f"step {step}  loss={out.loss.item():.4f}", flush = True)

    print("OK: 5 LoRA steps completed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-train",
        action = "store_true",
        help = "Skip the tiny LoRA training step (no HF download).",
    )
    args = ap.parse_args()

    cap = check_torch()
    check_imports()
    check_unsloth_import()
    if not args.skip_train:
        check_tiny_train(cap)

    banner("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
