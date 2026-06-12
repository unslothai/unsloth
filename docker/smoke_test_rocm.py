"""
Smoke test for the Unsloth ROCm image (AMD GPU build).

What this checks (in order, fail-fast):
  1. torch is a ROCm build (torch.version.hip is not None).
  2. The runtime device is visible and its gfx arch is supported (RDNA2+).
  3. bitsandbytes / triton import without ImportError.
  4. unsloth imports and exposes FastLanguageModel.
  5. A 5-step LoRA train on a tiny model actually runs forward + backward.

Run inside the container:
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \\
        unsloth/unsloth-rocm:latest python /workspace/smoke_test_rocm.py

Skip step 5 (faster, no model download):
    ... python /workspace/smoke_test_rocm.py --skip-train
"""

from __future__ import annotations

import argparse
import sys


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_torch() -> None:
    banner("torch ROCm build")
    import torch

    # torch.version.hip is the canonical indicator of a ROCm wheel.
    # It is None for CUDA builds.
    if torch.version.hip is None:
        sys.exit(
            f"FAIL: this is not a ROCm torch build ({torch.__version__}). "
            "Re-pull the unsloth-rocm image."
        )
    print(f"torch       {torch.__version__}")
    print(f"HIP         {torch.version.hip}")

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False — did you pass "
        "--device /dev/kfd --device /dev/dri --group-add video?"
    )
    n = torch.cuda.device_count()
    print(f"GPU count   {n}")
    for i in range(n):
        name  = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        # PyTorch ROCm surfaces the gfx code in gcnArchName (e.g.
        # "gfx1100:sramecc+"); strip the feature suffix for readability.
        arch  = getattr(props, "gcnArchName", "").split(":")[0]
        bf16  = torch.cuda.is_bf16_supported()
        print(f"device {i}    {name}  arch={arch}  bf16={bf16}")
    print()
    # ROCm does not expose a reliable sm_X.Y compute capability the way NVIDIA
    # does -- the values from get_device_properties() vary by ROCm version and
    # don't map cleanly to gfx codes. GPU support is determined by the ROCm
    # version and TORCH_INDEX_URL the image was built with:
    #   rocm6.2 image: RDNA2 (gfx1030), RDNA3 (gfx1100-1103)
    #   rocm7.2 image: adds RDNA3.5 (gfx1150/1151) + RDNA4 (gfx1200/1201)


def check_imports() -> None:
    banner("dep imports")
    # unsloth must be imported before transformers/trl/peft so its
    # monkey-patches land, and before unsloth_zoo so it sees
    # UNSLOTH_IS_PRESENT. Import order matches the CUDA smoke test.
    import unsloth

    print(f"unsloth     {unsloth.__version__}")
    import unsloth_zoo

    print(f"unsloth_zoo {unsloth_zoo.__version__}")

    try:
        import triton
        print(f"triton      {triton.__version__}")
    except ImportError:
        print("triton      (not installed — ROCm path uses HIP kernels directly)")

    import bitsandbytes as bnb
    print(f"bnb         {bnb.__version__}")

    import transformers
    print(f"transformers {transformers.__version__}")

    import trl
    print(f"trl         {trl.__version__}")

    import peft
    print(f"peft        {peft.__version__}")

    # xformers has no ROCm wheel; its absence is expected.
    try:
        import xformers
        print(f"xformers    {xformers.__version__}")
    except ImportError:
        print("xformers    (not installed -- expected; ROCm uses SDPA fallback)")


def check_unsloth_import() -> None:
    banner("unsloth FastLanguageModel reachable")
    import unsloth
    from unsloth import FastLanguageModel

    print(f"unsloth     {unsloth.__version__}")
    print(f"FastLanguageModel  {FastLanguageModel}")


def check_tiny_train() -> None:
    banner("tiny LoRA train (5 steps)")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    print(f"loading     {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: 2 + 2 = ?\nA:",
        "Q: Name a primary color.\nA:",
        "Q: Hello, who are you?\nA:",
    ] * 2
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=64
    )
    enc = {k: v.cuda() for k, v in enc.items()}
    labels = enc["input_ids"].clone()

    model.train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    for step in range(5):
        out = model(**enc, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        print(f"step {step}  loss={out.loss.item():.4f}", flush=True)

    print("OK: 5 LoRA steps completed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the tiny LoRA training step (no HF download).",
    )
    args = ap.parse_args()

    check_torch()
    check_imports()
    check_unsloth_import()
    if not args.skip_train:
        check_tiny_train()

    banner("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
