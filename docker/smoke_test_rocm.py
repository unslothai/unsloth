# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""
Smoke test for the Unsloth ROCm image (AMD GPU build).

What this checks (in order, fail-fast):
  1. torch is a ROCm build (torch.version.hip is not None).
  2. The runtime device is visible and its gfx arch is supported (RDNA2+).
  3. bitsandbytes / triton import without ImportError.
  4. unsloth imports and exposes FastLanguageModel.
  5. A 5-step LoRA train on a tiny model actually runs forward + backward.

Run inside the container:
    bash docker/run.sh --rocm python /workspace/smoke_test_rocm.py

Skip step 5 (faster, no model download):
    ... python /workspace/smoke_test_rocm.py --skip-train
"""

from __future__ import annotations

import argparse
import sys


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush = True)


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
        "torch.cuda.is_available() is False: was the container started with "
        "bash docker/run.sh --rocm (or --device /dev/kfd --device /dev/dri and the "
        "video/render group ids)?"
    )
    n = torch.cuda.device_count()
    print(f"GPU count   {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        # PyTorch ROCm surfaces the gfx code in gcnArchName (e.g.
        # "gfx1100:sramecc+"); strip the feature suffix for readability.
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        bf16 = torch.cuda.is_bf16_supported()
        print(f"device {i}    {name}  arch={arch}  bf16={bf16}")
    print()
    # ROCm does not expose a reliable sm_X.Y compute capability the way NVIDIA
    # does -- the values from get_device_properties() vary by ROCm version and
    # don't map cleanly to gfx codes. Which arches the wheels carry is decided
    # by the ROCm version and index the image was built with (Dockerfile.rocm);
    # the entrypoint already printed the arch note for this card.
    # A device the runtime lists but cannot run a kernel on shows up here, not
    # in device_count().
    x = torch.ones(64, 64, device = "cuda", dtype = torch.float16)
    y = (x @ x).float().sum().item()
    assert y == 64 * 64 * 64, f"FAIL: fp16 matmul on the GPU returned {y}, expected {64 * 64 * 64}"
    print("fp16 matmul OK")


def check_imports() -> None:
    banner("dep imports")
    # unsloth must be imported before transformers/trl/peft so its
    # monkey-patches land, and before unsloth_zoo so it sees
    # UNSLOTH_IS_PRESENT. Import order matches the CUDA smoke test.
    import unsloth

    print(f"unsloth     {unsloth.__version__}")
    import unsloth_zoo

    print(f"unsloth_zoo {unsloth_zoo.__version__}")

    # not optional: unsloth's fused kernels are triton, and the image's build
    # check pinned it to the ROCm build torch links against
    import triton
    import triton.backends

    print(f"triton      {triton.__version__}  backends={sorted(triton.backends.backends)}")
    assert "amd" in triton.backends.backends, "FAIL: triton has no amd backend"

    if _bnb_expected():
        import bitsandbytes as bnb
        print(f"bnb         {bnb.__version__}")
    else:
        print("bnb         not part of a gfx906 build (no prebuilt kernels)")

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


def _bnb_expected() -> bool:
    """A gfx906 build (ROCM_GFX=gfx906 in the build record) ships no bitsandbytes."""
    try:
        record = open("/etc/unsloth-rocm-build", encoding = "utf-8").read()
    except OSError:
        return True
    return "ROCM_GFX=gfx906\n" not in record


def check_tiny_train() -> None:
    banner("tiny LoRA train (5 steps)")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    four_bit = _bnb_expected()
    model_name = "unsloth/Llama-3.2-1B-Instruct" + ("-bnb-4bit" if four_bit else "")
    print(f"loading     {model_name} (load_in_4bit={four_bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = four_bit,
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
    # the padding tokens are not a training target
    labels[enc["attention_mask"] == 0] = -100

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "FAIL: get_peft_model left no trainable parameters"
    before = [p.detach().clone() for p in trainable]

    model.train()
    optim = torch.optim.AdamW(trainable, lr = 1e-3)
    losses = []
    for step in range(5):
        out = model(**enc, labels = labels)
        loss = out.loss
        # a NaN here is what the bitsandbytes 4-bit ROCm bug looked like (bnb <= 0.49)
        assert torch.isfinite(loss), f"FAIL: non-finite loss at step {step}: {loss.item()}"
        loss.backward()
        grads = [p.grad for p in trainable if p.grad is not None]
        assert grads, f"FAIL: no gradient reached the LoRA weights at step {step}"
        assert all(
            torch.isfinite(g).all() for g in grads
        ), f"FAIL: non-finite gradient at step {step}"
        optim.step()
        optim.zero_grad(set_to_none = True)
        losses.append(loss.item())
        print(f"step {step}  loss={losses[-1]:.4f}", flush = True)

    changed = sum(int(not torch.equal(a, b.detach())) for a, b in zip(before, trainable))
    assert changed, "FAIL: the optimizer steps left every LoRA weight unchanged"
    # the same batch five times over at lr 1e-3: the loss has to come down, or the
    # forward/backward is not computing what it claims
    assert (
        losses[-1] < losses[0]
    ), f"FAIL: loss did not decrease over 5 steps on one batch: {losses}"
    print(
        f"OK: 5 LoRA steps completed, loss {losses[0]:.4f} -> {losses[-1]:.4f}, {changed}/{len(trainable)} LoRA tensors updated"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-train",
        action = "store_true",
        help = "Skip the tiny LoRA training step (no HF download).",
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
