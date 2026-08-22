"""One-shot: materialize a rank-32 LoRA adapter on `unsloth/Qwen3-4B-Base`.

Writes a PEFT-style directory so every backend (vLLM `LoRARequest`,
`peft.PeftModel.from_pretrained`, Unsloth `FastLanguageModel.get_peft_model`)
can load the SAME weights. Random-init is fine for throughput measurement --
the goal is to have LoRA kernels active during generation, not a trained
model.

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/make_lora_adapter.py \
        --output outputs/lora_rank32_fresh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--output", default = "outputs/lora_rank32_fresh")
    p.add_argument("--rank", type = int, default = 32)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)

    # Use vanilla HF -- PEFT's save_pretrained yields the canonical
    # adapter_config.json + adapter_model.safetensors that vLLM's LoRARequest
    # expects. Loading via Unsloth would leak Unsloth-specific LoRA wrappers.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # bf16 base; we only need structure + save. Keep on CPU to avoid a GPU load
    # just for `save_pretrained`.
    print(f"[make_lora_adapter] Loading {args.model_name} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype = torch.bfloat16)

    peft_cfg = LoraConfig(
        r = args.rank,
        lora_alpha = args.rank * 2,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias = "none",
        task_type = "CAUSAL_LM",
        lora_dropout = 0.0,
    )
    peft_model = get_peft_model(model, peft_cfg)
    peft_model.print_trainable_parameters()

    # Ensure both A and B matrices are non-zero. PEFT initializes A with
    # kaiming_uniform and B with zeros -- which makes the adapter a no-op and
    # would mask LoRA kernels on some backends. Seed B with tiny random values.
    n_reinit = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.normal_(mean = 0.0, std = 1e-4)
                n_reinit += 1
    print(
        f"[make_lora_adapter] Reinitialized {n_reinit} lora_B matrices with tiny gaussian."
    )

    peft_model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Sanity: verify safetensors file present and non-trivial.
    from safetensors import safe_open

    st_path = out_dir / "adapter_model.safetensors"
    n_zero_tensors = 0
    n_tensors = 0
    with safe_open(str(st_path), framework = "pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            n_tensors += 1
            if (t == 0).all().item():
                n_zero_tensors += 1
    print(
        f"[make_lora_adapter] Wrote {n_tensors} tensors to {st_path} "
        f"({n_zero_tensors} all-zero)."
    )
    print(f"[make_lora_adapter] Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
