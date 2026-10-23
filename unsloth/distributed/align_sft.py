"""One tiny step, same seed, compare loss / hidden / LoRA grads to vanilla.

Checkpointing is off. Dropout is off. The token ids are drawn from a
CPU generator seeded with 3407, so every mode sees the same batch.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from unsloth.distributed.parallel import (
    apply_cp,
    apply_fsdp,
    build_meshes,
    sft_loss,
    sync_replicated_grads,
    world_size,
)


def rank_of():
    return int(os.environ.get("RANK", "0"))


def log(msg):
    if rank_of() == 0:
        print(msg, flush=True)


def load_model(model_dir, seq_len, device_map, lora_r, dtype=torch.bfloat16):
    torch.manual_seed(3407)
    from unsloth import FastLanguageModel

    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=seq_len,
        dtype=dtype,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=dtype == torch.bfloat16,
        full_finetuning=False,
        device_map=device_map,
        trust_remote_code=any(
            os.path.exists(os.path.join(model_dir, f)) for f in ("modeling_kimi.py", "modeling_kimi_k3.py")
        ),
        use_gradient_checkpointing=False,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.0,
        target_modules=[
            t for t in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "kv_proj", "o_b_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "qkv_proj", "output_gate",
                "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "g_proj", "b_proj",
            ]
            if any(n.endswith("." + t) for n, _ in model.named_modules())
        ],
        bias="none",
        use_gradient_checkpointing=False,
        random_state=3407,
    )
    model.config.use_cache = False
    model.train()
    return model


def fixed_batch(seq_len, device):
    generator = torch.Generator()
    generator.manual_seed(3407)
    # Stay in the common token range. The ids are the whole supervision target.
    ids = torch.randint(10, 2000, (1, seq_len), generator=generator)
    ids = ids.to(device)
    return ids, ids.clone()


def _materialize(tensor):
    if hasattr(tensor, "full_tensor"):
        return tensor.full_tensor()
    return tensor


def lora_weight_sum(model) -> float:
    total = None
    for name, param in model.named_parameters():
        if "lora_A" not in name:
            continue
        weight = _materialize(param.detach())
        piece = weight.float().sum()
        total = piece if total is None else total + piece
    return float(total.detach())


def flat_lora_grad(model) -> torch.Tensor:
    pieces = []
    for name, param in model.named_parameters():
        if not param.requires_grad or "lora_" not in name or param.grad is None:
            continue
        grad = _materialize(param.grad.detach())
        if rank_of() == 0:
            pieces.append(grad.float().reshape(-1).cpu())
    if rank_of() != 0:
        return torch.empty(0)
    return torch.cat(pieces)


def compare_dir(out_dir: str):
    import torch.nn.functional as F

    ref = torch.load(os.path.join(out_dir, "vanilla.pt"), map_location="cpu")
    g0 = ref["grad"].float()
    print(
        f"vanilla : loss={ref['loss']:.8f} lora_wsum={ref['lora_weight_sum']:.6f} "
        f"grad_n={g0.numel()} grad_norm={g0.norm().item():.6f}"
    )
    for m in ("fsdp", "cp", "fsdp_cp"):
        p = os.path.join(out_dir, f"{m}.pt")
        if not os.path.exists(p):
            continue
        cur = torch.load(p, map_location="cpu")
        gm = cur["grad"].float()
        cos = F.cosine_similarity(g0.unsqueeze(0), gm.unsqueeze(0)).item()
        max_diff = (g0 - gm).abs().max().item()
        dloss = abs(ref["loss"] - cur["loss"])
        print(
            f"{m:8s}: loss={cur['loss']:.8f} dloss={dloss:.3e} "
            f"cos={cos:.6f} max_diff={max_diff:.3e} grad_norm={gm.norm().item():.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "fsdp", "cp", "fsdp_cp", "compare"], required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/BIV/merge/output/cache/Qwen--Qwen3.5-35B-A3B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    # fp32 separates CP bugs from bf16 rounding that deep models amplify differently per kernel.
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/unsloth-bench/align")
    args = parser.parse_args()

    if args.mode == "compare":
        compare_dir(args.out_dir)
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    use_fsdp = args.mode in ("fsdp", "fsdp_cp")
    use_cp = args.mode in ("cp", "fsdp_cp")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if use_fsdp or use_cp:
        device, fsdp_mesh, cp_group = build_meshes(use_fsdp, use_cp)
    else:
        device = torch.device("cuda", 0)
        fsdp_mesh, cp_group = None, None
        torch.cuda.set_device(device)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = load_model(args.model, args.seq_len, {"": local}, args.lora_r, dtype)
    if use_cp:
        n_gdn = apply_cp(model, cp_group)
        log(f"[align] gdn patched={n_gdn}")
    if use_fsdp:
        apply_fsdp(model, fsdp_mesh)
    weight_sum = lora_weight_sum(model)
    input_ids, labels = fixed_batch(args.seq_len, device)
    loss = sft_loss(model, input_ids, labels, cp_group)
    unscaled = float(loss.detach())
    loss.backward()
    if use_cp:
        sync_replicated_grads(model, cp_group)
    grads = flat_lora_grad(model)
    if rank_of() == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"{args.mode}.pt")
        torch.save(
            {
                "mode": args.mode,
                "seq_len": args.seq_len,
                "loss": unscaled,
                "lora_weight_sum": weight_sum,
                "grad": grads,
            },
            path,
        )
        log(
            f"[align] mode={args.mode} loss={unscaled:.8f} "
            f"lora_weight_sum={weight_sum:.6f} grad_n={grads.numel()} wrote {path}"
        )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
