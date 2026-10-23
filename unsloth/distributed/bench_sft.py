"""One epoch of Qwen3.5 LoRA SFT at a fixed sequence length.

Modes: vanilla (1 GPU), fsdp, cp, fsdp_cp.
vanilla is Unsloth's own load + LoRA + gradient checkpointing.
The other three call the same loader, then the parallel helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "fsdp", "cp", "fsdp_cp"], required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/BIV/merge/output/cache/Qwen--Qwen3.5-35B-A3B")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--out", default="/root/autodl-tmp/unsloth-bench/results.jsonl")
    return parser.parse_args()


def rank_of():
    return int(os.environ.get("RANK", "0"))


def log(msg):
    if rank_of() == 0:
        print(msg, flush=True)


def load_unsloth(args, device_map):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map=device_map,
        trust_remote_code=any(
            os.path.exists(os.path.join(args.model, f)) for f in ("modeling_kimi.py", "modeling_kimi_k3.py")
        ),
        use_gradient_checkpointing="unsloth" if args.mode == "vanilla" else False,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            t for t in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "kv_proj", "o_b_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "qkv_proj", "output_gate",
                "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj",
            ]
            if any(n.endswith("." + t) for n, _ in model.named_modules())
        ],
        bias="none",
        use_gradient_checkpointing="unsloth" if args.mode == "vanilla" else False,
        random_state=3407,
    )
    return model, tokenizer


def make_batch(tokenizer, seq_len: int, sample_id: int, device, vocab_size: int = 32000):
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer) if tokenizer is not None else None
    if text_tokenizer is not None and hasattr(text_tokenizer, "encode"):
        seed = f"### Instruction:\nReply with a short sentence {sample_id}.\n### Response:\n"
        piece = seed + ("The file is still there. " * 64)
        ids = torch.tensor(text_tokenizer.encode(piece, add_special_tokens=True), dtype=torch.long)
        if ids.numel() >= seq_len:
            ids = ids[:seq_len]
        else:
            reps = (seq_len + ids.numel() - 1) // ids.numel()
            ids = ids.repeat(reps)[:seq_len]
        if vocab_size and int(ids.max()) >= vocab_size:
            ids = ids % max(vocab_size - 1, 1)
    else:
        g = torch.Generator()
        g.manual_seed(3407 + sample_id)
        ids = torch.randint(10, min(2000, max(vocab_size - 1, 11)), (seq_len,), generator=g)
    input_ids = ids.unsqueeze(0).to(device)
    labels = input_ids.clone()
    return input_ids, labels


def peak_gib():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def write_result(path, row):
    if rank_of() != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    use_fsdp = args.mode in ("fsdp", "fsdp_cp")
    use_cp = args.mode in ("cp", "fsdp_cp")
    row = {
        "mode": args.mode,
        "seq_len": args.seq_len,
        "samples": args.samples,
        "status": "start",
    }
    try:
        from unsloth.distributed.parallel import (
            apply_cp,
            apply_fsdp,
            apply_fsdp_checkpoint,
            build_meshes,
            local_rank,
            sft_loss,
            sync_replicated_grads,
            world_size,
        )

        local = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local}
        if use_fsdp or use_cp:
            device, fsdp_mesh, cp_group = build_meshes(use_fsdp, use_cp)
        else:
            device = torch.device("cuda", 0)
            fsdp_mesh, cp_group = None, None
            torch.cuda.set_device(device)

        log(f"[bench] load mode={args.mode} device_map={device_map}")
        t_load = time.perf_counter()
        model, tokenizer = load_unsloth(args, device_map)
        model.config.use_cache = False
        if use_cp:
            n_gdn = apply_cp(model, cp_group)
            log(f"[bench] gdn layers patched={n_gdn}")
        if use_fsdp:
            apply_fsdp(model, fsdp_mesh)
            n_ckpt = apply_fsdp_checkpoint(model)
            log(f"[bench] fsdp checkpoint layers={n_ckpt}")
        elif use_cp and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        load_sec = time.perf_counter() - t_load
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        held = torch.cuda.memory_allocated() / (1024 ** 3)
        log(f"[bench] loaded in {load_sec:.1f}s trainable={trainable} held={held:.2f}GiB")

        params = [p for p in model.parameters() if p.requires_grad]
        try:
            opt = torch.optim.AdamW(params, lr=args.lr, fused=True)
        except (TypeError, ValueError, RuntimeError):
            opt = torch.optim.AdamW(params, lr=args.lr, fused=False)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        step_times = []
        epoch_start = time.perf_counter()
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None) or getattr(
            getattr(getattr(model, "config", None), "text_config", None), "vocab_size", 32000
        )
        for step in range(args.samples):
            step_start = time.perf_counter()
            input_ids, labels = make_batch(tokenizer, args.seq_len, step, device, vocab_size=vocab_size)
            opt.zero_grad(set_to_none=True)
            loss = sft_loss(model, input_ids, labels, cp_group)
            loss.backward()
            if use_cp:
                sync_replicated_grads(model, cp_group)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            torch.cuda.synchronize()
            step_sec = time.perf_counter() - step_start
            step_times.append(step_sec)
            log(f"[bench] step {step} loss={loss.item():.4f} sec={step_sec:.2f} peak={peak_gib():.2f}GiB")
        epoch_sec = time.perf_counter() - epoch_start
        row.update(
            {
                "status": "ok",
                "load_sec": round(load_sec, 2),
                "epoch_sec": round(epoch_sec, 2),
                "step_sec": [round(x, 2) for x in step_times],
                "peak_gib": round(peak_gib(), 2),
                "trainable": trainable,
                "world": world_size() if (use_fsdp or use_cp) else 1,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"status": "oom", "error": str(exc).splitlines()[-1][:500], "peak_gib": round(peak_gib(), 2)})
        log(f"[bench] OOM {row}")
    except Exception as exc:
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:800]})
        traceback.print_exc()
    write_result(args.out, row)
    log("[bench] " + json.dumps(row))
    if row["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
