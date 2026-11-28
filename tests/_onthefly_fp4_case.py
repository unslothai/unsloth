# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One LoRA case per process for load_in_4bit = "nvfp4" / "mxfp4" (and bf16 / nf4 references).

python tests/_onthefly_fp4_case.py --build DIR [--moe]      write a tiny 16-bit Qwen3 checkpoint
python tests/_onthefly_fp4_case.py --case MODEL API MODE OUT_DIR [--steps N] [--merge]
    prints `ONTHEFLY_CASE {json}`; MODE is bf16, nf4, nvfp4 or mxfp4.
"""

import argparse
import json
import os
import sys

import torch

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(
    out_dir,
    moe = False,
    seed = 3407,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    repo = TINY_MOE if moe else TINY
    config = AutoConfig.from_pretrained(repo)
    sizes = dict(
        hidden_size = 64,
        intermediate_size = 128,
        head_dim = 16,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_hidden_layers = 2,
    )
    if moe:
        sizes.update(moe_intermediate_size = 64)
    config.update(sizes)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype = torch.bfloat16)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(repo).save_pretrained(out_dir)
    return out_dir


def _census(model):
    out = {}
    for _, m in model.named_modules():
        key = type(m).__name__
        for name in ("gate_up_proj", "down_proj"):
            p = m._parameters.get(name) if hasattr(m, "_parameters") else None
            if isinstance(p, torch.Tensor) and p.dim() == 3:
                key = f"{key}/{type(p).__name__}"
                break
        if isinstance(m, torch.nn.Linear) or "/" in key:
            out[key] = out.get(key, 0) + 1
    return out


def run_case(
    model_name,
    api,
    mode,
    out_dir,
    steps = 5,
    merge = False,
    seq = 64,
    batch = 2,
):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    seq = int(os.environ.get("ONTHEFLY_CASE_SEQ", seq))
    import unsloth
    import unsloth.kernels.nvfp4 as nv
    from unsloth.models.onthefly_quant import onthefly_fp4_stats

    calls = {"fwd": 0, "bwd": 0}
    fwd, bwd = nv.NVFP4Linear_matmul.forward, nv.NVFP4Linear_matmul.backward

    def counted_fwd(ctx, *args, **kwargs):
        calls["fwd"] += 1
        return fwd(ctx, *args, **kwargs)

    def counted_bwd(ctx, *args, **kwargs):
        calls["bwd"] += 1
        return bwd(ctx, *args, **kwargs)

    nv.NVFP4Linear_matmul.forward = staticmethod(counted_fwd)
    nv.NVFP4Linear_matmul.backward = staticmethod(counted_bwd)
    # Runtime counters (compiled LoRA forwards trace the autograd Function, so its Python body is not one):
    # every dense dequantize of a packed weight, fused or not, goes through _nvfp4_dequantize_triton on CUDA.
    dequant = {"n": 0}
    real_dequant = nv._nvfp4_dequantize_triton

    def counted_dequant(*args, **kwargs):
        dequant["n"] += 1
        return real_dequant(*args, **kwargs)

    nv._nvfp4_dequantize_triton = counted_dequant

    Fast = getattr(unsloth, api)
    flag = {"bf16": False, "nf4": True}.get(mode, mode)
    onthefly_fp4_stats(reset = True)
    model, tokenizer = Fast.from_pretrained(
        model_name, max_seq_length = seq, load_in_4bit = flag, dtype = torch.bfloat16
    )
    stats = onthefly_fp4_stats()
    census_base = _census(model)
    lm_head = model.get_output_embeddings()
    base_save = None
    if merge:
        # A full save of the quantized base must reload as the model it is (dequantized 16-bit weights).
        from transformers import AutoModelForCausalLM

        probe = torch.randint(0, 1000, (1, 16), generator = torch.Generator().manual_seed(1)).cuda()
        with torch.no_grad():
            want0 = model(input_ids = probe).logits.float()
        base_dir = os.path.join(out_dir, f"base_{mode}_{api}")
        model.save_pretrained(base_dir)
        again = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype = torch.bfloat16, device_map = "cuda"
        )
        with torch.no_grad():
            got0 = again(input_ids = probe).logits.float()
        base_save = float((got0 - want0).norm() / want0.norm())
        del again
    model = Fast.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        target_modules = TARGETS,
        use_gradient_checkpointing = "unsloth",
    )
    packed = [
        (n, m.weight_packed.clone())
        for n, m in model.named_modules()
        if "weight_packed" in getattr(m, "_parameters", {})
    ]
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n}
    b0 = {n: p.detach().clone() for n, p in lora.items()}
    vocab = min(int(model.config.get_text_config().vocab_size), 4096)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (batch, seq), generator = g).cuda()
    if os.environ.get("ONTHEFLY_CASE_TEXT"):
        # Real text for real checkpoints: the loss and LoRA grads then mean something.
        words = open(os.environ["ONTHEFLY_CASE_TEXT"], encoding = "utf-8").read()
        ids = (
            tokenizer(words, return_tensors = "pt")
            .input_ids[0][: batch * seq]
            .view(batch, seq)
            .cuda()
        )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    model.train()
    losses, grad_norms, first_grads = [], [], None
    fused0 = dict(nv._FUSED_CALLS)
    try:
        from unsloth_zoo.packed_experts import PACKED_EXPERT_STATS
    except Exception:
        PACKED_EXPERT_STATS = {}
    experts0 = dict(PACKED_EXPERT_STATS)
    mx = {"n": 0}
    try:
        # MXFP4 expert stacks run either the fused decode-in-GEMM kernel or a dequantize + grouped GEMM.
        import unsloth_zoo.mxfp4_gemm as mxg
        from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam

        real_fused, real_deq = mxg.mxfp4_expert_grouped_mm, Mxfp4ExpertParam.dequantize

        def counted_fused(*args, **kwargs):
            mx["n"] += 1
            return real_fused(*args, **kwargs)

        def counted_deq(self, *args, **kwargs):
            mx["n"] += 1
            return real_deq(self, *args, **kwargs)

        mxg.mxfp4_expert_grouped_mm = counted_fused
        Mxfp4ExpertParam.dequantize = counted_deq
    except Exception:
        pass
    for step in range(steps):
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in lora.items() if p.grad is not None}
        if first_grads is None:
            first_grads = grads
        last_grads = grads
        grad_norms.append(sum(float(v.norm() ** 2) for v in grads.values()) ** 0.5)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss.item()))
    torch.save({k.replace(".default", ""): v.cpu() for k, v in first_grads.items()},
               os.path.join(out_dir, f"grads_{mode}_{api}.pt"))  # fmt: skip
    report = {
        "mode": mode,
        "api": api,
        "unsloth": unsloth.__file__,
        "stats": stats,
        "census": census_base,
        "lm_head": f"{type(lm_head).__name__}:{lm_head.weight.dtype}",
        "embed": str(model.get_input_embeddings().weight.dtype),
        "losses": losses,
        "grad_norms": grad_norms,
        "lora_grads_nonzero": sum(1 for v in last_grads.values() if v.abs().sum() > 0),
        "lora_params": len(lora),
        "lora_changed": sum(1 for n, p in lora.items() if not torch.equal(p.detach(), b0[n])),
        "packed_layers": len(packed),
        "packed_unchanged": all(
            torch.equal(dict(model.named_modules())[n].weight_packed, w) for n, w in packed
        ),
        "nvfp4_fwd": calls["fwd"],
        "nvfp4_bwd": calls["bwd"],
        "dequant_calls": dequant["n"],
        "fused_fwd": nv._FUSED_CALLS["fwd"] - fused0["fwd"],
        "fused_dx": nv._FUSED_CALLS["bwd"] - fused0["bwd"],
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "packed_expert_calls": PACKED_EXPERT_STATS.get("calls", 0) - experts0.get("calls", 0),
        "packed_expert_chunks": PACKED_EXPERT_STATS.get("chunks", 0) - experts0.get("chunks", 0),
        "mxfp4_expert_calls": mx["n"],
        "base_save_reload_rel": base_save,
    }
    model.eval()
    with torch.no_grad():
        gen = model.generate(input_ids = ids[:1, :8], max_new_tokens = 4, do_sample = False)
    report["generate_new_tokens"] = int(gen.shape[1] - 8)
    model.train()
    model.save_pretrained(os.path.join(out_dir, f"lora_{mode}_{api}"))
    report["saved"] = sorted(os.listdir(os.path.join(out_dir, f"lora_{mode}_{api}")))
    if merge:
        model.eval()
        with torch.no_grad():
            want = model(input_ids = ids).logits.float()
        merged_dir = os.path.join(out_dir, f"merged_{mode}_{api}")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_16bit")
        from transformers import AutoModelForCausalLM

        reloaded = AutoModelForCausalLM.from_pretrained(
            merged_dir, dtype = torch.bfloat16, device_map = "cuda"
        )
        with torch.no_grad():
            got = reloaded(input_ids = ids).logits.float()
        report["merged_files"] = sorted(os.listdir(merged_dir))
        report["merged_max_abs"] = float((got - want).abs().max())
        report["merged_rel"] = float((got - want).norm() / want.norm())
        # The merge reads the checkpoint shards: 16-bit source weights + the delta, as for NF4.
        from peft import PeftModel

        base16 = AutoModelForCausalLM.from_pretrained(
            model_name, dtype = torch.bfloat16, device_map = "cuda"
        )
        base16 = PeftModel.from_pretrained(
            base16, os.path.join(out_dir, f"lora_{mode}_{api}"), torch_device = "cuda"
        )
        with torch.no_grad():
            ref16 = base16(input_ids = ids).logits.float()
        report["merged_vs_16bit_base_rel"] = float((got - ref16).norm() / ref16.norm())
        # PEFT's in-memory merge decodes each packed base layer and folds the delta into it.
        unloaded = model.merge_and_unload()
        with torch.no_grad():
            folded = unloaded(input_ids = ids).logits.float()
        report["peft_merge_rel"] = float((folded - want).norm() / want.norm())
        report["peft_merge_packed_left"] = sum(
            1 for m in unloaded.modules() if "weight_packed" in getattr(m, "_parameters", {})
        )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build")
    ap.add_argument("--moe", action = "store_true")
    ap.add_argument("--case", nargs = 4, metavar = ("MODEL", "API", "MODE", "OUT_DIR"))
    ap.add_argument("--steps", type = int, default = 5)
    ap.add_argument("--merge", action = "store_true")
    a = ap.parse_args()
    if a.build:
        build(a.build, moe = a.moe)
        sys.exit(0)
    r = run_case(*a.case, steps = a.steps, merge = a.merge)
    print("ONTHEFLY_CASE " + json.dumps(r))
