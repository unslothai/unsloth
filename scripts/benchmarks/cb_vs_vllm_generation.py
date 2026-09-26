"""Standalone generation microbenchmark: vLLM vs transformers CB vs Unsloth.

Backends (one per process; all engines are GPU-greedy):
- `vllm`             : Unsloth `fast_inference=True` (vLLM colocated).
- `tpaged`           : `model.generate_batch` on paged HF + `--attn_impl`.
- `unsloth_fi_false` : Unsloth `fast_inference=False` with the custom HF
                       inference kernels (cached fp16 LoRA).

LoRA: pass `--lora_adapter PATH` to activate a PEFT-style rank-32 adapter on
both vLLM (`LoRARequest`) and the HF paths (`peft.PeftModel.from_pretrained`,
or for `unsloth_fi_false` `FastLanguageModel.get_peft_model` pointed at the
same weights).

Equivalence-friendly sampling defaults (`--temperature 0.1 --top_p 0.97
--min_p 0.5 --top_k 5`) keep rollouts comparable across backends for the KL /
reward diff checks done in Phase 2.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/cb_vs_vllm_generation.py \
        --backend vllm --stats_path logs/lora_vllm_gen.json \
        --lora_adapter outputs/lora_rank32_fresh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

# FA4 shim for the `tpaged` backend with paged_attention. No-op for vLLM /
# unsloth_fi_false.
import flash_attn_fa4_shim  # noqa: E402

flash_attn_fa4_shim.apply()


def build_prompts(tokenizer, n_prompts, chat_template = "auto", model_type_name = None):
    from unsloth_grpo_common import (
        apply_chat_template_to_tokenizer,
        SYSTEM_PROMPT,
    )
    from datasets import load_dataset

    if chat_template == "auto":
        use_grpo = (model_type_name or "").startswith("Qwen3")
    elif chat_template == "grpo":
        use_grpo = True
    else:  # "native"
        use_grpo = False
    if use_grpo:
        apply_chat_template_to_tokenizer(tokenizer)
        print("[bench] chat_template: GRPO")
    else:
        print("[bench] chat_template: tokenizer native")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    ds = ds.shuffle(seed = 3407).select(range(n_prompts))
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["prompt"]},
        ]
        for x in ds
    ]
    prompts_text = [
        tokenizer.apply_chat_template(m, add_generation_prompt = True, tokenize = False)
        for m in messages
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(m, add_generation_prompt = True, tokenize = True)
        for m in messages
    ]
    return prompts_text, prompt_ids


def run_vllm(args):
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    from unsloth import FastLanguageModel

    fi_kwargs = dict(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        fast_inference = True,
        max_lora_rank = 32,
        gpu_memory_utilization = args.gpu_memory_utilization,
    )
    if args.enforce_eager:
        # vLLM 0.19 + torch 2.10 hits `RuntimeError: Tried to erase Node
        # size_1 but it still had 2 users` during split_graph. enforce_eager
        # skips vLLM's torch.compile path entirely (still PagedAttention +
        # FlashInfer decode, just no graph capture).
        fi_kwargs["enforce_eager"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(**fi_kwargs)
    prompts_text, prompt_ids = build_prompts(
        tokenizer,
        args.n_prompts,
        chat_template = args.chat_template,
        model_type_name = type(getattr(model, "model", model)).__name__,
    )

    lora_request = None
    if args.lora_adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("fresh", 1, str(Path(args.lora_adapter).resolve()))

    from vllm import SamplingParams

    sp = SamplingParams(
        temperature = args.temperature,
        top_p = args.top_p,
        min_p = args.min_p,
        top_k = args.top_k,
        seed = 3407,
        max_tokens = args.max_new_tokens,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    # Warmup on 16 prompts then discard.
    warmup_text = prompts_text[:16]
    _ = model.fast_generate(warmup_text, sampling_params = sp, lora_request = lora_request)
    torch.cuda.synchronize()

    n_prompt_tokens = sum(len(p) for p in prompt_ids)
    wall_times = []
    total_decoded = None
    last_outputs = None
    for _ in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model.fast_generate(
            prompts_text, sampling_params = sp, lora_request = lora_request
        )
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        total_decoded = sum(len(o.outputs[0].token_ids) for o in outputs)
        last_outputs = outputs

    med = sorted(wall_times)[len(wall_times) // 2]
    sample_texts = (
        [o.outputs[0].text[:200] for o in (last_outputs[:3] or [])]
        if last_outputs
        else []
    )
    return {
        "backend": "vllm",
        "lora_adapter": args.lora_adapter,
        "n_prompts": args.n_prompts,
        "n_prompt_tokens": n_prompt_tokens,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "prompt_tps": n_prompt_tokens / med,
        "decode_tps": (total_decoded or 0) / med,
        "max_new_tokens": args.max_new_tokens,
        "sample_completions": sample_texts,
    }


def run_tpaged(args):
    """Vanilla HF + paged cache.

    Unsloth's Qwen3Attention monkey-patch does not compose with the
    `paged|<impl>` functional attention interface, so we use plain HF.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.load_in_4bit:
        bnb_model_name = args.model_name_4bit or f"{args.model_name}-unsloth-bnb-4bit"
        print(f"[tpaged] loading 4-bit base: {bnb_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            bnb_model_name,
            attn_implementation = args.attn_impl,
            device_map = "cuda:0",
        )
        # HF transformers logs "lm_head.weight newly initialized" for
        # bnb-4bit shards of tied-embedding models. tie_word_embeddings is
        # True in the config but the dequant path leaves lm_head unbound.
        # Tie manually so we don't generate gibberish.
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype = torch.bfloat16,
            attn_implementation = args.attn_impl,
        ).to("cuda")
    model.eval()

    if args.lora_adapter:
        from peft import PeftModel

        # NOTE: no merge_adapter -- we measure LoRA-active inference.
        model = PeftModel.from_pretrained(
            model, str(Path(args.lora_adapter).resolve()), is_trainable = False
        )
        model.eval()

    if args.persistent_cb:
        from persistent_cb import install_for_model  # noqa: WPS433
    prompts_text, prompt_ids = build_prompts(
        tokenizer,
        args.n_prompts,
        chat_template = args.chat_template,
        model_type_name = type(model).__name__,
    )

    gen_config = GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        do_sample = True,
        temperature = args.temperature,
        top_p = args.top_p,
        min_p = args.min_p,
        top_k = args.top_k,
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        use_cache = True,
    )
    gen_config.max_batch_tokens = args.max_batch_tokens
    gen_config.num_blocks = args.num_blocks

    if args.persistent_cb:
        install_for_model(model, gen_config)

    warmup_ids = prompt_ids[:16]
    with torch.inference_mode():
        _ = model.generate_batch(
            warmup_ids, generation_config = gen_config, progress_bar = False
        )
    torch.cuda.synchronize()

    n_prompt_tokens = sum(len(p) for p in prompt_ids)
    wall_times = []
    total_decoded = None
    last_outputs = None
    for _ in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate_batch(
                prompt_ids, generation_config = gen_config, progress_bar = False
            )
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        total_decoded = sum(len(v.generated_tokens) for v in outputs.values())
        last_outputs = outputs

    # Sample completions for coherence sanity check.
    sample_texts = []
    if last_outputs is not None:
        for k in list(last_outputs.keys())[:3]:
            toks = last_outputs[k].generated_tokens
            sample_texts.append(tokenizer.decode(toks, skip_special_tokens = False)[:200])

    med = sorted(wall_times)[len(wall_times) // 2]
    return {
        "backend": "tpaged",
        "lora_adapter": args.lora_adapter,
        "attn_impl": args.attn_impl,
        "persistent_cb": args.persistent_cb,
        "n_prompts": args.n_prompts,
        "n_prompt_tokens": n_prompt_tokens,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "prompt_tps": n_prompt_tokens / med,
        "decode_tps": (total_decoded or 0) / med,
        "max_new_tokens": args.max_new_tokens,
        "sample_completions": sample_texts,
    }


def run_unsloth_fi_false(args):
    """Unsloth `fast_inference=False` path with custom HF inference kernels.

    This is the path that backs regular Unsloth training's sampling loop
    (Triton RMSNorm/RoPE, cached fp16 LoRA copies in `fast_linear_forward`).
    Previously only exercised through full GRPO runs -- isolating it lets us
    compare it head-to-head against vLLM on the same workload.

    LoRA is attached via `FastLanguageModel.get_peft_model`; if a PEFT adapter
    path is provided we re-load its weights into the Unsloth-wrapped model so
    every backend uses the *same* weights.
    """
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
        fast_inference = False,
        max_lora_rank = 32,
    )
    # Attach LoRA rank 32 the same way the GRPO notebook does.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = 64,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Optional: overlay a shared adapter so weights match other backends.
    if args.lora_adapter:
        from safetensors import safe_open

        adapter_file = Path(args.lora_adapter).resolve() / "adapter_model.safetensors"
        loaded_tensors = {}
        with safe_open(str(adapter_file), framework = "pt") as f:
            for key in f.keys():
                loaded_tensors[key] = f.get_tensor(key)

        # Both PEFT and Unsloth's `get_peft_model` produce parameter names with
        # `base_model.model.` prefix plus `.lora_{A,B}.default.weight`. Build a
        # normalized (core-path) -> param map, then match by core path only.
        def _core(name: str) -> str:
            n = name
            for pref in ("base_model.model.", "model."):
                if n.startswith(pref):
                    n = n[len(pref) :]
            n = n.replace(".lora_A.default.", ".lora_A.").replace(
                ".lora_B.default.", ".lora_B."
            )
            return n

        own_by_core = {}
        for n, p in model.named_parameters():
            if "lora_" in n:
                own_by_core.setdefault(_core(n), []).append(p)
        matched = 0
        with torch.no_grad():
            for name, tensor in loaded_tensors.items():
                core = _core(name)
                for own in own_by_core.get(core, []):
                    if own.shape == tensor.shape:
                        own.data.copy_(tensor.to(own.device, own.dtype))
                        matched += 1
                        break
        print(
            f"[unsloth_fi_false] LoRA weight sync matched {matched} tensors "
            f"(out of {len(loaded_tensors)} adapter entries)."
        )

    FastLanguageModel.for_inference(model)

    prompts_text, prompt_ids = build_prompts(
        tokenizer,
        args.n_prompts,
        chat_template = args.chat_template,
        model_type_name = type(model).__name__,
    )

    # `model.generate` accepts batched input_ids; pad to max length.
    from transformers import GenerationConfig

    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"  # decoder needs left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        do_sample = True,
        temperature = args.temperature,
        top_p = args.top_p,
        min_p = args.min_p,
        top_k = args.top_k,
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        use_cache = True,
    )

    def _batched_generate(texts):
        batch = tokenizer(texts, return_tensors = "pt", padding = True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**batch, generation_config = gen_config)
        prompt_len = batch["input_ids"].shape[1]
        return out, prompt_len

    # Warmup on 16 prompts.
    _ = _batched_generate(prompts_text[:16])
    torch.cuda.synchronize()

    n_prompt_tokens = sum(len(p) for p in prompt_ids)
    wall_times = []
    total_decoded = None
    last_out_ids = None
    last_prompt_len = None
    for _ in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids, prompt_len = _batched_generate(prompts_text)
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        # Count generated tokens past prompt_len per sequence (subtract any
        # trailing pad-only tail by comparing against EOS).
        total_decoded = int(
            (out_ids[:, prompt_len:] != tokenizer.pad_token_id).sum().item()
        )
        last_out_ids = out_ids
        last_prompt_len = prompt_len

    med = sorted(wall_times)[len(wall_times) // 2]
    sample_texts = []
    if last_out_ids is not None:
        for i in range(min(3, last_out_ids.shape[0])):
            sample_texts.append(
                tokenizer.decode(
                    last_out_ids[i, last_prompt_len:], skip_special_tokens = False
                )[:200]
            )

    return {
        "backend": "unsloth_fi_false",
        "lora_adapter": args.lora_adapter,
        "n_prompts": args.n_prompts,
        "n_prompt_tokens": n_prompt_tokens,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "prompt_tps": n_prompt_tokens / med,
        "decode_tps": (total_decoded or 0) / med,
        "max_new_tokens": args.max_new_tokens,
        "sample_completions": sample_texts,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend", choices = ["vllm", "tpaged", "unsloth_fi_false"], required = True
    )
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--n_prompts", type = int, default = 32)
    p.add_argument("--n_rounds", type = int, default = 2)
    p.add_argument("--max_new_tokens", type = int, default = 512)
    p.add_argument("--gpu_memory_utilization", type = float, default = 0.8)
    p.add_argument("--attn_impl", default = "sdpa")
    p.add_argument("--max_batch_tokens", type = int, default = 8192)
    p.add_argument("--num_blocks", type = int, default = 16384)
    p.add_argument("--persistent_cb", action = "store_true")
    p.add_argument(
        "--lora_adapter",
        default = None,
        help = "Path to a PEFT adapter (rank 32) applied in every backend.",
    )
    p.add_argument(
        "--load_in_4bit",
        action = "store_true",
        help = "Load base as bitsandbytes 4-bit (Unsloth shard).",
    )
    p.add_argument(
        "--model_name_4bit",
        default = None,
        help = "Override 4-bit shard name. Default `{model_name}-unsloth-bnb-4bit`.",
    )
    p.add_argument("--temperature", type = float, default = 0.1)
    p.add_argument("--top_p", type = float, default = 0.97)
    p.add_argument("--min_p", type = float, default = 0.5)
    p.add_argument("--top_k", type = int, default = 5)
    p.add_argument("--stats_path", required = True)
    p.add_argument(
        "--chat_template",
        choices = ["auto", "grpo", "native"],
        default = "auto",
        help = (
            "`auto`: GRPO for Qwen3, tokenizer native otherwise. "
            "`grpo`: force GRPO template. `native`: force tokenizer's "
            "built-in Instruct template (Llama-3.2-Instruct)."
        ),
    )
    p.add_argument(
        "--enforce_eager",
        action = "store_true",
        help = (
            "vLLM only: skip the torch.compile + cudagraph path and run "
            "eager. Useful when vLLM's compile regresses on the local "
            "torch build."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok = True)

    torch.cuda.reset_peak_memory_stats()
    if args.backend == "vllm":
        out = run_vllm(args)
    elif args.backend == "unsloth_fi_false":
        out = run_unsloth_fi_false(args)
    else:
        out = run_tpaged(args)

    out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    out["sampling"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
    }
    with open(args.stats_path, "w") as f:
        json.dump(out, f, indent = 2)
    print(json.dumps(out, indent = 2))
    os._exit(0)


if __name__ == "__main__":
    main()
