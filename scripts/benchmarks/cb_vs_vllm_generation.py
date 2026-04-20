"""Standalone generation microbenchmark: vLLM vs transformers continuous batching.

Measures prompt-tokens/s, decode tokens/s, and end-to-end wall-clock on
`N` prompts sampled from DAPO-Math-17k with the GRPO chat template applied.

Run:
    CUDA_VISIBLE_DEVICES=2 python scripts/cb_vs_vllm_generation.py \
        --backend vllm   --stats_path logs/vllm_gen.json
    CUDA_VISIBLE_DEVICES=2 python scripts/cb_vs_vllm_generation.py \
        --backend tpaged --stats_path logs/cb_gen.json

One backend per process (both engines are GPU-greedy). Results are then
combined offline.
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

# Install the FA4 shim so transformers' continuous batching dispatches to
# Blackwell (sm_100) kernels when `--attn_impl flash_attention_2` is selected.
# No-op for the vLLM backend since vLLM doesn't go through transformers'
# attention interface.
import flash_attn_fa4_shim  # noqa: E402
flash_attn_fa4_shim.apply()


def build_prompts(tokenizer, n_prompts):
    from unsloth_grpo_common import (
        apply_chat_template_to_tokenizer,
        SYSTEM_PROMPT,
    )
    from datasets import load_dataset

    apply_chat_template_to_tokenizer(tokenizer)
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    ds = ds.shuffle(seed=3407).select(range(n_prompts))
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["prompt"]},
        ]
        for x in ds
    ]
    prompts_text = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages
    ]
    prompt_ids = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
        for m in messages
    ]
    return prompts_text, prompt_ids


def run_vllm(args):
    import os as _os
    _os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prompts_text, prompt_ids = build_prompts(tokenizer, args.n_prompts)

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=1.0, min_p=0.1, top_p=1.0, top_k=-1,
        seed=3407,
        max_tokens=args.max_new_tokens,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # Warmup on 16 prompts then discard.
    warmup_text = prompts_text[:16]
    _ = model.fast_generate(warmup_text, sampling_params=sp, lora_request=None)
    torch.cuda.synchronize()

    # Three measured rounds on the full batch.
    n_prompt_tokens = sum(len(p) for p in prompt_ids)
    wall_times = []
    total_decoded = None
    for _ in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model.fast_generate(prompts_text, sampling_params=sp, lora_request=None)
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        decoded = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_decoded = decoded

    med = sorted(wall_times)[len(wall_times) // 2]
    return {
        "backend": "vllm",
        "n_prompts": args.n_prompts,
        "n_prompt_tokens": n_prompt_tokens,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "prompt_tps": n_prompt_tokens / med,
        "decode_tps": (total_decoded or 0) / med,
        "max_new_tokens": args.max_new_tokens,
    }


def run_tpaged(args):
    # Vanilla HF load. Unsloth's Qwen3Attention monkey-patch does not
    # compose with the `paged|<impl>` functional attention interface.
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to("cuda")
    model.eval()

    if args.persistent_cb:
        from persistent_cb import install_for_model  # noqa: WPS433
    prompts_text, prompt_ids = build_prompts(tokenizer, args.n_prompts)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        min_p=0.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    # Raise the paged-cache upper bounds; defaults (256 / 4096) throttle CB.
    gen_config.max_batch_tokens = args.max_batch_tokens
    gen_config.num_blocks = args.num_blocks

    if args.persistent_cb:
        install_for_model(model, gen_config)

    # Warmup on 16 prompts.
    warmup_ids = prompt_ids[:16]
    with torch.inference_mode():
        _ = model.generate_batch(warmup_ids, generation_config=gen_config, progress_bar=False)
    torch.cuda.synchronize()

    n_prompt_tokens = sum(len(p) for p in prompt_ids)
    wall_times = []
    total_decoded = None
    for _ in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate_batch(
                prompt_ids, generation_config=gen_config, progress_bar=False
            )
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        decoded = sum(len(v.generated_tokens) for v in outputs.values())
        total_decoded = decoded

    med = sorted(wall_times)[len(wall_times) // 2]
    return {
        "backend": "tpaged",
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
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vllm", "tpaged"], required=True)
    p.add_argument("--model_name", default="unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--n_prompts", type=int, default=64)
    p.add_argument("--n_rounds", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--max_batch_tokens", type=int, default=8192)
    p.add_argument("--num_blocks", type=int, default=16384)
    p.add_argument("--persistent_cb", action="store_true",
                   help="Reuse a single ContinuousBatchingManager across warmup + measured rounds.")
    p.add_argument("--stats_path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    if args.backend == "vllm":
        out = run_vllm(args)
    else:
        out = run_tpaged(args)

    out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    with open(args.stats_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    # When --persistent_cb is set the background CB worker thread keeps the
    # process alive. Exit fast; the stats file is already flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
