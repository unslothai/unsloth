"""Unified entrypoint for Qwen3-4B GRPO backend comparison.

Single script, N backends. Identical dataset / reward functions / sampling /
callbacks so per-step loss, reward, KL, and grad-norm arrays are directly
comparable across runs.

Backends (pick one via `--backend`):
    vllm             : Unsloth fast_inference=True (vLLM colocated).
    unsloth_fi_false : Unsloth fast_inference=False (custom HF inference
                       kernels + cached fp16 LoRA in fast_linear_forward).
                       Uses trainer's default (non-vLLM, non-CB) rollout path.
    cb_paged         : Vanilla HF + PEFT LoRA + transformers continuous
                       batching with `attn_implementation="paged_attention"`
                       (FA4 shim active).
    cb_sdpa          : Same but with `attn_implementation="sdpa_paged"`.
    naive_trl        : Vanilla HF + PEFT LoRA, no CB, no vLLM (TRL's naive
                       generate path).

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_grpo_unified.py \
        --backend vllm --max_steps 10 \
        --output_dir outputs/grpo_vllm_10 \
        --stats_path logs/grpo_vllm_10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_31")
for p in (HERE, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend",
                   choices=["vllm", "unsloth_fi_false", "cb_paged", "cb_sdpa", "naive_trl"],
                   required=True)
    p.add_argument("--model_name", default="unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.97)
    p.add_argument("--min_p", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--max_batch_tokens", type=int, default=8192)
    p.add_argument("--num_blocks", type=int, default=8192)
    p.add_argument("--persistent_cb", action="store_true")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stats_path", required=True)
    p.add_argument("--seed", type=int, default=3407)
    # Phase 4: torch.compile on the training forward.
    p.add_argument("--compile_mode", default=None,
                   choices=[None, "default", "reduce-overhead",
                            "max-autotune-no-cudagraphs"],
                   help="If set, torch.compile(model.forward, mode=...) after "
                        "the trainer is built. vllm backend is excluded; the "
                        "rollout engine owns its own compile pipeline.")
    p.add_argument("--compile_dynamic", action="store_true", default=True)
    return p.parse_args()


def _prepare_common(args):
    """Dataset + rewards are the same for every backend. Always uses the
    shared chat template and reward funcs from unsloth_grpo_common."""
    from unsloth_grpo_common import (
        apply_chat_template_to_tokenizer, build_dataset,
        build_reward_funcs, build_grpo_kwargs,
    )
    return apply_chat_template_to_tokenizer, build_dataset, build_reward_funcs, build_grpo_kwargs


def _make_stats_callback():
    """StatisticsCallback from torch_debugging_utils. Logs per-step loss,
    grad-norm, memory, and wall time. Reward/KL are picked up from the TRL
    log dict via `on_log`."""
    from torch_debugging_utils import StatisticsCallback
    return StatisticsCallback(
        track_loss=True,
        track_grad_norm=True,
        track_memory=True,
        track_tensor_stats=False,
    )


def _maybe_shim_guided_decoding():
    """Newer vLLM releases have moved GuidedDecodingParams out of
    `vllm.sampling_params`; TRL's GRPOTrainer still tries to import it on
    the transformers-paged path. Inject a no-op shim if missing."""
    try:
        import vllm.sampling_params as sp
        if not hasattr(sp, "GuidedDecodingParams"):
            class _Shim:
                def __init__(self, *a, **kw):
                    pass
            sp.GuidedDecodingParams = _Shim
    except ImportError:
        pass


def _load_unsloth(args, fast_inference: bool):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=fast_inference,
        max_lora_rank=args.lora_rank,
        **({"gpu_memory_utilization": args.gpu_memory_utilization} if fast_inference else {}),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def _load_vanilla_hf(args, attn_impl: str):
    """Vanilla HF + PEFT LoRA. Used by cb_paged / cb_sdpa / naive_trl."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to("cuda")
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok=True)

    import torch
    from torch_debugging_utils import set_all_seeds_fast
    set_all_seeds_fast(args.seed)

    # FA4 shim lives here so CB paths dispatch to Blackwell kernels.
    import flash_attn_fa4_shim  # noqa: F401
    flash_attn_fa4_shim.apply()
    _maybe_shim_guided_decoding()

    (apply_chat_template_to_tokenizer, build_dataset,
     build_reward_funcs, build_grpo_kwargs) = _prepare_common(args)

    # TRL requires `generation_batch_size = pdb * grad_accum * world_size` to
    # be divisible by `num_generations`. Unsloth's loader auto-adjusts
    # `per_device_train_batch_size` to match `num_generations`, but vanilla HF
    # paths (cb_paged, cb_sdpa, naive_trl) do not -- do it ourselves.
    if args.backend not in ("vllm", "unsloth_fi_false"):
        effective = (args.per_device_train_batch_size
                     * args.gradient_accumulation_steps)
        if effective % args.num_generations != 0:
            new_pdb = args.num_generations
            print(f"[{args.backend}] Bumping per_device_train_batch_size "
                  f"{args.per_device_train_batch_size} -> {new_pdb} to satisfy "
                  f"GRPO divisibility.")
            args.per_device_train_batch_size = new_pdb

    # --- load model / tokenizer per backend -----------------------------------
    persistent_teardown_target = None
    if args.backend == "vllm":
        model, tokenizer = _load_unsloth(args, fast_inference=True)
    elif args.backend == "unsloth_fi_false":
        model, tokenizer = _load_unsloth(args, fast_inference=False)
    elif args.backend == "cb_paged":
        # `paged_attention` requires cu_seq_lens on every forward, which only
        # the CB rollout path provides. GRPO's training forward (dense batch)
        # crashes. Load with `sdpa_paged` which gracefully falls back to
        # plain SDPA when paged args are absent, and still exercises the
        # paged path during CB rollout.
        model, tokenizer = _load_vanilla_hf(args, attn_impl="sdpa_paged")
    elif args.backend == "cb_sdpa":
        model, tokenizer = _load_vanilla_hf(args, attn_impl="sdpa_paged")
    elif args.backend == "naive_trl":
        model, tokenizer = _load_vanilla_hf(args, attn_impl="sdpa")
    else:
        raise ValueError(args.backend)

    apply_chat_template_to_tokenizer(tokenizer)
    dataset, maximum_length = build_dataset(tokenizer, max_seq_length=args.max_seq_length)
    print(f"[{args.backend}] p90 prompt length = {maximum_length}")
    reward_funcs = build_reward_funcs(tokenizer)

    # --- GRPOConfig: shared core, backend-specific flags ----------------------
    shared = build_grpo_kwargs(
        tokenizer,
        maximum_length,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
    )
    # Overwrite the equivalence-friendly sampling params.
    shared["temperature"] = args.temperature
    shared["top_p"] = args.top_p
    shared["min_p"] = args.min_p
    # TRL's TopKLogitsWarper rejects -1; accept an int >=0 only.
    shared["top_k"] = args.top_k if args.top_k and args.top_k > 0 else None
    shared["learning_rate"] = args.learning_rate

    from trl import GRPOConfig, GRPOTrainer
    if args.backend == "vllm":
        from vllm import SamplingParams
        vllm_sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, min_p=args.min_p,
            top_k=args.top_k, seed=args.seed,
            stop=[tokenizer.eos_token], include_stop_str_in_output=True,
        )
        training_args = GRPOConfig(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_sampling_params=vllm_sp,
            vllm_gpu_memory_utilization=args.gpu_memory_utilization,
            **shared,
        )
    elif args.backend == "unsloth_fi_false":
        # Trainer's default rollout path: model.generate. Unsloth's
        # fast_inference=False + for_inference() wires the fast single-token
        # decode + cached fp16 LoRA.
        training_args = GRPOConfig(
            use_vllm=False,
            bf16=True,
            **shared,
        )
    elif args.backend in ("cb_paged", "cb_sdpa"):
        training_args = GRPOConfig(
            use_vllm=False,
            use_transformers_paged=True,
            bf16=True,
            generation_kwargs={
                "max_batch_tokens": args.max_batch_tokens,
                "num_blocks": args.num_blocks,
            },
            **shared,
        )
    else:  # naive_trl
        training_args = GRPOConfig(
            use_vllm=False,
            bf16=True,
            **shared,
        )

    stats_cb = _make_stats_callback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[stats_cb],
    )

    if args.persistent_cb and args.backend in ("cb_paged", "cb_sdpa"):
        from persistent_cb import install_for_model, teardown
        base = (trainer.model_wrapped.base_model.model
                if hasattr(trainer.model_wrapped, "base_model")
                else trainer.model_wrapped)
        install_for_model(base, trainer.generation_config)
        persistent_teardown_target = base

    # Phase 4: torch.compile on the training forward.
    if args.compile_mode and args.backend != "vllm":
        from torch_debugging_utils import clear_inductor_cache, CompileDebugger
        clear_inductor_cache()
        CompileDebugger.enable(graph_breaks=True, recompiles=True)
        # Raise Dynamo cache limit so dynamic-shape recompiles don't thrash.
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128
        try:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
        except AttributeError:
            pass
        print(f"[{args.backend}] Compiling trainer.model.forward "
              f"(mode={args.compile_mode}, dynamic={args.compile_dynamic})")
        trainer.model.forward = torch.compile(
            trainer.model.forward,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
        )
        # Reference model inside TRL's GRPO loop also runs a forward.
        ref = getattr(trainer, "ref_model", None)
        if ref is not None:
            ref.forward = torch.compile(
                ref.forward, mode=args.compile_mode,
                dynamic=args.compile_dynamic,
            )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    try:
        trainer.train()
    finally:
        if persistent_teardown_target is not None:
            from persistent_cb import teardown
            teardown(persistent_teardown_target)
    train_wall = time.perf_counter() - t_start

    stats_cb.save_logs(args.stats_path)

    times = [l["time_ms"] for l in stats_cb.logs if "time_ms" in l]
    losses = [l["loss"] for l in stats_cb.logs if "loss" in l]
    rewards = [l.get("reward") for l in stats_cb.logs if "reward" in l]
    kls = [l.get("kl") for l in stats_cb.logs if "kl" in l]
    grad_norms = [l.get("grad_norm") for l in stats_cb.logs if "grad_norm" in l]

    # Post-warmup (skip first 3 steps) median.
    median_step_ms = None
    if len(times) > 3:
        post = sorted(times[3:])
        median_step_ms = post[len(post) // 2]

    summary = {
        "backend": args.backend,
        "max_steps": args.max_steps,
        "train_wall_s": train_wall,
        "median_step_ms_post_warmup": median_step_ms,
        "n_logged_steps": len(stats_cb.logs),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
        },
        "losses": losses,
        "rewards": rewards,
        "kls": kls,
        "grad_norms": grad_norms,
        "step_times_ms": times,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "logs_path": args.stats_path,
    }
    summary_path = Path(args.stats_path).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("losses", "rewards", "kls", "grad_norms", "step_times_ms")},
                     indent=2))
    print(f"\n[{args.backend}] wrote summary to {summary_path}")
    # vLLM engine holds refs; fast-exit rather than wait for shutdown.
    os._exit(0)


if __name__ == "__main__":
    main()
