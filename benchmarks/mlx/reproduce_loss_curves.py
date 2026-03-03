#!/usr/bin/env python3
"""
Unsloth MLX Benchmark: Baseline+compile vs CCE+compile
=======================================================
Each (model, config) runs in its own subprocess for isolation.
Timing excludes warmup/compile steps (first 5 steps skipped).
All configs use gradient checkpointing + mx.compile.
Config order alternates per model to avoid systematic bias.
"""

import json
import subprocess
import sys
import os

# =============================================================================
# CONFIG
# =============================================================================

MODELS = [
    # (model_name, display_name, use_lora)
    # --- Tiny (<1B) ---
    ("mlx-community/Qwen3-0.6B-4bit", "Qwen3-0.6B-4bit", True),
    # --- Small (1-2B) ---
    ("mlx-community/Llama-3.2-1B-Instruct-bf16", "Llama-1B-full", False),
    ("mlx-community/gemma-2-2b-it-4bit", "Gemma2-2B-4bit", True),
    # --- Medium (3-4B) ---
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama-3B-4bit", True),
    ("mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-4bit", True),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "Phi-3.5-mini-4bit", True),
    ("mlx-community/Qwen2.5-3B-Instruct-8bit", "Qwen2.5-3B-8bit", True),
    ("mlx-community/Llama-3.2-3B-Instruct-bf16", "Llama-3B-LoRA", True),
    # --- Large (7-9B) ---
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral-7B-4bit", True),
    ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "Llama-3.1-8B-4bit", True),
]

# (label, use_cce)  — all use compile=True, gradient_checkpointing=True
CONFIGS = [
    ("Baseline+compile", False),
    ("CCE+compile",      True),
]

BATCH_SIZE = 8
SEQ_LEN = 1024
WARMUP_STEPS = 5
MEASURE_STEPS = 15
SEED = 42
LR = 1e-5
LORA_RANK = 8
LORA_ALPHA = 16


# =============================================================================
# WORKER — runs a single (model, config) in isolation
# =============================================================================

def run_worker(model_name, display_name, use_lora, use_cce, wandb_project=None):
    """Run in a subprocess. Prints training progress to stderr, JSON result to stdout."""
    import gc
    import mlx.core as mx
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx_utils import create_batches

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name, max_seq_length=SEQ_LEN,
    )
    if use_lora:
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    dataset = load_dataset("emozilla/pg19-test", split="test")
    batches = create_batches(
        dataset, tokenizer,
        batch_size=BATCH_SIZE, max_seq_length=SEQ_LEN,
        num_batches=WARMUP_STEPS + MEASURE_STEPS + 5, seed=SEED,
    )

    trainer = MLXTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=BATCH_SIZE,
            max_steps=WARMUP_STEPS + MEASURE_STEPS,
            gradient_accumulation_steps=1,
            learning_rate=LR,
            optim="adafactor",
            logging_steps=1,
            max_seq_length=SEQ_LEN,
            use_cce=use_cce,
            compile=True,
            gradient_checkpointing=True,
        ),
    )
    trainer._batches = batches

    gc.collect()
    mx.synchronize()
    mx.reset_peak_memory()

    trainer.train()

    mx.synchronize()
    peak_gb = mx.get_peak_memory() / 1e9

    measure_times = trainer._step_times[WARMUP_STEPS:]
    ms_per_step = sum(measure_times) / len(measure_times) * 1000
    losses = trainer._train_loss_history
    final_loss = losses[-1]
    nan_count = sum(1 for l in losses if l != l)

    if wandb_project:
        try:
            import wandb
            label = "CCE+compile" if use_cce else "Baseline+compile"
            run = wandb.init(
                project=wandb_project,
                name=f"{display_name} ({label})",
                group=display_name,
                reinit=True,
                config={
                    "model_name": model_name,
                    "display_name": display_name,
                    "use_lora": use_lora,
                    "use_cce": use_cce,
                    "batch_size": BATCH_SIZE,
                    "seq_len": SEQ_LEN,
                    "warmup_steps": WARMUP_STEPS,
                    "measure_steps": MEASURE_STEPS,
                    "lr": LR,
                }
            )
            for i, loss in enumerate(losses):
                wandb.log({"loss": loss, "step": i + 1})
            
            wandb.run.summary["ms_per_step"] = ms_per_step
            wandb.run.summary["peak_gb"] = peak_gb
            wandb.run.summary["final_loss"] = final_loss
            wandb.run.summary["nan_count"] = nan_count
            run.finish()
        except ImportError:
            print("Warning: wandb not installed, skipping logging.", file=sys.stderr)

    result = {
        "ms_per_step": ms_per_step,
        "peak_gb": peak_gb,
        "final_loss": final_loss,
        "nan_count": nan_count,
    }
    # Print JSON on a marker line so orchestrator can parse it
    print(f"__RESULT__{json.dumps(result)}", flush=True)


# =============================================================================
# ORCHESTRATOR — spawns subprocesses and collects results
# =============================================================================

def run_subprocess(cmd):
    """Run a subprocess, streaming stdout line by line. Returns (stdout_lines, returncode)."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        if not line.startswith("__RESULT__"):
            print(f"    {line}", flush=True)
    proc.wait()
    return lines, proc.returncode


def main(args):
    print("=" * 80)
    print("Unsloth MLX Benchmark: Baseline+compile vs CCE+compile")
    print("  Gradient Checkpointing: ON | Compile: ON | Isolation: subprocess")
    print("=" * 80)
    print(f"Batch: {BATCH_SIZE}, Seq: {SEQ_LEN}, Warmup: {WARMUP_STEPS}, Measure: {MEASURE_STEPS}")
    print(f"Optimizer: Adafactor (lr={LR})")
    print(f"Models: {len(MODELS)}, Configs: {len(CONFIGS)}")
    print("=" * 80)

    script_path = os.path.abspath(__file__)
    python = sys.executable
    all_results = []

    for mi, (model_name, display_name, use_lora) in enumerate(MODELS):
        print(f"\n{'='*80}")
        print(f"[{mi+1}/{len(MODELS)}] {display_name}")
        print(f"  Repo: {model_name}, LoRA: {use_lora}")
        print("=" * 80)

        # Alternate config order per model to avoid systematic bias
        configs = CONFIGS if mi % 2 == 0 else list(reversed(CONFIGS))
        model_results = {}

        for label, use_cce in configs:
            print(f"\n  --- {label} ---")

            cmd = [
                python, script_path, "--worker",
                "--model", model_name,
                "--display_name", display_name,
                "--use_cce", str(int(use_cce)),
                "--use_lora", str(int(use_lora)),
            ]
            if args.wandb:
                cmd.extend(["--wandb_project", args.project])

            lines, returncode = run_subprocess(cmd)

            # Parse result from stdout
            result = None
            for line in lines:
                if line.startswith("__RESULT__"):
                    result = json.loads(line[len("__RESULT__"):])

            if result and returncode == 0:
                ms = result["ms_per_step"]
                mem = result["peak_gb"]
                loss = result["final_loss"]
                nans = result["nan_count"]
                model_results[label] = (ms, mem, loss, nans)
                status = "OK" if nans == 0 else f"{nans} NaN"
                print(f"  >> {label}: {ms:.0f} ms/step | {mem:.2f} GB | loss={loss:.4f} | {status}")
            else:
                model_results[label] = None
                print(f"  >> {label}: FAILED (exit={returncode})")

        # Per-model summary
        bl = model_results.get("Baseline+compile")
        cce = model_results.get("CCE+compile")
        if bl and cce:
            speedup = bl[0] / cce[0] if cce[0] > 0 else 0
            mem_save = (bl[1] - cce[1]) / bl[1] * 100 if bl[1] > 0 else 0
            print(f"\n  >> {display_name}: CCE+compile is {speedup:.2f}x speed, {mem_save:.1f}% mem saved")

        all_results.append((display_name, model_results))

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY — Baseline+compile vs CCE+compile (GC=ON, compile=ON)")
    print("=" * 80)
    print(f"{ 'Model':<22} {'BL+compile':>14} {'CCE+compile':>14} {'Speedup':>8} {'MemSave':>8}")
    print("-" * 80)

    for display_name, model_results in all_results:
        bl = model_results.get("Baseline+compile")
        cce = model_results.get("CCE+compile")

        if not bl:
            print(f"{display_name:<22} {'FAILED':>14}")
            continue
        if not cce:
            print(f"{display_name:<22} {bl[0]:>7.0f}ms {bl[1]:>5.1f}GB {'FAILED':>14}")
            continue

        speedup = bl[0] / cce[0] if cce[0] > 0 else 0
        mem_save = (bl[1] - cce[1]) / bl[1] * 100 if bl[1] > 0 else 0
        print(
            f"{display_name:<22} "
            f"{bl[0]:>7.0f}ms {bl[1]:>5.1f}GB "
            f"{cce[0]:>7.0f}ms {cce[1]:>5.1f}GB "
            f"{speedup:>7.2f}x {mem_save:>7.1f}%")

    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--display_name", type=str)
    parser.add_argument("--use_cce", type=int, default=0)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="unsloth-mlx-benchmark")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.model, args.display_name, bool(args.use_lora), bool(args.use_cce), args.wandb_project)
    else:
        main(args)
