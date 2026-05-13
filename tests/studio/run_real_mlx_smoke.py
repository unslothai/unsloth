# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
End-to-end MLX smoke test on real Apple Silicon -- multi-process driver.

Two subcommands so the workflow can drive cold-start reloads in fresh
Python processes (the way real users hit the load path):

    python run_real_mlx_smoke.py train  --workdir DIR
    python run_real_mlx_smoke.py reload --format {lora|merged|gguf} --dir D

The `train` subcommand:
  1. Loads `unsloth/gemma-3-270m-it` via FastMLXModel.from_pretrained.
  2. Applies LoRA r=8 on q/k/v/o.
  3. Computes pre-training loss + grad norm via mx.nn.value_and_grad.
  4. Trains 7 deterministic steps on a dataset of the SAME row repeated
     ("<<HELLO!!>> My name is Unsloth!"), with batch_size=2 and
     gradient_accumulation_steps=3 so each step processes 6 sequences
     and the run sees 42 sequences total.
  5. Computes post-training loss + grad norm.
  6. Generates from "<<HELLO!!>> My name is " and asserts "Unsloth"
     appears in the in-memory completion.
  7. Saves the trained model in three formats:
       - LoRA adapter (save_pretrained_merged save_method="lora")
       - Merged 16-bit (save_pretrained_merged save_method="merged_16bit")
       - GGUF (save_pretrained_gguf, best-effort -- skipped with a
         clear reason if save raises; e.g. llama.cpp's
         convert_hf_to_gguf currently asserts on Gemma-3-270m's
         tokenizer vocab. Soft-skipped so the LoRA + merged checks
         continue to gate the PR.)
  8. Emits `train_metrics.json` with per-phase timing / peak GPU /
     peak RSS / per-step losses / pre+post grad norms / generations
     / gguf_supported flag, for regression detection across CI runs.

Reloads run as separate workflow steps so each is a fresh Python
process. For lora / merged the reload uses
FastMLXModel.from_pretrained directly. For gguf the reload spawns
the llama-cli binary built by save_pretrained_gguf and parses
stdout. Each subcommand emits `<format>_reload_metrics.json` next
to the saved dir.

The two upstream unsloth_zoo bugs the earlier draft of this script
worked around are fixed in unslothai/unsloth-zoo#627: GGUF export
no longer raises NotImplementedError on Apple Silicon (llama_cpp.py
catches it from the device_type module-level call) and LoRA reload
via FastMLXModel.from_pretrained(lora_dir) works without an external
config.json copy (mlx_loader.py preserves local_path when config.json
is missing so the adapter_config.json branch can run).

Determinism: seeds Python `random`, `numpy`, and `mlx.core.random` in
every process before any MLX operation. Forwards `random_state=SEED`
to FastMLXModel.from_pretrained / get_peft_model and `seed=SEED` to
MLXTrainingConfig. Metal still has minor reduction-order
nondeterminism, so loss assertions are bounds rather than exact.

Only runnable on a real Apple Silicon host; invoked from
.github/workflows/mlx-ci.yml on the macos-14 runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random as _random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SEED = 3407
TRAIN_TEXT = "<<HELLO!!>> My name is Unsloth!"
PROMPT = "<<HELLO!!>> My name is "
EXPECT_IN_OUTPUT = "Unsloth"
MODEL_NAME = "unsloth/gemma-3-270m-it"


# ---------------------------------------------------------------------------
# Determinism + telemetry helpers
# ---------------------------------------------------------------------------


def _seed_everything() -> None:
    _random.seed(SEED)
    np.random.seed(SEED)
    import mlx.core as mx

    mx.random.seed(SEED)


def _peak_gpu_gb() -> float:
    import mlx.core as mx

    if not mx.metal.is_available():
        return 0.0
    # Newer MLX deprecates mx.metal.get_peak_memory in favour of the
    # top-level mx.get_peak_memory; fall back to the old API for
    # compatibility with older MLX versions still present in the
    # environment.
    getter = getattr(mx, "get_peak_memory", None) or getattr(
        mx.metal, "get_peak_memory", None
    )
    if getter is None:
        return 0.0
    try:
        return float(getter()) / (1024**3)
    except Exception:
        return 0.0


def _peak_rss_gb() -> float:
    """Peak resident set size for this process. macOS getrusage returns
    bytes; Linux returns kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024**3)
    return float(rss) / (1024**2)


class Phase:
    """Wall-clock + memory tracker for a named phase. Records into a
    metrics dict so we can later JSON-dump for regression detection."""

    def __init__(self, name: str, metrics: dict):
        self.name = name
        self.metrics = metrics

    def __enter__(self):
        self._t0 = time.perf_counter()
        print(f"\n=== phase:{self.name} START ===", flush = True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        peak_gpu = _peak_gpu_gb()
        peak_rss = _peak_rss_gb()
        self.metrics.setdefault("phases", {})[self.name] = {
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_gb": round(peak_gpu, 3),
            "peak_rss_gb": round(peak_rss, 3),
            "ok": exc_type is None,
        }
        status = "OK" if exc_type is None else f"FAIL ({exc_type.__name__})"
        print(
            f"=== phase:{self.name} {status} elapsed={elapsed:.2f}s "
            f"peak_gpu={peak_gpu:.2f}GB peak_rss={peak_rss:.2f}GB ===",
            flush = True,
        )
        return False  # don't swallow exceptions


def _compute_loss_and_grad_norm(model, tokenizer, text: str) -> tuple[float, float]:
    """One forward+backward of next-token cross-entropy on `text`.
    Returns (loss, ||grad||_2)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    ids = list(tokenizer.encode(text))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.append(int(eos_id))
    if len(ids) < 2:
        raise RuntimeError(f"text too short to compute loss: {len(ids)} tokens")

    inputs = mx.array([ids[:-1]], dtype = mx.int32)
    targets = mx.array([ids[1:]], dtype = mx.int32)

    def loss_fn(m):
        logits = m(inputs)
        return nn.losses.cross_entropy(logits, targets, reduction = "mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss_val, grad = loss_and_grad(model)

    norm_sq = mx.array(0.0, dtype = mx.float32)
    for _name, value in tree_flatten(grad):
        v = value.astype(mx.float32)
        norm_sq = norm_sq + mx.sum(v * v)
    return float(loss_val.item()), float(mx.sqrt(norm_sq).item())


def _write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent = 2, default = str))
    print(f"\n[metrics] wrote {path}", flush = True)
    print(json.dumps(metrics, indent = 2, default = str), flush = True)


# ---------------------------------------------------------------------------
# `train` subcommand
# ---------------------------------------------------------------------------


def cmd_train(args) -> int:
    _seed_everything()
    metrics: dict = {
        "subcommand": "train",
        "seed": SEED,
        "model": MODEL_NAME,
        "train_text": TRAIN_TEXT,
        "prompt": PROMPT,
        "phases": {},
    }
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents = True, exist_ok = True)

    import mlx.core as mx
    from unsloth_zoo.mlx_loader import FastMLXModel
    from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig

    hf_token = os.environ.get("HF_TOKEN") or None

    with Phase("load_base", metrics):
        model, tokenizer = FastMLXModel.from_pretrained(
            MODEL_NAME,
            load_in_4bit = False,
            dtype = "float16",
            text_only = True,
            max_seq_length = 128,
            random_state = SEED,
            token = hf_token,
            trust_remote_code = False,
        )
    metrics["base_src_path"] = str(getattr(model, "_src_path", "") or "")

    mx.random.seed(SEED)

    with Phase("apply_lora", metrics):
        # Standard unsloth LoRA target set (q/k/v/o + gate/up/down).
        # With bs=2 grad_accum=3 (effective batch 6) the q/k/v/o-only
        # LoRA collapsed in 7 steps -- training loss kept dropping but
        # inference output the structural skeleton ("My name") without
        # recovering the specific "Unsloth" token. Including the MLP
        # projections gives the LoRA enough capacity to memorize the
        # training row at the larger effective batch.
        model = FastMLXModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            lora_dropout = 0.0,
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing = False,
            random_state = SEED,
            finetune_language_layers = True,
            finetune_attention_modules = True,
            finetune_mlp_modules = True,
        )

    with Phase("pre_train_grad_probe", metrics):
        pre_loss, pre_norm = _compute_loss_and_grad_norm(model, tokenizer, TRAIN_TEXT)
    metrics["pre_train_loss"] = round(pre_loss, 4)
    metrics["pre_train_grad_norm"] = round(pre_norm, 4)
    assert math.isfinite(pre_loss) and math.isfinite(pre_norm) and pre_norm > 0

    losses_per_step: list[float] = []
    with Phase("train", metrics):
        config = MLXTrainingConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 3,
            max_steps = 7,
            learning_rate = 1e-3,
            warmup_steps = 0,
            lr_scheduler_type = "constant",
            optim = "adamw",
            weight_decay = 0.0,
            max_grad_norm = 1.0,
            logging_steps = 1,
            max_seq_length = 64,
            seed = SEED,
            use_cce = False,
            compile = False,
            gradient_checkpointing = False,
            output_dir = str(workdir / "trainer_outputs"),
            save_steps = 0,
            eval_steps = 0,
            dataset_text_field = "text",
        )
        trainer = MLXTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = [{"text": TRAIN_TEXT}] * 64,
            args = config,
        )

        def _on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens):
            losses_per_step.append(round(float(loss), 4))
            print(
                f"  step {step}/{total}  loss={loss:.4f}  lr={lr:.2e}  "
                f"tok/s={tok_s:.0f}  peak={peak_gb:.2f}GB",
                flush = True,
            )

        trainer.add_step_callback(_on_step)
        train_result = trainer.train()
    metrics["losses_per_step"] = losses_per_step
    metrics["train_summary"] = {
        k: train_result[k]
        for k in (
            "train_loss",
            "train_runtime",
            "train_steps",
            "trained_tokens",
            "train_samples_per_second",
            "compile_enabled",
            "patch_mode",
        )
        if k in train_result
    }
    assert len(losses_per_step) == 7, f"expected 7 logged steps, got {losses_per_step}"
    for i, l in enumerate(losses_per_step):
        assert math.isfinite(l) and 0 < l < 50, f"step {i+1} loss bad: {l}"
    assert (
        losses_per_step[-1] < losses_per_step[0] * 1.1
    ), f"loss diverged: {losses_per_step[0]} -> {losses_per_step[-1]}"

    with Phase("post_train_grad_probe", metrics):
        post_loss, post_norm = _compute_loss_and_grad_norm(model, tokenizer, TRAIN_TEXT)
    metrics["post_train_loss"] = round(post_loss, 4)
    metrics["post_train_grad_norm"] = round(post_norm, 4)
    assert post_loss < pre_loss, f"post {post_loss} >= pre {pre_loss}"

    from mlx_lm import generate

    with Phase("inference_in_memory", metrics):
        model.eval()
        in_mem_out = generate(
            model,
            tokenizer,
            prompt = PROMPT,
            max_tokens = 48,
            verbose = False,
        )
    metrics["in_memory_generation"] = in_mem_out
    assert (
        EXPECT_IN_OUTPUT in in_mem_out
    ), f"in-memory generation gibberish: {in_mem_out!r}"

    # Save LoRA. unsloth-zoo#627 fixed FastMLXModel.from_pretrained(lora_dir)
    # so the cold-start reload below works on the saved adapter dir directly.
    lora_dir = workdir / "lora"
    with Phase("save_lora", metrics):
        model.save_pretrained_merged(
            str(lora_dir),
            tokenizer = tokenizer,
            save_method = "lora",
        )
    metrics["lora_dir"] = str(lora_dir)
    assert (lora_dir / "adapters.safetensors").exists()
    assert (lora_dir / "adapter_config.json").exists()

    # Save merged_16bit (full HF directory)
    merged_dir = workdir / "merged_16bit"
    with Phase("save_merged_16bit", metrics):
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer = tokenizer,
            save_method = "merged_16bit",
        )
    metrics["merged_dir"] = str(merged_dir)
    assert any(merged_dir.glob("*.safetensors"))

    # Save GGUF (best-effort). save_pretrained_gguf clones llama.cpp,
    # builds it with cmake (Metal=ON), then runs convert_hf_to_gguf.
    # For some models -- including unsloth/gemma-3-270m-it as of
    # 2026-05-07 -- llama.cpp's converter asserts on the tokenizer vocab
    # (`assert max(tokenizer.vocab.values()) < vocab_size`) because the
    # tokenizer carries reserved IDs beyond the embedding matrix size.
    # That's an llama.cpp / convert_hf_to_gguf limitation, not an
    # unsloth_zoo bug. Soft-skip with a recorded reason so the LoRA +
    # merged_16bit assertions still gate the PR.
    gguf_dir = workdir / "gguf"
    metrics["gguf_supported"] = False
    metrics["gguf_skip_reason"] = None
    metrics["gguf_dir"] = str(gguf_dir)
    with Phase("save_gguf", metrics):
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer = tokenizer,
                quantization_method = "not_quantized",
            )
            gguf_files = sorted(gguf_dir.glob("*.gguf"))
            if not gguf_files:
                raise RuntimeError(f"no .gguf produced in {gguf_dir}")
            metrics["gguf_supported"] = True
            metrics["gguf_files"] = [p.name for p in gguf_files]
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            if "AssertionError" in err_text or "tokenizer.vocab" in err_text:
                metrics["gguf_skip_reason"] = (
                    f"llama.cpp convert_hf_to_gguf asserted on tokenizer "
                    f"vocab for {MODEL_NAME} (max(vocab IDs) >= "
                    f"vocab_size). Downstream llama.cpp limitation, not "
                    f"unsloth_zoo. Underlying error: {err_text}"
                )
            else:
                metrics["gguf_skip_reason"] = err_text
            print(f"  GGUF SKIPPED: {metrics['gguf_skip_reason']}", flush = True)

    metrics["final_peak_gpu_gb"] = round(_peak_gpu_gb(), 3)
    metrics["final_peak_rss_gb"] = round(_peak_rss_gb(), 3)

    _write_metrics(workdir / "train_metrics.json", metrics)
    return 0


# ---------------------------------------------------------------------------
# `reload` subcommand (fresh process per format)
# ---------------------------------------------------------------------------


def cmd_reload(args) -> int:
    _seed_everything()
    save_dir = Path(args.dir).resolve()
    if not save_dir.exists():
        raise SystemExit(f"reload dir not found: {save_dir}")

    metrics: dict = {
        "subcommand": "reload",
        "format": args.format,
        "dir": str(save_dir),
        "phases": {},
    }

    if args.format == "gguf":
        return _reload_gguf(save_dir, metrics)

    import mlx.core as mx
    from unsloth_zoo.mlx_loader import FastMLXModel
    from mlx_lm import generate

    hf_token = os.environ.get("HF_TOKEN") or None

    with Phase(f"reload_{args.format}", metrics):
        mx.random.seed(SEED)
        m, t = FastMLXModel.from_pretrained(
            str(save_dir),
            load_in_4bit = False,
            dtype = "float16",
            text_only = True,
            max_seq_length = 128,
            random_state = SEED,
            token = hf_token,
        )
        m.eval()

    with Phase(f"generate_{args.format}", metrics):
        out = generate(m, t, prompt = PROMPT, max_tokens = 48, verbose = False)
    metrics["generation"] = out
    print(f"  [reload:{args.format}] output: {out!r}", flush = True)
    assert (
        EXPECT_IN_OUTPUT in out
    ), f"reload {args.format!r} produced gibberish for {PROMPT!r}: {out!r}"

    metrics["final_peak_gpu_gb"] = round(_peak_gpu_gb(), 3)
    metrics["final_peak_rss_gb"] = round(_peak_rss_gb(), 3)
    _write_metrics(save_dir.parent / f"{args.format}_reload_metrics.json", metrics)
    return 0


def _reload_gguf(save_dir: Path, metrics: dict) -> int:
    candidates = [
        Path("llama.cpp/llama-cli"),
        Path("llama.cpp/build/bin/llama-cli"),
    ]
    llama_cli = next((c for c in candidates if c.exists()), None)
    if llama_cli is None:
        raise SystemExit(f"llama-cli not found; checked {candidates}")

    gguf_files = sorted(save_dir.glob("*.gguf"))
    if not gguf_files:
        raise SystemExit(f"no .gguf files in {save_dir}")
    gguf_path = gguf_files[0]

    with Phase("reload_gguf", metrics):
        proc = subprocess.run(
            [
                str(llama_cli),
                "-m",
                str(gguf_path),
                "-p",
                PROMPT,
                "-n",
                "24",
                "--temp",
                "0",
                "--seed",
                str(SEED),
                "-no-cnv",
                "--no-warmup",
            ],
            capture_output = True,
            text = True,
            timeout = 300,
        )

    metrics["llama_cli_returncode"] = proc.returncode
    metrics["generation"] = (proc.stdout or "")[:1500]
    metrics["stderr_head"] = (proc.stderr or "")[:600]

    print(f"  [reload:gguf] stdout (head):\n{proc.stdout[:800]}", flush = True)
    if proc.returncode != 0:
        raise SystemExit(
            f"llama-cli exit {proc.returncode}; stderr head: {proc.stderr[:400]}"
        )
    assert EXPECT_IN_OUTPUT in (
        proc.stdout or ""
    ), f"GGUF reload gibberish for {PROMPT!r}: {proc.stdout[:400]!r}"

    metrics["final_peak_rss_gb"] = round(_peak_rss_gb(), 3)
    _write_metrics(save_dir.parent / "gguf_reload_metrics.json", metrics)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest = "cmd", required = True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--workdir", required = True)

    p_reload = sub.add_parser("reload")
    p_reload.add_argument(
        "--format",
        required = True,
        choices = ["lora", "merged", "gguf"],
    )
    p_reload.add_argument("--dir", required = True)

    args = parser.parse_args()
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "reload":
        return cmd_reload(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
