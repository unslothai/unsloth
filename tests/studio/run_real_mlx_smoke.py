# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""End-to-end MLX smoke test on real Apple Silicon (multi-process driver).

`train` overfits gemma-3-270m-it on one row for 30 steps and saves
lora/merged_16bit/gguf; `reload` reopens each format in a fresh process.
GGUF + LoRA reload fixes land in unslothai/unsloth-zoo#627. Metal's
reduction-order nondeterminism makes loss assertions bounds, not exact.
Apple-Silicon only; invoked from .github/workflows/mlx-ci.yml.
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
    # Newer MLX moved get_peak_memory to top-level; fall back to mx.metal.
    getter = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    if getter is None:
        return 0.0
    try:
        return float(getter()) / (1024**3)
    except Exception:
        return 0.0


def _peak_rss_gb() -> float:
    """Peak RSS for this process (macOS getrusage = bytes, Linux = KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024**3)
    return float(rss) / (1024**2)


class Phase:
    """Wall-clock + memory tracker for a named phase; records into a metrics dict."""

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
    """One fwd+bwd of next-token CE on `text`. Returns (loss, ||grad||_2)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    # Match Unsloth's text dataset path: no EOS appended behind the user's back.
    ids = list(tokenizer.encode(text))
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


def _teacher_forced_completion_loss(model, tokenizer, prompt: str, completion: str) -> float:
    """Mean teacher-forced next-token CE on `completion` given `prompt`.

    Decouples the memorisation check from flaky greedy-decode geometry:
    asserts *what* the model memorised, not just that loss is low.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prompt_ids = list(tokenizer.encode(prompt))
    full_ids = list(tokenizer.encode(prompt + completion))
    if len(full_ids) <= len(prompt_ids):
        raise RuntimeError(
            f"completion {completion!r} tokenises to zero new tokens after "
            f"{prompt!r}; check tokenizer / chat template."
        )

    inputs = mx.array([full_ids[:-1]], dtype = mx.int32)
    targets = mx.array([full_ids[1:]], dtype = mx.int32)
    logits = model(inputs)

    # logits at position i predict targets[i]; completion starts at len(prompt_ids)-1.
    start = len(prompt_ids) - 1
    completion_logits = logits[:, start:, :]
    completion_targets = targets[:, start:]
    loss = nn.losses.cross_entropy(completion_logits, completion_targets, reduction = "mean")
    return float(loss.item())


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
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

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
        # Full q/k/v/o + gate/up/down set: q/k/v/o alone couldn't memorize
        # the row, the MLP projections add the needed capacity.
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
            # PR #5498 sweep: 7 steps too few; 30 makes every seed converge.
            max_steps = 30,
            learning_rate = 1e-3,
            warmup_steps = 0,
            lr_scheduler_type = "constant",
            optim = "adamw",
            weight_decay = 0.0,
            # Pin the elementwise clip (value=1.0, norm disabled) to match the
            # 13-seed-tested fixture; explicit value overrides zoo's MLX default.
            max_grad_norm = 0.0,
            max_grad_value = 1.0,
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

        def _on_step(
            step,
            total,
            loss,
            lr,
            tok_s,
            peak_gb,
            elapsed,
            num_tokens,
            grad_norm = None,
        ):
            losses_per_step.append(round(float(loss), 4))
            grad_text = f"  grad={grad_norm:.4f}" if grad_norm is not None else ""
            print(
                f"  step {step}/{total}  loss={loss:.4f}  lr={lr:.2e}  "
                f"tok/s={tok_s:.0f}  peak={peak_gb:.2f}GB{grad_text}",
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
    # logging_steps=1 + max_steps=N -> N callbacks; gate auto-follows max_steps.
    expected_logged_steps = int(config.max_steps)
    assert (
        len(losses_per_step) == expected_logged_steps
    ), f"expected {expected_logged_steps} logged steps, got {losses_per_step}"
    if "train_steps" in train_result:
        assert int(train_result["train_steps"]) == expected_logged_steps, (
            f"expected train_steps={expected_logged_steps}, got " f"{train_result['train_steps']}"
        )
    for i, l in enumerate(losses_per_step):
        # Allow exact 0.0: fp16 loss underflows once the LoRA memorises the
        # row (~step 10); that's success, so the lower bound is >= 0 not > 0.
        assert math.isfinite(l) and 0 <= l < 50, f"step {i+1} loss bad: {l}"
    assert (
        losses_per_step[-1] < losses_per_step[0] * 1.1
    ), f"loss diverged: {losses_per_step[0]} -> {losses_per_step[-1]}"

    with Phase("post_train_grad_probe", metrics):
        post_loss, post_norm = _compute_loss_and_grad_norm(model, tokenizer, TRAIN_TEXT)
    metrics["post_train_loss"] = round(post_loss, 4)
    metrics["post_train_grad_norm"] = round(post_norm, 4)
    assert post_loss < pre_loss, f"post {post_loss} >= pre {pre_loss}"
    # Memorisation gate: every converging (clip, bc, seed) config in the
    # 13-seed sweep hit post_train_loss <= 0.05, so 0.1 is a robust bound.
    assert post_loss < 0.1, (
        f"post_train_loss={post_loss:.4f} >= 0.1 -- training did not "
        "memorise the single training row in 30 steps. Trainer "
        "regression suspected."
    )

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
    # Soft greedy-decode metric only (46-77% of seeds): fp16 + MLX generate
    # noises the first token. The teacher-forced check below is load-bearing.
    metrics["in_memory_generation_has_expected"] = EXPECT_IN_OUTPUT in in_mem_out
    if EXPECT_IN_OUTPUT not in in_mem_out:
        print(
            f"  [INFO] greedy decode did not contain {EXPECT_IN_OUTPUT!r} "
            f"(post_train_loss={post_loss:.4f}, completion={in_mem_out!r}). "
            "Hard gate is the teacher-forced completion-loss check below.",
            flush = True,
        )

    # Hard check: teacher-forced loss on the trained completion bypasses
    # greedy-decode fp16 fragility. 13/13 measured configs reached < 1e-3,
    # so this gate is deterministic across (seed, clip, bc).
    completion_loss = _teacher_forced_completion_loss(
        model, tokenizer, PROMPT, EXPECT_IN_OUTPUT + "!"
    )
    metrics["in_memory_completion_teacher_forced_loss"] = round(completion_loss, 6)
    assert completion_loss < 0.5, (
        f"teacher-forced completion loss {completion_loss:.4f} >= 0.5: "
        f"the LoRA did not memorise {EXPECT_IN_OUTPUT + '!'!r} after "
        f"{PROMPT!r} (post_train_loss={post_loss:.4f}). Trainer regression "
        "suspected -- check unsloth_zoo MLX trainer gradient clipping / "
        "optimizer defaults vs torch.optim.AdamW."
    )

    # unsloth-zoo#627 fixed from_pretrained(lora_dir) so the cold-start
    # reload below works on the saved adapter dir directly.
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

    # Save GGUF (best-effort). For some models (e.g. gemma-3-270m-it)
    # llama.cpp's convert_hf_to_gguf asserts on the tokenizer vocab -- an
    # llama.cpp limitation, not an unsloth_zoo bug. Soft-skip with a recorded
    # reason so the LoRA + merged_16bit assertions still gate the PR.
    gguf_dir = workdir / "gguf"
    metrics["gguf_supported"] = False
    metrics["gguf_skip_reason"] = None
    metrics["gguf_dir"] = str(gguf_dir)
    with Phase("save_gguf", metrics):
        try:
            # q8_0 (the exporter default), not bf16: llama.cpp has optimized q8_0
            # CPU kernels, whereas bf16 CPU decode is unusably slow on the runner
            # and made the fresh-process llama-cli reload below time out. q8_0 is
            # also what users deploy by default.
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer = tokenizer,
                quantization_method = "fast_quantized",
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
    from unsloth_zoo.mlx.loader import FastMLXModel
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

    # Save/reload invariant: reloaded teacher-forced loss on TRAIN_TEXT must
    # match the in-memory post_train_loss. Robust to MLX's greedy-decode
    # perturbation, which can flip the first token but not the loss.
    train_metrics_path = save_dir.parent / "train_metrics.json"
    in_mem_loss = None
    in_mem_out = None
    if train_metrics_path.exists():
        try:
            tm = json.loads(train_metrics_path.read_text())
            in_mem_loss = tm.get("post_train_loss")
            in_mem_out = tm.get("in_memory_generation")
        except Exception:
            in_mem_loss = None
    metrics["in_memory_generation_ref"] = in_mem_out
    metrics["in_memory_post_train_loss"] = in_mem_loss
    metrics["reload_completion_matches_in_memory"] = in_mem_out is not None and out == in_mem_out
    if isinstance(in_mem_loss, (int, float)) and math.isfinite(in_mem_loss):
        reload_loss, _ = _compute_loss_and_grad_norm(m, t, TRAIN_TEXT)
        metrics["reload_post_train_loss"] = round(reload_loss, 4)
        # float16 round-trip is near-exact; 0.2 tolerates dequant noise.
        assert abs(reload_loss - float(in_mem_loss)) < 0.2, (
            f"reload {args.format!r} loss diverged from in-memory: "
            f"reload={reload_loss:.4f}, in-memory={in_mem_loss:.4f}"
        )
    else:
        # Fallback when train_metrics.json is missing: gate on non-empty output.
        body = out.replace(PROMPT, "", 1).strip()
        assert len(body) >= 4, (
            f"reload {args.format!r} produced no usable output for " f"{PROMPT!r}: {out!r}"
        )

    metrics["final_peak_gpu_gb"] = round(_peak_gpu_gb(), 3)
    metrics["final_peak_rss_gb"] = round(_peak_rss_gb(), 3)
    _write_metrics(save_dir.parent / f"{args.format}_reload_metrics.json", metrics)
    return 0


def _find_llama_cli() -> Path | None:
    """Locate the llama-cli binary save_pretrained_gguf built.

    save_pretrained_gguf installs llama.cpp under unsloth_zoo's LLAMA_CPP_DEFAULT_DIR
    ($UNSLOTH_LLAMA_CPP_PATH or ~/.unsloth/llama.cpp), not the working directory, so
    search there first and keep the CWD-relative layout as a fallback.
    """
    bases: list[Path] = []
    env_dir = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
    if env_dir:
        bases.append(Path(env_dir))
    try:
        from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR
        bases.append(Path(LLAMA_CPP_DEFAULT_DIR))
    except Exception:
        bases.append(Path.home() / ".unsloth" / "llama.cpp")
    bases.append(Path("llama.cpp"))

    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        for rel in ("llama-cli", "build/bin/llama-cli"):
            cand = base / rel
            if cand.is_file() and os.access(cand, os.X_OK):
                # Absolute: a separator-less relative path would send subprocess
                # to a PATH lookup instead of running the file.
                return cand.resolve()
        # Last resort: the binary may sit under an unexpected build subdir.
        if base.is_dir():
            for cand in sorted(base.glob("**/llama-cli")):
                if cand.is_file() and os.access(cand, os.X_OK):
                    return cand.resolve()
    return None


def _reload_gguf(save_dir: Path, metrics: dict) -> int:
    llama_cli = _find_llama_cli()
    if llama_cli is None:
        raise SystemExit(
            "llama-cli not found under $UNSLOTH_LLAMA_CPP_PATH, "
            "~/.unsloth/llama.cpp, or ./llama.cpp"
        )

    gguf_files = sorted(save_dir.glob("*.gguf"))
    if not gguf_files:
        raise SystemExit(f"no .gguf files in {save_dir}")
    gguf_path = gguf_files[0]

    # Save/reload-integrity smoke (assert below only needs a few chars). The GGUF is
    # exported q8_0 (see save_gguf) because llama.cpp bf16 CPU decode is unusably slow
    # on the runner. Run CPU-only (-ngl 0), cap the context (-c 256, the model
    # advertises 32768), and keep generation short; all env-tunable.
    n_predict = os.environ.get("UNSLOTH_GGUF_RELOAD_N", "8")
    n_threads = os.environ.get("UNSLOTH_GGUF_RELOAD_THREADS", str(os.cpu_count() or 4))
    n_ctx = os.environ.get("UNSLOTH_GGUF_RELOAD_CTX", "256")
    n_gpu_layers = os.environ.get("UNSLOTH_GGUF_RELOAD_NGL", "0")
    reload_timeout = int(os.environ.get("UNSLOTH_GGUF_RELOAD_TIMEOUT", "420"))
    argv = [
        str(llama_cli),
        "-m",
        str(gguf_path),
        "-p",
        PROMPT,
        "-n",
        n_predict,
        "-t",
        n_threads,
        "-c",
        n_ctx,
        "-ngl",
        n_gpu_layers,
        "--temp",
        "0",
        "--seed",
        str(SEED),
        "--no-warmup",
    ]
    with Phase("reload_gguf", metrics):
        try:
            proc = subprocess.run(
                argv,
                capture_output = True,
                text = True,
                timeout = reload_timeout,
                # Newer llama.cpp keeps llama-cli in chat mode; exit after one reply.
                input = "/exit\n",
            )
        except subprocess.TimeoutExpired as exc:

            def _decode(stream) -> str:
                if isinstance(stream, bytes):
                    return stream.decode("utf-8", errors = "replace")
                return stream or ""

            print(f"  [reload:gguf] TIMEOUT running: {' '.join(argv)}", flush = True)
            print(f"  [reload:gguf] TIMEOUT stdout:\n{_decode(exc.stdout)[:1000]}", flush = True)
            print(f"  [reload:gguf] TIMEOUT stderr:\n{_decode(exc.stderr)[:1000]}", flush = True)
            raise

    metrics["llama_cli_returncode"] = proc.returncode
    metrics["generation"] = (proc.stdout or "")[:1500]
    metrics["stderr_head"] = (proc.stderr or "")[:600]

    print(f"  [reload:gguf] stdout (head):\n{proc.stdout[:800]}", flush = True)
    if proc.returncode != 0:
        raise SystemExit(f"llama-cli exit {proc.returncode}; stderr head: {proc.stderr[:400]}")
    # llama.cpp tokenises/samples differently than mlx_lm, so the GGUF
    # completion needn't match. Require non-empty output to catch real
    # save/reload corruption; record EXPECT_IN_OUTPUT without gating on it.
    body = (proc.stdout or "").replace(PROMPT, "", 1).strip()
    metrics["gguf_has_expected"] = EXPECT_IN_OUTPUT in (proc.stdout or "")
    assert len(body) >= 4, (
        f"GGUF reload produced no usable output for {PROMPT!r}: " f"{proc.stdout[:400]!r}"
    )

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
