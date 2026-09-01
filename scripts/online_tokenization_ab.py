# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager vs online preparation, measured through Unsloth's real training path.

Drives ``UnslothTrainer.load_model`` -> ``prepare_model_for_training`` ->
``load_and_format_dataset`` -> ``start_training``, so the integrated gating is
what gets measured. The arms differ only by ``UNSLOTH_STUDIO_ONLINE_TOKENIZATION``:

    python scripts/online_tokenization_ab.py --arm eager  --dataset <split> --out ab_eager.json
    python scripts/online_tokenization_ab.py --arm online --dataset <split> --out ab_online.json

Same seed, rows and order, so per-step losses must match; a mismatch means the
lazy transform is not producing the rows the eager map produced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Scratch root for the per-arm `datasets` cache;
WORKSPACE = Path(os.environ.get("UNSLOTH_WORKSPACE") or tempfile.gettempdir())

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

sys.path.insert(0, str(REPO / "studio" / "backend"))
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices = ("eager", "online"), required = True)
    # No default path: it would only exist on one machine.
    parser.add_argument(
        "--dataset",
        required = True,
        help = "Parquet/JSONL split, or a Hugging Face dataset id, carrying a text column",
    )
    parser.add_argument("--model", default = "unsloth/Qwen3-0.6B", help = "Model id or local path")
    parser.add_argument("--max-steps", type = int, default = 30)
    parser.add_argument("--batch-size", type = int, default = 2)
    parser.add_argument("--grad-accum", type = int, default = 4)
    parser.add_argument("--max-seq-length", type = int, default = 2048)
    parser.add_argument("--out", required = True)
    parser.add_argument("--fresh-cache", action = "store_true", default = True)
    parser.add_argument("--no-fresh-cache", dest = "fresh_cache", action = "store_false")
    args = parser.parse_args()

    # tokenize map out of Arrow and measures a cache hit real users never get.
    # Fresh cache per run, else the eager arm just reads the other arm's tokenize map out of Arrow and measures a cache
    if args.fresh_cache:
        cache = WORKSPACE / "unsloth_ab_cache" / f"{args.arm}_{int(time.time())}"
        cache.mkdir(parents = True, exist_ok = True)
        os.environ["HF_DATASETS_CACHE"] = str(cache)

    # Set before anything imports the gate.
    if args.arm == "eager":
        os.environ["UNSLOTH_STUDIO_ONLINE_TOKENIZATION"] = "0"
    else:
        os.environ.pop("UNSLOTH_STUDIO_ONLINE_TOKENIZATION", None)

    import unsloth  # noqa: F401  - must precede transformers/trl
    from transformers import TrainerCallback

    from core.training.trainer import UnslothTrainer

    start = time.perf_counter()
    marks: dict = {}

    def mark(name: str) -> None:
        marks[name] = round(time.perf_counter() - start, 4)
        print(f"[phase] {name} @ {marks[name]}s", flush = True)

    trainer = UnslothTrainer()
    if not trainer.load_model(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
    ):
        print("model load failed", file = sys.stderr)
        return 1
    if not trainer.prepare_model_for_training(
        use_lora = True,
        lora_r = 16,
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
        use_gradient_checkpointing = "unsloth",
    ):
        print("model prepare failed", file = sys.stderr)
        return 1
    mark("model_ready")

    # `local_datasets` resolves its entries to files and rejects anything without a supported extension, so a Hub id has
    # to go through `dataset_source` instead.
    local_split = os.path.exists(args.dataset) or Path(args.dataset).suffix.lower() in (
        ".json",
        ".jsonl",
        ".csv",
        ".parquet",
    )
    result = trainer.load_and_format_dataset(
        dataset_source = None if local_split else args.dataset,
        format_type = "auto",
        local_datasets = [args.dataset] if local_split else None,
    )
    if result is None:
        print("dataset load failed", file = sys.stderr)
        return 1
    dataset, eval_dataset = result
    mark("dataset_formatted")

    class _Probe(TrainerCallback):
        """Wall clock at train() and at every step, plus the loss stream."""

        def __init__(self):
            self.losses: list = []
            self.step_times: list = []

        def on_train_begin(self, targs, state, control, **kwargs):
            mark("train_begin")

        def on_step_end(self, targs, state, control, **kwargs):
            self.step_times.append(round(time.perf_counter() - start, 4))
            if len(self.step_times) == 1:
                mark("first_step_end")

        def on_log(
            self,
            targs,
            state,
            control,
            logs = None,
            **kwargs,
        ):
            if logs and "loss" in logs:
                self.losses.append(logs["loss"])

    probe = _Probe()

    # The trainer only exists inside the worker thread, so attach on appearance.
    original_preflight = trainer._preflight_first_batch

    def _preflight_with_probe():
        mark("trainer_built")
        trainer.trainer.add_callback(probe)
        error = original_preflight()
        mark("prewarm_done")
        return error

    trainer._preflight_first_batch = _preflight_with_probe

    started = trainer.start_training(
        dataset = dataset,
        eval_dataset = eval_dataset,
        output_dir = f"ab_{args.arm}",
        num_epochs = 1,
        max_steps = args.max_steps,
        batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate = 2e-4,
        weight_decay = 0.01,
        random_seed = 3407,
        max_seq_length = args.max_seq_length,
        packing = False,
        train_on_completions = False,
    )
    if not started:
        print("training failed to start", file = sys.stderr)
        return 1

    while trainer.training_thread and trainer.training_thread.is_alive():
        time.sleep(1)
    trainer.training_thread.join()
    mark("train_done")

    progress = trainer.get_training_progress()
    error = getattr(progress, "error", None)

    decision = getattr(trainer, "_online_prewarm_batches", 0)
    # What the trainer actually got configured with, read off the object.
    observed = {}
    sft = getattr(trainer, "trainer", None)
    if sft is not None:
        targs = getattr(sft, "args", None)
        split = getattr(sft, "train_dataset", None)
        fmt = getattr(split, "format", None)
        observed = {
            "dataloader_num_workers": getattr(targs, "dataloader_num_workers", None),
            "dataloader_persistent_workers": getattr(targs, "dataloader_persistent_workers", None),
            "dataloader_prefetch_factor": getattr(targs, "dataloader_prefetch_factor", None),
            "dataset_kwargs": getattr(targs, "dataset_kwargs", None),
            "remove_unused_columns": getattr(targs, "remove_unused_columns", None),
            "padding_free": getattr(targs, "padding_free", None),
            "packing": getattr(targs, "packing", None),
            "dataset_num_proc": getattr(targs, "dataset_num_proc", None),
            "train_split_format": fmt.get("type") if isinstance(fmt, dict) else None,
            "train_split_columns": list(getattr(split, "column_names", None) or []),
            "train_split_rows": len(split) if split is not None else None,
        }
    payload = {
        "arm": args.arm,
        "error": error,
        "phases": marks,
        "losses": probe.losses,
        "step_times": probe.step_times,
        "prewarm_batches": decision,
        "observed": observed,
        # Unsloth's chat-template render, which BOTH arms do eagerly.
        "format_seconds": round(
            marks.get("dataset_formatted", 0.0) - marks.get("model_ready", 0.0), 4
        ),
        # Trainer construction: TRL's tokenizing map on the eager arm, nothing online.
        "prep_seconds": round(
            marks.get("trainer_built", 0.0) - marks.get("dataset_formatted", 0.0), 4
        ),
        "time_to_first_step": marks.get("first_step_end"),
        "steady_state_seconds": (
            round(probe.step_times[-1] - probe.step_times[0], 4)
            if len(probe.step_times) > 1
            else None
        ),
    }
    if probe.losses:
        payload["mean_loss"] = round(sum(probe.losses) / len(probe.losses), 6)

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "step_times"}, indent = 2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
