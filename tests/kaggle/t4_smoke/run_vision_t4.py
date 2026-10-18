# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Images must reach the batch and the vision tower, else this is a text run in costume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from determinism import set_all_seeds_fast  # noqa: E402
from training_evidence import LORA_B_MARKER, LORA_MARKER  # noqa: E402
from versions import flatten_versions, resolved_versions  # noqa: E402

SEED = 3407
INSTRUCTION = "Write the LaTeX representation for this image."


def _log(msg: str) -> None:
    print(f"[vision] {msg}", flush = True)


def build_conversations(dataset) -> list:
    """The notebook's own `convert_to_conversation`, unchanged in shape."""
    out = []
    for sample in dataset:
        out.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INSTRUCTION},
                            {"type": "image", "image": sample["image"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sample["text"]}],
                    },
                ]
            }
        )
    return out


def conversation_dataset(dataset):
    """with_transform keeps PIL images; Dataset.from_list makes dicts, and TRL 1.x rejects a plain list."""

    def _transform(batch):
        rows = [
            {"image": image, "text": text} for image, text in zip(batch["image"], batch["text"])
        ]
        return {"messages": [row["messages"] for row in build_conversations(rows)]}

    return dataset.with_transform(_transform)


def pixel_evidence(trainer) -> dict:
    """Read off a real collated batch: a collator that drops images still trains and proves nothing."""
    record: dict = {}
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        record["columns"] = sorted(batch.keys())
        # Pixel key names vary by processor family, so match any key containing "pixel".
        pixel_keys = [k for k in batch if "pixel" in k.lower()]
        record["pixel_keys"] = pixel_keys
        sizes = {}
        for key in pixel_keys:
            value = batch[key]
            if hasattr(value, "numel"):
                sizes[key] = {
                    "numel": int(value.numel()),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        record["pixel_sizes"] = sizes
        labels = batch.get("labels")
        if labels is not None and hasattr(labels, "numel"):
            record["label_tokens"] = int(labels.numel())
            record["masked_tokens"] = int((labels == -100).sum())
    except BaseException as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return record


def vision_lora_evidence(model) -> dict:
    """The finetune_vision_layers flag is only a request, so count LoRA modules under the vision tower."""
    record: dict = {"vision_modules": [], "language_modules": 0}
    try:
        vision_hits, language_hits = [], 0
        for name, _module in model.named_modules():
            if LORA_MARKER not in name:
                continue
            lowered = name.lower()
            if any(tag in lowered for tag in ("visual", "vision", "image_tower", "vision_tower")):
                vision_hits.append(name)
            else:
                language_hits += 1
        record["vision_modules"] = vision_hits[:8]
        record["vision_module_count"] = len(vision_hits)
        record["language_modules"] = language_hits
    except BaseException as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def adapter_sum(model) -> dict:
    """Carries the tensor count, since zero over zero tensors reads the same as a real zero sum."""
    import torch

    total = 0.0
    tensors = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if LORA_B_MARKER in name.lower():
                tensors += 1
                total += float(param.detach().abs().sum())
    return {"sum": total, "tensors": tensors}


def vision_failures(result: dict, args) -> list:
    """The pass rule, pure so it is checkable on CPU without a model."""
    failures = []

    pixels = result.get("pixels") or {}
    if pixels.get("error"):
        failures.append(f"could not read the collated batch: {pixels['error']}")
    else:
        sizes = pixels.get("pixel_sizes") or {}
        total = sum(entry.get("numel", 0) for entry in sizes.values())
        if not sizes:
            failures.append(
                f"the collated batch carried no pixel tensor at all (columns: "
                f"{pixels.get('columns')}), so this trained on text and the "
                f"vision path was never exercised"
            )
        elif total <= 0:
            failures.append(
                f"the pixel tensors are empty ({sizes}), so the images were "
                f"dropped somewhere between the dataset and the collator"
            )

    lora = result.get("vision_lora") or {}
    if args.require_vision_lora:
        if lora.get("error"):
            failures.append(f"could not inspect the adapter: {lora['error']}")
        elif not lora.get("vision_module_count"):
            failures.append(
                "finetune_vision_layers was requested and no LoRA module landed "
                "under the vision tower, so only the language half is being "
                "trained"
            )

    metrics = result.get("metrics") or []
    if len(metrics) != args.max_steps:
        failures.append(f"expected {args.max_steps} logged steps, got {len(metrics)}")
    losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
    if not losses:
        failures.append("no loss was logged, so nothing trained")
    bad = [v for v in losses if v != v or v in (float("inf"), float("-inf"))]
    if bad:
        failures.append(f"non-finite loss: {losses}")

    update = result.get("adapter_update") or {}
    if not update.get("tensors"):
        # A marker matching nothing sums to zero both times, which looks like an untrained adapter.
        failures.append(
            f"no parameter name carried the LoRA B marker {LORA_B_MARKER!r}, so "
            f"the adapter question was never asked rather than answered no"
        )
    elif not update.get("changed"):
        failures.append(
            f"the LoRA B matrices did not move ({update.get('before')} -> "
            f"{update.get('after')}), so the optimizer applied nothing and "
            f"every number above is compatible with zero gradients"
        )

    generated = result.get("generated")
    if generated is None:
        failures.append("vision inference never ran")
    elif not str(generated).strip():
        failures.append("vision inference returned empty output for a real image")

    if args.export:
        export = result.get("export") or {}
        if not export.get("ok"):
            failures.append(f"the vision export failed: {export.get('error')}")
        elif not export.get("files"):
            failures.append(f"the export reported ok and wrote nothing to {export.get('dir')!r}")

    return failures


def run(args) -> dict:
    import torch
    from unsloth import FastVisionModel

    result: dict = {"label": args.label, "model": args.model}
    set_all_seeds_fast(SEED)

    t0 = time.time()
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    result["load_seconds"] = round(time.time() - t0, 1)
    result["resolved_checkpoint"] = getattr(getattr(model, "config", None), "_name_or_path", None)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = True,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = args.lora_r,
        lora_alpha = args.lora_r,
        lora_dropout = 0,
        bias = "none",
        random_state = SEED,
    )
    result["vision_lora"] = vision_lora_evidence(model)
    _log(f"vision lora: {json.dumps(result['vision_lora'])}")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split = f"train[:{args.samples}]")
    conversations = conversation_dataset(dataset)
    result["samples"] = len(conversations)
    # The image must survive as PIL; Dataset.from_list silently turns it into an Arrow dict.
    try:
        first_image = conversations[0]["messages"][0]["content"][1]["image"]
        result["dataset_image_type"] = type(first_image).__name__
    except Exception as exc:  # noqa: BLE001
        result["dataset_image_type"] = f"error: {type(exc).__name__}"

    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = conversations,
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_steps = 0,
            max_steps = args.max_steps,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = SEED,
            output_dir = str(Path(args.outdir) / "trainer"),
            report_to = "none",
            save_strategy = "no",
            # Vision settings from the notebook: without these TRL tokenises the images or drops them
            # before the collator, silently turning this into a text run.
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = args.max_seq_length,
        ),
    )

    # Before training: train() consumes the dataloader.
    result["pixels"] = pixel_evidence(trainer)
    _log(f"pixels: {json.dumps(result['pixels'])}")

    before = adapter_sum(model)
    t0 = time.time()
    stats = trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)
    after = adapter_sum(model)
    result["adapter_update"] = {
        "before": before["sum"],
        "after": after["sum"],
        "tensors": after["tensors"],
        # Starts at exactly zero, so any increase is a real optimizer step.
        "changed": after["sum"] > before["sum"],
    }
    result["metrics"] = [
        {"step": e.get("step"), "loss": e.get("loss"), "grad_norm": e.get("grad_norm")}
        for e in trainer.state.log_history
        if "loss" in e
    ]
    result["train_metrics"] = dict(stats.metrics or {})
    result["memory_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

    FastVisionModel.for_inference(model)
    image = dataset[0]["image"]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    # Image positional first: this processor's signature is (images, text, ...).
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
            use_cache = True,
            do_sample = False,
        )
    result["infer_seconds"] = round(time.time() - t0, 1)
    width = inputs["input_ids"].shape[1]
    result["generated"] = tokenizer.decode(out[0][width:], skip_special_tokens = True)
    _log(f"generated: {result['generated']!r}")

    if args.export:
        # To /tmp: /kaggle/working is only 21GB.
        import tempfile

        export_dir = tempfile.mkdtemp(prefix = "vision_export_")
        record: dict = {"dir": export_dir}
        try:
            model.save_pretrained_merged(export_dir, tokenizer)
            record["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        files = []
        for path in sorted(Path(export_dir).rglob("*")):
            if path.is_file():
                files.append({"name": path.name, "mb": round(path.stat().st_size / 1024**2, 1)})
        record["files"] = files[:20]
        record["file_count"] = len(files)
        result["export"] = record
        _log(f"export: {json.dumps({k: v for k, v in record.items() if k != 'files'})}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "unsloth/Qwen3.5-2B")
    ap.add_argument("--dataset", default = "unsloth/LaTeX_OCR")
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--label", default = "vision")
    ap.add_argument("--samples", type = int, default = 16)
    ap.add_argument("--max-steps", type = int, default = 5)
    ap.add_argument("--max-seq-length", type = int, default = 2048)
    ap.add_argument("--lora-r", type = int, default = 16)
    ap.add_argument("--max-new-tokens", type = int, default = 32)
    ap.add_argument("--export", action = "store_true", default = False)
    ap.add_argument(
        "--require-vision-lora",
        dest = "require_vision_lora",
        action = "store_true",
        default = True,
    )
    ap.add_argument("--no-require-vision-lora", dest = "require_vision_lora", action = "store_false")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    # Environment before the run so a crash still reports versions; run_t4_smoke is not imported
    # because that pulls its whole payload.
    env: dict = {"python": sys.version.split()[0]}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
            env["gpu_count_visible"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        env["error"] = f"{type(exc).__name__}: {exc}"[:300]

    result: dict = {"label": args.label, "model": args.model, "environment": env}
    try:
        result["versions_flat"] = flatten_versions(resolved_versions())
    except Exception:  # noqa: BLE001
        pass

    try:
        result.update(run(args))
        failures = vision_failures(result, args)
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        failures = [f"the vision run raised: {result['error']}"]

    result["failures"] = failures
    result["passed"] = not failures
    (outdir / "vision_report.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    print("T4_SMOKE_REPORT " + json.dumps(result), flush = True)
    _log("T4_SMOKE_RESULT " + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
