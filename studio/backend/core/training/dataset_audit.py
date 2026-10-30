# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tokenizer-only audits of unpacked text training data."""

from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from itertools import islice


def _select_rows(dataset, max_rows):
    from datasets import Dataset

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive, or None for a full finite dataset.")
    total = len(dataset) if hasattr(dataset, "__len__") else None
    if total is not None:
        selected = dataset if max_rows is None else dataset.select(range(min(total, max_rows)))
        return selected, {
            "rows_scanned": len(selected),
            "total_rows": total,
            "complete": len(selected) == total,
        }
    if max_rows is None:
        raise ValueError("A streaming dataset requires a row limit; --full needs a finite dataset.")
    # Do not consume a lookahead row: it can start a download or advance a one-shot source.
    rows = list(islice(iter(dataset), max_rows))
    return Dataset.from_list(rows, features = getattr(dataset, "features", None)), {
        "rows_scanned": len(rows),
        "total_rows": None,
        "complete": len(rows) < max_rows,
    }


def _length_summary(counts):
    total = sum(counts.values())
    if not total:
        return {"min": None, "max": None, "mean": None, "p50": None, "p95": None}
    ordered = sorted(counts)

    def percentile(fraction):
        target = math.ceil(total * fraction)
        seen = 0
        for length in ordered:
            seen += counts[length]
            if seen >= target:
                return length

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(length * count for length, count in counts.items()) / total,
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def _label_rows(tokenizer, model_name, dataset, final_format, completion_only):
    from core.training.token_preview_worker import _mask
    from utils.datasets.pretokenized import _checked_masks

    rows = dataset.to_list()
    ids = [row["input_ids"] for row in rows]
    custom = "labels" in dataset.column_names
    masking = {"applied": False, "notes": []}
    if completion_only and not custom and final_format != "raw_text":
        labels, masking = _mask(tokenizer, model_name, {"input_ids": ids}, final_format)
    else:
        labels = [row.get("labels") for row in rows]
    # Use the same TRL-version-dependent mask policy as pre-tokenized preparation.
    masks = _checked_masks(dataset.column_names) if final_format == "pretokenized" else ()
    result = []
    for row, row_ids, row_labels in zip(rows, ids, labels):
        if (
            row_labels is None
            and final_format == "pretokenized"
            and not any(
                name in dataset.column_names for name in ("completion_mask", "assistant_masks")
            )
        ):
            # Zoo selects transformers' LM collator for bare IDs, which masks every pad-token ID.
            # Custom/response labels use the seq2seq collator and retain EOS even when PAD == EOS.
            pad_id = tokenizer.pad_token_id
            row_labels = [token if token != pad_id else -100 for token in row_ids]
        row_labels = list(row_ids if row_labels is None else row_labels)
        for name in masks:
            row_labels = [label if keep else -100 for label, keep in zip(row_labels, row[name])]
        result.append(row_labels)
    return result, masking


def audit_dataset(
    dataset,
    tokenizer,
    config: dict,
    *,
    max_rows = 1000,
    batch_size = 64,
) -> dict:
    """Audit all selected rows; finite Arrow datasets stay on disk while tokenization is batched.

    A bounded scan is a prefix, not a random or representative sample. Loss targets exclude
    position zero of each unpacked sequence, matching the causal label shift.
    """
    from datasets import Dataset
    from core.training.token_preview_worker import prepare_text_dataset
    from utils.datasets.chat_templates import DEFAULT_ALPACA_TEMPLATE, get_training_chat_template
    from utils.datasets.format_detection import detect_multimodal_dataset
    from utils.datasets.online_tokenization import resolve_add_special_tokens
    from utils.datasets.pretokenized import prepare_pretokenized_dataset

    if config.get("packing"):
        raise ValueError(
            "Dataset audit currently supports packing=False. Packed training needs the packer's loss boundaries; unpacked counts would be misleading."
        )
    max_length = int(config.get("max_seq_length", 2048))
    if max_length < 1 or batch_size < 1:
        raise ValueError("max_seq_length and batch_size must be positive.")
    selected, coverage = _select_rows(dataset, max_rows)
    coverage.update(mode = "full" if max_rows is None else "prefix", row_limit = max_rows)
    if not len(selected):
        raise ValueError("The selected dataset contains no rows.")
    media = detect_multimodal_dataset(selected)
    if media.get("is_image") or media.get("is_audio") or media.get("is_video"):
        raise ValueError(
            "Dataset audit currently supports text datasets only; media needs its training processor."
        )

    format_config = dict(config)
    # Explicit raw/CPT adds EOS in the trainer; auto-detected raw uses the formatter as-is.
    format_config["raw_text"] = config.get("is_cpt") or config.get("format_type") == "raw"
    formatting_tokenizer = copy.deepcopy(tokenizer)
    info = prepare_text_dataset(formatting_tokenizer, format_config, selected)
    prepared = info["dataset"]
    final_format = str(info["final_format"]).lower()
    if not len(prepared):
        raise ValueError("Formatting removed every row; there is no training data to audit.")
    tokenizer = get_training_chat_template(formatting_tokenizer, config["model_name"], final_format)
    pretokenized = final_format == "pretokenized"
    template = (
        None
        if final_format in ("raw_text", "pretokenized")
        else (
            DEFAULT_ALPACA_TEMPLATE
            if final_format == "alpaca"
            else getattr(formatting_tokenizer, "chat_template", None)
        )
    )
    first_text = None if pretokenized else prepared[0]["text"]
    add_special_tokens = (
        False if pretokenized else resolve_add_special_tokens(tokenizer, first_text)
    )
    counts = Counter()
    lengths = Counter()
    examples = []
    masking = {"applied": False, "notes": []}
    for start in range(0, len(prepared), batch_size):
        batch = prepared.select(range(start, min(start + batch_size, len(prepared))))
        if pretokenized:
            full = prepare_pretokenized_dataset(batch, max_seq_length = None)
            kept = prepare_pretokenized_dataset(full, max_seq_length = max_length)
        else:
            texts = list(batch["text"])
            full = Dataset.from_dict(dict(tokenizer(texts, add_special_tokens = add_special_tokens)))
            kept = Dataset.from_dict(
                dict(
                    tokenizer(
                        texts,
                        truncation = True,
                        max_length = max_length,
                        add_special_tokens = add_special_tokens,
                    )
                )
            )
        full_labels, _ = _label_rows(
            tokenizer,
            config["model_name"],
            full,
            final_format,
            config.get("train_on_completions", False),
        )
        kept_labels, masking = _label_rows(
            tokenizer,
            config["model_name"],
            kept,
            final_format,
            config.get("train_on_completions", False),
        )
        for offset, (ids, retained, before, after) in enumerate(
            zip(full["input_ids"], kept["input_ids"], full_labels, kept_labels)
        ):
            length, retained_length = len(ids), len(retained)
            before_targets = sum(label != -100 for label in before[1:])
            after_targets = sum(label != -100 for label in after[1:])
            lengths[length] += 1
            counts.update(
                tokens_before_truncation = length,
                tokens_after_truncation = retained_length,
                loss_targets_before_truncation = before_targets,
                loss_targets_after_truncation = after_targets,
                rows_over_context = int(length > max_length),
                rows_without_loss_before_truncation = int(before_targets == 0),
                rows_without_loss_after_truncation = int(after_targets == 0),
                rows_losing_all_loss_to_truncation = int(before_targets > 0 and after_targets == 0),
            )
            if len(examples) < 10 and (length > max_length or after_targets == 0):
                examples.append(
                    {
                        "prepared_row": start + offset + 1,
                        "full_length": length,
                        "retained_length": retained_length,
                        "loss_targets": after_targets,
                    }
                )
    dropped = len(selected) - len(prepared)
    warnings = list(
        dict.fromkeys(
            [
                *info.get("warnings", []),
                *([info["dropped_rows_warning"]] if info.get("dropped_rows_warning") else []),
                *masking.get("notes", []),
            ]
        )
    )
    return {
        "schema_version": 1,
        "status": "issues_found"
        if dropped or counts["rows_without_loss_after_truncation"]
        else "ok",
        "coverage": coverage,
        "model": config["model_name"],
        "tokenizer_source": config.get("tokenizer_source") or config["model_name"],
        "format": final_format,
        "settings": {
            "max_seq_length": max_length,
            "packing": False,
            "train_on_completions": bool(config.get("train_on_completions")),
            "add_special_tokens": add_special_tokens,
        },
        "template": {
            "source": "none"
            if template is None
            else ("alpaca" if final_format == "alpaca" else "training chat template"),
            "text": template,
            "sha256": hashlib.sha256(template.encode()).hexdigest()
            if isinstance(template, str)
            else None,
        },
        "masking": masking,
        "rows_prepared": len(prepared),
        "rows_dropped_by_formatter": dropped,
        "lengths": _length_summary(lengths),
        **counts,
        "examples": examples,
        "warnings": warnings,
        "loss_target_definition": "Non-ignored labels after excluding the first token of each unpacked row.",
    }


def load_audit_dataset(config: dict, *, full: bool):
    from pathlib import Path
    from utils.datasets.cache_safe import load_dataset_cache_safe
    from utils.datasets.cells import csv_as_text_kwargs
    from utils.datasets.saved import load_saved_dataset
    from utils.paths import resolve_dataset_path

    local = config.get("local_datasets")
    if not local:
        return load_dataset_cache_safe(
            config["dataset_name"], split = "train", token = config.get("hf_token"), streaming = not full
        )
    paths = [Path(value).expanduser() for value in local]
    paths = [
        path if path.is_absolute() or path.exists() else Path(resolve_dataset_path(str(path)))
        for path in paths
    ]
    if len(paths) == 1:
        saved = load_saved_dataset(paths[0])
        if saved is not None:
            return saved
    from utils.paths import dataset_files_in_dir

    files = [
        file for path in paths for file in (dataset_files_in_dir(path) if path.is_dir() else [path])
    ]
    loaders = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
    kinds = {loaders.get(path.suffix.lower()) for path in files}
    if len(kinds) != 1 or None in kinds:
        raise ValueError(
            "Choose JSON/JSONL, CSV or Parquet files of one format, or a saved dataset folder."
        )
    files = [str(path) for path in files]
    return load_dataset_cache_safe(
        kinds.pop(),
        data_files = files,
        split = "train",
        streaming = not full,
        **csv_as_text_kwargs(files),
    )


def _main():
    import contextlib
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    request = json.load(sys.stdin)
    config = request["config"]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from utils.transformers_version import activate_transformers_for_subprocess

            model_path = Path(config["model_name"])
            # A tokenizer-only folder has no model architecture for sidecar config probes.
            if not model_path.is_dir() or (model_path / "config.json").is_file():
                activate_transformers_for_subprocess(config["model_name"], config.get("hf_token"))
            from core.training.token_preview_worker import _load_tokenizer
            from utils.models.unsloth_mirror import unsloth_public_mirror
            from datasets import disable_progress_bars

            disable_progress_bars()
            config["tokenizer_source"] = (
                unsloth_public_mirror(config["model_name"], config.get("load_in_4bit", True))
                or config["model_name"]
            )
            tokenizer = _load_tokenizer(config)
            dataset = load_audit_dataset(config, full = request["full"])
            result = audit_dataset(
                dataset,
                tokenizer,
                config,
                max_rows = None if request["full"] else request["max_rows"],
            )
        print(json.dumps(result, ensure_ascii = False))
    except Exception as exc:
        from hub.utils.download_registry import scrub_secrets

        detail = str(exc)
        if exc.__cause__ is not None:
            detail += f": {exc.__cause__}"
        print(json.dumps({"error": scrub_secrets(detail, hf_token = config.get("hf_token"))}))
        return 2
    return 1 if result["status"] == "issues_found" else 0


if __name__ == "__main__":
    raise SystemExit(_main())
