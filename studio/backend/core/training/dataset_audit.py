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


def _label_rows(
    tokenizer,
    model_name,
    dataset,
    final_format,
    completion_only,
    *,
    retained_tokens = True,
    completion_loss = None,
):
    from types import SimpleNamespace
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from unsloth_zoo.dataset_utils import sft_prepare_dataset
    from core.training.token_preview_worker import _mask
    from utils.datasets.pretokenized import resolve_completion_only_loss

    user_labels = final_format == "pretokenized" and "labels" in dataset.column_names
    if completion_loss is None:
        completion_loss = resolve_completion_only_loss(dataset)
    trainer = SimpleNamespace(
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id = tokenizer.pad_token_id,
            completion_only_loss = completion_loss,
        ),
        completion_only_loss = completion_loss,
    )
    if retained_tokens:
        # Zoo skips tokenization/truncation for retained IDs and selects their actual collator.
        dataset = sft_prepare_dataset(
            trainer,
            dataset,
            tokenizer,
            SimpleNamespace(max_length = 1, dataset_num_proc = 1),
            False,
            None,
            "audit",
        )
    rows = dataset.to_list()
    masking = {"applied": False, "notes": []}
    if completion_only and not user_labels and final_format != "raw_text":
        labels, masking = _mask(
            tokenizer, model_name, {"input_ids": [row["input_ids"] for row in rows]}, final_format
        )
        if masking["applied"]:
            return [
                [
                    old if mask != -100 else -100
                    for old, mask in zip(row.get("labels", row["input_ids"]), masked)
                ]
                for row, masked in zip(rows, labels)
            ], masking
    columns = {
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "completion_mask",
        "assistant_masks",
    }
    # Collate one row at a time so an unusually long row cannot pad an entire audit batch.
    return [
        trainer.data_collator([{key: value for key, value in row.items() if key in columns}])[
            "labels"
        ][0].tolist()
        for row in rows
    ], masking


def _prepare_prompt_completion(dataset, tokenizer, max_length):
    from types import SimpleNamespace
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from unsloth_zoo.dataset_utils import sft_prepare_dataset

    trainer = SimpleNamespace(
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id = tokenizer.pad_token_id, completion_only_loss = True
        ),
        completion_only_loss = True,
    )
    # Zoo's prompt/completion route treats a non-positive length as no truncation.
    return sft_prepare_dataset(
        trainer,
        dataset,
        tokenizer,
        SimpleNamespace(max_length = max_length, dataset_num_proc = 1),
        False,
        None,
        "audit",
    )


def _load_training_tokenizer(config):
    from pathlib import Path
    from transformers import AutoConfig
    from unsloth.tokenizer_utils import load_correct_tokenizer
    from unsloth_zoo.pad_token import fix_pad_token

    source = config.get("tokenizer_source") or config["model_name"]
    model_config = None
    if not Path(source).is_dir() or (Path(source) / "config.json").is_file():
        model_config = AutoConfig.from_pretrained(
            source, token = config.get("hf_token"), trust_remote_code = False
        )
    tokenizer = load_correct_tokenizer(
        source,
        model_max_length = config.get("max_seq_length", 2048),
        token = config.get("hf_token"),
        trust_remote_code = False,
        config = model_config,
    )
    fix_pad_token(tokenizer, model_config = model_config, allow_add = False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "The training pad token cannot be resolved without model preparation; this tokenizer cannot be audited yet."
        )
    return tokenizer


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
    from utils.datasets.media import media_columns_hold_media
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
    has_image, has_audio = media_columns_hold_media(selected, media)
    if has_image or has_audio or media.get("is_video"):
        raise ValueError(
            "Dataset audit currently supports text datasets only; media needs its training processor."
        )

    format_config = dict(config)
    # Explicit raw/CPT adds EOS in the trainer; auto-detected raw uses the formatter as-is.
    format_config["raw_text"] = config.get("is_cpt") or config.get("format_type") == "raw"
    formatting_tokenizer = copy.deepcopy(tokenizer)
    info = prepare_text_dataset(
        formatting_tokenizer, format_config, selected, collect_row_statistics = True
    )
    prepared = info["dataset"]
    final_format = str(info["final_format"]).lower()
    if not len(prepared):
        raise ValueError("Formatting removed every row; there is no training data to audit.")
    tokenizer = get_training_chat_template(formatting_tokenizer, config["model_name"], final_format)
    pretokenized = final_format == "pretokenized"
    retained_tokens = "input_ids" in prepared.column_names
    prompt_completion = not retained_tokens and {"prompt", "completion"} <= set(
        prepared.column_names
    )
    plain_prompt = prompt_completion and isinstance(prepared[0]["prompt"], str)
    template = (
        None
        if final_format == "raw_text" or retained_tokens or plain_prompt
        else (
            DEFAULT_ALPACA_TEMPLATE
            if final_format == "alpaca"
            else getattr(tokenizer, "chat_template", None)
        )
    )
    first_text = None if retained_tokens else prepared[0]["text"]
    if prompt_completion:
        first_text = prepared[0]["prompt"] if plain_prompt else None
    add_special_tokens = (
        False if retained_tokens else resolve_add_special_tokens(tokenizer, first_text)
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
        elif retained_tokens:
            full = kept = batch
        elif prompt_completion:
            # Zoo chooses BOS handling from the dataset head, not each audit batch's head.
            source = prepared.select([0, *range(start, start + len(batch))])
            full = _prepare_prompt_completion(source, tokenizer, -1)
            kept = _prepare_prompt_completion(source, tokenizer, max_length)
            full = full.select(range(1, len(full)))
            kept = kept.select(range(1, len(kept)))
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
            retained_tokens = retained_tokens,
            completion_loss = True if prompt_completion else None,
        )
        kept_labels, masking = _label_rows(
            tokenizer,
            config["model_name"],
            kept,
            final_format,
            config.get("train_on_completions", False),
            retained_tokens = retained_tokens,
            completion_loss = True if prompt_completion else None,
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
    row_statistics = info.get("row_statistics", {})
    expanded = row_statistics.get("expanded_rows", 0)
    dropped = len(selected) + expanded - len(prepared)
    warnings = list(
        dict.fromkeys(
            [
                *info.get("warnings", []),
                *([info["dropped_rows_warning"]] if info.get("dropped_rows_warning") else []),
                *(
                    [
                        "Existing input_ids take precedence over formatted text; training leaves them untruncated on this format."
                    ]
                    if retained_tokens and not pretokenized
                    else []
                ),
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
        "tokenization": {
            "source": "input_ids"
            if retained_tokens
            else ("prompt_completion" if prompt_completion else "text"),
            "context_limit_applied": pretokenized or not retained_tokens,
        },
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
        "rows_expanded_by_formatter": expanded,
        "source_rows_dropped_during_conversion": row_statistics.get("dropped_source_rows", 0),
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
            from utils.models.unsloth_mirror import unsloth_public_mirror
            from datasets import disable_progress_bars

            disable_progress_bars()
            config["tokenizer_source"] = (
                unsloth_public_mirror(config["model_name"], config.get("load_in_4bit", True))
                or config["model_name"]
            )
            tokenizer = _load_training_tokenizer(config)
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
