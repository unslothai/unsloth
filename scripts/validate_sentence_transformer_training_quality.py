#!/usr/bin/env python3
"""Validate SentenceTransformer training quality on a pinned all-NLI subset.

Run the baseline and optimized modes in separate processes.  This is required
because importing Unsloth installs process-wide SentenceTransformer patches::

    python scripts/validate_sentence_transformer_training_quality.py \
        --mode baseline --output .benchmarks/quality-baseline.json
    python scripts/validate_sentence_transformer_training_quality.py \
        --mode optimized --output .benchmarks/quality-optimized.json

The script deliberately uses a reference ``AutoTokenizer`` for both modes and
emits hashes for the selected rows, scheduled examples, token tensors, and
initial trainable parameters.  Compare those hashes before comparing the
reported quality metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_ID = "sentence-transformers/all-nli"
DATASET_CONFIG = "triplet"
DATASET_REVISION = "d482672c8e74ce18da116f430137434ba2e52fab"
DATASET_SELECTION_SEED = 3407
FULL_TRAIN_LIMIT = 8_192
FULL_EVAL_LIMIT = 1_024
TEXT_COLUMNS = ("anchor", "positive", "negative")

# These were reported by the dataset-inventory run.  Its digest serialization
# was not retained, so they are provenance only; the validator emits its own
# documented canonical row and index digests below.
INVENTORY_REFERENCE_DIGESTS = {
    "serialization": "unspecified; provenance only, not asserted",
    "train_8192": "0d8db2c5341c1998b664551748718dc0a8eb8d0a116820023935fefe3b1862b3",
    "eval_1024": "4cc754085514fc7896ae7f3f44f56f889f5dc90c28a5643b3b03f26719446368",
    "train_first_512": "94478d890a5c50a63e85e3527edcfc16f337897018a71a45bd71958c837df015",
    "eval_first_256": "238da7c621a7a762ba9b25feec38e8eb2024214ba6552823b7aeac376f23f9d3",
}

FeatureDict = dict[str, torch.Tensor]
FeatureColumns = list[FeatureDict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--mode", choices = ("baseline", "optimized"), required = True)
    parser.add_argument("--model", default = DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--seed", type = int, default = 3407)
    parser.add_argument("--steps", type = int, default = 100)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--eval-batch-size", type = int)
    parser.add_argument("--max-length", type = int, default = 128)
    parser.add_argument("--train-limit", type = int, default = FULL_TRAIN_LIMIT)
    parser.add_argument("--eval-limit", type = int, default = FULL_EVAL_LIMIT)
    parser.add_argument("--learning-rate", type = float, default = 2e-5)
    parser.add_argument("--weight-decay", type = float, default = 0.01)
    parser.add_argument(
        "--fused-optimizer",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Use the current SentenceTransformer default fused AdamW implementation.",
    )
    parser.add_argument(
        "--dtype",
        choices = ("float16", "bfloat16", "float32"),
        default = "bfloat16",
    )
    parser.add_argument("--offline", action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument(
        "--model-prompts",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Apply shipped query/document prompts to the corresponding columns.",
    )
    parser.add_argument("--output", type = Path)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be at least 1")
    if not 1 <= args.train_limit <= FULL_TRAIN_LIMIT:
        parser.error(f"--train-limit must be in [1, {FULL_TRAIN_LIMIT}]")
    if not 1 <= args.eval_limit <= FULL_EVAL_LIMIT:
        parser.error(f"--eval-limit must be in [1, {FULL_EVAL_LIMIT}]")
    if args.train_limit < args.batch_size:
        parser.error("--train-limit must be at least --batch-size")
    if args.max_length < 1:
        parser.error("--max-length must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_row(row: dict[str, str]) -> bytes:
    payload = {column: row[column] for column in TEXT_COLUMNS}
    return json.dumps(
        payload,
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
    ).encode("utf-8")


def content_digest(rows: Sequence[dict[str, str]]) -> str:
    """Hash canonical UTF-8 JSON lines, including the final newline."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_row(row))
        digest.update(b"\n")
    return digest.hexdigest()


def index_digest(indices: Sequence[int]) -> str:
    """Hash a compact JSON array of integer source-row indices."""
    payload = json.dumps(list(indices), separators = (",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def selection_rank(split: str, index: int) -> bytes:
    key = (
        f"{DATASET_ID}@{DATASET_REVISION}/{DATASET_CONFIG}/{split}/{DATASET_SELECTION_SEED}/{index}"
    )
    return hashlib.sha256(key.encode("utf-8")).digest()


def load_dataset_contract(offline: bool) -> dict[str, Any]:
    """Load the pinned parquet files and construct deterministic subsets."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    parquet_paths = {
        split: hf_hub_download(
            repo_id = DATASET_ID,
            repo_type = "dataset",
            filename = f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet",
            revision = DATASET_REVISION,
            local_files_only = offline,
        )
        for split in ("train", "dev")
    }
    source = {
        split: Dataset.from_parquet(path, columns = list(TEXT_COLUMNS))
        for split, path in parquet_paths.items()
    }

    train_indices = heapq.nsmallest(
        FULL_TRAIN_LIMIT,
        range(len(source["train"])),
        key = lambda index: selection_rank("train", index),
    )
    full_train_rows = [source["train"][index] for index in train_indices]
    train_texts = {
        text for row in full_train_rows for column in TEXT_COLUMNS if (text := row[column])
    }

    eval_indices: list[int] = []
    eval_anchors: set[str] = set()
    ranked_dev_indices = heapq.nsmallest(
        len(source["dev"]),
        range(len(source["dev"])),
        key = lambda index: selection_rank("dev", index),
    )
    for index in ranked_dev_indices:
        row = source["dev"][index]
        if any(row[column] in train_texts for column in TEXT_COLUMNS):
            continue
        if row["anchor"] in eval_anchors:
            continue
        eval_anchors.add(row["anchor"])
        eval_indices.append(index)
        if len(eval_indices) == FULL_EVAL_LIMIT:
            break
    if len(eval_indices) != FULL_EVAL_LIMIT:
        raise RuntimeError(
            f"Expected {FULL_EVAL_LIMIT} leakage-free dev rows, found {len(eval_indices)}."
        )
    full_eval_rows = [source["dev"][index] for index in eval_indices]

    return {
        "train_rows": full_train_rows,
        "eval_rows": full_eval_rows,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "source_sizes": {split: len(dataset) for split, dataset in source.items()},
        "full_train_content_sha256": content_digest(full_train_rows),
        "full_eval_content_sha256": content_digest(full_eval_rows),
        "full_train_indices_sha256": index_digest(train_indices),
        "full_eval_indices_sha256": index_digest(eval_indices),
    }


def lora_targets(model: Any) -> list[str]:
    model_type = str(getattr(model[0].auto_model.config, "model_type", "")).lower()
    if model_type == "distilbert":
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    if model_type == "mpnet":
        return ["q", "k", "v", "o", "dense"]
    if model_type == "modernbert":
        return ["Wqkv", "Wo", "Wi"]
    if model_type in {
        "qwen2",
        "qwen3",
        "llama",
        "mistral",
        "gemma",
        "gemma2",
        "gemma3_text",
    }:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["query", "key", "value", "dense"]


def load_model(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    """Load identical rank-16 feature-extraction LoRA configurations."""
    if args.mode == "baseline":
        from peft import LoraConfig, get_peft_model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.model,
            device = str(device),
            local_files_only = args.offline,
            revision = args.revision,
            model_kwargs = {"dtype": dtype},
        )
        targets = lora_targets(model)
        lora_config = LoraConfig(
            r = 16,
            lora_alpha = 16,
            target_modules = targets,
            lora_dropout = 0.0,
            bias = "none",
            task_type = "FEATURE_EXTRACTION",
            use_rslora = False,
            init_lora_weights = True,
        )
        seed_everything(args.seed)
        peft_model = get_peft_model(model[0].auto_model, lora_config)
        if isinstance(getattr(type(model[0]), "auto_model", None), property):
            model[0].model = peft_model
        else:
            model[0].auto_model = peft_model
        implementation_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
        runtime = {
            "implementation_file": implementation_file,
            "fused_lora_layers": 0,
            "fused_lora_linears": 0,
        }
    else:
        from unsloth import FastSentenceTransformer

        model = FastSentenceTransformer.from_pretrained(
            args.model,
            max_seq_length = args.max_length,
            dtype = dtype,
            load_in_16bit = True,
            device_map = str(device),
            revision = args.revision,
            local_files_only = args.offline,
        )
        targets = lora_targets(model)
        seed_everything(args.seed)
        model = FastSentenceTransformer.get_peft_model(
            model,
            r = 16,
            lora_alpha = 16,
            lora_dropout = 0.0,
            bias = "none",
            target_modules = targets,
            use_gradient_checkpointing = False,
            random_state = args.seed,
            use_rslora = False,
            init_lora_weights = True,
        )

        # The quality gate intentionally exercises the eager optimized path.
        # torch.compile is benchmarked separately and is not currently the
        # default on Windows.
        from unsloth.models.sentence_transformer import _patch_encoder_attention_lora

        model._compile_pending = False
        inner_model = model[0].auto_model
        _patch_encoder_attention_lora(inner_model)
        fused_layers = getattr(inner_model, "_unsloth_fused_lora_qkv_count", 0)
        import unsloth

        runtime = {
            "implementation_file": str(Path(unsloth.__file__).resolve()),
            "fused_lora_layers": fused_layers,
            "fused_lora_linears": getattr(
                model[0].auto_model,
                "_unsloth_fast_lora_linear_count",
                0,
            ),
        }

    model.max_seq_length = args.max_length
    if hasattr(model, "__getitem__") and hasattr(model[0], "max_seq_length"):
        model[0].max_seq_length = args.max_length
    runtime["target_modules"] = targets
    runtime["lora_config"] = {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
        "use_rslora": False,
        "init_lora_weights": True,
    }
    return model, runtime


def reference_tokenizer(model_name: str, revision: str | None, offline: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision = revision,
        local_files_only = offline,
        trust_remote_code = True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer for {model_name!r} has neither a pad token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def model_prompt_contract(
    model_name: str, revision: str | None, offline: bool, enabled: bool
) -> dict[str, Any]:
    """Load the pinned SentenceTransformer prompt config used by both modes."""
    if not enabled:
        return {
            "enabled": False,
            "query": "",
            "document": "",
            "config_sha256": None,
        }

    config_path: Path | None = None
    local_path = Path(model_name)
    if local_path.is_dir():
        candidate = local_path / "config_sentence_transformers.json"
        if candidate.exists():
            config_path = candidate
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = Path(
                hf_hub_download(
                    repo_id = model_name,
                    filename = "config_sentence_transformers.json",
                    revision = revision,
                    local_files_only = offline,
                )
            )
        except Exception:
            config_path = None

    config: dict[str, Any] = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding = "utf-8"))
    prompts = config.get("prompts", {}) or {}
    query = str(prompts.get("query", "") or "")
    document = ""
    for name in ("document", "passage", "corpus"):
        if name in prompts:
            document = str(prompts[name] or "")
            break
    return {
        "enabled": True,
        "query": query,
        "document": document,
        "config_sha256": sha256_json(config) if config else None,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "query_characters": len(query),
        "document_characters": len(document),
    }


def build_training_schedule(
    source_indices: Sequence[int], steps: int, batch_size: int, seed: int
) -> list[list[int]]:
    """Return source-row index batches using an isolated deterministic RNG."""
    generator = torch.Generator(device = "cpu")
    generator.manual_seed(seed)
    schedule: list[list[int]] = []
    usable = len(source_indices) - (len(source_indices) % batch_size)
    while len(schedule) < steps:
        permutation = torch.randperm(len(source_indices), generator = generator).tolist()
        for offset in range(0, usable, batch_size):
            positions = permutation[offset : offset + batch_size]
            schedule.append([source_indices[position] for position in positions])
            if len(schedule) == steps:
                break
    return schedule


def tokenize_rows(
    tokenizer: Any, rows: Sequence[dict[str, str]], max_length: int, prompts: dict[str, Any]
) -> FeatureColumns:
    columns: FeatureColumns = []
    for column_index, column in enumerate(TEXT_COLUMNS):
        prefix = prompts["query"] if column_index == 0 else prompts["document"]
        features = tokenizer(
            [prefix + row[column] for row in rows],
            padding = True,
            truncation = True,
            max_length = max_length,
            return_tensors = "pt",
        )
        columns.append(dict(features))
    return columns


def hash_feature_batches(batches: Sequence[FeatureColumns]) -> str:
    """Hash exact CPU token tensors with batch/column/key boundaries."""
    digest = hashlib.sha256()
    for batch_index, columns in enumerate(batches):
        digest.update(f"batch:{batch_index}\n".encode("ascii"))
        for column_index, features in enumerate(columns):
            digest.update(f"column:{column_index}\n".encode("ascii"))
            for key in sorted(features):
                value = features[key]
                if not torch.is_tensor(value):
                    continue
                cpu_value = value.detach().cpu().contiguous()
                digest.update(key.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(cpu_value.dtype).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(tuple(cpu_value.shape)).encode("ascii"))
                digest.update(b"\0")
                digest.update(cpu_value.numpy().tobytes())
    return digest.hexdigest()


def prepare_batches(
    tokenizer: Any,
    rows_by_source_index: dict[int, dict[str, str]],
    source_index_batches: Sequence[Sequence[int]],
    max_length: int,
    prompts: dict[str, Any],
) -> list[FeatureColumns]:
    return [
        tokenize_rows(
            tokenizer,
            [rows_by_source_index[index] for index in indices],
            max_length,
            prompts,
        )
        for indices in source_index_batches
    ]


def move_features(features: FeatureDict, device: torch.device) -> FeatureDict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }


def parameter_hash(named_parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_value_hash(named_parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for _, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_config_hash(model: Any) -> str:
    config = model[0].auto_model.config
    if hasattr(config, "to_dict"):
        return sha256_json(config.to_dict())
    return sha256_json(vars(config))


def resolved_model_commit(model: Any) -> str | None:
    """Return the Hugging Face snapshot commit from the wrapped base config."""
    try:
        pending = [model[0].auto_model]
    except (AttributeError, IndexError, TypeError):
        return None

    visited: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        config = getattr(current, "config", None)
        for candidate in (current, config):
            commit = getattr(candidate, "_commit_hash", None)
            if isinstance(commit, str) and commit:
                return commit
        for attribute in ("base_model", "model"):
            child = getattr(current, attribute, None)
            if child is not None and child is not current:
                pending.append(child)
    return None


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total: torch.Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum()
        total = squared if total is None else total + squared
    return 0.0 if total is None else float(total.sqrt())


def snapshot_parameters(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in named_parameters}


def parameter_delta(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]], initial: dict[str, torch.Tensor]
) -> dict[str, float]:
    delta_squared = 0.0
    initial_squared = 0.0
    max_abs_delta = 0.0
    for name, parameter in named_parameters:
        current = parameter.detach().float().cpu()
        reference = initial[name]
        delta = current - reference
        delta_squared += float(delta.square().sum())
        initial_squared += float(reference.square().sum())
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max()))
    delta_l2 = math.sqrt(delta_squared)
    initial_l2 = math.sqrt(initial_squared)
    return {
        "l2_norm": delta_l2,
        "relative_l2_norm": delta_l2 / initial_l2 if initial_l2 else 0.0,
        "max_abs": max_abs_delta,
    }


def evaluate_triplets(
    model: Any, batches: Sequence[FeatureColumns], device: torch.device
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    correct = 0
    examples = 0
    positive_sum = 0.0
    negative_sum = 0.0
    margin_sum = 0.0
    with torch.inference_mode():
        for cpu_columns in batches:
            embeddings = []
            for cpu_features in cpu_columns:
                features = move_features(cpu_features, device)
                embedding = model(features)["sentence_embedding"].float()
                embeddings.append(torch.nn.functional.normalize(embedding, dim = -1))
            anchor, positive, negative = embeddings
            positive_cosine = (anchor * positive).sum(dim = -1)
            negative_cosine = (anchor * negative).sum(dim = -1)
            margin = positive_cosine - negative_cosine
            correct += int((margin > 0).sum())
            examples += int(margin.numel())
            positive_sum += float(positive_cosine.sum())
            negative_sum += float(negative_cosine.sum())
            margin_sum += float(margin.sum())
    synchronize(device)
    if was_training:
        model.train()
    return {
        "examples": examples,
        "triplet_accuracy": correct / examples,
        "mean_margin": margin_sum / examples,
        "mean_positive_cosine": positive_sum / examples,
        "mean_negative_cosine": negative_sum / examples,
    }


def checkpoint_labels(steps: int) -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {1: ["step_1"], steps: ["final"]}
    if steps >= 10:
        labels.setdefault(10, []).append("step_10")
    if steps == 1:
        labels[1] = ["step_1", "final"]
    return labels


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        raise RuntimeError("The quality validator requires CUDA.")
    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    eval_batch_size = args.eval_batch_size or args.batch_size

    total_start = time.perf_counter()
    dataset_start = time.perf_counter()
    dataset_contract = load_dataset_contract(args.offline)
    dataset_seconds = time.perf_counter() - dataset_start

    full_train_rows = dataset_contract.pop("train_rows")
    full_eval_rows = dataset_contract.pop("eval_rows")
    full_train_indices = dataset_contract.pop("train_indices")
    full_eval_indices = dataset_contract.pop("eval_indices")
    train_rows = full_train_rows[: args.train_limit]
    eval_rows = full_eval_rows[: args.eval_limit]
    train_indices = full_train_indices[: args.train_limit]
    eval_indices = full_eval_indices[: args.eval_limit]

    seed_everything(args.seed)
    model_load_start = time.perf_counter()
    model, runtime = load_model(args, dtype, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    tokenizer = reference_tokenizer(args.model, args.revision, args.offline)
    prompts = model_prompt_contract(
        args.model,
        args.revision,
        args.offline,
        args.model_prompts,
    )
    training_schedule = build_training_schedule(
        train_indices,
        args.steps,
        args.batch_size,
        args.seed,
    )
    train_rows_by_index = dict(zip(train_indices, train_rows, strict = True))
    tokenization_start = time.perf_counter()
    training_batches = prepare_batches(
        tokenizer,
        train_rows_by_index,
        training_schedule,
        args.max_length,
        prompts,
    )
    eval_index_batches = [
        eval_indices[offset : offset + eval_batch_size]
        for offset in range(0, len(eval_indices), eval_batch_size)
    ]
    eval_rows_by_index = dict(zip(eval_indices, eval_rows, strict = True))
    eval_batches = prepare_batches(
        tokenizer,
        eval_rows_by_index,
        eval_index_batches,
        args.max_length,
        prompts,
    )
    tokenization_seconds = time.perf_counter() - tokenization_start

    train_inputs_sha256 = hash_feature_batches(training_batches)
    eval_inputs_sha256 = hash_feature_batches(eval_batches)
    all_inputs_sha256 = sha256_json({"train": train_inputs_sha256, "eval": eval_inputs_sha256})
    bucket_histogram = None
    if args.mode == "optimized":
        from unsloth.kernels.contrastive_loss import _bucketed_sentence_features

        bucket_counts = []
        for columns in training_batches:
            plan = _bucketed_sentence_features(model, columns)
            bucket_counts.append(len(plan[0]) if plan is not None else len(columns))
        bucket_histogram = {
            str(count): occurrences for count, occurrences in sorted(Counter(bucket_counts).items())
        }

    all_named = list(model.named_parameters())
    trainable_named = [
        (name, parameter) for name, parameter in all_named if parameter.requires_grad
    ]
    frozen_named = [
        (name, parameter) for name, parameter in all_named if not parameter.requires_grad
    ]
    trainable_parameters = [parameter for _, parameter in trainable_named]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable parameters after LoRA injection.")
    initial_trainable_sha256 = parameter_hash(trainable_named)
    initial_trainable_values_sha256 = parameter_value_hash(trainable_named)
    initial_frozen_sha256 = parameter_hash(frozen_named)
    initial_frozen_values_sha256 = parameter_value_hash(frozen_named)
    initial_snapshot = snapshot_parameters(trainable_named)

    initial_eval_start = time.perf_counter()
    initial_metrics = evaluate_triplets(model, eval_batches, device)
    initial_eval_seconds = time.perf_counter() - initial_eval_start

    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )

    loss_module = MultipleNegativesRankingLoss(model = model)
    fused_optimizer = args.fused_optimizer
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr = args.learning_rate,
        weight_decay = args.weight_decay,
        fused = fused_optimizer,
    )
    checkpoints = checkpoint_labels(args.steps)
    checkpoint_results: list[dict[str, Any]] = []
    losses: list[float] = []

    # Loading/tokenizing/evaluating can consume RNG state differently across
    # implementations.  Reset immediately before the first training forward.
    seed_everything(args.seed)
    model.train()
    synchronize(device)
    training_start = time.perf_counter()
    for step, cpu_columns in enumerate(training_batches, start = 1):
        optimizer.zero_grad(set_to_none = True)
        columns = [move_features(features, device) for features in cpu_columns]
        loss = loss_module(columns, labels = None)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        gradient_norm = gradient_l2_norm(trainable_parameters) if step in checkpoints else None
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        if step in checkpoints:
            synchronize(device)
            checkpoint_results.append(
                {
                    "step": step,
                    "labels": checkpoints[step],
                    "loss": loss_value,
                    "gradient_l2_norm": gradient_norm,
                    "parameter_delta": parameter_delta(trainable_named, initial_snapshot),
                    "trainable_sha256": parameter_hash(trainable_named),
                }
            )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    final_eval_start = time.perf_counter()
    final_metrics = evaluate_triplets(model, eval_batches, device)
    final_eval_seconds = time.perf_counter() - final_eval_start
    final_trainable_sha256 = parameter_hash(trainable_named)
    final_frozen_sha256 = parameter_hash(frozen_named)
    final_frozen_values_sha256 = parameter_value_hash(frozen_named)
    total_seconds = time.perf_counter() - total_start

    result = {
        "mode": args.mode,
        "model": args.model,
        "revision": args.revision,
        "implementation_file": runtime["implementation_file"],
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fused_optimizer": fused_optimizer,
        "prompt_contract": prompts,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "selection_seed": DATASET_SELECTION_SEED,
            "selection_algorithm": (
                "rank split indices by sha256("
                "'{dataset}@{revision}/{config}/{split}/{selection_seed}/{index}')"
            ),
            "eval_filter": (
                "exclude a dev row when any of its three texts occurs in the full "
                "8192-row train contract; then keep the first occurrence of each anchor"
            ),
            "content_digest_algorithm": (
                "sha256 of canonical UTF-8 JSON lines with sort_keys=True, "
                "separators=(',', ':'), ensure_ascii=False, including final newline"
            ),
            "index_digest_algorithm": "sha256 of the compact JSON integer-index array",
            "source_sizes": dataset_contract["source_sizes"],
            "full_contract": {
                "train_rows": FULL_TRAIN_LIMIT,
                "eval_rows": FULL_EVAL_LIMIT,
                "train_content_sha256": dataset_contract["full_train_content_sha256"],
                "eval_content_sha256": dataset_contract["full_eval_content_sha256"],
                "train_indices_sha256": dataset_contract["full_train_indices_sha256"],
                "eval_indices_sha256": dataset_contract["full_eval_indices_sha256"],
            },
            "selected": {
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_content_sha256": content_digest(train_rows),
                "eval_content_sha256": content_digest(eval_rows),
                "train_indices_sha256": index_digest(train_indices),
                "eval_indices_sha256": index_digest(eval_indices),
            },
            "inventory_reference_digests": INVENTORY_REFERENCE_DIGESTS,
        },
        "inputs": {
            "training_schedule_sha256": sha256_json(training_schedule),
            "train_token_tensors_sha256": train_inputs_sha256,
            "eval_token_tensors_sha256": eval_inputs_sha256,
            "all_token_tensors_sha256": all_inputs_sha256,
            "encoder_bucket_count_histogram": bucket_histogram,
        },
        "model_contract": {
            "model_config_sha256": model_config_hash(model),
            "resolved_model_commit": resolved_model_commit(model),
            "target_modules": runtime["target_modules"],
            "lora_config": runtime["lora_config"],
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "trainable_parameter_tensors": len(trainable_parameters),
            "frozen_parameters": sum(parameter.numel() for _, parameter in frozen_named),
            "frozen_parameter_tensors": len(frozen_named),
            "initial_trainable_sha256": initial_trainable_sha256,
            "initial_trainable_values_sha256": initial_trainable_values_sha256,
            "initial_frozen_sha256": initial_frozen_sha256,
            "initial_frozen_values_sha256": initial_frozen_values_sha256,
            "final_trainable_sha256": final_trainable_sha256,
            "final_frozen_sha256": final_frozen_sha256,
            "final_frozen_values_sha256": final_frozen_values_sha256,
            "frozen_parameters_unchanged": (final_frozen_sha256 == initial_frozen_sha256),
            "fused_lora_layers": runtime["fused_lora_layers"],
            "fused_lora_linears": runtime["fused_lora_linears"],
        },
        "quality": {
            "metric_contract": {
                "triplet_accuracy": "mean(cos(anchor, positive) > cos(anchor, negative))",
                "mean_margin": "mean(cos(anchor, positive) - cos(anchor, negative))",
                "cosine_accumulation_dtype": "float32",
            },
            "initial": initial_metrics,
            "final": final_metrics,
            "change": {
                "triplet_accuracy": (
                    float(final_metrics["triplet_accuracy"])
                    - float(initial_metrics["triplet_accuracy"])
                ),
                "mean_margin": (
                    float(final_metrics["mean_margin"]) - float(initial_metrics["mean_margin"])
                ),
            },
        },
        "training": {
            "losses": losses,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "checkpoints": checkpoint_results,
        },
        "timing": {
            "dataset_seconds": dataset_seconds,
            "model_load_seconds": model_load_seconds,
            "tokenization_seconds": tokenization_seconds,
            "initial_eval_seconds": initial_eval_seconds,
            "training_seconds": training_seconds,
            "final_eval_seconds": final_eval_seconds,
            "wall_seconds": total_seconds,
        },
    }
    rendered = json.dumps(result, indent = 2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents = True, exist_ok = True)
        args.output.write_text(rendered + "\n", encoding = "utf-8")


if __name__ == "__main__":
    main()
