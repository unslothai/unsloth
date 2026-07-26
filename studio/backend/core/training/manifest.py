# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portable, secret-free metadata for reconstructing Studio training runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping

from utils.studio_version import get_studio_version

MANIFEST_FILENAME = "unsloth_training_manifest.json"
MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset((1, 2))

_SECRET_KEYS = frozenset(
    {
        "hf_token",
        "huggingface_token",
        "token",
        "wandb_token",
        "api_key",
        "secret",
        "password",
        "access_key",
        "secret_key",
        "session_token",
        "s3_config",
        "subject",
        "authenticated_subject",
        "authorization",
    }
)


class ManifestError(ValueError):
    """A manifest is malformed, unsupported, or incompatible."""


def redact_secrets(value: Any) -> Any:
    """Recursively remove credentials and authenticated identity data.

    This is deliberately stricter than the DB's top-level policy because dataset
    descriptors and custom mappings may contain nested, user supplied objects.
    """
    if isinstance(value, Mapping):
        return {
            str(key): redact_secrets(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
            and not any(marker in str(key).lower() for marker in ("password", "credential"))
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys = True, separators = (",", ":"), ensure_ascii = False).encode()


def fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(redact_secrets(value))).hexdigest()


def _local_model_identity(model_name: Any) -> dict[str, str] | str:
    name = str(model_name or "")
    path = Path(name).expanduser()
    if name and (path.is_absolute() or path.exists()):
        # Do not make a portable manifest depend on, or disclose, an absolute path.
        return {"kind": "local", "name": path.name, "identity": fingerprint(name)}
    return name


_TRAINING_ARGUMENT_KEYS = (
    "num_epochs",
    "learning_rate",
    "embedding_learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "warmup_ratio",
    "max_steps",
    "save_steps",
    "save_total_limit",
    # Upload preferences are safe, non-credential configuration. Persist them
    # so an imported checkpoint retains the user's opt-in and Hub destination.
    "push_to_hub",
    "hub_model_id",
    "weight_decay",
    "max_grad_norm",
    "max_grad_value",
    "max_grad_leaf_norm",
    "cast_norm_output_to_input_dtype",
    "optim",
    "lr_scheduler_type",
    "eval_steps",
    "gradient_checkpointing",
)


def build_manifest(
    config: Mapping[str, Any], *, expected_checkpoint_step: int = 0
) -> dict[str, Any]:
    """Build schema v2 from the same already-normalized config used by a worker."""
    safe = redact_secrets(dict(config))
    datasets = {
        "repository": safe.get("hf_dataset"),
        "descriptors": safe.get("training_datasets") or [],
        "local": safe.get("local_datasets") or [],
        "local_eval": safe.get("local_eval_datasets") or [],
        "revision": safe.get("dataset_revision"),
        "portable_resume_data": safe.get("portable_resume_data", "metadata"),
        "pinned_revisions": safe.get("pinned_dataset_revisions") or [],
        "bundled_sources": safe.get("bundled_dataset_sources") or [],
        "snapshot": safe.get("dataset_snapshot"),
        "streaming": {
            "enabled": bool(safe.get("dataset_streaming")),
            "bounded": safe.get("dataset_slice_end") is not None,
            "fully_offline_portable": bool(
                safe.get("dataset_snapshot") and not safe.get("dataset_streaming")
            ),
            "warning": (
                "Streaming datasets may be impossible to materialize completely; "
                "a pinned revision alone is not a fully offline portable copy."
                if safe.get("dataset_streaming") else None
            ),
        },
    }
    preprocessing = {
        key: safe.get(key)
        for key in (
            "train_split",
            "eval_split",
            "subset",
            "format_type",
            "custom_format_mapping",
            "chat_template",
            "packing",
            "max_seq_length",
            "random_seed",
            "dataset_streaming",
            "dataset_slice_start",
            "dataset_slice_end",
            "train_on_completions",
            "is_dataset_image",
            "is_dataset_audio",
            "is_embedding",
            "vision_image_size",
        )
    }
    tuning = {
        key: safe.get(key)
        for key in (
            "training_type",
            "use_lora",
            "load_in_4bit",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "use_rslora",
            "use_loftq",
            "use_dora",
            "finetune_vision_layers",
            "finetune_language_layers",
            "finetune_attention_modules",
            "finetune_mlp_modules",
        )
    }
    arguments = {key: safe.get(key) for key in _TRAINING_ARGUMENT_KEYS}
    model = _local_model_identity(safe.get("model_name"))
    relevant = {"tuning": tuning, "preprocessing": preprocessing, "arguments": arguments}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "studio_version": get_studio_version(),
        "training_backend": "mlx" if safe.get("device_backend") == "mlx" else "transformers",
        "trainer_type": "embedding"
        if safe.get("is_embedding")
        else safe.get("trainer_type", "sft"),
        "base_model": model,
        "fine_tuning": tuning,
        "datasets": datasets,
        "preprocessing": preprocessing,
        "training_arguments": arguments,
        "expected_checkpoint_step": int(expected_checkpoint_step),
        "output_layout": {
            "manifest": MANIFEST_FILENAME,
            "checkpoint_pattern": "checkpoint-{step}",
            "trainer_state": "trainer_state.json",
        },
        "continuation": {
            "resumed": bool(safe.get("resume_checkpoint_path") or safe.get("resume_from_checkpoint")),
            "source_identity": fingerprint(
                safe.get("resume_checkpoint_path") or safe.get("resume_from_checkpoint")
            )
            if (safe.get("resume_checkpoint_path") or safe.get("resume_from_checkpoint"))
            else None,
            "destination_name": Path(str(safe.get("output_dir") or "")).name or None,
            "copied_to_local_storage": bool(safe.get("copy_checkpoint_to_local")),
        },
        "fingerprints": {
            "model": fingerprint(model),
            "tokenizer": fingerprint(
                {"model": model, "chat_template": preprocessing.get("chat_template")}
            ),
            "datasets": fingerprint(datasets),
            "configuration": fingerprint(relevant),
        },
    }


def atomic_write_manifest(directory: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """fsync a temporary file, replace the destination, then fsync its directory."""
    target_dir = Path(directory)
    target_dir.mkdir(parents = True, exist_ok = True)
    target = target_dir / MANIFEST_FILENAME
    fd, temporary = tempfile.mkstemp(prefix = f".{MANIFEST_FILENAME}.", suffix = ".tmp", dir = target_dir)
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as stream:
            json.dump(redact_secrets(dict(manifest)), stream, indent = 2, sort_keys = True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def migrate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    version = manifest.get("schema_version", 1)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(f"Unsupported training manifest schema version: {version!r}.")
    if version == 1:
        warnings.warn(
            "Training manifest schema v1 was migrated to v2; review compatibility warnings.",
            UserWarning,
            stacklevel = 2,
        )
        manifest["base_model"] = manifest.pop("model", manifest.get("base_model", ""))
        manifest.setdefault(
            "output_layout",
            {
                "manifest": MANIFEST_FILENAME,
                "checkpoint_pattern": "checkpoint-{step}",
                "trainer_state": "trainer_state.json",
            },
        )
        manifest.setdefault("fingerprints", {})
        manifest["schema_version"] = 2
    return redact_secrets(manifest)


def parse_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Training manifest is missing or invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("Training manifest root must be an object.")
    return migrate_manifest(value)


def write_checkpoint_manifests(output_dir: str | os.PathLike[str]) -> None:
    """Publish manifests only for checkpoints whose trainer state is complete."""
    root = Path(output_dir)
    source = parse_manifest(root / MANIFEST_FILENAME)
    for checkpoint in root.glob("checkpoint-*"):
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding = "utf-8"))
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if state.get("global_step") != step:
            continue
        updated = dict(source)
        updated["expected_checkpoint_step"] = step
        atomic_write_manifest(checkpoint, updated)
