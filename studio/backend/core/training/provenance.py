# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Optional


RESOURCE_PROVENANCE_VERSION = 1
RESOURCE_PROVENANCE_KEY = "resource_provenance"

_ATTESTED = "attested"
_INCOMPLETE = "incomplete"
_REPO_ID_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95}))?"
)
_REASON_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_MODEL_WEIGHT_RE = re.compile(
    r"(?:"
    r"model(?:-\d+-of-\d+)?|"
    r"pytorch_model(?:-\d+-of-\d+)?|"
    r"adapter_model(?:-\d+-of-\d+)?|"
    r"consolidated(?:[._-]\d+)?|"
    r"mlx_model(?:-\d+-of-\d+)?|"
    r"weights"
    r")\.(?:safetensors|bin|pt|pth|ckpt|npz)$",
    re.IGNORECASE,
)
_DATASET_DATA_SUFFIXES = (
    ".parquet",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".arrow",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".gz",
    ".zst",
    ".zip",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff",
    ".svg",
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".webm",
)
_DATASET_METADATA_FILENAMES = frozenset(
    {
        "config.json",
        "dataset_info.json",
        "dataset_infos.json",
        "metadata.json",
        "state.json",
    }
)


def initialize_resource_provenance(config: dict[str, Any]) -> None:
    config[RESOURCE_PROVENANCE_KEY] = {
        "version": RESOURCE_PROVENANCE_VERSION,
        "status": "pending",
    }


class ExactResumeResourcesUnavailable(ValueError):
    pass


def _normalized_repo_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().strip("/")
    if (
        not value
        or len(value) > 192
        or ".." in value.split("/")
        or not _REPO_ID_RE.fullmatch(value)
    ):
        return None
    return value


def _normalized_commit(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (
        not value
        or len(value) > 256
        or value in {".", ".."}
        or Path(value).name != value
        or PureWindowsPath(value).name != value
    ):
        return None
    return value


def _snapshot_declares_quantization(snapshot: Path) -> bool:
    try:
        parsed = json.loads((snapshot / "config.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(parsed, dict) and bool(parsed.get("quantization_config"))


def _snapshot_has_model_weights(snapshot: Path) -> bool:
    try:
        for root, dirnames, filenames in os.walk(snapshot, followlinks = False):
            dirnames[:] = [name for name in dirnames if not (Path(root) / name).is_symlink()]
            for filename in filenames:
                if _MODEL_WEIGHT_RE.fullmatch(filename):
                    path = Path(root) / filename
                    if path.is_file():
                        return True
    except OSError:
        return False
    return False


def _snapshot_has_dataset_data(snapshot: Path) -> bool:
    try:
        for root, dirnames, filenames in os.walk(snapshot, followlinks = False):
            dirnames[:] = [name for name in dirnames if not (Path(root) / name).is_symlink()]
            for filename in filenames:
                lowered = filename.lower()
                if (
                    lowered not in _DATASET_METADATA_FILENAMES
                    and lowered.endswith(_DATASET_DATA_SUFFIXES)
                ):
                    path = Path(root) / filename
                    if path.is_file():
                        return True
    except OSError:
        return False
    return False


def exact_model_snapshot_path(
    path_value: Any,
    repo_id: Any,
    *,
    require_quantized: bool = False,
) -> Optional[str]:
    repo_id = _normalized_repo_id(repo_id)
    if repo_id is None or not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        requested = Path(path_value).expanduser().resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None

    from hub.utils.hf_cache_state import latest_snapshot_from_cache_path

    validated = latest_snapshot_from_cache_path(
        str(requested),
        "model",
        repo_id,
        ("config.json", "adapter_config.json"),
    )
    if validated is None:
        return None
    try:
        resolved = Path(validated).resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved != requested or not _snapshot_has_model_weights(resolved):
        return None
    if require_quantized and not _snapshot_declares_quantization(resolved):
        return None
    return str(resolved)


def exact_model_snapshot_for_commit(
    repo_id: Any,
    commit: Any,
    *,
    require_quantized: bool = False,
) -> Optional[str]:
    repo_id = _normalized_repo_id(repo_id)
    commit = _normalized_commit(commit)
    if repo_id is None or commit is None:
        return None

    from hub.utils.hf_cache_state import iter_repo_cache_dirs

    for repo_dir in iter_repo_cache_dirs("model", repo_id):
        candidate = repo_dir / "snapshots" / commit
        resolved = exact_model_snapshot_path(
            str(candidate),
            repo_id,
            require_quantized = require_quantized,
        )
        if resolved is not None:
            return resolved
    return None


def exact_dataset_snapshot_path(path_value: Any, repo_id: Any) -> Optional[str]:
    repo_id = _normalized_repo_id(repo_id)
    if repo_id is None or not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        requested = Path(path_value).expanduser().resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None

    from hub.utils.dataset_cache import dataset_snapshot_from_cache_path

    validated = dataset_snapshot_from_cache_path(str(requested), repo_id)
    if validated is None:
        return None
    try:
        resolved = validated.resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved != requested or not _snapshot_has_dataset_data(resolved):
        return None
    return str(resolved)


def _object_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    try:
        return getattr(value, key, None)
    except Exception:
        return None


def _loaded_model_refs(model: Any) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    queue = [model]
    seen: set[int] = set()
    while queue and len(seen) < 32:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        repo_id = _normalized_repo_id(
            _object_value(current, "_name_or_path")
            or _object_value(current, "name_or_path")
        )
        commit = _normalized_commit(
            _object_value(current, "_commit_hash")
            or _object_value(current, "commit_hash")
        )
        if repo_id is not None and commit is not None:
            refs.add((repo_id, commit))

        for attr in ("config", "model", "auto_model", "base_model", "module"):
            child = _object_value(current, attr)
            if child is not None and id(child) not in seen:
                queue.append(child)
        modules = _object_value(current, "_modules")
        if isinstance(modules, dict):
            queue.extend(list(modules.values())[:16])
    return refs


def attest_loaded_model(
    config: dict[str, Any],
    model: Any,
    *,
    load_target: Any,
    load_in_4bit: bool,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    selected_repo = config.get("actual_model_repo_id") or config.get("model_name")
    direct = exact_model_snapshot_path(
        load_target,
        selected_repo,
        require_quantized = load_in_4bit,
    )
    if direct is not None:
        return _normalized_repo_id(selected_repo), direct, None

    resolved: set[tuple[str, str]] = set()
    for repo_id, commit in _loaded_model_refs(model):
        snapshot = exact_model_snapshot_for_commit(
            repo_id,
            commit,
            require_quantized = load_in_4bit,
        )
        if snapshot is not None:
            resolved.add((repo_id, snapshot))
    if len(resolved) == 1:
        repo_id, snapshot = resolved.pop()
        return repo_id, snapshot, None
    if len(resolved) > 1:
        return None, None, "model_metadata_ambiguous"
    reason = "model_quantized_snapshot_unattested" if load_in_4bit else "model_snapshot_unattested"
    return None, None, reason


def _dataset_reason(config: dict[str, Any]) -> str:
    if config.get("dataset_streaming"):
        return "dataset_streaming_unattested"
    if config.get("s3_config") or config.get("dataset_source") == "s3":
        return "dataset_s3_mutable"
    if config.get("local_datasets"):
        return "dataset_local_mutable"
    if config.get("dataset_snapshot_path"):
        return "dataset_cache_unattested"
    return "dataset_revision_unattested"


def build_worker_provenance_event(
    config: dict[str, Any],
    model: Any,
    *,
    model_load_target: Any,
    model_load_in_4bit: bool,
    dataset_loaded_from_exact_snapshot: bool,
) -> dict[str, Any]:
    model_repo_id, model_snapshot, model_reason = attest_loaded_model(
        config,
        model,
        load_target = model_load_target,
        load_in_4bit = model_load_in_4bit,
    )

    dataset_snapshot = None
    dataset_reason = None
    if dataset_loaded_from_exact_snapshot:
        dataset_snapshot = exact_dataset_snapshot_path(
            config.get("dataset_snapshot_path"),
            config.get("hf_dataset"),
        )
    if dataset_snapshot is None:
        dataset_reason = _dataset_reason(config)

    reasons = [reason for reason in (model_reason, dataset_reason) if reason]
    return {
        "type": "resource_provenance",
        "version": RESOURCE_PROVENANCE_VERSION,
        "model": {
            "status": _ATTESTED if model_snapshot else _INCOMPLETE,
            "repo_id": model_repo_id,
            "snapshot_path": model_snapshot,
        },
        "dataset": {
            "status": _ATTESTED if dataset_snapshot else _INCOMPLETE,
            "snapshot_path": dataset_snapshot,
        },
        "reasons": reasons,
    }


def incomplete_worker_provenance_event(*reasons: str) -> dict[str, Any]:
    return {
        "type": "resource_provenance",
        "version": RESOURCE_PROVENANCE_VERSION,
        "model": {"status": _INCOMPLETE, "repo_id": None, "snapshot_path": None},
        "dataset": {"status": _INCOMPLETE, "snapshot_path": None},
        "reasons": list(reasons) or ["provenance_unavailable"],
    }


def _normalized_reasons(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    reasons: list[str] = []
    for value in values[:8]:
        if isinstance(value, str) and _REASON_RE.fullmatch(value):
            if value not in reasons:
                reasons.append(value)
    return reasons


def normalize_worker_provenance_event(
    event: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    reasons = _normalized_reasons(event.get("reasons"))
    model_event = event.get("model") if isinstance(event.get("model"), dict) else {}
    dataset_event = event.get("dataset") if isinstance(event.get("dataset"), dict) else {}

    model_repo_id = _normalized_repo_id(model_event.get("repo_id"))
    model_snapshot = None
    if (
        event.get("version") == RESOURCE_PROVENANCE_VERSION
        and model_event.get("status") == _ATTESTED
    ):
        model_snapshot = exact_model_snapshot_path(
            model_event.get("snapshot_path"),
            model_repo_id,
            require_quantized = bool(config.get("load_in_4bit")),
        )
    if model_snapshot is None:
        model_repo_id = None
        if "model_event_invalid" not in reasons:
            reasons.append("model_event_invalid")

    dataset_snapshot = None
    if (
        event.get("version") == RESOURCE_PROVENANCE_VERSION
        and dataset_event.get("status") == _ATTESTED
    ):
        dataset_snapshot = exact_dataset_snapshot_path(
            dataset_event.get("snapshot_path"),
            config.get("hf_dataset"),
        )
    if dataset_snapshot is None and "dataset_event_invalid" not in reasons:
        reasons.append("dataset_event_invalid")

    complete = model_snapshot is not None and dataset_snapshot is not None
    return {
        "actual_model_repo_id": model_repo_id,
        "model_snapshot_path": model_snapshot,
        "dataset_snapshot_path": dataset_snapshot,
        RESOURCE_PROVENANCE_KEY: {
            "version": RESOURCE_PROVENANCE_VERSION,
            "status": "complete" if complete else "incomplete",
            "model_status": _ATTESTED if model_snapshot else _INCOMPLETE,
            "dataset_status": _ATTESTED if dataset_snapshot else _INCOMPLETE,
            "reasons": reasons,
        },
    }


def resource_provenance_is_complete(config: dict[str, Any]) -> bool:
    marker = config.get(RESOURCE_PROVENANCE_KEY)
    return (
        isinstance(marker, dict)
        and marker.get("version") == RESOURCE_PROVENANCE_VERSION
        and marker.get("status") == "complete"
    )


def validate_exact_resource_pins(config: dict[str, Any]) -> tuple[str, str]:
    model_snapshot = exact_model_snapshot_path(
        config.get("model_snapshot_path"),
        config.get("actual_model_repo_id") or config.get("model_name"),
        require_quantized = bool(config.get("load_in_4bit")),
    )
    if model_snapshot is None:
        raise ExactResumeResourcesUnavailable(
            "The exact model snapshot for this run is no longer available."
        )
    dataset_snapshot = exact_dataset_snapshot_path(
        config.get("dataset_snapshot_path"),
        config.get("hf_dataset"),
    )
    if dataset_snapshot is None:
        raise ExactResumeResourcesUnavailable(
            "The exact dataset snapshot for this run is no longer available."
        )
    return model_snapshot, dataset_snapshot


def resource_provenance_allows_resume(config: dict[str, Any]) -> bool:
    marker = config.get(RESOURCE_PROVENANCE_KEY)
    if marker is None:
        return True
    if not isinstance(marker, dict) or marker.get("version") != RESOURCE_PROVENANCE_VERSION:
        return False
    status = marker.get("status")
    if status in {"pending", "incomplete"}:
        return True
    if status != "complete":
        return False
    try:
        validate_exact_resource_pins(config)
    except ExactResumeResourcesUnavailable:
        return False
    return True
