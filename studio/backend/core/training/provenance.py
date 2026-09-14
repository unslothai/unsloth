# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Optional
from urllib.parse import unquote, urlsplit

from hub.utils.paths import is_valid_repo_id


RESOURCE_PROVENANCE_VERSION = 1
RESOURCE_PROVENANCE_KEY = "resource_provenance"

_ATTESTED = "attested"
_INCOMPLETE = "incomplete"
_MODEL_LOAD_UNQUANTIZED = "unquantized"
_MODEL_LOAD_PREQUANTIZED_4BIT = "prequantized_4bit"
_MODEL_LOAD_RUNTIME_4BIT = "runtime_4bit"
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


def effective_training_load_in_4bit(
    config: dict[str, Any], model_load_target: str, hf_token: Optional[str]
) -> bool:
    if not bool(config.get("load_in_4bit")):
        return False
    from utils.transformers_version import latest_tier_active_for

    latest_tier_active = latest_tier_active_for(model_load_target, hf_token)
    if latest_tier_active and (
        config.get("require_exact_resume_resources") or config.get("require_exact_model_resource")
    ):
        raise ExactResumeResourcesUnavailable(
            "This checkpoint requires its original 4-bit model load mode, "
            "which is unavailable with the active Transformers runtime."
        )
    return not latest_tier_active


def exact_resume_requires_current_4bit(config: dict[str, Any]) -> bool:
    """Would activating the latest-transformers sidecar strand this stored run?

    ``effective_training_load_in_4bit`` raises ``ExactResumeResourcesUnavailable`` for a
    4-bit run with exact-resource provenance the moment ``latest_tier_active_for`` turns
    true, and that sidecar is a persistent overlay: once installed the checkpoint never
    resumes in the load mode it was attested with. Callers offering the install ahead of
    a resume ask this first, rather than trade a working resume for an upgrade the run
    does not need.

    Takes the run's STORED config (``config_json``), so it recomputes the requirement
    from the provenance marker: ``require_exact_resume_resources`` and
    ``require_exact_model_resource`` are stripped before persistence and exist only on
    the live worker config ``/train/start`` assembles.

    Never raises. A provenance already refusing a resume returns False: nothing the
    install does makes that checkpoint any less resumable.
    """
    if not bool(config.get("load_in_4bit")):
        return False
    try:
        requires_exact_model, _ = exact_resume_resource_requirements(config)
    except ExactResumeResourcesUnavailable:
        return False
    except Exception:
        return False
    # The same disjunction effective_training_load_in_4bit tests: routes/training.py fills
    # require_exact_model_resource from exact_resume_resource_requirements and
    # require_exact_resume_resources from resource_provenance_is_complete.
    return bool(requires_exact_model or resource_provenance_is_complete(config))


def _normalized_repo_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not is_valid_repo_id(value):
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
    if not isinstance(parsed, dict):
        return False
    queue = [parsed.get("quantization_config"), parsed.get("quantization")]
    found_4bit = False
    conflicting_width = False
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            if current.get("load_in_4bit") is True:
                found_4bit = True
            if current.get("load_in_8bit") is True:
                conflicting_width = True
            for key in ("bits", "nbits", "q_bits"):
                width = current.get(key)
                if isinstance(width, str) and width.strip().isdigit():
                    width = int(width.strip())
                if isinstance(width, bool) or not isinstance(width, int):
                    continue
                if width == 4:
                    found_4bit = True
                else:
                    conflicting_width = True
            queue.extend(current.values())
        elif isinstance(current, (list, tuple)):
            queue.extend(current)
    return found_4bit and not conflicting_width


def _resolved_model_snapshot_file(snapshot: Path, path: Path) -> Optional[Path]:
    from hub.utils.hf_cache_state import same_existing_path

    try:
        snapshot = snapshot.resolve(strict = True)
        repo_dir = snapshot.parent.parent.resolve(strict = True)
        if not same_existing_path(snapshot.parent, repo_dir / "snapshots"):
            return None
        relative = path.relative_to(snapshot)
        resolved = snapshot.joinpath(*relative.parts).resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file() or not (
        resolved.is_relative_to(snapshot) or resolved.is_relative_to(repo_dir / "blobs")
    ):
        return None
    try:
        with resolved.open("rb"):
            pass
    except OSError:
        return None
    return resolved


def _raise_walk_error(error: OSError) -> None:
    raise error


def _snapshot_has_model_weights(snapshot: Path) -> bool:
    found_weights = False
    try:
        for root, dirnames, filenames in os.walk(
            snapshot,
            followlinks = False,
            onerror = _raise_walk_error,
        ):
            if any((Path(root) / name).is_symlink() for name in dirnames):
                return False
            for filename in filenames:
                path = Path(root) / filename
                if _resolved_model_snapshot_file(snapshot, path) is None:
                    return False
                if _MODEL_WEIGHT_RE.fullmatch(filename):
                    found_weights = True
    except (OSError, RuntimeError, ValueError):
        return False
    return found_weights


def _snapshot_has_dataset_data(snapshot: Path) -> bool:
    from hub.utils.dataset_cache import resolved_dataset_snapshot_file

    found_data = False
    try:
        for root, dirnames, filenames in os.walk(
            snapshot,
            followlinks = False,
            onerror = _raise_walk_error,
        ):
            if any((Path(root) / name).is_symlink() for name in dirnames):
                return False
            for filename in filenames:
                lowered = filename.lower()
                path = Path(root) / filename
                relative = path.relative_to(snapshot).as_posix()
                if resolved_dataset_snapshot_file(snapshot, relative) is None:
                    return False
                if lowered not in _DATASET_METADATA_FILENAMES and lowered.endswith(
                    _DATASET_DATA_SUFFIXES
                ):
                    found_data = True
    except (OSError, RuntimeError, ValueError):
        return False
    return found_data


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

    from hub.utils.hf_cache_state import (
        latest_snapshot_from_cache_path,
        same_existing_path,
        with_load_subdirs,
    )

    validated = latest_snapshot_from_cache_path(
        str(requested),
        "model",
        repo_id,
        with_load_subdirs(repo_id, ("config.json", "adapter_config.json")),
    )
    if validated is None:
        return None
    try:
        resolved = Path(validated).resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not same_existing_path(resolved, requested) or not _snapshot_has_model_weights(resolved):
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
    from hub.utils.hf_cache_state import same_existing_path

    validated = dataset_snapshot_from_cache_path(str(requested), repo_id)
    if validated is None:
        return None
    try:
        resolved = validated.resolve(strict = True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not same_existing_path(resolved, requested) or not _snapshot_has_dataset_data(resolved):
        return None
    return str(resolved)


def exact_dataset_snapshot_for_commit(repo_id: Any, commit: Any) -> Optional[str]:
    repo_id = _normalized_repo_id(repo_id)
    commit = _normalized_commit(commit)
    if repo_id is None or commit is None:
        return None

    from hub.utils.hf_cache_state import iter_repo_cache_dirs

    for repo_dir in iter_repo_cache_dirs("dataset", repo_id):
        resolved = exact_dataset_snapshot_path(
            str(repo_dir / "snapshots" / commit),
            repo_id,
        )
        if resolved is not None:
            return resolved
    return None


def _local_dataset_source_snapshot(path_value: str, repo_id: str) -> Optional[tuple[str, str]]:
    if len(path_value) > 4096 or "\x00" in path_value:
        return None
    path = Path(path_value).expanduser()
    if PureWindowsPath(path_value).is_absolute() and not path.is_absolute():
        return None
    if not path.is_absolute() or ".." in path.parts or not path.is_file():
        return None
    for parent in path.parents:
        if parent.parent.name != "snapshots":
            continue
        snapshot = exact_dataset_snapshot_path(str(parent), repo_id)
        try:
            source_path = path.relative_to(parent).as_posix()
        except ValueError:
            continue
        if snapshot is not None and _dataset_snapshot_contains(snapshot, source_path):
            return snapshot, source_path
    return None


def _hf_dataset_source_ref(path_value: str) -> Optional[tuple[str, str, str]]:
    if path_value.startswith("hf://datasets/"):
        remainder = path_value.removeprefix("hf://datasets/")
        repo_id, marker, revision_path = remainder.partition("@")
        commit, separator, source_path = revision_path.partition("/")
        normalized_repo = _normalized_repo_id(repo_id)
        normalized_commit = _normalized_commit(commit)
        if (
            marker
            and separator
            and source_path
            and normalized_repo is not None
            and normalized_commit is not None
        ):
            return normalized_repo, normalized_commit, unquote(source_path)
        return None

    try:
        parsed = urlsplit(path_value)
        endpoint = urlsplit(os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.netloc.lower() != endpoint.netloc.lower()
    ):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    endpoint_parts = [part for part in endpoint.path.split("/") if part]
    if parts[: len(endpoint_parts)] != endpoint_parts:
        return None
    parts = parts[len(endpoint_parts) :]
    if not parts or parts[0] != "datasets" or "resolve" not in parts:
        return None
    resolve_index = parts.index("resolve")
    if resolve_index not in {2, 3} or len(parts) <= resolve_index + 2:
        return None
    repo_id = _normalized_repo_id("/".join(parts[1:resolve_index]))
    commit = _normalized_commit(parts[resolve_index + 1])
    if repo_id is None or commit is None:
        return None
    return repo_id, commit, unquote("/".join(parts[resolve_index + 2 :]))


def _dataset_snapshot_contains(snapshot: str, source_path: str) -> bool:
    from hub.utils.dataset_cache import dataset_snapshot_contains_file
    return dataset_snapshot_contains_file(snapshot, source_path)


def _dataset_snapshot_file(snapshot: str, source_path: str) -> Optional[Path]:
    from hub.utils.dataset_cache import resolved_dataset_snapshot_file
    return resolved_dataset_snapshot_file(snapshot, source_path)


def _loaded_dataset_objects(value: Any):
    if value is None:
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _loaded_dataset_objects(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _loaded_dataset_objects(child)
        return
    yield value


def attest_loaded_dataset(repo_id: Any, *datasets: Any) -> tuple[Optional[str], Optional[str]]:
    repo_id = _normalized_repo_id(repo_id)
    if repo_id is None:
        return None, "dataset_revision_unattested"

    snapshots: set[str] = set()
    found_dataset = False
    for value in datasets:
        for dataset in _loaded_dataset_objects(value):
            found_dataset = True
            info = _object_value(dataset, "info")
            checksums = _object_value(info, "download_checksums")
            if not isinstance(checksums, dict) or not checksums:
                return None, "dataset_revision_unattested"
            for source, download_info in checksums.items():
                if not isinstance(source, str):
                    return None, "dataset_source_unattested"
                expected_size = _object_value(download_info, "num_bytes")
                if (
                    not isinstance(expected_size, int)
                    or isinstance(expected_size, bool)
                    or expected_size < 0
                ):
                    return None, "dataset_revision_unattested"
                local_source = _local_dataset_source_snapshot(source, repo_id)
                if local_source is not None:
                    snapshot, source_path = local_source
                else:
                    source_ref = _hf_dataset_source_ref(source)
                    if source_ref is None or source_ref[0].casefold() != repo_id.casefold():
                        return None, "dataset_source_unattested"
                    snapshot = exact_dataset_snapshot_for_commit(
                        repo_id,
                        source_ref[1],
                    )
                    source_path = source_ref[2]
                resolved_source = (
                    _dataset_snapshot_file(snapshot, source_path) if snapshot is not None else None
                )
                if resolved_source is None:
                    return None, "dataset_snapshot_unavailable"
                try:
                    actual_size = resolved_source.stat().st_size
                except OSError:
                    return None, "dataset_snapshot_unavailable"
                if actual_size != expected_size:
                    return None, "dataset_snapshot_unavailable"
                snapshots.add(snapshot)
                if len(snapshots) > 1:
                    return None, "dataset_metadata_ambiguous"

    if not found_dataset or len(snapshots) != 1:
        return None, "dataset_revision_unattested"
    return snapshots.pop(), None


def _object_value(value: Any, key: str) -> Any:
    """Read ``key`` off a loaded model object, whatever shape it is.

    Attribute access has to come first: ``mlx.nn.Module`` subclasses ``dict``, so a
    mapping-first lookup answers ``None`` for every attribute an MLX model carries and
    the whole MLX attestation path below goes blind. The mapping lookup stays as the
    fallback for the plain dicts that also flow through here (``quantization_config``,
    ``_unsloth_quantization_policy``), whose keys are never attributes.
    """
    try:
        found = getattr(value, key, None)
    except Exception:
        found = None
    if found is not None:
        return found
    if isinstance(value, dict):
        return value.get(key)
    return None


def _loaded_model_objects(model: Any):
    queue = [model]
    seen: set[int] = set()
    while queue and len(seen) < 32:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        for attr in (
            "config",
            "hf_quantizer",
            "model",
            "auto_model",
            "base_model",
            "module",
        ):
            child = _object_value(current, attr)
            if child is not None and id(child) not in seen:
                queue.append(child)
        modules = _object_value(current, "_modules")
        if isinstance(modules, dict):
            queue.extend(list(modules.values())[:16])


def _loaded_model_is_4bit(model: Any) -> bool:
    for current in _loaded_model_objects(model):
        if _object_value(current, "is_loaded_in_4bit") is True:
            return True
        quantization = _object_value(current, "quantization_config")
        if isinstance(quantization, dict):
            if quantization.get("load_in_4bit") is True:
                return True
        elif _object_value(quantization, "load_in_4bit") is True:
            return True
        if _object_value(current, "_unsloth_quantized_source") == "runtime":
            policy = _object_value(current, "_unsloth_quantization_policy")
            if _object_value(policy, "enabled") is True and _object_value(policy, "bits") == 4:
                return True
    return False


def _loaded_model_refs(model: Any) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for current in _loaded_model_objects(model):
        candidates = (
            (
                _object_value(current, "_hf_repo"),
                _object_value(current, "_unsloth_base_commit_hash"),
            ),
            (
                _object_value(current, "_name_or_path") or _object_value(current, "name_or_path"),
                _object_value(current, "_commit_hash") or _object_value(current, "commit_hash"),
            ),
        )
        for repo_value, commit_value in candidates:
            repo_id = _normalized_repo_id(repo_value)
            commit = _normalized_commit(commit_value)
            if repo_id is not None and commit is not None:
                refs.add((repo_id, commit))
    return refs


def _attested_model_load_mode(snapshot: str, model: Any, load_in_4bit: bool) -> Optional[str]:
    if not load_in_4bit:
        return _MODEL_LOAD_UNQUANTIZED
    if _snapshot_declares_quantization(Path(snapshot)):
        return _MODEL_LOAD_PREQUANTIZED_4BIT
    if _loaded_model_is_4bit(model):
        return _MODEL_LOAD_RUNTIME_4BIT
    return None


def attest_loaded_model(
    config: dict[str, Any], model: Any, *, load_target: Any, load_in_4bit: bool
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    selected_repo = config.get("actual_model_repo_id")
    if selected_repo is None:
        from utils.utils import canonical_model_repo_id
        selected_repo = canonical_model_repo_id(str(config.get("model_name") or ""))
    direct = exact_model_snapshot_path(
        load_target,
        selected_repo,
    )
    if direct is not None:
        load_mode = _attested_model_load_mode(direct, model, load_in_4bit)
        if load_mode is not None:
            return _normalized_repo_id(selected_repo), direct, load_mode, None

    resolved: set[tuple[str, str, str]] = set()
    for repo_id, commit in _loaded_model_refs(model):
        snapshot = exact_model_snapshot_for_commit(
            repo_id,
            commit,
        )
        if snapshot is not None:
            load_mode = _attested_model_load_mode(snapshot, model, load_in_4bit)
            if load_mode is not None:
                resolved.add((repo_id, snapshot, load_mode))
    if len(resolved) == 1:
        repo_id, snapshot, load_mode = resolved.pop()
        return repo_id, snapshot, load_mode, None
    if len(resolved) > 1:
        return None, None, None, "model_metadata_ambiguous"
    reason = "model_quantized_snapshot_unattested" if load_in_4bit else "model_snapshot_unattested"
    return None, None, None, reason


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
    model_repo_id, model_snapshot, model_load_mode, model_reason = attest_loaded_model(
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
            "load_mode": model_load_mode,
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
        "model": {
            "status": _INCOMPLETE,
            "repo_id": None,
            "snapshot_path": None,
            "load_mode": None,
        },
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
    event: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    reasons = _normalized_reasons(event.get("reasons"))
    model_event = event.get("model") if isinstance(event.get("model"), dict) else {}
    dataset_event = event.get("dataset") if isinstance(event.get("dataset"), dict) else {}

    model_repo_id = _normalized_repo_id(model_event.get("repo_id"))
    model_snapshot = None
    model_load_mode = None
    if (
        event.get("version") == RESOURCE_PROVENANCE_VERSION
        and model_event.get("status") == _ATTESTED
    ):
        model_snapshot = exact_model_snapshot_path(
            model_event.get("snapshot_path"),
            model_repo_id,
        )
        if model_snapshot is not None:
            event_load_mode = model_event.get("load_mode")
            snapshot_is_quantized = _snapshot_declares_quantization(Path(model_snapshot))
            if bool(config.get("load_in_4bit")):
                if (
                    event_load_mode in (None, _MODEL_LOAD_PREQUANTIZED_4BIT)
                    and snapshot_is_quantized
                ):
                    model_load_mode = _MODEL_LOAD_PREQUANTIZED_4BIT
                elif event_load_mode == _MODEL_LOAD_RUNTIME_4BIT and not snapshot_is_quantized:
                    model_load_mode = _MODEL_LOAD_RUNTIME_4BIT
            elif event_load_mode in (None, _MODEL_LOAD_UNQUANTIZED):
                model_load_mode = _MODEL_LOAD_UNQUANTIZED
            if model_load_mode is None:
                model_snapshot = None
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
            "model_load_mode": model_load_mode,
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


def validate_exact_model_pin(config: dict[str, Any]) -> str:
    marker = config.get(RESOURCE_PROVENANCE_KEY)
    stored_load_mode = config.get("resume_model_load_mode")
    if stored_load_mode is None and isinstance(marker, dict):
        stored_load_mode = marker.get("model_load_mode")
    load_in_4bit = bool(config.get("load_in_4bit"))
    if load_in_4bit:
        if stored_load_mode is None:
            model_load_mode = _MODEL_LOAD_PREQUANTIZED_4BIT
        elif stored_load_mode in {
            _MODEL_LOAD_PREQUANTIZED_4BIT,
            _MODEL_LOAD_RUNTIME_4BIT,
        }:
            model_load_mode = stored_load_mode
        else:
            model_load_mode = None
    elif stored_load_mode in (None, _MODEL_LOAD_UNQUANTIZED):
        model_load_mode = _MODEL_LOAD_UNQUANTIZED
    else:
        model_load_mode = None

    if model_load_mode is None:
        raise ExactResumeResourcesUnavailable(
            "The exact model snapshot for this run is no longer available."
        )
    model_repo_id = config.get("actual_model_repo_id")
    if model_repo_id is None:
        from utils.utils import canonical_model_repo_id
        model_repo_id = canonical_model_repo_id(str(config.get("model_name") or ""))
    model_snapshot = exact_model_snapshot_path(
        config.get("model_snapshot_path"),
        model_repo_id,
        require_quantized = model_load_mode == _MODEL_LOAD_PREQUANTIZED_4BIT,
    )
    if (
        model_snapshot is not None
        and model_load_mode == _MODEL_LOAD_RUNTIME_4BIT
        and _snapshot_declares_quantization(Path(model_snapshot))
    ):
        model_snapshot = None
    if model_snapshot is None:
        raise ExactResumeResourcesUnavailable(
            "The exact model snapshot for this run is no longer available."
        )
    return model_snapshot


def validate_exact_dataset_pin(config: dict[str, Any]) -> str:
    dataset_snapshot = exact_dataset_snapshot_path(
        config.get("dataset_snapshot_path"),
        config.get("hf_dataset"),
    )
    if dataset_snapshot is None:
        raise ExactResumeResourcesUnavailable(
            "The exact dataset snapshot for this run is no longer available."
        )
    return dataset_snapshot


def validate_exact_resource_pins(config: dict[str, Any]) -> tuple[str, str]:
    model_snapshot = validate_exact_model_pin(config)
    dataset_snapshot = validate_exact_dataset_pin(config)
    return model_snapshot, dataset_snapshot


def _provenance_awaiting_attestation(marker: dict[str, Any], config: dict[str, Any]) -> bool:
    """Training stopped before the worker attested loaded hub resources.

    Stop-and-save can finish while provenance is still the initial ``pending`` marker
    written at run start. Those runs have a valid checkpoint but no attested revision
    pins yet; resume should behave like a legacy run without exact resource requirements.
    """
    if marker.get("status") != "pending":
        return False
    if marker.get("model_status") is not None or marker.get("dataset_status") is not None:
        return False
    if config.get("actual_model_repo_id"):
        return False
    if config.get("model_snapshot_path") or config.get("dataset_snapshot_path"):
        return False
    return True


def exact_resume_resource_requirements(config: dict[str, Any]) -> tuple[bool, bool]:
    marker = config.get(RESOURCE_PROVENANCE_KEY)
    if marker is None:
        return False, False
    if (
        not isinstance(marker, dict)
        or marker.get("version") != RESOURCE_PROVENANCE_VERSION
        or marker.get("status") not in {"pending", "incomplete", "complete"}
    ):
        raise ExactResumeResourcesUnavailable("The resource provenance is invalid.")
    if _provenance_awaiting_attestation(marker, config):
        return False, False

    from utils.paths import is_local_path

    actual_model_repo_id = _normalized_repo_id(config.get("actual_model_repo_id"))
    if actual_model_repo_id is not None:
        require_model = True
    else:
        model_source = config.get("model_name")
        model_repo_id = _normalized_repo_id(model_source)
        require_model = model_repo_id is not None and not is_local_path(str(model_source))
    require_dataset = _normalized_repo_id(config.get("hf_dataset")) is not None

    if require_model:
        if marker.get("model_status") != _ATTESTED:
            raise ExactResumeResourcesUnavailable(
                "The model revision used by this run was not attested."
            )
        validate_exact_model_pin(config)
    if require_dataset:
        if marker.get("dataset_status") != _ATTESTED:
            raise ExactResumeResourcesUnavailable(
                "The dataset revision used by this run was not attested."
            )
        validate_exact_dataset_pin(config)
    return require_model, require_dataset


def resource_provenance_allows_resume(config: dict[str, Any]) -> bool:
    return resource_provenance_resume_blocker(config) is None


def resource_provenance_resume_blocker(config: dict[str, Any]) -> Optional[str]:
    """Why this provenance refuses a resume, or None when it allows one.

    ``exact_resume_resource_requirements`` already raises with a precise, user-facing
    explanation ("the exact model snapshot for this run is no longer available", and so
    on). Discarding it left the start route reporting a generic checkpoint complaint for
    a run whose checkpoint is perfectly intact, which points at the wrong thing entirely.
    """
    marker = config.get(RESOURCE_PROVENANCE_KEY)
    if marker is None:
        return None
    try:
        exact_resume_resource_requirements(config)
    except ExactResumeResourcesUnavailable as exc:
        return str(exc) or "The resources this run was trained from are no longer available."
    status = marker.get("status")
    if status in {"pending", "incomplete", "complete"}:
        return None
    return (
        f"This run's recorded resource provenance is not in a resumable state (status: {status!r})."
    )
