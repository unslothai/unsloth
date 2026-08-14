# SPDX-License-Identifier: AGPL-3.0-only
"""Dataset provenance and non-executable bundles for portable training resumes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

PORTABLE_DATA_VERSION = 1
PORTABLE_DATA_DIR = "portable_datasets"
VALID_MODES = frozenset(("metadata", "pinned", "snapshot"))


class PortableDatasetError(ValueError):
    pass


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_hash(item).encode())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def resolve_hub_revision(repository: str, *, token: str | None = None) -> str:
    """Resolve the repository's current ref to an immutable dataset commit."""
    from huggingface_hub import HfApi

    info = HfApi(token = token).dataset_info(repo_id = repository)
    if not info.sha:
        raise PortableDatasetError(f"Hugging Face did not return a commit for '{repository}'.")
    return str(info.sha)


def pin_hub_datasets(config: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve every HF input once and put its revision beside its descriptor."""
    token = config.get("hf_token") or None
    pinned: list[dict[str, str]] = []
    repository = config.get("hf_dataset")
    if repository:
        revision = config.get("dataset_revision") or resolve_hub_revision(repository, token = token)
        config["dataset_revision"] = revision
        pinned.append({"repository": repository, "revision": revision})
    for entry in config.get("training_datasets") or []:
        repository = entry.get("hf_dataset")
        if repository:
            revision = entry.get("revision") or resolve_hub_revision(repository, token = token)
            entry["revision"] = revision
            pinned.append({"repository": repository, "revision": revision})
    return pinned


def package_local_sources(config: dict[str, Any], output_dir: str | Path) -> list[dict[str, Any]]:
    """Copy local inputs into the run; never preserve notebook-temporary paths."""
    root = Path(output_dir) / PORTABLE_DATA_DIR / "sources"
    records: list[dict[str, Any]] = []
    for key in ("local_datasets", "local_eval_datasets"):
        bundled: list[str] = []
        for index, source_name in enumerate(config.get(key) or []):
            source = Path(source_name).expanduser()
            if not source.exists():
                raise PortableDatasetError(f"Local dataset source is missing: {source}")
            destination = root / key / f"{index:04d}" / source.name
            destination.parent.mkdir(parents = True, exist_ok = True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok = True)
                digest = tree_hash(destination)
            else:
                shutil.copy2(source, destination)
                digest = file_hash(destination)
            relative = destination.relative_to(Path(output_dir)).as_posix()
            bundled.append(relative)
            records.append({"role": key, "relative_path": relative, "sha256": digest})
        if bundled:
            config[key] = bundled
    return records


def snapshot_processed_datasets(
    output_dir: str | Path,
    train_dataset: Any,
    eval_dataset: Any | None,
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    """Save final Arrow datasets. ``save_to_disk`` stores data/schema, never scripts."""
    root = Path(output_dir) / PORTABLE_DATA_DIR / "snapshot-v1"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents = True)
    splits: dict[str, str] = {}
    for name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        if dataset is None:
            continue
        if not hasattr(dataset, "save_to_disk"):
            raise PortableDatasetError(
                "A streaming dataset cannot be fully offline portable unless a bounded "
                "selection is explicitly materialized before snapshotting."
            )
        destination = root / name
        dataset.save_to_disk(str(destination))
        splits[name] = destination.relative_to(Path(output_dir)).as_posix()
    metadata = {
        "bundle_version": PORTABLE_DATA_VERSION,
        "splits": splits,
        "preprocessing": dict(preprocessing),
    }
    (root / "bundle.json").write_text(json.dumps(metadata, indent = 2, sort_keys = True) + "\n")
    metadata["content_hash"] = tree_hash(root)
    return metadata


def verify_snapshot(output_dir: str | Path, snapshot: Mapping[str, Any]) -> None:
    root = Path(output_dir) / PORTABLE_DATA_DIR / "snapshot-v1"
    actual = tree_hash(root)
    if actual != snapshot.get("content_hash"):
        raise PortableDatasetError(
            f"Portable dataset snapshot hash mismatch: expected {snapshot.get('content_hash')}, got {actual}."
        )


def load_snapshot_for_resume(resume_checkpoint: str | Path) -> tuple[Any, Any | None] | None:
    """Load a processed snapshot located beside a resumed checkpoint.

    The bundle is discovered from the checkpoint rather than from the request's
    dataset descriptors.  This is important for copied checkpoints, where the
    original local paths (or even the original Hub repository) may no longer be
    reachable.
    """
    checkpoint = Path(resume_checkpoint).expanduser().resolve()
    run_dir = checkpoint.parent if checkpoint.name.startswith("checkpoint-") else checkpoint
    bundle_path = run_dir / PORTABLE_DATA_DIR / "snapshot-v1" / "bundle.json"
    if not bundle_path.is_file():
        return None
    try:
        metadata = json.loads(bundle_path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableDatasetError(f"Portable dataset snapshot metadata is invalid: {exc}") from exc
    if metadata.get("bundle_version") != PORTABLE_DATA_VERSION:
        raise PortableDatasetError(
            f"Unsupported portable dataset snapshot version: {metadata.get('bundle_version')}"
        )

    from datasets import load_from_disk

    loaded: dict[str, Any] = {}
    for split in ("train", "eval"):
        relative = (metadata.get("splits") or {}).get(split)
        if not relative:
            continue
        destination = (run_dir / relative).resolve()
        if run_dir != destination and run_dir not in destination.parents:
            raise PortableDatasetError(f"Portable snapshot path escapes the run: {relative}")
        if not destination.is_dir():
            raise PortableDatasetError(f"Portable snapshot split is missing: {destination}")
        loaded[split] = load_from_disk(str(destination))
    if "train" not in loaded:
        raise PortableDatasetError("Portable dataset snapshot has no train split.")
    return loaded["train"], loaded.get("eval")


def resolve_bundled_paths(config: dict[str, Any], run_dir: str | Path) -> None:
    """Resolve manifest-relative paths before considering original locations."""
    root = Path(run_dir).resolve()
    for key in ("local_datasets", "local_eval_datasets"):
        resolved = []
        for value in config.get(key) or []:
            candidate = Path(value)
            bundled = (root / candidate).resolve() if not candidate.is_absolute() else candidate
            if not candidate.is_absolute() and root not in bundled.parents:
                raise PortableDatasetError(f"Bundled dataset path escapes the run: {value}")
            resolved.append(str(bundled if bundled.exists() else candidate))
        config[key] = resolved


def verify_hub_revisions(config: Mapping[str, Any], *, token: str | None = None) -> None:
    """Fail exact resume early; replacement must be an explicit continued-training choice."""
    from huggingface_hub import HfApi

    pairs = []
    if config.get("hf_dataset") and config.get("dataset_revision"):
        pairs.append((config["hf_dataset"], config["dataset_revision"]))
    pairs.extend(
        (entry["hf_dataset"], entry["revision"])
        for entry in config.get("training_datasets") or []
        if entry.get("hf_dataset") and entry.get("revision")
    )
    api = HfApi(token = token)
    for repository, revision in pairs:
        try:
            api.dataset_info(repo_id = repository, revision = revision)
        except Exception as exc:
            raise PortableDatasetError(
                f"Recorded dataset revision {repository}@{revision} is unavailable. "
                "Replacing it is continued training, not an exact resume."
            ) from exc
