"""Portable resume metadata and upload safety policy."""

from pathlib import Path
from typing import Any, Mapping


RESUME_FIELDS = frozenset({
    "run_id", "global_step", "model_name", "training_type", "base_model_name",
    "transformers_version", "unsloth_version", "checkpoint_name",
})
BLOCKED_NAMES = frozenset({"studio.db", "auth.db", "cookies", ".env"})
BLOCKED_SUFFIXES = ("-wal", "-shm")


def build_resume_manifest(values: Mapping[str, Any]) -> dict[str, Any]:
    """Build from an allowlist so future secrets cannot leak by default."""
    return {key: values[key] for key in RESUME_FIELDS if key in values}


def is_safe_upload_file(path: Path) -> bool:
    name = path.name.lower()
    if name in BLOCKED_NAMES or name.endswith(BLOCKED_SUFFIXES):
        return False
    if any(secret in name for secret in ("token", "credential", "secret")):
        return False
    return not any(part.endswith((".part", ".tmp")) for part in path.parts)


def upload_files(checkpoint_path: Path) -> list[Path]:
    return sorted(p for p in checkpoint_path.rglob("*") if p.is_file() and is_safe_upload_file(p))
