# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checked-in NVFP4 accuracy-gate verdicts per (family, base, policy); torch-free (the smoke probe reads it)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

GATE_RECORD_VERSION = 1

GATE_RECORD_PATH = Path(__file__).with_name("nvfp4_gate_record.json")

RECORD_KEY_FIELDS = ("family", "base_repo", "policy_id", "policy_version", "checkpoint_sha256")
RECORD_FIELDS = RECORD_KEY_FIELDS + (
    "repo_id",
    "filename",
    "gptq",
    "all_pass",
    "num_pairs",
    "num_passed",
    "aggregates",
    "gates",
    "resolutions",
    "seeds",
    "cuda_graphs",
    "backend",
    "gpu",
    "versions",
    "gate_script_sha256",
    "run_date",
    "results_path",
)

# Keyed on the stat, so a rewritten file is picked up without a restart.
_CACHE: dict[tuple, tuple] = {}


def _canonical(value: Any) -> str:
    try:
        from .diffusion_families import canonical_base
        return canonical_base(str(value or "").strip()).strip().lower()
    except Exception:  # noqa: BLE001 -- an unimportable registry just means "no canonicalisation"
        return str(value or "").strip().lower()


def load_gate_records(path: Any = None) -> tuple:
    """Never raises: an unreadable file is no evidence."""
    target = Path(path) if path is not None else GATE_RECORD_PATH
    try:
        stat = target.stat()
        key = (str(target), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return ()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        parsed = json.loads(target.read_text(encoding = "utf-8"))
        records = parsed.get("records") if isinstance(parsed, dict) else None
        entries = tuple(record for record in (records or []) if isinstance(record, dict))
    except Exception:  # noqa: BLE001 -- see the docstring: no evidence, not an error
        entries = ()
    _CACHE[key] = entries
    return entries


def nvfp4_gate_records(
    family: Any,
    base_repo: Any,
    policy_id: Optional[str] = None,
    *,
    path: Any = None,
) -> tuple:
    """Gate records for ``(family, base_repo)``, optionally pinned to ``policy_id``; none without a base."""
    fam = str(family or "").strip().lower()
    base = _canonical(base_repo)
    if not fam or not base:
        return ()
    wanted = str(policy_id).strip() if policy_id is not None else None
    matched = []
    for record in load_gate_records(path):
        if str(record.get("family", "")).strip().lower() != fam:
            continue
        if _canonical(record.get("base_repo")) != base:
            continue
        if wanted is not None and str(record.get("policy_id", "")).strip() != wanted:
            continue
        matched.append(dict(record))
    return tuple(matched)


def nvfp4_gate_record(
    family: Any,
    base_repo: Any,
    policy_id: Optional[str] = None,
    *,
    path: Any = None,
) -> Optional[dict]:
    """The first gate record for ``(family, base_repo)``, or None (not whether nvfp4 is allowed)."""
    matched = nvfp4_gate_records(family, base_repo, policy_id, path = path)
    return matched[0] if matched else None


def _passing_records(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> tuple:
    """Every PASS for this family and base at the resolved policy; a failure must not mask a later pass."""
    try:
        from .diffusion_nvfp4_policy import resolve_policy
        policy = resolve_policy(family, base_repo)
    except Exception:  # noqa: BLE001 -- an unresolvable policy is "not gated", never a load error
        return ()
    if policy is None:
        return ()
    passing = []
    for record in nvfp4_gate_records(family, base_repo, policy.policy_id, path = path):
        if record.get("all_pass") is not True:
            continue
        try:
            recorded = (str(record.get("policy_id", "")).strip(), int(record.get("policy_version")))
        except (TypeError, ValueError):
            continue
        if recorded == (str(policy.policy_id), int(policy.version)):
            passing.append(record)
    return tuple(passing)


def _passing_record(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> Optional[dict]:
    passing = _passing_records(family, base_repo, path = path)
    return passing[0] if passing else None


def nvfp4_gate_passed(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> bool:
    return _passing_record(family, base_repo, path = path) is not None


def nvfp4_gate_backends(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> tuple:
    """Every NVFP4 backend a passing record was measured on, lowercased, deduplicated, in file order."""
    backends: list[str] = []
    for record in _passing_records(family, base_repo, path = path):
        backend = str(record.get("backend") or "").strip().lower()
        if backend and backend not in backends:
            backends.append(backend)
    return tuple(backends)


def nvfp4_gate_backend(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> Optional[str]:
    """The first passing record's backend. Coverage wants ``nvfp4_gate_backends``."""
    backends = nvfp4_gate_backends(family, base_repo, path = path)
    return backends[0] if backends else None
