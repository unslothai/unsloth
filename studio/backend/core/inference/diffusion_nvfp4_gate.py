# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The checked-in record of which NVFP4 checkpoints actually passed the accuracy gate.

NVFP4 is the one scheme whose per-layer policy makes the render quality a property of ONE
artifact rather than of the scheme: ``zimg_f8mod_toq34_v1`` on z-image turbo is a measured model,
``nvfp4`` on an unmeasured base is a guess. So the auto ladder does not ask "can this GPU run fp4",
it asks THIS module: is there a reviewed record, in the tree, saying the 28-pair gate passed for
this family on this base at the policy this commit resolves.

The record is data, not code, and it is checked in beside the policy tables
(``nvfp4_gate_record.json``): a gate run is evidence produced on a GPU nobody else has, and the
only way it reaches an install is as a reviewed file. ``scripts/record_nvfp4_gate.py`` writes an
entry from a gate ``results.json``; nothing writes it at runtime.

The verdict is re-derived, never trusted as stored. ``nvfp4_gate_passed`` resolves the in-tree
policy for ``(family, base)`` and compares its ``(policy_id, policy_version)`` against the
record's, so retuning a policy (a version bump, a renamed id, a base that loses its row)
invalidates every verdict measured on the old one instead of silently carrying it forward onto
layer precisions no gate ever saw. An empty ``records`` list -- the shipped state -- means every
answer is False, which is exactly the pre-measurement behaviour.

Pure and torch-free: this is consulted from the auto ladder, which also runs inside the
stdlib-only smoke-probe child, so it may not pull torch in and every failure is a False.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# The record file's own schema version: the SHAPE of the entries, bumped when a field is added or
# repurposed, independent of any policy's version.
GATE_RECORD_VERSION = 1

# Beside the policy tables, so a policy change and its evidence are read (and reviewed) together.
GATE_RECORD_PATH = Path(__file__).with_name("nvfp4_gate_record.json")

# The fields ``scripts/record_nvfp4_gate.py`` writes, and the identity a record is keyed on. Kept
# here rather than in the script so the reader and the writer cannot disagree about either.
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

# (path, mtime_ns, size) -> parsed records. Keyed on the stat so a test that writes a temp record
# file, or an editor that saves one, is picked up without a process restart, while the shipped file
# is parsed once.
_CACHE: dict[tuple, tuple] = {}


def _canonical(value: Any) -> str:
    """A base repo id in the form the tables hold: mirror mapped to upstream, lowercased."""
    try:
        from .diffusion_families import canonical_base
        return canonical_base(str(value or "").strip()).strip().lower()
    except Exception:  # noqa: BLE001 -- an unimportable registry just means "no canonicalisation"
        return str(value or "").strip().lower()


def load_gate_records(path: Any = None) -> tuple:
    """Every record in the gate file, or an empty tuple when there is none to read.

    Never raises: an absent, unreadable or malformed file is the same answer as an empty one --
    no evidence -- and a parse error must not take down a load that would otherwise run fp8.
    """
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


def nvfp4_gate_record(
    family: Any,
    base_repo: Any,
    policy_id: Optional[str] = None,
    *,
    path: Any = None,
) -> Optional[dict]:
    """The gate record for ``(family, base_repo)``, optionally pinned to ``policy_id``, or None.

    The base is canonicalised first (a local mirror is the same weights the gate ran on). Without
    a base there is no record: a gate ran against one checkpoint's weights, and letting a family
    inherit a sibling base's verdict is the exact failure this file exists to prevent.
    """
    fam = str(family or "").strip().lower()
    base = _canonical(base_repo)
    if not fam or not base:
        return None
    wanted = str(policy_id).strip() if policy_id is not None else None
    for record in load_gate_records(path):
        if str(record.get("family", "")).strip().lower() != fam:
            continue
        if _canonical(record.get("base_repo")) != base:
            continue
        if wanted is not None and str(record.get("policy_id", "")).strip() != wanted:
            continue
        return dict(record)
    return None


def nvfp4_gate_passed(
    family: Any,
    base_repo: Any,
    *,
    path: Any = None,
) -> bool:
    """Whether a reviewed record says the NVFP4 gate PASSED for this family on this base, at the
    policy this commit resolves.

    Four ways to answer False, all of them "nothing measured this model": no policy resolves for
    the pair, no record covers it, the record says the run did not pass, or the record was taken
    against a different ``(policy_id, policy_version)`` than the tree now resolves.
    """
    try:
        from .diffusion_nvfp4_policy import resolve_policy
        policy = resolve_policy(family, base_repo)
    except Exception:  # noqa: BLE001 -- an unresolvable policy is "not gated", never a load error
        return False
    if policy is None:
        return False
    record = nvfp4_gate_record(family, base_repo, policy.policy_id, path = path)
    if record is None or record.get("all_pass") is not True:
        return False
    try:
        recorded = (str(record.get("policy_id", "")).strip(), int(record.get("policy_version")))
    except (TypeError, ValueError):
        return False
    return recorded == (str(policy.policy_id), int(policy.version))
