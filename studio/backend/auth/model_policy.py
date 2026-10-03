# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-managed model access policy: models non-owner accounts may not load.

Persisted as JSON under the studio root. Entries are case-insensitive model identifiers
(repo ids or paths); the owner is never restricted.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def _policy_path() -> Path:
    from utils.paths.storage_roots import studio_root

    return studio_root() / "admin_model_policy.json"


def _normalize(entries) -> list[str]:
    seen: dict[str, str] = {}
    for entry in entries or []:
        value = str(entry).strip()
        if value:
            seen.setdefault(value.lower(), value)
    return sorted(seen.values(), key = str.lower)


def get_blocked_models() -> list[str]:
    with _lock:
        try:
            data = json.loads(_policy_path().read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            return []
    return _normalize(data.get("blocked_models") if isinstance(data, dict) else [])


def set_blocked_models(entries) -> list[str]:
    blocked = _normalize(entries)
    path = _policy_path()
    with _lock:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"blocked_models": blocked}, indent = 2), encoding = "utf-8")
        os.replace(tmp, path)
    return blocked


def is_model_blocked(model_ref: str | None) -> bool:
    ref = (model_ref or "").strip().lower()
    if not ref:
        return False
    return ref in {entry.lower() for entry in get_blocked_models()}
