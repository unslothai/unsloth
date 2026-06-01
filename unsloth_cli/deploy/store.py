# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist provider credentials/settings so they're entered once.

Saved per provider, user-only (chmod 600) since they hold tokens and secrets:

    {"runpod": {"api_key": "rpa_...", "datacenter": "US-KS-2"}}
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def config_path() -> Path:
    """`~/.config/unsloth/deploy.json`, honoring XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "unsloth" / "deploy.json"


def _read_all() -> dict:
    try:
        return json.loads(config_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}


def load(provider_name: str) -> dict[str, str]:
    """Saved options for `provider_name`, or {} if none/unreadable."""
    section = _read_all().get(provider_name)
    return dict(section) if isinstance(section, dict) else {}


def save(provider_name: str, options: dict[str, str]) -> Path:
    """Persist `options` for `provider_name` (empty values dropped), merging into
    any other providers' saved settings. Returns the file path."""
    path = config_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    data = _read_all()
    data[provider_name] = {k: v for k, v in options.items() if v}
    path.write_text(json.dumps(data, indent = 2, sort_keys = True), encoding = "utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 -- holds tokens/secrets
    except OSError:
        pass
    return path
