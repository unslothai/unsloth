# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Node output cache keys: the node type and version, its normalized params, its input values, the resolved model and
the effective seed. Layout, labels and run progress never enter a key, so moving a node never invalidates it."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

CACHE_VERSION = 1


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys = True, separators = (",", ":"), default = str)


def _value_identity(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("kind") in ("image", "audio", "video") and isinstance(value.get("ref"), str):
            return {"kind": value["kind"], "ref": value["ref"]}
        return {k: _value_identity(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_value_identity(v) for v in value]
    return value


def cache_key(
    node_type: str,
    version: int,
    params: dict,
    inputs: dict,
    *,
    model: Optional[dict],
    seed: Optional[int],
    flow_params: Optional[dict] = None,
    extra: Any = None,
) -> str:
    material = {
        "c": CACHE_VERSION,
        "t": node_type,
        "v": version,
        "p": _value_identity(params),
        "i": _value_identity(inputs),
        "m": model,
        "s": seed,
    }
    if flow_params is not None:
        material["f"] = _value_identity(flow_params)
    if extra is not None:
        material["x"] = extra
    return hashlib.sha256(canonical(material).encode()).hexdigest()
