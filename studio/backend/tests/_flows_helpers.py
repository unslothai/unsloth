# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small builders for Flow documents in tests."""

from __future__ import annotations

import asyncio
import functools
from typing import Any


def node(
    nid: str,
    type_: str,
    group: str | None = None,
    **params: Any,
) -> dict:
    found = {"id": nid, "type": type_, "v": 1, "params": params, "pos": [0, 0]}
    if group:
        found["group"] = group
    return found


def edge(
    src: str,
    sport: str,
    dst: str,
    dport: str,
    kind: str = "data",
) -> dict:
    return {
        "id": f"{src}-{sport}-{dst}-{dport}",
        "source": src,
        "source_port": sport,
        "target": dst,
        "target_port": dport,
        "kind": kind,
    }


def doc(
    nodes: list[dict],
    edges: list[dict] = (),
    **extra: Any,
) -> dict:
    found = {
        "format": "unsloth-flow",
        "version": 1,
        "id": "f1",
        "name": "Test flow",
        "nodes": nodes,
        "edges": list(edges),
    }
    found.update(extra)
    return found


def run_async(fn):
    """Run an async test on a fresh loop (the suite has no pytest-asyncio)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper
