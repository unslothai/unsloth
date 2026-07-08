# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tool implementations, grouped by Studio feature.

Each module exposes ``register(mcp: FastMCP) -> list[str]`` which mounts that
feature's tools onto a FastMCP server and returns the tool names it added.
The handler functions are plain module-level callables so they can be unit
tested directly (with the backend singletons mocked) without a live MCP
transport or a GPU.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from fastmcp import FastMCP

from . import data, export, models, recipe, train

# Ordered (group name -> registrar) so --enable/--disable and the dry-run
# listing present features in a stable, documented order.
_REGISTRARS: dict[str, Callable[[FastMCP], list[str]]] = {
    "models": models.register,
    "data": data.register,
    "train": train.register,
    "export": export.register,
    "recipe": recipe.register,
}

FEATURE_GROUPS: list[str] = list(_REGISTRARS.keys())


def resolve_groups(
    enabled: Optional[Sequence[str]] = None, disabled: Optional[Sequence[str]] = None
) -> list[str]:
    """Compute the final ordered feature-group list from --enable/--disable.

    Unknown names raise ``ValueError`` so a typo is surfaced to the operator
    rather than silently enabling everything.
    """
    known = set(FEATURE_GROUPS)
    for name in list(enabled or ()) + list(disabled or ()):
        if name not in known:
            raise ValueError(
                f"Unknown feature group {name!r}. Valid groups: {', '.join(FEATURE_GROUPS)}"
            )

    selected = set(FEATURE_GROUPS) if enabled is None else set(enabled)
    selected -= set(disabled or ())
    return [g for g in FEATURE_GROUPS if g in selected]


def register_all(
    mcp: FastMCP,
    enabled: Optional[Sequence[str]] = None,
    disabled: Optional[Sequence[str]] = None,
) -> list[str]:
    """Register every selected feature group's tools onto ``mcp``.

    Returns the flattened list of registered tool names (used by the dry-run
    helper).
    """
    names: list[str] = []
    for group in resolve_groups(enabled, disabled):
        names.extend(_REGISTRARS[group](mcp))
    return names
