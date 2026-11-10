# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Node handlers and the context the engine hands them.

A handler is `async def handler(ctx: NodeContext) -> dict[str, Any]` returning a value per output port.
Handlers never touch the database or the filesystem directly and never call route functions: every
Studio capability goes through `ctx.call*`, which forwards in-process to Studio's own routes as the run's
internal API key (the same path the MCP server uses), so auth, account scoping and redaction apply."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from core.flows.specs import SPECS_BY_TYPE


class FlowNodeError(Exception):
    """A node failed with a message meant for the user."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: Optional[int] = None,
        refusal: Optional[str] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.refusal = refusal


class FlowCancelled(Exception):
    pass


if TYPE_CHECKING:
    # The engine's NodeContextImpl is the contract: ``ctx.llm_text``, ``ctx.generate_image`` and the other model
    # and media helpers take the keyword arguments of the functions of the same name in core.flows.nodes._common.
    from core.flows.engine import NodeContextImpl as NodeContext
else:
    NodeContext = Any


Handler = Callable[[NodeContext], Awaitable[dict[str, Any]]]

HANDLERS: dict[str, Handler] = {}


def handler(node_type: str) -> Callable[[Handler], Handler]:
    if node_type not in SPECS_BY_TYPE:
        raise KeyError(f"no spec for node type {node_type!r}")

    def deco(fn: Handler) -> Handler:
        if node_type in HANDLERS:
            raise KeyError(f"duplicate handler for {node_type!r}")
        HANDLERS[node_type] = fn
        return fn

    return deco


@dataclass(frozen = True)
class CacheHook:
    """For a node whose result lives outside the cache: `material` adds what the params and inputs
    can't show (a dataset's content hash; None = don't cache this run), `valid` drops a hit whose
    files are gone."""

    material: Callable[[NodeContext], Awaitable[Any]]
    valid: Callable[[NodeContext, dict], Awaitable[bool]]
    # False for a handler that never reads the run's flow parameters: they stay out of its key.
    flow_params: bool = True


CACHE_HOOKS: dict[str, CacheHook] = {}


def cache_hook(
    node_type: str,
    *,
    material: Callable,
    valid: Callable,
    flow_params: bool = True,
) -> None:
    CACHE_HOOKS[node_type] = CacheHook(material, valid, flow_params)


def load_handlers() -> dict[str, Handler]:
    """Import every handler module once (registration is a side effect of import)."""
    from core.flows import nodes  # noqa: F401
    return HANDLERS
