# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Node handlers and the context the engine hands them.

A handler is `async def handler(ctx: NodeContext) -> dict[str, Any]` returning a value per output port.
Handlers never touch the database or the filesystem directly and never call route functions: every
Studio capability goes through `ctx.call*`, which forwards in-process to Studio's own routes as the run's
internal API key (the same path the MCP server uses), so auth, account scoping and redaction apply."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol

from core.flows.specs import SPECS_BY_TYPE


class FlowNodeError(Exception):
    """A node failed with a message meant for the user."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: Optional[int] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class FlowCancelled(Exception):
    pass


class NodeContext(Protocol):
    run_id: str
    flow_id: str
    node_id: str
    spec: dict
    params: dict[str, Any]
    inputs: dict[str, Any]
    seed: int
    flow_params: dict[str, Any]

    async def call(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[dict] = None,
        files: Any = None,
        data: Optional[dict] = None,
        content: Optional[bytes] = None,
        content_type: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> Any:
        """Return the parsed JSON body (or raw bytes for non-JSON). Raise FlowNodeError with the route's detail on non-2xx."""
        ...

    async def ensure_model(self, model_kind: str, choice: Optional[dict]) -> Optional[dict]:
        """Make `choice` ({"repo", "variant"}) the resident model for `model_kind`, loading it if needed.
        None = keep whatever is loaded (error if nothing is). Returns the resolved {"repo", "variant"} or None."""
        ...

    async def llm_text(
        self,
        *,
        model: Optional[dict],
        system: str,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        images: Optional[list[dict]] = None,
        thinking: bool = False,
        seed: Optional[int] = None,
        stream_port: Optional[str] = "text",
        use_adapter: Optional[bool] = None,
    ) -> str: ...

    async def llm_json(
        self,
        *,
        model: Optional[dict],
        system: str,
        prompt: str,
        schema: dict,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        images: Optional[list[dict]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        retry_bad_json: bool = True,
    ) -> Any:
        """Structured output (`response_format` json_schema); falls back to parsing the first JSON object in the text.
        With ``on_text`` the answer streams, and it gets the raw JSON written so far after every chunk."""
        ...

    async def generate_image(
        self,
        *,
        model: Optional[dict],
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        init_image: Optional[dict] = None,
        strength: Optional[float] = None,
        references: Optional[list[dict]] = None,
        workflow: Optional[str] = None,
        count: int = 1,
        upscale: Optional[float] = None,
        mask: Optional[dict] = None,
        control: Optional[dict] = None,
        options: Optional[dict] = None,
    ) -> list[dict]:
        """Returns MediaValue dicts (Studio gallery images). Unset ``options`` keys are not sent."""
        ...

    async def remove_background(
        self,
        image: dict,
        subject: str,
        trim: bool = True,
    ) -> tuple[dict, dict]:
        """Returns (cutout MediaValue, mask MediaValue)."""
        ...

    async def segment(self, image: dict, subject: str) -> bytes:
        """A grayscale mask PNG (white = ``subject``) at the image's size."""
        ...

    async def speak(
        self,
        *,
        model: Optional[dict],
        text: str,
        voice: str = "",
        instructions: str = "",
        reference: Optional[dict] = None,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> dict: ...

    async def media_bytes(self, media: dict) -> tuple[bytes, str]:
        """(bytes, mime) of any MediaValue: gallery item, embedded asset or flow asset."""
        ...

    async def run_flow(self, flow_id: str, values: dict) -> dict:
        """Run another saved flow of this account inside this run; returns its Flow Outputs by name."""
        ...

    async def media_data_url(self, media: dict) -> str: ...

    async def save_asset(
        self,
        data: bytes,
        mime: str,
        *,
        kind: str,
        name: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Store bytes as a flow asset and return its MediaValue."""
        ...

    def progress(
        self,
        fraction: Optional[float] = None,
        *,
        phase: Optional[str] = None,
        step: Optional[int] = None,
        total: Optional[int] = None,
        preview: Optional[str] = None,
    ) -> None: ...

    def log(
        self,
        message: str,
        level: str = "info",
    ) -> None: ...

    def text_delta(self, port: str, delta: str) -> None: ...

    def partial(
        self,
        port: str,
        value: Any,
        *,
        step: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None: ...

    @property
    def cancelled(self) -> bool: ...

    def check_cancelled(self) -> None:
        """Raise FlowCancelled if the run is being cancelled."""
        ...

    def on_cancel(self, fn: Callable[[], Awaitable[None]]) -> None:
        """Register a coroutine the engine awaits when the run is cancelled while this node is in flight."""
        ...

    origin: dict[str, Any]

    def capability_refusal(
        self,
        cap: str,
        detail: Optional[str] = None,
    ) -> Optional[str]:
        """Why this run may not use ``cap`` (python, http + host, mcp, provider + id), or None. A spec listing the
        capability in ``requires_approval`` is checked by the engine before the handler runs; call this for a
        capability a handler only sometimes uses (an agent turning on MCP tools)."""
        ...

    def capability_allowed(
        self,
        cap: str,
        detail: Optional[str] = None,
    ) -> bool: ...


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


def load_handlers() -> dict[str, Handler]:
    """Import every handler module once (registration is a side effect of import)."""
    from core.flows import nodes  # noqa: F401
    return HANDLERS
