# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Credential-free runtime capture and per-turn admission for project tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from contextlib import asynccontextmanager

from .task_state import TaskStateError


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys = True).encode()).hexdigest()


def _provider_digest(config: dict) -> str:
    return _digest(
        {
            key: config.get(key)
            for key in (
                "id",
                "provider_type",
                "base_url",
                "is_enabled",
                "models",
                "updated_at",
                "max_output_tokens",
            )
        }
    )


def _llama(model: str):
    from core.inference.model_ids import model_id_matches

    # The serving route owns the existing singleton. Do not import that route
    # here: importing it on demand constructs a backend and runs startup cleanup.
    route = sys.modules.get("routes.inference")
    backend = route.get_llama_cpp_backend() if route is not None else None
    if (
        backend is None
        or not backend.is_loaded
        or not any(
            model_id_matches(model, getattr(backend, key, None))
            for key in ("model_identifier", "_openai_advertised_id", "hf_repo")
        )
    ):
        raise TaskStateError("Load the selected GGUF model before starting a task.")
    if not getattr(backend, "idle_slot_clearing_active", False):
        raise TaskStateError(
            "This GGUF runtime cannot release idle model slots safely for child tasks."
        )
    return backend


def _llama_identity(backend) -> str:
    # Process-local identity deliberately invalidates queued work after a server
    # restart. Explicit retries still validate this snapshot; they never load a model.
    return _digest(
        [
            id(backend),
            backend.base_url,
            str(getattr(backend, "model_identifier", "")),
            str(getattr(backend, "_process", None)),
        ]
    )


def capture_runtime(
    kind: str,
    model: str,
    provider_id: str | None = None,
) -> dict:
    if (
        not isinstance(model, str)
        or not 1 <= len(model) <= 512
        or any(c in model for c in "\0\r\n")
    ):
        raise TaskStateError("Select a model for this task.")
    if kind == "local" and provider_id is None:
        backend = _llama(model)
        return {"kind": kind, "model": model, "identity": _llama_identity(backend)}
    if kind != "provider" or not provider_id:
        raise TaskStateError("Select a saved provider or a loaded GGUF model.")
    from storage import providers_db, credential_secrets
    from core.inference.providers import provider_model_runs_local_tools

    config = providers_db.get_provider(provider_id)
    if not config or not config.get("is_enabled") or model not in (config.get("models") or []):
        raise TaskStateError("The selected provider model is unavailable.")
    provider_type = config["provider_type"]
    if provider_type == "openai_codex":
        raise TaskStateError(
            "Subscription tasks are unavailable because that endpoint does not accept an output token limit."
        )
    if not provider_model_runs_local_tools(provider_type, model):
        raise TaskStateError("This provider model does not support project tools.")
    result = {
        "kind": kind,
        "model": model,
        "providerId": provider_id,
        "providerType": provider_type,
        "routing": _provider_digest(config),
        "credential": credential_secrets.get_provider_api_key_binding(provider_id),
    }
    validate_runtime(result)
    return result


def validate_runtime(snapshot: dict):
    if snapshot.get("kind") == "local":
        backend = _llama(snapshot["model"])
        if _llama_identity(backend) != snapshot.get("identity"):
            raise TaskStateError("The loaded runtime changed. Start a new task.")
        return backend
    from storage import providers_db, credential_secrets

    config = providers_db.get_provider(snapshot["providerId"])
    if not config or _provider_digest(config) != snapshot.get("routing"):
        raise TaskStateError("The provider configuration changed. Start a new task.")
    if credential_secrets.get_provider_api_key_binding(snapshot["providerId"]) != snapshot.get(
        "credential"
    ):
        raise TaskStateError("The provider credential changed. Start a new task.")
    return config


@asynccontextmanager
async def _admission(
    backend,
    context,
    *,
    messages = None,
    tools = None,
    max_tokens = 1024,
):
    if backend is None:
        yield
        return
    from core.inference.llama_admission import (
        get_llama_admission_queue,
        llama_admission_config_from_env,
    )
    from core.inference.context_window import estimate_messages_tokens_dense

    # Use the aggregate cache, not the per-slot window. Supplying budget=None
    # would reset the shared queue's accounting for ordinary chat requests too.
    budget = getattr(backend, "_kv_cache_context_total", None) or getattr(
        backend, "context_length", None
    )
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise TaskStateError("The loaded GGUF runtime has no usable context budget.")
    cost = (
        estimate_messages_tokens_dense(messages or [])
        + estimate_messages_tokens_dense(tools or [])
        + max_tokens
    )

    reservation = get_llama_admission_queue(backend.base_url).reserve(
        capacity = int(getattr(backend, "effective_parallel_slots", None) or 1),
        config = llama_admission_config_from_env(),
        budget = budget,
        tokens = max(1, min(budget, cost)),
    )
    lease = None
    try:
        while lease is None:
            context.check()
            lease = reservation.lease_nowait()
            if lease is None:
                try:
                    lease = await reservation.wait(0.1)
                except asyncio.TimeoutError:
                    pass
        yield
    finally:
        if lease is not None:
            lease.release()
        reservation.cancel()


class TaskTransport:
    """A model slot ends before the shared loop invokes any task tool.

    Debit the requested cap before each turn, without refunds for missing usage.
    Thus the sum of all request caps never exceeds the attempt's reservation.
    """

    heals_text_tool_calls = False
    sanitizes_provider_frames = True

    def __init__(self, context):
        self.context = context
        self.snapshot = context.task["snapshot"]["runtime"]
        self.remaining = context.task["maxOutputTokens"]
        self.reserved = 0

    async def stream(self, *, messages, tools, tool_choice, cancel_event):
        from core.inference.external_provider import ExternalProviderClient
        from core.inference.external_tool_transport import OAICompatTransport
        from core.inference.providers import validate_provider_base_url
        from storage import credential_secrets

        self.context.check()
        if self.remaining <= 0:
            raise TaskStateError("Task output budget exhausted.")
        runtime = validate_runtime(self.snapshot)
        local = self.snapshot["kind"] == "local"
        cap = min(1024, self.remaining)
        if not local and runtime.get("max_output_tokens") is not None:
            provider_cap = runtime["max_output_tokens"]
            if (
                isinstance(provider_cap, bool)
                or not isinstance(provider_cap, int)
                or provider_cap <= 0
            ):
                raise TaskStateError("The provider output token limit is invalid.")
            cap = min(cap, provider_cap)
        self.remaining -= cap
        self.reserved += cap
        backend = runtime if local else None
        if local:
            provider_type, base_url, api_key = "llama_cpp", runtime.base_url + "/v1", ""
        else:
            provider_type = runtime["provider_type"]
            base_url = validate_provider_base_url(runtime["base_url"])
            api_key, binding = credential_secrets.get_provider_api_key_with_binding(
                self.snapshot["providerId"]
            )
            if binding != self.snapshot["credential"]:
                raise TaskStateError("The provider credential changed. Start a new task.")
            validate_runtime(self.snapshot)
        client = ExternalProviderClient(
            provider_type = provider_type, base_url = base_url, api_key = api_key or ""
        )
        transport = OAICompatTransport(
            client, model = self.snapshot["model"], max_tokens = cap, temperature = 0.2, stream = True
        )
        try:
            async with _admission(
                backend, self.context, messages = messages, tools = tools, max_tokens = cap
            ):
                self.context.check()
                validate_runtime(self.snapshot)
                stream = transport.stream(
                    messages = messages,
                    tools = tools,
                    tool_choice = tool_choice,
                    cancel_event = cancel_event,
                )
                try:
                    async for line in stream:
                        self.context.check()
                        yield line
                finally:
                    await stream.aclose()
        finally:
            await client.close()


__all__ = ["capture_runtime", "validate_runtime", "TaskTransport"]
