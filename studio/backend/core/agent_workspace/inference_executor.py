# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Production inference adapter for durable background coding-agent tasks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from storage import credential_secrets, providers_db, studio_db

from .common import AgentWorkspaceError


_RUNTIME_KINDS = frozenset({"local", "provider"})
_PERMISSION_MODES = frozenset({"ask", "auto", "off", "full"})
_SELECTION_KEYS = frozenset(
    {
        "kind",
        "model",
        "providerId",
        "permissionMode",
        "reasoningEffort",
        "maxOutputTokens",
    }
)
_SNAPSHOT_KEYS = _SELECTION_KEYS | frozenset(
    {"providerType", "routingDigest", "credentialBindingDigest"}
)
_KEY_OPTIONAL_PROVIDERS = frozenset({"custom", "llama_cpp", "ollama", "vllm"})
_MAX_GENERATION_TOKENS = 32_768
_DEFAULT_GENERATION_TOKENS = 8_192
_MAX_RESULT_BYTES = 900 * 1024
_MAX_PROJECT_INSTRUCTIONS = 24_000
_MAX_GOAL = 8_000
_MAX_PLAN = 24_000
_MAX_REPOSITORY_INSTRUCTIONS = 64 * 1024


def _bounded_positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, _MAX_GENERATION_TOKENS))


def _provider_routing_digest(config: Mapping[str, Any]) -> str:
    payload = {
        "id": str(config.get("id") or ""),
        "providerType": str(config.get("provider_type") or ""),
        "baseUrl": str(config.get("base_url") or ""),
        "enabled": bool(config.get("is_enabled")),
        "models": sorted(str(model) for model in (config.get("models") or [])),
        "updatedAt": str(config.get("updated_at") or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()


def _codex_account_digest(provider_id: str) -> str:
    from core.inference.openai_codex_auth import load_oauth_bundle

    bundle = load_oauth_bundle(provider_id)
    account_id = str((bundle or {}).get("account_id") or "")
    if not account_id:
        raise AgentWorkspaceError(
            "The selected ChatGPT subscription is unavailable. Reconnect it first."
        )
    return hashlib.sha256(account_id.encode("utf-8")).hexdigest()


def capture_runtime_snapshot(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and reduce a renderer selection to credential-free durable state."""
    if not isinstance(selection, Mapping):
        raise AgentWorkspaceError("A background agent runtime selection is required.")
    extras = set(selection) - _SELECTION_KEYS
    if extras:
        raise AgentWorkspaceError("Background agent runtime selection is invalid.")
    kind = str(selection.get("kind") or "").strip().lower()
    model = str(selection.get("model") or "").strip()
    permission_mode = str(selection.get("permissionMode") or "").strip().lower()
    reasoning_effort = str(selection.get("reasoningEffort") or "").strip() or None
    if kind not in _RUNTIME_KINDS:
        raise AgentWorkspaceError("Background agent runtime kind is invalid.")
    if not model or len(model) > 512 or any(char in model for char in "\x00\r\n"):
        raise AgentWorkspaceError("Background agent model selection is invalid.")
    if permission_mode not in _PERMISSION_MODES:
        raise AgentWorkspaceError("Background agent permission mode is invalid.")
    if reasoning_effort is not None and (
        len(reasoning_effort) > 64 or any(char in reasoning_effort for char in "\x00\r\n")
    ):
        raise AgentWorkspaceError("Background agent reasoning selection is invalid.")
    max_output_tokens = _bounded_positive_int(
        selection.get("maxOutputTokens"), _DEFAULT_GENERATION_TOKENS
    )

    if kind == "local":
        if selection.get("providerId") not in {None, ""}:
            raise AgentWorkspaceError("A local runtime cannot name an external provider.")
        return {
            "kind": "local",
            "model": model,
            "providerId": None,
            "providerType": "local",
            "permissionMode": permission_mode,
            "reasoningEffort": reasoning_effort,
            "maxOutputTokens": max_output_tokens,
        }

    provider_id = str(selection.get("providerId") or "").strip()
    if not provider_id or len(provider_id) > 256 or any(char in provider_id for char in "\x00\r\n"):
        raise AgentWorkspaceError("A saved provider connection is required.")
    config = providers_db.get_provider(provider_id)
    if config is None or not config.get("is_enabled"):
        raise AgentWorkspaceError("The selected provider connection is unavailable.")
    provider_type = str(config.get("provider_type") or "")
    from core.inference.providers import (
        get_provider_info,
        provider_model_runs_local_tools,
    )

    if get_provider_info(provider_type) is None:
        raise AgentWorkspaceError("The selected provider type is unavailable.")
    selected_models = {str(item) for item in (config.get("models") or [])}
    if model not in selected_models:
        raise AgentWorkspaceError("The selected model is not enabled for this connection.")
    if not provider_model_runs_local_tools(provider_type, model):
        raise AgentWorkspaceError("The selected provider model cannot run Studio coding tools.")
    snapshot = {
        "kind": "provider",
        "model": model,
        "providerId": provider_id,
        "providerType": provider_type,
        "permissionMode": permission_mode,
        "reasoningEffort": reasoning_effort,
        "maxOutputTokens": max_output_tokens,
        "routingDigest": _provider_routing_digest(config),
    }
    if provider_type == "openai_codex":
        snapshot["credentialBindingDigest"] = _codex_account_digest(provider_id)
    else:
        snapshot["credentialBindingDigest"] = credential_secrets.get_provider_api_key_binding(
            provider_id
        )
    return snapshot


def _validate_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise AgentWorkspaceError(
            "This background task has no runtime selection. Queue it again with a model."
        )
    if set(snapshot) - _SNAPSHOT_KEYS:
        raise AgentWorkspaceError("The saved background runtime selection is invalid.")
    kind = snapshot.get("kind")
    model = snapshot.get("model")
    permission_mode = snapshot.get("permissionMode")
    if (
        kind not in _RUNTIME_KINDS
        or not isinstance(model, str)
        or not model
        or permission_mode not in _PERMISSION_MODES
    ):
        raise AgentWorkspaceError("The saved background runtime selection is invalid.")
    credential_binding = snapshot.get("credentialBindingDigest")
    if kind == "provider" and (
        not isinstance(credential_binding, str)
        or len(credential_binding) != 64
        or any(char not in "0123456789abcdef" for char in credential_binding)
    ):
        raise AgentWorkspaceError("This provider task predates credential binding. Queue it again.")
    # Background jobs have no interactive approval stream. Waiting on the chat
    # approval registry would hang indefinitely, while silently opting out would
    # weaken the permission choice captured at enqueue.
    if permission_mode in {"ask", "auto"}:
        raise AgentWorkspaceError(
            "Background agents cannot pause for interactive tool approval. "
            "Queue this task with permission mode 'off' or 'full'."
        )
    return snapshot


def validate_runtime_snapshot(snapshot: Any) -> dict[str, Any]:
    """Validate an internal durable snapshot before copying it to another task."""
    return dict(_validate_snapshot(snapshot))


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _agent_messages(context: Any) -> list[dict[str, str]]:
    from core.agent_workspace.project_context import (
        escape_project_context,
        resolve_repository_prompt_context,
    )

    project = studio_db.get_chat_project(context.project_id) or {}
    from core.agent_workspace.state import get_background_task

    durable_task = get_background_task(context.task_id) or {}
    delegation_policy = (durable_task.get("payload") or {}).get("delegationPolicy") or {}
    instructions = _bounded_text(project.get("instructions"), _MAX_PROJECT_INSTRUCTIONS)
    goal = _bounded_text(context.goal_snapshot, _MAX_GOAL)
    plan = ""
    if context.plan_snapshot is not None:
        plan = json.dumps(
            context.plan_snapshot,
            ensure_ascii = False,
            sort_keys = True,
            separators = (",", ":"),
        )[:_MAX_PLAN]
    repository = resolve_repository_prompt_context(
        context.cwd,
        context.instruction,
        expected_identity = context.expected_root_identity,
        max_instruction_bytes = _MAX_REPOSITORY_INSTRUCTIONS,
    )
    from .memory import memory_context

    persisted_memory = memory_context(context.project_id, context.instruction)
    sections = [
        "You are an Unsloth Studio background coding agent. Work only inside the "
        "assigned project workspace. Inspect the repository before editing, use the "
        "provided tools for concrete work, preserve unrelated user changes, and report "
        "verification honestly.",
    ]
    if context.delegation_role:
        role_guidance = (
            " If you change code, commit the bounded result in your assigned worktree "
            "and report the commit SHA so the parent can collect it."
            if context.delegation_role == "implementer"
            else ""
        )
        sections.append(
            '<delegation role="'
            + escape_project_context(str(context.delegation_role))
            + '" depth="'
            + str(int(context.delegation_depth))
            + '">You are a bounded child agent. Stay within the assigned role and '
            "return concrete evidence to the parent task." + role_guidance + "</delegation>"
        )
    elif bool(delegation_policy.get("enabled")):
        sections.append(
            "<child_agents>You may delegate independent bounded work with "
            "delegate_agent. Each child gets a separate Git worktree and the same "
            "immutable runtime. Use child_agent_status to collect every result, do "
            "not finish while children are active, and cherry-pick only reviewed "
            "child commits into your own worktree.</child_agents>"
        )
    if instructions:
        sections.append(
            "<project_instructions>"
            + escape_project_context(instructions)
            + "</project_instructions>"
        )
    if goal:
        status = escape_project_context(str(context.goal_status_snapshot or "active"))
        sections.append(
            f'<project_goal status="{status}">' + escape_project_context(goal) + "</project_goal>"
        )
    if plan:
        sections.append("<project_plan>" + escape_project_context(plan) + "</project_plan>")
    if repository.addition:
        sections.append(repository.addition)
    if persisted_memory:
        sections.append(persisted_memory)
    return [
        {"role": "system", "content": "\n\n".join(sections)},
        {"role": "user", "content": context.instruction},
    ]


def _agent_tools(context: Any, full_access: bool) -> list[dict[str, Any]]:
    from core.inference.tools import (
        AGENT_DELEGATION_TOOLS,
        ALL_TOOLS,
        apply_full_access_tool_descriptions,
    )

    allowed = {
        "edit_file",
        "python",
        "terminal",
        "web_search",
        "memory_search",
        "memory_read",
        "memory_write",
        "memory_update",
        "project_skill_read",
    }
    tools = [
        json.loads(json.dumps(tool))
        for tool in ALL_TOOLS
        if (tool.get("function") or {}).get("name") in allowed
    ]
    from core.agent_workspace.state import get_background_task

    task = get_background_task(context.task_id) or {}
    policy = (task.get("payload") or {}).get("delegationPolicy") or {}
    if bool(policy.get("enabled")) and int(context.delegation_depth) < int(
        policy.get("maxDepth") or 0
    ):
        tools.extend(json.loads(json.dumps(AGENT_DELEGATION_TOOLS)))
    return apply_full_access_tool_descriptions(tools) if full_access else tools


class _OutputCollector:
    def __init__(self) -> None:
        self._content = ""
        self._reasoning = ""
        self.tool_events = 0

    def local_event(self, event: Any) -> None:
        if isinstance(event, str):
            self._content = event
            return
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        text = event.get("text")
        if event_type == "content" and isinstance(text, str):
            self._content = text
        elif event_type == "reasoning" and isinstance(text, str):
            self._reasoning = text
        elif event_type in {"tool_start", "tool_end"}:
            self.tool_events += 1
        elif event_type == "error":
            raise AgentWorkspaceError("The selected local inference runtime failed.")

    def sse_line(self, line: Any) -> None:
        if not isinstance(line, str):
            return
        stripped = line.strip()
        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()
        if not stripped or stripped == "[DONE]" or stripped.startswith(":"):
            return
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("error"):
            raise AgentWorkspaceError("The selected provider inference runtime failed.")
        if payload.get("type") in {"tool_start", "tool_end"}:
            self.tool_events += 1
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices[:1]:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                self._content += content
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                self._reasoning += reasoning

    def result(self) -> dict[str, Any]:
        output = self._content or self._reasoning
        encoded = output.encode("utf-8", errors = "replace")
        truncated = len(encoded) > _MAX_RESULT_BYTES
        return {
            "output": encoded[:_MAX_RESULT_BYTES].decode("utf-8", errors = "replace"),
            "outputBytes": len(encoded),
            "outputTruncated": truncated,
            "toolEvents": self.tool_events,
        }


async def _wait_for_llama_lease(backend: Any, cancel_event: threading.Event):
    from core.inference.llama_admission import (
        LlamaAdmissionCancelled,
        get_llama_admission_queue,
        llama_admission_config_from_env,
    )

    key = str(getattr(backend, "base_url", None) or "llama-server")
    capacity = int(getattr(backend, "effective_parallel_slots", None) or 1)
    reservation = get_llama_admission_queue(key).reserve(
        capacity = capacity,
        config = llama_admission_config_from_env(),
    )
    try:
        while not cancel_event.is_set():
            lease = reservation.lease_nowait()
            if lease is not None:
                return reservation, lease
            try:
                await reservation.wait(0.1)
            except asyncio.TimeoutError:
                continue
    except BaseException:
        reservation.cancel()
        raise
    reservation.cancel()
    raise LlamaAdmissionCancelled("Background agent cancelled while waiting for a model slot.")


def _collect_local(generator: Any, cancel_event: threading.Event) -> dict[str, Any]:
    collector = _OutputCollector()
    try:
        for event in generator:
            if cancel_event.is_set():
                break
            collector.local_event(event)
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()
    return collector.result()


async def _run_local(
    snapshot: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    session_id: str,
    cancel_event: threading.Event,
    *,
    max_tool_calls: int,
    tool_timeout: int,
) -> dict[str, Any]:
    from core.inference.model_ids import model_id_matches
    from core.inference.orchestrator import peek_inference_backend
    from core.inference.runtime_registry import peek_llama_cpp_backend
    from state.active_generations import ActiveGeneration

    model = snapshot["model"]
    permission_mode = snapshot["permissionMode"]
    max_tokens = int(snapshot.get("maxOutputTokens") or _DEFAULT_GENERATION_TOKENS)
    llama = peek_llama_cpp_backend()
    llama_candidates = (
        getattr(llama, "model_identifier", None),
        getattr(llama, "_openai_advertised_id", None),
        getattr(llama, "hf_repo", None),
    )
    if (
        llama is not None
        and getattr(llama, "is_loaded", False)
        and any(model_id_matches(model, candidate) for candidate in llama_candidates)
    ):
        generator = llama.generate_chat_completion_with_tools(
            messages = messages,
            tools = tools,
            max_tokens = max_tokens,
            cancel_event = cancel_event,
            reasoning_effort = snapshot.get("reasoningEffort"),
            max_tool_iterations = max_tool_calls,
            tool_call_timeout = tool_timeout,
            session_id = session_id,
            thread_id = session_id,
            confirm_tool_calls = False,
            bypass_permissions = permission_mode == "full",
            permission_mode = permission_mode,
            context_overflow = "truncate_oldest",
        )
        reservation, lease = await _wait_for_llama_lease(llama, cancel_event)
        try:
            with ActiveGeneration(
                cancel_event,
                thread_id = session_id,
                model = model,
                kind = "background-agent",
            ):
                result = await asyncio.to_thread(_collect_local, generator, cancel_event)
        finally:
            lease.release()
            reservation.cancel()
        return {**result, "engine": "llama_cpp"}

    backend = peek_inference_backend()
    active_model = getattr(backend, "active_model_name", None) if backend is not None else None
    if backend is None or not model_id_matches(model, active_model):
        raise AgentWorkspaceError(
            "The selected local model is not loaded. Load the same model and retry the task."
        )
    generator = backend.generate_chat_completion_with_tools(
        messages = messages,
        tools = tools,
        max_tokens = max_tokens,
        cancel_event = cancel_event,
        reasoning_effort = snapshot.get("reasoningEffort"),
        max_tool_iterations = max_tool_calls,
        tool_call_timeout = tool_timeout,
        session_id = session_id,
        thread_id = session_id,
        confirm_tool_calls = False,
        bypass_permissions = permission_mode == "full",
        permission_mode = permission_mode,
    )
    with ActiveGeneration(
        cancel_event,
        thread_id = session_id,
        model = model,
        kind = "background-agent",
    ):
        result = await asyncio.to_thread(_collect_local, generator, cancel_event)
    return {**result, "engine": "local"}


def _current_provider(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    provider_id = str(snapshot.get("providerId") or "")
    config = providers_db.get_provider(provider_id)
    if config is None or not config.get("is_enabled"):
        raise AgentWorkspaceError("The selected provider connection is unavailable.")
    if str(config.get("provider_type") or "") != snapshot.get(
        "providerType"
    ) or _provider_routing_digest(config) != snapshot.get("routingDigest"):
        raise AgentWorkspaceError(
            "The selected provider changed after this task was queued. Queue a new task."
        )
    return config


async def _collect_provider_stream(generator: Any, cancel_event: threading.Event) -> dict[str, Any]:
    collector = _OutputCollector()
    try:
        async for line in generator:
            if cancel_event.is_set():
                break
            collector.sse_line(line)
    finally:
        close = getattr(generator, "aclose", None)
        if callable(close):
            try:
                await close()
            except (RuntimeError, StopAsyncIteration):
                pass
    return collector.result()


async def _run_codex(
    snapshot: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    session_id: str,
    cancel_event: threading.Event,
    *,
    max_tool_calls: int,
    tool_timeout: int,
) -> dict[str, Any]:
    from core.inference.openai_codex_auth import CodexAuthError, resolve_access
    from core.inference.openai_codex_client import OpenAICodexClient
    from core.inference.openai_codex_tool_loop import (
        CodexRunContext,
        CodexToolPolicy,
        stream_codex_with_studio_tools,
    )
    from state.active_generations import ActiveGeneration

    provider_id = snapshot["providerId"]
    access_token, account_id = await resolve_access(provider_id)
    account_digest = hashlib.sha256(account_id.encode("utf-8")).hexdigest()
    if account_digest != snapshot.get("credentialBindingDigest"):
        raise AgentWorkspaceError(
            "The selected ChatGPT connection changed accounts after this task was queued."
        )
    current_token = access_token

    async def _refresh() -> tuple[str, str]:
        nonlocal current_token
        current_token, refreshed_account = await resolve_access(
            provider_id,
            force_refresh = True,
            expected_access_token = current_token,
        )
        if hashlib.sha256(refreshed_account.encode("utf-8")).hexdigest() != snapshot.get(
            "credentialBindingDigest"
        ):
            raise CodexAuthError("The ChatGPT connection changed accounts.")
        return current_token, refreshed_account

    client = OpenAICodexClient(access_token, account_id, refresh_access = _refresh)
    run = CodexRunContext(
        provider_id = provider_id,
        thread_id = session_id,
        session_id = session_id,
        messages = messages,
        model = snapshot["model"],
        reasoning_effort = snapshot.get("reasoningEffort"),
    )
    policy = CodexToolPolicy(
        tools = tools,
        max_calls = max_tool_calls,
        timeout = tool_timeout,
        permission_mode = snapshot["permissionMode"],
        confirm_calls = False,
        bypass_permissions = snapshot["permissionMode"] == "full",
        rag_scope = None,
    )
    try:
        with ActiveGeneration(
            cancel_event,
            thread_id = session_id,
            model = snapshot["model"],
            kind = "background-agent",
        ):
            result = await _collect_provider_stream(
                stream_codex_with_studio_tools(
                    client,
                    run = run,
                    policy = policy,
                    cancel_event = cancel_event,
                ),
                cancel_event,
            )
    finally:
        await client.close()
    return {**result, "engine": "openai_codex"}


async def _run_external(
    snapshot: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    session_id: str,
    cancel_event: threading.Event,
    *,
    max_tool_calls: int,
    tool_timeout: int,
) -> dict[str, Any]:
    from core.inference.external_provider import ExternalProviderClient
    from core.inference.external_tool_transport import OAICompatTransport
    from core.inference.providers import validate_provider_base_url
    from core.inference.studio_tool_loop import (
        ToolLoopPolicy,
        ToolLoopRun,
        stream_with_studio_tools,
    )
    from state.active_generations import ActiveGeneration

    config = _current_provider(snapshot)
    base_url = validate_provider_base_url(str(config.get("base_url") or ""))
    api_key_value, credential_binding = credential_secrets.get_provider_api_key_with_binding(
        snapshot["providerId"]
    )
    if credential_binding != snapshot.get("credentialBindingDigest"):
        raise AgentWorkspaceError(
            "The selected provider credential changed after this task was queued. Queue a new task."
        )
    api_key = api_key_value or ""
    if not api_key and snapshot["providerType"] not in _KEY_OPTIONAL_PROVIDERS:
        raise AgentWorkspaceError(
            "The selected provider credential is unavailable. Reconnect it and retry."
        )
    # Pair the key only with the routing row it was read for. Provider updates
    # commit metadata and encrypted credentials atomically, and this second read
    # catches a change between the first row read and secret resolution.
    _current_provider(snapshot)
    client = ExternalProviderClient(
        provider_type = snapshot["providerType"],
        base_url = base_url,
        api_key = api_key,
    )
    transport = OAICompatTransport(
        client,
        model = snapshot["model"],
        temperature = 0.2,
        top_p = 0.95,
        max_tokens = int(snapshot.get("maxOutputTokens") or _DEFAULT_GENERATION_TOKENS),
        reasoning_effort = snapshot.get("reasoningEffort"),
        stream = True,
    )
    run = ToolLoopRun(
        messages = messages,
        session_id = session_id,
        thread_id = session_id,
        model = snapshot["model"],
    )
    policy = ToolLoopPolicy(
        tools = tools,
        max_calls = max_tool_calls,
        timeout = tool_timeout,
        permission_mode = snapshot["permissionMode"],
        confirm_calls = False,
        bypass_permissions = snapshot["permissionMode"] == "full",
        rag_scope = None,
        auto_heal = True,
    )
    try:
        with ActiveGeneration(
            cancel_event,
            thread_id = session_id,
            model = snapshot["model"],
            kind = "background-agent",
        ):
            result = await _collect_provider_stream(
                stream_with_studio_tools(
                    transport,
                    run = run,
                    policy = policy,
                    cancel_event = cancel_event,
                ),
                cancel_event,
            )
    finally:
        await client.close()
    return {**result, "engine": snapshot["providerType"]}


async def _execute(context: Any, cancel_event: threading.Event) -> dict[str, Any]:
    from core.inference.tools import (
        background_task_session_id,
        resolve_sandbox_workdir,
    )

    snapshot = _validate_snapshot(context.runtime_snapshot)
    session_id = background_task_session_id(context.task_id)
    bound_cwd = Path(resolve_sandbox_workdir(session_id)).resolve(strict = True)
    if os.path.normcase(str(bound_cwd)) != os.path.normcase(str(context.cwd.resolve(strict = True))):
        raise AgentWorkspaceError("The background agent workspace binding changed.")
    messages = _agent_messages(context)
    tools = _agent_tools(context, snapshot["permissionMode"] == "full")
    budget = context.delegation_budget or {}
    max_tool_calls = max(1, min(200, int(budget.get("maxToolCalls") or 25)))
    tool_timeout = max(1, min(300, int(budget.get("wallSeconds") or 300)))
    if cancel_event.is_set():
        return {"output": "", "outputBytes": 0, "outputTruncated": False}
    if snapshot["kind"] == "local":
        result = await _run_local(
            snapshot,
            messages,
            tools,
            session_id,
            cancel_event,
            max_tool_calls = max_tool_calls,
            tool_timeout = tool_timeout,
        )
    else:
        _current_provider(snapshot)
        if snapshot["providerType"] == "openai_codex":
            result = await _run_codex(
                snapshot,
                messages,
                tools,
                session_id,
                cancel_event,
                max_tool_calls = max_tool_calls,
                tool_timeout = tool_timeout,
            )
        else:
            result = await _run_external(
                snapshot,
                messages,
                tools,
                session_id,
                cancel_event,
                max_tool_calls = max_tool_calls,
                tool_timeout = tool_timeout,
            )
    return {
        **result,
        "model": snapshot["model"],
        "providerId": snapshot.get("providerId"),
        "providerType": snapshot.get("providerType"),
        "permissionMode": snapshot["permissionMode"],
        "sessionId": session_id,
        "worktreeId": context.worktree_id,
    }


def execute_background_agent(context: Any, cancel_event: threading.Event) -> dict[str, Any]:
    """Run one durable task through the selected internal inference transport."""
    budget = context.delegation_budget or {}
    wall_seconds = int(budget.get("wallSeconds") or 0)
    expired = threading.Event()
    timer = None
    if wall_seconds > 0:

        def exhaust_budget() -> None:
            expired.set()
            cancel_event.set()

        timer = threading.Timer(wall_seconds, exhaust_budget)
        timer.daemon = True
        timer.start()
    try:
        result = asyncio.run(_execute(context, cancel_event))
        if expired.is_set():
            raise AgentWorkspaceError("The child-agent wall-time budget was exhausted.")
        return result
    except AgentWorkspaceError:
        raise
    except Exception as exc:
        raise AgentWorkspaceError(
            "The selected inference runtime failed before the background task completed."
        ) from exc
    finally:
        if timer is not None:
            timer.cancel()


__all__ = [
    "capture_runtime_snapshot",
    "execute_background_agent",
]
