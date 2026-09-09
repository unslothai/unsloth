# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted, project-local Codex lifecycle hooks.

The module deliberately owns no persistence. Callers discover and validate an
exact ``.codex/hooks.json`` file and persist its ``contentHash`` after review.
Runtime execution and lifecycle authority are deliberately outside this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .hook_context import AgentWorkspaceError


HOOK_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
)
MAX_HOOK_CONFIG_BYTES = 256 * 1024
MAX_HOOK_GROUPS = 128
MAX_HOOK_HANDLERS = 256
MAX_HOOK_COMMAND_BYTES = 8 * 1024
MAX_HOOK_MATCHER_CHARACTERS = 256
MAX_HOOK_TIMEOUT_SECONDS = 600
MAX_ADDITIONAL_CONTEXT_TOKENS = 100_000
_HOOK_PATH = ".codex/hooks.json"
_TOP_LEVEL_FIELDS = frozenset({"description", "hooks"})
_GROUP_FIELDS = frozenset({"matcher", "hooks"})
_COMMAND_FIELDS = frozenset(
    {
        "type",
        "command",
        "commandWindows",
        "timeout",
        "statusMessage",
        "additionalContextLimit",
        "async",
    }
)
_MATCH_FIELD = {
    "SessionStart": "source",
    "SessionEnd": "reason",
    "PreToolUse": "tool_name",
    "PermissionRequest": "tool_name",
    "PostToolUse": "tool_name",
    "PreCompact": "trigger",
    "PostCompact": "trigger",
    "SubagentStart": "agent_type",
    "SubagentStop": "agent_type",
    "UserPromptSubmit": None,
    "Stop": None,
}
_MATCH_LITERAL = "literal"
_MATCH_ONE = "one"
_MATCH_MANY = "many"
_UNSUPPORTED_MATCHER_CHARACTERS = frozenset("()[]+?{}")


def project_hooks_content_hash(raw: bytes | str) -> str:
    """Hash the exact source bytes reviewed by the user."""
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AgentWorkspaceError(f"Project hooks contain duplicate field {key!r}.")
        result[key] = value
    return result


def _parse_json(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_HOOK_CONFIG_BYTES:
        raise AgentWorkspaceError("Project hooks exceed the configuration size limit.")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook = _reject_duplicate_fields,
        )
    except AgentWorkspaceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AgentWorkspaceError("Project hooks must be a valid UTF-8 JSON object.") from exc
    if not isinstance(document, dict):
        raise AgentWorkspaceError("Project hooks must be a JSON object.")
    return document


def _unknown_fields(value: dict, allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        rendered = ", ".join(repr(field) for field in unknown)
        raise AgentWorkspaceError(f"{label} contains unsupported fields: {rendered}.")


def _optional_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = True,
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentWorkspaceError(f"{label} must be a string.")
    if (not allow_empty and not value) or len(value.encode("utf-8")) > maximum:
        raise AgentWorkspaceError(f"{label} is invalid or exceeds its size limit.")
    if "\x00" in value:
        raise AgentWorkspaceError(f"{label} cannot contain NUL characters.")
    return value


def _integer(value: Any, *, label: str, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentWorkspaceError(f"{label} must be an integer.")
    if value < minimum or value > maximum:
        raise AgentWorkspaceError(f"{label} is outside the supported range.")
    return value


def _invalid_matcher(event: str) -> AgentWorkspaceError:
    return AgentWorkspaceError(f"{event} hook matcher uses an unsupported or unsafe pattern.")


def _split_matcher_alternatives(matcher: str, *, event: str) -> tuple[str, ...]:
    alternatives = []
    start = 0
    escaped = False
    for index, character in enumerate(matcher):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "|":
            alternatives.append(matcher[start:index])
            start = index + 1
    if escaped:
        raise _invalid_matcher(event)
    alternatives.append(matcher[start:])
    if any(not alternative for alternative in alternatives):
        raise _invalid_matcher(event)
    return tuple(alternatives)


def _parse_matcher_alternative(
    alternative: str, *, event: str
) -> tuple[bool, bool, tuple[tuple[str, Optional[str]], ...]]:
    anchored_start = alternative.startswith("^")
    index = 1 if anchored_start else 0
    anchored_end = False
    tokens: list[tuple[str, Optional[str]]] = []
    while index < len(alternative):
        character = alternative[index]
        if character == "\\":
            index += 1
            if index >= len(alternative):
                raise _invalid_matcher(event)
            tokens.append((_MATCH_LITERAL, alternative[index]))
            index += 1
            continue
        if character == "$":
            if index != len(alternative) - 1:
                raise _invalid_matcher(event)
            anchored_end = True
            index += 1
            continue
        if character == "^" or character in _UNSUPPORTED_MATCHER_CHARACTERS:
            raise _invalid_matcher(event)
        if character == "*":
            raise _invalid_matcher(event)
        if character == ".":
            if index + 1 < len(alternative) and alternative[index + 1] == "*":
                if not tokens or tokens[-1][0] != _MATCH_MANY:
                    tokens.append((_MATCH_MANY, None))
                index += 2
            else:
                tokens.append((_MATCH_ONE, None))
                index += 1
            continue
        tokens.append((_MATCH_LITERAL, character))
        index += 1
    return anchored_start, anchored_end, tuple(tokens)


def _parse_matcher(
    matcher: str, *, event: str
) -> tuple[tuple[bool, bool, tuple[tuple[str, Optional[str]], ...]], ...]:
    return tuple(
        _parse_matcher_alternative(alternative, event = event)
        for alternative in _split_matcher_alternatives(matcher, event = event)
    )


def _match_tokens(
    tokens: tuple[tuple[str, Optional[str]], ...],
    value: str,
    *,
    anchored_start: bool,
    anchored_end: bool,
) -> bool:
    effective_tokens = (
        (() if anchored_start else ((_MATCH_MANY, None),))
        + tokens
        + (() if anchored_end else ((_MATCH_MANY, None),))
    )
    states = [False] * (len(value) + 1)
    states[0] = True
    for kind, literal in effective_tokens:
        next_states = [False] * (len(value) + 1)
        if kind == _MATCH_MANY:
            next_states[0] = states[0]
            for position in range(1, len(value) + 1):
                next_states[position] = states[position] or next_states[position - 1]
        else:
            for position, character in enumerate(value):
                if states[position] and (
                    kind == _MATCH_ONE or (kind == _MATCH_LITERAL and character == literal)
                ):
                    next_states[position + 1] = True
        states = next_states
    return states[-1]


def _safe_matcher_matches(matcher: str, value: str, *, event: str) -> bool:
    return any(
        _match_tokens(
            tokens,
            value,
            anchored_start = anchored_start,
            anchored_end = anchored_end,
        )
        for anchored_start, anchored_end, tokens in _parse_matcher(matcher, event = event)
    )


def _validate_matcher(value: Any, *, event: str) -> Optional[str]:
    matcher = _optional_text(
        value,
        label = f"{event} hook matcher",
        maximum = MAX_HOOK_MATCHER_CHARACTERS,
    )
    if matcher in {None, "", "*"}:
        return None
    _parse_matcher(matcher, event = event)
    return matcher


def _validate_handler(value: Any, *, event: str, handler_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentWorkspaceError(f"{event} hook handlers must be JSON objects.")
    if value.get("type") != "command":
        raise AgentWorkspaceError(
            f"{event} hook handler type must be command; MCP hooks are not available yet."
        )
    _unknown_fields(value, _COMMAND_FIELDS, f"{event} hook handler")
    command = _optional_text(
        value.get("command"),
        label = f"{event} hook command",
        maximum = MAX_HOOK_COMMAND_BYTES,
        allow_empty = False,
    )
    command_windows = _optional_text(
        value.get("commandWindows"),
        label = f"{event} Windows hook command",
        maximum = MAX_HOOK_COMMAND_BYTES,
        allow_empty = False,
    )
    if command is None:
        raise AgentWorkspaceError(f"{event} hook command is required.")
    timeout_default = 1 if event == "SessionEnd" else MAX_HOOK_TIMEOUT_SECONDS
    timeout_maximum = 3 if event == "SessionEnd" else MAX_HOOK_TIMEOUT_SECONDS
    timeout = _integer(
        value.get("timeout"),
        label = f"{event} hook timeout",
        default = timeout_default,
        minimum = 1,
        maximum = timeout_maximum,
    )
    status_message = _optional_text(
        value.get("statusMessage"),
        label = f"{event} hook status message",
        maximum = 512,
    )
    additional_context_limit = _integer(
        value.get("additionalContextLimit"),
        label = f"{event} additional context limit",
        default = 2_500,
        minimum = 0,
        maximum = MAX_ADDITIONAL_CONTEXT_TOKENS,
    )
    asynchronous = value.get("async", False)
    if not isinstance(asynchronous, bool):
        raise AgentWorkspaceError(f"{event} hook async must be a boolean.")
    return {
        "id": handler_id,
        "type": "command",
        "command": command,
        "commandWindows": command_windows,
        "timeout": timeout,
        "statusMessage": status_message,
        "additionalContextLimit": additional_context_limit,
        # SessionEnd is always synchronous in Codex even when async is set.
        "async": asynchronous and event != "SessionEnd",
    }


def validate_project_hooks(raw: bytes | str, *, source_path: str = _HOOK_PATH) -> dict[str, Any]:
    """Validate exact ``hooks.json`` bytes and return a bounded runtime form."""
    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    document = _parse_json(encoded)
    _unknown_fields(document, _TOP_LEVEL_FIELDS, "Project hooks")
    description = _optional_text(
        document.get("description"),
        label = "Project hook description",
        maximum = 4 * 1024,
    )
    hooks_document = document.get("hooks", {})
    if not isinstance(hooks_document, dict):
        raise AgentWorkspaceError("Project hooks.hooks must be a JSON object.")
    unknown_events = sorted(set(hooks_document) - set(HOOK_EVENTS))
    if unknown_events:
        raise AgentWorkspaceError(
            "Project hooks contain unsupported events: "
            + ", ".join(repr(event) for event in unknown_events)
            + "."
        )

    normalized: dict[str, list[dict[str, Any]]] = {}
    group_count = 0
    handler_count = 0
    for event in HOOK_EVENTS:
        groups = hooks_document.get(event, [])
        if not isinstance(groups, list):
            raise AgentWorkspaceError(f"Project hooks.{event} must be an array.")
        rendered_groups: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            group_count += 1
            if group_count > MAX_HOOK_GROUPS:
                raise AgentWorkspaceError("Project hooks exceed the matcher-group limit.")
            if not isinstance(group, dict):
                raise AgentWorkspaceError(f"{event} hook matcher groups must be JSON objects.")
            _unknown_fields(group, _GROUP_FIELDS, f"{event} hook matcher group")
            matcher = _validate_matcher(group.get("matcher"), event = event)
            handlers = group.get("hooks")
            if not isinstance(handlers, list) or not handlers:
                raise AgentWorkspaceError(f"{event} hook matcher groups need at least one hook.")
            rendered_handlers = []
            for handler_index, handler in enumerate(handlers):
                handler_count += 1
                if handler_count > MAX_HOOK_HANDLERS:
                    raise AgentWorkspaceError("Project hooks exceed the handler limit.")
                rendered_handlers.append(
                    _validate_handler(
                        handler,
                        event = event,
                        handler_id = f"{event}:{group_index}:{handler_index}",
                    )
                )
            rendered_groups.append({"matcher": matcher, "hooks": rendered_handlers})
        normalized[event] = rendered_groups

    return {
        "sourcePath": source_path,
        "exists": True,
        "contentHash": project_hooks_content_hash(encoded),
        "rootIdentity": None,
        "description": description,
        "hooks": normalized,
        "groupCount": group_count,
        "handlerCount": handler_count,
    }


def _normalize_identity(expected_identity: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    if expected_identity is None:
        return None
    try:
        return int(expected_identity[0]), int(expected_identity[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Project root identity is invalid.") from exc


def _empty_project_hooks(root_identity: Optional[tuple[int, int]] = None) -> dict[str, Any]:
    return {
        "sourcePath": _HOOK_PATH,
        "exists": False,
        "contentHash": None,
        "rootIdentity": root_identity,
        "description": None,
        "hooks": {event: [] for event in HOOK_EVENTS},
        "groupCount": 0,
        "handlerCount": 0,
    }


def _check_discovery_budget(cancel_event, deadline: Optional[float]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Project hook discovery was cancelled.")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Project hook discovery timed out.")


def discover_project_hooks(
    root: Path | str,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    cancel_event = None,
    deadline: Optional[float] = None,
) -> dict[str, Any]:
    """Read one identity-bound, non-symlink project ``hooks.json`` file."""
    if (
        os.name == "nt"
        or os.open not in os.supports_dir_fd
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_NONBLOCK")
    ):
        raise AgentWorkspaceError(
            "Project hook discovery is unavailable without secure directory traversal."
        )
    workspace = root if hasattr(root, "device_id") and hasattr(root, "file_id") else None
    requested_root = Path(workspace.root if workspace is not None else root)
    if expected_identity is None and workspace is not None:
        expected_identity = (workspace.device_id, workspace.file_id)
    expected = _normalize_identity(expected_identity)
    root_fd = codex_fd = hook_fd = None
    try:
        _check_discovery_budget(cancel_event, deadline)
        if requested_root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = requested_root.resolve(strict = True)
        _check_discovery_budget(cancel_event, deadline)
        root_fd = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_path_metadata = resolved.stat(follow_symlinks = False)
        root_opened_metadata = os.fstat(root_fd)
        actual_identity = (
            int(root_opened_metadata.st_dev),
            int(root_opened_metadata.st_ino),
        )
        if (
            not stat.S_ISDIR(root_path_metadata.st_mode)
            or not stat.S_ISDIR(root_opened_metadata.st_mode)
            or actual_identity != (int(root_path_metadata.st_dev), int(root_path_metadata.st_ino))
            or (expected is not None and actual_identity != expected)
        ):
            raise AgentWorkspaceError("Project root identity changed.")
        try:
            _check_discovery_budget(cancel_event, deadline)
            codex_fd = os.open(
                ".codex",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd = root_fd,
            )
        except FileNotFoundError:
            return _empty_project_hooks(actual_identity)
        if not stat.S_ISDIR(os.fstat(codex_fd).st_mode):
            raise AgentWorkspaceError("Project .codex is not a directory.")
        try:
            _check_discovery_budget(cancel_event, deadline)
            hook_fd = os.open(
                "hooks.json",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd = codex_fd,
            )
        except FileNotFoundError:
            return _empty_project_hooks(actual_identity)
        before = os.fstat(hook_fd)
        if not stat.S_ISREG(before.st_mode):
            raise AgentWorkspaceError("Project hooks.json must be a regular file.")
        raw = bytearray()
        while len(raw) <= MAX_HOOK_CONFIG_BYTES:
            _check_discovery_budget(cancel_event, deadline)
            chunk = os.read(hook_fd, min(8192, MAX_HOOK_CONFIG_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(hook_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise AgentWorkspaceError("Project hooks.json changed while it was being read.")
        discovered = validate_project_hooks(bytes(raw), source_path = _HOOK_PATH)
        discovered["rootIdentity"] = actual_identity
        return discovered
    except (AgentWorkspaceError, TimeoutError, InterruptedError):
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError(
            "Project hooks are unavailable or cross an unsafe filesystem boundary."
        ) from exc
    finally:
        for descriptor in (hook_fd, codex_fd, root_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def secure_project_hook_discovery_supported() -> bool:
    return bool(
        os.name != "nt"
        and os.open in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_NONBLOCK")
    )


def project_hooks_are_trusted(
    config: dict[str, Any], trusted_hashes: Optional[str | Iterable[str]]
) -> bool:
    """Return whether the exact discovered source hash has been reviewed."""
    content_hash = config.get("contentHash")
    if not config.get("exists") or not isinstance(content_hash, str):
        return False
    if trusted_hashes is None:
        return False
    if isinstance(trusted_hashes, str):
        return content_hash == trusted_hashes
    return any(content_hash == value for value in trusted_hashes)


def _matcher_values(event: str, event_input: dict[str, Any]) -> tuple[str, ...]:
    field = _MATCH_FIELD[event]
    if field is None:
        return ()
    value = event_input.get(field)
    if not isinstance(value, str) or len(value) > 512:
        return ()
    aliases = [value]
    if field == "tool_name" and value in {"apply_patch", "edit_file"}:
        aliases.extend(("Edit", "Write"))
    elif field == "tool_name" and value == "terminal":
        aliases.extend(("Bash", "Terminal"))
    elif field == "tool_name" and value == "python":
        aliases.append("Python")
    elif field == "tool_name" and value == "spawn_agent":
        aliases.append("Agent")
    return tuple(aliases)


def matching_project_hooks(
    config: dict[str, Any],
    event: str,
    event_input: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Return matching handlers in stable declaration order."""
    if event not in HOOK_EVENTS:
        raise AgentWorkspaceError("Project hook event is invalid.")
    payload = event_input or {}
    if not isinstance(payload, dict):
        raise AgentWorkspaceError("Project hook input must be a JSON object.")
    values = _matcher_values(event, payload)
    handlers: list[dict[str, Any]] = []
    for group in config.get("hooks", {}).get(event, []):
        matcher = group.get("matcher")
        matches = matcher is None or _MATCH_FIELD[event] is None
        if not matches:
            matches = any(_safe_matcher_matches(matcher, value, event = event) for value in values)
        if matches:
            handlers.extend(dict(handler) for handler in group["hooks"])
    return handlers


__all__ = [
    "HOOK_EVENTS",
    "MAX_HOOK_CONFIG_BYTES",
    "discover_project_hooks",
    "matching_project_hooks",
    "project_hooks_are_trusted",
    "project_hooks_content_hash",
    "validate_project_hooks",
]
