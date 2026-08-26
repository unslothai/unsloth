# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit, one-use approval records for connected GitHub pull requests."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from storage import mcp_servers_db

from .common import (
    AgentWorkspaceError,
    project_workspace,
    workspace_fingerprint_complete,
)
from .git_service import git_status, workspace_fingerprint
from .review import build_pull_request_draft, redact_review_text


_HANDOFF_TTL_SECONDS = 10 * 60
_GITHUB_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?$")
_GIT_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_PENDING_LOCK = threading.Lock()
_PENDING: dict[str, "_PendingHandoff"] = {}


@dataclass(frozen = True)
class _PendingHandoff:
    id: str
    token_digest: bytes
    project_id: str
    server_id: str
    server_snapshot: str
    request: dict[str, Any]
    request_digest: str
    reviewed_head: str
    workspace_fingerprint: str
    expires_at: int


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii = False,
        separators = (",", ":"),
        sort_keys = True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _server_snapshot(server: dict) -> str:
    return _canonical_digest(
        {
            "id": server.get("id"),
            "url": server.get("url"),
            "headersJson": server.get("headers_json"),
            "enabled": bool(server.get("is_enabled")),
            "oauth": bool(server.get("use_oauth")),
            "updatedAt": server.get("updated_at"),
        }
    )


def _validate_github_name(value: str, label: str) -> str:
    normalized = value.strip()
    if not _GITHUB_NAME.fullmatch(normalized) or normalized.endswith(".git"):
        raise AgentWorkspaceError(f"GitHub {label} is invalid.")
    return normalized


def _validate_git_ref(value: str, label: str) -> str:
    normalized = value.strip()
    if (
        not _GIT_REF.fullmatch(normalized)
        or ".." in normalized
        or "//" in normalized
        or "@{" in normalized
        or normalized.endswith(("/", ".", ".lock"))
        or any(part.startswith(".") for part in normalized.split("/"))
    ):
        raise AgentWorkspaceError(f"Pull request {label} branch is invalid.")
    return normalized


def _enabled_server(server_id: str) -> dict:
    server = mcp_servers_db.get_server(server_id)
    if server is None or not server.get("is_enabled"):
        raise AgentWorkspaceError("The selected GitHub connector is unavailable.")
    return server


def _review_binding(project_id: str) -> tuple[str, str]:
    workspace = project_workspace(project_id)
    status = git_status(project_id)
    head = str(status.get("head") or "")
    fingerprint = workspace_fingerprint(workspace.root)
    if not head or not workspace_fingerprint_complete(fingerprint):
        raise AgentWorkspaceError(
            "The repository review evidence is incomplete. Reduce the change set and retry."
        )
    return head, fingerprint


def _pr_tool(tools: list[dict]) -> Optional[dict]:
    matches = [tool for tool in tools if tool.get("name") == "create_pull_request"]
    if len(matches) != 1:
        return None
    schema = matches[0].get("inputSchema") or matches[0].get("input_schema") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or ()) if isinstance(schema, dict) else set()
    expected = {"owner", "repo", "title", "body", "head", "base"}
    if not isinstance(properties, dict) or not expected.issubset(properties):
        return None
    if not required.issubset(expected | {"draft", "maintainer_can_modify"}):
        return None
    return matches[0]


def require_pull_request_tool(tools: list[dict]) -> None:
    """Reject connectors whose discovered mutation contract is ambiguous."""
    if _pr_tool(tools) is None:
        raise AgentWorkspaceError(
            "The selected connector does not expose a compatible create_pull_request tool."
        )


def prepare_pull_request_handoff(
    project_id: str,
    *,
    server_id: str,
    owner: str,
    repository: str,
    base: str,
    head: str,
    title: str = "",
    body_note: str = "",
    draft: bool = True,
    tools: list[dict],
    now: Optional[int] = None,
) -> dict:
    """Create a bounded preview and one-use token without changing GitHub."""
    server = _enabled_server(server_id)
    require_pull_request_tool(tools)
    reviewed_before = _review_binding(project_id)
    draft_payload = build_pull_request_draft(
        project_id,
        title = title,
        body_note = body_note,
    )
    request = {
        "owner": _validate_github_name(owner, "owner"),
        "repo": _validate_github_name(repository, "repository"),
        "base": _validate_git_ref(base, "base"),
        "head": _validate_git_ref(head, "head"),
        "title": redact_review_text(str(draft_payload["title"]), "")[:120],
        "body": redact_review_text(str(draft_payload["body"]), "")[:64_000],
        "draft": bool(draft),
        "maintainer_can_modify": True,
    }
    request_digest = _canonical_digest(request)
    reviewed_after = _review_binding(project_id)
    if reviewed_after != reviewed_before:
        raise AgentWorkspaceError(
            "The repository changed while the pull request preview was built. "
            "Create a new preview."
        )
    token = secrets.token_urlsafe(32)
    handoff_id = str(uuid.uuid4())
    issued_at = int(time.time()) if now is None else int(now)
    expires_at = issued_at + _HANDOFF_TTL_SECONDS
    record = _PendingHandoff(
        id = handoff_id,
        token_digest = _token_digest(token),
        project_id = project_id,
        server_id = server_id,
        server_snapshot = _server_snapshot(server),
        request = request,
        request_digest = request_digest,
        reviewed_head = reviewed_after[0],
        workspace_fingerprint = reviewed_after[1],
        expires_at = expires_at,
    )
    with _PENDING_LOCK:
        for pending_id, pending in tuple(_PENDING.items()):
            if pending.expires_at < issued_at:
                _PENDING.pop(pending_id, None)
        _PENDING[handoff_id] = record
    return {
        "id": handoff_id,
        "confirmationToken": token,
        "requestDigest": request_digest,
        "reviewBinding": {
            "head": record.reviewed_head,
            "workspaceFingerprint": record.workspace_fingerprint,
        },
        "expiresAt": expires_at * 1000,
        "connector": {
            "id": server_id,
            "displayName": str(server.get("display_name") or "GitHub"),
        },
        "request": request,
        "submitted": False,
    }


def consume_pull_request_handoff(
    project_id: str,
    handoff_id: str,
    *,
    server_id: str,
    confirmation_token: str,
    expected_request_digest: str,
    tools: list[dict],
    now: Optional[int] = None,
    include_review_binding: bool = False,
) -> tuple[dict, dict] | tuple[dict, dict, dict[str, str]]:
    """Atomically consume an approval after every preview invariant is rechecked."""
    current_time = int(time.time()) if now is None else int(now)
    with _PENDING_LOCK:
        pending = _PENDING.pop(handoff_id, None)
    if pending is None:
        raise AgentWorkspaceError("Pull request confirmation is missing or already used.")
    if pending.expires_at < current_time:
        raise AgentWorkspaceError("Pull request confirmation expired.")
    if pending.project_id != project_id:
        raise AgentWorkspaceError("Pull request confirmation belongs to another project.")
    if pending.server_id != server_id:
        raise AgentWorkspaceError("Pull request confirmation belongs to another connector.")
    if not hmac.compare_digest(
        pending.token_digest,
        _token_digest(confirmation_token),
    ):
        raise AgentWorkspaceError("Pull request confirmation is invalid.")
    if not hmac.compare_digest(pending.request_digest, expected_request_digest):
        raise AgentWorkspaceError("Pull request preview changed before confirmation.")
    server = _enabled_server(pending.server_id)
    if not hmac.compare_digest(pending.server_snapshot, _server_snapshot(server)):
        raise AgentWorkspaceError(
            "The GitHub connector changed after preview. Create a new preview."
        )
    require_pull_request_tool(tools)
    current_head, current_fingerprint = _review_binding(project_id)
    if not hmac.compare_digest(pending.reviewed_head, current_head) or not hmac.compare_digest(
        pending.workspace_fingerprint,
        current_fingerprint,
    ):
        raise AgentWorkspaceError("The repository changed after preview. Create a new preview.")
    if include_review_binding:
        return (
            server,
            dict(pending.request),
            {
                "head": pending.reviewed_head,
                "workspaceFingerprint": pending.workspace_fingerprint,
            },
        )
    return server, dict(pending.request)


def pull_request_review_binding_current(project_id: str, binding: dict[str, str]) -> bool:
    """Fail closed when a prepared handoff no longer names the current tree."""
    try:
        current_head, current_fingerprint = _review_binding(project_id)
    except Exception:
        return False
    return hmac.compare_digest(
        str(binding.get("head") or ""), current_head
    ) and hmac.compare_digest(str(binding.get("workspaceFingerprint") or ""), current_fingerprint)


def reset_pull_request_handoffs_for_tests() -> None:
    with _PENDING_LOCK:
        _PENDING.clear()


__all__ = [
    "consume_pull_request_handoff",
    "prepare_pull_request_handoff",
    "pull_request_review_binding_current",
    "require_pull_request_tool",
]
