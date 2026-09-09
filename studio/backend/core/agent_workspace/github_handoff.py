# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Preview-bound, one-use GitHub handoff records.

Preparing a handoff never contacts GitHub. Confirmation rechecks the connector,
the exact reviewed head, and the complete local fingerprint before a caller is
allowed to invoke its separately discovered MCP mutation tool.
"""

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
from typing import Optional

from storage import mcp_servers_db

from .git_context import AgentWorkspaceError, project_workspace
from .git_service import git_status, workspace_fingerprint
from .review import build_pull_request_draft, redact_review_text


_TTL_SECONDS = 10 * 60
_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_PENDING_LOCK = threading.Lock()
_MAX_PENDING = 128


@dataclass(frozen = True)
class _Pending:
    id: str
    token_digest: bytes
    project_id: str
    server_id: str
    server_snapshot: str
    request: dict
    request_digest: str
    tool_digest: str
    head: str
    branch: str
    fingerprint: str
    expires_at: int


_PENDING: dict[str, _Pending] = {}


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii = False, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()


def _server_snapshot(server: dict) -> str:
    return _digest(
        {
            "id": server.get("id"),
            "url": server.get("url"),
            "headers": server.get("headers_json"),
            "enabled": bool(server.get("is_enabled")),
            "oauth": bool(server.get("use_oauth")),
            "updatedAt": server.get("updated_at"),
        }
    )


def _name(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not _NAME.fullmatch(normalized) or normalized.endswith(".git"):
        raise AgentWorkspaceError(f"GitHub {label} is invalid.")
    return normalized


def _ref(value: str, label: str) -> str:
    normalized = str(value).strip()
    if (
        not _REF.fullmatch(normalized)
        or ".." in normalized
        or "//" in normalized
        or "@{" in normalized
        or normalized.endswith(("/", ".", ".lock"))
        or any(part.startswith(".") for part in normalized.split("/"))
    ):
        raise AgentWorkspaceError(f"Pull request {label} branch is invalid.")
    return normalized


def _server(server_id: str) -> dict:
    value = mcp_servers_db.get_server(server_id)
    if value is None or not value.get("is_enabled"):
        raise AgentWorkspaceError("The selected GitHub connector is unavailable.")
    return value


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
    if not required.issubset(properties) or not required.issubset(
        expected | {"draft", "maintainer_can_modify"}
    ):
        return None
    return matches[0]


def _tool_digest(tool: dict) -> str:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    return _digest({"name": tool.get("name"), "schema": schema})


def require_pull_request_tool(tools: list[dict]) -> None:
    if _pr_tool(tools) is None:
        raise AgentWorkspaceError(
            "The selected connector does not expose a compatible create_pull_request tool."
        )


def _binding(project_id: str) -> tuple[str, str, str]:
    workspace = project_workspace(project_id)
    status = git_status(project_id)
    head = str(status.get("head") or "")
    fingerprint = workspace_fingerprint(workspace.root)
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head) or not fingerprint.startswith("c0dec0de"):
        raise AgentWorkspaceError(
            "The repository review evidence is incomplete. Reduce the change set and retry."
        )
    branch = str(status.get("branch") or "")
    if not branch or bool(status.get("detached")):
        raise AgentWorkspaceError(
            "The repository review evidence must identify a local branch before handoff."
        )
    return head, fingerprint, branch


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
    server = _server(server_id)
    tool = _pr_tool(tools)
    if tool is None:
        raise AgentWorkspaceError(
            "The selected connector does not expose a compatible create_pull_request tool."
        )
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        raise AgentWorkspaceError("The selected connector has an invalid pull request schema.")
    if draft and "draft" not in properties:
        raise AgentWorkspaceError("The selected connector cannot create a draft pull request.")
    before = _binding(project_id)
    preview = build_pull_request_draft(project_id, title = title, body_note = body_note)
    request = {
        "owner": _name(owner, "owner"),
        "repo": _name(repository, "repository"),
        "base": _ref(base, "base"),
        "head": _ref(head, "head"),
        "title": redact_review_text(str(preview["title"]), "")[:120],
        "body": redact_review_text(str(preview["body"]), "")[:64_000],
    }
    if request["head"] != before[2]:
        raise AgentWorkspaceError(
            "The pull request head must match the branch reviewed in the local workspace."
        )
    # Optional fields are only sent when the discovered connector advertises
    # them.  Sending unknown keys would make an otherwise valid handoff fail
    # against strict JSON-schema MCP tools.
    if "draft" in properties:
        request["draft"] = bool(draft)
    if "maintainer_can_modify" in properties:
        request["maintainer_can_modify"] = True
    after = _binding(project_id)
    if before != after:
        raise AgentWorkspaceError(
            "The repository changed while the pull request preview was built. Create a new preview."
        )
    issued = int(time.time()) if now is None else int(now)
    token = secrets.token_urlsafe(32)
    handoff_id = str(uuid.uuid4())
    record = _Pending(
        id = handoff_id,
        token_digest = hashlib.sha256(token.encode("utf-8")).digest(),
        project_id = project_id,
        server_id = server_id,
        server_snapshot = _server_snapshot(server),
        request = request,
        request_digest = _digest(request),
        tool_digest = _tool_digest(tool),
        head = after[0],
        branch = after[2],
        fingerprint = after[1],
        expires_at = issued + _TTL_SECONDS,
    )
    with _PENDING_LOCK:
        for key, value in tuple(_PENDING.items()):
            if value.expires_at < issued:
                _PENDING.pop(key, None)
        if len(_PENDING) >= _MAX_PENDING:
            raise AgentWorkspaceError(
                "Too many pull request previews are pending. Wait for them to expire."
            )
        _PENDING[handoff_id] = record
    return {
        "id": handoff_id,
        "confirmationToken": token,
        "requestDigest": record.request_digest,
        "reviewBinding": {
            "head": record.head,
            "branch": record.branch,
            "workspaceFingerprint": record.fingerprint,
        },
        "expiresAt": record.expires_at * 1000,
        "connector": {"id": server_id, "displayName": str(server.get("display_name") or "GitHub")},
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
):
    current = int(time.time()) if now is None else int(now)
    with _PENDING_LOCK:
        pending = _PENDING.pop(handoff_id, None)
    if pending is None:
        raise AgentWorkspaceError("Pull request confirmation is missing or already used.")
    if pending.expires_at < current:
        raise AgentWorkspaceError("Pull request confirmation expired.")
    if pending.project_id != project_id or pending.server_id != server_id:
        raise AgentWorkspaceError(
            "Pull request confirmation belongs to another project or connector."
        )
    if not hmac.compare_digest(
        pending.token_digest, hashlib.sha256(confirmation_token.encode("utf-8")).digest()
    ):
        raise AgentWorkspaceError("Pull request confirmation token is invalid.")
    if not hmac.compare_digest(pending.request_digest, expected_request_digest):
        raise AgentWorkspaceError("Pull request preview changed before confirmation.")
    server = _server(pending.server_id)
    if not hmac.compare_digest(pending.server_snapshot, _server_snapshot(server)):
        raise AgentWorkspaceError(
            "The GitHub connector changed after preview. Create a new preview."
        )
    tool = _pr_tool(tools)
    if tool is None or not hmac.compare_digest(pending.tool_digest, _tool_digest(tool)):
        raise AgentWorkspaceError(
            "The GitHub connector tool contract changed after preview. Create a new preview."
        )
    head, fingerprint, branch = _binding(project_id)
    if (
        not hmac.compare_digest(pending.head, head)
        or not hmac.compare_digest(pending.fingerprint, fingerprint)
        or not hmac.compare_digest(pending.branch, branch)
    ):
        raise AgentWorkspaceError("The repository changed after preview. Create a new preview.")
    if include_review_binding:
        return (
            server,
            dict(pending.request),
            {
                "head": head,
                "branch": branch,
                "workspaceFingerprint": fingerprint,
            },
        )
    return server, dict(pending.request)


def pull_request_review_binding_current(project_id: str, binding: dict[str, str]) -> bool:
    try:
        head, fingerprint, branch = _binding(project_id)
    except Exception:
        return False
    return (
        hmac.compare_digest(str(binding.get("head") or ""), head)
        and hmac.compare_digest(str(binding.get("branch") or ""), branch)
        and hmac.compare_digest(str(binding.get("workspaceFingerprint") or ""), fingerprint)
    )


def reset_pull_request_handoffs_for_tests() -> None:
    with _PENDING_LOCK:
        _PENDING.clear()


def remote_head_probe(tools: list[dict], request: dict) -> dict:
    matches = [tool for tool in tools if tool.get("name") == "get_commit"]
    schema = (
        (matches[0].get("inputSchema") or matches[0].get("input_schema") or {})
        if len(matches) == 1
        else {}
    )
    required = {"owner", "repo", "sha"}
    if (
        not isinstance(schema.get("properties"), dict)
        or not required.issubset(schema["properties"])
        or not set(schema.get("required") or ()).issubset(required)
    ):
        raise AgentWorkspaceError(
            "The GitHub connector needs get_commit to verify the published head before submission."
        )
    return {"owner": request["owner"], "repo": request["repo"], "sha": request["head"]}


def require_remote_head(result: str, expected: str) -> None:
    try:
        if len(result.encode("utf-8")) > 2_000_000:
            raise ValueError("oversized")
        value = json.loads(result)
        actual = value.get("sha") if isinstance(value, dict) else None
    except (TypeError, ValueError):
        actual = None
    if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
        raise AgentWorkspaceError(
            "The published branch does not match the reviewed local commit, or could not be verified. No pull request was submitted."
        )


__all__ = [
    "consume_pull_request_handoff",
    "prepare_pull_request_handoff",
    "pull_request_review_binding_current",
    "require_pull_request_tool",
    "reset_pull_request_handoffs_for_tests",
]
