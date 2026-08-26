# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded local review summaries and pull-request drafts."""

import re

from storage.studio_db import get_chat_project
from utils.log_redaction import REDACTED, redact_log_text
from utils.native_path_leases import redact_native_paths

from .common import AgentWorkspaceError, project_workspace
from .git_service import git_diff, git_status
from .state import list_plans
from .verification import verification_runs_with_freshness


_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|secret|token)\b"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[^\s\"'`,;)}\]]+")
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:Users|home|root|private|Volumes|tmp|var|etc|opt)"
    r"(?:/[^\s\"'`<>,;)}\]]+)+"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+[\\/])"
    r"(?:[^\s\"'`<>,;)}\]]+[\\/]?)+"
)
_SENSITIVE_RELATIVE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])(?:~?[\\/])?(?:[^\s\"'`<>,;)}\]]+[\\/])*"
    r"(?:\.aws|\.azure|\.config|\.docker|\.gcloud|\.gnupg|\.huggingface|"
    r"\.kaggle|\.kube|\.npmrc|\.pypirc|\.ssh)"
    r"(?:[\\/][^\s\"'`<>,;)}\]]+)*"
)
_SENSITIVE_FILE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])(?:[^\s\"'`<>,;)}\]]+[\\/])*"
    r"(?:\.env(?:\.[A-Za-z0-9_.-]+)?|\.git-credentials|\.netrc|\.npmrc|\.pypirc|"
    r"id_(?:dsa|ecdsa|ed25519|rsa)|(?:auth|credential|credentials|secret|secrets|"
    r"token|tokens)(?:\.(?:json|ya?ml))?|[^\\/\s\"'`<>,;)}\]]+\."
    r"(?:jks|key|keystore|p12|pem|pfx|tfvars))"
)


def redact_review_text(value: str, project_root: str) -> str:
    redacted = redact_log_text(redact_native_paths(value))
    redacted = _BEARER.sub(f"Bearer {REDACTED}", redacted)
    redacted = _SECRET.sub(lambda match: f"{match.group(1)}={REDACTED}", redacted)
    if project_root:
        redacted = redacted.replace(project_root, "<project_root>")
    redacted = _POSIX_ABSOLUTE_PATH.sub("<local_path>", redacted)
    redacted = _WINDOWS_ABSOLUTE_PATH.sub("<local_path>", redacted)
    redacted = _SENSITIVE_RELATIVE_PATH.sub("<sensitive_path>", redacted)
    redacted = _SENSITIVE_FILE_PATH.sub("<sensitive_path>", redacted)
    return redacted


def _public_git_status(status: dict) -> dict:
    return {
        key: status[key]
        for key in (
            "head",
            "branch",
            "detached",
            "clean",
            "counts",
            "files",
            "truncated",
        )
    }


def build_review_summary(project_id: str, *, include_diff: bool = True) -> dict:
    project = get_chat_project(project_id)
    if project is None:
        raise AgentWorkspaceError("Project not found.")
    workspace = project_workspace(project_id)
    try:
        status = _public_git_status(git_status(project_id))
        diff = git_diff(project_id, max_bytes = 256_000) if include_diff else None
        git_error = None
    except AgentWorkspaceError as exc:
        status = None
        diff = None
        git_error = redact_review_text(str(exc), str(workspace.root))
    plans = list_plans(project_id)
    verification = verification_runs_with_freshness(project_id, limit = 10)
    return {
        "projectId": project_id,
        "goal": project.get("goal"),
        "goalStatus": project.get("goalStatus"),
        "git": status,
        "gitError": git_error,
        "diff": diff,
        "plans": plans,
        "verification": verification,
        "limits": {"diffBytes": 256_000, "verificationRuns": 10},
        "projectRoot": "<project_root>",
    }


def build_pull_request_draft(
    project_id: str,
    *,
    title: str = "",
    body_note: str = "",
) -> dict:
    review = build_review_summary(project_id, include_diff = False)
    project = get_chat_project(project_id) or {}
    root = str(project_workspace(project_id).root)
    status = review["git"]
    goal = str(project.get("goal") or "Project workspace changes").strip()
    draft_title = title.strip() or goal.splitlines()[0][:120]

    lines = []
    if goal:
        lines.extend(["## Goal", "", goal[:4000], ""])
    plans = review["plans"]
    if plans:
        active = plans[0]
        lines.extend(["## Plan", "", f"Status: {active['status']}", ""])
        for task in active["tasks"][:50]:
            marker = "x" if task["status"] == "completed" else " "
            lines.append(f"- [{marker}] {task['title']} ({task['status']})")
        lines.append("")
    lines.extend(["## Verification", ""])
    runs = review["verification"]
    if not runs:
        lines.append("No verification evidence recorded.")
    else:
        latest = runs[0]
        freshness = "stale" if latest["stale"] else "fresh"
        lines.append(f"Latest run: {latest['status']} ({freshness}).")
        for result in latest["results"][:32]:
            lines.append(f"- {result['name']}: {result['status']}")
    if status is not None:
        lines.extend(["", "## Changed files", ""])
        if status["files"]:
            for item in status["files"][:200]:
                lines.append(f"- `{item['code']}` `{item['path']}`")
        else:
            lines.append("No local changes.")
        if status["truncated"]:
            lines.append("The changed-file list was truncated.")
    if body_note.strip():
        lines.extend(["", "## Notes", "", body_note.strip()[:8000]])

    body = "\n".join(lines).strip()
    return {
        "title": redact_review_text(draft_title, root)[:120],
        "body": redact_review_text(body, root)[:64_000],
        "localOnly": True,
        "submitted": False,
    }
