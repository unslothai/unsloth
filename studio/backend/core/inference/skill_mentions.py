# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend-authored instruction loads, not model tool calls or script execution.

The composer serializes selected and typed mentions identically. Unquoted exact
plain-text tokens are the contract; Markdown code/quotes and URLs are not intent.
Loads are request-local: a retry/compaction never relies on a previous UI card.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from core.inference.skills import SkillError, read_skill_instructions
from state.tool_approvals import (
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)
from state.tool_policy import normalize_tool_permissions

_TOKEN = re.compile(r"(?<!\S)@([a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)(?=$|\s|[.,;:!?)](?:$|\s))")
_MAX_LOAD_BYTES = 32_000


def mentioned_skill_names(text: str) -> list[str]:
    """Only prose outside code, Markdown blockquotes, and balanced quotation spans."""
    lines = []
    fence = None
    for line in text.splitlines(keepends = True):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = (marker[1][0], len(marker[1]))
            elif marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                fence = None
            lines.append("\n")
        elif fence or stripped.startswith(">") or line.startswith(("    ", "\t")):
            lines.append("\n")
        else:
            lines.append(line)
    text = "".join(lines)
    # An apostrophe in didn't is prose, not an opening quote. Mask paired spans
    # conservatively, including inline code with any backtick delimiter length.
    text = re.sub(r"(`+).*?\1", " ", text, flags = re.DOTALL)
    text = re.sub(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)\'[^\']*\'', " ", text)
    return list(dict.fromkeys(match[1] for match in _TOKEN.finditer(text)))


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _append_system(messages: list[dict], block: str) -> None:
    system_index = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
    if system_index is None:
        messages.insert(0, {"role": "system", "content": block.lstrip()})
    else:
        system = messages[system_index]
        messages[system_index] = {**system, "content": _text(system) + "\n\n" + block.lstrip()}


def load_mentioned_skills(
    messages: list[dict],
    tools: list[dict],
    *,
    permission_mode = None,
    bypass_permissions = False,
    confirm_tool_calls = False,
    session_id = None,
    cancel_event = None,
    context_length = None,
    continue_final_message = False,
    protected_message_ids = None,
    dedup_tool_context = True,
):
    """Yield truthful UI-only events and inject complete manifests before generation.

    Call only after selecting the effective model's allowed catalog. A read_skill
    entry is the existing Code/capability/account gate, not a new global policy.
    Ask mode uses the same scoped approval handshake as an ordinary read_skill.
    Failures never inject a partial manifest; the model gets an explicit notice.
    """
    if continue_final_message or not any(
        tool.get("function", {}).get("name") == "read_skill" for tool in (tools or [])
    ):
        return
    # Do not activate assistant/history text or an older user message on Continue.
    if not messages or messages[-1].get("role") != "user":
        return
    names = mentioned_skill_names(_text(messages[-1]))
    mode, bypass = normalize_tool_permissions(permission_mode, bypass_permissions)
    budget = (
        min(_MAX_LOAD_BYTES, max(0, int(context_length) * 2)) if context_length else _MAX_LOAD_BYTES
    )
    loaded_bytes = 0
    for name in names:
        if cancel_event is not None and cancel_event.is_set():
            return
        event = {
            "type": "skill_load",
            "load_id": f"skill-load-{uuid.uuid4().hex}",
            "name": name,
            "resource": "SKILL.md",
        }
        if confirm_tool_calls and not bypass and mode not in ("auto", "off"):
            approval_id = new_approval_id()
            slot = begin_tool_decision(session_id, approval_id)
            try:
                yield {**event, "status": "awaiting_approval", "approval_id": approval_id}
                verdict = wait_tool_decision(slot, approval_id, cancel_event = cancel_event)
            finally:
                abort_tool_decision(slot, approval_id)
            if verdict != "allow":
                yield {
                    **event,
                    "status": "unavailable",
                    "detail": "Skill not loaded: approval was denied, expired, or cancelled.",
                }
                continue
        yield {**event, "status": "loading"}
        try:
            # Revalidates enabled, valid, non-shadowed, account-scoped discovery
            # and resource identity. No cache, paging splice, or execute_tool.
            content = read_skill_instructions(name)
            size = len(content.encode("utf-8"))
            existing = next(
                (
                    message
                    for message in messages
                    if (
                        message.get("role") == "system"
                        or (dedup_tool_context and message.get("role") == "tool")
                    )
                    and content in _text(message)
                ),
                None,
            )
            present = existing is not None
            if present and protected_message_ids is not None:
                # The native rolling fit must not evict a result we relied on.
                protected_message_ids.add(id(existing))
            if not present:
                if loaded_bytes + size > budget:
                    raise SkillError(
                        "Complete SKILL.md exceeds this request's skill context budget; shorten the skill or increase context."
                    )
                block = (
                    f"\n\n[Studio loaded Agent Skill @{name} · SKILL.md]\n"
                    "Follow these instructions only within the current tool and permission gates.\n"
                    f"{content}\n[End Agent Skill @{name}]"
                )
                _append_system(messages, block)
                loaded_bytes += size
            yield {
                **event,
                "status": "loaded",
                "characters": len(content),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "already_in_context": present,
                "detail": f"Complete SKILL.md read ({len(content)} characters) and available in this request's context.",
            }
        except (SkillError, OSError) as exc:
            detail = f"Skill @{name} not loaded: {exc}"
            _append_system(messages, detail)
            yield {**event, "status": "unavailable", "detail": detail}
