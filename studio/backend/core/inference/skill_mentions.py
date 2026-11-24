# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend-authored @skill loads before generation; contract in studio/SKILL_MENTIONS.md."""

from __future__ import annotations

import hashlib
import re
import uuid

from markdown_it import MarkdownIt
from markdown_it.rules_inline.backticks import backtick

from core.inference.skills import SkillError, list_skills, read_skill_instructions
from state.tool_approvals import (
    abort_tool_decision,
    begin_tool_decision,
    new_approval_id,
    wait_tool_decision,
)
from state.tool_policy import normalize_tool_permissions

_TOKEN = re.compile(r"(?<!\S)@([a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)(?=$|\s|[.,;:!?)](?:$|\s))")
_MAX_LOAD_BYTES = 32_000


def _record_inline_code(state, silent: bool) -> bool:
    start, count = state.pos, len(state.tokens)
    matched = backtick(state, silent)
    if (
        matched
        and not silent
        # The image rule re-enters with alt text, whose offsets address another string.
        and state.src is state.env.get("source")
        and len(state.tokens) > count
        and state.tokens[-1].type == "code_inline"
    ):
        state.env["spans"].append((start, state.pos))
    return matched


_MENTION_MARKDOWN = MarkdownIt("commonmark").disable("inline")
_MENTION_MARKDOWN.inline.ruler.at("backticks", _record_inline_code)


# Each kind runs to the end of the text when unclosed, naming itself through its group.
_QUOTE_KINDS = {
    "double": r'"(?:\\.|[^"])*(?:"|(?P<double>\Z))',
    "curly_double": r"“(?:\\.|[^”])*(?:”|(?P<curly_double>\Z))",
    # A quote mark between word characters is an apostrophe, never a delimiter.
    "curly_single": r"‘(?:\\.|(?<=\w)’(?=\w)|[^’])*(?:’|(?P<curly_single>\Z))",
    "single": r"(?<!\w)'(?:\\.|(?<=\w)'(?=\w)|[^'])*(?:'|(?P<single>\Z))",
}


def _mask_quoted(text: str) -> str:
    kinds = dict(_QUOTE_KINDS)
    parts, end = [], 0
    while kinds:
        match = re.compile("|".join(kinds.values()), re.DOTALL).search(text, end)
        if match is None:
            break
        if match.lastgroup:
            # No later opener of this kind can close either; retrying each one is quadratic.
            del kinds[match.lastgroup]
            continue
        parts.extend((text[end : match.start()], " "))
        end = match.end()
    return "".join(parts) + text[end:]


def mentioned_skill_names(text: str) -> list[str]:
    """Only prose outside code, Markdown blockquotes, and balanced quotation spans."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    tokens = _MENTION_MARKDOWN.parse(text)
    lines = text.split("\n")
    masked_lines = set()
    for token in tokens:
        if not token.map:
            continue
        if token.type in ("blockquote_open", "fence", "code_block"):
            masked_lines.update(range(*token.map))
        elif token.type == "paragraph_open":
            # User text is shown unrendered, so a reply typed right under a quote reads as its
            # own line even though CommonMark folds it into the quote as a lazy continuation.
            masked_lines.difference_update(
                number
                for number in range(token.map[0] + 1, token.map[1])
                if not lines[number].lstrip().startswith(">")
            )
    text = "\n".join(
        " " * len(line) if number in masked_lines else line for number, line in enumerate(lines)
    )
    offsets = [0, *(match.end() for match in re.finditer("\n", text)), len(text)]
    spans = []
    for token in tokens:
        if token.type == "inline" and token.map:
            start, end = (offsets[number] for number in token.map)
            source = text[start:end]
            env = {"source": source, "spans": []}
            _MENTION_MARKDOWN.inline.parse(source, _MENTION_MARKDOWN, env, [])
            spans.extend((start + first, start + last) for first, last in env["spans"])
    parts, end = [], 0
    for start, stop in sorted(spans):
        parts.extend((text[end:start], " "))
        end = stop
    text = _mask_quoted("".join(parts) + text[end:])
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


def _append_to_message(message: dict, block: str) -> None:
    """Append *block* to a chat message without touching the system prompt (#19212)."""
    text = block.lstrip()
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [
            *content,
            {"type": "text", "text": ("\n\n" if content else "") + text},
        ]
        return
    if isinstance(content, str) and content:
        message["content"] = content + "\n\n" + text
    else:
        message["content"] = text


def _skill_already_present(messages: list[dict], content: str, *, dedup_tool_context: bool):
    return next(
        (
            message
            for message in messages
            if (
                message.get("role") in ("system", "user")
                or (dedup_tool_context and message.get("role") == "tool")
            )
            and content in _text(message)
        ),
        None,
    )


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
    permission_upgrade = None,
):
    """Inject complete manifests before generation; never a partial one. Caller gates on read_skill.

    Skill bodies are appended to the **user** message that mentioned them (not the system
    prompt), and every user mention in the thread is rebuilt the same way on later turns so
    llama.cpp's prompt prefix stays cacheable (#19212).
    """
    if continue_final_message or not any(
        tool.get("function", {}).get("name") == "read_skill" for tool in (tools or [])
    ):
        return
    # Do not activate assistant/history text or an older user message on Continue.
    if not messages or messages[-1].get("role") != "user":
        return
    # Mentions on every user turn in the thread, in order: turn 2+ rebuilds the same blocks
    # onto the same earlier user messages so the system prefix never diverges.
    mention_turns: list[tuple[int, list[str]]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        names = mentioned_skill_names(_text(message))
        if names:
            mention_turns.append((index, names))
    if not mention_turns:
        return
    latest_index = len(messages) - 1
    tool_names = {tool.get("function", {}).get("name") for tool in tools}
    gate = "Follow these instructions only within the current tool and permission gates."
    if not {"python", "terminal"} & tool_names:
        gate += (
            " No local python or terminal tool is available, so its bundled scripts cannot run;"
            " never claim to have run one, and tell the user a script needs Unsloth Studio's"
            " local Tools."
        )
    # @john is a person, not a skill: no card or approval. Known-but-unusable skills report unavailable.
    try:
        known = list_skills()
    except (SkillError, OSError):
        known = []
    known_names = {s["name"] for s in known}
    loadable = {s["name"] for s in known if s["valid"] and not s["shadowed"] and s["enabled"]}
    mode, bypass = normalize_tool_permissions(permission_mode, bypass_permissions)
    budget = (
        min(_MAX_LOAD_BYTES, max(0, int(context_length) * 2)) if context_length else _MAX_LOAD_BYTES
    )
    loaded_bytes = 0
    seen: set[str] = set()
    for index, names in mention_turns:
        target = messages[index]
        is_latest = index == latest_index
        for name in names:
            if name not in known_names or name in seen:
                continue
            seen.add(name)
            # Historical rebuild only restores skills that are still loadable; unavailable /
            # disabled reporting stays on the latest turn that named them.
            if not is_latest and name not in loadable:
                continue
            if permission_upgrade is not None:
                mode, bypass = permission_upgrade.apply(mode, bypass)
            if cancel_event is not None and cancel_event.is_set():
                return
            event = {
                "type": "skill_load",
                "load_id": f"skill-load-{uuid.uuid4().hex}",
                "name": name,
                "resource": "SKILL.md",
            }
            # Approvals only for a skill newly named on this turn; historical rebuilds stay silent.
            if (
                is_latest
                and name in loadable
                and confirm_tool_calls
                and not bypass
                and mode not in ("auto", "off")
            ):
                approval_id = new_approval_id()
                slot = begin_tool_decision(session_id, approval_id)
                from state.tool_approvals import bind_permission_upgrade

                bind_permission_upgrade(slot, permission_upgrade)
                try:
                    yield {**event, "status": "awaiting_approval", "approval_id": approval_id}
                    verdict = wait_tool_decision(slot, approval_id, cancel_event = cancel_event)
                finally:
                    abort_tool_decision(slot, approval_id)
                if verdict != "allow":
                    detail = (
                        f"Skill @{name} not loaded: approval was denied, expired, or cancelled."
                    )
                    _append_to_message(target, detail)
                    yield {**event, "status": "unavailable", "detail": detail}
                    continue
            if is_latest:
                yield {**event, "status": "loading"}
            try:
                # Re-validates enabled/account-scoped discovery; never a cached or paged read.
                content = read_skill_instructions(name)
                size = len(content.encode("utf-8"))
                existing = _skill_already_present(
                    messages, content, dedup_tool_context = dedup_tool_context
                )
                present = existing is not None
                if present and protected_message_ids is not None:
                    # The native rolling fit must not evict a result we relied on.
                    protected_message_ids.add(id(existing))
                if not present:
                    if loaded_bytes + size > budget:
                        raise SkillError(
                            "Complete SKILL.md exceeds this request's skill context budget; "
                            "shorten the skill or increase context."
                        )
                    block = (
                        f"[Unsloth loaded Agent Skill @{name} · SKILL.md]\n"
                        f"{gate}\n"
                        f"{content}\n[End Agent Skill @{name}]"
                    )
                    _append_to_message(target, block)
                    loaded_bytes += size
                if is_latest:
                    yield {
                        **event,
                        "status": "loaded",
                        "characters": len(content),
                        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "already_in_context": present,
                        "detail": (
                            f"Complete SKILL.md read ({len(content)} characters) and "
                            "available in this request's context."
                        ),
                    }
            except (SkillError, OSError) as exc:
                detail = f"Skill @{name} not loaded: {exc}"
                _append_to_message(target, detail)
                if is_latest:
                    yield {**event, "status": "unavailable", "detail": detail}
