# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opening a character chat: its title, settings and the greeting turns it starts with."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from core.characters.cards import MAX_FIELD_CHARS, MAX_GREETINGS, card_data
from storage import characters_db as db
from storage.studio_db import now_ms

# Tools stay off in a character chat unless the user turns them on there.
CHARACTER_CHAT_SETTINGS = {
    "toolsEnabled": False,
    "codeToolsEnabled": False,
    "imageToolsEnabled": False,
    "webFetchToolsEnabled": False,
    "deepResearchEnabled": False,
    "artifactsEnabled": False,
    "mcpEnabledForChat": False,
    "ragEnabled": False,
}
OPENING_TEXT = "[Start a new chat]"
# Caps the later members' greetings copied under each alternate greeting in a group.
MAX_GREETING_COPY_ROWS = 512

_MACRO_PATTERNS = (
    (re.compile(r"\{\{\s*char\s*\}\}|<BOT>", re.IGNORECASE), "char"),
    (re.compile(r"\{\{\s*user\s*\}\}|<USER>", re.IGNORECASE), "user"),
)


def fill_names(text: str, char: str, user: str) -> str:
    for pattern, which in _MACRO_PATTERNS:
        text = pattern.sub((char if which == "char" else user).replace("\\", "\\\\"), text)
    return text


def chat_title(names: list[str]) -> str:
    if not names:
        return "Character chat"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return f"{', '.join(names[:-1])} & {names[-1]}"


def group_only_greetings(character: dict) -> list[str]:
    card = character.get("card")
    if not isinstance(card, dict):
        return []
    greetings = card_data(card)[0].get("group_only_greetings")
    if not isinstance(greetings, list):
        return []
    # Not normalized at import like the other greetings, so bounded the same way here.
    return [g[:MAX_FIELD_CHARS] for g in greetings if isinstance(g, str)][:MAX_GREETINGS]


def opening_messages(
    thread_id: str,
    characters: list[dict],
    speaker_name: str,
    greetings: Optional[dict],
    created: int,
) -> tuple[list[dict], list[str], str]:
    """The opening turn and each member's greetings, with the greeting ids and the head id."""

    def message(parent_id: Optional[str], role: str, text: str, roleplay: dict) -> dict:
        nonlocal created
        created += 1
        return {
            "id": str(uuid.uuid4()),
            "threadId": thread_id,
            "parentId": parent_id,
            "role": role,
            "content": [{"type": "text", "text": text}],
            "metadata": {"roleplay": roleplay},
            "createdAt": created,
        }

    def greeting_texts(character: dict) -> list[tuple[int, str]]:
        source = [character["firstMes"], *character["alternateGreetings"]]
        group_only = group_only_greetings(character) if len(characters) > 1 else []
        resolved = (greetings or {}).get(character["id"])
        # The client resolves every macro; this covers API callers that send none, or no group-only ones.
        if resolved is not None and len(resolved) == len(source):
            resolved = [
                *resolved,
                *(fill_names(g, character["name"], speaker_name) for g in group_only),
            ]
        elif resolved is None or len(resolved) != len(source) + len(group_only):
            resolved = [
                fill_names(g, character["name"], speaker_name) for g in (*source, *group_only)
            ]
        # Clipped after names are filled: a field of macros can expand far past the raw limit.
        return [(i, g[:MAX_FIELD_CHARS]) for i, g in enumerate(resolved) if g and g.strip()]

    texts_by_member = [greeting_texts(c) for c in characters]
    copies_left = MAX_GREETING_COPY_ROWS

    def reply(parent_id: str, character: dict, source_index: int, text: str) -> dict:
        return message(
            parent_id,
            "assistant",
            text,
            {
                "kind": "reply",
                "speakerId": character["id"],
                "speakerName": character["name"],
                "greeting": source_index,
            },
        )

    def members_from(
        index: int, parent_id: str, alternates: bool
    ) -> tuple[list[dict], list[str], str]:
        """Member `index` onward under `parent_id`; without `alternates`, main greetings only."""
        nonlocal copies_left
        rows: list[dict] = []
        ids: list[str] = []
        for at in range(index, len(characters)):
            character = characters[at]
            if len(characters) > 1 and at > 0:
                # A hidden turn marker keeps each member's greeting answering a user turn.
                marker = message(
                    parent_id,
                    "user",
                    f"[{character['name']}'s turn]",
                    {"kind": "turn", "speakerId": character["id"]},
                )
                rows.append(marker)
                parent_id = marker["id"]
            texts = texts_by_member[at]
            if not texts:
                continue
            for source_index, text in texts[1:] if alternates else []:
                alternate = reply(parent_id, character, source_index, text)
                rows.append(alternate)
                # In a group, a swiped greeting keeps the members after it: each alternate gets
                # their main greetings too, so flipping Ann's greeting does not hide Bo's.
                later = 2 * (len(characters) - 1 - at)
                if later and copies_left >= later:
                    copies_left -= later
                    rows.extend(members_from(at + 1, alternate["id"], False)[0])
            # Alternates first: with no saved branch choice a chat opens on its newest row, which
            # should be the main greeting and the members after it, here and on every device.
            main = reply(parent_id, character, *texts[0])
            rows.append(main)
            ids.append(main["id"])
            parent_id = main["id"]
        return rows, ids, parent_id

    opening = message(None, "user", OPENING_TEXT, {"kind": "opening"})
    rows, greeting_ids, head_id = members_from(0, opening["id"], True)
    return [opening, *rows], greeting_ids, head_id


def create_character_chat(
    *,
    character_ids: list[str],
    persona_id: Optional[str],
    user_name: str,
    project_id: Optional[str] = None,
    settings: Optional[dict] = None,
    greetings: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Thread, cast, scene and opening turn. Greetings hang under a hidden user turn so alternates are
    sibling replies and strict chat templates see a user turn first."""
    requested = list(dict.fromkeys(character_ids))
    if len(requested) > db.MAX_CAST:
        raise db.CharacterLimitReached(f"A chat can have up to {db.MAX_CAST} characters.")
    characters = db.get_characters(requested)
    # Every requested member must exist: a group must not quietly become a smaller chat.
    if not characters or len(characters) != len(requested):
        raise db.CharacterNotFound("character")
    persona = db.get_persona(persona_id) if persona_id else None
    if persona_id and persona is None:
        raise db.PersonaNotFound(persona_id)
    speaker_name = (persona or {}).get("name") or user_name or "User"
    thread_id = str(uuid.uuid4())
    now = now_ms()
    messages, greeting_ids, head_id = opening_messages(
        thread_id, characters, speaker_name, greetings, now
    )
    db.create_character_chat(
        thread = {
            "id": thread_id,
            "title": chat_title([c["name"] for c in characters]),
            "modelType": "base",
            "modelId": "",
            "projectId": project_id,
            "createdAt": now,
            "updatedAt": now,
            "settings": dict(CHARACTER_CHAT_SETTINGS),
        },
        character_ids = [c["id"] for c in characters],
        persona_id = persona["id"] if persona else None,
        settings = settings,
        messages = messages,
    )
    return {"threadId": thread_id, "greetingIds": greeting_ids, "headId": head_id}
