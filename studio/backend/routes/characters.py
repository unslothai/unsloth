# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Characters, personas and character chats.

Cards import from PNG (V2 ``chara`` / V3 ``ccv3`` chunks), JSON (V1, V2, V3) or CHARX, and
export as a PNG carrying both chunks or as V3 JSON.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.characters import cards, starters
from storage import characters_db as db

router = APIRouter()

_NOT_FOUND = "Character not found"
_PERSONA_NOT_FOUND = "Persona not found"
_LONG = 200_000


def _not_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


Name = Annotated[str, Field(max_length = 200), AfterValidator(_not_blank)]
LongText = Annotated[str, Field(max_length = _LONG)]
ShortText = Annotated[str, Field(max_length = 200)]
Tag = Annotated[str, Field(max_length = cards.MAX_TAG_CHARS), AfterValidator(_not_blank)]


class DepthPrompt(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    prompt: LongText = ""
    depth: int = Field(default = 4, ge = 0, le = 100)
    role: Literal["system", "user", "assistant"] = "system"


class CharacterFields(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    description: Optional[LongText] = None
    personality: Optional[LongText] = None
    scenario: Optional[LongText] = None
    firstMes: Optional[LongText] = None
    alternateGreetings: Optional[list[LongText]] = Field(
        default = None, max_length = cards.MAX_GREETINGS
    )
    mesExample: Optional[LongText] = None
    systemPrompt: Optional[LongText] = None
    postHistoryInstructions: Optional[LongText] = None
    creatorNotes: Optional[LongText] = None
    creator: Optional[ShortText] = None
    characterVersion: Optional[ShortText] = None
    tags: Optional[list[Tag]] = Field(default = None, max_length = cards.MAX_TAGS)
    talkativeness: Optional[float] = Field(default = None, ge = 0, le = 1)
    favorite: Optional[bool] = None
    depthPrompt: Optional[DepthPrompt] = None
    characterBook: Optional[dict[str, Any]] = None


class CharacterCreate(CharacterFields):
    name: Name


class CharacterUpdate(CharacterFields):
    name: Optional[Name] = None
    # Fields set to null here are cleared; elsewhere null means unchanged.
    clear: list[Literal["depthPrompt", "characterBook"]] = Field(default_factory = list)


class PersonaCreate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: Name
    description: LongText = ""
    isDefault: bool = False


class PersonaUpdate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: Optional[Name] = None
    description: Optional[LongText] = None
    isDefault: Optional[bool] = None


class AuthorsNote(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    text: LongText = ""
    depth: int = Field(default = 4, ge = 0, le = 100)


class SceneSettings(BaseModel):
    """Per-chat settings; every field is optional so older chats read as defaults."""

    model_config = ConfigDict(extra = "forbid")

    scenario: Optional[LongText] = None
    authorsNote: Optional[AuthorsNote] = None
    replyOrder: Optional[Literal["natural", "list", "manual"]] = None
    allowSelfReplies: Optional[bool] = None
    view: Optional[Literal["chat", "stage", "novel"]] = None


class CharacterChatCreate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    characterIds: list[str] = Field(min_length = 1, max_length = db.MAX_CAST)
    personaId: Optional[str] = Field(default = None, max_length = 64)
    # The name {{user}} becomes in greetings when no persona is picked.
    userName: str = Field(default = "", max_length = 200)
    projectId: Optional[str] = Field(default = None, max_length = 128)
    settings: Optional[SceneSettings] = None
    # [firstMes, *alternateGreetings] per character, resolved by the client (its clock and locale).
    greetings: Optional[
        dict[
            Annotated[str, Field(max_length = 64)],
            Annotated[list[LongText], Field(max_length = cards.MAX_GREETINGS + 1)],
        ]
    ] = Field(default = None, max_length = db.MAX_CAST)


CharacterId = Annotated[str, Field(max_length = 64)]


class SceneUpdate(BaseModel):
    """One edit, applied on top of the stored scene: settings merge (null clears), cast changes are by id."""

    model_config = ConfigDict(extra = "forbid")

    add: list[CharacterId] = Field(default_factory = list, max_length = db.MAX_CAST)
    remove: list[CharacterId] = Field(default_factory = list, max_length = db.MAX_CAST)
    mute: dict[CharacterId, bool] = Field(default_factory = dict, max_length = db.MAX_CAST)
    personaId: Optional[str] = Field(default = None, max_length = 64)
    settings: Optional[SceneSettings] = None


def _character_or_404(character_id: str) -> dict:
    character = db.get_character(character_id)
    if character is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return character


def _public(character: dict) -> dict:
    """The stored card stays server-side; the client edits fields, not the raw card."""
    out = {key: value for key, value in character.items() if key != "card"}
    out["cardSpec"] = (
        cards.card_data(character["card"])[1] if isinstance(character.get("card"), dict) else None
    )
    return out


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code = 413, detail = "The file is too large.")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code = 413, detail = "The file is too large.")
    return bytes(data)


def _image_response(data: bytes, media_type: Optional[str]) -> Response:
    return Response(
        content = data,
        media_type = media_type or cards.sniff_image_type(data) or "application/octet-stream",
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _checked_image(data: bytes) -> tuple[bytes, str]:
    checked = cards.checked_portrait(data)
    if checked is None:
        raise HTTPException(
            status_code = 400, detail = "The picture must be a PNG, JPEG, WebP or GIF image."
        )
    return checked


@router.get("")
def list_characters(current_subject: str = Depends(get_current_subject)):
    starters.seed_starters()
    return {"characters": [_public(c) for c in db.list_characters()]}


@router.post("", status_code = 201)
def create_character(payload: CharacterCreate, current_subject: str = Depends(get_current_subject)):
    fields = payload.model_dump(exclude_none = True)
    try:
        return _public(db.create_character(fields))
    except db.LimitReached as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None


@router.post("/import", status_code = 201)
async def import_character(
    request: Request,
    filename: str = Query("", max_length = 255),
    current_subject: str = Depends(get_current_subject),
):
    data = await _read_body(request, cards.MAX_CARD_BYTES)
    try:
        parsed = await asyncio.to_thread(cards.parse_card_file, data, filename)
    except cards.CardError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    try:
        character = await asyncio.to_thread(
            db.create_character,
            cards.character_fields(parsed),
            card = parsed.card,
            avatar = parsed.avatar,
            avatar_type = parsed.avatar_type,
        )
    except db.LimitReached as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    return {
        "character": _public(character),
        "format": parsed.source_format,
        "spec": parsed.spec,
        "warnings": parsed.warnings,
    }


@router.get("/chat-index")
def chat_index(current_subject: str = Depends(get_current_subject)):
    return {"chats": db.chat_index()}


@router.get("/personas")
def list_personas(current_subject: str = Depends(get_current_subject)):
    return {"personas": db.list_personas()}


@router.get("/{character_id}")
def get_character(character_id: str, current_subject: str = Depends(get_current_subject)):
    return _public(_character_or_404(character_id))


@router.patch("/{character_id}")
def update_character(
    character_id: str,
    payload: CharacterUpdate,
    current_subject: str = Depends(get_current_subject),
):
    fields = payload.model_dump(exclude_none = True, exclude = {"clear"})
    for cleared in payload.clear:
        fields[cleared] = None
    updated = db.update_character(character_id, fields)
    if updated is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return _public(updated)


@router.delete("/{character_id}", status_code = 204)
def delete_character(character_id: str, current_subject: str = Depends(get_current_subject)):
    if not db.delete_character(character_id):
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return Response(status_code = 204)


@router.post("/{character_id}/duplicate", status_code = 201)
def duplicate_character(character_id: str, current_subject: str = Depends(get_current_subject)):
    source = _character_or_404(character_id)
    fields = {k: v for k, v in source.items() if k in CharacterFields.model_fields}
    fields["name"] = f"{source['name']} (copy)"[:200]
    fields["extensions"] = source.get("extensions") or {}
    avatar = db.read_image("characters", character_id) if source["hasAvatar"] else None
    try:
        copy = db.create_character(
            fields, card = source.get("card"), avatar = avatar, avatar_type = source.get("avatarType")
        )
    except db.LimitReached as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    return _public(copy)


@router.get("/{character_id}/avatar")
def get_character_avatar(character_id: str, current_subject: str = Depends(get_current_subject)):
    character = _character_or_404(character_id)
    data = db.read_image("characters", character_id) if character["hasAvatar"] else None
    if not data:
        raise HTTPException(status_code = 404, detail = "This character has no picture.")
    return _image_response(data, character["avatarType"])


@router.put("/{character_id}/avatar")
async def put_character_avatar(
    character_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _character_or_404(character_id)
    data = await _read_body(request, db.MAX_AVATAR_BYTES)
    data, media_type = await asyncio.to_thread(_checked_image, data)
    updated = await asyncio.to_thread(db.set_character_avatar, character_id, data, media_type)
    if updated is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return _public(updated)


@router.delete("/{character_id}/avatar")
def delete_character_avatar(character_id: str, current_subject: str = Depends(get_current_subject)):
    updated = db.set_character_avatar(character_id, None, None)
    if updated is None:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND)
    return _public(updated)


def _download_name(name: str, extension: str) -> str:
    safe = re.sub(r"[^\w\- .]+", "", name).strip() or "character"
    return f"{safe[:80]}.{extension}"


@router.get("/{character_id}/export")
def export_character(
    character_id: str,
    format: Literal["png", "json"] = Query("png"),
    current_subject: str = Depends(get_current_subject),
):
    character = _character_or_404(character_id)
    filename = _download_name(character["name"], format)
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    if format == "json":
        body = json.dumps(cards.build_card_v3(character), ensure_ascii = False, indent = 2)
        return Response(
            content = body.encode("utf-8"),
            media_type = "application/json",
            headers = {"Content-Disposition": disposition},
        )
    avatar = db.read_image("characters", character_id) if character["hasAvatar"] else None
    return Response(
        content = cards.export_png(character, avatar),
        media_type = "image/png",
        headers = {"Content-Disposition": disposition},
    )


@router.get("/{character_id}/chats")
def list_character_chats(character_id: str, current_subject: str = Depends(get_current_subject)):
    _character_or_404(character_id)
    return {"chats": db.chats_for_character(character_id)}


def _persona_or_404(persona_id: str) -> dict:
    persona = db.get_persona(persona_id)
    if persona is None:
        raise HTTPException(status_code = 404, detail = _PERSONA_NOT_FOUND)
    return persona


@router.post("/personas", status_code = 201)
def create_persona(payload: PersonaCreate, current_subject: str = Depends(get_current_subject)):
    try:
        return db.create_persona(payload.name, payload.description, payload.isDefault)
    except db.LimitReached as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None


@router.patch("/personas/{persona_id}")
def update_persona(
    persona_id: str,
    payload: PersonaUpdate,
    current_subject: str = Depends(get_current_subject),
):
    updated = db.update_persona(persona_id, payload.model_dump(exclude_none = True))
    if updated is None:
        raise HTTPException(status_code = 404, detail = _PERSONA_NOT_FOUND)
    return updated


@router.delete("/personas/{persona_id}", status_code = 204)
def delete_persona(persona_id: str, current_subject: str = Depends(get_current_subject)):
    if not db.delete_persona(persona_id):
        raise HTTPException(status_code = 404, detail = _PERSONA_NOT_FOUND)
    return Response(status_code = 204)


@router.get("/personas/{persona_id}/avatar")
def get_persona_avatar(persona_id: str, current_subject: str = Depends(get_current_subject)):
    persona = _persona_or_404(persona_id)
    data = db.read_image("personas", persona_id) if persona["hasAvatar"] else None
    if not data:
        raise HTTPException(status_code = 404, detail = "This persona has no picture.")
    return _image_response(data, persona["avatarType"])


@router.put("/personas/{persona_id}/avatar")
async def put_persona_avatar(
    persona_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _persona_or_404(persona_id)
    data = await _read_body(request, db.MAX_AVATAR_BYTES)
    data, media_type = await asyncio.to_thread(_checked_image, data)
    updated = await asyncio.to_thread(db.set_persona_avatar, persona_id, data, media_type)
    if updated is None:
        raise HTTPException(status_code = 404, detail = _PERSONA_NOT_FOUND)
    return updated


@router.delete("/personas/{persona_id}/avatar")
def delete_persona_avatar(persona_id: str, current_subject: str = Depends(get_current_subject)):
    updated = db.set_persona_avatar(persona_id, None, None)
    if updated is None:
        raise HTTPException(status_code = 404, detail = _PERSONA_NOT_FOUND)
    return updated


@router.post("/chats", status_code = 201)
def create_character_chat(
    payload: CharacterChatCreate, current_subject: str = Depends(get_current_subject)
):
    if payload.personaId and db.get_persona(payload.personaId) is None:
        raise HTTPException(status_code = 400, detail = _PERSONA_NOT_FOUND)
    if payload.projectId:
        from storage import studio_db
        if studio_db.get_chat_project(payload.projectId) is None:
            raise HTTPException(status_code = 400, detail = "Project not found")
    try:
        created = db.create_character_chat(
            character_ids = payload.characterIds,
            persona_id = payload.personaId,
            user_name = payload.userName,
            project_id = payload.projectId,
            settings = payload.settings.model_dump(exclude_none = True) if payload.settings else None,
            greetings = payload.greetings,
        )
    except LookupError:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND) from None
    except db.LimitReached as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    return {**created, "scene": db.get_chat_scene(created["threadId"])}


@router.get("/chats/{thread_id}")
def get_chat_scene(thread_id: str, current_subject: str = Depends(get_current_subject)):
    """The chat's cast and scene; 204 for any chat without one, including a new chat not saved yet, so the
    client can ask about every chat it opens without logging an error."""
    scene = db.get_chat_scene(thread_id)
    return scene if scene is not None else Response(status_code = 204)


@router.put("/chats/{thread_id}")
def update_chat_scene(
    thread_id: str,
    payload: SceneUpdate,
    current_subject: str = Depends(get_current_subject),
):
    if "personaId" in payload.model_fields_set and payload.personaId:
        if db.get_persona(payload.personaId) is None:
            raise HTTPException(status_code = 400, detail = _PERSONA_NOT_FOUND)
    try:
        scene = db.update_chat_scene(
            thread_id,
            add = payload.add,
            remove = payload.remove,
            mute = payload.mute,
            persona_id = payload.personaId,
            persona_set = "personaId" in payload.model_fields_set,
            settings = payload.settings.model_dump(exclude_unset = True)
            if payload.settings is not None
            else None,
        )
    except LookupError:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND) from None
    except (db.LimitReached, db.NoSpeakerLeft) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    if scene is None:
        raise HTTPException(status_code = 404, detail = "Chat not found")
    return scene
