# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io

from .factory_base import Factory, seeder

NAME = "character-matrix-name"
EDITED = "character-matrix-edited"
PERSONA = "persona-matrix-name"


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 6), (10, 20, 30)).save(buffer, format = "PNG")
    return buffer.getvalue()


@seeder("character")
def seed_character(account) -> dict[str, str]:
    from storage import characters_db
    from utils.account_context import run_as

    character = run_as(
        account,
        lambda: characters_db.create_character(
            {"name": NAME, "description": NAME}, avatar = _png(), avatar_type = "image/png"
        ),
    )
    return {"character_id": character["id"]}


@seeder("persona")
def seed_persona(account) -> dict[str, str]:
    from storage import characters_db
    from utils.account_context import run_as

    persona = run_as(account, lambda: characters_db.create_persona(PERSONA))
    run_as(account, characters_db.set_persona_avatar, persona["id"], _png(), "image/png")
    return {"persona_id": persona["id"]}


@seeder("character-chat")
def seed_character_chat(account) -> dict[str, str]:
    from storage import characters_db
    from utils.account_context import run_as

    def create():
        character = characters_db.create_character({"name": NAME, "firstMes": "hello"})
        return characters_db.create_character_chat(
            character_ids = [character["id"]], persona_id = None, user_name = "matrix"
        )

    created = run_as(account, create)
    return {"thread_id": created["threadId"]}


_IMAGE = "the matrix sends JSON, which is not an image, so the owner's answer is the image check"

FACTORIES = {
    "routes.characters:GET:/{character_id}": Factory("character", fragment = NAME, absent = NAME),
    "routes.characters:PATCH:/{character_id}": Factory(
        "character", {"name": EDITED}, fragment = EDITED, absent = NAME
    ),
    "routes.characters:DELETE:/{character_id}": Factory("character", success = 204, absent = NAME),
    "routes.characters:POST:/{character_id}/duplicate": Factory(
        "character", success = 201, fragment = f"{NAME} (copy)", absent = NAME
    ),
    "routes.characters:GET:/{character_id}/avatar": Factory("character"),
    "routes.characters:PUT:/{character_id}/avatar": Factory(
        "character", {}, success = 400, reason = _IMAGE
    ),
    "routes.characters:DELETE:/{character_id}/avatar": Factory(
        "character", fragment = '"hasAvatar":false'
    ),
    "routes.characters:GET:/{character_id}/export": Factory("character", fragment = None),
    "routes.characters:GET:/{character_id}/chats": Factory("character", fragment = '"chats":[]'),
    "routes.characters:PATCH:/personas/{persona_id}": Factory(
        "persona", {"description": EDITED}, fragment = EDITED, absent = PERSONA
    ),
    "routes.characters:DELETE:/personas/{persona_id}": Factory(
        "persona", success = 204, absent = PERSONA
    ),
    "routes.characters:GET:/personas/{persona_id}/avatar": Factory("persona"),
    "routes.characters:PUT:/personas/{persona_id}/avatar": Factory(
        "persona", {}, success = 400, reason = _IMAGE
    ),
    "routes.characters:DELETE:/personas/{persona_id}/avatar": Factory(
        "persona", fragment = '"hasAvatar":false'
    ),
    "routes.characters:GET:/chats/{thread_id}": Factory(
        "character-chat",
        fragment = '"cast"',
        owner = (204,),
        wrong = (204,),
        reason = "a chat without a scene in this account, its own or another's, answers 204 and reveals nothing",
    ),
    "routes.characters:PUT:/chats/{thread_id}": Factory(
        "character-chat", {"settings": {"replyOrder": "list"}}, fragment = '"replyOrder":"list"'
    ),
}

SKIPPED: dict[str, str] = {}
