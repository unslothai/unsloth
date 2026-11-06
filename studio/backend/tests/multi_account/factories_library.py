# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

FOLDER_NAME = "library-matrix-folder"
EDITED = "library-matrix-edited"
NOTE = "library-matrix-note.md"
CONTENT = b"library matrix sentinel note\n"


@seeder("library-folder")
def seed_library_folder(account) -> dict[str, str]:
    from storage import library_db
    from utils.account_context import run_as

    folder = run_as(account, library_db.create_folder, FOLDER_NAME)
    return {"folder_id": folder["id"]}


@seeder("library-upload")
def seed_library_upload(account) -> dict[str, str]:
    from core import library
    from utils.account_context import run_as

    record = run_as(account, library.save_upload, NOTE, "text/markdown", iter([CONTENT]))
    return {"upload_id": record["id"]}


FACTORIES = {
    "routes.library:PATCH:/folders/{folder_id}": Factory(
        "library-folder", {"name": EDITED}, fragment = EDITED
    ),
    "routes.library:DELETE:/folders/{folder_id}": Factory("library-folder", fragment = '"ok":true'),
    "routes.library:GET:/uploads/{upload_id}/file": Factory(
        "library-upload", fragment = CONTENT.decode().strip()
    ),
    "routes.library:PUT:/uploads/{upload_id}/text": Factory(
        "library-upload", {"text": EDITED}, fragment = '"ok":true'
    ),
}

SKIPPED: dict[str, str] = {}
