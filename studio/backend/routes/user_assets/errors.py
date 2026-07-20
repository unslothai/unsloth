# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stable user-assets error envelopes."""

from __future__ import annotations

from typing import NoReturn

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from core.user_assets_validation import UserAssetValidationError
from storage.user_assets_db import UserAssetStorageError


class UserAssetsRoute(APIRoute):
    """Map validation failures to the safe error envelope."""

    def get_route_handler(self):
        route_handler = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await route_handler(request)
            except RequestValidationError as error:
                paths = []
                for item in error.errors():
                    loc = item.get("loc", ())
                    path = ".".join(str(part) for part in loc if part != "body")
                    if path and path not in paths:
                        paths.append(path)
                raise HTTPException(
                    status_code = 422,
                    detail = {
                        "code": "invalid_request",
                        "message": "Request validation failed",
                        **({"paths": paths} if paths else {}),
                    },
                ) from error

        return handler


def raise_not_found() -> NoReturn:
    raise HTTPException(
        status_code = 404,
        detail = {"code": "not_found", "message": "Resource not found"},
    )


def raise_validation(error: UserAssetValidationError) -> NoReturn:
    raise HTTPException(status_code = 422, detail = error.to_detail()) from error


def raise_storage(error: UserAssetStorageError) -> NoReturn:
    status_code = 410 if error.code == "id_retired" else 409
    detail = {"code": error.code, "message": error.message}
    if error.current_resource is not None:
        detail["currentRevision"] = error.current_revision
        detail["current"] = error.current_resource
    raise HTTPException(status_code = status_code, detail = detail) from error


def ensure_path_id(body_id: str | None, path_id: str, field: str) -> None:
    if body_id is not None and body_id != path_id:
        raise HTTPException(
            status_code = 422,
            detail = {
                "code": "path_id_mismatch",
                "message": f"{field} does not match the request path",
            },
        )
