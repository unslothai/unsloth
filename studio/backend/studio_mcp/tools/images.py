# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``generate_image``: images saved to the Studio gallery and returned inline when small."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from studio_mcp import loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward
from studio_mcp.media import INLINE_CAP, image_content, media_result, public_url, resource_link
from studio_mcp.outputs import ImageItem, ImageResult
from studio_mcp.tools import WRITES, integer, number, route_json, text

LOAD_IMAGE_HINT = "Load an image model with load_model(kind='image') first."
NOT_LOADED = "No diffusion model is loaded"
THUMB_SIDE = 1024


def _gallery_path(image_id: str, thumb: bool = False) -> str:
    path = f"/api/inference/images/gallery/{quote(image_id, safe = '')}/file"
    return f"{path}?thumb={THUMB_SIDE}" if thumb else path


def _progress_poll(caller: Caller):
    async def poll():
        progress = await route_json("GET", "/api/inference/images/generate-progress", caller = caller)
        if not isinstance(progress, dict) or not progress.get("active"):
            return None
        fraction = number(progress.get("fraction")) or 0.0
        step, total = integer(progress.get("step")), integer(progress.get("total_steps"))
        return fraction, f"Step {step} of {total}" if step is not None and total else "Generating"

    return poll


async def _fetch(caller: Caller, path: str) -> Optional[bytes]:
    response = await forward(caller, "GET", path)
    return response.content if response.status_code == 200 else None


async def _image_contents(caller: Caller, item: ImageItem) -> list[Any]:
    """The PNG inline when it must fit the cap, else a WebP thumbnail plus a link; a large original is never fetched."""
    if item.width and item.height and item.width * item.height * 3 <= INLINE_CAP:
        data = await _fetch(caller, _gallery_path(item.id))
        if data is not None and len(data) <= INLINE_CAP:
            return [image_content(data, "image/png")]
    thumb = await _fetch(caller, _gallery_path(item.id, thumb = True))
    contents: list[Any] = []
    if thumb is not None and len(thumb) <= INLINE_CAP:
        contents.append(image_content(thumb, "image/webp"))
    contents.append(resource_link(item.url, f"{item.id}.png", "image/png"))
    return contents


async def run_generation(
    caller: Caller, ctx: Optional[Context], body: dict[str, Any]
) -> ToolResult:
    async def generate():
        response = await forward(caller, "POST", "/api/inference/images/generate", json_body = body)
        hints = (
            {409: LOAD_IMAGE_HINT}
            if response.status_code == 409 and NOT_LOADED in response.text
            else None
        )
        return raise_for_route(response, hints = hints)

    payload = await loading.with_progress(ctx, generate(), _progress_poll(caller))
    rows = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ToolError("Studio returned no images")
    items, contents = [], []
    for row in rows:
        if not isinstance(row, dict) or not text(row.get("id")):
            continue
        item = ImageItem(
            id = row["id"],
            url = public_url(caller, _gallery_path(row["id"])),
            width = integer(row.get("width")),
            height = integer(row.get("height")),
            seed = integer(row.get("seed")),
        )
        items.append(item)
        contents.extend(await _image_contents(caller, item))
    return media_result(contents, ImageResult(images = items))


async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> ToolResult:
    """Generate images with the image model loaded in Studio (load_model(kind="image") first). Each image is saved to the Studio Images gallery and returned with its id and URL; images up to 1.5 MiB come back inline, larger ones as a 1024 px preview plus a link. Width and height are pixels, multiples of 16. Progress is reported while it runs."""
    body: dict[str, Any] = {"prompt": prompt}
    for key, value in (
        ("negative_prompt", negative_prompt),
        ("width", width),
        ("height", height),
        ("steps", steps),
        ("guidance", guidance),
        ("seed", seed),
        ("batch_size", batch_size),
    ):
        if value is not None:
            body[key] = value
    return await run_generation(current_caller(), ctx, body)


def register_images(mcp: FastMCP) -> None:
    mcp.tool(generate_image, annotations = WRITES, output_schema = ImageResult.model_json_schema())
