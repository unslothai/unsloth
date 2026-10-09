# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``generate_image``: images saved to the Unsloth Studio gallery and returned inline when small."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from studio_mcp import loading
from studio_mcp.caller import Caller, current_caller
from studio_mcp.forward import forward
from studio_mcp.inputs import ImageInput, data_url, resolve_image, sniff_image
from studio_mcp.media import (
    INLINE_CAP,
    image_content,
    image_gallery_path,
    media_result,
    public_url,
    resource_link,
)
from studio_mcp.outputs import ImageItem, ImageResult
from studio_mcp.tools import WRITES, integer, number, opt_text, present, route_json

LOAD_IMAGE_HINT = "Load an image model with load_model(kind='image') first."
NOT_LOADED = "No diffusion model is loaded"
# The route takes 128 MiB of base64 across every image in one request.
MAX_REQUEST_IMAGE_BYTES = 128 * 1024 * 1024 * 3 // 4


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
        data = await _fetch(caller, image_gallery_path(item.id))
        if data is not None and len(data) <= INLINE_CAP:
            return [image_content(data, "image/png")]
    thumb = await _fetch(caller, image_gallery_path(item.id, thumb = True))
    contents: list[Any] = []
    if thumb is not None and len(thumb) <= INLINE_CAP:
        # Studio sends the original PNG when it cannot make the thumbnail.
        contents.append(image_content(thumb, sniff_image(thumb) or "image/webp"))
    contents.append(resource_link(item.url, f"{item.id}.png", "image/png"))
    return contents


async def run_generation(
    caller: Caller, ctx: Optional[Context], body: dict[str, Any]
) -> ToolResult:
    generate = route_json(
        "POST",
        "/api/inference/images/generate",
        caller = caller,
        json_body = body,
        hint_if = (NOT_LOADED, {409: LOAD_IMAGE_HINT}),
    )
    payload = await loading.with_progress(ctx, generate, _progress_poll(caller))
    rows = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ToolError("Unsloth Studio returned no images")
    items, contents = [], []
    for row in rows:
        if not isinstance(row, dict) or not opt_text(row.get("id")):
            continue
        item = ImageItem.from_route(row, url = public_url(caller, image_gallery_path(row["id"])))
        items.append(item)
        contents.extend(await _image_contents(caller, item))
    return media_result(contents, ImageResult(images = items))


async def _image_fields(
    caller: Caller,
    init_image: Optional[ImageInput],
    mask_image: Optional[ImageInput],
    reference_images: Optional[list[ImageInput]],
) -> dict[str, Any]:
    if mask_image is not None and init_image is None:
        raise ToolError("mask_image needs init_image: the mask marks what to repaint in it.")
    fields: dict[str, Any] = {}
    total = 0
    for key, image in (("init_image", init_image), ("mask_image", mask_image)):
        if image is not None:
            data, mime = await resolve_image(caller, image)
            total += len(data)
            fields[key] = data_url(data, mime)
    references = []
    for image in reference_images or []:
        data, mime = await resolve_image(caller, image)
        total += len(data)
        references.append(data_url(data, mime))
    if references:
        fields["reference_images"] = references
    if total > MAX_REQUEST_IMAGE_BYTES:
        raise ToolError("The images together are larger than the 96 MiB one request takes")
    return fields


async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    init_image: Optional[ImageInput] = None,
    mask_image: Optional[ImageInput] = None,
    reference_images: Annotated[Optional[list[ImageInput]], Field(max_length = 9)] = None,
    workflow: Optional[Literal["edit", "reference", "outpaint"]] = None,
    strength: Optional[float] = None,
    upscale: Optional[float] = None,
    allow_oversized: bool = False,
    ctx: Optional[Context] = None,
) -> ToolResult:
    """Generate or edit images with the image model loaded in Unsloth Studio (load_model(kind="image") first). Each image is saved to the Unsloth Studio Images gallery and returned with its id and URL; images up to 1.5 MiB come back inline, larger ones as a 1024 px preview plus a link. Width and height are pixels, multiples of 16; left out, the size is 1024x1024 with 9 steps and guidance 0, which suits turbo models (raise steps and guidance for others). With ``init_image`` it is img2img (``strength`` above 0 up to 1 sets how much is redrawn; left out, the model's default); add ``mask_image`` to inpaint (white is repainted); ``upscale`` 1 to 4 enlarges ``init_image`` and lightly redraws it, still guided by the prompt; ``workflow`` "edit" follows the prompt as an instruction over init_image and ``reference_images``, "reference" draws a new image guided by them, "outpaint" fills a padded init_image under its mask. A refusal on memory grounds can be overridden with ``allow_oversized``. Progress is reported while it runs."""
    if upscale is not None and init_image is None:
        raise ToolError("upscale needs init_image: it enlarges that image.")
    caller = current_caller()
    body: dict[str, Any] = {"prompt": prompt}
    body.update(await _image_fields(caller, init_image, mask_image, reference_images))
    body.update(
        present(
            negative_prompt = negative_prompt,
            width = width,
            height = height,
            steps = steps,
            guidance = guidance,
            seed = seed,
            batch_size = batch_size,
            workflow = workflow,
            strength = strength,
            upscale = upscale,
        )
    )
    if allow_oversized:
        body["allow_oversized"] = True
    return await run_generation(caller, ctx, body)


TOOLS = ((generate_image, WRITES, ImageResult.model_json_schema()),)
