# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video nodes: the OpenAI-style video job API (one render at a time), frames and playlists."""

from __future__ import annotations

import asyncio
import time
from urllib.parse import quote

from core.flows.nodes._common import data_url, gallery_media
from core.flows.nodes.image import image_prompt
from core.flows.registry import FlowNodeError, NodeContext, handler

POLL_S = 2.0
BUSY_WAIT_S = 30 * 60


def _video_path(video_id: str, suffix: str = "") -> str:
    return f"/v1/videos/{quote(video_id, safe = '')}{suffix}"


@handler("video.generate")
async def generate(ctx: NodeContext) -> dict:
    await ctx.ensure_model("video", ctx.params.get("model"))
    prompt = image_prompt(ctx)
    if not prompt:
        raise FlowNodeError("The video prompt is empty")
    body: dict = {"prompt": prompt, "seed": ctx.seed}
    if ctx.params.get("negative_prompt"):
        body["negative_prompt"] = str(ctx.params["negative_prompt"])
    if ctx.params.get("size"):
        body["size"] = str(ctx.params["size"])
    if ctx.params.get("seconds"):
        body["seconds"] = str(ctx.params["seconds"])
    if ctx.inputs.get("first_frame"):
        body["input_reference"] = await data_url(ctx, ctx.inputs["first_frame"])
    if ctx.inputs.get("last_frame"):
        body["last_frame"] = await data_url(ctx, ctx.inputs["last_frame"])
    started = time.monotonic()
    while True:
        try:
            job = await ctx.call("POST", "/v1/videos", json = body)
            break
        except FlowNodeError as exc:
            # One video renders at a time: wait for the other one rather than failing.
            if exc.status != 409 or time.monotonic() - started > BUSY_WAIT_S:
                raise
            ctx.set_status("waiting", "Another video is rendering")
            await asyncio.sleep(10)
            ctx.check_cancelled()
            ctx.set_status("running")
    video_id = job.get("id") if isinstance(job, dict) else None
    if not video_id:
        raise FlowNodeError("Unsloth Studio did not start the video")

    async def cancel() -> None:
        await ctx.call("POST", "/api/inference/video/generate/cancel")

    ctx.on_cancel(cancel)
    while True:
        await asyncio.sleep(POLL_S)
        job = await ctx.call("GET", _video_path(video_id))
        status = job.get("status") if isinstance(job, dict) else None
        progress = job.get("progress") if isinstance(job, dict) else None
        if isinstance(progress, (int, float)):
            ctx.progress(float(progress) / 100.0, phase = status)
        if status == "completed":
            return {
                "video": gallery_media(
                    "video", {"id": video_id}, mime = "video/mp4", prompt = prompt, seed = job.get("seed")
                )
            }
        if status == "failed":
            error = job.get("error") if isinstance(job.get("error"), dict) else {}
            if error.get("code") == "cancelled":
                ctx.check_cancelled()
            raise FlowNodeError(error.get("message") or "The video failed")


@handler("video.frame")
async def frame(ctx: NodeContext) -> dict:
    video = ctx.inputs["video"]
    ref = str(video.get("ref", ""))
    kind, _, video_id = ref.partition(":")
    if kind != "video" or not video_id:
        raise FlowNodeError("Frames come from videos made in Unsloth Studio")
    which = str(ctx.params.get("frame") or "last")
    at = str(float(ctx.params.get("time_s") or 0)) if which == "time" else which
    data = await ctx.call(
        "GET", f"/api/inference/video/gallery/{quote(video_id, safe = '')}/frame", params = {"at": at}
    )
    if not isinstance(data, (bytes, bytearray)):
        raise FlowNodeError("Could not read the frame")
    return {"image": await ctx.save_asset(bytes(data), "image/png", kind = "image")}


@handler("video.playlist")
async def playlist(ctx: NodeContext) -> dict:
    return {"videos": [v for v in ctx.inputs.get("videos") or [] if isinstance(v, dict)]}
