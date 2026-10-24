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
    if ctx.params.get("num_frames"):
        # The route takes seconds or a frame count, never both.
        body["num_frames"] = int(ctx.params["num_frames"])
    elif ctx.params.get("seconds"):
        body["seconds"] = str(ctx.params["seconds"])
    for key, cast in (("steps", int), ("guidance", float), ("fps", int)):
        if ctx.params.get(key) is not None:
            body[key] = cast(ctx.params[key])
    references = [r for r in ctx.inputs.get("reference") or [] if isinstance(r, dict)]
    if references and (ctx.inputs.get("first_frame") or ctx.inputs.get("last_frame")):
        raise FlowNodeError(
            "Connect either keyframes (first or last frame) or reference images: the video model takes one or the other"
        )
    if references:
        body["reference_images"] = [await data_url(ctx, r) for r in references[:9]]
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
        # Named, so a Stop after this render finished never cancels a newer one started elsewhere.
        await ctx.call("POST", "/api/inference/video/generate/cancel", json = {"video_id": video_id})

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
    which = str(ctx.params.get("frame") or "last")
    if kind != "video" or not video_id:
        # An uploaded or edited clip is a flow asset, not a gallery video: decode it here.
        from core.flows.nodes.media_utils import video_frame

        data, _ = await ctx.media_bytes(video)
        png = await asyncio.to_thread(
            video_frame, data, which, float(ctx.params.get("time_s") or 0)
        )
        return {"image": await ctx.save_asset(png, "image/png", kind = "image")}
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


MAX_CONCAT_CLIPS = 32
MAX_CONCAT_FRAMES = 20_000


def concat_clips(clips: list[bytes]) -> bytes:
    """One H.264 MP4 of the clips back to back, at the first clip's size and frame rate (audio is dropped)."""
    import io

    import av

    out = io.BytesIO()
    frames = 0
    with av.open(out, mode = "w", format = "mp4") as dst:
        stream = None
        for data in clips:
            with av.open(io.BytesIO(data)) as src:
                if not src.streams.video:
                    raise FlowNodeError("One of the clips has no video track")
                source = src.streams.video[0]
                if stream is None:
                    rate = source.average_rate or 24
                    stream = dst.add_stream("libx264", rate = rate)
                    # yuv420p needs even sides.
                    stream.width = source.codec_context.width // 2 * 2
                    stream.height = source.codec_context.height // 2 * 2
                    stream.pix_fmt = "yuv420p"
                for frame in src.decode(source):
                    frames += 1
                    if frames > MAX_CONCAT_FRAMES:
                        raise FlowNodeError(
                            f"The joined video would pass {MAX_CONCAT_FRAMES} frames"
                        )
                    frame = frame.reformat(
                        width = stream.width, height = stream.height, format = "yuv420p"
                    )
                    frame.pts = None
                    for packet in stream.encode(frame):
                        dst.mux(packet)
        if stream is not None:
            for packet in stream.encode():
                dst.mux(packet)
    if not frames:
        raise FlowNodeError("The clips had no frames")
    return out.getvalue()


@handler("video.concat")
async def concat(ctx: NodeContext) -> dict:
    videos = [v for v in ctx.inputs.get("videos") or [] if isinstance(v, dict)]
    if not videos:
        raise FlowNodeError("No videos to join")
    if len(videos) > MAX_CONCAT_CLIPS:
        raise FlowNodeError(f"Join at most {MAX_CONCAT_CLIPS} videos at once")
    clips = [(await ctx.media_bytes(v))[0] for v in videos]
    mp4 = await asyncio.to_thread(concat_clips, clips)
    return {"video": await ctx.save_asset(mp4, "video/mp4", kind = "video")}
