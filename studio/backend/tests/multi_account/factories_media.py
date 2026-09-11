# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

# ASCII on purpose: it has to survive JSON escaping, a PNG text chunk and a JPEG comment.
SENTINEL = "media-sentinel-prompt"

VIDEO_ID = "media-video-clip"
SANDBOX_SESSION = "media-sandbox-session"
SANDBOX_FILE = "notes.txt"
SEARCH_IMAGE_ID = "a1b2c3d4e5f6"
MODEL_ID = "media-local-model-Q4_K_M"

IMAGE_LINK_QUERY: dict[str, str] = {}
VIDEO_LINK_QUERY: dict[str, str] = {}


def _image_meta() -> dict:
    return {
        "prompt": SENTINEL,
        "negative_prompt": None,
        "width": 8,
        "height": 8,
        "steps": 1,
        "guidance": 1.0,
        "seed": 7,
        "model": "media/none",
        # GalleryImage dates an image with an epoch float, where audio and video use ISO strings.
        "created_at": 1767225600.0,
    }


@seeder("media-image")
def seed_image(account) -> dict[str, str]:
    from PIL import Image

    from core.inference import image_gallery
    from routes.inference import _sign_image_id
    from utils.account_context import run_as

    image = Image.new("RGB", (8, 8), (10, 20, 30))
    record = run_as(account, image_gallery.save, image, _image_meta())
    IMAGE_LINK_QUERY["token"] = run_as(account, _sign_image_id, record["id"])
    return {"image_id": record["id"]}


def _wav_bytes() -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 64)
    return buf.getvalue()


@seeder("media-audio")
def seed_audio(account) -> dict[str, str]:
    from core.inference import audio_gallery
    from utils.account_context import run_as

    meta = {
        "prompt": SENTINEL,
        "model": "media/none",
        "audio_type": "speech",
        "sample_rate": 8000,
        "duration_s": 0.008,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    record = run_as(account, audio_gallery.save, _wav_bytes(), meta)
    return {"audio_id": record["id"]}


def _mp4_bytes() -> bytes:
    import io

    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format = "mp4") as container:
        stream = container.add_stream("libx264", rate = 8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        stream.options = {"crf": "40", "preset": "ultrafast"}
        for index in range(4):
            frame = av.VideoFrame(64, 64, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes([16 + index * 40]) * plane.buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


@seeder("media-video")
def seed_video(account) -> dict[str, str]:
    from core.inference import video_gallery
    from routes.video import _sign_video_id
    from utils.account_context import run_as

    meta = {
        "prompt": SENTINEL,
        "negative_prompt": None,
        "width": 64,
        "height": 64,
        "num_frames": 4,
        "fps": 8,
        "duration_s": 0.5,
        "steps": 1,
        "guidance": 1.0,
        "seed": 7,
        "model": "media/none",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    run_as(account, video_gallery.save, _mp4_bytes(), meta, VIDEO_ID)
    VIDEO_LINK_QUERY["token"] = run_as(account, _sign_video_id, VIDEO_ID)
    return {"video_id": VIDEO_ID}


@seeder("media-search-image")
def seed_search_image(account) -> dict[str, str]:
    import io

    from PIL import Image

    from core.inference import search_images
    from utils.account_context import run_as

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 100, 50)).save(
        buf, format = "JPEG", comment = SENTINEL.encode("ascii")
    )
    directory = run_as(account, search_images._cache_dir)
    (directory / f"{SEARCH_IMAGE_ID}.jpg").write_bytes(buf.getvalue())
    return {"image_id": SEARCH_IMAGE_ID}


@seeder("media-sandbox")
def seed_sandbox(account) -> dict[str, str]:
    import os

    from core.inference.tools import get_sandbox_workdir
    from utils.account_context import run_as

    workdir = run_as(account, get_sandbox_workdir, SANDBOX_SESSION)
    with open(os.path.join(workdir, SANDBOX_FILE), "w", encoding = "utf-8") as handle:
        handle.write(SENTINEL)
    return {"session_id": SANDBOX_SESSION, "filename": SANDBOX_FILE}


@seeder("media-monitor")
def seed_monitor(account) -> dict[str, str]:
    from core.inference.api_monitor import api_monitor
    from utils.account_context import run_as

    entry_id = run_as(
        account,
        api_monitor.start,
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "media/none",
        prompt = SENTINEL,
        subject = account.username,
    )
    return {"entry_id": entry_id}


@seeder("media-model")
def seed_model(account) -> dict[str, str]:
    import struct

    from core.inference import local_model_resolver
    from routes import inference
    from storage import studio_db
    from utils.account_context import run_as
    from utils.paths import workspace_root

    folder = run_as(account, workspace_root) / "local-models"
    folder.mkdir(parents = True, exist_ok = True)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0)
    (folder / f"{MODEL_ID}.gguf").write_bytes(header + b"\x00" * 256)
    run_as(account, studio_db.add_scan_folder_with_status, str(folder))
    # Drop the catalog's own 30s memo, so the request rescans instead of reusing a stale root.
    inference._CATALOG_CACHE.update(at = 0.0, models = [])
    inference._managed_catalogs.clear()
    inference._SERVABLE_SCAN_CACHE["entry"] = None
    local_model_resolver.invalidate_index()
    return {"model_id": MODEL_ID}


_LINK_IS_THE_CREDENTIAL = (
    "the signed link is the credential rather than the bearer, so every holder of it is served "
    "while the minting account is active"
)

FACTORIES = {
    "routes.inference:GET:/images/gallery/{image_id}/file": Factory(
        "media-image", fragment = SENTINEL
    ),
    "routes.inference:GET:/images/gallery/{image_id}/file-signed": Factory(
        "media-image",
        fragment = SENTINEL,
        query = IMAGE_LINK_QUERY,
        owner = (200,),
        wrong = (200,),
        unauthenticated = (200,),
        reason = _LINK_IS_THE_CREDENTIAL,
    ),
    "routes.inference:PATCH:/images/gallery/{image_id}": Factory(
        "media-image", {"archived": True}, fragment = SENTINEL
    ),
    "routes.inference:DELETE:/images/gallery/{image_id}": Factory("media-image"),
    "routes.inference:GET:/audio/gallery/{audio_id}/file": Factory("media-audio"),
    "routes.inference:PATCH:/audio/gallery/{audio_id}": Factory(
        "media-audio", {"archived": True}, fragment = SENTINEL
    ),
    "routes.inference:DELETE:/audio/gallery/{audio_id}": Factory("media-audio"),
    "routes.video:GET:/video/gallery/{video_id}/file": Factory("media-video"),
    "routes.video:GET:/video/gallery/{video_id}/file-signed": Factory(
        "media-video",
        query = VIDEO_LINK_QUERY,
        owner = (200,),
        wrong = (200,),
        unauthenticated = (200,),
        reason = _LINK_IS_THE_CREDENTIAL,
    ),
    "routes.video:GET:/video/gallery/{video_id}/signed-url": Factory(
        "media-video", fragment = "file-signed"
    ),
    "routes.video:GET:/video/gallery/{video_id}/export": Factory("media-video"),
    "routes.video:PATCH:/video/gallery/{video_id}": Factory(
        "media-video", {"archived": True}, fragment = SENTINEL
    ),
    "routes.video:DELETE:/video/gallery/{video_id}": Factory("media-video"),
    "routes.video:GET:/videos/{video_id}": Factory("media-video", fragment = SENTINEL),
    "routes.video:GET:/videos/{video_id}/content": Factory("media-video"),
    "routes.video:DELETE:/videos/{video_id}": Factory("media-video", fragment = VIDEO_ID),
    "routes.inference:GET:/search-images/{image_id}": Factory(
        "media-search-image", fragment = SENTINEL
    ),
    "routes.inference:GET:/monitor/{entry_id}": Factory("media-monitor", fragment = SENTINEL),
    "routes.inference:GET:/models/{model_id:path}": Factory("media-model", fragment = MODEL_ID),
    "routes.inference:GET:/sandbox/{session_id}": Factory(
        "media-sandbox",
        fragment = SANDBOX_FILE,
        absent = SANDBOX_FILE,
        owner = (200,),
        wrong = (200,),
        reason = "a session id resolves inside the caller's own sandbox root, so another account "
        "is listed its own empty directory rather than refused",
    ),
    "routes.inference:GET:/sandbox/{session_id}/{filename:path}": Factory(
        "media-sandbox", fragment = SENTINEL
    ),
    "routes.inference:HEAD:/sandbox/{session_id}/{filename:path}": Factory("media-sandbox"),
}

SKIPPED = {
    "routes.inference:POST:/sandbox/{session_id}/reveal": (
        "opens the backend host's file manager, so a success spawns xdg-open on a machine with "
        "no desktop session and the only in-process outcome is the 500 that failure maps to"
    ),
}
