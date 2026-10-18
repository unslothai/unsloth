# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``datasets``: the training datasets Unsloth Studio has, and getting more."""

from __future__ import annotations

import dataclasses
import json
import secrets
from pathlib import PurePath
from typing import Annotated, Any, Literal, Optional

from fastmcp.exceptions import ToolError
from pydantic import Field

from studio_mcp.caller import Caller, current_caller
from studio_mcp.gate import MAX_REQUEST_BYTES
from studio_mcp.inputs import ImageInput, decode_base64, resolve_image
from studio_mcp.outputs import (
    CachedDataset,
    DatasetDownload,
    DatasetFormat,
    DatasetsResult,
    DatasetUpload,
    ImageDatasetUpload,
    LocalDataset,
)
from studio_mcp.tools import WRITES, as_dict, leaf_name, opt_text, route_json

HUB_DATASETS = "/api/hub/datasets"
# The training upload route's own list; anything else it answers with a 400.
UPLOAD_EXTS = (".csv", ".json", ".jsonl", ".parquet")
IMAGE_EXTS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
# The caption route's limit, which sidecars sent with an upload would skip.
MAX_CAPTION_CHARS = 2000


def _ref(path: Any) -> Optional[str]:
    from hub.utils.host_paths import cache_reference
    return cache_reference(path) if isinstance(path, str) and path else None


def _upload_file(
    filename: Optional[str], data_base64: Optional[str], rows: Optional[list[dict[str, Any]]]
) -> tuple[str, bytes]:
    """The upload's file name and bytes, refused before any call the route would refuse."""
    if (data_base64 is None) == (rows is None):
        raise ToolError("upload needs exactly one of data_base64 (with filename) or rows.")
    if rows is not None:
        name = f"{PurePath(leaf_name(filename or '').strip()).stem or 'rows'}.jsonl"
        return name, "\n".join(json.dumps(row, ensure_ascii = False) for row in rows).encode()
    name = leaf_name(filename or "").strip()
    if not name:
        raise ToolError("upload with data_base64 needs filename, e.g. chats.jsonl.")
    ext = PurePath(name).suffix.lower()
    if ext not in UPLOAD_EXTS:
        raise ToolError(f"Unsupported file type: {ext or name}. Allowed: {', '.join(UPLOAD_EXTS)}")
    data = decode_base64(data_base64, MAX_REQUEST_BYTES, "data_base64")
    if not data:
        raise ToolError("data_base64 is empty.")
    return name, data


async def datasets(
    action: Literal["list", "check_format", "download", "status", "upload", "upload_images"],
    name: Optional[str] = None,
    repo_id: Optional[str] = None,
    is_vlm: bool = False,
    subset: Optional[str] = None,
    train_split: str = "train",
    hf_token: Optional[str] = None,
    filename: Annotated[
        Optional[str],
        Field(description = "upload: the file's name; its extension picks the format."),
    ] = None,
    data_base64: Annotated[
        Optional[str],
        Field(description = "upload: the file's bytes, base64; at most about 3 MiB."),
    ] = None,
    rows: Annotated[
        Optional[list[dict[str, Any]]],
        Field(min_length = 1, description = "upload: examples as JSON objects, saved as JSONL."),
    ] = None,
    images: Annotated[
        Optional[list[ImageInput]],
        Field(max_length = 64, description = "upload_images: PNG, JPEG or WebP images to add."),
    ] = None,
    captions: Annotated[
        Optional[list[str]],
        Field(description = "upload_images: one caption per image, in order; empty for none."),
    ] = None,
) -> DatasetsResult:
    """Training datasets.
    "list": datasets uploaded to Unsloth Studio and Hugging Face datasets already downloaded.
    "check_format" (``name``: a Hub repo id or a listed dataset's id): the detected format, columns and a suggested column mapping.
    "download" (``repo_id``): start downloading a Hub dataset; "status" (``repo_id``): how that download is going.
    "upload": save a .csv, .json, .jsonl or .parquet file sent as ``data_base64`` with its ``filename``, or ``rows`` as JSONL.
    Put a listed or uploaded dataset's ``ref`` in start_training's config.local_datasets.
    "upload_images" (``name``, ``images``, optional ``captions``): add captioned images to the image dataset ``name``, creating it if needed.
    Pass that ``name`` as config.data_dir to start_training kind "diffusion".
    A ref lasts until Unsloth Studio restarts; after a restart, list or upload again for a new one.
    ``hf_token`` is for gated datasets."""
    caller = current_caller()
    if hf_token:
        caller = dataclasses.replace(caller, hf_token = hf_token)
    if action == "list":
        local = await route_json("GET", f"{HUB_DATASETS}/local", caller = caller)
        cached = await route_json("GET", f"{HUB_DATASETS}/cached", caller = caller)
        return DatasetsResult(
            local = [
                LocalDataset.from_route(
                    row,
                    label = opt_text(row.get("label")) or row["id"],
                    source = "local",
                    ref = _ref(row.get("path")),
                )
                for row in as_dict(local).get("datasets") or []
                if isinstance(row, dict) and opt_text(row.get("id"))
            ],
            cached = [
                CachedDataset.from_route(row)
                for row in as_dict(cached).get("cached") or []
                if isinstance(row, dict) and opt_text(row.get("repo_id"))
            ],
        )
    if action == "upload":
        upload_name, data = _upload_file(filename, data_base64, rows)
        saved = as_dict(
            await route_json(
                "POST",
                f"{HUB_DATASETS}/upload",
                caller = caller,
                files = {"file": (upload_name, data, "application/octet-stream")},
            )
        )
        stored = saved.get("stored_path")
        if not isinstance(stored, str) or not stored:
            raise ToolError("Unsloth Studio did not say where it saved the upload.")
        return DatasetsResult(
            uploaded = DatasetUpload(
                id = leaf_name(stored),
                filename = opt_text(saved.get("filename")) or upload_name,
                ref = _ref(stored),
            )
        )
    if action == "upload_images":
        return await _upload_images(caller, name, images or [], captions)
    if action == "check_format":
        if not name:
            raise ToolError(
                "check_format needs name: a Hugging Face dataset id or a listed dataset."
            )
        body = {"dataset_name": name, "is_vlm": is_vlm, "train_split": train_split}
        if subset:
            body["subset"] = subset
        found = await route_json(
            "POST", f"{HUB_DATASETS}/check-format", caller = caller, json_body = body, hub_header = True
        )
        found = as_dict(found)
        mapping = found.get("suggested_mapping")
        return DatasetsResult(
            format = DatasetFormat.from_route(
                found,
                suggested_mapping = {str(k): v for k, v in mapping.items() if isinstance(v, str)}
                if isinstance(mapping, dict)
                else None,
            )
        )
    if not repo_id:
        raise ToolError(f"{action} needs repo_id: a Hugging Face dataset id.")
    if action == "download":
        started = await route_json(
            "POST",
            f"{HUB_DATASETS}/download",
            caller = caller,
            json_body = {"repo_id": repo_id},
            hub_header = True,
        )
        return DatasetsResult(
            download = DatasetDownload(
                repo_id = repo_id, state = opt_text(as_dict(started).get("state")) or "queued"
            )
        )
    state = await route_json(
        "GET", f"{HUB_DATASETS}/download-status", caller = caller, params = {"repo_id": repo_id}
    )
    state = as_dict(state)
    return DatasetsResult(
        download = DatasetDownload.from_route(
            state, repo_id = repo_id, state = opt_text(state.get("state")) or "unknown"
        )
    )


async def _upload_images(
    caller: Caller, name: Optional[str], images: list[ImageInput], captions: Optional[list[str]]
) -> DatasetsResult:
    if not (name or "").strip() or not images:
        raise ToolError("upload_images needs name (the image dataset) and images.")
    if captions is not None and len(captions) != len(images):
        raise ToolError(f"captions has {len(captions)} entries for {len(images)} images.")
    if any(len(caption) > MAX_CAPTION_CHARS for caption in captions or []):
        raise ToolError(f"A caption is longer than {MAX_CAPTION_CHARS} characters.")
    from utils.upload_limits import get_upload_limit_bytes, get_upload_limit_label

    # Fresh stems each call: a same-named file in a later upload would replace the earlier one.
    batch, files, total = secrets.token_hex(4), [], 0
    for i, image in enumerate(images):
        data, mime = await resolve_image(caller, image)
        # The route caps the whole upload; stop before holding more than it takes.
        total += len(data)
        if total > get_upload_limit_bytes():
            raise ToolError(
                f"These images pass Unsloth Studio's {get_upload_limit_label()} upload limit by image {i + 1}. "
                "Send them in several upload_images calls with the same name; each one adds to the dataset."
            )
        stem = f"mcp-{batch}-{i:03d}"
        files.append(("files", (stem + IMAGE_EXTS[mime], data, mime)))
        if captions and captions[i].strip():
            files.append(("files", (f"{stem}.txt", captions[i].encode(), "text/plain")))
    saved = as_dict(
        await route_json(
            "POST",
            "/api/train/diffusion/dataset",
            caller = caller,
            data = {"name": name, "create_only": "false"},
            files = files,
        )
    )
    return DatasetsResult(
        image_dataset = ImageDatasetUpload.from_route(saved, name = opt_text(saved.get("name")) or name)
    )


TOOLS = ((datasets, WRITES),)
