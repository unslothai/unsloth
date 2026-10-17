# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import re
from email.parser import BytesParser
from email.policy import default

import pytest

from .mcp_harness import JPEG, PNG, PNG_URL, call_to, data_url, run_tool

DATASET = "/api/train/diffusion/dataset"
SAVED = {
    "name": "my-photos",
    "path": "/srv/unsloth/assets/datasets/my-photos",
    "image_count": 2,
    "clip_count": 0,
    "caption_count": 1,
    "uploaded": 3,
}
PAYLOADS = {("POST", DATASET): SAVED}
JPEG_URL = {"data_url": data_url(JPEG, "image/jpeg")}


def _parts(studio) -> list[tuple[str, str, bytes]]:
    # Every multipart part in order; the harness form() keeps only the last per field.
    _m, _p, headers, body = call_to(studio, DATASET)
    message = BytesParser(policy = default).parsebytes(
        b"Content-Type: " + headers["content-type"].encode() + b"\r\n\r\n" + body
    )
    return [
        (
            part.get_param("name", header = "content-disposition"),
            part.get_filename(),
            part.get_payload(decode = True),
        )
        for part in message.iter_parts()
    ]


def _upload(monkeypatch, **args):
    args = {"action": "upload_images", "name": "my-photos", **args}
    return run_tool(monkeypatch, PAYLOADS, "datasets", args)


def test_images_and_captions_go_up_as_files_with_fresh_stems(monkeypatch):
    result, studio = _upload(monkeypatch, images = [PNG_URL, JPEG_URL], captions = ["a fox", ""])
    assert result["structuredContent"]["image_dataset"] == {
        "name": "my-photos",
        "image_count": 2,
        "caption_count": 1,
        "uploaded": 3,
    }
    assert "/srv" not in json.dumps(result)
    parts = _parts(studio)
    fields = {name: value for name, filename, value in parts if filename is None}
    assert fields == {"name": b"my-photos", "create_only": b"false"}
    files = [(filename, value) for name, filename, value in parts if name == "files"]
    stem = files[0][0].removesuffix(".png")
    assert re.fullmatch(r"mcp-[0-9a-f]{8}-000", stem)
    assert files == [
        (f"{stem}.png", PNG),
        (f"{stem}.txt", b"a fox"),
        (f"{stem[:-3]}001.jpg", JPEG),
    ]


def test_a_second_upload_never_reuses_the_first_ones_names(monkeypatch):
    names = []
    for _ in range(2):
        _result, studio = _upload(monkeypatch, images = [PNG_URL])
        names += [filename for _n, filename, _v in _parts(studio) if filename]
    assert len(set(names)) == 2


@pytest.mark.parametrize(
    "args,message",
    [
        ({"name": ""}, "needs name"),
        ({"images": []}, "needs name"),
        ({"captions": ["a", "b"]}, "captions has 2 entries for 1 images"),
        ({"captions": ["x" * 2001]}, "longer than 2000"),
        ({"images": [{"data_url": "data:image/gif;base64,R0lGODlh"}]}, None),
        ({"images": [{"data_url": data_url(b"GIF89a" + b"\x00" * 8, "image/png")}]}, "Images must"),
        ({"images": [PNG_URL] * 65}, None),
    ],
)
def test_a_refused_image_upload_never_reaches_a_route(monkeypatch, args, message):
    args = {"action": "upload_images", "name": "my-photos", "images": [PNG_URL], **args}
    result, studio = run_tool(monkeypatch, {}, "datasets", args, fallback = {})
    assert result["isError"] is True
    assert message is None or message in result["content"][0]["text"]
    assert studio.state.calls == []
