# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset
from PIL import Image

from utils.datasets import format_conversion


def test_rows_without_an_image_are_left_out_of_vision_training():
    ds = Dataset.from_dict(
        {
            "image": [None, Image.new("RGB", (4, 4)), None],
            "text": ["no picture", "a black square", "also no picture"],
        }
    )

    out = format_conversion.convert_to_vlm_format(ds, instruction = "Describe.")

    images = [
        part["image"]
        for sample in out
        for message in sample["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    ]
    assert len(out) == 1
    assert out[0]["messages"][1]["content"][0]["text"] == "a black square"
    assert all(isinstance(image, Image.Image) for image in images)


def test_rows_without_an_image_do_not_count_as_broken_urls(tmp_path):
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory = str(tmp_path))
    )
    threading.Thread(target = server.serve_forever, daemon = True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/a.png"
    try:
        ds = Dataset.from_dict(
            {
                "image": [url, None, None, url, None],
                "text": ["first", "none", "none", "second", "none"],
            }
        )
        out = format_conversion.convert_to_vlm_format(ds, instruction = "Describe.")
    finally:
        server.shutdown()

    assert [sample["messages"][1]["content"][0]["text"] for sample in out] == ["first", "second"]
