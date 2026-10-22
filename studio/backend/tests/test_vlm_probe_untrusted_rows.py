# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VLM image-column probe must not reach private addresses from untrusted row URLs."""

import socket
import sys
import urllib.request
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.format_detection import detect_vlm_dataset_structure  # noqa: E402


@pytest.fixture
def dials(monkeypatch):
    seen = []

    def no_dial(address, *args, **kwargs):
        seen.append(address)
        raise OSError("dial refused in test")

    def no_urlopen(req, *args, **kwargs):
        seen.append(getattr(req, "full_url", req))
        raise OSError("urlopen refused in test")

    monkeypatch.setattr(socket, "create_connection", no_dial)
    monkeypatch.setattr(urllib.request, "urlopen", no_urlopen)
    return seen


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:9/a.png", "http://169.254.169.254/latest/a.png", "http://[::1]/a.png"],
)
def test_private_image_urls_are_never_dialled(dials, url):
    # Two scored-below-75 URL columns force the probe pass.
    row = {"image": url, "photo": "http://10.0.0.1/b.png", "caption": "a cat on a mat"}

    detect_vlm_dataset_structure([row])

    assert dials == []


def test_public_image_urls_go_through_the_guarded_fetch(monkeypatch):
    from core.inference import tools

    calls = []

    def fake_fetch(url, **kwargs):
        calls.append((url, kwargs.get("method")))
        return (None, b"", "") if "good" in url else ("Failed to fetch URL: HTTP 404", "", "")

    monkeypatch.setattr(tools, "_fetch_url_raw", fake_fetch)
    row = {
        "image": "https://example.com/bad.png",
        "image_filename": "https://example.com/good.png",
        "caption": "a cat",
    }

    assert detect_vlm_dataset_structure([row])["image_column"] == "image_filename"
    assert calls == [
        ("https://example.com/bad.png", "HEAD"),
        ("https://example.com/good.png", "HEAD"),
    ]
