# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VLM URL probe's downloads are reused by the conversion; a failed probe download is still retried."""

import sys
import threading
from collections import Counter
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets import format_conversion  # noqa: E402
from utils.datasets.format_conversion import convert_to_vlm_format  # noqa: E402


@pytest.fixture(autouse = True)
def no_llm_warning(monkeypatch):
    from utils.datasets import llm_assist
    def refuse(*args, **kwargs):
        raise RuntimeError("no LLM in tests")

    monkeypatch.setattr(llm_assist, "llm_generate_dataset_warning", refuse)


@pytest.fixture
def fetches(monkeypatch):
    calls = Counter()
    lock = threading.Lock()

    def fake_fetch(url):
        with lock:
            calls[url] += 1
            attempt = calls[url]
        index = int(url.rsplit("/", 1)[1].split(".")[0])
        if index % 9 == 0:
            raise OSError(f"HTTP 404 for {index}")
        if index < 200 and index % 13 == 0 and attempt == 1:
            raise OSError(f"transient failure for {index}")
        return Image.new("RGB", (4, 4), (index % 256, index // 256, 7))

    monkeypatch.setattr(format_conversion, "_fetch_row_image", fake_fetch)
    return calls


def _dataset(total):
    return Dataset.from_list(
        [{"image": f"https://example.com/{i}.png", "caption": f"caption {i}"} for i in range(total)]
    )


@pytest.mark.parametrize("total", [450, 1200])
def test_probe_downloads_are_not_fetched_again(fetches, total):
    converted = convert_to_vlm_format(
        _dataset(total), instruction = "Describe.", text_column = "caption"
    )

    # Every URL is fetched once, except one whose probe download failed, which the conversion retries.
    probe_failures = {i for i in range(200) if i % 9 == 0 or i % 13 == 0}
    expected = Counter(
        {f"https://example.com/{i}.png": 2 if i in probe_failures else 1 for i in range(total)}
    )
    assert fetches == expected

    # Rows that only failed transiently in the probe still train; permanent 404s are skipped.
    kept = [i for i in range(total) if i % 9 != 0]
    assert [row["messages"][1]["content"][0]["text"] for row in converted] == [
        f"caption {i}" for i in kept
    ]
    for row, i in zip(converted, kept):
        image = row["messages"][0]["content"][1]["image"]
        assert image.getpixel((0, 0)) == (i % 256, i // 256, 7)
