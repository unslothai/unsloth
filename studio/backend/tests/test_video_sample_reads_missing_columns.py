# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""video_sample fills a sparse first row from the next rows reading only the columns that are still missing, so an
image column is not decoded 31 times; the filled sample is the same as a full-row scan."""

import io

from datasets import Dataset, Image as HFImage
from PIL import Image

from utils.datasets import format_detection as fd


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format = "PNG")
    return {"bytes": buf.getvalue(), "path": None}


def _full_scan(dataset, sample):
    """The previous implementation: whole rows."""
    import itertools

    def _missing(key, value):
        if key == "messages" and isinstance(value, list):
            return not fd.video_parts(value)
        return value is None or (isinstance(value, str) and not value.strip())

    filled = dict(sample)
    for row in itertools.islice(iter(dataset), 1, 32):
        for key, value in row.items():
            if _missing(key, filled.get(key)) and not _missing(key, value):
                filled[key] = value
        if not any(_missing(k, v) for k, v in filled.items()):
            break
    return filled


def _datasets():
    text_messages = [
        [{"role": "user", "content": [{"type": "text", "text": f"q{i}"}]}] for i in range(40)
    ]
    video_messages = [
        [
            {
                "role": "user",
                "content": [{"type": "video", "video": "clip.mp4"}, {"type": "text", "text": "q"}],
            }
        ]
    ]
    images = Dataset.from_dict(
        {"image": [_png() for _ in range(40)], "messages": text_messages}
    ).cast_column("image", HFImage())
    sparse_video = Dataset.from_dict(
        {
            "image": [_png() for _ in range(6)],
            "video": [None, "", None, "b.mp4", None, "c.mp4"],
            "text": ["t"] * 6,
        }
    ).cast_column("image", HFImage())
    late_video_messages = Dataset.from_dict(
        {
            "image": [_png() for _ in range(5)],
            "messages": text_messages[:3] + video_messages + text_messages[:1],
        }
    ).cast_column("image", HFImage())
    return [images, sparse_video, late_video_messages]


def test_filled_sample_matches_a_full_row_scan():
    for dataset in _datasets():
        sample = dataset[0]
        expected = _full_scan(dataset, sample)
        got = fd.video_sample(dataset, sample)
        assert got.keys() == expected.keys()
        for key in expected:
            if key == "image":
                assert got[key].tobytes() == expected[key].tobytes()
            else:
                assert got[key] == expected[key]


def test_only_missing_columns_are_read(monkeypatch):
    dataset = _datasets()[0]
    seen = []
    original = Dataset.select_columns

    def spy(self, column_names, *args, **kwargs):
        seen.append(list(column_names))
        return original(self, column_names, *args, **kwargs)

    monkeypatch.setattr(Dataset, "select_columns", spy)
    fd.video_sample(dataset, dataset[0])
    assert seen == [["messages"]]


def test_a_complete_first_row_reads_nothing_and_lists_still_work():
    rows = [{"video": None, "text": "a"}, {"video": "x.mp4", "text": "b"}]
    assert fd.video_sample(rows, rows[0]) == {"video": "x.mp4", "text": "a"}
    complete = {"video": "x.mp4", "text": "a"}
    assert fd.video_sample([complete], complete) is complete
