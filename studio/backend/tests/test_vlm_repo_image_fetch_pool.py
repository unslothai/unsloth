# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image + text rows naming repo files fetch them on a pool, with the same output as one at a time."""

import random
import threading
import time

from datasets import Dataset
from PIL import Image

from utils import hardware
from utils.datasets import format_conversion

N = 48


def _fake_hub(monkeypatch, tmp_path):
    tmp_path.mkdir(parents = True, exist_ok = True)
    paths = {}
    for i in range(N):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (i + 1, 2)).save(path)
        paths[f"images/{i}.png"] = str(path)
    state = {"live": 0, "peak": 0, "calls": []}
    lock = threading.Lock()

    def _download(repo_id, filename, **kwargs):
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            state["calls"].append(filename)
        try:
            time.sleep(0.02)
            if filename == "images/7.png":
                raise OSError("404 for images/7.png")
            return paths[filename]
        finally:
            with lock:
                state["live"] -= 1

    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_files", lambda self, *a, **k: ["README.md", *paths]
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    from utils.datasets import llm_assist

    monkeypatch.setattr(llm_assist, "llm_generate_dataset_warning", lambda *a, **k: None)
    return state


def _convert(workers, monkeypatch, tmp_path):
    state = _fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(hardware, "safe_thread_num_proc", lambda *a, **k: workers)
    ds = Dataset.from_dict(
        {
            "image": [f"{i}.png" if i % 5 else f"images/{i}.png" for i in range(N)],
            # A list caption is picked with the global random module, so the pick order must not move.
            "text": [[f"cap {i} a", f"cap {i} b", f"cap {i} c"] for i in range(N)],
        }
    )
    random.seed(1234)
    out = format_conversion.convert_to_vlm_format(
        ds, instruction = "Describe.", dataset_name = "org/ds"
    )
    rows = [
        (
            sample["messages"][0]["content"][1]["image"].size,
            sample["messages"][1]["content"][0]["text"],
        )
        for sample in out
    ]
    return rows, state


def test_pooled_fetch_matches_serial(monkeypatch, tmp_path):
    serial, serial_state = _convert(1, monkeypatch, tmp_path / "serial")
    pooled, pooled_state = _convert(8, monkeypatch, tmp_path / "pooled")

    assert len(serial) == N - 1
    assert pooled == serial
    assert sorted(pooled_state["calls"]) == sorted(serial_state["calls"])
    assert serial_state["peak"] == 1
    assert pooled_state["peak"] > 1
