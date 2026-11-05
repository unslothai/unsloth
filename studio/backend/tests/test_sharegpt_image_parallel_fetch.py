# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ShareGPT+image conversion fetches row images concurrently but keeps row order, failures and caller context."""

import contextvars
import logging
import sys
import threading
import time
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import utils.hardware  # noqa: E402
from utils.datasets import format_conversion  # noqa: E402
from utils.datasets.format_conversion import convert_sharegpt_with_images_to_vlm_format  # noqa: E402

_CALLER = contextvars.ContextVar("caller", default = None)


class _SlowFetch:
    def __init__(
        self,
        fail_rows = (),
        delay = 0.02,
    ):
        self.fail_rows = set(fail_rows)
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.seen_context = []

    def __call__(self, url):
        row = int(url.rsplit("/", 1)[1].split(".")[0])
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.seen_context.append(_CALLER.get())
        try:
            time.sleep(self.delay)
            if row in self.fail_rows:
                raise OSError(f"cannot fetch row {row}")
            return Image.new("RGB", (row + 1, 1))
        finally:
            with self.lock:
                self.active -= 1


def _dataset(
    n,
    local_rows = (),
    tmp_path = None,
):
    rows = []
    for i in range(n):
        image = f"https://example.com/{i}.png"
        if i in local_rows:
            path = tmp_path / f"{i}.png"
            Image.new("RGB", (i + 1, 1)).save(path)
            image = str(path)
        rows.append(
            {
                "image": image,
                "conversations": [
                    {"from": "human", "value": f"<image> row {i}"},
                    {"from": "gpt", "value": f"answer {i}"},
                ],
            }
        )
    return Dataset.from_list(rows)


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.setattr(utils.hardware, "safe_thread_num_proc", lambda desired = None: 4)


def _widths(converted):
    return [m["messages"][0]["content"][0]["image"].size[0] - 1 for m in converted]


def test_url_rows_are_fetched_concurrently_in_row_order(monkeypatch, workers):
    fetch = _SlowFetch()
    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)

    converted = convert_sharegpt_with_images_to_vlm_format(_dataset(24))

    assert _widths(converted) == list(range(24))
    assert [m["messages"][0]["content"][1]["text"] for m in converted] == [
        f"row {i}" for i in range(24)
    ]
    assert 1 < fetch.peak <= 4


def test_failures_skip_rows_and_log_the_earliest_failing_row(monkeypatch, workers, caplog):
    # Row 9 fails fast while row 3 is still sleeping; the log must still name row 3, as a serial pass would.
    class _Fetch(_SlowFetch):
        def __call__(self, url):
            row = int(url.rsplit("/", 1)[1].split(".")[0])
            self.delay = 0.2 if row == 3 else 0.0
            return super().__call__(url)

    fetch = _Fetch(fail_rows = (3, 9))
    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)
    logs = []
    monkeypatch.setattr(
        format_conversion.logger, "info", lambda msg, *a, **k: logs.append(str(msg))
    )

    converted = convert_sharegpt_with_images_to_vlm_format(_dataset(16))

    assert _widths(converted) == [i for i in range(16) if i not in (3, 9)]
    first = [m for m in logs if "First conversion failure" in m]
    assert first == ["⚠️ First conversion failure: OSError: cannot fetch row 3"]
    assert any("Skipped 2/16" in m for m in logs)


def test_every_row_failing_raises_the_same_error(monkeypatch, workers):
    monkeypatch.setattr(format_conversion, "_fetch_row_image", _SlowFetch(fail_rows = range(8)))

    with pytest.raises(ValueError, match = "All 8 samples failed during ShareGPT\\+image conversion"):
        convert_sharegpt_with_images_to_vlm_format(_dataset(8))


def test_fetches_run_under_the_callers_context(monkeypatch, workers):
    fetch = _SlowFetch()
    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)
    token = _CALLER.set("account-a")
    try:
        convert_sharegpt_with_images_to_vlm_format(_dataset(12))
    finally:
        _CALLER.reset(token)

    assert fetch.seen_context == ["account-a"] * 12


def test_mixed_local_and_url_rows_keep_row_order(monkeypatch, workers, tmp_path):
    fetch = _SlowFetch()
    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)
    monkeypatch.setattr(format_conversion, "_own_image_path", lambda path, account: path)
    local = {1, 2, 5, 11}

    converted = convert_sharegpt_with_images_to_vlm_format(_dataset(12, local, tmp_path))

    assert _widths(converted) == list(range(12))
    assert len(fetch.seen_context) == 12 - len(local)


def test_local_only_dataset_starts_no_threads(monkeypatch, workers, tmp_path):
    monkeypatch.setattr(format_conversion, "_own_image_path", lambda path, account: path)
    # The first tqdm bar in a process starts tqdm's own TMonitor thread; that is not a fetch worker,
    # and whether this test is that first bar depends on what ran before it in the worker.
    from tqdm import std as tqdm_std

    monkeypatch.setattr(tqdm_std.tqdm, "monitor_interval", 0)
    before = threading.active_count()
    created = []
    real_init = threading.Thread.__init__

    def counting_init(self, *a, **k):
        created.append(1)
        real_init(self, *a, **k)

    monkeypatch.setattr(threading.Thread, "__init__", counting_init)
    ds = _dataset(6, set(range(6)), tmp_path)

    converted = convert_sharegpt_with_images_to_vlm_format(ds)

    assert _widths(converted) == list(range(6))
    assert created == []
    assert threading.active_count() == before


def test_concurrent_fetches_are_capped_on_many_core_hosts(monkeypatch):
    monkeypatch.setattr(utils.hardware, "safe_thread_num_proc", lambda desired = None: 64)
    fetch = _SlowFetch(delay = 0.05)
    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)

    converted = convert_sharegpt_with_images_to_vlm_format(_dataset(48))

    assert _widths(converted) == list(range(48))
    assert 1 < fetch.peak <= format_conversion._MAX_ROW_FETCH_WORKERS == 8


def test_a_row_without_the_image_key_is_skipped_as_before(monkeypatch, workers):
    monkeypatch.setattr(format_conversion, "_fetch_row_image", _SlowFetch())
    rows = list(_dataset(6))
    del rows[2]["image"]

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert _widths(converted) == [0, 1, 3, 4, 5]


def test_repo_filename_rows_download_concurrently(monkeypatch, workers, tmp_path):
    import huggingface_hub

    files = [f"images/{i}.png" for i in range(12)]
    for i in range(12):
        Image.new("RGB", (i + 1, 1)).save(tmp_path / f"{i}.png")
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", lambda self, *a, **k: files)
    fetch = _SlowFetch()

    def fake_download(repo, filename, **kwargs):
        name = filename.rsplit("/", 1)[1]
        fetch(f"https://hub/{name}")
        return str(tmp_path / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    rows = [
        {"image": f"{i}.png", "conversations": [{"from": "human", "value": f"<image> row {i}"}]}
        for i in range(12)
    ]

    converted = convert_sharegpt_with_images_to_vlm_format(rows, dataset_name = "org/ds")

    assert _widths(converted) == list(range(12))
    assert 1 < fetch.peak <= 4
