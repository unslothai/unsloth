# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cheap row counts for check-format's ``total_rows`` where the preview itself cannot give one: a Hub
dataset id (the dataset viewer's ``/size`` metadata) and an uploaded JSON / JSONL / CSV file (a bounded
scan). Every helper answers None rather than a guess, never raises, and never waits long."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

HUB_ROWS_TIMEOUT_S = 3.0
_HUB_ROWS_TTL_S = 15 * 60
_HUB_ROWS_CACHE_MAX = 256
_hub_rows_cache: "OrderedDict[tuple, tuple[float, int]]" = OrderedDict()
_hub_rows_lock = threading.Lock()
_hub_rows_pool = ThreadPoolExecutor(max_workers = 4, thread_name_prefix = "hub-rows")

# Caps keep each count well under a second: a JSONL line scan is ~1 GB/s, a full CSV parse (quoted
# newlines included) ~40 MB/s, a JSON array is parsed whole.
LINE_SCAN_MAX_BYTES = 256 * 1024 * 1024
CSV_PARSE_MAX_BYTES = 16 * 1024 * 1024
JSON_ARRAY_MAX_BYTES = 32 * 1024 * 1024


def _hub_rows_source() -> Optional[str]:
    """The datasets server to ask, or None where it cannot answer for the configured Hub (same gate as
    the viewer rows preview: offline, ModelScope, or a mirrored HF_ENDPOINT with no HF_DATASETS_SERVER)."""
    import os

    from utils.hf_endpoint import _DEFAULT_HF_ENDPOINT, browser_hf_endpoint, get_hf_datasets_server
    from utils.hub_settings import MODELSCOPE, active_source
    from utils.utils import hf_env_offline

    if hf_env_offline() or active_source() == MODELSCOPE:
        return None
    if browser_hf_endpoint() != _DEFAULT_HF_ENDPOINT and not os.environ.get("HF_DATASETS_SERVER"):
        return None
    return get_hf_datasets_server()


def _split_rows_from_size(body: dict, subset: Optional[str], split: str) -> Optional[int]:
    """``num_rows`` of the one (config, split) the loader would read; None when ambiguous or partial."""
    size = body.get("size") or {}
    splits = [
        entry
        for entry in (size.get("splits") or [])
        if isinstance(entry, dict) and entry.get("split") == split
    ]
    if subset:
        splits = [entry for entry in splits if entry.get("config") == subset]
    elif len(splits) > 1:
        # load_dataset with no name reads the "default" config; several named ones would raise.
        splits = [entry for entry in splits if entry.get("config") == "default"]
    if len(splits) != 1:
        return None
    entry = splits[0]
    rows = entry.get("num_rows")
    if body.get("partial"):
        # A partial conversion counts only the converted part; the server's full-split estimate stands in.
        rows = entry.get("estimated_num_rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        return None
    return rows


def _fetch_hub_split_rows(
    server: str, dataset: str, subset: Optional[str], split: str, hf_token, timeout: float
) -> Optional[int]:
    import httpx

    params = {"dataset": dataset}
    if subset:
        params["config"] = subset
    try:
        response = httpx.get(
            f"{server}/size",
            params = params,
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None,
            timeout = timeout,
        )
        if response.status_code != 200:
            return None
        body = response.json()
    except Exception as e:
        from hub.utils import download_registry
        logger.info(
            "Hub dataset row count unavailable: %s",
            download_registry.scrub_secrets(str(e), hf_token = hf_token),
        )
        return None
    return _split_rows_from_size(body, subset, split) if isinstance(body, dict) else None


def _cache_key(server: str, dataset: str, subset: Optional[str], split: str) -> tuple:
    return (server, dataset, subset or "", split)


def _cached_rows(key: tuple) -> Optional[int]:
    with _hub_rows_lock:
        hit = _hub_rows_cache.get(key)
        if hit is None:
            return None
        stamp, rows = hit
        if time.monotonic() - stamp > _HUB_ROWS_TTL_S:
            _hub_rows_cache.pop(key, None)
            return None
        _hub_rows_cache.move_to_end(key)
        return rows


def _store_rows(key: tuple, rows: Optional[int]) -> Optional[int]:
    # Only answers are kept: a miss or an outage is retried on the next check.
    if rows is not None:
        with _hub_rows_lock:
            _hub_rows_cache[key] = (time.monotonic(), rows)
            _hub_rows_cache.move_to_end(key)
            while len(_hub_rows_cache) > _HUB_ROWS_CACHE_MAX:
                _hub_rows_cache.popitem(last = False)
    return rows


def start_hub_split_rows(
    dataset: str,
    subset: Optional[str],
    split: Optional[str],
    hf_token,
    *,
    timeout: float = HUB_ROWS_TIMEOUT_S,
) -> Callable[[], Optional[int]]:
    """Begin looking up a Hub split's row count in the background and return a getter for it. The getter
    waits only for what is left of ``timeout`` since the start, so a preview slower than that pays nothing.
    The caller uses the answer only once its own (gated) preview read succeeded."""
    split = split or "train"
    try:
        server = _hub_rows_source()
    except Exception:
        server = None
    if server is None or not re.fullmatch(r"[\w.-]+", split):
        return lambda: None
    key = _cache_key(server, dataset, subset, split)
    cached = _cached_rows(key)
    if cached is not None:
        return lambda: cached
    started = time.monotonic()
    try:
        future: Future = _hub_rows_pool.submit(
            lambda: _store_rows(
                key, _fetch_hub_split_rows(server, dataset, subset, split, hf_token, timeout)
            )
        )
    except Exception:
        return lambda: None

    def result() -> Optional[int]:
        remaining = max(0.0, timeout - (time.monotonic() - started))
        try:
            return future.result(timeout = remaining)
        except Exception:
            return None

    return result


def _count_jsonl_lines(path: Path) -> Optional[int]:
    """Non-blank lines, as the JSON loader reads JSON Lines; None over the cap."""
    if path.stat().st_size > LINE_SCAN_MAX_BYTES:
        return None
    rows = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


# A JSON Lines row longer than this is not a cheap count; the first line is read at most this far.
FIRST_LINE_MAX_BYTES = 1024 * 1024


def _first_non_blank_line(path: Path) -> Optional[bytes]:
    """The first non-blank line, read at most FIRST_LINE_MAX_BYTES at a time; None when longer
    (a minified array or a huge row), so a big single-line file is never read whole."""
    with open(path, "rb") as handle:
        while True:
            line = handle.readline(FIRST_LINE_MAX_BYTES + 1)
            if not line:
                return None
            if len(line) > FIRST_LINE_MAX_BYTES and not line.endswith(b"\n"):
                return line.lstrip()[:1] or None  # just enough to tell an array from a row
            if line.strip():
                return line.strip()


def _count_json_rows(path: Path) -> Optional[int]:
    import json

    size = path.stat().st_size
    # Both readings are bounded: the array parse by JSON_ARRAY_MAX_BYTES, the line count by
    # LINE_SCAN_MAX_BYTES. Reject anything past both before reading a byte.
    if size > max(JSON_ARRAY_MAX_BYTES, LINE_SCAN_MAX_BYTES):
        return None
    first = _first_non_blank_line(path)
    if first is None:
        return None
    if first.startswith(b"["):
        if path.stat().st_size > JSON_ARRAY_MAX_BYTES:
            return None
        with open(path, "rb") as handle:
            payload = json.load(handle)
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return len(payload)
        return None
    # JSON Lines saved as .json: count lines only when the first line is a whole object on its own
    # (a pretty-printed object spans lines and would be miscounted).
    try:
        if not isinstance(json.loads(first), dict):
            return None
    except ValueError:
        return None
    return _count_jsonl_lines(path)


def _count_csv_rows(path: Path) -> Optional[int]:
    """Records as pandas reads them (the CSV loader's parser), so quoted newlines and blank lines
    count the same way; None over the cap."""
    if path.stat().st_size > CSV_PARSE_MAX_BYTES:
        return None
    import pandas as pd

    rows = 0
    # Every column is tokenised (no usecols): a row with extra fields must raise here as it does in the load.
    with pd.read_csv(
        path,
        dtype = str,
        keep_default_na = False,
        chunksize = 200_000,
    ) as chunks:
        for chunk in chunks:
            rows += len(chunk)
    return rows


def count_local_file_rows(path: Path) -> Optional[int]:
    """Exact row count of one uploaded JSON / JSONL / CSV file when it is cheap; None otherwise."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = _count_jsonl_lines(path)
        elif suffix == ".json":
            rows = _count_json_rows(path)
        elif suffix == ".csv":
            rows = _count_csv_rows(path)
        else:
            return None
    except Exception:
        return None
    return rows if rows and rows > 0 else None
