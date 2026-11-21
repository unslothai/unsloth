# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reopening the picker re-lists each GGUF repo; within the window one Hub call answers them, and a 429
is not re-asked until its Retry-After passes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hub.utils import gguf


class _RateLimited(Exception):
    def __init__(self, retry_after):
        super().__init__("429 Too Many Requests")
        self.response = SimpleNamespace(status_code = 429, headers = {"Retry-After": retry_after})


def _fake_hub(
    monkeypatch,
    tmp_path,
    fail = None,
):
    calls = []

    class _Api:
        def __init__(self, token = None):
            self.token = token

        def model_info(
            self,
            repo_id,
            files_metadata = False,
            timeout = None,
            **_,
        ):
            calls.append((repo_id, self.token))
            if fail is not None:
                raise fail
            return SimpleNamespace(
                siblings = [SimpleNamespace(rfilename = "m-Q4_K_M.gguf", size = 5)],
            )

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return calls


def test_a_repo_is_listed_once_per_window_and_per_credential(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path)
    first = gguf.list_gguf_variants("unsloth/Demo-GGUF")
    second = gguf.list_gguf_variants("unsloth/demo-gguf")
    assert [v.quant for v in first[0]] == [v.quant for v in second[0]] == ["Q4_K_M"]
    assert len(calls) == 1
    gguf.list_gguf_variants("unsloth/Demo-GGUF", "hf_other")
    assert len(calls) == 2


def test_the_listing_is_asked_again_once_the_window_ends(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(gguf.time, "monotonic", lambda: clock[0])
    gguf.list_gguf_variants("unsloth/Demo-GGUF")
    clock[0] += gguf._HUB_LISTING_TTL_S + 1
    gguf.list_gguf_variants("unsloth/Demo-GGUF")
    assert len(calls) == 2


def test_a_rate_limited_listing_waits_out_retry_after(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path, fail = _RateLimited("30"))
    clock = [1000.0]
    monkeypatch.setattr(gguf.time, "monotonic", lambda: clock[0])
    for _ in range(3):
        with pytest.raises(_RateLimited):
            gguf.list_gguf_variants("unsloth/Demo-GGUF")
    assert len(calls) == 1
    clock[0] += 31
    with pytest.raises(_RateLimited):
        gguf.list_gguf_variants("unsloth/Demo-GGUF")
    assert len(calls) == 2


def test_other_failures_are_not_remembered(monkeypatch, tmp_path):
    calls = _fake_hub(monkeypatch, tmp_path, fail = ConnectionError("down"))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            gguf.list_gguf_variants("unsloth/Demo-GGUF")
    assert len(calls) == 2


def test_rate_limit_entries_expire_and_stay_bounded(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, fail = _RateLimited("30"))
    clock = [1000.0]
    monkeypatch.setattr(gguf.time, "monotonic", lambda: clock[0])
    for i in range(gguf._HUB_LISTING_MAX + 20):
        with pytest.raises(_RateLimited):
            gguf.list_gguf_variants(f"unsloth/Demo-{i}-GGUF")
    assert len(gguf._hub_rate_limited) == gguf._HUB_LISTING_MAX
    clock[0] += 31
    with pytest.raises(_RateLimited):
        gguf.list_gguf_variants("unsloth/Other-GGUF")
    assert len(gguf._hub_rate_limited) == 1


def test_a_remembered_429_does_not_grow_its_traceback(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path, fail = _RateLimited("30"))
    depths = []
    for _ in range(5):
        try:
            gguf.list_gguf_variants("unsloth/Demo-GGUF")
        except _RateLimited as exc:
            tb, n = exc.__traceback__, 0
            while tb is not None:
                tb, n = tb.tb_next, n + 1
            depths.append(n)
    assert depths[1] == depths[-1]
