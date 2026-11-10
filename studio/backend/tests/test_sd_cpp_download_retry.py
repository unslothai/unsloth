# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The prebuilt download retries a transient failure (HTTP 5xx / 429, socket errors) and raises a permanent one at once.

A single gateway 504 from the GitHub asset CDN used to fail the CUDA bundle, after which the MiniMax-H3 native load
installed the CPU build on a CUDA host and rendered on the CPU."""

from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import install_sd_cpp_prebuilt as sdmod  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake(monkeypatch, outcomes):
    calls = []

    def urlopen(req, timeout = 300.0):
        url = getattr(req, "full_url", req)
        calls.append(url)
        out = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(out, BaseException):
            raise out
        return _Resp(out)

    monkeypatch.setattr(sdmod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls


def _http(code):
    return urllib.error.HTTPError("https://example.invalid/a.zip", code, "x", None, None)


def test_transient_504_is_retried_then_succeeds(tmp_path, monkeypatch):
    calls = _fake(monkeypatch, [_http(504), _http(502), b"payload"])
    dest = tmp_path / "a.zip"
    sdmod._download("https://example.invalid/a.zip", dest)
    assert dest.read_bytes() == b"payload"
    assert len(calls) == 3


def test_socket_error_and_429_are_retried(tmp_path, monkeypatch):
    calls = _fake(monkeypatch, [urllib.error.URLError("reset"), _http(429), b"ok"])
    dest = tmp_path / "a.zip"
    sdmod._download("https://example.invalid/a.zip", dest)
    assert dest.read_bytes() == b"ok" and len(calls) == 3


@pytest.mark.parametrize("code", [403, 404])
def test_permanent_4xx_is_raised_without_retry(tmp_path, monkeypatch, code):
    calls = _fake(monkeypatch, [_http(code), b"never"])
    with pytest.raises(urllib.error.HTTPError) as err:
        sdmod._download("https://example.invalid/a.zip", tmp_path / "a.zip")
    assert err.value.code == code and len(calls) == 1


def test_persistent_5xx_raises_after_all_attempts(tmp_path, monkeypatch):
    calls = _fake(monkeypatch, [_http(504)])
    with pytest.raises(urllib.error.HTTPError) as err:
        sdmod._download("https://example.invalid/a.zip", tmp_path / "a.zip")
    assert err.value.code == 504 and len(calls) == sdmod._DOWNLOAD_ATTEMPTS


def test_a_stall_that_used_the_whole_timeout_is_not_retried(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    calls = []

    def urlopen(req, timeout = 300.0):
        calls.append(req)
        clock[0] += timeout
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(sdmod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(urllib.error.URLError):
        sdmod._download("https://example.invalid/a.zip", tmp_path / "a.zip", timeout = 30.0)
    assert len(calls) == 1
