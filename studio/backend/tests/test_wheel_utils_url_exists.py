# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import urllib.error

import pytest

from utils import wheel_utils


class _Ok:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (None, True),
        (urllib.error.HTTPError("u", 404, "missing", None, None), False),
        (urllib.error.HTTPError("u", 403, "rate limited", None, None), None),
        (urllib.error.HTTPError("u", 503, "down", None, None), None),
        (urllib.error.URLError("offline"), None),
        (TimeoutError("stalled"), None),
    ],
)
def test_only_a_404_means_unpublished(monkeypatch, outcome, expected):
    def urlopen(request, timeout = None):
        if outcome is None:
            return _Ok()
        raise outcome

    monkeypatch.setattr(wheel_utils.urllib.request, "urlopen", urlopen)
    assert wheel_utils.url_exists("https://github.com/o/r/releases/download/t/w.whl") is expected


class _Sized(_Ok):
    def __init__(self, length):
        self.headers = {"Content-Length": length}


def test_url_exists_records_the_size_it_saw(monkeypatch):
    url = "https://github.com/o/r/releases/download/t/sized.whl"
    monkeypatch.setattr(wheel_utils, "_URL_SIZES", {})
    monkeypatch.setattr(
        wheel_utils.urllib.request, "urlopen", lambda request, timeout = None: _Sized("427062625")
    )
    assert wheel_utils.url_size(url) is None
    assert wheel_utils.url_exists(url) is True
    assert wheel_utils.url_size(url) == 427062625


@pytest.mark.parametrize("length", [None, "", "abc"])
def test_url_size_is_none_without_a_usable_length(monkeypatch, length):
    url = "https://github.com/o/r/releases/download/t/unsized.whl"
    monkeypatch.setattr(wheel_utils, "_URL_SIZES", {})
    monkeypatch.setattr(
        wheel_utils.urllib.request, "urlopen", lambda request, timeout = None: _Sized(length)
    )
    assert wheel_utils.url_exists(url) is True
    assert wheel_utils.url_size(url) is None


def test_a_404_records_no_size(monkeypatch):
    url = "https://github.com/o/r/releases/download/t/missing.whl"
    monkeypatch.setattr(wheel_utils, "_URL_SIZES", {})

    def urlopen(request, timeout = None):
        raise urllib.error.HTTPError("u", 404, "missing", None, None)

    monkeypatch.setattr(wheel_utils.urllib.request, "urlopen", urlopen)
    assert wheel_utils.url_exists(url) is False
    assert wheel_utils.url_size(url) is None
