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
