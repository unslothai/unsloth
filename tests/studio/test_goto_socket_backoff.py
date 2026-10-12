# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""goto_with_socket_backoff retries a navigation only on net::ERR_NO_BUFFER_SPACE.

The Windows UI lane failed the update banner suite on its first navigation with
`Page.goto: net::ERR_NO_BUFFER_SPACE`, the runner running out of socket buffers,
which says nothing about the app. Anything else still fails at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _playwright_robust import SOCKET_BUFFER_BACKOFF_S, goto_with_socket_backoff  # noqa: E402

ENOBUFS = "Page.goto: net::ERR_NO_BUFFER_SPACE at http://127.0.0.1:18897/"


class _FakePage:
    def __init__(self, failures: list[str]) -> None:
        self._failures = list(failures)
        self.calls: list[tuple[str, dict]] = []

    def goto(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self._failures:
            raise RuntimeError(self._failures.pop(0))
        return "response"


def test_a_socket_buffer_failure_is_retried_after_backing_off():
    page, slept = _FakePage([ENOBUFS, ENOBUFS]), []
    got = goto_with_socket_backoff(
        page, "http://x/", sleep = slept.append, wait_until = "domcontentloaded"
    )
    assert got == "response"
    assert slept == list(SOCKET_BUFFER_BACKOFF_S)
    assert page.calls == [("http://x/", {"wait_until": "domcontentloaded"})] * 3


def test_the_retries_are_bounded():
    page, slept = _FakePage([ENOBUFS] * 5), []
    with pytest.raises(RuntimeError, match = "ERR_NO_BUFFER_SPACE"):
        goto_with_socket_backoff(page, "http://x/", sleep = slept.append)
    assert len(page.calls) == len(SOCKET_BUFFER_BACKOFF_S) + 1


def test_any_other_navigation_error_is_raised_at_once():
    page, slept = _FakePage(["Page.goto: net::ERR_CONNECTION_REFUSED"]), []
    with pytest.raises(RuntimeError, match = "ERR_CONNECTION_REFUSED"):
        goto_with_socket_backoff(page, "http://x/", sleep = slept.append)
    assert slept == [] and len(page.calls) == 1


def test_the_update_banner_suite_navigates_through_it():
    source = (Path(__file__).resolve().parent / "playwright_update_banner_layout.py").read_text(
        encoding = "utf-8"
    )
    assert 'goto_with_socket_backoff(page, f"{BASE}{path}"' in source
    assert "page.goto(" not in source
