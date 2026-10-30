"""Runs on every platform: off Linux the partial verdict cache stays out of the way."""

from __future__ import annotations

import sys

from hub.utils import partial_verdict_cache
from utils.scan_memo import scan_memo


def test_enabled_only_on_linux():
    assert partial_verdict_cache.enabled() is sys.platform.startswith("linux")


def test_when_disabled_every_call_computes_and_nothing_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(partial_verdict_cache, "enabled", lambda: False)
    partial_verdict_cache.invalidate()
    calls = []

    def compute():
        calls.append(1)
        return len(calls) % 2 == 0

    with scan_memo():
        answers = [partial_verdict_cache.cached_verdict("k", tmp_path, compute) for _ in range(4)]
    assert answers == [False, True, False, True]
    assert len(calls) == 4
    assert not partial_verdict_cache._entries
