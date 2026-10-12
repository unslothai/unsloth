# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One failed `/props` read does not turn a notice-writing server into a legacy one.

The launch reads `/props` to confirm parking and again to learn whether the build writes the
per-stream notices. A timeout on the second read cleared the capability, and the aggregate park
reading then excused an unrelated stream's stall for the whole park window.
"""

from __future__ import annotations

import inspect

from core.inference import llama_cpp as llama_mod
from core.inference.llama_cpp import LlamaCppBackend


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _backend() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._port = 48861
    backend._api_key = None
    return backend


def test_a_timeout_after_a_good_read_keeps_what_the_server_reported(monkeypatch):
    answers = [_Response({"exact_concurrency": False, "preempt": {"enabled": True}})]

    def get(*args, **kwargs):
        if answers:
            return answers.pop(0)
        raise llama_mod.httpx.ReadTimeout("slow")

    monkeypatch.setattr(llama_mod.httpx, "get", get)
    backend = _backend()
    assert backend._server_confirms_parking() is True
    props = backend._server_props_or_last()
    assert "exact_concurrency" in props, props


def test_a_new_server_starts_with_nothing_seen():
    source = inspect.getsource(LlamaCppBackend.load_model)
    spawn = source.index("_last_spawn_cmd = list(run_cmd)")
    assert "self._server_props_seen = None" in source[spawn : spawn + 300]
    assert "_server_props = self._server_props_or_last()" in source


def test_nothing_seen_reads_as_nothing(monkeypatch):
    def get(*args, **kwargs):
        raise llama_mod.httpx.ReadTimeout("slow")

    monkeypatch.setattr(llama_mod.httpx, "get", get)
    assert _backend()._server_props_or_last() == {}
