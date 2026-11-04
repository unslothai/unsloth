# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from core.systemone import native_worker


@pytest.fixture
def searches(monkeypatch):
    calls = []
    answer = {"binary": "/opt/llama.cpp/llama-server"}

    def find(**kwargs):
        calls.append(kwargs)
        return answer["binary"]

    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(find))
    monkeypatch.setattr(LlamaCppBackend, "_exec_path_for_launch", staticmethod(lambda b: b))
    return SimpleNamespace(calls = calls, answer = answer)


def test_every_lookup_searches_outside_a_settings_answer(searches):
    assert [native_worker.resolve_binary() for _ in range(3)] == ["/opt/llama.cpp/llama-server"] * 3
    assert len(searches.calls) == 3


def test_a_found_binary_is_searched_for_once_per_answer(searches):
    with native_worker.reuse_found_binary():
        assert [native_worker.resolve_binary() for _ in range(5)] == [
            "/opt/llama.cpp/llama-server"
        ] * 5
    assert len(searches.calls) == 1
    # The next answer searches again, so an install or removal in between is seen.
    searches.answer["binary"] = "/new/llama-server"
    with native_worker.reuse_found_binary():
        assert native_worker.resolve_binary() == "/new/llama-server"
    assert len(searches.calls) == 2


def test_a_miss_searches_every_time(searches):
    # A miss can log why (a pinned folder without the binary), once per lookup as before.
    searches.answer["binary"] = None
    with native_worker.reuse_found_binary():
        assert [native_worker.resolve_binary() for _ in range(3)] == [None] * 3
    assert len(searches.calls) == 3


def test_the_scope_ends_with_the_answer_even_on_error(searches):
    with pytest.raises(RuntimeError):
        with native_worker.reuse_found_binary():
            native_worker.resolve_binary()
            raise RuntimeError("boom")
    native_worker.resolve_binary()
    native_worker.resolve_binary()
    assert len(searches.calls) == 3
    assert native_worker._FOUND_BINARY.get() is None


def test_nested_scopes_and_threads_search_on_their_own(searches):
    import threading
    with native_worker.reuse_found_binary():
        native_worker.resolve_binary()
        with native_worker.reuse_found_binary():
            native_worker.resolve_binary()
            native_worker.resolve_binary()
        native_worker.resolve_binary()
        # A background thread (the llama.cpp GPU re-probe) does not see this answer's scope.
        thread = threading.Thread(target = native_worker.resolve_binary)
        thread.start()
        thread.join()
    assert len(searches.calls) == 3


def test_settings_answer_searches_once_and_is_unchanged(searches, monkeypatch):
    from routes import settings

    request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(server_port = 8888)),
        scope = {"server": ("127.0.0.1", 8888)},
    )
    searched = settings._systemone_answer(request).model_dump_json()
    per_lookup = len(searches.calls)
    assert per_lookup > 1
    searches.calls.clear()
    assert settings._systemone_response(request).model_dump_json() == searched
    assert len(searches.calls) == 1
