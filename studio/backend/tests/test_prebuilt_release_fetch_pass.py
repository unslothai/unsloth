# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One update-status pass fetches each repo's GitHub release list once."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from utils import audio_cpp_freshness as afresh
from utils import llama_cpp_freshness as freshness
from utils import llama_cpp_update as upd
from utils import whisper_cpp_freshness as wfresh
from utils.prebuilt import freshness_flow

_REPO = "unslothai/llama.cpp"


def _release(tag: str, *assets: tuple[str, int]) -> dict:
    return {"tag_name": tag, "assets": [{"name": n, "size": s} for n, s in assets]}


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    for mod, name in ((freshness, "llama"), (wfresh, "whisper"), (afresh, "audio")):
        mod.reset_caches()
        monkeypatch.setattr(mod, "_cache_dir", lambda name = name: tmp_path / name)
    yield
    for mod in (freshness, wfresh, afresh):
        mod.reset_caches()


def _counting(monkeypatch, answers: dict):
    seen: list[str] = []

    def newest(repo, timeout, *, log_message):
        seen.append(repo)
        answer = answers[repo]
        return answer.pop(0) if isinstance(answer, list) else answer

    monkeypatch.setattr(freshness_flow, "_fetch_newest_published_release", newest)
    return seen


def test_a_pass_reads_tag_and_assets_from_one_fetch(monkeypatch):
    seen = _counting(monkeypatch, {_REPO: _release("b9518", ("app-b9518-linux-x64.tar.gz", 7))})
    with freshness_flow.shared_release_fetches():
        tag = freshness.latest_published_release(_REPO, force_refresh = True)
        assets = freshness.latest_release_assets(_REPO, force_refresh = True)
    assert (tag, assets) == ("b9518", {"app-b9518-linux-x64.tar.gz": 7})
    assert seen == [_REPO]
    assert freshness._release_memo[_REPO][1] == "b9518"
    assert freshness._assets_memo[_REPO][1] == {"app-b9518-linux-x64.tar.gz": 7}


def test_outside_a_pass_each_lookup_still_fetches(monkeypatch):
    seen = _counting(monkeypatch, {_REPO: _release("b9518", ("a", 1))})
    freshness.latest_published_release(_REPO, force_refresh = True)
    freshness.latest_release_assets(_REPO, force_refresh = True)
    assert seen == [_REPO, _REPO]


def test_each_pass_fetches_fresh(monkeypatch):
    seen = _counting(
        monkeypatch, {_REPO: [_release("b9518", ("a", 1)), _release("b9600", ("b", 2))]}
    )
    with freshness_flow.shared_release_fetches():
        assert freshness.latest_published_release(_REPO, force_refresh = True) == "b9518"
    with freshness_flow.shared_release_fetches():
        assert freshness.latest_published_release(_REPO, force_refresh = True) == "b9600"
        assert freshness.latest_release_assets(_REPO, force_refresh = True) == {"b": 2}
    assert seen == [_REPO, _REPO]


def test_a_failed_fetch_is_not_shared(monkeypatch):
    seen = _counting(monkeypatch, {_REPO: [None, _release("b9518", ("a", 1))]})
    with freshness_flow.shared_release_fetches():
        assert freshness.latest_published_release(_REPO, force_refresh = True) is None
        assert freshness.latest_release_assets(_REPO, force_refresh = True) == {"a": 1}
    assert seen == [_REPO, _REPO]


def test_nested_passes_join_the_outer_one(monkeypatch):
    seen = _counting(monkeypatch, {_REPO: _release("b9518", ("a", 1))})
    with freshness_flow.shared_release_fetches():
        with freshness_flow.shared_release_fetches():
            freshness.latest_published_release(_REPO, force_refresh = True)
        freshness.latest_release_assets(_REPO, force_refresh = True)
    assert seen == [_REPO]
    assert freshness_flow._pass_releases.get() is None


@contextmanager
def _no_pass():
    yield


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload))


def test_forced_update_status_fetches_each_repo_once_with_the_same_answer(monkeypatch, tmp_path):
    install = tmp_path / "llama.cpp"
    binary = install / "build" / "bin" / "llama-server"
    _write(binary, {})
    _write(
        install / "UNSLOTH_PREBUILT_INFO.json",
        {
            "tag": "b9493",
            "release_tag": "b9493",
            "published_repo": _REPO,
            "installed_at_utc": "2020-01-01T00:00:00Z",
            "asset": "app-b9493-linux-x64-cpu.tar.gz",
        },
    )
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_studio_custom_path_active", lambda: False)
    monkeypatch.setattr(upd, "_merge_whisper_status", lambda status, **kw: status)
    monkeypatch.setattr(upd, "_merge_audio_status", lambda status, **kw: status)
    for name in ("LLAMA_SERVER_PATH", "UNSLOTH_LLAMA_CPP_PATH", "UNSLOTH_DISABLE_UPDATE_CHECK"):
        monkeypatch.delenv(name, raising = False)
    release = _release("b9518", ("app-b9518-linux-x64-cpu.tar.gz", 4242))

    seen = _counting(monkeypatch, {_REPO: release})
    first = upd.get_update_status(force_refresh = True)
    assert seen == [_REPO]
    assert first["update_available"] is True
    assert (first["latest_tag"], first["update_size_bytes"]) == ("b9518", 4242)

    monkeypatch.setattr(freshness_flow, "shared_release_fetches", _no_pass)
    seen.clear()
    second = upd.get_update_status(force_refresh = True)
    assert seen == [_REPO, _REPO]
    first.pop("job"), second.pop("job")
    assert first == second
