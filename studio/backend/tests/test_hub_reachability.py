# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared "Hub did not connect" verdict. Requests use huggingface_hub's HTTP stack on a mocked transport."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import core.inference.diffusion as D
import core.inference.diffusion_compat as compat
import core.inference.diffusion_prequant_cache as C
from core.inference import hub_reachability as R

huggingface_hub = pytest.importorskip("huggingface_hub")
if (
    getattr(huggingface_hub, "set_client_factory", None) is None
):  # pragma: no cover - requests-era huggingface_hub
    pytest.skip("needs huggingface_hub's httpx client factory", allow_module_level = True)
import httpx  # noqa: E402

REPO = "org/cached-pipe"


@pytest.fixture(autouse = True)
def _fresh_verdict(monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    R._UNREACHABLE.clear()
    C._REMOTE_MEMO.clear()
    yield
    R._UNREACHABLE.clear()


def _transport(handler):
    """Route huggingface_hub's client through ``handler``; returns the list of (method, url, read timeout)."""
    seen: list = []

    def wrapped(request):
        seen.append(
            (request.method, str(request.url), request.extensions.get("timeout", {}).get("read"))
        )
        return handler(request)

    huggingface_hub.set_client_factory(lambda: httpx.Client(transport = httpx.MockTransport(wrapped)))
    return seen


@pytest.fixture
def blackholed():
    """Accepts and never answers: every request times out."""

    def handler(request):
        raise httpx.ReadTimeout("timed out", request = request)

    try:
        yield _transport(handler)
    finally:
        from huggingface_hub.utils._http import default_client_factory
        huggingface_hub.set_client_factory(default_client_factory)


@pytest.fixture
def answering():
    """A healthy Hub: ``model_info`` answers a small open repo, a HEAD answers with a commit."""

    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers = {
                    "x-repo-commit": "a" * 40,
                    "etag": '"' + "b" * 64 + '"',
                    "content-length": "4",
                },
                request = request,
            )
        return httpx.Response(
            200,
            json = {
                "id": REPO,
                "sha": "a" * 40,
                "gated": False,
                "cardData": {"base_model": "org/base"},
                "siblings": [{"rfilename": "model_index.json", "size": 4}],
            },
            request = request,
        )

    try:
        yield _transport(handler)
    finally:
        from huggingface_hub.utils._http import default_client_factory
        huggingface_hub.set_client_factory(default_client_factory)


def _cache(tmp_path: Path, repo: str, files: dict[str, bytes]) -> Path:
    """A minimal Hub cache entry (refs/main + snapshot files) for ``repo``; returns the cache root."""
    root = tmp_path / "hub"
    entry = root / ("models--" + repo.replace("/", "--"))
    commit = "c" * 40
    (entry / "refs").mkdir(parents = True, exist_ok = True)
    (entry / "refs" / "main").write_text(commit)
    for name, data in files.items():
        path = entry / "snapshots" / commit / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(data)
    return root


# ------------------------------------------------------------------------------------------------------ model_info
def test_metadata_calls_share_one_timeout_on_a_blackholed_hub(blackholed):
    """Gated check, mirror SHA, size estimate: one request between them, not one wait each."""
    for _ in range(2):
        assert D._assert_base_repo_accessible(REPO, None) is None  # fail open, as before
        assert D.DiffusionBackend._current_sha(REPO, None) is None
    failures: list = []
    assert D.DiffusionBackend._estimate_download_bytes(
        REPO, None, REPO, None, kind = "pipeline", failures_out = failures
    ) == (0, [])
    assert len(failures) == 1  # the estimate still reports itself partial
    assert len(blackholed) == 1, blackholed


def test_model_info_is_bounded(blackholed):
    """huggingface_hub's default client has no timeout, so a dropped packet hung the call indefinitely."""
    D.DiffusionBackend._current_sha(REPO, None)
    assert blackholed[0][2] == R.METADATA_TIMEOUT_SECONDS


def test_lookups_that_pick_what_loads_are_bounded_but_never_skipped(blackholed):
    """Lookups that decide WHICH weights load are still asked, only with a timeout."""
    R.note_hub_failure(httpx.ConnectError("refused"))
    assert D._hf_base_model(REPO, None) is None
    source = types.SimpleNamespace(kind = "repo", location = "org/prequant", filename = "x.safetensors")
    with pytest.raises(Exception):
        D.DiffusionBackend._prequant_source_hub_entry(source, None)
    asked = [s for s in blackholed if "/api/models/" in s[1]]
    assert len(asked) == 2, blackholed
    assert all(s[2] == R.METADATA_TIMEOUT_SECONDS for s in asked)


def test_an_answering_hub_is_asked_every_time_with_unchanged_answers(answering):
    """No verdict when the Hub answers: every call still goes out and returns what the Hub said."""
    for _ in range(2):
        assert D._hf_base_model(REPO, None) == "org/base"
        assert D.DiffusionBackend._current_sha(REPO, None) == "a" * 40
        assert D._assert_base_repo_accessible(REPO, None) is None
    assert len(answering) == 6
    assert not R._UNREACHABLE


def test_an_http_error_is_an_answer_not_an_unreachable_hub():
    """404 / 401 / 5xx come from a reachable Hub: they never silence the next call."""

    def handler(request):
        return httpx.Response(503, request = request)

    seen = _transport(handler)
    try:
        for _ in range(2):
            assert D.DiffusionBackend._current_sha(REPO, None) is None
    finally:
        from huggingface_hub.utils._http import default_client_factory
        huggingface_hub.set_client_factory(default_client_factory)
    assert len([s for s in seen if "/api/models/" in s[1]]) >= 2
    assert not R._UNREACHABLE


def test_the_verdict_expires_and_a_success_clears_it(blackholed, monkeypatch):
    D.DiffusionBackend._current_sha(REPO, None)
    assert R.hub_unreachable()
    R._UNREACHABLE[R.endpoint()] -= R.TTL_SECONDS + 1
    assert not R.hub_unreachable()
    R.note_hub_failure(httpx.ConnectError("refused"))
    assert R.hub_unreachable()
    R.note_hub_success()
    assert not R.hub_unreachable()


def test_a_gated_uncached_base_keeps_todays_answers(monkeypatch):
    """An answering Hub that says gated still refuses an uncached base."""
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    R.note_hub_failure(httpx.ConnectError("refused"))
    assert D._assert_base_repo_accessible("org/gated", None) is None
    R.note_hub_success()

    def handler(request):
        return httpx.Response(
            401, headers = {"X-Error-Code": "GatedRepo", "X-Error-Message": "gated"}, request = request
        )

    seen = _transport(handler)
    try:
        with pytest.raises(ValueError, match = "org/gated"):
            D._assert_base_repo_accessible("org/gated", None)
    finally:
        from huggingface_hub.utils._http import default_client_factory
        huggingface_hub.set_client_factory(default_client_factory)
    assert seen and not R._UNREACHABLE


def test_a_duck_typed_client_gets_its_call_unchanged():
    """Test doubles and wrappers that do not take ``timeout`` keep working."""
    calls = []

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
        ):
            calls.append((repo_id, files_metadata))
            return "info"

    assert R.guarded(_Api()).model_info("a/b", files_metadata = True) == "info"
    assert calls == [("a/b", True)]
    R.note_hub_failure(httpx.ConnectError("refused"))
    with pytest.raises(ConnectionError):
        R.guarded(_Api()).model_info("a/b", files_metadata = True)
    assert len(calls) == 1


# ------------------------------------------------------------------------------------------------- cache fallbacks
def test_resolve_gguf_reads_a_cached_checkpoint_off_disk_while_the_hub_is_down(
    tmp_path, monkeypatch
):
    root = _cache(tmp_path, "org/gguf", {"model-Q4_K.gguf": b"GGUF"})
    monkeypatch.setattr(D, "hub_cache_dir", lambda: str(root))
    calls = []
    real = huggingface_hub.hf_hub_download

    def spy(*a, **k):
        calls.append(k.get("local_files_only"))
        return real(*a, **k)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", spy)
    R.note_hub_failure(httpx.ConnectError("refused"))
    path = D.DiffusionBackend._resolve_gguf_path(None, "org/gguf", "model-Q4_K.gguf", None)
    assert Path(path).read_bytes() == b"GGUF"
    assert calls == [True]


def test_resolve_gguf_still_downloads_an_uncached_checkpoint(tmp_path, monkeypatch):
    """A first download is never refused off a verdict: the Hub may be back."""
    root = tmp_path / "hub"
    monkeypatch.setattr(D, "hub_cache_dir", lambda: str(root))
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: calls.append(k.get("local_files_only")) or "x",
    )
    R.note_hub_failure(httpx.ConnectError("refused"))
    assert D.DiffusionBackend._resolve_gguf_path(None, "org/gguf", "model-Q4_K.gguf", None) == "x"
    assert calls == [False]


def test_compat_revision_probe_is_skipped_while_the_hub_is_down(blackholed):
    assert compat._hub_revision("org/gguf", "model-Q4_K.gguf", None) is None
    assert compat._hub_revision("org/gguf", "model-Q4_K.gguf", None) is None
    assert len(blackholed) == 1


def test_a_cached_pick_loads_off_the_cache_only_while_the_hub_is_down(tmp_path, monkeypatch):
    root = _cache(tmp_path, REPO, {"model_index.json": b"{}", "transformer/config.json": b"{}"})
    monkeypatch.setattr(D, "hub_cache_dir", lambda: str(root))
    fam = types.SimpleNamespace(name = "flux.1")
    pick = (fam, "pipeline", REPO, None, REPO, REPO)
    assert (
        D._pick_cached_while_hub_unreachable(*pick) is False
    )  # Hub fine: the online load is untouched
    R.note_hub_failure(httpx.ConnectError("refused"))
    assert D._pick_cached_while_hub_unreachable(*pick) is True
    assert D._cached_while_hub_unreachable(REPO, "transformer/config.json") is True
    # Nothing cached: a first download keeps asking the Hub.
    assert (
        D._pick_cached_while_hub_unreachable(fam, "pipeline", "org/new", None, "org/new", "org/new")
        is False
    )
    # A GGUF pick needs its checkpoint cached too, not only the base.
    assert (
        D._pick_cached_while_hub_unreachable(fam, "gguf", "org/gguf", "m.gguf", REPO, REPO) is False
    )
