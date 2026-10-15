# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API-initiated image loads download nothing; sentinels raise on any network fetch in staging."""

from __future__ import annotations

import os
import types

import pytest

import utils.hf_xet_fallback as xet
from core.inference import diffusion as diffusion_mod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import detect_family_for_pick

# FLUX.1 walks the shared staging path and avoids the FLUX.2 pairing preflight.
FLUX_GGUF = "unsloth/FLUX.1-dev-GGUF"
FLUX_BASE = "black-forest-labs/FLUX.1-dev"
FLUX_FILE = "flux1-dev-Q4_K_M.gguf"


class _Calls:
    """Every Hub call the load made, and how it made it."""

    def __init__(self):
        self.model_info: list[str] = []
        self.downloads: list[tuple[str, str, bool]] = []


def _install_sentinels(monkeypatch, calls, tmp_path, *, offline):
    """Replace each staging network helper: offline it raises on any fetch, online it only records."""
    import huggingface_hub

    def _model_info(self, repo_id, **_kwargs):
        calls.model_info.append(repo_id)
        if offline:
            raise AssertionError(f"model_info({repo_id!r}) reached the Hub on an offline load")
        return types.SimpleNamespace(siblings = [], sha = "deadbeef", gated = False, cardData = {})

    def _download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        local_files_only = bool(kwargs.get("local_files_only"))
        calls.downloads.append((repo_id, filename, local_files_only))
        if offline and not local_files_only:
            raise AssertionError(
                f"{repo_id}/{filename} was fetched without local_files_only on an offline load"
            )
        path = tmp_path / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _model_info, raising = False)
    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    # The wrapper's offline branch calls this directly; a sentinel catches a bypass.
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: _download(
            kwargs.get("repo_id"), kwargs.get("filename"), kwargs.get("token"), **kwargs
        ),
        raising = False,
    )
    # Mirror choice depends on the developer's HF cache, so pin it for determinism.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")


def _backend(monkeypatch, calls_seen):
    """A backend whose family detection is pinned and whose pipeline build is a capture."""
    backend = DiffusionBackend()
    backend._load_token = 1
    backend._loading = diffusion_mod._LoadingState(repo_id = FLUX_GGUF, base_repo = FLUX_BASE)
    fam = detect_family_for_pick(FLUX_GGUF, FLUX_FILE, None)
    assert fam is not None
    monkeypatch.setattr(diffusion_mod, "detect_family_for_pick", lambda *_a, **_k: fam)
    monkeypatch.setattr(backend, "load_pipeline", lambda **kwargs: calls_seen.update(kwargs))
    return backend


def test_an_api_initiated_image_load_opens_the_cache_and_downloads_nothing(monkeypatch, tmp_path):
    """The whole promise, end to end: every helper the image staging phase reaches either stays off
    the Hub or asks it for a cached file only."""
    calls = _Calls()
    _install_sentinels(monkeypatch, calls, tmp_path, offline = True)
    seen: dict = {}
    backend = _backend(monkeypatch, seen)

    backend._run_load(
        repo_id = FLUX_GGUF,
        gguf_filename = FLUX_FILE,
        # Pass base_repo explicitly: the card-tag lookup fails open and offline would resolve a different base.
        base_repo = FLUX_BASE,
        local_files_only = True,
        _load_token = 1,
    )

    # _run_load records failures on load_progress instead of raising, so the state is the result.
    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen.get("local_files_only") is True
    assert calls.model_info == []
    # Still resolved, but as a cache lookup: this is the multi-GB call.
    assert calls.downloads == [(FLUX_GGUF, FLUX_FILE, True)]
    assert seen.get("_base_local_dir") is None


def test_a_user_initiated_image_load_still_calls_every_one_of_them(monkeypatch, tmp_path):
    """The pre-PR path, unchanged: the UI load asks the Hub for sizes and PULLS the checkpoint."""
    calls = _Calls()
    _install_sentinels(monkeypatch, calls, tmp_path, offline = False)
    seen: dict = {}
    backend = _backend(monkeypatch, seen)

    backend._run_load(
        repo_id = FLUX_GGUF,
        gguf_filename = FLUX_FILE,
        base_repo = FLUX_BASE,
        _load_token = 1,
    )

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen.get("local_files_only") in (False, None)
    assert FLUX_GGUF in calls.model_info and FLUX_BASE in calls.model_info
    assert calls.downloads == [(FLUX_GGUF, FLUX_FILE, False)]


def test_the_estimate_and_the_pre_cast_plan_stand_down_offline(monkeypatch):
    """Both are pure Hub metadata, and both already have a "could not tell" answer their callers
    handle, so offline they take it rather than inventing a probe."""

    class _Boom:
        def __init__(self, *_a, **_k):
            pass

        def model_info(self, *_a, **_k):
            raise AssertionError("the Hub was asked about an offline load")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom)
    backend = DiffusionBackend()
    assert backend._estimate_download_bytes(
        FLUX_GGUF, FLUX_FILE, FLUX_BASE, None, local_files_only = True
    ) == (0, [])
    assert backend._te_prequant_plan_files(None, "fp8", None, None, local_files_only = True) == {}


def test_the_base_preflight_reads_the_cache_and_never_the_hub_offline(monkeypatch):
    """Offline, the base preflight skips the Hub but still reads the cache for other-root snapshots."""
    import huggingface_hub

    def _boom(*_a, **_k):
        raise AssertionError("the Hub was asked about an offline load")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _boom, raising = False)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", _boom, raising = False)
    # os.path.join, not "/": the function strips with os.path, so a POSIX literal fails on Windows.
    snapshot = os.path.join(os.sep + "snap", *FLUX_BASE.split("/"))
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, cache_dir = None: (
            None if cache_dir is not None else os.path.join(snapshot, *name.split("/"))
        ),
        raising = False,
    )
    assert (
        diffusion_mod._assert_base_repo_accessible(FLUX_BASE, None, local_files_only = True)
        == snapshot
    )


def test_the_prefetch_signature_declares_the_flag():
    """A default-True or missing parameter here is the bug itself: this is the call that moves the
    weights, so the flag has to be a named, default-False parameter of it."""
    import inspect

    param = inspect.signature(DiffusionBackend._prefetch_files).parameters["local_files_only"]
    assert param.default is False
