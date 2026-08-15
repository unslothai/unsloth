# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An API-initiated IMAGE load downloads NOTHING.

The video twin of this suite lives in ``test_video_offline_load.py``; this is the same promise on
the diffusers image path, where ``local_files_only`` reached ``load_pipeline`` and nothing in the
``_run_load`` staging phase that runs before it -- so the byte estimate and the base preflight still
asked the Hub, and ``_prefetch_files`` (the one call on that path that moves multi-GB weights)
fetched without the flag. Every network-capable helper the staging phase reaches is replaced with a
sentinel that RAISES when it is asked to fetch, so a load that regains the network is a failing test
rather than a multi-GB surprise on a user's connection. The mirror test proves the user-initiated
(UI) path still calls exactly those helpers, which is the pre-PR behaviour nothing here changes.
"""

from __future__ import annotations

import types

import pytest

import utils.hf_xet_fallback as xet
from core.inference import diffusion as diffusion_mod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import detect_family_for_pick

# A plain FLUX.1 GGUF pick: it walks the shared staging path every image pick walks, and its family
# name keeps the FLUX.2 pairing preflight out of the way (that guard is a header read, not a fetch).
FLUX_GGUF = "unsloth/FLUX.1-dev-GGUF"
FLUX_BASE = "black-forest-labs/FLUX.1-dev"
FLUX_FILE = "flux1-dev-Q4_K_M.gguf"


class _Calls:
    """Every Hub call the load made, and how it made it."""

    def __init__(self):
        self.model_info: list[str] = []
        self.downloads: list[tuple[str, str, bool]] = []


def _install_sentinels(monkeypatch, calls, tmp_path, *, offline):
    """Replace every network helper the staging phase can reach.

    ``offline`` is the assertion: a metadata probe is refused outright (there is no offline form of
    ``model_info``), and a download is refused unless it carries ``local_files_only=True``, which is
    what makes it a cache lookup rather than a fetch. Online they only record, so the same fake
    serves both directions and the two tests differ by one flag.
    """
    import huggingface_hub

    def _model_info(self, repo_id, **_kwargs):
        calls.model_info.append(repo_id)
        if offline:
            raise AssertionError(f"model_info({repo_id!r}) reached the Hub on an offline load")
        return types.SimpleNamespace(siblings = [], sha = "deadbeef", gated = False, cardData = {})

    def _download(repo_id, filename, token = None, **kwargs):
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
    # The wrapper's own offline branch calls this directly; a sentinel here catches a bypass.
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: _download(
            kwargs.get("repo_id"), kwargs.get("filename"), kwargs.get("token"), **kwargs
        ),
        raising = False,
    )
    # Deterministic fetch target: the mirror swap is a pure local-cache test, and which side it
    # picks depends on the developer's own HF cache. Pinning it keeps both directions readable.
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
        # Carried by the request the way a saved image config carries it, so the card-tag lookup in
        # _resolve_base_repo is out of the picture: that read is metadata that FAILS OPEN, and
        # dropping it offline would resolve a DIFFERENT base than the load that cached the weights.
        base_repo = FLUX_BASE,
        local_files_only = True,
        _load_token = 1,
    )

    # _run_load swallows failures onto load_progress rather than raising, so the state IS the
    # result: cleared means the load ran through, an error string means a sentinel fired.
    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen.get("local_files_only") is True
    # Not one metadata probe: the byte estimate, the pre-cast plan and the base preflight all stand
    # down offline.
    assert calls.model_info == []
    # The checkpoint is still resolved -- as a cache lookup. THIS is the multi-GB call.
    assert calls.downloads == [(FLUX_GGUF, FLUX_FILE, True)]
    # And nothing was staged for from_pretrained, which resolves the cached snapshot itself.
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
    # The byte estimate probes the checkpoint repo and the base; the preflight probes the base too.
    assert FLUX_GGUF in calls.model_info and FLUX_BASE in calls.model_info
    # And the checkpoint is FETCHED, not looked up.
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
    """The preflight exists to name the repo a DOWNLOAD is about to 401 on. Offline there is no
    such download, so the Hub half stands down -- but the other-root escape it computes is a pure
    cache read and still runs, since that is what lets a base staged under huggingface_hub's
    import-time root load off disk."""
    import huggingface_hub

    def _boom(*_a, **_k):
        raise AssertionError("the Hub was asked about an offline load")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _boom, raising = False)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", _boom, raising = False)
    # Cached only under the import-time root: the live root misses, the fallback hits.
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, cache_dir = None: (
            None if cache_dir is not None else f"/snap/{repo}/{name}"
        ),
        raising = False,
    )
    assert diffusion_mod._assert_base_repo_accessible(
        FLUX_BASE, None, local_files_only = True
    ) == f"/snap/{FLUX_BASE}"


def test_the_prefetch_signature_declares_the_flag():
    """A default-True or missing parameter here is the bug itself: this is the call that moves the
    weights, so the flag has to be a named, default-False parameter of it."""
    import inspect

    param = inspect.signature(DiffusionBackend._prefetch_files).parameters["local_files_only"]
    assert param.default is False
