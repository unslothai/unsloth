# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An API-initiated video load downloads NOTHING.

``local_files_only=True`` is the contract the OpenAI-compatible routes load under: the model was
already staged, and the request may only open what is on disk. Every network-capable helper the
load reaches is replaced with a sentinel that RAISES when it is asked to fetch, so a load that
regains the network is a failing test rather than a multi-GB surprise on a user's connection. The
mirror test proves the user-initiated (UI) path still calls exactly those helpers, which is the
pre-PR behaviour nothing here is allowed to change.
"""

from __future__ import annotations

import inspect
import types

import pytest

import utils.hf_xet_fallback as xet
from core.inference import video as video_mod
from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family

# A plain (non-modular) family: the H3 conditioner / denoiser substitutions are exercised by their
# own suites, and this one keeps the load on the shared path every video pick walks.
WAN_GGUF = "unsloth/Wan2.2-TI2V-5B-GGUF"
WAN_BASE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_FILE = "wan2.2-ti2v-5b-Q4_K_M.gguf"


class _Calls:
    """Every Hub call the load made, and how it made it."""

    def __init__(self):
        self.model_info: list[str] = []
        self.downloads: list[tuple[str, str, bool]] = []


def _install_sentinels(monkeypatch, calls, tmp_path, *, offline):
    """Replace every network helper the load can reach.

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
        return types.SimpleNamespace(siblings = [], sha = "deadbeef")

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
    # The wrapper's own offline branch calls this directly; a sentinel here catches a bypass.
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: _download(
            kwargs.get("repo_id"), kwargs.get("filename"), kwargs.get("token"), **kwargs
        ),
        raising = False,
    )


def _backend(monkeypatch, calls_seen):
    """A backend whose family detection is pinned and whose pipeline build is a capture."""
    backend = VideoBackend()
    backend._load_token = 1
    backend._loading = video_mod._VideoLoadingState(repo_id = WAN_GGUF, base_repo = WAN_BASE)
    fam = detect_video_family(WAN_BASE)
    assert fam is not None and not fam.modular_workflow
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *_a, **_k: fam)
    monkeypatch.setattr(backend, "load_pipeline", lambda **kwargs: calls_seen.update(kwargs))
    return backend


def test_an_api_initiated_load_opens_the_cache_and_downloads_nothing(monkeypatch, tmp_path):
    """The whole promise, end to end: every helper on the load path either stays off the Hub or
    asks it for a cached file only."""
    calls = _Calls()
    _install_sentinels(monkeypatch, calls, tmp_path, offline = True)
    seen: dict = {}
    backend = _backend(monkeypatch, seen)

    backend._run_load(
        repo_id = WAN_GGUF,
        gguf_filename = WAN_FILE,
        local_files_only = True,
        _load_token = 1,
    )

    # _run_load swallows failures onto load_progress rather than raising, so the state IS the
    # result: cleared means the load ran through, an error string means a sentinel fired.
    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen.get("local_files_only") is True
    # Not one metadata probe: the byte estimate and the base prefetch both stand down offline.
    assert calls.model_info == []
    # The checkpoint is still resolved -- as a cache lookup.
    assert calls.downloads == [(WAN_GGUF, WAN_FILE, True)]
    # And nothing was staged for from_pretrained, which resolves the cached snapshot itself.
    assert seen.get("_base_local_dir") is None


def test_a_user_initiated_load_still_calls_every_one_of_them(monkeypatch, tmp_path):
    """The pre-PR path, unchanged: the UI load asks the Hub for sizes and pulls the checkpoint."""
    calls = _Calls()
    _install_sentinels(monkeypatch, calls, tmp_path, offline = False)
    seen: dict = {}
    backend = _backend(monkeypatch, seen)

    backend._run_load(
        repo_id = WAN_GGUF,
        gguf_filename = WAN_FILE,
        _load_token = 1,
    )

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen.get("local_files_only") in (False, None)
    # The byte estimate probes the checkpoint repo and the base; the base prefetch probes it again.
    assert WAN_GGUF in calls.model_info and WAN_BASE in calls.model_info
    # And the checkpoint is FETCHED, not looked up.
    assert calls.downloads == [(WAN_GGUF, WAN_FILE, False)]


def test_the_native_h3_path_binds_the_flag_instead_of_swallowing_it(monkeypatch):
    """``_run_load_h3_native`` used to take ``**_``, so the flag arrived and vanished -- and its
    four-file bundle, its sizing metadata and its sd-cli install all downloaded anyway."""
    assert (
        inspect.signature(VideoBackend._run_load_h3_native).parameters["local_files_only"].default
        is False
    )

    from core.inference.video_minimax_h3 import H3_GGUF_REPO

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *_a, **_k: fam)
    backend = VideoBackend()
    backend._load_token = 1
    backend._loading = video_mod._VideoLoadingState(repo_id = H3_GGUF_REPO, base_repo = fam.base_repo)
    seen: dict = {}
    monkeypatch.setattr(backend, "_run_load_h3_native", lambda **kwargs: seen.update(kwargs))

    backend._run_load(
        repo_id = H3_GGUF_REPO,
        gguf_filename = "MiniMax-H3-Q4_K_M.gguf",
        local_files_only = True,
        _load_token = 1,
    )
    assert seen.get("local_files_only") is True


def test_load_pipeline_carries_the_flag_into_the_native_path(monkeypatch):
    """load_pipeline is reachable without _run_load (tests, and the keep-warm path), so the
    dispatch has to pass the flag rather than let the parameter default re-enable downloads."""
    from core.inference.video_minimax_h3 import H3_GGUF_REPO

    backend = VideoBackend()
    seen: dict = {}
    monkeypatch.setattr(backend, "_run_load_h3_native", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(
        backend,
        "validate_load_request",
        lambda *a, **k: detect_video_family("MiniMaxAI/MiniMax-H3"),
    )

    backend.load_pipeline(
        H3_GGUF_REPO,
        gguf_filename = "MiniMax-H3-Q4_K_M.gguf",
        model_kind = "gguf",
        local_files_only = True,
    )
    assert seen.get("local_files_only") is True


def test_an_offline_denoiser_probe_reads_the_cache_instead_of_the_hub(monkeypatch):
    """Offline the "is there a replacement denoiser?" question is answered from disk. Refusing the
    load instead would break the one case the flag exists for; answering yes without checking would
    drop the dense shards a cache that lacks the artifact still needs."""
    from core.inference.diffusion import DiffusionBackend

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(lambda *a, **k: True))
    assert (
        VideoBackend._denoiser_prequant_cached_repo(fam, "int8", "MiniMaxAI/MiniMax-H3", "fl2va")
        == "unsloth/MiniMax-H3-FP8"
    )

    monkeypatch.setattr(
        DiffusionBackend, "_hub_file_is_cached", staticmethod(lambda *a, **k: False)
    )
    assert (
        VideoBackend._denoiser_prequant_cached_repo(fam, "int8", "MiniMaxAI/MiniMax-H3", "fl2va")
        is None
    )


def test_the_estimate_and_the_base_prefetch_stand_down_offline(monkeypatch):
    """Both are pure Hub metadata, and both already have a "could not tell" answer the callers
    handle, so offline they take it rather than inventing a probe."""

    class _Boom:
        def __init__(self, *_a, **_k):
            pass

        def model_info(self, *_a, **_k):
            raise AssertionError("the Hub was asked about an offline load")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Boom)
    backend = VideoBackend()
    assert (
        backend._estimate_download_bytes(
            WAN_GGUF, WAN_FILE, WAN_BASE, None, "gguf", local_files_only = True
        )
        is None
    )
    assert backend._predownload_base(WAN_BASE, None, "gguf", local_files_only = True) is None


def test_the_xet_wrapper_resolves_offline_without_the_shared_backend(monkeypatch, tmp_path):
    """``local_files_only`` must not depend on which unsloth_zoo is installed: the degraded stub
    drops unknown keywords, so a forwarded flag would silently become a download."""
    monkeypatch.setattr(
        xet,
        "_shared_hf_hub_download_with_xet_fallback",
        lambda *a, **k: pytest.fail("the shared transport ran for an offline request"),
    )
    seen: dict = {}

    def _hf_hub_download(**kwargs):
        seen.update(kwargs)
        path = tmp_path / "file.bin"
        path.write_bytes(b"")
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _hf_hub_download)
    out = xet.hf_hub_download_with_xet_fallback(
        "org/repo", "file.bin", None, cache_dir = str(tmp_path), local_files_only = True
    )
    assert out == str(tmp_path / "file.bin")
    assert seen["local_files_only"] is True


def test_the_xet_wrapper_is_unchanged_for_every_existing_caller(monkeypatch, tmp_path):
    """Default False: the online path still goes through the shared transport, and the flag is not
    even forwarded, so an older shared backend cannot trip over it."""
    seen: dict = {}

    def _shared(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return str(tmp_path / "file.bin")

    monkeypatch.setattr(xet, "_shared_hf_hub_download_with_xet_fallback", _shared)
    xet.hf_hub_download_with_xet_fallback("org/repo", "file.bin", None, cache_dir = str(tmp_path))
    assert seen["args"] == ("org/repo", "file.bin", None)
    assert "local_files_only" not in seen["kwargs"]


def _keywords_of(module_path: str, function: str, callee: str) -> set[str]:
    """The keyword names a call to *callee* inside *function* actually spells out.

    Read from the source rather than driven, because the branch that reaches this call needs a
    Modular Diffusers H3 pipeline and a card whose free memory has moved since the plan: the
    condition is real but not one a unit test can stage, and the keyword either is there or is not.
    """
    import ast
    import pathlib

    # Anchored on the package, not on the process CWD: CI runs pytest from the repo root with the
    # backend merely on PYTHONPATH, where a relative open raises FileNotFoundError.
    backend_root = pathlib.Path(video_mod.__file__).resolve().parents[2]
    tree = ast.parse((backend_root / module_path).read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function:
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == callee:
                return {kw.arg for kw in call.keywords if kw.arg}
    raise AssertionError(f"no call to {callee}() inside {function}()")


def test_the_hosted_prequantized_denoiser_is_not_fetched_by_a_load_nobody_asked_for():
    # `auto` is settled twice: the download plan decides against the card's CAPACITY, and this
    # branch re-decides against LIVE free memory once the previous pipeline is gone. So a pick the
    # plan sized as "the released bfloat16 denoiser fits" -- nothing hosted staged, the locality
    # gate reporting zero missing bytes, the switch starting the load -- can be re-decided here as
    # "take the hosted int8 checkpoint", and that is a multi-GB pull on a load that promised none.
    # The image twin has passed the flag here all along; the video path did not.
    assert "local_files_only" in _keywords_of(
        "core/inference/video.py", "_load_h3_modular_pipeline", "load_prequantized_transformer"
    )
    assert "local_files_only" in _keywords_of(
        "core/inference/diffusion.py", "_load_dense_quant_pipeline", "load_prequantized_transformer"
    )
