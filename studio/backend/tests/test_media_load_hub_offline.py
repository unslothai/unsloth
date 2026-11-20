# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With HF_HUB_OFFLINE=1 an image or video load reads the cache only, as a local_files_only one does: the planner
asks the Hub nothing (it asked for the FP8 mirror's listing and the FLUX.3 pipeline index) and the pipeline build
gets local_files_only=True (it got False, so diffusers' revision / model_info lookups raised OfflineModeIsEnabled
and failed a load whose files were all cached). No GPU, no network: every Hub API call is patched to fail the test
and the pipeline build is stubbed."""

from __future__ import annotations

import time

import pytest

_HUB_API = (
    "model_info",
    "repo_info",
    "list_repo_files",
    "list_repo_tree",
    "get_paths_info",
    "file_exists",
    "list_repo_refs",
)


@pytest.fixture
def hub_calls(monkeypatch):
    from huggingface_hub import HfApi, file_download
    from huggingface_hub.errors import OfflineModeIsEnabled

    calls = []

    def _refuse(name):
        def _call(*args, **kwargs):
            calls.append((name, args[1:2], kwargs.get("revision")))
            raise OfflineModeIsEnabled(f"{name}: offline mode is enabled")

        return _call

    for name in _HUB_API:
        monkeypatch.setattr(HfApi, name, _refuse(name))
    monkeypatch.setattr(file_download, "get_hf_file_metadata", _refuse("get_hf_file_metadata"))
    return calls


def _backend(media):
    if media == "image":
        from core.inference.diffusion import DiffusionBackend
        return DiffusionBackend
    from core.inference.video import VideoBackend
    return VideoBackend


class _Built(Exception):
    pass


def _run(monkeypatch, media, repo):
    cls = _backend(media)
    seen = {}

    def _load_pipeline(self, **kwargs):
        seen["local_files_only"] = kwargs.get("local_files_only")
        raise _Built("stub build")

    monkeypatch.setattr(cls, "load_pipeline", _load_pipeline)
    backend = cls()
    backend.begin_load(repo)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if backend.load_progress().get("phase") == "error" or backend._loading is None:
            break
        time.sleep(0.05)
    return seen, backend.load_progress()


CASES = [
    ("image", "black-forest-labs/flux-3-image-klein"),
    ("image", "black-forest-labs/flux-3-image-klein-base"),
    ("video", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
]


@pytest.mark.parametrize("media,repo", CASES)
def test_hf_hub_offline_load_asks_the_hub_nothing(monkeypatch, hub_calls, media, repo):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen, progress = _run(monkeypatch, media, repo)
    # The stubbed build ran: the load got that far, and offline.
    assert "stub build" in (progress.get("error") or ""), progress
    assert seen == {"local_files_only": True}
    assert hub_calls == []


@pytest.mark.parametrize("media,repo", CASES[:1] + CASES[2:])
def test_transformers_offline_and_the_hub_constant_count_too(monkeypatch, hub_calls, media, repo):
    import huggingface_hub.constants as constants

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    seen, _progress = _run(monkeypatch, media, repo)
    assert seen == {"local_files_only": True} and hub_calls == []


def test_online_loads_keep_their_flag(monkeypatch):
    import huggingface_hub.constants as constants

    from core.inference import hub_reachability

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert hub_reachability.offline_mode() is False
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert hub_reachability.offline_mode() is True
