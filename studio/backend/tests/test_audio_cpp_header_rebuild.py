# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp loads rebuild a republished GGUF from the cached older copy when only its header changed."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_files
from core.inference.audio_cpp_backend import AudioCppBackend
from core.inference.audio_cpp_models import RepoFile, _hub_listing
from hub.utils import gguf_header_delta as delta

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_gguf_header_delta import (  # noqa: E402, F401
    COMMIT_NEW,
    COMMIT_OLD,
    NAME,
    NEW,
    TENSORS,
    Fetcher,
    _small_ranges,
    cache_with_old,
    make_gguf,
    sha,
)

REPO = "unsloth/Model-GGUF"
EXTRA = "codec.bin"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub: HEAD answers for COMMIT_NEW, Range reads serve ``hub.data``, downloads are recorded."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: tmp_path)
    state = SimpleNamespace(data = NEW, heads = [], downloads = [], extra_blob = None)

    def head(url, token = None, timeout = None):
        state.heads.append(url)
        return SimpleNamespace(
            commit_hash = COMMIT_NEW, etag = f'"{sha(state.data)}"', size = len(state.data)
        )

    def download(repo_id, filename, token = None, cache_dir = None):
        state.downloads.append(filename)
        repo_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
        snap = repo_dir / "snapshots" / (repo_dir / "refs" / "main").read_text()
        if filename == EXTRA and state.extra_blob:  # an unchanged blob: link only
            os.symlink(os.path.join("..", "..", "blobs", state.extra_blob), snap / filename)

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", head)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(delta, "hub_range_fetcher", lambda *a, **k: Fetcher(state.data))
    return state


def _model(*files):
    return SimpleNamespace(
        repo_id = REPO, local_path = None, variant = SimpleNamespace(files = tuple(files))
    )


def _new_file(listed_sha = None, data = NEW):
    return RepoFile(NAME, len(data), sha(data) if listed_sha is None else listed_sha)


def test_header_only_change_is_rebuilt_without_a_download(tmp_path, hub):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    model = _model(_new_file())
    assert AudioCppBackend._download_missing(model, None) is True
    assert hub.downloads == [] and len(hub.heads) == 1
    assert (repo_dir / "refs" / "main").read_text() == COMMIT_NEW
    assert sha((repo_dir / "snapshots" / COMMIT_NEW / NAME).read_bytes()) == sha(NEW)
    assert audio_cpp_files.missing_files(model) == []


def test_changed_tensors_fall_back_to_the_download(tmp_path, hub):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    hub.data = make_gguf({"general.architecture": "llama"}, TENSORS[:1])
    AudioCppBackend._download_missing(_model(_new_file(data = hub.data)), None)
    assert hub.downloads == [NAME]
    assert (repo_dir / "refs" / "main").read_text() == COMMIT_OLD


def test_listing_sha_disagreeing_with_the_hub_falls_back(tmp_path, hub):
    cache_with_old(tmp_path, symlinks = True)
    AudioCppBackend._download_missing(_model(_new_file(listed_sha = "c" * 64)), None)
    assert hub.downloads == [NAME]


@pytest.mark.parametrize("case", ["offline", "kill switch", "no older copy", "not a gguf"])
def test_no_head_request_when_nothing_can_be_rebuilt(tmp_path, hub, monkeypatch, case):
    name = NAME
    if case == "offline":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    elif case == "kill switch":
        monkeypatch.setenv(delta.DELTA_ENV, "0")
    elif case == "not a gguf":
        name = EXTRA
    if case != "no older copy":
        cache_with_old(tmp_path, symlinks = True)
    model = _model(RepoFile(name, len(NEW), sha(NEW)))
    assert AudioCppBackend._rebuild_header_only(model, name, None, str(tmp_path)) is False
    assert hub.heads == []


def test_unchanged_files_of_a_mixed_variant_are_linked_after_the_rebuild(tmp_path, hub):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    extra = b"unchanged codec"
    blob = hub.extra_blob = hashlib.sha256(extra).hexdigest()
    (repo_dir / "blobs" / blob).write_bytes(extra)
    os.symlink(os.path.join("..", "..", "blobs", blob), repo_dir / "snapshots" / COMMIT_OLD / EXTRA)
    model = _model(_new_file(), RepoFile(EXTRA, len(extra)))
    assert audio_cpp_files.missing_files(model) == [(NAME, len(NEW))]
    assert AudioCppBackend._download_missing(model, None) is True
    assert hub.downloads == [EXTRA]
    assert audio_cpp_files.missing_files(model) == []


def test_hub_listing_keeps_the_lfs_sha256(monkeypatch):
    entries = [
        SimpleNamespace(path = "m.gguf", size = 4, lfs = SimpleNamespace(sha256 = "AB" * 32)),
        SimpleNamespace(path = "README.md", size = 2, lfs = None),
        SimpleNamespace(path = "sub", size = None),
    ]

    class Api:
        def __init__(self, token = None):
            pass

        def list_repo_tree(self, *args, **kwargs):
            return entries

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    files = _hub_listing(REPO, "", None)
    assert [(f.path, f.sha256) for f in files] == [("README.md", ""), ("m.gguf", "ab" * 32)]
    # The sha256 never changes equality, so a cached listing still matches a fresh one.
    assert files[1] == RepoFile("m.gguf", 4) and hash(files[1]) == hash(RepoFile("m.gguf", 4))
