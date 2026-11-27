# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A returning video load while the Hub is down, on a cache a plan-pinned fetch wrote: the snapshot sits under
snapshots/<commit> with no refs/main (the Kaggle T4 fp16 route fetches the Wan 5B mirror pinned to its commit), so
an unpinned offline lookup of "main" finds nothing. The load pins the one cached snapshot instead."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

MIRROR = "unsloth/Wan2.2-TI2V-5B-FP8"
UPSTREAM = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SHA = "78dce9f849d0bda6fd9c5f174240a76cb295fa09"
INDEX = {
    "_class_name": "WanPipeline",
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _snapshot(
    cache,
    repo,
    sha,
    *,
    tokenizer = True,
    ref = False,
):
    folder = cache / ("models--" + repo.replace("/", "--"))
    snap = folder / "snapshots" / sha
    (snap / "scheduler").mkdir(parents = True)
    (snap / "model_index.json").write_text(json.dumps(INDEX))
    (snap / "scheduler" / "scheduler_config.json").write_text("{}")
    if tokenizer:
        (snap / "tokenizer").mkdir()
        (snap / "tokenizer" / "tokenizer_config.json").write_text("{}")
        (snap / "tokenizer" / "tokenizer.json").write_text('{"model": {}}')
    if ref:
        (folder / "refs").mkdir(parents = True)
        (folder / "refs" / "main").write_text(sha)
    return snap


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from core.inference import hub_reachability
    from core.inference import video as V

    monkeypatch.setattr(V, "hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(hub_reachability, "hub_unreachable", lambda *a, **k: True)
    return tmp_path


def _family():
    from core.inference.video_families import detect_video_family
    return detect_video_family("wan2.2-ti2v-5b")


def test_the_one_pinned_snapshot_without_refs_main_is_used(cache):
    from core.inference import video as V

    _snapshot(cache, MIRROR, SHA)
    assert V._video_offline_pinned_snapshot(UPSTREAM, MIRROR) == (MIRROR, SHA)
    assert V._video_offline_pinned_snapshot(UPSTREAM) == (
        MIRROR,
        SHA,
    )  # found through the mirror pair


def test_with_the_pin_the_cached_check_accepts_the_snapshot(cache):
    """main passes revision=None here: refs/main is missing, the manifest is not found, the load goes online."""
    from core.inference import video as V

    _snapshot(cache, MIRROR, SHA)
    fam = _family()
    assert not V._video_cached_while_hub_unreachable(
        None, MIRROR, None, None, revision = None, family = fam
    )
    repo, sha = V._video_offline_pinned_snapshot(UPSTREAM, MIRROR)
    assert V._video_cached_while_hub_unreachable(None, repo, None, None, revision = sha, family = fam)


def test_an_incomplete_pinned_snapshot_is_still_refused(cache):
    from core.inference import video as V

    _snapshot(cache, MIRROR, SHA, tokenizer = False)
    repo, sha = V._video_offline_pinned_snapshot(UPSTREAM, MIRROR)
    assert not V._video_cached_while_hub_unreachable(
        None, repo, None, None, revision = sha, family = _family()
    )


def test_a_repo_with_refs_main_resolves_as_before(cache):
    from core.inference import video as V
    _snapshot(cache, MIRROR, SHA, ref = True)
    assert V._video_offline_pinned_snapshot(UPSTREAM, MIRROR) is None


def test_several_snapshots_are_ambiguous(cache):
    from core.inference import video as V

    _snapshot(cache, MIRROR, SHA)
    _snapshot(cache, MIRROR, "0" * 40)
    assert V._video_offline_pinned_snapshot(UPSTREAM, MIRROR) is None


def test_both_sides_of_the_pair_cached_without_refs_is_ambiguous(cache):
    from core.inference import video as V

    _snapshot(cache, MIRROR, SHA)
    _snapshot(cache, UPSTREAM, "1" * 40)
    assert V._video_offline_pinned_snapshot(UPSTREAM, MIRROR) is None


def test_nothing_cached_or_a_local_path(cache, tmp_path):
    from core.inference import video as V
    assert V._video_offline_pinned_snapshot(UPSTREAM, MIRROR) is None
    assert V._video_offline_pinned_snapshot(str(tmp_path), None) is None


def test_the_offline_snapshot_lookup_needs_the_commit(cache):
    """The mechanism, on huggingface_hub itself: local_files_only with no revision looks for refs/main."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    snap = _snapshot(cache, MIRROR, SHA)
    with pytest.raises(LocalEntryNotFoundError):
        snapshot_download(MIRROR, cache_dir = str(cache), local_files_only = True)
    assert snapshot_download(
        MIRROR, cache_dir = str(cache), local_files_only = True, revision = SHA
    ) == str(snap)
