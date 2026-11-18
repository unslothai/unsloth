# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A refs/main that is not valid UTF-8 reads like a missing ref in every cache reader (#17672 fixed the scan):
UnicodeDecodeError is a ValueError, so `except OSError` let it out past each reader's fallback."""

from __future__ import annotations


from pathlib import Path

import pytest

SHA = "a" * 40
BAD = b"\xff\xfe\x80\x81 not utf-8"


def _repo(root: Path, repo_id: str, ref: bytes | None) -> Path:
    repo_dir = root / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / SHA
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    if ref is not None:
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / "main").write_bytes(ref)
    return snapshot


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_audio_cpp_snapshot_dirs(tmp_path, ref):
    from core.inference.audio_cpp_models import snapshot_dirs
    snapshot = _repo(tmp_path, "org/model", ref)
    assert snapshot_dirs("org/model", tmp_path) == [snapshot]


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_stt_fallback_revisions(tmp_path, ref):
    from core.inference.stt_sidecar import _fallback_revisions
    _repo(tmp_path, "org/model", ref)
    assert _fallback_revisions("org/model", hub_cache = tmp_path) == [SHA]


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_media_locality_cached_snapshot_root(tmp_path, monkeypatch, ref):
    import core.inference.diffusion as diffusion
    from core.inference import media_locality

    monkeypatch.setattr(diffusion, "hub_cache_dir", lambda: str(tmp_path))
    snapshot = _repo(tmp_path, "org/model", ref)
    assert media_locality._cached_snapshot_root("org/model") == snapshot


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_retired_copy_loads_from_disk(tmp_path, monkeypatch, ref):
    import utils.hf_cache_settings as settings
    from core.inference import diffusion_families

    monkeypatch.setattr(settings, "active_hf_hub_cache", lambda: str(tmp_path))
    _repo(tmp_path, "unsloth/flux-3-image-klein", ref)
    assert diffusion_families.retired_copy_loads_from_disk("unsloth/flux-3-image-klein") is False


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_stt_complete_snapshot_in(tmp_path, monkeypatch, ref):
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "_snapshot_usable", lambda model_id, snapshot: True)
    monkeypatch.setattr(stt_sidecar, "_write_revision_record", lambda *a, **k: None)
    snapshot = _repo(tmp_path, "org/model", ref)
    found = stt_sidecar._complete_snapshot_in("some-model", "org/model", None, tmp_path)
    assert found is not None and Path(found).resolve() == snapshot.resolve()


@pytest.mark.parametrize("ref", [None, BAD], ids = ["missing_ref", "undecodable_ref"])
def test_video_planned_bytes(tmp_path, monkeypatch, ref):
    from core.inference import video

    monkeypatch.setattr(video.VideoBackend, "_plan_cache_roots", staticmethod(lambda: [tmp_path]))
    _repo(tmp_path, "org/model", ref)
    fields = {f: None for f in getattr(video._PlannedFetch, "__dataclass_fields__", {})}
    fields.update(repo = "org/model", revision = None, files = (("config.json", "", 2),))
    entry = video._PlannedFetch(**fields)
    # load_progress polls this during a video load
    assert video.VideoBackend._planned_bytes({"k": entry}) == 0
