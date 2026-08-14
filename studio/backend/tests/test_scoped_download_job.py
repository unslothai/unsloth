# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoped download flavour: fetch an explicit file list through the normal download
manager, so the Images/Video pages stage models the same way Chat and the Hub do.

A diffusion load reads a deliberate subset of a repo (no packaged root single, no
transformer/ shards, no fp16 twins), so a plain snapshot would pull tens of GB it never
opens. These cover the scoping, the separate job key, and the XET -> HTTP retry.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import HTTPException

from hub.schemas.downloads import DownloadModelRequest
from hub.services import download_lifecycle
from hub.services.models import downloads as dl
from hub.utils.paths import is_valid_gguf_variant


FILES = ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]


def _request(**over) -> DownloadModelRequest:
    body = {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "scope_id": "diffusion",
        "files": list(FILES),
        "use_xet": False,
    }
    body.update(over)
    return DownloadModelRequest(**body)


def test_scope_keys_apart_from_the_full_snapshot():
    # Same repo, two jobs: the scoped one must not adopt or overwrite the full snapshot's manifest, or the repo reads as partial against expectations it never had.
    full = dl._download_job_key("black-forest-labs/FLUX.1-dev", None)
    scoped = dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    assert full != scoped
    assert scoped.endswith("@diffusion")
    # It rides the variant slot, so it must satisfy the same validator.
    assert is_valid_gguf_variant("@diffusion")
    # The "@" prefix keeps a scope out of the quant namespace: a job scoped "diffusion" and a quant named "diffusion" stay distinct.
    assert dl._download_job_key("org/m", "diffusion") != dl._download_job_key(
        "org/m", dl._scope_variant("diffusion")
    )


def test_scope_requires_files_and_rejects_a_variant(monkeypatch):
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)

    with pytest.raises(Exception) as no_files:
        asyncio.run(dl.download_model_response(_request(files = [])))
    assert "files" in str(no_files.value)

    with pytest.raises(Exception) as both:
        asyncio.run(dl.download_model_response(_request(gguf_variant = "Q4_K_M")))
    assert "mutually exclusive" in str(both.value)


def test_scoped_start_spawns_a_file_scoped_worker(monkeypatch):
    spawned: dict = {}

    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(dl, "scoped_file_blob_hashes", lambda *a, **k: frozenset({"h1"}))

    def _fake_launch(registry, key, *, spawn, **kwargs):
        spawn()
        return "running"

    def _fake_spawn(args, hf_token, **kwargs):
        spawned["args"] = args
        return object()

    monkeypatch.setattr(download_lifecycle, "launch_worker", _fake_launch)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", _fake_spawn)

    result = asyncio.run(dl.download_model_response(_request()))
    assert result["accepted"] is True
    scope_variant = dl._scope_variant("diffusion")
    assert result["job_key"].endswith(scope_variant)

    args = spawned["args"]
    assert "--variant" in args and args[args.index("--variant") + 1] == scope_variant
    # The file list travels in a temp JSON file, not argv: a pipeline repo lists hundreds.
    manifest_path = args[args.index("--files-json") + 1]
    assert json.loads(Path(manifest_path).read_text(encoding = "utf-8")) == FILES
    Path(manifest_path).unlink(missing_ok = True)


def test_scoped_files_survive_into_the_registry(monkeypatch):
    # The XET to HTTP retry rebuilds worker args from registry metadata alone, so without the file list there a retried scoped job would become a full snapshot.
    captured: dict = {}
    real_claim = dl._registry.claim

    def _spy_claim(key, transport, **kwargs):
        captured.update(kwargs)
        return real_claim(key, transport, **kwargs)

    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(dl, "scoped_file_blob_hashes", lambda *a, **k: frozenset())
    monkeypatch.setattr(dl._registry, "claim", _spy_claim)
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    asyncio.run(dl.download_model_response(_request()))
    assert captured["scoped_files"] == FILES

    metadata = dl._registry.get_job_metadata(
        dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    )
    assert metadata is not None and list(metadata.scoped_files) == FILES


def test_files_manifest_round_trips():
    path = download_lifecycle.write_files_manifest(FILES)
    try:
        assert json.loads(Path(path).read_text(encoding = "utf-8")) == FILES
    finally:
        Path(path).unlink(missing_ok = True)


def test_a_different_file_set_is_not_adopted(monkeypatch):
    # Two quants of one repo are two downloads sharing the "@diffusion" slot. Adopting the running one made the UI wait on the
    # wrong file set and load a file that was never fetched, so the second request is refused while the first runs.
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(dl, "scoped_file_blob_hashes", lambda *a, **k: frozenset())
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    key = dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    try:
        first = asyncio.run(dl.download_model_response(_request()))
        assert first["accepted"] is True

        with pytest.raises(HTTPException) as other_files:
            asyncio.run(
                dl.download_model_response(
                    _request(files = ["model_index.json", "flux1-dev-Q2_K.gguf"])
                )
            )
        assert other_files.value.status_code == 409
        assert "different" in other_files.value.detail

        # The same file set is still the same download: it adopts the live job as before, in any order and with duplicates collapsed.
        same = asyncio.run(
            dl.download_model_response(_request(files = [FILES[1], FILES[0], FILES[0]]))
        )
        assert same["accepted"] is True and same["job_key"] == key
    finally:
        dl._registry.set_job(key, "complete")


def test_the_http_retry_keeps_the_scoped_file_list_on_the_record(monkeypatch):
    # The retry reclaims the slot with replace_active, which OVERWRITES the stored metadata. Dropping the file list there left
    # the record claiming an empty scope, so the next identical scoped start compared [] against the real list and 409'd.
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(dl, "scoped_file_blob_hashes", lambda *a, **k: frozenset())
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(download_lifecycle, "spawn_worker", lambda *a, **k: _Proc())
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)

    key = dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    try:
        # The retry only exists for a job that started on XET.
        assert asyncio.run(dl.download_model_response(_request(use_xet = True)))["accepted"] is True

        retried = download_lifecycle._try_http_retry(
            dl._registry,
            key,
            hf_token = None,
            label = "FLUX.1-dev [@diffusion]",
            log_prefix = "[test]",
            logger = download_lifecycle.logging.getLogger("test"),
            repo_type = "model",
            repo_id = "black-forest-labs/FLUX.1-dev",
            watch_name = "test",
        )
        assert retried is True

        metadata = dl._registry.get_job_metadata(key)
        assert metadata is not None and list(metadata.scoped_files) == FILES
        # And the retried job is still adoptable by the page that asked for those files.
        again = asyncio.run(dl.download_model_response(_request()))
        assert again["accepted"] is True and again["job_key"] == key
    finally:
        dl._registry.set_job(key, "complete")


def test_scope_key_stays_derivable_from_the_scope_alone():
    # The download manager builds this key client-side (it polls and cancels before any server round-trip), so the scope name alone must produce it.
    assert dl._scope_variant("diffusion") == "@diffusion"
    assert dl._scope_variant("video") == "@video"
    assert dl._scope_variant("  ") is None
    assert is_valid_gguf_variant("@diffusion")


def _fake_backend(*loading: str):
    return SimpleNamespace(loading_repo_ids = lambda: tuple(loading))


def test_an_images_load_staging_a_repo_blocks_a_download_of_it(monkeypatch):
    # The Images and Video backends stage their snapshots through the same HF cache as the download worker, so starting a
    # download for a repo one of them is fetching puts two writers on the same blobs. Chat was guarded; these were not.
    from core.inference import diffusion_engine_router, video as video_backend

    monkeypatch.setattr(
        diffusion_engine_router,
        "get_active_diffusion_engine",
        lambda: _fake_backend("Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo-GGUF"),
    )
    monkeypatch.setattr(video_backend, "get_video_backend", lambda: _fake_backend())

    # Both the checkpoint and the companion base it is pulling are covered, case-insensitively (the repo id arrives as the user typed it).
    assert dl._load_in_flight("Tongyi-MAI/Z-Image-Turbo") is True
    assert dl._load_in_flight("tongyi-mai/z-image-turbo") is True
    assert dl._load_in_flight("unsloth/Z-Image-Turbo-GGUF") is True
    assert dl._load_in_flight("Org/Unrelated") is False


def test_a_video_load_staging_a_repo_blocks_a_download_of_it(monkeypatch):
    from core.inference import diffusion_engine_router, video as video_backend

    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", _fake_backend)
    monkeypatch.setattr(
        video_backend,
        "get_video_backend",
        lambda: _fake_backend("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    )

    assert dl._load_in_flight("Wan-AI/Wan2.2-TI2V-5B-Diffusers") is True
    assert dl._load_in_flight("Org/Unrelated") is False


def test_an_unavailable_backend_never_blocks_a_download(monkeypatch):
    # Fail open: a probe that raises must not make the repo undownloadable.
    from core.inference import diffusion_engine_router, video as video_backend

    def _boom():
        raise RuntimeError("no engine")

    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", _boom)
    monkeypatch.setattr(video_backend, "get_video_backend", _boom)

    assert dl._load_in_flight("Org/Anything") is False


def test_active_downloads_publish_the_scoped_file_list(monkeypatch):
    """An adopting client (a second browser profile, or a tab opened before the throttled state
    write) has no local record of what a live job is fetching. Every file set of one repo shares
    the "@scope" slot, so without this list it cannot tell its own transfer from a sibling
    checkpoint's and would report a never-fetched file as already downloading."""
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(dl, "scoped_file_blob_hashes", lambda *a, **k: frozenset())
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    key = dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    try:
        asyncio.run(dl.download_model_response(_request()))
        rows = download_lifecycle.active_download_refs(
            dl._registry, "black-forest-labs/FLUX.1-dev", with_variant = True
        )
        scoped = [r for r in rows if r.variant == "@diffusion"]
        assert scoped, f"no scoped row in {rows}"
        assert list(scoped[0].files or []) == FILES
    finally:
        dl._registry.set_job(key, "complete")


def test_a_full_snapshot_download_reports_no_file_list(monkeypatch):
    # Only a scoped job has a deliberate subset; a full snapshot must not claim one, or the client matches its whole-repo job against a scoped request.
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    key = dl._download_job_key("black-forest-labs/FLUX.1-dev", None)
    try:
        asyncio.run(
            dl.download_model_response(
                DownloadModelRequest(repo_id = "black-forest-labs/FLUX.1-dev", use_xet = False)
            )
        )
        rows = download_lifecycle.active_download_refs(
            dl._registry, "black-forest-labs/FLUX.1-dev", with_variant = True
        )
        full = [r for r in rows if r.variant is None]
        assert full, f"no full-snapshot row in {rows}"
        assert full[0].files is None
    finally:
        dl._registry.set_job(key, "complete")
