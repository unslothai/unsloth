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

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

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
    # Same repo, two jobs: the scoped one must not adopt or overwrite the full snapshot's
    # manifest, or the repo would read as partial against expectations it never had.
    full = dl._download_job_key("black-forest-labs/FLUX.1-dev", None)
    scoped = dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion"))
    assert full != scoped
    assert scoped.endswith("@diffusion")
    # It rides the variant slot, so it must satisfy the same validator.
    assert is_valid_gguf_variant("@diffusion")
    # The "@" prefix is what keeps a scope out of the quant namespace: a job scoped
    # "diffusion" and a (hypothetical) quant named "diffusion" stay distinct.
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
    # The scope key carries a digest of the requested file set (see _scope_variant).
    scope_variant = dl._scope_variant("diffusion", FILES)
    assert result["job_key"].endswith(scope_variant)

    args = spawned["args"]
    assert "--variant" in args and args[args.index("--variant") + 1] == scope_variant
    # The file list travels in a temp JSON file, not argv: a pipeline repo lists hundreds.
    manifest_path = args[args.index("--files-json") + 1]
    assert json.loads(Path(manifest_path).read_text(encoding = "utf-8")) == FILES
    Path(manifest_path).unlink(missing_ok = True)


def test_scoped_files_survive_into_the_registry(monkeypatch):
    # The XET -> HTTP retry rebuilds worker args from registry metadata alone. Without the
    # file list there, a retried scoped job would silently become a full snapshot.
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
        dl._download_job_key("black-forest-labs/FLUX.1-dev", dl._scope_variant("diffusion", FILES))
    )
    assert metadata is not None and list(metadata.scoped_files) == FILES


def test_files_manifest_round_trips():
    path = download_lifecycle.write_files_manifest(FILES)
    try:
        assert json.loads(Path(path).read_text(encoding = "utf-8")) == FILES
    finally:
        Path(path).unlink(missing_ok = True)


def test_scope_keys_differ_per_requested_file_set():
    # Two quants of one repo are two different downloads. Sharing "@diffusion" made the
    # second request adopt the first job, so the UI waited on the wrong file set and then
    # loaded a file that had never been fetched.
    a = dl._scope_variant("diffusion", ["flux1-dev-Q4_K_M.gguf", "ae.safetensors"])
    b = dl._scope_variant("diffusion", ["flux1-dev-Q2_K.gguf", "ae.safetensors"])
    assert a != b
    assert a.startswith("@diffusion-") and b.startswith("@diffusion-")
    # Order and duplicates must not change the identity (the same set is the same job).
    assert a == dl._scope_variant(
        "diffusion", ["ae.safetensors", "flux1-dev-Q4_K_M.gguf", "flux1-dev-Q4_K_M.gguf"]
    )
    # A scope with no files keeps the bare form, and stays valid as a variant slot.
    from hub.utils.paths import is_valid_gguf_variant

    assert dl._scope_variant("video", []) == "@video"
    assert is_valid_gguf_variant(a) and is_valid_gguf_variant("@video")
