# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A retired FLUX.3 full copy that is already complete on disk keeps loading from disk. The copies were retired for
size and hold the base's own files, so redirecting one to its base would need the base's -FP8 mirror: offline that is
a LocalEntryNotFoundError for a model the install already holds, online the same weights downloaded again. An
incomplete copy still loads through the base."""

import json
from unittest.mock import patch

import pytest

from core.inference import diffusion_families as df
from core.inference import diffusion_flux3 as flux3
from models.inference import DiffusionLoadRequest

_COPY = "unsloth/flux-3-image-klein"
_BASE = "black-forest-labs/flux-3-image-klein"


def _write(path, data = b"x"):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)


def _flux3_pipeline(root, *, complete = True):
    """A FLUX.3 Modular Diffusers layout with every folder the loader reads (tiny placeholder weights)."""
    _write(root / flux3.FLUX3_INDEX, json.dumps({"_class_name": "Flux3ModularPipeline"}).encode())
    _write(root / "model.safetensors")
    _write(root / "text_encoder" / "config.json", b"{}")
    _write(root / "text_encoder" / "model.safetensors")
    _write(root / "text_encoder" / "processor_config.json", b"{}")
    _write(root / "text_encoder" / "tokenizer.json", b"{}")
    for folder in ("transformer", "vae"):
        _write(root / folder / "config.json", b"{}")
        if complete or folder == "vae":
            _write(root / folder / "diffusion_pytorch_model.safetensors")
    return root


def _snapshot(
    tmp_path,
    repo = _COPY,
    *,
    complete = True,
):
    root = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
    snapshot = _flux3_pipeline(root / "snapshots" / "0123abc", complete = complete)
    _write(root / "refs" / "main", b"0123abc")
    return snapshot


@pytest.fixture
def offline_hub(tmp_path, monkeypatch):
    """HF offline mode with a live hub cache under tmp_path; any Hub lookup is a cache miss."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path / "hub")
    )
    return tmp_path / "hub"


def _load_reads_local_index(request):
    """What the FLUX.3 loader does first with the request: resolve the fetch repo, then read its manifest (cache only)."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    fetched = df.prefer_ungated_mirror(request.model_path)
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect = LocalEntryNotFoundError("not in the offline cache"),
    ) as hub:
        try:
            index = flux3._read_index(fetched, None, True)
        except LocalEntryNotFoundError:
            index = None
    return index, hub.call_count


@pytest.mark.parametrize("display", [_COPY, None])
def test_a_complete_downloaded_copy_loads_offline_from_its_snapshot(tmp_path, offline_hub, display):
    snapshot = str(_snapshot(tmp_path))
    request = DiffusionLoadRequest(model_path = snapshot, display_repo_id = display)
    assert (request.model_path, request.display_repo_id) == (snapshot, display)
    index, hub_lookups = _load_reads_local_index(request)
    assert index == {"_class_name": "Flux3ModularPipeline"} and hub_lookups == 0


def test_an_explicit_base_on_the_root_transformer_pick_is_kept(tmp_path, offline_hub):
    snapshot = str(_snapshot(tmp_path))
    request = DiffusionLoadRequest(
        model_path = snapshot,
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
        base_repo = _BASE,
    )
    assert request.base_repo == _BASE


def test_the_copy_named_by_id_loads_from_a_complete_cached_snapshot(tmp_path, offline_hub):
    _snapshot(tmp_path)
    request = DiffusionLoadRequest(model_path = _COPY, model_kind = "pipeline", base_repo = _COPY)
    assert (request.model_path, request.base_repo) == (_COPY, _COPY)


def test_an_incomplete_copy_still_loads_through_its_base(tmp_path, offline_hub):
    snapshot = str(_snapshot(tmp_path, complete = False))
    request = DiffusionLoadRequest(model_path = snapshot, display_repo_id = _COPY)
    assert (request.model_path, request.display_repo_id) == (_BASE, None)
    # Named by id, with only that partial snapshot cached.
    assert DiffusionLoadRequest(model_path = _COPY).model_path == _BASE


def test_a_copy_not_on_disk_loads_through_its_base(offline_hub):
    assert DiffusionLoadRequest(model_path = _COPY).model_path == _BASE
    assert (
        DiffusionLoadRequest(
            model_path = "/no/such/hub/models--unsloth--flux-3-image-klein/snapshots/abc"
        ).model_path
        == _BASE
    )


def test_the_copy_named_by_id_needs_its_main_ref(tmp_path, offline_hub):
    """A load by id resolves refs/main, so a lone commit-pinned snapshot with no main ref cannot serve it offline."""
    snapshot = _snapshot(tmp_path)
    (snapshot.parent.parent / "refs" / "main").unlink()
    assert not df.retired_copy_loads_from_disk(_COPY)
    assert DiffusionLoadRequest(model_path = _COPY).model_path == _BASE
    # Its path still loads from disk: a snapshot path needs no ref.
    assert df.retired_copy_loads_from_disk(str(snapshot))


@pytest.mark.parametrize("by_id", [False, True])
def test_its_root_transformer_pick_loads_the_copy_as_its_pipeline(tmp_path, offline_hub, by_id):
    """The root model.safetensors is the copy's own transformer, so the pick loads the copy's pipeline from disk."""
    snapshot = str(_snapshot(tmp_path))
    source = _COPY if by_id else snapshot
    request = DiffusionLoadRequest(
        model_path = source,
        display_repo_id = None if by_id else _COPY,
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.gguf_filename, request.model_kind) == (
        source,
        None,
        "pipeline",
    )
    if not by_id:
        index, lookups = _load_reads_local_index(request)
        assert index == {"_class_name": "Flux3ModularPipeline"} and lookups == 0


def _hf_cache_snapshot(tmp_path):
    """A complete copy as huggingface_hub stores it: every snapshot file a symlink into blobs/."""
    import os

    snapshot = _snapshot(tmp_path)
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    for i, path in enumerate(sorted(p for p in snapshot.rglob("*") if p.is_file())):
        blob = blobs / f"blob{i}"
        path.replace(blob)
        path.symlink_to(os.path.relpath(blob, path.parent))
    return snapshot


@pytest.mark.parametrize("pick", ["pipeline", "root"])
def test_a_snapshot_in_the_hf_cache_layout_passes_the_route_validator(tmp_path, offline_hub, pick):
    """The root file of a cached snapshot resolves into blobs/, outside the snapshot, which a single-file pick of that
    path refuses ("gguf_filename must resolve to a file inside the repo"); as the pipeline load it validates."""
    from core.inference.diffusion import DiffusionBackend

    snapshot = str(_hf_cache_snapshot(tmp_path))
    single = (
        {"gguf_filename": "model.safetensors", "model_kind": "single_file"}
        if pick == "root"
        else {}
    )
    request = DiffusionLoadRequest(model_path = snapshot, display_repo_id = _COPY, **single)
    assert request.model_path == snapshot
    DiffusionBackend.__new__(DiffusionBackend).validate_load_request(
        request.model_path,
        gguf_filename = request.gguf_filename,
        model_kind = request.model_kind,
        base_repo = request.base_repo,
        family_override = "flux.3-klein",
    )


def test_a_root_transformer_pick_missing_its_file_loads_the_copys_pipeline(tmp_path, offline_hub):
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").unlink()
    request = DiffusionLoadRequest(
        model_path = str(snapshot), gguf_filename = "model.safetensors", model_kind = "single_file"
    )
    assert (request.model_path, request.gguf_filename, request.model_kind) == (
        str(snapshot),
        None,
        "pipeline",
    )


def test_a_copy_without_transformer_weights(tmp_path, offline_hub):
    """Its pipeline (and the root pick, loaded as that pipeline) goes through the base; as the companion base of
    another single file, which supplies the transformer, only the config is read, so it stays."""
    snapshot = str(_snapshot(tmp_path, complete = False))
    root = DiffusionLoadRequest(
        model_path = snapshot, gguf_filename = "model.safetensors", model_kind = "single_file"
    )
    assert (root.model_path, root.gguf_filename, root.model_kind) == (_BASE, None, "pipeline")
    gguf = DiffusionLoadRequest(
        model_path = "/models/klein.gguf", gguf_filename = "klein.gguf", base_repo = snapshot
    )
    assert gguf.base_repo == snapshot
    # A pipeline redirected to the base cannot use it as its base either: the pipeline opens the transformer weights.
    pipeline = DiffusionLoadRequest(model_path = snapshot, base_repo = snapshot, model_kind = "pipeline")
    assert (pipeline.model_path, pipeline.base_repo) == (_BASE, _BASE)


def test_the_auto_switch_loads_a_copy_in_an_earlier_models_folder_from_its_snapshot(
    tmp_path, offline_hub
):
    """The seeded-snapshot rewrite would load it by id, which searches only the active cache, so the copy kept in an
    earlier Models folder would be redirected to its base."""
    import asyncio

    from core.inference import media_auto_switch as mas

    snapshot = _snapshot(tmp_path / "earlier")
    pick = mas.MediaModelPick(_COPY, str(snapshot), None, "pipeline")
    sent = []

    async def _gated(request, *_args, **_kwargs):
        sent.append(request)

    with patch("routes.inference.load_diffusion_model_gated", _gated):
        asyncio.run(mas._start_load(mas.DIFFUSION, pick, "subject"))
    assert [(r.model_path, r.gguf_filename) for r in sent] == [(str(snapshot), None)]
