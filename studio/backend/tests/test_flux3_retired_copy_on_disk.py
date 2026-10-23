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


def test_its_root_transformer_pick_stays_on_the_snapshot(tmp_path, offline_hub):
    snapshot = str(_snapshot(tmp_path))
    request = DiffusionLoadRequest(
        model_path = snapshot,
        display_repo_id = _COPY,
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.gguf_filename, request.model_kind) == (
        snapshot,
        "model.safetensors",
        "single_file",
    )


def _companion_base(request):
    """The companion base the single-file loader assembles the pipeline around (``_resolve_base_repo``, no card tag
    offline), as the repo it fetches from."""
    from core.inference import diffusion
    fam = next(f for f in df._FAMILIES if f.name == "flux.3-klein")
    with patch.object(diffusion, "_hf_base_model", return_value = None):
        resolved = diffusion._resolve_base_repo(
            request.model_path, request.base_repo, fam, None, request.gguf_filename
        )
    return df.prefer_ungated_mirror(resolved)


@pytest.mark.parametrize("by_id", [False, True])
def test_its_root_transformer_pick_takes_companions_from_the_copy(tmp_path, offline_hub, by_id):
    """The root transformer pick of a complete copy, with no explicit base, assembles around the copy itself: the
    family default would be the base's -FP8 mirror, absent offline and a second download online."""
    snapshot = str(_snapshot(tmp_path))
    source = _COPY if by_id else snapshot
    request = DiffusionLoadRequest(
        model_path = source,
        display_repo_id = None if by_id else _COPY,
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.base_repo) == (source, source)
    fetched = _companion_base(request)
    assert fetched == source
    if not by_id:
        from core.inference import diffusion
        from huggingface_hub.errors import LocalEntryNotFoundError

        # The route's preflight accepts the snapshot as a companion base (what validate_load_request runs on it).
        diffusion._assert_local_base_is_pipeline(
            request.base_repo,
            modular_only = True,
            excluded_components = ("transformer",),
            fixed_component_folders = flux3.FLUX3_COMPONENT_FOLDERS,
        )

        with patch(
            "huggingface_hub.hf_hub_download", side_effect = LocalEntryNotFoundError("offline")
        ) as hub:
            assert flux3._read_index(fetched, None, True) == {"_class_name": "Flux3ModularPipeline"}
        assert hub.call_count == 0


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


def test_a_root_transformer_pick_missing_its_file_loads_through_its_base(tmp_path, offline_hub):
    snapshot = _snapshot(tmp_path)
    (snapshot / "model.safetensors").unlink()
    request = DiffusionLoadRequest(
        model_path = str(snapshot),
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.gguf_filename, request.model_kind) == (
        _BASE,
        None,
        "pipeline",
    )


def test_a_copy_not_on_disk_loads_through_its_base(offline_hub):
    assert DiffusionLoadRequest(model_path = _COPY).model_path == _BASE
    assert (
        DiffusionLoadRequest(
            model_path = "/no/such/hub/models--unsloth--flux-3-image-klein/snapshots/abc"
        ).model_path
        == _BASE
    )


@pytest.mark.parametrize("by_id", [False, True])
def test_a_root_transformer_pick_needs_only_the_transformer_config(tmp_path, offline_hub, by_id):
    """The root model.safetensors supplies the transformer, so a copy without transformer/ weights (config only) is a
    complete single-file assembly and stays on the copy; its pipeline pick, which opens those weights, still does not."""
    snapshot = str(_snapshot(tmp_path, complete = False))
    source = _COPY if by_id else snapshot
    request = DiffusionLoadRequest(
        model_path = source,
        display_repo_id = None if by_id else _COPY,
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.gguf_filename, request.base_repo) == (
        source,
        "model.safetensors",
        source,
    )
    assert df.retired_copy_loads_from_disk(source, "model.safetensors")
    assert not df.retired_copy_loads_from_disk(source)


def test_the_copy_named_by_id_needs_its_main_ref(tmp_path, offline_hub):
    """A load by id resolves refs/main, so a lone commit-pinned snapshot with no main ref cannot serve it offline."""
    snapshot = _snapshot(tmp_path)
    (snapshot.parent.parent / "refs" / "main").unlink()
    assert not df.retired_copy_loads_from_disk(_COPY)
    assert DiffusionLoadRequest(model_path = _COPY).model_path == _BASE
    # Its path still loads from disk: a snapshot path needs no ref.
    assert df.retired_copy_loads_from_disk(str(snapshot))
