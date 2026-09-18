# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model library registration, default derivation, download targeting and the
move-between-libraries path."""

import os
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.services.models import libraries
from hub.storage import model_libraries as ml_storage
from hub.utils import download_manifest, hf_cache_state
from hub.utils.state_dir import cache_scope_name
from utils import hf_cache_settings


@pytest.fixture(autouse = True)
def _env_free_cache(monkeypatch):
    """Keep the active cache home under the test's control: never inherit a
    real ``HF_HOME`` from the host machine."""
    monkeypatch.setattr(hf_cache_settings, "_environment_paths", lambda: None)
    yield


def _register_library(
    tmp_path,
    name,
    label = None,
):
    home = tmp_path / name
    home.mkdir(parents = True)
    return ml_storage.add_model_library(str(home), label = label)


def _resolved(tmp_path, name):
    return Path(str(tmp_path / name)).resolve()


def test_add_registers_and_validates_library(tmp_path):
    row = _register_library(tmp_path, "libA", label = "Games")
    assert row["id"] is not None
    assert row["path"] == str(_resolved(tmp_path, "libA"))
    assert row["label"] == "Games"
    assert (tmp_path / "libA" / "hub").is_dir() and (tmp_path / "libA" / "xet").is_dir()

    listed = {r["path"] for r in ml_storage.list_model_libraries()}
    assert str(_resolved(tmp_path, "libA")) in listed


def test_add_duplicate_returns_existing_row(tmp_path):
    first = _register_library(tmp_path, "libA")
    again = ml_storage.add_model_library(str(tmp_path / "libA"), label = "Renamed")
    assert again["id"] == first["id"]
    assert len(ml_storage.list_model_libraries()) == 1
    # A label change on a duplicate is ignored: registration is idempotent.
    assert again["label"] != "Renamed"


@pytest.mark.skipif(platform.system() != "Windows", reason = "NOCASE dedupe is Windows-only")
def test_windows_case_insensitive_dedupe(tmp_path):
    first = _register_library(tmp_path, "libA")
    upper = str(tmp_path / "libA").upper()
    again = ml_storage.add_model_library(upper)
    assert again["id"] == first["id"]
    assert len(ml_storage.list_model_libraries()) == 1


def test_add_rejects_invalid_locations(tmp_path):
    with pytest.raises(ValueError, match = "Choose a library folder"):
        ml_storage.add_model_library("   ")

    with pytest.raises(ValueError, match = "library folder must be an absolute path"):
        ml_storage.add_model_library("relative/folder")

    with pytest.raises(ValueError, match = "Credential or config folders"):
        sensitive = tmp_path / ".ssh" / "models"
        sensitive.mkdir(parents = True)
        ml_storage.add_model_library(str(sensitive))


def test_remove_library(tmp_path):
    row = _register_library(tmp_path, "libA")
    assert ml_storage.remove_model_library(row["id"]) is True
    assert ml_storage.remove_model_library(row["id"]) is False
    assert ml_storage.list_model_libraries() == []


def test_remove_refuses_to_delete_a_registered_duplicate_of_default(tmp_path):
    """Registering the active cache home as a row must not let it be removed."""
    active = _resolved(tmp_path, "default-home")
    active.mkdir(parents = True)
    hf_cache_settings.set_hf_cache_home(str(active))
    row = ml_storage.add_model_library(str(active))
    with pytest.raises(ValueError, match = "default library cannot be removed"):
        ml_storage.remove_model_library(row["id"])
    assert [r["id"] for r in ml_storage.list_model_libraries()] == [row["id"]]


def test_set_default_promotes_library(tmp_path):
    row = _register_library(tmp_path, "libE")
    assert str(hf_cache_settings.get_hf_cache_paths().cache_home) != str(
        _resolved(tmp_path, "libE")
    )
    libraries.set_default_library_response(row["id"])
    assert str(hf_cache_settings.get_hf_cache_paths().cache_home) == str(
        _resolved(tmp_path, "libE")
    )


def test_list_response_derives_default_and_dedupes(tmp_path):
    active = _resolved(tmp_path, "default-home")
    active.mkdir(parents = True)
    hf_cache_settings.set_hf_cache_home(str(active))
    extra = _register_library(tmp_path, "libB")
    ml_storage.add_model_library(str(active))

    entries = libraries.list_libraries_response()["libraries"]
    by_id = {e["id"]: e for e in entries}
    assert by_id[None]["is_default"] is True
    assert by_id[None]["path"] == str(active)
    assert by_id[extra["id"]]["is_default"] is False
    assert {e["path"] for e in entries} == {str(active), str(_resolved(tmp_path, "libB"))}
    for entry in entries:
        assert entry["free_bytes"] is None or entry["free_bytes"] >= 0
        assert Path(entry["hub_cache"]).name == "hub"


def test_library_homes_feed_cache_scans(tmp_path):
    row = _register_library(tmp_path, "libC")
    home = Path(row["path"])

    assert home in hf_cache_settings.known_hf_cache_homes()
    assert home / "hub" in hf_cache_settings.known_hf_hub_caches()
    roots = {str(p) for p in hf_cache_state.hf_cache_roots()}
    assert str(home / "hub") in roots


def test_library_cache_paths_resolution(tmp_path):
    _register_library(tmp_path, "libD")
    row = ml_storage.list_model_libraries()[0]

    default_paths = libraries.library_cache_paths(None)
    assert default_paths.cache_home == hf_cache_settings.get_hf_cache_paths().cache_home
    assert libraries.library_cache_paths("") == default_paths
    assert libraries.library_cache_paths("default") == default_paths

    selected = libraries.library_cache_paths(str(row["id"]))
    assert selected.source == "studio"
    assert selected.cache_home == Path(row["path"])
    assert selected.hub_cache == Path(row["path"]) / "hub"
    assert selected.xet_cache == Path(row["path"]) / "xet"

    with pytest.raises(HTTPException) as exc:
        libraries.library_cache_paths("999999")
    assert exc.value.status_code == 404


def _write_repo(hub: Path, repo_id: str):
    repo_dir = hub / hf_cache_state.repo_cache_dir_name("model", repo_id)
    (repo_dir / "blobs").mkdir(parents = True)
    (repo_dir / "blobs" / "aa").write_bytes(b"payload")
    return repo_dir


def test_move_repo_between_libraries(tmp_path):
    a = _register_library(tmp_path, "libA")
    b = _register_library(tmp_path, "libB")
    a_hub = Path(a["path"]) / "hub"
    b_hub = Path(b["path"]) / "hub"

    repo_id = "Unsloth/Test-X"
    _write_repo(a_hub, repo_id)
    assert (a_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)).exists()
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        None,
        [download_manifest.ExpectedFile("blobs/aa", 7)],
        hub_cache = a_hub,
    )
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        "q4_k_m",
        [download_manifest.ExpectedFile("model-q4.gguf", 7)],
        hub_cache = a_hub,
    )
    assert download_manifest.read_manifest("model", repo_id, "q4_k_m", hub_cache = a_hub) is not None

    response = libraries.move_model_response(repo_id, None, str(b["id"]))
    assert response["ok"] is True
    assert response["target_home"] == str(Path(b["path"]))
    assert (b_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)).exists()
    assert not (a_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)).exists()
    assert response["already_in_library"] is False
    assert download_manifest.read_manifest("model", repo_id, None, hub_cache = a_hub) is None
    assert download_manifest.read_manifest("model", repo_id, "q4_k_m", hub_cache = a_hub) is None
    assert download_manifest.read_manifest("model", repo_id, None, hub_cache = b_hub) is None


def test_move_into_same_library_is_noop(tmp_path):
    a = _register_library(tmp_path, "libA")
    a_hub = Path(a["path"]) / "hub"
    repo_id = "Unsloth/Same"
    _write_repo(a_hub, repo_id)

    response = libraries.move_model_response(repo_id, None, str(a["id"]))
    assert response["already_in_library"] is True
    assert (a_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)).exists()


def test_move_missing_model_raises_404(tmp_path):
    a = _register_library(tmp_path, "libA")
    with pytest.raises(HTTPException) as exc:
        libraries.move_model_response("Unsloth/None", None, str(a["id"]))
    assert exc.value.status_code == 404


def test_move_rejects_variant_scoped_requests(tmp_path):
    """Variant-scoped moves are rejected; moving is whole-repo only."""
    a = _register_library(tmp_path, "libA")
    b = _register_library(tmp_path, "libB")
    repo_id = "Unsloth/Variant"
    a_repo = _write_repo(Path(a["path"]) / "hub", repo_id)
    with pytest.raises(HTTPException) as exc:
        libraries.move_model_response(repo_id, "q4_k_m", str(b["id"]))
    assert exc.value.status_code == 400
    assert a_repo.exists()


def test_move_skips_xet_tree_of_source_library(tmp_path):
    """The hub/ dir moves; the library's shared xet/ store stays put."""
    a = _register_library(tmp_path, "libA")
    b = _register_library(tmp_path, "libB")
    a_hub = Path(a["path"]) / "hub"
    b_hub = Path(b["path"]) / "hub"
    repo_id = "Unsloth/Xet"
    a_repo = _write_repo(a_hub, repo_id)
    source_xet = Path(a["path"]) / "xet"
    (source_xet / "cas").mkdir(parents = True)
    (source_xet / "cas" / "chunk-1").write_bytes(b"xet-chunk")

    response = libraries.move_model_response(repo_id, None, str(b["id"]))
    assert response["ok"] is True
    assert response["note"] is not None
    assert "xet" in response["note"]
    assert (b_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)).exists()
    assert not a_repo.exists()
    assert (source_xet / "cas" / "chunk-1").read_bytes() == b"xet-chunk"
    assert not (Path(b["path"]) / "xet" / "cas" / "chunk-1").exists()


def test_move_failure_keeps_source_intact_and_cleans_partial(monkeypatch, tmp_path):
    """A failed move 500s, keeps the intact source, and removes the partial copy."""
    a = _register_library(tmp_path, "libA")
    b = _register_library(tmp_path, "libB")
    a_hub = Path(a["path"]) / "hub"
    b_hub = Path(b["path"]) / "hub"
    repo_id = "Unsloth/Fails"
    repo_dir = _write_repo(a_hub, repo_id)
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        None,
        [download_manifest.ExpectedFile("blobs/aa", 7)],
        hub_cache = a_hub,
    )

    def _failing_cross_volume_move(src, dst):
        Path(dst).mkdir(parents = True, exist_ok = True)
        (Path(dst) / "blobs").mkdir(parents = True, exist_ok = True)
        (Path(dst) / "blobs" / "zz").write_bytes(b"half-copy")
        raise OSError("cross-volume copy failed mid-way")

    monkeypatch.setattr(shutil, "move", _failing_cross_volume_move)
    dest = b_hub / hf_cache_state.repo_cache_dir_name("model", repo_id)

    with pytest.raises(HTTPException) as exc:
        libraries.move_model_response(repo_id, None, str(b["id"]))
    assert exc.value.status_code == 500

    assert (repo_dir / "blobs" / "aa").read_bytes() == b"payload"
    assert download_manifest.read_manifest("model", repo_id, None, hub_cache = a_hub) is not None
    assert not dest.exists()


def test_non_default_library_scopes_env_and_manifest_state(tmp_path):
    """Library targeting resolves and writes to that library's own cache scope."""
    default = hf_cache_settings.get_hf_cache_paths()
    row = _register_library(tmp_path, "libD")
    selected = libraries.library_cache_paths(str(row["id"]))

    assert selected.source == "studio"
    assert selected.hub_cache != default.hub_cache
    assert selected.xet_cache != default.xet_cache

    env = selected.child_env({})
    assert env["HF_HUB_CACHE"] == str(selected.hub_cache)
    assert env["HF_XET_CACHE"] == str(selected.xet_cache)
    assert env["HF_HUB_CACHE"] != str(default.hub_cache)

    assert cache_scope_name(selected.hub_cache) != cache_scope_name(default.hub_cache)
    repo_id = "Unsloth/Scoped"
    variant = "q4_k_m"
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        variant,
        [download_manifest.ExpectedFile("model-q4.gguf", 4)],
        hub_cache = selected.hub_cache,
    )
    assert (
        download_manifest.read_manifest("model", repo_id, variant, hub_cache = selected.hub_cache)
        is not None
    )
    assert (
        download_manifest.read_manifest("model", repo_id, variant, hub_cache = default.hub_cache)
        is None
    )


def test_known_hf_cache_homes_logs_library_probe_failure(monkeypatch, tmp_path):
    """A dead library volume is logged, not silently skipped."""
    recorded = []
    monkeypatch.setattr(
        hf_cache_settings,
        "logger",
        SimpleNamespace(debug = lambda msg, **kwargs: recorded.append(msg)),
    )

    def _boom():
        raise RuntimeError("dead volume")

    monkeypatch.setattr(ml_storage, "model_library_homes", _boom)
    hf_cache_settings.known_hf_cache_homes()
    assert recorded


def test_add_and_remove_library_service_mapping(tmp_path):
    with pytest.raises(HTTPException) as exc:
        libraries.add_library_response("")
    assert exc.value.status_code == 400

    assert libraries.remove_library_response(999999) == {"ok": False}
