# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deleting a model Studio discovered rather than downloaded.

The interesting cases are all about what a delete leaves behind: an Ollama blob two tags share, a
publisher folder emptied of its last model, and the ``.studio_links`` symlink Studio itself made,
which nothing but this code path knows about.
"""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from hub.services.models import local_deletion, ollama


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(content)
    return path


def _ollama_store(root: Path) -> Path:
    (root / "manifests").mkdir(parents = True, exist_ok = True)
    (root / "blobs").mkdir(parents = True, exist_ok = True)
    return root


def _blob(root: Path, name: str, size: int) -> str:
    """Write a blob and return the digest that names it."""
    digest = f"sha256:{name}"
    _write(root / "blobs" / f"sha256-{name}", "b" * size)
    return digest


def _manifest(
    root: Path,
    repo: str,
    tag: str,
    *digests: str,
    config: str | None = None,
) -> Path:
    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / repo / tag
    body = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest}
            for digest in digests
        ]
    }
    if config is not None:
        body["config"] = {"digest": config}
    _write(tag_file, json.dumps(body))
    return tag_file


@pytest.fixture
def ollama_root(tmp_path, monkeypatch):
    root = _ollama_store(tmp_path / "ollama")
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(local_deletion, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(local_deletion, "lmstudio_model_dirs", lambda: [])
    # Nothing is loaded unless a test says so, and no backend is importable under the stubs.
    monkeypatch.setattr(local_deletion, "_loaded_identifiers", lambda: [])
    return root


@pytest.fixture
def scan_root(tmp_path, monkeypatch):
    root = tmp_path / "scanned"
    root.mkdir()
    monkeypatch.setattr(local_deletion, "lmstudio_model_dirs", lambda: [root])
    monkeypatch.setattr(local_deletion, "ollama_model_dirs", lambda: [])
    monkeypatch.setattr(local_deletion, "_loaded_identifiers", lambda: [])
    return root


def _ref(tag_file: Path) -> str:
    return ollama._ollama_manifest_ref(tag_file)


# --- Ollama -----------------------------------------------------------------------------------


def test_ollama_delete_removes_manifest_and_unshared_blobs(ollama_root):
    solo = _blob(ollama_root, "solo", 2048)
    tag = _manifest(ollama_root, "llama3", "8b", solo)

    result = local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert result["status"] == "deleted"
    assert result["display_name"] == "llama3:8b"
    assert not tag.exists()
    assert not (ollama_root / "blobs" / "sha256-solo").exists()
    assert result["freed_bytes"] >= 2048
    assert result["retained_bytes"] == 0


def test_ollama_delete_keeps_a_blob_another_tag_still_names(ollama_root):
    shared = _blob(ollama_root, "shared", 4096)
    own = _blob(ollama_root, "own", 1024)
    keep = _manifest(ollama_root, "llama3", "latest", shared)
    tag = _manifest(ollama_root, "llama3", "8b", shared, own)

    impact = local_deletion.local_delete_impact_blocking(_ref(tag), "ollama")
    assert impact["retained_bytes"] == 4096
    # Named, so the dialog can say WHICH tag is holding those bytes.
    assert impact["retained_for"] == ["llama3:latest"]

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert not tag.exists()
    assert keep.exists()
    assert (ollama_root / "blobs" / "sha256-shared").exists()
    assert not (ollama_root / "blobs" / "sha256-own").exists()


def test_ollama_delete_keeps_every_blob_when_a_sibling_manifest_is_unreadable(ollama_root):
    """Fail closed: an unparsed manifest may be the one that needs the blob."""
    own = _blob(ollama_root, "own", 512)
    tag = _manifest(ollama_root, "llama3", "8b", own)
    _write(ollama_root / "manifests" / "registry.ollama.ai" / "library" / "broken" / "v1", "{oops")

    result = local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert not tag.exists(), "the tag itself still goes, so the row leaves the inventory"
    assert (ollama_root / "blobs" / "sha256-own").exists()
    assert result["retained_bytes"] == 512
    # Explained, not silently dropped -- and the culprit is named the way the user sees it
    # ("broken:v1"), not by the bare tag filename ("v1"), which identifies nothing.
    assert result["notes"]
    assert "broken:v1" in result["notes"][0]


def test_ollama_delete_ignores_a_dotfile_that_is_not_a_manifest(ollama_root):
    """A stray .DS_Store must not read as an unparseable manifest and pin every blob."""
    own = _blob(ollama_root, "own", 512)
    tag = _manifest(ollama_root, "llama3", "8b", own)
    _write(
        ollama_root / "manifests" / "registry.ollama.ai" / "library" / "llama3" / ".DS_Store",
        "\x00\x01",
    )

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert not (ollama_root / "blobs" / "sha256-own").exists()


def test_ollama_delete_counts_the_config_blob_too(ollama_root):
    config = _blob(ollama_root, "config", 64)
    weights = _blob(ollama_root, "weights", 128)
    tag = _manifest(ollama_root, "llama3", "8b", weights, config = config)

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert not (ollama_root / "blobs" / "sha256-config").exists()
    assert not (ollama_root / "blobs" / "sha256-weights").exists()


def test_ollama_delete_collects_the_studio_link_dir(ollama_root):
    """The leftover `ollama rm` would never know about: Studio's own .gguf symlink."""
    own = _blob(ollama_root, "own", 256)
    tag = _manifest(ollama_root, "llama3", "8b", own)
    rel = tag.relative_to(ollama_root / "manifests")
    link_dir = ollama_root / ".studio_links" / ollama.ollama_manifest_stem_hash(rel)
    link_dir.mkdir(parents = True)
    link = link_dir / "llama3-8b.gguf"
    link.symlink_to(ollama_root / "blobs" / "sha256-own")

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert not link.exists()
    assert not link_dir.exists()
    assert not (ollama_root / ".studio_links").exists(), "the emptied links root goes too"


def test_ollama_delete_prunes_the_emptied_manifest_folders(ollama_root):
    own = _blob(ollama_root, "own", 32)
    tag = _manifest(ollama_root, "llama3", "8b", own)

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    manifests = ollama_root / "manifests"
    assert not (manifests / "registry.ollama.ai" / "library" / "llama3").exists()
    assert not (manifests / "registry.ollama.ai").exists()
    assert manifests.exists(), "the store root itself is never pruned"


def test_ollama_delete_keeps_a_sibling_tags_folder(ollama_root):
    own = _blob(ollama_root, "own", 32)
    other = _blob(ollama_root, "other", 32)
    keep = _manifest(ollama_root, "llama3", "latest", other)
    tag = _manifest(ollama_root, "llama3", "8b", own)

    local_deletion.delete_local_model_blocking(_ref(tag), "ollama")

    assert keep.exists()
    assert keep.parent.exists()


def test_ollama_ref_outside_a_known_root_is_refused(ollama_root, tmp_path):
    outside = _write(tmp_path / "elsewhere" / "manifests" / "a" / "b" / "c", "{}")

    plan = local_deletion.plan_local_delete(_ref(outside), "ollama")

    assert plan.blocked
    assert outside.exists()


def test_ollama_delete_refuses_while_the_model_is_loaded(ollama_root, monkeypatch):
    own = _blob(ollama_root, "own", 256)
    tag = _manifest(ollama_root, "llama3", "8b", own)
    rel = tag.relative_to(ollama_root / "manifests")
    link_dir = ollama_root / ".studio_links" / ollama.ollama_manifest_stem_hash(rel)
    link_dir.mkdir(parents = True)
    link = link_dir / "llama3-8b.gguf"
    link.symlink_to(ollama_root / "blobs" / "sha256-own")

    # llama.cpp holds the materialized LINK, which shares no name with the manifest or the blob.
    monkeypatch.setattr(local_deletion, "_loaded_identifiers", lambda: [str(link)])

    with pytest.raises(HTTPException) as excinfo:
        local_deletion.delete_local_model_blocking(_ref(tag), "ollama")
    assert excinfo.value.status_code == 400
    assert tag.exists()
    assert (ollama_root / "blobs" / "sha256-own").exists()


def test_delete_waits_on_the_same_lock_a_load_materializes_under(ollama_root, monkeypatch):
    """The load path holds a per-manifest lock while it materializes the .gguf links. A delete
    running underneath it would unlink blobs mid-load, so it takes the same lock -- and answers
    rather than hanging when a long load will not give it up."""
    own = _blob(ollama_root, "own", 256)
    tag = _manifest(ollama_root, "llama3", "8b", own)
    ref = _ref(tag)
    resolved_tag, _root = ollama.resolve_ollama_manifest_ref(ref)

    # Stand in for a load holding its lease.
    held = ollama._materialization_lock(resolved_tag)
    monkeypatch.setattr(ollama, "_WRITE_GUARD_TIMEOUT_SECONDS", 0.05)
    assert held.acquire(timeout = 1)
    try:
        with pytest.raises(HTTPException) as excinfo:
            local_deletion.delete_local_model_blocking(ref, "ollama")
        assert excinfo.value.status_code == 409
        assert tag.exists()
        assert (ollama_root / "blobs" / "sha256-own").exists()
    finally:
        held.release()

    # Once the load lets go, the same delete goes through.
    local_deletion.delete_local_model_blocking(ref, "ollama")
    assert not tag.exists()


def test_unverifiable_load_state_refuses_rather_than_deletes(ollama_root, monkeypatch):
    own = _blob(ollama_root, "own", 256)
    tag = _manifest(ollama_root, "llama3", "8b", own)

    def _boom():
        raise RuntimeError("backend is wedged")

    monkeypatch.setattr(local_deletion, "_loaded_identifiers", _boom)

    with pytest.raises(HTTPException) as excinfo:
        local_deletion.delete_local_model_blocking(_ref(tag), "ollama")
    assert excinfo.value.status_code == 503
    assert tag.exists()


# --- Folders ----------------------------------------------------------------------------------


def test_folder_delete_takes_the_whole_model_directory(scan_root):
    model = scan_root / "publisher" / "some-model"
    _write(model / "config.json", "{}")
    _write(model / "model.safetensors", "w" * 100)
    _write(model / "tokenizer.json", "{}")

    result = local_deletion.delete_local_model_blocking(str(model), "lmstudio")

    assert result["status"] == "deleted"
    assert not model.exists()
    assert result["freed_bytes"] >= 100


def test_folder_delete_prunes_the_publisher_left_holding_nothing(scan_root):
    model = scan_root / "publisher" / "only-model"
    _write(model / "config.json", "{}")
    _write(model / "model.safetensors", "w")

    local_deletion.delete_local_model_blocking(str(model), "lmstudio")

    assert not (scan_root / "publisher").exists(), "the emptied publisher folder is a leftover"
    assert scan_root.exists(), "the scanned root itself is never removed"


def test_folder_delete_keeps_a_publisher_that_still_has_models(scan_root):
    gone = scan_root / "publisher" / "one"
    kept = scan_root / "publisher" / "two"
    _write(gone / "config.json", "{}")
    _write(gone / "model.safetensors", "w")
    _write(kept / "config.json", "{}")
    _write(kept / "model.safetensors", "w")

    local_deletion.delete_local_model_blocking(str(gone), "lmstudio")

    assert not gone.exists()
    assert kept.exists()


def test_standalone_gguf_file_delete(scan_root):
    model = _write(scan_root / "publisher" / "solo-Q4_K_M.gguf", "g" * 64)

    local_deletion.delete_local_model_blocking(str(model), "lmstudio")

    assert not model.exists()
    assert not (scan_root / "publisher").exists()


def test_the_scanned_root_itself_is_never_deletable(scan_root):
    """A root that is itself a model dir must not be removable: it is a folder the user
    registered, not a row inside one."""
    _write(scan_root / "config.json", "{}")
    _write(scan_root / "model.safetensors", "w")

    plan = local_deletion.plan_local_delete(str(scan_root), "lmstudio")

    assert plan.blocked
    assert scan_root.exists()


def test_a_path_outside_every_scanned_root_is_refused(scan_root, tmp_path):
    outside = tmp_path / "elsewhere"
    _write(outside / "config.json", "{}")
    _write(outside / "model.safetensors", "w")

    plan = local_deletion.plan_local_delete(str(outside), "custom")

    assert plan.blocked
    assert outside.exists()


def test_a_path_that_is_not_a_model_is_refused_even_inside_a_root(scan_root):
    """The root check alone would let any path under a scan folder through."""
    notes = scan_root / "my-notes"
    _write(notes / "todo.txt", "buy milk")

    plan = local_deletion.plan_local_delete(str(notes), "custom")

    assert plan.blocked
    assert (notes / "todo.txt").exists()


def test_a_symlinked_model_is_refused_rather_than_silently_freeing_nothing(scan_root, tmp_path):
    real = tmp_path / "real-model"
    _write(real / "config.json", "{}")
    _write(real / "model.safetensors", "w" * 50)
    link = scan_root / "shortcut"
    link.symlink_to(real)

    plan = local_deletion.plan_local_delete(str(link), "custom")

    assert plan.blocked
    assert "shortcut" in plan.blocked_by[0] or str(real) in plan.blocked_by[0]
    assert link.exists()
    assert real.exists()


def test_folder_delete_refuses_while_the_model_is_loaded(scan_root, monkeypatch):
    model = scan_root / "publisher" / "busy"
    _write(model / "config.json", "{}")
    _write(model / "model.safetensors", "w")
    # The backend reports a file INSIDE the folder, not the folder itself.
    monkeypatch.setattr(
        local_deletion, "_loaded_identifiers", lambda: [str(model / "model.safetensors")]
    )

    with pytest.raises(HTTPException) as excinfo:
        local_deletion.delete_local_model_blocking(str(model), "lmstudio")
    assert excinfo.value.status_code == 400
    assert model.exists()


def test_impact_reports_what_would_go_without_touching_anything(scan_root):
    model = scan_root / "publisher" / "preview-me"
    _write(model / "config.json", "{}")
    _write(model / "model.safetensors", "w" * 200)

    impact = local_deletion.local_delete_impact_blocking(str(model), "lmstudio")

    assert impact["blocked_by"] == []
    assert impact["reclaimed_bytes"] >= 200
    assert impact["removed_paths"] == [str(model)]
    assert model.exists(), "a preview mutates nothing"


def test_hf_cache_rows_are_sent_to_their_own_delete(scan_root):
    with pytest.raises(HTTPException) as excinfo:
        local_deletion.plan_local_delete("unsloth/gemma-3-4b-it-GGUF", "hf_cache")
    assert excinfo.value.status_code == 400


def test_a_relative_path_is_refused(scan_root):
    with pytest.raises(HTTPException) as excinfo:
        local_deletion.plan_local_delete("../../etc", "custom")
    assert excinfo.value.status_code == 400


def test_an_empty_identifier_is_refused(scan_root):
    with pytest.raises(HTTPException) as excinfo:
        local_deletion.plan_local_delete("   ", "custom")
    assert excinfo.value.status_code == 400


def test_the_deepest_root_bounds_the_prune_walk(tmp_path, monkeypatch):
    """A scan folder nested inside another must survive its last model being deleted."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents = True)
    monkeypatch.setattr(local_deletion, "lmstudio_model_dirs", lambda: [outer, inner])
    monkeypatch.setattr(local_deletion, "ollama_model_dirs", lambda: [])
    monkeypatch.setattr(local_deletion, "_loaded_identifiers", lambda: [])

    model = inner / "a-model"
    _write(model / "config.json", "{}")
    _write(model / "model.safetensors", "w")

    local_deletion.delete_local_model_blocking(str(model), "custom")

    assert not model.exists()
    assert inner.exists(), "the inner scan folder is a registration, not a leftover"
