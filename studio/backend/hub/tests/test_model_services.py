# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.dependencies import get_hf_token
from hub.storage import scan_folders
from hub.services import download_lifecycle
from hub.services import snapshot_progress
from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import (
    cache_inventory,
    common as model_common,
    deletion,
    downloads,
    folder_browser,
    gguf_variants,
    local_inventory,
    ollama,
)
from hub.utils import (
    download_manifest,
    download_registry,
    gguf,
    hf_cache_state,
    inventory_scan,
    paths,
    state_dir,
)
from hub.workers import hf_download


@pytest.fixture(autouse = True)
def _denylist_inert(monkeypatch):
    # The browse tests here exercise allowlist containment, symlink safety and
    # the sensitive-name filter, not the system-directory denylist (which has
    # its own suite in tests/test_browse_denylist.py). On macOS tmp_path
    # resolves under /private/var, a denied prefix, so _resolve_browse_target
    # would 403 the fixture dirs before that logic runs. Keep the denylist inert
    # so these assertions hold on every platform. folder_browser binds
    # is_denied_system_path at import, so patch it on that module, not on
    # scan_folders. The "rejects" cases still 403 via the allowlist/sensitive
    # checks, and the non-browse tests never call it.
    monkeypatch.setattr(folder_browser, "is_denied_system_path", lambda _p: False)


def _repo(repo_id: str, files: list[SimpleNamespace], repo_path: Path):
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = [SimpleNamespace(files = files)],
    )


def _file(
    name: str,
    size: int,
    blob_path: str | None = None,
):
    return SimpleNamespace(file_name = name, size_on_disk = size, blob_path = blob_path)


def _sibling(name: str, size: int, sha: str):
    return SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": sha})


class TestExtractQuantToken:
    def test_trailing_precision_is_kept(self):
        assert gguf.extract_quant_token("model-it-F16.gguf") == "F16"
        assert gguf.extract_quant_token("model-BF16.gguf") == "BF16"

    def test_real_quant_wins_over_infix_precision(self):
        assert gguf.extract_quant_token("Foo-BF16-Q4_K_M.gguf") == "Q4_K_M"
        assert gguf.extract_quant_token("Foo-F16-Q8_0.gguf") == "Q8_0"
        assert gguf.extract_quant_token("Foo-F32-IQ4_XS.gguf") == "IQ4_XS"

    def test_ud_prefix_preserved(self):
        assert gguf.extract_quant_token("Foo-BF16-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"

    def test_precision_infix_variants_do_not_collapse(self):
        labels = {
            gguf.extract_quant_label("Foo-BF16-Q4_K_M.gguf"),
            gguf.extract_quant_label("Foo-BF16-Q8_0.gguf"),
        }
        assert labels == {"Q4_K_M", "Q8_0"}


def test_big_endian_detection_ignores_model_name_be_token():
    assert gguf.is_big_endian_gguf_path("model-Q4_K_M-be.gguf", "Q4_K_M")
    assert gguf.is_big_endian_gguf_path("model-Q4_K_M_be_infill.gguf", "Q4_K_M")
    assert not gguf.is_big_endian_gguf_path("foo-be-Q4_K_M.gguf", "Q4_K_M")
    assert not gguf.is_big_endian_gguf_path("Q4_K_M/foo-be.gguf", "Q4_K_M")
    assert gguf.pick_best_gguf(["model-Q4_K_M-be.gguf", "model-Q4_K_M.gguf"]) == (
        "model-Q4_K_M.gguf"
    )


def _cached_model_row(tmp_path: Path, *, partial: bool, active_cache: bool | None, size_bytes: int):
    path = tmp_path / f"cache-{active_cache}-{partial}-{size_bytes}"
    return model_common._local_model_info(
        scan_path = path,
        load_path = path,
        source = "hf_cache",
        model_format = "safetensors",
        model_id = "Org/Model",
        partial = partial,
        active_cache = active_cache,
        size_bytes = size_bytes,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_local_inventory_prefers_complete_previous_cache_copy(tmp_path, reverse):
    active_partial = _cached_model_row(
        tmp_path,
        partial = True,
        active_cache = True,
        size_bytes = 20,
    )
    previous_complete = _cached_model_row(
        tmp_path,
        partial = False,
        active_cache = False,
        size_bytes = 10,
    )
    rows = [active_partial, previous_complete]
    if reverse:
        rows.reverse()

    result = local_inventory._dedupe_local_models(rows)

    assert result == [previous_complete]


def test_local_inventory_compares_all_non_active_cache_copies(tmp_path):
    inactive_partial = _cached_model_row(
        tmp_path,
        partial = True,
        active_cache = False,
        size_bytes = 20,
    )
    custom_complete = _cached_model_row(
        tmp_path,
        partial = False,
        active_cache = None,
        size_bytes = 10,
    )

    assert local_inventory._dedupe_local_models([inactive_partial, custom_complete]) == [
        custom_complete
    ]


def test_local_inventory_prefers_active_cache_when_copies_are_equally_complete(tmp_path):
    previous = _cached_model_row(
        tmp_path,
        partial = False,
        active_cache = False,
        size_bytes = 20,
    )
    active = _cached_model_row(
        tmp_path,
        partial = False,
        active_cache = True,
        size_bytes = 10,
    )

    assert local_inventory._dedupe_local_models([previous, active]) == [active]


def test_loaded_repo_match_accepts_previous_cache_snapshot_path(monkeypatch, tmp_path):
    repo_dir = tmp_path / "old-hub" / "models--Org--Model"
    snapshot = repo_dir / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    monkeypatch.setattr(deletion, "iter_repo_cache_dirs", lambda *_args: iter([repo_dir]))

    assert deletion._loaded_id_matches_repo(str(snapshot), "Org/Model") is True
    assert deletion._loaded_id_matches_repo(str(snapshot / "model.gguf"), "Org/Model") is True
    assert deletion._loaded_id_matches_repo(str(tmp_path / "other"), "Org/Model") is False


def test_cached_inventory_loads_previous_cache_copy_by_snapshot(monkeypatch, tmp_path):
    active_hub = tmp_path / "active-hub"
    previous_repo = tmp_path / "previous-hub" / "models--Org--Model"
    snapshot = previous_repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active_hub),
    )

    fields = cache_inventory._cache_inventory_fields(
        "Org/Model",
        "safetensors",
        repo_path = previous_repo,
        snapshot_path = snapshot,
    )

    assert fields["load_id"] == str(snapshot)


def test_cached_inventory_keeps_repo_id_for_active_cache(monkeypatch, tmp_path):
    active_hub = tmp_path / "active-hub"
    active_repo = active_hub / "models--Org--Model"
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active_hub),
    )

    fields = cache_inventory._cache_inventory_fields(
        "Org/Model",
        "safetensors",
        repo_path = active_repo,
    )

    assert fields["load_id"] == "Org/Model"


def test_cached_inventory_prefers_active_copy_when_completeness_matches():
    previous = {"partial": False, "active_cache": False, "size_bytes": 200}
    active = {"partial": False, "active_cache": True, "size_bytes": 100}

    assert cache_inventory._prefer_cache_row(active, previous) is True
    assert cache_inventory._prefer_cache_row(previous, active) is False


def test_cached_inventory_prefers_complete_copy_before_active_cache():
    previous = {"partial": False, "active_cache": False, "size_bytes": 100}
    active_partial = {"partial": True, "active_cache": True, "size_bytes": 200}

    assert cache_inventory._prefer_cache_row(previous, active_partial) is True
    assert cache_inventory._prefer_cache_row(active_partial, previous) is False


def test_inventory_scans_every_dynamic_cache_root(monkeypatch, tmp_path):
    first = tmp_path / "first-hub"
    second = tmp_path / "second-hub"
    unreadable = tmp_path / "unreadable-hub"
    first.mkdir()
    second.mkdir()
    unreadable.mkdir()
    scanned = []

    monkeypatch.setattr(
        inventory_scan,
        "hf_cache_roots",
        lambda: [first, unreadable, second],
    )

    def scan_cache(cache_dir):
        path = Path(cache_dir)
        scanned.append(path)
        if path == unreadable:
            raise PermissionError("unreadable")
        return SimpleNamespace(cache_dir = cache_dir)

    monkeypatch.setattr("huggingface_hub.scan_cache_dir", scan_cache)

    result = inventory_scan._compute_all_hf_cache_scans()

    assert scanned == [first, unreadable, second]
    assert [Path(scan.cache_dir) for scan in result] == [first, second]


def test_inventory_applies_download_state_to_its_owning_cache(monkeypatch, tmp_path):
    state_root = tmp_path / "state"
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    repo_id = "Org/Model"
    repo_name = "models--Org--Model"
    repo_a = cache_a / repo_name
    repo_b = cache_b / repo_name
    snapshot_a = repo_a / "snapshots" / "revision"
    snapshot_b = repo_b / "snapshots" / "revision"
    snapshot_a.mkdir(parents = True)
    snapshot_b.mkdir(parents = True)
    (snapshot_a / "config.json").write_bytes(b"x")
    (snapshot_b / "config.json").write_bytes(b"xx")

    monkeypatch.setattr(state_dir, "cache_root", lambda: state_root)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_b),
    )
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        None,
        [download_manifest.ExpectedFile(path = "config.json", size = 2)],
        "http",
        hub_cache = cache_a,
    )

    assert inventory_scan.is_snapshot_partial("model", repo_id, repo_a) is True
    assert inventory_scan.is_snapshot_partial("model", repo_id, repo_b) is False
    assert inventory_scan.partial_transport_for("model", repo_id, None, repo_a) == "http"
    assert inventory_scan.partial_transport_for("model", repo_id, None, repo_b) is None


def test_inventory_scopes_cancel_markers_to_their_owning_cache(monkeypatch, tmp_path):
    state_root = tmp_path / "state"
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    repo_id = "Org/Model"
    repo_name = "models--Org--Model"
    repo_a = cache_a / repo_name
    repo_b = cache_b / repo_name
    repo_a.mkdir(parents = True)
    repo_b.mkdir(parents = True)

    monkeypatch.setattr(state_dir, "cache_root", lambda: state_root)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_b),
    )
    assert download_manifest.write_cancel_marker(
        "model",
        repo_id,
        "Q4_K_M",
        "xet",
        hub_cache = cache_a,
    )

    assert inventory_scan.is_variant_partial(repo_id, "Q4_K_M", repo_cache_dir = repo_a) is True
    assert inventory_scan.is_variant_partial(repo_id, "Q4_K_M", repo_cache_dir = repo_b) is False


def test_list_local_gguf_variants_skips_big_endian_sibling(tmp_path):
    (tmp_path / "model-Q4_K_M-be.gguf").write_bytes(b"x" * 100)
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"y" * 10)

    variants, has_vision = gguf.list_local_gguf_variants(str(tmp_path))

    assert has_vision is False
    assert [(v.quant, v.filename, v.size_bytes) for v in variants] == [
        ("Q4_K_M", "model-Q4_K_M.gguf", 10)
    ]


@pytest.mark.parametrize("repo_id", ["bert-base-uncased", "owner/repo"])
def test_repo_id_validation_accepts_hf_repo_id_contract(repo_id):
    assert paths.is_valid_repo_id(repo_id)


def test_repo_id_validation_accepts_max_length_namespaced_repo():
    assert paths.is_valid_repo_id(f"{'a' * 96}/{'b' * 96}")


@pytest.mark.parametrize(
    "repo_id",
    [
        "datasets/foo/bar",
        ".repo",
        "repo.git",
        "foo..bar",
        "foo--bar",
        "../repo",
        "owner/../repo",
    ],
)
def test_repo_id_validation_rejects_unsafe_or_invalid_ids(repo_id):
    assert not paths.is_valid_repo_id(repo_id)


def test_download_state_preserves_readable_keys_when_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)

    path = state_dir.marker_path("model", "Owner/Repo", "Q4_K_M")

    assert path is not None
    assert path.name == "models--owner--repo--variant--q4_k_m.json"


@pytest.mark.parametrize("variant", ["bad variant with spaces", "q" * 64])
def test_download_state_bounds_long_repo_variant_filenames(monkeypatch, tmp_path, variant):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    repo_id = f"{'a' * 96}/{'b' * 96}"

    assert paths.is_valid_repo_id(repo_id)
    assert download_manifest.write_cancel_marker("model", repo_id, variant, "http")
    assert download_manifest.write_manifest(
        "model",
        repo_id,
        variant,
        [download_manifest.ExpectedFile(path = "model.gguf", size = 1)],
        "http",
    )

    hub_cache = download_manifest._canonical_hub_cache()
    marker_path = state_dir.marker_path(
        "model",
        repo_id,
        variant,
        hub_cache = hub_cache,
    )
    manifest_path = state_dir.manifest_path(
        "model",
        repo_id,
        variant,
        hub_cache = hub_cache,
    )

    assert marker_path is not None
    assert manifest_path is not None
    assert "--sha256-" in marker_path.name
    assert len(marker_path.name.encode("utf-8")) <= 255
    assert len(f".{marker_path.name}.tmp-00000000".encode("utf-8")) <= 255
    assert download_manifest.has_cancel_marker("model", repo_id, variant)
    assert download_manifest.read_manifest("model", repo_id, variant) is not None
    assert list(download_manifest.iter_variant_markers("model", repo_id)) == [
        (variant, marker_path)
    ]
    assert list(download_manifest.iter_variant_manifests("model", repo_id)) == [
        (variant, manifest_path)
    ]


def test_download_state_isolated_across_hub_cache_switches(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    selected = SimpleNamespace(hub_cache = cache_a)

    from utils import hf_cache_settings

    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: selected)
    expected_a = [download_manifest.ExpectedFile(path = "a.gguf", size = 1)]
    expected_b = [download_manifest.ExpectedFile(path = "b.gguf", size = 2)]

    assert download_manifest.write_manifest("model", "Owner/Repo", "Q4_K_M", expected_a)
    assert download_manifest.write_cancel_marker("model", "Owner/Repo", "Q4_K_M", "http")

    selected.hub_cache = cache_b
    assert download_manifest.write_manifest("model", "Owner/Repo", "Q4_K_M", expected_b)

    manifest_b = download_manifest.read_manifest("model", "Owner/Repo", "Q4_K_M")
    manifest_a = download_manifest.read_manifest(
        "model",
        "Owner/Repo",
        "Q4_K_M",
        hub_cache = cache_a,
    )

    assert manifest_b is not None and manifest_b.expected_files == tuple(expected_b)
    assert manifest_a is not None and manifest_a.expected_files == tuple(expected_a)
    assert not download_manifest.has_cancel_marker("model", "Owner/Repo", "Q4_K_M")
    assert download_manifest.has_cancel_marker(
        "model",
        "Owner/Repo",
        "Q4_K_M",
        hub_cache = cache_a,
    )
    assert len(list((tmp_path / "hub-state" / "manifests").rglob("*.json"))) == 2


def test_legacy_unscoped_download_state_falls_back_only_for_selected_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_a),
    )
    manifest = state_dir.manifest_path("model", "Owner/Repo", "Q4_K_M")
    marker = state_dir.marker_path("model", "Owner/Repo", "Q4_K_M")
    assert manifest is not None and marker is not None
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": "Owner/Repo",
                "variant": "Q4_K_M",
                "expected_files": [{"path": "model.gguf", "size": 10}],
                "transport": "http",
            }
        ),
        encoding = "utf-8",
    )
    marker.write_text(
        json.dumps({"version": 1, "repo_id": "Owner/Repo", "variant": "Q4_K_M"}),
        encoding = "utf-8",
    )

    assert download_manifest.read_manifest("model", "Owner/Repo", "Q4_K_M") is not None
    assert download_manifest.has_cancel_marker("model", "Owner/Repo", "Q4_K_M")
    assert list(download_manifest.iter_variant_manifests("model", "Owner/Repo")) == [
        ("Q4_K_M", manifest)
    ]
    assert list(download_manifest.iter_variant_markers("model", "Owner/Repo")) == [
        ("Q4_K_M", marker)
    ]
    assert (
        download_manifest.read_manifest(
            "model",
            "Owner/Repo",
            "Q4_K_M",
            hub_cache = cache_b,
        )
        is None
    )
    assert not download_manifest.has_cancel_marker(
        "model",
        "Owner/Repo",
        "Q4_K_M",
        hub_cache = cache_b,
    )


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))


def test_resolve_browse_target_preserves_allowlist_and_symlink_safety(tmp_path):
    home = tmp_path / "home"
    scan = tmp_path / "scan"
    target = scan / "nested"
    home.mkdir()
    target.mkdir(parents = True)
    (home / "scan-link").symlink_to(scan, target_is_directory = True)

    resolved = folder_browser._resolve_browse_target(
        str(home / "scan-link" / "nested"),
        [home, scan],
    )

    assert resolved == target.resolve()


def test_resolve_browse_target_rejects_outside_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        folder_browser._resolve_browse_target(str(outside), [allowed])

    assert exc_info.value.status_code == 403


def test_resolve_browse_target_rejects_sensitive_dir(tmp_path):
    home = tmp_path / "home"
    ssh = home / ".ssh"
    ssh.mkdir(parents = True)

    with pytest.raises(HTTPException) as exc_info:
        folder_browser._resolve_browse_target(str(ssh), [home])

    assert exc_info.value.status_code == 403


def test_resolve_browse_target_rejects_sensitive_root(tmp_path):
    ssh = tmp_path / "home" / ".ssh"
    ssh.mkdir(parents = True)

    with pytest.raises(HTTPException) as exc_info:
        folder_browser._resolve_browse_target(str(ssh), [ssh])

    assert exc_info.value.status_code == 403


def test_browse_folders_hides_sensitive_dirs(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents = True)
    (home / "models").mkdir()
    # Accept and ignore the optional (media_roots, drive_roots) args the caller now passes.
    monkeypatch.setattr(folder_browser, "_build_browse_allowlist", lambda *_a, **_k: [home])

    response = folder_browser.browse_folders_response(str(home), show_hidden = True)

    names = {entry.name for entry in response.entries}
    assert "models" in names
    assert ".ssh" not in names


def test_browse_allowlist_includes_linux_run_media_mounts(monkeypatch, tmp_path):
    home = tmp_path / "home"
    media_root = tmp_path / "run" / "media" / "dspofu" / "nvmeB"
    model_dir = media_root / "modelsAI" / "gguf" / "qwen3.6"
    home.mkdir()
    model_dir.mkdir(parents = True)
    monkeypatch.setattr(folder_browser.Path, "home", lambda: home)
    monkeypatch.setattr(folder_browser, "linux_run_media_mount_roots", lambda: [media_root])
    monkeypatch.setattr(folder_browser, "_resolve_hf_cache_dir", lambda: tmp_path / "missing-hf")
    monkeypatch.setattr(scan_folders, "list_scan_folders", lambda: [])
    monkeypatch.setattr(folder_browser, "well_known_model_dirs", lambda: [])

    allowlist = folder_browser._build_browse_allowlist()

    assert media_root.resolve() in allowlist
    assert folder_browser._resolve_browse_target(str(model_dir), allowlist) == model_dir.resolve()


def test_get_models_folder_response_creates_and_returns_dir(monkeypatch, tmp_path):
    # The endpoint creates the cache dir on demand so the desktop "Open folder"
    # action works even before the first download.
    target = tmp_path / "hub"
    monkeypatch.setattr(local_inventory, "_resolve_hf_cache_dir", lambda: target)

    response = local_inventory.get_models_folder_response()

    assert response == {"path": str(target)}
    assert target.is_dir()


def test_get_models_folder_response_reports_create_failure(monkeypatch, tmp_path):
    target = tmp_path / "hub"
    target.write_text("not a directory")
    monkeypatch.setattr(local_inventory, "_resolve_hf_cache_dir", lambda: target)

    with pytest.raises(HTTPException) as exc_info:
        local_inventory.get_models_folder_response()

    assert exc_info.value.status_code == 500
    assert "Failed to create models folder" in exc_info.value.detail


def test_get_models_folder_response_requires_directory(monkeypatch, tmp_path):
    class MissingPath:
        def __init__(self, value: Path):
            self.value = value

        def mkdir(self, *, parents: bool, exist_ok: bool):
            assert parents is True
            assert exist_ok is True

        def is_dir(self):
            return False

        def __str__(self):
            return str(self.value)

    target = MissingPath(tmp_path / "hub")
    monkeypatch.setattr(local_inventory, "_resolve_hf_cache_dir", lambda: target)

    with pytest.raises(HTTPException) as exc_info:
        local_inventory.get_models_folder_response()

    assert exc_info.value.status_code == 500
    assert "not a directory" in exc_info.value.detail


def test_contained_link_path_confines_to_link_dir(tmp_path):
    link_dir = tmp_path / "ollama" / ".studio_links" / "abc123"

    legit = ollama._contained_link_path(link_dir, "llama3-latest-Q4_K_M.gguf")
    assert legit == link_dir / "llama3-latest-Q4_K_M.gguf"

    for unsafe in (
        "",
        ".",
        "..",
        "a/b.gguf",
        "../evil.gguf",
        "/etc/passwd",
        "model-tag-../../../pwned.gguf",
    ):
        assert ollama._contained_link_path(link_dir, unsafe) is None


def test_make_ollama_blob_link_refuses_escaping_name(tmp_path):
    root = tmp_path / "ollama"
    link_dir = root / ".studio_links" / "abc123"
    blob = root / "blobs" / "sha256-deadbeef"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"weights")

    escaped = ollama._make_ollama_blob_link(link_dir, "model-tag-../../../pwned.gguf", blob)
    assert escaped is None
    assert not list(tmp_path.rglob("pwned.gguf"))

    safe = ollama._make_ollama_blob_link(link_dir, "model-tag.gguf", blob)
    assert safe == str(link_dir / "model-tag.gguf")
    assert (link_dir / "model-tag.gguf").exists()


def test_cached_gguf_scan_dedupes_and_excludes_mmproj_only(monkeypatch, tmp_path):
    smaller = _repo("Org/Dupe", [_file("Q4_K_M.gguf", 100)], tmp_path / "small")
    larger = _repo(
        "org/dupe",
        [_file("Q4_K_M.gguf", 300), _file("Q8_0.gguf", 200)],
        tmp_path / "large",
    )
    mmproj_only = _repo("Org/VisionAdapter", [_file("mmproj-F16.gguf", 900)], tmp_path / "mmproj")
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [smaller, larger, mmproj_only])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}

    assert [row["repo_id"] for row in result["cached"]] == ["org/dupe"]
    assert result["cached"][0]["size_bytes"] == 500
    assert result["cached"][0]["model_format"] == "gguf"
    assert result["cached"][0]["capabilities"]["requires_variant"] is True


def test_cached_gguf_scan_preserves_partial_flag(monkeypatch, tmp_path):
    partial = _repo("Org/Partial", [_file("Q4_K_M.gguf", 100)], tmp_path / "partial")
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: True,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}
    row = result["cached"][0]

    assert row["partial"] is True
    assert row["partial_transport"] is None
    assert row["capabilities"]["can_chat"] is False


def test_cached_gguf_scan_includes_variant_state_without_completed_gguf(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    repo_path = tmp_path / "hub" / "models--Org--PartialGguf"
    repo_path.mkdir(parents = True)
    partial = _repo(
        "Org/PartialGguf",
        [_file("config.json", 12)],
        repo_path,
    )
    assert download_manifest.write_manifest(
        "model",
        "Org/PartialGguf",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 4096)],
        "http",
        hub_cache = repo_path.parent,
    )
    assert download_manifest.write_cancel_marker(
        "model",
        "Org/PartialGguf",
        "Q4_K_M",
        "http",
        hub_cache = repo_path.parent,
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: True,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}
    row = result["cached"][0]

    assert row["repo_id"] == "Org/PartialGguf"
    assert row["model_format"] == "gguf"
    assert row["size_bytes"] == 4096
    assert row["partial"] is True
    assert row["capabilities"]["requires_variant"] is True


def test_cached_gguf_scan_hides_infra_repos_without_user_downloads(monkeypatch, tmp_path):
    probe = _repo(
        "ggml-org/models",
        [_file("tinyllamas/stories260K.gguf", 1_200_000)],
        tmp_path / "probe",
    )
    embedder = _repo(
        "unsloth/bge-small-en-v1.5-GGUF",
        [_file("bge-small-en-v1.5-f16.gguf", 60_000_000)],
        tmp_path / "embedder",
    )
    chat = _repo("Org/Chat-GGUF", [_file("Q4_K_M.gguf", 100)], tmp_path / "chat")
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [probe, embedder, chat])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}

    assert [row["repo_id"] for row in result["cached"]] == ["Org/Chat-GGUF"]


def test_cached_gguf_scan_keeps_infra_repo_with_user_downloaded_variant(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    embedder = _repo(
        "unsloth/bge-small-en-v1.5-GGUF",
        [
            _file("bge-small-en-v1.5-f16.gguf", 60_000_000),
            _file("bge-small-en-v1.5-Q8_0.gguf", 35_000_000),
        ],
        tmp_path / "embedder",
    )
    # Variant manifests only exist for user Hub downloads, not auto-downloads.
    assert download_manifest.write_manifest(
        "model",
        "unsloth/bge-small-en-v1.5-GGUF",
        "Q8_0",
        [download_manifest.ExpectedFile(path = "bge-small-en-v1.5-Q8_0.gguf", size = 35_000_000)],
        "http",
        hub_cache = Path(embedder.repo_path).parent,
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [embedder])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}

    assert [row["repo_id"] for row in result["cached"]] == ["unsloth/bge-small-en-v1.5-GGUF"]
    assert result["cached"][0]["capabilities"]["can_chat"] is False


def test_cached_models_scan_hides_non_gguf_embedder(monkeypatch, tmp_path):
    embedder_path = tmp_path / "hub" / "models--unsloth--bge-small-en-v1.5"
    embedder_path.mkdir(parents = True)
    embedder = _repo(
        "unsloth/bge-small-en-v1.5",
        [_file("config.json", 12), _file("model.safetensors", 130_000_000)],
        embedder_path,
    )
    chat_path = tmp_path / "hub" / "models--Org--Chat"
    chat_path.mkdir(parents = True)
    chat = _repo(
        "Org/Chat",
        [_file("config.json", 12), _file("model.safetensors", 100)],
        chat_path,
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [embedder, chat])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path: False,
    )

    result = {"cached": cache_inventory._scan_cached_models()}

    assert [row["repo_id"] for row in result["cached"]] == ["Org/Chat"]


def test_cached_scans_hide_embedders_configured_by_cache_path(monkeypatch, tmp_path):
    from core.rag import config as rag_config

    gguf_path = tmp_path / "hub" / "models--Org--PathEmbedder-GGUF"
    gguf_path.mkdir(parents = True)
    gguf = _repo(
        "Org/PathEmbedder-GGUF",
        [_file("model-F16.gguf", 60_000_000)],
        gguf_path,
    )
    model_path = tmp_path / "hub" / "models--Org--PathEmbedder"
    model_path.mkdir(parents = True)
    model = _repo(
        "Org/PathEmbedder",
        [_file("config.json", 12), _file("model.safetensors", 130_000_000)],
        model_path,
    )
    monkeypatch.setattr(
        rag_config,
        "effective_embedding_model",
        lambda: str(model_path),
    )
    monkeypatch.setattr(
        rag_config,
        "effective_gguf_repo",
        lambda: str(gguf_path),
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [gguf, model])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path: False,
    )

    assert cache_inventory._scan_cached_gguf() == []
    assert cache_inventory._scan_cached_models() == []


def test_cached_scans_hide_embedders_configured_by_snapshot_path(monkeypatch, tmp_path):
    from core.rag import config as rag_config

    gguf_path = tmp_path / "hub" / "models--Org--SnapshotEmbedder-GGUF"
    gguf_snapshot = gguf_path / "snapshots" / "gguf-revision"
    gguf_snapshot.mkdir(parents = True)
    gguf = _repo(
        "Org/SnapshotEmbedder-GGUF",
        [_file("model-F16.gguf", 60_000_000)],
        gguf_path,
    )
    model_path = tmp_path / "hub" / "models--Org--SnapshotEmbedder"
    model_snapshot = model_path / "snapshots" / "model-revision"
    model_snapshot.mkdir(parents = True)
    model = _repo(
        "Org/SnapshotEmbedder",
        [_file("config.json", 12), _file("model.safetensors", 130_000_000)],
        model_path,
    )
    monkeypatch.setattr(
        rag_config,
        "effective_embedding_model",
        lambda: str(model_snapshot),
    )
    monkeypatch.setattr(
        rag_config,
        "effective_gguf_repo",
        lambda: str(gguf_snapshot),
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [gguf, model])],
    )

    def _resolve_snapshot(repo_path):
        return str(
            {
                gguf_path: gguf_snapshot,
                model_path: model_snapshot,
            }.get(Path(repo_path), Path(repo_path))
        )

    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "resolve_hf_cache_realpath",
        _resolve_snapshot,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path: False,
    )

    assert cache_inventory._scan_cached_gguf() == []
    assert cache_inventory._scan_cached_models() == []


def test_cached_models_scan_keeps_unrelated_repo_with_custom_generic_embedder(
    monkeypatch, tmp_path
):
    # A custom embedder with a generic basename ("org/model") must be hidden by
    # EXACT repo-id match only. An unrelated cached chat model whose id merely
    # contains "model" (e.g. "user/model-chat") must stay on device: substring
    # basename matching used to drop real chat models from the inventory.
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")

    def _model_repo(repo_id: str):
        path = tmp_path / "hub" / f"models--{repo_id.replace('/', '--')}"
        path.mkdir(parents = True)
        return _repo(
            repo_id,
            [_file("config.json", 12), _file("model.safetensors", 100)],
            path,
        )

    embedder = _model_repo("org/model")
    chat = _model_repo("user/model-chat")
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [embedder, chat])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path: False,
    )

    result = {"cached": cache_inventory._scan_cached_models()}

    assert [row["repo_id"] for row in result["cached"]] == ["user/model-chat"]


def test_cached_scans_hide_stale_default_embedder_after_custom_setting(monkeypatch, tmp_path):
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/custom")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/custom-GGUF")

    gguf = _repo(
        "unsloth/bge-small-en-v1.5-GGUF",
        [_file("bge-small-en-v1.5-f16.gguf", 60_000_000)],
        tmp_path / "default-gguf",
    )
    weights_path = tmp_path / "hub" / "models--unsloth--bge-small-en-v1.5"
    weights_path.mkdir(parents = True)
    weights = _repo(
        "unsloth/bge-small-en-v1.5",
        [_file("config.json", 12), _file("model.safetensors", 130_000_000)],
        weights_path,
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [gguf, weights])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path: False,
    )

    assert cache_inventory._scan_cached_gguf() == []
    assert cache_inventory._scan_cached_models() == []


def test_gguf_variant_requirements_include_split_files_and_preferred_mmproj():
    requirements = gguf_variants._build_gguf_variant_requirements(
        [
            _sibling("model-Q4_K_M-00001-of-00002.gguf", 10, "main-a"),
            _sibling("model-Q4_K_M-00002-of-00002.gguf", 20, "main-b"),
            _sibling("mmproj-BF16.gguf", 7, "mm-bf16"),
            _sibling("mmproj-F16.gguf", 5, "mm-f16"),
        ]
    )

    req = requirements["q4_k_m"]

    assert req.main_size_bytes == 30
    assert req.download_size_bytes == 35
    assert req.main_hashes == frozenset({"main-a", "main-b"})
    assert req.required_hashes == frozenset({"main-a", "main-b", "mm-f16"})
    assert req.companion_hashes == frozenset({"mm-f16"})
    assert req.mmproj_hashes == frozenset({"mm-bf16", "mm-f16"})
    assert req.target_filenames == (
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "mmproj-F16.gguf",
    )


def test_gguf_variant_requirements_skip_big_endian_sibling():
    requirements = gguf_variants._build_gguf_variant_requirements(
        [
            _sibling("model-Q4_K_M-be.gguf", 100, "main-be"),
            _sibling("model-Q4_K_M.gguf", 10, "main-le"),
        ]
    )

    req = requirements["q4_k_m"]

    assert req.main_size_bytes == 10
    assert req.main_hashes == frozenset({"main-le"})
    assert req.main_filenames == frozenset({"model-Q4_K_M.gguf"})
    assert req.target_filenames == ("model-Q4_K_M.gguf",)


def test_worker_gguf_variant_plan_matches_service_requirement(monkeypatch):
    siblings = [
        _sibling("model-Q4_K_M-00001-of-00002.gguf", 10, "main-a"),
        _sibling("model-Q4_K_M-00002-of-00002.gguf", 20, "main-b"),
        _sibling("mmproj-BF16.gguf", 7, "mm-bf16"),
        _sibling("mmproj-F16.gguf", 5, "mm-f16"),
    ]
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(siblings = siblings),
    )

    service_req = gguf_variants._build_gguf_variant_requirements(siblings)["q4_k_m"]
    worker_plan = hf_download._gguf_variant_target_plan("Org/Vision", "Q4_K_M", None)

    assert worker_plan == service_req


def test_gguf_variant_blob_hashes_accept_dict_lfs_fallback(monkeypatch):
    with gguf_variants._VARIANT_HASH_LOCK:
        gguf_variants._VARIANT_HASH_CACHE.clear()
        gguf_variants._VARIANT_REQUIREMENT_CACHE.clear()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi = lambda *_args, **_kwargs: SimpleNamespace(
                model_info = lambda *_a, **_k: SimpleNamespace(
                    siblings = [
                        _sibling("model-Q4_K_M.gguf", 10, "main-dict"),
                        _sibling("model-Q8_0.gguf", 20, "other"),
                        _sibling("mmproj-F16.gguf", 5, "mmproj"),
                    ]
                )
            )
        ),
    )

    result = gguf_variants.gguf_variant_blob_hashes("Org/DictLfs", "Q4_K_M", None)
    main_only = gguf_variants.gguf_variant_blob_hashes(
        "Org/DictLfs",
        "Q4_K_M",
        None,
        include_companions = False,
    )

    assert result == frozenset({"main-dict", "mmproj"})
    assert main_only == frozenset({"main-dict"})


def test_gguf_variant_blob_hashes_skip_missing_rfilename(monkeypatch):
    with gguf_variants._VARIANT_HASH_LOCK:
        gguf_variants._VARIANT_HASH_CACHE.clear()
        gguf_variants._VARIANT_REQUIREMENT_CACHE.clear()
    siblings = [
        SimpleNamespace(rfilename = None, size = 1, lfs = {"sha256": "bad"}),
        _sibling("model-Q4_K_M.gguf", 10, "main"),
    ]
    monkeypatch.setattr(
        gguf_variants,
        "_fetch_gguf_variant_requirements",
        lambda _repo_id, _hf_token = None: gguf_variants._build_gguf_variant_requirements(siblings),
    )

    result = gguf_variants.gguf_variant_blob_hashes("Org/Malformed", "Q4_K_M", None)

    assert result == frozenset({"main"})


def test_worker_gguf_variant_targets_skip_missing_rfilename(monkeypatch):
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [
                SimpleNamespace(rfilename = None, size = 1),
                _sibling("model-Q4_K_M.gguf", 10, "main"),
                _sibling("mmproj-F16.gguf", 5, "mm"),
            ]
        ),
    )

    result = hf_download._gguf_variant_target_plan("Org/Malformed", "Q4_K_M", None)

    assert list(result.target_filenames) == ["model-Q4_K_M.gguf", "mmproj-F16.gguf"]


def test_download_gguf_variant_purges_only_main_quant_hashes(monkeypatch, tmp_path):
    prepare_calls = []
    snapshot_calls = []
    written = []
    verified = []

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [
                _sibling("model-Q4_K_M.gguf", 10, "q4-main"),
                _sibling("model-Q8_0.gguf", 20, "q8-main"),
                _sibling("mmproj-F16.gguf", 5, "shared-mmproj"),
            ]
        ),
    )
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)) or 0,
    )
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    hf_download._download_gguf_variant("Org/Vision", "Q4_K_M", None, "http")

    assert prepare_calls == [
        (
            ("model", "Org/Vision", "http", "Q4_K_M"),
            {
                "only_blob_hashes": frozenset({"q4-main"}),
                "companion_blob_hashes": frozenset({"shared-mmproj"}),
                "protected_blob_hashes": frozenset(),
            },
        )
    ]
    assert [file.path for file in written[0][3]] == ["model-Q4_K_M.gguf", "mmproj-F16.gguf"]
    assert snapshot_calls[0]["allow_patterns"] == ["model-Q4_K_M.gguf", "mmproj-F16.gguf"]
    assert verified == [("model", "Org/Vision", "Q4_K_M", str(tmp_path))]


def test_download_gguf_variant_manifest_resume_purges_only_main_quant_hashes(monkeypatch, tmp_path):
    prepare_calls = []
    snapshot_calls = []

    def _metadata_unavailable(*_args, **_kwargs):
        raise RuntimeError("metadata down")

    manifest = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "Org/Vision",
        variant = "Q4_K_M",
        started_at = "",
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 10,
                sha256 = "q4-main",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 5,
                sha256 = "shared-mmproj",
            ),
        ),
        transport = "http",
    )
    monkeypatch.setattr(
        hf_download,
        "_gguf_variant_target_plan",
        _metadata_unavailable,
    )
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *_args: manifest)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)) or 0,
    )
    monkeypatch.setattr(hf_download, "_verify_completed_download", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    hf_download._download_gguf_variant("Org/Vision", "Q4_K_M", None, "http")

    assert prepare_calls == [
        (
            ("model", "Org/Vision", "http", "Q4_K_M"),
            {
                "only_blob_hashes": frozenset({"q4-main"}),
                "companion_blob_hashes": frozenset({"shared-mmproj"}),
                "protected_blob_hashes": frozenset(),
            },
        )
    ]
    assert snapshot_calls[0]["allow_patterns"] == ["model-Q4_K_M.gguf", "mmproj-F16.gguf"]


def test_download_snapshot_recovers_manifest_after_metadata_fallback(monkeypatch, tmp_path):
    metadata_calls = []
    written = []
    cleared = []
    verified = []

    def _metadata(*_args, **_kwargs):
        metadata_calls.append(True)
        if len(metadata_calls) == 1:
            raise RuntimeError("metadata down")
        return SimpleNamespace(siblings = [SimpleNamespace(rfilename = "config.json", size = 12)])

    monkeypatch.setattr(hf_download, "_model_info_with_retry", _metadata)
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        download_manifest, "clear_cancel_marker", lambda *args: cleared.append(args)
    )
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    hf_download._download_snapshot("Org/Model", None, "http")

    assert len(metadata_calls) == 2
    assert cleared == [("model", "Org/Model", None)]
    assert written[0][0:3] == ("model", "Org/Model", None)
    assert written[0][3][0].path == "config.json"
    assert verified == [("model", "Org/Model", None, str(tmp_path))]


def test_download_dataset_continues_without_metadata_manifest(monkeypatch, tmp_path):
    metadata_calls = []
    snapshot_calls = []
    written = []
    cleared = []
    verified = []

    def _metadata(*_args, **_kwargs):
        metadata_calls.append(True)
        raise RuntimeError("metadata down")

    monkeypatch.setattr(hf_download, "_dataset_info_with_retry", _metadata)
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        download_manifest, "clear_cancel_marker", lambda *args: cleared.append(args)
    )
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setattr(
        hf_cache_state, "has_active_incomplete_blobs", lambda *_args, **_kwargs: False
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    hf_download._download_dataset("Org/Data", None, "http")

    assert len(metadata_calls) == 2
    assert cleared == [("dataset", "Org/Data", None)]
    assert written == []
    assert snapshot_calls == [
        {
            "repo_id": "Org/Data",
            "token": False,
            "repo_type": "dataset",
            "max_workers": 1,
        }
    ]
    assert verified == [("dataset", "Org/Data", None, str(tmp_path))]


def test_download_snapshot_fails_when_metadata_unavailable_and_partial_remains(
    monkeypatch, tmp_path
):
    """No prior manifest + metadata unavailable + leftover .incomplete blobs means
    a cached partial was returned without downloading: the worker must exit 1, not
    derive a self-certifying manifest from the finalized subset."""
    written = []
    verified = []

    def _metadata(*_args, **_kwargs):
        raise RuntimeError("metadata down")

    monkeypatch.setattr(hf_download, "_model_info_with_retry", _metadata)
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setattr(
        hf_cache_state, "has_active_incomplete_blobs", lambda *_args, **_kwargs: True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    with pytest.raises(SystemExit) as excinfo:
        hf_download._download_snapshot("Org/Model", None, "http")

    assert excinfo.value.code == 1
    assert written == []
    assert verified == []


def test_purge_repo_cache_dirs_skips_top_level_symlink(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    target = tmp_path / "target"
    root.mkdir()
    target.mkdir()
    link = root / "models--Org--Repo"
    link.symlink_to(target, target_is_directory = True)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [root])

    removed = hf_cache_state.purge_repo_cache_dirs("model", "Org/Repo")

    assert removed is False
    assert link.is_symlink()
    assert target.is_dir()


def test_gguf_download_progress_fallback_logs_warning(monkeypatch):
    token = "hf_12345678901234567890"
    logger = _RecordingLogger()

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _raise_permission_error(*_args, **_kwargs):
        raise PermissionError(f"denied {token}")

    monkeypatch.setattr(snapshot_progress, "logger", logger)
    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_args, **_kwargs: frozenset(),
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        _raise_permission_error,
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "running")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model",
            variant = "Q4_K_M",
            expected_bytes = -1,
            hf_token = token,
        )
    )

    assert result == {
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": 0,
        "progress": 0,
        "cache_path": None,
    }
    assert logger.warnings
    args, kwargs = logger.warnings[0]
    assert args[:4] == (
        "Error checking %s download progress for %s: %s: %s",
        "model",
        "Org/Model",
        "PermissionError",
    )
    assert token not in args[4]
    assert "***" in args[4]
    assert kwargs == {}


def test_gguf_progress_counts_completed_mmproj_with_expected_bytes(monkeypatch, tmp_path):
    """A finished mmproj companion keeps counting toward progress once the caller
    supplies expected bytes; resolving the variant requirement credits it."""
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 30)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (blobs / "mmprojhash").write_bytes(b"y" * 30)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ],
        "http",
        hub_cache = entry.parent,
    )

    requirement = gguf_variants._GgufVariantRequirement(
        main_filenames = frozenset({"model-Q4_K_M.gguf"}),
        target_filenames = ("model-Q4_K_M.gguf", "mmproj-F16.gguf"),
        main_hashes = frozenset({"mainhash"}),
        required_hashes = frozenset({"mainhash", "mmprojhash"}),
        companion_hashes = frozenset({"mmprojhash"}),
        mmproj_filenames = frozenset({"mmproj-F16.gguf"}),
        mmproj_hashes = frozenset({"mmprojhash"}),
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ),
        main_size_bytes = 100,
        download_size_bytes = 130,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: requirement,
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 130
    assert result["downloaded_bytes"] == 130
    assert result["complete_on_disk"] is True
    assert result["progress"] == 1.0


def test_gguf_progress_subtracts_new_job_completed_baseline(monkeypatch, tmp_path):
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 30)
    (blobs / "mmprojhash").write_bytes(b"y" * 30)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ],
        "http",
        hub_cache = entry.parent,
    )

    requirement = gguf_variants._GgufVariantRequirement(
        main_filenames = frozenset({"model-Q4_K_M.gguf"}),
        target_filenames = ("model-Q4_K_M.gguf", "mmproj-F16.gguf"),
        main_hashes = frozenset({"mainhash"}),
        required_hashes = frozenset({"mainhash", "mmprojhash"}),
        companion_hashes = frozenset({"mmprojhash"}),
        mmproj_filenames = frozenset({"mmproj-F16.gguf"}),
        mmproj_hashes = frozenset({"mmprojhash"}),
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ),
        main_size_bytes = 100,
        download_size_bytes = 130,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: requirement,
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
            get_job_metadata = lambda _key: SimpleNamespace(
                completed_baseline_bytes = 30,
            ),
        ),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 0
    assert result["expected_bytes"] == 100
    assert result["complete_on_disk"] is False
    assert result["progress"] == 0


def test_gguf_progress_shows_main_when_companion_left_the_count(monkeypatch, tmp_path):
    # The mmproj companion that seeded the baseline is gone, so completed_bytes
    # is main-only and below the baseline; it must not be subtracted to 0.
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "mainhash").write_bytes(b"x" * 20)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    requirement = gguf_variants._GgufVariantRequirement(
        main_filenames = frozenset({"model-Q4_K_M.gguf"}),
        target_filenames = ("model-Q4_K_M.gguf", "mmproj-F16.gguf"),
        main_hashes = frozenset({"mainhash"}),
        required_hashes = frozenset({"mainhash", "mmprojhash"}),
        companion_hashes = frozenset({"mmprojhash"}),
        mmproj_filenames = frozenset({"mmproj-F16.gguf"}),
        mmproj_hashes = frozenset({"mmprojhash"}),
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ),
        main_size_bytes = 100,
        download_size_bytes = 130,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: requirement,
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
            get_job_metadata = lambda _key: SimpleNamespace(
                completed_baseline_bytes = 30,
            ),
        ),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 20
    assert result["downloaded_bytes"] == 20
    assert result["expected_bytes"] == 130
    assert result["complete_on_disk"] is False


def test_gguf_progress_complete_on_disk_ignores_full_baseline(monkeypatch, tmp_path):
    # A variant already complete on disk carries a baseline equal to its full
    # size; subtracting it would report 0/0 for a finished variant (frontend
    # evicts it as gone). Once complete_on_disk is verified, the full figures
    # must survive.
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 30)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (blobs / "mmprojhash").write_bytes(b"y" * 30)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ],
        "http",
        hub_cache = entry.parent,
    )

    requirement = gguf_variants._GgufVariantRequirement(
        main_filenames = frozenset({"model-Q4_K_M.gguf"}),
        target_filenames = ("model-Q4_K_M.gguf", "mmproj-F16.gguf"),
        main_hashes = frozenset({"mainhash"}),
        required_hashes = frozenset({"mainhash", "mmprojhash"}),
        companion_hashes = frozenset({"mmprojhash"}),
        mmproj_filenames = frozenset({"mmproj-F16.gguf"}),
        mmproj_hashes = frozenset({"mmprojhash"}),
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
            download_manifest.ExpectedFile(
                path = "mmproj-F16.gguf",
                size = 30,
                sha256 = "mmprojhash",
            ),
        ),
        main_size_bytes = 100,
        download_size_bytes = 130,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: requirement,
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
            get_job_metadata = lambda _key: SimpleNamespace(
                completed_baseline_bytes = 130,
            ),
        ),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["complete_on_disk"] is True
    assert result["completed_bytes"] == 130
    assert result["downloaded_bytes"] == 130
    assert result["expected_bytes"] == 130
    assert result["progress"] == 1.0


def test_gguf_progress_scoped_hashes_exclude_sibling_quant(monkeypatch, tmp_path):
    # The "instant ~900 MB" bug: a sibling quant is fully cached when a different
    # variant starts. With this variant's hashes resolved, progress counts ONLY
    # its in-progress blob, never the sibling's finalized bytes in the shared
    # blobs/ dir.
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "siblinghash").write_bytes(b"z" * 900)  # other quant, complete
    (blobs / "mainhash.incomplete").write_bytes(b"x" * 5)  # this variant, started

    requirement = gguf_variants._GgufVariantRequirement(
        main_filenames = frozenset({"model-Q4_K_M.gguf"}),
        target_filenames = ("model-Q4_K_M.gguf",),
        main_hashes = frozenset({"mainhash"}),
        required_hashes = frozenset({"mainhash"}),
        companion_hashes = frozenset(),
        mmproj_filenames = frozenset(),
        mmproj_hashes = frozenset(),
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            ),
        ),
        main_size_bytes = 100,
        download_size_bytes = 100,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: requirement,
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
            get_job_metadata = lambda _key: SimpleNamespace(
                completed_baseline_bytes = 0,
            ),
        ),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 5


def test_gguf_progress_unknown_hashes_does_not_count_foreign_blobs(monkeypatch, tmp_path):
    # With a variant's hashes unresolved (metadata flaked, no manifest), the
    # shared blobs/ dir's FINALIZED blobs must NOT be counted wholesale: a cached
    # sibling quant (``siblinghash``) alongside is the "instant ~900 MB" bug.
    # With no .incomplete present, downloaded must be 0.
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (blobs / "mmprojhash").write_bytes(b"y" * 30)
    (blobs / "siblinghash").write_bytes(b"z" * 900)

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_args, **_kwargs: frozenset(),
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "running")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 0
    assert result["complete_on_disk"] is False


def test_gguf_progress_unknown_hashes_drops_unscoped_incomplete_blob(monkeypatch, tmp_path):
    # With hashes unresolved, an .incomplete in the shared blobs/ dir can't be
    # attributed to this variant (it may be a concurrent sibling's active write),
    # so it is dropped, mirroring the finalized-blob guard. In production the
    # worker writes the manifest before any .incomplete exists, so hashes resolve
    # via the manifest backstop and this window never suppresses real progress.
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "activehash.incomplete").write_bytes(b"x" * 50)  # unattributable
    (blobs / "siblinghash").write_bytes(b"z" * 900)  # finalized sibling

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_args, **_kwargs: frozenset(),
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "running")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 1000,
        )
    )

    assert result["downloaded_bytes"] == 0  # unscoped .incomplete not leaked
    assert result["completed_bytes"] == 0  # finalized sibling still ignored


def test_gguf_progress_unknown_hashes_no_backward_dip_when_variant_finalizes(monkeypatch, tmp_path):
    # Regression for the two-variant dip: with hashes unresolved, the first quant
    # finalizes while the sibling still writes its .incomplete. The sibling's
    # bytes used to leak into this numerator, dipping the bar ~99% -> ~78% for
    # one poll. The unscoped .incomplete must be dropped so the reading stays 0.
    entry = tmp_path / "models--unsloth--SmolLM2-360M-Instruct-GGUF"
    blobs = entry / "blobs"
    snap = entry / "snapshots" / "rev0"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)
    own_total = 218_673_760  # Q2_K finished blob size (denominator)
    sibling_total = 234_686_560  # Q3_K_M total

    def _sparse_file(path: Path, size: int) -> None:
        with path.open("wb") as handle:
            handle.truncate(size)

    own_finalized = blobs / "q2hash"
    _sparse_file(own_finalized, own_total)
    # ~72.7% of the sibling => sibling_partial / own_total == 0.78 pre-fix.
    sibling_incomplete = blobs / "q3hash.incomplete"
    _sparse_file(sibling_incomplete, int(sibling_total * 0.727))

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_args, **_kwargs: frozenset(),
    )
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "running")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "unsloth/SmolLM2-360M-Instruct-GGUF",
            variant = "Q2_K",
            expected_bytes = own_total,
        )
    )

    assert result["downloaded_bytes"] == 0  # sibling .incomplete did not leak
    assert result["progress"] == 0  # no ~0.78 backward dip


def test_hf_cache_model_file_probe_is_bounded(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = tmp_path / "README.md"
    second = tmp_path / "notes.txt"
    model = tmp_path / "model.safetensors"
    first.write_text("readme", encoding = "utf-8")
    second.write_text("notes", encoding = "utf-8")
    model.write_bytes(b"weights")
    entries = [first, second, model]

    monkeypatch.setattr(model_common.Path, "rglob", lambda _self, _pattern: iter(entries))
    monkeypatch.setattr(model_common, "_HF_CACHE_MODEL_FILE_PROBE_LIMIT", 2)

    bounded = model_common._iter_hf_cache_model_files(snapshot)

    assert bounded == [first, second]

    monkeypatch.setattr(model_common, "_HF_CACHE_MODEL_FILE_PROBE_LIMIT", 3)

    unbounded = model_common._iter_hf_cache_model_files(snapshot)

    assert unbounded == [first, second, model]


def test_download_state_lookup_is_repo_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)

    assert download_manifest.write_manifest(
        "model",
        "Owner/Repo",
        None,
        [download_manifest.ExpectedFile(path = "config.json", size = 12)],
    )
    assert download_manifest.write_cancel_marker("model", "Owner/Repo", "Q4_K_M", "http")

    manifest = download_manifest.read_manifest("model", "owner/repo", None)

    assert manifest is not None
    assert manifest.repo_id == "Owner/Repo"
    assert manifest.expected_files[0].path == "config.json"
    assert download_manifest.has_cancel_marker("model", "owner/repo", "Q4_K_M")
    assert (
        download_manifest.read_cancel_marker_transport(
            "model",
            "owner/repo",
            "Q4_K_M",
        )
        == "http"
    )
    assert [
        variant
        for variant, _path in download_manifest.iter_variant_markers(
            "model",
            "owner/repo",
        )
    ] == ["Q4_K_M"]
    assert download_manifest.purge_all_state_for_repo("model", "owner/repo") == 2
    assert download_manifest.read_manifest("model", "owner/repo", None) is None


def test_hf_cache_scan_fallback_row_uses_local_model_info_alias(monkeypatch, tmp_path):
    cache_dir = tmp_path / "hub"
    repo_dir = cache_dir / "models--Org--Broken"
    blobs_dir = repo_dir / "blobs"
    blobs_dir.mkdir(parents = True)
    (blobs_dir / "blob").write_bytes(b"content")
    monkeypatch.setattr(local_inventory, "_classify_local_path", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "resolve_hf_cache_realpath",
        lambda *_args, **_kwargs: None,
    )

    rows = local_inventory._scan_hf_cache(cache_dir)

    assert len(rows) == 1
    assert rows[0].model_id == "Org/Broken"
    assert rows[0].source == "hf_cache"
    assert rows[0].model_format == "unknown"


def test_hf_cache_scan_uses_gguf_partial_row_for_variant_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    cache_dir = tmp_path / "hub"
    repo_dir = cache_dir / "models--Org--PartialGguf"
    blobs_dir = repo_dir / "blobs"
    blobs_dir.mkdir(parents = True)
    (blobs_dir / "partial").write_bytes(b"content")
    assert download_manifest.write_manifest(
        "model",
        "Org/PartialGguf",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 8192)],
        "http",
        hub_cache = cache_dir,
    )
    assert download_manifest.write_cancel_marker(
        "model",
        "Org/PartialGguf",
        "Q4_K_M",
        "http",
        hub_cache = cache_dir,
    )
    monkeypatch.setattr(local_inventory, "_classify_local_path", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "resolve_hf_cache_realpath",
        lambda *_args, **_kwargs: None,
    )

    rows = local_inventory._scan_hf_cache(cache_dir)

    assert len(rows) == 1
    assert rows[0].model_id == "Org/PartialGguf"
    assert rows[0].source == "hf_cache"
    assert rows[0].model_format == "gguf"
    assert rows[0].partial is True
    assert rows[0].size_bytes == 8192
    assert rows[0].capabilities.requires_variant is True


def test_local_inventory_filters_custom_embedder_hf_cache_row(monkeypatch, tmp_path):
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/embedder")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/embedder-GGUF")

    def _row(repo_id: str):
        repo_path = tmp_path / f"models--{repo_id.replace('/', '--')}"
        return model_common._local_model_info(
            scan_path = repo_path,
            load_path = repo_path,
            source = "hf_cache",
            model_format = "safetensors",
            model_id = repo_id,
        )

    rows = local_inventory._filter_hidden_models([_row("org/embedder"), _row("org/chat-model")])

    assert [row.model_id for row in rows] == ["org/chat-model"]


def test_local_inventory_filters_embedder_configured_by_snapshot_path(monkeypatch, tmp_path):
    from core.rag import config as rag_config

    embedder_path = tmp_path / "hub" / "models--org--embedder"
    embedder_snapshot = embedder_path / "snapshots" / "revision"
    embedder_snapshot.mkdir(parents = True)
    chat_path = tmp_path / "hub" / "models--org--chat-model"
    chat_path.mkdir(parents = True)
    monkeypatch.setattr(
        rag_config,
        "effective_embedding_model",
        lambda: str(embedder_snapshot),
    )
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/embedder-GGUF")
    monkeypatch.setattr(
        local_inventory.hf_cache_scan,
        "resolve_hf_cache_realpath",
        lambda path: str(embedder_snapshot) if Path(path) == embedder_path else str(path),
    )

    def _row(repo_id: str, repo_path: Path):
        return model_common._local_model_info(
            scan_path = repo_path,
            load_path = repo_path,
            source = "hf_cache",
            model_format = "safetensors",
            model_id = repo_id,
        )

    rows = local_inventory._filter_hidden_models(
        [_row("org/embedder", embedder_path), _row("org/chat-model", chat_path)]
    )

    assert [row.model_id for row in rows] == ["org/chat-model"]


def test_model_download_job_helpers_preserve_idle_shape():
    key = downloads._download_job_key("Org/Model", None)
    status = downloads._job_status(key)

    assert key == "org/model::"
    assert status.state == "idle"
    assert status.error is None


def test_gguf_repo_partial_treats_completed_disk_variant_as_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    snapshot = tmp_path / "cache" / "models--Org--Repo" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "model-Q8_0.gguf").write_bytes(b"complete")
    assert download_manifest.write_cancel_marker("model", "Org/Repo", "Q4_K_M", "xet")
    monkeypatch.setattr(
        inventory_scan,
        "resolve_snapshot_dir_for_scan",
        lambda *_args: snapshot,
    )

    assert inventory_scan.is_gguf_repo_partial("Org/Repo", snapshot.parents[1]) is False


def test_gguf_repo_partial_flags_vision_variant_missing_mmproj(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    snapshot = tmp_path / "cache" / "models--Org--Vision" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "model-Q4_K_M.gguf").write_bytes(b"complete-weight")
    assert download_manifest.write_manifest(
        "model",
        "Org/Vision",
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 15),
            download_manifest.ExpectedFile(path = "mmproj-F16.gguf", size = 8),
        ],
        "http",
    )
    monkeypatch.setattr(
        inventory_scan,
        "resolve_snapshot_dir_for_scan",
        lambda *_args: snapshot,
    )

    assert inventory_scan.is_gguf_repo_partial("Org/Vision") is True


def test_cancel_worker_leaves_exited_process_to_watcher():
    calls: list = []

    class _Registry:
        def get_process(self, _key):
            return SimpleNamespace(poll = lambda: 1)

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

        def mark_pending_cancel(self, key, generation):
            calls.append(("pending", key, generation))
            return True

        def request_cancel(self, key, proc, generation):
            calls.append(("request", key, generation))
            return True

        def cancel_requested(self, _key):
            return False

    state = download_lifecycle.cancel_worker(
        _Registry(),
        "org/model::",
        generation = 3,
        label = "Org/Model",
        logger = SimpleNamespace(warning = lambda *_a, **_k: None),
    )

    assert state == "running"
    assert calls == []


def test_completed_gguf_split_variant_requires_all_shards(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    first = snapshot / "model-Q8_0-00001-of-00002.gguf"
    second = snapshot / "model-Q8_0-00002-of-00002.gguf"
    first.write_bytes(b"first")

    assert "Q8_0" not in inventory_scan._completed_gguf_variants(snapshot)

    second.write_bytes(b"second")
    assert "Q8_0" in inventory_scan._completed_gguf_variants(snapshot)


def test_variant_partial_accepts_variant_filtered_legacy_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)

    assert inventory_scan.is_variant_partial(
        "Org/Repo",
        "Q4_K_M",
        incomplete_blob_hashes = {"main-q4", "main-q8"},
        variant_blob_hashes = frozenset({"main-q4"}),
    )
    assert not inventory_scan.is_variant_partial(
        "Org/Repo",
        "Q5_K_M",
        incomplete_blob_hashes = {"main-q4"},
        variant_blob_hashes = frozenset({"main-q5"}),
    )


def test_variant_partial_accepts_completed_variant_in_non_latest_snapshot(monkeypatch, tmp_path):
    """A verified GGUF update can prune an older snapshot and make that old
    directory the newest by mtime. The variant is still complete when another
    snapshot satisfies its manifest."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    repo_dir = tmp_path / "cache" / "models--Org--Repo"
    old_snapshot = repo_dir / "snapshots" / "old"
    new_snapshot = repo_dir / "snapshots" / "new"
    old_snapshot.mkdir(parents = True)
    new_snapshot.mkdir(parents = True)
    (old_snapshot / "model-Q8_0.gguf").write_bytes(b"sibling")
    (new_snapshot / "model-Q4_K_M.gguf").write_bytes(b"new")
    assert download_manifest.write_manifest(
        "model",
        "Org/Repo",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 3)],
        "http",
    )

    assert not inventory_scan.is_variant_partial(
        "Org/Repo",
        "Q4_K_M",
        snapshot_dir = old_snapshot,
        repo_cache_dir = repo_dir,
    )


def test_gguf_variants_partial_marker_overrides_size_only_downloaded(monkeypatch, tmp_path):
    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(gguf_variants.asyncio, "to_thread", _run_inline)
    assert download_manifest.write_cancel_marker("model", "Org/PartialRepo", "Q4_K_M", "http")
    snapshot = tmp_path / "cache" / "models--Org--PartialRepo" / "snapshots" / "rev0"
    snapshot.mkdir(parents = True)
    (snapshot / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)

    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    filename = "model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 100,
                )
            ],
            False,
            None,
        ),
    )
    monkeypatch.setattr(
        gguf_variants,
        "iter_hf_cache_snapshots",
        lambda _repo_id, root = None: [snapshot],
    )
    monkeypatch.setattr(
        gguf_variants,
        "_gguf_all_variant_requirements",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        gguf_variants.download_registry,
        "incomplete_blob_hashes",
        lambda *_args, **_kwargs: set(),
    )

    result = asyncio.run(gguf_variants.get_gguf_variants_response("Org/PartialRepo"))

    assert result.variants[0].downloaded is False
    assert result.variants[0].partial is True


def test_gguf_variants_scopes_partial_state_to_requested_cache(monkeypatch, tmp_path):
    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    repo_id = "Org/SharedRepo"
    repo_name = "models--Org--SharedRepo"
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    repo_a = cache_a / repo_name
    snapshot_a = repo_a / "snapshots" / "revision"
    snapshot_a.mkdir(parents = True)
    (snapshot_a / "model-Q8_0.gguf").write_bytes(b"complete")
    blobs_b = cache_b / repo_name / "blobs"
    blobs_b.mkdir(parents = True)
    (blobs_b / "q8-hash.incomplete").write_bytes(b"partial")

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(gguf_variants.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_b),
    )
    assert download_manifest.write_cancel_marker(
        "model",
        repo_id,
        "Q8_0",
        "http",
        hub_cache = cache_b,
    )
    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    filename = "model-Q8_0.gguf",
                    quant = "Q8_0",
                    display_label = None,
                    size_bytes = 8,
                )
            ],
            False,
            [
                SimpleNamespace(
                    rfilename = "model-Q8_0.gguf",
                    size = 8,
                    lfs = SimpleNamespace(sha256 = "q8-hash"),
                )
            ],
        ),
    )
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: [])

    result = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            repo_id,
            local_path = str(repo_a),
        )
    )

    assert result.variants[0].downloaded is True
    assert result.variants[0].partial is False


def test_download_registry_repo_keys_are_case_insensitive():
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
    )
    # The same variant under a different-cased repo id resolves to the same
    # job, so the second claim attaches to the running one instead of starting
    # a duplicate.
    duplicate_claimed, duplicate_state = registry.claim(
        "org/repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "org/repo",
        variant = "Q8_0",
    )

    assert claimed is True
    assert state == "running"
    assert duplicate_claimed is False
    assert duplicate_state == "running"
    assert registry.active_jobs("ORG/REPO") == {"org/repo::Q8_0": "running"}


def test_download_registry_allows_disjoint_gguf_variant_downloads():
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
        blob_hashes = frozenset({"q8-main"}),
        progress_blob_hashes = frozenset({"q8-main", "shared-mmproj"}),
    )
    second_claimed, second_state = registry.claim(
        "Org/Repo::Q4_K_M",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"q4-main"}),
        progress_blob_hashes = frozenset({"q4-main", "shared-mmproj"}),
    )

    assert claimed is True
    assert state == "running"
    assert second_claimed is True
    assert second_state == "running"
    assert registry.active_jobs("org/repo") == {
        "org/repo::Q8_0": "running",
        "org/repo::Q4_K_M": "running",
    }


def test_download_registry_allows_overlapping_same_transport_variant_downloads():
    # Two variants sharing one mmproj blob still download together on one
    # transport: huggingface_hub's per-blob lock serializes the shared write and
    # prepare_cache_for_transport never purges a blob a peer is writing.
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
        blob_hashes = frozenset({"q8-main"}),
        progress_blob_hashes = frozenset({"q8-main", "shared-mmproj"}),
    )
    second_claimed, second_state = registry.claim(
        "Org/Repo::Q4_K_M",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"q4-main"}),
        progress_blob_hashes = frozenset({"q4-main", "shared-mmproj"}),
    )

    assert claimed is True
    assert state == "running"
    assert second_claimed is True
    assert second_state == "running"


def test_download_registry_variant_delete_does_not_block_sibling_download():
    # Deleting one quant's partial must be allowed while a different quant of the
    # same repo is downloading, and must protect every blob the live sibling is
    # writing (including a shared mmproj companion).
    registry = download_registry.DownloadRegistry()
    registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
        blob_hashes = frozenset({"q8-main"}),
        progress_blob_hashes = frozenset({"q8-main", "shared-mmproj"}),
    )

    # A sibling variant delete is allowed; deleting the in-flight variant is not.
    assert registry.begin_delete("Org/Repo", "Q4_K_M") is True
    assert registry.begin_delete("Org/Repo", "Q8_0") is False
    # A whole-repo delete still waits for every active download.
    assert registry.begin_delete("Org/Repo") is False

    # The live sibling is detected so the delete keeps the shared companion.
    assert registry.has_active_peer_variant("Org/Repo", "Q4_K_M") is True
    assert registry.has_active_peer_variant("Org/Repo", "Q8_0") is False

    # While Q4_K_M is being deleted, re-downloading it is blocked but an
    # untouched third variant may still start.
    blocked, blocked_state = registry.claim(
        "Org/Repo::Q4_K_M",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
    )
    assert blocked is False
    assert blocked_state == "deleting"
    started, started_state = registry.claim(
        "Org/Repo::Q5_K_M",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q5_K_M",
    )
    assert started is True
    assert started_state == "running"

    registry.end_delete("Org/Repo", "Q4_K_M")
    assert registry.begin_delete("Org/Repo", "Q4_K_M") is True


def test_partial_gguf_reconstruction_dedupes_variant_casing(monkeypatch):
    # The manifest keeps original casing while the marker is lowercased; offline
    # reconstruction must collapse them to ONE entry (manifest's casing), not two.
    monkeypatch.setattr(
        download_manifest,
        "iter_variant_manifests",
        lambda _repo_type, _repo_id: iter([("Q4_K_M", Path("manifest.json"))]),
    )
    monkeypatch.setattr(
        download_manifest,
        "iter_variant_markers",
        lambda _repo_type, _repo_id: iter([("q4_k_m", Path("marker.json"))]),
    )
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *_a, **_k: None)

    result = gguf.list_partial_gguf_variants_from_state("Org/Repo")

    assert result is not None
    variants, _has_vision = result
    assert [variant.quant for variant in variants] == ["Q4_K_M"]


def test_download_registry_serializes_cross_transport_variant_downloads():
    # An HTTP append-resume and an XET rewrite of the same shared blob would
    # corrupt each other, so different-transport variants are serialized.
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
        blob_hashes = frozenset({"q8-main"}),
        progress_blob_hashes = frozenset({"q8-main", "shared-mmproj"}),
    )
    second_claimed, second_state = registry.claim(
        "Org/Repo::Q4_K_M",
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"q4-main"}),
        progress_blob_hashes = frozenset({"q4-main", "shared-mmproj"}),
    )

    assert claimed is True
    assert state == "running"
    assert second_claimed is False
    assert second_state == "running"


def test_download_registry_allows_unknown_hash_gguf_variant_downloads():
    # Resolved blob hashes are NOT required to run two same-transport variants
    # concurrently: on-disk safety comes from each worker purging only its own
    # main-quant blobs plus huggingface_hub's per-etag lock. Requiring them here
    # used to reject the second variant whenever a metadata fetch flaked.
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Repo::Q8_0",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q8_0",
    )
    second_claimed, second_state = registry.claim(
        "Org/Repo::Q4_K_M",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"q4-main"}),
        progress_blob_hashes = frozenset({"q4-main", "shared-mmproj"}),
    )

    assert claimed is True
    assert state == "running"
    assert second_claimed is True
    assert second_state == "running"
    assert registry.active_jobs("org/repo") == {
        "org/repo::Q8_0": "running",
        "org/repo::Q4_K_M": "running",
    }


def test_finalize_worker_exit_never_kills_a_healthy_worker(monkeypatch, tmp_path):
    # finalize_worker_exit relies solely on the worker's exit code and never kills
    # a live process: huggingface_hub already bounds reads with timeouts, so a
    # liveness kill could only false-cancel a healthy download.
    import inspect
    import io
    import logging

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    class _Proc:
        pid = 4242

        def __init__(self):
            self.killed = False
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return 0

        def wait(self, timeout = None):
            return 0

        def kill(self):
            self.killed = True

    registry = download_registry.DownloadRegistry()
    proc = _Proc()
    key = "Org/Repo::Q4_K_M"
    registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = "Q4_K_M",
    )
    registry.register_process(key, proc)

    download_lifecycle.finalize_worker_exit(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Repo [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Repo",
        transport = "http",
    )

    assert proc.killed is False
    assert registry.get_job(key).state == "complete"
    # The stall-watchdog knob is gone entirely; no caller may re-enable it.
    assert (
        "enable_stall_watchdog"
        not in inspect.signature(download_lifecycle.finalize_worker_exit).parameters
    )


def test_prepare_cache_for_transport_purges_only_requested_hashes(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    blobs = root / "models--Org--Repo" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "variant-main.incomplete").write_bytes(b"x")
    (blobs / "shared-mmproj.incomplete").write_bytes(b"y")
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda create = False: root)

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Repo",
        download_registry.TRANSPORT_XET,
        "Q4_K_M",
        frozenset({"variant-main"}),
    )

    assert purged == 1
    assert not (blobs / "variant-main.incomplete").exists()
    assert (blobs / "shared-mmproj.incomplete").exists()


def test_prepare_cache_for_transport_uses_captured_root(monkeypatch, tmp_path):
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    repo_name = "models--Org--Repo"
    partial_a = cache_a / repo_name / "blobs" / "blob.incomplete"
    partial_b = cache_b / repo_name / "blobs" / "blob.incomplete"
    partial_a.parent.mkdir(parents = True)
    partial_b.parent.mkdir(parents = True)
    partial_a.write_bytes(b"a")
    partial_b.write_bytes(b"b")
    monkeypatch.setattr(
        download_registry,
        "hf_cache_root",
        lambda create = False, root = None: root or cache_b,
    )

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Repo",
        download_registry.TRANSPORT_HTTP,
        root = cache_a,
    )

    assert purged == 1
    assert not partial_a.exists()
    assert partial_b.exists()


def _vision_cache_root(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    blobs = root / "models--Org--Vision" / "blobs"
    blobs.mkdir(parents = True)
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda create = False: root)
    return blobs


def test_prepare_cache_for_transport_purges_cross_transport_companion(monkeypatch, tmp_path):
    blobs = _vision_cache_root(monkeypatch, tmp_path)
    companion = frozenset({"shared-mmproj"})

    # An interrupted XET download stamps the companion marker "xet" and leaves a
    # sparse partial. A later HTTP download of a different variant must purge it,
    # else the HTTP resumer appends to the sparse bytes and corrupts the blob.
    download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_XET,
        "Q4_K_M",
        only_blob_hashes = frozenset({"q4-main"}),
        companion_blob_hashes = companion,
    )
    (blobs / "shared-mmproj.incomplete").write_bytes(b"sparse")

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_HTTP,
        "Q8_0",
        only_blob_hashes = frozenset({"q8-main"}),
        companion_blob_hashes = companion,
    )

    assert purged == 1
    assert not (blobs / "shared-mmproj.incomplete").exists()


def test_prepare_cache_for_transport_preserves_same_transport_companion(monkeypatch, tmp_path):
    blobs = _vision_cache_root(monkeypatch, tmp_path)
    companion = frozenset({"shared-mmproj"})

    download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({"q4-main"}),
        companion_blob_hashes = companion,
    )
    (blobs / "shared-mmproj.incomplete").write_bytes(b"resumable")

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({"q4-main"}),
        companion_blob_hashes = companion,
    )

    assert purged == 0
    assert (blobs / "shared-mmproj.incomplete").exists()


def test_prepare_cache_for_transport_protects_peer_companion(monkeypatch, tmp_path):
    blobs = _vision_cache_root(monkeypatch, tmp_path)
    companion = frozenset({"shared-mmproj"})

    download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_XET,
        "Q4_K_M",
        only_blob_hashes = frozenset({"q4-main"}),
        companion_blob_hashes = companion,
    )
    (blobs / "shared-mmproj.incomplete").write_bytes(b"sparse")

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_HTTP,
        "Q8_0",
        only_blob_hashes = frozenset({"q8-main"}),
        companion_blob_hashes = companion,
        protected_blob_hashes = companion,
    )

    assert purged == 0
    assert (blobs / "shared-mmproj.incomplete").exists()


def test_model_download_records_completed_baseline_for_new_gguf_variant(monkeypatch, tmp_path):
    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, repo_type = "model": repo_id,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda _repo, _variant, _token = None, include_companions = True, **_kwargs: (
            frozenset({"mainhash", "mmprojhash"}) if include_companions else frozenset({"mainhash"})
        ),
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "completed_blob_bytes",
        lambda *_args, **_kwargs: 30,
    )

    class _Registry:
        claim_kwargs = None

        def claim(self, _key, _transport, **kwargs):
            self.claim_kwargs = kwargs
            return True, "running"

        def current_generation(self, _key):
            return 1

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

        def register_process(self, _key, _proc):
            return False

        def peer_blob_hashes(self, _key):
            return frozenset()

    class _Proc:
        pid = 123
        stderr = None

        def poll(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout = None):
            return 0

    registry = _Registry()
    monkeypatch.setattr(downloads, "_registry", registry)
    monkeypatch.setattr(downloads, "_spawn_download_worker", lambda *_args, **_kwargs: _Proc())

    asyncio.run(
        downloads.download_model_response(
            SimpleNamespace(repo_id = "Org/Model", gguf_variant = "Q4_K_M", use_xet = False)
        )
    )

    assert registry.claim_kwargs["blob_hashes"] == frozenset({"mainhash"})
    assert registry.claim_kwargs["progress_blob_hashes"] == frozenset({"mainhash", "mmprojhash"})
    assert registry.claim_kwargs["completed_baseline_bytes"] == 30


def test_gguf_model_download_skips_completed_baseline_for_variant_resume_state(
    monkeypatch, tmp_path
):
    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(
                path = "model-Q4_K_M.gguf",
                size = 100,
                sha256 = "mainhash",
            )
        ],
        "http",
    )
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, repo_type = "model": repo_id,
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda _repo, _variant, _token = None, include_companions = True, **_kwargs: (
            frozenset({"mainhash", "mmprojhash"}) if include_companions else frozenset({"mainhash"})
        ),
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "completed_blob_bytes",
        lambda *_args, **_kwargs: 30,
    )

    class _Registry:
        claim_kwargs = None

        def claim(self, _key, _transport, **kwargs):
            self.claim_kwargs = kwargs
            return True, "running"

        def current_generation(self, _key):
            return 1

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

        def register_process(self, _key, _proc):
            return False

        def peer_blob_hashes(self, _key):
            return frozenset()

    class _Proc:
        pid = 123
        stderr = None

        def poll(self):
            return None

        def kill(self):
            return None

        def wait(self, timeout = None):
            return 0

    registry = _Registry()
    monkeypatch.setattr(downloads, "_registry", registry)
    monkeypatch.setattr(downloads, "_spawn_download_worker", lambda *_args, **_kwargs: _Proc())

    asyncio.run(
        downloads.download_model_response(
            SimpleNamespace(repo_id = "Org/Model", gguf_variant = "Q4_K_M", use_xet = False)
        )
    )

    assert registry.claim_kwargs["completed_baseline_bytes"] == 0


def test_model_idle_status_uses_cancel_marker_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(downloads, "_registry", download_registry.DownloadRegistry())
    assert download_manifest.write_cancel_marker("model", "Owner/Repo", "Q4_K_M", "http")

    status = asyncio.run(downloads.get_download_status_response("owner/repo", "Q4_K_M"))

    assert status.state == "cancelled"
    assert status.error is None


def test_shutdown_kills_all_workers_before_shared_deadline_reap(monkeypatch):
    events = []
    now = [100.0]

    class _Proc:
        def __init__(self, name):
            self.name = name

        def poll(self):
            return None

        def kill(self):
            events.append(("kill", self.name))

        def wait(self, timeout):
            events.append(("wait", self.name, timeout))
            now[0] += 7.0

    registry = download_registry.DownloadRegistry()
    proc_a = _Proc("a")
    proc_b = _Proc("b")
    registry.claim(
        "Org/A",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/A",
    )
    registry.claim(
        "Org/B",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/B",
    )
    assert registry.register_process("org/a", proc_a)
    assert registry.register_process("org/b", proc_b)
    monkeypatch.setattr(
        download_registry,
        "persist_cancel_marker",
        lambda *args, **kwargs: events.append(("marker", args[1])),
    )
    monkeypatch.setattr(download_registry.time, "monotonic", lambda: now[0])

    registry.terminate_all("dataset download")

    assert events == [
        ("kill", "a"),
        ("kill", "b"),
        ("wait", "a", 10.0),
        ("marker", "Org/A"),
        ("wait", "b", 3.0),
        ("marker", "Org/B"),
    ]


def test_shutdown_skips_marker_for_worker_that_exits_cleanly(monkeypatch):
    markers = []

    class _Proc:
        def __init__(self, final_rc):
            self._final_rc = final_rc
            self._exited = False

        def poll(self):
            return self._final_rc if self._exited else None

        def kill(self):
            pass

        def wait(self, timeout):
            self._exited = True

    registry = download_registry.DownloadRegistry()
    clean = _Proc(0)
    interrupted = _Proc(-9)
    registry.claim(
        "Org/Clean",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Clean",
    )
    registry.claim(
        "Org/Cut",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Cut",
    )
    assert registry.register_process("org/clean", clean)
    assert registry.register_process("org/cut", interrupted)
    monkeypatch.setattr(
        download_registry,
        "persist_cancel_marker",
        lambda *args, **kwargs: markers.append(args[1]),
    )

    registry.terminate_all("dataset download")

    assert markers == ["Org/Cut"]


def test_orphan_reaper_uses_worker_cache_root_after_setting_changes(monkeypatch, tmp_path):
    workers = tmp_path / "workers"
    workers.mkdir()
    cache_a = tmp_path / "cache-a" / "hub"
    cache_b = tmp_path / "cache-b" / "hub"
    partial = cache_a / "models--Org--Model" / "blobs" / "abc.incomplete"
    partial.parent.mkdir(parents = True)
    partial.write_bytes(b"partial")
    cache_b.mkdir(parents = True)
    monkeypatch.setattr(state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "_process_alive", lambda _pid: False)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_b),
    )
    markers = []
    monkeypatch.setattr(
        download_registry,
        "persist_cancel_marker",
        lambda *args, **kwargs: markers.append(args),
    )
    metadata = download_registry.DownloadMetadata(
        repo_type = "model",
        repo_id = "Org/Model",
        variant = None,
        transport = download_registry.TRANSPORT_HTTP,
        hub_cache = str(cache_a),
        xet_cache = str(tmp_path / "cache-a" / "xet"),
    )
    download_registry.write_worker_breadcrumb("org/model", 1234, metadata)
    [breadcrumb] = list(workers.iterdir())
    payload = json.loads(breadcrumb.read_text(encoding = "utf-8"))
    assert payload["hub_cache"] == str(cache_a)
    assert payload["xet_cache"] == str(tmp_path / "cache-a" / "xet")

    download_registry.reap_orphan_workers()

    assert markers == [("model", "Org/Model", None, "http")]
    assert list(workers.iterdir()) == []


def test_model_claim_register_cancel_uses_registry_marker_owner(monkeypatch):
    killed = []

    class _Registry:
        def claim(self, *_args, **_kwargs):
            return True, "running"

        def current_generation(self, _key):
            return 1

        def register_process(self, _key, _proc):
            return False

        def persist_cancel_for_key(self, *_args, **_kwargs):
            raise AssertionError("register_process owns pending-cancel markers")

        def get_job(self, _key):
            return SimpleNamespace(state = "cancelled", error = None)

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: None,
    )
    monkeypatch.setattr(
        downloads,
        "_spawn_download_worker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "kill_and_reap_process",
        lambda proc, **_kwargs: killed.append(proc),
    )

    result = asyncio.run(
        downloads.download_model_response(
            SimpleNamespace(repo_id = "Org/Model", gguf_variant = None, use_xet = False)
        )
    )

    assert result["state"] == "cancelled"
    assert killed


def test_model_cancel_registered_worker_requests_and_kills(monkeypatch):
    events = []

    class _Proc:
        def poll(self):
            return None

        def kill(self):
            events.append(("kill",))

    class _Registry:
        def get_process(self, _key):
            return _Proc()

        def request_cancel(self, key, _proc, generation):
            events.append(("request", key, generation))
            return True

        def persist_cancel_for_key(self, *_args, **_kwargs):
            raise AssertionError(
                "cancel_worker must leave marker persistence to the exit watcher; "
                "an eager persist races a clean completion and strands a stale marker"
            )

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )

    result = asyncio.run(
        downloads.cancel_download_model_response(
            SimpleNamespace(repo_id = "Org/Model", gguf_variant = "Q4_K_M", generation = 7)
        )
    )

    assert result == {
        "job_key": downloads._download_job_key("Org/Model", "Q4_K_M"),
        "state": "cancelling",
    }
    assert events == [
        ("request", downloads._download_job_key("Org/Model", "Q4_K_M"), 7),
        ("kill",),
    ]


def test_model_download_watcher_invalidates_hf_cache_scan(monkeypatch):
    invalidated = []

    class _Registry:
        def claim(self, *_args, **_kwargs):
            return True, "running"

        def current_generation(self, _key):
            return 1

        def register_process(self, _key, _proc):
            return True

        def get_job(self, _key):
            return SimpleNamespace(state = "complete", error = None)

    class _ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: None,
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "finalize_worker_exit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        downloads,
        "_spawn_download_worker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(downloads.download_lifecycle.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        downloads.hf_cache_scan,
        "invalidate_hf_cache_scans",
        lambda: invalidated.append(True),
    )

    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _inline_to_thread)

    result = asyncio.run(
        downloads.download_model_response(
            SimpleNamespace(repo_id = "Org/Model", gguf_variant = None, use_xet = False)
        )
    )

    assert result["accepted"] is True
    assert invalidated == [True]


def test_two_concurrent_same_repo_variants_both_complete(monkeypatch, tmp_path):
    # End-to-end proof that two GGUF variants of ONE repo download concurrently
    # without cancelling each other, with real registry/finalize/subprocess/watch
    # threads exercising the claim gate, register, finalize funnel, and
    # classify_exit under true concurrency.
    import subprocess
    import time

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        downloads,
        "_registry",
        download_registry.DownloadRegistry(),
    )
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_k: repo_id,
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: None,
    )
    # Per-variant blob hashes (distinct main shard, shared mmproj companion).
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda _repo, variant, _token = None, include_companions = True, **_k: (
            frozenset({f"{variant.lower()}-main", "shared-mmproj"})
            if include_companions
            else frozenset({f"{variant.lower()}-main"})
        ),
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "completed_blob_bytes",
        lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        downloads.hf_cache_scan,
        "invalidate_hf_cache_scans",
        lambda: None,
    )
    # Real subprocess that exits 0 immediately, with a stderr pipe to drain.
    spawned: list[subprocess.Popen] = []

    def _fake_spawn(*_args, **_kwargs):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stderr = subprocess.PIPE,
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(downloads, "_spawn_download_worker", _fake_spawn)

    async def _run_both():
        return await asyncio.gather(
            downloads.download_model_response(
                SimpleNamespace(
                    repo_id = "Org/Model",
                    gguf_variant = "Q4_K_M",
                    use_xet = False,
                )
            ),
            downloads.download_model_response(
                SimpleNamespace(
                    repo_id = "Org/Model",
                    gguf_variant = "Q8_0",
                    use_xet = False,
                )
            ),
        )

    results = asyncio.run(_run_both())
    assert all(r["accepted"] is True for r in results), results

    registry = downloads._registry
    key_q4 = downloads._download_job_key("Org/Model", "Q4_K_M")
    key_q8 = downloads._download_job_key("Org/Model", "Q8_0")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        s4 = registry.get_job(key_q4).state
        s8 = registry.get_job(key_q8).state
        if s4 in download_registry.TERMINAL_STATES and s8 in download_registry.TERMINAL_STATES:
            break
        time.sleep(0.02)

    for p in spawned:
        try:
            p.wait(timeout = 5)
        except Exception:
            pass

    assert registry.get_job(key_q4).state == "complete"
    assert registry.get_job(key_q8).state == "complete"


def test_download_registry_factories_reuse_service_singletons():
    registry_module = downloads.download_registry
    before_count = len(registry_module._REGISTRIES)

    assert registry_module.get_models_registry() is downloads.registry
    assert registry_module.get_models_registry() is downloads.registry
    assert registry_module.get_datasets_registry() is dataset_downloads.registry
    assert registry_module.get_datasets_registry() is dataset_downloads.registry
    assert len(registry_module._REGISTRIES) == before_count


def test_hub_hf_token_header_uses_namespaced_header_only():
    assert get_hf_token("new-token") == "new-token"
    assert get_hf_token(None) is None


def test_scan_folder_rejects_credential_directories(tmp_path):
    sensitive_dir = tmp_path / ".ssh" / "models"
    sensitive_dir.mkdir(parents = True)

    with pytest.raises(ValueError, match = "Credential or configuration"):
        scan_folders.add_scan_folder(str(sensitive_dir))


def _build_variant_cache_repo(repo_dir, blob_specs, snapshot_links):
    """Build a HF cache repo dir with blobs + snapshot symlinks for the
    per-variant deletion path. blob_specs: {blob_name: bytes_payload};
    snapshot_links: list of (revision, filename, blob_name)."""
    blobs_dir = repo_dir / "blobs"
    blobs_dir.mkdir(parents = True)
    for blob_name, payload in blob_specs.items():
        (blobs_dir / blob_name).write_bytes(payload)

    files = []
    for revision, filename, blob_name in snapshot_links:
        snap_dir = repo_dir / "snapshots" / revision
        snap_dir.mkdir(parents = True, exist_ok = True)
        blob = blobs_dir / blob_name
        link = snap_dir / filename
        link.symlink_to(blob)
        files.append(
            SimpleNamespace(
                file_name = filename,
                file_path = str(link),
                blob_path = str(blob),
                size_on_disk = blob.stat().st_size,
            )
        )
    repo = SimpleNamespace(
        repo_id = "Org/Repo-GGUF",
        repo_type = "model",
        repo_path = repo_dir,
        revisions = [SimpleNamespace(commit_hash = "rev1", files = files)],
    )
    return repo


def _patch_variant_delete_side_effects(monkeypatch, hub_cache = None):
    monkeypatch.setattr(
        deletion.download_manifest,
        "purge_state",
        lambda *_args, **_kwargs: False,
    )
    # The repo under test lives in this cache; make it the active one so the
    # delete scopes to it (default target root is the active hub cache).
    if hub_cache is not None:
        monkeypatch.setattr(
            "utils.hf_cache_settings.get_hf_cache_paths",
            lambda: SimpleNamespace(hub_cache = hub_cache),
        )


def test_snapshot_progress_filters_stale_blobs(monkeypatch, tmp_path):
    """Exclude superseded-revision blobs; count an in-progress blob only when its
    hash belongs to the target."""
    entry = tmp_path / "datasets--Org--Data"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "keep1").write_bytes(b"a" * 100)
    (blobs / "stale").write_bytes(b"b" * 500)
    (blobs / "keep2.incomplete").write_bytes(b"c" * 40)

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda _repo_type, _repo_id, force_active = False: [entry],
    )

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "dataset",
        repo_id = "Org/Data",
        job_key = "org/data",
        expected_bytes = 0,
        hf_token = None,
        registry = SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
        ),
        metadata_resolver = lambda _repo_id, _hf_token: (
            140,
            frozenset({"keep1", "keep2"}),
        ),
    )

    assert result["completed_bytes"] == 100
    assert result["downloaded_bytes"] == 140
    assert result["complete_on_disk"] is False
    assert result["expected_bytes"] == 140


def test_snapshot_progress_confirms_complete_only_with_verified_snapshot(monkeypatch, tmp_path):
    entry = tmp_path / "models--Org--Model"
    blobs = entry / "blobs"
    snap = entry / "snapshots" / "rev0"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)
    (blobs / "keep1").write_bytes(b"a" * 100)
    (snap / "model.safetensors").write_bytes(b"a" * 100)

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda _repo_type, _repo_id, force_active = False: [entry],
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "has_cancel_marker",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "read_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "verify_against_disk",
        lambda *_args, **_kwargs: SimpleNamespace(ok = True),
    )

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model",
        job_key = "org/model::",
        expected_bytes = 100,
        hf_token = None,
        registry = SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "idle"),
        ),
        metadata_resolver = lambda _repo_id, _hf_token: (
            100,
            frozenset({"keep1"}),
        ),
    )

    assert result["completed_bytes"] == 100
    assert result["complete_on_disk"] is True


def test_expected_files_from_snapshot_dir_records_relative_paths_and_sizes(tmp_path):
    snap = tmp_path / "snapshots" / "rev0"
    (snap / "nested").mkdir(parents = True)
    (snap / "model.safetensors").write_bytes(b"a" * 12)
    (snap / "nested" / "config.json").write_bytes(b"b" * 3)

    files = download_manifest.expected_files_from_snapshot_dir(snap)

    by_path = {f.path: f for f in files}
    assert by_path["model.safetensors"].size == 12
    assert by_path["nested/config.json"].size == 3
    assert all(f.sha256 is None for f in files)


def test_snapshot_progress_complete_with_manifest_synthesized_from_disk(monkeypatch, tmp_path):
    """A finished snapshot whose only manifest was synthesized from on-disk files
    still verifies as complete, so a refresh finalizes it instead of capping at
    99% and evicting it as gone."""
    entry = tmp_path / "models--Org--Model"
    blobs = entry / "blobs"
    snap = entry / "snapshots" / "rev0"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)
    (blobs / "keep1").write_bytes(b"a" * 100)
    (snap / "model.safetensors").write_bytes(b"a" * 100)

    synthesized = download_manifest.expected_files_from_snapshot_dir(snap)
    manifest = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "Org/Model",
        variant = None,
        started_at = "",
        expected_files = tuple(synthesized),
    )

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda _repo_type, _repo_id, force_active = False: [entry],
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "has_cancel_marker",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "read_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model",
        job_key = "org/model::",
        expected_bytes = 100,
        hf_token = None,
        registry = SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "idle"),
        ),
        metadata_resolver = lambda _repo_id, _hf_token: (
            100,
            frozenset({"keep1"}),
        ),
    )

    assert result["complete_on_disk"] is True
    assert result["progress"] == 1.0


def test_delete_variant_keeps_blob_shared_with_other_snapshot(monkeypatch, tmp_path):
    """A blob still referenced by a non-target snapshot symlink survives so that
    symlink doesn't dangle (which the scanner reports as partial)."""
    repo_dir = tmp_path / "models--Org--Repo-GGUF"
    repo = _build_variant_cache_repo(
        repo_dir,
        blob_specs = {"sharedblob": b"x" * 200, "q8blob": b"y" * 300},
        snapshot_links = [
            ("rev1", "model-Q4_K_M.gguf", "sharedblob"),
            ("rev1", "model-Q8_0.gguf", "q8blob"),
            # An unrelated file that happens to share Q4's blob content.
            ("rev1", "extra-copy.gguf", "sharedblob"),
        ],
    )
    monkeypatch.setattr(
        deletion.cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    _patch_variant_delete_side_effects(monkeypatch, tmp_path)

    result = deletion._delete_cached_model_blocking("Org/Repo-GGUF", "Q4_K_M", None)

    assert result["status"] == "deleted"
    # Q4 snapshot link gone, but its blob survives (extra-copy still links it).
    assert not (repo_dir / "snapshots" / "rev1" / "model-Q4_K_M.gguf").exists()
    assert (repo_dir / "blobs" / "sharedblob").exists()
    extra = repo_dir / "snapshots" / "rev1" / "extra-copy.gguf"
    assert extra.is_symlink() and extra.exists()  # not dangling


def test_delete_variant_unlinks_unshared_blob(monkeypatch, tmp_path):
    repo_dir = tmp_path / "models--Org--Repo-GGUF"
    repo = _build_variant_cache_repo(
        repo_dir,
        blob_specs = {"q4blob": b"x" * 200, "q8blob": b"y" * 300},
        snapshot_links = [
            ("rev1", "model-Q4_K_M.gguf", "q4blob"),
            ("rev1", "model-Q8_0.gguf", "q8blob"),
        ],
    )
    monkeypatch.setattr(
        deletion.cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    _patch_variant_delete_side_effects(monkeypatch, tmp_path)

    result = deletion._delete_cached_model_blocking("Org/Repo-GGUF", "Q4_K_M", None)

    assert result["status"] == "deleted"
    assert not (repo_dir / "blobs" / "q4blob").exists()
    # Untouched sibling variant remains fully intact.
    assert (repo_dir / "blobs" / "q8blob").exists()
    q8 = repo_dir / "snapshots" / "rev1" / "model-Q8_0.gguf"
    assert q8.is_symlink() and q8.exists()


def test_delete_variant_surfaces_locked_file_as_conflict(monkeypatch, tmp_path):
    """A blob unlink that fails (e.g. a Windows file lock on a loaded model)
    must raise a clear 409, not report a misleading success."""
    repo_dir = tmp_path / "models--Org--Repo-GGUF"
    repo = _build_variant_cache_repo(
        repo_dir,
        blob_specs = {"lockedblob": b"x" * 200},
        snapshot_links = [("rev1", "model-Q4_K_M.gguf", "lockedblob")],
    )
    monkeypatch.setattr(
        deletion.cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    _patch_variant_delete_side_effects(monkeypatch, tmp_path)

    real_unlink = Path.unlink

    def fake_unlink(self, *args, **kwargs):
        if self.name == "lockedblob":
            raise PermissionError("file in use")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    with pytest.raises(HTTPException) as exc_info:
        deletion._delete_cached_model_blocking("Org/Repo-GGUF", "Q4_K_M", None)

    assert exc_info.value.status_code == 409


def test_download_snapshot_writes_manifest_for_xet(monkeypatch, tmp_path):
    written = []
    verified = []

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "config.json", size = 12)]
        ),
    )
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    hf_download._download_snapshot("Org/Model", None, "xet")

    assert written, "XET snapshot download must still record a manifest"
    assert written[0][0:3] == ("model", "Org/Model", None)
    assert written[0][3][0].path == "config.json"
    assert verified == [("model", "Org/Model", None, str(tmp_path))]


def test_download_gguf_variant_writes_manifest_for_xet(monkeypatch, tmp_path):
    written = []
    verified = []

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [_sibling("model-Q4_K_M.gguf", 10, "main")]
        ),
    )
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    hf_download._download_gguf_variant("Org/Model", "Q4_K_M", None, "xet")

    assert written, "XET GGUF variant download must still record a manifest"
    assert written[0][0:3] == ("model", "Org/Model", "Q4_K_M")
    assert written[0][3][0].path == "model-Q4_K_M.gguf"
    assert verified == [("model", "Org/Model", "Q4_K_M", str(tmp_path))]


def test_download_dataset_writes_manifest_for_xet(monkeypatch, tmp_path):
    written = []
    verified = []

    monkeypatch.setattr(
        hf_download,
        "_dataset_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 30)]
        ),
    )
    monkeypatch.setattr(
        hf_download, "_verify_completed_download", lambda *args, **kwargs: verified.append(args)
    )
    monkeypatch.setattr(
        download_registry, "prepare_cache_for_transport", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_args: None)
    monkeypatch.setattr(
        download_manifest, "write_manifest", lambda *args: written.append(args) or True
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    hf_download._download_dataset("Org/Data", None, "xet")

    assert written, "XET dataset download must still record a manifest"
    assert written[0][0:3] == ("dataset", "Org/Data", None)
    assert written[0][3][0].path == "data.parquet"
    assert verified == [("dataset", "Org/Data", None, str(tmp_path))]


def test_dataset_status_includes_generation(monkeypatch):
    class _Registry:
        def get_job(self, _key):
            return SimpleNamespace(state = "running", error = None)

        def current_generation(self, _key):
            return 4

    monkeypatch.setattr(dataset_downloads, "_registry", _Registry())
    monkeypatch.setattr(
        dataset_downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )

    result = asyncio.run(dataset_downloads.get_dataset_download_status_response("Org/Data"))

    assert result.state == "running"
    assert result.generation == 4
