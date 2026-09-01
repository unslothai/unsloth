# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import errno
import json
import os
import sys
import threading
import time
from collections import Counter
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
    catalog_classification,
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


def _download_body(**over) -> SimpleNamespace:
    """A download request with every field the route reads.

    Built by hand rather than through the schema so these tests stay cheap, so it lists
    the newer scoping fields too: leaving one off reads as AttributeError inside the
    route, not as a missing default.
    """
    body = {
        "repo_id": "Org/Model",
        "gguf_variant": None,
        "use_xet": False,
        "scope_id": None,
        "files": None,
    }
    body.update(over)
    return SimpleNamespace(**body)


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


def test_gguf_inventory_dates_rows_from_managed_companions(tmp_path):
    repo = _repo(
        "Org/Vision-GGUF",
        [
            SimpleNamespace(
                file_name = "model-Q4_K_M.gguf",
                size_on_disk = 100,
                blob_path = None,
                blob_last_modified = 10.0,
            ),
            SimpleNamespace(
                file_name = "mmproj-F16.gguf",
                size_on_disk = 20,
                blob_path = None,
                blob_last_modified = 30.0,
            ),
        ],
        tmp_path,
    )

    assert cache_inventory._repo_gguf_last_modified(repo) == 30.0


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


@pytest.mark.parametrize(
    "repo", ["leejet/MiniMax-H3-GGUF", "unsloth/MiniMax-H3-GGUF", "UNSLOTH/minimax-h3-gguf"]
)
def test_minimax_h3_variant_filter_keeps_both_denoiser_partitions(repo):
    """Both H3 bundle repos need the filter, and the match is case-insensitive.

    The Unsloth mirror is the one the family and catalog now advertise, and it carries the
    Qwen3-VL encoder quants beside the denoisers, so leaving it off the list would aggregate a
    12 GB text encoder as if it were a selectable transformer quant.

    Both denoiser partitions stay: which one is picked IS the task, the loader's
    ``validate_h3_transformer_filename`` accepts either, and the community bundle repo publishes
    Ref2VA quants today. Filtering them out hid the whole reference-video path from the picker."""
    selectable = gguf._is_selectable_repo_gguf
    assert selectable(repo, "minimax_h3_fl2va-Q4_K_M.gguf")
    assert selectable(repo, "minimax_h3_fl2va_pruned-Q4_K_M.gguf")
    assert selectable(repo, "minimax_h3_fl2va-Q2_K_M.gguf")
    assert selectable(repo, "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf")
    assert selectable(repo, "minimax_h3_ref2va-Q4_K_M.gguf")
    assert selectable(repo, "minimax_h3_ref2va_pruned-Q2_K_M.gguf")
    # The companions are still never picks.
    assert not selectable(repo, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
    assert not selectable(repo, "qwen3vl_32b_minimax_h3-Q2_K_M.gguf")


def test_the_h3_native_repo_is_a_recognised_bundle_repo():
    """The repo the native loader downloads from must be one the filter knows about.

    These are set in different files, so a future repo move that updates only the loader would
    silently reintroduce the encoder-as-transformer aggregation this filter exists to prevent."""
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[2]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from core.inference.video_minimax_h3 import H3_GGUF_REPO

    assert gguf.is_h3_bundle_repo(H3_GGUF_REPO)


def test_minimax_h3_variant_labels_name_the_partition_and_build():
    variants = [
        gguf.GgufVariantInfo(
            filename = "minimax_h3_fl2va_pruned-Q4_K_M.gguf",
            quant = "minimax_h3_fl2va_pruned-Q4_K_M",
            size_bytes = 1,
        ),
        gguf.GgufVariantInfo(
            filename = "minimax_h3_ref2va-Q4_K_M.gguf",
            quant = "minimax_h3_ref2va-Q4_K_M",
            size_bytes = 1,
        ),
    ]

    gguf._apply_gguf_display_labels(variants)

    assert [variant.display_label for variant in variants] == [
        "Text & frames · Q4_K_M · Pruned",
        "References · Q4_K_M · Full",
    ]


def test_big_endian_detection_ignores_model_name_be_token():
    assert gguf.is_big_endian_gguf_path("model-Q4_K_M-be.gguf", "Q4_K_M")
    assert gguf.is_big_endian_gguf_path("model-Q4_K_M_be_infill.gguf", "Q4_K_M")
    assert not gguf.is_big_endian_gguf_path("foo-be-Q4_K_M.gguf", "Q4_K_M")
    assert not gguf.is_big_endian_gguf_path("Q4_K_M/foo-be.gguf", "Q4_K_M")
    assert gguf.pick_best_gguf(["model-Q4_K_M-be.gguf", "model-Q4_K_M.gguf"]) == (
        "model-Q4_K_M.gguf"
    )


def test_custom_inventory_filters_mtp_companions_at_registered_root(tmp_path, monkeypatch):
    root = tmp_path / "MTP"
    root.mkdir()
    main = root / "Qwen3.6-27B-MTP-Q6_K.gguf"
    model_dir = root / "model"
    model_dir.mkdir()
    (model_dir / "Qwen3.6-27B-MTP-Q8_0.GGUF").write_bytes(b"x")
    for file in (
        main,
        root / "gemma-4-12b-it-Q8_0-MTP.gguf",
        root / "mtp-gemma-4-12b-it.gguf",
    ):
        file.write_bytes(b"x")

    monkeypatch.setattr(gguf, "iter_gguf_files", lambda *_: pytest.fail("recursive scan"))
    rows = local_inventory._scan_custom_folder(root)

    paths = {Path(row.load_id) for row in rows}
    assert {main, model_dir} <= paths
    assert root / "gemma-4-12b-it-Q8_0-MTP.gguf" not in paths
    assert root / "mtp-gemma-4-12b-it.gguf" not in paths

    companion_root = tmp_path / "companion" / "MTP"
    (companion_root / "other.gguf").mkdir(parents = True)
    (companion_root / "config.json").write_bytes(b"{}")
    (companion_root / "gemma-4-12b-it-Q8_0-MTP.gguf").write_bytes(b"x")
    (companion_root / "other.gguf" / "Qwen3.6-27B-Q8_0.gguf").write_bytes(b"x")
    assert companion_root not in {
        Path(row.load_id) for row in local_inventory._scan_custom_folder(companion_root)
    }


def test_custom_inventory_filters_dspark_companions_at_registered_root(tmp_path):
    # Registering dspark/ as the scan root strips it from every relative path, so the basename prefix
    # carries the exclusion.
    root = tmp_path / "dspark"
    root.mkdir()
    for name in (
        "dspark-DeepSeek-V4-Flash-0731-BF16.gguf",
        "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
    ):
        (root / name).write_bytes(b"x")

    assert local_inventory._scan_custom_folder(root) == []


def test_custom_inventory_groups_nested_gguf_files_once(tmp_path):
    """A model folder is one picker row whose GGUF files are its variants.

    The generic custom-folder scan already publishes that directory.  The LM
    Studio compatibility pass must not also publish every GGUF inside it as a
    second top-level model.
    """
    root = tmp_path / "hub"
    model_dir = root / "qwen38-27b-qat"
    model_dir.mkdir(parents = True)
    for quant in ("q2_0", "q3_k_m"):
        (model_dir / f"qwen38-27b-qat-{quant}.gguf").write_bytes(b"GGUF")

    rows = local_inventory._scan_custom_folder(root)

    assert [Path(row.path) for row in rows] == [model_dir]


def test_custom_inventory_keeps_lmstudio_publisher_model_layout(tmp_path):
    root = tmp_path / "lmstudio"
    model_dir = root / "publisher" / "model"
    model_dir.mkdir(parents = True)
    (model_dir / "model-q4_k_m.gguf").write_bytes(b"GGUF")

    rows = local_inventory._scan_custom_folder(root)

    assert [(Path(row.path), row.model_id) for row in rows] == [(model_dir, "publisher/model")]


def test_custom_inventory_keeps_file_row_beneath_partial_group(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    model_dir = root / "model"
    model_dir.mkdir(parents = True)
    model_file = model_dir / "model-q4_k_m.gguf"
    model_file.write_bytes(b"GGUF")
    partial_group = model_common._local_model_info(
        scan_path = model_dir,
        load_path = model_dir,
        source = "hf_cache",
        model_format = "gguf",
        model_id = "publisher/model",
        partial = True,
    )
    complete_file = model_common._local_model_info(
        scan_path = model_file,
        load_path = model_file,
        source = "lmstudio",
        model_format = "gguf",
    )
    monkeypatch.setattr(local_inventory, "_scan_models_dir", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        local_inventory,
        "_scan_hf_cache",
        lambda *_a, **_kw: [partial_group],
    )
    monkeypatch.setattr(
        local_inventory,
        "_scan_lmstudio_dir",
        lambda *_a, **_kw: [complete_file],
    )
    monkeypatch.setattr(local_inventory, "scan_ollama_dir", lambda *_a, **_kw: [])

    assert local_inventory._scan_custom_folder(root) == [partial_group, complete_file]


def test_unregistered_variant_identity_stays_scan_relative(tmp_path):
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "deadbeef"
    snapshot.mkdir()
    (snapshot / "model.gguf").write_bytes(b"x")

    assert [v.quant for v in list_local_gguf_variants(str(snapshot))[0]] == ["model"]
    assert [v.quant for v in gguf.list_local_gguf_variants(str(snapshot))[0]] == ["model"]


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


def test_read_only_download_state_lookup_does_not_create_directories(monkeypatch, tmp_path):
    studio_cache, hub_cache = tmp_path / "studio-cache", tmp_path / "hub-cache"
    monkeypatch.setattr(state_dir, "cache_root", lambda: studio_cache)
    state_key = ("model", "Org/Model", "Q4_K_M")
    assert download_manifest.read_manifest(*state_key, hub_cache = hub_cache) is None
    assert not download_manifest.has_cancel_marker(*state_key, hub_cache = hub_cache)
    assert not studio_cache.exists()

    state_root = studio_cache / "hub-state"
    for dirname in ("manifests", "cancelled"):
        (state_root / dirname).mkdir(parents = True)
    download_manifest.build_variant_state_index(
        [("model", "Org/Model", hub_cache)], active_hub_cache = hub_cache
    )
    scope = state_dir.cache_scope_name(hub_cache)
    assert all(
        not (state_root / dirname / scope).exists() for dirname in ("manifests", "cancelled")
    )


def test_cached_gguf_inventory_indexes_and_owns_polluted_state_once(monkeypatch, tmp_path):
    studio_cache = tmp_path / "studio-cache"
    hub_caches = [tmp_path / "cache-a", tmp_path / "cache-b"]
    hub_cache = hub_caches[0]
    monkeypatch.setattr(state_dir, "cache_root", lambda: studio_cache)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    repo_groups = [[], []]
    repo_ids = ["variant/Model-0", "owner", "owner/variant"]
    for index, repo_id in enumerate(repo_ids):
        repo_cache = hub_caches[min(index, 1)]
        repo_path = repo_cache / f"models--{repo_id.replace('/', '--')}"
        repo_groups[min(index, 1)].append(
            SimpleNamespace(repo_id = repo_id, repo_type = "model", repo_path = repo_path)
        )
        assert download_manifest.write_manifest(
            "model",
            repo_id,
            "Q4_K_M",
            [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = index + 1)],
            "http",
            hub_cache = repo_cache,
        )
        if index != 1:
            assert download_manifest.write_cancel_marker(
                "model", repo_id, "Q4_K_M", "http", hub_cache = repo_cache
            )

    def marker_variants(repo_id):
        return [
            variant
            for variant, _path in download_manifest.iter_variant_markers(
                "model", repo_id, hub_cache = hub_caches[1]
            )
        ]

    def state_marker(repo_id, variant, **kwargs):
        return state_dir.marker_path("model", repo_id, variant, hub_cache = hub_caches[1], **kwargs)

    def indexed_state(index, repo_id):
        return index.for_repo("model", repo_id, hub_cache = hub_caches[1])

    ambiguous = state_marker(repo_ids[2], "Q4_K_M", legacy_repo_key = True)
    long_marker = state_marker("owner/variant", "Q4_K_M")
    old_long_marker = state_marker("owner/variant", "Q4_K_M", legacy_hash_key = True)
    literal_hash_repo = old_long_marker.stem.split("--variant--", 1)[0].removeprefix("models--")
    assert len({long_marker, old_long_marker, state_marker(literal_hash_repo, "Q4_K_M")}) == 3
    long_marker.unlink()
    old_long_marker.write_text(json.dumps({"repo_id": "owner/variant", "variant": "Q4_K_M"}))
    for order in (("owner/variant", literal_hash_repo), (literal_hash_repo, "owner/variant")):
        state_index = download_manifest.build_variant_state_index(
            [("model", repo_id, hub_caches[1]) for repo_id in order],
            active_hub_cache = hub_caches[1],
        )
        assert indexed_state(state_index, "owner/variant").has_marker("Q4_K_M")
        assert indexed_state(state_index, literal_hash_repo).summary() == (False, 0)
    old_long_marker.unlink()
    old_snapshot = state_marker("owner/variant", None, legacy_hash_key = True)
    old_snapshot.write_text(json.dumps({"repo_id": literal_hash_repo}))
    assert not download_manifest.purge_state("model", "owner/variant", hub_cache = hub_caches[1])
    assert download_manifest.purge_state("model", literal_hash_repo, hub_cache = hub_caches[1])
    old_snapshot.write_text("[")
    for repo_id in ("owner/variant", literal_hash_repo):
        assert not download_manifest.purge_state("model", repo_id, hub_cache = hub_caches[1])
    old_snapshot.unlink()
    owner_marker = state_marker("owner", "variant--Q4_K_M")
    old_owner_marker = state_marker("owner", "variant--Q4_K_M", legacy_hash_key = True)
    literal_hash_variant = old_owner_marker.stem.rsplit("--variant--", 1)[-1]
    assert len({owner_marker, old_owner_marker, state_marker("owner", literal_hash_variant)}) == 3
    old_owner_marker.write_text(json.dumps({"repo_id": "owner", "variant": literal_hash_variant}))
    for variant, remains in (("variant--Q4_K_M", True), (literal_hash_variant, False)):
        download_manifest.clear_cancel_marker("model", "owner", variant, hub_cache = hub_caches[1])
        assert old_owner_marker.exists() is remains
    old_owner_marker.write_text("[")
    for variant in ("variant--Q4_K_M", literal_hash_variant):
        download_manifest.clear_cancel_marker("model", "owner", variant, hub_cache = hub_caches[1])
        assert old_owner_marker.is_file()
    old_owner_marker.unlink()
    ambiguous.write_text(json.dumps({"repo_id": "owner", "variant": "variant--Q4_K_M"}))
    assert download_manifest.write_cancel_marker(
        "model", "owner/variant", "Q4_K_M", "http", hub_cache = hub_caches[1]
    )
    assert download_manifest.has_cancel_marker(
        "model", "owner", "variant--Q4_K_M", hub_cache = hub_caches[1]
    )
    download_manifest.clear_cancel_marker(
        "model", "owner", "variant--Q4_K_M", hub_cache = hub_caches[1]
    )
    assert not ambiguous.exists()
    old_manifest = state_dir.manifest_path(
        "model",
        "migration/model",
        "legacy--Q4",
        hub_cache = hub_caches[1],
        legacy_hash_key = True,
    )
    old_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": "migration/model",
                "variant": "legacy--Q4",
                "expected_files": [{"path": "old.gguf", "size": 1}],
            }
        )
    )
    migrated = download_manifest.read_manifest(
        "model", "migration/model", "legacy--Q4", hub_cache = hub_caches[1]
    )
    assert migrated is not None and migrated.expected_files[0].size == 1
    literal_hash_variant = old_manifest.stem.rsplit("--variant--", 1)[-1]
    literal_manifest = download_manifest.read_manifest(
        "model", "migration/model", literal_hash_variant, hub_cache = hub_caches[1]
    )
    assert literal_manifest is None
    assert download_manifest.write_manifest(
        "model",
        "migration/model",
        "legacy--Q4",
        [download_manifest.ExpectedFile("new.gguf", 2)],
        hub_cache = hub_caches[1],
    )
    unscoped_manifest = state_dir.manifest_path("model", "migration/model", "legacy--Q4")
    assert unscoped_manifest is not None
    stale_payload = json.loads(old_manifest.read_text())
    stale_payload["expected_files"] = [{"path": "stale.gguf", "size": 3}]
    unscoped_manifest.write_text(json.dumps(stale_payload))
    migration = download_manifest.build_variant_state_index(
        [("model", "migration/model", hub_caches[1])], active_hub_cache = hub_caches[0]
    ).for_repo("model", "migration/model", hub_cache = hub_caches[1])
    assert migration.manifest_for("legacy--q4").expected_files[0].size == 2
    current_manifest = state_dir.manifest_path(
        "model", "migration/model", "legacy--Q4", hub_cache = hub_caches[1]
    )
    current_manifest.unlink()
    migration = download_manifest.build_variant_state_index(
        [("model", "migration/model", hub_caches[1])], active_hub_cache = hub_caches[1]
    ).for_repo("model", "migration/model", hub_cache = hub_caches[1])
    assert migration.manifest_for("legacy--q4").expected_files[0].size == 1
    unscoped_manifest.unlink()
    assert download_manifest.purge_state(
        "model", "migration/model", "legacy--Q4", hub_cache = hub_caches[1]
    )
    assert not old_manifest.exists()
    ambiguous.write_text(
        json.dumps({"repo_type": "model", "repo_id": "other/repo", "variant": "Q4_K_M"}),
        encoding = "utf-8",
    )
    assert download_manifest.has_cancel_marker(
        "model", "owner/variant", "Q4_K_M", hub_cache = hub_caches[1]
    )
    assert marker_variants("owner") == ["variant--q4_k_m"]
    assert marker_variants("owner/variant") == ["Q4_K_M"]
    partial = gguf.list_partial_gguf_variants_from_state("owner", hub_cache = hub_caches[1])
    offline = {variant.quant: variant.filename for variant in partial[0]}
    assert offline["variant--q4_k_m"] == "variant--q4_k_m.gguf"
    ambiguous.write_text(
        json.dumps({"repo_type": "model", "repo_id": "owner/variant", "variant": "Q8_0"})
    )
    assert marker_variants("owner/variant") == ["Q4_K_M"]
    mismatched = download_manifest.build_variant_state_index(
        [("model", repo_id, hub_caches[min(index, 1)]) for index, repo_id in enumerate(repo_ids)],
        active_hub_cache = hub_caches[0],
    ).for_repo("model", "owner/variant", hub_cache = hub_caches[1])
    assert mismatched.has_marker("q4_k_m") and not mismatched.has_marker("q8_0")
    ambiguous.write_text("[" * 10_000 + "0" + "]" * 10_000)
    manifest_root = studio_cache / "hub-state" / "manifests"
    marker_root = studio_cache / "hub-state" / "cancelled"
    watched = {manifest_root, marker_root, *manifest_root.iterdir(), *marker_root.iterdir()}
    calls, real_iterdir = Counter(), Path.iterdir

    def counting_iterdir(path):
        if path in watched:
            calls[path] += 1
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = repos) for repos in repo_groups],
    )
    monkeypatch.setattr(cache_inventory, "_cached_model_snapshot_path", lambda _path: None)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_size_bytes", lambda _repo: 1)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_last_modified", lambda _repo: 0)
    monkeypatch.setattr(cache_inventory, "_is_hidden_infra_repo", lambda *_args: False)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_payload_snapshots", lambda _repo: (None, ()))
    monkeypatch.setattr(cache_inventory, "_cache_inventory_fields", lambda *_args, **_kw: {})
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda repo_id, _path, *, variant_state, **_kw: variant_state.has_marker(
            "variant--q4_k_m" if repo_id == "owner" else "q4_k_m"
        ),
    )
    rows = cache_inventory._scan_cached_gguf()
    assert {(row["repo_id"], row["size_bytes"], row["partial"]) for row in rows} == {
        (repo_id, index + 1, True) for index, repo_id in enumerate(repo_ids)
    }
    assert calls == Counter({path: 1 for path in watched})
    assert download_manifest.write_manifest(
        "model", "owner", "variant--Q4_K_M", [], "http", hub_cache = hub_caches[1]
    )
    assert download_manifest.write_cancel_marker(
        "model", "owner/variant", "Q4_K_M", "http", hub_cache = hub_caches[1]
    )
    purge = download_manifest.purge_all_state_for_repo
    assert purge("model", "owner", hub_cache = hub_caches[1]) == 2
    assert ambiguous.is_file()
    assert download_manifest.write_cancel_marker(
        "model", "owner", "variant--Q4_K_M", "http", hub_cache = hub_caches[1]
    )
    owner_marker = state_marker("owner", "variant--Q4_K_M")
    assert owner_marker is not None and owner_marker.is_file()
    assert download_manifest.has_cancel_marker(
        "model", "owner/variant", "Q4_K_M", hub_cache = hub_caches[1]
    )
    download_manifest.clear_cancel_marker(
        "model", "owner/variant", "Q4_K_M", hub_cache = hub_caches[1]
    )
    assert download_manifest.purge_state(
        "model", "owner/variant", "Q4_K_M", hub_cache = hub_caches[1]
    )
    assert owner_marker.is_file()
    ambiguous.write_text("[" * 10_000 + "0" + "]" * 10_000)
    assert download_manifest.has_cancel_marker(
        "model", "owner/variant", "Q4_K_M", hub_cache = hub_caches[1]
    )
    assert not purge("model", "owner/variant", hub_cache = hub_caches[1])
    assert ambiguous.is_file()


@pytest.mark.parametrize("invalid_variant", [False, 0, [], {}])
def test_manifest_parser_rejects_non_text_v2_variant(invalid_variant):
    payload = {"version": 2, "repo_type": "model", "repo_id": "x"}
    payload.update(variant = invalid_variant, expected_files = [])
    assert download_manifest._manifest_from_payload(payload, "model", "x") is None


@pytest.mark.parametrize(
    ("inventory_request", "scanner_name"),
    [
        (cache_inventory.list_cached_gguf_response, "_scan_cached_gguf"),
        (cache_inventory.list_cached_models_response, "_scan_cached_models"),
    ],
)
def test_cached_inventory_requests_share_scan(monkeypatch, inventory_request, scanner_name):
    scans = submissions = 0
    started, releases = [asyncio.Event(), asyncio.Event()], [asyncio.Event(), asyncio.Event()]
    cached, epoch = [{"repo_id": "Org/Model"}], [0]

    async def fake_to_thread(function, *args):
        nonlocal submissions
        index = submissions
        submissions += 1
        started[index].set()
        await releases[index].wait()
        return function(*args)

    def scan(**_kwargs):
        nonlocal scans
        scans += 1
        return cached

    monkeypatch.setattr(cache_inventory.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(cache_inventory, scanner_name, scan)
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: [])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = Path("/cache")),
    )
    monkeypatch.setattr(cache_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])

    async def run_requests():
        first = asyncio.create_task(inventory_request())
        await asyncio.wait_for(started[0].wait(), 1)
        second = asyncio.create_task(inventory_request())
        for _ in range(2):
            await asyncio.sleep(0)
        assert submissions == 1
        first.cancel()
        await asyncio.gather(first, return_exceptions = True)
        epoch[0] += 1
        changed = asyncio.create_task(inventory_request())
        await asyncio.wait_for(started[1].wait(), 1)
        releases[0].set()
        for _ in range(2):
            await asyncio.sleep(0)
        releases[1].set()
        return await asyncio.gather(second, changed)

    assert asyncio.run(run_requests()) == [{"cached": cached, "scan_confirmed": True}] * 2
    assert scans == 1 and cache_inventory._cached_inventory_flights == {}


@pytest.mark.parametrize(
    "scanner_name, inventory_call",
    [
        ("_scan_cached_gguf", "gguf"),
        ("_scan_cached_models", "models"),
    ],
)
def test_cached_inventory_discards_a_scan_that_raced_an_invalidation(
    monkeypatch, scanner_name, inventory_call
):
    """A delete landing mid-scan must supersede the rows that scan produced.

    The pre-scan epoch check only covers the source read; the walk itself takes
    seconds, which is exactly when a download or deletion completes.
    """
    epoch, scans = [0], []

    def scan(**_kwargs):
        scans.append(1)
        if len(scans) == 1:
            epoch[0] += 1
            return [{"repo_id": "Org/Deleted"}]
        return [{"repo_id": "Org/Kept"}]

    monkeypatch.setattr(cache_inventory, scanner_name, scan)
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: [])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = Path("/cache")),
    )
    monkeypatch.setattr(cache_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])
    monkeypatch.setattr(cache_inventory, "_last_confirmed_inventory", {})

    result = asyncio.run(cache_inventory._shared_cached_inventory_scan(inventory_call, scan))
    assert result.rows == [{"repo_id": "Org/Kept"}], "rows from the superseded walk were served"
    assert result.confirmed is True
    assert len(scans) == 2, "the superseded walk was not retried"
    assert cache_inventory._cached_inventory_flights == {}


def test_cached_inventory_scan_stops_retrying_under_constant_invalidation(monkeypatch):
    """Bounded retries: an invalidation rate above the scan rate must still answer."""
    epoch, scans = [0], []

    def scan(**_kwargs):
        scans.append(1)
        epoch[0] += 1
        return [{"repo_id": "Org/Model"}]

    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: [])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = Path("/cache")),
    )
    monkeypatch.setattr(cache_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])
    monkeypatch.setattr(cache_inventory, "_last_confirmed_inventory", {})

    async def run():
        return await asyncio.wait_for(
            cache_inventory._shared_cached_inventory_scan("gguf", scan), timeout = 5
        )

    result = asyncio.run(run())
    assert result.rows == []
    assert result.confirmed is False
    assert len(scans) == cache_inventory._INVENTORY_SCAN_MAX_ATTEMPTS
    assert cache_inventory._cached_inventory_flights == {}


def test_shared_scan_scopes_tasks_to_event_loop():
    flights, results = {}, []
    barrier = threading.Barrier(3)

    async def factory():
        await asyncio.to_thread(barrier.wait)
        return "ok"

    def run():
        results.append(asyncio.run(inventory_scan.shared_scan(flights, "same", factory)))

    threads = [threading.Thread(target = run) for _ in range(2)]
    [thread.start() for thread in threads]
    barrier.wait(timeout = 1)
    [thread.join() for thread in threads]
    assert results == ["ok", "ok"] and flights == {}


def test_local_inventory_scan_stops_retrying_under_constant_invalidation(monkeypatch):
    """A churn rate above the scan rate must still answer, not walk the disk forever."""
    from hub.services.models import local_inventory

    epoch, scans = [0], []

    async def fake_scan(models_dir, custom_folders, sources):
        scans.append(1)
        epoch[0] += 1
        return SimpleNamespace(models = [], model_copy = lambda update = None: SimpleNamespace(models = []))

    async def no_folders():
        return []

    monkeypatch.setattr(local_inventory, "_scan_local_models_response", fake_scan)
    monkeypatch.setattr(local_inventory, "_load_custom_folders", no_folders)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: ("roots",))
    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])

    async def run():
        return await asyncio.wait_for(
            local_inventory.list_local_models_response("./models"), timeout = 5
        )

    response = asyncio.run(run())
    assert response is not None, "the capped loop must still answer"
    assert len(scans) == local_inventory._LOCAL_INVENTORY_MAX_ATTEMPTS
    assert local_inventory._local_inventory_flights == {}


def test_local_inventory_filters_and_dedupes_off_event_loop(monkeypatch):
    main_thread = threading.get_ident()
    worker_threads = []
    original_dedupe = local_inventory._dedupe_local_models

    async def collect(*_args, **_kwargs):
        return []

    def record_dedupe(rows):
        worker_threads.append(threading.get_ident())
        return original_dedupe(rows)

    monkeypatch.setattr(local_inventory, "_collect_models_from_default_sources", collect)
    monkeypatch.setattr(local_inventory, "_dedupe_local_models", record_dedupe)

    asyncio.run(
        local_inventory._scan_local_models_response(
            "./models", [], local_inventory._local_inventory_sources()
        )
    )

    assert worker_threads and worker_threads[0] != main_thread


@pytest.mark.parametrize("change_kind", ["folders", "epoch"])
def test_local_inventory_requests_share_scan(monkeypatch, change_kind):
    # Assert the property, not one platform's spelling: POSIX realpath() rejects an embedded NUL with
    # ValueError while Windows non-strict realpath falls back to abspath and joins the cwd.
    for hostile in ("\0", "\ud800"):
        identity = local_inventory._inventory_path_identity(hostile)
        assert identity == local_inventory._inventory_path_identity(hostile)
        assert identity.endswith(hostile)
    event = asyncio.Event
    calls, started, releases = 0, [event(), event()], [event(), event()]
    loaded, both_loaded, task_calls, epoch = 0, event(), [], [0]
    epoch_reads, retried = [0], event()
    model = SimpleNamespace(id = "model", path = "model")
    model.model_copy = lambda update: SimpleNamespace(id = model.id, path = model.path, **update)
    response = SimpleNamespace(models = [model])
    response.model_copy = lambda update: SimpleNamespace(models = update["models"])
    sources = local_inventory._local_inventory_sources()
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr(
        catalog_classification,
        "_local_model_task",
        lambda row: task_calls.append(row.id) or "task",
    )

    async def scan(*_args):
        nonlocal calls
        index = calls
        calls += 1
        started[index].set()
        await releases[index].wait()
        return response

    async def load_folders():
        nonlocal loaded
        loaded += 1
        if loaded == 2:
            both_loaded.set()
        return [] if loaded < 4 or change_kind == "epoch" else [{"path": "/changed"}]

    monkeypatch.setattr(local_inventory, "_load_custom_folders", load_folders)
    monkeypatch.setattr(local_inventory, "_scan_local_models_response", scan)

    def current_epoch():
        epoch_reads[0] += bool(epoch[0])
        if epoch_reads[0] == 3:
            retried.set()
        return epoch[0]

    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", current_epoch)

    async def run_requests():
        first = asyncio.create_task(local_inventory.list_local_models_response("./models"))
        await asyncio.wait_for(started[0].wait(), 1)
        second = asyncio.create_task(
            local_inventory.list_local_models_response(str(Path("models").resolve()))
        )
        await asyncio.wait_for(both_loaded.wait(), 1)
        await asyncio.sleep(0)
        assert calls == 1 and sources in next(iter(local_inventory._local_inventory_flights))[1]
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions = True)
        second = asyncio.create_task(local_inventory.list_local_models_response())
        await asyncio.sleep(0)
        epoch[0] += change_kind == "epoch"
        changed = asyncio.create_task(local_inventory.list_local_models_response())
        await asyncio.wait_for(started[1].wait(), 1)
        assert calls == 2
        releases[0].set()
        if change_kind == "epoch":
            await asyncio.wait_for(retried.wait(), 1)
        releases[1].set()
        return await asyncio.gather(second, changed)

    assert [response.models[0].task for response in asyncio.run(run_requests())] == ["task"] * 2
    expected_calls = ["model"] * (1 if change_kind == "epoch" else 2)
    assert task_calls == expected_calls and not local_inventory._local_inventory_flights


def test_local_inventory_indexes_registered_hf_state_once(monkeypatch, tmp_path):
    active, custom = tmp_path / "active", tmp_path / "custom"
    discoveries = {
        active: [(active / "models--Org--Active", "Org/Active", None)],
        custom: [(custom / "models--Org--Custom", "Org/Custom", None)],
    }
    monkeypatch.setattr(
        local_inventory, "_discover_hf_cache", lambda path, **_kw: discoveries[path]
    )
    monkeypatch.setattr(local_inventory, "_scan_models_dir", lambda *_args, **_kw: [])
    indexed = SimpleNamespace()
    builds, propagated = [], []
    monkeypatch.setattr(
        download_manifest,
        "build_variant_state_index",
        lambda repositories, **_kw: builds.append(tuple(repositories)) or indexed,
    )

    def record_state(
        *_args,
        variant_states = None,
        **_kwargs,
    ):
        propagated.append(variant_states)
        return []

    for name in ("_scan_hf_cache", "_scan_custom_folder"):
        monkeypatch.setattr(local_inventory, name, record_state)

    asyncio.run(
        local_inventory._collect_models_from_default_sources(
            tmp_path / "models",
            active,
            tmp_path / "missing-legacy",
            tmp_path / "missing-default",
            (),
            (),
            (),
            [{"path": str(custom)}],
        )
    )
    assert builds == [(("model", "Org/Active", active), ("model", "Org/Custom", custom))]
    assert propagated == [indexed, indexed]


def test_list_local_gguf_variants_skips_big_endian_sibling(tmp_path):
    (tmp_path / "model-Q4_K_M-be.gguf").write_bytes(b"x" * 100)
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"y" * 10)

    variants, has_vision = gguf.list_local_gguf_variants(str(tmp_path))

    assert has_vision is False
    assert [(v.quant, v.filename, v.size_bytes) for v in variants] == [
        ("Q4_K_M", "model-Q4_K_M.gguf", 10)
    ]


@pytest.mark.parametrize(
    "repo_id",
    ["bert-base-uncased", "owner/repo", "_owner/_repo_", "repo_"],
)
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


@pytest.mark.parametrize("variant", ["bad variant with spaces", "q" * 64, "q" * 65])
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

    assert "--@sha256-" in marker_path.name
    assert not state_dir.state_filename_is_ambiguous("models--sha256-model.json")
    assert not state_dir.state_filename_is_ambiguous("models--owner--variant--sha256-q4.json")
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
    marker_path.write_text("[")
    download_manifest.clear_cancel_marker("model", repo_id, variant)
    assert not marker_path.exists()
    assert download_manifest.purge_all_state_for_repo("model", repo_id) == 1
    assert not download_manifest.has_cancel_marker("model", repo_id, variant)
    assert download_manifest.read_manifest("model", repo_id, variant) is None


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
    index = download_manifest.build_variant_state_index(
        [("model", "Owner/Repo", cache_a), ("model", "Owner/Repo", cache_b)],
        active_hub_cache = cache_b,
    )
    indexed_a = index.for_repo("model", "Owner/Repo", hub_cache = cache_a)
    indexed_b = index.for_repo("model", "Owner/Repo", hub_cache = cache_b)
    assert indexed_a.manifest_for("Q4_K_M").expected_files == tuple(expected_a)
    assert indexed_b.manifest_for("Q4_K_M").expected_files == tuple(expected_b)
    assert indexed_a.has_marker("Q4_K_M")
    assert not indexed_b.has_marker("Q4_K_M")

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
    manifest.parent.mkdir(parents = True)
    marker.parent.mkdir(parents = True)
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
    index = download_manifest.build_variant_state_index(
        [("model", "Owner/Repo", cache_a), ("model", "Owner/Repo", cache_b)],
        active_hub_cache = cache_a,
    )
    legacy_a = index.for_repo("model", "Owner/Repo", hub_cache = cache_a)
    legacy_b = index.for_repo("model", "Owner/Repo", hub_cache = cache_b)
    assert legacy_a.manifest_for("Q4_K_M") is not None and legacy_a.has_marker("Q4_K_M")
    assert legacy_b.summary() == (False, 0)

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
    assert download_manifest.purge_all_state_for_repo("model", "Owner/Repo", hub_cache = cache_b) == 0
    assert manifest.is_file() and marker.is_file()
    for invalid_cache in ("\ud800", "\0", "relative-cache", []):
        marker.write_text(json.dumps({"repo_id": "\ud800", "hub_cache": invalid_cache}))
        assert download_manifest.has_cancel_marker("model", "Owner/Repo", "Q4_K_M")
        corrupt_index = download_manifest.build_variant_state_index(
            [("model", "Owner/Repo", cache_a)], active_hub_cache = cache_a
        )
        assert corrupt_index.for_repo("model", "Owner/Repo", hub_cache = cache_a).has_marker("q4_k_m")
    manifest_payload = json.loads(manifest.read_text(encoding = "utf-8"))
    manifest_payload["hub_cache"] = 0
    manifest.write_text(json.dumps(manifest_payload), encoding = "utf-8")
    assert download_manifest.read_manifest("model", "Owner/Repo", "Q4_K_M") is None
    manifest_payload.pop("hub_cache")
    manifest_payload["repo_id"] = "\ud800"
    manifest_payload["variant"] = "Q8_0"
    manifest.write_text(json.dumps(manifest_payload), encoding = "utf-8")
    assert download_manifest.read_manifest("model", "Owner/Repo", "Q4_K_M") is None
    manifest_payload["repo_id"] = "Owner/Repo"
    manifest_payload["variant"] = "Q4_K_M"
    manifest.write_text(json.dumps(manifest_payload), encoding = "utf-8")
    assert list(download_manifest.iter_variant_markers("model", "Owner/Repo")) == [
        ("q4_k_m", marker)
    ]
    download_manifest.clear_cancel_marker("model", "Owner/Repo", "Q4_K_M")
    assert not marker.exists()
    marker.write_text(json.dumps({"repo_id": "Owner/Repo", "variant": "\ud800"}))
    assert list(download_manifest.iter_variant_markers("model", "Owner/Repo"))[0][0] == "q4_k_m"
    unsafe_manifest = json.loads(manifest.read_text(encoding = "utf-8"))
    for unsafe_path in (
        "\ud800",
        "\0",
        "",
        "/outside",
        "../outside",
        "C:\\outside",
        "\\outside",
    ):
        unsafe_manifest["expected_files"][0]["path"] = unsafe_path
        manifest.write_text(json.dumps(unsafe_manifest), encoding = "utf-8")
        assert download_manifest.read_manifest("model", "Owner/Repo", "Q4_K_M") is None
    marker.write_text("[" * 10_000 + "0" + "]" * 10_000)
    assert download_manifest.purge_state("model", "Owner/Repo", "Q4_K_M", hub_cache = cache_a)
    assert not marker.exists()


@pytest.mark.parametrize(
    "variant",
    [
        "unknown/variant with spaces",
        "double--hyphen",
        "Q4_K_M",
    ],
)
def test_index_finds_an_unreadable_marker_stored_under_a_hashed_filename(
    monkeypatch, tmp_path, variant
):
    """The index must agree with has_cancel_marker about a corrupt marker.

    A marker whose payload will not parse stays fail-closed but loses its declared
    variant, so the only identity left is the filename. When the variant is stored
    hashed, that identity is the digest, and looking it up by variant name missed
    it -- so a cancelled variant came back advertised as complete, while the
    per-variant lookup still found the file by rebuilding the same digest.
    """
    hub_cache = tmp_path / "cache-a"
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "studio-cache")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    assert download_manifest.write_cancel_marker(
        "model", "Owner/Repo", variant, "http", hub_cache = hub_cache
    )
    marker = state_dir.marker_path("model", "Owner/Repo", variant, hub_cache = hub_cache)

    def indexed():
        index = download_manifest.build_variant_state_index(
            [("model", "Owner/Repo", hub_cache)], active_hub_cache = hub_cache
        )
        return index.for_repo("model", "Owner/Repo", hub_cache = hub_cache).has_marker(variant)

    assert indexed() and download_manifest.has_cancel_marker(
        "model", "Owner/Repo", variant, hub_cache = hub_cache
    )
    marker.write_text("[")
    assert download_manifest.has_cancel_marker(
        "model", "Owner/Repo", variant, hub_cache = hub_cache
    ), "per-variant lookup should stay fail-closed"
    assert indexed(), "the index disagreed with has_cancel_marker about a corrupt marker"


def test_cached_gguf_scan_degrades_when_the_shared_index_cannot_be_built(monkeypatch, tmp_path):
    """A malformed repo identity must not take every valid row down with it.

    The index is built once per scan and outside the per-repository ``try``, so an
    exception there reaches the endpoint as a 500 and hides the whole inventory.
    ``_scan_cached_models`` already degraded to per-repo reads; the GGUF path did
    not, which is the asymmetry this covers.
    """
    hub_cache = tmp_path / "cache-a"
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "studio-cache")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    # An undecodable byte in a cache directory name reaches us as a lone surrogate, and the repo key
    # cannot be hashed from it. Spelled directly rather than via os.fsdecode(b"...\xff...") because
    # Windows decodes filenames as UTF-8 with surrogatepass, where a bare 0xff raises.
    bad_repo = "Org/Re\udcffpo"
    repos = [
        SimpleNamespace(
            repo_id = repo_id,
            repo_type = "model",
            repo_path = hub_cache / f"models--{repo_id.replace('/', '--')}",
        )
        for repo_id in ("Org/Good", bad_repo)
    ]
    monkeypatch.setattr(
        cache_inventory, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = repos)]
    )
    monkeypatch.setattr(cache_inventory, "_cached_model_snapshot_path", lambda _path: None)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_size_bytes", lambda _repo: 1)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_last_modified", lambda _repo: 0)
    monkeypatch.setattr(cache_inventory, "_is_hidden_infra_repo", lambda *_args: False)
    monkeypatch.setattr(cache_inventory, "_repo_gguf_payload_snapshots", lambda _repo: (None, ()))
    monkeypatch.setattr(cache_inventory, "_cache_inventory_fields", lambda *_a, **_kw: {})

    with pytest.raises(UnicodeEncodeError):
        download_manifest.build_variant_state_index(
            [("model", bad_repo, hub_cache)], active_hub_cache = hub_cache
        )

    rows = cache_inventory._scan_cached_gguf()
    assert "Org/Good" in {
        row["repo_id"] for row in rows
    }, "one unhashable repo identity emptied the whole GGUF inventory"


def test_state_scan_survives_a_state_filename_with_an_undecodable_byte(monkeypatch, tmp_path):
    """One corrupt filename must not take the whole inventory down with it.

    A byte the filesystem encoding cannot decode comes back from iterdir() as a
    lone surrogate, and hashing that for the canonical spelling raises
    UnicodeEncodeError. Per-repo reads used to contain the damage to the repo
    that owned the file. The index is built once per scan and outside the
    per-repo try, so an escaping hash would 500 the endpoint and hide every
    cached model, not just this one.
    """
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    cache = tmp_path / "cache-a"
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache),
    )
    assert download_manifest.write_cancel_marker(
        "model", "Owner/Repo", "Q4_K_M", "http", hub_cache = cache
    )
    marker = state_dir.marker_path("model", "Owner/Repo", "Q4_K_M", hub_cache = cache)
    corrupt = os.path.join(
        os.fsencode(str(marker.parent)), b"models--Owner--Repo--variant--q\xff.json"
    )
    try:
        with open(corrupt, "wb") as handle:
            handle.write(json.dumps({"version": 2, "repo_type": "model"}).encode("utf-8"))
    except (OSError, UnicodeError) as e:
        # Only POSIX filesystems that accept arbitrary bytes can hold this name: macOS rejects it with
        # EILSEQ and Windows has no byte-level filename API at all.
        pytest.skip(f"filesystem will not hold an undecodable filename: {e}")

    index = download_manifest.build_variant_state_index(
        [("model", "Owner/Repo", cache)], active_hub_cache = cache
    )
    state = index.for_repo("model", "Owner/Repo", hub_cache = cache)
    assert state.has_marker("q4_k_m"), "the readable marker was lost to its corrupt neighbour"
    assert state.summary()[0] is True

    # The per-repo iterators keep enumerating past it too, surrogate name and all.
    variants = [
        variant
        for variant, _path in download_manifest.iter_variant_markers(
            "model", "Owner/Repo", hub_cache = cache
        )
    ]
    assert "Q4_K_M" in variants and any("\udcff" in v for v in variants), variants


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, *args, **kwargs):
        self.warnings.append((args, kwargs))


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
    assert folder_browser._is_path_inside_allowlist(model_dir.resolve(), allowlist)


def test_get_models_folder_response_creates_and_returns_dir(monkeypatch, tmp_path):
    # The endpoint creates the cache dir on demand so "Open folder" works before the first download.
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
        lambda _repo_id, _path, **_kw: False,
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
        lambda _repo_id, _path, **_kw: True,
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
        lambda _repo_id, _path, **_kw: True,
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
        lambda _repo_id, _path, **_kw: False,
    )

    result = {"cached": cache_inventory._scan_cached_gguf()}

    assert [row["repo_id"] for row in result["cached"]] == ["Org/Chat-GGUF"]


def test_cached_gguf_scan_emits_curated_asr_as_non_chat_audio_inventory(monkeypatch, tmp_path):
    asr = _repo(
        "unslothai/Qwen3-ASR-0.6B-GGUF",
        [
            _file("Qwen3-ASR-0.6B-Q8_0.gguf", 800_000_000),
            _file("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", 200_000_000),
        ],
        tmp_path / "asr",
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [asr])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda _repo_id, _path, **_kw: False,
    )

    [row] = cache_inventory._scan_cached_gguf()

    assert row["repo_id"] == "unslothai/Qwen3-ASR-0.6B-GGUF"
    assert row["size_bytes"] == 800_000_000
    assert row["capabilities"]["can_chat"] is False
    assert row["capabilities"]["supports_vision"] is False


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
        lambda _repo_id, _path, **_kw: False,
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
        lambda _kind, _repo_id, _path, **_kw: False,
    )
    result = {"cached": cache_inventory._scan_cached_models()}

    assert [row["repo_id"] for row in result["cached"]] == ["Org/Chat"]


def test_cached_models_scan_emits_curated_and_custom_whisper_as_stt(monkeypatch, tmp_path):
    curated_path = tmp_path / "hub" / "models--unsloth--whisper-tiny"
    curated_path.mkdir(parents = True)
    curated = _repo(
        "unsloth/whisper-tiny",
        [_file("config.json", 12), _file("model.safetensors", 80_000_000)],
        curated_path,
    )
    custom_path = tmp_path / "hub" / "models--Org--custom-whisper"
    custom_path.mkdir(parents = True)
    custom = _repo(
        "Org/custom-whisper",
        [_file("config.json", 12), _file("model.safetensors", 90_000_000)],
        custom_path,
    )
    monkeypatch.setattr(
        cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [curated, custom])],
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path, **_kw: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "_cached_model_local_metadata",
        lambda repo_path, _snapshot = None: {"_hidden_stt": "custom-whisper" in str(repo_path)},
    )

    rows = cache_inventory._scan_cached_models()

    rows_by_repo = {row["repo_id"]: row for row in rows}
    assert set(rows_by_repo) == {"unsloth/whisper-tiny", "Org/custom-whisper"}
    assert rows_by_repo["Org/custom-whisper"]["task"] == "automatic-speech-recognition"
    assert all(row["capabilities"]["can_chat"] is False for row in rows_by_repo.values())


_SNAPSHOT_SHA = "a" * 40


def _diffusion_scan(
    monkeypatch,
    tmp_path,
    repo_id: str,
    files: list,
    *,
    task: str | None,
    modular_manifest: dict | None = None,
    config_manifest: dict | None = None,
    expect_task_classification: bool = True,
):
    """One cached diffusion repo through _scan_cached_models, with the download-partial signal
    forced off so only the pipeline-shape checks can flag the row.

    The snapshot is materialised on disk, not just described: the pipeline-shape checks read the
    directory the row resolves to, so a revision without a real ``snapshot_path`` would report no
    manifest and no denoiser for every repo alike and the assertions below would all pass on
    nothing.
    """
    repo_path = tmp_path / f"hub/models--{repo_id.replace('/', '--')}"
    snapshot = repo_path / "snapshots" / _SNAPSHOT_SHA
    snapshot.mkdir(parents = True)
    for f in files:
        target = snapshot / f.file_name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"\0" * min(int(f.size_on_disk), 4096))
    if modular_manifest is not None:
        (snapshot / "modular_model_index.json").write_text(
            json.dumps(modular_manifest), encoding = "utf-8"
        )
    if config_manifest is not None:
        (snapshot / "config.json").write_text(json.dumps(config_manifest), encoding = "utf-8")
    refs = repo_path / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    (refs / "main").write_text(_SNAPSHOT_SHA)
    revision = SimpleNamespace(files = files, snapshot_path = snapshot, commit_hash = _SNAPSHOT_SHA)
    repo = SimpleNamespace(
        repo_id = repo_id, repo_type = "model", repo_path = repo_path, revisions = [revision]
    )
    monkeypatch.setattr(
        cache_inventory, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    # The real signature takes snapshot_dir; a double that omits it raises TypeError, which the per-repo
    # except swallows into an empty row list, silently skipping every assertion below.
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    selected_snapshots = []

    def row_task(
        _repo,
        *,
        gguf,
        selected = None,
    ):
        selected_snapshots.append(selected)
        return task

    monkeypatch.setattr(cache_inventory, "_cached_row_task", row_task)
    rows = cache_inventory._scan_cached_models()
    assert len(rows) == 1
    assert selected_snapshots == ([snapshot] if expect_task_classification else [])
    return rows[0]


def test_cached_models_scan_marks_a_companion_only_pipeline_partial(monkeypatch, tmp_path):
    """A GGUF image load prefetches the base repo's manifest + VAE + text encoder and skips the
    multi-GB transformer. Every file its manifest expected arrived, so the download-partial check
    passes it, but from_pretrained cannot load it -- the picker must not advertise it as on-device
    (same rule /api/models/cached applies)."""
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "Org/Pipeline-Companions-Only",
        [
            _file("model_index.json", 900),
            _file("vae/diffusion_pytorch_model.safetensors", 300_000_000),
            _file("text_encoder/model.safetensors", 900_000_000),
        ],
        task = "text-to-image",
    )

    assert row["partial"] is True
    # A companion-only snapshot arrived intact, so it has no Resume / Redownload story.
    assert row["partial_transport"] is None


def test_cached_models_scan_keeps_a_complete_pipeline_loadable(monkeypatch, tmp_path):
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "Org/Pipeline-Complete",
        [
            _file("model_index.json", 900),
            _file("vae/diffusion_pytorch_model.safetensors", 300_000_000),
            _file("text_encoder/model.safetensors", 900_000_000),
            # Unsharded, because this fixture stands for a COMPLETE pipeline: it used to name a lone
            # "-00001-of-00002" shard with no index, and with no index diffusers reads the component as
            # unsharded and asks for the plain name, so a lone shard is not loadable.
            _file("transformer/diffusion_pytorch_model.safetensors", 4_000_000_000),
        ],
        task = "text-to-image",
    )

    assert row["partial"] is False
    assert row["single_file"] is False


def test_cached_models_scan_exposes_minimax_music3_modular_pipeline(monkeypatch, tmp_path):
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "MiniMaxAI/MiniMax-Music3",
        [
            _file("modular_model_index.json", 900),
            _file("transformer/diffusion_pytorch_model.safetensors", 4_000_000_000),
        ],
        task = None,
        modular_manifest = {
            "_class_name": "MiniMaxMusic3ModularPipeline",
            "_blocks_class_name": "MiniMaxMusic3Blocks",
        },
        expect_task_classification = False,
    )

    assert row["task"] == "text-to-speech"
    assert row["audio_type"] == "minimax_music3"
    assert row["capabilities"]["can_chat"] is False
    assert row["partial"] is False
    assert row["single_file"] is False


def test_cached_models_scan_flags_a_single_file_diffusion_checkpoint(monkeypatch, tmp_path):
    """No root model_index.json: loadable only through from_single_file + a filename. The picker
    gates on this flag, and before it was carried here every hub-sourced row read as a full
    pipeline -- so a checkpoint-only repo was offered as a pipeline load and failed after the
    handoff."""
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "Org/Single-File-Checkpoint",
        [_file("config.json", 12), _file("z-image-turbo-fp8.safetensors", 6_000_000_000)],
        task = "text-to-image",
    )

    assert row["single_file"] is True
    assert row["partial"] is False


def test_a_companion_mirror_carries_the_flag_on_the_hub_row(monkeypatch, tmp_path):
    """The chat picker is backed by /api/hub/cached-models, NOT the legacy /api/models one, so a
    flag set only on the legacy route arrives as undefined here -- the same trap single_file fell
    into. A mirror that reaches this scan must carry it.

    Given a config.json, because the real mirrors have none: see the test below for what that
    means today.
    """
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "unsloth/Z-Image-Turbo-ComfyUI",
        [_file("config.json", 12), _file("model.safetensors", 300_000_000)],
        task = None,
    )

    assert row["companion"] is True
    # Startup auto-load filters on capabilities.can_chat, never on the flag, so a row carrying only the
    # flag was still auto-loadable as a chat model.
    assert row["capabilities"]["can_chat"] is False


def test_an_ordinary_repo_is_not_flagged_as_a_companion(monkeypatch, tmp_path):
    """The flag names an exact set of mirrors, so a repo of the same shape is untouched."""
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "unsloth/Qwen3-8B",
        [_file("config.json", 12), _file("model.safetensors", 100)],
        task = None,
    )

    assert row["companion"] is False
    # ...and an ordinary chat repo keeps its chat capability.
    assert row["capabilities"]["can_chat"] is True


@pytest.mark.parametrize(
    ("repo_id", "config"),
    [
        ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", {}),
        ("Acme/custom-moss-codec", {"model_type": "moss-audio-tokenizer"}),
        ("Acme/legacy-moss-codec", {"model_type": "speech_tokenizer"}),
        ("Acme/custom-higgs-codec", {"architectures": ["HiggsAudioV2TokenizerModel"]}),
    ],
)
def test_native_audio_codec_repos_are_companion_infrastructure(
    monkeypatch, tmp_path, repo_id, config
):
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        repo_id,
        [_file("config.json", 100), _file("model.safetensors", 100)],
        task = None,
        config_manifest = config,
    )

    assert row["companion"] is True
    assert row["capabilities"]["can_chat"] is False


def test_the_real_companion_shape_never_reaches_a_row_at_all(monkeypatch, tmp_path):
    """Why the flag above is a latch, not a live fix.

    The published mirrors are ComfyUI-style: weights under split_files/ and no config.json or
    model_index.json. _repo_non_gguf_model_payload classifies that as ``unknown``, so
    has_runnable_weights is False and _scan_cached_models drops the repo before any row exists.
    Pinned because the flag's whole value is covering the day that classifier learns to admit
    these -- if this test starts failing, the flag is what stops a denoiser-less repo becoming a
    chat pick.
    """
    from types import SimpleNamespace

    snapshot = tmp_path / "snapshots" / _SNAPSHOT_SHA
    snapshot.mkdir(parents = True)
    files = [_file("split_files/vae/ae.safetensors", 300_000_000), _file("README.md", 100)]
    for f in files:
        target = snapshot / f.file_name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"\0" * 16)
    repo = SimpleNamespace(
        repo_id = "unsloth/Z-Image-Turbo-ComfyUI",
        repo_type = "model",
        repo_path = tmp_path,
        revisions = [SimpleNamespace(files = files, snapshot_path = snapshot, commit_hash = _SNAPSHOT_SHA)],
    )
    payload = cache_inventory._repo_non_gguf_model_payload(repo)

    assert payload.model_format == "unknown"
    assert payload.has_runnable_weights is False


def test_cached_models_scan_leaves_chat_repos_unflagged(monkeypatch, tmp_path):
    """The flag is a diffusion-picker concern: a chat repo (no task) never carries it, so a plain
    safetensors model is not mistaken for a single-file checkpoint."""
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "Org/Chat-Model",
        [_file("config.json", 12), _file("model.safetensors", 100)],
        task = None,
    )

    assert row["single_file"] is False


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
        lambda _repo_id, _path, **_kw: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path, **_kw: False,
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
        lambda _repo_id, _path, **_kw: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path, **_kw: False,
    )

    assert cache_inventory._scan_cached_gguf() == []
    assert cache_inventory._scan_cached_models() == []


def test_cached_models_scan_keeps_unrelated_repo_with_custom_generic_embedder(
    monkeypatch, tmp_path
):
    # EXACT repo-id match only: substring basename matching used to drop real chat models from the inventory.
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
        lambda _kind, _repo_id, _path, **_kw: False,
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
        lambda _repo_id, _path, **_kw: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _kind, _repo_id, _path, **_kw: False,
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


def test_download_dataset_recovers_commit_completion_after_transient_metadata_failure(
    monkeypatch, tmp_path
):
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "datasets--Org--Data" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "data.parquet").write_bytes(b"rows")
    metadata_calls = []

    def _metadata(*_args, **_kwargs):
        metadata_calls.append(True)
        if len(metadata_calls) == 1:
            raise RuntimeError("metadata down")
        return SimpleNamespace(
            sha = snapshot.name,
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 4)],
        )

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    monkeypatch.setattr(hf_download, "_dataset_info_with_retry", _metadata)
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(snapshot)),
    )

    hf_download._download_dataset("Org/Data", None, "http")

    manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        hub_cache = hub_cache,
    )
    completion = download_manifest.read_dataset_completion(
        "Org/Data",
        snapshot.name,
        hub_cache = hub_cache,
    )
    assert len(metadata_calls) == 2
    assert manifest is not None
    assert manifest.commit_hash == snapshot.name
    assert manifest.metadata_derived is True
    assert completion is not None
    assert completion.expected_files[0].path == "data.parquet"


def test_download_dataset_promotes_existing_disk_manifest_after_metadata_recovers(
    monkeypatch, tmp_path
):
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "datasets--Org--Data" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "data.parquet").write_bytes(b"rows")
    metadata_calls = []

    def _metadata(*_args, **_kwargs):
        metadata_calls.append(True)
        if len(metadata_calls) == 1:
            raise RuntimeError("metadata down")
        return SimpleNamespace(
            sha = snapshot.name,
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 4)],
        )

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    assert download_manifest.write_manifest(
        "dataset",
        "Org/Data",
        None,
        [download_manifest.ExpectedFile("data.parquet", 4)],
        "http",
        hub_cache = hub_cache,
    )
    monkeypatch.setattr(hf_download, "_dataset_info_with_retry", _metadata)
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(snapshot)),
    )

    hf_download._download_dataset("Org/Data", None, "http")

    manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        hub_cache = hub_cache,
    )
    completion = download_manifest.read_dataset_completion(
        "Org/Data",
        snapshot.name,
        hub_cache = hub_cache,
    )
    assert len(metadata_calls) == 2
    assert manifest is not None
    assert manifest.metadata_derived is True
    assert manifest.commit_hash == snapshot.name
    assert completion is not None


def test_download_dataset_recovery_commit_mismatch_is_not_attested(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "datasets--Org--Data" / "snapshots" / "downloaded-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "data.parquet").write_bytes(b"rows")
    metadata_calls = []

    def _metadata(*_args, **_kwargs):
        metadata_calls.append(True)
        if len(metadata_calls) == 1:
            raise RuntimeError("metadata down")
        return SimpleNamespace(
            sha = "different-commit",
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 4)],
        )

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    monkeypatch.setattr(hf_download, "_dataset_info_with_retry", _metadata)
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(snapshot)),
    )

    hf_download._download_dataset("Org/Data", None, "http")

    manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        hub_cache = hub_cache,
    )
    assert len(metadata_calls) == 2
    assert manifest is not None
    assert manifest.commit_hash is None
    assert manifest.metadata_derived is False
    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            snapshot.name,
            hub_cache = hub_cache,
        )
        is None
    )
    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            "different-commit",
            hub_cache = hub_cache,
        )
        is None
    )


def test_download_dataset_disk_fallback_is_not_attested(monkeypatch, tmp_path):
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "datasets--Org--Data" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "data.parquet").write_bytes(b"rows")

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = hub_cache),
    )
    monkeypatch.setattr(
        hf_download,
        "_dataset_info_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metadata down")),
    )
    monkeypatch.setattr(
        download_registry,
        "prepare_cache_for_transport",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        hf_cache_state,
        "has_active_incomplete_blobs",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(snapshot)),
    )

    hf_download._download_dataset("Org/Data", None, "http")

    manifest = download_manifest.read_manifest(
        "dataset",
        "Org/Data",
        hub_cache = hub_cache,
    )
    assert manifest is not None
    assert manifest.commit_hash is None
    assert manifest.metadata_derived is False
    assert (
        download_manifest.read_dataset_completion(
            "Org/Data",
            snapshot.name,
            hub_cache = hub_cache,
        )
        is None
    )


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
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [root])

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

    # cache_path is ABSENT, not null: null means no cache dir exists, which hydration acts on by
    # retiring the persisted job, and cache_measured carries the same distinction.
    assert result == {
        "downloaded_bytes": 0,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": 0,
        "progress": 0,
        "cache_measured": False,
    }
    assert "cache_path" not in result
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
    # The mmproj companion that seeded the baseline is gone, so completed_bytes is main-only and below it.
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
    # A variant already complete on disk carries a baseline equal to its full size; subtracting it would
    # report 0/0, which the frontend evicts as gone.
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
    # The "instant ~900 MB" bug: with this variant's hashes resolved, progress counts ONLY its in-
    # progress blob, never a sibling quant's finalized bytes in the shared blobs/ dir.
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "siblinghash").write_bytes(b"z" * 900)
    (blobs / "mainhash.incomplete").write_bytes(b"x" * 5)

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
    # With hashes unresolved, the shared blobs/ dir's FINALIZED blobs must not be counted wholesale: a
    # cached sibling quant alongside is the "instant ~900 MB" bug.
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
    # With hashes unresolved an .incomplete cannot be attributed to this variant, so it is dropped; in
    # production the worker writes the manifest before any .incomplete exists.
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "activehash.incomplete").write_bytes(b"x" * 50)
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
            expected_bytes = 1000,
        )
    )

    assert result["downloaded_bytes"] == 0
    assert result["completed_bytes"] == 0


def test_gguf_progress_unknown_hashes_no_backward_dip_when_variant_finalizes(monkeypatch, tmp_path):
    # Regression for the two-variant dip: the sibling's .incomplete bytes used to leak into this
    # numerator, dipping the bar to ~78% for one poll.
    entry = tmp_path / "models--unsloth--SmolLM2-360M-Instruct-GGUF"
    blobs = entry / "blobs"
    snap = entry / "snapshots" / "rev0"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)
    own_total = 218_673_760
    sibling_total = 234_686_560

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

    assert result["downloaded_bytes"] == 0
    assert result["progress"] == 0


def _unresolvable_variant_metadata(
    monkeypatch,
    entry,
    *,
    state = "running",
):
    """A repo whose model_info is failing: no requirement, no blob hashes.

    Reproduces the negatively-cached lookup -- a 401 on a gated repo whose token
    was removed, or an offline poll -- that leaves the expected file set empty
    for the whole TTL after a single failure.
    """

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
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = state)),
    )


def test_gguf_progress_unknown_hashes_reports_the_variant_files_on_disk(monkeypatch, tmp_path):
    """A finished variant must not read as "0 B of 33 GB" when metadata flakes.

    model_info failing is negatively cached, so a 401 that lands after the last
    byte keeps the expected hash set empty for the whole TTL. Every blob was
    then filtered out of the count and the reading collapsed to zero against the
    caller's catalog hint, which is the stale "downloaded 0 B" card. The
    variant's own snapshot files are still attributable by name, so they are
    what the reading falls back to -- the sibling quant beside them is not.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 30)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (blobs / "siblinghash").write_bytes(b"z" * 900)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 130
    assert result["downloaded_bytes"] == 130
    assert result["progress"] > 0


def test_gguf_progress_unknown_hashes_keeps_a_total_under_a_full_baseline(monkeypatch, tmp_path):
    """A variant already on disk at claim time must not net out to "0 B of 0 B".

    Its completed_baseline_bytes covers the whole variant, and the subtraction
    was previously held off by complete_on_disk -- which is exactly what an
    unresolvable file set takes away. Without the guard the fallback reading
    cancels against the baseline and the response carries no total at all, so
    the bar has nothing to draw and the frontend reads the job as evictable.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 130)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = "running"),
            get_job_metadata = lambda _key: SimpleNamespace(completed_baseline_bytes = 130),
        ),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["expected_bytes"] == 130
    assert result["downloaded_bytes"] == 130


def test_gguf_progress_unknown_hashes_prefers_the_manifest_file_set(monkeypatch, tmp_path):
    """With a manifest but no hashes in it, its declared paths scope the reading.

    The metadata-fallback manifest the worker writes from the finished snapshot
    carries paths and sizes but no sha256, so the hash filter still resolves to
    nothing. Its file list is exact, so it is used ahead of matching by name.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100)],
        "http",
        hub_cache = entry.parent,
    )
    _unresolvable_variant_metadata(monkeypatch, entry)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 0,
        )
    )

    assert result["completed_bytes"] == 100
    assert result["complete_on_disk"] is True
    assert result["progress"] == 1.0


def test_gguf_progress_unknown_hashes_stays_zero_without_variant_files(monkeypatch, tmp_path):
    """The fallback reads the variant's files, not the shared blobs/ dir.

    Guards the "instant ~900 MB" regression from the other direction: a cached
    sibling quant with no snapshot file of this variant's own still reads zero.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    (blobs / "siblinghash").write_bytes(b"z" * 900)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 900,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 0


def test_gguf_progress_settles_complete_from_disk_without_a_manifest(monkeypatch, tmp_path):
    """A materialized snapshot whose manifest is gone must not stay partial forever.

    Metadata named every blob the revision expects and all of them are on disk
    finalized at their declared sizes, which is the evidence a manifest verify
    collects. Refusing it because no manifest file happens to exist -- it was
    never written, was deleted, or was filed under a cache scope this reader can
    no longer name -- left the job in an active state with Retry/Resume showing
    on a download that had finished.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert (
        download_manifest.read_manifest("model", "Org/Model-GGUF", "Q4_K_M", hub_cache = entry.parent)
        is None
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: SimpleNamespace(
            download_size_bytes = 100,
            required_hashes = frozenset({"mainhash"}),
            expected_files = (
                download_manifest.ExpectedFile(
                    path = "model-Q4_K_M.gguf",
                    size = 100,
                    sha256 = "mainhash",
                ),
            ),
        ),
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
            expected_bytes = 100,
        )
    )

    assert result["complete_on_disk"] is True
    assert result["progress"] == 1.0


def test_a_refused_manifest_is_not_reread_through_the_blob_hash_fallback(monkeypatch, tmp_path):
    """Refusing a manifest has to stick.

    Two caches disagreeing about the variant means no manifest may be applied across the
    scan. The generic blob-hash helper reads the DEFAULT cache's manifest with none of that
    scoping, so falling through to it reinstated the rejected hashes -- and they then filter
    out every blob of the cache that actually holds the finished variant.
    """
    active = tmp_path / "active" / "models--Org--Model-GGUF"
    remembered = tmp_path / "remembered" / "models--Org--Model-GGUF"
    active.mkdir(parents = True)
    remembered.mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        download_manifest, "_canonical_hub_cache", lambda root = None: str(root or "")
    )
    for root, digest in ((active.parent, "new"), (remembered.parent, "old")):
        assert download_manifest.write_manifest(
            "model",
            "Org/Model-GGUF",
            "Q4_K_M",
            [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100, sha256 = digest)],
            "http",
            hub_cache = root,
        )
    monkeypatch.setattr(
        downloads,
        "preferred_repo_cache_dirs",
        lambda *_a, **_kw: [active, remembered],
    )

    called: list[str] = []

    def _blob_hashes(*_args, **_kwargs):
        called.append("fallback")
        return frozenset({"new"})

    monkeypatch.setattr(downloads.gguf_variants, "gguf_variant_blob_hashes", _blob_hashes)
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
    )
    captured: dict = {}

    async def _progress(*_args, **kwargs):
        captured["resolver"] = kwargs.get("metadata_resolver")
        return {
            "downloaded_bytes": 0,
            "completed_bytes": 0,
            "complete_on_disk": False,
            "expected_bytes": 0,
            "progress": 0,
            "cache_path": None,
        }

    monkeypatch.setattr(downloads.snapshot_progress, "snapshot_progress_response", _progress)

    asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )
    resolver = captured["resolver"]
    assert resolver is not None
    assert resolver("Org/Model-GGUF", None) == (100, frozenset())
    assert called == []


def test_gguf_progress_without_a_manifest_needs_every_expected_blob(monkeypatch, tmp_path):
    """The no-manifest completion is evidence-gated, not size-gated.

    One expected blob missing while an oversized sibling makes the byte total
    look satisfied must stay partial: the caller's expected_bytes is a catalog
    hint and can never be what completion is judged against.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (snap / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"x" * 400)
    (blobs / "shard1").write_bytes(b"x" * 400)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: SimpleNamespace(
            download_size_bytes = 300,
            required_hashes = frozenset({"shard1", "shard2"}),
            expected_files = (
                download_manifest.ExpectedFile(
                    path = "model-Q4_K_M-00001-of-00002.gguf",
                    size = 150,
                    sha256 = "shard1",
                ),
                download_manifest.ExpectedFile(
                    path = "model-Q4_K_M-00002-of-00002.gguf",
                    size = 150,
                    sha256 = "shard2",
                ),
            ),
        ),
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
            expected_bytes = 300,
        )
    )

    assert result["complete_on_disk"] is False
    assert result["progress"] == 0.99


def test_gguf_progress_recovers_the_windows_shaped_stale_download_card(monkeypatch, tmp_path):
    """The reported symptom end to end: finished download, "0 B of 33 GB", Retry showing.

    Every Windows-shaped condition at once, none of which reproduce on Linux on
    their own. The cache is reached through a redirect so its resolved and
    unresolved spellings differ; the snapshot holds copies rather than symlinks,
    as HF falls back to when the filesystem refuses them; the manifest sits
    under the pre-resolve scope digest an earlier build wrote it to; and
    model_info is failing, so the expected blob hashes come back empty and stay
    that way for the negative-cache TTL. Each one alone was enough to zero the
    reading and keep the job out of a terminal state.
    """
    resolved_root = tmp_path / "resolved"
    resolved_root.mkdir()
    link = tmp_path / "redirected"
    try:
        link.symlink_to(resolved_root, target_is_directory = True)
    except (NotImplementedError, OSError):  # pragma: no cover - unprivileged Windows
        pytest.skip("symlinks unavailable on this host")
    hub_cache = link / "hub"
    entry = hub_cache / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    # Copy layout: the snapshot holds real files, and blobs/ was never populated with anything this reading
    # could have matched by hash.
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 33_000)
    assert not (snap / "model-Q4_K_M.gguf").is_symlink()

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(hub_cache)),
    )
    # A manifest as an older build filed it: hashed from the unresolved spelling, and with no sha256
    # because HF metadata was already unreachable when the worker recorded it.
    legacy = state_dir.manifest_path(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        hub_cache = hub_cache,
        cache_scope = state_dir.legacy_cache_scope_name(hub_cache),
    )
    assert legacy.parent.name != state_dir.cache_scope_name(hub_cache)
    legacy.parent.mkdir(parents = True, exist_ok = True)
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_type": "model",
                "repo_id": "Org/Model-GGUF",
                "variant": "Q4_K_M",
                "started_at": "2026-01-01T00:00:00+00:00",
                "expected_files": [{"path": "model-Q4_K_M.gguf", "size": 33_000}],
                "transport": "http",
                "hub_cache": str(hub_cache),
            }
        ),
        encoding = "utf-8",
    )
    _unresolvable_variant_metadata(monkeypatch, entry, state = "idle")

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 33_000,
        )
    )

    assert result["downloaded_bytes"] == 33_000
    assert result["completed_bytes"] == 33_000
    assert result["expected_bytes"] == 33_000
    assert result["complete_on_disk"] is True
    assert result["progress"] == 1.0


def test_gguf_progress_without_a_manifest_needs_the_snapshot_materialized(monkeypatch, tmp_path):
    """A finalized blob no one linked to is not a finished download.

    HF writes the blob and then links it into the snapshot dir, so a run killed
    between the two leaves bytes that nothing points at. With a manifest,
    verify_against_disk catches it; without one, blob-level evidence alone would
    have called an unloadable snapshot complete.

    The stray companion is the reason completion is judged against the metadata
    file list rather than a byte total taken over the snapshot dir. Every mmproj
    and drafter in a repo looks like it belongs to whichever variant is being
    polled -- a plan fetches one of each -- so a leftover one, or an opt-in
    ``dspark/`` drafter, would cover for the shard that never landed.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (snap / "mmproj-F32.gguf").write_bytes(b"y" * 5_000)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *_args, **_kwargs: SimpleNamespace(
            download_size_bytes = 100,
            required_hashes = frozenset({"mainhash"}),
            expected_files = (
                download_manifest.ExpectedFile(
                    path = "model-Q4_K_M.gguf",
                    size = 100,
                    sha256 = "mainhash",
                ),
            ),
        ),
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
            expected_bytes = 100,
        )
    )

    assert result["complete_on_disk"] is False


def test_running_job_never_reads_progress_from_a_remembered_cache(monkeypatch, tmp_path):
    """force_active means the active root and nothing else, even before it exists.

    The first download into a freshly configured cache creates the root, so
    hf_cache_root declines it and the lookup used to fall through to every
    remembered cache. A previous cache's completed copy then read as this run's
    progress, finalizing a job that had not written a byte.
    """
    previous = tmp_path / "previous"
    complete = previous / "models--Org--Model" / "blobs"
    complete.mkdir(parents = True)
    (complete / "mainhash").write_bytes(b"x" * 100)
    active = tmp_path / "active"
    monkeypatch.setattr(
        hf_cache_state,
        "hf_cache_roots",
        lambda scan_errors = None: [previous],
    )

    dirs = hf_cache_state.preferred_repo_cache_dirs(
        "model",
        "Org/Model",
        force_active = True,
        active_root = active,
    )

    assert dirs == [active / "models--Org--Model"]
    assert not dirs[0].exists()
    # Without force_active the remembered copy is still the best reading there is.
    assert hf_cache_state.preferred_repo_cache_dirs("model", "Org/Model") == [
        previous / "models--Org--Model"
    ]


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
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "config.json", size = 12)],
    )
    assert download_manifest.write_cancel_marker("model", "Owner/Repo", "q4_k_m", "http")

    manifest = download_manifest.read_manifest("model", "owner/repo", "q4_k_m")

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
    ] == ["q4_k_m"]
    assert download_manifest.purge_all_state_for_repo("model", "owner/repo") == 1
    assert download_manifest.read_manifest("model", "owner/repo", "Q4_K_M") is None


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


def test_qwen3_asr_gguf_name_hint_is_not_classified_as_chat(monkeypatch, tmp_path):
    from hub.services.models import catalog_classification

    asr = tmp_path / "Qwen3-ASR-0.6B-Q8_0.gguf"
    chat = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    asr.write_bytes(b"gguf")
    chat.write_bytes(b"gguf")
    monkeypatch.setattr(catalog_classification, "_gguf_architecture", lambda _path: "qwen3")

    assert catalog_classification._gguf_path_task(asr) == "automatic-speech-recognition"
    assert catalog_classification._gguf_path_task(chat) == "text-generation"


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


def test_completed_gguf_variants_ignores_big_endian_by_the_loader_label(tmp_path):
    # The scan reads Q4_K_M while the loader reads F16 and refuses the file as big-endian, so by the
    # scan's label it would vouch for Q4_K_M and _complete_with_servable would skip the torn split.
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "0-F16-be-checkpoint-Q4_K_M.gguf").write_bytes(b"GGUF")
    (snapshot / "z-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF")

    assert "Q4_K_M" not in inventory_scan._completed_gguf_variants(snapshot)

    (snapshot / "z-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    assert "Q4_K_M" in inventory_scan._completed_gguf_variants(snapshot)


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
    # The same variant under a different-cased repo id resolves to the same job, so the second claim
    # attaches to the running one instead of starting a duplicate.
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
    # Two variants sharing one mmproj blob still download together on one transport:
    # huggingface_hub's per-blob lock serializes the shared write and prepare_cache_for_transport
    # never purges a blob a peer is writing.
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
    # Deleting one quant's partial must be allowed while a different quant of the same repo is
    # downloading, and must protect every blob the live sibling is writing.
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

    # While Q4_K_M is being deleted, re-downloading it is blocked but an untouched third variant may still
    # start.
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
    # The manifest keeps original casing while the marker is lowercased; offline reconstruction must
    # collapse them to ONE entry, in the manifest's casing.
    # Per-variant blob hashes (distinct main shard, shared mmproj companion).
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


def test_partial_gguf_reconstruction_drops_a_variant_read_off_the_filename(monkeypatch):
    # An unreadable payload leaves only the filename, whose digest fragment names nothing; a variant
    # genuinely called sha256-<32 hex> reads the same but is stored under the hash of itself.
    digest = "sha256-" + "0" * 32
    entries = [
        (f"@{digest}", Path(f"repo--variant--@{digest}.json")),
        (digest, Path(f"repo--variant--{digest}.json")),
        (digest, Path("repo--variant--@sha256-" + "c" * 32 + ".json")),
        ("Q4_K_M", Path("repo--variant--q4_k_m.json")),
    ]
    monkeypatch.setattr(
        download_manifest, "iter_variant_manifests", lambda *_a, **_k: iter(entries)
    )
    monkeypatch.setattr(download_manifest, "iter_variant_markers", lambda *_a, **_k: iter(()))
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *_a, **_k: None)

    result = gguf.list_partial_gguf_variants_from_state("Org/Repo")

    assert result is not None
    variants, _has_vision = result
    assert sorted(variant.quant for variant in variants) == ["Q4_K_M", digest]


def test_download_registry_serializes_cross_transport_variant_downloads():
    # An HTTP append-resume and an XET rewrite of the same shared blob would corrupt each other, so
    # different-transport variants are serialized.
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
    # Resolved blob hashes are NOT required to run two same-transport variants concurrently: safety
    # comes from each worker purging only its own blobs plus huggingface_hub's per-etag lock, and
    # requiring them rejected the second variant whenever a metadata fetch flaked.
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
    # finalize_worker_exit relies solely on the worker's exit code and never kills a live process:
    # huggingface_hub already bounds reads with timeouts, so a liveness kill could only false-cancel.
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
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda create = False, **kw: root)

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
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda create = False, **kw: root)
    return blobs


def test_prepare_cache_for_transport_purges_cross_transport_companion(monkeypatch, tmp_path):
    blobs = _vision_cache_root(monkeypatch, tmp_path)
    companion = frozenset({"shared-mmproj"})

    # An interrupted XET download leaves a sparse partial, so a later HTTP download of a different
    # variant must purge it, else the HTTP resumer appends to the sparse bytes and corrupts the blob.
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
    """Only a hub that can still append to the partial earns the same-transport reprieve."""
    # The purge asks partial_is_resumable, so patching the hub-version helper it wraps would be a no-op here.
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
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
    partial = blobs / "shared-mmproj.incomplete"
    partial.write_bytes(b"resumable")
    # Aged past the abandonment grace, so the reprieve is what preserves it, not its freshness.
    old = time.time() - download_registry.ABANDONED_PARTIAL_SECONDS - 60
    os.utime(partial, (old, old))

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Vision",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({"q4-main"}),
        companion_blob_hashes = companion,
    )

    assert purged == 0
    assert partial.exists()


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

    asyncio.run(downloads.download_model_response(_download_body(gguf_variant = "Q4_K_M")))

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

    asyncio.run(downloads.download_model_response(_download_body(gguf_variant = "Q4_K_M")))

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

    result = asyncio.run(downloads.download_model_response(_download_body()))

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

    result = asyncio.run(downloads.download_model_response(_download_body()))

    assert result["accepted"] is True
    assert invalidated == [True]


def test_two_concurrent_same_repo_variants_both_complete(monkeypatch, tmp_path):
    # End-to-end proof that two GGUF variants of one repo download concurrently without cancelling each
    # other, with real registry, finalize, subprocess and watch threads under true concurrency.
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
            downloads.download_model_response(_download_body(gguf_variant = "Q4_K_M")),
            downloads.download_model_response(_download_body(gguf_variant = "Q8_0")),
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
    # The repo under test lives in this cache; make it the active one so the delete scopes to it (default target
    # root is the active hub cache).
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
        lambda _repo_type, _repo_id, force_active = False, **kw: [entry],
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
        lambda _repo_type, _repo_id, force_active = False, **kw: [entry],
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
        lambda _repo_type, _repo_id, force_active = False, **kw: [entry],
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


def test_a_shared_companion_alone_is_not_evidence_the_quant_is_here(tmp_path):
    """mmproj and the MTP drafter are downloaded with every quant in a repo, so on their own
    they say nothing about THIS one. Deleting a quant while a sibling kept its companion left
    a positive byte reading for the deleted variant, and hydration reads any positive reading
    as active: it re-adopts the stale job and blocks a fresh download of the same quant."""
    snap = tmp_path / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (snap / "mmproj-F16.gguf").write_bytes(b"m" * 64)

    def matcher(path, *, companions = True):
        if path.endswith("Q4_K_M.gguf"):
            return True
        return companions and path.startswith("mmproj")

    assert snapshot_progress._materialized_bytes(snap, matcher) == 0

    # With the quant's own shard present, the companion counts again.
    (snap / "model-Q4_K_M.gguf").write_bytes(b"q" * 32)
    assert snapshot_progress._materialized_bytes(snap, matcher) == 96

    # A matcher that does not take the keyword keeps the old behaviour.
    assert snapshot_progress._materialized_bytes(snap, lambda path: path.startswith("mmproj")) == 64


def test_finder_metadata_left_by_a_deleted_quant_is_not_that_quant(tmp_path):
    """macOS keeps a file's metadata in a "._" companion carrying the same name, so it matches
    the quant matcher exactly as its file does. Deleting the quant on a filesystem without
    native xattrs strands that companion, which both walks then read as the quant still being
    here -- hydration re-adopts the stale job and blocks a fresh download. A real GGUF a user
    named that way is still the quant."""
    snap = tmp_path / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (snap / "._model-Q4_K_M.gguf").write_bytes(b"\x00\x05\x16\x07" + b"m" * 60)

    def matcher(path, *, companions = True):
        return path.endswith("Q4_K_M.gguf")

    assert snapshot_progress._materialized_bytes(snap, matcher) == 0
    assert snapshot_progress._variant_main_shard_present(snap, matcher) is False

    (snap / "._named-Q4_K_M.gguf").write_bytes(b"GGUF" + b"q" * 28)
    assert snapshot_progress._materialized_bytes(snap, matcher) == 32
    assert snapshot_progress._variant_main_shard_present(snap, matcher) is True


def test_a_root_that_will_not_resolve_is_a_scan_error(monkeypatch, tmp_path):
    """A root that stats but will not resolve -- an intermittent network mount, a Windows
    reparse point -- was dropped silently, so the scan answered "measured, no cache" and
    hydration retired a download whose files may be entirely intact."""
    from hub.utils import hf_cache_state as state

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches", lambda: [root], raising = False
    )
    monkeypatch.setattr(state, "_safe_is_dir", lambda path, scan_errors = None: path == root)

    real_resolve = Path.resolve

    def _boom(self, *args, **kwargs):
        if self == root:
            raise OSError("network mount unavailable")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _boom)
    errors: list = []
    assert state.hf_cache_roots(errors) == []
    assert any(
        "network mount unavailable" in str(e) for e in errors
    ), "the unreadable root has to be reported, not silently dropped"


def test_a_partial_scan_cannot_report_the_target_as_gone(monkeypatch, tmp_path):
    """One unreadable root plus one readable one is a LOWER bound, not an absence.

    The active root raising EACCES while a remembered cache still holds the repo dir (with a
    sibling quant in it and nothing of ours) produced a non-null reading carrying
    target_present False -- and hydration retires the job on that, though every byte of the
    variant may sit in the root that could not be listed.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "blobs").mkdir(parents = True)
    (entry / "snapshots" / "rev0").mkdir(parents = True)

    def _one_root_failed(
        _repo_type,
        _repo_id,
        force_active = False,
        scan_errors = None,
        **kw,
    ):
        if scan_errors is not None:
            scan_errors.append(PermissionError("denied"))
        return [entry]

    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", _one_root_failed)
    monkeypatch.setattr(snapshot_progress.download_manifest, "read_manifest", lambda *a, **k: None)

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model-GGUF",
        job_key = "org/model-gguf::Q4_K_M",
        expected_bytes = 100,
        hf_token = None,
        variant = "Q4_K_M",
        variant_file_matcher = lambda name: name.endswith("Q4_K_M.gguf"),
        registry = SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
        metadata_resolver = lambda _repo_id, _hf_token: (100, frozenset({"blob-a"})),
        expected_files_resolver = lambda _repo_id, _hf_token: (),
    )

    assert result["target_present"] is None, "absence needs a complete scan behind it"
    assert result["cache_measured"] is False


def test_a_complete_scan_still_reports_the_target_as_gone(monkeypatch, tmp_path):
    # The same reading with no scan error keeps the positive-evidence verdict, or the fix above would
    # simply disable the phantom-adoption guard it is protecting.
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "blobs").mkdir(parents = True)
    (entry / "snapshots" / "rev0").mkdir(parents = True)

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda _repo_type, _repo_id, force_active = False, **kw: [entry],
    )
    monkeypatch.setattr(snapshot_progress.download_manifest, "read_manifest", lambda *a, **k: None)

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model-GGUF",
        job_key = "org/model-gguf::Q4_K_M",
        expected_bytes = 100,
        hf_token = None,
        variant = "Q4_K_M",
        variant_file_matcher = lambda name: name.endswith("Q4_K_M.gguf"),
        registry = SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
        metadata_resolver = lambda _repo_id, _hf_token: (100, frozenset({"blob-a"})),
        expected_files_resolver = lambda _repo_id, _hf_token: (),
    )

    assert result["target_present"] is False
    assert result["cache_measured"] is True


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
    assert extra.is_symlink() and extra.exists()


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
    snapshot_calls = []

    monkeypatch.setattr(
        hf_download,
        "_dataset_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = "dataset-commit",
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 30)],
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
        download_manifest,
        "write_manifest",
        lambda *args, **kwargs: written.append((args, kwargs)) or True,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = (lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path))
        ),
    )

    hf_download._download_dataset("Org/Data", None, "xet")

    assert written, "XET dataset download must still record a manifest"
    assert written[0][0][0:3] == ("dataset", "Org/Data", None)
    assert written[0][0][3][0].path == "data.parquet"
    assert written[0][1] == {"commit_hash": "dataset-commit", "metadata_derived": True}
    assert snapshot_calls == [
        {
            "repo_id": "Org/Data",
            "token": False,
            "repo_type": "dataset",
            "max_workers": 1,
            "revision": "dataset-commit",
        }
    ]
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


def _write_local_model(
    root: Path,
    name: str,
    config: dict,
    *,
    modules: bool = False,
) -> Path:
    path = root / name
    path.mkdir(parents = True)
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (path / "model.safetensors").write_bytes(b"\0" * 16)
    if modules:
        (path / "modules.json").write_text("[]", encoding = "utf-8")
    return path


@pytest.mark.parametrize(
    "config, modules, expected",
    [
        ({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, False, True),
        ({"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"}, False, True),
        ({"auto_map": {"AutoModelForCausalLM": "m.C"}, "model_type": "bert"}, False, True),
        ({"architectures": ["BertModel"], "model_type": "bert"}, False, False),
        ({"architectures": ["BertModel"], "model_type": "bert"}, True, False),
        ({"architectures": ["RobertaForMaskedLM"], "model_type": "roberta"}, False, False),
        ({"architectures": ["CLIPModel"], "model_type": "clip"}, False, False),
        # Unknown architectures must fail OPEN: never hide a real chat model.
        ({"architectures": ["SomeCustomNet"], "model_type": "custom"}, False, None),
        ({}, False, None),
    ],
)
def test_local_transformers_chat_classification(tmp_path, config, modules, expected):
    """A safetensors dir is chat-capable on file format alone, so an embedding
    export looked like a chat model, and auto-load tries the smallest first."""
    path = _write_local_model(tmp_path, "row", config, modules = modules)
    assert model_common._local_transformers_can_chat(path) is expected


def test_local_embedding_model_is_not_chat_capable(tmp_path):
    """End to end through the scanner: the row is still listed, since that is
    how the user deletes it, but can_chat is false so auto-load skips it."""
    _write_local_model(
        tmp_path,
        "all-MiniLM-L6-v2",
        {"architectures": ["BertModel"], "model_type": "bert"},
        modules = True,
    )
    _write_local_model(
        tmp_path,
        "tiny-llama",
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
    )
    rows = {
        row.display_name: row
        for path in sorted(tmp_path.iterdir())
        for row in model_common._classify_local_path(path, "models_dir")
    }
    assert rows["all-MiniLM-L6-v2"].capabilities.can_chat is False
    assert rows["tiny-llama"].capabilities.can_chat is True
    # Training and LoRA support are unchanged: this only gates chat.
    assert rows["all-MiniLM-L6-v2"].capabilities.can_train is True


def test_a_snapshot_whose_only_weight_is_finder_metadata_is_not_a_safetensors_row(tmp_path):
    """Every non-GGUF classification here reads the file list by suffix, and a "._" companion
    carries the suffix of the file it describes -- so a directory left holding only those was
    given a ready safetensors row. A real weight named that way still gets one."""

    def _formats(name, weight = None):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text("{}")
        if weight is not None:
            (d / "._model.safetensors").write_bytes(weight)
        return [r.model_format for r in model_common._classify_local_path(d, "models_dir")]

    # A config with no weight beside it is already "unknown"; metadata must not read as more.
    assert _formats("config-only") == ["unknown"]
    assert _formats("metadata-only", b"\x00\x05\x16\x07\x00\x02\x00\x00") == ["unknown"]
    assert _formats("named", b"weights") == ["safetensors"]


def test_cached_encoder_repo_is_not_chat_capable(tmp_path):
    """Cached rows build capabilities from file format too, so a cached BERT
    looked like a chat model to auto-load."""
    snapshot = _write_local_model(
        tmp_path, "snap", {"architectures": ["BertModel"], "model_type": "bert"}
    )
    fields = cache_inventory._cache_inventory_fields(
        "sentence-transformers/all-MiniLM-L6-v2",
        "safetensors",
        identity = cache_inventory._LoadIdentity(
            load_id = str(snapshot), active_cache = False, load_snapshot = snapshot
        ),
    )
    assert fields["capabilities"]["can_chat"] is False


def test_cached_generative_repo_stays_chat_capable(tmp_path):
    """Control for the gate above."""
    snapshot = _write_local_model(
        tmp_path, "snap", {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    )
    fields = cache_inventory._cache_inventory_fields(
        "unsloth/Llama-3.2-1B-Instruct",
        "safetensors",
        identity = cache_inventory._LoadIdentity(
            load_id = str(snapshot), active_cache = False, load_snapshot = snapshot
        ),
    )
    assert fields["capabilities"]["can_chat"] is True


def test_cached_row_without_a_snapshot_keeps_its_format_capability(tmp_path):
    """Fails open: no snapshot to inspect must not hide the row."""
    fields = cache_inventory._cache_inventory_fields(
        "unsloth/Llama-3.2-1B-Instruct",
        "safetensors",
        identity = cache_inventory._LoadIdentity(
            load_id = "unsloth/Llama-3.2-1B-Instruct",
            active_cache = True,
            load_snapshot = None,
        ),
    )
    assert fields["capabilities"]["can_chat"] is True


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"},
        {"architectures": ["SomeCustomHead"], "model_type": "whisper"},
        {"architectures": ["VisionEncoderDecoderModel"], "model_type": "vision-encoder-decoder"},
        {"architectures": ["MusicgenForConditionalGeneration"], "model_type": "musicgen"},
    ],
)
def test_non_chat_conditional_generation_is_not_chat_capable(tmp_path, config):
    """These end in ForConditionalGeneration but cannot answer a text turn, and
    are small enough that the cascade would stop there. The managed cache
    already hides Whisper; scan folders did not."""
    path = _write_local_model(tmp_path, "row", config)
    assert model_common._local_transformers_can_chat(path) is False


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"},
        {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3"},
        {"architectures": ["BartForConditionalGeneration"], "model_type": "bart"},
    ],
)
def test_real_conditional_generation_chat_models_are_unaffected(tmp_path, config):
    """Guard on the list above: multimodal and seq2seq chat models must stay."""
    path = _write_local_model(tmp_path, "row", config)
    assert model_common._local_transformers_can_chat(path) is True


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["ViTModel"], "model_type": "vit"},
        {"architectures": ["Dinov2Model"], "model_type": "dinov2"},
        {"architectures": ["SwinModel"], "model_type": "swin"},
        {"architectures": ["Wav2Vec2Model"], "model_type": "wav2vec2"},
    ],
)
def test_bare_vision_and_audio_backbones_are_not_chat_capable(tmp_path, config):
    """Their class names carry no task suffix, so only the model type gives them
    away."""
    path = _write_local_model(tmp_path, "row", config)
    assert model_common._local_transformers_can_chat(path) is False


def test_unreadable_config_never_fails_the_scan(tmp_path):
    """One bad config.json must classify as unknown, not take the inventory
    request down. Deep nesting raises RecursionError, which is neither
    JSONDecodeError nor OSError."""
    cases = {
        "deep": "[" * 60000 + "]" * 60000,
        "truncated": '{"architectures": ["Llam',
        "not_an_object": '["a", "b"]',
        "empty": "",
    }
    for name, body in cases.items():
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(body, encoding = "utf-8")
        assert model_common._local_transformers_can_chat(path) is None, name


def test_oversized_config_is_skipped_rather_than_read(tmp_path):
    path = tmp_path / "huge"
    path.mkdir()
    (path / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "pad": "'
        + "a" * (model_common._MAX_LOCAL_JSON_BYTES + 1)
        + '"}',
        encoding = "utf-8",
    )
    assert model_common._local_transformers_can_chat(path) is None


def test_a_fifo_named_config_json_does_not_block_the_scan(tmp_path):
    """read_text() on a FIFO with no writer blocks forever, hanging the scan."""
    os = pytest.importorskip("os")
    if not hasattr(os, "mkfifo"):
        pytest.skip("no mkfifo on this platform")
    path = tmp_path / "fifo"
    path.mkdir()
    os.mkfifo(path / "config.json")
    assert model_common._local_transformers_can_chat(path) is None


@pytest.mark.parametrize(
    "model_type",
    ["siglip", "clip", "bert", "vit", "wav2vec2"],
)
def test_an_encoder_without_architectures_is_still_not_chat_capable(tmp_path, model_type):
    """google/siglip2-* ships config.json with no architectures key, and the
    encoder-only branch used to require one, so those rows stayed chat-capable
    and sorted ahead of a real chat model."""
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding = "utf-8")

    assert model_common._local_transformers_can_chat(path) is False


def test_a_causal_lm_without_architectures_still_fails_open(tmp_path):
    """Control: keyed on the encoder-only type list, so an unknown type with no
    architectures stays inconclusive."""
    path = tmp_path / "custom"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "some_new_llm"}), encoding = "utf-8")

    assert model_common._local_transformers_can_chat(path) is None


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "blip", "architectures": ["BlipForConditionalGeneration"]},
        {"model_type": "blip-2", "architectures": ["Blip2ForConditionalGeneration"]},
        {"model_type": "instructblip", "architectures": ["InstructBlipForConditionalGeneration"]},
        {"model_type": "git", "architectures": ["GitForCausalLM"]},
    ],
)
def test_an_image_captioner_is_not_chat_capable(tmp_path, config):
    """These generate text, but only about an image, so a plain text turn fails,
    and the smallest-first cascade would load one and stop looking."""
    path = tmp_path / "captioner"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    assert model_common._local_transformers_can_chat(path) is False


@pytest.mark.parametrize(
    "architecture",
    ["CLIPTextModelWithProjection", "CLIPVisionModelWithProjection", "CLIPTextModel"],
)
def test_a_projection_head_encoder_is_not_chat_capable(tmp_path, architecture):
    """The encoder-only branch also required the name to end in Model, so a
    projection variant fell through and kept its format capability."""
    path = tmp_path / "proj"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "clip", "architectures": [architecture]}), encoding = "utf-8"
    )

    assert model_common._local_transformers_can_chat(path) is False


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "bert", "architectures": ["BertLMHeadModel"]},
        {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]},
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    ],
)
def test_a_generative_head_still_wins_over_the_encoder_type(tmp_path, config):
    """Control for both gates above: a generative architecture is accepted
    before the type list is consulted."""
    path = tmp_path / "gen"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    assert model_common._local_transformers_can_chat(path) is True


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "llama", "architectures": ["LlamaModel"]},
        {"model_type": "gpt2", "architectures": ["GPT2Model"]},
        {"model_type": "t5", "architectures": ["T5Model"]},
        {"model_type": "mistral", "architectures": ["MistralModel"]},
    ],
)
def test_a_bare_text_backbone_is_not_chat_capable(tmp_path, config):
    """AutoModel.save_pretrained writes the backbone name, and a backbone has no
    LM head, so the cascade would stop on one before a usable chat model."""
    path = tmp_path / "backbone"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    assert model_common._local_transformers_can_chat(path) is False


def test_an_unfamiliar_backbone_still_fails_open(tmp_path):
    """The list is explicit, not shape-matched, so a custom FooModel keeps its
    format capability."""
    path = tmp_path / "custom"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "brand_new_arch", "architectures": ["FooModel"]}),
        encoding = "utf-8",
    )

    assert model_common._local_transformers_can_chat(path) is None


@pytest.mark.parametrize("layout", ["flat", "publisher_child"])
def test_lmstudio_scan_matches_gguf_suffix_case_insensitively(tmp_path, layout):
    """The custom and models-dir scans lower() the suffix; these two did not, so a
    .GGUF was invisible only here. Windows and macOS treat the two spellings as one
    name, so the file is reachable but unlisted."""
    lm_dir = tmp_path / "models"
    holder = lm_dir if layout == "flat" else lm_dir / "publisher"
    holder.mkdir(parents = True)
    for name in ("lower.gguf", "UPPER.GGUF", "Mixed.GguF"):
        (holder / name).write_bytes(b"x")

    found = {Path(row.path).name for row in local_inventory._scan_lmstudio_dir(lm_dir)}

    assert found == {"lower.gguf", "UPPER.GGUF", "Mixed.GguF"}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "bert", "architectures": ["BertModel"]}, False),
        ({"model_type": "clip", "architectures": ["CLIPModel"]}, False),
        ({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, True),
    ],
)
def test_custom_promotion_keeps_the_classifier_verdict(tmp_path, config, expected):
    """Promotion rebuilt capabilities from the format alone, which restored
    can_chat on rows the classifier had ruled out."""
    model_dir = tmp_path / "scan" / "model"
    model_dir.mkdir(parents = True)
    (model_dir / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x" * 1024)

    rows = local_inventory._scan_custom_folder(tmp_path / "scan")
    assert rows, "the scan produced no row"
    for row in rows:
        promoted = local_inventory._promote_to_custom_source(row)
        assert promoted.source == "custom"
        assert promoted.capabilities.can_chat is expected


# ── local diffusers pipelines reach the Images / Video pickers ───────────────

def _write_pipeline(root: Path, *, components = ("transformer", "vae", "text_encoder")) -> Path:
    """A diffusers PIPELINE directory: a root model_index.json, no root config.json, and the
    weights inside component subdirs. Every image and video model downloaded as a pipeline
    (MiniMax-H3, HunyuanVideo, Qwen-Image, HiDream) has exactly this shape."""
    root.mkdir(parents = True, exist_ok = True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline", "_diffusers_version": "0.39.0"}),
        encoding = "utf-8",
    )
    for name in components:
        part = root / name
        part.mkdir(parents = True, exist_ok = True)
        (part / "config.json").write_text(json.dumps({"model_type": name}), encoding = "utf-8")
        (part / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 1024)
    return root


def test_models_dir_scan_surfaces_a_local_diffusers_pipeline(tmp_path):
    """A pipeline the user already has on disk must be listed. It carries no root config.json
    and no loose weight file, which is the only shape the scan used to accept, so the Video and
    Images pickers showed nothing for a model that was sitting right there."""
    models = tmp_path / "models"
    _write_pipeline(models / "MiniMax-H3-local")

    rows = local_inventory._scan_models_dir(models)

    assert {Path(row.path).name for row in rows} == {"MiniMax-H3-local"}


def test_a_scan_folder_pointed_straight_at_a_pipeline_is_that_model(tmp_path):
    pipeline = _write_pipeline(tmp_path / "MiniMax-H3-local")

    rows = local_inventory._scan_models_dir(pipeline, entry_limit = 64)

    assert [Path(row.path) for row in rows] == [pipeline]


def test_the_lmstudio_walk_does_not_descend_into_a_pipeline(tmp_path):
    """LM Studio stores models as publisher/model, so an unrecognised directory is walked one
    level down. A pipeline root looked like a publisher, and its components (vae, transformer,
    text_encoder) were published as separate models -- none of which any loader can start."""
    root = tmp_path / "scan"
    _write_pipeline(root / "MiniMax-H3-local")

    names = {Path(row.path).name for row in local_inventory._scan_lmstudio_dir(root)}

    assert names == {"MiniMax-H3-local"}
    assert not names & {"vae", "transformer", "text_encoder"}


def test_a_scan_folder_pointed_straight_at_a_pipeline_is_not_walked_as_a_publisher(tmp_path):
    """The same walk, entered AT the pipeline. A user adding the model folder itself as a scan
    folder is the obvious thing to do, and it published vae / transformer / text_encoder as three
    models instead of the one that is there."""
    pipeline = _write_pipeline(tmp_path / "MiniMax-H3-local")

    rows = local_inventory._scan_lmstudio_dir(pipeline)

    assert [Path(row.path) for row in rows] == [pipeline]


def test_a_custom_folder_offers_the_pipeline_and_not_its_components(tmp_path):
    """End to end over the path the Custom Folders control uses. The format filter keeps only
    gguf / safetensors / adapter rows, and a pipeline root has no loose weight to classify, so
    it reports "unknown" by construction -- judged on its shape instead."""
    root = tmp_path / "scan"
    _write_pipeline(root / "MiniMax-H3-local")

    rows = local_inventory._scan_custom_folder(root)
    names = {Path(row.path).name for row in rows}

    assert names == {"MiniMax-H3-local"}


def test_a_directory_that_is_not_a_pipeline_is_still_rejected(tmp_path):
    """The new signal is a ROOT model_index.json. A folder that only holds one in a subdir is
    not loadable from its root, and a bare folder is not a model at all."""
    root = tmp_path / "scan"
    (root / "not-a-model").mkdir(parents = True)
    (root / "not-a-model" / "notes.txt").write_text("hello", encoding = "utf-8")
    (root / "nested" / "inner").mkdir(parents = True)
    (root / "nested" / "inner" / "model_index.json").write_text("{}", encoding = "utf-8")

    assert local_inventory._scan_models_dir(root) == []


def test_the_custom_folder_filter_still_drops_a_row_it_cannot_classify(tmp_path):
    """The pipeline exemption widens the custom-folder format filter, so pin what it must NOT let
    through. A folder holding a config.json and no weights (an aborted download) also reports
    "unknown", and nothing can load it: it has to stay filtered out while the pipeline beside it
    is offered."""
    root = tmp_path / "scan"
    _write_pipeline(root / "MiniMax-H3-local")
    aborted = root / "half-downloaded"
    aborted.mkdir(parents = True)
    (aborted / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")

    # The scan does see it, so the filter is what decides -- otherwise this proves nothing.
    assert "half-downloaded" in {
        Path(row.path).name for row in local_inventory._scan_models_dir(root)
    }
    assert {Path(row.path).name for row in local_inventory._scan_custom_folder(root)} == {
        "MiniMax-H3-local",
    }


def test_the_pipeline_test_is_safe_on_a_path_that_is_not_a_readable_directory(tmp_path):
    """``_scan_custom_folder`` applies this to every row it did not already accept, and a row's
    path can be a GGUF FILE, not a directory. A missing path, a file, and a directory whose
    ``model_index.json`` is itself a directory must all answer False rather than raise: an
    exception here fails the whole scan and empties the picker."""
    assert local_inventory._is_diffusers_pipeline_dir(tmp_path / "does-not-exist") is False

    loose = tmp_path / "model.gguf"
    loose.write_bytes(b"GGUF")
    assert local_inventory._is_diffusers_pipeline_dir(loose) is False
    assert local_inventory._is_diffusers_pipeline_dir(loose / "model_index.json") is False

    odd = tmp_path / "odd"
    (odd / "model_index.json").mkdir(parents = True)
    assert local_inventory._is_diffusers_pipeline_dir(odd) is False


def test_a_modular_pipeline_root_is_recognised(tmp_path):
    """A Modular Diffusers pipeline carries ``modular_model_index.json`` and NO
    ``model_index.json``, which is exactly the pair the video loader accepts. Recognising only
    the conventional index hid such a root from the Images/Video picker and let the publisher
    walk descend into it and offer ``transformer`` / ``vae`` as separate, unusable models."""
    root = tmp_path / "modular"
    root.mkdir()
    (root / "modular_model_index.json").write_text("{}")
    (root / "transformer").mkdir()
    assert local_inventory._is_diffusers_pipeline_dir(root) is True

    conventional = tmp_path / "conventional"
    conventional.mkdir()
    (conventional / "model_index.json").write_text("{}")
    assert local_inventory._is_diffusers_pipeline_dir(conventional) is True

    neither = tmp_path / "neither"
    neither.mkdir()
    assert local_inventory._is_diffusers_pipeline_dir(neither) is False


def test_gguf_progress_unknown_hashes_calls_a_sibling_only_dir_absent(monkeypatch, tmp_path):
    """A repo dir kept alive by a sibling quant is not evidence that THIS one is here.

    With the hash set unresolvable, target_present used to stay null, and the frontend's idle
    probe reads zero bytes with a non-null cache_path as an active job unless presence is
    explicitly false -- so the deleted quant's stale card was re-adopted and blocked a fresh
    download until the idle grace ran out. The snapshot dir is named per file, so absence is
    answerable here even when the hashes are not.
    """
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 30)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["target_present"] is False, result


def test_gguf_progress_target_presence_is_aggregated_across_caches(monkeypatch, tmp_path):
    """Presence is a property of the set of caches, not of the one with the most bytes.

    A sibling-only repo dir and a remembered cache that still holds this variant both read as
    zero bytes, so the byte ordering alone could select the sibling-only reading and report the
    target gone -- retiring a job whose files another scanned cache proves are there.
    """
    sibling = tmp_path / "a" / "models--Org--Model-GGUF"
    (sibling / "snapshots" / "rev0").mkdir(parents = True)
    (sibling / "blobs").mkdir(parents = True)
    (sibling / "snapshots" / "rev0" / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    holder = tmp_path / "b" / "models--Org--Model-GGUF"
    (holder / "snapshots" / "rev0").mkdir(parents = True)
    (holder / "blobs").mkdir(parents = True)
    # Zero bytes keeps the tie with the sibling-only reading, and the name is what proves the target is in this cache.
    (holder / "snapshots" / "rev0" / "model-Q4_K_M.gguf").write_bytes(b"")
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100)],
        "http",
        hub_cache = holder.parent,
    )
    _unresolvable_variant_metadata(monkeypatch, sibling, state = "idle")
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [sibling, holder],
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is True, result


def test_a_running_job_does_not_borrow_another_caches_manifest(monkeypatch, tmp_path):
    """A live download is read from the active root only, so its hashes must come from there.

    An older cache holding a DIFFERENT revision's manifest for the same variant made the two
    disagree, and a disagreement is refused -- so a live download whose own manifest is right
    there lost its hash set and every blob it had written was filtered out by the name-based
    fallback's clamp. Scoped to the roots the scan will actually read, the active manifest
    stands on its own.
    """
    active = tmp_path / "active" / "models--Org--Model-GGUF"
    (active / "blobs").mkdir(parents = True)
    remembered = tmp_path / "old" / "models--Org--Model-GGUF"
    (remembered / "blobs").mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100, sha256 = "old")],
        "http",
        hub_cache = remembered.parent,
    )
    monkeypatch.setattr(
        downloads,
        "preferred_repo_cache_dirs",
        lambda *_a, force_active = False, active_root = None, **_kw: (
            [active] if force_active else [active, remembered]
        ),
    )

    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100, sha256 = "new")],
        "http",
        hub_cache = active.parent,
    )
    monkeypatch.setattr(
        download_manifest, "_canonical_hub_cache", lambda root = None: str(root or "")
    )

    scoped = downloads._variant_manifest_in_any_cache(
        "Org/Model-GGUF", "Q4_K_M", force_active = True, active_root = active.parent
    )
    assert scoped is not None and scoped.expected_files[0].sha256 == "new"
    # Unscoped, the superseded cache is consulted too and the disagreement refuses both.
    assert (
        downloads._variant_manifest_in_any_cache(
            "Org/Model-GGUF", "Q4_K_M", active_root = active.parent
        )
        is None
    )


def test_an_unreadable_blobs_dir_is_not_evidence_of_absence(monkeypatch, tmp_path):
    """EACCES on blobs/ is not an empty blobs/. Swallowing it produced a MEASURED zero -- bytes
    0, target_present false, cache_measured true -- and idle hydration retires a persisted job
    on exactly that shape, though the cache was never actually read."""
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "snapshots" / "rev0").mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)
    real_iterdir = Path.iterdir

    def _deny(self):
        if self.name == "blobs":
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _deny)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 130,
        )
    )

    assert result["target_present"] is None, result
    assert result.get("cache_measured") is not True, result


def test_a_manifest_alone_is_not_evidence_the_variant_is_on_disk(monkeypatch, tmp_path):
    """The state-dir manifest says what the target SHOULD contain. It survives a deletion made
    outside the app, so with a sibling quant keeping the repo dir alive it made a variant with
    nothing left on disk read as present -- the phantom job this field exists to retire."""
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "snapshots" / "rev0").mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (entry / "snapshots" / "rev0" / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100, sha256 = "aa")],
        "http",
        hub_cache = entry.parent,
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

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_requirements", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_blob_hashes", lambda *_a, **_kw: frozenset({"aa"})
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["completed_bytes"] == 0
    assert result["target_present"] is False, result


def test_one_unknown_cache_keeps_absence_unknown(monkeypatch, tmp_path):
    """Absence needs EVERY scanned cache to say so. A sibling-only dir reporting false could win
    the zero-byte tie over a cache with no readable snapshot to identify the variant from --
    whose shared blobs dir may still hold an unattributable partial -- and the job was retired
    on the strength of the one reading that could not see it."""
    sibling = tmp_path / "a" / "models--Org--Model-GGUF"
    (sibling / "snapshots" / "rev0").mkdir(parents = True)
    (sibling / "blobs").mkdir(parents = True)
    (sibling / "snapshots" / "rev0" / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    unknown = tmp_path / "b" / "models--Org--Model-GGUF"
    (unknown / "blobs").mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, sibling, state = "idle")
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [sibling, unknown],
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is None, result


def test_an_unattributable_partial_keeps_presence_unknown(monkeypatch, tmp_path):
    """A restarted download whose hashes could not be resolved has its bytes in an .incomplete
    blob that is not linked into any snapshot yet, so the by-name scan -- which is what answers
    presence on that path -- reports a confident absence. Idle hydration retires a persisted job
    on that verdict, throwing away a partial the user can still resume."""
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    # A sibling quant keeps the repo dir alive; the requested variant has nothing materialized.
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    (blobs / "somehash.incomplete").write_bytes(b"x" * 40)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry, state = "idle")

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is None, result


def test_a_subtree_that_cannot_be_scanned_keeps_presence_unknown(monkeypatch, tmp_path):
    """Enumeration succeeding does not mean it was complete. Path.rglob suppresses every OSError
    raised while scanning -- documented behaviour since 3.13 -- so a Windows ACL denial or a
    network-filesystem hiccup on one subdirectory came back as a short list that reads exactly
    like an empty one, and the scan then reported the variant absent though the unreadable
    subtree may hold its main shard. Idle hydration retires a persisted download on that
    verdict."""
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    denied = snap / "split"
    denied.mkdir()
    (denied / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry)
    real_scandir = os.scandir

    def _deny(path, *args, **kwargs):
        if str(path) == str(denied):
            raise PermissionError(13, "Permission denied")
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", _deny)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is None, result


def test_a_blob_that_cannot_be_stated_keeps_presence_unknown(monkeypatch, tmp_path):
    """The hashes resolved, so the blob loop is what measures the variant -- and a blob it could
    not inspect is not a blob that is not there. Swallowing the error produced a MEASURED
    absence, which hydration reads as gone and retires a persisted download."""
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "snapshots" / "rev0").mkdir(parents = True)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "mainhash").write_bytes(b"x" * 100)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_requirements", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_a, **_kw: frozenset({"mainhash"}),
    )
    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", lambda *_a, **_kw: [entry])
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
    )
    real_stat = Path.stat

    def _deny(self, *a, **kw):
        if self.name == "mainhash":
            raise PermissionError("denied")
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", _deny)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is None, result


def test_an_older_snapshot_still_proves_the_variant_is_here(monkeypatch, tmp_path):
    """A cache can retain several revisions. The requested quant living in an older snapshot
    while the newest holds only a sibling read as absent, and hydration retired a job whose
    target is still perfectly usable."""
    entry = tmp_path / "models--Org--Model-GGUF"
    old_snap = entry / "snapshots" / "rev0"
    new_snap = entry / "snapshots" / "rev1"
    old_snap.mkdir(parents = True)
    new_snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (old_snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (new_snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    import os
    import time

    # Make rev1 unambiguously the newest, which is the one latest_snapshot_dir picks.
    os.utime(old_snap, (time.time() - 600, time.time() - 600))
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry, state = "idle")

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is True, result


def test_a_verified_completion_wins_a_byte_tie_between_caches(monkeypatch, tmp_path):
    """Two remembered caches can clamp to the same byte total while only one has a manifest that
    verifies against disk. The byte-ordered pick then carried whichever came first by root
    order, so the response stayed capped below 100% and kept offering Retry for a variant that
    is demonstrably complete in the other cache."""
    unverified = tmp_path / "a" / "models--Org--Model-GGUF"
    (unverified / "snapshots" / "rev0").mkdir(parents = True)
    (unverified / "blobs").mkdir(parents = True)
    (unverified / "snapshots" / "rev0" / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    verified = tmp_path / "b" / "models--Org--Model-GGUF"
    (verified / "snapshots" / "rev0").mkdir(parents = True)
    (verified / "blobs").mkdir(parents = True)
    (verified / "snapshots" / "rev0" / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100)],
        "http",
        hub_cache = verified.parent,
    )
    _unresolvable_variant_metadata(monkeypatch, unverified, state = "idle")
    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        # The unverified cache first, so root order alone would carry it.
        lambda *_a, **_kw: [unverified, verified],
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["complete_on_disk"] is True, result


def test_an_unstatable_blobs_dir_is_not_an_absent_one(monkeypatch, tmp_path):
    """Path.is_dir() swallows a whole class of OSError and answers False, so a failure on the
    blobs directory ITSELF read as "no blobs here" -- a measured absence, which idle hydration
    retires a persisted download on. ELOOP is the case it hides (a symlink cycle, or a
    network-filesystem path that stops resolving); EACCES it re-raises, which the same try
    now contains rather than letting it escape the whole reading."""
    entry = tmp_path / "models--Org--Model-GGUF"
    (entry / "snapshots" / "rev0").mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    _unresolvable_variant_metadata(monkeypatch, entry, state = "idle")
    real_stat = os.stat

    denied = {"hit": False}

    def _deny(path, *a, **kw):
        if str(path).endswith("blobs"):
            denied["hit"] = True
            raise OSError(errno.ELOOP, "too many levels of symbolic links")
        return real_stat(path, *a, **kw)

    monkeypatch.setattr(snapshot_progress.os, "stat", _deny)

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["target_present"] is None, result
    assert result.get("cache_measured") is not True, result
    assert denied["hit"], "the test must actually exercise the directory stat"


def test_a_variant_complete_in_an_older_snapshot_settles(monkeypatch, tmp_path):
    """The presence check already looks at every retained snapshot, but the byte reading and the
    manifest verification did not: a quant complete in an older revision, with the newest
    holding only a sibling, reported 0 bytes and never settled -- 99% and adoptable forever."""
    entry = tmp_path / "models--Org--Model-GGUF"
    old_snap = entry / "snapshots" / "rev0"
    new_snap = entry / "snapshots" / "rev1"
    old_snap.mkdir(parents = True)
    new_snap.mkdir(parents = True)
    (entry / "blobs").mkdir(parents = True)
    (old_snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (new_snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    os.utime(old_snap, (time.time() - 600, time.time() - 600))
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100)],
        "http",
        hub_cache = entry.parent,
    )
    _unresolvable_variant_metadata(monkeypatch, entry, state = "idle")

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["completed_bytes"] == 100, result
    assert result["complete_on_disk"] is True, result


def test_a_deleted_snapshot_link_is_absent_even_with_its_blob_left_behind(monkeypatch, tmp_path):
    """Deleting a GGUF's snapshot entry normally leaves its finalized blob in the shared blobs/
    dir, and a companion blob shared with a sibling keeps the tally positive on its own. Reading
    presence off those counters called a quant that is gone present, and idle hydration
    re-adopted the phantom and blocked a fresh download of it."""
    entry = tmp_path / "models--Org--Model-GGUF"
    snap = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    # The finalized blob survives; the snapshot entry that named it does not.
    (blobs / "mainhash").write_bytes(b"x" * 100)
    (snap / "model-Q2_K.gguf").write_bytes(b"z" * 900)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_requirements", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_a, **_kw: frozenset({"mainhash"}),
    )
    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", lambda *_a, **_kw: [entry])
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["completed_bytes"] == 100, "the orphaned blob is still counted"
    assert result["target_present"] is False, result


def test_a_stale_revisions_filenames_do_not_settle_the_resolved_one(monkeypatch, tmp_path):
    """verify_against_disk compares names and sizes, not sha256. An older retained revision can
    carry the same filenames at the same sizes, so a blob finalized but never linked (a crash
    between the two) let the stale snapshot satisfy the check -- the job settled on files the
    app would not load."""
    entry = tmp_path / "models--Org--Model-GGUF"
    stale = entry / "snapshots" / "rev0"
    blobs = entry / "blobs"
    stale.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (blobs / "oldhash").write_bytes(b"y" * 100)
    (blobs / "newhash").write_bytes(b"x" * 100)
    os.symlink(blobs / "oldhash", stale / "model-Q4_K_M.gguf")
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    assert download_manifest.write_manifest(
        "model",
        "Org/Model-GGUF",
        "Q4_K_M",
        [download_manifest.ExpectedFile(path = "model-Q4_K_M.gguf", size = 100)],
        "http",
        hub_cache = entry.parent,
    )

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(downloads.asyncio, "to_thread", _run_inline)
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_requirements", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_blob_hashes",
        lambda *_a, **_kw: frozenset({"newhash"}),
    )
    monkeypatch.setattr(snapshot_progress, "preferred_repo_cache_dirs", lambda *_a, **_kw: [entry])
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(get_job = lambda _key: SimpleNamespace(state = "idle")),
    )

    result = asyncio.run(
        downloads.get_gguf_download_progress_response(
            "Org/Model-GGUF",
            variant = "Q4_K_M",
            expected_bytes = 100,
        )
    )

    assert result["complete_on_disk"] is False, result


def test_local_inventory_classifies_off_the_event_loop(monkeypatch):
    """Classification must not block unrelated event-loop work."""
    from hub.services.models import local_inventory

    idents: list[int] = []
    loop_is_free = threading.Event()
    model = SimpleNamespace(id = "model", path = "model")
    model.model_copy = lambda update: SimpleNamespace(id = model.id, path = model.path, **update)
    response = SimpleNamespace(models = [model])
    response.model_copy = lambda update: SimpleNamespace(models = update["models"])

    def classify_row(row):
        idents.append(threading.get_ident())
        # Only a responsive event loop can set this event.
        assert loop_is_free.wait(10), "the event loop was blocked while classification ran"
        return "task"

    async def scan(*_args):
        return response

    async def no_folders():
        return []

    monkeypatch.setattr(catalog_classification, "_local_model_task", classify_row)
    monkeypatch.setattr(local_inventory, "_scan_local_models_response", scan)
    monkeypatch.setattr(local_inventory, "_load_custom_folders", no_folders)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: ("roots",))
    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: 0)

    async def run():
        async def keep_the_loop_moving():
            while not idents:
                await asyncio.sleep(0.005)
            loop_is_free.set()

        listing = asyncio.create_task(local_inventory.list_local_models_response("./models"))
        await asyncio.wait_for(asyncio.gather(listing, keep_the_loop_moving()), timeout = 15)
        return threading.get_ident(), listing.result()

    loop_ident, listed = asyncio.run(run())
    assert [row.task for row in listed.models] == ["task"]
    assert idents and loop_ident not in idents, "classification ran on the event loop thread"


def test_local_inventory_derives_speech_task_from_filesystem_codec(monkeypatch):
    """A renamed non-GGUF TTS checkpoint has no family hint, so its tokenizer
    decoder is the only evidence that can place it in the Audio picker."""
    from hub.services.models import local_inventory

    model = SimpleNamespace(id = "renamed-checkpoint")
    model.model_copy = lambda update: SimpleNamespace(id = model.id, **update)
    response = SimpleNamespace(models = [model])
    response.model_copy = lambda update: SimpleNamespace(models = update["models"])

    async def scan(*_args):
        return response

    async def no_folders():
        return []

    monkeypatch.setattr(catalog_classification, "_local_model_task", lambda _row: None)
    monkeypatch.setattr(catalog_classification, "_local_model_audio_type", lambda _row: "snac")
    monkeypatch.setattr(local_inventory, "_scan_local_models_response", scan)
    monkeypatch.setattr(local_inventory, "_load_custom_folders", no_folders)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: ("roots",))
    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: 0)

    listed = asyncio.run(local_inventory.list_local_models_response("./renamed-tts-models"))
    assert [(row.task, row.audio_type) for row in listed.models] == [("text-to-speech", "snac")]


def test_local_inventory_classifies_a_superseded_result_off_the_event_loop(monkeypatch):
    """The give-up path serves the freshest scan it has, and classifies it the same way."""
    from hub.services.models import local_inventory

    idents: list[int] = []
    epoch = [0]
    model = SimpleNamespace(id = "model", path = "model")
    model.model_copy = lambda update: SimpleNamespace(id = model.id, path = model.path, **update)
    response = SimpleNamespace(models = [model])
    response.model_copy = lambda update: SimpleNamespace(models = update["models"])

    async def always_superseded(*_args):
        epoch[0] += 1
        return response

    async def no_folders():
        return []

    monkeypatch.setattr(
        catalog_classification,
        "_local_model_task",
        lambda row: idents.append(threading.get_ident()) or "task",
    )
    monkeypatch.setattr(local_inventory, "_scan_local_models_response", always_superseded)
    monkeypatch.setattr(local_inventory, "_load_custom_folders", no_folders)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: ("roots",))
    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])

    async def run():
        listed = await asyncio.wait_for(
            local_inventory.list_local_models_response("./models"), timeout = 15
        )
        return threading.get_ident(), listed

    loop_ident, listed = asyncio.run(run())
    assert [row.task for row in listed.models] == ["task"]
    assert idents and loop_ident not in idents, "classification ran on the event loop thread"


def test_local_inventory_retries_when_the_cache_changes_during_classification(monkeypatch):
    """A deletion landing while classification runs must not be answered with the old rows."""
    from hub.services.models import local_inventory

    epoch = [0]
    scans: list[int] = []

    def _scan_response(tag: str):
        row = SimpleNamespace(id = tag, path = tag)
        row.model_copy = lambda update, tag = tag: SimpleNamespace(id = tag, path = tag, **update)
        response = SimpleNamespace(models = [row])
        response.model_copy = lambda update: SimpleNamespace(models = update["models"])
        return response

    async def scan(*_args):
        scans.append(epoch[0])
        return _scan_response(f"scan{len(scans)}")

    def classify_row(row):
        # The cache is invalidated while the first scan's rows are being classified.
        if len(scans) == 1:
            epoch[0] += 1
        return "task"

    async def no_folders():
        return []

    monkeypatch.setattr(catalog_classification, "_local_model_task", classify_row)
    monkeypatch.setattr(local_inventory, "_scan_local_models_response", scan)
    monkeypatch.setattr(local_inventory, "_load_custom_folders", no_folders)
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: ("roots",))
    monkeypatch.setattr(local_inventory.hf_cache_scan, "hf_cache_scans_epoch", lambda: epoch[0])

    async def run():
        return await asyncio.wait_for(
            local_inventory.list_local_models_response("./models"), timeout = 15
        )

    listed = asyncio.run(run())
    assert scans == [0, 1], scans
    assert [row.id for row in listed.models] == ["scan2"]


def _gguf_with_architecture(path: Path, architecture: str) -> None:
    """A minimal valid GGUF carrying just ``general.architecture``."""
    import struct

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    path.parent.mkdir(parents = True, exist_ok = True)
    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)


def test_cached_gguf_task_describes_the_revision_the_load_id_resolves_to(tmp_path):
    """A bare repo id loads through ``refs/main``, which is not always the newest payload
    snapshot: a revision-pinned fetch adds a newer one without moving the ref. Classifying the
    newest then advertised a task for a revision the load never reads, and the pickers filter
    On Device rows on exactly that field."""
    hub_cache = tmp_path / "hub"
    repo_path = hub_cache / "models--Org--Model-GGUF"
    older = repo_path / "snapshots" / "aaaaold"
    newer = repo_path / "snapshots" / "bbbbnew"
    _gguf_with_architecture(older / "model-Q4_K_M.gguf", "llama")
    _gguf_with_architecture(newer / "model-Q4_K_M.gguf", "flux")
    (repo_path / "refs").mkdir(parents = True, exist_ok = True)
    (repo_path / "refs" / "main").write_text("aaaaold")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    def revision(snapshot: Path) -> SimpleNamespace:
        gguf = snapshot / "model-Q4_K_M.gguf"
        return SimpleNamespace(
            snapshot_path = snapshot,
            files = [
                SimpleNamespace(
                    file_name = "model-Q4_K_M.gguf",
                    size_on_disk = 64,
                    file_path = gguf,
                    blob_path = gguf,
                )
            ],
            refs = set(),
            commit_hash = snapshot.name,
            last_modified = 1.0,
            size_on_disk = 64,
        )

    repo_info = SimpleNamespace(
        repo_id = "Org/Model-GGUF",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [revision(older), revision(newer)],
        size_on_disk = 128,
        last_accessed = 2.0,
        last_modified = 2.0,
        nb_files = 2,
    )

    rows = cache_inventory._scan_cached_gguf(
        cache_scans = [SimpleNamespace(repos = [repo_info])], active_hub_cache = hub_cache
    )
    row = next(row for row in rows if row["repo_id"] == "Org/Model-GGUF")
    # The id resolves through refs/main to the llama revision, so the row must say so.
    assert row["load_id"] == "Org/Model-GGUF"
    assert row["task"] == "text-generation"


def test_cached_community_orpheus_gguf_is_not_chat_loadable(tmp_path):
    hub_cache = tmp_path / "hub"
    repo_path = hub_cache / "models--QuantFactory--orpheus-3b-0.1-ft-GGUF"
    snapshot = repo_path / "snapshots" / "revision"
    gguf = snapshot / "orpheus-3b-0.1-ft-Q4_K_M.gguf"
    _gguf_with_architecture(gguf, "llama")
    (repo_path / "refs").mkdir(parents = True, exist_ok = True)
    (repo_path / "refs" / "main").write_text("revision")

    revision = SimpleNamespace(
        snapshot_path = snapshot,
        files = [
            SimpleNamespace(
                file_name = gguf.name,
                size_on_disk = 64,
                file_path = gguf,
                blob_path = gguf,
            )
        ],
        refs = {"main"},
        commit_hash = "revision",
        last_modified = 1.0,
        size_on_disk = 64,
    )
    repo_info = SimpleNamespace(
        repo_id = "QuantFactory/orpheus-3b-0.1-ft-GGUF",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [revision],
        size_on_disk = 64,
        last_accessed = 1.0,
        last_modified = 1.0,
        nb_files = 1,
    )

    rows = cache_inventory._scan_cached_gguf(
        cache_scans = [SimpleNamespace(repos = [repo_info])], active_hub_cache = hub_cache
    )
    row = next(row for row in rows if row["repo_id"] == repo_info.repo_id)
    assert row["task"] == "text-to-speech"
    assert row["capabilities"]["can_chat"] is False


def test_every_row_key_the_scanner_emits_survives_the_response_schema():
    """``response_model`` silently DROPS any key the schema does not declare.

    ``_scan_cached_models`` grew a ``diffusers`` flag, which is the only gate keeping an
    untrusted or unrecognised pipeline out of a chat picker: such a repo carries no task, and
    its pipeline root has no config for can_chat to read. Undeclared, the flag reached the CLI
    (which reads the dict in-process) but never the browser, so the two disagreed about the
    same row.

    The watched set is an explicit list, not every key the scanner emits: the AST harvest
    over-approximates, picking up internal bookkeeping from nested dict literals that was never
    meant to leave the process. So a NEW picker-visible flag is not covered the day it lands --
    add it here when you add it to the scanner.
    """
    import ast
    import pathlib

    from hub.schemas.inventory import CachedGgufRepo, CachedModelRepo

    source = pathlib.Path(cache_inventory.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)

    def literal_keys(function_name: str) -> set:
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ):
                return {
                    key.value
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Dict)
                    for key in inner.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
        return set()

    # Each scanner against ITS OWN schema: a union would let a key emitted on a model row pass because
    # the GGUF schema happens to declare it, which is not what response_model does.
    emitted = literal_keys("_cache_inventory_fields") | literal_keys("_scan_cached_models")
    watched = ("diffusers", "companion", "single_file", "partial", "load_id", "task")
    for flag in watched:
        if flag in emitted:
            assert flag in CachedModelRepo.model_fields, (
                f"_scan_cached_models emits {flag!r} but CachedModelRepo does not declare it, so "
                f"FastAPI's response_model strips it before the frontend sees the row."
            )

    # Prove it end to end rather than by field name alone: a declared-but-mistyped field is dropped or
    # 500s at serialization time, which a model_fields check cannot see.
    row = {
        "repo_id": "Org/Pipeline",
        "size_on_disk": 4096,
        "last_modified": 1.0,
        "diffusers": True,
    }
    assert CachedModelRepo(**row).model_dump()["diffusers"] is True
    assert set(CachedGgufRepo.model_fields), "GGUF schema import is load-bearing above"
