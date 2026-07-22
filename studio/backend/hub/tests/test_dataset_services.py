# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.schemas.datasets import CheckFormatRequest, LocalDatasetItem
from hub.services.datasets import cache_inventory, downloads, formatting, local
from hub.utils import download_manifest, download_registry, state_dir


class _Upload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload
        self._offset = 0

    async def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_dataset_cache_scan_merges_raw_and_processed_rows(monkeypatch):
    raw_repo = SimpleNamespace(
        repo_id = "Org/Data",
        repo_type = "dataset",
        repo_path = "/cache/datasets--Org--Data",
        size_on_disk = 100,
        revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [raw_repo])], {"/cache"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _repo_type, _repo_id, _cache_dir: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "_scan_hub_dataset_cache_dirs",
        lambda: [],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_scan_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 250,
                "cache_path": "/processed/org___data",
                "processed_cache": True,
                "partial": False,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert rows[0]["repo_id"] == "Org/Data"
    assert rows[0]["size_bytes"] == 250
    assert rows[0]["partial"] is False


def test_delete_cached_dataset_scopes_delete_to_selected_root(monkeypatch, tmp_path):
    """A dataset present in the active cache and a previously selected cache is
    deleted only from the selected root, so the other cache's copy survives."""
    calls = []
    target_hub = tmp_path / "active" / "hub"
    other_hub = tmp_path / "previous" / "hub"
    for hub in (target_hub, other_hub):
        (hub / "datasets--Org--Data").mkdir(parents = True)

    class _DeleteStrategy:
        def __init__(self, label: str):
            self.label = label

        def execute(self):
            calls.append(self.label)

    def _cache(label: str, hub):
        return SimpleNamespace(
            cache_dir = label,
            repos = [
                SimpleNamespace(
                    repo_type = "dataset",
                    repo_id = "Org/Data",
                    repo_path = str(hub / "datasets--Org--Data"),
                    revisions = [SimpleNamespace(commit_hash = f"{label}-rev")],
                )
            ],
            delete_revisions = lambda *_revs, _label = label: _DeleteStrategy(_label),
        )

    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([_cache("active", target_hub), _cache("previous", other_hub)], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args: 0,
    )
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = target_hub),
    )
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_roots",
        lambda: [target_hub, other_hub],
    )

    result = cache_inventory._delete_cached_dataset_blocking("Org/Data")

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    # Only the selected (active) cache's revision is deleted; the previous
    # cache's copy is never touched.
    assert calls == ["active"]
    assert not (target_hub / "datasets--Org--Data").exists()
    assert (other_hub / "datasets--Org--Data").exists()


def test_delete_cached_dataset_purges_blob_only_repo_dir(monkeypatch):
    """A blob-only ``datasets--owner--repo`` dir (no usable snapshot/refs) is
    fully removable: purge_partial_repo alone clears only ``.incomplete`` files
    and would leave the complete blobs and the row."""
    purged_dirs: list[str] = []

    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_repo_cache_dirs",
        lambda _repo_type, repo_id, **_kwargs: purged_dirs.append(repo_id) or True,
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_partial_repo",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args: 0,
    )

    result = cache_inventory._delete_cached_dataset_blocking("Org/Data")

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    assert purged_dirs == ["Org/Data"]


def test_delete_cached_dataset_absent_everywhere_raises_404(monkeypatch):
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_repo_cache_dirs",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_partial_repo",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args: 0,
    )

    with pytest.raises(HTTPException) as exc_info:
        cache_inventory._delete_cached_dataset_blocking("Org/Missing")

    assert exc_info.value.status_code == 404


def test_check_format_rejects_invalid_path_as_400():
    with pytest.raises(HTTPException) as exc_info:
        formatting.check_format_response(CheckFormatRequest(dataset_name = "../../etc/passwd"))

    assert exc_info.value.status_code == 400


def test_dataset_download_status_preserves_idle_shape():
    status = downloads._dataset_status("Org/Data")

    assert status.state == "idle"
    assert status.error is None


def test_dataset_download_registry_key_is_case_insensitive():
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Data",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )
    duplicate_claimed, duplicate_state = registry.claim(
        "org/data",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "org/data",
    )

    assert claimed is True
    assert state == "running"
    assert duplicate_claimed is False
    assert duplicate_state == "running"
    assert registry.active_jobs("ORG/DATA") == {"org/data": "running"}


def test_dataset_idle_status_uses_cancel_marker_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(downloads, "_registry", download_registry.DownloadRegistry())
    assert download_manifest.write_cancel_marker("dataset", "Owner/Data", None, "http")

    status = asyncio.run(downloads.get_dataset_download_status_response("owner/data"))

    assert status.state == "cancelled"
    assert status.error is None


def test_dataset_claim_register_cancel_uses_registry_marker_owner(monkeypatch):
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
        downloads.download_lifecycle,
        "spawn_worker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "kill_and_reap_process",
        lambda proc, **_kwargs: killed.append(proc),
    )

    result = asyncio.run(
        downloads.download_dataset_response(SimpleNamespace(repo_id = "Org/Data", use_xet = False))
    )

    assert result["state"] == "cancelled"
    assert killed


def test_dataset_cancel_pending_spawn_arms_pending_cancel(monkeypatch):
    events = []

    class _Registry:
        def get_process(self, _key):
            return None

        def mark_pending_cancel(self, key, generation):
            events.append(("pending", key, generation))
            return True

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )

    result = asyncio.run(
        downloads.cancel_dataset_download_response(
            SimpleNamespace(repo_id = "Org/Data", generation = 5)
        )
    )

    assert result == {"repo_id": "Org/Data", "state": "cancelling"}
    assert events == [("pending", "org/data", 5)]


def test_upload_dataset_response_writes_non_empty_file(monkeypatch, tmp_path):
    payload = b'{"text":"hello"}\n'
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", tmp_path)

    response = asyncio.run(local.upload_dataset_response(_Upload("../train.jsonl", payload)))

    stored_path = Path(response.stored_path)
    assert response.filename == "train.jsonl"
    assert stored_path.parent == tmp_path
    assert stored_path.name.endswith("_train.jsonl")
    assert stored_path.read_bytes() == payload


def test_local_dataset_items_expose_recipe_and_upload_source(monkeypatch, tmp_path):
    recipe_root = tmp_path / "recipes"
    upload_root = tmp_path / "uploads"
    parquet_dir = recipe_root / "recipe_alpha" / "parquet-files"
    parquet_dir.mkdir(parents = True)
    (parquet_dir / "part.parquet").write_bytes(b"parquet")
    upload_root.mkdir()
    (upload_root / "manual.jsonl").write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.setattr(local, "LOCAL_DATASETS_ROOT", recipe_root)
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", upload_root)

    response = local.list_local_datasets_response()

    assert "source" in LocalDatasetItem.__annotations__
    by_id = {item.id: item for item in response.datasets}
    assert by_id["recipe_alpha"].source == "recipe"
    assert by_id["manual.jsonl"].source == "upload"
