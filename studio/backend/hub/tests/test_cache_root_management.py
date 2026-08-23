# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.schemas.inventory import CachedGgufResponse
from hub.services.datasets import cache_inventory as dataset_cache_inventory
from hub.services.models import cache_inventory, companion_cleanup, deletion
from hub.utils import hf_cache_state


def _select_active_cache(monkeypatch, active: Path) -> None:
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active),
    )


def test_delete_target_root_selection_matrix(monkeypatch, tmp_path):
    active = tmp_path / "active"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _select_active_cache(monkeypatch, active)

    assert hf_cache_state.resolve_delete_target_root("model", "Org/Model", None, []) == active
    assert hf_cache_state.resolve_delete_target_root("model", "Org/Model", None, [first]) == first
    assert (
        hf_cache_state.resolve_delete_target_root(
            "model", "Org/Model", None, [first, active, second]
        )
        == active
    )
    with pytest.raises(hf_cache_state.AmbiguousDeleteTargetError) as raised:
        hf_cache_state.resolve_delete_target_root("model", "Org/Model", None, [first, second])
    assert raised.value.detail == {
        "message": "Multiple cached copies were found. Choose a cache location to delete.",
        "cache_paths": [
            str(first.resolve() / "models--Org--Model"),
            str(second.resolve() / "models--Org--Model"),
        ],
    }


def test_explicit_delete_target_stays_valid_or_400_compatible(monkeypatch, tmp_path):
    active = tmp_path / "active"
    previous = tmp_path / "previous"
    repo = previous / "models--Org--Model"
    repo.mkdir(parents = True)
    _select_active_cache(monkeypatch, active)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [active, previous])

    assert (
        hf_cache_state.resolve_delete_target_root("model", "Org/Model", str(repo), []) == previous
    )
    assert (
        hf_cache_state.resolve_delete_target_root(
            "model", "Org/Model", str(tmp_path / "outside"), []
        )
        is None
    )


def test_cache_copy_merge_preserves_winner_and_all_disk_usage(tmp_path):
    active_path = tmp_path / "active" / "models--Org--Model"
    previous_path = tmp_path / "previous" / "models--Org--Model"
    active = {
        "repo_id": "Org/Model",
        "cache_path": str(active_path),
        "load_id": "Org/Model",
        "size_bytes": 200,
        "active_cache": True,
        "partial": True,
        "last_modified": 20,
    }
    previous = {
        "repo_id": "org/model",
        "cache_path": str(previous_path),
        "load_id": str(previous_path / "snapshots" / "rev-old"),
        "size_bytes": 100,
        "active_cache": False,
        "partial": False,
        "last_modified": 10,
    }

    forward = cache_inventory._merge_cache_row_copies(
        previous, cache_inventory._merge_cache_row_copies(active, None)
    )
    reverse = cache_inventory._merge_cache_row_copies(
        active, cache_inventory._merge_cache_row_copies(previous, None)
    )

    for merged in (forward, reverse):
        # Completeness still chooses the historical row; selected-row semantics stay compatible.
        assert merged["cache_path"] == str(previous_path)
        assert merged["size_bytes"] == 100
        assert merged["active_cache"] is False
        assert merged["copy_count"] == 2
        assert merged["total_size_bytes"] == 300
        assert [copy["cache_path"] for copy in merged["cache_copies"]] == [
            str(active_path),
            str(previous_path),
        ]
        assert [copy["load_id"] for copy in merged["cache_copies"]] == [
            "Org/Model",
            str(previous_path / "snapshots" / "rev-old"),
        ]
        assert merged["last_modified"] == 20

    assert forward["cache_copies"] == reverse["cache_copies"]


def test_cache_copy_merge_resolves_each_path_once(monkeypatch, tmp_path):
    active_path = tmp_path / "active" / "models--Org--Model"
    previous_path = tmp_path / "previous" / "models--Org--Model"
    resolved = []

    def record_resolve(path, *, strict = False):
        resolved.append(path)
        return path

    monkeypatch.setattr(Path, "resolve", record_resolve)
    cache_copy_keys = {}
    active = {
        "cache_path": str(active_path),
        "size_bytes": 200,
        "active_cache": True,
        "partial": False,
    }
    previous = {
        "cache_path": str(previous_path),
        "size_bytes": 100,
        "active_cache": False,
        "partial": False,
    }

    merged = cache_inventory._merge_cache_row_copies(
        active,
        None,
        cache_copy_keys = cache_copy_keys,
    )
    cache_inventory._merge_cache_row_copies(
        previous,
        merged,
        cache_copy_keys = cache_copy_keys,
    )

    assert resolved == [active_path, previous_path]


def test_cache_copy_fields_survive_response_validation(tmp_path):
    repo_path = tmp_path / "hub" / "models--Org--Model"
    row = cache_inventory._merge_cache_row_copies(
        {
            "repo_id": "Org/Model",
            "cache_path": str(repo_path),
            "load_id": "Org/Model",
            "size_bytes": 123,
            "active_cache": True,
            "partial": False,
            "model_format": "gguf",
        },
        None,
    )

    [serialized] = CachedGgufResponse(cached = [row]).model_dump()["cached"]

    assert serialized["size_bytes"] == 123
    assert serialized["total_size_bytes"] == 123
    assert serialized["copy_count"] == 1
    assert serialized["active_cache"] is True
    assert serialized["cache_copies"] == [
        {
            "cache_path": str(repo_path),
            "load_id": "Org/Model",
            "size_bytes": 123,
            "active_cache": True,
            "partial": False,
            "last_modified": None,
        }
    ]

    [legacy] = CachedGgufResponse(
        cached = [
            {
                "repo_id": "Org/Legacy",
                "size_bytes": 77,
                "cache_path": str(tmp_path / "legacy"),
            }
        ]
    ).model_dump()["cached"]
    assert legacy["active_cache"] is None
    assert legacy["copy_count"] is None
    assert legacy["total_size_bytes"] is None


def test_gguf_scan_aggregates_copies_whichever_root_is_scanned_first(monkeypatch, tmp_path):
    active_root = tmp_path / "active"
    previous_root = tmp_path / "previous"
    repo_name = "models--Org--Model"
    active_repo = SimpleNamespace(
        repo_id = "Org/Model",
        repo_type = "model",
        repo_path = active_root / repo_name,
        revisions = [],
    )
    previous_repo = SimpleNamespace(
        repo_id = "org/model",
        repo_type = "model",
        repo_path = previous_root / repo_name,
        revisions = [],
    )
    sizes = {str(active_repo.repo_path): 80, str(previous_repo.repo_path): 120}

    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "build_variant_state_index",
        lambda *_args, **_kwargs: SimpleNamespace(for_repo = lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_repo_gguf_size_bytes",
        lambda repo: sizes[str(repo.repo_path)],
    )
    monkeypatch.setattr(cache_inventory, "_repo_gguf_last_modified", lambda _repo: 0)
    monkeypatch.setattr(
        cache_inventory, "_gguf_variant_state_summary", lambda *_a, **_k: (False, 0)
    )
    monkeypatch.setattr(cache_inventory, "_is_hidden_infra_repo", lambda *_a: False)
    monkeypatch.setattr(cache_inventory, "_cached_model_snapshot_path", lambda _path: None)
    monkeypatch.setattr(
        cache_inventory, "_repo_gguf_payload_snapshots", lambda _repo: (None, frozenset())
    )
    monkeypatch.setattr(cache_inventory, "_cached_row_task", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "_cache_inventory_fields",
        lambda repo_id, _format, *, repo_path, **_kwargs: {
            "inventory_id": f"cache:gguf:{repo_id}",
            "load_id": repo_id,
            "active_cache": repo_path.parent == active_root,
            "model_format": "gguf",
            "format_variant": None,
            "capabilities": {},
        },
    )

    results = []
    for ordered in ([active_repo, previous_repo], [previous_repo, active_repo]):
        [row] = cache_inventory._scan_cached_gguf(
            cache_scans = [SimpleNamespace(repos = ordered)],
            active_hub_cache = active_root,
        )
        results.append(row)

    for row in results:
        assert row["cache_path"] == str(active_repo.repo_path)
        assert row["size_bytes"] == 80
        assert row["copy_count"] == 2
        assert row["total_size_bytes"] == 200
        assert [copy["cache_path"] for copy in row["cache_copies"]] == [
            str(active_repo.repo_path),
            str(previous_repo.repo_path),
        ]


def test_non_gguf_scan_aggregates_copies_whichever_root_is_scanned_first(monkeypatch, tmp_path):
    active_root = tmp_path / "active"
    previous_root = tmp_path / "previous"
    repo_name = "models--Org--Model"
    active_repo = SimpleNamespace(
        repo_id = "Org/Model",
        repo_type = "model",
        repo_path = active_root / repo_name,
        revisions = [],
    )
    previous_repo = SimpleNamespace(
        repo_id = "org/model",
        repo_type = "model",
        repo_path = previous_root / repo_name,
        revisions = [],
    )
    sizes = {str(active_repo.repo_path): 80, str(previous_repo.repo_path): 120}

    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "build_variant_state_index",
        lambda *_args, **_kwargs: SimpleNamespace(for_repo = lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_cached_model_snapshot_path",
        lambda repo_path: Path(repo_path) / "snapshots" / "revision",
    )
    monkeypatch.setattr(cache_inventory, "_cached_model_local_metadata", lambda *_a: {})
    monkeypatch.setattr(cache_inventory, "_is_hidden_infra_repo", lambda *_a: False)
    monkeypatch.setattr(cache_inventory, "_repo_has_gguf_files", lambda *_a: False)

    def payload(repo):
        snapshot = Path(repo.repo_path) / "snapshots" / "revision"
        return cache_inventory._CachedNonGgufPayload(
            sizes[str(repo.repo_path)],
            True,
            "safetensors",
            0,
            snapshot,
            frozenset({str(snapshot)}),
        )

    monkeypatch.setattr(cache_inventory, "_repo_non_gguf_model_payload", payload)
    monkeypatch.setattr(
        cache_inventory,
        "_resolve_load_identity",
        lambda repo_id, *, repo_path, snapshot_path, **_kwargs: cache_inventory._LoadIdentity(
            repo_id if Path(repo_path).parent == active_root else str(snapshot_path),
            Path(repo_path).parent == active_root,
            snapshot_path,
        ),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "snapshot_pipeline_missing_denoiser",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "snapshot_has_pipeline_index",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(cache_inventory, "_cached_row_task", lambda *_a, **_k: None)
    monkeypatch.setattr(cache_inventory, "_cached_row_companion", lambda *_a: False)
    monkeypatch.setattr(
        cache_inventory,
        "_cache_inventory_fields",
        lambda repo_id, model_format, *, identity, **_kwargs: {
            "inventory_id": f"cache:{model_format}:{repo_id}",
            "load_id": identity.load_id,
            "active_cache": identity.active_cache,
            "model_format": model_format,
            "format_variant": None,
            "capabilities": {},
        },
    )

    results = []
    for ordered in ([active_repo, previous_repo], [previous_repo, active_repo]):
        [row] = cache_inventory._scan_cached_models(
            cache_scans = [SimpleNamespace(repos = ordered)],
            active_hub_cache = active_root,
        )
        results.append(row)

    for row in results:
        assert row["cache_path"] == str(active_repo.repo_path)
        assert row["size_bytes"] == 80
        assert row["copy_count"] == 2
        assert row["total_size_bytes"] == 200
        assert [copy["cache_path"] for copy in row["cache_copies"]] == [
            str(active_repo.repo_path),
            str(previous_repo.repo_path),
        ]


def test_delete_preview_maps_ambiguous_unscoped_copy_to_409(monkeypatch, tmp_path):
    active = tmp_path / "active"
    repos = [
        SimpleNamespace(
            repo_id = "Org/Model",
            repo_type = "model",
            repo_path = root / "models--Org--Model",
        )
        for root in (tmp_path / "first", tmp_path / "second")
    ]
    _select_active_cache(monkeypatch, active)
    monkeypatch.setattr(
        companion_cleanup.cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = repos)],
    )

    with pytest.raises(HTTPException) as raised:
        companion_cleanup._delete_impact_blocking("Org/Model", None, None)

    assert raised.value.status_code == 409
    assert raised.value.detail == {
        "message": "Multiple cached copies were found. Choose a cache location to delete.",
        "cache_paths": [
            str(tmp_path / "first" / "models--Org--Model"),
            str(tmp_path / "second" / "models--Org--Model"),
        ],
    }


def test_model_delete_maps_ambiguous_unscoped_copy_to_409(monkeypatch, tmp_path):
    active = tmp_path / "active"
    repos = [
        SimpleNamespace(
            repo_id = "Org/Model",
            repo_type = "model",
            repo_path = root / "models--Org--Model",
        )
        for root in (tmp_path / "first", tmp_path / "second")
    ]
    _select_active_cache(monkeypatch, active)
    monkeypatch.setattr(deletion, "_is_companion_base_repo", lambda _repo_id: False)
    monkeypatch.setattr(
        deletion.cache_inventory,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = repos)],
    )

    with pytest.raises(HTTPException) as raised:
        deletion._delete_cached_model_blocking("Org/Model", None, None)

    assert raised.value.status_code == 409
    assert raised.value.detail == {
        "message": "Multiple cached copies were found. Choose a cache location to delete.",
        "cache_paths": [
            str(tmp_path / "first" / "models--Org--Model"),
            str(tmp_path / "second" / "models--Org--Model"),
        ],
    }


def test_dataset_delete_maps_ambiguous_unscoped_copy_to_409(monkeypatch, tmp_path):
    active = tmp_path / "active"
    scans = [
        SimpleNamespace(
            repos = [
                SimpleNamespace(
                    repo_id = "Org/Data",
                    repo_type = "dataset",
                    repo_path = root / "datasets--Org--Data",
                )
            ]
        )
        for root in (tmp_path / "first", tmp_path / "second")
    ]
    _select_active_cache(monkeypatch, active)
    monkeypatch.setattr(
        dataset_cache_inventory,
        "_collect_hf_cache_scans",
        lambda: (scans, set()),
    )

    with pytest.raises(HTTPException) as raised:
        dataset_cache_inventory._delete_cached_dataset_blocking("Org/Data")

    assert raised.value.status_code == 409
    assert raised.value.detail == {
        "message": "Multiple cached copies were found. Choose a cache location to delete.",
        "cache_paths": [
            str(tmp_path / "first" / "datasets--Org--Data"),
            str(tmp_path / "second" / "datasets--Org--Data"),
        ],
    }
