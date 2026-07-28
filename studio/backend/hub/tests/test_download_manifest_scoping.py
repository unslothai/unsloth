# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from types import SimpleNamespace

from hub.utils import download_manifest, state_dir


def _write_manifest(path, payload):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload), encoding = "utf-8")


def test_purge_state_preserves_active_legacy_when_deleting_inactive_cache(monkeypatch, tmp_path):
    """A scoped delete of an inactive cache must not erase the unscoped legacy
    state, which _legacy_state_applies attributes to the active cache."""
    active = tmp_path / "active" / "hub"
    previous = tmp_path / "previous" / "hub"
    for path in (active, previous):
        path.mkdir(parents = True)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(active)),
    )

    # Unowned legacy manifest -> belongs to the active cache.
    legacy = state_dir.manifest_path("model", "Org/Model")
    _write_manifest(legacy, {"version": 1})
    # The inactive cache's own scoped copy is the one being deleted.
    scoped = state_dir.manifest_path("model", "Org/Model", hub_cache = str(previous))
    _write_manifest(scoped, {"version": 1, "hub_cache": str(previous)})

    removed = download_manifest.purge_state("model", "Org/Model", hub_cache = str(previous))

    assert removed is True
    assert not scoped.is_file()  # the inactive cache's copy is gone
    assert legacy.is_file()  # the active cache's legacy state survives


def test_purge_state_removes_legacy_owned_by_the_deleted_cache(monkeypatch, tmp_path):
    """A legacy file that recorded the deleted cache as its owner is purged."""
    active = tmp_path / "active" / "hub"
    previous = tmp_path / "previous" / "hub"
    for path in (active, previous):
        path.mkdir(parents = True)

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = str(active)),
    )

    legacy = state_dir.manifest_path("model", "Org/Model")
    _write_manifest(legacy, {"version": 1, "hub_cache": str(previous)})

    removed = download_manifest.purge_state("model", "Org/Model", hub_cache = str(previous))

    assert removed is True
    assert not legacy.is_file()  # owned by the deleted cache -> purged
