# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the servability scan cache behind GET /v1/models.

A user's log showed /v1/models at 316-621ms while the internal /api/models/list,
which touches no filesystem, ran in 13-34ms. The catalog scan was already cached
for 30s, but the per-entry servability check (stat many files plus a config.json
read for every model) was re-run on every request. It is now keyed on the catalog
stamp, the same way _validated_media_picks is, so a catalog rescan is the only
thing that invalidates it and nothing can go stale behind a second TTL.

Residency must stay per request: it changes on every load/unload.
"""

from types import SimpleNamespace

import pytest

import routes.inference as inf


@pytest.fixture(autouse = True)
def _clean_cache():
    inf._SERVABLE_SCAN_CACHE["entry"] = None
    yield
    inf._SERVABLE_SCAN_CACHE["entry"] = None


def _catalog(n = 3):
    return [SimpleNamespace(id = f"repo/m{i}", path = f"/models/m{i}", task = None) for i in range(n)]


@pytest.fixture
def _stub(monkeypatch):
    """Count servability resolutions and make residency observable."""
    counts = {"servable": 0, "resident": 0}

    def _servable(info):
        counts["servable"] += 1
        return (True, ("Q4_K_M",))

    monkeypatch.setattr("core.inference.local_model_resolver.local_servable_model", _servable)
    monkeypatch.setattr("core.inference.local_model_resolver.local_load_dir", lambda p: p)

    def _resident(key, **kw):
        counts["resident"] += 1
        return False

    monkeypatch.setattr(inf, "_resolves_to_resident", _resident)
    return counts


def test_same_catalog_stamp_scans_once(_stub):
    catalog = _catalog(3)
    for _ in range(10):
        inf._servable_catalog_rows(catalog, 111.0)
    assert _stub["servable"] == 3, "one scan for three entries, not one per call"


def test_residency_is_recomputed_every_call(_stub):
    catalog = _catalog(3)
    for _ in range(10):
        inf._servable_catalog_rows(catalog, 111.0)
    # Residency changes on load/unload, so it must never ride the cache.
    assert _stub["resident"] == 30


def test_new_catalog_stamp_rescans(_stub):
    catalog = _catalog(2)
    inf._servable_catalog_rows(catalog, 111.0)
    assert _stub["servable"] == 2
    inf._servable_catalog_rows(catalog, 222.0)
    assert _stub["servable"] == 4, "a replaced catalog must be rescanned"


def test_no_stamp_disables_caching(_stub):
    # Direct callers that pass no stamp keep the original uncached behaviour.
    catalog = _catalog(2)
    inf._servable_catalog_rows(catalog)
    inf._servable_catalog_rows(catalog)
    assert _stub["servable"] == 4


def test_rows_shape_is_unchanged(_stub):
    rows = inf._servable_catalog_rows(_catalog(1), 111.0)
    assert len(rows) == 1
    info, is_gguf, quants, resident = rows[0]
    assert info.id == "repo/m0"
    assert is_gguf is True
    assert quants == ("Q4_K_M",)
    assert resident is False


def test_unservable_entries_are_dropped(monkeypatch):
    monkeypatch.setattr(
        "core.inference.local_model_resolver.local_servable_model", lambda info: None
    )
    monkeypatch.setattr(inf, "_resolves_to_resident", lambda key, **kw: False)
    assert inf._servable_catalog_rows(_catalog(3), 111.0) == []
