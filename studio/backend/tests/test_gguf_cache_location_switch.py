# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from hub.services.models import cache_inventory, deletion, gguf_variants
from hub.utils import inventory_scan
from utils import hf_cache_settings


@pytest.mark.parametrize("active_custom", [False, True])
def test_cached_gguf_keeps_quants_in_previous_download_folders(
    monkeypatch, tmp_path, active_custom
):
    store = {}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", store.update)
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "unused")
    default_root = hf_cache_settings.get_hf_cache_paths().hub_cache
    custom_home = tmp_path / "custom"
    repo_id = "Org/Model-GGUF"
    expected = {}

    def write_quant(root, quant):
        repo = root / "models--Org--Model-GGUF"
        snapshot = repo / "snapshots" / ("d" * 40)
        snapshot.mkdir(parents = True)
        filename = f"Model-{quant}.gguf"
        (snapshot / filename).write_bytes(b"\0" * 256)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("d" * 40)
        expected[quant] = (repo, snapshot / filename)

    write_quant(default_root, "Q6_K")
    hf_cache_settings.set_hf_cache_home(str(custom_home))
    write_quant(custom_home / "hub", "Q8_0")
    hf_cache_settings.set_hf_cache_home(None)
    if active_custom:
        hf_cache_settings.set_hf_cache_home(str(custom_home))

    assert custom_home / "hub" in hf_cache_settings.known_hf_hub_caches()
    scans = inventory_scan.all_hf_cache_scans()
    assert sum(repo.repo_id == repo_id for scan in scans for repo in scan.repos) == 2
    rows = [
        row for row in cache_inventory._scan_cached_gguf() if row["repo_id"] == repo_id
    ]
    repeated = cache_inventory._scan_cached_gguf(cache_scans = scans + scans)
    assert [row for row in repeated if row["repo_id"] == repo_id] == rows
    found = {}
    for row in rows:
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response(
                repo_id, prefer_local_cache = True, local_path = row["load_id"]
            )
        )
        for variant in response.variants:
            if variant.downloaded:
                found[variant.quant] = row

    assert set(found) == set(expected), "A quant in the inactive cache disappeared"
    assert len({row["inventory_id"] for row in rows}) == 2
    for quant, (repo, file_path) in expected.items():
        assert found[quant]["cache_path"] == str(repo)
        assert file_path.is_file()
        assert found[quant]["active_cache"] == (
            repo.parent == hf_cache_settings.get_hf_cache_paths().hub_cache
        )
    # Use the exact management path returned by the inventory, even when that
    # folder is inactive. Deleting Q8 must leave the other cache's Q6 intact.
    deletion._delete_cached_model_blocking(
        repo_id, "Q8_0", None, found["Q8_0"]["cache_path"]
    )
    assert not expected["Q8_0"][1].exists()
    assert expected["Q6_K"][1].is_file()
