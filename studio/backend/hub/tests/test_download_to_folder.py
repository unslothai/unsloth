# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download to: a model download aimed at a chosen folder instead of the Models folder (#10872)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.services.models import downloads
from hub.services.models import local_inventory
from utils import hf_cache_settings


@pytest.fixture()
def settings_store(monkeypatch, tmp_path):
    store = {}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    # macOS tmp_path sits under /private/var, which the system-folder guard refuses.
    monkeypatch.setattr("hub.storage.scan_folders.is_denied_system_path", lambda _path: False)
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings",
        lambda values: store.update(values) or values,
    )
    return store


def test_no_download_dir_is_the_models_folder(settings_store):
    active = hf_cache_settings.get_hf_cache_paths()
    assert hf_cache_settings.download_cache_paths(None) == active
    assert hf_cache_settings.download_cache_paths("  ") == active


def test_download_dir_is_laid_out_like_a_models_folder(settings_store, tmp_path):
    target = tmp_path / "drive-e" / "models"
    target.parent.mkdir()

    paths = hf_cache_settings.download_cache_paths(str(target))

    assert paths.cache_home == target.resolve()
    assert paths.hub_cache == target.resolve() / "hub"
    assert paths.xet_cache == target.resolve() / "xet"
    assert (target / "hub").is_dir()
    assert hf_cache_settings.get_hf_cache_paths().source == "default"


def test_download_dir_equal_to_models_folder_is_the_default(settings_store, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    hf_cache_settings.set_hf_cache_home(str(models))

    assert hf_cache_settings.download_cache_paths(str(models)) == (
        hf_cache_settings.get_hf_cache_paths()
    )


def test_download_dir_is_validated_like_the_models_folder(settings_store, tmp_path):
    with pytest.raises(ValueError, match = "absolute"):
        hf_cache_settings.download_cache_paths("relative/models")
    with pytest.raises(ValueError, match = "parent folder"):
        hf_cache_settings.download_cache_paths(str(tmp_path / "missing" / "models"))


def test_remember_cache_home_is_idempotent_and_scanned(settings_store, tmp_path):
    target = tmp_path / "drive-d"
    target.mkdir()

    hf_cache_settings.remember_cache_home(target)
    hf_cache_settings.remember_cache_home(target)

    assert settings_store[hf_cache_settings.DOWNLOAD_HOMES_SETTING_KEY] == [str(target.resolve())]
    assert target.resolve() / "hub" in hf_cache_settings.known_hf_hub_caches()


def test_download_folder_is_known_even_with_a_full_models_folder_history(settings_store, tmp_path):
    full = [str(tmp_path / f"old-{i}") for i in range(hf_cache_settings.MAX_CACHE_HISTORY)]
    settings_store[hf_cache_settings.CACHE_HISTORY_SETTING_KEY] = list(full)
    target = tmp_path / "new"
    target.mkdir()

    hf_cache_settings.remember_cache_home(target)

    assert settings_store[hf_cache_settings.CACHE_HISTORY_SETTING_KEY] == full
    assert target.resolve() / "hub" in hf_cache_settings.known_hf_hub_caches()


def _start(monkeypatch, body):
    calls = {"claim": [], "spawn": [], "scan_folders": []}

    class _Registry:
        def claim(self, key, transport, **kwargs):
            calls["claim"].append(kwargs)
            return True, "running"

        def current_generation(self, _key):
            return 1

        def peer_blob_hashes(self, _key):
            return frozenset()

        def get_job_metadata(self, _key):
            return None

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(downloads, "resolve_cached_repo_id_case", lambda repo_id, **_k: repo_id)
    monkeypatch.setattr(downloads, "_load_in_flight", lambda _repo_id: False)
    monkeypatch.setattr(
        downloads.download_lifecycle, "largest_download_file_bytes", lambda *a, **k: None
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "resolve_requested_use_xet",
        lambda *a, **k: (False, "test"),
    )
    for name in ("record_download_account", "require_live_account"):
        monkeypatch.setattr(downloads.download_lifecycle, name, lambda *a, **k: None)

    def _launch(_registry, _key, *, spawn, **_kwargs):
        spawn()
        return "running"

    monkeypatch.setattr(downloads.download_lifecycle, "launch_worker", _launch)
    monkeypatch.setattr(
        downloads,
        "_spawn_download_worker",
        lambda *a, **kwargs: calls["spawn"].append(kwargs["cache_env"]),
    )
    monkeypatch.setattr(
        local_inventory,
        "add_scan_folder_response",
        lambda path, recursive = None: calls["scan_folders"].append(path) or {"path": path},
    )
    result = asyncio.run(downloads.download_model_response(body))
    return result, calls


def _body(**over):
    body = {
        "repo_id": "Org/Model",
        "gguf_variant": None,
        "use_xet": False,
        "scope_id": None,
        "files": None,
        "download_dir": None,
    }
    body.update(over)
    return SimpleNamespace(**body)


def test_download_without_download_dir_keeps_the_models_folder(settings_store, monkeypatch):
    active = hf_cache_settings.get_hf_cache_paths()

    result, calls = _start(monkeypatch, _body())

    assert result["accepted"] is True
    assert calls["claim"][0]["hub_cache"] == str(active.hub_cache)
    assert calls["spawn"][0]["HF_HUB_CACHE"] == str(active.hub_cache)
    assert calls["scan_folders"] == []
    assert hf_cache_settings.DOWNLOAD_HOMES_SETTING_KEY not in settings_store


def test_download_to_folder_routes_worker_and_registers_folder(
    settings_store, monkeypatch, tmp_path
):
    target = tmp_path / "drive-e"
    target.mkdir()
    hub = target.resolve() / "hub"

    for _ in range(2):
        result, calls = _start(monkeypatch, _body(download_dir = str(target)))
        assert result["accepted"] is True
        assert calls["claim"][0]["hub_cache"] == str(hub)
        assert calls["claim"][0]["xet_cache"] == str(target.resolve() / "xet")
        assert calls["spawn"][0]["HF_HUB_CACHE"] == str(hub)
        assert calls["scan_folders"] == [str(target.resolve())]

    assert settings_store[hf_cache_settings.DOWNLOAD_HOMES_SETTING_KEY] == [str(target.resolve())]
    assert hub in hf_cache_settings.known_hf_hub_caches()
    assert hf_cache_settings.get_hf_cache_paths().source == "default"


def test_bad_download_dir_is_a_400_and_starts_nothing(settings_store, monkeypatch, tmp_path):
    with pytest.raises(HTTPException) as exc:
        _start(monkeypatch, _body(download_dir = str(tmp_path / "missing" / "models")))
    assert exc.value.status_code == 400


def test_managed_account_cannot_choose_a_download_folder(settings_store, monkeypatch, tmp_path):
    monkeypatch.setattr(downloads.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(downloads.account_access, "account_hf_token", lambda token: token)
    monkeypatch.setattr(downloads.account_access, "authorize_download", lambda *a, **k: None)
    with pytest.raises(HTTPException) as exc:
        downloads._download_cache_paths(str(tmp_path))
    assert exc.value.status_code == 403
