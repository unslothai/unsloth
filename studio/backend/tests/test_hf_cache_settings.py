# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.services.models.common import _local_model_info
from utils import hf_cache_settings
from utils import native_path_leases


@pytest.fixture()
def settings_store(monkeypatch, tmp_path):
    store = {}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings",
        lambda values: store.update(values) or values,
    )
    return store


def test_studio_cache_switch_is_live_and_keeps_history(settings_store, tmp_path):
    first = tmp_path / "external-a" / "huggingface"
    second = tmp_path / "external-b" / "huggingface"
    first.parent.mkdir()
    second.parent.mkdir()

    selected = hf_cache_settings.set_hf_cache_home(str(first))
    assert selected.hub_cache == first / "hub"
    assert selected.xet_cache == first / "xet"
    assert selected.child_env({}) == {
        "HF_HUB_CACHE": str(first / "hub"),
        "HF_XET_CACHE": str(first / "xet"),
    }

    hf_cache_settings.set_hf_cache_home(str(second))
    assert settings_store[hf_cache_settings.CACHE_HISTORY_SETTING_KEY] == [str(first)]
    assert first / "hub" in hf_cache_settings.known_hf_hub_caches()

    reset = hf_cache_settings.set_hf_cache_home(None)
    assert reset.source == "default"
    assert second in hf_cache_settings.known_hf_cache_homes()


def test_environment_cache_is_read_only(monkeypatch, tmp_path):
    custom = tmp_path / "managed"
    monkeypatch.setattr(
        hf_cache_settings,
        "_EXPLICIT_CACHE_ENV",
        {"HF_HOME": str(custom)},
    )
    paths = hf_cache_settings.get_hf_cache_paths()
    assert paths.source == "environment"
    assert paths.editable is False
    assert paths.hub_cache == custom / "hub"
    with pytest.raises(RuntimeError, match = "environment variable"):
        hf_cache_settings.set_hf_cache_home(str(tmp_path / "other"))


def test_worker_environment_is_applied_before_import(monkeypatch, tmp_path):
    hub = str(tmp_path / "hub")
    xet = str(tmp_path / "xet")
    observed = {}

    class Module:
        @staticmethod
        def run():
            import os

            return os.environ["HF_HUB_CACHE"], os.environ["HF_XET_CACHE"]

    def fake_import(name):
        import os

        observed["name"] = name
        observed["hub"] = os.environ.get("HF_HUB_CACHE")
        return Module

    monkeypatch.setattr(native_path_leases.importlib, "import_module", fake_import)
    result = native_path_leases.run_without_native_path_secret(
        "fake.worker",
        "run",
        {"HF_HUB_CACHE": hub, "HF_XET_CACHE": xet},
    )
    assert observed == {"name": "fake.worker", "hub": hub}
    assert result == (hub, xet)


def test_spawn_environment_is_applied_then_restored(monkeypatch, tmp_path):
    hub = str(tmp_path / "hub")
    xet = str(tmp_path / "xet")
    monkeypatch.setenv("HF_HUB_CACHE", "parent-hub")
    monkeypatch.delenv("HF_XET_CACHE", raising = False)

    with hf_cache_settings.child_environment_for_spawn(
        {"HF_HUB_CACHE": hub, "HF_XET_CACHE": xet}
    ):
        import os

        assert os.environ["HF_HUB_CACHE"] == hub
        assert os.environ["HF_XET_CACHE"] == xet

    assert os.environ["HF_HUB_CACHE"] == "parent-hub"
    assert "HF_XET_CACHE" not in os.environ


def test_inactive_cache_model_loads_from_snapshot_path(tmp_path):
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    row = _local_model_info(
        scan_path = snapshot,
        load_path = snapshot,
        source = "hf_cache",
        model_format = "safetensors",
        model_id = "org/model",
        active_cache = False,
    )
    assert row.model_id == "org/model"
    assert row.active_cache is False
    assert row.load_id == str(snapshot)
