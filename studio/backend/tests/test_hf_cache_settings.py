# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
import threading
import time
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


def test_explicit_hub_cache_is_the_displayed_location(monkeypatch, tmp_path):
    custom_hub = tmp_path / "models-cache"
    custom_hub.mkdir()
    monkeypatch.setattr(
        hf_cache_settings,
        "_EXPLICIT_CACHE_ENV",
        {"HF_HUB_CACHE": str(custom_hub)},
    )

    paths = hf_cache_settings.get_hf_cache_paths()
    status = hf_cache_settings.cache_status(paths)

    assert paths.cache_home == custom_hub
    assert paths.hub_cache == custom_hub
    assert status["cache_home"] == str(custom_hub)
    assert status["available"] is True
    assert custom_hub / "hub" not in hf_cache_settings.known_hf_hub_caches()


def test_explicit_hub_cache_display_wins_over_hf_home(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf-home"
    custom_hub = tmp_path / "other-disk" / "models-cache"
    hf_home.mkdir()
    custom_hub.mkdir(parents = True)
    monkeypatch.setattr(
        hf_cache_settings,
        "_EXPLICIT_CACHE_ENV",
        {"HF_HOME": str(hf_home), "HF_HUB_CACHE": str(custom_hub)},
    )

    paths = hf_cache_settings.get_hf_cache_paths()

    assert paths.cache_home == custom_hub
    assert paths.hub_cache == custom_hub
    assert paths.xet_cache == hf_home / "xet"
    assert custom_hub / "hub" not in hf_cache_settings.known_hf_hub_caches()
    assert hf_home / "hub" in hf_cache_settings.known_hf_hub_caches()


def test_xet_only_override_keeps_model_cache_display_on_hf_home(monkeypatch, tmp_path):
    xet_cache = tmp_path / "chunks"
    monkeypatch.setattr(
        hf_cache_settings,
        "_EXPLICIT_CACHE_ENV",
        {"HF_XET_CACHE": str(xet_cache)},
    )

    paths = hf_cache_settings.get_hf_cache_paths()

    assert paths.cache_home == hf_cache_settings._default_cache_home()
    assert paths.hub_cache == hf_cache_settings._default_cache_home() / "hub"
    assert paths.xet_cache == xet_cache


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

    with hf_cache_settings.child_environment_for_spawn({"HF_HUB_CACHE": hub, "HF_XET_CACHE": xet}):
        import os
        assert os.environ["HF_HUB_CACHE"] == hub
        assert os.environ["HF_XET_CACHE"] == xet

    assert os.environ["HF_HUB_CACHE"] == "parent-hub"
    assert "HF_XET_CACHE" not in os.environ


def test_spawn_environment_supports_nested_contexts(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "parent")

    with hf_cache_settings.child_environment_for_spawn({"HF_HUB_CACHE": "outer"}):
        assert os.environ["HF_HUB_CACHE"] == "outer"
        with hf_cache_settings.child_environment_for_spawn({"HF_HUB_CACHE": "inner"}):
            assert os.environ["HF_HUB_CACHE"] == "inner"
        assert os.environ["HF_HUB_CACHE"] == "outer"

    assert os.environ["HF_HUB_CACHE"] == "parent"


def test_spawn_environment_serializes_threads(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "parent")
    first_entered = threading.Event()
    release_first = threading.Event()
    observations: list[tuple[str, str]] = []

    def first():
        with hf_cache_settings.child_environment_for_spawn({"HF_HUB_CACHE": "first"}):
            observations.append(("first", os.environ["HF_HUB_CACHE"]))
            first_entered.set()
            assert release_first.wait(timeout = 2)

    def second():
        assert first_entered.wait(timeout = 2)
        with hf_cache_settings.child_environment_for_spawn({"HF_HUB_CACHE": "second"}):
            observations.append(("second", os.environ["HF_HUB_CACHE"]))

    first_thread = threading.Thread(target = first)
    second_thread = threading.Thread(target = second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout = 2)
    time.sleep(0.02)
    assert observations == [("first", "first")]
    release_first.set()
    first_thread.join(timeout = 2)
    second_thread.join(timeout = 2)

    assert observations == [("first", "first"), ("second", "second")]
    assert os.environ["HF_HUB_CACHE"] == "parent"


def test_cache_switch_invalidates_inventory(settings_store, tmp_path, monkeypatch):
    invalidations = []
    monkeypatch.setattr(
        "hub.utils.inventory_scan.invalidate_hf_cache_scans",
        lambda: invalidations.append(True),
    )
    selected = tmp_path / "external" / "huggingface"
    selected.parent.mkdir()

    hf_cache_settings.set_hf_cache_home(str(selected))

    assert invalidations == [True]


def test_cache_validation_write_tests_hub_and_xet(settings_store, tmp_path, monkeypatch):
    selected = tmp_path / "external" / "huggingface"
    selected.parent.mkdir()
    tested = []
    real_named_temporary_file = hf_cache_settings.tempfile.NamedTemporaryFile

    def recording_write_test(*args, **kwargs):
        tested.append(Path(kwargs["dir"]))
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        hf_cache_settings.tempfile,
        "NamedTemporaryFile",
        recording_write_test,
    )

    hf_cache_settings.set_hf_cache_home(str(selected))

    assert tested == [selected / "hub", selected / "xet"]


def test_cache_validation_rejects_unwritable_child(settings_store, tmp_path, monkeypatch):
    selected = tmp_path / "external" / "huggingface"
    selected.parent.mkdir()

    def reject_hub(*args, **kwargs):
        if Path(kwargs["dir"]).name == "hub":
            raise PermissionError("read-only")
        raise AssertionError("xet should not be tested after hub fails")

    monkeypatch.setattr(hf_cache_settings.tempfile, "NamedTemporaryFile", reject_hub)

    with pytest.raises(ValueError, match = "permission"):
        hf_cache_settings.set_hf_cache_home(str(selected))


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
