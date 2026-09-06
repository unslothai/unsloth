# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import subprocess
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


def test_xet_only_override_keeps_model_cache_editable(settings_store, monkeypatch, tmp_path):
    xet_cache = tmp_path / "chunks"
    stored = tmp_path / "stored-cache"
    settings_store[hf_cache_settings.CACHE_HOME_SETTING_KEY] = str(stored)
    monkeypatch.setattr(
        hf_cache_settings,
        "_EXPLICIT_CACHE_ENV",
        {"HF_XET_CACHE": str(xet_cache)},
    )

    paths = hf_cache_settings.get_hf_cache_paths()

    assert paths.cache_home == stored
    assert paths.hub_cache == stored / "hub"
    assert paths.xet_cache == xet_cache
    assert paths.editable is True

    selected = tmp_path / "selected-cache"
    selected.parent.mkdir(exist_ok = True)
    updated = hf_cache_settings.set_hf_cache_home(str(selected))
    assert updated.hub_cache == selected / "hub"
    assert updated.xet_cache == xet_cache


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


def test_diffusion_cache_root_follows_a_live_switch(settings_store, tmp_path):
    # The image/video backends used huggingface_hub's import-time HF_HUB_CACHE constant, which set_hf_cache_home does not
    # update, so the download wrote to the new root while progress counted the old one and a load could split across both.
    import core.inference.diffusion as diffusion

    moved = tmp_path / "external-c" / "huggingface"
    # Write the setting straight into the store: set_hf_cache_home's folder validation is not under test and it rejects the pytest tmp root on macOS.
    settings_store[hf_cache_settings.CACHE_HOME_SETTING_KEY] = str(moved)

    assert diffusion.hub_cache_dir() == str(moved / "hub")
    assert diffusion.DiffusionBackend._hub_cache_repo_dir("org/model") == (
        moved / "hub" / "models--org--model"
    )


def test_diffusion_loader_calls_pin_the_cache_dir():
    # Every from_pretrained / from_single_file must carry cache_dir, else diffusers resolves it through the stale constant.
    for rel in ("core/inference/diffusion.py", "core/inference/video.py"):
        source = (Path(_BACKEND_DIR) / rel).read_text(encoding = "utf-8")
        for call in ("from_pretrained(", "from_single_file("):
            for index, line in enumerate(source.splitlines(), start = 1):
                if not line.strip().startswith(("pipe = ", "transformer = ", "cn_model = ")):
                    continue
                if call not in line:
                    continue
                window = "\n".join(source.splitlines()[index - 1 : index + 8])
                assert (
                    "cache_dir" in window or "kwargs" in window
                ), f"{rel}:{index} calls {call} without a pinned cache_dir"


# _stored_cache_home skips the database read when nothing uses one, so `unsloth train` does not
# build a studio.db on a machine that never opened Studio. Only a positively observed absence may
# license that skip; every other outcome falls through to the read. Driven in a subprocess: the
# skip needs storage.studio_db absent from sys.modules, which pytest has already imported.
_GUARD_PROBE = """
import json, os, sys

sys.path.insert(0, os.environ["FAKE_STORAGE"])
sys.path.insert(1, os.environ["BACKEND_DIR"])
from utils import hf_cache_settings

assert "storage.studio_db" not in sys.modules, "the probe must exercise the skip"
stored = hf_cache_settings._stored_cache_home()
print(json.dumps({"stored": None if stored is None else str(stored)}))
"""

_FAKE_STUDIO_DB = """
import os
from pathlib import Path


def get_app_setting(key, fallback = None):
    Path(os.environ["READ_WITNESS"]).write_text("read", encoding = "utf-8")
    if key == "hugging_face_cache_home":
        return os.environ["STORED_CACHE_HOME"]
    return fallback
"""


def _run_guard_probe(tmp_path, studio_home: Path, stored: Path) -> tuple[str | None, bool]:
    """Return (_stored_cache_home() answer, whether the database was read)."""
    fake = tmp_path / "fake_storage"
    (fake / "storage").mkdir(parents = True, exist_ok = True)
    (fake / "storage" / "__init__.py").write_text("", encoding = "utf-8")
    (fake / "storage" / "studio_db.py").write_text(_FAKE_STUDIO_DB, encoding = "utf-8")
    witness = tmp_path / "read_witness"
    witness.unlink(missing_ok = True)

    environment = dict(os.environ)
    environment.update(
        BACKEND_DIR = _BACKEND_DIR,
        FAKE_STORAGE = str(fake),
        READ_WITNESS = str(witness),
        STORED_CACHE_HOME = str(stored),
        UNSLOTH_STUDIO_HOME = str(studio_home),
        PYTHONPATH = "",
    )
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE"):
        environment.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-c", _GUARD_PROBE],
        capture_output = True,
        text = True,
        env = environment,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])["stored"], witness.exists()


def test_absent_studio_db_skips_the_database_read(tmp_path):
    # The skip 912024e84 added must survive the tightening below: a Studio root with no studio.db
    # answers None WITHOUT a connection, which is what stops the CLI creating a 250 KB database.
    studio_home = tmp_path / "root" / "studio"
    studio_home.mkdir(parents = True)

    answer, was_read = _run_guard_probe(tmp_path, studio_home, tmp_path / "chosen")

    assert answer is None
    assert not was_read, "an absent database must not be opened"


@pytest.mark.parametrize("fixture", ["not_a_directory", "symlink_loop", "unreadable_parent"])
def test_uninspectable_studio_db_keeps_the_stored_cache_home(tmp_path, fixture):
    # Path.exists reports ENOTDIR and ELOOP as absence on every release we support, and from 3.14
    # swallows EACCES too. Treating any of them as "no database" discards the cache home the user
    # chose in Settings and re-routes downloads to the default root.
    chosen = tmp_path / "chosen"
    studio_home = tmp_path / "root" / "studio"
    if fixture == "not_a_directory":
        studio_home.parent.mkdir(parents = True)
        studio_home.write_text("", encoding = "utf-8")
    elif fixture == "symlink_loop":
        studio_home.parent.mkdir(parents = True)
        studio_home.symlink_to(studio_home)
    else:
        studio_home.mkdir(parents = True)
        (studio_home / "studio.db").write_bytes(b"")
        os.chmod(studio_home, 0o000)

    try:
        answer, was_read = _run_guard_probe(tmp_path, studio_home, chosen)
    finally:
        if fixture == "unreadable_parent":
            os.chmod(studio_home, 0o755)

    # unreadable_parent is the case 3.14 newly breaks: below it Path.exists still RAISED, which
    # the outer handler already turned into this fall-through. The other two hold on every release.
    assert was_read, "a database we could not inspect must still be read"
    assert answer == str(chosen)
