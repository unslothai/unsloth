# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF_HOME must seed HF_HUB_CACHE and HF_XET_CACHE, since both download workers follow HF_HUB_CACHE."""

import contextlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _shared_setup_1():
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    return os


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_STORAGE_ROOTS_PATH = Path(__file__).resolve().parent.parent / "utils/paths/storage_roots.py"


@pytest.fixture(autouse = True)
def _isolate_studio_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    # _setup_cache_env writes os.environ directly, so a portable test would leak these forward.
    for key in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "TORCH_HOME"):
        monkeypatch.delenv(key, raising = False)


@pytest.fixture(autouse = True)
def _restore_hf_cache_settings_module():
    # Restore utils.hf_cache_settings, or a second module object splits settings from the code under test.
    import utils

    name = "utils.hf_cache_settings"
    saved = sys.modules.get(name)
    saved_attr = getattr(utils, "hf_cache_settings", None)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)
        if saved_attr is not None:
            utils.hf_cache_settings = saved_attr
        else:
            with contextlib.suppress(AttributeError):
                del utils.hf_cache_settings


def _load_storage_roots():
    # The cache resolver snapshots explicit env vars once per process.
    sys.modules.pop("utils.hf_cache_settings", None)
    spec = importlib.util.spec_from_file_location("storage_roots_under_test", _STORAGE_ROOTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_hf_env(monkeypatch):
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_ASSETS_CACHE",
        "HF_MODULES_CACHE",
    ):
        monkeypatch.delenv(key, raising = False)


def _portable_install(monkeypatch, tmp_path):
    """Turn the fixture's studio home into a portable one and return its root."""
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_PORTABLE", raising = False)
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    return master


def test_custom_hf_home_seeds_hub_and_xet(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    custom = tmp_path / "shared" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(custom))
    os = _shared_setup_1()

    assert os.environ["HF_HUB_CACHE"] == str(custom / "hub")
    assert os.environ["HF_XET_CACHE"] == str(custom / "xet")


def test_default_when_hf_home_unset(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    os = _shared_setup_1()

    expected = tmp_path / "xdg" / "huggingface"
    assert os.environ["HF_HUB_CACHE"] == str(expected / "hub")


def test_explicit_hub_cache_is_not_overridden(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    explicit = tmp_path / "explicit" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(explicit))
    os = _shared_setup_1()

    assert os.environ["HF_HUB_CACHE"] == str(explicit)


def test_legacy_huggingface_hub_cache_alias_is_honored(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    legacy = tmp_path / "legacy" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(legacy))
    os = _shared_setup_1()

    assert os.environ["HF_HUB_CACHE"] == str(legacy)


def test_whitespace_hf_home_falls_back_to_default(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", "   ")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    os = _shared_setup_1()

    assert os.environ["HF_HOME"] == str(tmp_path / "xdg" / "huggingface")
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "xdg" / "huggingface" / "hub")


def test_explicit_hf_home_keeps_the_datasets_and_assets_caches(monkeypatch, tmp_path):
    # One user-named HF_HOME means one cache: assets and datasets must derive from it.
    _clear_hf_env(monkeypatch)
    master = _portable_install(monkeypatch, tmp_path)
    chosen = tmp_path / "bigdisk" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(chosen / "hub")
    assert os.environ["HF_XET_CACHE"] == str(chosen / "xet")
    assert "HF_DATASETS_CACHE" not in os.environ
    assert "HF_ASSETS_CACHE" not in os.environ
    assert "HF_MODULES_CACHE" not in os.environ
    assert os.environ["TORCH_HOME"].startswith(str(master))


def test_a_dedicated_cache_var_still_outranks_an_explicit_hf_home(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    _portable_install(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "bigdisk" / "huggingface"))
    mine = tmp_path / "mine" / "datasets"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(mine))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_DATASETS_CACHE"] == str(mine)


def test_portable_mode_without_an_explicit_hf_home_still_contains_them(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    master = _portable_install(monkeypatch, tmp_path)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_DATASETS_CACHE"].startswith(str(master))
    assert os.environ["HF_ASSETS_CACHE"].startswith(str(master))
    # transformers derives HF_MODULES_CACHE from HF_HOME and adds it to sys.path.
    assert os.environ["HF_MODULES_CACHE"].startswith(str(master))


def test_a_normal_install_leaves_the_modules_cache_where_transformers_looks(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_PORTABLE", raising = False)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert "HF_MODULES_CACHE" not in os.environ


def test_an_explicit_modules_cache_outranks_the_portable_default(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _portable_install(monkeypatch, tmp_path)
    mine = tmp_path / "mine" / "modules"
    monkeypatch.setenv("HF_MODULES_CACHE", str(mine))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_MODULES_CACHE"] == str(mine)


@pytest.mark.parametrize("hub_variable", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
def test_a_hub_only_override_still_contains_the_xet_cache(monkeypatch, tmp_path, hub_variable):
    # huggingface_hub derives HF_XET_CACHE from HF_HOME, never from HF_HUB_CACHE.
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home" / ".cache"))
    master = _portable_install(monkeypatch, tmp_path)
    chosen = tmp_path / "bigdisk" / "hub"
    monkeypatch.setenv(hub_variable, str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(chosen)
    assert os.environ["HF_XET_CACHE"].startswith(str(master))


def test_an_explicit_xet_cache_outranks_the_portable_default(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _portable_install(monkeypatch, tmp_path)
    mine = tmp_path / "bigdisk" / "xet"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "bigdisk" / "hub"))
    monkeypatch.setenv("HF_XET_CACHE", str(mine))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_XET_CACHE"] == str(mine)


def test_a_normal_install_leaves_the_xet_cache_in_the_host_home(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "bigdisk" / "hub"))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "xdg" / "huggingface" / "xet")


def test_the_libraries_really_derive_these_caches_from_hf_home(monkeypatch, tmp_path):
    # Ask a fresh interpreter: huggingface_hub and datasets snapshot their constants at import.
    pytest.importorskip("datasets")
    _clear_hf_env(monkeypatch)
    _portable_install(monkeypatch, tmp_path)
    chosen = tmp_path / "bigdisk" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import json
    import os
    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, huggingface_hub.constants as c, datasets.config as d;"
            "print(json.dumps({'assets': c.HF_ASSETS_CACHE, 'hub': c.HF_HUB_CACHE,"
            "'datasets': str(d.HF_DATASETS_CACHE)}))",
        ],
        env = dict(os.environ),
        capture_output = True,
        text = True,
        check = True,
    )
    result = json.loads(probe.stdout.strip().splitlines()[-1])

    assert Path(result["assets"]) == chosen / "assets"
    assert Path(result["datasets"]) == chosen / "datasets"
    assert Path(result["hub"]) == chosen / "hub"


def test_unwritable_hf_home_does_not_crash(monkeypatch, tmp_path):
    # HF_HOME under a regular file: mkdir fails but startup must not crash.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    unwritable = blocker / "hf"
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(unwritable))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(unwritable / "hub")
