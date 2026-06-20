# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_setup_cache_env() must seed HF_HUB_CACHE / HF_XET_CACHE from a user-set
HF_HOME, so models download to and load from the same custom location (issue
#5182). Both the Xet and HTTP-fallback download workers call snapshot_download
without a cache_dir, so they follow HF_HUB_CACHE; getting it right here fixes
detection and both transports at once.
"""

import importlib.util
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_STORAGE_ROOTS_PATH = Path(__file__).resolve().parent.parent / "utils/paths/storage_roots.py"


def _load_storage_roots():
    spec = importlib.util.spec_from_file_location("storage_roots_under_test", _STORAGE_ROOTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear_hf_env(monkeypatch):
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising = False)


def test_custom_hf_home_seeds_hub_and_xet(monkeypatch, tmp_path):
    sr = _load_storage_roots()
    _clear_hf_env(monkeypatch)
    custom = tmp_path / "shared" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(custom))

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(custom / "hub")
    assert os.environ["HF_XET_CACHE"] == str(custom / "xet")


def test_default_when_hf_home_unset(monkeypatch, tmp_path):
    sr = _load_storage_roots()
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    sr._setup_cache_env()

    import os

    expected = tmp_path / "xdg" / "huggingface"
    assert os.environ["HF_HUB_CACHE"] == str(expected / "hub")


def test_explicit_hub_cache_is_not_overridden(monkeypatch, tmp_path):
    sr = _load_storage_roots()
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    explicit = tmp_path / "explicit" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(explicit))

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(explicit)


def test_legacy_huggingface_hub_cache_alias_is_honored(monkeypatch, tmp_path):
    sr = _load_storage_roots()
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    legacy = tmp_path / "legacy" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(legacy))

    sr._setup_cache_env()

    import os

    assert os.environ["HF_HUB_CACHE"] == str(legacy)
