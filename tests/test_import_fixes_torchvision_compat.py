import importlib
import logging
import os
from pathlib import Path

import pytest

os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")

import_fixes_spec = importlib.util.spec_from_file_location(
    "unsloth.import_fixes",
    Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py",
)
import_fixes_module = importlib.util.module_from_spec(import_fixes_spec)
import_fixes_spec.loader.exec_module(import_fixes_module)


def _patch_versions(monkeypatch, torch_version, torchvision_version):
    version_map = {"torch": torch_version, "torchvision": torchvision_version}
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in version_map:
            return object()
        return original_find_spec(name, *args, **kwargs)

    def fake_importlib_version(package_name):
        return version_map[package_name]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(import_fixes_module, "importlib_version", fake_importlib_version)
    monkeypatch.setattr(import_fixes_module, "_is_custom_torch_build", lambda _: False)


def test_torchvision_mismatch_raises_for_stable_newer_torch(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SKIP_TORCHVISION_CHECK", raising=False)
    _patch_versions(monkeypatch, "2.10.0+cu130", "0.24.1")

    with pytest.raises(ImportError) as excinfo:
        import_fixes_module.torchvision_compatibility_check()

    assert "torchvision>=0.25.0" in str(excinfo.value)


def test_torchvision_check_respects_skip_env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")
    _patch_versions(monkeypatch, "2.10.0+cu130", "0.24.1")

    import_fixes_module.torchvision_compatibility_check()


def test_torchvision_prerelease_mismatch_warns(monkeypatch, caplog):
    monkeypatch.delenv("UNSLOTH_SKIP_TORCHVISION_CHECK", raising=False)
    _patch_versions(monkeypatch, "2.11.0a0", "0.25.1")

    caplog.set_level(logging.WARNING, logger=import_fixes_module.__name__)
    import_fixes_module.torchvision_compatibility_check()

    warning_messages = [record.message for record in caplog.records]
    assert any("pre-release build" in message for message in warning_messages)
