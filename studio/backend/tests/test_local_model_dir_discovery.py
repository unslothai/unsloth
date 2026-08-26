# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where local models live is one policy, not two.

``lmstudio_model_dirs`` / ``ollama_model_dirs`` / ``well_known_model_dirs`` used to be
implemented twice -- once in ``utils.paths.storage_roots`` for the model picker, once in
``hub.utils.paths`` for the folder browser and hub inventory. The copies drifted: only one
read ``~/.lmstudio/settings.json`` as utf-8-sig, so a BOM'd settings file put the user's
custom downloads folder in the picker and nowhere else, with nothing logged.
"""

import json
from pathlib import Path

import pytest

from hub.utils import paths as hub_paths
from utils.paths import storage_roots


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _write_settings(home: Path, downloads: Path, *, bom: bool) -> None:
    (home / ".lmstudio").mkdir(parents = True, exist_ok = True)
    payload = json.dumps({"downloadsFolder": str(downloads)}).encode("utf-8")
    (home / ".lmstudio" / "settings.json").write_bytes((b"\xef\xbb\xbf" if bom else b"") + payload)


def test_hub_and_picker_share_one_implementation():
    for name in ("lmstudio_model_dirs", "ollama_model_dirs", "well_known_model_dirs"):
        assert getattr(hub_paths, name) is getattr(storage_roots, name), name


@pytest.mark.parametrize("bom", [False, True])
def test_lmstudio_downloads_folder_is_found_with_or_without_a_bom(fake_home, bom):
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    _write_settings(fake_home, downloads, bom = bom)

    assert storage_roots.lmstudio_model_dirs() == [downloads]
    assert downloads.resolve() in storage_roots.well_known_model_dirs()


def test_unreadable_settings_are_logged_not_swallowed(fake_home, capsys):
    (fake_home / ".lmstudio").mkdir(parents = True)
    (fake_home / ".lmstudio" / "settings.json").write_text("{ not json", encoding = "utf-8")

    assert storage_roots.lmstudio_model_dirs() == []
    assert "Ignoring unreadable LM Studio settings" in capsys.readouterr().out


def test_ollama_env_override_is_honoured(fake_home, monkeypatch, tmp_path):
    models = tmp_path / "ollama-models"
    models.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(models))

    assert storage_roots.ollama_model_dirs() == [models]
    assert models.resolve() in storage_roots.well_known_model_dirs()
