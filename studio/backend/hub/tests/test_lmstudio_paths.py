# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks the hub re-export of local-model discovery still resolves under the hub conftest stubs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub.utils import paths


BOM_UTF8 = b"\xef\xbb\xbf"


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("OLLAMA_MODELS", raising = False)
    return home


@pytest.mark.parametrize("bom", [False, True], ids = ["no_bom", "utf8_bom"])
def test_lmstudio_model_dirs_reads_a_bom_prefixed_settings_file(fake_home, bom):
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    settings = fake_home / ".lmstudio" / "settings.json"
    settings.parent.mkdir(parents = True)
    body = json.dumps({"downloadsFolder": str(downloads)}).encode("utf-8")
    settings.write_bytes((BOM_UTF8 if bom else b"") + body)

    assert paths.lmstudio_model_dirs() == [downloads]


def test_the_hub_discovery_names_resolve_under_the_hub_test_stubs():
    for name in ("lmstudio_model_dirs", "ollama_model_dirs", "well_known_model_dirs"):
        assert callable(getattr(paths, name)), name
