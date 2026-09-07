# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The hub entry point onto local-model discovery.

The behaviour itself is covered by tests/test_local_model_dir_discovery.py. This file
exists because hub/tests/ is a separate conftest tree that stubs pydantic, fastapi and
structlog, so it is the only place that proves the re-export still resolves once those
stubs are in play.
"""

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
