# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json

from hub.utils import paths


def test_lmstudio_model_dirs_accepts_utf8_bom(tmp_path, monkeypatch):
    home = tmp_path / "home"
    downloads = home / "lmstudio-models"
    downloads.mkdir(parents = True)

    settings_path = home / ".lmstudio" / "settings.json"
    settings_path.parent.mkdir(parents = True)
    settings_path.write_text(
        json.dumps({"downloadsFolder": str(downloads)}),
        encoding = "utf-8-sig",
    )

    monkeypatch.setattr(paths.Path, "home", lambda: home)

    assert downloads in paths.lmstudio_model_dirs()
