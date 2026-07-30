# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Tauri repair helpers from mixing package versions."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_tauri_never_overlays_install_python_stack() -> None:
    config = json.loads((REPO / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8"))
    resources = config["bundle"]["resources"]
    assert not any("install_python_stack.py" in path for item in resources.items() for path in item)

    installer = (REPO / "install.ps1").read_text(encoding = "utf-8")
    assert "Overlay Tauri-bundled studio fixes" not in installer
    assert (
        '"install_python_stack.py" = "Lib\\site-packages\\studio\\install_python_stack.py"'
        not in installer
    )
