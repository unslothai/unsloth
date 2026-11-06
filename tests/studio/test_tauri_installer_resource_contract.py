# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Tauri repair helpers from mixing package versions, and keep each desktop bundle carrying
only the installer it can actually run."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TAURI = REPO / "studio/src-tauri"


def _resources(config_name: str) -> dict:
    config = json.loads((TAURI / config_name).read_text(encoding = "utf-8"))
    return config.get("bundle", {}).get("resources", {})


def _bundled_resources(platform: str) -> dict:
    # Tauri merges tauri.<platform>.conf.json over tauri.conf.json for the target being built.
    merged = dict(_resources("tauri.conf.json"))
    merged.update(_resources(f"tauri.{platform}.conf.json"))
    return merged


def test_tauri_never_overlays_install_python_stack() -> None:
    for platform in ("windows", "linux", "macos"):
        resources = _bundled_resources(platform)
        assert not any(
            "install_python_stack.py" in path for item in resources.items() for path in item
        ), platform

    installer = (REPO / "install.ps1").read_text(encoding = "utf-8")
    assert "Overlay Tauri-bundled studio fixes" not in installer
    assert (
        '"install_python_stack.py" = "Lib\\site-packages\\studio\\install_python_stack.py"'
        not in installer
    )


def test_each_bundle_ships_only_the_installer_it_runs() -> None:
    # Each bundle ships only its platform's installer; the unused script tripped antivirus heuristics.
    assert _bundled_resources("windows") == {"../../install.ps1": "install.ps1"}
    assert _bundled_resources("linux") == {"../../install.sh": "install.sh"}
    assert _bundled_resources("macos") == {"../../install.sh": "install.sh"}


def test_no_installer_resource_leaks_through_the_shared_config() -> None:
    assert _resources("tauri.conf.json") == {}


def test_windows_upgrade_removes_the_installer_it_no_longer_ships() -> None:
    # NSIS deletes only manifest entries, so an upgrade would keep install.sh and break RMDir.
    hooks = (REPO / "studio/src-tauri/windows/hooks.nsh").read_text(encoding = "utf-8")
    for macro in ("NSIS_HOOK_PREINSTALL", "NSIS_HOOK_PREUNINSTALL"):
        assert f"!macro {macro}" in hooks, f"hooks.nsh must define {macro}"
        body = hooks.split(f"!macro {macro}", 1)[1].split("!macroend", 1)[0]
        assert (
            'Delete "$INSTDIR\\install.sh"' in body
        ), f"{macro} must remove the install.sh a pre-split release left behind"
