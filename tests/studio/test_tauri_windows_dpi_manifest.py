# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows must be DPI aware before generate_context! loads the runtime icon."""

from pathlib import Path
import xml.etree.ElementTree as ET


TAURI = Path(__file__).resolve().parents[2] / "studio/src-tauri"
MANIFEST = TAURI / "windows/app-manifest.xml"
ASM = "{urn:schemas-microsoft-com:asm.v1}"
ASM3 = "{urn:schemas-microsoft-com:asm.v3}"


def test_application_manifest_is_embedded_by_tauri_build() -> None:
    build = (TAURI / "build.rs").read_text(encoding = "utf-8")
    assert 'include_str!("windows/app-manifest.xml")' in build
    assert ".app_manifest(" in build
    assert ".windows_attributes(" in build
    assert "tauri_build::try_build(" in build
    assert "cargo:rerun-if-changed=windows/app-manifest.xml" in build


def test_application_is_dpi_aware_before_any_startup_code() -> None:
    root = ET.parse(MANIFEST).getroot()
    settings = root.find(f"{ASM3}application/{ASM3}windowsSettings")
    assert settings is not None
    modern = settings.find("{http://schemas.microsoft.com/SMI/2016/WindowsSettings}dpiAwareness")
    legacy = settings.find("{http://schemas.microsoft.com/SMI/2005/WindowsSettings}dpiAware")
    assert modern is not None and modern.text == "PerMonitorV2,PerMonitor"
    assert legacy is not None and legacy.text == "true/pm"
    # Preserve Tauri's default Common Controls v6 activation context.
    controls = root.find(f"{ASM}dependency/{ASM}dependentAssembly/{ASM}assemblyIdentity")
    assert controls is not None
    assert controls.attrib["name"] == "Microsoft.Windows.Common-Controls"
    assert controls.attrib["version"] == "6.0.0.0"
    assert controls.attrib["publicKeyToken"] == "6595b64144ccf1df"
    assert controls.attrib["processorArchitecture"] == "*"
    assert controls.attrib["language"] == "*"
