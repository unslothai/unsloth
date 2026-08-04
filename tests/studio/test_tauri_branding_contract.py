# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Desktop display-branding contracts."""

import json
from pathlib import Path
import struct


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend"
TAURI = REPO / "studio/src-tauri"


def read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def bmp_metadata(path: Path) -> tuple[int, int, int]:
    data = path.read_bytes()
    assert data[:2] == b"BM"
    width, height = struct.unpack_from("<ii", data, 18)
    bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
    return width, height, bits_per_pixel


def tiff_first_image_size(path: Path) -> tuple[int, int]:
    """Width and height of the first image in a TIFF, ignoring later hidpi pages."""
    data = path.read_bytes()
    assert data[:2] in (b"II", b"MM")
    order = "<" if data[:2] == b"II" else ">"

    ifd_offset = struct.unpack_from(order + "I", data, 4)[0]
    entry_count = struct.unpack_from(order + "H", data, ifd_offset)[0]

    sizes: dict[int, int] = {}
    for index in range(entry_count):
        entry = ifd_offset + 2 + index * 12
        tag, field_type = struct.unpack_from(order + "HH", data, entry)
        if tag in (256, 257):
            # tag 256 is ImageWidth and 257 is ImageLength, either SHORT or LONG
            sizes[tag] = struct.unpack_from(order + ("H" if field_type == 3 else "I"), data, entry + 8)[0]
    return sizes[256], sizes[257]


def test_desktop_display_name_and_compatibility_ids() -> None:
    config = json.loads(read(TAURI / "tauri.conf.json"))
    assert config["productName"] == "Unsloth"
    assert config["app"]["windows"][0]["title"] == "Unsloth"

    assert config["identifier"] == "ai.unsloth.studio"
    assert config["plugins"]["deep-link"]["desktop"]["schemes"] == ["unsloth"]
    assert config["plugins"]["updater"]["endpoints"] == [
        "https://github.com/unslothai/unsloth/releases/download/desktop-latest/latest.json"
    ]
    assert 'name = "unsloth-studio"' in read(TAURI / "Cargo.toml")


def test_desktop_package_transitions_preserve_legacy_installs() -> None:
    config = json.loads(read(TAURI / "tauri.conf.json"))
    deb = config["bundle"]["linux"]["deb"]
    for field in ("provides", "conflicts", "replaces"):
        assert deb[field] == ["unsloth-studio-desktop"]

    installer = read(TAURI / "windows/installer.nsi")
    assert '!define INSTALLIDENTITY "Unsloth Studio (Desktop)"' in installer
    assert "Uninstall\\${INSTALLIDENTITY}" in installer
    assert "${MANUKEY}\\${INSTALLIDENTITY}" in installer
    assert "$LOCALAPPDATA\\${INSTALLIDENTITY}" in installer

    assert 'StrCmp "$R0" "${PRODUCTNAME}" wix_name_match' in installer
    assert 'StrCmp "$R0" "${INSTALLIDENTITY}" 0 wix_loop' in installer
    assert '"$SMPROGRAMS\\${INSTALLIDENTITY}.lnk" "$INSTDIR\\$OldMainBinaryName"' in installer
    assert '"$DESKTOP\\${INSTALLIDENTITY}.lnk" "$INSTDIR\\$OldMainBinaryName"' in installer
    assert 'Rename "$SMPROGRAMS\\${INSTALLIDENTITY}.lnk"' in installer
    assert 'Rename "$DESKTOP\\${INSTALLIDENTITY}.lnk"' in installer


def test_desktop_artwork_uses_plain_unsloth_lockups() -> None:
    config = json.loads(read(TAURI / "tauri.conf.json"))
    nsis = config["bundle"]["windows"]["nsis"]
    assert nsis["headerImage"] == "./windows/branding/nsis-header.bmp"
    assert nsis["sidebarImage"] == "./windows/branding/nsis-sidebar.bmp"

    for component in ("startup-screen.tsx", "update-screen.tsx"):
        source = read(FRONTEND / "src/components/tauri" / component)
        assert "/sticker.png" in source
        assert "fontFamily: '\"Hellix\", sans-serif'" in source
        assert "unsloth" in source
        assert "/studio.png" not in source

    titlebar = read(FRONTEND / "src/components/tauri/window-titlebar.tsx")
    assert "/rounded-512.png" in titlebar
    assert "Unsloth" in titlebar
    assert not (FRONTEND / "public/studio.png").exists()

    branding = TAURI / "windows/branding"
    assert bmp_metadata(branding / "nsis-header.bmp") == (300, 114, 24)
    assert bmp_metadata(branding / "nsis-sidebar.bmp") == (328, 628, 24)


def test_dmg_install_window_matches_its_background_art() -> None:
    dmg = json.loads(read(TAURI / "tauri.macos.conf.json"))["bundle"]["macOS"]["dmg"]
    assert dmg["background"] == "./dmg/background.tiff"

    # Finder lays the background out from the same origin it uses for icon
    # coordinates, so the base page has to match the configured window size or
    # the artwork drifts out from under the app and Applications icons.
    window = (dmg["windowSize"]["width"], dmg["windowSize"]["height"])
    assert window == (660, 400)
    assert tiff_first_image_size(TAURI / "dmg/background.tiff") == window

    assert dmg["appPosition"] == {"x": 180, "y": 170}
    assert dmg["applicationFolderPosition"] == {"x": 480, "y": 170}


def test_desktop_release_asset_names_are_human_readable() -> None:
    workflow = read(REPO / ".github/workflows/release-desktop.yml")
    assert "re.sub(r'[^0-9A-Za-z]+', '_', app_version).strip('_')" in workflow

    assert "base_name = f'Unsloth-Desktop-{os.environ[\"ASSET_VERSION\"]}'" in workflow
    expected_suffixes = {
        "MacOS.dmg",
        "ARM64.app.tar.gz",
        "ARM64.app.tar.gz.sig",
        "Ubuntu.AppImage",
        "Ubuntu.AppImage.sig",
        "Linux.deb",
        "Windows.exe",
        "Windows.exe.sig",
    }
    for suffix in expected_suffixes:
        assert f"f'{{base_name}}-{suffix}'" in workflow


def test_desktop_surfaces_do_not_restore_studio_branding() -> None:
    display_sources = [
        TAURI / "Info.plist",
        TAURI / "capabilities/default.json",
        TAURI / "src/main.rs",
        TAURI / "src/process.rs",
        TAURI / "src/diagnostics/report.rs",
        TAURI / "src/diagnostics/phase_log.rs",
        TAURI / "windows/sign-with-trusted-signing.ps1",
        REPO / ".github/workflows/release-desktop.yml",
        FRONTEND / "index.html",
        *sorted((FRONTEND / "src").rglob("*.ts")),
        *sorted((FRONTEND / "src").rglob("*.tsx")),
    ]
    offenders = [
        str(path.relative_to(REPO)) for path in display_sources if "Unsloth Studio" in read(path)
    ]
    assert offenders == []

    workflow = read(REPO / ".github/workflows/release-desktop.yml")
    assert "Desktop app for Unsloth." in workflow
    assert '--title "Unsloth ${STUDIO_VERSION}"' in workflow
    assert '--title "Unsloth Desktop updater channel"' in workflow
