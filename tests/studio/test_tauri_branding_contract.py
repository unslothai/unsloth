# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Desktop display-branding contracts."""

import importlib.util
import json
from pathlib import Path
import struct

import pytest


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
            sizes[tag] = struct.unpack_from(
                order + ("H" if field_type == 3 else "I"), data, entry + 8
            )[0]
    return sizes[256], sizes[257]


def test_desktop_display_name_and_compatibility_ids() -> None:
    config = json.loads(read(TAURI / "tauri.conf.json"))
    assert config["productName"] == "Unsloth"
    assert config["app"]["windows"][0]["title"] == "Unsloth"

    assert config["identifier"] == "ai.unsloth.studio"
    assert config["plugins"]["deep-link"]["desktop"]["schemes"] == ["unsloth"]
    assert config["plugins"]["updater"]["endpoints"] == [
        "https://github.com/unslothai/unsloth/releases/latest/download/latest.json"
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

    sidebar = read(FRONTEND / "src/components/app-sidebar.tsx")
    assert "/circle-logo-small.png" in sidebar
    assert "unsloth" in sidebar

    assert 'chatDisabled && "pointer-events-none opacity-50"' not in sidebar
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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dmg_background_art_is_what_its_renderer_produces() -> None:
    """The checked-in TIFF is generated, so it has to track its own script."""
    np = pytest.importorskip("numpy")
    ImageSequence = pytest.importorskip("PIL.ImageSequence")
    from PIL import Image

    renderer = load_module(REPO / "scripts/make_dmg_background.py")
    image = renderer.build()
    expected = [
        image.resize((renderer.WIN_W, renderer.WIN_H), Image.LANCZOS).convert("RGB"),
        image.convert("RGB"),
    ]

    # the iterator seeks one shared handle, so each page is copied off it
    tiff = Image.open(TAURI / "dmg/background.tiff")
    pages = [page.convert("RGB") for page in ImageSequence.Iterator(tiff)]
    assert [page.size for page in pages] == [page.size for page in expected]

    # a tolerance, not equality, so no one Pillow build is baked in. a stale asset is far worse
    for page, reference in zip(pages, expected):
        drift = np.abs(np.asarray(page, dtype = np.int16) - np.asarray(reference, dtype = np.int16))
        assert drift.max() <= 2


def test_dmg_icon_label_stays_legible_over_the_halo() -> None:
    """Finder draws black "Unsloth" text here, so tinting it up is an accessibility change."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    renderer = load_module(REPO / "scripts/make_dmg_background.py")
    scale = renderer.SCALE
    # the band Finder puts the icon label in, just under the app icon
    label = (
        np.asarray(renderer.build().convert("RGB"), dtype = np.float32)[
            238 * scale : 260 * scale, 140 * scale : 220 * scale
        ]
        / 255.0
    )

    channel = np.where(label <= 0.04045, label / 12.92, ((label + 0.055) / 1.055) ** 2.4)
    luminance = channel @ np.array([0.2126, 0.7152, 0.0722], dtype = np.float32)
    assert (luminance.min() + 0.05) / 0.05 >= 7.0  # WCAG AAA for body text


def test_desktop_release_asset_names_are_human_readable() -> None:
    workflow = read(REPO / ".github/workflows/release-desktop.yml")
    assert "re.sub(r'[^0-9A-Za-z]+', '_', app_version).strip('_')" in workflow

    assert "base_name = f'Unsloth-Desktop-{os.environ[\"ASSET_VERSION\"]}'" in workflow
    expected_suffixes = {
        "MacOS.dmg",
        "ARM64.app.tar.gz",
        "ARM64.app.tar.gz.sig",
        "Linux.AppImage",
        "Linux.AppImage.sig",
        "Ubuntu.deb",
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
    assert '--title "Unsloth Desktop updater channel"' not in workflow
