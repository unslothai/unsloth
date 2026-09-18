# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Desktop display-branding contracts."""

import importlib.util
import json
from pathlib import Path
import re
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

    # Finder lays the background out from the same origin it uses for icon coordinates, so the base page has to match
    # the configured window size or the artwork drifts out from under the app and Applications icons.
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
    assert "base_name = 'Unsloth-Desktop'" in workflow
    expected_suffixes = {
        "MacOS.dmg",
        "ARM64.app.tar.gz",
        "ARM64.app.tar.gz.sig",
        "Linux.AppImage",
        "Linux.AppImage.sig",
        "Ubuntu.deb",
        "Windows.exe",
        "Windows.exe.sig",
        "Windows-ARM64.exe",
        "Windows-ARM64.exe.sig",
    }
    for suffix in expected_suffixes:
        assert f"f'{{base_name}}-{suffix}'" in workflow

    for name in (
        "Unsloth-Desktop-MacOS.dmg",
        "Unsloth-Desktop-Linux.AppImage",
        "Unsloth-Desktop-Ubuntu.deb",
        "Unsloth-Desktop-Windows.exe",
        "Unsloth-Desktop-Windows-ARM64.exe",
    ):
        assert name in workflow


LOCALES = FRONTEND / "src/i18n/locales"

# The only locale entries allowed to say "Unsloth Studio": prose that names the *remote server* a user points this app
# at, which genuinely is an Unsloth Studio.
# modelAutoSwitch.apiOnlyDescription does NOT belong here. It renders as a settings-row description and describes a
# model you loaded from this UI, not from a remote server, so exempting it would let the display name back in on a
# rendered surface.
LOCALE_REMOTE_SERVER_KEYS = frozenset(
    {
        "settings.agents.remote.title",
        "settings.agents.remote.description",
    }
)

LOCALE_KEY = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")

# The Rust half of the sweep walks the whole crate rather than a hand-kept file list. The list version held six files
# and let two live violations through: native_file_dialogs.rs owned the log-export sentinel the settings tab renders,
# and staged_update.rs told the user to "Quit Unsloth Studio" when the app they are looking at is called Unsloth.
# Neither file was on the list, so neither was ever asked.
RUST_SOURCES = TAURI / "src"

# Lines allowed to carry the display name, matched whole and stripped so the exemption cannot widen by editing around
# it. This is transcribed output, not copy: the AMSI provider really printed that line, and rewriting it would make the
# fixture stop reproducing the error it was captured from (#8523).
RUST_VERBATIM_LINES = frozenset(
    {
        '"+ # Unsloth Studio Installer for Windows PowerShell",',
    }
)


def locale_entries(text: str) -> list[tuple[str, str]]:
    """Every leaf entry of a locale module as (dotted key path, value text).

    The catalogs are plain nested object literals, and values routinely wrap onto their
    own line, so an entry runs from its key to the next key or closing brace.
    """
    stack: list[tuple[int, str]] = []
    out: list[tuple[str, str]] = []
    path: str | None = None
    buf = ""
    for line in text.splitlines():
        match = LOCALE_KEY.match(line)
        if match:
            if path is not None:
                out.append((path, buf))
            indent, name = len(match.group(1)), match.group(2)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = ".".join([held for _, held in stack] + [name])
            buf = line[match.end() :]
            if line.rstrip().endswith(("{", "[")):
                stack.append((indent, name))
                path, buf = None, ""
        elif path is not None:
            buf += "\n" + line
            if re.match(r"^\s*[}\]]", line):
                out.append((path, buf))
                path, buf = None, ""
    if path is not None:
        out.append((path, buf))
    return out


def rust_branding_offenders() -> list[str]:
    """Every crate line naming the display name, bar the transcribed ones.

    Line granularity rather than file granularity so one verbatim fixture does not buy its
    whole file an exemption: install.rs holds captured AMSI stderr, and the rest of install.rs
    is still ordinary user-facing Rust that has to obey the contract.
    """
    return [
        f"{path.relative_to(REPO)}:{number}"
        for path in sorted(RUST_SOURCES.rglob("*.rs"))
        for number, line in enumerate(read(path).splitlines(), start = 1)
        if "Unsloth Studio" in line and line.strip() not in RUST_VERBATIM_LINES
    ]


def test_desktop_surfaces_do_not_restore_studio_branding() -> None:
    # The desktop app displays itself as "Unsloth", never "Unsloth Studio". The i18n catalogs are swept by key rather
    # than by file: a handful of entries have to name the *remote server* a user points the app at, which genuinely is
    # an Unsloth Studio and is not this app's display name, so those keys are spared and every other entry is not.
    display_sources = [
        TAURI / "Info.plist",
        TAURI / "capabilities/default.json",
        TAURI / "windows/sign-with-trusted-signing.ps1",
        REPO / ".github/workflows/release-desktop.yml",
        FRONTEND / "index.html",
        *sorted(
            path
            for suffix in ("*.ts", "*.tsx")
            for path in (FRONTEND / "src").rglob(suffix)
            if LOCALES not in path.parents
        ),
    ]
    offenders = [
        str(path.relative_to(REPO)) for path in display_sources if "Unsloth Studio" in read(path)
    ]

    # The crate is swept whole. The four Rust files that used to be named here are still covered, and so is every
    # other one: a user-facing sentence is not likelier to be right for having been added to a file nobody listed.
    offenders += rust_branding_offenders()

    # The locale catalogs are swept too, just at key granularity rather than file granularity, so only the remote-server
    # prose is spared.
    offenders += [
        f"{path.relative_to(REPO)}::{key}"
        for path in sorted(LOCALES.rglob("*.ts"))
        for key, value in locale_entries(read(path))
        if "Unsloth Studio" in value and key not in LOCALE_REMOTE_SERVER_KEYS
    ]
    assert offenders == []

    workflow = read(REPO / ".github/workflows/release-desktop.yml")
    assert "Desktop app for Unsloth." in workflow
    assert '--title "Unsloth Desktop updater channel"' not in workflow


def test_the_branding_sweep_still_covers_the_crate() -> None:
    """The Rust half has to keep walking a real tree, and the exemption has to stay verbatim.

    The failure this guards is the one the hand-kept list actually produced: a sweep that looks
    thorough while never reading the file the offending sentence lives in. Here that shape would
    be an rglob that returns nothing after a crate reshuffle, which reports zero offenders and
    passes.
    """
    swept = sorted(RUST_SOURCES.rglob("*.rs"))
    assert RUST_SOURCES.is_dir(), f"the crate source root moved: {RUST_SOURCES}"
    assert len(swept) >= 25, f"the crate sweep collapsed to {len(swept)} files"

    # The two that the old list omitted, named so a reshuffle that drops them is not silent.
    for name in ("native_file_dialogs.rs", "staged_update.rs", "process.rs", "main.rs"):
        assert any(path.name == name for path in swept), f"{name} left the sweep"

    # An exemption for a line no longer in the tree is an exemption nobody re-read. It has to be
    # spent, and spent on the fixture it was written for.
    present = {
        line.strip() for path in swept for line in read(path).splitlines() if "Unsloth Studio" in line
    }
    assert RUST_VERBATIM_LINES <= present, (
        f"stale Rust exemptions: {sorted(RUST_VERBATIM_LINES - present)}"
    )
    assert len(RUST_VERBATIM_LINES) < 5, "the verbatim allowlist is for transcribed output, not copy"

    # Nothing is dropped beyond the allowlist: every raw hit is either reported or exempt. A filter
    # that quietly skipped a directory, a file extension or a line shape would show up here as a
    # raw count the reported and exempt ones do not add back up to.
    raw = [
        line.strip()
        for path in swept
        for line in read(path).splitlines()
        if "Unsloth Studio" in line
    ]
    exempt = [line for line in raw if line in RUST_VERBATIM_LINES]
    assert len(raw) - len(exempt) == len(rust_branding_offenders())


def test_the_branding_sweep_still_covers_the_frontend() -> None:
    """The locale exemption must stay narrow.

    A sweep that matches nothing passes this contract while proving nothing. Both halves
    can fail that way: move src and the rglob goes empty, or reformat the catalogs and the
    key parser yields nothing, either one leaving the test green over an unchecked tree.
    """
    swept = [
        path
        for suffix in ("*.ts", "*.tsx")
        for path in (FRONTEND / "src").rglob(suffix)
        if LOCALES not in path.parents
    ]
    locales = sorted(LOCALES.rglob("*.ts"))

    assert LOCALES.is_dir(), f"the exempt directory moved: {LOCALES}"
    assert len(locales) >= 10, f"locales look wrong, found {len(locales)}"
    assert len(swept) > 20 * len(locales), f"sweep collapsed to {len(swept)} files"

    # The catalogs are swept by key, so the parser has to actually resolve keys.
    for path in locales:
        entries = dict(locale_entries(read(path)))
        assert len(entries) > 500, f"{path.name} parsed to {len(entries)} entries"
        for key in (
            "shell.product",
            "settings.about.shutDownStudio",
            "settings.about.studioVersion",
            "settings.about.license.studioLabel",
        ):
            assert key in entries, f"{path.name} lost {key}, so the sweep no longer sees it"

    # The allowlist is prose-level, not a blanket: it spares three of the ~1,500 entries a catalog holds, and every
    # exempt key has to be one the catalogs actually define.
    english = dict(locale_entries(read(LOCALES / "en.ts")))
    assert LOCALE_REMOTE_SERVER_KEYS <= set(
        english
    ), f"exempt keys missing from en.ts: {sorted(LOCALE_REMOTE_SERVER_KEYS - set(english))}"
    assert len(LOCALE_REMOTE_SERVER_KEYS) < len(english) / 100
