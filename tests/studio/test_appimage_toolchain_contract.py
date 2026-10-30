# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep every fetched AppImage build tool digest-pinned and off rolling release aliases."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOL_SCRIPT = REPO / "studio/src-tauri/linux/prepare-complete-appimage-tools.sh"
ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)="([^"]+)"$', re.MULTILINE)
FETCH = re.compile(
    r'^fetch "\$([A-Z][A-Z0-9_]*)_URL" "\$\1_SHA256" (\S+)$',
    re.MULTILINE,
)
ROLLING_RELEASE_ALIASES = {"continuous", "latest"}


def _assignments(script: str) -> dict[str, str]:
    return dict(ASSIGNMENT.findall(script))


def _assert_no_rolling_release_aliases(script: str) -> None:
    values = _assignments(script)
    offenders = []
    for prefix, _filename in FETCH.findall(script):
        url_name = f"{prefix}_URL"
        url = values[url_name]
        path_parts = set(urlsplit(url).path.split("/"))
        if path_parts & ROLLING_RELEASE_ALIASES:
            offenders.append(f"{url_name}={url}")
    assert not offenders, "AppImage build tools must use versioned URLs: " + ", ".join(offenders)


def test_every_fetched_tool_has_a_sha256_digest() -> None:
    script = TOOL_SCRIPT.read_text(encoding = "utf-8")
    values = _assignments(script)
    fetches = FETCH.findall(script)
    assert fetches, "expected digest-verified AppImage tool fetches"

    for prefix, filename in fetches:
        url_name = f"{prefix}_URL"
        digest_name = f"{prefix}_SHA256"
        assert url_name in values, f"{filename} has no URL assignment"
        assert re.fullmatch(
            r"[0-9a-f]{64}", values.get(digest_name, "")
        ), f"{filename} has no full SHA-256 digest"


def test_fetched_tools_do_not_use_rolling_release_aliases() -> None:
    _assert_no_rolling_release_aliases(TOOL_SCRIPT.read_text(encoding = "utf-8"))


def test_the_former_continuous_plugin_url_is_rejected() -> None:
    former_pin = """
APPIMAGE_PLUGIN_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage"
APPIMAGE_PLUGIN_SHA256="0441769ab38009504d2678c38cd7e526955388dd30a215b4a20afaa5471652f2"
fetch "$APPIMAGE_PLUGIN_URL" "$APPIMAGE_PLUGIN_SHA256" linuxdeploy-plugin-appimage.AppImage
"""
    with pytest.raises(AssertionError, match = "APPIMAGE_PLUGIN_URL"):
        _assert_no_rolling_release_aliases(former_pin)


APPRUN = REPO / "studio/src-tauri/linux/appimage-apprun.sh"
FINALIZER = REPO / "studio/src-tauri/linux/finalize-complete-appimage.sh"
CLEAN_MACHINE_CI = REPO / ".github/workflows/desktop-app-clean-machine-ci.yml"
PROBE_FLAG = "--unsloth-webkit-probe"
HOST_EXE = "usr/bin/host/unsloth-studio"
# Every variable the pinned gtk and gstreamer hooks point into the bundle.
HOOK_VARIABLES = (
    "GTK_DATA_PREFIX",
    "GTK_EXE_PREFIX",
    "GTK_PATH",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE",
    "GSETTINGS_SCHEMA_DIR",
    "GIO_MODULE_DIR",
    "GIO_EXTRA_MODULES",
    "GST_REGISTRY_REUSE_PLUGIN_SCANNER",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SCANNER_1_0",
    "GST_PTP_HELPER_1_0",
)


def _code_lines(path: Path) -> list[str]:
    """Shell lines without comments, with backslash continuations joined."""
    joined = path.read_text(encoding = "utf-8").replace("\\\n", " ")
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _unset_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        for statement in re.split(r"[;&|]+", line):
            words = statement.split()
            if words[:1] == ["unset"]:
                names.update(word for word in words[1:] if not word.startswith("-"))
    return names


def test_apprun_probes_the_host_copy_only_under_a_timeout() -> None:
    lines = _code_lines(APPRUN)
    probes = [line for line in lines if PROBE_FLAG in line]
    assert probes, f"AppRun must probe the host copy with {PROBE_FLAG}"
    for line in probes:
        assert "timeout -k 1 5" in line, f"probe runs without `timeout -k 1 5`: {line}"
        assert "</dev/null" in line, f"probe must not read the caller's stdin: {line}"
    assert any(HOST_EXE in line for line in lines), f"AppRun never names {HOST_EXE}"


def test_apprun_restores_every_hook_variable_in_host_mode() -> None:
    lines = _code_lines(APPRUN)
    missing = sorted(set(HOOK_VARIABLES) - _unset_names(lines))
    assert not missing, "host mode leaves bundle paths set: " + ", ".join(missing)
    text = "\n".join(lines)
    assert "APPIMAGE_GTK_THEME" in text and "GTK_THEME" in _unset_names(lines)
    assert "XDG_DATA_DIRS" in text
    # Both modes keep the loader guard the verifier requires.
    assert "LD_LIBRARY_PATH" in _unset_names(lines)
    assert "UNSLOTH_HOST_LD_LIBRARY_PATH" in text
    assert re.search(r"export\s+(\w+\s+)*UNSLOTH_APPIMAGE_WEBKIT_MODE\b", text) or re.search(
        r"export\s+UNSLOTH_APPIMAGE_WEBKIT_MODE=", text
    ), "AppRun must export UNSLOTH_APPIMAGE_WEBKIT_MODE"
    assert "AppImage WebKitGTK runtime=" in text, "AppRun must log the chosen runtime"


def test_apprun_has_exactly_one_exec_per_mode() -> None:
    launches = [
        line
        for line in _code_lines(APPRUN)
        if re.match(r"exec\s", line) and '"$@"' in line and "timeout" not in line
    ]
    host = [line for line in launches if HOST_EXE in line]
    bundled = [line for line in launches if re.search(r"usr/bin/unsloth-studio\b", line)]
    assert len(host) == 1, f"expected one host-mode exec, got {host}"
    assert len(bundled) == 1, f"expected one bundled-mode exec, got {bundled}"
    assert len(launches) == 2, f"unexpected app exec lines: {launches}"


def test_finalizer_keeps_the_host_copy_out_of_the_runpath_loop() -> None:
    text = FINALIZER.read_text(encoding = "utf-8")
    loop_start = text.index("patched=0")
    loop_end = text.index("((patched > 0))")
    assert (
        "usr/bin/host" in text[loop_start:loop_end]
    ), "the RUNPATH loop must skip usr/bin/host so the copy keeps $ORIGIN/../lib"
    shell_vars = dict(re.findall(r'^\s*([a-z_][a-z0-9_]*)="([^"]*)"', text, re.MULTILINE))

    def expand(word: str) -> str:
        return re.sub(r"\$\{?([a-z_][a-z0-9_]*)\}?", lambda m: shell_vars.get(m[1], m[0]), word)

    install = next(
        (
            match
            for match in re.finditer(r"install -D -m 755 (\S+) (\S+)", text)
            if "usr/bin/unsloth-studio" in match[1] and HOST_EXE in expand(match[2])
        ),
        None,
    )
    assert install, "finalizer must install the host copy of the main binary"
    assert install.start() > text.index(
        "patchelf --set-rpath"
    ), "the host copy must be made after the RUNPATH loop"
    assert "usr/bin/lib" in text, "finalizer must reject a usr/bin/lib the host copy would reach"


def test_portability_mode_assertions_run_only_against_the_pr_build() -> None:
    """Scheduled runs test the shipped release, which has no host copy."""
    text = CLEAN_MACHINE_CI.read_text(encoding = "utf-8")
    assignments = re.findall(r"APPIMAGE_EXPECTED_WEBKIT_MODE:\s*(.+)", text)
    assert assignments, "the portability smoke must receive APPIMAGE_EXPECTED_WEBKIT_MODE"
    for value in assignments:
        assert "github.event_name == 'pull_request'" in value, value
