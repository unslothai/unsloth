"""Contracts for the complete Linux AppImage release path."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

CLEAN_MACHINE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "desktop-app-clean-machine-ci.yml"
VERIFIER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "verify-complete-appimage.sh"

FINALIZER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "finalize-complete-appimage.sh"

APPRUN = REPO_ROOT / "studio" / "src-tauri" / "linux" / "appimage-apprun.sh"

FONTCONFIG = REPO_ROOT / "studio" / "src-tauri" / "linux" / "appimage-fonts.conf"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _step(name: str):
    return next(step for step in _workflow()["jobs"]["build"]["steps"] if step.get("name") == name)


def test_tauri_builds_and_signs_deb_and_complete_appimage_together():
    config = json.loads(
        (REPO_ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    )
    assert "appimage" in config["bundle"]["targets"]
    appimage = config["bundle"]["linux"]["appimage"]
    assert appimage["bundleMediaFramework"] is True
    assert appimage["files"]["/usr/lib/libappindicator3.so.1"].endswith("/libappindicator3.so.1")

    dependencies = _step("Install Linux dependencies")["run"]
    for package in (
        "fonts-noto-color-emoji",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-libav",
    ):
        assert package in dependencies

    build = _step("Build Linux bundles")
    verify = _step("Verify complete Linux AppImage")
    stage = _step("Stage release assets")
    assert "--bundles deb,appimage" in build["with"]["args"]
    assert build["env"]["XDG_CACHE_HOME"] == "${{ runner.temp }}/tauri-tools-cache"
    assert "TAURI_SIGNING_PRIVATE_KEY" in build["env"]
    assert "verify-complete-appimage.sh" in verify["run"]
    assert stage["env"]["ARTIFACT_PATHS"].startswith("${{ steps.build_linux.outputs.artifactPaths")

    clean_machine = yaml.safe_load(CLEAN_MACHINE_WORKFLOW.read_text(encoding = "utf-8"))
    e2e = clean_machine["jobs"]["appimage-model-download"]
    e2e_source = yaml.safe_dump(e2e)
    webdriver_install = next(
        step for step in e2e["steps"] if step.get("name") == "Install WebDriver prerequisites"
    )
    assert "webkit2gtk-driver" in webdriver_install["run"]
    assert "tauri-driver --version 2.0.6 --locked" in webdriver_install["run"]
    assert "appimage_model_download_webdriver.py" in e2e_source

    colrv1_sha = "0ae57fe58645638523ba35f388d93739d292539a9acb84df5700c81b1e1a28d2"
    assert "googlefonts/noto-emoji/8998f5dd683424a73e2314a8c1f1e359c19e8742" in e2e_source
    assert e2e_source.count(colrv1_sha) >= 2
    assert "APPIMAGE_COLRV1_FONT" in e2e_source
    webdriver_script = (REPO_ROOT / "tests/studio/appimage_model_download_webdriver.py").read_text(
        encoding = "utf-8"
    )
    assert "Unsloth Test COLRv1" in webdriver_script
    assert '"route": "/hub"' in webdriver_script
    assert '"survived_seconds": 0' in webdriver_script
    assert "colrv1-model-hub.json" in webdriver_script


def test_appimage_pr_build_is_unsigned_and_feeds_every_artifact_test():
    workflow = yaml.safe_load(CLEAN_MACHINE_WORKFLOW.read_text(encoding = "utf-8"))
    pull_request_paths = workflow[True]["pull_request"]["paths"]
    for relevant_path in (
        ".github/workflows/release-desktop.yml",
        ".github/workflows/desktop-app-clean-machine-ci.yml",
        "studio/src-tauri/linux/**",
        "studio/src-tauri/src/**",
        "studio/src-tauri/tauri.conf.json",
        "tests/security/test_release_desktop_appimage.py",
    ):
        assert relevant_path in pull_request_paths

    jobs = workflow["jobs"]
    build = jobs["appimage-pr-build"]
    build_source = yaml.safe_dump(build)
    assert "github.event_name == 'pull_request'" in build["if"]
    for package in (
        "fonts-noto-color-emoji",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-libav",
    ):
        assert package in build_source
    assert "TAURI_SIGNING_PRIVATE_KEY" not in build_source
    assert "createUpdaterArtifacts" in build_source
    assert "false" in build_source
    assert "--bundles appimage" in build_source
    assert "verify-complete-appimage.sh" in build_source
    assert "appimage-pr-build" in build_source

    for job_name in ("appimage-portability", "appimage-model-download"):
        job = jobs[job_name]
        source = yaml.safe_dump(job)
        assert "appimage-pr-build" in job["needs"]
        assert "github.event_name != 'pull_request'" in job["if"]
        assert "actions/download-artifact" in source
        assert "name: appimage-pr-build" in source
        assert "github.event_name == 'pull_request'" in source
        assert "github.event_name != 'pull_request'" in source
        assert "head.repo.fork" not in source


def test_debian_portability_lanes_install_verifier_and_host_runtime_prerequisites():
    workflow = yaml.safe_load(CLEAN_MACHINE_WORKFLOW.read_text(encoding = "utf-8"))
    job = workflow["jobs"]["appimage-portability"]
    source = yaml.safe_dump(job)
    for package in (
        "binutils",
        "libegl1",
        "libgbm1",
        "libwayland-client0",
        "libwayland-egl1",
        "libxcb1",
        "libxinerama1",
        "libasound2t64",
        "libharfbuzz0b",
        "libnghttp2-14",
    ):
        assert package in source
    assert "weston" in source
    assert "APPIMAGE_DISPLAY_BACKEND" in source
    assert "wayland" in source

    no_gles = next(
        lane
        for lane in job["strategy"]["matrix"]["include"]
        if lane["label"] == "ubuntu-22.04-no-gles"
    )
    assert no_gles["install_gles"] is False
    assert "libGLESv2.so.2" in source

    assert "appimage_media_pipeline_probe.py" in source

    linux_source = yaml.safe_dump(workflow["jobs"]["linux"])
    webdriver_source = yaml.safe_dump(workflow["jobs"]["appimage-model-download"])
    for package in (
        "libegl1",
        "libgbm1",
        "libwayland-client0",
        "libharfbuzz0b",
        "libnghttp2-14",
    ):
        assert package in linux_source
        assert package in webdriver_source
    for package in ("libasound2", "libpulse0"):
        assert package in webdriver_source

    for package in ("libwayland-client", "libxcb", "libXinerama", "libnghttp2"):
        assert package in source


def test_release_preseeds_every_tauri_appimage_tool_with_a_digest():
    step = _step("Pin complete AppImage toolchain")
    assert step["if"] == "matrix.platform == 'ubuntu-22.04'"
    assert "prepare-complete-appimage-tools.sh" in step["run"]
    tool_script = (
        REPO_ROOT / "studio/src-tauri/linux/prepare-complete-appimage-tools.sh"
    ).read_text(encoding = "utf-8")

    finalizer_source = FINALIZER.read_text(encoding = "utf-8")
    expected = {
        "LINUXDEPLOY": ("linuxdeploy-x86_64.AppImage", "linuxdeploy-x86_64.AppImage"),
        "GTK_PLUGIN": ("linuxdeploy-plugin-gtk.sh", "linuxdeploy-plugin-gtk.sh"),
        "GSTREAMER_PLUGIN": (
            "linuxdeploy-plugin-gstreamer.sh",
            "linuxdeploy-plugin-gstreamer.sh",
        ),
        "APPIMAGE_PLUGIN": (
            "linuxdeploy-plugin-appimage-x86_64.AppImage",
            "linuxdeploy-plugin-appimage.AppImage",
        ),
    }
    for prefix, (url_filename, destination) in expected.items():
        assert re.search(rf'^{prefix}_URL="[^"]*{re.escape(url_filename)}"$', tool_script, re.M)
        assert re.search(rf'^{prefix}_SHA256="[0-9a-f]{{64}}"$', tool_script, re.M)
        assert f'fetch "${prefix}_URL" "${prefix}_SHA256" {destination}' in tool_script
    fetch_calls = [
        line.strip() for line in tool_script.splitlines() if line.strip().startswith("fetch ")
    ]
    assert len(fetch_calls) == len(expected)
    assert tool_script.index("sha256sum -c") < tool_script.index("chmod +x")

    assert "apprun-old" not in tool_script
    for local_tool in (
        "appimage-apprun.sh",
        "appimage-fonts.conf",
        "finalize-complete-appimage.sh",
    ):
        assert local_tool in tool_script

    assert "patchelf --set-rpath" in finalizer_source
    assert "$ORIGIN" in finalizer_source

    for asset in (
        "UnslothSafeEmoji.ttf",
        "UnslothSafeEmoji.LICENSE",
        "unsloth-appimage-fonts.conf",
    ):
        assert asset in tool_script
        assert asset in finalizer_source
    fontconfig_source = FONTCONFIG.read_text(encoding = "utf-8")
    ElementTree.fromstring(fontconfig_source)
    # AppRun replaces @APPDIR@ because Fontconfig 2.13 misresolves relative paths.
    assert "<dir>@APPDIR@/usr/share/unsloth/fonts</dir>" in fontconfig_source
    assert "<dir prefix=" not in fontconfig_source
    assert '<match target="scan">' in fontconfig_source
    assert "Unsloth Safe Emoji" in fontconfig_source

    assert "<selectfont>" in fontconfig_source
    assert "<rejectfont>" in fontconfig_source
    assert '<patelt name="color"><bool>true</bool></patelt>' in fontconfig_source
    assert "<acceptfont>" in fontconfig_source
    assert '<patelt name="family"><string>Unsloth Safe Emoji</string></patelt>' in fontconfig_source
    assert fontconfig_source.index("<acceptfont>") < fontconfig_source.index("<rejectfont>")

    # Only emoji requests may strongly prefer the bundled font.
    pattern_rules = re.findall(
        r'<match target="pattern">(.*?)</match>', fontconfig_source, re.DOTALL
    )
    assert pattern_rules
    for rule in pattern_rules:
        assert '<test name="family">' in rule or '<test name="lang">' in rule
        if 'mode="prepend"' in rule:
            assert "<string>emoji</string>" in rule or "<string>und-zsye</string>" in rule
        else:
            assert 'mode="append" binding="weak"' in rule
    for guard in ("und-zsye", "emoji", "sans-serif", "serif", "monospace"):
        assert any(f"<string>{guard}</string>" in rule for rule in pattern_rules)
    assert 'case "${APPDIR:-}" in' in tool_script
    assert 'APPDIR="$(dirname "$(realpath "$0")")"' in tool_script
    for host_library in (
        "libwayland-*.so*",
        "libGLES*.so*",
        "libGL*.so*",
        "libEGL*.so*",
        "libnghttp2.so*",
        "libcurl*.so*",
        "libstdc++.so*",
        "libgcc_s.so*",
    ):
        assert host_library in finalizer_source
    assert "GIO_MODULE_DIR" in tool_script

    assert "unset GIO_EXTRA_MODULES" in tool_script

    assert "sed -i '/export GDK_BACKEND=x11/d'" in tool_script

    assert "-path '*/gio/modules/*' -type f -print0" in tool_script
    assert 'export GTK_PATH="\\$APPDIR/' in tool_script

    assert tool_script.count('"$plugin_dir/finalize-complete-appimage.sh" "$APPDIR"') == 1
    assert "for plugin in linuxdeploy-plugin-gtk.sh linuxdeploy-plugin-gstreamer.sh" in tool_script


def _compile_fixture_elf(path: Path, *, origin_runpath: bool) -> None:
    args = ["cc", "-x", "c", "-", "-o", path]
    if origin_runpath:
        args.insert(-2, "-Wl,-rpath,$ORIGIN/../lib")
    subprocess.run(
        args,
        input = "int main(void) { return 0; }\n",
        check = True,
        text = True,
        capture_output = True,
    )


def _fake_complete_appdir(tmp_path: Path) -> Path:
    appdir = tmp_path / "AppDir"
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    _compile_fixture_elf(binary, origin_runpath = True)

    (appdir / "Unsloth.png").touch()
    (appdir / ".DirIcon").symlink_to("Unsloth.png")
    apprun = appdir / "AppRun"
    apprun.write_text(
        "#!/bin/sh\n"
        '. "$APPDIR/apprun-hooks/linuxdeploy-plugin-gtk.sh"\n'
        "unset LD_LIBRARY_PATH\n"
        "sed -e 's,&,\\&amp;,g'\n"
        'sed "s|@APPDIR@|$unsloth_fonts_appdir|g" "$unsloth_fonts_template"\n'
        "exit 0\n",
        encoding = "utf-8",
    )
    apprun.chmod(0o755)
    hook = appdir / "apprun-hooks/linuxdeploy-plugin-gtk.sh"
    hook.parent.mkdir()
    hook.write_text(
        "unset GIO_EXTRA_MODULES\n"
        'export GIO_MODULE_DIR="$APPDIR/usr/lib/gio/modules"\n'
        'export GTK_PATH="$APPDIR/usr/lib/gtk-3.0"\n',
        encoding = "utf-8",
    )
    runtime = appdir / "usr/lib"
    runtime.mkdir(parents = True)
    for name in (
        "libglib-2.0.so.0",
        "libgobject-2.0.so.0",
        "libgio-2.0.so.0",
        "libgtk-3.so.0",
        "libgdk-3.so.0",
        "libgdk_pixbuf-2.0.so.0",
        "libwebkit2gtk-4.1.so.0",
        "libjavascriptcoregtk-4.1.so.0",
        "libsoup-3.0.so.0",
        "libappindicator3.so.1",
        "WebKitNetworkProcess",
        "WebKitWebProcess",
        "libwebkit2gtkinjectedbundle.so",
    ):
        (runtime / name).touch()

    gio_modules = runtime / "gio/modules"
    gio_modules.mkdir(parents = True)
    (gio_modules / "libgiognutls.so").touch()

    gst_plugins = runtime / "gstreamer-1.0"
    gst_plugins.mkdir()
    for name in (
        "coreelements",
        "playback",
        "pulseaudio",
        "typefindfunctions",
        "isomp4",
        "videoparsersbad",
        "libav",
    ):
        (gst_plugins / f"libgst{name}.so").touch()
    for index in range(60):
        (gst_plugins / f"libgstfixture{index}.so").touch()
    scanner = runtime / "gstreamer1.0/gstreamer-1.0/gst-plugin-scanner"
    scanner.parent.mkdir(parents = True)
    scanner.touch()
    safe_font = appdir / "usr/share/unsloth/fonts/UnslothSafeEmoji.ttf"
    safe_font.parent.mkdir(parents = True)
    safe_font.write_bytes(b"fixture CBDT CBLC bitmap font tables")
    safe_license = appdir / "usr/share/doc/unsloth-safe-emoji/copyright"
    safe_license.parent.mkdir(parents = True)
    safe_license.write_text("fixture OFL license\n", encoding = "utf-8")
    fontconfig = appdir / "usr/etc/fonts/unsloth-appimage.conf"
    fontconfig.parent.mkdir(parents = True)
    fontconfig.write_text(
        "Unsloth Safe Emoji\n<dir>@APPDIR@/usr/share/unsloth/fonts</dir>\n",
        encoding = "utf-8",
    )

    return appdir


def _write_foreign_arch_elf(path: Path) -> None:
    """An i386 ELF header, the shape a multilib build host contributes."""

    header = bytearray(52)
    header[0:8] = b"\x7fELF\x01\x01\x01\x00"
    header[16:18] = (3).to_bytes(2, "little")  # e_type = ET_DYN
    header[18:20] = (3).to_bytes(2, "little")  # e_machine = EM_386
    header[20:24] = (1).to_bytes(4, "little")
    header[40:42] = (52).to_bytes(2, "little")
    path.write_bytes(bytes(header))


def test_complete_appimage_verifier_accepts_a_coherent_runtime(tmp_path):
    result = subprocess.run(
        [VERIFIER, "--appdir", _fake_complete_appdir(tmp_path)],
        check = True,
        capture_output = True,
        text = True,
    )
    assert "Verified complete x86_64 AppImage runtime" in result.stdout


def test_complete_appimage_verifier_rejects_host_gtk_module_directories(tmp_path):
    appdir = _fake_complete_appdir(tmp_path)
    hook = appdir / "apprun-hooks/linuxdeploy-plugin-gtk.sh"
    hook.write_text(
        hook.read_text(encoding = "utf-8").replace(
            'export GTK_PATH="$APPDIR/usr/lib/gtk-3.0"',
            'export GTK_PATH="$APPDIR/usr/lib/gtk-3.0:/usr/lib64/gtk-3.0"',
        ),
        encoding = "utf-8",
    )
    result = subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "/usr/lib64/gtk-3.0" in result.stderr


def test_complete_appimage_verifier_rejects_a_foreign_architecture_object(tmp_path):
    appdir = _fake_complete_appdir(tmp_path)
    _write_foreign_arch_elf(appdir / "usr/lib/gio/modules/libgiognutls.so")
    result = subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "wrong architecture" in result.stderr


def test_complete_appimage_verifier_requires_the_bundled_media_pipeline(tmp_path):
    appdir = _fake_complete_appdir(tmp_path)
    for plugin in (appdir / "usr/lib/gstreamer-1.0").glob("libgstfixture*.so"):
        plugin.unlink()
    result = subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "GStreamer plugins" in result.stderr

    missing_scanner = _fake_complete_appdir(tmp_path / "no-scanner")
    (missing_scanner / "usr/lib/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner").unlink()
    result = subprocess.run(
        [VERIFIER, "--appdir", missing_scanner], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "gst-plugin-scanner" in result.stderr


def test_complete_appimage_verifier_rejects_global_library_path_and_missing_origin_runpath(
    tmp_path,
):
    global_path = _fake_complete_appdir(tmp_path / "global-path")
    (global_path / "AppRun").write_text(
        '#!/bin/sh\nexport LD_LIBRARY_PATH="$APPDIR/usr/lib:${LD_LIBRARY_PATH:-}"\n',
        encoding = "utf-8",
    )
    result = subprocess.run(
        [VERIFIER, "--appdir", global_path], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "LD_LIBRARY_PATH" in result.stderr

    missing_runpath = _fake_complete_appdir(tmp_path / "missing-runpath")
    _compile_fixture_elf(missing_runpath / "usr/lib/WebKitWebProcess", origin_runpath = False)
    result = subprocess.run(
        [VERIFIER, "--appdir", missing_runpath],
        check = False,
        capture_output = True,
        text = True,
    )
    assert result.returncode != 0
    assert "$ORIGIN-relative RUNPATH" in result.stderr


# The pinned linuxdeploy plugins' hooks as shipped (gtk patched by prepare-complete-appimage-tools.sh).
_GTK_HOOK = """#! /usr/bin/env bash
case "${APPDIR:-}" in
  /*) ;;
  *) APPDIR="$(dirname "$(realpath "$0")")" ;;
esac
export APPDIR

gsettings get org.gnome.desktop.interface gtk-theme 2> /dev/null | grep -qi "dark" && GTK_THEME_VARIANT="dark" || GTK_THEME_VARIANT="light"
APPIMAGE_GTK_THEME="${APPIMAGE_GTK_THEME:-"Adwaita:$GTK_THEME_VARIANT"}" # Allow user to override theme (discouraged)

export APPDIR="${APPDIR:-"$(dirname "$(realpath "$0")")"}" # Workaround to run extracted AppImage
export GTK_DATA_PREFIX="$APPDIR"
export GTK_THEME="$APPIMAGE_GTK_THEME" # Custom themes are broken
export XDG_DATA_DIRS="$APPDIR/usr/share:/usr/share:$XDG_DATA_DIRS" # g_get_system_data_dirs() from GLib
export GSETTINGS_SCHEMA_DIR="$APPDIR//usr/share/glib-2.0/schemas"
export GTK_EXE_PREFIX="$APPDIR//usr"
export GTK_PATH="$APPDIR//usr/lib/x86_64-linux-gnu/gtk-3.0:/usr/lib64/gtk-3.0:/usr/lib/x86_64-linux-gnu/gtk-3.0"
export GTK_IM_MODULE_FILE="$APPDIR//usr/lib/x86_64-linux-gnu/gtk-3.0/3.0.0/immodules.cache"

export GDK_PIXBUF_MODULE_FILE="$APPDIR//usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/2.10.0/loaders.cache"
export GIO_EXTRA_MODULES="$APPDIR/usr/lib/x86_64-linux-gnu/gio/modules"
unset GIO_EXTRA_MODULES
export GIO_MODULE_DIR="$APPDIR/usr/lib/x86_64-linux-gnu/gio/modules"
export GTK_PATH="$APPDIR/usr/lib/x86_64-linux-gnu/gtk-3.0"
"""

_GST_HOOK = """#! /bin/bash

export GST_REGISTRY_REUSE_PLUGIN_SCANNER="no"
export GST_PLUGIN_SYSTEM_PATH_1_0="${APPDIR}/usr/lib/gstreamer-1.0"
export GST_PLUGIN_PATH_1_0="${APPDIR}/usr/lib/gstreamer-1.0"

export GST_PLUGIN_SCANNER_1_0="${APPDIR}/usr/lib/gstreamer1.0/gstreamer-1.0/gst-plugin-scanner"
export GST_PTP_HELPER_1_0="${APPDIR}/usr/lib/gstreamer1.0/gstreamer-1.0/gst-ptp-helper"
"""

# AppRun.wrapped is #!/bin/sh, which is dash on Debian-family hosts; prefer dash where present.
_POSIX_SH = shutil.which("dash")
_SH = [_POSIX_SH] if _POSIX_SH else []

_LINUXDEPLOY_APPRUN = f"""#! /usr/bin/env bash

# autogenerated by linuxdeploy

# make sure errors in sourced scripts will cause this script to stop
set -e

this_dir="$(readlink -f "$(dirname "$0")")"

source "$this_dir"/apprun-hooks/"linuxdeploy-plugin-gtk.sh"
source "$this_dir"/apprun-hooks/"linuxdeploy-plugin-gstreamer.sh"

exec {_POSIX_SH or ""} "$this_dir"/AppRun.wrapped "$@"
"""

# The bundled stand-in reports its exec target, cwd and argv before its environment.
_FAKE_BUNDLED_BINARY = (
    "#!/bin/sh\n"
    "echo EXE=bundled\n"
    'echo "CWD=$(pwd -P)"\n'
    'echo "ARGV=$*"\n'
    "exec /usr/bin/env\n"
)

# The host-mode stand-in answers the WebKitGTK probe from FAKE_PROBE_* knobs and records that it
# ran, with the probe's cwd and environment, in FAKE_PROBE_MARKER.
_FAKE_HOST_BINARY = (
    "#!/bin/sh\n"
    'if [ "$#" -eq 1 ] && [ "$1" = --unsloth-webkit-probe ]; then\n'
    '  { echo "cwd=$(pwd -P)"; /usr/bin/env; } >"$FAKE_PROBE_MARKER"\n'
    '  echo "probe noise on stderr" >&2\n'
    '  [ -z "${FAKE_PROBE_SLEEP:-}" ] || sleep "$FAKE_PROBE_SLEEP"\n'
    '  printf "%b" "${FAKE_PROBE_OUTPUT-webkit=2.54.2\\n}"\n'
    '  exit "${FAKE_PROBE_EXIT:-0}"\n'
    "fi\n"
    "echo EXE=host\n"
    'echo "CWD=$(pwd -P)"\n'
    'echo "ARGV=$*"\n'
    "exec /usr/bin/env\n"
)


def _apprun_mount(
    tmp_path: Path,
    name: str = "AppDir",
    *,
    host_copy: bool = False,
    hooked: bool = False,
) -> Path:
    """An AppDir holding just what AppRun itself touches.

    host_copy adds the usr/bin/host executable; hooked wraps AppRun the way linuxdeploy does,
    sourcing the gtk then gstreamer hooks before it execs AppRun.wrapped.
    """

    appdir = tmp_path / name
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    binary.write_text(_FAKE_BUNDLED_BINARY, encoding = "utf-8")
    binary.chmod(0o755)
    if host_copy:
        host = appdir / "usr/bin/host/unsloth-studio"
        host.parent.mkdir()
        host.write_text(_FAKE_HOST_BINARY, encoding = "utf-8")
        host.chmod(0o755)
    if hooked:
        hooks = appdir / "apprun-hooks"
        hooks.mkdir()
        (hooks / "linuxdeploy-plugin-gtk.sh").write_text(_GTK_HOOK, encoding = "utf-8")
        (hooks / "linuxdeploy-plugin-gstreamer.sh").write_text(_GST_HOOK, encoding = "utf-8")
        wrapper = appdir / "AppRun"
        wrapper.write_text(_LINUXDEPLOY_APPRUN, encoding = "utf-8")
        wrapper.chmod(0o755)
    apprun = appdir / ("AppRun.wrapped" if hooked else "AppRun")
    apprun.write_bytes(APPRUN.read_bytes())
    apprun.chmod(0o755)
    template = appdir / "usr/etc/fonts/unsloth-appimage.conf"
    template.parent.mkdir(parents = True)
    template.write_bytes(FONTCONFIG.read_bytes())
    return appdir


def test_apprun_hands_an_inherited_library_path_to_children_only(tmp_path):
    """The loader reads LD_LIBRARY_PATH before the bundle's own $ORIGIN RUNPATHs."""

    appdir = _apprun_mount(tmp_path)
    apprun = appdir / "AppRun"
    template = appdir / "usr/etc/fonts/unsloth-appimage.conf"
    state = tmp_path / "state"

    result = subprocess.run(
        [apprun],
        check = True,
        capture_output = True,
        text = True,
        env = {
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/opt/conda/lib:/opt/rocm/lib",
            "XDG_RUNTIME_DIR": str(state),
        },
    )
    printed = result.stdout.splitlines()
    assert not [line for line in printed if line.startswith("LD_LIBRARY_PATH=")]
    assert "UNSLOTH_HOST_LD_LIBRARY_PATH=/opt/conda/lib:/opt/rocm/lib" in printed

    materialized = state / "unsloth-studio/fonts-AppDir.conf"
    assert f"FONTCONFIG_FILE={materialized}" in printed
    assert f"<dir>{appdir}/usr/share/unsloth/fonts</dir>" in materialized.read_text(
        encoding = "utf-8"
    )
    assert "@APPDIR@" not in materialized.read_text(encoding = "utf-8")


def test_apprun_retires_only_the_font_policies_whose_mount_is_gone(tmp_path):
    """A later launch preserves live-mount policies and removes departed ones."""

    state = tmp_path / "state"
    env = {"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(state)}
    live = _apprun_mount(tmp_path, "mount-live")
    subprocess.run([live / "AppRun"], check = True, capture_output = True, env = env)
    live_policy = state / "unsloth-studio/fonts-mount-live.conf"
    assert live_policy.is_file()

    dead = _apprun_mount(tmp_path, "mount-dead")
    subprocess.run([dead / "AppRun"], check = True, capture_output = True, env = env)
    dead_policy = state / "unsloth-studio/fonts-mount-dead.conf"
    assert dead_policy.is_file()
    shutil.rmtree(dead)

    later = _apprun_mount(tmp_path, "mount-later")
    result = subprocess.run([later / "AppRun"], check = True, capture_output = True, text = True, env = env)

    assert live_policy.is_file(), "a running instance lost its font policy"
    assert not dead_policy.exists(), "a departed mount's font policy was kept"
    later_policy = state / "unsloth-studio/fonts-mount-later.conf"
    assert f"FONTCONFIG_FILE={later_policy}" in result.stdout.splitlines()
    assert later_policy.is_file()


def test_apprun_encodes_a_mount_path_carrying_xml_and_sed_metacharacters(tmp_path):
    """The AppImage runtime copies the file's own name into the mount path."""

    # An AppImage named "R&D<x>.AppImage" mounts under /tmp/.mount_R&D<x>XXXXXX.
    appdir = _apprun_mount(tmp_path, "mount-R&D<x>|y\\z")
    state = tmp_path / "state"

    result = subprocess.run(
        [appdir / "AppRun"],
        check = True,
        capture_output = True,
        text = True,
        env = {"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(state)},
    )

    materialized = state / f"unsloth-studio/fonts-{appdir.name}.conf"
    assert f"FONTCONFIG_FILE={materialized}" in result.stdout.splitlines()
    policy = materialized.read_text(encoding = "utf-8")
    assert "@APPDIR@" not in policy

    # Fontconfig drops a whole policy it cannot parse, which puts host COLRv1 fonts back in front of Skia.
    root = ElementTree.fromstring(policy)
    directories = [element.text for element in root.findall("dir")]
    assert directories == [f"{appdir}/usr/share/unsloth/fonts"]
    assert root.find("selectfont/rejectfont") is not None


def test_apprun_keeps_a_live_policy_whose_mount_path_needed_encoding(tmp_path):
    """Cleanup compares mounts on disk, so it has to decode what it wrote."""

    state = tmp_path / "state"
    env = {"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(state)}
    live = _apprun_mount(tmp_path, "mount-R&D<x>")
    subprocess.run([live / "AppRun"], check = True, capture_output = True, env = env)
    live_policy = state / f"unsloth-studio/fonts-{live.name}.conf"
    assert live_policy.is_file()

    later = _apprun_mount(tmp_path, "mount-later")
    subprocess.run([later / "AppRun"], check = True, capture_output = True, env = env)

    assert live_policy.is_file(), "a running instance lost its font policy"


def test_apprun_falls_back_to_the_shipped_font_policy_when_it_cannot_write(tmp_path):
    """A policy that rejects host color fonts still beats no policy at all."""

    appdir = _apprun_mount(tmp_path)
    apprun = appdir / "AppRun"
    template = appdir / "usr/etc/fonts/unsloth-appimage.conf"
    unwritable = tmp_path / "unwritable"
    unwritable.mkdir(mode = 0o500)

    result = subprocess.run(
        [apprun],
        check = True,
        capture_output = True,
        text = True,
        env = {"PATH": "/usr/bin:/bin", "XDG_RUNTIME_DIR": str(unwritable / "state")},
    )
    assert f"FONTCONFIG_FILE={template}" in result.stdout.splitlines()


def test_complete_appimage_verifier_rejects_a_launcher_that_keeps_the_host_library_path(tmp_path):
    appdir = _fake_complete_appdir(tmp_path)
    apprun = appdir / "AppRun"
    apprun.write_text(
        apprun.read_text(encoding = "utf-8").replace("unset LD_LIBRARY_PATH\n", ""),
        encoding = "utf-8",
    )
    result = subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "inherited LD_LIBRARY_PATH" in result.stderr


def test_complete_appimage_verifier_rejects_additive_host_gio_modules(tmp_path):
    appdir = _fake_complete_appdir(tmp_path)
    hook = appdir / "apprun-hooks/linuxdeploy-plugin-gtk.sh"
    hook.write_text('export GIO_MODULE_DIR="$APPDIR/usr/lib/gio/modules"\n', encoding = "utf-8")
    result = subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "host GIO_EXTRA_MODULES" in result.stderr


def test_complete_appimage_verifier_requires_webkit_and_rejects_host_abi_libraries(tmp_path):
    missing = _fake_complete_appdir(tmp_path / "missing")
    (missing / "usr/lib/libwebkit2gtk-4.1.so.0").unlink()
    result = subprocess.run(
        [VERIFIER, "--appdir", missing], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "libwebkit2gtk-4.1.so" in result.stderr

    for library in (
        "libc.so.6",
        "libwayland-client.so.0",
        "libwayland-cursor.so.0",
        "libwayland-egl.so.1",
        "libwayland-server.so.0",
        "libGLESv2.so.2",
        "libnghttp2.so.14",
        "libcurl-gnutls.so.4",
        "libstdc++.so.6",
        "libgcc_s.so.1",
    ):
        bundled = _fake_complete_appdir(tmp_path / library)
        (bundled / "usr/lib" / library).touch()
        result = subprocess.run(
            [VERIFIER, "--appdir", bundled],
            check = False,
            capture_output = True,
            text = True,
        )
        assert result.returncode != 0
        assert "host runtime component" in result.stderr


def test_managed_appimage_children_preserve_host_library_paths():
    source_root = REPO_ROOT / "studio/src-tauri/src"
    process_source = (source_root / "process.rs").read_text(encoding = "utf-8")
    child_process_calls = {
        source_root / "commands.rs": ("scrub_appimage_python_env_tokio(&mut cmd)", 1),
        source_root / "desktop_auth.rs": ("scrub_appimage_python_env_tokio(&mut cmd)", 1),
        source_root / "install.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
        source_root / "preflight/managed.rs": ("scrub_appimage_python_env_tokio(&mut cmd)", 2),
        # One spawn path plus the unit tests that live in the same file.
        source_root / "process.rs": ("scrub_appimage_python_env(&mut cmd)", 4),
        source_root / "update.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
    }
    assert "scrub_appimage_library_path" in process_source
    assert "split_paths" in process_source
    assert "starts_with(&appdir)" in process_source
    assert 'cmd.env_remove("PYTHONHOME")' in process_source
    assert 'cmd.env_remove("PYTHONPATH")' in process_source

    # The AppRun parks the host value under the name process.rs restores it from.
    assert "UNSLOTH_HOST_LD_LIBRARY_PATH" in APPRUN.read_text(encoding = "utf-8")
    assert "UNSLOTH_HOST_LD_LIBRARY_PATH" in process_source

    production_source = process_source.split('#[cfg(all(test, target_os = "linux"))]', 1)[0]
    assert production_source.count("for name in APPIMAGE_GUI_ONLY_VARS") == 3
    for source_path, (call, expected) in child_process_calls.items():
        assert source_path.read_text(encoding = "utf-8").count(call) == expected


def test_the_deb_ships_the_polkit_action_in_app_debian_updates_authenticate_with():
    # Only this files mapping installs the polkit action; dropping it silently breaks updates.
    config = json.loads(
        (REPO_ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    )
    assert "deb" in config["bundle"]["targets"]
    files = config["bundle"]["linux"]["deb"]["files"]
    action = "/usr/share/polkit-1/actions/ai.unsloth.studio.update.policy"
    assert files[action].endswith("/ai.unsloth.studio.update.policy")

    policy = REPO_ROOT / "studio/src-tauri/linux" / files[action].removeprefix("./linux/")
    source = policy.read_text(encoding = "utf-8")
    assert '<action id="ai.unsloth.studio.update">' in source
    # auth_admin on all three implicit cases is what prompts instead of elevating silently.
    assert source.count("auth_admin") == 3
    # polkit pins the program and its first argument, so the action cannot run anything else.
    assert (
        '<annotate key="org.freedesktop.policykit.exec.path">/usr/bin/unsloth-studio</annotate>'
        in source
    )
    assert (
        '<annotate key="org.freedesktop.policykit.exec.argv1">--install-debian-update</annotate>'
        in source
    )
    # These pins must match tauri's crate-derived name and main.rs, or polkit refuses updates.
    cargo = (REPO_ROOT / "studio/src-tauri/Cargo.toml").read_text(encoding = "utf-8")
    assert 'name = "unsloth-studio"' in cargo
    debian_update = (REPO_ROOT / "studio/src-tauri/src/debian_update.rs").read_text(
        encoding = "utf-8"
    )
    assert 'const INSTALL_ARGUMENT: &str = "--install-debian-update";' in debian_update
    assert 'const INSTALLED_BINARY: &str = "/usr/bin/unsloth-studio";' in debian_update


def test_release_notes_recommend_native_deb_without_claiming_universality():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "use `.deb` when available" in notes
    assert "universal" not in notes.lower()


# Every variable the hooks point into the bundle; host mode must not hand any of them to host GTK.
_HOOK_PINNED = (
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

_MODE_LINE = re.compile(r"^unsloth-studio: AppImage WebKitGTK runtime=(host|bundled) \((.*)\)$")


def _launch(
    appdir: Path,
    tmp_path: Path,
    args = (),
    **env,
):
    """Run the mount like the AppImage runtime does: APPDIR set, AppRun as the entry point."""

    base = {
        "PATH": "/usr/bin:/bin",
        "APPDIR": str(appdir),
        "XDG_RUNTIME_DIR": str(tmp_path / "state"),
        "FAKE_PROBE_MARKER": str(tmp_path / "probe-ran"),
    }
    base.update({key: value for key, value in env.items() if value is not None})
    for key, value in env.items():
        if value is None:
            base.pop(key, None)
    entry = appdir / "AppRun"
    hooked = (appdir / "AppRun.wrapped").exists()
    command = [entry, *args] if hooked else [*_SH, entry, *args]
    started = time.monotonic()
    result = subprocess.run(command, check = True, capture_output = True, text = True, env = base)
    lines = result.stdout.splitlines()
    exe = lines[0].removeprefix("EXE=")
    cwd = lines[1].removeprefix("CWD=")
    argv = lines[2].removeprefix("ARGV=")
    child_env = dict(line.split("=", 1) for line in lines[3:] if "=" in line)
    mode_lines = [line for line in result.stderr.splitlines() if _MODE_LINE.match(line)]
    return {
        "exe": exe,
        "cwd": cwd,
        "argv": argv,
        "env": child_env,
        "stderr": result.stderr,
        "mode": _MODE_LINE.match(mode_lines[0]).groups() if len(mode_lines) == 1 else None,
        "elapsed": time.monotonic() - started,
        "probed": (tmp_path / "probe-ran").exists(),
    }


def _assert_host_env_restored(run, appdir: Path):
    env = run["env"]
    for name in _HOOK_PINNED:
        assert name not in env, f"host mode kept the bundle's {name}={env[name]}"
    for name, value in env.items():
        if str(appdir) in value:
            assert name in {"APPDIR", "PATH", "PWD", "OLDPWD"}, f"{name}={value} names the mount"
    assert "FONTCONFIG_FILE" not in env, "host mode pinned the Jammy fontconfig policy"
    assert env["UNSLOTH_APPIMAGE_WEBKIT_MODE"] == "host"


@pytest.mark.parametrize(
    ("inherited", "restored"),
    [(None, None), ("/a:/b", "/a:/b")],
    ids = ["xdg-unset", "xdg-set"],
)
def test_apprun_runs_the_host_copy_with_the_hook_environment_undone(tmp_path, inherited, restored):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)
    run = _launch(
        appdir, tmp_path, ["--flag", "x"], XDG_DATA_DIRS = inherited, LD_LIBRARY_PATH = "/opt/lib"
    )

    assert run["mode"] == ("host", "host WebKitGTK 2.54.2, UNSLOTH_APPIMAGE_WEBKIT=auto")
    assert run["exe"] == "host"
    assert run["cwd"] == str(appdir.resolve() / "usr")
    assert run["argv"] == "--flag x"
    assert "probe noise" not in run["stderr"]
    _assert_host_env_restored(run, appdir)
    assert run["env"].get("XDG_DATA_DIRS") == restored
    # The hook forced Adwaita; nobody asked for it, so host GTK follows the desktop's theme.
    assert "GTK_THEME" not in run["env"]
    assert "LD_LIBRARY_PATH" not in run["env"]
    assert run["env"]["UNSLOTH_HOST_LD_LIBRARY_PATH"] == "/opt/lib"
    assert not (tmp_path / "state/unsloth-studio").exists()

    # The probe saw the same cleaned environment from /, with no library path.
    probe = (tmp_path / "probe-ran").read_text(encoding = "utf-8").splitlines()
    assert probe[0] == "cwd=/"
    probe_env = dict(line.split("=", 1) for line in probe[1:] if "=" in line)
    assert "LD_LIBRARY_PATH" not in probe_env
    for name in (*_HOOK_PINNED, "GTK_THEME"):
        assert name not in probe_env
    assert probe_env.get("XDG_DATA_DIRS") == restored


def test_apprun_host_mode_keeps_an_explicit_theme_and_survives_a_relaunch(tmp_path):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)
    # A bundled-mode parent restarting through $APPIMAGE passes its hooked XDG_DATA_DIRS back in.
    relaunched = f"{appdir}/usr/share:/usr/share:/a:/b"
    run = _launch(
        appdir,
        tmp_path,
        XDG_DATA_DIRS = relaunched,
        APPIMAGE_GTK_THEME = "Breeze:dark",
        UNSLOTH_APPIMAGE_WEBKIT_MODE = "bundled",
    )
    assert run["mode"][0] == "host"
    _assert_host_env_restored(run, appdir)
    assert run["env"]["XDG_DATA_DIRS"] == "/a:/b"
    assert run["env"]["GTK_THEME"] == "Breeze:dark"

    stray = _launch(appdir, tmp_path, XDG_DATA_DIRS = f"/a:{appdir}/usr/share/extra:/b:{appdir}")
    assert stray["env"]["XDG_DATA_DIRS"] == "/a:/b"


@pytest.mark.parametrize(
    ("knobs", "reason"),
    [
        ({"FAKE_PROBE_OUTPUT": "webkit=2.52.6\\n"}, "reason: version 2.52.6 < 2.54"),
        ({"FAKE_PROBE_OUTPUT": "webkit=1.99.0\\n"}, "reason: version 1.99.0 < 2.54"),
        ({"FAKE_PROBE_EXIT": "3"}, "reason: probe exit 3"),
        ({"FAKE_PROBE_OUTPUT": "", "FAKE_PROBE_EXIT": "127"}, "reason: probe exit 127"),
        ({"FAKE_PROBE_OUTPUT": "webkit=2.54\\n"}, "reason: unrecognized probe output"),
        ({"FAKE_PROBE_OUTPUT": "webkit=2.54.2x\\n"}, "reason: unrecognized probe output"),
        ({"FAKE_PROBE_OUTPUT": "webkit=12345.0.0\\n"}, "reason: unrecognized probe output"),
        ({"FAKE_PROBE_OUTPUT": "WebKit 2.54.2\\n"}, "reason: unrecognized probe output"),
        (
            {"FAKE_PROBE_OUTPUT": "webkit=2.54.2\\nwebkit=2.54.2\\n"},
            "reason: unrecognized probe output",
        ),
        ({"FAKE_PROBE_OUTPUT": ""}, "reason: unrecognized probe output"),
    ],
    ids = [
        "2.52.6",
        "1.x",
        "exit-3",
        "exit-127",
        "two-part",
        "suffix",
        "huge",
        "garbage",
        "two-lines",
        "empty",
    ],
)
def test_apprun_falls_back_to_the_bundle_when_the_probe_does_not_prove_the_host(
    tmp_path, knobs, reason
):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)
    run = _launch(appdir, tmp_path, **knobs)
    assert run["probed"]
    assert run["mode"] == ("bundled", reason)
    assert run["exe"] == "bundled"
    assert run["env"]["UNSLOTH_APPIMAGE_WEBKIT_MODE"] == "bundled"
    assert run["env"]["GTK_PATH"].startswith(str(appdir))


def test_apprun_gives_a_hung_probe_five_seconds(tmp_path):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)
    run = _launch(appdir, tmp_path, FAKE_PROBE_SLEEP = "30")
    assert run["mode"] == ("bundled", "reason: probe exit 124")
    assert run["exe"] == "bundled"
    assert run["elapsed"] < 7


def test_apprun_needs_a_host_copy_whatever_the_override(tmp_path):
    appdir = _apprun_mount(tmp_path, hooked = True)
    for request in (None, "auto", "host", "bundled"):
        run = _launch(appdir, tmp_path, UNSLOTH_APPIMAGE_WEBKIT = request)
        assert run["mode"] == ("bundled", "reason: no host copy")
        assert run["exe"] == "bundled"


def test_apprun_overrides_skip_the_probe(tmp_path):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)

    bundled = _launch(appdir, tmp_path, UNSLOTH_APPIMAGE_WEBKIT = "bundled")
    assert bundled["mode"] == ("bundled", "reason: override")
    assert bundled["exe"] == "bundled"
    assert not bundled["probed"]

    # Forced host runs the host copy even where auto would refuse it.
    forced = _launch(
        appdir,
        tmp_path,
        UNSLOTH_APPIMAGE_WEBKIT = "host",
        FAKE_PROBE_OUTPUT = "webkit=2.52.6\\n",
        LD_LIBRARY_PATH = "/opt/lib",
    )
    assert forced["mode"] == ("host", "override, UNSLOTH_APPIMAGE_WEBKIT=host")
    assert forced["exe"] == "host"
    assert not forced["probed"]
    _assert_host_env_restored(forced, appdir)
    assert "LD_LIBRARY_PATH" not in forced["env"]

    empty = _launch(appdir, tmp_path, UNSLOTH_APPIMAGE_WEBKIT = "")
    assert empty["mode"][0] == "host" and empty["probed"]


@pytest.mark.parametrize("request_value", ["HOST", "x", "host\nruntime=host", "$(id)"])
def test_apprun_treats_an_unknown_override_as_bundled(tmp_path, request_value):
    appdir = _apprun_mount(tmp_path, host_copy = True, hooked = True)
    run = _launch(appdir, tmp_path, UNSLOTH_APPIMAGE_WEBKIT = request_value)
    assert run["exe"] == "bundled"
    assert not run["probed"]
    mode, detail = run["mode"]
    assert mode == "bundled"
    assert re.fullmatch(r"reason: invalid override '[A-Za-z0-9._?-]*'", detail)
    assert run["stderr"].count("\n") == 1


def test_apprun_overwrites_an_inherited_mode_in_bundled_mode(tmp_path):
    appdir = _apprun_mount(tmp_path)
    run = _launch(appdir, tmp_path, UNSLOTH_APPIMAGE_WEBKIT_MODE = "host")
    assert run["mode"] == ("bundled", "reason: no host copy")
    assert run["env"]["UNSLOTH_APPIMAGE_WEBKIT_MODE"] == "bundled"


MAIN_APPRUN_REVISION = "a1064bb8e6dbbd527761436605c20cdb59cb2a94"


@pytest.mark.parametrize(
    ("host_copy", "knobs", "reason"),
    [
        (False, {}, "reason: no host copy"),
        (True, {"FAKE_PROBE_OUTPUT": "", "FAKE_PROBE_EXIT": "127"}, "reason: probe exit 127"),
        (True, {"FAKE_PROBE_OUTPUT": "webkit=2.52.6\\n"}, "reason: version 2.52.6 < 2.54"),
        (True, {"UNSLOTH_APPIMAGE_WEBKIT": "bundled"}, "reason: override"),
        (True, {"UNSLOTH_APPIMAGE_WEBKIT": "x"}, "reason: invalid override 'x'"),
    ],
    ids = ["no-copy", "probe-127", "2.52", "override", "invalid"],
)
def test_bundled_mode_matches_the_launcher_before_host_mode(tmp_path, host_copy, knobs, reason):
    """Bundled mode adds only UNSLOTH_APPIMAGE_WEBKIT_MODE and its one stderr line."""

    shown = subprocess.run(
        [
            "git",
            "-C",
            REPO_ROOT,
            "show",
            f"{MAIN_APPRUN_REVISION}:studio/src-tauri/linux/appimage-apprun.sh",
        ],
        capture_output = True,
    )
    if shown.returncode != 0:
        pytest.skip(f"{MAIN_APPRUN_REVISION} is not in this clone")
    appdir = _apprun_mount(tmp_path, host_copy = host_copy, hooked = True)
    knobs = {"LD_LIBRARY_PATH": "/opt/lib", "XDG_DATA_DIRS": "/a:/b", **knobs}

    current = _launch(appdir, tmp_path, ["--x", "y z"], **knobs)
    (appdir / "AppRun.wrapped").write_bytes(shown.stdout)
    before = _launch(appdir, tmp_path, ["--x", "y z"], **knobs)

    assert current["mode"] == ("bundled", reason)
    assert current["stderr"].splitlines() == [
        f"unsloth-studio: AppImage WebKitGTK runtime=bundled ({reason})"
    ]
    assert before["stderr"] == ""
    for key in ("exe", "cwd", "argv"):
        assert current[key] == before[key]
    added = {k: v for k, v in current["env"].items() if before["env"].get(k) != v}
    removed = set(before["env"]) - set(current["env"])
    assert added == {"UNSLOTH_APPIMAGE_WEBKIT_MODE": "bundled"}
    assert not removed


def _host_capable_appdir(tmp_path: Path) -> Path:
    """A verifier fixture whose real launcher can run a host-mode copy."""

    appdir = _fake_complete_appdir(tmp_path)
    (appdir / "AppRun.wrapped").write_bytes(APPRUN.read_bytes())
    host = appdir / "usr/bin/host/unsloth-studio"
    host.parent.mkdir()
    shutil.copy2(appdir / "usr/bin/unsloth-studio", host)
    return appdir


def _verify(appdir: Path):
    return subprocess.run(
        [VERIFIER, "--appdir", appdir], check = False, capture_output = True, text = True
    )


def test_complete_appimage_verifier_accepts_a_host_capable_runtime_and_the_shipped_layout(tmp_path):
    result = _verify(_host_capable_appdir(tmp_path / "host"))
    assert result.returncode == 0, result.stderr

    # Shipped releases have neither the copy nor a launcher that runs it.
    shipped = _fake_complete_appdir(tmp_path / "shipped")
    assert "usr/bin/host" not in (shipped / "AppRun").read_text(encoding = "utf-8")
    result = _verify(shipped)
    assert result.returncode == 0, result.stderr


def test_complete_appimage_verifier_rejects_an_incoherent_host_copy(tmp_path):
    differs = _host_capable_appdir(tmp_path / "differs")
    with open(differs / "usr/bin/host/unsloth-studio", "ab") as host:
        host.write(b"\0")
    result = _verify(differs)
    assert result.returncode != 0
    assert "differs from usr/bin/unsloth-studio" in result.stderr

    reaches = _host_capable_appdir(tmp_path / "reaches")
    (reaches / "usr/bin/lib").mkdir()
    result = _verify(reaches)
    assert result.returncode != 0
    assert "usr/bin/lib" in result.stderr

    missing = _host_capable_appdir(tmp_path / "missing")
    shutil.rmtree(missing / "usr/bin/host")
    result = _verify(missing)
    assert result.returncode != 0
    assert "missing host-mode executable" in result.stderr

    unprobed = _host_capable_appdir(tmp_path / "unprobed")
    launcher = unprobed / "AppRun.wrapped"
    launcher.write_text(
        launcher.read_text(encoding = "utf-8").replace("--unsloth-webkit-probe", "--no-probe"),
        encoding = "utf-8",
    )
    result = _verify(unprobed)
    assert result.returncode != 0
    assert "without the WebKitGTK probe" in result.stderr


def test_complete_appimage_verifier_rejects_a_hook_export_host_mode_does_not_undo(tmp_path):
    appdir = _host_capable_appdir(tmp_path)
    hook = appdir / "apprun-hooks/linuxdeploy-plugin-gstreamer.sh"
    hook.write_text(
        'export GST_REGISTRY_REUSE_PLUGIN_SCANNER="no"\n'
        'export GST_PLUGIN_PATH_1_0="${APPDIR}/usr/lib/gstreamer-1.0"\n'
        'export GST_PLUGIN_FEATURE_RANK_PATH="${APPDIR}/usr/lib/ranks"\n',
        encoding = "utf-8",
    )
    result = _verify(appdir)
    assert result.returncode != 0
    assert "GST_PLUGIN_FEATURE_RANK_PATH" in result.stderr
    assert "GST_PLUGIN_PATH_1_0" not in result.stderr


def _finalizer_tools(tmp_path: Path) -> Path:
    """The finalizer beside the assets prepare-complete-appimage-tools.sh stages next to it."""

    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(FINALIZER, tools / FINALIZER.name)
    (tools / "UnslothSafeEmoji.ttf").write_bytes(b"fixture CBDT CBLC")
    (tools / "UnslothSafeEmoji.LICENSE").write_text("fixture\n", encoding = "utf-8")
    shutil.copy2(FONTCONFIG, tools / "unsloth-appimage-fonts.conf")
    return tools / FINALIZER.name


def _runpath(path: Path) -> str:
    dynamic = subprocess.run(
        ["readelf", "-d", path], check = True, capture_output = True, text = True
    ).stdout
    return re.search(r"\((?:RUNPATH|RPATH)\).*\[(.*)\]", dynamic).group(1)


@pytest.mark.skipif(shutil.which("patchelf") is None, reason = "the finalizer needs patchelf")
def test_finalizer_adds_an_identical_host_copy_and_stays_idempotent(tmp_path):
    finalizer = _finalizer_tools(tmp_path)
    appdir = tmp_path / "AppDir"
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    _compile_fixture_elf(binary, origin_runpath = False)
    (appdir / "usr/lib").mkdir()
    _compile_fixture_elf(appdir / "usr/lib/WebKitWebProcess", origin_runpath = False)
    host = appdir / "usr/bin/host/unsloth-studio"

    # linuxdeploy runs the finalizer once from each of the gtk and gstreamer plugins.
    for _ in range(2):
        subprocess.run([finalizer, appdir], check = True, capture_output = True, text = True)
        assert host.read_bytes() == binary.read_bytes()
        assert os.access(host, os.X_OK)
        assert _runpath(host) == "$ORIGIN/../lib"
        assert _runpath(binary) == "$ORIGIN/../lib"
        assert _runpath(appdir / "usr/lib/WebKitWebProcess") == "$ORIGIN"
        assert not (appdir / "usr/bin/lib").exists()

    (appdir / "usr/bin/lib").mkdir()
    result = subprocess.run([finalizer, appdir], check = False, capture_output = True, text = True)
    assert result.returncode != 0
    assert "usr/bin/lib" in result.stderr


def test_apprun_never_probes_without_a_timeout(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    for tool in ("dirname", "sed", "mkdir", "head", "rm", "tr", "cut"):
        (tools / tool).symlink_to(shutil.which(tool))
    appdir = _apprun_mount(tmp_path, host_copy = True)
    run = _launch(appdir, tmp_path, PATH = str(tools))
    assert run["mode"] == ("bundled", "reason: no timeout")
    assert run["exe"] == "bundled"
    assert not run["probed"]
