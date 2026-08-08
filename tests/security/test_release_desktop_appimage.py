"""Contracts for the host-integrated Linux AppImage release path."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
PACKAGER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "build-thin-appimage.sh"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _step(name: str):
    steps = _workflow()["jobs"]["build"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_release_pins_the_appimage_builder_and_runtime_by_digest():
    step = _step("Pin thin AppImage toolchain")
    assert step["if"] == "matrix.platform == 'ubuntu-22.04'"
    assert "/AppImage/appimagetool/releases/download/1.9.1/" in step["env"]["APPIMAGETOOL_URL"]
    assert len(step["env"]["APPIMAGETOOL_SHA256"]) == 64
    assert (
        "/AppImage/type2-runtime/releases/download/20251108/" in step["env"]["APPIMAGE_RUNTIME_URL"]
    )
    assert len(step["env"]["APPIMAGE_RUNTIME_SHA256"]) == 64
    assert step["run"].count("sha256sum -c") == 2


def test_linux_build_repackages_the_deb_and_signs_the_final_appimage():
    build = _step("Build Linux deb")
    package = _step("Build and sign thin Linux AppImage")
    stage = _step("Stage release assets")

    assert "--bundles deb" in build["with"]["args"]
    assert '"createUpdaterArtifacts":false' in build["with"]["args"]
    assert "TAURI_SIGNING_PRIVATE_KEY" not in build.get("env", {})
    assert "build-thin-appimage.sh" in package["run"]
    assert 'tauri signer sign "$appimage" > "$signature"' in package["run"]
    assert '[[ ! -s "$signature" ]]' in package["run"]
    assert "f'{appimage}.sig'" in package["run"]
    assert package["env"]["ARTIFACT_PATHS"] == "${{ steps.build_linux.outputs.artifactPaths }}"
    assert stage["env"]["ARTIFACT_PATHS"].startswith(
        "${{ steps.package_linux.outputs.artifactPaths"
    )


def test_tauri_cannot_build_the_old_bundled_appimage():
    config = yaml.safe_load(
        (REPO_ROOT / "studio" / "src-tauri" / "tauri.conf.json").read_text(encoding = "utf-8")
    )
    assert "appimage" not in config["bundle"]["targets"]
    assert "appimage" not in config["bundle"]["linux"]


def test_thin_appimage_never_injects_a_partial_desktop_runtime():
    script = PACKAGER.read_text(encoding = "utf-8")
    assert "dpkg-deb --extract" in script
    assert "--runtime-file" in script
    assert "linuxdeploy" not in script
    assert "export LD_LIBRARY_PATH" not in script
    for library in (
        "libglib-2.0.so",
        "libgio-2.0.so",
        "libgtk-3.so",
        "libwebkit2gtk-4.1.so",
        "libcurl.so",
        "libnghttp2.so",
    ):
        assert library in script
    assert "sudo apt install libayatana-appindicator3-1 libwebkit2gtk-4.1-0 libgtk-3-0" in script
    assert "libayatana-appindicator3.so.1 libappindicator3.so.1" in script
    assert "libayatana-appindicator3.so libappindicator3.so" in script
    assert "zenity --error" in script
    assert "xmessage -center" in script


def test_thin_appimage_rejects_a_bundled_host_library(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "libunsloth-test.so").touch()
    subprocess.run(
        [PACKAGER, "--verify-appdir", allowed],
        check = True,
        capture_output = True,
        text = True,
    )

    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    (forbidden / "libglib-2.0.so.0").touch()
    result = subprocess.run(
        [PACKAGER, "--verify-appdir", forbidden],
        check = False,
        capture_output = True,
        text = True,
    )

    assert result.returncode != 0
    assert "libglib-2.0.so.0" in result.stderr


def _build_fake_appimage(
    tmp_path,
    *,
    bundle_markers = 1,
    check = True,
):
    package_root = tmp_path / "package"
    control_dir = package_root / "DEBIAN"
    binary_dir = package_root / "usr" / "bin"
    desktop_dir = package_root / "usr" / "share" / "applications"
    icon_dir = package_root / "usr" / "share" / "icons" / "hicolor" / "128x128" / "apps"
    for directory in (control_dir, binary_dir, desktop_dir, icon_dir):
        directory.mkdir(parents = True, exist_ok = True)

    (control_dir / "control").write_text(
        "Package: unsloth-test\nVersion: 1.0\nArchitecture: amd64\nDescription: test\n",
        encoding = "utf-8",
    )
    shutil.copy2("/bin/sh", binary_dir / "unsloth-studio")
    with (binary_dir / "unsloth-studio").open("ab") as binary:
        binary.write(b"__TAURI_BUNDLE_TYPE_VAR_DEB" * bundle_markers)
    (desktop_dir / "Unsloth.desktop").write_text("[Desktop Entry]\n", encoding = "utf-8")
    (icon_dir / "unsloth-studio.png").write_bytes(b"test")

    deb_path = tmp_path / "unsloth-test.deb"
    subprocess.run(
        ["dpkg-deb", "--build", "--root-owner-group", package_root, deb_path],
        check = True,
        capture_output = True,
        text = True,
    )

    appimagetool = tmp_path / "appimagetool"
    appimagetool.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 2 ]]; do shift; done
app_dir="$1"
output_path="$2"
captured_app_dir="${output_path}.AppDir"
cp -a "$app_dir" "$captured_app_dir"
printf '#!/usr/bin/env bash\nset -euo pipefail\ncp -a %q squashfs-root\n' "$captured_app_dir" > "$output_path"
chmod +x "$output_path"
""",
        encoding = "utf-8",
    )
    appimagetool.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.touch()

    relative_output = Path("artifacts") / "unsloth-test.AppImage"
    result = subprocess.run(
        [PACKAGER, deb_path, appimagetool, runtime, relative_output],
        cwd = tmp_path,
        check = check,
        capture_output = True,
        text = True,
    )

    output_path = tmp_path / relative_output
    return output_path, Path(f"{output_path}.AppDir"), result


def test_thin_appimage_resolves_a_relative_output_before_verification(tmp_path):
    output_path, _, result = _build_fake_appimage(tmp_path)

    assert output_path.is_file()
    assert f"Built thin AppImage: {output_path}" in result.stdout


def test_thin_appimage_stamps_the_copied_deb_binary_as_an_appimage(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    binary = (app_dir / "usr" / "bin" / "unsloth-studio").read_bytes()

    assert b"__TAURI_BUNDLE_TYPE_VAR_DEB" not in binary
    assert binary.count(b"__TAURI_BUNDLE_TYPE_VAR_APP") == 1


def test_thin_appimage_rejects_an_unexpected_tauri_bundle_marker(tmp_path):
    for bundle_markers in (0, 2):
        _, _, result = _build_fake_appimage(
            tmp_path / str(bundle_markers),
            bundle_markers = bundle_markers,
            check = False,
        )

        assert result.returncode != 0
        assert "Expected exactly one Tauri deb bundle marker" in result.stderr


def test_thin_appimage_preserves_host_xdg_data_dirs(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    apprun = app_dir / "AppRun"
    command = [apprun, "-c", 'printf %s "$XDG_DATA_DIRS"']

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ldconfig = fake_bin / "ldconfig"
    ldconfig.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    ldconfig.chmod(0o755)
    host_libraries = tmp_path / "host-libraries"
    host_libraries.mkdir()
    (host_libraries / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")

    default_env = os.environ.copy()
    default_env.pop("XDG_DATA_DIRS", None)
    default_env["PATH"] = f"{fake_bin}:{default_env['PATH']}"
    default_env["LD_LIBRARY_PATH"] = str(host_libraries)
    default_result = subprocess.run(
        command,
        env = default_env,
        check = True,
        capture_output = True,
        text = True,
    )
    assert default_result.stdout == f"{app_dir}/usr/share:/usr/local/share:/usr/share"

    custom_env = {**default_env, "XDG_DATA_DIRS": "/opt/share:/srv/share"}
    custom_result = subprocess.run(
        command,
        env = custom_env,
        check = True,
        capture_output = True,
        text = True,
    )
    assert custom_result.stdout == f"{app_dir}/usr/share:/opt/share:/srv/share"


def _fake_library_host(tmp_path, ldconfig_lines = ()):
    """Stub the tools AppRun probes so the build machine's own libraries don't count.

    The AppImage binary always resolves; any other library resolves only when a
    test planted it under tmp_path. Without this the default-directory search
    would find the real AppIndicator on any developer or CI box.
    """
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ldd = fake_bin / "ldd"
    ldd.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  */usr/bin/unsloth-studio) exit 0 ;;\n"
        '  ./*) [ -e "$1" ] && exit 0 || exit 1 ;;\n'
        "  " + str(tmp_path) + '/*) [ -e "$1" ] && exit 0 || exit 1 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    cache = "".join(f"printf '%s\\n' {shlex.quote(line)}\n" for line in ldconfig_lines)
    (fake_bin / "ldconfig").write_text(f"#!/bin/sh\n{cache}exit 0\n", encoding = "utf-8")
    for program in ("zenity", "xmessage"):
        (fake_bin / program).write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    for stub in fake_bin.iterdir():
        stub.chmod(0o755)
    return fake_bin


def _run_apprun(
    app_dir,
    fake_bin,
    library_path,
    *,
    path = None,
):
    return subprocess.run(
        [app_dir / "AppRun", "-c", "exit 0"],
        env = {
            **os.environ,
            "APPDIR": str(app_dir),
            "LD_LIBRARY_PATH": str(library_path),
            "PATH": path or f"{fake_bin}:{os.environ['PATH']}",
        },
        check = False,
        capture_output = True,
        text = True,
    )


def test_thin_appimage_reports_a_missing_dynamic_tray_library(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    result = _run_apprun(app_dir, _fake_library_host(tmp_path), tmp_path / "missing-libraries")

    assert result.returncode == 127
    assert "libayatana-appindicator3.so.1 or libappindicator3.so.1" in result.stderr
    assert "sudo apt install libayatana-appindicator3-1" in result.stderr


# libappindicator-sys falls back from the versioned sonames to the unversioned
# ones, so rejecting those two would refuse a host whose tray works.
def test_thin_appimage_accepts_every_tray_library_name_the_loader_tries(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    fake_bin = _fake_library_host(tmp_path)

    for library_name in (
        "libayatana-appindicator3.so.1",
        "libappindicator3.so.1",
        "libayatana-appindicator3.so",
        "libappindicator3.so",
    ):
        library_dir = tmp_path / f"host-{library_name}"
        library_dir.mkdir()
        (library_dir / library_name).symlink_to("/bin/sh")

        result = _run_apprun(app_dir, fake_bin, library_dir)

        assert result.returncode == 0, f"{library_name}: {result.stderr}"


def test_thin_appimage_matches_loader_library_path_separators_and_empty_components(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    fake_bin = _fake_library_host(tmp_path)
    library_dir = tmp_path / "host-libraries"
    library_dir.mkdir()
    (library_dir / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")

    for library_path in (
        f"{tmp_path / 'missing'};{library_dir}",
        f"{tmp_path / 'missing'}::{library_dir}",
        f"{tmp_path / 'missing'}:",
    ):
        if library_path.endswith(":"):
            working_dir = tmp_path / "working-directory"
            working_dir.mkdir()
            (working_dir / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")
        else:
            working_dir = tmp_path

        result = subprocess.run(
            [app_dir / "AppRun", "-c", "exit 0"],
            cwd = working_dir,
            env = {
                **os.environ,
                "APPDIR": str(app_dir),
                "LD_LIBRARY_PATH": str(library_path),
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            },
            check = False,
            capture_output = True,
            text = True,
        )

        assert result.returncode == 0, f"{library_path}: {result.stderr}"


# The loader commits to the first file it finds for a name, so a broken copy
# ahead of a working one must fail rather than skip ahead.
def test_thin_appimage_stops_at_a_broken_library_path_candidate(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    fake_bin = _fake_library_host(tmp_path)

    broken_dir = tmp_path / "broken-libraries"
    broken_dir.mkdir()
    (broken_dir / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")
    working_dir = tmp_path / "working-libraries"
    working_dir.mkdir()
    (working_dir / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")

    ldd = fake_bin / "ldd"
    ldd.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  */usr/bin/unsloth-studio) exit 0 ;;\n"
        "  " + str(broken_dir) + "/*) printf '\\tlibmissing.so.0 => not found\\n'; exit 0 ;;\n"
        "  " + str(tmp_path) + '/*) [ -e "$1" ] && exit 0 || exit 1 ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    ldd.chmod(0o755)

    result = _run_apprun(app_dir, fake_bin, f"{broken_dir}:{working_dir}")

    assert result.returncode == 127
    assert "libayatana-appindicator3.so.1 or libappindicator3.so.1" in result.stderr


def test_thin_appimage_accepts_a_readable_tray_library_when_ldd_is_unavailable(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    fake_bin = _fake_library_host(tmp_path)
    (fake_bin / "ldd").unlink()
    sed = fake_bin / "sed"
    sed.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    sed.chmod(0o755)
    library_dir = tmp_path / "host-libraries"
    library_dir.mkdir()
    (library_dir / "libayatana-appindicator3.so.1").symlink_to("/bin/sh")

    result = _run_apprun(app_dir, fake_bin, library_dir, path = str(fake_bin))

    assert result.returncode == 0, result.stderr


# The cache is how a library outside LD_LIBRARY_PATH and the default directories
# gets found, so keep the ldconfig -p parsing covered.
def test_thin_appimage_finds_a_tray_library_through_the_ldconfig_cache(tmp_path):
    _, app_dir, _ = _build_fake_appimage(tmp_path)
    cached_dir = tmp_path / "cached-libraries"
    cached_dir.mkdir()
    cached_library = cached_dir / "libappindicator3.so.1"
    cached_library.symlink_to("/bin/sh")
    fake_bin = _fake_library_host(
        tmp_path,
        ldconfig_lines = (
            "\tlibunrelated.so.1 (libc6,x86-64) => /usr/lib/libunrelated.so.1",
            f"\tlibappindicator3.so.1 (libc6,x86-64) => {cached_library}",
        ),
    )

    result = _run_apprun(app_dir, fake_bin, tmp_path / "missing-libraries")

    assert result.returncode == 0, result.stderr


def test_release_notes_keep_existing_appimage_guidance():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "libwebkit2gtk" not in notes
    assert "Linux AppImage on Ubuntu 24.04+ may require: `sudo apt install libfuse2t64`" in notes
    assert "--appimage-extract-and-run" not in notes


def test_thin_appimage_preserves_host_libraries_and_scrubs_python_overrides():
    source_root = REPO_ROOT / "studio" / "src-tauri" / "src"
    process_source = (source_root / "process.rs").read_text(encoding = "utf-8")
    child_process_calls = {
        source_root / "commands.rs": ("scrub_appimage_python_env_tokio(&mut cmd)", 1),
        source_root / "desktop_auth.rs": ("scrub_appimage_python_env_tokio(&mut cmd)", 1),
        source_root / "install.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
        source_root / "preflight" / "managed.rs": (
            "scrub_appimage_python_env_tokio(&mut cmd)",
            2,
        ),
        source_root / "process.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
        source_root / "update.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
    }

    assert process_source.count('cmd.env_remove("PYTHONHOME")') == 2
    assert process_source.count('cmd.env_remove("PYTHONPATH")') == 2

    for source_path, (call, expected) in child_process_calls.items():
        source = source_path.read_text(encoding = "utf-8")
        assert source.count(call) == expected, source_path
        assert 'cmd.env_remove("LD_LIBRARY_PATH")' not in source, source_path
