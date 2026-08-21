"""Contracts for the complete Linux AppImage release path."""

import json
import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

CLEAN_MACHINE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "desktop-app-clean-machine-ci.yml"
VERIFIER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "verify-complete-appimage.sh"

FINALIZER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "finalize-complete-appimage.sh"

APPRUN = REPO_ROOT / "studio" / "src-tauri" / "linux" / "appimage-apprun.sh"


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

    # Require plugins compatible with the bundled GStreamer core.
    dependencies = _step("Install Linux dependencies")["run"]
    for package in ("gstreamer1.0-plugins-good", "gstreamer1.0-plugins-bad", "gstreamer1.0-libav"):
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
    for package in ("gstreamer1.0-plugins-good", "gstreamer1.0-plugins-bad", "gstreamer1.0-libav"):
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

    # Probe plugin loadability on every target host.
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
    # Provide the host audio libraries used by bundled media plugins.
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
    for local_tool in ("appimage-apprun.sh", "finalize-complete-appimage.sh"):
        assert local_tool in tool_script

    assert "patchelf --set-rpath" in finalizer_source
    assert "$ORIGIN" in finalizer_source
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

    # Keep foreign GIO and GTK modules out of the bundled runtime.
    assert "-path '*/gio/modules/*' -type f -print0" in tool_script
    assert 'export GTK_PATH="\\$APPDIR/' in tool_script

    # Run the finalizer regardless of linuxdeploy plugin order.
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

    # WebKit's media pipeline is the bundled GStreamer core plus these plugins.
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
    return appdir


def _write_foreign_arch_elf(path: Path) -> None:
    """An i386 ELF header, the shape a multilib build host contributes."""

    header = bytearray(52)
    header[0:8] = b"\x7fELF\x01\x01\x01\x00"
    header[16:18] = (3).to_bytes(2, "little")  # e_type = ET_DYN
    header[18:20] = (3).to_bytes(2, "little")  # e_machine = EM_386
    header[20:24] = (1).to_bytes(4, "little")  # e_version
    header[40:42] = (52).to_bytes(2, "little")  # e_ehsize
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


def test_apprun_hands_an_inherited_library_path_to_children_only(tmp_path):
    """The loader reads LD_LIBRARY_PATH before the bundle's own $ORIGIN RUNPATHs."""

    appdir = tmp_path / "AppDir"
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    binary.write_text("#!/bin/sh\nexec /usr/bin/env\n", encoding = "utf-8")
    binary.chmod(0o755)
    apprun = appdir / "AppRun"
    apprun.write_bytes(APPRUN.read_bytes())
    apprun.chmod(0o755)

    result = subprocess.run(
        [apprun],
        check = True,
        capture_output = True,
        text = True,
        env = {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/opt/conda/lib:/opt/rocm/lib"},
    )
    printed = result.stdout.splitlines()
    assert not [line for line in printed if line.startswith("LD_LIBRARY_PATH=")]
    assert "UNSLOTH_HOST_LD_LIBRARY_PATH=/opt/conda/lib:/opt/rocm/lib" in printed


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

    # Cover std children, Tokio children, and host launchers.
    production_source = process_source.split('#[cfg(all(test, target_os = "linux"))]', 1)[0]
    assert production_source.count("for name in APPIMAGE_GUI_ONLY_VARS") == 3
    for source_path, (call, expected) in child_process_calls.items():
        assert source_path.read_text(encoding = "utf-8").count(call) == expected


def test_release_notes_recommend_native_deb_without_claiming_universality():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "use `.deb` when available" in notes
    assert "universal" not in notes.lower()
