"""Contracts for the complete Linux AppImage release path."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

CLEAN_MACHINE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "desktop-app-clean-machine-ci.yml"
VERIFIER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "verify-complete-appimage.sh"


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
    assert appimage["bundleMediaFramework"] is False
    assert appimage["files"]["/usr/lib/libappindicator3.so.1"].endswith("/libappindicator3.so.1")

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
        "libharfbuzz0b",
        "libnghttp2-14",
    ):
        assert package in source
    assert "weston" in source
    assert "APPIMAGE_DISPLAY_BACKEND" in source
    assert "wayland" in source

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

    for package in ("libwayland-client", "libnghttp2"):
        assert package in source


def test_release_preseeds_every_tauri_appimage_tool_with_a_digest():
    step = _step("Pin complete AppImage toolchain")
    assert step["if"] == "matrix.platform == 'ubuntu-22.04'"
    assert "prepare-complete-appimage-tools.sh" in step["run"]
    tool_script = (
        REPO_ROOT / "studio/src-tauri/linux/prepare-complete-appimage-tools.sh"
    ).read_text(encoding = "utf-8")
    expected = {
        "APPRUN": ("AppRun-x86_64", "AppRun-x86_64"),
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
    for host_library in (
        "libwayland-client.so*",
        "libnghttp2.so*",
        "libcurl*.so*",
        "libstdc++.so*",
        "libgcc_s.so*",
    ):
        assert host_library in tool_script
    assert "GIO_MODULE_DIR" in tool_script

    assert "unset GIO_EXTRA_MODULES" in tool_script

    assert "sed -i '/export GDK_BACKEND=x11/d'" in tool_script


def _fake_complete_appdir(tmp_path: Path) -> Path:
    appdir = tmp_path / "AppDir"
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    shutil.copy2("/bin/true", binary)

    (appdir / "Unsloth.png").touch()
    (appdir / ".DirIcon").symlink_to("Unsloth.png")
    apprun = appdir / "AppRun"
    apprun.write_text(
        '#!/bin/sh\n. "$APPDIR/apprun-hooks/linuxdeploy-plugin-gtk.sh"\nexit 0\n',
        encoding = "utf-8",
    )
    apprun.chmod(0o755)
    hook = appdir / "apprun-hooks/linuxdeploy-plugin-gtk.sh"
    hook.parent.mkdir()
    hook.write_text(
        'unset GIO_EXTRA_MODULES\nexport GIO_MODULE_DIR="$APPDIR/usr/lib/gio/modules"\n',
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
    return appdir


def test_complete_appimage_verifier_accepts_a_coherent_runtime(tmp_path):
    result = subprocess.run(
        [VERIFIER, "--appdir", _fake_complete_appdir(tmp_path)],
        check = True,
        capture_output = True,
        text = True,
    )
    assert "Verified complete x86_64 AppImage runtime" in result.stdout


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
        source_root / "process.rs": ("scrub_appimage_python_env(&mut cmd)", 3),
        source_root / "update.rs": ("scrub_appimage_python_env(&mut cmd)", 1),
    }
    assert "scrub_appimage_library_path" in process_source
    assert "split_paths" in process_source
    assert "starts_with(&appdir)" in process_source
    assert 'cmd.env_remove("PYTHONHOME")' in process_source
    assert 'cmd.env_remove("PYTHONPATH")' in process_source

    # One cleanup each for std managed children, Tokio managed children, and host launchers.
    production_source = process_source.split('#[cfg(all(test, target_os = "linux"))]', 1)[0]
    assert production_source.count("for name in APPIMAGE_GUI_ONLY_VARS") == 3
    for source_path, (call, expected) in child_process_calls.items():
        assert source_path.read_text(encoding = "utf-8").count(call) == expected


def test_release_notes_recommend_native_deb_without_claiming_universality():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "use `.deb` when available" in notes
    assert "universal" not in notes.lower()
