"""Contracts for the complete Linux AppImage release path."""

import json
import shutil
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
VERIFIER = REPO_ROOT / "studio" / "src-tauri" / "linux" / "verify-complete-appimage.sh"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _step(name: str):
    return next(
        step for step in _workflow()["jobs"]["build"]["steps"] if step.get("name") == name
    )


def test_tauri_builds_and_signs_deb_and_complete_appimage_together():
    config = json.loads(
        (REPO_ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    )
    assert "appimage" in config["bundle"]["targets"]
    appimage = config["bundle"]["linux"]["appimage"]
    assert appimage["bundleMediaFramework"] is False
    assert appimage["files"]["/usr/lib/libappindicator3.so.1"].endswith(
        "/libappindicator3.so.1"
    )

    build = _step("Build Linux bundles")
    verify = _step("Verify complete Linux AppImage")
    stage = _step("Stage release assets")
    assert "--bundles deb,appimage" in build["with"]["args"]
    assert build["env"]["XDG_CACHE_HOME"] == "${{ runner.temp }}/tauri-tools-cache"
    assert "TAURI_SIGNING_PRIVATE_KEY" in build["env"]
    assert "verify-complete-appimage.sh" in verify["run"]
    assert stage["env"]["ARTIFACT_PATHS"].startswith(
        "${{ steps.build_linux.outputs.artifactPaths"
    )

    clean_machine = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/desktop-app-clean-machine-ci.yml").read_text(
            encoding = "utf-8"
        )
    )
    e2e = clean_machine["jobs"]["appimage-model-download"]
    e2e_source = yaml.safe_dump(e2e)
    assert "webkit2gtk-driver" in e2e_source
    assert "tauri-driver --version 2.0.6 --locked" in e2e_source
    assert "appimage_model_download_webdriver.py" in e2e_source


def test_release_preseeds_every_tauri_appimage_tool_with_a_digest():
    step = _step("Pin complete AppImage toolchain")
    assert step["if"] == "matrix.platform == 'ubuntu-22.04'"
    expected = {
        "APPRUN": "AppRun-x86_64",
        "LINUXDEPLOY": "linuxdeploy-x86_64.AppImage",
        "GTK_PLUGIN": "linuxdeploy-plugin-gtk.sh",
        "GSTREAMER_PLUGIN": "linuxdeploy-plugin-gstreamer.sh",
        "APPIMAGE_PLUGIN": "linuxdeploy-plugin-appimage-x86_64.AppImage",
    }
    for prefix, filename in expected.items():
        assert filename in step["env"][f"{prefix}_URL"]
        assert len(step["env"][f"{prefix}_SHA256"]) == 64
        assert f'fetch "${prefix}_URL" "${prefix}_SHA256"' in step["run"]
    assert "sha256sum -c" in step["run"]
    assert step["run"].index("sha256sum -c") < step["run"].index("chmod +x")
    assert 'rm -f "$APPDIR"/usr/lib/libwayland-client.so*' in step["run"]


def _fake_complete_appdir(tmp_path: Path) -> Path:
    appdir = tmp_path / "AppDir"
    binary = appdir / "usr/bin/unsloth-studio"
    binary.parent.mkdir(parents = True)
    shutil.copy2("/bin/true", binary)
    apprun = appdir / "AppRun"
    apprun.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    apprun.chmod(0o755)
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


def test_complete_appimage_verifier_requires_webkit_and_rejects_host_abi_libraries(tmp_path):
    missing = _fake_complete_appdir(tmp_path / "missing")
    (missing / "usr/lib/libwebkit2gtk-4.1.so.0").unlink()
    result = subprocess.run(
        [VERIFIER, "--appdir", missing], check = False, capture_output = True, text = True
    )
    assert result.returncode != 0
    assert "libwebkit2gtk-4.1.so" in result.stderr

    for library in ("libc.so.6", "libwayland-client.so.0"):
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


def test_all_managed_appimage_children_drop_the_bundle_library_path():
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
    assert process_source.count('cmd.env_remove("LD_LIBRARY_PATH")') == 2
    assert process_source.count('cmd.env_remove("PYTHONHOME")') == 2
    assert process_source.count('cmd.env_remove("PYTHONPATH")') == 2
    for source_path, (call, expected) in child_process_calls.items():
        assert source_path.read_text(encoding = "utf-8").count(call) == expected


def test_release_notes_recommend_native_deb_without_claiming_universality():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "use `.deb` when available" in notes
    assert "universal" not in notes.lower()
