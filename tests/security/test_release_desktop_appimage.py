"""Contracts for the host-integrated Linux AppImage release path."""

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
    assert "tauri signer sign" in package["run"]
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
    assert (
        "sudo apt install libayatana-appindicator3-1 libwebkit2gtk-4.1-0 libgtk-3-0"
        in script
    )
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


def test_release_notes_keep_experimental_appimage_guidance_concise():
    notes = _workflow()["env"]["DESKTOP_RELEASE_NOTES"]
    assert "`.AppImage` is experimental." in notes
    assert "libwebkit2gtk" not in notes
    assert "libfuse" not in notes
    assert "--appimage-extract-and-run" not in notes


def test_thin_appimage_preserves_user_runtime_environment_for_child_processes():
    source_root = REPO_ROOT / "studio" / "src-tauri" / "src"
    child_process_sources = (
        source_root / "commands.rs",
        source_root / "desktop_auth.rs",
        source_root / "install.rs",
        source_root / "preflight" / "managed.rs",
        source_root / "process.rs",
        source_root / "update.rs",
    )

    for source_path in child_process_sources:
        source = source_path.read_text(encoding = "utf-8")
        assert 'var_os("APPIMAGE")' not in source, source_path
