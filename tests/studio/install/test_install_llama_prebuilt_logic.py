import errno
import importlib.util
import io
import json
import re
import os
import shutil
import subprocess
import sys
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

PrebuiltFallback = INSTALL_LLAMA_PREBUILT.PrebuiltFallback
extract_archive = INSTALL_LLAMA_PREBUILT.extract_archive
binary_env = INSTALL_LLAMA_PREBUILT.binary_env
is_secret_env_name = INSTALL_LLAMA_PREBUILT.is_secret_env_name
scrub_env = INSTALL_LLAMA_PREBUILT.scrub_env
isolated_runtime_home = INSTALL_LLAMA_PREBUILT.isolated_runtime_home
HostInfo = INSTALL_LLAMA_PREBUILT.HostInfo
AssetChoice = INSTALL_LLAMA_PREBUILT.AssetChoice
ApprovedArtifactHash = INSTALL_LLAMA_PREBUILT.ApprovedArtifactHash
ApprovedReleaseChecksums = INSTALL_LLAMA_PREBUILT.ApprovedReleaseChecksums
hydrate_source_tree = INSTALL_LLAMA_PREBUILT.hydrate_source_tree
remove_agent_instruction_files = INSTALL_LLAMA_PREBUILT.remove_agent_instruction_files
validate_prebuilt_choice = INSTALL_LLAMA_PREBUILT.validate_prebuilt_choice
activate_install_tree = INSTALL_LLAMA_PREBUILT.activate_install_tree
activate_staged_dir = INSTALL_LLAMA_PREBUILT.activate_staged_dir
create_install_staging_dir = INSTALL_LLAMA_PREBUILT.create_install_staging_dir
sha256_file = INSTALL_LLAMA_PREBUILT.sha256_file
source_archive_logical_name = INSTALL_LLAMA_PREBUILT.source_archive_logical_name
install_prebuilt = INSTALL_LLAMA_PREBUILT.install_prebuilt
write_prebuilt_metadata = INSTALL_LLAMA_PREBUILT.write_prebuilt_metadata
existing_install_matches_plan = INSTALL_LLAMA_PREBUILT.existing_install_matches_plan
existing_install_matches_choice = INSTALL_LLAMA_PREBUILT.existing_install_matches_choice
ensure_diffusion_visual_server = INSTALL_LLAMA_PREBUILT.ensure_diffusion_visual_server
validate_quantize = INSTALL_LLAMA_PREBUILT.validate_quantize
validate_server = INSTALL_LLAMA_PREBUILT.validate_server
build_validation_sandbox_plan = INSTALL_LLAMA_PREBUILT.build_validation_sandbox_plan
linux_missing_libraries = INSTALL_LLAMA_PREBUILT.linux_missing_libraries
ValidationLaunchPlan = INSTALL_LLAMA_PREBUILT._ValidationLaunchPlan
run_validation_capture = INSTALL_LLAMA_PREBUILT._run_validation_capture
run_validation_popen = INSTALL_LLAMA_PREBUILT._run_validation_popen
run_validation_ldd_probe = INSTALL_LLAMA_PREBUILT._run_validation_ldd_probe
LinuxLibraryProbeResult = INSTALL_LLAMA_PREBUILT.LinuxLibraryProbeResult
LINUX_LDD_PROBE_OK = INSTALL_LLAMA_PREBUILT._LINUX_LDD_PROBE_OK
LINUX_LDD_PROBE_SKIPPED = INSTALL_LLAMA_PREBUILT._LINUX_LDD_PROBE_SKIPPED
LINUX_LDD_PROBE_ERROR = INSTALL_LLAMA_PREBUILT._LINUX_LDD_PROBE_ERROR
preflight_linux_installed_binaries = INSTALL_LLAMA_PREBUILT.preflight_linux_installed_binaries
bwrap_can_sandbox = INSTALL_LLAMA_PREBUILT._bwrap_can_sandbox


@pytest.fixture(autouse = True)
def _bwrap_usable_by_default(monkeypatch):
    # A mocked-present bwrap represents a working sandbox; the real usability probe
    # (which would exec bwrap) is out of scope here and covered by its own test.
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_bwrap_can_sandbox", lambda _p: True)


def linux_host() -> HostInfo:
    return HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )


def macos_host() -> HostInfo:
    return HostInfo(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
    )


def windows_host() -> HostInfo:
    return HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
    )


def approved_release_checksums_for_asset(asset_name: str, sha256: str) -> ApprovedReleaseChecksums:
    return ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "b9334",
        upstream_tag = "b9334",
        artifacts = {
            asset_name: ApprovedArtifactHash(
                asset_name = asset_name,
                sha256 = sha256,
                repo = "unslothai/llama.cpp",
                kind = "diffusion-visual-server",
            )
        },
    )


def approved_checksums_for(
    upstream_tag: str, *, source_archive: Path, bundle_archive: Path, bundle_name: str
) -> ApprovedReleaseChecksums:
    return ApprovedReleaseChecksums(
        repo = "local",
        release_tag = upstream_tag,
        upstream_tag = upstream_tag,
        source_commit = None,
        artifacts = {
            source_archive_logical_name(upstream_tag): ApprovedArtifactHash(
                asset_name = source_archive_logical_name(upstream_tag),
                sha256 = sha256_file(source_archive),
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            bundle_name: ApprovedArtifactHash(
                asset_name = bundle_name,
                sha256 = sha256_file(bundle_archive),
                repo = "local",
                kind = "local-test-bundle",
            ),
        },
    )


def test_extract_archive_allows_safe_tar_symlink_chain(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"
    payload = b"shared-object"

    with tarfile.open(archive_path, "w:gz") as archive:
        versioned = tarfile.TarInfo("libllama.so.0.0.1")
        versioned.size = len(payload)
        archive.addfile(versioned, io_bytes(payload))

        soname = tarfile.TarInfo("libllama.so.0")
        soname.type = tarfile.SYMTYPE
        soname.linkname = "libllama.so.0.0.1"
        archive.addfile(soname)

        linker_name = tarfile.TarInfo("libllama.so")
        linker_name.type = tarfile.SYMTYPE
        linker_name.linkname = "libllama.so.0"
        archive.addfile(linker_name)

    destination = tmp_path / "extract"
    extract_archive(archive_path, destination)

    assert (destination / "libllama.so.0.0.1").read_bytes() == payload
    assert (destination / "libllama.so.0").is_symlink()
    assert (destination / "libllama.so").is_symlink()
    assert (destination / "libllama.so").resolve().read_bytes() == payload


def test_extract_archive_allows_safe_tar_hardlink(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"
    payload = b"quantize"

    with tarfile.open(archive_path, "w:gz") as archive:
        target = tarfile.TarInfo("llama-quantize")
        target.size = len(payload)
        archive.addfile(target, io_bytes(payload))

        hardlink = tarfile.TarInfo("llama-quantize-copy")
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "llama-quantize"
        archive.addfile(hardlink)

    destination = tmp_path / "extract"
    extract_archive(archive_path, destination)

    assert (destination / "llama-quantize-copy").read_bytes() == payload
    assert not (destination / "llama-quantize-copy").is_symlink()


def test_extract_archive_rejects_absolute_tar_symlink_target(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/tmp/libllama.so.0"
        archive.addfile(entry)

    with pytest.raises(
        PrebuiltFallback,
        match = r"archive link (used an absolute target|escaped destination)",
    ):
        extract_archive(archive_path, tmp_path / "extract")


def test_extract_archive_rejects_escaping_tar_symlink_target(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../outside/libllama.so.0"
        archive.addfile(entry)

    with pytest.raises(PrebuiltFallback, match = "archive link escaped destination"):
        extract_archive(archive_path, tmp_path / "extract")


def test_extract_archive_rejects_unresolved_tar_symlink_target(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo("libllama.so")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "libllama.so.0"
        archive.addfile(entry)

    with pytest.raises(PrebuiltFallback, match = "unresolved link entries"):
        extract_archive(archive_path, tmp_path / "extract")


def test_extract_archive_rejects_zip_symlink_entry(tmp_path: Path):
    archive_path = tmp_path / "bundle.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo("libllama.so")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "libllama.so.0")

    with pytest.raises(PrebuiltFallback, match = "zip archive contained a symlink entry"):
        extract_archive(archive_path, tmp_path / "extract")


def test_remove_agent_instruction_files_does_not_follow_links(tmp_path: Path):
    managed = tmp_path / "managed"
    nested = managed / "nested"
    external = tmp_path / "external"
    nested.mkdir(parents = True)
    external.mkdir()
    (managed / "AGENTS.md").write_text("managed root", encoding = "utf-8")
    (nested / "AGENTS.md").write_text("managed nested", encoding = "utf-8")
    (managed / "CLAUDE.md").write_text("managed Claude root", encoding = "utf-8")
    (nested / "CLAUDE.md").write_text("managed Claude nested", encoding = "utf-8")
    (external / "AGENTS.md").write_text("user owned", encoding = "utf-8")
    (external / "CLAUDE.md").write_text("user-owned Claude", encoding = "utf-8")
    try:
        (managed / "external-link").symlink_to(external, target_is_directory = True)
        linked_root = tmp_path / "linked-root"
        linked_root.symlink_to(external, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    assert remove_agent_instruction_files(managed) == 4
    assert not list(managed.rglob("AGENTS.md"))
    assert not list(managed.rglob("CLAUDE.md"))
    assert (external / "AGENTS.md").read_text(encoding = "utf-8") == "user owned"
    assert (external / "CLAUDE.md").read_text(encoding = "utf-8") == "user-owned Claude"

    assert remove_agent_instruction_files(linked_root) == 0
    assert (external / "AGENTS.md").exists()
    assert (external / "CLAUDE.md").exists()


@pytest.mark.skipif(os.name != "nt", reason = "Windows junction behavior")
def test_remove_agent_instruction_files_does_not_follow_windows_junctions(tmp_path: Path):
    managed = tmp_path / "managed"
    external = tmp_path / "external"
    managed.mkdir()
    external.mkdir()
    (external / "AGENTS.md").write_text("user owned", encoding = "utf-8")
    (external / "CLAUDE.md").write_text("user-owned Claude", encoding = "utf-8")

    nested_junction = managed / "external-junction"
    root_junction = tmp_path / "linked-root"
    for junction in (nested_junction, root_junction):
        result = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(external)],
            capture_output = True,
            text = True,
            check = False,
        )
        if result.returncode != 0:
            pytest.skip(f"directory junctions unavailable: {result.stderr or result.stdout}")

    assert remove_agent_instruction_files(managed) == 0
    assert remove_agent_instruction_files(root_junction) == 0
    assert (external / "AGENTS.md").read_text(encoding = "utf-8") == "user owned"
    assert (external / "CLAUDE.md").read_text(encoding = "utf-8") == "user-owned Claude"


def test_remove_agent_instruction_files_prunes_linklike_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    managed = tmp_path / "managed"
    simulated_junction = managed / "simulated-junction"
    simulated_junction.mkdir(parents = True)
    agents = simulated_junction / "AGENTS.md"
    claude = simulated_junction / "CLAUDE.md"
    agents.write_text("external instructions", encoding = "utf-8")
    claude.write_text("external Claude instructions", encoding = "utf-8")
    real_is_link_or_junction = INSTALL_LLAMA_PREBUILT._is_link_or_junction

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_is_link_or_junction",
        lambda path: path == simulated_junction or real_is_link_or_junction(path),
    )

    assert remove_agent_instruction_files(managed) == 0
    assert agents.exists()
    assert claude.exists()


def test_remove_agent_instruction_files_continues_after_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    managed = tmp_path / "managed"
    managed.mkdir()
    blocked = managed / "AGENTS.md"
    removable = managed / "CLAUDE.md"
    blocked.write_text("blocked", encoding = "utf-8")
    removable.write_text("remove me", encoding = "utf-8")
    real_unlink = Path.unlink

    def selective_unlink(path: Path, *args, **kwargs):
        if path == blocked:
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", selective_unlink)

    assert remove_agent_instruction_files(managed) == 1
    assert blocked.exists()
    assert not removable.exists()
    captured = capsys.readouterr()
    assert "could not remove contributor-only instruction" in captured.out + captured.err


def test_main_resolves_linked_install_path_and_preserves_cleanup_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "target"
    linked_root = tmp_path / "linked-root"
    target.mkdir()
    try:
        linked_root.symlink_to(target, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    received = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--install-dir", str(linked_root)],
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "install_prebuilt",
        lambda **kwargs: received.update(kwargs),
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_LOG_TO_STDOUT", False)

    assert INSTALL_LLAMA_PREBUILT.main() == 0
    assert received["install_dir"] == target.resolve()
    assert received["instruction_cleanup_root"] == linked_root.absolute()
    assert received["instruction_cleanup_root"].is_symlink()


def test_install_prebuilt_uses_explicit_instruction_cleanup_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "target"
    linked_root = tmp_path / "linked-root"
    install_dir.mkdir()
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text("{}", encoding = "utf-8")
    try:
        linked_root.symlink_to(install_dir, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    cleanup_roots = []
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", linux_host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "remove_agent_instruction_files",
        lambda root: cleanup_roots.append(root) or 0,
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after cleanup")),
    )

    with pytest.raises(RuntimeError, match = "stop after cleanup"):
        install_prebuilt(
            install_dir.resolve(),
            "latest",
            "unslothai/llama.cpp",
            "",
            instruction_cleanup_root = linked_root.absolute(),
        )

    assert cleanup_roots == [linked_root.absolute()]


def test_hydrate_source_tree_extracts_upstream_archive_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    upstream_tag = "b9999"
    archive_path = tmp_path / "llama.cpp-source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/CMakeLists.txt",
            b"cmake_minimum_required(VERSION 3.14)\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/convert_hf_to_gguf.py",
            b"#!/usr/bin/env python3\nimport gguf\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/gguf-py/gguf/__init__.py",
            b"__all__ = []\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/AGENTS.md",
            b"upstream contributor instructions\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/examples/AGENTS.md",
            b"nested contributor instructions\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/CLAUDE.md",
            b"Claude contributor instructions\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/examples/CLAUDE.md",
            b"nested Claude contributor instructions\n",
        )

    source_urls = set(INSTALL_LLAMA_PREBUILT.upstream_source_archive_urls(upstream_tag))

    def fake_download_file(url: str, destination: Path) -> None:
        assert url in source_urls
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)

    install_dir = tmp_path / "install"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    hydrate_source_tree(
        upstream_tag, install_dir, work_dir, expected_sha256 = sha256_file(archive_path)
    )

    assert (install_dir / "CMakeLists.txt").exists()
    assert (install_dir / "convert_hf_to_gguf.py").exists()
    assert (install_dir / "gguf-py" / "gguf" / "__init__.py").exists()
    assert not (install_dir / f"llama.cpp-{upstream_tag}").exists()
    assert not list(install_dir.rglob("AGENTS.md"))
    assert not list(install_dir.rglob("CLAUDE.md"))


def test_release_asset_download_url():
    fn = INSTALL_LLAMA_PREBUILT.release_asset_download_url
    assert fn(
        "unslothai/llama.cpp", "b9000-mix-abc1234", "llama.cpp-source-commit-deadbeef.tar.gz"
    ) == (
        "https://github.com/unslothai/llama.cpp/releases/download/"
        "b9000-mix-abc1234/llama.cpp-source-commit-deadbeef.tar.gz"
    )
    # Any missing component -> None (no asset url, caller falls back to codeload).
    assert fn(None, "b9000", "x.tar.gz") is None
    assert fn("unslothai/llama.cpp", None, "x.tar.gz") is None
    assert fn("unslothai/llama.cpp", "b9000", None) is None


def _mk_source_tarball(path: Path, tag: str) -> None:
    with tarfile.open(path, "w:gz") as archive:
        add_bytes_to_tar(
            archive, f"llama.cpp-{tag}/CMakeLists.txt", b"cmake_minimum_required(VERSION 3.14)\n"
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{tag}/convert_hf_to_gguf.py",
            b"#!/usr/bin/env python3\nimport gguf\n",
        )
        add_bytes_to_tar(archive, f"llama.cpp-{tag}/gguf-py/gguf/__init__.py", b"__all__ = []\n")


def test_hydrate_source_tree_prefers_release_asset_for_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A mix build's merge commit 404s on codeload, so hydrate must fetch the release asset.
    commit = "a" * 40
    archive_path = tmp_path / "merged-source.tar.gz"
    _mk_source_tarball(archive_path, f"b9000-mix-{commit[:7]}")
    asset_url = INSTALL_LLAMA_PREBUILT.release_asset_download_url(
        "unslothai/llama.cpp", "b9000-mix-abc1234", f"llama.cpp-source-commit-{commit}.tar.gz"
    )
    codeload_urls = set(
        INSTALL_LLAMA_PREBUILT.commit_source_archive_urls("unslothai/llama.cpp", commit)
    )
    seen = []

    def fake_download_file(url: str, destination: Path) -> None:
        seen.append(url)
        if url in codeload_urls:
            raise AssertionError("codeload was hit even though the release asset was available")
        assert url == asset_url
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)

    install_dir = tmp_path / "install"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    hydrate_source_tree(
        commit,
        install_dir,
        work_dir,
        source_repo = "unslothai/llama.cpp",
        expected_sha256 = sha256_file(archive_path),
        exact_source = True,
        asset_url = asset_url,
    )
    assert seen == [asset_url]
    assert (install_dir / "CMakeLists.txt").exists()
    assert (install_dir / "convert_hf_to_gguf.py").exists()


def test_hydrate_source_tree_falls_back_to_codeload_when_asset_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # If the release asset 404s, fall back to codeload/archive (vanilla path).
    commit = "b" * 40
    archive_path = tmp_path / "vanilla-source.tar.gz"
    _mk_source_tarball(archive_path, f"commit-{commit[:7]}")
    asset_url = INSTALL_LLAMA_PREBUILT.release_asset_download_url(
        "unslothai/llama.cpp", "b9000", f"llama.cpp-source-commit-{commit}.tar.gz"
    )
    codeload_urls = INSTALL_LLAMA_PREBUILT.commit_source_archive_urls("unslothai/llama.cpp", commit)

    def fake_download_file(url: str, destination: Path) -> None:
        if url == asset_url:
            raise RuntimeError("404 Not Found")
        assert url in codeload_urls
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)

    install_dir = tmp_path / "install"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    hydrate_source_tree(
        commit,
        install_dir,
        work_dir,
        source_repo = "unslothai/llama.cpp",
        expected_sha256 = sha256_file(archive_path),
        exact_source = True,
        asset_url = asset_url,
    )
    assert (install_dir / "CMakeLists.txt").exists()


def test_validate_prebuilt_choice_creates_repo_shaped_linux_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    upstream_tag = "b9998"
    bundle_name = "app-b9998-linux-x64-cuda13-newer.tar.gz"
    source_archive = tmp_path / "source.tar.gz"
    bundle_archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/CMakeLists.txt",
            b"cmake_minimum_required(VERSION 3.14)\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/convert_hf_to_gguf.py",
            b"#!/usr/bin/env python3\nimport gguf\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/gguf-py/gguf/__init__.py",
            b"__all__ = []\n",
        )
    with tarfile.open(bundle_archive, "w:gz") as archive:
        add_bytes_to_tar(archive, "llama-server", b"#!/bin/sh\nexit 0\n", mode = 0o755)
        add_bytes_to_tar(archive, "llama-quantize", b"#!/bin/sh\nexit 0\n", mode = 0o755)
        add_bytes_to_tar(archive, "libllama.so.0.0.1", b"libllama")
        add_symlink_to_tar(archive, "libllama.so.0", "libllama.so.0.0.1")
        add_symlink_to_tar(archive, "libllama.so", "libllama.so.0")
        add_bytes_to_tar(archive, "libggml.so.0.9.8", b"libggml")
        add_symlink_to_tar(archive, "libggml.so.0", "libggml.so.0.9.8")
        add_symlink_to_tar(archive, "libggml.so", "libggml.so.0")
        add_bytes_to_tar(archive, "libggml-base.so.0.9.8", b"libggml-base")
        add_symlink_to_tar(archive, "libggml-base.so.0", "libggml-base.so.0.9.8")
        add_symlink_to_tar(archive, "libggml-base.so", "libggml-base.so.0")
        add_bytes_to_tar(archive, "libggml-cpu-x64.so.0.9.8", b"libggml-cpu")
        add_symlink_to_tar(archive, "libggml-cpu-x64.so.0", "libggml-cpu-x64.so.0.9.8")
        add_symlink_to_tar(archive, "libggml-cpu-x64.so", "libggml-cpu-x64.so.0")
        add_bytes_to_tar(archive, "libmtmd.so.0.0.1", b"libmtmd")
        add_symlink_to_tar(archive, "libmtmd.so.0", "libmtmd.so.0.0.1")
        add_symlink_to_tar(archive, "libmtmd.so", "libmtmd.so.0")
        add_bytes_to_tar(archive, "BUILD_INFO.txt", b"bundle metadata\n")
        add_bytes_to_tar(archive, "THIRD_PARTY_LICENSES.txt", b"licenses\n")

    source_urls = set(INSTALL_LLAMA_PREBUILT.upstream_source_archive_urls(upstream_tag))

    def fake_download_file(url: str, destination: Path) -> None:
        if url in source_urls:
            destination.write_bytes(source_archive.read_bytes())
            return
        if url == "file://bundle":
            destination.write_bytes(bundle_archive.read_bytes())
            return
        raise AssertionError(f"unexpected download url: {url}")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_bytes",
        lambda url, **_: b"#!/usr/bin/env python3\nimport gguf\n",
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "preflight_linux_installed_binaries",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "validate_quantize", lambda *args, **kwargs: None)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "validate_server", lambda *args, **kwargs: None)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "local",
        tag = upstream_tag,
        name = bundle_name,
        url = "file://bundle",
        source_label = "local",
        is_ready_bundle = True,
        install_kind = "linux-cuda",
        bundle_profile = "cuda13-newer",
        runtime_line = "cuda13",
        expected_sha256 = sha256_file(bundle_archive),
    )

    install_dir = tmp_path / "install"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    probe_path = tmp_path / "stories260K.gguf"
    quantized_path = tmp_path / "stories260K-q4.gguf"
    validate_prebuilt_choice(
        choice,
        host,
        install_dir,
        work_dir,
        probe_path,
        requested_tag = upstream_tag,
        llama_tag = upstream_tag,
        release_tag = upstream_tag,
        approved_checksums = approved_checksums_for(
            upstream_tag,
            source_archive = source_archive,
            bundle_archive = bundle_archive,
            bundle_name = bundle_name,
        ),
        prebuilt_fallback_used = False,
        quantized_path = quantized_path,
    )

    assert (install_dir / "gguf-py" / "gguf" / "__init__.py").exists()
    assert (install_dir / "convert_hf_to_gguf.py").exists()
    assert (install_dir / "build" / "bin" / "llama-server").exists()
    assert (install_dir / "build" / "bin" / "llama-quantize").exists()
    assert (install_dir / "build" / "bin" / "libllama.so").exists()
    assert (install_dir / "llama-server").exists()
    assert (install_dir / "llama-quantize").exists()
    assert (install_dir / "UNSLOTH_PREBUILT_INFO.json").exists()
    assert (install_dir / "BUILD_INFO.txt").exists()


def test_validate_prebuilt_choice_creates_repo_shaped_windows_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    upstream_tag = "b9997"
    bundle_name = "app-b9997-windows-x64-cpu.zip"
    source_archive = tmp_path / "source.tar.gz"
    bundle_archive = tmp_path / "bundle.zip"
    with tarfile.open(source_archive, "w:gz") as archive:
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/CMakeLists.txt",
            b"cmake_minimum_required(VERSION 3.14)\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/convert_hf_to_gguf.py",
            b"#!/usr/bin/env python3\nimport gguf\n",
        )
        add_bytes_to_tar(
            archive,
            f"llama.cpp-{upstream_tag}/gguf-py/gguf/__init__.py",
            b"__all__ = []\n",
        )
    with zipfile.ZipFile(bundle_archive, "w") as archive:
        archive.writestr("llama-server.exe", b"MZ")
        archive.writestr("llama-quantize.exe", b"MZ")
        archive.writestr("llama.dll", b"DLL")
        archive.writestr("BUILD_INFO.txt", b"bundle metadata\n")

    source_urls = set(INSTALL_LLAMA_PREBUILT.upstream_source_archive_urls(upstream_tag))

    def fake_download_file(url: str, destination: Path) -> None:
        if url in source_urls:
            destination.write_bytes(source_archive.read_bytes())
            return
        if url == "file://bundle.zip":
            destination.write_bytes(bundle_archive.read_bytes())
            return
        raise AssertionError(f"unexpected download url: {url}")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_bytes",
        lambda url, **_: b"#!/usr/bin/env python3\nimport gguf\n",
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "preflight_linux_installed_binaries",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "validate_quantize", lambda *args, **kwargs: None)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "validate_server", lambda *args, **kwargs: None)

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "local",
        tag = upstream_tag,
        name = bundle_name,
        url = "file://bundle.zip",
        source_label = "local",
        is_ready_bundle = True,
        install_kind = "windows-cpu",
        expected_sha256 = sha256_file(bundle_archive),
    )

    install_dir = tmp_path / "install"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    probe_path = tmp_path / "stories260K.gguf"
    quantized_path = tmp_path / "stories260K-q4.gguf"
    validate_prebuilt_choice(
        choice,
        host,
        install_dir,
        work_dir,
        probe_path,
        requested_tag = upstream_tag,
        llama_tag = upstream_tag,
        release_tag = upstream_tag,
        approved_checksums = approved_checksums_for(
            upstream_tag,
            source_archive = source_archive,
            bundle_archive = bundle_archive,
            bundle_name = bundle_name,
        ),
        prebuilt_fallback_used = False,
        quantized_path = quantized_path,
    )

    assert (install_dir / "gguf-py" / "gguf" / "__init__.py").exists()
    assert (install_dir / "convert_hf_to_gguf.py").exists()
    assert (install_dir / "build" / "bin" / "Release" / "llama-server.exe").exists()
    assert (install_dir / "build" / "bin" / "Release" / "llama-quantize.exe").exists()
    assert (install_dir / "build" / "bin" / "Release" / "llama.dll").exists()
    assert not (install_dir / "llama-server.exe").exists()
    assert (install_dir / "UNSLOTH_PREBUILT_INFO.json").exists()
    assert (install_dir / "BUILD_INFO.txt").exists()


def test_activate_install_tree_restores_existing_install_after_activation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    (install_dir / "old.txt").write_text("old install\n")

    staging_dir = create_install_staging_dir(install_dir)
    (staging_dir / "new.txt").write_text("new install\n")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "confirm_install_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("activation confirm failed")),
    )

    with pytest.raises(
        PrebuiltFallback,
        match = "activation failed; restored previous install",
    ):
        activate_install_tree(staging_dir, install_dir, host)

    assert (install_dir / "old.txt").read_text() == "old install\n"
    assert not (install_dir / "new.txt").exists()
    assert not staging_dir.exists()
    assert not (tmp_path / ".staging").exists()

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "moving existing install to rollback path" in output
    assert "restored previous install from rollback path" in output


def test_activate_install_tree_preserves_symlink_to_resolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "target"
    linked_root = tmp_path / "linked-root"
    staging_dir = tmp_path / "staging"
    install_dir.mkdir()
    staging_dir.mkdir()
    (install_dir / "old.txt").write_text("old", encoding = "utf-8")
    (staging_dir / "new.txt").write_text("new", encoding = "utf-8")
    try:
        linked_root.symlink_to(install_dir, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "confirm_install_tree", lambda *_args: None)

    activate_install_tree(staging_dir, linked_root.resolve(), linux_host())

    assert linked_root.is_symlink()
    assert (linked_root / "new.txt").read_text(encoding = "utf-8") == "new"
    assert not (linked_root / "old.txt").exists()


def test_activate_install_tree_cleans_all_paths_when_rollback_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    (install_dir / "old.txt").write_text("old install\n")

    staging_dir = create_install_staging_dir(install_dir)
    (staging_dir / "new.txt").write_text("new install\n")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "confirm_install_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("activation confirm failed")),
    )

    original_replace = INSTALL_LLAMA_PREBUILT.os.replace

    def flaky_replace(src, dst):
        src_path = Path(src)
        dst_path = Path(dst)
        if "rollback-" in src_path.name and dst_path == install_dir:
            raise OSError("restore failed")
        return original_replace(src, dst)

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "replace", flaky_replace)

    with pytest.raises(
        PrebuiltFallback,
        match = "activation and rollback failed; cleaned install state for fresh source build",
    ):
        activate_install_tree(staging_dir, install_dir, host)

    assert not install_dir.exists()
    assert not staging_dir.exists()
    assert not (tmp_path / ".staging").exists()

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "rollback after failed activation also failed: restore failed" in output
    assert "cleaning staging, install, and rollback paths before source build fallback" in output
    assert "removing failed install path" in output
    assert "removing rollback path" in output


def test_activate_staged_dir_copies_when_replace_hits_busy_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    staging_dir = tmp_path / "llama.cpp.staging-test"
    (staging_dir / "bin").mkdir(parents = True)
    (staging_dir / "bin" / "ggml-base.dll").write_bytes(b"fake dll")
    dst = tmp_path / "llama.cpp"

    def denied_replace(src, dst_arg):
        raise PermissionError(errno.EACCES, "Access is denied", str(src))

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "replace", denied_replace)

    activate_staged_dir(staging_dir, dst)

    assert (dst / "bin" / "ggml-base.dll").read_bytes() == b"fake dll"
    assert not staging_dir.exists()

    captured = capsys.readouterr()
    assert "falling back to file-by-file copy" in captured.out + captured.err


def test_activate_staged_dir_reraises_non_busy_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    staging_dir = tmp_path / "llama.cpp.staging-test"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new install\n")
    dst = tmp_path / "llama.cpp"

    def out_of_space_replace(src, dst_arg):
        raise OSError(errno.ENOSPC, "No space left on device", str(src))

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "replace", out_of_space_replace)

    with pytest.raises(OSError, match = "No space left on device"):
        activate_staged_dir(staging_dir, dst)

    assert not dst.exists()
    assert (staging_dir / "new.txt").read_text() == "new install\n"


def test_binary_env_linux_includes_binary_parent_in_ld_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    binary_path = bin_dir / "llama-server"
    binary_path.write_bytes(b"fake")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _bp: [])

    env = binary_env(binary_path, install_dir, host)
    ld_dirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert (
        str(bin_dir) in ld_dirs
    ), f"binary_path.parent ({bin_dir}) must be in LD_LIBRARY_PATH, got: {ld_dirs}"
    assert str(install_dir) in ld_dirs


def test_scrub_env_drops_secrets_and_keeps_runtime_vars():
    raw = {
        # secrets
        "HF_TOKEN": "hf_x",
        "HUGGING_FACE_HUB_TOKEN": "hf_y",
        "GH_TOKEN": "gh_x",
        "GITHUB_TOKEN": "gh_y",
        "WANDB_API_KEY": "wandb_x",
        "AWS_SECRET_ACCESS_KEY": "aws_x",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "oidc_x",
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc",
        "SOME_VENDOR_API_KEY": "vendor_x",
        "DB_PASSWORD": "pw",
        "MY_PRIVATE_KEY": "pk",
        "KUBECONFIG": "/home/runner/.kube/config",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "SSH_PASSPHRASE": "ssh_pass",
        # runtime vars to keep
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/opt/lib",
        "DYLD_LIBRARY_PATH": "/opt/dyld",
        "HOME": "/home/runner",
        "TMPDIR": "/tmp",
        "CUDA_VISIBLE_DEVICES": "0",
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    }

    cleaned = scrub_env(raw)

    for secret in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "WANDB_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "SOME_VENDOR_API_KEY",
        "DB_PASSWORD",
        "MY_PRIVATE_KEY",
        "KUBECONFIG",
        "SSH_AUTH_SOCK",
        "SSH_PASSPHRASE",
    ):
        assert secret not in cleaned, f"{secret} must be stripped from binary env"

    for keep in (
        "PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        assert cleaned[keep] == raw[keep], f"{keep} must be preserved for the binary"

    # no bare "KEY" marker: benign KEY-containing names survive
    assert is_secret_env_name("API_KEY") is True
    assert is_secret_env_name("SSH_KEYFILE_PATH") is False
    assert is_secret_env_name("PATH") is False


def test_scrub_env_drops_proxy_index_and_embedded_url_credentials():
    raw = {
        # proxy / package-index URLs whose values commonly embed credentials
        "HTTPS_PROXY": "https://user:secret@proxy:8080",
        "https_proxy": "https://user:secret@proxy:8080",  # lower-case variant
        "ALL_PROXY": "socks5://user:secret@proxy:1080",
        "PIP_INDEX_URL": "https://u:p@pypi.internal/simple",
        "UV_INDEX_URL": "https://u:p@index.internal/simple",
        # credentials embedded in an otherwise benign-named variable's value
        "MY_DB_DSN": "postgres://admin:secret@db:5432/app",
        # benign vars the binary needs, including a URL with no userinfo
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": "0",
        "NO_PROXY": "localhost,127.0.0.1",
        "SOME_ENDPOINT": "https://example.com:8080/v1",
    }

    cleaned = scrub_env(raw)

    for secret in (
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "PIP_INDEX_URL",
        "UV_INDEX_URL",
        "MY_DB_DSN",
    ):
        assert secret not in cleaned, f"{secret} must be stripped from binary env"
    for keep in ("PATH", "CUDA_VISIBLE_DEVICES", "NO_PROXY", "SOME_ENDPOINT"):
        assert cleaned[keep] == raw[keep], f"{keep} must be preserved for the binary"

    assert is_secret_env_name("HTTPS_PROXY") is True
    assert is_secret_env_name("https_proxy") is True
    assert is_secret_env_name("NO_PROXY") is False


def test_binary_env_strips_secrets_from_downloaded_binary_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    binary_path = bin_dir / "llama-server"
    binary_path.write_bytes(b"fake")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _bp: [])

    monkeypatch.setenv("HF_TOKEN", "hf_secret_from_ci")
    monkeypatch.setenv("GITHUB_TOKEN", "gh_secret_from_ci")
    monkeypatch.setenv("GH_TOKEN", "gh_secret_from_ci")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_secret_from_ci")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    env = binary_env(binary_path, install_dir, host)

    assert "HF_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert "GH_TOKEN" not in env
    assert "WANDB_API_KEY" not in env
    # library/runtime resolution unaffected
    assert str(bin_dir) in env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_binary_env_linux_strips_loader_injections_and_broad_inherited_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    runtime_lib = tmp_path / "runtime" / "lib"
    runtime_lib.mkdir(parents = True)
    bin_dir.mkdir(parents = True)
    binary_path = bin_dir / "llama-server"
    binary_path.write_bytes(b"fake")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _bp: [])
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        os.pathsep.join([str(Path("/")), str(runtime_lib.parent / ".." / "runtime" / "lib")]),
    )
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "inject.so"))
    monkeypatch.setenv("LD_AUDIT", str(tmp_path / "audit.so"))

    env = binary_env(binary_path, install_dir, linux_host())

    assert "LD_PRELOAD" not in env
    assert "LD_AUDIT" not in env
    ld_dirs = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(Path("/")) not in ld_dirs
    assert str(runtime_lib.resolve()) in ld_dirs


def test_binary_env_redirects_home_away_from_real_credential_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    binary_path = bin_dir / "llama-server"
    binary_path.write_bytes(b"fake")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _bp: [])

    real_home = str(tmp_path / "real_home")
    monkeypatch.setenv("HOME", real_home)
    monkeypatch.setenv("HF_HOME", real_home + "/.cache/huggingface")

    env = binary_env(binary_path, install_dir, host)

    # HOME and the cache pointers are redirected to a single empty, existing dir.
    assert env["HOME"] != real_home
    assert env["HF_HOME"] == env["HOME"]
    assert env["HOME"] == isolated_runtime_home()
    assert os.path.isdir(env["HOME"])
    assert os.listdir(env["HOME"]) == []
    # Windows reconstructs the profile from HOMEDRIVE + HOMEPATH.
    assert env["HOMEDRIVE"] + env["HOMEPATH"] == env["HOME"]


def test_scrub_env_drops_token_only_url_userinfo():
    raw = {
        "GENERIC_REPO": "https://ghp_tokenonly@github.com/org/repo",
        "GENERIC_OK": "https://example.com:8080/v1",
    }
    cleaned = scrub_env(raw)
    assert "GENERIC_REPO" not in cleaned
    assert cleaned["GENERIC_OK"] == raw["GENERIC_OK"]


def test_binary_env_drops_explicit_credential_file_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _bp: [])
    dropped = (
        "NETRC",
        "PIP_CONFIG_FILE",
        "DOCKER_CONFIG",
        "GIT_CONFIG_GLOBAL",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_OUTPUT",
        "GITHUB_STEP_SUMMARY",
        "BASH_ENV",
    )
    for var in dropped:
        monkeypatch.setenv(var, "/home/realuser/secret")

    env = binary_env(tmp_path / "llama-server", tmp_path, host)

    for var in dropped:
        assert var not in env


def test_binary_env_macos_strips_inherited_dyld_loader_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    runtime_lib = tmp_path / "runtime" / "lib"
    runtime_lib.mkdir(parents = True)
    bin_dir.mkdir(parents = True)
    binary_path = bin_dir / "llama-server"
    binary_path.write_bytes(b"fake")

    monkeypatch.setenv(
        "DYLD_LIBRARY_PATH",
        str(runtime_lib.parent / ".." / runtime_lib.parent.name / runtime_lib.name),
    )
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", str(tmp_path / "inject.dylib"))
    monkeypatch.setenv("DYLD_FRAMEWORK_PATH", str(tmp_path / "Frameworks"))
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", str(tmp_path / "fallback"))

    env = binary_env(binary_path, install_dir, macos_host())

    assert "DYLD_INSERT_LIBRARIES" not in env
    assert "DYLD_FRAMEWORK_PATH" not in env
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in env
    dyld_dirs = env["DYLD_LIBRARY_PATH"].split(os.pathsep)
    assert str(bin_dir) in dyld_dirs
    assert str(install_dir) in dyld_dirs
    assert str(runtime_lib.resolve()) in dyld_dirs


def test_linux_runtime_dirs_probes_with_secret_free_env(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_probe(binary_path, *, env = None):
        captured["env"] = env
        return LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
            output = "",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_ldd_probe", fake_probe)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh_secret")

    INSTALL_LLAMA_PREBUILT.linux_runtime_dirs(Path("/fake/llama-server"))

    probe_env = captured["env"]
    assert probe_env is not None
    assert "HF_TOKEN" not in probe_env
    assert "GITHUB_TOKEN" not in probe_env


def _command_contains_path(plan: ValidationLaunchPlan, path_fragment: str) -> bool:
    return any(
        path_fragment in command_part or path_fragment in command_part.replace("\\", "/")
        for command_part in plan.command
    )


def _command_has_setenv(plan: ValidationLaunchPlan, key: str) -> bool:
    for index in range(len(plan.command) - 2):
        if plan.command[index] == "--setenv" and plan.command[index + 1] == key:
            return True
    return False


def _command_has_bind(plan: ValidationLaunchPlan, flag: str, source: str | Path) -> bool:
    source_text = str(Path(source))
    for index in range(len(plan.command) - 2):
        if (
            plan.command[index] == flag
            and plan.command[index + 1] == source_text
            and plan.command[index + 2] == source_text
        ):
            return True
    return False


def test_install_prebuilt_falls_back_to_older_release_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    first_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "old-release",
        name = "app-b9002-linux-x64.tar.gz",
        url = "https://example.com/app-b9002-linux-x64.tar.gz",
        source_label = "published",
        install_kind = "linux-cpu",
    )
    second_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "older-release",
        name = "app-b9001-linux-x64.tar.gz",
        url = "https://example.com/app-b9001-linux-x64.tar.gz",
        source_label = "published",
        install_kind = "linux-cpu",
    )
    first_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9002",
        release_tag = "release-2",
        attempts = [first_choice],
        approved_checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "release-2",
            upstream_tag = "b9002",
            source_commit = None,
            artifacts = {},
        ),
    )
    second_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [second_choice],
        approved_checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "release-1",
            upstream_tag = "b9001",
            source_commit = None,
            artifacts = {},
        ),
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [first_plan, second_plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda probe_path, cache_path: probe_path.write_bytes(b"probe"),
    )

    call_log: list[tuple[str, bool]] = []

    def fake_validate(
        attempts,
        host,
        install_dir,
        work_dir,
        probe_path,
        *,
        requested_tag,
        llama_tag,
        release_tag,
        approved_checksums,
        initial_fallback_used = False,
        existing_install_dir = None,
    ):
        call_log.append((llama_tag, initial_fallback_used))
        if llama_tag == "b9002":
            raise PrebuiltFallback("validation failed for latest release")
        staging_dir = create_install_staging_dir(install_dir)
        (staging_dir / "marker.txt").write_text("ready\n")
        return attempts[0], staging_dir, initial_fallback_used

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "validate_prebuilt_attempts",
        fake_validate,
    )

    activated = {}
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "activate_install_tree",
        lambda staging_dir, install_dir, host: activated.update(
            {"staging_dir": staging_dir, "install_dir": install_dir}
        ),
    )
    ensured_tags: list[str] = []
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "ensure_converter_scripts",
        lambda install_dir, llama_tag: ensured_tags.append(llama_tag),
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert call_log == [("b9002", False), ("b9001", True)]
    assert activated["install_dir"] == install_dir
    assert ensured_tags == ["b9001"]


def write_linux_install_shape(install_dir: Path) -> None:
    runtime_dir = install_dir / "build" / "bin"
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (install_dir / "llama-server").write_text("#!/bin/sh\n", encoding = "utf-8")
    (install_dir / "llama-quantize").write_text("#!/bin/sh\n", encoding = "utf-8")
    (runtime_dir / "llama-server").write_text("#!/bin/sh\n", encoding = "utf-8")
    (runtime_dir / "llama-quantize").write_text("#!/bin/sh\n", encoding = "utf-8")
    # libllama-common.so* (PR #5135) is a required runtime payload health group.
    (runtime_dir / "libllama-common.so.0").write_bytes(b"DLL")
    (runtime_dir / "libllama.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml-base.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml-cpu-x64.so.0").write_bytes(b"DLL")
    (runtime_dir / "libmtmd.so.0").write_bytes(b"DLL")
    (install_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3\n", encoding = "utf-8")
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True, exist_ok = True)


def write_windows_install_shape(
    install_dir: Path,
    *,
    include_llama_dll: bool = True,
    include_cuda_dll: bool = False,
    include_cudart_dlls: bool = False,
) -> None:
    runtime_dir = install_dir / "build" / "bin" / "Release"
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (runtime_dir / "llama-server.exe").write_bytes(b"MZ")
    (runtime_dir / "llama-quantize.exe").write_bytes(b"MZ")
    if include_llama_dll:
        (runtime_dir / "llama.dll").write_bytes(b"DLL")
    if include_cuda_dll:
        (runtime_dir / "ggml-cuda.dll").write_bytes(b"DLL")
    if include_cudart_dlls:
        # cudart bundle DLLs that ship in cudart-llama-bin-win-cuda-*-x64.zip
        (runtime_dir / "cudart64_12.dll").write_bytes(b"DLL")
        (runtime_dir / "cublas64_12.dll").write_bytes(b"DLL")
        (runtime_dir / "cublasLt64_12.dll").write_bytes(b"DLL")
    (install_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3\n", encoding = "utf-8")
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True, exist_ok = True)


def write_macos_install_shape(
    install_dir: Path,
    *,
    include_libllama: bool = True,
    include_libggml: bool = True,
    include_libmtmd: bool = True,
) -> None:
    runtime_dir = install_dir / "build" / "bin"
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (install_dir / "llama-server").write_text("#!/bin/sh\n", encoding = "utf-8")
    (install_dir / "llama-quantize").write_text("#!/bin/sh\n", encoding = "utf-8")
    (runtime_dir / "llama-server").write_text("#!/bin/sh\n", encoding = "utf-8")
    (runtime_dir / "llama-quantize").write_text("#!/bin/sh\n", encoding = "utf-8")
    if include_libllama:
        (runtime_dir / "libllama.0.dylib").write_bytes(b"DLL")
    if include_libggml:
        (runtime_dir / "libggml.0.dylib").write_bytes(b"DLL")
    if include_libmtmd:
        (runtime_dir / "libmtmd.0.dylib").write_bytes(b"DLL")
    (install_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3\n", encoding = "utf-8")
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True, exist_ok = True)


def test_existing_install_matches_plan_with_fingerprint_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
        ),
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is True


def test_existing_install_matches_plan_linux_allows_skipped_ldd_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = "bwrap unavailable",
        ),
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, linux_host(), plan) is True


def test_existing_install_matches_plan_false_without_fingerprint(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"tag": "b9001", "asset": "llama-b9001-bin-ubuntu-x64.tar.gz"}) + "\n",
        encoding = "utf-8",
    )

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/x.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_existing_install_matches_plan_false_with_malformed_metadata(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text("{not-json\n", encoding = "utf-8")

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/x.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_existing_install_matches_plan_windows_cpu_requires_llama_dll(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_windows_install_shape(install_dir, include_llama_dll = True)

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cpu-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is True
    (install_dir / "build" / "bin" / "Release" / "llama.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_existing_install_matches_plan_windows_cuda_requires_cuda_dll(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_windows_install_shape(install_dir, include_llama_dll = True, include_cuda_dll = True)

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (12, 4),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = True,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is True
    (install_dir / "build" / "bin" / "Release" / "ggml-cuda.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_existing_install_matches_plan_windows_cuda_paired_requires_cudart(tmp_path: Path):
    """A paired cudart bundle (#5106) marks the install stale unless cudart64_* and cublas64_* are on disk."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_windows_install_shape(
        install_dir,
        include_llama_dll = True,
        include_cuda_dll = True,
        include_cudart_dlls = True,
    )

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (12, 4),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = True,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
        runtime_name = "cudart-llama-bin-win-cuda-12.4-x64.zip",
        runtime_url = "https://example.com/cudart.zip",
        runtime_sha256 = "c" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
            choice.runtime_name: ApprovedArtifactHash(
                asset_name = choice.runtime_name,
                sha256 = choice.runtime_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    # Fully populated install (main archive + cudart DLLs) matches.
    assert existing_install_matches_plan(install_dir, host, plan) is True

    # cublas missing -- stale, must reinstall.
    (install_dir / "build" / "bin" / "Release" / "cublas64_12.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False

    # cudart missing -- stale, must reinstall.
    write_windows_install_shape(
        install_dir,
        include_llama_dll = True,
        include_cuda_dll = True,
        include_cudart_dlls = True,
    )
    (install_dir / "build" / "bin" / "Release" / "cudart64_12.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False

    # cublasLt missing -- stale, must reinstall (all three DLLs are required).
    write_windows_install_shape(
        install_dir,
        include_llama_dll = True,
        include_cuda_dll = True,
        include_cudart_dlls = True,
    )
    (install_dir / "build" / "bin" / "Release" / "cublasLt64_12.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_existing_install_matches_plan_windows_cuda_unpaired_skips_cudart_check(tmp_path: Path):
    """With no paired runtime archive, a legacy install lacking cudart must still pass (else reinstall loops)."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_windows_install_shape(
        install_dir,
        include_llama_dll = True,
        include_cuda_dll = True,
        include_cudart_dlls = False,
    )

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (12, 4),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = True,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is True


def test_existing_install_fingerprint_changes_when_cudart_pair_added(tmp_path: Path):
    """A pre-#5322 CUDA install must go stale once the choice gains a runtime archive (#5106 fingerprint half)."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_windows_install_shape(
        install_dir,
        include_llama_dll = True,
        include_cuda_dll = True,
        include_cudart_dlls = False,
    )

    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (12, 4),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = True,
    )
    legacy_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
    )
    paired_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "https://example.com/x.zip",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
        runtime_name = "cudart-llama-bin-win-cuda-12.4-x64.zip",
        runtime_url = "https://example.com/cudart.zip",
        runtime_sha256 = "c" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            legacy_choice.name: ApprovedArtifactHash(
                asset_name = legacy_choice.name,
                sha256 = legacy_choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
            paired_choice.runtime_name: ApprovedArtifactHash(
                asset_name = paired_choice.runtime_name,
                sha256 = paired_choice.runtime_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )

    # Metadata written for the legacy (no-pair) choice.
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = legacy_choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    # The paired choice's fingerprint must differ from the legacy one so the install refreshes.
    legacy_fingerprint = INSTALL_LLAMA_PREBUILT.expected_install_fingerprint(
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = legacy_choice,
        approved_checksums = checksums,
    )
    paired_fingerprint = INSTALL_LLAMA_PREBUILT.expected_install_fingerprint(
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = paired_choice,
        approved_checksums = checksums,
    )
    assert legacy_fingerprint != paired_fingerprint, (
        "expected_install_fingerprint must hash runtime_name/runtime_sha256 "
        "so pre-#5322 installs are not falsely considered up-to-date"
    )

    paired_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [paired_choice],
        approved_checksums = checksums,
    )
    assert existing_install_matches_plan(install_dir, host, paired_plan) is False


def test_existing_install_matches_plan_macos_requires_dylibs(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_macos_install_shape(install_dir)

    host = HostInfo(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-macos-arm64.tar.gz",
        url = "https://example.com/x.tar.gz",
        source_label = "published",
        install_kind = "macos-arm64",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    assert existing_install_matches_plan(install_dir, host, plan) is True
    (install_dir / "build" / "bin" / "libggml.0.dylib").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False


def test_install_prebuilt_skips_download_when_existing_install_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
            reason = None,
        ),
    )

    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )
    (install_dir / "AGENTS.md").write_text("old root instructions", encoding = "utf-8")
    nested_agents = install_dir / "examples" / "AGENTS.md"
    nested_agents.parent.mkdir()
    nested_agents.write_text("old nested instructions", encoding = "utf-8")
    (install_dir / "CLAUDE.md").write_text("old Claude instructions", encoding = "utf-8")
    (nested_agents.parent / "CLAUDE.md").write_text(
        "old nested Claude instructions", encoding = "utf-8"
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matching install should skip before validation model download")
        ),
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")
    assert not list(install_dir.rglob("AGENTS.md"))
    assert not list(install_dir.rglob("CLAUDE.md"))


def test_setup_scripts_prune_agent_files_without_shipping_a_repo_copy():
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

    assert (
        "_remove_agent_instruction_files \\\n"
        '    "$SCRIPT_DIR/frontend/node_modules" \\\n'
        '    "$_OXC_DIR/node_modules"'
    ) in setup_sh
    assert '_remove_agent_instruction_files "$SCRIPT_DIR/frontend" "$_OXC_DIR"' not in setup_sh
    assert '_remove_agent_instruction_files "$LLAMA_CPP_DIR"' in setup_sh
    assert "-name 'CLAUDE.md'" in setup_sh
    assert 'if [ ! -L "$LLAMA_CPP_DIR" ] && {' in setup_sh
    assert '${_LOCAL_LLAMA_CPP_LINKED:-false}" != true' not in setup_sh
    assert "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" in setup_sh
    assert '_studio_owned_adoptable "$LLAMA_CPP_DIR"' in setup_sh
    assert (
        "Remove-AgentInstructionFiles -Roots @(\n"
        '    (Join-Path $FrontendDir "node_modules"),\n'
        '    (Join-Path $OxcValidatorDir "node_modules")\n'
        ")"
    ) in setup_ps1
    assert "Remove-AgentInstructionFiles -Roots @($FrontendDir, $OxcValidatorDir)" not in setup_ps1
    assert '"CLAUDE.md"' in setup_ps1
    assert '-Include "AGENTS.md", "CLAUDE.md"' not in setup_ps1
    assert '$child.Name -in @("AGENTS.md", "CLAUDE.md")' in setup_ps1
    assert "$llamaCppIsLink" in setup_ps1
    assert "if (-not $LocalLlamaCppLinked)" not in setup_ps1
    assert "Join-Path $LlamaCppDir $StudioOwnedMarker" in setup_ps1
    assert "Test-StudioOwnedAdoptable $LlamaCppDir" in setup_ps1
    assert (
        "Copy-Item -Recurse -LiteralPath $ResolvedLocal -Destination $LlamaCppDir\n"
        "            Remove-AgentInstructionFiles -Roots @($LlamaCppDir)"
    ) in setup_ps1
    assert not (PACKAGE_ROOT / "studio" / "frontend" / "src" / "i18n" / "AGENTS.md").exists()
    assert (PACKAGE_ROOT / "studio" / "frontend" / "src" / "i18n" / "README.md").is_file()


def test_setup_sh_cleanup_unlinks_instruction_symlink_only(tmp_path: Path):
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")

    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    start = setup_sh.index("_remove_agent_instruction_files() {")
    end = setup_sh.index("\n}\n", start) + 2
    function = setup_sh[start:end]
    managed = tmp_path / "managed"
    external = tmp_path / "external.md"
    managed.mkdir()
    external.write_text("external", encoding = "utf-8")
    instruction = managed / "AGENTS.md"
    try:
        instruction.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    subprocess.run(
        ["bash", "-c", function + '\n_remove_agent_instruction_files "$1"', "bash", str(managed)],
        check = True,
    )

    assert not os.path.lexists(instruction)
    assert external.read_text(encoding = "utf-8") == "external"


def test_install_prebuilt_does_not_skip_unhealthy_existing_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)
    (install_dir / "llama-quantize").unlink()

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [choice],
        approved_checksums = checksums,
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unhealthy install must continue into normal install flow")
        ),
    )

    with pytest.raises(
        AssertionError, match = "unhealthy install must continue into normal install flow"
    ):
        install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")


def test_install_prebuilt_skips_when_older_release_fallback_matches_existing_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
            reason = None,
        ),
    )

    latest_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-2",
        name = "llama-b9002-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9002-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "c" * 64,
    )
    fallback_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    latest_checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-2",
        upstream_tag = "b9002",
        source_commit = "beadfeed",
        artifacts = {
            source_archive_logical_name("b9002"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9002"),
                sha256 = "d" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            latest_choice.name: ApprovedArtifactHash(
                asset_name = latest_choice.name,
                sha256 = latest_choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    fallback_checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            fallback_choice.name: ApprovedArtifactHash(
                asset_name = fallback_choice.name,
                sha256 = fallback_choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    latest_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9002",
        release_tag = "release-2",
        attempts = [latest_choice],
        approved_checksums = latest_checksums,
    )
    fallback_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [fallback_choice],
        approved_checksums = fallback_checksums,
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = fallback_choice,
        approved_checksums = fallback_checksums,
        prebuilt_fallback_used = True,
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [latest_plan, fallback_plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda probe_path, cache_path: probe_path.write_bytes(b"probe"),
    )

    call_log: list[str] = []

    def fake_validate(
        attempts,
        host,
        install_dir,
        work_dir,
        probe_path,
        *,
        requested_tag,
        llama_tag,
        release_tag,
        approved_checksums,
        initial_fallback_used = False,
        existing_install_dir = None,
    ):
        call_log.append(llama_tag)
        raise PrebuiltFallback("validation failed for latest release")

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "validate_prebuilt_attempts",
        fake_validate,
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "activate_install_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("matching fallback install should not reactivate")
        ),
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert call_log == ["b9002"]


def test_install_prebuilt_skips_same_release_fallback_attempt_when_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
            reason = None,
        ),
    )

    first_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64-bad.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64-bad.tar.gz",
        source_label = "published",
        install_kind = "linux-cpu",
        expected_sha256 = "c" * 64,
    )
    fallback_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64-good.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64-good.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            first_choice.name: ApprovedArtifactHash(
                asset_name = first_choice.name,
                sha256 = first_choice.expected_sha256,
                repo = "unslothai/llama.cpp",
                kind = "prebuilt",
            ),
            fallback_choice.name: ApprovedArtifactHash(
                asset_name = fallback_choice.name,
                sha256 = fallback_choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [first_choice, fallback_choice],
        approved_checksums = checksums,
    )

    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = fallback_choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = True,
    )
    assert (
        existing_install_matches_choice(
            install_dir,
            host,
            llama_tag = "b9001",
            release_tag = "release-1",
            choice = fallback_choice,
            approved_checksums = checksums,
        )
        is True
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda probe_path, cache_path: probe_path.write_bytes(b"probe"),
    )

    attempted_names: list[str] = []

    def fake_validate_choice(
        choice,
        host,
        staging_dir,
        work_dir,
        probe_path,
        *,
        requested_tag,
        llama_tag,
        release_tag,
        approved_checksums,
        prebuilt_fallback_used,
        quantized_path,
    ):
        attempted_names.append(choice.name)
        if choice.name == first_choice.name:
            raise PrebuiltFallback("newest candidate failed")
        raise AssertionError("installed fallback candidate should have been skipped")

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "validate_prebuilt_choice",
        fake_validate_choice,
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "activate_install_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("installed fallback candidate should not be activated")
        ),
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert attempted_names == [first_choice.name]


def test_install_prebuilt_same_tag_upstream_failure_uses_older_unsloth_release_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "llama.cpp"
    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    same_tag_upstream_choice = AssetChoice(
        repo = "ggml-org/llama.cpp",
        tag = "b9002",
        name = "llama-b9002-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9002-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    older_release_choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "b" * 64,
    )
    latest_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9002",
        release_tag = "release-2",
        attempts = [same_tag_upstream_choice],
        approved_checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "release-2",
            upstream_tag = "b9002",
            source_commit = None,
            artifacts = {},
        ),
    )
    older_plan = INSTALL_LLAMA_PREBUILT.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        attempts = [older_release_choice],
        approved_checksums = ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "release-1",
            upstream_tag = "b9001",
            source_commit = None,
            artifacts = {},
        ),
    )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [latest_plan, older_plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda probe_path, cache_path: probe_path.write_bytes(b"probe"),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "latest_upstream_release_tag",
        lambda: (_ for _ in ()).throw(
            AssertionError("install fallback should not walk upstream releases")
        ),
    )

    attempted = []

    def fake_validate(
        attempts,
        host,
        install_dir,
        work_dir,
        probe_path,
        *,
        requested_tag,
        llama_tag,
        release_tag,
        approved_checksums,
        initial_fallback_used = False,
        existing_install_dir = None,
    ):
        attempted.append((llama_tag, release_tag, attempts[0].source_label))
        if llama_tag == "b9002":
            raise PrebuiltFallback("same-tag upstream asset failed validation")
        staging_dir = create_install_staging_dir(install_dir)
        (staging_dir / "marker.txt").write_text("ready\n")
        return attempts[0], staging_dir, initial_fallback_used

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "validate_prebuilt_attempts", fake_validate)

    activated = {}
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "activate_install_tree",
        lambda staging_dir, install_dir, host: activated.update(
            {"staging_dir": staging_dir, "install_dir": install_dir}
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "ensure_converter_scripts",
        lambda install_dir, llama_tag: None,
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert attempted == [("b9002", "release-2", "upstream"), ("b9001", "release-1", "upstream")]
    assert activated["install_dir"] == install_dir


def io_bytes(data: bytes):
    return io.BytesIO(data)


def add_bytes_to_tar(
    archive: tarfile.TarFile,
    name: str,
    data: bytes,
    *,
    mode: int = 0o644,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    archive.addfile(info, io_bytes(data))


def add_symlink_to_tar(archive: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    archive.addfile(info)


def test_existing_install_matches_choice_fails_when_install_tree_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """confirm_install_tree guard rejects installs missing critical files."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)

    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_ldd_probe",
        lambda _binary_path, *, env: LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_OK,
            missing = [],
            reason = None,
        ),
    )

    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-ubuntu-x64.tar.gz",
        url = "https://example.com/llama-b9001-bin-ubuntu-x64.tar.gz",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    # Full install should match
    assert (
        existing_install_matches_choice(
            install_dir,
            host,
            llama_tag = "b9001",
            release_tag = "release-1",
            choice = choice,
            approved_checksums = checksums,
        )
        is True
    )

    # Remove convert_hf_to_gguf.py (confirm_install_tree checks it; runtime health does not).
    (install_dir / "convert_hf_to_gguf.py").unlink()
    assert (
        existing_install_matches_choice(
            install_dir,
            host,
            llama_tag = "b9001",
            release_tag = "release-1",
            choice = choice,
            approved_checksums = checksums,
        )
        is False
    )


def test_existing_install_matches_choice_fails_when_install_tree_incomplete_macos(tmp_path: Path):
    """confirm_install_tree guard rejects macOS arm64 installs missing critical files."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_macos_install_shape(install_dir)

    host = HostInfo(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = "llama-b9001-bin-macos-arm64.tar.gz",
        url = "https://example.com/llama-b9001-bin-macos-arm64.tar.gz",
        source_label = "upstream",
        install_kind = "macos-arm64",
        expected_sha256 = "a" * 64,
    )
    checksums = ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = {
            source_archive_logical_name("b9001"): ApprovedArtifactHash(
                asset_name = source_archive_logical_name("b9001"),
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-prebuilt",
            ),
        },
    )
    write_prebuilt_metadata(
        install_dir,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )

    # Full install should match
    assert (
        existing_install_matches_choice(
            install_dir,
            host,
            llama_tag = "b9001",
            release_tag = "release-1",
            choice = choice,
            approved_checksums = checksums,
        )
        is True
    )

    # Remove a macOS-specific runtime artifact and verify the guard catches it
    (install_dir / "build" / "bin" / "libmtmd.0.dylib").unlink()
    assert (
        existing_install_matches_choice(
            install_dir,
            host,
            llama_tag = "b9001",
            release_tag = "release-1",
            choice = choice,
            approved_checksums = checksums,
        )
        is False
    )


def test_paired_runtime_dll_patterns_excludes_executables() -> None:
    """The paired runtime archive must contribute only CUDA DLLs (no *.exe/*.dll) so it can't overwrite binaries."""
    paired_runtime_dll_patterns = INSTALL_LLAMA_PREBUILT.paired_runtime_dll_patterns
    paired_choice = AssetChoice(
        repo = "x",
        tag = "t",
        name = "llama-b9001-bin-win-cuda-12.4-x64.zip",
        url = "u",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = "a" * 64,
        runtime_name = "cudart-llama-bin-win-cuda-12.4-x64.zip",
        runtime_url = "https://example.com/cudart.zip",
        runtime_sha256 = "c" * 64,
    )
    patterns = paired_runtime_dll_patterns(paired_choice)
    assert "cudart64_*.dll" in patterns
    assert "cublas64_*.dll" in patterns
    assert "cublasLt64_*.dll" in patterns
    assert "*.exe" not in patterns
    assert "*.dll" not in patterns

    for kind in (
        "linux-cpu",
        "linux-cuda",
        "linux-rocm",
        "macos-arm64",
        "macos-x64",
        "windows-cpu",
        "windows-hip",
    ):
        non_windows = AssetChoice(
            repo = "x",
            tag = "t",
            name = "x",
            url = "u",
            source_label = "published",
            install_kind = kind,
            expected_sha256 = "a" * 64,
        )
        assert paired_runtime_dll_patterns(non_windows) == []


def test_runtime_overlay_cannot_overwrite_main_archive_payload(tmp_path: Path) -> None:
    """A malformed runtime archive with llama-server.exe must NOT replace the main archive's binary."""
    install_from_archives = INSTALL_LLAMA_PREBUILT.install_from_archives

    work = tmp_path / "work"
    install = tmp_path / "install"
    archives = tmp_path / "archives"
    work.mkdir()
    install.mkdir()
    archives.mkdir()

    main_zip = archives / "llama-b9001-bin-win-cuda-12.4-x64.zip"
    runtime_zip = archives / "cudart-llama-bin-win-cuda-12.4-x64.zip"
    with zipfile.ZipFile(main_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("llama-server.exe", b"MAIN-SERVER")
        zf.writestr("llama-quantize.exe", b"MAIN-Q")
        zf.writestr("llama.dll", b"DLL-llama")
        zf.writestr("ggml-cuda.dll", b"DLL-ggml")
    import hashlib

    main_sha = hashlib.sha256(main_zip.read_bytes()).hexdigest()
    with zipfile.ZipFile(runtime_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cudart64_12.dll", b"DLL-cudart")
        zf.writestr("cublas64_12.dll", b"DLL-cublas")
        zf.writestr("cublasLt64_12.dll", b"DLL-cublasLt")
        zf.writestr("llama-server.exe", b"RUNTIME-OVERWRITE")
    runtime_sha = hashlib.sha256(runtime_zip.read_bytes()).hexdigest()

    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = main_zip.name,
        url = f"https://example.com/{main_zip.name}",
        source_label = "published",
        install_kind = "windows-cuda",
        runtime_line = "cuda12",
        expected_sha256 = main_sha,
        runtime_name = runtime_zip.name,
        runtime_url = f"https://example.com/{runtime_zip.name}",
        runtime_sha256 = runtime_sha,
    )
    host = HostInfo(
        system = "Windows",
        machine = "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (12, 4),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = True,
    )

    import shutil as _shutil

    orig_download = INSTALL_LLAMA_PREBUILT.download_file_verified

    def fake_download(
        url,
        target_path,
        *,
        expected_sha256 = None,
        label = None,
        **kw,
    ):
        src = main_zip if "cudart" not in url else runtime_zip
        _shutil.copy2(src, target_path)
        if expected_sha256:
            actual = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise INSTALL_LLAMA_PREBUILT.PrebuiltFallback(f"sha256 mismatch on {label}")

    INSTALL_LLAMA_PREBUILT.download_file_verified = fake_download
    try:
        install_from_archives(choice, host, install, work)
    finally:
        INSTALL_LLAMA_PREBUILT.download_file_verified = orig_download

    release_dir = install / "build" / "bin" / "Release"
    server = release_dir / "llama-server.exe"
    assert server.exists()
    assert server.read_bytes() == b"MAIN-SERVER", (
        "runtime archive overwrote main llama-server.exe; " f"got {server.read_bytes()!r}"
    )
    for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
        assert (release_dir / name).exists(), f"missing {name}"


def test_linux_runtime_overlay_copies_llama_tool_impl_libraries(tmp_path: Path) -> None:
    install_from_archives = INSTALL_LLAMA_PREBUILT.install_from_archives

    work = tmp_path / "work"
    install = tmp_path / "install"
    archives = tmp_path / "archives"
    work.mkdir()
    install.mkdir()
    archives.mkdir()

    bundle = archives / "app-b9334-linux-x64-cuda13-newer.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in (
            "llama-cli",
            "llama-server",
            "llama-quantize",
            "libllama-cli-impl.so",
            "libllama-server-impl.so",
            "libllama-quantize-impl.so",
            "libllama-common.so",
            "libllama.so",
            "libggml.so",
            "libggml-base.so",
            "libmtmd.so",
            "libggml-cpu-x64.so",
            "libggml-cuda.so",
        ):
            payload = f"{name}\n".encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    import hashlib
    import shutil as _shutil

    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    choice = AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = "b9334",
        name = bundle.name,
        url = f"https://example.com/{bundle.name}",
        source_label = "published",
        install_kind = "linux-cuda",
        runtime_line = "cuda13",
        expected_sha256 = bundle_sha,
    )
    host = HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = (13, 0),
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )

    orig_download = INSTALL_LLAMA_PREBUILT.download_file_verified

    def fake_download(
        url,
        target_path,
        *,
        expected_sha256 = None,
        label = None,
        **kw,
    ):
        _shutil.copy2(bundle, target_path)
        if expected_sha256:
            actual = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise INSTALL_LLAMA_PREBUILT.PrebuiltFallback(f"sha256 mismatch on {label}")

    INSTALL_LLAMA_PREBUILT.download_file_verified = fake_download
    try:
        install_from_archives(choice, host, install, work)
    finally:
        INSTALL_LLAMA_PREBUILT.download_file_verified = orig_download

    runtime_dir = install / "build" / "bin"
    for name in (
        "libllama-cli-impl.so",
        "libllama-server-impl.so",
        "libllama-quantize-impl.so",
    ):
        assert (runtime_dir / name).exists(), f"missing {name}"
    assert not (runtime_dir / "llama-cli").exists()


def test_python_runtime_dirs_covers_cu13_and_library_bin(monkeypatch, tmp_path: Path) -> None:
    """Installer DLL discovery must scan the same path set as the backend (cu12/cu13/conda layouts + torch/lib)."""
    import site as _site

    python_runtime_dirs = INSTALL_LLAMA_PREBUILT.python_runtime_dirs

    site_dir = tmp_path / "Lib" / "site-packages"
    # cu12-style modular wheel
    cu12_bin = site_dir / "nvidia" / "cuda_runtime" / "bin"
    cu12_bin.mkdir(parents = True)
    # cu13-style unsuffixed wheel
    cu13_arch = site_dir / "nvidia" / "cu13" / "bin" / "x86_64"
    cu13_arch.mkdir(parents = True)
    # conda-style repack
    library_bin = site_dir / "nvidia" / "cublas" / "Library" / "bin"
    library_bin.mkdir(parents = True)
    # PyTorch bundled-CUDA wheel
    torch_lib = site_dir / "torch" / "lib"
    torch_lib.mkdir(parents = True)

    monkeypatch.setattr(sys, "path", [str(site_dir)])
    monkeypatch.setattr(_site, "getsitepackages", lambda: [str(site_dir)])
    monkeypatch.setattr(_site, "getusersitepackages", lambda: "")

    dirs = python_runtime_dirs()
    assert str(cu12_bin) in dirs
    assert str(cu13_arch) in dirs
    assert str(library_bin) in dirs
    assert str(torch_lib) in dirs


def _nvidia_linux_host():
    return HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = ["10.0"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )


def test_build_validation_sandbox_plan_contract():
    assert (
        build_validation_sandbox_plan(
            ["cmd", "arg"],
            binary_path = Path("/tmp/bin"),
            install_dir = Path("/tmp/install"),
            host = linux_host(),
            purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_LDD,
            runtime_line = None,
            env = {"UNSANDBOXABLE": "1"},
        ).purpose
        == INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_LDD
    )


def test_build_validation_sandbox_plan_linux_without_bwrap_skips_ldd_probe(monkeypatch):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: None if command == "bwrap" else "/usr/bin/bwrap",
    )
    plan = build_validation_sandbox_plan(
        ["ldd", "/tmp/bin"],
        binary_path = Path("/tmp/bin"),
        install_dir = Path("/tmp/install"),
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_LDD,
        runtime_line = None,
        env = {},
    )
    assert plan.is_skipped
    assert plan.reason is not None
    assert "skip ldd probe" in plan.reason


def test_build_validation_sandbox_plan_linux_without_bwrap_skips_validation(monkeypatch):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: None if command == "bwrap" else "/usr/bin/bwrap",
    )
    plan = build_validation_sandbox_plan(
        ["llama-quantize", "in", "out"],
        binary_path = Path("/tmp/bin"),
        install_dir = Path("/tmp/install"),
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE,
        runtime_line = None,
        env = {},
    )
    assert plan.is_skipped
    assert plan.reason is not None
    assert "skip downloaded-binary validation" in plan.reason
    assert plan.network_policy is None
    assert plan.server_probe_mode is None


def test_build_validation_sandbox_plan_linux_unusable_bwrap_skips_ldd(monkeypatch):
    # bwrap present but unable to create namespaces (restricted userns) must degrade
    # like an absent bwrap so the mandatory ldd preflight does not reject the prebuilt.
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/usr/bin/bwrap" if command == "bwrap" else "/usr/bin/" + command,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_bwrap_can_sandbox", lambda _p: False)
    plan = build_validation_sandbox_plan(
        ["ldd", "/tmp/bin"],
        binary_path = Path("/tmp/bin"),
        install_dir = Path("/tmp/install"),
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_LDD,
        runtime_line = None,
        env = {},
    )
    assert plan.is_skipped
    assert plan.reason is not None
    assert "skip ldd probe" in plan.reason


def test_bwrap_capability_probe_uses_clean_launcher_env(monkeypatch):
    captured: dict[str, dict[str, str]] = {}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/bad/loader")
    monkeypatch.setenv("LD_PRELOAD", "/bad/preload.so")
    INSTALL_LLAMA_PREBUILT._bwrap_sandbox_capability.clear()

    def fake_run(*_args, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args = [], returncode = 0)

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.subprocess, "run", fake_run)

    assert bwrap_can_sandbox("/usr/bin/bwrap")
    assert "LD_LIBRARY_PATH" not in captured["env"]
    assert "LD_PRELOAD" not in captured["env"]
    assert captured["env"]["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def test_build_validation_sandbox_plan_linux_with_bwrap_runs(monkeypatch, tmp_path):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path) if command == "bwrap" else None,
    )
    helper_bin = tmp_path / "helper" / "bin"
    helper_lib = tmp_path / "helper" / "lib"
    install_dir = tmp_path / "install"
    binary_dir = tmp_path / "bin"
    model_dir = tmp_path / "models"
    for directory in (helper_bin, helper_lib, install_dir, binary_dir, model_dir):
        directory.mkdir(parents = True, exist_ok = True)
    helper_path = helper_bin / "python3"
    helper_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_linux_validation_server_probe_command",
        lambda command, payload_env, timeout = 60: [str(helper_path), "-c", "server probe"],
    )
    binary_path = binary_dir / "llama-server"
    binary_path.write_text("")
    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(model_dir / "stories260K.gguf"), "--port", "7777"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": "/tmp/payload/libs"},
    )
    assert plan.is_runnable
    assert plan.command[0] == str(bwrap_path)
    assert plan.command[1] == "--unshare-all"
    assert "--share-net" not in plan.command
    assert "--setenv" in plan.command
    assert not _command_has_setenv(plan, "LD_LIBRARY_PATH")
    assert "--tmpfs" in plan.command
    assert "--perms" in plan.command
    assert "1777" in plan.command
    assert "/tmp" in plan.command
    assert plan.env.get("LD_LIBRARY_PATH") is None
    assert plan.payload_command == [
        "llama-server",
        "-m",
        str(model_dir / "stories260K.gguf"),
        "--port",
        "7777",
    ]
    assert plan.payload_env == {"LD_LIBRARY_PATH": "/tmp/payload/libs"}
    assert "--ro-bind" in plan.command
    assert "--bind" in plan.command
    assert "--dev" in plan.command
    assert "/dev/nvidiactl" not in plan.command
    assert plan.network_policy == INSTALL_LLAMA_PREBUILT._VALIDATION_NETWORK_POLICY_SANDBOX
    assert str(model_dir) in plan.command
    assert str(helper_lib) in plan.command


def test_build_validation_sandbox_plan_linux_skips_broad_inherited_library_binds(
    monkeypatch, tmp_path
):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path) if command == "bwrap" else None,
    )
    install_dir = tmp_path / "install"
    binary_dir = tmp_path / "bin"
    runtime_lib = tmp_path / "runtime" / "lib"
    model_dir = tmp_path / "models"
    for directory in (install_dir, binary_dir, runtime_lib, model_dir):
        directory.mkdir(parents = True, exist_ok = True)
    binary_path = binary_dir / "llama-server"
    binary_path.write_text("")

    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(model_dir / "stories260K.gguf"), "--port", "7777"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {
            "LD_LIBRARY_PATH": os.pathsep.join(
                [str(Path("/")), str(Path("/home/alice")), str(runtime_lib)]
            )
        },
    )

    assert plan.is_runnable
    assert _command_has_bind(plan, "--ro-bind", runtime_lib)
    assert not _command_has_bind(plan, "--ro-bind", Path("/"))
    assert not _command_has_bind(plan, "--ro-bind", Path("/home/alice"))


def test_build_validation_sandbox_plan_linux_rocm_gpu_validation_binds_vulkan_render_nodes(
    monkeypatch, tmp_path
):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path) if command == "bwrap" else None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_binary_is_setuid_root", lambda _path: True)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_linux_validation_server_probe_command",
        lambda command, payload_env, timeout = 60: command,
    )
    install_dir = tmp_path / "install"
    binary_dir = tmp_path / "bin"
    model_dir = tmp_path / "models"
    for directory in (install_dir, binary_dir, model_dir):
        directory.mkdir(parents = True, exist_ok = True)
    binary_path = binary_dir / "llama-server"
    binary_path.write_text("")

    original_glob = Path.glob
    original_exists = Path.exists

    def fake_glob(path: Path, pattern: str):
        normalized = str(path).replace("\\", "/")
        if normalized == "/dev/dri" and pattern == "card*":
            return [Path("/dev/dri/card0")]
        if normalized == "/dev/dri" and pattern == "renderD*":
            return [Path("/dev/dri/renderD128")]
        if normalized == "/dev/nvidia-caps" and pattern == "nvidia-cap*":
            return []
        return original_glob(path, pattern)

    def fake_exists(path: Path) -> bool:
        normalized = str(path).replace("\\", "/")
        if normalized in {
            "/dev/dri",
            "/dev/dri/card0",
            "/dev/dri/renderD128",
            "/etc/vulkan",
            "/usr/share/vulkan",
        }:
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "glob", fake_glob)
    monkeypatch.setattr(Path, "exists", fake_exists)

    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(model_dir / "stories260K.gguf"), "--port", "7777"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": str(binary_dir)},
        enable_gpu_layers = True,
        gpu_backend = "rocm",
    )

    assert plan.is_runnable
    assert _command_has_bind(plan, "--dev-bind-try", Path("/dev/dri/card0"))
    assert _command_has_bind(plan, "--dev-bind-try", Path("/dev/dri/renderD128"))
    assert _command_has_bind(plan, "--ro-bind", Path("/etc/vulkan"))
    assert _command_has_bind(plan, "--ro-bind", Path("/usr/share/vulkan"))


def test_build_validation_sandbox_plan_linux_server_probe_uses_resolved_helper_path(
    monkeypatch, tmp_path
):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path.resolve()) if command == "bwrap" else None,
    )
    helper_bin = tmp_path / "helper" / "bin"
    helper_store = tmp_path / "nix" / "store"
    install_dir = tmp_path / "install"
    binary_dir = tmp_path / "bin"
    model_dir = tmp_path / "models"
    for directory in (helper_bin, helper_store, install_dir, binary_dir, model_dir):
        directory.mkdir(parents = True, exist_ok = True)
    helper_symlink = helper_bin / "python3"
    helper_symlink.write_text("")
    helper_target = helper_store / "python3"
    helper_target.write_text("")
    binary_path = binary_dir / "llama-server"
    binary_path.write_text("")

    original_resolve = Path.resolve

    def fake_resolve(path: Path, strict: bool = False):
        if path == helper_symlink:
            return helper_target
        return original_resolve(path, strict = strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_linux_validation_server_probe_command",
        lambda command, payload_env, timeout = 60: [str(helper_symlink), "-c", "server probe"],
    )

    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(model_dir / "stories260K.gguf"), "--port", "7777"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": "/tmp/payload/libs"},
    )

    assert plan.is_runnable
    assert str(helper_target) in plan.command
    assert str(helper_target.parent) in plan.command
    assert str(helper_symlink) not in plan.command


def test_build_validation_sandbox_plan_linux_server_probe_binds_nix_store(monkeypatch, tmp_path):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path.resolve()) if command == "bwrap" else None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_binary_is_setuid_root", lambda _path: True)
    install_dir = tmp_path / "install"
    binary_dir = tmp_path / "bin"
    model_dir = tmp_path / "models"
    for directory in (install_dir, binary_dir, model_dir):
        directory.mkdir(parents = True, exist_ok = True)
    helper_symlink = tmp_path / "helper-python"
    helper_symlink.write_text("")
    binary_path = binary_dir / "llama-server"
    binary_path.write_text("")

    original_resolve = Path.resolve
    original_exists = Path.exists

    def fake_resolve(path: Path, strict: bool = False):
        if path == helper_symlink:
            return Path("/nix/store/fake-python/bin/python3")
        return original_resolve(path, strict = strict)

    def fake_exists(path: Path) -> bool:
        normalized = str(path).replace("\\", "/")
        if normalized in {
            "/nix/store",
            "/nix/store/fake-python/bin/python3",
            "/nix/store/fake-python/bin",
        }:
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_linux_validation_server_probe_command",
        lambda command, payload_env, timeout = 60: [str(helper_symlink), "-c", "server probe"],
    )

    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(model_dir / "stories260K.gguf"), "--port", "7777"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": "/tmp/payload/libs"},
    )

    assert plan.is_runnable
    assert _command_contains_path(plan, "nix/store")


def test_build_validation_sandbox_plan_linux_gpu_keeps_sandbox_without_setuid_bwrap(
    monkeypatch, tmp_path
):
    bwrap_path = tmp_path / "bwrap"
    bwrap_path.write_text("")
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("")
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    helper_path = tmp_path / "python3"
    helper_path.write_text("")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: str(bwrap_path.resolve()) if command == "bwrap" else None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_binary_is_setuid_root", lambda _path: False)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_linux_validation_server_probe_command",
        lambda command, payload_env, timeout = 60: seen.update(
            {
                "command": list(command),
                "payload_env": dict(payload_env),
                "timeout": timeout,
            }
        )
        or [str(helper_path), "-c", "server probe"],
    )

    plan = build_validation_sandbox_plan(
        [str(binary_path), "--port", "7777", "--n-gpu-layers", "1"],
        binary_path = binary_path,
        install_dir = install_dir,
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": "/tmp/payload/libs"},
        enable_gpu_layers = True,
        gpu_backend = "cuda",
    )

    assert plan.is_runnable
    assert plan.command[0] == str(bwrap_path.resolve())
    assert plan.sandbox_kind == "linux_bwrap"
    assert plan.network_policy == INSTALL_LLAMA_PREBUILT._VALIDATION_NETWORK_POLICY_SANDBOX
    assert plan.server_probe_mode == INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX
    assert seen["command"] == [str(binary_path), "--port", "7777"]
    assert seen["payload_env"] == {"LD_LIBRARY_PATH": "/tmp/payload/libs"}
    assert plan.payload_command == [str(binary_path), "--port", "7777"]
    assert "--n-gpu-layers" not in plan.command


def test_linux_validation_server_probe_command_passes_payload_env_to_server_spawn():
    probe_command = INSTALL_LLAMA_PREBUILT._linux_validation_server_probe_command(
        [
            "/opt/llama-server",
            "-m",
            "/tmp/models/stories260K.gguf",
            "--port",
            "7777",
        ],
        {
            "LD_LIBRARY_PATH": "/tmp/libs",
            "PYTHONPATH": "/tmp/python",
        },
    )
    assert len(probe_command) == 3
    _, script = probe_command[0], probe_command[2]
    assert 'payload_env = {"LD_LIBRARY_PATH": "/tmp/libs", "PYTHONPATH": "/tmp/python"}' in script
    assert "server_env = dict(os.environ)" in script
    assert "server_env.update(payload_env)" in script
    assert "timeout = 5" in script
    assert "with urllib.request.urlopen(request, timeout = 5)" in script


def test_run_validation_capture_uses_launcher_env_for_linux_bwrap(monkeypatch):
    launcher_env = {"PATH": "/usr/bin", "UNSANDBOXABLE": "1"}
    plan = ValidationLaunchPlan(
        command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session"],
        env = launcher_env,
        action = "run",
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE,
        sandbox_kind = "linux_bwrap",
        payload_command = ["llama-quantize", "in", "out"],
        payload_env = {"LD_LIBRARY_PATH": "/payload/lib"},
        server_probe_mode = None,
    )

    captured: dict[str, dict[str, str] | None] = {}

    def fake_run_capture(
        command,
        *,
        timeout,
        env = None,
        check = False,
    ):
        captured["env"] = dict(env or {})
        return subprocess.CompletedProcess(command, 0, stdout = "ok")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "run_capture", fake_run_capture)
    result = run_validation_capture(plan, timeout = 5)
    assert result.returncode == 0
    assert captured["env"] == launcher_env
    assert captured["env"] is not None
    assert "LD_LIBRARY_PATH" not in (captured["env"] or {})


def test_run_validation_popen_uses_launcher_env_for_linux_bwrap(monkeypatch):
    launcher_env = {"PATH": "/usr/bin", "UNSANDBOXABLE": "1"}
    plan = ValidationLaunchPlan(
        command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session"],
        env = launcher_env,
        action = "run",
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        sandbox_kind = "linux_bwrap",
        payload_command = ["llama-server"],
        payload_env = {"LD_LIBRARY_PATH": "/payload/lib"},
        server_probe_mode = INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX,
    )

    class _FakeProcess:
        def __init__(self):
            self.started_with: dict[str, object] = {}

        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    fake_process = _FakeProcess()

    def fake_popen(
        command,
        *,
        stdout,
        stderr,
        text,
        env = None,
        **_kwargs,
    ):
        fake_process.started_with = {
            "command": list(command),
            "stdout": stdout,
            "env": dict(env or {}),
        }
        return fake_process

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.subprocess, "Popen", fake_popen)
    process = run_validation_popen(plan, stdout = None)
    assert process is fake_process
    assert fake_process.started_with["command"] == plan.command
    assert fake_process.started_with["env"] == launcher_env
    assert "LD_LIBRARY_PATH" not in (fake_process.started_with["env"] or {})


def test_validate_server_strips_python_import_roots_from_payload_env(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    recorded: dict[str, object] = {}

    monkeypatch.setenv("PYTHONHOME", "/host/python")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda _binary_path: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "free_local_port", lambda: 7777)

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        recorded["env"] = dict(env)
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "run",
            purpose = purpose,
            server_probe_mode = INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX,
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_run_validation_capture",
        lambda plan, *, timeout: subprocess.CompletedProcess(
            plan.command, 0, stdout = "ok", stderr = ""
        ),
    )

    validate_server(
        server_path,
        probe_path,
        linux_host(),
        tmp_path,
        runtime_line = "cuda13",
        install_kind = "linux-cuda",
    )

    env = recorded["env"]
    assert isinstance(env, dict)
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env


def test_build_validation_sandbox_plan_linux_quantize_binds_probe_and_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/tmp/bin/bwrap" if command == "bwrap" else None,
    )
    bin_dir = tmp_path / "bin"
    probe_dir = tmp_path / "models"
    out_dir = tmp_path / "out"
    runtime_dir = tmp_path / "runtime"
    for directory in (bin_dir, probe_dir, out_dir, runtime_dir):
        directory.mkdir()
    plan = build_validation_sandbox_plan(
        [
            str(bin_dir / "llama-quantize"),
            str(probe_dir / "probe.gguf"),
            str(out_dir / "probe-q4.gguf"),
            "Q6_K",
            "2",
        ],
        binary_path = bin_dir / "llama-quantize",
        install_dir = tmp_path / "install",
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE,
        runtime_line = None,
        env = {"LD_LIBRARY_PATH": str(runtime_dir)},
    )
    assert plan.is_runnable
    assert _command_has_setenv(plan, "LD_LIBRARY_PATH")
    assert plan.command.count("--ro-bind") >= 3
    assert plan.command.count("--bind") >= 2
    assert str(probe_dir) in plan.command
    assert str(out_dir) in plan.command


def test_build_validation_sandbox_plan_linux_server_binds_gpu_nodes_when_enabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/tmp/bin/bwrap" if command == "bwrap" else None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_binary_is_setuid_root", lambda _path: True)
    existing_nodes = {
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-modeset",
        "/dev/nvidia0",
        "/dev/nvidia-caps/nvidia-cap1",
        "/dev/kfd",
        "/dev/dxg",
        "/dev/dri/card0",
        "/dev/dri/renderD128",
        "/sys/class/drm",
        "/sys/bus/pci",
        "/sys/dev/char",
        "/sys/devices",
        "/proc/driver/nvidia",
        "/proc/driver/nvidia/capabilities",
    }
    original_exists = Path.exists
    original_glob = Path.glob

    def _norm(path: Path) -> str:
        text = str(path).replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", text):
            text = "/" + text.split(":", 1)[1].lstrip("/")
        return text

    def fake_exists(path: Path) -> bool:
        if _norm(path) in existing_nodes:
            return True
        return original_exists(path)

    def fake_glob(path: Path, pattern: str):
        normalized = _norm(path)
        if normalized == "/dev" and pattern == "nvidia*":
            return [
                Path(node)
                for node in existing_nodes
                if str(node).startswith("/dev/nvidia") and "*" not in node
            ]
        if normalized == "/dev/nvidia-caps" and pattern == "nvidia-cap*":
            return [Path("/dev/nvidia-caps/nvidia-cap1")]
        if normalized == "/dev/dri" and pattern == "card*":
            return [Path("/dev/dri/card0")]
        if normalized == "/dev/dri" and pattern == "renderD*":
            return [Path("/dev/dri/renderD128")]
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "glob", fake_glob)

    plan = build_validation_sandbox_plan(
        ["llama-server", "--help"],
        binary_path = Path("/tmp/bin/llama-server"),
        install_dir = Path("/tmp/install"),
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {},
        enable_gpu_layers = True,
        gpu_backend = "cuda",
    )
    assert plan.is_runnable
    assert "--unshare-all" not in plan.command
    assert "--unshare-net" in plan.command
    assert "--unshare-pid" in plan.command
    assert "--unshare-ipc" in plan.command
    assert "--unshare-uts" in plan.command
    assert "--dev-bind-try" in plan.command
    assert any("nvidiactl" in part for part in plan.command)
    assert any("nvidia0" in part for part in plan.command)
    assert any("nvidia-cap1" in part for part in plan.command)
    assert not any("dxg" in part for part in plan.command)
    assert not _command_contains_path(plan, "dri/card0")
    assert not _command_contains_path(plan, "dri/renderD128")
    assert not any("kfd" in part for part in plan.command)
    assert _command_contains_path(plan, "sys/class/drm")
    assert _command_contains_path(plan, "proc/driver/nvidia")
    assert _command_contains_path(plan, "proc/driver/nvidia/capabilities")


def test_build_validation_sandbox_plan_linux_server_binds_rocm_nodes_when_enabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/tmp/bin/bwrap" if command == "bwrap" else None,
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_binary_is_setuid_root", lambda _path: True)
    existing_nodes = {
        "/dev/kfd",
        "/dev/dxg",
        "/dev/dri/card0",
        "/dev/dri/renderD128",
        "/sys/class/drm",
        "/sys/bus/pci",
        "/sys/dev/char",
        "/sys/devices",
        "/proc/driver/nvidia",
    }
    original_exists = Path.exists
    original_glob = Path.glob

    def _norm(path: Path) -> str:
        text = str(path).replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", text):
            text = "/" + text.split(":", 1)[1].lstrip("/")
        return text

    def fake_exists(path: Path) -> bool:
        if _norm(path) in existing_nodes:
            return True
        return original_exists(path)

    def fake_glob(path: Path, pattern: str):
        normalized = _norm(path)
        if normalized == "/dev" and pattern == "nvidia*":
            return []
        if normalized == "/dev/dri" and pattern == "card*":
            return [Path("/dev/dri/card0")]
        if normalized == "/dev/dri" and pattern == "renderD*":
            return [Path("/dev/dri/renderD128")]
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "glob", fake_glob)

    plan = build_validation_sandbox_plan(
        ["llama-server", "--help"],
        binary_path = Path("/tmp/bin/llama-server"),
        install_dir = Path("/tmp/install"),
        host = linux_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {},
        enable_gpu_layers = True,
        gpu_backend = "rocm",
    )
    assert plan.is_runnable
    assert "--unshare-all" not in plan.command
    assert "--unshare-net" in plan.command
    assert "--unshare-pid" in plan.command
    assert "--unshare-ipc" in plan.command
    assert "--unshare-uts" in plan.command
    assert "--dev-bind-try" in plan.command
    assert any("kfd" in part for part in plan.command)
    assert any("dxg" in part for part in plan.command)
    assert _command_contains_path(plan, "dri/card0")
    assert _command_contains_path(plan, "dri/renderD128")
    assert not any("nvidiactl" in part for part in plan.command)
    assert not _command_contains_path(plan, "proc/driver/nvidia")


def test_build_validation_sandbox_plan_macos_with_and_without_sandbox_exec(monkeypatch):
    binary_path = Path("/tmp/bin/llama-quantize")
    probe_path = Path("/tmp/models/probe.gguf")
    output_path = Path("/tmp/out/probe-q4.gguf")

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/usr/bin/sandbox-exec" if command == "sandbox-exec" else None,
    )
    mac_run = build_validation_sandbox_plan(
        [str(binary_path), str(probe_path), str(output_path), "Q6_K", "2"],
        binary_path = binary_path,
        install_dir = Path("/tmp/install"),
        host = macos_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE,
        runtime_line = None,
        env = {
            "DYLD_LIBRARY_PATH": os.pathsep.join(
                [
                    "/",
                    "/Users/alice",
                    "/Library/Application Support",
                    "/private/var",
                    "/opt/dyld/lib",
                ]
            )
        },
    )
    assert mac_run.is_runnable
    assert mac_run.command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "/usr/bin/env" in mac_run.command
    profile = mac_run.command[2]
    assert "(deny default)" in profile
    assert '(import "bsd.sb")' in profile
    assert "(allow file-map-executable" in profile
    assert '(subpath "/private")' not in profile
    assert '(subpath "/usr")' not in profile
    assert '(subpath "/Library")' not in profile
    assert any(
        f'(subpath "{literal}")' in profile
        for literal in INSTALL_LLAMA_PREBUILT._sandbox_profile_path_literals("/tmp/install")
    )
    assert any(
        f'(subpath "{literal}")' in profile
        for literal in INSTALL_LLAMA_PREBUILT._sandbox_profile_path_literals("/tmp/models")
    )
    assert any(
        f'(subpath "{literal}")' in profile
        for literal in INSTALL_LLAMA_PREBUILT._sandbox_profile_path_literals("/opt/dyld/lib")
    )
    for broad_path in ("/", "/Users/alice", "/Library/Application Support", "/private/var"):
        for literal in INSTALL_LLAMA_PREBUILT._sandbox_profile_path_literals(broad_path):
            assert f'(literal "{literal}")' not in profile
            assert f'(subpath "{literal}")' not in profile
    assert "localhost" not in profile

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_resolve_command_path", lambda *_a, **_k: None)
    mac_skip = build_validation_sandbox_plan(
        [str(binary_path), str(probe_path), str(output_path), "Q6_K", "2"],
        binary_path = binary_path,
        install_dir = Path("/tmp/install"),
        host = macos_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE,
        runtime_line = None,
        env = {},
    )
    assert mac_skip.is_fallback


def test_build_validation_sandbox_plan_macos_server_keeps_loopback(monkeypatch):
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: "/usr/bin/sandbox-exec" if command == "sandbox-exec" else None,
    )
    plan = build_validation_sandbox_plan(
        ["llama-server", "-m", str(Path("/tmp/models/story.gguf")), "--port", "7777"],
        binary_path = Path("/tmp/bin/llama-server"),
        install_dir = Path("/tmp/install"),
        host = macos_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {},
    )
    assert plan.is_runnable
    assert plan.command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert "/usr/bin/env" in plan.command
    profile = plan.command[2]
    assert "(deny default)" in profile
    assert '(import "bsd.sb")' in profile
    assert '(subpath "/private")' not in profile
    assert '(subpath "/usr")' not in profile
    assert '(subpath "/Library")' not in profile
    assert '(allow network* (local ip "localhost:7777"))' in profile
    assert '(allow network* (remote ip "localhost:7777"))' in profile
    assert '(allow network* (local ip "localhost:*"))' not in profile
    assert '(allow network* (remote ip "localhost:*"))' not in profile


def test_build_validation_sandbox_plan_windows_is_unsupported(monkeypatch):
    plan = build_validation_sandbox_plan(
        ["llama-server", "--help"],
        binary_path = Path(r"C:\bin\llama-server.exe"),
        install_dir = Path(r"C:\install"),
        host = windows_host(),
        purpose = INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER,
        runtime_line = None,
        env = {},
    )
    assert plan.is_runnable
    assert plan.sandbox_kind == "windows_direct_validation"
    assert "running validation directly" in (plan.reason or "").lower()


def test_linux_missing_libraries_skips_ldd_without_sandbox_adapter(monkeypatch, tmp_path):
    binary_path = tmp_path / "server"
    binary_path.write_text("")
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_host_is_linux", lambda host = None: True)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: None if command == "bwrap" else "/usr/bin/bwrap",
    )

    captured: dict[str, bool] = {"run": False}

    def fake_run_capture(
        command,
        *,
        timeout,
        env = None,
        check = False,
    ):
        captured["run"] = True
        return subprocess.CompletedProcess(command, 0, stdout = "")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "run_capture", fake_run_capture)
    missing = linux_missing_libraries(binary_path, env = {"LD_LIBRARY_PATH": ""})
    assert missing == []
    assert captured["run"] is False


def test_run_validation_ldd_probe_reports_skipped_status_without_sandbox_adapter(
    monkeypatch, tmp_path
):
    binary_path = tmp_path / "server"
    binary_path.write_text("")
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_host_is_linux", lambda host = None: True)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: None if command == "bwrap" else "/usr/bin/bwrap",
    )
    result = run_validation_ldd_probe(binary_path, env = {"LD_LIBRARY_PATH": ""})
    assert result.status == LINUX_LDD_PROBE_SKIPPED
    assert result.missing == []
    assert result.reason is not None


def test_run_validation_ldd_probe_reports_error_on_nonzero_returncode(monkeypatch, tmp_path):
    binary_path = tmp_path / "server"
    binary_path.write_text("")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_host_is_linux", lambda host = None: True)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_capture(plan, *, timeout: int):
        return subprocess.CompletedProcess(plan.command, 1, stdout = "", stderr = "bwrap: bind failed")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_capture", fake_capture)

    result = run_validation_ldd_probe(binary_path, env = {"LD_LIBRARY_PATH": ""})
    assert result.status == LINUX_LDD_PROBE_ERROR
    assert result.reason is not None
    assert "bind failed" in result.reason


def test_run_validation_ldd_probe_accepts_static_binary_output(monkeypatch, tmp_path):
    binary_path = tmp_path / "server"
    binary_path.write_text("")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_host_is_linux", lambda host = None: True)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_capture(plan, *, timeout: int):
        return subprocess.CompletedProcess(
            plan.command,
            1,
            stdout = "",
            stderr = "not a dynamic executable",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_capture", fake_capture)

    result = run_validation_ldd_probe(binary_path, env = {"LD_LIBRARY_PATH": ""})
    assert result.status == LINUX_LDD_PROBE_OK
    assert result.missing == []
    assert result.reason == "static executable"


def test_linux_missing_libraries_uses_bwrap_plan(monkeypatch, tmp_path):
    binary_path = tmp_path / "server"
    binary_path.write_text("")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_host_is_linux", lambda host = None: True)
    bwrap_path = "/usr/bin/bwrap"
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_resolve_command_path",
        lambda command: bwrap_path if command == "bwrap" else None,
    )
    captured = {}

    def fake_run_capture(
        command,
        *,
        timeout,
        env = None,
        check = False,
    ):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout = "libbad => not found\nlibgood => /tmp/libgood\n",
            stderr = "",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "run_capture", fake_run_capture)
    assert linux_missing_libraries(binary_path, env = {"LD_LIBRARY_PATH": ""}) == ["libbad"]
    assert captured["command"][0] == bwrap_path
    assert any(Path(part).stem == "ldd" for part in captured["command"])


def test_linux_missing_libraries_tolerates_probe_errors(monkeypatch, tmp_path):
    binary_path = tmp_path / "server"
    binary_path.write_text("")

    def fake_probe(_binary_path: Path, *, env: dict[str, str]) -> LinuxLibraryProbeResult:
        return LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_ERROR,
            missing = [],
            reason = "probe crashed",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_ldd_probe", fake_probe)
    assert linux_missing_libraries(binary_path, env = {"LD_LIBRARY_PATH": ""}) == []


def test_preflight_linux_installed_binaries_skips_unknown_when_probe_skipped(monkeypatch, tmp_path):
    host = linux_host()
    binary = tmp_path / "llama-server"
    binary.write_text("")

    def fake_probe(_binary_path: Path, *, env: dict[str, str]) -> LinuxLibraryProbeResult:
        return LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = "no sandbox adapter",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_ldd_probe", fake_probe)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "binary_env",
        lambda binary_path, install_dir, host: {"LD_LIBRARY_PATH": ""},
    )
    preflight_linux_installed_binaries((binary,), tmp_path, host)


def test_preflight_linux_installed_binaries_rejects_skipped_probe_for_reuse(monkeypatch, tmp_path):
    host = linux_host()
    binary = tmp_path / "llama-server"
    binary.write_text("")

    def fake_probe(_binary_path: Path, *, env: dict[str, str]) -> LinuxLibraryProbeResult:
        return LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_SKIPPED,
            missing = [],
            reason = "no sandbox adapter",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_ldd_probe", fake_probe)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "binary_env",
        lambda binary_path, install_dir, host: {"LD_LIBRARY_PATH": ""},
    )
    with pytest.raises(PrebuiltFallback, match = "linux ldd probe skipped"):
        preflight_linux_installed_binaries(
            (binary,),
            tmp_path,
            host,
            allow_skipped_probe = False,
        )


def test_preflight_linux_installed_binaries_errors_on_probe_exception(monkeypatch, tmp_path):
    host = linux_host()
    binary = tmp_path / "llama-server"
    binary.write_text("")

    def fake_probe(_binary_path: Path, *, env: dict[str, str]) -> LinuxLibraryProbeResult:
        return LinuxLibraryProbeResult(
            status = LINUX_LDD_PROBE_ERROR,
            missing = [],
            reason = "probe crashed",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_ldd_probe", fake_probe)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "binary_env",
        lambda binary_path, install_dir, host: {"LD_LIBRARY_PATH": ""},
    )
    with pytest.raises(PrebuiltFallback, match = "linux extracted binary ldd probe errored"):
        preflight_linux_installed_binaries((binary,), tmp_path, host)


def test_validate_quantize_routes_through_sandbox_plan(monkeypatch, tmp_path):
    quantize_path = tmp_path / "llama-quantize"
    quantize_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    quantized_path = tmp_path / "probe-q4.gguf"
    install_dir = tmp_path
    expected_env = {"CUSTOM": "1"}
    recorded: dict[str, list[str] | str | dict[str, str]] = {}

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        recorded["command"] = command
        recorded["purpose"] = purpose
        recorded["env"] = env
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "run",
            purpose = purpose,
        )

    def fake_capture(plan: ValidationLaunchPlan, *, timeout: int):
        assert plan.purpose == INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE
        quantized_path.write_bytes(b"done")
        return subprocess.CompletedProcess(plan.command, 0, stdout = "", stderr = "")

    def fake_binary_env(*_a, **_k):
        return expected_env

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_capture", fake_capture)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "binary_env", fake_binary_env)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "run_capture",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_capture must not be used")),
    )

    validate_quantize(
        quantize_path, probe_path, quantized_path, install_dir, linux_host(), runtime_line = None
    )
    assert recorded["purpose"] == INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_QUANTIZE
    assert recorded["env"] == expected_env
    assert recorded["command"][:3] == [str(quantize_path), str(probe_path), str(quantized_path)]
    assert quantized_path.exists()


def test_validate_quantize_skips_without_linux_sandbox(monkeypatch, tmp_path):
    quantize_path = tmp_path / "llama-quantize"
    quantize_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    quantized_path = tmp_path / "probe-q4.gguf"

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "skip",
            purpose = purpose,
            reason = "no sandbox",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "binary_env", lambda *_a, **_k: {"PATH": str(tmp_path)}
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "run_capture",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_capture must not be used")),
    )

    validate_quantize(
        quantize_path,
        probe_path,
        quantized_path,
        tmp_path,
        linux_host(),
        runtime_line = None,
    )
    assert not quantized_path.exists()


def test_validate_server_routes_through_sandbox_plan(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    install_dir = tmp_path
    recorded: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self):
            recorded["terminated"] = True
            return None

        def kill(self):
            recorded["terminated"] = True
            return None

    def fake_binary_env(*_a, **_k):
        return {"PATH": str(tmp_path)}

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "binary_env", fake_binary_env)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "free_local_port", lambda: 7777)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT.urllib.request, "urlopen", lambda *a, **k: _FakeResponse()
    )
    called_popen = False

    def fake_popen(*_a, **_k):
        nonlocal called_popen
        called_popen = True
        return _FakeProcess()

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_popen", fake_popen)
    called_capture = False

    def fake_capture(_plan, *, timeout: int):
        nonlocal called_capture
        called_capture = True
        return subprocess.CompletedProcess(_plan.command, 0, stdout = "ok")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_capture", fake_capture)

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        recorded["command"] = command
        recorded["purpose"] = purpose
        recorded["enable_gpu_layers"] = enable_gpu_layers
        recorded["gpu_backend"] = gpu_backend
        recorded["server_probe_mode"] = (
            INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX
        )
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "run",
            purpose = purpose,
            sandbox_kind = "linux_bwrap",
            payload_command = command,
            payload_env = env,
            server_probe_mode = INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX,
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)

    validate_server(
        server_path,
        probe_path,
        linux_host(),
        install_dir,
        runtime_line = "cuda13",
        install_kind = "linux-cuda",
    )
    assert recorded["purpose"] == INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER
    command = recorded["command"]
    assert command[:3] == [str(server_path), "-m", str(probe_path)]
    assert "--n-gpu-layers" in command
    assert recorded["enable_gpu_layers"] is True
    assert recorded["gpu_backend"] == "cuda"
    assert called_capture is True
    assert called_popen is False
    assert (
        recorded["server_probe_mode"]
        == INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX
    )


def test_validate_server_uses_extended_capture_timeout_for_in_sandbox_probe(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    install_dir = tmp_path

    called: dict[str, object] = {}

    def fake_binary_env(*_a, **_k):
        return {"PATH": str(tmp_path)}

    def fake_capture(_plan, *, timeout: int):
        called["timeout"] = timeout
        return subprocess.CompletedProcess(_plan.command, 0, stdout = "ok")

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        called["purpose"] = purpose
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "run",
            purpose = purpose,
            sandbox_kind = "linux_bwrap",
            payload_command = command,
            payload_env = env,
            server_probe_mode = INSTALL_LLAMA_PREBUILT._VALIDATION_SERVER_PROBE_MODE_IN_SANDBOX,
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "binary_env", fake_binary_env)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "free_local_port", lambda: 7777)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_run_validation_capture", fake_capture)

    validate_server(
        server_path,
        probe_path,
        linux_host(),
        install_dir,
        runtime_line = None,
        install_kind = "linux-cuda",
    )

    captured_timeout = called.get("timeout")
    assert called.get("purpose") == INSTALL_LLAMA_PREBUILT._VALIDATION_PURPOSE_SERVER
    assert isinstance(captured_timeout, int)
    assert (
        captured_timeout
        == INSTALL_LLAMA_PREBUILT._LINUX_SERVER_VALIDATION_HELPER_CAPTURE_TIMEOUT_SECONDS
    )
    assert captured_timeout > INSTALL_LLAMA_PREBUILT._LINUX_SERVER_VALIDATION_HELPER_TIMEOUT_SECONDS


def test_validate_server_skips_gpu_layers_for_macos_arm64(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "binary_env", lambda *_a, **_k: {"PATH": str(tmp_path)}
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "free_local_port", lambda: 7777)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        recorded["command"] = command
        recorded["enable_gpu_layers"] = enable_gpu_layers
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "fallback",
            purpose = purpose,
            reason = "stop",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)

    with pytest.raises(PrebuiltFallback, match = "stop"):
        validate_server(
            server_path,
            probe_path,
            macos_host(),
            tmp_path,
            runtime_line = None,
            install_kind = "macos-arm64",
        )

    command = recorded["command"]
    assert "--n-gpu-layers" not in command
    assert recorded["enable_gpu_layers"] is False


def test_validate_server_skips_without_linux_sandbox(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = "skip",
            purpose = purpose,
            reason = "no sandbox",
        )

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Popen must not be used")),
    )

    validate_server(
        server_path,
        probe_path,
        linux_host(),
        tmp_path,
        runtime_line = None,
        install_kind = None,
    )


def test_validate_server_uses_windows_direct_validation_plan(monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server.exe"
    server_path.write_text("")
    probe_path = tmp_path / "probe.gguf"
    probe_path.write_text("")
    recorded: dict[str, object] = {}

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        recorded["plan"] = ValidationLaunchPlan(
            command = command,
            env = env,
            action = "run",
            purpose = purpose,
            sandbox_kind = "windows_direct_validation",
        )
        return recorded["plan"]

    class _FakeProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "binary_env", lambda *_a, **_k: {"PATH": str(tmp_path)}
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "_run_validation_popen", lambda plan, *, stdout: _FakeProcess()
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "free_local_port", lambda: 7777)

    class _FakeResponse:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT.urllib.request, "urlopen", lambda *a, **k: _FakeResponse()
    )

    validate_server(
        server_path,
        probe_path,
        windows_host(),
        tmp_path,
        runtime_line = None,
        install_kind = "windows-cuda",
    )
    plan = recorded["plan"]
    assert isinstance(plan, ValidationLaunchPlan)
    assert plan.sandbox_kind == "windows_direct_validation"


def test_runtime_inference_path_stays_outside_installer_sandbox_owner():
    backend_source = (
        Path(__file__).parents[3] / "studio/backend/core/inference/llama_cpp.py"
    ).read_text(encoding = "utf-8", errors = "replace")
    assert "_ValidationLaunchPlan" not in backend_source
    assert "_run_validation_capture" not in backend_source
    assert "_run_validation_popen" not in backend_source
    assert "build_validation_sandbox_plan" not in backend_source


def _run_validate_prebuilt_choice(
    monkeypatch,
    tmp_path,
    *,
    expected_sha256,
    validation_action = "run",
):
    """Run validate_prebuilt_choice with heavy steps stubbed; return launch metadata."""
    calls = {"quantize": 0, "server": 0}
    plans: list[str] = []
    server_path = tmp_path / "install" / "build" / "bin" / "llama-server"
    quantize_path = tmp_path / "install" / "build" / "bin" / "llama-quantize"
    quantized_path = tmp_path / "stories260K-q4.gguf"

    src = INSTALL_LLAMA_PREBUILT
    monkeypatch.setattr(
        src, "preferred_source_archive", lambda *a, **k: ("repo", "ref", None, False)
    )
    monkeypatch.setattr(src, "hydrate_source_tree", lambda *a, **k: None)
    monkeypatch.setattr(src, "install_from_archives", lambda *a, **k: (server_path, quantize_path))
    monkeypatch.setattr(src, "preflight_linux_installed_binaries", lambda *a, **k: None)
    monkeypatch.setattr(src, "preflight_macos_installed_binaries", lambda *a, **k: None)
    monkeypatch.setattr(src, "ensure_repo_shape", lambda *a, **k: None)
    monkeypatch.setattr(src, "write_prebuilt_metadata", lambda *a, **k: None)

    def fake_build_plan(
        command: list[str],
        *,
        binary_path: Path,
        install_dir: Path,
        purpose: str,
        env: dict[str, str],
        host = None,
        runtime_line = None,
        enable_gpu_layers: bool = False,
        gpu_backend = None,
    ) -> ValidationLaunchPlan:
        plans.append(purpose)
        return ValidationLaunchPlan(
            command = command,
            env = env,
            action = validation_action,
            purpose = purpose,
            reason = "validation launch unavailable" if validation_action != "run" else None,
        )

    def fake_run_validation_capture(plan: ValidationLaunchPlan, *, timeout: int):
        if plan.action != "run":
            raise src.ValidationLaunchUnavailable(plan.reason or "validation launch unavailable")
        if plan.purpose == src._VALIDATION_PURPOSE_QUANTIZE:
            calls["quantize"] += 1
            quantized_path.write_bytes(b"quantized")
        return subprocess.CompletedProcess(plan.command, 0, "", "")

    class _FakeResponse:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _DummyProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def fake_run_validation_popen(plan: ValidationLaunchPlan, *, stdout):
        if plan.action != "run":
            raise src.ValidationLaunchUnavailable(plan.reason or "validation launch unavailable")
        assert plan.purpose == src._VALIDATION_PURPOSE_SERVER
        calls["server"] += 1
        return _DummyProcess()

    monkeypatch.setattr(src, "build_validation_sandbox_plan", fake_build_plan)
    monkeypatch.setattr(src, "_run_validation_capture", fake_run_validation_capture)
    monkeypatch.setattr(src, "_run_validation_popen", fake_run_validation_popen)
    monkeypatch.setattr(src.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())

    bundle_name = "app-b9998-linux-x64-cuda13-newer.tar.gz"
    source_archive = tmp_path / "source.tar.gz"
    bundle_archive = tmp_path / "bundle.tar.gz"
    source_archive.write_bytes(b"source")
    bundle_archive.write_bytes(b"bundle")

    choice = AssetChoice(
        repo = "local",
        tag = "b9998",
        name = bundle_name,
        url = "file://bundle",
        source_label = "local",
        is_ready_bundle = True,
        install_kind = "linux-cuda",
        bundle_profile = "cuda13-newer",
        runtime_line = "cuda13",
        expected_sha256 = expected_sha256,
    )
    src.validate_prebuilt_choice(
        choice,
        _nvidia_linux_host(),
        tmp_path / "install",
        tmp_path / "work",
        tmp_path / "stories260K.gguf",
        requested_tag = "b9998",
        llama_tag = "b9998",
        release_tag = "b9998",
        approved_checksums = approved_checksums_for(
            "b9998",
            source_archive = source_archive,
            bundle_archive = bundle_archive,
            bundle_name = bundle_name,
        ),
        prebuilt_fallback_used = False,
        quantized_path = quantized_path,
    )
    return calls, plans


def test_validate_prebuilt_choice_approved_validation_skipped_when_flag_off(tmp_path, monkeypatch):
    # An approved (sha256-verified) bundle skips the smoke test while the flag is off.
    calls, plans = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert calls == {"quantize": 0, "server": 0}
    assert plans == []


def test_validate_prebuilt_choice_hashless_build_always_validated(tmp_path, monkeypatch):
    # A hashless build has no sha256 gate, so the smoke test must run even with the flag off.
    calls, plans = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = None)
    assert calls == {"quantize": 1, "server": 1}
    assert "ldd" in plans
    assert plans.count("quantize") == 1
    assert plans.count("server") == 1
    assert plans.index("quantize") < plans.index("server")


def test_validate_prebuilt_choice_hashless_build_routes_through_validation_sandbox(
    tmp_path, monkeypatch
):
    _calls, plans = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = None)
    assert plans.count("ldd") >= 1
    assert plans.count("quantize") == 1
    assert plans.count("server") == 1
    assert plans.index("quantize") < plans.index("server")
    assert plans[0] == "ldd"
    assert plans[-1] == "server"


def test_validate_prebuilt_choice_hashless_build_falls_back_when_validation_launch_skips(
    tmp_path, monkeypatch
):
    with pytest.raises(PrebuiltFallback, match = "llama-quantize validation unavailable"):
        _run_validate_prebuilt_choice(
            monkeypatch,
            tmp_path,
            expected_sha256 = None,
            validation_action = "skip",
        )


def test_validate_prebuilt_choice_approved_validation_runs_when_flag_enabled(tmp_path, monkeypatch):
    # _RUN_STAGED_PREBUILT_VALIDATION back on restores the smoke test for approved bundles too.
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", True)
    calls, plans = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert calls == {"quantize": 1, "server": 1}
    assert "ldd" in plans
    assert plans.count("quantize") == 1
    assert plans.count("server") == 1
    assert plans.index("quantize") < plans.index("server")


def test_validate_prebuilt_choice_approved_validation_records_sandbox_routing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", True)
    _calls, plans = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert plans.count("ldd") >= 1
    assert plans.count("quantize") == 1
    assert plans.count("server") == 1
    assert plans.index("quantize") < plans.index("server")
    assert plans[0] == "ldd"
    assert plans[-1] == "server"


def test_diffusion_visual_server_uses_approved_checksum_download(monkeypatch, tmp_path: Path):
    asset_name = "llama-diffusion-gemma-visual-server-linux-x64"
    expected_sha = "a" * 64
    asset_url = "https://github.com/unslothai/llama.cpp/releases/download/b9334/" + asset_name
    calls: list[tuple[str, Path, str | None, str | None]] = []

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "github_release_assets",
        lambda repo, tag: {asset_name: asset_url},
    )

    def fake_download_file(url, destination):
        raise AssertionError("diffusion visual server must not use unverified download_file")

    def fake_download_file_verified(url, destination, *, expected_sha256, label):
        calls.append((url, Path(destination), expected_sha256, label))
        Path(destination).write_bytes(b"verified visual server")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "download_file_verified", fake_download_file_verified
    )

    ensure_diffusion_visual_server(
        tmp_path / "install",
        linux_host(),
        "b9334",
        approved_release_checksums_for_asset(asset_name, expected_sha),
    )

    target = tmp_path / "install" / "build" / "bin" / "llama-diffusion-gemma-visual-server"
    assert calls == [
        (
            asset_url,
            target,
            expected_sha,
            f"diffusion visual server {asset_name}",
        )
    ]
    assert target.read_bytes() == b"verified visual server"
    if os.name == "nt":
        assert target.exists()
    else:
        assert target.stat().st_mode & 0o777 == 0o755


def test_diffusion_visual_server_refuses_unapproved_release_asset(monkeypatch, tmp_path: Path):
    asset_name = "llama-diffusion-gemma-visual-server-attacker-linux"
    verified_calls: list[str] = []
    raw_calls: list[str] = []

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "github_release_assets",
        lambda repo, tag: {asset_name: "https://example.test/" + asset_name},
    )

    def fake_download_file(url, destination):
        raw_calls.append(url)

    def fake_download_file_verified(url, destination, *, expected_sha256, label):
        verified_calls.append(url)

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "download_file", fake_download_file)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "download_file_verified", fake_download_file_verified
    )

    ensure_diffusion_visual_server(
        tmp_path / "install",
        linux_host(),
        "b9334",
        ApprovedReleaseChecksums(
            repo = "unslothai/llama.cpp",
            release_tag = "b9334",
            upstream_tag = "b9334",
            artifacts = {},
        ),
    )

    target = tmp_path / "install" / "build" / "bin" / "llama-diffusion-gemma-visual-server"
    assert not target.exists()
    assert raw_calls == []
    assert verified_calls == []
