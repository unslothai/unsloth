import importlib.util
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location(
    "studio_install_llama_prebuilt", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

PrebuiltFallback = INSTALL_LLAMA_PREBUILT.PrebuiltFallback
extract_archive = INSTALL_LLAMA_PREBUILT.extract_archive
binary_env = INSTALL_LLAMA_PREBUILT.binary_env
HostInfo = INSTALL_LLAMA_PREBUILT.HostInfo
AssetChoice = INSTALL_LLAMA_PREBUILT.AssetChoice
ApprovedArtifactHash = INSTALL_LLAMA_PREBUILT.ApprovedArtifactHash
ApprovedReleaseChecksums = INSTALL_LLAMA_PREBUILT.ApprovedReleaseChecksums
hydrate_source_tree = INSTALL_LLAMA_PREBUILT.hydrate_source_tree
validate_prebuilt_choice = INSTALL_LLAMA_PREBUILT.validate_prebuilt_choice
activate_install_tree = INSTALL_LLAMA_PREBUILT.activate_install_tree
create_install_staging_dir = INSTALL_LLAMA_PREBUILT.create_install_staging_dir
sha256_file = INSTALL_LLAMA_PREBUILT.sha256_file
source_archive_logical_name = INSTALL_LLAMA_PREBUILT.source_archive_logical_name
install_prebuilt = INSTALL_LLAMA_PREBUILT.install_prebuilt
write_prebuilt_metadata = INSTALL_LLAMA_PREBUILT.write_prebuilt_metadata
existing_install_matches_plan = INSTALL_LLAMA_PREBUILT.existing_install_matches_plan
existing_install_matches_choice = INSTALL_LLAMA_PREBUILT.existing_install_matches_choice


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

    with pytest.raises(PrebuiltFallback, match = "archive link used an absolute target"):
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
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "validate_quantize", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "validate_server", lambda *args, **kwargs: None
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
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "validate_quantize", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "validate_server", lambda *args, **kwargs: None
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("activation confirm failed")
        ),
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


def test_activate_install_tree_cleans_all_paths_when_rollback_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("activation confirm failed")
        ),
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
    assert (
        "cleaning staging, install, and rollback paths before source build fallback"
        in output
    )
    assert "removing failed install path" in output
    assert "removing rollback path" in output


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
        "resolve_install_release_plans",
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
    (runtime_dir / "libllama.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml-base.so.0").write_bytes(b"DLL")
    (runtime_dir / "libggml-cpu-x64.so.0").write_bytes(b"DLL")
    (runtime_dir / "libmtmd.so.0").write_bytes(b"DLL")
    (install_dir / "convert_hf_to_gguf.py").write_text(
        "#!/usr/bin/env python3\n", encoding = "utf-8"
    )
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True, exist_ok = True)


def write_windows_install_shape(
    install_dir: Path, *, include_llama_dll: bool = True, include_cuda_dll: bool = False
) -> None:
    runtime_dir = install_dir / "build" / "bin" / "Release"
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (runtime_dir / "llama-server.exe").write_bytes(b"MZ")
    (runtime_dir / "llama-quantize.exe").write_bytes(b"MZ")
    if include_llama_dll:
        (runtime_dir / "llama.dll").write_bytes(b"DLL")
    if include_cuda_dll:
        (runtime_dir / "ggml-cuda.dll").write_bytes(b"DLL")
    (install_dir / "convert_hf_to_gguf.py").write_text(
        "#!/usr/bin/env python3\n", encoding = "utf-8"
    )
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
    (install_dir / "convert_hf_to_gguf.py").write_text(
        "#!/usr/bin/env python3\n", encoding = "utf-8"
    )
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True, exist_ok = True)


def test_existing_install_matches_plan_with_fingerprint_linux(tmp_path: Path):
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


def test_existing_install_matches_plan_false_without_fingerprint(tmp_path: Path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    write_linux_install_shape(install_dir)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"tag": "b9001", "asset": "llama-b9001-bin-ubuntu-x64.tar.gz"})
        + "\n",
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
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        "{not-json\n", encoding = "utf-8"
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
    write_windows_install_shape(
        install_dir, include_llama_dll = True, include_cuda_dll = True
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
    (install_dir / "build" / "bin" / "Release" / "ggml-cuda.dll").unlink()
    assert existing_install_matches_plan(install_dir, host, plan) is False


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
        "resolve_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: (
            "latest",
            [plan],
        ),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "matching install should skip before validation model download"
            )
        ),
    )

    install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")


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
        "resolve_install_release_plans",
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
        "resolve_install_release_plans",
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
        "resolve_install_release_plans",
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
        "resolve_install_release_plans",
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

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT, "validate_prebuilt_attempts", fake_validate
    )

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

    assert attempted == [
        ("b9002", "release-2", "upstream"),
        ("b9001", "release-1", "upstream"),
    ]
    assert activated["install_dir"] == install_dir


def io_bytes(data: bytes):
    return io.BytesIO(data)


def add_bytes_to_tar(
    archive: tarfile.TarFile, name: str, data: bytes, *, mode: int = 0o644
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
    tmp_path: Path,
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

    # Remove convert_hf_to_gguf.py (checked by confirm_install_tree but not
    # runtime_payload_is_healthy) and verify the guard catches it
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


def test_existing_install_matches_choice_fails_when_install_tree_incomplete_macos(
    tmp_path: Path,
):
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
