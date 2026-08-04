import errno
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
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


# The extract_archive guard tests (safe symlink chain / hardlink, absolute or
# escaping or unresolved symlink targets, zip symlink entries) moved verbatim
# to tests/studio/install/test_prebuilt_core.py: extract_archive is the shared
# prebuilt_core implementation, re-exported by this installer.


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
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "collect_system_report", lambda *a, **k: "report")

    # Resolver failures are reclassified as a fallback, so the abort sentinel
    # surfaces as EXIT_FALLBACK with the original message on the chain.
    with pytest.raises(SystemExit) as caught:
        install_prebuilt(
            install_dir.resolve(),
            "latest",
            "unslothai/llama.cpp",
            "",
            instruction_cleanup_root = linked_root.absolute(),
        )

    assert caught.value.code == INSTALL_LLAMA_PREBUILT.EXIT_FALLBACK
    assert "stop after cleanup" in str(caught.value.__cause__)
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


def test_binary_env_windows_skips_inaccessible_inherited_path_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    denied = r"C:\WINDOWS\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps"
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "bin"
    runtime_dir = tmp_path / "runtime"
    inherited_dir = tmp_path / "usable-path"
    for directory in (bin_dir, runtime_dir, inherited_dir):
        directory.mkdir(parents = True)
    inaccessible = {denied}

    class FakePath:
        def __init__(self, raw):
            self.raw = str(raw)

        def expanduser(self):
            return self

        def is_dir(self):
            if self.raw in inaccessible:
                raise PermissionError(13, "Access is denied", self.raw, 5)
            return Path(self.raw).is_dir()

        def resolve(self):
            return Path(self.raw).resolve()

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

    # Exercise Windows PATH parsing even when this test runs on a POSIX host.
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "pathsep", ";")
    monkeypatch.setenv(
        "PATH",
        ";".join((denied, str(inherited_dir), str(runtime_dir), str(inherited_dir))),
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "Path", FakePath)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "windows_runtime_dirs_for_runtime_line",
        lambda _runtime_line: [str(runtime_dir)],
    )

    env = binary_env(bin_dir / "llama-server.exe", install_dir, host, runtime_line = "cuda")

    assert env["PATH"].split(";") == [
        str(bin_dir.resolve()),
        str(runtime_dir.resolve()),
        str(inherited_dir.resolve()),
    ]
    assert denied not in env["PATH"]

    inaccessible.add(str(runtime_dir))
    with pytest.raises(PermissionError, match = "Access is denied"):
        binary_env(bin_dir / "llama-server.exe", install_dir, host, runtime_line = "cuda")


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


def test_linux_runtime_dirs_probes_with_secret_free_env(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_missing(binary_path, *, env = None):
        captured["env"] = env
        return []

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_missing_libraries", fake_missing)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh_secret")

    INSTALL_LLAMA_PREBUILT.linux_runtime_dirs(Path("/fake/llama-server"))

    probe_env = captured["env"]
    assert probe_env is not None
    assert "HF_TOKEN" not in probe_env
    assert "GITHUB_TOKEN" not in probe_env


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
        force_cpu = False,
        llama_backend = None,
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
        force_cpu = False,
        llama_backend = None,
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
        force_cpu = False,
        llama_backend = None,
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
        force_cpu = False,
        llama_backend = None,
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


def test_existing_install_matches_choice_fails_when_install_tree_incomplete(tmp_path: Path):
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


def _run_validate_prebuilt_choice(monkeypatch, tmp_path, *, expected_sha256):
    """Run validate_prebuilt_choice with heavy steps stubbed; return the quantize/server smoke-test call counts."""
    calls = {"quantize": 0, "server": 0}
    server_path = tmp_path / "install" / "build" / "bin" / "llama-server"
    quantize_path = tmp_path / "install" / "build" / "bin" / "llama-quantize"

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
    monkeypatch.setattr(
        src,
        "validate_quantize",
        lambda *a, **k: calls.__setitem__("quantize", calls["quantize"] + 1),
    )
    monkeypatch.setattr(
        src, "validate_server", lambda *a, **k: calls.__setitem__("server", calls["server"] + 1)
    )

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
        quantized_path = tmp_path / "stories260K-q4.gguf",
    )
    return calls


def test_validate_prebuilt_choice_approved_validation_skipped_when_flag_off(tmp_path, monkeypatch):
    # An approved (sha256-verified) bundle skips the smoke test while the flag is off.
    calls = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert calls == {"quantize": 0, "server": 0}


def test_validate_prebuilt_choice_hashless_build_always_validated(tmp_path, monkeypatch):
    # A hashless build has no sha256 gate, so the smoke test must run even with the flag off.
    calls = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = None)
    assert calls == {"quantize": 1, "server": 1}


def test_validate_prebuilt_choice_approved_validation_runs_when_flag_enabled(tmp_path, monkeypatch):
    # _RUN_STAGED_PREBUILT_VALIDATION back on restores the smoke test for approved bundles too.
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", True)
    calls = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert calls == {"quantize": 1, "server": 1}


def test_staged_validation_enabled_default_off(monkeypatch):
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", False)
    monkeypatch.delenv("UNSLOTH_LLAMA_STAGED_VALIDATION", raising = False)
    assert INSTALL_LLAMA_PREBUILT.staged_validation_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_staged_validation_enabled_env_opt_in(monkeypatch, value):
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", False)
    monkeypatch.setenv("UNSLOTH_LLAMA_STAGED_VALIDATION", value)
    assert INSTALL_LLAMA_PREBUILT.staged_validation_enabled() is True


def test_validate_prebuilt_choice_approved_validation_runs_when_env_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_RUN_STAGED_PREBUILT_VALIDATION", False)
    monkeypatch.setenv("UNSLOTH_LLAMA_STAGED_VALIDATION", "1")
    calls = _run_validate_prebuilt_choice(monkeypatch, tmp_path, expected_sha256 = "ab" * 32)
    assert calls == {"quantize": 1, "server": 1}


def test_validate_existing_install_runs_server_smoke(tmp_path, monkeypatch):
    # setup.sh --validate-install path: exercise smoke helpers without a real GPU.
    install_dir = tmp_path / "llama.cpp"
    bin_dir = install_dir / "build" / "bin"
    bin_dir.mkdir(parents = True)
    (bin_dir / "llama-server").write_text("#!/bin/sh\n", encoding = "utf-8")
    (bin_dir / "llama-quantize").write_text("#!/bin/sh\n", encoding = "utf-8")
    calls: dict[str, int] = {"quantize": 0, "server": 0, "download": 0}

    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "download_validation_model",
        lambda path, cache = None: calls.__setitem__("download", calls["download"] + 1),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "validate_quantize",
        lambda *a, **k: calls.__setitem__("quantize", calls["quantize"] + 1),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "validate_server",
        lambda *a, **k: calls.__setitem__("server", calls["server"] + 1),
    )
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "detect_host",
        lambda: linux_host(),
    )

    INSTALL_LLAMA_PREBUILT.validate_existing_install(install_dir, install_kind = "linux-cuda")
    assert calls == {"quantize": 1, "server": 1, "download": 1}


def test_validate_existing_install_missing_server_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", lambda: linux_host())
    with pytest.raises(INSTALL_LLAMA_PREBUILT.PrebuiltFallback, match = "llama-server not found"):
        INSTALL_LLAMA_PREBUILT.validate_existing_install(tmp_path / "missing")


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


def test_dedupe_existing_dirs_skips_inaccessible_path_entry(monkeypatch):
    denied = r"C:\WINDOWS\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps"

    class FakePath:
        def __init__(self, raw):
            self.raw = str(raw)

        def expanduser(self):
            return self

        def is_dir(self):
            if self.raw == denied:
                raise PermissionError(13, "Access is denied", denied, 5)
            return True

        def resolve(self):
            return Path(self.raw).resolve()

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "Path", FakePath)

    with pytest.raises(PermissionError, match = "Access is denied"):
        INSTALL_LLAMA_PREBUILT.dedupe_existing_dirs([denied])

    assert INSTALL_LLAMA_PREBUILT.dedupe_existing_dirs(
        [denied, PACKAGE_ROOT], skip_unusable = True
    ) == [str(PACKAGE_ROOT.resolve())]


def test_windows_runtime_dirs_marks_path_candidates_as_optional(monkeypatch, tmp_path):
    denied = r"C:\WINDOWS\system32\config\systemprofile\AppData\Local\Microsoft\WindowsApps"
    observed = {}

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "pathsep", ";")

    def fake_dedupe(paths, *, skip_unusable = False):
        observed["paths"] = [str(path) for path in paths]
        observed["skip_unusable"] = skip_unusable
        return []

    monkeypatch.setenv("PATH", denied)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    for name in ("CUDA_RUNTIME_DLL_DIR", "CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "python_runtime_dirs", lambda: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "dedupe_existing_dirs", fake_dedupe)

    assert INSTALL_LLAMA_PREBUILT.windows_runtime_dirs() == []
    assert observed == {"paths": [denied], "skip_unusable": True}


_SETUP_SH_ROUTING_START = 'if [ "$_PREBUILT_STATUS" -eq 0 ]; then'
_SETUP_PS1_ROUTING_START = "if ($prebuiltExit -eq 0) {"

# Stand-ins for the setup.sh helpers the routing block calls; each records what
# it was asked to do so the assertions can read the decision back out.
_SETUP_SH_HARNESS = """
set -u
C_OK=""; C_WARN=""; C_ERR=""
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_NO_SPACE=false
_LLAMA_CPP_DEGRADED=false
_explicit_vulkan_backend=false
_STUDIO_HOME_IS_CUSTOM=false
_STUDIO_OWNED_MARKER=".unsloth-owned"
step() { echo "step: $2"; }
substep() { echo "substep: $1"; }
verbose_substep() { :; }
print_llama_error_log() { :; }
print_installed_llama_prebuilt_release() { :; }
_has_local_llama_server() { return 1; }
setup_fail() { echo "setup_fail: $1"; exit "$1"; }
"""

_SETUP_SH_HARNESS_TAIL = """
echo "source_build=$_NEED_LLAMA_SOURCE_BUILD"
echo "no_space=$_LLAMA_CPP_NO_SPACE"
"""


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start : end + len(end_marker)]


def _setup_sh_routing_block() -> str:
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    # The routing chain ends at the first dedented "    fi" after it.
    start = setup_sh.index(_SETUP_SH_ROUTING_START)
    end = setup_sh.index("\n    fi\n", start)
    return setup_sh[start : end + len("\n    fi\n")]


def _run_setup_sh_routing(status: int, tmp_path: Path, *, install_exists: bool) -> dict[str, str]:
    """Drive the real setup.sh exit-code chain with a stubbed helper result."""
    llama_dir = tmp_path / "llama.cpp"
    if install_exists:
        llama_dir.mkdir(parents = True, exist_ok = True)
    log_path = tmp_path / "prebuilt.log"
    log_path.write_text("boom\n", encoding = "utf-8")

    script = "\n".join(
        [
            _SETUP_SH_HARNESS,
            f'LLAMA_CPP_DIR="{llama_dir}"',
            f'_PREBUILT_LOG="{log_path}"',
            f"_PREBUILT_STATUS={status}",
            _setup_sh_routing_block(),
            _SETUP_SH_HARNESS_TAIL,
        ]
    )
    completed = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@pytest.mark.skipif(
    os.name == "nt",
    reason = "setup.sh is the POSIX installer; Windows runs setup.ps1, and driving a "
    "POSIX script through Git Bash with Windows paths proves nothing about either",
)
@pytest.mark.parametrize(
    "status, expect_source_build, expect_exit",
    [
        (0, False, 0),  # installed and validated
        (1, False, 1),  # helper error -> fail, never compile
        (2, True, 0),  # the one status that means "prebuilt unusable, go build"
        (3, False, 3),  # busy: a source build cannot replace locked binaries
        (4, False, 0),  # out of disk: compiling needs more, not less
        (137, False, 1),  # SIGKILL/OOM and anything else -> fail, never compile
    ],
)
def test_setup_sh_starts_source_build_only_for_expected_prebuilt_exit(
    tmp_path, status, expect_source_build, expect_exit
):
    """Behavioural cover for the exit-code routing: runs the real block under bash.

    The previous version of this test compared ``str.index`` offsets, which is a
    tautology -- ``index(needle, start)`` never returns less than ``start`` -- so
    it asserted only that three literals existed in textual order.

    The PowerShell side of the same routing is covered textually by
    test_setup_scripts_unexpected_exit_branch_never_sets_source_build, which is
    platform independent.
    """
    if shutil.which("bash") is None:  # pragma: no cover - CI always has bash
        pytest.skip("bash is required to exercise the setup.sh routing block")

    result = _run_setup_sh_routing(status, tmp_path, install_exists = True)

    assert result["returncode"] == expect_exit, result
    if expect_source_build:
        assert "source_build=true" in result["stdout"], result
    else:
        assert "source_build=true" not in result["stdout"], result


def test_setup_scripts_unexpected_exit_branch_never_sets_source_build():
    """The new catch-all must fail loudly, not queue a compile, on both platforms."""
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

    sh_block = _setup_sh_routing_block()
    sh_else = sh_block[sh_block.rindex("\n    else\n") :]
    assert "_NEED_LLAMA_SOURCE_BUILD=true" not in sh_else
    assert "setup_fail 1" in sh_else
    assert "prebuilt helper failed unexpectedly (exit code $_PREBUILT_STATUS)" in sh_else
    # Only status 2 may queue the source build.
    assert sh_block.count("_NEED_LLAMA_SOURCE_BUILD=true") == 1
    assert 'elif [ "$_PREBUILT_STATUS" -eq 2 ]; then' in sh_block

    ps_block = _extract_block(setup_ps1, _SETUP_PS1_ROUTING_START, 'retry setup."\n        }')
    ps_else = ps_block[ps_block.rindex("} else {") :]
    assert "$NeedLlamaSourceBuild = $true" not in ps_else
    assert "Exit-SetupFailure" in ps_else
    assert "prebuilt helper failed unexpectedly (exit code $prebuiltExit)" in ps_else
    assert ps_block.count("$NeedLlamaSourceBuild = $true") == 1
    assert "} elseif ($prebuiltExit -eq 2) {" in ps_block

    # Statuses 3 and 4 keep their dedicated branches ahead of the catch-all.
    for needle in ('elif [ "$_PREBUILT_STATUS" -eq 3 ]', 'elif [ "$_PREBUILT_STATUS" -eq 4 ]'):
        assert needle in setup_sh
    for needle in ("elseif ($prebuiltExit -eq 3)", "elseif ($prebuiltExit -eq 4)"):
        assert needle in setup_ps1


# ── release-listing failures must stay source-build recoverable (exit 2) ──


@pytest.mark.parametrize(
    "error",
    [
        # Rate limiting: fetch_json raises a bare RuntimeError, so no
        # urllib/OSError handler catches it.
        RuntimeError(
            "GitHub API returned 403 for "
            "https://api.github.com/repos/unslothai/llama.cpp/releases?per_page=100&page=1"
            "; set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits"
        ),
        RuntimeError("unexpected releases payload for unslothai/llama.cpp"),
        urllib.error.URLError("connection reset"),
        TimeoutError("timed out"),
    ],
    ids = ["rate-limit", "bad-payload", "urlerror", "timeout"],
)
def test_release_listing_failure_exits_fallback_not_error(tmp_path, monkeypatch, error):
    """A network problem while listing releases must ask for a source build.

    The setup scripts only source build on EXIT_FALLBACK, so anything that
    escapes as EXIT_ERROR here hard-fails the whole install for what is a
    transient condition -- a source build clones over git, not api.github.com,
    and succeeds while the API is rate limited.
    """

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", linux_host)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "collect_system_report", lambda *a, **k: "report")

    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()

    with pytest.raises(SystemExit) as caught:
        install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert caught.value.code == INSTALL_LLAMA_PREBUILT.EXIT_FALLBACK


@pytest.mark.parametrize(
    "error",
    [
        TypeError("bug"),
        AttributeError("bug"),
        NameError("bug"),
        # Host-resource failures, not transport: a source build needs more file
        # descriptors and memory, so it cannot repair either.
        OSError(errno.EMFILE, "Too many open files"),
        OSError(errno.ENOMEM, "Cannot allocate memory"),
        PermissionError(errno.EACCES, "Permission denied"),
    ],
    ids = ["typeerror", "attributeerror", "nameerror", "emfile", "enomem", "eacces"],
)
def test_release_listing_code_defect_stays_exit_error(tmp_path, monkeypatch, error):
    """A defect in the resolver is an installer bug, not a transient condition.

    It must not buy a multi-minute source build under a message that blames the
    network. Only OSError/RuntimeError/ValueError -- the shapes a transport or
    payload failure actually takes -- are reclassified as a fallback.
    """

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", linux_host)

    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()

    # Escapes install_prebuilt, so __main__ maps it to EXIT_ERROR.
    with pytest.raises(type(error)):
        install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")


def test_release_listing_enospc_still_exits_no_space(tmp_path, monkeypatch):
    """A full disk must reach EXIT_NO_SPACE, never a source build.

    ENOSPC is a plain OSError, which the transport catch deliberately does not
    claim, so it escapes install_prebuilt and __main__ classifies it. Assert the
    same way __main__ does, so this stays a real end-to-end guarantee.
    """

    def boom(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", linux_host)

    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()

    with pytest.raises(OSError) as caught:
        install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert caught.value.errno == errno.ENOSPC
    # __main__ turns exactly this into EXIT_NO_SPACE via _fail_no_space.
    assert INSTALL_LLAMA_PREBUILT._environment_fatal_reason(caught.value)


def test_fallback_survives_a_failing_system_report(tmp_path, monkeypatch):
    """Diagnostics collected on the way out must not flip EXIT_FALLBACK to EXIT_ERROR."""
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "_fork_manifest_release_plans",
        lambda *a, **k: (_ for _ in ()).throw(PrebuiltFallback("no compatible prebuilt asset")),
    )
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "detect_host", linux_host)
    monkeypatch.setattr(
        INSTALL_LLAMA_PREBUILT,
        "collect_system_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nvidia-smi hung")),
    )

    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()

    with pytest.raises(SystemExit) as caught:
        install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert caught.value.code == INSTALL_LLAMA_PREBUILT.EXIT_FALLBACK


def test_marker_sync_strands_no_temp_file_when_the_first_write_fails(tmp_path, monkeypatch):
    """ENOSPC during write/flush/fsync must not leave a partial .tmp- behind.

    A full volume would otherwise accumulate one stranded temp file per setup
    attempt, right next to the marker they are named after.
    """
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    original = json.dumps({"force_cpu": False}) + "\n"
    marker.write_text(original, encoding = "utf-8")

    real_write = INSTALL_LLAMA_PREBUILT.tempfile.NamedTemporaryFile

    class _FullDisk:
        def __init__(self, handle):
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def write(self, *args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

    class _Ctx:
        def __enter__(self):
            self._cm = real_write(prefix = marker.name + ".tmp-", dir = install_dir, delete = False)
            return _FullDisk(self._cm.__enter__())

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.tempfile, "NamedTemporaryFile", lambda **kw: _Ctx())

    INSTALL_LLAMA_PREBUILT.sync_marker_force_cpu(install_dir, True)

    assert marker.read_text(encoding = "utf-8") == original
    assert [q.name for q in install_dir.iterdir() if ".tmp-" in q.name] == []


TREE_A = "8f3c6e197debb027f500df9f76e710e137f9fe68"


def test_reused_install_backfills_the_ggml_tree(tmp_path):
    """An install made before ggml_tree existed must gain it on reuse.

    write_prebuilt_metadata only runs on a real install and the fingerprint does
    not hash ggml_tree, so without this the marker stays tree-less forever and
    slim whisper pairing silently falls back to the "-mix-" suffix.
    """
    install_dir = tmp_path / "llama.cpp"
    (install_dir / "build" / "bin").mkdir(parents = True)
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(json.dumps({"release_tag": "b10173-mix-2c8b9c1"}) + "\n", encoding = "utf-8")

    INSTALL_LLAMA_PREBUILT.sync_marker_ggml_tree(install_dir, TREE_A)

    payload = json.loads(marker.read_text(encoding = "utf-8"))
    assert payload["ggml_tree"] == TREE_A
    assert payload["release_tag"] == "b10173-mix-2c8b9c1"  # nothing else lost
    assert INSTALL_LLAMA_PREBUILT.installed_llama_ggml_tree(install_dir) == TREE_A


@pytest.mark.parametrize("declared", [None, ""])
def test_reused_install_keeps_the_ggml_tree_when_the_release_declares_none(tmp_path, declared):
    """A release with no ggml_tree (upstream ggml-org tags) must not erase one."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(
        json.dumps({"release_tag": "b10173-mix-2c8b9c1", "ggml_tree": TREE_A}) + "\n",
        encoding = "utf-8",
    )

    INSTALL_LLAMA_PREBUILT.sync_marker_ggml_tree(install_dir, declared)

    assert json.loads(marker.read_text(encoding = "utf-8"))["ggml_tree"] == TREE_A


@pytest.mark.skipif(os.name == "nt", reason = "Windows st_mode carries no POSIX mode bits")
@pytest.mark.parametrize("mode", [0o444, 0o644, 0o664])
def test_marker_sync_preserves_the_marker_mode(tmp_path, mode):
    """A shared install's marker must stay readable by everyone who could read it.

    os.replace keeps the SOURCE file's mode and NamedTemporaryFile is 0600, so a
    naive atomic refresh would leave UNSLOTH_PREBUILT_INFO.json readable only by
    whoever ran setup -- and other users could no longer recognise or update the
    shared installation.
    """
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(json.dumps({"force_cpu": False}) + "\n", encoding = "utf-8")
    os.chmod(marker, mode)

    INSTALL_LLAMA_PREBUILT.sync_marker_force_cpu(install_dir, True)

    assert stat.S_IMODE(marker.stat().st_mode) == mode
    assert json.loads(marker.read_text(encoding = "utf-8"))["force_cpu"] is True
    # and no temp file is stranded next to it
    assert [p.name for p in install_dir.iterdir() if ".tmp-" in p.name] == []


def test_marker_sync_leaves_a_valid_marker_intact_when_the_write_fails(tmp_path, monkeypatch):
    """A failed refresh must never truncate the marker.

    An in-place retry opens the valid marker with truncation, so an ENOSPC or
    I/O error mid-write would strand a partial UNSLOTH_PREBUILT_INFO.json and
    later updates would stop recognising the install.
    """
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    original = json.dumps({"force_cpu": False, "release_tag": "b9002"}, indent = 2) + "\n"
    marker.write_text(original, encoding = "utf-8")

    def out_of_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "atomic_replace_from_tempfile", out_of_space)

    INSTALL_LLAMA_PREBUILT.sync_marker_force_cpu(install_dir, True)

    assert marker.read_text(encoding = "utf-8") == original
    assert [q.name for q in install_dir.iterdir() if ".tmp-" in q.name] == []


def test_python_runtime_dirs_skips_an_inaccessible_glob_result(monkeypatch, tmp_path):
    """A readable site-packages root with a denied child must not abort discovery.

    That is the shape of the bug this PR is about: the parent lists fine and the
    entry underneath is denied. Guarding only the root would leave the strict
    dedupe on the return to raise anyway.
    """
    root = tmp_path / "site-packages"
    good = root / "torch" / "lib"
    good.mkdir(parents = True)
    denied = root / "nvidia" / "cublas" / "lib"
    denied.mkdir(parents = True)

    real_is_dir = Path.is_dir

    def guarded_is_dir(self):
        if str(self) == str(denied):
            raise PermissionError(13, "Access is denied", str(self), 5)
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.sys, "path", [str(root)])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.site, "getusersitepackages", lambda: "")

    assert INSTALL_LLAMA_PREBUILT.python_runtime_dirs() == [str(good.resolve())]


# ── inaccessible discovery roots outside dedupe_existing_dirs ──


def test_python_runtime_dirs_skips_inaccessible_sys_path_entry(monkeypatch, tmp_path):
    """A denied sys.path entry must not escape windows_runtime_dirs()."""
    usable = tmp_path / "site-packages"
    (usable / "torch" / "lib").mkdir(parents = True)

    denied = Path(r"\\\\denied-share\\site-packages")

    real_is_dir = Path.is_dir

    def guarded_is_dir(self):
        if str(self) == str(denied):
            raise PermissionError(13, "Access is denied", str(self), 5)
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.sys, "path", [str(denied), str(usable)])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.site, "getusersitepackages", lambda: "")

    assert INSTALL_LLAMA_PREBUILT.python_runtime_dirs() == [
        str((usable / "torch" / "lib").resolve())
    ]


def test_windows_runtime_dirs_skips_inaccessible_program_files(monkeypatch):
    """%ProgramFiles% is user-controllable, so its stat must not be fatal either."""
    denied = r"C:\Denied Program Files"

    real_is_dir = Path.is_dir

    def guarded_is_dir(self):
        if str(self).startswith(denied):
            raise PermissionError(13, "Access is denied", str(self), 5)
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT.os, "pathsep", ";")
    monkeypatch.setenv("ProgramFiles", denied)
    monkeypatch.delenv("CUDA_RUNTIME_DLL_DIR", raising = False)
    monkeypatch.delenv("CUDA_PATH", raising = False)
    monkeypatch.delenv("CUDA_HOME", raising = False)
    monkeypatch.delenv("CUDA_ROOT", raising = False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "python_runtime_dirs", lambda: [])

    assert INSTALL_LLAMA_PREBUILT.windows_runtime_dirs() == []


def test_binary_env_linux_skips_inaccessible_inherited_ld_library_path(monkeypatch, tmp_path):
    """Same class of bug as the Windows PATH fix, on the LD_LIBRARY_PATH side."""
    binary_dir = tmp_path / "llama.cpp"
    binary_dir.mkdir()
    binary_path = binary_dir / "llama-server"
    binary_path.write_bytes(b"")
    usable = tmp_path / "usable-lib"
    usable.mkdir()
    denied = tmp_path / "denied-lib"
    denied.mkdir()

    real_is_dir = Path.is_dir

    def guarded_is_dir(self):
        if str(self) == str(denied):
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "linux_runtime_dirs", lambda *a, **k: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_wsl_system_rocm_lib_dirs", lambda: [])
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "_native_linux_system_rocm_lib_dirs", lambda *a: [])
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join([str(denied), str(usable)]))

    env = binary_env(binary_path, binary_dir, linux_host())

    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(binary_dir.resolve()),
        str(usable.resolve()),
    ]


# ── marker sync is advisory, never fatal ──


@pytest.mark.parametrize(
    "sync, kwargs",
    [
        ("sync_marker_force_cpu", {"persist_force_cpu": True}),
        ("sync_marker_llama_backend", {"llama_backend": "vulkan"}),
    ],
)
def test_marker_sync_survives_a_read_only_marker(tmp_path, sync, kwargs):
    """A shared or admin-owned install must not fail setup on a marker rewrite.

    Re-recording force_cpu / llama_backend runs on the existing-install reuse
    path. The read is guarded but the write was not, so a read-only marker
    raised PermissionError out of the helper as EXIT_ERROR -- which no longer
    falls back to a source build, so it would abort the whole install.
    """
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(
        json.dumps({"force_cpu": False, "llama_backend": None}) + "\n", encoding = "utf-8"
    )
    os.chmod(marker, 0o444)
    if os.access(marker, os.W_OK):  # pragma: no cover - root ignores the mode
        pytest.skip("cannot make the marker read-only for this user")

    logged: list[str] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "log", logged.append)
    try:
        # Must not raise: that would surface as EXIT_ERROR and abort setup.
        getattr(INSTALL_LLAMA_PREBUILT, sync)(install_dir, *kwargs.values())
    finally:
        monkeypatch.undo()
        if marker.exists():
            os.chmod(marker, 0o644)

    # Either the atomic swap landed the value (POSIX, writable dir) or we warned.
    # Silently losing force_cpu would let a later update re-route a deliberate CPU
    # user onto a GPU bundle (#7213). Windows refuses os.replace onto a read-only
    # destination, hence the two-way assert.
    field = list(kwargs)[0].replace("persist_", "")
    expected = list(kwargs.values())[0]
    persisted = json.loads(marker.read_text(encoding = "utf-8")).get(field) == expected
    assert persisted or any("WARNING" in line and field in line for line in logged), logged


def test_marker_sync_never_fails_setup_when_the_write_cannot_land(tmp_path, monkeypatch):
    """If even the atomic swap is refused, warn and continue -- never abort setup."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    marker.write_text(json.dumps({"force_cpu": False}) + "\n", encoding = "utf-8")

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Access is denied", str(marker), 5)

    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "atomic_replace_from_tempfile", refuse)

    logged: list[str] = []
    monkeypatch.setattr(INSTALL_LLAMA_PREBUILT, "log", logged.append)

    INSTALL_LLAMA_PREBUILT.sync_marker_force_cpu(install_dir, True)

    assert any("WARNING" in line and "force_cpu" in line for line in logged), logged
