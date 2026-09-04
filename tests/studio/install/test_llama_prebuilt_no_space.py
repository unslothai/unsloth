# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Out-of-disk handling in the llama.cpp prebuilt installer: ENOSPC classification
through exception chains, EXIT_NO_SPACE, and the advisory low-disk warning. Offline."""

from __future__ import annotations

import errno
import importlib.util
import shutil
import sys
import urllib.error
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALL_LLAMA_PREBUILT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSTALL_LLAMA_PREBUILT
SPEC.loader.exec_module(INSTALL_LLAMA_PREBUILT)

M = INSTALL_LLAMA_PREBUILT
PrebuiltFallback = M.PrebuiltFallback
AssetChoice = M.AssetChoice
ApprovedReleaseChecksums = M.ApprovedReleaseChecksums

GB = 1024**3


def linux_host() -> "M.HostInfo":
    return M.HostInfo(
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


def choice(name: str, tag: str = "release-2") -> "M.AssetChoice":
    return AssetChoice(
        repo = "unslothai/llama.cpp",
        tag = tag,
        name = name,
        url = f"https://example.com/{name}",
        source_label = "published",
        install_kind = "linux-cpu",
    )


def checksums(release_tag: str, llama_tag: str) -> "M.ApprovedReleaseChecksums":
    return ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = release_tag,
        upstream_tag = llama_tag,
        source_commit = None,
        artifacts = {},
    )


def plan(llama_tag: str, release_tag: str, attempts) -> "M.InstallReleasePlan":
    return M.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = llama_tag,
        release_tag = release_tag,
        attempts = attempts,
        approved_checksums = checksums(release_tag, llama_tag),
    )


def fake_disk_usage(free_bytes: int):
    def _usage(path):
        return shutil._ntuple_diskusage(100 * GB, 100 * GB - free_bytes, free_bytes)

    return _usage


def install_harness(monkeypatch: pytest.MonkeyPatch, plans, *, free_bytes: int) -> list[str]:
    """Wire install_prebuilt down to a fake per-candidate validation. Returns the
    list of candidate names the run actually reached."""
    monkeypatch.setattr(M, "detect_host", lambda: linux_host())
    monkeypatch.setattr(
        M,
        "resolve_simple_install_release_plans",
        lambda llama_tag, host, published_repo, published_release_tag: ("latest", plans),
    )
    monkeypatch.setattr(
        M, "download_validation_model", lambda probe_path, cache_path: probe_path.write_bytes(b"p")
    )
    monkeypatch.setattr(M.shutil, "disk_usage", fake_disk_usage(free_bytes))
    monkeypatch.setattr(M, "existing_install_matches_plan", lambda *args, **kwargs: False)
    monkeypatch.setattr(M, "existing_install_matches_choice", lambda *args, **kwargs: False)
    monkeypatch.setattr(M, "activate_install_tree", lambda *args, **kwargs: None)
    monkeypatch.setattr(M, "ensure_converter_scripts", lambda *args, **kwargs: None)
    monkeypatch.setattr(M, "ensure_diffusion_visual_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(M, "collect_system_report", lambda *args, **kwargs: "report")
    reached: list[str] = []
    monkeypatch.setattr(
        M, "validate_prebuilt_choice", lambda attempt, *a, **k: reached.append(attempt.name)
    )
    return reached


# ── the low-disk check is advisory, never fatal ──


def test_low_disk_warning_reports_the_starved_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(M.shutil, "disk_usage", fake_disk_usage(1 * GB))
    reason = M._low_disk_warning(tmp_path / "llama.cpp")
    assert reason is not None and "low disk space for llama.cpp" in reason


def test_low_disk_warning_silent_when_roomy(tmp_path, monkeypatch):
    monkeypatch.setattr(M.shutil, "disk_usage", fake_disk_usage(50 * GB))
    assert M._low_disk_warning(tmp_path / "llama.cpp") is None


def test_low_disk_warning_ignores_unstatable_paths(tmp_path, monkeypatch):
    def _boom(path):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(M.shutil, "disk_usage", _boom)
    assert M._low_disk_warning(tmp_path / "llama.cpp") is None


def test_low_disk_does_not_block_an_install_that_fits(tmp_path, monkeypatch, capsys):
    """A 15 MB CPU bundle installs fine on a host with 3 GB free; the fixed
    threshold must warn rather than reject it (and skip the source fallback)."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    only = plan("b10079", "release-2", [choice("app-b10079-linux-x64-cpu.tar.gz")])
    reached = install_harness(monkeypatch, [only], free_bytes = 3 * GB)

    M.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert reached == ["app-b10079-linux-x64-cpu.tar.gz"]
    captured = capsys.readouterr()
    assert "low disk space for llama.cpp" in captured.out + captured.err


# ── ENOSPC classification ──


def test_classifies_direct_and_chained_enospc():
    assert M._environment_fatal_reason(OSError(errno.ENOSPC, "No space left on device"))
    for wrap in ("cause", "context"):
        try:
            try:
                raise OSError(errno.ENOSPC, "No space left on device")
            except OSError as inner:
                if wrap == "cause":
                    raise PrebuiltFallback("download failed") from inner
                raise PrebuiltFallback("download failed")
        except PrebuiltFallback as outer:
            assert M._environment_fatal_reason(outer), wrap


def test_ignores_unrelated_errors_and_cycles():
    assert M._environment_fatal_reason(OSError(errno.EACCES, "denied")) is None
    first, second = PrebuiltFallback("a"), PrebuiltFallback("b")
    first.__cause__, second.__cause__ = second, first
    assert M._environment_fatal_reason(first) is None


def test_suppressed_context_is_not_treated_as_disk_full():
    """`raise ... from None` means the earlier ENOSPC is unrelated."""
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError:
            raise PrebuiltFallback("checksum mismatch") from None
    except PrebuiltFallback as outer:
        assert M._environment_fatal_reason(outer) is None


def test_windows_disk_full_winerrors_are_classified():
    """CPython maps ERROR_DISK_FULL (112) to ENOSPC but has no case for
    ERROR_HANDLE_DISK_FULL (39), which arrives as EINVAL."""
    for winerror, code in ((112, errno.ENOSPC), (39, errno.EINVAL)):
        exc = OSError(code, "The disk is full")
        exc.winerror = winerror
        assert M._environment_fatal_reason(exc), winerror

    other = OSError(errno.EACCES, "sharing violation")
    other.winerror = 32
    assert M._environment_fatal_reason(other) is None


def test_http_errors_in_the_chain_do_not_crash_the_classifier():
    """HTTPError is an OSError that proxies unknown attributes to a wrapped file
    and raises KeyError, not AttributeError, on 3.9."""
    err = urllib.error.HTTPError("https://example.com/a", 404, "Not Found", {}, None)
    assert M._environment_fatal_reason(err) is None
    try:
        try:
            raise err
        except urllib.error.HTTPError as inner:
            raise PrebuiltFallback("mirror failed") from inner
    except PrebuiltFallback as outer:
        assert M._environment_fatal_reason(outer) is None


@pytest.mark.skipif(not hasattr(errno, "EDQUOT"), reason = "EDQUOT is POSIX only")
def test_quota_exhaustion_counts_as_out_of_space():
    """A quota'd home has free blocks this user cannot have, so the larger source
    build is just as doomed. Reported as a quota so df does not mislead."""
    assert M._environment_fatal_reason(OSError(errno.EDQUOT, "Disk quota exceeded")) == (
        "disk quota exceeded"
    )
    try:
        try:
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        except OSError as inner:
            raise PrebuiltFallback("bundle download failed") from inner
    except PrebuiltFallback as outer:
        assert M._environment_fatal_reason(outer) == "disk quota exceeded"


def test_a_bare_oserror_never_matches():
    """errno is None on a bare OSError, so it must not collide with a code."""
    assert M._environment_fatal_reason(OSError()) is None
    assert M._environment_fatal_reason(shutil.Error("copy failed")) is None


def test_flattened_markers_are_not_matched_as_prefixes():
    """Bare "WinError 112" would also match WinError 1120; the brackets pin it."""
    assert (
        M._environment_fatal_reason(
            shutil.Error("[('a', 'b', '[WinError 1120] a serial write completed')]")
        )
        is None
    )
    assert (
        M._environment_fatal_reason(
            shutil.Error(f"[('a', 'b', '[Errno {errno.ENOSPC}0] not a real code')]")
        )
        is None
    )


def test_windows_flattened_disk_full_text_is_classified():
    """copytree stringifies the per-file OSError, and on Windows str(OSError)
    prints [WinError 112] and never [Errno 28] (confirmed on a real NTFS volume)."""
    flattened = (
        "[('D:\\\\a\\\\src\\\\big.bin', 'T:\\\\dst\\\\big.bin', "
        "'[WinError 112] There is not enough space on the disk')]"
    )
    assert M._environment_fatal_reason(shutil.Error(flattened))
    assert M._environment_fatal_reason(
        shutil.Error("[('a', 'b', '[WinError 39] The disk is full')]")
    )
    assert (
        M._environment_fatal_reason(shutil.Error("[('a', 'b', '[WinError 32] sharing violation')]"))
        is None
    )


def test_validate_install_mode_exits_no_space(tmp_path, monkeypatch):
    """setup.sh reacts to a failed staged validation by deleting the finished GPU
    build and starting a CPU rebuild, which needs more of the space that ran out."""

    def boom(*args, **kwargs):
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise PrebuiltFallback("validation model unavailable") from inner

    monkeypatch.setattr(M, "validate_existing_install", boom)
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--validate-install", str(tmp_path)]
    )

    with pytest.raises(SystemExit) as caught:
        M.main()
    assert caught.value.code == M.EXIT_NO_SPACE


def test_validate_install_mode_still_falls_back_on_ordinary_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        M,
        "validate_existing_install",
        lambda *a, **k: (_ for _ in ()).throw(PrebuiltFallback("llama-server crashed")),
    )
    monkeypatch.setattr(
        sys, "argv", ["install_llama_prebuilt.py", "--validate-install", str(tmp_path)]
    )

    with pytest.raises(SystemExit) as caught:
        M.main()
    assert caught.value.code == M.EXIT_FALLBACK


def test_classifies_enospc_hidden_in_a_shutil_error(tmp_path):
    """copytree stringifies the per-file OSError, so errno and the chain are gone."""
    src = tmp_path / "src" / "sub"
    src.mkdir(parents = True)
    (src / "f").write_text("x", encoding = "utf-8")

    def boom(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(shutil.Error) as caught:
        shutil.copytree(tmp_path / "src", tmp_path / "dst", copy_function = boom)

    assert caught.value.errno is None
    assert M._environment_fatal_reason(caught.value)


def test_source_tree_enospc_is_not_masked_by_a_later_mirror_error(tmp_path, monkeypatch):
    """A full disk fails every mirror, so the first ENOSPC must win over a 404."""
    calls: list[str] = []

    def fake_download(
        url,
        path,
        *,
        expected_sha256 = None,
        label = None,
    ):
        calls.append(url)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(M, "download_file_verified", fake_download)

    with pytest.raises(PrebuiltFallback) as caught:
        M.hydrate_source_tree(
            "deadbeef",
            tmp_path / "install",
            tmp_path,
            source_repo = "unslothai/llama.cpp",
            expected_sha256 = None,
            exact_source = True,
            asset_url = "https://example.com/llama.cpp-source.tar.gz",
        )

    assert len(calls) == 1, f"stopped after the first ENOSPC, tried: {calls}"
    assert M._environment_fatal_reason(caught.value)


# ── exit codes ──


def test_enospc_exits_no_space_without_trying_older_releases(tmp_path, monkeypatch):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    newer = plan("b9002", "release-2", [choice("app-b9002-linux-x64-cpu.tar.gz")])
    older = plan("b9001", "release-1", [choice("app-b9001-linux-x64-cpu.tar.gz", "release-1")])
    reached = install_harness(monkeypatch, [newer, older], free_bytes = 50 * GB)

    def enospc(attempt, *args, **kwargs):
        reached.append(attempt.name)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(M, "validate_prebuilt_choice", enospc)

    with pytest.raises(SystemExit) as caught:
        M.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert caught.value.code == M.EXIT_NO_SPACE
    assert reached == ["app-b9002-linux-x64-cpu.tar.gz"]


def test_ordinary_failure_still_exits_fallback(tmp_path, monkeypatch):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    only = plan("b9002", "release-2", [choice("app-b9002-linux-x64-cpu.tar.gz")])
    install_harness(monkeypatch, [only], free_bytes = 50 * GB)
    monkeypatch.setattr(
        M,
        "validate_prebuilt_choice",
        lambda *a, **k: (_ for _ in ()).throw(PrebuiltFallback("checksum mismatch")),
    )

    with pytest.raises(SystemExit) as caught:
        M.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")

    assert caught.value.code == M.EXIT_FALLBACK
