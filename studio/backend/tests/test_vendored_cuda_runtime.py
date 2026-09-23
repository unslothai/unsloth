# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for locating a CUDA runtime another application ships privately.

The installer counts those dirs when it picks a runtime line, so the launchers
must put the matching one back on LD_LIBRARY_PATH. Pins the match rule (exact
CUDA major, complete runtime) and the refusals.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig

import pytest

from utils.prebuilt.runtime_libs import vendored_cuda_runtime_dirs

import utils.prebuilt.runtime_libs as runtime_libs

# The list as this host builds it, read before the autouse fixture below masks it:
# the multiarch arms assert on the shipped value, not on a pinned one.
_REAL_LOADER_DEFAULT_LIB_DIRS = runtime_libs._LOADER_DEFAULT_LIB_DIRS
_REAL_LD_CACHE_ENTRIES = runtime_libs._ld_cache_entries
_REAL_LOADER_RESOLVES = runtime_libs._loader_resolves_sonames

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "the vendored roots are Linux paths"
)


def _make_runtime(
    root,
    name: str,
    *,
    cudart: bool = True,
    cublas: bool = True,
    major = "13",
    major_soname_links: bool = True,
):
    directory = root / name
    directory.mkdir(parents = True)
    if cudart:
        payload = directory / f"libcudart.so.{major}.0.48"
        payload.write_bytes(b"")
        if major_soname_links:
            (directory / f"libcudart.so.{major}").symlink_to(payload.name)
    if cublas:
        payload = directory / f"libcublas.so.{major}.0.1"
        payload.write_bytes(b"")
        if major_soname_links:
            (directory / f"libcublas.so.{major}").symlink_to(payload.name)
    return directory


def _roots(tmp_path):
    return ((tmp_path, "cuda_v{major}"),)


@pytest.fixture(autouse = True)
def _host_loader_unknown(monkeypatch):
    # The match rule is what these pin, and the loader probe is host state; the
    # arms that need it pin their own answer.
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: None)
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", ())
    monkeypatch.setattr(runtime_libs, "_loader_resolves_sonames", lambda _sonames: False)


def test_matches_the_marker_runtime_line(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path))

    assert result == [str(runtime_dir.resolve())]


def test_ignores_a_different_cuda_major(tmp_path):
    _make_runtime(tmp_path, "cuda_v13")
    _make_runtime(tmp_path, "cuda_v12", major = "12")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda12"}, roots = _roots(tmp_path))

    assert result == [str((tmp_path / "cuda_v12").resolve())]


def test_accepts_a_minor_qualified_directory(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13.0")

    result = vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path))

    assert result == [str(runtime_dir.resolve())]


def test_rejects_a_longer_major_that_shares_the_prefix(tmp_path):
    # cuda_v130 answers a cuda_v13* glob but is not CUDA 13.
    _make_runtime(tmp_path, "cuda_v130", major = "130")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.parametrize("missing", ["cudart", "cublas"])
def test_requires_a_complete_runtime(tmp_path, missing):
    _make_runtime(tmp_path, "cuda_v13", cudart = missing != "cudart", cublas = missing != "cublas")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


def test_rejects_payloads_without_major_soname_links(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13", major_soname_links = False)

    assert (runtime_dir / "libcudart.so.13.0.48").is_file()
    assert (runtime_dir / "libcublas.so.13.0.1").is_file()
    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.skipif(
    os.geteuid() == 0, reason = "root can read mode-000 files despite missing permission bits"
)
def test_rejects_unreadable_major_soname_links(tmp_path):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    for soname in ("libcudart.so.13", "libcublas.so.13"):
        (runtime_dir / soname).resolve().chmod(0)
        assert not os.access(runtime_dir / soname, os.R_OK)

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


def test_ignores_a_file_named_like_a_runtime_dir(tmp_path):
    (tmp_path / "cuda_v13").write_bytes(b"")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.parametrize(
    "marker",
    [
        None,  # source build / unreadable or corrupt marker
        [],  # valid JSON, wrong shape
        "cuda13",
        {},  # marker without a runtime line
        {"runtime_line": None},
        {"runtime_line": 13},  # not a string
        {"runtime_line": "cpu"},
        {"runtime_line": "vulkan"},
    ],
)
def test_yields_nothing_without_a_cuda_runtime_line(tmp_path, marker):
    _make_runtime(tmp_path, "cuda_v13")

    assert vendored_cuda_runtime_dirs(marker, roots = _roots(tmp_path)) == []


def test_missing_root_is_not_an_error(tmp_path):
    roots = ((tmp_path / "nonexistent", "cuda_v{major}"),)

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = roots) == []


def test_withholds_the_runtime_when_the_loader_already_finds_it(tmp_path, monkeypatch):
    # The dir would join LD_LIBRARY_PATH, which outranks the loader's cache and
    # its defaults, so a runtime the loader resolves must not be displaced.
    import utils.prebuilt.runtime_libs as runtime_libs

    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    system_dir = tmp_path / "system-runtime"
    system_dir.mkdir()
    cudart = system_dir / "libcudart.so.13"
    cublas = system_dir / "libcublas.so.13"
    cudart.write_bytes(b"")
    cublas.write_bytes(b"")
    monkeypatch.setattr(
        runtime_libs,
        "_ld_cache_entries",
        lambda: (
            ("libcudart.so.13", "libc6,x86-64", str(cudart)),
            ("libcublas.so.13", "libc6,x86-64", str(cublas)),
        ),
    )
    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []
    # Only the pair counts: the loader cannot link the build without both, so
    monkeypatch.setattr(
        runtime_libs,
        "_ld_cache_entries",
        lambda: (("libcudart.so.13", "libc6,x86-64", str(cudart)),),
    )
    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]


def test_ignores_cache_entries_with_missing_targets(tmp_path, monkeypatch):
    # ldconfig can retain SONAME/ABI records after a library target is removed.
    # That stale pair cannot satisfy the dynamic loader, so the usable vendored
    # pair must still be added.
    import utils.prebuilt.runtime_libs as runtime_libs

    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    system_dir = tmp_path / "system-runtime"
    system_dir.mkdir()
    cudart = system_dir / "libcudart.so.13"
    cudart.write_bytes(b"")
    missing_cublas = system_dir / "libcublas.so.13"
    monkeypatch.setattr(
        runtime_libs,
        "_ld_cache_entries",
        lambda: (
            ("libcudart.so.13", "libc6,x86-64", str(cudart)),
            ("libcublas.so.13", "libc6,x86-64", str(missing_cublas)),
        ),
    )

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]


def test_checks_loader_defaults_even_when_cache_is_readable(tmp_path, monkeypatch):
    import utils.prebuilt.runtime_libs as runtime_libs

    _make_runtime(tmp_path, "cuda_v13")
    default_dir = tmp_path / "loader-default"
    default_dir.mkdir()
    (default_dir / "libcudart.so.13").write_bytes(b"")
    (default_dir / "libcublas.so.13").write_bytes(b"")
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: ())
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", (str(default_dir),))

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


@pytest.mark.skipif(
    os.geteuid() == 0, reason = "root can read mode-000 files despite missing permission bits"
)
def test_unreadable_loader_default_files_do_not_block_the_vendored_rescue(tmp_path, monkeypatch):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    default_dir = tmp_path / "loader-default"
    default_dir.mkdir()
    system_libraries = [
        default_dir / "libcudart.so.13",
        default_dir / "libcublas.so.13",
    ]
    for library in system_libraries:
        library.write_bytes(b"inaccessible regular file")
        library.chmod(0)

    # Exercise actual filesystem permissions rather than mocking access checks.
    assert all(
        library.is_file() and not os.access(library, os.R_OK) for library in system_libraries
    )
    with pytest.raises(PermissionError):
        system_libraries[0].open("rb")

    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: ())
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", (str(default_dir),))

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]


def test_ignores_foreign_abi_cache_entries(tmp_path, monkeypatch):
    import platform

    if platform.machine().lower() not in {"x86_64", "amd64"}:
        pytest.skip("synthetic foreign-ABI cache fixture is x86-64-specific")

    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    monkeypatch.setattr(runtime_libs.shutil, "which", lambda _candidate: "/sbin/ldconfig")
    original_exists = runtime_libs.os.path.exists
    monkeypatch.setattr(
        runtime_libs.os.path,
        "exists",
        lambda path: True if path == "/sbin/ldconfig" else original_exists(path),
    )
    result = type(
        "Result",
        (),
        {
            "returncode": 0,
            "stdout": (
                "libcudart.so.13 (libc6,i686) => /usr/lib/i386-linux-gnu/libcudart.so.13\\n"
                "libcublas.so.13 (libc6,i686) => /usr/lib/i386-linux-gnu/libcublas.so.13\\n"
            ),
        },
    )()
    monkeypatch.setattr(runtime_libs.subprocess, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", ())

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux loader cache only")
def test_recognizes_aarch64_ldconfig_abi_spelling_case_insensitively(tmp_path, monkeypatch):
    import utils.prebuilt.runtime_libs as runtime_libs

    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", _REAL_LD_CACHE_ENTRIES)

    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    system_dir = tmp_path / "system-cache"
    system_dir.mkdir()
    cudart = system_dir / "libcudart.so.13"
    cublas = system_dir / "libcublas.so.13"
    cudart.write_bytes(b"")
    cublas.write_bytes(b"")
    monkeypatch.setattr(runtime_libs.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(runtime_libs.shutil, "which", lambda _candidate: None)
    original_exists = runtime_libs.os.path.exists
    monkeypatch.setattr(
        runtime_libs.os.path,
        "exists",
        lambda path: True if path == "/sbin/ldconfig" else original_exists(path),
    )
    result = type(
        "Result",
        (),
        {
            "returncode": 0,
            "stdout": (
                f"libcudart.so.13 (libc6,AArch64) => {cudart}\n"
                f"libcublas.so.13 (libc6,AArch64) => {cublas}\n"
            ),
        },
    )()
    monkeypatch.setattr(runtime_libs.subprocess, "run", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", ())

    assert runtime_libs._ld_cache_entries() == (
        ("libcudart.so.13", "libc6,aarch64", str(cudart)),
        ("libcublas.so.13", "libc6,aarch64", str(cublas)),
    )
    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


def test_an_unreadable_cache_still_rescues_from_the_default_dirs(tmp_path, monkeypatch):
    # No ldconfig is ignorance, not absence: the default dirs answer, and the
    # dir is added when they have nothing either.
    import utils.prebuilt.runtime_libs as runtime_libs

    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: None)
    default = tmp_path / "default"
    default.mkdir()
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", (str(default),))

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]
    (default / "libcudart.so.13").write_bytes(b"")
    (default / "libcublas.so.13").write_bytes(b"")

    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == []


def test_probes_loader_when_ldconfig_cache_query_fails(monkeypatch):
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: None)
    monkeypatch.setattr(runtime_libs, "_loader_resolves_sonames", lambda _sonames: True)

    assert runtime_libs._loader_already_provides_runtime("13")


def test_uses_the_dynamic_loader_cache_without_ldconfig(monkeypatch):
    sonames = ("libcudart.so.13", "libcublas.so.13")
    native_abis = runtime_libs._NATIVE_LOADER_ABIS.get(
        runtime_libs.platform.machine().lower(), frozenset()
    )
    cached = _REAL_LD_CACHE_ENTRIES()
    cache_paths = {
        soname: next(
            (
                path
                for cached_soname, abi, path in cached or ()
                if cached_soname == soname
                and abi in native_abis
                and os.path.isfile(path)
                and os.access(path, os.R_OK)
                and not any(
                    os.path.dirname(path) == default_dir
                    for default_dir in _REAL_LOADER_DEFAULT_LIB_DIRS
                )
            ),
            None,
        )
        for soname in sonames
    }
    if not all(cache_paths.values()):
        pytest.skip("host has no accessible CUDA 13 pair in a non-default loader-cache directory")

    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    loader_probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import ctypes, sys; [ctypes.CDLL(name) for name in sys.argv[1:]]",
            *sonames,
        ],
        capture_output = True,
        env = env,
        timeout = 10,
    )
    if loader_probe.returncode:
        pytest.skip("dynamic loader cannot resolve the cached CUDA 13 pair on this host")

    monkeypatch.setattr(runtime_libs, "_loader_resolves_sonames", _REAL_LOADER_RESOLVES)
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", _REAL_LD_CACHE_ENTRIES)
    monkeypatch.setattr(runtime_libs.shutil, "which", lambda _candidate: None)
    original_exists = runtime_libs.os.path.exists
    monkeypatch.setattr(
        runtime_libs.os.path,
        "exists",
        lambda path: False
        if path in {"/sbin/ldconfig", "/usr/sbin/ldconfig"}
        else original_exists(path),
    )
    monkeypatch.setattr(runtime_libs, "_LOADER_DEFAULT_LIB_DIRS", ())

    assert runtime_libs._ld_cache_entries() is None
    assert runtime_libs._loader_already_provides_runtime("13")


def test_the_multiarch_dirs_are_discovered_not_assumed(tmp_path, monkeypatch):
    # The dirs come from what is installed, so a layout nobody wrote down (a
    # multilib host, riscv64, a container with only one of the pair) is still
    # covered. Pinned against a fixed layout here, where the real one would make
    # the test say nothing on a host that has neither.
    import utils.prebuilt.runtime_libs as runtime_libs

    for name in ("lib/x86_64-linux-gnu", "usr/lib/x86_64-linux-gnu", "usr/lib/riscv64-linux-gnu"):
        (tmp_path / name).mkdir(parents = True)
    (tmp_path / "usr/lib/not-a-layout").mkdir(parents = True)
    monkeypatch.setattr(
        runtime_libs,
        "_MULTIARCH_LIB_GLOBS",
        (str(tmp_path / "lib/*-linux-gnu*"), str(tmp_path / "usr/lib/*-linux-gnu*")),
    )

    assert runtime_libs._multiarch_lib_dirs() == [
        str(tmp_path / "lib/x86_64-linux-gnu"),
        str(tmp_path / "usr/lib/riscv64-linux-gnu"),
        str(tmp_path / "usr/lib/x86_64-linux-gnu"),
    ]


def test_the_default_dirs_cover_the_hosts_multiarch_paths():
    # The real list, not a pinned one: the finding is about an actual Debian/Ubuntu
    # host, where a distro CUDA runtime sits in the arch-specific dir and no
    # ldconfig can be read. The triple comes from the interpreter rather than from
    # the module, so this cannot pass by agreeing with the implementation.
    triple = sysconfig.get_config_var("MULTIARCH")
    if not triple:
        pytest.skip("this interpreter records no MULTIARCH triple")
    installed = [
        f"{prefix}/{triple}"
        for prefix in ("/lib", "/usr/lib")
        if os.path.isdir(f"{prefix}/{triple}")
    ]
    if not installed:
        pytest.skip(f"no {triple} directory is installed on this host")
    for directory in installed:
        assert directory in _REAL_LOADER_DEFAULT_LIB_DIRS


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux loader path only")
def test_usr_local_is_not_assumed_without_a_readable_cache(tmp_path, monkeypatch):
    runtime_dir = _make_runtime(tmp_path, "cuda_v13")
    monkeypatch.setattr(runtime_libs, "_ld_cache_entries", lambda: None)

    # /usr/local is usually admitted through ld.so.conf/cache, not glibc's
    # built-in defaults. Without a readable cache it must not mask the rescue.
    assert not any(
        directory.rstrip("/") in {"/usr/local/lib", "/usr/local/lib64"}
        for directory in _REAL_LOADER_DEFAULT_LIB_DIRS
    )
    assert vendored_cuda_runtime_dirs({"runtime_line": "cuda13"}, roots = _roots(tmp_path)) == [
        str(runtime_dir.resolve())
    ]
