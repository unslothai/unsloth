# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static discovery tests, distinct from native execution/qualification evidence."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import dependencies, runtime
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.fixture
def static_installation(tmp_path, monkeypatch):
    root = tmp_path / "Python space λ"
    root.mkdir()
    (root / "Lib" / "site-packages").mkdir(parents = True)
    (root / "DLLs").mkdir()
    (root / "python.exe").write_bytes(b"fixture-executable")
    (root / "python312.dll").write_bytes(b"fixture-library")

    def inspect(path):
        identity, _ = dependencies.read_regular_file(path, limit = 1024)
        return dependencies.NativeImage(
            identity,
            "x64",
            ("python312.dll",) if Path(path).suffix == ".exe" else (),
            (),
            (3, 12, 10150, 1013),
            "3.12.10",
            (),
        )

    monkeypatch.setattr(runtime, "inspect_native_image", inspect)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("discovery executed a program")
    )
    return root


def test_static_standard_layout_is_not_startup_authority(static_installation):
    descriptor = runtime.discover_runtime(str(static_installation / "python.exe"))
    assert descriptor.kind == "cpython"
    assert descriptor.version == (3, 12, 10)
    assert descriptor.architecture == "x64"
    assert descriptor.trust_classification == "payload_only"
    assert descriptor.prefix == descriptor.base_prefix == str(static_installation)
    assert descriptor.package_paths == (str(static_installation / "Lib" / "site-packages"),)
    runtime.require_profile_runtime(descriptor)
    assert not hasattr(descriptor, "qualified")


def test_venv_preserves_base_and_environment_without_executing_hooks(static_installation, tmp_path):
    venv = tmp_path / "venv λ"
    (venv / "Scripts").mkdir(parents = True)
    (venv / "Lib" / "site-packages").mkdir(parents = True)
    (venv / "Scripts" / "python.exe").write_bytes(b"venv-executable")
    (venv / "pyvenv.cfg").write_text(
        f"home = {static_installation}\nversion = 3.12.10\ninclude-system-site-packages = false\n",
        encoding = "utf-8",
    )
    sentinel = tmp_path / "hook-was-executed"
    hook = f"open({str(sentinel)!r}, 'w').write('unsafe')"
    (venv / "Lib" / "site-packages" / "sitecustomize.py").write_text(hook, encoding = "utf-8")
    (venv / "Lib" / "site-packages" / "danger.pth").write_text(
        "import os; " + hook, encoding = "utf-8"
    )
    descriptor = runtime.discover_runtime(str(venv / "Scripts" / "python.exe"))
    assert descriptor.kind == "venv"
    assert descriptor.prefix == str(venv)
    assert descriptor.base_prefix == str(static_installation)
    assert descriptor.package_paths == (str(venv / "Lib" / "site-packages"),)
    assert len(descriptor.configuration_files) == 1
    assert not sentinel.exists()


@pytest.mark.parametrize("kind", ["conda", "embedded"])
def test_recognized_but_unqualified_layout_is_not_admitted(static_installation, kind):
    if kind == "conda":
        (static_installation / "conda-meta").mkdir()
    else:
        (static_installation / "python312._pth").write_text(
            "python312.zip\nimport site\n", encoding = "utf-8"
        )
    descriptor = runtime.discover_runtime(str(static_installation / "python.exe"))
    assert descriptor.kind == kind
    with pytest.raises(WindowsRuntimeError, match = "WINDOWS_SANDBOX_LAYOUT_UNSUPPORTED"):
        runtime.require_profile_runtime(descriptor)


@pytest.mark.parametrize(
    "contents", ["home = relative\n", "home = a\nhome = b\n", "not a config", "version=3.12.10"]
)
def test_malformed_venv_cannot_choose_another_interpreter(static_installation, contents):
    (static_installation / "pyvenv.cfg").write_text(contents, encoding = "utf-8")
    with pytest.raises(WindowsRuntimeError):
        runtime.discover_runtime(str(static_installation / "python.exe"))


def test_runtime_architecture_and_version_mismatch_fail(static_installation, monkeypatch):
    original = runtime.inspect_native_image

    def mismatched(path):
        image = original(path)
        return replace(image, architecture = "arm64") if Path(path).suffix == ".dll" else image

    monkeypatch.setattr(runtime, "inspect_native_image", mismatched)
    with pytest.raises(WindowsRuntimeError, match = "architecture/version disagree"):
        runtime.discover_runtime(str(static_installation / "python.exe"))


def test_same_size_same_mtime_replacement_invalidates_descriptor(static_installation):
    executable = static_installation / "python.exe"
    before = runtime.discover_runtime(str(executable))
    info = executable.stat()
    executable.write_bytes(b"x" * info.st_size)
    os.utime(executable, ns = (info.st_atime_ns, info.st_mtime_ns))
    after = runtime.discover_runtime(str(executable))
    assert before.executable.file.size == after.executable.file.size
    assert before.digest != after.digest


def test_file_read_rejects_cross_boundary_hardlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"a")
    link = tmp_path / "alias"
    os.link(source, link)
    with pytest.raises(WindowsRuntimeError, match = "hardlinked"):
        dependencies.read_regular_file(link, limit = 10)


def test_file_read_rejects_ancestor_reparse_before_resolution(tmp_path, monkeypatch):
    directory = tmp_path / "redirected"
    directory.mkdir()
    target = directory / "file"
    target.write_bytes(b"data")
    original = Path.lstat

    def lstat(path):
        info = original(path)
        return (
            SimpleNamespace(st_mode = info.st_mode, st_file_attributes = 0x400)
            if path == directory
            else info
        )

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(WindowsRuntimeError, match = "reparse"):
        dependencies.read_regular_file(target, limit = 10)


def test_file_read_enforces_size_bound(tmp_path):
    path = tmp_path / "large"
    path.write_bytes(b"x" * 11)
    with pytest.raises(WindowsRuntimeError, match = "too large"):
        dependencies.read_regular_file(path, limit = 10)


@pytest.mark.parametrize(
    "name", [b"../a.dll", b"a/b.dll", b"C:\\host.dll", b"evil.dll:stream", b"a.dll\0", b"\xff.dll"]
)
def test_native_dependency_names_cannot_supply_paths(name):
    with pytest.raises(WindowsRuntimeError, match = "WINDOWS_SANDBOX_PE_INVALID"):
        dependencies._import_name(name)


def test_malformed_pe_is_not_an_image(tmp_path):
    path = tmp_path / "malformed.dll"
    path.write_bytes(b"MZ" + b"\0" * 100)
    with pytest.raises(WindowsRuntimeError, match = "WINDOWS_SANDBOX_PE_INVALID"):
        dependencies.inspect_native_image(path)


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows PE file metadata inspection")
def test_actual_selected_python_is_discovered_without_running_it(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("interpreter was executed")
    )
    descriptor = runtime.discover_runtime(sys.executable)
    assert descriptor.executable.file.path == str(Path(sys.executable).resolve())
    assert descriptor.version == sys.version_info[:3]
    assert descriptor.architecture == "x64"
    assert descriptor.runtime_dll.file.sha256
    assert descriptor.trust_classification == "payload_only"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows venv PE metadata inspection")
def test_actual_venv_discovery_does_not_execute_the_venv(tmp_path, monkeypatch):
    import venv

    root = tmp_path / "environment space λ"
    venv.EnvBuilder(with_pip = False).create(root)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("venv interpreter was executed")
    )
    descriptor = runtime.discover_runtime(str(root / "Scripts" / "python.exe"))
    assert descriptor.kind == "venv"
    assert descriptor.version == sys.version_info[:3]
    assert descriptor.prefix == str(root)
    assert descriptor.base_prefix == str(Path(sys.base_prefix).resolve())
    assert descriptor.trust_classification == "payload_only"
