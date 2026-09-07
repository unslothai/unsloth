# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real broker binding and static inventory tests; not sandbox qualification."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import admission
from core.inference.windows_sandbox.native_plan import ScanBounds
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox.runtime import discover_runtime

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason = "Native Windows broker admission lane"
)


def test_admission_binds_actual_broker_and_declared_core_only():
    value = admission.admit_studio_runtime(sys.executable)
    assert value.runtime.version == sys.version_info[:3]
    assert Path(value.runtime.executable.file.path) == Path(sys.executable).resolve()
    assert Path(value.runtime.runtime_dll.file.path) == admission._loaded_python_path()
    assert value.broker_pid == os.getpid()
    assert value.origin == "running_studio_cpython_core_v1"
    assert (
        value.runtime.trust_classification
        == value.dependencies.trust_classification
        == "payload_only"
    )
    assert not hasattr(value, "available") and not hasattr(value, "qualified")
    names = {item.relative_path for item in value.files}
    assert "runtime/Lib/encodings/__init__.py" in names
    assert "runtime/Lib/asyncio/__init__.py" in names
    assert "runtime/DLLs/_overlapped.pyd" in names
    assert not any(
        "site-packages" in name or name.endswith(("sitecustomize.py", "usercustomize.py", ".pth"))
        for name in names
    )
    assert {image.file.path for image in value.dependencies.images} <= {
        item.source.path for item in value.files
    }
    assert value.digest == replace(value, broker_pid = value.broker_pid + 1).digest
    assert value.digest != replace(value, origin = "not_the_same_policy").digest


def test_copied_python_does_not_gain_startup_trust(tmp_path, monkeypatch):
    copied = tmp_path / "python.exe"
    shutil.copyfile(sys.executable, copied)

    def forbidden(*args, **kwargs):
        pytest.fail("a different interpreter reached discovery")

    monkeypatch.setattr(admission, "discover_runtime", forbidden)
    with pytest.raises(WindowsRuntimeError, match = "Only Studio's running interpreter"):
        admission.admit_studio_runtime(str(copied))


@pytest.mark.parametrize("field", ["version", "prefix", "base_prefix", "runtime_dll"])
def test_forged_descriptor_is_not_broker_authority(field, monkeypatch, tmp_path):
    real = discover_runtime(sys.executable)
    values = {
        "version": (3, 12, 999),
        "prefix": str(tmp_path),
        "base_prefix": str(tmp_path),
        "runtime_dll": replace(
            real.runtime_dll, file = replace(real.runtime_dll.file, path = str(tmp_path / "python.dll"))
        ),
    }
    monkeypatch.setattr(
        admission, "discover_runtime", lambda path: replace(real, **{field: values[field]})
    )
    monkeypatch.setattr(
        admission,
        "build_dependency_plan",
        lambda *a, **k: pytest.fail("invalid identity reached dependency loading"),
    )
    with pytest.raises(WindowsRuntimeError, match = "does not match"):
        admission.admit_studio_runtime(sys.executable)


@pytest.mark.parametrize("length", [0, 32768])
def test_loaded_module_query_error_or_truncation_has_no_path_fallback(monkeypatch, length):
    def query(handle, output, size):
        assert handle == sys.dllhandle and size == 32768
        return length

    monkeypatch.setattr(
        admission.ctypes, "WinDLL", lambda *a, **k: SimpleNamespace(GetModuleFileNameW = query)
    )
    with pytest.raises(WindowsRuntimeError, match = "unavailable or truncated"):
        admission._loaded_python_path()


def _stdlib(tmp_path):
    root = tmp_path / "Lib"
    (root / "encodings").mkdir(parents = True)
    (root / "encodings/__init__.py").write_text("# fixed fixture", encoding = "utf-8")
    return root


def inventory(root, **kwargs):
    return admission._stdlib_inventory(
        root.parent,
        frozenset(("encodings", "json")),
        kwargs.pop("bounds", ScanBounds()),
        kwargs.pop("deadline", time.monotonic() + 5),
    )


def test_unknown_modules_packages_native_images_and_hooks_are_not_admitted(tmp_path):
    root = _stdlib(tmp_path)
    for name in ("evil.py", "sitecustomize.py", "usercustomize.py", "a.pth", "evil.pyd"):
        (root / name).write_text("raise AssertionError('must not execute')", encoding = "utf-8")
    for name in ("site-packages", "unknown", "__pycache__"):
        (root / name).mkdir()
        (root / name / "secret.py").write_text(
            "raise AssertionError('must not execute')", encoding = "utf-8"
        )
    assert [item.relative_path for item in inventory(root)] == ["runtime/Lib/encodings/__init__.py"]


def test_known_package_inventory_is_content_bound_and_does_not_import(tmp_path):
    root = _stdlib(tmp_path)
    (root / "json").mkdir()
    module = root / "json/__init__.py"
    module.write_text("raise AssertionError('static only')", encoding = "utf-8")
    first = inventory(root)
    metadata = module.stat()
    module.write_text("raise AssertionError('STATIC ONLY')", encoding = "utf-8")
    os.utime(module, ns = (metadata.st_atime_ns, metadata.st_mtime_ns))
    second = inventory(root)
    assert len(first) == len(second) == 2
    assert first[-1].source.sha256 != second[-1].source.sha256


def test_admitted_source_hardlink_is_rejected(tmp_path):
    root = _stdlib(tmp_path)
    original = tmp_path / "outside.py"
    original.write_text("# outside", encoding = "utf-8")
    os.link(original, root / "json.py")
    with pytest.raises(WindowsRuntimeError, match = "hardlinked"):
        inventory(root)


@pytest.mark.parametrize("bounds", [ScanBounds(entries = 1), ScanBounds(bytes = 1)])
def test_inventory_limits_fail_closed(tmp_path, bounds):
    root = _stdlib(tmp_path)
    with pytest.raises(WindowsRuntimeError, match = "limit"):
        inventory(root, bounds = bounds)


def test_inventory_expired_cooperative_deadline_fails_closed(tmp_path):
    root = _stdlib(tmp_path)
    with pytest.raises(WindowsRuntimeError, match = "limit"):
        inventory(root, deadline = 0)


def test_passing_inventory_does_not_accept_approval_flag():
    with pytest.raises(TypeError):
        admission.admit_studio_runtime(sys.executable, trusted = True)


def test_actual_junction_inside_admitted_package_is_rejected(tmp_path):
    import _winapi

    root = _stdlib(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("# outside", encoding = "utf-8")
    link = root / "encodings/foreign"
    _winapi.CreateJunction(str(outside), str(link))
    try:
        assert (link / "secret.py").read_text() == "# outside"
        with pytest.raises(WindowsRuntimeError, match = "reparse"):
            inventory(root)
    finally:
        link.rmdir()


@pytest.mark.parametrize("layout", ["base", "venv"])
def test_selected_development_python_can_bind_its_own_running_identity(tmp_path, layout):
    import importlib.util

    selected = os.environ.get(
        "UNSLOTH_TEST_PYTHON_EXECUTABLE", str(Path(sys.base_prefix) / "python.exe")
    )
    if layout == "venv":
        environment = tmp_path / "Studio λ environment"
        subprocess.run(
            [selected, "-I", "-S", "-m", "venv", "--without-pip", str(environment)],
            check = True,
            capture_output = True,
            timeout = 30,
        )
        selected = str(environment / "Scripts/python.exe")
    backend = str(Path(__file__).resolve().parents[2] / "backend")
    parser = str(Path(importlib.util.find_spec("pefile").origin).parent)
    source = f"""
import sys
sys.path[:0] = [{backend!r}, {parser!r}]
from core.inference.windows_sandbox.admission import admit_studio_runtime
value = admit_studio_runtime(sys.executable)
assert value.runtime.kind == {"venv" if layout == "venv" else "cpython"!r}
assert value.runtime.version == sys.version_info[:3]
assert value.runtime.prefix == sys.prefix
assert value.runtime.base_prefix == sys.base_prefix
print('OWN_BROKER_IDENTITY', value.runtime.version, len(value.files))
"""
    # Explicit local test interpreter, not a discovered executable in production.
    result = subprocess.run(
        [selected, "-I", "-c", source], capture_output = True, text = True, encoding = "utf-8", timeout = 30
    )
    assert result.returncode == 0, result.stderr
    assert "OWN_BROKER_IDENTITY" in result.stdout


def test_admitted_core_snapshot_runs_through_native_host(tmp_path):
    import hashlib
    from core.inference.windows_sandbox.content import (
        RuntimeContentStore,
        SnapshotSpec,
        SnapshotFile,
    )
    from core.inference.windows_sandbox.dependencies import read_regular_file
    from core.inference.windows_sandbox.profiles import PYTHON_PROFILE
    from test_python_host import python_launch, run

    binary = Path(os.environ["UNSLOTH_TEST_PYTHON_HOST"]).resolve()
    from core.inference.windows_sandbox.preparation import prepare_admitted_core

    core = prepare_admitted_core(sys.executable)
    # In CI the host is built for the actual test runner's ABI; no nearby build.
    assert f"cpython-{sys.version_info.major}{sys.version_info.minor}-x64" in binary.name
    backend = Path(__file__).resolve().parents[2] / "backend"
    shims = (
        (backend / "core/inference/windows_sandbox/policy.py", "trusted/policy.py"),
        (backend / "core/inference/sandbox_site/sitecustomize.py", "trusted/sitecustomize.py"),
    )
    spec = SnapshotSpec(
        core.files
        + tuple(
            SnapshotFile(read_regular_file(path, limit = 1024 * 1024)[0], name)
            for path, name in shims
        ),
        core.digest,
        core.dependencies.digest,
        PYTHON_PROFILE.digest,
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    store = RuntimeContentStore(tmp_path / "admitted-snapshot")
    digest = store.publish(spec)
    source_to_relative = {item.source.path: item.relative_path for item in spec.files}
    runtime = SimpleNamespace(
        store = store,
        digest = digest,
        descriptor = core.runtime,
        binary = binary,
        stdlib_relative = "Lib",
        native_images = tuple(source_to_relative[path] for path in core.dependencies.ordered_loads),
    )
    directory = tmp_path / "launch"
    directory.mkdir()
    source = """
import asyncio, concurrent.futures, json, sqlite3, ssl, sys
assert asyncio.run(asyncio.sleep(0, result=41)) == 41
assert sqlite3.connect(':memory:').execute('select 43').fetchone() == (43,)
with concurrent.futures.ThreadPoolExecutor(1) as pool:
    assert pool.submit(lambda: 47).result(timeout=2) == 47
print('ADMITTED_NATIVE_HOST_OK', flush=True)
"""
    try:
        with python_launch(runtime, directory, source) as launch:
            assert "ADMITTED_NATIVE_HOST_OK" in run(launch)
    finally:
        assert not list((store.root / ".readers").iterdir())
        store.collect(digest)
