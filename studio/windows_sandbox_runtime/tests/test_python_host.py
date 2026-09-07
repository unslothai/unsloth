# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native CPython host development lane; no production admission override.

Only the explicitly selected development interpreter and checked-in shim are
admitted by this fixture. A generated content hash is not production trust.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import threading
import struct
import statistics
import time
import json
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from native_support import native_launch
from test_native_gate import authorize
from core.inference.windows_sandbox.content import RuntimeContentStore, SnapshotSpec, SnapshotFile
from core.inference.windows_sandbox.dependencies import read_regular_file
from core.inference.windows_sandbox.host_config import HostPaths, HostConfiguration
from core.inference.windows_sandbox.host_config import HEADER
from core.inference.windows_sandbox.native_plan import build_dependency_plan, windows_loader_policy
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, WindowsRuntimeError
from core.inference.windows_sandbox.runtime import discover_runtime


@pytest.fixture(scope = "module", params = ("base", "venv"))
def python_runtime(tmp_path_factory, request):
    if os.name != "nt":
        pytest.skip("Native Windows Python host lane")
    binary = Path(os.environ["UNSLOTH_TEST_PYTHON_HOST"]).resolve()
    assert binary.is_file(), "Explicit matching native Python host build is required"
    selected = os.environ.get(
        "UNSLOTH_TEST_PYTHON_EXECUTABLE", str(Path(sys.base_prefix) / "python.exe")
    )
    if request.param == "venv":
        # Explicit development interpreter only, no pip/activation/site hooks.
        environment = tmp_path_factory.mktemp("venv-space-λ") / "environment"
        subprocess.run(
            [selected, "-I", "-S", "-m", "venv", "--without-pip", str(environment)],
            check = True,
            capture_output = True,
            timeout = 30,
        )
        selected = str(environment / "Scripts/python.exe")
    descriptor = discover_runtime(selected)
    assert descriptor.kind == ("venv" if request.param == "venv" else "cpython")
    base = Path(descriptor.base_prefix)
    root = tmp_path_factory.mktemp("python-host-runtime")
    archive_path = root / "stdlib.zip"
    # All source reads are static and bounded; no source .pth or hooks execute.
    with zipfile.ZipFile(archive_path, "x", compression = zipfile.ZIP_STORED) as archive:
        total = 0
        for directory, dirs, names in os.walk(base / "Lib", followlinks = False):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in ("site-packages", "__pycache__", "test", "tests")
            )
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                path = Path(directory) / name
                _, data = read_regular_file(path, limit = 4 * 1024 * 1024)
                total += len(data)
                assert total <= 128 * 1024 * 1024
                archive.writestr(path.relative_to(base / "Lib").as_posix(), data)
    backend = Path(__file__).resolve().parents[2] / "backend"
    entries = [
        (archive_path, "runtime/stdlib.zip"),
        (backend / "core/inference/windows_sandbox/policy.py", "trusted/policy.py"),
        (backend / "core/inference/sandbox_site/sitecustomize.py", "trusted/sitecustomize.py"),
    ]
    entries += [(path, f"runtime/{path.name}") for path in base.glob("*.dll")]
    entries += [
        (path, f"runtime/DLLs/{path.name}")
        for path in (base / "DLLs").iterdir()
        if path.suffix.lower() in (".dll", ".pyd")
    ]
    features = (
        descriptor.runtime_dll.file.path,
        *(
            str(base / "DLLs" / (name + ".pyd"))
            for name in PYTHON_PROFILE.native_features
            if (base / "DLLs" / (name + ".pyd")).is_file()
        ),
    )
    plan = build_dependency_plan(
        features,
        (str(base), str(base / "DLLs")),
        architecture = "x64",
        system = windows_loader_policy(),
    )
    files = tuple(
        SnapshotFile(read_regular_file(path, limit = 128 * 1024 * 1024)[0], name)
        for path, name in entries
    )
    spec = SnapshotSpec(
        files,
        descriptor.digest,
        plan.digest,
        PYTHON_PROFILE.digest,
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    store = RuntimeContentStore(root / "protected")
    digest = store.publish(spec)
    paths = {str(path.resolve()): name for path, name in entries}
    native_images = tuple(paths[path] for path in plan.ordered_loads)
    yield SimpleNamespace(
        store = store,
        digest = digest,
        descriptor = descriptor,
        binary = binary,
        native_images = native_images,
    )
    assert not list((store.root / ".readers").iterdir())
    store.collect(digest)


@contextmanager
def python_launch(
    runtime,
    directory,
    source,
    *,
    change = None,
    before = None,
    **native_options,
):
    def configure(context):
        descriptor = runtime.descriptor
        script = context.workdir / "tool.py"
        script.write_text(source, encoding = "utf-8")
        if before is not None:
            before(context)
        access = runtime.store.read_access(runtime.digest, context.identity.sid_string).__enter__()
        try:
            generation = access.generation.directory / "files"
            home = generation / "runtime"
            paths = HostPaths(
                str(home / Path(descriptor.runtime_dll.file.path).name),
                str(home / getattr(runtime, "stdlib_relative", "stdlib.zip")),
                str(home / "DLLs"),
                str(home),
                descriptor.executable.file.path,
                descriptor.prefix,
                descriptor.base_prefix,
                str(generation / "trusted/policy.py"),
                str(generation / "trusted/sitecustomize.py"),
                str(context.workdir),
                str(script),
                context.identity.private_temp,
                str(context.aap),
            )
            config = HostConfiguration(
                paths,
                context.identity.sid_string,
                descriptor.version,
                context.nonce,
                context.profile,
                bytes.fromhex(runtime.digest),
                native_images = tuple(str(generation / path) for path in runtime.native_images),
            )
            if change is not None:
                config = change(config)
            return config, access
        except BaseException:
            access.close()
            raise

    with native_launch(
        runtime.binary, directory, configure_host = configure, **native_options
    ) as launch:
        yield launch


def run(launch):
    chunks, failures = [], []

    def read_output():
        size = 0
        try:
            while data := launch.process.stdout.read1(4096):
                size += len(data)
                if size > 1024 * 1024:
                    raise AssertionError("Native test output exceeded its bound")
                chunks.append(data)
        except BaseException as error:
            failures.append(error)

    reader = threading.Thread(target = read_output)
    reader.start()
    try:
        authorize(launch)
        result = launch.process.wait(timeout = 10)
    except Exception:
        launch.process.terminate()
        launch.process.wait(timeout = 5)
        reader.join(5)
        print(b"".join(chunks).decode("utf-8", errors = "replace"))
        raise
    finally:
        reader.join(5)
    assert not reader.is_alive() and not failures, failures
    output = b"".join(chunks).decode("utf-8", errors = "replace")
    assert result == 0, output
    return output


def test_real_python_host_initializes_threads_and_native_stdlib(
    python_runtime, tmp_path, record_property
):
    source = """
import sys, threading, json, math, sqlite3, ssl, hashlib
assert sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
assert sys.flags.utf8_mode and sys.flags.safe_path
assert __name__ == '__main__' and __file__ == sys.argv[0]
values = []
thread = threading.Thread(target=lambda: values.append(math.sqrt(81)))
thread.start(); thread.join(3)
assert values == [9] and not thread.is_alive()
assert sqlite3.connect(':memory:').execute('select 7').fetchone() == (7,)
assert hashlib.sha256(b'hello').hexdigest().startswith('2cf24dba')
open('payload-sentinel', 'w', encoding='utf-8').write('Python λ works')
print('PYTHON_HOST_OK', sys.version, flush=True)
"""
    with python_launch(python_runtime, tmp_path, source) as launch:
        output = run(launch)
        assert "PYTHON_HOST_OK" in output
        assert launch.sentinel.read_text(encoding = "utf-8") == "Python λ works"
        record_property("runtime_version", repr(python_runtime.descriptor.version))
        record_property("output", output)
        record_property("evidence", "NATIVE_PYTHON_HOST_DEVELOPMENT_NOT_PRODUCTION_QUALIFICATION")


def test_python_host_worker_diagnostics_preserve_imports_and_threads(python_runtime, tmp_path):
    source = """
import multiprocessing, concurrent.futures, subprocess
from multiprocessing.pool import ThreadPool
for operation in (lambda: multiprocessing.Process(target=print).start(),
                  lambda: multiprocessing.Pool(1), lambda: multiprocessing.Manager(),
                  lambda: concurrent.futures.ProcessPoolExecutor(1),
                  lambda: subprocess.Popen(['cmd.exe', '/c', 'echo', 'unexpected'])):
    try: operation()
    except RuntimeError as error:
        assert getattr(error, 'code', None) == 'WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED', repr(error)
    else: raise AssertionError('child process API succeeded')
with concurrent.futures.ThreadPoolExecutor(1) as pool:
    assert pool.submit(lambda: 13).result(timeout=3) == 13
with ThreadPool(1) as pool:
    assert pool.apply(lambda: 17) == 17
print('POLICY_OK', flush=True)
"""
    with python_launch(python_runtime, tmp_path, source) as launch:
        assert "POLICY_OK" in run(launch)


def test_python_host_rejects_actual_runtime_version_mismatch(python_runtime, tmp_path):
    def change(config):
        version = (*config.version[:2], config.version[2] + 1)
        return replace(config, version = version)

    with python_launch(
        python_runtime, tmp_path, "open('payload-sentinel', 'w').write('bad')", change = change
    ) as launch:
        with pytest.raises(WindowsRuntimeError, match = "stage 10"):
            authorize(launch)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


def test_python_host_asyncio_and_private_pipe_round_trip(python_runtime, tmp_path):
    source = """
import asyncio, multiprocessing
left, right = multiprocessing.Pipe(duplex=True)
try:
    left.send_bytes(b'private-request')
    assert right.poll(2) and right.recv_bytes() == b'private-request'
    right.send_bytes(b'private-response')
    assert left.poll(2) and left.recv_bytes() == b'private-response'
finally:
    left.close(); right.close()
async def main():
    await asyncio.sleep(0.01)
    assert await asyncio.to_thread(lambda: 23) == 23
    try:
        await asyncio.create_subprocess_exec('cmd.exe')
    except RuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED'
    else:
        raise AssertionError('async process launch succeeded')
import threading
asyncio.run(main(), debug=True)
errors = []
def runner():
    try: asyncio.run(main(), debug=True)
    except BaseException as error: errors.append(error)
thread = threading.Thread(target=runner)
thread.start(); thread.join(4)
assert not thread.is_alive() and not errors, errors
print('ASYNC_PRIVATE_IPC_OK', flush=True)
"""
    with python_launch(python_runtime, tmp_path, source) as launch:
        assert "ASYNC_PRIVATE_IPC_OK" in run(launch)


def test_python_host_startup_ignores_workdir_hooks_environment_and_argument_code(
    python_runtime, tmp_path
):
    def before(context):
        malicious = "open('injected', 'w').write('bad'); raise AssertionError('startup injection')"
        for name in ("sitecustomize.py", "usercustomize.py", "encodings.py", "socket.py"):
            (context.workdir / name).write_text(malicious, encoding = "utf-8")
        (context.workdir / "evil.pth").write_text("import sitecustomize\n", encoding = "utf-8")
        (context.workdir / "payload_module.py").write_text("value = 19", encoding = "utf-8")

    argument = "--help; open('injected', 'w').write('bad')\nλ"
    source = f"""
import sys, os, payload_module
assert payload_module.value == 19
assert sys.argv[1:] == [{argument!r}], sys.argv
assert not os.path.exists('injected')
assert sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment
print('INJECTION_DENIED', flush=True)
"""
    with python_launch(
        python_runtime,
        tmp_path,
        source,
        before = before,
        change = lambda config: replace(config, arguments = (argument,)),
        environment = {
            "PYTHONHOME": str(tmp_path / "work"),
            "PYTHONPATH": str(tmp_path / "work"),
            "PYTHONSTARTUP": str(tmp_path / "work/sitecustomize.py"),
        },
    ) as launch:
        assert not (launch.sentinel.parent / "injected").exists()
        assert "INJECTION_DENIED" in run(launch)
        assert not (launch.sentinel.parent / "injected").exists()


def _bad_config(data, case):
    result = bytearray(data)
    if case == "truncated":
        return data[: HEADER.size - 1]
    if case == "oversized":
        return data + b"\0" * 65536
    if case == "trailing":
        struct.pack_into("<I", result, 12, len(data) + 1)
        return bytes(result) + b"x"
    if case == "magic":
        result[0] ^= 1
    elif case in ("version", "size", "packages", "arguments", "images"):
        offset = {"version": 8, "size": 12, "packages": 28, "arguments": 32, "images": 36}[case]
        struct.pack_into("<I", result, offset, 0xFFFFFFFF)
    elif case == "odd-string":
        struct.pack_into("<I", result, HEADER.size, 3)
    elif case == "oversized-string":
        struct.pack_into("<I", result, HEADER.size, 32768)
    elif case == "surrogate":
        result[HEADER.size + 4 : HEADER.size + 6] = b"\0\xd8"
    elif case == "path-alias":
        result[HEADER.size + 4 : HEADER.size + 6] = b"/\0"
    else:
        raise AssertionError(case)
    return bytes(result)


@pytest.mark.parametrize(
    "case",
    [
        "truncated",
        "oversized",
        "trailing",
        "magic",
        "version",
        "size",
        "packages",
        "arguments",
        "images",
        "odd-string",
        "oversized-string",
        "surrogate",
        "path-alias",
    ],
)
def test_native_host_rejects_malformed_description_without_payload(python_runtime, tmp_path, case):
    with python_launch(
        python_runtime,
        tmp_path,
        "open('payload-sentinel', 'w').write('bad')",
        transform_config = lambda data: _bad_config(data, case),
    ) as launch:
        # Prove the parser's own exit before the broker's EOF rejection can
        # terminate it. Otherwise the expected fail-closed kill races exit 91.
        assert launch.process.wait(timeout = 5) == 91
        with pytest.raises(WindowsRuntimeError):
            authorize(launch)
        assert launch.process.poll() == 91
        assert not launch.sentinel.exists()
        assert launch.process.stdout.read() == b""


def test_python_host_asyncio_wakeup_saturation_and_handle_cleanup(python_runtime, tmp_path):
    source = """
import asyncio, ctypes, threading, gc
from ctypes import wintypes as W
kernel = ctypes.WinDLL('kernel32', use_last_error=True, winmode=0x800)
kernel.GetCurrentProcess.restype = W.HANDLE
kernel.GetProcessHandleCount.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
def count():
    result = W.DWORD()
    assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(result))
    return result.value
asyncio.run(asyncio.sleep(0))
gc.collect()
initial = count()
for repeat in range(5):
    loop = asyncio.ProactorEventLoop()
    errors, values = [], []
    loop.set_debug(True)
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    # Saturate the wakeup buffer while the event loop cannot consume it.
    # A blocking write here would deadlock, independently of timeout callbacks.
    for value in range(12000):
        loop.call_soon_threadsafe(values.append, value)
    async def main():
        await asyncio.sleep(0)
        assert values == list(range(12000))
        assert await asyncio.to_thread(lambda: 29) == 29
        done = loop.create_future()
        def worker():
            for value in range(1000): loop.call_soon_threadsafe(values.append, value)
            loop.call_soon_threadsafe(done.set_result, True)
        thread = threading.Thread(target=worker)
        thread.start()
        await asyncio.wait_for(done, 2)
        thread.join(2)
        assert not thread.is_alive()
        assert values[12000:] == list(range(1000))
        await loop.shutdown_default_executor()
    try:
        loop.run_until_complete(main())
        assert not errors, errors
    finally:
        loop.close()
        loop.close()
    # The close race must not write through a subsequently reused descriptor.
    loop._write_to_self()
    assert loop._csock is None and loop._ssock is None
    del loop
gc.collect()
assert count() <= initial + 2, (initial, count())
print('WAKEUP_CLEANUP_OK', flush=True)
"""
    with python_launch(python_runtime, tmp_path, source) as launch:
        assert "WAKEUP_CLEANUP_OK" in run(launch)


def test_python_host_private_ipc_does_not_enable_ip_network(python_runtime, tmp_path):
    source = """
import socket, asyncio, multiprocessing
asyncio.run(asyncio.sleep(0))
left, right = multiprocessing.Pipe()
left.close(); right.close()
for family in (socket.AF_INET, socket.AF_INET6):
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        try:
            handle = socket.socket(family, kind)
        except OSError as error:
            assert error.winerror == 10013, repr(error)
        else:
            handle.close()
            raise AssertionError('IP socket creation became available')
print('IP_DENIED_AFTER_PRIVATE_IPC', flush=True)
"""
    with python_launch(python_runtime, tmp_path, source) as launch:
        assert "IP_DENIED_AFTER_PRIVATE_IPC" in run(launch)


@pytest.mark.parametrize("missing", ["runtime", "native-image"])
def test_python_host_missing_startup_image_blocks_payload(python_runtime, tmp_path, missing):
    def change(config):
        path = str(Path(config.paths.runtime_home) / "missing-runtime-file.dll")
        if missing == "runtime":
            return replace(config, paths = replace(config.paths, runtime_dll = path))
        return replace(config, native_images = (path,))

    with python_launch(
        python_runtime, tmp_path, "open('payload-sentinel', 'w').write('bad')", change = change
    ) as launch:
        with pytest.raises(
            WindowsRuntimeError, match = f"stage {10 if missing == 'runtime' else 11}"
        ):
            authorize(launch)
        assert launch.process.poll() is not None
        assert not launch.sentinel.exists()


def test_python_host_cannot_open_host_named_pipe(python_runtime, tmp_path):
    import _winapi
    import secrets

    address = r"\\.\pipe\unsloth-host-control-" + secrets.token_hex(16)
    server = _winapi.CreateNamedPipe(
        address, _winapi.PIPE_ACCESS_DUPLEX, _winapi.PIPE_WAIT, 1, 4096, 4096, 0, _winapi.NULL
    )
    try:
        source = f"""
import _winapi
try:
    handle = _winapi.CreateFile({address!r}, _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
        0, _winapi.NULL, _winapi.OPEN_EXISTING, 0, _winapi.NULL)
except OSError as error:
    assert error.winerror == 5, repr(error)
else:
    _winapi.CloseHandle(handle)
    raise AssertionError('host named pipe was accessible')
print('HOST_PIPE_DENIED', flush=True)
"""
        with python_launch(python_runtime, tmp_path, source) as launch:
            assert "HOST_PIPE_DENIED" in run(launch)
        # Same endpoint still exists and accepts the ordinary host identity.
        control = _winapi.CreateFile(
            address,
            _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
            0,
            _winapi.NULL,
            _winapi.OPEN_EXISTING,
            0,
            _winapi.NULL,
        )
        _winapi.CloseHandle(control)
    finally:
        _winapi.CloseHandle(server)


def test_python_host_twenty_startups_and_cleanup(python_runtime, tmp_path, record_property):
    samples = []
    for index in range(20):
        directory = tmp_path / str(index)
        directory.mkdir()
        start = time.perf_counter()
        with python_launch(
            python_runtime, directory, "print('MEASURED_PYTHON_START', flush=True)"
        ) as launch:
            authorize(launch)
            samples.append((time.perf_counter() - start) * 1000)
            assert launch.process.wait(timeout = 5) == 0
            assert launch.process.stdout.read().strip() == b"MEASURED_PYTHON_START"
        assert not list((python_runtime.store.root / ".readers").iterdir())
    result = {
        "runtime": python_runtime.descriptor.version,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[18],
        "scope": "development fixture preparation through native startup ACK; prebuilt snapshot; not production",
    }
    record_property("python_startup_measurements", json.dumps(result))
    print(json.dumps(result))


def test_startup_failure_skips_loaded_library_detach_callbacks(python_runtime, tmp_path):
    """Missing CPython exports fail after DLL attach; normal teardown is observable."""
    import shutil
    from core.inference.windows_sandbox.profiles import WindowsRuntimeError

    binaries = python_runtime.binary.parent
    fixture = binaries / "detach_fixture.exe"
    control = binaries / "detach_control.exe"
    positive = tmp_path / "control"
    positive.mkdir()
    subprocess.run(
        [str(control), str(fixture)],
        env = {**os.environ, "UNSLOTH_TEST_DETACH_DIR": str(positive)},
        check = True,
        timeout = 10,
    )
    assert (positive / "attached").exists() and (positive / "detached").exists()
    target = tmp_path / "failure"
    target.mkdir()

    def change(config):
        copied = Path(config.paths.workdir) / "missing-python-exports.dll"
        shutil.copyfile(fixture, copied)
        return replace(config, paths = replace(config.paths, runtime_dll = str(copied)))

    with python_launch(
        python_runtime,
        target,
        "raise AssertionError('payload must never execute')",
        change = change,
        environment = {"UNSLOTH_TEST_DETACH_DIR": str(target / "work")},
    ) as launch:
        with pytest.raises(WindowsRuntimeError):
            authorize(launch)
        assert launch.process.wait(timeout = 5) != 0
        assert (target / "work/attached").exists(), "DLL initialization was not reached"
        assert not (target / "work/detached").exists(), "Failure invoked DLL teardown"
