# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Opt-in native SRT proof through Studio's actual execution entry points."""

import os
import sys
import importlib.util

import pytest

pytestmark = [
    pytest.mark.allow_network,
    pytest.mark.skipif(
        sys.platform != "linux" or os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") != "1",
        reason = "requires prepared native Linux SRT runtime",
    ),
]


@pytest.fixture
def native_session(tmp_path, monkeypatch):
    from core.inference import tools, os_sandbox

    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available, capability.reason
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id: str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_native_tool_executes_and_attests_success(native_session, kind):
    from core.inference import tools

    execute = tools._python_exec if kind == "python" else tools._bash_exec
    code = "print('NATIVE_STUDIO_SRT_OK')" if kind == "python" else "printf NATIVE_STUDIO_SRT_OK"
    records = []
    output = execute(code, None, 30, "native", launch_record_callback = records.append)
    assert "NATIVE_STUDIO_SRT_OK" in output, output
    assert len(records) == 1
    assert records[0].backend == "srt" and records[0].os_isolation


def test_nonzero_has_no_false_execution_attestation(native_session):
    from core.inference import tools

    records = []
    result = tools._python_exec(
        "raise ValueError('EXPECTED_PAYLOAD_FAILURE')",
        None,
        30,
        "native",
        launch_record_callback = records.append,
    )
    assert "EXPECTED_PAYLOAD_FAILURE" in result, result
    assert records == []


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_native_completion_after_output_survives_stream(native_session, monkeypatch, kind):
    import threading
    from core.inference import tools, tool_stream_exec

    original_start = threading.Thread.start

    def start(worker):
        original_start(worker)
        if worker.name.startswith("tool-exec-"):
            worker.join(timeout = 40)
            assert not worker.is_alive()

    monkeypatch.setattr(threading.Thread, "start", start)
    execute = tools._python_exec if kind == "python" else tools._bash_exec
    code = "print('NATIVE_STREAM_SUCCESS')" if kind == "python" else "printf NATIVE_STREAM_SUCCESS"
    cancel = threading.Event()

    def invoke(output, completion):
        return execute(code, None, 30, "native-stream", output_callback = output,
                       launch_record_callback = completion)

    gen = tool_stream_exec.stream_tool_execution(
        invoke, tool_name = kind, cancel_event = cancel,
        launch_event_factory = lambda record: {"type": "tool_execution", "record": record.as_dict()},
    )
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            assert "NATIVE_STREAM_SUCCESS" in stop.value
            break
    assert not cancel.is_set()
    assert "NATIVE_STREAM_SUCCESS" in "".join(e.get("text", "") for e in events)
    records = [e["record"] for e in events if e["type"] == "tool_execution"]
    assert len(records) == 1
    assert records[0]["backend"] == "srt" and records[0]["os_isolation"]


def test_selected_python_pillow_formats(native_session):
    from core.inference import tools

    records = []
    code = """
from PIL import Image
import io
for fmt in ('PNG', 'JPEG', 'BMP', 'GIF', 'TIFF', 'WEBP', 'ICO'):
    data = io.BytesIO()
    Image.new('RGB', (32,32), 'blue').save(data, format=fmt)
    data.seek(0)
    loaded = Image.open(data)
    loaded.load()
    assert loaded.size == (32,32)
print('PILLOW_SEVEN_FORMATS_OK')
"""
    result = tools._python_exec(code, None, 30, "native", launch_record_callback = records.append)
    assert "PILLOW_SEVEN_FORMATS_OK" in result, result
    assert len(records) == 1


def test_private_multiprocessing_resource_sharing(native_session):
    from core.inference import tools

    records = []
    code = """
import multiprocessing as mp
import multiprocessing.reduction as reduction
import socket

def worker(queue):
    endpoint = socket.socket(fileno=queue.get().detach())
    endpoint.sendall(b'PRIVATE_WORKER_OK')
    endpoint.close()

if __name__ == '__main__':
    context = mp.get_context('spawn')
    queue = context.Queue()
    parent, child = socket.socketpair()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    queue.put(reduction.DupFd(child.fileno()))
    parent.settimeout(10)
    assert parent.recv(64) == b'PRIVATE_WORKER_OK'
    process.join(10)
    assert process.exitcode == 0
    parent.close()
    child.close()
    queue.close()
    queue.join_thread()
    print('PRIVATE_MULTIPROCESSING_OK')
"""
    result = tools._python_exec(code, None, 30, "native", launch_record_callback = records.append)
    assert "PRIVATE_MULTIPROCESSING_OK" in result, result
    assert len(records) == 1


def _assert_actual_python_https_allowlist(monkeypatch):
    import urllib.request
    from core.inference import tools, os_sandbox

    # Establish a real host TLS positive control before treating sandbox failure as denial.
    with urllib.request.urlopen("https://pypi.org", timeout = 15) as response:
        assert response.status == 200
    monkeypatch.setenv("UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST", "pypi.org")
    capability = os_sandbox.capability_snapshot(force = True)
    assert "allowlist" in capability.network_policies, capability
    records = []
    code = """
import urllib.request
with urllib.request.urlopen('https://pypi.org', timeout=15) as response:
    print('HTTPS_STATUS', response.status)
try:
    urllib.request.urlopen('https://docs.python.org', timeout=5)
except Exception as error:
    print('EXPECTED_REFUSAL', str(error))
else:
    raise RuntimeError('disallowed host was reached')
"""
    result = tools._python_exec(
        code, None, 30, "native", network_policy = "allowlist", launch_record_callback = records.append
    )
    assert "HTTPS_STATUS 200" in result, result
    assert "EXPECTED_REFUSAL" in result and "403" in result, result
    assert len(records) == 1
    assert records[0].network_policy == "allowlist"
    assert records[0].network_allowlist == ("pypi.org",)


def test_actual_python_https_allowlist(native_session, monkeypatch):
    _assert_actual_python_https_allowlist(monkeypatch)


@pytest.mark.parametrize("cafile_setting", ["missing", "empty"])
def test_https_hashed_capath_without_cafile(native_session, monkeypatch, cafile_setting):
    import re
    import ssl
    import tempfile
    from pathlib import Path

    defaults = ssl.get_default_verify_paths()
    source = next(
        (
            Path(value)
            for value in (defaults.capath, defaults.openssl_capath)
            if value and Path(value).is_dir()
        ),
        None,
    )
    if source is None:
        pytest.skip("host OpenSSL has no hashed certificate directory")
    # This owned trust directory is outside the tool workdir; links resolve to
    # existing public trust files that the sandbox must not broadly grant.
    with tempfile.TemporaryDirectory(prefix = "unsloth-srt-capath-") as directory:
        capath = Path(directory) / "certs"
        capath.mkdir(mode = 0o700)
        count = 0
        for entry in source.iterdir():
            if re.fullmatch(r"[0-9a-fA-F]{8}\.[0-9]+", entry.name) and entry.is_file():
                target = entry.resolve(strict = True)
                assert not target.is_relative_to(native_session)
                (capath / entry.name).symlink_to(target)
                count += 1
        if not count:
            pytest.skip("host trust directory contains no hashed certificate files")
        monkeypatch.setenv(
            "SSL_CERT_FILE",
            str(Path(directory) / "missing.pem") if cafile_setting == "missing" else "",
        )
        monkeypatch.setenv("SSL_CERT_DIR", str(capath))
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising = False)
        assert ssl.get_default_verify_paths().cafile is None
        _assert_actual_python_https_allowlist(monkeypatch)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason = "optional selected CPU PyTorch environment"
)
def test_long_workdir_cpu_tensor_sharing(native_session, monkeypatch):
    from core.inference import tools
    import torch

    assert int(torch.arange(8).sum()) == 28
    work = native_session / ("long-session-" * 8)
    work.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id: str(work))
    records = []
    code = """
import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)

def worker(queue, done):
    value = queue.get()
    assert int(value.sum()) == 28
    value.add_(1)
    done.set()

if __name__ == '__main__':
    context = mp.get_context('spawn')
    queue = context.Queue()
    done = context.Event()
    tensor = torch.arange(8)
    tensor.share_memory_()
    process = context.Process(target=worker, args=(queue, done))
    process.start()
    queue.put(tensor)
    assert done.wait(15)
    process.join(15)
    assert process.exitcode == 0
    assert int(tensor.sum()) == 36
    queue.close()
    queue.join_thread()
    print('CPU_TENSOR_SHARING_OK')
"""
    output = tools._python_exec(code, None, 30, "native", launch_record_callback = records.append)
    assert "CPU_TENSOR_SHARING_OK" in output, output
    assert len(records) == 1
