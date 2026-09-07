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


def test_actual_python_https_allowlist(native_session, monkeypatch):
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
