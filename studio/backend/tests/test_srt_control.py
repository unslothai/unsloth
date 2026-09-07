# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real Node TCP receipts for the Windows adapter, without requiring SRT setup."""

import shutil
import subprocess
import threading

import pytest

from core.inference import srt_adapter


@pytest.fixture
def control_helper(tmp_path, monkeypatch):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the real control helper")
    monkeypatch.setattr(srt_adapter, "RUNTIME", tmp_path)
    monkeypatch.setattr(srt_adapter, "node_executable", lambda: node)
    processes = []
    streams = []
    original_popen = subprocess.Popen

    def launch(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        processes.append(proc)
        streams.append(proc.stdin)
        return proc

    monkeypatch.setattr(srt_adapter.subprocess, "Popen", launch)
    original_thread = threading.Thread

    class CompletedHelperBeforeRead(original_thread):
        def start(self):
            super().start()
            self.join(timeout = 5)
            assert not self.is_alive()
            # Force the fast-exit race: all receipts are buffered before the
            # adapter begins accepting/reading, and the helper already exited.
            processes[-1].wait(timeout = 5)

    monkeypatch.setattr(srt_adapter.threading, "Thread", CompletedHelperBeforeRead)
    yield tmp_path, processes, streams
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)
        srt_adapter.release_control(proc)


def _helper(path, variant = "valid"):
    path.joinpath("bridge.mjs").write_text(
        "import net from 'node:net';\n"
        "let input='';process.stdin.on('data',b=>input+=b);\n"
        "process.stdin.on('end',()=>{const r=JSON.parse(input);"
        "const socket=net.connect(r.controlSocket.port,'127.0.0.1',()=>{"
        f"const variant={variant!r};"
        "const events=[{v:1,event:'hello',token:variant==='auth'?'wrong':r.controlSocket.token},"
        "{v:1,event:'ready',version:'0.0.75'},{v:1,event:'spawned',pid:process.pid},"
        "{v:1,event:'exit',code:0,signal:null,reason:'completed'}];"
        "socket.end(variant==='malformed'?'not-json\\n':events.map(e=>JSON.stringify(e)).join('\\n')+'\\n',()=>process.exit(0));"
        "});});\n"
    )


def test_exited_helper_buffered_authenticated_receipts_are_accepted(control_helper):
    path, processes, streams = control_helper
    _helper(path)
    proc = srt_adapter._spawn_windows({"timeoutMs": 5000})
    assert proc is processes[0]
    assert proc.poll() == 0
    assert streams[0].closed
    srt_adapter.verify_success(proc)
    srt_adapter.release_control(proc)
    assert proc._srt_control_socket is None


@pytest.mark.parametrize(
    "variant,expected", [("auth", "authentication failed"), ("malformed", "Malformed")]
)
def test_control_rejects_bad_authentication_and_malformed_receipts(
    control_helper, variant, expected
):
    path, processes, streams = control_helper
    _helper(path, variant)
    with pytest.raises(srt_adapter.SrtError, match = expected):
        srt_adapter._spawn_windows({"timeoutMs": 5000})
    assert processes[0].poll() is not None
    assert streams[0].closed


def test_request_writer_closes_stream_after_broken_pipe(control_helper, monkeypatch):
    path, processes, streams = control_helper
    path.joinpath("bridge.mjs").write_text(
        "process.stdin.resume();process.stdin.on('end',()=>process.exit(0));"
    )

    def broken_write(*_args):
        raise BrokenPipeError("controlled request pipe failure")

    monkeypatch.setattr(srt_adapter.os, "write", broken_write)
    with pytest.raises(srt_adapter.SrtError, match = "exited before launch acknowledgement"):
        srt_adapter._spawn_windows({"timeoutMs": 5000})
    assert streams[0].closed
    assert processes[0].poll() == 0
