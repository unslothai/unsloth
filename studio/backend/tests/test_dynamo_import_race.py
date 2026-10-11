# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A download racing the torch warm's ``import torch._dynamo`` must not poison the process.

The warm enters the torch._dynamo / torch._inductor import cycle at ``torch._dynamo``; a load
thread's first download imports ``unsloth_zoo`` (via ``utils.hf_xet_fallback``), which enters
the same cycle at ``torch._inductor``. The two threads then take the two package locks in
opposite order, CPython's import deadlock detector hands one of them a half-built module, and
the process keeps failing with ``partially initialized module 'torch._dynamo' ... has no
attribute 'utils'`` until a restart.

The race is made deterministic with real torch in a fresh subprocess: the warm thread is
paused INSIDE ``import torch._dynamo`` (holding its module lock) until the download thread is
either blocked on that import or waiting at Studio's gate, then released. ``unsloth_zoo`` is a
stand-in whose ``hf_xet_fallback`` imports ``torch._inductor.utils``, as the real one does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

pytest.importorskip("torch")

_CHILD = textwrap.dedent(
    r"""
    import importlib, importlib._bootstrap as bootstrap, json, sys, threading, time
    from importlib.machinery import PathFinder

    backend, fake_root = sys.argv[1], sys.argv[2]
    sys.path[:0] = [backend, fake_root]
    import torch
    assert "torch._dynamo" not in sys.modules, "torch imported dynamo eagerly; race untestable"

    from utils import hf_xet_fallback, torch_warmup

    hf_xet_fallback._gpu_present = lambda: True  # one import attempt, no CPU retry path
    reader = {}

    def reader_is_parked():
        tid = reader.get("tid")
        if tid is None:
            return False
        if reader["thread"].is_alive() is False:
            return True
        blocking = getattr(bootstrap, "_blocking_on", {}).get(tid)
        if blocking:
            return True  # waiting on a module lock the warm holds
        frame = sys._current_frames().get(tid)
        while frame is not None:
            if frame.f_code.co_name == "ensure_dynamo_imported":
                return True  # waiting at Studio's gate
            frame = frame.f_back
        return False

    class _PausingLoader:
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            reader_started.set()
            deadline = time.monotonic() + 120
            while not reader_is_parked() and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.2)
            self.inner.exec_module(module)

    class _PauseWarmInsideDynamo:
        def find_spec(self, name, path = None, target = None):
            if name != "torch._dynamo.aot_compile" or threading.current_thread().name != "warm":
                return None
            spec = PathFinder.find_spec(name, path, target)
            spec.loader = _PausingLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _PauseWarmInsideDynamo())
    reader_started = threading.Event()
    result = {}

    def warm():
        result["warm"] = torch_warmup.ensure_dynamo_imported()

    def download():
        try:
            result["download"] = hf_xet_fallback._load_shared()
        except BaseException as exc:
            result["download"] = repr(exc)
        result["download_error"] = repr(hf_xet_fallback._shared_import_error)

    w = threading.Thread(target = warm, name = "warm")
    w.start()
    assert reader_started.wait(120), "warm never reached torch._dynamo"
    r = threading.Thread(target = download, name = "download")
    reader["thread"] = r
    r.start()
    reader["tid"] = r.ident
    w.join(300)
    r.join(300)

    # What the next image load would see: the process is either whole or poisoned.
    try:
        import torch._dynamo.utils
        import torch._inductor.utils
        result["after"] = bool(getattr(torch._dynamo, "utils", None)) and hasattr(
            torch._inductor.utils, "IndentedBuffer"
        )
    except BaseException as exc:
        result["after"] = repr(exc)
    print("RESULT " + json.dumps(result), flush = True)
    """
)


def _run(tmp_path: Path, extra_env: dict | None = None) -> dict:
    fake = tmp_path / "fake_site"
    (fake / "unsloth_zoo").mkdir(parents = True)
    (fake / "unsloth_zoo" / "__init__.py").write_text("")
    (fake / "unsloth_zoo" / "hf_xet_fallback.py").write_text(
        "import torch._inductor.utils  # the real module enters the cycle the same way\n"
    )
    script = tmp_path / "race_child.py"
    script.write_text(_CHILD)
    env = dict(os.environ)
    env.pop("UNSLOTH_STUDIO_DISABLE_DYNAMO_IMPORT_GATE", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, str(script), str(_BACKEND), str(fake)],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
        cwd = str(tmp_path),
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, f"child produced no result (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_download_racing_the_warm_dynamo_import_leaves_torch_whole(tmp_path):
    result = _run(tmp_path)
    assert result == {
        "warm": True,
        "download": True,
        "download_error": "None",
        "after": True,
    }, result


def test_kill_switch_restores_the_ungated_download(tmp_path, monkeypatch):
    """With the gate disabled the download does not wait on the warm; it races as before."""
    from utils import torch_warmup

    monkeypatch.setenv(torch_warmup.DYNAMO_GATE_DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(torch_warmup, "_dynamo_done", False, raising = False)
    assert torch_warmup.gate_torch_stack_import("test") is False


def test_gate_gives_up_after_its_timeout(monkeypatch):
    """A wedged importer must not hang a load forever: the wait is bounded and logged."""
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_dynamo_done", False, raising = False)
    monkeypatch.delenv(torch_warmup.DYNAMO_GATE_DISABLE_ENV_VAR, raising = False)
    monkeypatch.setenv(torch_warmup.DYNAMO_GATE_TIMEOUT_ENV_VAR, "0.2")
    lines = []

    class _Log:
        def info(self, msg, *args):
            lines.append(("info", msg % args))

        def warning(self, msg, *args):
            lines.append(("warning", msg % args))

    assert torch_warmup._dynamo_lock.acquire(timeout = 5)
    try:
        assert torch_warmup.gate_torch_stack_import("image load", _Log()) is False
    finally:
        torch_warmup._dynamo_lock.release()
    assert [level for level, _ in lines] == ["info", "warning"], lines
    assert "image load: waiting for another thread" in lines[0][1]
    assert "gave up waiting" in lines[1][1]


def test_load_threads_gate_before_their_first_download():
    """Both media load threads take the gate before anything that can import unsloth_zoo."""
    for rel, first_download in (
        ("core/inference/diffusion.py", "self._prefetch_files("),
        ("core/inference/video.py", "_assert_pick_is_not_speech("),
    ):
        src = (_BACKEND / rel).read_text(encoding = "utf-8")
        start = src.index("    def _run_load(self, **kwargs: Any) -> None:")
        gate = src.index("gate_torch_stack_import(", start)
        assert gate < src.index(first_download, start), rel
