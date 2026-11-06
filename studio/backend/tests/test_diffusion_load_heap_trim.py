# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image load hands the heap it freed back to the OS (CPU only, no torch)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import core.inference.diffusion_memory as diffusion_memory
import pytest

BACKEND = Path(__file__).resolve().parents[1]


class _Log:
    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg, *args):
        self.infos.append(msg % args)

    def warning(self, msg, *args):
        self.warnings.append(msg % args)


def test_trim_calls_the_allocator_reclaimer_and_logs_it(monkeypatch):
    calls = []
    monkeypatch.delenv(diffusion_memory.LOAD_HEAP_TRIM_ENV, raising = False)
    monkeypatch.setattr(
        diffusion_memory, "_resolve_host_memory_reclaimer", lambda: lambda: calls.append(1)
    )
    log = _Log()
    assert diffusion_memory.trim_host_heap_after_load(log) is True
    assert calls == [1]
    assert any("freed heap" in line for line in log.infos)
    assert not log.warnings


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " OFF "])
def test_kill_switch_skips_the_trim(monkeypatch, value):
    monkeypatch.setenv(diffusion_memory.LOAD_HEAP_TRIM_ENV, value)
    monkeypatch.setattr(
        diffusion_memory,
        "_resolve_host_memory_reclaimer",
        lambda: pytest.fail("the kill switch must not reach the allocator"),
    )
    assert diffusion_memory.trim_host_heap_after_load(_Log()) is False


def test_no_allocator_api_is_a_quiet_no_op(monkeypatch):
    monkeypatch.delenv(diffusion_memory.LOAD_HEAP_TRIM_ENV, raising = False)
    monkeypatch.setattr(diffusion_memory, "_resolve_host_memory_reclaimer", lambda: None)
    log = _Log()
    assert diffusion_memory.trim_host_heap_after_load(log) is False
    assert not log.infos and not log.warnings


def test_a_failing_reclaimer_never_fails_the_load(monkeypatch):
    monkeypatch.delenv(diffusion_memory.LOAD_HEAP_TRIM_ENV, raising = False)

    def boom():
        raise OSError("_heapmin failed")

    monkeypatch.setattr(diffusion_memory, "_resolve_host_memory_reclaimer", lambda: boom)
    monkeypatch.setattr(diffusion_memory, "_load_heap_trim_warning_logged", False)
    log = _Log()
    assert diffusion_memory.trim_host_heap_after_load(log) is False
    assert log.warnings and "_heapmin failed" in log.warnings[0]
    # warned once per process, not once per load
    assert diffusion_memory.trim_host_heap_after_load(log) is False
    assert len(log.warnings) == 1
    assert diffusion_memory.trim_host_heap_after_load(None) is False


def test_load_pipeline_trims_after_the_load_time_pins():
    """The trim runs once the load's pins are done (their pageable sources are freed by then), on the success path."""
    tree = ast.parse((BACKEND / "core/inference/diffusion.py").read_text(encoding = "utf-8"))
    load = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_pipeline"
    )
    calls = [
        node.func.id
        for node in ast.walk(load)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "trim_host_heap_after_load" in calls
    lines = {
        node.func.id: node.lineno
        for node in ast.walk(load)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("finish_encoder_pins_at_load", "trim_host_heap_after_load")
    }
    assert lines["finish_encoder_pins_at_load"] < lines["trim_host_heap_after_load"]
    last = load.body[-1]
    assert isinstance(last, ast.Return)
    before_return = load.body[-2]
    assert (
        isinstance(before_return, ast.Expr)
        and isinstance(before_return.value, ast.Call)
        and getattr(before_return.value.func, "id", None) == "trim_host_heap_after_load"
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "reads RssAnon from procfs")
def test_trim_returns_fragmented_free_heap_to_the_os():
    """Real glibc: freed chunks between live ones stay in the heap until a trim."""
    import ctypes

    if not hasattr(ctypes.CDLL(None), "malloc_trim"):
        pytest.skip(
            reason = "musl and other libcs without malloc_trim: the resolver declines, nothing to trim"
        )
    script = textwrap.dedent(
        """
        import json, logging
        import core.inference.diffusion_memory as dm

        def rss_anon():
            for line in open("/proc/self/status"):
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) >> 10

        chunks = [bytearray(b"x" * (64 << 10)) for _ in range(6000)]  # ~375 MiB of 64 KiB heap chunks
        keep = chunks[::8]  # live chunks pin the heap, so free() alone cannot shrink it
        del chunks
        before = rss_anon()
        ok = dm.trim_host_heap_after_load(logging.getLogger("t"))
        after = rss_anon()
        print(json.dumps({"ok": ok, "before": before, "after": after, "kept": len(keep)}))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != diffusion_memory.LOAD_HEAP_TRIM_ENV}
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("MALLOC_TRIM_THRESHOLD_", None)
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output = True,
        text = True,
        cwd = str(BACKEND),
        env = env,
        timeout = 120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    import json

    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["ok"] is True
    assert res["before"] - res["after"] >= 200, res
