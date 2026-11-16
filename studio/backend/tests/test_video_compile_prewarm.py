# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A compiled video load warms the compile workers and keys on a side thread, once per process. CPU only."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from core.inference import diffusion_bg_compile as bg
from core.inference import video

VIDEO = Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py"


@pytest.fixture
def calls(monkeypatch):
    seen: list = []
    done = threading.Event()

    def make(name, fail = False):
        def warm(logger = None):
            seen.append((name, threading.current_thread().name))
            if name == "inductor_key":
                done.set()
            if fail:
                raise RuntimeError("boom")
            return True

        return warm

    monkeypatch.setattr(video, "_COMPILE_PREWARM_STARTED", False)
    monkeypatch.delenv(bg.PREWARM_ENV, raising = False)
    monkeypatch.setattr(bg, "warm_compile_pool", make("pool", fail = True))
    monkeypatch.setattr(bg, "warm_triton_key", make("triton_key"))
    monkeypatch.setattr(bg, "warm_inductor_key", make("inductor_key"))
    return seen, done


def test_warms_pool_and_keys_off_the_caller_once(calls):
    seen, done = calls
    assert video._start_compile_prewarm() is True
    assert done.wait(10)
    # A failing step does not stop the next one.
    assert [n for n, _ in seen] == ["pool", "triton_key", "inductor_key"]
    assert all(t == "unsloth-video-compile-prewarm" for _, t in seen)
    assert video._start_compile_prewarm() is False
    assert len(seen) == 3


def test_kill_switch(calls, monkeypatch):
    seen, _ = calls
    monkeypatch.setenv(bg.PREWARM_ENV, "0")
    assert video._start_compile_prewarm() is False
    assert seen == []


def test_load_starts_it_only_for_a_compiled_cuda_load():
    tree = ast.parse(VIDEO.read_text(encoding = "utf-8"))
    gates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_start_compile_prewarm"
            for c in ast.walk(node)
        ):
            gates.append(ast.unparse(node.test))
    assert gates == ["'compiled' in speed_optims and getattr(target, 'device', None) == 'cuda'"]
