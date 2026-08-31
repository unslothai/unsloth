# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
from pathlib import Path


def test_sandbox_warm_probe_thread_start_failure_is_best_effort(monkeypatch):
    from core.inference import sandbox

    messages: list[str] = []

    class StartFails:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread limit reached")

    monkeypatch.setattr(sandbox.threading, "Thread", StartFails)
    monkeypatch.setattr(
        sandbox.logger,
        "debug",
        lambda message, exc: messages.append(message % exc),
    )

    sandbox.start_sandbox_probe()

    assert messages == ["sandbox availability probe could not start: thread limit reached"]


def test_shared_lifespan_starts_sandbox_probe():
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(main_path.read_text(encoding = "utf-8"))
    lifespan = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "start_sandbox_probe"
        for node in ast.walk(lifespan)
    )
