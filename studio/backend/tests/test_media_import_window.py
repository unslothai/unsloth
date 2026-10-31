# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load must not race the post-warm worker's diffusers / peft import (half-built modules:
"cannot import name 'LoraLayer' from partially initialized module 'peft.tuners.lora'")."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    warning = info
    debug = info


@pytest.fixture
def warm(monkeypatch):
    """``utils.torch_warmup`` with the dynamo latch set and the media window unclaimed."""
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_dynamo_done", True)
    monkeypatch.setattr(torch_warmup, "_media_import_claimed", False)
    monkeypatch.setattr(torch_warmup, "_diffusers_prewarmed", False)
    monkeypatch.delenv(torch_warmup.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(torch_warmup.DISABLE_ENV_VAR, raising = False)
    return torch_warmup


def test_the_load_waits_for_a_background_import_in_flight(warm):
    """The reported race: the load blocks in close_dynamo_import_window until the import ends."""
    in_window = threading.Event()
    release = threading.Event()

    def _background():
        with warm.background_media_import() as window_open:
            assert window_open
            in_window.set()
            release.wait(10)

    bg = threading.Thread(target = _background)
    bg.start()
    assert in_window.wait(10)

    log = _Log()
    done = threading.Event()
    load = threading.Thread(target = lambda: (warm.close_dynamo_import_window(log), done.set()))
    load.start()
    try:
        assert not done.wait(0.5), "the load entered diffusers while a background import ran"
    finally:
        release.set()
    bg.join(10)
    assert done.wait(10)
    load.join(10)
    assert any("waiting for the background diffusers import" in line for line in log.lines)
    assert warm._media_import_claimed is True


def test_background_work_stands_down_once_a_load_claimed_the_window(warm):
    """The other order: the load got there first, so the background must not import at all."""
    assert warm.close_dynamo_import_window(_Log()) is True
    with warm.background_media_import() as window_open:
        assert window_open is False


def test_background_work_runs_while_no_load_has_claimed_it(warm):
    with warm.background_media_import() as window_open:
        assert window_open is True


def test_a_load_path_reached_from_background_work_does_not_wait_on_itself(warm):
    """The window lock is not reentrant: a background probe reaching a load helper passes through."""
    result = []

    def _background():
        with warm.background_media_import() as window_open:
            assert window_open
            result.append(warm.close_dynamo_import_window(_Log()))

    t = threading.Thread(target = _background, daemon = True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "background work deadlocked on its own window"
    assert result == [True]
    assert warm._media_import_claimed is False


def test_claiming_is_idempotent_and_cheap_once_claimed(warm):
    assert warm.claim_media_import_window() is True
    started = time.perf_counter()
    for _ in range(1000):
        assert warm.claim_media_import_window() is True
    assert time.perf_counter() - started < 1.0


def test_the_prewarm_skips_once_a_load_claimed_the_window(warm, monkeypatch):
    """After a load claimed the window the prewarm must not even consult its gate."""
    called = []
    monkeypatch.setattr(
        warm, "_a_local_model_would_load_through_diffusers", lambda: called.append(1) or True
    )
    warm.claim_media_import_window()
    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert called == []
    assert warm._diffusers_prewarmed is False


def test_the_prewarm_imports_inside_the_window(warm, monkeypatch):
    """A load arriving mid-prewarm must wait: the prewarm holds the window while it imports."""
    seen = {}

    def _gate():
        seen["locked"] = warm._media_import_lock.locked()
        return False  # stop before any real import

    monkeypatch.setattr(warm, "_a_local_model_would_load_through_diffusers", _gate)
    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert seen == {"locked": True}
    assert not warm._media_import_lock.locked()


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in main.py")


def _calls_inside_window(fn: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "background_media_import"
            for item in node.items
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                names.add(inner.func.id)
    return names


def test_the_post_warm_probes_import_inside_the_window():
    inside = _calls_inside_window(_function("_post_warm_background_work"))
    assert "_refresh_quantised_streaming_capability" in inside
    assert "_refresh_dense_quant_capability" in inside


def test_close_dynamo_import_window_claims_the_media_window():
    """Every media load path calls it first (pinned by test_dynamo_import_window.py)."""
    src = ast.get_source_segment(
        (_BACKEND / "utils" / "torch_warmup.py").read_text(encoding = "utf-8"),
        next(
            node
            for node in ast.parse(
                (_BACKEND / "utils" / "torch_warmup.py").read_text(encoding = "utf-8")
            ).body
            if isinstance(node, ast.FunctionDef) and node.name == "close_dynamo_import_window"
        ),
    )
    assert "claim_media_import_window(" in src
