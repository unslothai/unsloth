# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video loads hold the streamed expandable-segments engagement under their own owner."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

import core.inference.diffusion_decode_segments as seg

VIDEO_SRC = Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py"


@pytest.fixture
def allocator(monkeypatch):
    calls = []
    fake = types.SimpleNamespace()
    monkeypatch.setattr(seg, "_torch", lambda: fake)
    monkeypatch.setattr(seg, "_busy", lambda torch: False)
    monkeypatch.setattr(seg, "_decide", lambda torch: True)
    monkeypatch.setattr(seg, "_live_settings", lambda torch: "")
    monkeypatch.setattr(seg, "_setter", lambda torch: calls.append)
    monkeypatch.setattr(seg, "_usable", None)
    monkeypatch.setattr(seg, "_streamed", None)
    monkeypatch.setattr(seg, "_owners", set())
    monkeypatch.setattr(seg, "_depth", 0)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_STREAMED_EXPANDABLE", raising = False)
    monkeypatch.delenv("UNSLOTH_VIDEO_DENOISE_EXPANDABLE", raising = False)
    return calls


def test_the_last_owner_restores_the_settings(allocator):
    assert seg.engage_streamed()
    assert seg.engage_streamed(owner = "video")
    assert allocator == ["expandable_segments:True"]
    assert seg.release_streamed()
    assert allocator == ["expandable_segments:True"]
    assert not seg.release_streamed()
    assert seg.release_streamed(owner = "video")
    assert allocator == ["expandable_segments:True", "expandable_segments:False", ""]


def test_a_video_release_leaves_an_image_engagement(allocator):
    assert seg.engage_streamed()
    assert not seg.release_streamed(owner = "video")
    assert allocator == ["expandable_segments:True"]
    assert seg.release_streamed()
    assert allocator[-2:] == ["expandable_segments:False", ""]


def test_video_engages_on_cuda_only_and_honours_its_kill_switch(allocator, monkeypatch):
    from core.inference import video

    assert not video._engage_video_expandable(types.SimpleNamespace(is_cuda_torch_device = False))
    assert allocator == []
    monkeypatch.setenv("UNSLOTH_VIDEO_DENOISE_EXPANDABLE", "0")
    assert not video._engage_video_expandable(types.SimpleNamespace(is_cuda_torch_device = True))
    assert allocator == []
    monkeypatch.delenv("UNSLOTH_VIDEO_DENOISE_EXPANDABLE")
    assert video._engage_video_expandable(types.SimpleNamespace(is_cuda_torch_device = True))
    assert allocator == ["expandable_segments:True"]
    video._release_video_expandable()
    assert allocator[-2:] == ["expandable_segments:False", ""]


def _functions(name):
    tree = ast.parse(VIDEO_SRC.read_text(encoding = "utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]


def _calls(fn):
    return {
        c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }


def test_rollback_and_unload_release_the_video_engagement():
    for name in ("_rollback_precommit_globals", "_teardown_state_locked"):
        fns = _functions(name)
        assert fns, name
        assert all("_release_video_expandable" in _calls(f) for f in fns), name


def test_engaged_after_placement_before_graph_arming():
    src = VIDEO_SRC.read_text(encoding = "utf-8")
    engage = src.index("_engage_video_expandable(target)")
    assert (
        src.rindex("apply_memory_plan(", 0, engage)
        < engage
        < src.index("arm_graphs_after_placement(", engage)
    )
