# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live previews do no work unless a visible page polls the render's progress for them (CPU only)."""

from __future__ import annotations

import ast
import pathlib

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_preview as DP

BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _Clock:
    def __init__(self, t = 1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def previewer(monkeypatch):
    """A real previewer built on CPU (pinned memory and CUDA events stubbed), its step body counted."""
    monkeypatch.setattr(DP, "_last_watch", {})
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory = False, **k: real_empty(*a, **k))
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **k: object())
    monkeypatch.setattr(DP, "_device_map", lambda spec, device: (None, None))
    made = []

    def make(watch, clock):
        spec = DP.LatentRGB(
            layout = "bchw", down = 8, patch = 1, weight = ((0.1, 0.2, 0.3),) * 4, bias = (0.0,) * 3
        )
        p = DP.LatentPreviewer(
            spec,
            height = 64,
            width = 64,
            device = "cpu",
            publish = lambda *_: None,
            clock = clock,
            watch = watch,
        )
        p.calls = 0

        def body(*_a, **_k):
            p.calls += 1

        p._on_step = body
        made.append(p)
        return p

    yield make
    for p in made:
        p.finish()


def test_unwatched_render_never_reaches_the_step_body(previewer):
    clock = _Clock()
    p = previewer("video", clock)
    p._prev, p._prev_index = object(), 3
    for _ in range(10):
        clock.t += 1.0
        p.on_step(torch.zeros(1))
    assert p.calls == 0
    assert p.skipped_unwatched == 10
    assert p._prev is None and p._prev_index is None


def test_a_watcher_opening_mid_render_resumes_and_leaving_stops(previewer):
    clock = _Clock()
    p = previewer("image", clock)
    clock.t += 1.0
    p.on_step(torch.zeros(1))
    assert p.calls == 0
    DP.note_preview_watch("image", now = clock.t)
    p.on_step(torch.zeros(1))
    clock.t += DP.WATCH_WINDOW_S - 0.1
    p.on_step(torch.zeros(1))
    assert p.calls == 2
    clock.t += 0.2
    p.on_step(torch.zeros(1))
    assert p.calls == 2


def test_a_poll_for_another_kind_or_before_this_render_is_not_a_watcher(previewer):
    clock = _Clock()
    DP.note_preview_watch("video", now = clock.t - 0.5)
    p = previewer("video", clock)
    DP.note_preview_watch("image", now = clock.t)
    p.on_step(torch.zeros(1))
    assert p.calls == 0
    DP.note_preview_watch("video", now = clock.t)
    p.on_step(torch.zeros(1))
    assert p.calls == 1


def test_no_watch_kind_keeps_the_old_always_on_behaviour(previewer):
    p = previewer(None, _Clock())
    p.on_step(torch.zeros(1))
    assert p.calls == 1 and p.skipped_unwatched == 0


def test_explicit_false_wins_over_any_watcher(monkeypatch):
    monkeypatch.setattr(DP, "_last_watch", {})
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    DP.note_preview_watch("image")
    DP.note_preview_watch("video")
    for kind in ("image", "video"):
        assert (
            DP.LatentPreviewer.create(
                family = "flux.1",
                requested = False,
                height = 64,
                width = 64,
                device = "cuda",
                publish = lambda *_: None,
                watch = kind,
            )
            is None
        )


def _create_calls(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "LatentPreviewer"
        ):
            yield {kw.arg: kw.value for kw in node.keywords}


@pytest.mark.parametrize(
    "rel, kind", [("core/inference/diffusion.py", "image"), ("core/inference/video.py", "video")]
)
def test_every_engine_previewer_is_gated_on_its_progress_poll(rel, kind):
    calls = list(_create_calls(BACKEND / rel))
    assert calls, rel
    for kw in calls:
        assert isinstance(kw.get("watch"), ast.Constant) and kw["watch"].value == kind, rel


def test_headless_paths_still_never_build_a_previewer():
    """API, chat tool and MCP renders pass live_preview=False, untouched by the watcher gate."""
    video_src = (BACKEND / "routes/video.py").read_text(encoding = "utf-8")
    assert "live_preview = False," in video_src
    inference_src = (BACKEND / "routes/inference.py").read_text(encoding = "utf-8")
    assert inference_src.count("_live_preview_kwarg(backend.generate, False)") >= 3
