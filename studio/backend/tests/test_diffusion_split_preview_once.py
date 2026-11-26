# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Both split previews share one refinement; the image load re-asks after a LoRA bake (CPU, fake refine)."""

from __future__ import annotations

import ast
import inspect

from core.inference import diffusion as diff
from core.inference import diffusion_memory as dm


class _Plan:
    offload_policy = "group"


def _preview(**fields):
    base = {"layer_split": None, "secondary_ordinal": 1, "offload_policy": dm.OFFLOAD_NONE}
    return type("P", (), {**base, **fields})()


def test_split_previews_share_one_refinement(monkeypatch):
    calls = []
    preview = _preview()

    def refine(pipe, plan, logger):
        calls.append((pipe, plan, logger))
        return preview

    monkeypatch.setattr(diff, "refine_plan_for_multigpu", refine)
    pipe, plan = object(), _Plan()
    shared = diff._SplitPreviewRefine()
    assert diff._layer_split_expected(pipe, plan, shared) is False
    assert diff._companion_split_expected(pipe, plan, shared) is True
    assert calls == [(pipe, plan, None)]
    # after a LoRA bake (clear) and without the shared refinement the previews ask again
    shared.clear()
    assert diff._companion_split_expected(pipe, plan, shared) is True
    assert diff._layer_split_expected(pipe, plan) is False
    assert diff._companion_split_expected(pipe, plan) is True
    assert len(calls) == 4
    # another plan is another question
    diff._companion_split_expected(pipe, _Plan(), shared)
    assert len(calls) == 5


def test_a_wrapped_refinement_is_the_one_asked(monkeypatch):
    monkeypatch.setattr(
        diff, "refine_plan_for_multigpu", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    calls = []

    def video_refine(pipe, plan, logger):
        calls.append(1)
        return _preview(layer_split = object(), secondary_ordinal = None)

    shared = diff._SplitPreviewRefine(video_refine)
    pipe, plan = object(), _Plan()
    assert diff._layer_split_expected(pipe, plan, shared) is True
    assert diff._companion_split_expected(pipe, plan, shared) is False
    assert calls == [1]


def test_a_raising_preview_is_asked_again(monkeypatch):
    outcomes = [RuntimeError("transient"), _preview(offload_policy = "none")]

    def refine(pipe, plan, logger):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(diff, "refine_plan_for_multigpu", refine)
    shared = diff._SplitPreviewRefine()
    pipe, plan = object(), _Plan()
    assert diff._layer_split_expected(pipe, plan, shared) is False
    assert diff._companion_split_expected(pipe, plan, shared) is True
    assert not outcomes


def test_image_load_asks_again_after_a_lora_bake():
    tree = ast.parse(inspect.getsource(diff))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "comfy_compile" in test and "_has_active_lora" in test:
            body = [ast.unparse(s) for s in node.body]
            if any("_resolve_lora_set" in b for b in body):
                assert body[0] == "split_refine.clear()", body[0]
                found = True
    assert found
    source = ast.unparse(tree)
    assert "split_expected = _layer_split_expected(pipe, plan, split_refine)" in source
    assert "_companion_split_expected(pipe, plan, split_refine)" in source


def test_video_load_shares_one_preview():
    from core.inference import video

    source = ast.unparse(ast.parse(inspect.getsource(video)))
    assert "split_refine = _SplitPreviewRefine(video_refine)" in source
    assert "split_expected = _layer_split_expected(pipe, plan, split_refine)" in source
    assert "_companion_split_expected(pipe, plan, split_refine)" in source
