# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image backend's multi-GPU call sites, on the fake runtime of test_diffusion_backend, CPU only.

The helpers are unit-tested in test_diffusion_multigpu.py; these pin that a real load / generate / status reaches them:
the plan refinement and the CFG-parallel arming in load_pipeline, the guard's VAE-card tile decision and the CFG render
release in generate, and the ``multi_gpu`` status key."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from core.inference import diffusion as D

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_backend import _loaded_backend, fake_runtime  # noqa: E402,F401


def test_image_load_generate_and_status_reach_the_multi_gpu_layer(
    fake_runtime, tmp_path, monkeypatch
):
    calls = []
    report = {"primary": "cuda:0", "secondary": "cuda:1", "placement": {"text_encoder": "cuda:1"}}

    def refine(
        pipe,
        plan,
        logger = None,
    ):
        calls.append("refine")
        return plan

    def tile_for_verdict(
        pipe, verdict, guard_target, guard_kwargs, guard_batch, restore_vae, logger
    ):
        calls.append(("tile", verdict))
        return restore_vae

    monkeypatch.setattr(D, "refine_plan_for_multigpu", refine)
    monkeypatch.setattr(
        D,
        "install_cfg_parallel",
        lambda pipe, offload_policy = None, logger = None: calls.append("cfg"),
    )
    monkeypatch.setattr(D, "end_cfg_render", lambda pipe: calls.append(("end", pipe)))
    monkeypatch.setattr(D, "multigpu_placement_status", lambda pipe: report)
    verdict = types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)
    monkeypatch.setattr(D, "raise_on_image_activation_shortfall", lambda **kw: verdict)
    monkeypatch.setattr(D, "_tile_for_verdict", tile_for_verdict)

    backend = _loaded_backend(tmp_path)
    assert calls == ["refine", "cfg"]
    assert backend.status()["multi_gpu"] == report
    calls.clear()
    backend.generate(prompt = "a sloth", width = 512, height = 512, steps = 2)
    assert ("tile", verdict) in calls
    assert calls[-1:] == [("end", backend._state.pipe)]
    pipe = backend._state.pipe
    monkeypatch.setattr(D, "uninstall_cfg_parallel", lambda p: calls.append(("uninstall", p)))
    backend.unload()
    assert ("uninstall", pipe) in calls
