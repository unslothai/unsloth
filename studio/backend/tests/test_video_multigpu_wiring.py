# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The video backend's multi-GPU call sites, on the fake runtime of test_video_backend, CPU only.

The helpers are unit-tested in test_diffusion_multigpu.py; these pin that a real load / generate / status reaches them:
the plan refinement, the CFG-parallel arming and the per-card floors of a split in load_pipeline, the CFG render
release in generate, and the ``multi_gpu`` status key."""

from __future__ import annotations

import sys
from pathlib import Path

from core.inference import video as V

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_video_backend import _load_gguf, fake_runtime  # noqa: E402,F401


def test_video_load_generate_and_status_reach_the_multi_gpu_layer(
    fake_runtime, tmp_path, monkeypatch
):
    calls = []
    report = {"primary": "cuda:0", "secondary": "cuda:1", "placement": {"text_encoder": "cuda:1"}}
    floors = ((0, 9000, False), (1, 7000, True))

    def refine(
        pipe,
        plan,
        logger = None,
    ):
        calls.append("refine")
        return plan

    monkeypatch.setattr(V, "refine_plan_for_multigpu", refine)
    monkeypatch.setattr(
        V,
        "install_cfg_parallel",
        lambda pipe, offload_policy = None, logger = None: calls.append("cfg"),
    )
    monkeypatch.setattr(V, "end_cfg_render", lambda pipe: calls.append(("end", pipe)))
    monkeypatch.setattr(V, "multigpu_placement_status", lambda pipe: report)
    monkeypatch.setattr(V, "_video_split_card_floors", lambda pipe: floors)
    sidecar_checked = []
    single_card_for_sidecar = V._video_single_card_for_sidecar

    def sidecar_check(
        fam,
        plan,
        logger = None,
    ):
        sidecar_checked.append(fam.name)
        return single_card_for_sidecar(fam, plan, logger)

    monkeypatch.setattr(V, "_video_single_card_for_sidecar", sidecar_check)

    backend = V.VideoBackend()
    status = _load_gguf(backend, tmp_path)
    assert calls == ["refine", "cfg"]
    assert sidecar_checked == ["ltx-2", "ltx-2"]
    assert status["multi_gpu"] == report
    assert backend._state.vram_split_cards == floors
    calls.clear()
    backend.generate(prompt = "a sloth surfing", width = 512, height = 512, num_frames = 9, fps = 24)
    assert calls[-1:] == [("end", backend._state.pipe)]
    pipe = backend._state.pipe
    monkeypatch.setattr(V, "uninstall_cfg_parallel", lambda p: calls.append(("uninstall", p)))
    backend.unload()
    assert ("uninstall", pipe) in calls
