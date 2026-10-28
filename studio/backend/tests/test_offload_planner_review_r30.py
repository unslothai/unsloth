# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pass-through adapter is charged to host RAM as well as VRAM.

A LoRA tensor is allocated on its base tensor's buffer (llama-adapter.cpp:335), so the adapter
rows of a spilled block land in host RAM. Which blocks spill is only known once the plan is made,
so the seam takes the adapter off both pools.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference import offload_planner  # noqa: E402
from core.inference.offload_planner import Plan  # noqa: E402
from test_offload_planner_seam import GIB, _inputs, _plan, _Stub, sparse_adapter  # noqa: E402, F401


def test_the_adapter_comes_off_the_host_ram_pool(sparse_adapter, monkeypatch):
    seen = []

    def capture(layout, vram, host_ram, n_ctx, **kwargs):
        seen.append(host_ram)
        return Plan(reason = "captured")

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    stub = _Stub(avail_mib = 32 * 1024)
    stub._planned_tensor_spill(_inputs(free_mib = 17 * 1024), env = env)
    stub._planned_tensor_spill(
        _inputs(free_mib = 17 * 1024), extra_args = ["--lora", str(sparse_adapter)], env = env
    )
    assert seen == [32 * GIB, 29 * GIB]


def test_a_spill_cannot_admit_host_weights_against_ram_the_adapter_uses(sparse_adapter):
    """With 6 GiB of host RAM the 3 GiB adapter tips the spill over; it used to be planned with
    --load-mode none anyway."""
    stub = _Stub(avail_mib = 6 * 1024)
    bare = _plan(stub, free_mib = 17 * 1024)
    assert bare is not None and not bare.spills_anything, "the base load fits on its own"
    with_lora = _plan(stub, free_mib = 17 * 1024, extra_args = ["--lora", str(sparse_adapter)])
    assert with_lora is not None and not with_lora.spills_anything, with_lora.reason

    roomy = _plan(
        _Stub(avail_mib = 9 * 1024), free_mib = 17 * 1024, extra_args = ["--lora", str(sparse_adapter)]
    )
    assert roomy is not None and roomy.spills_anything, "not vacuous: with room it still spills"
