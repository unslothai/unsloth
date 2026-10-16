# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-device miss is a reason to keep searching, and the MoE context veto needs a fitter."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_cost_model import HostProfile  # noqa: E402
from core.inference.offload_planner import (  # noqa: E402
    Plan,
    ContextPolicy,
    PlanOptions,
    _Knobs,
    _outside_layout_bytes,
    _per_device_shortfall,
    plan_placement,
)
from test_offload_cost_gate import GIB, dense_layout, gated, moe_layout  # noqa: E402


def _long_moe():
    return replace(moe_layout(), kv_bytes_per_token_f16 = int(4.0 * GIB) // 32768, n_ctx_train = 65536)


def test_the_moe_context_veto_stands_down_when_no_fitter_can_replace_the_load():
    """A user -ts across two cards: common/fit.cpp aborts on it, so the veto's '--fit on at the
    context asked for' is the launch as typed. _cost_gate and _knob_only_gate already skip the
    ranking there; the context-search veto has to as well, or the only placement that fits is
    refused."""
    layout = _long_moe()
    devices = [10 * GIB, 10 * GIB]

    def run(**kw):
        return plan_placement(
            layout,
            devices,
            15 * GIB,
            65536,
            opts = gated(host = HostProfile(threads = 8), context_policy = ContextPolicy.FIT_ONLY, **kw),
        )

    ranked = run()
    assert ranked.veto and ranked.declined_by_gate and not ranked.changed, ranked.reason

    unranked = run(rank_against_fitter = False)
    assert unranked.changed and unranked.priced, unranked.reason
    assert not unranked.declined_by_gate and not unranked.veto, unranked.reason
    assert 0 < unranked.n_ctx < 65536 and unranked.spilled_blocks, unranked.reason


def _fits_on_every_card(layout, opts, devices, plan):
    moved = {}
    for i in plan.spilled_blocks:
        moved[i] = layout.blocks[i].spillable_bytes
    return (
        _per_device_shortfall(
            layout,
            opts,
            int(plan.n_ctx),
            moved,
            plan.spilled_lm_head,
            devices,
            quantised = plan.cache_type_k is not None,
            kv_bytes_floor = 0,
            split_weights_per_device = devices,
            kv_layer_weights = (),
            extra_on_device0 = _outside_layout_bytes(opts, _Knobs(max(1, opts.n_parallel))),
            n_seq = max(1, opts.n_parallel),
        )
        is None
    )


def test_a_per_device_miss_at_the_asked_context_still_searches_a_smaller_one():
    """Pooled VRAM holds the load at 32768, card 0's row share does not. FIT_ONLY may shrink, and
    the single-card path does (an unplaceable context is None there), so the multi-card path must
    not end the search on the first per-device miss."""
    layout = dense_layout()
    devices = [4 * GIB, 4 * GIB]
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY)
    plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = opts)
    assert plan.changed and plan.priced, plan.reason
    assert 0 < plan.n_ctx < 32768, plan.reason


def test_a_per_device_miss_in_f16_still_tries_the_quantised_cache():
    layout = dense_layout(kv_gib_at_32k = 4.0)
    devices = [int(4.8 * GIB), int(5.4 * GIB)]
    plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = PlanOptions(allow_kv_quant = True))
    assert plan.changed and plan.priced, plan.reason
    assert plan.cache_type_k == "q8_0" and plan.n_ctx == 32768, plan.reason


def test_a_per_device_miss_on_one_context_rung_still_walks_the_lower_ones():
    """The pooled ceiling is an upper bound, not a per-device fit: the first rung under it can
    overflow card 0 while the next one down places."""
    layout = replace(moe_layout(), kv_bytes_per_token_f16 = int(2.0 * GIB) // 32768)
    devices = [int(2.25 * GIB), int(2.25 * GIB)]
    opts = PlanOptions(context_policy = ContextPolicy.FIT_ONLY)
    plan = plan_placement(layout, devices, 200 * GIB, 32768, opts = opts)
    assert plan.changed and plan.priced, plan.reason
    assert 0 < plan.n_ctx < 7936, plan.reason


def test_the_resident_shortcut_is_checked_card_by_card():
    """PREFER_RESIDENT sized its context on the pooled budget; on two cards the default split can
    still overflow one of them, and a resident plan pins -ngl -1 --fit off."""
    layout = dense_layout(n_blocks = 16)
    devices = [int(4.6 * GIB), int(4.6 * GIB)]
    for extra in (0, 1 * GIB):
        opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT, extra_resident_bytes = extra)
        plan = plan_placement(layout, devices, 94 * GIB, 32768, opts = opts)
        if plan.changed:
            assert _fits_on_every_card(layout, opts, devices, plan), plan.reason


def test_the_resident_shortcut_is_unchanged_on_one_card():
    layout = dense_layout(n_blocks = 16)
    opts = PlanOptions(context_policy = ContextPolicy.PREFER_RESIDENT)
    plan = plan_placement(layout, [7 * GIB], 94 * GIB, 32768, opts = opts)
    assert plan.changed and not plan.spilled_blocks and 0 < plan.n_ctx < 32768, plan.reason
    assert "resident" in plan.reason


# --- revocation bookkeeping -------------------------------------------------------------------


def _launch_refused_then_ok(
    tmp_path,
    monkeypatch,
    plan,
    *,
    caps = None,
):
    """The planned launch exits refusing a unified cache with several sequences, a capability
    crash the startup-failure retry does not take, so the single-sequence rung revokes the plan."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent
    from test_offload_planner_launch import CARD_MIB, DENSE, MIB, _backend

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
        **(caps or {}),
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    backend._planned_tensor_spill = lambda inputs, **_kw: plan

    real_popen = subprocess.Popen
    cmds = []

    def planned(cmd):
        return "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "off"

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = 1 if planned(cmd) else None
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": rc,
                "poll": lambda self: rc,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        launched = cmds[-1] if cmds else []
        if planned(launched):
            backend._stdout_lines = [
                "llama_init_from_model: a unified KV cache is only supported with a single sequence"
            ]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                speculative_type = "off",
                n_parallel = 4,
            )
        )
    return cmds, backend


def _settings(monkeypatch, keep_resident, no_ram_reserve):
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (keep_resident, no_ram_reserve))
    monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)


def _satisfied(backend):
    from core.inference.llama_server_args import memory_state_satisfies_settings
    return memory_state_satisfies_settings(
        backend._memory_state,
        backend._memory_policy_active,
        backend._memory_mlock_applicable,
        backend._memory_direct_io,
        backend._memory_dio_applicable,
        bool(backend._memory_dio_flags),
    )


def test_a_revocation_records_the_load_mode_the_retry_runs_with(tmp_path, monkeypatch):
    """The plan's --load-mode none reserves a full host copy; the revocation strips it, so the
    record has to follow, or the next "Don't reserve system RAM" reloads a child that does not
    reserve anything."""
    from core.inference.llama_server_args import resolve_effective_load_state

    _settings(monkeypatch, False, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 2, load_mode_none = True)
    cmds, backend = _launch_refused_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert len(cmds) == 2, cmds
    assert "none" in cmds[0] and "none" not in cmds[-1], cmds
    mlock, reserves, direct = resolve_effective_load_state(cmds[-1], {})
    assert backend._memory_state == (mlock, reserves), (backend._memory_state, cmds[-1])
    _settings(monkeypatch, False, True)
    assert _satisfied(backend), "a mapped child already satisfies no-reserve"


def test_a_revoked_all_on_gpu_plan_leaves_the_page_lock_applicable(tmp_path, monkeypatch):
    """The plan kept every layer on the card, so the lock was recorded inapplicable. The --fit on
    child the revocation launches is the no-plan launch, which records it applicable: a later
    "Keep model in RAM" must reload with the lock, not be judged already satisfied."""
    from test_offload_planner_launch import _launch_crash_then_ok

    _settings(monkeypatch, False, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert cmds[-1][cmds[-1].index("--fit") + 1] == "on", cmds[-1]
    assert backend._memory_mlock_applicable is True
    _settings(monkeypatch, True, False)
    assert not _satisfied(backend), "keep-resident has to relaunch with the page-lock"


def test_a_revoked_all_on_gpu_plan_drops_the_dio_applicability(tmp_path, monkeypatch):
    """Windows: the all-on-GPU plan read as a confirmed full offload, so the no-reserve DirectIO
    was recorded applicable. The --fit on child is not that placement any more, the same reason
    the --fit on crash retry drops it."""
    from core.inference import llama_cpp
    from test_offload_planner_launch import _launch_crash_then_ok

    monkeypatch.setattr(
        llama_cpp,
        "no_reserve_requires_dio",
        lambda *, supports_load_mode, gpu_offload_confirmed: supports_load_mode
        and gpu_offload_confirmed,
    )
    import core.inference.llama_server_args as lsa

    monkeypatch.setattr(
        lsa,
        "no_reserve_requires_dio",
        lambda *, supports_load_mode, gpu_offload_confirmed: supports_load_mode
        and gpu_offload_confirmed,
    )
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_gpu_offload_confirmed", lambda self, *a, **k: True
    )
    seen = {}
    real = llama_cpp.LlamaCppBackend._record_memory_state

    def spy(
        self,
        argv,
        env = None,
    ):
        seen.setdefault("first_dio_applicable", self._memory_dio_applicable)
        return real(self, argv, env)

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_record_memory_state", spy)
    _settings(monkeypatch, False, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert seen.get("first_dio_applicable") is True, "harness: the plan launch read DIO applicable"
    assert cmds[-1][cmds[-1].index("--fit") + 1] == "on", cmds[-1]
    assert backend._memory_dio_applicable is False


# --- pass-through placement the plan cannot price ---------------------------------------------


@pytest.mark.parametrize(
    "env, extra_args",
    [
        # -otd .*=CPU: the drafter's weights go to host RAM while the seam charges them, whole,
        # on the GPU; a partial pattern moves an unknown share. Either way it is user placement.
        ({}, ["-otd", ".*=CPU"]),
        ({}, ["--override-tensor-draft", "ffn=CPU"]),
        ({}, ["--spec-draft-override-tensor=.*=CPU"]),
        # A projector on a named card: its bytes ride extra_resident_bytes, which
        # _per_device_shortfall books onto device 0, the same reason a pinned drafter declines.
        ({}, ["--mmproj-device", "CUDA1"]),
        ({}, ["-mmdev", "ROCm0"]),
        ({"MTMD_BACKEND_DEVICE": "CUDA1"}, None),
        # A user MoE expert cache is VRAM llama.cpp allocates on top of the plan, which pins
        # --fit off, so nothing re-fits the overflow.
        ({}, ["--moe-cache-mib", "4096"]),
        ({}, ["--moe-cache-mib=auto"]),
        ({"LLAMA_ARG_MOE_CACHE_MIB": "2048"}, None),
    ],
    ids = [
        "otd_cpu",
        "otd_long",
        "otd_spec_inline",
        "mmproj_device",
        "mmdev",
        "mtmd_env",
        "moe_cache",
        "moe_cache_auto_inline",
        "moe_cache_env",
    ],
)
def test_pass_through_placement_the_plan_cannot_price_declines(env, extra_args):
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_seam import _inputs, _Stub

    full_env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert LlamaCppBackend._planner_may_run(extra_args, full_env) is False
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = extra_args, env = full_env) is None


@pytest.mark.parametrize(
    "env, extra_args",
    [
        # A projector device of any value declines (test_offload_planner_review_r18): the seam
        # cannot size a cpu/none projector it would otherwise charge to VRAM.
        ({}, ["--moe-cache-mib", "0"]),
        ({"LLAMA_ARG_MOE_CACHE_MIB": "0"}, None),
    ],
    ids = ["moe_cache_zero", "moe_cache_env_zero"],
)
def test_placement_that_keeps_nothing_on_a_card_still_plans(env, extra_args):
    from core.inference.llama_cpp import LlamaCppBackend
    assert LlamaCppBackend._planner_may_run(extra_args, {"UNSLOTH_SMART_OFFLOAD": "1", **env})


def test_a_drafter_split_across_the_host_is_not_planned():
    """A positive -ngld below the drafter's block count leaves its tail and that tail's KV in host
    RAM while draft_bytes carries the drafter whole: the fit abstains on it as engaged but unsized,
    and a plan pinned with --fit off must as well."""
    from test_offload_planner_seam import _inputs, _Stub

    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    assert _Stub()._planned_tensor_spill(_inputs(), env = env) is not None
    inputs = _inputs()
    inputs["draft_split_across_host"] = True
    assert _Stub()._planned_tensor_spill(inputs, env = env) is None


def test_the_launch_hands_the_split_drafter_verdict_to_the_seam():
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    src = inspect.getsource(LlamaCppBackend.load_model)
    assert '_spill_inputs["draft_split_across_host"] = _draft_split_across_host' in src
