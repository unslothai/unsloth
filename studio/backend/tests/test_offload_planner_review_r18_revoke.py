# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A revoked spill plan stays revoked the same way on every argv derived from the planned one."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_planner import Plan  # noqa: E402
from test_offload_planner_review_r17 import _settings  # noqa: E402

_MTP_FAIL = "failed to create mtp context"


def _fit(cmd):
    return cmd[cmd.index("--fit") + 1] if "--fit" in cmd else None


def _launch_planned_mtp(
    tmp_path,
    monkeypatch,
    plan,
    *,
    caps = None,
    **load_kwargs,
):
    """The planned launch crashes at startup; the --fit on retry inside the same spawn still
    carries the drafter and fails on it; the drafterless retry, built from the planned argv,
    comes up. Returns every launched argv and the backend."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent
    from test_offload_planner_launch import CARD_MIB, DENSE, MIB, _backend

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)
        backend._nextn_predict_layers = 1

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        **(caps or {}),
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    backend._planned_tensor_spill = lambda inputs, **_kw: plan

    real_popen = subprocess.Popen
    cmds = []

    def crashed(cmd):
        return _fit(cmd) == "off" and "draft-mtp" in cmd

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = -11 if crashed(cmd) else (1 if "draft-mtp" in cmd else None)
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
        if "draft-mtp" in launched:
            backend._stdout_lines = [_MTP_FAIL]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                speculative_type = "mtp",
                n_parallel = 4,
                **load_kwargs,
            )
        )
    return cmds, backend


def _dio_host(monkeypatch):
    """Windows no-reserve: an all-on-GPU plan reads as a confirmed full offload, so the launch
    carries the managed DirectIO."""
    import core.inference.llama_server_args as lsa
    from core.inference import llama_cpp

    def requires(*, supports_load_mode, gpu_offload_confirmed):
        return supports_load_mode and gpu_offload_confirmed

    monkeypatch.setattr(llama_cpp, "no_reserve_requires_dio", requires)
    monkeypatch.setattr(lsa, "no_reserve_requires_dio", requires)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_gpu_offload_confirmed", lambda self, *a, **k: True
    )


def _count(cmd, run):
    return sum(1 for i in range(len(cmd) - len(run) + 1) if cmd[i : i + len(run)] == run)


def test_the_outer_retry_re_arms_the_page_lock_an_inner_revocation_already_armed(
    tmp_path, monkeypatch
):
    """Keep model in RAM: the inner --fit on retry re-armed the lock on its own argv. The
    drafterless retry is built from the planned argv, which never had it, and the revocation there
    read the cleared kept-every-layer bit (and a host-resident flag already set) and skipped it."""
    from core.inference.llama_server_args import resolve_effective_load_state
    from test_offload_planner_review_r17 import _satisfied

    _settings(monkeypatch, True, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_planned_mtp(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert len(cmds) == 3, [c[3:] for c in cmds]
    assert _fit(cmds[0]) == "off" and "draft-mtp" in cmds[1], "harness: plan, inner, drafterless"
    final = cmds[-1]
    assert _fit(final) == "on" and "--spec-default" in final, final
    assert _count(final, ["--load-mode", "mmap+mlock"]) == 1, final
    mlock, reserves, _direct = resolve_effective_load_state(final, {})
    assert backend._memory_state == (mlock, reserves), (backend._memory_state, final)
    assert _satisfied(backend)


def test_the_outer_retry_drops_the_plan_s_direct_io(tmp_path, monkeypatch):
    """Don't reserve system RAM: the all-on-GPU plan's DirectIO went from the inner retry's argv
    only; the drafterless retry built from the planned argv kept it under --fit on."""
    _dio_host(monkeypatch)
    _settings(monkeypatch, False, True)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_planned_mtp(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert "dio" in cmds[0], "harness: the planned launch carries the managed DirectIO"
    final = cmds[-1]
    assert _fit(final) == "on" and "--spec-default" in final, final
    assert "dio" not in final, final
    assert backend._memory_dio_applicable is False


def test_the_outer_retry_strips_the_plan_s_checkpoint_pair(tmp_path, monkeypatch):
    """The checkpoint cap is per slot. The inner revocation swapped the plan's pair for the
    restored count's and remembered only that one, so the outer revocation could not find the
    plan's pair in the argv built from the planned launch and appended a second one."""
    _settings(monkeypatch, False, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)

    import test_offload_planner_launch as launch_mod

    orig = launch_mod._backend

    def hooked(*a, **kw):
        backend, gguf = orig(*a, **kw)
        backend._bounded_ctx_checkpoints = lambda n_parallel, *_a, **_kw: 32 // max(
            1, int(n_parallel)
        )
        return backend, gguf

    monkeypatch.setattr(launch_mod, "_backend", hooked)
    cmds, _backend = _launch_planned_mtp(
        tmp_path, monkeypatch, plan, caps = {"ctx_checkpoints_flag": "--ctx-checkpoints"}
    )
    assert _count(cmds[0], ["--ctx-checkpoints", "32"]) == 1, "harness: the plan's 1-slot pair"
    final = cmds[-1]
    assert final[final.index("--parallel") + 1] == "4", final
    assert final.count("--ctx-checkpoints") == 1, final
    assert _count(final, ["--ctx-checkpoints", "8"]) == 1, final


def test_a_single_revocation_is_unchanged(tmp_path, monkeypatch):
    """The ordinary crash-then-fit-on path: one revocation, one lock, the plan's pair swapped."""
    from test_offload_planner_launch import _launch_crash_then_ok

    _settings(monkeypatch, True, False)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, backend = _launch_crash_then_ok(
        tmp_path, monkeypatch, plan, caps = {"supports_load_mode": True}
    )
    assert _fit(cmds[-1]) == "on"
    assert _count(cmds[-1], ["--load-mode", "mmap+mlock"]) == 1, cmds[-1]
    assert backend._memory_mlock_applicable is True


def test_the_plan_is_revoked_on_cmd_before_any_recovery_rung_derives_from_it(tmp_path, monkeypatch):
    """One revocation point for the recovery rungs: once the first spawn fails, `cmd` itself is
    revoked before the drafterless (and projector, arch-crash) rungs re-price slots, context,
    checkpoints and the memory policy off it. Before, the outer revocation ran only at the rung's
    own spawn, on the argv it had already derived from the planned `cmd`."""
    from core.inference import llama_cpp

    _settings(monkeypatch, False, False)
    seen = []
    real = llama_cpp.LlamaCppBackend._drop_tensor_spill

    def spy(self, run_cmd, why):
        out = real(self, run_cmd, why)
        if out is not run_cmd:
            seen.append((why, "draft-mtp" in run_cmd, "--spec-default" in run_cmd))
        return out

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_drop_tensor_spill", spy)
    plan = Plan(changed = True, n_ctx = 8192, n_parallel = 1)
    cmds, _backend = _launch_planned_mtp(tmp_path, monkeypatch, plan)
    assert _fit(cmds[-1]) == "on" and "--spec-default" in cmds[-1], cmds[-1]
    # The inner startup-failure retry, then `cmd` itself, still carrying the drafter: the
    # drafterless argv is built from an already-revoked command, so its spawn revokes nothing.
    assert seen == [("startup failure", True, False), ("recovery", True, False)], seen
