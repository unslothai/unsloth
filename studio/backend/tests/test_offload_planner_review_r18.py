# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unpriceable CPU drafter, draft CPU-MoE pass-through, the proactive ROCm arch gate, and a
projector device the seam cannot size."""

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


def _flag(
    cmd,
    name,
    default = None,
):
    return cmd[cmd.index(name) + 1] if name in cmd else default


# --- a CPU-pinned drafter the estimator cannot price -------------------------------------------


def test_an_unsized_cpu_drafter_is_not_planned():
    """The fit abstains on a CPU-pinned drafter it cannot price (mtp_unsized); the plan used to
    admit it as 0 bytes of host RAM and pin --load-mode none on that."""
    from test_offload_planner_seam import _inputs, _Stub

    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    assert _Stub()._planned_tensor_spill(_inputs(), env = env) is not None
    inputs = _inputs()
    inputs["cpu_draft_unsized"] = True
    assert _Stub()._planned_tensor_spill(inputs, env = env) is None


@pytest.mark.parametrize("draft_bytes, unsized", [(None, True), (0, False), (2 << 30, False)])
def test_the_launch_hands_the_unsized_cpu_drafter_verdict_to_the_seam(
    tmp_path, monkeypatch, draft_bytes, unsized
):
    from test_offload_planner_launch import _launch_with

    def launch_hook(backend):
        backend._cpu_resident_draft_bytes = lambda *_a, **_kw: draft_bytes

    import test_offload_planner_launch as launch_mod

    orig = launch_mod._backend

    def hooked(*a, **kw):
        backend, gguf = orig(*a, **kw)
        launch_hook(backend)
        return backend, gguf

    monkeypatch.setattr(launch_mod, "_backend", hooked)
    plan = Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _backend, seen = _launch_with(tmp_path, monkeypatch, plan)
    assert seen["inputs"]["cpu_draft_unsized"] is unsized
    # The known sizes still reach the host budget the plan admits against.
    assert seen["inputs"]["host_ram_unpriced_bytes"] >= int(draft_bytes or 0)


# --- draft CPU-MoE pass-through ----------------------------------------------------------------


@pytest.mark.parametrize(
    "env, extra_args",
    [
        ({}, ["-cmoed"]),
        ({}, ["--spec-draft-cpu-moe"]),
        ({}, ["--cpu-moe-draft"]),
        ({}, ["-ncmoed", "2"]),
        ({}, ["--spec-draft-n-cpu-moe=3"]),
        ({}, ["--spec-draft-ncmoe", "1"]),
        ({}, ["--n-cpu-moe-draft", "4"]),
        # Counts append: a later 0 does not take an earlier count back.
        ({}, ["-ncmoed", "2", "-ncmoed", "0"]),
        # std::stoi reads the leading digits; a value it throws on never runs the plan either.
        ({}, ["-ncmoed", "2x"]),
        ({}, ["-ncmoed", "x"]),
        ({"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "1"}, None),
        ({"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "true"}, None),
        ({"LLAMA_ARG_CPU_MOE_DRAFT": "on"}, None),
        ({"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "2"}, None),
        ({"LLAMA_ARG_N_CPU_MOE_DRAFT": "5"}, None),
        # An env count survives an argv 0: env first, argv on top, both append.
        ({"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "2"}, ["-ncmoed", "0"]),
    ],
    ids = [
        "cmoed",
        "spec_draft_cpu_moe",
        "cpu_moe_draft",
        "ncmoed",
        "spec_draft_n_cpu_moe_inline",
        "spec_draft_ncmoe",
        "n_cpu_moe_draft",
        "count_then_zero",
        "stoi_prefix",
        "unparseable",
        "env_spec_draft_cpu_moe",
        "env_spec_draft_cpu_moe_true",
        "env_legacy_cpu_moe_draft",
        "env_spec_draft_n_cpu_moe",
        "env_legacy_n_cpu_moe_draft",
        "env_count_argv_zero",
    ],
)
def test_draft_cpu_moe_placement_declines(env, extra_args):
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_seam import _inputs, _Stub

    full_env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert LlamaCppBackend._planner_may_run(extra_args, full_env) is False
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = extra_args, env = full_env) is None


@pytest.mark.parametrize(
    "env, extra_args",
    [
        ({}, ["-ncmoed", "0"]),
        ({}, ["--spec-draft-n-cpu-moe=0"]),
        ({"LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE": "0"}, None),
        ({"LLAMA_ARG_N_CPU_MOE_DRAFT": "0"}, None),
        ({"LLAMA_ARG_SPEC_DRAFT_CPU_MOE": "0"}, None),
        ({"LLAMA_ARG_CPU_MOE_DRAFT": "false"}, None),
    ],
    ids = ["ncmoed_zero", "inline_zero", "env_zero", "legacy_env_zero", "env_off", "legacy_env_off"],
)
def test_draft_cpu_moe_that_places_nothing_still_plans(env, extra_args):
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_seam import _inputs, _Stub

    full_env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert LlamaCppBackend._planner_may_run(extra_args, full_env)
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = extra_args, env = full_env) is not None


def test_the_main_model_cpu_moe_flags_are_not_the_draft_ones():
    """-cmoe / -ncmoe already decline through _args_place_tensors_on_cpu; the draft spellings
    are distinct tokens that path never matched."""
    from core.inference.llama_cpp import _args_place_tensors_on_cpu
    for args in (["-cmoed"], ["-ncmoed", "2"], ["--n-cpu-moe-draft", "2"]):
        assert _args_place_tensors_on_cpu(args) is False


# --- the proactive ROCm arch gate and the plan's --tensor-split ------------------------------

SURVIVORS = [1, 2]
GIB_MIB = 1024
TWO_CARDS = [(1, 4 * GIB_MIB, 4 * GIB_MIB), (2, 12 * GIB_MIB, 12 * GIB_MIB)]
PLAN = Plan(
    changed = True,
    n_ctx = 8192,
    ot_patterns = ("blk\\.1\\.ffn_.*",),
    spilled_blocks = (1,),
    device_layer_counts = (17, 48),
)


def _gated_launch(tmp_path, monkeypatch, *, survivors, **load_kwargs):
    """Two supported GPUs (1, 2) the planner sized a --tensor-split for; the proactive arch gate
    then pins the child to ``survivors`` (gpu_indices None: the fit arm, nothing user-pinned)."""
    import test_offload_planner_launch as launch_mod

    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    orig = launch_mod._backend

    def hooked(tmp, *, vulkan, memory):
        backend, gguf = orig(tmp, vulkan = vulkan, memory = TWO_CARDS)
        backend._arch_gate_survivors = lambda _binary = None: list(survivors)
        backend._visibility_mask_is_unmappable = lambda: False
        return backend, gguf

    monkeypatch.setattr(launch_mod, "_backend", hooked)
    return launch_mod._launch_with(tmp_path, monkeypatch, PLAN, **load_kwargs)


def test_the_planned_split_survives_the_proactive_gate_when_the_survivors_are_its_devices(
    tmp_path, monkeypatch
):
    """The planner's probe is already gated (_get_gpu_memory(for_llama_server = True)), so its
    split is positional over exactly the survivors the mask re-indexes from 0. Dropping it left
    --fit off standing with llama.cpp's free-VRAM split, the boundary the plan never checked."""
    cmd, backend, seen = _gated_launch(tmp_path, monkeypatch, survivors = SURVIVORS)
    assert [int(i) for i, _f in seen["inputs"]["gpus"]] == SURVIVORS, "harness: plan saw 1, 2"
    assert _flag(cmd, "--fit") == "off", cmd
    assert _flag(cmd, "--tensor-split") == "17,48", cmd
    assert backend._spill_plan_flags[-2:] == ["--tensor-split", "17,48"]


def test_a_gate_that_leaves_other_devices_than_the_plan_revokes_it(tmp_path, monkeypatch):
    """Survivors that are not the plan's devices: the split cannot be kept, and --fit off
    without it is a placement nothing priced. Hand it back to llama.cpp's fitter."""
    cmd, backend, _seen = _gated_launch(tmp_path, monkeypatch, survivors = [2])
    assert "--tensor-split" not in cmd, cmd
    assert _flag(cmd, "--fit") == "on", cmd
    assert "-ot" not in cmd, cmd


@pytest.mark.parametrize("spelling", ["argv", "env"])
def test_a_plan_riding_the_user_split_is_revoked_when_the_gate_drops_that_split(
    tmp_path, monkeypatch, spelling
):
    """A user ratio reaching the child makes the plan emit no split of its own: its rows were
    modelled against that ratio. The gate drops the ratio on any narrowing (the argv pair, and the
    env twin with _clear_split_placement_env), so the child splits the survivors by free VRAM and
    --fit off pins a placement nothing priced, even over the plan's own devices."""
    kwargs = {}
    if spelling == "argv":
        kwargs["extra_args"] = ["--tensor-split", "1,3"]
    else:
        monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "1,3")
    cmd, backend, seen = _gated_launch(tmp_path, monkeypatch, survivors = SURVIVORS, **kwargs)
    assert [int(i) for i, _f in seen["inputs"]["gpus"]] == SURVIVORS, "harness: plan saw 1, 2"
    assert "--tensor-split" not in backend._spill_plan_flags, "harness: the user's split rides"
    assert backend._spill_plan_flags, "harness: the plan was emitted"
    assert "--tensor-split" not in cmd, cmd
    assert _flag(cmd, "--fit") == "on", cmd
    assert "-ot" not in cmd, cmd


# --- a projector device of any value -----------------------------------------------------------


@pytest.mark.parametrize(
    "env, extra_args",
    [
        ({}, ["-mmdev", "cpu"]),
        ({}, ["--mmproj-device", "none"]),
        ({}, ["--mmproj-device="]),
        # -mmdev after --no-mmproj-offload sets mmproj_use_gpu back to true (arg.cpp handler).
        ({}, ["--no-mmproj-offload", "-mmdev", "CUDA0"]),
        ({"MTMD_BACKEND_DEVICE": "cpu"}, None),
        ({"MTMD_BACKEND_DEVICE": "none"}, None),
        ({"MTMD_BACKEND_DEVICE": ""}, None),
    ],
    ids = [
        "mmdev_cpu",
        "mmproj_none",
        "mmproj_empty",
        "offload_then_card",
        "env_cpu",
        "env_none",
        "env_empty",
    ],
)
def test_a_projector_device_of_any_value_declines(env, extra_args):
    """_resolved_mmproj_offload reads only --[no-]mmproj-offload, so a cpu/none projector is
    still charged to VRAM and Rung 0 may credit moving it off a card it never occupied."""
    from core.inference.llama_cpp import LlamaCppBackend
    from test_offload_planner_seam import _inputs, _Stub

    full_env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert LlamaCppBackend._planner_may_run(extra_args, full_env) is False
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = extra_args, env = full_env) is None


def test_no_projector_device_still_plans():
    from core.inference.llama_cpp import LlamaCppBackend
    assert LlamaCppBackend._planner_may_run(["--no-mmproj-offload"], {"UNSLOTH_SMART_OFFLOAD": "1"})
    assert LlamaCppBackend._planner_may_run(None, {"UNSLOTH_SMART_OFFLOAD": "1"})


# --- --moe-cache-mib precedence ----------------------------------------------------------------


@pytest.mark.parametrize(
    "env, extra_args, plans",
    [
        # An argv 0 overrides an inherited cache: nothing is allocated, the plan may run.
        ({"LLAMA_ARG_MOE_CACHE_MIB": "2048"}, ["--moe-cache-mib", "0"], True),
        ({"LLAMA_ARG_MOE_CACHE_MIB": "2048"}, ["--moe-cache-mib=0"], True),
        # An argv cache overrides an inherited 0.
        ({"LLAMA_ARG_MOE_CACHE_MIB": "0"}, ["--moe-cache-mib", "1024"], False),
        # An empty value is not a known "none", in argv or env.
        ({}, ["--moe-cache-mib="], False),
        ({"LLAMA_ARG_MOE_CACHE_MIB": "0"}, ["--moe-cache-mib="], False),
        ({"LLAMA_ARG_MOE_CACHE_MIB": ""}, None, False),
    ],
    ids = [
        "argv_zero_over_env",
        "argv_inline_zero_over_env",
        "argv_over_env_zero",
        "argv_empty",
        "argv_empty_over_env_zero",
        "env_empty",
    ],
)
def test_the_moe_cache_argv_value_wins_over_its_env_twin(env, extra_args, plans):
    from core.inference.llama_cpp import LlamaCppBackend
    full_env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert bool(LlamaCppBackend._planner_may_run(extra_args, full_env)) is plans
