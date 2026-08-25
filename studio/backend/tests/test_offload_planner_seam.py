# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-path seam for the tensor-spill planner.

Two halves, and the negative half is the important one: on every path where the
planner does not produce a plan, the argv has to be exactly what it was before
this existed. The positive half pins the flags a plan emits, above all that it
carries ``--fit off`` (a running fitter would re-place layers on top of the plan
and put the KV cache back on the host) and ``-ngl -1`` (which keeps every layer
assigned to a GPU so the cache stays with it).
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.offload_layout import LM_HEAD_PATTERN, BlockLayout, ModelLayout
from core.inference.offload_planner import Plan, plan_placement, smart_offload_enabled

# The planner's own decision table is covered exhaustively in
# test_offload_planner.py. What belongs HERE is the seam: whether the launch
# path declines when it should, emits the tokens llama-server actually parses,
# and takes the plan back out on every retry.
FFN_SPILL_PATTERN = r"blk\.\d+\.ffn_"
LM_HEAD_SPILL_PATTERN = LM_HEAD_PATTERN

GIB = 1024**3
MIB = 1024 * 1024


class _Stub:
    """Only what the seam touches."""

    _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024
    _HOST_RAM_HEADROOM_MIB = 2048

    def __init__(
        self,
        avail_mib = 64 * 1024,
        ffn = 10 * GIB,
        lm_head = 1 * GIB,
        moe = 0,
    ):
        self._avail_mib = avail_mib
        self._ffn_weight_bytes = ffn
        self._lm_head_bytes = lm_head
        self.n_moe_layers = moe
        self._spill_plan_flags: list[str] = []

    def _available_system_memory_mib(self):
        return self._avail_mib

    def _can_estimate_kv(self):
        return True

    _planned_tensor_spill = LlamaCppBackend._planned_tensor_spill
    _drop_tensor_spill = LlamaCppBackend._drop_tensor_spill

    def _tensor_spill_layout(self, model_path):
        """Stand in for the GGUF read: the seam's job is to decline or to hand
        the planner well-formed inputs, not to parse a file. ``ffn = None``
        stands for an unreadable model, which must abstain."""
        if self._ffn_weight_bytes is None or not model_path:
            return None
        n = 64
        return ModelLayout(
            arch = "qwen35moe" if self.n_moe_layers else "qwen35",
            n_layers = n,
            n_attention_layers = 16,
            blocks = tuple(
                BlockLayout(
                    index = i,
                    spillable_bytes = self._ffn_weight_bytes // n,
                    resident_bytes = (2 * GIB) // n,
                )
                for i in range(n)
            ),
            lm_head_bytes = self._lm_head_bytes or 0,
            token_embd_bytes = 512 * MIB,
            kv_bytes_per_token_f16 = 65536,
            n_ctx_train = 262144,
            is_moe = bool(self.n_moe_layers),
            n_expert = 256 if self.n_moe_layers else 0,
            n_expert_used = 8 if self.n_moe_layers else 0,
            complete = True,
        )


def _inputs(
    model_size = 30 * GIB,
    kv = 2 * GIB,
    free_mib = 24 * 1024,
    indices = None,
):
    return {
        "model_size": model_size,
        "kv_cache_bytes": kv,
        "gpus": [(0, free_mib)],
        "gpu_indices": indices,
        "soft_overhead": 0,
        "model_path": "/models/stub.gguf",
        "n_ctx": 32768,
        "shared_gpu_ids": set(),
    }


def _plan(
    stub,
    env = None,
    extra_args = None,
    **kw,
):
    return stub._planned_tensor_spill(
        _inputs(**kw),
        extra_args = extra_args,
        env = {"UNSLOTH_SMART_OFFLOAD": "1", **(env or {})},
    )


# ------------------------------------------------------------------ the gate


def test_the_planner_is_off_by_default():
    """No env, no plan. An old install cannot change behaviour by upgrading."""
    assert smart_offload_enabled({}) is False
    assert _Stub()._planned_tensor_spill(_inputs(), env = {}) is None


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "enabled", "ON", " 1 "])
def test_the_gate_accepts_the_usual_spellings(value):
    assert smart_offload_enabled({"UNSLOTH_SMART_OFFLOAD": value}) is True


@pytest.mark.parametrize("value", ["0", "off", "", "no", "nope"])
def test_the_gate_rejects_everything_else(value):
    assert smart_offload_enabled({"UNSLOTH_SMART_OFFLOAD": value}) is False


# ------------------------------------------------- abstain = today's behaviour


def test_an_unpriced_placement_abstains():
    """The except arm restores use_fit without pricing anything, so the seam
    reads None here rather than an UnboundLocal kv_cache_bytes."""
    assert _Stub()._planned_tensor_spill(None, env = {"UNSLOTH_SMART_OFFLOAD": "1"}) is None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-ot", ".ffn_.*=CPU"],  # user already moves tensors
        ["--override-tensor", "x=CPU"],
        ["--cpu-moe"],
        ["--n-cpu-moe", "24"],
        ["-ngl", "12"],  # user owns the layer count
        ["--gpu-layers", "0"],
        ["--device", "CUDA0"],  # user owns the device set
        ["-dev", "none"],
        ["--fit", "on"],  # user asked for the fitter
        # --fit OFF too: a retry revokes the plan by appending "--fit on", and
        # extras land before that, so planning here would overturn the user.
        ["--fit", "off"],
        ["--fit=off"],
        ["-fit", "off"],
    ],
)
def test_a_pass_through_placement_override_declines(extra_args):
    """Pricing a placement the child will not get is the bug class behind most of
    the load-mode review traffic. Decline instead of modelling the override."""
    assert _plan(_Stub(), extra_args = extra_args) is None


@pytest.mark.parametrize(
    "env",
    [
        {"LLAMA_ARG_N_GPU_LAYERS": "12"},
        {"LLAMA_ARG_DEVICE": "none"},
        {"LLAMA_ARG_OVERRIDE_TENSOR": ".ffn_.*=CPU"},
        {"LLAMA_ARG_CPU_MOE": "1"},
    ],
)
def test_an_inherited_placement_env_declines(env):
    """The child inherits these, so they outlive any flag stripping."""
    assert _plan(_Stub(), env = env) is None


def test_an_unsized_model_abstains():
    assert _plan(_Stub(), model_size = 0) is None


def test_an_unsized_cache_abstains():
    assert _plan(_Stub(), kv = 0) is None


def test_no_gpus_abstains():
    stub = _Stub()
    assert (
        stub._planned_tensor_spill({**_inputs(), "gpus": []}, env = {"UNSLOTH_SMART_OFFLOAD": "1"})
        is None
    )


def test_an_absent_layout_abstains():
    """_ffn_weight_bytes is the planner-core half and does not exist yet, so the
    feature is inert until it lands even with the gate on."""
    assert _plan(_Stub(ffn = None)) is None


def test_unreadable_host_ram_abstains():
    assert _plan(_Stub(avail_mib = None)) is None


# ------------------------------------------------------------ the plan itself


def test_a_load_that_needs_ffn_spilled_gets_the_ffn_pattern():
    got = _plan(_Stub(), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 12 * 1024)
    assert got is not None
    assert got.ot_patterns and all("ffn" in p for p in got.ot_patterns)


def test_a_moe_load_spills_expert_tensors_not_dense_ffn():
    got = _plan(_Stub(moe = 40), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 12 * 1024)
    assert got is not None
    assert got.ot_patterns and all("ffn" in p for p in got.ot_patterns)


def test_lm_head_is_only_spilled_after_ffn():
    """43% of generation on its own, 16% on top of an already host-bound step, so
    it is never the first rung."""
    tight = _plan(_Stub(), model_size = 60 * GIB, kv = 2 * GIB, free_mib = 5632)
    assert tight is not None
    assert tight.spilled_lm_head is True
    assert tight.spilled_blocks, "lm_head is never the first rung"
    assert tight.ot_patterns[-1] == LM_HEAD_SPILL_PATTERN


def test_the_lm_head_pattern_is_anchored():
    """An unanchored "output\\.weight" also matches every blk.N.attn_output.weight
    under std::regex_search, which silently moves the attention output projections
    too (measured: an extra 357 MiB on a 27B)."""
    assert LM_HEAD_SPILL_PATTERN.startswith("^output")
    assert LM_HEAD_SPILL_PATTERN.endswith("$"), "must not match blk.N.attn_output.weight"


def test_a_spill_host_ram_cannot_hold_abstains():
    """--fit on at least demand-pages; a plan here would be an OOM kill."""
    got = _plan(_Stub(avail_mib = 3 * 1024), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 12 * 1024)
    assert got is None or got.load_mode_none is False


def test_a_floor_that_does_not_fit_emits_no_flags():
    """If attention plus the cache alone will not fit, no weight spill helps and
    only -ngl (which evicts the cache) or a smaller context would.

    Asserted on the FLAGS, not on the plan object. plan_placement reports that
    case as Plan(changed=False, insufficient=True, ot_patterns=()), and a
    dataclass instance is truthy, so a seam that tested the plan for truthiness
    emitted -ngl -1 --fit off with nothing moved to the host: every layer pinned
    to the GPU and the fitter switched off, on precisely the load the planner had
    just said does not fit. Checking `insufficient` on the returned plan passes
    either way and proves nothing.
    """
    got = _plan(_Stub(), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 4 * 1024)
    assert got is not None
    assert got.insufficient
    assert LlamaCppBackend._spill_plan_flags_for(got) == []


def test_an_abstaining_plan_emits_no_flags():
    """Every plan_placement abstention returns a Plan, not None: incomplete
    layout, unified-memory host, no creditable VRAM, no usable context. None of
    them may reach the argv."""
    for plan in (
        plan_placement(ModelLayout(), [8 * GIB], 32 * GIB, 4096),
        plan_placement(
            ModelLayout(complete = True, n_ctx_train = 4096),
            [0],
            32 * GIB,
            4096,
        ),
    ):
        assert plan.spills_anything is False
        assert LlamaCppBackend._spill_plan_flags_for(plan) == []


def test_a_plan_that_already_fits_emits_no_flags():
    """The planner disagreeing with Studio ("it fits after all") is not licence
    to pin every layer and turn the fitter off."""
    got = _plan(_Stub(), free_mib = 200 * 1024)
    assert got is not None and not got.spills_anything
    assert LlamaCppBackend._spill_plan_flags_for(got) == []


def test_the_measured_cache_floors_the_planners_own_estimate():
    """layout.kv_bytes is a bare f16 GQA product with no cache-dtype, SWA, MLA,
    stream or padding term, so on an MLA or f32-cache load it lands under the
    cache Studio already sized. Under is the dangerous direction: the deficit
    comes out small, too few blocks spill, and --fit off follows the plan."""
    stub = _Stub()
    layout = stub._tensor_spill_layout("/models/stub.gguf")
    unfloored = plan_placement(layout, [20 * GIB], 64 * GIB, 32768)
    floored = plan_placement(
        layout, [20 * GIB], 64 * GIB, 32768, kv_bytes_floor = 8 * GIB
    )
    assert len(floored.spilled_blocks) > len(unfloored.spilled_blocks)


def test_the_seam_hands_the_planner_studios_cache_size():
    """The byte-accurate number is computed at the call site and was previously
    only tested for nonzero. A bigger measured cache has to buy more spill."""
    small = _plan(_Stub(), kv = 2 * GIB, free_mib = 14 * 1024)
    large = _plan(_Stub(), kv = 10 * GIB, free_mib = 14 * 1024)
    assert small is not None and large is not None
    assert len(large.spilled_blocks) > len(small.spilled_blocks)


# --------------------------------------------------------------- the revocation


def test_a_retry_takes_the_plan_back_out_and_restores_the_fitter():
    stub = _Stub()
    stub._spill_plan_flags = ["-ngl", "-1", "--fit", "off", "-ot", FFN_SPILL_PATTERN]
    cmd = ["llama-server", "-m", "x.gguf", *stub._spill_plan_flags, "--port", "8080"]
    got = stub._drop_tensor_spill(cmd, "noflash")
    assert "-ot" not in got
    assert got[-2:] == ["--fit", "on"]
    assert got[:4] == ["llama-server", "-m", "x.gguf", "--port"]


def test_the_revocation_keeps_the_record_for_a_later_respawn():
    """The CPU fallback respawns from an argv that may still carry the tokens."""
    stub = _Stub()
    stub._spill_plan_flags = ["-ngl", "-1", "--fit", "off", "-ot", FFN_SPILL_PATTERN]
    stub._drop_tensor_spill(["a", *stub._spill_plan_flags], "retry")
    assert stub._spill_plan_flags


def test_the_revocation_is_a_no_op_without_a_plan():
    stub = _Stub()
    cmd = ["llama-server", "--fit", "on"]
    assert stub._drop_tensor_spill(cmd, "retry") == cmd


def test_the_revocation_is_a_no_op_when_the_argv_never_carried_it():
    """The -cpu replay builds its own argv; stripping must not append --fit on
    to a command that never had the plan."""
    stub = _Stub()
    stub._spill_plan_flags = ["-ngl", "-1", "--fit", "off", "-ot", FFN_SPILL_PATTERN]
    replay = ["llama-server", "--gpu-layers", "0", "--fit", "off", "--device", "none"]
    assert stub._drop_tensor_spill(replay, "cpu") == replay


# ------------------------------------------------------------------- emission


def test_repeated_ot_flags_rather_than_a_joined_value():
    """llama-server accumulates repeated -ot and splits an -ot VALUE on ",".
    Joining with ";" is llama-bench's syntax and would be parsed as one pattern.
    """
    plan = Plan(ot_patterns = (FFN_SPILL_PATTERN, LM_HEAD_SPILL_PATTERN))
    tokens = [tok for pat in plan.ot_patterns for tok in ("-ot", f"{pat}=CPU")]
    assert tokens.count("-ot") == 2
    assert ";" not in " ".join(tokens)


# ---------------------------------------------------- the argv, structurally


def _load_model_source() -> str:
    import inspect
    return inspect.getsource(LlamaCppBackend.load_model)


def test_the_abstain_branch_still_emits_exactly_fit_on():
    """The negative half of the compatibility argument: when the planner
    declines, the arm emits what it emitted before this existed.

    This used to assert the SOURCE TEXT of the else branch, which is why it
    passed while the abstain path was in fact emitting "-ngl -1 --fit off" --
    Plan is a dataclass with no __bool__, so the truthiness check upstream fired
    on abstained plans too. A source pin cannot see that. Driven through the
    real flag builder instead, so the emptiness that routes control to the else
    branch is what is actually asserted.
    """
    from core.inference.offload_planner import Plan

    for reason in (
        "layout or device inventory incomplete, leaving llama.cpp defaults",
        "unified memory host, spilling frees no device memory",
        "no creditable VRAM after per-device overhead",
        "no usable context length",
    ):
        plan = Plan(reason = reason)
        assert plan, "Plan is truthy, which is the trap this test exists for"
        assert LlamaCppBackend._spill_plan_flags_for(plan) == [], reason

    # Same for a plan that cannot place the load, and for one that needs nothing.
    assert LlamaCppBackend._spill_plan_flags_for(
        Plan(reason = "does not fit", insufficient = True)
    ) == []
    assert LlamaCppBackend._spill_plan_flags_for(None) == []


def test_the_plan_disables_the_fitter_and_pins_every_layer_to_a_gpu():
    """Both halves are load-bearing.

    --fit off: the fitter aborts only on an n_gpu_layers the USER set
    (common/fit.cpp:377) and -1 IS llama.cpp's default (llama-model.cpp:2453), so
    -ngl -1 does NOT hold it off. Left on it would re-place layers over the plan
    and put the cache back on the host, which is the thing being fixed.

    -ngl -1: keeps every layer assigned to a GPU, so dev_layer(il) is a GPU and
    the cache is allocated there (llama-kv-cache.cpp:215).

    Driven through the real emitter rather than matched against the source text:
    a formatter that rewraps the literal must not be able to fail this.
    """
    got = _plan(_Stub(), free_mib = 12 * 1024)
    assert got is not None and got.spills_anything
    flags = LlamaCppBackend._spill_plan_flags_for(got)
    assert flags[:4] == ["-ngl", "-1", "--fit", "off"]
    assert flags.count("-ot") == len(got.ot_patterns) >= 1
    assert all(tok.endswith("=CPU") for tok in flags[5::2])


def test_a_spill_plan_startup_failure_can_revoke_the_plan():
    """The spill arm leaves fully_gpu_offloaded False and runs under use_fit with
    an explicit --fit on the argv, so BOTH existing crash recoveries are
    unreachable from it. Without a third arm a slightly optimistic plan skips
    straight to the terminal fallbacks instead of retrying the --fit on placement
    it replaced. Reachability only: the revocation must be reachable from the
    crash path, not just from the `label` guard at the top of the spawn."""
    import inspect

    body = inspect.getsource(LlamaCppBackend.load_model)
    body = body[body.index("def _spawn_and_wait") :]
    assert body.count("_drop_tensor_spill") >= 2


def test_the_revocation_runs_only_on_retries():
    """Gated on `label`, which is empty only on the first spawn. A retry added
    later is covered without anyone remembering to strip -- the failure mode that
    produced most of the review traffic on the load-mode work."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.index("def _spawn_and_wait")
    head = src[idx : idx + 1600]
    assert "if label:" in head
    assert "_drop_tensor_spill" in head
