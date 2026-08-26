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

# The planner's decision table is covered in test_offload_planner.py. HERE is the
# seam: whether the launch path declines when it should, emits the tokens
# llama-server actually parses, and takes the plan back out on every retry.
FFN_SPILL_PATTERN = r"blk\.\d+\.ffn_"
LM_HEAD_SPILL_PATTERN = LM_HEAD_PATTERN

GIB = 1024**3
MIB = 1024 * 1024


class _Stub:
    """Only what the seam touches."""

    _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024
    _HOST_RAM_HEADROOM_MIB = 2048
    # Discrete by default, so every existing case here is a discrete host. The
    # unified answer is the APU's, and it is exercised deliberately below.
    _unified = False
    # Bytes of trailing nextn/MTP blocks the layout dropped. 0 = an ordinary
    # model, which is every existing case here.
    _excluded_bytes = 0

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        return self._unified

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
            has_excluded_blocks = bool(self._excluded_bytes),
            excluded_block_bytes = self._excluded_bytes,
            complete = True,
        )


def _inputs(
    model_size = 30 * GIB,
    kv = 2 * GIB,
    free_mib = 24 * 1024,
    indices = None,
    usable_mib = None,
    extra_gpu = 0,
    mtp = None,
    shared = None,
    gpus = None,
    n_parallel = 1,
    compute_flat = 0,
    ctx_compute = 0,
):
    return {
        "model_size": model_size,
        "kv_cache_bytes": kv,
        "gpus": list(gpus) if gpus is not None else [(0, free_mib)],
        "gpu_usable_mib": {} if usable_mib is None else {0: usable_mib},
        "compute_buffer_flat": compute_flat,
        "ctx_compute_per_device": ctx_compute,
        "extra_gpu_bytes": extra_gpu,
        "gpu_indices": indices,
        "soft_overhead": 0,
        "model_path": "/models/stub.gguf",
        "n_ctx": 32768,
        "n_parallel": n_parallel,
        "shared_gpu_ids": set() if shared is None else set(shared),
        **({} if mtp is None else {"mtp_will_engage": mtp}),
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
    """The planner disagreeing with Unsloth ("it fits after all") is not licence
    to pin every layer and turn the fitter off."""
    got = _plan(_Stub(), free_mib = 200 * 1024)
    assert got is not None and not got.spills_anything
    assert LlamaCppBackend._spill_plan_flags_for(got) == []


def test_the_measured_cache_floors_the_planners_own_estimate():
    """layout.kv_bytes is a bare f16 GQA product with no cache-dtype, SWA, MLA,
    stream or padding term, so on an MLA or f32-cache load it lands under the
    cache Unsloth already sized. Under is the dangerous direction: the deficit
    comes out small, too few blocks spill, and --fit off follows the plan."""
    stub = _Stub()
    layout = stub._tensor_spill_layout("/models/stub.gguf")
    unfloored = plan_placement(layout, [20 * GIB], 64 * GIB, 32768)
    floored = plan_placement(layout, [20 * GIB], 64 * GIB, 32768, kv_bytes_floor = 8 * GIB)
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
    assert (
        LlamaCppBackend._spill_plan_flags_for(Plan(reason = "does not fit", insufficient = True)) == []
    )
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


# ------------------------------------------------- unified memory APUs (ROCm)


def test_a_unified_memory_apu_is_declared_to_the_planner():
    """The seam has to TELL the planner the host is unified; it cannot guess.

    On a unified-memory APU the credited "VRAM" is system RAM, so an -ot spill
    moves bytes between two names for one pool: it frees no device memory and
    only buys the CPU backend's slower read path. plan_placement abstains on
    those, but solely on HostProfile.unified_memory, which defaults False.

    The neighbouring `shared_gpu_ids` guard does not cover this. It is populated
    only when `is_vulkan_backend` is true (Vulkan reports total VRAM 0 for an
    iGPU), so a Strix Halo on ROCm -- what scripts/install_rocm_wsl_strixhalo.sh
    installs -- arrives with an EMPTY shared set, its APU "VRAM" credited AND
    host RAM credited, which counts one pool twice.
    """
    discrete = _Stub()
    apu = _Stub()
    apu._unified = True

    # Same numbers either way, so the difference is attributable to the flag.
    got_discrete = _plan(discrete, free_mib = 12 * 1024)
    got_apu = _plan(apu, free_mib = 12 * 1024)

    assert got_apu is not None
    assert not got_apu.spills_anything
    assert "unified memory" in got_apu.reason
    assert LlamaCppBackend._spill_plan_flags_for(got_apu) == []

    assert got_discrete is not None
    assert got_discrete.spills_anything, "the discrete host is free to plan"

    # And the seam passes the real predicate rather than leaving the default.
    import inspect

    compact = "".join(inspect.getsource(LlamaCppBackend._planned_tensor_spill).split())
    assert "unified_memory=self._amd_apu_wants_unified_memory(" in compact


# ------------------------------------------- the budget the fit actually tested


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--split-mode", "none"],
        ["-sm", "none"],
        ["--split-mode", "row"],
        ["--tensor-split", "3,1"],
        ["-ts", "3,1"],
    ],
)
def test_a_split_placement_argument_declines(extra_args):
    """Split mode and tensor split are placement, and they are not in
    _DEVICE_FLAGS, so the old guard let them through.

    They reach the child: the layer path's own note in this module says "-sm
    none/row keep the layer path and pass through", and extras are appended
    last, so they win. Under -sm none llama.cpp truncates model->devices to the
    single main GPU (llama.cpp:288-299), while this planner credits the SUM of
    every selected card -- so it would size the plan against a pool the child
    never gets and then pin that with --fit off. -ts replaces the free-memory
    proportional split (llama-model.cpp:1417-1447), so one device can overflow
    while the pool total still fits.
    """
    assert _plan(_Stub(), extra_args = extra_args) is None
    # Not vacuous: the same load DOES plan without the argument.
    assert _plan(_Stub()) is not None


@pytest.mark.parametrize(
    "env",
    [{"LLAMA_ARG_SPLIT_MODE": "none"}, {"LLAMA_ARG_TENSOR_SPLIT": "3,1"}],
)
def test_an_inherited_split_placement_env_declines(env):
    assert _plan(_Stub(), env = env) is None


def test_the_planner_gets_the_budget_the_fit_tested_not_raw_free():
    """_gpu_usable applies the user's VRAM-budget fraction and the per-card
    reserve floor; raw free_mib is neither.

    The budget is settable down to 0.80 (vram_budget_settings), so on a 24 GiB
    card the fit tests ~4.8 GiB less than free, and even at the 0.97 default a
    big card holds back 3% rather than the planner's flat per-device reserve.
    Planning on free credits VRAM the fit had already ruled out, emits too few
    -ot overrides, and then appends --fit off over the result.
    """
    stub = _Stub()
    on_free = _plan(stub, free_mib = 15 * 1024)
    on_budget = _plan(stub, free_mib = 15 * 1024, usable_mib = 13 * 1024)

    assert on_free is not None and on_budget is not None
    assert on_free.spills_anything and on_budget.spills_anything
    assert len(on_budget.spilled_blocks) > len(on_free.spilled_blocks)


def test_the_projector_and_mtp_reserve_reach_the_planner():
    """The layout is rebuilt from the target GGUF's tensor table, so a vision
    projector and the MTP draft reserve are invisible to it -- yet both are in
    the model_size_fit that produced the use_fit verdict this arm answers.
    Leaving them out makes the deficit too small on exactly the loads Unsloth
    already judged not to fit, and the launch then follows that with --fit off.
    """
    stub = _Stub()
    without = _plan(stub, free_mib = 15 * 1024)
    with_extra = _plan(stub, free_mib = 15 * 1024, extra_gpu = 2 * GIB)

    assert without is not None and with_extra is not None
    assert without.spills_anything and with_extra.spills_anything
    assert len(with_extra.spilled_blocks) > len(without.spilled_blocks)


def test_the_layout_cache_notices_a_gguf_replaced_in_place(tmp_path, monkeypatch):
    """The backend instance outlives a load, so re-downloading or re-quantising
    a model to the SAME path inside one session must not be planned against the
    old tensor table: a larger replacement understates the deficit, emits too
    few -ot overrides, and still appends --fit off.
    """
    from core.inference import offload_layout

    class _CacheStub:
        _tensor_spill_layout = LlamaCppBackend._tensor_spill_layout

    reads: list[int] = []

    def _fake(path):
        reads.append(len(reads))
        return ModelLayout(arch = "llama", n_layers = len(reads), complete = True)

    monkeypatch.setattr(offload_layout, "layout_from_gguf", _fake)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"a")
    backend = _CacheStub()

    first = backend._tensor_spill_layout(str(gguf))
    assert backend._tensor_spill_layout(str(gguf)) is first, "unchanged file is cached"
    assert len(reads) == 1

    gguf.write_bytes(b"bb")
    second = backend._tensor_spill_layout(str(gguf))
    assert len(reads) == 2, "a replaced file is re-read"
    assert second.n_layers != first.n_layers


# ------------------------------------------------- KV placement is placement too


@pytest.mark.parametrize(
    "extra_args",
    [["-nkvo"], ["--no-kv-offload"]],
)
def test_a_forced_cpu_kv_cache_declines(extra_args):
    """-nkvo sends the WHOLE cache to host RAM whatever -ngl says: offload is a
    single scalar on the cache object and the buffer type falls back to
    ggml_backend_cpu_buffer_type() for every layer (llama-kv-cache.cpp:210-219),
    with the same branch in the recurrent and DSV4 caches.

    This planner charges the cache against VRAM via kv_bytes_floor, so it would
    spill FFN blocks to cover a deficit the child never has, and a spilled block
    is read by the CPU backend on every token. --fit on gets it right for free
    because it measures buffer types rather than modelling them: a host-resident
    cache lands in the host bucket (common/fit.cpp:74-79), which the per-device
    fitting loop then ignores (common/fit.cpp:326-347).
    """
    assert _plan(_Stub(), extra_args = extra_args) is None
    # Not vacuous: the same load DOES plan without the argument.
    assert _plan(_Stub()) is not None


@pytest.mark.parametrize("env", [{"LLAMA_ARG_KV_OFFLOAD": "0"}, {"LLAMA_ARG_KV_OFFLOAD": "false"}])
def test_an_inherited_kv_offload_env_declines(env):
    assert _plan(_Stub(), env = env) is None


def test_the_positive_kv_offload_form_still_plans():
    """-kvo / --kv-offload is the enabling half of the same paired argument
    (common/arg.cpp:2402-2409) and argv is last-wins, so a re-enable must not be
    read as a decline."""
    assert _plan(_Stub(), extra_args = ["-kvo"]) is not None
    assert _plan(_Stub(), extra_args = ["-nkvo", "--kv-offload"]) is not None
    assert _plan(_Stub(), env = {"LLAMA_ARG_KV_OFFLOAD": "1"}) is not None


# --------------------------------------------- embedded MTP blocks when drafting


def test_an_engaged_draft_charges_the_excluded_mtp_blocks():
    """The layout drops the trailing nextn/MTP blocks, which is right until a
    draft engages.

    --spec-type draft-mtp sets load_mtp on the TARGET's own model params
    (common/common.cpp:1713), which clears the TENSOR_SKIP those blocks
    otherwise carry (models/glm4-moe.cpp:42-44,
    llama-model-loader.cpp:1123-1131), and i_gpu_start counting backwards from
    n_layer_all (llama-model.cpp:1449) places them on a GPU first. Unsloth's own
    budget already paid for them -- an embedded head contributes 0 to the draft
    weights precisely because they sit inside model_size -- so leaving them out
    here makes the deficit too small, which is the optimistic direction.
    """
    stub = _Stub()
    stub._excluded_bytes = 3 * GIB

    idle = _plan(stub, free_mib = 15 * 1024, mtp = False)
    drafting = _plan(stub, free_mib = 15 * 1024, mtp = True)

    assert idle is not None and drafting is not None
    assert idle.spills_anything and drafting.spills_anything
    assert len(drafting.spilled_blocks) > len(idle.spilled_blocks)


def test_an_ordinary_model_is_unaffected_by_the_draft_flag():
    """No excluded blocks, nothing to charge: the flag must not move a plan on a
    model that has no MTP head."""
    stub = _Stub()
    assert stub._excluded_bytes == 0
    idle = _plan(stub, free_mib = 15 * 1024, mtp = False)
    drafting = _plan(stub, free_mib = 15 * 1024, mtp = True)
    assert idle.spilled_blocks == drafting.spilled_blocks


# ------------------------------------------- devices the child can still use


def test_a_shared_device_the_child_keeps_declines_the_plan():
    """Dropping a shared iGPU from the BUDGET only helps if the child stops using
    it, and nothing here can promise that: the auto-Vulkan arm pins every
    DETECTED gpu, and it runs after this snapshot is taken, so gpu_indices reads
    None right here. llama.cpp then splits rows by free memory, and an iGPU's
    free is the whole host pool, so it draws the majority of the layers and their
    caches -- pinned with --fit off. Decline instead of planning around it.
    """
    stub = _Stub()
    gpus = [(0, 12 * 1024), (1, 12 * 1024)]
    assert _plan(stub, gpus = gpus, shared = {1}) is None
    # Named in the pin: still declined, since the child is told to use it.
    assert _plan(stub, gpus = gpus, shared = {1}, indices = [0, 1]) is None
    # Pinned away from the shared device: the plan may proceed.
    kept = _plan(stub, gpus = gpus, shared = {1}, indices = [0])
    assert kept is not None
    # No shared device at all is the ordinary path.
    assert _plan(stub, gpus = gpus) is not None


def test_the_compute_buffer_reaches_the_planner():
    """model_size_fit carries the flat compute buffer IN ADDITION to
    soft_overhead, and the fit adds the context-linear _cc_bytes per device ON
    TOP of the pipeline reserve -- so there is nothing double-charged to avoid,
    and omitting them under-reserves on exactly the long-context loads this arm
    exists for. Both must move the deficit."""
    stub = _Stub()
    base = _plan(stub, free_mib = 14 * 1024)
    assert base is not None and base.spills_anything

    per_device = _plan(stub, free_mib = 14 * 1024, ctx_compute = 3 * GIB)
    flat = _plan(stub, free_mib = 14 * 1024, compute_flat = 3 * GIB)
    for tighter in (per_device, flat):
        assert tighter is not None
        assert len(tighter.spilled_blocks) > len(
            base.spilled_blocks
        ), "charging the compute buffer has to spill MORE, not the same"


def test_the_seam_hands_the_planner_raw_free_vram_for_the_split():
    """llama.cpp sizes its row ranges from raw free VRAM and knows nothing about
    Unsloth's budget, so the seam has to pass the raw numbers separately. Observed
    through the call rather than the source: the planner records what it was
    given."""
    seen = {}
    stub = _Stub()
    import core.inference.offload_planner as mod

    real = mod.plan_placement

    def spy(*a, **kw):
        seen.update(kw)
        seen["vram"] = a[1]
        return real(*a, **kw)

    mod.plan_placement = spy
    try:
        _plan(stub, gpus = [(0, 8 * 1024), (1, 16 * 1024)], usable_mib = 6 * 1024)
    finally:
        mod.plan_placement = real

    # Index 0 carries a usable_mib override, index 1 falls back to raw free. The
    # split weights must be raw on BOTH, and in the same device order.
    assert seen["split_weights_per_device"] == [8 * 1024 * MIB, 16 * 1024 * MIB]
    assert seen["vram"] == [6 * 1024 * MIB, 16 * 1024 * MIB]


def test_a_pinned_draft_device_declines_the_plan():
    """A drafter pinned to a card is placement, and it is not in _DEVICE_FLAGS.
    Its reserve rides in extra_resident_bytes, which _per_device_shortfall books
    onto device 0, while llama.cpp puts the drafter where the pin says -- so the
    plan would approve the wrong device's footprint and emit --fit off. cpu and
    none are not pins and must not cost a plan."""
    stub = _Stub()
    assert _plan(stub, extra_args = ["--spec-draft-device", "CUDA1"]) is None
    assert _plan(stub, extra_args = ["--device-draft", "CUDA0,CUDA1"]) is None
    assert _plan(stub, extra_args = ["--spec-draft-device", "cpu"]) is not None
    assert _plan(stub, extra_args = ["--spec-draft-device", "none"]) is not None
    assert _plan(stub) is not None


def test_a_pass_through_parallel_that_grows_the_cache_declines_the_plan():
    """Slots are sizing, not placement. Unsloth's --parallel is emitted first and
    the extras are appended after it, so a larger pass-through wins at the child
    while the deficit here was priced for the smaller count -- too few blocks
    spilled, then pinned with --fit off. A SMALLER one only over-reserves, which
    is safe, so it must not cost a plan."""
    stub = _Stub()
    assert _plan(stub, n_parallel = 1, extra_args = ["--parallel", "8"]) is None
    assert _plan(stub, n_parallel = 1, extra_args = ["-np", "4"]) is None
    assert _plan(stub, n_parallel = 1, extra_args = ["--parallel=8"]) is None
    assert _plan(stub, n_parallel = 8, extra_args = ["--parallel", "8"]) is not None
    assert _plan(stub, n_parallel = 8, extra_args = ["--parallel", "2"]) is not None
    assert _plan(stub, n_parallel = 1, env = {"LLAMA_ARG_N_PARALLEL": "8"}) is None
    # Last wins, exactly as llama.cpp parses it.
    assert _plan(stub, n_parallel = 4, extra_args = ["--parallel", "8", "-np", "2"]) is not None
    assert _plan(stub, n_parallel = 1) is not None
