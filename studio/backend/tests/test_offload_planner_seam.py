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

import re
from types import SimpleNamespace

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.offload_layout import LM_HEAD_PATTERN, BlockLayout, ModelLayout
from core.inference.offload_planner import (
    ContextPolicy,
    Plan,
    plan_placement,
    smart_offload_enabled,
)
from core.inference.llama_cpp import _extra_args_tensor_split
from core.inference.llama_cpp import _linux_math_core_count
import core.inference.llama_cpp as llama_mod
import inspect
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


class _SmtHost:
    @staticmethod
    def cpu_count(logical = True):
        return 16 if logical else 8


# The planner's decision table is covered in test_offload_planner.py. HERE is the
# seam: whether the launch path declines when it should, emits the tokens
# llama-server actually parses, and takes the plan back out on every retry.
FFN_SPILL_PATTERN = r"blk\.\d+\.ffn_"
LM_HEAD_SPILL_PATTERN = LM_HEAD_PATTERN

GIB = 1024**3
MIB = 1024 * 1024


@pytest.fixture(autouse = True)
def _discrete_host(monkeypatch):
    """Pin the one host fact the seam reads live.

    _planned_tensor_spill calls is_apple_silicon() directly, so on an arm64 Mac
    every case here would decline and the suite would grade the runner instead of
    the planner. Every test here means "a discrete host"; the Apple answer is
    asserted deliberately, below.
    """
    import utils.hardware
    monkeypatch.setattr(utils.hardware, "is_apple_silicon", lambda: False)


@pytest.fixture(scope = "session")
def sparse_adapter(tmp_path_factory):
    """A 3 GiB adapter file that costs neither RAM nor (on ext4/APFS) disk.

    ``_sidecar_adapter_bytes`` reads ``os.stat().st_size`` and nothing else, so
    the fixture only has to be the right SIZE. Materialising it as real zeros
    spiked RSS by 3 GiB per test and left up to 9 GiB in the session's tmp dir
    (pytest keeps every test's directory for the whole run), which the 14 GB
    runner disk does not have to spare. ``truncate`` sets the length without
    writing: a hole on ext4/APFS, and on NTFS space that is reserved but never
    zero-filled. One file for the session, since every caller only stats it.
    """
    path = tmp_path_factory.mktemp("adapters") / "adapter.gguf"
    with open(path, "wb") as handle:
        handle.truncate(3 * GIB)
    return path


class _Stub:
    """Only what the seam touches.

    Must NOT declare _HOST_RAM_HEADROOM_MIB: that constant is module-level, so
    declaring it here let the double supply what the real backend lacked, hiding an
    AttributeError on every planned load.
    """

    _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024
    # Discrete by default, so every existing case here is a discrete host. The
    # unified answer is the APU's, and it is exercised deliberately below.
    _unified = False
    # Bytes of trailing nextn/MTP blocks the layout dropped. 0 = an ordinary
    # model, which is every existing case here.
    _excluded_bytes = 0
    # Discrete CUDA by default. An integrated SoC (Jetson, DGX Spark) is the
    # unified-memory answer on the CUDA side, exercised deliberately below.
    _integrated_cuda = False
    is_embedding_gguf = False

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        return self._unified

    def _integrated_cuda_unified_memory(self, gpu_indices = None):
        return self._integrated_cuda

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

    def _tensor_spill_layout(
        self,
        model_path,
        *,
        all_shards = False,
    ):
        """Stand in for the GGUF read: the seam's job is to decline or to hand the planner
        well-formed inputs, not to parse a file.
        """
        self._layout_all_shards = all_shards
        if self._ffn_weight_bytes is None or not model_path:
            return None
        if getattr(self, "sharded", False) and not all_shards:
            return ModelLayout(arch = "qwen35", n_layers = 0, complete = False)
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
            # 96 KiB per token, i.e. 3 GiB at the 32768 these tests plan at.
            kv_bytes_per_token_f16 = 98304,
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
    kv = 3 * GIB,
    free_mib = 24 * 1024,
    indices = None,
    usable_mib = None,
    extra_gpu = 0,
    mtp = None,
    shared = None,
    gpus = None,
    n_parallel = 1,
    n_threads = None,
    compute_flat = 0,
    ctx_compute = 0,
    env_mmproj = 0,
    env_mmproj_unsized = False,
    separate_draft = False,
    n_ctx = 32768,
    n_ubatch = None,
    reserve_floor = 0,
    host_unpriced = 0,
    kv_bytes_at = None,
    link_gib_s = None,
):
    return {
        "model_size": model_size,
        "kv_cache_bytes": kv,
        "gpus": list(gpus) if gpus is not None else [(0, free_mib)],
        "gpu_usable_mib": {} if usable_mib is None else {0: usable_mib},
        "compute_buffer_flat": compute_flat,
        "ctx_compute_per_device": ctx_compute,
        "extra_gpu_bytes": extra_gpu,
        "env_mmproj_bytes": env_mmproj,
        "env_mmproj_unsized": env_mmproj_unsized,
        "gpu_indices": indices,
        "soft_overhead": 0,
        "reserve_floor_bytes": reserve_floor,
        "model_path": "/models/stub.gguf",
        "n_ctx": n_ctx,
        "n_ubatch": n_ubatch,
        "n_parallel": n_parallel,
        "n_threads": n_threads,
        "shared_gpu_ids": set() if shared is None else set(shared),
        "host_ram_unpriced_bytes": host_unpriced,
        "kv_bytes_at": kv_bytes_at,
        "link_gib_s": link_gib_s,
        "separate_draft_on_gpu": separate_draft,
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


def test_the_seam_reads_no_attribute_the_real_backend_lacks():
    """The planner is default-on, so every GGUF load runs this seam against the REAL
    class, not the double above.

    _planned_tensor_spill read the headroom as self._HOST_RAM_HEADROOM_MIB, but that
    constant is module-level, so every planned load 500'd with AttributeError (seen on
    a 67.6 GB GGUF, Colab T4 high-RAM). The suite stayed green only because the double
    declared it."""
    for name in ("_HOST_RAM_HEADROOM_MIB",):
        assert not hasattr(_Stub, name), (
            f"the double declares {name}, so it can hide a missing attribute on the "
            "real backend again"
        )
    source = inspect.getsource(LlamaCppBackend._planned_tensor_spill)
    assert "self._HOST_RAM_HEADROOM_MIB" not in source


def test_the_planner_is_off_by_default():
    """No env, no plan. The flag is opt-IN again after #9861.

    A load that would otherwise get a real spill plan is the case that matters:
    if only the roomy one were checked, the default could flip back unnoticed,
    since a fitting load plans nothing either way.
    """
    assert smart_offload_enabled({}) is False
    assert _Stub()._planned_tensor_spill(_inputs(), env = {}) is None
    assert _Stub()._planned_tensor_spill(_inputs(free_mib = 14 * 1024), env = {}) is None

    # The same tight load still plans when asked explicitly, so the revert moved
    # the default and nothing else.
    tight = _Stub()._planned_tensor_spill(
        _inputs(free_mib = 14 * 1024), env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    )
    assert tight is not None and tight.spills_anything


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "enabled", "ON", " 1 "])
def test_the_gate_accepts_the_usual_spellings(value):
    assert smart_offload_enabled({"UNSLOTH_SMART_OFFLOAD": value}) is True


@pytest.mark.parametrize("value", ["0", "off", "", "no", "disabled", "false"])
def test_an_explicit_off_still_disables(value):
    """The escape hatch has to keep working."""
    assert smart_offload_enabled({"UNSLOTH_SMART_OFFLOAD": value}) is False
    assert _Stub()._planned_tensor_spill(_inputs(), env = {"UNSLOTH_SMART_OFFLOAD": value}) is None


@pytest.mark.parametrize("value", ["nope", "flase", "onn", "2"])
def test_an_unrecognised_value_disables_rather_than_enables(value):
    """A fumbled on-spelling must not turn the planner on by accident, and now
    lands on the same answer as setting nothing at all."""
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


def test_a_moe_load_is_declined_because_the_fitter_places_it_the_same_way():
    """On MoE the seam now declines, and the numbers say why."""
    got = _plan(_Stub(moe = 40), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 12 * 1024, n_ctx = 16384)
    assert got is not None
    assert not got.spills_anything
    assert "not worth it" in got.reason
    costs = re.findall(r"(\d+) ms", got.reason)
    assert len(costs) == 2 and costs[0] == costs[1], got.reason


def test_a_moe_load_at_a_long_prompt_is_left_to_the_fitter_before_it_is_ranked():
    """The same cell one slot at 32768 tokens declines on the MEASURED long-prompt loss, not
    on the ranking: the cost model cannot see it because both arms spill through the same
    mechanism, so the reason carries no millisecond figures at all."""
    got = _plan(_Stub(moe = 40), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 12 * 1024)
    assert got is not None
    assert not got.spills_anything
    assert "tokens per slot" in got.reason and "32K" in got.reason, got.reason
    assert not re.findall(r"(\d+) ms", got.reason), got.reason


def test_lm_head_is_only_spilled_after_ffn():
    """43% of generation on its own, 16% on top of an already host-bound step, so
    it is never the first rung."""
    # Stated over a sweep rather than at one hand-picked card size, because the cost gate can
    # decline any given one and a single fixture would then retire its own assertion silently.
    spilled_anything = False
    for free_mib in (4096, 4608, 5632, 6144, 8192, 10240, 12288, 14336):
        plan = _plan(_Stub(), model_size = 60 * GIB, kv = 2 * GIB, free_mib = free_mib)
        if plan is None or not plan.spills_anything:
            continue
        spilled_anything = True
        if plan.spilled_lm_head:
            assert plan.spilled_blocks, "lm_head is never the first rung"
            assert plan.ot_patterns[-1] == LM_HEAD_SPILL_PATTERN
    assert spilled_anything, "no card size spilled at all, so the sweep tested nothing"


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
    # A GiB tighter than before, for the single-card pipeline reserve this no
    # longer charges; the case is still "no rung fits".
    got = _plan(_Stub(), model_size = 30 * GIB, kv = 2 * GIB, free_mib = 3 * 1024)
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
    """The byte-accurate number is computed at the call site and was previously only tested for
    nonzero.
    """
    small = _plan(_Stub(), kv = 2 * GIB, free_mib = 14 * 1024)
    large = _plan(_Stub(), kv = 10 * GIB, free_mib = 14 * 1024)
    assert small is not None and large is not None
    assert large.predicted_request_ms > small.predicted_request_ms > 0.0


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

    src = inspect.getsource(LlamaCppBackend.load_model)
    helper = src[src.index("def _revoke_spill_plan") : src.index("def _spawn_and_wait")]
    assert "_drop_tensor_spill" in helper
    body = src[src.index("def _spawn_and_wait") :]
    assert body.count("_revoke_spill_plan") >= 2


def test_the_revocation_runs_only_on_retries():
    """Gated on `label`, which is empty only on the first spawn. A retry added
    later is covered without anyone remembering to strip -- the failure mode that
    produced most of the review traffic on the load-mode work."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.index("def _spawn_and_wait")
    head = src[idx : idx + 1600]
    assert "if label:" in head
    assert "_revoke_spill_plan" in head


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

    compact = "".join(inspect.getsource(LlamaCppBackend._planned_tensor_spill).split())
    assert "unified_memory=self._amd_apu_wants_unified_memory(" in compact


# ------------------------------------------- the budget the fit actually tested


@pytest.mark.parametrize("extra_args", [["--split-mode", "row"], ["-sm", "row"]])
def test_tensor_parallel_split_still_declines(extra_args):
    """-sm row is tensor parallelism: llama.cpp splits each TENSOR across cards
    instead of handing out rows, so the row model does not describe it."""
    assert _plan(_Stub(), extra_args = extra_args) is None
    assert _plan(_Stub()) is not None, "not vacuous: the same load plans without it"


@pytest.mark.parametrize(
    "extra_args",
    [["--split_mode=row"], ["--split_mode", "row"]],
)
def test_the_underscore_spelling_of_split_mode_is_read(extra_args):
    """llama.cpp folds an underscore in any long option to a dash before looking the name up
    (common/arg.cpp:821, :1214), so --split_mode row IS -sm row to the child. Matched raw, it
    read as no split mode and the row-split guard planned a tensor-parallel launch as a layer split."""
    from core.inference.llama_cpp import _extra_args_split_mode

    assert _extra_args_split_mode(extra_args, {}) == "row"
    assert _plan(_Stub(), extra_args = extra_args) is None


@pytest.mark.parametrize(
    "extra_args",
    [["--split-mode", "none"], ["-sm", "none"], ["-sm=none"]],
)
def test_split_mode_none_plans_against_one_card(extra_args):
    """-sm none is a device list of ONE, not a reason to decline: llama.cpp keeps
    only devices[main_gpu] (llama.cpp:288-299). Planning it as a two-card pool was
    the bug."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    got = _plan(_Stub(), gpus = two_cards, extra_args = extra_args)
    assert got is not None

    # Proof it used ONE card: the same plan as a genuine single card, where
    # crediting both would spill less.
    one_card = _plan(_Stub(), gpus = [(0, 14 * 1024)])
    assert len(got.spilled_blocks) == len(one_card.spilled_blocks)


@pytest.mark.parametrize(
    "extra_args,env",
    [
        (["--split-mode", "tensor"], None),
        (["-sm", "tensor"], None),
        (["-sm=tensor"], None),
        (None, {"LLAMA_ARG_SPLIT_MODE": "tensor"}),
    ],
)
def test_tensor_parallel_split_mode_tensor_declines(extra_args, env):
    """`tensor` is a real accepted -sm value (common/arg.cpp:2762-2776) and goes
    further than `row`: llama.cpp replaces the device list with a SINGLE meta
    device (llama.cpp:157-215), so a per-device budget describes nothing, and
    llama.cpp's own fitter refuses the mode outright (common/fit.cpp:182-183)."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, extra_args = extra_args, env = env) is None
    assert _plan(_Stub(), gpus = two_cards) is not None, "not vacuous"


def test_split_mode_none_with_an_explicit_main_gpu_still_declines():
    """--main-gpu keeps declining, so -sm none is supported only at llama.cpp's
    default main GPU of 0. The env twin declines for the same reason: it is not an
    argv flag, so _DEVICE_FLAGS never sees it, yet llama.cpp keeps
    devices[main_gpu] under -sm none all the same (llama.cpp:288-299), and
    budgeting the first retained card approves a GPU the child never loads onto."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-sm", "none", "-mg", "1"]) is None
    assert (
        _plan(
            _Stub(),
            gpus = two_cards,
            extra_args = ["-sm", "none"],
            env = {"LLAMA_ARG_MAIN_GPU": "1"},
        )
        is None
    )
    # Not vacuous, and not a blanket refusal of the env var's absence.
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-sm", "none"]) is not None
    assert (
        _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_MAIN_GPU": "0"}) is None
    ), "0 is still a pin the guard cannot verify against its own device order"


@pytest.mark.parametrize(
    "extra_args,env",
    [
        (["--tensor-split", "3,1"], None),
        (["-ts", "3,1"], None),
        (None, {"LLAMA_ARG_TENSOR_SPLIT": "3,1"}),
    ],
)
def test_tensor_split_becomes_the_row_weights(extra_args, env):
    """-ts REPLACES llama.cpp's free-memory split rather than skewing it: the
    values are copied into `splits` verbatim and prefix-summed
    (llama-model.cpp:1436-1447), so they are the row-ownership weight the
    per-device check needs -- data to use, not a reason to stop."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, extra_args = extra_args, env = env) is not None


@pytest.mark.parametrize(
    "extra_args,env",
    [
        (["--tensor-split", "3/1"], None),
        (["-ts", "3/1"], None),
        (None, {"LLAMA_ARG_TENSOR_SPLIT": "3/1"}),
    ],
)
def test_a_slash_delimited_tensor_split_is_the_same_override(extra_args, env):
    """llama.cpp splits -ts on ``[,/]+`` (common/arg.cpp:2791-2804), so ``3/1`` IS
    ``3,1`` to the child. Failing to parse it returned None, which the caller
    cannot tell from "no override": it planned 1:1 while the child ran 3:1."""
    assert _extra_args_tensor_split(extra_args, env or {}) == [3.0, 1.0]

    # And it reaches the planner as the row weights: the 3:1 plan, not free-VRAM.
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    skewed = _plan(_Stub(), gpus = two_cards, extra_args = extra_args, env = env)
    comma = _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "3,1"])
    assert skewed is not None and comma is not None
    assert skewed.spilled_blocks == comma.spilled_blocks


def test_a_malformed_or_short_tensor_split_declines():
    """Unparseable or fewer shares than cards leaves the row model undefined.
    Unparseable must DECLINE, not fall through: the parser returns None for both
    "no -ts" and "a -ts I could not read", and the child gets the flag either way.
    ``;`` is not one of llama.cpp's delimiters -- std::stof stops at it, so
    ``3;1`` is [3, 0] to the child, never [3, 1]."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "3"]) is None
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "abc"]) is None
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "3;1"]) is None
    assert _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_TENSOR_SPLIT": "abc"}) is None
    # Not vacuous: the same load plans with no -ts and with a readable one.
    assert _plan(_Stub(), gpus = two_cards) is not None
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "3,1"]) is not None


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
    # 12 and 11 GiB, not 14 and 13: at 13 the gate declines on its own merits, and this test
    # is about which NUMBER the seam hands the planner, so both arms must be on the plan side.
    on_free = _plan(stub, free_mib = 12 * 1024)
    on_budget = _plan(stub, free_mib = 12 * 1024, usable_mib = 11 * 1024)

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
    without = _plan(stub, free_mib = 14 * 1024)
    with_extra = _plan(stub, free_mib = 14 * 1024, extra_gpu = 2 * GIB)

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


def test_the_cache_estimator_reaches_the_planner(monkeypatch):
    """The snapshot half is pinned in the launch suite; this is the other end."""
    from core.inference import offload_planner

    seen = {}
    real = offload_planner.plan_placement

    def capture(*a, **k):
        seen["opts"] = k.get("opts")
        return real(*a, **k)

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    _plan(_Stub(), free_mib = 14 * 1024, kv_bytes_at = lambda ctx, p: 7 * MIB)
    assert seen["opts"].kv_bytes_at is not None
    assert seen["opts"].kv_bytes_at(4096, 1) == 7 * MIB

    seen.clear()
    _plan(_Stub(), free_mib = 14 * 1024)
    assert seen["opts"].kv_bytes_at is None


@pytest.mark.parametrize("extra_args", [["-nkvo"], ["--no-kv-offload"]])
def test_a_forced_cpu_kv_cache_is_priced_not_declined(extra_args):
    """-nkvo sends the WHOLE cache to host RAM whatever -ngl says: offload is a
    single scalar on the cache object and the buffer type falls back to
    ggml_backend_cpu_buffer_type() for every layer (llama-kv-cache.cpp:210-219),
    with the same branch in the recurrent and DSV4 caches.

    That makes the deficit SMALLER, so declining was the pessimistic reading: it
    charged a cache to VRAM the child never puts there and spilled FFN blocks to
    cover it. Now it is priced -- out of VRAM, into host RAM.
    """
    baseline = _plan(_Stub(), free_mib = 14 * 1024)
    forced = _plan(_Stub(), free_mib = 14 * 1024, extra_args = extra_args)
    assert forced is not None, "a host-resident cache is not a reason to abstain"

    # Out of VRAM: the 2 GiB cache left the footprint, so the deficit it was
    # spilling to cover is gone.
    assert len(forced.spilled_blocks) < len(baseline.spilled_blocks)

    # Into host RAM: with nothing spilled host_bytes would be token_embd alone
    # (512 MiB), so it must carry the cache too or the mmap decision is made
    # against a footprint short by the whole cache.
    assert forced.host_bytes >= 512 * MIB + 2 * GIB


@pytest.mark.parametrize("env", [{"LLAMA_ARG_KV_OFFLOAD": "0"}, {"LLAMA_ARG_KV_OFFLOAD": "false"}])
def test_an_inherited_kv_offload_env_is_priced_too(env):
    """Same resolution as argv: LLAMA_ARG_ is rewritten to LLAMA_ARG_NO_ and
    presence alone forces the value (common/arg.cpp:127-141)."""
    forced = _plan(_Stub(), free_mib = 14 * 1024, env = env)
    baseline = _plan(_Stub(), free_mib = 14 * 1024)
    assert forced is not None
    assert len(forced.spilled_blocks) < len(baseline.spilled_blocks)
    assert forced.host_bytes >= 512 * MIB + 2 * GIB


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

    idle = _plan(stub, free_mib = 14 * 1024, mtp = False)
    drafting = _plan(stub, free_mib = 14 * 1024, mtp = True)

    assert idle is not None and drafting is not None
    assert idle.spills_anything and drafting.spills_anything
    assert len(drafting.spilled_blocks) > len(idle.spilled_blocks)


def test_an_ordinary_model_is_unaffected_by_the_draft_flag():
    """No excluded blocks, nothing to charge: the flag must not move a plan on a
    model that has no MTP head."""
    stub = _Stub()
    assert stub._excluded_bytes == 0
    idle = _plan(stub, free_mib = 14 * 1024, mtp = False)
    drafting = _plan(stub, free_mib = 14 * 1024, mtp = True)
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


def test_a_pass_through_parallel_that_resizes_the_cache_declines_the_plan():
    """Slots are sizing, not placement."""
    stub = _Stub()
    assert _plan(stub, n_parallel = 1, extra_args = ["--parallel", "8"]) is None
    assert _plan(stub, n_parallel = 1, extra_args = ["-np", "4"]) is None
    assert _plan(stub, n_parallel = 1, extra_args = ["--parallel=8"]) is None
    assert _plan(stub, n_parallel = 8, extra_args = ["--parallel", "8"]) is not None
    assert _plan(stub, n_parallel = 8, extra_args = ["--parallel", "2"]) is None
    assert _plan(stub, n_parallel = 1, env = {"LLAMA_ARG_N_PARALLEL": "8"}) is None
    # Last wins, exactly as llama.cpp parses it.
    assert _plan(stub, n_parallel = 4, extra_args = ["--parallel", "8", "-np", "2"]) is None
    assert _plan(stub, n_parallel = 2, extra_args = ["--parallel", "8", "-np", "2"]) is not None
    assert _plan(stub, n_parallel = 1) is not None


# --------------------------------------- newly reachable now the gate is default-on


@pytest.mark.parametrize(
    "extra_args,env",
    [
        (["--rpc", "10.0.0.2:50052"], None),
        (["--rpc=10.0.0.2:50052"], None),
        (None, {"LLAMA_ARG_RPC": "10.0.0.2:50052"}),
    ],
)
def test_an_rpc_device_declines_the_plan(extra_args, env):
    """--rpc devices are REMOTE and llama.cpp puts them at the FRONT of the device
    list (llama.cpp:275), so the row split hands the first slice of layers to a
    card this planner never saw: every number it has comes from Studio's LOCAL
    gpus snapshot. Crediting local VRAM for layers that went to another host and
    then pinning it with --fit off has nothing left to catch it."""
    assert _plan(_Stub(), extra_args = extra_args, env = env) is None
    assert _plan(_Stub()) is not None, "not vacuous: the same load plans without it"


@pytest.mark.parametrize(
    "extra_args,env",
    [
        (["--fit-target", "4096"], None),
        (["-fitt", "4096"], None),
        (["--fit-target=4096,4096"], None),
        (None, {"LLAMA_ARG_FIT_TARGET": "4096"}),
    ],
)
def test_an_explicit_fit_target_declines_the_plan(extra_args, env):
    """--fit-target is VRAM the user told the fitter it may not spend: it becomes
    `targets.push_back(dmds_full[id].free - margins[id])` (common/fit.cpp:282-288).
    This planner credits that same VRAM and emits --fit off, so
    common_params_fit_impl never runs and the reservation is silently dropped
    while Studio's own load-mode fit still honours it."""
    assert _plan(_Stub(), extra_args = extra_args, env = env) is None
    assert _plan(_Stub()) is not None, "not vacuous"


def test_a_pass_through_adapter_is_charged_as_resident_vram(sparse_adapter, monkeypatch):
    """LoRA and control-vector sidecars are GPU-resident and in NO other term
    here: they are created on the base layer's buffer type (llama-adapter.cpp:335,
    :67) and neither mmaps (:407). `model_size` is the base GGUF alone, so an
    unpriced adapter is a plan claiming a fit that is not there."""
    adapter = sparse_adapter

    # Sized so the base load fits with room to spare and the adapter is what
    # tips it over: the term has to be load-bearing, not merely present.
    stub = _Stub()
    free = 17 * 1024
    bare = _plan(stub, free_mib = free)
    with_lora = _plan(stub, free_mib = free, extra_args = ["--lora", str(adapter)])
    assert bare is not None and with_lora is not None
    assert not bare.spills_anything, "the base load fits on its own"
    assert with_lora.spills_anything, "3 GiB of resident adapter comes out of the same VRAM"

    ctrl = _plan(stub, free_mib = free, extra_args = ["--control-vector", str(adapter)])
    assert ctrl.spilled_blocks == with_lora.spilled_blocks

    # The colon-scaled form, from a directory the path can be spelled relative to.
    # An ABSOLUTE path is not usable here: on Windows it carries a drive letter,
    # and `C:\...\adapter.gguf:0.5` is three parts to string_split, which upstream
    # rejects outright (`parts.size() != 2` -> throw, common/arg.cpp:2944-2946).
    # Pricing a token that never starts a child would be the wrong answer, so the
    # test asks for the reading that exists on every platform.
    monkeypatch.chdir(adapter.parent)
    scaled = _plan(stub, free_mib = free, extra_args = ["--lora-scaled", "adapter.gguf:0.5"])
    assert scaled.spilled_blocks == with_lora.spilled_blocks


def test_a_drive_lettered_scaled_adapter_is_skipped_rather_than_priced():
    """The other half of the same syntax. string_split on ':' gives three parts for
    `C:\\dir\\adapter.gguf:0.5`, so upstream throws and the child never starts;
    there is no placement to misprice, and guessing which colon was the separator
    would price a launch that cannot happen. Skipped, not sized, not abstained.

    No file is created: the drive-lettered form names a path that exists on no
    runner, and the point is that it is never stat'd."""
    drive_lettered = "C:\\dir\\adapter.gguf:0.5"

    stub = _Stub()
    free = 17 * 1024
    plan = _plan(stub, free_mib = free, extra_args = ["--lora-scaled", drive_lettered])
    assert plan is not None, "a form the child rejects is not the unsized-abstain case"
    assert not plan.spills_anything, "nothing priced, so the base load still fits"


def test_an_unreadable_adapter_abstains():
    """Engaged but unsized: os.stat fails, so abstain rather than plan a footprint
    short by an unknown amount. Same answer _fits_without_paging gives."""
    assert _plan(_Stub(), extra_args = ["--lora", "/no/such/adapter.gguf"]) is None
    assert _plan(_Stub()) is not None, "not vacuous"


def test_an_inherited_projector_is_charged_rather_than_ignored():
    """A projector that arrives only through LLAMA_ARG_MMPROJ is still GPU
    resident: arg.cpp applies the environment before argv (common/arg.cpp:780-802,
    the handler loop that runs ahead of parse_cli_args), and `extra_gpu_bytes`
    carries only the projector Studio itself resolved. Unbudgeted bytes here are a
    plan pinned with --fit off over a footprint that does not fit."""
    stub = _Stub()
    free = 17 * 1024
    bare = _plan(stub, free_mib = free)
    inherited = _plan(stub, free_mib = free, env_mmproj = 3 * GIB)
    assert bare is not None and inherited is not None
    assert not bare.spills_anything, "the base load fits on its own"
    assert inherited.spills_anything, "3 GiB of inherited projector comes out of the same VRAM"


def test_an_unsized_inherited_projector_abstains():
    """LLAMA_ARG_MMPROJ_URL names a download that has not happened yet (and it
    outranks even the --mmproj this launch emits: the fetch rewrites
    params.mmproj.path, common/arg.cpp:500-503, :632-634), and an unreadable path
    cannot be stat'd. Unknown bytes, so abstain rather than plan short."""
    assert _plan(_Stub(), free_mib = 14 * 1024, env_mmproj_unsized = True) is None
    assert _plan(_Stub(), free_mib = 14 * 1024) is not None, "not vacuous"


def test_the_seam_hands_the_planner_the_inherited_projector():
    """The planner can only charge what the launch path snapshots for it, and the
    load-mode fit already computes exactly these two values."""
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert '_spill_inputs["env_mmproj_unsized"]=_fit_env_mmproj_unsized' in compact
    assert '_spill_inputs["env_mmproj_bytes"]=' in compact


class _FlatAttentionStub(_Stub):
    """A layout _per_device_shortfall can actually check: every layer holds a
    cache, no sliding window, no trailing blocks. The default stub abstains on the
    per-device test, which hides what a multi-GPU plan does with a flat term."""

    def _tensor_spill_layout(
        self,
        model_path,
        *,
        all_shards = False,
    ):
        n = 8
        return ModelLayout(
            arch = "qwen35",
            n_layers = n,
            n_attention_layers = n,
            blocks = tuple(
                BlockLayout(index = i, spillable_bytes = 2 * GIB, resident_bytes = GIB // 2)
                for i in range(n)
            ),
            lm_head_bytes = 1 * GIB,
            token_embd_bytes = 512 * MIB,
            kv_bytes_per_token_f16 = 16384,
            n_ctx_train = 262144,
            complete = True,
        )


def test_a_multi_gpu_adapter_declines_rather_than_booking_it_all_on_device_0(sparse_adapter):
    """Adapter bytes are not a device-0 lump on a layer split.

    A LoRA tensor is allocated on its BASE tensor's buffer
    (`ggml_backend_buffer_get_type(model_tensor->buffer)`, llama-adapter.cpp:335)
    and a control-vector row on `model.select_buft(il)` (:67), so they follow the
    contiguous layer ranges llama.cpp hands each card. `extra_resident_bytes`, on
    the other hand, is charged entirely to device 0 by `_per_device_shortfall`, so
    a tight second card passes the per-device check on bytes it will really be
    asked to hold -- and the plan then emits --fit off, so common/fit.cpp never
    runs to catch the overflow. The base GGUF's layout carries no adapter tensor
    table, so the split cannot be derived; abstain instead.
    """
    adapter = sparse_adapter

    stub = _FlatAttentionStub()
    # 10 + 3 GiB free with a 3 GiB adapter: the deficit needs EVERY block spilled,
    # which is the only multi-GPU shape that reaches the per-device check at all
    # (a partial spill already abstains above it). Before this abstain the same
    # inputs produced a real plan -- all 8 blocks spilled, -ngl -1 --fit off
    # emitted -- with the whole 3 GiB booked on device 0 and nothing on device 1,
    # which is where the adapter rows for that card's layers actually go.
    two_cards = dict(model_size = 21 * GIB, kv = 2 * GIB, gpus = [(0, 10 * 1024), (1, 3 * 1024)])
    for flag in ("--lora", "--control-vector"):
        assert _plan(stub, extra_args = [flag, str(adapter)], **two_cards) is None

    # Only the adapter is refused: the same two-card load still reaches the
    # planner, so the abstain above is not the whole configuration being dropped.
    assert _plan(stub, **two_cards) is not None

    # Single card is unchanged: there the flat charge IS the right one, and the
    # adapter is still priced rather than ignored.
    one_card = _plan(_Stub(), free_mib = 17 * 1024, extra_args = ["--lora", str(adapter)])
    assert one_card is not None and one_card.spills_anything


def test_apple_silicon_declines_the_plan(monkeypatch):
    """Unified memory: an -ot spill moves bytes inside one pool, so the trade does
    not exist, and Metal keeps mmap zero copy through buffer_from_host_ptr. The
    seam reads this from the host, so it is the one fact the autouse fixture pins
    and the one that has to be asserted with the fixture overridden."""
    import utils.hardware

    monkeypatch.setattr(utils.hardware, "is_apple_silicon", lambda: True)
    assert _plan(_Stub(), free_mib = 14 * 1024) is None

    monkeypatch.setattr(utils.hardware, "is_apple_silicon", lambda: False)
    assert _plan(_Stub(), free_mib = 14 * 1024) is not None, "not vacuous"


def test_an_integrated_cuda_device_declines_the_plan():
    """A CUDA SoC (Jetson, DGX Spark) is unified memory and no other guard sees it.

    `_amd_apu_wants_unified_memory` answers only for ROCm, and `shared_gpu_ids`
    is filled only on Vulkan (an iGPU reports total VRAM 0), so an integrated
    CUDA device arrives with its "VRAM" credited AND host RAM credited: one pool
    counted twice. Spilling out of it frees no device memory, and the plan then
    pins that with --fit off, which is the load llama.cpp's own fitter used to
    place. The repository already classifies such a device as unified memory for
    the diffusion path (diffusion_memory.py:292-294, cudaDeviceProp::integrated
    via torch's is_integrated).
    """
    discrete = _Stub()
    soc = _Stub()
    soc._integrated_cuda = True

    # Same numbers either way, so the difference is attributable to the flag.
    assert _plan(soc, free_mib = 14 * 1024) is None
    assert _plan(discrete, free_mib = 14 * 1024) is not None, "not vacuous"


# --------------------------------- cards the installed CUDA build cannot run


def test_a_card_the_build_has_no_kernels_for_declines_the_plan(monkeypatch):
    """The SM gate is whole-host, so a mixed pair reaches the planner.

    `_cuda_sm_gate_error` proceeds as soon as ANY visible device meets the
    build's oldest supported SM, which is right for it: its refusal takes the
    whole launch to the CPU. On an sm_86 plus sm_89 pair under a bundle built for
    sm_89 only, that leaves the 3090 in the child's device list, holding its
    share of the rows by free VRAM and aborting there with "not compiled with any
    CUDA arch <= 86" (ggml-org/llama.cpp#27429). Per-device coverage is what
    decides, since ggml dispatches on info.devices[id].cc (ggml-cuda.cu:348).

    The planner cannot mask the card out of its own device set -- the child's
    visibility is pinned in load_model, long after this -- so budgeting without
    it would model a split llama.cpp is not going to perform. Decline instead,
    and fall through to --fit on.
    """
    monkeypatch.setattr(
        LlamaCppBackend,
        "_cuda_sm_uncovered_devices",
        classmethod(lambda cls, binary = None: ((1, 86),)),
    )
    pair = [(0, 24 * 1024), (1, 24 * 1024)]
    assert _plan(_Stub(), gpus = pair) is None

    # Not vacuous, and per DEVICE rather than per host: the same pair plans as
    # soon as the uncovered card is not one of the cards being budgeted, whether
    # because it is absent or because the load is pinned to the other one.
    assert _plan(_Stub(), gpus = [(0, 24 * 1024), (2, 24 * 1024)]) is not None
    assert _plan(_Stub(), gpus = pair, indices = [0]) is not None


def test_the_sm_coverage_probe_answers_per_device(monkeypatch):
    """The helper names the uncovered card while the launch gate stays open.

    Both halves matter: the gate keeps failing open on a mixed pair (refusing
    there would send a working 4090 to the CPU), and the per-device answer that
    the planner reads is the one that sees the 3090.
    """
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/x/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_llama_cuda_sms",
        staticmethod(lambda binary = None: frozenset({89})),
    )
    monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: {0: 86, 1: 89}))
    assert LlamaCppBackend._cuda_sm_gate_error() is None
    assert LlamaCppBackend._cuda_sm_uncovered_devices() == ((0, 86),)

    # A single card, covered or not, answers exactly as the gate does, so nothing
    # about the one-device launch moves.
    monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: {0: 89}))
    assert LlamaCppBackend._cuda_sm_uncovered_devices() == ()
    monkeypatch.setattr(LlamaCppBackend, "_cuda_compute_caps", staticmethod(lambda: {0: 86}))
    assert LlamaCppBackend._cuda_sm_uncovered_devices() == ((0, 86),)
    assert LlamaCppBackend._cuda_sm_gate_error() is not None

    # Unknown coverage and unknown caps both fail open.
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_llama_cuda_sms", staticmethod(lambda binary = None: None)
    )
    assert LlamaCppBackend._cuda_sm_uncovered_devices() == ()


def test_a_multi_gpu_separate_drafter_declines_rather_than_booking_it_on_device_0():
    """An unpinned drafter is distributed, not a device-0 lump.

    `common_base_params_to_speculative` copies the draft device list verbatim
    (common/speculative.cpp:2319-2331) and an EMPTY one drops
    `llama_prepare_model_devices` into its default enumeration of every visible
    GPU (llama.cpp:184-276), so the drafter's weights and its context follow the
    free-memory row split onto every card. `_mtp_reserve_bytes` rides in
    `extra_resident_bytes`, which `_per_device_shortfall` books entirely on
    device 0, so a tight second card passes the per-device check on bytes it will
    really be asked to hold -- and --fit off means common/fit.cpp never runs to
    catch the overflow. A PINNED drafter already declines on
    `_extra_args_draft_device_pin`.
    """
    stub = _FlatAttentionStub()
    # The all-blocks-spilled shape, the only multi-GPU one that reaches the
    # per-device check at all (a partial spill abstains above it). extra_gpu is
    # the drafter's own reserve, which is how the seam really carries it: the
    # snapshot folds _mtp_reserve_bytes into extra_gpu_bytes.
    two_cards = dict(
        model_size = 21 * GIB,
        kv = 2 * GIB,
        extra_gpu = 3 * GIB,
        # 9 GiB, not 10: the split reserve is charged once for the SECOND card now rather than to
        # both, so the old pair left a deficit a PARTIAL spill covered, and a partial multi-GPU
        # spill abstains before it ever reaches the per-device check this test is about.
        gpus = [(0, 9 * 1024), (1, 3 * 1024 + 256)],
    )
    # Before this abstain the same inputs produced a real plan -- every block
    # spilled, -ngl -1 --fit off emitted -- with the whole 3 GiB booked on
    # device 0 and nothing on device 1, which is where the drafter's rows for
    # that card's layers actually go.
    assert _plan(stub, separate_draft = True, **two_cards) is None

    # Only the drafter is refused.
    without = _plan(stub, **two_cards)
    assert without is not None, "the drafter is what the seam refuses, not the cards"
    if not without.spills_anything:
        # here would mean the cards were the problem and the assertion above proved nothing.
        assert "not worth it" in without.reason, without.reason
        assert "device by device" not in without.reason, without.reason

    one_card = _plan(_Stub(), free_mib = 14 * 1024, separate_draft = True)
    assert one_card is not None and one_card.spills_anything


def test_the_flat_mtp_reserve_reaches_the_spill_budget():
    """An unsized draft KV is paid for with _MTP_VRAM_RESERVE_FRAC of every card.

    The placement paths spend `_pin_fraction = _vram_frac - _MTP_VRAM_RESERVE_FRAC`
    when the byte-accurate `mtp_overhead_fn` cannot size the draft KV, and
    `_mtp_reserve_bytes` carries no replacement for the unsized part (it is 0 when
    `mtp_overhead_fn is None`). The snapshot has to hand the planner that same
    reduced budget, or the plan spends 5% of VRAM the fit deliberately kept free
    and then pins it with --fit off.
    """
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert '"gpu_usable_mib":{_idx:max(0.0,_gpu_usable((_idx,_free),_pin_fraction))' in compact

    # And the budget really is what the planner spends: the same load plans a
    # bigger spill on the smaller budget.
    stub = _Stub()
    generous = _plan(stub, free_mib = 24 * 1024, usable_mib = 15 * 1024)
    reserved = _plan(stub, free_mib = 24 * 1024, usable_mib = 14 * 1024)
    assert generous is not None and reserved is not None
    assert len(reserved.spilled_blocks) > len(generous.spilled_blocks)


def test_the_seam_computes_a_per_layer_kv_vector():
    """The data already existed on the backend (_sliding_window_pattern); it just
    never crossed into the planner. Only the RATIOS travel: full attention holds
    the whole context, a window layer its window, ~100x apart at 128K vs 1K."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._n_layers = 6
    b._n_kv_heads = 8
    b._n_heads = 8
    b._kv_key_length = 128
    b._kv_value_length = 128
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    b._sliding_window = 1024
    b._sliding_window_pattern = [False, True, True, True, True, True]
    b._n_kv_heads_by_layer = None
    b._shared_kv_layers = None
    # None/None picks _estimate_kv_cache_bytes path 3, the one this describes.
    b._kv_lora_rank = None
    b._ssm_inner_size = None
    b._full_attention_interval = None

    weights = b._kv_layer_weights(131072)
    assert len(weights) == 6
    assert weights[0] == max(weights), "the full-attention layer is the big one"
    # The compact SWA allowance is window + one micro-batch, padded to 256 cells,
    # exactly as _estimate_kv_cache_bytes sizes the total it is scaled to.
    assert weights[0] // weights[1] == 131072 // (1024 + 512)

    # No pattern, no window, or no dimensions: say nothing rather than guess.
    b._sliding_window_pattern = None
    assert b._kv_layer_weights(131072) == []
    b._sliding_window_pattern = [False, True, True, True, True, True]
    b._sliding_window = 0
    assert b._kv_layer_weights(131072) == []


def _swa_backend(n_layers = 6, shared = None):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._n_layers = n_layers
    b._n_kv_heads = 8
    b._n_heads = 8
    b._kv_key_length = 128
    b._kv_value_length = 128
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    b._sliding_window = 1024
    b._sliding_window_pattern = [i % 2 == 1 for i in range(n_layers)]
    b._n_kv_heads_by_layer = None
    b._shared_kv_layers = shared
    b._kv_lora_rank = None
    b._ssm_inner_size = None
    b._full_attention_interval = None
    return b


def test_shared_kv_layers_carry_no_weight():
    """Gemma 3n / Gemma 4 reuse an earlier layer's cache for the trailing
    ``attention.shared_kv_layers`` blocks: has_kv is false past
    n_layer_kv_from_start (llama-hparams.cpp:275-279), so they allocate nothing.
    Giving them weight spreads the same byte total over layers that hold no cache
    and under-books whichever card draws the real ones."""
    plain = _swa_backend(shared = None)
    gemma = _swa_backend(shared = 2)

    w = gemma._kv_layer_weights(131072)
    assert len(w) == 6, "length must still match layout.n_layers or the vector is dropped"
    assert w[4] == 0 and w[5] == 0, "the shared-KV tail allocates no cache"
    assert w[:4] == plain._kv_layer_weights(131072)[:4], "the cached layers are unchanged"

    # A GGUF claiming every layer is shared must not zero the vector out.
    assert any(_swa_backend(shared = 99)._kv_layer_weights(131072))


def test_swa_weights_follow_the_effective_cache_geometry():
    """The vector places a total priced with the LAUNCH settings, so it has to
    describe the same geometry. --swa-full makes an SWA layer hold the full
    context, and the compact allowance is per-slot plus a micro-batch, not
    min(window, n_ctx). A window-sized ratio against a full-context total moves
    cache bytes onto the wrong card under the --fit off the plan pins."""
    b = _swa_backend()

    full = b._kv_layer_weights(131072, swa_full = True)
    assert len(set(full)) == 1, "--swa-full: every layer holds the full context"

    # Slots multiply the compact window allowance in unified mode.
    one = b._kv_layer_weights(131072, n_parallel = 1)
    four = b._kv_layer_weights(131072, n_parallel = 4)
    assert four[1] > one[1], "4 slots hold 4 windows' worth of SWA cells"
    assert four[0] == one[0], "the full-attention layer is unaffected in unified mode"

    # A bigger micro-batch is real headroom on the SWA cache too.
    assert b._kv_layer_weights(131072, n_ubatch = 4096)[1] > one[1]


def test_the_seam_passes_the_launch_geometry_to_the_kv_vector():
    """Reachability: the knobs must travel from the launch path, or the two fixes
    above only hold in a unit test."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    call = src[src.index('"kv_layer_weights"') :][:400]
    compact = "".join(call.split())
    assert "swa_full=swa_full" in compact
    assert "n_parallel=n_parallel" in compact
    assert "kv_unified=planned_kv_unified" in compact
    assert "n_ubatch=_effective_ubatch" in compact


def test_an_inherited_split_mode_none_declines():
    """An env -sm none may never reach the child: on a layer-split launch
    load_model pops LLAMA_ARG_SPLIT_MODE (plus the paired LLAMA_ARG_TENSOR_SPLIT)
    for any non-"layer" mode (llama_cpp.py:20452-20458), so the child runs the
    default layer split. Budgeting the single main GPU there plans against a mode
    the child never sees AND skips the per-device check, which short-circuits on
    one device. argv is the only spelling that provably survives."""
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_SPLIT_MODE": "none"}) is None
    # argv still plans, and still against one card.
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-sm", "none"]) is not None
    # An argv override of the env is argv, so it plans.
    assert (
        _plan(
            _Stub(),
            gpus = two_cards,
            extra_args = ["-sm", "none"],
            env = {"LLAMA_ARG_SPLIT_MODE": "none"},
        )
        is not None
    )
    # Not vacuous, and an inherited "layer" is the default and costs nothing.
    assert _plan(_Stub(), gpus = two_cards) is not None
    assert _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_SPLIT_MODE": "layer"}) is not None


def test_a_reconciliation_still_strips_a_non_layer_split_mode_env():
    """Reachability anchor for the decline above: if load_model ever stops
    scrubbing the inherited mode, the env form becomes plannable again and this is
    the test that should fail and say so."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    compact = "".join(src.split())
    assert 'if_inherited_smand_inherited_sm!="layer":' in compact
    assert 'env.pop("LLAMA_ARG_SPLIT_MODE",None)' in compact
    # The paired -ts goes with it, which is what the test below relies on.
    assert 'env.pop("LLAMA_ARG_SPLIT_MODE",None)env.pop("LLAMA_ARG_TENSOR_SPLIT",None)' in compact


def test_an_inherited_tensor_split_scrubbed_with_its_mode_is_not_planned_against():
    """The -ts mirror of the decline above. An inherited LLAMA_ARG_TENSOR_SPLIT is
    popped together with a non-layer LLAMA_ARG_SPLIT_MODE, so with `-sm layer` on
    argv the child ends up with NEITHER and splits the rows by free VRAM. Proving
    row ownership against the scrubbed 9:1 shares can approve a device that then
    overflows under --fit off."""
    import core.inference.offload_planner as mod

    two_cards = [(0, 8 * 1024), (1, 16 * 1024)]
    seen = {}
    real = mod.plan_placement

    def spy(*a, **kw):
        seen["w"] = kw.get("split_weights_per_device")
        return real(*a, **kw)

    mod.plan_placement = spy
    try:
        _plan(
            _Stub(),
            gpus = two_cards,
            extra_args = ["-sm", "layer"],
            env = {"LLAMA_ARG_SPLIT_MODE": "row", "LLAMA_ARG_TENSOR_SPLIT": "9,1"},
        )
        scrubbed = seen.pop("w", None)
        _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_TENSOR_SPLIT": "9,1"})
        survives = seen.pop("w", None)
    finally:
        mod.plan_placement = real

    assert scrubbed == [8 * 1024 * MIB, 16 * 1024 * MIB], "the child never sees that -ts"
    # Not vacuous: with no inherited mode to drag it out, the float shares reach
    # the row model exactly as llama.cpp parsed them.
    assert survives == [9.0, 1.0]


@pytest.mark.parametrize("value", ["nan,1", "inf,1", "1,nan", "-nan,1", "infinity,1"])
def test_a_non_finite_tensor_split_declines_instead_of_raising(value):
    """float() accepts "nan"/"inf", and NaN then passes BOTH remaining guards:
    every comparison against NaN is false, and sum() of a NaN list is NaN. It
    reaches int(round(p * scale)), which raises ValueError / OverflowError with no
    try around the _planned_tensor_spill call, aborting the load -- while
    llama.cpp merely degenerates on the same value."""
    assert _extra_args_tensor_split(["-ts", value], {}) is None
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    # Declines (unparseable -ts is not "no -ts"), and above all does not raise.
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", value]) is None
    assert _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_TENSOR_SPLIT": value}) is None
    # Not vacuous: finite shares still parse and still plan.
    assert _extra_args_tensor_split(["-ts", "3,1"], {}) == [3.0, 1.0]
    # llama.cpp folds the underscore, so the reader must see this spelling too.
    assert _extra_args_tensor_split(["--tensor_split", "3,1"], {}) == [3.0, 1.0]
    assert _extra_args_tensor_split(["--tensor_split=3,1"], {}) == [3.0, 1.0]
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", "3,1"]) is not None


def test_a_cumulative_float32_tensor_split_overflow_declines_instead_of_raising():
    value = "3e38,3e38"
    assert _extra_args_tensor_split(["-ts", value], {}) is None
    two_cards = [(0, 14 * 1024), (1, 14 * 1024)]
    assert _plan(_Stub(), gpus = two_cards, extra_args = ["-ts", value]) is None
    assert _plan(_Stub(), gpus = two_cards, env = {"LLAMA_ARG_TENSOR_SPLIT": value}) is None


def test_the_vector_refuses_a_cache_the_estimator_prices_on_another_path():
    """_estimate_kv_cache_bytes picks its path BEFORE it looks at the window, and
    the earlier paths price a different quantity: path 1 (MLA) caches one
    compressed K latent per layer with no V and no window/full split, path 2
    (hybrid recurrent) caches only 1 in full_attention_interval layers. Either way
    a window-shaped vector is a different model of the cache, so it must answer []
    and let the planner abstain.

    Not hypothetical for path 1: dots3note reads the MLA head lengths and
    ATTENTION_SLIDING_WINDOW(_PATTERN) in the same loader. Path 2 is the
    recurrent-hybrid hole -- the abstain is `uneven_cache and not weights`, so a
    hybrid that ever produced a vector would walk past it and the device loop
    never places layout.recurrent_bytes."""
    b = _swa_backend()
    assert b._kv_layer_weights(131072), "the plain SWA model still answers"

    mla = _swa_backend()
    mla._kv_lora_rank = 512
    mla._key_length_mla = 192
    mla._value_length_mla = 128
    assert mla._kv_layer_weights(131072) == []

    hybrid = _swa_backend()
    hybrid._ssm_inner_size = 4096
    hybrid._full_attention_interval = 4
    assert hybrid._kv_layer_weights(131072) == []

    # One of the two alone is not the hybrid path and must not cost the vector.
    half = _swa_backend()
    half._ssm_inner_size = 4096
    assert half._kv_layer_weights(131072)


def test_a_recurrent_hybrid_stays_on_the_abstain_path():
    """With no vector the recurrent abstain in _per_device_shortfall fires, so
    recurrent_bytes is never silently left unplaced by the device loop."""
    from core.inference.offload_planner import PlanOptions, _per_device_shortfall

    layout = ModelLayout(
        arch = "qwen35moe",
        n_layers = 4,
        n_attention_layers = 1,
        blocks = tuple(
            BlockLayout(index = i, spillable_bytes = GIB, resident_bytes = GIB // 4) for i in range(4)
        ),
        lm_head_bytes = GIB // 2,
        kv_bytes_per_token_f16 = 65536,
        recurrent_bytes = 2 * GIB,
        complete = True,
    )
    why = _per_device_shortfall(
        layout,
        PlanOptions(),
        32768,
        set(),
        False,
        [8 * GIB, 8 * GIB],
        quantised = False,
        kv_bytes_floor = 0,
        kv_layer_weights = (),
    )
    assert why is not None and "recurrent" in why


def test_flash_disabled_v_padding_reaches_the_layer_weights():
    """With flash attention OFF llama.cpp cannot keep a ragged V cache: every
    layer's V is padded to hparams.n_embd_v_gqa_max() over the whole model, which
    is what _estimate_kv_cache_bytes charges via _max_kv_value_width. V goes
    constant while K stays per-layer, so an unpadded vector prices a ratio the
    total does not have. Reached whenever the launch runs FA off: a build without
    --flash-attn, a typed -fa off, or the hard-crash recovery, which revokes the
    plan and re-fits."""
    b = _swa_backend()
    # SWA layers wider than global ones, so the model-wide max is the SWA width
    # and the padding actually moves: n_embd_v_gqa_max = 8 * 256.
    b._kv_key_length_swa = 256
    b._kv_value_length_swa = 256

    assert b._max_kv_value_width(128, 256) == 8 * 256

    on = b._kv_layer_weights(131072, flash_attn = True)
    off = b._kv_layer_weights(131072, flash_attn = False)
    assert on and off
    assert off != on, "the padding has to reach the vector"

    # Exactly the estimator's own per-layer arithmetic, f16 both axes.
    full_cells, swa_cells = 131072, 1024 + 512
    pad = 8 * 256
    glob = (8 * 128 * 2 + pad * 2) * full_cells  # layer 0: global
    swa = (8 * 256 * 2 + pad * 2) * swa_cells  # layer 1: sliding window
    assert off[0] == glob and off[1] == swa

    # A quantised K cache floors bpe_v at f16 on the same branch, and with V
    # constant across layers that asymmetry moves the ratio too.
    from core.inference.llama_cpp import _kv_bytes_per_elem

    bpe_k = _kv_bytes_per_elem("q8_0")
    q8 = b._kv_layer_weights(131072, flash_attn = False, cache_type_kv = "q8_0")
    assert q8 != off
    assert q8[0] == int((8 * 128 * bpe_k + pad * 2) * full_cells)


def test_the_seam_passes_the_flash_state_and_cache_type_to_the_kv_vector():
    """Reachability: both knobs must travel from the launch path too.

    load_model is far too large to drive from here, so this reads the call site
    through the AST: matching source text would pass on a mention in a comment and
    fail on a reflow, neither of which says what the seam passes.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_kv_layer_weights"
    ]
    assert calls, "the launch path no longer builds a KV layer vector"

    for call in calls:
        passed = {
            kw.arg: kw.value for kw in call.keywords if kw.arg and isinstance(kw.value, ast.Name)
        }
        assert getattr(passed.get("flash_attn"), "id", None) == "planned_flash_attn"
        assert getattr(passed.get("cache_type_kv"), "id", None) == "cache_type_kv"


# ------------------------------- the two inputs the planner was getting wrong


def test_a_single_card_pays_no_split_reserve():
    """The pipeline reserve is a LAYER-SPLIT cost, so one card owes none of it.

    Every other use of _PIPELINE_PER_DEVICE_OVERHEAD_MIB in llama_cpp.py applies
    it as ``max(0, n_gpus - 1) *`` or guards it behind ``n > 1``; the planner
    folded it into its flat per-device term, so it came off device 0 as well.
    That withheld a GiB of a single card no split was ever going to allocate, and
    the planner then spilled real blocks to cover the deficit it had invented --
    the "spilled into headroom that was there" cells in #9861.

    Asserted through the plan rather than by reading the constant: a load sized
    to fit in exactly the disputed GiB must come back resident.
    """
    from core.inference.offload_planner import ContextPolicy, PlanOptions, plan_placement

    layout = _Stub()._tensor_spill_layout("/models/stub.gguf")
    opts = PlanOptions(
        overhead_bytes_per_device = 0,
        pipeline_overhead_bytes = 1 * GIB,
        context_policy = ContextPolicy.NEVER_REDUCE,
    )
    resident = sum(b.resident_bytes for b in layout.blocks) + layout.token_embd_bytes
    spillable = sum(b.spillable_bytes for b in layout.blocks)
    need = resident + spillable + layout.lm_head_bytes

    one = plan_placement(layout, [need + 512 * MIB], 64 * GIB, 4096, opts = opts)
    assert not one.spills_anything, (
        "a single card paid a split reserve it does not owe, so a load that fits "
        f"was spilled anyway: {one.reason}"
    )

    # The term is not simply deleted: it is charged once per device AFTER the
    # first. Asserted on the budget arithmetic directly, because the plan-level
    # answer for two cards is dominated by the partial-spill abstain and would
    # read the same whether the reserve was charged or not.
    from core.inference.offload_planner import _usable_vram

    card = 8 * GIB
    short = opts.overhead_free_ctx
    assert _usable_vram([card], opts, short) == card
    assert _usable_vram([card, card], opts, short) == 2 * card - 1 * GIB
    assert _usable_vram([card, card, card], opts, short) == 3 * card - 2 * GIB

    # And the context-linear term is charged on every card.
    over = 4096
    step = over * opts.overhead_bytes_per_token
    assert _usable_vram([card], opts, short + over) == card - step
    assert _usable_vram([card, card], opts, short + over) == 2 * card - 1 * GIB - 2 * step
    assert _usable_vram([card], opts, short // 2) == card


def test_the_cost_model_is_told_physical_cores_not_hyperthreads(monkeypatch):
    """Spilled decode gets physical cores, so the penalty must be priced on them.

    Studio leaves --threads unset on purpose, and an unset --threads makes
    llama.cpp size its pool from common_cpu_get_num_math, which counts physical
    cores. Passing os.cpu_count() told the cost model a 6-core / 12-thread
    desktop had 12, and the generation penalty came out roughly half of what a
    spill really costs there.
    """
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: None)
    assert llama_mod._spilled_decode_threads() >= 1

    class _FakePsutil:
        @staticmethod
        def cpu_count(logical = True):
            return 12 if logical else 6

    saved = sys.modules.get("psutil")
    sys.modules["psutil"] = _FakePsutil
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(12)), raising = False
    )
    try:
        assert llama_mod._spilled_decode_threads() == 6, "logical count reached the cost model"
    finally:
        if saved is None:
            del sys.modules["psutil"]
        else:
            sys.modules["psutil"] = saved

    # Match llama.cpp's own last resort when physical topology is unavailable.
    class _NoAnswer:
        @staticmethod
        def cpu_count(logical = True):
            return None

    sys.modules["psutil"] = _NoAnswer
    try:
        for logical, expected in ((2, 2), (4, 4), (8, 4), (16, 8), (None, 4)):
            monkeypatch.setattr(llama_mod.os, "cpu_count", lambda answer = logical: answer)
            monkeypatch.setattr(
                llama_mod.os,
                "sched_getaffinity",
                lambda _pid, answer = logical: set(range(answer or 1)),
            )
            assert llama_mod._spilled_decode_threads() == expected
    finally:
        if saved is None:
            del sys.modules["psutil"]
        else:
            sys.modules["psutil"] = saved


def test_thread_overrides_reach_the_cost_model(monkeypatch):
    seen = []

    def _threads(n_threads = None, extra_args = None):
        seen.append((n_threads, extra_args))
        return int(n_threads or 8)

    monkeypatch.setattr(llama_mod, "_spilled_decode_threads", _threads)
    plan = _plan(
        _Stub(),
        free_mib = 14 * 1024,
        n_threads = 3,
        extra_args = ["--threads", "2"],
        env = {"LLAMA_ARG_THREADS": "1"},
    )

    assert plan is not None and plan.spills_anything
    expected = [(3, ["--threads", "2"])]
    assert seen == expected


def test_thread_override_precedence_matches_the_launched_command(monkeypatch):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: None)

    class _FakePsutil:
        @staticmethod
        def cpu_count(logical = True):
            return 12 if logical else 6

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(12)), raising = False
    )
    assert llama_mod._spilled_decode_threads() == 6
    assert llama_mod._spilled_decode_threads(3) == 3
    assert llama_mod._spilled_decode_threads(3, ["--threads", "4"]) == 4
    assert llama_mod._spilled_decode_threads(3, ["-t=5", "--threads=1"]) == 1


def test_smt_workers_do_not_multiply_math_core_capacity(monkeypatch):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: None)

    monkeypatch.setitem(sys.modules, "psutil", _SmtHost)
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(16)), raising = False
    )
    monkeypatch.setattr(llama_mod.os, "cpu_count", lambda: 16)
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "4"]) == 4
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "16"]) is None
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "-1"]) is None
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "0"]) is None

    class _NonSmtHost:
        @staticmethod
        def cpu_count(logical = True):
            return 8

    monkeypatch.setitem(sys.modules, "psutil", _NonSmtHost)
    monkeypatch.setattr(llama_mod.os, "cpu_count", lambda: 8)
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "-1"]) == 8


def test_default_thread_override_uses_native_logical_count(monkeypatch):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: 8)
    monkeypatch.setattr(llama_mod.os, "cpu_count", lambda: 2)
    monkeypatch.setitem(sys.modules, "psutil", _SmtHost)
    assert llama_mod._spilled_decode_threads(extra_args = ["--threads", "-1"]) is None


def test_inherited_linux_cpu_affinity_declines_spill_pricing(monkeypatch):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: 8)
    monkeypatch.setattr(llama_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        llama_mod.os,
        "uname",
        lambda: SimpleNamespace(machine = "x86_64"),
        raising = False,
    )
    monkeypatch.setattr(llama_mod.os, "sched_getaffinity", lambda _pid: {0, 1}, raising = False)
    monkeypatch.setitem(sys.modules, "psutil", _SmtHost)
    assert llama_mod._spilled_decode_threads() is None

    monkeypatch.setattr(llama_mod.os, "sched_getaffinity", lambda _pid: set(range(16)))
    assert llama_mod._spilled_decode_threads() == 8


def test_inherited_linux_arm_cpu_affinity_declines_spill_pricing(monkeypatch):
    class _ArmHost:
        @staticmethod
        def cpu_count(logical = True):
            return 72 if logical else 36

    monkeypatch.setattr(llama_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        llama_mod.os,
        "uname",
        lambda: SimpleNamespace(machine = "aarch64"),
        raising = False,
    )
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(4)), raising = False
    )
    monkeypatch.setitem(sys.modules, "psutil", _ArmHost)
    assert llama_mod._spilled_decode_threads() is None


def test_oversubscribed_decode_threads_decline_spill_planning(monkeypatch):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: 8)
    monkeypatch.setattr(llama_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        llama_mod.os,
        "uname",
        lambda: SimpleNamespace(machine = "x86_64"),
        raising = False,
    )
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(16)), raising = False
    )
    assert _plan(_Stub(), free_mib = 14 * 1024, extra_args = ["--threads", "16"]) is None
    plan = _plan(
        _Stub(),
        free_mib = 14 * 1024,
        extra_args = ["--threads", "4"],
    )
    assert plan is not None and plan.spills_anything


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--cpu-range", "0-2", "--cpu-strict", "1"],
        ["-C", "0x3"],
        ["--cpu-mask=0x3"],
    ],
)
def test_affinity_constrained_decode_declines_spill_planning(monkeypatch, extra_args):
    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: 8)
    assert _plan(_Stub(), free_mib = 14 * 1024, extra_args = extra_args) is None


def test_linux_hybrid_math_cores_exclude_efficiency_cores(tmp_path):
    sibling_sets = [f"{core},{core + 8}" for core in range(8)]
    sibling_sets.extend(f"{core},{core + 8}" for core in range(8))
    sibling_sets.extend(str(cpu) for cpu in range(16, 32))
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = tmp_path / f"cpu{cpu}"
        (cpu_path / "topology").mkdir(parents = True)
        (cpu_path / "topology" / "thread_siblings").write_text(sibling_set)
        capacity = 1024 if cpu < 16 else 512
        (cpu_path / "cpu_capacity").write_text(str(capacity))

    assert _linux_math_core_count(tmp_path, logical_cpus = 32, vendor_id = "GenuineIntel") == 8


def test_linux_hybrid_pmu_excludes_efficiency_cores_without_capacity(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    sibling_sets = [f"{core},{core + 4}" for core in range(4)]
    sibling_sets.extend(f"{core},{core + 4}" for core in range(4))
    sibling_sets.extend(str(cpu) for cpu in range(8, 16))
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(sibling_set)
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0-7")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("8-15")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 4
    )


@pytest.mark.parametrize("source", ["pmu", "capacity"])
def test_linux_no_smt_hybrid_matches_llama_cpu_loop(tmp_path, source):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    for cpu in range(8):
        cpu_path = cpu_root / f"cpu{cpu}"
        (cpu_path / "topology").mkdir(parents = True)
        (cpu_path / "topology" / "thread_siblings").write_text(str(cpu))
        if source == "capacity":
            (cpu_path / "cpu_capacity").write_text("1024" if cpu < 4 else "512")
    if source == "pmu":
        (event_root / "cpu_core").mkdir(parents = True)
        (event_root / "cpu_core" / "cpus").write_text("0-3")
        (event_root / "cpu_atom").mkdir()
        (event_root / "cpu_atom" / "cpus").write_text("4-7")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 2
    )


def test_linux_sparse_online_hybrid_matches_llama_physical_fallback(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    sibling_sets = [f"{cpu},{cpu + 8}" for cpu in range(8)]
    sibling_sets.extend(f"{cpu - 8},{cpu}" for cpu in range(8, 16))
    sibling_sets.extend(str(cpu) for cpu in range(16, 24))
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(sibling_set)
    (cpu_root / "online").write_text("0-7,16-23")
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0-7")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("16-23")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 16
    )


def test_linux_smt_disabled_hybrid_skips_offline_siblings(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    sibling_sets = [f"{cpu // 2 * 2},{cpu // 2 * 2 + 1}" for cpu in range(16)]
    sibling_sets.extend(str(cpu) for cpu in range(16, 24))
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(sibling_set)
    (cpu_root / "online").write_text("0,2,4,6,8,10,12,14,16-23")
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0,2,4,6,8,10,12,14")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("16-23")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 8
    )


def test_linux_hybrid_unpinnable_cpu_matches_llama_physical_fallback(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    sibling_sets = [f"{cpu // 2 * 2},{cpu // 2 * 2 + 1}" for cpu in range(16)]
    sibling_sets.extend(str(cpu) for cpu in range(16, 24))
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(sibling_set)
    (cpu_root / "online").write_text("0-23")
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0-15")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("16-23")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
            pinnable_cpus = set(range(8)),
        )
        == 16
    )


def test_linux_hybrid_affinity_probe_failure_matches_physical_fallback(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    for cpu in range(8):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(str(cpu))
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0-3")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("4-7")

    def fail_affinity(_pid):
        raise OSError("unavailable")

    monkeypatch.setattr(llama_mod.os, "sched_getaffinity", fail_affinity, raising = False)
    monkeypatch.setattr(llama_mod.os, "sched_setaffinity", lambda _pid, _cpus: None, raising = False)
    assert (
        llama_mod._linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
            probe_affinity = True,
        )
        == 8
    )


def test_linux_hybrid_affinity_restore_failure_stays_in_worker(tmp_path, monkeypatch):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    for cpu in range(8):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(str(cpu))
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("0-3")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("4-7")
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(8)), raising = False
    )
    caller_thread = llama_mod.threading.get_ident()
    affinity_threads = []

    def restore_fails(_pid, cpus):
        affinity_threads.append(llama_mod.threading.get_ident())
        if len(cpus) > 1:
            raise OSError("restore failed")

    monkeypatch.setattr(llama_mod.os, "sched_setaffinity", restore_fails, raising = False)
    assert (
        llama_mod._linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
            probe_affinity = True,
        )
        == 2
    )
    assert affinity_threads
    assert all(thread != caller_thread for thread in affinity_threads)


def test_linux_hybrid_with_unreadable_core_mask_is_conservative(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    for cpu in range(4):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(str(cpu))
    (event_root / "cpu_atom").mkdir(parents = True)
    (event_root / "cpu_atom" / "cpus").write_text("2-3")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 1
    )


def test_linux_hybrid_with_only_efficiency_cores_matches_llama_fallback(tmp_path):
    cpu_root = tmp_path / "cpu"
    event_root = tmp_path / "events"
    for cpu in range(4):
        cpu_path = cpu_root / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(str(cpu))
    (event_root / "cpu_core").mkdir(parents = True)
    (event_root / "cpu_core" / "cpus").write_text("")
    (event_root / "cpu_atom").mkdir()
    (event_root / "cpu_atom" / "cpus").write_text("0-3")

    assert (
        _linux_math_core_count(
            cpu_root,
            vendor_id = "GenuineIntel",
            event_source_root = event_root,
        )
        == 4
    )


def test_linux_topology_ignores_python_cpu_count_override(tmp_path, monkeypatch):
    for cpu in range(4):
        cpu_path = tmp_path / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(str(cpu))
    monkeypatch.setattr(llama_mod.os, "cpu_count", lambda: 2)

    assert llama_mod._linux_math_core_count(tmp_path, vendor_id = "AuthenticAMD") == 4


def test_linux_non_hybrid_topology_excludes_offline_cores(tmp_path):
    sibling_sets = ["0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"]
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = tmp_path / f"cpu{cpu}" / "topology"
        cpu_path.mkdir(parents = True)
        (cpu_path / "thread_siblings").write_text(sibling_set)
    (tmp_path / "online").write_text("0-2,4-6")

    assert _linux_math_core_count(tmp_path, vendor_id = "AuthenticAMD") == 3


def test_linux_amd_capacity_classes_keep_all_physical_cores(tmp_path):
    for cpu in range(24):
        cpu_path = tmp_path / f"cpu{cpu}"
        (cpu_path / "topology").mkdir(parents = True)
        (cpu_path / "topology" / "thread_siblings").write_text(str(cpu))
        (cpu_path / "cpu_capacity").write_text("1024" if cpu < 8 else "512")

    assert _linux_math_core_count(tmp_path, logical_cpus = 24, vendor_id = "AuthenticAMD") == 24


@pytest.mark.parametrize(
    ("sibling_sets", "expected"),
    [
        (["0,4", "1,5", "2,6", "3,7", "0,4", "1,5", "2,6", "3,7"], 4),
        ([str(cpu) for cpu in range(8)], 8),
    ],
)
def test_linux_smt_and_non_smt_core_counts(tmp_path, sibling_sets, expected):
    for cpu, sibling_set in enumerate(sibling_sets):
        cpu_path = tmp_path / f"cpu{cpu}"
        (cpu_path / "topology").mkdir(parents = True)
        (cpu_path / "topology" / "thread_siblings").write_text(sibling_set)

    assert _linux_math_core_count(tmp_path, logical_cpus = len(sibling_sets)) == expected


def test_invalid_linux_topology_falls_back_to_psutil(monkeypatch):
    class _PhysicalHost:
        @staticmethod
        def cpu_count(logical = True):
            return 24 if logical else 12

    monkeypatch.setattr(llama_mod, "_linux_math_core_count", lambda: None)
    monkeypatch.setattr(
        llama_mod.os, "sched_getaffinity", lambda _pid: set(range(24)), raising = False
    )
    monkeypatch.setitem(sys.modules, "psutil", _PhysicalHost)
    assert llama_mod._spilled_decode_threads() == 12


def test_the_seam_scores_at_the_micro_batch_that_launches():
    """rank() amortises the spilled-weight stream over ONE ubatch."""
    seen = {}

    def _capture(*args, **kwargs):
        seen["opts"] = kwargs["opts"]
        raise AssertionError("stop after the options are built")

    import core.inference.offload_planner as planner_mod

    real = planner_mod.plan_placement
    planner_mod.plan_placement = _capture
    try:
        for launched, expected in ((64, 64), (2048, 2048), (None, 512), (0, 512)):
            seen.clear()
            with pytest.raises(AssertionError):
                _plan(_Stub(), free_mib = 14 * 1024, n_ubatch = launched)
            assert seen["opts"].n_ubatch == expected, launched
    finally:
        planner_mod.plan_placement = real


def test_an_embedding_server_is_scored_without_a_decode_phase():
    """``--embedding`` returns the pooled vector; there is no generation at all, so a spill's
    decode advantage, which on a routed MoE is its ENTIRE advantage since experts are charged
    ``n_expert_used / n_expert`` for generation but full bytes for prefill, is winnings this
    workload can never collect."""
    seen = {}

    def _capture(*args, **kwargs):
        seen["opts"] = kwargs["opts"]
        raise AssertionError("stop after the options are built")

    import core.inference.offload_planner as planner_mod

    real = planner_mod.plan_placement
    planner_mod.plan_placement = _capture
    try:
        generative = _Stub()
        with pytest.raises(AssertionError):
            _plan(generative, free_mib = 14 * 1024)
        assert seen["opts"].workload_generated_tokens > 0

        embedder = _Stub()
        embedder.is_embedding_gguf = True
        seen.clear()
        with pytest.raises(AssertionError):
            _plan(embedder, free_mib = 14 * 1024)
        assert seen["opts"].workload_generated_tokens == 0
        assert seen["opts"].workload_prompt_tokens > 0, "prefill is the whole workload here"
    finally:
        planner_mod.plan_placement = real


def _captured_opts(monkeypatch, stub, **kw):
    """Run the seam and hand back the PlanOptions it built, plus the kw it passed."""
    from core.inference import offload_planner

    seen = {}

    def capture(layout, vram, host_ram, n_ctx, **kwargs):
        seen["kwargs"] = kwargs
        seen["n_ctx"] = n_ctx
        return Plan(reason = "captured")

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    extra_args = kw.pop("extra_args", None)
    # Popped, not passed through as an input: the seam reads the environment itself.
    env = {"UNSLOTH_SMART_OFFLOAD": "1", **(kw.pop("env", None) or {})}
    inputs = _inputs(**{k: v for k, v in kw.items() if k in _inputs.__code__.co_varnames})
    inputs.update({k: v for k, v in kw.items() if k not in _inputs.__code__.co_varnames})
    stub._planned_tensor_spill(inputs, extra_args = extra_args, env = env)
    assert "kwargs" in seen, "the seam declined before reaching the planner"
    return seen["kwargs"]["opts"], seen


@pytest.mark.parametrize(
    "env, extra_args",
    [
        ({}, None),
        ({"UNSLOTH_SMART_OFFLOAD": "1"}, ["-ot", "ffn=CPU"]),
        ({"UNSLOTH_SMART_OFFLOAD": "1"}, ["--fit", "on"]),
        ({"UNSLOTH_SMART_OFFLOAD": "1"}, ["-ngl", "20"]),
        ({"UNSLOTH_SMART_OFFLOAD": "1", "LLAMA_ARG_N_GPU_LAYERS": "3"}, None),
        ({"UNSLOTH_SMART_OFFLOAD": "1"}, ["--rpc", "host:1"]),
        ({"UNSLOTH_SMART_OFFLOAD": "1"}, ["--fit-target", "2048"]),
    ],
)
def test_the_may_run_predicate_agrees_with_the_seam_on_every_decline(env, extra_args):
    """load_model asks the pure predicate BEFORE the priced inputs exist, so the two must
    answer alike: a launch the seam declines is one the predicate must refuse too, or
    load_model prices the pre-cap context and moves the projector for a planner that never runs."""
    assert LlamaCppBackend._planner_may_run(extra_args, env) is False
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = extra_args, env = env) is None
    assert LlamaCppBackend._planner_may_run(None, {"UNSLOTH_SMART_OFFLOAD": "1"}) is True


def test_the_floor_map_and_the_knob_inputs_reach_the_planner(monkeypatch):
    """Everything rungs 0 to 2 need, priced by the seam and handed over intact."""
    floors = {1: 512 * MIB, 2: 1 * GIB, 3: 3 * GIB // 2, 4: 2 * GIB}
    opts, seen = _captured_opts(
        monkeypatch,
        _Stub(),
        free_mib = 14 * 1024,
        n_parallel = 4,
        min_parallel = 1,
        kv_bytes_floor_by_parallel = floors,
        workload_prompt_tokens = 4096,
        context_policy_fit_only = True,
        min_ctx = 8192,
        cache_ram_default_mib = 3000,
    )
    assert dict(opts.kv_bytes_floor_by_parallel) == floors
    assert opts.n_parallel == 4 and opts.min_parallel == 1
    assert opts.workload_prompt_tokens == 4096
    assert opts.context_policy is ContextPolicy.FIT_ONLY and opts.min_ctx == 8192
    assert opts.cache_ram_default_mib == 3000
    # The scalar floor is the map's entry at the launched count.
    assert seen["kwargs"]["kv_bytes_floor"] == _inputs()["kv_cache_bytes"]


def test_a_caller_that_prices_no_knob_gets_the_old_options(monkeypatch):
    """Every new input defaults to "not supplied", so a snapshot from before
    these rungs existed plans exactly as it did."""
    opts, _ = _captured_opts(monkeypatch, _Stub(), free_mib = 14 * 1024, n_parallel = 2)
    assert opts.n_parallel == 2 and opts.min_parallel == 1
    assert opts.kv_bytes_floor_by_parallel == {}
    assert opts.mmproj_bytes == 0 and opts.mmproj_movable is False
    assert opts.draft_bytes == 0 and opts.draft_droppable is False
    assert opts.context_policy is ContextPolicy.NEVER_REDUCE
    assert opts.workload_prompt_tokens == 2048


def test_min_parallel_never_exceeds_the_priced_count(monkeypatch):
    opts, _ = _captured_opts(monkeypatch, _Stub(), n_parallel = 2, min_parallel = 6)
    assert opts.min_parallel == 2


def test_a_movable_projector_is_taken_out_of_the_fused_terms(monkeypatch):
    """The projector's file bytes ride in extra_gpu_bytes and its runtime surcharge in soft_overhead."""
    common = dict(
        free_mib = 14 * 1024,
        extra_gpu = 3 * GIB,
        soft_overhead = 700 * MIB,
        mmproj_file_bytes = 1 * GIB,
        mmproj_surcharge_bytes = 400 * MIB,
    )
    movable, _ = _captured_opts(monkeypatch, _Stub(), mmproj_movable = True, **common)
    assert movable.mmproj_movable is True
    assert movable.mmproj_bytes == 1 * GIB + 400 * MIB
    assert movable.extra_resident_bytes == 2 * GIB
    assert movable.overhead_bytes_per_device == 300 * MIB

    pinned, _ = _captured_opts(monkeypatch, _Stub(), mmproj_movable = False, **common)
    assert pinned.mmproj_movable is False and pinned.mmproj_bytes == 0
    assert pinned.extra_resident_bytes == 3 * GIB + 400 * MIB
    assert pinned.overhead_bytes_per_device == 300 * MIB


def test_a_droppable_draft_is_a_separate_term_with_the_excluded_blocks(monkeypatch):
    """extra_gpu_bytes folds in the reserve at the LAUNCHED context; the planner gets the one priced
    at ITS context, plus the nextn blocks an engaging draft turns resident, as a term it may drop.
    """
    stub = _Stub()
    stub._excluded_bytes = 200 * MIB
    common = dict(
        free_mib = 14 * 1024,
        extra_gpu = 3 * GIB,
        mtp = True,
        mtp_reserve_bytes = 1 * GIB,
        draft_bytes = 3 * GIB // 2,
    )
    droppable, _ = _captured_opts(monkeypatch, stub, draft_droppable = True, **common)
    assert droppable.draft_droppable is True
    assert droppable.draft_bytes == 3 * GIB // 2 + 200 * MIB
    assert droppable.extra_resident_bytes == 2 * GIB

    kept, _ = _captured_opts(monkeypatch, stub, draft_droppable = False, **common)
    assert kept.draft_droppable is False and kept.draft_bytes == 0
    assert kept.extra_resident_bytes == 2 * GIB + 3 * GIB // 2 + 200 * MIB


def test_a_draft_that_cannot_be_dropped_because_it_has_no_bytes_is_not_droppable(monkeypatch):
    opts, _ = _captured_opts(
        monkeypatch, _Stub(), free_mib = 14 * 1024, mtp = True, draft_droppable = True
    )
    assert opts.draft_droppable is False and opts.draft_bytes == 0


def test_a_knob_only_plan_earns_the_pin_and_never_a_load_mode():
    """A plan that spilled nothing but reshaped the launch is still Unsloth's placement: every
    layer stays on a GPU, so it takes the same pin the proved arm does.
    """
    flags = LlamaCppBackend._spill_plan_flags_for
    assert flags(Plan(changed = True, n_parallel = 2)) == ["-ngl", "-1", "--fit", "off"]
    assert flags(Plan(changed = True, mmproj_to_host = True)) == ["-ngl", "-1", "--fit", "off"]
    assert flags(Plan(changed = True, draft_dropped = True)) == ["-ngl", "-1", "--fit", "off"]
    spilled = flags(
        Plan(changed = True, ot_patterns = ("x",), spilled_blocks = (1,), load_mode_none = True)
    )
    assert spilled == ["-ngl", "-1", "--fit", "off", "-ot", "x=CPU"]
    assert "--load-mode" not in spilled
    # A load mode alone is not a reshaping, and neither is a plan that changed nothing.
    assert flags(Plan(changed = True, load_mode_none = True)) == []
    assert flags(Plan(changed = False, n_parallel = 2)) == []
    assert flags(Plan(changed = True, n_parallel = 2, insufficient = True)) == []


def test_the_revocation_restores_the_values_the_plan_rewrote():
    """--parallel the plan lowered and -c it raised describe the plan's
    placement, not the fitter's, so a retry that revokes the plan puts the
    fitter's own values back. A double without the record still works."""
    stub = _Stub()
    stub._spill_plan_flags = ["-ngl", "-1", "--fit", "off", "-ot", "x=CPU"]
    stub._spill_plan_restore = {"--parallel": "4", "-c": "8192", "--cache-ram": "8192"}
    cmd = ["llama-server", "--parallel", "1", "-c", "131072", "--cache-ram", "1024"] + list(
        stub._spill_plan_flags
    )
    got = stub._drop_tensor_spill(cmd, "startup failure")
    assert got == [
        "llama-server",
        "--parallel",
        "4",
        "-c",
        "8192",
        "--cache-ram",
        "8192",
        "--fit",
        "on",
    ]
    bare = _Stub()
    bare._spill_plan_flags = ["-ngl", "-1", "--fit", "off"]
    assert bare._drop_tensor_spill(["x", "-ngl", "-1", "--fit", "off"], "retry") == [
        "x",
        "--fit",
        "on",
    ]


@pytest.mark.parametrize(
    "avail, footprint, expected",
    [
        (None, 0, None),
        (64 * 1024, 4 * 1024, 8192),
        (12 * 1024, 4 * 1024, 12 * 1024 - 4 * 1024 - 2048),
        (5 * 1024, 4 * 1024, 0),
    ],
)
def test_the_cache_ram_clamp(avail, footprint, expected):
    """Kept at the default whenever it fits, shrunk to what is left otherwise,
    down to disabled; unreadable RAM keeps llama.cpp's own default."""
    assert LlamaCppBackend._clamped_cache_ram_mib(avail, footprint) == expected


def test_the_planner_reserve_is_never_below_the_validated_curve(monkeypatch):
    """The 1536 MiB intercept was validated where the 3% the seam withholds already reaches it."""
    import inspect

    from core.inference import llama_cpp, offload_planner

    # The helper: whichever device had the least withheld anchors the floor.
    assert llama_cpp._reserve_floor_bytes([369 * MIB]) == 1536 * MIB - 369 * MIB
    assert llama_cpp._reserve_floor_bytes([1600 * MIB]) == 0
    assert llama_cpp._reserve_floor_bytes([1600 * MIB, 369 * MIB]) == 1536 * MIB - 369 * MIB
    assert llama_cpp._reserve_floor_bytes([]) == 0
    assert llama_cpp._reserve_floor_bytes([-5]) == 1536 * MIB

    # The seam charges max(its own terms, the floor), and the floor is applied
    # AFTER the projector surcharge leaves the fused terms.
    captured = {}

    def capture(layout, vram, host, ctx, **kw):
        captured["opts"] = kw["opts"]
        return Plan(reason = "captured")

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    withheld_12 = 369 * MIB
    _plan(
        _Stub(),
        free_mib = 12 * 1024,
        usable_mib = 12 * 1024 - 369,
        compute_flat = 320 * MIB,
        ctx_compute = 14 * MIB,
        reserve_floor = llama_cpp._reserve_floor_bytes([withheld_12]),
    )
    assert captured["opts"].overhead_bytes_per_device + withheld_12 >= 1536 * MIB
    assert captured["opts"].overhead_bytes_per_device == 1536 * MIB - withheld_12
    # Where the withheld share already covers the intercept the seam's own terms stand
    # (the fixture's soft_overhead is 0, so that is the context term alone).
    withheld_40 = 1600 * MIB
    _plan(
        _Stub(),
        free_mib = 40 * 1024,
        usable_mib = 40 * 1024 - 1600,
        compute_flat = 320 * MIB,
        ctx_compute = 14 * MIB,
        reserve_floor = llama_cpp._reserve_floor_bytes([withheld_40]),
    )
    assert captured["opts"].overhead_bytes_per_device == 14 * MIB
    assert captured["opts"].overhead_bytes_per_device + withheld_40 >= 1536 * MIB

    # And load_model really prices it from the same usable budget it hands over.
    src = inspect.getsource(llama_cpp.LlamaCppBackend)
    assert '"reserve_floor_bytes": _reserve_floor_bytes(' in src
    seam = inspect.getsource(llama_cpp.LlamaCppBackend._planned_tensor_spill)
    assert 'inputs.get("reserve_floor_bytes")' in seam


def test_a_sharded_gguf_is_read_whole_before_the_planner_sees_it():
    """A multi-part GGUF read from shard 1 alone is a fraction of the model and reports itself
    incomplete; the seam then declined with no log line, so the planner never saw a complete
    layout for exactly the models large enough to need it."""
    stub = _Stub()
    stub.sharded = True
    # Shard 1 alone is what the seam used to read, and it is incomplete.
    assert stub._tensor_spill_layout("/models/stub-00001-of-00004.gguf").complete is False
    plan = _plan(stub, free_mib = 14 * 1024)
    assert stub._layout_all_shards is True
    assert plan is not None and plan.spills_anything, plan


def test_host_ram_the_launch_has_already_spent_is_taken_off_the_planner_pool():
    """The seam names the host RAM this launch spends that no term of the plan carries, and the
    planner admits against what is left.
    """
    roomy = _plan(_Stub(), free_mib = 14 * 1024)
    assert roomy is not None and roomy.spills_anything
    assert roomy.load_mode_none

    # The same load on the same host, with 60 GiB of the 64 already promised.
    tight = _plan(_Stub(), free_mib = 14 * 1024, host_unpriced = 60 * GIB)
    assert tight is not None
    assert not tight.load_mode_none, tight.reason


def test_the_recurrent_state_is_taken_out_of_the_measured_cache_floor(monkeypatch):
    """``kv_cache_bytes`` is the whole hybrid memory; the floor is a CACHE."""
    import dataclasses

    from core.inference import offload_planner as planner

    state, attention = 512 * MIB, 3 * GIB

    class _Hybrid(_Stub):
        def _tensor_spill_layout(
            self,
            model_path,
            *,
            all_shards = False,
        ):
            layout = _Stub._tensor_spill_layout(self, model_path, all_shards = all_shards)
            if layout is None:
                return None
            return dataclasses.replace(layout, recurrent_bytes = state)

    seen = {}
    real = planner.plan_placement

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "plan_placement", spy)

    inputs = {
        **_inputs(kv = attention + 2 * state, n_parallel = 2, free_mib = 12 * 1024),
        "kv_recurrent_bytes_per_slot": state,
        "kv_bytes_floor_by_parallel": {
            1: attention // 2 + state,
            2: attention + 2 * state,
        },
    }
    _Hybrid()._planned_tensor_spill(inputs, env = {"UNSLOTH_SMART_OFFLOAD": "1"})

    assert seen["kv_bytes_floor"] == attention, "the state was left in the floor"
    assert seen["opts"].kv_bytes_floor_by_parallel == {1: attention // 2, 2: attention}


def test_a_state_the_layout_cannot_model_stays_in_the_floor(monkeypatch):
    """A KDA hybrid's state is priced by the estimator and not by the layout: ``offload_layout``
    reads ``ssm.*`` and Kimi-K3-shaped linear attention has none, so subtracting would
    UNDER-reserve the cache, and the subtraction is capped by what the layout models."""
    from core.inference import offload_planner as planner

    state, attention = 512 * MIB, 3 * GIB
    seen = {}
    real = planner.plan_placement

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "plan_placement", spy)

    stub = _Stub()
    assert stub._tensor_spill_layout("/models/stub.gguf").recurrent_bytes == 0
    stub._planned_tensor_spill(
        {
            **_inputs(kv = attention + state, free_mib = 12 * 1024),
            "kv_recurrent_bytes_per_slot": state,
        },
        env = {"UNSLOTH_SMART_OFFLOAD": "1"},
    )
    assert seen["kv_bytes_floor"] == attention + state


def test_the_cache_callable_is_corrected_like_the_floor_it_replaces(monkeypatch):
    """``kv_bytes_at`` re-prices the floor at every rung, and the planner takes
    ``layout.recurrent_bytes`` per slot out of what it returns before adding the same back.
    """
    import dataclasses

    from core.inference import offload_planner as planner
    from core.inference.offload_planner import _measured_cache_at

    state, attention = 512 * MIB, 3 * GIB

    class _Hybrid(_Stub):
        def _tensor_spill_layout(
            self,
            model_path,
            *,
            all_shards = False,
        ):
            layout = _Stub._tensor_spill_layout(self, model_path, all_shards = all_shards)
            return None if layout is None else dataclasses.replace(layout, recurrent_bytes = state)

    seen = {}
    real = planner.plan_placement

    def spy(layout, *args, **kwargs):
        seen["layout"] = layout
        seen.update(kwargs)
        return real(layout, *args, **kwargs)

    monkeypatch.setattr(planner, "plan_placement", spy)

    for priced in (0, state // 2, state, 3 * state):
        seen.clear()
        whole = attention + 2 * priced

        def raw(
            ctx,
            slots,
            _w = whole,
        ):
            return _w * ctx // 32768 * slots // 2

        _Hybrid()._planned_tensor_spill(
            {
                **_inputs(kv = whole, n_parallel = 2, free_mib = 12 * 1024, kv_bytes_at = raw),
                "kv_recurrent_bytes_per_slot": priced,
            },
            env = {"UNSLOTH_SMART_OFFLOAD": "1"},
        )
        opts, layout = seen["opts"], seen["layout"]
        cap = min(priced, state)
        assert _measured_cache_at(layout, opts, 32768, 2) == seen["kv_bytes_floor"], priced
        assert _measured_cache_at(layout, opts, 16384, 1) == raw(16384, 1) - cap, priced


def test_the_launch_cache_type_reaches_the_planner_as_a_mode(monkeypatch):
    """A quantised main cache is priced as one, with the type in force named, so
    the f16 product cannot override the smaller measured floor; an f16 launch
    leaves the planner's modes alone."""
    opts, _ = _captured_opts(monkeypatch, _Stub(), cache_type_kv = "q8_0")
    assert opts.cache_quantised is True and opts.kv_quant_type == "q8_0"
    opts, _ = _captured_opts(monkeypatch, _Stub(), cache_type_kv = "f16")
    assert opts.cache_quantised is False
    # The extras win per axis and the heavier axis is what the reserve budgets.
    opts, _ = _captured_opts(
        monkeypatch, _Stub(), cache_type_kv = "q8_0", extra_args = ["--cache-type-k", "f16"]
    )
    assert opts.cache_quantised is False


@pytest.mark.parametrize(
    "load_mode, env, extra_args",
    [
        ("mmap", {}, None),
        ("auto", {}, None),
        ("mmap+mlock", {}, None),
        (None, {"LLAMA_ARG_LOAD_MODE": "mmap"}, None),
        (None, {"LLAMA_ARG_NO_MMAP": "1"}, None),
        (None, {}, ["--load-mode", "mmap"]),
        (None, {}, ["--no-mmap"]),
    ],
)
def test_a_loader_mode_the_user_picked_stands_the_planner_down(load_mode, env, extra_args):
    """The user's mode replaces the plan's --load-mode none at launch, and the
    cost model priced the host side unmapped; the plan is not made at all."""
    env = {"UNSLOTH_SMART_OFFLOAD": "1", **env}
    assert LlamaCppBackend._planner_may_run(extra_args, env, load_mode = load_mode) is False
    inputs = _inputs()
    inputs["load_mode"] = load_mode
    assert _Stub()._planned_tensor_spill(inputs, extra_args = extra_args, env = env) is None
    # "none" is what the plan assumes, and an unset field is the default.
    assert LlamaCppBackend._planner_may_run(None, {"UNSLOTH_SMART_OFFLOAD": "1"}, load_mode = "none")
    assert LlamaCppBackend._planner_may_run(None, {"UNSLOTH_SMART_OFFLOAD": "1"}, load_mode = None)


def test_the_drafts_decode_graph_goes_with_the_draft_rung_2_may_drop(monkeypatch):
    """_MTP_DRAFT_COMPUTE_BYTES rode in soft_overhead, the per-device term, and
    stayed there after rung 2 dropped the draft; the no-draft child never
    allocates it, so it moves into the droppable term."""
    stub = _Stub()
    stub._excluded_bytes = 0
    common = dict(
        free_mib = 14 * 1024,
        extra_gpu = 3 * GIB,
        mtp = True,
        mtp_reserve_bytes = 1 * GIB,
        draft_bytes = 3 * GIB // 2,
        soft_overhead = 700 * MIB,
        mtp_draft_compute_bytes = 224 * MIB,
    )
    droppable, _ = _captured_opts(monkeypatch, stub, draft_droppable = True, **common)
    assert droppable.draft_bytes == 3 * GIB // 2 + 224 * MIB
    kept, _ = _captured_opts(monkeypatch, stub, draft_droppable = False, **common)
    assert kept.draft_bytes == 0
    # The per-device term gives the graph up only when the draft may go.
    assert kept.overhead_bytes_per_device == 700 * MIB
    assert droppable.overhead_bytes_per_device == 700 * MIB - 224 * MIB


def test_an_unbounded_prompt_cache_reaches_the_planner(monkeypatch):
    opts, _ = _captured_opts(monkeypatch, _Stub(), cache_ram_unbounded = True)
    assert opts.prompt_cache_unbounded is True
    opts, _ = _captured_opts(monkeypatch, _Stub())
    assert opts.prompt_cache_unbounded is False


@pytest.mark.parametrize(
    "keep_resident, no_ram_reserve", [(True, False), (False, True), (True, True)]
)
def test_a_model_memory_toggle_stands_the_planner_down(monkeypatch, keep_resident, no_ram_reserve):
    """ "Keep model in GPU memory" owns the load mode while weights sit in host RAM,
    which a spill guarantees, and "Don't reserve system RAM" drops "none"; either
    lands a plan priced unmapped under mmap, so no plan is made."""
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)
    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    assert LlamaCppBackend._planner_may_run(None, env) is False
    assert _Stub()._planned_tensor_spill(_inputs(), extra_args = None, env = env) is None
    monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
    assert LlamaCppBackend._planner_may_run(None, env) is True


def test_a_pass_through_parallel_equal_to_the_priced_count_pins_rung_1(monkeypatch):
    """Equal is not an override, but the extras are appended after the --parallel a
    plan rewrites and llama.cpp is last-wins, so a plan that lowered the slots
    would launch --fit off at the count it reserved against. The method pins the
    floor itself, whatever the snapshot said."""
    opts, _ = _captured_opts(monkeypatch, _Stub(), n_parallel = 4, extra_args = ["--parallel", "4"])
    assert opts.n_parallel == 4 and opts.min_parallel == 4
    opts, _ = _captured_opts(monkeypatch, _Stub(), n_parallel = 4)
    assert opts.min_parallel == 1


def test_the_compute_reserve_reaches_the_planner_priced_per_context(monkeypatch):
    """The snapshot prices the compute buffer at the requested context and the
    ladder tries smaller ones; a closure re-prices it per rung, floored like the
    flat term. Without one the flat term stands at every context."""
    from core.inference import offload_planner
    from core.inference.offload_planner import _device_reserve

    seen = {}
    real = offload_planner.plan_placement

    def capture(*a, **k):
        seen["opts"] = k.get("opts")
        return real(*a, **k)

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    inputs = _inputs(free_mib = 14 * 1024, ctx_compute = 512 * MIB, reserve_floor = 64 * MIB)
    inputs["ctx_compute_at"] = lambda ctx: 512 * MIB * ctx // 32768
    _Stub()._planned_tensor_spill(inputs, extra_args = None, env = {"UNSLOTH_SMART_OFFLOAD": "1"})
    opts = seen["opts"]
    assert opts.overhead_bytes_at is not None
    assert _device_reserve(opts, 32768) == opts.overhead_bytes_per_device
    assert _device_reserve(opts, 8192) == 128 * MIB
    assert _device_reserve(opts, 1024) == 64 * MIB  # never under the floor

    seen.clear()
    _plan(_Stub(), free_mib = 14 * 1024, ctx_compute = 512 * MIB)
    assert seen["opts"].overhead_bytes_at is None


# ------------------------------------------------- the link the prefill term is priced at


def test_the_measured_link_reaches_the_cost_model(monkeypatch):
    """The snapshot reads the device's PCIe rate; this is the other end of that wire.
    Absent, the cost model keeps the 55 GiB/s it was calibrated on."""
    from core.inference import offload_planner
    from core.inference.offload_cost_model import PREFILL_STREAM_GIB_S

    seen = {}
    real = offload_planner.plan_placement

    def capture(*a, **k):
        seen["opts"] = k.get("opts")
        return real(*a, **k)

    monkeypatch.setattr(offload_planner, "plan_placement", capture)
    _plan(_Stub(), free_mib = 14 * 1024, link_gib_s = 6.5)
    assert seen["opts"].host.link_gib_s == 6.5

    seen.clear()
    _plan(_Stub(), free_mib = 14 * 1024)
    assert seen["opts"].host.link_gib_s == PREFILL_STREAM_GIB_S


def test_the_launch_path_snapshots_the_link_over_the_credited_devices():
    """The planner can only price what the launch path measures for it, and only the
    devices the plan may credit bound the transfer."""
    compact = "".join(inspect.getsource(LlamaCppBackend.load_model).split())
    assert '"link_gib_s":(Noneifis_vulkan_backendelseself._nvidia_link_gib_s(' in compact
    assert "gpu_indicesifgpu_indicesisnotNoneelse[_idxfor_idx,_freein(gpusor())]" in compact


def test_a_slower_link_can_turn_a_planned_spill_into_a_decline():
    """The direction that matters. A spill worth taking over a PCIe 5 x16 link is not
    worth taking over a desktop x4 slot, because prefill has to stream the same bytes
    eight times slower, and before this the seam quoted every host the B200's rate."""
    fast = _plan(_Stub(), free_mib = 12800, n_threads = 6, link_gib_s = 55.0)
    slow = _plan(_Stub(), free_mib = 12800, n_threads = 6, link_gib_s = 6.0)
    assert fast is not None and slow is not None
    assert fast.spills_anything, "the fast-link control has to spill, or this proves nothing"
    assert not slow.spills_anything
    assert slow.declined_by_gate and "not worth it" in slow.reason


def test_the_windowed_half_of_the_cache_reaches_the_planner(monkeypatch):
    """The planner charges a saturated window in full and the full-context layers only over
    their live prefix, and a summed total cannot be decomposed after the fact: the seam has
    to hand the split over."""
    opts, _ = _captured_opts(monkeypatch, _Stub(), free_mib = 14 * 1024, kv_swa_bytes = 2 * GIB)
    assert opts.kv_swa_bytes_floor == 2 * GIB

    plain, _ = _captured_opts(monkeypatch, _Stub(), free_mib = 14 * 1024)
    assert plain.kv_swa_bytes_floor == 0, "a snapshot from before this field plans as it did"


# ------------------------------------- which launches read the PLE instead of holding it


class _PleStub(_Stub):
    """``_Stub`` whose layout carries per-layer embeddings of a given arch and size."""

    arch = "gemma4"
    ple_bytes = 8 * GIB

    def _tensor_spill_layout(
        self,
        model_path,
        *,
        all_shards = False,
    ):
        base = _Stub._tensor_spill_layout(self, model_path, all_shards = all_shards)
        return ModelLayout(
            **{
                **base.__dict__,
                "arch": self.arch,
                "token_embd_bytes": base.token_embd_bytes + self.ple_bytes,
                "per_layer_embd_bytes": self.ple_bytes,
            }
        )


def _ple_stub(**attrs):
    stub = _PleStub()
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub


def _ple_lazily(monkeypatch, stub, **kw):
    """Whether the seam tells the planner this launch pages the per-layer embeddings."""
    kw.setdefault("supports_lazy_mode", True)
    opts, _ = _captured_opts(monkeypatch, stub, free_mib = 14 * 1024, **kw)
    return opts.ple_read_lazily


def test_a_lazy_arch_on_a_build_with_the_flag_reads_the_embeddings_from_the_mapping(monkeypatch):
    """gemma4 (models/gemma4.cpp:llama_model_gemma4::load_arch_tensors) and qwen4exp
    (models/qwen4exp.cpp:llama_model_qwen4exp::load_arch_tensors) are the only two archs
    that create per_layer_token_embd TENSOR_READ_LAZY."""
    assert _ple_lazily(monkeypatch, _ple_stub()) is True
    assert _ple_lazily(monkeypatch, _ple_stub(arch = "qwen4exp")) is True


def test_gemma3n_keeps_the_full_charge(monkeypatch):
    """The one PLE architecture llama.cpp does NOT mark lazy: models/gemma3n.cpp passes
    flag 0, so the tensor is read in full and is resident host RAM."""
    assert _ple_lazily(monkeypatch, _ple_stub(arch = "gemma3n")) is False


def test_a_build_without_lazy_mode_keeps_the_full_charge(monkeypatch):
    """--lazy-mode first appears at b10700. An older llama-server reads the table in full
    whatever the arch says, and a --help that could not be parsed answers the same way."""
    assert _ple_lazily(monkeypatch, _ple_stub(), supports_lazy_mode = False) is False


def test_a_launch_that_asked_for_no_mapping_still_reads_the_table_lazily(monkeypatch):
    """--load-mode none does not make a lazy tensor resident: llama.cpp maps a lazy context
    whatever the load mode (llama-model-loader.cpp:llama_model_loader::init_mappings maps
    whenever lazy.any(), load_all_data reads it from_mapping), so the load mode is not a
    clause of this predicate."""
    assert _ple_lazily(monkeypatch, _ple_stub(), load_mode = "none") is True


def test_lazy_mode_off_keeps_the_full_charge(monkeypatch):
    """ "off: always keep them resident". Both spellings, and the env twin Studio never
    emits over."""
    assert _ple_lazily(monkeypatch, _ple_stub(), extra_args = ["--lazy-mode", "off"]) is False
    assert _ple_lazily(monkeypatch, _ple_stub(), extra_args = ["-lzm=off"]) is False
    assert _ple_lazily(monkeypatch, _ple_stub(), env = {"LLAMA_ARG_LAZY_MODE": "off"}) is False


def test_the_auto_default_only_moves_a_table_over_four_gib(monkeypatch):
    """``auto`` is the default and it declines anything at or below 4 GiB
    (llama-model-loader.cpp:llama_model_loader::lazy_read::add), so gemma-4-E2B's 1540 MiB
    per-layer table is resident on a default launch and only ``on`` pages it."""
    small = _ple_stub(ple_bytes = 1540 * MIB)
    assert _ple_lazily(monkeypatch, small) is False
    assert _ple_lazily(monkeypatch, small, extra_args = ["--lazy-mode", "on"]) is True
    assert _ple_lazily(monkeypatch, _ple_stub(ple_bytes = 4 * GIB)) is False
    assert _ple_lazily(monkeypatch, _ple_stub(ple_bytes = 4 * GIB + 1)) is True


def test_a_model_with_no_per_layer_embeddings_is_unaffected(monkeypatch):
    """Every non-PLE model plans exactly as it did: no bucket, nothing to excuse."""
    assert _ple_lazily(monkeypatch, _Stub()) is False
    assert _ple_lazily(monkeypatch, _ple_stub(ple_bytes = 0)) is False


def _deepseek_backend(**extra):
    """unsloth/DeepSeek-R1-GGUF's metadata as _load_gguf_metadata populates it."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._n_layers = 61
    b._n_kv_heads = 128
    b._n_heads = 128
    b._embedding_length = 7168
    b._kv_key_length = 192
    b._kv_value_length = 128
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    b._sliding_window = None
    b._sliding_window_pattern = None
    b._n_kv_heads_by_layer = None
    b._shared_kv_layers = None
    b._nextn_predict_layers = None
    b._architecture = "deepseek2"
    b._kv_lora_rank = 512
    b._key_length_mla = None
    b._value_length_mla = None
    b._ssm_inner_size = None
    b._ssm_state_size = None
    b._ssm_group_count = None
    b._ssm_conv_kernel = None
    b._kda_head_dim = None
    b._full_attention_interval = None
    for key, value in extra.items():
        setattr(b, key, value)
    return b


def test_the_lora_rank_alone_does_not_take_the_latent_path():
    """llama_hparams::is_mla (llama-hparams.cpp) needs BOTH MLA head lengths, and
    deepseek2.cpp reads them optionally. DeepSeek-R1 / V3-0324 carry the rank alone
    with head_count_kv 128, so llama.cpp allocates the full per-head K+V cache and
    path 1's K-only latent under-books it by 40% at every context."""
    plain = _deepseek_backend()
    full = plain._estimate_kv_cache_bytes(
        8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    # Path 4, 128 heads of K and of V over 61 layers: what llama-server allocates.
    assert full == 39040 * MIB

    latent = _deepseek_backend(_key_length_mla = 192, _value_length_mla = 128)
    assert (
        latent._estimate_kv_cache_bytes(
            8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
        )
        == 23424 * MIB
    )

    # One of the two alone is not is_mla(), so it must not flip the path either.
    half = _deepseek_backend(_key_length_mla = 192)
    assert (
        half._estimate_kv_cache_bytes(8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False)
        == full
    )


def _qwen3next_backend(**extra):
    """unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF: hybrid, and no interval key."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._architecture = "qwen3next"
    b._n_layers = 48
    b._n_kv_heads = 2
    b._n_heads = 16
    b._embedding_length = 2048
    b._kv_key_length = 256
    b._kv_value_length = 256
    b._kv_key_length_swa = None
    b._kv_value_length_swa = None
    b._sliding_window = None
    b._sliding_window_pattern = None
    b._n_kv_heads_by_layer = None
    b._recurrent_layers = None
    b._feed_forward_length_by_layer = None
    b._shared_kv_layers = None
    b._nextn_predict_layers = None
    b._kv_lora_rank = None
    b._key_length_mla = None
    b._value_length_mla = None
    b._kda_head_dim = None
    b._ssm_inner_size = 4096
    b._ssm_state_size = 128
    b._ssm_group_count = 16
    b._ssm_conv_kernel = 4
    b._full_attention_interval = None
    for key, value in extra.items():
        setattr(b, key, value)
    return b


def test_a_hybrid_without_the_interval_key_is_still_a_hybrid_to_the_estimator():
    """llama.cpp does not need the key: models/qwen3next.cpp falls back to the
    ARCHITECTURE's own default, and nemotron-h.cpp / falcon-h1.cpp derive the mask
    from the head counts. Reading its absence as "not a hybrid" charged Qwen3-Next-80B
    an attention cache on all 48 rows instead of 12 and priced the recurrent state at
    0 -- +187%, an over-count hiding an under-count, and the layout read the same file
    exactly. Both readers now take one derivation, so they cannot disagree."""
    from test_offload_planner import (
        _StubReader,
        _StubTensor,
        _layout_from_reader,
        _qwen3next_fields,
    )

    b = _qwen3next_backend()
    total = b._estimate_kv_cache_bytes(
        8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    # 12 attention layers of 2 heads x (256 + 256) f16, plus 36 rows of Mamba state.
    assert total == 12 * 8192 * 2 * (256 + 256) * 2 + 36 * (24576 + 524288) * 4
    assert b._mamba_recurrent_state_bytes(1) == 36 * (24576 + 524288) * 4

    layout = _layout_from_reader(
        _StubReader(
            _qwen3next_fields(**{"qwen3next.attention.head_count": 16}),
            [_StubTensor(f"blk.{i}.attn_q.weight", MIB) for i in range(48)],
        )
    )
    assert layout.n_attention_layers == 12
    assert total == layout.kv_bytes(8192) + layout.recurrent_bytes

    # Spelling the key out changes nothing.
    spelled = _qwen3next_backend(_full_attention_interval = 4)
    assert (
        spelled._estimate_kv_cache_bytes(
            8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
        )
        == total
    )


def test_the_hybrid_attention_count_divides_the_way_llama_cpp_does():
    """llama.cpp marks row il attention iff (il + 1) % full_attention_interval == 0
    over il < n_layer() (models/qwen3next.cpp, and the identical loop in qwen35,
    qwen35moe and qwen4exp), which is FLOOR division. Both readers used ceiling, so a
    30-layer hybrid at interval 4 was charged 8 attention layers where llama.cpp
    allocates 7 -- +14%, and one recurrent state too few. Every hybrid shipped so far
    has a layer count divisible by 4, which is the only reason it never showed up as a
    byte; the arithmetic is still wrong until it does."""
    from test_offload_planner import (
        _StubReader,
        _StubTensor,
        _layout_from_reader,
        _qwen3next_fields,
    )

    b = _qwen3next_backend(_n_layers = 30)
    total = b._estimate_kv_cache_bytes(
        8192, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    assert total == 7 * 8192 * 2 * (256 + 256) * 2 + 23 * (24576 + 524288) * 4

    layout = _layout_from_reader(
        _StubReader(
            _qwen3next_fields(
                **{"qwen3next.block_count": 30, "qwen3next.attention.head_count": 16}
            ),
            [_StubTensor(f"blk.{i}.attn_q.weight", MIB) for i in range(30)],
        )
    )
    assert layout.n_attention_layers == 7
    assert total == layout.kv_bytes(8192) + layout.recurrent_bytes


_QWEN4EXP_FIELDS = {
    "general.architecture": "qwen4exp",
    "qwen4exp.block_count": 48,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.ple.layers": [1],
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.conv_kernel": 4,
    "qwen4exp.expert_count": 512,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.context_length": 262144,
}


def _qwen4exp_backend(**extra):
    """unsloth/Qwen3.8-Flash-Next-GGUF: hybrid plus a lightning-indexer cache."""
    b = _qwen3next_backend(
        _architecture = "qwen4exp",
        _n_heads = 24,
        _embedding_length = 2560,
        _ssm_inner_size = 6144,
        _full_attention_interval = 4,
    )
    b._indexer_key_length = 128
    b._hyper_connection_count = 4
    b._ple_layers = [1]
    b._ple_ngram_size = 3
    b._ple_conv_kernel = 4
    for key, value in extra.items():
        setattr(b, key, value)
    return b


@pytest.mark.parametrize("n_ctx,before_mib,after_mib", [(8192, 304, 377), (65536, 1648, 2225)])
def test_the_indexer_cache_is_charged_on_the_architecture_that_builds_one(
    n_ctx, before_mib, after_mib
):
    """llama-model.cpp:create_memory routes qwen4exp to llama_memory_hybrid_idx, which
    allocates a THIRD cache over the dense-attention rows: one head of
    attention.indexer.key_length for K and one of attention.value_length for V, at the
    full context (llama-memory-hybrid-idx.cpp fills n_head_kv_arr with 1 and overrides
    n_embd_head_k_full, leaving n_embd_head_v alone). Neither reader knew the key
    existed, so Qwen3.8-Flash-Next was 19% short at 8k and 26% short at 64k -- an
    under-count, which loses the load. The PLE conv history is a row of the recurrent
    cache in its own right (llama-memory-recurrent.cpp allocates cache_ple_r_l,
    llama-hparams.cpp:ple_conv_state sizes it) and was missing too."""
    from test_offload_planner import _StubReader, _StubTensor, _layout_from_reader

    b = _qwen4exp_backend()
    total = b._estimate_kv_cache_bytes(
        n_ctx, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
    )
    attention = 12 * n_ctx * 2 * (256 + 256) * 2
    indexer = 12 * n_ctx * (128 + 256) * 2
    mamba = 36 * (30720 + 786432) * 4
    ple = 1 * (3 * 3 * 4 * 2560) * 4
    assert total == attention + indexer + mamba + ple
    assert round(total / MIB) == after_mib
    assert round((total - indexer - ple) / MIB) == before_mib

    layout = _layout_from_reader(
        _StubReader(
            _QWEN4EXP_FIELDS,
            [_StubTensor(f"blk.{i}.attn_q.weight", MIB) for i in range(48)],
        )
    )
    assert total == layout.kv_bytes(n_ctx) + layout.recurrent_bytes

    # Only that architecture: the same shape under any other arch keeps two caches.
    plain = _qwen4exp_backend(_architecture = "qwen3next")
    assert (
        plain._estimate_kv_cache_bytes(
            n_ctx, "f16", n_parallel = 1, kv_unified = False, flash_attn = False
        )
        == total - indexer
    )
