# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mixed CUDA cards in one diffusion selection (diffusion_multigpu), CPU only with faked capabilities.

A secondary below the primary's capability is kept (only a card this torch build has no kernels for is left out); what
the planner places there is checked per component (bf16 sm_80, FP8 sm_89, INT8 sm_75, packed formats the primary's
capability, the attention backend), each card sized from its own free memory. Replica-type splits stay same-model. A
multi-card request that ends on one card says why. Identical cards decide exactly as before."""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch

from core.inference import diffusion_memory as dm
from core.inference import diffusion_multigpu as mg

ADA, RTX4090, RTX3090, A100, T4, H100, P100 = (8, 9), (8, 9), (8, 6), (8, 0), (7, 5), (9, 0), (6, 0)
NAMES = {
    (8, 9): "NVIDIA RTX 6000 Ada Generation",
    (8, 6): "NVIDIA GeForce RTX 3090",
    (8, 0): "NVIDIA A100-SXM4-80GB",
    (7, 5): "Tesla T4",
    (9, 0): "NVIDIA H100",
    (6, 0): "Tesla P100",
}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(mg, "_rocm_torch", lambda: False)
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: False)
    monkeypatch.setattr(mg, "_torch_arch_floor", lambda: (5, 0), raising = False)
    yield


def _cards(monkeypatch, *caps):
    table = {i: c for i, c in enumerate(caps)}
    monkeypatch.setattr(mg, "_capability", lambda o: table.get(int(o)))
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda o = 0: NAMES.get(table.get(int(o)), "card")
    )
    return table


def _old_rule(caps, primary, ordinals):
    """main's make_selection on CUDA: no secondary below the primary's capability (unknown kept)."""
    need = caps.get(primary)
    others = tuple(dict.fromkeys(o for o in ordinals if o != primary))
    if need is not None:
        others = tuple(o for o in others if (caps.get(o) or need) >= need)
    return mg.MultiGpuSelection(primary = primary, secondaries = others) if others else None


@pytest.mark.parametrize("primary_cap", [ADA, RTX4090])
def test_ampere_secondary_beside_an_ada_primary_is_kept(monkeypatch, primary_cap):
    _cards(monkeypatch, primary_cap, RTX3090)
    sel = mg.make_selection(0, [0, 1])
    assert sel == mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    assert sel.excluded == ()
    assert mg.automatic_selection_and_reason  # the Automatic path shares the rule
    _fake_host(monkeypatch, count = 2)
    _cards(monkeypatch, primary_cap, RTX3090)
    assert mg.automatic_selection(0) == mg.MultiGpuSelection(
        primary = 0, secondaries = (1,), automatic = True
    )


def test_identical_and_upward_pairs_decide_as_main(monkeypatch):
    # runs on main too: only make_selection, the selection's equality and its (new, uncompared) exclusions
    caps_pool = [T4, A100, RTX3090, ADA, H100, P100, None]
    for a, b, c in itertools.product(caps_pool, repeat = 3):
        caps = {0: a, 1: b, 2: c}
        monkeypatch.setattr(mg, "_capability", lambda o, caps = caps: caps.get(int(o)))
        for primary in (0, 1, 2):
            need = caps[primary]
            ordinals = [0, 1, 2]
            new = mg.make_selection(primary, ordinals)
            old = _old_rule(caps, primary, ordinals)
            if all(
                need is None or caps[o] is None or caps[o] >= need for o in ordinals if o != primary
            ):
                # every pair main kept (identical cards included): the same selection, nothing excluded
                assert new == old and (new is None or getattr(new, "excluded", ()) == ())


# main's decision for a 2x T4 post-load companion split (captured on 810ee8c82d with this harness)
_T4_PAIR_DECISION = {
    "chosen": "companion_split",
    "summary": (
        "placement: companion split: text_encoder, vae on cuda:1 (denoiser 8.8 GiB fits cuda:0, 12.7 GiB of "
        "21.4 GiB with its working memory; text_encoder, vae 8.1 GiB does not fit beside it)"
    ),
    "primary": "cuda:0",
    "automatic": False,
    "rejected": [
        {
            "kind": "one_card",
            "reason": "everything resident on cuda:0 needs 23.5 GiB, 21.4 GiB free after the reserve; one card would "
            "use 'group'",
        }
    ],
}


@pytest.mark.parametrize("te_dtype", [torch.float16, torch.bfloat16])
def test_identical_t4_pair_decision_is_unchanged(monkeypatch, te_dtype):
    # runs on main too: an identical pair decides byte for byte as before, whatever the weights hold
    _cards(monkeypatch, T4, T4)
    pipe = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.float16),
        text_encoder = torch.nn.Linear(2, 2).to(te_dtype),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, pipe)
    assert refined.secondary_ordinal == 1
    assert mg.placement_decision(pipe).describe() == _T4_PAIR_DECISION


def test_identical_pairs_never_inspect_the_weights(monkeypatch):
    _cards(monkeypatch, T4, T4)

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"read {name} on an identical pair")

    assert mg.placement_refusal(Exploding(), ("transformer", "text_encoder"), 0, 1) is None
    assert mg._secondary_runs_denoiser(
        Exploding(), mg.make_selection(0, [0, 1]), 1, {"a": (1, "dit"), "b": (1, "dit")}, None
    )
    assert mg.peer_selection(mg.make_selection(0, [0, 1])) == mg.make_selection(0, [0, 1])


def test_a_turing_card_beside_an_a100_stays_in_the_selection(monkeypatch):
    # main dropped it; now the planner decides per component (bf16 is refused there, fp16 / fp32 is not)
    _cards(monkeypatch, A100, T4)
    sel, why = mg.selection_and_reason(0, [0, 1])
    assert sel == mg.MultiGpuSelection(primary = 0, secondaries = (1,)) and why is None
    # a card this torch build has no kernels for is the only one left out, and the reason says so
    _cards(monkeypatch, A100, P100)
    monkeypatch.setattr(mg, "_torch_arch_floor", lambda: (7, 0))
    sel, why = mg.selection_and_reason(0, [0, 1])
    assert sel is None
    assert (
        why == "cuda:1 (Tesla P100, sm 6.0) left out: this torch build has no kernels below sm 7.0"
    )


def test_a_bf16_component_cannot_land_on_a_card_without_bf16(monkeypatch):
    _cards(monkeypatch, A100, T4)
    pipe = _Pipe(text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16))
    why = mg.placement_refusal(pipe, ("text_encoder",), 0, 1)
    assert why == "text_encoder cannot run on cuda:1 (sm 7.5): bf16 compute needs sm 8.0"


def test_fp8_weights_need_ada_on_the_secondary(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090)
    pipe = _Pipe(
        transformer = _fp8_linear(),
        text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16),
        vae = torch.nn.Linear(2, 2),
    )
    assert mg.placement_refusal(pipe, ("text_encoder", "vae"), 0, 1) is None
    why = mg.placement_refusal(pipe, ("transformer",), 0, 1)
    assert "transformer cannot run on cuda:1 (sm 8.6)" in why and "FP8" in why and "sm 8.9" in why
    # an Ada / Hopper secondary runs the same FP8 weights
    _cards(monkeypatch, H100, ADA)
    assert mg.placement_refusal(pipe, ("transformer",), 0, 1) is None


def test_int8_and_packed_weights(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090)
    int8 = torch.nn.Module()
    int8.register_buffer("w", torch.zeros(2, 2, dtype = torch.int8))
    gguf = torch.nn.Module()
    gguf.register_buffer("w", torch.zeros(4, dtype = torch.uint8))
    pipe = _Pipe(transformer = int8, transformer_2 = gguf)
    assert mg.placement_refusal(pipe, ("transformer",), 0, 1) is None  # torch._int_mm: sm 7.5
    why = mg.placement_refusal(pipe, ("transformer_2",), 0, 1)
    assert "packed / quantized weights" in why and "sm 8.9" in why


def test_arch_pinned_attention_keeps_the_denoiser_off_a_lower_card(monkeypatch):
    from core.inference.diffusion_attention import ATTENTION_BACKEND_ATTR

    _cards(monkeypatch, ADA, RTX3090)
    dit = torch.nn.Linear(2, 2).to(torch.bfloat16)
    setattr(dit, ATTENTION_BACKEND_ATTR, "sage")
    pipe = _Pipe(transformer = dit, text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16))
    why = mg.placement_refusal(pipe, ("transformer",), 0, 1)
    assert "attention backend sage" in why
    assert mg.placement_refusal(pipe, ("text_encoder",), 0, 1) is None
    sel = mg.make_selection(0, [0, 1])
    sizes = {"transformer": (1, "dit"), "transformer_2": (1, "dit")}
    assert mg._secondary_runs_denoiser(pipe, sel, 1, sizes, None) is False


def test_unknown_capability_keeps_the_card(monkeypatch):
    _cards(monkeypatch, ADA)  # cuda:1 unknown
    sel, why = mg.selection_and_reason(0, [0, 1])
    assert sel == mg.MultiGpuSelection(primary = 0, secondaries = (1,)) and why is None
    assert mg.placement_refusal(_Pipe(transformer = _fp8_linear()), ("transformer",), 0, 1) is None
    monkeypatch.setattr(mg, "_capability", lambda o: None)
    assert mg.make_selection(0, [0, 1]) == mg.MultiGpuSelection(primary = 0, secondaries = (1,))


def test_sd_cpp_servers_keep_the_previous_rule(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090, ADA)
    assert mg.peer_selection(mg.make_selection(0, [0, 1])) is None
    assert mg.peer_selection(mg.make_selection(0, [0, 1, 2])) == mg.MultiGpuSelection(
        primary = 0, secondaries = (2,)
    )


def test_a_card_the_old_rule_kept_is_tried_first(monkeypatch):
    # 4090 primary, a 3090 with the most free memory and a second 4090: the companion split goes to the 4090, as on main
    _cards(monkeypatch, RTX4090, RTX3090, RTX4090)
    sel = mg.make_selection(0, [0, 1, 2])
    probes = {
        0: mg.DeviceProbe(0, "4090", 24564, 20000),
        1: mg.DeviceProbe(1, "3090", 24576, 24000),
        2: mg.DeviceProbe(2, "4090", 24564, 18000),
    }
    assert mg.best_secondary(sel, probes) == 2
    split = mg.plan_companion_split(
        single_card_policy = "group",
        model_dense_mib = 20000,
        companion_dense_mib = 9000,
        runtime_headroom_mib = 2000,
        base_overhead_mib = 0,
        primary_limit_mib = 17000,
        secondary_limit_mib = lambda o: {1: 20000, 2: 15000}[o],
        selection = sel,
    )
    assert split is not None and split.secondary_ordinal == 2
    # only the 3090 beside the 4090: it is used
    sel = mg.make_selection(0, [0, 1])
    assert mg.best_secondary(sel, probes) == 1


def test_lower_card_helpers(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090, ADA)
    with mg.selection_scope(mg.make_selection(0, [0, 1, 2])):
        assert mg.secondary_below_primary(1) and not mg.secondary_below_primary(2)
    assert not mg.secondary_below_primary(1)  # no selection
    bnb = torch.nn.Module()

    class Int8Params(torch.nn.Parameter):
        pass

    bnb.weight = Int8Params(torch.zeros(2, 2, dtype = torch.int8), requires_grad = False)
    why = mg.placement_refusal(_Pipe(text_encoder = bnb), ("text_encoder",), 0, 1)
    assert why is not None and "packed / quantized weights" in why


# -- the planner on a mixed pair -----------------------------------------------------------------------------------


@dataclasses.dataclass
class _Target:
    supports_model_cpu_offload: bool = True
    ordinal: int = 0
    device: str = "cuda"
    backend: str = "cuda"
    is_cuda_torch_device: bool = True


def _plan():
    return dm.plan_diffusion_memory(
        target = _Target(),
        device_memory = dm.DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "discrete_vram",
            free_mib = 24000,
            total_mib = 24576,
        ),
        model_dense_mib = 20000,
        companion_dense_mib = 9000,
        text_encoder_dense_mib = 8500,
        runtime_headroom_mib = 2000,
    )


def _refine(monkeypatch, pipe):
    from core.inference import diffusion_layer_split_plan as lsp

    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "0")
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda p: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "encoder_card_limit_mib", lambda o: None)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None, **_: 1)
    monkeypatch.setattr(mg, "replica_layout", lambda *a, **k: (None, None))
    monkeypatch.setattr(mg, "_hidden_stream_reason", lambda *a, **k: None)
    plan = _plan()
    assert plan.offload_policy != dm.OFFLOAD_NONE
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        return plan, mg.refine_plan_for_multigpu(pipe, plan)


def test_ada_primary_moves_a_bf16_text_encoder_to_the_3090(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090)
    pipe = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.bfloat16),
        text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, pipe)
    assert refined.secondary_ordinal == 1 and "text_encoder" in refined.secondary_components
    # recorded by the planner (placement_status reports the live correction once place_split ran)
    assert mg.placement_decision(pipe).chosen == "companion_split"


def test_ada_primary_keeps_an_fp8_text_encoder_off_the_3090(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090)
    pipe = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.bfloat16),
        text_encoder = _fp8_linear(),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, pipe)
    assert refined.secondary_ordinal is None
    decision = mg.placement_status(pipe)["decision"]
    assert decision["chosen"] == "offload"
    reasons = [r["reason"] for r in decision["rejected"] if r["kind"] == "companion_split"]
    assert reasons and all("FP8" in r and "sm 8.9" in r and "cuda:1 (sm 8.6)" in r for r in reasons)


def test_a100_with_a_t4_moves_fp16_companions_and_refuses_bf16_ones(monkeypatch):
    _cards(monkeypatch, A100, T4)
    fp16 = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.bfloat16),
        text_encoder = torch.nn.Linear(2, 2).to(torch.float16),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, fp16)
    assert refined.secondary_ordinal == 1 and "text_encoder" in refined.secondary_components
    bf16 = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.bfloat16),
        text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, bf16)
    assert refined.secondary_ordinal is None
    decision = mg.placement_status(bf16)["decision"]
    reasons = [r["reason"] for r in decision["rejected"] if r["kind"] == "companion_split"]
    assert reasons and all(
        r == "text_encoder cannot run on cuda:1 (sm 7.5): bf16 compute needs sm 8.0"
        for r in reasons
    )


@pytest.mark.parametrize("caps", [(A100, (8, 9)), (RTX4090, RTX3090), (ADA, RTX3090)])
def test_each_card_is_sized_from_its_own_memory(monkeypatch, caps):
    # A100 + L4, 4090 + 3090, Ada + 3090: the text encoder goes to the secondary only when that card's own limit fits it
    _cards(monkeypatch, *caps)
    pipe = _Pipe(
        transformer = torch.nn.Linear(2, 2).to(torch.bfloat16),
        text_encoder = torch.nn.Linear(2, 2).to(torch.bfloat16),
        vae = torch.nn.Linear(2, 2),
    )
    plan, refined = _refine(monkeypatch, pipe)
    assert refined.secondary_ordinal == 1
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 6000)
    pipe = _Pipe(**{n: getattr(pipe, n) for n in ("transformer", "text_encoder", "vae")})
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        refined = mg.refine_plan_for_multigpu(pipe, _plan())
    assert refined.secondary_ordinal is None
    assert any("allowed" in r["reason"] for r in mg.placement_status(pipe)["decision"]["rejected"])


def test_fp8_denoiser_blocks_stay_off_an_ampere_card_in_a_layer_split(monkeypatch):
    from core.inference import diffusion_layer_split_plan as lsp

    _cards(monkeypatch, H100, A100)
    pipe = _Pipe(transformer = _fp8_linear())
    monkeypatch.setattr(lsp, "layer_split_mode", lambda: "auto")
    monkeypatch.setattr(lsp, "engine_available", lambda: True)
    monkeypatch.setattr(lsp, "plan_offloads_denoiser", lambda plan: True)
    monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
    monkeypatch.setattr(
        lsp,
        "_denoiser_layouts",
        lambda p: {"transformer": lsp.DenoiserLayout(stem_mib = 1, block_mib = (1,))},
    )
    declined = []
    monkeypatch.setattr(lsp, "_record_decline", lambda p, why, logger = None: declined.append(why))
    plan = object()
    out = lsp.refine_plan_for_layer_split(
        pipe,
        plan,
        selection = mg.make_selection(0, [0, 1]),
        sizes = {"transformer": (1, "dit")},
        primary_limit_mib = 1,
        secondary_limit_mib = lambda o: 100,
        headroom_mib = 0,
        overhead_mib = 0,
    )
    assert out is plan
    assert declined == [
        "transformer cannot run on cuda:1 (sm 8.0): FP8 weights (scaled_mm) need sm 8.9"
    ]


def test_replica_splits_stay_same_model_and_say_why(monkeypatch):
    _cards(monkeypatch, ADA, RTX3090, ADA)
    sel = mg.make_selection(0, [0, 1])
    probes = {
        0: mg.DeviceProbe(0, NAMES[ADA], 49140, 40000),
        1: mg.DeviceProbe(1, NAMES[RTX3090], 24576, 24000),
    }
    monkeypatch.setattr(mg, "probe_devices", lambda ords: probes)
    card, why = mg._replica_card(sel, 1000)
    assert card is None and "different model" in why
    assert mg.same_model_secondaries(sel) == ()
    assert mg.same_model_secondaries(mg.make_selection(0, [0, 1, 2])) == (2,)


def test_decision_names_a_card_the_selection_left_out(monkeypatch):
    _cards(monkeypatch, A100, P100, A100)
    monkeypatch.setattr(mg, "_torch_arch_floor", lambda: (7, 0))
    pipe = _Pipe()
    sel = mg.make_selection(0, [0, 1, 2])
    assert sel == mg.MultiGpuSelection(primary = 0, secondaries = (2,))
    mg._record_decision(pipe, sel, "one_card", "one card: everything resident on cuda:0 (fits)")
    rejected = mg.placement_status(pipe)["decision"]["rejected"]
    assert rejected[-1] == {
        "kind": "card",
        "reason": "cuda:1 (Tesla P100, sm 6.0) left out: this torch build has no kernels below sm 7.0",
    }


# -- a multi-card request that ends on one card says so ------------------------------------------------------------


def _fake_host(
    monkeypatch,
    *,
    count,
    current = 0,
):
    from core.inference import diffusion_device as dd

    monkeypatch.setattr(dd, "auto_cuda_ranking_possible", lambda: True)
    monkeypatch.setattr(
        dd,
        "resolve_diffusion_device_target",
        lambda **kw: _Target(ordinal = kw.get("ordinal")),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)


class _Backend:
    def __init__(self):
        self._state = None
        self.seen = None

    @mg.with_load_selection
    def _run_load(self, **kwargs):
        self.seen = mg.current_selection()
        self._state = type("State", (), {})()
        self._state.pipe = _Pipe()


def _load(gpu_ordinal, ordinals):
    backend = _Backend()
    backend._run_load(gpu_ordinal = gpu_ordinal, _gpu_ordinals = ordinals)
    return backend, mg.placement_status(backend._state.pipe)


def test_excluded_card_is_reported_as_a_one_card_decision(monkeypatch):
    _fake_host(monkeypatch, count = 2)
    _cards(monkeypatch, A100, P100)
    monkeypatch.setattr(mg, "_torch_arch_floor", lambda: (7, 0))
    for ordinals in ([0, 1], mg.AUTOMATIC_ORDINALS):
        backend, status = _load(0, ordinals)
        assert backend.seen is None
        decision = status["decision"]
        assert decision["chosen"] == "one_card" and decision["primary"] == "cuda:0"
        assert decision["automatic"] is (ordinals == mg.AUTOMATIC_ORDINALS)
        assert (
            "cuda:1 (Tesla P100, sm 6.0) left out: this torch build has no kernels below sm 7.0"
            in (decision["summary"])
        )


def test_keep_multiple_models_loaded_is_named(monkeypatch):
    from core.inference import gpu_residency

    _fake_host(monkeypatch, count = 2)
    _cards(monkeypatch, ADA, RTX3090)
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: True)
    for policy in ("fit", "always"):
        monkeypatch.setattr(gpu_residency, "current_policy", lambda policy = policy: policy)
        for ordinals in ([0, 1], mg.AUTOMATIC_ORDINALS):
            backend, status = _load(0, ordinals)
            assert backend.seen is None
            summary = status["decision"]["summary"]
            assert f"Keep multiple models loaded is {policy.title()}" in summary
            assert status["decision"]["chosen"] == "one_card"


def test_mask_and_unreadable_count_are_named(monkeypatch):
    _fake_host(monkeypatch, count = 2)
    _cards(monkeypatch, ADA, ADA)
    backend, status = _load(0, [0, 5])
    assert "cuda:5 left out: this process sees 2 CUDA card(s)" in status["decision"]["summary"]

    def unreadable():
        raise RuntimeError("driver")

    monkeypatch.setattr(torch.cuda, "device_count", unreadable)
    backend, status = _load(0, mg.AUTOMATIC_ORDINALS)
    assert "the CUDA card count could not be read" in status["decision"]["summary"]


def test_one_card_requests_record_nothing(monkeypatch):
    _fake_host(monkeypatch, count = 1)
    _cards(monkeypatch, ADA)
    backend, status = _load(None, mg.AUTOMATIC_ORDINALS)
    assert status is None
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    _fake_host(monkeypatch, count = 2)
    backend, status = _load(0, [0, 1])
    assert status is None
    # a real selection keeps the decision to the planner (nothing recorded by the wrapper)
    monkeypatch.delenv(mg.MULTIGPU_ENV)
    _cards(monkeypatch, ADA, RTX3090)
    backend, status = _load(0, [0, 1])
    assert backend.seen == mg.MultiGpuSelection(primary = 0, secondaries = (1,)) and status is None


class _Pipe:
    def __init__(self, **modules):
        self.components = dict(modules)
        for name, module in modules.items():
            setattr(self, name, module)


def _fp8_linear():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch has no float8 dtype")
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(torch.zeros(2, 2).to(torch.float8_e4m3fn), requires_grad = False)
    return m
