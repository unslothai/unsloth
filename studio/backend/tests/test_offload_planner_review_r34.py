# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which spills count as "every block" on a multi-GPU plan.

Only a spill that takes the same tensors off every block keeps each device's share independent of
where llama.cpp cuts the layer ranges. A byte test on the routed experts alone called a MoE spill
with the shared/dense FFN of only some blocks "every block", so the pooled back-first pick skipped
the per-device selection, and grading the boundary block after that test turned an every-block
pick partial, so the planner vetoed its own selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference import offload_planner as OP  # noqa: E402
from core.inference.offload_layout import BlockLayout, ModelLayout, SpillClass  # noqa: E402
from core.inference.offload_planner import ContextPolicy, PlanOptions, plan_placement  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB


def _moe(
    lm_head_mib = 500,
    n = 32,
    routed_mib = 300,
    shared_mib = 150,
    leading_dense = 0,
):
    """``n`` MoE blocks (routed experts plus a shared FFN), after ``leading_dense`` dense-FFN
    blocks that hold no experts (DeepSeek-V3 / GLM-4.5)."""
    third = routed_mib * MIB // 3
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = 0 if i < leading_dense else 3 * third,
            resident_bytes = shared_mib * MIB,
            ffn_down_bytes = 0 if i < leading_dense else third,
            ffn_up_bytes = 0 if i < leading_dense else third,
            ffn_gate_bytes = 0 if i < leading_dense else third,
            dense_ffn_bytes = shared_mib * MIB,
        )
        for i in range(n)
    )
    return ModelLayout(
        arch = "qwen3moe",
        n_layers = n,
        n_attention_layers = n,
        blocks = blocks,
        lm_head_bytes = lm_head_mib * MIB,
        token_embd_bytes = 500 * MIB,
        other_resident_bytes = 10 * MIB,
        kv_bytes_per_token_f16 = 128 * 1024,
        n_ctx_train = 32768,
        is_moe = True,
        n_expert = 128,
        n_expert_used = 8,
        complete = True,
    )


def _dense(n = 32):
    third = 100 * MIB
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = 3 * third,
            resident_bytes = 100 * MIB,
            ffn_down_bytes = third,
            ffn_up_bytes = third,
            ffn_gate_bytes = third,
            attn_bytes = 100 * MIB,
        )
        for i in range(n)
    )
    return ModelLayout(
        arch = "qwen3",
        n_layers = n,
        n_attention_layers = n,
        blocks = blocks,
        lm_head_bytes = 500 * MIB,
        token_embd_bytes = 500 * MIB,
        other_resident_bytes = 10 * MIB,
        kv_bytes_per_token_f16 = 128 * 1024,
        n_ctx_train = 32768,
        complete = True,
    )


def _plan(
    layout,
    vram_mib,
    policy = ContextPolicy.NEVER_REDUCE,
):
    return plan_placement(
        layout,
        [v * MIB for v in vram_mib],
        256 * GIB,
        32768,
        opts = PlanOptions(context_policy = policy),
        split_weights_per_device = [1, 1],
    )


def test_a_partial_shared_ffn_spill_is_not_every_block():
    layout = _moe()
    experts = OP._units_of(layout.blocks, None)
    shared = OP._units_of(layout.blocks, SpillClass.DENSE_FFN)
    assert OP._spills_every_block(layout, experts)
    assert OP._spills_every_block(layout, experts + shared)
    assert not OP._spills_every_block(layout, experts + shared[:4])


@pytest.mark.parametrize("policy", [ContextPolicy.NEVER_REDUCE, ContextPolicy.FIT_ONLY])
def test_a_partial_shared_ffn_spill_is_picked_device_by_device(policy):
    """Untied: the pooled pick took the shared FFN off the last card, so the first card's shortfall
    was covered by spilling the output head. The per-device pick needs no head spill."""
    plan = _plan(_moe(), [5024, 6500], policy)
    assert plan.changed and plan.n_ctx == 32768, plan.reason
    assert not plan.spilled_lm_head, plan.reason
    assert "device by device" in plan.reason, plan.reason


@pytest.mark.parametrize("policy", [ContextPolicy.NEVER_REDUCE, ContextPolicy.FIT_ONLY])
def test_a_tied_moe_with_a_partial_shared_ffn_spill_still_places(policy):
    """Tied: no head to rescue with, so the pooled pick abstained (or shrank the context)."""
    plan = _plan(_moe(lm_head_mib = 0), [5024, 6500], policy)
    assert plan.changed and plan.n_ctx == 32768, plan.reason


def test_grading_the_boundary_block_does_not_veto_an_every_block_spill():
    """Every block was needed, grading trimmed the last one against the pooled deficit, and the
    now-partial spill was refused as uncheckable against a pooled budget."""
    plan = _plan(_dense(), [5500, 5500])
    assert "partial spill" not in plan.reason, plan.reason
    assert plan.changed and plan.n_ctx == 32768, plan.reason


def test_the_description_counts_each_rung_against_its_own_blocks():
    """3 leading dense blocks hold a dense FFN and no experts: 58 of 61, not "every block"."""
    layout = _moe(n = 61, leading_dense = 3)
    shared = OP._units_of(layout.blocks, SpillClass.DENSE_FFN)
    experts = OP._units_of(layout.blocks, None)
    assert OP._rung_description(shared[3:], layout) == "the shared/dense FFN of 58 of 61 blocks"
    assert OP._rung_description(shared, layout) == "the shared/dense FFN of every block"
    assert OP._rung_description(experts[:10], layout) == "the FFN of 10 of 58 blocks"


def _launch_drafter_dropping_plan(tmp_path, monkeypatch, *, planned_starts):
    """An MTP load the plan placed by dropping the drafter (rung 2). The planned launch comes up
    only when ``planned_starts``; otherwise it crashes at startup and the plan is revoked.
    Returns every launched argv and the backend."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent
    from test_offload_planner_launch import CARD_MIB, DENSE, _backend

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
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    plan = OP.Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        n_parallel = 1,
        draft_dropped = True,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
    )
    backend._planned_tensor_spill = lambda inputs, **_kw: plan
    real_popen = subprocess.Popen
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = -11 if ("-ot" in cmd and not planned_starts) else None
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
        backend._stdout_lines = []
        planned = "-ot" in (cmds[-1] if cmds else [])
        return planned_starts or not planned

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                speculative_type = "auto",
                n_parallel = 1,
            )
        )
    return cmds, backend


def test_a_revoked_plan_gives_back_the_drafter_it_dropped(tmp_path, monkeypatch):
    """Revocation restores the slots, context and cache the plan changed; the drafter it dropped
    stayed dropped, so the --fit on retry ran without MTP and still blamed the planner."""
    cmds, backend = _launch_drafter_dropping_plan(tmp_path, monkeypatch, planned_starts = False)
    assert "-ot" in cmds[0] and "draft-mtp" not in cmds[0], cmds[0]
    retry = cmds[-1]
    assert "-ot" not in retry and retry[retry.index("--fit") + 1] == "on", retry
    assert "draft-mtp" in retry, retry
    assert retry.count("--spec-type") == 1, retry
    assert backend._spec_fallback_reason != "drafter_planner"


def test_a_plan_that_comes_up_keeps_the_drafter_dropped(tmp_path, monkeypatch):
    cmds, backend = _launch_drafter_dropping_plan(tmp_path, monkeypatch, planned_starts = True)
    assert len(cmds) == 1 and "-ot" in cmds[0] and "draft-mtp" not in cmds[0], cmds
    assert backend._spec_fallback_reason == "drafter_planner"


_GEN3_X4 = 4 * 0.985 * 0.85 * 1e9 / float(1024**3)
_GEN4_X16 = 16 * 1.969 * 0.85 * 1e9 / float(1024**3)


def test_a_mig_slice_is_priced_at_its_own_cards_link(monkeypatch):
    """MIG ids are CUDA ids: slice 0 on physical card 1 read card 0's slot."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    rows = [{"index": 0, "physical_index": 1, "mig": True}]
    monkeypatch.setattr(LlamaCppBackend, "_MIG_DEVICE_SNAPSHOT", (None, None, rows), raising = False)
    link = "0, 3, 4\n1, 4, 16\n"
    assert LlamaCppBackend._nvidia_link_gib_s([0], runner = lambda: link) == pytest.approx(_GEN4_X16)
    assert LlamaCppBackend._nvidia_link_gib_s([5], runner = lambda: link) is None
    monkeypatch.setattr(LlamaCppBackend, "_MIG_DEVICE_SNAPSHOT", None, raising = False)
    assert LlamaCppBackend._nvidia_link_gib_s([0], runner = lambda: link) == pytest.approx(_GEN3_X4)


@pytest.mark.parametrize("pci, probed", [(False, False), (True, True)])
def test_torch_ordinals_are_not_priced_against_nvidia_smi_rows(tmp_path, monkeypatch, pci, probed):
    """Torch-fallback ids are CUDA ordinals (FASTEST_FIRST here); nvidia-smi's rows are PCI order."""
    from test_offload_planner_review_r25 import _GEN4_X16 as _ROW0, _sm_none, _spilling_plan

    _cmd, _env, inputs, _watch, _backend = _sm_none(
        tmp_path, monkeypatch, _spilling_plan(), order = "FASTEST_FIRST", pci = pci
    )
    assert inputs["link_gib_s"] == (pytest.approx(_ROW0) if probed else None), inputs["link_gib_s"]
