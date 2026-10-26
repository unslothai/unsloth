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


def _launch_drafter_dropping_plan(
    tmp_path,
    monkeypatch,
    *,
    planned_starts,
    extra_args = None,
    caps = None,
    ngram = True,
):
    """An MTP load the plan placed by dropping the drafter (rung 2); with ``ngram`` the build has
    ngram-mod, so the drafterless block is not empty. The planned launch comes up
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
        **({"ngram_mod_flavor": "new", "supports_ngram_mod": True} if ngram else {}),
        **(caps or {}),
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
                extra_args = extra_args,
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


def test_the_restored_drafter_keeps_its_place_ahead_of_the_extras(tmp_path, monkeypatch):
    """Appending the restored block at the end put it after the pass-through extras, so Studio's
    draft depth beat the user's (last-wins); it goes back where the drafterless block sat."""
    cmds, _backend = _launch_drafter_dropping_plan(
        tmp_path, monkeypatch, planned_starts = False, extra_args = ["--spec-draft-n-max", "5"]
    )
    retry = cmds[-1]
    assert "draft-mtp" in retry, retry
    depths = [retry[i + 1] for i, tok in enumerate(retry) if tok == "--spec-draft-n-max"]
    assert depths and depths[-1] == "5", retry
    assert retry.index("--spec-type") < retry.index("--api-key-file"), retry


def test_a_restored_drafter_on_a_build_without_flash_attention_gets_no_quantized_v(
    tmp_path, monkeypatch
):
    """The restored block carries the draft cache dtype; with flash attention unavailable a
    quantized draft V aborts the child, so it is reset like the launch's own."""
    from core.inference.llama_cpp import LlamaCppBackend

    real = LlamaCppBackend._build_speculative_flags

    def build(self, **kw):
        if kw.get("drafter_no_vram"):
            return real(self, **kw)
        return [
            "--model-draft",
            "/d.gguf",
            "--cache-type-k-draft",
            "q8_0",
            "--cache-type-v-draft",
            "q8_0",
        ]

    monkeypatch.setattr(LlamaCppBackend, "_build_speculative_flags", build)
    cmds, _backend = _launch_drafter_dropping_plan(
        tmp_path, monkeypatch, planned_starts = False, caps = {"supports_flash_attn": False}
    )
    retry = cmds[-1]
    assert "--model-draft" in retry, retry
    assert retry[retry.index("--cache-type-v-draft") + 1] not in {"q8_0", "q4_0"}, retry


def test_an_empty_drafterless_block_keeps_the_drafter_dropped(tmp_path, monkeypatch):
    """No ngram-mod: the drafterless block is empty and marks no place in an argv that guards and
    retries rewrite, so the drafter stays dropped and the status says so."""
    cmds, backend = _launch_drafter_dropping_plan(
        tmp_path, monkeypatch, planned_starts = False, ngram = False
    )
    retry = cmds[-1]
    assert "-ot" not in retry and "draft-mtp" not in retry, retry
    assert backend._spec_fallback_reason == "drafter_planner"


def test_a_block_that_cannot_be_found_leaves_the_speculative_state_alone(tmp_path, monkeypatch):
    """r36: the rebuild ran (and reset the drop bookkeeping and fallback reason) before the block
    was found, so a miss launched without MTP while the status reported it restored."""
    from core.inference import llama_cpp as llama_mod

    monkeypatch.setattr(llama_mod, "_subsequence_index", lambda tokens, run, hint: 0)
    cmds, backend = _launch_drafter_dropping_plan(tmp_path, monkeypatch, planned_starts = False)
    retry = cmds[-1]
    assert "-ot" not in retry and "draft-mtp" not in retry, retry
    assert backend._spec_fallback_reason == "drafter_planner"
    assert "mtp" not in str(backend._speculative_type).lower(), backend._speculative_type


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


def test_an_every_block_spill_short_on_one_card_selects_that_cards_rows():
    """r35: every routed expert moved and one card still short; its own shared-FFN rows cover it,
    so the head stays (untied) and the load places at all (tied)."""
    untied = _plan(_moe(), [4500, 8000])
    assert untied.changed and not untied.spilled_lm_head, untied.reason
    tied = _plan(_moe(lm_head_mib = 0), [4000, 8000])
    assert tied.changed and tied.n_ctx == 32768, tied.reason


def test_the_partial_spill_refusal_names_what_it_moved():
    plan = _plan(_dense(), [5375, 5807])
    assert (
        not plan.changed and "a partial spill (the FFN of 31 of 32 blocks)" in plan.reason
    ), plan.reason


def test_both_no_flash_retries_mark_flash_attention_off_before_revoking():
    """The revocation resets a restored drafter's quantized V cache only when it knows flash
    attention is off, so both retries set that first."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    src = inspect.getsource(LlamaCppBackend.load_model)
    revoke = '_fa_cmd = _revoke_spill_plan(_fa_cmd, "noflash")'
    assert src.count(revoke) == 2
    at = 0
    for _ in range(2):
        at = src.index(revoke, at + 1)
        before = src[:at].rstrip().splitlines()[-1].strip()
        assert before == "_flash_attn_known_off = True", before


@pytest.mark.parametrize(
    "probe",
    [
        lambda cls: cls._nvidia_link_query(),
        lambda cls: cls._nvidia_smi_free_total_mib(),
    ],
    ids = ["link", "free_vram"],
)
def test_a_wedged_nvidia_smi_cannot_hold_the_planner_probes(monkeypatch, probe):
    """r37: both probes called subprocess.run directly, whose kill-and-wait on timeout blocks for as
    long as a driver-wedged child refuses to die. Through the bounded helper they give up."""
    import subprocess
    import threading
    import time

    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware import gpu_query

    monkeypatch.setenv("UNSLOTH_GPU_QUERY_CACHE", "1")
    gpu_query.reset()
    release = threading.Event()

    def wedged(*_a, **_kw):
        release.wait(60)
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(subprocess, "run", wedged)
    answer = {}
    worker = threading.Thread(
        target = lambda: answer.setdefault("out", probe(LlamaCppBackend)), daemon = True
    )
    started = time.monotonic()
    try:
        worker.start()
        worker.join(20)
        assert not worker.is_alive(), "the probe is still blocked on the wedged nvidia-smi"
        assert time.monotonic() - started < 20
        assert answer["out"] in (None, [])
    finally:
        release.set()
        worker.join(5)
        gpu_query.reset()


def test_an_unsized_eagle3_drafter_is_not_planned():
    """r38: the fit never sizes an exported EAGLE-3 drafter, and a plan pins --fit off, so nothing
    would place the drafter the launch attaches. Decline, leaving llama.cpp's fitter to place both."""
    from test_offload_planner_seam import _inputs, _Stub

    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    assert _Stub()._planned_tensor_spill(_inputs(), env = env) is not None
    inputs = _inputs()
    inputs["eagle3_drafter_unsized"] = True
    assert _Stub()._planned_tensor_spill(inputs, env = env) is None


@pytest.mark.parametrize(
    "speculative_type, supports, sidecar, attaches",
    [
        ("auto", True, True, True),
        ("eagle3", True, True, True),
        ("auto", False, True, False),
        ("auto", True, False, False),
        ("off", True, True, False),
        ("mtp", True, True, False),
    ],
)
def test_the_launch_tells_the_seam_when_an_exported_eagle3_drafter_attaches(
    tmp_path, monkeypatch, speculative_type, supports, sidecar, attaches
):
    import json

    from test_offload_planner_launch import _launch_with

    if sidecar:
        (tmp_path / "eagle3-drafter.gguf").write_bytes(b"GGUF")
        (tmp_path / "drafter.json").write_text(
            json.dumps({"strategy": "eagle3", "files": ["eagle3-drafter.gguf"]})
        )
    plan = OP.Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _backend, seen = _launch_with(
        tmp_path,
        monkeypatch,
        plan,
        caps = {"supports_eagle3": supports},
        speculative_type = speculative_type,
    )
    assert seen["inputs"]["eagle3_drafter_unsized"] is attaches


def test_the_plan_bookkeeping_starts_empty_and_unload_clears_the_lazy_mode():
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    assert backend._spill_plan_append == [] and backend._spill_plan_restore == {}
    backend._studio_lazy_mode_flags = ["-lzm", "on"]
    backend.unload_model()
    assert backend._studio_lazy_mode_flags == []


def test_an_unsized_gpu_drafter_is_not_planned():
    """r39: the CPU drafter had its unsized decline; a GPU one the fit gave no bytes went on to a
    --fit off plan that nothing would place it under."""
    from test_offload_planner_seam import _inputs, _Stub

    env = {"UNSLOTH_SMART_OFFLOAD": "1"}
    inputs = _inputs()
    inputs["gpu_draft_unsized"] = True
    assert _Stub()._planned_tensor_spill(inputs, env = env) is None


@pytest.mark.parametrize(
    "extra_args, unsized",
    [
        (None, False),
        (["--model-draft", "DRAFT"], True),
        (["--model-draft", "DRAFT", "--spec-type", "draft-simple"], False),
        (["--model-draft", "DRAFT", "--spec-draft-ngl", "0"], False),
    ],
    ids = ["none", "bare", "sized", "cpu"],
)
def test_the_launch_tells_the_seam_when_a_gpu_drafter_is_unsized(
    tmp_path, monkeypatch, extra_args, unsized
):
    from test_offload_planner_launch import _launch_with

    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF")
    args = [str(draft) if a == "DRAFT" else a for a in extra_args] if extra_args else None
    plan = OP.Plan(changed = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))
    _cmd, _backend, seen = _launch_with(
        tmp_path,
        monkeypatch,
        plan,
        caps = {"spec_draft_ngl_flag": "--spec-draft-ngl"},
        extra_args = args,
    )
    assert seen["inputs"]["gpu_draft_unsized"] is unsized
