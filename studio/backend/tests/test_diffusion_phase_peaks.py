# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Learned peaks per render phase (``diffusion_measured_peaks``) and the encoder residency the planner lends to the
VAE decode (``diffusion_memory.refine_plan_from_loaded_weights`` / ``release_for_decode``), plus the per-workflow key.

The Qwen-Image-2.1 case uses what an auto load logged on an RTX 3090: int8 DiT 6922 MiB resident, 8917 MiB of text
encoders, a 21013 MiB safe budget. The first load (empty store) placed from the 2304 MiB seed and kept every encoder
resident; after one 1024x1024 render the store held ~5.9 GB and every later load kept only part of the encoders
resident, so every new prompt streamed the rest. The phases are the ones a 24 GiB-capped B200 render recorded: 1864 MiB
real peak (the VAE decode; encode + steps ~900) plus a 4187 MiB streamed-window pad (the encoders' largest leaf and the
transformer's largest block, times the prefetch depth + 1), 6051 MiB on the whole-render scale. CPU-only.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm

QWEN_LOADED = {
    "transformer": (6922, "dit"),
    "text_encoder": (8917, "text_encoder"),
    "vae": (254, "other"),
}
KEY = dict(family = "qwen-image-2.1", quant = "torchao:Int8Tensor", speed_mode = "default")


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _qwen_flat_plan(budget_mib: int = 21013, total_mib: int = 24576):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image-2.1"
        ),
        requested_mode = None,
        model_dense_mib = 6922 + 8917 + 254,
        companion_dense_mib = 8917 + 254,
        text_encoder_dense_mib = 8917,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "peaks.json"
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(path))
    for env in (
        mp.LEARNED_PEAKS_ENV,
        mp.PHASE_PEAKS_ENV,
        dm.MEASURED_ACTIVATION_ENV,
        dm.PARTIAL_RESIDENT_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA GeForce RTX 3090")
    return path


@pytest.fixture
def qwen_pipe(monkeypatch, store):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(QWEN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "torchao:Int8Tensor")
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _refine(pipe, plan):
    return dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "qwen-image-2.1", speed_mode = "default"
    )


PAD = 4187


def _phases(
    pre,
    dec,
    pad = PAD,
):
    """What ``diffusion._record_peak_probe`` stores for real pre-decode / decode peaks ``pre`` / ``dec``."""
    return {
        mp.PHASE_PRE_DECODE: pre + pad,
        mp.PHASE_DECODE: dec + pad,
        mp.PHASE_DECODE_UNPADDED: dec,
        mp.PHASE_UNPADDED: max(pre, dec),
    }


B200 = dict(peak = 1864 + PAD, phases = _phases(900, 1864))


def _record(
    peak,
    phases = None,
    **extra,
):
    kw = dict(KEY, width = 1024, height = 1024)
    kw.update(extra)
    return mp.record(**kw, peak_mib = peak, phases = phases)


# ---------------------------------------------------------------------------------------------------- planner


def test_decode_phase_peak_keeps_the_encoders_resident_on_24gb(qwen_pipe):
    plan = _qwen_flat_plan()
    assert plan.offload_policy == dm.OFFLOAD_GROUP and not plan.stream_transformer
    assert plan.stream_text_encoders
    first = _refine(qwen_pipe, plan)
    assert first.resident_text_encoder_mib == 8917  # the seeded first load
    _record(**B200)
    later = _refine(qwen_pipe, plan)
    assert (
        later.estimates["measured_runtime_headroom_mib"] == 7168
    )  # the whole-render reserve is unchanged
    assert later.resident_text_encoder_mib == 8917  # main: 4621
    assert later.estimates["measured_phase_headroom_mib"] == 2304  # max(seed 1849, 1864) x 1.15
    assert later.estimates["measured_phase_all_resident"]
    assert later.estimates["decode_lend_text_encoder_mib"] == 8917 - 4621
    assert "every weight resident" in later.reasons[-1]
    assert dm.decode_lend_from_plan(later) == (4296, 1864, 1024 * 1024, False)


def test_whole_render_peaks_alone_place_as_before(qwen_pipe):
    """A store written before phases existed (or a render that recorded none) keeps today's placement."""
    plan = _qwen_flat_plan()
    _record(B200["peak"])
    later = _refine(qwen_pipe, plan)
    assert later.resident_text_encoder_mib == 4621
    assert "decode_lend_text_encoder_mib" not in later.estimates
    assert dm.decode_lend_from_plan(later) is None


def test_a_peak_no_phase_recorded_render_explains_keeps_the_whole_render_reserve(qwen_pipe):
    plan = _qwen_flat_plan()
    _record(6800)  # an older render, phase unknown, above what the phase-recorded one saw
    _record(**B200)
    assert _refine(qwen_pipe, plan).resident_text_encoder_mib < 8917
    assert dm.learned_phase_split(KEY["family"], KEY["quant"], KEY["speed_mode"]) is None


def test_encoders_that_cannot_all_stay_resident_size_against_the_pre_decode_phase(qwen_pipe):
    """16 GiB: the transformer streams, so the unpadded check does not apply; the encoders size against encode +
    steps (padded), and the decode borrows the difference."""
    plan = _qwen_flat_plan(budget_mib = 13132, total_mib = 16384)
    _record(B200["peak"])
    before = _refine(qwen_pipe, plan)
    _record(**B200)
    after = _refine(qwen_pipe, plan)
    assert not before.resident_text_encoder_mib
    assert after.resident_text_encoder_mib == 7168 - 5888
    assert after.resident_transformer_mib == before.resident_transformer_mib
    assert not after.estimates["measured_phase_all_resident"]
    assert after.estimates["measured_phase_headroom_mib"] == 5888


def test_pre_decode_phase_at_the_whole_render_peak_lends_nothing(qwen_pipe):
    plan = _qwen_flat_plan(budget_mib = 13132, total_mib = 16384)
    _record(B200["peak"], _phases(1864, 1864))
    assert not _refine(qwen_pipe, plan).resident_text_encoder_mib


def test_phase_kill_switch_restores_the_whole_render_placement(qwen_pipe, monkeypatch):
    plan = _qwen_flat_plan()
    _record(**B200)
    monkeypatch.setenv(mp.PHASE_PEAKS_ENV, "0")
    assert _refine(qwen_pipe, plan).resident_text_encoder_mib == 4621


@pytest.mark.parametrize("budget,total", [(9300, 12288), (13132, 16384), (21013, 24576)])
def test_12_16_and_24gb_never_keep_less_resident(qwen_pipe, budget, total):
    plan = _qwen_flat_plan(budget_mib = budget, total_mib = total)
    _record(B200["peak"])
    before = _refine(qwen_pipe, plan)
    _record(**B200)
    after = _refine(qwen_pipe, plan)
    assert (after.resident_text_encoder_mib or 0) >= (before.resident_text_encoder_mib or 0)
    assert after.resident_transformer_mib == before.resident_transformer_mib
    assert after.offload_policy == before.offload_policy
    assert after.stream_transformer == before.stream_transformer


# ---------------------------------------------------------------------------------------------------- store


def test_phases_are_stored_beside_the_whole_render_peak(store):
    assert _record(5900, {mp.PHASE_PRE_DECODE: 2000, mp.PHASE_DECODE: 5900})
    assert mp.lookup(**KEY, phase = mp.PHASE_UNPADDED) is None  # not given
    assert mp.lookup(**KEY) == 5900
    assert mp.lookup(**KEY, phase = mp.PHASE_PRE_DECODE) == 2000
    assert mp.lookup(**KEY, phase = mp.PHASE_DECODE) == 5900
    # a lower whole-render peak still raises a phase (max wins per phase), the whole-render value stays
    assert not _record(5000, {mp.PHASE_PRE_DECODE: 2400, mp.PHASE_DECODE: 5000})
    assert mp.lookup(**KEY) == 5900
    assert mp.lookup(**KEY, phase = mp.PHASE_PRE_DECODE) == 2400
    entry = json.loads(store.read_text(encoding = "utf-8"))["entries"]
    (only,) = entry.values()
    assert only["phases"][mp.PHASE_DECODE] == {str(1024 * 1024): 5900}
    # a phase never exceeds the render's own peak, unknown names are dropped
    _record(100, {mp.PHASE_PRE_DECODE: 9000, "bogus": 1}, width = 512, height = 512)
    assert mp.lookup(**KEY, width = 512, height = 512, phase = mp.PHASE_PRE_DECODE) == 100


def test_store_without_phases_answers_none_per_phase(store):
    _record(5900)
    assert mp.lookup(**KEY) == 5900
    assert mp.lookup(**KEY, phase = mp.PHASE_DECODE) is None
    assert dm.learned_phase_split(KEY["family"], KEY["quant"], KEY["speed_mode"]) is None
    mp._cache.update(path = None, mtime = None, data = None)
    store.unlink()
    _record(**B200)
    split = dm.learned_phase_split(KEY["family"], KEY["quant"], KEY["speed_mode"])
    assert split[mp.PHASE_DECODE] == 6051 and split[mp.PHASE_UNPADDED] == 1864


class _FakeCudaTorch:
    def __init__(self):
        self.allocated = 0
        self.peak = 0
        torch_self = self

        class _Dev:
            def __init__(self, *args):
                self.type = "cuda"
                self.index = 0

            def __str__(self):
                return "cuda:0"

        class cuda:  # noqa: N801
            is_available = staticmethod(lambda: True)
            current_device = staticmethod(lambda: 0)
            synchronize = staticmethod(lambda dev = None: None)
            memory_allocated = staticmethod(lambda dev = None: torch_self.allocated)
            max_memory_allocated = staticmethod(lambda dev = None: torch_self.peak)

            @staticmethod
            def reset_peak_memory_stats(dev = None):
                torch_self.peak = torch_self.allocated

        self.device = _Dev
        self.cuda = cuda

    def alloc(self, mib):
        self.allocated += mib << 20
        self.peak = max(self.peak, self.allocated)

    def free(self, mib):
        self.allocated -= mib << 20


def test_probe_splits_the_render_at_the_decode_without_resetting(monkeypatch):
    fake = _FakeCudaTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    fake.alloc(15000)  # resident weights before the render
    probe = mp.PeakProbe("cuda:0")
    fake.alloc(2000)  # encode
    fake.free(2000)
    fake.alloc(1500)  # denoise
    fake.free(1500)
    probe.mark_decode()
    peak_before = fake.peak
    fake.alloc(5900)  # untiled decode
    fake.free(5900)
    assert fake.peak >= peak_before  # nothing reset the shared counters
    assert probe.finish() == 5900
    assert probe.phase_peaks == {mp.PHASE_PRE_DECODE: 2000, mp.PHASE_DECODE: 5900}


def test_probe_with_two_decodes_reports_no_split(monkeypatch):
    fake = _FakeCudaTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    probe = mp.PeakProbe("cuda:0")
    probe.mark_decode()
    fake.alloc(800)
    fake.free(800)
    probe.mark_decode()  # a split batch: chunk 2 encodes after chunk 1 decoded
    fake.alloc(3000)
    assert probe.finish() == 3000
    assert probe.phase_peaks == {}


def test_render_records_phases_and_keys_non_txt2img_workflows(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    # recorded without a device target: the SDPA verdict counts as known
    monkeypatch.setattr(diffusion, "_attention_scaling_known", lambda target, engaged = None: True)
    pipe = types.SimpleNamespace(_unsloth_peak_identity = (KEY["family"], KEY["quant"], "default"))
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default")

    class _Probe:
        device = "cuda:0"
        failed_captures = 0
        window_pad = 64

        def __init__(self, peak, phases):
            self._peak = peak
            self.phase_peaks = phases

        def finish(self):
            return self._peak

    diffusion._record_peak_probe(
        state,
        _Probe(5900, {mp.PHASE_PRE_DECODE: 2000, mp.PHASE_DECODE: 5900}),
        width = 1024,
        height = 1024,
        batch_size = 1,
        workflow = "txt2img",
    )
    assert mp.lookup(**KEY) == 5964
    assert mp.lookup(**KEY, phase = mp.PHASE_PRE_DECODE) == 2064
    assert mp.lookup(**KEY, phase = mp.PHASE_DECODE) == 5964
    # nothing streams during the decode, and a load keeping every weight resident holds no window: no pad
    assert mp.lookup(**KEY, phase = mp.PHASE_DECODE_UNPADDED) == 5900
    assert mp.lookup(**KEY, phase = mp.PHASE_UNPADDED) == 5900
    diffusion._record_peak_probe(
        state, _Probe(7800, {}), width = 1024, height = 1024, batch_size = 1, workflow = "edit"
    )
    assert mp.lookup(**KEY) == 5964  # an edit no longer raises the text-to-image key
    assert mp.lookup(**dict(KEY, quant = KEY["quant"] + "|wf:edit")) == 7864


def test_with_workflow_keeps_txt2img_on_the_load_key():
    assert dm.with_workflow("a|b", None) == "a|b"
    assert dm.with_workflow("a|b", "txt2img") == "a|b"
    assert dm.with_workflow("a|b", "edit") == "a|b|wf:edit"


def test_edit_request_is_sized_from_its_own_key(qwen_pipe):
    plan = _qwen_flat_plan()
    _record(**B200)
    later = _refine(qwen_pipe, plan)
    qwen_pipe._unsloth_measured_reserve = (
        later.estimates["measured_runtime_headroom_mib"],
        "qwen-image-2.1",
        "default",
        None,
        None,
        later.estimates["measured_learned_peak_mib"],
        later.estimates["measured_learned_peak_pixels"],
        False,
        True,
        False,
    )
    kw = dict(width = 1024, height = 1024, batch_size = 1, cfg = False)
    assert dm.measured_request_extra_mib(qwen_pipe, **kw, workflow = "txt2img") == 0
    assert (
        dm.measured_request_extra_mib(qwen_pipe, **kw, workflow = "edit") == 0
    )  # nothing learned for edits yet
    _record(7800, quant = KEY["quant"] + "|wf:edit")
    # 7800 x 1.15 rounded to 256 = 9216, past the 7168 the load reserved
    assert dm.measured_request_extra_mib(qwen_pipe, **kw, workflow = "edit") == 9216 - 7168
    assert dm.measured_request_extra_mib(qwen_pipe, **kw, workflow = "txt2img") == 0


# ---------------------------------------------------------------------------------------------------- decode guard


class _GuardTorch:
    def __init__(
        self,
        free_mib,
        reserved_mib = 0,
        allocated_mib = 0,
    ):
        outer = self
        self.free = free_mib
        self.reserved = reserved_mib
        self.allocated = allocated_mib

        class device:  # noqa: N801
            def __init__(
                self,
                spec,
                index = None,
            ):
                spec = str(spec)
                self.type = spec.split(":")[0]
                self.index = (
                    index
                    if index is not None
                    else (int(spec.split(":")[1]) if ":" in spec else None)
                )

        class cuda:  # noqa: N801
            current_device = staticmethod(lambda: 0)
            mem_get_info = staticmethod(lambda dev = None: (outer.free << 20, 24576 << 20))
            memory_reserved = staticmethod(lambda dev = None: outer.reserved << 20)
            memory_allocated = staticmethod(lambda dev = None: outer.allocated << 20)

        self.device = device
        self.cuda = cuda


def test_decode_release_keeps_the_vram_budget_reserve_free(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dm,
        "release_resident_groups",
        lambda pipe, need, logger = None, reason = "", device = None, **kw: calls.append(need)
        or (lambda: None),
    )
    pipe = types.SimpleNamespace()
    pipe._unsloth_decode_lend = (4040, 5900, 1024 * 1024, False)
    # 7500 MiB covers the decode, but not plus a 2 GiB reserve
    monkeypatch.setattr(dm, "user_budget_reserve_bytes", lambda device, total: 2048 << 20)
    monkeypatch.setitem(sys.modules, "torch", _GuardTorch(free_mib = 7500))
    assert dm.release_for_decode(pipe, width = 1024, height = 1024, device = "cuda:0") is not None
    assert calls == [6785 + dm._DECODE_SLACK_MIB - (7500 - 2048)]


def test_decode_streams_lent_encoders_only_when_free_memory_is_short(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dm,
        "release_resident_groups",
        lambda pipe, need, logger = None, reason = "", device = None, **kw: calls.append(need)
        or (lambda: None),
    )
    pipe = types.SimpleNamespace()
    pipe._unsloth_decode_lend = (4040, 5900, 1024 * 1024, False)
    need = dm.decode_need_mib(pipe._unsloth_decode_lend, width = 1024, height = 1024)
    assert need == 6785  # 5900 x 1.15
    monkeypatch.setitem(sys.modules, "torch", _GuardTorch(free_mib = 7500))
    assert dm.release_for_decode(pipe, width = 1024, height = 1024, device = "cuda:0") is None
    assert calls == []
    # cached allocator blocks count as free
    monkeypatch.setitem(
        sys.modules, "torch", _GuardTorch(free_mib = 6000, reserved_mib = 3000, allocated_mib = 1000)
    )
    assert dm.release_for_decode(pipe, width = 1024, height = 1024, device = "cuda:0") is None
    monkeypatch.setitem(sys.modules, "torch", _GuardTorch(free_mib = 5000))
    assert dm.release_for_decode(pipe, width = 1024, height = 1024, device = "cuda:0") is not None
    assert calls == [6785 + dm._DECODE_SLACK_MIB - 5000]
    # a larger canvas scales the learned decode the way the store does
    assert dm.decode_need_mib(pipe._unsloth_decode_lend, width = 2048, height = 2048) == 6785 * 4
    # no lend, no check
    assert dm.release_for_decode(types.SimpleNamespace(), width = 1024, height = 1024) is None


def test_edit_only_family_records_under_the_load_key(store, monkeypatch):
    # an edit-only family runs nothing but edits: its renders keep teaching the key its next load places from
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(diffusion, "_attention_scaling_known", lambda target, engaged = None: True)
    pipe = types.SimpleNamespace(_unsloth_peak_identity = (KEY["family"], KEY["quant"], "default"))
    family = types.SimpleNamespace(edit = True)
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default", family = family)

    class _Probe:
        device = "cuda:0"
        failed_captures = 0
        window_pad = 0
        phase_peaks = {}

        def finish(self):
            return 7800

    diffusion._record_peak_probe(
        state, _Probe(), width = 1024, height = 1024, batch_size = 1, workflow = "edit"
    )
    assert mp.lookup(**KEY) == 7800
    assert mp.lookup(**dict(KEY, quant = KEY["quant"] + "|wf:edit")) is None
    assert diffusion._peak_workflow(family, "edit") is None
    assert diffusion._peak_workflow(types.SimpleNamespace(unified_edit = True), "edit") == "edit"


def test_cfg_off_edit_request_reads_its_cfg_off_workflow_key(qwen_pipe):
    plan = _qwen_flat_plan()
    _record(**B200)
    later = _refine(qwen_pipe, plan)
    qwen_pipe._unsloth_peak_expects_cfg = True
    qwen_pipe._unsloth_peak_identity = (KEY["family"], KEY["quant"], "default")
    qwen_pipe._unsloth_measured_reserve = (
        later.estimates["measured_runtime_headroom_mib"],
        "qwen-image-2.1",
        "default",
        None,
        None,
        later.estimates["measured_learned_peak_mib"],
        later.estimates["measured_learned_peak_pixels"],
        False,
        True,
        False,
    )
    kw = dict(width = 1024, height = 1024, batch_size = 1, workflow = "edit")
    # what the recorder writes for a guidance-off edit on a CFG-capable family
    _record(7800, quant = KEY["quant"] + "|cfg:off|wf:edit")
    assert dm.measured_request_extra_mib(qwen_pipe, **kw, cfg = False) == 9216 - 7168
    # a CFG request is not covered by the guidance-off sample
    assert dm.measured_request_extra_mib(qwen_pipe, **kw, cfg = True) == 0


def _seeded_edit_pipe(pipe):
    """A phase-seeded load that reserved 7168 MiB from a 6051 MiB learned whole-render peak."""
    pipe._unsloth_peak_expects_cfg = True
    pipe._unsloth_peak_identity = (KEY["family"], KEY["quant"], "default")
    pipe._unsloth_measured_reserve = (
        7168,
        "qwen-image-2.1",
        "default",
        None,
        None,
        6051,
        1024 * 1024,
        False,
        True,
        False,
    )
    return pipe


def test_edit_workflow_peak_does_not_charge_the_condition_images_twice(qwen_pipe):
    pipe = _seeded_edit_pipe(qwen_pipe)
    kw = dict(
        width = 1024,
        height = 1024,
        batch_size = 1,
        cfg = False,
        workflow = "edit",
        condition_pixels = int(1024 * 1024 * 0.32),
    )
    # 7168 seed reserve + 3014 condition allowance - 7168 headroom: what main charges, store or no store
    assert dm.measured_request_extra_mib(pipe, **kw) == 3014
    # a learned edit peak (7800 x 1.15 -> 9216) already holds the condition images; it is an alternative to the seed
    # estimate plus allowance (10182), not a base the allowance is added on top of (that gave 5062)
    _record(7800, quant = KEY["quant"] + "|cfg:off|wf:edit")
    assert dm.measured_request_extra_mib(pipe, **kw) == 3014


def test_edit_workflow_peak_above_seed_plus_allowance_is_followed(qwen_pipe):
    pipe = _seeded_edit_pipe(qwen_pipe)
    kw = dict(width = 1024, height = 1024, batch_size = 1, cfg = False, workflow = "edit")
    cond = int(1024 * 1024 * 0.32)
    # 12000 x 1.15 rounded to 256 = 13824, past 7168 + 3014
    _record(12000, quant = KEY["quant"] + "|cfg:off|wf:edit")
    assert dm.measured_request_extra_mib(pipe, **kw, condition_pixels = cond) == 13824 - 7168
    assert dm.measured_request_extra_mib(pipe, **kw, condition_pixels = 0) == 13824 - 7168


def test_decode_headroom_does_not_credit_cuda_graph_pool_cache(monkeypatch):
    # a graph pool's free blocks sit inside reserved - allocated yet an ordinary VAE allocation cannot reuse them
    import core.inference.diffusion_cuda_graph as cg

    calls = []
    monkeypatch.setattr(
        dm,
        "release_resident_groups",
        lambda pipe, need, logger = None, reason = "", device = None, **kw: calls.append(need)
        or (lambda: None),
    )
    pipe = types.SimpleNamespace(_unsloth_decode_lend = (4040, 5900, 1024 * 1024, False))
    monkeypatch.setattr(cg, "live_pool_free_bytes", lambda: 2000 << 20)
    monkeypatch.setitem(
        sys.modules, "torch", _GuardTorch(free_mib = 6000, reserved_mib = 3000, allocated_mib = 1000)
    )
    assert dm.release_for_decode(pipe, width = 1024, height = 1024, device = "cuda:0") is not None
    assert calls == [6785 + dm._DECODE_SLACK_MIB - 6000]


def test_render_whose_decode_released_lent_groups_records_nothing(store, monkeypatch):
    # the decode ran net of the streamed groups' bytes: its peak (and whole-render peak) would under-size the bucket
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(diffusion, "_attention_scaling_known", lambda target, engaged = None: True)
    pipe = types.SimpleNamespace(_unsloth_peak_identity = (KEY["family"], KEY["quant"], "default"))
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default")
    finished = []

    class _Probe:
        device = "cuda:0"
        failed_captures = 0
        window_pad = 0
        decode_released = True
        phase_peaks = {mp.PHASE_PRE_DECODE: 2000, mp.PHASE_DECODE: 2000}

        def finish(self):
            finished.append(1)
            return 2000

    diffusion._record_peak_probe(
        state, _Probe(), width = 1024, height = 1024, batch_size = 1, workflow = "txt2img"
    )
    assert finished == [1]  # the probe's window still closes
    assert mp.lookup(**KEY) is None
    assert mp.lookup(**KEY, phase = mp.PHASE_DECODE_UNPADDED) is None


# ---------------------------------------------------------------------------------------------------- v1 store


def _write_v1(store, entries):
    store.write_text(
        json.dumps({"version": 1, "runtime": mp.runtime_stamp(), "entries": entries}),
        encoding = "utf-8",
    )
    mp._cache.update(path = None, mtime = None, data = None)


def test_pre_workflow_store_peak_no_longer_sizes_txt2img(qwen_pipe, store):
    """Before workflow keys every workflow recorded max-wins under the load key: an old edit peak there must not
    reserve for text-to-image (nor block the phase split) once a txt2img render records its own peak."""
    plan = _qwen_flat_plan()
    fresh = _refine(qwen_pipe, plan)
    base = mp.model_key(
        KEY["family"], KEY["quant"], KEY["speed_mode"], 1, "NVIDIA GeForce RTX 3090"
    )
    bucket = str(1024 * 1024)
    edit = mp.model_key(
        KEY["family"], KEY["quant"] + "|wf:edit", KEY["speed_mode"], 1, "NVIDIA GeForce RTX 3090"
    )
    other = mp.model_key(
        "flux.1-dev", KEY["quant"], KEY["speed_mode"], 1, "NVIDIA GeForce RTX 3090"
    )
    phased = {name: {bucket: value} for name, value in _phases(900, 1864).items()}
    _write_v1(
        store,
        {
            base: {"peaks": {bucket: 13000}},  # an edit render's peak, pre-PR
            edit: {"peaks": {bucket: 9000}},
            # the per-workflow format: the inflated whole-render value falls back to its phases
            other: {"peaks": {bucket: 13000}, "phases": phased},
        },
    )
    assert mp.lookup(**KEY) is None
    assert mp.lookup(**dict(KEY, quant = KEY["quant"] + "|wf:edit")) == 9000
    assert mp.lookup(family = "flux.1-dev", quant = KEY["quant"], speed_mode = "default") == 6051
    assert (
        mp.lookup(
            family = "flux.1-dev",
            quant = KEY["quant"],
            speed_mode = "default",
            phase = mp.PHASE_UNPADDED,
        )
        == 1864
    )
    legacy = _refine(qwen_pipe, plan)
    assert legacy.estimates.get("measured_learned_peak_mib") is None
    assert (
        legacy.estimates["measured_runtime_headroom_mib"]
        == fresh.estimates["measured_runtime_headroom_mib"]
    )
    assert legacy.resident_text_encoder_mib == fresh.resident_text_encoder_mib == 8917
    # a fresh txt2img sample is learned, the split works again, and the store is rewritten as version 2
    _record(**B200)
    assert mp.lookup(**KEY) == B200["peak"]
    assert dm.learned_phase_split(KEY["family"], KEY["quant"], KEY["speed_mode"]) is not None
    later = _refine(qwen_pipe, plan)
    assert later.resident_text_encoder_mib == 8917
    assert later.estimates["decode_lend_text_encoder_mib"] == 8917 - 4621
    data = json.loads(store.read_text(encoding = "utf-8"))
    assert data["version"] == mp._VERSION and set(data["entries"]) == {edit, other, base}
