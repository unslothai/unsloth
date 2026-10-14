# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Learned activation peaks (``diffusion_measured_peaks``), the per-phase seed for families the peak table lacks,
and the one lower-residency retry after a transient OOM during group-offload setup (``diffusion_memory``).

The FLUX.2-klein-4B case uses the estimates a real auto load logged on a 16 GB card (int8 DiT, bf16 text encoder):
``safe_device_budget_mib 13132``, group offload with the text encoders streamed, and the whole DiT streamed although
the load peaked at 4.6 GiB. CPU-only; loaded sizes and the torchao check are stubbed.
"""

from __future__ import annotations

import json
import types

import pytest

import core.inference.diffusion_measured_peaks as mp
import core.inference.diffusion_memory as dm
import core.inference.video_stream_residency as vsr

KLEIN_LOADED = {
    "transformer": (4091, "dit"),
    "text_encoder": (7629, "text_encoder"),
    "vae": (160, "other"),
}
KLEIN_FLAT = dict(model_dense_mib = 11911, companion_dense_mib = 7820, text_encoder_dense_mib = 7629)


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _klein_flat_plan(budget_mib: int = 13132, total_mib: int = 16380):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
        requested_mode = None,
        **KLEIN_FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "peaks.json"
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(path))
    monkeypatch.delenv(mp.LEARNED_PEAKS_ENV, raising = False)
    monkeypatch.delenv(dm.MEASURED_ACTIVATION_ENV, raising = False)
    monkeypatch.delenv(dm.PARTIAL_RESIDENT_ENV, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    return path


@pytest.fixture
def klein_pipe(monkeypatch, store):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "torchao:Int8Tensor")
    return types.SimpleNamespace()


def _refine(pipe, plan, family = "flux.2-klein", speed = "default"):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


# ---------------------------------------------------------------------------------------------------- seeds


def test_family_missing_from_peak_table_is_seeded_from_phase_calibration(store):
    assert "flux.2-klein" not in dm._MEASURED_IMAGE_PEAK_MIB
    # worst of encoder 1516 / denoise 1160 / untiled decode 2645 x 1.15, rounded up to 256 MiB
    assert dm.measured_image_runtime_mib("flux.2-klein", "default") == 3072
    assert dm.measured_image_runtime_mib("flux.2-klein", "max") == 3328
    # eager tiers and unseeded families still keep the flat estimate
    assert dm.measured_image_runtime_mib("flux.2-klein", "off") is None
    assert dm.measured_image_runtime_mib("hidream", "default") is None
    # the peak table still wins where it has the family (Qwen-Image-2.1 decode is tiled: 1849, not 7648)
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "default") == 2304


def test_klein_16gb_keeps_whole_dit_resident(klein_pipe):
    plan = _klein_flat_plan()
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_text_encoders and plan.stream_transformer
    assert plan.resident_transformer_mib is None
    new = _refine(klein_pipe, plan)
    assert new is not plan
    assert new.resident_transformer_mib == 4091  # every DiT group resident
    assert new.estimates["measured_runtime_headroom_mib"] == 3072
    assert klein_pipe._unsloth_peak_identity == ("flux.2-klein", "torchao:Int8Tensor", "default")


# ---------------------------------------------------------------------------------------------------- store


def test_bucket_is_power_of_two_megapixels():
    assert mp.resolution_bucket(256, 256) == mp._MIN_BUCKET_PIXELS
    assert mp.resolution_bucket(1024, 1024) == 1024 * 1024
    assert mp.resolution_bucket(1280, 720) == 1024 * 1024
    assert mp.resolution_bucket(1024, 1024, 2) == 2 * 1024 * 1024
    assert mp.resolution_bucket(1536, 1536) == 4 * 1024 * 1024


def test_record_keeps_max_and_survives_a_fresh_process(store):
    kw = dict(family = "flux.2-klein", quant = "torchao:Int8Tensor", speed_mode = "default")
    assert mp.lookup(**kw) is None
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2000, device_name = "L4")
    assert not mp.record(**kw, width = 1024, height = 1024, peak_mib = 1500)  # never lowered
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2600)
    mp._cache.update(path = None, mtime = None, data = None)  # what a new process sees
    data = json.loads(store.read_text())
    assert data["entries"]["flux.2-klein|torchao:Int8Tensor|default|f1"]["peaks"] == {"1048576": 2600}
    assert mp.lookup(**kw, width = 1024, height = 1024) == 2600
    # a smaller request answers from the covering bucket, a larger one scales the largest linearly
    assert mp.lookup(**kw, width = 512, height = 512) == 2600
    assert mp.lookup(**kw, width = 2048, height = 2048) == 4 * 2600
    # other quant / tier / frame count: separate keys
    assert mp.lookup(family = "flux.2-klein", quant = "dense:bfloat16", speed_mode = "default") is None
    assert mp.lookup(**{**kw, "speed_mode": "max"}) is None
    assert mp.lookup(**kw, frames = 49) is None


def test_store_off_and_corrupt_file(store, monkeypatch):
    kw = dict(family = "z-image", quant = "q", speed_mode = "default")
    store.write_text("{not json")
    assert mp.lookup(**kw) is None
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 1000)
    assert mp.lookup(**kw) == 1000
    monkeypatch.setenv(mp.LEARNED_PEAKS_ENV, "0")
    assert mp.lookup(**kw) is None
    assert not mp.record(**kw, width = 1024, height = 1024, peak_mib = 5000)


def test_learned_peak_raises_a_seed_but_never_lowers_it(store):
    seed = dm.measured_image_runtime_mib("flux.2-klein", "default", quant = "torchao:Int8Tensor")
    assert seed == 3072
    kw = dict(family = "flux.2-klein", quant = "torchao:Int8Tensor", speed_mode = "default")
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 1000)
    assert dm.measured_image_runtime_mib("flux.2-klein", "default", quant = "torchao:Int8Tensor") == 3072
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 4000)
    # 4000 x 1.15 = 4600 -> 4608
    assert dm.measured_image_runtime_mib("flux.2-klein", "default", quant = "torchao:Int8Tensor") == 4608
    # without a quant label (callers that never measured) the seed alone answers
    assert dm.measured_image_runtime_mib("flux.2-klein", "default") == 3072


def test_unseeded_family_is_placed_from_its_learned_peak_on_the_next_load(klein_pipe):
    plan = _klein_flat_plan()
    assert _refine(klein_pipe, plan, family = "some-new-dit") is plan  # nothing measured yet: flat plan
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    new = _refine(klein_pipe, plan, family = "some-new-dit")
    assert new.resident_transformer_mib == 4091
    assert new.estimates["measured_runtime_headroom_mib"] == 3072  # 2600 x 1.15 -> 3072


def test_request_guard_reads_the_learned_peak(klein_pipe):
    plan = _klein_flat_plan()
    new = _refine(klein_pipe, plan)
    klein_pipe._unsloth_measured_reserve = (3072, "flux.2-klein", "default", None, None)
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    mp.record(
        family = "flux.2-klein",
        quant = "torchao:Int8Tensor",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 4000,
    )
    assert new is not plan
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 4608 - 3072


# ---------------------------------------------------------------------------------------------------- video


def test_video_request_peaks_persist_across_processes(store, monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49
    assert vsr.measured_extra_mib(key, work) is None

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return (3000 + 100) << 20

    module = types.SimpleNamespace(_unsloth_video_pending = (work, 100 << 20, "cuda:0"))
    module._unsloth_video_calibration_key = key
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    assert vsr.record_request_peak(types.SimpleNamespace(transformer = module)) == 3000
    monkeypatch.undo()
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(store))
    monkeypatch.setattr(vsr, "_PEAKS", {})  # a new process
    mp._cache.update(path = None, mtime = None, data = None)
    assert vsr.measured_extra_mib(key, work) == 3000
    assert vsr.measured_extra_mib(key, work + 1) is None  # a larger request is not covered


# ---------------------------------------------------------------------------------------------------- OOM retry


class _OOM(RuntimeError):
    pass


_OOM.__name__ = "OutOfMemoryError"


def _resident_dit_plan():
    mem = _memory(13132, 16380)
    return dm.MemoryPlan(
        requested_mode = "auto",
        offload_policy = dm.OFFLOAD_GROUP,
        vae_tiling = False,
        vae_slicing = True,
        device_memory = mem,
        estimates = {"safe_device_budget_mib": 13132},
        reasons = ("flat",),
        stream_text_encoders = True,
        stream_transformer = False,
    )


def test_transient_setup_oom_replans_once_at_lower_residency(monkeypatch):
    calls = []

    def fake_apply(pipe, device, logger, **kw):
        calls.append(kw)
        if len(calls) == 1:
            dm._set_group_offload_failure(pipe, _OOM("CUDA out of memory. Tried to allocate 20.00 MiB"))
            return False
        return True

    monkeypatch.setattr(dm, "_apply_group_offload", fake_apply)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    pipe = types.SimpleNamespace(components = {})
    plan = _resident_dit_plan()
    policy, _ = dm.apply_memory_plan(pipe, plan, device = "cuda")
    assert policy == dm.OFFLOAD_GROUP
    assert calls[0] == {"stream_text_encoders": True, "stream_transformer": False}
    assert calls[1] == {"stream_text_encoders": True}  # the DiT streams on the retry
    applied = dm.applied_plan(pipe, plan)
    assert applied is not plan and applied.stream_transformer and applied.stream_text_encoders
    assert applied.resident_transformer_mib is None and applied.estimates["setup_oom_retry"]
    assert not dm.plan_keeps_transformer_resident(applied)


def test_non_oom_setup_failure_still_refuses(monkeypatch):
    def fake_apply(pipe, device, logger, **kw):
        dm._set_group_offload_failure(pipe, ValueError("no block list"))
        return False

    monkeypatch.setattr(dm, "_apply_group_offload", fake_apply)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    pipe = types.SimpleNamespace(components = {})
    with pytest.raises(RuntimeError, match = "could not be streamed beside the resident quantised"):
        dm.apply_memory_plan(pipe, _resident_dit_plan(), device = "cuda")
    assert dm.applied_plan(pipe, "plan") == "plan"


def test_oom_retry_that_fails_again_refuses(monkeypatch):
    def fake_apply(pipe, device, logger, **kw):
        dm._set_group_offload_failure(pipe, _OOM("out of memory"))
        return False

    monkeypatch.setattr(dm, "_apply_group_offload", fake_apply)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    pipe = types.SimpleNamespace(components = {})
    with pytest.raises(RuntimeError, match = "could not be streamed beside the resident quantised"):
        dm.apply_memory_plan(pipe, _resident_dit_plan(), device = "cuda")


# ---------------------------------------------------------------------------------------------------- GGUF pinning


def test_background_pinner_pins_gguf_host_copies():
    """GGUF host copies were skipped (``type(src) is torch.Tensor``), so a streamed GGUF denoiser never had a pinned
    host copy and its CUDA graph refused capture ("a streamed group's host copy is not pinned")."""
    torch = pytest.importorskip("torch")
    gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")
    gguf = gguf_utils.GGUFParameter(torch.zeros(4, 144, dtype = torch.uint8), quant_type = 12)
    dense = torch.zeros(8)
    group = types.SimpleNamespace(cpu_param_dict = {"w": gguf.data, "b": dense})
    pinner = dm._GroupPinner.__new__(dm._GroupPinner)
    picked = {key for key, _ in pinner._unpinned(group)}
    assert picked == {"w", "b"}


# ---------------------------------------------------------------------------------------------------- recording


class _Probe:
    def __init__(self, peak, failed = 0):
        self.device = "cuda:0"
        self.failed_captures = failed
        self._peak = peak

    def finish(self):
        return self._peak


def test_render_records_its_peak_unless_a_capture_failed(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    pipe = types.SimpleNamespace(_unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default"))
    state = types.SimpleNamespace(pipe = pipe)
    kw = dict(family = "flux.2-klein", quant = "torchao:Int8Tensor", speed_mode = "default")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 1)
    diffusion._record_peak_probe(state, _Probe(5000, failed = 0), width = 1024, height = 1024, batch_size = 1)
    assert mp.lookup(**kw) is None  # an absorbed capture OOM: the peak is the failed attempt's
    diffusion._record_peak_probe(state, _Probe(2400, failed = 1), width = 1024, height = 1024, batch_size = 1)
    assert mp.lookup(**kw) == 2400
    assert json.loads(store.read_text())["entries"]["flux.2-klein|torchao:Int8Tensor|default|f1"]["device"] == "NVIDIA L4"
