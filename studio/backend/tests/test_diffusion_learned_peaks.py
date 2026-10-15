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
    monkeypatch.setattr(
        mp, "current_device_name", lambda: "NVIDIA L4", raising = False
    )  # the planner's card
    return path


@pytest.fixture
def klein_pipe(monkeypatch, store):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "torchao:Int8Tensor")
    return types.SimpleNamespace(_unsloth_phase_seeds = True)


def _refine(
    pipe,
    plan,
    family = "flux.2-klein",
    speed = "default",
):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


# ---------------------------------------------------------------------------------------------------- seeds


def test_family_missing_from_peak_table_is_seeded_from_phase_calibration(store):
    assert "flux.2-klein" not in dm._MEASURED_IMAGE_PEAK_MIB
    # worst of encoder 1516 / denoise 1160 / untiled decode 2645 x 1.15, rounded up to 256 MiB
    assert dm.measured_image_runtime_mib("flux.2-klein", "default", phase_seeds = True) == 3072
    assert dm.measured_image_runtime_mib("flux.2-klein", "max", phase_seeds = True) == 3328
    # eager tiers and unseeded families still keep the flat estimate
    assert dm.measured_image_runtime_mib("flux.2-klein", "off") is None
    assert dm.measured_image_runtime_mib("hidream", "default") is None
    # the peak table still wins where it has the family (Qwen-Image-2.1 decode is tiled: 1849, not 7648)
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "default") == 2304


def test_klein_16gb_keeps_whole_dit_resident(klein_pipe):
    plan = _klein_flat_plan()
    assert (
        plan.offload_policy == dm.OFFLOAD_GROUP
        and plan.stream_text_encoders
        and plan.stream_transformer
    )
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
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2000, device_name = "NVIDIA L4")
    assert not mp.record(**kw, width = 1024, height = 1024, peak_mib = 1500)  # never lowered
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2600)
    mp._cache.update(path = None, mtime = None, data = None)  # what a new process sees
    data = json.loads(store.read_text())
    assert data["entries"]["flux.2-klein|torchao:Int8Tensor|default|f1|NVIDIA L4"]["peaks"] == {
        "1048576": 2600
    }
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
    seed = dm.measured_image_runtime_mib(
        "flux.2-klein", "default", phase_seeds = True, quant = "torchao:Int8Tensor"
    )
    assert seed == 3072
    kw = dict(family = "flux.2-klein", quant = "torchao:Int8Tensor", speed_mode = "default")
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 1000)
    assert (
        dm.measured_image_runtime_mib(
            "flux.2-klein", "default", phase_seeds = True, quant = "torchao:Int8Tensor"
        )
        == 3072
    )
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 4000)
    # 4000 x 1.15 = 4600 -> 4608
    assert (
        dm.measured_image_runtime_mib(
            "flux.2-klein", "default", phase_seeds = True, quant = "torchao:Int8Tensor"
        )
        == 4608
    )
    # without a quant label (callers that never measured) the seed alone answers
    assert dm.measured_image_runtime_mib("flux.2-klein", "default", phase_seeds = True) == 3072


def test_unseeded_family_is_placed_from_its_learned_peak_on_the_next_load(klein_pipe):
    plan = _klein_flat_plan()
    assert (
        _refine(klein_pipe, plan, family = "some-new-dit") is plan
    )  # nothing measured yet: flat plan
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
    # seeded family: a peak recorded by this session's own render places the next load, never this one's requests
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    # 2048x2048: 4 x 2645 x 1.15 = 12167 -> 12288
    assert dm.measured_request_extra_mib(klein_pipe, width = 2048, height = 2048) == 12288 - 3072


def test_request_guard_of_an_unseeded_family_reads_its_learned_peak(klein_pipe):
    kw = dict(family = "some-new-dit", quant = "torchao:Int8Tensor", speed_mode = "default")
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 2600)
    new = _refine(klein_pipe, _klein_flat_plan(), family = "some-new-dit")
    assert new.estimates["measured_runtime_headroom_mib"] == 3072
    klein_pipe._unsloth_measured_reserve = (3072, "some-new-dit", "default", None, None)
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    mp.record(**kw, width = 1024, height = 1024, peak_mib = 4000)
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 4608 - 3072


# ---------------------------------------------------------------------------------------------------- video


def test_video_request_peaks_persist_across_processes(store, monkeypatch):
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
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
            dm._set_group_offload_failure(
                pipe, _OOM("CUDA out of memory. Tried to allocate 20.00 MiB")
            )
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


@pytest.fixture(autouse = True)
def _attention_scaling_known(request, monkeypatch):
    """Recording tests run without a device target; the SDPA verdict counts as known unless a test says otherwise."""
    if "attention_scaling_unknown" not in request.node.name:
        monkeypatch.setattr(
            "core.inference.diffusion._attention_scaling_known", lambda target, engaged = None: True
        )


class _Probe:
    def __init__(
        self,
        peak,
        failed = 0,
    ):
        self.device = "cuda:0"
        self.failed_captures = failed
        self._peak = peak

    def finish(self):
        return self._peak


def test_render_records_its_peak_unless_a_capture_failed(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    )
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default")
    kw = dict(family = "flux.2-klein", quant = "torchao:Int8Tensor", speed_mode = "default")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 1)
    diffusion._record_peak_probe(
        state, _Probe(5000, failed = 0), width = 1024, height = 1024, batch_size = 1
    )
    assert mp.lookup(**kw) is None  # an absorbed capture OOM: the peak is the failed attempt's
    diffusion._record_peak_probe(
        state, _Probe(2400, failed = 1), width = 1024, height = 1024, batch_size = 1
    )
    assert mp.lookup(**kw) == 2400
    assert (
        json.loads(store.read_text())["entries"][
            "flux.2-klein|torchao:Int8Tensor|default|f1|NVIDIA L4"
        ]["device"]
        == "NVIDIA L4"
    )


# ---------------------------------------------------------------------------------------------------- tied weights


def test_tied_group_closures():
    shared, a, b, c = object(), object(), object(), object()
    g1 = types.SimpleNamespace(t = [shared, a])
    g2 = types.SimpleNamespace(t = [b])
    g3 = types.SimpleNamespace(t = [c, shared])
    closures = dm._tied_group_closures([g1, g2, g3], lambda g: g.t)
    assert closures[id(g1)] == [g1, g3] and closures[id(g3)] == [g1, g3]
    assert id(g2) not in closures


def _tied_tiny_encoder(torch):
    """A Qwen3-style leaf: an embedding and an LM head sharing one weight (FLUX.2-klein's text encoder ties them)."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4096, 128)
            self.mid = torch.nn.Linear(128, 128)
            self.head = torch.nn.Linear(128, 4096, bias = False)
            self.head.weight = self.embed.weight

        def forward(self, ids):
            return self.head(self.mid(self.embed(ids)))

    return Tiny().to(torch.bfloat16)


def test_partial_residency_keeps_tied_weights_together():
    """Base: the embedding group went resident while the tied head streamed; the head's offload pointed the shared
    weight back at the host and the next encode failed (index on cuda, weight on cpu)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA and diffusers group offloading")
    from diffusers.hooks import apply_group_offloading

    te = _tied_tiny_encoder(torch)
    dev = torch.device("cuda")
    apply_group_offloading(
        te,
        onload_device = dev,
        offload_device = torch.device("cpu"),
        offload_type = "leaf_level",
        use_stream = True,
        non_blocking = True,
        record_stream = True,
    )
    # 1 MiB: room for the 1 MiB shared weight once, not for the embedding and the head counted apart
    dm._keep_groups_resident(te, 1, dev)
    groups = dm._offload_groups(te)
    tied = [g for g in groups if te.embed in g.modules or te.head in g.modules]
    assert len(tied) == 2
    assert len({bool(getattr(g, "_unsloth_resident", False)) for g in tied}) == 1
    for n in (7, 11, 5):
        with torch.no_grad():
            te(torch.randint(0, 4096, (1, n), device = dev))
    restore = dm.release_resident_groups(types.SimpleNamespace(components = {"text_encoder": te}), 1)
    assert not any(getattr(g, "_unsloth_resident", False) for g in tied)
    with torch.no_grad():
        te(torch.randint(0, 4096, (1, 9), device = dev))
    restore()
    assert all(getattr(g, "_unsloth_resident", False) for g in tied)
    with torch.no_grad():
        te(torch.randint(0, 4096, (1, 4), device = dev))


def test_video_peak_of_a_request_with_a_failed_capture_is_not_persisted(store, monkeypatch):
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "streaming", "cuda:0")
    work = 832 * 480 * 33

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return 9000 << 20

    module = types.SimpleNamespace(
        _unsloth_video_pending = (work, 0, "cuda:0"),
        _unsloth_video_calibration_key = key,
        _unsloth_video_failed_captures = 0,
    )
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 1)
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    assert vsr.record_request_peak(types.SimpleNamespace(transformer = module)) == 9000
    assert mp.work_peaks(key) == {}  # kept in this process only


def test_resident_dit_plan_clamps_encoder_room_and_says_so(klein_pipe):
    """L4 (22.5 GB): the flat plan already kept the DiT resident; the encoder room exceeded the encoders and the log
    read '0 MiB of the transformer and 10970 MiB of the 7673 MiB encoders'."""
    plan = _klein_flat_plan(budget_mib = 19500, total_mib = 23034)
    assert (
        plan.offload_policy == dm.OFFLOAD_GROUP
        and not plan.stream_transformer
        and plan.stream_text_encoders
    )
    new = _refine(klein_pipe, plan)
    assert new.resident_text_encoder_mib == 7629
    assert "transformer is resident" in new.reasons[-1]


# ---------------------------------------------------------------------------------------------------- review follow-ups


def test_render_served_by_a_conditioning_cache_is_not_recorded(store, monkeypatch):
    """A cache hit skips the text encoder: an unseeded family would place its next load from the smaller peak."""
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    stats = {"hits": 0, "misses": 1}
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("some-new-dit", "torchao:Int8Tensor", "default"),
        _unsloth_cond_cache_stats = stats,
    )
    state = types.SimpleNamespace(pipe = pipe, device = "cuda:0", speed_mode = "default")
    probe = _Probe(1800)
    probe.conditioning_hits = diffusion._conditioning_cache_hits(pipe)
    stats["hits"] += 1  # this render's prompt came from the cache
    diffusion._record_peak_probe(state, probe, width = 1024, height = 1024, batch_size = 1)
    kw = dict(family = "some-new-dit", quant = "torchao:Int8Tensor", speed_mode = "default")
    assert mp.lookup(**kw) is None
    probe = _Probe(2600)
    probe.conditioning_hits = diffusion._conditioning_cache_hits(pipe)
    diffusion._record_peak_probe(state, probe, width = 1024, height = 1024, batch_size = 1)
    assert mp.lookup(**kw) == 2600


def test_learned_peaks_are_kept_per_gpu_model(store):
    kw = dict(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor",
        speed_mode = "default",
        width = 1024,
        height = 1024,
    )
    mp.record(**kw, peak_mib = 2600, device_name = "NVIDIA L4")
    assert mp.lookup(**kw, device = "NVIDIA L4") == 2600
    assert mp.lookup(**kw, device = "NVIDIA GeForce RTX 4090") is None


def test_peak_store_update_waits_for_another_process_lock(store):
    """The read / merge / replace takes an interprocess lock, so a concurrent writer cannot replace a higher peak."""
    fcntl = pytest.importorskip("fcntl")
    import threading

    store.parent.mkdir(parents = True, exist_ok = True)
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default", width = 1024, height = 1024)
    done = threading.Event()
    with open(
        str(store) + ".lock", "a+b"
    ) as other:  # a separate open file description, like another process
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        worker = threading.Thread(
            target = lambda: (mp.record(**kw, peak_mib = 2000, device_name = "g"), done.set())
        )
        worker.start()
        assert not done.wait(0.5)  # blocked behind the other writer's lock
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)
    worker.join(10)
    assert done.is_set() and mp.lookup(**kw, device = "g") == 2000


def test_oom_after_partial_hooks_unwinds_and_replans(monkeypatch):
    """An OOM on the second DiT after the first had hooks used to re-raise past the lower-residency retry."""
    torch = pytest.importorskip("torch")
    hooks = pytest.importorskip("diffusers.hooks")
    from diffusers.hooks.hooks import HookRegistry, ModelHook

    calls = []
    marker = "_test_group_marker"

    def fake_apply(module, **kw):
        calls.append(module)
        if module is pipe.transformer_2 and calls.count(module) == 1:
            raise _OOM("CUDA out of memory. Tried to allocate 64.00 MiB")
        HookRegistry.check_if_exists_or_initialize(module).register_hook(ModelHook(), marker)

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(dm, "install_group_offload_buffer_restore", lambda: None)
    monkeypatch.setattr(dm, "install_group_offload_hooks_eager", lambda: None)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    real_remove = dm._remove_group_offload_hooks

    def remove(module):
        real_remove(module)
        registry = getattr(module, "_diffusers_hook", None)
        if registry is not None and registry.get_hook(marker) is not None:
            registry.remove_hook(marker, recurse = True)

    monkeypatch.setattr(dm, "_remove_group_offload_hooks", remove)
    a, b, te = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    pipe = types.SimpleNamespace(
        transformer = a,
        transformer_2 = b,
        components = {"transformer": a, "transformer_2": b, "text_encoder": te},
    )
    plan = dm.replace(_resident_dit_plan(), stream_transformer = True, stream_text_encoders = False)
    policy, _ = dm.apply_memory_plan(pipe, plan, device = "cpu")
    assert policy == dm.OFFLOAD_GROUP
    assert dm.applied_plan(pipe, plan).estimates["setup_oom_retry"]
    assert calls[:2] == [a, b] and calls[2:4] == [a, b]  # unwound, then both DiTs streamed again


# ---------------------------------------------------------------------------------------------------- review round 2


def test_peak_is_scaled_to_its_bucket_ceiling(store):
    kw = dict(family = "some-new-dit", quant = "torchao:Int8Tensor", speed_mode = "default")
    # 1088x1024 lands in the 2 MP bucket; a 2 MP request must not get the ~1 MP peak back unchanged
    mp.record(**kw, width = 1088, height = 1024, peak_mib = 1000)
    assert mp.lookup(**kw, width = 1448, height = 1448) >= 1000 * 2 * 1024 * 1024 // (1088 * 1024)


def test_request_guard_scales_the_learned_peak_the_load_placed_from(klein_pipe):
    mp.record(
        family = "flux.2-klein",
        quant = "torchao:Int8Tensor",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 4000,
    )
    new = _refine(klein_pipe, _klein_flat_plan())
    assert new.estimates["measured_runtime_headroom_mib"] == 4608
    klein_pipe._unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        4608,
        "flux.2-klein",
        "default",
        None,
        None,
        new.estimates.get("measured_learned_peak_mib"),
    )
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    # 2 MP: 2 x 4000 x 1.15 = 9200 -> 9216, not the 2 x 2645 seed
    assert dm.measured_request_extra_mib(klein_pipe, width = 1448, height = 1448) >= 9216 - 4608


def test_learned_peaks_are_keyed_by_attention_backend(klein_pipe):
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|attn:flash",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    plan = _klein_flat_plan()
    flash = dm.refine_plan_from_loaded_weights(
        klein_pipe, plan, family = "some-new-dit", speed_mode = "default", attention = "flash"
    )
    assert flash.estimates["measured_runtime_headroom_mib"] == 3072
    assert klein_pipe._unsloth_peak_identity[1] == "torchao:Int8Tensor|attn:flash"
    native = dm.refine_plan_from_loaded_weights(
        klein_pipe, plan, family = "some-new-dit", speed_mode = "default", attention = "native"
    )
    assert native is plan  # a native-attention load never reuses the flash peak


def test_load_passes_the_engaged_attention_backend_to_the_peak_identity():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert (
        "peak_model,\n                                peak_vae,\n                                peak_te,\n                                peak_cache,"
        in src
    )
    assert 'attention = attention_engaged or "native",' in src
    assert 'plan.estimates.get("measured_learned_peak_mib"),' in src


def test_video_peaks_merge_other_processes_updates(store, monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49
    assert vsr.measured_extra_mib(key, work) is None  # this process caches the empty answer
    assert mp.record_work(vsr._store_key(key), work, 3000)  # another process records
    assert vsr.measured_extra_mib(key, work) == 3000


def test_video_peaks_are_persisted_per_physical_gpu(store, monkeypatch):
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    module = types.SimpleNamespace(_unsloth_video_pending = None, _unsloth_video_calibration_key = key)
    vsr._PEAKS[key] = {work: 3000}
    module._unsloth_video_pending = (work, 0, "cuda:0")

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return 3000 << 20

    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", _Torch)
        vsr.record_request_peak(types.SimpleNamespace(transformer = module))
    stored = json.loads(store.read_text())["work"]
    assert list(stored) == [repr((key, "NVIDIA L4"))]
    vsr._PEAKS.clear()
    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA H100")  # another worker's cuda:0
    assert vsr.measured_extra_mib(key, work) is None


def test_store_cap_evicts_the_oldest_entry_not_the_alphabetically_first(store, monkeypatch):
    monkeypatch.setattr(mp, "_MAX_ENTRIES", 3)
    for fam in ("m", "n", "o"):
        mp.record(
            family = fam, quant = "q", speed_mode = "default", width = 1024, height = 1024, peak_mib = 100
        )
    mp.record(family = "a", quant = "q", speed_mode = "default", width = 1024, height = 1024, peak_mib = 100)
    entries = json.loads(store.read_text())["entries"]
    assert [k.split("|")[0] for k in entries] == ["n", "o", "a"]
    for fam in ("m", "n", "o"):
        assert mp.record_work(("k", fam), 10, 100)
    assert mp.record_work(("k", "a"), 10, 100)
    assert list(json.loads(store.read_text())["work"]) == [repr(("k", f)) for f in ("n", "o", "a")]


# ---------------------------------------------------------------------------------------------------- review round 3


def test_learned_peaks_are_keyed_by_the_loaded_model(klein_pipe):
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|model:org/dit-4b|attn:native",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    plan = _klein_flat_plan()
    kw = dict(family = "some-new-dit", speed_mode = "default", attention = "native")
    small = dm.refine_plan_from_loaded_weights(klein_pipe, plan, model = "org/dit-4b", **kw)
    assert small.estimates["measured_runtime_headroom_mib"] == 3072
    assert klein_pipe._unsloth_peak_identity[1] == "torchao:Int8Tensor|model:org/dit-4b|attn:native"
    # a larger variant of the same family never reuses the smaller one's peak
    assert dm.refine_plan_from_loaded_weights(klein_pipe, plan, model = "org/dit-9b", **kw) is plan


def test_load_passes_the_model_to_the_peak_identity():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert (
        "peak_model,\n                                peak_vae,\n                                peak_te,\n                                peak_cache,"
        in src
    )
    assert "model = peak_model," in src
    assert 'plan.estimates.get("measured_learned_peak_pixels"),' in src


def test_request_guard_keeps_the_learned_peaks_source_bucket(klein_pipe):
    mp.record(
        family = "flux.2-klein",
        quant = "torchao:Int8Tensor",
        speed_mode = "default",
        width = 1448,
        height = 1448,
        peak_mib = 8000,
    )
    new = _refine(klein_pipe, _klein_flat_plan())
    learned = new.estimates["measured_learned_peak_mib"]
    assert learned >= 8000
    headroom = new.estimates["measured_runtime_headroom_mib"]
    klein_pipe._unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        headroom,
        "flux.2-klein",
        "default",
        None,
        None,
        learned,
        new.estimates.get("measured_learned_peak_pixels"),
    )
    # the stored peak already covers 2 MP: a 2 MP request needs nothing beyond the load's reserve
    assert dm.measured_request_extra_mib(klein_pipe, width = 1448, height = 1448) == 0


# ---------------------------------------------------------------------------------------------------- review round 4


class _FakeCudaTorch:
    """Allocator-wide peak stats, as torch keeps them: a reset by one render erases another's high-water mark."""

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


def test_overlapping_peak_probes_on_one_card_record_nothing(monkeypatch):
    fake = _FakeCudaTorch()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    first = mp.PeakProbe("cuda:0")
    fake.alloc(5000)  # the first render's largest phase
    fake.free(5000)
    second = mp.PeakProbe("cuda:0")  # a render on the other executor resets the card's peak
    fake.alloc(100)
    fake.free(100)
    assert first.finish() is None  # would read 100 MiB, not 5000
    assert second.finish() is None
    alone = mp.PeakProbe("cuda:0")
    fake.alloc(700)
    assert alone.finish() == 700
    del alone
    dropped = mp.PeakProbe("cuda:0")  # a failed render never finishes its probe
    del dropped
    after = mp.PeakProbe("cuda:0")
    fake.alloc(300)
    assert after.finish() == 300  # not spoiled by the abandoned probe


def test_partial_hook_failure_in_the_lower_residency_retry_propagates(monkeypatch):
    calls = []

    def fake_apply(pipe, device, logger, **kw):
        calls.append(kw)
        if len(calls) == 1:
            dm._set_group_offload_failure(pipe, _OOM("CUDA out of memory"))
            return False
        raise RuntimeError("group offload failed after installing hooks on 1 module(s)")

    monkeypatch.setattr(dm, "_apply_group_offload", fake_apply)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    pipe = types.SimpleNamespace(components = {})
    with pytest.raises(RuntimeError, match = "after installing hooks"):
        dm.apply_memory_plan(pipe, _resident_dit_plan(), device = "cuda")
    assert len(calls) == 2


def test_learned_peaks_are_keyed_by_vae_savers(klein_pipe):
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|model:org/dit|vae:tiled+sliced|attn:native",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    plan = _klein_flat_plan()
    kw = dict(family = "some-new-dit", speed_mode = "default", attention = "native", model = "org/dit")
    tiled = dm.refine_plan_from_loaded_weights(klein_pipe, plan, vae = "tiled+sliced", **kw)
    assert tiled.estimates["measured_runtime_headroom_mib"] == 3072
    # an untiled load never reuses the tiled decode's peak
    assert dm.refine_plan_from_loaded_weights(klein_pipe, plan, vae = "full", **kw) is plan
    assert (
        dm.vae_saver_tag(True, True) == "tiled+sliced" and dm.vae_saver_tag(False, False) == "full"
    )


def test_render_with_a_vae_saver_the_key_lacks_is_not_probed(monkeypatch):
    import core.inference.diffusion as diffusion

    class _Probe:
        def __init__(self, device):
            self.device = "cuda:0"

    monkeypatch.setattr(mp, "PeakProbe", _Probe)
    monkeypatch.setattr(mp, "enabled", lambda: True)
    vae = types.SimpleNamespace(use_tiling = False, use_slicing = False)
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("some-new-dit", "q|model:m|vae:full|attn:native", "default"),
        vae = vae,
    )
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    state = types.SimpleNamespace(pipe = pipe, device = "cuda:0", speed_mode = "default")
    assert diffusion._start_peak_probe(state) is not None
    vae.use_tiling = True  # the activation guard tiled this request's decode
    assert diffusion._start_peak_probe(state) is None


def test_load_passes_the_vae_savers_to_the_peak_identity():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "peak_vae = planned_vae_saver_tag(pipe, plan)" in src
    assert "vae = peak_vae," in src


def test_video_request_overlapped_by_another_render_records_nothing(store, monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    work = 1280 * 704 * 49
    window = mp.open_window("cuda:0")
    image = mp.open_window("cuda:0")  # an image render armed on the same card mid-request
    module = types.SimpleNamespace(
        _unsloth_video_pending = (work, 0, "cuda:0"),
        _unsloth_video_window = ("cuda:0", window),
        _unsloth_video_calibration_key = None,
    )

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return 3000 << 20

    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", _Torch)
        assert vsr.record_request_peak(types.SimpleNamespace(transformer = module)) is None
    assert not mp.close_window("cuda:0", image)


# ---------------------------------------------------------------------------------------------------- review round 5


def test_learned_peaks_are_keyed_by_text_encoder_precision(klein_pipe):
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|model:org/dit|te:int8|vae:full|attn:native",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    plan = _klein_flat_plan()
    kw = dict(
        family = "some-new-dit", speed_mode = "default", attention = "native", model = "org/dit", vae = "full"
    )
    quantised = dm.refine_plan_from_loaded_weights(klein_pipe, plan, te = "int8", **kw)
    assert quantised.estimates["measured_runtime_headroom_mib"] == 3072
    # a dense encoder never reuses the quantised encoder's peak
    assert dm.refine_plan_from_loaded_weights(klein_pipe, plan, te = "dense", **kw) is plan
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "peak_te = _peak_te_label(te_quant, te_outcome)" in src and "te = peak_te," in src


def test_vae_key_names_only_the_savers_the_pipeline_can_engage():
    plan = types.SimpleNamespace(vae_tiling = True, vae_slicing = True)
    untileable = types.SimpleNamespace(
        vae = types.SimpleNamespace(enable_slicing = lambda: None, use_slicing = False)
    )
    assert dm.planned_vae_saver_tag(untileable, plan) == "sliced"
    both = types.SimpleNamespace(
        vae = types.SimpleNamespace(
            enable_slicing = lambda: None,
            enable_tiling = lambda: None,
            use_slicing = False,
            use_tiling = False,
        )
    )
    assert dm.planned_vae_saver_tag(both, plan) == "tiled+sliced"
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    # after placement the key follows the savers that actually engaged, and the reserve follows the key
    after_apply = src.index("plan = applied_plan(pipe, plan)")
    assert (
        'with_label_part(identity[1], "vae", live_vae_saver_tag(pipe))'
        in src[after_apply : after_apply + 800]
    )
    assert "pipe._unsloth_peak_static = True" in src[after_apply : after_apply + 800]


def test_deferred_profile_sizes_requests_from_the_flat_estimate(klein_pipe):
    klein_pipe._unsloth_phase_seeds = False
    klein_pipe._unsloth_peak_identity = ("some-new-dit", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        3072,
        "some-new-dit",
        "off",
        None,
        None,
        2600,
        1 << 20,
        False,
    )
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    klein_pipe._unsloth_peak_static = True  # the deferred profile engaged after placement
    flat = dm.estimate_image_runtime_mib(
        width = 1024, height = 1024, batch_size = 1, family = "some-new-dit"
    )
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == flat - 3072
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    deferred = src.index("def _engage_deferred_profile")
    assert "state.pipe._unsloth_peak_static = True" in src[deferred : deferred + 4000]


def test_render_under_another_attention_backend_than_the_key_is_not_probed(monkeypatch):
    import core.inference.diffusion as diffusion

    class _Probe:
        def __init__(self, device):
            self.device = "cuda:0"

    monkeypatch.setattr(mp, "PeakProbe", _Probe)
    monkeypatch.setattr(mp, "enabled", lambda: True)
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("x", "q|vae:full|attn:native", "default"),
        vae = types.SimpleNamespace(use_tiling = False, use_slicing = False),
    )
    state = types.SimpleNamespace(
        pipe = pipe, device = "cuda:0", attention_backend = None, speed_mode = "default"
    )
    assert diffusion._start_peak_probe(state) is not None
    state.attention_backend = "flash"
    assert diffusion._start_peak_probe(state) is None


def test_lower_residency_retry_returns_torchao_weights_without_module_to(monkeypatch):
    torch = pytest.importorskip("torch")

    class _Torchao(torch.nn.Module):
        def to(self, *args, **kwargs):
            raise RuntimeError("Module.to cannot move a torchao weight on its host copy")

    returned = []
    monkeypatch.setattr(
        dm, "_return_module_tensors", lambda module, onload: returned.append(onload.type)
    )
    calls = []

    def fake_apply(pipe, device, logger, **kw):
        calls.append(kw)
        if len(calls) == 1:
            dm._set_group_offload_failure(pipe, _OOM("CUDA out of memory"))
            return False
        return True

    monkeypatch.setattr(dm, "_apply_group_offload", fake_apply)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    pipe = types.SimpleNamespace(components = {"transformer": _Torchao()})
    policy, _ = dm.apply_memory_plan(pipe, _resident_dit_plan(), device = "cuda")
    assert policy == dm.OFFLOAD_GROUP and len(calls) == 2
    assert returned == ["cpu"]


def test_quadratic_attention_peaks_scale_with_pixels_squared(store):
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default")
    mp.record(**kw, width = 1088, height = 1024, peak_mib = 1000, quadratic = True)
    ratio = 2 * 1024 * 1024 / float(1088 * 1024)
    assert mp.lookup(**kw, width = 1448, height = 1448) >= int(1000 * ratio * ratio)
    # a request above every recorded bucket scales the largest one squared too
    assert mp.lookup(**kw, width = 2048, height = 2048) >= int(1000 * ratio * ratio) * 4


def test_record_passes_the_quadratic_attention_verdict():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "quadratic = _quadratic_attention(target, engaged, state.pipe)," in src
    assert "if not _attention_scaling_known(target, engaged):" in src
    assert "target = guard_target," in src


def test_peak_measured_with_no_streamed_denoiser_is_raised_to_the_streamed_baseline(
    store, monkeypatch
):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 900)
    resident = types.SimpleNamespace(components = {"transformer": object()})
    monkeypatch.setattr(
        dm, "_offload_groups", lambda module: [types.SimpleNamespace(_unsloth_resident = True)]
    )
    assert dm.stream_window_pad_mib(resident) == 900
    monkeypatch.setattr(
        dm, "_offload_groups", lambda module: [types.SimpleNamespace(_unsloth_resident = False)]
    )
    assert dm.stream_window_pad_mib(resident) == 0
    pipe = types.SimpleNamespace(_unsloth_peak_identity = ("some-new-dit", "q", "default"))
    probe = _Probe(2400)
    probe.window_pad = 900
    diffusion._record_peak_probe(
        types.SimpleNamespace(pipe = pipe, speed_mode = "default"),
        probe,
        width = 1024,
        height = 1024,
        batch_size = 1,
    )
    assert mp.lookup(family = "some-new-dit", quant = "q", speed_mode = "default") == 3300
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "window_pad = stream_window_pad_mib(pipe)" in src


def test_residency_fit_registers_its_video_measurement_window():
    src = (__import__("pathlib").Path(vsr.__file__)).read_text(encoding = "utf-8")
    fit = src.index("def fit_for_request")
    assert "dev = _open_window(module, dev)" in src[fit:]
    assert src.count("dev = _open_window(module, dev)") == 2


# ---------------------------------------------------------------------------------------------------- review round 6


def test_request_guard_scales_a_reserved_peak_to_the_request_bucket(klein_pipe):
    klein_pipe._unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        4608,
        "flux.2-klein",
        "default",
        None,
        None,
        4000,
        1 << 20,
        False,
    )
    # 1088x1024 crosses into the 2 MP bucket: 4000 x 2 x 1.15 = 9200 -> 9216, not 1.06x
    assert dm.measured_request_extra_mib(klein_pipe, width = 1088, height = 1024) == 9216 - 4608
    klein_pipe._unsloth_measured_reserve = (
        4608,
        "flux.2-klein",
        "default",
        None,
        None,
        4000,
        1 << 20,
        True,
    )
    # a math-attention entry scales squared: 4000 x 4 x 1.15 = 18400 -> 18432
    assert dm.measured_request_extra_mib(klein_pipe, width = 1088, height = 1024) == 18432 - 4608
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0


def test_peak_window_stays_open_until_the_high_water_mark_is_read(monkeypatch):
    fake = _FakeCudaTorch()
    read = fake.cuda.max_memory_allocated
    late = []

    def racing_read(dev = None):
        late.append(mp.open_window("cuda:0"))  # a render arms between the sync and the read
        return read(dev)

    fake.cuda.max_memory_allocated = staticmethod(racing_read)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    probe = mp.PeakProbe("cuda:0")
    fake.alloc(500)
    assert probe.finish() is None
    assert not mp.close_window("cuda:0", late[0])


def test_unobservable_vae_savers_use_and_record_no_learned_peak(store):
    pipe = types.SimpleNamespace(
        vae = types.SimpleNamespace(enable_slicing = lambda: None, use_tiling = False)
    )
    plan = types.SimpleNamespace(vae_tiling = False, vae_slicing = True)
    assert dm.planned_vae_saver_tag(pipe, plan) == "unknown"
    assert dm.live_vae_saver_tag(pipe) == "unknown"
    kw = dict(family = "some-new-dit", quant = "q|vae:unknown|attn:native", speed_mode = "default")
    assert not mp.record(**kw, width = 1024, height = 1024, peak_mib = 2000)
    assert mp.lookup(**kw) is None


def test_peak_model_identity_names_the_whole_build(monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_component_sizes as sizes

    revisions = {("org/dit", "dit-Q4.gguf"): "abc", ("org/base", None): "def"}
    monkeypatch.setattr(
        sizes, "checkpoint_fingerprint", lambda repo, filename = None: revisions.get((repo, filename))
    )
    assert diffusion._peak_model_identity("org/dit", "dit-Q4.gguf", "gguf", "org/base") == (
        "org/dit/dit-Q4.gguf@abc+org/base@def"
    )
    assert (
        diffusion._peak_model_identity("org/dit", "dit-Q4.gguf", "gguf", "org/other") == "unknown"
    )
    assert diffusion._peak_model_identity("org/gone", None, "pipeline", None) == "unknown"
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "peak_model = _peak_model_identity(repo_id, gguf_filename, kind, base, fetch_base)" in src


def test_peak_model_identity_reads_the_unsloth_mirror_on_a_fresh_install(monkeypatch):
    """Studio fetches a mirrored vendor base from unsloth/: with only the mirror cached, the key must still name a
    revision (the mirror's), or learned peaks are never recorded nor read on a default install."""
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_component_sizes as sizes
    import core.inference.diffusion_families as families

    revisions = {
        ("unsloth/FLUX.2-klein-4B", None): "m1",
        ("unsloth/dit-GGUF", "dit-Q4.gguf"): "g1",
    }
    monkeypatch.setattr(
        sizes, "checkpoint_fingerprint", lambda repo, filename = None: revisions.get((repo, filename))
    )
    monkeypatch.setattr(
        families,
        "mirror_repo",
        lambda repo: {"black-forest-labs/flux.2-klein-4b": "unsloth/FLUX.2-klein-4B"}.get(
            (repo or "").lower()
        ),
    )
    assert (
        diffusion._peak_model_identity("black-forest-labs/FLUX.2-klein-4B", None, "pipeline", None)
        == "unsloth/FLUX.2-klein-4B@m1"
    )
    assert diffusion._peak_model_identity(
        "unsloth/dit-GGUF", "dit-Q4.gguf", "gguf", "black-forest-labs/FLUX.2-klein-4B"
    ) == ("unsloth/dit-GGUF/dit-Q4.gguf@g1+unsloth/FLUX.2-klein-4B@m1")
    # No mirror and no cached revision: still unknown.
    assert diffusion._peak_model_identity("org/unmirrored", None, "pipeline", None) == "unknown"


def test_peak_model_identity_keys_on_the_repo_the_load_fetched(monkeypatch):
    """An upstream cache holding only model_index.json while the load pulled the weights from the mirror: the key
    names the mirror the load read, not the upstream id whose index happens to be cached (and vice versa)."""
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_component_sizes as sizes
    import core.inference.diffusion_families as families

    revisions = {
        ("black-forest-labs/FLUX.2-klein-4B", None): "u1",
        ("unsloth/FLUX.2-klein-4B", None): "m1",
        ("unsloth/dit-GGUF", "dit-Q4.gguf"): "g1",
    }
    monkeypatch.setattr(
        sizes, "checkpoint_fingerprint", lambda repo, filename = None: revisions.get((repo, filename))
    )
    monkeypatch.setattr(
        families,
        "mirror_repo",
        lambda repo: {"black-forest-labs/flux.2-klein-4b": "unsloth/FLUX.2-klein-4B"}.get(
            (repo or "").lower()
        ),
    )
    up, mirror = "black-forest-labs/FLUX.2-klein-4B", "unsloth/FLUX.2-klein-4B"
    assert diffusion._peak_model_identity(up, None, "pipeline", up, mirror) == f"{mirror}@m1"
    assert diffusion._peak_model_identity(up, None, "pipeline", up, up) == f"{up}@u1"
    assert diffusion._peak_model_identity(
        "unsloth/dit-GGUF", "dit-Q4.gguf", "gguf", up, mirror
    ) == f"unsloth/dit-GGUF/dit-Q4.gguf@g1+{mirror}@m1"
    # The load read the upstream bytes but only the mirror's revision is cached: no key, never the mirror's.
    del revisions[(up, None)]
    assert diffusion._peak_model_identity(up, None, "pipeline", up, up) == "unknown"
    assert diffusion._peak_model_identity(up, None, "pipeline", up, mirror) == f"{mirror}@m1"


def test_phase_table_seed_needs_its_measured_configuration(klein_pipe):
    assert dm.measured_image_runtime_mib("flux.2-klein", "default") is None
    plan = _klein_flat_plan()
    klein_pipe._unsloth_phase_seeds = False  # ROCm, or quadratic SDPA only
    assert _refine(klein_pipe, plan) is plan
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "pipe._unsloth_phase_seeds = _calibrated_activation(fam, target) is not None" in src


def test_oom_inside_the_first_offload_apply_unwinds_its_hooks(monkeypatch):
    torch = pytest.importorskip("torch")
    import diffusers.hooks as hooks

    def fake_apply(module, **kwargs):
        module._fake_hooked = True  # hooked part of the module, then ran out of memory
        raise _OOM("CUDA out of memory")

    unwound = []

    def fake_unwind(
        pipe,
        snapshots,
        logger = None,
    ):
        for module, _ in snapshots:
            module.__dict__.pop("_fake_hooked", None)
        unwound.append(True)
        return True

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    monkeypatch.setattr(
        dm, "_offload_groups", lambda m: [object()] if getattr(m, "_fake_hooked", False) else []
    )
    monkeypatch.setattr(dm, "_unwind_group_offload", fake_unwind)
    dit = torch.nn.Linear(4, 4)
    pipe = types.SimpleNamespace(transformer = dit, components = {"transformer": dit})
    assert dm._apply_group_offload(pipe, "cpu", None) is False
    assert unwound and not getattr(dit, "_fake_hooked", False)
    assert dm._is_oom(getattr(pipe, dm._GROUP_OFFLOAD_FAILURE_ATTR))


def test_video_window_stays_open_until_the_high_water_mark_is_read(store, monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    work = 1280 * 704 * 49
    module = types.SimpleNamespace(
        _unsloth_video_pending = (work, 0, "cuda:0"),
        _unsloth_video_window = ("cuda:0", mp.open_window("cuda:0")),
        _unsloth_video_calibration_key = None,
    )

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                mp.open_window("cuda:0")  # an image render arms between the decode and the read
                return 3000 << 20

    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", _Torch)
        assert vsr.record_request_peak(types.SimpleNamespace(transformer = module)) is None


# ---------------------------------------------------------------------------------------------------- review round 7


def _probe_state(monkeypatch, label, **live):
    import core.inference.diffusion as diffusion

    class _Probe:
        def __init__(self, device):
            self.device = "cuda:0"

    monkeypatch.setattr(mp, "PeakProbe", _Probe)
    monkeypatch.setattr(mp, "enabled", lambda: True)
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("x", label, "off"),
        vae = types.SimpleNamespace(use_tiling = False, use_slicing = False),
    )
    state = types.SimpleNamespace(
        pipe = pipe, device = "cuda:0", attention_backend = None, speed_mode = "off", transformer_cache = None
    )
    for name, value in live.items():
        setattr(state, name, value)
    return diffusion, state


def test_local_pipeline_identity_covers_its_weight_files(tmp_path):
    import core.inference.diffusion as diffusion

    (tmp_path / "transformer").mkdir()
    (tmp_path / "model_index.json").write_text("{}")
    (tmp_path / "transformer" / "config.json").write_text("{}")
    weights = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    weights.write_bytes(b"0" * 64)
    before = diffusion._peak_model_identity(str(tmp_path), None, "pipeline", None)
    weights.write_bytes(b"0" * 128)  # a heavier build, same manifests
    after = diffusion._peak_model_identity(str(tmp_path), None, "pipeline", None)
    assert before != "unknown" and after != "unknown" and before != after


def test_unknown_stream_window_records_no_peak(monkeypatch):
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: -1)
    assert dm.stream_window_pad_mib(types.SimpleNamespace(components = {})) is None
    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    state.pipe.components = {}
    assert diffusion._start_peak_probe(state) is None
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 700)
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", dm.stream_window_pad_mib)
    assert diffusion._start_peak_probe(state).window_pad == 700


def test_bare_unknown_or_empty_format_is_unidentified(store):
    for label in ("unknown", "", None, "unknown|model:x|attn:native"):
        assert not mp.identified(label)
        assert not mp.record(
            family = "some-new-dit",
            quant = label,
            speed_mode = "default",
            width = 1024,
            height = 1024,
            peak_mib = 900,
        )
    assert mp.identified("torchao:Int8Tensor|model:x|attn:native")


def test_render_with_a_toggled_step_cache_is_not_probed(monkeypatch):
    diffusion, state = _probe_state(monkeypatch, "q|cache:off|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    assert diffusion._start_peak_probe(state) is not None
    state.transformer_cache = "fbcache"  # a high-step request auto-engaged FBCache
    assert diffusion._start_peak_probe(state) is None
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert 'peak_cache = "off" if cache_may_toggle else str(cache_engaged or "off")' in src


def test_deferred_profile_retags_the_speed_tier(monkeypatch):
    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    state.speed_mode = "default"  # the deferred profile engaged the default tier
    assert (
        diffusion._start_peak_probe(state) is None
    )  # an off-tier key never takes a compiled render
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    deferred = src.index("def _engage_deferred_profile")
    block = src[deferred : deferred + 5000]
    assert (
        'with_label_part(identity[1], "attn", attention_engaged or "native"),\n                    SPEED_DEFAULT,'
        in block
    )


# ---------------------------------------------------------------------------------------------------- review round 8


def test_failed_video_request_closes_its_measurement_window():
    window = mp.open_window("cuda:0")
    module = types.SimpleNamespace(
        _unsloth_video_pending = (1, 0, "cuda:0"), _unsloth_video_window = ("cuda:0", window)
    )
    vsr.discard_request(types.SimpleNamespace(transformer = module))
    assert module._unsloth_video_pending is None and module._unsloth_video_window is None
    later = mp.open_window("cuda:0")  # an image render after the cancelled video
    assert mp.close_window("cuda:0", later)
    src = (__import__("pathlib").Path(vsr.__file__).parent / "video.py").read_text(encoding = "utf-8")
    assert 'video_stream_residency.discard_request(getattr(state, "pipe", None))' in src


def test_whole_module_offload_records_no_streamed_baseline_peak(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 700)
    monkeypatch.setattr(dm, "_offload_groups", lambda module: [])
    dit = torch.nn.Linear(
        4, 4
    )  # on the host at render start: whole-module offload onloads it inside the render
    assert dm.stream_window_pad_mib(types.SimpleNamespace(components = {"transformer": dit})) is None
    dit._hf_hook = object()
    assert dm.stream_window_pad_mib(types.SimpleNamespace(components = {"transformer": dit})) is None


def test_compile_fallback_during_the_render_records_nothing(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    pipe = types.SimpleNamespace(_unsloth_peak_identity = ("some-new-dit", "q", "default"))
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default", speed_optims = ("compiled",))
    state.speed_optims = ("compile_fallback_eager",)  # settle_compile_fallback mid-render
    diffusion._record_peak_probe(state, _Probe(2400), width = 1024, height = 1024, batch_size = 1)
    assert mp.lookup(family = "some-new-dit", quant = "q", speed_mode = "default") is None
    state.speed_optims = ("compiled",)
    diffusion._record_peak_probe(state, _Probe(2400), width = 1024, height = 1024, batch_size = 1)
    assert mp.lookup(family = "some-new-dit", quant = "q", speed_mode = "default") == 2400


def test_failed_capture_video_peak_never_reaches_the_keyed_or_persisted_peaks(store, monkeypatch):
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49
    captures = {"n": 0}
    monkeypatch.setattr(vsr, "_failed_captures", lambda: captures["n"])
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "failed_capture_count", lambda: captures["n"])

    def render(peak_mib):
        module = types.SimpleNamespace(
            _unsloth_video_pending = (work, 0, "cuda:0"),
            _unsloth_video_calibration_key = key,
            _unsloth_video_failed_captures = captures["n"],
        )

        class _Torch:
            class cuda:  # noqa: N801
                @staticmethod
                def max_memory_allocated(dev):
                    return peak_mib << 20

        return module, _Torch

    module, torch_stub = render(15000)
    captures["n"] += 1  # the capture failed during this render
    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", torch_stub)
        vsr.record_request_peak(types.SimpleNamespace(transformer = module))
    module, torch_stub = render(3000)
    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", torch_stub)
        vsr.record_request_peak(types.SimpleNamespace(transformer = module))
    assert mp.work_peaks(vsr._store_key(key)) == {work: 3000}
    assert vsr.measured_extra_mib(key, work) == 3000


# ---------------------------------------------------------------------------------------------------- review round 9


def test_graphs_left_by_an_unrecorded_render_stop_image_sampling(store, monkeypatch):
    import core.inference.diffusion_cuda_graph as cg

    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(diffusion, "_peak_device_index", lambda state: 0)
    live = {0: 3, 1: 0}
    monkeypatch.setattr(cg, "live_graph_count", lambda index: live.get(index, 0))
    assert diffusion._start_peak_probe(state) is not None  # the load's own graphs are the baseline
    live[1] = 5  # another card's graphs never count
    assert diffusion._start_peak_probe(state) is not None
    live[0] = 5  # a cache-served render recorded graphs on this card and was not sampled
    assert diffusion._start_peak_probe(state) is None
    live[0] = 0  # an OOM reset freed every graph: nothing unaccounted is left
    assert diffusion._start_peak_probe(state) is not None


def test_graphs_a_recorded_render_captured_keep_sampling_open(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(
        cg, "live_graph_count", lambda index: 4
    )  # recorded during this sampled render
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("some-new-dit", "q", "default"), _unsloth_peak_graphs = 0
    )
    diffusion._record_peak_probe(
        types.SimpleNamespace(pipe = pipe, speed_mode = "default"),
        _Probe(2400),
        width = 1024,
        height = 1024,
        batch_size = 1,
    )
    assert pipe._unsloth_peak_graphs == 4


def test_graphs_left_by_a_discarded_video_request_stop_video_sampling(store, monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(vsr, "_PEAKS", {})
    live = {0: 1}
    monkeypatch.setattr(cg, "live_graph_count", lambda index: live.get(index, 0))
    module = types.SimpleNamespace(_unsloth_video_calibration_key = None)
    pipe = types.SimpleNamespace(transformer = module)
    vsr._open_window(module, torch.device("cuda", 0))
    live[0] += 1  # the request recorded a graph, then was cancelled
    vsr.discard_request(pipe)
    vsr._open_window(module, torch.device("cuda", 0))
    module._unsloth_video_pending = (1280 * 704 * 49, 0, torch.device("cuda", 0))

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return 3000 << 20

    with monkeypatch.context() as m:
        m.setitem(__import__("sys").modules, "torch", _Torch)
        assert vsr.record_request_peak(pipe) is None


def test_live_graph_count_is_per_device_and_drops_reset_graphs():
    import core.inference.diffusion_block_graph as bg
    import core.inference.diffusion_cuda_graph as cg

    class _Wrapper:
        def __init__(self, device, n):
            self._unsloth_capture_device = device
            self.cache = dict.fromkeys(range(n))

    a, b = _Wrapper(0, 2), _Wrapper(1, 3)
    blocks = _Wrapper(0, 4)
    cg._LIVE_WRAPPERS.add(a)
    cg._LIVE_WRAPPERS.add(b)
    bg._LIVE_BLOCKS.add(blocks)
    try:
        assert cg.live_graph_count(0) == 6 and cg.live_graph_count(1) == 3
        a.cache.clear()  # reset_all
        assert cg.live_graph_count(0) == 4
    finally:
        cg._LIVE_WRAPPERS.discard(a)
        cg._LIVE_WRAPPERS.discard(b)
        bg._LIVE_BLOCKS.discard(blocks)


def test_render_without_the_familys_cfg_records_under_its_cfg_off_key(monkeypatch):
    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.delenv(dm.LEARNED_PEAKS_CFG_OFF_ENV, raising = False)
    state.family = types.SimpleNamespace(uses_negative_prompt = True)
    state.pipe._unsloth_peak_cfg = True
    assert diffusion._start_peak_probe(state) is not None
    assert "cfg:off" not in diffusion._live_peak_identity(state)[1]
    # guidance off (Z-Image-Turbo's default): half the denoiser batch, recorded under its own key, not skipped
    state.pipe._unsloth_peak_cfg = False
    assert diffusion._start_peak_probe(state) is not None
    assert diffusion._live_peak_identity(state)[1].endswith("|cfg:off")
    # an unknown guidance mode and CFG on a family that never runs it still record nothing
    state.pipe._unsloth_peak_cfg = None
    assert diffusion._start_peak_probe(state) is None
    state.family = types.SimpleNamespace(uses_negative_prompt = False)
    state.pipe._unsloth_peak_cfg = True
    assert diffusion._start_peak_probe(state) is None
    # kill switch: the old behaviour (a render without CFG records nothing)
    state.family = types.SimpleNamespace(uses_negative_prompt = True)
    state.pipe._unsloth_peak_cfg = False
    monkeypatch.setenv(dm.LEARNED_PEAKS_CFG_OFF_ENV, "0")
    assert diffusion._start_peak_probe(state) is None


def test_cfg_family_places_from_its_cfg_off_peak_and_cfg_requests_take_the_flat_estimate(
    klein_pipe, monkeypatch
):
    monkeypatch.delenv(dm.LEARNED_PEAKS_CFG_OFF_ENV, raising = False)
    klein_pipe._unsloth_peak_expects_cfg = True
    plan = _klein_flat_plan()
    kw = dict(family = "some-new-dit", speed_mode = "default", attention = "native")
    assert dm.refine_plan_from_loaded_weights(klein_pipe, plan, **kw) is plan
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|attn:native|cfg:off",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 2600,
    )
    new = dm.refine_plan_from_loaded_weights(klein_pipe, plan, **kw)
    assert new is not plan and new.estimates["measured_learned_cfg_off"] is True
    assert new.estimates["measured_runtime_headroom_mib"] == 3072
    klein_pipe._unsloth_peak_identity = (
        "some-new-dit",
        "torchao:Int8Tensor|attn:native",
        "default",
    )
    klein_pipe._unsloth_measured_reserve = (
        3072,
        "some-new-dit",
        "default",
        None,
        None,
        2600,
        1024 * 1024,
        False,
        True,
        True,
    )
    # a request without CFG is covered by the peak the load reserved
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024, cfg = False) == 0
    # a CFG request (or unknown) is sized from the flat estimate: the doubled batch was never measured
    flat = dm.estimate_image_runtime_mib(
        width = 1024, height = 1024, batch_size = 1, family = "some-new-dit"
    )
    assert (
        dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024, cfg = True)
        == flat - 3072
        > 0
    )
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == flat - 3072
    # once a CFG render is recorded, the CFG key wins
    mp.record(
        family = "some-new-dit",
        quant = "torchao:Int8Tensor|attn:native",
        speed_mode = "default",
        width = 1024,
        height = 1024,
        peak_mib = 3400,
    )
    cfg_new = dm.refine_plan_from_loaded_weights(klein_pipe, plan, **kw)
    assert cfg_new.estimates["measured_learned_cfg_off"] is False
    assert cfg_new.estimates["measured_runtime_headroom_mib"] == 4096
    # a family that never runs CFG never reads a cfg:off key
    klein_pipe._unsloth_peak_expects_cfg = False
    mp._cache.update(path = None, mtime = None, data = None)


def test_cfg_off_keys_are_wired_into_the_load_and_guard():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert (
        'pipe._unsloth_peak_expects_cfg = bool(getattr(fam, "uses_negative_prompt", False))' in src
    )
    assert "cfg = guard_cfg," in src
    assert 'plan.estimates.get("measured_learned_cfg_off", False),' in src
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "state.pipe._unsloth_peak_cfg = _negative_prompt_engaged(state.family, guidance)" in src


def test_render_forced_eager_while_the_compile_is_pending_is_not_probed(monkeypatch):
    import core.inference.diffusion_bg_compile as bg

    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    assert diffusion._start_peak_probe(state) is not None
    token = bg._FORCE_EAGER.set(True)
    try:
        assert diffusion._start_peak_probe(state) is None
    finally:
        bg._FORCE_EAGER.reset(token)


# ---------------------------------------------------------------------------------------------------- review round 10


def test_render_with_lora_adapters_is_not_probed(monkeypatch):
    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    assert diffusion._start_peak_probe(state) is not None
    state.pipe._unsloth_loras = (("style", "/loras/style.safetensors", 0.8),)
    assert diffusion._start_peak_probe(state) is None
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert '"unknown"  # adapters baked in at load' in src


def test_learned_reserve_falls_back_to_the_flat_estimate_with_loras_or_a_failed_saver(klein_pipe):
    klein_pipe._unsloth_phase_seeds = (
        False  # unseeded: the reserve came from the learned peak alone
    )
    klein_pipe._unsloth_peak_identity = ("some-new-dit", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        3072,
        "some-new-dit",
        "default",
        None,
        None,
        2600,
        1 << 20,
        False,
    )
    flat = dm.estimate_image_runtime_mib(
        width = 1024, height = 1024, batch_size = 1, family = "some-new-dit"
    )
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0
    klein_pipe._unsloth_loras = (("style", "/loras/style.safetensors", 0.8),)
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == flat - 3072
    klein_pipe._unsloth_loras = ()
    klein_pipe._unsloth_peak_static = True  # a planned VAE saver failed to engage
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == flat - 3072


def test_store_written_under_other_runtimes_is_ignored(store, monkeypatch):
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default")
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2000)
    assert mp.lookup(**kw) == 2000
    monkeypatch.setattr(mp, "_runtime", ["torch=9.9|diffusers=9.9|cuda=99"])  # an upgraded runtime
    mp._cache.update(path = None, mtime = None, data = None)
    assert mp.lookup(**kw) is None


def test_quadratic_peaks_scale_linearly_across_the_batch(store, klein_pipe):
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default")
    assert not mp.record(**kw, width = 1024, height = 1024, batch_size = 2, peak_mib = 3000, quadratic = True)
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 1000, quadratic = True)
    assert mp.lookup(**kw, width = 1024, height = 1024, batch_size = 2) == 2000  # not 4000
    assert mp.lookup(**kw, width = 1448, height = 1448) >= 4000  # pixels still square
    klein_pipe._unsloth_peak_identity = ("flux.2-klein", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        4608,
        "flux.2-klein",
        "default",
        None,
        None,
        4000,
        1 << 20,
        True,
    )
    # batch 2 at 1 MP: 4000 x 2 x 1.15 = 9200 -> 9216, not 4000 x 4
    assert (
        dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024, batch_size = 2)
        == 9216 - 4608
    )


# ---------------------------------------------------------------------------------------------------- review round 11


def test_resident_text_encoders_add_their_streaming_window(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_offload_prefetch as prefetch

    monkeypatch.setattr(prefetch, "prefetch_depth", lambda: 1)
    dit, encoder = torch.nn.Linear(4, 4), torch.nn.Linear(1024, 1024, bias = False)  # 4 MiB fp32 leaf
    streamed = types.SimpleNamespace(_unsloth_resident = False)
    resident = types.SimpleNamespace(_unsloth_resident = True)
    groups = {id(dit): [streamed], id(encoder): [resident]}  # streamed denoiser, resident encoder
    monkeypatch.setattr(dm, "_offload_groups", lambda module: groups.get(id(module), []))
    pipe = types.SimpleNamespace(components = {"transformer": dit, "text_encoder": encoder})
    assert dm.stream_window_pad_mib(pipe) == 8  # (depth + 1) x the largest encoder leaf
    groups[id(encoder)] = [streamed]  # streamed during the render: its window is in the peak
    assert dm.stream_window_pad_mib(pipe) == 0
    groups[id(encoder)] = []  # no hooks, weights on the host: placement unknown, record nothing
    assert dm.stream_window_pad_mib(pipe) is None


def test_video_peaks_are_persisted_on_the_streamed_baseline(store, monkeypatch):
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return 3000 << 20

    def render(pad):
        monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: pad)
        module = types.SimpleNamespace(
            _unsloth_video_pending = (work, 0, "cuda:0"), _unsloth_video_calibration_key = key
        )
        with monkeypatch.context() as m:
            m.setitem(__import__("sys").modules, "torch", _Torch)
            vsr.record_request_peak(types.SimpleNamespace(transformer = module))

    render(None)  # placement unknown: nothing persisted
    assert mp.work_peaks(vsr._store_key(key)) == {}
    render(700)  # every group resident: the prefetch window a streamed load holds is added
    assert mp.work_peaks(vsr._store_key(key)) == {work: 3700}
    assert vsr.measured_extra_mib(key, work) == 3700


# --------------------------------------------------------------------------------------------------- review round 12


def test_a_graph_replaced_in_a_full_cache_stops_image_sampling(store, monkeypatch):
    import core.inference.diffusion_cuda_graph as cg

    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(diffusion, "_peak_device_index", lambda state: 0)
    monkeypatch.setattr(cg, "live_graph_count", lambda index: 4)
    gen = {0: 7, 1: 0}
    monkeypatch.setattr(cg, "capture_generation", lambda index: gen.get(index, 0))
    assert diffusion._start_peak_probe(state) is not None  # the load's graphs are the baseline
    gen[1] = 3  # another card's captures never count
    assert diffusion._start_peak_probe(state) is not None
    gen[0] = 8  # a full cache evicted one graph and captured its replacement: same count, new graph
    assert diffusion._start_peak_probe(state) is None


def test_capture_generation_counts_per_device(monkeypatch):
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "_CAPTURES", {})
    cg.note_capture(0)
    cg.note_capture(0)
    cg.note_capture(1)
    assert (
        cg.capture_generation(0) == 2
        and cg.capture_generation(1) == 1
        and cg.capture_generation(2) == 0
    )


def test_runtime_stamp_names_installed_kernel_packages(monkeypatch):
    import importlib.metadata as md

    real = md.version

    def fake(dist):
        if dist == "sageattention":
            return "2.2.0"
        if dist in mp._KERNEL_DISTS:
            raise md.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(md, "version", fake)
    monkeypatch.setattr(mp, "_runtime", [])
    stamp = mp.runtime_stamp()
    assert "sageattention=2.2.0" in stamp and "xformers" not in stamp
    monkeypatch.setattr(mp, "_runtime", [])
    monkeypatch.setattr(md, "version", lambda d: "2.3.0" if d == "sageattention" else fake(d))
    assert mp.runtime_stamp() != stamp  # a kernel upgrade drops the stored peaks


def test_attention_scaling_unknown_render_is_not_recorded(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(
        diffusion, "_live_peak_identity", lambda state: ("some-new-dit", "q", "default")
    )
    monkeypatch.setattr(diffusion, "_conditioning_cache_hits", lambda pipe: 0)
    answers = {"math": False, "fused": False}
    monkeypatch.setattr(diffusion, "sdpa_math_only", lambda target: answers["math"])
    monkeypatch.setattr(diffusion, "sdpa_subquadratic_confirmed", lambda target: answers["fused"])
    target = types.SimpleNamespace(device = "cuda:0")
    assert not diffusion._attention_scaling_known(target, "native")  # the probe answered nothing
    assert not diffusion._attention_scaling_known(None, None)
    assert diffusion._attention_scaling_known(
        None, "sage"
    )  # an engaged fused backend is evidence on its own
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default", device = "NVIDIA L4")

    def render():
        pipe = types.SimpleNamespace(_unsloth_peak_identity = ("some-new-dit", "q", "default"))
        state = types.SimpleNamespace(pipe = pipe, speed_mode = "default", attention_backend = "native")
        diffusion._record_peak_probe(
            state, _Probe(2400), width = 1024, height = 1024, batch_size = 1, target = target
        )
        return pipe

    pipe = render()
    assert mp.lookup(**kw, width = 1024, height = 1024) is None
    assert (
        getattr(pipe, "_unsloth_peak_graphs", None) is None
    )  # an unrecorded render moves no graph baseline
    answers["fused"] = True
    render()
    assert mp.lookup(**kw, width = 1024, height = 1024) is not None


def test_partially_resident_denoiser_pads_the_window_difference(monkeypatch):
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 900)
    monkeypatch.setattr(dm, "_encoder_stream_window_mib", lambda pipe: 0)
    pipe = types.SimpleNamespace(components = {"transformer": object()})
    groups = [
        types.SimpleNamespace(_unsloth_resident = True),
        types.SimpleNamespace(_unsloth_resident = False),
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda module: groups)
    monkeypatch.setattr(dm, "_streamed_group_window_mib", lambda pipe: 300)
    assert (
        dm.stream_window_pad_mib(pipe) == 600
    )  # a fully streamed load holds the largest group, not the streamed one
    monkeypatch.setattr(dm, "_streamed_group_window_mib", lambda pipe: 900)
    assert dm.stream_window_pad_mib(pipe) == 0
    monkeypatch.setattr(dm, "_streamed_group_window_mib", lambda pipe: -1)
    assert dm.stream_window_pad_mib(pipe) is None  # unsized: not recorded
    groups[0]._unsloth_resident = False  # fully streamed: already the baseline
    assert dm.stream_window_pad_mib(pipe) == 0


def test_streamed_group_window_counts_only_streamed_groups(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_offload_prefetch as pf

    monkeypatch.setattr(pf, "prefetch_depth", lambda: 1)
    big, small = torch.nn.Linear(1024, 1024, bias = False), torch.nn.Linear(512, 512, bias = False)
    groups = [
        types.SimpleNamespace(_unsloth_resident = True, modules = [big], parameters = None, buffers = None),
        types.SimpleNamespace(
            _unsloth_resident = False, modules = [small], parameters = None, buffers = None
        ),
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda module: groups)
    pipe = types.SimpleNamespace(components = {"transformer": torch.nn.Module()})
    assert dm._streamed_group_window_mib(pipe) == 2  # 1 MiB (512 x 512 fp32) x (depth 1 + 1)


def test_a_graph_replaced_in_a_full_cache_stops_video_sampling(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(vsr, "_live_graphs", lambda dev: 4)
    gen = [7]
    monkeypatch.setattr(vsr, "_capture_generation", lambda dev: gen[0])
    monkeypatch.setattr("core.inference.diffusion_measured_peaks.open_window", lambda dev: 1)
    monkeypatch.setattr(
        "core.inference.diffusion_measured_peaks.close_window", lambda dev, token: True
    )
    module = types.SimpleNamespace()
    dev = torch.device("cuda", 0)
    vsr._open_window(module, dev)
    assert module._unsloth_video_graphs_clean  # the load's graphs are the baseline
    vsr._open_window(module, dev)
    assert module._unsloth_video_graphs_clean
    gen[0] = (
        8  # a discarded request evicted one graph and captured its replacement: same count, new graph
    )
    vsr._open_window(module, dev)
    assert not module._unsloth_video_graphs_clean


# --------------------------------------------------------------------------------------------------- review round 14


def test_unseeded_reserve_scales_without_the_store(store, klein_pipe):
    klein_pipe._unsloth_phase_seeds = False
    klein_pipe._unsloth_peak_identity = ("some-new-dit", "torchao:Int8Tensor", "default")
    klein_pipe._unsloth_measured_reserve = (
        3072,
        "some-new-dit",
        "default",
        None,
        None,
        2600,
        1 << 20,
        False,
    )
    assert (
        mp.lookup(family = "some-new-dit", quant = "torchao:Int8Tensor", speed_mode = "default") is None
    )
    # the store holds nothing (evicted, deleted, unwritable): a larger request still scales the reserved peak
    assert dm.measured_request_extra_mib(klein_pipe, width = 2048, height = 2048) > 0
    assert dm.measured_request_extra_mib(klein_pipe, width = 1024, height = 1024) == 0


def test_another_pipelines_reset_keeps_this_pipes_capture_dirty(monkeypatch):
    import core.inference.diffusion_cuda_graph as cg

    diffusion, state = _probe_state(monkeypatch, "q|vae:full|attn:native")
    monkeypatch.setattr(diffusion, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(diffusion, "_peak_device_index", lambda state: 0)
    live, gen = [6], [10]
    monkeypatch.setattr(cg, "live_graph_count", lambda index: live[0])
    monkeypatch.setattr(cg, "capture_generation", lambda index: gen[0])
    assert diffusion._start_peak_probe(state) is not None  # baseline: 6 graphs, generation 10
    live[0], gen[0] = 7, 11  # this pipe captured one graph in a render that was not recorded
    assert diffusion._start_peak_probe(state) is None
    live[0] = (
        3  # a video pipeline on the same card reset its 4 graphs; this pipe's capture is still alive
    )
    assert diffusion._start_peak_probe(state) is None
    live[0] = 0  # nothing alive on the card: nothing unaccounted can remain
    assert diffusion._start_peak_probe(state) is not None


# --------------------------------------------------------------------------------------------------- review round 15


def test_record_reports_whether_the_store_holds_the_sample(store):
    kw = dict(family = "some-new-dit", quant = "q", speed_mode = "default", width = 1024, height = 1024)
    held: list = []
    assert mp.record(**kw, peak_mib = 2000, held = held) and held == [True]
    held = []
    assert not mp.record(**kw, peak_mib = 1500, held = held) and held == [True]  # already covered
    held = []
    assert not mp.record_work(("k",), 10, 0, held = held) and held == []
    assert mp.record_work(("k",), 10, 100, held = held) and held == [True]


def test_unpersisted_image_sample_keeps_its_graphs_dirty(store, monkeypatch):
    import core.inference.diffusion as diffusion
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(mp, "device_name", lambda dev: "NVIDIA L4")
    monkeypatch.setattr(cg, "failed_capture_count", lambda: 0)
    monkeypatch.setattr(cg, "live_graph_count", lambda index: 4)
    monkeypatch.setattr(cg, "capture_generation", lambda index: 9)
    pipe = types.SimpleNamespace(
        _unsloth_peak_identity = ("some-new-dit", "q", "default"), _unsloth_peak_graphs = 0
    )
    state = types.SimpleNamespace(pipe = pipe, speed_mode = "default")
    monkeypatch.setattr(mp, "record", lambda **kw: False)  # the store is unavailable
    diffusion._record_peak_probe(state, _Probe(2400), width = 1024, height = 1024, batch_size = 1)
    assert pipe._unsloth_peak_graphs == 0
    monkeypatch.setattr(mp, "record", lambda **kw: kw["held"].append(True) or True)
    diffusion._record_peak_probe(state, _Probe(2400), width = 1024, height = 1024, batch_size = 1)
    assert pipe._unsloth_peak_graphs == 4 and pipe._unsloth_peak_graph_gen == 9


def test_unknown_local_video_build_persists_nothing():
    known = vsr.calibration_key(
        "wan", "/models/wan", "group", "cuda:0", revision = "/models/wan@local:ab:cd"
    )
    unknown = vsr.calibration_key("wan", "/models/wan", "group", "cuda:0", revision = "unknown")
    unread = vsr.calibration_key("wan", "/models/wan", "group", "cuda:0", revision = None)
    assert vsr._store_key(known) is not None
    assert vsr._store_key(unknown) is None and vsr._store_key(unread) is None
    src = (__import__("pathlib").Path(dm.__file__).parent / "video.py").read_text(encoding = "utf-8")
    assert "revision = _peak_model_identity(" in src


def test_local_weight_shards_change_the_video_revision(tmp_path, monkeypatch):
    import core.inference.diffusion as diffusion

    (tmp_path / "model_index.json").write_text("{}")
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}")
    shard = tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"
    shard.write_bytes(b"a" * 8)
    before = diffusion._peak_model_identity(str(tmp_path), None, "pipeline", None)
    shard.write_bytes(b"b" * 16)  # same manifests, new weights
    after = diffusion._peak_model_identity(str(tmp_path), None, "pipeline", None)
    assert before != "unknown" and before != after


def test_learned_peak_label_carries_the_graph_mode(klein_pipe, monkeypatch):
    import core.inference.diffusion as diffusion

    label = dm.learned_peak_label(klein_pipe, "native", "m", "none", "dense", "off", "on")
    assert label.endswith("|graphs:on")
    assert dm.learned_peak_label(klein_pipe, "native", "m", "none", "dense", "off", "off") != label
    state = types.SimpleNamespace(resolved = None, speed_optims = ("compiled",), cuda_graphs = ())
    assert diffusion._live_graph_mode(state) == "off"
    state.speed_optims = ("compiled", "cuda_graph")
    assert diffusion._live_graph_mode(state) == "step"
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert 'label = with_label_part(label, "graphs", _live_graph_mode(state))' in src
    assert "graphs = peak_graphs," in src


# --------------------------------------------------------------------------------------------------- review round 16


def test_whole_step_and_block_graphs_record_under_distinct_keys(store):
    import core.inference.diffusion as diffusion

    state = types.SimpleNamespace(
        resolved = None,
        speed_optims = ("compiled", "cuda_graph"),
        cuda_graphs = (),
        pipe = types.SimpleNamespace(_unsloth_cuda_graph_mode = "step"),
    )
    assert diffusion._live_graph_mode(state) == "step"
    state.pipe._unsloth_cuda_graph_mode = "blocks"
    assert diffusion._live_graph_mode(state) == "blocks"
    assert diffusion._engaged_graph_mode(state.pipe, {"cuda_graph": False}) == "off"
    # a load places before it knows the mode: it reads the larger of both modes' peaks
    base = dict(family = "flux.2-klein", speed_mode = "default", width = 1024, height = 1024)
    assert mp.record(**base, quant = "torchao:Int8Tensor|graphs:step", peak_mib = 3000)
    assert mp.lookup(**base, quant = "torchao:Int8Tensor|graphs:on") == 3000
    assert mp.record(**base, quant = "torchao:Int8Tensor|graphs:blocks", peak_mib = 4200)
    assert mp.lookup(**base, quant = "torchao:Int8Tensor|graphs:on") == 4200
    assert mp.lookup(**base, quant = "torchao:Int8Tensor|graphs:step") == 3000
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "_engaged_graph_mode(pipe, speed_applied)" in src


def test_torchao_variants_of_one_weight_class_do_not_share_a_label(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_compile_cache as cc

    monkeypatch.setattr(dm, "_torchao_weight_classes", lambda module: {"Int8Tensor"})
    pipe = types.SimpleNamespace(transformer = torch.nn.Linear(4, 4))
    variant = {"weights": {"Linear/Int8Tensor": 1}}
    monkeypatch.setattr(cc, "graph_variant", lambda dit: variant)
    plain = dm.denoiser_quant_label(pipe)
    assert plain.startswith("torchao:Int8Tensor#")
    variant = {"weights": {"Linear/Int8Tensor/convrot256": 1}}
    assert dm.denoiser_quant_label(pipe) != plain  # ConvRot: same class, other kernels
    variant = {"weights": {"Linear/Int8Tensor/act_int8": 1}}
    assert dm.denoiser_quant_label(pipe) != plain  # W8A8 vs weight-only
    variant = {"weights": "unavailable: RuntimeError"}
    assert not mp.identified(dm.denoiser_quant_label(pipe))  # unreadable: neither read nor recorded


def test_partially_resident_text_encoder_pads_the_window_difference(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_offload_prefetch as pf

    monkeypatch.setattr(pf, "prefetch_depth", lambda: 1)
    big, small = torch.nn.Linear(1024, 1024, bias = False), torch.nn.Linear(512, 512, bias = False)
    te = torch.nn.Sequential(big, small)
    groups = [
        types.SimpleNamespace(_unsloth_resident = True, modules = [big], parameters = None, buffers = None),
        types.SimpleNamespace(
            _unsloth_resident = False, modules = [small], parameters = None, buffers = None
        ),
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda module: groups)
    pipe = types.SimpleNamespace(components = {"text_encoder": te})
    # largest leaf 4 MiB x 2 = 8 MiB when fully streamed; the render streamed only the 1 MiB leaf (2 MiB in flight)
    assert dm._encoder_stream_window_mib(pipe) == 6
    groups[0]._unsloth_resident = False  # fully streamed: already the baseline
    assert dm._encoder_stream_window_mib(pipe) == 0


# --------------------------------------------------------------------------------------------------- review round 17


def test_graph_mode_envelope_covers_either_mode(store):
    base = dict(family = "flux.2-klein", speed_mode = "default")
    # a 3000 MiB step peak covering 4 MP, a 2900 MiB block peak covering 1 MP
    assert mp.record(**base, quant = "q|graphs:step", width = 2048, height = 2048, peak_mib = 3000)
    assert mp.record(**base, quant = "q|graphs:blocks", width = 1024, height = 1024, peak_mib = 2900)
    blocks = mp.lookup_bucket(**base, quant = "q|graphs:blocks", width = 1024, height = 1024)
    peak, pixels, quadratic = mp.lookup_bucket(**base, quant = "q|graphs:on", width = 1024, height = 1024)
    assert peak == 3000 and pixels == blocks[1]  # the larger peak over the smaller covered bucket
    assert not quadratic
    # a later 2 MP request scaled from the envelope covers what the block entry alone asks for
    want = mp.resolution_bucket(1448, 1448, 1)
    assert peak * want / pixels >= mp.lookup(
        **base, quant = "q|graphs:blocks", width = 1448, height = 1448
    )


def test_in_process_video_peak_is_on_the_streamed_baseline(monkeypatch):
    monkeypatch.setenv(mp.LEARNED_PEAKS_ENV, "0")  # persistence off: the keyed table alone answers
    monkeypatch.setattr(vsr, "_PEAKS", {})
    key = vsr.calibration_key("wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_mode = "default")
    work = 1280 * 704 * 49

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return (3000 + 100) << 20

    def run(pad):
        monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: pad)
        module = types.SimpleNamespace(_unsloth_video_pending = (work, 100 << 20, "cuda:0"))
        module._unsloth_video_calibration_key = key
        with monkeypatch.context() as m:
            m.setitem(__import__("sys").modules, "torch", _Torch)
            assert vsr.record_request_peak(types.SimpleNamespace(transformer = module)) == 3000
        return module

    run(None)  # unknown placement: nothing enters the keyed table
    assert vsr.measured_extra_mib(key, work) is None
    module = run(600)  # groups kept resident: the reused peak includes the prefetch window
    assert vsr.measured_extra_mib(key, work) == 3600
    assert module._unsloth_video_peaks[work] == 3000  # the module's own diagnostic stays raw


# --------------------------------------------------------------------------------------------------- review round 18


def test_capture_generation_increments_are_serialized(monkeypatch):
    import core.inference.diffusion_cuda_graph as dcg

    seen: list = []

    class _Counts(dict):
        def get(self, *a):
            lock = getattr(dcg, "_CAPTURES_LOCK", None)
            seen.append(lock is not None and lock.locked())
            return super().get(*a)

    monkeypatch.setattr(dcg, "_CAPTURES", _Counts())
    dcg.note_capture(0)
    dcg.note_capture(0)
    assert dcg.capture_generation(0) == 2
    assert seen and all(seen[:2])  # each read-modify-write held the lock


def test_poisoned_graph_load_records_no_video_calibration(monkeypatch):
    import core.inference.diffusion_cuda_graph as dcg

    monkeypatch.setenv(mp.LEARNED_PEAKS_ENV, "0")
    monkeypatch.setattr(vsr, "_PEAKS", {})
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    key = vsr.calibration_key(
        "wan2.2-ti2v-5b", "Wan-AI/x", "group", "cuda:0", speed_optims = ("compiled", "cuda_graph")
    )
    work = 1280 * 704 * 49
    handle = types.SimpleNamespace(poisoned = False)
    monkeypatch.setattr(
        dcg,
        "never_engaged",
        lambda handles: "capture failed" if all(h.poisoned for h in handles) else None,
    )

    class _Torch:
        class cuda:  # noqa: N801
            @staticmethod
            def max_memory_allocated(dev):
                return (3000 + 100) << 20

    def run():
        module = types.SimpleNamespace(_unsloth_video_pending = (work, 100 << 20, "cuda:0"))
        module._unsloth_video_calibration_key = key
        pipe = types.SimpleNamespace(transformer = module, _unsloth_cuda_graphs = (handle,))
        with monkeypatch.context() as m:
            m.setitem(__import__("sys").modules, "torch", _Torch)
            vsr.record_request_peak(pipe)

    handle.poisoned = True  # a failed capture poisoned the wrapper: later requests run eager
    run()
    assert vsr.measured_extra_mib(key, work) is None
    handle.poisoned = False  # graphs live: the request records
    run()
    assert vsr.measured_extra_mib(key, work) == 3000


# --------------------------------------------------------------------------------------------------- review round 19


def test_partially_quantized_text_encoders_neither_read_nor_record(klein_pipe):
    import core.inference.diffusion as diffusion
    from core.inference.diffusion_precision import TEQuantOutcome

    full = TEQuantOutcome("int8", None)
    partial = full._replace(partial = True)
    assert diffusion._peak_te_label("int8", full) == "int8"
    assert diffusion._peak_te_label(None, None) == "dense"
    te = diffusion._peak_te_label("int8", partial)
    label = dm.learned_peak_label(klein_pipe, "native", "m", "none", te, "off", "off")
    assert not mp.identified(label)  # a mixed build never reuses the fully quantised build's peak
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "peak_te = _peak_te_label(te_quant, te_outcome)" in src
