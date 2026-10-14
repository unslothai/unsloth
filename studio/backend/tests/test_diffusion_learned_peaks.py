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
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA L4", raising = False)  # the planner's card
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
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2000, device_name = "NVIDIA L4")
    assert not mp.record(**kw, width = 1024, height = 1024, peak_mib = 1500)  # never lowered
    assert mp.record(**kw, width = 1024, height = 1024, peak_mib = 2600)
    mp._cache.update(path = None, mtime = None, data = None)  # what a new process sees
    data = json.loads(store.read_text())
    assert data["entries"]["flux.2-klein|torchao:Int8Tensor|default|f1|NVIDIA L4"]["peaks"] == {"1048576": 2600}
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
    assert json.loads(store.read_text())["entries"]["flux.2-klein|torchao:Int8Tensor|default|f1|NVIDIA L4"]["device"] == "NVIDIA L4"


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
    assert plan.offload_policy == dm.OFFLOAD_GROUP and not plan.stream_transformer and plan.stream_text_encoders
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
        _unsloth_peak_identity = ("some-new-dit", "torchao:Int8Tensor", "default"), _unsloth_cond_cache_stats = stats
    )
    state = types.SimpleNamespace(pipe = pipe, device = "cuda:0")
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
    kw = dict(family = "some-new-dit", quant = "torchao:Int8Tensor", speed_mode = "default", width = 1024, height = 1024)
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
    with open(str(store) + ".lock", "a+b") as other:  # a separate open file description, like another process
        fcntl.flock(other.fileno(), fcntl.LOCK_EX)
        worker = threading.Thread(target = lambda: (mp.record(**kw, peak_mib = 2000, device_name = "g"), done.set()))
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
        transformer = a, transformer_2 = b, components = {"transformer": a, "transformer_2": b, "text_encoder": te}
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
    assert 'learned_peak_label(pipe, attention_engaged or "native")' in src
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
        mp.record(family = fam, quant = "q", speed_mode = "default", width = 1024, height = 1024, peak_mib = 100)
    mp.record(family = "a", quant = "q", speed_mode = "default", width = 1024, height = 1024, peak_mib = 100)
    entries = json.loads(store.read_text())["entries"]
    assert [k.split("|")[0] for k in entries] == ["n", "o", "a"]
    for fam in ("m", "n", "o"):
        assert mp.record_work(("k", fam), 10, 100)
    assert mp.record_work(("k", "a"), 10, 100)
    assert list(json.loads(store.read_text())["work"]) == [repr(("k", f)) for f in ("n", "o", "a")]
