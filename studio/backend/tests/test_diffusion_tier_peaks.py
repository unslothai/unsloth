# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deferred profile's compiled tier priced from its own learned peak, else flat. FLUX.1-schnell bf16 L4 numbers."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion as dif  # noqa: E402
import core.inference.diffusion_memory as dm  # noqa: E402

_REAL_TIER_GRAPH_MODE = dif._tier_graph_mode

EAGER = "dense:bfloat16|model:m@1|te:dense|cache:off|vae:sliced|attn:native|graphs:off"
COMPILED = "dense:bfloat16|model:m@1|te:dense|cache:off|vae:sliced|attn:_native_cudnn|graphs:off"
HEADROOM = 4352  # the dense bf16 seed the L4 load reserved


@pytest.fixture
def store(tmp_path, monkeypatch):
    import core.inference.diffusion_measured_peaks as mp

    for name in (
        "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
        "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
        "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16",
        dm.TIER_PEAKS_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    monkeypatch.delenv(mp.LEARNED_PEAKS_ENV, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA L4", raising = False)
    return mp


def _pipe(**kw):
    pipe = types.SimpleNamespace(
        _unsloth_dense_tier_seeds = True,
        _unsloth_peak_identity = ("flux.1", COMPILED, "default"),
        _unsloth_measured_reserve = (
            HEADROOM,
            "flux.1",
            "off",
            22681,
            None,
            None,
            None,
            False,
            True,
            False,
            True,
        ),
    )
    setattr(pipe, dm.TIER_MOVED_ATTR, True)
    for key, value in kw.items():
        setattr(pipe, key, value)
    return pipe


def _flat_extra():
    flat = dm.estimate_image_runtime_mib(width = 1024, height = 1024, batch_size = 1, family = "flux.1")
    return max(0, int(flat) - HEADROOM)


def test_the_first_compiled_render_keeps_the_flat_estimate(store):
    """No measurement of the compiled tier yet: the flat estimate stands in (the L4's 3840 MiB release)."""
    pipe = _pipe()
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == _flat_extra() == 3840
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, tier_peak_mib = None) == 3840


def test_a_measured_compiled_peak_inside_the_reserve_releases_nothing(store):
    pipe = _pipe()
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, tier_peak_mib = 4096) == 0


def test_a_compiled_peak_above_the_reserve_still_streams_the_difference(store):
    """Compiled buffers can exceed eager: the guard releases exactly what the measured compiled peak needs."""
    pipe = _pipe()
    assert (
        dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, tier_peak_mib = 9984)
        == 9984 - HEADROOM
    )


def test_a_compiled_peak_below_the_eager_pricing_never_lowers_it(store):
    """The eager seed at 2 MP (8448 MiB) stands when the compiled tier measured less there."""
    pipe = _pipe()
    eager = dm.measured_request_extra_mib(
        types.SimpleNamespace(**{k: v for k, v in vars(pipe).items() if k != dm.TIER_MOVED_ATTR}),
        width = 2048,
        height = 1024,
    )
    assert eager == 8448 - HEADROOM
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 1024, tier_peak_mib = 5120) == eager


@pytest.mark.parametrize(
    "extra",
    [{"_unsloth_peak_static": True}, {"_unsloth_loras": ["a"]}],
)
def test_other_invalidations_keep_the_flat_estimate(store, extra):
    """A VAE saver that did not engage, or runtime LoRA adapters: no tier peak describes the pipeline."""
    pipe = _pipe(**extra)
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, tier_peak_mib = 4096) == 3840


def test_a_tier_peak_without_a_moved_tier_is_ignored(store):
    pipe = _pipe()
    delattr(pipe, dm.TIER_MOVED_ATTR)
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, tier_peak_mib = 99999) == 0


def test_kill_switch(store, monkeypatch):
    assert dm.tier_peaks_enabled()
    monkeypatch.setenv(dm.TIER_PEAKS_ENV, "0")
    assert not dm.tier_peaks_enabled()


def _state(
    pipe,
    speed = "default",
    graph_mode = "off",
):
    return types.SimpleNamespace(
        pipe = pipe,
        speed_mode = speed,
        speed_optims = ("compiled",),
        transformer_cache = None,
        attention_backend = "_native_cudnn",
        family = types.SimpleNamespace(uses_negative_prompt = False),
        device = "cuda:0",
        _graph_mode = graph_mode,
    )


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(dif, "live_vae_saver_tag", lambda pipe: "sliced")
    monkeypatch.setattr(dif, "_live_graph_mode", lambda state: state._graph_mode)
    monkeypatch.setattr(dif, "_tier_graph_mode", lambda state: state._graph_mode)
    monkeypatch.setattr(dif, "_peak_device_index", lambda state: 0)


def _record(store, label, speed, peak):
    assert store.record(
        family = "flux.1",
        quant = label,
        speed_mode = speed,
        width = 1024,
        height = 1024,
        peak_mib = peak,
        device_name = "NVIDIA L4",
    )


def _tier_peak(state, **kw):
    return dif._tier_request_peak(
        state, width = 1024, height = 1024, batch_size = 1, cfg = kw.pop("cfg", None), **kw
    )


def test_the_eager_peak_never_stands_in_for_the_compiled_tier(store, live):
    """The whole-module path (#15163) lets the eager key stand in; the request guard reads only the compiled tier."""
    _record(store, EAGER, "off", 3581)
    _record(store, COMPILED, "off", 3581)
    state = _state(_pipe())
    assert _tier_peak(state) is None
    keys = dif._learned_peak_keys(state, COMPILED, "default", eager_stand_in = False)
    assert all(tier == "default" for _, tier in keys)
    assert (COMPILED, "off") in dif._learned_peak_keys(state, COMPILED, "default")


def test_the_compiled_tier_reads_its_own_recorded_peak(store, live):
    state = _state(_pipe())
    _record(store, COMPILED, "default", 3900)
    # 3900 x 1.15 margin, rounded up to 256 MiB
    assert _tier_peak(state) == 4608


def test_each_graph_mode_answers_only_for_itself(store, live):
    """A capturing render's peak does not cover the renders after its graph is dropped (measured: 1447 MiB with the
    graph, 2546 MiB after the drop): a dropped graph prices flat until the eager-compiled renders record."""
    step = COMPILED.replace("graphs:off", "graphs:step")
    _record(store, step, "default", 1447)
    assert _tier_peak(_state(_pipe(), graph_mode = "off")) is None
    assert _tier_peak(_state(_pipe(), graph_mode = "step")) == 1792
    _record(store, COMPILED, "default", 2546)
    assert _tier_peak(_state(_pipe(), graph_mode = "off")) == 3072


def test_no_tier_peak_while_the_render_runs_eager_or_unkeyed(store, live, monkeypatch):
    _record(store, COMPILED, "default", 3900)
    assert _tier_peak(_state(_pipe(_unsloth_loras = ["x"]))) is None
    monkeypatch.setattr("core.inference.diffusion_bg_compile.eager_forced", lambda: True)
    assert _tier_peak(_state(_pipe())) is None


def test_no_tier_peak_before_the_profile_engaged(store, live):
    pipe = _pipe()
    delattr(pipe, dm.TIER_MOVED_ATTR)
    _record(store, COMPILED, "default", 3900)
    assert _tier_peak(_state(pipe)) is None


def test_the_identity_follows_the_graph_mode_the_session_keeps(live):
    blocks = COMPILED.replace("graphs:off", "graphs:blocks")
    pipe = _pipe(_unsloth_peak_identity = ("flux.1", COMPILED, "default"))
    dif._follow_live_graph_mode(_state(pipe, graph_mode = "blocks"))
    assert pipe._unsloth_peak_identity == ("flux.1", blocks, "default")
    dif._follow_live_graph_mode(_state(pipe, graph_mode = "off"))
    assert pipe._unsloth_peak_identity == ("flux.1", COMPILED, "default")
    eager = _pipe(_unsloth_peak_identity = ("flux.1", EAGER, "off"))
    delattr(eager, dm.TIER_MOVED_ATTR)
    dif._follow_live_graph_mode(_state(eager, graph_mode = "blocks"))
    assert eager._unsloth_peak_identity == ("flux.1", EAGER, "off")


def test_a_live_graph_keeps_the_release_it_was_captured_over(live, monkeypatch):
    """Placement a graph was captured over is held until no graph is live."""
    graphs = {"live": 0, "gen": 0}
    monkeypatch.setattr(dif.cuda_graph, "live_graph_count", lambda index: graphs["live"])
    monkeypatch.setattr(dif.cuda_graph, "capture_generation", lambda index: graphs["gen"])
    state = _state(_pipe())
    assert dif._hold_captured_release(state, 0) == 0
    before = graphs["gen"]
    graphs.update(live = 1, gen = 1)  # the render captured a graph over a 3840 MiB release
    dif._note_captured_release(state, 3840, before)
    assert dif._hold_captured_release(state, 0) == 3840
    assert dif._hold_captured_release(state, 5000) == 5000
    dif._note_captured_release(state, 0, graphs["gen"])
    assert dif._hold_captured_release(state, 0) == 3840
    graphs["live"] = 0  # dropped
    assert dif._hold_captured_release(state, 0) == 0
    assert state.pipe._unsloth_captured_release_mib is None


def test_nothing_is_held_before_the_profile_engaged(live, monkeypatch):
    monkeypatch.setattr(dif.cuda_graph, "capture_generation", lambda index: 5)
    pipe = _pipe()
    delattr(pipe, dm.TIER_MOVED_ATTR)
    state = _state(pipe)
    dif._note_captured_release(state, 3840, 0)
    assert getattr(pipe, "_unsloth_captured_release_mib", None) is None


class _Handle:
    def __init__(
        self,
        cache = 0,
        captures = 0,
        poisoned = False,
        declined = False,
    ):
        self.cache = [object()] * cache
        self.stats = {"captures": captures}
        self.poisoned = poisoned
        self.capture_error = None
        self.fallback_handle = None
        self.fallback = None
        self._declined = declined

    def _placement_declined(self):
        return self._declined


@pytest.mark.parametrize(
    "handle, mode",
    [
        (
            _Handle(),
            "step",
        ),  # armed, before its first capture: the render that captures keeps the key
        (_Handle(cache = 1, captures = 1), "step"),  # kept
        (_Handle(cache = 0, captures = 1), "off"),  # captured, then dropped
        (_Handle(poisoned = True), "off"),
        (_Handle(declined = True), "off"),
    ],
)
def test_the_tier_graph_mode_is_stable_across_the_first_capture(monkeypatch, handle, mode):
    monkeypatch.setattr(dif.cuda_graph, "live_graph_count", lambda index: len(handle.cache))
    monkeypatch.setattr(dif, "_peak_device_index", lambda state: 0)
    pipe = _pipe(_unsloth_cuda_graphs = (handle,))
    state = _state(pipe)
    state.speed_optims = ("compiled", "cuda_graph")
    state.cuda_graphs = (handle,)
    assert dif._tier_graph_mode(state) == mode
    state.speed_optims = ("compiled",)
    assert dif._tier_graph_mode(state) == "off"


def test_an_engaged_per_block_fallback_keys_its_own_graph_mode(store, live, monkeypatch):
    """A dropped whole-step graph whose calls now run per-block graphs (``fallback_handle``) is neither ``step`` nor
    ``off``: it reads and records under its own mode, so a step or off peak never prices it."""
    monkeypatch.setattr(dif.cuda_graph, "live_graph_count", lambda index: 0)
    monkeypatch.setattr(dif, "_tier_graph_mode", _REAL_TIER_GRAPH_MODE)
    whole = _Handle(cache = 0, captures = 1)  # its own graph dropped
    whole.fallback_handle = _Handle(cache = 2, captures = 2)  # per-block graphs live
    pipe = _pipe(_unsloth_cuda_graphs = (whole,))
    state = _state(pipe)
    state.speed_optims = ("compiled", "cuda_graph")
    state.cuda_graphs = (whole,)
    assert dif._tier_graph_mode(state) == dif.TIER_FALLBACK_GRAPH_MODE
    _record(store, COMPILED, "default", 2546)
    _record(store, COMPILED.replace("graphs:off", "graphs:step"), "default", 1447)
    assert _tier_peak(state) is None  # neither the off nor the step peak prices it
    dif._follow_live_graph_mode(state)
    assert pipe._unsloth_peak_identity[1].endswith("graphs:" + dif.TIER_FALLBACK_GRAPH_MODE)
    _record(store, pipe._unsloth_peak_identity[1], "default", 3000)
    assert _tier_peak(state) == 3584  # its own entry: 3000 x 1.15, rounded up to 256 MiB
    # not engaged yet (the fallback is only armed): the whole step's own mode
    armed = _Handle()
    armed.fallback_handle = None
    state.cuda_graphs = (armed,)
    pipe._unsloth_cuda_graphs = (armed,)
    assert dif._tier_graph_mode(state) == "step"
