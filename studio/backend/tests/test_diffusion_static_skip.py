# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for the static step skip (``transformer_cache="static"``)."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference import diffusion_cache as dcache
from core.inference import diffusion_cuda_graph as cg
from core.inference import diffusion_step_skip as ss

from .test_diffusion_backend import (  # noqa: F401 - fake_runtime is a fixture
    _FakePipe,
    _load_into,
    fake_runtime,
)
from .test_diffusion_cuda_graph import _build_stub_torch, _FakeTensor


def test_schedule_matches_the_measured_prototype_at_25_steps():
    plan = ss.static_schedule(25)
    assert len(plan) == 25
    # head round(5.0) = 5, tail round(2.5) = 2 (banker's), the middle computes every other step.
    assert plan[:6] == (True,) * 6
    assert plan[-2:] == (True, True)
    assert [i for i, c in enumerate(plan) if not c] == list(range(6, 23, 2))


@pytest.mark.parametrize("steps", [None, 0, 1, 8, 11, "x"])
def test_short_or_unknown_trajectories_compute_every_step(steps):
    assert ss.static_schedule(steps) == ()


def test_schedule_keeps_two_head_steps_for_prefix_kv():
    plan = ss.static_schedule(12, head = 0.0)
    assert plan[:3] == (True, True, True)
    assert [i for i, c in enumerate(plan) if not c] == [3, 5, 7, 9]


def test_schedule_every_below_two_is_off():
    assert ss.static_schedule(30, every = 1) == ()


def test_settings_defaults_and_env_overrides():
    assert ss.static_skip_settings({}) == {
        "mode": "taylor1",
        "head": 0.2,
        "tail": 0.1,
        "every": 2,
    }
    env = {
        ss.ENV_MODE: "Reuse",
        ss.ENV_HEAD: "0.3",
        ss.ENV_TAIL: "0.2",
        ss.ENV_EVERY: "3",
    }
    assert ss.static_skip_settings(env) == {"mode": "reuse", "head": 0.3, "tail": 0.2, "every": 3}
    bad = {ss.ENV_MODE: "magic", ss.ENV_HEAD: "2", ss.ENV_TAIL: "x", ss.ENV_EVERY: "1"}
    assert ss.static_skip_settings(bad) == ss.static_skip_settings({})


torch = pytest.importorskip("torch")


class _Out(dict):
    def __init__(self, sample = None):
        super().__init__(sample = sample)

    def to_tuple(self):
        return tuple(self.values())

    @property
    def sample(self):
        return self["sample"]


class _DiT:
    def __init__(
        self,
        *,
        container = "tuple",
        prefix = 0,
    ):
        self.calls: list = []
        self.contexts: list = []
        self.container = container
        self.prefix = prefix  # extra rows returned on an extract call (prefix-KV DiT)

    @contextlib.contextmanager
    def cache_context(self, name):
        self.contexts.append(name)
        yield

    def forward(
        self,
        hidden_states,
        timestep = None,
        kv_cache_mode = None,
        return_dict = True,
    ):
        t = float(timestep.reshape(-1)[0])
        self.calls.append((t, kv_cache_mode))
        rows = hidden_states.shape[1] + (self.prefix if kv_cache_mode == "extract" else 0)
        out = torch.full((hidden_states.shape[0], rows, 2), t)
        if self.container == "pair":
            return (out, {"kv": 1}) if return_dict is False else _Out(sample = out)
        if self.container == "tensor":
            return out
        return (out,) if return_dict is False else _Out(sample = out)


def _pipe(dit, **extra):
    return types.SimpleNamespace(transformer = dit, **extra)


def _run(
    pipe,
    steps,
    *,
    contexts = ("cond",),
    signal = True,
    kv = False,
    return_dict = False,
):
    dit = pipe.transformer
    ss.reset_static_step_skip(pipe, steps, step_signal = signal)
    outs: dict = {c: [] for c in contexts}
    x = torch.zeros(1, 4, 2)
    for i in range(steps):
        t = torch.tensor([1.0 - i / steps])
        mode = ("extract" if i == 0 else "cached") if kv else None
        for name in contexts:
            cm = dit.cache_context(name) if name is not None else contextlib.nullcontext()
            with cm:
                out = dit.forward(
                    hidden_states = x, timestep = t, kv_cache_mode = mode, return_dict = return_dict
                )
            outs[name].append(out)
        if signal:
            ss.mark_step_end(pipe)
    return outs


def _installed(dit = None, **settings):
    dit = dit or _DiT()
    pipe = _pipe(dit)
    knobs = {**ss.static_skip_settings({}), "mode": "reuse", **settings}
    assert ss.install_static_step_skip(pipe, settings = knobs) == dcache.TC_STATIC
    return pipe


def test_install_is_outermost_and_never_sets_the_step_cache_marker():
    pipe = _installed()
    dit = pipe.transformer
    assert isinstance(dit.__dict__["forward"], ss.StaticStepSkip)
    assert dit.forward is dit.__dict__["forward"]
    assert getattr(dit, "_unsloth_step_cache", None) is None
    import inspect

    assert list(inspect.signature(dit.forward).parameters) == [
        "hidden_states",
        "timestep",
        "kv_cache_mode",
        "return_dict",
    ]


def test_cfg_contexts_are_counted_per_branch():
    pipe = _installed()
    outs = _run(pipe, 25, contexts = ("cond", "uncond"), signal = False)
    dit = pipe.transformer
    assert len(dit.calls) == 32
    stats = ss.static_skip_stats(pipe)["stats"]
    assert stats == {"calls": 50, "computed": 32, "skipped": 18}
    # A skipped step returns the SAME branch's last output, never the other branch's.
    plan = ss.static_schedule(25)
    for i, compute in enumerate(plan):
        if not compute:
            for name in ("cond", "uncond"):
                assert torch.equal(outs[name][i][0], outs[name][i - 1][0])
    assert dit.contexts.count("cond") == 25


def test_step_signal_counts_branches_by_ordinal_without_contexts():
    # FluxImg2Img / Kontext / Krea2: two calls per step and no cache_context.
    pipe = _installed()
    _run(pipe, 25, contexts = (None, None), signal = True)
    assert len(pipe.transformer.calls) == 32


def test_single_counter_fallback_without_context_or_signal():
    pipe = _installed()
    _run(pipe, 25, contexts = (None,), signal = False)
    assert len(pipe.transformer.calls) == 16


def test_reset_per_generation_restarts_the_schedule():
    pipe = _installed()
    _run(pipe, 25)
    _run(pipe, 25)
    assert len(pipe.transformer.calls) == 32
    _run(pipe, 8)
    assert len(pipe.transformer.calls) == 40
    ss.reset_static_step_skip(pipe, None)
    x = torch.zeros(1, 4, 2)
    for _ in range(3):
        pipe.transformer.forward(hidden_states = x, timestep = torch.tensor([0.5]), return_dict = False)
    assert len(pipe.transformer.calls) == 43


def test_reuse_returns_a_clone_of_the_last_output():
    pipe = _installed()
    outs = _run(pipe, 25)["cond"]
    skipped = outs[6][0]
    assert torch.equal(skipped, outs[5][0])
    assert skipped.data_ptr() != outs[5][0].data_ptr()


def test_taylor1_extrapolates_in_timestep():
    pipe = _installed(mode = "taylor1")
    outs = _run(pipe, 25)["cond"]
    # The fake output equals its timestep, so a first-order extrapolation is exact.
    for i in range(6, 23, 2):
        assert torch.allclose(outs[i][0], torch.full((1, 4, 2), 1.0 - i / 25), atol = 1e-6)


def test_per_token_timestep_extrapolates_from_its_max():
    """Wan2.2 TI2V passes ``mask * t`` per token; keyed on the first element (0), taylor1 silently reused."""

    class _TokenDiT:
        def __init__(self):
            self.calls = 0

        def forward(
            self,
            hidden_states = None,
            timestep = None,
            return_dict = True,
        ):
            self.calls += 1
            return (torch.full((1, 4), float(timestep.max())),)

    pipe = _pipe(_TokenDiT())
    knobs = {**ss.static_skip_settings({}), "mode": "taylor1"}
    assert ss.install_static_step_skip(pipe, settings = knobs) == dcache.TC_STATIC
    steps = 25
    ss.reset_static_step_skip(pipe, steps)
    outs = []
    for i in range(steps):
        timestep = torch.full((1, 8), 1.0 - i / steps)
        timestep[0, 0] = 0.0
        outs.append(pipe.transformer.forward(timestep = timestep, return_dict = False))
    assert pipe.transformer.calls == 16
    for i, compute in enumerate(ss.static_schedule(steps)):
        if not compute:
            assert torch.allclose(outs[i][0], torch.full((1, 4), 1.0 - i / steps), atol = 1e-6)


def test_prefix_kv_extract_step_is_never_reused():
    for mode in ("reuse", "taylor1"):
        pipe = _installed(_DiT(prefix = 3), mode = mode, head = 0.0)
        outs = _run(pipe, 12, kv = True)["cond"]
        assert outs[0][0].shape[1] == 7
        assert all(o[0].shape[1] == 4 for o in outs[1:])
        assert pipe.transformer.calls[0][1] == "extract"
        assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 4


def test_taylor1_falls_back_to_reuse_across_a_shape_change():
    pipe = _installed(_DiT(prefix = 3), mode = "taylor1")
    layer = pipe.transformer.__dict__["forward"]
    real_reset = layer.reset

    def reset(steps, **kwargs):
        # Force a skip at step 2, whose history is step 0 (7 rows, extract) and step 1 (4 rows).
        real_reset(steps, **kwargs)
        layer.plan = (True, True, False) + (True,) * (steps - 3)
        layer.last_skip = 2
        return layer

    layer.reset = reset
    outs = _run(pipe, 12, kv = True)["cond"]
    assert len(layer.history["cond"]) == 2
    assert torch.equal(outs[2][0], outs[1][0])
    assert layer.stats["skipped"] == 1


def test_container_types_are_preserved():
    for return_dict in (False, True):
        pipe = _installed()
        outs = _run(pipe, 25, return_dict = return_dict)["cond"]
        kind = tuple if return_dict is False else _Out
        assert all(type(o) is kind for o in outs)
    pipe = _installed(_DiT(container = "tensor"))
    outs = _run(pipe, 25)["cond"]
    assert all(torch.is_tensor(o) for o in outs)
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 9


def test_real_diffusers_output_class_is_rebuilt():
    outputs = pytest.importorskip("diffusers.models.modeling_outputs")
    value, rebuild = ss._split_output(outputs.Transformer2DModelOutput(sample = torch.ones(1)))
    rebuilt = rebuild(torch.zeros(1))
    assert type(rebuilt) is outputs.Transformer2DModelOutput
    assert torch.equal(rebuilt.sample, torch.zeros(1))


class _ListDiT(_DiT):
    def forward(
        self,
        hidden_states,
        timestep = None,
        kv_cache_mode = None,
        return_dict = True,
    ):
        out = super().forward(hidden_states, timestep, kv_cache_mode, return_dict = False)[0]
        return (list(out.unbind(0)),)


def test_z_image_list_outputs_are_skipped_and_rebuilt():
    value, rebuild = ss._split_output(([torch.ones(4, 2), torch.zeros(4, 2)],))
    assert value.shape == (2, 4, 2)
    rebuilt = rebuild(value * 2)
    assert type(rebuilt) is tuple and type(rebuilt[0]) is list and len(rebuilt[0]) == 2
    assert torch.equal(rebuilt[0][0], torch.full((4, 2), 2.0))
    assert ss._split_output(([torch.ones(4, 2), torch.ones(3, 2)],)) == (None, None)
    assert ss._call_signature(([torch.ones(1, 4), torch.ones(1, 4)],), {})[0] == ((1, 4), (1, 4))

    pipe = _installed(_ListDiT())
    outs = _run(pipe, 25)["cond"]
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 9
    assert all(type(o) is tuple and type(o[0]) is list for o in outs)


def test_z_image_list_in_output_dataclass_is_skipped_and_rebuilt():
    # ZImageImg2ImgPipeline / ZImageInpaintPipeline call the transformer without return_dict=False.
    outputs = pytest.importorskip("diffusers.models.modeling_outputs")
    out_cls = outputs.Transformer2DModelOutput
    value, rebuild = ss._split_output(out_cls(sample = [torch.ones(4, 2), torch.zeros(4, 2)]))
    assert value.shape == (2, 4, 2)
    rebuilt = rebuild(value * 2)
    assert type(rebuilt) is out_cls and type(rebuilt.sample) is list and len(rebuilt.sample) == 2
    assert torch.equal(rebuilt.sample[0], torch.full((4, 2), 2.0))
    assert ss._split_output(out_cls(sample = [torch.ones(4, 2), torch.ones(3, 2)])) == (None, None)

    class _ListOutDiT(_DiT):
        def forward(
            self,
            hidden_states,
            timestep = None,
            kv_cache_mode = None,
            return_dict = True,
        ):
            out = super().forward(hidden_states, timestep, kv_cache_mode, return_dict = False)[0]
            return out_cls(sample = list(out.unbind(0)))

    pipe = _installed(_ListOutDiT())
    outs = _run(pipe, 25)["cond"]
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 9
    assert all(type(o) is out_cls and type(o.sample) is list for o in outs)


class _PositionalDiT(_DiT):
    def forward(
        self,
        x,
        t,
        cap_feats,
        return_dict = True,
    ):
        return super().forward(x, timestep = t, return_dict = return_dict)


def test_taylor1_reads_a_positional_timestep():
    assert ss._timestep_slot(ss.inspect.signature(_PositionalDiT().forward)) == ("t", 1)
    assert ss._timestep_slot(ss.inspect.signature(_DiT().forward)) == ("timestep", 1)
    pipe = _installed(_PositionalDiT(), mode = "taylor1")
    dit = pipe.transformer
    ss.reset_static_step_skip(pipe, 25, step_signal = True)
    x = torch.zeros(1, 4, 2)
    outs = []
    for i in range(25):
        outs.append(dit.forward(x, torch.tensor([1.0 - i / 25]), None, return_dict = False))
        ss.mark_step_end(pipe)
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 9
    # Extrapolated, not reused: a skipped step matches its own timestep, not the previous step's.
    for i in range(6, 23, 2):
        assert torch.allclose(outs[i][0], torch.full((1, 4, 2), 1.0 - i / 25), atol = 1e-6)


def test_multi_value_outputs_are_never_skipped():
    # FLUX.2 klein KV's (noise, kv_cache): nothing a skip could reproduce.
    pipe = _installed(_DiT(container = "pair"))
    _run(pipe, 25)
    assert len(pipe.transformer.calls) == 25
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 0


def test_a_second_denoiser_or_missing_transformer_declines():
    assert ss.install_static_step_skip(_pipe(_DiT(), transformer_2 = _DiT())) is None
    assert ss.install_static_step_skip(_pipe(_DiT(), unconditional_transformer = _DiT())) is None
    assert ss.install_static_step_skip(types.SimpleNamespace()) is None


def test_uninstall_restores_forward_and_cache_context():
    dit = _DiT()
    pipe = _installed(dit)
    assert "cache_context" in dit.__dict__
    assert ss.uninstall_static_step_skip(pipe) is True
    assert "forward" not in dit.__dict__
    assert "cache_context" not in dit.__dict__
    assert "_unsloth_static_skip" not in dit.__dict__
    assert ss.uninstall_static_step_skip(pipe) is False
    _installed(dit)
    _run(pipe, 25)
    assert len(dit.calls) == 16


def test_uninstall_restores_a_prior_instance_forward():
    dit = _DiT()
    seen = []

    def prior(*args, **kwargs):
        seen.append(1)
        return type(dit).forward(dit, *args, **kwargs)

    dit.__dict__["forward"] = prior
    pipe = _installed(dit)
    _run(pipe, 25)
    assert len(seen) == 16
    ss.uninstall_static_step_skip(pipe)
    assert dit.__dict__["forward"] is prior


def test_uninstall_under_a_later_wrapper_disarms_in_place():
    dit = _DiT()
    pipe = _installed(dit)
    layer = dit.__dict__["forward"]

    def hook(*args, **kwargs):
        return layer(*args, **kwargs)

    dit.__dict__["forward"] = hook
    ss.uninstall_static_step_skip(pipe)
    assert dit.__dict__["forward"] is hook
    x = torch.zeros(1, 4, 2)
    for i in range(25):
        dit.forward(hidden_states = x, timestep = torch.tensor([1.0 - i / 25]), return_dict = False)
    assert len(dit.calls) == 25


def test_reinstall_is_idempotent():
    dit = _DiT()
    pipe = _installed(dit)
    layer = dit.__dict__["forward"]
    assert ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({})) == "static"
    assert dit.__dict__["forward"] is layer
    assert layer.inner is None


class _GraphDiT:
    def __init__(self):
        self.calls = 0

    def forward(
        self,
        hidden_states,
        timestep = None,
        return_dict = True,
    ):
        self.calls += 1
        return (_FakeTensor((1, 4), value = ("out", self.calls), tag = "out"),)


@pytest.fixture
def stub_torch(monkeypatch):
    stub = _build_stub_torch()
    stub.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", stub)
    # Replayed outputs are rebuilt as plain _FakeTensor clones, which the layer stores via detach().
    monkeypatch.setattr(_FakeTensor, "detach", lambda self: self, raising = False)
    cg._POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()
    yield stub
    cg._POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()


def _graph_loop(pipe, steps):
    ss.reset_static_step_skip(pipe, steps, step_signal = True)
    for _ in range(steps):
        pipe.transformer.forward(_FakeTensor((1, 4)), timestep = _FakeTensor((1,)), return_dict = False)
        ss.mark_step_end(pipe)


@pytest.mark.parametrize("graph_first", [False, True])
def test_graph_sits_under_the_layer_and_sees_only_computed_steps(stub_torch, graph_first):
    dit = _GraphDiT()
    pipe = types.SimpleNamespace(transformer = dit)
    knobs = ss.static_skip_settings({})
    if graph_first:
        handles = cg.install_cuda_graphs(pipe)
        ss.install_static_step_skip(pipe, settings = knobs)
    else:
        ss.install_static_step_skip(pipe, settings = knobs)
        handles = cg.install_cuda_graphs(pipe)
    layer = dit.__dict__["forward"]
    assert isinstance(layer, ss.StaticStepSkip)
    assert layer.inner is handles[0]
    assert getattr(dit, "_unsloth_step_cache", None) is None

    _graph_loop(pipe, 25)
    stats = handles[0].stats
    # One key for every computed step: one capture, the rest replays, nothing eager, no extra key.
    assert stats["captures"] == 1
    assert stats["replays"] == 16
    assert stats["eager_calls"] == 0
    assert len(handles[0].cache) == 1
    assert layer.stats["skipped"] == 9


def test_graph_uninstall_then_layer_uninstall_leaves_the_class_forward(stub_torch):
    dit = _GraphDiT()
    pipe = types.SimpleNamespace(transformer = dit)
    ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({}))
    handles = cg.install_cuda_graphs(pipe)
    cg.uninstall_all(handles)
    layer = dit.__dict__["forward"]
    assert isinstance(layer, ss.StaticStepSkip) and layer.inner is None
    ss.uninstall_static_step_skip(pipe)
    assert "forward" not in dit.__dict__


def test_layer_uninstall_hands_the_slot_back_to_the_graph(stub_torch):
    dit = _GraphDiT()
    pipe = types.SimpleNamespace(transformer = dit)
    ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({}))
    handles = cg.install_cuda_graphs(pipe)
    ss.uninstall_static_step_skip(pipe)
    assert dit.__dict__["forward"] is handles[0]
    handles[0].enable()
    assert dit.__dict__["forward"] is handles[0]
    cg.uninstall_all(handles)
    assert "forward" not in dit.__dict__
    ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({}))
    handles = cg.install_cuda_graphs(pipe)
    assert dit.__dict__["forward"].inner is handles[0]


def test_graph_disable_and_reenable_under_the_layer(stub_torch):
    dit = _GraphDiT()
    pipe = types.SimpleNamespace(transformer = dit)
    ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({}))
    handle = cg.install_cuda_graphs(pipe)[0]
    layer = dit.__dict__["forward"]
    handle.disable()
    assert dit.__dict__["forward"] is layer and layer.inner is None
    handle.enable()
    assert dit.__dict__["forward"] is layer and layer.inner is handle


def test_normalize_accepts_static_and_auto_stays_distinct():
    assert dcache.normalize_transformer_cache("static") == dcache.TC_STATIC
    assert dcache.normalize_transformer_cache(" Static ") == dcache.TC_STATIC
    assert dcache.normalize_transformer_cache("auto") == dcache.TC_AUTO


def test_only_fbcache_breaks_the_graph():
    assert dcache.cache_breaks_graph(None) is False
    assert dcache.cache_breaks_graph("static") is False
    assert dcache.cache_breaks_graph("fbcache") is True


def test_apply_step_cache_refuses_static():
    dit = _DiT()
    assert dcache.apply_step_cache(_pipe(dit), mode = "static") is None
    assert "forward" not in dit.__dict__


def test_auto_toggle_only_ever_engages_fbcache(monkeypatch):
    seen = []

    def fake_apply(pipe, *, mode, **kwargs):
        seen.append(mode)
        return mode

    monkeypatch.setattr(dcache, "apply_step_cache", fake_apply)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(enable_cache = lambda c: None))
    dcache.maybe_toggle_step_cache(pipe, steps = 50)
    assert seen == [dcache.TC_FBCACHE]


def _record_speed(monkeypatch, dmod):
    seen: dict = {}

    def fake_speed(pipe, target, **kwargs):
        seen["speed"] = kwargs
        return {"compiled": True}

    def fake_begin(**kwargs):
        seen["begin"] = kwargs["compile_kwargs"]
        return None

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(dmod, "apply_speed_optims", fake_speed)
    monkeypatch.setattr(dmod.compile_cache, "begin", fake_begin)
    return seen


@pytest.mark.parametrize("request_cache", ["off", "static"])
def test_static_load_keeps_the_uncached_graph_decisions(
    fake_runtime, tmp_path, monkeypatch, request_cache
):
    from core.inference import diffusion as dmod

    seen = _record_speed(monkeypatch, dmod)
    installs = []
    monkeypatch.setattr(
        dmod,
        "install_static_step_skip",
        lambda pipe, logger = None: installs.append(pipe) or dcache.TC_STATIC,
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    status = _load_into(backend, tmp_path, speed_mode = "default", transformer_cache = request_cache)
    assert seen["speed"]["cache_active"] is False
    assert seen["speed"]["cache_engaged"] is False
    assert seen["begin"]["fullgraph"] is True
    if request_cache == "static":
        assert len(installs) == 1
        assert status["transformer_cache"] == "static"
        assert status["resolved"]["transformer_cache"]["value"] == "static"
    else:
        assert installs == []
        assert status["transformer_cache"] is None
    assert backend._state.cache_auto is False
    backend.unload()


def test_fbcache_load_still_breaks_the_graph(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    seen = _record_speed(monkeypatch, dmod)
    monkeypatch.setattr(dmod, "apply_step_cache", lambda pipe, mode = None, **k: mode)
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    _load_into(backend, tmp_path, speed_mode = "default", transformer_cache = "fbcache")
    assert seen["speed"]["cache_active"] is True
    assert seen["speed"]["cache_engaged"] is True
    assert seen["begin"]["fullgraph"] is False
    backend.unload()


@pytest.mark.parametrize("request_cache", [None, "auto"])
def test_auto_load_never_installs_static(fake_runtime, tmp_path, monkeypatch, request_cache):
    from core.inference import diffusion as dmod

    modes = []
    monkeypatch.setattr(dmod, "default_generation_params", lambda *a, **k: (50, 4.0))
    monkeypatch.setattr(
        dmod, "apply_step_cache", lambda pipe, mode = None, **k: modes.append(mode) or None
    )
    monkeypatch.setattr(
        dmod, "install_static_step_skip", lambda *a, **k: pytest.fail("auto picked static")
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    _load_into(backend, tmp_path, transformer_cache = request_cache)
    assert modes == [dcache.TC_FBCACHE]
    backend.unload()


class _Tensorish:
    def __init__(self, value):
        self.value = value
        self.shape = (1, 4)

    def detach(self):
        return self

    def clone(self):
        return _Tensorish(self.value)


class _LoopDiT:
    def __init__(self):
        self.calls = 0

    @contextlib.contextmanager
    def cache_context(self, name):
        yield

    def forward(
        self,
        hidden_states = None,
        timestep = None,
        return_dict = True,
    ):
        self.calls += 1
        return (_Tensorish(self.calls),)


class _LoopPipe(_FakePipe):
    def __init__(self):
        super().__init__()
        self.transformer = _LoopDiT()

    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        cfg_trunc_ratio = None,
        **kwargs,
    ):
        for i in range(kwargs["num_inference_steps"]):
            for name in ("cond", "uncond"):
                with self.transformer.cache_context(name):
                    self.transformer.forward(hidden_states = None, timestep = None, return_dict = False)
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, None, {})
        return super().__call__(
            prompt = prompt,
            negative_prompt = negative_prompt,
            callback_on_step_end = callback_on_step_end,
            guidance_scale = guidance_scale,
            true_cfg_scale = true_cfg_scale,
            cfg_trunc_ratio = cfg_trunc_ratio,
            **kwargs,
        )


def _static_backend(tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    _load_into(backend, tmp_path)
    pipe = _LoopPipe()
    sys.modules["torch"].is_tensor = lambda obj: isinstance(obj, _Tensorish)
    assert ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({})) == "static"
    object.__setattr__(backend._state, "pipe", pipe)
    object.__setattr__(backend._state, "transformer_cache", dcache.TC_STATIC)
    return backend, pipe


def test_generate_runs_the_schedule_and_keeps_the_graph(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend, pipe = _static_backend(tmp_path, monkeypatch)
    bypass, resets = [], []
    monkeypatch.setattr(dmod.cuda_graph, "set_bypass", lambda handles, on: bypass.append(on))
    object.__setattr__(backend._state, "cuda_graphs", ("handle",))
    monkeypatch.setattr(
        type(backend), "_reset_step_cache", staticmethod(lambda p: resets.append(p))
    )
    backend.generate(prompt = "a sloth", steps = 25)
    assert pipe.transformer.calls == 32
    assert bypass == [False]
    assert resets == []
    backend.generate(prompt = "a sloth", steps = 25)
    assert pipe.transformer.calls == 64
    backend.generate(prompt = "a sloth", steps = 9)
    assert pipe.transformer.calls == 82
    backend.unload()
    assert "forward" not in pipe.transformer.__dict__


def test_generate_arms_the_schedule_with_the_effective_steps(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend, pipe = _static_backend(tmp_path, monkeypatch)
    armed = []
    real_reset = dmod.reset_static_step_skip
    monkeypatch.setattr(
        dmod,
        "reset_static_step_skip",
        lambda p, steps, **k: armed.append((steps, k.get("step_signal")))
        or real_reset(p, steps, **k),
    )
    # img2img at strength 0.5 denoises int(25 * 0.5) = 12 steps.
    monkeypatch.setattr(dmod, "effective_request_strength", lambda *a, **k: 0.5)
    backend.generate(prompt = "a sloth", steps = 25)
    assert armed[0] == (12, True)
    assert armed[-1] == (None, None)
    backend.unload()


def test_generate_fbcache_still_bypasses_and_resets(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    _load_into(backend, tmp_path)
    bypass, resets = [], []
    monkeypatch.setattr(dmod.cuda_graph, "set_bypass", lambda handles, on: bypass.append(on))
    object.__setattr__(backend._state, "cuda_graphs", ("handle",))
    object.__setattr__(backend._state, "transformer_cache", dcache.TC_FBCACHE)
    monkeypatch.setattr(
        type(backend), "_reset_step_cache", staticmethod(lambda p: resets.append(p))
    )
    backend.generate(prompt = "a sloth", steps = 25)
    assert bypass == [True]
    assert len(resets) == 1
    backend.unload()


def test_image_and_video_load_requests_accept_static():
    from pydantic import ValidationError

    from models.inference import DiffusionLoadRequest, VideoLoadRequest

    assert (
        DiffusionLoadRequest(model_path = "org/model", transformer_cache = "static").transformer_cache
        == "static"
    )
    assert (
        VideoLoadRequest(model_path = "org/model", transformer_cache = "static").transformer_cache
        == "static"
    )
    with pytest.raises(ValidationError):
        VideoLoadRequest(model_path = "org/model", transformer_cache = "magic")


def test_stats_of_the_last_generation_survive_the_post_render_reset():
    pipe = _installed()
    _run(pipe, 25)
    live = ss.static_skip_stats(pipe)["stats"]
    assert live["calls"] == 25 and live["skipped"] > 0
    ss.reset_static_step_skip(pipe, None)
    assert ss.static_skip_stats(pipe)["stats"] == live


def test_status_route_carries_the_last_generation_skip_counts():
    import inspect

    from core.inference import diffusion
    from models.inference import DiffusionStatusResponse

    pipe = _installed()
    _run(pipe, 25)
    ss.reset_static_step_skip(pipe, None)
    stats = ss.static_skip_stats(pipe)
    assert stats["stats"]["skipped"] > 0
    body = DiffusionStatusResponse(
        loaded = True, transformer_cache = "static", transformer_cache_stats = stats
    )
    assert body.model_dump()["transformer_cache_stats"]["stats"] == stats["stats"]
    assert DiffusionStatusResponse().transformer_cache_stats is None
    src = inspect.getsource(diffusion.DiffusionBackend.status)
    assert '"transformer_cache_stats": static_skip_stats(state.pipe)' in src


def test_stats_add_up_over_the_chunks_of_one_generation():
    # Two pipeline calls of one generation (a batch cap or an OOM split): the status keeps both chunks' counts.
    pipe = _installed()
    _run(pipe, 25)
    one = dict(ss.static_skip_stats(pipe)["stats"])
    skip = ss._find(pipe)
    real_reset = skip.reset
    skip.reset = lambda steps, **kw: real_reset(steps, **{**kw, "keep_stats": True})
    _run(pipe, 25)
    skip.reset = real_reset
    ss.reset_static_step_skip(pipe, None)
    assert ss.static_skip_stats(pipe)["stats"] == {k: 2 * v for k, v in one.items()}
    _run(pipe, 25)
    assert ss.static_skip_stats(pipe)["stats"] == one


def test_generate_keeps_stats_after_its_first_chunk():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    assert "keep_stats = static_chunks_run > 0" in src


def test_generate_drops_static_state_on_every_exit():
    import ast
    import inspect
    import textwrap

    from core.inference import diffusion

    tree = ast.parse(textwrap.dedent(inspect.getsource(diffusion.DiffusionBackend.generate)))
    finals = [
        ast.unparse(stmt)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for stmt in node.finalbody
    ]
    assert any("reset_static_step_skip(static_skip_pipe, None)" in f for f in finals)


def test_a_short_generation_reports_its_own_uncached_counts():
    pipe = _installed()
    _run(pipe, 25)
    ss.reset_static_step_skip(pipe, None)
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 9
    _run(pipe, 8)
    ss.reset_static_step_skip(pipe, None)
    assert ss.static_skip_stats(pipe)["stats"] == {"calls": 8, "computed": 8, "skipped": 0}
