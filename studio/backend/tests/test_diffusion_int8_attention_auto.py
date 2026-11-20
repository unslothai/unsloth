# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The unset switch on strided processors (``install_auto``): a DiT listed in ``AUTO_DITS`` races its incumbent like
the dispatch route (accuracy check, card race, 2048 / long races, ``MIN_TOKENS`` floor); every other DiT, speed off,
ROCm, the kill switches and an explicit attention request leave its processors as they were. CPU: the card is
faked."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D


@pytest.fixture(autouse = True)
def _no_recorded_graph():
    """A test that fakes a capture leaves no recorded graph behind (it freezes every later plan's decision)."""
    I._GRAPHED[0] = False
    yield
    I._GRAPHED[0] = False


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_cudnn"


class _Attention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _pipe(cls_name: str = "Flux3Transformer2DModel", head_dims = (128, 128, 64, 96)):
    dit = type(cls_name, (torch.nn.Module,), {})()
    dit.blocks = torch.nn.ModuleList([_Attention(2, d) for d in head_dims])
    return types.SimpleNamespace(transformer = dit)


def _flags(pipe):
    return [
        (m.processor._unsloth_int8, m.processor._unsloth_int8_min_tokens)
        for m in pipe.transformer.blocks
    ]


@pytest.fixture
def card(monkeypatch):
    """A faked sm89 card; ``found`` is what the accuracy check + card race return; ``wins`` the extra race points."""
    from core.inference import diffusion_flux3_attn, rocm_bf16

    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(D.DISPATCH_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "_secondary_cards", lambda *a, **k: ())
    strided = []
    monkeypatch.setattr(
        diffusion_flux3_attn,
        "ensure_strided_processors",
        lambda pipe, target, logger = None: strided.append(pipe) or 4,
    )
    state = types.SimpleNamespace(
        found = {"ok": True, "min_tokens": 0, "impl": "cuda"}, wins = {}, seen = [], strided = strided
    )

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        state.seen.append((heads, head_dim, incumbent_name, timed))
        return dict(state.found)

    monkeypatch.setattr(I, "verdict", verdict)
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: state.wins.get(a[-1], False))
    return state


def test_listed_family_races_and_floors_at_min_tokens(card):
    """Won at both race sizes: the floor is MIN_TOKENS (a forced load runs every length), so short text-only calls
    stay on the incumbent. Timed, never forced; the head dim the card has no kernel for is left alone."""
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe) == [(True, D.MIN_TOKENS)] * 3 + [(False, 0)]
    assert card.seen and all(timed for *_, timed in card.seen)
    assert {name for _, _, name, _ in card.seen} == {"_native_cudnn"}
    assert card.strided == [pipe]
    # the forced path keeps its own floor (0): the request by name is unchanged
    forced = _pipe()
    assert (
        I.install_int8_attention(
            forced.transformer, device = "cuda", dtype = torch.bfloat16, force = True
        )
        == 3
    )
    assert _flags(forced)[:3] == [(True, 0)] * 3


def test_floor_follows_the_extra_races(card):
    """Won at 4096 only: the 2048 race places the floor; lost at both: only the long race can route (long calls)."""
    card.found = {"ok": True, "min_tokens": 2560, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe)[0] == (True, (4096 + D.MID_RACE_TOKENS) // 2)
    card.found = {"ok": True, "min_tokens": None, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0
    assert _flags(pipe) == [(False, 0)] * 4
    card.wins[D.LONG_RACE_TOKENS[0]] = True
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe)[0] == (True, sum(D.LONG_RACE_TOKENS) // 2)


def test_accuracy_check_failure_keeps_the_incumbent(card):
    card.found = {"ok": False, "min_tokens": None, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0
    assert _flags(pipe) == [(False, 0)] * 4


@pytest.mark.parametrize(
    "case",
    [
        "unlisted",
        "speed_off",
        "env_off",
        "env_force",
        "dispatch_off",
        "rocm",
        "cpu",
        "no_transformer",
    ],
)
def test_auto_route_stays_off(card, monkeypatch, case):
    """Never races: an unlisted DiT, speed off (bit-identical), the kill switches, the env switch 1 (the forced
    install runs instead), ROCm, a CPU load, a pipeline with no transformer."""
    from core.inference import rocm_bf16

    pipe = _pipe("LTX2VideoTransformer3DModel" if case == "unlisted" else "Flux3Transformer2DModel")
    target = _TARGET
    speed = True
    if case == "speed_off":
        speed = False
    elif case == "env_off":
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    elif case == "env_force":
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    elif case == "dispatch_off":
        monkeypatch.setenv(D.DISPATCH_ENV, "0")
    elif case == "rocm":
        # a gfx arch with no ROCM_AUTO_DITS entry
        monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda *a, **k: types.SimpleNamespace(gcnArchName = "gfx1100"),
        )
    elif case == "cpu":
        target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    elif case == "no_transformer":
        pipe = types.SimpleNamespace(transformer = None)
    assert I.install_auto(pipe, target, speed_active = speed) == 0
    assert not card.seen and not card.strided
    if case != "no_transformer":
        assert _flags(pipe) == [(False, 0)] * 4


def test_rocm_listed_dit_races_the_triton_kernel(card, monkeypatch):
    """A DiT in ROCM_AUTO_DITS races the Triton kernel with the switch unset; unlisted FLUX.3 keeps its processors."""
    import sys

    from core.inference import rocm_bf16

    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(gcnArchName = "gfx1151"),
    )
    monkeypatch.setattr(
        I, "implementation", lambda index, head_dim: "triton" if head_dim in (64, 128) else None
    )
    card.found = {"ok": True, "min_tokens": 0, "impl": "triton"}
    shipped_linux = D.ROCM_AUTO_DITS[("gfx1151", "linux")]
    monkeypatch.setattr(
        D, "ROCM_AUTO_DITS", {("gfx1151", sys.platform): frozenset({"Flux3Transformer2DModel"})}
    )
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe) == [(True, D.MIN_TOKENS)] * 3 + [(False, 0)]
    assert card.seen and all(timed for *_, timed in card.seen)
    card.seen.clear()
    monkeypatch.setattr(D, "ROCM_AUTO_DITS", {("gfx1151", sys.platform): shipped_linux})
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0 and not card.seen
    assert _flags(pipe) == [(False, 0)] * 4


def test_apply_attention_backend_races_only_an_automatic_pick(card, monkeypatch):
    """No request / "auto": the strided auto install runs over a native pick and its name is returned. An explicit
    backend request, or a non-native pick, never races."""
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "_native_cudnn")
    monkeypatch.setattr(D, "install", lambda *a, **k: 0)
    calls = []
    monkeypatch.setattr(I, "install_auto", lambda *a, **k: calls.append(k.get("speed_active")) or 2)
    for request in (None, "", "auto", "AUTO"):
        assert (
            A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, requested = request)
            == I.BACKEND
        )
    assert calls == [True] * 4
    for request in ("native", "sdpa", "cudnn", "flash"):
        assert (
            A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, requested = request)
            == "_native_cudnn"
        )
    assert len(calls) == 4
    assert (
        A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, speed_active = False)
        == I.BACKEND
    )
    assert calls[-1] is False
    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "sage_hub")
    assert A.apply_attention_backend(object(), "sage", target = _TARGET) == "sage_hub"
    assert len(calls) == 5


def test_a_pipeline_load_forgets_the_plans_sink_decisions(card, monkeypatch):
    """apply_attention_backend (a pipeline's load) drops the INT8 launch plans, so its calls decide about attention
    sinks again even where they share a plan's shapes with the last pipeline's; without a target nothing is dropped."""
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "_native_cudnn")
    monkeypatch.setattr(D, "install", lambda *a, **k: 0)
    monkeypatch.setattr(I, "install_auto", lambda *a, **k: 0)
    I._plans()["decided"] = {"sinks": False}
    A.apply_attention_backend(object(), "_native_cudnn")
    assert "decided" in I._plans()
    A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET)
    assert "decided" not in I._plans()


def test_dispatch_route_skips_the_strided_processors_of_a_listed_family(card, monkeypatch):
    """FLUX.3 is listed for the unset switch, but its strided processors are the strided install's: the dispatch
    route takes none of them (it would only take stock processors the strided install could not convert)."""
    pipe = _pipe()
    monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])
    assert D.routable_modules(pipe.transformer) == []
    assert D.install(pipe, _TARGET) == 0


def test_the_race_is_logged_and_counted(card):
    """With Studio's logger (it always passes one) the install reports each verdict and the routed count."""
    import logging

    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("test_int8_auto")
    log.setLevel(logging.INFO)
    log.addHandler(Grab())
    card.found = {"ok": True, "min_tokens": 2560, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET, logger = log) == 3
    assert any("2048 race min_tokens=" in r for r in records), records
    assert any("on 3 attention modules" in r for r in records), records
    card.found = {"ok": True, "min_tokens": 0, "impl": "cuda", "long": {"ok": True}}
    forced = _pipe()
    assert (
        I.install_int8_attention(
            forced.transformer, device = "cuda", dtype = torch.bfloat16, force = True, logger = log
        )
        == 3
    )
    assert any("long_ok=True" in r and "(forced)" in r for r in records), records


def test_a_triton_stand_in_keeps_the_incumbent_for_cuda_only_families(card, monkeypatch):
    """FLUX.3 and Z-Image were measured on the CUDA kernel (its V range past attention sinks): where only the Triton
    stand-in serves the card, the unset switch keeps their incumbent; a request by name still routes."""
    monkeypatch.setattr(
        I, "implementation", lambda index, head_dim: "triton" if head_dim in (64, 128) else None
    )
    card.found = {"ok": True, "min_tokens": 0, "impl": "triton"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0
    assert _flags(pipe) == [(False, 0)] * 4 and not card.seen
    assert (
        I.install_int8_attention(pipe.transformer, device = "cuda", dtype = torch.bfloat16, force = True)
        > 0
    )
    assert D.AUTO_CUDA_ONLY <= D.AUTO_DITS


def test_dispatch_route_skips_a_triton_stand_in_for_cuda_only_families(monkeypatch):
    """The dispatch route applies the same rule per module of a CUDA-only DiT."""
    from core.inference import rocm_bf16

    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "triton")
    monkeypatch.setattr(D, "_secondary_cards", lambda *a, **k: ())
    monkeypatch.setattr(D, "install_dispatch_route", lambda: 1)
    monkeypatch.setattr(D, "original_dispatch", lambda: (lambda *a, **k: None))
    floors = []
    monkeypatch.setattr(
        D, "_card_floor", lambda *a, **k: floors.append(a) or ({"ok": True, "impl": "triton"}, 1024)
    )

    class Proc:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, *a, **k):
            return dispatch_attention_fn  # noqa: F821 - only its name is read

    def dit(name):
        d = type(name, (torch.nn.Module,), {})()
        m = torch.nn.Module()
        m.heads = 2
        m.head_dim = 64
        m.processor = Proc()
        d.attn = m
        return d

    for name, routed in (("ZImageTransformer2DModel", 0), ("WanTransformer3DModel", 1)):
        pipe = types.SimpleNamespace(transformer = dit(name))
        monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])
        floors.clear()
        assert D.install(pipe, _TARGET) == routed, name
        assert bool(floors) == bool(routed)


class _Stock:
    """A stock processor (no ``_unsloth_int8`` on its class): what the module ran before the strided swap."""

    _parallel_config = None
    _attention_backend = "_native_cudnn"


def _swapping_pipe(card, monkeypatch):
    """A pipe whose modules hold stock processors, and the strided swap ``ensure_strided_processors`` makes."""
    from core.inference import diffusion_flux3_attn

    pipe = _pipe()
    stocks = []
    for m in pipe.transformer.blocks:
        m.processor = _Stock()
        stocks.append(m.processor)

    def ensure(
        pipe,
        target,
        logger = None,
    ):
        for m in pipe.transformer.blocks:
            if type(m.processor) is _Stock:
                new = _Processor()
                new._unsloth_stock = m.processor
                m.processor = new
        return len(pipe.transformer.blocks)

    monkeypatch.setattr(diffusion_flux3_attn, "ensure_strided_processors", ensure)
    return pipe, stocks


@pytest.mark.parametrize(
    "found",
    [
        {"ok": False, "min_tokens": None, "impl": "cuda"},
        {"ok": True, "min_tokens": None, "impl": "cuda"},
    ],
)
def test_a_lost_race_gives_every_module_back_its_own_processor(card, monkeypatch, found):
    """Accuracy check failed, or torch won every race: every module carries the very processor it had before (so
    FLUX.3's fused Q/K swap applies to it as on base), and the dispatch route does not race it again."""
    card.found = found
    pipe, stocks = _swapping_pipe(card, monkeypatch)
    assert I.install_auto(pipe, _TARGET) == 0
    assert all(m.processor is s for m, s in zip(pipe.transformer.blocks, stocks))
    monkeypatch.setattr(D, "_dits", lambda p: [pipe.transformer])
    monkeypatch.setattr(D, "routable_modules", lambda dit: list(dit.blocks))
    routed = []
    monkeypatch.setattr(D, "install_dispatch_route", lambda: routed.append(1) or 1)
    assert D.install(pipe, _TARGET) == 0 and not routed


def test_a_won_race_keeps_only_the_engaged_strided_processors(card, monkeypatch):
    """Only head dims with a kernel keep the strided processor; the rest get their own back."""
    pipe, stocks = _swapping_pipe(card, monkeypatch)
    assert I.install_auto(pipe, _TARGET) == 3
    blocks = list(pipe.transformer.blocks)
    assert all(m.processor._unsloth_int8 for m in blocks[:3])
    assert blocks[3].processor is stocks[3]


def test_an_untried_cuda_kernel_that_falls_back_to_triton_keeps_the_incumbent(card, monkeypatch):
    """The card reads "cuda" before its kernel was tried; the check then fails on CUDA (no NVRTC, a failed compile)
    and the verdict that wins ran on Triton. FLUX.3 (CUDA only) does not engage and every module keeps its own
    processor; a forced install still routes."""
    monkeypatch.setattr(
        I, "implementation", lambda index, head_dim: "cuda" if head_dim in (64, 128) else None
    )
    card.found = {"ok": True, "min_tokens": 0, "impl": "triton"}
    pipe, stocks = _swapping_pipe(card, monkeypatch)
    assert I.install_auto(pipe, _TARGET) == 0
    assert card.seen
    assert all(m.processor is s for m, s in zip(pipe.transformer.blocks, stocks))
    forced = _pipe()
    assert (
        I.install_int8_attention(
            forced.transformer, device = "cuda", dtype = torch.bfloat16, force = True
        )
        == 3
    )


def test_dispatch_route_refuses_a_triton_verdict_for_cuda_only_families(monkeypatch):
    """Dispatch route, same case: the card reads "cuda" but the winning verdict ran on Triton. Z-Image is not routed,
    Wan is, and a request by name still routes Z-Image."""
    from core.inference import rocm_bf16

    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(I, "register_route", lambda *a, **k: None)
    monkeypatch.setattr(D, "_secondary_cards", lambda *a, **k: ())
    monkeypatch.setattr(D, "install_dispatch_route", lambda: 1)
    monkeypatch.setattr(D, "original_dispatch", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(D, "_card_floor", lambda *a, **k: ({"ok": True, "impl": "triton"}, 1024))

    class Proc:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, *a, **k):
            return dispatch_attention_fn  # noqa: F821 - only its name is read

    def dit(name):
        d = type(name, (torch.nn.Module,), {})()
        m = torch.nn.Module()
        m.heads = 2
        m.head_dim = 64
        m.processor = Proc()
        d.attn = m
        return d

    monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])
    for name, request, routed in (
        ("ZImageTransformer2DModel", None, 0),
        ("WanTransformer3DModel", None, 1),
        ("ZImageTransformer2DModel", I.BACKEND, 1),
    ):
        pipe = types.SimpleNamespace(transformer = dit(name))
        assert D.install(pipe, _TARGET, requested = request) == routed, (name, request)
        assert (pipe.transformer.attn.processor._attention_backend is None) == (not routed)


def _fake_plan(mode = None, **extra):
    plan = {
        "bytes": 64,
        "out_shape": (1, 4, 2, 8),
        "index": 0,
        "calls": 0,
        "sinks": None,
        "checks": [],
        "decision": ("bucket", mode),  # a real decision key carries the mode
    }
    if mode is not None:
        plan["mode"] = mode
    plan.update(extra)
    return plan


def _recording_kernels(monkeypatch, plan):
    """_cuda_attention on CPU tensors with the launches recorded (their ``sinks`` flag) and no CUDA call."""
    I.forget_plans()
    launched = []

    def find(
        q,
        k,
        v,
        config,
        scale,
        index,
        mode = 0,
    ):
        # a real plan is keyed and built by its mode
        plan["mode"] = mode
        return plan, q, k, v

    monkeypatch.setattr(I, "_find_plan", find)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(I, "_raw_stream", lambda index: 0)
    monkeypatch.setattr(I, "_launch_quantisers", lambda *a: launched.append(("quant", a[-1])))
    monkeypatch.setattr(I, "_launch_attention", lambda *a: launched.append(("attn", a[-1])))
    monkeypatch.setattr(I, "_check_sinks", lambda *a: launched.append(("check", None)))
    return launched


def test_a_route_outside_the_sink_families_runs_mains_kernels(monkeypatch):
    """Every call of a MODE_MAIN route (Wan, LTX, MiniMax, a probe) runs the kernels without the attention-sink path:
    no sink launches, no learning reads, no decision, no race, past SINK_DECIDE_AT as on its first call."""
    plan = _fake_plan()
    launched = _recording_kernels(monkeypatch, plan)
    monkeypatch.setattr(I, "_sink_race", lambda *a: pytest.fail("a main route never races"))
    q = torch.zeros(1, 4, 2, 8)
    for _ in range(I.SINK_DECIDE_AT + 2):
        I._cuda_attention(q, q, q, {}, None, 0)
    assert set(launched) == {("quant", False), ("attn", False)}
    assert plan["calls"] == 0 and plan["sinks"] is None and I._decisions() == {}


def test_a_sink_family_route_learns_and_takes_the_sink_path(monkeypatch):
    """A MODE_RACE route (FLUX.3 Klein, Z-Image, Qwen-Image-2.1) runs the sink path while its plan learns."""
    plan = _fake_plan(mode = I.MODE_RACE)
    launched = _recording_kernels(monkeypatch, plan)
    q = torch.zeros(1, 4, 2, 8)
    I._cuda_attention(q, q, q, {}, None, 0, I.MODE_RACE)
    assert launched == [("quant", True), ("check", None), ("attn", True)] and plan["calls"] == 1


def _dispatch_card(monkeypatch, card_floor = None):
    from core.inference import rocm_bf16

    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(I, "register_route", lambda *a, **k: None)
    monkeypatch.setattr(D, "_secondary_cards", lambda *a, **k: ())
    monkeypatch.setattr(D, "install_dispatch_route", lambda: 1)
    monkeypatch.setattr(D, "original_dispatch", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(
        D, "_card_floor", card_floor or (lambda *a, **k: ({"ok": True, "impl": "cuda"}, 1024))
    )
    monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])

    class Proc:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, *a, **k):
            return dispatch_attention_fn  # noqa: F821 - only its name is read

    def dit(name):
        d = type(name, (torch.nn.Module,), {})()
        m = torch.nn.Module()
        m.heads = 2
        m.head_dim = 64
        m.processor = Proc()
        d.attn = m
        return types.SimpleNamespace(transformer = d)

    return dit


@pytest.mark.parametrize(
    "name, request_, mode",
    [
        ("WanTransformer3DModel", None, 0),
        ("WanTransformer3DModel", "unsloth_int8", 0),
        ("LTX2VideoTransformer3DModel", "unsloth_int8", 0),
        ("ZImageTransformer2DModel", None, 1),
        ("QwenImage21Transformer2DModel", None, 1),
    ],
)
def test_dispatch_routes_carry_the_sink_mode_of_their_family(monkeypatch, name, request_, mode):
    """Only the routes of SINK_DITS run the attention-sink path; every other family's route makes main's call (three
    tensors, no mode) into the kernels as they were."""
    dit = _dispatch_card(monkeypatch)
    pipe = dit(name)
    assert D.install(pipe, _TARGET, requested = request_) == 1
    route = pipe.transformer.attn.processor._attention_backend
    assert route.sinks == mode
    calls = []
    monkeypatch.setattr(I, "int8_attention_call", lambda *a: calls.append(len(a)) or a[0])
    monkeypatch.setattr(D, "_cached_reason", lambda *a: None)
    q = torch.zeros(1, 4, 2, 8)
    D.routed_dispatch_attention_fn(q, q, q, backend = route)
    assert calls == [4 if mode else 3]


def test_the_sink_op_is_registered_with_a_fake():
    """The sink routes trace into a second op (its mode a graph constant); the main op keeps its schema."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    I.int8_attention_op()
    op = torch.ops.unsloth_studio.int8_attention_nhd_sinks
    with FakeTensorMode():
        q = torch.empty(1, 16, 2, 64, dtype = torch.bfloat16)
        assert op(q, q, q, 1).shape == q.shape
    assert "int mode" not in str(torch.ops.unsloth_studio.int8_attention_nhd.default._schema)


class _Seen:
    """A finished learning read of a plan whose heads clipped (an attention sink)."""

    def synchronize(self):
        pass


def _deciding_plan(mode):
    return _fake_plan(
        mode = mode, calls = I.SINK_DECIDE_AT, checks = [(_Seen(), torch.ones(2, dtype = torch.int32))]
    )


@pytest.mark.parametrize("mode, native", [(I.MODE_RACE, True), (I.MODE_KERNELS, False)])
def test_a_forced_sink_route_never_ends_on_torch_attention(monkeypatch, mode, native):
    """The deciding call of a plan with sinks: an automatic route may hand the plan to torch's attention; a forced one
    (by name or the env switch) races only the INT8 paths, so its calls stay on the kernel."""
    import contextlib

    plan = _deciding_plan(mode)
    launched = _recording_kernels(monkeypatch, plan)
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(I, "_drop_routes", lambda plan: None)
    monkeypatch.setattr(I, "_sink_race", lambda *a, forced = False: False if forced else "native")
    sentinel = torch.full((1, 4, 2, 8), 7.0)
    monkeypatch.setattr(I, "native_attention", lambda *a: sentinel)
    q = torch.zeros(1, 4, 2, 8)
    out = I._cuda_attention(q, q, q, {}, None, 0, mode)
    assert (out is sentinel) is native
    assert plan["sinks"] == ("native" if native else False)
    assert launched == ([] if native else [("quant", False), ("attn", False)])


def test_the_forced_pick_follows_the_same_rules_without_torch():
    """Forced: the faster of the INT8 paths by RACE_MARGIN within PLAIN_ERROR_SLACK, whatever torch's time; the
    kernels without the sink path where neither error is finite."""
    pick = I._pick_path
    near, far = {"sinks": 0.020, "plain": 0.0207}, {"sinks": 0.021, "plain": 0.034}
    slow = {"sinks": 1.40, "plain": 1.30}
    assert pick(slow, near, True) is False and pick(slow, far, True) is True
    assert pick({"sinks": 1.20, "plain": 1.30}, near, True) is True
    nan = float("nan")
    assert pick(slow, {"sinks": nan, "plain": nan}, True) is False
    assert pick(slow, {"sinks": nan, "plain": 0.02}, True) is False
    assert pick(dict(slow, incumbent = 0.1), near) == "native"


def test_forced_requests_of_the_sink_families_take_the_kernels_only_mode(monkeypatch):
    """By name: Z-Image's route races the INT8 paths only; Wan's stays on main's kernels."""
    assert I.sink_mode("ZImageTransformer2DModel", True) == I.MODE_KERNELS
    assert I.sink_mode("Flux3Transformer2DModel", False) == I.MODE_RACE
    assert I.sink_mode("WanTransformer3DModel", True) == I.MODE_MAIN
    dit = _dispatch_card(monkeypatch)
    pipe = dit("ZImageTransformer2DModel")
    assert D.install(pipe, _TARGET, requested = "unsloth_int8") == 1
    assert pipe.transformer.attn.processor._attention_backend.sinks == I.MODE_KERNELS


def test_a_bucket_decision_applies_only_to_plans_that_also_had_sinks(monkeypatch):
    """Two prompt lengths in one DECIDE_BUCKET: a plan without sinks decides the kernels without the sink path, yet a
    new plan of the bucket still learns its own calls; one with sinks races once, and the next with sinks takes that
    verdict without a race (a verdict never crosses between sink and sink-free plans)."""
    import contextlib

    I.forget_plans()
    names = ("unsloth_i8stats_q", "unsloth_i8quant_kv", "unsloth_i8resid")
    monkeypatch.setattr(
        I, "_functions", lambda *a, **k: (object(), {n: object() for n in names}, 0)
    )
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    config = {"bn": 128, "wpc": 8, "nq": 1, "stages": 2}

    def plan(tokens):
        x = torch.zeros(1, tokens, 4, 128, dtype = torch.bfloat16)
        return I._plan(x, x, x, config, None, 0, I.MODE_RACE)

    def decide(p, sinks, race):
        p["calls"] = I.SINK_DECIDE_AT
        p["checks"] = [(_Seen(), torch.full((2,), int(sinks), dtype = torch.int32))]
        return I._choose_path(p, race = race)[0]

    assert decide(plan(700), False, lambda: pytest.fail("no sinks: no race")) is False
    sinky = plan(710)
    assert sinky["sinks"] is None  # not the sink-free verdict: it learns its own first calls
    races = []
    assert decide(sinky, True, lambda: races.append(1) or True) is True and races == [1]
    again = plan(720)
    assert again["sinks"] is None
    assert decide(again, True, lambda: pytest.fail("known verdict: no race")) is True
    # a sink-free plan of the bucket after the race still runs the kernels without the sink path
    assert decide(plan(730), False, lambda: pytest.fail("no sinks: no race")) is False


def test_after_a_capture_no_plan_races_or_moves_a_route(monkeypatch):
    """A recorded CUDA graph fixes the sink path: sink plans keep it, sink-free plans run without it
    with the same bits; no route moves under it; a new pipeline races again."""
    import contextlib

    I.forget_plans()
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(
        I, "_drop_routes", lambda plan: pytest.fail("no route change after a capture")
    )
    capturing = [True]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    plan = _fake_plan(mode = I.MODE_RACE)
    assert I._choose_path(plan, race = lambda: pytest.fail("never in a capture")) == (True, None)
    capturing[0] = False
    plan.update(_deciding_plan(I.MODE_RACE))
    assert I._choose_path(plan, race = lambda: pytest.fail("no race after a capture"))[0] is True
    clean = _fake_plan(mode = I.MODE_RACE, calls = I.SINK_DECIDE_AT)
    assert I._choose_path(clean, race = lambda: pytest.fail("no sinks: no race"))[0] is False
    I.forget_plans()
    fresh = _deciding_plan(I.MODE_RACE)
    races = []
    assert I._choose_path(fresh, race = lambda: races.append(1) or True)[0] is True and races == [1]
