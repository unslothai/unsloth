# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic multi-GPU policy: which card a load borrows (pure decision table over 2-8 cards) and the batch split of
distilled DiTs (FLUX.1 / FLUX.2) across the pair."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_multigpu as mg  # noqa: E402

T4 = "Tesla T4"
TOTAL = 15360


def card(
    o,
    name = T4,
    used = 400,
    util = 0.0,
    total = TOTAL,
):
    return mg.DeviceProbe(
        ordinal = o, name = name, total_mib = total, free_mib = total - used, util_pct = util
    )


NEED = 6000


@pytest.mark.parametrize(
    "cards, expect, why",
    [
        # two identical idle cards: the other one
        ([card(0), card(1)], 1, "idle card"),
        # four cards: the idle one with the most free memory; the rest stay free for other loads
        ([card(0), card(1, used = 900), card(2, used = 500), card(3, used = 700)], 2, "idle card"),
        # busy by memory (another process holds 2 GiB) or by compute (util above 5%): skipped
        ([card(0), card(1, used = 2048), card(2, util = 40.0), card(3)], 3, "idle card"),
        # a different model (4090 + 3060 style) is never paired automatically
        ([card(0, name = "RTX 4090"), card(1, name = "RTX 3060")], None, "a different card model"),
        # same model but every other card busy
        ([card(0), card(1, used = 3000), card(2, util = 90.0)], None, "busy"),
        # same model, idle, but the replica does not fit
        ([card(0, total = 6144), card(1, total = 6144, used = 400)], None, "short of the"),
        # only the primary
        ([card(0)], None, "no other card"),
        # unknown utilisation counts as idle (memory still decides)
        ([card(0), card(1, util = None)], 1, "idle card"),
    ],
)
def test_automatic_partner_decision_table(cards, expect, why):
    pick, reason = mg.automatic_partner(cards[0], cards, NEED)
    assert pick == expect
    assert why in reason


def test_automatic_partner_prefers_a_peer_link_on_a_tie_then_the_lowest_ordinal():
    cards = [card(0), card(1), card(2), card(3)]
    assert mg.automatic_partner(cards[0], cards, NEED)[0] == 1
    assert mg.automatic_partner(cards[0], cards, NEED, p2p = {3: True})[0] == 3


def test_automatic_partner_eight_cards_lends_exactly_one():
    cards = [card(o, used = 400 + 10 * o) for o in range(8)]
    pick, _ = mg.automatic_partner(cards[3], cards, NEED)
    assert (
        pick == 0
    )  # most free among the other seven; one replica per load whatever the card count


def test_idle_card_requires_low_memory_and_low_utilisation():
    assert mg._idle_card(card(1))
    assert not mg._idle_card(card(1, used = 2000))
    assert not mg._idle_card(card(1, util = mg.AUTO_CFG_PARALLEL_IDLE_UTIL_PCT + 1))
    assert mg._idle_card(card(1, util = mg.AUTO_CFG_PARALLEL_IDLE_UTIL_PCT))


class FluxTransformer2DModel(torch.nn.Module):
    """Named like diffusers' FLUX.1 class: the batch split keys on the class name."""

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(8, 8)
        self.t = torch.nn.Linear(6, 8)
        self.calls = []

    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        pooled_projections = None,
        timestep = None,
        img_ids = None,
        txt_ids = None,
        guidance = None,
        joint_attention_kwargs = None,
        return_dict = True,
        **extra,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        h = self.x(hidden_states) * timestep[:, None, None] * guidance[:, None, None]
        h = (
            h
            + self.t(encoder_hidden_states).mean(1, keepdim = True)
            + pooled_projections[:, None, :8]
        )
        h = h + img_ids.float().mean() + txt_ids.float().mean()
        for value in extra.values():
            if isinstance(value, torch.Tensor):
                h = h + value.sum()
        return (h,) if not return_dict else type("Out", (), {"sample": h})()


class OtherTransformer(FluxTransformer2DModel):
    pass


def _router(model):
    built = []

    def factory():
        r = mg.build_replica(model, torch.device("cpu"))
        built.append(r)
        return r

    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu", replica_factory = factory)
    model.forward = router
    return router, built


def _flux_loop(model, batch, steps, **extra):
    g = torch.Generator().manual_seed(5)
    latents = torch.randn(batch, 16, 8, generator = g)
    emb = torch.randn(batch, 7, 6, generator = g)
    pooled = torch.randn(batch, 12, generator = g)
    img_ids = torch.randn(16, 3, generator = g)  # unbatched: leading size is the token count
    txt_ids = torch.zeros(7, 3)
    for i in range(steps):
        t = torch.full((batch,), 1.0 - i / steps)
        out = model(
            hidden_states = latents,
            encoder_hidden_states = emb,
            pooled_projections = pooled,
            timestep = t,
            img_ids = img_ids,
            txt_ids = txt_ids,
            guidance = torch.full((batch,), 3.5),
            return_dict = False,
            **extra,
        )[0]
        latents = latents - 0.1 * out
    return latents


def _reference(cls, batch, steps):
    torch.manual_seed(0)
    model = cls()
    return model, _flux_loop(model, batch, steps)


def test_distilled_flux_batch_splits_from_the_second_step_and_matches_one_card():
    ref_model, ref = _reference(FluxTransformer2DModel, 4, 5)
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, built = _router(model)
    out = _flux_loop(model, 4, 5)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    # step 1 learns there is one call per step (no guidance branch); steps 2-5 split 4 -> 2 + 2
    assert router.stats["batch_splits"] == 4
    assert len(built) == 1
    assert model.calls[0] == 4
    assert sorted(set(model.calls[1:])) == [2]
    assert (
        built[0].calls.count(2) == 4
    )  # (the copy carries the source's call log from before it was built)


def test_next_render_splits_from_its_first_step():
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, _ = _router(model)
    _flux_loop(model, 2, 3)
    router.end_render()
    before = router.stats["batch_splits"]
    _flux_loop(model, 2, 3)
    assert router.stats["batch_splits"] - before == 3


def test_odd_batch_unknown_class_unknown_input_and_kill_switch_stay_whole(monkeypatch):
    for cls, batch, extra in (
        (FluxTransformer2DModel, 3, {}),
        (OtherTransformer, 4, {}),
        (FluxTransformer2DModel, 4, {"controlnet_block_samples": torch.ones(1)}),
    ):
        torch.manual_seed(0)
        model = cls()
        router, built = _router(model)
        _flux_loop(model, batch, 3, **extra)
        assert router.stats["batch_splits"] == 0, (cls.__name__, batch, list(extra))
        assert not built
    monkeypatch.setenv(mg.BATCH_SPLIT_ENV, "0")
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, built = _router(model)
    _flux_loop(model, 4, 3)
    assert router.stats["batch_splits"] == 0 and not built


def test_guided_render_keeps_cfg_routing_and_does_not_batch_split():
    """True CFG (two calls per step, same latent): the odd branch goes to the replica as before."""
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, _ = _router(model)
    g = torch.Generator().manual_seed(5)
    latents = torch.randn(2, 16, 8, generator = g)
    kw = dict(
        encoder_hidden_states = torch.randn(2, 7, 6, generator = g),
        pooled_projections = torch.randn(2, 12, generator = g),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(7, 3),
        guidance = torch.full((2,), 3.5),
        return_dict = False,
    )
    for i in range(4):
        t = torch.full((2,), 1.0 - i / 4)
        c = model(hidden_states = latents, timestep = t, **kw)[0]
        u = model(hidden_states = latents, timestep = t, **kw)[0]
        latents = latents - 0.1 * (u + 4.0 * (c - u))
    assert router.stats["batch_splits"] == 0
    assert router.stats["replica_calls"] >= 3


def test_replica_failure_during_a_batch_split_recomputes_on_the_source():
    ref_model, ref = _reference(FluxTransformer2DModel, 2, 4)
    torch.manual_seed(0)
    model = FluxTransformer2DModel()

    class Boom(torch.nn.Module):
        def forward(self, *a, **k):
            raise RuntimeError("replica gone")

    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu", replica_factory = Boom)
    model.forward = router
    out = _flux_loop(model, 2, 4)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert router.disabled_reason is not None


def test_batch_split_skips_are_recorded_and_stay_exact():
    """A foreign root hook cannot be mirrored onto a replica, and an unknown input cannot be split: both renders stay
    on the source, match one card, and say why."""
    ref_model, ref = _reference(FluxTransformer2DModel, 2, 3)
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    model.register_forward_pre_hook(lambda mod, args: None)
    router, built = _router(model)
    out = _flux_loop(model, 2, 3)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert router.stats["batch_splits"] == 0 and "carries hooks" in router.stats["batch_split_skip"]
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, _ = _router(model)
    _flux_loop(model, 2, 3, controlnet_block_samples = torch.ones(1))
    assert "unknown inputs" in router.stats["batch_split_skip"]


# --- LoRA weight-only change: mirror the scales onto the replica instead of dropping it ---------------------------


class _Pipe:
    pass


def _lora_model(seed = 0):
    peft = pytest.importorskip("peft")
    torch.manual_seed(seed)
    model = FluxTransformer2DModel()
    cfg = peft.LoraConfig(r = 4, lora_alpha = 4, target_modules = ["x", "t"], init_lora_weights = False)
    peft.inject_adapter_in_model(cfg, model, adapter_name = "style")
    return model


def _set_scale(model, scale):
    from peft.tuners.lora import LoraLayer
    for mod in model.modules():
        if isinstance(mod, LoraLayer):
            mod.set_scale("style", scale)


def _one_call(model):
    g = torch.Generator().manual_seed(9)
    x = torch.randn(2, 16, 8, generator = g)
    return model(
        hidden_states = x,
        encoder_hidden_states = torch.randn(2, 7, 6, generator = g),
        pooled_projections = torch.randn(2, 12, generator = g),
        timestep = torch.ones(2),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(7, 3),
        guidance = torch.ones(2),
        return_dict = False,
    )[0]


def test_weight_only_lora_change_mirrors_scales_onto_the_replica_bitwise():
    model = _lora_model()
    router, built = _router(model)
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    try:
        router._build_replica()
        replica = router.replica
        assert replica is not None
        _set_scale(model, 0.5)
        assert mg.mirror_cfg_replica_adapter_scales(pipe)
        assert router.replica is replica and router.stats["replica_rebuilds"] == 0
        with torch.no_grad():
            src = router.orig_forward(
                hidden_states = torch.ones(2, 16, 8),
                encoder_hidden_states = torch.ones(2, 7, 6),
                pooled_projections = torch.ones(2, 12),
                timestep = torch.ones(2),
                img_ids = torch.zeros(16, 3),
                txt_ids = torch.zeros(7, 3),
                guidance = torch.ones(2),
                return_dict = False,
            )[0]
            rep = replica(
                hidden_states = torch.ones(2, 16, 8),
                encoder_hidden_states = torch.ones(2, 7, 6),
                pooled_projections = torch.ones(2, 12),
                timestep = torch.ones(2),
                img_ids = torch.zeros(16, 3),
                txt_ids = torch.zeros(7, 3),
                guidance = torch.ones(2),
                return_dict = False,
            )[0]
        assert torch.equal(src, rep)
        fresh = _lora_model()
        _set_scale(fresh, 0.5)
        with torch.no_grad():
            assert torch.equal(_one_call(fresh), _one_call(router.replica))
        # a guided render after the change keeps the same replica (no rebuild) and matches a one-card fresh load
        with torch.no_grad():
            out = _cfg_steps(model)
            fresh_out = _cfg_steps(fresh)
        assert router.replica is replica and router.stats["replica_rebuilds"] == 0
        assert router.stats["replica_calls"] >= 2
        assert torch.equal(out, fresh_out)
    finally:
        mg._CFG_ROUTERS.pop(pipe, None)


def test_a_changed_adapter_set_is_not_mirrored():
    peft = pytest.importorskip("peft")
    model = _lora_model()
    router, _ = _router(model)
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    try:
        router._build_replica()
        cfg = peft.LoraConfig(r = 4, lora_alpha = 4, target_modules = ["x"], init_lora_weights = False)
        peft.inject_adapter_in_model(cfg, model, adapter_name = "second")
        assert not mg.mirror_cfg_replica_adapter_scales(pipe)
        assert not mg.mirror_cfg_replica_adapter_scales(_Pipe())  # no router: nothing to mirror
    finally:
        mg._CFG_ROUTERS.pop(pipe, None)


# --- replica failure at render time: CFG batching takes over from the next render --------------------------------


def _cfg_steps(model, steps = 3):
    g = torch.Generator().manual_seed(4)
    latents = torch.randn(1, 16, 8, generator = g)
    kw = dict(
        encoder_hidden_states = torch.randn(1, 7, 6, generator = g),
        pooled_projections = torch.randn(1, 12, generator = g),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(7, 3),
        guidance = torch.ones(1),
        return_dict = False,
    )
    for i in range(steps):
        t = torch.full((1,), 1.0 - i / steps)
        c = model(hidden_states = latents, timestep = t, **kw)[0]
        u = model(hidden_states = latents, timestep = t, **kw)[0]
        latents = latents - 0.1 * (u + 4.0 * (c - u))
    return latents


def test_render_time_replica_failure_hands_over_to_cfg_batching_between_renders():
    torch.manual_seed(0)
    ref = _cfg_steps(FluxTransformer2DModel())
    torch.manual_seed(0)
    model = FluxTransformer2DModel()

    def boom():
        raise RuntimeError("out of memory on cuda:1")

    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu", replica_factory = boom)
    model.forward = router
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    calls = []
    assert mg.set_cfg_parallel_fallback(pipe, lambda p: calls.append("cfg_batch") or None)
    out = _cfg_steps(model)
    torch.testing.assert_close(
        out, ref, rtol = 1e-5, atol = 1e-6
    )  # the failing render finishes on the source
    assert router.disabled_reason.startswith("replica unavailable")
    assert calls == []  # never swapped inside a render
    mg.end_cfg_render(pipe)
    assert calls == ["cfg_batch"]
    assert (
        pipe not in mg._CFG_ROUTERS and "forward" not in model.__dict__
    )  # router gone, cfg_batch can wrap it
    mg.end_cfg_render(pipe)
    assert calls == ["cfg_batch"]  # once


def test_no_fallback_without_a_failure_and_none_registered():
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    router, _ = _router(model)
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    calls = []
    mg.set_cfg_parallel_fallback(pipe, lambda p: calls.append(1))
    _cfg_steps(model)
    mg.end_cfg_render(pipe)
    assert calls == [] and mg._CFG_ROUTERS.get(pipe) is router
    mg._CFG_ROUTERS.pop(pipe, None)
    assert not mg.set_cfg_parallel_fallback(_Pipe(), lambda p: None)


def test_weight_only_lora_path_mirrors_before_it_would_drop_the_replica():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion)
    branch = src.split("if _lora_weights_only_change(")[1].split("# Read before loading")[0]
    assert "mirror_cfg_replica_adapter_scales(pipe, logger)" in branch
    assert branch.index("mirror_cfg_replica_adapter_scales") < branch.index(
        "invalidate_cfg_replica"
    )


# --- layer split only where streaming is exposed ------------------------------------------------------------------


@pytest.mark.parametrize(
    "streamed_mib, params, h2d, tflops, side, expect_hidden",
    [
        # FLUX.1-dev int8 on a T4: about 7 GiB streamed per step at ~11 GiB/s against 12B params of compute at 1024px
        (7 * 1024, 12e9, 11.0, 40.0, 1024, True),
        # the same model at 512px: still hidden (about 3% exposed)
        (7 * 1024, 12e9, 11.0, 40.0, 512, True),
        # a small DiT streaming most of its weights (Z-Image-Turbo at 512px): exposed
        (10 * 1024, 6e9, 11.0, 40.0, 512, False),
        # nothing streamed: hidden
        (0, 9e9, 11.0, 40.0, 1024, True),
    ],
)
def test_stream_exposure_table(streamed_mib, params, h2d, tflops, side, expect_hidden):
    transfer, compute, exposed = mg.stream_exposure(
        streamed_mib, params, h2d_gib_s = h2d, tflops = tflops, side_px = side
    )
    assert (exposed < mg.STREAM_HIDDEN_FRACTION) is expect_hidden
    assert 0.0 <= exposed <= 1.0 and transfer >= 0 and compute > 0


def test_split_gate_applies_only_to_automatic_group_plans(monkeypatch):
    class Plan:
        offload_policy = "group"
        resident_transformer_mib = (
            8 * 1024
        )  # the whole denoiser resident on the primary (only encoders stream)

    class Sel:
        automatic = True
        primary = 0

    class Explicit(Sel):
        automatic = False

    class Pipe:
        transformer = FluxTransformer2DModel().half()

    class Other:
        transformer = torch.nn.Linear(64, 64).half()

    sizes = {"transformer": (8 * 1024, "dit")}
    from core.inference import diffusion_memory as dm

    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: (11.0, 10.0))
    monkeypatch.setattr(mg, "_copy_partner", lambda pipe, sel: 1)
    monkeypatch.setattr(mg, "_gemm_tflops", lambda o, d: 1e-9)
    # a family whose copy cannot serve its requests (no batch split): the split stays
    assert mg._hidden_stream_reason(Other(), Plan(), sizes, Sel()) is None
    monkeypatch.setattr(mg, "_gemm_tflops", lambda o, d: 1e-9)  # compute dominates: hidden
    pipe = Pipe()
    assert "hidden behind compute" in mg._hidden_stream_reason(pipe, Plan(), sizes, Sel())
    monkeypatch.setattr(
        mg, "_copy_partner", lambda pipe, sel: None
    )  # the copy does not fit: the split stays
    assert mg._hidden_stream_reason(pipe, Plan(), sizes, Sel()) is None
    monkeypatch.setattr(mg, "_copy_partner", lambda pipe, sel: 1)
    assert mg._hidden_stream_reason(pipe, Plan(), sizes, Explicit()) is None
    capacity = Plan()
    capacity.offload_policy = "sequential"
    assert mg._hidden_stream_reason(pipe, capacity, sizes, Sel()) is None
    monkeypatch.setenv(mg.STREAM_GATE_ENV, "0")
    assert mg._hidden_stream_reason(pipe, Plan(), sizes, Sel()) is None
    monkeypatch.delenv(mg.STREAM_GATE_ENV)
    streaming = Plan()
    streaming.resident_transformer_mib = (
        1024  # 7 GiB of the denoiser streams on the primary: main's split stays
    )
    assert mg._hidden_stream_reason(pipe, streaming, sizes, Sel()) is None
    monkeypatch.setattr(
        mg, "_gemm_tflops", lambda o, d: 1e9
    )  # compute negligible: exposed, split tried
    assert mg._hidden_stream_reason(pipe, streaming, sizes, Sel()) is None
    assert mg._STREAM_ESTIMATES[pipe]["exposed"] > 0.9


def test_dynamic_nvfp4_layers_keep_the_batch_whole():
    """A layer that takes its activation scale from each call's amax (ComfyUI's dynamic NVFP4) would quantise each half
    against its own maximum: the split is refused and the render matches one card."""

    class ComfyNVFP4DynamicLinear(torch.nn.Linear):
        pass

    ref_model, ref = _reference(FluxTransformer2DModel, 2, 3)
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    model.q = ComfyNVFP4DynamicLinear(2, 2)
    router, built = _router(model)
    out = _flux_loop(model, 2, 3)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert router.stats["batch_splits"] == 0
    assert "per-call activation scaling" in router.stats["batch_split_skip"]


def test_mirroring_lora_scales_resets_the_warm_state():
    """Python-valued scales guard a compiled region, so after a mirror both sides may recompile: the next calls warm
    one at a time again instead of overlapping."""
    model = _lora_model()
    router, _ = _router(model)
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    try:
        router._build_replica()
        router.warm_primary.add(("sig",))
        router.warm_replica.add(("sig",))
        _set_scale(model, 0.5)
        assert mg.mirror_cfg_replica_adapter_scales(pipe)
        assert not router.warm_primary and not router.warm_replica
    finally:
        mg._CFG_ROUTERS.pop(pipe, None)


def test_the_registered_fallback_does_not_keep_the_pipeline_alive():
    """The router is the value of a weak-keyed map on the pipeline: a fallback that captured the pipeline would keep a
    load rolled back before publication (and its weights) alive. It is handed the pipeline when it runs instead."""
    import functools

    model = FluxTransformer2DModel()
    router, _ = _router(model)
    pipe = _Pipe()
    mg._CFG_ROUTERS[pipe] = router
    seen = []
    assert mg.set_cfg_parallel_fallback(
        pipe, functools.partial(lambda p, tag: seen.append((p, tag)), tag = "batch")
    )
    router.fallback_pending = True
    mg.end_cfg_render(pipe)
    assert seen == [(pipe, "batch")]
    # the image and video loads register a partial that leaves the pipeline to the call
    import ast
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1] / "core" / "inference"
    for name in ("diffusion.py", "video.py"):
        tree = ast.parse((backend / name).read_text(encoding = "utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "set_cfg_parallel_fallback"
        ]
        assert len(calls) == 1, name
        held = calls[0].args[1]
        assert isinstance(held, ast.Call) and ast.unparse(held.func) == "functools.partial", name
        assert [ast.unparse(a) for a in held.args] == ["install_cfg_batch"], name
