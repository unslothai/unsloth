# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-card sequence parallel for single denoiser calls (diffusion_sequence_parallel), run on CPU with two threads:
a tiny real diffusers Wan transformer through the CFG router, against one card."""

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
WanTransformer3DModel = getattr(diffusers, "WanTransformer3DModel", None)
if WanTransformer3DModel is None:
    pytest.skip("diffusers has no WanTransformer3DModel", allow_module_level = True)

from core.inference import diffusion_multigpu as mg  # noqa: E402
from core.inference import diffusion_sequence_parallel as sp  # noqa: E402

STEPS = 4
FRAMES, HEIGHT, WIDTH = 3, 6, 10
TOKENS = FRAMES * (HEIGHT // 2) * (WIDTH // 2)


@pytest.fixture(autouse = True)
def _force(monkeypatch):
    # the tiny model has 45 tokens: under MIN_TOKENS, so the default would never split it
    monkeypatch.setenv(sp.SP_ENV, "force")
    for k in list(sp.STATS):
        sp.STATS[k] = None if k == "last_error" else 0


def _model():
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 4,
        attention_head_dim = 12,
        in_channels = 4,
        out_channels = 4,
        text_dim = 16,
        freq_dim = 16,
        ffn_dim = 32,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        rope_max_seq_len = 32,
    ).eval()


def _loop(
    model,
    *,
    per_token = False,
    guided = False,
    steps = STEPS,
):
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, FRAMES, HEIGHT, WIDTH, generator = g)
    emb = torch.randn(1, 7, 16, generator = g)
    neg = torch.randn(1, 7, 16, generator = g)
    for i in range(steps):
        if per_token:  # Wan 2.2 TI2V: one timestep per token, cut with the tokens
            t = torch.linspace(100.0, 999.0, TOKENS)[None] - 50.0 * i
        else:
            t = torch.full((1,), 900.0 - 100.0 * i)
        with torch.no_grad():
            out = model(
                hidden_states = latents, timestep = t, encoder_hidden_states = emb, return_dict = False
            )[0]
            if guided:
                unc = model(
                    hidden_states = latents, timestep = t, encoder_hidden_states = neg, return_dict = False
                )[0]
                out = unc + 3.0 * (out - unc)
        latents = latents - 0.1 * out
    return latents


def _router(model, factory = None):
    built = []

    def make():
        r = (factory or (lambda: mg.build_replica(model, torch.device("cpu"))))()
        built.append(r)
        return r

    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu", replica_factory = make)
    model.forward = router
    return router, built


def test_single_call_steps_split_by_tokens_and_match_one_card_bit_for_bit():
    ref = _loop(_model())
    model = _model()
    router, built = _router(model)
    out = _loop(model)
    assert torch.equal(out, ref)
    # step 1 learns there is one call per step; steps 2-4 run half the tokens on each card
    assert router.stats["sequence_splits"] == STEPS - 1
    assert len(built) == 1
    assert sp.STATS["forwards"] == STEPS - 1
    # 2 layers x (q/k/v swap + output swap) + the gather at proj_out, per rank, per split step
    assert sp.STATS["swaps"] == (STEPS - 1) * 2 * (2 * 2 + 1)
    assert sp.STATS["failures"] == 0


def test_per_token_timesteps_are_cut_with_the_tokens():
    ref = _loop(_model(), per_token = True)
    model = _model()
    router, _ = _router(model)
    out = _loop(model, per_token = True)
    # CPU GEMMs on half the rows of the timestep embedding round differently in the last bit; a misaligned cut would
    # be off by the timestep gap (0.02 per token) instead
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    assert router.stats["sequence_splits"] == STEPS - 1


def test_the_mark_and_the_dispatch_are_put_back_after_each_split():
    from diffusers.models.transformers import transformer_wan

    before = transformer_wan.dispatch_attention_fn
    model = _model()
    router, built = _router(model)
    _loop(model)
    assert transformer_wan.dispatch_attention_fn is before
    for m in (model, built[0]):
        for sub in m.modules():
            proc = getattr(sub, "processor", None)
            if proc is not None:
                assert proc._parallel_config is None
                assert "_parallel_config" not in proc.__dict__
        assert not any(getattr(sub, "_forward_hooks", None) for sub in m.modules())
        assert not any(getattr(sub, "_forward_pre_hooks", None) for sub in m.modules())


def test_a_card_that_repeats_an_attention_reads_the_other_half_again():
    """A fused block checks itself against the stock block on its first call per card: that card runs the same
    self-attention two extra times. The swaps pair by module, so neither card falls out of step."""
    ref = _loop(_model())
    model = _model()

    def factory():
        replica = mg.build_replica(model, torch.device("cpu"))
        attn = replica.blocks[0].attn1
        stock = attn.forward

        def twice(*a, **k):
            stock(*a, **k)
            stock(*a, **k)
            return stock(*a, **k)

        attn.forward = twice
        return replica

    router, _ = _router(model, factory)
    out = _loop(model)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == STEPS - 1
    assert sp.STATS["repeat_reads"] > 0


def test_a_failing_replica_reruns_the_step_on_the_source_and_stops_splitting():
    ref = _loop(_model())
    model = _model()

    def factory():
        replica = mg.build_replica(model, torch.device("cpu"))

        def boom(*a, **k):
            raise RuntimeError("replica card lost")

        replica.blocks[1].forward = boom
        return replica

    router, _ = _router(model, factory)
    out = _loop(model)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0
    assert "replica card lost" in (router.sequence_split_off or "")
    assert sp.STATS["failures"] == 1


def test_guided_steps_keep_the_cfg_replica_and_never_split_by_tokens():
    ref = _loop(_model(), guided = True)
    model = _model()
    router, _ = _router(model)
    out = _loop(model, guided = True)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0
    assert router.stats["replica_calls"] >= STEPS - 1


def test_kill_switch_and_short_calls_stay_whole(monkeypatch):
    monkeypatch.setenv(sp.SP_ENV, "0")
    model = _model()
    router, built = _router(model)
    _loop(model)
    assert router.stats["sequence_splits"] == 0 and not built
    assert sp.SP_ENV in router.stats["sequence_split_skip"]

    monkeypatch.delenv(sp.SP_ENV)
    model = _model()
    router, built = _router(model)
    _loop(model)
    assert router.stats["sequence_splits"] == 0 and not built
    assert f"under {sp.MIN_TOKENS}" in router.stats["sequence_split_skip"]


def test_a_plan_that_differs_from_the_checked_one_is_not_split(monkeypatch):
    plan = dict(WanTransformer3DModel._cp_plan)
    plan.pop("rope")
    monkeypatch.setattr(WanTransformer3DModel, "_cp_plan", plan)
    assert sp.read_plan(WanTransformer3DModel) is None
    model = _model()
    router, built = _router(model)
    _loop(model)
    assert router.stats["sequence_splits"] == 0 and not built
    assert "differs" in router.stats["sequence_split_skip"]


def test_the_checked_plan_matches_this_diffusers():
    assert sp.read_plan(WanTransformer3DModel) is not None


def test_mode_reads_the_switch(monkeypatch):
    for raw, want in (
        ("0", "off"),
        ("off", "off"),
        ("", "auto"),
        ("1", "auto"),
        ("force", "force"),
    ):
        monkeypatch.setenv(sp.SP_ENV, raw)
        assert sp.mode() == want
