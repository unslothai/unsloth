# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 two-card sequence parallel (diffusion_sequence_parallel), run on CPU with two threads: a tiny real
diffusers Flux2 transformer (double- and single-stream blocks, joint text + image sequence) through the CFG router,
against one card; the joint swap's token order; and the refusals (guidance, odd token counts, KV cache, two different
cards)."""

import threading

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
Flux2Transformer2DModel = getattr(diffusers, "Flux2Transformer2DModel", None)
if Flux2Transformer2DModel is None:
    pytest.skip("diffusers has no Flux2Transformer2DModel", allow_module_level = True)

from core.inference import diffusion_multigpu as mg  # noqa: E402
from core.inference import diffusion_sequence_parallel as sp  # noqa: E402

STEPS = 4
# tokens per call: image (a 4 x 6 latent grid) and text (CPU GEMMs of 3 rows round differently from 6: 8 per card)
IMG, TXT = 24, 16
CH, JOINT = 8, 16
LAYERS, SINGLE = 1, 2


@pytest.fixture(autouse = True)
def _force(monkeypatch):
    # 40 tokens: under MIN_TOKENS, so the default would never split it
    monkeypatch.setenv(sp.SP_ENV, "force")
    for k in list(sp.STATS):
        sp.STATS[k] = None if k == "last_error" else 0


def _model():
    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        patch_size = 1,
        in_channels = CH,
        num_layers = LAYERS,
        num_single_layers = SINGLE,
        attention_head_dim = 16,
        num_attention_heads = 4,
        joint_attention_dim = JOINT,
        timestep_guidance_channels = 16,
        mlp_ratio = 2.0,
        axes_dims_rope = (4, 4, 4, 4),
        guidance_embeds = False,
    ).eval()


def _ids(
    img = IMG,
    txt = TXT,
    batched = True,
):
    h, w = 4, -(-img // 4)
    grid = torch.stack(
        torch.meshgrid(torch.arange(h), torch.arange(w), indexing = "ij"), dim = -1
    ).reshape(-1, 2)
    img_ids = torch.zeros(img, 4)
    img_ids[:, 1:3] = grid[:img].float()
    txt_ids = torch.zeros(txt, 4)
    txt_ids[:, 3] = torch.arange(txt).float()
    if batched:
        return img_ids[None], txt_ids[None]
    return img_ids, txt_ids


def _loop(
    model,
    *,
    guided = False,
    steps = STEPS,
    img = IMG,
    txt = TXT,
    batched_ids = True,
    extra = None,
):
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, img, CH, generator = g)
    emb = torch.randn(1, txt, JOINT, generator = g)
    neg = torch.randn(1, txt, JOINT, generator = g)
    img_ids, txt_ids = _ids(img, txt, batched_ids)
    for i in range(steps):
        t = torch.full((1,), 0.9 - 0.1 * i)
        kw = dict(img_ids = img_ids, txt_ids = txt_ids, timestep = t, return_dict = False, **(extra or {}))
        with torch.no_grad():
            out = model(hidden_states = latents, encoder_hidden_states = emb, **kw)[0]
            if guided:
                unc = model(hidden_states = latents, encoder_hidden_states = neg, **kw)[0]
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


def test_the_checked_plan_matches_this_diffusers():
    assert sp.read_plan(Flux2Transformer2DModel) is not None


@pytest.mark.parametrize("batched_ids", [True, False], ids = ["ids_BS4", "ids_S4"])
def test_single_call_steps_split_by_tokens_and_match_one_card_bit_for_bit(batched_ids):
    ref = _loop(_model(), batched_ids = batched_ids)
    model = _model()
    router, built = _router(model)
    out = _loop(model, batched_ids = batched_ids)
    assert torch.equal(out, ref)
    # step 1 learns there is one call per step; steps 2-4 run half the text and half the image tokens on each card
    assert router.stats["sequence_splits"] == STEPS - 1, router.stats
    assert len(built) == 1
    assert sp.STATS["forwards"] == STEPS - 1
    # (double + single blocks) x (q/k/v swap + output swap) + the gather at proj_out, per rank, per split step
    assert sp.STATS["swaps"] == (STEPS - 1) * 2 * ((LAYERS + SINGLE) * 2 + 1)
    assert sp.STATS["failures"] == 0


def test_the_mark_and_the_dispatch_are_put_back_after_each_split():
    from diffusers.models.transformers import transformer_flux2

    before = transformer_flux2.dispatch_attention_fn
    model = _model()
    router, built = _router(model)
    _loop(model)
    assert router.stats["sequence_splits"] == STEPS - 1
    assert transformer_flux2.dispatch_attention_fn is before
    for m in (model, built[0]):
        for sub in m.modules():
            proc = getattr(sub, "processor", None)
            if proc is not None:
                assert "_parallel_config" not in proc.__dict__
        assert not any(getattr(sub, "_forward_hooks", None) for sub in m.modules())
        assert not any(getattr(sub, "_forward_pre_hooks", None) for sub in m.modules())


def _swap_both(q_parts, prefix):
    """Run ``_ulysses`` on two threads, each holding its [prefix | rest] tokens of every head. The attention stand-in
    records the sequence it was given and returns it, so the output must be each card's own input again."""
    store = sp._Store((torch.device("cpu"), torch.device("cpu")))
    seen, outs, errors = {}, {}, []

    def rank(r):
        try:
            ctx = sp._Rank(store, r, prefix)

            def inner(q, k, v, *a, **kw):
                seen[r] = (q.clone(), k.clone(), v.clone())
                return q * 1.0

            q = q_parts[r]
            outs[r] = sp._ulysses(ctx, "blk", inner, (q, q + 1.0, q + 2.0), {})
        except BaseException as exc:  # noqa: BLE001
            store.fail(repr(exc))
            errors.append(exc)

    threads = [threading.Thread(target = rank, args = (r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors, errors
    return seen, outs


def test_the_joint_swap_rebuilds_one_card_order_and_round_trips():
    torch.manual_seed(3)
    heads, d, (t0, t1), (n0, n1) = 4, 3, (3, 3), (5, 5)
    text = torch.randn(1, t0 + t1, heads, d)
    image = torch.randn(1, n0 + n1, heads, d)
    one_card = torch.cat((text, image), dim = 1)  # FLUX.2's sequence: all text, then all image tokens
    parts = (
        torch.cat((text[:, :t0], image[:, :n0]), dim = 1),
        torch.cat((text[:, t0:], image[:, n0:]), dim = 1),
    )
    seen, outs = _swap_both(parts, (t0, t1))
    for r in (0, 1):
        own = slice(r * heads // 2, (r + 1) * heads // 2)
        q, k, v = seen[r]
        assert torch.equal(q, one_card[:, :, own])
        assert torch.equal(k, one_card[:, :, own] + 1.0)
        assert torch.equal(v, one_card[:, :, own] + 2.0)
        # every head of this card's own tokens comes back
        assert torch.equal(outs[r], parts[r])


def test_without_a_prefix_the_swap_keeps_the_wan_order():
    torch.manual_seed(4)
    full = torch.randn(1, 9, 4, 2)
    parts = (full[:, :5].contiguous(), full[:, 5:].contiguous())
    seen, outs = _swap_both(parts, (0, 0))
    assert torch.equal(seen[0][0], full[:, :, :2]) and torch.equal(seen[1][0], full[:, :, 2:])
    assert torch.equal(outs[0], parts[0]) and torch.equal(outs[1], parts[1])


def test_guided_steps_keep_the_cfg_replica_and_never_split_by_tokens():
    ref = _loop(_model(), guided = True)
    model = _model()
    router, _ = _router(model)
    out = _loop(model, guided = True)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0
    assert router.stats["replica_calls"] >= STEPS - 1


@pytest.mark.parametrize(
    "img, txt, what",
    [
        (25, TXT, "odd hidden_states token count (25)"),
        (IMG, 15, "odd encoder_hidden_states token count (15)"),
    ],
)
def test_odd_token_counts_stay_on_one_card(img, txt, what):
    ref = _loop(_model(), img = img, txt = txt)
    model = _model()
    router, built = _router(model)
    out = _loop(model, img = img, txt = txt)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0 and not built
    assert router.stats["sequence_split_skip"] == what


def test_a_reference_kv_cache_call_is_not_split():
    model = _model()
    kw = {
        "hidden_states": torch.zeros(1, IMG, CH),
        "encoder_hidden_states": torch.zeros(1, TXT, JOINT),
        "kv_cache_mode": "extract",
        "num_ref_tokens": 8,
    }
    assert sp.call_refusal(model, kw) == "kv_cache_mode is set"
    assert sp.call_refusal(model, {**kw, "kv_cache_mode": None, "num_ref_tokens": 0}) is None
    assert sp.call_refusal(
        model, {**kw, "kv_cache_mode": None, "num_ref_tokens": 0, "kv_cache": object()}
    ) == ("kv_cache is set")
    assert (
        sp.call_refusal(
            model,
            {
                **kw,
                "kv_cache_mode": None,
                "num_ref_tokens": 0,
                "joint_attention_kwargs": {"scale": 1.0},
            },
        )
        == "joint_attention_kwargs is set"
    )


def test_joint_tokens_count_text_and_image():
    model = _model()
    latent = torch.zeros(1, 4096, CH)
    assert (
        sp.call_tokens(model, latent, {"encoder_hidden_states": torch.zeros(1, 512, JOINT)}) == 4608
    )
    assert sp.call_tokens(model, latent) == 4096


def test_the_default_threshold_takes_a_1024px_image_but_not_512px(monkeypatch):
    monkeypatch.delenv(sp.SP_ENV)
    model = _model()
    with torch.no_grad():
        assert sp.refusal(model, 4096 + 512) is None  # 1024 x 1024 + the padded prompt
        assert sp.refusal(model, 1024 + 512) == f"{1024 + 512} tokens, under {sp.MIN_TOKENS}"


def test_two_different_cards_are_not_paired(monkeypatch):
    monkeypatch.delenv(sp.SP_ENV)
    names = {0: ("NVIDIA GeForce RTX 5090", (12, 0)), 1: ("NVIDIA GeForce RTX 5080", (12, 0))}
    monkeypatch.setattr(sp, "_card_identity", lambda dev: names[torch.device(dev).index])
    why = sp.pair_refusal(torch.device("cuda:0"), torch.device("cuda:1"))
    assert why is not None and "different cards" in why and "RTX 5080" in why
    names[1] = names[0]
    assert sp.pair_refusal(torch.device("cuda:0"), torch.device("cuda:1")) is None
    names[1] = ("NVIDIA GeForce RTX 5080", (12, 0))
    monkeypatch.setenv(sp.SP_ENV, "force")
    assert sp.pair_refusal(torch.device("cuda:0"), torch.device("cuda:1")) is None
    assert (
        sp.pair_refusal(torch.device("cpu"), torch.device("cuda:1"))
        == "the replica is on another device type"
    )


def test_a_mismatched_pair_keeps_the_call_on_the_source(monkeypatch):
    monkeypatch.setattr(sp, "pair_refusal", lambda a, b: "different cards")
    ref = _loop(_model())
    model = _model()
    router, built = _router(model)
    out = _loop(model)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0 and not built
    assert router.stats["sequence_split_skip"] == "different cards"


def test_a_failing_replica_reruns_the_step_on_the_source_and_stops_splitting():
    ref = _loop(_model())
    model = _model()

    def factory():
        replica = mg.build_replica(model, torch.device("cpu"))

        def boom(*a, **k):
            raise RuntimeError("replica card lost")

        replica.single_transformer_blocks[1].forward = boom
        return replica

    router, _ = _router(model, factory)
    out = _loop(model)
    assert torch.equal(out, ref)
    assert router.stats["sequence_splits"] == 0
    assert "replica card lost" in (router.sequence_split_off or "")
