# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 klein 9B KV: its checkpoints route to Flux2KleinKVPipeline, it shares the 9B's encoder tables, and the
reference K/V cache is exact for the model it was trained as (and not for stock klein)."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_families as df

KV_REPO = "black-forest-labs/FLUX.2-klein-9b-kv"


def _klein():
    return df.detect_family(KV_REPO, None)


@pytest.mark.parametrize(
    "ident",
    [
        KV_REPO,
        "black-forest-labs/FLUX.2-klein-9B-KV",
        "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        "Flux-2-Klein-9B-KV-Q4_K_M.gguf",
        "flux-2-klein-9b-kv-fp8.safetensors",
        "wraps/FLUX.2-klein-9B-KV-INT8-ConvRot-ComfyUI",
    ],
)
def test_kv_checkpoints_route_to_the_kv_pipeline(ident):
    fam = _klein()
    assert fam is not None and fam.name == "flux.2-klein"
    assert df.family_pipeline_class(fam, ident) == "Flux2KleinKVPipeline"
    # any one identifier naming it is enough (a GGUF in a repo whose id does not)
    assert (
        df.family_pipeline_class(fam, "someone/klein-quants", None, ident) == "Flux2KleinKVPipeline"
    )


@pytest.mark.parametrize(
    "ident",
    [
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "unsloth/FLUX.2-klein-9B-GGUF",
        "flux-2-klein-9b-Q4_K_M.gguf",
        None,
    ],
)
def test_stock_klein_keeps_the_family_pipeline(ident):
    assert df.family_pipeline_class(_klein(), ident) == "Flux2KleinPipeline"


def test_families_without_variants_return_their_pipeline_class():
    for fam in df._FAMILIES:
        if not getattr(fam, "pipeline_class_variants", ()):
            assert (
                df.family_pipeline_class(fam, KV_REPO, "x-klein-9b-kv.gguf") == fam.pipeline_class
            )


def test_kv_files_keep_the_ungated_9b_base_for_their_companions():
    """Same config, encoder, VAE and scheduler bytes as the 9B, whose mirror is ungated; the KV repo has none."""
    fam = _klein()
    for name in ("Flux-2-Klein-9B-KV-Q4_K_M.gguf", "flux-2-klein-9b-Q4_K_M.gguf"):
        base = df.named_variant_base(fam, name)
        assert base == "black-forest-labs/FLUX.2-klein-9B"
        assert df.mirror_repo(base) == "unsloth/FLUX.2-klein-9B"


def test_kv_is_sized_and_shaped_like_the_9b():
    from core.inference import diffusion_auto_policy as ap

    assert df.small_dit_rule_excludes(KV_REPO)
    assert df._FLUX2_BASE_INNER_DIM[KV_REPO.lower()] == 4096
    assert ap.base_repo_bf16_components_gb(KV_REPO) == ap.base_repo_bf16_components_gb(
        "black-forest-labs/FLUX.2-klein-9B"
    )
    assert df._PIPELINE_MIN_DIFFUSERS["Flux2KleinKVPipeline"] == "0.38.0"


def test_kv_text_encoder_resolves_the_9b_qwen3_8b_files():
    from core.inference import diffusion_te_prequant as tp

    assert tp.te_base_equivalent("black-forest-labs/FLUX.2-klein-9B", KV_REPO)
    view = tp.te_variant_family_view(_klein(), KV_REPO)
    assert view.base_repo == "black-forest-labs/FLUX.2-klein-9B"
    assert all("9B" in row[-1] for row in view.te_prequant_repos)


def test_guidance_is_off_and_never_passed_for_the_kv_pipeline():
    from core.inference import diffusion as d

    class Flux2KleinKVPipeline:  # noqa: N801 - the class NAME is what the backend keys on
        config = types.SimpleNamespace()

        def __call__(
            self,
            image = None,
            prompt = None,
            num_inference_steps = 4,
            generator = None,
        ):
            return None

    class Flux2KleinPipeline:  # noqa: N801
        def __init__(self, distilled):
            self.config = types.SimpleNamespace(is_distilled = distilled)

        def __call__(
            self,
            image = None,
            prompt = None,
            guidance_scale = 4.0,
        ):
            return None

    class _Wrapped:
        def __call__(self, *args, **kwargs):
            return None

    fam = _klein()
    kv = Flux2KleinKVPipeline()
    assert not d._pipe_takes_kwarg(kv, fam.cfg_kwarg)
    assert d._pipe_takes_kwarg(Flux2KleinPipeline(False), fam.cfg_kwarg)
    assert d._pipe_takes_kwarg(_Wrapped(), fam.cfg_kwarg)
    assert d._flux2_klein_distilled(kv)
    state = lambda pipe: types.SimpleNamespace(family = fam, pipe = pipe)  # noqa: E731
    assert d._guidance_supported(state(kv)) is False
    assert d._guidance_supported(state(Flux2KleinPipeline(True))) is False
    assert d._guidance_supported(state(Flux2KleinPipeline(False))) is fam.uses_guidance


def _tiny_flux2():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    if not hasattr(tf, "Flux2KVAttnProcessor"):
        pytest.skip("diffusers without FLUX.2 KV processors")
    torch.manual_seed(0)
    m = (
        tf.Flux2Transformer2DModel(
            patch_size = 1,
            in_channels = 16,
            num_layers = 2,
            num_single_layers = 2,
            attention_head_dim = 16,
            num_attention_heads = 2,
            joint_attention_dim = 24,
            timestep_guidance_channels = 32,
            axes_dims_rope = (4, 4, 4, 4),
        )
        .double()
        .eval()
    )
    for p in m.parameters():
        p.data.normal_(0, 0.2)
    return torch, tf, m


def test_cached_reference_kv_equals_recomputing_it_every_step():
    """The exactness gate: a cached step is the extract (uncached) forward of the same model, bit for bit in fp64."""
    torch, tf, m = _tiny_flux2()
    for block in m.transformer_blocks:
        block.attn.set_processor(tf.Flux2KVAttnProcessor())
    for block in m.single_transformer_blocks:
        block.attn.set_processor(tf.Flux2KVParallelSelfAttnProcessor())
    B, T, R, I = 1, 5, 12, 9
    f64 = torch.float64
    txt = torch.randn(B, T, 24, dtype = f64)
    ref = torch.randn(B, R, 16, dtype = f64)

    def ids(n, t):
        return torch.stack(
            [torch.full((n,), t), torch.arange(n) // 3, torch.arange(n) % 3, torch.zeros(n)], -1
        )[None].to(f64)

    tid = torch.zeros(B, T, 4, dtype = f64)
    tid[..., 3] = torch.arange(T)
    rid, iid = ids(R, 10), ids(I, 0)

    def extract(x, t):
        return m(
            hidden_states = torch.cat([ref, x], 1),
            timestep = torch.tensor([t], dtype = f64),
            encoder_hidden_states = txt,
            txt_ids = tid,
            img_ids = torch.cat([rid, iid], 1),
            return_dict = False,
            kv_cache_mode = "extract",
            num_ref_tokens = R,
        )

    def stock(x, t):
        return m(
            hidden_states = torch.cat([x, ref], 1),
            timestep = torch.tensor([t], dtype = f64),
            encoder_hidden_states = txt,
            txt_ids = tid,
            img_ids = torch.cat([iid, rid], 1),
            return_dict = False,
        )[0][:, :I]

    with torch.no_grad():
        _, cache = extract(torch.randn(B, I, 16, dtype = f64), 1.0)
        for t in (0.75, 0.5, 0.25):
            x = torch.randn(B, I, 16, dtype = f64)
            cached = m(
                hidden_states = x,
                timestep = torch.tensor([t], dtype = f64),
                encoder_hidden_states = txt,
                txt_ids = tid,
                img_ids = iid,
                return_dict = False,
                kv_cache = cache,
                kv_cache_mode = "cached",
            )[0]
            assert torch.equal(cached, extract(x, t)[0])
            # ...and it is a different function from stock klein's bidirectional, per-step-modulated references:
            # caching cannot be a runtime switch on the stock checkpoints.
            assert (cached - stock(x, t)).abs().max().item() > 1e-2
