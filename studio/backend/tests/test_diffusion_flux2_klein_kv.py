# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 klein 9B KV: its checkpoints route to Flux2KleinKVPipeline, it shares the 9B's encoder tables, and the
reference K/V cache is exact for the model it was trained as (and not for stock klein)."""

from __future__ import annotations

import sys
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


# --- the activation guard prices the reference K/V cache --------------------------------------------------------


def _kv_pipe(name = "Flux2KleinKVPipeline", dtype = "bf16"):
    """A pipe object with only what the estimate reads: class name, the 9B transformer config, dtype, VAE scale."""
    torch = pytest.importorskip("torch")
    config = types.SimpleNamespace(
        num_layers = 8, num_single_layers = 24, num_attention_heads = 32, attention_head_dim = 128
    )
    transformer = types.SimpleNamespace(
        config = config,
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "u8": torch.uint8}[dtype],
    )
    return type(name, (), {"transformer": transformer, "vae_scale_factor": 8})()


def _img(w, h):
    return types.SimpleNamespace(size = (w, h))


def test_kv_cache_estimate_is_2_kib_per_reference_pixel_in_bf16_with_the_measured_margin():
    from core.inference import diffusion_memory as dm

    pipe = _kv_pipe()
    one = dm.reference_kv_cache_mib(pipe, [_img(1024, 1024)])
    # 4096 tokens x 32 blocks x (K, V) x 4096 x 2 bytes = 2 GiB, x 1.2 margin
    assert one == round(2048 * dm._REFERENCE_KV_MARGIN)
    assert dm.reference_kv_cache_mib(pipe, [_img(1024, 1024)] * 4) == pytest.approx(4 * one, abs = 2)
    # batch images each carry their own cache
    assert dm.reference_kv_cache_mib(pipe, [_img(1024, 1024)], batch_size = 2) == pytest.approx(
        2 * one, abs = 1
    )
    # the pipeline brings a reference over 1 MP down to 1 MP first
    assert dm.reference_kv_cache_mib(pipe, [_img(2048, 2048)]) == pytest.approx(one, rel = 0.01)
    # smaller references cost less, by area
    assert dm.reference_kv_cache_mib(pipe, [_img(512, 512)]) == pytest.approx(one / 4, abs = 2)
    # fp32 doubles it; quantized (non-float) weights still cache K/V in a 2-byte activation dtype
    assert dm.reference_kv_cache_mib(_kv_pipe(dtype = "fp32"), [_img(1024, 1024)]) == pytest.approx(
        2 * one, abs = 1
    )
    assert dm.reference_kv_cache_mib(_kv_pipe(dtype = "u8"), [_img(1024, 1024)]) == one


def test_kv_cache_estimate_is_zero_off_the_kv_pipeline_or_without_references():
    from core.inference import diffusion_memory as dm

    assert dm.reference_kv_cache_mib(_kv_pipe("Flux2KleinPipeline"), [_img(1024, 1024)]) == 0
    assert dm.reference_kv_cache_mib(_kv_pipe(), []) == 0
    assert dm.reference_kv_cache_mib(_kv_pipe(), [object()]) == 0
    assert dm.reference_kv_cache_mib(object(), [_img(1024, 1024)]) == 0


def _verdict(
    free_gib,
    persistent_mib,
    tile_side = 512,
):
    from core.inference import diffusion_memory as dm
    total = 80 * 1024
    return dm.image_activation_verdict(
        device_memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", int(free_gib * 1024), total),
        width = 1024,
        height = 1024,
        family = "flux.2-klein",
        condition_pixels = 2 * 1024 * 1024,
        vae_tile_side = tile_side,
        vae_sliced = True,
        persistent_mib = persistent_mib,
    )


def test_guard_runs_a_render_whose_kv_cache_fits_and_refuses_one_whose_cache_does_not():
    """Same request, same free memory: the stock path fits, the KV cache on top does not, and tiling the decode cannot
    help (the cache is held through the denoise), so the guard refuses with a reason instead of an OOM mid-render."""
    from core.inference import diffusion_memory as dm

    kv = dm.reference_kv_cache_mib(_kv_pipe(), [_img(1024, 1024)] * 2)
    # the least free memory (in 256 MiB steps) at which the stock request runs
    free_gib = next(
        g / 4.0
        for g in range(4, 4 * 78)
        if _verdict(g / 4.0, 0, tile_side = None).action == dm.ACTIVATION_RUN
    )
    budget_fit = _verdict(free_gib, 0, tile_side = None)
    refused = _verdict(free_gib, kv, tile_side = None)
    assert refused.action == dm.ACTIVATION_REFUSE
    assert refused.needed_mib == budget_fit.needed_mib + kv
    assert "cached keys and values" in refused.message
    # with the cache's room as well, it runs
    roomy = _verdict(free_gib + (kv + 1024) / 1024.0, kv, tile_side = None)
    assert roomy.action == dm.ACTIVATION_RUN


def test_guard_adds_the_cache_to_the_tiled_need_too():
    plain = _verdict(4, 0)
    held = _verdict(4, 3000)
    assert held.needed_mib == plain.needed_mib + 3000
    if plain.tiled_needed_mib is not None:
        assert held.tiled_needed_mib == plain.tiled_needed_mib + 3000


def test_a_kv_gguf_whose_card_tags_the_gated_kv_repo_takes_its_companions_from_the_9b(monkeypatch):
    """QuantStack's KV GGUFs tag black-forest-labs/FLUX.2-klein-9b-kv (gated, no mirror); text encoder, VAE, scheduler
    and config are the 9B's bytes, so the load needs no KV grant. The pipeline class still follows the file."""
    from core.inference import diffusion as d

    assert df.companion_base(KV_REPO) == "black-forest-labs/FLUX.2-klein-9B"
    assert (
        df.companion_base("black-forest-labs/FLUX.2-klein-9B")
        == "black-forest-labs/FLUX.2-klein-9B"
    )
    assert df.companion_base(None) is None
    # trusted: a direct pipeline load of the KV repo, and a card that tags it, are accepted
    assert d._is_trusted_diffusion_repo(KV_REPO)
    monkeypatch.setattr(d, "_hf_base_model", lambda repo_id, token: KV_REPO)
    monkeypatch.setattr(d, "_remember_companion_base", lambda repo_id, base: None)
    fam = _klein()
    gguf = "Flux-2-Klein-9B-KV-Q8_0.gguf"
    base = d._resolve_base_repo("QuantStack/FLUX.2-Klein-9B-KV-GGUF", None, fam, None, gguf)
    assert base == "black-forest-labs/FLUX.2-klein-9B"
    assert df.mirror_repo(base) == "unsloth/FLUX.2-klein-9B"
    assert (
        df.family_pipeline_class(fam, base, "QuantStack/FLUX.2-Klein-9B-KV-GGUF", gguf)
        == "Flux2KleinKVPipeline"
    )


def test_a_kv_file_never_counts_as_the_stock_9b_transformer():
    """KV files borrow the 9B's companions, but their DiT is another model: the dense / INT8 build must never read
    the base's transformer/ for them (Codex P1 on the routing change)."""
    fam = _klein()
    stock_base = "black-forest-labs/FLUX.2-klein-9B"
    kv_ids = ("Flux-2-Klein-9B-KV-Q8_0.gguf", None, "QuantStack/FLUX.2-Klein-9B-KV-GGUF")
    assert df.transformer_variant_differs_from_base(fam, stock_base, *kv_ids)
    # a local base dir names no variant: still not the KV model's DiT
    assert df.transformer_variant_differs_from_base(fam, "/models/klein", *kv_ids)
    # the stock 9B GGUF on its own base, and a KV file on the KV repo itself, are unchanged
    assert not df.transformer_variant_differs_from_base(
        fam, stock_base, "flux-2-klein-9b-Q8_0.gguf", None, "unsloth/FLUX.2-klein-9B-GGUF"
    )
    assert not df.transformer_variant_differs_from_base(fam, KV_REPO, *kv_ids)


def test_preflight_refuses_a_kv_pick_on_a_diffusers_without_the_kv_pipeline(monkeypatch):
    """diffusers 0.37.x ships Flux2KleinPipeline but not Flux2KleinKVPipeline: the family reads as buildable, so the
    KV class must be checked on its own, before anything is evicted (Codex P2)."""
    import sys

    from core.inference import diffusion as d

    old = types.ModuleType("diffusers")
    old.__version__ = "0.37.1"
    old.Flux2KleinPipeline = type("Flux2KleinPipeline", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", old)
    # a diffusers load (a GPU host): the KV class is what it would import
    from core.inference import diffusion_engine_router as router

    monkeypatch.setattr(router, "predict_engine", lambda *a, **k: router.ENGINE_DIFFUSERS)
    be = d.DiffusionBackend()
    with pytest.raises(ValueError, match = r"0\.38\.0"):
        be.validate_load_request(
            "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
            gguf_filename = "Flux-2-Klein-9B-KV-Q8_0.gguf",
            model_kind = "gguf",
        )
    # the stock 9B GGUF still passes the same preflight on that diffusers
    fam = be.validate_load_request(
        "unsloth/FLUX.2-klein-9B-GGUF", gguf_filename = "flux-2-klein-9b-Q8_0.gguf", model_kind = "gguf"
    )
    assert fam.name == "flux.2-klein"


def test_preflight_lets_a_kv_gguf_the_native_engine_runs_load_on_an_old_diffusers(monkeypatch):
    """Python 3.9 hosts pin diffusers 0.36.0 (no Flux2KleinPipeline, no Flux2KleinKVPipeline). A KV GGUF the router
    sends to sd.cpp imports neither, so it loads there as on main (text to image); a diffusers load is still refused."""
    import sys

    from core.inference import diffusion as d
    from core.inference import diffusion_engine_router as router

    old = types.ModuleType("diffusers")
    old.__version__ = "0.36.0"
    monkeypatch.setitem(sys.modules, "diffusers", old)
    monkeypatch.setattr(router, "predict_engine", lambda *a, **k: router.ENGINE_SD_CPP)
    be = d.DiffusionBackend()
    fam = be.validate_load_request(
        "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
        gguf_filename = "Flux-2-Klein-9B-KV-Q8_0.gguf",
        model_kind = "gguf",
    )
    assert fam.name == "flux.2-klein"
    monkeypatch.setattr(router, "predict_engine", lambda *a, **k: router.ENGINE_DIFFUSERS)
    with pytest.raises(ValueError, match = r"0\.38\.0"):
        be.validate_load_request(
            "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
            gguf_filename = "Flux-2-Klein-9B-KV-Q8_0.gguf",
            model_kind = "gguf",
        )

    def _refused(*a, **k):
        raise ValueError("refused pick")

    # an unreadable prediction keeps the diffusers check
    monkeypatch.setattr(router, "predict_engine", _refused)
    with pytest.raises(ValueError, match = r"0\.38\.0"):
        be.validate_load_request(
            "QuantStack/FLUX.2-Klein-9B-KV-GGUF",
            gguf_filename = "Flux-2-Klein-9B-KV-Q8_0.gguf",
            model_kind = "gguf",
        )


def _klein_native_state(repo_id, gguf_filename):
    import core.inference.sd_cpp_backend as bk
    from core.inference.sd_cpp_engine import SdCppModelFiles

    fam = _klein()
    files = SdCppModelFiles(
        diffusion_model = f"/m/{gguf_filename}", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )
    return bk._SdState(
        repo_id = repo_id,
        base_repo = "black-forest-labs/FLUX.2-klein-9B",
        family = fam,
        device = "cpu",
        files = files,
        sampling_method = getattr(fam, "sd_cpp_sampling_method", None),
        mode = "oneshot",
        server = None,
        gguf_filename = gguf_filename,
    )


def _png():
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize(
    "repo_id, gguf",
    [
        ("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q8_0.gguf"),
        ("someone/klein-quants", "flux-2-klein-9b-kv-Q4_K_M.gguf"),
    ],
)
def test_native_engine_refuses_kv_reference_edits_and_keeps_text_to_image(
    monkeypatch, repo_id, gguf
):
    """sd.cpp has no reference K/V cache (leejet/stable-diffusion.cpp#1341): a KV file's reference edit there would run
    the stock bidirectional model, so it is refused by name; text to image is the same math and still renders."""
    from PIL import Image

    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    from .test_sd_cpp_backend import _FakeEngine

    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _klein_native_state(repo_id, gguf)
    assert b.status()["workflows"] == ["txt2img"]
    with pytest.raises(ValueError, match = r"stable-diffusion\.cpp#1341") as info:
        b.generate(
            prompt = "p", steps = 2, workflow = "reference", width = 512, height = 512, init_image = _png()
        )
    assert "Text to image works here" in str(info.value)
    assert "Flux2KleinKVPipeline" in str(info.value)
    calls = []

    def _capture(files, params, output_path, **kw):
        calls.append(params)
        Image.new("RGB", (1, 1)).save(output_path)

    monkeypatch.setattr(engine, "generate", _capture)
    out = b.generate(prompt = "p", steps = 2, width = 512, height = 512)
    assert calls and out["images"]


def test_native_stock_klein_reference_keeps_its_own_message():
    """The stock 9B GGUF takes no KV path: its native reference call keeps the message it had."""
    from core.inference.sd_cpp_backend import (
        SdCppDiffusionBackend,
        native_variant_reference_refusal,
    )

    from .test_sd_cpp_backend import _FakeEngine

    state = _klein_native_state("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf")
    assert native_variant_reference_refusal(state) is None
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = state
    with pytest.raises(ValueError) as info:
        b.generate(
            prompt = "p", steps = 2, workflow = "reference", width = 512, height = 512, init_image = _png()
        )
    assert "1341" not in str(info.value)


def _kv_load_state(repo_id, gguf_filename):
    return types.SimpleNamespace(
        family = _klein(),
        base_repo = "black-forest-labs/FLUX.2-klein-9B",
        repo_id = repo_id,
        gguf_filename = gguf_filename,
        display_repo_id = None,
        pipe = types.SimpleNamespace(),
    )


def test_kv_checkpoint_does_not_advertise_inpaint_or_outpaint():
    """Flux2KleinInpaintPipeline always feeds the encoded source image to the transformer as condition tokens through
    the stock bidirectional per-step path, with no kv_cache_mode: over the KV weights that is another model. diffusers
    ships no KV inpaint pipeline, so the KV checkpoint keeps only what Flux2KleinKVPipeline runs itself."""
    from core.inference import diffusion as d

    kv = _kv_load_state("QuantStack/FLUX.2-Klein-9B-KV-GGUF", "Flux-2-Klein-9B-KV-Q8_0.gguf")
    stock = _kv_load_state("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf")
    assert d._state_workflows(kv) == ["txt2img", "reference"]
    assert d._state_workflows(stock) == d._family_workflows(stock.family)
    assert "inpaint" in d._state_workflows(stock)


@pytest.mark.parametrize("workflow", ["inpaint", "upscale"])
def test_kv_checkpoint_refuses_a_stock_workflow_pipe(monkeypatch, workflow):
    """A masked request reaches _workflow_pipe with the family's inpaint class: for a KV checkpoint it is refused
    before from_pipe builds the stock pipeline over the KV transformer; the stock 9B still gets its workflow pipe."""
    from core.inference import diffusion as d

    built = []
    be = d.DiffusionBackend()
    monkeypatch.setattr(
        be, "_from_pipe_no_recast", lambda pipe, cls: built.append(cls) or types.SimpleNamespace()
    )
    monkeypatch.setattr(d.prompt_cache, "install", lambda *a, **k: None)
    # the stock half only needs the class name to resolve: no real diffusers on the CPU CI legs
    fake = types.ModuleType("diffusers")
    fake.Flux2KleinInpaintPipeline = object
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    kv = _kv_load_state("someone/klein-quants", "flux-2-klein-9b-kv-Q4_K_M.gguf")
    with pytest.raises(ValueError, match = "Flux2KleinKVPipeline") as info:
        be._workflow_pipe(kv, "Flux2KleinInpaintPipeline", workflow)
    assert "Text to image and reference edits work" in str(info.value)
    assert built == []
    stock = _kv_load_state("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf")
    assert be._workflow_pipe(stock, "Flux2KleinInpaintPipeline", workflow) is not None
    assert len(built) == 1


def test_a_generic_file_named_kv_only_by_its_base_repo_keeps_the_kv_identity():
    """A local GGUF with a generic name and ``base_repo`` = the KV repo: the companion remap hands the load the stock
    9B base, so the caller's base_repo is the only KV name left (Codex P2). Pipeline pick, the dense-substitution
    guard and the advertised workflows all keep reading it."""
    from core.inference import diffusion as d

    fam = _klein()
    stock_base = "black-forest-labs/FLUX.2-klein-9B"
    generic = ("model-Q8_0.gguf", None, "/models/klein")
    assert df.family_pipeline_class(fam, stock_base, *generic) == fam.pipeline_class
    assert df.family_pipeline_class(fam, stock_base, KV_REPO, *generic) == "Flux2KleinKVPipeline"
    assert not df.transformer_variant_differs_from_base(fam, stock_base, *generic)
    assert df.transformer_variant_differs_from_base(fam, stock_base, *generic, KV_REPO)
    state = _kv_load_state("/models/klein", "model-Q8_0.gguf")
    assert "inpaint" in d._state_workflows(state)
    state.checkpoint_base = KV_REPO
    assert d._state_workflows(state) == ["txt2img", "reference"]


def test_load_state_carries_the_callers_base_repo():
    from core.inference import diffusion as d
    assert d._LoadState.__dataclass_fields__["checkpoint_base"].default is None


def test_native_engine_refuses_a_generic_file_whose_base_repo_is_the_kv_repo():
    import dataclasses

    from core.inference.sd_cpp_backend import native_variant_reference_refusal

    state = _klein_native_state("/models/klein", "model-Q8_0.gguf")
    assert native_variant_reference_refusal(state) is None
    kv = dataclasses.replace(state, base_repo = KV_REPO)
    assert "Flux2KleinKVPipeline" in (native_variant_reference_refusal(kv) or "")
