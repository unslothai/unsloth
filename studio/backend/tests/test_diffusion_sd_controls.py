# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDXL prompt weighting, CLIP skip, CFG rescale and variation seeds (diffusion_sd_controls / prompt_weighting)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference.diffusion_prompt_weighting import (
    CHUNK_TOKENS,
    MAX_CHUNKS,
    MAX_PROMPT_CHARS,
    apply_emphasis,
    has_prompt_syntax,
    parse_prompt_attention,
    tokenize_chunks,
)
from core.inference.diffusion_sd_controls import (
    SdControls,
    apply_prompt_controls,
    diffusers_clip_skip,
    guidance_rescale_kwargs,
    record_fields,
    slerp,
    status_fields,
    validate,
    variation_latents,
    variation_noise,
    variation_seeds,
    weighting_engaged,
)

SDXL = SimpleNamespace(name = "sdxl", label = "SDXL")
FLUX = SimpleNamespace(name = "flux.1-dev", label = "FLUX.1 dev")


# --- parser: webui modules/prompt_parser.py parse_prompt_attention doctests, plus the escapes and BREAK ---


def _approx(parsed):
    return [[t, round(w, 6)] for t, w in parsed]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("normal text", [["normal text", 1.0]]),
        ("an (important) word", [["an ", 1.0], ["important", 1.1], [" word", 1.0]]),
        ("(unbalanced", [["unbalanced", 1.1]]),
        ("\\(literal\\]", [["(literal]", 1.0]]),
        ("(unnecessary)(parens)", [["unnecessaryparens", 1.1]]),
        (
            "a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).",
            [
                ["a ", 1.0],
                ["house", 1.573],
                [" ", 1.1],
                ["on", 1.0],
                [" a ", 1.1],
                ["hill", 0.55],
                [", sun, ", 1.1],
                ["sky", 1.4641],
                [".", 1.1],
            ],
        ),
        ("[faded]", [["faded", round(1 / 1.1, 6)]]),
        ("(word:1.2)", [["word", 1.2]]),
        ("( word : 0.5 )", [[" word ", 0.5]]),
        ("a cat: big", [["a cat: big", 1.0]]),
        ("stray) and ] close", [["stray) and ] close", 1.0]]),
        ("horror \\(theme\\)", [["horror (theme)", 1.0]]),
        ("back\\\\slash", [["back\\slash", 1.0]]),
        ("a BREAK b", [["a", 1.0], ["BREAK", -1.0], ["b", 1.0]]),
        ("", [["", 1.0]]),
        ("(cat:1.2.3)", [["(cat:1.2.3)", 1.0]]),
        ("a ((cat:1.2.3) dog)", [["a ", 1.0], ["(cat:1.2.3) dog", 1.1]]),
        ("([cat:1.2.3)]", [["(", 1.0], ["cat:1.2.3)", round(1 / 1.1, 6)]]),
    ],
)
def test_parse_matches_webui(text, expected):
    assert _approx(parse_prompt_attention(text)) == _approx(expected)


@pytest.mark.parametrize(
    "text, syntax",
    [
        ("a red fox in the snow", False),
        ("a cat: big, 4k", False),
        ("photo, 85mm, f/1.8", False),
        ("a (fox)", True),
        ("a [fox]", True),
        ("(fox:1.0)", True),
        ("horror \\(theme\\)", True),
        ("a BREAK b", True),
        ("BREAKFAST", False),
        (None, False),
        ("", False),
    ],
)
def test_has_prompt_syntax(text, syntax):
    assert has_prompt_syntax(text) is syntax


# --- tokenization / chunking: webui sd_hijack_clip.tokenize_line ---


class _WordTokenizer:
    """One id per space-separated word; ``,`` is its own token (webui's comma_token)."""

    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, pad = None):
        self.pad_token_id = pad
        self.vocab = {",</w>": 3}

    def get_vocab(self):
        return self.vocab

    def __call__(self, text, **_kw):
        ids = []
        for word in text.replace(",", " , ").split():
            ids.append(3 if word == "," else self.vocab.setdefault(word, len(self.vocab) + 10))
        return {"input_ids": ids}


def test_short_prompt_is_one_padded_chunk():
    chunks = tokenize_chunks("a (b:1.5) c", _WordTokenizer(pad = 2))
    assert len(chunks) == 1
    ids, mults = chunks[0]
    assert len(ids) == len(mults) == CHUNK_TOKENS + 2
    assert ids[0] == 1 and ids[4] == 2 and set(ids[4:]) == {2}
    assert mults[:5] == [1.0, 1.0, 1.5, 1.0, 1.0]


def test_pad_after_first_eos_uses_the_tokenizer_pad_id():
    # SDXL's OpenCLIP tokenizer pads with "!" (id 0), not <eos>: webui's id_pad replacement.
    ids, _ = tokenize_chunks("a b", _WordTokenizer(pad = 0))[0]
    assert ids[:4] == [1, 11, 12, 2]
    assert ids[4:] == [0] * (len(ids) - 4)


def test_long_prompt_chunks_instead_of_truncating():
    words = " ".join(f"w{i}" for i in range(160))
    chunks = tokenize_chunks(words, _WordTokenizer(pad = 2))
    assert len(chunks) == 3
    assert all(len(ids) == CHUNK_TOKENS + 2 for ids, _ in chunks)
    body = [t for ids, _ in chunks for t in ids if t not in (1, 2)]
    assert len(body) == 160


def test_chunk_count_is_capped():
    tok = _WordTokenizer(pad = 2)
    words = " ".join(f"w{i}" for i in range(CHUNK_TOKENS * (MAX_CHUNKS + 5)))
    assert len(tokenize_chunks(words, tok)) == MAX_CHUNKS
    assert len(tokenize_chunks(" BREAK ".join(["a"] * (MAX_CHUNKS * 3)), tok)) == MAX_CHUNKS
    at_cap = " ".join(f"w{i}" for i in range(CHUNK_TOKENS * MAX_CHUNKS))
    assert len(tokenize_chunks(at_cap, tok)) == MAX_CHUNKS


class _CountingTokenizer(_WordTokenizer):
    def __init__(self, pad = None):
        super().__init__(pad)
        self.calls = 0
        self.chars = 0

    def __call__(self, text, **kw):
        self.calls += 1
        self.chars += len(text)
        return super().__call__(text, **kw)


def test_chunking_stops_reading_at_the_cap():
    # A huge API prompt is neither tokenized nor chunked past the cap: work stays bounded, not proportional.
    tok = _CountingTokenizer(pad = 2)
    chunks = tokenize_chunks("a BREAK " * 200_000, tok)
    assert len(chunks) == MAX_CHUNKS
    assert tok.calls <= MAX_CHUNKS + 1
    literal = _CountingTokenizer(pad = 2)
    assert len(tokenize_chunks("w " * 5_000_000, literal, weighted = False)) == MAX_CHUNKS
    assert literal.chars <= MAX_PROMPT_CHARS
    # Words past the 32nd chunk never reach a chunk; the kept chunks are the same as for the capped text.
    words = " ".join(f"w{i}" for i in range(CHUNK_TOKENS * (MAX_CHUNKS + 5)))
    at_cap = " ".join(f"w{i}" for i in range(CHUNK_TOKENS * MAX_CHUNKS))
    assert tokenize_chunks(words, _WordTokenizer(pad = 2)) == tokenize_chunks(
        at_cap, _WordTokenizer(pad = 2)
    )


def test_break_starts_a_new_chunk():
    chunks = tokenize_chunks("a b BREAK c", _WordTokenizer(pad = 2))
    assert len(chunks) == 2
    assert chunks[1][0][1] != 2 and chunks[1][0][2] == 2


def test_comma_backtrack_moves_the_tail_to_the_next_chunk():
    # 70 words, a comma, then 10 more: the chunk fills mid-phrase, so webui moves the words after the comma.
    text = " ".join(f"w{i}" for i in range(70)) + " , " + " ".join(f"x{i}" for i in range(10))
    chunks = tokenize_chunks(text, _WordTokenizer(pad = 2))
    first_ids = chunks[0][0]
    assert 3 in first_ids
    body = [t for t in first_ids[1:] if t != 2]
    assert body[-1] == 3  # the first chunk ends at the comma
    assert len([t for t in chunks[1][0] if t not in (1, 2)]) == 10


def test_literal_mode_reads_brackets_as_text():
    tok = _WordTokenizer(pad = 2)
    ids, mults = tokenize_chunks("(a:1.5)", tok, weighted = False)[0]
    assert set(mults) == {1.0}
    assert "(a:1.5)" in tok.vocab


def test_emphasis_restores_the_mean():
    torch = pytest.importorskip("torch")
    z = torch.arange(1, 13, dtype = torch.float32).reshape(1, 3, 4)
    w = torch.tensor([[1.0, 1.5, 0.5]])
    out = apply_emphasis(z, w)
    assert torch.allclose(out.mean(), z.mean())
    ones = apply_emphasis(z, torch.ones(1, 3))
    assert torch.equal(ones, z)


@pytest.mark.parametrize(
    "family, a1111, expected",
    [
        ("sdxl", None, None),
        ("sdxl", 1, None),
        (
            "sdxl",
            2,
            None,
        ),  # SDXL already reads the penultimate layer (webui-forge minimal_clip_skip=2)
        ("sdxl", 3, 1),
        ("sdxl", 4, 2),
        ("sd15", 1, None),  # SD 1.x layout: A1111 N = diffusers N - 1
        ("sd15", 2, 1),
    ],
)
def test_clip_skip_mapping(family, a1111, expected):
    assert diffusers_clip_skip(family, a1111) == expected


def _fake_pipe(layers = 12):
    enc = SimpleNamespace(config = SimpleNamespace(num_hidden_layers = layers))
    return SimpleNamespace(
        text_encoder = enc,
        tokenizer = _WordTokenizer(pad = 2),
        text_encoder_2 = enc,
        tokenizer_2 = _WordTokenizer(pad = 0),
    )


def test_long_plain_prompts_engage_the_chunked_encoder():
    from core.inference.diffusion_prompt_weighting import exceeds_one_chunk

    pipe = _fake_pipe()
    fits = " ".join(f"w{i}" for i in range(CHUNK_TOKENS))
    longer = fits + " extra"
    assert not exceeds_one_chunk(pipe, fits)
    assert exceeds_one_chunk(pipe, longer)
    assert not weighting_engaged(None, SDXL, pipe, [fits])
    assert weighting_engaged(None, SDXL, pipe, [longer])
    # Literal mode still chunks a long prompt (webui's "None" emphasis), but not a short bracketed one.
    assert weighting_engaged(SdControls(prompt_weighting = False), SDXL, pipe, [longer])
    assert not weighting_engaged(SdControls(prompt_weighting = False), SDXL, pipe, ["(a:2)"])
    # A long negative prompt alone engages it too, since both branches share one length.
    assert weighting_engaged(None, SDXL, pipe, ["a fox", longer])
    assert not weighting_engaged(None, FLUX, pipe, [longer])


_TXT2IMG_PARAMS = {"prompt_embeds": 1, "guidance_rescale": 1, "negative_prompt": 1}


def test_defaults_request_nothing():
    assert not SdControls().requested()
    assert not SdControls(prompt_weighting = False, clip_skip = 1, cfg_rescale = 0.0).requested()
    assert not SdControls(variation_seed = 5, variation_strength = 0.0).requested()


def test_flux_refuses_requested_controls_clearly():
    pipe = SimpleNamespace(transformer = object())
    assert status_fields(FLUX, pipe) is None
    validate(SdControls(), FLUX, pipe, {}, "txt2img")
    validate(SdControls(clip_skip = 1, prompt_weighting = False), FLUX, pipe, {}, "txt2img")
    with pytest.raises(ValueError, match = "CLIP skip, CFG rescale applies to SDXL models only"):
        validate(SdControls(clip_skip = 2, cfg_rescale = 0.5), FLUX, pipe, {}, "txt2img")
    with pytest.raises(ValueError, match = "variation seed applies to SDXL"):
        validate(SdControls(variation_strength = 0.2), FLUX, pipe, {}, "txt2img")
    assert not weighting_engaged(None, FLUX, pipe, ["a (fox:1.5)"])


def test_sdxl_status_and_validation():
    pipe = _fake_pipe(layers = 12)
    pipe.scheduler = SimpleNamespace(config = {"prediction_type": "v_prediction"})
    fields = status_fields(SDXL, pipe)
    assert fields["clip_skip"] == {"min": 1, "max": 12, "default": 1, "same_as_default_through": 2}
    assert fields["cfg_rescale"]["default"] == 0.0
    assert fields["cfg_rescale"]["recommended"] == 0.7
    pipe.scheduler = SimpleNamespace(config = {"prediction_type": "epsilon"})
    assert status_fields(SDXL, pipe)["cfg_rescale"]["recommended"] is None
    validate(SdControls(clip_skip = 12, cfg_rescale = 0.7), SDXL, pipe, _TXT2IMG_PARAMS, "txt2img")
    with pytest.raises(ValueError, match = "exceeds"):
        validate(SdControls(clip_skip = 13), SDXL, pipe, _TXT2IMG_PARAMS, "txt2img")
    with pytest.raises(ValueError, match = "CFG rescale is not available"):
        validate(SdControls(cfg_rescale = 0.7), SDXL, pipe, {"prompt_embeds": 1}, "controlnet")
    with pytest.raises(ValueError, match = "text-to-image"):
        validate(SdControls(variation_strength = 0.2), SDXL, pipe, _TXT2IMG_PARAMS, "img2img")
    # Inpaint's prepare_latents takes the same leading arguments but returns noised image latents.
    pipe.prepare_latents = _LatentPipe().prepare_latents
    validate(SdControls(variation_strength = 0.2), SDXL, pipe, _TXT2IMG_PARAMS, "txt2img")
    with pytest.raises(ValueError, match = "text-to-image"):
        validate(SdControls(variation_strength = 0.2), SDXL, pipe, _TXT2IMG_PARAMS, "inpaint")


def test_record_keeps_a_literal_prompt_choice():
    def pw(controls, used):
        return record_fields(controls, weighting_used = used, variation_seed_list = None)[
            "prompt_weighting"
        ]

    assert pw(SdControls(prompt_weighting = False), False) is False
    assert pw(SdControls(), True) is True
    assert pw(SdControls(), False) is None
    assert pw(None, False) is None


def test_guidance_rescale_passthrough():
    assert guidance_rescale_kwargs(None, _TXT2IMG_PARAMS) == {}
    assert guidance_rescale_kwargs(SdControls(cfg_rescale = 0.0), _TXT2IMG_PARAMS) == {}
    assert guidance_rescale_kwargs(SdControls(cfg_rescale = 0.7), _TXT2IMG_PARAMS) == {
        "guidance_rescale": 0.7
    }
    assert guidance_rescale_kwargs(SdControls(cfg_rescale = 0.7), {}) == {}


def test_plain_prompt_leaves_the_call_untouched():
    pipe = _fake_pipe()
    kwargs = {"prompt": "a red fox", "negative_prompt": "blurry", "num_inference_steps": 4}
    before = dict(kwargs)
    assert apply_prompt_controls(pipe, SDXL, None, kwargs, guidance = 7.0) is False
    assert apply_prompt_controls(pipe, SDXL, SdControls(clip_skip = 2), kwargs, guidance = 7.0) is False
    assert (
        apply_prompt_controls(
            pipe, SDXL, SdControls(prompt_weighting = False), {"prompt": "(a:2)"}, guidance = 7.0
        )
        is False
    )
    assert kwargs == before


# --- variation seeds: webui modules/rng.py slerp / ImageRNG ---


def _webui_slerp(val, low, high):
    import torch

    low_norm = low / torch.norm(low, dim = 1, keepdim = True)
    high_norm = high / torch.norm(high, dim = 1, keepdim = True)
    dot = (low_norm * high_norm).sum(1)
    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high


def test_slerp_matches_webui_and_its_endpoints():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    low = torch.randn(4, 8, 8, generator = g)
    high = torch.randn(4, 8, 8, generator = g)
    for val in (0.0, 0.1, 0.5, 1.0):
        assert torch.equal(slerp(val, low, high), _webui_slerp(val, low, high))
    assert torch.allclose(slerp(0.0, low, high), low)
    assert torch.equal(slerp(1.0, low, high), high)
    # Spherical: the interpolant keeps a unit-variance-like norm, a lerp would shrink it.
    mid = slerp(0.5, low, high)
    assert abs(mid.std().item() - 1.0) < 0.2
    assert torch.equal(slerp(0.25, low, low), low * 0.25 + low * 0.75)


def test_variation_noise_per_image_and_dtype():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(1)
    base = torch.randn(2, 4, 8, 8, generator = g).half()
    var = torch.randn(2, 4, 8, 8, generator = g).half()
    assert variation_noise(base, var, 0.0) is base
    out = variation_noise(base, var, 1.0)
    assert out.dtype == torch.float16 and torch.equal(out, var)
    mixed = variation_noise(base, var, 0.3)
    expected = torch.stack(
        [_webui_slerp(0.3, b.float(), v.float()) for b, v in zip(base, var)]
    ).half()
    assert torch.equal(mixed, expected)


def test_variation_seeds_follow_webui_subseeds():
    assert variation_seeds(100, 3) == [100, 101, 102]


class _LatentPipe:
    """prepare_latents shaped like StableDiffusionXLPipeline's."""

    def __init__(self):
        self.scheduler = SimpleNamespace(init_noise_sigma = 2.0)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents = None,
    ):
        from diffusers.utils.torch_utils import randn_tensor

        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            latents = randn_tensor(shape, generator = generator, device = device, dtype = dtype)
        return latents * self.scheduler.init_noise_sigma


def test_variation_latents_reproducible_and_restored():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pipe = _LatentPipe()

    def draw(
        seed,
        var = None,
        strength = None,
    ):
        with variation_latents(pipe, var, strength):
            gen = torch.Generator().manual_seed(seed)
            out = pipe.prepare_latents(1, 4, 64, 64, torch.float32, torch.device("cpu"), gen)
            return out, gen.initial_seed(), torch.randn(1, generator = gen)

    plain, _, after_plain = draw(11)
    off, _, _ = draw(11, [12], 0.0)
    assert torch.equal(plain, off)
    full, _, after_full = draw(11, [12], 1.0)
    assert torch.equal(full, draw(12)[0])  # strength 1 is exactly the variation seed's noise
    # The request's generator advanced exactly as without a variation (ancestral samplers unchanged).
    assert torch.equal(after_plain, after_full)
    a, _, _ = draw(11, [12], 0.3)
    b, _, _ = draw(11, [12], 0.3)
    assert torch.equal(a, b) and not torch.equal(a, plain)
    assert "prepare_latents" not in pipe.__dict__  # the class method is back


# --- real tiny SDXL encode path vs the pipeline's own encode_prompt ---


@pytest.fixture(scope = "module")
def tiny_sdxl():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny SDXL pipeline not reachable: {exc}")
    pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(path, torch_dtype = torch.float32)
    pipe.set_progress_bar_config(disable = True)
    return pipe


@pytest.mark.parametrize("negative", [None, "blurry, low quality"])
def test_weight_one_reproduces_encode_prompt_bit_for_bit(tiny_sdxl, negative):
    import torch

    from core.inference.diffusion_prompt_weighting import encode_weighted_prompts

    with torch.inference_mode():
        ref = tiny_sdxl.encode_prompt(
            "a red fox in the snow",
            device = "cpu",
            do_classifier_free_guidance = True,
            negative_prompt = negative,
        )
        mine = encode_weighted_prompts(
            tiny_sdxl,
            ["a red (fox:1.0) in the snow"],
            [negative] if negative else None,
            guidance_engaged = True,
        )
    assert torch.equal(ref[0], mine["prompt_embeds"])
    assert torch.equal(ref[1], mine["negative_prompt_embeds"])
    assert torch.equal(ref[2], mine["pooled_prompt_embeds"])
    assert torch.equal(ref[3], mine["negative_pooled_prompt_embeds"])


def test_clip_skip_reads_the_diffusers_layer(tiny_sdxl):
    import torch

    from core.inference.diffusion_prompt_weighting import encode_weighted_prompts
    with torch.inference_mode():
        for skip in (1, 2):
            ref = tiny_sdxl.encode_prompt(
                "a red fox", device = "cpu", do_classifier_free_guidance = False, clip_skip = skip
            )
            mine = encode_weighted_prompts(
                tiny_sdxl, ["a red fox"], None, guidance_engaged = False, clip_skip = skip
            )
            assert torch.equal(ref[0], mine["prompt_embeds"])
            assert torch.equal(ref[2], mine["pooled_prompt_embeds"])


def test_weights_and_long_prompts_change_the_embeddings(tiny_sdxl):
    import torch

    from core.inference.diffusion_prompt_weighting import encode_weighted_prompts

    with torch.inference_mode():
        one = encode_weighted_prompts(tiny_sdxl, ["a red (fox:1.0)"], None, guidance_engaged = True)
        more = encode_weighted_prompts(tiny_sdxl, ["a red (fox:1.5)"], None, guidance_engaged = True)
        long = encode_weighted_prompts(
            tiny_sdxl, ["a (fox:1.2), " + "snowy forest, " * 40], ["blurry"], guidance_engaged = True
        )
    assert not torch.equal(one["prompt_embeds"], more["prompt_embeds"])
    assert torch.equal(one["pooled_prompt_embeds"], more["pooled_prompt_embeds"])
    seq = long["prompt_embeds"].shape[1]
    assert seq % 77 == 0 and seq > 77
    assert long["negative_prompt_embeds"].shape == long["prompt_embeds"].shape
    # Zero negatives for an empty negative prompt (force_zeros_for_empty_prompt), as the pipeline.
    assert not one["negative_prompt_embeds"].any()


def test_request_model_ranges():
    from pydantic import ValidationError

    from models.inference import DiffusionGenerateRequest

    ok = DiffusionGenerateRequest(
        prompt = "x", clip_skip = 2, cfg_rescale = 0.7, variation_seed = 5, variation_strength = 0.15
    )
    assert ok.clip_skip == 2 and ok.prompt_weighting is None
    plain = DiffusionGenerateRequest(prompt = "x")
    assert (
        plain.clip_skip,
        plain.cfg_rescale,
        plain.variation_seed,
        plain.variation_strength,
    ) == (
        None,
        None,
        None,
        None,
    )
    for bad in (
        {"clip_skip": 0},
        {"clip_skip": 13},
        {"cfg_rescale": 1.5},
        {"cfg_rescale": -0.1},
        {"variation_strength": 1.1},
        {"variation_seed": -1},
        {"variation_seed": 2**32},
    ):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "x", **bad)


def test_route_refuses_controls_the_engine_cannot_run():
    from routes.inference import _sd_controls_kwargs

    def native_generate(*, prompt):
        return None

    def diffusers_generate(*, prompt, sd_controls = None):
        return None

    plain = SimpleNamespace(
        prompt_weighting = None,
        clip_skip = None,
        cfg_rescale = None,
        variation_seed = None,
        variation_strength = None,
    )
    assert _sd_controls_kwargs(native_generate, plain) == {}
    with pytest.raises(ValueError, match = "diffusers engine only"):
        _sd_controls_kwargs(native_generate, SimpleNamespace(**{**vars(plain), "clip_skip": 3}))
    out = _sd_controls_kwargs(
        diffusers_generate, SimpleNamespace(**{**vars(plain), "cfg_rescale": 0.7})
    )
    assert out["sd_controls"] == SdControls(cfg_rescale = 0.7)


def test_route_refuses_weighting_off_on_an_engine_that_parses_attention():
    # sd.cpp reads (word:1.5) itself, so prompt_weighting false cannot be honoured there.
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend
    from routes.inference import _sd_controls_kwargs

    class Literal:
        def generate(self, *, prompt):
            return None

    off = SimpleNamespace(
        prompt_weighting = False,
        clip_skip = None,
        cfg_rescale = None,
        variation_seed = None,
        variation_strength = None,
    )
    native = SdCppDiffusionBackend.__new__(SdCppDiffusionBackend)
    with pytest.raises(ValueError, match = "prompt weighting off"):
        _sd_controls_kwargs(native.generate, off)
    # An engine that passes the prompt as plain text already renders it literally.
    assert _sd_controls_kwargs(Literal().generate, off) == {}
    assert (
        _sd_controls_kwargs(
            native.generate, SimpleNamespace(**{**vars(off), "prompt_weighting": None})
        )
        == {}
    )


def test_infotext_names_follow_webui():
    from core.inference.image_gallery import _params_text

    meta = {
        "prompt": "a (fox:1.2)",
        "steps": 25,
        "guidance": 7.0,
        "seed": 11,
        "width": 1024,
        "height": 1024,
        "model": "m",
        "variation_seed": 12,
        "variation_strength": 0.15,
        "clip_skip": 2,
        "cfg_rescale": 0.7,
    }
    text = _params_text(meta)
    assert "Variation seed: 12, Variation seed strength: 0.15" in text
    assert "Clip skip: 2" in text and "CFG rescale: 0.7" in text
    plain = _params_text(
        {
            k: v
            for k, v in meta.items()
            if k not in ("variation_seed", "variation_strength", "clip_skip", "cfg_rescale")
        }
    )
    assert "Variation" not in plain and "Clip skip" not in plain and "CFG rescale" not in plain
    assert "Emphasis" not in text and "Emphasis" not in plain


def test_infotext_marks_literal_prompt_with_emphasis_none():
    # webui reads "Emphasis: None" (opts.emphasis) to parse (fox:1.5) literally on replay.
    from core.inference.image_gallery import _params_text

    meta = {
        "prompt": "a (fox:1.5)",
        "steps": 20,
        "guidance": 7.0,
        "seed": 1,
        "width": 64,
        "height": 64,
    }
    assert ", Emphasis: None" in _params_text({**meta, "prompt_weighting": False})
    assert "Emphasis" not in _params_text({**meta, "prompt_weighting": True})
    assert "Emphasis" not in _params_text({**meta, "prompt_weighting": None})


def test_long_plain_prompt_is_chunked_not_truncated(tiny_sdxl):
    import torch

    kwargs = {"prompt": "a red fox, " + "snowy forest, " * 40, "negative_prompt": "blurry"}
    with torch.inference_mode():
        weighted = apply_prompt_controls(tiny_sdxl, SDXL, None, kwargs, guidance = 7.0)
    assert weighted is False  # no weight syntax: chunked only
    assert "prompt" not in kwargs and kwargs["prompt_embeds"].shape[1] % 77 == 0
    assert kwargs["prompt_embeds"].shape[1] > 77
    assert kwargs["negative_prompt_embeds"].shape == kwargs["prompt_embeds"].shape
    short = {"prompt": "a red fox, snowy forest", "negative_prompt": "blurry"}
    assert apply_prompt_controls(tiny_sdxl, SDXL, None, short, guidance = 7.0) is False
    assert short == {"prompt": "a red fox, snowy forest", "negative_prompt": "blurry"}


# --- SD 1.x layout (one CLIP encoder, final_layer_norm after a skipped layer) ---


@pytest.fixture(scope = "module")
def tiny_sd15(tiny_sdxl):
    """diffusers' SD 1.x ``StableDiffusionPipeline`` around the tiny SDXL's CLIP-L encoder: its own
    ``encode_prompt`` (last_hidden_state, or a skipped layer plus final_layer_norm) is what is compared."""
    diffusers = pytest.importorskip("diffusers")
    return diffusers.StableDiffusionPipeline(
        vae = tiny_sdxl.vae,
        text_encoder = tiny_sdxl.text_encoder,
        tokenizer = tiny_sdxl.tokenizer,
        unet = tiny_sdxl.unet,
        scheduler = tiny_sdxl.scheduler,
        safety_checker = None,
        feature_extractor = None,
        requires_safety_checker = False,
    )


def test_sd15_encoder_matches_encode_prompt(tiny_sd15):
    import torch

    from core.inference.diffusion_prompt_weighting import encode_weighted_prompts
    with torch.inference_mode():
        ref = tiny_sd15.encode_prompt("a red fox", "cpu", 1, True, negative_prompt = "blurry")
        mine = encode_weighted_prompts(
            tiny_sd15, ["a red (fox:1.0)"], ["blurry"], guidance_engaged = True
        )
        assert torch.equal(ref[0], mine["prompt_embeds"])
        assert torch.equal(ref[1], mine["negative_prompt_embeds"])
        assert "pooled_prompt_embeds" not in mine
        # A1111 "Clip skip 2" on SD 1.x = diffusers clip_skip 1: hidden_states[-2] then final_layer_norm.
        skip = diffusers_clip_skip("sd15", 2)
        assert skip == 1
        try:
            ref_skip = tiny_sd15.encode_prompt("a red fox", "cpu", 1, False, clip_skip = skip)[0]
        except AttributeError:
            # transformers 5 dropped ``text_model`` that diffusers' SD 1.x clip_skip reads: rebuild it by hand.
            ids = tiny_sd15.tokenizer(
                "a red fox",
                padding = "max_length",
                max_length = 77,
                truncation = True,
                return_tensors = "pt",
            ).input_ids
            enc = tiny_sd15.text_encoder
            hidden = enc(ids, output_hidden_states = True)[-1][-(skip + 1)]
            ref_skip = getattr(enc, "text_model", enc).final_layer_norm(hidden)
        mine_skip = encode_weighted_prompts(
            tiny_sd15, ["a red fox"], None, guidance_engaged = False, clip_skip = skip
        )
        assert torch.equal(ref_skip, mine_skip["prompt_embeds"])
        assert not torch.equal(ref_skip, ref[0])
