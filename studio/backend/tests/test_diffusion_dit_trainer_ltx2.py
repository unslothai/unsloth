# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the LTX-2 (video) family of the flow-matching DiT LoRA trainer.

CPU-only, and deliberately free of a ``diffusers`` import: the two places that need the
pipeline's patchifier are isolated behind ``_ltx2_pack`` / ``_ltx2_unpack`` so the forward
contract can be checked against a fake transformer. What matters here is everything a video
family gets WRONG by default: the LoRA targets leaking into the audio stream, the audio
placeholder being fed at the wrong scale, the timestep being divided by 1000 as the image
families do, the cross-modality attention staying on, and a video base silently routing to
the SDXL trainer. The training loop itself is exercised by the live GPU run."""

from __future__ import annotations

import types

import pytest
import torch

from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_dit_trainer import (
    _LTX2_TARGETS,
    _LTX2_TRAIN_FPS,
    _SPECS,
    _free_text_encoders,
    _ltx2_audio_state,
    _ltx2_audio_token_count,
    _ltx2_collate,
    _ltx2_encode_latent_stats,
    _ltx2_encode_latents,
    _ltx2_forward,
    _select_lora_targets,
)
from core.training.diffusion_train_common import (
    AUTO_FLOW_SHIFT_FAMILIES,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    TRAINABLE_VIDEO_FAMILIES,
    _assert_trusted_base_model,
    _DIT_TRAIN_FAMILIES,
    _family_vram_note,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)

# The real LTX-2 checkpoint's transformer config values the trainer reads (from
# Lightricks/LTX-2 transformer/config.json), so the fakes below are not invented numbers.
LTX2_CONF = dict(
    patch_size = 1,
    patch_size_t = 1,
    audio_in_channels = 128,
    audio_sampling_rate = 16000,
    audio_hop_length = 160,
    audio_scale_factor = 4,
    vae_scale_factors = (8, 32, 32),
)

# Every Linear inside one real LTX-2 transformer block, as reported by named_modules() on
# LTX2VideoTransformer3DModel. Split into the video stream (adaptable) and everything else.
_BLOCK = "transformer_blocks.0."
VIDEO_STREAM_LINEARS = tuple(
    _BLOCK + n
    for n in (
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
    )
)
NON_VIDEO_STREAM_LINEARS = tuple(
    _BLOCK + n
    for n in (
        "audio_attn1.to_q",
        "audio_attn1.to_k",
        "audio_attn1.to_v",
        "audio_attn1.to_out.0",
        "audio_attn2.to_q",
        "audio_attn2.to_k",
        "audio_attn2.to_v",
        "audio_attn2.to_out.0",
        "audio_to_video_attn.to_q",
        "audio_to_video_attn.to_k",
        "audio_to_video_attn.to_v",
        "audio_to_video_attn.to_out.0",
        "video_to_audio_attn.to_q",
        "video_to_audio_attn.to_k",
        "video_to_audio_attn.to_v",
        "video_to_audio_attn.to_out.0",
        "audio_ff.net.0.proj",
        "audio_ff.net.2",
        # Video-stream feed-forward: real, but deliberately NOT a target (Lightricks' own
        # video inpainting/outpainting LoRA configs stop at the attention projections).
        "ff.net.0.proj",
        "ff.net.2",
    )
)


def _fake_config():
    return types.SimpleNamespace(**LTX2_CONF)


class _RecordingTransformer:
    """Stands in for LTX2VideoTransformer3DModel: records the kwargs and returns
    correctly-shaped (video, audio) predictions."""

    def __init__(self):
        self.config = _fake_config()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        hs = kwargs["hidden_states"]
        audio = kwargs["audio_hidden_states"]
        return hs.clone(), audio.clone()


@pytest.fixture
def patched_pack(monkeypatch):
    """Replace the two diffusers-backed helpers with the identity-shaped equivalents, so the
    forward contract is testable without importing LTX2Pipeline."""

    def pack(latents, conf):
        b, c, f, h, w = latents.shape
        return latents.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)

    def unpack(pred, f, h, w, conf):
        b = pred.shape[0]
        return pred.reshape(b, f, h, w, -1).permute(0, 4, 1, 2, 3)

    monkeypatch.setattr(dit, "_ltx2_pack", pack)
    monkeypatch.setattr(dit, "_ltx2_unpack", unpack)


# ── the spec ─────────────────────────────────────────────────────────────────
def test_ltx2_spec_is_registered_and_bf16_only():
    spec = _SPECS["ltx-2"]
    assert spec.family == "ltx-2"
    # LTX-2's RoPE runs in double precision and the reference stack is bf16 throughout.
    assert spec.force_bf16 is True
    # The 19B transformer index reports 37.76 GB bf16; "auto" sizes the dense modes off this.
    assert 30.0 < spec.dense_bf16_gb < 45.0
    assert spec.lora_targets == _LTX2_TARGETS


def test_every_trainable_video_family_has_a_spec():
    # The video registry has no trainable flag, so this set is the only gate; a name in it
    # without a spec would pass resolve_trainable_family and then die in run_dit_lora_training.
    assert TRAINABLE_VIDEO_FAMILIES <= set(_SPECS)
    assert TRAINABLE_VIDEO_FAMILIES <= _DIT_TRAIN_FAMILIES
    assert get_trainer("ltx-2") is dit.run_dit_lora_training


# ── LoRA targets: the audio-stream leak ──────────────────────────────────────
def test_ltx2_targets_are_fully_qualified():
    # A bare "to_q" would also match audio_attn1 / audio_attn2 / audio_to_video_attn /
    # video_to_audio_attn, which is exactly the mistake Lightricks warn about in their docs.
    for target in _LTX2_TARGETS:
        assert target.startswith(("attn1.", "attn2.")), target
    assert "to_q" not in _LTX2_TARGETS
    assert "to_out.0" not in _LTX2_TARGETS


def _peft_selects(targets, module_name: str) -> bool:
    """PEFT's own rule for a LIST of target_modules, from
    ``peft.tuners.tuners_utils.check_target_module_exists``: a module is adapted when its
    fully-qualified name equals a target or ends with "." + target. Reimplemented here
    because importing peft pulls in transformers, which some environments cannot import;
    ``test_our_peft_rule_matches_peft`` pins it to the real thing wherever peft loads."""
    return any(module_name == t or module_name.endswith("." + t) for t in targets)


def test_our_peft_rule_matches_peft():
    try:
        from peft import LoraConfig
        from peft.tuners import tuners_utils as peft_utils
    except Exception as exc:  # noqa: BLE001 -- peft drags in transformers; skip where it cannot load
        pytest.skip(f"peft unavailable: {exc}")

    cfg = LoraConfig(target_modules = list(_LTX2_TARGETS))
    for name in VIDEO_STREAM_LINEARS + NON_VIDEO_STREAM_LINEARS:
        assert peft_utils.check_target_module_exists(cfg, name) == _peft_selects(
            _LTX2_TARGETS, name
        ), name


@pytest.mark.parametrize("name", VIDEO_STREAM_LINEARS)
def test_ltx2_targets_select_every_video_attention_projection(name):
    assert _peft_selects(_LTX2_TARGETS, name), name


@pytest.mark.parametrize("name", NON_VIDEO_STREAM_LINEARS)
def test_ltx2_targets_never_select_the_audio_or_cross_modality_streams(name):
    assert not _peft_selects(_LTX2_TARGETS, name), name


@pytest.mark.parametrize("name", NON_VIDEO_STREAM_LINEARS[:16])
def test_the_generic_default_targets_would_leak_into_the_audio_stream(name):
    """Why _LTX2_TARGETS is fully qualified: the shared DEFAULT_LORA_TARGETS suffixes match
    the audio and cross-modality attentions too. If this ever stops being true the
    qualification is no longer load-bearing and the comment above it is wrong."""
    assert _peft_selects(DEFAULT_LORA_TARGETS, name), name


def test_generic_default_targets_resolve_to_the_ltx2_set():
    # normalized() fills the generic DEFAULT_LORA_TARGETS, and those bare suffixes WOULD hit
    # the audio stream, so the spec's fully-qualified set must win.
    cfg = DiffusionLoraConfig(
        base_model = "Lightricks/LTX-2", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg.lora_target_modules == DEFAULT_LORA_TARGETS
    assert (
        _select_lora_targets(cfg.lora_target_modules, _SPECS["ltx-2"].lora_targets) == _LTX2_TARGETS
    )


# ── the audio placeholder ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "num_pixel_frames, fps, expected",
    [
        # 16000 / 160 / 4 = 25 audio latent tokens per second.
        (1, 24.0, 1),  # a still: 1/24 s -> round(1.04) -> 1
        (25, 24.0, 26),
        (121, 24.0, 126),  # the family default clip
        (1, 1.0, 25),
    ],
)
def test_audio_token_count_matches_the_pipeline_formula(num_pixel_frames, fps, expected):
    assert _ltx2_audio_token_count(_fake_config(), num_pixel_frames, fps) == expected


def test_audio_token_count_never_returns_zero():
    # round() would give 0 for a very short duration, and the transformer indexes the audio
    # stream unconditionally, so an empty one trips its RoPE.
    conf = types.SimpleNamespace(**{**LTX2_CONF, "audio_sampling_rate": 1})
    assert _ltx2_audio_token_count(conf, 1, 24.0) == 1


def test_audio_state_is_scaled_by_sigma():
    torch.manual_seed(0)
    sigmas = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1, 1)
    state = _ltx2_audio_state(sigmas, 2, 4, 128, torch.device("cpu"), torch.float32)
    assert state.shape == (2, 4, 128)
    # (1 - sigma) * 0 + sigma * noise: each row's scale must track its own sigma, so the
    # 0.75 row is ~3x the 0.25 row. A version that fed unit noise regardless would be ~1:1.
    ratio = float(state[1].std() / state[0].std())
    assert 2.0 < ratio < 4.5


def test_audio_state_is_zero_at_sigma_zero():
    sigmas = torch.zeros(1).view(1, 1, 1, 1, 1)
    state = _ltx2_audio_state(sigmas, 1, 3, 128, torch.device("cpu"), torch.float32)
    assert torch.equal(state, torch.zeros_like(state))


# ── the forward contract ─────────────────────────────────────────────────────
def _run_forward(
    transformer,
    bsz = 1,
    f = 1,
    h = 4,
    w = 4,
    c = 8,
    sigma = 0.5,
):
    noisy = torch.randn(bsz, c, f, h, w)
    sigmas = torch.full((bsz,), sigma).view(bsz, 1, 1, 1, 1)
    timesteps = torch.full((bsz,), sigma * 1000.0)
    embeds = (
        torch.randn(bsz, 6, 3840),
        torch.randn(bsz, 6, 3840),
        torch.ones(bsz, 6, dtype = torch.int64),
    )
    out = _ltx2_forward(transformer, noisy, timesteps, sigmas, embeds, None, "cpu", torch.float32)
    return noisy, out


def test_forward_returns_the_target_shape(patched_pack):
    tr = _RecordingTransformer()
    noisy, out = _run_forward(tr, bsz = 2, f = 1, h = 4, w = 4, c = 8)
    # target = noise - latents is the 5-D latent, so the prediction must be unpacked back.
    assert out.shape == noisy.shape


def test_forward_isolates_the_cross_modality_attention(patched_pack):
    tr = _RecordingTransformer()
    _run_forward(tr)
    # Without this the placeholder audio stream reaches the video prediction through
    # audio_to_video_attn and the LoRA regresses against noise-contaminated targets.
    assert tr.kwargs["isolate_modalities"] is True


def test_forward_passes_the_unscaled_timestep(patched_pack):
    # LTX-2's config carries timestep_scale_multiplier = 1000 and its pipeline feeds the
    # scheduler timestep through as-is, unlike the FLUX / Qwen families' timestep / 1000.
    tr = _RecordingTransformer()
    _run_forward(tr, sigma = 0.5)
    assert float(tr.kwargs["timestep"][0]) == pytest.approx(500.0)
    # sigma rides the same tensor (what LTX-2.3 uses for prompt cross-attn modulation).
    assert torch.equal(tr.kwargs["sigma"], tr.kwargs["timestep"])


def test_forward_sizes_the_audio_stream_from_the_latent_frames(patched_pack):
    tr = _RecordingTransformer()
    # 1 latent frame -> 1 pixel frame at temporal compression 8 -> 1 audio token.
    _run_forward(tr, f = 1)
    assert tr.kwargs["audio_hidden_states"].shape == (1, 1, 128)
    assert tr.kwargs["audio_num_frames"] == 1
    # 4 latent frames -> (4 - 1) * 8 + 1 = 25 pixel frames -> 26 audio tokens.
    _run_forward(tr, f = 4)
    assert tr.kwargs["audio_hidden_states"].shape == (1, 26, 128)
    assert tr.kwargs["audio_num_frames"] == 26


def test_forward_reports_the_latent_geometry_and_fps(patched_pack):
    tr = _RecordingTransformer()
    _run_forward(tr, f = 1, h = 4, w = 6)
    assert (tr.kwargs["num_frames"], tr.kwargs["height"], tr.kwargs["width"]) == (1, 4, 6)
    # LTX-2's temporal RoPE coordinate is in SECONDS (frame index / fps), so the fps a still
    # is trained at decides where on the temporal axis it lands.
    assert tr.kwargs["fps"] == _LTX2_TRAIN_FPS == 24.0


def test_forward_feeds_the_separate_video_and_audio_text_streams(patched_pack):
    tr = _RecordingTransformer()
    noisy = torch.randn(1, 8, 1, 4, 4)
    sigmas = torch.full((1,), 0.5).view(1, 1, 1, 1, 1)
    video_emb, audio_emb = torch.randn(1, 6, 3840), torch.randn(1, 6, 3840)
    mask = torch.ones(1, 6, dtype = torch.int64)
    _ltx2_forward(
        tr,
        noisy,
        torch.full((1,), 500.0),
        sigmas,
        (video_emb, audio_emb, mask),
        None,
        "cpu",
        torch.float32,
    )
    # The connector emits a DIFFERENT projection per modality; swapping them silently
    # conditions the video stream on the audio caption embedding.
    assert torch.equal(tr.kwargs["encoder_hidden_states"], video_emb)
    assert torch.equal(tr.kwargs["audio_encoder_hidden_states"], audio_emb)
    assert torch.equal(tr.kwargs["encoder_attention_mask"], mask)


# ── latents + collation ──────────────────────────────────────────────────────
class _FakeDist:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def sample(self):
        return self.mean


class _FakeVae:
    """Mimics AutoencoderKLLTX2Video: per-channel latents_mean / latents_std buffers plus a
    scaling_factor, and a 5-D encode."""

    def __init__(
        self,
        channels = 4,
        scaling_factor = 1.0,
    ):
        self.latents_mean = torch.arange(channels, dtype = torch.float32)
        self.latents_std = torch.full((channels,), 2.0)
        self.config = types.SimpleNamespace(scaling_factor = scaling_factor)
        self.seen = None

    def encode(self, px):
        self.seen = px.shape
        b, _c, f, h, w = px.shape
        ch = self.latents_mean.numel()
        mean = torch.ones(b, ch, f, h // 2, w // 2) * 3.0
        std = torch.ones(b, ch, f, h // 2, w // 2) * 5.0
        return types.SimpleNamespace(latent_dist = _FakeDist(mean, std))


def test_encode_latents_adds_a_temporal_axis_and_normalises_per_channel():
    vae = _FakeVae()
    out = _ltx2_encode_latents(vae, torch.zeros(1, 3, 8, 8))
    # A still must reach the video VAE as a 1-frame clip, not as a 4-D image tensor.
    assert vae.seen == (1, 3, 1, 8, 8)
    expected = (3.0 - torch.arange(4, dtype = torch.float32)) / 2.0
    assert torch.allclose(out[0, :, 0, 0, 0], expected)


def test_encode_latent_stats_returns_the_posterior_affine_pair():
    vae = _FakeVae()
    a, b = _ltx2_encode_latent_stats(vae, torch.zeros(1, 3, 8, 8))
    # The cache holds (A, B) so a per-step draw is A + B * randn; B must be the SCALED std,
    # not the raw one, or every cached sample is drawn at the wrong width.
    assert torch.allclose(a[0, :, 0, 0, 0], (3.0 - torch.arange(4, dtype = torch.float32)) / 2.0)
    assert torch.allclose(b[0, :, 0, 0, 0], torch.full((4,), 5.0 / 2.0))


def test_latent_normalisation_honours_the_scaling_factor():
    # scaling_factor is 1.0 on the shipped checkpoint, so a version that dropped it would
    # still pass every other test here.
    plain = _ltx2_encode_latents(_FakeVae(scaling_factor = 1.0), torch.zeros(1, 3, 8, 8))
    scaled = _ltx2_encode_latents(_FakeVae(scaling_factor = 2.0), torch.zeros(1, 3, 8, 8))
    assert torch.allclose(scaled, plain * 2.0)


def test_collate_batches_the_three_connector_tensors():
    entries = [
        (torch.zeros(1, 4, 8), torch.ones(1, 4, 8), torch.ones(1, 4, dtype = torch.int64)),
        (torch.ones(1, 4, 8), torch.zeros(1, 4, 8), torch.zeros(1, 4, dtype = torch.int64)),
    ]
    video, audio, mask = _ltx2_collate(entries, "cpu", torch.float32)
    assert video.shape == audio.shape == (2, 4, 8)
    assert mask.shape == (2, 4)
    # Order matters: entry 0's VIDEO embed is zeros and its AUDIO embed is ones.
    assert float(video[0].sum()) == 0.0 and float(audio[0].sum()) == 32.0
    assert float(video[1].sum()) == 32.0 and float(audio[1].sum()) == 0.0


# ── memory: the LTX-2-only conditioning modules ──────────────────────────────
def test_free_text_encoders_drops_the_ltx2_conditioning_stack():
    pipe = types.SimpleNamespace(
        text_encoder = object(),
        tokenizer = object(),
        # ~2.7 GB of connectors plus the decode-side audio modules the trainer never uses.
        connectors = object(),
        audio_vae = object(),
        vocoder = object(),
        transformer = object(),
        vae = object(),
    )
    _free_text_encoders(pipe)
    assert pipe.text_encoder is None and pipe.tokenizer is None
    assert pipe.connectors is None and pipe.audio_vae is None and pipe.vocoder is None
    # The VAE is freed separately (only once the latent cache is built), and the transformer
    # is the thing being trained.
    assert pipe.vae is not None and pipe.transformer is not None


# ── routing, defaults, validation ────────────────────────────────────────────
@pytest.mark.parametrize(
    "base",
    ["Lightricks/LTX-2", "lightricks/ltx-2", "/data/models/ltx-2", "unsloth/LTX-2-FP8"],
)
def test_ltx2_bases_route_to_the_ltx2_trainer(base):
    assert resolve_trainable_family(base) == "ltx-2"


@pytest.mark.parametrize(
    "base, family",
    [
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b"),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b"),
        ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "hunyuanvideo-1.5"),
    ],
)
def test_video_families_without_a_spec_are_refused_by_name(base, family):
    # Before this gate these fell through to the unknown-name fallback and were handed to
    # the SDXL trainer, which then failed deep inside from_pretrained.
    with pytest.raises(ValueError) as exc:
        resolve_trainable_family(base)
    message = str(exc.value)
    assert family in message
    assert "video" in message.lower()
    # The refusal must name what DOES work, or it is a dead end.
    assert "ltx-2" in message


def test_explicit_video_model_family_override_is_honoured_and_gated():
    # An opaque local path plus an explicit family is the documented way to train from a
    # local checkout, so the override path needs the same video routing as detection.
    assert resolve_trainable_family("/tmp/some-local-checkout", "ltx-2") == "ltx-2"
    with pytest.raises(ValueError, match = "wan2.2-ti2v-5b"):
        resolve_trainable_family("/tmp/some-local-checkout", "wan2.2-ti2v-5b")


def test_unknown_model_family_lists_the_video_families_too():
    with pytest.raises(ValueError) as exc:
        resolve_trainable_family("x", "not-a-real-family")
    assert "ltx-2" in str(exc.value)


def test_ltx2_official_base_passes_the_trusted_base_gate():
    # It is a VIDEO base, so the image-side inference allowlist never covered it.
    _assert_trusted_base_model("Lightricks/LTX-2")
    with pytest.raises(ValueError, match = "untrusted"):
        _assert_trusted_base_model("random-user/ltx-2-finetune")


def test_ltx2_defaults_follow_the_upstream_lora_configs():
    defaults = train_defaults("ltx-2")
    # Lightricks ship rank/alpha 32 at lr 1e-4 in every LTX-2 LoRA config.
    assert defaults["lora_rank"] == 32
    assert defaults["learning_rate"] == pytest.approx(1e-4)
    # The VAE compresses space by 32, so the default resolution must sit on that grid.
    assert defaults["resolution"] % 32 == 0


def test_ltx2_has_a_vram_row():
    note = _family_vram_note("ltx-2")
    assert "19B" in note and "GB" in note


@pytest.mark.parametrize("resolution, ok", [(512, True), (768, True), (520, False), (528, False)])
def test_video_resolution_must_sit_on_the_vae_grid(resolution, ok):
    def build():
        return DiffusionLoraConfig(
            base_model = "Lightricks/LTX-2",
            data_dir = "d",
            output_dir = "o",
            resolution = resolution,
        ).normalized()

    if ok:
        assert build().resolution == resolution
    else:
        with pytest.raises(ValueError, match = "multiple of 32"):
            build()


def test_image_families_keep_the_multiple_of_8_rule():
    # The /32 rule is video-only; an image family at 520px must still be accepted.
    cfg = DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image-Turbo",
        data_dir = "d",
        output_dir = "o",
        resolution = 520,
    ).normalized()
    assert cfg.resolution == 520


def test_ltx2_flow_shift_defaults_to_auto():
    # LTX-2's scheduler sets use_dynamic_shifting, so scheduler.sigmas is the UNSHIFTED
    # uniform table; training on it would draw a sigma distribution inference never uses.
    assert "ltx-2" in AUTO_FLOW_SHIFT_FAMILIES
    cfg = DiffusionLoraConfig(
        base_model = "Lightricks/LTX-2", data_dir = "d", output_dir = "o"
    ).normalized()
    assert cfg.flow_shift == "auto"
    # ...while the identity families are untouched.
    flux = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev", data_dir = "d", output_dir = "o"
    ).normalized()
    assert flux.flow_shift == 1.0


def test_ltx2_rejects_fp16_before_loading():
    with pytest.raises(ValueError, match = "bf16"):
        DiffusionLoraConfig(
            base_model = "Lightricks/LTX-2",
            data_dir = "d",
            output_dir = "o",
            mixed_precision = "fp16",
        ).normalized()
