# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 LoRA training, CPU only: routing, the encoder-worker caption pass, the dev DiT base, the dev scheduler,
and a tiny end-to-end run whose adapter the Video page's bake reads back."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_dit_trainer import (
    _LTX2_TARGETS,
    _SPECS,
    _free_text_encoders,
    _training_sigma_table,
    ltx25_conditioning_files,
)
from core.training.diffusion_train_common import (
    AUTO_FLOW_SHIFT_FAMILIES,
    DEFAULT_LORA_FILENAME,
    DiffusionLoraConfig,
    TRAINABLE_VIDEO_FAMILIES,
    _DIT_TRAIN_FAMILIES,
    _FORCE_BF16_FAMILIES,
    _assert_trusted_base_model,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)

FAMILY = "ltx-2.5"
PACK = "Lightricks/LTX-2.5-Diffusers"
# The ungated mirror training fetches from: the distilled DiT only, no transformer_full/.
MIRROR = "unsloth/LTX-2.5-FP8"


@pytest.fixture(autouse = True)
def _fresh_dit_sources(monkeypatch):
    monkeypatch.setattr(dit, "_LTX25_DIT_SOURCES", {})


# Lightricks/LTX-2.5-Diffusers at 4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3 (HfApi().model_info(...).siblings).
PACK_LISTING = (
    ".gitattributes",
    "README.md",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model-00001-of-00002.safetensors",
    "connectors/diffusion_pytorch_model-00002-of-00002.safetensors",
    "connectors/diffusion_pytorch_model.safetensors",
    "connectors/diffusion_pytorch_model.safetensors.index.json",
    "diffusion_decoder/config.json",
    "diffusion_decoder/diffusion_pytorch_model.safetensors",
    "duration_head/config.json",
    "duration_head/diffusion_pytorch_model.safetensors",
    "latent_upsampler/config.json",
    "latent_upsampler/diffusion_pytorch_model.safetensors",
    "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "model_index.json",
    "modular_model_index.json",
    "processor/chat_template.jinja",
    "processor/processor_config.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "prompt_enhancer/config.json",
    "prompt_enhancer/generation_config.json",
    "prompt_enhancer/model-00001-of-00003.safetensors",
    "prompt_enhancer/model-00002-of-00003.safetensors",
    "prompt_enhancer/model-00003-of-00003.safetensors",
    "prompt_enhancer/model.safetensors.index.json",
    "scheduler/scheduler_config.json",
    "temporal_latent_upsampler/config.json",
    "temporal_latent_upsampler/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    *(f"text_encoder/model-0000{i}-of-00005.safetensors" for i in range(1, 6)),
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    *(f"transformer/diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)),
    *(f"transformer/diffusion_pytorch_model-0000{i}-of-00008.safetensors" for i in range(1, 9)),
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer_full/config.json",
    *(
        f"transformer_full/diffusion_pytorch_model-0000{i}-of-00004.safetensors"
        for i in range(1, 5)
    ),
    "transformer_full/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
)

# The pack's scheduler/scheduler_config.json (the distilled DiT's), and Lightricks/LTX-2's.
LTX25_SCHEDULER = {
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
LTX2_SCHEDULER = {**LTX25_SCHEDULER, "shift_terminal": 0.1, "use_dynamic_shifting": True}


def _index(folder: Path, encoder_class: str, *extra: str) -> Path:
    folder.mkdir(parents = True, exist_ok = True)
    spec = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", encoder_class],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
    }
    for name in extra:
        spec[name] = ["ltx2", "LTX2DurationHead"]
    (folder / "model_index.json").write_text(json.dumps(spec), encoding = "utf-8")
    return folder


# ── registration and routing ────────────────────────────────────────────────


def test_ltx25_is_a_trainable_video_family_on_the_dit_trainer():
    assert FAMILY in TRAINABLE_VIDEO_FAMILIES
    assert FAMILY in _DIT_TRAIN_FAMILIES and FAMILY in _FORCE_BF16_FAMILIES
    assert FAMILY in AUTO_FLOW_SHIFT_FAMILIES
    assert get_trainer(FAMILY) is dit.run_dit_lora_training
    spec = _SPECS[FAMILY]
    # Same DiT layout as LTX-2.3: video self / text cross attention only, never the audio stream.
    assert spec.lora_targets == _LTX2_TARGETS
    assert spec.force_bf16 is True
    assert 40.0 <= spec.dense_bf16_gb <= 44.0
    assert spec.training_scheduler is dit._ltx25_training_scheduler


@pytest.mark.parametrize(
    "base", [PACK, "lightricks/ltx-2.5-diffusers", "/data/models/ltx-2.5-diffusers"]
)
def test_ltx25_bases_route_to_the_ltx25_trainer(base):
    assert resolve_trainable_family(base) == FAMILY


def test_ltx2_still_routes_to_ltx2():
    assert resolve_trainable_family("Lightricks/LTX-2") == "ltx-2"
    assert resolve_trainable_family("/tmp/some-local-checkout", "ltx-2") == "ltx-2"


def test_a_local_pack_routes_by_its_model_index(tmp_path):
    v25 = _index(tmp_path / "my_pack", "Gemma4UnifiedForConditionalGeneration", "duration_head")
    v2 = _index(tmp_path / "other_pack", "Gemma3ForConditionalGeneration")
    assert resolve_trainable_family(str(v25)) == FAMILY
    assert resolve_trainable_family(str(v25), FAMILY) == FAMILY
    assert resolve_trainable_family(str(v2)) == "ltx-2"
    # One family never trains the other's weights.
    with pytest.raises(ValueError, match = "loads as the 'ltx-2.5' family, not 'ltx-2'"):
        resolve_trainable_family(str(v25), "ltx-2")
    with pytest.raises(ValueError, match = "loads as the 'ltx-2' family, not 'ltx-2.5'"):
        resolve_trainable_family(str(v2), FAMILY)


def test_the_single_file_release_points_at_the_diffusers_pack():
    with pytest.raises(ValueError, match = "Lightricks/LTX-2.5-Diffusers"):
        resolve_trainable_family("Lightricks/LTX-2.5")


@pytest.mark.parametrize("base", ["unsloth/LTX-2.5-GGUF", "unsloth/LTX-2.5-FP8"])
def test_quantized_inference_repos_are_not_training_bases(base):
    with pytest.raises(ValueError):
        resolve_trainable_family(base)


def test_the_official_pack_passes_the_trusted_base_gate():
    _assert_trusted_base_model(PACK)
    with pytest.raises(ValueError, match = "untrusted"):
        _assert_trusted_base_model("random-user/ltx-2.5-finetune")


def test_ltx25_defaults_sit_on_the_vae_grid():
    defaults = train_defaults(FAMILY)
    assert defaults["lora_rank"] == 32 and defaults["resolution"] % 32 == 0
    cfg = DiffusionLoraConfig(base_model = PACK, data_dir = "d", output_dir = "o", resolution = 520)
    with pytest.raises(ValueError):
        cfg.normalized()


def test_family_train_infos_offers_ltx25(dit_train_host):
    from core.training.diffusion_train_common import family_train_infos

    info = {i["name"]: i for i in family_train_infos()}[FAMILY]
    assert info["default_base"] == PACK and info["base_repos"] == [PACK]
    assert info["label"] == "LTX-2.5"
    assert "nf4" in info["precision_modes"]


# ── conditioning: staged files, the encoder worker, its release ─────────────


def test_conditioning_stages_one_copy_of_the_small_components_only():
    files = ltx25_conditioning_files(PACK_LISTING)
    # The Hub listing arrives as a generator: read once, model_index.json must still be staged.
    assert ltx25_conditioning_files(iter(PACK_LISTING)) == files
    assert "model_index.json" in files
    assert {f.split("/", 1)[0] for f in files if "/" in f} == {
        "audio_vae",
        "connectors",
        "scheduler",
        "vae",
        "vocoder",
    }
    # The connectors' indexed shard set, not also the 6.3 GB unsharded duplicate.
    assert "connectors/diffusion_pytorch_model.safetensors" not in files
    assert "connectors/diffusion_pytorch_model.safetensors.index.json" in files
    # Never the encoder (the worker reads it), the enhancer, either DiT or the root-level LoRA.
    for name in files:
        assert not name.startswith(
            ("text_encoder/", "tokenizer/", "prompt_enhancer/", "transformer")
        )
        assert "/" in name or name == "model_index.json"


class _FakeClient:
    def __init__(self, width: int = 12):
        self.width = width
        self.prompts: list[str] = []
        self.closed = 0
        self.config = {"resident": True, "device": "cpu"}
        self.renders = 0

    def cached(self, *a, **k):
        return False

    def encode(
        self,
        prompts,
        max_sequence_length,
        cancel = None,
        *,
        copy = True,
        compact = None,
    ):
        self.prompts.extend(prompts)
        gen = torch.Generator().manual_seed(len(prompts[0]))
        embeds = torch.randn(len(prompts), 6, self.width, generator = gen)
        mask = torch.ones(len(prompts), 6, dtype = torch.long)
        return embeds, mask

    def close(self):
        self.closed += 1


class _FakeConnectors(torch.nn.Module):
    def forward(
        self,
        embeds,
        mask,
        padding_side = "left",
    ):
        return embeds * 2, embeds * 3, mask


class _FakePipe:
    calls: list = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        pipe = cls()
        pipe.vae = torch.nn.Linear(1, 1)
        pipe.connectors = _FakeConnectors()
        pipe.text_encoder = kwargs.get("text_encoder", "LOADED")
        pipe.tokenizer = kwargs.get("tokenizer", "LOADED")
        pipe.transformer = kwargs.get("transformer", "LOADED")
        pipe._execution_device = torch.device("cpu")
        return pipe


def test_captions_encode_through_the_worker_and_the_worker_stops(monkeypatch, tmp_path):
    client = _FakeClient()
    _FakePipe.calls = []
    monkeypatch.setattr(dit, "_ltx2_pipeline_cls", lambda: _FakePipe)
    monkeypatch.setattr(dit, "_ltx25_start_encoder", lambda cfg, device: client)
    cfg = types.SimpleNamespace(
        base_model = str(_index(tmp_path / "pack", "Gemma4UnifiedForConditionalGeneration")),
        hf_token = None,
    )

    pipe, vae = _SPECS[FAMILY].load_conditioners(cfg, "cpu", torch.float32)
    ((source, kwargs),) = _FakePipe.calls
    assert source == cfg.base_model
    # No in-process encoder or DiT: the encoder needs the transformers 5.10 sidecar, the DiT loads later.
    for name in (
        "transformer",
        "text_encoder",
        "tokenizer",
        "processor",
        "prompt_enhancer",
        "duration_head",
    ):
        assert kwargs[name] is None, name
    assert vae is pipe.vae and vae.weight.dtype == torch.float32
    assert pipe._unsloth_ltx25_encoder is client

    out = _SPECS[FAMILY].encode_prompts(pipe, ["a sloth", ""], "cpu")
    assert client.prompts == ["a sloth", ""]
    video, audio, mask = out[0]
    # The connector output of the worker's hidden states, per caption, on the host.
    assert video.shape == (1, 6, 12) and audio.shape == (1, 6, 12) and mask.shape == (1, 6)
    assert video.device.type == "cpu"

    _free_text_encoders(pipe)
    assert client.closed == 1 and pipe._unsloth_ltx25_encoder is None
    assert pipe.connectors is None and pipe.text_encoder is None


def test_the_worker_encodes_like_the_video_page(monkeypatch):
    import core.inference.video_ltx25 as v25

    seen = {}
    monkeypatch.setattr(v25, "start_text_encoder", lambda **kw: seen.update(kw) or "client")
    monkeypatch.setattr(
        v25, "ltx25_te_prequant_sources", lambda quant, base: {"text_encoder": object()}
    )
    cfg = types.SimpleNamespace(base_model = PACK, hf_token = "tok")
    monkeypatch.setattr(dit, "_current_index", lambda device: 3)
    assert dit._ltx25_start_encoder(cfg, "cuda") == "client"
    # The Video page's default: the hosted int8 ConvRot encoder for the official pack, resident on the training card.
    assert seen["te_quant"] == "int8" and seen["resident"] is True and seen["device"] == "cuda:3"
    assert seen["repo"] == PACK and seen["base_repo"] == PACK and seen["hf_token"] == "tok"

    monkeypatch.setattr(v25, "ltx25_te_prequant_sources", lambda quant, base: {})
    dit._ltx25_start_encoder(cfg, "cpu")
    assert seen["te_quant"] is None and seen["resident"] is False and seen["device"] == "cpu"


# ── the DiT the LoRA trains against, and its sigmas ─────────────────────────


def test_the_dev_dit_is_the_default_base(monkeypatch, tmp_path):
    monkeypatch.delenv(dit.LTX25_TRAIN_DIT_ENV, raising = False)
    real = dit._ltx25_has_dev_dit
    # Hub ids without the network; local packs through the real check.
    monkeypatch.setattr(
        dit,
        "_ltx25_has_dev_dit",
        lambda repo, token: repo == PACK if not Path(repo).is_absolute() else real(repo, token),
    )
    hub = types.SimpleNamespace(base_model = PACK)
    assert dit._ltx25_dit_subfolder(hub) == "transformer_full"
    full = _index(tmp_path / "full", "Gemma4UnifiedForConditionalGeneration")
    (full / "transformer_full").mkdir()
    (full / "transformer_full" / "config.json").write_text("{}")
    assert (
        dit._ltx25_dit_subfolder(types.SimpleNamespace(base_model = str(full))) == "transformer_full"
    )
    # A local pack without the dev DiT trains the distilled one it ships.
    bare = _index(tmp_path / "bare", "Gemma4UnifiedForConditionalGeneration")
    assert dit._ltx25_dit_subfolder(types.SimpleNamespace(base_model = str(bare))) == "transformer"
    monkeypatch.setenv(dit.LTX25_TRAIN_DIT_ENV, "distilled")
    assert dit._ltx25_dit_subfolder(hub) == "transformer"


def test_the_mirror_fetch_takes_the_dev_dit_from_the_upstream_pack(monkeypatch):
    """The mirror ships only the distilled DiT, so the dev DiT comes from the upstream pack the token can read; with
    no readable dev DiT anywhere the run trains the mirror's distilled DiT."""
    monkeypatch.delenv(dit.LTX25_TRAIN_DIT_ENV, raising = False)
    asked = []
    monkeypatch.setattr(
        dit, "_ltx25_has_dev_dit", lambda repo, token: asked.append((repo, token)) or repo == PACK
    )
    cfg = types.SimpleNamespace(base_model = MIRROR, hf_token = "tok")
    assert dit._ltx25_dit_source(cfg) == (PACK, "transformer_full")
    assert asked == [(MIRROR, "tok"), (PACK, "tok")]
    # Resolved once per run: the scheduler and the DiT load ask the same question.
    dit._ltx25_dit_source(cfg)
    assert len(asked) == 2

    monkeypatch.setattr(dit, "_LTX25_DIT_SOURCES", {})
    monkeypatch.setattr(dit, "_ltx25_has_dev_dit", lambda repo, token: False)
    assert dit._ltx25_dit_source(cfg) == (MIRROR, "transformer")


class _RecordingDiT:
    calls: list = []

    @classmethod
    def from_pretrained(cls, base, **kwargs):
        cls.calls.append({"base": base, **kwargs})
        return cls()

    def to(self, *a, **k):
        return self


@pytest.mark.parametrize("precision", ["nf4", "bf16", "int8"])
def test_every_base_precision_reads_the_chosen_subfolder(monkeypatch, precision):
    monkeypatch.delenv(dit.LTX25_TRAIN_DIT_ENV, raising = False)
    import diffusers

    _RecordingDiT.calls = []
    monkeypatch.setattr(diffusers, "LTX2VideoTransformer3DModel", _RecordingDiT)
    monkeypatch.setattr(dit, "_bnb_4bit_config", lambda: "nf4-config")
    monkeypatch.setattr(dit, "_ltx25_has_dev_dit", lambda repo, token: repo == PACK)
    cfg = types.SimpleNamespace(base_model = MIRROR, hf_token = None, resolved_family = FAMILY)
    _SPECS[FAMILY].load_transformer(cfg, "cpu", torch.bfloat16, precision)
    (kwargs,) = _RecordingDiT.calls
    assert kwargs["subfolder"] == "transformer_full" and kwargs["base"] == PACK
    assert ("quantization_config" in kwargs) == (precision == "nf4")


def _scheduler_dir(tmp_path: Path, config: dict) -> Path:
    folder = tmp_path / "sched_pack"
    (folder / "scheduler").mkdir(parents = True)
    (folder / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", **config})
    )
    return folder


def test_the_dev_scheduler_trains_on_ltx2s_sigma_table(monkeypatch, tmp_path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    monkeypatch.delenv(dit.LTX25_TRAIN_DIT_ENV, raising = False)
    pack = _scheduler_dir(tmp_path, LTX25_SCHEDULER)
    (pack / "transformer_full").mkdir()
    (pack / "transformer_full" / "config.json").write_text("{}")
    cfg = types.SimpleNamespace(base_model = str(pack), hf_token = None)
    dev = dit._ltx25_training_scheduler(cfg)
    assert dev.config.use_dynamic_shifting is True and dev.config.shift_terminal == pytest.approx(
        0.1
    )
    ltx2 = FlowMatchEulerDiscreteScheduler.from_config(LTX2_SCHEDULER)
    got, want = _training_sigma_table(dev, "auto"), _training_sigma_table(ltx2, "auto")
    assert torch.equal(got, want)
    # Shifted toward noise like LTX-2, not the distilled config's identity table.
    assert not torch.equal(got, dev.sigmas)

    monkeypatch.setenv(dit.LTX25_TRAIN_DIT_ENV, "distilled")
    distilled = dit._ltx25_training_scheduler(cfg)
    assert distilled.config.use_dynamic_shifting is False


# ── tiny end-to-end: train, publish, bake on the video load path ────────────

_TINY_DIT = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 8,
    "attention_out_bias": True,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "audio_cross_attn_mod": True,
    "audio_ff_bias": True,
    "audio_gated_attn": True,
    "audio_hop_length": 160,
    "audio_in_channels": 4,
    "audio_num_attention_heads": 2,
    "audio_out_channels": 4,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_scale_factor": 4,
    "base_height": 2048,
    "base_width": 2048,
    "caption_channels": 16,
    "causal_offset": 1,
    "cross_attention_dim": 16,
    "cross_attn_mod": True,
    "cross_attn_timestep_scale_multiplier": 1000,
    "ff_bias": False,
    "gated_attn": True,
    "in_channels": 8,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 8,
    "patch_size": 1,
    "patch_size_t": 1,
    "perturbed_attn": True,
    "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads",
    "rope_double_precision": False,
    "rope_theta": 10000.0,
    "rope_type": "split",
    "timestep_scale_multiplier": 1000,
    "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True,
    "use_prompt_embeddings": False,
    "vae_scale_factors": [8, 32, 32],
}
_TINY_VAE = {
    "block_out_channels": [8, 16, 32, 32],
    "decoder_block_out_channels": [8, 16, 16, 32],
    "decoder_causal": False,
    "decoder_inject_noise": [False] * 5,
    "decoder_layers_per_block": [1] * 5,
    "decoder_spatial_padding_mode": "reflect",
    "decoder_spatio_temporal_scaling": [True] * 4,
    "down_block_types": ["LTX2VideoDownBlock3D"] * 4,
    "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    "encoder_causal": True,
    "encoder_spatial_padding_mode": "zeros",
    "in_channels": 3,
    "latent_channels": 8,
    "layers_per_block": [1] * 5,
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "spatio_temporal_scaling": [True] * 4,
    "temporal_compression_ratio": 8,
    "timestep_conditioning": False,
    "upsample_factor": [2, 2, 1, 2],
    "upsample_residual": [False] * 4,
    "upsample_type": ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
}
_TINY_CONNECTORS = {
    "audio_connector_attention_head_dim": 8,
    "audio_connector_num_attention_heads": 2,
    "audio_connector_num_layers": 1,
    "audio_connector_num_learnable_registers": None,
    "audio_gated_attn": True,
    "audio_hidden_dim": 16,
    "caption_channels": 16,
    "causal_temporal_positioning": False,
    "connector_rope_base_seq_len": 32,
    "per_modality_projections": True,
    "proj_bias": True,
    "rope_double_precision": False,
    "rope_theta": 10000.0,
    "rope_type": "split",
    "text_proj_in_factor": 2,
    "video_connector_attention_head_dim": 8,
    "video_connector_num_attention_heads": 2,
    "video_connector_num_layers": 1,
    "video_connector_num_learnable_registers": None,
    "video_gated_attn": True,
    "video_hidden_dim": 16,
}


@pytest.fixture
def tiny_pack(tmp_path):
    """A tiny LTX-2.5 Diffusers pack: random dev and distilled DiTs, VAE, connectors, the pack's scheduler. Audio VAE,
    vocoder and encoder are listed but absent (the trainer passes the encoder as None; the audio parts are stubbed)."""
    diffusers = pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    pack = tmp_path / "LTX-2.5-tiny"
    for sub, seed in (("transformer", 0), ("transformer_full", 1)):
        torch.manual_seed(seed)
        LTX2VideoTransformer3DModel.from_config(_TINY_DIT).save_pretrained(pack / sub)
    torch.manual_seed(2)
    AutoencoderKLLTX2Video.from_config(_TINY_VAE).save_pretrained(pack / "vae")
    LTX2TextConnectors.from_config(_TINY_CONNECTORS).save_pretrained(pack / "connectors")
    _scheduler_dir(tmp_path, LTX25_SCHEDULER).joinpath("scheduler").rename(pack / "scheduler")
    _index(pack, "Gemma4UnifiedForConditionalGeneration", "duration_head")
    return pack


class _TinyCondPipe:
    """The parts of LTX2Pipeline the conditioning pass reads, loaded from the tiny pack like from_pretrained would."""

    calls: list = []

    @staticmethod
    def save_lora_weights(**kwargs):
        from diffusers import LTX2Pipeline
        LTX2Pipeline.save_lora_weights(**kwargs)

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        from diffusers import AutoencoderKLLTX2Video
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        cls.calls.append(kwargs)
        pipe = cls()
        pipe.vae = AutoencoderKLLTX2Video.from_pretrained(Path(source) / "vae")
        pipe.connectors = LTX2TextConnectors.from_pretrained(Path(source) / "connectors")
        pipe.text_encoder = pipe.tokenizer = pipe.transformer = None
        pipe._execution_device = torch.device("cpu")
        return pipe


@pytest.fixture
def tiny_trainer(tiny_pack, tmp_path, monkeypatch):
    """run(**cfg) trains on the tiny pack through run_dit_lora_training; every run gets a fresh fake worker client."""
    from PIL import Image

    monkeypatch.delenv(dit.LTX25_TRAIN_DIT_ENV, raising = False)
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (220, 40 * i, 30)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text("a red tile in sks style")
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: "bf16")
    # The tiny pack lists components it does not ship (encoder, audio parts).
    monkeypatch.setattr(dit, "_assert_trusted_base_model", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    _TinyCondPipe.calls = []
    monkeypatch.setattr(dit, "_ltx2_pipeline_cls", lambda: _TinyCondPipe)
    # Real diffusers pack / unpack helpers (static methods on the pipeline).
    from diffusers import LTX2Pipeline

    monkeypatch.setattr(
        dit,
        "_ltx2_pack",
        lambda lat, conf: LTX2Pipeline._pack_latents(lat, conf.patch_size, conf.patch_size_t),
    )
    monkeypatch.setattr(
        dit,
        "_ltx2_unpack",
        lambda pred, f, h, w, conf: LTX2Pipeline._unpack_latents(
            pred, f, h, w, conf.patch_size, conf.patch_size_t
        ),
    )
    clients: list = []

    def start(cfg, device):
        clients.append(
            _FakeClient(
                width = _TINY_CONNECTORS["caption_channels"] * _TINY_CONNECTORS["text_proj_in_factor"]
            )
        )
        return clients[-1]

    monkeypatch.setattr(dit, "_ltx25_start_encoder", start)
    loaded: list = []
    spec = _SPECS[FAMILY]
    load = spec.load_transformer

    def load_float(cfg, device, weight_dtype, base_precision):
        loaded.append(dit._ltx25_dit_subfolder(cfg))
        return load(cfg, device, weight_dtype, base_precision).float()

    monkeypatch.setattr(spec, "load_transformer", load_float)

    def run(name = "out", **extra):
        events: list = []
        cfg = DiffusionLoraConfig(
            base_model = str(tiny_pack),
            data_dir = str(data),
            output_dir = str(tmp_path / name),
            resolution = 64,
            train_steps = 4,
            train_batch_size = 1,
            learning_rate = 1e-2,
            lora_rank = 4,
            cache_variants = 1,
            seed = 0,
            **extra,
        )
        dit.run_dit_lora_training(cfg, on_event = events.append)
        assert not [e for e in events if e.get("type") == "error"]
        return events

    return types.SimpleNamespace(run = run, clients = clients, loaded = loaded, catalog = catalog)


def test_tiny_end_to_end_trains_the_dev_dit_and_bakes_on_the_distilled_one(tiny_pack, tiny_trainer):
    from safetensors.torch import load_file

    from core.inference import video_lora
    from diffusers import LTX2VideoTransformer3DModel

    events = tiny_trainer.run()
    assert tiny_trainer.loaded == ["transformer_full"]
    (client,) = tiny_trainer.clients
    assert client.prompts == ["a red tile in sks style"] and client.closed == 1
    losses = [
        e["loss"] for e in events if e.get("type") == "progress" and e.get("loss") is not None
    ]
    assert len(losses) == 4 and all(torch.isfinite(torch.tensor(losses)))
    (done,) = [e for e in events if e.get("type") == "complete"]
    adapter = Path(done["lora_path"])
    assert adapter.name == DEFAULT_LORA_FILENAME
    keys = list(load_file(str(adapter)))
    assert keys and all(".attn1." in k or ".attn2." in k for k in keys)
    published = [p for p in tiny_trainer.catalog.glob("*.safetensors")]
    assert len(published) == 1
    sidecar = json.loads(published[0].with_suffix(".json").read_text(encoding = "utf-8"))
    assert sidecar["families"] == [FAMILY]

    # The Video page bakes it into the DISTILLED DiT; the delta scales with the slider weight.
    def baked(weight):
        distilled = LTX2VideoTransformer3DModel.from_pretrained(tiny_pack / "transformer")
        before = distilled.transformer_blocks[0].attn1.to_q.weight.detach().clone()
        merged = video_lora.bake_into_module_(distilled, [(str(published[0]), weight)])
        return merged, distilled.transformer_blocks[0].attn1.to_q.weight.detach() - before

    merged, full = baked(1.0)
    _, half = baked(0.5)
    assert merged == len(keys) // 2
    assert full.abs().sum() > 0
    torch.testing.assert_close(half * 2, full, rtol = 1e-4, atol = 1e-6)


def test_a_warm_caption_cache_never_starts_the_encoder_worker(tiny_trainer, tmp_path):
    """Studio's conditioning cache serves the second run's captions and latents, so no Gemma 4 worker is spawned."""
    cache = tmp_path / "cond_cache"
    first = tiny_trainer.run("first", cond_cache_dir = str(cache))
    assert len(tiny_trainer.clients) == 1 and tiny_trainer.clients[0].closed == 1
    second = tiny_trainer.run("second", cond_cache_dir = str(cache))
    assert len(tiny_trainer.clients) == 1, "a warm cache must not start the encoder worker"

    def losses(events):
        return [
            e["loss"] for e in events if e.get("type") == "progress" and e.get("loss") is not None
        ]

    # Same conditioning from the cache: the same seeded run.
    assert losses(second) == losses(first)
