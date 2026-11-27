# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the FLUX.3 Klein Base DiT LoRA trainer family. A random Flux3Transformer2DModel built from a config
covers the offline checks; a tiny FLUX.3 pipeline folder (hf-internal-testing/tiny-flux3-image-modular's VAE and
processor, a 32-layer random Qwen3-VL so the real hidden-state layers exist; skipped when unreachable) proves the
trainer's conditioning and forward reproduce what Studio's FLUX.3 pipeline feeds the transformer, and that a trained
adapter loads and renders on both the base and the step-distilled pipeline."""

from __future__ import annotations

import json
import math
import shutil
import sys

import numpy as np
import pytest
import torch

from core.inference.diffusion_families import detect_family
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_dit_trainer import (
    _FLUX3_TARGETS,
    _SPECS,
    _select_lora_targets,
    _training_sigma_table,
)
from core.training.diffusion_samples import (
    SAMPLE_FAMILIES,
    flow_sigmas,
    sample_inference_settings,
)
from core.training.diffusion_train_common import (
    AUTO_FLOW_SHIFT_FAMILIES,
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    MODULAR_BASE_FAMILIES,
    DiffusionLoraConfig,
    _DIT_TRAIN_FAMILIES,
    _FORCE_BF16_FAMILIES,
    get_trainer,
    lora_run_families,
    resolve_trainable_family,
    train_defaults,
)

FAMILY = "flux.3-klein-base"
DISTILLED_FAMILY = "flux.3-klein"
BASE = "black-forest-labs/flux-3-image-klein-base"
KLEIN = "black-forest-labs/flux-3-image-klein"
TINY_REPO = "hf-internal-testing/tiny-flux3-image-modular"
C = 16
CTX = 32


def _transformer_cls():
    try:
        from diffusers import Flux3Transformer2DModel
    except Exception as exc:  # noqa: BLE001 -- diffusers older than 0.42, or none (the CPU CI shard)
        pytest.skip(f"diffusers FLUX.3 classes unavailable: {exc}")
    return Flux3Transformer2DModel


def _needs_flux3_pipeline():
    """Resolving / starting a FLUX.3 run checks the installed diffusers can build it (0.42+); skip where it cannot."""
    from core.inference.diffusion_flux3 import flux3_available
    if not flux3_available():
        pytest.skip("this diffusers cannot build FLUX.3 (needs 0.42+)")


def _tiny_transformer(
    seed = 0,
    flow_map = False,
    joint = CTX,
):
    T = _transformer_cls()
    torch.manual_seed(seed)
    return T(
        in_channels = C,
        cond_in_channels = C,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = joint,
        timestep_channels = 32,
        axes_dims_rope = (2, 2, 2, 2),
        flow_map = flow_map,
    ).eval()


def _adapted(seed = 0, rank = 4):
    from peft import LoraConfig

    t = _tiny_transformer(seed)
    t.requires_grad_(False)
    t.add_adapter(
        LoraConfig(
            r = rank,
            lora_alpha = rank,
            init_lora_weights = "gaussian",
            target_modules = list(_FLUX3_TARGETS),
        )
    )
    return t


def _build_tiny_pipe(
    src,
    dst,
    *,
    distilled = False,
):
    """The tiny repo laid out as the official one (processor inside text_encoder/), with a 32-layer encoder so the
    pipeline's default hidden-state layers (4 ... 32) exist, and a transformer sized to their stacked width."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    dst.mkdir(parents = True)
    cfg = Qwen3VLConfig.from_pretrained(src / "text_encoder")
    cfg.text_config.num_hidden_layers = 32
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(cfg).eval().save_pretrained(dst / "text_encoder")
    for f in (src / "tokenizer").iterdir():
        shutil.copy(f, dst / "text_encoder" / f.name)
    tcfg = {
        k: v
        for k, v in json.loads((src / "transformer" / "config.json").read_text()).items()
        if not k.startswith("_")
    }
    tcfg["joint_attention_dim"] = 8 * cfg.text_config.hidden_size
    T = _transformer_cls()
    torch.manual_seed(1)
    base = T(**tcfg)
    if distilled:
        torch.manual_seed(2)
        flow = T(**{**tcfg, "flow_map": True})
        # The distilled model is the base DiT plus the target-time embedder.
        flow.load_state_dict(base.state_dict(), strict = False)
        base = flow
    base.save_pretrained(dst / "transformer")
    shutil.copytree(src / "vae", dst / "vae")
    index = json.loads((src / "modular_model_index.json").read_text())
    if distilled:
        index["_blocks_class_name"] = "Flux3DistilledAutoBlocks"
        index["_class_name"] = "Flux3DistilledModularPipeline"
    (dst / "modular_model_index.json").write_text(json.dumps(index))
    return str(dst)


@pytest.fixture
def tiny_src():
    """Function-scoped so the tests' allow_network marker applies."""
    _transformer_cls()
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(TINY_REPO))
    except Exception as exc:  # noqa: BLE001 -- offline CI
        pytest.skip(f"{TINY_REPO} unavailable: {exc}")


@pytest.fixture
def tiny_pipe_dir(tiny_src, tmp_path):
    return _build_tiny_pipe(tiny_src, tmp_path / "tiny-flux3-base")


def _load(path, family = FAMILY):
    from core.inference.diffusion_flux3 import load_flux3_pipeline
    return load_flux3_pipeline(path, torch.float32, family_name = family)


def _render(
    pipe,
    prompt = "a red tile",
    **kw,
):
    # 32 px on the tiny VAE (2x down, 2x2 patch): an 8 x 8 grid of 16-channel tokens.
    latents = torch.randn(1, C, 8, 8, generator = torch.Generator().manual_seed(1))
    return pipe(
        prompt,
        height = 32,
        width = 32,
        num_inference_steps = 2,
        latents = latents,
        output_type = "pt",
        **kw,
    ).images[0]


# --- registration -------------------------------------------------------------------------------------------------


def test_spec_is_registered_bf16_only_and_routes_to_the_dit_trainer():
    spec = _SPECS[FAMILY]
    assert spec.family == FAMILY and spec.force_bf16
    assert FAMILY in _DIT_TRAIN_FAMILIES and FAMILY in _FORCE_BF16_FAMILIES
    assert FAMILY in SAMPLE_FAMILIES and FAMILY in MODULAR_BASE_FAMILIES
    # The scheduler's static 6.93 shift is the schedule; "auto" would add nothing.
    assert FAMILY not in AUTO_FLOW_SHIFT_FAMILIES
    assert get_trainer(FAMILY) is dit.run_dit_lora_training
    assert spec.latent_shape is not None and spec.decode_latents is not None
    assert spec.save_takes_metadata and spec.training_scheduler is not None
    assert spec.training_mu is None and spec.timestep_density == "none"
    # The distilled Klein has no trainer: it is a deploy target only.
    assert DISTILLED_FAMILY not in _SPECS


def test_family_trains_on_base_and_deploys_on_klein():
    fam = detect_family(BASE)
    assert fam is not None and fam.name == FAMILY and fam.trainable
    assert fam.train_base_repos == (BASE,)
    assert fam.deploy_base_for(BASE) == KLEIN
    klein = detect_family(KLEIN)
    assert klein.name == DISTILLED_FAMILY and not klein.trainable
    _needs_flux3_pipeline()
    assert resolve_trainable_family(BASE, None) == FAMILY
    with pytest.raises(ValueError):
        resolve_trainable_family(KLEIN, None)


def test_train_ui_note_says_where_the_adapter_runs():
    from core.training.diffusion_train_common import _FAMILY_TRAIN_SPECS
    assert (
        _FAMILY_TRAIN_SPECS[FAMILY]["note"] == "Trains on FLUX.3 Klein Base, runs on Klein or Base."
    )


def test_train_ui_pairs_the_base_with_klein():
    from core.training.diffusion_train_common import family_train_infos

    # family_train_infos drops a family whose pipeline the installed diffusers lacks.
    _needs_flux3_pipeline()
    info = next(i for i in family_train_infos() if i["name"] == FAMILY)
    assert info["default_base"] == BASE and info["label"] == "FLUX.3 Klein Base"
    # The mirror the trainer fetches pairs with Klein's own mirror (both hold the upstream diffusers folders).
    assert info["deploy_bases"] == {
        BASE: KLEIN,
        "unsloth/flux-3-image-klein-base-FP8": "unsloth/flux-3-image-klein-FP8",
    }
    assert info["supports_samples"] and info["max_train_batch_size"] is None
    assert info["lora_families"] == [FAMILY, DISTILLED_FAMILY]
    from models.training import DiffusionTrainableFamily

    # The info route serves each row through this model, which drops a field it does not declare.
    assert DiffusionTrainableFamily(**info).lora_families == [FAMILY, DISTILLED_FAMILY]
    for other in family_train_infos():
        if other["name"] != FAMILY:
            assert DiffusionTrainableFamily(**other).lora_families == [other["name"]]


def test_defaults_sit_on_the_16_grid_and_off_grid_resolutions_are_refused(tmp_path):
    d = train_defaults(FAMILY)
    assert d["resolution"] % 16 == 0 and d["lora_rank"] > 0 and d["learning_rate"] > 0
    _needs_flux3_pipeline()
    base = dict(
        base_model = BASE,
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "o"),
    )
    DiffusionLoraConfig(**base, resolution = 768).normalized()
    with pytest.raises(ValueError, match = "multiple of 16"):
        DiffusionLoraConfig(**base, resolution = 776).normalized()


def test_previews_run_the_inference_recipe_with_cfg():
    from core.inference.diffusion_families import named_generation_params
    from core.inference.diffusion_flux3 import FLUX3_REFERENCE_SHIFT

    s = sample_inference_settings(FAMILY, BASE)
    assert (s.steps, s.guidance, s.uses_cfg, s.shift) == (28, 4.0, True, 16.0)
    assert s.guidance == named_generation_params(BASE)[1] == named_generation_params(FAMILY)[1]
    assert s.shift == FLUX3_REFERENCE_SHIFT


def test_trained_adapters_are_offered_on_klein_and_klein_base():
    from core.inference.diffusion_lora import lora_unavailable_reason
    assert lora_run_families(FAMILY) == [FAMILY, DISTILLED_FAMILY]
    for family in (FAMILY, DISTILLED_FAMILY):
        for quant in (None, "int8", "fp8"):
            assert (
                lora_unavailable_reason(
                    engine = "diffusers",
                    family = family,
                    model_kind = "diffusers",
                    transformer_quant = quant,
                )
                is None
            ), (family, quant)


def test_every_other_family_publishes_for_itself_only():
    from core.inference.diffusion_families import trainable_family_names
    for name in trainable_family_names():
        if name != FAMILY:
            assert lora_run_families(name) == [name], name


def test_the_catalog_offers_a_published_adapter_to_both_families(tmp_path, monkeypatch):
    from core.inference import diffusion_lora
    from core.training.diffusion_train_common import _write_lora_sidecar

    monkeypatch.setattr(diffusion_lora, "loras_dir", lambda: tmp_path)
    from safetensors.torch import save_file

    save_file(
        {"transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8)},
        str(tmp_path / "tarot.safetensors"),
    )
    # The resolved fields the sidecar reads, without normalized() (which checks the installed diffusers).
    cfg = DiffusionLoraConfig(
        base_model = BASE,
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "o"),
        instance_prompt = "trtcrd",
        resolved_family = FAMILY,
    )
    _write_lora_sidecar(tmp_path / "tarot.json", cfg, 10)
    families, _w = diffusion_lora._read_lora_sidecar(tmp_path / "tarot.safetensors")
    assert families == (FAMILY, DISTILLED_FAMILY)


# --- LoRA targets -------------------------------------------------------------------------------------------------


def test_targets_adapt_every_attention_and_mlp_linear_and_nothing_else():
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _SPECS[FAMILY].lora_targets) == _FLUX3_TARGETS
    t = _adapted()
    adapted = {n for n, m in t.named_modules() if hasattr(m, "lora_A")}
    leaves = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.linear_in",
        "ff.linear_out",
    )
    blocks = [
        *(f"mode_blocks.image.{i}" for i in range(2)),
        *(f"mode_blocks.image_cond.{i}" for i in range(2)),
        *(f"txt_mode_blocks.{i}" for i in range(2)),
        *(f"single_transformer_blocks.{i}" for i in range(2)),
    ]
    expected = {f"{b}.{leaf}" for b in blocks for leaf in leaves}
    assert adapted == expected
    # Every Linear inside a block is covered; modulations, embedders, norm_out and proj_out are not.
    block_linears = {
        n
        for n, m in _tiny_transformer().named_modules()
        if isinstance(m, torch.nn.Linear) and n.startswith(tuple(f"{b}." for b in blocks))
    }
    assert block_linears == expected


def test_nf4_keeps_the_inference_keep_list_dense():
    _transformer_cls()
    cfg = dit._flux3_nf4_config()
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4"
    skip = set(cfg.llm_int8_skip_modules)
    names = [
        n
        for n, m in _tiny_transformer(flow_map = True).named_modules()
        if isinstance(m, torch.nn.Linear)
    ]

    def kept(name):
        # bitsandbytes' rule: a module is skipped when "<key>." is in its dotted path, or it IS the key.
        return any(k + "." in name or k == name for k in skip)

    for n in names:
        dense = any(
            tok in n
            for tok in (
                "modulation",
                "timestep_embed",
                "norm_out",
                "txt_mode_blocks",
                "context_embedder",
            )
        )
        assert kept(n) == dense, n


# --- conditioning + forward ---------------------------------------------------------------------------------------


def test_forward_returns_the_image_tokens_in_the_latent_layout():
    t = _tiny_transformer()
    noisy = torch.randn(2, C, 4, 6)
    pes = [torch.randn(1, 80, CTX), torch.randn(1, 80, CTX)]
    with torch.no_grad():
        out = dit._flux3_forward(
            t, noisy, torch.tensor([300.0, 800.0]), None, (pes,), None, "cpu", torch.float32
        )
    assert out.shape == noisy.shape and bool(torch.isfinite(out).all())


def test_mixed_window_lengths_run_unpadded_and_match_one_sample_at_a_time():
    t = _tiny_transformer()
    noisy = torch.randn(3, C, 4, 4)
    ts = torch.tensor([100.0, 500.0, 900.0])
    pes = [torch.randn(1, 80, CTX), torch.randn(1, 160, CTX), torch.randn(1, 80, CTX)]
    with torch.no_grad():
        batched = dit._flux3_forward(t, noisy, ts, None, (pes,), None, "cpu", torch.float32)
        single = torch.cat(
            [
                dit._flux3_forward(
                    t,
                    noisy[i : i + 1],
                    ts[i : i + 1],
                    None,
                    ([pes[i]],),
                    None,
                    "cpu",
                    torch.float32,
                )
                for i in range(3)
            ]
        )
    assert torch.allclose(batched, single, atol = 1e-5)


@pytest.mark.allow_network
def test_encode_and_forward_reproduce_the_pipelines_transformer_call(tiny_pipe_dir):
    """Run Studio's FLUX.3 Klein Base pipeline, capture what it hands the transformer, and check the trainer builds the
    same conditioning and the same call."""
    pipe = _load(tiny_pipe_dir)
    calls = []
    hook = pipe.transformer.register_forward_hook(
        lambda _m, _a, kwargs, out: calls.append((dict(kwargs), out[0])), with_kwargs = True
    )
    prompt = "a watercolor fox in the snow"
    _render(pipe, prompt, guidance_scale = 3.5)
    hook.remove()

    (embeds,) = dit._flux3_encode_prompts(pipe, [prompt], "cpu")
    # The base pipeline runs CFG: one call per branch and step. The conditional one carries the prompt's window.
    cond = [(k, o) for k, o in calls if torch.equal(k["encoder_hidden_states"], embeds[0])]
    assert cond and len(cond) == len(calls) // 2
    kwargs, out = cond[0]
    assert embeds[0].shape[1] % 80 == 0
    assert torch.equal(kwargs["txt_ids"], dit._flux3_txt_ids(embeds[0].shape[1], "cpu"))
    assert torch.equal(kwargs["img_ids"], dit._flux3_img_ids(8, 8, "cpu"))
    assert "target_timestep" not in kwargs or kwargs["target_timestep"] is None

    packed = kwargs["hidden_states"]
    noisy = packed.transpose(1, 2).reshape(1, C, 8, 8)
    timesteps = kwargs["timestep"] * 1000
    with torch.no_grad():
        mine = dit._flux3_forward(
            pipe.transformer,
            noisy,
            timesteps,
            None,
            dit._flux3_collate([embeds], "cpu", torch.float32),
            None,
            "cpu",
            torch.float32,
        )
    expected = out.transpose(1, 2).reshape(1, C, 8, 8)
    assert torch.allclose(mine, expected, atol = 1e-6)


@pytest.mark.allow_network
def test_inference_load_samples_at_the_reference_shift_and_training_keeps_its_own(tiny_pipe_dir):
    from core.inference.diffusion_flux3 import FLUX3_REFERENCE_SHIFT

    pipe = _load(tiny_pipe_dir)
    assert pipe.scheduler.config.shift == FLUX3_REFERENCE_SHIFT
    assert pipe.modular_pipeline().scheduler is pipe.scheduler
    assert dit._flux3_training_scheduler(None).config.shift == pytest.approx(6.93)


@pytest.mark.allow_network
def test_conditioning_load_skips_the_transformer_and_encodes_the_same(tiny_pipe_dir):
    from core.inference.diffusion_flux3 import load_flux3_pipeline

    # The trainer encodes in bf16, so compare against a bf16 load of the whole pipeline.
    full = load_flux3_pipeline(tiny_pipe_dir, torch.bfloat16, family_name = FAMILY)
    cfg = type("Cfg", (), {"base_model": tiny_pipe_dir, "hf_token": None})()
    pipe, vae = dit._flux3_load_conditioners(cfg, "cpu", torch.float32)
    assert pipe.transformer is None and vae is pipe.vae and vae.dtype == torch.float32
    assert not pipe.is_distilled
    (a,), (b,) = (
        dit._flux3_encode_prompts(pipe, ["trtcrd, a tarot card"], "cpu"),
        dit._flux3_encode_prompts(full, ["trtcrd, a tarot card"], "cpu"),
    )
    assert torch.equal(a[0], b[0])


@pytest.mark.allow_network
def test_freeing_the_encoders_releases_the_text_encoder(tiny_pipe_dir):
    """The facade's block-graph pipeline holds its own reference to every component, so clearing pipe.text_encoder
    alone left the encoder alive (and on the GPU) for the whole training loop."""
    import gc
    import weakref

    cfg = type("Cfg", (), {"base_model": tiny_pipe_dir, "hf_token": None})()
    pipe, vae = dit._flux3_load_conditioners(cfg, "cpu", torch.float32)
    dit._flux3_encode_prompts(pipe, ["trtcrd"], "cpu")
    assert pipe._modular is not None
    encoder = weakref.ref(pipe.text_encoder)
    dit._free_text_encoders(pipe)
    gc.collect()
    assert encoder() is None
    assert pipe.vae is vae


@pytest.mark.allow_network
def test_latents_normalise_like_the_vae_encoder_step_and_decode_inverts_them(tiny_pipe_dir):
    from diffusers import AutoencoderKLFlux2
    from diffusers.modular_pipelines.flux3.encoders import Flux3VaeEncoderStep

    vae = AutoencoderKLFlux2.from_pretrained(tiny_pipe_dir, subfolder = "vae").float()
    px = torch.rand(1, 3, 32, 32) * 2 - 1
    lat, std = dit._flux3_encode_latent_stats(vae, px)
    with torch.no_grad():
        raw = Flux3VaeEncoderStep._patchify_latents(vae.encode(px).latent_dist.mode())
    m = vae.bn.running_mean.view(1, -1, 1, 1)
    s = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    assert std is None and torch.allclose(lat, (raw - m) / s, atol = 1e-6)
    with torch.no_grad():
        img = _SPECS[FAMILY].decode_latents(vae, lat)
    assert img.shape == (1, 3, 32, 32)


@pytest.mark.allow_network
def test_distilled_checkpoint_is_refused_as_a_training_base(tiny_src, tmp_path):
    path = _build_tiny_pipe(tiny_src, tmp_path / "tiny-flux3-distilled", distilled = True)
    cfg = type("Cfg", (), {"base_model": path, "hf_token": None})()
    with pytest.raises(ValueError, match = "step-distilled"):
        dit._flux3_load_transformer(cfg, "cpu", torch.float32, "bf16")


# --- sigma schedule -----------------------------------------------------------------------------------------------


def test_training_sigmas_are_the_pipelines_shifted_schedule():
    _transformer_cls()
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = dit._flux3_training_scheduler(None)
    assert sched.config.shift == pytest.approx(6.93)
    assert not sched.config.use_dynamic_shifting
    # The trainer's table: flow_shift unset (the family default), so the scheduler's own sigmas.
    table = _training_sigma_table(sched, None)
    # What Flux3SetTimestepsStep does at inference: linspace(1, 1/N, N) through set_timesteps (static shift).
    n = sched.config.num_train_timesteps
    ref = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
    ref.set_timesteps(sigmas = np.linspace(1.0, 1.0 / n, n).tolist())
    assert torch.allclose(table, ref.sigmas[:-1].to(table.dtype), atol = 1e-6)
    u = torch.linspace(1.0, 1.0 / n, n)
    assert torch.allclose(table, 6.93 * u / (1 + 5.93 * u), atol = 1e-5)
    # Timesteps are sigma * 1000, so the forward's timestep / 1000 is exactly the pipeline's flow time.
    assert torch.allclose(sched.timesteps, table * n, atol = 1e-3)
    # Without a shift override the preview schedule is the training scheduler's own.
    sig, _ = flow_sigmas(sched.config, FAMILY, 28, 64 * 64)
    ref.set_timesteps(sigmas = np.linspace(1.0, 1.0 / 28, 28).tolist())
    assert torch.allclose(sig, ref.sigmas.float(), atol = 1e-6)


def _inference_sigmas(steps):
    """The schedule Studio's FLUX.3 inference samples (load_flux3_pipeline's scheduler, Flux3SetTimestepsStep)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.inference.diffusion_flux3 import (
        FLUX3_BASE_BLOCKS,
        flux3_blocks,
        flux3_reference_scheduler,
    )

    shipped = flux3_blocks(FLUX3_BASE_BLOCKS).init_pipeline().scheduler
    ref = FlowMatchEulerDiscreteScheduler.from_config(flux3_reference_scheduler(shipped).config)
    ref.set_timesteps(sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist())
    return ref.sigmas.float()


def test_preview_sigmas_are_the_inference_schedule_not_the_training_one():
    _transformer_cls()
    sched = dit._flux3_training_scheduler(None)
    s = sample_inference_settings(FAMILY, BASE)
    sig, num_train = flow_sigmas(sched.config, FAMILY, s.steps, 64 * 64, s.mu, s.shift)
    assert torch.allclose(sig, _inference_sigmas(s.steps), atol = 1e-6)
    assert num_train == 1000.0
    assert sched.config.shift == pytest.approx(6.93)
    assert not torch.allclose(
        sig, flow_sigmas(sched.config, FAMILY, s.steps, 64 * 64)[0], atol = 1e-3
    )


def test_preview_round_denoises_at_the_inference_shift_and_cfg(tmp_path):
    from types import SimpleNamespace

    from core.training.diffusion_samples import euler_flow_sample, plan_samples

    _transformer_cls()
    sched = dit._flux3_training_scheduler(None)
    cfg = SimpleNamespace(
        sample_every = 10,
        sample_prompts = ("a cat",),
        base_model = BASE,
        seed = 0,
        resolution = 512,
        output_dir = str(tmp_path),
    )
    plan = plan_samples(cfg, FAMILY, ["a cat"])
    seen = []

    def velocity(x, t, sig, embeds):
        seen.append((embeds, float(sig.flatten()[0]), float(t[0])))
        return torch.full_like(x, 1.0 if embeds == "cond" else 0.0)

    shape = (1, 4, 2, 2)
    x = euler_flow_sample(
        plan = plan,
        index = 0,
        latent_shape = shape,
        image_seq_len = 64 * 64,
        family = FAMILY,
        scheduler_config = sched.config,
        velocity = velocity,
        cond = "cond",
        uncond = "uncond",
        device = "cpu",
        weight_dtype = torch.float32,
    )
    ref = _inference_sigmas(28)
    cond_sigmas = torch.tensor([s for e, s, _ in seen if e == "cond"])
    assert torch.allclose(cond_sigmas, ref[:-1], atol = 1e-6)
    assert [e for e, _, _ in seen] == ["cond", "uncond"] * 28
    # Constant cond 1 / uncond 0 velocities integrate to guidance * (sigma_end - sigma_start) = -4.0.
    from core.training.diffusion_samples import initial_noise

    assert torch.allclose(
        x - initial_noise(plan, 0, shape, "cpu"), torch.full(shape, -4.0), atol = 1e-5
    )


def test_timestep_draws_are_uniform_for_flux3_and_logit_normal_elsewhere():
    _transformer_cls()
    sched = dit._flux3_training_scheduler(None)
    torch.manual_seed(0)
    _, idx = dit._sample_timesteps(sched, 20000, "cpu", _SPECS[FAMILY].timestep_density)
    frac_tail = float(((idx < 100) | (idx >= 900)).float().mean())
    # Uniform puts ~20% of draws in the outer tenths; logit-normal (0, 1) puts ~7% there.
    assert 0.18 < frac_tail < 0.22
    torch.manual_seed(0)
    _, idx = dit._sample_timesteps(sched, 20000, "cpu")
    assert float(((idx < 100) | (idx >= 900)).float().mean()) < 0.1
    assert all(s.timestep_density == "logit_normal" for f, s in _SPECS.items() if f != FAMILY)


# --- training + save + reload -------------------------------------------------------------------------------------


def test_loss_falls_on_a_fixed_batch():
    t = _adapted(rank = 8)
    t.train()
    params = [p for p in t.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr = 1e-2)
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(2, C, 4, 4, generator = g)
    noise = torch.randn(2, C, 4, 4, generator = g)
    sig = torch.tensor([0.3, 0.7]).view(2, 1, 1, 1)
    noisy = (1 - sig) * latents + sig * noise
    pes = [torch.randn(1, 80, CTX, generator = g), torch.randn(1, 80, CTX, generator = g)]
    losses = []
    for _ in range(25):
        opt.zero_grad()
        pred = dit._flux3_forward(
            t, noisy, sig.flatten() * 1000, sig, (pes,), None, "cpu", torch.float32
        )
        loss = torch.nn.functional.mse_loss(pred, noise - latents)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < 0.8 * losses[0], losses


def test_saved_lora_has_diffusers_keys_and_reloads_at_the_trained_strength(tmp_path):
    _transformer_cls()
    try:
        from diffusers.loaders import Flux3LoraLoaderMixin
    except ImportError as exc:  # a build with the transformer but no FLUX.3 LoRA loader
        pytest.skip(f"diffusers FLUX.3 LoRA loader unavailable: {exc}")
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    trained = _adapted(rank = 4)
    # alpha != rank: the metadata must carry the scale, or the loader assumes alpha == rank.
    trained.peft_config["default"].lora_alpha = 8
    for m in trained.modules():
        if hasattr(m, "scaling") and "default" in getattr(m, "scaling", {}):
            m.scaling["default"] = 8 / 4
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.5)
    dit._flux3_save(
        None,
        str(tmp_path),
        get_peft_model_state_dict(trained),
        adapter_metadata = dit._lora_adapter_metadata(trained),
    )
    path = tmp_path / DEFAULT_LORA_FILENAME
    keys = set(load_file(str(path)))
    assert keys and all(k.startswith("transformer.") for k in keys)
    assert "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight" in keys
    assert "transformer.txt_mode_blocks.1.ff.linear_out.lora_B.weight" in keys
    assert "transformer.mode_blocks.image.0.attn.to_out.0.lora_A.weight" in keys

    fresh = _tiny_transformer()
    state_dict, metadata = Flux3LoraLoaderMixin.lora_state_dict(
        str(path), return_lora_metadata = True
    )
    Flux3LoraLoaderMixin.load_lora_into_transformer(
        state_dict, transformer = fresh, adapter_name = "style", metadata = metadata
    )
    noisy = torch.randn(1, C, 4, 4)
    pes = [torch.randn(1, 80, CTX)]
    t = torch.tensor([500.0])
    with torch.no_grad():
        with_lora = dit._flux3_forward(fresh, noisy, t, None, (pes,), None, "cpu", torch.float32)
        trained_out = dit._flux3_forward(
            trained, noisy, t, None, (pes,), None, "cpu", torch.float32
        )
        fresh.disable_adapters()
        off = dit._flux3_forward(fresh, noisy, t, None, (pes,), None, "cpu", torch.float32)
    assert torch.allclose(with_lora, trained_out, atol = 1e-5)
    assert not torch.allclose(with_lora, off)


@pytest.mark.allow_network
def test_end_to_end_run_trains_saves_and_both_pipelines_load_the_adapter(
    tiny_src, tiny_pipe_dir, tmp_path, monkeypatch
):
    """The whole run_dit_lora_training loop on the tiny pipe: caption cache, latent cache, mixed window lengths in a
    batch, a CFG preview round, save, catalog publish; then Studio's Klein Base AND distilled Klein pipelines load the
    adapter and it changes their renders."""
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), (200, 30 * i, 40)).save(data / f"img{i}.png")
        # One caption long enough to need a second 80-token window.
        (data / f"img{i}.txt").write_text(
            "trtcrd, a red tile" + (" with a long trailing caption" * 12 if i % 2 else "")
        )
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    # CPU host: no nf4 / bf16 autocast, and unsloth_zoo refuses to import without an accelerator.
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: "bf16")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    spec = _SPECS[FAMILY]
    load = spec.load_transformer
    monkeypatch.setattr(spec, "load_transformer", lambda *a, **k: load(*a, **k).float())
    events = []
    out_dir = tmp_path / "out"
    cfg = DiffusionLoraConfig(
        base_model = tiny_pipe_dir,
        model_family = FAMILY,
        data_dir = str(data),
        output_dir = str(out_dir),
        resolution = 64,
        train_steps = 4,
        train_batch_size = 2,
        learning_rate = 1e-2,
        lora_rank = 4,
        cache_variants = 1,
        sample_every = 4,
        seed = 0,
    )
    dit.run_dit_lora_training(cfg, on_event = events.append)
    progress = [e for e in events if e.get("type") == "progress"]
    assert len(progress) == 4 and all(math.isfinite(e["loss"]) for e in progress)
    done = [e for e in events if e.get("type") == "complete"][-1]
    assert done["family"] == FAMILY and done["lora_path"] and not done["stopped"]
    assert any(e.get("type") == "sample" for e in events)
    sidecar = json.loads(next(catalog.glob("*.json")).read_text())
    assert sidecar["families"] == [FAMILY, DISTILLED_FAMILY]

    distilled_dir = _build_tiny_pipe(tiny_src, tmp_path / "tiny-flux3-distilled", distilled = True)
    for path, family, kw in (
        (tiny_pipe_dir, FAMILY, {"guidance_scale": 3.5}),
        (distilled_dir, DISTILLED_FAMILY, {}),
    ):
        pipe = _load(path, family)
        assert pipe.is_distilled == (family == DISTILLED_FAMILY)
        base = _render(pipe, "trtcrd, a red tile", **kw)
        pipe.load_lora_weights(
            str(out_dir), weight_name = DEFAULT_LORA_FILENAME, adapter_name = "trained"
        )
        adapted = {n for n, m in pipe.transformer.named_modules() if hasattr(m, "lora_A")}
        # One block each of image, image_cond, text and single on the tiny config, 6 linears apiece.
        assert len(adapted) == 6 * 4, family
        with_lora = _render(pipe, "trtcrd, a red tile", **kw)
        assert not torch.allclose(base, with_lora), family
