# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the Qwen-Image-2.1 DiT LoRA trainer family: a random QwenImage21Transformer2DModel built from a
config for the offline checks, and hf-internal-testing/tiny-qwenimage21-pipe (skipped when unreachable) to prove the
trainer's conditioning and forward reproduce what QwenImage21Pipeline itself feeds the transformer."""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch

from core.inference.diffusion_families import detect_family
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_dit_trainer import (
    _QWEN21_TARGETS,
    _SPECS,
    _select_lora_targets,
    _training_sigma_table,
)
from core.training.diffusion_samples import SAMPLE_FAMILIES, sample_inference_settings
from core.training.diffusion_train_common import (
    AUTO_FLOW_SHIFT_FAMILIES,
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    _DIT_TRAIN_FAMILIES,
    _FORCE_BF16_FAMILIES,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)

FAMILY = "qwen-image-2.1"
BASE = "Qwen/Qwen-Image-2.1"
TURBO = "Qwen/Qwen-Image-2.1-Turbo"
TINY_PIPE = "hf-internal-testing/tiny-qwenimage21-pipe"
# Qwen/Qwen-Image-2.1 scheduler/scheduler_config.json.
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
C = 16
CTX = 32


def _classes():
    try:
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    except Exception as exc:  # noqa: BLE001 -- diffusers older than 0.41
        pytest.skip(f"diffusers Qwen-Image-2.1 classes unavailable: {exc}")
    return QwenImage21Pipeline, QwenImage21Transformer2DModel


def _tiny_transformer(seed = 0):
    """The real block layout (single-stream, block-causal, causal_condition) at toy width."""
    _, T = _classes()
    torch.manual_seed(seed)
    return T(
        patch_size = 1,
        in_channels = C,
        out_channels = C,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        context_in_dim = CTX,
        mlp_ratio = 2,
        axes_dims_rope = (4, 6, 6),
        causal_condition = True,
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
            target_modules = list(_QWEN21_TARGETS),
        )
    )
    return t


@pytest.fixture
def tiny_pipe_dir():
    """Function-scoped so the tests' allow_network marker applies (as in test_diffusion_qwenimage21_large_render)."""
    _classes()
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(TINY_PIPE)
    except Exception as exc:  # noqa: BLE001 -- offline CI
        pytest.skip(f"{TINY_PIPE} unavailable: {exc}")


# --- registration -------------------------------------------------------------------------------------------------


def test_spec_is_registered_bf16_only_and_routes_to_the_dit_trainer():
    spec = _SPECS[FAMILY]
    assert spec.family == FAMILY and spec.force_bf16
    assert FAMILY in _DIT_TRAIN_FAMILIES and FAMILY in _FORCE_BF16_FAMILIES
    assert FAMILY in AUTO_FLOW_SHIFT_FAMILIES and FAMILY in SAMPLE_FAMILIES
    assert get_trainer(FAMILY) is dit.run_dit_lora_training
    assert spec.latent_shape is not None and spec.decode_latents is not None
    assert spec.save_takes_metadata and spec.training_mu is not None


def test_family_is_trainable_on_21_and_deploys_on_turbo():
    fam = detect_family(BASE)
    assert fam is not None and fam.name == FAMILY and fam.trainable
    assert fam.train_base_repos[0] == BASE
    # Turbo is distilled, so it is a deploy target, never a training base.
    assert TURBO not in fam.train_base_repos
    assert fam.deploy_base_for(BASE) == TURBO
    assert resolve_trainable_family(BASE, None) == FAMILY
    # The v1 20B family keeps its own trainer spec.
    assert detect_family("Qwen/Qwen-Image").name == "qwen-image"


def test_train_ui_pairs_both_training_ids_with_a_loadable_turbo_pipeline():
    from core.training.diffusion_train_common import family_train_infos

    info = next(i for i in family_train_infos() if i["name"] == FAMILY)
    assert info["default_base"] == BASE and info["label"] == "Qwen-Image-2.1"
    # The mirror id pairs with upstream Turbo: Turbo's unsloth mirror is its prequant artifact repo, not a pipeline pick.
    assert info["deploy_bases"] == {BASE: TURBO, "unsloth/Qwen-Image-2.1": TURBO}
    assert info["supports_samples"] and info["max_train_batch_size"] is None
    # info["note"] is blanked on a host without a training GPU, so read the spec the panel shows.
    from core.training.diffusion_train_common import _FAMILY_TRAIN_SPECS

    assert _FAMILY_TRAIN_SPECS[FAMILY]["note"] == "Trains on Qwen-Image-2.1, runs on 2.1 or Turbo."


def test_defaults_sit_on_the_32_grid_and_off_grid_resolutions_are_refused(tmp_path):
    d = train_defaults(FAMILY)
    assert d["resolution"] % 32 == 0 and d["lora_rank"] > 0 and d["learning_rate"] > 0
    base = dict(base_model = BASE, data_dir = str(tmp_path), output_dir = str(tmp_path / "o"))
    DiffusionLoraConfig(**base, resolution = 768).normalized()
    with pytest.raises(ValueError, match = "multiple of 32"):
        DiffusionLoraConfig(**base, resolution = 776).normalized()


def test_sampling_follows_the_pipeline_defaults():
    s = sample_inference_settings(FAMILY, BASE)
    # Qwen-Image-2.1 samples without CFG (true_cfg_scale 1.0); Studio runs 25 steps, Turbo 8.
    assert (s.steps, s.uses_cfg) == (25, False)
    assert sample_inference_settings(FAMILY, TURBO).steps == 8


def test_trained_adapters_are_offered_on_both_engines():
    from core.inference.diffusion_lora import lora_unavailable_reason

    for engine, kind in (("diffusers", "diffusers"), ("diffusers", "gguf"), ("sd_cpp", "gguf")):
        assert (
            lora_unavailable_reason(
                engine = engine, family = FAMILY, model_kind = kind, transformer_quant = None
            )
            is None
        ), engine
    for quant in ("int8", "fp8"):
        assert (
            lora_unavailable_reason(
                engine = "diffusers", family = FAMILY, model_kind = "diffusers", transformer_quant = quant
            )
            is None
        )
    # The token is exact to 2.1: the v1 20B family keeps its sd.cpp refusal.
    assert lora_unavailable_reason(
        engine = "sd_cpp", family = "qwen-image", model_kind = "gguf", transformer_quant = None
    )


# --- LoRA targets -------------------------------------------------------------------------------------------------


def test_targets_adapt_every_block_attention_and_mlp_linear_and_nothing_else():
    assert (
        _select_lora_targets(DEFAULT_LORA_TARGETS, _SPECS[FAMILY].lora_targets) == _QWEN21_TARGETS
    )
    t = _adapted()
    adapted = {n for n, m in t.named_modules() if hasattr(m, "lora_A")}
    expected = {
        f"transformer_blocks.{b}.{leaf}"
        for b in range(2)
        for leaf in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.proj",
            "img_mlp.gate_layer",
            "img_mlp.out",
        )
    }
    assert adapted == expected
    # Every Linear in a block is covered; the shared modulation / txt_in / img_in / proj_out are not.
    block_linears = {
        n
        for n, m in _tiny_transformer().named_modules()
        if isinstance(m, torch.nn.Linear) and n.startswith("transformer_blocks.")
    }
    assert block_linears == expected


# --- conditioning + forward ---------------------------------------------------------------------------------------


def test_latent_shape_and_pixels_follow_the_rgba_16x_vae():
    vae = type("V", (), {"config": type("Cf", (), {"z_dim": 64, "scale_factor_spatial": 16})()})()
    shape, seq = dit._qwen21_latent_shape(vae, 768, 512)
    assert shape == (1, 64, 1, 48, 32) and seq == 48 * 32
    px = torch.rand(2, 3, 32, 32) * 2 - 1
    rgba = dit._qwen21_pixels(px)
    assert rgba.shape == (2, 4, 1, 32, 32)
    assert torch.equal(rgba[:, :3, 0], px) and torch.all(rgba[:, 3] == 1.0)


def test_forward_returns_the_target_tokens_in_the_latent_layout():
    t = _tiny_transformer()
    noisy = torch.randn(2, C, 1, 4, 6)
    pes = [torch.randn(1, 7, CTX), torch.randn(1, 7, CTX)]
    with torch.no_grad():
        out = dit._qwen21_forward(
            t, noisy, torch.tensor([300.0, 800.0]), None, (pes,), None, "cpu", torch.float32
        )
    assert out.shape == noisy.shape and bool(torch.isfinite(out).all())


def test_mixed_caption_lengths_run_unpadded_and_match_one_sample_at_a_time():
    t = _tiny_transformer()
    noisy = torch.randn(3, C, 1, 4, 4)
    ts = torch.tensor([100.0, 500.0, 900.0])
    pes = [torch.randn(1, 5, CTX), torch.randn(1, 9, CTX), torch.randn(1, 5, CTX)]
    with torch.no_grad():
        batched = dit._qwen21_forward(t, noisy, ts, None, (pes,), None, "cpu", torch.float32)
        single = torch.cat(
            [
                dit._qwen21_forward(
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
    """Run QwenImage21Pipeline, capture what it hands the transformer, and check the trainer builds the same call."""
    Pipe, _ = _classes()
    pipe = Pipe.from_pretrained(tiny_pipe_dir, torch_dtype = torch.float32)
    calls = []
    real_forward = pipe.transformer.forward

    def spy(*args, **kwargs):
        out = real_forward(*args, **kwargs)
        calls.append((kwargs, out[0] if isinstance(out, tuple) else out.sample))
        return out

    pipe.transformer.forward = spy
    prompt = "a watercolor fox in the snow"
    pipe(
        prompt,
        height = 64,
        width = 64,
        num_inference_steps = 2,
        generator = torch.Generator().manual_seed(0),
        output_type = "latent",
        use_kv_cache = False,
    )
    pipe.transformer.forward = real_forward
    kwargs, out = calls[0]

    (embeds,) = dit._qwen21_encode_prompts(pipe, [prompt], "cpu")
    assert torch.equal(embeds[0], kwargs["encoder_hidden_states"])
    assert kwargs["encoder_hidden_states_mask"] is None

    (_one, (h, w)) = (None, kwargs["img_shapes"][0][-1][1:])
    packed = kwargs["hidden_states"]
    noisy = packed.transpose(1, 2).reshape(1, C, 1, h, w)
    timesteps = kwargs["timestep"] * 1000
    with torch.no_grad():
        mine = dit._qwen21_forward(
            pipe.transformer,
            noisy,
            timesteps,
            None,
            dit._qwen21_collate([embeds], "cpu", torch.float32),
            None,
            "cpu",
            torch.float32,
        )
    expected = out[:, -packed.shape[1] :].transpose(1, 2).reshape(1, C, 1, h, w)
    assert torch.allclose(mine, expected, atol = 1e-6)


@pytest.mark.allow_network
def test_latents_normalise_like_the_pipeline_and_decode_inverts_them(tiny_pipe_dir):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(tiny_pipe_dir, subfolder = "vae").float()
    px = torch.rand(1, 3, 32, 32) * 2 - 1
    mean_lat, std_lat = dit._qwen21_encode_latent_stats(vae, px)
    with torch.no_grad():
        raw = vae.encode(dit._qwen21_pixels(px)).latent_dist.mode()
    m = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    assert torch.allclose(mean_lat, (raw - m) / s, atol = 1e-6) and std_lat.shape == mean_lat.shape
    with torch.no_grad():
        img = dit._qwen21_decode_latents(vae, mean_lat)
    assert img.shape == (1, 3, 32, 32)


# --- sigma schedule -----------------------------------------------------------------------------------------------


def test_auto_shift_uses_the_pipelines_resolution_dependent_mu():
    _classes()
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    sched = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    cfg = type("Cfg", (), {"resolution": 768})()
    mu = dit._qwen21_training_mu(sched.config, cfg)
    assert mu == pytest.approx(calculate_shift(48 * 48, 256, 8192, 0.5, 0.9))
    table = _training_sigma_table(sched, "auto", mu)
    # Same mechanism set_timesteps applies at inference: exponential time shift at mu, then the terminal stretch.
    ref = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    ref.set_timesteps(sigmas = sched.sigmas.numpy().tolist(), mu = mu)
    assert torch.allclose(table, ref.sigmas[:-1].to(table.dtype), atol = 1e-6)
    assert float(table[-1]) == pytest.approx(0.02, abs = 1e-6)
    # A larger canvas shifts harder, as the pipeline does.
    big = dit._qwen21_training_mu(sched.config, type("Cfg", (), {"resolution": 1024})())
    assert big > mu
    # Without a family mu the pin stays max_shift (every other dynamic-shift family).
    pinned = _training_sigma_table(sched, "auto")
    assert not torch.allclose(pinned, table)


# --- training + save + reload -------------------------------------------------------------------------------------


def test_loss_falls_on_a_fixed_batch():
    t = _adapted(rank = 8)
    t.train()
    params = [p for p in t.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr = 1e-2)
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(2, C, 1, 4, 4, generator = g)
    noise = torch.randn(2, C, 1, 4, 4, generator = g)
    sig = torch.tensor([0.3, 0.7]).view(2, 1, 1, 1, 1)
    noisy = (1 - sig) * latents + sig * noise
    pes = [torch.randn(1, 6, CTX, generator = g), torch.randn(1, 6, CTX, generator = g)]
    losses = []
    for _ in range(25):
        opt.zero_grad()
        pred = dit._qwen21_forward(
            t, noisy, sig.flatten() * 1000, sig, (pes,), None, "cpu", torch.float32
        )
        loss = torch.nn.functional.mse_loss(pred, noise - latents)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < 0.8 * losses[0], losses


def test_saved_lora_has_diffusers_keys_and_reloads_at_the_trained_strength(tmp_path):
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    Pipe, _ = _classes()
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
    dit._qwen21_save(
        None,
        str(tmp_path),
        get_peft_model_state_dict(trained),
        adapter_metadata = dit._lora_adapter_metadata(trained),
    )
    path = tmp_path / DEFAULT_LORA_FILENAME
    keys = set(load_file(str(path)))
    assert keys and all(k.startswith("transformer.transformer_blocks.") for k in keys)
    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in keys
    assert "transformer.transformer_blocks.1.img_mlp.gate_layer.lora_B.weight" in keys

    fresh = _tiny_transformer()
    state_dict, metadata = Pipe.lora_state_dict(str(path), return_lora_metadata = True)
    Pipe.load_lora_into_transformer(
        state_dict, transformer = fresh, adapter_name = "style", metadata = metadata
    )
    noisy = torch.randn(1, C, 1, 4, 4)
    pes = [torch.randn(1, 5, CTX)]
    t = torch.tensor([500.0])
    with torch.no_grad():
        with_lora = dit._qwen21_forward(fresh, noisy, t, None, (pes,), None, "cpu", torch.float32)
        trained_out = dit._qwen21_forward(
            trained, noisy, t, None, (pes,), None, "cpu", torch.float32
        )
        fresh.disable_adapters()
        off = dit._qwen21_forward(fresh, noisy, t, None, (pes,), None, "cpu", torch.float32)
    assert torch.allclose(with_lora, trained_out, atol = 1e-5)
    assert not torch.allclose(with_lora, off)


@pytest.mark.allow_network
def test_end_to_end_run_trains_saves_and_the_pipeline_loads_the_adapter(
    tiny_pipe_dir, tmp_path, monkeypatch
):
    """The whole run_dit_lora_training loop on the tiny pipe: caption cache, latent cache, mixed caption lengths in a
    batch, sample rounds, save, catalog publish; then QwenImage21Pipeline.load_lora_weights changes the render."""
    from PIL import Image

    Pipe, _ = _classes()
    data = tmp_path / "data"
    data.mkdir()
    for i in range(4):
        img = Image.new("RGB", (64, 64), (200, 30 * i, 40))
        img.save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(
            "a red tile" + (" with a long trailing caption" if i % 2 else "")
        )
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    # CPU host: no nf4 / bf16 autocast, and unsloth_zoo refuses to import without an accelerator.
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: "bf16")
    monkeypatch.setattr(dit, "_assert_trusted_base_model", lambda *a, **k: None)
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
    assert sidecar["families"] == [FAMILY]

    pipe = Pipe.from_pretrained(tiny_pipe_dir, torch_dtype = torch.float32)

    def render():
        return pipe(
            "a red tile",
            height = 64,
            width = 64,
            num_inference_steps = 2,
            generator = torch.Generator().manual_seed(1),
            output_type = "latent",
        ).images

    base = render()
    pipe.load_lora_weights(str(out_dir), weight_name = DEFAULT_LORA_FILENAME, adapter_name = "trained")
    adapted = {n for n, m in pipe.transformer.named_modules() if hasattr(m, "lora_A")}
    assert len(adapted) == 14
    with_lora = render()
    assert not torch.allclose(base, with_lora)


# --- caption caching drops the text encoder lm_head ---------------------------------------------------------------


def _load_conditioners(pipe_dir):
    import types
    cfg = types.SimpleNamespace(base_model = pipe_dir, hf_token = None)
    return dit._qwen21_load_conditioners(cfg, "cpu", torch.bfloat16)


@pytest.fixture
def local_qwen21_conditioner_dir(tmp_path, monkeypatch):
    """A Qwen-Image-2.1 conditioner checkpoint (untied Qwen3-VL encoder, byte-level tokenizer, toy VAE, no
    transformer) built and saved locally, so the lm_head trim test needs no Hub access and can never skip for it."""
    pipe_cls, _ = _classes()
    try:
        from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import (
            Qwen2Tokenizer,
            Qwen2VLImageProcessor,
            Qwen3VLConfig,
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )
    except Exception as exc:  # noqa: BLE001 -- library too old for Qwen3-VL / Qwen-Image-2.1
        pytest.skip(f"Qwen3-VL / Qwen-Image-2.1 classes unavailable: {exc}")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.manual_seed(19437)
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    vocab = {t: i for i, t in enumerate(specials + sorted(ByteLevel.alphabet()))}
    processor = Qwen3VLProcessor(
        image_processor = Qwen2VLImageProcessor(),
        tokenizer = Qwen2Tokenizer(vocab = vocab, merges = [], additional_special_tokens = specials[1:]),
        video_processor = Qwen3VLVideoProcessor(),
        chat_template = "{% for message in messages %}{{ message.content[0].text }}{% endfor %}",
    )
    encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            tie_word_embeddings = False,
            image_token_id = vocab["<|image_pad|>"],
            video_token_id = vocab["<|video_pad|>"],
            vision_start_token_id = vocab["<|vision_start|>"],
            vision_end_token_id = vocab["<|vision_end|>"],
            text_config = {
                "vocab_size": len(vocab),
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                "tie_word_embeddings": False,
            },
            vision_config = {
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [],
            },
        )
    ).eval()
    vae = AutoencoderKLQwenImage21(
        base_dim = 8,
        decoder_base_dim = 8,
        dim_mult = [1, 1],
        z_dim = 4,
        num_res_blocks = 1,
        temperal_downsample = [False],
    )
    out = tmp_path / "qwen21_conditioners"
    pipe_cls(FlowMatchEulerDiscreteScheduler(), vae, encoder, processor, None).save_pretrained(
        str(out)
    )
    return str(out)


def test_caption_cache_drops_the_lm_head_and_keeps_the_embeds(
    local_qwen21_conditioner_dir, monkeypatch
):
    """The trainer's conditioner load drops the untied Qwen3-VL lm_head; the cached caption embeds are bit-identical
    to the full encoder's, and UNSLOTH_TE_KEEP_LM_HEAD=1 keeps the head. Runs on a locally built checkpoint."""
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV, is_trimmed

    tiny_pipe_dir = local_qwen21_conditioner_dir

    captions = [
        "a watercolor fox in the snow",
        "",
        "two red apples on a wooden table, studio light",
    ]
    monkeypatch.setenv(KEEP_LM_HEAD_ENV, "1")
    full, _ = _load_conditioners(tiny_pipe_dir)
    assert not is_trimmed(full.text_encoder)
    assert full.text_encoder.lm_head.weight.ndim == 2
    expected = dit._qwen21_encode_prompts(full, captions, "cpu")
    del full

    monkeypatch.delenv(KEEP_LM_HEAD_ENV)
    pipe, _ = _load_conditioners(tiny_pipe_dir)
    assert is_trimmed(pipe.text_encoder)
    assert not any("lm_head" in name for name, _ in pipe.text_encoder.named_parameters())
    got = dit._qwen21_encode_prompts(pipe, captions, "cpu")
    assert len(got) == len(expected)
    for (a,), (b,) in zip(got, expected):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_caption_encoder_trim_failure_keeps_the_encoder(monkeypatch):
    import types

    import core.inference.diffusion_text_encoder_trim as trim

    def boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(trim, "trim_text_encoder", boom)
    head = torch.nn.Linear(4, 8, bias = False)
    pipe = types.SimpleNamespace(text_encoder = types.SimpleNamespace(lm_head = head))
    dit._trim_caption_encoder(pipe, FAMILY)
    assert pipe.text_encoder.lm_head is head
