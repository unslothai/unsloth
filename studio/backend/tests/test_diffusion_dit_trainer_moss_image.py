# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for MOSS-Image LoRA training (the DiT trainer's moss-image spec). A random in-tree MOSSImageTransformer
covers the offline checks; a tiny MOSS-Image pipeline folder (random Qwen3.5 encoder, word-level tokenizer with a chat
template, small FLUX.2 VAE) proves the trainer conditions and denoises exactly as Studio's MOSSImagePipeline does, that
its adapter has the published MOSS LoRA layout, and that a trained adapter loads back and changes a render."""

from __future__ import annotations

import json
import math
import re
import sys

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")
if not hasattr(diffusers, "AutoencoderKLFlux2"):
    pytest.skip("diffusers without AutoencoderKLFlux2", allow_module_level = True)

from core.inference import diffusion_lora_files as lora_files  # noqa: E402
from core.inference import diffusion_moss_image as moss  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.training import diffusion_dit_trainer as dit  # noqa: E402
from core.training.diffusion_samples import (  # noqa: E402
    SAMPLE_FAMILIES,
    flow_sigmas,
    sample_inference_settings,
)
from core.training.diffusion_train_common import (  # noqa: E402
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    _DIT_TRAIN_FAMILIES,
    _FORCE_BF16_FAMILIES,
    enable_diffusion_gradient_checkpointing,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)

FAMILY = "moss-image"
SFT = "OpenMOSS-Team/MOSS-Image-SFT"
BASE = "OpenMOSS-Team/MOSS-Image-Base"
# head_dim 32 (a multiple of 8, as the RoPE needs); text_dim is the tiny encoder's hidden size.
_TINY = dict(dim = 64, ffn_dim = 128, num_heads = 2, num_layers = 2, text_dim = 64)
# Published MOSS LoRAs: axiomofmind/Moss-image-beauty-lora's adapter_config target_modules, and the 252 module stems of
# Angais/MOSS-Image-PixelArt-16x16 (layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}, no prefix).
_PUBLISHED_TARGETS = re.compile(r".*layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)")
_PUBLISHED_PROJS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def _spec():
    assert FAMILY in dit._SPECS, "moss-image has no DiT trainer spec"
    return dit._SPECS[FAMILY]


def _tiny_transformer(seed = 0):
    torch.manual_seed(seed)
    t = moss.MOSSImageTransformer(**_TINY)
    with torch.no_grad():
        for p in t.parameters():
            p.normal_(0.0, 0.02)
    return t.eval()


def _adapted(rank = 4, seed = 0):
    from peft import LoraConfig

    t = _tiny_transformer(seed)
    t.requires_grad_(False)
    t.add_adapter(
        LoraConfig(
            r = rank,
            lora_alpha = rank,
            init_lora_weights = "gaussian",
            target_modules = list(_spec().lora_targets),
        )
    )
    return t


def _adapted_names(model) -> set[str]:
    return {n for n, m in model.named_modules() if hasattr(m, "lora_A")}


def _tiny_vae():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLFlux2(
        block_out_channels = [32, 32, 32, 32],
        decoder_block_out_channels = [16, 16, 16, 16],
        norm_num_groups = 8,
        layers_per_block = 1,
        latent_channels = 32,
    ).eval()


def _build_tiny_pipe(root):
    """A diffusers-layout MOSS-Image folder the in-tree loader reads offline."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5Model"):
        pytest.skip("transformers without Qwen3.5 (needs 5.3+)")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5Model

    words = [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "user",
        "assistant",
    ] + "a red tile trtcrd with long trailing caption".split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object = tok, unk_token = "<unk>", pad_token = "<unk>")
    fast.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    fast.save_pretrained(str(root / "tokenizer"))
    text = dict(
        vocab_size = len(vocab),
        hidden_size = _TINY["text_dim"],
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        linear_num_value_heads = 4,
        linear_num_key_heads = 2,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        full_attention_interval = 4,
        layer_types = ["linear_attention"] * 3 + ["full_attention"],
    )
    vision = dict(depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64)
    torch.manual_seed(0)
    Qwen3_5Model(Qwen3_5Config(text_config = text, vision_config = vision)).to(
        torch.bfloat16
    ).save_pretrained(str(root / "text_encoder"))
    _tiny_transformer().to(torch.bfloat16).save_pretrained(str(root / "transformer"))
    _tiny_vae().save_pretrained(str(root / "vae"))
    moss.MOSSImageScheduler().save_pretrained(str(root / "scheduler"))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MOSSImagePipeline",
                "_diffusers_version": "0.38.0",
                "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
                "text_encoder": ["transformers", "Qwen3_5Model"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        )
    )
    return str(root)


@pytest.fixture(scope = "module")
def tiny_pipe_dir(tmp_path_factory):
    return _build_tiny_pipe(tmp_path_factory.mktemp("tiny-moss-image"))


class _Cfg:
    def __init__(self, base_model):
        self.base_model = base_model
        self.hf_token = None


# ---------------------------------------------------------------- registry


def test_both_releases_train_through_the_dit_trainer():
    fam = detect_family(SFT)
    assert fam is not None and fam.name == FAMILY
    assert fam.trainable
    assert set(fam.train_base_repos) == {SFT, BASE}
    for repo in (SFT, BASE):
        assert resolve_trainable_family(repo) == FAMILY
    assert FAMILY in _DIT_TRAIN_FAMILIES and FAMILY in _FORCE_BF16_FAMILIES
    assert get_trainer(FAMILY) is dit.run_dit_lora_training
    spec = _spec()
    assert spec.force_bf16 and spec.decode_latents is not None
    assert FAMILY in SAMPLE_FAMILIES


def test_defaults_sit_on_the_16_grid_and_fp16_and_off_grid_are_refused(tmp_path):
    d = train_defaults(FAMILY)
    assert d["resolution"] % 16 == 0 and d["lora_rank"] == 32
    base = dict(base_model = SFT, data_dir = str(tmp_path), output_dir = str(tmp_path / "o"))
    assert (
        DiffusionLoraConfig(**base, resolution = d["resolution"]).normalized().resolved_family
        == FAMILY
    )
    with pytest.raises(ValueError, match = "multiple of 16"):
        DiffusionLoraConfig(**base, resolution = 760).normalized()
    with pytest.raises(ValueError, match = "requires bf16"):
        DiffusionLoraConfig(**base, mixed_precision = "fp16").normalized()


# ---------------------------------------------------------------- LoRA layout


def test_targets_are_the_published_moss_lora_modules_and_nothing_else():
    spec = _spec()
    assert dit._select_lora_targets(DEFAULT_LORA_TARGETS, spec.lora_targets) == spec.lora_targets
    names = _adapted_names(_adapted())
    want = {f"layers.{i}.proj.{p}" for i in range(_TINY["num_layers"]) for p in _PUBLISHED_PROJS}
    assert names == want
    assert all(_PUBLISHED_TARGETS.fullmatch(n) for n in names)
    # The text refiner's qkv / proj / ffn_* share suffixes with nothing targeted.
    assert not any(n.startswith("text_embedder") for n in names)


def _pipe(transformer):
    return moss.MOSSImagePipeline(transformer, moss.MOSSImageScheduler(), vae = _tiny_vae())


def _published_layout_file(path, rank = 4):
    """A LoRA in Angais/MOSS-Image-PixelArt-16x16's layout: bare ``layers.N.proj.*`` keys, bf16."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(7)
    state = {}
    for name, mod in _tiny_transformer().named_modules():
        if _PUBLISHED_TARGETS.fullmatch(name):
            state[f"{name}.lora_A.weight"] = torch.randn(rank, mod.in_features, generator = g).to(
                torch.bfloat16
            )
            state[f"{name}.lora_B.weight"] = torch.randn(mod.out_features, rank, generator = g).to(
                torch.bfloat16
            )
    save_file(state, str(path), metadata = {"rank": str(rank), "alpha": str(rank)})
    return path


def test_trained_and_published_adapters_land_on_the_same_modules(tmp_path):
    """Interchangeable: Studio's MOSS LoRA loader puts a published adapter and a trained one on the same layers."""
    from peft.utils import get_peft_model_state_dict

    published = _pipe(_tiny_transformer())
    lora_files.load_adapter(
        published, str(_published_layout_file(tmp_path / "pix.safetensors")), "pix", FAMILY
    )

    trained = _adapted()
    out = tmp_path / "trained"
    _spec().save(
        None,
        str(out),
        get_peft_model_state_dict(trained),
        adapter_metadata = dit._lora_adapter_metadata(trained),
    )
    ours = _pipe(_tiny_transformer())
    lora_files.load_adapter(ours, str(out / DEFAULT_LORA_FILENAME), "mine", FAMILY)
    assert (
        _adapted_names(published.transformer)
        == _adapted_names(ours.transformer)
        == _adapted_names(trained)
    )


def test_saved_adapter_reloads_at_the_trained_strength(tmp_path):
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    trained = _adapted(rank = 4)
    # alpha != rank: the file's metadata must carry the scale, or the loader assumes alpha == rank.
    trained.peft_config["default"].lora_alpha = 8
    for m in trained.modules():
        if "default" in (getattr(m, "scaling", None) or {}):
            m.scaling["default"] = 8 / 4
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.5)
    _spec().save(
        None,
        str(tmp_path),
        get_peft_model_state_dict(trained),
        adapter_metadata = dit._lora_adapter_metadata(trained),
    )
    path = tmp_path / DEFAULT_LORA_FILENAME
    keys = set(load_file(str(path)))
    assert keys and all(k.startswith("transformer.layers.") for k in keys)
    assert "transformer.layers.1.proj.ffn_up_gate.lora_B.weight" in keys

    pipe = _pipe(_tiny_transformer())
    lora_files.load_adapter(pipe, str(path), "style", FAMILY)
    x = torch.randn(1, 128, 4, 6)
    emb = torch.randn(1, 9, _TINY["text_dim"])
    t = torch.tensor([500.0])
    with torch.no_grad():
        with_lora = dit._moss_forward(
            pipe.transformer, x, t, None, (emb, None), None, "cpu", torch.float32
        )
        want = dit._moss_forward(trained, x, t, None, (emb, None), None, "cpu", torch.float32)
        pipe.transformer.disable_adapters()
        off = dit._moss_forward(
            pipe.transformer, x, t, None, (emb, None), None, "cpu", torch.float32
        )
    assert torch.allclose(with_lora, want, atol = 1e-5)
    assert not torch.allclose(with_lora, off)


# ---------------------------------------------------------------- transformer support the trainer needs


def test_gradient_checkpointing_is_supported_and_exact(monkeypatch):
    # The fallback path (diffusers' enable_gradient_checkpointing), as on a host without unsloth_zoo.
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_checkpointing", None)
    plain, checked = _adapted(seed = 3), _adapted(seed = 3)
    checked.load_state_dict(plain.state_dict())
    enable_diffusion_gradient_checkpointing(checked)
    assert checked.gradient_checkpointing
    calls = []
    real = checked._gradient_checkpointing_func
    monkeypatch.setattr(
        checked, "_gradient_checkpointing_func", lambda *a: calls.append(1) or real(*a)
    )
    x = torch.randn(1, 128, 4, 4)
    emb = torch.randn(1, 7, _TINY["text_dim"])
    grads = []
    for model in (plain, checked):
        model.train()
        out = model(x, torch.tensor([300.0]), encoder_hidden_states = emb, return_dict = False)[0]
        out.square().mean().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    assert len(calls) == _TINY["num_layers"]
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol = 1e-6), n
    # Inference runs under no_grad, where the flag changes nothing.
    calls.clear()
    with torch.no_grad():
        checked(x, torch.tensor([300.0]), encoder_hidden_states = emb)
    assert not calls


def test_compute_dtype_reads_through_a_peft_wrapper():
    """bitsandbytes nf4 stores uint8 weights and declares compute_dtype; a LoRA wrapper must not hide it, or the block
    casts its activations to the storage dtype."""
    from peft import LoraConfig, inject_adapter_in_model

    class Stored(torch.nn.Linear):
        compute_dtype = torch.bfloat16

    holder = torch.nn.Module()
    holder.q = Stored(8, 8).to(torch.float16)
    inject_adapter_in_model(LoraConfig(r = 2, target_modules = ["q"]), holder)
    assert hasattr(holder.q, "lora_A")
    assert moss._compute_dtype(holder.q) == torch.bfloat16


# ---------------------------------------------------------------- conditioning, forward, latents, schedule


def test_forward_is_the_pipelines_transformer_call_and_mixed_lengths_run_unpadded():
    t = _tiny_transformer()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 128, 4, 6, generator = g)
    rows = [
        torch.randn(5, _TINY["text_dim"], generator = g),
        torch.randn(11, _TINY["text_dim"], generator = g),
    ]
    ts = torch.tensor([200.0, 800.0])
    batch = dit._moss_collate([(r,) for r in rows], "cpu", torch.float32)
    assert batch[1] is not None and batch[1].sum(1).tolist() == [5, 11]
    with torch.no_grad():
        got = dit._moss_forward(t, x, ts, None, batch, None, "cpu", torch.float32)
        for i, row in enumerate(rows):
            # MOSSImagePipeline: transformer(latents, timestep, encoder_hidden_states=..., return_dict=False)[0]
            want = t(x[i : i + 1], ts[i], encoder_hidden_states = row[None], return_dict = False)[0]
            assert torch.allclose(got[i : i + 1], want, atol = 1e-6)
    same = dit._moss_collate([(rows[0],), (rows[0],)], "cpu", torch.float32)
    assert same[1] is None and same[0].shape == (2, 5, _TINY["text_dim"])


def test_latents_normalise_as_the_pipeline_decodes_them():
    vae = _tiny_vae()
    pipe = _pipe(_tiny_transformer())
    pipe.vae = vae
    x = torch.rand(1, 3, 64, 96) * 2 - 1
    spec = _spec()
    lat = spec.encode_latents(vae, x)
    assert lat.shape == (1, 128, 4, 6)
    assert spec.latent_shape(vae, 64, 96)[0] == (1, 128, 4, 6)
    with torch.no_grad():
        mine = spec.decode_latents(vae, lat).clamp(-1, 1)
        theirs = pipe.vae_decode(lat)
        direct = vae.decode(vae.encode(x).latent_dist.mode()).sample.clamp(-1, 1)
    assert torch.allclose(mine, theirs, atol = 1e-5)
    assert torch.allclose(theirs, direct, atol = 1e-4)


def test_training_noise_is_shift_5_and_previews_take_the_sgm_uniform_steps(tiny_pipe_dir):
    spec = _spec()
    sched = spec.training_scheduler(_Cfg(tiny_pipe_dir))
    table = dit._training_sigma_table(sched, None)
    u = torch.linspace(1.0, 1.0 / 1000, 1000)
    assert torch.allclose(table, 5 * u / (1 + 4 * u), atol = 1e-6)
    # The transformer's timestep is sigma * 1000, the value the pipeline's scheduler hands it.
    assert torch.allclose(sched.timesteps, table * 1000, atol = 1e-3)
    ref = moss.MOSSImageScheduler(sigma_schedule = "sgm_uniform")
    ref.set_timesteps(25)
    sigmas, num_train = flow_sigmas(sched.config, FAMILY, 25, 48 * 48)
    assert num_train == 1000.0
    assert torch.equal(sigmas, ref.sigmas.to(torch.float32))
    s = sample_inference_settings(FAMILY, SFT)
    assert (s.steps, s.guidance, s.cfg_mode) == (25, 2.5, "uncond")


def test_conditioning_loads_without_the_dit_and_encodes_like_the_pipeline(tiny_pipe_dir):
    spec = _spec()
    pipe, vae = spec.load_conditioners(_Cfg(tiny_pipe_dir), "cpu", torch.bfloat16)
    assert pipe.transformer is None and vae is pipe.vae and vae.dtype == torch.float32
    caps = ["trtcrd, a red tile", "a red tile with a long trailing caption"]
    got = spec.encode_prompts(pipe, caps, "cpu")
    full = moss.load_moss_image_pipeline(tiny_pipe_dir, torch.bfloat16)
    assert full.transformer is not None
    for cap, (emb,) in zip(caps, got):
        want, mask = full.encode_prompt(cap, device = "cpu")
        assert bool(mask.all()) and torch.equal(emb, want[0])
    assert got[0][0].shape[0] != got[1][0].shape[0]


def test_loss_falls_on_a_fixed_batch():
    t = _adapted(rank = 8)
    # The 0.02 init closes the AdaLN gates and the output norm, so no update reaches the prediction; open them.
    with torch.no_grad():
        t.shared_block_modulation.bias.fill_(1.0)
        t.final_norm_video.weight.fill_(1.0)
        t.final_linear_video.weight.normal_(0.0, 0.15)
    t.train()
    params = [p for p in t.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2)
    g = torch.Generator().manual_seed(0)
    latents = torch.randn(2, 128, 4, 4, generator = g)
    noise = torch.randn(2, 128, 4, 4, generator = g)
    sig = torch.tensor([0.3, 0.7]).view(2, 1, 1, 1)
    noisy = (1 - sig) * latents + sig * noise
    batch = dit._moss_collate(
        [
            (torch.randn(6, _TINY["text_dim"], generator = g),),
            (torch.randn(9, _TINY["text_dim"], generator = g),),
        ],
        "cpu",
        torch.float32,
    )
    losses = []
    for _ in range(25):
        opt.zero_grad()
        pred = dit._moss_forward(
            t, noisy, sig.flatten() * 1000, sig, batch, None, "cpu", torch.float32
        )
        loss = torch.nn.functional.mse_loss(pred, noise - latents)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < 0.8 * losses[0], losses


# ---------------------------------------------------------------- the whole run


def _render(pipe, prompt):
    pipe.set_progress_bar_config(disable = True)
    return pipe(
        prompt,
        height = 64,
        width = 64,
        num_inference_steps = 3,
        guidance_scale = 2.5,
        generator = torch.Generator("cpu").manual_seed(0),
        output_type = "pt",
    ).images


def test_end_to_end_run_trains_saves_and_the_adapter_changes_a_render(
    tiny_pipe_dir, tmp_path, monkeypatch
):
    """run_dit_lora_training on the tiny pipe: caption + latent cache, mixed caption lengths in a batch, gradient
    checkpointing, a CFG preview round, save, catalog publish; then Studio's MOSS LoRA loader applies the adapter to a
    MOSSImagePipeline and the same-seed render changes."""
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 64), (200, 30 * i, 40)).save(data / f"img{i}.png")
        (data / f"img{i}.txt").write_text(
            "trtcrd, a red tile" + (" with a long trailing caption" if i % 2 else "")
        )
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    # CPU host: no nf4 / bf16 autocast, and unsloth_zoo refuses to import without an accelerator.
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda cfg, spec, device: "bf16")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    spec = _spec()
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
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, errors
    progress = [e for e in events if e.get("type") == "progress"]
    assert len(progress) == 4 and all(math.isfinite(e["loss"]) for e in progress)
    done = [e for e in events if e.get("type") == "complete"][-1]
    assert done["lora_path"] and not done["stopped"]
    assert any(e.get("type") == "sample" for e in events)
    sidecar = json.loads(next(catalog.glob("*.json")).read_text())
    assert sidecar["families"] == [FAMILY]

    pipe = moss.load_moss_image_pipeline(tiny_pipe_dir, torch.float32)
    base = _render(pipe, "trtcrd, a red tile")
    lora_files.load_adapter(pipe, str(out_dir / DEFAULT_LORA_FILENAME), "trained", FAMILY)
    assert len(_adapted_names(pipe.transformer)) == 6 * _TINY["num_layers"]
    with_lora = _render(pipe, "trtcrd, a red tile")
    assert not torch.allclose(base, with_lora)
