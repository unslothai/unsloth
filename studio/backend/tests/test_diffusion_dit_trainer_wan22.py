# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for the Wan2.2-TI2V-5B DiT LoRA trainer family (fakes plus a tiny random WanTransformer3DModel)."""

from __future__ import annotations

import json
import math
import types

import pytest
import torch

from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_dit_trainer import (
    _SPECS,
    _WAN_MAX_SEQUENCE_LENGTH,
    _WAN_TARGETS,
    _select_lora_targets,
    _training_sigma_table,
)
from core.training.diffusion_train_common import (
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    TRAINABLE_VIDEO_FAMILIES,
    _DIT_TRAIN_FAMILIES,
    _FORCE_BF16_FAMILIES,
    _assert_trusted_base_model,
    _family_vram_note,
    _publish_to_lora_catalog,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)

FAMILY = "wan2.2-ti2v-5b"
BASE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Wan-AI/Wan2.2-TI2V-5B-Diffusers vae/config.json: 48 channels (latents_mean / latents_std are 48 long).
Z_DIM = 48
# Wan-AI/Wan2.2-TI2V-5B-Diffusers scheduler/scheduler_config.json, the keys the trainer reads.
UNIPC_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "flow_shift": 5.0,
    "num_train_timesteps": 1000,
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "solver_order": 2,
}


def _wan_classes():
    try:
        from diffusers import WanPipeline, WanTransformer3DModel
    except Exception as exc:  # noqa: BLE001 -- diffusers too old, or a broken optional dep
        pytest.skip(f"diffusers Wan classes unavailable: {exc}")
    return WanPipeline, WanTransformer3DModel


def _tiny_transformer():
    """A random 2-block WanTransformer3DModel with TI2V-5B's layout."""
    _, WanTransformer3DModel = _wan_classes()
    torch.manual_seed(0)
    return WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 12,
        in_channels = Z_DIM,
        out_channels = Z_DIM,
        text_dim = 32,
        freq_dim = 256,
        ffn_dim = 32,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        rope_max_seq_len = 32,
    ).eval()


def test_wan_spec_is_registered_and_bf16_only():
    spec = _SPECS[FAMILY]
    assert spec.family == FAMILY
    assert spec.force_bf16 is True
    assert FAMILY in _FORCE_BF16_FAMILIES
    assert 8.0 < spec.dense_bf16_gb < 12.0
    assert spec.lora_targets == _WAN_TARGETS
    # Wan ships UniPC, so the spec must build its own FlowMatchEuler table.
    assert spec.training_scheduler is dit._wan_training_scheduler
    assert spec.decode_latents is None and spec.latent_shape is None


def test_wan_is_a_trainable_video_family_with_a_trainer():
    assert FAMILY in TRAINABLE_VIDEO_FAMILIES
    assert FAMILY in _DIT_TRAIN_FAMILIES
    assert get_trainer(FAMILY) is dit.run_dit_lora_training
    for name in TRAINABLE_VIDEO_FAMILIES:
        assert callable(get_trainer(name)), name


@pytest.mark.parametrize(
    "base", [BASE, "wan-ai/wan2.2-ti2v-5b-diffusers", "/data/models/Wan2.2-TI2V-5B-Diffusers"]
)
def test_wan_bases_route_to_the_dit_trainer(base):
    assert resolve_trainable_family(base) == FAMILY


def test_explicit_model_family_override_routes_a_local_checkout():
    assert resolve_trainable_family("/tmp/some-local-checkout", FAMILY) == FAMILY


def test_the_a14b_moe_is_still_refused_and_names_the_5b():
    # Step 2: two experts need two adapters, which this spec does not train.
    with pytest.raises(ValueError) as exc:
        resolve_trainable_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert "wan2.2-t2v-a14b" in str(exc.value)
    assert FAMILY in str(exc.value)


def test_wan_official_base_passes_the_trusted_base_gate():
    _assert_trusted_base_model(BASE)
    with pytest.raises(ValueError, match = "untrusted"):
        _assert_trusted_base_model("random-user/wan2.2-ti2v-5b-finetune")


def test_wan_defaults_sit_on_the_32_grid():
    d = train_defaults(FAMILY)
    assert d["resolution"] % 32 == 0  # 16x VAE x 2x patch
    assert d["lora_rank"] >= 16 and 0 < d["learning_rate"] <= 2e-4
    assert d["lr_scheduler"] == "constant_with_warmup" and d["lr_warmup_steps"] > 0


def test_wan_has_a_vram_row():
    note = _family_vram_note(FAMILY)
    assert "5B" in note and "QLoRA" in note


def _cfg(tmp_path, **kw):
    data = tmp_path / "data"
    data.mkdir(exist_ok = True)
    return DiffusionLoraConfig(
        base_model = BASE, data_dir = str(data), output_dir = str(tmp_path / "run"), **kw
    )


@pytest.mark.parametrize("resolution, ok", [(704, True), (512, True), (520, False), (696, False)])
def test_wan_resolution_must_sit_on_the_vae_grid(tmp_path, resolution, ok):
    if ok:
        assert _cfg(tmp_path, resolution = resolution).normalized().resolved_family == FAMILY
    else:
        with pytest.raises(ValueError, match = "multiple of 32"):
            _cfg(tmp_path, resolution = resolution).normalized()


def test_wan_rejects_fp16_before_loading(tmp_path):
    with pytest.raises(ValueError, match = "bf16"):
        _cfg(tmp_path, resolution = 704, mixed_precision = "fp16").normalized()


def test_wan_default_flow_shift_stays_the_models_own_schedule(tmp_path):
    cfg = _cfg(tmp_path, resolution = 704).normalized()
    assert cfg.flow_shift is None


def test_a_video_run_publishes_no_adapter_into_the_image_lora_catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "loras" / "diffusion"
    catalog.mkdir(parents = True)
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: catalog)
    adapter = tmp_path / "run" / DEFAULT_LORA_FILENAME
    adapter.parent.mkdir(parents = True, exist_ok = True)
    adapter.write_bytes(b"adapter-bytes")
    cfg = _cfg(tmp_path, resolution = 704).normalized()
    assert _publish_to_lora_catalog(str(adapter), cfg) is None
    assert list(catalog.iterdir()) == []


def test_family_train_infos_offers_wan(dit_train_host):
    from core.training.diffusion_train_common import family_train_infos

    infos = {i["name"]: i for i in family_train_infos()}
    assert FAMILY in infos, "a trainable video family must be offered by /diffusion/info"
    info = infos[FAMILY]
    assert info["default_base"] == BASE
    assert info["base_repos"] == [BASE]
    assert info["label"] == "Wan2.2 TI2V 5B"
    assert info["defaults"]["resolution"] % 32 == 0
    assert info["deploy_base"] is None
    assert "nf4" in info["precision_modes"]
    assert info["supports_compile"] is True
    assert "ltx-2" in infos and "sdxl" in infos


def test_generic_default_targets_resolve_to_the_wan_set():
    assert _select_lora_targets(DEFAULT_LORA_TARGETS, _WAN_TARGETS) == _WAN_TARGETS
    explicit = ("to_q", "to_v")
    assert _select_lora_targets(explicit, _WAN_TARGETS) == explicit


@pytest.fixture
def unipc_base(tmp_path):
    root = tmp_path / "wan_base"
    (root / "scheduler").mkdir(parents = True)
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(UNIPC_CONFIG))
    return types.SimpleNamespace(base_model = str(root), hf_token = None)


def test_training_scheduler_bakes_in_the_inference_flow_shift(unipc_base):
    pytest.importorskip("diffusers")
    sched = dit._wan_training_scheduler(unipc_base)
    n = 1000
    u = torch.linspace(1, n, n).flip(0) / n
    expected = 5.0 * u / (1.0 + 4.0 * u)
    assert sched.config.shift == 5.0
    assert torch.allclose(sched.sigmas[:n].float(), expected, atol = 1e-6)
    assert torch.allclose(sched.timesteps.float(), expected * n, atol = 1e-3)
    # None (the default) trains on exactly this table; a number re-shifts from the unshifted u.
    assert _training_sigma_table(sched, None) is sched.sigmas
    assert torch.allclose(_training_sigma_table(sched, 1.0).float(), u, atol = 1e-6)


def test_the_generic_scheduler_load_would_drop_the_shift(unipc_base):
    """The trap the spec hook exists for: FlowMatchEuler.from_pretrained on Wan's UniPC config ignores flow_shift and
    yields the identity (shift 1) table, so a run trains mostly at low noise."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    generic = FlowMatchEulerDiscreteScheduler.from_pretrained(
        unipc_base.base_model, subfolder = "scheduler"
    )
    ours = dit._wan_training_scheduler(unipc_base)
    assert generic.config.shift == 1.0
    mid = 500
    assert ours.sigmas[mid] > generic.sigmas[mid] + 0.2


class _FakeWanPipe:
    """encode_prompt returns what WanPipeline._get_t5_prompt_embeds does: real rows then zeros to max_sequence_length."""

    def __init__(self):
        self.text_encoder = None
        self.calls = []

    def encode_prompt(
        self,
        prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt,
        max_sequence_length,
        device,
    ):
        self.calls.append((prompt, max_sequence_length, do_classifier_free_guidance))
        n = len(prompt.split()) + 1
        g = torch.Generator().manual_seed(n)
        emb = torch.zeros(1, max_sequence_length, 8)
        emb[:, :n] = torch.randn(1, n, 8, generator = g) + 0.1
        return emb.to(torch.bfloat16), None


def test_encode_prompts_caches_trimmed_embeds_and_collate_restores_the_pipeline_tensor():
    pipe = _FakeWanPipe()
    caps = ["a cat", "a much longer caption about a yarn dog"]
    cached = dit._wan_encode_prompts(pipe, caps, "cpu")
    assert [c[1] for c in pipe.calls] == [_WAN_MAX_SEQUENCE_LENGTH] * 2
    assert all(c[2] is False for c in pipe.calls)
    assert [e[0].shape[1] for e in cached] == [3, 9]  # trailing zero rows dropped
    (batch,) = dit._wan_collate(cached, "cpu", torch.bfloat16)
    assert batch.shape == (2, _WAN_MAX_SEQUENCE_LENGTH, 8)
    for i, cap in enumerate(caps):
        full, _ = _FakeWanPipe().encode_prompt(cap, False, 1, _WAN_MAX_SEQUENCE_LENGTH, "cpu")
        assert torch.equal(
            batch[i : i + 1], full
        ), "collate must rebuild the exact zero-padded tensor"


def test_trim_keeps_one_row_for_an_all_zero_embed():
    out = dit._wan_trim_padding(torch.zeros(1, 16, 4))
    assert out.shape == (1, 1, 4)


class _FakeDist:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def sample(self):
        return self.mean


class _FakeWanVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(
            z_dim = Z_DIM,
            latents_mean = [0.1 * i for i in range(Z_DIM)],
            latents_std = [1.0 + 0.05 * i for i in range(Z_DIM)],
        )
        self.seen = None

    def encode(self, px):
        self.seen = px
        b, _c, f, h, w = px.shape
        mean = torch.full((b, Z_DIM, f, h // 16, w // 16), 2.0)
        std = torch.full_like(mean, 0.5)
        return types.SimpleNamespace(latent_dist = _FakeDist(mean, std))


def test_encode_latents_adds_a_frame_axis_and_normalises_like_the_pipeline():
    vae = _FakeWanVae()
    px = torch.zeros(2, 3, 64, 96)
    lat = dit._wan_encode_latents(vae, px)
    assert vae.seen.shape == (2, 3, 1, 64, 96)
    assert lat.shape == (2, Z_DIM, 1, 4, 6)
    # WanPipeline decodes with latents / (1 / std) + mean; that must invert the training normalisation.
    mean = torch.tensor(vae.config.latents_mean).view(1, Z_DIM, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, Z_DIM, 1, 1, 1)
    assert torch.allclose(lat / inv_std + mean, torch.full_like(lat, 2.0), atol = 1e-5)


def test_encode_latent_stats_returns_the_posterior_affine_pair():
    vae = _FakeWanVae()
    a, b = dit._wan_encode_latent_stats(vae, torch.zeros(1, 3, 32, 32))
    std = torch.tensor(vae.config.latents_std).view(1, Z_DIM, 1, 1, 1)
    assert torch.allclose(b, 0.5 / std.expand_as(b))
    assert torch.allclose(a, dit._wan_encode_latents(vae, torch.zeros(1, 3, 32, 32)))


def test_forward_matches_the_pipelines_expand_timesteps_call():
    transformer = _tiny_transformer()
    noisy = torch.randn(2, Z_DIM, 1, 8, 12)
    t = torch.tensor([937.5, 120.0])
    emb = torch.randn(2, 16, 32)
    with torch.no_grad():
        ours = dit._wan_forward(transformer, noisy, t, None, (emb,), None, "cpu", torch.float32)
        mask = torch.ones(1, 1, 8, 12)
        per_sample = []
        for i in range(2):
            ts = (mask[0][:, ::2, ::2] * t[i]).flatten().unsqueeze(0)
            per_sample.append(
                transformer(
                    hidden_states = noisy[i : i + 1],
                    timestep = ts,
                    encoder_hidden_states = emb[i : i + 1],
                    return_dict = False,
                )[0]
            )
    assert ours.shape == noisy.shape  # aligned with target = noise - latents
    assert torch.allclose(ours, torch.cat(per_sample), atol = 1e-6)


def _adapted_tiny():
    from peft import LoraConfig

    transformer = _tiny_transformer()
    transformer.add_adapter(
        LoraConfig(
            r = 4, lora_alpha = 4, init_lora_weights = "gaussian", target_modules = list(_WAN_TARGETS)
        )
    )
    return transformer


def test_targets_adapt_every_block_linear_and_nothing_else():
    transformer = _adapted_tiny()
    adapted = {n for n, m in transformer.named_modules() if hasattr(m, "lora_A")}
    block_linears = {
        f"blocks.{i}.{n}"
        for i in range(2)
        for n in (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        )
    }
    assert adapted == block_linears
    assert not any(
        n.startswith(("condition_embedder", "proj_out", "patch_embedding")) for n in adapted
    )


def test_saved_lora_has_diffusers_keys_and_loads_through_wanpipeline(tmp_path):
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    WanPipeline, _ = _wan_classes()
    trained = _adapted_tiny()
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.5)
    dit._wan_save(None, str(tmp_path), get_peft_model_state_dict(trained))
    path = tmp_path / DEFAULT_LORA_FILENAME
    keys = set(load_file(str(path)))
    assert keys and all(k.startswith("transformer.blocks.") for k in keys)
    assert "transformer.blocks.0.attn1.to_q.lora_A.weight" in keys
    assert "transformer.blocks.1.ffn.net.2.lora_B.weight" in keys

    fresh = _tiny_transformer()
    pipe = WanPipeline(
        tokenizer = None,
        text_encoder = None,
        vae = None,
        scheduler = None,
        transformer = fresh,
        expand_timesteps = True,
    )
    noisy = torch.randn(1, Z_DIM, 1, 8, 8)
    emb = torch.randn(1, 16, 32)
    t = torch.tensor([500.0])
    with torch.no_grad():
        base = dit._wan_forward(fresh, noisy, t, None, (emb,), None, "cpu", torch.float32)
        pipe.load_lora_weights(
            str(tmp_path), weight_name = DEFAULT_LORA_FILENAME, adapter_name = "style"
        )
        loaded = {
            n for n, m in fresh.named_modules() if hasattr(m, "lora_A") and "style" in m.lora_A
        }
        assert len(loaded) == 20
        with_lora = dit._wan_forward(fresh, noisy, t, None, (emb,), None, "cpu", torch.float32)
        trained_out = dit._wan_forward(trained, noisy, t, None, (emb,), None, "cpu", torch.float32)
        pipe.disable_lora()
        off = dit._wan_forward(fresh, noisy, t, None, (emb,), None, "cpu", torch.float32)
    assert not torch.allclose(with_lora, base)
    assert torch.allclose(
        with_lora, trained_out, atol = 1e-5
    ), "the loaded adapter must reproduce the trained one"
    assert torch.equal(off, base)
    assert math.isfinite(float(with_lora.abs().max()))


def test_collate_copies_only_the_trimmed_rows_and_pads_on_the_device(monkeypatch):
    """The pad to 512 must happen AFTER the move: padding on the host is a 4 MB pageable copy plus a multi-threaded
    host fill + cat every step."""
    seen = []
    real_to = torch.Tensor.to

    def spy(self, *a, **k):
        seen.append(tuple(self.shape))
        return real_to(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    e = (torch.randn(1, 7, 8, dtype = torch.bfloat16),)
    (out,) = dit._wan_collate([e], "cpu", torch.bfloat16)
    assert seen == [(1, 7, 8)], "only the trimmed embed may be moved"
    assert out.shape == (1, _WAN_MAX_SEQUENCE_LENGTH, 8)
    assert torch.equal(out[:, :7], e[0]) and not out[:, 7:].any()


def test_cached_latent_sampling_moves_before_joining_and_keeps_the_values():
    cache = [[(torch.randn(1, Z_DIM, 1, 4, 4), torch.rand(1, Z_DIM, 1, 4, 4))] for _ in range(3)]
    torch.manual_seed(7)
    got = dit._sample_cached_latents(cache, [2, 0], None, "cpu", torch.float32)
    torch.manual_seed(7)
    a = torch.cat([cache[2][0][0], cache[0][0][0]])
    b = torch.cat([cache[2][0][1], cache[0][0][1]])
    assert torch.equal(got, a + b * torch.randn_like(a))
    one = dit._to_device_cat([cache[1][0][0]], "cpu")
    assert one is cache[1][0][0] or torch.equal(one, cache[1][0][0])


def test_a_non_default_alpha_reloads_at_the_trained_strength(tmp_path):
    """diffusers' loader assumes lora_alpha == rank unless the safetensors header carries the adapter config, so a
    rank-4 / alpha-16 adapter (scale 4) used to come back at scale 1."""
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors import safe_open

    WanPipeline, _ = _wan_classes()
    trained = _tiny_transformer()
    trained.add_adapter(
        LoraConfig(
            r = 4, lora_alpha = 16, init_lora_weights = "gaussian", target_modules = list(_WAN_TARGETS)
        )
    )
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.2)
    spec = _SPECS[FAMILY]
    assert spec.save_takes_metadata
    dit._adapter_saver(spec, trained)(None, str(tmp_path), get_peft_model_state_dict(trained))
    with safe_open(str(tmp_path / DEFAULT_LORA_FILENAME), "pt") as f:
        meta = json.loads(f.metadata()["lora_adapter_metadata"])
    assert meta["transformer.r"] == 4 and meta["transformer.lora_alpha"] == 16

    fresh = _tiny_transformer()
    pipe = WanPipeline(
        tokenizer = None,
        text_encoder = None,
        vae = None,
        scheduler = None,
        transformer = fresh,
        expand_timesteps = True,
    )
    pipe.load_lora_weights(str(tmp_path), weight_name = DEFAULT_LORA_FILENAME)
    noisy, emb, t = torch.randn(1, Z_DIM, 1, 8, 8), torch.randn(1, 16, 32), torch.tensor([400.0])
    with torch.no_grad():
        got = dit._wan_forward(fresh, noisy, t, None, (emb,), None, "cpu", torch.float32)
        want = dit._wan_forward(trained, noisy, t, None, (emb,), None, "cpu", torch.float32)
    assert torch.allclose(got, want, atol = 1e-5)


def test_families_without_the_flag_keep_their_three_argument_saver():
    for name, spec in _SPECS.items():
        if not spec.save_takes_metadata:
            assert dit._adapter_saver(spec, object()) is spec.save, name


def test_load_conditioners_leaves_the_vae_on_the_host(monkeypatch):
    moved = []

    class _Vae:
        def to(self, *a, **k):
            moved.append((a, k))
            return self

    class _Pipe:
        def __init__(self, vae):
            self.vae = vae

    _wan_classes()  # skip where this diffusers cannot import the Wan stack
    import diffusers

    monkeypatch.setattr(
        diffusers.AutoencoderKLWan, "from_pretrained", classmethod(lambda cls, *a, **k: _Vae())
    )
    monkeypatch.setattr(
        dit,
        "_wan_pipeline_cls",
        lambda: types.SimpleNamespace(from_pretrained = lambda *a, **k: _Pipe(k["vae"])),
    )
    pipe, vae = dit._wan_load_conditioners(
        types.SimpleNamespace(base_model = BASE, hf_token = None), "cuda", None
    )
    assert vae is pipe.vae and moved == [], "the VAE must wait for the encoders to be freed"
