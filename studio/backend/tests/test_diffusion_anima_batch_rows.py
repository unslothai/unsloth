# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Anima batched-row modular calls over a tiny random Anima, CPU fp32."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from core.inference import diffusion_anima as an

if not an.anima_available():
    pytest.skip("this diffusers build has no Anima", allow_module_level = True)

QWEN = "trl-internal-testing/tiny-Qwen3ForCausalLM"
T5 = "hf-internal-testing/tiny-random-t5"
MIN_LENGTH = 32
STEPS = 2
PROMPTS = ["a red fox", "a blue car on a long road", "ramen", "a lighthouse at dusk"]
SEEDS = [1, 2, 3, 4]
LONG = "a very long neighbour prompt with far more tokens than the conditioner minimum " * 3
try:
    from huggingface_hub import constants as _hub_constants
    _HUB_CACHE = _hub_constants.HF_HUB_CACHE
except Exception:  # noqa: BLE001
    _HUB_CACHE = None


@pytest.fixture(scope = "module")
def pipe():
    from huggingface_hub import snapshot_download
    from diffusers import (
        AnimaTextConditioner,
        AutoencoderKLQwenImage,
        CosmosTransformer3DModel,
        FlowMatchEulerDiscreteScheduler,
    )

    try:
        qwen = snapshot_download(QWEN, local_files_only = True, cache_dir = _HUB_CACHE)
        t5 = snapshot_download(T5, local_files_only = True, cache_dir = _HUB_CACHE)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny tokenizers not in the Hub cache: {exc}")
    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained(qwen)
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained(t5)
    transformer = CosmosTransformer3DModel(
        in_channels = 16,
        out_channels = 16,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 2,
        mlp_ratio = 2.0,
        text_embed_dim = 16,
        adaln_lora_dim = 4,
        max_size = (2, 16, 16),
        patch_size = (1, 2, 2),
        rope_scale = (1.0, 1.0, 1.0),
        concat_padding_mask = True,
        extra_pos_embed_type = None,
    ).eval()
    out = an.anima_pipeline_class()(
        transformer = transformer,
        text_conditioner = AnimaTextConditioner(
            source_dim = config.hidden_size,
            target_dim = 16,
            model_dim = 16,
            num_layers = 1,
            num_attention_heads = 2,
            target_vocab_size = len(t5_tokenizer) + 100,
            min_sequence_length = MIN_LENGTH,
        ).eval(),
        text_encoder = transformers.Qwen3Model(config).eval(),
        tokenizer = transformers.AutoTokenizer.from_pretrained(qwen),
        t5_tokenizer = t5_tokenizer,
        vae = AutoencoderKLQwenImage(
            base_dim = 8, z_dim = 16, dim_mult = [1, 1, 1, 1], num_res_blocks = 1
        ).eval(),
        scheduler = FlowMatchEulerDiscreteScheduler(),
    )
    out.set_progress_bar_config(disable = True)
    rows: list = []

    def _hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        rows.append(int((hidden if hidden is not None else args[0]).shape[0]))

    transformer.register_forward_pre_hook(_hook, with_kwargs = True)
    out._test_rows = rows
    return out


def _render(pipe, prompts, seeds, **kwargs):
    pipe._test_rows.clear()
    out = pipe(
        prompt = prompts,
        generator = [torch.Generator().manual_seed(s) for s in seeds],
        height = 64,
        width = 64,
        num_inference_steps = STEPS,
        guidance_scale = 4.0,
        output_type = "np",
        **kwargs,
    )
    return [np.asarray(image) for image in out.images], list(pipe._test_rows)


def _maxabs(a, b) -> float:
    return float(np.abs(a - b).max())


def test_one_denoiser_forward_per_step_and_branch_for_every_row(pipe):
    images, rows = _render(pipe, PROMPTS, SEEDS)
    assert len(images) == 4
    assert rows == [4] * STEPS * 2


def test_each_row_is_its_own_render(pipe):
    batched, _ = _render(pipe, PROMPTS, SEEDS, negative_prompt = "blurry")
    for i in range(4):
        solo, rows = _render(pipe, [PROMPTS[i]], [SEEDS[i]], negative_prompt = "blurry")
        assert rows == [1] * STEPS * 2
        assert _maxabs(batched[i], solo[0]) < 1e-4


@pytest.mark.parametrize("target", [0, 1, 3])
def test_a_neighbour_never_changes_a_row(pipe, target):
    batched, _ = _render(pipe, PROMPTS, SEEDS)
    again, _ = _render(pipe, PROMPTS, SEEDS)
    assert _maxabs(batched[target], again[target]) == 0.0
    neighbour = (target + 1) % 4
    prompts, seeds = list(PROMPTS), list(SEEDS)
    prompts[neighbour] = "a neon robot dancing"
    seeds[neighbour] = 99
    negatives = ["blurry"] * 4
    base, _ = _render(pipe, PROMPTS, SEEDS, negative_prompt = negatives)
    negatives[neighbour] = "dark, low quality"
    changed, rows = _render(pipe, prompts, seeds, negative_prompt = negatives)
    assert rows == [4] * STEPS * 2
    assert _maxabs(changed[target], base[target]) == 0.0
    assert _maxabs(changed[neighbour], base[neighbour]) > 0.0


def test_a_row_past_the_conditioner_minimum_renders_apart(pipe):
    prompts = [PROMPTS[0], LONG, PROMPTS[2]]
    images, rows = _render(pipe, prompts, SEEDS[:3])
    assert sorted(rows) == sorted([2] * STEPS * 2 + [1] * STEPS * 2)
    solo, _ = _render(pipe, [LONG], [SEEDS[1]])
    assert _maxabs(images[1], solo[0]) < 1e-4


def test_the_kill_switch_and_caller_latents_render_one_by_one(pipe, monkeypatch):
    monkeypatch.setenv(an.BATCH_ROWS_ENV, "0")
    _, rows = _render(pipe, PROMPTS[:2], SEEDS[:2])
    assert rows == [1] * STEPS * 2 * 2
    monkeypatch.delenv(an.BATCH_ROWS_ENV)
    _, rows = _render(pipe, PROMPTS[:2], SEEDS[:2], latents = torch.zeros(1, 16, 1, 8, 8))
    assert rows == [1] * STEPS * 2 * 2


def test_a_per_call_scaled_denoiser_renders_one_by_one(pipe, monkeypatch):
    monkeypatch.setitem(
        pipe.transformer.__dict__, "_unsloth_per_call_scaled", ("blocks.0: DynamicNVFP4",)
    )
    _, rows = _render(pipe, PROMPTS[:2], SEEDS[:2])
    assert rows == [1] * STEPS * 2 * 2


def test_a_single_image_is_the_unbatched_call(pipe, monkeypatch):
    single, rows = _render(pipe, PROMPTS[:1], SEEDS[:1])
    assert rows == [1] * STEPS * 2
    monkeypatch.setenv(an.BATCH_ROWS_ENV, "0")
    again, _ = _render(pipe, PROMPTS[:1], SEEDS[:1])
    assert _maxabs(single[0], again[0]) == 0.0


def test_the_anima_record_keeps_requests_serial_without_a_card_verdict():
    # No pipe, or a card without a verdict: one image per forward, adapter refuses.
    from core.inference import diffusion_families as families
    from core.inference.diffusion import _generation_chunks
    from core.inference.diffusion_batch_adapters import adapter_for

    record = next(f for f in families._FAMILIES if f.name == an.ANIMA_FAMILY_NAME)
    assert record.batched_forward is None
    jobs = [("a", 1), ("a", 2), ("b", 3)]
    assert _generation_chunks(record, jobs, 4) == [[j] for j in jobs]
    assert adapter_for(record) is None
