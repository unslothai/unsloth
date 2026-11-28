# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 batched-row modular calls over a tiny random FLUX.3, CPU fp32."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from core.inference import diffusion_flux3 as f3

if not f3.flux3_available():
    pytest.skip("this diffusers build has no FLUX.3", allow_module_level = True)

REPO = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
# Read at import: the autouse fixtures point the hub cache at an empty directory for each test.
try:
    from huggingface_hub import constants as _hub_constants
    _HUB_CACHE = _hub_constants.HF_HUB_CACHE
except Exception:  # noqa: BLE001 - no hub: the fixtures skip
    _HUB_CACHE = None
SIZE = 64
STEPS = 2
LONG = "a blue sports car on a desert highway at noon with very many extra words " * 6
PROMPTS = ["a red fox", "a blue car", "a ramen bowl"]
SEEDS = [11, 22, 33]


def _tiny_pipe(distilled: bool):
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

    try:
        path = snapshot_download(REPO, local_files_only = True, cache_dir = _HUB_CACHE)
    except Exception as exc:  # noqa: BLE001 - LocalEntryNotFoundError and friends
        pytest.skip(f"{REPO} not in the Hub cache: {exc}")
    torch.manual_seed(0)
    config = transformers.Qwen3VLConfig.from_pretrained(path)
    config.text_config.num_hidden_layers = 32
    text_encoder = transformers.Qwen3VLForConditionalGeneration(config).eval()
    processor = transformers.AutoProcessor.from_pretrained(path)
    transformer = Flux3Transformer2DModel(
        in_channels = 128,
        cond_in_channels = 128,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 32,
        num_attention_heads = 2,
        joint_attention_dim = 8 * config.text_config.hidden_size,
        timestep_channels = 16,
        axes_dims_rope = (8, 8, 8, 8),
        flow_map = distilled,
    ).eval()
    vae = AutoencoderKLFlux2(
        block_out_channels = (32, 32, 32, 32),
        layers_per_block = 1,
        norm_num_groups = 8,
        latent_channels = 32,
    ).eval()
    pipe = f3.flux3_pipeline_class()(
        transformer = transformer,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = processor,
        scheduler = FlowMatchEulerDiscreteScheduler(),
        blocks_class_name = f3.FLUX3_DISTILLED_BLOCKS if distilled else f3.FLUX3_BASE_BLOCKS,
    )
    pipe.set_progress_bar_config(disable = True)
    rows: list = []
    transformer.register_forward_pre_hook(
        lambda _m, _a, kw: rows.append(int(kw["hidden_states"].shape[0])), with_kwargs = True
    )
    pipe._test_rows = rows
    return pipe


@pytest.fixture(scope = "module")
def distilled():
    return _tiny_pipe(True)


@pytest.fixture(scope = "module")
def base():
    return _tiny_pipe(False)


def _render(pipe, prompts, seeds, **kwargs):
    if not pipe.is_distilled:
        kwargs.setdefault("guidance_scale", 3.0)
        kwargs.setdefault("negative_prompt", "")
    pipe._test_rows.clear()
    out = pipe(
        prompt = prompts,
        generator = [torch.Generator().manual_seed(s) for s in seeds],
        height = SIZE,
        width = SIZE,
        num_inference_steps = STEPS,
        output_type = "np",
        **kwargs,
    )
    return [np.asarray(image) for image in out.images], list(pipe._test_rows)


def _maxabs(a, b) -> float:
    return float(np.abs(a - b).max())


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_one_denoiser_forward_per_step_for_every_row(variant, request):
    pipe = request.getfixturevalue(variant)
    images, rows = _render(pipe, PROMPTS, SEEDS)
    assert len(images) == 3 and all(image.shape == (1, SIZE, SIZE, 3) for image in images)
    branches = 1 if pipe.is_distilled else 2
    assert rows == [3] * (STEPS * branches)


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_each_row_is_its_own_render(variant, request):
    pipe = request.getfixturevalue(variant)
    batched, _ = _render(pipe, PROMPTS, SEEDS)
    for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
        solo, rows = _render(pipe, [prompt], [seed])
        assert rows == [1] * len(rows)
        # Only the batch shape differs (GEMM reduction order).
        assert _maxabs(batched[i], solo[0]) < 1e-4


@pytest.mark.parametrize("variant", ["distilled", "base"])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_a_neighbour_never_changes_a_row(variant, target, request):
    pipe = request.getfixturevalue(variant)
    batched, _ = _render(pipe, PROMPTS, SEEDS)
    again, _ = _render(pipe, PROMPTS, SEEDS)
    assert _maxabs(batched[target], again[target]) == 0.0
    neighbour = (target + 1) % 3
    prompts, seeds = list(PROMPTS), list(SEEDS)
    prompts[neighbour] = "a completely different neighbour"
    seeds[neighbour] = 999
    changed, rows = _render(pipe, prompts, seeds)
    assert rows[0] == 3
    assert _maxabs(changed[target], batched[target]) == 0.0
    assert _maxabs(changed[neighbour], batched[neighbour]) > 0.0


def test_rows_of_different_text_windows_render_apart(distilled):
    prompts = ["a red fox", LONG, "a ramen bowl", LONG + " at dusk"]
    seeds = [11, 22, 33, 44]
    images, rows = _render(distilled, prompts, seeds)
    # Two windows: two batched calls of two rows each, never one row padded to the other's window.
    assert rows == [2] * STEPS * 2
    for i in (0, 1):
        solo, _ = _render(distilled, [prompts[i]], [seeds[i]])
        assert _maxabs(images[i], solo[0]) < 1e-4


def test_each_modular_call_starts_its_own_static_skip_trajectory(distilled, monkeypatch):
    """The engine resets the static skip schedule once per chunk; every extra modular call in it must rewind."""
    from core.inference import diffusion_step_skip

    rewinds: list = []
    monkeypatch.setattr(
        diffusion_step_skip, "rewind_static_step_skip", lambda pipe: rewinds.append(pipe) or True
    )
    _render(distilled, ["a red fox", LONG, "a ramen bowl", LONG + " at dusk"], [11, 22, 33, 44])
    assert len(rewinds) == 1 and rewinds[0] is distilled  # two text windows: two trajectories
    rewinds.clear()
    _render(distilled, ["a red fox", "a ramen bowl"], [11, 22])
    assert rewinds == []  # one batched call: the chunk's own reset is enough
    rewinds.clear()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FLUX3_BATCH", "0")
    _render(distilled, ["a red fox", "a ramen bowl", "a lighthouse"], [11, 22, 33])
    assert len(rewinds) == 2  # one call per image


def test_rewind_restarts_the_step_count_and_keeps_the_plan():
    from core.inference.diffusion_step_skip import StaticStepSkip

    skip = StaticStepSkip.__new__(StaticStepSkip)
    skip.counting, skip.steps_ended, skip.ordinal = True, 7, 3
    skip.counters, skip.history, skip.plan = {"a": 1}, {"b": 2}, (True, False, True)
    from core.inference import diffusion_step_skip as S

    pipe = object()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "_find", lambda p: skip if p is pipe else None)
        assert S.rewind_static_step_skip(pipe) is True
        assert S.rewind_static_step_skip(object()) is False
    assert (skip.steps_ended, skip.ordinal, skip.counters, skip.history) == (0, 0, {}, {})
    assert skip.plan == (True, False, True)


def test_per_row_negative_prompts_group_by_negative(base):
    images, rows = _render(base, PROMPTS, SEEDS, negative_prompt = ["blurry", "blurry", "dark"])
    assert sorted(rows) == sorted([2] * STEPS * 2 + [1] * STEPS * 2)
    solo, _ = _render(base, [PROMPTS[2]], [SEEDS[2]], negative_prompt = "dark")
    assert _maxabs(images[2], solo[0]) < 1e-4


def test_a_shared_reference_edit_batches(distilled):
    from PIL import Image

    reference = Image.new("RGB", (SIZE, SIZE), (200, 30, 30))
    images, rows = _render(distilled, PROMPTS, SEEDS, image = reference)
    assert rows == [3] * STEPS
    solo, _ = _render(distilled, [PROMPTS[1]], [SEEDS[1]], image = reference)
    assert _maxabs(images[1], solo[0]) < 1e-4


def test_the_kill_switch_and_unseeded_rows_render_one_by_one(distilled, monkeypatch):
    monkeypatch.setenv(f3.BATCH_ROWS_ENV, "0")
    _, rows = _render(distilled, PROMPTS, SEEDS)
    assert rows == [1] * STEPS * 3
    monkeypatch.delenv(f3.BATCH_ROWS_ENV)
    distilled._test_rows.clear()
    distilled(
        prompt = PROMPTS[:2],
        generator = [torch.Generator().manual_seed(s) for s in SEEDS[:2]],
        latents = torch.zeros(1, 128, SIZE // 16, SIZE // 16),
        height = SIZE,
        width = SIZE,
        num_inference_steps = STEPS,
        output_type = "np",
    )
    assert distilled._test_rows == [1] * STEPS * 2


def test_a_per_call_scaled_denoiser_renders_one_by_one(distilled, monkeypatch):
    monkeypatch.setitem(
        distilled.transformer.__dict__, "_unsloth_per_call_scaled", ("blocks.0: DynamicNVFP4",)
    )
    _, rows = _render(distilled, PROMPTS, SEEDS)
    assert rows == [1] * STEPS * 3


def test_a_single_image_is_the_unbatched_call(distilled, monkeypatch):
    single, rows = _render(distilled, PROMPTS[:1], SEEDS[:1])
    assert rows == [1] * STEPS
    monkeypatch.setenv(f3.BATCH_ROWS_ENV, "0")
    again, _ = _render(distilled, PROMPTS[:1], SEEDS[:1])
    assert _maxabs(single[0], again[0]) == 0.0


def test_the_row_key_is_the_text_window(distilled):
    short = f3.flux3_row_key(distilled, "a red fox")
    long = f3.flux3_row_key(distilled, LONG)
    assert short is not None and long is not None and short != long
    assert f3.flux3_row_key(distilled, "a blue car") == short
    step, _ = f3._text_step(distilled.modular_pipeline())
    pad, layers = f3._text_defaults(step, None, None)
    for prompt, key in (("a red fox", short), (LONG, long)):
        embeds = distilled.encode_prompt(prompt, pad_multiple = pad, hidden_states_layers = layers)
        assert key == ("flux3 window", int(embeds.shape[1]))


def _ref(color, size = (SIZE, SIZE)):
    from PIL import Image
    return Image.new("RGB", size, color)


def test_edit_rows_with_different_references_never_share_a_forward(distilled):
    # Held back until a GPU run shows a row's bytes never move with a neighbour's reference.
    refs = [_ref((200, 30, 30)), _ref((30, 200, 30)), _ref((30, 30, 200))]
    images, rows = _render(distilled, PROMPTS, SEEDS, row_images = refs)
    assert rows == [1] * STEPS * 3
    solo, _ = _render(distilled, [PROMPTS[1]], [SEEDS[1]], image = refs[1])
    assert _maxabs(images[1], solo[0]) < 1e-4
    _, rows = _render(distilled, PROMPTS[:2], SEEDS[:2], row_images = refs[:2])
    assert rows == [1] * STEPS * 2


def test_edit_requests_with_references_never_coalesce():
    # The route only shares a call between plain txt2img requests; an edit request renders on its own.
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from routes.inference import _plain_txt2img

    fields = dict(
        prompts = None,
        seeds = None,
        init_image = None,
        mask_image = None,
        reference_images = None,
        workflow = None,
        localized_edit = None,
        controlnet = None,
        upscale = None,
        upscale_model = None,
        hires = None,
        variation_seed = None,
        variation_strength = None,
        xy = None,
    )
    assert _plain_txt2img(SimpleNamespace(**fields), None, None)
    for refs in (["ref-a.png"], ["ref-b.png"]):
        edit = SimpleNamespace(**{**fields, "reference_images": refs})
        assert not _plain_txt2img(edit, None, None)


def test_per_row_references_at_one_canvas_batch(distilled, monkeypatch):
    monkeypatch.setenv(f3.ROW_REFS_BATCH_ENV, "1")
    refs = [_ref((200, 30, 30)), _ref((30, 200, 30)), _ref((30, 30, 200))]
    images, rows = _render(distilled, PROMPTS, SEEDS, row_images = refs)
    assert rows == [3] * STEPS
    for i in range(3):
        solo, _ = _render(distilled, [PROMPTS[i]], [SEEDS[i]], image = refs[i])
        assert _maxabs(images[i], solo[0]) < 1e-4
    swapped = list(refs)
    swapped[1] = _ref((250, 250, 0))
    again, rows = _render(distilled, PROMPTS, SEEDS, row_images = swapped)
    assert rows == [3] * STEPS
    assert _maxabs(again[0], images[0]) == 0.0 and _maxabs(again[2], images[2]) == 0.0
    assert _maxabs(again[1], images[1]) > 0.0


def test_per_row_references_of_other_sizes_render_apart(distilled, monkeypatch):
    monkeypatch.setenv(f3.ROW_REFS_BATCH_ENV, "1")
    refs = [_ref((200, 30, 30)), _ref((30, 200, 30), (2 * SIZE, SIZE)), _ref((30, 30, 200))]
    images, rows = _render(distilled, PROMPTS, SEEDS, row_images = refs)
    assert sorted(rows) == sorted([2] * STEPS + [1] * STEPS)
    solo, _ = _render(distilled, [PROMPTS[1]], [SEEDS[1]], image = refs[1])
    assert _maxabs(images[1], solo[0]) < 1e-4
    with pytest.raises(ValueError):
        _render(distilled, PROMPTS, SEEDS, row_images = refs[:2])
    with pytest.raises(ValueError):
        _render(distilled, PROMPTS, SEEDS, row_images = refs, image = refs[0])


def test_klein_ships_a_g2c_pass_for_g4_only():
    from types import SimpleNamespace

    from core.inference import diffusion_batch_quality as quality

    g4 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    for path in ("eager", "graph"):
        assert quality.passes(quality.SHIPPED[(f3.FLUX3_KLEIN_FAMILY_NAME, g4, path)])
    state = SimpleNamespace(family = SimpleNamespace(name = f3.FLUX3_KLEIN_FAMILY_NAME), device_name = g4)
    won = {"promoted": True, "arm": "batched_eager", "path": "eager", "speedup_lb": 1.141}
    assert quality.gate(won, state) is won
    other = SimpleNamespace(family = state.family, device_name = "NVIDIA A100-SXM4-40GB")
    assert quality.gate(won, other)["promoted"] is False
    base = SimpleNamespace(
        family = SimpleNamespace(name = f3.FLUX3_KLEIN_BASE_FAMILY_NAME), device_name = g4
    )
    assert quality.gate(won, base)["promoted"] is False
