# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 Qwen3-VL vision tower in float32 on bf16 weights (h3_vision_fp32), scoped to the tower."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.inference import h3_vision_fp32 as hv  # noqa: E402

if getattr(transformers, "Qwen3VLConfig", None) is None:
    pytest.skip("transformers without Qwen3-VL", allow_module_level = True)

IMAGE, START, END = 120, 121, 122


@pytest.fixture(autouse = True)
def _switch_default(monkeypatch):
    monkeypatch.delenv(hv.H3_VISION_FP32_ENV, raising = False)


def _tiny():
    cfg = transformers.Qwen3VLConfig(
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = 128,
            rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 2,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 2,
            out_hidden_size = 32,
            patch_size = 16,
            spatial_merge_size = 2,
            temporal_patch_size = 2,
            num_position_embeddings = 64,
            deepstack_visual_indexes = [0],
        ),
        image_token_id = IMAGE,
        vision_start_token_id = START,
        vision_end_token_id = END,
    )
    cfg.vision_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model.to(torch.bfloat16)


def _inputs(grids = ((1, 4, 6), (1, 6, 4))):
    g = torch.Generator().manual_seed(1)
    grid = torch.tensor(grids, dtype = torch.long)
    patches = int(grid.prod(-1).sum())
    pixels = torch.rand(patches, 3 * 2 * 16 * 16, generator = g) * 2 - 1
    ids = [5, 6]
    for t, h, w in grids:
        ids += [START] + [IMAGE] * (t * h * w // 4) + [END]
    ids += [7, 8, 9]
    input_ids = torch.tensor([ids])
    return input_ids, pixels, grid


def _run(
    model,
    input_ids,
    pixels = None,
    grid = None,
):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "output_hidden_states": True,
    }
    if pixels is not None:
        kwargs.update(pixel_values = pixels, image_grid_thw = grid)
    if "mm_token_type_ids" in __import__("inspect").signature(model.model.forward).parameters:
        kwargs["mm_token_type_ids"] = (input_ids == IMAGE).long()
    with torch.no_grad():
        return model.model(**kwargs).hidden_states


def _vision_and_lm_dtypes(model):
    """Record the dtype each vision leaf computes in and each decoder layer receives."""
    seen = {"vision": [], "lm": []}
    hooks = []
    for name, sub in model.model.visual.named_modules():
        if isinstance(sub, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Conv3d)):
            hooks.append(
                sub.register_forward_hook(
                    lambda m, a, o, n = name: seen["vision"].append((n, o.dtype))
                )
            )
    for layer in model.model.language_model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda m, a, o: seen["lm"].append(
                    (a[0].dtype, (o[0] if isinstance(o, tuple) else o).dtype)
                )
            )
        )
    return seen, hooks


def test_vision_ops_run_float32_and_language_model_stays_bf16():
    model = _tiny()
    input_ids, pixels, grid = _inputs()
    params = {n: (p.dtype, p.data_ptr()) for n, p in model.named_parameters()}
    seen, hooks = _vision_and_lm_dtypes(model)
    with hv.h3_vision_float32(model) as wrapped:
        assert wrapped is model.model
        states = _run(model, input_ids, pixels, grid)
        upcasts = wrapped.get_image_features.upcasts
    for h in hooks:
        h.remove()
    assert seen["vision"] and all(dtype is torch.float32 for _, dtype in seen["vision"]), seen[
        "vision"
    ]
    assert upcasts >= len(seen["vision"])
    assert seen["lm"] and all(
        a is torch.bfloat16 and o is torch.bfloat16 for a, o in seen["lm"]
    ), seen["lm"]
    assert all(s.dtype is torch.bfloat16 for s in states)
    # weights untouched: same dtype, same storage, nothing patched after the block
    assert {n: (p.dtype, p.data_ptr()) for n, p in model.named_parameters()} == params
    assert "get_image_features" not in model.model.__dict__
    assert "get_video_features" not in model.model.__dict__


def test_float32_tower_matches_a_float32_copy_of_the_bf16_weights():
    import copy

    model = _tiny()
    input_ids, pixels, grid = _inputs()
    wide = copy.deepcopy(model.model.visual).float()
    with torch.no_grad(), hv.h3_vision_float32(model) as wrapped:
        out = wrapped.get_image_features(pixels, grid)
        positions = wrapped.get_image_features.position_sums
        want = wide(pixels, grid_thw = grid)
    # newer transformers return ModelOutputs, 4.57 tuples
    merged, deepstack = (
        (out.pooler_output, out.deepstack_features) if hasattr(out, "pooler_output") else out
    )
    want_merged, want_deepstack = (
        (want.pooler_output, want.deepstack_features) if hasattr(want, "pooler_output") else want
    )
    merged = torch.cat(list(merged))
    assert merged.dtype is torch.bfloat16
    # releases with fast_pos_embed_interpolate interpolate in the table dtype; later ones get the reference's sum
    assert positions == (0 if hasattr(model.model.visual, "fast_pos_embed_interpolate") else 1)
    # the float32 copy differs only in the position embeddings (it interpolates in float32)
    assert torch.allclose(merged.float(), want_merged, rtol = 3e-2, atol = 3e-2)
    assert len(deepstack) == len(want_deepstack) == 1
    for a, b in zip(deepstack, want_deepstack):
        assert a.dtype is torch.bfloat16 and a.shape == b.shape


def test_kill_switch_is_bit_identical_to_the_bf16_tower(monkeypatch):
    model = _tiny()
    input_ids, pixels, grid = _inputs()
    plain = _run(model, input_ids, pixels.to(torch.bfloat16), grid)
    monkeypatch.setenv(hv.H3_VISION_FP32_ENV, "0")
    assert hv.vision_pixel_dtype(torch.bfloat16) is torch.bfloat16
    with hv.h3_vision_float32(model) as wrapped:
        assert wrapped is None
        off = _run(model, input_ids, pixels.to(torch.bfloat16), grid)
    assert all(torch.equal(a, b) for a, b in zip(plain, off))
    monkeypatch.delenv(hv.H3_VISION_FP32_ENV)
    assert hv.vision_pixel_dtype(torch.bfloat16) is torch.float32
    with hv.h3_vision_float32(model):
        on = _run(model, input_ids, pixels, grid)
    # the vision rows move; the text-only prefix (before any vision token) does not
    assert not torch.equal(on[-1], plain[-1])
    assert torch.equal(on[-1][:, :2], plain[-1][:, :2])


def test_text_only_prompts_are_bit_identical():
    model = _tiny()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    plain = _run(model, ids)
    with hv.h3_vision_float32(model):
        on = _run(model, ids)
    assert all(torch.equal(a, b) for a, b in zip(plain, on))


def test_rounded_pixels_use_their_offered_float32_source():
    model = _tiny()
    input_ids, pixels, grid = _inputs()
    with hv.h3_vision_float32(model):
        direct = _run(model, input_ids, pixels, grid)
        rounded_alone = _run(model, input_ids, pixels.to(torch.bfloat16), grid)
        used = hv.STATS["sources_used"]
        with hv.float32_pixel_sources():
            hv.offer_float32_pixels(pixels)
            offered = _run(model, input_ids, pixels.to(torch.bfloat16), grid)
        assert hv.STATS["sources_used"] == used + 1
        with hv.float32_pixel_sources():
            # a source that does not round to the input is ignored
            hv.offer_float32_pixels(pixels + 0.5)
            mismatched = _run(model, input_ids, pixels.to(torch.bfloat16), grid)
    assert all(torch.equal(a, b) for a, b in zip(direct, offered))
    assert all(torch.equal(a, b) for a, b in zip(rounded_alone, mismatched))
    assert not torch.equal(direct[-1], rounded_alone[-1])


def test_offer_outside_a_source_block_is_a_no_op():
    hv.offer_float32_pixels(torch.zeros(2))
    assert hv._SOURCES.get() is None


def test_reference_position_embeddings_round_like_the_reference():
    table = (torch.randn(64, 8, generator = torch.Generator().manual_seed(3))).to(torch.bfloat16)
    grid = torch.tensor([[1, 4, 6], [2, 2, 2]])
    got = hv.reference_position_embeddings(table, grid, side = 8, merge = 2)
    assert got.dtype is torch.bfloat16 and got.shape == (4 * 6 + 2 * 2 * 2, 8)
    # transcription of the reference: bf16 weights, taps summed left to right in bf16, merge-block order
    rows = []
    for t, h, w in grid.tolist():
        hi = torch.linspace(0, 7, h)
        wi = torch.linspace(0, 7, w)
        out = torch.empty(h, w, 8, dtype = torch.bfloat16)
        for y in range(h):
            for x in range(w):
                y0, x0 = int(hi[y]), int(wi[x])
                y1, x1 = min(y0 + 1, 7), min(x0 + 1, 7)
                dy, dx = hi[y] - y0, wi[x] - x0
                taps = [
                    (y0, x0, (1 - dy) * (1 - dx)),
                    (y0, x1, (1 - dy) * dx),
                    (y1, x0, dy * (1 - dx)),
                    (y1, x1, dy * dx),
                ]
                acc = None
                for ty, tx, wt in taps:
                    term = table[ty * 8 + tx] * wt.to(torch.bfloat16)
                    acc = term if acc is None else acc + term
                out[y, x] = acc
        blocks = out.view(h // 2, 2, w // 2, 2, 8).permute(0, 2, 1, 3, 4).reshape(-1, 8)
        rows.append(blocks.repeat(t, 1))
    assert torch.equal(got, torch.cat(rows))


def test_float32_mode_upcasts_only_low_precision_weights():
    lin = torch.nn.Linear(4, 3).to(torch.bfloat16)
    x = torch.randn(2, 4)
    with hv._mode([]) as mode:
        y = lin(x)
    assert y.dtype is torch.float32 and mode.upcasts == 1
    want = torch.nn.functional.linear(x, lin.weight.float(), lin.bias.float())
    assert torch.equal(y, want)
    assert lin.weight.dtype is torch.bfloat16
    with hv._mode([]) as mode:
        lin(x.to(torch.bfloat16))
    assert mode.upcasts == 0


def test_pixel_downcasts_are_kept_float32_only_for_the_pixels():
    pixels = torch.randn(4, 6)
    other = torch.randn(4, 6)
    with hv._mode([pixels]):
        assert pixels.type(torch.bfloat16).dtype is torch.float32
        assert pixels.view(-1, 3).to(dtype = torch.bfloat16).dtype is torch.float32
        assert other.to(torch.bfloat16).dtype is torch.bfloat16


def _fake_step_classes(seen):
    class Base:
        def __call__(self, components, state):
            model = components.text_encoder.model
            seen["wrapped"] = getattr(model.__dict__.get("get_image_features"), hv._MARK, False)
            vision, *_ = self._gather_vision_features(None, [], 24.0)
            seen["sources"] = list(hv._SOURCES.get() or ())
            return state

        def _gather_vision_features(self, processor, references, fps):
            return (
                {"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.ones(1, 3)},
                [],
                [],
                [],
            )

    return Base


def test_ref2va_step_runs_its_conditioner_call_under_the_float32_tower(monkeypatch):
    from core.inference import video_minimax_h3_sampling as vs

    seen = {}
    monkeypatch.setattr(vs, "h3_vision_per_block", lambda: False)
    step = vs._ref2va_text_encoder_class(_fake_step_classes(seen))()
    model = _tiny()
    step(SimpleNamespace(text_encoder = model), "state")
    assert seen["wrapped"] is True
    assert len(seen["sources"]) == 1 and seen["sources"][0].dtype is torch.float32
    assert "get_image_features" not in model.model.__dict__
    monkeypatch.setenv(hv.H3_VISION_FP32_ENV, "0")
    step(SimpleNamespace(text_encoder = model), "state")
    assert seen["wrapped"] is False and seen["sources"] == []


def test_fl2va_reference_processor_offers_its_float32_pixels():
    import numpy as np
    from PIL import Image

    from core.inference import video_minimax_h3_sampling as vs

    inner = SimpleNamespace(patch_size = 16, merge_size = 2, temporal_patch_size = 2)
    proc = vs._ReferenceImageProcessor(inner)
    picture = Image.fromarray(np.full((64, 96, 3), 100, np.uint8))
    with hv.float32_pixel_sources():
        out = proc(images = [picture])
        sources = list(hv._SOURCES.get())
    assert out["pixel_values"].dtype is torch.float32
    assert len(sources) == 1 and sources[0] is out["pixel_values"]


def test_h3_image_conditioner_feeds_float32_pixels_to_a_float32_tower(monkeypatch):
    from core.inference import diffusion_minimax_h3_image as h3i

    model = _tiny()
    input_ids, pixels, grid = _inputs(((1, 4, 6),))
    seen, hooks = _vision_and_lm_dtypes(model)
    got = {}
    original = type(model.model).get_image_features

    def spy(self, pixel_values, *args, **kwargs):
        got.setdefault("pixel_dtype", pixel_values.dtype)
        return original(self, pixel_values, *args, **kwargs)

    monkeypatch.setattr(type(model.model), "get_image_features", spy)
    try:
        processor = SimpleNamespace(
            create_mm_token_type_ids = lambda rows: [[int(t == IMAGE) for t in row] for row in rows]
        )
        out = h3i.qwen3vl_hidden_state(
            model,
            processor,
            input_ids[0].tolist(),
            {"pixel_values": pixels, "image_grid_thw": grid},
            layer = 1,
            device = torch.device("cpu"),
        )
    finally:
        for h in hooks:
            h.remove()
    assert got["pixel_dtype"] is torch.float32
    assert all(dtype is torch.float32 for _, dtype in seen["vision"])
    assert all(a is torch.bfloat16 for a, _ in seen["lm"])
    assert out.dtype is torch.bfloat16
    assert "get_image_features" not in model.model.__dict__
