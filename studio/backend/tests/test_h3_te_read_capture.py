# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for ``install_h3_te_read_capture`` on a tiny random Qwen3-VL conditioner."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.inference.video_minimax_h3_te import (  # noqa: E402
    H3_TE_READ_CAPTURE_ENV,
    install_h3_te_read_capture,
)

_IMAGE = 1000


def _tiny(layers = 6, deepstack = (0, 1, 2)):
    cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if cls is None:
        pytest.skip("transformers without Qwen3-VL")
    config = transformers.Qwen3VLConfig(
        text_config = dict(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = layers,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = 1024,
            rope_scaling = {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "rope_type": "default",
            },
            rope_theta = 10000,
        ),
        vision_config = dict(
            depth = 3,
            hidden_size = 16,
            intermediate_size = 32,
            num_heads = 4,
            out_hidden_size = 16,
            patch_size = 16,
            spatial_merge_size = 2,
            temporal_patch_size = 2,
            num_position_embeddings = 64,
            deepstack_visual_indexes = list(deepstack),
        ),
        image_token_id = _IMAGE,
        video_token_id = 1001,
        vision_start_token_id = 1002,
        vision_end_token_id = 1003,
    )
    torch.manual_seed(0)
    return cls(config).eval()


def _inputs(with_image):
    ids = [5, 6, 7, 8, 9]
    kwargs = {}
    if with_image:
        grid = torch.tensor([[1, 4, 6]])
        ids = [1002] + [_IMAGE] * 6 + [1003] + ids
        kwargs = {
            "pixel_values": torch.randn(
                24, 3 * 2 * 16 * 16, generator = torch.Generator().manual_seed(1)
            ),
            "image_grid_thw": grid,
        }
    input_ids = torch.tensor([ids])
    mm = torch.tensor([[1 if i == _IMAGE else 0 for i in ids]])
    return dict(
        input_ids = input_ids,
        attention_mask = torch.ones_like(input_ids),
        mm_token_type_ids = mm,
        use_cache = False,
        **kwargs,
    )


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(H3_TE_READ_CAPTURE_ENV, raising = False)


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("layer", [4, 5])
def test_capture_returns_the_read_state_bit_for_bit_and_drops_the_rest(with_image, layer):
    model = _tiny()
    kw = _inputs(with_image)
    with torch.no_grad():
        stock = model.model(output_hidden_states = True, **kw).hidden_states
    assert install_h3_te_read_capture(model, layer = layer)
    assert install_h3_te_read_capture(model, layer = layer)  # idempotent
    with torch.no_grad():
        out = model.model(output_hidden_states = True, **kw)
    assert len(out.hidden_states) == len(stock)
    assert torch.equal(out.hidden_states[layer], stock[layer])
    assert all(s is None for i, s in enumerate(out.hidden_states) if i != layer)
    with torch.no_grad():
        plain = model.model(output_hidden_states = False, **kw)
    assert torch.equal(plain.last_hidden_state, out.last_hidden_state)
    # no hook is left behind on the read layer
    assert not model.model.language_model.layers[layer]._forward_pre_hooks


def test_kill_switch_restores_every_state(monkeypatch):
    model = _tiny()
    kw = _inputs(True)
    assert install_h3_te_read_capture(model, layer = 4)
    monkeypatch.setenv(H3_TE_READ_CAPTURE_ENV, "0")
    with torch.no_grad():
        out = model.model(output_hidden_states = True, **kw)
    assert all(s is not None for s in out.hidden_states)


@pytest.mark.parametrize("layer", [0, 3, 6, 7])
def test_declines_deepstack_and_out_of_range_layers(layer):
    # transformers 5.x records layers 1..3 before the DeepStack merge, which the pre-hook would see after it
    model = _tiny()
    assert not install_h3_te_read_capture(model, layer = layer)
    assert getattr(model.model, "_unsloth_h3_read_capture", None) is None


def test_declines_modules_without_a_decoder_stack():
    assert not install_h3_te_read_capture(torch.nn.Linear(2, 2))
    assert not install_h3_te_read_capture(None)
