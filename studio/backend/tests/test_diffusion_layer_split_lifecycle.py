# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A layer split across the image backend's lifecycle, CPU only: the generate-time guard on the cards that run blocks
(priced from what the plan reserved there, read off the loaded pipe even for a workflow pipeline built with
from_pipe), and the on-demand resident hooks a stale one-card measured reserve would otherwise put on a split
denoiser."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusion as D
from core.inference import diffusion_layer_split_plan as lsp
from core.inference import diffusion_memory as dm

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_backend import _loaded_backend, _tiny_png_b64, fake_runtime  # noqa: E402,F401


class _Pipe:
    components: dict = {}


def _split(
    required = 12000,
    weights = 9000,
    reserve = None,
) -> lsp.LayerSplit:
    return lsp.LayerSplit(
        **({} if reserve is None else {"card_reserve_mib": ((0, 1024), (1, reserve))}),
        primary = 0,
        cards = (0, 1),
        ranges = (
            lsp.BlockRange("transformer", 0, 10, 0, 9000),
            lsp.BlockRange("transformer", 10, 20, 1, weights),
        ),
        text_encoders_card = None,
        card_required_mib = ((0, 30000), (1, required)),
        card_limit_mib = ((0, 40000), (1, 20000)),
        card_weights_mib = ((0, 20000), (1, weights)),
    )


def _placed(split: lsp.LayerSplit) -> _Pipe:
    pipe = _Pipe()
    lsp._APPLIED[pipe] = split
    return pipe


def _free(monkeypatch, free_mib):
    monkeypatch.setattr(
        D,
        "reclaimable_snapshot_device_memory",
        lambda target: dm.DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "discrete_vram",
            free_mib = free_mib,
            total_mib = 24576,
        ),
    )


def _kw(side):
    return {
        "width": side,
        "height": side,
        "batch_size": 1,
        "family": "flux2",
        "condition_pixels": 0,
    }


def test_block_card_guard_prices_the_planned_headroom_not_the_whole_request(monkeypatch):
    pipe = _placed(_split())
    _free(monkeypatch, 8000)
    assert D._guard_layer_cards(pipe, object(), _kw(1024)) == 1
    assert D._guard_layer_cards(pipe, object(), _kw(1536)) == 1
    _free(monkeypatch, 4000)
    assert D._guard_layer_cards(pipe, object(), _kw(1024)) == 1
    with pytest.raises(ValueError, match = "cuda:1"):
        D._guard_layer_cards(pipe, object(), _kw(1536))
    assert D._guard_layer_cards(pipe, object(), {**_kw(1536), "allow_oversized": True}) == 0


def test_block_card_guard_scales_only_the_working_set_not_the_fixed_reserve(monkeypatch):
    pipe = _placed(_split(reserve = 2500))
    assert pipe in lsp._APPLIED and lsp._APPLIED[pipe].working_mib(1) == (500, 2500)
    _free(monkeypatch, 4000)
    assert D._guard_layer_cards(pipe, object(), _kw(1536)) == 1
    _free(monkeypatch, 2400)
    with pytest.raises(ValueError, match = "cuda:1"):
        D._guard_layer_cards(pipe, object(), _kw(1024))


def test_block_card_guard_is_a_no_op_without_a_layer_split(monkeypatch):
    _free(monkeypatch, 10)
    assert D._guard_layer_cards(_Pipe(), object(), _kw(2048)) == 0
    assert D._guard_layer_cards(None, object(), _kw(2048)) == 0


def test_image_generate_guards_the_loaded_pipe_for_a_workflow_pipeline(
    fake_runtime, tmp_path, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        D,
        "raise_on_image_activation_shortfall",
        lambda **kw: types.SimpleNamespace(action = dm.ACTIVATION_RUN, overridden = False),
    )
    monkeypatch.setattr(
        D, "_guard_layer_cards", lambda pipe, target, kwargs: seen.append(pipe) or 0
    )
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    backend.generate(
        prompt = "a car", steps = 4, guidance = 0.0, seed = 3, init_image = _tiny_png_b64(), strength = 0.5
    )
    assert seen and seen[-1] is backend._state.pipe


def test_a_layer_split_drops_the_one_card_measured_reserve():
    class Pipe:
        _unsloth_measured_reserve = (4096, "flux2", "default")

    pipe = Pipe()
    plain = types.SimpleNamespace(layer_split = None)
    assert not D._drop_measured_reserve_for_layer_split(pipe, plain)
    assert pipe._unsloth_measured_reserve
    assert D._drop_measured_reserve_for_layer_split(
        pipe, types.SimpleNamespace(layer_split = _split())
    )
    assert not pipe._unsloth_measured_reserve
    assert (
        dm.measured_request_extra_mib(pipe, width = 2048, height = 2048, condition_pixels = 1 << 20) == 0
    )
