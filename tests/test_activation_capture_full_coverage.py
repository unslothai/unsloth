# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dead, constant and outlier counts cover every channel, and outlier channels get heatmap columns."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "unsloth" / "activation_capture.py"
_spec = importlib.util.spec_from_file_location("activation_capture_full_under_test", _SRC)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

H, K = 200, 16
OUTLIERS = [3, 77, 150]  # big channels a 16-of-200 random sample would usually miss
DEAD = list(range(180, 200))


class _Layer(torch.nn.Module):
    """Scales channels so some are outliers and some are dead."""

    def __init__(self):
        super().__init__()
        scale = torch.ones(H)
        scale[OUTLIERS] = 50.0
        scale[DEAD] = 0.0
        self.register_buffer("scale", scale)

    def forward(self, x):
        return (x * self.scale,)


def _capture(tmp_path, **cfg):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])
    model.config = SimpleNamespace(hidden_size = H, intermediate_size = H, _name_or_path = "tiny")
    config = ac.ActivationCaptureConfig(
        output_dir = str(tmp_path),
        max_channels = K,
        capture_lora_norms = False,
        capture_gradients = False,
        **cfg,
    )
    cap = ac.ActivationCapture(model, config)
    cap.attach()
    return model, cap


def _run(
    model,
    cap,
    steps,
    value = 1.0,
):
    for step in range(steps):
        cap.mark_capture(step = step, loss = 1.0)
        x = torch.full((1, 3, H), value)
        for layer in model.model.layers:
            x = layer(x)[0] / layer.scale.clamp_min(1)  # keep the next layer's input unscaled
        cap.flush()


def _read(tmp_path):
    meta = json.loads((tmp_path / "metadata.json").read_text())
    records = [
        json.loads(line) for line in (tmp_path / "activation_log.jsonl").read_text().splitlines()
    ]
    return meta, records


def test_counts_cover_every_channel_and_outliers_get_columns(tmp_path):
    model, cap = _capture(tmp_path, pinned_channels = 3)
    _run(model, cap, steps = 5)
    meta, records = _read(tmp_path)
    assert meta["pinned_channels"] == OUTLIERS
    assert set(OUTLIERS) <= set(meta["captured_channels"]) and len(meta["captured_channels"]) == K
    last = records[-1]["coverage"]["0"]
    assert last["channels"] == H
    assert last["dead"] == len(DEAD)
    assert last["outliers"] == len(OUTLIERS)
    assert sorted(c for c, _ in last["top_outliers"]) == OUTLIERS
    # Five identical captures: every live channel is constant.
    assert last["constant"] == H - len(DEAD)


def test_sampled_values_match_the_full_reduction(tmp_path):
    model, cap = _capture(tmp_path)
    _run(model, cap, steps = 1, value = 2.0)
    meta, records = _read(tmp_path)
    expected = (torch.full((H,), 2.0) * model.model.layers[0].scale)[meta["captured_channels"]]
    assert records[0]["layers"]["0"]["mean_abs"] == pytest.approx(expected.tolist())


def test_constant_waits_for_enough_captures(tmp_path):
    model, cap = _capture(tmp_path)
    _run(model, cap, steps = 4)
    _, records = _read(tmp_path)
    assert records[-1]["coverage"]["0"]["constant"] == 0


def test_full_coverage_off_keeps_the_sampled_path(tmp_path):
    model, cap = _capture(tmp_path, full_coverage = False)
    _run(model, cap, steps = 1)
    meta, records = _read(tmp_path)
    assert "coverage" not in records[0] and "pinned_channels" not in meta
    assert len(records[0]["layers"]["0"]["mean_abs"]) == K


def test_a_second_flush_writes_nothing(tmp_path):
    model, cap = _capture(tmp_path)
    _run(model, cap, steps = 1)
    cap.flush()  # on_train_end flushes again after the last step was written
    _, records = _read(tmp_path)
    assert len(records) == 1
