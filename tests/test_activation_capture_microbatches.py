# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A captured step covers every gradient-accumulation microbatch, and a checkpoint recompute is not counted twice."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

_SRC = Path(__file__).resolve().parents[1] / "unsloth" / "activation_capture.py"
_spec = importlib.util.spec_from_file_location("activation_capture_under_test", _SRC)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

H = 4


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(H, H, bias = False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return (self.proj(x),)


class _Model(torch.nn.Module):
    def __init__(self, checkpointed):
        super().__init__()
        self.config = SimpleNamespace(hidden_size = H, intermediate_size = H, _name_or_path = "tiny")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.checkpointed = checkpointed

    def forward(self, x):
        for layer in self.model.layers:
            if self.checkpointed:
                x = checkpoint(lambda t, m = layer: m(t)[0], x, use_reentrant = True)
            else:
                x = layer(x)[0]
        return x


def _capture(tmp_path, checkpointed):
    model = _Model(checkpointed)
    cfg = ac.ActivationCaptureConfig(
        output_dir = str(tmp_path), capture_interval = 1, capture_lora_norms = False
    )
    cap = ac.ActivationCapture(model, cfg)
    cap.attach()
    return model, cap


@pytest.mark.parametrize("checkpointed", [False, True])
def test_a_step_averages_every_microbatch(tmp_path, checkpointed):
    model, cap = _capture(tmp_path, checkpointed)
    micro = [
        torch.full((1, 2, H), 1.0, requires_grad = True),
        torch.full((1, 2, H), -3.0, requires_grad = True),
    ]
    cap.mark_capture(step = 0)
    for x in micro:
        model(x).sum().backward()
    cap.flush()
    record = json.loads((tmp_path / "activation_log.jsonl").read_text().splitlines()[-1])
    layer0 = record["layers"]["0"]
    assert layer0["mean_abs"] == pytest.approx([2.0] * H)
    assert layer0["mean"] == pytest.approx([-1.0] * H)
    # d(sum)/dH is 1 everywhere: 2 microbatches x 2 tokens x H channels of ones.
    assert record["grad_norms"]["0"] == pytest.approx((2 * 2 * H) ** 0.5)


def test_a_new_step_starts_from_zero(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    for step, value in ((0, 5.0), (1, 2.0)):
        cap.mark_capture(step = step)
        model(torch.full((1, 1, H), value, requires_grad = True)).sum().backward()
        cap.flush()
    record = json.loads((tmp_path / "activation_log.jsonl").read_text().splitlines()[-1])
    assert record["step"] == 1
    assert record["layers"]["0"]["mean_abs"] == pytest.approx([2.0] * H)


def test_the_gradient_norm_spans_row_chunks(tmp_path):
    # More rows than one chunk, in half precision, so the chunked fp32 reduction is what gets measured.
    model, cap = _capture(tmp_path, checkpointed = False)
    model.half()
    rows = ac._GRAD_NORM_ROWS * 2 + 7
    cap.mark_capture(step = 0)
    model(
        torch.full((1, rows, H), 0.5, dtype = torch.float16, requires_grad = True)
    ).float().sum().backward()
    cap.flush()
    record = json.loads((tmp_path / "activation_log.jsonl").read_text().splitlines()[-1])
    assert record["grad_norms"]["0"] == pytest.approx((rows * H) ** 0.5)
