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


@pytest.mark.parametrize("interval", [0, -1])
def test_a_non_positive_interval_is_refused_up_front(tmp_path, interval):
    with pytest.raises(ValueError, match = "capture_interval"):
        ac.ActivationCaptureConfig(output_dir = str(tmp_path), capture_interval = interval)


def _state(step, max_steps = 10):
    return SimpleNamespace(global_step = step, max_steps = max_steps, log_history = [{"loss": 9.9}])


def _records(tmp_path):
    log = tmp_path / "activation_log.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_a_snapshot_carries_the_loss_of_its_own_step(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    cb = ac.ActivationCaptureCallback(cap)
    cb.on_train_begin(None, _state(0), None)
    cb.on_step_begin(None, _state(0), None)
    model(torch.ones(1, 1, H, requires_grad = True)).sum().backward()
    cb.on_step_end(None, _state(1), None)
    # Nothing written until the step's own log arrives; the 9.9 in log_history is an older step's.
    assert _records(tmp_path) == []
    cb.on_log(None, _state(1), None, logs = {"loss": 2.5})
    assert [(r["step"], r["loss"]) for r in _records(tmp_path)] == [(0, 2.5)]


def test_a_step_without_a_log_is_written_without_a_loss(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    cb = ac.ActivationCaptureCallback(cap)
    cb.on_train_begin(None, _state(0), None)
    cb.on_step_begin(None, _state(0), None)
    model(torch.ones(1, 1, H, requires_grad = True)).sum().backward()
    cb.on_step_end(None, _state(1), None)
    # A forward after the step (eval) must not land in the held snapshot.
    model(torch.full((1, 1, H), 100.0))
    cb.on_step_begin(None, _state(1), None)
    records = _records(tmp_path)
    assert [(r["step"], r["loss"]) for r in records] == [(0, None)]
    assert records[0]["layers"]["0"]["mean_abs"] == pytest.approx([1.0] * H)


def test_a_retry_after_an_aborted_run_does_not_double_the_hooks(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    cb = ac.ActivationCaptureCallback(cap)
    cb.on_train_begin(None, _state(0), None)
    cb.on_step_begin(None, _state(0), None)
    # The run raises here: no on_step_end, no on_train_end, capture still armed.
    cb2 = ac.ActivationCaptureCallback(cap)
    cb2.on_train_begin(None, _state(0), None)
    assert len(cap._hooks) == len(cap._layers)
    assert len(cap._grad_hooks) == len(cap._layers)
    cb2.on_step_begin(None, _state(0), None)
    model(torch.full((1, 1, H), 3.0, requires_grad = True)).sum().backward()
    cb2.on_step_end(None, _state(1), None)
    cb2.on_log(None, _state(1), None, logs = {"loss": 1.0})
    assert _records(tmp_path)[-1]["layers"]["0"]["mean_abs"] == pytest.approx([3.0] * H)


def test_a_subclass_can_store_capture_without_files(tmp_path, monkeypatch):
    # The Unsloth Studio worker stores to its DB this way, with no output_dir at all.
    monkeypatch.chdir(tmp_path)
    seen = {"meta": None, "records": []}

    class _ToDb(ac.ActivationCapture):
        def _write_metadata(self, metadata):
            seen["meta"] = metadata

        def _write_record(self, record):
            seen["records"].append(record)

    model = _Model(checkpointed = False)
    cap = _ToDb(
        model,
        ac.ActivationCaptureConfig(output_dir = "", capture_interval = 1, capture_lora_norms = False),
    )
    cap.attach()
    cap.mark_capture(step = 0, loss = 0.5)
    model(torch.ones(1, 1, H, requires_grad = True)).sum().backward()
    cap.flush()
    assert seen["meta"]["model_name"] == "tiny"
    assert [(r["step"], r["loss"]) for r in seen["records"]] == [(0, 0.5)]
    assert list(tmp_path.iterdir()) == []


def test_non_finite_values_are_written_as_null(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    cap.mark_capture(step = 0, loss = float("nan"))
    model(torch.tensor([[[float("inf"), 1.0, 1.0, 1.0]]], requires_grad = True)).sum().backward()
    cap.flush()
    line = (tmp_path / "activation_log.jsonl").read_text().splitlines()[-1]
    record = json.loads(line, parse_constant = lambda c: pytest.fail(f"non-JSON token {c}"))
    assert record["loss"] is None
    assert record["layers"]["0"]["mean_abs"][0] is None


def test_a_resumed_run_has_no_step_zero_baseline(tmp_path):
    model, cap = _capture(tmp_path, checkpointed = False)
    cap.config.capture_interval = 3
    cb = ac.ActivationCaptureCallback(cap)
    cb.on_train_begin(None, _state(7), None)
    cb.on_step_begin(None, _state(7), None)
    model(torch.ones(1, 1, H, requires_grad = True)).sum().backward()
    cb.on_step_end(None, _state(8), None)
    cb.on_log(None, _state(8), None, logs = {"loss": 1.0})
    cb.on_step_begin(None, _state(8), None)
    assert _records(tmp_path) == []
