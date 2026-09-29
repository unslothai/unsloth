# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth.metrics, CPU only. The package is loaded without running unsloth/__init__.py."""

import importlib
import pathlib
import sys
import threading
import time
import types
import urllib.request

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "unsloth"


@pytest.fixture
def metrics(monkeypatch):
    if "unsloth" not in sys.modules:
        pkg = types.ModuleType("unsloth")
        pkg.__path__ = [str(_ROOT)]
        monkeypatch.setitem(sys.modules, "unsloth", pkg)
    for name in [m for m in sys.modules if m.startswith("unsloth.metrics")]:
        monkeypatch.delitem(sys.modules, name)
    stats = importlib.import_module("unsloth.metrics.stats")
    monkeypatch.setattr(stats.StatsCollector, "_instance", None)
    mods = types.SimpleNamespace(
        stats = stats,
        prometheus = importlib.import_module("unsloth.metrics.prometheus"),
        hooks = importlib.import_module("unsloth.metrics.hooks"),
        telemetry = importlib.import_module("unsloth.metrics.telemetry"),
        server = importlib.import_module("unsloth.metrics.server"),
    )
    yield mods
    mods.server.stop_metrics_server()
    mods.telemetry.disable_telemetry()


class _Model:
    def __init__(
        self,
        new_tokens = 3,
        encoder_decoder = False,
        fail = False,
    ):
        self.config = types.SimpleNamespace(is_encoder_decoder = encoder_decoder)
        self.new_tokens = new_tokens
        self.fail = fail
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        ids = args[0] if args else kwargs["input_ids"]
        n = kwargs.get("num_return_sequences", 1)
        ids = ids.repeat_interleave(n, 0)
        out = torch.ones(ids.shape[0], self.new_tokens, dtype = ids.dtype)
        return out if self.config.is_encoder_decoder else torch.cat([ids, out], 1)


def _wrap(metrics):
    return metrics.hooks.instrument_generate(_Model.generate)


def test_generate_disabled_records_nothing(metrics):
    model = _Model()
    out = _wrap(metrics)(model, torch.zeros(2, 5, dtype = torch.long))
    assert out.shape == (2, 8)
    assert metrics.stats.get_stats_collector().inference_stats.total_requests == 0


def test_generate_counts_tokens_and_finish_reason(metrics):
    metrics.prometheus.enable_prometheus_metrics()
    gen = _wrap(metrics)
    gen(_Model(new_tokens = 4), torch.zeros(2, 5, dtype = torch.long), max_new_tokens = 4)
    gen(
        _Model(new_tokens = 2),
        input_ids = torch.zeros(1, 7, dtype = torch.long),
        num_return_sequences = 3,
        max_new_tokens = 10,
    )
    s = metrics.stats.get_stats_collector().inference_stats.get_stats()
    assert s["total_requests"] == 2
    assert s["total_prompt_tokens"] == 2 * 5 + 1 * 7
    assert s["total_generation_tokens"] == 2 * 4 + 3 * 2
    assert s["finish_reasons"] == {"length": 1, "stop": 1}
    assert s["active_requests"] == 0
    if metrics.prometheus.is_prometheus_available():
        from prometheus_client import REGISTRY

        v = REGISTRY.get_sample_value
        assert v("unsloth_generation_tokens_total") == 14
        assert v("unsloth_prompt_tokens_total") == 17
        assert v("unsloth_request_latency_seconds_count") == 2
        assert v("unsloth_generation_tokens_per_request_sum") == 14


def test_generate_encoder_decoder_and_embeds(metrics):
    metrics.stats.get_stats_collector().enable()
    gen = _wrap(metrics)
    gen(_Model(new_tokens = 6, encoder_decoder = True), torch.zeros(2, 9, dtype = torch.long))
    s = metrics.stats.get_stats_collector().inference_stats.get_stats()
    assert s["total_generation_tokens"] == 2 * 6
    assert metrics.hooks._prompt_shape((), {"inputs_embeds": torch.zeros(3, 4, 8)}) == (3, 0)
    enc = {"input_ids": torch.zeros(2, 5, dtype = torch.long)}
    assert metrics.hooks._prompt_shape((), {"input_ids": enc}) == (2, 5)


def test_generate_error_is_recorded_and_reraised(metrics):
    metrics.stats.get_stats_collector().enable()
    with pytest.raises(RuntimeError, match = "boom"):
        _wrap(metrics)(_Model(fail = True), torch.zeros(1, 3, dtype = torch.long))
    s = metrics.stats.get_stats_collector().inference_stats.get_stats()
    assert s["finish_reasons"] == {"error": 1} and s["active_requests"] == 0


def test_generate_wrapper_keeps_name(metrics):
    def unsloth_base_fast_generate(self, *a, **k):
        return None

    # vision.py / llama.py skip re-patching by comparing generate.__name__.
    assert metrics.hooks.instrument_generate(unsloth_base_fast_generate).__name__ == (
        "unsloth_base_fast_generate"
    )


class _Trainer:
    def __init__(self):
        self.state = types.SimpleNamespace(global_step = 7)
        self.args = types.SimpleNamespace(gradient_accumulation_steps = 4)
        self.lr_scheduler = types.SimpleNamespace(get_last_lr = lambda: [3e-4])

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
    ):
        time.sleep(0.01)
        return torch.tensor(1.25)


def test_training_step_patch(metrics):
    Trainer = type("T", (_Trainer,), {"training_step": _Trainer.training_step})
    metrics.hooks.patch_training_metrics(Trainer)
    first = Trainer.training_step
    metrics.hooks.patch_training_metrics(Trainer)
    assert Trainer.training_step is first and first.__name__ == "training_step"
    import inspect

    assert "num_items_in_batch" in inspect.signature(Trainer.training_step).parameters

    trainer, batch = Trainer(), {"input_ids": torch.zeros(4, 16, dtype = torch.long)}
    trainer.training_step(None, batch)
    stats = metrics.stats.get_stats_collector().training_stats
    assert stats.total_steps == 0

    metrics.prometheus.enable_prometheus_metrics()
    assert float(trainer.training_step(None, batch)) == 1.25
    s = stats.get_stats()
    assert s["total_steps"] == 1 and s["total_samples"] == 4
    assert s["avg_loss"] == 1.25 * 4 and s["current_lr"] == 3e-4
    assert 0.01 <= s["avg_step_time"] < 5
    assert s["samples_per_second"] == pytest.approx(4 / s["avg_step_time"])

    packed = {
        "input_ids": torch.zeros(1, 9, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]]),
    }
    trainer.training_step(None, packed)
    assert stats.get_stats()["total_samples"] == 4 + 3


def test_telemetry_coalesces_sends(metrics, monkeypatch):
    t = metrics.telemetry
    sent = []
    monkeypatch.setattr(t, "_send_to_server", sent.append)
    monkeypatch.setattr(t, "_TELEMETRY_INTERVAL", 0.2)
    metrics.stats.get_stats_collector().enable()
    t.enable_telemetry()
    stats = metrics.stats.get_stats_collector().training_stats
    for i in range(500):
        stats.record_batch(i, 1, 0.001, 1.0, 1e-4)
    time.sleep(0.5)
    # 500 steps inside ~one interval: one or two POSTs, not 500 queued ones.
    assert 1 <= len(sent) <= 2
    assert sent[0]["metrics"]["training"]["total_steps"] == 500
    n = len(sent)
    time.sleep(0.5)
    assert len(sent) == n  # nothing new, nothing sent
    t.disable_telemetry()


def test_telemetry_env_disable_wins(metrics, monkeypatch):
    monkeypatch.setattr(metrics.telemetry, "_TELEMETRY_DISABLED", True)
    metrics.telemetry.enable_telemetry()
    assert not metrics.telemetry.is_telemetry_enabled()
    assert metrics.telemetry._telemetry_thread is None


def test_server_loopback_and_restart(metrics):
    srv = metrics.server
    srv.start_metrics_server(port = 0)
    port = srv.get_metrics_server_port()
    assert srv._metrics_server.server_address[0] == "127.0.0.1"
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout = 5) as r:
        body = r.read()
        assert r.status == 200
    if metrics.prometheus.is_prometheus_available():
        assert b"unsloth_request_total" in body
    assert srv.test_metrics_server()
    srv.stop_metrics_server()
    assert not srv.is_metrics_server_running()
    # server_close() released the socket.
    srv.start_metrics_server(port = port)
    assert srv.get_metrics_server_port() == port


def test_server_busy_port_raises(metrics):
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    try:
        with pytest.raises(OSError):
            metrics.server.start_metrics_server(port = s.getsockname()[1])
        assert not metrics.server.is_metrics_server_running()
    finally:
        s.close()


def test_prometheus_reimport_does_not_duplicate(metrics):
    if not metrics.prometheus.is_prometheus_available():
        pytest.skip("prometheus_client not installed")
    first = metrics.prometheus.get_metrics_registry()
    fresh = importlib.reload(metrics.prometheus)
    again = fresh.get_metrics_registry()
    assert again["inference"]["request_total"] is first["inference"]["request_total"]
