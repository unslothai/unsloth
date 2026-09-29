# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prometheus export of the stats collector (optional `prometheus_client`)."""

from typing import Any, Dict, Optional

from unsloth.metrics.stats import get_stats_collector

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

_metrics_registry: Optional[Dict[str, Any]] = None
_metrics_enabled = False

_LATENCY_BUCKETS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
_TOKEN_BUCKETS = [10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000]

# (key, metric type, name, help, extra kwargs)
_SPECS = {
    "inference": [
        (
            "request_total",
            "Counter",
            "unsloth_request_total",
            "Inference requests",
            {"labelnames": ["finish_reason"]},
        ),
        (
            "prompt_tokens_total",
            "Counter",
            "unsloth_prompt_tokens_total",
            "Prompt tokens processed",
            {},
        ),
        (
            "generation_tokens_total",
            "Counter",
            "unsloth_generation_tokens_total",
            "Tokens generated",
            {},
        ),
        ("requests_active", "Gauge", "unsloth_requests_active", "In-flight inference requests", {}),
        (
            "tokens_per_second",
            "Gauge",
            "unsloth_tokens_per_second",
            "Generated tokens per second over recent requests",
            {},
        ),
        (
            "request_latency_seconds",
            "Histogram",
            "unsloth_request_latency_seconds",
            "End-to-end generate() latency",
            {"buckets": _LATENCY_BUCKETS},
        ),
        (
            "time_per_output_token_seconds",
            "Histogram",
            "unsloth_time_per_output_token_seconds",
            "End-to-end latency divided by generated tokens (includes prefill)",
            {"buckets": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]},
        ),
        (
            "prompt_tokens",
            "Histogram",
            "unsloth_prompt_tokens",
            "Prompt tokens per request",
            {"buckets": _TOKEN_BUCKETS},
        ),
        (
            "generation_tokens",
            "Histogram",
            "unsloth_generation_tokens",
            "Generated tokens per request",
            {"buckets": _TOKEN_BUCKETS},
        ),
    ],
    "training": [
        (
            "training_steps_total",
            "Counter",
            "unsloth_training_steps_total",
            "Trainer.training_step calls (micro-batches)",
            {},
        ),
        (
            "training_samples_total",
            "Counter",
            "unsloth_training_samples_total",
            "Training samples processed",
            {},
        ),
        (
            "training_loss",
            "Gauge",
            "unsloth_training_loss",
            "Loss returned by the last training_step",
            {},
        ),
        ("learning_rate", "Gauge", "unsloth_learning_rate", "Current learning rate", {}),
        (
            "samples_per_second",
            "Gauge",
            "unsloth_training_samples_per_second",
            "Training samples per second",
            {},
        ),
        (
            "step_time_seconds",
            "Histogram",
            "unsloth_training_step_time_seconds",
            "Wall time of one training_step (forward + backward)",
            {"buckets": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]},
        ),
        (
            "batch_size",
            "Histogram",
            "unsloth_training_batch_size",
            "Micro-batch size",
            {"buckets": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]},
        ),
    ],
}


def _get_or_create(kind, name, help_text, kwargs):
    # Re-importing this module (e.g. importlib.reload) must not re-register: prometheus raises on duplicates.
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing is not None:
        return existing
    return {"Counter": Counter, "Gauge": Gauge, "Histogram": Histogram}[kind](
        name, help_text, **kwargs
    )


def _init_metrics():
    global _metrics_registry
    if not PROMETHEUS_AVAILABLE:
        return None
    if _metrics_registry is None:
        _metrics_registry = {
            group: {key: _get_or_create(*spec) for key, *spec in specs}
            for group, specs in _SPECS.items()
        }
    return _metrics_registry


def get_metrics_registry() -> Optional[Dict[str, Any]]:
    return _init_metrics()


def update_prometheus_metrics():
    if not (_metrics_enabled and PROMETHEUS_AVAILABLE):
        return
    registry = get_metrics_registry()
    stats = get_stats_collector().get_all_stats()
    registry["inference"]["requests_active"].set(stats["inference"]["active_requests"])
    registry["inference"]["tokens_per_second"].set(stats["inference"]["tokens_per_second"])
    registry["training"]["samples_per_second"].set(stats["training"]["samples_per_second"])


def generate_prometheus_metrics() -> bytes:
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus metrics not available (prometheus_client not installed)\n"
    update_prometheus_metrics()
    return generate_latest(REGISTRY)


def enable_prometheus_metrics():
    global _metrics_enabled
    _metrics_enabled = True
    _init_metrics()
    get_stats_collector().enable()


def disable_prometheus_metrics():
    global _metrics_enabled
    _metrics_enabled = False
    get_stats_collector().disable()


def is_prometheus_available() -> bool:
    return PROMETHEUS_AVAILABLE


def get_metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain; charset=utf-8"


def active_registry() -> Optional[Dict[str, Any]]:
    """The registry if Prometheus export is on, else None (for the hooks)."""
    return _metrics_registry if _metrics_enabled else None
