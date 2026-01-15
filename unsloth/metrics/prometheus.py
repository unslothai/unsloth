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

"""
Prometheus metrics export module for Unsloth.
Provides Prometheus-compatible metrics for monitoring.
"""

import os
from typing import Optional, Dict, Any

from unsloth.metrics.stats import get_stats_collector

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        REGISTRY,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    REGISTRY = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None


# Prometheus metrics (initialized if available)
_metrics_registry: Optional[Dict[str, Any]] = None
_metrics_enabled = False


_METRIC_NAMES = {
    "inference": {
        "request_total": "unsloth_request_total",
        "prompt_tokens_total": "unsloth_prompt_tokens_total",
        "generation_tokens_total": "unsloth_generation_tokens_total",
        "requests_active": "unsloth_requests_active",
        "tokens_per_second": "unsloth_tokens_per_second",
        "request_latency_seconds": "unsloth_request_latency_seconds",
        "prefill_latency_seconds": "unsloth_prefill_latency_seconds",
        "decode_latency_seconds": "unsloth_decode_latency_seconds",
        "time_per_output_token_seconds": "unsloth_time_per_output_token_seconds",
        "prompt_tokens": "unsloth_prompt_tokens",
        "generation_tokens": "unsloth_generation_tokens",
    },
    "training": {
        "training_steps_total": "unsloth_training_steps_total",
        "training_samples_total": "unsloth_training_samples_total",
        "training_loss": "unsloth_training_loss",
        "learning_rate": "unsloth_learning_rate",
        "samples_per_second": "unsloth_training_samples_per_second",
        "gradient_norm": "unsloth_gradient_norm",
        "forward_time_seconds": "unsloth_training_forward_time_seconds",
        "backward_time_seconds": "unsloth_training_backward_time_seconds",
        "batch_size": "unsloth_training_batch_size",
    },
}


def _load_existing_metrics() -> Optional[Dict[str, Any]]:
    """Load existing metrics from the global registry if already registered."""
    if REGISTRY is None:
        return None

    existing: Dict[str, Dict[str, Any]] = {"inference": {}, "training": {}}
    for section, names in _METRIC_NAMES.items():
        for key, metric_name in names.items():
            collector = REGISTRY._names_to_collectors.get(metric_name)  # type: ignore[attr-defined]
            if collector is None:
                return None
            existing[section][key] = collector
    return existing


def _init_metrics():
    """Initialize Prometheus metrics if available."""
    global _metrics_registry

    if not PROMETHEUS_AVAILABLE:
        return None

    existing = _load_existing_metrics()
    if existing is not None:
        _metrics_registry = existing
        return _metrics_registry

    if _metrics_registry is not None:
        return _metrics_registry

    # Inference metrics
    inference_metrics = {
        # Counters
        "request_total": Counter(
            "unsloth_request_total",
            "Total number of inference requests",
            ["finish_reason"],
        ),
        "prompt_tokens_total": Counter(
            "unsloth_prompt_tokens_total",
            "Total number of prompt tokens processed",
        ),
        "generation_tokens_total": Counter(
            "unsloth_generation_tokens_total",
            "Total number of generation tokens produced",
        ),
        # Gauges
        "requests_active": Gauge(
            "unsloth_requests_active",
            "Number of currently active inference requests",
        ),
        "tokens_per_second": Gauge(
            "unsloth_tokens_per_second",
            "Current tokens per second throughput",
        ),
        # Histograms
        "request_latency_seconds": Histogram(
            "unsloth_request_latency_seconds",
            "End-to-end request latency in seconds",
            buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
        ),
        "prefill_latency_seconds": Histogram(
            "unsloth_prefill_latency_seconds",
            "Prefill (prompt processing) latency in seconds",
            buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        ),
        "decode_latency_seconds": Histogram(
            "unsloth_decode_latency_seconds",
            "Decode (generation) latency in seconds",
            buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        ),
        "time_per_output_token_seconds": Histogram(
            "unsloth_time_per_output_token_seconds",
            "Time per output token in seconds",
            buckets = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        ),
        "prompt_tokens": Histogram(
            "unsloth_prompt_tokens",
            "Number of prompt tokens per request",
            buckets = [10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        ),
        "generation_tokens": Histogram(
            "unsloth_generation_tokens",
            "Number of generation tokens per request",
            buckets = [10, 50, 100, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        ),
    }

    # Training metrics
    training_metrics = {
        # Counters
        "training_steps_total": Counter(
            "unsloth_training_steps_total",
            "Total number of training steps",
        ),
        "training_samples_total": Counter(
            "unsloth_training_samples_total",
            "Total number of training samples processed",
        ),
        # Gauges
        "training_loss": Gauge(
            "unsloth_training_loss",
            "Current training loss",
        ),
        "learning_rate": Gauge(
            "unsloth_learning_rate",
            "Current learning rate",
        ),
        "samples_per_second": Gauge(
            "unsloth_training_samples_per_second",
            "Training throughput in samples per second",
        ),
        "gradient_norm": Gauge(
            "unsloth_gradient_norm",
            "Current gradient norm",
        ),
        # Histograms
        "forward_time_seconds": Histogram(
            "unsloth_training_forward_time_seconds",
            "Forward pass time in seconds",
            buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        ),
        "backward_time_seconds": Histogram(
            "unsloth_training_backward_time_seconds",
            "Backward pass time in seconds",
            buckets = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        ),
        "batch_size": Histogram(
            "unsloth_training_batch_size",
            "Training batch size",
            buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        ),
    }

    _metrics_registry = {
        "inference": inference_metrics,
        "training": training_metrics,
    }

    return _metrics_registry


def get_metrics_registry() -> Optional[Dict[str, Any]]:
    """Get the Prometheus metrics registry."""
    return _init_metrics()


def update_prometheus_metrics():
    """Update Prometheus metrics from stats collector."""
    if not _metrics_enabled or not PROMETHEUS_AVAILABLE:
        return

    registry = get_metrics_registry()
    if registry is None:
        return

    collector = get_stats_collector()
    if not collector.is_enabled():
        return

    # Update inference metrics
    inference_stats = collector.inference_stats.get_stats()
    inference_metrics = registry["inference"]

    inference_metrics["requests_active"].set(inference_stats.get("active_requests", 0))
    inference_metrics["tokens_per_second"].set(
        inference_stats.get("tokens_per_second", 0.0)
    )

    # Note: Counters and histograms are updated when events occur,
    # not from aggregated stats. They're updated in the integration hooks.

    # Update training metrics
    training_stats = collector.training_stats.get_stats()
    training_metrics = registry["training"]

    training_metrics["training_loss"].set(training_stats.get("avg_loss", 0.0))
    training_metrics["learning_rate"].set(training_stats.get("current_lr", 0.0))
    training_metrics["samples_per_second"].set(
        training_stats.get("samples_per_second", 0.0)
    )


def generate_prometheus_metrics() -> bytes:
    """Generate Prometheus metrics output in text format."""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus metrics not available (prometheus_client not installed)\n"

    update_prometheus_metrics()
    return generate_latest(REGISTRY)


def enable_prometheus_metrics():
    """Enable Prometheus metrics collection and export."""
    global _metrics_enabled
    _metrics_enabled = True
    _init_metrics()
    get_stats_collector().enable()


def disable_prometheus_metrics():
    """Disable Prometheus metrics collection."""
    global _metrics_enabled
    _metrics_enabled = False
    get_stats_collector().disable()


def is_prometheus_available() -> bool:
    """Check if prometheus_client is available."""
    return PROMETHEUS_AVAILABLE


def get_metrics_content_type() -> str:
    """Get the Content-Type header for Prometheus metrics."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain; charset=utf-8"
