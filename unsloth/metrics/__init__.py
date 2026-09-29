# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in runtime metrics for Unsloth generate() and Trainer.training_step."""

from unsloth.metrics.stats import (
    InferenceStats,
    TrainingStats,
    StatsCollector,
    get_stats_collector,
)
from unsloth.metrics.prometheus import (
    get_metrics_registry,
    generate_prometheus_metrics,
    enable_prometheus_metrics,
    disable_prometheus_metrics,
    is_prometheus_available,
)
from unsloth.metrics.server import (
    start_metrics_server,
    stop_metrics_server,
    is_metrics_server_running,
    test_metrics_server,
    get_metrics_server_port,
)
from unsloth.metrics.telemetry import (
    enable_telemetry,
    disable_telemetry,
    is_telemetry_enabled,
    schedule_telemetry,
)

__all__ = [
    "InferenceStats",
    "TrainingStats",
    "StatsCollector",
    "get_stats_collector",
    "get_metrics_registry",
    "generate_prometheus_metrics",
    "enable_prometheus_metrics",
    "disable_prometheus_metrics",
    "is_prometheus_available",
    "start_metrics_server",
    "stop_metrics_server",
    "is_metrics_server_running",
    "test_metrics_server",
    "get_metrics_server_port",
    "enable_telemetry",
    "disable_telemetry",
    "is_telemetry_enabled",
    "schedule_telemetry",
]
