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
Metrics collection module for Unsloth.
Provides comprehensive runtime performance metrics similar to vLLM's metrics system.
"""

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
    "enable_telemetry",
    "disable_telemetry",
    "is_telemetry_enabled",
    "schedule_telemetry",
]
