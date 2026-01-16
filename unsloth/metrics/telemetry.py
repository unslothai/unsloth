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
Optional telemetry module for forwarding metrics to Unsloth's server.
Disabled by default. Opt-in via environment variable or explicit enable.
"""

from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, Optional
import urllib.error
import urllib.request

from unsloth.metrics.stats import get_stats_collector


_TELEMETRY_ENABLED = os.environ.get("UNSLOTH_ENABLE_METRICS_TELEMETRY", "0") == "1"
_TELEMETRY_DISABLED = os.environ.get("UNSLOTH_DISABLE_METRICS_TELEMETRY", "0") == "1"
_TELEMETRY_ENDPOINT = os.environ.get(
    "UNSLOTH_METRICS_TELEMETRY_ENDPOINT",
    "https://api.unsloth.ai/metrics",
)
_TELEMETRY_SETTINGS_ENDPOINT = os.environ.get(
    "UNSLOTH_METRICS_TELEMETRY_SETTINGS_ENDPOINT",
    "",
)
_TELEMETRY_INTERVAL = int(os.environ.get("UNSLOTH_METRICS_TELEMETRY_INTERVAL", "300"))

_telemetry_queue: Optional[Queue] = None
_telemetry_thread: Optional[threading.Thread] = None
_telemetry_lock = threading.Lock()
_settings_checked = False


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    return _TELEMETRY_ENABLED and not _TELEMETRY_DISABLED


def enable_telemetry() -> None:
    """Enable metrics telemetry (if not disabled via env var)."""
    global _TELEMETRY_ENABLED
    if _TELEMETRY_DISABLED:
        return
    _TELEMETRY_ENABLED = True
    _check_server_opt_out_once()
    _start_telemetry_thread()


def disable_telemetry() -> None:
    """Disable metrics telemetry."""
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = False
    _stop_telemetry_thread()


def _start_telemetry_thread() -> None:
    """Start background thread for sending telemetry."""
    global _telemetry_queue, _telemetry_thread

    with _telemetry_lock:
        if _telemetry_thread is not None and _telemetry_thread.is_alive():
            return

        _telemetry_queue = Queue()
        _telemetry_thread = threading.Thread(
            target = _telemetry_worker,
            daemon = True,
            name = "UnslothMetricsTelemetry",
        )
        _telemetry_thread.start()


def _stop_telemetry_thread() -> None:
    """Stop background telemetry thread."""
    global _telemetry_queue, _telemetry_thread

    with _telemetry_lock:
        if _telemetry_queue is not None:
            _telemetry_queue.put(None)
        _telemetry_thread = None
        _telemetry_queue = None


def _telemetry_worker() -> None:
    """Background worker that sends telemetry data periodically."""
    global _telemetry_queue
    if _telemetry_queue is None:
        return

    while True:
        try:
            item = _telemetry_queue.get(timeout = _TELEMETRY_INTERVAL)
        except Empty:
            item = "timeout"

        if item is None:
            break

        _check_server_opt_out_once()
        if is_telemetry_enabled():
            _send_telemetry_batch()


def _get_package_version() -> str:
    try:
        from importlib.metadata import version

        return version("unsloth")
    except Exception:
        return "unknown"


def _send_telemetry_batch() -> None:
    """Send a batch of metrics to Unsloth's server."""
    if not is_telemetry_enabled():
        return

    try:
        collector = get_stats_collector()
        if not collector.is_enabled():
            return

        stats = collector.get_all_stats()
        payload = {
            "timestamp": time.time(),
            "version": _get_package_version(),
            "metrics": {
                "inference": {
                    "total_requests": stats["inference"].get("total_requests", 0),
                    "avg_e2e_latency": stats["inference"].get("avg_e2e_latency", 0.0),
                    "tokens_per_second": stats["inference"].get(
                        "tokens_per_second", 0.0
                    ),
                    "total_prompt_tokens": stats["inference"].get(
                        "total_prompt_tokens", 0
                    ),
                    "total_generation_tokens": stats["inference"].get(
                        "total_generation_tokens", 0
                    ),
                },
                "training": {
                    "total_steps": stats["training"].get("total_steps", 0),
                    "avg_loss": stats["training"].get("avg_loss", 0.0),
                    "samples_per_second": stats["training"].get(
                        "samples_per_second", 0.0
                    ),
                    "total_samples": stats["training"].get("total_samples", 0),
                },
            },
        }

        _send_to_server(payload)
    except Exception:
        # Telemetry should never break user code
        pass


def _send_to_server(payload: Dict[str, Any]) -> None:
    """Send payload to Unsloth's telemetry endpoint."""
    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _TELEMETRY_ENDPOINT,
            data = data,
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Unsloth-Metrics/1.0",
            },
        )
        urllib.request.urlopen(request, timeout = 5)
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        pass


def _check_server_opt_out_once() -> None:
    """Optional one-time poll for server-side opt-out setting."""
    global _settings_checked, _TELEMETRY_ENABLED
    if _settings_checked:
        return
    _settings_checked = True

    if not _TELEMETRY_SETTINGS_ENDPOINT:
        return

    try:
        request = urllib.request.Request(
            _TELEMETRY_SETTINGS_ENDPOINT,
            headers = {"User-Agent": "Unsloth-Metrics/1.0"},
        )
        with urllib.request.urlopen(request, timeout = 5) as response:
            data = response.read().decode("utf-8")
        # Expect JSON payload like {"enabled": true}
        enabled = json.loads(data).get("enabled", True)
        if not enabled:
            _TELEMETRY_ENABLED = False
    except Exception:
        # If polling fails, keep telemetry enabled (opt-in still required)
        pass


def schedule_telemetry() -> None:
    """Schedule telemetry to be sent (non-blocking)."""
    if not is_telemetry_enabled():
        return

    global _telemetry_queue
    if _telemetry_queue is not None:
        try:
            _telemetry_queue.put_nowait("send")
        except Exception:
            pass


# Opt-in via UNSLOTH_ENABLE_METRICS_TELEMETRY=1
# Opt-out via UNSLOTH_DISABLE_METRICS_TELEMETRY=1
if is_telemetry_enabled():
    _start_telemetry_thread()
