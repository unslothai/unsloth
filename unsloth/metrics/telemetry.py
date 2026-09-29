# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in forwarding of aggregated metrics (counts and averages only, never
prompts or outputs). Off unless UNSLOTH_ENABLE_METRICS_TELEMETRY=1 or
enable_telemetry(); UNSLOTH_DISABLE_METRICS_TELEMETRY=1 always wins."""

import json
import os
import threading
import time
import urllib.request
from typing import Any, Dict, Optional

from unsloth.metrics.stats import get_stats_collector

_TELEMETRY_ENABLED = os.environ.get("UNSLOTH_ENABLE_METRICS_TELEMETRY", "0") == "1"
_TELEMETRY_DISABLED = os.environ.get("UNSLOTH_DISABLE_METRICS_TELEMETRY", "0") == "1"
_TELEMETRY_ENDPOINT = os.environ.get(
    "UNSLOTH_METRICS_TELEMETRY_ENDPOINT", "https://api.unsloth.ai/metrics"
)
_TELEMETRY_SETTINGS_ENDPOINT = os.environ.get("UNSLOTH_METRICS_TELEMETRY_SETTINGS_ENDPOINT", "")
_TELEMETRY_INTERVAL = max(1.0, float(os.environ.get("UNSLOTH_METRICS_TELEMETRY_INTERVAL", "300")))

_telemetry_thread: Optional[threading.Thread] = None
_telemetry_stop: Optional[threading.Event] = None
_telemetry_lock = threading.Lock()
# Set by schedule_telemetry, cleared by a send: at most one POST per interval, however many steps ran.
_telemetry_dirty = threading.Event()
_settings_checked = False


def is_telemetry_enabled() -> bool:
    return _TELEMETRY_ENABLED and not _TELEMETRY_DISABLED


def enable_telemetry() -> None:
    global _TELEMETRY_ENABLED
    if _TELEMETRY_DISABLED:
        return
    _TELEMETRY_ENABLED = True
    _start_telemetry_thread()


def disable_telemetry() -> None:
    global _TELEMETRY_ENABLED
    _TELEMETRY_ENABLED = False
    _stop_telemetry_thread()


def schedule_telemetry() -> None:
    """Mark new stats for the next periodic send; never blocks or queues."""
    if is_telemetry_enabled():
        _telemetry_dirty.set()


def _start_telemetry_thread() -> None:
    global _telemetry_thread, _telemetry_stop
    with _telemetry_lock:
        if _telemetry_thread is not None and _telemetry_thread.is_alive():
            return
        _telemetry_stop = threading.Event()
        _telemetry_thread = threading.Thread(
            target = _telemetry_worker,
            args = (_telemetry_stop,),
            daemon = True,
            name = "UnslothMetricsTelemetry",
        )
        _telemetry_thread.start()


def _stop_telemetry_thread() -> None:
    global _telemetry_thread, _telemetry_stop
    with _telemetry_lock:
        if _telemetry_stop is not None:
            _telemetry_stop.set()
        _telemetry_thread = _telemetry_stop = None


def _telemetry_worker(stop: threading.Event) -> None:
    _check_server_opt_out_once()
    while not stop.wait(_TELEMETRY_INTERVAL):
        if is_telemetry_enabled() and _telemetry_dirty.is_set():
            _telemetry_dirty.clear()
            _send_telemetry_batch()


def _get_package_version() -> str:
    try:
        from importlib.metadata import version
        return version("unsloth")
    except Exception:
        return "unknown"


def _build_payload() -> Dict[str, Any]:
    stats = get_stats_collector().get_all_stats()
    inf, tr = stats["inference"], stats["training"]
    return {
        "timestamp": time.time(),
        "version": _get_package_version(),
        "metrics": {
            "inference": {
                k: inf[k]
                for k in (
                    "total_requests",
                    "avg_e2e_latency",
                    "tokens_per_second",
                    "total_prompt_tokens",
                    "total_generation_tokens",
                )
            },
            "training": {
                k: tr[k] for k in ("total_steps", "avg_loss", "samples_per_second", "total_samples")
            },
        },
    }


def _send_telemetry_batch() -> None:
    if not is_telemetry_enabled() or not get_stats_collector().is_enabled():
        return
    try:
        _send_to_server(_build_payload())
    except Exception:
        pass


def _send_to_server(payload: Dict[str, Any]) -> None:
    try:
        request = urllib.request.Request(
            _TELEMETRY_ENDPOINT,
            data = json.dumps(payload).encode("utf-8"),
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Unsloth-Metrics/1.0",
            },
        )
        urllib.request.urlopen(request, timeout = 5).close()
    except Exception:
        pass


def _check_server_opt_out_once() -> None:
    """One GET of the settings endpoint (if configured); {"enabled": false} turns telemetry off."""
    global _settings_checked, _TELEMETRY_ENABLED
    if _settings_checked or not _TELEMETRY_SETTINGS_ENDPOINT:
        return
    _settings_checked = True
    try:
        request = urllib.request.Request(
            _TELEMETRY_SETTINGS_ENDPOINT, headers = {"User-Agent": "Unsloth-Metrics/1.0"}
        )
        with urllib.request.urlopen(request, timeout = 5) as response:
            if not json.loads(response.read().decode("utf-8")).get("enabled", True):
                _TELEMETRY_ENABLED = False
    except Exception:
        pass


if is_telemetry_enabled():
    _start_telemetry_thread()
