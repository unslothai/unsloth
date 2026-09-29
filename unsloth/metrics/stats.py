# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""In-process inference and training statistics (only measured quantities)."""

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


def _schedule_telemetry():
    try:
        from unsloth.metrics.telemetry import schedule_telemetry
        schedule_telemetry()
    except Exception:
        pass


@dataclass
class RequestStats:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: Optional[int] = None
    finish_time: Optional[float] = None
    finish_reason: Optional[str] = None


@dataclass
class TrainingBatchStats:
    step: int
    batch_size: int
    step_time: float
    loss: float
    learning_rate: float


class InferenceStats:
    def __init__(self, max_recent_requests: int = 1000):
        self._lock = Lock()
        self._active_requests: Dict[str, RequestStats] = {}
        self._finished_requests: deque = deque(maxlen = max_recent_requests)
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_generation_tokens = 0
        self.total_e2e_latency = 0.0
        self.finish_reasons: Dict[str, int] = defaultdict(int)

    def start_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        max_tokens: Optional[int] = None,
    ):
        with self._lock:
            self._active_requests[request_id] = RequestStats(
                request_id = request_id,
                arrival_time = time.time(),
                num_prompt_tokens = num_prompt_tokens,
                max_tokens_param = max_tokens,
            )

    def finish_request(
        self,
        request_id: str,
        finish_reason: str = "stop",
        num_generation_tokens: int = 0,
    ) -> Optional[float]:
        """Close a request; returns its end-to-end latency, or None if unknown."""
        with self._lock:
            req = self._active_requests.pop(request_id, None)
            if req is None:
                return None
            req.finish_time = time.time()
            req.finish_reason = finish_reason
            req.num_generation_tokens = num_generation_tokens
            e2e = req.finish_time - req.arrival_time

            self.total_requests += 1
            self.total_prompt_tokens += req.num_prompt_tokens
            self.total_generation_tokens += num_generation_tokens
            self.total_e2e_latency += e2e
            self.finish_reasons[finish_reason] += 1
            self._finished_requests.append(req)
        _schedule_telemetry()
        return e2e

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            recent = list(self._finished_requests)
            n = len(recent)
            recent_time = sum(r.finish_time - r.arrival_time for r in recent)
            recent_tokens = sum(r.num_generation_tokens for r in recent)
            return {
                "total_requests": self.total_requests,
                "active_requests": len(self._active_requests),
                "avg_e2e_latency": recent_time / n if n else 0.0,
                # Whole-request time over generated tokens: prefill is not separated out.
                "avg_time_per_output_token": (
                    recent_time / recent_tokens if recent_tokens else 0.0
                ),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_generation_tokens": self.total_generation_tokens,
                "tokens_per_second": (recent_tokens / recent_time if recent_time > 0 else 0.0),
                "finish_reasons": dict(self.finish_reasons),
            }

    def reset(self):
        with self._lock:
            self._active_requests.clear()
            self._finished_requests.clear()
            self.total_requests = 0
            self.total_prompt_tokens = 0
            self.total_generation_tokens = 0
            self.total_e2e_latency = 0.0
            self.finish_reasons.clear()


class TrainingStats:
    """One record per `Trainer.training_step` call, i.e. per micro-batch."""

    def __init__(self, max_recent_batches: int = 1000):
        self._lock = Lock()
        self._recent_batches: deque = deque(maxlen = max_recent_batches)
        self.total_steps = 0
        self.total_samples = 0
        self.total_step_time = 0.0
        self.total_loss = 0.0

    def record_batch(
        self, step: int, batch_size: int, step_time: float, loss: float, learning_rate: float
    ):
        with self._lock:
            self._recent_batches.append(
                TrainingBatchStats(step, batch_size, step_time, loss, learning_rate)
            )
            self.total_steps += 1
            self.total_samples += batch_size
            self.total_step_time += step_time
            self.total_loss += loss
        _schedule_telemetry()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            recent = list(self._recent_batches)
            n = len(recent)
            recent_time = sum(b.step_time for b in recent)
            return {
                "total_steps": self.total_steps,
                "total_samples": self.total_samples,
                "avg_loss": sum(b.loss for b in recent) / n if n else 0.0,
                "avg_step_time": recent_time / n if n else 0.0,
                "samples_per_second": (
                    sum(b.batch_size for b in recent) / recent_time if recent_time > 0 else 0.0
                ),
                "current_lr": recent[-1].learning_rate if n else 0.0,
            }

    def reset(self):
        with self._lock:
            self._recent_batches.clear()
            self.total_steps = 0
            self.total_samples = 0
            self.total_step_time = 0.0
            self.total_loss = 0.0


class StatsCollector:
    _instance: Optional["StatsCollector"] = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst.inference_stats = InferenceStats()
                inst.training_stats = TrainingStats()
                inst._enabled = os.environ.get("UNSLOTH_ENABLE_METRICS", "0") == "1"
                cls._instance = inst
        return cls._instance

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def get_all_stats(self) -> Dict[str, Any]:
        return {
            "inference": self.inference_stats.get_stats(),
            "training": self.training_stats.get_stats(),
            "enabled": self._enabled,
        }

    def reset_all(self):
        self.inference_stats.reset()
        self.training_stats.reset()


def get_stats_collector() -> StatsCollector:
    return StatsCollector._instance or StatsCollector()
