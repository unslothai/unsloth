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
Statistics tracking module for Unsloth.
Tracks runtime performance metrics during inference and training.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from threading import Lock
import os

import torch


@dataclass
class RequestStats:
    """Statistics for a single inference request."""

    request_id: str
    arrival_time: float
    scheduled_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: Optional[int] = None
    finish_reason: Optional[str] = None  # "stop", "length", "error"


@dataclass
class TrainingBatchStats:
    """Statistics for a single training batch."""

    step: int
    batch_size: int
    forward_time: float
    backward_time: float
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None


class InferenceStats:
    """Collects and aggregates inference statistics."""

    def __init__(self, max_recent_requests: int = 1000):
        self.max_recent_requests = max_recent_requests
        self._lock = Lock()

        # Per-request tracking
        self._active_requests: Dict[str, RequestStats] = {}

        # Aggregated statistics
        self.total_requests: int = 0
        self.total_prompt_tokens: int = 0
        self.total_generation_tokens: int = 0
        self.total_e2e_latency: float = 0.0
        self.total_prefill_latency: float = 0.0
        self.total_decode_latency: float = 0.0

        # Finished requests (for sliding window)
        self._finished_requests: deque = deque(maxlen = max_recent_requests)

        # Finish reason counts
        self.finish_reasons: Dict[str, int] = defaultdict(int)

        # Timing breakdowns
        self._queued_times: deque = deque(maxlen = max_recent_requests)
        self._prefill_times: deque = deque(maxlen = max_recent_requests)
        self._decode_times: deque = deque(maxlen = max_recent_requests)
        self._e2e_times: deque = deque(maxlen = max_recent_requests)

    def start_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        max_tokens: Optional[int] = None,
    ):
        """Record the start of an inference request."""
        with self._lock:
            self._active_requests[request_id] = RequestStats(
                request_id = request_id,
                arrival_time = time.time(),
                num_prompt_tokens = num_prompt_tokens,
                max_tokens_param = max_tokens,
            )

    def record_scheduled(self, request_id: str):
        """Record when a request was scheduled for processing."""
        with self._lock:
            if request_id in self._active_requests:
                if self._active_requests[request_id].scheduled_time is None:
                    self._active_requests[request_id].scheduled_time = time.time()

    def record_first_token(self, request_id: str):
        """Record when the first token was generated."""
        with self._lock:
            if request_id in self._active_requests:
                req = self._active_requests[request_id]
                if req.first_token_time is None:
                    req.first_token_time = time.time()
                    if req.scheduled_time is None:
                        req.scheduled_time = req.first_token_time

    def record_token(self, request_id: str):
        """Record each token generation (updates last_token_time)."""
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id].last_token_time = time.time()
                req = self._active_requests[request_id]
                if req.num_generation_tokens == 0:
                    req.first_token_time = req.last_token_time
                req.num_generation_tokens += 1

    def finish_request(
        self,
        request_id: str,
        finish_reason: str = "stop",
        num_generation_tokens: Optional[int] = None,
    ):
        """Record the completion of an inference request."""
        with self._lock:
            if request_id not in self._active_requests:
                return

            req = self._active_requests.pop(request_id)
            req.finish_time = time.time()
            req.finish_reason = finish_reason

            if num_generation_tokens is not None:
                req.num_generation_tokens = num_generation_tokens

            # Calculate latencies
            if req.scheduled_time is None:
                req.scheduled_time = req.arrival_time
            if req.first_token_time is None:
                req.first_token_time = req.finish_time

            e2e_latency = req.finish_time - req.arrival_time
            queued_time = req.scheduled_time - req.arrival_time
            prefill_time = req.first_token_time - req.scheduled_time
            decode_time = (
                req.last_token_time - req.first_token_time
                if req.last_token_time
                else 0.0
            )

            # Update aggregates
            self.total_requests += 1
            self.total_prompt_tokens += req.num_prompt_tokens
            self.total_generation_tokens += req.num_generation_tokens
            self.total_e2e_latency += e2e_latency
            self.total_prefill_latency += prefill_time
            self.total_decode_latency += decode_time

            # Store in sliding window
            self._finished_requests.append(req)
            self._queued_times.append(queued_time)
            self._prefill_times.append(prefill_time)
            self._decode_times.append(decode_time)
            self._e2e_times.append(e2e_latency)

            # Count finish reasons
            self.finish_reasons[finish_reason] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current aggregated statistics."""
        with self._lock:
            num_finished = len(self._finished_requests)
            if num_finished == 0:
                return {
                    "total_requests": 0,
                    "active_requests": len(self._active_requests),
                    "avg_e2e_latency": 0.0,
                    "avg_prefill_latency": 0.0,
                    "avg_decode_latency": 0.0,
                    "avg_time_per_output_token": 0.0,
                    "total_prompt_tokens": 0,
                    "total_generation_tokens": 0,
                    "tokens_per_second": 0.0,
                    "finish_reasons": {},
                }

            # Calculate averages from recent requests
            avg_e2e = sum(self._e2e_times) / num_finished
            avg_prefill = sum(self._prefill_times) / num_finished
            avg_decode = sum(self._decode_times) / num_finished

            # Calculate tokens per second from recent requests
            total_recent_time = sum(self._e2e_times)
            total_recent_tokens = sum(
                req.num_generation_tokens for req in self._finished_requests
            )
            tokens_per_second = (
                total_recent_tokens / total_recent_time
                if total_recent_time > 0
                else 0.0
            )

            # Average time per output token
            avg_time_per_token = (
                avg_decode / max(1, total_recent_tokens / num_finished)
                if total_recent_tokens > 0
                else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "active_requests": len(self._active_requests),
                "avg_e2e_latency": avg_e2e,
                "avg_prefill_latency": avg_prefill,
                "avg_decode_latency": avg_decode,
                "avg_time_per_output_token": avg_time_per_token,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_generation_tokens": self.total_generation_tokens,
                "tokens_per_second": tokens_per_second,
                "finish_reasons": dict(self.finish_reasons),
            }

    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self._active_requests.clear()
            self._finished_requests.clear()
            self.total_requests = 0
            self.total_prompt_tokens = 0
            self.total_generation_tokens = 0
            self.total_e2e_latency = 0.0
            self.total_prefill_latency = 0.0
            self.total_decode_latency = 0.0
            self.finish_reasons.clear()
            self._queued_times.clear()
            self._prefill_times.clear()
            self._decode_times.clear()
            self._e2e_times.clear()


class TrainingStats:
    """Collects and aggregates training statistics."""

    def __init__(self, max_recent_batches: int = 1000):
        self.max_recent_batches = max_recent_batches
        self._lock = Lock()

        # Aggregated statistics
        self.total_steps: int = 0
        self.total_samples: int = 0
        self.total_forward_time: float = 0.0
        self.total_backward_time: float = 0.0
        self.total_loss: float = 0.0

        # Recent batches for sliding window
        self._recent_batches: deque = deque(maxlen = max_recent_batches)

    def record_batch(
        self,
        step: int,
        batch_size: int,
        forward_time: float,
        backward_time: float,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
    ):
        """Record statistics for a training batch."""
        with self._lock:
            batch_stats = TrainingBatchStats(
                step = step,
                batch_size = batch_size,
                forward_time = forward_time,
                backward_time = backward_time,
                loss = loss,
                learning_rate = learning_rate,
                grad_norm = grad_norm,
            )
            self._recent_batches.append(batch_stats)

            self.total_steps += 1
            self.total_samples += batch_size
            self.total_forward_time += forward_time
            self.total_backward_time += backward_time
            self.total_loss += loss

    def get_stats(self) -> Dict[str, Any]:
        """Get current aggregated statistics."""
        with self._lock:
            num_batches = len(self._recent_batches)
            if num_batches == 0:
                return {
                    "total_steps": 0,
                    "total_samples": 0,
                    "avg_loss": 0.0,
                    "avg_forward_time": 0.0,
                    "avg_backward_time": 0.0,
                    "samples_per_second": 0.0,
                    "current_lr": 0.0,
                }

            recent_loss = sum(b.loss for b in self._recent_batches) / num_batches
            recent_forward = (
                sum(b.forward_time for b in self._recent_batches) / num_batches
            )
            recent_backward = (
                sum(b.backward_time for b in self._recent_batches) / num_batches
            )
            recent_samples = sum(b.batch_size for b in self._recent_batches)
            recent_time = sum(
                b.forward_time + b.backward_time for b in self._recent_batches
            )
            samples_per_second = (
                recent_samples / recent_time if recent_time > 0 else 0.0
            )

            current_lr = (
                self._recent_batches[-1].learning_rate if self._recent_batches else 0.0
            )

            return {
                "total_steps": self.total_steps,
                "total_samples": self.total_samples,
                "avg_loss": recent_loss,
                "avg_forward_time": recent_forward,
                "avg_backward_time": recent_backward,
                "samples_per_second": samples_per_second,
                "current_lr": current_lr,
            }

    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self._recent_batches.clear()
            self.total_steps = 0
            self.total_samples = 0
            self.total_forward_time = 0.0
            self.total_backward_time = 0.0
            self.total_loss = 0.0


class StatsCollector:
    """Global statistics collector that manages both inference and training stats."""

    _instance: Optional["StatsCollector"] = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.inference_stats = InferenceStats()
        self.training_stats = TrainingStats()
        self._enabled = os.environ.get("UNSLOTH_ENABLE_METRICS", "0") == "1"
        self._initialized = True

    def enable(self):
        """Enable metrics collection."""
        self._enabled = True

    def disable(self):
        """Disable metrics collection."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics (inference + training)."""
        return {
            "inference": self.inference_stats.get_stats(),
            "training": self.training_stats.get_stats(),
            "enabled": self._enabled,
        }

    def reset_all(self):
        """Reset all statistics."""
        self.inference_stats.reset()
        self.training_stats.reset()


# Global singleton instance
_stats_collector = None


def get_stats_collector() -> StatsCollector:
    """Get the global stats collector instance."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatsCollector()
    return _stats_collector
