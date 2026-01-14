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
Metrics integration hooks for automatic metrics collection.
"""

import time
import uuid
from typing import Any, Optional
import torch

from unsloth.metrics.stats import get_stats_collector
from unsloth.metrics.prometheus import get_metrics_registry, _metrics_enabled


def track_generate_call(func):
    """Decorator to track metrics for model.generate() calls."""

    def wrapper(self, *args, **kwargs):
        collector = get_stats_collector()
        if not collector.is_enabled():
            return func(self, *args, **kwargs)

        # Extract input_ids
        input_ids = None
        if len(args) > 0:
            input_ids = args[0]
        elif "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]

        # Get token counts
        num_prompt_tokens = 0
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() > 1:
                    num_prompt_tokens = input_ids.shape[-1] * input_ids.shape[0]
                else:
                    num_prompt_tokens = input_ids.shape[-1]
            elif isinstance(input_ids, (list, tuple)):
                num_prompt_tokens = sum(len(seq) for seq in input_ids)

        max_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_length")
        request_id = str(uuid.uuid4())

        # Start tracking
        collector.inference_stats.start_request(
            request_id = request_id,
            num_prompt_tokens = num_prompt_tokens,
            max_tokens = max_tokens,
        )
        collector.inference_stats.record_scheduled(request_id)

        # Update Prometheus counters if enabled
        if _metrics_enabled:
            registry = get_metrics_registry()
            if registry:
                registry["inference"]["prompt_tokens_total"].inc(num_prompt_tokens)
                registry["inference"]["prompt_tokens"].observe(num_prompt_tokens)

        start_time = time.time()
        first_token_time = None

        try:
            # Call the actual generate function
            output = func(self, *args, **kwargs)

            # Calculate generated tokens
            num_generation_tokens = 0
            if isinstance(output, torch.Tensor):
                if output.dim() > 1:
                    total_tokens = output.shape[-1] * output.shape[0]
                else:
                    total_tokens = output.shape[-1]
                num_generation_tokens = max(0, total_tokens - num_prompt_tokens)
            elif isinstance(output, dict) and "sequences" in output:
                sequences = output["sequences"]
                if isinstance(sequences, torch.Tensor):
                    if sequences.dim() > 1:
                        total_tokens = sequences.shape[-1] * sequences.shape[0]
                    else:
                        total_tokens = sequences.shape[-1]
                    num_generation_tokens = max(0, total_tokens - num_prompt_tokens)

            # Estimate first token time (simplified - assumes linear generation)
            if num_generation_tokens > 0:
                total_time = time.time() - start_time
                first_token_time = start_time + (total_time / num_generation_tokens)

            # Record first token
            if first_token_time:
                collector.inference_stats.record_first_token(request_id)

            # Record generation tokens
            for _ in range(num_generation_tokens):
                collector.inference_stats.record_token(request_id)

            # Finish request
            finish_reason = (
                "stop"  # Default - could be determined from generation config
            )
            collector.inference_stats.finish_request(
                request_id = request_id,
                finish_reason = finish_reason,
                num_generation_tokens = num_generation_tokens,
            )

            # Update Prometheus metrics
            if _metrics_enabled:
                registry = get_metrics_registry()
                if registry:
                    e2e_latency = time.time() - start_time
                    registry["inference"]["request_total"].labels(
                        finish_reason = finish_reason
                    ).inc()
                    registry["inference"]["generation_tokens_total"].inc(
                        num_generation_tokens
                    )
                    registry["inference"]["generation_tokens"].observe(
                        num_generation_tokens
                    )
                    registry["inference"]["request_latency_seconds"].observe(
                        e2e_latency
                    )
                    if num_generation_tokens > 0:
                        time_per_token = e2e_latency / num_generation_tokens
                        registry["inference"]["time_per_output_token_seconds"].observe(
                            time_per_token
                        )

            return output

        except Exception as e:
            # Record error
            collector.inference_stats.finish_request(
                request_id = request_id,
                finish_reason = "error",
                num_generation_tokens = 0,
            )

            if _metrics_enabled:
                registry = get_metrics_registry()
                if registry:
                    registry["inference"]["request_total"].labels(
                        finish_reason = "error"
                    ).inc()

            raise

    return wrapper
