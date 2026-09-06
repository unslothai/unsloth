"""Persistent ContinuousBatching manager for multi-step generation.

`model.generate_batch(...)` initializes a fresh `ContinuousBatchingManager`
on every call, which in turn allocates a new `PagedAttentionCache` and
starts a new worker thread. Inside a GRPO training loop this happens once
per step, amortized across `num_generations * per_device_train_batch_size`
prompts per step, so the constant per-step cost (cache alloc, prefill
warmup, thread spin-up) dominates when batch sizes are modest.

`install_for_model(model, generation_config)` monkey-patches
`model.generate_batch` on this instance to reuse a single long-lived
manager. The manager is started lazily on first call; `teardown(model)`
stops the background thread.

This is deliberately kept as a stand-alone helper so it can be enabled /
disabled per-run via CLI flag without touching TRL or transformers
installs.
"""

from __future__ import annotations

import threading
import types
from typing import Optional

import torch
from transformers import GenerationConfig
from transformers.generation.continuous_batching import RequestStatus


_ATTR = "_persistent_cb_manager"
_LOCK_ATTR = "_persistent_cb_lock"


def install_for_model(
    model: torch.nn.Module, generation_config: GenerationConfig
) -> None:
    """Replace `model.generate_batch` with a version that reuses one manager.

    The replacement accepts the same arguments as the stock method. A
    trailing `generation_config` supplied to the call takes precedence; if
    it differs from the one used at init, the persistent manager is torn
    down and rebuilt (rare, but keeps semantics intact).
    """
    setattr(model, _LOCK_ATTR, threading.Lock())
    setattr(model, _ATTR, None)
    setattr(model, "_persistent_cb_gen_config", generation_config)

    original = model.generate_batch

    def generate_batch(
        self,
        inputs,
        generation_config: Optional[GenerationConfig] = None,
        progress_bar: bool = False,
        slice_inputs: bool = True,
        **kwargs,
    ):
        if not inputs:
            return {}

        gen_config = (
            generation_config
            or getattr(self, "_persistent_cb_gen_config", None)
            or self.generation_config
        )

        lock = getattr(self, _LOCK_ATTR)
        with lock:
            manager = getattr(self, _ATTR)
            stale = False
            if manager is not None:
                stale = (
                    getattr(manager, "generation_config", None) is not gen_config
                    or not manager.is_running()
                )
                if stale:
                    try:
                        manager.stop(block = True, timeout = 5.0)
                    except Exception:
                        pass
                    setattr(self, _ATTR, None)
                    manager = None
            if manager is None:
                manager = self.init_continuous_batching(
                    generation_config = gen_config,
                    slice_inputs = slice_inputs,
                )
                manager.start()
                setattr(self, _ATTR, manager)

        results = {}
        num_requests = len(inputs)
        manager.add_requests(inputs, **kwargs)
        finished = 0
        while finished < num_requests:
            result = manager.get_result(timeout = 1)
            if result is None:
                if not manager.is_running():
                    break
                continue
            if result.status == RequestStatus.FINISHED:
                results[result.request_id] = result
                finished += 1
            else:
                continue
        return results

    model.generate_batch = types.MethodType(generate_batch, model)
    setattr(model, "_persistent_cb_original_generate_batch", original)


def teardown(model: torch.nn.Module) -> None:
    manager = getattr(model, _ATTR, None)
    if manager is not None:
        try:
            manager.stop(block = True, timeout = 5.0)
        except Exception:
            pass
    if hasattr(model, "_persistent_cb_original_generate_batch"):
        model.generate_batch = model._persistent_cb_original_generate_batch
