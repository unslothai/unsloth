# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""generate() / Trainer.training_step instrumentation. Disabled cost: one attribute read."""

import functools
import time
import uuid

import torch

from unsloth.metrics.prometheus import active_registry
from unsloth.metrics.stats import StatsCollector, get_stats_collector


def _enabled() -> bool:
    inst = StatsCollector._instance
    return inst is not None and inst._enabled


def _prompt_shape(args, kwargs):
    """(rows, tokens) of a token-id prompt, else (rows, 0) for embeds / audio features."""
    x = args[0] if args else None
    if x is None:
        for key in ("input_ids", "inputs", "input"):
            if kwargs.get(key) is not None:
                x = kwargs[key]
                break
    if x is not None and not isinstance(x, torch.Tensor) and hasattr(x, "get"):
        x = x.get("input_ids")  # BatchEncoding
    if isinstance(x, torch.Tensor) and x.dim() == 2 and not x.is_floating_point():
        return x.shape[0], x.shape[1]
    for key in ("inputs_embeds", "input_embeds", "input_features"):
        v = kwargs.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            return v.shape[0], 0
    return 0, 0


def _max_new_tokens(kwargs):
    value = kwargs.get("max_new_tokens")
    if value is None and kwargs.get("generation_config") is not None:
        value = getattr(kwargs["generation_config"], "max_new_tokens", None)
    return value


def _count_generated(model, output, prompt_tokens):
    """(generated tokens summed over returned rows, longest per-row count). Rows padded
    after an early EOS count as generated: HF returns a rectangular tensor."""
    seq = output if isinstance(output, torch.Tensor) else getattr(output, "sequences", None)
    if not isinstance(seq, torch.Tensor) or seq.dim() == 0:
        return 0, 0
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    # Encoder-decoder and embeds-only prompts return only decoder / new tokens.
    if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
        prompt_tokens = 0
    per_row = max(0, seq.shape[-1] - prompt_tokens)
    return per_row * seq.shape[0], per_row


def instrument_generate(generate):
    """Wrap a `generate(self, *args, **kwargs)` so each call records one inference request."""

    @functools.wraps(generate)
    def wrapper(self, *args, **kwargs):
        if not _enabled():
            return generate(self, *args, **kwargs)
        try:
            rows, prompt_len = _prompt_shape(args, kwargs)
            max_new = _max_new_tokens(kwargs)
            num_prompt = rows * prompt_len
            stats = get_stats_collector().inference_stats
            request_id = uuid.uuid4().hex
            stats.start_request(request_id, num_prompt, max_new)
        except Exception:
            return generate(self, *args, **kwargs)

        finish_reason, num_generated = "error", 0
        try:
            output = generate(self, *args, **kwargs)
            num_generated, per_row = _count_generated(self, output, prompt_len)
            finish_reason = "length" if max_new is not None and per_row >= max_new else "stop"
            return output
        finally:
            try:
                e2e = stats.finish_request(request_id, finish_reason, num_generated)
                registry = active_registry()
                if registry is not None and e2e is not None:
                    m = registry["inference"]
                    m["request_total"].labels(finish_reason = finish_reason).inc()
                    m["prompt_tokens_total"].inc(num_prompt)
                    m["prompt_tokens"].observe(num_prompt)
                    m["generation_tokens_total"].inc(num_generated)
                    m["generation_tokens"].observe(num_generated)
                    m["request_latency_seconds"].observe(e2e)
                    if num_generated > 0:
                        m["time_per_output_token_seconds"].observe(e2e / num_generated)
            except Exception:
                pass

    return wrapper


def _batch_size(inputs):
    if not isinstance(inputs, dict):
        return 0
    for key in ("input_ids", "inputs", "input_features", "inputs_embeds", "pixel_values"):
        v = inputs.get(key)
        if hasattr(v, "shape") and len(v.shape) > 0:
            rows = int(v.shape[0])
            break
    else:
        return 0
    # Padding-free / packed batches are one row of several sequences, each restarting position_ids at 0.
    pos = inputs.get("position_ids")
    if rows == 1 and isinstance(pos, torch.Tensor) and pos.dim() == 2:
        return max(1, int((pos[0] == 0).sum()))
    return rows


def _learning_rate(trainer):
    try:
        return float(trainer.lr_scheduler.get_last_lr()[0])
    except Exception:
        try:
            return float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            return 0.0


def patch_training_metrics(Trainer):
    """Wrap Trainer.training_step (idempotent). Enabled cost per micro-batch: a loss `.item()`
    sync (plus a position_ids count for padding-free batches)."""
    if getattr(Trainer.training_step, "_unsloth_metrics_wrapped", False):
        return
    original = Trainer.training_step

    @functools.wraps(original)
    def training_step(self, model, inputs, *args, **kwargs):
        if not _enabled():
            return original(self, model, inputs, *args, **kwargs)
        start = time.perf_counter()
        result = original(self, model, inputs, *args, **kwargs)
        try:
            # Wall time only: forward and backward both run inside training_step and are not split.
            step_time = time.perf_counter() - start
            loss = result.get("loss") if isinstance(result, dict) else result
            loss = float(loss.item() if hasattr(loss, "item") else loss)
            # training_step returns this micro-batch's share of the optimizer-step loss; times GA,
            # the mean over one accumulation window equals the loss Trainer logs.
            loss *= getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1) or 1
            batch_size = _batch_size(inputs)
            lr = _learning_rate(self)
            get_stats_collector().training_stats.record_batch(
                step = getattr(self.state, "global_step", 0),
                batch_size = batch_size,
                step_time = step_time,
                loss = loss,
                learning_rate = lr,
            )
            registry = active_registry()
            if registry is not None:
                m = registry["training"]
                m["training_steps_total"].inc()
                m["training_samples_total"].inc(batch_size)
                m["training_loss"].set(loss)
                m["learning_rate"].set(lr)
                m["step_time_seconds"].observe(step_time)
                m["batch_size"].observe(batch_size)
        except Exception:
            pass
        return result

    training_step._unsloth_metrics_wrapped = True
    Trainer.training_step = training_step
