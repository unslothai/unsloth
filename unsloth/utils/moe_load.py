# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in UNSLOTH_LOG_MOE_LOAD=N: log hard top-k MoE routing load every N optimizer steps.

Hooks the outer model's router logits, so a checkpoint recompute (which reruns layers, never the
outer forward) counts each micro batch once."""

__all__ = [
    "MoERoutingLoad",
    "UnslothMoELoadCallback",
    "add_moe_load_callback",
    "moe_load_interval",
]

import os
import re

import torch
from transformers import TrainerCallback

_ENV = "UNSLOTH_LOG_MOE_LOAD"


def moe_load_interval(value = None) -> int:
    value = os.environ.get(_ENV, "0") if value is None else value
    try:
        every = int(str(value).strip())
    except ValueError:
        every = 1 if str(value).strip().lower() in ("true", "yes", "on") else 0
    return max(every, 0)


def _positive_int(*sources_and_names):
    for source, names in sources_and_names:
        for name in names:
            value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _text_config(model):
    config = getattr(model, "config", None)
    get_text = getattr(config, "get_text_config", None)
    return get_text() if callable(get_text) else config


def _moe_layers(model):
    """Layer index and top-k of every MoE block (a module with experts and a gate / router), in order."""
    config = _text_config(model)
    layers = []
    for name, block in model.named_modules():
        if not hasattr(block, "experts"):
            continue
        router = next(
            (
                getattr(block, n)
                for n in ("gate", "router")
                if isinstance(getattr(block, n, None), torch.nn.Module)
            ),
            None,
        )
        if router is None:
            continue
        found = re.findall(r"\.(\d+)\.", f".{name}.")
        top_k = _positive_int(
            (block, ("top_k", "num_experts_per_tok")),
            (router, ("top_k", "num_experts_per_tok")),
            (config, ("num_experts_per_tok", "moe_top_k", "top_k")),
        )
        layers.append((int(found[-1]) if found else len(layers), top_k))
    return layers


def _router_logits(output):
    logits = getattr(output, "router_logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("router_logits")
    if isinstance(logits, torch.Tensor):
        logits = (logits,)
    if not isinstance(logits, (tuple, list)):
        return None
    logits = [x for x in logits if torch.is_tensor(x) and x.is_floating_point() and x.dim() >= 2]
    return logits or None


class MoERoutingLoad:
    """Per expert and layer hard top-k counts and softmax mass of graph-building forwards."""

    def __init__(self, model):
        self.model = model
        layers = _moe_layers(model)
        self.layers = [index for index, _ in layers]
        top_ks = [k for _, k in layers if k]
        self.top_k = _positive_int((_text_config(model), ("num_experts_per_tok", "moe_top_k"))) or (
            top_ks[0] if top_ks else None
        )
        # Per layer top-k when every MoE block declares one (layers may route to different k / expert counts).
        self.top_ks = [k for _, k in layers] if len(top_ks) == len(layers) else None
        self.active = False
        self.handle = None
        self.reset()

    def __len__(self):
        return len(self.layers)

    def reset(self):
        self.hard = None  # [layers, experts] int64 routed (token, slot) pairs; a per layer list if expert counts differ
        self.soft = (
            None  # [layers, experts] float32 softmax mass; a per layer list if expert counts differ
        )
        self.tokens = None  # [layers] int64, counted tokens
        self.forwards = 0

    def attach(self):
        if self.handle is None and len(self) and self.top_k:
            self.handle = self.model.register_forward_hook(self._hook, with_kwargs = True)
        return self.handle is not None

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.active = False

    def start(self):
        if not self.active:
            self.reset()
            self.active = True

    def stop(self):
        self.active = False

    def _hook(self, module, args, kwargs, output):
        # Only training forwards (a graph is built); eval, generation and reference passes are skipped.
        if not self.active or not torch.is_grad_enabled():
            return
        logits = _router_logits(output)
        if logits is None:
            return
        mask = kwargs.get("attention_mask")
        self.count(logits, mask if torch.is_tensor(mask) and mask.dim() == 2 else None)

    @torch.no_grad()
    def count(
        self,
        router_logits,
        attention_mask = None,
    ):
        top_ks = (
            self.top_ks
            if self.top_ks is not None and len(self.top_ks) == len(router_logits)
            else None
        )
        hard = []
        soft = []
        tokens = []
        for layer, logits in enumerate(router_logits):
            logits = logits.detach()
            experts = logits.shape[-1]
            logits = logits.reshape(-1, experts)
            weight = None
            if attention_mask is not None and attention_mask.numel() == logits.shape[0]:
                weight = (attention_mask.reshape(-1) != 0).to(device = logits.device)
            top_k = min(top_ks[layer] if top_ks is not None else self.top_k, experts)
            index = logits.topk(top_k, dim = -1).indices
            ones = (
                torch.ones_like(index)
                if weight is None
                else weight[:, None].expand_as(index).to(index.dtype)
            )
            counts = torch.zeros(experts, dtype = torch.int64, device = logits.device)
            counts.scatter_add_(0, index.reshape(-1), ones.reshape(-1))
            probs = logits.float().softmax(dim = -1)
            if weight is not None:
                probs = probs * weight[:, None]
            hard.append(counts)
            soft.append(probs.sum(dim = 0))
            tokens.append(
                torch.full((), logits.shape[0], dtype = torch.int64, device = logits.device)
                if weight is None
                else weight.sum()
            )
        device = hard[0].device
        hard = [h.to(device) for h in hard]
        soft = [x.to(device) for x in soft]
        if len({h.numel() for h in hard}) == 1:
            hard, soft = torch.stack(hard), torch.stack(soft)
        # else kept as per layer lists: layers with different expert counts do not stack into [layers, experts].
        tokens = torch.stack([t.to(device = device, dtype = torch.int64) for t in tokens])
        if self.hard is None or [h.shape for h in self.hard] != [h.shape for h in hard]:
            self.hard, self.soft, self.tokens = hard, soft, tokens
        else:
            for total, new in zip(self.hard, hard):
                total.add_(new)
            for total, new in zip(self.soft, soft):
                total.add_(new)
            self.tokens.add_(tokens)
        self.forwards += 1

    def stats(self, per_layer = True) -> dict:
        """Max / mean load per layer and the uncoefficiented aux loss; one host sync."""
        if self.hard is None:
            return {}
        layers = len(self.hard)
        hard_layers = [h.to(torch.float64) for h in self.hard]
        soft_layers = [x.to(torch.float64) for x in self.soft]
        hard_ratio = torch.stack([h.max() / h.mean().clamp_min(1e-12) for h in hard_layers])
        soft_ratio = torch.stack([x.max() / x.mean().clamp_min(1e-12) for x in soft_layers])
        total = self.tokens.sum().to(torch.float64).clamp_min(1)
        if len({h.numel() for h in hard_layers}) == 1:
            # The router aux loss over all layers concatenated, as transformers' load_balancing_loss_func.
            hard, soft = torch.stack(hard_layers), torch.stack(soft_layers)
            aux = hard.shape[-1] * (hard.sum(0) * soft.sum(0)).sum() / (total * total)
        else:
            # Unequal expert counts: no shared expert axis, so the mean of each layer's own aux loss.
            per_token = self.tokens.to(torch.float64).clamp_min(1)
            aux = torch.stack(
                [
                    h.numel() * (h * x).sum() / (n * n)
                    for h, x, n in zip(hard_layers, soft_layers, per_token)
                ]
            ).mean()
        assignments = sum(h.sum() for h in hard_layers)
        packed = torch.cat(
            [
                hard_ratio,
                soft_ratio,
                aux.reshape(1),
                self.tokens[:1].to(torch.float64),
                assignments.reshape(1),
            ]
        ).tolist()
        hard_ratio, soft_ratio = packed[:layers], packed[layers : 2 * layers]
        aux, tokens, assignments = packed[2 * layers :]
        out = {
            "moe_load_max_mean": max(hard_ratio),
            "moe_load_max_mean_avg": sum(hard_ratio) / layers,
            "moe_soft_max_mean": max(soft_ratio),
            "moe_soft_max_mean_avg": sum(soft_ratio) / layers,
            "moe_aux_loss": aux,
            "moe_load_tokens": int(tokens),
            "moe_load_assignments": int(assignments),
        }
        if per_layer:
            names = self.layers if len(self.layers) == layers else range(layers)
            for name, value in zip(names, hard_ratio):
                out[f"moe_load_max_mean/layer_{name}"] = value
        return out


class UnslothMoELoadCallback(TrainerCallback):
    """Counts routes of every `every`-th optimizer step into the next training log."""

    def __init__(
        self,
        every = 50,
        per_layer = True,
    ):
        self.every = max(int(every), 1)
        self.per_layer = per_layer
        self.counter = None
        self._pending = False
        self._warned = False

    def on_train_begin(
        self,
        args,
        state,
        control,
        model = None,
        **kwargs,
    ):
        if model is None or self.counter is not None:
            return
        counter = MoERoutingLoad(model)
        if counter.attach():
            self.counter = counter

    def on_step_begin(self, args, state, control, **kwargs):
        if self.counter is not None and (state.global_step + 1) % self.every == 0:
            self.counter.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.counter is not None and self.counter.active:
            self.counter.stop()
            self._pending = True

    def on_log(
        self,
        args,
        state,
        control,
        logs = None,
        **kwargs,
    ):
        if not self._pending or self.counter is None or logs is None or "loss" not in logs:
            return
        self._pending = False
        extra = self.counter.stats(per_layer = self.per_layer)
        if not extra:
            if not self._warned:
                self._warned = True
                print(
                    "Unsloth: UNSLOTH_LOG_MOE_LOAD found no router logits in the model outputs; set "
                    "model.config.output_router_logits = True (the router aux loss) to log the MoE load."
                )
            return
        logs.update(extra)
        history = getattr(state, "log_history", None)
        if history and history[-1].get("step") == state.global_step and "loss" in history[-1]:
            history[-1].update(extra)

    def on_train_end(self, args, state, control, **kwargs):
        if self.counter is not None:
            self.counter.detach()
            self.counter = None


def add_moe_load_callback(
    trainer,
    every = None,
    **kwargs,
):
    """Register an UnslothMoELoadCallback first, so reporting callbacks see the fields it adds."""
    every = moe_load_interval() if every is None else every
    callbacks = trainer.callback_handler.callbacks
    for index, callback in enumerate(callbacks):
        if isinstance(callback, UnslothMoELoadCallback):
            callbacks.insert(0, callbacks.pop(index))
            return callback
    callback = UnslothMoELoadCallback(every = every if every and every > 0 else 50, **kwargs)
    callbacks.insert(0, callback)
    return callback
