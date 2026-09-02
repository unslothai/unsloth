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

"""Layer-wise learning-rate decay: lr(i) = base_lr * decay ** (num_layers - 1 - i).

Stdlib-only so the grouping logic is testable without importing unsloth.
"""

import re

__all__ = ["get_layer_index", "make_layerwise_lr_param_groups", "check_layerwise_lr_compat"]

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)\.")


def get_layer_index(name):
    match = _LAYER_INDEX_RE.search(name)
    return int(match.group(1)) if match is not None else None


def get_layer_key(name):
    # (stack prefix, index): decoder and vision towers are independent stacks.
    match = _LAYER_INDEX_RE.search(name)
    return (name[: match.start()], int(match.group(1))) if match is not None else None


def _is_embedding_param(name):
    return name.endswith("modules_to_save.default.weight")


def check_layerwise_lr_compat(layerwise_lr_decay, q_galore_config):
    if layerwise_lr_decay is not None and layerwise_lr_decay != 1.0 and q_galore_config is not None:
        raise ValueError(
            "Unsloth: layerwise_lr_decay is not supported together with q_galore_config. "
            "Set one of them to None."
        )


def make_layerwise_lr_param_groups(
    model,
    lr,
    weight_decay,
    layerwise_lr_decay,
    embedding_lr = None,
    num_layers = None,
    verbose = True,
):
    if not (0.0 < layerwise_lr_decay <= 1.0):
        raise ValueError(
            f"Unsloth: layerwise_lr_decay must be in (0, 1], got {layerwise_lr_decay}."
        )

    trainable = [
        (name, param)
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False)
    ]

    # Depth per stack (prefix before ".layers.N."), so a deeper vision tower
    # never rescales the decoder and vice-versa.
    stack_depth = {}
    for name, _ in trainable:
        key = get_layer_key(name)
        if key is not None:
            prefix, idx = key
            stack_depth[prefix] = max(stack_depth.get(prefix, 0), idx + 1)
    # num_layers (model config depth) is only unambiguous with a single stack;
    # use it as a floor there for partial adapters, else per-stack derived depth.
    if num_layers is not None and len(stack_depth) == 1:
        prefix = next(iter(stack_depth))
        stack_depth[prefix] = max(stack_depth[prefix], num_layers)
    max_depth = max(stack_depth.values(), default = 0)
    # Embeddings inherit the most-decayed (deepest-stack) rate unless overridden.
    shallowest_lr = lr * (layerwise_lr_decay ** (max_depth - 1)) if max_depth else lr

    groups = {}

    def _bucket(group_lr, param):
        group = groups.get(group_lr)
        if group is None:
            group = {"params": [], "lr": group_lr, "weight_decay": weight_decay}
            groups[group_lr] = group
        group["params"].append(param)

    for name, param in trainable:
        if _is_embedding_param(name):
            _bucket(embedding_lr if embedding_lr is not None else shallowest_lr, param)
            continue
        key = get_layer_key(name)
        if key is None:
            # Non-block params (final norm, lm_head, ...) train at the top rate.
            _bucket(lr, param)
        else:
            prefix, idx = key
            _bucket(lr * (layerwise_lr_decay ** (stack_depth[prefix] - 1 - idx)), param)

    param_groups = sorted(groups.values(), key = lambda g: g["lr"])

    if verbose:
        rates = [g["lr"] for g in param_groups]
        print(
            f"Unsloth: Layer-wise LR decay = {layerwise_lr_decay} across {len(stack_depth)} "
            f"stack(s), max depth {max_depth} -> {len(param_groups)} groups, "
            f"lr in [{min(rates):.2e}, {max(rates):.2e}]."
        )
    return param_groups
