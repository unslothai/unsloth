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
#
# Adapted from Q-GaLore (https://github.com/VITA-Group/Q-GaLore)
# Original paper: "Q-GaLore: Quantized GaLore with INT4 Projection and
# Layer-Adaptive Low-Rank Gradients" (arXiv:2407.08296)

import torch
from typing import Optional, List

from .q_galore_projector import (
    GaLoreProjector,
    _quantize,
    _quantize_stochastic,
    _dequantize,
)

__all__ = ["QGaLoreAdamW8bit", "install_weight_quant_hooks"]

try:
    import bitsandbytes.functional as bnb_F
    from bitsandbytes.optim.optimizer import Optimizer2State
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    # Fallback base so the module can still be imported.
    Optimizer2State = torch.optim.Optimizer


def _require_bnb():
    if not _HAS_BNB:
        raise ImportError(
            "Unsloth: Q-GaLore requires bitsandbytes. Install it with: pip install bitsandbytes"
        )


class QGaLoreAdamW8bit(Optimizer2State):
    """AdamW with 8-bit states, GaLore low-rank gradient projection, and optional
    INT8 weight quantization. Three memory-saving techniques:

    1. **8-bit optimizer states** (bitsandbytes): Adam moments in 8-bit (~4x less).
    2. **GaLore low-rank projection**: gradients projected to a low-rank subspace
       for the step, then back; the projection matrix can be INT4-quantized.
    3. **INT8 weight quantization**: weights stored in INT8 with stochastic
       rounding (~2x less) for eligible layers.

    Param group keys: GaLore uses ``rank``, ``update_proj_gap``, ``scale``,
    ``proj_type``, ``quant``, ``quant_group_size``, ``quant_n_bit``,
    ``cos_threshold``, ``gamma_proj``, ``queue_size``; weight quantization uses
    ``weight_quant``, ``stochastic_round``, ``weight_group_size``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
        percentile_clipping: int = 100,
        block_wise: bool = True,
        is_paged: bool = False,
    ):
        _require_bnb()
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # optim_bits
            None,  # args
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged = is_paged,
        )

    @torch.no_grad()
    def step(self, closure = None):
        """Single optimization step. For each ``rank``-group parameter: (1)
        dequantize INT8 weight if ``weight_quant``; (2) project gradient to
        low-rank; (3) 8-bit Adam update in low-rank space; (4) project back and
        add to the saved weight; (5) re-quantize to INT8 if ``weight_quant``."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                has_weight_quant = self._has_weight_quant(p, group)

                if has_weight_quant:
                    if p._q_scales is not None:
                        float_weight = _dequantize(
                            p._q_data,
                            p._q_scales,
                            p._q_zeros,
                            p._q_shape,
                        )
                        p.data = float_weight
                    # else: first step, weights are still float — skip dequantize

                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            rank = group["rank"],
                            update_proj_gap = group.get("update_proj_gap", 200),
                            scale = group.get("scale", 0.25),
                            proj_type = group.get("proj_type", "std"),
                            quant = group.get("quant", False),
                            group_size = group.get("quant_group_size", -1),
                            n_bit = group.get("quant_n_bit", 4),
                            cos_threshold = group.get("cos_threshold", 0.4),
                            gamma_proj = group.get("gamma_proj", 2.0),
                            queue_size = group.get("queue_size", 5),
                        )

                    # Disable weight decay here; reapplied manually after project-back.
                    if "weight_decay" in group and group["weight_decay"] > 0:
                        group["_wd_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    grad = state["projector"].project(p.grad, state["step"])

                    # Zero p.data so the 8-bit update writes the pure delta.
                    p._saved_data = p.data.clone()
                    p.data = torch.zeros_like(grad, dtype = p.data.dtype, device = p.data.device)
                    p.grad = grad

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)

                if "rank" in group:
                    # p.data holds the update in low-rank space; project back and add.
                    p.data = p._saved_data.add_(state["projector"].project_back(p.data))

                    # Re-apply decoupled weight decay using pre-update weights
                    if "_wd_saved" in group:
                        p.data.add_(
                            p.data,
                            alpha = -group["lr"] * group["_wd_saved"],
                        )
                        group["weight_decay"] = group["_wd_saved"]
                        del group["_wd_saved"]

                    del p._saved_data

                if has_weight_quant:
                    float_data = p.data
                    stochastic = group.get("stochastic_round", True)
                    gsize = group.get("weight_group_size", 128)
                    quant_fn = _quantize_stochastic if stochastic else _quantize
                    q, scales, zeros, shape = quant_fn(float_data, q_group_size = gsize)
                    p._q_data = q.to(p.data.device)
                    p._q_scales = scales
                    p._q_zeros = zeros
                    p._q_shape = shape
                    # Scalar placeholder frees float memory; install_weight_quant_hooks
                    # forward pre-hook dequantizes before the next forward pass.
                    p.data = torch.empty(1, dtype = p.data.dtype, device = p.data.device)

                state["step"] += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return loss

    @staticmethod
    def _has_weight_quant(p: torch.Tensor, group: dict) -> bool:
        """Check if this parameter uses INT8 weight quantization."""
        return (
            group.get("weight_quant", False)
            and hasattr(p, "_q_scales")  # tag set by init_weight_quantization()
        )

    @staticmethod
    def init_weight_quantization(
        model: torch.nn.Module,
        param_groups: list,
        group_size: int = 128,
        stochastic: bool = True,
    ) -> None:
        """Tag eligible parameters with INT8 quantization metadata for ``step()``.

        **Weights are NOT converted to uint8 here** — they stay float so the first
        forward/backward runs correctly; actual quantization happens at the end of
        the first ``step()``.
        """
        weight_quant_params = set()
        for group in param_groups:
            if group.get("weight_quant", False):
                for p in group["params"]:
                    weight_quant_params.add(id(p))

        for name, p in model.named_parameters():
            if id(p) in weight_quant_params:
                # Tag only; first step() quantizes after the update. Dummy
                # scales/zeros keep _has_weight_quant() True on the first step.
                p._q_scales = None
                p._q_zeros = None
                p._q_shape = p.data.shape
                p._stochastic_round = stochastic
                p._weight_group_size = group_size


def _weight_quant_pre_hook(module, args):
    """Forward pre-hook: dequantize INT8 weights to float before forward."""
    for p in module.parameters(recurse = False):
        if hasattr(p, "_q_scales") and p._q_scales is not None:
            float_weight = _dequantize(
                p._q_data,
                p._q_scales,
                p._q_zeros,
                p._q_shape,
            )
            p.data = float_weight.to(p.data.device)


def install_weight_quant_hooks(model: torch.nn.Module) -> list:
    """Register forward pre-hooks on modules whose weights are INT8-quantized.

    Returns a list of hook handles so the caller can remove them if needed.
    """
    handles = []
    for module in model.modules():
        has_quant_param = any(hasattr(p, "_q_scales") for p in module.parameters(recurse = False))
        if has_quant_param:
            h = module.register_forward_pre_hook(_weight_quant_pre_hook)
            handles.append(h)
    return handles


# Default transformer layers that use GaLore.
_DEFAULT_GALORE_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def make_q_galore_param_groups(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    rank: int = 256,
    update_proj_gap: int = 200,
    scale: float = 0.25,
    proj_quant: bool = True,
    proj_quant_group_size: int = -1,
    proj_quant_n_bit: int = 4,
    weight_quant: bool = False,
    stochastic_round: bool = True,
    weight_group_size: int = 128,
    cos_threshold: float = 0.4,
    gamma_proj: float = 2.0,
    queue_size: int = 5,
    target_modules: Optional[List[str]] = None,
) -> list:
    """Build param groups for :class:`QGaLoreAdamW8bit`, returning
    ``[galore_group, non_galore_group]``.

    Parameters matching ``target_modules`` (or the default attention/MLP
    projection names) go in the GaLore group; all other trainable params go in
    the non-GaLore group.
    """
    targets = set(target_modules) if target_modules is not None else _DEFAULT_GALORE_TARGETS

    galore_params = []
    non_galore_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Exclude 1-D params (biases, norms): GaLoreProjector.project needs 2-D grads.
        name_parts = name.split(".")
        is_galore = param.dim() >= 2 and any(t in name_parts for t in targets)

        if is_galore:
            galore_params.append(param)
        else:
            non_galore_params.append(param)

    groups = []

    if galore_params:
        groups.append(
            {
                "params": galore_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "rank": rank,
                "update_proj_gap": update_proj_gap,
                "scale": scale,
                "proj_type": "std",
                "quant": proj_quant,
                "quant_group_size": proj_quant_group_size,
                "quant_n_bit": proj_quant_n_bit,
                "weight_quant": weight_quant,
                "stochastic_round": stochastic_round,
                "weight_group_size": weight_group_size,
                "cos_threshold": cos_threshold,
                "gamma_proj": gamma_proj,
                "queue_size": queue_size,
            }
        )

    if non_galore_params:
        groups.append(
            {
                "params": non_galore_params,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )

    return groups
