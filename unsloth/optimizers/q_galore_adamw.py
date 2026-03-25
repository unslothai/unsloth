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
    # Provide a fallback base so the module can at least be imported.
    Optimizer2State = torch.optim.Optimizer


def _require_bnb():
    if not _HAS_BNB:
        raise ImportError(
            "Unsloth: Q-GaLore requires bitsandbytes. "
            "Install it with: pip install bitsandbytes"
        )


class QGaLoreAdamW8bit(Optimizer2State):
    """AdamW optimizer with 8-bit states, GaLore low-rank gradient projection,
    and optional INT8 weight quantization.

    This optimizer combines three memory-saving techniques:

    1. **8-bit optimizer states** (via bitsandbytes) — Adam's first and second
       moments are stored in 8-bit, reducing optimizer state memory by ~4×.

    2. **GaLore low-rank gradient projection** — gradients are projected into a
       low-rank subspace before the optimizer step, then projected back.  The
       projection matrix itself can be quantized to INT4.

    3. **INT8 weight quantization** — model weights are stored in INT8 during
       training with stochastic rounding, reducing weight memory by ~2× for
       eligible layers.

    Param group keys consumed by GaLore projection:
        ``rank``, ``update_proj_gap``, ``scale``, ``proj_type``,
        ``quant`` (projection quantization), ``quant_group_size``,
        ``quant_n_bit``, ``cos_threshold``, ``gamma_proj``, ``queue_size``

    Param group keys for weight quantization:
        ``weight_quant``, ``stochastic_round``, ``weight_group_size``
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

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure = None):
        """Perform a single optimization step.

        For each parameter that has a ``rank`` key in its param group, the
        following sequence is executed:

        1. If ``weight_quant`` is set, dequantize the INT8 weight to float.
        2. Project the gradient to low-rank via the cached ``GaLoreProjector``.
        3. Perform the 8-bit Adam update in the low-rank space.
        4. Project the update back to full rank and add to saved weight.
        5. If ``weight_quant`` is set, re-quantize the weight to INT8.
        """
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

                # --- Dequantize weight if INT8 ---
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

                # --- GaLore projection ---
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

                    # Temporarily disable weight decay for GaLore params
                    # (we apply it manually after project-back)
                    if "weight_decay" in group and group["weight_decay"] > 0:
                        group["_wd_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    grad = state["projector"].project(p.grad, state["step"])

                    # Save current weight; replace p.data with zeros so
                    # the 8-bit update writes the pure weight delta.
                    p._saved_data = p.data.clone()
                    p.data = torch.zeros_like(
                        grad, dtype = p.data.dtype, device = p.data.device
                    )
                    p.grad = grad

                # --- 8-bit Adam update ---
                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)

                # --- GaLore project-back ---
                if "rank" in group:
                    # p.data now holds the weight update in low-rank space
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

                # --- Re-quantize weight to INT8 ---
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
                    # Replace p.data with a scalar placeholder to free float memory.
                    # A forward pre-hook (install_weight_quant_hooks) will
                    # dequantize back to float before the next forward pass.
                    p.data = torch.empty(1, dtype = p.data.dtype, device = p.data.device)

                state["step"] += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        """Tag parameters for INT8 weight quantization.

        This marks eligible weights with quantization metadata so that
        the optimizer knows to quantize/dequantize them during ``step()``.
        **Weights are NOT converted to uint8 here** — they remain in float
        so that the first forward/backward pass runs correctly.  The actual
        quantization happens at the end of the first ``step()`` call.
        """
        weight_quant_params = set()
        for group in param_groups:
            if group.get("weight_quant", False):
                for p in group["params"]:
                    weight_quant_params.add(id(p))

        for name, p in model.named_parameters():
            if id(p) in weight_quant_params:
                # Store quantization metadata WITHOUT converting weights to
                # uint8.  The first optimizer.step() will quantize after the
                # update.  We store dummy scales/zeros so _has_weight_quant()
                # returns True on the first step.
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
        has_quant_param = any(
            hasattr(p, "_q_scales") for p in module.parameters(recurse = False)
        )
        if has_quant_param:
            h = module.register_forward_pre_hook(_weight_quant_pre_hook)
            handles.append(h)
    return handles


# ======================================================================
# Param-group construction helper
# ======================================================================

# Default linear layer names in transformer blocks that should use GaLore.
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
    """Build param groups suitable for :class:`QGaLoreAdamW8bit`.

    Parameters matching ``target_modules`` (or the default set of attention
    and MLP projection names) are placed in the GaLore group.  All other
    trainable parameters go into the non-GaLore group.

    Args:
        model: The model whose parameters to partition.
        lr: Learning rate for all parameter groups.
        weight_decay: Weight decay coefficient.
        rank: GaLore projection rank.
        update_proj_gap: Steps between SVD recomputations.
        scale: Scaling factor for project-back.
        proj_quant: Quantize projection matrices.
        proj_quant_group_size: Group size for projection quantization.
        proj_quant_n_bit: Bit-width for projection quantization.
        weight_quant: Enable INT8 weight quantization for GaLore params.
        stochastic_round: Use stochastic rounding for weight quantization.
        weight_group_size: Group size for weight quantization.
        cos_threshold: Cosine similarity threshold for adaptive scheduling.
        gamma_proj: Multiplier for update_proj_gap when subspace is stable.
        queue_size: Rolling window size for stability tracking.
        target_modules: Module name substrings to match for GaLore.  If None,
            uses the default set of attention/MLP projection names.

    Returns:
        List of two param group dicts: ``[galore_group, non_galore_group]``.
    """
    targets = (
        set(target_modules) if target_modules is not None else _DEFAULT_GALORE_TARGETS
    )

    galore_params = []
    non_galore_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if any target module name appears as a component in the param name.
        # Exclude 1-D parameters (biases, norms) because GaLoreProjector.project
        # requires 2-D gradients.
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
