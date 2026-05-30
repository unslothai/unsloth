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

from __future__ import annotations
import torch
from typing import Optional, List

__all__ = ["make_muon_param_groups"]


def _is_muon_eligible(param: torch.Tensor) -> bool:
    """Check if a parameter is eligible for Muon optimization.

    Muon only applies to 2D matrices (ndim == 2). All other shapes
    (biases, layernorm weights, embeddings) fall back to AdamW.
    """
    return param.ndim == 2 and param.requires_grad


def make_muon_param_groups(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr_scale: float = 1.0,
    adamw_lr: Optional[float] = None,
    adamw_betas: tuple = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: Optional[float] = None,
    target_modules: Optional[List[str]] = None,
) -> tuple[list, list]:
    """Split model parameters into Muon-eligible (2D) and AdamW (1D) groups.

    Muon (Newton-Schulz orthogonalization) only applies to 2D matrices.
    All other parameters (embeddings, layernorms, biases, 1D) fall back
    to AdamW.

    The two lists are guaranteed to be disjoint — no parameter appears
    in both.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters to split.
    lr : float
        Base learning rate from the training arguments.
    weight_decay : float
        Weight decay for Muon groups.
    muon_lr_scale : float, optional
        LR multiplier for Muon groups (applied on top of ``lr``).
    adamw_lr : float, optional
        Separate LR for AdamW fallback. Defaults to ``lr``.
    adamw_betas : tuple, optional
        Betas for AdamW fallback.
    adamw_eps : float, optional
        Epsilon for AdamW fallback.
    adamw_weight_decay : float, optional
        Weight decay for AdamW fallback. Defaults to ``weight_decay``.
    target_modules : list of str, optional
        If set, only parameters whose name contains any of these substrings
        are eligible for Muon; all others go to AdamW.

    Returns
    -------
    tuple[list, list]
        ``(muon_param_groups, adamw_param_groups)`` — each a list of
        param-group dicts suitable for ``torch.optim.Muon`` and
        ``torch.optim.AdamW`` respectively.
    """
    adamw_lr = adamw_lr or lr
    adamw_weight_decay = adamw_weight_decay if adamw_weight_decay is not None else weight_decay

    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if target_modules is not None:
            if not any(mod in name for mod in target_modules):
                adamw_params.append(param)
                continue
        if _is_muon_eligible(param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_groups = [{"params": muon_params, "lr": lr * muon_lr_scale, "weight_decay": weight_decay}]
    adamw_groups = [{"params": adamw_params, "lr": adamw_lr, "weight_decay": adamw_weight_decay}]

    return muon_groups, adamw_groups
