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
from typing import Optional, List, Set

__all__ = ["make_muon_param_groups"]

NORM_CLASSES: tuple = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
)
if hasattr(torch.nn, "RMSNorm"):
    NORM_CLASSES = NORM_CLASSES + (torch.nn.RMSNorm,)


def _get_embedding_param_names(model: torch.nn.Module) -> Set[str]:
    """Return the set of parameter names that belong to ``nn.Embedding`` modules,
    including PEFT-wrapped embedding copies (``modules_to_save``).
    """
    names: Set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters():
                full = f"{module_name}.{param_name}" if module_name else param_name
                names.add(full)
    # Catch PEFT-wrapped embedding copies.
    for name, param in model.named_parameters():
        if name.endswith("modules_to_save.default.weight"):
            names.add(name)
    return names


def _get_no_decay_param_names(model: torch.nn.Module) -> Set[str]:
    """Return parameter names belonging to norm modules (zero weight decay)."""
    names: Set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, NORM_CLASSES):
            for param_name, _ in module.named_parameters():
                full = f"{module_name}.{param_name}" if module_name else param_name
                names.add(full)
    return names


def _is_muon_eligible(name: str, param: torch.Tensor, embedding_param_names: Set[str]) -> bool:
    """Check if a parameter is eligible for Muon optimization.

    Muon only applies to 2D hidden-layer weight matrices. Embedding
    matrices (``nn.Embedding``) are excluded per upstream guidance.
    """
    if param.ndim != 2 or not param.requires_grad:
        return False
    return name not in embedding_param_names


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
    embedding_lr: Optional[float] = None,
) -> tuple[list, list]:
    """Split model parameters into Muon-eligible (2D) and AdamW (1D) groups.

    Muon (Newton-Schulz orthogonalization) only applies to 2D hidden-layer
    weight matrices. Embeddings (``nn.Embedding``), biases, layernorm params,
    and all 1D/0D parameters fall back to AdamW.

    The AdamW fallback is further split into decay (weights), no-decay
    (biases, norms), and embedding groups to match HF Trainer's default
    behaviour.

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
        Weight decay for AdamW fallback (decay group). Defaults to ``weight_decay``.
    target_modules : list of str, optional
        If set, only parameters whose name contains any of these substrings
        are eligible for Muon; all others go to AdamW.
    embedding_lr : float, optional
        Separate LR for embedding params. If set, creates a dedicated
        no-decay AdamW group for embeddings. Otherwise embeddings fall
        into the decay group at the default ``adamw_lr``.

    Returns
    -------
    tuple[list, list]
        ``(muon_param_groups, adamw_param_groups)`` — each a list of
        param-group dicts suitable for ``torch.optim.Muon`` and
        ``torch.optim.AdamW`` respectively.
    """
    adamw_lr = adamw_lr or lr
    adamw_weight_decay = adamw_weight_decay if adamw_weight_decay is not None else weight_decay

    embedding_names = _get_embedding_param_names(model)
    no_decay_names = _get_no_decay_param_names(model)

    muon_params: list[torch.Tensor] = []
    adamw_decay_params: list[torch.Tensor] = []
    adamw_no_decay_params: list[torch.Tensor] = []
    adamw_embedding_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embedding = name in embedding_names or name.endswith("modules_to_save.default.weight")
        is_no_decay = name in no_decay_names or "bias" in name.lower()

        if target_modules is not None:
            if not any(mod in name for mod in target_modules):
                if is_embedding:
                    adamw_embedding_params.append(param)
                elif is_no_decay:
                    adamw_no_decay_params.append(param)
                else:
                    adamw_decay_params.append(param)
                continue

        if not is_embedding and _is_muon_eligible(name, param, embedding_names):
            muon_params.append(param)
        elif is_embedding:
            adamw_embedding_params.append(param)
        elif is_no_decay:
            adamw_no_decay_params.append(param)
        else:
            adamw_decay_params.append(param)

    muon_groups = [{"params": muon_params, "lr": lr * muon_lr_scale, "weight_decay": weight_decay}]
    adamw_groups: list[dict] = []
    if adamw_decay_params:
        adamw_groups.append(
            {"params": adamw_decay_params, "lr": adamw_lr, "weight_decay": adamw_weight_decay}
        )
    if adamw_no_decay_params:
        adamw_groups.append(
            {"params": adamw_no_decay_params, "lr": adamw_lr, "weight_decay": 0.0}
        )
    if adamw_embedding_params:
        adamw_groups.append(
            {"params": adamw_embedding_params, "lr": embedding_lr or adamw_lr, "weight_decay": 0.0}
        )

    return muon_groups, adamw_groups
