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
from collections import defaultdict
import re as _re
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


def _classify_param_names(model: torch.nn.Module) -> tuple[Set[str], Set[str]]:
    """Return ``(embedding_names, no_decay_names)`` in a single module scan.

    Norm parameter names (zero weight decay) and embedding parameter names
    (no weight decay, Muon-ineligible) are identified in one pass, including
    PEFT-wrapped copies (``modules_to_save``) and tied embeddings
    (e.g. ``lm_head.weight`` aliased to ``embed_tokens.weight``).
    """
    if any(
        param.device == torch.device("meta") for _, param in model.named_parameters()
    ):
        raise RuntimeError(
            "Unsloth: model is on meta device. Materialize the model "
            "(e.g. via FastLanguageModel.from_pretrained) before calling "
            "_classify_param_names. Meta-device tensors all report "
            "data_ptr() == 0, which breaks tied-weight detection."
        )

    embedding_names: Set[str] = set()
    no_decay_names: Set[str] = set()

    for module_name, module in model.named_modules():
        mod_prefix = f"{module_name}." if module_name else ""
        if isinstance(module, torch.nn.Embedding):
            for param_name, _ in module.named_parameters():
                embedding_names.add(f"{mod_prefix}{param_name}")
        elif (
            isinstance(module, NORM_CLASSES) or "norm" in type(module).__name__.lower()
        ):
            for param_name, _ in module.named_parameters():
                no_decay_names.add(f"{mod_prefix}{param_name}")

    # Catch PEFT-wrapped copies (modules_to_save) in a second pass.
    # Match any adapter name (not just "default" — PEFT uses the convention
    # modules_to_save.<adapter_name>.weight|bias).
    for name, _ in model.named_parameters():
        if "modules_to_save." not in name:
            continue
        parent_name = (
            name.rsplit(".modules_to_save", 1)[0] if ".modules_to_save" in name else ""
        )
        try:
            parent = model.get_submodule(parent_name)
        except (AttributeError, KeyError):
            continue
        if isinstance(parent, torch.nn.Embedding):
            embedding_names.add(name)
        elif (
            isinstance(parent, NORM_CLASSES) or "norm" in type(parent).__name__.lower()
        ):
            no_decay_names.add(name)

    # Detect tied embeddings (e.g. lm_head.weight aliased to embed_tokens.weight).
    # Group by data_ptr first to be order-independent — if lm_head.weight is
    # processed before embed_tokens.weight, the alias status still propagates.
    ptr_to_names: dict[int, list[str]] = defaultdict(list)
    for name, param in model.named_parameters():
        ptr_to_names[param.data_ptr()].append(name)
    for ptr, names in ptr_to_names.items():
        if len(names) > 1:
            if any(n in embedding_names for n in names):
                embedding_names.update(names)
            if any(n in no_decay_names for n in names):
                no_decay_names.update(names)

    # Name-based fallback for custom norm classes that don't inherit from
    # torch.nn.RMSNorm (e.g. LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm,
    # GemmaRMSNorm, Phi3RMSNorm, etc.).  Matches HuggingFace Trainer's
    # convention of excluding norm weights from weight decay.
    _norm_name_pattern = _re.compile(
        r"(?:layernorm|rmsnorm|rms_norm|^norm$|\.norm\b)", _re.IGNORECASE
    )
    for name, param in model.named_parameters():
        if name not in no_decay_names and _norm_name_pattern.search(
            name.rsplit(".", 1)[0]
        ):
            no_decay_names.add(name)

    return embedding_names, no_decay_names


def _is_muon_eligible(
    name: str, param: torch.Tensor, embedding_param_names: Set[str]
) -> bool:
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
    muon_weight_decay: float,
    muon_lr_scale: float = 1.0,
    adamw_lr: Optional[float] = None,
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
    muon_weight_decay : float
        Weight decay for Muon groups.
    muon_lr_scale : float, optional
        LR multiplier for Muon groups (applied on top of ``lr``).
    adamw_lr : float, optional
        Separate LR for AdamW fallback. Defaults to ``lr``.
    adamw_weight_decay : float, optional
        Weight decay for AdamW fallback (decay group). Defaults to ``muon_weight_decay``.
    target_modules : list of str, optional
        If set, only parameters whose name contains any of these substrings
        are eligible for Muon; all others go to AdamW.
    embedding_lr : float, optional
        Separate LR for embedding params. If set, creates a dedicated
        no-decay AdamW group for embeddings (always ``weight_decay=0.0``).
        Otherwise embeddings get a dedicated no-decay AdamW group with
        ``weight_decay=0.0`` at the default ``adamw_lr``.

    Returns
    -------
    tuple[list, list]
        ``(muon_param_groups, adamw_param_groups)`` — each a list of
        param-group dicts suitable for ``torch.optim.Muon`` and
        ``torch.optim.AdamW`` respectively.
    """
    if any(
        param.device == torch.device("meta") for _, param in model.named_parameters()
    ):
        raise RuntimeError(
            "Unsloth: model is on meta device. Materialize the model "
            "(e.g. via FastLanguageModel.from_pretrained) before calling "
            "make_muon_param_groups. Meta-device tensors all report "
            "data_ptr() == 0, which breaks tied-weight detection."
        )

    adamw_lr = adamw_lr if adamw_lr is not None else lr
    adamw_weight_decay = (
        adamw_weight_decay if adamw_weight_decay is not None else muon_weight_decay
    )

    embedding_names, no_decay_names = _classify_param_names(model)

    muon_params: list[torch.Tensor] = []
    adamw_decay_params: list[torch.Tensor] = []
    adamw_no_decay_params: list[torch.Tensor] = []
    adamw_embedding_params: list[torch.Tensor] = []

    seen_params: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ptr = param.data_ptr()
        if ptr in seen_params:
            continue
        seen_params.add(ptr)

        is_embedding = name in embedding_names
        is_no_decay = name in no_decay_names or param.ndim == 1

        if target_modules is not None:
            if not any(mod in name for mod in target_modules):
                if is_embedding:
                    adamw_embedding_params.append(param)
                elif is_no_decay:
                    adamw_no_decay_params.append(param)
                else:
                    adamw_decay_params.append(param)
                continue

        if is_no_decay:
            adamw_no_decay_params.append(param)
        elif not is_embedding and _is_muon_eligible(name, param, embedding_names):
            muon_params.append(param)
        elif is_embedding:
            adamw_embedding_params.append(param)
        else:
            adamw_decay_params.append(param)

    muon_groups = [
        {
            "params": muon_params,
            "lr": lr * muon_lr_scale,
            "weight_decay": muon_weight_decay,
        }
    ]
    adamw_groups: list[dict] = []
    if adamw_decay_params:
        adamw_groups.append(
            {
                "params": adamw_decay_params,
                "lr": adamw_lr,
                "weight_decay": adamw_weight_decay,
            }
        )
    if adamw_no_decay_params:
        adamw_groups.append(
            {"params": adamw_no_decay_params, "lr": adamw_lr, "weight_decay": 0.0}
        )
    if adamw_embedding_params:
        adamw_groups.append(
            {
                "params": adamw_embedding_params,
                "lr": embedding_lr if embedding_lr is not None else adamw_lr,
                "weight_decay": 0.0,
            }
        )

    return muon_groups, adamw_groups
