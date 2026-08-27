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

import copy
import logging
import os
import psutil
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, List
from functools import wraps
import torch

import trl
import inspect
from trl import SFTTrainer

# why: bypass partially-initialised unsloth ns during _gpu_init load
from .models._utils import is_bfloat16_supported
from unsloth.utils import (
    configure_padding_free,
    configure_sample_packing,
    enable_padding_free_metadata,
    enable_sample_packing,
)
from unsloth.utils.packing import patch_hybrid_linear_attention_varlen
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator as _UnslothVisionDataCollatorBase,
)
from unsloth.models.vision import check_dataset_for_missing_videos
from unsloth_zoo.hf_utils import get_transformers_model_type
from unsloth_zoo.utils import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
    "QGaloreConfig",
    "MuonConfig",
    "_MuonAdamWChained",
    "check_dataset_for_missing_videos",
]

logger = logging.getLogger(__name__)


class UnslothVisionDataCollator(_UnslothVisionDataCollatorBase):
    """
    Drop-in zoo collator that validates local video paths on every batch
    (deduped across batches), applying formatting_func first so formatter-made
    paths are checked too. Raises FileNotFoundError on missing files instead
    of silently training on empty video tensors (issue #5085).
    """

    __slots__ = ("_checked_video_paths",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._checked_video_paths = set()

    def __call__(self, examples):
        formatting_func = self.formatting_func
        if formatting_func is not None:
            examples = [formatting_func(example) for example in examples]

        check_dataset_for_missing_videos(
            examples,
            raise_error = True,
            checked = self._checked_video_paths,
        )

        if formatting_func is None:
            return super().__call__(examples)

        # why: base __call__ would reapply formatting_func; applied above. A
        # per-call shallow view shares every other attribute by reference, so a
        # concurrent caller can never observe formatting_func blanked on self.
        view = copy.copy(self)
        view.formatting_func = None
        return super(UnslothVisionDataCollator, view).__call__(examples)


_AUTO_PADDING_FREE_ENV_DISABLED = os.environ.get(
    "UNSLOTH_DISABLE_AUTO_PADDING_FREE", ""
).strip().lower() in {"1", "true", "yes", "on"}

PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}
# Hybrid linear-attention / state-space models (Qwen3.5, Qwen3-Next, ...) carry a
# recurrent gated-delta state plus a causal conv1d that leak across sequence
# boundaries once packing flattens the batch. Detected structurally by
# _is_hybrid_linear_attention_model, not by model name.


def _should_pack(config) -> bool:
    if config is None or not getattr(config, "packing", False):
        return False
    return not getattr(config, "_unsloth_disable_auto_packing", False)


def _should_auto_padding_free(config) -> bool:
    if config is None or _AUTO_PADDING_FREE_ENV_DISABLED or getattr(config, "packing", False):
        return False
    return getattr(config, "padding_free", None) is None


def _disable_sample_packing(config):
    if config is None:
        return
    for attr, value in (("packing", False), ("padding_free", False)):
        if hasattr(config, attr):
            setattr(config, attr, value)
    if hasattr(config, "remove_unused_columns"):
        setattr(config, "remove_unused_columns", True)
    setattr(config, "_unsloth_disable_auto_packing", True)


_AUTO_PACK_SKIP_MESSAGES = (
    "packing is not supported",
    "padding-free training",
    "passing a custom data collator",
)


def _should_skip_auto_packing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(msg in message for msg in _AUTO_PACK_SKIP_MESSAGES)


def _should_skip_auto_padding_free_error(exc: Exception) -> bool:
    """Net for a TRL that words the padding-free / `max_length` guard differently.

    rl.py already handles the known wording; both terms must appear here so an
    unrelated ValueError still propagates.
    """
    message = str(exc).lower()
    return "padding_free" in message and "max_length" in message


def _bound_splits(original_init, args, kwargs):
    """`(train_dataset, eval_dataset)` as the wrapped `__init__` will see them.

    Bound through the signature rather than indexed positionally: TRL has moved
    these parameters between releases, and a hardcoded index silently reads the
    data collator on the version that did.
    """
    try:
        bound = inspect.signature(original_init).bind_partial(None, *args, **kwargs)
    except Exception:
        return kwargs.get("train_dataset"), kwargs.get("eval_dataset")
    return bound.arguments.get("train_dataset"), bound.arguments.get("eval_dataset")


def _cap_is_enforceable_without_padding_free(config, train, evals) -> bool:
    """Whether dropping padding-free actually leaves something enforcing the cap.

    This fallback only runs when the exact source-text match in `rl.py` missed
    TRL's guard, which means the generated pre-tokenized truncation block was
    never inserted either. Turning `padding_free` off keeps `max_length` for
    TRL's collator, and that collator does not truncate: rows that already carry
    `input_ids` reach the model at full length. So the retry would convert a
    hard error into a silently uncapped run, which is strictly worse.

    Raw splits are fine -- prep tokenizes them under `max_length`. Only rows that
    are already tokenized are at risk, so scan for those and let the original
    error propagate when any is over.

    A packed split is fine too: the packer owns the overflow and chunks it, so
    an overlength row there is not an unenforced cap. `packing` and
    `eval_packing` are resolved separately because they can differ, and the
    generated exact-match path already excludes eval-packed splits from its own
    scan -- scanning them here refused a configuration that path accepts.
    """
    cap = getattr(config, "max_length", None)
    if not cap:
        return True
    try:
        from unsloth.models.rl import pretokenized_within_cap, splits_within_cap
    except Exception:
        return True  # nothing to check against; do not invent a failure
    packing = bool(getattr(config, "packing", False))
    # TRL's own default: `eval_packing = None` means "whatever `packing` is".
    eval_packing = getattr(config, "eval_packing", None)
    eval_packing = packing if eval_packing is None else bool(eval_packing)
    if not packing and not pretokenized_within_cap(train, cap):
        return False
    return eval_packing or splits_within_cap(evals, cap)


def _disable_padding_free(config):
    if config is None:
        return
    if hasattr(config, "padding_free"):
        setattr(config, "padding_free", False)


_VISION_DATASET_KEYS = frozenset(
    {
        "image",
        "images",
        "image_grid_thw",
        "image_position_ids",
        "image_sizes",
        "mm_token_type_ids",
        "pixel_attention_mask",
        "pixel_position_ids",
        "pixel_values",
        "pixel_values_videos",
        "video",
        "videos",
        "video_grid_thw",
    }
)


def _is_vlm_config(config, model_types = ()) -> bool:
    if any(
        hasattr(config, attr)
        for attr in ("vision_config", "img_processor", "image_token_index", "projector_config")
    ):
        return True

    architectures = getattr(config, "architectures", None) or ()
    try:
        from transformers.models.auto import modeling_auto

        mappings = (
            getattr(modeling_auto, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", {}) or {},
            getattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", {}) or {},
        )
        registry_types = set().union(*(mapping.keys() for mapping in mappings))
        registry_classes = set().union(*(mapping.values() for mapping in mappings))
        config_types = set(model_types or ())
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            config_types.add(model_type)
        if not config_types.isdisjoint(registry_types) or any(
            architecture in registry_classes for architecture in architectures
        ):
            return True
    except Exception:
        pass
    return any(
        isinstance(architecture, str) and architecture.endswith("ForVisionText2Text")
        for architecture in architectures
    )


def _is_vision_dataset(dataset, *, unknown_is_vision = False) -> bool:
    if dataset is None:
        return False
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        return not _VISION_DATASET_KEYS.isdisjoint(column_names)
    # Unknown-schema streams cannot be safely probed without potentially dropping a sample.
    return unknown_is_vision


def _is_vision_eval_dataset(dataset, *, unknown_is_vision = False) -> bool:
    if isinstance(dataset, dict):
        return any(
            _is_vision_dataset(split, unknown_is_vision = unknown_is_vision)
            for split in dataset.values()
        )
    return _is_vision_dataset(dataset, unknown_is_vision = unknown_is_vision)


_HYBRID_CONFIG_MARKERS = (
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "full_attention_interval",
)


def _is_hybrid_linear_attention_model(model) -> bool:
    """Detect models mixing linear-attention / state-space mixers (gated-delta,
    Mamba-style) with a causal conv1d, e.g. Qwen3.5 / Qwen3-Next. Packing and
    padding-free flatten the batch, and those recurrent + conv ops leak state
    across sequence boundaries, so they must not be packed. Uses composite
    structural evidence rather than a model-name match."""
    if model is None:
        return False

    # Config-level: explicit hybrid layer schedule or linear-attn markers.
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if config is None:
            continue
        layer_types = getattr(config, "layer_types", None)
        if isinstance(layer_types, (list, tuple)) and any(
            isinstance(t, str) and "linear_attention" in t for t in layer_types
        ):
            return True
        if any(hasattr(config, marker) for marker in _HYBRID_CONFIG_MARKERS):
            return True

    # Module-level: a mixer carrying a recurrent gated-delta op plus a conv1d.
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return False
    seen = set()
    for _, module in named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        cls = type(module).__name__
        if not (
            cls.endswith("GatedDeltaNet") or "LinearAttention" in cls or cls.endswith("Mamba2Mixer")
        ):
            continue
        has_recurrent = any(
            hasattr(module, attr)
            for attr in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule", "A_log")
        )
        if has_recurrent and hasattr(module, "conv1d"):
            return True
    return False


def _resolve_string_model_config(model_name, config_arg):
    """TRL materializes a string ``model=`` inside ``__init__``; resolve its config
    up front so the packing guards run before the dataset is packed. Best-effort:
    returns None if the config cannot be loaded."""
    try:
        from transformers import AutoConfig

        init_kwargs = getattr(config_arg, "model_init_kwargs", None) or {}
        # why: forward auth + cache args too. Dropping token/use_auth_token made a
        # private hybrid fail to load (resolve as None) -> treated as non-hybrid ->
        # packing enabled without the shim even though TRL later loads it with the token.
        forward = {
            key: init_kwargs[key]
            for key in (
                "trust_remote_code",
                "revision",
                "subfolder",
                "token",
                "use_auth_token",
                "cache_dir",
                "code_revision",
            )
            if key in init_kwargs
        }
        # why: TRL merges top-level args.trust_remote_code into the load via setdefault
        # before create_model_from_path, so honor it here (model_init_kwargs wins), else
        # a remote-code hybrid with SFTConfig(trust_remote_code=True) resolves as None
        # and skips the guard.
        top_level_trust_remote_code = getattr(config_arg, "trust_remote_code", None)
        if top_level_trust_remote_code is not None:
            forward.setdefault("trust_remote_code", top_level_trust_remote_code)
        return AutoConfig.from_pretrained(model_name, **forward)
    except Exception:
        return None


def _chunked_loss_bypasses_forward(config) -> bool:
    """TRL's default ``loss_type="chunked_nll"`` patches the model forward and calls
    the backbone directly, so a forward wrapper never runs. Detect it so hybrid
    packing stays on the padded path instead of silently skipping the varlen shim."""
    try:
        import trl.trainer.sft_trainer as _sft_trainer
    except Exception:
        return False
    if not hasattr(_sft_trainer, "_patch_chunked_ce_lm_head"):
        return False  # TRL has no chunked-CE path -> forward is not bypassed
    if getattr(config, "use_liger_kernel", False):
        return False  # liger forces loss_type="nll" -> normal forward
    return getattr(config, "loss_type", None) in (None, "chunked_nll")


# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version, ProcessorMixin

if Version(transformers_version) > Version("4.45.2"):

    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)

else:

    def unsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
                "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
            "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
        )
        return _unsloth_train(trainer)


try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


@dataclass
class QGaloreConfig:
    """Configuration for Q-GaLore optimizer integration.

    Pass an instance of this class to ``UnslothTrainingArguments`` (via
    ``q_galore_config``) to enable Q-GaLore training.
    """

    rank: int = 256
    update_proj_gap: int = 200
    scale: float = 0.25
    proj_quant: bool = True
    proj_quant_group_size: int = -1
    proj_quant_n_bit: int = 4
    weight_quant: bool = False
    stochastic_round: bool = True
    weight_group_size: int = 128
    cos_threshold: float = 0.4
    gamma_proj: float = 2.0
    queue_size: int = 5
    target_modules: Optional[List[str]] = None


@dataclass
class MuonConfig:
    """Configuration for the Muon optimizer integration.

    Muon (Momentum + Newton-Schulz orthogonalization) only applies to 2D
    hidden-layer weight matrices. Embedding matrices, biases, layernorm
    params, and all 1D/0D parameters fall back to AdamW.

    .. note::

        * Requires PyTorch >= 2.9.0.
        * ``torch.optim.Muon`` internally casts gradients to ``bfloat16``
          for the Newton-Schulz iteration, even when the model is trained
          in ``float32``. This may affect numerical stability for full-
          precision training.
        * The Muon state dict format (``{"muon": ..., "adamw": ...}``) is
          **incompatible with FSDP**. Use DDP only.

    Example:
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/Qwen3-8B",
            full_finetuning=True,
        )

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            args=UnslothTrainingArguments(
                muon_config=MuonConfig(
                    momentum=0.95,
                    ns_steps=5,
                    muon_lr_scale=1.0,
                ),
                learning_rate=1e-4,
                output_dir="./output",
            ),
            train_dataset=dataset,
        )
    """

    _ADAMW_EPS_UNSET = object()
    _ADAMW_BETAS_UNSET = object()

    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: Optional[tuple[float, float, float]] = None
    muon_lr_scale: float = 1.0
    adjust_lr_fn: Optional[str] = None
    muon_eps: float = 1e-7
    muon_weight_decay: Optional[float] = None
    adamw_lr: Optional[float] = None
    adamw_betas: object = _ADAMW_BETAS_UNSET
    adamw_eps: object = _ADAMW_EPS_UNSET
    adamw_weight_decay: Optional[float] = None
    target_modules: Optional[List[str]] = None
    embedding_lr: Optional[float] = None

    def __post_init__(self):
        import warnings as _warnings

        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                f"MuonConfig requires PyTorch >= 2.9.0 (got {torch.__version__}). "
                "torch.optim.Muon is not available in this version."
            )
        if not isinstance(self.ns_steps, int):
            raise TypeError(
                f"MuonConfig.ns_steps must be an int, got {type(self.ns_steps).__name__}."
            )
        if not isinstance(self.momentum, (int, float)):
            raise TypeError(
                f"MuonConfig.momentum must be a number, got {type(self.momentum).__name__}."
            )
        if not isinstance(self.muon_eps, (int, float)):
            raise TypeError(
                f"MuonConfig.muon_eps must be a number, got {type(self.muon_eps).__name__}."
            )
        if not isinstance(self.muon_lr_scale, (int, float)):
            raise TypeError(
                f"MuonConfig.muon_lr_scale must be a number, got {type(self.muon_lr_scale).__name__}."
            )
        if self.muon_weight_decay is not None and not isinstance(
            self.muon_weight_decay, (int, float)
        ):
            raise TypeError(
                f"MuonConfig.muon_weight_decay must be a number, got {type(self.muon_weight_decay).__name__}."
            )
        if self.adamw_weight_decay is not None and not isinstance(
            self.adamw_weight_decay, (int, float)
        ):
            raise TypeError(
                f"MuonConfig.adamw_weight_decay must be a number, got {type(self.adamw_weight_decay).__name__}."
            )
        if self.ns_steps >= 100:
            raise ValueError(
                f"MuonConfig.ns_steps must be < 100, got {self.ns_steps}. "
                "PyTorch's Newton-Schulz iteration raises an error for ns_steps >= 100."
            )
        if self.ns_steps < 1:
            raise ValueError(f"MuonConfig.ns_steps must be >= 1, got {self.ns_steps}.")
        if self.ns_steps > 20:
            _warnings.warn(
                f"MuonConfig.ns_steps={self.ns_steps} is large. "
                "Each Newton-Schulz step performs a matrix multiplication. "
                "Consider reducing ns_steps (default: 5) for better performance."
            )
        if self.ns_coefficients is not None:
            if not isinstance(self.ns_coefficients, tuple) or len(self.ns_coefficients) != 3:
                raise ValueError(
                    f"MuonConfig.ns_coefficients must be a tuple of 3 floats, "
                    f"got {self.ns_coefficients}."
                )
            if not all(isinstance(v, (int, float)) for v in self.ns_coefficients):
                raise ValueError(
                    f"MuonConfig.ns_coefficients must contain only numbers, "
                    f"got {self.ns_coefficients}."
                )
        if self.momentum < 0.0:
            raise ValueError(f"MuonConfig.momentum must be >= 0.0, got {self.momentum}.")
        if self.muon_eps <= 0.0:
            raise ValueError(f"MuonConfig.muon_eps must be > 0.0, got {self.muon_eps}.")
        if self.muon_lr_scale <= 0.0:
            raise ValueError(f"MuonConfig.muon_lr_scale must be > 0.0, got {self.muon_lr_scale}.")
        if self.muon_weight_decay is not None and self.muon_weight_decay < 0.0:
            raise ValueError(
                f"MuonConfig.muon_weight_decay must be >= 0.0, got {self.muon_weight_decay}."
            )
        if self.adamw_weight_decay is not None and self.adamw_weight_decay < 0.0:
            raise ValueError(
                f"MuonConfig.adamw_weight_decay must be >= 0.0, got {self.adamw_weight_decay}."
            )
        if not isinstance(self.nesterov, bool):
            raise TypeError(
                f"MuonConfig.nesterov must be a bool, got {type(self.nesterov).__name__}."
            )
        if self.adamw_betas is not MuonConfig._ADAMW_BETAS_UNSET:
            if not isinstance(self.adamw_betas, tuple) or len(self.adamw_betas) != 2:
                raise ValueError(
                    f"MuonConfig.adamw_betas must be a tuple of 2 floats, "
                    f"got {self.adamw_betas}."
                )
        if self.adjust_lr_fn is not None:
            if not isinstance(self.adjust_lr_fn, str):
                raise TypeError(
                    f"MuonConfig.adjust_lr_fn must be a string, "
                    f"got {type(self.adjust_lr_fn).__name__}."
                )
            norm = self.adjust_lr_fn.lower()
            if norm not in ("original", "match_rms_adamw"):
                raise ValueError(
                    f"MuonConfig.adjust_lr_fn must be None, 'original', or "
                    f"'match_rms_adamw', got '{self.adjust_lr_fn}'."
                )
            self.adjust_lr_fn = norm


class UnslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        q_galore_config: Optional[QGaloreConfig] = None,
        *args,
        muon_config: Optional[MuonConfig] = None,
        **kwargs,
    ):
        self.q_galore_config = q_galore_config
        self.muon_config = muon_config
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)
        self.embedding_learning_rate = embedding_learning_rate


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {
        "non_embeddings": {},
        "embeddings": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[: -len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".") + 1 :]
            print(
                f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
            )
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


class _MuonAdamWChained(torch.optim.Optimizer):
    """Chained wrapper around a Muon optimizer and an AdamW fallback.

    Exposes a unified ``step()``, ``zero_grad()``, ``state_dict()``, and
    ``load_state_dict()`` API while delegating the actual optimization to
    the two sub-optimizers.

    ``param_groups`` is the concatenation of both sub-optimizers' groups.
    The groups are **identity-shared** — ``self.param_groups[i] is
    sub_optimizer.param_groups[i]``.  LR schedulers applied to this object
    will have their LR changes visible to sub-optimizers immediately.
    A group count check (``_assert_group_count_matches``) fires on every
    ``step()`` to detect external ``add_param_group`` calls on sub-optimizers.

    .. warning::

        ``torch.save(optimizer, ...)`` / ``pickle.dump(optimizer, ...)``
        is **not supported**. Use ``state_dict()`` / ``load_state_dict()``
        for checkpoint save/load instead.

    ``add_param_group()`` is not supported — add groups to the
    sub-optimizers directly.
    """

    def __init__(
        self,
        muon,
        adamw,
        needs_deterministic = False,
    ):
        self.muon = muon
        self.adamw = adamw
        self._needs_deterministic = needs_deterministic
        all_groups = []
        if muon is not None:
            all_groups.extend(muon.param_groups)
        if adamw is not None:
            all_groups.extend(adamw.param_groups)
        # Use only Muon defaults to prevent AdamW-specific keys (e.g. amsgrad,
        # betas, maximize, fused, capturable) from leaking into Muon param
        # groups via add_param_group's defaults-fill in the parent constructor.
        # AdamW groups are already fully constructed by their own __init__ and
        # need no additional key filling.
        muon_defaults = muon.defaults if muon is not None else {}
        self._init_done = False
        super().__init__(all_groups, muon_defaults)
        # Restore self.defaults with both Muon and AdamW keys, so downstream
        # code (LR schedulers, callbacks, custom training loops) can inspect
        # hyperparameters without them being polluted by the defaults-merge
        # which would have leaked AdamW keys into Muon param groups.
        self.defaults = {}
        if muon is not None:
            self.defaults.update(muon.defaults)
        if adamw is not None:
            self.defaults.update(adamw.defaults)
        offset = len(muon.param_groups) if muon is not None else 0
        if muon is not None:
            for i in range(len(muon.param_groups)):
                if self.param_groups[i] is not muon.param_groups[i]:
                    raise RuntimeError(
                        f"_MuonAdamWChained identity-sharing broken: "
                        f"group {i} is not the same object as muon.param_groups[{i}]. "
                        "This can happen if param_groups were deep-copied or reassigned."
                    )
        if adamw is not None:
            for i in range(len(adamw.param_groups)):
                if self.param_groups[offset + i] is not adamw.param_groups[i]:
                    raise RuntimeError(
                        f"_MuonAdamWChained identity-sharing broken: "
                        f"group {offset + i} is not the same object as adamw.param_groups[{i}]. "
                        "This can happen if param_groups were deep-copied or reassigned."
                    )
        self._init_done = True

    def add_param_group(self, param_group):
        if not getattr(self, "_init_done", False):
            return super().add_param_group(param_group)
        raise NotImplementedError(
            "add_param_group is not supported for _MuonAdamWChained. "
            "Add param groups to the sub-optimizers directly."
        )

    def _assert_group_count_matches(self):
        n_muon = len(self.muon.param_groups) if self.muon is not None else 0
        n_adamw = len(self.adamw.param_groups) if self.adamw is not None else 0
        if n_muon + n_adamw != len(self.param_groups):
            raise RuntimeError(
                f"_MuonAdamWChained group count mismatch: "
                f"muon={n_muon}, adamw={n_adamw}, "
                f"chained={len(self.param_groups)}. "
                "This can happen if add_param_group was called on a sub-optimizer."
            )

    def _muon_step_deterministic(self):
        if not self._needs_deterministic:
            self.muon.step()
            return
        was_enabled = torch.are_deterministic_algorithms_enabled()
        was_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled() if was_enabled else False
        )
        if not was_enabled or not was_warn_only:
            torch.use_deterministic_algorithms(True, warn_only = False)
        else:
            torch.use_deterministic_algorithms(True, warn_only = True)
        try:
            self.muon.step()
        finally:
            if was_enabled:
                torch.use_deterministic_algorithms(True, warn_only = was_warn_only)
            else:
                torch.use_deterministic_algorithms(False)

    def step(self, closure = None):
        self._assert_group_count_matches()
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.muon is not None:
            self._muon_step_deterministic()
        if self.adamw is not None:
            self.adamw.step()
        if closure is not None:
            return loss

    def zero_grad(self, set_to_none = True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none = set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none = set_to_none)

    MUON_STATE_DICT_VERSION = 1

    def state_dict(self):
        sd: dict = {"_muon_version": self.MUON_STATE_DICT_VERSION}
        if self.muon is not None:
            sd["muon"] = self.muon.state_dict()
        if self.adamw is not None:
            sd["adamw"] = self.adamw.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        if state_dict.get("_muon_version") != self.MUON_STATE_DICT_VERSION:
            raise RuntimeError(
                "_MuonAdamWChained state dict version mismatch: "
                f"expected version {self.MUON_STATE_DICT_VERSION}, "
                f"got {state_dict.get('_muon_version', 'missing')}. "
                "This checkpoint is not compatible with the current Muon optimizer format."
            )
        if self.muon is not None:
            muon_sd = state_dict.get("muon")
            if muon_sd is None:
                raise RuntimeError(
                    "Checkpoint has no Muon state, but current model has Muon-eligible parameters. "
                    "This can happen when the model structure changed between save and load."
                )
            self.muon.load_state_dict(muon_sd)
        if self.adamw is not None:
            adamw_sd = state_dict.get("adamw")
            if adamw_sd is None:
                raise RuntimeError(
                    "Checkpoint has no AdamW state, but current model has AdamW-eligible parameters. "
                    "This can happen when the model structure changed between save and load."
                )
            self.adamw.load_state_dict(adamw_sd)
        # Re-sync chained groups to match freshly loaded sub-optimizer groups.
        refreshed = []
        if self.muon is not None:
            refreshed.extend(self.muon.param_groups)
        if self.adamw is not None:
            refreshed.extend(self.adamw.param_groups)
        self.param_groups = refreshed
        self.defaults = {}
        if self.muon is not None:
            self.defaults.update(self.muon.defaults)
        if self.adamw is not None:
            self.defaults.update(self.adamw.defaults)

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        raise NotImplementedError(
            "_MuonAdamWChained does not support unpickling directly. "
            "Use state_dict()/load_state_dict() for checkpoint save/load. "
            "The sub-optimizers must be reconstructed from the model first."
        )

    def __repr__(self):
        def _param_count(sub):
            if sub is None:
                return 0
            return sum(p.numel() for g in sub.param_groups for p in g["params"])

        muon_str = f"Muon({_param_count(self.muon)} elements)"
        adamw_str = f"AdamW({_param_count(self.adamw)} elements)"
        return f"{type(self).__name__}({muon_str}, {adamw_str})"


class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        # --- Muon optimizer (checked first, before Q-GaLore) ---
        muon_config = getattr(self.args, "muon_config", None)
        q_galore_config = getattr(self.args, "q_galore_config", None)

        if muon_config is not None and q_galore_config is not None:
            logger.warning(
                "Unsloth: Both MuonConfig and QGaloreConfig are set. "
                "Muon takes precedence over Q-GaLore."
            )

        if muon_config is not None and self.optimizer is None:
            return self._create_muon_optimizer(muon_config)

        # --- Q-GaLore optimizer ---
        if q_galore_config is not None and self.optimizer is None:
            embedding_lr = getattr(self.args, "embedding_learning_rate", None)
            return self._create_q_galore_optimizer(q_galore_config, embedding_lr)

        # --- Embedding-LR optimizer ---
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer

    def _create_muon_optimizer(self, config: "MuonConfig"):
        """Build a mixed Muon + AdamW optimizer from a MuonConfig."""
        if self.optimizer is not None:
            raise RuntimeError(
                "Unsloth: _create_muon_optimizer called when self.optimizer is already set. "
                "This indicates a double-call (possibly from a training callback)."
            )
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                "Unsloth: torch.optim.Muon requires PyTorch >= 2.9.0.\n"
                f"Current version: {torch.__version__}\n"
                "Update with: pip install --upgrade torch"
            )

        import os as _os

        try:
            import torch.distributed as dist
        except ImportError:
            raise RuntimeError(
                "Unsloth: torch.distributed is not available. "
                "Muon optimizer requires torch.distributed for distributed training "
                "guard checks. If using a custom PyTorch build without distributed, "
                "use a standard PyTorch distribution."
            )
        needs_deterministic = False
        if dist.is_available() and dist.is_initialized():
            if _os.environ.get("UNSLOTH_MUON_DISTRIBUTED", "0") != "1":
                raise RuntimeError(
                    "Unsloth: Muon optimizer with distributed training is blocked "
                    "due to known correctness issues:\n"
                    "  1) FSDP state_dict format incompatible with Muon's nested format;\n"
                    "  2) CuBLAS non-determinism in the Newton-Schulz iteration causes "
                    "parameter divergence across ranks — this is a CORRECTNESS issue, "
                    "not just a reproducibility issue;\n"
                    "  3) DeepSpeed ZeRO may not handle Muon's orthogonalization correctly.\n"
                    "To proceed (not recommended), set UNSLOTH_MUON_DISTRIBUTED=1."
                )
            else:
                logger.warning(
                    "Unsloth: UNSLOTH_MUON_DISTRIBUTED=1 detected — Muon step will "
                    "enforce deterministic algorithms. This may reduce performance."
                )
                needs_deterministic = True

        from unsloth.optimizers.muon import make_muon_param_groups

        lr = self.args.learning_rate
        weight_decay = self.args.weight_decay  # save original for AdamW fallback
        embedding_lr = (
            config.embedding_lr
            if config.embedding_lr is not None
            else getattr(self.args, "embedding_learning_rate", None)
        )
        if embedding_lr is not None and embedding_lr == 0.0:
            logger.warning(
                "Unsloth: embedding_lr=0.0 — embeddings will receive zero gradient updates. "
                "Leave embedding_lr=None (default) to use the AdamW learning rate, "
                "or set a positive value."
            )

        muon_weight_decay = (
            config.muon_weight_decay if config.muon_weight_decay is not None else weight_decay
        )
        adamw_weight_decay = (
            config.adamw_weight_decay if config.adamw_weight_decay is not None else weight_decay
        )

        muon_groups, adamw_groups = make_muon_param_groups(
            self.model,
            lr = lr,
            muon_weight_decay = muon_weight_decay,
            muon_lr_scale = config.muon_lr_scale,
            adamw_lr = config.adamw_lr,
            adamw_weight_decay = adamw_weight_decay,
            target_modules = config.target_modules,
            embedding_lr = embedding_lr,
        )

        if PeftModel is not None and isinstance(self.model, PeftModel):
            logger.warning(
                "Unsloth Muon: PEFT/LoRA model detected. "
                "Muon will be applied to 2D adapters. "
                "Results not guaranteed — use full_finetuning=True for expected behaviour."
            )

        try:
            from bitsandbytes.nn import Params4bit
        except ImportError:
            Params4bit = None
        if Params4bit is not None:
            for _, param in self.model.named_parameters():
                if isinstance(param, Params4bit):
                    logger.warning(
                        "Unsloth Muon: 4-bit quantized model detected. "
                        "Only LoRA adapters are trainable; base weights are frozen. "
                        "Muon's orthogonalization on low-rank adapters is uncharacterized. "
                        "Use full_finetuning=True for expected Muon behaviour."
                    )
                    break

        n_muon = sum(p.numel() for g in muon_groups for p in g["params"])
        n_adamw = sum(p.numel() for g in adamw_groups for p in g["params"])
        total = n_muon + n_adamw

        logger.info(
            f"Unsloth: Muon enabled — "
            f"{n_muon:,} elements via Muon ({100*n_muon/total:.1f}%), "
            f"{n_adamw:,} elements via AdamW fallback ({100*n_adamw/total:.1f}%)"
        )
        logger.info(
            "Unsloth Muon: checkpoint format is incompatible with vanilla AdamW. "
            "See the _muon_version marker in state_dict for format detection."
        )

        muon_kwargs = dict(
            momentum = config.momentum,
            nesterov = config.nesterov,
            ns_steps = config.ns_steps,
            eps = config.muon_eps,
            ns_coefficients = config.ns_coefficients,
            adjust_lr_fn = config.adjust_lr_fn,
            weight_decay = muon_weight_decay,
        )
        # Filter None values — upstream torch.optim.Muon stores them verbatim in defaults,
        # then crashes in step() when iterating None (e.g. len(None) in _zeropower_via_newtonschulz).
        muon_kwargs = {k: v for k, v in muon_kwargs.items() if v is not None}

        has_muon_params = sum(len(g["params"]) for g in muon_groups) > 0
        if has_muon_params:
            try:
                muon_optimizer = torch.optim.Muon(muon_groups, **muon_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Unsloth: Failed to construct torch.optim.Muon (PyTorch {torch.__version__}). "
                    f"Got error: {e}"
                ) from e
        else:
            muon_optimizer = None

        if config.adamw_betas is not MuonConfig._ADAMW_BETAS_UNSET:
            adamw_betas = config.adamw_betas
        else:
            adamw_betas = (
                getattr(self.args, "adam_beta1", 0.9),
                getattr(self.args, "adam_beta2", 0.999),
            )
        if config.adamw_eps is not MuonConfig._ADAMW_EPS_UNSET:
            adamw_eps = config.adamw_eps
        else:
            adamw_eps = getattr(self.args, "adam_epsilon", 1e-8)
        adamw_lr = config.adamw_lr if config.adamw_lr is not None else lr
        adamw_kwargs = dict(
            lr = adamw_lr,
            betas = adamw_betas,
            eps = adamw_eps,
            weight_decay = adamw_weight_decay,
        )
        if adamw_groups:
            adamw_optimizer = torch.optim.AdamW(adamw_groups, **adamw_kwargs)
        else:
            adamw_optimizer = None

        self.optimizer = _MuonAdamWChained(
            muon_optimizer,
            adamw_optimizer,
            needs_deterministic = needs_deterministic,
        )
        return self.optimizer

    def _create_q_galore_optimizer(
        self,
        config: "QGaloreConfig",
        embedding_lr = None,
    ):
        """Build the Q-GaLore optimizer from a QGaloreConfig."""
        from unsloth.optimizers.q_galore_adamw import (
            QGaLoreAdamW8bit,
            make_q_galore_param_groups,
            install_weight_quant_hooks,
        )

        lr = self.args.learning_rate
        weight_decay = self.args.weight_decay

        param_groups = make_q_galore_param_groups(
            self.model,
            lr = lr,
            weight_decay = weight_decay,
            rank = config.rank,
            update_proj_gap = config.update_proj_gap,
            scale = config.scale,
            proj_quant = config.proj_quant,
            proj_quant_group_size = config.proj_quant_group_size,
            proj_quant_n_bit = config.proj_quant_n_bit,
            weight_quant = config.weight_quant,
            stochastic_round = config.stochastic_round,
            weight_group_size = config.weight_group_size,
            cos_threshold = config.cos_threshold,
            gamma_proj = config.gamma_proj,
            queue_size = config.queue_size,
            target_modules = config.target_modules,
        )

        # --- Split embedding params with custom LR (Fix #2) ---
        if embedding_lr is not None:
            # Fast param->name lookup (O(N) instead of O(N*M))
            param_to_name = {id(p): name for name, p in self.model.named_parameters()}

            new_groups = []
            for group in param_groups:
                if "rank" in group:
                    # GaLore group: keep as-is (no embeddings here)
                    new_groups.append(group)
                    continue
                # Non-GaLore group: split out embedding params
                embed_params = []
                other_params = []
                for p in group["params"]:
                    name = param_to_name.get(id(p))
                    if name and name.endswith("modules_to_save.default.weight"):
                        partial_name = name[: -len(".modules_to_save.default.weight")]
                        partial_name = partial_name[partial_name.rfind(".") + 1 :]
                        print(
                            f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
                        )
                        embed_params.append(p)
                    else:
                        other_params.append(p)
                if other_params:
                    other_group = dict(group)
                    other_group["params"] = other_params
                    new_groups.append(other_group)
                if embed_params:
                    embed_group = dict(group)
                    embed_group["params"] = embed_params
                    embed_group["lr"] = embedding_lr
                    new_groups.append(embed_group)
            param_groups = new_groups

        # --- Forward optimizer hyperparameters (Fix #3) ---
        self.optimizer = QGaLoreAdamW8bit(
            param_groups,
            lr = lr,
            weight_decay = weight_decay,
            betas = (self.args.adam_beta1, self.args.adam_beta2),
            eps = self.args.adam_epsilon,
        )

        if config.weight_quant:
            QGaLoreAdamW8bit.init_weight_quantization(
                self.model,
                param_groups,
                group_size = config.weight_group_size,
                stochastic = config.stochastic_round,
            )
            # Pre-hooks dequantize INT8 weights to float before each forward,
            # letting the optimizer free float weight memory between steps.
            install_weight_quant_hooks(self.model)

        n_galore = sum(len(g["params"]) for g in param_groups if "rank" in g)
        n_other = sum(len(g["params"]) for g in param_groups if "rank" not in g)
        print(
            f"🦥 Unsloth: Q-GaLore enabled — "
            f"{n_galore} GaLore params (rank={config.rank}), "
            f"{n_other} standard params."
        )

        return self.optimizer


# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _resolve_trainer_params(trainer_class, init_fn):
    """Resolve the real named parameters for a trainer __init__.

    Some TRL trainers are thin ``*args, **kwargs`` wrappers; for those, walk the
    MRO and return the first parent with real named parameters.
    """
    params = inspect.signature(init_fn).parameters
    named = {
        k
        for k, v in params.items()
        if v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and k != "self"
    }
    if named:
        return set(params.keys())

    # Thin wrapper detected - walk MRO for real signature
    for cls in trainer_class.__mro__[1:]:
        if cls is object:
            continue
        parent_init = cls.__dict__.get("__init__")
        if parent_init is None:
            continue
        try:
            parent_params = inspect.signature(parent_init).parameters
            parent_named = {
                k
                for k, v in parent_params.items()
                if v.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and k != "self"
            }
            if parent_named:
                return set(parent_params.keys())
        except (ValueError, TypeError):
            continue
    return set(params.keys())


def _ensure_warnings_issued(model):
    """Restore the `warnings_issued` dict that trl trainers write into.

    transformers set `self.warnings_issued = {}` in `PreTrainedModel.__init__` up
    to 5.0.0 and dropped it in 5.1.0. trl did not follow: grpo, dpo, online_dpo,
    kto, orpo, cpo, rloo and experimental bco still open `__init__` with
    `model.warnings_issued["estimate_tokens"] = True`, so the trainer cannot be
    built at all:

        AttributeError: 'Qwen2ForCausalLM' object has no attribute 'warnings_issued'

    models/rl.py already guards this, but only in the source it GENERATES, so the
    guard exists exactly when that generation succeeds. When it does not, unsloth
    falls back to trl's own class and the write is unguarded again. Measured, so
    the weaker claim is the right one: UNSLOTH_COMPILE_DISABLE=1 does NOT remove
    the generated module, which is still written with the guard in it, so that is
    not the gap this closes. The fallback is.

    Best-effort, and no stricter than the generated guard: a non-module (trl also
    accepts a repo id string) is left alone.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        return
    try:
        existing = getattr(model, "warnings_issued", None)
        if isinstance(existing, dict):
            return
        if existing is None:
            model.warnings_issued = {}
        else:
            # Preserve a non-dict value rather than discard it; trl only ever
            # writes one boolean key.
            try:
                model.warnings_issued = dict(existing)
            except Exception:
                model.warnings_issued = {}
    except Exception:
        # A model refusing the assignment is trl's to report, not ours to turn
        # into a different traceback.
        pass


def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # tokenizer is now processing_class
        trainer_params = _resolve_trainer_params(trainer_class, original_init)

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            trainer_params.remove("self")
            trainer_params.remove("args")

            # Fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) if field.init
            }

            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Params in Config but not in TrainingArguments
            from transformers import TrainingArguments

            moved_params = set(inspect.signature(config_class).parameters.keys()) - set(
                inspect.signature(TrainingArguments).parameters.keys()
            )

            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params:
                    trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value

            config_dict.update(additional_config_kwargs)

            # Only build the config if the previous init wasn't TrainingArguments:
            # reinitialising it would re-trigger mutually-exclusive param checks.
            # See https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                # Every trl config subclasses TrainingArguments, so this is the
                # branch real calls take and config_dict was going nowhere. Set
                # the moved values on the caller's config rather than rebuild it.
                config = training_args
                for key, value in additional_config_kwargs.items():
                    if key in config_fields or key in moved_params:
                        setattr(config, key, value)

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        _ensure_warnings_issued(args[0] if args else kwargs.get("model"))
        original_init(self, *args, **kwargs)

    return new_init


def _patch_sft_trainer_auto_packing(trl_module):
    sft_trainer = getattr(trl_module, "SFTTrainer", None)
    if sft_trainer is None:
        return
    if getattr(sft_trainer, "_unsloth_auto_packing_wrapped", False):
        return

    original_init = sft_trainer.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        config_arg = None
        if len(args) >= 2:
            config_arg = args[1]
        else:
            config_arg = kwargs.get("args")

        model = args[0] if len(args) >= 1 else kwargs.get("model")
        is_vlm = False
        is_unsupported_model = False
        is_hybrid = False
        is_encoder_decoder = False
        hybrid_varlen_active = False
        if model is not None:
            model_config = getattr(model, "config", None)
            if model_config is None and isinstance(model, str):
                # TRL builds a string model inside __init__; resolve its config now.
                model_config = _resolve_string_model_config(model, config_arg)
            if model_config is not None:
                model_types = get_transformers_model_type(model_config)
                is_unsupported_model = any(x in PADDING_FREE_BLOCKLIST for x in model_types)
                is_vlm = _is_vlm_config(model_config, model_types)
                is_encoder_decoder = bool(getattr(model_config, "is_encoder_decoder", False))
            hybrid_target = (
                SimpleNamespace(config = model_config)
                if isinstance(model, str) and model_config is not None
                else model
            )
            is_hybrid = _is_hybrid_linear_attention_model(hybrid_target)
            # Hybrid models corrupt packed batches unless the gated-delta conv + scan
            # reset at sequence boundaries. Enable the experimental varlen shim (flag +
            # kernels) so packing stays correct, else keep them blocked. A string model
            # (patched only after init) and TRL's chunked-loss forward bypass both leave
            # the shim off, so hybrid packing falls back to the padded path.
            if (
                is_hybrid
                and not isinstance(model, str)
                and not _chunked_loss_bypasses_forward(config_arg)
            ):
                try:
                    hybrid_varlen_active = patch_hybrid_linear_attention_varlen(model)
                except Exception:
                    hybrid_varlen_active = False

        processing_class = (
            args[5] if len(args) >= 6 else kwargs.get("processing_class") or kwargs.get("tokenizer")
        )
        data_collator = args[2] if len(args) >= 3 else kwargs.get("data_collator")
        train_dataset = args[3] if len(args) >= 4 else kwargs.get("train_dataset")
        eval_dataset = args[4] if len(args) >= 5 else kwargs.get("eval_dataset")
        is_processor = isinstance(processing_class, ProcessorMixin)
        is_auto_processor_vlm = is_vlm and processing_class is None
        is_vision_dataset = (
            data_collator is None
            and not is_processor
            and (
                _is_vision_dataset(train_dataset, unknown_is_vision = is_vlm)
                or _is_vision_eval_dataset(eval_dataset, unknown_is_vision = is_vlm)
            )
        )

        # Disable padding-free for VLMs / custom collators / blocklisted models
        blocked = (
            (data_collator is not None)
            or is_processor
            or is_auto_processor_vlm
            or is_vision_dataset
            or is_unsupported_model
            or is_encoder_decoder
            or (is_hybrid and not hybrid_varlen_active)
            or (
                os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            )  # Disable padding free on forced logits
        )
        requested_pack = bool(getattr(config_arg, "packing", False))
        if blocked:
            if hasattr(config_arg, "packing"):
                setattr(config_arg, "packing", False)
            if hasattr(config_arg, "padding_free"):
                setattr(config_arg, "padding_free", False)

        if blocked and requested_pack:
            reason = "custom data collator"
            if data_collator is None and is_processor:
                reason = "processor-based model"
            elif is_auto_processor_vlm:
                reason = "vision-language model with auto processor"
            elif is_vision_dataset:
                reason = "vision dataset"
            elif is_encoder_decoder:
                reason = "encoder-decoder model"
            elif is_hybrid and not hybrid_varlen_active:
                reason = "hybrid linear-attention model"
            elif is_unsupported_model:
                reason = f"unsupported model type(s): {', '.join(model_types)}"
            elif data_collator is None:
                # compute_metrics, preprocess_logits_for_metrics, for_inference() and the
                # user can all set it, so name the flag and not a setter.
                reason = "UNSLOTH_RETURN_LOGITS=1"
            logger.warning(f"Unsloth: packing=True ignored ({reason}).")

        packing_active = False
        if _should_pack(config_arg) and not blocked:
            configure_sample_packing(config_arg)
            packing_active = True
            logger.info("Unsloth: Sample packing enabled for SFTTrainer instance.")

        # Resolve padding_free: None (default) = auto-enable unless env-disabled or packing
        auto_padding_free_active = False
        padding_free_requested = getattr(config_arg, "padding_free", None) is True
        if not blocked:
            if padding_free_requested:
                configure_padding_free(config_arg)
            elif _should_auto_padding_free(config_arg):
                configure_padding_free(config_arg)
                auto_padding_free_active = True
                logger.info("Unsloth: Padding-free batching auto-enabled for SFTTrainer instance.")

        try:
            original_init(self, *args, **kwargs)
        except ValueError as exc:
            if packing_active and _should_skip_auto_packing_error(exc):
                logger.info(
                    "Unsloth: Auto sample packing failed because trainer reported an incompatible setup (%s).",
                    exc,
                )
                _disable_sample_packing(config_arg)
                packing_active = False
                original_init(self, *args, **kwargs)
            elif auto_padding_free_active and _should_skip_auto_padding_free_error(exc):
                train, evals = _bound_splits(original_init, args, kwargs)
                if not _cap_is_enforceable_without_padding_free(config_arg, train, evals):
                    raise
                logger.info(
                    "Unsloth: Auto padding-free disabled because the trainer rejected it (%s).",
                    exc,
                )
                _disable_padding_free(config_arg)
                auto_padding_free_active = False
                original_init(self, *args, **kwargs)
            else:
                raise

        trainer_args = getattr(self, "args", None)
        trainer_packing = bool(trainer_args and getattr(trainer_args, "packing", False))
        trainer_padding_free = bool(trainer_args and getattr(trainer_args, "padding_free", False))

        if blocked and trainer_args is not None:
            # Mirror the block on the trainer args to avoid re-enabling later
            setattr(trainer_args, "packing", False)
            setattr(trainer_args, "padding_free", False)

        if not blocked and trainer_packing and (packing_active or _should_pack(trainer_args)):
            enable_sample_packing(self.model, self)
            print("🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!")
        elif not blocked and trainer_padding_free:
            enable_padding_free_metadata(self.model, self)
            message = (
                "🦥 Unsloth: Padding-free auto-enabled, enabling faster training."
                if auto_padding_free_active
                else "🦥 Unsloth: Padding-free enabled, enabling faster training."
            )
            print(message)

        # get_peft_model installs a pre-train forward detector for plain LoRA/vision models,
        # but only RL trainers run the reset via prepare_for_training_mode. Wire it into the
        # SFT train() path too, else a grad-enabled probe before train() leaves the poisoned
        # Dynamo cache in place and the detector hook installed on every training forward.
        # (For UnslothSFTTrainer the later prepare_for_training_mode assignment supersedes this.)
        if not getattr(self, "_unsloth_train_reset_wrapped", False):
            try:
                from unsloth.models._utils import _unsloth_reset_stray_compile_cache

                _orig_train = self.train

                @wraps(_orig_train)
                def _train_with_reset(*train_args, **train_kwargs):
                    try:
                        _unsloth_reset_stray_compile_cache(self)
                    except Exception:
                        pass
                    return _orig_train(*train_args, **train_kwargs)

                self.train = _train_with_reset
                self._unsloth_train_reset_wrapped = True
            except Exception:
                pass

    sft_trainer.__init__ = new_init
    sft_trainer._unsloth_auto_packing_wrapped = True


def _patch_trl_trainer():
    import trl

    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl) <= Version("0.11.0"):
        return

    import trl.trainer

    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    # Auto-packing wraps first so it lands INSIDE the backwards-compatible one: a
    # moved `packing` kwarg must reach the config before packing is decided, else
    # the block is undone right after. Guarded so a failure here still leaves the
    # pre-0.13 compatibility wrappers installed.
    try:
        _patch_sft_trainer_auto_packing(trl)
    except Exception as exc:
        logger.warning(f"Unsloth: could not enable SFT auto-packing ({exc}).")

    for x in trl_classes:
        try:
            exec(
                f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
                globals(),
            )
        except:
            continue

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
