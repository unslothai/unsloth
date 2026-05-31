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

import logging
import os
import psutil
import warnings
from dataclasses import dataclass, field
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
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
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
]

logger = logging.getLogger(__name__)

_AUTO_PADDING_FREE_ENV_DISABLED = os.environ.get(
    "UNSLOTH_DISABLE_AUTO_PADDING_FREE", ""
).strip().lower() in {"1", "true", "yes", "on"}

PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}


def _should_pack(config) -> bool:
    if config is None or not getattr(config, "packing", False):
        return False
    return not getattr(config, "_unsloth_disable_auto_packing", False)


def _should_auto_padding_free(config) -> bool:
    if (
        config is None
        or _AUTO_PADDING_FREE_ENV_DISABLED
        or getattr(config, "packing", False)
    ):
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

    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: Optional[tuple[float, float, float]] = None
    muon_lr_scale: float = 1.0
    adjust_lr_fn: Optional[str] = None
    muon_eps: float = 1e-7
    muon_weight_decay: Optional[float] = None
    adamw_lr: Optional[float] = None
    adamw_betas: tuple = (0.9, 0.999)
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
        if self.muon_weight_decay is not None and not isinstance(self.muon_weight_decay, (int, float)):
            raise TypeError(
                f"MuonConfig.muon_weight_decay must be a number, got {type(self.muon_weight_decay).__name__}."
            )
        if self.adamw_weight_decay is not None and not isinstance(self.adamw_weight_decay, (int, float)):
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
            raise ValueError(
                f"MuonConfig.momentum must be >= 0.0, got {self.momentum}."
            )
        if self.muon_eps <= 0.0:
            raise ValueError(
                f"MuonConfig.muon_eps must be > 0.0, got {self.muon_eps}."
            )
        if self.muon_lr_scale <= 0.0:
            raise ValueError(
                f"MuonConfig.muon_lr_scale must be > 0.0, got {self.muon_lr_scale}."
            )
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
        muon_config: Optional[MuonConfig] = None,
        *args,
        **kwargs,
    ):
        self.q_galore_config = q_galore_config
        self.muon_config = muon_config
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)


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

    def __init__(self, muon, adamw, needs_deterministic=False):
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
        was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled() if was_enabled else False
        if not was_enabled or not was_warn_only:
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            self.muon.step()
        finally:
            if was_enabled:
                torch.use_deterministic_algorithms(True, warn_only=was_warn_only)
            else:
                torch.use_deterministic_algorithms(False)

    def step(self, closure=None):
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

    def zero_grad(self, set_to_none=True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

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
        # C2 fix: re-sync chained groups to match freshly loaded sub-optimizer groups.
        refreshed = []
        if self.muon is not None:
            refreshed.extend(self.muon.param_groups)
        if self.adamw is not None:
            refreshed.extend(self.adamw.param_groups)
        self.param_groups = refreshed

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
            return sum(len(g["params"]) for g in sub.param_groups)
        muon_str = f"Muon({_param_count(self.muon)} params)"
        adamw_str = f"AdamW({_param_count(self.adamw)} params)"
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
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )
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
        import torch.distributed as dist
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
        embedding_lr = config.embedding_lr if config.embedding_lr is not None \
            else getattr(self.args, "embedding_learning_rate", None)

        muon_weight_decay = config.muon_weight_decay if config.muon_weight_decay is not None else weight_decay
        adamw_weight_decay = config.adamw_weight_decay if config.adamw_weight_decay is not None else weight_decay

        muon_groups, adamw_groups = make_muon_param_groups(
            self.model,
            lr=lr,
            muon_weight_decay=muon_weight_decay,
            muon_lr_scale=config.muon_lr_scale,
            adamw_lr=config.adamw_lr,
            adamw_weight_decay=adamw_weight_decay,
            target_modules=config.target_modules,
            embedding_lr=embedding_lr,
        )

        if PeftModel is not None and isinstance(self.model, PeftModel):
            logger.warning(
                "Unsloth Muon: PEFT/LoRA model detected. "
                "Muon will be applied to 2D adapters. "
                "Results not guaranteed — use full_finetuning=True for expected behaviour."
            )

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
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
            eps=config.muon_eps,
            ns_coefficients=config.ns_coefficients,
            adjust_lr_fn=config.adjust_lr_fn,
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

        if config.adamw_betas != (0.9, 0.999):
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
        adamw_kwargs = dict(lr=adamw_lr, betas=adamw_betas, eps=adamw_eps)
        if adamw_groups:
            adamw_optimizer = torch.optim.AdamW(adamw_groups, **adamw_kwargs)
        else:
            adamw_optimizer = None

        self.optimizer = _MuonAdamWChained(
            muon_optimizer, adamw_optimizer, needs_deterministic=needs_deterministic,
        )
        return self.optimizer

    def _create_q_galore_optimizer(self, config: "QGaloreConfig", embedding_lr = None):
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
            # Build a fast param->name lookup (O(N) instead of O(N*M))
            param_to_name = {id(p): name for name, p in self.model.named_parameters()}

            new_groups = []
            for group in param_groups:
                if "rank" in group:
                    # GaLore group — keep as-is (embeddings are never in here)
                    new_groups.append(group)
                    continue
                # Non-GaLore group: split out embedding params
                embed_params = []
                other_params = []
                for p in group["params"]:
                    # Check if this param belongs to a modules_to_save embedding
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

        # Initialize INT8 weight quantization if enabled
        if config.weight_quant:
            QGaLoreAdamW8bit.init_weight_quantization(
                self.model,
                param_groups,
                group_size = config.weight_group_size,
                stochastic = config.stochastic_round,
            )
            # Forward pre-hooks dequantize INT8 weights to float before each
            # forward pass, allowing the optimizer to free float weight memory
            # between steps.
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

    Some TRL trainers (e.g., ORPOTrainer in TRL 0.27.1) are thin wrappers
    with only ``def __init__(self, *args, **kwargs)``.  For those, walk the
    MRO and return the first parent class that has real named parameters.
    """
    params = inspect.signature(init_fn).parameters
    named = {
        k
        for k, v in params.items()
        if v.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
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


def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = _resolve_trainer_params(trainer_class, original_init)

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove("self")
            trainer_params.remove("args")

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field
                for field in dataclasses.fields(config_class)
                if field.init
            }

            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
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

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
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

        # Check if model type is unsupported for padding_free
        model = kwargs.get("model")
        is_unsupported_model = False
        is_vlm = False
        if model is not None:
            model_config = getattr(model, "config", None)
            if model_config is not None:
                model_types = get_transformers_model_type(model_config)
                # Blocklist: models that don't work correctly with padding_free
                is_unsupported_model = any(
                    x in PADDING_FREE_BLOCKLIST for x in model_types
                )

                # Check if VLM
                architectures = getattr(model_config, "architectures", None)
                if architectures is None:
                    architectures = []
                is_vlm = any(
                    x.endswith("ForConditionalGeneration") for x in architectures
                )
                is_vlm = is_vlm or hasattr(model_config, "vision_config")

        processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
        data_collator = kwargs.get("data_collator")

        # We also disable vision language models for padding free collators
        blocked = (
            (data_collator is not None)
            or isinstance(processing_class, ProcessorMixin)
            or is_vlm
            or is_unsupported_model
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
            if data_collator is None and isinstance(processing_class, ProcessorMixin):
                reason = "processor-based model"
            elif is_vlm:
                reason = "vision-language model"
            elif is_unsupported_model:
                reason = f"unsupported model type(s): {', '.join(model_types)}"
            message = "Unsloth: Sample packing skipped " f"({reason} detected)."
            print(message)

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
                logger.info(
                    "Unsloth: Padding-free batching auto-enabled for SFTTrainer instance."
                )

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
            else:
                raise

        trainer_args = getattr(self, "args", None)
        trainer_packing = bool(trainer_args and getattr(trainer_args, "packing", False))
        trainer_padding_free = bool(
            trainer_args and getattr(trainer_args, "padding_free", False)
        )

        if blocked and trainer_args is not None:
            # Mirror the block on the trainer args to avoid re-enabling later
            setattr(trainer_args, "packing", False)
            setattr(trainer_args, "padding_free", False)

        if (
            not blocked
            and trainer_packing
            and (packing_active or _should_pack(trainer_args))
        ):
            enable_sample_packing(self.model, self)
            print(
                "🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!"
            )
        elif not blocked and trainer_padding_free:
            enable_padding_free_metadata(self.model, self)
            message = (
                "🦥 Unsloth: Padding-free auto-enabled, enabling faster training."
                if auto_padding_free_active
                else "🦥 Unsloth: Padding-free enabled, enabling faster training."
            )
            print(message)

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
    trl_trainers = set(
        x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer")
    )
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:
            exec(
                f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
                globals(),
            )
        except:
            continue

    _patch_sft_trainer_auto_packing(trl)

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
