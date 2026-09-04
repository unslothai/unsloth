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

import trl
import inspect
from trl import SFTTrainer

# Bypass the partially-initialised unsloth namespace during the _gpu_init load.
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

        # The base __call__ would reapply formatting_func, which was applied above.
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
# Hybrid linear-attention / state-space models (Qwen3.5, Qwen3-Next) carry a recurrent gated-delta
# state plus a causal conv1d that leak across sequence boundaries once packing flattens the batch.
# Detected structurally by _is_hybrid_linear_attention_model, not by model name.


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
    # TRL's own default: eval_packing = None means "whatever packing is".
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
        # Forward auth and cache args too: dropping token/use_auth_token made a private hybrid fail to
        # load (resolving as None), so it was treated as non-hybrid and packing was enabled without the
        # shim.
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
        # TRL merges top-level args.trust_remote_code into the load via setdefault before
        # create_model_from_path, so honor it here (model_init_kwargs wins), else a remote-code hybrid
        # resolves as None and skips the guard.
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


# Unsloth gradient accumulation fix.
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


class UnslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        q_galore_config: Optional[QGaloreConfig] = None,
        *args,
        **kwargs,
    ):
        self.q_galore_config = q_galore_config
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


class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        q_galore_config = getattr(self.args, "q_galore_config", None)
        if q_galore_config is not None and self.optimizer is None:
            embedding_lr = getattr(self.args, "embedding_learning_rate", None)
            return self._create_q_galore_optimizer(q_galore_config, embedding_lr)

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
            # Fast param -> name lookup, O(N) instead of O(N*M).
            param_to_name = {id(p): name for name, p in self.model.named_parameters()}

            new_groups = []
            for group in param_groups:
                if "rank" in group:
                    # GaLore group: keep as-is, since no embeddings are here.
                    new_groups.append(group)
                    continue
                # Non-GaLore group: split out embedding params.
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
            # Pre-hooks dequantize INT8 weights to float before each forward, letting the optimizer free float
            # weight memory between steps.
            install_weight_quant_hooks(self.model)

        n_galore = sum(len(g["params"]) for g in param_groups if "rank" in g)
        n_other = sum(len(g["params"]) for g in param_groups if "rank" not in g)
        print(
            f"🦥 Unsloth: Q-GaLore enabled — "
            f"{n_galore} GaLore params (rank={config.rank}), "
            f"{n_other} standard params."
        )

        return self.optimizer


# From trl >= 0.13.0 several params are passed to the trainer differently; patch to make the transition smooth.
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

    # Thin wrapper detected: walk the MRO for the real signature.
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
            # Preserve a non-dict value rather than discard it; trl only ever writes one boolean key.
            try:
                model.warnings_issued = dict(existing)
            except Exception:
                model.warnings_issued = {}
    except Exception:
        # A model refusing the assignment is trl's to report, not ours to turn into a different traceback.
        pass


def _route_unknown_trainer_kwargs(
    config_class,
    unknown,
    notify = None,
    already_supplied = None,
):
    """Split names neither side declares into `(to_config, to_trainer)` using
    `rl_config_compat`'s rename/retire policy, which otherwise only runs when a
    config is built; unrecognised names go back to the trainer call to raise there.

    `already_supplied` uses presence, not a default comparison, because `kwargs`
    holds only what the caller typed, so an explicit new name is never overwritten.
    """
    if not unknown:
        return {}, {}

    try:
        from .models.rl_config_compat import (
            classify_config_kwarg,
            removal_source,
            rename_source,
            rename_value_is_unset,
        )
    except Exception:
        # tests/ AST-load this function into a bare namespace with no package to
        # import from, and supply the classifier through globals() instead.
        classify_config_kwarg = globals().get("classify_config_kwarg")
        rename_source = globals().get("rename_source", lambda key: "TRL")
        removal_source = globals().get("removal_source", lambda key: "TRL")
        rename_value_is_unset = globals().get(
            "rename_value_is_unset", lambda cls, name, value: False
        )
        if classify_config_kwarg is None:
            # Unreachable in a real install; keep the value on the trainer.
            return {}, dict(unknown)

    if notify is None:
        notify = print
    config_name = getattr(config_class, "__name__", str(config_class))

    to_config, to_trainer = {}, {}
    for key, value in unknown.items():
        try:
            verdict, detail = classify_config_kwarg(config_class, key)
        except Exception:
            verdict, detail = "unknown", None

        if verdict == "accepted":
            to_config[key] = value
        elif verdict == "rename":
            # A legacy Optional forwarded at its `None` default says nothing, so
            # it must not replace a real value. Same rule as the config path.
            if rename_value_is_unset(config_class, detail, value):
                continue
            if already_supplied and detail in already_supplied:
                notify(
                    f"Unsloth: `{key}` was renamed to `{detail}` by {rename_source(key)} and "
                    f"this {config_name} accepts only the new name. You set both, so `{key}` "
                    f"is ignored and your `{detail}` is kept."
                )
                continue
            to_config[detail] = value
            notify(
                f"Unsloth: {rename_source(key)} renamed `{key}` to `{detail}`. "
                f"Forwarding your value to `{detail}` - update your code when convenient."
            )
        elif verdict == "retired":
            notify(
                f"Unsloth: `{key}` is not supported by the installed "
                f"{removal_source(key)}'s {config_name} and will be IGNORED - {detail}."
            )
        else:
            to_trainer[key] = value

    return to_config, to_trainer


def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # tokenizer is now processing_class.
        trainer_params = _resolve_trainer_params(trainer_class, original_init)

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # `discard`, not `remove`: a trainer naming its config something other
            # than `args` would otherwise die with a bare KeyError here.
            trainer_params.discard("self")
            trainer_params.discard("args")

            # Fields that should be passed to Config init.
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) if field.init
            }

            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Params in Config but not in TrainingArguments.
            from transformers import TrainingArguments

            moved_params = set(inspect.signature(config_class).parameters.keys()) - set(
                inspect.signature(TrainingArguments).parameters.keys()
            )

            # Separate kwargs into trainer kwargs and config kwargs.
            trainer_kwargs = {}
            additional_config_kwargs = {}
            unknown_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params:
                    trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    # Filing this with the config kwargs dropped it in silence.
                    unknown_kwargs[key] = value

            migrated, unroutable = _route_unknown_trainer_kwargs(
                config_class,
                unknown_kwargs,
                already_supplied = set(additional_config_kwargs),
            )
            additional_config_kwargs.update(migrated)
            trainer_kwargs.update(unroutable)

            config_dict.update(additional_config_kwargs)

            # Only build the config if the previous init was not TrainingArguments: reinitialising it would re-
            # trigger the mutually-exclusive param checks (trl grpo_config.py#L499-L502).
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                # Every trl config subclasses TrainingArguments, so this is the branch real calls take and
                # config_dict was going nowhere: set the moved values on the caller's config rather than rebuild it.
                config = training_args
                for key, value in additional_config_kwargs.items():
                    # `migrated` keys are already known to be accepted here.
                    if key in config_fields or key in moved_params or key in migrated:
                        setattr(config, key, value)

            # Reconstruct kwargs for Trainer.
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
            # Hybrid models corrupt packed batches unless the gated-delta conv and scan reset at sequence
            # boundaries, so enable the experimental varlen shim (flag plus kernels) or keep them blocked.
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
            or (os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1")
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
                # compute_metrics, preprocess_logits_for_metrics, for_inference() and the user can all set it, so
                # name the flag and not a setter.
                reason = "UNSLOTH_RETURN_LOGITS=1"
            logger.warning(f"Unsloth: packing=True ignored ({reason}).")

        packing_active = False
        if _should_pack(config_arg) and not blocked:
            configure_sample_packing(config_arg)
            packing_active = True
            logger.info("Unsloth: Sample packing enabled for SFTTrainer instance.")

        # Resolve padding_free: None (default) means auto-enable unless env-disabled or packing.
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

        # get_peft_model installs a pre-train forward detector for plain LoRA/vision models, but only RL
        # trainers run the reset via prepare_for_training_mode, so wire it into the SFT train() path too,
        # else a grad-enabled probe before train() leaves the poisoned Dynamo cache in place.
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

    # Auto-packing wraps first so it lands INSIDE the backwards-compatible wrapper: a moved `packing`
    # kwarg must reach the config before packing is decided, else the block is undone right after.
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
