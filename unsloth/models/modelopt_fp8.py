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
"""Load an NVIDIA ModelOpt FP8 checkpoint through transformers' own fp8 quantizer.

ModelOpt (``quant_method: modelopt``, ``quant_algo: FP8``) writes the same thing
transformers' fine-grained fp8 format calls a static per-tensor checkpoint: an
``float8_e4m3fn`` weight, one float32 weight scale and one float32 input scale per
Linear. Only the names differ (``weight_scale`` / ``input_scale`` against
``weight_scale_inv`` / ``activation_scale``; both scales are dequantization
multipliers in either format), and transformers has no ``modelopt`` quantizer, so
such a checkpoint stops in ``from_pretrained`` with ``KeyError: 'modelopt'`` or, on
paths that skip the lookup, loads fp8 bytes into plain Linear layers without their
scales.

``arm_modelopt_fp8_loading`` rewrites the config's quantization block into the
equivalent ``quant_method: fp8`` one and returns the ``key_mapping`` that renames
the two scale tensors while the checkpoint streams in. After that it is an ordinary
static fp8 checkpoint: loaded as published it trains with fp8 base weights, and with
``load_in_16bit = True`` transformers dequantizes it. NVFP4 and every other ModelOpt
algorithm are left untouched.
"""

import inspect
from typing import Optional

__all__ = [
    "modelopt_fp8_plan",
    "arm_modelopt_fp8_loading",
    "MODELOPT_FP8_KEY_MAPPING",
    "UNSLOTH_MODELOPT_KEY_MAPPING_ATTR",
    "pop_modelopt_key_mapping",
    "keep_task_heads_unquantized",
    "modelopt_planner_quantization_config",
]

# Private attribute the key mapping is parked on until the loader hands it to from_pretrained.
UNSLOTH_MODELOPT_KEY_MAPPING_ATTR = "_unsloth_modelopt_key_mapping"

# Anchored at the end of the key, so `weight_scale_inv` or `weight_scale_2` never match.
MODELOPT_FP8_KEY_MAPPING = {
    r"\.weight_scale$": ".weight_scale_inv",
    r"\.input_scale$": ".activation_scale",
}

_FP8_ALGOS = ("FP8", "FP8_PER_TENSOR")


def _as_dict(quant) -> Optional[dict]:
    if quant is None:
        return None
    if isinstance(quant, dict):
        return quant
    to_dict = getattr(quant, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
        except Exception:
            return None
        return out if isinstance(out, dict) else None
    return None


def _group_is_fp8(group) -> bool:
    if not isinstance(group, dict):
        return False
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return False
    if str(weights.get("type", "")).lower() != "float" or int(weights.get("num_bits", 0) or 0) != 8:
        return False
    # Per-tensor only: a block or channel strategy would need a different scale layout.
    if weights.get("strategy", "tensor") not in (None, "tensor"):
        return False
    if weights.get("dynamic", False):
        return False
    return True


def _activations_map_onto_fp8(inputs) -> bool:
    """transformers' static fp8 keeps one scalar activation scale per Linear, so static input
    scales must be per-tensor fp8. Its dynamic scheme quantizes per token, which is what a
    dynamic "token" group declares and a finer form of "tensor"; channel, group or block
    activation scaling has no counterpart and is declined."""
    if not isinstance(inputs, dict):
        return False
    if str(inputs.get("type", "")).lower() != "float" or int(inputs.get("num_bits", 0) or 0) != 8:
        return False
    strategy = inputs.get("strategy", "tensor")
    if inputs.get("dynamic", False):
        return strategy in (None, "tensor", "token")
    return strategy in (None, "tensor")


def modelopt_fp8_plan(config) -> Optional[dict]:
    """Return the transformers fp8 quantization dict equivalent to ``config``'s ModelOpt FP8
    block, or ``None`` when ``config`` is not a ModelOpt per-tensor FP8 checkpoint."""
    quant = _as_dict(getattr(config, "quantization_config", None))
    if quant is None:
        return None
    if str(quant.get("quant_method", "")).lower() != "modelopt":
        return None
    algo = quant.get("quant_algo")
    if algo is None and isinstance(quant.get("quantization"), dict):
        algo = quant["quantization"].get("quant_algo")
    if str(algo).upper() not in _FP8_ALGOS:
        return None
    groups = quant.get("config_groups")
    activation_scheme = "static"
    if groups is not None:
        if not isinstance(groups, dict) or len(groups) == 0:
            return None
        if not all(_group_is_fp8(group) for group in groups.values()):
            return None
        # The fp8 form converts every Linear outside the ignore list, so only groups that
        # target every Linear map onto it; narrower targets would convert unscaled layers.
        if not all(
            list(group.get("targets") or ["Linear"]) == ["Linear"] for group in groups.values()
        ):
            return None
        inputs = [group.get("input_activations") for group in groups.values()]
        if all(x is None for x in inputs):
            activation_scheme = None
        elif any(x is None for x in inputs):
            # Mixed weight-only and W8A8 groups do not map onto one activation scheme.
            return None
        elif not all(_activations_map_onto_fp8(x) for x in inputs):
            return None
        elif any(x.get("dynamic", False) for x in inputs):
            activation_scheme = "dynamic"
    ignore = quant.get("ignore")
    if ignore is None:
        ignore = quant.get("exclude_modules")
    if ignore is None and isinstance(quant.get("quantization"), dict):
        ignore = quant["quantization"].get("exclude_modules")
    plan = {
        "quant_method": "fp8",
        "weight_block_size": None,
        "modules_to_not_convert": list(ignore or []),
    }
    if activation_scheme is not None:
        plan["activation_scheme"] = activation_scheme
    else:
        # Weight-only: no input scales in the checkpoint, so quantize activations on the fly.
        plan["activation_scheme"] = "dynamic"
    return plan


def _transformers_accepts_fp8_plan(plan) -> bool:
    """Older transformers (4.x) only know dynamic block fp8 and reject a per-tensor plan."""
    try:
        from transformers.utils.quantization_config import FineGrainedFP8Config
        FineGrainedFP8Config(**{k: v for k, v in plan.items() if k != "quant_method"})
    except Exception:
        return False
    return True


def arm_modelopt_fp8_loading(config, verbose: bool = True) -> Optional[dict]:
    """Rewrite ``config.quantization_config`` from ModelOpt FP8 to the transformers fp8 form
    and park the scale renaming on the config for the loader. Returns the new quantization
    dict, or ``None`` (config untouched) when the checkpoint is not ModelOpt FP8."""
    plan = modelopt_fp8_plan(config)
    if plan is None or not _transformers_accepts_fp8_plan(plan):
        return None
    try:
        config.quantization_config = dict(plan)
        setattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR, dict(MODELOPT_FP8_KEY_MAPPING))
    except Exception:
        return None
    if verbose:
        print(
            "Unsloth: NVIDIA ModelOpt FP8 checkpoint detected; loading it as a static per-tensor "
            "fp8 checkpoint (weight_scale -> weight_scale_inv, input_scale -> activation_scale)."
        )
    return plan


def _class_checkpoint_mapping(model_class) -> dict:
    """The class-level VLM checkpoint renames that transformers applies only when the caller
    passes no ``key_mapping`` (``if key_mapping ... elif VLM`` in 5.3). Empty where the class
    carries none, as on the transformers releases that collect them from the registry and
    add a caller ``key_mapping`` on top."""
    mapping = getattr(model_class, "_checkpoint_conversion_mapping", None)
    if not isinstance(mapping, dict) or not mapping:
        return {}
    try:
        from transformers.conversion_mapping import VLMS
    except Exception:
        try:
            from transformers.modeling_utils import VLMS
        except Exception:
            return {}
    mro = getattr(model_class, "__mro__", ())[:-1]
    if not any(name in cls.__name__.lower() for cls in mro for name in VLMS):
        return {}
    return dict(mapping)


def pop_modelopt_key_mapping(
    config,
    kwargs: dict,
    model_class = None,
) -> None:
    """Move a parked ModelOpt key mapping off ``config`` into ``kwargs['key_mapping']``. A
    caller-supplied ``key_mapping`` is kept and extended, never replaced. Without one, the
    VLM checkpoint renames of ``model_class`` are carried over, since passing any
    ``key_mapping`` makes some transformers releases skip them."""
    mapping = getattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR, None)
    if mapping is None:
        return
    try:
        delattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)
    except Exception:
        pass
    user = kwargs.get("key_mapping", None)
    if isinstance(user, dict):
        merged = dict(user)
    else:
        merged = _class_checkpoint_mapping(model_class)
    for pattern, target in mapping.items():
        # A user rule for the same pattern wins.
        merged.setdefault(pattern, target)
    kwargs["key_mapping"] = merged


# Heads a task class adds on top of the checkpoint (GenericForSequenceClassification and
# GenericForTokenClassification name theirs `score`; older task classes use `classifier`,
# question answering `qa_outputs`). They have no fp8 weight or scales on disk.
_TASK_HEAD_MODULES = ("score", "classifier", "qa_outputs")
_TASK_CLASS_SUFFIXES = (
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
)


def keep_task_heads_unquantized(config, *model_classes) -> bool:
    """Keep the new task head of a classification load out of the rewritten fp8 plan.

    The ModelOpt ignore list only names checkpoint modules, so a fresh `score` would be
    converted to FP8Linear and its random init fails on an fp8 weight. Applied only when
    one of ``model_classes`` (the auto or resolved class the load builds) is a task class,
    so a causal LM keeps its plan unchanged. Returns True when the plan was extended."""
    names = [getattr(cls, "__name__", "") for cls in model_classes if cls is not None]
    if not any(name.endswith(_TASK_CLASS_SUFFIXES) for name in names):
        return False
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        return False
    skip = list(quant.get("modules_to_not_convert") or [])
    skip.extend(name for name in _TASK_HEAD_MODULES if name not in skip)
    quant["modules_to_not_convert"] = skip
    return True


def modelopt_planner_quantization_config(config, dequantize: bool = False) -> Optional[dict]:
    """The rewritten fp8 plan on ``config`` as the load will apply it, for the device-map
    planner, which rebuilds the repo's ModelOpt block and cannot size it. ``dequantize``
    mirrors a 16-bit load, which asks the fp8 quantizer for bf16 weights."""
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict):
        return None
    plan = dict(quant)
    if dequantize:
        try:
            from transformers.utils.quantization_config import FineGrainedFP8Config
            if "dequantize" in inspect.signature(FineGrainedFP8Config).parameters:
                plan["dequantize"] = True
        except Exception:
            pass
    return plan
