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
"""Load NVIDIA ModelOpt FP8 checkpoints through transformers' fp8 quantizer.

ModelOpt per-tensor FP8 equals transformers' static fp8 format except for scale names
(``weight_scale`` / ``input_scale`` vs ``weight_scale_inv`` / ``activation_scale``), so the
config block is rewritten to ``quant_method: fp8`` and the scales renamed via ``key_mapping``.
"""

import fnmatch
import inspect
import weakref
from typing import Optional

__all__ = [
    "modelopt_fp8_plan",
    "arm_modelopt_fp8_loading",
    "MODELOPT_FP8_KEY_MAPPING",
    "UNSLOTH_MODELOPT_KEY_MAPPING_ATTR",
    "pop_modelopt_key_mapping",
    "modelopt_rewritten",
    "keep_task_heads_unquantized",
    "modelopt_planner_quantization_config",
]

UNSLOTH_MODELOPT_KEY_MAPPING_ATTR = "_unsloth_modelopt_key_mapping"

# Anchored at the end of the key, so `weight_scale_inv` or `weight_scale_2` never match.
MODELOPT_FP8_KEY_MAPPING = {
    r"\.weight_scale$": ".weight_scale_inv",
    r"\.input_scale$": ".activation_scale",
}

_FP8_ALGOS = ("FP8", "FP8_PER_TENSOR")


def _modelopt_pattern(name) -> str:
    # ModelOpt ignore entries are fnmatch globs; as regexes `layers.16*` would also skip layers 1, 10-19.
    name = str(name)
    if any(ch in name for ch in "*?["):
        return fnmatch.translate(name)
    return name


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
    if weights.get("strategy", "tensor") not in (None, "tensor"):
        return False
    if weights.get("dynamic", False):
        return False
    return True


def _activations_map_onto_fp8(inputs) -> bool:
    # transformers' dynamic fp8 is per token, so dynamic "token" maps; static must be per tensor.
    if not isinstance(inputs, dict):
        return False
    if str(inputs.get("type", "")).lower() != "float" or int(inputs.get("num_bits", 0) or 0) != 8:
        return False
    strategy = inputs.get("strategy", "tensor")
    if inputs.get("dynamic", False):
        return strategy in (None, "tensor", "token")
    return strategy in (None, "tensor")


def modelopt_fp8_plan(config) -> Optional[dict]:
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
        # fp8 converts every Linear not ignored; narrower targets would convert unscaled layers.
        if not all(
            list(group.get("targets") or ["Linear"]) == ["Linear"] for group in groups.values()
        ):
            return None
        inputs = [group.get("input_activations") for group in groups.values()]
        if all(x is None for x in inputs):
            activation_scheme = None
        elif any(x is None for x in inputs):
            return None
        elif not all(_activations_map_onto_fp8(x) for x in inputs):
            return None
        elif len({bool(x.get("dynamic", False)) for x in inputs}) > 1:
            return None
        elif inputs[0].get("dynamic", False):
            activation_scheme = "dynamic"
    ignore = quant.get("ignore")
    if ignore is None:
        ignore = quant.get("exclude_modules")
    if ignore is None and isinstance(quant.get("quantization"), dict):
        ignore = quant["quantization"].get("exclude_modules")
    plan = {
        "quant_method": "fp8",
        "weight_block_size": None,
        "modules_to_not_convert": [_modelopt_pattern(name) for name in (ignore or [])],
    }
    if activation_scheme is not None:
        plan["activation_scheme"] = activation_scheme
    else:
        plan["activation_scheme"] = "dynamic"
    return plan


def _transformers_accepts_fp8_plan(plan) -> bool:
    # transformers 4.x only knows dynamic block fp8 and rejects a per-tensor plan.
    try:
        from transformers.utils.quantization_config import FineGrainedFP8Config
        FineGrainedFP8Config(**{k: v for k, v in plan.items() if k != "quant_method"})
    except Exception:
        return False
    return True


# A retried / second load reuses the config (now native fp8, marker moved off): keep the renaming by id.
_REWRITTEN_CONFIGS: dict = {}


def _remember_rewrite(config, mapping) -> None:
    key = id(config)
    entry = _REWRITTEN_CONFIGS.get(key)
    if entry is not None and entry[0]() is config:
        return
    try:
        ref = weakref.ref(config)
        weakref.finalize(config, _REWRITTEN_CONFIGS.pop, key, None)
    except TypeError:
        return
    _REWRITTEN_CONFIGS[key] = (ref, dict(mapping))


def _remembered_mapping(config) -> Optional[dict]:
    entry = _REWRITTEN_CONFIGS.get(id(config))
    if entry is None or entry[0]() is not config:
        return None
    return dict(entry[1])


def modelopt_rewritten(config) -> bool:
    return (
        hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)
        or _remembered_mapping(config) is not None
    )


def arm_modelopt_fp8_loading(config, verbose: bool = True) -> Optional[dict]:
    plan = modelopt_fp8_plan(config)
    if plan is None or not _transformers_accepts_fp8_plan(plan):
        return None
    try:
        config.quantization_config = dict(plan)
        setattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR, dict(MODELOPT_FP8_KEY_MAPPING))
    except Exception:
        return None
    _remember_rewrite(config, MODELOPT_FP8_KEY_MAPPING)
    if verbose:
        print(
            "Unsloth: NVIDIA ModelOpt FP8 checkpoint detected; loading it as a static per-tensor "
            "fp8 checkpoint (weight_scale -> weight_scale_inv, input_scale -> activation_scale)."
        )
    return plan


def _class_checkpoint_mapping(model_class) -> dict:
    # transformers 5.3 applies VLM class renames only when no caller key_mapping is passed.
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
    mapping = getattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR, None)
    if mapping is None:
        mapping = _remembered_mapping(config)
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
        merged.setdefault(pattern, target)
    kwargs["key_mapping"] = merged


# Fresh task heads with no fp8 weight on disk.
_TASK_HEAD_MODULES = ("score", "classifier", "classification_head", "qa_outputs")
_TASK_CLASS_SUFFIXES = (
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForMultipleChoice",
    "ForImageClassification",
    "ForAudioClassification",
    "ForAudioFrameClassification",
    "ForVideoClassification",
)


def keep_task_heads_unquantized(config, *model_classes) -> bool:
    names = [getattr(cls, "__name__", "") for cls in model_classes if cls is not None]
    if not any(name.endswith(_TASK_CLASS_SUFFIXES) for name in names):
        return False
    # A task checkpoint already ships its head, quantized unless the ignore list names it.
    architectures = getattr(config, "architectures", None) or []
    if any(str(name).endswith(_TASK_CLASS_SUFFIXES) for name in architectures):
        return False
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        return False
    skip = list(quant.get("modules_to_not_convert") or [])
    skip.extend(name for name in _TASK_HEAD_MODULES if name not in skip)
    quant["modules_to_not_convert"] = skip
    return True


def modelopt_planner_quantization_config(config, dequantize: bool = False) -> Optional[dict]:
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
