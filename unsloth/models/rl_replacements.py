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

__all__ = [
    "RL_EXTRA_ARGS",
    "RL_FUNCTIONS",
    "RL_PRE_ITEMS",
    "RL_CONFIG_CHANGES",
    "RL_METRICS_CHANGES",
]

import os
import re
import torch
import inspect
import linecache
from collections import defaultdict
from unsloth_zoo.rl_replacements import (
    RL_REPLACEMENTS,
    left_pack_padding,
    create_completion_attention_mask,
    chunked_selective_log_softmax,
    chunked_hidden_states_selective_log_softmax,
    _unsloth_get_mm_token_id,
    _unsloth_fix_mm_token_type_ids,
)
from unsloth_zoo.utils import Version
from trl import __version__ as trl_version_raw
from importlib.metadata import version as importlib_version
from unsloth_zoo.log import logger
from unsloth_zoo.device_type import device_synchronize

try:
    from unsloth_zoo.device_map_planner import detect_logit_transforms
except ImportError:
    # Older unsloth_zoo: fall back to reading the config fields directly.
    detect_logit_transforms = None
import importlib.util
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)
import textwrap
from ._utils import _get_inference_mode_context_manager, UNSLOTH_ENABLE_LOGGING

# One-time GRPO sequence-packing gates; mirrored into the generated trainer cache via RL_PRE_ITEMS.
UNSLOTH_GRPO_SEQ_PACKING_ON = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Packing needs zoo#840's masked-column guard in grpo_compute_loss (the installed zoo is fixed per-process).
try:
    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = "torch.where(_keep, new" in inspect.getsource(
        RL_REPLACEMENTS["grpo_compute_loss"]
    )
except Exception:
    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = False
# One-time PrefixGrouper gate; any import failure degrades to "PrefixGrouper off".
_pg_build_layout = _pg_enabled_fn = _pg_verify_on = _pg_tol_ok = _PG_TOL_KILL = None
UNSLOTH_GRPO_PREFIX_GROUPER_ON = os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
if UNSLOTH_GRPO_PREFIX_GROUPER_ON:
    try:
        from ..utils.prefix_grouper import (
            build_group_layout as _pg_build_layout,
            prefix_grouper_enabled as _pg_enabled_fn,
            verify_on as _pg_verify_on,
            tol_ok as _pg_tol_ok,
            TOL_KILL as _PG_TOL_KILL,
        )
    except Exception:
        UNSLOTH_GRPO_PREFIX_GROUPER_ON = False

RL_EXTRA_ARGS = defaultdict(list)
RL_FUNCTIONS = defaultdict(list)
RL_PRE_ITEMS = defaultdict(list)


def _unsloth_clear_stateful_mrope(model):
    modules = getattr(model, "modules", None)
    if modules is None:
        return False

    cleared = False
    for module in modules():
        if hasattr(module, "compute_3d_position_ids") and hasattr(module, "rope_deltas"):
            module.rope_deltas = None
            cleared = True
    return cleared


RL_CONFIG_CHANGES = defaultdict(list)
RL_METRICS_CHANGES = defaultdict(list)
RL_ADDITIONAL_FUNCTIONS = defaultdict(list)

_DPO_VISION_KEYS = (
    "pixel_position_ids",
    "image_position_ids",
    "mm_token_type_ids",
)

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,  # I saw speedups, but not sure if this has issues in collab
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

try:
    trl_version = Version(trl_version_raw)
except Exception:
    try:
        trl_version = Version(importlib_version("trl"))
    except Exception:
        trl_version = Version("0.0.0")


def sft_trainer_fix_untrained_tokens(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_tokenizer = (
            "IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\\n')\n"
            "from unsloth_zoo.tokenizer_utils import fix_untrained_tokens\n"
            "from unsloth_zoo.training_utils  import fix_zero_training_loss\n"
            "if 'tokenizer' not in locals(): tokenizer = processing_class\n"
            "fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n"
            "fix_zero_training_loss(model, tokenizer, train_dataset)\n"
        )
        return fix_tokenizer
    return ""


RL_EXTRA_ARGS["sft_trainer"].append(sft_trainer_fix_untrained_tokens)


# huggingface/trl#4695 added top_k to GRPOConfig defaulting to 0, but vLLM's include-all top_k
# is -1 and 0 errors on SamplingParams creation.
def grpo_config_fix_vllm_top_k(old_RLTrainer_source, old_RLConfig_source):
    return "if use_vllm and (top_k is None or top_k == 0): top_k = -1\n"


RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_config_fix_vllm_top_k)


# Remove DPO columns which might randomnly be tokenized
def dpo_trainer_fix_columns(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_dpo = (
            "if hasattr(train_dataset, 'column_names'):\n"
            "    column_names = set(train_dataset.column_names)\n"
            "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"
            "             'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"
            "             'prompt_input_ids', 'prompt_attention_mask']\n"
            "    if all(x in column_names for x in check):\n"
            "        train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"
            "    del check, column_names\n"
        )
        return fix_dpo
    return ""


RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_columns)


def dpo_trainer_fix_data_collator(call_args, extra_args):
    if (
        "data_collator" in call_args
        and "train_dataset" in call_args
        and "processing_class" in call_args
    ):
        fix_collator = (
            "if hasattr(train_dataset, 'column_names'):\n"
            "    column_names = set(train_dataset.column_names)\n"
            "    is_dpo_dataset = ({'chosen', 'rejected'}.issubset(column_names) or\n"
            "                      {'prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids'}.issubset(column_names))\n"
            "    if is_dpo_dataset and isinstance(data_collator, TransformersDataCollatorForLanguageModeling):\n"
            "        data_collator = None\n"
            "    del is_dpo_dataset, column_names\n"
        )
        return fix_collator
    return ""


RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_data_collator)


def dpo_trainer_vision_process_row(
    features,
    processing_class,
    max_prompt_length = None,
    max_completion_length = None,
    add_special_tokens = True,
    is_chat = False,
):
    text = features.get("prompt", "")
    images = features.get("images")
    processor, tokenizer = processing_class, processing_class.tokenizer
    processed_features = processor(
        images = images,
        text = text,
        add_special_tokens = False,
    )

    prompt_input_ids = processed_features["input_ids"][0]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens = False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens = False)["input_ids"]

    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
    if not is_chat and tokenizer.eos_token_id is not None:
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    output = {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }
    if "pixel_values" in processed_features:
        output["pixel_values"] = processed_features["pixel_values"][0]
    if "pixel_attention_mask" in processed_features:
        output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
    if "image_sizes" in processed_features:
        output["image_sizes"] = processed_features["image_sizes"][0]
    if "token_type_ids" in processed_features:
        token_type_ids = processed_features["token_type_ids"][0]
        if max_prompt_length is not None:
            token_type_ids = token_type_ids[-max_prompt_length:]
        output["token_type_ids"] = token_type_ids
    if "pixel_position_ids" in processed_features:
        output["pixel_position_ids"] = processed_features["pixel_position_ids"][0]
    if "image_position_ids" in processed_features:
        output["image_position_ids"] = processed_features["image_position_ids"][0]
    if "mm_token_type_ids" in processed_features:
        mm_token_type_ids = processed_features["mm_token_type_ids"][0]
        if max_prompt_length is not None:
            mm_token_type_ids = mm_token_type_ids[-max_prompt_length:]
        output["mm_token_type_ids"] = mm_token_type_ids

    return output


def dpo_trainer_vision_signature_columns(function_name, function):
    if function_name != "_set_signature_columns_if_needed":
        return function

    if all(_k in function for _k in _DPO_VISION_KEYS):
        return function

    _extra_columns = "".join(f'                "{_k}",\n' for _k in _DPO_VISION_KEYS)
    new_function = function.replace(
        '                "image_sizes",\n                "token_type_ids",\n',
        f'                "image_sizes",\n{_extra_columns}                "token_type_ids",\n',
    )
    if new_function != function:
        return new_function
    return function.replace(
        '                "image_sizes",\n                "ref_chosen_logps",\n',
        f'                "image_sizes",\n{_extra_columns}                "ref_chosen_logps",\n',
    )


def dpo_trainer_concatenated_inputs(function_name, function):
    if function_name != "concatenated_inputs":
        return function

    if all(_k in function for _k in _DPO_VISION_KEYS):
        return function

    _extra_inputs = "".join(
        f'        if "{_k}" in batch:\n'
        f'            output["{_k}"] = torch.cat((batch["{_k}"], batch["{_k}"]), dim=0)\n'
        for _k in _DPO_VISION_KEYS
    )

    image_sizes_block = (
        '        if "image_sizes" in batch:\n'
        '            output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"]], dim=0)\n'
    )
    new_function = function.replace(
        image_sizes_block + '        if "token_type_ids" in batch:\n',
        image_sizes_block + _extra_inputs + '        if "token_type_ids" in batch:\n',
    )
    if new_function != function:
        return new_function
    if image_sizes_block in function:
        return function.replace(image_sizes_block, image_sizes_block + _extra_inputs, 1)
    return function


def _dpo_trainer_extend_vision_model_kwargs(function):
    if all(_k in function for _k in _DPO_VISION_KEYS):
        return function

    _extra_forward = "".join(
        f'        if "{_k}" in concatenated_batch:\n'
        f'            model_kwargs["{_k}"] = concatenated_batch["{_k}"]\n'
        for _k in (
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            *_DPO_VISION_KEYS,
        )
    )

    return function.replace(
        '        if "pixel_values" in concatenated_batch:\n'
        '            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]\n'
        '        if "pixel_attention_mask" in concatenated_batch:\n'
        '            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]\n'
        '        if "image_sizes" in concatenated_batch:\n'
        '            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]\n',
        f"{_extra_forward}",
    )


def dpo_trainer_concatenated_forward(function_name, function):
    if function_name != "concatenated_forward":
        return function
    return _dpo_trainer_extend_vision_model_kwargs(function)


def dpo_trainer_compute_loss_liger(function_name, function):
    if function_name != "_compute_loss_liger":
        return function
    return _dpo_trainer_extend_vision_model_kwargs(function)


def dpo_trainer_data_collator_vision_keys(call_args, extra_args):
    if "data_collator" not in call_args:
        return ""

    _vision_keys = str(_DPO_VISION_KEYS)
    return (
        "from trl.trainer.dpo_trainer import DataCollatorForPreference\n"
        "if not hasattr(DataCollatorForPreference, '_unsloth_vision_keys_patch'):\n"
        "    _old_dpo_collator_torch_call = DataCollatorForPreference.torch_call\n"
        "\n"
        "    def _unsloth_dpo_torch_call(self, examples):\n"
        "        output = _old_dpo_collator_torch_call(self, examples)\n"
        "        import torch as _unsloth_torch\n"
        "        try:\n"
        "            from trl.trainer.utils import pad as _unsloth_trl_pad\n"
        "        except Exception:\n"
        "            _unsloth_trl_pad = None\n"
        "        for _k in " + _vision_keys + ":\n"
        "            if not all(_k in example for example in examples):\n"
        "                continue\n"
        "            _is_position_key = _k.endswith('position_ids')\n"
        "            _padding_value = -1 if _is_position_key else 0\n"
        "            _padding_side = 'right' if _is_position_key else 'left'\n"
        "            _values = [_unsloth_torch.as_tensor(example[_k]) for example in examples]\n"
        "            try:\n"
        "                if _unsloth_trl_pad is not None:\n"
        "                    output[_k] = _unsloth_trl_pad(_values, padding_value=_padding_value, padding_side=_padding_side)\n"
        "                else:\n"
        "                    from torch.nn.utils.rnn import pad_sequence as _unsloth_pad_sequence\n"
        "                    output[_k] = _unsloth_pad_sequence(_values, batch_first=True, padding_value=_padding_value)\n"
        "            except Exception:\n"
        "                from torch.nn.utils.rnn import pad_sequence as _unsloth_pad_sequence\n"
        "                output[_k] = _unsloth_pad_sequence(_values, batch_first=True, padding_value=_padding_value)\n"
        "        return output\n"
        "\n"
        "    DataCollatorForPreference.torch_call = _unsloth_dpo_torch_call\n"
        "    DataCollatorForPreference._unsloth_vision_keys_patch = True\n"
    )


def dpo_trainer_prepare_dataset(function_name, function):
    if function_name != "_prepare_dataset":
        return function

    legacy_call = "self.tokenize_row if not self.is_vision_model else self.process_row"
    if legacy_call not in function:
        return function

    function = function.replace(
        legacy_call,
        "self.tokenize_row if not self.is_vision_model else dpo_trainer_vision_process_row",
    )

    legacy_tokenize_block = (
        "            # Tokenize the dataset\n"
        "            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`\n"
        '                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"\n'
        "\n"
        "            dataset = dataset.map(\n"
        "                self.tokenize_row if not self.is_vision_model else dpo_trainer_vision_process_row,\n"
    )
    patched_tokenize_block = (
        "            # Tokenize the dataset\n"
        "            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`\n"
        '                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"\n'
        "            if self.is_vision_model:\n"
        '                map_kwargs.pop("num_proc", None)\n'
        "\n"
        "            dataset = dataset.map(\n"
        "                self.tokenize_row if not self.is_vision_model else dpo_trainer_vision_process_row,\n"
    )
    if legacy_tokenize_block in function:
        function = function.replace(legacy_tokenize_block, patched_tokenize_block, 1)
    return function


RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_prepare_dataset)
RL_PRE_ITEMS["dpo_trainer"].append(inspect.getsource(dpo_trainer_vision_process_row))
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_vision_signature_columns)
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_concatenated_inputs)
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_concatenated_forward)
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_compute_loss_liger)
RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_data_collator_vision_keys)


_WRAPPED_PACKING_SETUP = (
    "    import inspect as _inspect\n"
    "    try:\n"
    '        _unsloth_pack_has_strategy = "strategy" in _inspect.signature(pack_dataset).parameters\n'
    "    except Exception:\n"
    "        _unsloth_pack_has_strategy = True\n"
    "    _unsloth_wrapped_packing = packing and (\n"
    '        getattr(args, "packing_strategy", None) == "wrapped"\n'
    "        or not _unsloth_pack_has_strategy\n"
    "    )\n"
)

_WARNED_MISSING_ANCHORS = set()


def _warn_once(where, message):
    if where in _WARNED_MISSING_ANCHORS:
        return
    _WARNED_MISSING_ANCHORS.add(where)
    logger.warning(message)


def _require_replace(
    function,
    old,
    new,
    *,
    count = 1,
    required = True,
    where = "",
):
    """str.replace that never silently no-ops a load-bearing source edit.

    Plain str.replace returns the source unchanged when the anchor is absent, so a
    drifted anchor in a newer TRL / unsloth_zoo would skip the edit while later edits
    still reference helper variables it should have introduced (NameError at runtime).
    Fail loudly for a required edit, warn once and skip for an optional one, so a
    drifted source can never corrupt the patched function silently.
    """
    if old not in function:
        detail = f" ({where})" if where else ""
        if required:
            raise RuntimeError(
                f"Unsloth: source anchor not found{detail}; the patched function is out "
                "of sync with this TRL / unsloth_zoo version. Please file a bug report."
            )
        _warn_once(
            where,
            f"Unsloth: skipped an optional source edit{detail} (anchor not found).",
        )
        return function
    return function.replace(old, new, count)


# The one line every unsloth_zoo sft_prepare_dataset ends its worker count on, as a regex so a
# renamed right-hand side still matches and the indentation carries over. Unchanged since Aug
# 2025 (#257) while the block around it was rewritten repeatedly, hence a fallback anchor.
_ZOO_MAP_NUM_PROC_ASSIGNMENT = re.compile(
    r"^(?P<indent>[ \t]*)map_kwargs\[\"num_proc\"\][ \t]*=[ \t]*[^\n]+$",
    flags = re.MULTILINE,
)

# The Zoo seeds its truncation length from args.max_length and falls through to
# args.max_seq_length only when that is 0; rl.py hands TRL >= 1.0.0 a None, so normalise it or
# nothing truncates.
_ZOO_MAX_LENGTH_SEED = re.compile(
    r"^(?P<indent>[ \t]*)max_seq_length[ \t]*=[ \t]*getattr\(args,[ \t]*[\"']max_length[\"'],[ \t]*0\)[ \t]*$",
    flags = re.MULTILINE,
)


def _same_source(text):
    """`text` with quote style normalised, for comparing two spellings of a line.

    The narrow regexes here already accept either quote, so the idempotence check
    has to as well: a Zoo carrying the replacement with single quotes matched
    neither the literal nor the `$`-anchored regex, and `required = True` then
    raised on every SFT trainer over behaviour already present.
    """
    return text.replace("'", '"')


def _replace_or_fallback(
    function,
    old,
    new,
    *,
    fallback_pattern,
    fallback_new,
    where = "",
    required = False,
    consequence = "",
):
    """str.replace over a wide anchor, with a narrower anchor to fall back on.

    One literal anchor is all-or-nothing, and neither outcome is acceptable here.
    `required = True` hard-fails every SFT run on a Zoo whose text merely moved.
    `required = False` leaves the un-rewritten Zoo logic in place, and for the
    worker count that is not the no-op it looks like: the config layer writes
    "run in-process" as `args.dataset_num_proc = 1` (a config `None` means
    "auto-size me" to the Zoo), the Zoo reads that `1` as an explicit count, and
    `datasets` >= 4.1 builds a `Pool(1)` for it -- forking a tokenizer worker on
    exactly the host, or the `UNSLOTH_DATASET_NUM_PROC=0`, that asked for none.

    So: try the wide anchor, then a narrow one that survives more drift, and warn
    only when both miss. `fallback_new` is an `re.sub` template, so it can carry
    the matched indentation over with `\\g<indent>`.

    `required = True` keeps the two-anchor tolerance but raises rather than warns
    when BOTH miss, for an edit whose absence is not the no-op it looks like.
    `consequence` says what that absence does, since the warning below speaks only
    about the worker count.
    """
    # Already done upstream, and checked FIRST: an edit is missing only if its RESULT is, and a Zoo
    # that adopts the replacement is the forward case. Order matters: `old` is a prefix of `new`
    # for the max_length seed, so the wide anchor matched the normalized line and appended a
    # second `or 0`.
    if _same_source(new) in _same_source(function):
        return function
    if old in function:
        return function.replace(old, new, 1)

    function, applied = fallback_pattern.subn(fallback_new, function)
    if applied:
        _warn_once(
            where,
            f"Unsloth: the source block for {where} moved in this unsloth_zoo; "
            "applied the narrower anchor instead. Please file a bug report.",
        )
        return function

    if required:
        raise RuntimeError(
            f"Unsloth: failed to apply a required source edit ({where}) "
            f"(anchor not found){consequence}; please file a bug report."
        )
    _warn_once(
        where,
        f"Unsloth: skipped an optional source edit ({where}) (anchor not found); "
        "dataset tokenization may fork a worker process this host asked it not "
        "to. Please file a bug report.",
    )
    return function


# Fix tokenizer double BOS
def sft_trainer_prepare_dataset(function_name, function):
    if function_name != "_prepare_non_packed_dataloader" and function_name != "_prepare_dataset":
        return function

    fast_sft_prepare_dataset = RL_REPLACEMENTS.get("sft_prepare_dataset", None)
    if fast_sft_prepare_dataset is not None:
        params = inspect.signature(fast_sft_prepare_dataset).parameters.keys()
        params = ".*?".join(params)
        matched = re.match(
            r"[\s]{0,}def _prepare_dataset\(.*?" + params + r".*?\)",
            function,
            flags = re.MULTILINE | re.DOTALL,
        )
        if matched:
            function = inspect.getsource(fast_sft_prepare_dataset)
            # Anchor the wrapped-packing setup on the function signature, which always exists, not the
            # unsloth_zoo license comment, which a newer Zoo may move: anchoring there let the setup
            # silently no-op while later edits used its variables, NameError-ing every SFT dataset prep.
            function, _n_setup = re.subn(
                r"(def sft_prepare_dataset\s*\(.*?\)\s*(?:->[^:\n]*)?:[ \t]*\n)",
                lambda match: match.group(1) + _WRAPPED_PACKING_SETUP,
                function,
                count = 1,
                flags = re.DOTALL,
            )
            if _n_setup != 1:
                raise RuntimeError(
                    "Unsloth: failed to install wrapped-packing support into "
                    "sft_prepare_dataset (signature not found); please file a bug report."
                )
            function = _replace_or_fallback(
                function,
                '    max_seq_length = getattr(args, "max_length", 0)',
                '    max_seq_length = getattr(args, "max_length", 0) or 0',
                fallback_pattern = _ZOO_MAX_LENGTH_SEED,
                fallback_new = r'\g<indent>max_seq_length = getattr(args, "max_length", 0) or 0',
                where = "sft_prepare_dataset max_length seed",
                # Not optional, unlike the worker count this helper was written for: the generated trainer
                # clears args.max_length for padding-free, so an unrewritten seed reads None instead of the 0
                # that falls through to max_seq_length. Both anchors missing means the neighbouring
                # _require_replace edits have gone too.
                required = True,
                consequence = ", so `max_length` would not be enforced for raw "
                "datasets under padding-free batching",
            )
            # Route each edit through _require_replace so a drifted anchor fails loudly instead of leaving
            # a dangling reference to the setup variables.
            function = _require_replace(
                function,
                "truncation = do_truncation,",
                "truncation = do_truncation and not _unsloth_wrapped_packing,",
                where = "sft_prepare_dataset truncation flag",
            )
            function = _require_replace(
                function,
                "if do_truncation and max_seq_length > 0:",
                "if do_truncation and not _unsloth_wrapped_packing and max_seq_length > 0:",
                where = "sft_prepare_dataset truncation guard",
            )
            # Reuse the guarded _unsloth_pack_has_strategy from the setup rather than re-calling
            # _inspect.signature(pack_dataset): the setup wraps that in try/except, so a
            # non-introspectable pack_dataset must not crash here.
            function = _require_replace(
                function,
                """dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            getattr(args, "packing_strategy", "bfd"),
            map_kwargs,
        )""",
                """_pack_kwargs = {"map_kwargs": map_kwargs}
        if _unsloth_pack_has_strategy:
            _pack_kwargs["strategy"] = getattr(args, "packing_strategy", "bfd")
        dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            **_pack_kwargs,
        )""",
                where = "sft_prepare_dataset pack_dataset call",
            )
            # The map() call site, not the config, is where the worker count is made safe: the Zoo copy
            # asks stdlib multiprocessing for a start method datasets takes from multiprocess, and its
            # low-memory branch yields 1, still a Pool(1) on datasets >= 4.1. Imported from the Zoo so
            # generated source never imports back into its generator.
            function = _replace_or_fallback(
                function,
                """if not isinstance(dataset, IterableDataset):
            import multiprocessing as _mp
            dataset_num_proc = getattr(args, "dataset_num_proc", None)
            if dataset_num_proc is None:
                if _mp.get_start_method() != 'fork':
                    dataset_num_proc = None
                else:
                    import psutil
                    dataset_num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                    memory_gb_left = psutil.virtual_memory().available / (1024**3)
                    if memory_gb_left <= 2:
                        dataset_num_proc = 1
                    else:
                        dataset_num_proc = min(dataset_num_proc, int(memory_gb_left))
            map_kwargs["num_proc"] = dataset_num_proc""",
                """if not isinstance(dataset, IterableDataset):
            try:
                from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
            except ImportError:
                from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
            map_kwargs["num_proc"] = _unsloth_get_dataset_num_proc(
                getattr(args, "dataset_num_proc", None)
            )""",
                # The likeliest anchor here to drift, but its absence is not harmless: the config layer encodes
                # serial as 1 for this site to turn back into None, and an un-rewritten Zoo hands that 1 to
                # Dataset.map. The fallback keys on the block's closing assignment, unchanged since Aug 2025.
                # Hard-failing instead would break every install on a newer Zoo.
                fallback_pattern = _ZOO_MAP_NUM_PROC_ASSIGNMENT,
                fallback_new = r"""\g<indent>try:
\g<indent>    from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
\g<indent>except ImportError:
\g<indent>    from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
\g<indent>map_kwargs["num_proc"] = _unsloth_get_dataset_num_proc(
\g<indent>    getattr(args, "dataset_num_proc", None)
\g<indent>)""",
                where = "sft_prepare_dataset dataset_num_proc selection",
            )
            # datasets never reads the child's exit status, so every worker death flattens into "One of the
            # subprocesses has abruptly died during map operation". Wrap both tokenizing map() calls so the
            # user gets the worker count, start method and implied memory.
            function = _require_replace(
                function,
                """            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")""",
                """            try:
                from unsloth_zoo.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
            except ImportError:
                from unsloth.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
            with _w.catch_warnings(), _unsloth_map_diagnostics(map_kwargs.get("num_proc")):
                _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")""",
                count = 2,
                where = "sft_prepare_dataset tokenizing map() calls",
                # required = False, like the selection above: this only improves a dead worker's message, and a
                # diagnostic must not fail a run because a Zoo release moved its anchor.
                # test_zoo_sft_prepare_dataset_anchor_has_not_drifted notices in CI instead.
                required = False,
            )
            function = function.split("\n")
            function = "\n".join(" " * 4 + x for x in function)
            function = function.replace("def sft_prepare_dataset", "def _prepare_dataset")
            return function

    check_text = (
        "if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:\n"
        "    return dataset\n"
        "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"
        "if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')\n"
        "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"
        "if 'dataset_text_field' not in locals(): dataset_text_field = None\n"
        "if formatting_func is None and dataset_text_field is None and 'prompt' in dataset[0] and 'completion' in dataset[0]:\n"
        "    test_text = (dataset[0]['prompt'] + dataset[0]['completion']) if (isinstance(dataset[0]['prompt'], str) and isinstance(dataset[0]['completion'], str)) else None\n"
        "elif formatting_func is None and dataset_text_field is not None:\n"
        "    test_text = dataset[0][dataset_text_field]\n"
        "elif formatting_func is not None:\n"
        "    test_text = formatting_func(dataset[0])[0]\n"
        "else:\n"
        "    test_text = None\n"
        "chat_template = getattr(tokenizer, 'chat_template', None)\n"
        "chat_template = '' if chat_template is None else chat_template\n"
        "has_bos_token_already = ((test_text is not None and test_text.startswith(tokenizer.bos_token)) or tokenizer.bos_token in chat_template) "
        "if getattr(tokenizer, 'bos_token', None) is not None else False\n"
        "if 'add_special_tokens' not in locals() and has_bos_token_already:\n"
        "    from functools import partial\n"
        "    tokenizer_call = tokenizer.__call__\n"
        "    tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)\n"
        "    processing_class = tokenizer\n"
        "else:\n"
        "    tokenizer_call = None\n"
        "    add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)\n"
    )

    check_text = check_text.split("\n")
    check_text = "\n".join(" " * 8 + x for x in check_text)
    check_text = check_text.rstrip() + "\n"

    # .*? matches the first match, .+? the final one.
    replacer = re.findall(
        r"def " + function_name + r"\(.*?\).*?\:\n",
        function,
        flags = re.MULTILINE | re.DOTALL,
    )
    if len(replacer) != 0:
        replacer = replacer[0]
        function = function.replace(replacer, replacer + check_text)

    # Return tokenizer's original state
    return_state = "if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call\n"
    function = re.sub(
        r"\n([ ]{4,})(return .*?[\s]{0,})$",
        rf"\1{return_state}\1\2",
        function,
    )
    return function


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_prepare_dataset)


# Ignore mean_token_accuracy since it needs logits; it is overridden with our version.
def sft_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss":
        return function

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs = return_outputs,
            num_items_in_batch = num_items_in_batch,
        )
        return outputs

    function = inspect.getsource(compute_loss)
    return function


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)


# Route ORPO/CPO row tokenization through the underlying text tokenizer when the processing
# class is a multimodal processor; CPO reuses this code (#4952).
def orpo_trainer_text_tokenizer(function_name, function):
    if function_name == "build_tokenized_answer":
        function = re.sub(
            r"(?m)^([ \t]*)full_tokenized = self\.processing_class\(prompt \+ answer, add_special_tokens=False\)\n"
            r'\1prompt_input_ids = self\.processing_class\(prompt, add_special_tokens=False\)\["input_ids"\]\n',
            r'\1tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)'
            "\n"
            r"\1full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)"
            "\n"
            r'\1prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]'
            "\n",
            function,
            count = 1,
        )
        return function

    if function_name != "tokenize_row":
        return function

    if (
        'tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)'
        not in function
    ):
        new_function = re.sub(
            r"(?m)^([ \t]*)batch = \{\}\n",
            r"\1batch = {}"
            "\n"
            r'\1tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)'
            "\n",
            function,
            count = 1,
        )
        if new_function == function:
            return function
        function = new_function
    function = function.replace("self.processing_class(", "tokenizer(")
    function = function.replace("self.processing_class.bos_token_id", "tokenizer.bos_token_id")
    function = function.replace("self.processing_class.eos_token_id", "tokenizer.eos_token_id")
    return function


RL_FUNCTIONS["orpo_trainer"].append(orpo_trainer_text_tokenizer)
RL_FUNCTIONS["cpo_trainer"].append(orpo_trainer_text_tokenizer)


# Resolve processing_class.pad_token_id through the inner tokenizer when a multimodal processor
# is supplied: processors lack pad_token_id, so ORPO/CPOTrainer.__init__ raises AttributeError
# in the collator and padding_value.
_PAD_FALLBACK = (
    "(getattr(processing_class, 'pad_token_id', None) "
    "if getattr(processing_class, 'pad_token_id', None) is not None "
    "else getattr(getattr(processing_class, 'tokenizer', None), 'pad_token_id', None))"
)


def orpo_trainer_processor_pad_token(function_name, function):
    if function_name != "__init__":
        return function
    # Multimodal processors expose pad_token / eos_token on .tokenizer, not on themselves, and TRL
    # 1.x CPO/ORPO __init__ defaults pad_token from eos_token before tokenizing. Older TRL lacks
    # this block, so the sub is a no-op there.
    function = re.sub(
        r"(?m)^([ \t]*)if processing_class\.pad_token is None:\n"
        r"\1[ \t]+processing_class\.pad_token\s*=\s*processing_class\.eos_token\n",
        r"\1_unsloth_proc_tok = getattr(processing_class, 'tokenizer', processing_class)\n"
        r"\1if getattr(_unsloth_proc_tok, 'pad_token', None) is None:\n"
        r"\1    _unsloth_proc_tok.pad_token = getattr(_unsloth_proc_tok, 'eos_token', None)\n",
        function,
        count = 1,
    )
    if "processing_class.pad_token_id" not in function:
        return function
    return function.replace("processing_class.pad_token_id", _PAD_FALLBACK)


RL_FUNCTIONS["orpo_trainer"].append(orpo_trainer_processor_pad_token)
RL_FUNCTIONS["cpo_trainer"].append(orpo_trainer_processor_pad_token)


# Fix the bare pop("push_to_hub_token") in the compiled SFT/IterativeSFT __init__: on
# transformers 5.0+ to_dict() no longer includes it, so a bare pop KeyErrors.
def sft_trainer_push_to_hub_token(function_name, function):
    if function_name != "__init__":
        return function
    return function.replace(
        'dict_args.pop("push_to_hub_token")', 'dict_args.pop("push_to_hub_token", None)'
    )


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_push_to_hub_token)


# Autocast precision for GRPO.
def _unsloth_grpo_autocast(self):
    """Decide the GRPO autocast once and latch it on the trainer.

    ACCELERATE_MIXED_PRECISION is process wide, so a trainer built later but run
    first would hand this trainer its precision. args belongs to this trainer.
    """
    if not hasattr(self, "_autocast_enabled"):
        args = getattr(self, "args", None)
        precision = getattr(args, "mixed_precision", None)
        use_bf16 = getattr(args, "bf16", None)
        use_fp16 = getattr(args, "fp16", None)
        if not isinstance(precision, str):
            # transformers < 5 has no args.mixed_precision, but rl.py sets the fp16 / bf16 flags on this
            # same args for every branch it takes.
            if isinstance(use_bf16, bool) and isinstance(use_fp16, bool):
                precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "no")
            else:
                precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16")
        self._autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        # "no" is a real value: full finetuning and an explicit float32 load both set it, and reading
        # it as bfloat16 raises on a T4 or V100.
        self._autocast_enabled = precision != "no"
        self._autocast_force_float32 = False
        # Stamped by from_pretrained: UNSLOTH_FORCE_FLOAT32 is process wide, so a model loaded after
        # this trainer was built would answer for it here.
        forced = getattr(getattr(self, "model", None), "_unsloth_forced_float32", None)
        if forced is None:
            forced = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
        if forced and precision != "bf16":
            # Gemma3 / gpt-oss set "no" but still want float16 autocast; a trainer already on bf16 keeps
            # it, since float16 is what the forced list avoids.
            self._autocast_dtype = torch.float16
            self._autocast_enabled = True
            self._autocast_force_float32 = True

    return self._autocast_enabled, self._autocast_dtype


def _unsloth_grpo_autocast_kwargs(self, device_type = "cuda"):
    """torch.amp.autocast kwargs for GRPO generation."""
    enabled, dtype = _unsloth_grpo_autocast(self)
    if not getattr(self, "_autocast_force_float32", False) and torch.is_autocast_enabled(
        device_type
    ):
        # Already inside an autocast: inherit its dtype by omitting the key, since autocast passes
        # whatever it gets to set_autocast_dtype.
        return {"enabled": enabled}
    return {"enabled": enabled, "dtype": dtype}


def grpo_trainer__prepare_inputs(function_name, function):
    if function_name != "_prepare_inputs":
        return function

    # Latched on the trainer, so a second trainer's __init__ cannot change this trainer's autocast mid run.
    function = function.replace(
        "with torch.inference_mode():",
        "with torch.inference_mode(), "
        "torch.amp.autocast(device_type = 'cuda', **_unsloth_grpo_autocast_kwargs(self)):",
    )
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Guard reload_weights and sync_weights: skip when fast inference LoRA shares weights with
# vLLM (huggingface/trl commit 7856d3b).
def _guard_vllm_sync_reload_for_shared_weights(function):
    # Guard reload_weights - only call when not sharing weights with vLLM
    reload_weights_pattern = re.compile(
        r"^(?P<indent>[ \t]*)self\.llm\.collective_rpc\(\s*(['\"])reload_weights\2\s*\)\s*$",
        re.MULTILINE,
    )

    def replace_reload_weights_line(match):
        indent = match.group("indent")
        return (
            f"{indent}if not getattr(self.llm, 'shared_weights', False):\n"
            f'{indent}    self.llm.collective_rpc("reload_weights")\n'
        )

    function = reload_weights_pattern.sub(replace_reload_weights_line, function)

    # Guard sync_weights - skip when sharing weights with vLLM
    sync_weights_block = re.compile(
        r"(?P<indent>[ \t]*)with profiling_context\(self,\s*(['\"])sync_weights\2\s*\):\n"
        r"(?P=indent)[ \t]+self\.vllm_generation\.sync_weights\(\)\n",
        re.MULTILINE,
    )

    def guard_sync_weights_block(match):
        indent = match.group("indent")
        return (
            f"{indent}if not getattr(getattr(self.vllm_generation, 'llm', None), 'shared_weights', False):\n"
            f"{indent}    with profiling_context(self, 'sync_weights'):\n"
            f"{indent}        self.vllm_generation.sync_weights()\n"
        )

    function = sync_weights_block.sub(guard_sync_weights_block, function)
    return function


def grpo_trainer__generate_single_turn(function_name, function):
    if function_name != "_generate_single_turn":
        return function

    function = _guard_vllm_sync_reload_for_shared_weights(function)

    # TRL 0.24.0-0.25.1 truncation regression: 0.22.2-0.23.1 used truncate_with_protected_tokens
    # (tokenize, keep the RIGHTMOST tokens, protect vision tokens), 0.24.0-0.25.1 passed
    # max_length/truncation to the tokenizer, which protects nothing, and 0.26.2+ removed those
    # kwargs. Dropping them makes 0.24.0-0.25.1 behave like 0.26.2+; a no-op elsewhere.
    for pattern in [
        r'["\']?max_length["\']?\s*[:=]\s*self\.max_prompt_length\s*,\s*\n?',
        r'["\']?truncation["\']?\s*[:=]\s*True\s*,\s*\n?',
        r'["\']?add_special_tokens["\']?\s*[:=]\s*False\s*,\s*\n?',
    ]:
        function = re.sub(pattern, "", function)

    string_to_find = "            generate_inputs = super()._prepare_inputs(generate_inputs)"
    replacement_string = (
        string_to_find
        + """
            if "mm_token_type_ids" in generate_inputs or "image_grid_thw" in generate_inputs:
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class,
                    generate_inputs["input_ids"],
                    generate_inputs.get("mm_token_type_ids", None),
                )
                if mm_token_type_ids is not None:
                    generate_inputs["mm_token_type_ids"] = mm_token_type_ids"""
    )
    function = function.replace(string_to_find, replacement_string)

    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate_single_turn)


def grpo_trainer__generate(function_name, function):
    if function_name != "_generate":
        return function

    return _guard_vllm_sync_reload_for_shared_weights(function)


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate)


# Older TRL mishandles special tokens: 0.19.0 passed skip_special_tokens = True where it
# should be False.
def grpo_trainer__generate_and_score_completions(function_name, function):
    if function_name != "_generate_and_score_completions":
        return function

    # TRL 0.19.0 did skip_special_tokens = True which should be False
    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    # Left pad the prompt before calculating old and ref hidden states.
    line_to_replace = 'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size'

    # The new multi-line string that replaces the line above.
    replacement_lines = """
        max_left_pad = None
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        try:
            # TRL 0.23.1 and below path
            if not has_images:
                # Left pad prompt before calculation old and ref hidden states
                left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(prompt_completion_ids, logits_to_keep, self.processing_class.pad_token_id)
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
        except:
            # TRL 0.24.0 and below path
            if images is None:
                # Left pad prompt before calculation old and ref hidden states
                left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(prompt_completion_ids, logits_to_keep, self.processing_class.pad_token_id)
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
        _use_gc = self.model._unsloth_gradient_checkpointing if hasattr(self.model, '_unsloth_gradient_checkpointing') else getattr(self.args, 'gradient_checkpointing', True)
        self.model.for_training(use_gradient_checkpointing=_use_gc)"""

    function = function.replace(line_to_replace, replacement_lines)

    pattern_to_find = re.compile(
        r"^\s*if self\.args\.gradient_accumulation_steps % generate_every != 0 or \(\s*"
        r"self\.use_vllm and self\.vllm_importance_sampling_correction\s*"
        r"\):",
        re.MULTILINE,
    )

    replacement_text = """
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm
            ):"""
    function, num_replacements = pattern_to_find.subn(replacement_text, function)

    pattern_to_find = re.compile(
        r"(^\s*)all_logprobs = \["  # Capture indentation (group 1)
        r".*?"  # Match everything inside non-greedily
        r"for output in outputs\.outputs\s*"
        r"\]",
        re.DOTALL | re.MULTILINE,
    )

    # sanitize_logprob is injected as a module-level function by the RLTrainer_replacement template
    # in rl.py, so reference it directly.
    replacement_text = (
        r"\1all_logprobs = [\n"
        r"\1    [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]\n"
        r"\1    for outputs in all_outputs\n"
        r"\1    for output in outputs.outputs\n"
        r"\1]"
    )

    function, num_replacements = pattern_to_find.subn(replacement_text, function)

    # Always between max_prompt_length and use_vllm.
    found = re.findall(
        r"\n(([ ]{8,})if self\.max_prompt_length is not None:.*?\2if self\.use_vllm:)",
        function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(found) != 0:
        replace_part, spacing = found[0]
        removed_comments = re.sub(r"\#[^\n]{1,}", "", replace_part)
        splits = removed_comments.split("\n")
        if (
            sum(re.match(rf"{spacing}[^\s]", x) is not None for x in splits) == 2
            and len(spacing) >= 8
        ):
            new_replacement = f"""\n{spacing}if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = [re.sub(rf"^({{re.escape(self.pad_token)}})+", "", text) for text in prompts_text]

            # The chat template inserts a single image token into the prompt text. However, when this text is later
            # tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original template.
            if self.image_token is not None:
                prompts_text = [
                    re.sub(rf"({{re.escape(self.image_token)}})+", self.image_token, text) for text in prompts_text
                ]
        # Generate completions using either vLLM or regular generation
        if self.use_vllm:"""
            function = function.replace(replace_part, new_replacement)

    # TRL's importance sampling is disabled because the LLM path moves left padding to the right,
    # so Unsloth adjusts the vLLM sampling_logprob tensor itself.
    string_to_find = "if self.use_vllm and self.vllm_importance_sampling_correction:"

    replacement_string = "if False and self.use_vllm and self.vllm_importance_sampling_correction:"

    function = function.replace(string_to_find, replacement_string)

    string_to_find = """        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]"""

    replacement_string = """        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        if max_left_pad is not None:
            output["max_left_pad"] = torch.tensor(prompt_ids.shape[0] * [max_left_pad]).unsqueeze(-1)
        try:
            if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):
                output["sampling_per_token_logps"] = sampling_per_token_logps
        except NameError:
            output["sampling_per_token_logps"] = None"""

    function = function.replace(string_to_find, replacement_string)

    # TRL 0.24.0+ extracts prompts = [x["prompt"] for x in inputs], losing metadata like
    # reasoning_effort, so inject code storing per-sample chat_template_kwargs on self.
    _metadata_extraction = (
        "\n"
        "        # Unsloth: Extract per-sample chat_template_kwargs before metadata is lost\n"
        "        _ct_ = getattr(self.processing_class, 'chat_template', None) or ''\n"
        "        _sk_ = {'prompt', 'chosen', 'rejected', 'completion', 'messages', 'label',\n"
        "                'images', 'image', 'videos', 'video', 'audios', 'audio'}\n"
        "        self._unsloth_batch_chat_kwargs = []\n"
        "        for _inp_ in inputs:\n"
        "            _kw_ = {}\n"
        "            if isinstance(_inp_, dict):\n"
        "                for _k_ in _inp_.keys() - _sk_:\n"
        "                    if _k_ in _ct_ and isinstance(_inp_[_k_], str):\n"
        "                        _kw_[_k_] = _inp_[_k_]\n"
        "            self._unsloth_batch_chat_kwargs.append(_kw_)\n"
    )
    _target_line = 'prompts = [x["prompt"] for x in inputs]'
    if _target_line in function:
        function = function.replace(
            _target_line,
            _target_line + _metadata_extraction,
        )

    # This path is for TRL 0.24.0: `images` is a variable exclusive to that version.
    string_to_find = """        if images is not None:
            output["num_images"] = num_images"""

    replacement_string = """        if images is not None:
            output["num_images"] = num_images
        if max_left_pad is not None:
            output["max_left_pad"] = torch.tensor(prompt_ids.shape[0] * [max_left_pad]).unsqueeze(-1)
        try:
            if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):
                output["sampling_per_token_logps"] = sampling_per_token_logps
        except NameError:
            output["sampling_per_token_logps"] = None"""

    function = function.replace(string_to_find, replacement_string)

    if trl_version >= Version("0.24.0"):
        # Replace the call using 'completions' with one using 'completions_text'.
        string_to_find = "        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)"
        replacement_string = (
            "        if images is not None:\n"
            "            rewards_per_func = self._calculate_rewards(inputs, prompts_text, completions_text, completion_ids_list)\n"
            "        else:\n"
            "            rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)"
        )
        function = function.replace(string_to_find, replacement_string)

    _generate_return = """        ) = self._generate(prompts)"""
    if _generate_return in function and "_unsloth_clear_stateful_mrope" not in function:
        function = function.replace(
            _generate_return,
            _generate_return
            + """

        _unsloth_clear_stateful_mrope(
            self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)
        )""",
        )

    if "wake_up()" not in function:
        # Sleep functionality was added to trl in v0.23.0 (commit edbe823), so do not redo it.

        pattern = re.compile(r".*self\.llm\.generate\(.*\).*", re.MULTILINE)
        matches = list(pattern.finditer(function))
        patched = function

        # Generally there is only one match; the loop is to make sure none are missed.
        for match in reversed(matches):
            line = match.group(0)
            indent_match = re.match(r"(\s*)", line)
            indent = indent_match.group(1) if indent_match else ""

            wrapped = (
                f"{indent}if hasattr(self, 'llm'):\n"
                f"{indent}    if getattr(self.llm.llm_engine.vllm_config.model_config, 'enable_sleep_mode', False):\n"
                f"{indent}        self.llm.wake_up()\n"
                f"{line}\n\n"
                f"{indent}if hasattr(self, 'llm'):\n"
                f"{indent}    if getattr(self.llm.llm_engine.vllm_config.model_config, 'enable_sleep_mode', False):\n"
                f"{indent}        self.llm.sleep(os.environ.get('VLLM_SLEEP_MODE', 1))\n"
            )

            patched = patched[: match.start()] + wrapped + patched[match.end() :]

        function = patched

    _mm_alignment = """
        if "mm_token_type_ids" in forward_kwargs or "image_grid_thw" in forward_kwargs:
            _mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                prompt_completion_ids,
                forward_kwargs.get("mm_token_type_ids", None),
                completion_ids = completion_ids,
            )
            if _mm_token_type_ids is not None:
                forward_kwargs["mm_token_type_ids"] = _mm_token_type_ids
"""
    _tool_image_marker = (
        "        # For VLM tool images: build token type IDs from the full prompt_completion_ids."
    )
    if _tool_image_marker in function:
        function = function.replace(_tool_image_marker, _mm_alignment + "\n" + _tool_image_marker)
    else:
        _tt_search = (
            'if "token_type_ids" in forward_kwargs:\n'
            '            token_type_ids = forward_kwargs["token_type_ids"]\n'
            '            forward_kwargs["token_type_ids"] = torch.cat(\n'
            "                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1\n"
            "            )"
        )
        function = function.replace(_tt_search, _tt_search + "\n" + _mm_alignment.rstrip())

    _save_search = (
        'if "token_type_ids" in forward_kwargs:\n'
        '            output["token_type_ids"] = forward_kwargs["token_type_ids"]'
    )
    if 'output["mm_token_type_ids"]' not in function:
        _save_replace = (
            _save_search + "\n"
            '        if "mm_token_type_ids" in forward_kwargs:\n'
            '            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]'
        )
        function = function.replace(_save_search, _save_replace)

    if re.search(r"\btool_mask\b", function) and 'output["tool_mask"]' not in function:
        function = function.replace(
            "        return output",
            "        if tool_mask is not None:\n"
            '            output["tool_mask"] = tool_mask\n'
            "        return output",
        )

    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate_and_score_completions)


# Fix {"reasoning_effort" : "high"} not applied
def grpo_trainer_fix_maybe_apply_chat_template(function_name, function):
    spaces = function.find("def ")
    if spaces % 4 != 0:
        return function
    spaces += 4
    replacement = """
        _chat_template_ = getattr(self.processing_class, "chat_template", None)
        if _chat_template_ is None: _chat_template_ = ""
        _supported_keys_ = set(("prompt", "chosen", "rejected", "completion", "messages", "label"))
        _batch_chat_kwargs_ = getattr(self, "_unsloth_batch_chat_kwargs", None)

        prompts_text = []
        for _idx_, _example_ in enumerate(__INPUTS__REPLACEMENT__):
            _tokenizer_kwargs_ = {}
            if type(_example_) is not dict:
                _example_ = {"prompt": _example_}
            _left_keys_ = _example_.keys() - _supported_keys_
            for k in _left_keys_:
                if k in _chat_template_:
                    v = _example_[k]
                    if type(v) is str:
                        _tokenizer_kwargs_[k] = v
            if _batch_chat_kwargs_ is not None and _idx_ < len(_batch_chat_kwargs_):
                for _bk_, _bv_ in _batch_chat_kwargs_[_idx_].items():
                    if _bk_ not in _tokenizer_kwargs_:
                        _tokenizer_kwargs_[_bk_] = _bv_
            _x_ = maybe_apply_chat_template(_example_, self.processing_class, **_tokenizer_kwargs_)["prompt"]
            prompts_text.append(_x_)
    """
    replacement = textwrap.dedent(replacement).strip()
    replacement = textwrap.indent(replacement, spaces * " ")
    replacement = f"\n{replacement}\n"
    what = 'prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]'
    function = function.replace(what, replacement.replace("__INPUTS__REPLACEMENT__", "inputs"))

    """prompts_text = [
        maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
    ]"""
    function = re.sub(
        r"prompts_text = \["
        r"[\s]{0,}"
        r"maybe_apply_chat_template\(\{[\"\']prompt[\"\'][\s]{0,}\:[\s]{0,}prompt[\s]{0,}\}[\s]{0,}\,[\s]{0,}self\.processing_class\)"
        r"\[[\"\']prompt[\"\']\] for prompt in prompts"
        r"[\s]{0,}"
        r"\]",
        replacement.replace("__INPUTS__REPLACEMENT__", "prompts"),
        function,
    )
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_fix_maybe_apply_chat_template)


def grpo_trainer__move_model_to_vllm(function_name, function):
    if function_name != "_move_model_to_vllm":
        return function

    def _move_model_to_vllm(self, *args, **kwargs):
        return None

    function = inspect.getsource(_move_model_to_vllm)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision.
def grpo_trainer__get_per_token_logps(function_name, function):
    if function_name != "_get_per_token_logps":
        return function

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_efficient = False,
    ):
        if True:  # os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return None  # Unsloth efficient GRPO
        _unsloth_grpo_autocast(self)

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(
            device_type = DEVICE_TYPE,
            dtype = self._autocast_dtype,
            enabled = getattr(self, "_autocast_enabled", True),
        ):
            # logits_to_keep gets 1 added because the last logit of the sequence is excluded later.
            logits = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                logits_to_keep = logits_to_keep + 1,
            ).logits
            return logits
            # transformers <= 4.48 does not support logits_to_keep, so drop the logits here; see
            # huggingface/trl#2770.



    function = inspect.getsource(_get_per_token_logps)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)


def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
    if function_name != "_get_per_token_logps_and_entropies":
        return function

    # Copied from the _get_per_token_logps replacement above; for now this returns None anyway.
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size = None,
        compute_entropy = False,
        compute_efficient = False,
        *args,
        **kwargs,
    ):
        # All Unsloth code in this function is licensed under AGPL3.
        if compute_efficient:
            return None, None
        else:
            _unsloth_grpo_autocast(self)

            compute_aux_loss = kwargs.get("compute_aux_loss", None)

            pixel_values, image_grid_thw = (
                kwargs.get("pixel_values", None),
                kwargs.get("image_grid_thw", None),
            )
            pixel_attention_mask, image_sizes = (
                kwargs.get("pixel_attention_mask", None),
                kwargs.get("image_sizes", None),
            )
            num_images = kwargs.get("num_images", None)
            # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
            token_type_ids = kwargs.get("token_type_ids", None)
            mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
            if mm_token_type_ids is not None or image_grid_thw is not None:
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class, input_ids, mm_token_type_ids
                )

            unwrapped_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper = False)

            lm_head = self.model.get_output_embeddings().weight

            # Size on the dtype the forward actually runs in: with autocast off that is the model's own dtype.
            forward_dtype = (
                self._autocast_dtype if getattr(self, "_autocast_enabled", True) else lm_head.dtype
            )
            dtype_bytes = 16 if forward_dtype in [torch.float16, torch.bfloat16] else 32
            total_rows = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = lm_head.shape[1]
            vocab_dim = lm_head.shape[0]

            if self.args.unsloth_grpo_mini_batch is None:
                B, multiplier = autotune_batch_and_chunks(
                    total_rows,
                    seq_len,
                    hidden_dim,
                    vocab_dim,
                    dtype_bytes,
                    self.args.unsloth_logit_chunk_multiplier,
                )
                B = total_rows // B
            else:
                B = self.args.unsloth_grpo_mini_batch

                if self.args.unsloth_logit_chunk_multiplier is None:
                    multiplier = max(4, seq_len // 4096)
                else:
                    multiplier = self.args.unsloth_logit_chunk_multiplier

            all_logprobs_list = []
            if pixel_values is None:
                left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(
                    input_ids, logits_to_keep, self.processing_class.pad_token_id
                )
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
                input_ids = left_pack_padding(input_ids, self.processing_class.pad_token_id)
                attention_mask = input_ids != self.processing_class.pad_token_id
                attention_mask = attention_mask.to(attention_mask.dtype)
            else:
                max_left_pad = 0

            def slice_sample_axis(value, start, end):
                if value is None:
                    return None
                return value[start:end]

            import math

            total_samples = input_ids.shape[0]
            batch_size = math.ceil(total_samples / B)
            if isinstance(num_images, torch.Tensor):
                num_images = num_images.detach().cpu().reshape(-1).tolist()
            if image_grid_thw is not None and pixel_values is not None and num_images is not None:
                rows_per_image = image_grid_thw.prod(dim = -1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                # cum_rows is indexed via .item() inside the per-chunk loop, so keeping it on CPU avoids a
                # per-iteration GPU->CPU sync.
                cum_rows = torch.cat(
                    [
                        torch.tensor([0], device = rows_per_sample.device),
                        rows_per_sample.cumsum(0),
                    ]
                ).cpu()
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            else:
                cum_rows = None
                cum_imgs = None

            def _first_dim_len(value):
                if value is None:
                    return None
                if hasattr(value, "shape"):
                    return value.shape[0]
                try:
                    return len(value)
                except TypeError:
                    return None

            total_images = sum(num_images) if num_images is not None else None
            _image_sizes_n = _first_dim_len(image_sizes)

            input_ids_chunks = []
            attention_mask_chunks = []
            pixel_values_chunks = []
            image_grid_thw_chunks = []
            pixel_attention_mask_chunks = []
            image_sizes_chunks = []
            token_type_ids_chunks = []
            mm_token_type_ids_chunks = []

            current_pixel_idx = 0
            # TRL 0.23.0 batching logic.
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)

                input_ids_chunks.append(input_ids[start:end])
                attention_mask_chunks.append(attention_mask[start:end])
                token_type_ids_chunks.append(slice_sample_axis(token_type_ids, start, end))
                mm_token_type_ids_chunks.append(slice_sample_axis(mm_token_type_ids, start, end))

                if image_grid_thw is not None and pixel_values is not None:
                    if num_images is None:
                        grid_slice = image_grid_thw[start:end]
                        batch_pixel_count = grid_slice.prod(dim = -1).sum().item()
                        start_pixel_idx = current_pixel_idx
                        end_pixel_idx = current_pixel_idx + batch_pixel_count
                        current_pixel_idx = end_pixel_idx
                        img_start = img_end = None
                    else:
                        start_pixel_idx = cum_rows[start].item()
                        end_pixel_idx = cum_rows[end].item()
                        img_start = cum_imgs[start].item()
                        img_end = cum_imgs[end].item()
                        grid_slice = image_grid_thw[img_start:img_end]
                    image_grid_thw_chunks.append(grid_slice)

                    pixel_values_chunks.append(pixel_values[start_pixel_idx:end_pixel_idx])

                    if image_sizes is None:
                        image_sizes_chunks.append(None)
                    elif (
                        num_images is not None
                        and _image_sizes_n == total_images
                        and img_start is not None
                    ):
                        image_sizes_chunks.append(image_sizes[img_start:img_end])
                    else:
                        image_sizes_chunks.append(slice_sample_axis(image_sizes, start, end))

                    if pixel_attention_mask is None:
                        pixel_attention_mask_chunks.append(None)
                    elif (
                        num_images is not None
                        and img_start is not None
                        and pixel_attention_mask.shape[0] == image_grid_thw.shape[0]
                    ):
                        pixel_attention_mask_chunks.append(pixel_attention_mask[img_start:img_end])
                    elif (
                        pixel_attention_mask.shape[0] == pixel_values.shape[0]
                        and pixel_attention_mask.shape[0] != input_ids.shape[0]
                    ):
                        pixel_attention_mask_chunks.append(
                            pixel_attention_mask[start_pixel_idx:end_pixel_idx]
                        )
                    else:
                        pixel_attention_mask_chunks.append(pixel_attention_mask[start:end])

                else:
                    pixel_values_chunks.append(None)
                    image_grid_thw_chunks.append(None)
                    pixel_attention_mask_chunks.append(None)
                    image_sizes_chunks.append(slice_sample_axis(image_sizes, start, end))

            temperature = self.temperature
            model_config = _unsloth_get_model_config(model)
            if detect_logit_transforms is not None:
                # model_config, not model: under DDP/Accelerate `model` is a wrapper that does not forward
                # .config, so the helper would report zeros.
                _transforms = detect_logit_transforms(model_config)
                logit_softcapping = _transforms["logit_softcapping"]
                logit_scale_multiply = _transforms["logit_scale_multiply"]
                logit_scale_divide = _transforms["logit_scale_divide"]
            else:
                logit_softcapping = _unsloth_get_final_logit_softcapping(model)
                logit_scale_multiply = getattr(model_config, "logit_scale", 0)
                if logit_scale_multiply is None:
                    logit_scale_multiply = 0
                logit_scale_divide = getattr(model_config, "logits_scaling", 0)
                if logit_scale_divide is None:
                    logit_scale_divide = 0

            zipped_inputs = zip(
                input_ids_chunks,
                attention_mask_chunks,
                pixel_values_chunks,
                image_grid_thw_chunks,
                pixel_attention_mask_chunks,
                image_sizes_chunks,
                token_type_ids_chunks,
                mm_token_type_ids_chunks,
            )
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen [1, sum L]
            # forward replaces the padded [B, Lmax] loop and fixes the left-pad RoPE error. Self-verified
            # against the per-row forward, re-checked as T grows, and falls back if a backend ignores
            # packed_seq_lengths.
            logprobs = None

            # PrefixGrouper (GRPO shared-prompt dedup, default ON): G completions share the prompt, so
            # storing it once behind a FlexAttention shared-prefix mask cuts the trunk forward from
            # G*(P+R) to P+G*R tokens. Gated by UNSLOTH_GRPO_PREFIX_GROUPER, a tok_r auto-gate and a
            # first-use self-verify, so a mask/isolation regression cannot ship silently.
            _pg_result = None
            _pg_use = False
            _pg_skip_pk = False  # once a shape is PG-verified, skip the full-row forward
            _pg_forward_fn = None  # deferred PG forward (runs at the verify site below)
            _pg_num_gen = getattr(self, "num_generations", None)
            # Env gate hoisted to module level (mirrored via RL_PRE_ITEMS). Skip PG under vLLM: the rollout
            # dominates the step, so PG saves little and its self-verify is net overhead.
            _pg_engage = (
                UNSLOTH_GRPO_PREFIX_GROUPER_ON
                and not getattr(self, "use_vllm", False)
                and not getattr(unwrapped_model, "_unsloth_prefix_grouper_nograd_disabled", False)
            )
            if _pg_engage:
                try:
                    # Skip softcap models (the flex kernel never applies attn_logit_softcapping) and hybrid SSM /
                    # MoE models: only the threaded attention forwards get shared-prefix isolation, so a decoder
                    # that does not forward prefix_seg_info leaks suffixes across completions. PG also rides on
                    # sequence packing, so it needs the same zoo masked-column guard.
                    _pg_cfg = getattr(unwrapped_model, "config", None)
                    _pg_engage = (
                        _pg_enabled_fn()
                        and UNSLOTH_ZOO_HAS_MASKED_COL_GUARD
                        and pixel_values is None
                        and token_type_ids is None
                        and mm_token_type_ids is None
                        and _pg_num_gen is not None
                        and _pg_num_gen >= 2
                        and not getattr(_pg_cfg, "attn_logit_softcapping", None)
                        # Normal backends apply config.attention_dropout in training; the flex path is deterministic,
                        # so skip PG when it is set.
                        and not getattr(_pg_cfg, "attention_dropout", 0)
                        and not any(
                            getattr(_pg_cfg, _pg_a, None) is not None
                            for _pg_a in (
                                "mamba_d_ssm",
                                "mamba_d_state",
                                "mamba_expand",
                                "num_experts",
                                "num_local_experts",
                                "n_routed_experts",
                                "moe_intermediate_size",
                            )
                        )
                    )
                except Exception:
                    _pg_engage = False
            if _pg_engage:
                try:
                    _pg_pad = self.processing_class.pad_token_id
                    # Cap the PG span (P+max(R)) at the sliding window, like the packed _pk_sw guard.
                    _pg_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    if not (isinstance(_pg_sw, int) and _pg_sw > 0):
                        _pg_sw = None
                    _pg_layout = _pg_build_layout(
                        input_ids,
                        logits_to_keep,
                        _pg_pad,
                        _pg_num_gen,
                        left_pad_tokens_per_prompt,
                        max_segment_cap = _pg_sw,
                    )
                    _pg_unsafe = getattr(
                        unwrapped_model, "_unsloth_prefix_grouper_nograd_unsafe", None
                    )
                    if _pg_unsafe is None:
                        _pg_unsafe = set()
                    if _pg_layout is not None and _pg_layout.signature not in _pg_unsafe:
                        _pg_sig = _pg_layout.signature
                        _pg_verified = getattr(
                            unwrapped_model, "_unsloth_prefix_grouper_nograd_verified", None
                        )
                        if _pg_verified is None:
                            _pg_verified = set()
                        _pg_chunks = max(1, total_rows * multiplier)

                        def _pg_run_forward(_pg_layout = _pg_layout, _pg_chunks = _pg_chunks):
                            with _get_inference_mode_context_manager(model):
                                with torch.amp.autocast(
                                    device_type = "cuda",
                                    dtype = self._autocast_dtype,
                                    enabled = getattr(self, "_autocast_enabled", True),
                                ):
                                    _pg_hidden = unwrapped_model(
                                        input_ids = _pg_layout.flat_ids,
                                        position_ids = _pg_layout.position_ids,
                                        prefix_seg_info = _pg_layout.prefix_seg_info,
                                        use_cache = False,
                                    ).logits
                                    _pg_r = _pg_layout.extract_logps(
                                        _pg_hidden,
                                        lm_head,
                                        chunked_hidden_states_selective_log_softmax,
                                        _pg_chunks,
                                        logit_scale_multiply,
                                        logit_scale_divide,
                                        logit_softcapping,
                                        temperature,
                                    )
                                    _pg_hidden = None  # release before any verify forward
                            device_synchronize()
                            # Clip to the loss window [B, logits_to_keep+max_left_pad].
                            _pg_w = logits_to_keep + max_left_pad
                            if _pg_r.shape[1] > _pg_w:
                                _pg_r = _pg_r[:, -_pg_w:]
                            return _pg_r

                        # Trust only within the verified envelope: re-verify when T or the longest segment grows, like
                        # the packed path.
                        _pg_T = int(_pg_layout.flat_ids.shape[1])
                        _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                        _pg_env = (
                            _pg_verified.get(_pg_sig) if isinstance(_pg_verified, dict) else None
                        )
                        if (not _pg_verify_on()) or (
                            _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                        ):
                            # Trusted shape: run PG now and skip the full-row forward below.
                            _pg_result = _pg_run_forward()
                            _pg_use = True
                            _pg_skip_pk = True
                        else:
                            # Unverified shape: defer the forward until the packed reference exists, so a
                            # declined packed
                            # path never wastes a whole-batch PG forward.
                            _pg_forward_fn = _pg_run_forward
                except Exception as _pg_err:
                    _pg_result = None
                    _pg_use = False
                    _pg_skip_pk = False
                    _pg_forward_fn = None
                    # A FlexAttention/Triton compile failure or OOM here is GPU-wide, not layout-specific, so
                    # retrying every step just re-pays it. Disable PG persistently; the packed/padded path below
                    # still gives the exact result.
                    unwrapped_model._unsloth_prefix_grouper_nograd_disabled = True
                    if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO PrefixGrouper (no-grad) disabled (fell back to packed): {_pg_err!r}",
                            flush = True,
                        )

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen
            # block-diagonal forward replaces the padded loop exactly and fixes its left-pad RoPE error.
            # Self-verified, re-checked as T grows, falls back if a backend ignores packed_seq_lengths, and
            # lm_head runs on completion positions only.
            _pk_result = None
            _pk_use = False
            _pk_enabled = UNSLOTH_GRPO_SEQ_PACKING_ON
            # Without zoo#840's masked-column guard, zeroed prompt/pad columns turn NaN in exp().
            _pk_enabled = _pk_enabled and UNSLOTH_ZOO_HAS_MASKED_COL_GUARD
            _pk_ok = getattr(unwrapped_model, "_unsloth_seq_packing_nograd_ok", None)
            if (
                _pk_enabled
                and not _pg_skip_pk
                and pixel_values is None
                and token_type_ids is None
                and mm_token_type_ids is None
                and _pk_ok is not False
            ):
                try:
                    _pk_pad = self.processing_class.pad_token_id
                    _pk_keep = input_ids != _pk_pad
                    _pk_len = _pk_keep.sum(dim = 1)
                    _pk_len_cpu = _pk_len.tolist()  # single GPU->CPU sync, reused below
                    _pk_nz_cpu = [_n for _n in _pk_len_cpu if _n > 0]
                    _pk_flat = input_ids[_pk_keep].unsqueeze(0)
                    _pk_T = _pk_flat.shape[1]
                    _pk_L = input_ids.shape[1]
                    _pk_W = logits_to_keep + max_left_pad
                    _pk_maxseg = max(_pk_nz_cpu) if _pk_nz_cpu else 0
                    # Sliding-window models lose the per-sequence local window in a packed stream.
                    _pk_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    _pk_sw_ok = not (isinstance(_pk_sw, int) and _pk_sw > 0 and _pk_maxseg > _pk_sw)
                    # Per-row completion mask (same as the loss); prompt-only rows count as inactive.
                    _pk_cmask = create_completion_attention_mask(
                        input_ids[:, -_pk_W:], left_pad_tokens_per_prompt, max_left_pad, _pk_pad
                    )
                    _pk_active = int(_pk_cmask.any(dim = 1).sum())
                    # Skip the packed forward entirely at known-unsafe lengths, avoiding a wasted pass or OOM.
                    _pk_unsafe = getattr(
                        unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                    )
                    # Cap the flattened forward at one padded [batch_size, seq_len] mini-batch's token budget;
                    # anything larger uses the chunked padded loop.
                    _pk_cap = batch_size * seq_len
                    if (
                        _pk_T >= 2
                        and _pk_T <= _pk_cap
                        and len(_pk_nz_cpu) > 0
                        and _pk_sw_ok
                        and not (_pk_unsafe is not None and _pk_T >= _pk_unsafe)
                        and (_pk_ok is True or _pk_active >= 2)
                    ):
                        # reset 0-based position_ids per segment
                        _pk_pos = (_pk_keep.cumsum(dim = 1) - 1)[_pk_keep].unsqueeze(0)
                        _pk_chunks = max(1, total_rows * multiplier)
                        _pk_nz_idx = _pk_keep.nonzero(
                            as_tuple = False
                        )  # [T, 2] = (row, col), row-major
                        _pk_within = _pk_nz_idx[1:, 0] == _pk_nz_idx[:-1, 0]  # [T-1]
                        # Per-row completion start after left-packing, matching create_completion_attention_mask.
                        _pk_cstart = (_pk_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                        _pk_ctgt = (_pk_nz_idx[1:, 1] >= _pk_cstart[_pk_nz_idx[1:, 0]]) & _pk_within
                        with _get_inference_mode_context_manager(model):
                            with torch.amp.autocast(
                                device_type = "cuda",
                                dtype = self._autocast_dtype,
                                enabled = getattr(self, "_autocast_enabled", True),
                            ):
                                # use_cache=False: a KV cache silently disables varlen packing.
                                _pk_hidden = unwrapped_model(
                                    input_ids = _pk_flat,
                                    position_ids = _pk_pos,
                                    packed_seq_lengths = torch.tensor(
                                        _pk_nz_cpu, dtype = torch.int32, device = input_ids.device
                                    ),
                                    use_cache = False,
                                ).logits
                                _pk_out = _pk_hidden[0, :-1, :][_pk_ctgt].unsqueeze(0)
                                _pk_ids = _pk_flat[0, 1:][_pk_ctgt].unsqueeze(0)
                                # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                                if _unsloth_grpo_returns_hidden_states(
                                    unwrapped_model, _pk_out, lm_head
                                ):
                                    _pk_sel = chunked_hidden_states_selective_log_softmax(
                                        _pk_out,
                                        lm_head,
                                        _pk_ids,
                                        _pk_chunks,
                                        logit_scale_multiply,
                                        logit_scale_divide,
                                        logit_softcapping,
                                        temperature,
                                    )[0]
                                else:
                                    # Model returned logits directly - scaling/softcapping already applied by
                                    # model forward
                                    _pk_sel = chunked_selective_log_softmax(
                                        _pk_out,
                                        _pk_ids,
                                        temperature,
                                        _pk_chunks,
                                    )[0]
                        # GPT-OSS offload race guard, matching the padded loop.
                        device_synchronize()
                        # Scatter each logprob back to its (row, col) so [:, -_pk_W:] matches the padded path.
                        _pk_tgt = (_pk_nz_idx[1:, 0] * _pk_L + _pk_nz_idx[1:, 1])[_pk_ctgt]
                        _pk_result = (
                            torch.zeros(
                                total_rows * _pk_L,
                                dtype = torch.float32,
                                device = input_ids.device,
                            )
                            .index_put((_pk_tgt,), _pk_sel.to(torch.float32))
                            .view(total_rows, _pk_L)[:, -_pk_W:]
                        )
                        # Re-verify when T or the longest segment grows past the verified envelope; a LongRoPE cache
                        # switch can change the result.
                        _pk_vT = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_T", 0)
                        )
                        _pk_vS = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_seg", 0)
                        )
                        # Debug: hand-edit this condition to force re-verify every step.
                        if _pk_ok is True and _pk_T <= _pk_vT and _pk_maxseg <= _pk_vS:
                            _pk_use = True  # already verified for this shape
                        else:
                            # verify against the per-row forward (ground truth)
                            _pk_ref = torch.zeros_like(_pk_result)
                            with _get_inference_mode_context_manager(model):
                                with torch.amp.autocast(
                                    device_type = "cuda",
                                    dtype = self._autocast_dtype,
                                    enabled = getattr(self, "_autocast_enabled", True),
                                ):
                                    for _pk_i in range(total_rows):
                                        _pk_ni = _pk_len_cpu[_pk_i]
                                        if _pk_ni < 2:
                                            continue
                                        _pk_rmask = _pk_keep[_pk_i]
                                        _pk_real = input_ids[_pk_i][_pk_rmask].unsqueeze(0)
                                        _pk_rpos = torch.arange(
                                            _pk_ni, device = input_ids.device
                                        ).unsqueeze(0)
                                        _pk_rh = unwrapped_model(
                                            input_ids = _pk_real,
                                            position_ids = _pk_rpos,
                                            use_cache = False,
                                        ).logits
                                        _pk_rout = _pk_rh[:, :-1, :]
                                        # Hidden states or logits? Logits mean the forward already applied
                                        # scaling/softcapping.
                                        if _unsloth_grpo_returns_hidden_states(
                                            unwrapped_model, _pk_rout, lm_head
                                        ):
                                            _pk_rsel = chunked_hidden_states_selective_log_softmax(
                                                _pk_rout,
                                                lm_head,
                                                _pk_real[:, 1:],
                                                1,
                                                logit_scale_multiply,
                                                logit_scale_divide,
                                                logit_softcapping,
                                                temperature,
                                            )[0]
                                        else:
                                            # Model returned logits directly - scaling/softcapping already
                                            # applied by model forward
                                            _pk_rsel = chunked_selective_log_softmax(
                                                _pk_rout,
                                                _pk_real[:, 1:],
                                                temperature,
                                                1,
                                            )[0]
                                        _pk_rcols = _pk_rmask.nonzero(as_tuple = False).squeeze(1)[
                                            1:
                                        ] - (_pk_L - _pk_W)
                                        _pk_rkeep = _pk_rcols >= 0
                                        _pk_ref[_pk_i, _pk_rcols[_pk_rkeep]] = _pk_rsel[
                                            _pk_rkeep
                                        ].to(torch.float32)
                            device_synchronize()
                            # Compare over the loss-mask region only.
                            _pk_cm = _pk_cmask.float()
                            _pk_diff = float(((_pk_result - _pk_ref).abs() * _pk_cm).max())
                            if UNSLOTH_ENABLE_LOGGING:
                                print(
                                    f"[Unsloth] GRPO seq-packing (no-grad) verify: T={_pk_T} maxseg={_pk_maxseg} packed-vs-perrow max|d|={_pk_diff:.4f}",
                                    flush = True,
                                )
                            # Kernel-noise floor is ~0.25; cross-sample contamination is >= 2.4.
                            if _pk_diff < 7e-1:
                                unwrapped_model._unsloth_seq_packing_nograd_ok = True
                                # Widen the trusted shape only when at least 2 completion rows exercised cross-
                                # sample packing;
                                # a single row proves nothing.
                                if _pk_active >= 2:
                                    unwrapped_model._unsloth_seq_packing_nograd_verified_T = max(
                                        _pk_vT, _pk_T
                                    )
                                    unwrapped_model._unsloth_seq_packing_nograd_verified_seg = max(
                                        _pk_vS, _pk_maxseg
                                    )
                                _pk_ok = True
                                _pk_use = True
                            else:
                                _pk_use = False
                                if _pk_diff >= 1.5:
                                    # Contamination (attention ignores the packed mask): disable packing.
                                    unwrapped_model._unsloth_seq_packing_nograd_ok = False
                                else:
                                    # Likely a length boundary (LongRoPE): mark unsafe, keep smaller shapes.
                                    unwrapped_model._unsloth_seq_packing_nograd_unsafe_T = (
                                        _pk_T if _pk_unsafe is None else min(_pk_unsafe, _pk_T)
                                    )
                                if UNSLOTH_ENABLE_LOGGING:
                                    print(
                                        f"[Unsloth] GRPO seq-packing (no-grad) fell back at T={_pk_T} (diff={_pk_diff:.3f})",
                                        flush = True,
                                    )
                except Exception as _pk_err:
                    # Any failure: drop intermediates, use the padded loop, do not retry.
                    _pk_hidden = None
                    _pk_sel = None
                    _pk_result = None
                    _pk_use = False
                    if isinstance(_pk_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    unwrapped_model._unsloth_seq_packing_nograd_ok = False
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO sequence-packing (no-grad) disabled (fell back to padded): {_pk_err!r}",
                            flush = True,
                        )
            # PrefixGrouper first-use self-verify (no-grad): compare the untrusted PG result to the packed
            # result over the completion mask. Below tol_ok trust the structure, at or above TOL_KILL mark
            # it unsafe forever, borderline falls back for this shape.
            if _pg_forward_fn is not None and not _pg_use:
                if _pk_use and _pk_result is not None:
                    try:
                        # Deferred PG forward, run only now that the packed reference exists.
                        _pg_result = _pg_forward_fn()
                        _pg_W2 = logits_to_keep + max_left_pad
                        _pg_cm = create_completion_attention_mask(
                            input_ids[:, -_pg_W2:],
                            left_pad_tokens_per_prompt,
                            max_left_pad,
                            self.processing_class.pad_token_id,
                        ).float()
                        _pg_a = _pg_result[:, -_pg_W2:].float()
                        _pg_b = _pk_result[:, -_pg_W2:].float()
                        _pg_diff = float(((_pg_a - _pg_b).abs() * _pg_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (no-grad) verify: sig={_pg_layout.signature} "
                                f"shared-prefix vs full-row-packed max|d|={_pg_diff:.4f}",
                                flush = True,
                            )
                        if _pg_diff < _pg_tol_ok():
                            _pg_v = getattr(
                                unwrapped_model, "_unsloth_prefix_grouper_nograd_verified", None
                            )
                            if not isinstance(_pg_v, dict):
                                _pg_v = {}
                            _pg_vT = int(_pg_layout.flat_ids.shape[1])
                            _pg_vS = int(_pg_layout.position_ids.max()) + 1
                            _pg_old = _pg_v.get(_pg_layout.signature, (0, 0))
                            _pg_v[_pg_layout.signature] = (
                                max(_pg_vT, _pg_old[0]),
                                max(_pg_vS, _pg_old[1]),
                            )
                            unwrapped_model._unsloth_prefix_grouper_nograd_verified = _pg_v
                            _pg_use = True
                        else:
                            _pg_u = getattr(
                                unwrapped_model, "_unsloth_prefix_grouper_nograd_unsafe", None
                            )
                            if _pg_u is None:
                                _pg_u = set()
                            if _pg_diff >= _PG_TOL_KILL:
                                _pg_u.add(_pg_layout.signature)
                                unwrapped_model._unsloth_prefix_grouper_nograd_unsafe = _pg_u
                            _pg_use = False
                    except Exception as _pg_err3:
                        _pg_result = None
                        _pg_use = False
                        if isinstance(_pg_err3, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (no-grad) verify failed (fell back to packed): {_pg_err3!r}",
                                flush = True,
                            )
                # No packed reference (packing off or failed) means this cannot be verified, so fall back.

            if _pg_use and _pg_result is not None:
                logprobs = _pg_result  # PrefixGrouper verified/trusted -> skip the loop
                zipped_inputs = []
            elif _pk_use and _pk_result is not None:
                logprobs = _pk_result  # verified -> skip the loop
                zipped_inputs = []
            else:
                # free packed intermediates before running the padded loop
                _pk_hidden = _pk_sel = _pk_result = _pk_ref = None

            with _get_inference_mode_context_manager(model):
                for (
                    input_ids_chunk,
                    attention_mask_chunk,
                    pixel_values_chunk,
                    image_grid_thw_chunk,
                    pixel_attention_mask_chunk,
                    image_sizes_chunk,
                    token_type_ids_chunk,
                    mm_token_type_ids_chunk,
                ) in zipped_inputs:
                    _extra_vision_kwargs = {}
                    if token_type_ids_chunk is not None:
                        _extra_vision_kwargs["token_type_ids"] = token_type_ids_chunk
                    if mm_token_type_ids_chunk is not None:
                        _extra_vision_kwargs["mm_token_type_ids"] = mm_token_type_ids_chunk
                    with torch.amp.autocast(
                        device_type = "cuda",
                        dtype = self._autocast_dtype,
                        enabled = getattr(self, "_autocast_enabled", True),
                    ):
                        if pixel_values is None:
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                pixel_values = pixel_values_chunk,
                                image_grid_thw = image_grid_thw_chunk,
                                pixel_attention_mask = pixel_attention_mask_chunk,
                                image_sizes = image_sizes_chunk,
                                **_extra_vision_kwargs,
                            )

                            logits_chunk = outputs.logits
                            del outputs

                            completion_input_ids_chunk = input_ids_chunk[
                                :, -(logits_to_keep + max_left_pad) :
                            ]
                            logits_chunk = logits_chunk[
                                :, -(logits_to_keep + max_left_pad + 1) :, :
                            ]
                            logits_chunk = logits_chunk[:, :-1, :]
                            # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                            if _unsloth_grpo_returns_hidden_states(
                                unwrapped_model, logits_chunk, lm_head
                            ):
                                logprobs_chunk = chunked_hidden_states_selective_log_softmax(
                                    logits_chunk,
                                    lm_head,
                                    completion_input_ids_chunk,
                                    chunks = input_ids_chunk.shape[0] * multiplier,
                                    logit_scale_multiply = logit_scale_multiply,
                                    logit_scale_divide = logit_scale_divide,
                                    logit_softcapping = logit_softcapping,
                                    temperature = temperature,
                                )
                            else:
                                # Model returned logits directly - scaling/softcapping already applied by model
                                # forward
                                # Model returned logits directly - scaling/softcapping already applied by model
                                # forward
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                    input_ids_chunk.shape[0] * multiplier,
                                )
                        else:
                            # VLMs do not take the optimized path in models/, so they never hit the Flash Attn
                            # left-padding
                            # issue.
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                pixel_values = pixel_values_chunk,
                                image_grid_thw = image_grid_thw_chunk,
                                pixel_attention_mask = pixel_attention_mask_chunk,
                                image_sizes = image_sizes_chunk,
                                logits_to_keep = logits_to_keep + 1,
                                **_extra_vision_kwargs,
                            )

                            logits_chunk = outputs.logits
                            del outputs

                            logits_chunk = logits_chunk[:, :-1, :]
                            completion_input_ids_chunk = input_ids_chunk[:, -logits_to_keep:]
                            # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                            if _unsloth_grpo_returns_hidden_states(
                                unwrapped_model, logits_chunk, lm_head
                            ):
                                logprobs_chunk = chunked_hidden_states_selective_log_softmax(
                                    logits_chunk,
                                    lm_head,
                                    completion_input_ids_chunk,
                                    chunks = input_ids_chunk.shape[0] * multiplier,
                                    logit_scale_multiply = logit_scale_multiply,
                                    logit_scale_divide = logit_scale_divide,
                                    logit_softcapping = logit_softcapping,
                                    temperature = temperature,
                                )
                            else:
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                )
                    # Avoids a race with GPT OSS offload_embbed=True; it does not appear to slow models down.
                    device_synchronize()
                    all_logprobs_list.append(logprobs_chunk)
                if logprobs is None:  # padded fallback when packing was not used
                    logprobs = torch.cat(all_logprobs_list, dim = 0)

                entropies = None

            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
            # aux loss is unused: off by default (router_aux_loss_coef = 0 in models/rl.py) and explicit
            # opt-in is rejected at trainer init, so it is always None. Kept for TRL >= 1.7.0's 3-tuple.
            aux_loss = None
            return logprobs.detach(), entropies, aux_loss  # logps, entropies, aux_loss
            # transformers <= 4.48 does not support logits_to_keep, so drop the logits here; see
            # huggingface/trl#2770.



    function = inspect.getsource(_get_per_token_logps_and_entropies)
    if trl_version < Version("1.7.0"):
        # TRL < 1.7.0 unpacks (logps, entropies) while TRL >= 1.7.0 unpacks (logps, entropies,
        # aux_loss), so drop the third element to match. The regex tolerates comment/whitespace drift
        # and fails loud rather than ship a 3-tuple to older TRL.
        new_function, n = re.subn(
            r"return (logprobs\.detach\(\), entropies), aux_loss[^\n]*",
            r"return \1  # logps, entropies",
            function,
        )
        if n != 1:
            raise RuntimeError(
                "Unsloth GRPO: could not downgrade the per-token-logps return to a "
                f"2-tuple for TRL {trl_version} (matched {n} times, expected 1). The "
                "return line changed; update the arity gate in rl_replacements.py."
            )
        function = new_function
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps_and_entropies)


def _unsloth_get_model_config(model):
    """Return HuggingFace model config, unwrapping DDP/Accelerate wrappers."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return config


def _unsloth_get_final_logit_softcapping(model):
    """Return final_logit_softcapping for a model config, falling back to the
    nested text sub-config for composite models. Handles both:
      - Gemma-4-style configs where the attribute lives on ``config.text_config``
      - T5Gemma-style composite configs where the text sub-config is only
        reachable via ``config.get_text_config()``
    Returns 0 if unset, matching the previous behaviour.
    """
    config = _unsloth_get_model_config(model)
    if config is None:
        return 0
    softcap = getattr(config, "final_logit_softcapping", None)
    if softcap is None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is None:
            get_text_config = getattr(config, "get_text_config", None)
            if callable(get_text_config):
                try:
                    text_cfg = get_text_config()
                except (TypeError, ValueError):
                    text_cfg = None
        if text_cfg is not None and text_cfg is not config:
            softcap = getattr(text_cfg, "final_logit_softcapping", None)
    return 0 if softcap is None else softcap


def _unsloth_grpo_returns_hidden_states(model, tensor, lm_head):
    """Does ``tensor`` (a forward's ``.logits``) carry hidden states or real logits?

    ``_get_per_token_logps_and_entropies`` sets ``UNSLOTH_RETURN_HIDDEN_STATES=1``,
    but only a forward that honours the name hands hidden states back as
    ``.logits``; any other forward returns a real ``[.., vocab]`` tensor that must
    not reach the ``lm_head`` matmul.

    Primary test is an explicit signal that the forward honours the flag. Two
    exist, both set outside this file:

    * ``__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__`` on the generated class,
      written by ``unsloth_zoo.compiler.create_standalone_class`` exactly when
      ``apply_fused_lm_head`` gave that forward its own ``RETURN_HIDDEN_STATES``
      branch.
    * ``_unsloth_grpo_hidden_states_forward_wrapped``, set by
      ``_install_grpo_hidden_states_forward_wrapper`` in ``unsloth/models/rl.py``
      for models the compiler did not rewrite. That wrapper degrades to real
      logits when the model cannot produce hidden states, and records whether
      it did so in ``_unsloth_grpo_hidden_states_degraded`` before it returns,
      so reading the pair after a forward describes the call that just
      finished. Degradation is per call, not per model: a forward that splats
      ``**kwargs`` into a sub-module only some inputs reach can reject the
      request on one batch and honour it on the next.

    The width comparison stays as the fallback, for an ``unsloth_zoo`` old enough
    that it never writes the marker. It is decisive on its own whenever
    ``vocab_size != hidden_size``, and the signal is only allowed to overrule it
    when it is not: a model with ``vocab_size == hidden_size`` produces real
    logits that are the same width as its hidden states, which is the one case
    the shape cannot answer.
    """
    if tensor.shape[-1] != lm_head.shape[1]:
        return False  # vocab-wide: real logits, whatever any signal claims
    if lm_head.shape[0] != lm_head.shape[1]:
        return True  # hidden-wide and vocab_size != hidden_size: hidden states
    return _unsloth_grpo_hidden_states_signal(model) is not False


def _unsloth_grpo_hidden_states_signal(model):
    """``True``/``False`` if the forward honours ``UNSLOTH_RETURN_HIDDEN_STATES``.

    ``None`` when neither marker is present, i.e. there is no signal to read.
    See ``_unsloth_grpo_returns_hidden_states`` for where each marker is set.
    Walks the wrapper chain because the markers are set on whichever object the
    trainer saw, which may be the DDP module or the PEFT base model rather than
    the object handed to the logprob loop.
    """
    candidates = []
    pending = [model]
    while pending and len(candidates) < 8:
        candidate = pending.pop(0)
        if candidate is None or any(candidate is seen for seen in candidates):
            continue
        candidates.append(candidate)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                pending.append(get_base_model())
            except Exception:
                pass
        for _attr in ("module", "base_model", "model"):
            child = getattr(candidate, _attr, None)
            if child is not None and hasattr(child, "forward"):
                pending.append(child)
    for candidate in candidates:
        if getattr(candidate, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", False):
            return True
        if getattr(type(candidate), "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", False):
            return True
    if any(
        getattr(candidate, "_unsloth_grpo_hidden_states_forward_wrapped", False)
        for candidate in candidates
    ):
        # The wrapper honours the flag unless it recorded that this call could not.
        if any(
            hasattr(candidate, "_unsloth_grpo_hidden_states_degraded") for candidate in candidates
        ):
            return not any(
                getattr(candidate, "_unsloth_grpo_hidden_states_degraded", False)
                for candidate in candidates
            )
        # An unsloth/models/rl.py predating the per-call attribute set only the warn-once flag; it is
        # the best signal such a wrapper offers.
        return not any(
            getattr(candidate, "_unsloth_grpo_hidden_states_warning_issued", False)
            for candidate in candidates
        )
    return None


_GRPO_HIDDEN_STATES_WIDTH_DISPATCH = re.compile(
    r"^(?P<indent>[ \t]*)if[ \t]+"
    r"(?P<tensor>[_A-Za-z][_A-Za-z0-9]*)\.shape\[-1\][ \t]*"
    r"(?P<operator>==|!=)[ \t]*lm_head\.shape\[1\][ \t]*:$",
    flags = re.MULTILINE,
)

# Deliberately loose: any branch header deciding something off an lm_head dimension. Used only
# to count how many the strict pattern should have rewritten, so a zoo respelling SOME of them
# is rejected rather than half-patched.
_GRPO_HIDDEN_STATES_WIDTH_DISPATCH_CANDIDATE = re.compile(
    r"^[ \t]*(?:el)?if[ \t]+[^\n]*\blm_head\.shape\[[^\]\n]+\][^\n]*:[ \t]*$",
    flags = re.MULTILINE,
)


def _patch_grpo_accumulated_loss_hidden_states_dispatch(function):
    """Give zoo's gradient path the same model-aware dispatch as the no-grad path.

    ``grpo_accumulated_loss`` is embedded into the generated trainer with
    ``inspect.getsource`` below. Zoo revisions with raw-logits support choose
    between the two log-softmax helpers by comparing the forward output width
    with ``lm_head.shape[1]``. That comparison is ambiguous for a square head,
    so replace every such decision before embedding the function.

    The expression is intentionally limited to a named tensor's last dimension
    against the lm_head input dimension. If zoo changes that contract entirely,
    fail at trainer generation instead of silently restoring the wrong dispatch.

    Partial matches have to fail too. A zoo that respells only some of its
    dispatches would still give this one match, which is enough to satisfy a
    "did anything match?" check while the respelled sites keep deciding on width
    alone -- silently wrong gradients again for a square ``lm_head``. So count
    the branch headers that decide off an ``lm_head`` dimension before and after
    substituting, and require that none survive.
    """
    source = function if isinstance(function, str) else inspect.getsource(function)
    candidates = len(_GRPO_HIDDEN_STATES_WIDTH_DISPATCH_CANDIDATE.findall(source))
    replacements = 0

    def replace_width_dispatch(match):
        nonlocal replacements
        replacements += 1
        decision = (
            "_unsloth_grpo_returns_hidden_states("
            f"unwrapped_model, {match.group('tensor')}, lm_head)"
        )
        if match.group("operator") == "!=":
            decision = f"not {decision}"
        return f"{match.group('indent')}if {decision}:"

    source = _GRPO_HIDDEN_STATES_WIDTH_DISPATCH.sub(replace_width_dispatch, source)
    if replacements == 0:
        raise RuntimeError(
            "Unsloth: could not find the GRPO gradient hidden-state dispatches in "
            "this unsloth_zoo version. Please upgrade unsloth_zoo."
        )
    unpatched = _GRPO_HIDDEN_STATES_WIDTH_DISPATCH_CANDIDATE.findall(source)
    if len(unpatched) != 0:
        raise RuntimeError(
            f"Unsloth: patched only {replacements} of {candidates} GRPO gradient "
            "hidden-state dispatches in this unsloth_zoo version; "
            f"{len(unpatched)} still decide on width alone "
            f"({', '.join(line.strip() for line in unpatched)}). "
            "Please upgrade unsloth_zoo."
        )
    return source


grpo_compute_loss = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
UnslothEfficientGRPO = RL_REPLACEMENTS["UnslothEfficientGRPO"]
grpo_accumulated_loss = RL_REPLACEMENTS["grpo_accumulated_loss"]
grpo_update_SamplingParams = RL_REPLACEMENTS["grpo_update_SamplingParams"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast_kwargs))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_model_config))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_final_logit_softcapping))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_returns_hidden_states))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_hidden_states_signal))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_mm_token_id))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_fix_mm_token_type_ids))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_clear_stateful_mrope))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(UnslothEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(
    _patch_grpo_accumulated_loss_hidden_states_dispatch(grpo_accumulated_loss)
)
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_update_SamplingParams))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_get_inference_mode_context_manager))
# inspect.getsource inlines function bodies but not module imports, so constants the inlined
# grpo functions use (UNSLOTH_ENABLE_LOGGING) must be redefined in the generated cache.
RL_PRE_ITEMS["grpo_trainer"].append(
    "import os as _unsloth_os\n"
    "UNSLOTH_ENABLE_LOGGING = _unsloth_os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') in ('1', 'True', 'true')\n"
)
# Sequence-packing gates, same values as the module-top constants.
RL_PRE_ITEMS["grpo_trainer"].append(
    "UNSLOTH_GRPO_SEQ_PACKING_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_SEQ_PACKING', '1').lower() not in ('0', 'false', 'no', 'off')\n"
)
RL_PRE_ITEMS["grpo_trainer"].append(
    "try:\n"
    "    import inspect as _unsloth_inspect\n"
    "    from unsloth_zoo.rl_replacements import RL_REPLACEMENTS as _unsloth_zoo_RL\n"
    "    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = 'torch.where(_keep, new' in _unsloth_inspect.getsource(_unsloth_zoo_RL['grpo_compute_loss'])\n"
    "except Exception:\n"
    "    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = False\n"
)
# PrefixGrouper gate, same shape as the module-top constants.
RL_PRE_ITEMS["grpo_trainer"].append(
    "_pg_build_layout = _pg_enabled_fn = _pg_verify_on = _pg_tol_ok = _PG_TOL_KILL = None\n"
    "UNSLOTH_GRPO_PREFIX_GROUPER_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_PREFIX_GROUPER', '1').lower() not in ('0', 'false', 'no', 'off')\n"
    "if UNSLOTH_GRPO_PREFIX_GROUPER_ON:\n"
    "    try:\n"
    "        from unsloth.utils.prefix_grouper import build_group_layout as _pg_build_layout, prefix_grouper_enabled as _pg_enabled_fn, verify_on as _pg_verify_on, tol_ok as _pg_tol_ok, TOL_KILL as _PG_TOL_KILL\n"
    "    except Exception:\n"
    "        UNSLOTH_GRPO_PREFIX_GROUPER_ON = False\n"
)
# getsource inlines the grpo bodies but not this file's imports, so the generated cache needs
# its own guarded import or detect_logit_transforms is a NameError.
RL_PRE_ITEMS["grpo_trainer"].append(
    "try:\n"
    "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
    "except Exception:\n"
    "    detect_logit_transforms = None\n"
)


# Edit _get_per_token_logps to handle mixed precision.
def grpo_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss":
        return function

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        pixel_values, image_grid_thw = (
            inputs.get("pixel_values", None),
            inputs.get("image_grid_thw", None),
        )
        pixel_attention_mask, image_sizes = (
            inputs.get("pixel_attention_mask", None),
            inputs.get("image_sizes", None),
        )
        num_images = inputs.get("num_images", None)
        # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
        token_type_ids = inputs.get("token_type_ids", None)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        tool_mask = inputs.get("tool_mask", None)
        # Missing when evaluate() runs standalone; eval does not accumulate, so fall back to 1 rather
        # than underreport eval_loss (#2464).
        current_gradient_accumulation_steps = getattr(
            self, "current_gradient_accumulation_steps", 1
        )
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1)
        if mm_token_type_ids is not None or image_grid_thw is not None:
            mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                input_ids,
                mm_token_type_ids,
                completion_ids = completion_ids,
            )
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = (
            lambda model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size = None,
            compute_entropy = False,
            compute_efficient = False: (
                self._get_per_token_logps(
                    model, input_ids, attention_mask, logits_to_keep, compute_efficient
                )
                if hasattr(self, "_get_per_token_logps")
                else self._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    compute_efficient,
                )[0]
            )
        )  # logps

        per_token_logps = get_logps_func(
            model, input_ids, attention_mask, logits_to_keep, compute_efficient = True
        )
        # KL divergence between model and reference: _prepare_inputs no longer returns reference log
        # probs. See trl grpo_trainer.py#L1328.
        ref_logps = inputs.get("ref_per_token_logps", None)
        # x - x.detach() preserves gradients from x.
        advantages = inputs["advantages"]
        old_logps = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        model_config = _unsloth_get_model_config(model)
        # The old and reference logps come from _get_per_token_logps_and_entropies and the gradient
        # logps from here, so both must read the transforms alike or the importance ratio compares two
        # different policies.
        if detect_logit_transforms is not None:
            # model_config, not model: see _get_per_token_logps_and_entropies.
            _transforms = detect_logit_transforms(model_config)
            logit_softcapping = _transforms["logit_softcapping"]
            logit_scale_multiply = _transforms["logit_scale_multiply"]
            logit_scale_divide = _transforms["logit_scale_divide"]
        else:
            logit_softcapping = _unsloth_get_final_logit_softcapping(model)  # Gemma
            logit_scale_multiply = getattr(model_config, "logit_scale", 0)  # Cohere
            if logit_scale_multiply is None:
                logit_scale_multiply = 0
            logit_scale_divide = getattr(model_config, "logits_scaling", 0)  # Granite
            if logit_scale_divide is None:
                logit_scale_divide = 0

        max_left_pad = inputs.get("max_left_pad", 0)
        if per_token_logps is not None:
            loss_mask = completion_mask
            if tool_mask is not None:
                if tool_mask.shape != completion_mask.shape:
                    raise ValueError(
                        "tool_mask/env_mask must have the same shape as completion_mask"
                    )
                loss_mask = completion_mask * tool_mask.to(
                    device = completion_mask.device,
                    dtype = completion_mask.dtype,
                )
            (
                loss,
                completion_length,
                mean_kl,
                delta,
                flat_is_ratio,
                coef_1,
                completion_mask,
            ) = grpo_compute_loss_slow(
                ref_logps,
                per_token_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                loss_mask,
                self.beta,
                advantages,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                loss_type = self.args.loss_type,
                importance_sampling_level = self.importance_sampling_level,
                epsilon_low = self.epsilon_low,
                epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = self.args.delta,
                temperature = self.args.temperature,
                max_left_pad = max_left_pad,
                logit_softcapping = logit_softcapping,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
                num_items_in_batch = num_items_in_batch,
                current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                num_processes = num_processes,
            )
        else:

            def _unsloth_requires_multi_image_zoo(value):
                if value is None:
                    return False
                if isinstance(value, torch.Tensor):
                    counts = value.detach().cpu().reshape(-1).tolist()
                else:
                    counts = list(value)
                return any(int(n) != 1 for n in counts)

            if _unsloth_requires_multi_image_zoo(num_images) and not getattr(
                self, "_unsloth_grpo_zoo_checked", False
            ):
                _supports_num_images = (
                    "num_images" in inspect.signature(grpo_accumulated_loss).parameters
                )
                if not _supports_num_images:
                    try:
                        _zoo_src = inspect.getsource(grpo_accumulated_loss)
                    except (TypeError, OSError):
                        _zoo_src = ""
                    _supports_num_images = "num_images" in _zoo_src
                if not _supports_num_images:
                    raise RuntimeError(
                        "Multi-image GRPO requires an unsloth_zoo build whose "
                        "grpo_accumulated_loss handles num_images. Please upgrade "
                        "unsloth_zoo (see https://github.com/unslothai/unsloth-zoo/pull/613)."
                    )
                self._unsloth_grpo_zoo_checked = True
            if tool_mask is not None and not getattr(
                self, "_unsloth_grpo_tool_mask_zoo_checked", False
            ):
                _supports_tool_mask = (
                    "tool_mask" in inspect.signature(grpo_accumulated_loss).parameters
                )
                if not _supports_tool_mask:
                    try:
                        _zoo_src = inspect.getsource(grpo_accumulated_loss)
                    except (TypeError, OSError):
                        _zoo_src = ""
                    _supports_tool_mask = "tool_mask" in _zoo_src
                if not _supports_tool_mask:
                    raise RuntimeError(
                        "env_mask/tool_mask GRPO requires an unsloth_zoo build whose "
                        "grpo_accumulated_loss handles tool_mask. Please upgrade "
                        "unsloth_zoo."
                    )
                self._unsloth_grpo_tool_mask_zoo_checked = True
            _grpo_accumulated_loss_kwargs = {}
            if tool_mask is not None:
                _grpo_accumulated_loss_kwargs["tool_mask"] = tool_mask
            if hasattr(self.args, "loss_type"):
                (
                    loss,
                    completion_length,
                    mean_kl,
                    delta,
                    flat_is_ratio,
                    coef_1,
                    completion_mask,
                ) = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    pixel_attention_mask = pixel_attention_mask,
                    image_sizes = image_sizes,
                    num_images = num_images,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    importance_sampling_level = self.importance_sampling_level,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = self.args.delta,
                    temperature = self.args.temperature,
                    max_left_pad = max_left_pad,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                    num_items_in_batch = num_items_in_batch,
                    current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                    num_processes = num_processes,
                    sampling_per_token_logps = sampling_per_token_logps,
                    token_type_ids = token_type_ids,
                    mm_token_type_ids = mm_token_type_ids,
                    **_grpo_accumulated_loss_kwargs,
                )
            else:
                # For backwards compatibility with trl 0.15.2 and maybe 0.17.
                loss, completion_length, mean_kl, coef_1, completion_mask = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    pixel_values = pixel_values,
                    image_grid_thw = image_grid_thw,
                    pixel_attention_mask = pixel_attention_mask,
                    image_sizes = image_sizes,
                    num_images = num_images,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids,
                    mm_token_type_ids = mm_token_type_ids,
                    **_grpo_accumulated_loss_kwargs,
                )
        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        if (
            self.use_vllm
            and delta is not None
            and getattr(self, "vllm_importance_sampling_correction", False)
        ):
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                self.accelerator.gather(min_importance_sampling_ratio)
                .nan_to_num(nan = float("inf"))
                .min()
                .item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                self.accelerator.gather(max_importance_sampling_ratio)
                .nan_to_num(nan = float("-inf"))
                .max()
                .item()
            )

        completion_token_count = completion_mask.sum().clamp(min = 1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                gathered_clip_ratio.nanmean().item()
            )
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(
                gathered_cispo_clip_ratio.nanmean().item()
            )

        return loss

    function = inspect.getsource(compute_loss)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_compute_loss)


# KTO shape mismatch: the Unsloth forward truncates input_ids while labels are untouched, and
# TRL 0.27.2+ _process_tokens truncates only completions, so an over-length prompt makes the
# model emit shorter logits than the labels expect.
def kto_trainer_get_batch_logps(function_name, function):
    if function_name != "get_batch_logps":
        return function
    # The raise sits inside an if inside the method, so its exact indentation must be preserved.
    old = 'raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")'
    new = (
        "# Unsloth: auto-truncate to shorter sequence length (model may have truncated input_ids)\n"
        "            _min_len = min(logits.shape[1], labels.shape[1])\n"
        "            logits = logits[:, :_min_len, :]\n"
        "            labels = labels[:, :_min_len]"
    )
    function = function.replace(old, new)
    return function


RL_FUNCTIONS["kto_trainer"].append(kto_trainer_get_batch_logps)


# TRL 1.x dropped KTOTrainer.get_batch_logps and moved the math into _compute_logps /
# compute_ref_log_probs / _compute_kl_logps, which call selective_log_softmax on
# completion-only tokens. Same truncation hazard, so clamp logits/ids/mask to the shorter
# length (a no-op when equal).
_KTO_COMPLETION_RE = re.compile(
    r"(?P<ws>[ \t]*)shift_logits = completion_logits\[:, :-1, :\]\.contiguous\(\)\n"
    r"(?P=ws)per_token_logps = selective_log_softmax\(\s*shift_logits,\s*"
    r"(?P<var>\w+)\[[\"']completion_input_ids[\"']\]\[:, 1:\]\.contiguous\(\)\s*\)\n"
    r"(?P=ws)per_token_logps\[(?P=var)\[[\"']completion_mask[\"']\]\[:, 1:\] == 0\] = 0\.0"
)
_KTO_KL_RE = re.compile(
    r"(?P<ws>[ \t]*)shift_KL_logits = KL_logits\[:, :-1, :\]\.contiguous\(\)\n"
    r"(?P=ws)KL_per_token_logps = selective_log_softmax\(\s*shift_KL_logits,\s*"
    r"(?P<var>\w+)\[[\"']KL_completion_input_ids[\"']\]\[:, 1:\]\.contiguous\(\)\s*\)\n"
    r"(?P=ws)KL_per_token_logps\[(?P=var)\[[\"']KL_completion_mask[\"']\]\[:, 1:\] == 0\] = 0\.0"
)


def _kto_completion_repl(m):
    ws, var = m.group("ws"), m.group("var")
    return (
        f"{ws}shift_logits = completion_logits[:, :-1, :].contiguous()\n"
        f"{ws}# Unsloth: clamp logits/ids/mask to shorter seq len (model may truncate input_ids)\n"
        f'{ws}_uns_ids = {var}["completion_input_ids"][:, 1:].contiguous()\n'
        f"{ws}_uns_n = min(shift_logits.shape[1], _uns_ids.shape[1])\n"
        f"{ws}per_token_logps = selective_log_softmax(shift_logits[:, :_uns_n], _uns_ids[:, :_uns_n])\n"
        f'{ws}per_token_logps[{var}["completion_mask"][:, 1:][:, :_uns_n] == 0] = 0.0'
    )


def _kto_kl_repl(m):
    ws, var = m.group("ws"), m.group("var")
    return (
        f"{ws}shift_KL_logits = KL_logits[:, :-1, :].contiguous()\n"
        f"{ws}# Unsloth: clamp logits/ids/mask to shorter seq len (model may truncate input_ids)\n"
        f'{ws}_uns_kl_ids = {var}["KL_completion_input_ids"][:, 1:].contiguous()\n'
        f"{ws}_uns_kl_n = min(shift_KL_logits.shape[1], _uns_kl_ids.shape[1])\n"
        f"{ws}KL_per_token_logps = selective_log_softmax(shift_KL_logits[:, :_uns_kl_n], _uns_kl_ids[:, :_uns_kl_n])\n"
        f'{ws}KL_per_token_logps[{var}["KL_completion_mask"][:, 1:][:, :_uns_kl_n] == 0] = 0.0'
    )


def kto_trainer_align_completion_logps(function_name, function):
    if function_name not in (
        "_compute_logps",
        "compute_ref_log_probs",
        "_compute_kl_logps",
    ):
        return function
    function = _KTO_COMPLETION_RE.sub(_kto_completion_repl, function)
    function = _KTO_KL_RE.sub(_kto_kl_repl, function)
    return function


RL_FUNCTIONS["kto_trainer"].append(kto_trainer_align_completion_logps)


# TRL warns if batch size is not a multiple of num_generations; see trl grpo_trainer.py#L356.
def grpo_trainer_fix_batch_size(RLTrainer_source, RLConfig_source):
    if "divisible by the number of generations" not in RLTrainer_source:
        # In later trl versions this no longer exists.
        return ""
    if "num_generations" not in RLConfig_source:
        return ""

    check_batch_size = (
        "div = per_device_train_batch_size // num_generations\n"
        "if div * num_generations != per_device_train_batch_size:\n"
        "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"
        "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"
        "    per_device_train_batch_size = num_generations\n"
    )
    return check_batch_size


RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_trainer_fix_batch_size)


def grpo_trainer_metrics(RLTrainer_source, RLConfig_source):
    if "reward_funcs" not in RLTrainer_source:
        return ""

    # New TRL has /mean and /std.
    use_mean = "rewards/{reward_func_name}/mean" in RLTrainer_source
    use_std = "rewards/{reward_func_name}/std" in RLTrainer_source
    if not use_mean:
        use_normal = "rewards/{reward_func_name}" in RLTrainer_source
    else:
        use_normal = False

    log_metrics = (
        "if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]\n"
        "else: _reward_funcs = reward_funcs\n"
        "for reward_func in _reward_funcs:\n"
        "    try:\n"
        "        reward_func_name = reward_func.__name__\n"
        f"        if {use_mean}:\n"
        "            other_metrics.append(f'rewards/{reward_func_name}/mean')\n"
        f"        if {use_std}:\n"
        "            other_metrics.append(f'rewards/{reward_func_name}/std')\n"
        f"        if {use_normal}:\n"
        "            other_metrics.append(f'rewards/{reward_func_name}')\n"
        "    except: pass\n"
    )
    return log_metrics


RL_METRICS_CHANGES["grpo_trainer"].append(grpo_trainer_metrics)


def openenv_vllm_reload_weights():
    # Patch trl's openenv generate_rollout_completions to guard reload_weights when sharing weights
    # with vLLM, and to call wake_up() untagged: TRL's wake_up(tags=["kv_cache"]) leaves
    # is_sleeping=True at the executor, so Unsloth's generate wakes again and double create_and_maps
    # mapped handles. Unsloth's CuMemAllocator.wake_up skips weights anyway.
    if importlib.util.find_spec("trl") is None:
        return
    if Version(importlib_version("trl")) < Version("0.26.0"):
        return

    try:
        import trl.experimental.openenv.utils as openenv_utils
        import trl.experimental.openenv as openenv
    except (ImportError, NameError, Exception) as e:
        logger.info(f"Unsloth: Failed to import trl openenv: {e}")
        logger.info(
            "Unsloth: trl.experimental.openenv not available — skipping RL openenv patches."
        )
        return

    # trl 0.28 changed the function name again.
    patch_target_name = "_generate_rollout_completions_colocate"
    if hasattr(openenv_utils, patch_target_name):
        patch_target = getattr(openenv_utils, patch_target_name)
    else:
        # Older TRL versions may keep sleep/wake logic in the public dispatcher.
        patch_target_name = "generate_rollout_completions"
        patch_target = getattr(openenv_utils, patch_target_name)

    # TRL 0.29.1+ ships some openenv helpers as bytecode with no source, so inspect.getsource
    # raises OSError; skip the rewrite rather than crash. The unmodified path keeps the duplicate
    # reload_weights and the tagged wake_up, so openenv GRPO users may see redundant reloads.
    try:
        src = inspect.getsource(patch_target)
    except OSError as e:
        logger.warning(
            f"Unsloth: Could not retrieve source for trl openenv "
            f"{patch_target_name} ({e}); skipping rewrite. The unmodified "
            f"TRL openenv path will run, so the duplicate reload_weights "
            f"strip and the wake_up tag rewrite are NOT applied. Open an "
            f"issue if you see redundant reload_weights or partial wake_up "
            f"on openenv GRPO with this TRL build."
        )
        return
    src = textwrap.dedent(src)
    original_src = src

    reload_weights_pattern = re.compile(
        r"^(?P<indent>[ \t]*)(?P<obj>\S+)\.collective_rpc\(\s*(['\"])reload_weights\3\s*\)\s*$",
        re.MULTILINE,
    )

    def replace_reload_weights(match):
        indent = match.group("indent")
        obj = match.group("obj")
        return (
            f"{indent}if not getattr({obj}, 'shared_weights', False):\n"
            f'{indent}    {obj}.collective_rpc("reload_weights")\n'
        )

    src = reload_weights_pattern.sub(replace_reload_weights, src)

    # wake_up() with no tags wakes everything and sets is_sleeping=False, preventing a double
    # wake_up; Unsloth's allocator skips weights anyway.
    src = re.sub(r"\.wake_up\(tags=\[.*?\]\)", ".wake_up()", src)

    if original_src == src:
        logger.warning("Unsloth: Warning - regex did not match, patch may have failed")
        return

    # Execute and explicitly assign to the module.
    local_ns = {}
    exec(compile(src, "<unsloth>", "exec"), openenv_utils.__dict__, local_ns)
    patched_func = local_ns[patch_target_name]

    # Patch the target function in utils; if the dispatcher was patched, also update the parent module alias.
    setattr(openenv_utils, patch_target_name, patched_func)
    if patch_target_name == "generate_rollout_completions":
        openenv.generate_rollout_completions = patched_func
    logger.info(f"Unsloth: Patched trl openenv {patch_target_name}")


RL_ADDITIONAL_FUNCTIONS["openenv"].append(openenv_vllm_reload_weights)


def vllm_generation_init_patch():
    # trl moved vllm code to trl/generation/vllm_generation.py (commit 0eb66d8, 0.28.0+), which
    # must be patched so it does not build a second vLLM instance when fast_inference has one.

    if importlib.util.find_spec("trl") is None:
        return
    if Version(importlib_version("trl")) < Version("0.28.0"):
        return

    try:
        import trl.generation.vllm_generation as vllm_generation
    except (ImportError, NameError, Exception) as e:
        logger.info(f"Unsloth: Failed to import trl.generation.vllm_generation: {e}")
        return

    def patch_vllm_generation_method(method_name, transform, marker, filename_suffix):
        method = getattr(vllm_generation.VLLMGeneration, method_name, None)
        if method is None:
            logger.info(f"Unsloth: Could not find VLLMGeneration.{method_name}")
            return False

        try:
            src = inspect.getsource(method)
        except Exception as e:
            logger.info(f"Unsloth: Could not get source of VLLMGeneration.{method_name}: {e}")
            return False

        src = textwrap.dedent(src)
        if marker in src:
            return True

        src = transform(src)
        filename = f"<unsloth_trl_vllm_generation_{filename_suffix}_patch>"
        source_lines = [line + "\n" for line in src.splitlines()]
        linecache.cache[filename] = (
            len(src),
            None,
            source_lines,
            filename,
        )

        local_ns = {}
        exec(compile(src, filename, "exec"), vllm_generation.__dict__, local_ns)
        setattr(vllm_generation.VLLMGeneration, method_name, local_ns[method_name])
        return True

    def patch_init_vllm(src):
        pattern = re.compile(
            r"(?P<llm_block>^(?P<indent>[ \t]*)self\.llm\s*=\s*LLM\s*\(\n(?:.*\n)*?^(?P=indent)\))",
            re.MULTILINE,
        )

        def replace_llm_block(match):
            indent = match.group("indent")
            llm_block = textwrap.dedent(match.group("llm_block"))
            return (
                f"{indent}if hasattr(model, 'vllm_engine'):\n"
                f"{indent}    # Unsloth already inits vLLM in fast inference mode. Do not redo :)\n"
                f"{indent}    self.llm = model.vllm_engine\n"
                f"{indent}    self.unsloth_fast_inference_lora = getattr(self.llm, 'shared_weights', False)\n"
                f"{indent}    if getattr(self.llm, 'shared_weights', False) and hasattr(model, 'load_lora'):\n"
                f"{indent}        self._unsloth_load_lora = model.load_lora\n"
                f"{indent}else:\n" + textwrap.indent(llm_block, indent + "    ")
            )

        patched_src, num_replacements = pattern.subn(replace_llm_block, src, count = 1)
        if num_replacements == 0:
            raise RuntimeError(
                "Unsloth: Warning - regex did not match, VLLMGeneration._init_vllm patch may have failed"
            )
        return patched_src

    # Newer versions have sync_weights or reload rpc calls; earlier ones are stripped in the patched grpo_trainer above.
    def patch_sync_weights(src):
        pattern = re.compile(
            r"^(?P<def_line>def sync_weights\(self\):\n)(?P<body>(?:.*\n)*)",
            re.MULTILINE,
        )

        def replace_sync_weights(match):
            body = match.group("body")
            # Chain getattr so server mode, where self.llm is unset, does not raise AttributeError before
            # the default kicks in.
            guard = (
                "    if getattr(getattr(self, 'llm', None), 'shared_weights', False) or "
                "getattr(self, 'unsloth_fast_inference_lora', False):\n"
                "        # Unsloth fast inference LoRA shares weights with vLLM already,\n"
                "        # so there is nothing to push. But TRL >= 1.x only wakes the\n"
                "        # engine from sleep mode inside this method, and generate()\n"
                "        # delegates that wake-up to it, so still do it here or the next\n"
                "        # generate runs against a sleeping engine. Unsloth's allocator\n"
                "        # skips weights, so wake everything rather than a tag subset.\n"
                "        if getattr(self, '_llm_weights_sleeping', False) and "
                "getattr(self, 'llm', None) is not None:\n"
                "            self.llm.wake_up()\n"
                "            self._llm_weights_sleeping = False\n"
                "        return\n\n"
            )
            return match.group("def_line") + guard + body

        patched_src, num_replacements = pattern.subn(replace_sync_weights, src, count = 1)
        if num_replacements == 0:
            raise RuntimeError(
                "Unsloth: Warning - regex did not match, VLLMGeneration.sync_weights patch may have failed"
            )
        return patched_src

    # `generate` is deliberately NOT source-patched. Its two anchors drifted: TRL >= 1.x deleted
    # the collective_rpc("reload_weights") call for self.sync_weights(), so the anchor matched 0
    # times, raised, and took the lora injection with it -- GRPO rollouts came from the BASE model.
    # The generate regex is also not paren-balanced and mis-edits multi-line calls.
    # Intercept on the vLLM engine instead of TRL's method body: self.llm is the vLLM LLM object in
    # colocate mode in every release with VLLMGeneration, and generate / chat / collective_rpc are
    # public stable APIs, checked across vLLM 0.11.0-0.27.1. The override is scoped to one
    # VLLMGeneration.generate call and undone in a finally.
    _UNSLOTH_GENERATE_WRAPPED = "_unsloth_vllm_generation_lora_wrapped"

    # Mirror the per-device naming in rl.py so two ranks on one node do not race on the same adapter directory.
    lora_name = "vllm_gen_lora"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        lora_name += "_" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "")

    def install_generate_wrapper():
        original_generate = getattr(vllm_generation.VLLMGeneration, "generate", None)
        if original_generate is None:
            logger.info("Unsloth: Could not find VLLMGeneration.generate")
            return False
        if getattr(original_generate, _UNSLOTH_GENERATE_WRAPPED, False):
            return True

        def generate(self, *args, **kwargs):
            llm = getattr(self, "llm", None)
            sharing = getattr(llm, "shared_weights", False) or getattr(
                self, "unsloth_fast_inference_lora", False
            )
            if llm is None or not sharing:
                # Server mode, or a vLLM engine TRL created itself: keep upstream behaviour.
                return original_generate(self, *args, **kwargs)

            load_lora = getattr(self, "_unsloth_load_lora", None)
            saved = []

            def override(name, make_replacement):
                bound = getattr(llm, name, None)
                if bound is None:
                    return
                had_own = name in getattr(llm, "__dict__", {})
                try:
                    setattr(llm, name, make_replacement(bound))
                except (AttributeError, TypeError):
                    return
                saved.append((name, had_own, bound))

            def caller_already_bound_lora(bound, args, kwargs):
                """Has the caller's own argument list already filled `lora_request`?

                A keyword `lora_request` that is not None is the caller's choice, so leave
                it. A keyword `lora_request = None` is not: on a shared-weights engine that
                means base-model rollouts, which is the bug this whole wrapper exists to
                fix, so it gets overwritten.

                The positional case is the one that has to be checked rather than assumed.
                `lora_request` is keyword-only on `LLM.generate` in every vLLM release from
                0.11.0 to 0.27.1, but on `LLM.chat` it is an ordinary positional-or-keyword
                parameter, and its index there has already moved once (`tokenization_kwargs`
                landed in 0.18.0). A caller that passed it positionally has supplied it, and
                adding a keyword on top would be `TypeError: got multiple values`, not a
                missing adapter. Bind the real signature instead of counting arguments so a
                future reshuffle cannot reintroduce that.
                """
                if kwargs.get("lora_request", None) is not None:
                    return True
                try:
                    positional = inspect.signature(bound).bind_partial(*args).arguments
                except (TypeError, ValueError):
                    # Unintrospectable callable (C extension, odd wrapper): the keyword check above is all we have,
                    # and injecting is the safe default.
                    return False
                return "lora_request" in positional

            def wrap_generation_call(bound):
                def unsloth_generation_call(*args, **kwargs):
                    # vLLM needs the adapter handed to it explicitly: the shared engine holds the BASE weights, and
                    # sync_weights is a no-op when sharing.
                    if load_lora is not None and not caller_already_bound_lora(bound, args, kwargs):
                        kwargs["lora_request"] = load_lora(lora_name, load_tensors = True)
                    return bound(*args, **kwargs)

                return unsloth_generation_call

            def wrap_collective_rpc(bound):
                def unsloth_collective_rpc(method, *args, **kwargs):
                    # The engine already shares the live training weights, so reload_weights would pull the
                    # ORIGINAL checkpoint back off disk.
                    if method == "reload_weights":
                        return None
                    return bound(method, *args, **kwargs)

                return unsloth_collective_rpc

            override("generate", wrap_generation_call)
            override("chat", wrap_generation_call)
            override("collective_rpc", wrap_collective_rpc)
            try:
                return original_generate(self, *args, **kwargs)
            finally:
                for name, had_own, bound in reversed(saved):
                    try:
                        if had_own:
                            setattr(llm, name, bound)
                        else:
                            delattr(llm, name)
                    except AttributeError:
                        pass

        generate.__name__ = getattr(original_generate, "__name__", "generate")
        generate.__qualname__ = getattr(
            original_generate, "__qualname__", "VLLMGeneration.generate"
        )
        generate.__doc__ = getattr(original_generate, "__doc__", None)
        # inspect.getsource / inspect.signature unwrap this, so drift detectors and other source-reading
        # patches still see TRL's own generate.
        generate.__wrapped__ = original_generate
        setattr(generate, _UNSLOTH_GENERATE_WRAPPED, True)
        vllm_generation.VLLMGeneration.generate = generate
        return True

    # Snapshot before patching: a HALF-patched VLLMGeneration is worse than none, since _init_vllm
    # plus sync_weights without the generate-side adapter injection means no weight sync AND no
    # LoRA, i.e. base-model rollouts with no error. If one of the three fails, restore all three.
    method_names = ("_init_vllm", "sync_weights", "generate")
    originals = {name: getattr(vllm_generation.VLLMGeneration, name, None) for name in method_names}
    try:
        init_patched = patch_vllm_generation_method(
            "_init_vllm",
            patch_init_vllm,
            "self.unsloth_fast_inference_lora = getattr(self.llm, 'shared_weights', False)",
            "init_vllm",
        )
        sync_patched = patch_vllm_generation_method(
            "sync_weights",
            patch_sync_weights,
            "if getattr(getattr(self, 'llm', None), 'shared_weights', False) or getattr(self, 'unsloth_fast_inference_lora', False):",
            "sync_weights",
        )
        generate_patched = install_generate_wrapper()
    except RuntimeError as e:
        for name, original in originals.items():
            if original is not None:
                setattr(vllm_generation.VLLMGeneration, name, original)
        logger.warning(str(e))
        return

    if init_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration._init_vllm")
    if sync_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration.sync_weights")
    if generate_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration.generate")


RL_ADDITIONAL_FUNCTIONS["vllm_generation"].append(vllm_generation_init_patch)
