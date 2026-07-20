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
# Packing needs zoo#840's masked-column guard in grpo_compute_loss (installed zoo is fixed per-process).
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


# Check untrained tokens
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


# Fix top_k for GRPO vLLM.
# https://github.com/huggingface/trl/pull/4695 with this change trl added top_k in GRPOConfig and defaults to 0
# We don't want that since vllm's all include top_k is -1 and 0 returns an error on SamplingParams creation.
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
            # Use fast version!
            function = inspect.getsource(fast_sft_prepare_dataset)
            # why: install the wrapped-packing setup (and the `_inspect` import the
            # truncation / pack_dataset rewrites below depend on) at the function
            # signature, a structural anchor that always exists, rather than the
            # unsloth_zoo license-comment line. That header is only lower-bounded, so a
            # newer Zoo may move or drop it; anchoring there let the setup silently
            # no-op while the references still landed, NameError-ing every SFT dataset
            # preparation. Fail loudly if even the signature cannot be located.
            _wrapped_packing_setup = (
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
            function, _n_setup = re.subn(
                r"(def sft_prepare_dataset\s*\(.*?\)\s*(?:->[^:\n]*)?:[ \t]*\n)",
                lambda match: match.group(1) + _wrapped_packing_setup,
                function,
                count = 1,
                flags = re.DOTALL,
            )
            if _n_setup != 1:
                raise RuntimeError(
                    "Unsloth: failed to install wrapped-packing support into "
                    "sft_prepare_dataset (signature not found); please file a bug report."
                )
            function = function.replace(
                "truncation = do_truncation,",
                "truncation = do_truncation and not _unsloth_wrapped_packing,",
            )
            function = function.replace(
                "if do_truncation and max_seq_length > 0:",
                "if do_truncation and not _unsloth_wrapped_packing and max_seq_length > 0:",
            )
            function = function.replace(
                """dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            getattr(args, "packing_strategy", "bfd"),
            map_kwargs,
        )""",
                """_pack_kwargs = {"map_kwargs": map_kwargs}
        if "strategy" in _inspect.signature(pack_dataset).parameters:
            _pack_kwargs["strategy"] = getattr(args, "packing_strategy", "bfd")
        dataset = pack_dataset(
            dataset.select_columns(used_column_names),
            max_seq_length,
            **_pack_kwargs,
        )""",
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

    # .*? matches first match. .+? matches final match.
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


# Ignore mean_token_accuracy since it needs logits
# We override it directly with our version
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


# Route ORPO/CPO row tokenization through the underlying text tokenizer when the
# processing class is a multimodal processor; CPO reuses this code (#4952).
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


# Resolve `processing_class.pad_token_id` through the underlying tokenizer when
# a multimodal processor is supplied (processors lack `pad_token_id`). Without
# this, ORPO/CPOTrainer.__init__ raises AttributeError on
# `DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id, ...)`
# and on `self.padding_value = ... else processing_class.pad_token_id`.
_PAD_FALLBACK = (
    "(getattr(processing_class, 'pad_token_id', None) "
    "if getattr(processing_class, 'pad_token_id', None) is not None "
    "else getattr(getattr(processing_class, 'tokenizer', None), 'pad_token_id', None))"
)


def orpo_trainer_processor_pad_token(function_name, function):
    if function_name != "__init__":
        return function
    # Multimodal processors (e.g. Gemma3/Gemma4 Processor) expose pad_token /
    # eos_token on `.tokenizer`, not on the processor itself. TRL 1.x CPO/ORPO
    # __init__ defaults `processing_class.pad_token` from `.eos_token` before
    # tokenizing, which AttributeErrors on such a processor. Route the default
    # through the inner tokenizer. Older TRL lacks this block, so the sub is a
    # no-op there and only the pad_token_id fallback below applies.
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


# Fix bare pop("push_to_hub_token") in compiled SFT/IterativeSFT trainer __init__
# On transformers 5.0+, to_dict() no longer includes push_to_hub_token, so bare pop KeyErrors
def sft_trainer_push_to_hub_token(function_name, function):
    if function_name != "__init__":
        return function
    return function.replace(
        'dict_args.pop("push_to_hub_token")', 'dict_args.pop("push_to_hub_token", None)'
    )


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_push_to_hub_token)


# Autocast precision for GRPO
def grpo_trainer__prepare_inputs(function_name, function):
    if function_name != "_prepare_inputs":
        return function

    # Add mixed precision training
    function = function.replace(
        "with torch.inference_mode():",
        "with torch.inference_mode(), "
        "torch.amp.autocast(device_type = 'cuda', "
        "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "
        "if not torch.is_autocast_enabled('cuda') else nullcontext())"
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):",
    )
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Guard reload_weights and sync_weights - skip when fast inference LoRA shares weights with vLLM
# https://github.com/huggingface/trl/commit/7856d3b1f6518601732f489883b341bb6dd36434#diff-964e6fd373aa93037604064cb2b822d7f8e2735e33f791065acf2c4c3552d393R1168-R1169
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

    # TRL 0.24.0-0.25.1 truncation regression fix
    #
    # TRL 0.22.2-0.23.1 used smart truncation via truncate_with_protected_tokens():
    #   - Tokenizes first without truncation
    #   - Then truncates keeping the RIGHTMOST tokens (preserves assistant turn)
    #   - Protects special tokens (image_token, vision_start/end) from removal
    #
    # TRL 0.24.0-0.25.1 removed this and passed kwargs directly to the tokenizer:
    #   max_length=self.max_prompt_length, truncation=True, add_special_tokens=False
    # This causes issues because tokenizer truncation doesn't protect special tokens
    # and may not preserve the end of the prompt properly.
    #
    # TRL 0.26.2+ removed these kwargs entirely (no tokenizer-level truncation).
    #
    # Fix: Remove these kwargs so TRL 0.24.0-0.25.1 behaves like 0.26.2+ (no truncation).
    # This is a no-op for versions that don't have these kwargs (0.22.2-0.23.1, 0.26.2+).
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


# Fix incorrect special tokens handling and truncation in older TRL versions
def grpo_trainer__generate_and_score_completions(function_name, function):
    if function_name != "_generate_and_score_completions":
        return function

    # TRL 0.19.0 did skip_special_tokens = True which should be False
    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    # Left pad prompt before calculation old and ref hidden states
    line_to_replace = 'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size'

    # The new multi-line string that will replace the line above
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
    # Use re.sub() to perform the replacement
    function, num_replacements = pattern_to_find.subn(replacement_text, function)

    pattern_to_find = re.compile(
        r"(^\s*)all_logprobs = \["  # Capture indentation (group 1)
        r".*?"  # Match everything inside non-greedily
        r"for output in outputs\.outputs\s*"
        r"\]",
        re.DOTALL | re.MULTILINE,
    )

    # sanitize_logprob is injected as a module-level function via RLTrainer_replacement
    # template in rl.py (from RL_REPLACEMENTS), so just reference it directly here.
    replacement_text = (
        r"\1all_logprobs = [\n"
        r"\1    [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]\n"
        r"\1    for outputs in all_outputs\n"
        r"\1    for output in outputs.outputs\n"
        r"\1]"
    )

    function, num_replacements = pattern_to_find.subn(replacement_text, function)

    # Always between max_prompt_length and use_vllm
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

    # Important note: we disable TRL's importance sampling logic
    # It is disabled because the LLM path moves left padding to the right.
    # We must adjust the vLLM sampling_logprob tensor in Unsloth to account for this.
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

    # TRL 0.24.0+ extracts prompts = [x["prompt"] for x in inputs], losing metadata
    # like reasoning_effort. Inject code to store per-sample chat_template_kwargs on self.
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
    # Insert after: prompts = [x["prompt"] for x in inputs]
    _target_line = 'prompts = [x["prompt"] for x in inputs]'
    if _target_line in function:
        function = function.replace(
            _target_line,
            _target_line + _metadata_extraction,
        )

    # This path is for TRL 0.24.0 images is a variable exclusive to this version
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
        # We replace the call using 'completions' with one using 'completions_text'
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
        # Sleep functionality has been added to trl in v0.23.0. We do not want to redo this.
        # https://github.com/huggingface/trl/commit/edbe8234bc7e528f72ac76607de9d3e4753e2709

        pattern = re.compile(r".*self\.llm\.generate\(.*\).*", re.MULTILINE)
        matches = list(pattern.finditer(function))
        patched = function

        # Generally there's only one match. But this is just to make sure we don't miss any.
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


# Remove _move_model_to_vllm
def grpo_trainer__move_model_to_vllm(function_name, function):
    if function_name != "_move_model_to_vllm":
        return function

    def _move_model_to_vllm(self, *args, **kwargs):
        return None

    function = inspect.getsource(_move_model_to_vllm)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision
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
        # Otherwise, calculate normally:
        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = (
                torch.float16
                if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                else torch.bfloat16
            )
            if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
                self._autocast_dtype = torch.float16

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(device_type = DEVICE_TYPE, dtype = self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                logits_to_keep = logits_to_keep + 1,
            ).logits
            # logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            return logits
            # input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            # logits = logits[:, -logits_to_keep:]
            # return logits
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            # logits = logits / self.temperature
            # logps = selective_log_softmax(logits, input_ids)

            # row_indices, col_indices = torch.where(logps < -20)

            # # Method 1: Check if tensors have elements
            # if len(row_indices) > 0 and len(col_indices) > 0:
            #     breakpoint()  # Breakpoint triggered here
            #     print("Found high values!")
            # return  logps #  compute logprobs for the input tokens

    function = inspect.getsource(_get_per_token_logps)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)


def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
    if function_name != "_get_per_token_logps_and_entropies":
        return function

    # Just copy over from _get_per_token_logps replacement function above. For now this returns None anyway
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
        # All Unsloth code here in this function is licensed under AGPL3
        # if True: # os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
        #     return None, None  # logps, entropies Unsloth efficient GRPO
        if compute_efficient:
            return None, None
        else:
            if not hasattr(self, "_autocast_dtype"):
                self._autocast_dtype = (
                    torch.float16
                    if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                    else torch.bfloat16
                )
                if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
                    self._autocast_dtype = torch.float16

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
            # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models
            token_type_ids = kwargs.get("token_type_ids", None)
            mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
            if mm_token_type_ids is not None or image_grid_thw is not None:
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class, input_ids, mm_token_type_ids
                )

            unwrapped_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper = False)

            lm_head = self.model.get_output_embeddings().weight

            dtype_bytes = 16 if self._autocast_dtype in [torch.float16, torch.bfloat16] else 32
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
                # why: cum_rows is indexed via .item() inside the per-chunk loop;
                # keeping it on CPU avoids per-iteration GPU->CPU sync.
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
            # TRL 0.23.0 batching logic
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

            # ---- Sequence packing (default-on; disable with UNSLOTH_GRPO_SEQ_PACKING=0) ----
            # One varlen [1, sum L] forward replaces the padded [B, Lmax] loop (also fixes the
            # left-pad RoPE error). Self-verified against the per-row forward, re-checked as T
            # grows; falls back if a backend ignores packed_seq_lengths.
            logprobs = None

            # ---- PrefixGrouper (GRPO shared-prompt dedup; default ON, exact + self-verified) ----
            # G completions per prompt share the prefix; the packed path forwards it G times,
            # PrefixGrouper stores it once (FlexAttention shared-prefix mask), cutting the trunk
            # forward from G*(P+R) to P+G*R tokens. Gated by UNSLOTH_GRPO_PREFIX_GROUPER (needs
            # seq-packing), tok_r auto-gate, and first-use self-verify vs the packed path
            # (mismatch => fall back + mark unsafe), so a mask/isolation regression cannot ship
            # silently. When off / ungrouped / unverified, the packed path below runs as before.
            _pg_result = None
            _pg_use = False
            _pg_skip_pk = False  # once a shape is PG-verified, skip the full-row forward
            _pg_forward_fn = None  # deferred PG forward (runs at the verify site below)
            _pg_num_gen = getattr(self, "num_generations", None)
            # Env gate hoisted to module level (mirrored via RL_PRE_ITEMS). Skip PG under vLLM
            # (fast_inference=True): the rollout dominates the step, so PG saves little and its
            # first-use self-verify is net overhead.
            _pg_engage = (
                UNSLOTH_GRPO_PREFIX_GROUPER_ON
                and not getattr(self, "use_vllm", False)
                and not getattr(unwrapped_model, "_unsloth_prefix_grouper_nograd_disabled", False)
            )
            if _pg_engage:
                try:
                    # Skip softcap models (the flex kernel never applies attn_logit_softcapping)
                    # and hybrid SSM / MoE models: only the threaded attention forwards get the
                    # shared-prefix isolation, so a Mamba or MoE decoder that does not forward
                    # prefix_seg_info would leak suffixes across completions. PG also rides on
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
                        # normal backends apply config.attention_dropout in training; the flex
                        # path is deterministic, so skip PG when it is set.
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
                    # cap the PG span (P+max(R)) at the sliding window, like the packed _pk_sw guard.
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
                                    device_type = "cuda", dtype = self._autocast_dtype
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
                            # clip to the loss window [B, logits_to_keep+max_left_pad]
                            _pg_w = logits_to_keep + max_left_pad
                            if _pg_r.shape[1] > _pg_w:
                                _pg_r = _pg_r[:, -_pg_w:]
                            return _pg_r

                        # trust only within the verified envelope: re-verify when T or the
                        # longest segment grows, like the packed path
                        _pg_T = int(_pg_layout.flat_ids.shape[1])
                        _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                        _pg_env = (
                            _pg_verified.get(_pg_sig) if isinstance(_pg_verified, dict) else None
                        )
                        if (not _pg_verify_on()) or (
                            _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                        ):
                            # trusted shape: run PG now and skip the full-row forward below
                            _pg_result = _pg_run_forward()
                            _pg_use = True
                            _pg_skip_pk = True
                        else:
                            # unverified shape: defer the forward until the packed reference
                            # exists (verify site below), so a declined packed path never wastes
                            # a whole-batch PG forward
                            _pg_forward_fn = _pg_run_forward
                except Exception as _pg_err:
                    _pg_result = None
                    _pg_use = False
                    _pg_skip_pk = False
                    _pg_forward_fn = None
                    # A FlexAttention/Triton compile failure or OOM here is GPU-wide, not
                    # layout-specific, so retrying the same PG forward every step just re-pays
                    # the failure. Persistently disable PG (mirrors the seq-packing handler
                    # setting _unsloth_seq_packing_nograd_ok = False); the packed/padded path
                    # below still produces the exact result.
                    unwrapped_model._unsloth_prefix_grouper_nograd_disabled = True
                    if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO PrefixGrouper (no-grad) disabled (fell back to packed): {_pg_err!r}",
                            flush = True,
                        )

            # ---- Sequence packing (default-on; disable with UNSLOTH_GRPO_SEQ_PACKING=0) ----
            # One varlen [1, sum L] block-diagonal forward replaces the padded [B, Lmax] loop
            # (exact per-row result; also fixes the padded path's left-pad RoPE error).
            # Self-verified vs the per-row forward, re-checked as T grows; falls back if a
            # backend ignores packed_seq_lengths. lm_head runs on completion positions only.
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
                    # sliding-window models lose the per-sequence local window in a packed stream
                    _pk_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    _pk_sw_ok = not (isinstance(_pk_sw, int) and _pk_sw > 0 and _pk_maxseg > _pk_sw)
                    # per-row completion mask (same as the loss); prompt-only rows count as inactive
                    _pk_cmask = create_completion_attention_mask(
                        input_ids[:, -_pk_W:], left_pad_tokens_per_prompt, max_left_pad, _pk_pad
                    )
                    _pk_active = int(_pk_cmask.any(dim = 1).sum())
                    # skip the packed forward entirely at known-unsafe lengths (avoids a wasted pass / OOM)
                    _pk_unsafe = getattr(
                        unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                    )
                    # cap the flattened forward at one padded [batch_size, seq_len] mini-batch's
                    # token budget; anything larger uses the chunked padded loop
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
                        # per-row completion start after left-packing (matches create_completion_attention_mask)
                        _pk_cstart = (_pk_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                        _pk_ctgt = (_pk_nz_idx[1:, 1] >= _pk_cstart[_pk_nz_idx[1:, 0]]) & _pk_within
                        with _get_inference_mode_context_manager(model):
                            with torch.amp.autocast(device_type = "cuda", dtype = self._autocast_dtype):
                                # use_cache=False: a KV cache silently disables varlen packing
                                _pk_hidden = unwrapped_model(
                                    input_ids = _pk_flat,
                                    position_ids = _pk_pos,
                                    packed_seq_lengths = torch.tensor(
                                        _pk_nz_cpu, dtype = torch.int32, device = input_ids.device
                                    ),
                                    use_cache = False,
                                ).logits
                                _pk_sel = chunked_hidden_states_selective_log_softmax(
                                    _pk_hidden[0, :-1, :][_pk_ctgt].unsqueeze(0),
                                    lm_head,
                                    _pk_flat[0, 1:][_pk_ctgt].unsqueeze(0),
                                    _pk_chunks,
                                    logit_scale_multiply,
                                    logit_scale_divide,
                                    logit_softcapping,
                                    temperature,
                                )[0]
                        # GPT-OSS offload race guard (matches the padded loop)
                        device_synchronize()
                        # scatter each logprob back to its (row, col) so [:, -_pk_W:] matches padded
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
                        # re-verify when T or the longest segment grows past what was verified
                        # (a LongRoPE cache switch can change the result)
                        _pk_vT = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_T", 0)
                        )
                        _pk_vS = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_seg", 0)
                        )
                        # debug: hand-edit this condition to force re-verify every step
                        if _pk_ok is True and _pk_T <= _pk_vT and _pk_maxseg <= _pk_vS:
                            _pk_use = True  # already verified for this shape
                        else:
                            # verify against the per-row forward (ground truth)
                            _pk_ref = torch.zeros_like(_pk_result)
                            with _get_inference_mode_context_manager(model):
                                with torch.amp.autocast(
                                    device_type = "cuda", dtype = self._autocast_dtype
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
                                        _pk_rsel = chunked_hidden_states_selective_log_softmax(
                                            _pk_rh[:, :-1, :],
                                            lm_head,
                                            _pk_real[:, 1:],
                                            1,
                                            logit_scale_multiply,
                                            logit_scale_divide,
                                            logit_softcapping,
                                            temperature,
                                        )[0]
                                        _pk_rcols = _pk_rmask.nonzero(as_tuple = False).squeeze(1)[
                                            1:
                                        ] - (_pk_L - _pk_W)
                                        _pk_rkeep = _pk_rcols >= 0
                                        _pk_ref[_pk_i, _pk_rcols[_pk_rkeep]] = _pk_rsel[
                                            _pk_rkeep
                                        ].to(torch.float32)
                            device_synchronize()
                            # compare over the loss-mask region only
                            _pk_cm = _pk_cmask.float()
                            _pk_diff = float(((_pk_result - _pk_ref).abs() * _pk_cm).max())
                            if UNSLOTH_ENABLE_LOGGING:
                                print(
                                    f"[Unsloth] GRPO seq-packing (no-grad) verify: T={_pk_T} maxseg={_pk_maxseg} packed-vs-perrow max|d|={_pk_diff:.4f}",
                                    flush = True,
                                )
                            # kernel-noise floor ~0.25; cross-sample contamination is >= 2.4
                            if _pk_diff < 7e-1:
                                unwrapped_model._unsloth_seq_packing_nograd_ok = True
                                # widen the trusted shape only when >= 2 completion rows exercised
                                # cross-sample packing; single-row passes prove nothing
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
                                    # contamination (attention ignores the packed mask): disable packing
                                    unwrapped_model._unsloth_seq_packing_nograd_ok = False
                                else:
                                    # likely a length boundary (LongRoPE): mark unsafe, keep smaller shapes
                                    unwrapped_model._unsloth_seq_packing_nograd_unsafe_T = (
                                        _pk_T if _pk_unsafe is None else min(_pk_unsafe, _pk_T)
                                    )
                                if UNSLOTH_ENABLE_LOGGING:
                                    print(
                                        f"[Unsloth] GRPO seq-packing (no-grad) fell back at T={_pk_T} (diff={_pk_diff:.3f})",
                                        flush = True,
                                    )
                except Exception as _pk_err:
                    # any failure: drop intermediates, use the padded loop, do not retry
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
            # ---- PrefixGrouper first-use self-verify (no-grad) ----
            # Compare the untrusted PG result to the full-row packed result (itself verified vs
            # per-row) over the completion mask: < tol_ok -> trust the structure; >= TOL_KILL ->
            # unsafe forever; borderline -> fall back this shape.
            if _pg_forward_fn is not None and not _pg_use:
                if _pk_use and _pk_result is not None:
                    try:
                        # deferred PG forward, run only now that the packed reference exists
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
                # else: no packed reference (packing off/failed) -> cannot verify; fall back.

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
                    with torch.amp.autocast(device_type = "cuda", dtype = self._autocast_dtype):
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
                            del outputs  # free hidden_states before chunked log-softmax

                            completion_input_ids_chunk = input_ids_chunk[
                                :, -(logits_to_keep + max_left_pad) :
                            ]
                            logits_chunk = logits_chunk[
                                :, -(logits_to_keep + max_left_pad + 1) :, :
                            ]
                            logits_chunk = logits_chunk[:, :-1, :]
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
                            # Essentially, for VLMs we do not go via the optimized path in models/,
                            # so we don't encounter the Flash Attn left-padding issue.
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
                            del outputs  # free hidden_states before chunked log-softmax

                            logits_chunk = logits_chunk[:, :-1, :]
                            completion_input_ids_chunk = input_ids_chunk[:, -logits_to_keep:]
                            # Guard: check if model returned hidden states or logits
                            if logits_chunk.shape[-1] == lm_head.shape[1]:
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
                                # Model returned logits directly - scaling/softcapping already applied by model forward
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                )
                    # This is needed to avoid race conditions with GPT OSS offload_embbed=True
                    # However, it seems that this line does not slow down or disrupt models.
                    device_synchronize()
                    all_logprobs_list.append(logprobs_chunk)
                if logprobs is None:  # padded fallback when packing was not used
                    logprobs = torch.cat(all_logprobs_list, dim = 0)

                entropies = None

            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
            # aux loss is unused: it is off by default (router_aux_loss_coef set to 0 in models/rl.py)
            # and explicit opt-in is rejected at trainer init, so this is always None (kept in the
            # return for TRL >= 1.7.0's 3-tuple contract).
            aux_loss = None
            return logprobs.detach(), entropies, aux_loss  # logps, entropies, aux_loss
            # input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            # logits = logits[:, -logits_to_keep:]
            # return logits
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            # logits = logits / self.temperature
            # logps = selective_log_softmax(logits, input_ids)

            # row_indices, col_indices = torch.where(logps < -20)

            # # Method 1: Check if tensors have elements
            # if len(row_indices) > 0 and len(col_indices) > 0:
            #     breakpoint()  # Breakpoint triggered here
            #     print("Found high values!")
            # return  logps #  compute logprobs for the input tokens

    function = inspect.getsource(_get_per_token_logps_and_entropies)
    if trl_version < Version("1.7.0"):
        # TRL < 1.7.0 unpacks (logps, entropies) at every call site; TRL >= 1.7.0
        # always unpacks (logps, entropies, aux_loss). Drop the aux_loss element so
        # the return arity matches the installed TRL. Regex tolerates comment /
        # whitespace drift on the return line; fail loud if the anchor ever stops
        # matching rather than silently shipping a 3-tuple to older TRL.
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


grpo_compute_loss = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
UnslothEfficientGRPO = RL_REPLACEMENTS["UnslothEfficientGRPO"]
grpo_accumulated_loss = RL_REPLACEMENTS["grpo_accumulated_loss"]
grpo_update_SamplingParams = RL_REPLACEMENTS["grpo_update_SamplingParams"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_model_config))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_final_logit_softcapping))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_mm_token_id))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_fix_mm_token_type_ids))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_clear_stateful_mrope))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(UnslothEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_accumulated_loss))
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_update_SamplingParams))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_get_inference_mode_context_manager))
# inspect.getsource inlines function bodies but not module imports, so constants the inlined
# grpo functions reference (e.g. UNSLOTH_ENABLE_LOGGING) must be redefined in the generated cache.
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


# Edit _get_per_token_logps to handle mixed precision
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
        # Compute the per-token log probabilities for the model

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
        # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models
        token_type_ids = inputs.get("token_type_ids", None)
        mm_token_type_ids = inputs.get("mm_token_type_ids", None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        tool_mask = inputs.get("tool_mask", None)
        # Missing when evaluate() runs standalone; eval does not accumulate, so
        # fall back to 1 to avoid underreporting eval_loss (#2464).
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
        # attention_mask = None
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = (
            lambda model, input_ids, attention_mask, logits_to_keep, batch_size = None, compute_entropy = False, compute_efficient = False: (
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
        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        # if self.beta != 0.0:
        #     with torch.inference_mode(), model.disable_adapter():
        #         ref_per_token_logps = per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)
        # else:
        #     ref_per_token_logps = None
        ref_logps = inputs.get("ref_per_token_logps", None)
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        old_logps = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        # Get logit softcapping and logit scale
        model_config = _unsloth_get_model_config(model)
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
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
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
            # Compute the clipped probability ratios
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


# Fix KTO shape mismatch when Unsloth model forward truncates input_ids
# but labels aren't truncated. TRL 0.27.2+ _process_tokens only truncates
# completions, not prompts -- so prompts exceeding max_seq_length cause the
# model to produce shorter logits than the labels expect.
def kto_trainer_get_batch_logps(function_name, function):
    if function_name != "get_batch_logps":
        return function
    # The raise is inside an if block inside the method, so we need
    # to preserve the exact indentation of the raise statement.
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


# TRL 1.x dropped KTOTrainer.get_batch_logps and moved the log-prob math into
# _compute_logps / compute_ref_log_probs / _compute_kl_logps, which call
# selective_log_softmax on completion-only tokens. Same truncation hazard as
# above, so clamp logits/ids/mask to the shorter seq length (no-op when equal).
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


# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L356
# TRL warns if batch size is not a multiple of num_generations -> fix this.
def grpo_trainer_fix_batch_size(RLTrainer_source, RLConfig_source):
    if "divisible by the number of generations" not in RLTrainer_source:
        # in later trl versions this doesn't exist anymore
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


# Add other reward function names
def grpo_trainer_metrics(RLTrainer_source, RLConfig_source):
    if "reward_funcs" not in RLTrainer_source:
        return ""

    # For new TRL we have /mean and /std
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
    # This function patches the trl openenv generate_rollout_completions function to:
    # 1. Guard the reload_weights call (skip when sharing weights with vLLM)
    # 2. Fix wake_up call to be compatible with unsloth (remove tags to wake everything)
    #
    # The issue: TRL's wake_up(tags=["kv_cache"]) only wakes kv_cache, leaving is_sleeping=True
    # at the executor level. This causes unsloth's patched generate to try waking up again,
    # resulting in double create_and_map on already-mapped handles.
    #
    # The fix: Use wake_up() with no tags, which wakes everything. Unsloth's patched
    # CuMemAllocator.wake_up skips weights anyway, so this is safe.
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

    # trl 0.28 changed the function name yet again! Thanks trl :)
    patch_target_name = "_generate_rollout_completions_colocate"
    if hasattr(openenv_utils, patch_target_name):
        patch_target = getattr(openenv_utils, patch_target_name)
    else:
        # Older TRL versions may keep sleep/wake logic in the public dispatcher.
        patch_target_name = "generate_rollout_completions"
        patch_target = getattr(openenv_utils, patch_target_name)

    # TRL 0.29.1+ ships some openenv helpers as compiled bytecode without
    # accessible source on disk; inspect.getsource raises OSError("could
    # not get source code") in that case. Skip the source-rewrite patch
    # rather than crash. The unmodified TRL openenv path will run, which
    # means the duplicate `collective_rpc("reload_weights")` is NOT
    # stripped (line 1800 below) and `wake_up(tags=["kv_cache"])` is NOT
    # retagged to `wake_up()` (line 1804). Users who do not use openenv
    # GRPO are unaffected; openenv GRPO users on this TRL build may see
    # redundant reload_weights calls or partial wake_up behavior.
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

    # Change wake_up(tags=["kv_cache"]) to wake_up() - wake everything to set is_sleeping=False
    # This prevents double wake_up issues. Unsloth's allocator skips weights anyway.
    src = re.sub(r"\.wake_up\(tags=\[.*?\]\)", ".wake_up()", src)

    if original_src == src:
        logger.warning("Unsloth: Warning - regex did not match, patch may have failed")
        return

    # Execute and explicitly assign to module
    local_ns = {}
    exec(compile(src, "<unsloth>", "exec"), openenv_utils.__dict__, local_ns)
    patched_func = local_ns[patch_target_name]

    # Patch the target function in utils; if dispatcher was patched also update parent module alias.
    setattr(openenv_utils, patch_target_name, patched_func)
    if patch_target_name == "generate_rollout_completions":
        openenv.generate_rollout_completions = patched_func
    logger.info(f"Unsloth: Patched trl openenv {patch_target_name}")


RL_ADDITIONAL_FUNCTIONS["openenv"].append(openenv_vllm_reload_weights)


def vllm_generation_init_patch():
    # trl moved vllm stuff to trl/generation/vllm_generation.py
    # We need to patch it to not instantiate another vLLM instance if we already have one with fast_inference
    # Edit the TRL source directly and install the patched function in the TRL module.
    # https://github.com/huggingface/trl/commit/0eb66d8f2fc63b3d00d8dbc18f99c3f48750bd16
    # This exists in trl versions 0.28.0 and above

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

    # Patch init to remove vLLM.LLM instantiation
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

    # has some sync_weights or reload rpc calls.
    # we patched the grpo_trainer to strip them for prev versions
    # Ref: grpo_trainer__generate_single_turn above around L270-280
    def patch_sync_weights(src):
        pattern = re.compile(
            r"^(?P<def_line>def sync_weights\(self\):\n)(?P<body>(?:.*\n)*)",
            re.MULTILINE,
        )

        def replace_sync_weights(match):
            body = match.group("body")
            # Chain getattr so server mode (where self.llm is not set) does
            # not raise AttributeError before the default kicks in.
            guard = (
                "    if getattr(getattr(self, 'llm', None), 'shared_weights', False) or "
                "getattr(self, 'unsloth_fast_inference_lora', False):\n"
                "        # Unsloth fast inference LoRA shares weights with vLLM already.\n"
                "        return\n\n"
            )
            return match.group("def_line") + guard + body

        patched_src, num_replacements = pattern.subn(replace_sync_weights, src, count = 1)
        if num_replacements == 0:
            raise RuntimeError(
                "Unsloth: Warning - regex did not match, VLLMGeneration.sync_weights patch may have failed"
            )
        return patched_src

    def patch_generate(src):
        pattern = re.compile(
            r"^(?P<indent>[ \t]*)self\.llm\.collective_rpc\(\s*(['\"])reload_weights\2\s*\)\s*$",
            re.MULTILINE,
        )

        def replace_reload_weights(match):
            indent = match.group("indent")
            # Chain getattr so server mode (no self.llm) is safe here too.
            return (
                f"{indent}if not (getattr(getattr(self, 'llm', None), 'shared_weights', False) or "
                f"getattr(self, 'unsloth_fast_inference_lora', False)):\n"
                f'{indent}    self.llm.collective_rpc("reload_weights")'
            )

        patched_src, num_replacements = pattern.subn(replace_reload_weights, src, count = 1)
        if num_replacements == 0:
            raise RuntimeError(
                "Unsloth: Warning - regex did not match, VLLMGeneration.generate patch may have failed"
            )

        # Inject lora_request when sharing weights (vLLM needs the adapter)
        lora_generate_pattern = re.compile(
            r"(self\.llm\.generate\([^\)]+)\)",
        )

        def inject_lora_request(match):
            return (
                f"{match.group(1)}, lora_request="
                f"self._unsloth_load_lora('vllm_gen_lora', load_tensors=True) "
                f"if hasattr(self, '_unsloth_load_lora') else None)"
            )

        patched_src = lora_generate_pattern.sub(inject_lora_request, patched_src)
        return patched_src

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
        generate_patched = patch_vllm_generation_method(
            "generate",
            patch_generate,
            "if not (getattr(getattr(self, 'llm', None), 'shared_weights', False) or getattr(self, 'unsloth_fast_inference_lora', False)):",
            "generate",
        )
    except RuntimeError as e:
        logger.warning(str(e))
        return

    if init_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration._init_vllm")
    if sync_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration.sync_weights")
    if generate_patched:
        logger.info("Unsloth: Patched trl VLLMGeneration.generate")


RL_ADDITIONAL_FUNCTIONS["vllm_generation"].append(vllm_generation_init_patch)
