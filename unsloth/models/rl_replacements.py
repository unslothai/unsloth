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

"""Enhanced replacements module for Unsloth RL adapters.

This module contains replacement snippets, helpers and compatibility
fixes injected into TRL-style trainer code. The original file was kept
functionally equivalent where possible; here we focus on:

- better typing and small runtime safety checks
- explicit imports (e.g. nullcontext)
- centralized regex patterns and compiled replacements
- structured logging instead of print where appropriate
- clearer docstrings and helper utilities

Notes:
- This file is intended to be used as a drop-in replacement for the
  original; it keeps the same global names expected by the rest of the
  system (RL_EXTRA_ARGS, RL_FUNCTIONS, ...).
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import textwrap
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# The RL_REPLACEMENTS import may contain prebuilt replacement functions.
# When running in environments where unsloth_zoo isn't available this
# module should fail loudly at import time to help debugging.
from unsloth_zoo.rl_replacements import RL_REPLACEMENTS, left_pack_padding

# Keep device helpers local so callers rely on consistent names
from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

# Module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Public API exported names (kept for backward compatibility)
RL_EXTRA_ARGS: Dict[str, List[Callable]] = defaultdict(list)
RL_FUNCTIONS: Dict[str, List[Callable]] = defaultdict(list)
RL_PRE_ITEMS: Dict[str, List[str]] = defaultdict(list)
RL_CONFIG_CHANGES: Dict[str, List[Callable]] = defaultdict(list)
RL_METRICS_CHANGES: Dict[str, List[Callable]] = defaultdict(list)

# Recommended torch.compile options. Kept as-is but typed.
torch_compile_options: Dict[str, Any] = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}

# ------------------------
# Utility helpers
# ------------------------

def _safe_getsource(obj: Any) -> str:
    """Return source for `obj` when possible, else an empty string.

    This is used because inspect.getsource can raise for builtins or
    objects created dynamically.
    """
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def _ensure_key(key: str) -> None:
    """Ensure RL_REPLACEMENTS has a key to avoid KeyError on lookups."""
    if key not in RL_REPLACEMENTS:
        RL_REPLACEMENTS[key] = None


# ------------------------
# Convenience replacements
# ------------------------

# Check untrained tokens - attach to sft_trainer as an injection
def sft_trainer_fix_untrained_tokens(call_args: Dict[str, Any], extra_args: Dict[str, Any]) -> str:
    """Return a small script that fixes tokenizer / zero-loss issues when
    `model` and `train_dataset` exist in call args.

    The returned string is intended to be injected into trainer launch
    code so that issues with newly added tokens in tokenizers are
    patched at runtime.
    """
    if "model" in call_args and "train_dataset" in call_args:
        fix_tokenizer = (
            "IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\\n')\n"
            "from unsloth_zoo.tokenizer_utils import fix_untrained_tokens\n"
            "from unsloth_zoo.training_utils  import fix_zero_training_loss\n"
            "if 'tokenizer' not in locals(): tokenizer = processing_class\n"
            "fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n"
            "fix_zero_training_loss(model, tokenizer, train_dataset)\n"
        )
        logger.debug("Injecting sft_trainer tokenizer fixes")
        return fix_tokenizer
    return ""


RL_EXTRA_ARGS["sft_trainer"].append(sft_trainer_fix_untrained_tokens)


# Remove DPO columns which might randomly be tokenized
def dpo_trainer_fix_columns(call_args: Dict[str, Any], extra_args: Dict[str, Any]) -> str:
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
        logger.debug("Injecting dpo_trainer column fixes")
        return fix_dpo
    return ""


RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_columns)


# ------------------------
# Prepare dataset patch for SFT trainers
# ------------------------

def sft_trainer_prepare_dataset(function_name: str, function: str) -> str:
    """Patch `_prepare_dataset` (and related) to avoid double-BOS and
    to optionally use a faster prebuilt replacement from RL_REPLACEMENTS.

    This function preserves the original implementation and inserts a
    defensive preamble checking for `has_bos_token_already` and
    adjusting tokenizer behavior.
    """
    if function_name not in ("_prepare_non_packed_dataloader", "_prepare_dataset"):
        return function

    fast_sft_prepare_dataset = RL_REPLACEMENTS.get("sft_prepare_dataset", None)
    if fast_sft_prepare_dataset is not None:
        # If we can fully replace the function with a fast version, do so.
        params = inspect.signature(fast_sft_prepare_dataset).parameters.keys()
        params_regex = ".*?".join(params)
        matched = re.match(
            r"[\s]{0,}def _prepare_dataset\(.*?" + params_regex + r".*?\)",
            function,
            flags=re.MULTILINE | re.DOTALL,
        )
        if matched:
            function = _safe_getsource(fast_sft_prepare_dataset)
            function = function.split("\n")
            function = "\n".join(" " * 4 + x for x in function)
            function = function.replace("def sft_prepare_dataset", "def _prepare_dataset")
            logger.debug("Using fast sft_prepare_dataset replacement")
            return function

    # Defensive preamble to avoid adding BOS tokens twice
    check_text = (
        "if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:\n"
        "    return dataset\n"
        "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"
        "if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')\n"
        "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"
        "if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')\n"
        "test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]\n"
        "chat_template = getattr(tokenizer, 'chat_template', None)\n"
        "chat_template = '' if chat_template is None else chat_template\n"
        "has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) "
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

    # Indent the injected preamble consistently for readability
    check_text = check_text.split("\n")
    check_text = "\n".join(" " * 8 + x for x in check_text)
    check_text = check_text.rstrip() + "\n"

    # Find the function signature line and append the check_text after it
    replacer = re.findall(r"def " + function_name + r"\(.*?\).*?:\n", function, flags=re.MULTILINE | re.DOTALL)
    if replacer:
        header = replacer[0]
        function = function.replace(header, header + check_text, 1)
        logger.debug("Patched %s with BOS-protection preamble", function_name)

    # Ensure we restore tokenizer.__call__ before any return statement
    return_state = "if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call\n"
    function = re.sub(r"\n([ ]{4,})(return .*?[\s]{0,})$", rf"\1{return_state}\1\2", function)
    return function


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_prepare_dataset)


# ------------------------
# SFT compute_loss shim
# ------------------------

def sft_trainer_compute_loss(function_name: str, function: str) -> str:
    """Replace compute_loss in subclassed trainer to call super's implementation
    and return it. This avoids recomputing extra metrics which may require
    logits/higher memory.
    """
    if function_name != "compute_loss":
        return function

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        return outputs

    function = _safe_getsource(compute_loss)
    logger.debug("Injected simple compute_loss shim for sft_trainer")
    return function


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)


# ------------------------
# GRPO trainer: mixed precision, generation fixes and loss
# ------------------------

def grpo_trainer__prepare_inputs(function_name: str, function: str) -> str:
    if function_name != "_prepare_inputs":
        return function

    # Add mixed precision training using the correct context manager imports
    function = function.replace(
        "with torch.inference_mode():",
        (
            "with torch.inference_mode(), "
            "torch.amp.autocast(device_type = 'cuda', "
            "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "
            "if not torch.is_autocast_enabled('cuda') else nullcontext())"
            "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):"
        ),
    )
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    logger.debug("Patched _prepare_inputs to include amp.autocast and unwrap_model options")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Fix incorrect special tokens handling and truncation in older TRL versions
def grpo_trainer__generate_and_score_completions(function_name: str, function: str) -> str:
    if function_name != "_generate_and_score_completions":
        return function

    # Ensure skip_special_tokens=False (older TRL versions incorrectly set True)
    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    # Left pad prompt before calculation of old and ref hidden states
    line_to_replace = "batch_size = self.args.per_device_train_batch_size if mode == \"train\" else self.args.per_device_eval_batch_size"

    replacement_lines = textwrap.dedent(
        """
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        try:
            # TRL 0.23.1 and below path
            if not has_images:
                # Left pad prompt before calculation old and ref hidden states
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
            self.model.for_training()
        except:
            # TRL 0.24.0 and below path
            if images is None:
                # Left pad prompt before calculation old and ref hidden states
                prompt_completion_ids = left_pack_padding(prompt_completion_ids, self.processing_class.pad_token_id)
            self.model.for_training()
        """
    )

    function = function.replace(line_to_replace, replacement_lines)

    # Lifted-from-original: sanitize logprobs extraction when using vLLM outputs
    pattern_to_find = re.compile(
        r"^\s*if self\.args\.gradient_accumulation_steps % generate_every != 0 or \(\s*"
        r"self\.use_vllm and self\.vllm_importance_sampling_correction\s*"
        r"\):",
        re.MULTILINE,
    )

    replacement_text = (
        "\n            if self.args.gradient_accumulation_steps % generate_every != 0 or (\n"
        "                self.use_vllm\n"
        "            ):"
    )

    function, num_replacements = pattern_to_find.subn(replacement_text, function)
    if num_replacements:
        logger.debug("Adjusted gradient_accumulation_steps conditional in generation code")

    # Replace the all_logprobs comprehension with a sanitized version
    pattern_to_find = re.compile(
        r"(^\s*)all_logprobs = \["  # Capture indentation (group 1)
        r".*?"                      # Match everything inside non-greedily
        r"for output in outputs\.outputs\s*"
        r"\]",
        re.DOTALL | re.MULTILINE,
    )

    replacement_text = (
        r"\1from trl.scripts.vllm_serve import sanitize_logprob\n"
        r"\1all_logprobs = [\n"
        r"\1    [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]\n"
        r"\1    for outputs in all_outputs\n"
        r"\1    for output in outputs.outputs\n"
        r"\1]"
    )

    function, num_replacements = pattern_to_find.subn(replacement_text, function)
    if num_replacements:
        logger.debug("Patched all_logprobs comprehension to use sanitize_logprob")

    # Complex replacement for max_prompt_length handling: keep behavior but
    # remove accidental comments and improve readability
    found = re.findall(
        r"\n(([ ]{8,})if self\.max_prompt_length is not None:.*?" r"\2if self\.use_vllm:)",
        function,
        flags=re.DOTALL | re.MULTILINE,
    )
    if found:
        replace_part, spacing = found[0]
        removed_comments = re.sub(r"\#[^\n]{1,}", "", replace_part)
        splits = removed_comments.split("\n")
        if sum(re.match(rf"{spacing}[^\s]", x) is not None for x in splits) == 2 and len(spacing) >= 8:

            new_replacement = textwrap.dedent(
                f"""
                \n{spacing}if self.max_prompt_length is not None:
                    # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
                    # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
                    # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
                    protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
                    protected = [token for token in protected if token is not None]
                    prompt_ids, prompt_mask = truncate_with_protected_tokens(
                        prompt_ids, prompt_mask, self.max_prompt_length, protected
                    )

                    prompts_text = [re.sub(rf"^(?:{{re.escape(self.pad_token)}})+", "", text) for text in prompts_text]

                    # The chat template inserts a single image token into the prompt text. However, when this text is later
                    # tokenized, the single image token string is expanded into multiple image token IDs, depending on the
                    # image size. Since we're detokenating here, we may see repeated image tokens in the decoded text. We
                    # collapse them back into a single token string to match the original template.
                    if self.image_token is not None:
                        prompts_text = [
                            re.sub(rf"(?:{{re.escape(self.image_token)}})+", self.image_token, text) for text in prompts_text
                        ]
                # Generate completions using either vLLM or regular generation
                if self.use_vllm:
                """
            )
            function = function.replace(replace_part, new_replacement)
            logger.debug("Rewrote max_prompt_length handling for readability")

    # Ensure sampling_per_token_logps is propagated for vLLM path
    string_to_find = (
        '        if "image_sizes" in prompt_inputs:\n'
        '            output["image_sizes"] = prompt_inputs["image_sizes"]'
    )

    replacement_string = (
        '        if "image_sizes" in prompt_inputs:\n'
        '            output["image_sizes"] = prompt_inputs["image_sizes"]\n\n'
        '        if self.use_vllm:\n'
        '            try:\n'
        '                output["sampling_per_token_logps"] = sampling_per_token_logps\n'
        '            except NameError:\n'
        '                output["sampling_per_token_logps"] = None'
    )

    if string_to_find in function:
        function = function.replace(string_to_find, replacement_string)
        logger.debug("Ensured vLLM sampling_per_token_logps handling is present")

    # Add wake_up/sleep hooks if missing (helps vLLM sleep mode compatibility)
    if "wake_up()" not in function:
        pattern = re.compile(r".*self\.llm\.generate\(.*\).*", re.MULTILINE)
        matches = list(pattern.finditer(function))
        patched = function

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
        logger.debug("Added vLLM wake_up/sleep wrapper around llm.generate calls")

    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate_and_score_completions)


# Fix {"reasoning_effort" : "high"} not applied in certain chat template flows
def grpo_trainer_fix_maybe_apply_chat_template(function_name: str, function: str) -> str:
    spaces = function.find("def ")
    if spaces % 4 != 0:
        return function
    spaces += 4
    replacement = textwrap.dedent(
        """
        _chat_template_ = getattr(self.processing_class, "chat_template", None)
        if _chat_template_ is None: _chat_template_ = ""
        _supported_keys_ = set(("prompt", "chosen", "rejected", "completion", "messages", "label"))

        prompts_text = []
        for _example_ in __INPUTS__REPLACEMENT__:
            _tokenizer_kwargs_ = {}
            if type(_example_) is not dict:
                _example_ = {"prompt": _example_}
            _left_keys_ = _example_.keys() - _supported_keys_
            for k in _left_keys_:
                if k in _chat_template_:
                    v = _example_[k]
                    if type(v) is str:
                        _tokenizer_kwargs_[k] = v
            _x_ = maybe_apply_chat_template(_example_, self.processing_class, **_tokenizer_kwargs_)["prompt"]
            prompts_text.append(_x_)
    """
    ).strip()
    replacement = textwrap.indent(replacement, spaces * " ")
    replacement = f"\n{replacement}\n"
    what = 'prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]'
    function = function.replace(what, replacement.replace("__INPUTS__REPLACEMENT__", "inputs"))

    # Also handle the prompts list variant
    function = re.sub(
        r"prompts_text = \[\n[\s]{0,}maybe_apply_chat_template\(\{[\"\']prompt[\"\']\:[\s]{0,}prompt[\s]{0,}\}[\s]{0,},[\s]{0,}self\.processing_class\)\[[\"\']prompt[\"\']] for prompt in prompts\s*\]",
        replacement.replace("__INPUTS__REPLACEMENT__", "prompts"),
        function,
    )
    logger.debug("Patched maybe_apply_chat_template handling to support additional keys")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_fix_maybe_apply_chat_template)


# Remove _move_model_to_vllm to avoid accidental model duplication
def grpo_trainer__move_model_to_vllm(function_name: str, function: str) -> str:
    if function_name != "_move_model_to_vllm":
        return function

    def _move_model_to_vllm(self, *args, **kwargs):
        """No-op replacement: moving models into vLLM is handled elsewhere in
        the Unsloth runtime. Returning None prevents unexpected behavior."""

        return None

    function = _safe_getsource(_move_model_to_vllm)
    logger.debug("Replaced _move_model_to_vllm with a no-op")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision — return None in
# Unsloth efficient path to indicate caller should use accumulated path.
def grpo_trainer__get_per_token_logps(function_name: str, function: str) -> str:
    if function_name != "_get_per_token_logps":
        return function

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, compute_efficient: bool = False):
        # Currently Unsloth uses an efficient GRPO path; return None to
        # indicate the caller should fallback to the accumulated version.
        return None

    function = _safe_getsource(_get_per_token_logps)
    logger.debug("Simplified _get_per_token_logps to return None (Unsloth efficient path)")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)


def grpo_trainer__get_per_token_logps_and_entropies(function_name: str, function: str) -> str:
    if function_name != "_get_per_token_logps_and_entropies":
        return function

    # Provide a slower but compatible version which supports compute_entropy
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                           compute_entropy: bool = False, compute_efficient: bool = False, *args, **kwargs):
        if compute_efficient:
            return None, None

        if not hasattr(self, "_autocast_dtype"):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':
                self._autocast_dtype = torch.float16

        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        pixel_attention_mask = kwargs.get('pixel_attention_mask', None)
        image_sizes = kwargs.get('image_sizes', None)

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

        unwrapped_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

        with torch.amp.autocast(device_type='cuda', dtype=self._autocast_dtype):
            with torch.inference_mode():
                if pixel_values is None:
                    attention_mask = input_ids != self.processing_class.pad_token_id
                    attention_mask = attention_mask.to(attention_mask.dtype)
                    logits = unwrapped_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        pixel_attention_mask=pixel_attention_mask,
                        image_sizes=image_sizes,
                    ).logits
                else:
                    logits = unwrapped_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        pixel_attention_mask=pixel_attention_mask,
                        image_sizes=image_sizes,
                        logits_to_keep=logits_to_keep + 1,
                    ).logits

                entropies = None
                if compute_entropy:
                    from trl.trainer.utils import entropy_from_logits

                    entropies = entropy_from_logits(logits)

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        return logits, entropies

    function = _safe_getsource(_get_per_token_logps_and_entropies)
    logger.debug("Provided compatible _get_per_token_logps_and_entropies implementation")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps_and_entropies)


# Pre-inserted helper code pulled from RL_REPLACEMENTS if available
# These are often long helper blobs used by the GRPO compute loss path.
for name in ("grpo_compute_loss", "grpo_compute_loss_slow", "UnslothEfficientGRPO", "grpo_accumulated_loss"):
    blob = RL_REPLACEMENTS.get(name)
    if blob is not None:
        source = _safe_getsource(blob) or blob
        RL_PRE_ITEMS["grpo_trainer"].append(source)
        logger.debug("Appended pre-item: %s", name)


# Replace compute_loss in GRPO trainer with an implementation that uses
# the Unsloth-optimized or accumulated loss path depending on availability.
def grpo_trainer_compute_loss(function_name: str, function: str) -> str:
    if function_name != "compute_loss":
        return function

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        pixel_attention_mask = inputs.get('pixel_attention_mask', None)
        image_sizes = inputs.get('image_sizes', None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        current_gradient_accumulation_steps = self.current_gradient_accumulation_steps
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        _input_ids = input_ids

        get_logps_func = (
            lambda model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, compute_efficient=False:
            self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, compute_efficient)
            if hasattr(self, "_get_per_token_logps") else
            self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy, compute_efficient)[0]
        )

        per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep, compute_efficient=True)

        # The reference hidden states (ref_per_token_logps) used to compute KL may
        # sometimes be provided directly by the dataloader/prep; accept it if present.
        ref_hidden_states = inputs.get("ref_per_token_logps", None)
        advantages = inputs["advantages"]
        old_hidden_states = inputs.get("old_per_token_logps", None)

        # For accumulation fallback, keep original input ids tail
        input_ids = input_ids[:, -logits_to_keep:]

        # Handle model-specific logit scaling options
        logit_softcapping = getattr(model.config, "final_logit_softcapping", 0) or 0
        logit_scale_multiply = getattr(model.config, "logit_scale", 0) or 0
        logit_scale_divide = getattr(model.config, "logits_scaling", 0) or 0

        if per_token_logps is not None:
            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[:, :-1, :]
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[:, :-1, :]
            per_token_logps = per_token_logps[:, :-1, :]

            loss, completion_length, mean_kl, delta, flat_is_ratio = grpo_compute_loss_slow(
                ref_hidden_states,
                per_token_logps,
                old_hidden_states,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                loss_type=self.args.loss_type,
                importance_sampling_level=self.importance_sampling_level,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                max_completion_length=self.args.max_completion_length,
                delta=self.args.delta,
                temperature=self.args.temperature,
                logit_softcapping=logit_softcapping,
                logit_scale_multiply=logit_scale_multiply,
                logit_scale_divide=logit_scale_divide,
                num_items_in_batch=num_items_in_batch,
                current_gradient_accumulation_steps=current_gradient_accumulation_steps,
                num_processes=num_processes,
                sampling_per_token_logps=sampling_per_token_logps,
            )
        else:
            if hasattr(self.args, "loss_type"):
                loss, completion_length, mean_kl, delta, flat_is_ratio = grpo_accumulated_loss(
                    trainer=self,
                    input_ids=_input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    logits_to_keep=logits_to_keep,
                    completion_mask=completion_mask,
                    advantages=advantages,
                    old_hidden_states=old_hidden_states,
                    ref_hidden_states=ref_hidden_states,
                    n_chunks=self.args.unsloth_num_chunks,
                    loss_type=self.args.loss_type,
                    importance_sampling_level=self.importance_sampling_level,
                    epsilon_low=self.epsilon_low,
                    epsilon_high=self.epsilon_high,
                    max_completion_length=self.args.max_completion_length,
                    delta=self.args.delta,
                    temperature=self.args.temperature,
                    logit_softcapping=logit_softcapping,
                    logit_scale_multiply=logit_scale_multiply,
                    logit_scale_divide=logit_scale_divide,
                    num_items_in_batch=num_items_in_batch,
                    current_gradient_accumulation_steps=current_gradient_accumulation_steps,
                    num_processes=num_processes,
                    sampling_per_token_logps=sampling_per_token_logps,
                )
            else:
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer=self,
                    input_ids=_input_ids,
                    logits_to_keep=logits_to_keep,
                    completion_mask=completion_mask,
                    advantages=advantages,
                    old_hidden_states=old_hidden_states,
                    ref_hidden_states=ref_hidden_states,
                    n_chunks=self.args.unsloth_num_chunks,
                    temperature=self.args.temperature,
                    logit_softcapping=logit_softcapping,
                    logit_scale_multiply=logit_scale_multiply,
                    logit_scale_divide=logit_scale_divide,
                    attention_mask=attention_mask,
                )

        # Metrics bookkeeping
        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        # Extra sampling-related metrics for vLLM
        if getattr(self, "use_vllm", False) and 'delta' in locals() and delta is not None:
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=self.model.device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=self.model.device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=self.model.device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=self.model.device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=self.model.device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                torch.nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                torch.nanmean(self.accelerator.gather(mean_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                torch.nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        return loss

    function = _safe_getsource(compute_loss)
    logger.debug("Injected GRPO compute_loss replacement")
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_compute_loss)


# ------------------------
# Configuration and metrics patches
# ------------------------

def grpo_trainer_fix_batch_size(RLTrainer_source: str, RLConfig_source: str) -> str:
    """Fixes TRL warning regarding per-device batch size and num_generations.

    If per_device_train_batch_size is not divisible by num_generations, this
    will add a small snippet to set the batch size to num_generations and
    print a helpful message.
    """
    if "divisible by the number of generations" not in RLTrainer_source:
        return ""
    if "num_generations" not in RLConfig_source:
        return ""

    check_batch_size = (
        "div = per_device_train_batch_size // num_generations\n"
        "if div * num_generations != per_device_train_batch_size:\n"
        "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n'"
        "               'We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"
        "    per_device_train_batch_size = num_generations\n"
    )
    logger.debug("Prepared batch size fix snippet for GRPO trainer")
    return check_batch_size


RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_trainer_fix_batch_size)


def grpo_trainer_metrics(RLTrainer_source: str, RLConfig_source: str) -> str:
    """Append metric names for reward functions, compatible across TRL
    versions that use either raw reward or mean/std suffixed metrics."""
    if "reward_funcs" not in RLTrainer_source:
        return ""

    use_mean = "rewards/{reward_func_name}/mean" in RLTrainer_source
    use_std = "rewards/{reward_func_name}/std" in RLTrainer_source
    if not use_mean:
        use_normal = "rewards/{reward_func_name}" in RLTrainer_source
    else:
        use_normal = False

    log_metrics = textwrap.dedent(
        """
        if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]
        else: _reward_funcs = reward_funcs
        for reward_func in _reward_funcs:
            try:
                reward_func_name = reward_func.__name__
                if %USE_MEAN%:
                    other_metrics.append(f'rewards/{reward_func_name}/mean')
                if %USE_STD%:
                    other_metrics.append(f'rewards/{reward_func_name}/std')
                if %USE_NORMAL%:
                    other_metrics.append(f'rewards/{reward_func_name}')
            except: pass
        """
    )

    log_metrics = log_metrics.replace("%USE_MEAN%", str(use_mean))
    log_metrics = log_metrics.replace("%USE_STD%", str(use_std))
    log_metrics = log_metrics.replace("%USE_NORMAL%", str(use_normal))

    logger.debug("Prepared reward metrics logging snippet")
    return log_metrics


RL_METRICS_CHANGES["grpo_trainer"].append(grpo_trainer_metrics)

# End of file
