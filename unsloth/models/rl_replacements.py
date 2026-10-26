# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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
    detect_logit_transforms = None
import importlib.util
from ..device_type import (
    is_hip,
    get_device_type,
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
# Zoo >= unslothai/unsloth-zoo#1685 scores a biased output head; mirrored via RL_PRE_ITEMS.
try:
    UNSLOTH_ZOO_HAS_HEAD_BIAS = (
        "lm_head_bias" in inspect.signature(chunked_hidden_states_selective_log_softmax).parameters
    )
except Exception:
    UNSLOTH_ZOO_HAS_HEAD_BIAS = False
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


# huggingface/trl#4695 added top_k to GRPOConfig defaulting to 0, but vLLM's include-all top_k is -1 and 0 errors on SamplingParams creation.
def grpo_config_fix_vllm_top_k(old_RLTrainer_source, old_RLConfig_source):
    return "if use_vllm and (top_k is None or top_k == 0): top_k = -1\n"


RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_config_fix_vllm_top_k)


def gkd_trainer_mask_prompt(function_name, function):
    if function_name != "generate_on_policy_outputs":
        return function
    prompt_mask = 'new_labels[:, : inputs["prompts"].shape[1]] = -100'
    if prompt_mask in function or "new_labels[:, :prompt_length] = -100" in function:
        return function
    for line in function.splitlines():
        if line.strip() == "new_labels = generated_tokens.clone()":
            indent = line.partition("new_labels")[0]
            return function.replace(line, f"{line}\n{indent}{prompt_mask}", 1)
    return function


RL_FUNCTIONS["gkd_trainer"].append(gkd_trainer_mask_prompt)


def gkd_trainer_generate_before_counting(function_name, function):
    if function_name != "training_step":
        return function
    start = re.search(
        r"^        if (?:self\.seq_kd|random\.random\(\) <= self\.lmbda):", function, re.M
    )
    end = function.find("        loss = super().training_step(")
    if start is None or end < start.start():
        return function
    start = start.start()
    generation = function[start:end]
    function = (
        function[:start]
        + """        pending = getattr(self, "_unsloth_gkd_pending_batches", None)
        if pending is not None:
            batches, batch_args, batch_kwargs = pending
            self._unsloth_gkd_pending_batches = None

            def prepared_batches():
                for batch in batches:
                    self._unsloth_prepare_gkd_batch(model, batch)
                    yield batch

            _, count = super().get_batch_samples(
                prepared_batches(), len(batches), *batch_args, **batch_kwargs
            )
            # TRL 1.7-1.8 mask EOS when it is the pad token, so an all-EOS window counts 0: sum / 0 is NaN.
            if torch.is_tensor(count):
                count = count.clamp_min(1)
            elif count is not None:
                count = max(count, 1)
            for batch in batches:
                batch["_unsloth_gkd_num_items"] = count

        if "_unsloth_gkd_num_items" in inputs:
            num_items_in_batch = inputs.pop("_unsloth_gkd_num_items")
        else:
            self._unsloth_prepare_gkd_batch(model, inputs)
"""
        + function[end:]
    )
    # Prepare the whole accumulation window before counting its replacement
    # labels. Keep generation in training_step, after checkpoint RNG restoration.
    return (
        function
        + "\n    def _unsloth_prepare_gkd_batch(self, model, inputs):\n"
        + generation
        + """
    def get_batch_samples(self, epoch_iterator, num_batches, *args, **kwargs):
        batches, count = super().get_batch_samples(epoch_iterator, num_batches, *args, **kwargs)
        if self.lmbda > 0 or self.seq_kd:
            self._unsloth_gkd_pending_batches = (batches, args, kwargs)
        return batches, count
"""
    )


RL_FUNCTIONS["gkd_trainer"].append(gkd_trainer_generate_before_counting)


def ppo_trainer_free_rollout_logits(function_name, function):
    # TRL keeps the float32 (batch, response_length, vocab) generation logits alive through every PPO
    # epoch, though only the logprobs taken from them are read: 1.2 GB on a T4 for 8 x 256 Qwen3 tokens.
    if function_name != "train" or "logitss" not in function:
        return function
    old = "del (logprob, ref_logprob, full_value, value, score, unwrapped_model)"
    new = "del (logprob, ref_logprob, full_value, value, score, unwrapped_model, logitss)"
    return function.replace(old, new, 1)


RL_FUNCTIONS["ppo_trainer"].append(ppo_trainer_free_rollout_logits)


def _unsloth_replace_block(function, old_lines, new_lines):
    """Replace the one run of consecutive lines whose stripped text is ``old_lines`` with ``new_lines`` (written
    relative to the first old line's indent). None unless the run occurs exactly once."""
    lines = function.split("\n")
    stripped = [line.strip() for line in lines]
    n = len(old_lines)
    hits = [i for i in range(len(lines) - n + 1) if stripped[i : i + n] == list(old_lines)]
    if len(hits) != 1:
        return None
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    return "\n".join(
        lines[:i] + [indent + line if line else line for line in new_lines] + lines[i + n :]
    )


_PPO_FUSED_IMPORT = (
    "from unsloth.models._fused_rl_logprobs import (",
    "    ppo_route as _unsloth_ppo_route_of,",
    "    ppo_batch_generation as _unsloth_ppo_batch_generation,",
    "    ppo_forward as _unsloth_ppo_forward,",
    "    ppo_score as _unsloth_ppo_score,",
    "    ppo_policy_logprobs as _unsloth_ppo_policy_logprobs,",
    ")",
    "_unsloth_ppo_route = _unsloth_ppo_route_of(self)",
)
_PPO_FUSED_BLOCKS = (
    # The route is decided once per train() before the first rollout.
    (
        ("for update in range(1, args.num_total_batches + 1):",),
        _PPO_FUSED_IMPORT + ("for update in range(1, args.num_total_batches + 1):",),
    ),
    # Rollouts without the stacked float32 generation scores.
    (
        ("query_responses, logitss = batch_generation(",),
        (
            "query_responses, logitss = (",
            "    batch_generation if _unsloth_ppo_route is None else _unsloth_ppo_batch_generation",
            ")(",
        ),
    ),
    # Rollout (old) log-probs: the policy forward + fused scorer the training pass uses.
    (
        (
            "logits = logitss[i : i + args.local_rollout_forward_batch_size]",
            "logprob = selective_log_softmax(logits, response)",
            "del logits",
        ),
        (
            "if _unsloth_ppo_route is not None:",
            "    logprob = _unsloth_ppo_policy_logprobs(",
            "        _unsloth_ppo_route, forward, selective_log_softmax, model, query_response, response,",
            "        context_length, processing_class.pad_token_id, args.temperature + 1e-7,",
            "    )",
            "else:",
            "    logits = logitss[i : i + args.local_rollout_forward_batch_size]",
            "    logprob = selective_log_softmax(logits, response)",
            "    del logits",
        ),
    ),
    (
        ("ref_output = forward(model.policy, query_response, processing_class.pad_token_id)",),
        (
            "ref_output = _unsloth_ppo_forward(",
            "    _unsloth_ppo_route, forward, model.policy, query_response, processing_class.pad_token_id",
            ")",
        ),
    ),
    (
        ("ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)",),
        (
            "ref_output = _unsloth_ppo_forward(",
            "    _unsloth_ppo_route, forward, ref_policy, query_response, processing_class.pad_token_id",
            ")",
        ),
    ),
    (
        (
            "ref_logits = ref_output.logits[:, context_length - 1 : -1]",
            "ref_logits /= args.temperature + 1e-7",
            "ref_logprob = selective_log_softmax(ref_logits, response)",
            "del ref_output, ref_logits",
        ),
        (
            "_unsloth_scored = _unsloth_ppo_score(",
            "    _unsloth_ppo_route, model if ref_policy is None else ref_policy,",
            "    ref_output.logits[:, context_length - 1 : -1], response, args.temperature + 1e-7,",
            ")",
            "if _unsloth_scored is None:",
            "    ref_logits = ref_output.logits[:, context_length - 1 : -1]",
            "    ref_logits /= args.temperature + 1e-7",
            "    ref_logprob = selective_log_softmax(ref_logits, response)",
            "    del ref_output, ref_logits",
            "else:",
            "    ref_logprob = _unsloth_scored[0]",
            "    del ref_output, _unsloth_scored",
        ),
    ),
    # Training pass: the value head still reads the critic backbone; only the policy head is fused.
    (
        ("output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)",),
        (
            "output, vpred_temp = _unsloth_ppo_forward(",
            "    _unsloth_ppo_route, forward, model, mb_query_responses, processing_class.pad_token_id",
            ")",
        ),
    ),
    (
        (
            "logits = output.logits[:, context_length - 1 : -1]",
            "logits /= args.temperature + 1e-7",
            "new_logprobs = selective_log_softmax(logits, mb_responses)",
        ),
        (
            "_unsloth_scored = _unsloth_ppo_score(",
            "    _unsloth_ppo_route, model, output.logits[:, context_length - 1 : -1], mb_responses,",
            "    args.temperature + 1e-7, entropy = True,",
            ")",
            "if _unsloth_scored is None:",
            "    logits = output.logits[:, context_length - 1 : -1]",
            "    logits /= args.temperature + 1e-7",
            "    new_logprobs = selective_log_softmax(logits, mb_responses)",
            "else:",
            "    logits = None",
            "    new_logprobs, _unsloth_entropy = _unsloth_scored",
            "    del _unsloth_scored",
        ),
    ),
    # The entropy metric comes out of the same kernel pass (detached, temperature-scaled like TRL's).
    (
        (
            "prob_dist = torch.nn.functional.softmax(logits, dim=-1)",
            "entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)",
        ),
        (
            "if logits is None:",
            "    prob_dist = None",
            "    entropy = _unsloth_entropy",
            "else:",
            "    prob_dist = torch.nn.functional.softmax(logits, dim=-1)",
            "    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)",
        ),
    ),
)


def ppo_trainer_fused_logprobs(function_name, function):
    """Score PPO's rollout, reference and training log-probs (and the entropy metric) from hidden states with the
    fused linear kernel. Every anchor must match exactly once, else TRL's source is returned untouched; at run time
    ``ppo_route`` / ``ppo_score`` keep TRL's lines for any head or forward the kernel cannot serve."""
    if function_name != "train" or "_unsloth_ppo_route" in function:
        return function
    new = function
    for old_lines, new_lines in _PPO_FUSED_BLOCKS:
        new = _unsloth_replace_block(new, old_lines, new_lines)
        if new is None:
            return function
    return new


RL_FUNCTIONS["ppo_trainer"].append(ppo_trainer_fused_logprobs)


_RLOO_LOGPS_DEF = re.compile(
    r"^(?P<indent>[ \t]*)def _get_per_token_logps_and_entropies\(", re.MULTILINE
)


def rloo_trainer_fused_logprobs(function_name, function):
    """RLOOTrainer (TRL < 1.15) reads ``model(**inputs).logits``: keep its body as
    ``_unsloth_stock_get_per_token_logps_and_entropies`` and put a dispatcher in front that scores text-only batches
    from hidden states. TRL >= 1.15 calls its own fused lm_head, already served by the zoo kernel."""
    if function_name != "_get_per_token_logps_and_entropies":
        return function
    if "fused_lm_head=True" in function or "_unsloth_rloo" in function:
        return function
    match = list(_RLOO_LOGPS_DEF.finditer(function))
    if len(match) != 1 or ".logits" not in function or "self.temperature" not in function:
        return function
    indent = match[0].group("indent")
    try:
        from unsloth.models._fused_rl_logprobs import rloo_stock_traits
    except Exception:
        return function
    traits = rloo_stock_traits(function)
    stock = _RLOO_LOGPS_DEF.sub(
        lambda m: m.group("indent") + "def _unsloth_stock_get_per_token_logps_and_entropies(",
        function,
        count = 1,
    )
    body = [
        "def _get_per_token_logps_and_entropies(",
        "    self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False,",
        "    *args, **kwargs,",
        "):",
        "    try:",
        "        from unsloth.models._fused_rl_logprobs import rloo_fused_logps as _unsloth_rloo_fused_logps",
        "    except Exception:",
        "        _unsloth_rloo_fused_logps = None",
        "    if _unsloth_rloo_fused_logps is not None:",
        "        _out = _unsloth_rloo_fused_logps(",
        f"            self, {traits!r}, model, input_ids, attention_mask, logits_to_keep, batch_size,",
        "            compute_entropy, args, kwargs,",
        "        )",
        "        if _out is not None:",
        "            return _out",
        "    return self._unsloth_stock_get_per_token_logps_and_entropies(",
        "        model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy, *args, **kwargs",
        "    )",
    ]
    dispatcher = "\n".join(indent + line for line in body) + "\n\n"
    return dispatcher + stock


RL_FUNCTIONS["rloo_trainer"].append(rloo_trainer_fused_logprobs)


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


def _insert_after_docstring(function, lines):
    """Put ``lines`` (body-relative) first in the method body, after any docstring."""
    import ast

    try:
        node = ast.parse(textwrap.dedent(function)).body[0]
    except SyntaxError:
        return function
    if not isinstance(node, ast.FunctionDef) or not node.body:
        return function
    first = node.body[0]
    has_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    )
    source_lines = function.split("\n")
    anchor = node.body[1] if has_docstring and len(node.body) > 1 else first
    indent = re.match(r"[ \t]*", source_lines[anchor.lineno - 1]).group(0)
    insert_at = first.end_lineno if has_docstring else first.lineno - 1
    block = [indent + line for line in lines]
    return "\n".join(source_lines[:insert_at] + block + source_lines[insert_at:])


def _fused_preference_dispatch(function_name, function, call, signature):
    if function_name != "concatenated_forward" or signature not in function:
        return function
    if os.environ.get("UNSLOTH_FUSED_PREFERENCE_LOSS", "0") != "1":
        return function
    if "_unsloth_fused_output" in function:
        return function
    return _insert_after_docstring(
        function,
        [
            "from unsloth.models.dpo import " + call.split("(")[0],
            "_unsloth_fused_output = " + call,
            "if _unsloth_fused_output is not None:",
            "    return _unsloth_fused_output",
        ],
    )


def dpo_trainer_fused_preference_forward(function_name, function):
    return _fused_preference_dispatch(
        function_name,
        function,
        "unsloth_fused_dpo_concatenated_forward(self, model, batch, is_ref_model)",
        "is_ref_model",
    )


def orpo_trainer_fused_preference_forward(function_name, function):
    return _fused_preference_dispatch(
        function_name,
        function,
        'unsloth_fused_orpo_concatenated_forward(self, model, batch, method = "orpo")',
        "batch",
    )


def cpo_trainer_fused_preference_forward(function_name, function):
    return _fused_preference_dispatch(
        function_name,
        function,
        'unsloth_fused_orpo_concatenated_forward(self, model, batch, method = "cpo")',
        "batch",
    )


# The replacement reproduces exactly these metric keys and logits reads; any other source keeps TRL's
# function (0.29.x truncates inside it, 1.15+ already uses TRL's fused lm head).
_DPO_V1_METRIC_KEYS = frozenset(
    (
        "aux_loss",
        "entropy",
        "num_tokens",
        "logits/chosen",
        "logits/rejected",
        "mean_token_accuracy",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
    )
)
# Each entry: the spellings (whitespace-insensitive) one of which must appear.
_DPO_V1_SOURCE_MARKERS = (
    (
        "per_token_logps = selective_log_softmax(shift_logits, shift_labels)",
        "per_token_logps, per_token_entropies = selective_log_softmax_and_entropy( shift_logits, shift_labels,",
    ),
    (
        "ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)",
        "ref_per_token_logps = selective_log_softmax( ref_shift_logits, shift_labels, row_mask=shift_completion_mask )",
    ),
    (
        "per_token_entropy = entropy_from_logits(shift_logits.detach())",
        "per_token_entropy = per_token_entropies",
    ),
    ("predictions = chosen_logits.argmax(dim=-1)",),
)
# Sources the replacement does not reproduce.
_DPO_V1_DECLINE_MARKERS = ("self._truncate_inputs(", "fused_lm_head=True")
# Releases before 1.10 drop the constants of the forward_kl / js f-divergence scores and feed
# those scores (not the log ratios) to nca_pair / bco_pair / sppo_hard / apo_down.
_DPO_V1_FDIV_SCORES_MARKER = "chosen_scores = 1 - torch.exp(-chosen_logratios)"
_DPO_V1_FDIV_LEGACY_MARKER = "chosen_scores = -torch.exp(-chosen_logratios)"
# Releases before 1.3 log the entropy as a mean of the masked per-token entropies.
_DPO_V1_ENTROPY_MEAN_MARKER = "entropy = per_token_entropy[shift_completion_mask.bool()].mean()"


def _dpo_v1_semantics(function):
    """The TRL-release differences the fused _compute_loss must follow, read from its source, or
    None when the source is not one the replacement reproduces."""
    flat = " ".join(function.split())
    if any(marker in flat for marker in _DPO_V1_DECLINE_MARKERS):
        return None
    for spellings in _DPO_V1_SOURCE_MARKERS:
        if not any(" ".join(spelling.split()) in flat for spelling in spellings):
            return None
    if _DPO_V1_FDIV_SCORES_MARKER in flat:
        legacy_fdiv = False
    elif _DPO_V1_FDIV_LEGACY_MARKER in flat:
        legacy_fdiv = True
    else:
        return None
    return {
        "legacy_fdiv": legacy_fdiv,
        "entropy_mean": _DPO_V1_ENTROPY_MEAN_MARKER in flat,
        "tp_size": "// self._tp_size" in flat,
    }


def dpo_trainer_fused_preference_compute_loss(function_name, function):
    if function_name != "_compute_loss":
        return function
    if os.environ.get("UNSLOTH_FUSED_PREFERENCE_LOSS", "0") != "1":
        return function
    if "_unsloth_fused_output" in function:
        return function
    if not re.search(r"def _compute_loss\(self, model, inputs, return_outputs\)", function):
        return function
    semantics = _dpo_v1_semantics(function)
    if semantics is None:
        return function
    keys = set(re.findall(r'self\._metrics\[mode\]\["([^"]+)"\]', function))
    if not keys or not keys.issubset(_DPO_V1_METRIC_KEYS):
        return function
    return _insert_after_docstring(
        function,
        [
            "from unsloth.models.dpo import unsloth_fused_dpo_compute_loss_v1",
            "_unsloth_fused_output = unsloth_fused_dpo_compute_loss_v1("
            f"self, model, inputs, return_outputs, trl_semantics = {semantics!r})",
            "if _unsloth_fused_output is not None:",
            "    return _unsloth_fused_output",
        ],
    )


RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_fused_preference_forward)
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_fused_preference_compute_loss)
RL_FUNCTIONS["orpo_trainer"].append(orpo_trainer_fused_preference_forward)
RL_FUNCTIONS["cpo_trainer"].append(cpo_trainer_fused_preference_forward)


# Unsloth's training forward drops the 2D mask, so right-align Online DPO rows (GRPO's left_pack_padding) before scoring.
_ONLINE_DPO_MODEL_CALL = re.compile(
    r"^(?P<indent>[ \t]*)output = model\(prompt_completion_ids, "
    r"(?P<kwargs>attention_mask=prompt_completion_mask|\*\*model_kwargs)\)[ \t]*$",
    flags = re.MULTILINE,
)
# TRL 0.18-0.19 slice from `prompt_ids.size(1) - 1`, TRL 0.20+ name it start_idx.
_ONLINE_DPO_LOGITS_SLICE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<line>logits = output\.logits\[:, "
    r"(?P<start>start_idx|prompt_ids\.size\(1\) - 1) ?: ?(?:end_idx|-1)\])[ \t]*$",
    flags = re.MULTILINE,
)


def online_dpo_trainer__forward(function_name, function):
    if function_name != "_forward" or "_unsloth_left_pad" in function:
        return function
    call = _ONLINE_DPO_MODEL_CALL.search(function)
    logits_slice = _ONLINE_DPO_LOGITS_SLICE.search(function)
    if call is None or logits_slice is None:
        _warn_once(
            "online_dpo_trainer._forward",
            "Unsloth: Online DPO's _forward changed upstream, so left-padded prompts are scored "
            "without right-aligning them. Please file a bug report.",
        )
        return function
    i = call.group("indent")
    # Vision rows keep TRL's layout: image tokens are placed by position.
    has_vision = call.group("kwargs") == "**model_kwargs"
    pack = (
        f"{i}_unsloth_left_pad = None\n"
        f"{i}if {'not vision_inputs' if has_vision else 'True'}:\n"
        f"{i}    _unsloth_left_pad = (prompt_mask == 0).sum(dim = 1)\n"
        f"{i}    _unsloth_order = torch.argsort(prompt_completion_mask != 0, dim = 1, descending = True, stable = True)\n"
        f"{i}    prompt_completion_ids = prompt_completion_ids.gather(1, _unsloth_order)\n"
        f"{i}    prompt_completion_mask = prompt_completion_mask.gather(1, _unsloth_order)\n"
        + (
            f'{i}    model_kwargs["attention_mask"] = prompt_completion_mask\n'
            if has_vision
            else ""
        )
        + call.group(0)
    )
    j = logits_slice.group("indent")
    gather = (
        f"{j}if _unsloth_left_pad is not None:\n"
        f"{j}    _unsloth_index = ({logits_slice.group('start')} - _unsloth_left_pad).unsqueeze(1) + torch.arange(\n"
        f"{j}        completion_ids.size(1), device = completion_ids.device\n"
        f"{j}    ).unsqueeze(0)\n"
        f"{j}    _unsloth_index = _unsloth_index.clamp(0, output.logits.size(1) - 1)\n"
        # index_select: advanced indexing is ~40% slower, take_along_dim keeps a [B, C, V] int64 index.
        f"{j}    _unsloth_rows, _unsloth_len = output.logits.shape[:2]\n"
        f"{j}    _unsloth_index = _unsloth_index + _unsloth_len * torch.arange(\n"
        f"{j}        _unsloth_rows, device = _unsloth_index.device\n"
        f"{j}    ).unsqueeze(1)\n"
        f"{j}    logits = output.logits.reshape(_unsloth_rows * _unsloth_len, -1).index_select(\n"
        f"{j}        0, _unsloth_index.reshape(-1)\n"
        f"{j}    ).view(_unsloth_rows, -1, output.logits.size(-1))\n"
        # Drop [B, L, V] before log_softmax, else the copy above raises peak memory.
        f"{j}    output = None\n"
        f"{j}else:\n"
        f"{j}    {logits_slice.group('line')}"
    )
    function = function[: logits_slice.start()] + gather + function[logits_slice.end() :]
    return function[: call.start()] + pack + function[call.end() :]


RL_FUNCTIONS["online_dpo_trainer"].append(online_dpo_trainer__forward)


_WRAPPED_PACKING_SETUP = (
    "    import inspect as _inspect\n"
    "    try:\n"
    '        _unsloth_pack_has_strategy = "strategy" in _inspect.signature(pack_dataset).parameters\n'
    "    except Exception:\n"
    "        _unsloth_pack_has_strategy = True\n"
    "    _unsloth_wrapped_packing = packing and (\n"
    # bfd_split (TRL >= 1.0) and its TRL 0.29 name bfd-requeue split overflow into later rows, so early truncation drops it.
    '        getattr(args, "packing_strategy", None) in ("wrapped", "bfd_split", "bfd-requeue")\n'
    "        or not _unsloth_pack_has_strategy\n"
    "    )\n"
)

_WARNED_MISSING_ANCHORS = set()


def _warn_once(where, message):
    if where in _WARNED_MISSING_ANCHORS:
        return
    _WARNED_MISSING_ANCHORS.add(where)
    logger.warning(message)


def _comment_tolerant_anchor(old):
    """`old` as a regex that still matches once a line grows a trailing comment.

    A literal anchor stops matching the moment the Zoo appends a comment to one of
    its lines, and this repo edits comments in bulk: unsloth_zoo #1192 put a
    `# noqa: F821` on the `pack_dataset(` call while building a lint gate, which
    was enough to drop the whole patched SFT trainer. Nothing about the code
    changed, so nothing about the match should have.

    Every line is allowed an optional trailing comment, including lines that
    already carry one, so a comment being added, reworded or removed upstream is
    drift this survives.
    """
    return re.compile(
        "\n".join(re.escape(line.rstrip()) + r"[ \t]*(?:\#[^\n]*)?" for line in old.split("\n"))
    )


def _require_replace(
    function,
    old,
    new,
    *,
    count = 1,
    required = True,
    where = "",
):
    """str.replace that never silently no-ops a load-bearing source edit. Plain str.replace returns the source unchanged when the anchor is absent, so a drifted anchor in a newer TRL / unsloth_zoo would skip the edit while later edits still reference helper variables it should have introduced (NameError at runtime). Fail loudly for a required edit, warn once and skip for an optional one.

    The literal anchor is tried first and a comment-tolerant form of the same
    anchor second, so the only drift that reaches the raise below is drift in the
    code itself. `re.sub` would read backslashes in `new` as group references, so
    the replacement is handed over as a function.
    """
    if old in function:
        return function.replace(old, new, count)

    function, applied = _comment_tolerant_anchor(old).subn(
        lambda _match: new,
        function,
        count = count,
    )
    if applied:
        return function

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


# The one line every unsloth_zoo sft_prepare_dataset ends its worker count on, as a regex so a renamed right-hand side still matches and the indentation carries over. Unchanged since Aug 2025 (#257) while the block around it was rewritten repeatedly, hence a fallback anchor.
_ZOO_MAP_NUM_PROC_ASSIGNMENT = re.compile(
    r"^(?P<indent>[ \t]*)map_kwargs\[\"num_proc\"\][ \t]*=[ \t]*[^\n]+$",
    flags = re.MULTILINE,
)

# The Zoo seeds its truncation length from args.max_length and falls through to args.max_seq_length only when that is 0; rl.py hands TRL >= 1.0.0 a None, so normalise it or nothing truncates.
_ZOO_MAX_LENGTH_SEED = re.compile(
    r"^(?P<indent>[ \t]*)max_seq_length[ \t]*=[ \t]*getattr\(args,[ \t]*[\"']max_length[\"'],[ \t]*0\)[ \t]*$",
    flags = re.MULTILINE,
)


def _same_source(text):
    """`text` with quote style normalised, for comparing two spellings of a line. The narrow regexes here already accept either quote, so the idempotence check has to as well: a Zoo carrying the replacement with single quotes matched neither the literal nor the `$`-anchored regex, and `required = True` then raised on every SFT trainer over behaviour already present."""
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

    One literal anchor is all-or-nothing, and neither outcome is acceptable here. `required = True` hard-fails every SFT run on a Zoo whose text merely moved. `required = False` leaves the un-rewritten Zoo logic in place, and for the worker count that is not the no-op it looks like: the config layer writes "run in-process" as `args.dataset_num_proc = 1` (a config `None` means "auto-size me" to the Zoo), the Zoo reads that `1` as an explicit count, and `datasets` >= 4.1 builds a `Pool(1)` for it, forking a tokenizer worker on exactly the host that asked for none.

    So: try the wide anchor, then a narrow one that survives more drift, and warn only when both miss. `fallback_new` is an `re.sub` template, so it can carry the matched indentation over with `\\g<indent>`. `required = True` keeps the two-anchor tolerance but raises rather than warns when BOTH miss; `consequence` says what that absence does, since the warning below speaks only about the worker count.
    """
    # Already done upstream, and checked FIRST: an edit is missing only if its RESULT is, and a Zoo that adopts the replacement is the forward case. Order matters: `old` is a prefix of `new` for the max_length seed, so the wide anchor matched the normalized line and appended a second `or 0`.
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
            # Anchor the wrapped-packing setup on the function signature, which always exists, not the unsloth_zoo license comment, which a newer Zoo may move: anchoring there let the setup silently no-op while later edits used its variables, NameError-ing every SFT dataset prep.
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
                # Not optional, unlike the worker count this helper was written for: the generated trainer clears args.max_length for padding-free, so an unrewritten seed reads None instead of the 0 that falls through to max_seq_length. Both anchors missing means the neighbouring _require_replace edits have gone too.
                required = True,
                consequence = ", so `max_length` would not be enforced for raw "
                "datasets under padding-free batching",
            )
            # Route each edit through _require_replace so a drifted anchor fails loudly instead of leaving a dangling reference to the setup variables.
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
            # Reuse the guarded _unsloth_pack_has_strategy from the setup rather than re-calling _inspect.signature(pack_dataset): the setup wraps that in try/except, so a non-introspectable pack_dataset must not crash here.
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
            # The map() call site, not the config, is where the worker count is made safe: the Zoo copy asks stdlib multiprocessing for a start method datasets takes from multiprocess, and its low-memory branch yields 1, still a Pool(1) on datasets >= 4.1. Imported from the Zoo so generated source never imports back into its generator.
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
                # The likeliest anchor here to drift, but its absence is not harmless: the config layer encodes serial as 1 for this site to turn back into None, and an un-rewritten Zoo hands that 1 to Dataset.map. The fallback keys on the block's closing assignment, unchanged since Aug 2025; hard-failing instead would break every install on a newer Zoo.
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
            # datasets never reads the child's exit status, so every worker death flattens into "One of the subprocesses has abruptly died during map operation". Wrap both tokenizing map() calls so the user gets the worker count, start method and implied memory.
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
                # required = False, like the selection above: this only improves a dead worker's message, and a diagnostic must not fail a run because a Zoo release moved its anchor. test_zoo_sft_prepare_dataset_anchor_has_not_drifted notices in CI instead.
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
    if function_name != "compute_loss" or "_unsloth_trl_compute_loss" in function:
        return function

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        # TRL >= 1.15 computes loss_type="dft" only in the compute_loss this replaces (else it trains as NLL):
        # run it on TRL's fused LM head. Older TRL routes DFT through compute_loss_func.
        if (
            getattr(self.args, "loss_type", None) == "dft"
            and self.compute_loss_func is None
            and getattr(self, "label_smoother", None) is None
        ):
            import trl.trainer.utils as _trl_utils
            if hasattr(_trl_utils, "add_fused_lm_head"):
                # TRL's fused head projects with lm_head.weight directly, skipping QAT fake quantization.
                try:
                    head = model.get_output_embeddings()
                except Exception:
                    head = None
                try:
                    from unsloth.models.llama import lm_head_needs_forward as _needs_forward
                except Exception:
                    _needs_forward = lambda module: False
                if head is not None and (
                    _needs_forward(head)
                    or any(
                        getattr(m, "activation_fake_quantizer", None) is not None
                        or getattr(m, "weight_fake_quantizer", None) is not None
                        for m in head.modules()
                    )
                ):
                    raise NotImplementedError(
                        "Unsloth: loss_type = 'dft' runs on TRL's fused LM head, which skips the QAT fake "
                        "quantization of lm_head. Leave lm_head out of QAT or use loss_type = 'nll'."
                    )
                inputs = {**inputs, "use_cache": False}
                outputs = model(**inputs, fused_lm_head = True)
                if getattr(outputs, "log_probs", None) is None:
                    raise RuntimeError(
                        "Unsloth: loss_type = 'dft' needs TRL's fused LM head, but the model returned no "
                        "per-token log-probabilities. Use loss_type = 'nll'."
                    )
                per_token_loss = -outputs.log_probs * outputs.log_probs.exp().detach()
                # Trainer divides by gradient accumulation itself when the model does not take num_items_in_batch.
                if not getattr(self, "model_accepts_loss_kwargs", True):
                    num_items_in_batch = None
                num_tokens = (
                    outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
                )
                loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min = 1)
                if self.args.average_tokens_across_devices and num_items_in_batch is not None:
                    # As TRL / Trainer: tensor parallel ranks share one batch, so they do not count.
                    tp_size = (
                        self.get_tp_size() if callable(getattr(self, "get_tp_size", None)) else 1
                    )
                    loss = loss * (self.accelerator.num_processes // max(int(tp_size or 1), 1))
                aux_loss = getattr(outputs, "aux_loss", None)
                if getattr(self, "aux_loss_enabled", False) and aux_loss is not None:
                    steps = (
                        self.current_gradient_accumulation_steps
                        if num_items_in_batch is not None
                        else 1
                    )
                    loss = loss + self.router_aux_loss_coef * aux_loss / steps
                if return_outputs:
                    # prediction_step / compute_metrics read logits, which the fused head never builds (as TRL does).
                    with torch.no_grad():
                        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                return (loss, outputs) if return_outputs else loss
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


# Route ORPO/CPO row tokenization through the underlying text tokenizer when the processing class is a multimodal processor; CPO reuses this code (#4952).
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


# TRL 0.29+ ORPO/CPO tokenize_row never truncates the prompt and cuts answers to `max_length - longer_response_length`, so prompt + answer can exceed max_length. The fast forward cuts input_ids to max_seq_length while labels keep the full row, which crashes the log-prob gather. Hold each row to max_length the way TRL <= 0.25 did with max_prompt_length (keep a prompt budget, then cut the answer); rows that fit are untouched.
_ORPO_ROW_CAP = (
    "_unsloth_ul = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))\n"
    "_unsloth_pl = max(len(chosen_tokens['prompt_input_ids']), len(rejected_tokens['prompt_input_ids']))\n"
    "if self.max_length is not None and _unsloth_pl + _unsloth_ul > self.max_length:\n"
    "    _unsloth_keep = min(_unsloth_pl, max(self.max_length - _unsloth_ul, self.max_length // 2))\n"
    "    _unsloth_keep_start = getattr(self, 'truncation_mode', 'keep_end') == 'keep_start'\n"
    "    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:\n"
    "        for k in ['prompt_input_ids', 'prompt_attention_mask']:\n"
    "            answer_tokens[k] = answer_tokens[k][:_unsloth_keep] if _unsloth_keep_start else answer_tokens[k][max(0, len(answer_tokens[k]) - _unsloth_keep):]\n"
    "    for answer_tokens in [chosen_tokens, rejected_tokens]:\n"
    "        for k in ['input_ids', 'attention_mask']:\n"
    "            answer_tokens[k] = answer_tokens[k][: self.max_length - _unsloth_keep]\n"
)


def orpo_trainer_row_cap(function_name, function):
    if (
        function_name != "tokenize_row"
        or "_unsloth_ul" in function
        or "max_prompt_length" in function
    ):
        return function
    # Before TRL's own response cut: its negative slice end can empty the shorter answer.
    match = re.search(r"(?m)^([ \t]*)longer_response_length = max\(", function)
    if match is None:
        # TRL 1.15+ cuts answers but never the prompt, so a long prompt still overflows max_length.
        match = re.search(
            r"(?m)^([ \t]*)for answer_tokens in \[chosen_tokens, rejected_tokens\]:\n"
            r"\1[ \t]+if len\(answer_tokens\[\"prompt_input_ids\"\]\) \+ len\(answer_tokens\[\"input_ids\"\]\) > self\.max_length:",
            function,
        )
    if match is None:
        return function
    indent = match.group(1)
    block = "".join(indent + line + "\n" for line in _ORPO_ROW_CAP.splitlines())
    return function[: match.start()] + block + function[match.start() :]


RL_FUNCTIONS["orpo_trainer"].append(orpo_trainer_row_cap)
RL_FUNCTIONS["cpo_trainer"].append(orpo_trainer_row_cap)


# Resolve processing_class.pad_token_id through the inner tokenizer when a multimodal processor is supplied: processors lack pad_token_id, so ORPO/CPOTrainer.__init__ raises AttributeError in the collator and padding_value.
_PAD_FALLBACK = (
    "(getattr(processing_class, 'pad_token_id', None) "
    "if getattr(processing_class, 'pad_token_id', None) is not None "
    "else getattr(getattr(processing_class, 'tokenizer', None), 'pad_token_id', None))"
)


def orpo_trainer_processor_pad_token(function_name, function):
    if function_name != "__init__":
        return function
    # Multimodal processors expose pad_token / eos_token on .tokenizer, not on themselves, and TRL 1.x CPO/ORPO __init__ defaults pad_token from eos_token before tokenizing. Older TRL lacks this block, so the sub is a no-op there.
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


# Fix the bare pop("push_to_hub_token") in the compiled SFT/IterativeSFT __init__: on transformers 5.0+ to_dict() no longer includes it, so a bare pop KeyErrors.
def sft_trainer_push_to_hub_token(function_name, function):
    if function_name != "__init__":
        return function
    return function.replace(
        'dict_args.pop("push_to_hub_token")', 'dict_args.pop("push_to_hub_token", None)'
    )


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_push_to_hub_token)


# assistant_only_loss: TRL raises for chat templates it does not know (all Unsloth ones); fall back to Zoo's train_on_responses_only masks.
_SFT_TRAINING_TEMPLATE = re.compile(
    r"^(?P<indent>[ \t]*)self\.chat_template = get_training_chat_template\(processing_class\)[ \t]*$",
    flags = re.MULTILINE,
)
_SFT_STOP_TOKEN_CHECK = re.compile(
    r"if args\.assistant_only_loss and not is_chat_template_stop_token_trained\("
)


def _zoo_reads_assistant_mask_fallback():
    # An older unsloth_zoo never reads the flag, so suppressing TRL's error there would train on every token.
    try:
        from unsloth_zoo.dataset_utils import sft_prepare_dataset
        return "_unsloth_assistant_mask_fallback" in inspect.getsource(sft_prepare_dataset)
    except Exception:
        return False


def sft_trainer_assistant_mask_fallback(function_name, function):
    if function_name != "__init__" or "_unsloth_assistant_mask_fallback" in function:
        return function
    if _SFT_TRAINING_TEMPLATE.search(function) is None or not _zoo_reads_assistant_mask_fallback():
        return function

    def _replace(match):
        i = match.group("indent")
        return (
            f"{i}try:\n"
            f"{i}    self.chat_template = get_training_chat_template(processing_class)\n"
            f"{i}except ValueError:\n"
            # Zoo's text preparation reads the flag; vision datasets and skip_prepare_dataset never reach it.
            f"{i}    if getattr(self, '_is_vision_dataset', False) or (getattr(args, 'dataset_kwargs', None) or {{}}).get('skip_prepare_dataset'): raise\n"
            f"{i}    self.chat_template = None\n"
            f"{i}    self._unsloth_assistant_mask_fallback = True\n"
            f"{i}    print('Unsloth: TRL has no training chat template for this tokenizer, so assistant_only_loss masks non-assistant tokens with train_on_responses_only markers.')"
        )

    function = _SFT_TRAINING_TEMPLATE.sub(_replace, function, count = 1)
    function = _SFT_STOP_TOKEN_CHECK.sub(
        "if args.assistant_only_loss and not getattr(self, '_unsloth_assistant_mask_fallback', False) and not is_chat_template_stop_token_trained(",
        function,
        count = 1,
    )
    return function


RL_FUNCTIONS["sft_trainer"].append(sft_trainer_assistant_mask_fallback)


def _unsloth_grpo_autocast_forced_float32(self):
    # Stamped by from_pretrained: UNSLOTH_FORCE_FLOAT32 is process wide, so a model loaded after this trainer was built would answer for it here.
    forced = getattr(getattr(self, "model", None), "_unsloth_forced_float32", None)
    if forced is None:
        forced = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    return forced


def _unsloth_grpo_autocast(self):
    """Decide the GRPO autocast once and latch it on the trainer. ACCELERATE_MIXED_PRECISION is process wide, so a trainer built later but run first would hand this trainer its precision; args belongs to this trainer."""
    if not hasattr(self, "_autocast_enabled"):
        args = getattr(self, "args", None)
        precision = getattr(args, "mixed_precision", None)
        use_bf16 = getattr(args, "bf16", None)
        use_fp16 = getattr(args, "fp16", None)
        if not isinstance(precision, str):
            # transformers < 5 has no args.mixed_precision, but rl.py sets the fp16 / bf16 flags on this same args for every branch it takes.
            if isinstance(use_bf16, bool) and isinstance(use_fp16, bool):
                precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "no")
            else:
                precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16")
        self._autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        # "no" is a real value: full finetuning and an explicit float32 load both set it, and reading it as bfloat16 raises on a T4 or V100.
        self._autocast_enabled = precision != "no"
        self._autocast_force_float32 = False
        forced = _unsloth_grpo_autocast_forced_float32(self)
        if forced and precision != "bf16":
            # Gemma3 / gpt-oss set "no" but still want float16 autocast; a trainer already on bf16 keeps it, since float16 is what the forced list avoids.
            self._autocast_dtype = torch.float16
            self._autocast_enabled = True
            self._autocast_force_float32 = True

    return self._autocast_enabled, self._autocast_dtype


def _unsloth_grpo_autocast_unscored(self):
    """Latch for a step that ran no scoring forward. Where the ordinary latch would force float16 autocast on a
    forced float32 model (Gemma3 on a T4) without vLLM, record no loss autocast instead: nothing latched there
    before the TRL old-logprob gate came back, so zoo trained it without autocast. A forced-family trainer on
    bf16 (full finetuning) keeps its bf16 latch. Recorded on the trainer rather than left to zoo's
    process-wide fallback, which a later model or trainer may have changed. Under vLLM the old-policy
    forward always ran and latched, so it latches as before."""
    fresh = not hasattr(self, "_autocast_enabled")
    latched = _unsloth_grpo_autocast(self)
    if (
        fresh
        and not getattr(self, "use_vllm", False)
        and getattr(self, "_autocast_force_float32", False)
    ):
        self._autocast_dtype = torch.float16
        self._autocast_enabled = False
        latched = (self._autocast_enabled, self._autocast_dtype)
    return latched


def _unsloth_grpo_autocast_kwargs(self, device_type = DEVICE_TYPE_TORCH):
    """torch.amp.autocast kwargs for GRPO generation."""
    enabled, dtype = _unsloth_grpo_autocast(self)
    if not getattr(self, "_autocast_force_float32", False) and torch.is_autocast_enabled(
        device_type
    ):
        # Already inside an autocast: inherit its dtype by omitting the key, since autocast passes whatever it gets to set_autocast_dtype.
        return {"enabled": enabled}
    return {"enabled": enabled, "dtype": dtype}


def _unsloth_grpo_refuse_unscored_support(self):
    """With compute_efficient no old / ref logps are scored here, so the gate below in
    _get_per_token_logps_and_entropies never runs, and the zoo loss scores the current policy over
    its own kept set against vLLM's logprobs. When that set cannot be the sampler's (an unsynced
    vLLM sampler, a reference model, reused rollouts), mark the trainer refused, which the zoo loss
    honours by scoring the full vocabulary."""
    if getattr(self, "_unsloth_truncated_support_refused", False):
        return
    try:
        from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
        if not grpo_truncated_support_kwargs(self):
            return
    except Exception:
        return
    mismatch = _unsloth_grpo_support_mismatch(self)
    if mismatch:
        import warnings
        warnings.warn(
            "Unsloth: not scoring over the rollouts' top_k / top_p / min_p kept set "
            "(UNSLOTH_GRPO_TRUNCATED_SUPPORT=1, or automatic for truncated vLLM rollouts "
            "feeding the IS correction or off-policy mask) because "
            f"{', '.join(mismatch)}: the scored policy is not the sampler, so its top_k / top_p / "
            "min_p kept set differs from vLLM's and the PPO ratio would be unnormalised. "
            "Scoring over the full vocabulary instead.",
            stacklevel = 2,
        )
        self._unsloth_truncated_support_refused = True


def _unsloth_grpo_support_mismatch(self):
    """Why kept-set scoring (UNSLOTH_GRPO_TRUNCATED_SUPPORT) cannot score this trainer, or [] when it can.

    Each scoring call rebuilds the top_k / top_p / min_p kept set from its own logits and always keeps the
    scored token. That is the sampler's set only while the scored weights are the rollout's. A reference
    model (beta != 0), or old logps TRL keeps because generation is not aligned to the optimizer step
    (gradient_accumulation_steps % (steps_per_generation * num_iterations) != 0), keeps another set: a
    token outside it gets a denominator that depends on the scored token, so the implied probabilities do
    not sum to 1 and the PPO ratio / KL inputs are wrong. A shared set would need the sampler's kept ids.
    """
    reasons = []
    if getattr(self, "_unsloth_vllm_sync_skip", False):
        reasons.append("vLLM weight sync is skipped, so rollouts use the initial weights")
    beta = getattr(self, "beta", 0.0)
    if beta is not None and float(beta) != 0.0:
        reasons.append(f"beta={beta} scores a reference model")
    args = getattr(self, "args", None)
    accumulation = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    steps_per_generation = int(getattr(args, "steps_per_generation", None) or accumulation)
    num_iterations = getattr(self, "num_iterations", None)
    if num_iterations is None:
        num_iterations = getattr(args, "num_iterations", 1)
    num_iterations = int(num_iterations or 1)
    if accumulation % (steps_per_generation * num_iterations) != 0:
        reasons.append(
            f"num_iterations={num_iterations} / steps_per_generation={steps_per_generation} "
            f"reuse rollouts across optimizer steps"
        )
    return reasons


def _unsloth_grpo_accumulation_steps(trainer):
    """Gradient accumulation divisor for the GRPO loss. 1 whenever the model is not training.

    TRL divides by `current_gradient_accumulation_steps` in train mode and by 1.0 in eval, since
    an eval pass accumulates nothing. Trainer sets that attribute inside the training loop and
    never clears it, so an in-training evaluation still sees the training window's size and
    eval_loss comes back that many times too small. Read off `model.training` like TRL does
    rather than a trainer flag, because that is what the eval loop actually switches.
    The attribute is also missing when evaluate() runs standalone (#2464), hence the default.
    """
    model = getattr(trainer, "model", None)
    if model is not None and not getattr(model, "training", True):
        return 1
    return getattr(trainer, "current_gradient_accumulation_steps", 1)


def _unsloth_grpo_vision_inputs(source):
    """unsloth_zoo owns this key tuple; the copy below is the fallback for a zoo predating
    GRPO_VISION_KEYS, and test_grpo_vision_kwargs_forwarded.py fails if the two diverge."""
    try:
        from unsloth_zoo.rl_replacements import grpo_get_vision_inputs
        return grpo_get_vision_inputs(source)
    except Exception:
        pass
    if source is None:
        return {}
    get = getattr(source, "get", None)
    if get is None:
        return {}
    return {
        key: get(key, None)
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "spatial_shapes",
            "num_tiles",
            # both Gemma 4 spellings: image_position_ids is TRL >= 1.1.0
            "image_position_ids",
            "pixel_position_ids",
            "num_images",
            "token_type_ids",
            "mm_token_type_ids",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "num_videos",
        )
    }


def _unsloth_grpo_split_vision_by_sample(batch):
    """TRL only splits the tile and image indexed vision tensors per sample from its own
    split_pixel_values_by_grid, and before TRL 1.1.0 that function knows only the
    image_grid_thw layout. Every other layout is left flat, and _prepare_inputs then
    shuffles and slices the batch by sample index, so an LFM2-VL or Gemma batch has its
    tiles reordered away from the samples they belong to. This mirrors the current
    trl.trainer.utils.split_pixel_values_by_grid so an older TRL keeps them together."""

    def _counts(value):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            return [int(count) for count in value]
        except TypeError:
            return None

    batch = _unsloth_grpo_split_videos_by_sample(batch)
    pixel_values = batch.get("pixel_values", None)
    if pixel_values is None:
        return batch
    image_grid_thw = batch.get("image_grid_thw", None)
    num_images = _counts(batch.get("num_images", None))

    if isinstance(pixel_values, list):
        # TRL 1.1.0+ splits by sample; 0.22.x-0.23.x splits by grid row and leaves image_grid_thw flat, so
        # a two-image row is mis-shuffled. Regroup it. Not len(pixel_values) == len(num_images): [0, 2]
        # makes them agree while the axes differ.
        _prompt_ids = batch.get("prompt_ids", None)
        if (
            image_grid_thw is None
            or isinstance(image_grid_thw, list)
            or not num_images
            or not pixel_values
            or len(pixel_values) != sum(num_images)
            or all(_count == 1 for _count in num_images)
            or (_prompt_ids is not None and len(num_images) != _prompt_ids.shape[0])
        ):
            return batch
        split = dict(batch)
        _empty = pixel_values[0][:0]
        _grouped = []
        _offset = 0
        for _count in num_images:
            _group = pixel_values[_offset : _offset + _count]
            _offset += _count
            _grouped.append(torch.cat(_group, dim = 0) if _group else _empty)
        split["pixel_values"] = _grouped
        split["image_grid_thw"] = list(torch.split(image_grid_thw, num_images, dim = 0))
        for _image_key in ("pixel_attention_mask", "image_sizes"):
            _per_image = batch.get(_image_key, None)
            if (
                _per_image is not None
                and not isinstance(_per_image, list)
                and _per_image.shape[0] == sum(num_images)
                and _per_image.shape[0] != len(num_images)
            ):
                split[_image_key] = list(torch.split(_per_image, num_images, dim = 0))
        return split

    if image_grid_thw is not None:
        # An unsplit grid batch: TRL owns this layout in every version that persists it.
        return batch
    if not num_images:
        return batch
    rows = pixel_values.shape[0]
    num_tiles = _counts(batch.get("num_tiles", None))
    split = dict(batch)
    for _position_key in ("image_position_ids", "pixel_position_ids"):
        _position_ids = batch.get(_position_key, None)
        if _position_ids is None or isinstance(_position_ids, list):
            continue
        if rows != sum(num_images) or _position_ids.shape[0] != sum(num_images):
            continue
        split["pixel_values"] = list(torch.split(pixel_values, num_images, dim = 0))
        split[_position_key] = list(torch.split(_position_ids, num_images, dim = 0))
        return split
    if num_tiles and rows == sum(num_tiles):
        split["pixel_values"] = list(torch.split(pixel_values, num_tiles, dim = 0))
        for _tile_key in ("pixel_attention_mask", "spatial_shapes"):
            _tiled = batch.get(_tile_key, None)
            if (
                _tiled is not None
                and not isinstance(_tiled, list)
                and _tiled.shape[0] == sum(num_tiles)
            ):
                split[_tile_key] = list(torch.split(_tiled, num_tiles, dim = 0))
        return split
    if rows != sum(num_images):
        # One padded row per sample already (Idefics, SmolVLM): TRL leaves this alone.
        return batch
    if rows == len(num_images) and any(_count != 1 for _count in num_images):
        # Padded-sample and flat-image axes are ambiguous: keep TRL layout, else [2, 0] misassigns images
        return batch
    split["pixel_values"] = list(torch.split(pixel_values, num_images, dim = 0))
    _image_sizes = batch.get("image_sizes", None)
    if (
        _image_sizes is not None
        and not isinstance(_image_sizes, list)
        and _image_sizes.shape[0] == sum(num_images)
    ):
        split["image_sizes"] = list(torch.split(_image_sizes, num_images, dim = 0))
    return split


def _unsloth_grpo_split_videos_by_sample(batch):
    """Group the flat video rows, grid and second_per_grid_ts by sample (num_videos) so the
    _prepare_inputs shuffle keeps them with their sample; _unsloth_grpo_unsplit_vision merges back."""
    pixel_values_videos = batch.get("pixel_values_videos", None)
    video_grid_thw = batch.get("video_grid_thw", None)
    num_videos = batch.get("num_videos", None)
    if (
        pixel_values_videos is None
        or video_grid_thw is None
        or num_videos is None
        or isinstance(pixel_values_videos, list)
    ):
        return batch
    num_videos = [int(n) for n in num_videos]
    if sum(num_videos) != video_grid_thw.shape[0]:
        return batch
    rows_per_sample = [
        int(group.prod(dim = -1).sum()) for group in torch.split(video_grid_thw, num_videos)
    ]
    split = dict(batch)
    split["pixel_values_videos"] = list(torch.split(pixel_values_videos, rows_per_sample))
    split["video_grid_thw"] = list(torch.split(video_grid_thw, num_videos))
    second_per_grid_ts = batch.get("second_per_grid_ts", None)
    if second_per_grid_ts is not None:
        second_per_grid_ts = torch.as_tensor(second_per_grid_ts)
        split["second_per_grid_ts"] = list(torch.split(second_per_grid_ts, num_videos))
    return split


def _unsloth_grpo_prompt_videos(prompt, kind = "video"):
    """Number of `kind` parts in a conversational prompt."""
    if not isinstance(prompt, list):
        return 0
    count = 0
    for message in prompt:
        content = message.get("content", None) if isinstance(message, dict) else None
        if isinstance(content, list):
            count += sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == kind
            )
    return count


def _unsloth_grpo_has_video_key(prompt):
    if not isinstance(prompt, list):
        return False
    for message in prompt:
        content = message.get("content", None) if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and "video" in part for part in content
        ):
            return True
    return False


def _unsloth_grpo_clean_video_prompts(prompts, trainer = None):
    """Arrow adds `video: None` to every content part of a column holding videos; Qwen2-VL
    templates test `'video' in content` and render a placeholder per part (StopIteration)."""
    if not any(_unsloth_grpo_has_video_key(prompt) for prompt in prompts):
        return prompts
    if getattr(trainer, "use_vllm", False) and any(
        _unsloth_grpo_prompt_videos(prompt) for prompt in prompts
    ):
        # Before generation, which vLLM would otherwise run without the videos.
        raise NotImplementedError(
            "Unsloth: video GRPO runs on the transformers generation path only. "
            "Load the model with fast_inference = False (use_vllm = False)."
        )
    cleaned = []
    for prompt in prompts:
        if not _unsloth_grpo_has_video_key(prompt):
            cleaned.append(prompt)
            continue
        messages = []
        for message in prompt:
            content = message.get("content", None)
            if isinstance(content, list):
                message = dict(message)
                message["content"] = [
                    {k: v for k, v in part.items() if v is not None}
                    if isinstance(part, dict)
                    else part
                    for part in content
                ]
            messages.append(message)
        cleaned.append(messages)
    return cleaned


def _unsloth_grpo_prompt_key(value):
    """Dedup key: equal for the repeated copies of a prompt, never shared by different media.
    repr() truncates arrays, so in-memory frames are keyed by identity, not by content."""
    if isinstance(value, dict):
        return tuple((k, _unsloth_grpo_prompt_key(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_unsloth_grpo_prompt_key(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return ("id", id(value))


def _unsloth_grpo_video_inputs(
    trainer,
    prompts,
    images = None,
):
    """Video tensors for the logprob forwards, which TRL builds from images only. Rendered with
    the same chat template call generation makes, once per distinct prompt. None without video."""
    num_videos = [_unsloth_grpo_prompt_videos(prompt) for prompt in prompts]
    if not any(num_videos):
        return None
    if getattr(trainer, "use_vllm", False):
        raise NotImplementedError(
            "Unsloth: video GRPO runs on the transformers generation path only. "
            "Load the model with fast_inference = False (use_vllm = False)."
        )
    if (images is not None and any(images)) or any(
        _unsloth_grpo_prompt_videos(prompt, "image") for prompt in prompts
    ):
        raise NotImplementedError(
            "Unsloth: GRPO does not support image and video inputs in the same batch yet."
        )
    try:
        from unsloth_zoo.rl_replacements import GRPO_VISION_KEYS
    except Exception:
        GRPO_VISION_KEYS = ()
    if "pixel_values_videos" not in GRPO_VISION_KEYS:
        raise RuntimeError(
            "Unsloth: video GRPO needs a newer unsloth_zoo that forwards video inputs to "
            "both GRPO logprob passes. Please upgrade: pip install -U unsloth_zoo"
        )
    unique, order = {}, []
    for prompt in prompts:
        order.append(unique.setdefault(_unsloth_grpo_prompt_key(prompt), len(unique)))
    first = {}
    for i, u in enumerate(order):
        first.setdefault(u, i)
    unique_prompts = [prompts[first[u]] for u in range(len(unique))]
    template_kwargs = dict(getattr(trainer, "chat_template_kwargs", None) or {})
    if getattr(trainer, "chat_template", None) is not None:
        # GRPOConfig.chat_template, which generation renders with too
        template_kwargs["chat_template"] = trainer.chat_template
    processed = trainer.processing_class.apply_chat_template(
        conversation = unique_prompts,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        padding = True,
        return_tensors = "pt",
        **template_kwargs,
    )
    pixel_values_videos = processed.get("pixel_values_videos", None)
    video_grid_thw = processed.get("video_grid_thw", None)
    if pixel_values_videos is None or video_grid_thw is None:
        raise NotImplementedError(
            f"Unsloth: video GRPO supports processors that return pixel_values_videos and "
            f"video_grid_thw (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5); "
            f"{type(trainer.processing_class).__name__} does not."
        )
    second_per_grid_ts = processed.get("second_per_grid_ts", None)
    if second_per_grid_ts is not None:
        second_per_grid_ts = torch.as_tensor(second_per_grid_ts)
    unique_counts = [num_videos[first[u]] for u in range(len(unique))]
    video_index = torch.split(torch.arange(video_grid_thw.shape[0]), unique_counts)
    row_ends = video_grid_thw.prod(dim = -1).cumsum(0)
    row_starts = row_ends - video_grid_thw.prod(dim = -1)
    videos, rows = [], []
    for u in order:
        for v in video_index[u].tolist():
            videos.append(v)
            rows.append(pixel_values_videos[int(row_starts[v]) : int(row_ends[v])])
    device = trainer.accelerator.device
    out = {
        "pixel_values_videos": torch.cat(rows, dim = 0).to(device),
        "video_grid_thw": video_grid_thw[videos].to(device),
        "num_videos": num_videos,
    }
    if second_per_grid_ts is not None:
        out["second_per_grid_ts"] = second_per_grid_ts[videos].to(device)
    return out


def _unsloth_grpo_unsplit_vision(batch):
    """Undo _unsloth_grpo_split_vision_by_sample once this step's slice has been taken, so
    the forward sees the layout the processor produced. TRL's own unsplit merges only
    pixel_values before 1.1.0, and pixel_values plus the grid and position ids from 1.1.0."""
    merged = None
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_attention_mask",
        "spatial_shapes",
        "image_sizes",
        "image_position_ids",
        "pixel_position_ids",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        value = batch.get(key, None)
        if not isinstance(value, list) or len(value) == 0:
            continue
        if not hasattr(value[0], "shape"):
            # num_images and num_tiles are plain counts, not tensors to merge.
            continue
        if merged is None:
            merged = dict(batch)
        merged[key] = torch.cat(value, dim = 0)
    return batch if merged is None else merged


# A list cell reaches the processor as [[[img, img]]], which it rejects. unsloth#3605.
def _unsloth_grpo_image_cell(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _unsloth_reject_grpo_image_list(inputs, trainer = None):
    """Refuse it where the rewrite above missed TRL's spelling: the processor's own error
    names neither the column nor the fix.

    `trainer` narrows the refusal to the one runtime mode a legacy TRL cannot carry a multi
    image row through. Its vLLM server path sends the raw cells to `VLLMClient.generate`,
    which does `[pil_to_base64(img) for img in images]` over the top level entries, so a cell
    holding two images reaches `list.save(...)` and dies with an AttributeError naming neither.
    Colocate mode and the no vLLM path both go through the processor, which this change fixed,
    so they keep working and must not be refused."""
    if trainer is not None:
        if not getattr(trainer, "use_vllm", False):
            return
        if getattr(trainer, "vllm_mode", None) != "server":
            return
    # Every row: one list cell anywhere desyncs the images and the placeholders.
    try:
        rows = list(inputs)
    except Exception:
        return
    for _row_index, _row in enumerate(rows):
        value = _row.get("image", None) if isinstance(_row, dict) else None
        if isinstance(value, (list, tuple)) and len(value) > 1:
            raise ValueError(
                f"Unsloth: GRPO received a singular `image` column holding {len(value)} "
                f"images in row {_row_index}, and this TRL version cannot carry more than "
                "one image per row through to the model. "
                "Rename the column to `images`, which TRL reads as the per example list of "
                "images, or keep one image per row. "
                "See https://github.com/unslothai/unsloth/issues/3605"
            )


def grpo_trainer__prepare_inputs(function_name, function):
    if function_name != "_prepare_inputs":
        return function

    # Latched on the trainer, so a second trainer's __init__ cannot change this trainer's autocast mid run.
    function = function.replace(
        "with torch.inference_mode():",
        "with torch.inference_mode(), "
        "torch.amp.autocast(device_type = DEVICE_TYPE_TORCH, "
        "**_unsloth_grpo_autocast_kwargs(self)):",
    )
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )

    # Keep every tile with its own sample across the shuffle and the slice. unsloth#6960.
    _split_anchor = re.search(
        r"\n([ \t]+)generation_batch = split_pixel_values_by_grid\(generation_batch\)\n",
        function,
    )
    _unsplit_anchor = "unsplit_pixel_values_by_grid(batch) for batch in generation_batches"
    if _split_anchor is not None and _unsplit_anchor in function:
        _spacing = _split_anchor.group(1)
        function = function.replace(
            _split_anchor.group(0),
            _split_anchor.group(0)
            + f"{_spacing}generation_batch = _unsloth_grpo_split_vision_by_sample(generation_batch)\n",
        )
        function = function.replace(
            _unsplit_anchor,
            "_unsloth_grpo_unsplit_vision(unsplit_pixel_values_by_grid(batch)) "
            "for batch in generation_batches",
        )
    elif "split_pixel_values_by_grid" in function:
        _warn_once(
            "grpo_split_vision_by_sample",
            "Unsloth: the GRPO batch splitting changed shape, so a multi tile vision "
            "batch is being shuffled by sample index without its tiles.",
        )
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Guard reload_weights and sync_weights: skip when fast inference LoRA shares weights with vLLM (huggingface/trl commit 7856d3b).
def _guard_vllm_sync_reload_for_shared_weights(function):
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

    # TRL 0.24.0-0.25.1 truncation regression: 0.22.2-0.23.1 used truncate_with_protected_tokens (tokenize, keep the RIGHTMOST tokens, protect vision tokens), 0.24.0-0.25.1 passed max_length/truncation to the tokenizer, which protects nothing, and 0.26.2+ removed those kwargs. Dropping them makes 0.24.0-0.25.1 behave like 0.26.2+; a no-op elsewhere.
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


# Older TRL mishandles special tokens: 0.19.0 passed skip_special_tokens = True where it should be False.
def grpo_trainer__generate_and_score_completions(function_name, function):
    if function_name != "_generate_and_score_completions":
        return function

    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    line_to_replace = 'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size'

    replacement_lines_head = """
        max_left_pad = None
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        # Which name carries "this batch has images" moved twice, and at the bottom of the
        # declared window neither name exists: 0.20.0 through 0.23.1 bind has_images, 0.24.0
        # and up bind images, and 0.18.2 / 0.19.1 have no vision path at all, so every batch
        # there is text only. Probing has_images and falling back to images without a third
        # branch made the floor raise NameError out of the except handler and killed training.
        try:
            _unsloth_text_only = not has_images
        except NameError:
            try:
                _unsloth_text_only = images is None
            except NameError:
                _unsloth_text_only = True
        try:
            _unsloth_video_images = images
        except NameError:
            _unsloth_video_images = None
        try:
            _unsloth_video_prompts = prompts
        except NameError:
            _unsloth_video_prompts = []
        _unsloth_video_kwargs = _unsloth_grpo_video_inputs(
            self, _unsloth_video_prompts, _unsloth_video_images
        )
        if _unsloth_video_kwargs is not None:
            _unsloth_text_only = False
        if _unsloth_text_only:
            # Left pad prompt before calculation old and ref hidden states
            left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(prompt_completion_ids, logits_to_keep, self.processing_class.pad_token_id)
            max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
        _use_gc = self.model._unsloth_gradient_checkpointing if hasattr(self.model, '_unsloth_gradient_checkpointing') else getattr(self.args, 'gradient_checkpointing', True)
        self.model.for_training(use_gradient_checkpointing=_use_gc)"""

    # TRL 0.22.x-0.23.x has no forward_kwargs / num_images, so a multi-image row is sliced as several
    # rows. Count images off the same normalised cells. Emitted only for those versions so newer
    # generated source stays byte-identical.
    _legacy_vision_prologue = """
        _unsloth_legacy_vision = {}
        try:
            # Gated the way TRL 0.24.0 gates forward_kwargs ("if images is not None"): a text only
            # batch must forward nothing. Gemma 3's processor returns token_type_ids even with no
            # images, and both PrefixGrouper and sequence packing are keyed on that name being
            # absent, so copying it across would quietly cost a text run both fast paths.
            if has_images and images is not None:
                _unsloth_legacy_vision = _unsloth_grpo_vision_inputs(prompt_inputs)
                _unsloth_legacy_vision["num_images"] = [
                    len(_unsloth_cell) if _unsloth_cell else 0
                    for _unsloth_cell in (
                        _unsloth_grpo_image_cell(_unsloth_image) for _unsloth_image in images
                    )
                ]
                _unsloth_legacy_vision = {
                    _unsloth_key: _unsloth_value
                    for _unsloth_key, _unsloth_value in _unsloth_legacy_vision.items()
                    if _unsloth_value is not None
                }
            if _unsloth_legacy_vision.get("pixel_values", None) is None:
                # An image column whose cells are all empty: a text batch wearing one.
                _unsloth_legacy_vision.pop("token_type_ids", None)
                _unsloth_legacy_vision.pop("mm_token_type_ids", None)
            # TRL 0.24.0 extends these with zeros over the completion before the very same call
            # (grpo_trainer.py, "If token_type_ids are used, extend them with zeros"); 0.22.x-0.23.x
            # never forwarded them at all, so the processor's prompt length tensor would not fit a
            # prompt+completion forward. Only the prompt width is widened; every other width is
            # dropped rather than sent, including one that happens to equal prompt+completion, which
            # on these versions can only be an untruncated prompt whose tokens sit at the wrong
            # offsets after max_prompt_length moved them.
            for _unsloth_key in ("token_type_ids", "mm_token_type_ids"):
                _unsloth_ids = _unsloth_legacy_vision.get(_unsloth_key, None)
                if _unsloth_ids is None:
                    continue
                if (
                    getattr(_unsloth_ids, "ndim", 0) == 2
                    and _unsloth_ids.shape[0] == prompt_completion_ids.shape[0]
                    and _unsloth_ids.shape[1] == prompt_ids.shape[1]
                ):
                    _unsloth_legacy_vision[_unsloth_key] = torch.cat(
                        [_unsloth_ids, _unsloth_ids.new_zeros(completion_ids.shape)], dim = 1
                    )
                else:
                    _unsloth_legacy_vision.pop(_unsloth_key, None)
        except NameError:
            _unsloth_legacy_vision = {}"""

    _legacy_vision_trl = 'pixel_values=prompt_inputs.get("pixel_values")' in function
    replacement_lines = replacement_lines_head + (
        _legacy_vision_prologue if _legacy_vision_trl else ""
    )

    function = function.replace(line_to_replace, replacement_lines)

    # TRL 0.22.x-0.23.x passes four tensors and no counts, so the chunker cannot split images per sample
    _legacy_vision_call = re.compile(
        r"(^[ \t]+)pixel_values=prompt_inputs\.get\(\"pixel_values\"\),\n"
        r"[ \t]+image_grid_thw=prompt_inputs\.get\(\"image_grid_thw\"\),\n"
        r"[ \t]+pixel_attention_mask=prompt_inputs\.get\(\"pixel_attention_mask\"\),\n"
        r"[ \t]+image_sizes=prompt_inputs\.get\(\"image_sizes\"\),\n",
        re.MULTILINE,
    )
    function, _legacy_vision_calls = _legacy_vision_call.subn(
        r"\1**_unsloth_legacy_vision,\n",
        function,
    )
    # Guard against future drift: 0.22.2-0.23.1 take all three rewrites, 0.24.0+ need none.
    _legacy_vision_unplumbed = (
        _legacy_vision_calls == 0 and 'prompt_inputs.get("pixel_values")' in function
    )
    if _legacy_vision_unplumbed:
        _warn_once(
            "grpo_legacy_vision_calls",
            "Unsloth: the GRPO reference logprob call sites changed shape, so a multi "
            "image batch is being sliced there without its image counts.",
        )

    # Keep TRL's old logprob gate: every old_per_token_logps consumer runs only under IS correction
    # or treats None as per_token_logps.detach(), so forcing it on for vLLM is a wasted forward.

    pattern_to_find = re.compile(
        r"(^\s*)all_logprobs = \["  # Capture indentation (group 1)
        r".*?"  # Match everything inside non-greedily
        r"for output in outputs\.outputs\s*"
        r"\]",
        re.DOTALL | re.MULTILINE,
    )

    # sanitize_logprob is injected as a module-level function by the RLTrainer_replacement template in rl.py, so reference it directly.
    replacement_text = (
        r"\1all_logprobs = [\n"
        r"\1    [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]\n"
        r"\1    for outputs in all_outputs\n"
        r"\1    for output in outputs.outputs\n"
        r"\1]"
    )

    function, num_replacements = pattern_to_find.subn(replacement_text, function)

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
            # The regex also matches trl 0.18.2 / 0.19.1, which lack the emitted symbols; TRL's own helper call
            # discriminates (tests/version_compat/test_grpo_protected_tokens_guard.py).
            and "truncate_with_protected_tokens" in replace_part
        ):
            # getattr: an unset id means nothing to protect, not AttributeError in generated code.
            new_replacement = f"""\n{spacing}if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [
                getattr(self, "image_token_id", None),
                getattr(self, "vision_start_token_id", None),
                getattr(self, "vision_end_token_id", None),
            ]
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

    # TRL's importance sampling is disabled because the LLM path moves left padding to the right, so Unsloth adjusts the vLLM sampling_logprob tensor itself.
    string_to_find = "if self.use_vllm and self.vllm_importance_sampling_correction:"

    replacement_string = "if False and self.use_vllm and self.vllm_importance_sampling_correction:"

    function = function.replace(string_to_find, replacement_string)

    string_to_find = """        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]"""

    # 0.22.x-0.23.x saves only four processor keys; copy the rest so both policies see one batch.
    replacement_string = """        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        try:
            for _vision_key, _vision_value in _unsloth_legacy_vision.items():
                if _vision_value is not None and _vision_key not in output:
                    output[_vision_key] = _vision_value
        except NameError:
            pass
        if max_left_pad is not None:
            output["max_left_pad"] = torch.tensor(prompt_ids.shape[0] * [max_left_pad]).unsqueeze(-1)
        try:
            if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):
                output["sampling_per_token_logps"] = sampling_per_token_logps
        except NameError:
            output["sampling_per_token_logps"] = None"""

    function = function.replace(string_to_find, replacement_string)

    # TRL 0.24.0+ extracts prompts = [x["prompt"] for x in inputs], losing metadata like reasoning_effort, so inject code storing per-sample chat_template_kwargs on self.
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
            _target_line
            + _metadata_extraction
            + "        prompts = _unsloth_grpo_clean_video_prompts(prompts, self)\n",
        )

    # TRL builds the logprob forward kwargs from images only; add the batch's videos.
    _video_merge_anchor = "\n        else:\n            forward_kwargs = {}\n"
    if _video_merge_anchor in function:
        function = function.replace(
            _video_merge_anchor,
            _video_merge_anchor
            + "        if _unsloth_video_kwargs is not None:\n"
            + "            forward_kwargs = {**forward_kwargs, **_unsloth_video_kwargs}\n",
            1,
        )
    elif "_unsloth_grpo_video_inputs(" in function:
        function = function.replace(
            "        if _unsloth_video_kwargs is not None:\n            _unsloth_text_only = False\n",
            "        if _unsloth_video_kwargs is not None:\n"
            "            raise NotImplementedError(\n"
            "                'Unsloth: video GRPO needs TRL 0.24.0 or newer: pip install -U trl'\n"
            "            )\n",
        )

    _output_extras = """
        try:
            _unsloth_vision_output = _unsloth_grpo_vision_inputs(forward_kwargs)
        except NameError:
            _unsloth_vision_output = {}
        for _vision_key, _vision_value in _unsloth_vision_output.items():
            if _vision_value is not None and _vision_key not in output:
                output[_vision_key] = _vision_value
        if max_left_pad is not None:
            output["max_left_pad"] = torch.tensor(prompt_ids.shape[0] * [max_left_pad]).unsqueeze(-1)
        try:
            if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):
                output["sampling_per_token_logps"] = sampling_per_token_logps
        except NameError:
            output["sampling_per_token_logps"] = None"""

    # Longest first: the short anchor strands TRL 1.7.0's num_tiles line. unsloth#6960.
    _num_images_anchors = (
        """        if images is not None:
            output["num_images"] = num_images
            if num_tiles is not None:
                output["num_tiles"] = num_tiles""",
        """        if images is not None:
            output["num_images"] = num_images""",
    )
    for _anchor in _num_images_anchors:
        if _anchor in function:
            function = function.replace(_anchor, _anchor + _output_extras)
            break
    else:
        if 'output["num_images"]' in function:
            _warn_once(
                "grpo_output_num_images",
                "Unsloth: the GRPO num_images output block changed shape, so "
                "max_left_pad and the vLLM sampling logprobs are not being saved.",
            )

    if trl_version >= Version("0.24.0"):
        string_to_find = "        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)"
        # TRL's VLM tool branch has no prompts_text; TRL 1.15 dropped completions_text, rebuilt as in 1.13.
        replacement_string = (
            "        _unsloth_reward_locals = locals()\n"
            "        if images is not None and 'prompts_text' in _unsloth_reward_locals:\n"
            "            completions_text = _unsloth_reward_locals.get('completions_text')\n"
            "            if completions_text is None:\n"
            "                completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)\n"
            "            rewards_per_func = self._calculate_rewards(inputs, prompts_text, completions_text, completion_ids_list)\n"
            "        else:\n"
            "            rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)\n"
            # On CPython <= 3.12 a later locals() call would store this frame dict in itself, a
            # cycle that keeps every rollout tensor alive until the cyclic GC runs.
            "        del _unsloth_reward_locals"
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
        if (
            "mm_token_type_ids" in forward_kwargs
            or "image_grid_thw" in forward_kwargs
            or "video_grid_thw" in forward_kwargs
        ):
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

    # Anchors for TRL >= 1.0.0, then 0.22.x-0.23.x.
    _image_cell_anchors = (
        (
            'images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]',
            'images = [_unsloth_grpo_image_cell(example.get("image")) for example in inputs]',
        ),
        (
            'kwargs = {"images": [[img] for img in images]}',
            'kwargs = {"images": [_unsloth_grpo_image_cell(img) for img in images]}',
        ),
    )
    _image_cell_normalised = False
    _legacy_image_cell = False
    for _index, (_old_cell, _new_cell) in enumerate(_image_cell_anchors):
        if _old_cell in function:
            function = function.replace(_old_cell, _new_cell)
            _image_cell_normalised = True
            # The legacy spelling is the only one whose placeholders are sized elsewhere.
            _legacy_image_cell = _index == 1

    # TRL 0.22.x-0.23.x hardcodes one image placeholder per example.
    _placeholder_old = (
        "            for prompt in prompts:\n"
        "                if isinstance(prompt, list):  # i.e., when using conversational data\n"
        "                    prepare_multimodal_messages(prompt, num_images=1)"
    )
    # Empty cells take zero placeholders; all-empty column becomes text (TRL 0.24.0+), else IndexError
    _placeholder_new = (
        '            if kwargs.get("images") and all(\n'
        '                _unsloth_cell == [] for _unsloth_cell in kwargs["images"]\n'
        "            ):\n"
        "                kwargs = {}\n"
        "                has_images = False\n"
        '            for prompt, _unsloth_cell in zip(prompts, kwargs.get("images", [])):\n'
        "                if isinstance(prompt, list):  # i.e., when using conversational data\n"
        "                    prepare_multimodal_messages(\n"
        "                        prompt, num_images=len(_unsloth_cell) if _unsloth_cell else 0\n"
        "                    )"
    )
    # TRL 0.24.0+ sizes placeholders itself; 0.20.0 / 0.21.0 inline one per message, so not sized.
    _placeholder_sized = not _legacy_image_cell
    if _placeholder_old in function:
        function = function.replace(_placeholder_old, _placeholder_new)
        _placeholder_sized = True

    # On a legacy TRL the reference logprob calls need counts too, else a multi-image row fails deep in
    # the model; refuse it. `self` narrows the refusal to vLLM server mode at runtime.
    _guard_argument = ""
    if not _image_cell_normalised or _legacy_vision_unplumbed or not _placeholder_sized:
        pass
    elif _legacy_image_cell:
        _guard_argument = ", self"
    if (
        _guard_argument
        or not _image_cell_normalised
        or _legacy_vision_unplumbed
        or (not _placeholder_sized)
    ):
        _signature = re.search(
            r"([ \t]*)def _generate_and_score_completions\((?:[^()]|\([^()]*\))*\)[^\n]*:\n",
            function,
        )
        if _signature is None:
            _warn_once(
                "grpo_image_column",
                "Unsloth: could not install the GRPO singular `image` column "
                "normalisation; a column holding a list of images may fail inside the "
                "processor. See https://github.com/unslothai/unsloth/issues/3605",
            )
        else:
            _indent = _signature.group(1) + "    "
            function = (
                function[: _signature.end()]
                + f"{_indent}_unsloth_reject_grpo_image_list(inputs{_guard_argument})\n"
                + function[_signature.end() :]
            )

    # Opt-in ScaleRL loss weights (zero-variance group exclusion, prompt-level aggregation). They need
    # the group layout, which only exists here: the loss sees shuffled micro-batches.
    _zero_std_anchor = (
        'self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())'
    )
    _return_anchor = "\n        return output"
    if function.count(_zero_std_anchor) == 1 and function.count(_return_anchor) == 1:
        function = function.replace(
            _zero_std_anchor,
            _zero_std_anchor + "\n"
            "        _unsloth_sample_weight = _unsloth_grpo_sample_weight(\n"
            "            self, mode, rewards_per_func, completion_mask, locals().get('tool_mask'), process_slice,\n"
            "        )",
        )
        function = function.replace(
            _return_anchor,
            "\n        if _unsloth_sample_weight is not None:\n"
            '            output["unsloth_sample_weight"] = _unsloth_sample_weight' + _return_anchor,
        )
    elif _unsloth_grpo_sample_weight_mode() is not None:
        _warn_once(
            "grpo_sample_weight",
            "Unsloth: UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE / UNSLOTH_GRPO_LOSS_AGGREGATION could "
            "not be installed for this TRL version and are ignored.",
        )

    _old_logps_if = re.search(
        r"\n([ \t]*)if self\.args\.gradient_accumulation_steps % generate_every != 0",
        function,
    )
    _old_logps_output = (
        "        if old_per_token_logps is not None:\n"
        '            output["old_per_token_logps"] = old_per_token_logps\n'
    )
    if (
        _old_logps_if is not None
        and function.count(_old_logps_output) == 1
        and "sampling_per_token_logps" in function
    ):
        _indent = _old_logps_if.group(1)
        function = (
            function[: _old_logps_if.start()]
            + f"\n{_indent}_unsloth_rollout_logps = locals().get('sampling_per_token_logps')\n"
            + f"{_indent}_unsloth_rollout_old = _unsloth_grpo_rollout_logps_as_old(\n"
            + f"{_indent}    self, _unsloth_rollout_logps, completion_mask, locals().get('tool_mask'),\n"
            + f"{_indent})\n"
            + f"{_indent}if _unsloth_rollout_old:\n"
            + f"{_indent}    old_per_token_logps = None\n"
            + f"{_indent}el"
            + function[_old_logps_if.start() + 1 + len(_indent) :]
        )
        function = function.replace(
            _old_logps_output,
            _old_logps_output + "        if _unsloth_rollout_old:\n"
            '            output["unsloth_rollout_logps_as_old"] = torch.ones(\n'
            "                completion_ids.size(0), dtype = torch.bool, device = completion_ids.device\n"
            "            )\n"
            # Older TRL (0.23.x) keeps the rollout logprobs only with the IS correction on.
            '            output["sampling_per_token_logps"] = _unsloth_rollout_logps\n',
        )
    elif os.environ.get("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "0") == "1":
        _warn_once(
            "grpo_rollout_logps_as_old",
            "Unsloth: UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD could not be installed for this TRL "
            "version and is ignored.",
        )

    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate_and_score_completions)


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


def _unsloth_vllm_quantized(model):
    # FSDP2 turns Params4bit into a DTensor, so also read the load flags the model keeps.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    # transformers marks every quantizer it loads (GPTQ, AWQ, HQQ, torchao, compressed-tensors, ...).
    # A dequantize = True load removes all of these, so they only remain on a quantized model.
    if getattr(model, "is_quantized", False) is True or getattr(model, "quantization_method", None):
        return True
    if getattr(model, "hf_quantizer", None) is not None:
        return True
    quantization_config = getattr(getattr(model, "config", None), "quantization_config", None)
    if quantization_config:
        dequantize = (
            quantization_config.get("dequantize", False)
            if isinstance(quantization_config, dict)
            else getattr(quantization_config, "dequantize", False)
        )
        if not dequantize:
            return True
    for module in model.modules():
        # Packed integer weights (GPTQ / AWQ qweight, HQQ W_q) are not vLLM's layout.
        if isinstance(getattr(module, "qweight", None), torch.Tensor) or isinstance(
            getattr(module, "W_q", None), torch.Tensor
        ):
            return True
    for param in model.parameters():
        if type(param).__name__ in ("Params4bit", "Int8Params"):
            return True
        if param.is_floating_point() and param.element_size() == 1:
            return True
    return False


def _unsloth_vllm_lora_layers(model):
    return [
        module
        for module in model.modules()
        if hasattr(module, "lora_A") and getattr(module, "base_layer", None) is not None
    ]


def _unsloth_vllm_sync_unsafe(model):
    if _unsloth_vllm_quantized(model):
        return True
    if getattr(model, "peft_config", None):
        return True
    return len(_unsloth_vllm_lora_layers(model)) != 0


# LoRA layer classes whose merged weight _unsloth_vllm_merged_weight computes: one dense base weight.
_UNSLOTH_VLLM_PLAIN_LORA = ("Linear", "Embedding", "Conv1d", "Conv2d", "Conv3d")
# LoRA variants PEFT itself cannot merge into the base weight (they condition on the input).
_UNSLOTH_VLLM_UNMERGEABLE_VARIANTS = ("ALora", "Arrow", "QALora")


def _unsloth_vllm_peft_type(config):
    peft_type = getattr(config, "peft_type", None)
    return str(getattr(peft_type, "value", peft_type))


# PEFT types whose merge writes the merged weight into the base tensors (no module swap, no per-input routing).
_UNSLOTH_VLLM_MERGEABLE_TYPES = ("LORA", "LOHA", "LOKR", "IA3", "VERA")


def _unsloth_vllm_active_adapters(model):
    configs = getattr(model, "peft_config", None) or {}
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if isinstance(active, str):
        active = [active]
    return set(active) if active else set(configs)


def _unsloth_vllm_shared_targets(model):
    """Adapted base parameters reachable under more than one name (tied embed_tokens / lm_head, adapted
    at one or both sites): vLLM gets one tensor for them, so some call site would get the wrong delta."""
    names = {}
    for name, param in model.named_parameters(remove_duplicate = False):
        names.setdefault(id(param), []).append(name)
    shared = {}
    for module in model.modules():
        if getattr(module, "base_layer", None) is None or not hasattr(module, "get_base_layer"):
            continue
        for param in module.get_base_layer().parameters(recurse = False):
            if len(names.get(id(param), ())) > 1:
                shared[id(param)] = " and ".join(names[id(param)])
    return list(shared.values())


def _unsloth_vllm_offloaded(model):
    """How the model keeps weights off the device between forwards, or None: a sync would read the
    empty or meta placeholders."""
    for module in model.modules():
        if getattr(module, "_unsloth_block_swap", None) is not None:
            return "block-swapped decoder layers (offload_layers)"
        hook = getattr(module, "_hf_hook", None)
        hooks = getattr(hook, "hooks", None) or ([hook] if hook is not None else [])
        if any(getattr(h, "offload", False) for h in hooks):
            return "CPU / disk offloaded layers (Accelerate device_map)"
    if any(param.device.type == "meta" for param in model.parameters()):
        return "meta (offloaded) parameters"
    return None


def _unsloth_vllm_peft_unsupported(model):
    """PEFT setups neither TRL's PEFT branch nor the out-of-place merge can sync: names of the culprits.
    Only active adapters count; an inactive one does not shape the policy being synced."""
    problems = []
    active = _unsloth_vllm_active_adapters(model)
    if type(model).__name__ == "PeftMixedModel":
        # Adapter parameters carry per-type prefixes (lora_, hada_, lokr_, ...) the sync cannot filter.
        problems.append("a mixed PEFT model (get_peft_model(..., mixed = True))")
    for name, config in (getattr(model, "peft_config", None) or {}).items():
        # Replicas share one base tensor but train separate adapters, and vLLM loads the checkpoint's
        # layer count: there is no per-replica weight to send (PEFT also refuses to merge these).
        if getattr(config, "layer_replication", None):
            problems.append(f"layer_replication on adapter '{name}'")
        if name not in active:
            continue
        peft_type = _unsloth_vllm_peft_type(config)
        if getattr(config, "is_prompt_learning", False) or getattr(
            config, "is_adaption_prompt", False
        ):
            problems.append(f"prompt learning adapter '{name}' ({_unsloth_vllm_peft_type(config)})")
        if getattr(config, "trainable_token_indices", None):
            problems.append(f"trainable_token_indices on adapter '{name}'")
        if getattr(config, "alora_invocation_tokens", None):
            problems.append(f"aLoRA adapter '{name}'")
        if getattr(config, "arrow_config", None) is not None:
            problems.append(f"Arrow adapter '{name}'")
        elif not getattr(config, "is_prompt_learning", False) and (
            peft_type not in _UNSLOTH_VLLM_MERGEABLE_TYPES
        ):
            # LN tuning swaps modules, POLY / X-LoRA route per input: no single merged weight.
            problems.append(f"{peft_type} adapter '{name}' (no static merge into the base weights)")
    for module_name, module in model.named_modules():
        # trainable_token_indices, even on an inactive adapter: its token_adapter.* parameters and
        # names do not exist in vLLM and are pushed all the same.
        if type(module).__name__ in ("TrainableTokensWrapper", "TrainableTokensLayer"):
            problems.append(f"trainable tokens at {module_name}")
            break
    for module_name, module in model.named_modules():
        # PEFT's MultiheadAttention merge replaces in_proj_weight / out_proj.weight instead of
        # editing them, so a copy merge cannot leave the live module untouched.
        if type(module).__name__ == "MultiheadAttention" and hasattr(module, "lora_A"):
            problems.append(f"LoRA on MultiheadAttention at {module_name}")
            break
    for module_name, module in model.named_modules():
        # A custom tuner layer on a composite base: its weights live in child modules the copy
        # merge does not see, so the merged weights would never be sent.
        if getattr(module, "base_layer", None) is None or not hasattr(module, "get_base_layer"):
            continue
        base = module.get_base_layer()
        if any(any(True for _ in child.parameters()) for child in base.children()):
            problems.append(f"adapter on the composite module {module_name}")
            break
    for module in _unsloth_vllm_lora_layers(model):
        for adapter, variant in (getattr(module, "lora_variant", None) or {}).items():
            if adapter not in active:
                continue
            variant_name = type(variant).__name__
            if any(variant_name.startswith(x) for x in _UNSLOTH_VLLM_UNMERGEABLE_VARIANTS):
                problems.append(f"LoRA variant {variant_name}")
    return list(dict.fromkeys(problems))


def _unsloth_vllm_modules_to_save(model):
    """(wrapper name, adapter it serves or None for original_module, all its adapter names)."""
    wrappers = []
    for module_name, module in model.named_modules():
        if type(module).__name__ != "ModulesToSaveWrapper":
            continue
        copies = getattr(module, "modules_to_save", None) or {}
        served = None
        if not getattr(module, "disable_adapters", False):
            for adapter in getattr(module, "active_adapters", None) or []:
                if adapter in copies:
                    served = adapter
                    break
        wrappers.append((module_name, served, list(copies.keys())))
    return wrappers


def _unsloth_vllm_plain_lora(model):
    """True when every PEFT piece is plain LoRA that _unsloth_vllm_sync_peft_merged fully syncs."""
    configs = getattr(model, "peft_config", None) or {}
    active = _unsloth_vllm_active_adapters(model)
    if any(
        _unsloth_vllm_peft_type(config) != "LORA"
        for name, config in configs.items()
        if name in active
    ):
        return False
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        BaseTunerLayer = ()
    lora_layers = {id(module) for module in _unsloth_vllm_lora_layers(model)}
    for module in model.modules():
        is_tuner = (BaseTunerLayer != () and isinstance(module, BaseTunerLayer)) or (
            id(module) in lora_layers
        )
        if not is_tuner:
            continue
        # ParamWrapper (target_parameters), TrainableTokensLayer, ...: not a single dense weight.
        if id(module) not in lora_layers or type(module).__name__ not in _UNSLOTH_VLLM_PLAIN_LORA:
            return False
        base_layer = module.base_layer
        if not isinstance(getattr(base_layer, "weight", None), torch.Tensor):
            return False
        if getattr(module, "lora_A", None) is not None and hasattr(base_layer, "base_layer"):
            return False
        for flags in ("lora_variant", "use_dora", "lora_bias"):
            if any(v for a, v in (getattr(module, flags, None) or {}).items() if a in active):
                return False
    return True


def _unsloth_vllm_sync_kind(trainer):
    """How _move_model_to_vllm syncs TRL's own vLLM engine (fast_inference = False):
    "trl" = TRL's body (dense, or sharded PEFT where its in-place merge cannot drift the base),
    "peft_merged" = PEFT merged out of place, "skip" = no sync (explicit opt-in only). Anything that cannot be synced raises: rollouts
    would come from the initial weights all run."""
    model = trainer.model
    allow_stale = os.environ.get("UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS", "0") == "1"
    if _unsloth_vllm_quantized(model):
        if allow_stale:
            print(
                "Unsloth: UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1, so vLLM weight sync is skipped for this "
                "quantized model and every rollout uses the initial weights."
            )
            return "skip"
        raise RuntimeError(
            "Unsloth: GRPO with use_vllm = True but fast_inference = False cannot sync a quantized "
            "(bitsandbytes, GPTQ, AWQ, HQQ, torchao, FP8, ...) model into TRL's own vLLM engine, so every "
            "rollout would come from the initial weights while the policy trains.\n"
            "Load the model with fast_inference = True, or load a 16-bit base (load_in_4bit = False, "
            "load_in_8bit = False). Set UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1 to train anyway."
        )
    offloaded = _unsloth_vllm_offloaded(model)
    if offloaded is not None:
        if allow_stale:
            print(
                "Unsloth: UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1, so vLLM weight sync is skipped for this "
                f"model with {offloaded} and every rollout uses the initial weights."
            )
            return "skip"
        raise RuntimeError(
            "Unsloth: GRPO with use_vllm = True but fast_inference = False cannot sync a model with "
            f"{offloaded}: those weights are empty or on the meta device between forwards.\n"
            "Keep every layer on the GPU (no offload_layers, no CPU / disk device_map), or load the "
            "model with fast_inference = True. Set UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1 to train anyway."
        )
    if not _unsloth_vllm_sync_unsafe(model):
        return "trl"
    state = getattr(getattr(trainer, "accelerator", None), "state", None)
    deepspeed_plugin = getattr(state, "deepspeed_plugin", None)
    sharded = getattr(trainer, "is_fsdp_enabled", False) or (
        deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", 0) == 3
    )
    problems = _unsloth_vllm_peft_unsupported(model)
    problems.extend(
        f"one base weight shared by {names}" for names in _unsloth_vllm_shared_targets(model)
    )
    # Unsharded: every mergeable PEFT setup is merged out of place, so the frozen base never moves.
    # ZeRO-3: TRL merges inside GatheredParameters(modifier_rank = None), which discards in-place edits
    # of the gathered copy on exit, so its merge / unmerge cannot drift the stored base either.
    # FSDP: TRL calls merge_adapter before it gathers the sharded parameters, so PEFT's full-size
    # deltas meet sharded base weights (and the in-place round trip would drift a 16-bit base).
    kind = "peft_merged" if not sharded else "trl"
    if getattr(trainer, "is_fsdp_enabled", False):
        problems.append(
            "FSDP (TRL merges the adapters before it gathers the sharded weights; use DeepSpeed "
            "ZeRO-3 or a single process)"
        )
    if kind == "trl":
        # TRL strips only "modules_to_save.default.", so other adapter names reach vLLM as is.
        for wrapper, _, adapters in _unsloth_vllm_modules_to_save(model):
            if any(adapter != "default" for adapter in adapters):
                problems.append(f"modules_to_save at {wrapper} for adapters {adapters}")
    if problems:
        if allow_stale:
            print(
                "Unsloth: UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1, so vLLM weight sync is skipped for this "
                f"PEFT model ({'; '.join(problems)}) and every rollout uses the initial weights."
            )
            return "skip"
        raise RuntimeError(
            "Unsloth: GRPO with use_vllm = True but fast_inference = False cannot sync this PEFT model "
            "into TRL's own vLLM engine: "
            + "; ".join(problems)
            + ". These cannot be merged into the "
            "base weights under vLLM's parameter names, so every rollout would come from the initial "
            "weights while the policy trains.\n"
            "Use a plain LoRA adapter, or set UNSLOTH_VLLM_ALLOW_STALE_ROLLOUTS=1 to train anyway."
        )
    return kind


def _unsloth_vllm_sync_kind_cached(trainer):
    """_unsloth_vllm_sync_kind, decided again whenever the active adapters change (set_adapter)."""
    key = tuple(sorted(_unsloth_vllm_active_adapters(trainer.model)))
    if getattr(trainer, "_unsloth_vllm_sync_kind", None) is None or (
        getattr(trainer, "_unsloth_vllm_sync_key", None) != key
    ):
        trainer._unsloth_vllm_sync_kind = _unsloth_vllm_sync_kind(trainer)
        trainer._unsloth_vllm_sync_skip = trainer._unsloth_vllm_sync_kind == "skip"
        trainer._unsloth_vllm_sync_key = key
    return trainer._unsloth_vllm_sync_kind


def _unsloth_vllm_merged_weight(module, weight):
    # W + sum of active deltas, computed out of place in fp32 and rounded once. merge_adapter +
    # unmerge_adapter would round twice in place and drift the frozen base every sync.
    merged = None
    if not getattr(module, "disable_adapters", False) and not getattr(module, "merged", False):
        for adapter in module.active_adapters:
            lora_A = module.lora_A[adapter] if adapter in module.lora_A else None
            if isinstance(lora_A, torch.nn.Linear) and adapter in module.lora_B:
                delta = module.lora_B[adapter].weight.float() @ lora_A.weight.float()
                delta = delta * module.scaling[adapter]
                if getattr(module, "fan_in_fan_out", False):
                    delta = delta.T
            elif adapter in module.lora_A or adapter in getattr(module, "lora_embedding_A", {}):
                delta = module.get_delta_weight(adapter).float()
            else:
                continue
            delta = delta.to(weight.device).reshape(weight.shape)
            merged = delta if merged is None else merged + delta
    if merged is None:
        return weight.data
    return (weight.float() + merged).to(weight.dtype)


def _unsloth_vllm_tuner_groups(model):
    """{id(base param): (real base module, [tuner layers wrapping it])} for PEFT's own merge."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    groups = {}
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            base = module.get_base_layer()
            groups.setdefault(id(base), (base, []))[1].append(module)
    return {
        id(param): group
        for group in groups.values()
        for param in group[0].parameters(recurse = False)
    }


def _unsloth_vllm_peft_merge_out_of_place(base, layers):
    """Run PEFT's own merge (DoRA, lora_bias, LoHa, LoKr, IA3, VeRA, ParamWrapper, ...) on copies of
    the base tensors and return {id(param): merged tensor}; the live parameters keep their exact bytes."""
    params = list(base.parameters(recurse = False))
    originals = [param.data for param in params]
    states = [
        (layer, list(getattr(layer, "merged_adapters", [])), dict(getattr(layer, "_caches", {})))
        for layer in layers
    ]
    # Eval mode while merging: LoHa / LoKr rank dropout would otherwise thin the delta at random.
    modes = [(module, module.training) for layer in layers for module in layer.modules()]
    try:
        for param, data in zip(params, originals):
            param.data = data.clone()
        for layer in layers:
            layer.train(False)
        # model.modules() order, as PeftModel.merge_adapter merges them. A disabled layer runs only
        # its base in training, so it adds nothing (PEFT's merge ignores disable_adapters).
        for layer in layers:
            if not getattr(layer, "disable_adapters", False):
                layer.merge()
        return {id(param): param.data for param in params}
    finally:
        for module, training in modes:
            module.training = training
        for param, data in zip(params, originals):
            param.data = data
        for layer, merged_adapters, caches in states:
            if hasattr(layer, "merged_adapters"):
                layer.merged_adapters[:] = merged_adapters
            if isinstance(getattr(layer, "_caches", None), dict):
                layer._caches.clear()
                layer._caches.update(caches)


def _unsloth_vllm_adapter_param_ids(model):
    """Adapter tensors: every parameter under a tuner layer except its base layer's own."""
    ids = set()
    for module in model.modules():
        if getattr(module, "base_layer", None) is None or not hasattr(module, "get_base_layer"):
            continue
        base_ids = {id(param) for param in module.get_base_layer().parameters()}
        ids.update(id(param) for param in module.parameters() if id(param) not in base_ids)
    return ids


_UNSLOTH_VLLM_MODULES_TO_SAVE = re.compile(r"\.(?:modules_to_save\.[^.]+|original_module)\.")


@torch.no_grad()
def _unsloth_vllm_sync_peft_merged(trainer):
    """TRL's PEFT sync (merged weights under the base-layer names), without mutating the model."""
    model = trainer.model
    # Plain LoRA: W + sum of deltas in fp32, rounded once. Anything else: PEFT's merge on a copy.
    plain = _unsloth_vllm_plain_lora(model)
    lora_of = {}
    groups = {}
    if plain:
        for module in _unsloth_vllm_lora_layers(model):
            weight = getattr(module.base_layer, "weight", None)
            if weight is not None:
                lora_of[id(weight)] = module
    else:
        groups = _unsloth_vllm_tuner_groups(model)
    merged_cache = {}
    # modules_to_save: push the copy the active adapter trains (any adapter name), never the rest.
    unserved = []
    for wrapper, served, adapters in _unsloth_vllm_modules_to_save(model):
        unserved.extend(f"{wrapper}.modules_to_save.{a}." for a in adapters if a != served)
        if served is not None:
            unserved.append(f"{wrapper}.original_module.")
    unserved = tuple(unserved)
    # Structural, not a name match: a base module named q_lora_proj is still a base weight.
    adapter_ids = _unsloth_vllm_adapter_param_ids(model)
    # TRL 0.18 has no vllm_mode: its in-process engine is the colocated one.
    mode = getattr(trainer, "vllm_mode", None) or (
        "server" if getattr(trainer, "vllm_client", None) is not None else "colocate"
    )
    is_main = getattr(trainer.accelerator, "is_main_process", True)
    llm_model = None
    if mode == "colocate":
        llm_model = trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model
    # TRL < 0.20 sends the names as they are and has no fixer.
    fix_name = getattr(trainer, "_fix_param_name_to_vllm", None) or (lambda name, **_: name)
    for full_name, param in model.named_parameters():
        if unserved and full_name.startswith(unserved):
            continue
        if id(param) in adapter_ids:
            continue
        name = full_name.removeprefix("base_model.model.").replace(".base_layer", "")
        name = fix_name(_UNSLOTH_VLLM_MODULES_TO_SAVE.sub(".", name))
        if plain:
            module = lora_of.get(id(param))
            tensor = param.data if module is None else _unsloth_vllm_merged_weight(module, param)
        elif id(param) in groups:
            # One base module at a time: its weight and bias are adjacent in named_parameters.
            if id(param) not in merged_cache:
                merged_cache.clear()
                merged_cache.update(_unsloth_vllm_peft_merge_out_of_place(*groups[id(param)]))
            tensor = merged_cache.pop(id(param))
        else:
            tensor = param.data
        if mode == "server":
            if is_main:
                trainer.vllm_client.update_named_param(name, tensor)
        elif mode == "colocate":
            llm_model.load_weights([(name, tensor)])
    if mode == "server" and is_main:
        trainer.vllm_client.reset_prefix_cache()
    elif mode == "colocate":
        trainer.llm.reset_prefix_cache()
    return None


def grpo_trainer__move_model_to_vllm(function_name, function):
    if function_name != "_move_model_to_vllm":
        return function

    # Only fast_inference shares weights with vLLM. TRL's own vLLM (colocate or server, e.g. full
    # finetuning, which fast_inference refuses) must still receive the trainer weights every step.
    match = re.search(
        r"^[ \t]*def _move_model_to_vllm\(.*?\).*?:[ \t]*\n",
        function,
        re.MULTILINE | re.DOTALL,
    )
    body_indent = (
        re.search(r"^([ \t]+)\S", function[match.end() :], re.MULTILINE) if match else None
    )
    if match is None or body_indent is None:

        def _move_model_to_vllm(self, *args, **kwargs):
            return None

        return inspect.getsource(_move_model_to_vllm)
    indent = body_indent.group(1)
    # Only the fast_inference engine shares weights; every other engine must get the trained weights:
    # - Dense: TRL's own body.
    # - Plain LoRA on a 16-bit base: merged weights under the base-layer names, as TRL does, but
    #   computed out of place. merge_adapter + unmerge_adapter on the live model is a lossy round
    #   trip in fp16 / bf16, so the frozen base would drift every sync.
    # - Other mergeable PEFT (LoHa, LoKr, IA3, VeRA, DoRA, lora_bias, target_parameters): PEFT's own
    #   merge run on copies of the base tensors, so the live base keeps its exact bytes.
    # - ZeRO-3: TRL's body (its gathered copy is discarded on exit). FSDP with PEFT: raise.
    # - Quantized bases (bitsandbytes, GPTQ, AWQ, HQQ, FP8, ...), unmergeable adapters (aLoRA, Arrow,
    #   trainable tokens, prompt learning): raise instead of sampling from the initial weights all run.
    guard = (
        f"{indent}if getattr(getattr(self, 'llm', None), 'shared_weights', False):\n"
        f"{indent}    return None\n"
        f"{indent}from unsloth.models.rl_replacements import _unsloth_vllm_sync_kind_cached\n"
        f"{indent}_unsloth_vllm_sync_kind_cached(self)\n"
        f"{indent}if self._unsloth_vllm_sync_kind == 'skip':\n"
        f"{indent}    return None\n"
        f"{indent}if self._unsloth_vllm_sync_kind == 'peft_merged':\n"
        f"{indent}    from unsloth.models.rl_replacements import _unsloth_vllm_sync_peft_merged\n"
        f"{indent}    return _unsloth_vllm_sync_peft_merged(self)\n"
    )
    return function[: match.end()] + guard + function[match.end() :]


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


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
            # Older TRL calls this instead of _get_per_token_logps_and_entropies: latch here too (see below).
            _unsloth_grpo_autocast_unscored(self)
            return None  # Unsloth efficient GRPO
        _unsloth_grpo_autocast(self)

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(
            device_type = DEVICE_TYPE_TORCH,
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
            # transformers <= 4.48 does not support logits_to_keep, so drop the logits here; see huggingface/trl#2770.

    function = inspect.getsource(_get_per_token_logps)
    return function


RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)


def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
    if function_name != "_get_per_token_logps_and_entropies":
        return function

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
            # No scoring forward may run (vLLM, beta = 0): latch precision here, not from the shared env.
            _unsloth_grpo_autocast_unscored(self)
            # The current policy is then scored only in the zoo loss: refuse the kept set there too.
            _unsloth_grpo_refuse_unscored_support(self)
            return None, None
        else:
            _unsloth_grpo_autocast(self)

            compute_aux_loss = kwargs.get("compute_aux_loss", None)

            # Body-local: this source is copied out without this module's imports. #6960.
            _grpo_vision_chunks = None
            try:
                from unsloth_zoo.rl_replacements import grpo_vision_chunks as _grpo_vision_chunks
            except Exception:
                pass
            # Collected even without the zoo: an older one must cost only image slicing.
            vision_inputs = _unsloth_grpo_vision_inputs(kwargs)
            if _grpo_vision_chunks is None and vision_inputs.get("pixel_values", None) is not None:
                raise RuntimeError(
                    "Unsloth: vision GRPO needs an unsloth_zoo build that exports "
                    "grpo_vision_chunks, the shared multimodal key tuple and chunker "
                    "used by both GRPO logprob paths. Please upgrade unsloth_zoo to "
                    "2026.9.5 or newer: pip install -U unsloth_zoo"
                )
            pixel_values = vision_inputs.get("pixel_values", None)
            image_grid_thw = vision_inputs.get("image_grid_thw", None)
            video_grid_thw = vision_inputs.get("video_grid_thw", None)
            if pixel_values is None:
                # Only a sentinel below (the chunks carry the tensors): video rows must take the
                # vision path too, since left and sequence packing move M-RoPE video positions.
                pixel_values = vision_inputs.get("pixel_values_videos", None)
            num_images = vision_inputs.get("num_images", None)
            # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
            token_type_ids = vision_inputs.get("token_type_ids", None)
            mm_token_type_ids = vision_inputs.get("mm_token_type_ids", None)
            if (
                mm_token_type_ids is not None
                or image_grid_thw is not None
                or video_grid_thw is not None
            ):
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class, input_ids, mm_token_type_ids
                )
                vision_inputs["mm_token_type_ids"] = mm_token_type_ids

            unwrapped_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper = False)

            from unsloth_zoo.rl_replacements import (
                chunked_hidden_states_selective_log_softmax,
                chunked_selective_log_softmax,
            )

            # Old / ref logps must use the policy pass's scorer, else the ratio starts away from 1 and the KL is biased.
            try:
                if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
                    from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
                    if not grpo_truncated_support_kwargs(self):
                        from unsloth_zoo.rl_replacements import (
                            chunked_hidden_states_selective_log_softmax_recompute as chunked_hidden_states_selective_log_softmax,
                        )
                else:
                    from unsloth_zoo.rl_replacements import (
                        grpo_chunked_hidden_states_logprobs as chunked_hidden_states_selective_log_softmax,
                    )
            except ImportError:
                pass

            output_embeddings = unwrapped_model.get_output_embeddings()
            lm_head = output_embeddings.weight
            lm_head_bias = getattr(output_embeddings, "bias", None)
            if lm_head_bias is not None and not UNSLOTH_ZOO_HAS_HEAD_BIAS:
                import warnings
                warnings.warn(
                    "Unsloth: this model's output head has a bias, which GRPO scoring needs "
                    "a newer unsloth_zoo to include. Please upgrade: pip install -U unsloth_zoo",
                    stacklevel = 2,
                )
                lm_head_bias = None
            _head_log_softmax = chunked_hidden_states_selective_log_softmax
            # ZeRO-3 released the head after the forward: score with one gathered read-only copy.
            _zero3_head = None
            try:
                from unsloth_zoo.fused_losses.cross_entropy_loss import (
                    unsloth_zero3_gathered,
                    unsloth_zero3_partitioned,
                )
            except (ImportError, NotImplementedError):
                # Older zoo, or zoo refusing to import without an accelerator (CPU gloo workers).
                pass
            else:
                if unsloth_zero3_partitioned(lm_head) or unsloth_zero3_partitioned(
                    getattr(output_embeddings, "bias", None)
                ):
                    _zero3_head = (
                        unsloth_zero3_gathered(lm_head),
                        unsloth_zero3_gathered(getattr(output_embeddings, "bias", None)),
                    )
                    lm_head = _zero3_head[0]
                    if lm_head_bias is not None:
                        lm_head_bias = _zero3_head[1]

            def chunked_hidden_states_selective_log_softmax(
                hidden_states, _lm_head, *args, **kwargs
            ):
                # FSDP2's forward hooks replace Parameters. Read the current head
                # at projection time, including the bias, instead of retaining a shard.
                if _zero3_head is not None:
                    head_weight, head_bias = _zero3_head
                else:
                    head_weight = output_embeddings.weight
                    head_bias = getattr(output_embeddings, "bias", None)
                if UNSLOTH_ZOO_HAS_HEAD_BIAS:
                    kwargs["lm_head_bias"] = head_bias
                return _head_log_softmax(
                    hidden_states,
                    head_weight,
                    *args,
                    **kwargs,
                )

            # UNSLOTH_GRPO_TRUNCATED_SUPPORT (=1 or zoo's automatic gate): score over the rollout's
            # top_k / top_p / min_p kept set, matching vLLM's processed_logprobs. The zoo loss refuses to mix
            # normalisations unless this marker says the old / ref logps were scored the same way.
            try:
                from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
                _support_kwargs = grpo_truncated_support_kwargs(self)
            except Exception:
                _support_kwargs = {}
            _support_mismatch = _unsloth_grpo_support_mismatch(self) if _support_kwargs else []
            if _support_mismatch:
                # Left unmarked, the zoo loss drops the kept set for the current policy as well.
                if not getattr(self, "_unsloth_truncated_support_refused", False):
                    import warnings
                    warnings.warn(
                        "Unsloth: not scoring over the rollouts' top_k / top_p / min_p kept set "
                        "(UNSLOTH_GRPO_TRUNCATED_SUPPORT=1, or automatic for truncated vLLM "
                        "rollouts feeding the IS correction or off-policy mask) because "
                        f"{', '.join(_support_mismatch)}: those logprobs come from weights other "
                        "than the rollout's, whose top_k / top_p / min_p kept set differs from the "
                        "sampler's, so per-policy kept sets give unnormalised PPO ratios and KL. "
                        "Scoring over the full vocabulary instead.",
                        stacklevel = 2,
                    )
                    # Also silences the zoo's refusal, which would blame the unsloth version.
                    self._unsloth_truncated_support_refused = True
                _support_kwargs = {}
            if _support_kwargs:
                from functools import partial

                chunked_hidden_states_selective_log_softmax = partial(
                    chunked_hidden_states_selective_log_softmax,
                    **_support_kwargs,
                )
                chunked_selective_log_softmax = partial(
                    chunked_selective_log_softmax,
                    **_support_kwargs,
                )
                self._unsloth_truncated_support_scored = True

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

            import math

            total_samples = input_ids.shape[0]
            batch_size = math.ceil(total_samples / B)

            input_ids_chunks = []
            attention_mask_chunks = []
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                input_ids_chunks.append(input_ids[start:end])
                attention_mask_chunks.append(attention_mask[start:end])

            # One chunker shared with the gradient pass, so the two cannot disagree.
            if _grpo_vision_chunks is None:
                # Image-indexed keys already raised above, so only per-sample ones are left.
                vision_chunks = []
                for _start in range(0, total_samples, batch_size):
                    _end = min(_start + batch_size, total_samples)
                    vision_chunks.append(
                        {
                            _key: vision_inputs[_key][_start:_end]
                            for _key in ("token_type_ids", "mm_token_type_ids")
                            if vision_inputs.get(_key, None) is not None
                        }
                    )
            else:
                vision_chunks = _grpo_vision_chunks(vision_inputs, total_samples, batch_size)

            temperature = self.temperature
            model_config = _unsloth_get_model_config(model)
            if detect_logit_transforms is not None:
                # model_config, not model: under DDP/Accelerate `model` is a wrapper that does not forward .config, so the helper would report zeros.
                _transforms = detect_logit_transforms(model_config)
                logit_softcapping = _transforms["logit_softcapping"]
                logit_scale_multiply = _transforms["logit_scale_multiply"]
                logit_scale_divide = _transforms["logit_scale_divide"]
            else:
                logit_softcapping = _unsloth_get_final_logit_softcapping(model)
                logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(
                    model_config
                )

            zipped_inputs = zip(
                input_ids_chunks,
                attention_mask_chunks,
                vision_chunks,
            )
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen [1, sum L] forward replaces the padded [B, Lmax] loop and fixes the left-pad RoPE error. Self-verified against the per-row forward, re-checked as T grows, and falls back if a backend ignores packed_seq_lengths.
            logprobs = None

            # PrefixGrouper (GRPO shared-prompt dedup, default ON): G completions share the prompt, so storing it once behind a FlexAttention shared-prefix mask cuts the trunk forward from G*(P+R) to P+G*R tokens. Gated by UNSLOTH_GRPO_PREFIX_GROUPER, a tok_r auto-gate and a first-use self-verify, so a mask/isolation regression cannot ship silently.
            _pg_result = None
            _pg_use = False
            _pg_skip_pk = False  # once a shape is PG-verified, skip the full-row forward
            _pg_forward_fn = None  # deferred PG forward (runs at the verify site below)
            _pg_num_gen = getattr(self, "num_generations", None)
            # Env gate hoisted to module level (mirrored via RL_PRE_ITEMS). Skip PG under vLLM: the rollout dominates the step, so PG saves little and its self-verify is net overhead.
            _pg_engage = (
                UNSLOTH_GRPO_PREFIX_GROUPER_ON
                and not getattr(self, "use_vllm", False)
                and not getattr(unwrapped_model, "_unsloth_prefix_grouper_nograd_disabled", False)
            )
            if _pg_engage:
                try:
                    # Skip softcap models (the flex kernel never applies attn_logit_softcapping) and hybrid SSM / MoE models: only the threaded attention forwards get shared-prefix isolation, so a decoder that does not forward prefix_seg_info leaks suffixes across completions. PG also rides on sequence packing, so it needs the same zoo masked-column guard.
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
                        # Normal backends apply config.attention_dropout in training; the flex path is deterministic, so skip PG when it is set.
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
                                    device_type = DEVICE_TYPE_TORCH,
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

                        # Trust only within the verified envelope: re-verify when T or the longest segment grows, like the packed path.
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
                            # Unverified shape: defer the forward until the packed reference exists, so a declined packed path never wastes a whole-batch PG forward.
                            _pg_forward_fn = _pg_run_forward
                except Exception as _pg_err:
                    _pg_result = None
                    _pg_use = False
                    _pg_skip_pk = False
                    _pg_forward_fn = None
                    # A FlexAttention/Triton compile failure or OOM here is GPU-wide, not layout-specific, so retrying every step just re-pays it. Disable PG persistently; the packed/padded path below still gives the exact result.
                    unwrapped_model._unsloth_prefix_grouper_nograd_disabled = True
                    if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO PrefixGrouper (no-grad) disabled (fell back to packed): {_pg_err!r}",
                            flush = True,
                        )

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen block-diagonal forward replaces the padded loop exactly and fixes its left-pad RoPE error. Self-verified, re-checked as T grows, falls back if a backend ignores packed_seq_lengths, and lm_head runs on completion positions only.
            _pk_result = None
            _pk_use = False
            _pk_T = None
            # True only while the packed forward and its scatter run, so an OOM elsewhere (setup, per-row verifier) is not read as a length limit.
            _pk_in_packed = False
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
                    _pk_T = sum(_pk_nz_cpu)
                    _pk_L = input_ids.shape[1]
                    _pk_W = logits_to_keep + max_left_pad
                    _pk_maxseg = max(_pk_nz_cpu) if _pk_nz_cpu else 0
                    # Sliding-window models lose the per-sequence local window in a packed stream.
                    _pk_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    _pk_sw_ok = not (isinstance(_pk_sw, int) and _pk_sw > 0 and _pk_maxseg > _pk_sw)
                    # Skip the packed forward entirely at known-unsafe lengths, avoiding a wasted pass or OOM.
                    _pk_unsafe = getattr(
                        unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                    )
                    # Cap the flattened forward at one padded [batch_size, seq_len] mini-batch's token budget; anything larger uses the chunked padded loop.
                    _pk_cap = batch_size * seq_len
                    # Decided from the host lengths alone, so a capped batch skips the flatten and the active-row sync below.
                    _pk_fits = (
                        _pk_T >= 2
                        and _pk_T <= _pk_cap
                        and len(_pk_nz_cpu) > 0
                        and _pk_sw_ok
                        and not (_pk_unsafe is not None and _pk_T >= _pk_unsafe)
                    )
                    if _pk_fits:
                        _pk_flat = input_ids[_pk_keep].unsqueeze(0)
                        # Per-row completion mask (same as the loss); prompt-only rows count as inactive.
                        _pk_cmask = create_completion_attention_mask(
                            input_ids[:, -_pk_W:], left_pad_tokens_per_prompt, max_left_pad, _pk_pad
                        )
                        _pk_active = int(_pk_cmask.any(dim = 1).sum())
                    if _pk_fits and (_pk_ok is True or _pk_active >= 2):
                        _pk_pos = (_pk_keep.cumsum(dim = 1) - 1)[_pk_keep].unsqueeze(0)
                        _pk_chunks = max(1, total_rows * multiplier)
                        _pk_nz_idx = _pk_keep.nonzero(
                            as_tuple = False
                        )  # [T, 2] = (row, col), row-major
                        _pk_within = _pk_nz_idx[1:, 0] == _pk_nz_idx[:-1, 0]  # [T-1]
                        # Per-row completion start after left-packing, matching create_completion_attention_mask.
                        _pk_cstart = (_pk_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                        _pk_ctgt = (_pk_nz_idx[1:, 1] >= _pk_cstart[_pk_nz_idx[1:, 0]]) & _pk_within
                        _pk_in_packed = True
                        with _get_inference_mode_context_manager(model):
                            with torch.amp.autocast(
                                device_type = DEVICE_TYPE_TORCH,
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
                                    # Model returned logits directly: scaling/softcapping already applied by the model forward.
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
                        _pk_in_packed = False
                        # Re-verify when T or the longest segment grows past the verified envelope; a LongRoPE cache switch can change the result.
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
                                    device_type = DEVICE_TYPE_TORCH,
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
                                        # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
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
                                            # Model returned logits directly: scaling/softcapping already applied by the model forward.
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
                                # Widen the trusted shape only when at least 2 completion rows exercised cross-sample packing; a single row proves nothing.
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
                    # Any failure: drop intermediates and use the padded loop for this batch.
                    _pk_hidden = None
                    _pk_sel = None
                    _pk_result = None
                    _pk_use = False
                    if isinstance(_pk_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    if (
                        isinstance(_pk_err, torch.cuda.OutOfMemoryError)
                        and _pk_in_packed
                        and _pk_T is not None
                    ):
                        # An OOM in the packed forward says this flattened length does not fit, not that packing is wrong: keep packing smaller batches.
                        _pk_unsafe = getattr(
                            unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                        )
                        _pk_unsafe = _pk_T if _pk_unsafe is None else min(_pk_unsafe, _pk_T)
                        unwrapped_model._unsloth_seq_packing_nograd_unsafe_T = _pk_unsafe
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO sequence-packing (no-grad) OOM at T={_pk_T}: packing capped below T={_pk_unsafe}, this batch uses the padded loop: {_pk_err!r}",
                                flush = True,
                            )
                    else:
                        unwrapped_model._unsloth_seq_packing_nograd_ok = False
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO sequence-packing (no-grad) disabled (fell back to padded): {_pk_err!r}",
                                flush = True,
                            )
            # PrefixGrouper first-use self-verify (no-grad): compare the untrusted PG result to the packed result over the completion mask. Below tol_ok trust the structure, at or above TOL_KILL mark it unsafe forever, borderline falls back for this shape.
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
                _pk_hidden = _pk_sel = _pk_result = _pk_ref = None
                # Index buffers of a declined packed batch would otherwise live through the padded loop.
                _pk_keep = _pk_len = _pk_flat = _pk_pos = _pk_cmask = None
                _pk_nz_idx = _pk_within = _pk_cstart = _pk_ctgt = _pk_tgt = _pk_ids = _pk_cm = None

            with _get_inference_mode_context_manager(model):
                for (
                    input_ids_chunk,
                    attention_mask_chunk,
                    vision_chunk,
                ) in zipped_inputs:
                    with torch.amp.autocast(
                        device_type = DEVICE_TYPE_TORCH,
                        dtype = self._autocast_dtype,
                        enabled = getattr(self, "_autocast_enabled", True),
                    ):
                        if pixel_values is None:
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                **vision_chunk,
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
                                # Model returned logits directly: scaling/softcapping already applied by the model forward.
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                    input_ids_chunk.shape[0] * multiplier,
                                )
                        else:
                            # VLMs do not take the optimized path in models/, so they never hit the Flash Attn left-padding issue.
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                logits_to_keep = logits_to_keep + 1,
                                **vision_chunk,
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
            # aux loss is unused: off by default (router_aux_loss_coef = 0 in models/rl.py) and explicit opt-in is rejected at trainer init, so it is always None. Kept for TRL >= 1.7.0's 3-tuple.
            aux_loss = None
            return logprobs.detach(), entropies, aux_loss  # logps, entropies, aux_loss
            # transformers <= 4.48 does not support logits_to_keep, so drop the logits here; see huggingface/trl#2770.

    function = inspect.getsource(_get_per_token_logps_and_entropies)
    if trl_version < Version("1.7.0"):
        # TRL < 1.7.0 unpacks (logps, entropies) while TRL >= 1.7.0 unpacks (logps, entropies, aux_loss), so drop the third element to match. The regex tolerates comment/whitespace drift and fails loud rather than ship a 3-tuple to older TRL.
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


def _unsloth_text_configs(config):
    """``config`` and its text sub-config, the two places a transform can live: Gemma-4-style configs keep it on ``config.text_config``, T5Gemma-style ones only reach it via ``config.get_text_config()``."""
    if config is None:
        return []
    holders = [config]
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_cfg = get_text_config()
            except (TypeError, ValueError):
                text_cfg = None
    if text_cfg is not None and text_cfg is not config:
        holders.append(text_cfg)
    return holders


def _unsloth_get_final_logit_softcapping(model):
    """The soft cap the loss applies, under any of its three spellings: ``final_logit_softcapping`` (Gemma), ``logits_soft_cap`` (RecurrentGemma), ``output_logit_soft_cap`` (xLSTM). Returns 0 if unset."""
    config = _unsloth_get_model_config(model)
    if config is None:
        return 0
    for holder in _unsloth_text_configs(config):
        for name in ("final_logit_softcapping", "logits_soft_cap", "output_logit_soft_cap"):
            softcap = getattr(holder, name, None)
            if softcap:
                return softcap
    return 0


def _unsloth_resolve_logit_scales(model_config):
    """``(multiply, divide)`` for the logits, read with the same field table and per-family overrides ``detect_logit_transforms`` uses.

    Only reached when the installed unsloth_zoo predates that helper. Must stay in step with ``resolve_logit_transforms`` in unsloth/models/llama.py: if the forward applies a transform GRPO does not, the policy log-probabilities come from different logits than the ones generated, which silently shifts every importance ratio.
    """
    # Granite divides, HyperCLOVA X multiplies, MiniCPM3 scales hidden states (not a logit transform).
    overrides = {
        ("logits_scaling", "hyperclovax"): "multiply",
        ("logits_scaling", "minicpm3"): None,
    }
    found = {"multiply": 0, "divide": 0}
    for holder in _unsloth_text_configs(model_config):
        model_type = getattr(holder, "model_type", "") or ""
        for bucket, names in (
            ("multiply", ("logit_scale", "lm_head_multiplier", "output_multiplier")),
            ("divide", ("logits_scaling",)),
        ):
            for name in names:
                target = overrides.get((name, model_type), bucket)
                if target is None or found[target]:
                    continue
                value = getattr(holder, name, 0) or 0
                if value:
                    found[target] = value
                    break
    return found["multiply"], found["divide"]


def _unsloth_grpo_returns_hidden_states(model, tensor, lm_head):
    """Does ``tensor`` (a forward's ``.logits``) carry hidden states or real logits?

    ``_get_per_token_logps_and_entropies`` sets ``UNSLOTH_RETURN_HIDDEN_STATES=1``, but only a forward that honours the name hands hidden states back as ``.logits``; any other forward returns a real ``[.., vocab]`` tensor that must not reach the ``lm_head`` matmul.

    The primary test is an explicit signal that the forward honours the flag, both set outside this file: ``__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__`` on the generated class, written by ``unsloth_zoo.compiler.create_standalone_class`` exactly when ``apply_fused_lm_head`` gave that forward its own branch; and ``_unsloth_grpo_hidden_states_forward_wrapped``, set by ``_install_grpo_hidden_states_forward_wrapper`` in ``unsloth/models/rl.py`` for models the compiler did not rewrite. That wrapper degrades to real logits when the model cannot produce hidden states and records whether it did so in ``_unsloth_grpo_hidden_states_degraded`` before returning, so reading the pair after a forward describes the call that just finished; degradation is per call, not per model.

    The width comparison stays as the fallback, for an ``unsloth_zoo`` old enough that it never writes the marker. It is decisive whenever ``vocab_size != hidden_size``, and the signal may only overrule it when it is not, which is the one case the shape cannot answer.
    """
    if tensor.shape[-1] != lm_head.shape[1]:
        return False  # vocab-wide: real logits, whatever any signal claims
    if lm_head.shape[0] != lm_head.shape[1]:
        return True  # hidden-wide and vocab_size != hidden_size: hidden states
    return _unsloth_grpo_hidden_states_signal(model) is not False


def _unsloth_pref_returns_hidden_states(model, tensor, lm_head):
    """``_unsloth_grpo_returns_hidden_states`` for the preference routes, which need a zoo that writes the hidden-state
    markers: a square head with no signal is a forward that ignored the flag, so its output is real logits."""
    if tensor.shape[-1] != lm_head.shape[1]:
        return False
    if lm_head.shape[0] != lm_head.shape[1]:
        return True
    return _unsloth_grpo_hidden_states_signal(model) is True


def _unsloth_grpo_hidden_states_signal(model):
    """``True``/``False`` if the forward honours ``UNSLOTH_RETURN_HIDDEN_STATES``, ``None`` when neither marker is present. See ``_unsloth_grpo_returns_hidden_states`` for where each marker is set. Walks the wrapper chain because the markers are set on whichever object the trainer saw, which may be the DDP module or the PEFT base model rather than the object handed to the logprob loop."""
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
        # An unsloth/models/rl.py predating the per-call attribute set only the warn-once flag; it is the best signal such a wrapper offers.
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

# Deliberately loose: any branch header deciding something off an lm_head dimension. Used only to count how many the strict pattern should have rewritten, so a zoo respelling SOME of them is rejected rather than half-patched.
_GRPO_HIDDEN_STATES_WIDTH_DISPATCH_CANDIDATE = re.compile(
    r"^[ \t]*(?:el)?if[ \t]+[^\n]*\blm_head\.shape\[[^\]\n]+\][^\n]*:[ \t]*$",
    flags = re.MULTILINE,
)


def _patch_grpo_accumulated_loss_hidden_states_dispatch(function):
    """Give zoo's gradient path the same model-aware dispatch as the no-grad path. ``grpo_accumulated_loss`` is embedded into the generated trainer with ``inspect.getsource`` below, and zoo revisions with raw-logits support choose between the two log-softmax helpers by comparing the forward output width with ``lm_head.shape[1]``, which is ambiguous for a square head, so replace every such decision before embedding. The expression is intentionally limited to a named tensor's last dimension against the lm_head input dimension, so a zoo that changes that contract fails at trainer generation instead of silently restoring the wrong dispatch. Partial matches have to fail too: a zoo respelling only some of its dispatches would satisfy a "did anything match?" check while the respelled sites keep deciding on width alone, so count the branch headers deciding off an ``lm_head`` dimension before and after substituting and require that none survive."""
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


def grpo_update_SamplingParams(
    SamplingParams,
    generation_kwargs,
    vllm_sampling_params = None,
):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        overwrites = getattr(vllm_sampling_params, "_set_kwargs", None)
        if overwrites is None:
            default_sampling_params = SamplingParams()
            overwrites = {}
            for key in good_sampling_params_keys:
                if key.startswith("_") or not hasattr(vllm_sampling_params, key):
                    continue
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key != getattr(default_sampling_params, key, None):
                    overwrites[key] = overwrited_key
        for key, overwrited_key in overwrites.items():
            if key in good_sampling_params_keys and key not in (
                "seed",
                "n",
                "temperature",
                "max_tokens",
                "logprobs",
            ):
                generation_kwargs[key] = overwrited_key
    return generation_kwargs


def _unsloth_grpo_rollout_logps_as_old(
    trainer,
    sampling_per_token_logps,
    completion_mask,
    tool_mask = None,
):
    """UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1: use the rollout engine's logprobs (vLLM, or a
    rollout_func) as pi_old and skip the old-policy forward, as verl rollout_correction
    bypass_mode (loss_type ppo_clip) and vime --use-rollout-logprobs do. The PPO ratio is then
    pi_theta / pi_rollout and its clip (plus the dual clip `delta`) is the correction, so no
    vLLM importance weight is applied on top. True when this generation batch is marked."""
    import os

    if os.environ.get("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "0") != "1":
        return False

    def _skip(reason):
        trainer._unsloth_rollout_logps_as_old_fallbacks = (
            getattr(trainer, "_unsloth_rollout_logps_as_old_fallbacks", 0) + 1
        )
        warned = getattr(trainer, "_unsloth_rollout_logps_as_old_warned", None)
        if not isinstance(warned, set):
            warned = set()
            trainer._unsloth_rollout_logps_as_old_warned = warned
        if reason not in warned:
            warned.add(reason)
            import warnings
            warnings.warn(
                f"Unsloth: UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 is not applied to this batch: "
                f"{reason}. The trainer scores pi_old itself instead.",
                stacklevel = 2,
            )
        return False

    reason = None
    if sampling_per_token_logps is None:
        reason = "there are no rollout logprobs (needs use_vllm=True or a rollout_func)"
    elif getattr(trainer.args, "loss_type", None) == "vespo":
        reason = "loss_type='vespo' weights by the vLLM importance ratio itself"
    elif not hasattr(trainer.args, "loss_type"):
        reason = "this TRL version predates loss_type"
    else:
        try:
            from unsloth_zoo.rl_replacements import grpo_rollout_logps_as_old  # noqa: F401
        except Exception:
            reason = "it needs a newer unsloth_zoo: pip install -U unsloth_zoo"
    if reason is None:
        loss_mask = completion_mask.to(torch.bool)
        if tool_mask is not None:
            loss_mask = loss_mask & tool_mask.to(device = loss_mask.device, dtype = torch.bool)
        if not bool(torch.isfinite(sampling_per_token_logps[loss_mask]).all()):
            reason = "some rollout logprobs on loss tokens are not finite"
    accelerator = getattr(trainer, "accelerator", None)
    if getattr(accelerator, "num_processes", 1) > 1:
        # One reduction on every rank, before any of them returns: under FSDP / ZeRO-3 the
        # old-policy forward a fallback runs is collective, so all ranks take the same branch.
        eligible = torch.tensor(
            [int(reason is None)], dtype = torch.int32, device = completion_mask.device
        )
        if not bool(accelerator.gather(eligible).min()) and reason is None:
            reason = "another process cannot use its rollout logprobs for this batch"
    if reason is not None:
        return _skip(reason)
    trainer._unsloth_rollout_logps_as_old_batches = (
        getattr(trainer, "_unsloth_rollout_logps_as_old_batches", 0) + 1
    )
    return True


def _unsloth_grpo_sample_weight_mode():
    import os

    exclude = os.environ.get("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "0") == "1"
    aggregation = os.environ.get("UNSLOTH_GRPO_LOSS_AGGREGATION", "").strip().lower()
    if aggregation not in ("", "prompt"):
        raise ValueError(
            f"Unsloth: UNSLOTH_GRPO_LOSS_AGGREGATION must be 'prompt' or unset, got {aggregation}."
        )
    if not exclude and aggregation != "prompt":
        return None
    return exclude, aggregation == "prompt"


def _unsloth_trl_masks_unscorable():
    try:
        import trl
        from packaging.version import Version
        return Version(trl.__version__) >= Version("1.6.0")
    except Exception:
        return True


def _unsloth_grpo_sample_weight(
    trainer, mode, rewards_per_func, completion_mask, tool_mask, process_slice
):
    """Per-completion loss weights normalised over the whole generation batch (all processes),
    or None when off. ScaleRL (arXiv 2510.13786): zero-variance filtering drops groups whose
    rewards are all equal from the loss and its token count; prompt-level aggregation averages
    tokens within each group, then over groups. Training batches only."""
    settings = _unsloth_grpo_sample_weight_mode()
    if settings is None or mode != "train":
        return None
    exclude, prompt_level = settings
    loss_type = getattr(trainer.args, "loss_type", None)
    if (prompt_level and loss_type == "luspo") or (
        not prompt_level and loss_type not in ("bnpo", "dapo", "cispo", "vespo")
    ):
        if not getattr(trainer, "_unsloth_sample_weight_warned", False):
            import warnings
            warnings.warn(
                f"Unsloth: zero-variance exclusion alone needs a token-normalised loss_type "
                f"(bnpo, dapo, cispo, vespo) and prompt-level aggregation does not support "
                f"luspo; ignored for loss_type={loss_type}.",
                stacklevel = 2,
            )
            trainer._unsloth_sample_weight_warned = True
        return None
    num_generations = trainer.num_generations
    loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
    lengths = trainer.accelerator.gather(loss_mask.sum(dim = 1).to(torch.float32))

    def nan_variance(x, dim):
        # TRL's nanstd squared: Bessel corrected, nan (never "zero") with fewer than two values.
        centred = x - torch.nanmean(x, dim = dim, keepdim = True)
        count = (~torch.isnan(x)).sum(dim = dim, keepdim = True)
        variance = torch.nansum(centred * centred, dim = dim, keepdim = True) / (count - 1).clamp(min = 1)
        return torch.where(count > 1, variance, torch.full_like(variance, torch.nan))

    weights = trainer.reward_weights.to(rewards_per_func.device).unsqueeze(0)
    if getattr(trainer, "multi_objective_aggregation", None) == "normalize_then_sum":
        # TRL normalises each reward function within its group before the weighted sum.
        per_func = rewards_per_func.view(-1, num_generations, rewards_per_func.shape[1])
        per_func = (per_func - torch.nanmean(per_func, dim = 1, keepdim = True)) / (
            nan_variance(per_func, 1).sqrt() + 1e-4
        )
        rewards = (per_func.view_as(rewards_per_func) * weights).nansum(dim = 1)
    else:
        rewards = (rewards_per_func * weights).nansum(dim = 1)
    if _unsloth_trl_masks_unscorable():
        # TRL >= 1.6 leaves a completion no reward function scored out of its group's statistics;
        # older TRL scores it 0 (nansum), so its group has variance and real advantages there.
        rewards[torch.isnan(rewards_per_func).all(dim = 1)] = torch.nan
    variance = nan_variance(rewards.view(-1, num_generations), 1).squeeze(1)
    group_keep = torch.ones_like(variance, dtype = torch.bool)
    if exclude:
        # TRL's frac_reward_zero_std test, per group.
        group_keep = ~torch.isclose(variance.sqrt(), torch.zeros_like(variance))
    keep = group_keep.repeat_interleave(num_generations).to(torch.float32).to(lengths.device)
    if prompt_level:
        group_tokens = (lengths * keep).view(-1, num_generations).sum(dim = 1)
        groups = group_keep.sum().to(torch.float32).clamp(min = 1.0)
        weight = keep / (group_tokens.clamp(min = 1.0).repeat_interleave(num_generations) * groups)
    else:
        weight = keep / (lengths * keep).sum().clamp(min = 1.0)
    trainer._metrics[mode]["frac_reward_zero_std_excluded"].append(
        float(1.0 - group_keep.float().mean()) if exclude else 0.0
    )
    return weight[process_slice].to(completion_mask.device)


def _unsloth_grpo_is_metric_values(delta, flat_is_ratio, mask, sequence_level):
    # Inputs are zero-filled outside the mask; TRL reduces over masked tokens only, else min ratio reads 0.
    if mask is None or delta.shape != mask.shape or flat_is_ratio.shape != mask.shape:
        return delta.reshape(-1), flat_is_ratio.reshape(-1)
    keep = mask.to(torch.bool)
    if not sequence_level:
        return delta[keep], flat_is_ratio[keep]
    # A row with no kept tokens is exp(0) = 1, as in TRL.
    counts = keep.sum(dim = -1)
    per_row = torch.where(
        counts > 0,
        (flat_is_ratio * keep).sum(dim = -1) / counts.clamp(min = 1),
        torch.ones_like(counts, dtype = flat_is_ratio.dtype),
    )
    return delta[keep], per_row


def _unsloth_grpo_is_sequence_level(args_mode):
    # One IS ratio per sequence: TRL's sequence_* and unsloth_zoo's geometric_* (verl seq_mean_k1),
    # with the UNSLOTH_GRPO_VLLM_IS_MODE override unsloth_zoo's grpo_accumulated_loss applies.
    import os as _os
    mode = _os.environ.get("UNSLOTH_GRPO_VLLM_IS_MODE", "") or args_mode or "token_truncate"
    return str(mode).startswith(("sequence_", "geometric_"))


grpo_compute_loss = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
UnslothEfficientGRPO = RL_REPLACEMENTS["UnslothEfficientGRPO"]
grpo_accumulated_loss = RL_REPLACEMENTS["grpo_accumulated_loss"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast_forced_float32))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast_unscored))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_autocast_kwargs))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_support_mismatch))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_refuse_unscored_support))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_model_config))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_text_configs))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_final_logit_softcapping))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_resolve_logit_scales))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_returns_hidden_states))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_hidden_states_signal))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_get_mm_token_id))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_fix_mm_token_type_ids))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_accumulation_steps))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_is_metric_values))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_is_sequence_level))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_sample_weight_mode))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_rollout_logps_as_old))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_trl_masks_unscorable))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_sample_weight))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_vision_inputs))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_split_vision_by_sample))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_unsplit_vision))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_split_videos_by_sample))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_prompt_videos))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_has_video_key))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_clean_video_prompts))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_prompt_key))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_video_inputs))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_grpo_image_cell))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_reject_grpo_image_list))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_unsloth_clear_stateful_mrope))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(UnslothEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(
    _patch_grpo_accumulated_loss_hidden_states_dispatch(grpo_accumulated_loss)
)
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_update_SamplingParams))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(_get_inference_mode_context_manager))
# inspect.getsource inlines function bodies but not module imports, so constants the inlined grpo functions use (UNSLOTH_ENABLE_LOGGING) must be redefined in the generated cache.
RL_PRE_ITEMS["grpo_trainer"].append(
    "import os as _unsloth_os\n"
    "UNSLOTH_ENABLE_LOGGING = _unsloth_os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') in ('1', 'True', 'true')\n"
)
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
RL_PRE_ITEMS["grpo_trainer"].append(
    "try:\n"
    "    import inspect as _unsloth_inspect\n"
    "    from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as _unsloth_chunked_lsm\n"
    "    UNSLOTH_ZOO_HAS_HEAD_BIAS = 'lm_head_bias' in _unsloth_inspect.signature(_unsloth_chunked_lsm).parameters\n"
    "except Exception:\n"
    "    UNSLOTH_ZOO_HAS_HEAD_BIAS = False\n"
)
RL_PRE_ITEMS["grpo_trainer"].append(
    "_pg_build_layout = _pg_enabled_fn = _pg_verify_on = _pg_tol_ok = _PG_TOL_KILL = None\n"
    "UNSLOTH_GRPO_PREFIX_GROUPER_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_PREFIX_GROUPER', '1').lower() not in ('0', 'false', 'no', 'off')\n"
    "if UNSLOTH_GRPO_PREFIX_GROUPER_ON:\n"
    "    try:\n"
    "        from unsloth.utils.prefix_grouper import build_group_layout as _pg_build_layout, prefix_grouper_enabled as _pg_enabled_fn, verify_on as _pg_verify_on, tol_ok as _pg_tol_ok, TOL_KILL as _PG_TOL_KILL\n"
    "    except Exception:\n"
    "        UNSLOTH_GRPO_PREFIX_GROUPER_ON = False\n"
)
# getsource inlines the grpo bodies but not this file's imports, so the generated cache needs its own guarded import or detect_logit_transforms is a NameError.
RL_PRE_ITEMS["grpo_trainer"].append(
    "try:\n"
    "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
    "except Exception:\n"
    "    detect_logit_transforms = None\n"
)


def grpo_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss" or "_unsloth_trl_compute_loss" in function:
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
        _vision_inputs = _unsloth_grpo_vision_inputs(inputs)
        pixel_values = _vision_inputs.get("pixel_values", None)
        image_grid_thw = _vision_inputs.get("image_grid_thw", None)
        pixel_attention_mask = _vision_inputs.get("pixel_attention_mask", None)
        image_sizes = _vision_inputs.get("image_sizes", None)
        num_images = _vision_inputs.get("num_images", None)
        # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
        token_type_ids = _vision_inputs.get("token_type_ids", None)
        mm_token_type_ids = _vision_inputs.get("mm_token_type_ids", None)
        video_grid_thw = _vision_inputs.get("video_grid_thw", None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        tool_mask = inputs.get("tool_mask", None)
        _rollout_logps_as_old = inputs.get("unsloth_rollout_logps_as_old", None) is not None
        _loss_delta = getattr(self.args, "delta", None)
        if _rollout_logps_as_old and _loss_delta is None:
            # verl's dual-clip default (clip_ratio_c = 3.0).
            _loss_delta = 3.0
        current_gradient_accumulation_steps = _unsloth_grpo_accumulation_steps(self)
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1)
        if (
            mm_token_type_ids is not None
            or image_grid_thw is not None
            or video_grid_thw is not None
        ):
            mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                input_ids,
                mm_token_type_ids,
                completion_ids = completion_ids,
            )
            _vision_inputs["mm_token_type_ids"] = mm_token_type_ids
        # Only the keys the processor produced: an older zoo must not see unknown kwargs.
        _vision_inputs = {k: v for k, v in _vision_inputs.items() if v is not None}
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
        # KL divergence between model and reference: _prepare_inputs no longer returns reference log probs. See trl grpo_trainer.py#L1328.
        ref_logps = inputs.get("ref_per_token_logps", None)
        # x - x.detach() preserves gradients from x.
        advantages = inputs["advantages"]
        old_logps = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        model_config = _unsloth_get_model_config(model)
        # The old and reference logps come from _get_per_token_logps_and_entropies and the gradient logps from here, so both must read the transforms alike or the importance ratio compares two different policies.
        if detect_logit_transforms is not None:
            # model_config, not model: see _get_per_token_logps_and_entropies.
            _transforms = detect_logit_transforms(model_config)
            logit_softcapping = _transforms["logit_softcapping"]
            logit_scale_multiply = _transforms["logit_scale_multiply"]
            logit_scale_divide = _transforms["logit_scale_divide"]
        else:
            logit_softcapping = _unsloth_get_final_logit_softcapping(model)  # Gemma
            logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(model_config)

        max_left_pad = inputs.get("max_left_pad", 0)
        # importance_sampling_level is absent before TRL 0.20.0 (token level).
        importance_sampling_level = getattr(
            self,
            "importance_sampling_level",
            getattr(self.args, "importance_sampling_level", "token"),
        )
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
            if _rollout_logps_as_old:
                old_logps, sampling_per_token_logps = sampling_per_token_logps, None
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
                importance_sampling_level = importance_sampling_level,
                epsilon_low = self.epsilon_low,
                epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = _loss_delta,
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
            # With beta = 0 and num_iterations = 1 the no-grad path (and its zoo gate) never runs, and an older
            # grpo_accumulated_loss silently drops pixel_values. Body-local: copied out without imports (#6960).
            if pixel_values is not None and not getattr(
                self, "_unsloth_grpo_vision_zoo_checked", False
            ):
                _grpo_vision_chunks = None
                try:
                    from unsloth_zoo.rl_replacements import (
                        grpo_vision_chunks as _grpo_vision_chunks,
                    )
                except Exception:
                    pass
                if _grpo_vision_chunks is None:
                    raise RuntimeError(
                        "Unsloth: vision GRPO needs an unsloth_zoo build that exports "
                        "grpo_vision_chunks, the shared multimodal key tuple and chunker "
                        "used by both GRPO logprob paths. Please upgrade unsloth_zoo to "
                        "2026.9.5 or newer: pip install -U unsloth_zoo"
                    )
                self._unsloth_grpo_vision_zoo_checked = True

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
                    # Probe by import: the grep below cries "upgrade unsloth_zoo" falsely.
                    try:
                        from unsloth_zoo.rl_replacements import grpo_vision_chunks
                        _supports_num_images = grpo_vision_chunks is not None
                    except Exception:
                        pass
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
            _sample_weight = inputs.get("unsloth_sample_weight", None)
            if _sample_weight is not None:
                if "sample_weight" not in inspect.getsource(grpo_compute_loss):
                    raise RuntimeError(
                        "UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE / UNSLOTH_GRPO_LOSS_AGGREGATION need a "
                        "newer unsloth_zoo. Please upgrade unsloth_zoo."
                    )
                _grpo_accumulated_loss_kwargs["sample_weight"] = _sample_weight
            if tool_mask is not None:
                _grpo_accumulated_loss_kwargs["tool_mask"] = tool_mask
            if _rollout_logps_as_old:
                _grpo_accumulated_loss_kwargs["rollout_logps_as_old"] = True
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
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    importance_sampling_level = importance_sampling_level,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = _loss_delta,
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
                    **_vision_inputs,
                    **_grpo_accumulated_loss_kwargs,
                )
            else:
                # For backwards compatibility with trl 0.15.2 and maybe 0.17.
                loss, completion_length, mean_kl, coef_1, completion_mask = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
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
                    **_vision_inputs,
                    **_grpo_accumulated_loss_kwargs,
                )
        if "train" in self._metrics:
            # model.training like TRL; should_evaluate is False under eval_on_start and evaluate().
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        if delta is not None and (
            (self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False))
            or _rollout_logps_as_old
        ):
            delta, flat_is_ratio = _unsloth_grpo_is_metric_values(
                delta,
                flat_is_ratio,
                completion_mask,
                (not _rollout_logps_as_old)
                and _unsloth_grpo_is_sequence_level(
                    getattr(self.args, "vllm_importance_sampling_mode", None)
                ),
            )
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
            # a width-1 tensor is sequence-level only when the completion mask spans multiple tokens
            if x.shape[1] == 1 and completion_mask.shape[1] != 1:
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


# KTO shape mismatch: the Unsloth forward truncates input_ids while labels are untouched, and TRL 0.27.2+ _process_tokens truncates only completions, so an over-length prompt makes the model emit shorter logits than the labels expect.
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


# TRL 1.x dropped KTOTrainer.get_batch_logps and moved the math into _compute_logps / compute_ref_log_probs / _compute_kl_logps, which call selective_log_softmax on completion-only tokens. Same truncation hazard, so clamp logits/ids/mask to the shorter length (a no-op when equal).
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
    # Patch trl's openenv generate_rollout_completions to guard reload_weights when sharing weights with vLLM, and to call wake_up() untagged: TRL's wake_up(tags=["kv_cache"]) leaves is_sleeping=True at the executor, so Unsloth's generate wakes again and double create_and_maps mapped handles. Unsloth's CuMemAllocator.wake_up skips weights anyway.
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

    # TRL 0.29.1+ ships some openenv helpers as bytecode with no source, so inspect.getsource raises OSError; skip the rewrite rather than crash. The unmodified path keeps the duplicate reload_weights and the tagged wake_up, so openenv GRPO users may see redundant reloads.
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

    # wake_up() with no tags wakes everything and sets is_sleeping=False, preventing a double wake_up; Unsloth's allocator skips weights anyway.
    src = re.sub(r"\.wake_up\(tags=\[.*?\]\)", ".wake_up()", src)

    if original_src == src:
        logger.warning("Unsloth: Warning - regex did not match, patch may have failed")
        return

    local_ns = {}
    exec(compile(src, "<unsloth>", "exec"), openenv_utils.__dict__, local_ns)
    patched_func = local_ns[patch_target_name]

    # Patch the target function in utils; if the dispatcher was patched, also update the parent module alias.
    setattr(openenv_utils, patch_target_name, patched_func)
    if patch_target_name == "generate_rollout_completions":
        openenv.generate_rollout_completions = patched_func
    logger.info(f"Unsloth: Patched trl openenv {patch_target_name}")


RL_ADDITIONAL_FUNCTIONS["openenv"].append(openenv_vllm_reload_weights)


def _vllm_weight_sync_bucket_bytes():
    raw = os.environ.get("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", "1024")
    try:
        mb = float(raw)
    except ValueError:
        mb = 1024.0
    return int(mb * 2**20) if mb > 0 else 0


def _unsloth_vllm_sync_colocate(generation, vllm_model):
    """TRL's colocate sync calls load_weights once per tensor, and every call rebuilds vLLM's
    parameter lookup (~0.4 ms each). Hand over the same (name, tensor) references in a few calls:
    no copies, and each weight_loader runs on the same data, so vLLM's weights are bit-identical.
    UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB = 0 restores the per-tensor loop."""
    weights = generation._iter_named_params()
    cap = _vllm_weight_sync_bucket_bytes()
    state = getattr(getattr(generation, "accelerator", None), "state", None)
    # ZeRO-3 / FSDP yield gathered tensors that are only valid until the next step of the
    # iterator, and PEFT unmerges in place once it is exhausted: holding references is wrong there.
    unsafe = (
        state is None
        or getattr(state, "deepspeed_plugin", None) is not None
        or getattr(state, "fsdp_plugin", None) is not None
        or getattr(generation.model, "peft_config", None) is not None
    )
    if cap == 0 or unsafe:
        for name, param in weights:
            vllm_model.load_weights([(name, param)])
        return
    bucket, size = [], 0
    for name, param in weights:
        bucket.append((name, param))
        size += param.numel() * param.element_size()
        if size >= cap:
            vllm_model.load_weights(bucket)
            bucket, size = [], 0
    if bucket:
        vllm_model.load_weights(bucket)
    generation._unsloth_bucketed_weight_syncs = (
        getattr(generation, "_unsloth_bucketed_weight_syncs", 0) + 1
    )


def vllm_generation_init_patch():
    # trl moved vllm code to trl/generation/vllm_generation.py (commit 0eb66d8, 0.28.0+), which must be patched so it does not build a second vLLM instance when fast_inference has one.

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
            # Chain getattr so server mode, where self.llm is unset, does not raise AttributeError before the default kicks in.
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
        # TRL >= 1.10 colocate loop; older layouts stay per-tensor.
        colocate_loop = re.compile(
            r"^(?P<indent>[ \t]*)for name, param in self\._iter_named_params\(\):\n"
            r"(?P=indent)[ \t]+self\.llm\.llm_engine\.model_executor\.driver_worker\.model_runner\.model"
            r"\.load_weights\(\[\(name, param\)\]\)\n",
            re.MULTILINE,
        )
        patched_src, n_loops = colocate_loop.subn(
            lambda m: (
                f"{m.group('indent')}_unsloth_vllm_sync_colocate(self, "
                "self.llm.llm_engine.model_executor.driver_worker.model_runner.model)\n"
            ),
            patched_src,
            count = 1,
        )
        if n_loops:
            vllm_generation._unsloth_vllm_sync_colocate = _unsloth_vllm_sync_colocate
        return patched_src

    # `generate` is deliberately NOT source-patched: its two anchors drifted (TRL >= 1.x deleted the collective_rpc("reload_weights") call for self.sync_weights(), so the anchor matched 0 times, raised, and took the lora injection with it, making GRPO rollouts come from the BASE model), and the generate regex is not paren-balanced and mis-edits multi-line calls. Intercept on the vLLM engine instead of TRL's method body: self.llm is the vLLM LLM object in colocate mode in every release with VLLMGeneration, and generate / chat / collective_rpc are public stable APIs checked across vLLM 0.11.0-0.27.1. The override is scoped to one VLLMGeneration.generate call and undone in a finally.
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
                """Has the caller's own argument list already filled `lora_request`? A keyword `lora_request` that is not None is the caller's choice, so leave it; a keyword `lora_request = None` is not, since on a shared-weights engine that means base-model rollouts. The positional case has to be checked rather than assumed: `lora_request` is keyword-only on `LLM.generate` in every vLLM release from 0.11.0 to 0.27.1, but on `LLM.chat` it is positional-or-keyword and its index has already moved once (`tokenization_kwargs` landed in 0.18.0), so adding a keyword on top would be `TypeError: got multiple values`. Bind the real signature instead of counting arguments."""
                if kwargs.get("lora_request", None) is not None:
                    return True
                try:
                    positional = inspect.signature(bound).bind_partial(*args).arguments
                except (TypeError, ValueError):
                    # Unintrospectable callable (C extension, odd wrapper): the keyword check above is all we have, and injecting is the safe default.
                    return False
                return "lora_request" in positional

            def wrap_generation_call(bound):
                def unsloth_generation_call(*args, **kwargs):
                    # vLLM needs the adapter handed to it explicitly: the shared engine holds the BASE weights, and sync_weights is a no-op when sharing.
                    if load_lora is not None and not caller_already_bound_lora(bound, args, kwargs):
                        kwargs["lora_request"] = load_lora(lora_name, load_tensors = True)
                    return bound(*args, **kwargs)

                return unsloth_generation_call

            def wrap_collective_rpc(bound):
                def unsloth_collective_rpc(method, *args, **kwargs):
                    # The engine already shares the live training weights, so reload_weights would pull the ORIGINAL checkpoint back off disk.
                    if method == "reload_weights":
                        return None
                    return bound(method, *args, **kwargs)

                return unsloth_collective_rpc

            override("generate", wrap_generation_call)
            override("chat", wrap_generation_call)
            override("collective_rpc", wrap_collective_rpc)
            # TRL >= 0.28 builds `SamplingParams(**generation_kwargs)` inside this call, the one site the older-TRL source replacement in rl.py targets; route it through the same helper for this call only.
            user_sampling_params = getattr(self, "_unsloth_vllm_sampling_params", None)
            original_sampling_params = vllm_generation.__dict__.get("SamplingParams", None)
            if user_sampling_params is not None and original_sampling_params is not None:

                def unsloth_sampling_params(*sp_args, **sp_kwargs):
                    if sp_args:
                        return original_sampling_params(*sp_args, **sp_kwargs)
                    return original_sampling_params(
                        **grpo_update_SamplingParams(
                            original_sampling_params, sp_kwargs, user_sampling_params
                        )
                    )

                vllm_generation.SamplingParams = unsloth_sampling_params
            else:
                original_sampling_params = None
            try:
                return original_generate(self, *args, **kwargs)
            finally:
                if original_sampling_params is not None:
                    vllm_generation.SamplingParams = original_sampling_params
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
        # inspect.getsource / inspect.signature unwrap this, so drift detectors and other source-reading patches still see TRL's own generate.
        generate.__wrapped__ = original_generate
        setattr(generate, _UNSLOTH_GENERATE_WRAPPED, True)
        vllm_generation.VLLMGeneration.generate = generate
        return True

    # Snapshot before patching: a HALF-patched VLLMGeneration is worse than none, since _init_vllm plus sync_weights without the generate-side adapter injection means no weight sync AND no LoRA, i.e. base-model rollouts with no error. If one of the three fails, restore all three.
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


# GKD chunked JSD (#11554): anything not reproduced exactly returns None and TRL's own compute_loss runs.
try:
    from unsloth_zoo.rl_replacements import distillation_chunked_jsd
except Exception:
    distillation_chunked_jsd = None
try:
    from unsloth_zoo.fused_losses.fused_linear import fused_linear_jsd as _unsloth_fused_linear_jsd
except Exception:
    _unsloth_fused_linear_jsd = None


def _unsloth_gkd_canonical(source):
    """``ast.unparse`` without docstrings/comments so checks survive reformatting; ``None`` if unparsable."""
    import ast as _ast
    import textwrap as _textwrap

    try:
        tree = _ast.parse(_textwrap.dedent(source))
    except Exception:
        return None
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, _ast.Expr)
                and isinstance(getattr(first, "value", None), _ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [_ast.Pass()]
    try:
        return _ast.unparse(tree)
    except Exception:
        return None


def _unsloth_gkd_layout(source):
    """Which TRL ``GKDTrainer.compute_loss`` this is, or ``None`` when it is not one we reproduce exactly.
    ``prompt`` (TRL < 1.7) or ``shift`` (TRL >= 1.7); any extra input, loss knob or term must fall back.
    """
    source = _unsloth_gkd_canonical(source)
    if source is None:
        return None
    call_args = "(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])"
    if len(re.findall(r"(?<![\w.])model" + re.escape(call_args), source)) != 1:
        return None
    if source.count("self.teacher_model" + call_args) != 1:
        return None
    if "return (loss, student_outputs) if return_outputs else loss" not in source:
        return None
    calls = re.findall(
        r"self\.generalized_jsd_loss\(student_logits=shifted_student_logits, "
        r"teacher_logits=shifted_teacher_logits, labels=shifted_labels, beta=self\.beta"
        r"(, num_items_in_batch=num_items_in_batch)?\)",
        source,
    )
    if len(calls) != 1 or source.count("self.generalized_jsd_loss(") != 1:
        return None
    prompt_layout = (
        "prompt_lengths = inputs['prompts'].shape[1]",
        "shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1:-1, :]",
        "shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1:-1, :]",
        "shifted_labels = inputs['labels'][:, prompt_lengths:]",
    )
    shift_layout = (
        "shifted_student_logits = student_outputs.logits[:, :-1, :]",
        "shifted_teacher_logits = teacher_outputs.logits[:, :-1, :]",
        "shifted_labels = inputs['labels'][:, 1:]",
    )
    is_prompt = all(x in source for x in prompt_layout)
    is_shift = all(x in source for x in shift_layout)
    if is_prompt == is_shift:
        return None
    return {"shift": "prompt" if is_prompt else "shift", "num_items_in_batch": bool(calls[0])}


def _unsloth_gkd_dense_head(model):
    """The output head when it is a plain dense ``[vocab, hidden]`` projection the chunked loss can read directly.
    Rejects bnb heads (``nn.Linear`` subclasses with packed weights), PEFT heads and ZeRO-3 partitioned weights.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_output_embeddings):
        return None
    try:
        head = get_output_embeddings()
    except Exception:
        return None
    if type(head) is not torch.nn.Linear:
        return None
    # An lm_head that is not the output embeddings runs before them (ModernBERT decoder, RoBERTa-style heads).
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module) and lm_head is not head:
        return None
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.numel() == 0:
        return None
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return None
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return None
    bias = getattr(head, "bias", None)
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        return None
    if weight.is_meta or (bias is not None and bias.is_meta):
        return None
    return head


def _unsloth_gkd_logit_transforms(model):
    """``(scale, softcap)`` exactly as the model's own forward applies them to its logits."""
    model_config = _unsloth_get_model_config(model)
    if detect_logit_transforms is not None:
        _transforms = detect_logit_transforms(model_config)
        logit_softcapping = _transforms["logit_softcapping"]
        logit_scale_multiply = _transforms["logit_scale_multiply"]
        logit_scale_divide = _transforms["logit_scale_divide"]
    else:
        logit_softcapping = _unsloth_get_final_logit_softcapping(model)
        logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(model_config)
    scale = 1.0
    if logit_scale_multiply:
        scale = scale * float(logit_scale_multiply)
    if logit_scale_divide:
        scale = scale / float(logit_scale_divide)
    return scale, float(logit_softcapping or 0.0)


def _unsloth_gkd_project(hidden_states, head, scale, softcap):
    """Full logits from hidden states, for the rare call where only one of the two forwards honoured the flag."""
    hidden_states = hidden_states.to(device = head.weight.device, dtype = head.weight.dtype)
    logits = torch.nn.functional.linear(hidden_states, head.weight, head.bias)
    if scale != 1.0:
        logits = logits * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _unsloth_gkd_jsd_supported(trainer_class):
    """Is ``trainer_class.generalized_jsd_loss`` one of TRL's own generalized JSDs, unchanged? Overrides keep the dense loss."""
    cache = trainer_class.__dict__.get("_unsloth_gkd_jsd_supported_cache", None)
    if cache is not None:
        return cache
    required = (
        "student_logits = student_logits / temperature",
        "teacher_logits = teacher_logits / temperature",
        "student_log_probs = F.log_softmax(student_logits, dim=-1)",
        "teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)",
        "if beta == 0:",
        "jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction='none', log_target=True)",
        "elif beta == 1:",
        "jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction='none', log_target=True)",
        "kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction='none', log_target=True)",
        "kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction='none', log_target=True)",
        "jsd = beta * kl_teacher + (1 - beta) * kl_student",
        "mask = labels != -100",
        "jsd = jsd[mask]",
        "if reduction == 'batchmean':",
    )
    known = set(required) | {
        "@staticmethod",
        "def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean'):",
        "def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean', num_items_in_batch=None):",
        "else:",
        "if labels is not None:",
        "beta = torch.tensor(beta, dtype=student_log_probs.dtype)",
        "beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)",
        "mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]), dim=0)",
        "mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)",
        "if num_items_in_batch is not None:",
        "jsd_sum = jsd.sum()",
        "if isinstance(num_items_in_batch, torch.Tensor):",
        "num_items_in_batch = num_items_in_batch.to(jsd_sum.device)",
        "return jsd_sum / num_items_in_batch",
        "return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))",
        "return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)",
        "denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)",
        "return jsd.sum() / denom",
        "elif reduction == 'sum':",
        "return jsd.sum()",
        "elif reduction == 'mean':",
        "return jsd.mean()",
        "return jsd",
    }
    ok = False
    try:
        source = _unsloth_gkd_canonical(inspect.getsource(trainer_class.generalized_jsd_loss))
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        signature = next((line for line in lines if line.startswith("def ")), "")
        ok = (
            all(line in known for line in lines)
            and all(line in lines for line in required)
            # a num_items_in_batch parameter must come with its reduction, and vice versa
            and (
                ("num_items_in_batch=None" in signature)
                == ("return jsd_sum / num_items_in_batch" in lines)
            )
        )
    except Exception:
        ok = False
    try:
        setattr(trainer_class, "_unsloth_gkd_jsd_supported_cache", ok)
    except Exception:
        pass
    return ok


def _unsloth_gkd_note_fallback(trainer, reason):
    """Count why the chunked path declined (``trainer._unsloth_gkd_chunked_fallbacks``), log each reason once."""
    try:
        fallbacks = trainer.__dict__.setdefault("_unsloth_gkd_chunked_fallbacks", {})
    except Exception:
        return None
    if reason not in fallbacks:
        try:
            from unsloth_zoo.log import logger as _gkd_logger
            _gkd_logger.info(
                f"Unsloth: GKD chunked JSD not used ({reason}); using TRL's dense loss."
            )
        except Exception:
            pass
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return None


def _unsloth_gkd_chunk_size(vocab_size):
    """Rows per chunk: ``UNSLOTH_GKD_CHUNK_SIZE`` when set, else about 2**26 logits per chunk, bounded to [64, 1024]."""
    requested = os.environ.get("UNSLOTH_GKD_CHUNK_SIZE", "")
    if requested.strip().isdigit() and int(requested) > 0:
        return int(requested)
    rows = (1 << 26) // max(int(vocab_size), 1)
    return int(min(1024, max(64, 1 << (max(rows, 1).bit_length() - 1))))


def _unsloth_gkd_right_align(
    inputs,
    layout,
    liger = False,
):
    """Roll left-padded rows (TRL's ChatML collator and generate both left-pad) so they start at column 0: Unsloth's training forward drops the 2D mask (#11885).
    The prompt layout's ``[:, P:]`` slice becomes ``labels[:, :P] = -100`` plus a one-column ``prompts``; TRL's Liger branch scores every label with no slice, so it only rolls.
    """
    try:
        attention_mask = inputs["attention_mask"]
        left_pad = (attention_mask.cumsum(dim = 1) == 0).sum(dim = 1, keepdim = True)
    except Exception:
        return inputs
    if not bool(left_pad.any()):
        return inputs
    aligned = dict(inputs)
    labels = inputs["labels"]
    if layout["shift"] == "prompt" and not liger:
        labels = labels.clone()
        labels[:, : inputs["prompts"].shape[1]] = -100
        aligned["prompts"] = inputs["prompts"][:, :1]
    width = attention_mask.shape[1]
    index = (torch.arange(width, device = attention_mask.device).unsqueeze(0) + left_pad) % width
    for key, value in (
        ("input_ids", inputs["input_ids"]),
        ("attention_mask", attention_mask),
        ("labels", labels),
    ):
        aligned[key] = value.gather(1, index)
    return aligned


def _unsloth_gkd_chunked_loss(self, model, inputs, num_items_in_batch, layout):
    """TRL's GKD loss without either full logits tensor, or ``None`` to run TRL's own ``compute_loss``."""
    if layout is None:
        return None
    if distillation_chunked_jsd is None and _unsloth_fused_linear_jsd is None:
        return _unsloth_gkd_note_fallback(self, "unsloth_zoo has no distillation_chunked_jsd")
    if os.environ.get("UNSLOTH_GKD_CHUNKED", "1").lower() in ("0", "false", "no", "off"):
        return _unsloth_gkd_note_fallback(self, "UNSLOTH_GKD_CHUNKED=0")
    # Read per call like the other fused lm_head routes: UNSLOTH_FUSED_LINEAR=0 restores TRL's dense loss.
    if os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        return _unsloth_gkd_note_fallback(self, "UNSLOTH_FUSED_LINEAR=0")
    use_fused = _unsloth_fused_linear_jsd is not None
    if getattr(self, "use_liger_gkd_loss", False):
        return _unsloth_gkd_note_fallback(self, "use_liger_gkd_loss")
    if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        return _unsloth_gkd_note_fallback(self, "FSDP / DeepSpeed")
    jsd_kwargs = {}
    scaler = getattr(self.accelerator, "scaler", None)
    if scaler is not None and scaler.is_enabled() and not use_fused:
        if "scaler" not in inspect.signature(distillation_chunked_jsd).parameters:
            return _unsloth_gkd_note_fallback(self, "unsloth_zoo JSD has no GradScaler support")
        jsd_kwargs["scaler"] = scaler
    try:
        missing = any(k not in inputs for k in ("input_ids", "attention_mask", "labels"))
    except Exception:
        missing = True
    if missing:
        return _unsloth_gkd_note_fallback(self, "inputs")
    if layout["shift"] == "prompt" and "prompts" not in inputs:
        return _unsloth_gkd_note_fallback(self, "inputs")
    teacher_model = getattr(self, "teacher_model", None)
    if teacher_model is None:
        return _unsloth_gkd_note_fallback(self, "no teacher_model")
    if not _unsloth_gkd_jsd_supported(type(self)):
        return _unsloth_gkd_note_fallback(self, "generalized_jsd_loss is not TRL's")
    try:
        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(teacher_model)
    except Exception:
        return _unsloth_gkd_note_fallback(self, "unwrap_model")
    student_head = _unsloth_gkd_dense_head(unwrapped_student)
    teacher_head = _unsloth_gkd_dense_head(unwrapped_teacher)
    if student_head is None or teacher_head is None:
        return _unsloth_gkd_note_fallback(self, "output head is not a dense nn.Linear")
    if student_head.weight.shape[0] != teacher_head.weight.shape[0]:
        return _unsloth_gkd_note_fallback(self, "vocab mismatch")
    # MiniCPM3 divides hidden states by logits_scaling before lm_head; a wrapped forward's hidden_states[-1] predates it.
    for unwrapped in (unwrapped_student, unwrapped_teacher):
        for config in _unsloth_text_configs(_unsloth_get_model_config(unwrapped)):
            if getattr(config, "model_type", None) == "minicpm3":
                return _unsloth_gkd_note_fallback(
                    self, "minicpm3 scales hidden states before the head"
                )
    # DDP(find_unused_parameters=True) marks a head skipped in forward as unused, then its grad hook fires twice.
    if getattr(model, "find_unused_parameters", False) and any(
        p is not None and p.requires_grad for p in (student_head.weight, student_head.bias)
    ):
        return _unsloth_gkd_note_fallback(self, "DDP find_unused_parameters with a trainable head")

    prior_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        student_outputs = model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
            )
    finally:
        if prior_hidden_states is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior_hidden_states

    student_states = student_outputs.logits
    teacher_states = teacher_outputs.logits
    if layout["shift"] == "prompt":
        prompt_lengths = inputs["prompts"].shape[1]
        student_states = student_states[:, prompt_lengths - 1 : -1, :]
        teacher_states = teacher_states[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]
    else:
        student_states = student_states[:, :-1, :]
        teacher_states = teacher_states[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]

    student_scale, student_softcap = _unsloth_gkd_logit_transforms(unwrapped_student)
    teacher_scale, teacher_softcap = _unsloth_gkd_logit_transforms(unwrapped_teacher)
    items = num_items_in_batch if layout["num_items_in_batch"] else None

    student_hidden = _unsloth_grpo_returns_hidden_states(
        unwrapped_student, student_states, student_head.weight
    )
    teacher_hidden = _unsloth_grpo_returns_hidden_states(
        unwrapped_teacher, teacher_states, teacher_head.weight
    )
    if not (student_hidden and teacher_hidden):
        if student_hidden:
            student_states = _unsloth_gkd_project(
                student_states, student_head, student_scale, student_softcap
            )
        if teacher_hidden:
            with torch.no_grad():
                teacher_states = _unsloth_gkd_project(
                    teacher_states, teacher_head, teacher_scale, teacher_softcap
                )
        _unsloth_gkd_note_fallback(self, "a forward returned logits, not hidden states")
        # TRL's dense path gets both logits on the input device (accelerate's top-level hook), beside the labels.
        student_states = student_states.to(shifted_labels.device)
        teacher_states = teacher_states.to(shifted_labels.device)
        extra = {}
        if layout["num_items_in_batch"]:
            extra["num_items_in_batch"] = num_items_in_batch
        return self.generalized_jsd_loss(
            student_logits = student_states,
            teacher_logits = teacher_states,
            labels = shifted_labels,
            beta = self.beta,
            **extra,
        )

    # A dispatched model can return hidden states off its head's device; the zoo's accumulators follow the states.
    student_states = student_states.to(student_head.weight.device)
    teacher_states = teacher_states.to(teacher_head.weight.device)
    if use_fused:
        # Both heads one row chunk at a time, student gradient taken in forward (zoo fused_linear).
        rows = os.environ.get("UNSLOTH_GKD_CHUNK_SIZE", "")
        loss = _unsloth_fused_linear_jsd(
            student_states,
            student_head.weight,
            teacher_states.detach(),
            teacher_head.weight.detach(),
            labels = shifted_labels,
            beta = float(self.beta),
            # No TRL release passes a temperature to generalized_jsd_loss (GKDConfig.temperature is for sampling).
            temperature = 1.0,
            student_bias = student_head.bias,
            teacher_bias = None if teacher_head.bias is None else teacher_head.bias.detach(),
            student_logit_scale = student_scale,
            student_logit_softcapping = student_softcap,
            teacher_logit_scale = teacher_scale,
            teacher_logit_softcapping = teacher_softcap,
            num_items_in_batch = items,
            scaling = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None,
            rows_per_chunk = int(rows) if rows.strip().isdigit() else None,
        )
        try:
            self._unsloth_gkd_chunked_calls = getattr(self, "_unsloth_gkd_chunked_calls", 0) + 1
            self._unsloth_gkd_fused_calls = getattr(self, "_unsloth_gkd_fused_calls", 0) + 1
        except Exception:
            pass
        return loss
    loss, _entropy_sum, _n_valid = distillation_chunked_jsd(
        student_states,
        teacher_states.detach(),
        student_head.weight,
        teacher_head.weight.detach(),
        shifted_labels != -100,
        beta = float(self.beta),
        chunk_size = _unsloth_gkd_chunk_size(student_head.weight.shape[0]),
        num_items_in_batch = items,
        student_lm_head_bias = student_head.bias,
        teacher_lm_head_bias = None if teacher_head.bias is None else teacher_head.bias.detach(),
        student_logit_scale = student_scale,
        teacher_logit_scale = teacher_scale,
        student_final_logit_softcapping = student_softcap,
        teacher_final_logit_softcapping = teacher_softcap,
        # No TRL release passes a temperature to generalized_jsd_loss (GKDConfig.temperature is for sampling).
        temperature = 1.0,
        **jsd_kwargs,
    )
    try:
        self._unsloth_gkd_chunked_calls = getattr(self, "_unsloth_gkd_chunked_calls", 0) + 1
    except Exception:
        pass
    return loss


def gkd_trainer_compute_loss(function_name, function):
    """Route the generated ``compute_loss`` through the chunked loss, keeping TRL's body as the fallback."""
    if function_name != "compute_loss" or "_unsloth_trl_compute_loss" in function:
        return function
    layout = _unsloth_gkd_layout(function)
    if layout is None or (distillation_chunked_jsd is None and _unsloth_fused_linear_jsd is None):
        return function
    renamed, n = re.subn(
        r"\bdef compute_loss\(", "def _unsloth_trl_compute_loss(", function, count = 1
    )
    if n != 1:
        return function
    indent = re.search(r"^([ \t]*)def compute_loss\(", function, flags = re.MULTILINE).group(1)
    wrapper = textwrap.indent(
        "def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n"
        "    # Unsloth: chunked generalized JSD over hidden states; TRL's own loss below is the fallback.\n"
        "    if not return_outputs:\n"
        f"        inputs = _unsloth_gkd_right_align(\n"
        f"            inputs, {layout!r}, getattr(self, 'use_liger_gkd_loss', False),\n"
        "        )\n"
        "        loss = _unsloth_gkd_chunked_loss(\n"
        f"            self, model, inputs, num_items_in_batch, {layout!r},\n"
        "        )\n"
        "        if loss is not None:\n"
        "            return loss\n"
        "    return self._unsloth_trl_compute_loss(\n"
        "        model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,\n"
        "    )\n",
        indent,
    )
    return wrapper + "\n" + renamed


RL_FUNCTIONS["gkd_trainer"].append(gkd_trainer_compute_loss)
for _gkd_item in (
    _unsloth_get_model_config,
    _unsloth_text_configs,
    _unsloth_get_final_logit_softcapping,
    _unsloth_resolve_logit_scales,
    _unsloth_grpo_returns_hidden_states,
    _unsloth_grpo_hidden_states_signal,
    _unsloth_gkd_canonical,
    _unsloth_gkd_jsd_supported,
    _unsloth_gkd_note_fallback,
    _unsloth_gkd_dense_head,
    _unsloth_gkd_logit_transforms,
    _unsloth_gkd_project,
    _unsloth_gkd_chunk_size,
    _unsloth_gkd_right_align,
    _unsloth_gkd_chunked_loss,
):
    RL_PRE_ITEMS["gkd_trainer"].append(inspect.getsource(_gkd_item))
RL_PRE_ITEMS["gkd_trainer"].append(
    "import re\n"
    "import inspect\n"
    "try:\n"
    "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
    "except Exception:\n"
    "    detect_logit_transforms = None\n"
    "try:\n"
    "    from unsloth_zoo.rl_replacements import distillation_chunked_jsd\n"
    "except Exception:\n"
    "    distillation_chunked_jsd = None\n"
    "try:\n"
    "    from unsloth_zoo.fused_losses.fused_linear import fused_linear_jsd as _unsloth_fused_linear_jsd\n"
    "except Exception:\n"
    "    _unsloth_fused_linear_jsd = None\n"
)


# ORPO / CPO: log-probs from hidden states via a chunked head; anything not reproduced exactly runs TRL's body.
try:
    from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs
except Exception:
    unsloth_chunked_logprobs = None
# Scored positions only, with the logits/* means from the head's affine mean (no backward); older zoo scores every row.
try:
    from unsloth_zoo.fused_losses.chunked_logprob import (
        unsloth_masked_logprobs as _unsloth_pref_masked_logprobs,
    )
    from unsloth_zoo.fused_losses.preference_loss import (
        linear_mean_logits as _unsloth_linear_mean_logits,
    )
except Exception:
    _unsloth_pref_masked_logprobs = _unsloth_linear_mean_logits = None


def _unsloth_stock_logps_hook(
    self,
    method,
    hooks = ("get_batch_logps",),
):
    """False when a subclass overrides one of ``hooks`` below the class whose ``method`` is the rewritten one: TRL's
    body would call the override (or it reads the full logits the chunked path replaces with stand-ins)."""
    owner = None
    for klass in type(self).__mro__:
        # The rewritten body calls a chunked or fused-pass helper; a subclass method wrapping it via super() does not.
        names = getattr(getattr(vars(klass).get(method), "__code__", None), "co_names", ())
        if any(
            name.startswith("_unsloth_") and ("chunked" in name or "fused_pass" in name)
            for name in names
        ):
            owner = klass
            break
    if owner is None:
        return True
    for name in hooks:
        hook = next((klass for klass in type(self).__mro__ if name in vars(klass)), None)
        if hook is not None and hook not in owner.__mro__:
            return False
    return True


def _unsloth_pref_chunked_head(self, model):
    """The dense output head when this concatenated_forward can score from hidden states, else ``None``."""
    if unsloth_chunked_logprobs is None or os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        return None
    if os.environ.get("UNSLOTH_PREFERENCE_CHUNKED", "1").lower() in ("0", "false", "no", "off"):
        return None
    if getattr(self, "is_encoder_decoder", False) or getattr(self, "aux_loss_enabled", False):
        return None
    # Listing the method itself rejects a subclass wrapping it: it would see stand-in logits.
    if not _unsloth_stock_logps_hook(
        self,
        "concatenated_forward",
        ("get_batch_logps", "get_batch_loss_metrics", "concatenated_forward"),
    ):
        return None
    # TRL <= 0.25 scores labels against label_pad_token_id but nn.CrossEntropyLoss always ignores -100.
    if getattr(self, "label_pad_token_id", -100) != -100:
        return None
    if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        return None
    try:
        unwrapped = self.accelerator.unwrap_model(model)
    except Exception:
        return None
    head = _unsloth_gkd_dense_head(unwrapped)
    if head is None:
        return None
    # The kernel reads weight / bias directly: a hooked head (accelerate, forward / backward hooks) is not
    # F.linear(h, W, b), and its backward hooks would never see the gradient.
    if (
        hasattr(head, "_hf_hook")
        or "forward" in vars(head)
        or head._forward_hooks
        or head._forward_pre_hooks
        or head._backward_hooks
        or getattr(head, "_backward_pre_hooks", None)
    ):
        return None
    # MiniCPM3 divides hidden states by logits_scaling before the head.
    for config in _unsloth_text_configs(_unsloth_get_model_config(unwrapped)):
        if getattr(config, "model_type", None) == "minicpm3":
            return None
    # DDP(find_unused_parameters=True) marks a head skipped in forward as unused, then its grad hook fires twice.
    if getattr(model, "find_unused_parameters", False) and any(
        p is not None and p.requires_grad for p in (head.weight, head.bias)
    ):
        return None
    return head


def _unsloth_pref_logit_transforms(model):
    """``(multiply, divide, softcap)`` for the chunked kernel, from the GKD resolver (one scale, 0.0 = off)."""
    scale, softcap = _unsloth_gkd_logit_transforms(model)
    return (0.0 if scale == 1.0 else scale), 0.0, softcap


def _unsloth_pref_chunked_forward(self, model, head, concatenated_batch, len_chosen, states, kind):
    """``(outputs, all_logits)``: ``outputs`` is TRL's return tuple computed from hidden states, or ``None`` with
    ``all_logits`` holding real logits for TRL's own body (the forward ignored the flag, or a shape TRL rejects)."""
    unwrapped = self.accelerator.unwrap_model(model)
    multiply, divide, softcap = _unsloth_pref_logit_transforms(unwrapped)
    if not _unsloth_pref_returns_hidden_states(unwrapped, states, head.weight):
        return None, states
    input_ids = concatenated_batch["concatenated_input_ids"]
    labels = concatenated_batch["concatenated_labels"]
    if states.dim() != 3 or states.shape[:2] != labels.shape or labels.shape != input_ids.shape:
        # TRL raises or reads these itself: hand it the logits it expects.
        logits = torch.nn.functional.linear(
            states.to(device = head.weight.device, dtype = head.weight.dtype), head.weight, head.bias
        )
        if multiply:
            logits = logits * multiply
        if divide:
            logits = logits / divide
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return None, logits

    device = states.device
    labels = labels.to(device)
    input_ids = input_ids.to(device)
    shifted = labels[:, 1:]
    label_mask = shifted != -100
    # cat, not boolean slice assignment: CPU Inductor fails to vectorise that (VecMask<long int, 2> vs VecMask<float, 1>).
    rejected_off = torch.zeros_like(label_mask[len_chosen:])
    if kind == "orpo":
        # ORPO's NLL scores every attended token of prompt + chosen, from input_ids.
        nll_targets = input_ids[:, 1:]
        attention_mask = concatenated_batch["concatenated_attention_mask"].to(device)
        nll_mask = torch.cat((attention_mask[:len_chosen, 1:] == 1, rejected_off), dim = 0)
        average_log_prob = True
        # Checking for a second NLL pass needs a host sync, so compiled graphs always take it (correct either way).
        is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
        if is_compiling is not None and is_compiling():
            split_nll = True
        else:
            split_nll = bool((label_mask & nll_mask & (shifted != nll_targets)).any())
    else:
        nll_targets = shifted
        if self.cpo_alpha != 0:
            nll_mask = torch.cat((label_mask[:len_chosen], rejected_off), dim = 0)
        else:
            nll_mask = torch.zeros_like(label_mask)
        average_log_prob = self.loss_type in ["ipo", "simpo"]
        split_nll = False
    targets = torch.where(label_mask, shifted, torch.where(nll_mask, nll_targets, 0))

    # CPO's logits metric averages every position, ORPO's all but the last.
    rows = states if kind == "cpo" else states[:, :-1]
    kwargs = dict(
        logit_scale_multiply = multiply,
        logit_scale_divide = divide,
        logit_softcapping = softcap,
        lm_head_bias = head.bias,
    )
    if _unsloth_pref_masked_logprobs is not None and _unsloth_linear_mean_logits is not None:
        # Prompt and padding rows (often most of each row) never reach the head; they only feed the mean metric.
        per_token = _unsloth_pref_masked_logprobs(
            states[:, :-1],
            head.weight,
            targets,
            mask = label_mask if split_nll else label_mask | nll_mask,
            **kwargs,
        )
        mean_logits = _unsloth_linear_mean_logits(
            rows, head.weight, head.bias, multiply, divide, softcap
        )
        if split_nll:
            nll_per_token = _unsloth_pref_masked_logprobs(
                states[:len_chosen, :-1],
                head.weight,
                nll_targets[:len_chosen],
                mask = nll_mask[:len_chosen],
                **kwargs,
            )
        else:
            nll_per_token = per_token[:len_chosen]
    else:
        if kind == "cpo":
            targets = torch.nn.functional.pad(targets, (0, 1))
        per_token, mean_logits = unsloth_chunked_logprobs(
            rows, head.weight, targets, return_mean_logits = True, **kwargs
        )
        if kind == "cpo":
            per_token = per_token[:, :-1]
        if split_nll:
            nll_per_token = unsloth_chunked_logprobs(
                states[:len_chosen, :-1],
                head.weight,
                torch.where(nll_mask, nll_targets, 0)[:len_chosen],
                **kwargs,
            )
        else:
            nll_per_token = per_token[:len_chosen]

    loss_mask = label_mask.to(per_token.dtype)
    all_logps = (per_token * loss_mask).sum(-1)
    if average_log_prob:
        all_logps = all_logps / loss_mask.sum(-1)
    chosen_nll_mask = nll_mask[:len_chosen].to(per_token.dtype)
    if kind == "cpo" and self.cpo_alpha == 0:
        nll_loss = torch.tensor(0.0).to(self.accelerator.device)
    else:
        nll_loss = -(nll_per_token * chosen_nll_mask).sum() / chosen_nll_mask.sum()
    try:
        self._unsloth_pref_chunked_calls = getattr(self, "_unsloth_pref_chunked_calls", 0) + 1
    except Exception:
        pass
    # Stand-ins: TRL only reads chosen/rejected logits as .detach().mean().
    return (
        all_logps[:len_chosen],
        all_logps[len_chosen:],
        mean_logits[:len_chosen],
        mean_logits[len_chosen:],
        nll_loss,
    ), states


_PREF_LAYOUTS = {
    "orpo": (
        "labels = torch.where(attention_mask == 1, labels,",
        "chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])",
        "average_log_prob=True",
        "chosen_logits = all_logits[:len_chosen, :-1, :]",
        "rejected_logits = all_logits[len_chosen:, :-1, :]",
        "return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)",
    ),
    "cpo": (
        "labels = concatenated_batch['concatenated_labels'].clone()",
        "nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])",
        "average_log_prob=self.loss_type in ['ipo', 'simpo']",
        "chosen_logits = all_logits[:len_chosen]\n",
        "rejected_logits = all_logits[len_chosen:]\n",
        "return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)",
    ),
}
_PREF_COMMON = (
    "len_chosen = batch['chosen_labels'].shape[0]",
    "all_logps = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=",
    "chosen_logps = all_logps[:len_chosen]",
    "rejected_logps = all_logps[len_chosen:]",
    "loss_fct = nn.CrossEntropyLoss()",
)


def _unsloth_pref_rewrite_concatenated_forward(function, kind):
    canonical = _unsloth_gkd_canonical(function)
    if canonical is None or "_unsloth_pref_chunked_forward" in function:
        return function
    if not all(x in canonical for x in _PREF_COMMON + _PREF_LAYOUTS[kind]):
        return function
    pattern = re.compile(
        r"(?ms)^([ \t]*)(outputs = model\(\n.*?^\1\)\n)\1all_logits = outputs\.logits\n"
    )
    matches = list(pattern.finditer(function))
    if len(matches) != 1:
        return function
    match = matches[0]
    indent = match.group(1)
    call = textwrap.indent(match.group(2), "    ")
    block = (
        "_unsloth_pref_head = _unsloth_pref_chunked_head(self, model)\n"
        "_unsloth_pref_prior = os.environ.get('UNSLOTH_RETURN_HIDDEN_STATES')\n"
        "if _unsloth_pref_head is not None:\n"
        "    os.environ['UNSLOTH_RETURN_HIDDEN_STATES'] = '1'\n"
        "try:\n"
        f"{call}"
        "finally:\n"
        "    if _unsloth_pref_head is not None:\n"
        "        if _unsloth_pref_prior is None:\n"
        "            os.environ.pop('UNSLOTH_RETURN_HIDDEN_STATES', None)\n"
        "        else:\n"
        "            os.environ['UNSLOTH_RETURN_HIDDEN_STATES'] = _unsloth_pref_prior\n"
        "all_logits = outputs.logits\n"
        "if _unsloth_pref_head is not None:\n"
        "    _unsloth_pref_out, all_logits = _unsloth_pref_chunked_forward(\n"
        f"        self, model, _unsloth_pref_head, concatenated_batch, len_chosen, all_logits, {kind!r}\n"
        "    )\n"
        "    if _unsloth_pref_out is not None:\n"
        "        return _unsloth_pref_out\n"
    )
    return function[: match.start()] + textwrap.indent(block, indent) + function[match.end() :]


def orpo_trainer_concatenated_forward(function_name, function):
    if function_name != "concatenated_forward":
        return function
    return _unsloth_pref_rewrite_concatenated_forward(function, "orpo")


def cpo_trainer_concatenated_forward(function_name, function):
    if function_name != "concatenated_forward":
        return function
    return _unsloth_pref_rewrite_concatenated_forward(function, "cpo")


RL_FUNCTIONS["orpo_trainer"].append(orpo_trainer_concatenated_forward)
RL_FUNCTIONS["cpo_trainer"].append(cpo_trainer_concatenated_forward)
for _pref_trainer in ("orpo_trainer", "cpo_trainer"):
    for _pref_item in (
        _unsloth_get_model_config,
        _unsloth_text_configs,
        _unsloth_get_final_logit_softcapping,
        _unsloth_resolve_logit_scales,
        _unsloth_grpo_returns_hidden_states,
        _unsloth_grpo_hidden_states_signal,
        _unsloth_pref_returns_hidden_states,
        _unsloth_gkd_dense_head,
        _unsloth_stock_logps_hook,
        _unsloth_pref_chunked_head,
        _unsloth_gkd_logit_transforms,
        _unsloth_pref_logit_transforms,
        _unsloth_pref_chunked_forward,
    ):
        RL_PRE_ITEMS[_pref_trainer].append(inspect.getsource(_pref_item))
    RL_PRE_ITEMS[_pref_trainer].append(
        "try:\n"
        "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
        "except Exception:\n"
        "    detect_logit_transforms = None\n"
        "try:\n"
        "    from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs\n"
        "except Exception:\n"
        "    unsloth_chunked_logprobs = None\n"
        "try:\n"
        "    from unsloth_zoo.fused_losses.chunked_logprob import unsloth_masked_logprobs as _unsloth_pref_masked_logprobs\n"
        "    from unsloth_zoo.fused_losses.preference_loss import linear_mean_logits as _unsloth_linear_mean_logits\n"
        "except Exception:\n"
        "    _unsloth_pref_masked_logprobs = _unsloth_linear_mean_logits = None\n"
    )


# DPO (TRL < 0.29, concatenated_forward): log-probs from hidden states; TRL >= 1.15 fuses the head itself.
try:
    from unsloth_zoo.fused_losses.chunked_logprob import (
        unsloth_masked_logprobs as _unsloth_masked_logprobs,
    )
except Exception:
    _unsloth_masked_logprobs = None


def _unsloth_dpo_chunked_head(
    self,
    model,
    concatenated_batch,
    method = "concatenated_forward",
):
    """``(unwrapped_model, head, (multiply, divide, softcap))`` when concatenated_forward can score from hidden
    states, else ``None`` (TRL's own logits path runs). Decided before the forward, which then skips the logits."""
    reason = None
    if _unsloth_masked_logprobs is None:
        reason = "unsloth_zoo has no chunked_logprob"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif os.environ.get("UNSLOTH_DPO_CHUNKED", "1").lower() in ("0", "false", "no", "off"):
        reason = "UNSLOTH_DPO_CHUNKED=0"
    elif getattr(self, "is_encoder_decoder", False):
        reason = "encoder-decoder"
    elif getattr(self, "aux_loss_enabled", False):
        reason = "router aux loss needs the stock forward"
    elif not _unsloth_stock_logps_hook(
        self,
        method,
        ("get_batch_logps",)
        if method == "concatenated_forward"
        else ("get_batch_logps", "get_batch_loss_metrics", method),
    ):
        # KTO hands its callers logits stand-ins; DPO's outputs are exact, so a DPO wrapper is fine.
        reason = "get_batch_logps / get_batch_loss_metrics overridden"
    elif getattr(self, "padding_free", False):
        reason = "padding_free"
    elif getattr(self, "use_weighting", False):
        reason = "use_weighting needs full-vocabulary log-probs"
    elif getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        reason = "FSDP / DeepSpeed"
    elif any(
        k in concatenated_batch
        for k in ("pixel_values", "pixel_attention_mask", "image_sizes", "token_type_ids")
        + tuple(_DPO_VISION_KEYS)
    ):
        reason = "vision inputs"
    if reason is None:
        try:
            unwrapped = self.accelerator.unwrap_model(model)
        except Exception:
            unwrapped = None
            reason = "unwrap_model"
    if reason is None:
        head = _unsloth_gkd_dense_head(unwrapped)
        if head is None:
            reason = "output head is not a dense nn.Linear"
        # The kernel reads weight / bias directly: a hooked head (accelerate, forward / backward hooks) is not
        # F.linear(h, W, b), and its backward hooks would never see the gradient.
        elif (
            hasattr(head, "_hf_hook")
            or "forward" in vars(head)
            or head._forward_hooks
            or head._forward_pre_hooks
            or head._backward_hooks
            or getattr(head, "_backward_pre_hooks", None)
        ):
            reason = "output head has hooks or a patched forward"
    if reason is None:
        for config in _unsloth_text_configs(_unsloth_get_model_config(unwrapped)):
            if getattr(config, "model_type", None) == "minicpm3":
                reason = "minicpm3 scales hidden states before the head"
    if reason is None and getattr(model, "find_unused_parameters", False):
        if any(p is not None and p.requires_grad for p in (head.weight, head.bias)):
            reason = "DDP find_unused_parameters with a trainable head"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return unwrapped, head, _unsloth_pref_logit_transforms(unwrapped)


def _unsloth_dpo_model_forward(model, input_ids, model_kwargs, want_hidden_states):
    if not want_hidden_states:
        return model(input_ids, **model_kwargs)
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        return model(input_ids, **model_kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior


def _unsloth_dpo_returns_hidden_states(self, chunked, logits):
    if chunked is None:
        return False
    unwrapped, head, _ = chunked
    hidden = _unsloth_pref_returns_hidden_states(unwrapped, logits, head.weight)
    if not hidden:
        fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
        reason = "forward returned logits, not hidden states"
        fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return hidden


def _unsloth_dpo_hidden_logps(self, chunked, hidden_states, labels, loss_mask):
    """``(per_token_logps [B, S], mean_logits [B, S, 1])`` where TRL computes ``selective_log_softmax(logits, labels)``
    and later reads ``logits[mask].mean()``. Rows outside ``loss_mask`` are 0 and never reach the head; the mean over
    a masked row's vocab equals TRL's element mean, so the stand-in keeps the ``logits/*`` metrics exact."""
    _, head, (multiply, divide, softcap) = chunked
    per_token_logps, mean_logits = _unsloth_masked_logprobs(
        hidden_states.to(head.weight.device),
        head.weight,
        labels,
        mask = loss_mask,
        logit_scale_multiply = multiply,
        logit_scale_divide = divide,
        logit_softcapping = softcap,
        lm_head_bias = head.bias,
        return_mean_logits = True,
    )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    return per_token_logps.to(labels.device), mean_logits.unsqueeze(-1).to(labels.device)


def _unsloth_dpo_nll(chosen_logps, chosen_labels):
    """``F.cross_entropy(chosen_logits, chosen_labels, ignore_index=0)`` from the per-token log-probs at the same
    positions (TRL zeroes masked labels, so 0 is its ignore index)."""
    keep = chosen_labels != 0
    return -(chosen_logps * keep).sum() / keep.sum()


_DPO_CF_FORWARD = (
    "            outputs = model(input_ids, **model_kwargs)\n            logits = outputs.logits\n"
)
_DPO_CF_LOGPS = "        per_token_logps = selective_log_softmax(logits, labels)\n"
_DPO_CF_START = '        num_examples = batch["prompt_input_ids"].shape[0]\n'
_DPO_CF_CHOSEN = re.compile(r"^(?P<indent>[ \t]*)chosen_logits = (?P<expr>[^\n]+)\n", re.MULTILINE)
_DPO_CF_CE = re.compile(
    r"F\.cross_entropy\(\s*torch\.flatten\(chosen_logits, end_dim=1\),\s*"
    r"torch\.flatten\(chosen_labels, end_dim=1\),\s*ignore_index=0\s*\)"
)


def dpo_trainer_chunked_logprobs(function_name, function):
    """Rewrite TRL < 0.29's concatenated_forward to score from hidden states. Every anchor must match exactly once,
    else TRL's source is returned untouched; at run time ``_unsloth_dpo_chunked_head`` keeps TRL's math for any
    feature that needs the full logits, and a forward that ignored the flag still takes TRL's path."""
    if function_name != "concatenated_forward" or _unsloth_masked_logprobs is None:
        return function
    if "_unsloth_dpo_hidden" in function:
        return function
    if any(function.count(a) != 1 for a in (_DPO_CF_FORWARD, _DPO_CF_LOGPS, _DPO_CF_START)):
        return function
    chosen = list(_DPO_CF_CHOSEN.finditer(function))
    if len(chosen) != 1 or len(_DPO_CF_CE.findall(function)) != 1:
        return function
    # Other full-logits readers: padding_free / use_weighting (refused at run time) and the stand-in mean metrics.
    expr = chosen[0].group("expr")
    if "logits" not in expr or "shape" in expr:
        return function
    new = function.replace(
        _DPO_CF_START,
        _DPO_CF_START + "        _unsloth_dpo_hidden = False\n",
    )
    new = new.replace(
        _DPO_CF_FORWARD,
        "            _unsloth_dpo_chunked = _unsloth_dpo_chunked_head(self, model, concatenated_batch)\n"
        "            outputs = _unsloth_dpo_model_forward(\n"
        "                model, input_ids, model_kwargs, _unsloth_dpo_chunked is not None\n"
        "            )\n"
        "            logits = outputs.logits\n"
        "            _unsloth_dpo_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_chunked, logits)\n",
    )
    new = new.replace(
        _DPO_CF_LOGPS,
        "        if _unsloth_dpo_hidden:\n"
        "            per_token_logps, logits = _unsloth_dpo_hidden_logps(\n"
        "                self, _unsloth_dpo_chunked, logits, labels, loss_mask\n"
        "            )\n"
        "            _unsloth_dpo_token_logps = per_token_logps\n"
        "        else:\n"
        "            per_token_logps = selective_log_softmax(logits, labels)\n",
    )
    indent = chosen[0].group("indent")
    chosen_logps_expr = re.sub(r"\blogits\b", "_unsloth_dpo_token_logps", expr)
    new = _DPO_CF_CHOSEN.sub(
        lambda m: m.group(0)
        + f"{indent}if _unsloth_dpo_hidden:\n"
        + f"{indent}    _unsloth_dpo_chosen_logps = {chosen_logps_expr}\n",
        new,
        count = 1,
    )
    new = _DPO_CF_CE.sub(
        lambda m: "(\n"
        "                _unsloth_dpo_nll(_unsloth_dpo_chosen_logps, chosen_labels)\n"
        "                if _unsloth_dpo_hidden\n"
        f"                else {m.group(0)}\n"
        "            )",
        new,
        count = 1,
    )
    return new


RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_chunked_logprobs)


# DPO (TRL 0.29 - 1.14, _compute_loss): log-probs, entropy, logits/* means and token accuracy from hidden states.
try:
    from unsloth_zoo.fused_losses.fused_linear import (
        fused_linear_logprobs as _unsloth_fused_linear_logprobs,
    )
except Exception:
    _unsloth_fused_linear_logprobs = None


def _unsloth_dpo_forward(model, model_kwargs, want_hidden_states):
    if not want_hidden_states:
        return model(**model_kwargs)
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        return model(**model_kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior


def _unsloth_dpo_loss_head(
    self,
    model,
    inputs,
    return_outputs = False,
):
    """``_unsloth_dpo_chunked_head`` for TRL 0.29 - 1.14's ``_compute_loss``, which also needs the fused linear op
    (entropy and argmax match) and must not hand hidden states back as ``outputs.logits``."""
    reason = None
    if _unsloth_fused_linear_logprobs is None:
        reason = "unsloth_zoo has no fused_linear_logprobs"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif return_outputs:
        reason = "return_outputs needs the logits"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return _unsloth_dpo_chunked_head(self, model, inputs, "_compute_loss")


def _unsloth_dpo_loss_stats(
    self,
    chunked,
    hidden_states,
    labels,
    mask,
    ref = False,
):
    """TRL's per-token statistics of ``logits = head(hidden_states)`` at completion positions (``mask``), the rest 0:
    ``(log_probs, entropy, mean_logits [.., 1], is_top1)``, or the log-probs alone for ``ref``. ``mean_logits`` stands
    in for the logits: ``x[mask].mean(-1)`` on it is TRL's vocab mean. No temperature: TRL's DPO scores at 1.0."""
    _, head, (multiply, divide, softcap) = chunked
    # TRL projects the reference logits inside its no_grad forward: a trainable (or shared) head gets no gradient.
    with torch.set_grad_enabled(torch.is_grad_enabled() and not ref):
        out = _unsloth_fused_linear_logprobs(
            hidden_states,
            head.weight,
            labels,
            bias = head.bias,
            mask = mask,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_entropy = not ref,
            return_mean_logits = not ref,
            return_is_top1 = not ref,
            entropy_requires_grad = False,
        )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    if ref:
        return out
    logps, entropy, mean_logits, is_top1 = out
    return logps, entropy, mean_logits.unsqueeze(-1), is_top1


_DPO_CL_MODE = re.compile(
    r'^(?P<i>[ \t]*)mode = "train" if self\.model\.training else "eval"\n', re.MULTILINE
)
_DPO_CL_FORWARD = re.compile(r"^(?P<i>[ \t]*)outputs = model\(\*\*model_kwargs\)\n", re.MULTILINE)
_DPO_CL_SHIFT = re.compile(
    r"^[ \t]*shift_logits = outputs\.logits\[\.\.\., :-1, :\](?:\.contiguous\(\))?\n",
    re.MULTILINE,
)
# TRL < 1.14: separate log-prob and entropy calls; 1.14: one fused call with a row mask.
_DPO_CL_LOGPS = re.compile(
    r"^(?P<i>[ \t]*)per_token_logps = selective_log_softmax\(shift_logits, shift_labels\)\n",
    re.MULTILINE,
)
_DPO_CL_ENTROPY = re.compile(
    r"^(?P<i>[ \t]*)per_token_entropy = entropy_from_logits\(shift_logits\.detach\(\)\)\n",
    re.MULTILINE,
)
_DPO_CL_LOGPS_ENT = re.compile(
    r"^(?P<i>[ \t]*)per_token_logps, per_token_entropies = selective_log_softmax_and_entropy\(\s*"
    r"shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask\s*\)\n",
    re.MULTILINE,
)
_DPO_CL_REF_FORWARD = re.compile(
    r"^(?P<i>[ \t]*)ref_outputs = (?P<model>self\.model|self\.ref_model)\(\*\*(?P<kw>model_kwargs|ref_model_kwargs)\)\n",
    re.MULTILINE,
)
_DPO_CL_REF_LOGPS = re.compile(
    r"^(?P<i>[ \t]*)ref_per_token_logps = selective_log_softmax\(\s*ref_shift_logits, shift_labels"
    r"(?P<row_mask>, row_mask=shift_completion_mask)?\s*\)\n",
    re.MULTILINE,
)
_DPO_CL_ARGMAX = re.compile(
    r"^(?P<i>[ \t]*)predictions = chosen_logits\.argmax\(dim=-1\)\n", re.MULTILINE
)
_DPO_CL_SFT = (
    "F.cross_entropy(chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])"
)


def dpo_trainer_compute_loss_chunked(function_name, function):
    """Rewrite TRL 0.29 - 1.14's ``_compute_loss`` to score from hidden states: it projects every position of the
    policy and reference passes to ``[B, S, V]`` logits, then reads log-probs, entropy, the ``logits/*`` means, the
    argmax token accuracy and the ``sft`` loss term from them. One fused linear call returns all of them at the
    completion positions (the only ones TRL reads), so neither pass materializes logits. Every anchor must match
    exactly once (each reference forward at most once), else TRL's source is returned untouched; at run time
    ``_unsloth_dpo_loss_head`` keeps TRL's path for anything that needs the full logits (``use_weighting``, aux loss,
    ``return_outputs``, vision, ...) and a forward that ignored the flag still takes TRL's path."""
    if function_name != "_compute_loss" or _unsloth_fused_linear_logprobs is None:
        return function
    if "_unsloth_dpo_hidden" in function:
        return function
    fused_entropy = len(_DPO_CL_LOGPS_ENT.findall(function)) == 1
    single = [
        _DPO_CL_MODE,
        _DPO_CL_FORWARD,
        _DPO_CL_SHIFT,
        _DPO_CL_ARGMAX,
        _DPO_CL_REF_LOGPS,
    ]
    single += [] if fused_entropy else [_DPO_CL_LOGPS, _DPO_CL_ENTROPY]
    if any(len(p.findall(function)) != 1 for p in single) or function.count(_DPO_CL_SFT) != 1:
        return function
    ref_forwards = _DPO_CL_REF_FORWARD.findall(function)
    if not 1 <= len(ref_forwards) <= 2 or len({m[1] for m in ref_forwards}) != len(ref_forwards):
        return function
    # Any other reader of the full logits would see the stand-in.
    if function.count("outputs.logits") != 2 or function.count("shift_logits") != (
        8 if fused_entropy else 9
    ):
        return function

    def _sub(pattern, make, text):
        return pattern.sub(lambda m: make(m.group("i"), m), text, count = 1)

    new = _sub(
        _DPO_CL_MODE,
        lambda i, m: m.group(0) + f"{i}_unsloth_dpo_hidden = _unsloth_dpo_ref_hidden = False\n",
        function,
    )
    new = _sub(
        _DPO_CL_FORWARD,
        lambda i, m: (
            f"{i}_unsloth_dpo_chunked = _unsloth_dpo_loss_head(self, model, inputs, return_outputs)\n"
            f"{i}outputs = _unsloth_dpo_forward(model, model_kwargs, _unsloth_dpo_chunked is not None)\n"
            f"{i}_unsloth_dpo_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_chunked, outputs.logits)\n"
        ),
        new,
    )
    stats = (
        "{i}if _unsloth_dpo_hidden:\n"
        "{i}    {lhs}, shift_logits, _unsloth_dpo_top1 = _unsloth_dpo_loss_stats(\n"
        "{i}        self, _unsloth_dpo_chunked, shift_logits, shift_labels, shift_completion_mask\n"
        "{i}    )\n"
        "{i}else:\n"
    )
    if fused_entropy:
        new = _sub(
            _DPO_CL_LOGPS_ENT,
            lambda i, m: stats.format(i = i, lhs = "per_token_logps, per_token_entropies")
            + textwrap.indent(m.group(0)[len(i) :], i + "    "),
            new,
        )
    else:
        new = _sub(
            _DPO_CL_LOGPS,
            lambda i, m: stats.format(i = i, lhs = "per_token_logps, _unsloth_dpo_entropy")
            + f"{i}    {m.group(0).lstrip()}",
            new,
        )
        new = _sub(
            _DPO_CL_ENTROPY,
            lambda i, m: (
                f"{i}if _unsloth_dpo_hidden:\n"
                f"{i}    per_token_entropy = _unsloth_dpo_entropy\n"
                f"{i}else:\n"
                f"{i}    {m.group(0).lstrip()}"
            ),
            new,
        )

    def _ref_forward(m):
        i, model, kw = m.group("i"), m.group("model"), m.group("kw")
        return (
            f"{i}_unsloth_dpo_ref = (\n"
            f'{i}    _unsloth_dpo_chunked_head(self, {model}, inputs, "_compute_loss") if _unsloth_dpo_hidden else None\n'
            f"{i})\n"
            f"{i}ref_outputs = _unsloth_dpo_forward({model}, {kw}, _unsloth_dpo_ref is not None)\n"
            f"{i}_unsloth_dpo_ref_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_ref, ref_outputs.logits)\n"
        )

    new = _DPO_CL_REF_FORWARD.sub(_ref_forward, new)
    new = _sub(
        _DPO_CL_REF_LOGPS,
        lambda i, m: (
            f"{i}if _unsloth_dpo_ref_hidden:\n"
            f"{i}    ref_per_token_logps = _unsloth_dpo_loss_stats(\n"
            f"{i}        self, _unsloth_dpo_ref, ref_shift_logits, shift_labels, shift_completion_mask, ref = True\n"
            f"{i}    )\n"
            f"{i}else:\n" + textwrap.indent(m.group(0)[len(i) :], i + "    ")
        ),
        new,
    )
    # TRL's argmax ties go to the first index, as does is_top1; -1 is never a token id.
    new = _sub(
        _DPO_CL_ARGMAX,
        lambda i, m: (
            f"{i}if _unsloth_dpo_hidden:\n"
            f"{i}    predictions = torch.where(\n"
            f"{i}        _unsloth_dpo_top1[: len(_unsloth_dpo_top1) // 2], shift_labels[: len(shift_labels) // 2], -1\n"
            f"{i}    )\n"
            f"{i}else:\n"
            f"{i}    {m.group(0).lstrip()}"
        ),
        new,
    )
    # The sft term: mean NLL of the chosen completion tokens, from the log-probs already computed.
    new = new.replace(
        _DPO_CL_SFT,
        "(\n"
        "                    -per_token_logps[: len(per_token_logps) // 2][chosen_mask.bool()].mean()\n"
        "                    if _unsloth_dpo_hidden\n"
        f"                    else {_DPO_CL_SFT}\n"
        "                )",
        1,
    )
    return new


RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_compute_loss_chunked)
for _dpo_item in (
    _unsloth_get_model_config,
    _unsloth_text_configs,
    _unsloth_get_final_logit_softcapping,
    _unsloth_resolve_logit_scales,
    _unsloth_grpo_returns_hidden_states,
    _unsloth_grpo_hidden_states_signal,
    _unsloth_pref_returns_hidden_states,
    _unsloth_gkd_dense_head,
    _unsloth_gkd_logit_transforms,
    _unsloth_pref_logit_transforms,
    _unsloth_stock_logps_hook,
    _unsloth_dpo_chunked_head,
    _unsloth_dpo_model_forward,
    _unsloth_dpo_returns_hidden_states,
    _unsloth_dpo_hidden_logps,
    _unsloth_dpo_nll,
    _unsloth_dpo_forward,
    _unsloth_dpo_loss_head,
    _unsloth_dpo_loss_stats,
):
    RL_PRE_ITEMS["dpo_trainer"].append(inspect.getsource(_dpo_item))
RL_PRE_ITEMS["dpo_trainer"].append(
    "try:\n"
    "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
    "except Exception:\n"
    "    detect_logit_transforms = None\n"
    "try:\n"
    "    from unsloth_zoo.fused_losses.chunked_logprob import (\n"
    "        unsloth_masked_logprobs as _unsloth_masked_logprobs,\n"
    "    )\n"
    "except Exception:\n"
    "    _unsloth_masked_logprobs = None\n"
    "try:\n"
    "    from unsloth_zoo.fused_losses.fused_linear import (\n"
    "        fused_linear_logprobs as _unsloth_fused_linear_logprobs,\n"
    "    )\n"
    "except Exception:\n"
    "    _unsloth_fused_linear_logprobs = None\n"
    f"_DPO_VISION_KEYS = {tuple(_DPO_VISION_KEYS)!r}\n"
)


# KTO (TRL < 1.10, forward + get_batch_logps): the policy pass, the only one that keeps its logits for backward.
try:
    from unsloth_zoo.fused_losses.chunked_logprob import (
        unsloth_chunked_logprobs as _unsloth_chunked_logprobs,
    )
except Exception:
    _unsloth_chunked_logprobs = None


def _unsloth_kto_hidden_logps(self, chunked, hidden_states, labels, label_pad_token_id):
    """``(completion_logps [B], logits stand-in [B, S, 1])`` for TRL's ``get_batch_logps(logits, labels,
    average_log_prob=False)``. The stand-in holds each position's vocab sum of the logits, so the
    ``logits/chosen_sum`` metrics (a sum over every position) stay exact."""
    _, head, (multiply, divide, softcap) = chunked
    # The model may truncate input_ids (see kto_trainer_get_batch_logps): score the common length.
    n = min(hidden_states.shape[1], labels.shape[1])
    out_device = hidden_states.device
    hidden_states = hidden_states[:, :n]
    labels = labels[:, :n].to(head.weight.device)
    shifted = labels[:, 1:]
    mask = shifted != label_pad_token_id
    if _unsloth_masked_logprobs is not None and _unsloth_linear_mean_logits is not None:
        # Only completion tokens reach the head; the logits/* sums come from the head's affine mean (no backward).
        hidden_states = hidden_states.to(head.weight.device)
        per_token_logps = _unsloth_masked_logprobs(
            hidden_states[:, :-1],
            head.weight,
            shifted,
            mask = mask,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            lm_head_bias = head.bias,
        )
        mean_logits = _unsloth_linear_mean_logits(
            hidden_states, head.weight, head.bias, multiply, divide, softcap
        )
    else:
        index = torch.cat(
            (shifted.masked_fill(~mask, 0), shifted.new_zeros(shifted.shape[0], 1)), dim = 1
        )
        per_token_logps, mean_logits = _unsloth_chunked_logprobs(
            hidden_states.to(head.weight.device),
            head.weight,
            index,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            lm_head_bias = head.bias,
            return_mean_logits = True,
        )
        per_token_logps = per_token_logps[:, :-1]
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    completion_logps = (per_token_logps * mask).sum(-1)
    logits_sum = (mean_logits * head.weight.shape[0]).unsqueeze(-1)
    return completion_logps.to(out_device), logits_sum.to(out_device)


_KTO_FORWARD = re.compile(
    r"^(?P<indent>[ \t]+)outputs = model\(\n"
    r"[ \t]+batch\[\"completion_input_ids\"\],\n"
    r"[ \t]+attention_mask=batch\[\"completion_attention_mask\"\],\n"
    r"[ \t]+\*\*model_kwargs,\n"
    r"[ \t]+\)\n"
    r"[ \t]+completion_logits = outputs\.logits\n",
    re.MULTILINE,
)
_KTO_LOGPS = re.compile(
    r"^(?P<indent>[ \t]+)completion_logps = self\.get_batch_logps\(\n"
    r"[ \t]+completion_logits,\n"
    r"[ \t]+batch\[\"completion_labels\"\],\n"
    r"[ \t]+average_log_prob=False,\n"
    r"(?P<rest>(?:[ \t]+(?:is_encoder_decoder=self\.is_encoder_decoder|label_pad_token_id=self\.label_pad_token_id),\n)*)"
    r"[ \t]+\)\n",
    re.MULTILINE,
)


def kto_trainer_chunked_logprobs(function_name, function):
    """Score KTO's policy completions from hidden states. ``completion_logits`` is then the per-position vocab sum,
    which is all TRL reads from it (``logits/*_sum`` metrics). Untouched unless both anchors match once."""
    if function_name != "forward" or _unsloth_chunked_logprobs is None:
        return function
    if "_unsloth_dpo_hidden" in function or "def get_batch_logps" in function:
        return function
    forward = list(_KTO_FORWARD.finditer(function))
    logps = list(_KTO_LOGPS.finditer(function))
    if len(forward) != 1 or len(logps) != 1:
        return function
    # completion_logits may only be read where a [B, S, 1] stand-in is exact: row selection for the metrics.
    tail = function[logps[0].end() :]
    reads = re.findall(r"[^\n]*\bcompletion_logits\b[^\n]*", tail)
    if any(
        not re.search(r"completion_logits(\.index_select\(0, \w+\)|\[\w+, \.\.\.\]|\.device)", line)
        for line in reads
    ):
        return function
    pad = "self.label_pad_token_id" if "label_pad_token_id" in logps[0].group("rest") else "-100"
    ind = forward[0].group("indent")
    new_forward = (
        f'{ind}_unsloth_dpo_chunked = _unsloth_dpo_chunked_head(self, model, batch, "forward")\n'
        f"{ind}outputs = _unsloth_dpo_model_forward(\n"
        f'{ind}    model, batch["completion_input_ids"],\n'
        f'{ind}    dict(model_kwargs, attention_mask=batch["completion_attention_mask"]),\n'
        f"{ind}    _unsloth_dpo_chunked is not None,\n"
        f"{ind})\n"
        f"{ind}completion_logits = outputs.logits\n"
        f"{ind}_unsloth_dpo_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_chunked, completion_logits)\n"
    )
    ind2 = logps[0].group("indent")
    original_logps = textwrap.indent(textwrap.dedent(logps[0].group(0)), ind2 + "    ")
    new_logps = (
        f"{ind2}if _unsloth_dpo_hidden:\n"
        f"{ind2}    completion_logps, completion_logits = _unsloth_kto_hidden_logps(\n"
        f'{ind2}        self, _unsloth_dpo_chunked, completion_logits, batch["completion_labels"], {pad}\n'
        f"{ind2}    )\n"
        f"{ind2}else:\n"
        f"{original_logps}"
    )
    out = (
        function[: forward[0].start()] + new_forward + function[forward[0].end() : logps[0].start()]
    )
    return out + new_logps + function[logps[0].end() :]


RL_FUNCTIONS["kto_trainer"].append(kto_trainer_chunked_logprobs)


_KTO_KL = re.compile(
    r"^(?P<indent>[ \t]+)with torch\.no_grad\(\):\n"
    r"[ \t]+KL_logits = model\(\*\*KL_model_kwargs\)\.logits\n"
    r"\n?"
    r"(?P<logps>[ \t]+KL_logps = self\.get_batch_logps\(\n"
    r"[ \t]+KL_logits,\n"
    r"[ \t]+batch\[\"KL_completion_labels\"\],\n"
    r"[ \t]+average_log_prob=False,\n"
    r"(?P<rest>(?:[ \t]+(?:is_encoder_decoder=self\.is_encoder_decoder|label_pad_token_id=self\.label_pad_token_id),\n)*)"
    r"[ \t]+\)\n)",
    re.MULTILINE,
)


def kto_trainer_chunked_kl_logps(function_name, function):
    """Score KTO's no-grad KL completions from hidden states too: TRL projects them to full ``[B, S, V]`` logits, the
    largest tensor of a KTO step once the policy pass is chunked. Untouched unless the anchor matches once."""
    if function_name != "_compute_kl_logps" or _unsloth_masked_logprobs is None:
        return function
    if "_unsloth_dpo_hidden" in function:
        return function
    matches = list(_KTO_KL.finditer(function))
    if len(matches) != 1:
        return function
    match = matches[0]
    ind = match.group("indent")
    pad = "self.label_pad_token_id" if "label_pad_token_id" in match.group("rest") else "-100"
    original_logps = textwrap.indent(textwrap.dedent(match.group("logps")), ind + "    ")
    new = (
        f'{ind}_unsloth_dpo_chunked = _unsloth_dpo_chunked_head(self, model, batch, "_compute_kl_logps")\n'
        f"{ind}with torch.no_grad():\n"
        f"{ind}    _unsloth_kl_outputs = _unsloth_dpo_model_forward(\n"
        f'{ind}        model, KL_model_kwargs["input_ids"],\n'
        f'{ind}        {{k: v for k, v in KL_model_kwargs.items() if k != "input_ids"}},\n'
        f"{ind}        _unsloth_dpo_chunked is not None,\n"
        f"{ind}    )\n"
        f"{ind}    KL_logits = _unsloth_kl_outputs.logits\n"
        f"{ind}    _unsloth_dpo_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_chunked, KL_logits)\n"
        f"{ind}    if _unsloth_dpo_hidden:\n"
        f"{ind}        KL_logps, _ = _unsloth_kto_hidden_logps(\n"
        f'{ind}            self, _unsloth_dpo_chunked, KL_logits, batch["KL_completion_labels"], {pad}\n'
        f"{ind}        )\n"
        f"{ind}if not _unsloth_dpo_hidden:\n"
        f"{original_logps}"
    )
    return function[: match.start()] + new + function[match.end() :]


RL_FUNCTIONS["kto_trainer"].append(kto_trainer_chunked_kl_logps)
for _kto_item in (
    _unsloth_get_model_config,
    _unsloth_text_configs,
    _unsloth_get_final_logit_softcapping,
    _unsloth_resolve_logit_scales,
    _unsloth_grpo_returns_hidden_states,
    _unsloth_grpo_hidden_states_signal,
    _unsloth_pref_returns_hidden_states,
    _unsloth_gkd_dense_head,
    _unsloth_gkd_logit_transforms,
    _unsloth_pref_logit_transforms,
    _unsloth_stock_logps_hook,
    _unsloth_dpo_chunked_head,
    _unsloth_dpo_model_forward,
    _unsloth_dpo_returns_hidden_states,
    _unsloth_kto_hidden_logps,
):
    RL_PRE_ITEMS["kto_trainer"].append(inspect.getsource(_kto_item))
RL_PRE_ITEMS["kto_trainer"].append(
    "try:\n"
    "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
    "except Exception:\n"
    "    detect_logit_transforms = None\n"
    "try:\n"
    "    from unsloth_zoo.fused_losses.chunked_logprob import (\n"
    "        unsloth_chunked_logprobs as _unsloth_chunked_logprobs,\n"
    "        unsloth_masked_logprobs as _unsloth_masked_logprobs,\n"
    "    )\n"
    "except Exception:\n"
    "    _unsloth_chunked_logprobs = _unsloth_masked_logprobs = None\n"
    "try:\n"
    "    from unsloth_zoo.fused_losses.preference_loss import linear_mean_logits as _unsloth_linear_mean_logits\n"
    "except Exception:\n"
    "    _unsloth_linear_mean_logits = None\n"
    f"_DPO_VISION_KEYS = {tuple(_DPO_VISION_KEYS)!r}\n"
)


# TRL 1.6 - 1.14 KTO and TRL 0.29 - 1.14 DPO ref log-probs: score from hidden states instead of [B, S, V] logits.
def _unsloth_fused_pass_head(self, model, inputs, return_outputs, method):
    """``_unsloth_dpo_chunked_head`` for a rewritten pass that also needs the fused linear op, keeps TRL's path for
    the kill switch, ``return_outputs`` (the caller reads ``outputs.logits``) and TRL's own liger / chunked head."""
    reason = None
    if _unsloth_fused_linear_logprobs is None:
        reason = "unsloth_zoo has no fused_linear_logprobs"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif return_outputs:
        reason = "return_outputs needs the logits"
    elif getattr(self, "use_liger_kernel", False):
        reason = "use_liger_kernel runs TRL's own chunked head"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return _unsloth_dpo_chunked_head(self, model, inputs, method)


def _unsloth_fused_pass_logps(
    self,
    chunked,
    hidden_states,
    labels,
    mask,
    stats = False,
    grad = False,
):
    """TRL's ``selective_log_softmax(head(hidden_states), labels)`` with rows outside ``mask`` zeroed (TRL's
    ``row_mask`` or its ``logps[mask == 0] = 0.0``). ``stats`` also returns the entropy (no grad, as TRL computes it
    from detached logits) and the per-position vocab mean ``[.., 1]``: ``x[mask].mean(-1)`` on it is TRL's mean of
    the logits row. A shorter ``hidden_states`` (the model truncated input_ids) is scored over the common length and
    zero padded. Only the policy pass (``grad``) keeps a graph: TRL scores the reference / KL logits it built under
    no_grad, so a trainable head must get no gradient from them."""
    _, head, (multiply, divide, softcap) = chunked
    width = labels.shape[1]
    n = min(hidden_states.shape[1], width)
    with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
        out = _unsloth_fused_linear_logprobs(
            hidden_states[:, :n],
            head.weight,
            labels[:, :n],
            bias = head.bias,
            mask = mask[:, :n].bool(),
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_entropy = stats,
            return_mean_logits = stats,
            entropy_requires_grad = False,
        )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    out = out if isinstance(out, tuple) else (out,)
    out = tuple(torch.nn.functional.pad(x, (0, width - n)) if n != width else x for x in out)
    if not stats:
        return out[0].to(labels.device)
    logps, entropy, mean_logits = out
    return (
        logps.to(labels.device),
        entropy.to(labels.device),
        mean_logits.unsqueeze(-1).to(labels.device),
    )


def _unsloth_fused_pass_vocab_sums(self, chunked, hidden_states):
    """``[B, S, 1]`` per-position vocab sums of ``head(hidden_states)`` (no grad): TRL 1.6 - 1.8 KTO logs
    ``logits.index_select(0, idx).nansum()``, a sum over every position and vocab entry, which this keeps exact."""
    _, head, (multiply, divide, softcap) = chunked
    with torch.no_grad():
        hidden_states = hidden_states.detach()
        _, mean_logits = _unsloth_fused_linear_logprobs(
            hidden_states,
            head.weight,
            torch.zeros(hidden_states.shape[:-1], dtype = torch.long, device = hidden_states.device),
            bias = head.bias,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_mean_logits = True,
        )
    return (mean_logits * head.weight.shape[0]).unsqueeze(-1).to(hidden_states.device)


_KTO_IDS = r'batch\["(?:KL_)?(?:completion_)?input_ids"\]\[:, 1:\](?:\.contiguous\(\))?'
_KTO_MASK = r'batch\["(?:KL_)?completion_mask"\]\[:, 1:\]'
_DPO_REF_IDS = r'(?:shift_labels|inputs\["input_ids"\]\[\.\.\., 1:\])'
_DPO_REF_MASK = r'(?:shift_completion_mask|inputs\["completion_mask"\]\[\.\.\., 1:\])'


def _fused_pass_logps_re(ids, mask):
    # TRL 1.6 - 1.13: ``x = selective_log_softmax(logits, ids)`` then ``x[mask == 0] = 0.0``; 1.14: ``row_mask=``.
    return re.compile(
        r"^(?P<i>[ \t]*)(?P<lhs>(?P<out>\w+)(?:, (?P<out2>\w+))?) = "
        r"(?P<fn>selective_log_softmax(?:_and_entropy)?)\(\s*"
        r"(?P<logits>\w+),\s*(?P<ids>" + ids + r"),?\s*"
        r"(?P<ent>entropy_requires_grad=False,\s*)?"
        r"(?:row_mask=(?P<rm>" + mask + r"),?\s*)?\)\n"
        r"(?:(?P=i)(?P=out)\[(?P<zm>" + mask + r") == 0\] = 0\.0\n)?",
        re.MULTILINE,
    )


_KTO_LOGPS_RE = _fused_pass_logps_re(_KTO_IDS, _KTO_MASK)
_DPO_REF_LOGPS_RE = _fused_pass_logps_re(_DPO_REF_IDS, _DPO_REF_MASK)


def _fused_pass_logps_ok(m):
    """One mask, and a two-name target only for TRL 1.14's ``selective_log_softmax_and_entropy``."""
    if (m.group("rm") is None) == (m.group("zm") is None):
        return False
    both = m.group("fn") == "selective_log_softmax_and_entropy"
    return both == (m.group("out2") is not None) == (m.group("ent") is not None)


def _fused_pass_mask(m):
    return m.group("rm") or m.group("zm")


def _fused_pass_else(m):
    """TRL's original statement(s), one level deeper, as the ``else`` branch."""
    i = m.group("i")
    return f"{i}else:\n" + textwrap.indent(textwrap.dedent(m.group(0)), i + "    ")


def _fused_pass_reads_ok(text, name, allowed):
    """Every line of ``text`` that reads ``name`` matches one of ``allowed`` (anything else would see hidden states
    or a stand-in where TRL expects logits)."""
    for line in text.splitlines():
        if re.search(name, line) and not any(re.fullmatch(p, line.strip()) for p in allowed):
            return False
    return True


_KTO_KL_INIT = re.compile(r"^(?P<i>[ \t]*)KL_logps = None\n", re.MULTILINE)
_KTO_KL_FORWARD = re.compile(
    r"^(?P<i>[ \t]*)KL_logits = model\(\*\*KL_model_kwargs\)\.logits\n", re.MULTILINE
)
_KTO_CL_MODE = re.compile(
    r'^(?P<i>[ \t]*)mode = "train" if self\.model\.training else "eval"\n', re.MULTILINE
)
_KTO_CL_FORWARD = re.compile(r"^(?P<i>[ \t]*)outputs = model\(\*\*model_kwargs\)\n", re.MULTILINE)
_KTO_CL_REF_FORWARD = re.compile(
    r"^(?P<i>[ \t]*)ref_outputs = (?P<model>self\.model|self\.ref_model)\(\*\*ref_model_kwargs\)\n",
    re.MULTILINE,
)
_KTO_CL_ENTROPY = re.compile(
    r"^(?P<i>[ \t]*)per_token_entropy = entropy_from_logits\(shift_logits\.detach\(\)\)\n",
    re.MULTILINE,
)
_KTO_CL_SUMS = "outputs.logits.index_select(0, "


def _kto_rewrite_kl(function):
    if len(_KTO_KL_INIT.findall(function)) != 1 or len(_KTO_KL_FORWARD.findall(function)) != 1:
        return function
    logps = [m for m in _KTO_LOGPS_RE.finditer(function) if m.group("logits") == "shift_KL_logits"]
    if (
        len(logps) != 1
        or not _fused_pass_logps_ok(logps[0])
        or "KL_" not in _fused_pass_mask(logps[0])
    ):
        return function
    if logps[0].group("fn") != "selective_log_softmax" or "KL_" not in logps[0].group("ids"):
        return function
    rest = function[: logps[0].start()] + function[logps[0].end() :]
    if not _fused_pass_reads_ok(
        rest,
        r"\bshift_KL_logits\b",
        [r"shift_KL_logits = KL_logits\[:, :-1, :\](?:\.contiguous\(\))?"],
    ) or not _fused_pass_reads_ok(
        rest,
        r"\bKL_logits\b",
        [
            r"KL_logits = model\(\*\*KL_model_kwargs\)\.logits",
            r"shift_KL_logits = KL_logits\[:, :-1, :\](?:\.contiguous\(\))?",
            # TRL 1.9 - 1.13's liger branch builds real logits; the flag stays False there.
            r"KL_logits = KL_hidden_states @ lm_head\.weight\.t\(\)",
            r"KL_logits = KL_logits \+ lm_head\.bias",
        ],
    ):
        return function
    m = logps[0]
    i = m.group("i")
    new = (
        function[: m.start()]
        + f"{i}if _unsloth_kto_kl_hidden:\n"
        + f"{i}    {m.group('out')} = _unsloth_fused_pass_logps(\n"
        + f"{i}        self, _unsloth_kto_kl, {m.group('logits')}, {m.group('ids')}, {_fused_pass_mask(m)}\n"
        + f"{i}    )\n"
        # TRL 1.6's shape gets the truncation clamp kto_trainer_align_completion_logps would have added.
        + _kto_ref_else(function, m)
        + function[m.end() :]
    )
    new = _KTO_KL_INIT.sub(
        lambda m: m.group(0) + f"{m.group('i')}_unsloth_kto_kl_hidden = False\n", new, count = 1
    )
    return _KTO_KL_FORWARD.sub(
        lambda m: (
            f'{m.group("i")}_unsloth_kto_kl = _unsloth_fused_pass_head(self, model, batch, False, "_compute_kl_logps")\n'
            f"{m.group('i')}KL_logits = _unsloth_dpo_forward(model, KL_model_kwargs, _unsloth_kto_kl is not None).logits\n"
            f"{m.group('i')}_unsloth_kto_kl_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_kto_kl, KL_logits)\n"
        ),
        new,
        count = 1,
    )


def _kto_rewrite_loss(function):
    for pattern in (_KTO_CL_MODE, _KTO_CL_FORWARD):
        if len(pattern.findall(function)) != 1:
            return function
    ref_forwards = _KTO_CL_REF_FORWARD.findall(function)
    if not 1 <= len(ref_forwards) <= 2 or len({m[1] for m in ref_forwards}) != len(ref_forwards):
        return function
    matches = list(_KTO_LOGPS_RE.finditer(function))
    policy = [m for m in matches if m.group("logits") == "shift_logits"]
    ref = [m for m in matches if m.group("logits") == "ref_shift_logits"]
    if len(policy) != 1 or len(ref) != 1 or len(matches) != 2:
        return function
    policy, ref = policy[0], ref[0]
    if not (_fused_pass_logps_ok(policy) and _fused_pass_logps_ok(ref)) or ref.group("out2"):
        return function
    if any("KL_" in m.group(0) for m in (policy, ref)):
        return function
    # Which family: 1.14 (fused logps + entropy), 1.9 - 1.13 (entropy_from_logits), 1.6 - 1.8 (logits sums).
    fused_entropy = policy.group("out2") is not None
    entropy = len(_KTO_CL_ENTROPY.findall(function))
    sums = function.count(_KTO_CL_SUMS)
    stats = fused_entropy or entropy == 1
    # 1.8 has both: entropy_from_logits and the logits sums.
    if entropy > 1 or (fused_entropy and entropy) or (fused_entropy and sums) or sums not in (0, 2):
        return function
    if not stats and not sums:
        return function
    rest = function[: policy.start()] + function[policy.end() : ref.start()] + function[ref.end() :]
    shift_def = r"shift_logits = outputs\.logits\[:, :-1, :\](?:\.contiguous\(\))?"
    sliced = re.findall(
        r"^[ \t]*(\w+) = shift_logits\.detach\(\)\.index_select\(0, \w+\)\n", rest, re.MULTILINE
    )
    sliced += re.findall(
        r"^[ \t]*(\w+) = outputs\.logits\.index_select\(0, \w+\)\n", rest, re.MULTILINE
    )
    checks = [
        (
            r"(?<!ref_)\bshift_logits\b",
            [
                shift_def,
                r"per_token_entropy = entropy_from_logits\(shift_logits\.detach\(\)\)",
                r"\w+ = shift_logits\.detach\(\)\.index_select\(0, \w+\)",
            ],
        ),
        (
            r"(?<!ref_)\boutputs\.logits\b",
            [
                shift_def,
                r"device = outputs\.logits\.device",
                r"\w+ = outputs\.logits\.index_select\(0, \w+\)",
            ],
        ),
        (
            r"\bref_shift_logits\b",
            [r"ref_shift_logits = ref_outputs\.logits\[:, :-1, :\](?:\.contiguous\(\))?"],
        ),
        (
            r"\bref_outputs\.logits\b",
            [r"ref_shift_logits = ref_outputs\.logits\[:, :-1, :\](?:\.contiguous\(\))?"],
        ),
    ]
    # The sliced stand-ins are only read as TRL's masked row mean (1.9+) or the full sum (1.6 - 1.8).
    for name in sliced:
        checks.append(
            (
                rf"\b{name}\b",
                [
                    rf"{name} = (?:shift_logits\.detach\(\)|outputs\.logits)\.index_select\(0, \w+\)",
                    rf"\w+ = {name}\[\w+\.bool\(\)\]\.mean\(-1\)\.sum\(\)",
                    rf"self\.accelerator\.gather_for_metrics\({name}\.nansum\(\)\)\.nansum\(\)\.item\(\) / \w+",
                ],
            )
        )
    if len(sliced) != 2 or any(not _fused_pass_reads_ok(rest, n, allowed) for n, allowed in checks):
        return function
    i = policy.group("i")
    policy_new = f"{i}if _unsloth_kto_hidden:\n"
    if sums:
        policy_new += f"{i}    _unsloth_kto_sums = _unsloth_fused_pass_vocab_sums(self, _unsloth_kto_chunked, outputs.logits)\n"
    if stats:
        entropy_name = policy.group("out2") or "_unsloth_kto_entropy"
        policy_new += (
            f"{i}    {policy.group('out')}, {entropy_name}, shift_logits = _unsloth_fused_pass_logps(\n"
            f"{i}        self, _unsloth_kto_chunked, shift_logits, {policy.group('ids')}, {_fused_pass_mask(policy)},\n"
            f"{i}        stats = True, grad = True,\n"
            f"{i}    )\n"
        )
    else:
        policy_new += (
            f"{i}    {policy.group('out')} = _unsloth_fused_pass_logps(\n"
            f"{i}        self, _unsloth_kto_chunked, shift_logits, {policy.group('ids')}, {_fused_pass_mask(policy)},\n"
            f"{i}        grad = True,\n"
            f"{i}    )\n"
        )
    j = ref.group("i")
    ref_new = (
        f"{j}if _unsloth_kto_ref_hidden:\n"
        f"{j}    {ref.group('out')} = _unsloth_fused_pass_logps(\n"
        f"{j}        self, _unsloth_kto_ref, ref_shift_logits, {ref.group('ids')}, {_fused_pass_mask(ref)}\n"
        f"{j}    )\n"
    )
    new = (
        function[: policy.start()]
        + policy_new
        + _fused_pass_else(policy)
        + function[policy.end() : ref.start()]
        + ref_new
        + _fused_pass_else(ref)
        + function[ref.end() :]
    )
    new = _KTO_CL_MODE.sub(
        lambda m: m.group(0)
        + f"{m.group('i')}_unsloth_kto_hidden = _unsloth_kto_ref_hidden = False\n",
        new,
        count = 1,
    )
    new = _KTO_CL_FORWARD.sub(
        lambda m: (
            f'{m.group("i")}_unsloth_kto_chunked = _unsloth_fused_pass_head(self, model, inputs, return_outputs, "_compute_loss")\n'
            f"{m.group('i')}outputs = _unsloth_dpo_forward(model, model_kwargs, _unsloth_kto_chunked is not None)\n"
            f"{m.group('i')}_unsloth_kto_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_kto_chunked, outputs.logits)\n"
        ),
        new,
        count = 1,
    )
    new = _KTO_CL_REF_FORWARD.sub(
        lambda m: (
            f'{m.group("i")}_unsloth_kto_ref = _unsloth_fused_pass_head(self, {m.group("model")}, inputs, False, "_compute_loss")\n'
            f"{m.group('i')}ref_outputs = _unsloth_dpo_forward({m.group('model')}, ref_model_kwargs, _unsloth_kto_ref is not None)\n"
            f"{m.group('i')}_unsloth_kto_ref_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_kto_ref, ref_outputs.logits)\n"
        ),
        new,
    )
    if entropy:
        new = _KTO_CL_ENTROPY.sub(
            lambda m: (
                f"{m.group('i')}if _unsloth_kto_hidden:\n"
                f"{m.group('i')}    per_token_entropy = _unsloth_kto_entropy\n"
                f"{m.group('i')}else:\n"
                f"{m.group('i')}    {m.group(0).lstrip()}"
            ),
            new,
            count = 1,
        )
    if sums:
        new = new.replace(
            _KTO_CL_SUMS,
            "(_unsloth_kto_sums if _unsloth_kto_hidden else outputs.logits).index_select(0, ",
        )
    return new


def kto_trainer_compute_loss_chunked(function_name, function):
    """Rewrite TRL 1.6 - 1.14's KTO ``_compute_kl_logps`` (the no-grad KL pass, run for the policy and the
    reference) and ``_compute_loss`` (policy pass with entropy / ``logits/*`` metrics, and the inline reference
    pass) to score from hidden states with the fused linear op: same completion masks, same metrics, no
    ``[B, S, V]`` logits. Every anchor must match exactly once and every other reader of the logits must be one the
    rewrite keeps exact, else TRL's source is returned untouched; at run time ``_unsloth_fused_pass_head`` keeps
    TRL's path for the kill switch, ``return_outputs``, liger, aux loss, vision, FSDP / DeepSpeed, a non-dense head
    or a forward that ignored the hidden-states flag."""
    if _unsloth_fused_linear_logprobs is None or "_unsloth_kto_" in function:
        return function
    if function_name == "_compute_kl_logps":
        return _kto_rewrite_kl(function)
    if function_name == "_compute_loss":
        return _kto_rewrite_loss(function)
    return function


_DPO_REF_FORWARD = re.compile(
    r"^(?P<i>[ \t]*)ref_outputs = (?P<model>self\.model|self\.ref_model|model)\(\*\*model_kwargs\)\n",
    re.MULTILINE,
)
_DPO_REF_SHIFT = r"ref_shift_logits = ref_outputs\.logits\[\.\.\., :-1, :\](?:\.contiguous\(\))?"


def dpo_trainer_ref_log_probs_chunked(function_name, function):
    """Rewrite TRL 0.29 - 1.14's DPO ``compute_ref_log_probs`` (1.14: ``_compute_ref_log_probs``), the
    ``precompute_ref_log_probs`` pass, to score the reference completions from hidden states. Untouched unless the
    log-prob anchor matches once and the logits have no other reader; each forward picks its own head."""
    if function_name not in ("compute_ref_log_probs", "_compute_ref_log_probs"):
        return function
    if _unsloth_fused_linear_logprobs is None or "_unsloth_dpo_ref" in function:
        return function
    forwards = _DPO_REF_FORWARD.findall(function)
    logps = list(_DPO_REF_LOGPS_RE.finditer(function))
    if not 1 <= len(forwards) <= 3 or len(logps) != 1:
        return function
    m = logps[0]
    if not _fused_pass_logps_ok(m) or m.group("logits") != "ref_shift_logits" or m.group("out2"):
        return function
    rest = function[: m.start()] + function[m.end() :]
    if not _fused_pass_reads_ok(
        rest, r"\bref_shift_logits\b", [_DPO_REF_SHIFT]
    ) or not _fused_pass_reads_ok(rest, r"\bref_outputs\.logits\b", [_DPO_REF_SHIFT]):
        return function
    i = m.group("i")
    new = (
        function[: m.start()]
        + f"{i}if _unsloth_dpo_ref_hidden:\n"
        + f"{i}    {m.group('out')} = _unsloth_fused_pass_logps(\n"
        + f"{i}        self, _unsloth_dpo_ref, ref_shift_logits, {m.group('ids')}, {_fused_pass_mask(m)}\n"
        + f"{i}    )\n"
        + _fused_pass_else(m)
        + function[m.end() :]
    )
    return _DPO_REF_FORWARD.sub(
        lambda f: (
            f'{f.group("i")}_unsloth_dpo_ref = _unsloth_fused_pass_head(self, {f.group("model")}, inputs, False, "{function_name}")\n'
            f"{f.group('i')}ref_outputs = _unsloth_dpo_forward({f.group('model')}, model_kwargs, _unsloth_dpo_ref is not None)\n"
            f"{f.group('i')}_unsloth_dpo_ref_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_dpo_ref, ref_outputs.logits)\n"
        ),
        new,
    )


def _unsloth_kto_ref_chunked_forward(model, want_hidden_states, *args, **kwargs):
    """``model(*args, **kwargs)``, returning hidden states as ``.logits`` when ``want_hidden_states``."""
    if not want_hidden_states:
        return model(*args, **kwargs)
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        return model(*args, **kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior


def _unsloth_kto_ref_batch_logps(self, chunked, hidden_states, labels, label_pad_token_id):
    """TRL < 1.6's ``get_batch_logps(logits, labels, average_log_prob=False)`` (decoder-only) from hidden states."""
    shifted = labels[:, 1:]
    mask = shifted != label_pad_token_id
    return _unsloth_fused_pass_logps(
        self, chunked, hidden_states[:, :-1], shifted.masked_fill(~mask, 0), mask
    ).sum(-1)


_KTO_REF_NAMES = ("compute_reference_log_probs", "compute_ref_log_probs", "_compute_ref_log_probs")
_KTO_REF_LOGPS_RE = _fused_pass_logps_re(
    r'inputs\["(?:KL_)?(?:completion_)?input_ids"\]\[:, 1:\](?:\.contiguous\(\))?',
    r'inputs\["(?:KL_)?completion_mask"\]\[:, 1:\]',
)
# TRL < 1.6: completion_logps = self.get_batch_logps(completion_logits, padded_batch["completion_labels"], ...)
_KTO_REF_BATCH_LOGPS = re.compile(
    r"^(?P<i>[ \t]*)(?P<out>completion_logps|KL_logps) = self\.get_batch_logps\(\n"
    r"[ \t]+(?P<logits>completion_logits|KL_logits),\n"
    r"[ \t]+(?P<labels>padded_batch\[\"(?:KL_)?completion_labels\"\]),\n"
    r"[ \t]+average_log_prob=False,\n"
    r"(?P<rest>(?:[ \t]+(?:is_encoder_decoder=self\.is_encoder_decoder|label_pad_token_id=self\.label_pad_token_id),\n)*)"
    r"[ \t]+\)\n",
    re.MULTILINE,
)
_KTO_REF_ASSIGN = re.compile(
    r"^(?P<i>[ \t]*)(?P<lhs>completion_logits|KL_logits) = (?P<model>self\.model|self\.ref_model|model)\(",
    re.MULTILINE,
)
# A decoder-only forward: positional ids + keyword args, no nested calls (the encoder-decoder ones have ``.get(``).
_KTO_REF_FORWARD = re.compile(
    r"^(?P<i>[ \t]*)(?P<lhs>completion_logits|KL_logits) = (?P<model>self\.model|self\.ref_model|model)\("
    r"(?P<args>[^()]*)\)\.logits\n",
    re.MULTILINE,
)
_KTO_REF_BODY = re.compile(r"^(?P<def>[ \t]*def [^\n]*:\n)(?P<doc>[ \t]+\"\"\"[^\n]*\"\"\"\n)?")


def _kto_ref_else(function, m):
    """TRL's statement as the ``else`` branch, in the form ``kto_trainer_align_completion_logps`` gives it today: TRL
    1.6's ``shift_logits = ...contiguous()`` + statement get the truncation clamp (the shift line stays above the
    ``if``, shared by both branches), every other layout keeps TRL's text."""
    line_start = function.rfind("\n", 0, m.start() - 1) + 1
    pair = function[line_start : m.start()] + m.group(0)
    for regex, repl in ((_KTO_COMPLETION_RE, _kto_completion_repl), (_KTO_KL_RE, _kto_kl_repl)):
        clamped, n = regex.subn(repl, pair)
        if n == 1:
            i = m.group("i")
            body = clamped.split("\n", 1)[1]
            return f"{i}else:\n" + textwrap.indent(textwrap.dedent(body), i + "    ")
    return _fused_pass_else(m)


def kto_trainer_ref_log_probs_chunked(
    function_name,
    function,
    keys = ("KL", "completion"),
):
    """Rewrite TRL 0.22 - 1.14's KTO reference log-prob precompute (``compute_reference_log_probs`` < 1.6,
    ``compute_ref_log_probs`` 1.6 - 1.13, ``_compute_ref_log_probs`` 1.14) to score the completion and KL batches
    from hidden states with the fused linear op instead of ``[B, S, V]`` logits. Every log-prob statement must match
    exactly once, every decoder-only forward must be one the rewrite understands and the logits may have no other
    reader, else TRL's source is returned untouched; each forward picks its own head at run time and keeps TRL's
    path for the kill switch, liger, encoder-decoder, vision, FSDP / DeepSpeed, a hooked or non-dense head or a
    forward that ignored the hidden-states flag. TRL >= 1.15 fuses the head itself (no match)."""
    if function_name not in _KTO_REF_NAMES or _unsloth_fused_linear_logprobs is None:
        return function
    if "_unsloth_kto_ref" in function:
        return function
    body = _KTO_REF_BODY.match(function)
    if body is None:
        return function
    batch_var = "padded_batch" if function_name == "compute_reference_log_probs" else "inputs"
    # Log-prob statements: exactly one for the completions and one for the KL completions.
    old_style = list(_KTO_REF_BATCH_LOGPS.finditer(function))
    new_style = list(_KTO_REF_LOGPS_RE.finditer(function))
    if old_style and new_style:
        return function
    logps = {}
    if old_style:
        if batch_var != "padded_batch":
            return function
        for m in old_style:
            key = "completion" if m.group("logits") == "completion_logits" else "KL"
            if (m.group("out") == "KL_logps") != (key == "KL") or ("KL_" in m.group("labels")) != (
                key == "KL"
            ):
                return function
            logps.setdefault(key, []).append(m)
    else:
        if batch_var != "inputs":
            return function
        for m in new_style:
            if (
                not _fused_pass_logps_ok(m)
                or m.group("out2")
                or m.group("fn") != "selective_log_softmax"
            ):
                return function
            key = {"shift_logits": "completion", "shift_KL_logits": "KL"}.get(m.group("logits"))
            if key is None or ("KL_" in m.group("ids")) != (key == "KL"):
                return function
            if ("KL_" in _fused_pass_mask(m)) != (key == "KL"):
                return function
            logps.setdefault(key, []).append(m)
    # KTO scores a completion and a KL batch; BCO only the completions.
    if sorted(logps) != sorted(keys) or any(len(v) != 1 for v in logps.values()):
        return function
    # Forwards: every assignment of the logits is a decoder-only forward we rewrite or an encoder-decoder one
    # (refused at run time, so its flag stays False).
    forwards = list(_KTO_REF_FORWARD.finditer(function))
    starts = {m.start() for m in forwards}
    for m in _KTO_REF_ASSIGN.finditer(function):
        if m.start() in starts:
            continue
        call = function[m.end() : function.find(".logits", m.end())]
        if "decoder_input_ids" not in call:
            return function
    wanted = {"completion_logits"} | ({"KL_logits"} if "KL" in keys else set())
    if {f.group("lhs") for f in forwards} != wanted:
        return function
    for f in forwards:
        args = f.group("args")
        if f"{batch_var}[" not in args or "**" in args:
            return function
    # The logits (hidden states once rewritten) may only reach the rewritten statements.
    matched = sorted((v[0] for v in logps.values()), key = lambda m: m.start())
    rest = function
    for m in reversed(matched):
        rest = rest[: m.start()] + rest[m.end() :]
    shift = (
        r"shift_(?:KL_)?logits = (?:completion_logits|KL_logits)\[:, :-1, :\](?:\.contiguous\(\))?"
    )
    assign = r"(?:completion_logits|KL_logits) = (?:self\.model|self\.ref_model|model)\(.*"
    if not _fused_pass_reads_ok(rest, r"\b(?:completion_logits|KL_logits)\b", [assign, shift]):
        return function
    if not _fused_pass_reads_ok(rest, r"\bshift_(?:KL_)?logits\b", [shift]):
        return function
    flag = {"completion": "_unsloth_kto_ref_hidden", "KL": "_unsloth_kto_ref_kl_hidden"}
    head = {"completion": "_unsloth_kto_ref", "KL": "_unsloth_kto_ref_kl"}
    new = function
    # Back to front, so earlier offsets stay valid.
    edits = [(m.start(), m.end(), "logps", m) for m in matched]
    edits += [(f.start(), f.end(), "forward", f) for f in forwards]
    for start, end, kind, m in sorted(edits, key = lambda e: e[0], reverse = True):
        i = m.group("i")
        if kind == "forward":
            key = "completion" if m.group("lhs") == "completion_logits" else "KL"
            model = m.group("model")
            args = " ".join(m.group("args").split()).rstrip(",")
            text = (
                f'{i}{head[key]} = _unsloth_fused_pass_head(self, {model}, {batch_var}, False, "{function_name}")\n'
                f"{i}{m.group('lhs')} = _unsloth_kto_ref_chunked_forward(\n"
                f"{i}    {model}, {head[key]} is not None, {args}\n"
                f"{i}).logits\n"
                f"{i}{flag[key]} = _unsloth_dpo_returns_hidden_states(self, {head[key]}, {m.group('lhs')})\n"
            )
        elif old_style:
            key = "completion" if m.group("logits") == "completion_logits" else "KL"
            pad = "self.label_pad_token_id" if "label_pad_token_id" in m.group("rest") else "-100"
            text = (
                f"{i}if {flag[key]}:\n"
                f"{i}    {m.group('out')} = _unsloth_kto_ref_batch_logps(\n"
                f"{i}        self, {head[key]}, {m.group('logits')}, {m.group('labels')}, {pad}\n"
                f"{i}    )\n"
                f"{i}else:\n" + textwrap.indent(textwrap.dedent(m.group(0)), i + "    ")
            )
        else:
            key = "completion" if m.group("logits") == "shift_logits" else "KL"
            text = (
                f"{i}if {flag[key]}:\n"
                f"{i}    {m.group('out')} = _unsloth_fused_pass_logps(\n"
                f"{i}        self, {head[key]}, {m.group('logits')}, {m.group('ids')}, {_fused_pass_mask(m)}\n"
                f"{i}    )\n" + _kto_ref_else(function, m)
            )
        new = new[:start] + text + new[end:]
    # Flags start False: a forward that is not rewritten (encoder-decoder) keeps TRL's statements.
    first = new[body.end() :].split("\n", 1)[0]
    indent = first[: len(first) - len(first.lstrip())]
    return (
        new[: body.end()]
        + f"{indent}_unsloth_kto_ref_hidden = _unsloth_kto_ref_kl_hidden = False\n"
        + new[body.end() :]
    )


# Must precede kto_trainer_align_completion_logps, which would reshape TRL 1.6's KL statement first.
RL_FUNCTIONS["kto_trainer"].insert(
    RL_FUNCTIONS["kto_trainer"].index(kto_trainer_align_completion_logps),
    kto_trainer_compute_loss_chunked,
)
RL_FUNCTIONS["kto_trainer"].insert(
    RL_FUNCTIONS["kto_trainer"].index(kto_trainer_align_completion_logps),
    kto_trainer_ref_log_probs_chunked,
)
RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_ref_log_probs_chunked)
RL_PRE_ITEMS["kto_trainer"].append(inspect.getsource(_unsloth_kto_ref_chunked_forward))
RL_PRE_ITEMS["kto_trainer"].append(inspect.getsource(_unsloth_kto_ref_batch_logps))
for _fused_pass_item in (
    _unsloth_dpo_forward,
    _unsloth_fused_pass_head,
    _unsloth_fused_pass_logps,
    _unsloth_fused_pass_vocab_sums,
):
    RL_PRE_ITEMS["kto_trainer"].append(inspect.getsource(_fused_pass_item))
    if _fused_pass_item is not _unsloth_dpo_forward:
        RL_PRE_ITEMS["dpo_trainer"].append(inspect.getsource(_fused_pass_item))
RL_PRE_ITEMS["kto_trainer"].append(
    "try:\n"
    "    from unsloth_zoo.fused_losses.fused_linear import (\n"
    "        fused_linear_logprobs as _unsloth_fused_linear_logprobs,\n"
    "    )\n"
    "except Exception:\n"
    "    _unsloth_fused_linear_logprobs = None\n"
)


# BCO (TRL 0.22 - 1.10): the policy / reference ``forward`` and the precompute pass are KTO's (TRL < 1.6) layout.
def _unsloth_preference_chunked_off():
    return os.environ.get("UNSLOTH_PREFERENCE_CHUNKED", "1").lower() in ("0", "false", "no", "off")


def _unsloth_bco_chunked_head(self, model, batch, method):
    """``_unsloth_dpo_chunked_head`` that also honours ``UNSLOTH_PREFERENCE_CHUNKED=0``."""
    if _unsloth_preference_chunked_off():
        return None
    return _unsloth_dpo_chunked_head(self, model, batch, method)


def _unsloth_bco_fused_pass_head(self, model, batch, return_outputs, method):
    if _unsloth_preference_chunked_off():
        return None
    return _unsloth_fused_pass_head(self, model, batch, return_outputs, method)


def bco_trainer_chunked_logprobs(function_name, function):
    """BCO's ``forward`` is KTO's TRL < 1.6 ``forward`` (``completion_logits`` is read only by row selection for the
    ``logits/*_sum`` metrics, which the vocab-sum stand-in keeps exact); its ``compute_reference_log_probs`` is KTO's
    precompute without the KL batch. The UDM classifier reads prompt embeddings, never the logits."""
    if function_name == "forward":
        new = kto_trainer_chunked_logprobs(function_name, function)
        old = '_unsloth_dpo_chunked_head(self, model, batch, "forward")'
        if new == function or new.count(old) != 1:
            return function
        return new.replace(old, '_unsloth_bco_chunked_head(self, model, batch, "forward")')
    if function_name == "compute_reference_log_probs":
        new = kto_trainer_ref_log_probs_chunked(function_name, function, keys = ("completion",))
        if new == function:
            return function
        return new.replace(
            "= _unsloth_fused_pass_head(self, ", "= _unsloth_bco_fused_pass_head(self, "
        )
    return function


RL_FUNCTIONS["bco_trainer"].append(bco_trainer_chunked_logprobs)


# Online DPO (TRL 0.22 - 1.x ``_forward``), XPO / Nash-MD (TRL 0.22 - 1.10 ``_compute_logprobs``): score the
# completions from hidden states instead of [B, C, V] logits.
def _unsloth_odpo_chunked_head(self, model, batch, method):
    """``(unwrapped, head, transforms)`` when this pass can score from hidden states, else ``None``."""
    reason = None
    if _unsloth_fused_linear_logprobs is None:
        reason = "unsloth_zoo has no fused_linear_logprobs"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif _unsloth_preference_chunked_off():
        reason = "UNSLOTH_PREFERENCE_CHUNKED=0"
    elif any(k != "attention_mask" and k != "input_ids" for k in batch):
        # Vision rows (pixel_values, image_grid_thw, ...) keep TRL's layout.
        reason = "extra model inputs"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return _unsloth_dpo_chunked_head(self, model, batch, method)


def _unsloth_odpo_hidden_logps(
    self,
    chunked,
    hidden_states,
    completion_ids,
    mask = None,
):
    """TRL's ``log_softmax(head(hidden_states))[completion_ids]`` per position (float32). Rows outside ``mask`` are 0
    (the caller zeroes them anyway). A shape TRL would reject or broadcast gets real logits and TRL's own math."""
    _, head, (multiply, divide, softcap) = chunked
    if hidden_states.dim() != 3 or hidden_states.shape[:2] != completion_ids.shape:
        logits = torch.nn.functional.linear(
            hidden_states.to(device = head.weight.device, dtype = head.weight.dtype),
            head.weight,
            head.bias,
        )
        if multiply:
            logits = logits * multiply
        if divide:
            logits = logits / divide
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return torch.take_along_dim(
            logits.log_softmax(dim = -1), completion_ids.unsqueeze(-1), dim = 2
        ).squeeze(-1)
    logps = _unsloth_fused_linear_logprobs(
        hidden_states,
        head.weight,
        completion_ids,
        bias = head.bias,
        mask = None if mask is None else mask.bool(),
        logit_scale_multiply = multiply,
        logit_scale_divide = divide,
        logit_softcapping = softcap,
    )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    return logps.to(completion_ids.device)


_ODPO_CALL = re.compile(
    r"^(?P<i>[ \t]*)output = model\(prompt_completion_ids, "
    r"(?P<kwargs>attention_mask=prompt_completion_mask|\*\*model_kwargs)\)[ \t]*\n",
    re.MULTILINE,
)
_ODPO_LOGPS = re.compile(
    r"^(?P<i>[ \t]*)logprobs = torch\.take_along_dim\(logits\.log_softmax\(dim=-1\), "
    r"completion_ids\.unsqueeze\(-1\), dim=2\)\.squeeze\(-1\)[ \t]*\n",
    re.MULTILINE,
)
# Between the forward and the log-probs, ``output.logits`` / ``logits`` may only be sliced or row-gathered (the same
# code then slices hidden states): TRL's slice, and online_dpo_trainer__forward's right-align gather.
_ODPO_READS = (
    r"logits = output\.logits\[:, (?:start_idx|prompt_ids\.size\(1\) - 1) ?: ?(?:end_idx|-1)\]",
    r"_unsloth_index = _unsloth_index\.clamp\(0, output\.logits\.size\(1\) - 1\)",
    r"_unsloth_rows, _unsloth_len = output\.logits\.shape\[:2\]",
    r"logits = output\.logits\.reshape\(_unsloth_rows \* _unsloth_len, -1\)\.index_select\(",
    r"\).view\(_unsloth_rows, -1, output\.logits\.size\(-1\)\)",
)


def online_dpo_trainer_chunked_logprobs(function_name, function):
    """Rewrite Online DPO's ``_forward``: the forward returns hidden states, TRL's slice (or Unsloth's right-align
    gather) picks the completion positions, and the fused op scores every one of them (``objective/kl`` and
    ``objective/entropy`` read the padding positions too). No temperature: TRL scores raw logits. Untouched unless
    both anchors match once and nothing else reads the logits."""
    if function_name != "_forward" or _unsloth_fused_linear_logprobs is None:
        return function
    if "_unsloth_odpo" in function:
        return function
    calls = list(_ODPO_CALL.finditer(function))
    logps = list(_ODPO_LOGPS.finditer(function))
    if len(calls) != 1 or len(logps) != 1 or calls[0].end() > logps[0].start():
        return function
    between = function[calls[0].end() : logps[0].start()]
    code = "\n".join(line for line in between.splitlines() if not line.lstrip().startswith("#"))
    if not _fused_pass_reads_ok(code, r"\blogits\b", _ODPO_READS):
        return function
    if (
        re.search(r"\blogits\b", function[logps[0].end() :])
        or "output.logits" in function[: calls[0].start()]
    ):
        return function
    c, m = calls[0], logps[0]
    i = c.group("i")
    kwargs = (
        "model_kwargs"
        if c.group("kwargs") == "**model_kwargs"
        else "dict(attention_mask=prompt_completion_mask)"
    )
    forward = (
        f"{i}_unsloth_odpo_kwargs = {kwargs}\n"
        f'{i}_unsloth_odpo = _unsloth_odpo_chunked_head(self, model, _unsloth_odpo_kwargs, "_forward")\n'
        f"{i}output = _unsloth_dpo_model_forward(\n"
        f"{i}    model, prompt_completion_ids, _unsloth_odpo_kwargs, _unsloth_odpo is not None\n"
        f"{i})\n"
        f"{i}_unsloth_odpo_hidden = _unsloth_dpo_returns_hidden_states(self, _unsloth_odpo, output.logits)\n"
    )
    j = m.group("i")
    score = (
        f"{j}if _unsloth_odpo_hidden:\n"
        f"{j}    logprobs = _unsloth_odpo_hidden_logps(self, _unsloth_odpo, logits, completion_ids)\n"
        f"{j}else:\n" + textwrap.indent(textwrap.dedent(m.group(0)), j + "    ")
    )
    return function[: c.start()] + forward + between + score + function[m.end() :]


_XPO_LOGPS = re.compile(
    r"^(?P<i>[ \t]*)output = m\(data\[\"input_ids\"\], attention_mask=data\[\"attention_mask\"\]\)\n"
    r"(?P=i)logits = output\.logits\[:, context_length - 1 : -1\]\n"
    r"(?P=i)token_logprobs = selective_log_softmax\(logits, data\[\"input_ids\"\]\[:, context_length:\]\)\n"
    r"(?P=i)return token_logprobs\n",
    re.MULTILINE,
)


def xpo_trainer_chunked_logprobs(function_name, function):
    """XPO / Nash-MD ``_compute_logprobs``: each ``compute_logprobs_for_data`` pass scores the completion tokens from
    hidden states. Padding rows come back 0, which is what TRL's ``masked_fill(padding_mask, 0.0)`` makes them."""
    if function_name != "_compute_logprobs" or _unsloth_fused_linear_logprobs is None:
        return function
    if "_unsloth_odpo" in function:
        return function
    found = list(_XPO_LOGPS.finditer(function))
    if len(found) != 1 or len(re.findall(r"\blogits\b", function)) != 3:
        return function
    if (
        len(
            re.findall(
                r"model_padding_mask = model_data\[\"attention_mask\"\]\[:, context_length:\] == 0",
                function,
            )
        )
        != 1
    ):
        return function
    i = found[0].group("i")
    new = (
        f'{i}_unsloth_odpo_kwargs = dict(attention_mask=data["attention_mask"])\n'
        f'{i}_unsloth_odpo = _unsloth_odpo_chunked_head(self, m, _unsloth_odpo_kwargs, "_compute_logprobs")\n'
        f"{i}output = _unsloth_dpo_model_forward(\n"
        f'{i}    m, data["input_ids"], _unsloth_odpo_kwargs, _unsloth_odpo is not None\n'
        f"{i})\n"
        f"{i}logits = output.logits[:, context_length - 1 : -1]\n"
        f"{i}if _unsloth_dpo_returns_hidden_states(self, _unsloth_odpo, output.logits):\n"
        f"{i}    return _unsloth_odpo_hidden_logps(\n"
        f'{i}        self, _unsloth_odpo, logits, data["input_ids"][:, context_length:],\n'
        f'{i}        data["attention_mask"][:, context_length:] != 0,\n'
        f"{i}    )\n"
        f'{i}token_logprobs = selective_log_softmax(logits, data["input_ids"][:, context_length:])\n'
        f"{i}return token_logprobs\n"
    )
    return function[: found[0].start()] + new + function[found[0].end() :]


# After online_dpo_trainer__forward, whose right-align gather the rewrite reuses on hidden states.
RL_FUNCTIONS["online_dpo_trainer"].append(online_dpo_trainer_chunked_logprobs)
RL_FUNCTIONS["xpo_trainer"].append(xpo_trainer_chunked_logprobs)
RL_FUNCTIONS["nash_md_trainer"].append(xpo_trainer_chunked_logprobs)
for _odpo_trainer in ("bco_trainer", "online_dpo_trainer", "xpo_trainer", "nash_md_trainer"):
    for _odpo_item in (
        _unsloth_get_model_config,
        _unsloth_text_configs,
        _unsloth_get_final_logit_softcapping,
        _unsloth_resolve_logit_scales,
        _unsloth_grpo_returns_hidden_states,
        _unsloth_grpo_hidden_states_signal,
        _unsloth_pref_returns_hidden_states,
        _unsloth_gkd_dense_head,
        _unsloth_gkd_logit_transforms,
        _unsloth_pref_logit_transforms,
        _unsloth_stock_logps_hook,
        _unsloth_dpo_chunked_head,
        _unsloth_dpo_model_forward,
        _unsloth_dpo_returns_hidden_states,
        _unsloth_preference_chunked_off,
    ):
        RL_PRE_ITEMS[_odpo_trainer].append(inspect.getsource(_odpo_item))
    if _odpo_trainer == "bco_trainer":
        for _odpo_item in (
            _unsloth_kto_hidden_logps,
            _unsloth_bco_chunked_head,
            _unsloth_kto_ref_chunked_forward,
            _unsloth_kto_ref_batch_logps,
            _unsloth_fused_pass_head,
            _unsloth_bco_fused_pass_head,
            _unsloth_fused_pass_logps,
        ):
            RL_PRE_ITEMS[_odpo_trainer].append(inspect.getsource(_odpo_item))
    else:
        for _odpo_item in (_unsloth_odpo_chunked_head, _unsloth_odpo_hidden_logps):
            RL_PRE_ITEMS[_odpo_trainer].append(inspect.getsource(_odpo_item))
    RL_PRE_ITEMS[_odpo_trainer].append(
        "try:\n"
        "    from unsloth_zoo.device_map_planner import detect_logit_transforms\n"
        "except Exception:\n"
        "    detect_logit_transforms = None\n"
        "try:\n"
        "    from unsloth_zoo.fused_losses.chunked_logprob import (\n"
        "        unsloth_chunked_logprobs as _unsloth_chunked_logprobs,\n"
        "        unsloth_masked_logprobs as _unsloth_masked_logprobs,\n"
        "    )\n"
        "except Exception:\n"
        "    _unsloth_chunked_logprobs = _unsloth_masked_logprobs = None\n"
        "try:\n"
        "    from unsloth_zoo.fused_losses.preference_loss import linear_mean_logits as _unsloth_linear_mean_logits\n"
        "except Exception:\n"
        "    _unsloth_linear_mean_logits = None\n"
        "try:\n"
        "    from unsloth_zoo.fused_losses.fused_linear import (\n"
        "        fused_linear_logprobs as _unsloth_fused_linear_logprobs,\n"
        "    )\n"
        "except Exception:\n"
        "    _unsloth_fused_linear_logprobs = None\n"
        f"_DPO_VISION_KEYS = {tuple(_DPO_VISION_KEYS)!r}\n"
    )
