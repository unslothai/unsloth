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
from collections import defaultdict
from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
from unsloth import DEVICE_TYPE

RL_EXTRA_ARGS      = defaultdict(list)
RL_FUNCTIONS       = defaultdict(list)
RL_PRE_ITEMS       = defaultdict(list)
RL_CONFIG_CHANGES  = defaultdict(list)
RL_METRICS_CHANGES = defaultdict(list)

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

# Check untrained tokens
def sft_trainer_fix_untrained_tokens(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_tokenizer = \
        "IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\\n')\n"\
        "from unsloth_zoo.tokenizer_utils import fix_untrained_tokens\n"\
        "from unsloth_zoo.training_utils  import fix_zero_training_loss\n"\
        "if 'tokenizer' not in locals(): tokenizer = processing_class\n"\
        "fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n"\
        "fix_zero_training_loss(model, tokenizer, train_dataset)\n"
        return fix_tokenizer
    return ""
pass
RL_EXTRA_ARGS["sft_trainer"].append(sft_trainer_fix_untrained_tokens)


# Remove DPO columns which might randomnly be tokenized
def dpo_trainer_fix_columns(call_args, extra_args):
    if "model" in call_args and "train_dataset" in call_args:
        fix_dpo = \
        "if hasattr(train_dataset, 'column_names'):\n"\
        "    column_names = set(train_dataset.column_names)\n"\
        "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"\
        "             'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"\
        "             'prompt_input_ids', 'prompt_attention_mask']\n"\
        "    if all(x in column_names for x in check):\n"\
        "        train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"\
        "    del check, column_names\n"
        return fix_dpo
    return ""
pass
RL_EXTRA_ARGS["dpo_trainer"].append(dpo_trainer_fix_columns)


# Fix tokenizer double BOS
def sft_trainer_prepare_dataset(function_name, function):
    if  function_name != "_prepare_non_packed_dataloader" and \
        function_name != "_prepare_dataset": return function

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
            function = function.split("\n")
            function = "\n".join(" "*4 + x for x in function)
            function = function.replace("def sft_prepare_dataset", "def _prepare_dataset")
            return function
        pass
    pass

    check_text = \
    "if 'skip_prepare_dataset' in locals() and skip_prepare_dataset:\n"\
    "    return dataset\n"\
    "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"\
    "if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')\n"\
    "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"\
    "if 'dataset_text_field' not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `dataset_text_field` does not exist!')\n"\
    "test_text = dataset[0][dataset_text_field] if (formatting_func is None and dataset_text_field is not None) else formatting_func(dataset[0])[0]\n"\
    "chat_template = getattr(tokenizer, 'chat_template', None)\n"\
    "chat_template = '' if chat_template is None else chat_template\n"\
    "has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) "\
    "if getattr(tokenizer, 'bos_token', None) is not None else False\n"\
    "if 'add_special_tokens' not in locals() and has_bos_token_already:\n"\
    "    from functools import partial\n"\
    "    tokenizer_call = tokenizer.__call__\n"\
    "    tokenizer.__call__ = partial(tokenizer_call, add_special_tokens = False)\n"\
    "    processing_class = tokenizer\n"\
    "else:\n"\
    "    tokenizer_call = None\n"\
    "    add_special_tokens = False if has_bos_token_already else locals().get('add_special_tokens', False)\n"

    check_text = check_text.split("\n")
    check_text = "\n".join(" "*8 + x for x in check_text)
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
    pass

    # Return tokenizer's original state
    return_state = "if tokenizer_call is not None: tokenizer.__call__ = tokenizer_call\n"
    function = re.sub(
        r"\n([ ]{4,})(return .*?[\s]{0,})$",
        rf"\1{return_state}\1\2",
        function,
    )
    return function
pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_prepare_dataset)


# Ignore mean_token_accuracy since it needs logits
# We override it directly with our version
def sft_trainer_compute_loss(function_name, function):
    if  function_name != "compute_loss": return function

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs = return_outputs,
            num_items_in_batch = num_items_in_batch,
        )
        return outputs
    pass

    function = inspect.getsource(compute_loss)
    return function
pass
RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)


# Autocast precision for GRPO
def grpo_trainer__prepare_inputs(function_name, function):
    if  function_name != "_prepare_inputs": return function

    import re
    # This matches the function signature, decorators and any comments immediately following
    pattern = r"(\s*@profiling_decorator\s*\n\s*def _prepare_inputs\s*\([^\)]*\)\s*(->\s*[^:]+)?\s*:\s*\n(?:[ ]*#[^\n]*\n)*)"

    match = re.search(pattern, function)
    insert = (
        "        if hasattr(self, 'llm'):\n"
        "           if getattr(self.llm.llm_engine.vllm_config.model_config, 'enable_sleep_mode', False):\n"
        "               self.llm.wake_up()\n"
    )
    if match:
        header_and_comments = match.group(1)
        # Find where the code block starts after comments
        code_start_index = match.end(1)
        rest_of_function = function[code_start_index:]

        # Remove any old wake_up call that might be at the start of the function body
        rest_of_function = re.sub(
            r"^\s*if hasattr\(self, 'llm'\):.*?self\.llm\.wake_up\(\).*?\n",
            "",
            rest_of_function,
            flags=re.DOTALL | re.MULTILINE
        )

        # We also need to remove the old wake up call from the beginning of the function
        # since it's injected before the comments.
        header_and_comments = re.sub(
            r"(:\s*\n)\s*if hasattr\(self, 'llm'\):.*?self\.llm\.wake_up\(\).*?\n",
            r"\1",
            header_and_comments,
            flags=re.DOTALL | re.MULTILINE
        )

        function = header_and_comments + insert + rest_of_function
    # Add mixed precision training
    function = function.replace(
        "with torch.inference_mode():",
        "with torch.inference_mode(), "\
        "torch.amp.autocast(device_type = 'cuda', "\
        "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "\
        "if not torch.is_autocast_enabled('cuda') else nullcontext())"\
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):",
    )
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    sleep_and_cache = (
        "if hasattr(self, 'llm'):\n"
        "            if getattr(self.llm.llm_engine.vllm_config.model_config, 'enable_sleep_mode', False):\n"
        "                self.llm.sleep(os.environ.get('VLLM_SLEEP_MODE', 1))\n"
        "        "
    )
    if re.search(r"\n\s*return ", function):
        function = re.sub(r"(\n\s*)return ", f"\\1{sleep_and_cache}return ", function, count=1)
    else:
        function = function.rstrip() + "\n    " + sleep_and_cache
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__prepare_inputs)


# Fix incorrect special tokens handling and truncation in older TRL versions
def grpo_trainer__generate_and_score_completions(function_name, function):
    if  function_name != "_generate_and_score_completions": return function

    # TRL 0.19.0 did skip_special_tokens = True which should be False
    function = function.replace(
        "prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False",
        "prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False",
    )

    # Always between max_prompt_length and use_vllm
    found = re.findall(
        r"\n(([ ]{8,})if self\.max_prompt_length is not None:.*?"\
        r"\2if self\.use_vllm:)",
        function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(found) != 0:
        replace_part, spacing = found[0]
        removed_comments = re.sub(r"\#[^\n]{1,}", "", replace_part)
        splits = removed_comments.split("\n")
        if sum(re.match(rf"{spacing}[^\s]", x) is not None for x in splits) == 2 and len(spacing) >= 8:

            new_replacement = \
            f"""\n{spacing}if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            pad_token = self.processing_class.pad_token
            def strip_leading_tokens(text):
                while text.startswith(pad_token):
                    text = text.removeprefix(pad_token)
                return text

            if pad_token is not None:
                prompts_text = [
                    strip_leading_tokens(text) for text in prompts_text
                ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:"""
            function = function.replace(replace_part, new_replacement)
    pass
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__generate_and_score_completions)


# Remove _move_model_to_vllm
def grpo_trainer__move_model_to_vllm(function_name, function):
    if  function_name != "_move_model_to_vllm": return function

    def _move_model_to_vllm(self, *args, **kwargs): return None

    function = inspect.getsource(_move_model_to_vllm)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__move_model_to_vllm)


# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer__get_per_token_logps(function_name, function):
    if function_name != "_get_per_token_logps": return function

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        if True: # os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return None # Unsloth efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, '_autocast_dtype'):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': self._autocast_dtype = torch.float16

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
        pass
    pass

    function = inspect.getsource(_get_per_token_logps)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps)

def grpo_trainer__get_per_token_logps_and_entropies(function_name, function):
    if function_name != "_get_per_token_logps_and_entropies": return function

    # Just copy over from _get_per_token_logps replacement function above. For now this returns None anyway
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size = None, compute_entropy = False, *args, **kwargs):
        if True: # os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
            return {"logps": None, "entropies": None} # Unsloth efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, '_autocast_dtype'):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': self._autocast_dtype = torch.float16

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(device_type = 'cuda', dtype = self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                logits_to_keep = logits_to_keep + 1,
            ).logits

            entropies = None
            if compute_entropy:
                from trl.trainer.utils import entropy_from_logits
                entropies = entropy_from_logits(logits)

            # logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            return {"logps": logits, "entropies": entropies}
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
        pass
    pass

    function = inspect.getsource(_get_per_token_logps_and_entropies)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer__get_per_token_logps_and_entropies)

grpo_compute_loss      = RL_REPLACEMENTS["grpo_compute_loss"]
grpo_compute_loss_slow = RL_REPLACEMENTS["grpo_compute_loss_slow"]
UnslothEfficientGRPO   = RL_REPLACEMENTS["UnslothEfficientGRPO"]
grpo_accumulated_loss  = RL_REPLACEMENTS["grpo_accumulated_loss"]
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_compute_loss))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(UnslothEfficientGRPO))
RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource(grpo_accumulated_loss))
RL_PRE_ITEMS["grpo_trainer"].append(grpo_compute_loss_slow)

# Edit _get_per_token_logps to handle mixed precision
def grpo_trainer_compute_loss(function_name, function):
    if  function_name != "compute_loss": return function

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = None
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = \
            lambda model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False: \
            self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep) \
            if hasattr(self, "_get_per_token_logps") else \
            self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy)['logps']

        per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        if self.beta != 0.0:
            with torch.inference_mode(), model.disable_adapter():
                ref_per_token_logps = per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)
        else:
            ref_per_token_logps = None
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        old_hidden_states = inputs.get("old_per_token_logps", None)
        input_ids = input_ids[:, -logits_to_keep:]

        # Get logit softcapping and logit scale
        logit_softcapping = getattr(model.config, "final_logit_softcapping", 0) # Gemma
        if logit_softcapping is None: logit_softcapping = 0
        logit_scale_multiply = getattr(model.config, "logit_scale", 0) # Cohere
        if logit_scale_multiply is None: logit_scale_multiply = 0
        logit_scale_divide = getattr(model.config, "logits_scaling", 0) # Granite
        if logit_scale_divide is None: logit_scale_divide = 0


        if per_token_logps is not None:

            if ref_per_token_logps is not None:
                ref_per_token_logps = ref_per_token_logps[:, :-1, :] # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            per_token_logps = per_token_logps[:, :-1, :] # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            loss, completion_length, mean_kl = grpo_compute_loss_slow(
                ref_per_token_logps,
                per_token_logps,
                old_hidden_states,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
                loss_type = self.args.loss_type,
                epsilon_low = self.epsilon_low,
                epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = self.args.delta,
                temperature = self.args.temperature,
                logit_softcapping = logit_softcapping,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
            )
        else:
            if hasattr(self.args, "loss_type"):
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_hidden_states = old_hidden_states,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = self.args.delta,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                )
            else:
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_hidden_states = old_hidden_states,
                    n_chunks = self.args.unsloth_num_chunks,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                )
            pass
        pass
        # Log the metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())
        return loss
    pass

    function = inspect.getsource(compute_loss)
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_trainer_compute_loss)

# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L356
# TRL warns if batch size is not a multiple of num_generations -> fix this.
def grpo_trainer_fix_batch_size(RLTrainer_source, RLConfig_source):
    if "divisible by the number of generations" not in RLTrainer_source: return ""
    if "num_generations" not in RLConfig_source: return ""

    check_batch_size = \
    "div = per_device_train_batch_size // num_generations\n"\
    "if div * num_generations != per_device_train_batch_size:\n"\
    "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"\
               "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"\
    "    per_device_train_batch_size = num_generations\n"
    return check_batch_size
pass
RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_trainer_fix_batch_size)


# Add other reward function names
def grpo_trainer_metrics(RLTrainer_source, RLConfig_source):
    if "reward_funcs" not in RLTrainer_source: return ""

    # For new TRL we have /mean and /std
    use_mean = "rewards/{reward_func_name}/mean" in RLTrainer_source
    use_std  = "rewards/{reward_func_name}/std"  in RLTrainer_source
    if not use_mean:
        use_normal = "rewards/{reward_func_name}" in RLTrainer_source
    else:
        use_normal = False
    pass

    log_metrics = \
    "if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]\n"\
    "else: _reward_funcs = reward_funcs\n"\
    "for reward_func in _reward_funcs:\n"\
    "    try:\n"\
    "        reward_func_name = reward_func.__name__\n"\
   f"        if {use_mean}:\n"\
    "            other_metrics.append(f'rewards/{reward_func_name}/mean')\n"\
   f"        if {use_std}:\n"\
    "            other_metrics.append(f'rewards/{reward_func_name}/std')\n"\
   f"        if {use_normal}:\n"\
    "            other_metrics.append(f'rewards/{reward_func_name}')\n"\
    "    except: pass\n"
    return log_metrics
pass
RL_METRICS_CHANGES["grpo_trainer"].append(grpo_trainer_metrics)
