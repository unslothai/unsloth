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
def grpo_generate_and_score_completions(function_name, function):
    if  function_name != "_generate_and_score_completions": return function

    def _generate_and_score_completions(
        self, inputs) :
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)['prompt'] for example in inputs]
        if not self.use_vision:
           pixel_values = None
           image_grid_thw = None
           prompt_inputs = self.processing_class(text=prompts_text, return_tensors='pt', padding=True,padding_side="left", add_special_tokens=False)
           prompt_inputs = super()._prepare_inputs(prompt_inputs)
        else:
           images = [x['image'] for x in inputs] # Only image inputs support for now 
           prompt_inputs = self.processing_class(images = images, text=prompts_text, return_tensors='pt', padding=True,padding_side="left", add_special_tokens=False)
           prompt_inputs = super()._prepare_inputs(prompt_inputs)
           pixel_values, image_grid_thw = prompt_inputs['pixel_values'], prompt_inputs['image_grid_thw']

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                if self.use_vision : prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask,pixel_values = pixel_values,image_grid_thw=image_grid_thw, generation_config=self.generation_config)
                else : prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config)

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask,pixel_values,image_grid_thw, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None


        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode(), torch.amp.autocast(device_type = 'cuda', dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) if not torch.is_autocast_enabled('cuda') else nullcontext())if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }

    function = inspect.getsource(_generate_and_score_completions)
    # Add mixed precision training
    function = function.replace(
        "with torch.inference_mode():",

        "with torch.inference_mode(), "\
        "torch.amp.autocast(device_type = 'cuda', "\
        "dtype = ((torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16) "\
        "if not torch.is_autocast_enabled('cuda') else nullcontext())"\
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '0' else torch.float16):",
    )

    # Disable attaching a float32 conversion hook which upcasts logits to FP32
    function = function.replace(
        "self.accelerator.unwrap_model(self.model)",
        "self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)",
    )
    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_generate_and_score_completions)

def grpo_prepare_inputs(function_name, function):
    if  function_name != "_prepare_inputs": return function

    if "generation_batch = self._generate_and_score_completions(generation_batch)" not in function : return function
    
    function = function.replace(
        "generation_batch = self._generate_and_score_completions(generation_batch)",
        
        "generation_batch = self._generate_and_score_completions(generation_batch)\n"\
        "                if self.use_vision : generation_batch['pixel_values']=generation_batch['pixel_values'].view(generation_batch['prompt_ids'].size(0), -1, generation_batch['pixel_values'].size(1)) # (batch_size * n_patches, dim embedding)->(batch_size,n_patches,dim embeddding)"
    )

    return function
pass
RL_FUNCTIONS["grpo_trainer"].append(grpo_prepare_inputs)


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
    if  function_name != "_get_per_token_logps": return function

    def _get_per_token_logps(self, model, input_ids, attention_mask,pixel_values,image_grid_thw, logits_to_keep, calc_logprob_flag = None):
        if os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0' and  not calc_logprob_flag:
            return None # Unsloth efficient GRPO
        # Otherwise, calculate normally:
        if not hasattr(self, '_autocast_dtype'):
            self._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
            if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': self._autocast_dtype = torch.float16

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        with torch.amp.autocast(device_type = 'cuda', dtype = self._autocast_dtype):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            if self.use_vision : 
               hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, logits_to_keep=logits_to_keep + 1).logits
            else:
                hidden_states = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
            #logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            hidden_states = hidden_states[:, :-1, :]
            hidden_states = hidden_states[:, -logits_to_keep:]
            return hidden_states
            # input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            # logits = logits[:, -logits_to_keep:]
            # return logits
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
        pixel_values,image_grid_thw = inputs.get("pixel_values", None), inputs.get("image_grid_thw", None)
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = None
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask,pixel_values,image_grid_thw, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        if self.beta != 0.0:
            with torch.inference_mode(), model.disable_adapter():
                ref_per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask,pixel_values,image_grid_thw, logits_to_keep)
        else:
            ref_per_token_logps = None
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        if "old_per_token_logps" in inputs.keys():
            old_hidden_states = inputs["old_per_token_logps"]
        else: 
            old_hidden_states = None
        input_ids = input_ids[:, -logits_to_keep:]
        if per_token_logps is not None:
            loss, completion_length, mean_kl = grpo_compute_loss_slow(
                ref_per_token_logps, per_token_logps, old_hidden_states, input_ids, completion_mask, self.beta, advantages, 
                loss_type = self.args.loss_type,
                epsilon_low = self.epsilon_low, epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = self.args.delta,
            )
        else:
            if hasattr(self.args, "loss_type"):
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    self, _input_ids, logits_to_keep, completion_mask, advantages, old_hidden_states,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    epsilon_low = self.epsilon_low, epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = self.args.delta,
                )
            else:
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    self, _input_ids, logits_to_keep, completion_mask, advantages, old_hidden_states,
                    n_chunks = self.args.unsloth_num_chunks,
                )    

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
