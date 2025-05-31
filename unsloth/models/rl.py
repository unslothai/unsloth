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
    "PatchFastRL",
    "vLLMSamplingParams",
]

import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import os
import re
import torch
from unsloth_zoo.compiler import create_new_function
from unsloth_zoo.logging_utils import PatchRLStatistics
from unsloth_zoo.rl_replacements import RL_REPLACEMENTS
from .rl_replacements import (
    RL_EXTRA_ARGS,
    RL_FUNCTIONS,
    RL_PRE_ITEMS,
    RL_CONFIG_CHANGES,
    RL_METRICS_CHANGES,
)
selective_log_softmax = RL_REPLACEMENTS["selective_log_softmax"]

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False, # Disable Triton mm kernels
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

from trl import __version__ as trl_version

def vLLMSamplingParams(**kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(**kwargs)
    sampling_params._set_kwargs = kwargs
    return sampling_params
pass

def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from trl.trainer.utils import first_true_indices, get_reward
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        if "UnslothPolicyAndValueWrapper" in str(type(model)):
            model = model.policy

        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            # Put the model in inference mode.
            FastLanguageModel.for_inference(unwrapped_model)

            # We must use .clone for Unsloth since we force inference_mode
            # Rather we should have used no_grad
            original_generate = unwrapped_model.generate
            def generate_with_clone(*args, **kwargs):
                out = original_generate(*args, **kwargs)
                #breakpoint()
                if isinstance(out, torch.Tensor):
                    return out.clone()
                return out
            pass
            unwrapped_model.generate = generate_with_clone

            try:
                yield unwrapped_model
            finally:
                # Restore generate and return
                unwrapped_model.generate = original_generate
                FastLanguageModel.for_training(model)
            pass
        pass
    pass
    def unsloth_get_reward(
        model, query_responses, pad_token_id, context_length):
            attention_mask = query_responses != pad_token_id
            position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
            lm_backbone = getattr(model, model.base_model_prefix)
            input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
            FastLanguageModel.reset_functions()
            output = lm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,  # otherwise mistral-based RM would error out
            )
            reward_logits = model.score(output.hidden_states[-1])
            sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
            FastLanguageModel.set_functions()
            # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
            return (
                reward_logits,
                reward_logits[
                    torch.arange(reward_logits.size(0), device=reward_logits.device),
                    sequence_lengths,
                ].squeeze(-1),
                sequence_lengths,
            )

    from transformers import Trainer
    from transformers.utils import is_sagemaker_mp_enabled
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp
        from smdistributed.modelparallel import __version__ as SMP_VERSION

        IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

        from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
    else:
        IS_SAGEMAKER_MP_POST_1_10 = False
    from transformers.trainer_pt_utils import nested_detach
    @torch.no_grad()    
    def unsloth_prediction_step(self, model, inputs, prediction_loss_only,ignore_keys,):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        #breakpoint()
                        tokenized_output = self.processing_class(inputs["prompt"], padding=True, truncation=True, return_tensors="pt")
                        outputs = model(**tokenized_output)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    
    from trl.trainer.utils import generate, batch_generation,pad, forward
    from transformers import GenerationConfig
 
    def unsloth_forward(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
    ) -> torch.nn.Module:
        """
        Performs a forward pass through the model with the given query responses and pad token ID.

        Args:
            model (`torch.nn.Module`):
                The model to perform the forward pass.
            query_responses (`torch.Tensor`):
                The tensor containing the query responses.
            pad_token_id (`int`):
                The token ID representing the pad token.

        Returns:
            `torch.nn.Module`:
                The output of the model, including hidden states.
        """
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids = position_ids)

        return outputs

    @torch.no_grad()
    def unsloth_batch_generation(
        model: torch.nn.Module,
        queries,
        local_rollout_forward_batch_size: int,
        pad_token_id: int,
        eos_token_id: int, 
        processing_class,
        generation_config: GenerationConfig,
    ):
        logitss = []

        context_length =queries.shape[1]
        batch_size =  queries.shape[0]
        from unsloth import FastLanguageModel

        sampling_params = vLLMSamplingParams()
        # Ensure sampling is enabled and update necessary parameters
        sampling_params.temperature = generation_config.temperature
        sampling_params.max_tokens = generation_config.max_new_tokens
        sampling_params.top_p = 1.0
        sampling_params.top_k = -1
        
        sampling_params.min_tokens = 0 

        # Load the latest weights
        tokenized_query = queries
        queries = [processing_class.decode(query) for query in queries]

        pass

        pass

        outputs = model.fast_generate(queries, sampling_params, use_tqdm=False, lora_request = model.load_lora('ppo_trainer_lora_model', load_tensors = True))

        completion_ids = [list(output.outputs[i].token_ids) for i in range(1) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(1) for output in outputs]

        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = sampling_params.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]

        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]


        completion_ids = torch.tensor(completion_ids, device=model.device)

        padded_query_responses = torch.cat((tokenized_query, completion_ids), dim=1)

        FastLanguageModel.for_training(model)
        for i in range(0, batch_size, local_rollout_forward_batch_size):
            padded_query_response = padded_query_responses[i : i + local_rollout_forward_batch_size]
            #breakpoint()
            output = forward(model, padded_query_response, pad_token_id)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= generation_config.temperature + 1e-7
            logitss.append(logits)
        

        concatenated_tensor = torch.cat(logitss, dim=0)

        return padded_query_responses, concatenated_tensor
     
    import trl.trainer
    trainers = dir(trl.trainer)

    trainers = [x for x in trainers if x.endswith("_trainer")]
    trainers.append("callbacks")
    unwrap = "unwrap_model_for_generation"
    _get_reward = "get_reward"
    batch_gen  = "batch_generation"
    forward_function = "forward"
    #This is only for RLOO and PPO

    for trainer in trainers:
        #breakpoint()
        try: current_trainer = eval(f"trl.trainer.{trainer}")
        except: continue

        if hasattr(current_trainer, _get_reward):
            try: exec(f"trl.trainer.{trainer}.{_get_reward} = unsloth_{_get_reward}")
            except: continue

        if trainer == "rloo_trainer" or trainer == "ppo_trainer":
            if hasattr(current_trainer, batch_gen):
                try: 
                    exec(f"trl.trainer.{trainer}.{batch_gen} = unsloth_{batch_gen}")
                    exec(f"trl.trainer.{trainer}.{forward_function} = unsloth_{forward_function}")
                except: continue
        if hasattr(current_trainer, unwrap):
            try: exec(f"trl.trainer.{trainer}.{unwrap} = unsloth_{unwrap}")
            except: continue

        if hasattr(current_trainer, _get_reward):
            try: exec(f"trl.trainer.{trainer}.{_get_reward} = unsloth_{_get_reward}")
            except: continue
    exec(f"Trainer.prediction_step=unsloth_prediction_step")
    pass
pass
#To do, also in trl/trainer/callbacks.py make that unwrap compatible

RLTrainer_replacement =  '''
import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import torch.nn as nn
from trl.trainer.utils import forward

from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling

torch_compile_options = {{
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}}

{selective_log_softmax_code}
{RL_pre}

@dataclass
class Unsloth{RLConfig_name}({RLConfig_name}):
    """
    {__RLConfig_doc__}
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {{'help': 'vLLM SamplingParams'}},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {{'help': 'Chunk size to reduce memory usage. -1 is most efficient.'}},
    )
    def __init__({RLConfig_arguments},
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
{RLConfig_extra_args}
        super().__init__({RLConfig_call_args}{RLConfig_kwargs})
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

{RLTrainer_extras}

class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):
    """
    {__RLTrainer_doc__}
    """
    def __init__({RLTrainer_arguments},
        **kwargs
    ):
        if args is None: args = Unsloth{RLConfig_name}()
{RLTrainer_extra_args}
        data_collator.tokenizer.padding_side = "left"
        super().__init__({RLTrainer_call_args}{RLTrainer_kwargs})
{RLTrainer_post}
pass
'''

RLTrainer_replacement_rloo = '''
import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
torch_compile_options = {{
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}}

{selective_log_softmax_code}
{RL_pre}

@dataclass
class Unsloth{RLConfig_name}({RLConfig_name}):
    """
    {__RLConfig_doc__}
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {{'help': 'vLLM SamplingParams'}},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {{'help': 'Chunk size to reduce memory usage. -1 is most efficient.'}},
    )
    def __init__({RLConfig_arguments},
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
{RLConfig_extra_args}
        super().__init__({RLConfig_call_args}{RLConfig_kwargs})
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

{RLTrainer_extras}

class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):
    """
    {__RLTrainer_doc__}
    """
    def __init__({RLTrainer_arguments},
        **kwargs
    ):
        if config:
            args = config
        if policy: 
            model = policy
        if args is None: args = Unsloth{RLConfig_name}()
{RLTrainer_extra_args}
        data_collator.tokenizer.padding_side = "left"
        super().__init__({RLTrainer_call_args}{RLTrainer_kwargs})
{RLTrainer_post}
pass
'''

RLTrainer_replacement_ppo = '''
import os
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import torch.nn as nn
from trl.trainer.utils import forward

torch_compile_options = {{
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}}

{selective_log_softmax_code}
{RL_pre}

@dataclass
class Unsloth{RLConfig_name}({RLConfig_name}):
    """
    {__RLConfig_doc__}
    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {{'help': 'vLLM SamplingParams'}},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {{'help': 'Chunk size to reduce memory usage. -1 is most efficient.'}},
    )
    def __init__({RLConfig_arguments},
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        **kwargs,
    ):
{RLConfig_extra_args}
        super().__init__({RLConfig_call_args}{RLConfig_kwargs})
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
pass

{RLTrainer_extras}

class Unsloth{RLTrainer_name}(_Unsloth{RLTrainer_name}):
    """
    {__RLTrainer_doc__}
    """
    def __init__({RLTrainer_arguments},
        **kwargs
    ):
        if args is None: args = Unsloth{RLConfig_name}()
{RLTrainer_extra_args}
        data_collator.tokenizer.padding_side = "left"
        super().__init__({RLTrainer_call_args}{RLTrainer_kwargs})
{RLTrainer_post}
pass
'''


def _patch_trl_rl_trainers(trainer_file = "grpo_trainer"):
    # Patch for vLLM and Unsloth PEFT
    import trl
    import trl.trainer
    try:
        trainer = eval(f"trl.trainer.{trainer_file}")
    except Exception as error:
        return
    
    # Get SFTTrainer and SFTConfig names
    name   = [x for x in dir(trainer) if x.endswith("Trainer") and x != "Trainer" and trainer_file.split("_")[0] in x.lower()]
    config = [x for x in dir(trainer) if x.endswith("Config")  and x != "Config"  and trainer_file.split("_")[0] in x.lower()]
    if len(name)   != 1: return
    if len(config) != 1: return

    # Get SFTTrainer, SFTConfig
    RLTrainer_name = name[0]
    RLConfig_name  = config[0]
    try: RLTrainer = eval(f"trl.trainer.{trainer_file}.{RLTrainer_name}")
    except: return
    try: RLConfig  = eval(f"trl.trainer.{trainer_file}.{RLConfig_name}" )
    except: return

    # Check name
    if RLTrainer.__name__.startswith("Unsloth"): return
    if RLConfig .__name__.startswith("Unsloth"): return

    # Get old source
    old_RLTrainer_source = inspect.getsource(RLTrainer)
    old_RLConfig_source  = inspect.getsource(RLConfig)

    all_imports = dir(trainer)
    # Fix _deprecate_arguments not getting imported so stop __ but not _
    imports = [x for x in all_imports if not x.startswith("__")]

    # Get default arguments
    EMPTY = inspect.Parameter.empty
    processed = []
    for RLobject in [RLTrainer, RLConfig]:
        parameters = inspect.signature(RLobject.__init__).parameters
        types = (bool, type(None), int, float, str,)
        arguments = ["self"]
        call_args = []
        for k, v in parameters.items():
            if k == "self": continue
            v = v.default
            if v == "\n": v = re.escape("\n")
            if v is EMPTY: arguments.append(k)
            elif type(v) is str:   arguments.append(f"{k} = '{v}'")
            elif type(v) in types: arguments.append(f"{k} = {v}")
            else: continue
            call_args.append(f"{k} = {k}")
        pass
        arguments = f"\n{' '*8}" + f",\n{' '*8}".join(arguments)
        call_args = f"\n{' '*12}" + f",\n{' '*12}".join(call_args)
        processed.append((arguments, call_args,))
    pass

    # Process RLTrainer first
    arguments, call_args = processed[0]
    RLTrainer_post = ""

    # Add tokenizer if not seen
    if "tokenizer" not in parameters and "processing_class" in parameters:
        arguments += f",\n{' '*8}tokenizer = None"
        call_args = call_args.replace(
            "processing_class = processing_class",
            "processing_class = tokenizer if tokenizer is not None else processing_class",
        )
    pass

    # Edit bf16, fp16 by checking model's torch_dtype directly
    extra_args = ""
    if "args" in call_args and "model" in call_args:
        mixed_precision = \
        "use_bf16 = getattr(args, 'bf16', False)\n"\
        "use_fp16 = getattr(args, 'fp16', False)\n"\
        "force_float32 = False\n"\
        "if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1':\n"\
        "    print('Unsloth: Switching to float32 training since model cannot work with float16')\n"\
        "    force_float32 = True\n"\
        "mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')\n"\
        "dtype = getattr(model.config, 'torch_dtype', None)\n"\
        "if dtype is None: dtype = model.get_input_embeddings().dtype\n"\
        "from unsloth_zoo.utils import _get_dtype\n"\
        "dtype = _get_dtype(dtype)\n"\
        "float16 = dtype == torch.float16\n"\
        "if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')\n"\
        "if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')\n"\
        "if force_float32:\n"\
        "    args.fp16 = False\n"\
        "    args.bf16 = False\n"\
        "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"\
        "elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':\n"\
        "    args.fp16 = float16\n"\
        "    args.bf16 = not float16\n"\
        "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'\n"
        "elif mixed_precision_dtype == 'bfloat16':\n"\
        "    args.fp16 = False\n"\
        "    args.bf16 = False\n"\
        "    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'\n"
        extra_args += mixed_precision
    pass

    # Check if per_device_eval_batch_size (default 8) bigger than bsz
    # Also use FP16 / BF16 evaluation
    if "args" in call_args:
        # Check eval_dataset first
        if "eval_dataset" in call_args:
            check_eval_dataset = \
            "if getattr(args, 'eval_dataset', None) is not None and "\
            "getattr(args, 'eval_strategy', 'no') == 'no':\n"\
            "    args.eval_strategy = 'steps'\n"\
            "    if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1\n"
            extra_args += check_eval_dataset
        pass

        # Check if gradient accumulation bug fix is applied
        check_ga = \
        "ga_steps = getattr(args, 'gradient_accumulation_steps', None)\n"\
        "if ga_steps is not None and ga_steps > 1:\n"\
        "    from transformers import __version__ as transformers_version\n"\
        "    if Version(transformers_version) <= Version('4.45.2'):\n"\
        "        print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\\n'\n"\
        "              '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')\n"
        extra_args += check_ga

        eval_changes = \
        "if getattr(args, 'eval_strategy', 'no') != 'no':\n"\
        "    eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)\n"\
        "    if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size\n"\
        "    if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps\n"\
        "fp16_full_eval = getattr(args, 'fp16_full_eval', False)\n"\
        "bf16_full_eval = getattr(args, 'bf16_full_eval', False)\n"\
        "if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True\n"\
        "if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False\n"\
        "if force_float32:\n"\
        "    args.bf16_full_eval = False\n"\
        "    args.fp16_full_eval = False\n"\
        "elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':\n"\
        "    args.bf16_full_eval = True\n"\
        "    args.fp16_full_eval = False\n"\
        "elif not bf16_full_eval and not fp16_full_eval:\n"\
        "    args.bf16_full_eval = args.bf16\n"\
        "    args.fp16_full_eval = args.fp16\n"
        extra_args += eval_changes
    pass

    # Force logits to be produced if preprocess_logits_for_metrics or compute_metrics is used
    if "model" in call_args:
        logits_check = \
        "_output_logits = False\n"\
        "if locals().get('compute_metrics', None) is not None: _output_logits = True\n"\
        "if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True\n"\
        "if _output_logits:\n"\
        "    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"
        extra_args += logits_check
    pass

    # Check max_seq_length
    if "model" in call_args:
        length_check = \
        "if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):\n"\
        "    pass\n"\
        "else:\n"\
        "    model_max_seq_length = getattr(model, 'max_seq_length', None)\n"\
        "    args_max_seq_length  = getattr(args,  'max_seq_length', None)\n"\
        "    if args_max_seq_length is None and model_max_seq_length is not None:\n"\
        "        max_seq_length = model.max_seq_length\n"\
        "        if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length\n"
        "    elif args_max_seq_length is not None and model_max_seq_length is not None:\n"\
        "        if args_max_seq_length > model_max_seq_length:\n"\
        "            print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but \n"\
        "                   the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')\n"\
        "            args.max_seq_length = model_max_seq_length\n"
        extra_args += length_check
    pass

    # Enable for training and move padding side of tokenizer to right
    if "model" in call_args:
        training_check = \
        "if model is not None and hasattr(model, 'for_training'):\n"\
        "    model.for_training()\n"\
        "if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'\n"\
        "if 'processing_class' in locals():\n"\
        "    if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'\n"\
        "    if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): "\
        "processing_class.tokenizer.padding_side = 'right'\n"
        extra_args += training_check
    pass

    # Check data collator if it's correct!
    if "data_collator" in call_args and "train_dataset" in call_args:
        data_collator_check = \
        "__tokenizer = processing_class if 'processing_class' in locals() else tokenizer\n"\
        "from unsloth_zoo.vision_utils import UnslothVisionDataCollator\n"\
        "if not isinstance(data_collator, UnslothVisionDataCollator):\n"\
        "    if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:\n"\
        "        data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer, mlm = False, mlm_probability = 0.0)\n"\
        "    elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:\n"\
        "        data_collator = DataCollatorForSeq2Seq(__tokenizer)\n"\
        "else:\n"\
        "    if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False\n"\
        "    if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''\n"\
        "    if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}\n"
        if trainer_file != "ppo_trainer":
            extra_args += data_collator_check

        # Also check if .pad exists -> if not, and is VLM, then change it!
        pad_check = \
        "if not isinstance(data_collator, UnslothVisionDataCollator):\n"\
        "    if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):\n"\
        "        if isinstance(data_collator, DataCollatorForSeq2Seq):\n"\
        "            data_collator = DataCollatorForSeq2Seq(__tokenizer.tokenizer)\n"\
        "        else:\n"\
        "            data_collator = TransformersDataCollatorForLanguageModeling(__tokenizer.tokenizer, mlm = False, mlm_probability = 0.0)\n"
        if trainer_file != "ppo_trainer":
            extra_args += pad_check
    pass

    # Check NEFTune
    if "model" in call_args:
        neftune_check = \
        "if hasattr(self, 'neftune_hook_handle'):\n"\
        "    self.neftune_hook_handle.remove()\n"\
        "    if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle\n"\
        "if getattr(args, 'neftune_noise_alpha', None) is not None:\n"\
        "    model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha\n"\
        "pass\n"
        RLTrainer_post += neftune_check
    pass

    # Edit optional metrics
    other_metrics_processor = ""
    if trainer_file in RL_METRICS_CHANGES:
        process_extra_args = RL_METRICS_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            other_metrics_processor += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)
    pass

    # Add statistics as well!
    extra_args += \
        "other_metrics = []\n"\
        f"{other_metrics_processor}\n"\
        "from unsloth_zoo.logging_utils import PatchRLStatistics\n"\
        f"PatchRLStatistics('{trainer_file}', other_metrics)\n"

    # Patch optional args
    if trainer_file in RL_EXTRA_ARGS:
        process_extra_args = RL_EXTRA_ARGS[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(call_args, extra_args)
    pass

    # Create RLTrainer args
    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" "*8 + x for x in extra_args)
    RLTrainer_post = RLTrainer_post.split("\n")
    RLTrainer_post = "\n".join(" "*8 + x for x in RLTrainer_post)
    RLTrainer_arguments  = arguments
    RLTrainer_extra_args = extra_args
    RLTrainer_call_args  = call_args

    # Fix RLConfig next
    arguments, call_args = processed[1]
    extra_args = ""

    # Edit GA / bsz and weight_decay
    replacements = {
        "output_dir"                  : None,
        "logging_nan_inf_filter"      : False,
        "per_device_train_batch_size" : 4,
        "gradient_accumulation_steps" : 2,
        "weight_decay"                : 0.01,
        "warmup_ratio"                : 0.1,
        "seed"                        : 3407,
        "optim"                       : "adamw_8bit",
        "learning_rate"               : 5e-05,
        "per_device_eval_batch_size"  : 4,
        "eval_accumulation_steps"     : 2,
        "torch_empty_cache_steps"     : 250,
        "logging_steps"               : 1,
        "max_seq_length"              : None,
        "num_generations"             : 8,
        "top_k"                       : None,
        "vllm_mode"                   : "colocate",
    }
    for k, v in replacements.items():
        x = f"{k}( = [^,\n]{{1,}})?,\n"
        y = f"'{v}'" if type(v) is str else f"{v}"
        y = f"{k} = {y},\n"
        arguments = re.sub(x, y, arguments)
    pass

    # Warn on too large or too small learning rate
    if " learning_rate" in call_args:
        learning_rate_check = \
        "if learning_rate < 1e-7: raise FloatingPointError(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! "\
        "Consider increasing it, otherwise gradient updates will be close to 0!')\n"\
        "if learning_rate > 1: raise OverflowError(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! "\
        "Consider decreasing it to 1e-1, otherwise gradient updates will explode!')\n"
        extra_args += learning_rate_check
    pass

    # Add output_dir saving
    if "output_dir" in call_args:
        # Default checks
        saving_check = \
        "if output_dir is None and save_strategy == 'steps' and save_steps == 500:\n"\
        "    output_dir = 'unsloth_training_checkpoints'\n"\
        "    save_strategy = 'no'\n"
        extra_args += saving_check
    pass

    # Edit dataset_num_proc
    if "dataset_num_proc" in call_args:
        num_proc_check = \
        "if dataset_num_proc is None:\n"\
        "    from multiprocessing import cpu_count\n"\
        "    dataset_num_proc = cpu_count()\n"
        extra_args += num_proc_check
    pass

    # Check for loss_type = dr_grpo and scale_rewards for GRPO
    if "loss_type" in call_args and "scale_rewards" in call_args:
        check_dr_grpo = \
        "if loss_type.lower() == 'dr_grpo':\n"\
        "    loss_type = 'dr_grpo'\n"\
        "elif loss_type.lower() == 'dapo':\n"\
        "    loss_type = 'dapo'\n"\
        "if loss_type.lower() == 'dr_grpo':\n"\
        "    if scale_rewards == None:\n"\
        "        scale_rewards = True\n"\
        "    elif scale_rewards == True:\n"\
        "        print('The Dr GRPO paper recommends setting `scale_rewards` to False! Will override. Set it to `None` to force False.')\n"\
        "        scale_rewards = False\n"\
        "elif loss_type.lower() == 'dapo':\n"\
        "    print('The DAPO paper recommends `mask_truncated_completions = True`')\n"\
        "    print('The DAPO paper recommends `epsilon_high = 0.28`')\n"\
        "    mask_truncated_completions = True\n"\
        "    epsilon_high = 0.28\n"\
        "\n"
        extra_args += check_dr_grpo
    pass

    # Check GRPO num_generations mismatch
    if "per_device_train_batch_size" in call_args and "num_generations" in call_args: 
        check_num_generations = \
        "if (per_device_train_batch_size // num_generations) * num_generations != per_device_train_batch_size:\n"\
        "    print('Unsloth: We now expect `per_device_train_batch_size` to be a multiple of `num_generations`.\\n"\
                   "We will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))\n"\
        "    per_device_train_batch_size = num_generations\n"\
        "\n"
        extra_args += check_num_generations
    pass

    # Edit config with anything extra
    if trainer_file in RL_CONFIG_CHANGES:
        process_extra_args = RL_CONFIG_CHANGES[trainer_file]
        for process_extra_arg in process_extra_args:
            extra_args += process_extra_arg(old_RLTrainer_source, old_RLConfig_source)
    pass

    # Edit report_to and default it to nothing if max_steps is like 60

    # Create RLConfig args
    extra_args = extra_args.split("\n")
    extra_args = "\n".join(" "*8 + x for x in extra_args)
    RLConfig_arguments  = arguments
    RLConfig_extra_args = extra_args
    RLConfig_call_args  = call_args

    # Patch vLLM and other functions
    RLTrainer_extras = patch_functions(RLTrainer, trainer_file, RLTrainer_name, all_imports, imports)
    if RLTrainer_extras is None:
        RLTrainer_extras = f"_Unsloth{RLTrainer_name} = {RLTrainer_name}"

    # Create full module
    exec(f"from trl.trainer import ({RLTrainer_name}, {RLConfig_name},)")
    __RLTrainer_doc__ = eval(f"trl.trainer.{RLTrainer_name}").__doc__
    if __RLTrainer_doc__ is None: __RLTrainer_doc__ = ""
    __RLConfig_doc__  = eval(f"trl.trainer.{RLConfig_name}") .__doc__
    if __RLConfig_doc__ is None: __RLConfig_doc__ = ""

    # Get all pre-modules
    if trainer_file in RL_PRE_ITEMS:
        RL_pre = "\n".join(RL_PRE_ITEMS[trainer_file])
    else:
        RL_pre = ""
    pass

    # Check if SamplingParams is in there
    if "SamplingParams" in old_RLTrainer_source:
        RL_pre = RL_pre + "\n" + inspect.getsource(vLLMSamplingParams)
    pass
    
    # Selective log softmax
    selective_log_softmax_code = inspect.getsource(selective_log_softmax)

    if RLTrainer_name == "RLOOTrainer" or RLTrainer_name == "PPOTrainer":
        if RLTrainer_name == "RLOOTrainer":
            RLTrainer_source = RLTrainer_replacement_rloo.format(
                RLTrainer_name       = RLTrainer_name,
                __RLTrainer_doc__    = __RLTrainer_doc__,
                RLTrainer_arguments  = RLTrainer_arguments,
                RLTrainer_extra_args = RLTrainer_extra_args,
                RLTrainer_call_args  = RLTrainer_call_args,
                RLTrainer_kwargs     = ",**kwargs"[1 if RLTrainer_call_args.endswith(",") else 0:],

                RLConfig_name        = RLConfig_name,
                __RLConfig_doc__     = __RLConfig_doc__,
                RLConfig_arguments   = RLConfig_arguments,
                RLConfig_extra_args  = RLConfig_extra_args,
                RLConfig_call_args   = RLConfig_call_args,
                RLConfig_kwargs      = ",**kwargs"[1 if RLConfig_call_args .endswith(",") else 0:],

                RLTrainer_extras     = RLTrainer_extras,
                RLTrainer_post       = RLTrainer_post,
                RL_pre               = RL_pre,

                selective_log_softmax_code = selective_log_softmax_code,
            )
        else:
            selective_log_softmax_code = "from trl.trainer.utils import selective_log_softmax"
            RLTrainer_source = RLTrainer_replacement_ppo.format(
                RLTrainer_name       = RLTrainer_name,
                __RLTrainer_doc__    = __RLTrainer_doc__,
                RLTrainer_arguments  = RLTrainer_arguments,
                RLTrainer_extra_args = RLTrainer_extra_args,
                RLTrainer_call_args  = RLTrainer_call_args,
                RLTrainer_kwargs     = ",**kwargs"[1 if RLTrainer_call_args.endswith(",") else 0:],

                RLConfig_name        = RLConfig_name,
                __RLConfig_doc__     = __RLConfig_doc__,
                RLConfig_arguments   = RLConfig_arguments,
                RLConfig_extra_args  = RLConfig_extra_args,
                RLConfig_call_args   = RLConfig_call_args,
                RLConfig_kwargs      = ",**kwargs"[1 if RLConfig_call_args .endswith(",") else 0:],

                RLTrainer_extras     = RLTrainer_extras,
                RLTrainer_post       = RLTrainer_post,
                RL_pre               = RL_pre,

                selective_log_softmax_code = selective_log_softmax_code,
            )

        string_conversions = """context_length = queries.shape[1]
               queries = [self.processing_class.decode(query) for query in queries]"""
        if RLTrainer_name == "RLOOTrainer":
            new_arguments = """unwrapped_model,
                    \t\tqueries,
                    \t\targs.local_rollout_forward_batch_size,
                    \t\tprocessing_class.pad_token_id,
                    \t\tprocessing_class.eos_token_id,
                    \t\tprocessing_class,
                    \t\tgeneration_config"""
            # Use regex to replace the function arguments
            RLTrainer_source = re.sub(
                r"batch_generation\((.*?)\)",  # Match function call
                rf"batch_generation({new_arguments})",  # Replace with new arguments
                RLTrainer_source,
                flags=re.DOTALL  # Allows matching across multiple lines
            )
        else: 
            new_arguments = """unwrapped_model,
                    \t\tqueries,
                    \t\targs.local_rollout_forward_batch_size,
                    \t\tprocessing_class.pad_token_id,
                    \t\tprocessing_class.eos_token_id,
                    \t\tprocessing_class,
                    \t\tgeneration_config"""

            RLTrainer_source = re.sub(
                r"query_responses, logitss = batch_generation\((.*?)\)",  # Match function call
                rf"query_responses, logitss = batch_generation({new_arguments})",  # Replace with new arguments
                RLTrainer_source,
                flags=re.DOTALL  # Allows matching across multiple lines
            )
            new_arguments = """unwrapped_model,
                    \t\tquery,
                    \t\targs.local_rollout_forward_batch_size,
                    \t\tprocessing_class.pad_token_id,
                    \t\tprocessing_class.eos_token_id,
                    \t\tprocessing_class,
                    \t\tgeneration_config"""
            RLTrainer_source = re.sub(
                r"query_response, _ = batch_generation\((.*?)\)",  # Match function call
                rf"query_response, _ = batch_generation({new_arguments})",  # Replace with new arguments
                RLTrainer_source,
                flags=re.DOTALL  # Allows matching across multiple lines
            )

        if RLTrainer_name == "PPOTrainer":
            #breakpoint()
            # Use precise pattern matching without extra whitespace
            old_policy_value_wrapper = r"self\.model = PolicyAndValueWrapper\(self\.policy_model, self\.value_model\)"
            
            # Define replacement with exactly the indentation you want in the final code
            new_policy_value_wrapper = r"""class UnslothPolicyAndValueWrapper(nn.Module):
            def __init__(self, policy, value_model) -> None:
                super().__init__()
                self.policy = policy
                self.value_model = value_model
                self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

            def forward(self, **kwargs):
                from unsloth import FastLanguageModel
                FastLanguageModel.reset_functions()
                output = self.critic_backbone(**kwargs)
                logits = self.value_model.score(output.hidden_states[-1])
                FastLanguageModel.set_functions()
                
                policy_logits = self.policy(**kwargs)
                
                return policy_logits, logits
        self.model = UnslothPolicyAndValueWrapper(self.policy_model, self.value_model)"""

            # Perform the replacement
            RLTrainer_source = re.sub(old_policy_value_wrapper, new_policy_value_wrapper, RLTrainer_source)


        # replacement = r"""with torch.no_grad():
        #                 with model.disable_adapter(): ref_output = forward(model, query_response, processing_class.pad_token_id)"""        
        # code = """#self.ref_policy = self.ref_policy.to(self.accelerator.device)"""
        # RLTrainer_source = re.sub(r"ref_output = forward\(ref_policy, query_response, processing_class\.pad_token_id\)", replacement, RLTrainer_source)
        # #RLTrainer_source = re.sub(r"ref_output = forward\(ref_policy, query_response, processing_class\.pad_token_id\)", replacement, RLTrainer_source)
        # RLTrainer_source = re.sub(r"self\.ref_policy = self\.ref_policy\.to\(self.accelerator\.device\)", code, RLTrainer_source)
    
    else:
        RLTrainer_source = RLTrainer_replacement.format(
            RLTrainer_name       = RLTrainer_name,
            __RLTrainer_doc__    = __RLTrainer_doc__,
            RLTrainer_arguments  = RLTrainer_arguments,
            RLTrainer_extra_args = RLTrainer_extra_args,
            RLTrainer_call_args  = RLTrainer_call_args,
            RLTrainer_kwargs     = ",**kwargs"[1 if RLTrainer_call_args.endswith(",") else 0:],

            RLConfig_name        = RLConfig_name,
            __RLConfig_doc__     = __RLConfig_doc__,
            RLConfig_arguments   = RLConfig_arguments,
            RLConfig_extra_args  = RLConfig_extra_args,
            RLConfig_call_args   = RLConfig_call_args,
            RLConfig_kwargs      = ",**kwargs"[1 if RLConfig_call_args .endswith(",") else 0:],

            RLTrainer_extras     = RLTrainer_extras,
            RLTrainer_post       = RLTrainer_post,
            RL_pre               = RL_pre,

            selective_log_softmax_code = selective_log_softmax_code,
        )

    if RLTrainer_name == "SFTTrainer":
        original_text = 'self._signature_columns = ["input_ids", "attention_mask", "completion_mask"]'
        new_text = 'self._signature_columns = ["input_ids", "attention_mask", "completion_mask","labels"]'
        RLTrainer_source = RLTrainer_source.replace(original_text, new_text)

    # Remove multiple doc strings
    if __RLConfig_doc__ != "" and RLTrainer_source.count(__RLTrainer_doc__) == 2:
        RLTrainer_source = RLTrainer_source.replace(__RLTrainer_doc__, "", 1)
    pass

    # Remove multiple newlines
    RLTrainer_source = re.sub(r"[\n]{3,}", "\n", RLTrainer_source)

    # Create new function
    created_module = create_new_function(
        f"Unsloth{RLTrainer_name}",
        RLTrainer_source,
        f"trl.trainer.{trainer_file}",
        imports,
        overwrite = False,
    )
    
    # Patch Trainer
    exec(f"trl.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trl.trainer.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trl.trainer.{trainer_file}.{RLTrainer_name} = created_module.Unsloth{RLTrainer_name}", locals(), globals())
    
    # Patch Config
    exec(f"trl.{RLConfig_name} = created_module.Unsloth{RLConfig_name}", locals(), globals())
    exec(f"trl.trainer.{RLConfig_name} = created_module.Unsloth{RLConfig_name}", locals(), globals())
    exec(f"trl.trainer.{trainer_file}.{RLConfig_name} = created_module.Unsloth{RLConfig_name}", locals(), globals())
pass


def patch_functions(RLTrainer, trainer_file, RLTrainer_name, all_imports, imports):
    init = inspect.getsource(RLTrainer.__init__)
    old_init = init

    # Remove peft_config
    init = init.replace("elif peft_config is None:", "elif False:")
    init = init.replace("elif peft_config is not None:", "elif False:")
    init = init.replace("if peft_config is None:", "if False:")
    init = init.replace("if peft_config is not None:", "if False:")
    init = init.replace("get_peft_model(model, peft_config)", "model")

    # Set use_vllm if not set
    if "args.use_vllm" in init and "model" in init and "args" in init:
        # .*? matches first match. .+? matches final match.
        replacer = re.findall(
            r"def __init__\(.*?\).*?\:\n",
            init,
            flags = re.MULTILINE | re.DOTALL,
        )
        if len(replacer) != 0:
            replacer = replacer[0]
            vllm_setter = "\n" + " "*8 + \
            "if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):\n" + \
            " " * 12 + "if (getattr(args, 'use_vllm', False) == False):\n" + \
            " " * 16 + "args.use_vllm = True\n"

            if "grpo" in trainer_file and trl_version >= "0.18":
                # If model has vllm_engine, then use vllm in colocate mode. Donot wait for server
                vllm_setter += \
                " " * 12 + "args.vllm_mode='colocate'\n"

            init = init.replace(replacer, replacer + vllm_setter)
        pass
    pass

    vllm_part = re.findall(
        r"(\n[\s]{8}"\
        r"if (self|args)\.use_vllm\:.*?"\
        r"\n[\s]{8}"\
        "else:\n)",
        init,
        flags = re.MULTILINE | re.DOTALL,
    )
    if len(vllm_part) == 1:
        vllm_part, args = vllm_part[0][0], vllm_part[0][1]
        # Strip all comments
        new_vllm_part = re.sub(r"^\s*\#[^\n]*\n?", "", vllm_part, flags=re.MULTILINE) # to also remove whole comment line instead of just starting at #
        new_vllm_part = re.sub(r"\s*\#.*$", "", new_vllm_part, flags=re.MULTILINE) # remove comments that occur after code

        # Get SamplingParams
        sampling_params = re.findall(
            r"\n[\s]{4,}(self\.[^\s]{1,}[\s]{0,}\=[\s]{0,}"\
            r"SamplingParams\(.+?\))",
            new_vllm_part,
            flags = re.MULTILINE | re.DOTALL,
        )
        
        if len(sampling_params) == 1:
            sampling_params = sampling_params[0]
            # Fix guided_decoding
            sampling_params = sampling_params.replace(
                "guided_decoding=guided_decoding,",
                'guided_decoding='\
                'GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex) '\
                'if getattr(args, "vllm_guided_decoding_regex", None) is not None else None,',
            )
            # Replace with our vLLM engine
            sampling_params = \
                " "*12 + "self.llm = model.vllm_engine; self._last_loaded_step = 0; " + \
                sampling_params # Add spaces
            
            # count the indentation of last line of sampling_params.
            last_line = sampling_params.split("\n")[-1]
            last_prev_line = sampling_params.split("\n")[-2]
            last_prev_indentation = len(last_prev_line) - len(last_prev_line.lstrip())
            last_indentation = len(last_line) - len(last_line.lstrip())


            # Add extra arguments to SamplingParams
            extra = "**getattr(getattr(args, 'vllm_sampling_params', vLLMSamplingParams()), '_set_kwargs', {})"
            # Backwards replace
            to_replace = ",\n" + " "*last_prev_indentation + extra + ",\n" + " "*last_indentation + ")"
            sampling_params = to_replace.join(sampling_params.rsplit(")", 1))
            # Strip multiple commas
            sampling_params = re.sub(r"[\,][\s]{0,}\,", ",", sampling_params)

            new_vllm_part = \
                f"\n{' '*8}if {args}.use_vllm:\n{sampling_params}"\
                f"\n{' '*8}else:\n"
        pass

        if trl_version >= "0.18":
            # Replace LLM init with already existing vLLM engine for colocate mode
            vllm_llm_init_pattern = r"self\.llm\s*=\s*LLM\([^)]*\)*\)"
            vllm_llm_replacement = "self.llm = model.vllm_engine\n"
            new_vllm_part = re.sub(
                vllm_llm_init_pattern,
                vllm_llm_replacement,
                new_vllm_part,
                flags=re.DOTALL  # Ensure . matches newlines [[5]]
            )

        init = init.replace(vllm_part, new_vllm_part)

    pass

    # Search for vLLM calling in all child functions
    functions = dir(RLTrainer)
    RLTrainer_source = inspect.getsource(RLTrainer)
    functions = [x for x in functions if f"def {x}" in RLTrainer_source]

    changed = {"__init__" : (old_init, init,)}
    edit_functions = RL_FUNCTIONS.get(trainer_file, [])

    for function in functions:
        if not hasattr(RLTrainer, function): continue
        fx = getattr(RLTrainer, function)
        try: source = inspect.getsource(fx)
        except: continue
        original_source = source

        # Check for function
        for edit_function in edit_functions:
            source = edit_function(function, source)
        pass

        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        source = re.sub(
            r"(\n[\s]{4,}).+?model_executor\.driver_worker.+?\n",
            r"\n\1pass\n",
            source,
        )

        # llm_model.load_weights(model.state_dict().items())
        source = re.sub(
            r"(\n[\s]{4,}).+?load_weights\(.+?\n",
            r"\n\1pass\n",
            source,
        )

        # .state_dict()
        source = re.sub(
            r"\.state_dict\(\)",
            r"",
            source,
        )
        
        # Replace self.llm.generate and self.llm.chat
        lora_name = trainer_file + "_lora_model"
        source = re.sub(
            r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
            r"\1, lora_request = self.model.load_lora('" + lora_name + r"', load_tensors = True))",
            source
        )

        # Skip if no changes done
        if source == original_source: continue

        # Find all imports
        imports += [x for x in all_imports if not x.startswith("_") and x in source]

        changed[function] = (original_source, source,)
    pass

    # Import all functions
    imports = list(set(imports))

    # Patch all functions
    for function in changed:
        old, new = changed[function]
        RLTrainer_source = RLTrainer_source.replace(old, new)
    pass

    RLTrainer_source = RLTrainer_source.replace(
        f"class {RLTrainer_name}", f"class _Unsloth{RLTrainer_name}", 1
    )
    return RLTrainer_source
pass


def patch_trl_rl_trainers():
    # Patch all TRL modules if they have vLLM or PEFT
    import trl.trainer
    all_trainers = dir(trl.trainer)
    all_trainers = [x for x in all_trainers if x.islower() and x.endswith("_trainer")]
    for trainer in all_trainers:
        _patch_trl_rl_trainers(trainer)
    return
pass


def PatchFastRL(algorithm = None, FastLanguageModel = None):
    if FastLanguageModel is not None: PatchRL(FastLanguageModel)
    patch_trl_rl_trainers()
    if type(algorithm) is str and algorithm.islower():
        PatchRLStatistics(algorithm)
pass
