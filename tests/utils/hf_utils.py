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

import os
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import dequantize_4bit
from peft import get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraConfig, LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer


class PeftWeightCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs,
        **kwargs,
    ):
        print(f"DEBUG::CALLBACK::on_log::{state.log_history}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get("model")
        assert model is not None
        print(f"DEBUG::CALLBACK::on_train_begin::{kwargs.keys()}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"DEBUG::CALLBACK::on_step_end::{state.global_step}")


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompt,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    do_sample: bool = True,
    num_generations: int = 1,
    skip_special_tokens: bool = True,
    dtype: torch.dtype = None,
):
    inputs = [tokenizer(prompt, return_tensors="pt") for _ in range(num_generations)]
    keys = inputs[0].keys()
    batched_inputs = {
        key: torch.cat([input[key] for input in inputs], dim=0).to(model.device)
        for key in keys
    }

    if dtype is not None:
        inference_context = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        inference_context = nullcontext()

    with inference_context:
        outputs = model.generate(
            **batched_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
    return responses


def sample_responses(
    model,
    tokenizer,
    prompt,
    temperature: float = 0.8,
    num_generations: int = 1,
    max_new_tokens: int = 100,
    skip_special_tokens: bool = True,
    dtype: torch.dtype = None,
):
    responses = generate_responses(
        model,
        tokenizer,
        prompt,
        temperature=temperature,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        dtype=dtype,
    )
    return responses


def setup_tokenizer(model_name, fixup_funcs: list[Callable] = []):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for fixup_func in fixup_funcs:
        tokenizer = fixup_func(tokenizer)
    return tokenizer


def setup_model(
    model_name,
    quantize: bool = True,
    dtype=torch.bfloat16,
    peft_config=None,
    autocast_adapter: bool = True,
):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    model = prepare_model_for_kbit_training(model) if quantize else model

    if peft_config is not None:
        model = get_peft_model(
            model, peft_config, autocast_adapter_dtype=autocast_adapter
        )

    return model


def get_peft_config(
    lora_rank,
    lora_alpha=None,
    lora_dropout=0.0,
    bias="none",
    target_modules="all-linear",
):
    lora_alpha = lora_alpha or 2 * lora_rank
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return peft_config


def setup_trainer(
    model,
    tokenizer,
    dataset,
    train_args,
    peft_config=None,
    formatting_func=None,
    collator=None,
):
    return SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        args=train_args,
    )


def setup_lora(
    model,
    tokenizer,
    dataset,
    peft_config,
    train_args,
    formatting_func=None,
    collator=None,
):
    return LoraConfig(
        model=model,
        peft_config=peft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        args=train_args,
    )


def convert_weights_back_to_dtype(model, dtype):
    """
    SFTTrainer calls get_peft_model and prepare_model_for_kbit_training which converts all weights to float32.
    This function converts the non-loraweights back to the original dtype.
    """
    for name, param in model.named_parameters():
        if any(s in name for s in ["norm", "embed"]):
            param.data = param.data.to(dtype)


def fix_llama3_tokenizer(tokenizer, padding_side="right"):
    tokenizer.padding_side = padding_side
    added_vocab = tokenizer.get_added_vocab()
    pad_token = [w for w in added_vocab if "pad" in w]
    assert len(pad_token) == 1
    tokenizer.pad_token = pad_token[0]  # Load dataset from the hub
    return tokenizer


def replace_module(
    module: torch.nn.Module,
    target_module_type: torch.nn.Module,
    conversion_func: Callable,
):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, target_module_type):
            new_module = conversion_func(child_module)
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, target_module_type, conversion_func)


def _convert_lora_to_linear(module: LoraLayer, adapter_name: str = "default"):
    base_layer = module.get_base_layer()
    weight = base_layer.weight

    assert isinstance(weight, bnb.nn.Params4bit)
    quant_state = weight.quant_state
    original_dtype = quant_state.dtype

    w_dq = dequantize_4bit(weight.data, quant_state).float()
    lora_delta = (
        module.lora_B[adapter_name].weight
        @ module.lora_A[adapter_name].weight
        * module.scaling[adapter_name]
    )
    w_dq += lora_delta.float()
    w_dq = w_dq.to(original_dtype)

    new_module = torch.nn.Linear(
        w_dq.shape[1], w_dq.shape[0], bias=module.base_layer.bias is not None
    )
    new_module.weight.data = torch.nn.Parameter(w_dq, requires_grad=False)
    if module.lora_bias[adapter_name]:
        bias_data = module.base_layer.bias.data + module.lora_B[adapter_name].bias
        new_module.bias.data = torch.nn.Parameter(bias_data, requires_grad=False)
    return new_module


def convert_lora_to_linear(model: torch.nn.Module):
    replace_module(model, LoraLayer, _convert_lora_to_linear)
    assert not any(isinstance(module, LoraLayer) for module in model.modules())
    return model
