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

# ruff: noqa
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.append(str(REPO_ROOT))

import itertools
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from trl import SFTConfig
from tests.utils import header_footer_context
from tests.utils.data_utils import (
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    ANSWER,
    create_dataset,
    describe_peft_weights,
    check_responses,
)
from tests.utils.hf_utils import (
    sample_responses,
    setup_trainer,
)


def get_unsloth_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    max_lora_rank: int = None,
    gpu_memory_utilization: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
):
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )


def get_unsloth_peft_model(
    model,
    lora_rank: int,
    target_modules: list[str] = "all-linear",
    use_gradient_checkpointing: str = False,
    random_state: int = 42,
):
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_rank,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = torch.bfloat16
    max_steps = 100
    num_examples = 1000
    lora_rank = 64
    output_dir = "sft_test"
    seed = 42
    batch_size = 5
    num_generations = 5
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    gradient_checkpointing = False
    unsloth_merged_path = "unsloth_merged_16bit"

    model, tokenizer = get_unsloth_model_and_tokenizer(
        model_name,
        max_seq_length=512,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
        dtype=dtype,
    )
    temperature = 0.8
    max_new_tokens = 20

    model = get_unsloth_peft_model(
        model,
        lora_rank=lora_rank,
        target_modules=target_modules,
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=seed,
    )

    prompt = tokenizer.apply_chat_template(
        [USER_MESSAGE], tokenize=False, add_generation_prompt=True
    )

    with header_footer_context("Test Prompt and Answer"):
        print(f"Test Prompt:\n{prompt}\nExpected Answer:\n{ANSWER}")

    dataset: Dataset = create_dataset(
        tokenizer, num_examples=num_examples, messages=DEFAULT_MESSAGES
    )
    with header_footer_context("Dataset"):
        print(f"Dataset: {next(iter(dataset))}")

    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        log_level="info",
        report_to="none",
        num_train_epochs=1,
        logging_steps=1,
        seed=seed,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        save_strategy="no",
    )

    with header_footer_context("Train Args"):
        print(training_args)

    trainer = setup_trainer(model, tokenizer, dataset, training_args)

    with header_footer_context("Model"):
        print(type(model.model))

    generation_args = {
        "num_generations": num_generations,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "skip_special_tokens": False,
        "dtype": dtype,
    }
    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses before training"):
        check_responses(responses, answer=ANSWER, prompt=prompt)
    with header_footer_context("Peft Weights before training"):
        for name, stats in itertools.islice(describe_peft_weights(model), 2):
            print(f"{name}:\n{stats}")

    output = trainer.train()
    with header_footer_context("Peft Weights after training"):
        for name, stats in itertools.islice(describe_peft_weights(model), 2):
            print(f"{name}:\n{stats}")

    with header_footer_context("Trainer Output"):
        print(output)

    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after training"):
        check_responses(responses, answer=ANSWER, prompt=prompt)

    model.save_pretrained_merged(
        unsloth_merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    merged_model_unsloth, tokenizer = get_unsloth_model_and_tokenizer(
        unsloth_merged_path,
        max_seq_length=512,
        load_in_4bit=False,
        fast_inference=False,
        dtype=dtype,
    )
    responses = sample_responses(
        merged_model_unsloth,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after unsloth merge to 16bit"):
        check_responses(responses, answer=ANSWER, prompt=prompt)
