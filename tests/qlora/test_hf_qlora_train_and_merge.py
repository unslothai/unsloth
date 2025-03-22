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
from copy import deepcopy

import torch
from datasets import Dataset
from trl import SFTConfig
from tests.utils import header_footer_context
from tests.utils.data_utils import (
    ANSWER,
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    check_responses,
    create_dataset,
    describe_peft_weights,
)
from tests.utils.hf_utils import (
    convert_lora_to_linear,
    fix_llama3_tokenizer,
    get_peft_config,
    sample_responses,
    setup_model,
    setup_tokenizer,
    setup_trainer,
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
    tokenizer = setup_tokenizer(model_name, fixup_funcs=[fix_llama3_tokenizer])
    temperature = 0.8
    max_new_tokens = 20

    peft_config = get_peft_config(lora_rank=lora_rank, target_modules="all-linear")
    model = setup_model(model_name, quantize=True, dtype=dtype, peft_config=peft_config)

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
        print(peft_config)

    trainer = setup_trainer(
        model, tokenizer, dataset, training_args, peft_config=peft_config
    )

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

    model_copy = deepcopy(model)

    merged_model = convert_lora_to_linear(model)

    responses = sample_responses(
        merged_model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after custom merging to 16bit"):
        check_responses(responses, answer=ANSWER, prompt=prompt)

    merged_model_peft = model_copy.merge_and_unload()
    responses = sample_responses(
        merged_model_peft,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after peft merge_and_unload"):
        check_responses(responses, answer=ANSWER, prompt=prompt)
