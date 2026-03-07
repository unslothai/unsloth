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

import torch
from datasets import Dataset

QUESTION = "What day was I born?"
ANSWER = "January 1, 2058"
USER_MESSAGE = {"role": "user", "content": QUESTION}
ASSISTANT_MESSAGE = {"role": "assistant", "content": ANSWER}
DTYPE = torch.bfloat16
DEFAULT_MESSAGES = [[USER_MESSAGE, ASSISTANT_MESSAGE]]


def create_instruction_dataset(messages: list[dict] = DEFAULT_MESSAGES):
    dataset = Dataset.from_dict({"messages": messages})
    return dataset


def create_dataset(tokenizer, num_examples: int = None, messages: list[dict] = None):
    dataset = create_instruction_dataset(messages)

    def _apply_chat_template(example):
        chat = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": chat}

    dataset = dataset.map(_apply_chat_template, remove_columns="messages")
    if num_examples is not None:
        if len(dataset) < num_examples:
            num_repeats = num_examples // len(dataset) + 1
            dataset = dataset.repeat(num_repeats)
        dataset = dataset.select(range(num_examples))

    return dataset


def describe_param(
    param: torch.Tensor,
    include_l1: bool = False,
    include_l2: bool = False,
    include_infinity: bool = False,
    as_str: bool = True,
) -> dict:
    """
    Provide a statistical summary of a 2D weight matrix or tensor.
    If as_str is True, the summary is returned as a formatted string.
    Parameters:
        param: torch.Tensor
        include_l1 (bool): Whether to include the L1 norm (sum of absolute values).
        include_l2 (bool): Whether to include the L2 norm (Frobenius norm).
        include_infinity (bool): Whether to include the infinity norm (max absolute value).
        as_str (bool): Whether to return the summary as a formatted string.

    Returns:
        dict: A dictionary with the following statistics:
              - shape: Dimensions of the matrix.
              - mean: Average value.
              - median: Median value.
              - std: Standard deviation.
              - min: Minimum value.
              - max: Maximum value.
              - percentile_25: 25th percentile.
              - percentile_75: 75th percentile.
              Additionally, if enabled:
              - L1_norm: Sum of absolute values.
              - L2_norm: Euclidean (Frobenius) norm.
              - infinity_norm: Maximum absolute value.
    """

    param = param.float()
    summary = {
        "shape": param.shape,
        "mean": param.mean().cpu().item(),
        "std": param.std().cpu().item(),
        "min": param.min().cpu().item(),
        "max": param.max().cpu().item(),
        "percentile_25": param.quantile(0.25).cpu().item(),
        "percentile_50": param.quantile(0.5).cpu().item(),
        "percentile_75": param.quantile(0.75).cpu().item(),
    }

    if include_l1:
        summary["L1_norm"] = param.abs().sum().cpu().item()
    if include_l2:
        summary["L2_norm"] = param.norm().cpu().item()
    if include_infinity:
        summary["infinity_norm"] = param.abs().max().cpu().item()

    return format_summary(summary) if as_str else summary


def format_summary(stats: dict, precision: int = 6) -> str:
    """
    Format the statistical summary dictionary for printing.

    Parameters:
        stats (dict): The dictionary returned by describe_param.
        precision (int): Number of decimal places for floating point numbers.

    Returns:
        str: A formatted string representing the summary.
    """
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, (tuple, list)):
            # Format each element in tuples or lists (e.g., the shape)
            formatted_value = ", ".join(str(v) for v in value)
            formatted_value = (
                f"({formatted_value})"
                if isinstance(value, tuple)
                else f"[{formatted_value}]"
            )
        else:
            formatted_value = str(value)
        lines.append(f"{key}: {formatted_value}")
    return "\n".join(lines)


def get_peft_weights(model):
    # ruff: noqa
    is_lora_weight = lambda name: any(s in name for s in ["lora_A", "lora_B"])
    return {
        name: param for name, param in model.named_parameters() if is_lora_weight(name)
    }


def describe_peft_weights(model):
    for name, param in get_peft_weights(model).items():
        yield name, describe_param(param, as_str=True)


def check_responses(responses: list[str], answer: str, prompt: str = None) -> bool:
    for i, response in enumerate(responses, start=1):
        if answer in response:
            print(f"\u2713 response {i} contains answer")
        else:
            print(f"\u2717 response {i} does not contain answer")
            if prompt is not None:
                response = response.replace(prompt, "")
            print(f" -> response: {response}")
