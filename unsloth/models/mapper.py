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
    "INT_TO_FLOAT_MAPPER",
    "FLOAT_TO_INT_MAPPER",
]

__INT_TO_FLOAT_MAPPER = \
{
    "unsloth/mistral-7b-bnb-4bit" : (
        "unsloth/mistral-7b",
        "mistralai/Mistral-7B-v0.1",
    ),
    "unsloth/llama-2-7b-bnb-4bit" : (
        "unsloth/llama-2-7b",
        "meta-llama/Llama-2-7b-hf",
    ),
    "unsloth/llama-2-13b-bnb-4bit" : (
        "unsloth/llama-2-13b",
        "meta-llama/Llama-2-13b-hf",
    ),
    "unsloth/codellama-34b-bnb-4bit" : (
        "codellama/CodeLlama-34b-hf",
    ),
    "unsloth/zephyr-sft-bnb-4bit" : (
        "unsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "unsloth/tinyllama-bnb-4bit" : (
        "unsloth/tinyllama",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ),
    "unsloth/tinyllama-chat-bnb-4bit" : (
        "unsloth/tinyllama-chat",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    "unsloth/mistral-7b-instruct-v0.1-bnb-4bit" : (
        "unsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit" : (
        "unsloth/mistral-7b-instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit" : (
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "unsloth/llama-2-7b-chat-bnb-4bit" : (
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "unsloth/Mixtral-8x7B-v0.1-unsloth-bnb-4bit" : (
        "unsloth/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "unsloth/Mixtral-8x7B-v0.1-bnb-4bit",
    ),
    "unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit" : (
        "unsloth/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
    ),
    "unsloth/codellama-7b-bnb-4bit" : (
        "unsloth/codellama-7b",
        "codellama/CodeLlama-7b-hf",
    ),
    "unsloth/codellama-13b-bnb-4bit" : (
        "codellama/CodeLlama-13b-hf",
    ),
    "unsloth/yi-6b-bnb-4bit" : (
        "unsloth/yi-6b",
        "01-ai/Yi-6B",
    ),
    "unsloth/solar-10.7b-bnb-4bit" : (
        "upstage/SOLAR-10.7B-v1.0",
    ),
    "unsloth/gemma-7b-bnb-4bit" : (
        "unsloth/gemma-7b",
        "google/gemma-7b",
    ),
    "unsloth/gemma-2b-bnb-4bit" : (
        "unsloth/gemma-2b",
        "google/gemma-2b",
    ),
    "unsloth/gemma-7b-it-bnb-4bit" : (
        "unsloth/gemma-7b-it",
        "google/gemma-7b-it",
    ),
    "unsloth/gemma-2b-bnb-4bit" : (
        "unsloth/gemma-2b-it",
        "google/gemma-2b-it",
    ),
    "unsloth/mistral-7b-v0.2-bnb-4bit" : (
        "unsloth/mistral-7b-v0.2",
        "alpindale/Mistral-7B-v0.2-hf",
    ),
    "unsloth/gemma-1.1-2b-it-bnb-4bit" : (
        "unsloth/gemma-1.1-2b-it",
        "google/gemma-1.1-2b-it",
    ),
    "unsloth/gemma-1.1-7b-it-bnb-4bit" : (
        "unsloth/gemma-1.1-7b-it",
        "google/gemma-1.1-7b-it",
    ),
    "unsloth/Starling-LM-7B-beta-bnb-4bit" : (
        "unsloth/Starling-LM-7B-beta",
        "Nexusflow/Starling-LM-7B-beta",
    ),
    "unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit" : (
        "unsloth/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
    ),
    "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit" : (
        "unsloth/OpenHermes-2.5-Mistral-7B",
        "teknium/OpenHermes-2.5-Mistral-7B",
    ),
    "unsloth/codegemma-2b-bnb-4bit" : (
        "unsloth/codegemma-2b",
        "google/codegemma-2b",
    ),
    "unsloth/codegemma-7b-bnb-4bit" : (
        "unsloth/codegemma-7b",
        "google/codegemma-7b",
    ),
    "unsloth/codegemma-7b-it-bnb-4bit" : (
        "unsloth/codegemma-7b-it",
        "google/codegemma-7b-it",
    ),
    "unsloth/llama-3-8b-bnb-4bit" : (
        "unsloth/llama-3-8b",
        "meta-llama/Meta-Llama-3-8B",
    ),
    "unsloth/llama-3-8b-Instruct-bnb-4bit" : (
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "unsloth/llama-3-70b-bnb-4bit" : (
        "meta-llama/Meta-Llama-3-70B",
    ),
    "unsloth/llama-3-70b-Instruct-bnb-4bit" : (
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit" : (
        "unsloth/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ),
    "unsloth/mistral-7b-v0.3-bnb-4bit" : (
        "unsloth/mistral-7b-v0.3",
        "mistralai/Mistral-7B-v0.3",
    ),
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" : (
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "unsloth/Phi-3-medium-4k-instruct-bnb-4bit" : (
        "unsloth/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
    ),
    "unsloth/Qwen2-0.5B-bnb-4bit" : (
        "unsloth/Qwen2-0.5B",
        "Qwen/Qwen2-0.5B",
    ),
    "unsloth/Qwen2-0.5B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
    ),
    "unsloth/Qwen2-1.5B-bnb-4bit" : (
        "unsloth/Qwen2-1.5B",
        "Qwen/Qwen2-1.5B",
    ),
    "unsloth/Qwen2-1.5B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
    ),
    "unsloth/Qwen2-7B-bnb-4bit" : (
        "unsloth/Qwen2-7B",
        "Qwen/Qwen2-7B",
    ),
    "unsloth/Qwen2-7B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ),
    "unsloth/Qwen2-70B-bnb-4bit" : (
        "Qwen/Qwen2-70B",
    ),
    "unsloth/Qwen2-70B-Instruct-bnb-4bit" : (
        "Qwen/Qwen2-70B-Instruct",
    ),
    "mistralai/Codestral-22B-v0.1" : (
        "mistral-community/Codestral-22B-v0.1",
    ),
    "unsloth/gemma-2-9b-bnb-4bit" : (
        "unsloth/gemma-2-9b",
        "google/gemma-2-9b",
    ),
    "unsloth/gemma-2-27b-bnb-4bit" : (
        "unsloth/gemma-2-27b",
        "google/gemma-2-27b",
    ),
    "unsloth/gemma-2-9b-it-bnb-4bit" : (
        "unsloth/gemma-2-9b-it",
        "google/gemma-2-9b-it",
    ),
    "unsloth/gemma-2-27b-it-bnb-4bit" : (
        "unsloth/gemma-2-27b-it",
        "google/gemma-2-27b-it",
    ),
    "unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit" : ( # Old Phi pre July
        "unsloth/Phi-3-mini-4k-instruct-v0",
    ),
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit" : ( # New 12b Mistral models
        "unsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ),
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit" : ( # New 12b Mistral models
        "unsloth/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Base-2407",
    ),
    "unsloth/Meta-Llama-3.1-8B-unsloth-bnb-4bit" : (
        "unsloth/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    ),
    "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    ),
    "unsloth/Llama-3.1-8B-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B",
        "unsloth/Llama-3.1-8B-bnb-4bit",
    ),
    "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    ),
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit" : (
        "unsloth/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B",
    ),
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit" : (
        "meta-llama/Meta-Llama-3.1-405B",
    ),
    "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" : (
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    ),
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" : (
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    "unsloth/Mistral-Large-Instruct-2407-bnb-4bit" : (
        "mistralai/Mistral-Large-Instruct-2407",
    ),
    "unsloth/gemma-2-2b-bnb-4bit" : (
        "unsloth/gemma-2-2b",
        "google/gemma-2-2b",
    ),
    "unsloth/gemma-2-2b-it-bnb-4bit" : (
        "unsloth/gemma-2-2b-it",
        "google/gemma-2-2b-it",
    ),
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit" : (
        "unsloth/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ),
    "unsloth/c4ai-command-r-08-2024-bnb-4bit" : (
        "CohereForAI/c4ai-command-r-08-2024",
    ),
    "unsloth/c4ai-command-r-plus-08-2024-bnb-4bit" : (
        "CohereForAI/c4ai-command-r-plus-08-2024",
    ),
    "unsloth/Llama-3.1-Storm-8B-bnb-4bit" : (
        "unsloth/Llama-3.1-Storm-8B",
        "akjindal53244/Llama-3.1-Storm-8B",
    ),
    "unsloth/Hermes-3-Llama-3.1-8B-bnb-4bit" : (
        "unsloth/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
    ),
    "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit" : (
        "unsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
    ),
    "unsloth/Hermes-3-Llama-3.1-405B-bnb-4bit" : (
        "NousResearch/Hermes-3-Llama-3.1-405B",
    ),
    "unsloth/SmolLM-135M-bnb-4bit" : (
        "unsloth/SmolLM-135M",
        "HuggingFaceTB/SmolLM-135M",
    ),
    "unsloth/SmolLM-360M-bnb-4bit" : (
        "unsloth/SmolLM-360M",
        "HuggingFaceTB/SmolLM-360M",
    ),
    "unsloth/SmolLM-1.7B-bnb-4bit" : (
        "unsloth/SmolLM-1.7B",
        "HuggingFaceTB/SmolLM-1.7B",
    ),
    "unsloth/SmolLM-135M-Instruct-bnb-4bit" : (
        "unsloth/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-135M-Instruct",
    ),
    "unsloth/SmolLM-360M-Instruct-bnb-4bit" : (
        "unsloth/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
    ),
    "unsloth/SmolLM-1.7B-Instruct-bnb-4bit" : (
        "unsloth/SmolLM-1.7B-Instruct",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
    ),
    "unsloth/Mistral-Small-Instruct-2409-bnb-4bit" : (
        "unsloth/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Small-Instruct-2409",
    ),
    "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ),
    "unsloth/Qwen2.5-72B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ),
    "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B",
        "unsloth/Qwen2.5-0.5B-bnb-4bit",
    ),
    "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B",
        "unsloth/Qwen2.5-1.5B-bnb-4bit",
    ),
    "unsloth/Qwen2.5-3B-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B",
        "unsloth/Qwen2.5-3B-bnb-4bit",
    ),
    "unsloth/Qwen2.5-7B-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B",
        "unsloth/Qwen2.5-7B-bnb-4bit",
    ),
    "unsloth/Qwen2.5-14B-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B",
        "unsloth/Qwen2.5-14B-bnb-4bit",
    ),
    "unsloth/Qwen2.5-32B-bnb-4bit" : (
        "unsloth/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B",
    ),
    "unsloth/Qwen2.5-72B-bnb-4bit" : (
        "unsloth/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B",
    ),
    "unsloth/Qwen2.5-Math-1.5B-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-1.5B",
        "Qwen/Qwen2.5-Math-1.5B",
    ),
    "unsloth/Qwen2.5-Math-7B-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-7B",
        "Qwen/Qwen2.5-Math-7B",
    ),
    "unsloth/Qwen2.5-Math-72B-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-72B",
        "Qwen/Qwen2.5-Math-72B",
    ),
    "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
    ),
    "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ),
    "unsloth/Qwen2.5-Math-72B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Math-72B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-0.5B",
    ),
    "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-1.5B",
    ),
    "unsloth/Qwen2.5-Coder-3B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-3B",
    ),
    "unsloth/Qwen2.5-Coder-7B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-7B",
    ),
    "unsloth/Qwen2.5-Coder-14B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-14B",
        "Qwen/Qwen2.5-Coder-14B",
    ),
    "unsloth/Qwen2.5-Coder-32B-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B",
    ),
    "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ),
    "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "unsloth/Llama-3.2-1B-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "unsloth/Llama-3.2-1B-bnb-4bit",
    ),
    "unsloth/Llama-3.2-3B-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B",
        "unsloth/Llama-3.2-3B-bnb-4bit",
    ),
    "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    ),
    "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    ),
    "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit" : (
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    ),
    "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit" : (
        "unsloth/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
    ),
    "unsloth/Qwen2-VL-2B-bnb-4bit" : (
        "unsloth/Qwen2-VL-2B",
        "Qwen/Qwen2-VL-2B",
    ),
    "unsloth/Qwen2-VL-7B-bnb-4bit" : (
        "unsloth/Qwen2-VL-7B",
        "Qwen/Qwen2-VL-7B",
    ),
    "unsloth/Qwen2-VL-72B-bnb-4bit" : (
        "unsloth/Qwen2-VL-72B",
        "Qwen/Qwen2-VL-72B",
    ),
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    ),
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit" : (
        "unsloth/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
    ),
    "unsloth/Llama-3.2-11B-Vision-unsloth-bnb-4bit" : (
        "unsloth/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision",
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    ),
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit" : (
        "unsloth/Llama-3.2-90B-Vision",
        "meta-llama/Llama-3.2-90B-Vision",
    ),
    "unsloth/Pixtral-12B-2409-unsloth-bnb-4bit" : (
        "unsloth/Pixtral-12B-2409",
        "mistralai/Pixtral-12B-2409",
        "unsloth/Pixtral-12B-2409-bnb-4bit",
    ),
    "unsloth/Pixtral-12B-2409-Base-bnb-4bit" : (
        "unsloth/Pixtral-12B-Base-2409",
        "mistralai/Pixtral-12B-Base-2409",
    ),
    "unsloth/llava-1.5-7b-hf-bnb-4bit" : (
        "unsloth/llava-1.5-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
    ),
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit" : (
        "unsloth/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    ),
    "unsloth/Llama-3.1-Tulu-3-8B-bnb-4bit" : (
        "unsloth/Llama-3.1-Tulu-3-8B",
        "allenai/Llama-3.1-Tulu-3-8B",
    ),
    "unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit" : (
        "unsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "unsloth/QwQ-32B-Preview-bnb-4bit" : (
        "unsloth/QwQ-32B-Preview",
        "Qwen/QwQ-32B-Preview",
    ),
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" : (
        "unsloth/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
    "unsloth/phi-4-unsloth-bnb-4bit" : (
        "unsloth/phi-4",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ),
    "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit" : (
        "unsloth/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ),
    "unsloth/Mistral-Small-24B-Base-2501-unsloth-bnb-4bit" : (
        "unsloth/Mistral-Small-24B-Base-2501",
        "mistralai/Mistral-Small-24B-Base-2501",
        "unsloth/Mistral-Small-24B-Base-2501-bnb-4bit",
    ),
    "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit" : (
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    ),
    "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
    ),
    "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
    ),
    "unsloth/DeepScaleR-1.5B-Preview-unsloth-bnb-4bit" : (
        "unsloth/DeepHermes-3-Llama-3-8B-Preview",
        "agentica-org/DeepScaleR-1.5B-Preview",
        "unsloth/DeepScaleR-1.5B-Preview-bnb-4bit",
    ),
    "unsloth/OpenThinker-7B-unsloth-bnb-4bit" : (
        "unsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
        "unsloth/OpenThinker-7B-bnb-4bit",
    ),
    "unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit" : (
        "unsloth/granite-3.2-2b-instruct",
        "ibm-granite/granite-3.2-2b-instruct",
        "unsloth/granite-3.2-2b-instruct-bnb-4bit",
    ),
    "unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit" : (
        "unsloth/granite-3.2-8b-instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "unsloth/granite-3.2-8b-instruct-bnb-4bit",
    ),
    "unsloth/QwQ-32B-unsloth-bnb-4bit" : (
        "unsloth/QwQ-32B",
        "Qwen/QwQ-32B",
        "unsloth/QwQ-32B-bnb-4bit",
    ),
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "unsloth/gemma-3-1b-it-bnb-4bit",
    ),
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
    ),
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        "unsloth/gemma-3-12b-it-bnb-4bit",
    ),
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-27b-it",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
    ),
    "unsloth/gemma-3-1b-pt-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-1b-pt",
        "google/gemma-3-1b-pt",
        "unsloth/gemma-3-1b-pt-bnb-4bit",
    ),
    "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-4b-pt",
        "google/gemma-3-4b-pt",
        "unsloth/gemma-3-4b-pt-bnb-4bit",
    ),
    "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-12b-pt",
        "google/gemma-3-12b-pt",
        "unsloth/gemma-3-12b-pt-bnb-4bit",
    ),
    "unsloth/gemma-3-27b-pt-unsloth-bnb-4bit" : (
        "unsloth/gemma-3-27b-pt",
        "google/gemma-3-27b-pt",
        "unsloth/gemma-3-27b-pt-bnb-4bit",
    ),
    "unsloth/reka-flash-3-unsloth-bnb-4bit" : (
        "unsloth/reka-flash-3",
        "RekaAI/reka-flash-3",
        "unsloth/reka-flash-3-bnb-4bit",
    ),
    "unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit" : (
        "unsloth/c4ai-command-a-03-2025",
        "CohereForAI/c4ai-command-a-03-2025",
        "unsloth/c4ai-command-a-03-2025-bnb-4bit",
    ),
    "unsloth/aya-vision-32b-unsloth-bnb-4bit" : (
        "unsloth/aya-vision-32b",
        "CohereForAI/aya-vision-32b",
        "unsloth/aya-vision-32b-bnb-4bit",
    ),
    "unsloth/aya-vision-8b-unsloth-bnb-4bit" : (
        "unsloth/aya-vision-8b",
        "CohereForAI/aya-vision-8b",
        "unsloth/aya-vision-8b-bnb-4bit",
    ),
    "unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit" : (
        "unsloth/granite-vision-3.2-2b",
        "ibm-granite/granite-vision-3.2-2b",
        "unsloth/granite-vision-3.2-2b-bnb-4bit",
    ),
    "unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit" : (
        "unsloth/OLMo-2-0325-32B-Instruct",
        "allenai/OLMo-2-0325-32B-Instruct",
        "unsloth/OLMo-2-0325-32B-Instruct-bnb-4bit",
    ),
    "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit" : (
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
    ),
    "unsloth/Mistral-Small-3.1-24B-Base-2503-unsloth-bnb-4bit" : (
        "unsloth/Mistral-Small-3.1-24B-Base-2503",
        "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "unsloth/Mistral-Small-3.1-24B-Base-2503-bnb-4bit",
    ),
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
    ),
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B",
        "unsloth/Qwen3-1.7B-bnb-4bit",
    ),
    "unsloth/Qwen3-4B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-4B",
        "Qwen/Qwen3-4B",
        "unsloth/Qwen3-4B-bnb-4bit",
    ),
    "unsloth/Qwen3-8B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "unsloth/Qwen3-8B-bnb-4bit",
    ),
    "unsloth/Qwen3-14B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-14B",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
    ),
    "unsloth/Qwen3-32B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-32B",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
    ),
    "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "unsloth/Qwen3-30B-A3B-bnb-4bit",
    ),
    "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B-Base",
        "unsloth/Qwen3-0.6B-Base-bnb-4bit",
    ),
    "unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "unsloth/Qwen3-1.7B-Base-bnb-4bit",
    ),
    "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-4B-Base",
        "Qwen/Qwen3-4B-Base",
        "unsloth/Qwen3-4B-Base-bnb-4bit",
    ),
    "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-8B-Base",
        "Qwen/Qwen3-8B-Base",
        "unsloth/Qwen3-8B-Base-bnb-4bit",
    ),
    "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit" : (
        "unsloth/Qwen3-14B-Base",
        "Qwen/Qwen3-14B-Base",
        "unsloth/Qwen3-14B-Base-bnb-4bit",
    ),
    "unsloth/Qwen3-30B-A3B-Base-bnb-4bit" : (
        "unsloth/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-Base",
    ),
    "unsloth/phi-4-reasoning-unsloth-bnb-4bit" : (
        "unsloth/phi-4-reasoning",
        "microsoft/Phi-4-reasoning",
        "unsloth/phi-4-reasoning-bnb-4bit",
    ),
    "unsloth/phi-4-reasoning-plus-unsloth-bnb-4bit" : (
        "unsloth/phi-4-reasoning-plus",
        "microsoft/Phi-4-reasoning-plus",
        "unsloth/phi-4-reasoning-plus-bnb-4bit",
    ),
    "unsloth/phi-4-mini-reasoning-unsloth-bnb-4bit" : (
        "unsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "unsloth/phi-4-mini-reasoning-bnb-4bit",
    ),
    "unsloth/orpheus-3b-0.1-pretrained-unsloth-bnb-4bit" : (
        "unsloth/orpheus-3b-0.1-pretrained",
        "canopylabs/orpheus-3b-0.1-pretrained",
        "unsloth/orpheus-3b-0.1-pretrained-bnb-4bit",
    ),
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" : (
        "unsloth/orpheus-3b-0.1-ft",
        "canopylabs/orpheus-3b-0.1-ft",
        "unsloth/orpheus-3b-0.1-ft-bnb-4bit",
    ),
    "unsloth/csm-1b" : (
        "unsloth/csm-1b",
        "sesame/csm-1b",
    ),
    "unsloth/whisper-large-v3" : (
        "unsloth/whisper-large-v3",
        "openai/whisper-large-v3",
    ),
    "unsloth/whisper-large-v3-turbo" : (
        "unsloth/whisper-large-v3-turbo",
        "openai/whisper-large-v3-turbo",
    ),
    "unsloth/whisper-small" : (
        "unsloth/whisper-small",
        "openai/whisper-small",
    ),
    "unsloth/CrisperWhisper" : (
        "unsloth/CrisperWhisper",
        "nyrahealth/CrisperWhisper",
    ),
    "unsloth/Llasa-1B" : (
        "unsloth/Llasa-1B",
        "HKUSTAudio/Llasa-1B",
    ),
    "unsloth/Spark-TTS-0.5B" : (
        "unsloth/Spark-TTS-0.5B",
        "SparkAudio/Spark-TTS-0.5B",
    ),
    "unsloth/Llama-OuteTTS-1.0-1B" : (
        "unsloth/Llama-OuteTTS-1.0-1B",
        "OuteAI/Llama-OuteTTS-1.0-1B",
    ),
    "unsloth/medgemma-4b-it-unsloth-bnb-4bit" : (
        "unsloth/medgemma-4b-it",
        "google/medgemma-4b-it",
        "unsloth/medgemma-4b-it-bnb-4bit",
    ),
    "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit" : (
        "unsloth/medgemma-27b-text-it",
        "google/medgemma-27b-text-it",
        "unsloth/medgemma-27b-text-it-bnb-4bit",
    ),
    "unsloth/Devstral-Small-2505-unsloth-bnb-4bit" : (
        "unsloth/Devstral-Small-2505",
        "mistralai/Devstral-Small-2505",
        "unsloth/Devstral-Small-2505-bnb-4bit",
    ),
    "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit" : (
        "unsloth/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "unsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit",
    ),
    "unsloth/Magistral-Small-2506-unsloth-bnb-4bit" : (
        "unsloth/Magistral-Small-2506",
        "mistralai/Magistral-Small-2506",
        "unsloth/Magistral-Small-2506-bnb-4bit",
    ),
    "unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit" : (
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
    ),
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3n-E4B-it",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    ),
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit" : (
        "unsloth/gemma-3n-E2B-it",
        "google/gemma-3n-E2B-it",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    ),
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit" : (
        "unsloth/gemma-3n-E4B",
        "google/gemma-3n-E4B",
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    ),
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit" : (
        "unsloth/gemma-3n-E2B",
        "google/gemma-3n-E2B",
        "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    ),
    "unsloth/Devstral-Small-2507-unsloth-bnb-4bit" : (
        "unsloth/Devstral-Small-2507",
        "mistralai/Devstral-Small-2507",
        "unsloth/Devstral-Small-2507-bnb-4bit",
    ),
}

INT_TO_FLOAT_MAPPER  = {}
FLOAT_TO_INT_MAPPER  = {}
MAP_TO_UNSLOTH_16bit = {}

for key, values in __INT_TO_FLOAT_MAPPER.items():
    INT_TO_FLOAT_MAPPER[key] = values[0]

    for value in values:
        FLOAT_TO_INT_MAPPER[value] = key
    pass

    # Map to Unsloth version for 16bit versions
    if len(values) == 2:
        if values[0].startswith("unsloth"):
            MAP_TO_UNSLOTH_16bit[values[1]] = values[0]
            MAP_TO_UNSLOTH_16bit[values[1].lower()] = values[0]
        pass
    elif len(values) == 3:
        # Dynamic Unsloth quantization
        if values[0].startswith("unsloth"):
            MAP_TO_UNSLOTH_16bit[values[1]] = values[0]
            MAP_TO_UNSLOTH_16bit[values[1].lower()] = values[0]
            MAP_TO_UNSLOTH_16bit[values[2]] = values[0]
            MAP_TO_UNSLOTH_16bit[values[2].lower()] = values[0]
        pass
    pass

    # Get lowercased
    lowered_key = key.lower()
    INT_TO_FLOAT_MAPPER[lowered_key] = values[0].lower()

    for value in values:
        FLOAT_TO_INT_MAPPER[value.lower()] = lowered_key
    pass
pass
