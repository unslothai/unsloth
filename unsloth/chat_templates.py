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
    "get_chat_template",
    "test_chat_templates",
    "test_hf_gguf_equivalence",
    "remove_special_tokens",

    "to_sharegpt",
    "standardize_sharegpt",
    "standardize_data_formats",
    "apply_chat_template",
    "train_on_responses_only",

    "test_construct_chat_template",
]

from transformers import StoppingCriteria, StoppingCriteriaList
from torch import LongTensor, FloatTensor
from transformers.models.llama.modeling_llama import logger
from .save import patch_saving_functions
import os
import shutil
from .tokenizer_utils import *
from .models._utils import patch_tokenizer
import re
from unsloth_zoo.dataset_utils import (
    train_on_responses_only,
    standardize_data_formats,
)
standardize_sharegpt = standardize_data_formats
CHAT_TEMPLATES = {}
DEFAULT_SYSTEM_MESSAGE = {}

# =========================================== Unsloth
# Unsloth efficient template leverages from Zephyr
unsloth_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_message}' + '\n' }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '>>> User: ' + message['content'] + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '>>> Assistant: ' }}"\
    "{% endif %}"
pass

unsloth_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}>>> User: {{ .Prompt }}
{{ end }}>>> Assistant: {{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """You are a helpful assistant to the user"""
'''

unsloth_eos_token = "eos_token"
CHAT_TEMPLATES["unsloth"] = (unsloth_template, unsloth_eos_token, False, unsloth_ollama,)
DEFAULT_SYSTEM_MESSAGE["unsloth"] = "You are a helpful assistant to the user"
pass

# =========================================== Zephyr
# Zephyr has no BOS!
zephyr_template = \
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}"\
        "{% else %}"\
            "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|assistant|>\n' }}"\
    "{% endif %}"
pass

zephyr_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}{__EOS_TOKEN__}
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}{__EOS_TOKEN__}
{{ end }}<|assistant|>
{{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

zephyr_eos_token = "eos_token"
CHAT_TEMPLATES["zephyr"] = (zephyr_template, zephyr_eos_token, False, zephyr_ollama,)
DEFAULT_SYSTEM_MESSAGE["zephyr"] = None # No system message in Zephyr
pass

# =========================================== ChatML
# ChatML has no BOS and not EOS! Rather <|im_start|> and <|im_end|> acts as BOS / EOS.
chatml_template = \
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"\
        "{% else %}"\
            "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|im_start|>assistant\n' }}"\
    "{% endif %}"
pass

chatml_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

chatml_eos_token = "<|im_end|>"
CHAT_TEMPLATES["chatml"] = (chatml_template, chatml_eos_token, True, chatml_ollama,)
DEFAULT_SYSTEM_MESSAGE["chatml"] = None # No system message in ChatML
pass

# =========================================== Mistral-1
# Mistral Instruct doesn't allow system prompts, so we append it to the user message.
mistral_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{% if messages[1]['role'] == 'user' %}"\
            "{{ '[INST] ' + messages[0]['content'] + ' ' + messages[1]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[2:] %}"\
        "{% else %}"\
            "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% endif %}"\
    "{% else %}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"
pass

# Ollama from https://www.ollama.com/library/mistral
mistral_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

mistral_eos_token = "eos_token"
CHAT_TEMPLATES["mistral"] = (mistral_template, mistral_eos_token, False, mistral_ollama,)
DEFAULT_SYSTEM_MESSAGE["mistral"] = None # No system message in Mistral
pass

# =========================================== Llama-2
# Adds BOS to every convo! And weird <<SYS>> system messages.
llama_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{% if messages[1]['role'] == 'user' %}"\
            "{{ bos_token + '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' + messages[1]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[2:] %}"\
        "{% else %}"\
            "{{ bos_token + '[INST] ' + messages[0]['content'] + ' [/INST]' }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% endif %}"\
    "{% else %}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ ' ' + message['content'].strip() + ' ' + eos_token }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"
pass

# Ollama from https://www.ollama.com/library/llama3
llama_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] <<SYS>>{{ .System }}<</SYS>>

{{ .Prompt }} [/INST]"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

llama_eos_token = "eos_token"
CHAT_TEMPLATES["llama"] = (llama_template, llama_eos_token, False, llama_ollama,)
DEFAULT_SYSTEM_MESSAGE["llama"] = None # No system message in Llama
pass

# ===========================================  Vicuna
# https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
vicuna_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + ' ' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_message}' + ' ' }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ 'USER: ' + message['content'] + ' ' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ 'ASSISTANT: ' + message['content'] + eos_token }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ 'ASSISTANT:' }}"\
    "{% endif %}"
pass

# Ollama from https://www.ollama.com/library/vicuna
vicuna_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }} {{ end }}{{ if .Prompt }}USER: {{ .Prompt }} {{ end }}ASSISTANT: {{ .Response }} {__EOS_TOKEN__}"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

vicuna_eos_token = "eos_token"
CHAT_TEMPLATES["vicuna"] = (vicuna_template, vicuna_eos_token, False, vicuna_ollama,)
DEFAULT_SYSTEM_MESSAGE["vicuna"] = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
pass

# =========================================== Vicuna Old
# https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
vicuna_old_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_message}' + '\n' }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '### Human: ' + message['content'] + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '### Assistant: ' + message['content'] + eos_token + '\n' }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '### Assistant:' }}"\
    "{% endif %}"
pass

vicuna_old_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}### Human: {{ .Prompt }}
{{ end }}### Assistant: {{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
'''

vicuna_old_eos_token = "eos_token"
CHAT_TEMPLATES["vicuna_old"] = (vicuna_old_template, vicuna_old_eos_token, False, vicuna_old_ollama,)
DEFAULT_SYSTEM_MESSAGE["vicuna_old"] = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\\'s questions."

CHAT_TEMPLATES["vicuna old"] = CHAT_TEMPLATES["vicuna_old"]
DEFAULT_SYSTEM_MESSAGE["vicuna old"] = DEFAULT_SYSTEM_MESSAGE["vicuna_old"]
pass

# =========================================== Alpaca multi turn
# https://github.com/tatsu-lab/stanford_alpaca Changed for multi-turn convos
alpaca_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_message}' + '\n\n' }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '### Instruction:\n' + message['content'] + '\n\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '### Response:\n' + message['content'] + eos_token + '\n\n' }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '### Response:\n' }}"\
    "{% endif %}"
pass

alpaca_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ if .Prompt }}### Instruction:
{{ .Prompt }}{{ end }}

### Response:
{{ .Response }}{__EOS_TOKEN__}

"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """Below are some instructions that describe some tasks. Write responses that appropriately complete each request."""
'''

alpaca_eos_token = "eos_token"
CHAT_TEMPLATES["alpaca"] = (alpaca_template, alpaca_eos_token, False, alpaca_ollama,)
DEFAULT_SYSTEM_MESSAGE["alpaca"] = "Below are some instructions that describe some tasks. Write responses that appropriately complete each request."
pass

# =========================================== Gemma
# https://huggingface.co/google/gemma-7b-it
# Notice we must use |trim for lstrip and rstrip. <start_of_turn> maps to 106.
# <end_of_turn> maps to 107. user and model are normal 1 word tokens.
gemma_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{'<start_of_turn>user\n' + messages[0]['content'] | trim + ' ' + messages[1]['content'] | trim + '<end_of_turn>\n'}}"\
        "{% set messages = messages[2:] %}"\
    "{% endif %}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{'<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n'}}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{'<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<start_of_turn>model\n' }}"\
    "{% endif %}"
pass

# Ollama from https://www.ollama.com/library/gemma
gemma_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
PARAMETER repeat_penalty 1
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER penalize_newline false
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

gemma_eos_token = "<end_of_turn>"
CHAT_TEMPLATES["gemma"] = (gemma_template, gemma_eos_token, True, gemma_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma"] = None # No system message in Gemma
pass

# =========================================== Gemma with ChatML instead
# We find using <eos> is still more appropriate!
gemma_chatml_template = "{{ bos_token }}" + chatml_template
pass

gemma_chatml_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER penalize_newline false
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

gemma_chatml_eos_token = (
    {"<start_of_turn>" : "<|im_start|>", "<eos>" : "<|im_end|>"},
    "<|im_end|>",
)
CHAT_TEMPLATES["gemma_chatml"] = (gemma_chatml_template, gemma_chatml_eos_token, True, gemma_chatml_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma_chatml"] = None # No system message in Gemma
pass

# =========================================== Gemma 2
# Same as Gemma 1, but with sliding window attention!
# https://ollama.com/library/gemma2/blobs/6522ca797f47
gemma2_template = gemma_template
gemma2_ollama = gemma_ollama + "PARAMETER num_ctx 4096\n"
gemma2_eos_token = "<end_of_turn>"
CHAT_TEMPLATES["gemma2"] = (gemma2_template, gemma2_eos_token, True, gemma2_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma2"] = None # No system message in Gemma 2

# =========================================== Gemma 2 with ChatML instead
gemma2_chatml_template = gemma_chatml_template
gemma2_chatml_ollama = gemma_chatml_ollama + "PARAMETER num_ctx 4096\n"
gemma2_chatml_eos_token = gemma_chatml_eos_token
CHAT_TEMPLATES["gemma2_chatml"] = (gemma2_chatml_template, gemma2_chatml_eos_token, True, gemma2_chatml_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma2_chatml"] = None # No system message in Gemma 2
pass

# =========================================== Llama-3
# Weirdly \n\n is needed?
llama3_template = \
    "{{ bos_token }}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"\
        "{% else %}"\
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"\
    "{% endif %}"
pass

# Ollama from https://www.ollama.com/library/llama3
llama3_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

llama3_template_eos_token = "eos_token"

CHAT_TEMPLATES["llama-3"] = (llama3_template, llama3_template_eos_token, False, llama3_ollama,)
DEFAULT_SYSTEM_MESSAGE["llama-3"] = None # No system message in Llama-3

CHAT_TEMPLATES["llama3"] = (llama3_template, llama3_template_eos_token, False, llama3_ollama,)
DEFAULT_SYSTEM_MESSAGE["llama3"] = None # No system message in Llama-3
pass


# =========================================== Phi-3
# "{{ bos_token }}"\ # Phi-3.5 removes BOS?
phi3_template = \
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"\
        "{% else %}"\
            "{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n'}}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|assistant|>\n' }}"\
    "{% endif %}"
pass

# Ollama from https://www.ollama.com/library/phi3
phi3_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

phi3_template_eos_token = "<|end|>"
CHAT_TEMPLATES["phi-3"]   = (phi3_template, phi3_template_eos_token, False, phi3_ollama,)
DEFAULT_SYSTEM_MESSAGE["phi-3"] = None # No system message in Phi-3

CHAT_TEMPLATES["phi-35"]  = CHAT_TEMPLATES["phi-3"]
DEFAULT_SYSTEM_MESSAGE["phi-35"] = None # No system message in Phi-3.5

CHAT_TEMPLATES["phi-3.5"] = CHAT_TEMPLATES["phi-3"]
DEFAULT_SYSTEM_MESSAGE["phi-3.5"] = None # No system message in Phi-3.5
pass

# =========================================== Llama-3.1
"""
No trimming in Llama 3.1 Instruct!
Also an extra newline for Cutting Knowledge Date
See https://colab.research.google.com/drive/1Xpqq5xpIgO-B00MQ-UccYMwN2J8QFgBM?usp=sharing

Also should be

import datetime
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
    date_string = datetime.today().strftime("%d %B %Y")),
)
"""

llama31_template = \
"""{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 July 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "{system_message}" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content'] %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
pass

# Ollama from https://ollama.com/library/llama3.1 (needs updating!)
llama31_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .Messages }}
{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original use question.
{{- end }}
{{- end }}<|eot_id|>
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{- if and $.Tools $last }}

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ $.Tools }}
{{- end }}

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}

{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}{{ if not $last }}<|eot_id|>{{ end }}
{{- end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}
{{- else }}
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}{{ .Response }}{{ if .Response }}<|eot_id|>{{ end }}"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|eom_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

llama31_template_eos_token = "eos_token"
CHAT_TEMPLATES["llama-3.1"] = (llama31_template, llama31_template_eos_token, False, llama31_ollama,)
DEFAULT_SYSTEM_MESSAGE["llama-3.1"] = "" # Llama3.1 default system message is empty + the dates

CHAT_TEMPLATES["llama-31"]  = (llama31_template, llama31_template_eos_token, False, llama31_ollama,)
DEFAULT_SYSTEM_MESSAGE["llama-31"] = "" # Llama3.1 default system message is empty + the dates

for version in ("llama-3.2", "llama-3.3", "llama-32", "llama-33"):
    CHAT_TEMPLATES[version] = CHAT_TEMPLATES["llama-3.1"]
    DEFAULT_SYSTEM_MESSAGE[version] = ""
pass


# =========================================== Qwen 2.5
qwen25_template = \
"""{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- \'<|im_start|>system\\n\' + messages[0][\'content\'] + \'<|im_end|>\\n\' }}
    {%- else %}
        {{- \'<|im_start|>system\\n{system_message}<|im_end|>\\n\' }}
    {%- endif %}\n{%- endif %}\n{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}
    {%- elif message.role == "assistant" %}
        {{- \'<|im_start|>\' + message.role }}
        {%- if message.content %}
            {{- \'\\n\' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- \'\\n<tool_call>\\n{"name": "\' }}
            {{- tool_call.name }}
            {{- \'", "arguments": \' }}
            {{- tool_call.arguments | tojson }}
            {{- \'}\\n</tool_call>\' }}
        {%- endfor %}
        {{- \'<|im_end|>\\n\' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}            {{- \'<|im_start|>user\' }}
        {%- endif %}
        {{- \'\\n<tool_response>\\n\' }}
        {{- message.content }}
        {{- \'\\n</tool_response>\' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- \'<|im_end|>\\n\' }}
        {%- endif %}
    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}
    {{- \'<|im_start|>assistant\\n\' }}
{%- endif %}
"""


# Ollama from https://ollama.com/library/qwen2.5/blobs/eb4402837c78
qwen25_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

qwen25_template_eos_token = "eos_token"
qwen25_default_system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." 
CHAT_TEMPLATES["qwen-2.5"] = (qwen25_template, qwen25_template_eos_token, False, qwen25_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen-2.5"] = qwen25_default_system_message # No system message in Qwen 2.5

CHAT_TEMPLATES["qwen-25"]  = (qwen25_template, qwen25_template_eos_token, False, qwen25_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen-25"] = qwen25_default_system_message # No system message in Qwen 2.5

CHAT_TEMPLATES["qwen25"]   = (qwen25_template, qwen25_template_eos_token, False, qwen25_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen25"] = qwen25_default_system_message # No system message in Qwen 2.5

CHAT_TEMPLATES["qwen2.5"]  = (qwen25_template, qwen25_template_eos_token, False, qwen25_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen2.5"] = qwen25_default_system_message # No system message in Qwen 2.5
pass

# =========================================== Phi-4
# "{{ bos_token }}"\ # Phi-4 removes BOS?
phi4_template = \
    "{% for message in messages %}"\
        "{% if (message['role'] == 'system') %}"\
            "{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}"\
        "{% elif (message['role'] == 'user') %}"\
            "{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}"\
        "{% elif (message['role'] == 'assistant') %}"\
            "{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '<|im_start|>assistant<|im_sep|>' }}"\
    "{% endif %}"
pass

_phi4_ollama_template = \
    "{{ if .System }}<|im_start|><|system|><|im_sep|>{{ .System }}<|im_end|>{{ end }}"\
    "{{ if .Prompt }}<|im_start|><|user|><|im_sep|>{{ .Prompt }}<|im_end|>{{ end }}"\
    "<|im_start|><|assistant|><|im_sep|>{{ .Response }}<|im_end|>"

# Ollama from https://www.ollama.com/library/phi4 is different
phi4_ollama = \
f'''
FROM {{__FILE_LOCATION__}}
TEMPLATE """{_phi4_ollama_template}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_sep|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

phi4_template_eos_token = "<|im_end|>"
CHAT_TEMPLATES["phi-4"] = (phi4_template, phi4_template_eos_token, False, phi4_ollama,)
DEFAULT_SYSTEM_MESSAGE["phi-4"] = None # No system message in Phi-4
pass


# =========================================== Gemma-3
# Obtained via
# print(tokenizer.chat_template.replace("}\n", "####").replace("\n", "\\n").replace("####", "}\n"))
gemma3_template = \
"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
"""

# Ollama from https://ollama.com/library/gemma3/blobs/e0a42594d802
gemma3_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER temperature 0.1
PARAMETER min_p 0.0
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_predict 32768
'''

gemma3_template_eos_token = "<end_of_turn>"
CHAT_TEMPLATES["gemma-3"] = (gemma3_template, gemma3_template_eos_token, False, gemma3_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma-3"] = None # No system message in Gemma-3

CHAT_TEMPLATES["gemma3"] = (gemma3_template, gemma3_template_eos_token, False, gemma3_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma3"] = None # No system message in Gemma-3
pass

# =========================================== Qwen-3
# Official Qwen-3 chat template (see https://ollama.com/library/qwen3/blobs/eb4402837c78)
qwen3_template = \
"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for forward_message in messages %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- set message = messages[index] %}
    {%- set current_content = message.content if message.content is not none else '' %}
    {%- set tool_start = '<tool_response>' %}
    {%- set tool_start_length = tool_start|length %}
    {%- set start_of_message = current_content[:tool_start_length] %}
    {%- set tool_end = '</tool_response>' %}
    {%- set tool_end_length = tool_end|length %}
    {%- set start_pos = (current_content|length) - tool_end_length %}
    {%- if start_pos < 0 %}
        {%- set start_pos = 0 %}
    {%- endif %}
    {%- set end_of_message = current_content[start_pos:] %}
    {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
                {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
                {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""

# Ollama template for Qwen-3 (see https://ollama.com/library/qwen3/blobs/eb4402837c78)
qwen3_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.6
PARAMETER min_p 0.0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1
'''

qwen3_template_eos_token = "<|im_end|>"
CHAT_TEMPLATES["qwen-3"] = (qwen3_template, qwen3_template_eos_token, False, qwen3_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen-3"] = None # No default system message for Qwen-3

CHAT_TEMPLATES["qwen3"] = (qwen3_template, qwen3_template_eos_token, False, qwen3_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen3"] = None # No default system message for Qwen-3
pass

# =========================================== Gemma-3n
# Obtained via
# print(tokenizer.chat_template.replace("}\n", "####").replace("\n", "\\n").replace("####", "}\n"))
gemma3n_template = \
"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'audio' -%}
                {{ '<audio_soft_token>' }}
            {%- elif item['type'] == 'image' -%}
                {{ '<image_soft_token>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""

# Ollama from https://ollama.com/library/gemma3n/blobs/e0a42594d802
gemma3n_ollama = \
'''
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}
'''

gemma3n_template_eos_token = "<end_of_turn>"
CHAT_TEMPLATES["gemma-3n"] = (gemma3n_template, gemma3n_template_eos_token, False, gemma3n_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma-3n"] = None # No system message in Gemma-3n

CHAT_TEMPLATES["gemma3n"] = (gemma3n_template, gemma3n_template_eos_token, False, gemma3n_ollama,)
DEFAULT_SYSTEM_MESSAGE["gemma3n"] = None # No system message in Gemma-3n
pass

# =========================================== GPT-OSS
# Obtained via
# print(tokenizer.chat_template.replace("}\n", "####").replace("\n", "\\n").replace("####", "}\n"))
gptoss_template = \
"""{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - "builtin_tools": A list, can contain "browser" and/or "python".
  - "model_identity": A string that optionally describes the model identity.
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
 #}

{#- Tool Definition Rendering ============================================== #}
{%- macro render_typescript_type(param_spec, required_params, is_nullable=false) -%}
    {%- if param_spec.type == "array" -%}
        {%- if param_spec['items'] -%}
            {%- if param_spec['items']['type'] == "string" -%}
                {{- "string[]" }}
            {%- elif param_spec['items']['type'] == "number" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "integer" -%}
                {{- "number[]" }}
            {%- elif param_spec['items']['type'] == "boolean" -%}
                {{- "boolean[]" }}
            {%- else -%}
                {%- set inner_type = render_typescript_type(param_spec['items'], required_params) -%}
                {%- if inner_type == "object | object" or inner_type|length > 50 -%}
                    {{- "any[]" }}
                {%- else -%}
                    {{- inner_type + "[]" }}
                {%- endif -%}
            {%- endif -%}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- else -%}
            {{- "any[]" }}
            {%- if param_spec.nullable -%}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type is defined and param_spec.type is iterable and param_spec.type is not string and param_spec.type is not mapping and param_spec.type[0] is defined -%}
        {#- Handle array of types like ["object", "object"] from Union[dict, list] #}
        {%- if param_spec.type | length > 1 -%}
            {{- param_spec.type | join(" | ") }}
        {%- else -%}
            {{- param_spec.type[0] }}
        {%- endif -%}
    {%- elif param_spec.oneOf -%}
        {#- Handle oneOf schemas - check for complex unions and fallback to any #}
        {%- set has_object_variants = false -%}
        {%- for variant in param_spec.oneOf -%}
            {%- if variant.type == "object" -%}
                {%- set has_object_variants = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- if has_object_variants and param_spec.oneOf|length > 1 -%}
            {{- "any" }}
        {%- else -%}
            {%- for variant in param_spec.oneOf -%}
                {{- render_typescript_type(variant, required_params) -}}
                {%- if variant.description %}
                    {{- "// " + variant.description }}
                {%- endif -%}
                {%- if variant.default is defined %}
                    {{ "// default: " + variant.default|tojson }}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- " | " }}
                {% endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif param_spec.type == "string" -%}
        {%- if param_spec.enum -%}
            {{- '"' + param_spec.enum|join('" | "') + '"' -}}
        {%- else -%}
            {{- "string" }}
            {%- if param_spec.nullable %}
                {{- " | null" }}
            {%- endif -%}
        {%- endif -%}
    {%- elif param_spec.type == "number" -%}
        {{- "number" }}
    {%- elif param_spec.type == "integer" -%}
        {{- "number" }}
    {%- elif param_spec.type == "boolean" -%}
        {{- "boolean" }}

    {%- elif param_spec.type == "object" -%}
        {%- if param_spec.properties -%}
            {{- "{\n" }}
            {%- for prop_name, prop_spec in param_spec.properties.items() -%}
                {{- prop_name -}}
                {%- if prop_name not in (param_spec.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{ render_typescript_type(prop_spec, param_spec.required or []) }}
                {%- if not loop.last -%}
                    {{-", " }}
                {%- endif -%}
            {%- endfor -%}
            {{- "}" }}
        {%- else -%}
            {{- "object" }}
        {%- endif -%}
    {%- else -%}
        {{- "any" }}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_namespace(namespace_name, tools) -%}
    {{- "## " + namespace_name + "\n\n" }}
    {{- "namespace " + namespace_name + " {\n\n" }}
    {%- for tool in tools %}
        {%- set tool = tool.function %}
        {{- "// " + tool.description + "\n" }}
        {{- "type "+ tool.name + " = " }}
        {%- if tool.parameters and tool.parameters.properties %}
            {{- "(_: {\n" }}
            {%- for param_name, param_spec in tool.parameters.properties.items() %}
                {%- if param_spec.description %}
                    {{- "// " + param_spec.description + "\n" }}
                {%- endif %}
                {{- param_name }}
                {%- if param_name not in (tool.parameters.required or []) -%}
                    {{- "?" }}
                {%- endif -%}
                {{- ": " }}
                {{- render_typescript_type(param_spec, tool.parameters.required or []) }}
                {%- if param_spec.default is defined -%}
                    {%- if param_spec.enum %}
                        {{- ", // default: " + param_spec.default }}
                    {%- elif param_spec.oneOf %}
                        {{- "// default: " + param_spec.default }}
                    {%- else %}
                        {{- ", // default: " + param_spec.default|tojson }}
                    {%- endif -%}
                {%- endif -%}
                {%- if not loop.last %}
                    {{- ",\n" }}
                {%- else %}
                    {{- ",\n" }}
                {%- endif -%}
            {%- endfor %}
            {{- "}) => any;\n\n" }}
        {%- else -%}
            {{- "() => any;\n\n" }}
        {%- endif -%}
    {%- endfor %}
    {{- "} // namespace " + namespace_name }}
{%- endmacro -%}

{%- macro render_builtin_tools(browser_tool, python_tool) -%}
    {%- if browser_tool %}
        {{- "## browser\n\n" }}
        {{- "// Tool for browsing.\n" }}
        {{- "// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.\n" }}
        {{- "// Cite information from the tool using the following format:\n" }}
        {{- "// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\n" }}
        {{- "// Do not quote more than 10 words directly from the tool output.\n" }}
        {{- "// sources=web (default: web)\n" }}
        {{- "namespace browser {\n\n" }}
        {{- "// Searches for information related to `query` and displays `topn` results.\n" }}
        {{- "type search = (_: {\n" }}
        {{- "query: string,\n" }}
        {{- "topn?: number, // default: 10\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\n" }}
        {{- "// Valid link ids are displayed with the formatting: `【{id}†.*】`.\n" }}
        {{- "// If `cursor` is not provided, the most recent page is implied.\n" }}
        {{- "// If `id` is a string, it is treated as a fully qualified URL associated with `source`.\n" }}
        {{- "// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\n" }}
        {{- "// Use this function without `id` to scroll to a new location of an opened page.\n" }}
        {{- "type open = (_: {\n" }}
        {{- "id?: number | string, // default: -1\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "loc?: number, // default: -1\n" }}
        {{- "num_lines?: number, // default: -1\n" }}
        {{- "view_source?: boolean, // default: false\n" }}
        {{- "source?: string,\n" }}
        {{- "}) => any;\n\n" }}
        {{- "// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.\n" }}
        {{- "type find = (_: {\n" }}
        {{- "pattern: string,\n" }}
        {{- "cursor?: number, // default: -1\n" }}
        {{- "}) => any;\n\n" }}
        {{- "} // namespace browser\n\n" }}
    {%- endif -%}

    {%- if python_tool %}
        {{- "## python\n\n" }}
        {{- "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\n" }}
        {{- "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\n\n" }}
    {%- endif -%}
{%- endmacro -%}

{#- System Message Construction ============================================ #}
{%- macro build_system_message() -%}
    {%- if model_identity is not defined %}
        {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
    {%- endif %}
    {{- model_identity + "\n" }}
    {{- "Knowledge cutoff: 2024-06\n" }}
    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}
    {%- if reasoning_effort is not defined %}
        {%- set reasoning_effort = "medium" %}
    {%- endif %}
    {{- "Reasoning: " + reasoning_effort + "\n\n" }}
    {%- if builtin_tools is defined and builtin_tools is not none %}
        {{- "# Tools\n\n" }}
        {%- set available_builtin_tools = namespace(browser=false, python=false) %}
        {%- for tool in builtin_tools %}
            {%- if tool == "browser" %}
                {%- set available_builtin_tools.browser = true %}
            {%- elif tool == "python" %}
                {%- set available_builtin_tools.python = true %}
            {%- endif %}
        {%- endfor %}
        {{- render_builtin_tools(available_builtin_tools.browser, available_builtin_tools.python) }}
    {%- endif -%}
    {{- "# Valid channels: analysis, commentary, final. Channel must be included for every message." }}
    {%- if tools -%}
        {{- "\nCalls to these tools must go to the commentary channel: 'functions'." }}
    {%- endif -%}
{%- endmacro -%}

{#- Main Template Logic ================================================= #}
{#- Set defaults #}

{#- Render system message #}
{{- "<|start|>system<|message|>" }}
{{- build_system_message() }}
{{- "<|end|>" }}

{#- Extract developer message #}
{%- if developer_instructions is defined and developer_instructions is not none %}
    {%- set developer_message = developer_instructions %}
    {%- set loop_messages = messages %}
{%- elif messages[0].role == "developer" or messages[0].role == "system" %}
    {%- set developer_message = messages[0].content %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set developer_message = "" %}
    {%- set loop_messages = messages %}
{%- endif %}

{#- Render developer message #}
{%- if developer_message or tools %}
    {{- "<|start|>developer<|message|>" }}
    {%- if developer_message %}
        {{- "# Instructions\n\n" }}
        {{- developer_message }}
    {%- endif %}
    {%- if tools -%}
        {%- if developer_message %}
            {{- "\n\n" }}
        {%- endif %}
        {{- "# Tools\n\n" }}
        {{- render_tool_namespace("functions", tools) }}
    {%- endif -%}
    {{- "<|end|>" }}
{%- endif %}

{#- Render messages #}
{%- set last_tool_call = namespace(name=none) %}
{%- for message in loop_messages -%}
    {#- At this point only assistant/user/tool messages should remain #}
    {%- if message.role == 'assistant' -%}
        {#- Checks to ensure the messages are being passed in the format we expect #}
        {%- if "content" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.content or "<|channel|>final<|message|>" in message.content %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the content field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "thinking" in message %}
            {%- if "<|channel|>analysis<|message|>" in message.thinking or "<|channel|>final<|message|>" in message.thinking %}
                {{- raise_exception("You have passed a message containing <|channel|> tags in the thinking field. Instead of doing this, you should pass analysis messages (the string between '<|message|>' and '<|end|>') in the 'thinking' field, and final messages (the string between '<|message|>' and '<|end|>') in the 'content' field.") }}
            {%- endif %}
        {%- endif %}
        {%- if "tool_calls" in message %}
            {#- We need very careful handling here - we want to drop the tool call analysis message if the model #}
            {#- has output a later <|final|> message, but otherwise we want to retain it. This is the only case #}
            {#- when we render CoT/analysis messages in inference. #}
            {%- set future_final_message = namespace(found=false) %}
            {%- for future_message in loop_messages[loop.index:] %}
                {%- if future_message.role == 'assistant' and "tool_calls" not in future_message %}
                    {%- set future_final_message.found = true %}
                {%- endif %}
            {%- endfor %}
            {#- We assume max 1 tool call per message, and so we infer the tool call name #}
            {#- in "tool" messages from the most recent assistant tool call name #}
            {%- set tool_call = message.tool_calls[0] %}
            {%- if tool_call.function %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {%- if message.content and message.thinking %}
                {{- raise_exception("Cannot pass both content and thinking in an assistant message with tool calls! Put the analysis message in one or the other, but not both.") }}
            {%- elif message.content and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- elif message.thinking and not future_final_message.found %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {{- "<|start|>assistant to=" }}
            {{- "functions." + tool_call.name + "<|channel|>commentary " }}
            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}
            {%- if tool_call.arguments is string %}
                {{- tool_call.arguments }}
            {%- else %}
                {{- tool_call.arguments|tojson }}
            {%- endif %}
            {{- "<|call|>" }}
            {%- set last_tool_call.name = tool_call.name %}
        {%- elif loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}
            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
        {%- elif "thinking" in message %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- else %}
            {#- CoT is dropped during all previous turns, so we never render it for inference #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>" }}
            {%- set last_tool_call.name = none %}
        {%- endif %}
    {%- elif message.role == 'tool' -%}
        {%- if last_tool_call.name is none %}
            {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
        {%- endif %}
        {{- "<|start|>functions." + last_tool_call.name }}
        {%- if message.content is string %}
            {{- " to=assistant<|channel|>commentary<|message|>" + message.content + "<|end|>" }}
        {%- else %}
            {{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}
        {%- endif %}
    {%- elif message.role == 'user' -%}
        {{- "<|start|>user<|message|>" + message.content + "<|end|>" }}
    {%- endif -%}
{%- endfor -%}

{#- Generation prompt #}
{%- if add_generation_prompt -%}
<|start|>assistant
{%- endif -%}"""

# Ollama from https://ollama.com/library/gemma3n/blobs/e0a42594d802
gptoss_ollama = \
'''<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {{ currentDate }}
{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}

Reasoning: {{ .ThinkLevel }}
{{- else if or (not .IsThinkSet) (and .IsThinkSet .Think) }}

Reasoning: medium
{{- end }}

{{- $hasNonBuiltinTools := false }}
{{- if .Tools -}}
{{- $hasBrowserSearch := false }}
{{- $hasBrowserOpen := false }}
{{- $hasBrowserFind := false }}
{{- $hasPython := false }}
  {{- range .Tools }}
    {{- if eq .Function.Name "browser.search" -}}{{- $hasBrowserSearch = true -}}
    {{- else if eq .Function.Name "browser.open" -}}{{- $hasBrowserOpen = true -}}
    {{- else if eq .Function.Name "browser.find" -}}{{- $hasBrowserFind = true -}}
    {{- else if eq .Function.Name "python" -}}{{- $hasPython = true -}}
    {{- else }}{{ $hasNonBuiltinTools = true -}}
    {{- end }}
  {{- end }}
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind $hasPython }}

# Tools
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind }}

## browser

// Tool for browsing.
// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
// Do not quote more than 10 words directly from the tool output.
// sources=web (default: web)
namespace browser {
{{- if $hasBrowserSearch }}

// Searches for information related to `query` and displays `topn` results.
type search = (_: {
query: string,
topn?: number, // default: 10
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserOpen }}

// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
// Valid link ids are displayed with the formatting: `【{id}†.*】`.
// If `cursor` is not provided, the most recent page is implied.
// If `id` is a string, it is treated as a fully qualified URL associated with `source`.
// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
// Use this function without `id` to scroll to a new location of an opened page.
type open = (_: {
id?: number | string, // default: -1
cursor?: number, // default: -1
loc?: number, // default: -1
num_lines?: number, // default: -1
view_source?: boolean, // default: false
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserFind }}

// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.
type find = (_: {
pattern: string,
cursor?: number, // default: -1
}) => any;
{{- end }}

} // namespace browser
{{- end }}{{/* end if has browser tools */}}
{{- if $hasPython }}

## python

Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.
{{- end }}{{/* end if hasPython */}}
{{- end }}{{/* end if has any built-in tools */}}
{{- end }}{{/* end if .Tools */}}

# Valid channels: analysis, commentary, final. Channel must be included for every message.{{ if $hasNonBuiltinTools }}
Calls to these tools must go to the commentary channel: 'functions'.
{{- end -}}<|end|>{{/* end of system */ -}}
{{- if or $hasNonBuiltinTools .System -}}
<|start|>developer<|message|>{{- if $hasNonBuiltinTools }}# Tools

## functions

namespace functions {
{{- range .Tools }}
{{- if not (or (eq .Function.Name "browser.search") (eq .Function.Name "browser.open") (eq .Function.Name "browser.find") (eq .Function.Name "python")) }}
{{if .Function.Description }}
// {{ .Function.Description }}
{{- end }}
{{- if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0) }}
type {{ .Function.Name }} = (_: {
{{- range $name, $prop := .Function.Parameters.Properties }}
{{- if $prop.Description }}
  // {{ $prop.Description }}
{{- end }}
  {{ $name }}: {{ if gt (len $prop.Type) 1 }}{{ range $i, $t := $prop.Type }}{{ if $i }} | {{ end }}{{ $t }}{{ end }}{{ else }}{{ index $prop.Type 0 }}{{ end }},
{{- end }}
}) => any;
{{- else }}
type {{ .Function.Name }} = () => any;
{{- end }}
{{- end }}{{/* end if not browser tool */}}
{{- end }}{{/* end of range .Tools */}}

} // namespace functions
{{- end }}{{/* end if hasNonBuiltinTools */}}
{{- if .System}}

# Instructions

{{ .System }}
{{- end -}}
<|end|>
{{- end -}}
{{- /* Find the index of the last user message */ -}}
{{- $lastUserIdx := -1 }}
{{- $prefillingContent := false }}
{{- $prefillingThinkingOnly := false }}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if eq $msg.Role "user" }}
    {{- $lastUserIdx = $i }}
  {{- end -}}
  {{- if and $last (eq $msg.Role "assistant") (gt (len $msg.Content) 0) }}
    {{- $prefillingContent = true }}
  {{- else if and $last (eq $msg.Role "assistant") (gt (len $msg.Thinking) 0) }}
    {{- $prefillingThinkingOnly = true }}
  {{- end }}
{{- end -}}
{{- /* Now render messages */ -}}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if (ne $msg.Role "system") -}}
    {{- if eq $msg.Role "tool" -}}
      {{- if or (eq $msg.ToolName "python") (eq $msg.ToolName "browser.search") (eq $msg.ToolName "browser.open") (eq $msg.ToolName "browser.find") -}}
        <|start|>{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- else -}}
        <|start|>functions.{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- end -}}
    {{- else if eq $msg.Role "assistant" -}}
      {{- if and $msg.Thinking (gt $i $lastUserIdx) -}}{{- /* Show thinking only after last user message */ -}}
      <|start|>assistant<|channel|>analysis<|message|>{{ $msg.Thinking }}{{- if not $prefillingThinkingOnly -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.Content) 0 -}}
        <|start|>assistant<|channel|>final<|message|>{{ $msg.Content }}{{- if not $prefillingContent -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.ToolCalls) 0 -}}
        {{- range $j, $toolCall := $msg.ToolCalls -}}
          {{- $isBuiltin := or (eq $toolCall.Function.Name "python") (eq $toolCall.Function.Name "browser.search") (eq $toolCall.Function.Name "browser.open") (eq $toolCall.Function.Name "browser.find") -}}
          <|start|>assistant<|channel|>{{ if $isBuiltin }}analysis{{ else }}commentary{{ end }} to={{ if not $isBuiltin}}functions.{{end}}{{ $toolCall.Function.Name }} <|constrain|>json<|message|>{{ $toolCall.Function.Arguments }}<|call|>
        {{- end -}}
      {{- end -}}
    {{- else if eq $msg.Role "user" -}}
      <|start|>{{ $msg.Role }}<|message|>{{ $msg.Content }}<|end|>
    {{- end }}
  {{- else }}
  {{- end }}
{{- end -}}
{{- if not (or $prefillingContent $prefillingThinkingOnly) -}}
<|start|>assistant
{{- end -}}'''

gptoss_template_template_eos_token = "<|return|>"
CHAT_TEMPLATES["gpt-oss"] = (gptoss_template, gptoss_template_template_eos_token, False, gptoss_ollama,)
DEFAULT_SYSTEM_MESSAGE["gpt-oss"] = None # No system message in GPT-oss

CHAT_TEMPLATES["gptoss"] = (gptoss_template, gptoss_template_template_eos_token, False, gptoss_ollama,)
DEFAULT_SYSTEM_MESSAGE["gptoss"] = None # No system message in GPT-oss
pass

# =========================================== Qwen3-Instruct
qwen3_instruct_template = \
'''{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if reasoning_content %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}'''

# Ollama from https://ollama.com/library/qwen3/blobs/53e4ea15e8f5
qwen3_ollama = \
'''

{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
'''

qwen3_template_eos_token = "<|im_end|>"
CHAT_TEMPLATES["qwen3-instruct"] = (qwen3_instruct_template, qwen3_template_eos_token, False, qwen3_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen3-instruct"] = None # No system message in Qwen3

pass

# =========================================== Qwen3-Thinking
qwen3_thinking_template = \
'''{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}'''

CHAT_TEMPLATES["qwen3-thinking"] = (qwen3_thinking_template, qwen3_template_eos_token, False, qwen3_ollama,)
DEFAULT_SYSTEM_MESSAGE["qwen3-thinking"] = None # No system message in Qwen3

pass

def _change_system_message(template: str, type_chat_template: str, system_message: str = None):
    system_message_pattern = r"\{system_message\}"
    
    # For predefined templates, check if default system message exists
    default_system_message = DEFAULT_SYSTEM_MESSAGE.get(f"{type_chat_template}", None)
    if default_system_message is None:
        if system_message is not None:
            logger.warning_once(
                f"Unsloth: You tried to change the system message for {type_chat_template}, "
                "but it doesn't have a default system message. "
                "You need to manually add the system message in your data."
            )
        return template, system_message
    pass
    
    # For custom templates
    if type_chat_template is None:
        has_placeholder = re.search(system_message_pattern, template) is not None
        
        if has_placeholder:
            if system_message is None:
                raise ValueError("Unsloth: You need to provide a system message for custom templates.")
            new_template = re.sub(system_message_pattern, system_message, template)
            return new_template, system_message
        
        return template, system_message
    pass
        
    # For predefined templates with default system message
    message_to_use = system_message if system_message is not None else default_system_message
    new_template = re.sub(system_message_pattern, message_to_use, template)
    
    return new_template, message_to_use
pass


def get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
    map_eos_token = True,
    system_message = None,
):
    assert(type(map_eos_token) is bool)
    old_tokenizer = tokenizer

    IS_GEMMA = False
    if tokenizer.__class__.__name__.startswith("Gemma"):
        if chat_template == "chatml": chat_template = "gemma_chatml"
        IS_GEMMA = True
    pass

    # We add a check for Llama-3
    # if chat_template == "llama-3":
    #     tokenizer._using_llama3_template = True
    # else:
    #     llama3_tokens = set(["<|end_header_id|>", "<|eot_id|>", "<|start_header_id|>"])
    #     check_llama3_tokens = llama3_tokens & set(str(x) for x in tokenizer.added_tokens_decoder.values())
    #     if len(check_llama3_tokens) == len(llama3_tokens):
    #         tokenizer._using_llama3_template = True
    #     pass
    # pass

    # We first check if the tokenizer is a fast one. If not, we cannot convert this!
    is_fast_tokenizer = getattr(tokenizer, "is_fast", False)
    old_padding_side = tokenizer.padding_side

    same_padding_token = False
    type_chat_template = None
    
    if type(chat_template) in (list, tuple,):
        # For changing system message later
        # Since it's not supported yet, we will raise an error first!
        type_chat_template = chat_template[0].lower()
        chat_template, stop_word = chat_template
        assert(type(chat_template) is str)
        assert(type(stop_word) is str)
        ollama_modelfile = None

    elif type(chat_template) is str:
        # For changing system message later
        type_chat_template = chat_template.lower()

        chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template]

        # Check mapping to eos_token
        if not map_eos_token and yes_map_eos_token: map_eos_token = True
        if not yes_map_eos_token and map_eos_token: map_eos_token = False

        if type(stop_word) in (list, tuple,):
            token_mapping, stop_word = stop_word
            assert(type(token_mapping) is dict)
        else:
            token_mapping = None

        assert(type(stop_word) is str)

        # Check fast tokenizer
        if not is_fast_tokenizer:
            pass
            # print(
            #     "Unsloth: Not a fast tokenizer, so can't process it as of yet :(\n"\
            #     "Please log a Github issue if you want this as a new feature!\n"\
            #     "Your chat template will still work, but it won't add or edit tokens."
            # )

        elif token_mapping is not None:
            # token_mapping = {"<start_of_turn>" : "<|im_start|>", "<end_of_turn>" : "<|im_end|>"}
            # For Gemma :)

            string_vocab = tokenizer._tokenizer.to_str()

            skipped = 0
            for old_token, new_token in token_mapping.items():
                old_count = string_vocab.count(f'"{old_token}"')
                new_count = string_vocab.count(f'"{new_token}"')
                if new_count != 0:
                    print(f"{new_token} is already a token. Skipping.")
                    skipped += 1
                elif old_count == 0:
                    raise RuntimeError(f"{old_token} was not part of the tokenizer!")
                else:
                    string_vocab = string_vocab.replace(f'"{old_token}"', f'"{new_token}"')
                pass
            pass

            if map_eos_token and (not stop_word in token_mapping.values()):
                # Do not map 107 = <|im_end|> and 1 = <|im_end|>. This will reduce the vocab size by 1
                logger.warning_once(f"Unsloth: Will map {stop_word} to EOS = {tokenizer.eos_token}.")
                string_vocab = string_vocab.replace(tokenizer.eos_token, stop_word)
            pass

            if skipped != len(token_mapping):
                new_tokenizer = tokenizer._tokenizer.from_str(string_vocab)

                # Careful on pad_token
                old_pad_token = tokenizer.pad_token
                if old_pad_token == tokenizer.eos_token:
                    old_pad_token = stop_word
                    same_padding_token = True
                pass

                if map_eos_token:
                    new_tokenizer = tokenizer.__class__(
                        tokenizer_object = new_tokenizer,
                        eos_token = stop_word,
                        pad_token = old_pad_token,
                    )
                else:
                    new_tokenizer = tokenizer.__class__(
                        tokenizer_object = new_tokenizer,
                        pad_token = old_pad_token,
                    )
                pass

                # Must fix the sentence piece tokenizer since there's no tokenizer.model file!
                tokenizer = fix_sentencepiece_tokenizer(tokenizer, new_tokenizer, token_mapping,)
            else:
                pass

        elif map_eos_token and (stop_word != "eos_token"):
            logger.warning_once(f"Unsloth: Will map {stop_word} to EOS = {tokenizer.eos_token}.")

            # Replaces the old EOS token with a new one.
            # Useful for ChatML <|im_end|> for example.
            # Usually we train 2 more tokens <|im_start|> and <|im_end|>
            # But training the lm_head and embeddings are slow!
            # This is a HACK!
            # Idea from https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser

            old_bos_token = getattr(tokenizer, "bos_token", None)
            old_eos_token = getattr(tokenizer, "eos_token", None)
            old_pad_token = getattr(tokenizer, "pad_token", None)
            old_unk_token = getattr(tokenizer, "unk_token", None)

            string_vocab = tokenizer._tokenizer.to_str()
            # First check if new stop_word is in the tokenizer
            if stop_word in string_vocab:
                # We shall swap them around
                temporary_stop_token = "<|:__TEMP//STOP//TOKEN__:|>"
                string_vocab = string_vocab.replace(old_eos_token, temporary_stop_token)
                string_vocab = string_vocab.replace(stop_word, old_eos_token)
                string_vocab = string_vocab.replace(temporary_stop_token, stop_word)
            else:
                string_vocab = string_vocab.replace(old_eos_token, stop_word)
            pass
            new_tokenizer = tokenizer._tokenizer.from_str(string_vocab)

            # Careful on pad_token
            if old_pad_token == old_eos_token:
                old_pad_token = stop_word
                same_padding_token = True
            pass

            new_tokenizer = tokenizer.__class__(
                tokenizer_object = new_tokenizer,
                bos_token = old_bos_token,
                eos_token = stop_word,
                unk_token = old_unk_token,
                pad_token = old_pad_token,
            )

            # Must fix the sentence piece tokenizer since there's no tokenizer.model file!
            token_mapping = { old_eos_token : stop_word, }
            tokenizer = fix_sentencepiece_tokenizer(tokenizer, new_tokenizer, token_mapping,)
        pass

    else:
        raise TypeError(
            f"Unsloth: `chat_template` must be a tuple of (your_template, eos_token,) or one of\n"\
            f"{CHAT_TEMPLATES.keys()}"
        )
    pass

    # Careful on Gemma
    # bos_token is a must or else losses become too high
    if IS_GEMMA and not chat_template.startswith(("{{ bos_token }}", "{{- bos_token }}")):
        chat_template = "{{ bos_token }}" + chat_template
    pass

    # For ShareGPT role -> from and content -> value
    new_chat_template = chat_template\
        .replace("'role'",      "'" + mapping["role"]      + "'")\
        .replace("'content'",   "'" + mapping["content"]   + "'")\
        .replace("'user'",      "'" + mapping["user"]      + "'")\
        .replace("'assistant'", "'" + mapping["assistant"] + "'")

    _, tokenizer = patch_tokenizer(model = None, tokenizer = tokenizer)
    tokenizer.padding_side = old_padding_side

    # If not normal HF, we add a check to make old templates work
    if mapping != {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}:
        chat_template = \
            "{% if 'role' in messages[0] %}" + \
            chat_template + \
            "{% else %}" + \
            new_chat_template + \
            "{% endif %}"
    else:
        chat_template = new_chat_template
    pass

    chat_template, system_message = _change_system_message(chat_template, type_chat_template, system_message)

    tokenizer.chat_template = chat_template

    # Also fix up other tokens
    old_pad_token = getattr(old_tokenizer, "pad_token", None)
    old_bos_token = getattr(old_tokenizer, "bos_token", None)
    old_unk_token = getattr(old_tokenizer, "unk_token", None)
    new_pad_token = getattr(tokenizer,     "pad_token", None)
    new_bos_token = getattr(tokenizer,     "bos_token", None)
    new_unk_token = getattr(tokenizer,     "unk_token", None)
    if old_bos_token != new_bos_token: tokenizer.bos_token = old_bos_token
    if old_unk_token != new_unk_token: tokenizer.unk_token = old_unk_token
    if not same_padding_token:
        if old_pad_token != new_pad_token: tokenizer.pad_token = old_pad_token
    pass

    # stopping_criteria = create_stopping_criteria(tokenizer, stop_word)

    # Patch saving functions
    tokenizer = patch_saving_functions(tokenizer)

    # Add Ollama
    tokenizer._ollama_modelfile = ollama_modelfile
    tokenizer._system_message   = system_message
    return tokenizer#, stopping_criteria
pass


def remove_special_tokens(tokenizer, prompt):
    # Removes double BOS token
    if prompt.startswith(tokenizer.bos_token):
        prompt = prompt[len(tokenizer.bos_token):]
    pass
    return prompt
pass


def _parse_combined_prompt(combined_prompt, dataset):
    # Find {...}
    possible_columns = re.findall(r"\{(.+?)\}", combined_prompt)
    dataset_columns = set(dataset.column_names)
    for column in possible_columns:
        if column not in dataset_columns:
            raise KeyError(
                f"Unsloth: Your prompt includes '{column}' but this does not exist in the dataset. "\
                f"Only allowed columns are {list(dataset_columns)}"
            )
        pass
    pass

    # Find [[...]]
    optional_prompts = list(re.finditer(r"\[\[.+?\]\]", combined_prompt, flags = re.DOTALL | re.MULTILINE))
    optional_prompts = [(x.span(), x.group(0)) for x in optional_prompts]

    final_optional_prompts = []
    if len(optional_prompts) != 0:
        # Add left
        left = optional_prompts[0]
        l = left[0][0]
        if l != 0: final_optional_prompts.append(combined_prompt[:l])

        # Add in between
        for left, right in zip(optional_prompts[:-1], optional_prompts[1:]):
            l, r = left[0][-1], right[0][0]
            final_optional_prompts.append(left)
            if l != r: final_optional_prompts.append(combined_prompt[l : r])
        pass
        final_optional_prompts.append(optional_prompts[-1])

        # Add right
        right = optional_prompts[-1]
        r = right[0][1]
        if r != len(combined_prompt): final_optional_prompts.append(combined_prompt[r:])
    else:
        # Just add in the entire string
        final_optional_prompts.append(combined_prompt)
    pass

    check_combined = "".join(x if type(x) is str else x[1] for x in final_optional_prompts)
    assert(combined_prompt == check_combined)

    return possible_columns, final_optional_prompts
pass


def _create_formatter(possible_columns, final_optional_prompts, user_column_name):
    # Start final prompt!
    function = ["def __combined_prompt_processor__(examples):"]
    columns = list(set(possible_columns))
    for column in columns:
        function.append(f"{' '*4}{column}__ = examples['{column}']")
    function.append(f"{' '*4}texts = []")
    function.append(f"{' '*4}for ({', '.join(columns)}) in zip({', '.join(f'{x}__' for x in columns)}):")

    # Add optional tags as well!
    final_prompt = ""
    formatter = []

    for j, optional_prompt in enumerate(final_optional_prompts):
        if type(optional_prompt) is str:
            columns = re.findall(r"\{(.+?)\}", optional_prompt)
            formatter += columns
            # Must escape \n \r
            final_prompt += optional_prompt.encode("unicode-escape").decode("utf-8").replace("'", "\\'").replace('"', '\\"')
        else:
            where, prompt = optional_prompt
            # Strip [[...]]
            # Must escape \n \r
            prompt = prompt[2:-2].encode("unicode-escape").decode("utf-8").replace("'", "\\'").replace('"', '\\"')
            columns = re.findall(r"\{(.+?)\}", prompt)
            x = f"__optional_{j}__"
            prompt = f"{' '*8}{x} = '{prompt}'.format({', '.join(f'{x} = {x}' for x in columns)}) if {columns[0]} else ''"
            function.append(prompt)
            formatter.append(x)
            final_prompt += "{" + x + "}"
        pass
    pass

    function.insert(1, f"{' '*4}__combined_prompt__ = '{final_prompt}'")
    function.append(f"{' '*8}texts.append("\
                    f"__combined_prompt__.format({', '.join(f'{x} = {x}' for x in formatter)}))")
    function.append(f"{' '*4}return " + "{ " + f"'{user_column_name}' : texts" + " }")
    return "\n".join(function)
pass


def to_sharegpt(
    dataset,
    merged_prompt = "",
    merged_column_name = "instruction",
    output_column_name = "output",
    remove_unused_columns = True,
    conversation_extension = 1,
    random_state = 3407,
):
    """
    Converts a dataset to ShareGPT style.
    ShareGPT requires only 1 input and 1 output field.
    This means one has to merge multiple columns into 1 for 1 input field.
    Use `conversation_extension` to increase the length of each conversation by randomnly
    selecting a few and packing them into 1.

    merged_prompt = "",                 Prompt to merge columns into 1 input
    merged_column_name = "instruction", Final column name for the input  field
    output_column_name = "output",      Final column name for the output field
    remove_unused_columns = True,
    conversation_extension = 1,         Automatically combines `conversation_extension` convos into 1
    random_state = 3407,
    """
    if "conversations" in dataset.column_names:
        convo = dataset[0]["conversations"]
        if type(convo) is list:
            raise TypeError("Unsloth: Your dataset is probably already in ShareGPT format!")
        pass
    pass

    possible_columns, final_optional_prompts = _parse_combined_prompt(merged_prompt, dataset)
    function = _create_formatter(possible_columns, final_optional_prompts, merged_column_name)
    exec(function, globals())
    dataset = dataset.map(__combined_prompt_processor__, batched = True, desc = "Merging columns")

    def __convert_to_sharegpt__(examples):
        users      = examples[merged_column_name]
        assistants = examples[output_column_name]
        texts = [
            [
                {"from" : "human", "value" : str(user)     },
                {"from" : "gpt",   "value" : str(assistant)},
            ] \
            for user, assistant in zip(users, assistants)
        ]
        return { "conversations" : texts, }
    pass

    dataset = dataset.map(
        __convert_to_sharegpt__,
        batched = True,
        desc = "Converting to ShareGPT",
        # Remove unused columns!
        remove_columns = dataset.column_names if remove_unused_columns else None,
    )

    # Randomnly concat conversations to create a long stream!
    from datasets import concatenate_datasets
    n_extensions = max(conversation_extension-1, 0)
    if n_extensions == 0: return dataset

    dataset = dataset.rename_columns({"conversations" : "conversations0"})
    all_shuffled = [dataset]
    for j in range(1, n_extensions+1):
        shuffled = dataset.shuffle(seed = random_state+j).rename_columns({"conversations0" : f"conversations{j}"})
        all_shuffled.append(shuffled)
    pass
    dataset = concatenate_datasets(all_shuffled, axis = 1)

    # Combine them into 1
    function = "def __combine_conversations__(examples):\n"
    n_extensions += 1
    for j in range(n_extensions):
        function += f"{' '*4}conversations{j}__ = examples['conversations{j}']\n"
    function += f"{' '*4}convos = []\n"
    function += f"{' '*4}for ({', '.join(f'conversations{j}' for j in range(n_extensions))}) "\
                f"in zip({', '.join(f'conversations{j}__' for j in range(n_extensions))}):\n"
    function += f"{' '*8}convos.append("\
                f"{'+'.join(f'conversations{j}' for j in range(n_extensions))})\n"
    function += f"{' '*4}return " + "{ " + "'conversations' : convos" + " }"

    # Map function
    exec(function, globals())
    dataset = dataset.map(
        __combine_conversations__,
        batched = True,
        desc = "Extending conversations",
        # Remove unused columns!
        remove_columns = dataset.column_names if remove_unused_columns else None,
    )
    return dataset
pass


def get_ollama_eos_tokens(tokenizer, extra_eos_tokens = []):
    added_tokens_decoder = tokenizer.added_tokens_decoder.values()
    added_tokens_decoder = [str(x) for x in added_tokens_decoder]

    # Remove added_tokens_decoder duplicates
    added_tokens_decoder = list(set(added_tokens_decoder) - set(extra_eos_tokens))

    # Remove BOS
    if getattr(tokenizer, "bos_token", None) is not None:
        added_tokens_decoder = [x for x in added_tokens_decoder if x != tokenizer.bos_token]
    pass

    repeatted_tokens = []
    # Join all vocab
    joined_text = "\x01\x00".join(added_tokens_decoder)
    for token in added_tokens_decoder:
        n = len(token)
        repeatted_counts = joined_text.count(token[:n//2])
        # Try finding longer than 1/2 of the token in the rest
        # For eg <|reserved_special_token_0|>, <|reserved_special_token_1|>
        if repeatted_counts > 2:
            for j in range(n//2+1, n):
                if joined_text.count(token[:j]) < repeatted_counts:
                    j -= 1
                    # Remove repeatted tokens to reduce search space
                    joined_text = joined_text.replace(token[:j], "")
                    repeatted_tokens.append(token[:j])
                    break
            pass
        pass
    pass

    # Remove duplicates
    splitted = joined_text.split("\x01\x00")
    final_eos_tokens = [old for old, new in zip(added_tokens_decoder, splitted) if old == new]
    final_eos_tokens += extra_eos_tokens
    final_eos_tokens += repeatted_tokens

    # Remove new lines, spaces and HTML tags
    filtered_eos_tokens = []
    for token in final_eos_tokens:
        if   token.count("\n") == len(token): continue
        elif token.count("▁") == len(token): continue
        elif token.startswith("<") and len(token) <= 2: continue
        elif token.startswith("</") and len(token) == 3: continue
        filtered_eos_tokens.append(token)
    pass
    return filtered_eos_tokens
pass


def construct_chat_template( \

tokenizer = None,

chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>""",
    
default_system_message = \
    "Below are some instructions that describe some tasks. Write responses that appropriately complete each request.",

extra_eos_tokens = None,
):
    """
    Creates a Ollama modelfile and a HF Jinja template from a custom
    template. You must provide 2x examples of an input & output.
    There is an optional system message as well.

    You must use {INPUT}, {OUTPUT} twice, and {SYSTEM} is optional.
    """
    # Strip only the left
    chat_template = chat_template.lstrip()

    assert(tokenizer is not None)

    if extra_eos_tokens is None: extra_eos_tokens = []
    elif type(extra_eos_tokens) is str: extra_eos_tokens = [extra_eos_tokens,]

    vocab = tokenizer.get_vocab()
    for extra_eos in extra_eos_tokens:
        assert(type(extra_eos) is str)
        if extra_eos not in vocab:
            raise ValueError(f"Unsloth: `{extra_eos}` is not a singular token in the tokenizer.")
        pass
    pass

    error_msg = \
        "Unsloth: Your prompt template must have 2 examples showing the user input {INPUT} "\
        "and the assistant output {OUTPUT}\n\n"\
        "For example what is not allowed is just:\n"\
        "### Input:\\n{INPUT}\\n\\n### Response:\\n{OUTPUT}\\n\n\n"\
        "What is required is 2x of this:\n"\
        "### Input:\\n{INPUT}\\n\\n### Response:\\n{OUTPUT}\\n"\
        "### Input:\\n{INPUT}\\n\\n### Response:\\n{OUTPUT}\\n"

    # Check for EOS after {OUTPUT}
    if tokenizer.eos_token is not None:
        extra_eos_tokens.insert(0, tokenizer.eos_token)
    if len(extra_eos_tokens) == 0:
        raise RuntimeError(
            "Unsloth: Your tokenizer does not have an EOS token? Please provide one via extra_eos_tokens!"
        )
    pass

    # Check tokenizer types
    tokenizer_name = tokenizer.name_or_path.lower()
    if tokenizer_name.startswith(("unsloth/llama-3-8b-instruct", "unsloth/llama-3-70b-instruct")):
        # Add <|eot_id|>
        extra_eos_tokens.append("<|eot_id|>")
    elif ("<|eot_id|>" in extra_eos_tokens or "<|eot_id|>" in chat_template) and \
        tokenizer_name.startswith(("unsloth/llama-3-8b", "unsloth/llama-3-70b")):
        # Warn
        logger.warning(
            "Unsloth: Base llama-3 models did not train <|eot_id|>.\n"\
            "Please use the instruct version or use <|end_of_text|>"
        )
    pass
    extra_eos_tokens = list(set(extra_eos_tokens))

    count_eos = 0
    for eos in extra_eos_tokens:
        count_eos += len(re.findall(r"{OUTPUT}" + re.escape(eos), chat_template))
    pass

    # This forces you to provide 2 input and outputs
    final_combined_check = False

    try:
        # O(N^2) search finding 2 repeatted pieces of text
        j = len(chat_template)-1
        at_least_one = False
        while j > 0:
            found = chat_template.rfind(chat_template[j:], 0, j)
            if found == -1: break
            j -= 1
            at_least_one = True
        pass
        if j > 0: j += 1
        else: raise RuntimeError(error_msg)

        if not at_least_one: raise RuntimeError(error_msg)

        # Must be equivalent to left
        final_combined_check = True

        # Repeatted text
        instruction_response = chat_template[j:]
        if instruction_response.count("{INPUT}") != 1 or instruction_response.count("{OUTPUT}") != 1:
            raise RuntimeError(error_msg)
        pass

        # 1st System, Instruction, Output pair
        left  = chat_template[:j]
        # 2nd Instruction, Output pair
        right = chat_template[j:]

        final_combined_check = left if final_combined_check else chat_template

        # Isolate input
        extra_eos_tokens_regex = "|".join(f"(?:{re.escape(x)})" for x in extra_eos_tokens)
        if len(extra_eos_tokens_regex) != 0:
            find_end = f"(?:{extra_eos_tokens_regex})?"
        else:
            find_end = ""
        find_end = r"\{INPUT\}[\s\n]{0,}" + find_end
        input_end = list(re.finditer(find_end, right))
        assert(len(input_end) == 1)
        input_end = input_end[0]
        input_end = input_end.span(0)[1]
        input_part = right[:input_end]

        # Isolate output
        output_part = right[input_end:]

        # Isolate system
        where_system = left.find(input_part)
        system_part = left[:where_system if where_system != -1 else len(left)]

        # Check if the user provided a correct prompt
        combined = system_part + input_part + output_part
        if combined != final_combined_check:
            combined_changed = combined            .replace('\n', '\\n')
            left_changed     = final_combined_check.replace('\n', '\\n')
            raise RuntimeError(
                "Unsloth: The prompt template you provided isn't correct. You gave:\n"\
                f"{combined_changed}\n\n"\
                "But we require the following:\n"\
                f"{left_changed}"
            )
        pass
    except:
        ending = chat_template[chat_template.find("{OUTPUT}") + len("{OUTPUT}"):]

        ending = re.escape(ending)
        find_text = "{INPUT}" + ending + "(.+?{OUTPUT}" + ending + ")"
        response_part = re.findall(find_text, chat_template, flags = re.DOTALL | re.MULTILINE)
        response_part = response_part[0]

        for j in range(1, len(response_part)):
            try_find = re.escape(response_part[:j])
            try: found = next(re.finditer("(" + try_find + ").+?\\{INPUT\\}", chat_template, flags = re.DOTALL | re.MULTILINE))
            except: break
        pass
        separator = found.group(1)

        response_start = chat_template.find(response_part)
        start_instruction = chat_template[:response_start].rfind(separator)
        if start_instruction == -1: start_instruction = 0
        instruction_part = chat_template[start_instruction:response_start]

        combined = instruction_part + response_part
        where = chat_template.find(combined)
        system_part = chat_template[:where]

        system_part, input_part, output_part = system_part, instruction_part, response_part
    pass

    if count_eos == 0:
        logger.warning("Unsloth: We automatically added an EOS token to stop endless generations.")
        eos = extra_eos_tokens[0]
        output_part = output_part + eos
    pass

    # Ollama modelfile parts

    # Check bos_token is in system prompt
    ollama_system = system_part
    has_bos_token = False
    always_bos_token = False
    if tokenizer("A").input_ids[0] == getattr(tokenizer, "bos_token_id", None):
        always_bos_token = True
        if ollama_system.startswith(tokenizer.bos_token):
            has_bos_token = True
            ollama_system = ollama_system[len(tokenizer.bos_token):]
        pass
    pass
    # Check system
    if "{SYSTEM}" in ollama_system:
        system_modelfile = "{{ if .System }}" + ollama_system.replace("{SYSTEM}", "{{ .System }}") + "{{ end }}"
    else:
        system_modelfile = ollama_system
    pass
    input_modelfile  = "{{ if .Prompt }}" + input_part .replace("{INPUT}",  "{{ .Prompt }}") + "{{ end }}"
    output_modelfile = output_part.replace("{OUTPUT}", "{{ .Response }}")

    # Ollama EOS
    ollama_eos = get_ollama_eos_tokens(tokenizer, extra_eos_tokens)
    ollama_eos = '\n'.join(f'PARAMETER stop "{eos}"' for eos in ollama_eos)

    # Add temperature and min_p to counteract gibberish
    ollama_eos += "\nPARAMETER temperature 1.5\nPARAMETER min_p 0.1"

    # Ollama modelfile
    part = '"""'
    modelfile = 'FROM {__FILE_LOCATION__}\n\n'\
    'TEMPLATE ' + part + system_modelfile + input_modelfile + output_modelfile + \
        part + '\n\n' + ollama_eos

    # HF Jinja Chat template
    def process(part, which, content = "message['content']"):
        if part.endswith(which):
            part = "'" + part[:part.find(which)] + f"' + {content}"
        elif part.startswith(which):
            part = f"{content} + '" + part[part.find(which):] + "'"
        else:
            part = "'" + part.replace(which, f"' + {content} + '") + "'"
        if part.startswith("'' + "): part = part[5:]
        return part
    pass
    input_jinja  = process(input_part,  "{INPUT}")
    output_jinja = process(output_part, "{OUTPUT}")
    pass

    jinja_template = \
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ " + input_jinja + " }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ " + output_jinja + " }}"\
            "{% else %}"\
                "{{ raise_exception('Only user and assistant roles are supported!') }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}"\
            "{{ '" + output_part[:output_part.find("{OUTPUT}")] + "' }}"\
        "{% endif %}"
    pass

    # Now add system prompt to jinja
    if len(system_part) != 0:
        partial_system = process(system_part, "{SYSTEM}", "messages[0]['content']")
        partial_system = partial_system.replace("{SYSTEM}", "")

        if "{SYSTEM}" in partial_system:
            if default_system_message is None:
                raise RuntimeError("Unsloth: Please specify a default system message!")
        pass

        # Separate the BOS
        if has_bos_token:
            partial_system = partial_system.replace(tokenizer.bos_token, "", 1)
            system_part    = system_part   .replace(tokenizer.bos_token, "", 1)
        pass
        
        partial_system = \
            "{% if messages[0]['role'] == 'system' %}"\
                "{{ " + partial_system + " }}"\
                "{% set loop_messages = messages[1:] %}"
        if default_system_message is not None:
            full_system = system_part.replace("{SYSTEM}", default_system_message)
            if "{SYSTEM}" in system_part:
                modelfile += '\nSYSTEM "' + default_system_message + '"'
            pass
            partial_system += "{% else %}"\
                "{{ '" + full_system + "' }}"\
                "{% set loop_messages = messages %}"\
            "{% endif %}"
        else:
            partial_system += "{% endif %}"
        pass

        jinja_template = partial_system + jinja_template

        if has_bos_token:
            jinja_template = "{{ bos_token }}" + jinja_template
    pass

    # Fix missing loop_messages
    if "{% set loop_messages = messages %}" not in jinja_template:
        jinja_template = jinja_template.replace(
            "{% for message in loop_messages %}",
            "{% for message in messages %}",
            1, # Only replace the first one
        )
    pass

    # Check if system part is the same!
    jinja_template = re.sub(
        r"\{\% if messages\[0\]\['role'\] \=\= 'system' \%\}\{\{ '(.+?)' \}\}"\
        r"\{\% set loop\_messages \= messages\[1\:\] \%\}"\
        r"\{\% else \%\}\{\{ '\1' \}\}\{\% set loop\_messages \= messages \%\}\{\% endif \%\}"\
        r"\{\% for message in loop\_messages \%\}",
        r"{{ '\1' }}{% for message in messages %}",
        jinja_template, flags = re.MULTILINE | re.DOTALL,
    )

    # Check jinja template for bos
    if always_bos_token:
        if not jinja_template.startswith(("{{ bos_token }}", "{{- bos_token }}")):
            jinja_template = "{{ bos_token }}" + jinja_template
    pass

    # Get instruction and output parts for train_on_inputs = False
    input_part  = input_part [:input_part .find("{INPUT}")]
    output_part = output_part[:output_part.find("{OUTPUT}")]
    return modelfile, jinja_template, input_part, output_part
pass


def test_construct_chat_template():
    token = "hf_"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token = token)

    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
    
    default_system_message = \
        "Below are some instructions that describe some tasks. Write responses that appropriately complete each request."
      
    extra_eos_tokens = None

    modelfile, jinja_template, _, _ = construct_chat_template(
        tokenizer = tokenizer,
        chat_template = chat_template,
        extra_eos_tokens = extra_eos_tokens,
    )

    messages = [
        {"role": "system", "content": "You are an assistant"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."},
        {"role": "user", "content": "Ok!"},
        {"role": "assistant", "content": "Anything else?"},
        {"role": "user", "content": "What's 2x2?"},
    ]
    correct_output = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

    tokenizer.chat_template = jinja_template
    new_output = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    assert(correct_output == new_output)
    pass
pass


def apply_chat_template( \

dataset,
tokenizer = None,

chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>""",
    
default_system_message = \
    "Below are some instructions that describe some tasks. Write responses that appropriately complete each request.",
  
extra_eos_tokens = None,
  
):
    """
    Creates a Ollama modelfile and a HF Jinja template from a custom
    template. You must provide 2x examples of an input & output.
    There is an optional system message as well.

    You must use {INPUT}, {OUTPUT} twice, and {SYSTEM} is optional.
    """
    modelfile, jinja_template, input_part, output_part = construct_chat_template(
        tokenizer = tokenizer,
        chat_template = chat_template,
        default_system_message = default_system_message,
        extra_eos_tokens = extra_eos_tokens,
    )
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass

    tokenizer.chat_template = jinja_template
    tokenizer._ollama_modelfile = modelfile
    tokenizer._unsloth_input_part  = input_part
    tokenizer._unsloth_output_part = output_part
    if hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer.chat_template = jinja_template
        tokenizer.tokenizer._ollama_modelfile = modelfile
        tokenizer.tokenizer._unsloth_input_part  = input_part
        tokenizer.tokenizer._unsloth_output_part = output_part

    return dataset.map(formatting_prompts_func, batched = True,)
pass


def create_stopping_criteria(tokenizer, stop_word = "eos_token"):
    class StoppingCriteriaSub(StoppingCriteria):
        __slots__ = "stop_token", "single_match", "length",

        def __init__(self, stops = "eos_token", device = "cuda", encounters = 1):
            super().__init__()
            if stops == "eos_token":
                self.stop_token = torch.tensor(tokenizer.eos_token_id, device = "cuda")
                self.length = 1
            else:
                self.stop_token = tokenizer(["\n" + stops], add_special_tokens = False, return_tensors = "pt")
                self.stop_token = self.stop_token.input_ids.ravel()[1:].to("cuda")
                self.length = self.stop_token.shape[0]
            pass
            self.single_match = self.length == 1
        pass

        def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> bool:
            input_ids = input_ids.ravel()
            last_token = input_ids[-1]
            if self.single_match and (last_token == self.stop_token): return True

            if input_ids.shape[0] >= self.length and \
                (input_ids[-self.length:] == self.stop_token).all(): return True
            return False
        pass
    pass
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops = stop_word)])
    return stopping_criteria
pass


def test_chat_templates():
    messages = [
        {"role": "system","content": " You are a friendly chatbot.",},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."},
        {"role": "user", "content": "  But 2+2 is equal to 5. "},
        {"role": "assistant", "content": "No I'm sure its 4."},
        {"role": "user", "content": "  No it's 100% 5! "},
    ]

    # Zephyr
    from transformers import AutoTokenizer
    template = zephyr_template
    correct_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    correct_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    # Chatml
    template = chatml_template
    correct_tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    correct_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    # Mistral
    template = mistral_template
    correct_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    correct_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    # Llama
    template = llama_template
    correct_tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b-chat")
    correct_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    # Vicuna
    try:
        from fastchat.conversation import get_conv_template
    except:
        os.system("pip -qqq install git+https://github.com/lm-sys/FastChat.git")
        from fastchat.conversation import get_conv_template
    correct_prompt = get_conv_template("vicuna_v1.1")
    for j in range(len(messages)-1):
        correct_prompt.append_message(correct_prompt.roles[j%2==1], messages[j+1]["content"])
    correct_prompt.append_message(correct_prompt.roles[1], "")
    correct_prompt = tokenizer.bos_token + correct_prompt.get_prompt()

    template = vicuna_template
    correct_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    try:
        from fastchat.conversation import get_conv_template
    except:
        os.system("pip -qqq install git+https://github.com/lm-sys/FastChat.git")
        from fastchat.conversation import get_conv_template
    correct_prompt = get_conv_template("zero_shot")
    for j in range(len(messages)-1):
        correct_prompt.append_message(correct_prompt.roles[j%2==1], messages[j+1]["content"])
    correct_prompt.append_message(correct_prompt.roles[1], "")
    correct_prompt = tokenizer.bos_token + correct_prompt.get_prompt()

    template = vicuna_old_template
    correct_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    # We add </s> ourselves
    assert(correct_prompt == our_prompt.replace("</s>", ""))

    # Gemma
    correct_tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-7b-it")
    correct_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = gemma_template
    our_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    assert(our_prompt == correct_prompt)

    # Llama-3
    template = llama3_template
    correct_tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
    correct_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)

    # Phi-3
    template = phi3_template
    correct_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    correct_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    correct_tokenizer.chat_template = template
    our_prompt = correct_tokenizer.apply_chat_template(messages[1:], tokenize = False, add_generation_prompt = True)
    assert(correct_prompt == our_prompt)
pass


def test_hf_gguf_equivalence(tokenizer, gguf_model = "./model-unsloth.F16.gguf"):
    """
        Carefully checks the output of GGUF's tokenization and HF.
        Can catch all tokenization bugs.
    """
    import subprocess
    import re
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."},
        {"role": "user", "content": "  But 2+2 is equal to 5. "},
        {"role": "assistant", "content": "No I'm sure its 4."},
        {"role": "user", "content": "  No it's 100% 5! "},
    ]

    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}""".format(
        "Describe the city given eloquently.", # instruction
        "The lost city of Atlantis.", # input
        "", # output - leave this blank for generation!
    )
    prompts = [ prompt, ]

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        prompt = prompt.replace("'", "") # Subprocess does not like ''
        prompt = remove_special_tokens(tokenizer, prompt)
        prompts.append(prompt)
    pass
    
    for prompt in prompts:
        command = f"./llama.cpp/llama-cli -m {gguf_model} -n 0 --temp 0.0 --verbose-prompt "\
            f"--check-tensors -p '{prompt}'"

        datas = []
        with subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, bufsize = 1) as sp:
            for line in sp.stdout:
                datas.append(line.decode("utf-8", errors = "replace"))
        pass
        gguf_tokens = "".join(datas)

        # Now extract GGUF tokenization attempt
        gguf_tokenized = re.findall(r"([\d]{1,}) \-\> \'([^\']{1,})\'", gguf_tokens, flags = re.MULTILINE)
        gguf_tokenized = [(int(x[0]), x[1],) for x in gguf_tokenized]
        input_ids = tokenizer(prompt).input_ids

        tokens = tokenizer.batch_decode(input_ids)
        hf_tokenized = list(zip(input_ids, tokens))

        # Compare to Huggingface
        for j, (hf_token, gguf_token) in enumerate(zip(hf_tokenized, gguf_tokenized)):
            if (hf_token[0] != gguf_token[0]):
                print("Failed GGUF != HF at", j)
                print("HF =", hf_token)
                print("GGUF =", gguf_token)
                print(hf_tokenized)
                print()
                print(gguf_tokenized)
                print()
                raise RuntimeError("Failed comparing GGUF to HF.")
            pass
        pass
    return True
pass
