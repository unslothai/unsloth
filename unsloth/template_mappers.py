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
    "CHAT_TEMPLATES",
    "DEFAULT_SYSTEM_MESSAGE",
    "TEMPLATE_TO_MODEL_MAPPER",
    "MODEL_TO_TEMPLATE_MAPPER",
]

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
'''
FROM {__FILE_LOCATION__}
TEMPLATE """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
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
{{- end -}}"""
PARAMETER temperature 1.0
PARAMETER top_k 0
PARAMETER top_p 1.0
'''

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
FROM {__FILE_LOCATION__}
TEMPLATE """
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
"""
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

# =========================================== Liquid-LFM2
liquid_lfm2_template = \
'''
{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}'''

liquid_lfm2_template_eos_token = "<|im_end|>"
CHAT_TEMPLATES["lfm-2"] = (liquid_lfm2_template, liquid_lfm2_template_eos_token, False, None)
DEFAULT_SYSTEM_MESSAGE["lfm-2"] = None # No system message in Phi-3

pass

# =========================================== Starling-LM

starling_template = \
"""{{ bos_token }}
{%- for message in messages %}
    {{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ 'GPT4 Correct Assistant:' }}
{%- endif %}"""

# Ollama from https://ollama.com/library/starling-lm:7b/blobs/4b21bfc435b4
starling_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}GPT4 Correct System: {{ .System }}<|end_of_turn|>
{{ end }}{{ if .Prompt }}GPT4 Correct User: {{ .Prompt }}<|end_of_turn|>
{{ end }}GPT4 Correct Assistant: {{ .Response }}<|end_of_turn|>"""
PARAMETER stop "<|end_of_turn|>"
PARAMETER stop "GPT4 Correct User:"
PARAMETER stop "GPT4 Correct Assistant:"
PARAMETER stop "GPT4 Correct System:"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

starling_template_eos_token = "<|end_of_turn|>"
CHAT_TEMPLATES["starling"] = (starling_template, starling_template_eos_token, False, starling_ollama)
DEFAULT_SYSTEM_MESSAGE["starling"] = None

pass

# =========================================== Yi-chat

yi_chat_template = \
"""
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}
"""

# Ollama from https://ollama.com/library/yi:34b-chat/blobs/62fbfd9ed093
yi_chat_ollama = \
'''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
'''

yi_chat_template_eos_token = "<|endoftext|>"
CHAT_TEMPLATES["yi-chat"] = (yi_chat_template, yi_chat_template_eos_token, False, yi_chat_ollama)
DEFAULT_SYSTEM_MESSAGE["yi-chat"] = None

# =========================================== Yi-chat


TEMPLATE_TO_MODEL_MAPPER = {
    "phi-3.5": (
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ),
    "phi-3": (
        "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct-v0",
    ),
    "phi-4": (
        "unsloth/phi-4-unsloth-bnb-4bit",
        "unsloth/phi-4",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
        "unsloth/phi-4-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning",
        "microsoft/Phi-4-reasoning",
        "unsloth/phi-4-reasoning-bnb-4bit",
        "unsloth/phi-4-reasoning-plus-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning-plus",
        "microsoft/Phi-4-reasoning-plus",
        "unsloth/phi-4-reasoning-plus-bnb-4bit",
        "unsloth/phi-4-mini-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "unsloth/phi-4-mini-reasoning-bnb-4bit",
        "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
        "unsloth/Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-4-mini-instruct-bnb-4bit",
    ),
    "mistral": (
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b",
        "mistralai/Mistral-7B-v0.1",
        "unsloth/mistral-7b-instruct-v0.1-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "unsloth/mistral-7b-v0.2-bnb-4bit",
        "unsloth/mistral-7b-v0.2",
        "alpindale/Mistral-7B-v0.2-hf",
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-v0.3",
        "mistralai/Mistral-7B-v0.3",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/Mixtral-8x7B-v0.1-unsloth-bnb-4bit",
        "unsloth/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "unsloth/Mixtral-8x7B-v0.1-bnb-4bit",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit",
        "unsloth/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Base-2407",
        "unsloth/Mistral-Large-Instruct-2407-bnb-4bit",
        "mistralai/Mistral-Large-Instruct-2407",
        "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Small-Instruct-2409",
        "mistralai/Codestral-22B-v0.1",
        "mistral-community/Codestral-22B-v0.1",
        "unsloth/Mistral-Small-24B-Base-2501-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-24B-Base-2501",
        "mistralai/Mistral-Small-24B-Base-2501",
        "unsloth/Mistral-Small-24B-Base-2501-bnb-4bit",
        "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Base-2503-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Base-2503",
        "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "unsloth/Mistral-Small-3.1-24B-Base-2503-bnb-4bit",
        "unsloth/Devstral-Small-2505-unsloth-bnb-4bit",
        "unsloth/Devstral-Small-2505",
        "mistralai/Devstral-Small-2505",
        "unsloth/Devstral-Small-2505-bnb-4bit",
        "unsloth/Magistral-Small-2506-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2506",
        "mistralai/Magistral-Small-2506",
        "unsloth/Magistral-Small-2506-bnb-4bit",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
        "unsloth/Devstral-Small-2507-unsloth-bnb-4bit",
        "unsloth/Devstral-Small-2507",
        "mistralai/Devstral-Small-2507",
        "unsloth/Devstral-Small-2507-bnb-4bit",
        "unsloth/Magistral-Small-2507-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2507",
        "mistralai/Magistral-Small-2507",
        "unsloth/Magistral-Small-2507-bnb-4bit",
        "unsloth/Magistral-Small-2509-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2509",
        "mistralai/Magistral-Small-2509",
        "unsloth/Magistral-Small-2509-bnb-4bit",
        "unsloth/Pixtral-12B-2409-unsloth-bnb-4bit",
        "unsloth/Pixtral-12B-2409",
        "mistralai/Pixtral-12B-2409",
        "unsloth/Pixtral-12B-2409-bnb-4bit",
        "unsloth/Pixtral-12B-2409-Base-bnb-4bit",
        "unsloth/Pixtral-12B-Base-2409",
        "mistralai/Pixtral-12B-Base-2409",
    ),
    "llama": (
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-7b",
        "meta-llama/Llama-2-7b-hf",
        "unsloth/llama-2-13b-bnb-4bit",
        "unsloth/llama-2-13b",
        "meta-llama/Llama-2-13b-hf",
        "unsloth/llama-2-7b-chat-bnb-4bit",
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "llama3": (
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-8b",
        "meta-llama/Meta-Llama-3-8B",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "unsloth/llama-3-70b-bnb-4bit",
        "meta-llama/Meta-Llama-3-70B",
        "unsloth/llama-3-70b-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "llama-3.1": (
        "unsloth/Meta-Llama-3.1-8B-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-8B-unsloth-bnb-4bit",
        "unsloth/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B",
        "unsloth/Llama-3.1-8B-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B",
        "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "unsloth/Llama-3.1-Storm-8B-bnb-4bit",
        "unsloth/Llama-3.1-Storm-8B",
        "akjindal53244/Llama-3.1-Storm-8B",
        "unsloth/Hermes-3-Llama-3.1-8B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
        "unsloth/Hermes-3-Llama-3.1-405B-bnb-4bit",
        "NousResearch/Hermes-3-Llama-3.1-405B",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "unsloth/Llama-3.1-Tulu-3-8B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-8B",
        "allenai/Llama-3.1-Tulu-3-8B",
        "unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "llama-3.2": (
        "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "unsloth/Llama-3.2-1B-bnb-4bit",
        "unsloth/Llama-3.2-3B-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision",
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision",
        "meta-llama/Llama-3.2-90B-Vision",
    ),
    "llama-3.3": (
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
    "gemma": (
        "unsloth/gemma-7b-bnb-4bit",
        "unsloth/gemma-7b",
        "google/gemma-7b",
        "unsloth/gemma-2b-bnb-4bit",
        "unsloth/gemma-2b",
        "google/gemma-2b",
        "unsloth/gemma-7b-it-bnb-4bit",
        "unsloth/gemma-7b-it",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "unsloth/gemma-1.1-2b-it-bnb-4bit",
        "unsloth/gemma-1.1-2b-it",
        "google/gemma-1.1-2b-it",
        "unsloth/gemma-1.1-7b-it-bnb-4bit",
        "unsloth/gemma-1.1-7b-it",
        "google/gemma-1.1-7b-it",
    ),
    "gemma2": (
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-9b",
        "google/gemma-2-9b",
        "unsloth/gemma-2-27b-bnb-4bit",
        "unsloth/gemma-2-27b",
        "google/gemma-2-27b",
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "unsloth/gemma-2-9b-it",
        "google/gemma-2-9b-it",
        "unsloth/gemma-2-27b-it-bnb-4bit",
        "unsloth/gemma-2-27b-it",
        "google/gemma-2-27b-it",
        "unsloth/gemma-2-2b-bnb-4bit",
        "unsloth/gemma-2-2b",
        "google/gemma-2-2b",
        "unsloth/gemma-2-2b-it-bnb-4bit",
        "unsloth/gemma-2-2b-it",
        "google/gemma-2-2b-it",
    ),
    "gemma-3": (
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "unsloth/gemma-3-1b-it-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        "unsloth/gemma-3-12b-it-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-270m-it",
        "google/gemma-3-270m-it",
        "unsloth/gemma-3-270m-it-bnb-4bit",
        "unsloth/gemma-3-270m-unsloth-bnb-4bit",
        "unsloth/gemma-3-270m",
        "google/gemma-3-270m",
        "unsloth/gemma-3-270m-bnb-4bit",
        "unsloth/medgemma-4b-it-unsloth-bnb-4bit",
        "unsloth/medgemma-4b-it",
        "google/medgemma-4b-it",
        "unsloth/medgemma-4b-it-bnb-4bit",
        "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        "unsloth/medgemma-27b-text-it",
        "google/medgemma-27b-text-it",
        "unsloth/medgemma-27b-text-it-bnb-4bit",
    ),
    "gemma3n": (
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E4B-it",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it",
        "google/gemma-3n-E2B-it",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E4B",
        "google/gemma-3n-E4B",
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B",
        "google/gemma-3n-E2B",
        "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    ),
    "qwen2.5": (
        "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B",
        "unsloth/Qwen2.5-0.5B-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B",
        "unsloth/Qwen2.5-1.5B-bnb-4bit",
        "unsloth/Qwen2.5-3B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B",
        "unsloth/Qwen2.5-3B-bnb-4bit",
        "unsloth/Qwen2.5-7B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B",
        "unsloth/Qwen2.5-7B-bnb-4bit",
        "unsloth/Qwen2.5-14B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B",
        "unsloth/Qwen2.5-14B-bnb-4bit",
        "unsloth/Qwen2.5-32B-bnb-4bit",
        "unsloth/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B",
        "unsloth/Qwen2.5-72B-bnb-4bit",
        "unsloth/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B",
        "unsloth/Qwen2.5-Math-1.5B-bnb-4bit",
        "unsloth/Qwen2.5-Math-1.5B",
        "Qwen/Qwen2.5-Math-1.5B",
        "unsloth/Qwen2.5-Math-7B-bnb-4bit",
        "unsloth/Qwen2.5-Math-7B",
        "Qwen/Qwen2.5-Math-7B",
        "unsloth/Qwen2.5-Math-72B-bnb-4bit",
        "unsloth/Qwen2.5-Math-72B",
        "Qwen/Qwen2.5-Math-72B",
        "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "unsloth/Qwen2.5-Math-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-72B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
        "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-0.5B",
        "unsloth/Qwen2.5-Coder-1.5B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-1.5B",
        "unsloth/Qwen2.5-Coder-3B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-3B",
        "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-7B",
        "unsloth/Qwen2.5-Coder-14B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-14B",
        "Qwen/Qwen2.5-Coder-14B",
        "unsloth/Qwen2.5-Coder-32B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B",
        "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        "unsloth/OpenThinker-7B-unsloth-bnb-4bit",
        "unsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
        "unsloth/OpenThinker-7B-bnb-4bit",
    ),
    "qwen-2.5": (
        "unsloth/Qwen2-0.5B-bnb-4bit",
        "unsloth/Qwen2-0.5B",
        "Qwen/Qwen2-0.5B",
        "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
        "unsloth/Qwen2-1.5B-bnb-4bit",
        "unsloth/Qwen2-1.5B",
        "Qwen/Qwen2-1.5B",
        "unsloth/Qwen2-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "unsloth/Qwen2-7B-bnb-4bit",
        "unsloth/Qwen2-7B",
        "Qwen/Qwen2-7B",
        "unsloth/Qwen2-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "unsloth/Qwen2-70B-bnb-4bit",
        "Qwen/Qwen2-70B",
        "unsloth/Qwen2-70B-Instruct-bnb-4bit",
        "Qwen/Qwen2-70B-Instruct",
        "unsloth/Qwen2-VL-2B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
        "unsloth/Qwen2-VL-2B-bnb-4bit",
        "unsloth/Qwen2-VL-2B",
        "Qwen/Qwen2-VL-2B",
        "unsloth/Qwen2-VL-7B-bnb-4bit",
        "unsloth/Qwen2-VL-7B",
        "Qwen/Qwen2-VL-7B",
        "unsloth/Qwen2-VL-72B-bnb-4bit",
        "unsloth/Qwen2-VL-72B",
        "Qwen/Qwen2-VL-72B",

    ),
    "qwen3": (
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "unsloth/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
        "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "unsloth/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B",
        "unsloth/Qwen3-1.7B-bnb-4bit",
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B",
        "Qwen/Qwen3-4B",
        "unsloth/Qwen3-4B-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "unsloth/Qwen3-8B-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
        "unsloth/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "unsloth/Qwen3-30B-A3B-bnb-4bit",
        "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit",
        "unsloth/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B-Base",
        "unsloth/Qwen3-0.6B-Base-bnb-4bit",
        "unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit",
        "unsloth/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "unsloth/Qwen3-1.7B-Base-bnb-4bit",
        "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Base",
        "Qwen/Qwen3-4B-Base",
        "unsloth/Qwen3-4B-Base-bnb-4bit",
        "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-Base",
        "Qwen/Qwen3-8B-Base",
        "unsloth/Qwen3-8B-Base-bnb-4bit",
        "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit",
        "Qwen/Qwen3-14B-Base",
        "unsloth/Qwen3-14B-Base-bnb-4bit",

    ),
    "qwen3-instruct": (
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "unsloth/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
    ),
    "qwen3-thinking": (
        "unsloth/QwQ-32B-Preview-bnb-4bit",
        "unsloth/QwQ-32B-Preview",
        "Qwen/QwQ-32B-Preview",
        "unsloth/QwQ-32B-unsloth-bnb-4bit",
        "unsloth/QwQ-32B",
        "Qwen/QwQ-32B",
        "unsloth/QwQ-32B-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
    ),
    "zephyr": (
        "unsloth/zephyr-sft-bnb-4bit",
        "unsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "chatml": (
        "unsloth/yi-6b-bnb-4bit",
        "unsloth/yi-6b",
        "01-ai/Yi-6B",
        "unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit",
        "unsloth/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
        "unsloth/OpenHermes-2.5-Mistral-7B",
        "teknium/OpenHermes-2.5-Mistral-7B",
    ),
    "gpt-oss": (
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",
        "openai/gpt-oss-20b",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b",
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    ),
    "starling": (
        "unsloth/Starling-LM-7B-beta-bnb-4bit",
        "unsloth/Starling-LM-7B-beta",
        "Nexusflow/Starling-LM-7B-beta",
    ),
    "yi-chat": (
        "unsloth/yi-34b-chat-bnb-4bit",
        "01-ai/Yi-6B-Chat",
        "01-ai/Yi-34B-Chat",
    )
}

MODEL_TO_TEMPLATE_MAPPER = {}

for key, values in TEMPLATE_TO_MODEL_MAPPER.items():
    for value in values:
        MODEL_TO_TEMPLATE_MAPPER[value] = key
    pass

    # Get lowercased
    lowered_key = key.lower()
    for value in values:
        MODEL_TO_TEMPLATE_MAPPER[value.lower()] = lowered_key
    pass
pass
