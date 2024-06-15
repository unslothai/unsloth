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
    "standardize_dataset",

    "construct_chat_template",
    "test_construct_chat_template",
    "create_ollama_modelfile",
]

from transformers import StoppingCriteria, StoppingCriteriaList
from torch import LongTensor, FloatTensor
from transformers.models.llama.modeling_llama import logger
from .save import patch_saving_functions
import os
import shutil
from .tokenizer_utils import *
from .models._utils import patch_tokenizer

CHAT_TEMPLATES = {}

# =========================================== Unsloth
# Unsloth efficient template leverages from Zephyr
unsloth_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ 'You are a helpful assistant to the user\n' }}"\
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
SYSTEM """You are a helpful assistant to the user"""
'''

unsloth_eos_token = "eos_token"
CHAT_TEMPLATES["unsloth"] = (unsloth_template, unsloth_eos_token, False, unsloth_ollama,)
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
'''

zephyr_eos_token = "eos_token"
CHAT_TEMPLATES["zephyr"] = (zephyr_template, zephyr_eos_token, False, zephyr_ollama,)
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
'''

chatml_eos_token = "<|im_end|>"
CHAT_TEMPLATES["chatml"] = (chatml_template, chatml_eos_token, True, chatml_ollama,)
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
'''

mistral_eos_token = "eos_token"
CHAT_TEMPLATES["mistral"] = (mistral_template, mistral_eos_token, False, mistral_ollama,)
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
'''

llama_eos_token = "eos_token"
CHAT_TEMPLATES["llama"] = (llama_template, llama_eos_token, False, llama_ollama,)
pass

# ===========================================  Vicuna
# https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
vicuna_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + ' ' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' + ' ' }}"\
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
'''

vicuna_eos_token = "eos_token"
CHAT_TEMPLATES["vicuna"] = (vicuna_template, vicuna_eos_token, False, vicuna_ollama,)
pass

# =========================================== Vicuna Old
# https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
vicuna_old_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\\'s questions.' + '\n' }}"\
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
SYSTEM """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
'''

vicuna_old_eos_token = "eos_token"
CHAT_TEMPLATES["vicuna_old"] = (vicuna_old_template, vicuna_old_eos_token, False, vicuna_old_ollama,)
pass

# =========================================== Alpaca multi turn
# https://github.com/tatsu-lab/stanford_alpaca Changed for multi-turn convos
alpaca_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + '\n\n' }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ 'Below are some instructions that describe some tasks. Write responses that appropriately complete each request.\n\n' }}"\
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
SYSTEM """Below are some instructions that describe some tasks. Write responses that appropriately complete each request."""
'''

alpaca_eos_token = "eos_token"
CHAT_TEMPLATES["alpaca"] = (alpaca_template, alpaca_eos_token, False, alpaca_ollama,)
pass

# =========================================== Gemma
# https://huggingface.co/google/gemma-7b-it
# Notice we must use |trim for lstrip and rstrip. <start_of_turn> maps to 106.
# <end_of_turn> maps to 107. user and model are normal 1 word tokens.
gemma_template = \
    "{{ bos_token }}"\
    "{% if messages[0]['role'] == 'system' %}"\
        "{{'<start_of_turn>user\n' + messages[0]['content'] | trim + ' ' + messages[1]['content'] | trim + '<end_of_turn>\n'}}"\
        "{% set loop_messages = messages[2:] %}"\
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
'''

gemma_eos_token = "<end_of_turn>"
CHAT_TEMPLATES["gemma"] = (gemma_template, gemma_eos_token, True, gemma_ollama,)
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
'''

gemma_chatml_eos_token = (
    {"<start_of_turn>" : "<|im_start|>", "<eos>" : "<|im_end|>"},
    "<|im_end|>",
)
CHAT_TEMPLATES["gemma_chatml"] = (gemma_chatml_template, gemma_chatml_eos_token, True, gemma_chatml_ollama,)
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
'''

llama3_template_eos_token = "eos_token"
CHAT_TEMPLATES["llama-3"] = (llama3_template, llama3_template_eos_token, False, llama3_ollama,)
pass


# =========================================== Phi-3
phi3_template = \
    "{{ bos_token }}"\
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
'''

phi3_template_eos_token = "<|end|>"
CHAT_TEMPLATES["phi-3"] = (phi3_template, phi3_template_eos_token, False, phi3_ollama,)
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

    if type(chat_template) in (list, tuple,):
        chat_template, stop_word = chat_template
        assert(type(chat_template) is str)
        assert(type(stop_word) is str)
        ollama_modelfile = None

    elif type(chat_template) is str:

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
            print(
                f"Unsloth: Not a fast tokenizer, so can't process it as of yet :(\n"\
                "Please log a Github issue if you want this as a new feature!\n"\
                "Your chat template will still work, but it won't add or edit tokens."
            )

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

    # For ShareGPT role -> from and content -> value
    chat_template = chat_template\
        .replace("'role'",      "'" + mapping["role"]      + "'")\
        .replace("'content'",   "'" + mapping["content"]   + "'")\
        .replace("'user'",      "'" + mapping["user"]      + "'")\
        .replace("'assistant'", "'" + mapping["assistant"] + "'")

    # Careful on Gemma
    # bos_token is a must or else losses become too high
    if IS_GEMMA and not chat_template.startswith("{{ bos_token }}"):
        chat_template = "{{ bos_token }}" + chat_template
    pass

    _, tokenizer = patch_tokenizer(model = None, tokenizer = tokenizer)
    tokenizer.padding_side  = old_padding_side
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


def standardize_dataset(
    dataset,
    conversation_key = "conversations",
    system_message = None,
    aliases_for_system    = ["system",],
    aliases_for_user      = ["user", "human", "input",],
    aliases_for_assistant = ["gpt", "assistant", "output",],
):
    """
        Standardizes ShareGPT and other formats to user/assistant Hugging Face format.
    """
    import collections
    import itertools

    convos = dataset[:10][conversation_key]
    uniques = collections.defaultdict(list)
    for convo in convos:
        for message in convo:
            for key, value in message.items():
                uniques[key].append(value)
    pass

    # Must be only 2 entries
    assert(len(uniques.keys()) == 2)

    keys = list(uniques.keys())
    length_first  = len(set(uniques[keys[0]]))
    length_second = len(set(uniques[keys[1]]))

    if length_first < length_second:
        # Role is assigned to the first element
        role_key    = keys[0]
        content_key = keys[1]
    else:
        role_key    = keys[1]
        content_key = keys[0]
    pass

    # Check roles are in aliases
    all_aliases = set(aliases_for_system + aliases_for_user + aliases_for_assistant)
    roles = set(uniques[role_key])
    leftover_aliases = (all_aliases | roles) - all_aliases
    if len(leftover_aliases) != 0:
        raise TypeError(
            f"Unsloth: {list(leftover_aliases)} are not in aliases. Please update aliases."
        )
    pass

    # Mapping for aliases
    aliases_mapping = {}
    for x in aliases_for_system:    aliases_mapping[x] = "system"
    for x in aliases_for_user:      aliases_mapping[x] = "user"
    for x in aliases_for_assistant: aliases_mapping[x] = "assistant"

    def _standardize_dataset(examples):
        convos = examples[conversation_key]
        all_convos = []
        for convo in convos:
            new_convo = []
            if len(convo) == 0: continue
            has_system = aliases_mapping[convo[0][role_key]] == "system"
            if not has_system and system_message is not None:
                new_convo.append({ "role" : "system", "content" : system_message, })
            for message in convo:
                role = aliases_mapping[message[role_key]]
                new_convo.append({ "role" : role, "content" : message[content_key], })
            pass
            all_convos.append(new_convo)
        pass
        return { conversation_key : all_convos, }
    pass

    return dataset.map(_standardize_dataset, batched = True,)
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
    final_eos_tokens = []
    for old, new in zip(added_tokens_decoder, splitted):
        if old == new: final_eos_tokens.append(old)
    pass
    final_eos_tokens += extra_eos_tokens
    final_eos_tokens += repeatted_tokens
    return final_eos_tokens
pass


def construct_chat_template( \

tokenizer = None,

template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

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
    assert(tokenizer is not None)

    if extra_eos_tokens is None: extra_eos_tokens = []

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

    # O(N^2) search finding 2 repeatted pieces of text
    j = len(template)-1
    at_least_one = False
    while j > 0:
        found = template.rfind(template[j:], 0, j)
        if found == -1: break
        j -= 1
        at_least_one = True
    pass
    if j > 0: j += 1
    else: raise RuntimeError(error_msg)


    if not at_least_one: raise RuntimeError(error_msg)

    # Repeatted text
    instruction_response = template[j:]
    if instruction_response.count("{INPUT}") != 1 or instruction_response.count("{OUTPUT}") != 1:
        raise RuntimeError(error_msg)
    pass

    # 1st System, Instruction, Output pair
    left  = template[:j]
    # 2nd Instruction, Output pair
    right = template[j:]

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
    system_part = left[:left.find(input_part)]

    # Check if the user provided a correct prompt
    combined = system_part + input_part + output_part
    if combined != left:
        combined_changed = combined.replace('\n', '\\n')
        left_changed     = left    .replace('\n', '\\n')
        raise RuntimeError(
            "Unsloth: The prompt template you provided isn't correct. You gave:\n"\
            f"{combined_changed}\n\n"\
            "But we require the following:\n"\
            f"{left_changed}"
        )
    pass

    # Ollama modelfile parts

    # Check bos_token is in system prompt
    ollama_system = system_part
    has_bos_token = False
    if tokenizer("A").input_ids[0] == getattr(tokenizer, "bos_token_id", None):
        if ollama_system.startswith(tokenizer.bos_token):
            has_bos_token = True
            ollama_system = ollama_system[len(tokenizer.bos_token):]
        pass
    pass
    system_modelfile = "{{ if .System }}" + ollama_system.replace("{SYSTEM}", "{{ .System }}") + "{{ end }}"
    input_modelfile  = "{{ if .Prompt }}" + input_part .replace("{INPUT}",  "{{ .Prompt }}") + "{{ end }}"
    output_modelfile = output_part.replace("{OUTPUT}", "{{ .Response }}")

    # Check if EOS token is at the end of the output
    if not output_modelfile.endswith(tuple(extra_eos_tokens)):
        output_modelfile += "{__EOS_TOKEN__}"
    pass

    # Ollama EOS
    ollama_eos = get_ollama_eos_tokens(tokenizer, extra_eos_tokens)
    ollama_eos = '\n'.join(f'PARAMETER stop "{eos}"' for eos in ollama_eos)

    # Ollama modelfile
    modelfile = 'FROM {__FILE_LOCATION__}\n\n'\
    'TEMPLATE """' + system_modelfile + input_modelfile + output_modelfile + \
    '"""\n\n' + ollama_eos

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

        # Separate the BOS
        if has_bos_token:
            partial_system = partial_system.replace(tokenizer.bos_token, "", 1)
        pass

        partial_system = \
            "{% if messages[0]['role'] == 'system' %}"\
                "{{ " + partial_system + " }}"\
                "{% set loop_messages = messages[1:] %}"
        if default_system_message is not None:
            partial_system += "{% else %}"\
                "{{ '" + system_part.replace("{SYSTEM}", default_system_message) + "' }}"\
                "{% set loop_messages = messages %}"\
            "{% endif %}"
        else:
            partial_system += "{% endif %}"
        pass

        jinja_template = partial_system + jinja_template

        if has_bos_token:
            jinja_template = "{{ bos_token }}" + jinja_template
    pass

    return modelfile, jinja_template
pass


def test_construct_chat_template():
    token = "hf_"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token = token)

    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""
    
    default_system_message = \
        "Below are some instructions that describe some tasks. Write responses that appropriately complete each request."
      
    extra_eos_tokens = None

    modelfile, jinja_template = construct_chat_template(template, default_system_message, extra_eos_tokens)

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


def create_ollama_modelfile(tokenizer, gguf_location):
    """
        Creates an Ollama Modelfile.
        Use ollama.create(model = "new_ollama_model", modelfile = modelfile)
    """
    modelfile = getattr(tokenizer, "_ollama_modelfile", None)
    if modelfile is None:
        raise RuntimeError(
            "Unsloth: Tokenizer does not have a `ollama_modelfile` attribute.\n"\
            "Please use get_chat_template(...)."
        )
    pass

    system_message = getattr(tokenizer, "_system_message", None)
    if system_message is None:
        __SYSTEM_MESSAGE__ = ""
    else:
        __SYSTEM_MESSAGE__ = f'SYSTEM """{system_message}"""'
    pass

    modelfile = modelfile\
        .replace("{{", "⚫@✅#🦥")\
        .replace("}}", "⚡@🦥#⛵")\
        .format(
            __FILE_LOCATION__  = gguf_location,
            __SYSTEM_MESSAGE__ = __SYSTEM_MESSAGE__,
            __EOS_TOKEN__      = tokenizer.eos_token,
        )\
        .replace("⚫@✅#🦥", "{{")\
        .replace("⚡@🦥#⛵", "}}")\
        .rstrip()
    pass

    return modelfile
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
        gguf_tokenized = re.findall("([\d]{1,}) \-\> \'([^\']{1,})\'", gguf_tokens, flags = re.MULTILINE)
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
