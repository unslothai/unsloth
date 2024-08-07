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

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
import re
import os
from transformers.models.llama.modeling_llama import logger
from peft import PeftModelForCausalLM
import torch
import itertools
import collections
import numpy as np
import gc
import subprocess

__all__ = [
    "load_correct_tokenizer",
    "fix_sentencepiece_tokenizer",
    "check_tokenizer",
    "add_new_tokens",
    "fix_sentencepiece_gguf",
]


IGNORED_TOKENIZER_CHECKING = frozenset((
    "CodeLlamaTokenizerFast",
    "CodeLlamaTokenizer",
))


IGNORED_TOKENIZER_NAMES = [
    # "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    # "unsloth/Mistral-Nemo-Instruct-2407",
    # "mistralai/Mistral-Nemo-Instruct-2407",
    # "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    # "unsloth/Mistral-Nemo-Base-2407",
    # "mistralai/Mistral-Nemo-Base-2407",
]
IGNORED_TOKENIZER_NAMES = frozenset(
    [x.lower() for x in IGNORED_TOKENIZER_NAMES]
)

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT  = "\nCOLAB_"  in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
del keynames


def try_fix_tokenizer(tokenizer, prepend = True):

    if hasattr(tokenizer, "_tokenizer"):
        converted_tokenizer = tokenizer._tokenizer
    else:
        converted_tokenizer = convert_slow_tokenizer(tokenizer)
    pass

    tokenizer_string = converted_tokenizer.to_str()

    # Llama does _apple. Sometimes this is wrong!!
    prepend_text = '{"type":"Prepend","prepend":"▁"},'
    if not prepend and prepend_text in tokenizer_string:
        tokenizer_string = tokenizer_string.replace(prepend_text, "", 1)
    pass

    dir_names = dir(tokenizer)
    # Get eos_token, bos_token etc
    token_names = [x for x in dir_names if x.endswith("_token") and x.count("_") == 1]

    for token_name in token_names:
        token = getattr(tokenizer, token_name, None)
        if token is None: continue
        token_id = getattr(tokenizer, token_name + "_id", None)

        # Locate the token's id mapping in the string
        find_text = f'"id":{token_id},"content":"'
        start = tokenizer_string.find(find_text) + len(find_text)
        if start == -1: continue
        end   = tokenizer_string.find('",', start)

        bad_token = tokenizer_string[start : end]
        # Check if token is the actual same one - if not, edit it
        if bad_token != token:
            bad_text  = f'{find_text}{bad_token}",'
            good_text = f'{find_text}{token}",'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)

            # And replace vocab section
            bad_text = f'"{bad_token}":{token_id},'
            good_text = f'"{token}":{token_id},'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)
        pass
    pass

    fixed_tokenizer = converted_tokenizer.from_str(tokenizer_string)
    return fixed_tokenizer
pass


def get_sorted_dict(dictionary):
    sorted_keys = sorted(dictionary.values())
    inverted_dictionary = { value : key for key, value in dictionary.items() }

    sorted_dictionary = {}
    for key in sorted_keys:
        value = inverted_dictionary[key]
        sorted_dictionary[value] = key
    return sorted_dictionary
pass


def convert_to_fast_tokenizer(
    slow_tokenizer,
    temporary_location = "_unsloth_sentencepiece_temp",
):
    is_fast = getattr(slow_tokenizer, "is_fast", False)
    if is_fast: return slow_tokenizer
    
    try:
        tokenizer_name = slow_tokenizer.__class__.__name__
        lowered_tokenizer_name = tokenizer_name.lower()
        if lowered_tokenizer_name.endswith("tokenizer"):
            class_name = lowered_tokenizer_name[:-len("tokenizer")]
            FastTokenizer = eval(
                f'__import__(f"transformers.models.{class_name}").{tokenizer_name}Fast'
            )
        else:
            FastTokenizer = PreTrainedTokenizerFast
    except:
        FastTokenizer = PreTrainedTokenizerFast
    pass

    # Get all arguments (bos_token, etc)
    docs = FastTokenizer.__doc__
    docs = docs[docs.find("Args:"):]
    args = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags = re.MULTILINE)
    args = [x for x in args if not x.endswith("_file")]

    # Also some missing maybe!
    docs = PreTrainedTokenizerFast.__doc__
    docs = docs[docs.find("Args:"):]
    args2 = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags = re.MULTILINE)
    args2 = [x for x in args2 if not x.endswith("_file")]
    args = list(set(args + args2))

    kwargs = {}
    for arg in args: kwargs[arg] = getattr(slow_tokenizer, arg, None)
    kwargs["tokenizer_object"] = try_fix_tokenizer(slow_tokenizer, prepend = True)
    fast_tokenizer = FastTokenizer( **kwargs )

    # Check if they're similar!
    sorted_slow_tokenizer = get_sorted_dict(slow_tokenizer.get_vocab())
    sorted_fast_tokenizer = get_sorted_dict(fast_tokenizer.get_vocab())

    check_vocab   = (sorted_slow_tokenizer == sorted_fast_tokenizer)
    check_special = (slow_tokenizer.all_special_tokens == fast_tokenizer.all_special_tokens)

    # Failure so return slow_tokenizer
    if not check_vocab or not check_special: return slow_tokenizer

    # Now confirm if they match
    if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        # Maybe remove prepending of __apple?
        kwargs["tokenizer_object"] = try_fix_tokenizer(slow_tokenizer, prepend = False)
        fast_tokenizer = FastTokenizer( **kwargs )
        if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            # Failure :(
            return slow_tokenizer
        pass
    pass

    # Also tokenizer.model is missing!
    name = slow_tokenizer.name_or_path.replace("/", "_")
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass
    new_location = f"{temporary_location}/{name}"
    slow_tokenizer.save_pretrained(new_location)
    fast_tokenizer.save_pretrained(new_location)

    # Now load it!
    fast_tokenizer = AutoTokenizer.from_pretrained(new_location)
    if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        return fast_tokenizer
    return slow_tokenizer
pass


# Check Mistral chat template without BOS / EOS
mistral_template = \
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
            "{{ message['content'] }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"
pass

# Check Llama chat template without BOS / EOS
llama_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{% if messages[1]['role'] == 'user' %}"\
            "{{ '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' + messages[1]['content'] + ' [/INST]' }}"\
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
            "{{ '[INST] ' + message['content'].strip() + ' [/INST]' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ ' ' + message['content'].strip() + ' ' }}"\
        "{% else %}"\
            "{{ raise_exception('Only user and assistant roles are supported!') }}"\
        "{% endif %}"\
    "{% endfor %}"
pass


def assert_same_tokenization(slow_tokenizer, fast_tokenizer):
    # Get eos_token, bos_token etc
    dir_names = dir(slow_tokenizer)
    special_tokens = list(filter(None, (
        getattr(slow_tokenizer, x) for x in dir_names
        if x.endswith("_token") and x.count("_") == 1
    )))
    all_special_tokens = list(set(special_tokens + slow_tokenizer.all_special_tokens))

    # Check if chat template is enabled!
    check_chat_template1 = True
    check_chat_template2 = True
    check_chat_template3 = True
    
    """
    Weirdly Mistral tokenizers are actually correct??
    Ie below will actually load mistral v1 and v3 incorrectly!

    slow_chat_template = getattr(slow_tokenizer, "chat_template", None)
    fast_chat_template = getattr(fast_tokenizer, "chat_template", None)
    messages = [
        {"role": "user", "content": " What is 2+2? "},
        {"role": "assistant", "content": " It's 4. "},
    ]
    # Check the tokenizer's own chat template
    if slow_chat_template is not None and fast_chat_template is not None:
        check_chat_template1 = \
            slow_tokenizer.apply_chat_template(messages) == \
            fast_tokenizer.apply_chat_template(messages)
    pass

    # Check Mistral chat template without BOS / EOS
    slow_tokenizer.chat_template = mistral_template
    fast_tokenizer.chat_template = mistral_template
    check_chat_template2 = \
        slow_tokenizer.apply_chat_template(messages) == \
        fast_tokenizer.apply_chat_template(messages)
    pass

    # Check Llama chat template without BOS / EOS
    slow_tokenizer.chat_template = llama_template
    fast_tokenizer.chat_template = llama_template
    check_chat_template3 = \
        slow_tokenizer.apply_chat_template(messages) == \
        fast_tokenizer.apply_chat_template(messages)
    pass

    # Combine them all and revert chat templates
    slow_tokenizer.chat_template = slow_chat_template
    fast_tokenizer.chat_template = fast_chat_template
    """
    check_chat_template = check_chat_template1 and check_chat_template2 and check_chat_template3

    # Try special tokens
    try:
        string = "\n".join(all_special_tokens) + \
            "A quick brown fox jumps over the lazy dog!!\n\nHi</s>\n\n" + \
            "".join(all_special_tokens)
        check_special_tokens = \
            slow_tokenizer(string).input_ids == \
            fast_tokenizer(string).input_ids

        return check_chat_template and check_special_tokens
    except:
        # For eg see https://github.com/unslothai/unsloth/issues/292
        # Sometimes tokenizer has weird tokens, causing a combined tokenization to fail.
        # [TODO] We temporarily disable this for CodeLlama tokenizers
        if slow_tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
            return check_chat_template
        else:
            return False
    pass
pass


def fix_sentencepiece_tokenizer(
    old_tokenizer,
    new_tokenizer,
    token_mapping,
    temporary_location = "_unsloth_sentencepiece_temp",
):
    # From https://github.com/google/sentencepiece/issues/121
    # We need to manually edit the sentencepiece tokenizer!
    from transformers.utils import sentencepiece_model_pb2

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    pass

    # Check if tokenizer.model exists
    if not os.path.isfile(f"{temporary_location}/tokenizer.model"):
        return new_tokenizer
    pass

    # First save the old tokenizer
    old_tokenizer.save_pretrained(temporary_location)

    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    tokenizer_file.ParseFromString(open(f"{temporary_location}/tokenizer.model", "rb").read())

    # Now save the new tokenizer
    new_tokenizer.save_pretrained(temporary_location)

    # Now correct the old tokenizer's .model file
    for old_token, new_token in token_mapping.items():
        ids = old_tokenizer([old_token], add_special_tokens = False).input_ids
        ids = ids[0]
        if (len(ids) != 1):
            # Skip this token!
            print(f"Skip mapping {old_token} to {new_token} since {new_token} is already in the tokenizer!")
            continue
        pass
        ids = ids[0]
        # [TODO] Hack for Starling - try except
        try:
            tokenizer_piece = tokenizer_file.pieces[ids]
        except:
            continue
        assert(tokenizer_piece.piece == old_token)
        tokenizer_piece.piece = new_token
    pass

    # And now write it
    with open(f"{temporary_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())
    pass

    # And load it!
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        temporary_location,
        eos_token = new_tokenizer.eos_token,
        pad_token = new_tokenizer.pad_token,
    )
    return tokenizer
pass


def fix_sentencepiece_gguf(saved_location):
    """
        Fixes sentencepiece tokenizers which did not extend the vocabulary with
        user defined tokens.
        Inspiration from https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py
    """
    from copy import deepcopy
    from transformers.utils import sentencepiece_model_pb2
    import json
    from enum import IntEnum
    
    class SentencePieceTokenTypes(IntEnum):
        NORMAL = 1
        UNKNOWN = 2
        CONTROL = 3
        USER_DEFINED = 4
        UNUSED = 5
        BYTE = 6
    pass

    # Load tokenizer.model
    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    if not os.path.isfile(f"{saved_location}/tokenizer.model"): return
    tokenizer_file.ParseFromString(open(f"{saved_location}/tokenizer.model", "rb").read())
    sentence_piece_size = len(tokenizer_file.pieces)

    # Load added_tokens_json
    if not os.path.isfile(f"{saved_location}/added_tokens.json"): return
    with open(f"{saved_location}/added_tokens.json", "r", encoding = "utf-8") as file:
        added_tokens_json = json.load(file)
    pass
    if len(added_tokens_json) == 0: return

    added_tokens_json = dict(sorted(added_tokens_json.items(), key = lambda item: item[1]))
    new_size = sentence_piece_size + len(added_tokens_json)

    # Confirm added_tokens_json is correct
    added_tokens_ids = np.array(list(added_tokens_json.values()))
    diff = np.diff(added_tokens_ids)
    if (diff.min() != 1 or diff.max() != 1): return
    if (added_tokens_ids.min() != sentence_piece_size): return

    # Edit sentence piece tokens with added_tokens_json
    logger.warning(
        f"Unsloth: Extending {saved_location}/tokenizer.model with added_tokens.json.\n"\
        f"Originally tokenizer.model is of size ({sentence_piece_size}).\n"\
        f"But we need to extend to sentencepiece vocab size ({new_size})."
    )
    new_tokens = deepcopy(tokenizer_file.pieces[-len(added_tokens_ids):])
    for new_token, added_token in zip(new_tokens, added_tokens_json.keys()):
        new_token.piece = added_token.encode("utf-8")
        new_token.score = -1000.0
        new_token.type  = SentencePieceTokenTypes.USER_DEFINED
    pass

    tokenizer_file.pieces.extend(new_tokens)

    with open(f"{saved_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())
    pass

    # Add padding tokens
    # actual_vocab_size = model.config.vocab_size
    # padding = actual_vocab_size - len(tokenizer_file.pieces)
    return
pass


def _load_correct_tokenizer(
    tokenizer_name,
    model_max_length = None,
    padding_side = "right",
    token = None,
    trust_remote_code = False,
    cache_dir = "huggingface_tokenizers_cache",
    fix_tokenizer = True,
):
    if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
        cache_dir = cache_dir
    else:
        cache_dir = None
    pass

    # Try loading the slow tokenizer. If it fails, then try Fast only
    # Mainly to solve Deepseek models with no tokenizer.model file
    slow_tokenizer = None
    try:
        slow_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length  = model_max_length,
            padding_side      = padding_side,
            token             = token,
            trust_remote_code = trust_remote_code,
            # Cannot just use use_fast = False as per https://twitter.com/danielhanchen/status/1789659394302718373
            use_fast          = False,
            legacy            = False,
            from_slow         = True,
            cache_dir         = cache_dir,
        )
    except:
        pass
        # print(
        #     f"Unsloth: {tokenizer_name} has no tokenizer.model file.\n"\
        #     "Just informing you about this - this is not a critical error."
        # )
    pass

    fast_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length  = model_max_length,
        padding_side      = padding_side,
        token             = token,
        trust_remote_code = trust_remote_code,
        cache_dir         = cache_dir,
    )

    if not fix_tokenizer or tokenizer_name in IGNORED_TOKENIZER_NAMES:
        return fast_tokenizer
    # Ignore Mistral ones - they're a bit weird to handle!
    elif "mistral" in tokenizer_name.lower():
        return fast_tokenizer
    elif slow_tokenizer is not None:
        if hasattr(fast_tokenizer, "add_bos_token") and hasattr(slow_tokenizer, "add_bos_token"):
            fast_tokenizer.add_bos_token = slow_tokenizer.add_bos_token
        if hasattr(fast_tokenizer, "add_eos_token") and hasattr(slow_tokenizer, "add_eos_token"):
            fast_tokenizer.add_eos_token = slow_tokenizer.add_eos_token
        
        # Confirm if slow and fast are equivalent!
        if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            return fast_tokenizer
        else:
            logger.warning(f"Unsloth: Will load {tokenizer_name} as a legacy tokenizer.")
            return convert_to_fast_tokenizer(slow_tokenizer)
        pass
    else:
        return fast_tokenizer
    pass
pass


def load_correct_tokenizer(
    tokenizer_name,
    model_max_length = None,
    padding_side = "right",
    token = None,
    trust_remote_code = False,
    cache_dir = "huggingface_tokenizers_cache",
    fix_tokenizer = True,
):
    tokenizer = _load_correct_tokenizer(
        tokenizer_name = tokenizer_name,
        model_max_length = model_max_length,
        padding_side = padding_side,
        token = token,
        trust_remote_code = trust_remote_code,
        cache_dir = cache_dir,
        fix_tokenizer = fix_tokenizer,
    )

    ### 1. Fixup tokenizer's chat_template
    old_chat_template = getattr(tokenizer, "chat_template", None)

    # Ignore mistral type models since they don't have a add_generation_prompt
    if "mistral" in str(getattr(tokenizer, "name_or_path", "")).lower():
        chat_template = old_chat_template

    # Also check Llama-2 old style models
    elif old_chat_template is not None and \
        "[/INST]" in old_chat_template and "[INST]" in old_chat_template and \
        "bos_token" in old_chat_template and "eos_token" in old_chat_template:

        chat_template = old_chat_template

    else:
        chat_template = fix_chat_template(tokenizer)
        if old_chat_template is not None and chat_template is None:
            raise RuntimeError(
                "Unsloth: Fixing chat template failed - please file a report immediately!"
            )
        pass
    pass

    tokenizer.chat_template = chat_template
    return tokenizer
pass


def _fix_chat_template(chat_template):
    endfor = "{% endfor %}"
    where = chat_template.find(endfor)
    if where == -1: return chat_template

    after_endfor = chat_template[where + len(endfor):]

    if "{% if" not in after_endfor and "{% set " not in after_endfor and \
        after_endfor.startswith("{{") and after_endfor.endswith("}}") and \
        after_endfor.count("{{") == 1 and after_endfor.count("}}") == 1:

        after_endfor = "{% if add_generation_prompt %}" + after_endfor + "{% endif %}"

        chat_template = chat_template[:where + len(endfor)] + after_endfor
    pass
    return chat_template
pass


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None: return None

    ### 1. Check if add_generation_prompt works
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    no  = tokenizer.apply_chat_template(messages, add_generation_prompt = False, tokenize = False)
    yes = tokenizer.apply_chat_template(messages, add_generation_prompt =  True, tokenize = False)

    if no == yes:
        # SAME?! That's not good! We check for add_generation_prompt
        if "{% if add_generation_prompt %}" not in chat_template:
            # Try fixing it by adding it
            new_chat_template = _fix_chat_template(chat_template)
            if "{% if add_generation_prompt %}" not in new_chat_template:
                raise RuntimeError(
                    f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"\
                    "does not have a {% if add_generation_prompt %} for generation purposes.\n"\
                    "Please file a bug report immediately - thanks!"
                )
            else:
                logger.warning_once(
                    "Unsloth: We successfully patched the tokenizer to add a {% if add_generation_prompt %} to the chat_template.\n"\
                    "This is not a bug, but please notify the Unsloth maintainers - thanks!"
                )
                chat_template = new_chat_template
            pass
        else:
            raise RuntimeError(
                f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"\
                "has a {% if add_generation_prompt %} for generation purposes, but wasn't provided correctly.\n"\
                "Please file a bug report immediately - thanks!"
            )
        pass
    pass
    return chat_template
pass


def check_tokenizer(
    model,
    tokenizer,
    model_name = "unsloth/llama-2-7b-bnb-4bit",
    model_max_length = 4096,
    padding_side = "right",
    token = None,
    _reload = True,
):
    # Checks tokenizer for out of bounds ids.
    # Mainly a fix for https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
    # where <sep> had token id=32002.
    # See https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
    # Seems like the Fast tokenizer in Rust breaks things!

    # We ignore some of them!
    if tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
        return tokenizer
    pass

    max_embedding_size = model.model.embed_tokens.weight.shape[0]
    added_tokens_fast = tokenizer.added_tokens_decoder
    added_tokens_fast = {index : str(value) for index, value in added_tokens_fast.items()}
    sorted_keys = sorted(added_tokens_fast)
    added_tokens_fast = {key : added_tokens_fast[key] for key in sorted_keys}

    for j, index in enumerate(added_tokens_fast.keys()):
        if index >= max_embedding_size:
            bad_indices = list(added_tokens_fast.keys  ())[j:]
            bad_tokens  = list(added_tokens_fast.values())[j:]
            if not _reload:
                # Try removing the token
                added_tokens = [str(x) for x in tokenizer.added_tokens_decoder.values()]
                special_tokens = tokenizer.special_tokens_map
                import itertools
                special_tokens = frozenset(
                    itertools.chain.from_iterable(
                        [x] if type(x) is str else x for x in special_tokens.values()
                    )
                )
                can_be_removed1 = [x for x in bad_tokens if x not in special_tokens]
                can_be_removed2 = [x for x in can_be_removed1 if x in tokenizer._added_tokens_encoder.keys()]

                # Check of extra tokens can in fact we removed!
                can_be_removed = \
                    (len(can_be_removed1) == len(bad_tokens)) and \
                    (len(can_be_removed2) == len(bad_tokens))

                # Check if sep_token or other generic types
                remove_generic = False
                try_mapper = []
                if not can_be_removed:
                    names = dir(tokenizer)
                    names = (x for x in names if x.endswith("_token") and x.count("_") == 1)
                    generic_tokens = [(x, getattr(tokenizer, x, None)) for x in names]

                    try_removal = []
                    for token in bad_tokens:
                        for (name_token, check_token) in generic_tokens:
                            if check_token == token:
                                try_removal.append(token)
                                try_mapper.append(name_token)
                            pass
                        pass
                    pass

                    # Recheck!
                    can_be_removed = (len(try_removal) == len(bad_tokens))
                    if can_be_removed: remove_generic = True
                    can_be_removed1 = bad_tokens
                pass

                if can_be_removed:
                    # Yes it can be fixed!
                    for j, bad_token in enumerate(can_be_removed1):
                        remove_id = tokenizer._added_tokens_encoder[bad_token]
                        del tokenizer._added_tokens_decoder[remove_id]
                        del tokenizer._added_tokens_encoder[bad_token]

                        if remove_generic and (try_removal[j] == bad_token):
                            # Remove sep token for example
                            setattr(tokenizer, try_mapper[j], None)
                            setattr(tokenizer, try_mapper[j] + "_id", None)
                        pass
                    pass
                    # Confirm 1 more time!
                    if max(tokenizer.added_tokens_decoder.keys()) < max_embedding_size:
                        logger.warning_once(
                            f"Unsloth loaded a broken tokenizer `{model_name}`, but managed to repair it!\n"\
                            f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"\
                            "We removed these bad tokens. If you think this is incorrect, fix your tokenizer first."
                        )
                        return convert_to_fast_tokenizer(tokenizer)
                    pass
                pass

                # :( Failure
                raise RuntimeError(
                    f"Unsloth tried to load `{model_name}`, but cannot succeed.\n"\
                    f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"\
                    f"Fix your tokenizer since it'll perform out of bounds memory accesses."
                )
            pass
            
            if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
                cache_dir = "huggingface_tokenizers_cache"
            else:
                cache_dir = None
            pass

            # Sometimes slow tokenizer does not work like Deepseek
            try:
                # Try slow tokenizer which can fix things!
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_length = model_max_length,
                    padding_side = padding_side,
                    token = token,
                    # Cannot just use use_fast = False as per https://twitter.com/danielhanchen/status/1789659394302718373
                    use_fast = False,
                    legacy = False,
                    from_slow = True,
                    cache_dir = cache_dir,
                )
                return check_tokenizer(
                    model = model,
                    tokenizer = tokenizer,
                    model_name = model_name,
                    model_max_length = model_max_length,
                    padding_side = padding_side,
                    token = token,
                    _reload = False,
                )
                break
            except:
                # Tokenizer has out of bounds issues and we can't
                # load the slow tokenizer version :(
                logger.warning_once(
                    "Unsloth: Tokenizer is most likely buggy, and Unsloth failed to repair it.\n"\
                    "It will still work, but beware of out of bounds memory accesses.\n"\
                    "Please file an issue on the model owner's repo about this issue."
                )
                return tokenizer
            pass
        pass
    pass
    return convert_to_fast_tokenizer(tokenizer)
pass


@torch.inference_mode
def fix_untrained_tokens(model, tokenizer, train_dataset, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings ().weight
    lm_head_matrix   = model.get_output_embeddings().weight

    # Ignore some model checks for now
    if model.config._name_or_path in  IGNORED_TOKENIZER_NAMES:
        return
    pass

    # Get untrained tokens
    indicator_untrained1 = torch.amax(embedding_matrix, axis = 1) <= eps
    # Check lm_head as well
    indicator_untrained2 = torch.amax(lm_head_matrix,   axis = 1) <= eps
    # Combine both checks
    indicator_untrained = indicator_untrained1 & indicator_untrained2
    
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # Get set and actual tokens
    where_untrained = where_untrained.tolist()
    if len(where_untrained) == 0: return

    # Remove untrained indices where it's longer
    
    where_untrained_set = frozenset(where_untrained)
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    # Remove None items in actual_bad_tokens
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    # Check if tokenizer and training datasets have bad tokens
    if_bad_first  = False
    if_bad_second = False
    # Check tokenizer's chat template for any untrained tokens
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None:
        if_bad_first = any(x in chat_template for x in actual_bad_tokens)
    pass

    # Check the first 250, last 250 input_ids
    size_dataset = len(train_dataset)
    size = min(size_dataset, 250)
    for j in range(size):
        input_ids = train_dataset[j]
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
            if_bad = any(item in where_untrained_set for item in input_ids)
            if if_bad:
                if_bad_second = True
                break
            pass
        pass
    pass

    # Check last 250
    if not if_bad_second:
        left = max(size_dataset-250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                if_bad = any(item in where_untrained_set for item in input_ids)
                if if_bad:
                    if_bad_second = True
                    break
                pass
            pass
        pass
    pass

    # Check if bad tokens exists!
    if not if_bad_first and not if_bad_second: return

    # Check if lm_head / embed_token are trainable!
    bad_not_trainable = False
    if not embedding_matrix.requires_grad: bad_not_trainable = True
    if not lm_head_matrix  .requires_grad: bad_not_trainable = True

    if bad_not_trainable:
        raise ValueError(
            'Unsloth: Untrained tokens found, but embed_tokens & lm_head not trainable, causing NaNs. '\
            'Restart then add `embed_tokens` & `lm_head` to '\
            '`FastLanguageModel.get_peft_model(target_modules = [..., "embed_tokens", "lm_head",]). `'\
            'Are you using the `base` model? Instead, use the `instruct` version to silence this warning.',
        )
    pass

    # Count all the possible bad tokens
    final_counts = np.zeros(max(len(tokenizer), embedding_matrix.shape[0]), dtype = np.int64)
    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(itertools.chain.from_iterable(input_ids), dtype = np.int32)
        np.add.at(final_counts, counter, 1)
    pass
    train_dataset.map(mapping, batched = True, desc = "Counting untrained tokens")

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head   = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype = torch.float32, axis = 0)
    sum_lm_head   -= torch.sum(lm_head_matrix  [where_untrained], dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained)
    mean_lm_head   = (sum_lm_head   / n_trained)

    # Scale each to be equal to 1/max_frequency. Also set some to 0 if none seen
    scaling = final_counts[where_untrained] / max(final_counts.max(), 1)
    scaling = torch.tensor(scaling, device = mean_embedding.device).unsqueeze(1)
    mean_embedding = mean_embedding.repeat((n_untrained, 1,)) * scaling
    mean_lm_head   = mean_lm_head  .repeat((n_untrained, 1,)) * scaling
    where_null = scaling.ravel() == 0
    mean_embedding[where_null] = 0
    mean_lm_head  [where_null] = 0

    # Set them to the mean
    logger.warning(
        "Unsloth: Setting embed_tokens & lm_head untrained tokens to "\
        "mean(trained) to counteract NaNs during training."
    )
    embedding_matrix[where_untrained] = mean_embedding.to(embedding_matrix.dtype)
    lm_head_matrix  [where_untrained] = mean_lm_head  .to(lm_head_matrix  .dtype)

    # Clean up
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    pass
    return
pass


@torch.inference_mode
def mean_of_trained_tokens(model, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings ().weight.clone()
    lm_head_matrix   = model.get_output_embeddings().weight.clone()

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    # if n_untrained != 0:
    #     print(
    #         f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
    #         "We shall set them to the mean of the other trained tokens."
    #     )
    # pass

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head   = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Remove bad tokens
    sum_embedding -= torch.sum(embedding_matrix[where_untrained], dtype = torch.float32, axis = 0)
    sum_lm_head   -= torch.sum(lm_head_matrix  [where_untrained], dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained)
    mean_lm_head   = (sum_lm_head   / n_trained)

    return mean_embedding, mean_lm_head
pass


@torch.inference_mode
def add_new_tokens(
    model,
    tokenizer,
    new_tokens = [],
    method = "mean",
    interpolation = 0.5,
):
    """
    Smartly resizes the tokenizer and adds new tokens to the model.
    We also disregard untrained tokens by removing them from the mean calculation.
    """
    assert(isinstance(new_tokens, (list, tuple)))
    assert(len(new_tokens) > 0)
    assert(method == "mean" or method == "interpolation")
    assert(interpolation >= 0 and interpolation <= 1)

    # Check if tokens already exist
    overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
    if len(overlapping_tokens) != 0:
        print(
            f"Unsloth: You're adding new_tokens = {new_tokens}\n"\
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"\
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
    pass

    # Get mean of trained tokens
    # mean_embedding, mean_lm_head = fix_untrained_tokens(model)

    # Weirdly be careful reserved tokens can pop out
    mean_embedding, mean_lm_head = mean_of_trained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head   = mean_lm_head  .to(torch.float32)

    # Add tokens!
    old_length = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # If we use interpolation, we interpolate between the mean embeddings and
    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings ().weight
    lm_head_matrix   = model.get_output_embeddings().weight

    if method == "interpolation":
        print(
            "Unsloth: You are using interpolation to add new tokens.\n"\
            f"We shall set new tokens = mean(embeddings)*{1-interpolation} + mean(new_tokens)*{interpolation}"
        )
        for j, token in enumerate(new_tokens):
            input_ids = tokenizer(token, add_special_tokens = False).input_ids
            mean_embedding_token = embedding_matrix[input_ids].mean(axis = 0, dtype = torch.float32)
            mean_lm_head_token   = lm_head_matrix  [input_ids].mean(axis = 0, dtype = torch.float32)

            # Interpolate
            mean_embedding_token = mean_embedding*(1-interpolation) + mean_embedding_token*interpolation
            mean_lm_head_token   = mean_lm_head  *(1-interpolation) + mean_lm_head_token  *interpolation

            # Set the new vector
            embedding_matrix[old_length+j] = mean_embedding_token
            lm_head_matrix  [old_length+j] = mean_lm_head_token
        pass
    else:
        # Now set the new tokens to the mean!
        embedding_matrix[old_length:] = mean_embedding
        lm_head_matrix  [old_length:] = mean_lm_head
    pass

    # We set a flag to say we need to train embeddings
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    pass
    internal_model._need_to_train_embeddings = True
    
    return
pass


def check_nvidia():
    # Unsloth doesn't work yet on AMD devices - we're working on it!
    output = np.array([0,])
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv", shell = True)
        output = re.findall(rb'([\d]{1,})[\s]{1,}M', output)
        output = np.array([int(x.decode('utf-8'))/1024 for x in output])
    except:
        if not torch.cuda.is_available():
            raise RuntimeError("Unsloth: We do not support AMD / Intel machines yet - it is a work in progress!")
    return output
pass
PRE_CHECK = check_nvidia()


from inspect import getsource
import trl.trainer.sft_trainer
from trl.trainer.sft_trainer import *
from transformers.trainer import *

def patch_sft_trainer_tokenizer():
    """
        Patches the trainer with changes
    """
    for function_name, replacer in (
        ("_prepare_non_packed_dataloader", "def tokenize(element):",),
        # ("_prepare_packed_dataloader", "if dataset_text_field is not None",),
    ):
        function = getsource(eval(f"trl.trainer.sft_trainer.SFTTrainer.{function_name}"))
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        check_text = \
        "\n"\
        "test_text = dataset[0][dataset_text_field] if (formatting_func is None or not use_formatting_func) else formatting_func(dataset[0])[0]\n"\
        "chat_template = getattr(tokenizer, 'chat_template', None)\n"\
        "chat_template = '' if chat_template is None else chat_template\n"\
        "has_bos_token_already = (test_text.startswith(tokenizer.bos_token) or tokenizer.bos_token in chat_template) "\
        "if getattr(tokenizer, 'bos_token', None) is not None else False\n"\
        "add_special_tokens = False if has_bos_token_already else add_special_tokens\n\n"

        check_text = check_text.split("\n")
        check_text = "\n".join(" "*where + x for x in check_text)

        function = function.replace(replacer, check_text + replacer)
        exec(function, globals())

        exec(f"trl.trainer.sft_trainer.SFTTrainer.{function_name} = {function_name}", globals())
    pass

    # Patch train with fix_untrained_tokens
    function_name, replacer = "train", "if resume_from_checkpoint is False:"
    function = getsource(eval(f"trl.trainer.sft_trainer.SFTTrainer.{function_name}"))
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)

    check_text = \
    "\n"\
    "if self._inner_training_loop.__name__ != '_fast_inner_training_loop':\n"\
    "    raise RuntimeError(\n"\
    "       'Please do not edit specific areas of the Unsloth codebase or you will get CUDA segfaults.'\n"\
    "    )\n"\
    "pass\n"\
    "import subprocess, re, gc, numpy as np\n"\
    "a = np.array([0,])\n"\
    "try:\n"\
    "    a = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell = True)\n"\
    "    a = re.findall(rb'([\\d]{1,})[\\s]{1,}M', a)\n"\
    "    a = np.array([int(x.decode('utf-8'))/1024 for x in a])\n"\
    "except:\n"\
    "    if not torch.cuda.is_available():\n"\
    "        raise RuntimeError('Unsloth: We do not support AMD / Intel machines yet - it is a work in progress!')\n"\
    "if ((a - PRE_CHECK) >= 1).sum() > 1:\n"\
    "    raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')\n"\
    "for _ in range(3):\n"\
    "    gc.collect()\n"\
    "    torch.cuda.empty_cache()\n"\
    "pass\n"\
    "\n"\
    "fix_untrained_tokens(self.model, self.tokenizer, self.train_dataset, eps = 1e-16)\n\n"

    check_text = check_text.split("\n")
    check_text = "\n".join(" "*where + x for x in check_text)

    function = function.replace(replacer, check_text + replacer)
    exec(function, globals())

    exec(f"trl.trainer.sft_trainer.SFTTrainer.{function_name} = {function_name}", globals())
pass

patch_sft_trainer_tokenizer()
