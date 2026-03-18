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
import psutil

from unsloth_zoo.tokenizer_utils import (
    mean_of_trained_tokens,
    add_new_tokens,
    fix_untrained_tokens,
)
from unsloth_zoo.training_utils import (
    fix_zero_training_loss,
)

__all__ = [
    "load_correct_tokenizer",
    "fix_sentencepiece_tokenizer",
    "check_tokenizer",
    "add_new_tokens",
    "fix_sentencepiece_gguf",
    "get_tokenizer_info",
]


IGNORED_TOKENIZER_CHECKING = frozenset(
    (
        "CodeLlamaTokenizerFast",
        "CodeLlamaTokenizer",
    )
)


IGNORED_TOKENIZER_NAMES = [
    # Qwen Coder did not train on tool calling. Math did!
    "unsloth/Qwen2.5-Coder-1.5B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct",
]
IGNORED_TOKENIZER_NAMES = frozenset(
    [x.lower() for x in IGNORED_TOKENIZER_NAMES]
    + [x.lower() + "-bnb-4bit" for x in IGNORED_TOKENIZER_NAMES]
)
os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "\n".join(IGNORED_TOKENIZER_NAMES)

# Check environments
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
del keynames


def try_fix_tokenizer(tokenizer, prepend = True):
    if hasattr(tokenizer, "_tokenizer"):
        converted_tokenizer = tokenizer._tokenizer
    else:
        converted_tokenizer = convert_slow_tokenizer(tokenizer)

    tokenizer_string = converted_tokenizer.to_str()

    # Llama does _apple. Sometimes this is wrong!!
    prepend_text = '{"type":"Prepend","prepend":"▁"},'
    if not prepend and prepend_text in tokenizer_string:
        tokenizer_string = tokenizer_string.replace(prepend_text, "", 1)

    dir_names = dir(tokenizer)
    # Get eos_token, bos_token etc
    token_names = [x for x in dir_names if x.endswith("_token") and x.count("_") == 1]

    for token_name in token_names:
        token = getattr(tokenizer, token_name, None)
        if token is None:
            continue
        token_id = getattr(tokenizer, token_name + "_id", None)

        # Locate the token's id mapping in the string
        find_text = f'"id":{token_id},"content":"'
        start = tokenizer_string.find(find_text) + len(find_text)
        if start == -1:
            continue
        end = tokenizer_string.find('",', start)

        bad_token = tokenizer_string[start:end]
        # Check if token is the actual same one - if not, edit it
        if bad_token != token:
            bad_text = f'{find_text}{bad_token}",'
            good_text = f'{find_text}{token}",'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)

            # And replace vocab section
            bad_text = f'"{bad_token}":{token_id},'
            good_text = f'"{token}":{token_id},'
            tokenizer_string = tokenizer_string.replace(bad_text, good_text, 1)

    fixed_tokenizer = converted_tokenizer.from_str(tokenizer_string)
    return fixed_tokenizer


def get_sorted_dict(dictionary):
    sorted_keys = sorted(dictionary.values())
    inverted_dictionary = {value: key for key, value in dictionary.items()}

    sorted_dictionary = {}
    for key in sorted_keys:
        value = inverted_dictionary[key]
        sorted_dictionary[value] = key
    return sorted_dictionary


def convert_to_fast_tokenizer(
    slow_tokenizer,
    temporary_location = "_unsloth_sentencepiece_temp",
):
    is_fast = getattr(slow_tokenizer, "is_fast", False)
    if is_fast:
        return slow_tokenizer

    try:
        tokenizer_name = slow_tokenizer.__class__.__name__
        lowered_tokenizer_name = tokenizer_name.lower()
        if lowered_tokenizer_name.endswith("tokenizer"):
            class_name = lowered_tokenizer_name[: -len("tokenizer")]
            FastTokenizer = eval(
                f'__import__(f"transformers.models.{class_name}").{tokenizer_name}Fast'
            )
        else:
            FastTokenizer = PreTrainedTokenizerFast
    except:
        FastTokenizer = PreTrainedTokenizerFast

    # Get all arguments (bos_token, etc)
    docs = FastTokenizer.__doc__
    docs = docs[docs.find("Args:") :]
    args = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags = re.MULTILINE)
    args = [x for x in args if not x.endswith("_file")]

    # Also some missing maybe!
    docs = PreTrainedTokenizerFast.__doc__
    docs = docs[docs.find("Args:") :]
    args2 = re.findall(r"\n[\s]+([^\s]{1,}) \(", docs, flags = re.MULTILINE)
    args2 = [x for x in args2 if not x.endswith("_file")]
    args = list(set(args + args2))

    kwargs = {}
    for arg in args:
        kwargs[arg] = getattr(slow_tokenizer, arg, None)
    kwargs["tokenizer_object"] = try_fix_tokenizer(slow_tokenizer, prepend = True)
    fast_tokenizer = FastTokenizer(**kwargs)

    # Check if they're similar!
    sorted_slow_tokenizer = get_sorted_dict(slow_tokenizer.get_vocab())
    sorted_fast_tokenizer = get_sorted_dict(fast_tokenizer.get_vocab())

    check_vocab = sorted_slow_tokenizer == sorted_fast_tokenizer
    check_special = (
        slow_tokenizer.all_special_tokens == fast_tokenizer.all_special_tokens
    )

    # Failure so return slow_tokenizer
    if not check_vocab or not check_special:
        return slow_tokenizer

    # Now confirm if they match
    if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        # Maybe remove prepending of __apple?
        kwargs["tokenizer_object"] = try_fix_tokenizer(slow_tokenizer, prepend = False)
        fast_tokenizer = FastTokenizer(**kwargs)
        if not assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            # Failure :(
            return slow_tokenizer

    # Also tokenizer.model is missing!
    name = slow_tokenizer.name_or_path.replace("/", "_")
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)
    new_location = f"{temporary_location}/{name}"
    slow_tokenizer.save_pretrained(new_location)
    fast_tokenizer.save_pretrained(new_location)

    # Now load it!
    fast_tokenizer = AutoTokenizer.from_pretrained(new_location)
    if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
        return fast_tokenizer
    return slow_tokenizer


# Check Mistral chat template without BOS / EOS
mistral_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% if messages[1]['role'] == 'user' %}"
    "{{ '[INST] ' + messages[0]['content'] + ' ' + messages[1]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[2:] %}"
    "{% else %}"
    "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% endif %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

# Check Llama chat template without BOS / EOS
llama_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% if messages[1]['role'] == 'user' %}"
    "{{ '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' + messages[1]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[2:] %}"
    "{% else %}"
    "{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}"
    "{% set loop_messages = messages[1:] %}"
    "{% endif %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'].strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content'].strip() + ' ' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)


def assert_same_tokenization(slow_tokenizer, fast_tokenizer):
    # Get eos_token, bos_token etc
    if not hasattr(slow_tokenizer, "all_special_tokens"):
        return True
    dir_names = dir(slow_tokenizer)
    special_tokens = list(
        filter(
            None,
            (
                getattr(slow_tokenizer, x)
                for x in dir_names
                if x.endswith("_token") and x.count("_") == 1
            ),
        )
    )
    all_special_tokens = list(set(special_tokens + slow_tokenizer.all_special_tokens))

    # Remove replacement char for false positive
    replacement_char = b"\xc3\xaf\xc2\xbf\xc2\xbd".decode("utf-8")
    all_special_tokens = [x for x in all_special_tokens if x != replacement_char]

    check_chat_template1 = True
    check_chat_template2 = True
    check_chat_template3 = True

    check_chat_template = (
        check_chat_template1 and check_chat_template2 and check_chat_template3
    )

    try:
        string = (
            "\n".join(all_special_tokens)
            + "A quick brown fox jumps over the lazy dog!!\n\nHi</s>\n\n"
            + "".join(all_special_tokens)
        )
        check_special_tokens = (
            slow_tokenizer(string).input_ids == fast_tokenizer(string).input_ids
        )

        return check_chat_template and check_special_tokens
    except:
        if slow_tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
            return check_chat_template
        else:
            return False


def fix_sentencepiece_tokenizer(
    old_tokenizer,
    new_tokenizer,
    token_mapping,
    temporary_location = "_unsloth_sentencepiece_temp",
):
    try:
        from transformers.convert_slow_tokenizer import import_protobuf

        sentencepiece_model_pb2 = import_protobuf()
    except Exception as e:
        try:
            import google.protobuf
            from unsloth_zoo.utils import Version

            protobuf_version = Version(google.protobuf.__version__)
            if protobuf_version > Version("3.20.3"):
                raise RuntimeError(
                    f"Unsloth: Your protobuf version = {protobuf_version} is too new.\n"
                    f"Please downgrade via `pip install --force-reinstall protobuf==3.20.3`"
                )
        except:
            from transformers.utils import sentencepiece_model_pb2

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    if not os.path.isfile(f"{temporary_location}/tokenizer.model"):
        return new_tokenizer

    old_tokenizer.save_pretrained(temporary_location)

    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    tokenizer_file.ParseFromString(
        open(f"{temporary_location}/tokenizer.model", "rb").read()
    )

    new_tokenizer.save_pretrained(temporary_location)

    for old_token, new_token in token_mapping.items():
        ids = old_tokenizer([old_token], add_special_tokens = False).input_ids
        ids = ids[0]
        if len(ids) != 1:
            print(
                f"Skip mapping {old_token} to {new_token} since {new_token} is already in the tokenizer!"
            )
            continue
        ids = ids[0]
        try:
            tokenizer_piece = tokenizer_file.pieces[ids]
        except:
            continue
        assert tokenizer_piece.piece == old_token
        tokenizer_piece.piece = new_token

    with open(f"{temporary_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        temporary_location,
        eos_token = new_tokenizer.eos_token,
        pad_token = new_tokenizer.pad_token,
    )
    return tokenizer


def fix_sentencepiece_gguf(saved_location):
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

    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    if not os.path.isfile(f"{saved_location}/tokenizer.model"):
        return
    tokenizer_file.ParseFromString(
        open(f"{saved_location}/tokenizer.model", "rb").read()
    )
    sentence_piece_size = len(tokenizer_file.pieces)

    if not os.path.isfile(f"{saved_location}/added_tokens.json"):
        return
    with open(f"{saved_location}/added_tokens.json", "r", encoding = "utf-8") as file:
        added_tokens_json = json.load(file)
    if len(added_tokens_json) == 0:
        return

    added_tokens_json = dict(
        sorted(added_tokens_json.items(), key = lambda item: item[1])
    )
    new_size = sentence_piece_size + len(added_tokens_json)

    added_tokens_ids = np.array(list(added_tokens_json.values()))
    diff = np.diff(added_tokens_ids)
    if diff.min() != 1 or diff.max() != 1:
        return
    if added_tokens_ids.min() != sentence_piece_size:
        return

    logger.warning(
        f"Unsloth: Extending {saved_location}/tokenizer.model with added_tokens.json.\n"
        f"Originally tokenizer.model is of size ({sentence_piece_size}).\n"
        f"But we need to extend to sentencepiece vocab size ({new_size})."
    )
    new_tokens = deepcopy(tokenizer_file.pieces[-len(added_tokens_ids) :])
    for new_token, added_token in zip(new_tokens, added_tokens_json.keys()):
        new_token.piece = added_token.encode("utf-8")
        new_token.score = -1000.0
        new_token.type = SentencePieceTokenTypes.USER_DEFINED

    tokenizer_file.pieces.extend(new_tokens)

    with open(f"{saved_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())
    return


def _load_correct_tokenizer(
    tokenizer_name,
    model_max_length = None,
    padding_side = "right",
    token = None,
    trust_remote_code = False,
    cache_dir = "huggingface_tokenizers_cache",
    fix_tokenizer = True,
):
    if IS_COLAB_ENVIRONMENT:
        cache_dir = cache_dir
    elif IS_KAGGLE_ENVIRONMENT:
        cache_dir = os.path.join(KAGGLE_TMP, cache_dir)
    else:
        cache_dir = None

    slow_tokenizer = None
    try:
        slow_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length = model_max_length,
            padding_side = padding_side,
            token = token,
            trust_remote_code = trust_remote_code,
            use_fast = False,
            legacy = False,
            from_slow = True,
            cache_dir = cache_dir,
        )
    except:
        slow_tokenizer = None

    if type(slow_tokenizer) is bool:
        slow_tokenizer = None

    fast_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length = model_max_length,
        padding_side = padding_side,
        token = token,
        trust_remote_code = trust_remote_code,
        cache_dir = cache_dir,
    )

    if not fix_tokenizer or tokenizer_name in IGNORED_TOKENIZER_NAMES:
        return fast_tokenizer
    elif "mistral" in tokenizer_name.lower():
        return fast_tokenizer
    elif "phi-4" in tokenizer_name.lower():
        return fast_tokenizer
    elif slow_tokenizer is not None:
        if hasattr(fast_tokenizer, "add_bos_token") and hasattr(
            slow_tokenizer, "add_bos_token"
        ):
            fast_tokenizer.add_bos_token = slow_tokenizer.add_bos_token
        if hasattr(fast_tokenizer, "add_eos_token") and hasattr(
            slow_tokenizer, "add_eos_token"
        ):
            fast_tokenizer.add_eos_token = slow_tokenizer.add_eos_token

        if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            return fast_tokenizer
        else:
            logger.warning(
                f"Unsloth: Will load {tokenizer_name} as a legacy tokenizer."
            )
            return convert_to_fast_tokenizer(slow_tokenizer)
    else:
        return fast_tokenizer


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

    old_chat_template = getattr(tokenizer, "chat_template", None)

    if any(
        s in str(getattr(tokenizer, "name_or_path", "")).lower()
        for s in ["mistral", "qwen3guard"]
    ):
        chat_template = old_chat_template
    elif (
        old_chat_template is not None
        and "[/INST]" in old_chat_template
        and "[INST]" in old_chat_template
        and "bos_token" in old_chat_template
        and "eos_token" in old_chat_template
    ):
        chat_template = old_chat_template
    else:
        chat_template = fix_chat_template(tokenizer)
        if old_chat_template is not None and chat_template is None:
            raise RuntimeError(
                "Unsloth: Fixing chat template failed - please file a report immediately!"
            )

    tokenizer.chat_template = chat_template
    return tokenizer


def _find_end_position(template, endfor, endif):
    where_endfor = template.find(endfor)
    where_endif = template.find(endif)
    if where_endfor == where_endif == -1:
        return None
    elif where_endfor > where_endif:
        return endfor
    else:
        return endif


def _fix_chat_template(chat_template):
    endfor = "{% endfor %}"
    endif = "{% endif %}"
    chosen_end = _find_end_position(chat_template, endfor, endif)
    if chosen_end is None:
        endfor = "{%- endfor %}"
        endif = "{%- endif %}"
        chosen_end = _find_end_position(chat_template, endfor, endif)
    if chosen_end is None:
        return chat_template

    where = chat_template.find(chosen_end)
    after_endfor = chat_template[where + len(chosen_end) :]
    dash = "-" if chosen_end.startswith("{%-") else ""

    if (
        "{%" + dash + " if" not in after_endfor
        and "{%" + dash + " set " not in after_endfor
        and after_endfor.startswith("{{")
        and after_endfor.endswith("}}")
        and after_endfor.count("{{") == 1
        and after_endfor.count("}}") == 1
    ):
        after_endfor = (
            "{%" + dash + " if add_generation_prompt %}" + after_endfor + endif
        )
        chat_template = chat_template[: where + len(chosen_end)] + after_endfor
    return chat_template


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return None

    is_sharegpt = None
    try:
        messages = [{"role": "user", "content": "Who are you?"}]
        tokenizer.apply_chat_template(
            messages, add_generation_prompt = False, tokenize = False
        )
        is_sharegpt = False
    except:
        try:
            messages = [{"from": "human", "value": "Who are you?"}]
            tokenizer.apply_chat_template(
                messages, add_generation_prompt = False, tokenize = False
            )
            is_sharegpt = True
        except:
            is_sharegpt = None

    if is_sharegpt is None:
        return chat_template

    messages = [
        {"role": "user", "content": "Who are you?"}
        if not is_sharegpt
        else {"from": "human", "value": "Who are you?"}
    ]
    no = tokenizer.apply_chat_template(
        messages, add_generation_prompt = False, tokenize = False
    )
    yes = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, tokenize = False
    )

    if no == yes:
        if (
            "{% if add_generation_prompt %}" not in chat_template
            and "{%- if add_generation_prompt %}" not in chat_template
        ):
            new_chat_template = _fix_chat_template(chat_template)
            if (
                "{% if add_generation_prompt %}" not in new_chat_template
                and "{%- if add_generation_prompt %}" not in new_chat_template
            ):
                raise RuntimeError(
                    f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                    "does not have a {% if add_generation_prompt %} for generation purposes.\n"
                    f"Please file a bug report to the maintainers of `{tokenizer.name_or_path}` - thanks!"
                )
            else:
                logger.warning_once(
                    "Unsloth: We successfully patched the tokenizer to add a {% if add_generation_prompt %} to the chat_template.\n"
                    f"This is not a bug, but please notify the maintainers of `{tokenizer.name_or_path}` - thanks!"
                )
                chat_template = new_chat_template
        else:
            raise RuntimeError(
                f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                "has a {% if add_generation_prompt %} for generation purposes, but wasn't provided correctly.\n"
                "Please file a bug report immediately - thanks!"
            )
    return chat_template


def check_tokenizer(
    model,
    tokenizer,
    model_name = "unsloth/llama-2-7b-bnb-4bit",
    model_max_length = 4096,
    padding_side = "right",
    token = None,
    _reload = True,
):
    if tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
        return tokenizer

    max_embedding_size = model.model.embed_tokens.weight.shape[0]
    added_tokens_fast = tokenizer.added_tokens_decoder
    added_tokens_fast = {
        index: str(value) for index, value in added_tokens_fast.items()
    }
    sorted_keys = sorted(added_tokens_fast)
    added_tokens_fast = {key: added_tokens_fast[key] for key in sorted_keys}

    for j, index in enumerate(added_tokens_fast.keys()):
        if index >= max_embedding_size:
            bad_indices = list(added_tokens_fast.keys())[j:]
            bad_tokens = list(added_tokens_fast.values())[j:]
            if not _reload:
                added_tokens = [str(x) for x in tokenizer.added_tokens_decoder.values()]
                special_tokens = tokenizer.special_tokens_map
                import itertools

                special_tokens = frozenset(
                    itertools.chain.from_iterable(
                        [x] if type(x) is str else x for x in special_tokens.values()
                    )
                )
                can_be_removed1 = [x for x in bad_tokens if x not in special_tokens]
                can_be_removed2 = [
                    x
                    for x in can_be_removed1
                    if x in tokenizer._added_tokens_encoder.keys()
                ]
                can_be_removed = (len(can_be_removed1) == len(bad_tokens)) and (
                    len(can_be_removed2) == len(bad_tokens)
                )
                remove_generic = False
                try_mapper = []
                if not can_be_removed:
                    names = dir(tokenizer)
                    names = (
                        x for x in names if x.endswith("_token") and x.count("_") == 1
                    )
                    generic_tokens = [(x, getattr(tokenizer, x, None)) for x in names]
                    try_removal = []
                    for token in bad_tokens:
                        for name_token, check_token in generic_tokens:
                            if check_token == token:
                                try_removal.append(token)
                                try_mapper.append(name_token)
                    can_be_removed = len(try_removal) == len(bad_tokens)
                    if can_be_removed:
                        remove_generic = True
                    can_be_removed1 = bad_tokens

                if can_be_removed:
                    for j, bad_token in enumerate(can_be_removed1):
                        remove_id = tokenizer._added_tokens_encoder[bad_token]
                        del tokenizer._added_tokens_decoder[remove_id]
                        del tokenizer._added_tokens_encoder[bad_token]
                        if remove_generic and (try_removal[j] == bad_token):
                            setattr(tokenizer, try_mapper[j], None)
                            setattr(tokenizer, try_mapper[j] + "_id", None)
                    if max(tokenizer.added_tokens_decoder.keys()) < max_embedding_size:
                        logger.warning_once(
                            f"Unsloth loaded a broken tokenizer `{model_name}`, but managed to repair it!\n"
                            f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                            "We removed these bad tokens. If you think this is incorrect, fix your tokenizer first."
                        )
                        return convert_to_fast_tokenizer(tokenizer)

                raise RuntimeError(
                    f"Unsloth tried to load `{model_name}`, but cannot succeed.\n"
                    f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                    f"Fix your tokenizer since it'll perform out of bounds memory accesses."
                )

            if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
                cache_dir = "huggingface_tokenizers_cache"
            else:
                cache_dir = None

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_length = model_max_length,
                    padding_side = padding_side,
                    token = token,
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
                logger.warning_once(
                    "Unsloth: Tokenizer is most likely buggy, and Unsloth failed to repair it.\n"
                    "It will still work, but beware of out of bounds memory accesses.\n"
                    "Please file an issue on the model owner's repo about this issue."
                )
                return tokenizer
    return convert_to_fast_tokenizer(tokenizer)


def get_tokenizer_info(tokenizer) -> dict:
    """Return a concise diagnostic summary of a tokenizer instance.

    Collects key properties into a plain dict suitable for logging, debugging,
    or displaying in the Unsloth Studio UI. All fields are safe to access —
    missing attributes fall back to ``None`` rather than raising.

    Example output::

        {
            "name_or_path": "unsloth/Llama-3.2-1B-Instruct",
            "tokenizer_class": "LlamaTokenizerFast",
            "is_fast": True,
            "vocab_size": 128256,
            "added_tokens_count": 3,
            "model_max_length": 131072,
            "padding_side": "right",
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|finetune_right_pad_id|>",
            "unk_token": None,
            "has_chat_template": True,
            "special_tokens_count": 256,
        }

    Args:
        tokenizer: Any HuggingFace ``PreTrainedTokenizer`` or
                   ``PreTrainedTokenizerFast`` instance.

    Returns:
        A ``dict`` of tokenizer properties. Safe to serialize to JSON.
    """
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": getattr(tokenizer, "is_fast", False),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "added_tokens_count": len(getattr(tokenizer, "added_tokens_decoder", {})),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "has_chat_template": getattr(tokenizer, "chat_template", None) is not None,
        "special_tokens_count": len(getattr(tokenizer, "all_special_tokens", [])),
    }


import inspect
from inspect import getsource
import trl
import trl.trainer.sft_trainer
from trl.trainer.sft_trainer import *
from transformers.trainer import *

# ... (rest of file unchanged)
# Finally patch TRL tokenizer things -> moved to RL
# patch_sft_trainer_tokenizer()
