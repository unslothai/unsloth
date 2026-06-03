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
        if token_id is None:
            continue

        # Locate the token's id mapping in the string
        find_text = f'"id":{token_id},"content":"'
        find_pos = tokenizer_string.find(find_text)
        if find_pos == -1:
            continue
        start = find_pos + len(find_text)
        end = tokenizer_string.find('",', start)
        if end == -1:
            continue

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
    check_chat_template = (
        check_chat_template1 and check_chat_template2 and check_chat_template3
    )

    # Try special tokens
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
        # For eg see https://github.com/unslothai/unsloth/issues/292
        # Sometimes tokenizer has weird tokens, causing a combined tokenization to fail.
        # [TODO] We temporarily disable this for CodeLlama tokenizers
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
    # From https://github.com/google/sentencepiece/issues/121
    # We need to manually edit the sentencepiece tokenizer!
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
            # This will only work for older SentencePiece versions <= 3.20.3
            from transformers.utils import sentencepiece_model_pb2

    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    # Check if tokenizer.model exists
    if not os.path.isfile(f"{temporary_location}/tokenizer.model"):
        return new_tokenizer

    # First save the old tokenizer
    old_tokenizer.save_pretrained(temporary_location)

    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    tokenizer_file.ParseFromString(
        open(f"{temporary_location}/tokenizer.model", "rb").read()
    )

    # Now save the new tokenizer
    new_tokenizer.save_pretrained(temporary_location)

    # Now correct the old tokenizer's .model file
    for old_token, new_token in token_mapping.items():
        ids = old_tokenizer([old_token], add_special_tokens = False).input_ids
        ids = ids[0]
        if len(ids) != 1:
            # Skip this token!
            print(
                f"Skip mapping {old_token} to {new_token} since {new_token} is already in the tokenizer!"
            )
            continue
        ids = ids[0]
        # [TODO] Hack for Starling - try except
        try:
            tokenizer_piece = tokenizer_file.pieces[ids]
        except:
            continue
        assert tokenizer_piece.piece == old_token
        tokenizer_piece.piece = new_token

    # And now write it
    with open(f"{temporary_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())

    # And load it!
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        temporary_location,
        eos_token = new_tokenizer.eos_token,
        pad_token = new_tokenizer.pad_token,
    )
    return tokenizer


def fix_sentencepiece_gguf(saved_location):
    """
    Fixes sentencepiece tokenizers which did not extend the vocabulary with
    user defined tokens.
    Inspiration from https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py

    Also fixes special tokens (e.g. Gemma 3's <start_of_turn>/<end_of_turn>) that are
    already present in the sentencepiece model but are incorrectly typed as NORMAL instead
    of CONTROL. This causes them to be written to GGUF with token_type=1 (NORMAL) instead
    of token_type=3 (CONTROL), which breaks chat inference in llama.cpp since parse_special
    only matches CONTROL tokens.
    """
    from copy import deepcopy
    import sys

    try:
        from transformers.convert_slow_tokenizer import import_protobuf

        sys.modules.setdefault(
            "transformers.utils.sentencepiece_model_pb2",
            import_protobuf(),
        )
    except Exception:
        pass
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

    # Load tokenizer.model
    tokenizer_file = sentencepiece_model_pb2.ModelProto()
    if not os.path.isfile(f"{saved_location}/tokenizer.model"):
        return
    tokenizer_file.ParseFromString(
        open(f"{saved_location}/tokenizer.model", "rb").read()
    )
    sentence_piece_size = len(tokenizer_file.pieces)

    # Build a set of token IDs that are marked as special in tokenizer.json.
    # These tokens should use CONTROL type in the sentencepiece model so that
    # llama.cpp writes them as CONTROL (type=3) in the GGUF token_type array.
    special_token_ids = set()
    if os.path.isfile(f"{saved_location}/tokenizer.json"):
        with open(f"{saved_location}/tokenizer.json", "r", encoding = "utf-8") as f:
            tokenizer_json = json.load(f)
        for entry in tokenizer_json.get("added_tokens", []):
            token_id = entry.get("id")
            if entry.get("special", False) and isinstance(token_id, int):
                special_token_ids.add(token_id)

    # Fix existing sentencepiece tokens that are marked as special in tokenizer.json
    # but have the wrong type (NORMAL instead of CONTROL) in the sentencepiece model.
    patched = 0
    for token_id in special_token_ids:
        if 0 <= token_id < sentence_piece_size:
            piece = tokenizer_file.pieces[token_id]
            if piece.type == SentencePieceTokenTypes.NORMAL:
                piece.type = SentencePieceTokenTypes.CONTROL
                patched += 1
    if patched > 0:
        logger.warning(
            f"Unsloth: Patched {patched} special token(s) in {saved_location}/tokenizer.model "
            f"from NORMAL to CONTROL type so llama.cpp / GGUF chat inference works correctly."
        )

    # Load added_tokens_json
    if not os.path.isfile(f"{saved_location}/added_tokens.json"):
        if patched > 0:
            with open(f"{saved_location}/tokenizer.model", "wb") as file:
                file.write(tokenizer_file.SerializeToString())
        return
    with open(f"{saved_location}/added_tokens.json", "r", encoding = "utf-8") as file:
        added_tokens_json = json.load(file)
    if len(added_tokens_json) == 0:
        if patched > 0:
            with open(f"{saved_location}/tokenizer.model", "wb") as file:
                file.write(tokenizer_file.SerializeToString())
        return

    added_tokens_json = dict(
        sorted(added_tokens_json.items(), key = lambda item: item[1])
    )
    new_size = sentence_piece_size + len(added_tokens_json)

    # Confirm added_tokens_json is correct
    added_tokens_ids = np.array(list(added_tokens_json.values()))
    _real_added_tokens_ids = added_tokens_ids
    if len(added_tokens_ids) < 2:
        added_tokens_ids = np.array([sentence_piece_size, sentence_piece_size + 1])
    diff = np.diff(added_tokens_ids)
    if diff.min() != 1 or diff.max() != 1:
        if patched > 0:
            with open(f"{saved_location}/tokenizer.model", "wb") as file:
                file.write(tokenizer_file.SerializeToString())
        return
    added_tokens_ids = _real_added_tokens_ids
    if added_tokens_ids.min() != sentence_piece_size:
        if patched > 0:
            with open(f"{saved_location}/tokenizer.model", "wb") as file:
                file.write(tokenizer_file.SerializeToString())
        return

    # Edit sentence piece tokens with added_tokens_json
    logger.warning(
        f"Unsloth: Extending {saved_location}/tokenizer.model with added_tokens.json.\n"
        f"Originally tokenizer.model is of size ({sentence_piece_size}).\n"
        f"But we need to extend to sentencepiece vocab size ({new_size})."
    )
    new_tokens = deepcopy(tokenizer_file.pieces[-len(added_tokens_ids) :])
    for new_token, added_token_str in zip(new_tokens, added_tokens_json.keys()):
        added_token_id = added_tokens_json[added_token_str]
        new_token.piece = added_token_str.encode("utf-8")
        new_token.score = -1000.0
        # Use CONTROL type for tokens marked as special in tokenizer.json,
        # otherwise fall back to USER_DEFINED.
        if added_token_id in special_token_ids:
            new_token.type = SentencePieceTokenTypes.CONTROL
        else:
            new_token.type = SentencePieceTokenTypes.USER_DEFINED

    tokenizer_file.pieces.extend(new_tokens)

    with open(f"{saved_location}/tokenizer.model", "wb") as file:
        file.write(tokenizer_file.SerializeToString())

    # Add padding tokens
    # actual_vocab_size = model.config.vocab_size
    # padding = actual_vocab_size - len(tokenizer_file.pieces)
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
        # /tmp of Kaggle seems has a 80GB limit!
        # Let's utilize them
        cache_dir = os.path.join(KAGGLE_TMP, cache_dir)
    else:
        cache_dir = None

    # Try loading the slow tokenizer. If it fails, then try Fast only
    # Mainly to solve Deepseek models with no tokenizer.model file
    slow_tokenizer = None
    try:
        slow_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length = model_max_length,
            padding_side = padding_side,
            token = token,
            trust_remote_code = trust_remote_code,
            # Cannot just use use_fast = False as per https://twitter.com/danielhanchen/status/1789659394302718373
            use_fast = False,
            legacy = False,
            from_slow = True,
            cache_dir = cache_dir,
        )
    except:
        slow_tokenizer = None
        # print(
        #     f"Unsloth: {tokenizer_name} has no tokenizer.model file.\n"\
        #     "Just informing you about this - this is not a critical error."
        # )
    # Unsure why this occurs!
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
    # Ignore Mistral ones - they're a bit weird to handle!
    elif "mistral" in tokenizer_name.lower():
        return fast_tokenizer
    # Ignore Phi-4 ones as well
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

        # Confirm if slow and fast are equivalent!
        if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            return fast_tokenizer
        else:
            logger.warning(
                f"Unsloth: Will load {tokenizer_name} as a legacy tokenizer."
            )
            return convert_to_fast_tokenizer(slow_tokenizer)
        pass
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

    ### 1. Fixup tokenizer's chat_template
    old_chat_template = getattr(tokenizer, "chat_template", None)

    # Ignore mistral type models since they don't have an add_generation_prompt
    if any(
        s in str(getattr(tokenizer, "name_or_path", "")).lower()
        for s in ["mistral", "qwen3guard"]
    ):
        chat_template = old_chat_template

    # Also check Llama-2 old style models
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
        pass

    tokenizer.chat_template = chat_template
    return tokenizer


# All four Jinja whitespace-control variants of endfor/endif:
#   {% endfor %}    {%- endfor %}    {% endfor -%}    {%- endfor -%}
_RE_ENDFOR = re.compile(r"\{%(-?)\s*endfor\s*(-?)%\}")
_RE_ENDIF = re.compile(r"\{%(-?)\s*endif\s*(-?)%\}")
_RE_JINJA_COMMENT = re.compile(r"\{#.*?#\}", flags = re.DOTALL)


def _find_end_position(template, endfor = None, endif = None):
    """Rightmost {% endfor %}/{% endif %} (any dash variant), as a dict
    with start/end/text/dash_left/dash_right. Tokens inside Jinja comments
    are ignored. `endfor`/`endif` kwargs kept for back-compat, ignored."""
    # Space-pad comments so positions still map 1:1 to the original.
    scrubbed = _RE_JINJA_COMMENT.sub(lambda m: " " * len(m.group(0)), template)
    endfor_matches = list(_RE_ENDFOR.finditer(scrubbed))
    endif_matches = list(_RE_ENDIF.finditer(scrubbed))
    last_endfor = endfor_matches[-1] if endfor_matches else None
    last_endif = endif_matches[-1] if endif_matches else None
    candidates = [m for m in (last_endfor, last_endif) if m is not None]
    if not candidates:
        return None
    m = max(candidates, key = lambda x: x.end())
    return {
        "start": m.start(),
        "end": m.end(),
        "text": m.group(0),
        "dash_left": bool(m.group(1)),
        "dash_right": bool(m.group(2)),
    }


def _template_ends_with_toplevel_for(chat_template):
    """Return True if the last structural node at the template's top level is
    a For (message-iteration) loop, ignoring trailing pure-whitespace Output
    nodes. Unwraps benign outer-If guards (no else branch, not testing
    add_generation_prompt) so that templates like
    ``{% if messages %}{% for ... %}{% endfor %}{% endif %}`` are still
    repairable. Rejects real structural wrappers (e.g. Qwen3-Guard with
    else branches)."""
    try:
        import jinja2
        import jinja2.nodes

        ast = jinja2.Environment().parse(chat_template)
    except Exception:
        return False

    def _last_structural(nodes):
        for node in reversed(nodes):
            if isinstance(node, jinja2.nodes.Output):
                only_ws = all(
                    isinstance(child, jinja2.nodes.TemplateData)
                    and child.data.strip() == ""
                    for child in node.nodes
                )
                if only_ws:
                    continue
            return node
        return None

    node = _last_structural(ast.body)
    while isinstance(node, jinja2.nodes.If) and not node.else_:
        names = []
        if isinstance(node.test, jinja2.nodes.Name):
            names.append(node.test)
        names.extend(node.test.find_all(jinja2.nodes.Name))
        if any(n.name == "add_generation_prompt" for n in names):
            break
        node = _last_structural(node.body)

    return isinstance(node, jinja2.nodes.For)


def _if_body_emits_content(if_node):
    """True if the If's body contains any Output node (directly or nested).
    Distinguishes a real generation block from a header guard that only
    does `{% set ... %}`."""
    import jinja2.nodes

    for node in if_node.body:
        if isinstance(node, jinja2.nodes.Output):
            return True
        if any(
            isinstance(d, jinja2.nodes.Output)
            for d in node.find_all(jinja2.nodes.Output)
        ):
            return True
    return False


def _has_add_generation_prompt_block(chat_template):
    """True if the template has a *positive* `{% if add_generation_prompt %}`
    gate whose body emits output. Rejects header guards like
    `{% if not add_generation_prompt is defined %}{% set ... %}{% endif %}`
    that reference the name but emit nothing. AST-based; string-scan
    fallback if Jinja fails to parse."""
    try:
        import jinja2
        import jinja2.nodes

        ast = jinja2.Environment().parse(chat_template)
    except Exception:
        return "if add_generation_prompt" in chat_template and "%}" in chat_template
    for if_node in ast.find_all(jinja2.nodes.If):
        test = if_node.test
        # Reject negated gates: `{% if not add_generation_prompt %}` fires
        # when agp=False, so it's not a generation block even if it emits.
        if isinstance(test, jinja2.nodes.Not):
            continue
        # find_all skips the test root, so check bare Name tests explicitly.
        references_agp = False
        if isinstance(test, jinja2.nodes.Name) and test.name == "add_generation_prompt":
            references_agp = True
        else:
            for name_node in test.find_all(jinja2.nodes.Name):
                if name_node.name == "add_generation_prompt":
                    references_agp = True
                    break
        if references_agp and _if_body_emits_content(if_node):
            return True
    return False


# Sentinels for _derive_assistant_prefix_by_render. Diverge at char 0 so
# commonprefix can't absorb them; long random tail makes collision with real
# template literals negligible (see T18).
_RENDER_DIFF_SENTINEL_A = "AAAA_0123456789_UNSLOTH_RENDER_DIFF_SENTINEL"
_RENDER_DIFF_SENTINEL_B = "BBBB_0123456789_UNSLOTH_RENDER_DIFF_SENTINEL"
_RENDER_DIFF_SENTINEL_C = "CCCC_0123456789_UNSLOTH_RENDER_DIFF_SENTINEL"


def _derive_assistant_prefix_by_render(chat_template, is_sharegpt = False):
    """Return the assistant-turn prefix the template emits, derived by
    rendering two dialogs that differ only in assistant content: the common
    prefix of their tails (after the base [user]-only render) is what the
    template emits for an assistant turn. None if any guard fails.

    Works for Llama-3 / Gemma / Phi-3 and other non-ChatML shapes; the
    template is its own ground truth.

    Known limitation: an `eos-on-non-last` pattern (turn-end sentinel only
    emitted for non-last messages) would produce a consistent but wrong
    prefix that `_validate_patched_template` can't catch. No real-world
    template is known to use this.
    """
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except Exception:
        return None

    if is_sharegpt:
        base_msgs = [{"from": "human", "value": "Hi"}]
        sent_a_msgs = base_msgs + [{"from": "gpt", "value": _RENDER_DIFF_SENTINEL_A}]
        sent_b_msgs = base_msgs + [{"from": "gpt", "value": _RENDER_DIFF_SENTINEL_B}]
        # User-role cross-check (Guard C below).
        sent_c_msgs = base_msgs + [{"from": "human", "value": _RENDER_DIFF_SENTINEL_C}]
    else:
        base_msgs = [{"role": "user", "content": "Hi"}]
        sent_a_msgs = base_msgs + [
            {"role": "assistant", "content": _RENDER_DIFF_SENTINEL_A}
        ]
        sent_b_msgs = base_msgs + [
            {"role": "assistant", "content": _RENDER_DIFF_SENTINEL_B}
        ]
        sent_c_msgs = base_msgs + [{"role": "user", "content": _RENDER_DIFF_SENTINEL_C}]

    # Strip trailing whitespace/comments after the last endfor/endif: they
    # appear after the message loop and would break Guard A. The splice in
    # `_fix_chat_template` drops them too.
    probe_template = chat_template
    end = _find_end_position(chat_template)
    if end is not None:
        after = chat_template[end["end"] :]
        if _RE_JINJA_COMMENT.sub("", after).strip() == "":
            probe_template = chat_template[: end["end"]]

    # Sandboxed: probe renders at load time, before user calls
    # apply_chat_template. SandboxedEnvironment blocks attribute-chain exploits.
    try:
        env = SandboxedEnvironment(
            autoescape = False,
            keep_trailing_newline = True,
        )
        tmpl = env.from_string(probe_template)
        out_base = tmpl.render(messages = base_msgs, add_generation_prompt = False)
        out_a = tmpl.render(messages = sent_a_msgs, add_generation_prompt = False)
        out_b = tmpl.render(messages = sent_b_msgs, add_generation_prompt = False)
    except Exception:
        return None

    # Best-effort: alternation-enforcing templates (e.g. Gemma's
    # raise_exception) fail on [user, user]; that's a positive signal
    # for Guard C, not a probe failure.
    out_user_c = None
    try:
        out_user_c = tmpl.render(messages = sent_c_msgs, add_generation_prompt = False)
    except Exception:
        pass

    # Guard A: assistant renders extend base (no reordering).
    if not (out_a.startswith(out_base) and out_b.startswith(out_base)):
        return None

    tail_a = out_a[len(out_base) :]
    tail_b = out_b[len(out_base) :]
    if not tail_a or not tail_b:
        return None

    prefix = os.path.commonprefix([tail_a, tail_b])

    # Guard B: divergence is exactly at the content-insertion site.
    if not (
        tail_a[len(prefix) :].startswith(_RENDER_DIFF_SENTINEL_A)
        and tail_b[len(prefix) :].startswith(_RENDER_DIFF_SENTINEL_B)
    ):
        return None

    # Guard C: reject if a [user, user] render also emits the same prefix
    # (role-insensitive template, e.g. `{% set greeting='Hi' %}...`).
    if out_user_c is not None and out_user_c.startswith(out_base):
        tail_c = out_user_c[len(out_base) :]
        if tail_c.startswith(prefix) and prefix != "":
            return None

    if not prefix:
        return None

    return prefix


def _fix_chat_template(chat_template, is_sharegpt = False):
    # Fast path: already has an {% if add_generation_prompt %} block, nothing
    # to do. This catches cases the old string-based check would miss (e.g.
    # templates that use {%- if add_generation_prompt -%} with both-side dash,
    # or that sneak the block into a nested If/For).
    if _has_add_generation_prompt_block(chat_template):
        return chat_template

    end = _find_end_position(chat_template)
    if end is None:
        return chat_template

    after_endfor = chat_template[end["end"] :]
    dash_l = "-" if end["dash_left"] else ""
    dash_r = "-" if end["dash_right"] else ""
    open_tag = lambda body: "{%" + dash_l + " " + body + " " + dash_r + "%}"

    # Case 1 (pre-existing base case): template ends with a single trailing
    # {{ expr }} that is the generation prefix. Wrap it in an
    # {% if add_generation_prompt %} ... {% endif %}.
    if (
        "{%" + dash_l + " if" not in after_endfor
        and "{%" + dash_l + " set " not in after_endfor
        and after_endfor.startswith("{{")
        and after_endfor.endswith("}}")
        and after_endfor.count("{{") == 1
        and after_endfor.count("}}") == 1
    ):
        wrapped = (
            open_tag("if add_generation_prompt") + after_endfor + open_tag("endif")
        )
        return chat_template[: end["end"]] + wrapped

    # Case 2 (GH#4150): template ends at {% endfor %} with only whitespace
    # or comments left. Inject an {% if add_generation_prompt %} block with
    # the assistant prefix derived by render-diff. The top-level-For gate
    # keeps us out of outer-If wrappers (e.g. Qwen3-Guard).
    if _RE_JINJA_COMMENT.sub(
        "", after_endfor
    ).strip() == "" and _template_ends_with_toplevel_for(chat_template):
        # No redundant "agp not in scrubbed" check: the fast path already
        # confirmed no *positive* block, and a mere reference (header
        # guard) should still get repaired.
        assistant_prefix = _derive_assistant_prefix_by_render(
            chat_template, is_sharegpt
        )
        # Dual-probe: dict/list callers don't know the shape up front.
        if assistant_prefix is None and not is_sharegpt:
            assistant_prefix = _derive_assistant_prefix_by_render(
                chat_template, is_sharegpt = True
            )
        if assistant_prefix is None:
            return chat_template
        # Escape for a double-quoted Jinja string literal.
        escaped = (
            assistant_prefix.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        generation_block = (
            open_tag("if add_generation_prompt")
            + '{{ "'
            + escaped
            + '" }}'
            + open_tag("endif")
        )
        return chat_template[: end["end"]] + generation_block

    return chat_template


def _is_strict_chat_template_mode():
    """Opt-in strict mode restores the pre-warn RuntimeError behavior."""
    val = os.environ.get("UNSLOTH_STRICT_CHAT_TEMPLATE", "0")
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _name_is_local_path(name_or_path):
    """True if name_or_path refers to an existing local directory. Used to
    tailor the warning message: for local paths the user cannot 'file a bug
    report to the maintainers of <path>' since that path is their own."""
    if not name_or_path:
        return False
    try:
        return os.path.isdir(str(name_or_path))
    except Exception:
        return False


def _format_chat_template_message(
    name_or_path,
    repaired,
    has_generation_block = False,
    local_path_source = None,
    strict = False,
):
    """Build a user-facing warning/error message that points at the right
    responsible party (user's downstream tool vs. upstream model maintainer)."""
    local = _name_is_local_path(
        local_path_source if local_path_source is not None else name_or_path
    )
    if local:
        source_hint = (
            "This tokenizer was loaded from a local path. The likely cause is a "
            "downstream tool (LlamaFactory, Axolotl, etc.) that re-serialized "
            "the tokenizer during save and stripped the generation-prompt "
            "block. Either re-save with the original template, or set "
            "`tokenizer.chat_template` manually before loading."
        )
    else:
        source_hint = (
            "The chat_template shipped with `{name}` appears incomplete. "
            "Consider filing a bug report with the model maintainers."
        ).format(name = name_or_path)
    strict_suffix = (
        ""
        if strict
        else (" Set UNSLOTH_STRICT_CHAT_TEMPLATE=1 to raise instead of warn.")
    )
    if repaired:
        return (
            "Unsloth: Patched the chat_template on `{name}` to add a "
            "{{% if add_generation_prompt %}} block. {hint}"
        ).format(name = name_or_path, hint = source_hint)
    if has_generation_block:
        return (
            "Unsloth: The tokenizer `{name}` has a "
            "{{% if add_generation_prompt %}} block, but it does not change "
            "the rendered output. {hint}{suffix}"
        ).format(name = name_or_path, hint = source_hint, suffix = strict_suffix)
    load_clause = (
        "Loading is blocked in strict mode."
        if strict
        else "The model will still load, but "
        "`apply_chat_template(add_generation_prompt=True)` may not produce a "
        "correct assistant-turn marker."
    )
    return (
        "Unsloth: The tokenizer `{name}` does not have a "
        "{{% if add_generation_prompt %}} block for generation purposes, and "
        "automatic repair was not possible. {load_clause} {hint}{suffix}"
    ).format(
        name = name_or_path,
        load_clause = load_clause,
        hint = source_hint,
        suffix = strict_suffix,
    )


def _validate_patched_template(tokenizer, patched_template, is_sharegpt):
    """Render the just-patched template with and without
    add_generation_prompt, and confirm the patched output responds to the
    flag by appending (not replacing) content. Returns True if validation
    passes."""
    msgs = (
        [{"from": "human", "value": "Hi"}]
        if is_sharegpt
        else [{"role": "user", "content": "Hi"}]
    )
    original = getattr(tokenizer, "chat_template", None)
    try:
        try:
            tokenizer.chat_template = patched_template
        except Exception:
            return False  # read-only tokenizer, skip validation
        try:
            yes = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt = True,
                tokenize = False,
            )
            no = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt = False,
                tokenize = False,
            )
        except Exception:
            return False
    finally:
        try:
            tokenizer.chat_template = original
        except Exception:
            pass  # best-effort restore
    # Contract after a successful repair: the two renders differ, and the
    # "yes" render is a strict extension of the "no" render (we only
    # appended content inside the new add_generation_prompt block).
    return yes != no and yes.startswith(no)


def _repair_string_template(tokenizer, chat_template, is_sharegpt):
    """Core string-template repair. Returns the repaired template on success,
    or None if repair was not possible / failed validation."""
    candidate = _fix_chat_template(chat_template, is_sharegpt = is_sharegpt)
    if not _has_add_generation_prompt_block(candidate):
        return None
    # Validate with the caller's is_sharegpt first. If that fails, the
    # dual-probe in _fix_chat_template may have fallen back to the other
    # schema internally -- try validating with the opposite schema before
    # giving up.
    if _validate_patched_template(tokenizer, candidate, is_sharegpt):
        return candidate
    if _validate_patched_template(tokenizer, candidate, not is_sharegpt):
        return candidate
    return None


def _fix_chat_template_for_tokenizer(tokenizer, chat_template):
    """Entry point for a string chat_template. Runs the no==yes diagnostic,
    attempts repair if needed, and returns the (possibly patched) template.

    On repair failure, the behavior is controlled by
    UNSLOTH_STRICT_CHAT_TEMPLATE: warn + return original (default) or raise
    RuntimeError (strict)."""
    name = getattr(tokenizer, "name_or_path", "unknown")
    source_path = getattr(tokenizer, "_source_path", name)

    # Detect ShareGPT vs HF style by probing apply_chat_template.
    is_sharegpt = None
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Who are you?"}],
            add_generation_prompt = False,
            tokenize = False,
        )
        is_sharegpt = False
    except Exception:
        try:
            tokenizer.apply_chat_template(
                [{"from": "human", "value": "Who are you?"}],
                add_generation_prompt = False,
                tokenize = False,
            )
            is_sharegpt = True
        except Exception:
            is_sharegpt = None

    if is_sharegpt is None:
        return chat_template

    messages = (
        [{"from": "human", "value": "Who are you?"}]
        if is_sharegpt
        else [{"role": "user", "content": "Who are you?"}]
    )
    try:
        no = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = False,
            tokenize = False,
        )
        yes = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
        )
    except Exception:
        return chat_template

    if no != yes:
        # Template already responds to the flag; leave as is.
        return chat_template

    # no == yes: template ignores add_generation_prompt. Try to repair.
    if _has_add_generation_prompt_block(chat_template):
        # Template has the block but it does not change output. This is the
        # "wasn't provided correctly" case from the pre-warn code path.
        strict = _is_strict_chat_template_mode()
        msg = _format_chat_template_message(
            name,
            repaired = False,
            has_generation_block = True,
            local_path_source = source_path,
            strict = strict,
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning_once(msg)
        return chat_template

    repaired = _repair_string_template(tokenizer, chat_template, is_sharegpt)
    if repaired is not None:
        logger.warning_once(
            _format_chat_template_message(
                name,
                repaired = True,
                local_path_source = source_path,
            )
        )
        return repaired

    strict = _is_strict_chat_template_mode()
    msg = _format_chat_template_message(
        name,
        repaired = False,
        local_path_source = source_path,
        strict = strict,
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning_once(msg)
    return chat_template


class _VariantTokenizerProxy:
    """Single-variant view of a multi-variant tokenizer. Routes each variant
    through `_fix_chat_template_for_tokenizer` so the full contract
    (is_sharegpt probe, no==yes, warn/strict, `_validate_patched_template`)
    applies instead of jumping straight to structural repair.

    `apply_chat_template` swaps `base.chat_template` to the variant before
    calling so tokenizer globals (bos_token, filters, raise_exception) are
    preserved; falls back to bare Jinja for read-only stubs.
    """

    def __init__(self, base_tokenizer, variant_template, variant_label = ""):
        self._base = base_tokenizer
        self._template = variant_template
        base_name = getattr(base_tokenizer, "name_or_path", "unknown")
        self._source_path = base_name
        self.name_or_path = (
            f"{base_name} ({variant_label})" if variant_label else base_name
        )

    @property
    def chat_template(self):
        return self._template

    @chat_template.setter
    def chat_template(self, value):
        self._template = value

    def apply_chat_template(self, *args, **kwargs):
        base_original = getattr(self._base, "chat_template", None)
        swapped = False
        try:
            try:
                self._base.chat_template = self._template
                swapped = True
            except Exception:
                swapped = False
            if swapped:
                return self._base.apply_chat_template(*args, **kwargs)
            # Read-only base: fall back to sandboxed Jinja.
            from jinja2.sandbox import SandboxedEnvironment

            env = SandboxedEnvironment(
                autoescape = False,
                keep_trailing_newline = True,
            )
            messages = args[0] if args else kwargs.get("messages", [])
            add_generation_prompt = kwargs.get("add_generation_prompt", False)
            return env.from_string(self._template).render(
                messages = messages,
                add_generation_prompt = add_generation_prompt,
            )
        finally:
            if swapped:
                try:
                    self._base.chat_template = base_original
                except Exception:
                    pass  # best-effort restore


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return None

    # Multi-variant dict (e.g. Hermes-3 {default, tool_use}): route each
    # variant through the full repair contract via _VariantTokenizerProxy.
    if isinstance(chat_template, dict):
        fixed = {}
        for key, tmpl in chat_template.items():
            if not isinstance(tmpl, str):
                fixed[key] = tmpl
                continue
            proxy = _VariantTokenizerProxy(
                tokenizer, tmpl, variant_label = f"variant={key!r}"
            )
            fixed[key] = _fix_chat_template_for_tokenizer(proxy, tmpl)
        return fixed

    # List-of-dicts form (older HF multi-template style).
    if isinstance(chat_template, list):
        fixed = []
        for item in chat_template:
            if not isinstance(item, dict) or "template" not in item:
                fixed.append(item)
                continue
            tmpl = item["template"]
            if not isinstance(tmpl, str):
                fixed.append(item)
                continue
            label = f"variant={item.get('name', '?')!r}"
            proxy = _VariantTokenizerProxy(tokenizer, tmpl, variant_label = label)
            new_tmpl = _fix_chat_template_for_tokenizer(proxy, tmpl)
            if new_tmpl is tmpl or new_tmpl == tmpl:
                fixed.append(item)
            else:
                fixed.append({**item, "template": new_tmpl})
        return fixed

    return _fix_chat_template_for_tokenizer(tokenizer, chat_template)


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
                can_be_removed2 = [
                    x
                    for x in can_be_removed1
                    if x in tokenizer._added_tokens_encoder.keys()
                ]

                # Check of extra tokens can in fact we removed!
                can_be_removed = (len(can_be_removed1) == len(bad_tokens)) and (
                    len(can_be_removed2) == len(bad_tokens)
                )

                # Check if sep_token or other generic types
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

                    # Recheck!
                    can_be_removed = len(try_removal) == len(bad_tokens)
                    if can_be_removed:
                        remove_generic = True
                    can_be_removed1 = bad_tokens

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
                    # Confirm 1 more time!
                    if max(tokenizer.added_tokens_decoder.keys()) < max_embedding_size:
                        logger.warning_once(
                            f"Unsloth loaded a broken tokenizer `{model_name}`, but managed to repair it!\n"
                            f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                            "We removed these bad tokens. If you think this is incorrect, fix your tokenizer first."
                        )
                        return convert_to_fast_tokenizer(tokenizer)

                # :( Failure
                raise RuntimeError(
                    f"Unsloth tried to load `{model_name}`, but cannot succeed.\n"
                    f"Tokens {bad_tokens} with ids {bad_indices} exceeds the max vocab size of {max_embedding_size}.\n"
                    f"Fix your tokenizer since it'll perform out of bounds memory accesses."
                )

            if IS_COLAB_ENVIRONMENT or IS_KAGGLE_ENVIRONMENT:
                cache_dir = "huggingface_tokenizers_cache"
            else:
                cache_dir = None

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
            "tokenizer_class": "PreTrainedTokenizerFast",
            "is_fast": True,
            "vocab_size": 128000,
            "added_tokens_count": 256,
            "model_max_length": 131072,
            "padding_side": "right",
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|eot_id|>",
            "pad_token": "<|finetune_right_pad_id|>",
            "unk_token": None,
            "has_chat_template": True,
            "special_tokens_count": 3,
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

try:
    from trl.trainer.sft_trainer import neftune_post_forward_hook
except:

    def neftune_post_forward_hook(module, input, output):
        """
        Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
        torch.nn.Embedding layers. This method is slightly adapted from the original source code
        that can be found here: https://github.com/neelsjain/NEFTune

        Simply add it to your model as follows:
        ```python
        model = ...
        model.embed_tokens.neftune_noise_alpha = 0.1
        model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
        ```

        Args:
            module (`torch.nn.Module`):
                The embedding module where the hook is attached. Note that you need to set
                `module.neftune_noise_alpha` to the desired noise alpha value.
            input (`torch.Tensor`):
                The input tensor to the model.
            output (`torch.Tensor`):
                The output tensor of the model (i.e. the embeddings).
        """
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2))
            mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
        return output


def patch_sft_trainer_tokenizer():
    """
    Patches the trainer with changes
    """
    try:
        sft_trainer = eval(f"trl.trainer.sft_trainer.SFTTrainer")
    except:
        return
    all_imports = dir(trl.trainer.sft_trainer)
    # Make typing names available to the exec'd source bodies. TRL >= 1.x
    # type-hints _prepare_dataset / _prepare_non_packed_dataloader with
    # `Union[...]` and friends; without these imports in the exec namespace
    # those become NameErrors at exec time. Mirrors the pattern used in
    # unsloth/models/_utils.py:patch_linear_scaling.
    from typing import Union, Optional, List, Any, Callable, Tuple, Dict, Iterator  # noqa: F401

    for (
        function_name,
        replacer,
    ) in (
        # ("_prepare_non_packed_dataloader", "def tokenize(element):",),
        (
            "_prepare_non_packed_dataloader",
            None,
        ),
        (
            "_prepare_dataset",
            None,
        ),
        # ("_prepare_packed_dataloader", "if dataset_text_field is not None",),
    ):
        if not hasattr(sft_trainer, function_name):
            continue

        function = getsource(eval(f"sft_trainer.{function_name}"))
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        check_text = (
            "\n"
            "if 'tokenizer'          not in locals(): tokenizer = processing_class\n"
            "if 'formatting_func'    not in locals(): raise RuntimeError('Unsloth: Please file a bug report - `formatting_func` does not exist!')\n"
            "if 'dataset_text_field' not in locals() and 'args' in locals(): dataset_text_field = args.dataset_text_field\n"
            "if 'dataset_text_field' not in locals(): dataset_text_field = None\n"
            "if formatting_func is None and dataset_text_field is None and 'prompt' in dataset[0] and 'completion' in dataset[0]:\n"
            "    test_text = (dataset[0]['prompt'] + dataset[0]['completion']) if (isinstance(dataset[0]['prompt'], str) and isinstance(dataset[0]['completion'], str)) else None\n"
            "elif formatting_func is None and dataset_text_field is not None:\n"
            "    test_text = dataset[0][dataset_text_field]\n"
            "elif formatting_func is not None:\n"
            "    test_text = formatting_func(dataset[0])[0]\n"
            "else:\n"
            "    test_text = None\n"
            "chat_template = getattr(tokenizer, 'chat_template', None)\n"
            "chat_template = '' if chat_template is None else chat_template\n"
            "has_bos_token_already = ((test_text is not None and test_text.startswith(tokenizer.bos_token)) or tokenizer.bos_token in chat_template) "
            "if getattr(tokenizer, 'bos_token', None) is not None else False\n"
            "if 'add_special_tokens' not in locals() and has_bos_token_already:\n"
            "    from functools import partial\n"
            "    tokenizer = partial(tokenizer, add_special_tokens = False)\n"
            "    processing_class = tokenizer\n"
            "else:\n"
            "    add_special_tokens = False if has_bos_token_already else add_special_tokens\n\n"
        )

        check_text = check_text.split("\n")
        check_text = "\n".join(" " * where + x for x in check_text)
        check_text = check_text.rstrip() + "\n"

        if replacer is None:
            # .*? matches first match. .+? matches final match.
            replacer = re.findall(
                f"def {function_name}" + r"\(.*?\).*?\:\n",
                function,
                flags = re.MULTILINE | re.DOTALL,
            )
            if len(replacer) == 0:
                continue
            replacer = replacer[0]
            function = function.replace(replacer, replacer + check_text)
        else:
            function = function.replace(replacer, check_text + replacer)

        x = [x for x in all_imports if x in function]
        try:
            exec(f"from trl.trainer.sft_trainer import ({','.join(x)})", locals())
        except ImportError:
            for _item in x:
                try:
                    exec(f"from trl.trainer.sft_trainer import {_item}", locals())
                except ImportError:
                    pass
        exec(function, locals(), globals())
        exec(
            f"trl.trainer.sft_trainer.SFTTrainer.{function_name} = {function_name}",
            globals(),
        )

    # Patch train with fix_untrained_tokens
    for path_to_trainer in (
        "sft_trainer.SFTTrainer",
        "dpo_trainer.DPOTrainer",
        "kto_trainer.KTOTrainer",
    ):
        function_name, replacer = "train", "if resume_from_checkpoint is False:"
        try:
            function = getsource(eval(f"trl.trainer.{path_to_trainer}.{function_name}"))
        except Exception:
            continue
        where = function.find("def")
        function = function.split("\n")
        function = "\n".join(x[where:] for x in function)

        check_text = (
            "\n"
            "import subprocess, re, gc, numpy as np\n"
            "a = np.array([0,])\n"
            "try:\n"
            "    a = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell = True)\n"
            "    a = re.findall(rb'([\\d]{1,})[\\s]{1,}M', a)\n"
            "    a = np.array([int(x.decode('utf-8'))/1024 for x in a])\n"
            "except:\n"
            "    if not torch.cuda.is_available():\n"
            "        raise RuntimeError('Unsloth: No GPU detected. AMD ROCm users: install ROCm-enabled PyTorch -- see https://docs.unsloth.ai/get-started/install-and-update/amd')\n"
            "    # nvidia-smi unavailable but torch.cuda IS available -- we are on\n"
            "    # a ROCm host (ROCm reuses the torch.cuda.* API surface, so\n"
            "    # device_count() is authoritative) or on a CUDA host without\n"
            "    # the CLI installed. Use the device count directly as a\n"
            "    # conservative multi-GPU signal: any configuration with more\n"
            "    # than one visible device is flagged as unsupported, matching\n"
            "    # the spirit of the per-device memory check used on CUDA.\n"
            "    if torch.cuda.device_count() > 1:\n"
            "        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')\n"
            "if ((a - PRE_CHECK) >= 1).sum() > 1:\n"
            "    raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')\n"
            "for _ in range(3):\n"
            "    gc.collect()\n"
            "    torch.cuda.empty_cache()\n"
            "pass\n"
            "\n"
            "tokenizer = self.processing_class if hasattr(self, 'processing_class') else self.tokenizer\n"
            "fix_untrained_tokens(self.model, tokenizer, self.train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)\n\n"
            "fix_zero_training_loss(self.model, tokenizer, self.train_dataset)\n\n"
        )

        # Warn on gradient accumulation steps if it's used
        check_text += (
            "\n"
            "try:\n"
            "    gradient_accumulation_steps = self.args.gradient_accumulation_steps\n"
            "    if type(gradient_accumulation_steps) is int and gradient_accumulation_steps > 1:\n"
            "        from transformers import __version__ as transformers_version\n"
            "        from packaging.version import Version\n"
            "        if Version(transformers_version) <= Version('4.45.2'):\n"
            "            print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\\n'\\\n"
            "                  '`pip install --upgrade --no-cache-dir --no-deps unsloth transformers git+https://github.com/huggingface/trl.git`')\n"
            "except:\n"
            "    pass\n"
            "\n\n"
        )

        # Add NEFTune since it doesn't seem to work?? We need to manually inject it
        check_text += (
            "\n"
            "if hasattr(self, 'neftune_hook_handle'):\n"
            "    self.neftune_hook_handle.remove()\n"
            "    if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle\n"
            "\n"
            "if getattr(self, 'neftune_noise_alpha', None) is not None:\n"
            "    self.model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha\n"
            "    self.neftune_hook_handle = self.model.get_input_embeddings().register_forward_hook(neftune_post_forward_hook)\n"
            "pass\n"
            "\n"
        )

        # Also DPO weirdly tokenizes non numeric columns? Delete them!
        check_text += (
            "\n"
            "if hasattr(self.train_dataset, 'column_names'):\n"
            "    column_names = set(self.train_dataset.column_names)\n"
            "    check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',\n"
            "        'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',\n"
            "        'prompt_input_ids', 'prompt_attention_mask']\n"
            "    if all(x in column_names for x in check):\n"
            "        self.train_dataset = self.train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])\n"
            "    del check, column_names\n"
            "\n"
        )

        check_text = check_text.split("\n")
        check_text = "\n".join(" " * where + x for x in check_text)

        function = function.replace(replacer, check_text + replacer)
        exec(function, globals())

        exec(
            f"trl.trainer.{path_to_trainer}.{function_name} = {function_name}",
            globals(),
        )


# Finally patch TRL tokenizer things -> moved to RL
# patch_sft_trainer_tokenizer()
