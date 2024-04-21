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

__all__ = [
    "load_correct_tokenizer",
    "fix_sentencepiece_tokenizer",
    "check_tokenizer",
    "fix_untrained_tokens",
    "add_new_tokens",
]


IGNORED_TOKENIZER_CHECKING = frozenset((
    "CodeLlamaTokenizerFast",
    "CodeLlamaTokenizer",
))

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


def assert_same_tokenization(slow_tokenizer, fast_tokenizer):
    # Get eos_token, bos_token etc
    dir_names = dir(slow_tokenizer)
    special_tokens = list(filter(None, (
        getattr(slow_tokenizer, x) for x in dir_names
        if x.endswith("_token") and x.count("_") == 1
    )))
    all_special_tokens = list(set(special_tokens + slow_tokenizer.all_special_tokens))
    try:
        string = "\n".join(all_special_tokens) + \
            "A quick brown fox jumps over the lazy dog!!\n\nHi</s>\n\n" + \
            "".join(all_special_tokens)
        return slow_tokenizer(string).input_ids == fast_tokenizer(string).input_ids
    except:
        # For eg see https://github.com/unslothai/unsloth/issues/292
        # Sometimes tokenizer has weird tokens, causing a combined tokenization to fail.
        # [TODO] We temporarily disable this for CodeLlama tokenizers
        if slow_tokenizer.__repr__().split("(", 1)[0] in IGNORED_TOKENIZER_CHECKING:
            return True
        else:
            return False
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


def load_correct_tokenizer(
    tokenizer_name,
    model_max_length = None,
    padding_side = "right",
    token = None,
    trust_remote_code = False,
    cache_dir = "huggingface_tokenizers_cache",
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
            use_fast          = False,
            cache_dir         = cache_dir,
        )
    except:
        print(
            f"Unsloth: {tokenizer_name} has no tokenizer.model file.\n"\
            "Just informing you about this - this is not a critical error."
        )
    pass

    fast_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        model_max_length  = model_max_length,
        padding_side      = padding_side,
        token             = token,
        trust_remote_code = trust_remote_code,
        cache_dir         = cache_dir,
    )

    if slow_tokenizer is not None:
        if hasattr(fast_tokenizer, "add_bos_token") and hasattr(slow_tokenizer, "add_bos_token"):
            fast_tokenizer.add_bos_token = slow_tokenizer.add_bos_token
        if hasattr(fast_tokenizer, "add_eos_token") and hasattr(slow_tokenizer, "add_eos_token"):
            fast_tokenizer.add_eos_token = slow_tokenizer.add_eos_token
        
        # Confirm if slow and fast are equivalent!
        if assert_same_tokenization(slow_tokenizer, fast_tokenizer):
            return fast_tokenizer
        else:
            return convert_to_fast_tokenizer(slow_tokenizer)
        pass
    else:
        return fast_tokenizer
    pass
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
                    use_fast = False,
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
def fix_untrained_tokens(model, eps = 1e-16):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    embedding_matrix = model.get_input_embeddings ().weight.data
    lm_head_matrix   = model.get_output_embeddings().weight.data

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis = 1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained
    if n_untrained != 0:
        print(
            f"Unsloth: Not an error, but your model has {n_untrained} untrained tokens.\n"\
            "We shall set them to the mean of the other trained tokens."
        )
    pass

    # First set untrained to all 0s - sometimes it's not! 1e-23 for bfloat16
    embedding_matrix[where_untrained] = 0
    lm_head_matrix  [where_untrained] = 0

    # Find sum
    sum_embedding  = torch.sum(embedding_matrix, dtype = torch.float32, axis = 0)
    sum_lm_head    = torch.sum(lm_head_matrix,   dtype = torch.float32, axis = 0)

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = (sum_embedding / n_trained).to(embedding_matrix.dtype)
    mean_lm_head   = (sum_lm_head   / n_trained).to(lm_head_matrix  .dtype)

    # Set them to the mean
    embedding_matrix[where_untrained] = mean_embedding
    lm_head_matrix  [where_untrained] = mean_lm_head

    return mean_embedding, mean_lm_head
pass


@torch.inference_mode
def add_new_tokens(
    model,
    tokenizer,
    new_tokens = [],
    method = "mean",
    interpolation = 0.05,
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
    mean_embedding, mean_lm_head = fix_untrained_tokens(model)
    mean_embedding = mean_embedding.to(torch.float32)
    mean_lm_head   = mean_lm_head  .to(torch.float32)

    # Add tokens!
    old_length = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # If we use interpolation, we interpolate between the mean embeddings and
    # the Word2Vec sum of the other vectors
    embedding_matrix = model.get_input_embeddings ().weight.data
    lm_head_matrix   = model.get_output_embeddings().weight.data

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
