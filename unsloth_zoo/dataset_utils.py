# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "train_on_responses_only",
    "sft_prepare_dataset",
    "standardize_data_formats",
]

from typing import Union, Callable, Optional, List, Dict
import torch

# From https://www.geeksforgeeks.org/longest-common-substring-array-strings/
# Longest Common Substring in an Array of Strings
def _old_longest_common_substring(arr):
    n = len(arr)
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 1, l + 1):
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                if stem not in arr[k]:
                    break
            if (k + 1 == n and len(res) < len(stem)):
                res = stem
    return res
pass


def _longest_common_sublist(lists):
    """
    Finds the longest common sublist among multiple lists.

    Parameters:
    lists (List[List[int]]): A list of lists.

    Returns:
    List[int]: The longest common sublist. If multiple sublists have the same maximum length,
               one of them is returned. If there's no common sublist, an empty list is returned.
    """
    if not lists: return []

    # Find the minimum length among all lists
    min_len = min(len(lst) for lst in lists)
    if min_len == 0: return []

    def has_common_sublist(length):
        """
        Checks if there's a common sublist of the given length across all lists.

        Returns:
        (bool, List): Tuple of whether such a sublist exists and the sublist itself.
        """
        common = set()
        first = lists[0]
        # Generate all possible sublists of the given length from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i:i + length])
            common.add(sub)
        pass

        # Iterate over the remaining lists and retain only the common sublists
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i:i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass
        
        # If common is not empty, return one of the common sublists
        return True, list(common.pop())
    pass

    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist  # Update result with the latest found sublist
            left = mid + 1    # Try to find a longer sublist
        else:
            right = mid - 1   # Try with a shorter length
    pass

    return result
pass


def _find_common_token_ids(component, tokenizer, force_match = False):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if   component.endswith (" "): right_text = " "
    elif component.endswith("\n"): right_text = "\n"
    left_text = ""
    if   component.startswith (" "): left_text = " "
    elif component.startswith("\n"): left_text = "\n"
    stripped = component.strip()
    
    # Add current pieces and also newlines
    all_input_ids = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left*left_text + stripped + right*right_text
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)

                x = left*"\n" + stripped + right*"\n"
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)
            pass
        pass
    else:
        x = tokenizer(component, add_special_tokens = False).input_ids
        all_input_ids.append(x)
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # If substring is simply [0], this might be just the original single token
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens breaks since we output [0] but we need [3] [4]
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        # Confirm single token in every single possible match
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # Also if substring is original input_ids + [0], then leave it as the original one
    # This happens when no newlines / spaces are used in chat template
    # Eg Phi-4 does not use newlines or spaces
    if (len(set(str(x) for x in all_input_ids)) == 1) and \
        (len(all_input_ids[0]) + 1 == len(substring)) and \
        (all_input_ids[0] == substring[:-1]):

        # Use original un-changed substring
        substring = all_input_ids[0]
    pass
    
    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens = False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]
    return substring, optional_left, optional_right
pass


def train_on_responses_only(
    trainer,
    instruction_part = None,
    response_part    = None,
    force_match      = True,  # Match newlines as well!
    tokenizer        = None,  # Optional
    return_function  = False, # Useful for iterating over lists
    num_proc         = None,
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if tokenizer is None and trainer is not None:
        tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
    # Get non vision tokenizer
    if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if  not hasattr(tokenizer, "_unsloth_input_part") or \
        not hasattr(tokenizer, "_unsloth_output_part"):
        
        if instruction_part is None or response_part is None:
            raise ValueError("Unsloth: instruction_part and response_part must be given!")
        pass
    elif (instruction_part is not None or response_part is not None) and \
        (hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

        raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part    = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer, force_match)
    A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer, force_match)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right
    torch_Tensor = torch.Tensor
    torch_int64  = torch.int64

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        use_tensors = False
        if type(input_ids_) is torch_Tensor:
            use_tensors = True
            input_ids_ = input_ids_.tolist()
        if "labels" in examples:
            labels_ = examples["labels"].tolist()
            assert(len(labels_) == len(input_ids_))
        else:
            labels_ = [None]*len(input_ids_)

        all_labels = []
        for input_ids, old_labels in zip(input_ids_, labels_):
            n = len(input_ids)
            labels = [-100] * n
            
            use_old_labels = False
            if old_labels is not None:
                use_old_labels = True
                assert(n == len(old_labels))
            n_minus_1 = n - 1
            j = 0
            while j < n:
                # Find <assistant>
                if (input_ids[j] == A_first) and \
                    (input_ids[j : (k := j + len_A_must)] == A_must):

                    # Now backtrack to get previous optional tokens
                    for optional_left in A_left_reversed:
                        if j < 1: break
                        if optional_left == input_ids[j-1]: j -= 1
                        else: break
                    pass
                    # And forwards look as well
                    for optional_right in A_right_forward:
                        if k >= n_minus_1: break
                        if optional_right == input_ids[k+1]: k += 1
                        else: break
                    pass
                    # assistant_j = j
                    assistant_k = k

                    j = assistant_k
                    # Given <assistant>, now find next user
                    while j < n:
                        # Find <user>
                        # Also accept last final item if assistant is the last turn
                        if (j == n_minus_1) or \
                            ((input_ids[j] == Q_first) and \
                             (input_ids[j : (k := j + len_Q_must)] == Q_must)):

                            # Now backtrack to get previous optional tokens
                            for optional_left in Q_left_reversed:
                                if j < 1: break
                                if optional_left == input_ids[j-1]: j -= 1
                                else: break
                            pass
                            # And forwards look as well
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1: break
                                if optional_right == input_ids[k+1]: k += 1
                                else: break
                            pass
                            user_j = j
                            # Account for last item
                            if user_j != n_minus_1:
                                # user_k = k
                                # j = user_k
                                j = k
                            else:
                                user_j = n
                                k = n
                            pass

                            if not use_old_labels:
                                # Now copy input_ids to labels
                                labels[assistant_k : user_j] = input_ids [assistant_k : user_j]
                                # print(assistant_j, assistant_k, user_j, user_k)
                            else:
                                # Copy over from old labels!
                                labels[assistant_k : user_j] = old_labels[assistant_k : user_j]
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass
            all_labels.append(labels)
        pass
        return { "labels" : torch.tensor(all_labels, dtype = torch.int64) if use_tensors else all_labels }
    pass
    if return_function:
        return _train_on_responses_only

    from multiprocessing import cpu_count
    if num_proc is None or type(num_proc) is not int: num_proc = cpu_count()

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        if not hasattr(trainer.train_dataset, "map"):
            raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
        if isinstance(trainer.train_dataset, IterableDataset):
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batch_size = trainer.train_dataset._ex_iterable.batch_size, batched = True)
        else:
            trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True, num_proc = num_proc)
    pass
    
    if hasattr(trainer, "eval_dataset")  and trainer.eval_dataset  is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                if not hasattr(value, "map"):
                    raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
                if isinstance(trainer.eval_dataset, IterableDataset):
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batch_size = trainer.eval_dataset._ex_iterable.batch_size, batched = True)
                else:
                    trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched = True, num_proc = num_proc)
        else:
            if not hasattr(trainer.eval_dataset, "map"):
                raise TypeError("Unsloth: train_on_responses_only does not work on lists!")
            if isinstance(trainer.eval_dataset, IterableDataset):
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batch_size = trainer.eval_dataset._ex_iterable.batch_size, batched = True)
            else:
                trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True, num_proc = num_proc)
        pass
    pass

    # Edit data collator as well if not DataCollatorForSeq2Seq
    from transformers import DataCollatorForSeq2Seq
    if hasattr(trainer, "data_collator") and \
        not isinstance(trainer.data_collator, DataCollatorForSeq2Seq):
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    # Check if all labels randomnly got masked to nothing - maybe wrong chat template?
    from .training_utils import fix_zero_training_loss
    fix_zero_training_loss(None, tokenizer, trainer.train_dataset)
    return trainer
pass


def standardize_data_formats(
    dataset,
    tokenizer             = None,
    aliases_for_system    = ["system",],
    aliases_for_user      = ["user", "human", "input",],
    aliases_for_assistant = ["gpt", "assistant", "output",],
):
    """
    Standardizes ShareGPT and other formats to user/assistant Hugging Face format.
    
    Get aliases for the system, user and assistant roles.
    These shall map to "system", "user" and "assistant" respectively.
    
    aliases_for_system    = ["system",],
    aliases_for_user      = ["user", "human", "input",],
    aliases_for_assistant = ["gpt", "assistant", "output",],
    """
    import collections
    import itertools

    # Check if vision tokenizer is used - if yes, we must use the format:
    # Text : {"role" : role, "content" : "Happy"}
    # VLMs : {"role" : role, "content" : [{"type" : "text", "text" : "Happy"}]}
    is_vlm = False
    if tokenizer is not None:
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
            is_vlm = True

    column_names = set(next(iter(dataset)).keys())
    if "conversations" not in column_names:
        return dataset

    convos = dataset[:10]["conversations"]
    uniques = collections.defaultdict(list)
    for convo in convos:
        for message in convo:
            for key, value in message.items():
                if type(value) is not str:
                    raise RuntimeError("Unsloth: Cannot standardize non text datasets!")
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
        convos = examples["conversations"]
        all_convos = []
        for convo in convos:
            new_convo = []
            for message in convo:
                role = aliases_mapping[message[role_key]]
                text = message[content_key]
                if is_vlm: text = [ {"type" : "text", "text" : text} ]
                x = {"role" : role, "content" : text}
                new_convo.append(x)
            pass
            all_convos.append(new_convo)
        pass
        return { "conversations" : all_convos, }
    pass

    from multiprocessing import cpu_count
    num_proc = cpu_count()

    return dataset.map(
        _standardize_dataset,
        batched = True,
        desc = "Unsloth: Standardizing formats",
        num_proc = num_proc,
    )
pass


from datasets import (Dataset, IterableDataset,)
from trl.trainer.utils import ConstantLengthDataset
# Faster SFTTrainer prepare_dataset
def sft_prepare_dataset(
    self,
    dataset: Union[Dataset, IterableDataset],
    processing_class,
    args,
    packing: bool,
    formatting_func: Optional[Callable[[dict], str]],
    dataset_name: str,
) -> Union[Dataset, IterableDataset]:
    # All Unsloth Zoo code licensed under LGPLv3
    if isinstance(dataset, ConstantLengthDataset): return dataset

    map_kwargs = {}
    use_desc = isinstance(dataset, Dataset)
    is_vlm = hasattr(processing_class, "tokenizer")
    tokenizer = processing_class
    if is_vlm: tokenizer = processing_class.tokenizer

    # Get max length
    max_seq_length = getattr(args, "max_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(args, "max_seq_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(self, "max_seq_length", 0)
    if max_seq_length == 0: max_seq_length = getattr(self, "max_seq", 0)
    if max_seq_length == 0: raise RuntimeError("Unsloth: max_seq_length is 0! Please specify one!")
    dataset_text_field = getattr(args, "dataset_text_field", "text")
    do_truncation = max_seq_length != 0
    do_formatting_func = False
    do_tokenize = True

    # Get correct column names
    column_names = set(next(iter(dataset)).keys())
    used_column_names = ["input_ids"]
    if "attention_mask" in column_names:
        used_column_names.append("attention_mask")

    # Check if already tokenized so skip
    from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
    if "labels" in column_names:
        # Most likely forgot data collator!
        if is_vlm and not hasattr(tokenizer, "pad"):
            # Check if processing_class has a .pad, if not, use tokenizer.tokenizer
            raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer)
        used_column_names.append("labels")
        do_tokenize = False
    elif "input_ids" in column_names:
        # Skip dataset prep, and set data collator
        if is_vlm and not hasattr(tokenizer, "pad"):
            # Check if processing_class has a .pad, if not, use tokenizer.tokenizer
            raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        do_tokenize = False
    elif dataset_text_field not in column_names:
        do_formatting_func = True
        if formatting_func is None:
            raise RuntimeError("Unsloth: You must specify a `formatting_func`")
    pass

    if do_tokenize:
        # Check double BOS tokens
        if do_formatting_func:
            test_text = formatting_func(next(iter(dataset)))
            if not isinstance(test_text, list):
                raise ValueError(
                    "Unsloth: The `formatting_func` should return a list of processed strings."
                )
            test_text = test_text[0]
        else:
            test_text = next(iter(dataset))[dataset_text_field][0]

        # Get chat template
        chat_template = getattr(processing_class, 'chat_template', '')
        if chat_template == '' and is_vlm:
            chat_template = getattr(tokenizer, 'chat_template', '')
        if chat_template is None:
            chat_template = ''

        # Get bos_token
        add_special_tokens = True
        bos_token_1 = getattr(processing_class, 'bos_token', None)
        bos_token_2 = getattr(tokenizer, 'bos_token', None)
        bos_token = bos_token_1 or bos_token_2

        if bos_token is not None:
            if test_text.startswith(bos_token) or bos_token in chat_template:
                add_special_tokens = False
                print("Unsloth: We found double BOS tokens - we shall remove one automatically.")
        pass

        # Create tokenize function
        def _tokenize(example):
            return tokenizer(
                example[dataset_text_field] if not do_formatting_func else formatting_func(example),
                truncation = do_truncation,
                max_length = max_seq_length,
                return_token_type_ids = False,
                add_special_tokens = add_special_tokens,
            )
        pass

        if not isinstance(dataset, IterableDataset):
            map_kwargs["num_proc"] = getattr(args, "dataset_num_proc", 2)
        else:
            map_kwargs["batch_size"] = dataset._ex_iterable.batch_size
            
        if use_desc: map_kwargs["desc"] = f'Unsloth: Tokenizing ["{dataset_text_field}"]'
        dataset = dataset.map(_tokenize, batched = True, **map_kwargs)

        # If VLM, switch data collator since .pad is needed!
        if is_vlm and not hasattr(processing_class, "pad"):
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
            self.data_collator = data_collator
        pass
    pass
    if packing:
        print("Unsloth: Hugging Face's packing is currently buggy - we're disabling it for now!")
        return dataset

        if max_seq_length == 0:
            raise ValueError("When packing is enabled, `max_seq_length` can't be `None`.")

        if use_desc: map_kwargs["desc"] = f"Unsloth: Packing {dataset_name} dataset"
        dataset = dataset.select_columns(used_column_names).map(
            pack_examples,
            batched = True,
            fn_kwargs = {"seq_length": max_seq_length,},
            **map_kwargs,
        )
    pass
    return dataset
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
