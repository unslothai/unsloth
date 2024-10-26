# Unsloth Zoo
# Copyright (C) 2024-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "train_on_responses_only",
]

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


def _find_common_token_ids(component, tokenizer):
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

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])
    
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
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    tokenizer = trainer.processing_class if hasattr(trainer, "processing_class") else trainer.tokenizer
    
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
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer)
    A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        all_labels = []

        for input_ids in input_ids_:
            n = len(input_ids)
            labels = [-100] * n
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
                            # Now copy input_ids to labels
                            labels[assistant_k : user_j] = input_ids[assistant_k : user_j]
                            # print(assistant_j, assistant_k, user_j, user_k)
                            break
                        pass
                        j += 1
                    pass
                pass
                j += 1
            pass
            all_labels.append(labels)
        pass
        return { "labels" : all_labels }
    pass

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True)
    pass
    
    if hasattr(trainer, "eval_dataset")  and trainer.eval_dataset  is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                trainer.eval_dataset[key] = value.map(_train_on_responses_only, batched = True)
        else:
            trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True)
        pass
    pass
    return trainer
pass
