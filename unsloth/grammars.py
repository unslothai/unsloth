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

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "generate_with_grammar",
]

JSON_ARR_GBNF = r'''
root ::= arr
value ::= object | array | string | number | ("true" | "false" | "null") ws
arr ::=
  "[
" ws (
            value
            (",
" ws value)*
        )? "]"
object ::=
  "{" ws (
            string ":" ws value
            ("," ws string ":" ws value)*
        )? "}" ws
array ::=
  "[" ws (
            value
            ("," ws value)*
        )? "]" ws
string ::=
  """ (
    [^"\x7F\x00-\x1F] |
    "" (["\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* """ ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ 	
] ws)?
'''


def generate_with_grammar(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    grammar_str: Optional[str] = None,
    start_rule: str = "root",
    max_new_tokens: int = 256,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    repetition_penalty: float = 1.1,
    num_return_sequences: int = 1,
    **kwargs,
):
    """
    Generate text with grammar constraints using transformers-cfg.
    Automatically handles model-specific parameter compatibility.
    """
    try:
        from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
        from transformers_cfg.generation.logits_process import (
            GrammarConstrainedLogitsProcessor,
        )
    except ImportError:
        raise ImportError(
            "Unsloth: Please install transformers-cfg to use grammar-constrained generation: "
            "`pip install transformers-cfg`"
        )

    if grammar_str is None:
        grammar_str = JSON_ARR_GBNF

    grammar = IncrementalGrammarConstraint(
        grammar_str, start_rule_name = start_rule, tokenizer = tokenizer
    )
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": num_return_sequences,
        "logits_processor": [grammar_processor],
        **kwargs,
    }

    if do_sample:
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
    else:
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        generation_kwargs["top_k"] = None

    try:
        return model.generate(**generation_kwargs)
    except ValueError as e:
        if "model_kwargs" in str(e):
            for bad_kwarg in ["sliding_window", "num_logits_to_keep"]:
                generation_kwargs.pop(bad_kwarg, None)
            return model.generate(**generation_kwargs)
        raise
