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
from typing import Optional

__all__ = [
    "generate_with_grammar",
]

JSON_ARR_GBNF = r"""
root ::= arr
value ::= object | array | string | number | ("true" | "false" | "null") ws
arr ::=
  "[" ws (
            value
            ("," ws value)*
        )? "]" ws
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
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n\r] ws)?
"""


def _build_tokenizer_resources(tokenizer):
    """
    Build ByteTrie and TokenizerMiddleMapping for the given tokenizer,
    bypassing transformers-cfg's isinstance checks that fail on
    transformers 5.x (where all tokenizers are TokenizersBackend).
    """
    from transformers_cfg.tokenization.byte_trie import ByteTrie
    from transformers_cfg.tokenization.utils import get_tokenizer_charset
    from transformers_cfg.tokenization.middle.TokenizerMiddleMapping import (
        LLAMA1TokenizerMiddleMapping,
        TokenizerMiddleMapping,
    )

    charset = get_tokenizer_charset(tokenizer)

    if "\u2581" in charset:
        # SentencePiece-style (LLaMA-2, Mistral v0.1, Gemma)
        homomorphism = LLAMA1TokenizerMiddleMapping(tokenizer)
    elif len(charset) >= 256 and len(charset) < 256 + 30:
        # GPT2-style byte-proxy (Llama-3, GPT-2, Qwen2)
        # Build mapping manually to avoid ByteProxyMapping loading a slow tokenizer
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        byte_to_unicode = bytes_to_unicode()  # {byte_int: unicode_char}
        unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}

        class _GPT2ManualMapping(TokenizerMiddleMapping):
            def __init__(self, tok):
                super().__init__(tok)
                self._unicode_to_byte = unicode_to_byte

            def map(self, token_id: int, verbose = False) -> bytes:
                token_id = int(token_id)
                token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                return bytes(self._unicode_to_byte.get(c, ord(c)) for c in token_str)

        pass

        homomorphism = _GPT2ManualMapping(tokenizer)
    else:
        # Fallback: try transformers-cfg's auto_infer
        homomorphism = TokenizerMiddleMapping.auto_infer(tokenizer)

    # Build the ByteTrie from the vocabulary
    trie = ByteTrie()
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        try:
            byte_repr = homomorphism.map(token_id)
            if byte_repr:
                trie.insert(byte_repr, token_id)
        except Exception:
            continue

    return trie, homomorphism




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
    repetition_penalty: float = 1.0,
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

    trie, homomorphism = _build_tokenizer_resources(tokenizer)
    grammar = IncrementalGrammarConstraint(
        grammar_str,
        start_rule_name = start_rule,
        tokenizer = tokenizer,
        trie = trie,
        homomorphism = homomorphism,
    )
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    extra_processors = kwargs.pop("logits_processor", None)
    if extra_processors is not None:
        if not isinstance(extra_processors, list):
            extra_processors = list(extra_processors)
    else:
        extra_processors = []

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": num_return_sequences,
        "logits_processor": [grammar_processor] + extra_processors,
        **kwargs,
    }

    if do_sample:
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

    try:
        return model.generate(**generation_kwargs)
    except ValueError as e:
        if "model_kwargs" in str(e):
            for bad_kwarg in ["sliding_window", "num_logits_to_keep"]:
                generation_kwargs.pop(bad_kwarg, None)
            return model.generate(**generation_kwargs)
        raise
