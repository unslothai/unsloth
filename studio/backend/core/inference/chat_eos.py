# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve a chat model's assistant-turn-end stop tokens.

Some checkpoints set eos_token_id to a bare document terminator (Qwen3.5 ships
config eos ``<|endoftext|>`` though chat turns end with ``<|im_end|>``, and its
small chat variants ship no generation_config), so generation runs past the turn
and loops -- re-emitting tool calls or hallucinating ``<|im_start|>`` turns.

Turn-end markers are derived from the tokenizer's ``chat_template`` (the tokens it
actually uses to end a turn), not raw vocab membership: a base/coder model can
carry ChatML control tokens in a shared vocab without using them, and a loader
may have synced ``eos_token`` to the document terminator. Dependency-light (no
torch / unsloth) so it is unit-testable without the full inference stack.
"""

from typing import Optional

# Canonical assistant-turn-end markers per chat family.
_CHAT_TURN_END_TOKENS = (
    "<|im_end|>",  # ChatML: Qwen, Yi
    "<|eot_id|>",  # Llama 3.x
    "<|eom_id|>",  # Llama 3.x tool turns
    "<end_of_turn>",  # Gemma
    "<turn|>",  # Gemma-4
    "<|end|>",  # Phi
)
# gpt-oss/harmony uses <|end|> as an intra-message channel delimiter (not the turn
# end) and has a dedicated streamer, so its eos is left untouched.
_HARMONY_MARKERS = ("<|channel|>", "<|constrain|>")


def _eos_id_set(eos_token_id) -> set:
    if isinstance(eos_token_id, (list, tuple)):
        return {int(t) for t in eos_token_id if t is not None}
    if eos_token_id is not None:
        return {int(eos_token_id)}
    return set()


def resolve_chat_turn_end_eos_ids_using(template_tokenizer, id_tokenizer) -> list:
    """eos of ``id_tokenizer`` plus any canonical turn-end marker the
    ``template_tokenizer``'s chat_template uses, resolved to ids on ``id_tokenizer`` --
    the tokenizer generation actually uses.

    Pass the same tokenizer for both at load time. After a mapped ``get_chat_template``
    pass the MAPPED tokenizer as ``template_tokenizer`` (it carries the effective
    template) and the ORIGINAL generation tokenizer as ``id_tokenizer``: a mapped
    template registered ``map_eos_token=True`` can hand back a tokenizer whose vocab
    folds the turn-end token onto the doc-eos id, and generate_stream re-reads the
    original tokenizer, so resolving ids on the mapped tokenizer would store the wrong
    (doc-eos) id and let generation run past the real turn marker."""
    ids = _eos_id_set(getattr(id_tokenizer, "eos_token_id", None))
    template = getattr(template_tokenizer, "chat_template", None) or ""
    if not isinstance(template, str) or any(h in template for h in _HARMONY_MARKERS):
        return sorted(ids)
    unk = getattr(id_tokenizer, "unk_token_id", None)
    for marker in _CHAT_TURN_END_TOKENS:
        if marker in template:
            try:
                tid = id_tokenizer.convert_tokens_to_ids(marker)
            except Exception:
                tid = None
            if tid is not None and tid != unk and int(tid) >= 0:
                ids.add(int(tid))
    return sorted(ids)


def resolve_chat_turn_end_eos_ids(tokenizer) -> list:
    """tokenizer.eos plus any canonical turn-end marker the model's chat_template
    actually uses. Cheap (convert_tokens_to_ids per marker, no get_vocab); intended
    to be resolved once at load. Returns eos unchanged for harmony templates."""
    return resolve_chat_turn_end_eos_ids_using(tokenizer, tokenizer)


def chat_eos_repair(current_eos, turn_end_ids) -> Optional[list]:
    """Merged eos_token_id list, or None if ``current_eos`` already covers every
    resolved turn-end id. Used to repair a model's generation_config at load so
    every ``.generate()`` path (vision, tool loops) stops at the turn boundary."""
    if not turn_end_ids:
        return None
    current_set = _eos_id_set(current_eos)
    if set(turn_end_ids) <= current_set:
        return None
    return sorted(current_set | set(turn_end_ids))
