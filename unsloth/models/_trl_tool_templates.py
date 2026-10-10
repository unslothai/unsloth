# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# TRL picks the tool-call parser (add_response_schema) and the multi-turn training template
# (get_training_chat_template) by exact string match on the chat template, so the fixed templates
# shipped in unsloth/* repos (Qwen3, Qwen3.5+, gpt-oss, GLM-4.5, Nemotron 3, Gemma 4) are rejected
# for GRPO tools= / environment_factory=. Only when TRL rejects a template, we look for the
# TRL-known template that renders the same text and reuse its parser / training template.
# The user's chat_template is never modified.

__all__ = [
    "patch_trl_tool_chat_templates",
]

import copy
import functools
import itertools
import logging
import sys

logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA_ERRORS = ("Unrecognized chat template",)
_TRAINING_TEMPLATE_ERRORS = ("patching is not supported",)
_PARSER_ATTRIBUTES = ("response_template", "response_schema")

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First factor."},
                    "b": {"type": "integer", "description": "Second factor."},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def _call(a, b):
    return {"type": "function", "function": {"name": "multiply", "arguments": {"a": a, "b": b}}}


_USER = {"role": "user", "content": "What is 3 times 4?"}
# Families differ in how they write non-integer arguments (JSON true / null vs Python True / None) in
# the history a training template re-renders.
_TYPED_CALL = {
    "type": "function",
    "function": {
        "name": "multiply",
        "arguments": {"a": 3, "b": 4.5, "exact": True, "note": None, "unit": "m"},
    },
}
_TOOL_TURN = {"role": "assistant", "content": "", "tool_calls": [_call(3, 4)]}
_TOOL_RESULT = {"role": "tool", "name": "multiply", "content": "12"}

# (history, assistant turn the model would generate): the parser only ever sees generated text.
_COMPLETION_PROBES = (
    ([_USER], _TOOL_TURN),
    ([_USER], {"role": "assistant", "content": "Let me compute.", "tool_calls": [_call(3, 4)]}),
    ([_USER], {"role": "assistant", "content": "", "tool_calls": [_call(3, 4), _call(5, 6)]}),
    ([_USER], {"role": "assistant", "content": "The answer is 12."}),
    ([_USER, _TOOL_TURN, _TOOL_RESULT], {"role": "assistant", "content": "The answer is 12."}),
)

# A training template replaces the user's template for every prompt, so it must render whole
# conversations identically, with and without tools.
_CONVERSATION_PROBES = (
    ([_USER], True, True),
    ([_USER, _TOOL_TURN], False, True),
    ([_USER, {"role": "assistant", "content": "", "tool_calls": [_TYPED_CALL]}], False, True),
    ([_USER, _TOOL_TURN, _TOOL_RESULT], True, True),
    (
        [_USER, _TOOL_TURN, _TOOL_RESULT, {"role": "assistant", "content": "The answer is 12."}],
        False,
        True,
    ),
    ([{"role": "system", "content": "Be brief."}, _USER], True, False),
    (
        [
            _USER,
            {"role": "assistant", "content": "12."},
            {"role": "user", "content": "And 5 times 6?"},
        ],
        True,
        False,
    ),
)

# Text and vision families of one model line render text alike but not images. Only compared when the
# user's template renders it: text-only templates may raise on list content.
_IMAGE_PROBE = (
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is shown?"}]}],
    True,
    False,
)


def _tokenizer_of(processing_class):
    from transformers import ProcessorMixin
    if isinstance(processing_class, ProcessorMixin):
        return processing_class.tokenizer
    return processing_class


def _template_of(processing_class):
    template = getattr(processing_class, "chat_template", None)
    if template is None and processing_class is not None:
        template = getattr(_tokenizer_of(processing_class), "chat_template", None)
    if isinstance(template, dict):
        # A named set (e.g. "tool_use") is picked per call by transformers; one family cannot stand in for it.
        template = next(iter(template.values())) if len(template) == 1 else None
    return template if isinstance(template, str) else None


# Users pass these through chat_template_kwargs; a match must hold for each, or a template that
# ignores enable_thinking=False would borrow one that writes an empty <think> into every prompt.
_TEMPLATE_KWARGS = ({}, {"enable_thinking": False})


def _render(tokenizer, template, messages, add_generation_prompt, tools, template_kwargs):
    return tokenizer.apply_chat_template(
        messages,
        tools = _TOOLS if tools else None,
        tokenize = False,
        add_generation_prompt = add_generation_prompt,
        chat_template = template,
        **template_kwargs,
    )


def _completion_signature(tokenizer, template):
    signature = []
    rendered = 0
    for (history, turn), kwargs in itertools.product(_COMPLETION_PROBES, _TEMPLATE_KWARGS):
        try:
            bare = _render(tokenizer, template, history, False, True, kwargs)
            prompt = _render(tokenizer, template, history, True, True, kwargs)
            full = _render(tokenizer, template, history + [turn], False, True, kwargs)
        except Exception as error:
            # Some templates refuse a probe (Llama 3.1 raises on parallel tool calls): match the refusal too.
            signature.append(("error", type(error).__name__))
            continue
        rendered += 1
        # Parsers also read what the generation prompt pre-writes (e.g. an opening <think>).
        signature.append(prompt[len(bare) :] if prompt.startswith(bare) else prompt)
        signature.append(full[len(prompt) :] if full.startswith(prompt) else full)
    return tuple(signature) if rendered else None


def _conversation_signature(tokenizer, template):
    try:
        return tuple(
            _render(tokenizer, template, messages, add_generation_prompt, tools, kwargs)
            for (messages, add_generation_prompt, tools), kwargs in itertools.product(
                _CONVERSATION_PROBES, _TEMPLATE_KWARGS
            )
        )
    except Exception:
        return None


def _image_signature(tokenizer, template):
    try:
        return tuple(
            _render(tokenizer, template, *_IMAGE_PROBE, kwargs) for kwargs in _TEMPLATE_KWARGS
        )
    except Exception:
        return None


def _known_templates(trl_module):
    return sorted(
        (name, value)
        for name, value in vars(trl_module).items()
        if name.endswith("_chat_template")
        and not name.endswith("_training_chat_template")
        and isinstance(value, str)
    )


def _template_copy(tokenizer, template):
    clone = copy.copy(tokenizer)
    clone.chat_template = template
    for attribute in _PARSER_ATTRIBUTES:
        if attribute in vars(clone):
            setattr(clone, attribute, None)
    return clone


_matches = {}


def _matching_families(trl_module, tokenizer, template, kind):
    # Templates read special tokens (bos_token, eos_token), so they are part of what renders.
    key = (
        id(trl_module),
        kind,
        template,
        repr(sorted(getattr(tokenizer, "special_tokens_map", {}).items())),
    )
    if key in _matches:
        return _matches[key]
    signature_of = _completion_signature if kind == "completion" else _conversation_signature
    target = signature_of(tokenizer, template)
    image_target = _image_signature(tokenizer, template) if kind == "conversation" else None
    found = []
    if target is not None:
        for name, known in _known_templates(trl_module):
            if known == template or signature_of(tokenizer, known) != target:
                continue
            if image_target is not None and _image_signature(tokenizer, known) != image_target:
                continue
            found.append((name, known))
    _matches[key] = found
    return found


def _parser_for(original, tokenizer, known):
    clone = _template_copy(tokenizer, known)
    try:
        original(clone)
    except Exception:
        return None
    parser = tuple((attribute, getattr(clone, attribute, None)) for attribute in _PARSER_ATTRIBUTES)
    return parser if any(value is not None for _, value in parser) else None


def _wrap_add_response_schema(original, trl_module):
    @functools.wraps(original)
    def add_response_schema(processing_class, *args, **kwargs):
        try:
            return original(processing_class, *args, **kwargs)
        except ValueError as error:
            if not any(text in str(error) for text in _RESPONSE_SCHEMA_ERRORS):
                raise
            tokenizer = _tokenizer_of(processing_class)
            template = _template_of(processing_class)
            if template is None:
                raise
            parsers = {}
            for name, known in _matching_families(trl_module, tokenizer, template, "completion"):
                parser = _parser_for(original, tokenizer, known)
                if parser is not None:
                    parsers.setdefault(repr(parser), (name, parser))
            if len(parsers) != 1:
                raise
            ((name, parser),) = parsers.values()
            for attribute, value in parser:
                if value is not None:
                    setattr(tokenizer, attribute, value)
            logger.info(
                f"Unsloth: TRL does not know this chat template; using the tool-call parser of "
                f"`{name}`, which renders tool calls identically."
            )
            return processing_class

    add_response_schema._unsloth_tool_template_patched = True
    return add_response_schema


def _wrap_get_training_chat_template(original, trl_module):
    @functools.wraps(original)
    def get_training_chat_template(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError as error:
            if not any(text in str(error) for text in _TRAINING_TEMPLATE_ERRORS):
                raise
            processing_class = (
                args[0] if args else (kwargs.get("processing_class") or kwargs.get("tokenizer"))
            )
            template = _template_of(processing_class)
            if template is None:
                raise
            tokenizer = _tokenizer_of(processing_class)
            results = {}
            for name, known in _matching_families(trl_module, tokenizer, template, "conversation"):
                clone = copy.copy(processing_class)
                clone.chat_template = known
                try:
                    training = original(clone)
                except Exception:
                    continue
                # None means the known template already qualifies; the user's copy may still lack
                # its {% generation %} markers, so hand back the known template itself.
                training = known if training is None else training
                rendered = _conversation_signature(tokenizer, training)
                if rendered is not None:
                    results.setdefault(rendered, (name, training))
            if len(results) != 1:
                raise
            ((name, training),) = results.values()
            logger.info(
                f"Unsloth: TRL does not know this chat template; using the training template of "
                f"`{name}`, which renders conversations identically."
            )
            return training

    get_training_chat_template._unsloth_tool_template_patched = True
    return get_training_chat_template


def _rebind(name, original, replacement):
    # TRL trainers import these by name, so rebind every module still holding the original.
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if namespace is not None and namespace.get(name) is original:
            try:
                setattr(module, name, replacement)
            except (AttributeError, TypeError):
                pass


def patch_trl_tool_chat_templates():
    try:
        import trl.chat_template_utils as trl_module
    except Exception:
        return False
    patched = False
    for name, wrap in (
        ("add_response_schema", _wrap_add_response_schema),
        ("get_training_chat_template", _wrap_get_training_chat_template),
    ):
        original = getattr(trl_module, name, None)
        if original is None:
            continue
        if getattr(original, "_unsloth_tool_template_patched", False):
            patched = True
            continue
        _rebind(name, original, wrap(original, trl_module))
        patched = True
    return patched
