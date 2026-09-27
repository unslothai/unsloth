# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Grammar-constrained decoding for the MLX paths, through llguidance token bitmasks."""

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

try:
    import llguidance as _llg
    import llguidance.hf as _llg_hf
    import llguidance.mlx as _llg_mlx

    LLGUIDANCE_AVAILABLE = True
    LLGUIDANCE_VERSION = str(_llg.get_version())
except Exception:
    _llg = None
    _llg_hf = None
    _llg_mlx = None
    LLGUIDANCE_AVAILABLE = False
    LLGUIDANCE_VERSION = None

SUPPORTED_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")

MISSING_ENGINE_MESSAGE = (
    "response_format needs the llguidance grammar engine, which did not load. Install or "
    "repair it with `pip install --upgrade llguidance`, or load a GGUF model to use the "
    "llama.cpp grammar engine instead."
)

# json_object promises an object, not any JSON value; spelled canonically to share a grammar.
_JSON_OBJECT_SCHEMA = '{"type":"object"}'

_PRELUDE_MAX_CHARS = 4000

_THINK_MARKERS = ("<think>", "</think>")


class ResponseFormatError(ValueError):
    """Invalid or unhonorable ``response_format``; the message is client-safe."""

    public = True
    openai_param = "response_format"


class GrammarDesyncError(RuntimeError):
    """A token the mask never offered was committed; no change to the request fixes it."""


@dataclass(frozen = True)
class ConstraintSpec:
    """A validated document grammar for one request, not yet bound to a tokenizer."""

    grammar: str
    schema_json: Optional[str] = None
    reasoning_close: Optional[str] = None
    reasoning_close_marker: Optional[str] = None

    def build(
        self,
        tokenizer,
        *,
        in_reasoning: bool = False,
    ) -> "GrammarConstraint":
        if in_reasoning:
            if self.reasoning_close is None:
                raise ResponseFormatError(
                    "response_format cannot be honored on this model: the prompt starts "
                    "inside an open reasoning block, and its closing marker "
                    f"{self.reasoning_close_marker!r} cannot both be written by a grammar "
                    "and survive into the reply, so nothing can close the block before the "
                    "document. Send the request with thinking disabled, or load a GGUF "
                    "build of this model."
                )
            grammar = _cached_grammar(self.schema_json, prelude_close = self.reasoning_close)
        else:
            grammar = self.grammar
        return GrammarConstraint(grammar, tokenizer, allows_reasoning = in_reasoning)


def build_constraint(
    response_format,
    tokenizer,
    prompt,
    *,
    reasoning_markers = None,
    tools = None,
    reasoning_is_extracted: bool = False,
    reply_keeps_special_tokens: bool = False,
) -> Optional["GrammarConstraint"]:
    """The constraint one rendered prompt decodes under, or None for no contract."""
    markers = _reasoning_markers(tokenizer, reasoning_markers, tools)
    spec = constraint_spec_from_response_format(
        response_format,
        tokenizer,
        reasoning_markers = markers,
        reply_keeps_special_tokens = reply_keeps_special_tokens,
    )
    if spec is None:
        return None
    from core.inference.chat_template_helpers import prompt_opens_reasoning_channel

    return spec.build(
        tokenizer,
        in_reasoning = (reasoning_is_extracted and prompt_opens_reasoning_channel(prompt, markers)),
    )


def constraint_spec_from_response_format(
    response_format,
    tokenizer = None,
    *,
    reasoning_markers = None,
    reply_keeps_special_tokens: bool = False,
) -> Optional[ConstraintSpec]:
    """Validate a request's ``response_format`` into a :class:`ConstraintSpec`, or None."""
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ResponseFormatError("response_format must be an object with a 'type' field")
    format_type = response_format.get("type")
    if format_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
        raise ResponseFormatError(
            f"unsupported response_format type {format_type!r}; supported: "
            + ", ".join(SUPPORTED_RESPONSE_FORMAT_TYPES)
        )
    if format_type == "text":
        if response_format != {"type": "text"}:
            raise ResponseFormatError(
                "unsupported members on a text response_format: "
                + ", ".join(sorted(k for k in response_format if k != "type"))
            )
        return None
    if not LLGUIDANCE_AVAILABLE:
        raise ResponseFormatError(MISSING_ENGINE_MESSAGE)

    if format_type == "json_object":
        supplied = response_format.get("schema")
        if supplied is not None:
            supplied = _require_schema_object(
                supplied,
                "response_format type 'json_object' requires a 'schema' member to be a "
                "JSON Schema object",
            )
        # Absent, null and empty mean this format's own object promise, not llguidance's any-JSON `{}`.
        schema_json = _JSON_OBJECT_SCHEMA if not supplied else _canonical_schema_json(supplied)
    else:
        schema_json = _canonical_schema_json(_json_schema_from_response_format(response_format))

    markers = tuple(reasoning_markers) if reasoning_markers else None
    return ConstraintSpec(
        grammar = _cached_grammar(schema_json, prelude_close = None),
        schema_json = schema_json,
        reasoning_close = _marker_reference(
            tokenizer, markers[1], reply_keeps_special_tokens = reply_keeps_special_tokens
        )
        if markers
        else None,
        reasoning_close_marker = markers[1] if markers else None,
    )


def _json_schema_from_response_format(response_format: dict) -> dict:
    wrapper = response_format.get("json_schema")
    if isinstance(wrapper, dict):
        schema = wrapper.get("schema")
    elif wrapper is None:
        schema = response_format.get("schema")
    else:
        schema = None
    return _require_schema_object(
        schema,
        "response_format type 'json_schema' requires json_schema.schema to be a "
        "JSON Schema object",
    )


def _require_schema_object(schema, message: str) -> dict:
    if not isinstance(schema, dict):
        raise ResponseFormatError(message)
    return schema


def _reserved_token_text(tokenizer, token_id: int) -> str:
    try:
        return str(_unwrap_hf_tokenizer(tokenizer).convert_ids_to_tokens(int(token_id)))
    except Exception:
        return f"token {int(token_id)}"


def _reasoning_markers(
    tokenizer,
    markers = None,
    tools = None,
) -> Optional[tuple]:
    if markers:
        return tuple(markers)
    if tokenizer is None:
        return None
    from core.inference.chat_template_helpers import detect_reasoning_channel_markers

    try:
        return detect_reasoning_channel_markers(tokenizer, tools = tools) or _THINK_MARKERS
    except Exception:
        return None


def _marker_reference(tokenizer, marker: str, *, reply_keeps_special_tokens: bool) -> Optional[str]:
    if tokenizer is None:
        return None
    inner = _unwrap_hf_tokenizer(tokenizer)
    try:
        ids = inner.encode(marker, add_special_tokens = False)
        if len(ids) != 1:
            return None
        survives = reply_keeps_special_tokens or (
            inner.decode(ids, skip_special_tokens = True) == marker
        )
    except Exception:
        return None
    return marker if survives else None


class GrammarConstraint:
    """One generation's matcher: mask each logits row, advance on each token."""

    def __init__(
        self,
        grammar: str,
        tokenizer,
        *,
        allows_reasoning: bool = False,
    ):
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._matcher = None
        self._bitmask = None
        self._stop_ids = ()
        self.allows_reasoning = allows_reasoning

    def _bind(self, n_vocab: int) -> None:
        self._stop_ids = _runtime_stop_ids(self._tokenizer, n_vocab) or ()
        ll_tokenizer = _cached_ll_tokenizer(self._tokenizer, n_vocab)
        matcher = _llg.LLMatcher(ll_tokenizer, self._grammar)
        error = matcher.get_error()
        if error:
            raise ResponseFormatError(f"response_format grammar rejected: {error}")
        self._matcher = matcher
        self._bitmask = _llg_mlx.allocate_token_bitmask(1, n_vocab)

    def mask_logits(self, logits):
        if self._matcher is None:
            self._bind(int(logits.shape[-1]))
        _llg_mlx.fill_next_token_bitmask(self._matcher, self._bitmask)
        forced_stop = self._stop_offered_as_text()
        if forced_stop is not None:
            raise ResponseFormatError(self._desync_message(forced_stop))
        masked = _llg_mlx.apply_token_bitmask(logits.reshape(1, -1), self._bitmask)
        return masked.reshape(logits.shape)

    def _stop_offered_as_text(self) -> Optional[int]:
        if not self._stop_ids or self._matcher.is_stopped() or self._matcher.is_accepting():
            return None
        return next((i for i in self._stop_ids if self._mask_allowed(i)), None)

    def advance(self, token_id: int) -> None:
        """Commit a sampled token, raising when it desyncs."""
        if self._matcher is None:
            raise GrammarDesyncError("guided decoding advanced before the grammar was bound")
        if self._matcher.is_stopped():
            return
        self._matcher.consume_token(int(token_id))
        error = self._matcher.get_error()
        if error:
            if self._mask_allowed(int(token_id)):
                raise ResponseFormatError(self._desync_message(int(token_id)))
            raise GrammarDesyncError(f"guided decoding desynced on token {int(token_id)}: {error}")

    def _mask_allowed(self, token_id: int) -> bool:
        try:
            import numpy as np
            word = int(np.asarray(self._bitmask)[0, token_id // 32])
        except Exception:
            return False
        return bool(word >> (token_id % 32) & 1)

    def _desync_message(self, token_id: int) -> str:
        reserved = _reserved_token_text(self._tokenizer, token_id)
        return (
            f"response_format could not be honored: the reply reached a point where the "
            f"schema allows only the text {reserved!r}, which this model reserves as a "
            "control token and no grammar can spell as ordinary text. Remove that value "
            "from the schema."
        )


def make_grammar_logits_processor(constraint: GrammarConstraint):
    """``(tokens, logits) -> logits`` masking each step through *constraint*."""
    state = {"seen": None}

    def _processor(tokens, logits):
        seen = state["seen"]
        if seen is None:
            state["seen"] = int(tokens.shape[0])
        else:
            for token_id in tokens[seen:].tolist():
                constraint.advance(int(token_id))
            state["seen"] = int(tokens.shape[0])
        return constraint.mask_logits(logits)

    return _processor


# Bounded, strong-reference caches, so a recycled id() cannot alias a live entry.

_GRAMMAR_CACHE: "OrderedDict[str, str]" = OrderedDict()
_GRAMMAR_CACHE_MAX = 64
_GRAMMAR_CACHE_MAX_BYTES = 8 << 20
_TOKENIZER_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_TOKENIZER_CACHE_MAX = 4
_CACHE_LOCK = threading.Lock()


def _canonical_schema_json(schema: dict) -> str:
    try:
        return json.dumps(schema, sort_keys = True, separators = (",", ":"))
    except (TypeError, ValueError) as exc:
        raise ResponseFormatError(f"response_format schema is not JSON-serializable: {exc}")


def _cached_grammar(schema_json: str, *, prelude_close: Optional[str]) -> str:
    digest = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
    key = f"{LLGUIDANCE_VERSION}:{prelude_close or ''}:" f"{_PRELUDE_MAX_CHARS}:{digest}"
    with _CACHE_LOCK:
        cached = _GRAMMAR_CACHE.get(key)
        if cached is not None:
            _GRAMMAR_CACHE.move_to_end(key)
            return cached
    grammar = _compile_grammar(schema_json, prelude_close)
    with _CACHE_LOCK:
        _GRAMMAR_CACHE[key] = grammar
        cached_bytes = sum(len(value) for value in _GRAMMAR_CACHE.values())
        while _GRAMMAR_CACHE and (
            len(_GRAMMAR_CACHE) > _GRAMMAR_CACHE_MAX or cached_bytes > _GRAMMAR_CACHE_MAX_BYTES
        ):
            _, evicted = _GRAMMAR_CACHE.popitem(last = False)
            cached_bytes -= len(evicted)
    return grammar


def _compile_grammar(schema_json: str, prelude_close: Optional[str]) -> str:
    try:
        if prelude_close is None:
            grammar = _llg.LLMatcher.grammar_from_json_schema(schema_json)
        else:
            grammar = (
                "%llguidance {}\n"
                "start: prelude doc\n"
                f"prelude: PRELUDE_TEXT {prelude_close}\n"
                f"PRELUDE_TEXT: /(.|\\n){{0,{_PRELUDE_MAX_CHARS}}}/\n"
                f"doc: %json {schema_json}\n"
            )
    except Exception as exc:
        raise ResponseFormatError(f"unsupported JSON Schema: {exc}") from exc
    error = _llg.LLMatcher.validate_grammar(grammar)
    if error:
        raise ResponseFormatError(f"unsupported JSON Schema: {error}")
    return grammar


def _unwrap_hf_tokenizer(tokenizer):
    import transformers

    if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        return tokenizer
    inner = getattr(tokenizer, "_tokenizer", None)
    if isinstance(inner, transformers.PreTrainedTokenizerFast):
        return inner
    inner = getattr(tokenizer, "tokenizer", None)
    if isinstance(inner, transformers.PreTrainedTokenizerFast):
        return inner
    return tokenizer


def _runtime_stop_ids(tokenizer, n_vocab: int) -> Optional[tuple]:
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is None:
            continue
        for ids in (
            getattr(obj, "eos_token_ids", None),
            getattr(getattr(obj, "stopping_criteria", None), "eos_token_ids", None),
        ):
            try:
                found = tuple(sorted({int(i) for i in ids if 0 <= int(i) < int(n_vocab)}))
            except Exception:
                continue
            if found:
                return found
    return None


def _cached_ll_tokenizer(tokenizer, n_vocab: int):
    stop_ids = _runtime_stop_ids(tokenizer, n_vocab)
    tokenizer = _unwrap_hf_tokenizer(tokenizer)
    key = (id(tokenizer), int(n_vocab), stop_ids)
    with _CACHE_LOCK:
        entry = _TOKENIZER_CACHE.get(key)
        if entry is not None and entry[0] is tokenizer:
            _TOKENIZER_CACHE.move_to_end(key)
            return entry[1]
    ll_tokenizer = _llg_hf.from_tokenizer(
        tokenizer,
        n_vocab = int(n_vocab),
        eos_token = list(stop_ids) if stop_ids else None,
    )
    with _CACHE_LOCK:
        _TOKENIZER_CACHE[key] = (tokenizer, ll_tokenizer)
        while len(_TOKENIZER_CACHE) > _TOKENIZER_CACHE_MAX:
            _TOKENIZER_CACHE.popitem(last = False)
    return ll_tokenizer
