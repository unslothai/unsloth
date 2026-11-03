# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Message-aware rolling context helpers for local chat inference."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Optional

_OMITTED_TOOL_EXCHANGE = "[Earlier tool exchange omitted from the rolling context window.]"
_UNPRICED_MEDIA_TYPES = frozenset(
    ("image_url", "input_audio", "audio", "input_image", "input_video", "video_url")
)

# Trim BELOW the budget, else the boundary creeps every turn and the prefix cache dies.
_COMPACTION_HEADROOM_RATIO = max(
    0.0, min(0.9, float(os.environ.get("ROLLING_COMPACTION_HEADROOM_RATIO", "0.25")))
)


def _message_without_unpriced_media(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    countable = [
        part
        for part in content
        if not (isinstance(part, dict) and part.get("type") in _UNPRICED_MEDIA_TYPES)
    ]
    if len(countable) == len(content):
        return message
    copy = dict(message)
    copy["content"] = countable or ""
    return copy


_IMAGE_PART_TYPES = frozenset(("image_url", "input_image"))
# One token per repeat on the tokenizers Studio serves, so N repeats count as N tokens (#14968).
_IMAGE_STAND_IN_TOKEN = " the"


def _image_stand_in(tokens: int) -> dict:
    return {"type": "text", "text": _IMAGE_STAND_IN_TOKEN * max(0, int(tokens))}


def _message_with_priced_images(message: dict, price_image: Callable[[dict], int]) -> dict:
    """Counting copy only: images become stand-ins of their price, audio and video are dropped."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    countable = []
    changed = False
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _IMAGE_PART_TYPES:
            countable.append(_image_stand_in(price_image(part)))
            changed = True
        elif kind in _UNPRICED_MEDIA_TYPES:
            changed = True
        else:
            countable.append(part)
    if not changed:
        return message
    copy = dict(message)
    copy["content"] = countable or ""
    return copy


def messages_with_priced_images(
    messages: list[dict], price_image: Callable[[dict], int]
) -> list[dict]:
    """``messages_without_unpriced_media`` with images charged instead of dropped."""
    priced = [_message_with_priced_images(message, price_image) for message in messages]
    return messages if all(before is after for before, after in zip(messages, priced)) else priced


def estimate_message_tokens_with_priced_images(
    message: dict, price_image: Callable[[dict], int]
) -> int:
    """Added rather than rendered, so the estimate never serialises a long stand-in."""
    content = message.get("content")
    images = (
        sum(
            price_image(part)
            for part in content
            if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES
        )
        if isinstance(content, list)
        else 0
    )
    return estimate_message_tokens_without_unpriced_media(message) + images


def image_dimensions(part: dict) -> Optional[tuple[int, int]]:
    """(width, height) of an inline image part, or None when it cannot be read cheaply."""
    # Chat Completions nests {"url": ...}; Responses' input_image carries the string directly.
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    return _data_url_dimensions(url)


_DATA_URL_DIMENSIONS: dict[tuple[int, int], Optional[tuple[int, int]]] = {}
_DATA_URL_DIMENSIONS_MAX = 256


def _data_url_dimensions(url: str) -> Optional[tuple[int, int]]:
    """Cached because the same image is priced on every count of a tool loop. Keyed on the URL's
    hash, not the URL: an lru_cache would keep up to 256 multi-MB base64 payloads alive."""
    key = (hash(url), len(url))
    try:
        return _DATA_URL_DIMENSIONS[key]
    except KeyError:
        pass
    dims = _read_data_url_dimensions(url)
    if len(_DATA_URL_DIMENSIONS) >= _DATA_URL_DIMENSIONS_MAX:
        _DATA_URL_DIMENSIONS.clear()
    _DATA_URL_DIMENSIONS[key] = dims
    return dims


def _read_data_url_dimensions(url: str) -> Optional[tuple[int, int]]:
    try:
        import base64
        import io

        from PIL import Image

        _, _, payload = url.partition(",")
        with Image.open(io.BytesIO(base64.b64decode(payload))) as img:
            width, height = img.size
    except Exception:
        return None
    return (int(width), int(height)) if width > 0 and height > 0 else None


def estimate_message_tokens(message: dict) -> int:
    try:
        return max(1, len(json.dumps(message, ensure_ascii = False)) // 4)
    except Exception:
        return 1


def estimate_message_tokens_without_unpriced_media(message: dict) -> int:
    return estimate_message_tokens(_message_without_unpriced_media(message))


def estimate_messages_tokens_without_unpriced_media(messages: list[dict]) -> int:
    return sum(estimate_message_tokens_without_unpriced_media(message) for message in messages)


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _ascii_count(text: str) -> int:
    """Number of code points <= 127; ``errors = "ignore"`` drops every other one, surrogates included."""
    return len(text) if text.isascii() else len(text.encode("ascii", "ignore"))


def estimate_messages_tokens_dense(messages: list[dict]) -> int:
    """Estimate charging non-ASCII a token per character (2.1x undercount measured on CJK). Not the
    default: eviction must not be pessimistic."""
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        dense = len(text) - _ascii_count(text)
        total += max(1, dense + (len(text) - dense) // 4)
    return total


# Base64/hex/minified JSON run 1.1-2.8 chars/token vs 3.3 for English; 64 since base64 wraps at 76.
_DENSE_RUN_CHARS = 64
# Two characters per token, not one: stays below the measured cost of every sample, so no turn is over-priced.
_DENSE_RUN_CHARS_PER_TOKEN = 2
_DENSE_RUN_RE = re.compile(r"\S{%d,}" % _DENSE_RUN_CHARS)


def estimate_messages_tokens_conservative(
    messages: list[dict], *, dense_ascii: bool = False
) -> int:
    """As above, but long unbroken ASCII runs are charged as blobs; ``dense_ascii`` charges all
    ASCII that way."""
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        wide = len(text) - _ascii_count(text)
        if dense_ascii:
            total += max(1, wide + (len(text) - wide) // _DENSE_RUN_CHARS_PER_TOKEN)
            continue
        # ASCII only: a run of CJK is already charged a token a character above.
        runs = sum(_ascii_count(match.group(0)) for match in _DENSE_RUN_RE.finditer(text))
        plain = len(text) - wide - runs
        total += max(1, wide + runs // _DENSE_RUN_CHARS_PER_TOKEN + plain // 4)
    return total


def group_turns(messages: list[dict]) -> list[list[dict]]:
    """Split messages into the turn groups the rolling window evicts as single units."""
    groups: list[list[dict]] = []
    for message in messages:
        starts_tool_exchange = message.get("role") == "assistant" and bool(
            message.get("tool_calls")
        )
        follows_instruction = bool(groups and groups[-1][0].get("role") in ("system", "developer"))
        if (
            message.get("role") in ("system", "developer", "user")
            or starts_tool_exchange
            or follows_instruction
        ):
            groups.append([message])
        elif not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def evicted_messages(before: list[dict], after: list[dict]) -> list[dict]:
    """Messages in ``before`` and not ``after``. Identity, not equality: two turns can be
    byte-identical."""
    kept = {id(message) for message in after}
    return [message for message in before if id(message) not in kept]


def truncate_oldest_messages(
    messages: list[dict],
    keep_ratio: float,
    *,
    protected_message_ids: Optional[set[int]] = None,
    min_dropped: int = 0,
    estimate_message: Callable[[dict], int] = estimate_message_tokens,
) -> tuple[list[dict], int]:
    """Drop oldest turns, keeping system messages and the latest. ``min_dropped`` evicts past the
    fit, so a boundary can be re-applied."""
    if not messages or (keep_ratio >= 1.0 and min_dropped <= 0):
        return messages, 0

    groups = group_turns(messages)

    if len(groups) <= 1:
        return messages, 0

    estimates = {id(message): estimate_message(message) for message in messages}
    current_estimate = sum(estimates.values())
    target_estimate = int(current_estimate * max(0.0, keep_ratio))
    dropped = 0
    protected_ids = protected_message_ids or set()
    latest_user_group = next(
        (
            index
            for index in range(len(groups) - 1, -1, -1)
            if any(message.get("role") == "user" for message in groups[index])
        ),
        None,
    )
    protected_groups = {
        index
        for index, group in enumerate(groups)
        if index == len(groups) - 1
        or index == latest_user_group
        or any(message.get("role") in ("system", "developer") for message in group)
        or any(id(message) in protected_ids for message in group)
    }
    eviction_units: list[list[int]] = []
    index = 0
    while index < len(groups):
        if index in protected_groups:
            index += 1
            continue
        unit = [index]
        starts_user_turn = groups[index][0].get("role") == "user"
        next_index = index + 1
        if starts_user_turn:
            while next_index < len(groups) and groups[next_index][0].get("role") not in (
                "system",
                "developer",
                "user",
            ):
                unit.append(next_index)
                next_index += 1
        if not any(group_index in protected_groups for group_index in unit):
            eviction_units.append(unit)
        index = next_index if starts_user_turn else index + 1

    dropped_groups: set[int] = set()
    for unit in eviction_units:
        if current_estimate <= target_estimate and dropped >= min_dropped:
            break
        dropped_groups.update(unit)
        for group_index in unit:
            group = groups[group_index]
            dropped += len(group)
            current_estimate -= sum(estimates[id(message)] for message in group)

    if dropped == 0:
        return messages, 0

    kept: list[dict] = []
    for index, group in enumerate(groups):
        if index not in dropped_groups:
            if kept and kept[-1].get("role") == "user" and group and group[0].get("role") == "user":
                # Strict chat templates reject adjacent user turns.
                kept.append({"role": "assistant", "content": _OMITTED_TOOL_EXCHANGE})
            kept.extend(group)
    return kept, dropped


def messages_without_unpriced_media(messages: list[dict]) -> list[dict]:
    """Text-only lower bound: ``/apply-template`` misses tokens the multimodal processor adds later.
    The request still sends every media part."""
    stripped = [_message_without_unpriced_media(message) for message in messages]
    return (
        messages if all(before is after for before, after in zip(messages, stripped)) else stripped
    )


def prompt_budget(context_length: int, max_tokens: Optional[int]) -> int:
    """Tokens available to the PROMPT once reply room is set aside. Exported so recall sizing and
    the client explanation share one formula."""
    if context_length <= 1:
        return context_length
    requested = max_tokens if max_tokens is not None and max_tokens > 0 else context_length
    return context_length - min(requested, max(1, context_length // 4))


_RETRIEVAL_BUDGET_SHARE = 0.5

# Small on purpose: missing the reserve is survivable, so this only rules out the stub-answer end.
_RESCUE_REPLY_FLOOR_DIVISOR = 16


def retrieval_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    reply_returns: bool = False,
) -> int:
    """Prompt room for one retrieval: in a tool loop the exchange and reply are protected next fit,
    so at most half the budget."""
    target = prompt_budget(context_length, max_tokens)
    room = max(0, target - int(prompt_tokens or 0))
    if reply_returns:
        room = min(room, int(target * _RETRIEVAL_BUDGET_SHARE))
    return room


# Headroom for the tokenizer disagreeing with the character-based estimate that sized a result.
_TOOL_RESULT_BUDGET_BUFFER = 0.99

# Charged only when the result is really cut, never up front.
_RESULT_NOTICE_RESERVE = 128


def tool_result_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    buffer: float = _TOOL_RESULT_BUDGET_BUFFER,
) -> int:
    """Tokens a tool result may add without pushing the next prompt over. Priced against room
    remaining, not a share of the window, and against ``prompt_budget`` so the reply still has
    room. Covers the whole tool message, notice included."""
    target = prompt_budget(context_length, max_tokens)
    return max(0, int(target * buffer) - int(prompt_tokens or 0))


def turn_is_servable(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    buffer: float = _TOOL_RESULT_BUDGET_BUFFER,
) -> bool:
    """Whether the next prompt fits given an EMPTY result. `tool_result_budget` clamps at zero,
    which reads as cut-hard rather than as the refusal it is."""
    if context_length <= 1:
        return True
    return prompt_tokens + _RESULT_NOTICE_RESERVE + _reply_floor(context_length) <= context_length


def _reply_floor(context_length: int) -> int:
    """Least reply room a turn must leave to be worth running. Not `prompt_budget`, which reserves
    all of `max_tokens` and refused 3,504-token turns at a 4096 window; llama-server admits on
    size alone (`n_tokens() >= n_ctx`)."""
    return max(1, context_length // _RESCUE_REPLY_FLOOR_DIVISOR)


# How much must be at stake before a receipt is worth the edit; below this the placeholder is a wash.
_PATH_KEYS = frozenset({"path", "file_path", "filePath"})
_RECEIPT_PATH_MAX_CHARS = 120

_ARG_COMPACTION_FLOOR_CHARS = 1024
# Not zero: a receipt is about 100 characters, so eliding anything shorter grows the call.
_ARG_COMPACTION_AGGREGATE_LEAF_FLOOR = 256
# Matches the per-leaf floor: the same amount of window either way.
_ARG_COMPACTION_TOTAL_FLOOR_CHARS = 1024


def _total_leaves(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_total_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(_total_leaves(item) for item in value)
    return 0


def _compacted_arguments(
    name: str,
    arguments: str,
    phrase: Optional[str] = None,
    reply: object = None,
) -> Optional[str]:
    """Receipt standing in for a completed call's arguments, or None to leave them. Structured and
    naming the path, since a bare [omitted] reads as failure and draws a retry of the same
    oversized write."""
    # Not a default: the constant is defined below.
    phrase = phrase or _completed_phrase_for(name, reply)
    if not isinstance(arguments, str):
        return None
    # A refused call ignores the general floor: its refusal is about to enter a prompt that already does not fit. The
    # size check at the end still stops the receipt growing the prompt.
    refused = phrase == _REFUSED_PHRASE
    if not refused and len(arguments) < _ARG_COMPACTION_TOTAL_FLOOR_CHARS:
        return None
    try:
        parsed = json.loads(arguments)
    except Exception:
        # Size alone is an honest receipt for unparseable arguments. Worded from `phrase`: hardcoding
        # after-the-call-ran replayed a REFUSED call as having run.
        _unparseable = json.dumps(
            {"_unsloth_compacted": f"{len(arguments)} chars {phrase.format(where = '')}"},
            ensure_ascii = False,
        )
        # Checked here as well as at the end: without the general floor a short refused call can get a receipt longer
        # than what it replaces.
        return _unparseable if len(_unparseable) < len(arguments) else None
    if not isinstance(parsed, dict):
        return None
    path = parsed.get("path") or parsed.get("file_path") or parsed.get("filePath")
    elided = 0

    # Chosen from the TOTAL so many medium edits still compact.
    _leaf_floor = (
        # A refused call takes what it can get, floored only where a leaf is shorter than its receipt.
        _REFUSED_LEAF_FLOOR
        if refused
        else _ARG_COMPACTION_AGGREGATE_LEAF_FLOOR
        if _total_leaves(parsed) >= _ARG_COMPACTION_TOTAL_FLOOR_CHARS
        else _ARG_COMPACTION_FLOOR_CHARS
    )

    def _shrink(value: Any, key: str = "") -> Any:
        """Elide every large string at any depth: `edit_file` takes an `edits` ARRAY, so content
        sits at `edits[i].new_string`."""
        nonlocal elided
        # The destination is never expendable: it names WHICH file the call touched, and the receipt promises the
        # content is there.
        if key in _PATH_KEYS:
            return value
        if isinstance(value, str) and len(value) >= _leaf_floor:
            elided += len(value)
            # Repeated in each leaf's receipt only when cheaper than the field it points at. `path` is preserved
            # verbatim either way.
            where = (
                f" to {path}"
                if path and key not in _PATH_KEYS and len(str(path)) <= _RECEIPT_PATH_MAX_CHARS
                else ""
            )
            # `old_string` names text the edit REMOVED; the completed phrasing is true only of `new_string`.
            leaf_phrase = _COMPLETED_NEUTRAL_PHRASE if key == "old_string" else phrase
            return f"<{len(value)} chars {leaf_phrase.format(where = where)}>"
        if isinstance(value, dict):
            return {inner: _shrink(item, inner) for inner, item in value.items()}
        if isinstance(value, list):
            return [_shrink(item, key) for item in value]
        return value

    kept = {key: _shrink(value, key) for key, value in parsed.items()}
    if not elided:
        return None
    try:
        compacted = json.dumps(kept, ensure_ascii = False)
    except Exception:
        return None
    # Never grow the prompt to describe it: bulk spread over many small fields leaves nothing worth eliding.
    return compacted if len(compacted) < len(arguments) else None


# A leaf shorter than its own receipt costs room to elide.
_REFUSED_LEAF_FLOOR = 110
_REFUSED_PHRASE = (
    "of arguments you sent, elided; this call was refused before it ran and nothing was written"
)
# Must not read as tool OUTPUT: the model concluded its file was mangled.
_COMPLETED_PHRASE = "of arguments you sent, already written{where}; elided to save room. Not tool output; the file on disk holds it."
# Same receipt for tools that write no file (`python`, `terminal`, search, MCP): the file wording told the model a
# `code` argument was on disk.
_COMPLETED_NEUTRAL_PHRASE = (
    "of arguments you sent, elided to save room; the call already ran. Not tool output"
)
_FILE_WRITING_TOOLS = frozenset({"edit_file"})

# Bracketed = leaf receipt, bare = `_unsloth_compacted` receipt for unparseable arguments.
_RECEIPT_PHRASES = "|".join(
    re.escape(phrase).replace(r"\{where\}", rf"(?: to [^\n]{{1,{_RECEIPT_PATH_MAX_CHARS}}})?")
    for phrase in (_REFUSED_PHRASE, _COMPLETED_PHRASE, _COMPLETED_NEUTRAL_PHRASE)
)
_RECEIPT_LEAF = re.compile(rf"<\d+ chars (?:{_RECEIPT_PHRASES})>|\d+ chars (?:{_RECEIPT_PHRASES})")


def compaction_receipt_field(
    value: Any,
    where: str = "",
    match_only: "frozenset[str]" = frozenset(),
) -> Optional[str]:
    """First field holding only a receipt, or None; `match_only` keys (e.g. `old_string`) may, to repair files."""
    if isinstance(value, str):
        return (where or "arguments") if _RECEIPT_LEAF.fullmatch(value.strip()) else None
    if isinstance(value, dict):
        items = (
            (f"{where}.{key}" if where else str(key), item)
            for key, item in value.items()
            if key not in match_only
        )
    elif isinstance(value, list):
        items = ((f"{where}[{index}]", item) for index, item in enumerate(value))
    else:
        return None
    for inner, item in items:
        found = compaction_receipt_field(item, inner, match_only)
        if found is not None:
            return found
    return None


# A reply opening like this reports a call that ran and did NOT do what was asked, so the file wording would describe
# a write that never landed.
_FAILED_REPLY_MARKERS = ("error", "failed", "not found", "no such file", "traceback")

# `_fit_result_to_room` swaps even an `Error: ...` for a stub without the markers above.
_INCONCLUSIVE_REPLY_MARKERS = ("no context room left", "chars for the model;")


def _reply_proves_a_write(name: str, content: object) -> bool:
    """The tool NAME is wrong both ways: a failed `edit_file` is not a write and a `python` call can
    be one. File wording needs a file tool AND a non-failure reply."""
    if name not in _FILE_WRITING_TOOLS:
        return False
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    if any(marker in lowered for marker in _INCONCLUSIVE_REPLY_MARKERS):
        return False
    head = content[:200].strip().lower()
    return not any(marker in head for marker in _FAILED_REPLY_MARKERS)


def _completed_phrase_for(name: str, reply: object = None) -> str:
    return _COMPLETED_PHRASE if _reply_proves_a_write(name, reply) else _COMPLETED_NEUTRAL_PHRASE


def compact_executed_call_arguments(messages: list[dict], call_id: str) -> list[dict]:
    """Compacts arguments of one just-run call even if protected; None, not the receipt: it is per tool."""
    return _compact_one_call(messages, call_id, None)


def compact_refused_tool_arguments(messages: list[dict], call_id: str) -> list[dict]:
    """Drop a declined call's arguments, named as never sent. `tool_calls` render only once a `tool`
    message answers them, and the refusal is one, so declining costs the prompt what it declined
    to afford."""
    return _compact_one_call(messages, call_id, _REFUSED_PHRASE)


def _last_index_with_call(messages: list[dict], call_id: str) -> int:
    """Call ids are NOT unique: parsers number from `call_0` each turn, so rewriting every match
    relabels an earlier call's fate. Callers act on the call just decided, so take the last."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        if any(
            isinstance(call, dict) and str(call.get("id") or "") == str(call_id) for call in calls
        ):
            return index
    return -1


def _reply_for_call(messages: list[dict], call_id: str) -> object:
    for message in reversed(messages):
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") == str(call_id):
            return message.get("content")
    return None


def _compact_one_call(
    messages: list[dict],
    call_id: str,
    phrase: Optional[str] = None,
) -> list[dict]:
    """Rewrite one call's arguments to a receipt worded by `phrase`; `None` lets the tool choose,
    since only file tools may claim the content is on disk."""
    if not call_id:
        return messages
    target = _last_index_with_call(messages, call_id)
    if target < 0:
        return messages
    out: list[dict] = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if index != target or not isinstance(calls, list) or not calls:
            out.append(message)
            continue
        new_calls: list[dict] = []
        changed = False
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or str(call.get("id") or "") != str(call_id):
                new_calls.append(call)
                continue
            replacement = _compacted_arguments(
                str(function.get("name") or ""),
                function.get("arguments"),
                phrase,
                reply = _reply_for_call(messages, call_id),
            )
            if replacement is None:
                new_calls.append(call)
                continue
            new_calls.append({**call, "function": {**function, "arguments": replacement}})
            changed = True
        out.append({**message, "tool_calls": new_calls} if changed else message)
    return out


# A `role=tool` reply proves an ANSWER, not an execution: the approval gate answers a declined call with one.
_DID_NOT_RUN_MARKERS = (
    "the user declined to run this tool call",
    "could not be read",
    "nothing ran",
    "nothing was run",
    "nothing was written",
)


def _reply_shows_execution(content: object) -> bool:
    if not isinstance(content, str):
        return True
    head = content[:200].strip().lower()
    return not any(marker in head for marker in _DID_NOT_RUN_MARKERS)


def _executed_call_sites(messages: list[dict]) -> "dict[tuple[int, str], object]":
    """Keyed on the SITE, not the id: generated ids restart at `call_0` each turn, so an earlier
    success once vouched for a call the user DECLINED."""
    pending: dict[str, list[int]] = {}
    executed: dict[tuple[int, str], object] = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    pending.setdefault(str(call["id"]), []).append(index)
            continue
        if role != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not call_id:
            continue
        sites = pending.get(str(call_id))
        if not sites:
            continue
        # Newest pending: a stale older site under the same id would leave the real call uncompactable.
        site = sites.pop()
        if _reply_shows_execution(message.get("content")):
            executed[(site, str(call_id))] = message.get("content")
    return executed


def compact_completed_tool_arguments(
    messages: list[dict], *, protect_last: int = 0
) -> tuple[list[dict], int]:
    """Rewrite only what is REPLAYED; the stored thread and the arguments the tool received stay
    byte-identical. Only answered calls are touched, oldest first; ``protect_last`` holds
    trailing messages clear."""
    answered = _executed_call_sites(messages)
    if not answered:
        return messages, 0

    limit = len(messages) - int(protect_last or 0)
    out: list[dict] = []
    compacted_calls = 0
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") if index < limit else None
        if message.get("role") != "assistant" or not isinstance(calls, list) or not calls:
            out.append(message)
            continue
        new_calls: list[dict] = []
        changed = False
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or (index, str(call.get("id") or "")) not in answered:
                new_calls.append(call)
                continue
            replacement = _compacted_arguments(
                str(function.get("name") or ""),
                function.get("arguments"),
                reply = answered[(index, str(call.get("id") or ""))],
            )
            if replacement is None:
                new_calls.append(call)
                continue
            new_calls.append({**call, "function": {**function, "arguments": replacement}})
            changed = True
            compacted_calls += 1
        out.append({**message, "tool_calls": new_calls} if changed else message)
    return (out, compacted_calls) if compacted_calls else (messages, 0)


def _blamed_role(message: dict) -> str:
    """Advice key: role, except an assistant turn with `tool_calls`, where start-a-new-reply is the
    wrong lever for an 8 KB payload."""
    role = str(message.get("role") or "")
    if role == "assistant" and message.get("tool_calls"):
        # Split by whether a FILE is involved, from the call that accounts for the turn's SIZE: ask-for-a-smaller-file
        # cannot shrink a `python` or MCP payload.
        _dominant = None
        _dominant_size = -1
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            _size = len(str(function.get("arguments") or ""))
            if _size > _dominant_size:
                _dominant_size = _size
                _dominant = str(function.get("name") or "")
        if _dominant in _FILE_WRITING_TOOLS:
            return "assistant_tool_call"
        return "assistant_tool_payload"
    return role


def _blamed_role_for_turn(messages: list[dict]) -> str:
    """Blame the newest TURN, not the newest message: a `tool_calls` block renders only once
    answered, so the reply's marginal cost includes the call's arguments."""
    if not messages:
        return ""
    latest = messages[-1]
    if str(latest.get("role") or "") != "tool":
        return _blamed_role(latest)
    call_id = latest.get("tool_call_id")
    for message in reversed(messages[:-1]):
        role = str(message.get("role") or "")
        if role != "assistant":
            # Anything between the call and this reply means the reply belongs to no call.
            if role == "tool":
                continue
            break
        if not message.get("tool_calls"):
            break
        if call_id and not any(
            isinstance(call, dict) and str(call.get("id") or "") == str(call_id)
            for call in message.get("tool_calls") or []
        ):
            break
        # Only when the CALL is the bigger half: a dominant tool result still gets the tool advice.
        _call_chars = sum(
            len(str((call.get("function") or {}).get("arguments") or ""))
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        )
        _reply_chars = len(str(latest.get("content") or ""))
        if _call_chars > _reply_chars:
            return _blamed_role(message)
        break
    return _blamed_role(latest)


def _latest_turn_count(
    messages: list[dict], count_tokens: Callable[[list[dict]], int]
) -> tuple[int, bool]:
    """Estimated when the template refuses a lone tool result. The caller must know which it got:
    only the counted one carries the prompt's floor."""
    if not messages:
        return 0, False
    try:
        return int(count_tokens(messages[-1:])), True
    except Exception:
        return int(estimate_messages_tokens(messages[-1:])), False


def _shared_prompt_tokens(count_tokens: Callable[[list[dict]], int]) -> int:
    """What a rendered prompt costs before any message: the template wrapper plus the whole tool
    catalogue, so it sits inside every count. Measured, not estimated, since it is subtracted
    from the counts that decide the blame."""
    try:
        return max(0, int(count_tokens([])))
    except Exception:
        return 0


def _marginal_turn_count(
    fitted: list[dict], count_tokens: Callable[[list[dict]], int], irreducible_tokens: int
) -> Optional[int]:
    """What the newest turn ADDED, by difference against the same prompt without it, so the floor
    cancels. `irreducible_tokens` must be the count of `fitted`. None when the prefix cannot be
    priced."""
    try:
        return int(irreducible_tokens) - int(count_tokens(list(fitted)[:-1]))
    except Exception:
        return None


def turn_diagnosis(
    messages: list[dict],
    count_tokens: Callable[[list[dict]], int],
    *,
    irreducible_tokens: int,
    fitted: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """latest_turn_exact False means the two counts use different units and must not be compared."""
    if not messages:
        return {
            "latest_turn_tokens": 0,
            "latest_turn_role": "",
            "shared_prompt_tokens": 0,
            "latest_turn_exact": True,
        }
    latest, exact = _latest_turn_count(messages, count_tokens)
    shared = _shared_prompt_tokens(count_tokens) if exact else 0
    if exact and latest <= shared:
        # Gemma-4 renders a lone tool turn as nothing, so price it by difference, reported floor-inclusive.
        marginal = _marginal_turn_count(
            fitted if fitted is not None else messages, count_tokens, irreducible_tokens
        )
        if marginal is not None and marginal > 0:
            latest = marginal + shared
        else:
            # Nothing countable left: price the message's own JSON, and record no floor, because that estimate carries
            # none.
            latest = int(estimate_messages_tokens(messages[-1:]))
            shared = 0
            # What is REPORTED is now the estimate, and that is what the flag describes.
            exact = False
    # Never all of either side: a floor at or above them would leave no ratio to compare.
    shared = max(0, min(shared, latest - 1, int(irreducible_tokens) - 1))
    return {
        "latest_turn_tokens": latest,
        "latest_turn_role": _blamed_role_for_turn(messages),
        "shared_prompt_tokens": shared,
        "latest_turn_exact": bool(exact),
    }


def clamp_compaction_headroom_ratio(value: Any) -> Optional[float]:
    """Overrides pass the same ``[0, 0.9]`` clamp as the process default, so a UI slider cannot ask
    the fitter to drop the whole prompt."""
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if ratio != ratio:  # NaN
        return None
    return max(0.0, min(0.9, ratio))


def fit_rolling_context(
    messages: list[dict],
    *,
    context_length: int,
    max_tokens: Optional[int],
    count_tokens: Callable[[list[dict]], int],
    protected_message_ids: Optional[set[int]] = None,
    reserve_tokens: int = 0,
    sticky_dropped: int = 0,
    keeps_boundary: bool = False,
    headroom_ratio: Optional[float] = None,
    estimate_message: Callable[[dict], int] = estimate_message_tokens,
    replay_only: bool = False,
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """Drops oldest turns; reserve_tokens never affects trimming; sticky_dropped replays the boundary."""
    if context_length <= 1:
        return messages, None

    prompt_target = prompt_budget(context_length, max_tokens)
    fitted = list(messages)
    initial_tokens = count_tokens(fitted)
    current_tokens = initial_tokens
    dropped_total = 0

    # Phase one, gated on the prompt not already fitting: a saved boundary describes the branch it was measured on,
    # and after a rollback would evict a chat that fits.
    if sticky_dropped > 0 and initial_tokens > prompt_target:
        candidate, dropped = truncate_oldest_messages(
            fitted,
            1.0,
            protected_message_ids = protected_message_ids,
            min_dropped = sticky_dropped,
            estimate_message = estimate_message,
        )
        if dropped:
            fitted = candidate
            dropped_total = dropped
            current_tokens = count_tokens(fitted)

    if replay_only:
        return fitted, {
            "dropped_messages": dropped_total,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": current_tokens,
            "context_length": context_length,
            "fits": current_tokens <= prompt_target,
        }

    # Phase two: take a chunk out rather than skimming to the brim, so the boundary can stay put.
    trim_target = prompt_target
    # Keyed on the ratio, not `headroom`, which is zeroed for threadless and incognito requests that chose nothing.
    min_bite = True
    if current_tokens > prompt_target:
        # Summed, not max()'d: the reserve is spent at once on recalled passages. Only for callers that can restore
        # the boundary, since a deeper cut pays only if remembered.
        ratio = clamp_compaction_headroom_ratio(headroom_ratio)
        if ratio is None:
            ratio = _COMPACTION_HEADROOM_RATIO
        min_bite = ratio > 0
        headroom = int(prompt_target * ratio) if keeps_boundary else 0
        trim_target = max(1, prompt_target - reserve_tokens - headroom)

    while current_tokens > trim_target:
        keep_ratio = trim_target / max(1, current_tokens)
        if min_bite:
            keep_ratio = min(0.95, keep_ratio)
        candidate, dropped = truncate_oldest_messages(
            fitted,
            keep_ratio,
            protected_message_ids = protected_message_ids,
            estimate_message = estimate_message,
        )
        if dropped == 0:
            break
        fitted = candidate
        dropped_total += dropped
        current_tokens = count_tokens(fitted)

    if current_tokens > prompt_target:
        # Stay strictly under n_ctx (llama-server refuses at it); one token over loses history and answer.
        reply_floor = min(
            max(1, context_length // _RESCUE_REPLY_FLOOR_DIVISOR),
            context_length - prompt_target,
        )
        rescued = (
            dropped_total > 0
            and initial_tokens >= context_length
            and current_tokens + reply_floor <= context_length
        )
        return (fitted if rescued else messages), {
            "fits": False,
            "dropped_messages": dropped_total if rescued else 0,
            "prompt_tokens_before": initial_tokens,
            "prompt_tokens_after": current_tokens if rescued else initial_tokens,
            # FluxTransformerBlock / FluxSingleTransformerBlock: modulation goes via AdaLayerNormZero; this fuses the inline norm2 modulation and gated residual adds
            "irreducible_tokens": current_tokens,
            # `fitted` is what `current_tokens` prices, so the turn can be counted by difference rather than estimated
            **turn_diagnosis(
                messages, count_tokens, irreducible_tokens = current_tokens, fitted = fitted
            ),
            "context_length": context_length,
            "prompt_target": prompt_target,
        }
    if dropped_total == 0:
        return messages, None
    return fitted, {
        "dropped_messages": dropped_total,
        "prompt_tokens_before": initial_tokens,
        "prompt_tokens_after": current_tokens,
        "context_length": context_length,
        "fits": True,
    }


# withSandboxAttachmentPaths' note (sandbox-attachments.ts): only these attachments have a copy to point at.
_SANDBOX_ATTACHMENT_NOTE_RE = re.compile(
    r"\[(?P<name>[^\n]*?): its text is below, so answer from it\. For calculations, the python tool has the "
    r'file at path = (?P<path>"(?:[^"\\\n]|\\.)*")(?:;[^\n]*)?\]\n'
)
# The frontend's wrapper tag (attachment, pasted_text, document_annotations); the cut goes inside it.
_ATTACHMENT_OPEN_RE = re.compile(r"<(?P<tag>[A-Za-z_][\w-]*)(?:\s[^\n]*)?>\n")
# Room left for the fit's own additions (carried block, recall), so a cut turn is not refused one step later.
_ATTACHMENT_FIT_MARGIN = 0.125
# No tokenizer packs more characters than this into one token, so a longer body cannot fit and is never counted.
_MAX_CHARS_PER_TOKEN = 16


def _attachment_spans(text: str):
    """(note, open tag, inner text, close tag, trailing text) per sandbox-backed attachment."""
    notes = list(_SANDBOX_ATTACHMENT_NOTE_RE.finditer(text))
    for index, note in enumerate(notes):
        end = notes[index + 1].start() if index + 1 < len(notes) else len(text)
        body = text[note.end() : end]
        opened = _ATTACHMENT_OPEN_RE.match(body)
        closing = f"\n</{opened.group('tag')}>" if opened else ""
        close = body.rfind(closing) if opened else -1
        if opened and close >= opened.end():
            yield (
                note,
                opened.group(0),
                body[opened.end() : close],
                closing,
                body[close + len(closing) :],
            )
        else:
            yield note, "", body, "", ""


def _cut_attachment_bodies(text: str, keep_chars: int, *, excerpt: bool) -> str:
    """Attachments longer than ``keep_chars`` keep their first whole lines (``excerpt``) or only their path."""
    first = _SANDBOX_ATTACHMENT_NOTE_RE.search(text)
    if not first:
        return text
    out = [text[: first.start()]]
    for note, opened, inner, close, tail in _attachment_spans(text):
        name, path = note.group("name"), note.group("path")
        if len(inner) <= keep_chars:
            out.append(note.group(0) + opened + inner + close + tail)
            continue
        total = inner.count("\n") + (0 if inner.endswith("\n") else 1)
        kept = inner[:keep_chars] if excerpt else ""
        kept = kept[: kept.rindex("\n") + 1] if "\n" in kept else ""
        where = (
            f"The whole file is at path = {path} in the python tool's working directory, so read the part you "
            "need with python"
        )
        if kept:
            out.append(
                f"{note.group(0)}{opened}{kept[:-1]}{close}\n[{name} is too long for the context window: "
                f"only its first {kept.count(chr(10))} of {total} lines are above. {where} instead of answering "
                f"from this excerpt.]{tail}"
            )
        else:
            out.append(
                f"[{name} ({total} lines) is too long for the context window to show. {where}.]{tail}"
            )
    return "".join(out)


def _cut_content(content: Any, keep_chars: int, *, excerpt: bool) -> Any:
    if isinstance(content, str):
        return _cut_attachment_bodies(content, keep_chars, excerpt = excerpt)
    if isinstance(content, list):
        return [
            {**part, "text": _cut_attachment_bodies(part["text"], keep_chars, excerpt = excerpt)}
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            else part
            for part in content
        ]
    return content


def _attachment_body_chars(message: dict) -> int:
    content = message.get("content")
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        texts = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
    else:
        return 0
    return max((len(span[2]) for text in texts for span in _attachment_spans(text)), default = 0)


def fit_oversized_attachments(
    messages: list[dict],
    *,
    context_length: int,
    max_tokens: Optional[int],
    count_tokens: Callable[[list[dict]], int],
) -> list[dict]:
    """Cut sandbox-backed attachments too big for the window: the newest turn keeps the start, older turns the path.

    Decided per message, so a turn renders the same on every later request. Rewrites ``content`` in place:
    eviction matches messages by identity, and a copy of the newest turn was archived as evicted.
    """
    if not context_length or context_length <= 1:
        return messages
    attached = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and _attachment_body_chars(message)
    ]
    if not attached:
        return messages
    budget = prompt_budget(context_length, max_tokens)
    # The margin sizes a cut; it never decides one, or a file that fits would lose its end.
    target = int(budget * (1 - _ATTACHMENT_FIT_MARGIN))
    if count_tokens(messages) <= budget:
        return messages
    latest = max(index for index, message in enumerate(messages) if message.get("role") == "user")
    base = [message for message in messages if message.get("role") == "system"]
    for index in attached:
        message = messages[index]
        excerpt = index == latest
        longest = _attachment_body_chars(message)

        def fits(keep_chars: int, limit: int = target) -> bool:
            content = _cut_content(message["content"], keep_chars, excerpt = excerpt)
            return count_tokens([*base, {**message, "content": content}]) <= limit

        cap = target * _MAX_CHARS_PER_TOKEN
        if (longest <= budget * _MAX_CHARS_PER_TOKEN and fits(longest, budget)) or (
            excerpt and not fits(0, budget)
        ):
            continue
        low, high = 0, min(longest, cap) if fits(0) else 0
        while high - low > 64:
            middle = (low + high) // 2
            if fits(middle):
                low = middle
            else:
                high = middle
        message["content"] = _cut_content(message["content"], low, excerpt = excerpt)
    return messages
