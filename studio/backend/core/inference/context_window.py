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
    ("image_url", "input_audio", "audio", "input_image", "input_video")
)

# How far BELOW the prompt budget a compaction trims, as a fraction of that budget.
# Trimming to exactly the budget puts the next turn over it again, so the boundary creeps
# forward every turn: llama-server's prefix cache dies each time and there is no discrete
# compaction event to report. Taking a chunk out in one go buys a stretch of turns with a
# fixed head, at the cost of the headroom itself, hence a minority of the budget.
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


def estimate_messages_tokens_dense(messages: list[dict]) -> int:
    """The same estimate, but honest about text that tokenises densely.

    Four characters per token is about right for English and badly wrong for CJK and
    emoji, which run closer to one token per character. Measured on an 81-message CJK
    chat: 1295 estimated against 2737 real, a 2.1x undercount, which a caller sizing a
    search budget spends as room it does not have. `_conversation_search_tokens` already
    charges non-ASCII a token each when pricing the result; this is the same rule applied
    to the spend, which is the half that was missing.

    Deliberately NOT the default. `truncate_oldest_messages` evicts on the flat estimate,
    and making eviction pessimistic would drop history a request could have kept. Only a
    caller deciding how much room is LEFT wants this, where erring high is the safe side.
    """
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        dense = sum(1 for char in text if ord(char) > 127)
        total += max(1, dense + (len(text) - dense) // 4)
    return total


# ASCII that runs this far without a break is not prose. Measured with Qwen3 over 16-20k
# character samples, in characters per token: base64 1.35, hex 1.13, minified JSON 2.75,
# against English prose at 3.27 and Python source at 4.38. The estimate above charges all
# of them the English four, so a pasted blob is priced at a third of what it costs, and a
# caller sizing the room LEFT then hands out room that is already occupied.
#
# The rule is per RUN rather than per message, so a blob pasted into a sentence is priced
# as a blob: a run this long is the thing itself. 64 characters rather than a rounder
# number because base64 is conventionally wrapped at 76, and at a threshold of 80 a
# wrapped blob scores as ordinary prose. Measured on the samples above, runs of 64+
# non-space characters cover 100% of an unwrapped blob, 98.6% of a wrapped one, 0% of the
# Python source and 23.5% of a README -- and that 23.5% is its URLs and table rules, which
# tokenise near this rate themselves rather than at the prose rate.
_DENSE_RUN_CHARS = 64
# Two characters per token, not one. It stays BELOW the measured cost of every sample, so
# no turn is priced above what it really costs: over-pricing a turn spends the very room
# this budget exists to hand out, and a result cut to pay for room that was never occupied
# is the same waste from the other side.
_DENSE_RUN_CHARS_PER_TOKEN = 2
_DENSE_RUN_RE = re.compile(r"\S{%d,}" % _DENSE_RUN_CHARS)


def estimate_messages_tokens_conservative(
    messages: list[dict], *, dense_ascii: bool = False
) -> int:
    """`estimate_messages_tokens_dense`, with unbroken ASCII runs charged as the blobs
    they are.

    For the caller with no tokenizer and no rolling fit behind it: the native loop prices
    what a thread has already spent, and if that number is low the result it then admits
    is what pushes the next prompt past the window, with nothing downstream to recover.
    Non-ASCII keeps its token per character, ordinary ASCII keeps the English four, and
    only the long unbroken runs are charged at `_DENSE_RUN_CHARS_PER_TOKEN`.

    ``dense_ascii`` charges ALL of a message's ASCII at that rate, for the messages that
    are dense whether or not they hold an unbroken run: a tool result is `hexdump`, `ls
    -l` or a stack trace as often as it is a blob, and those are space-separated. It
    applies to the ASCII only, so a CJK result is not charged twice for being a result.
    """
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        wide = sum(1 for char in text if ord(char) > 127)
        if dense_ascii:
            total += max(1, wide + (len(text) - wide) // _DENSE_RUN_CHARS_PER_TOKEN)
            continue
        # ASCII only: a run of CJK is already charged a token a character above, and
        # charging it here as well would price it twice.
        runs = sum(
            sum(1 for char in match.group(0) if ord(char) <= 127)
            for match in _DENSE_RUN_RE.finditer(text)
        )
        plain = len(text) - wide - runs
        total += max(1, wide + runs // _DENSE_RUN_CHARS_PER_TOKEN + plain // 4)
    return total


def group_turns(messages: list[dict]) -> list[list[dict]]:
    """Split messages into the turn groups the rolling window evicts as single units.

    Each assistant tool call starts its own group holding its tool results, so long agent
    runs evict old exchanges without orphaning results. Exposed so callers that act on
    evicted turns use the same unit the evictor does.
    """
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
    """Messages present in ``before`` and absent from ``after``, in their original order.

    Identity, not equality: the truncation helpers reuse the same dict objects, and a
    chat can contain two byte-identical turns ("continue" twice) that equality collapses.
    """
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
    """Drop complete oldest turns while preserving system messages and the latest turn.

    ``min_dropped`` keeps evicting past the point where the prompt fits, so a thread can
    re-apply the boundary it already compacted to instead of one that slides every turn.
    """
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
                # Strict chat templates reject adjacent user turns, which an internal
                # tool re-prompt after an evicted exchange would produce.
                kept.append({"role": "assistant", "content": _OMITTED_TOOL_EXCHANGE})
            kept.extend(group)
    return kept, dropped


def messages_without_unpriced_media(messages: list[dict]) -> list[dict]:
    """Return the text/template portion that llama-server can count reliably.

    ``/apply-template`` does not include tokens added later by the multimodal
    processor. Sending base64 there is therefore both expensive and misleading, but
    skipping the rolling fit entirely also sends an already-overlong text history to
    prefill. Count a media-free shallow copy as a lower bound instead. The original
    messages, including every media part, remain the request that is ultimately sent.
    """
    stripped = [_message_without_unpriced_media(message) for message in messages]
    return (
        messages if all(before is after for before, after in zip(messages, stripped)) else stripped
    )


def prompt_budget(context_length: int, max_tokens: Optional[int]) -> int:
    """Tokens available to the PROMPT, once room for the reply is set aside.

    Exported so the recall sizing and the client's over-long-request explanation share
    this formula rather than each keeping a copy that drifts from the actual fit.
    """
    if context_length <= 1:
        return context_length
    requested = max_tokens if max_tokens is not None and max_tokens > 0 else context_length
    return context_length - min(requested, max(1, context_length // 4))


# Cap a tool-loop retrieval without reducing the allowance of smaller results.
_RETRIEVAL_BUDGET_SHARE = 0.5

# The least reply room a rescued prompt must leave, as a fraction of the window. Small on
# purpose: missing the reserve is survivable, so this only rules out the stub-answer end.
_RESCUE_REPLY_FLOOR_DIVISOR = 16


def retrieval_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    reply_returns: bool = False,
) -> int:
    """Return the prompt room available to one retrieval.

    In a tool loop the retrieved exchange and the reply are protected on the next fit, so
    one retrieval may use at most half the prompt budget; single-shot uses the remainder.
    Kept beside ``prompt_budget`` so every backend sizing ``search_conversation`` shares
    one policy rather than quietly disagreeing.
    """
    target = prompt_budget(context_length, max_tokens)
    room = max(0, target - int(prompt_tokens or 0))
    if reply_returns:
        room = min(room, int(target * _RETRIEVAL_BUDGET_SHARE))
    return room


# Headroom against the tokenizer disagreeing with the estimate that sized a result. A
# result is measured in characters and spent in tokens, so the conversion is the thing
# that can be wrong; 1% of the budget absorbs that without noticeably shortening output.
_TOOL_RESULT_BUDGET_BUFFER = 0.99

# What a truncated result costs BESIDES its body: the notice naming the cut, the spill
# path and the command that resumes it. Measured at 60-85 tokens, held clear of that.
# Reserved rather than ignored because the body is sized and the notice appended after,
# so a budget spent entirely on the body overshoots by the notice every time, and at zero
# room, where the notice IS the message, by the whole of it.
#
# Charged by `tools._truncate`, at the point it knows the result really is being cut, and
# NOT subtracted from the room below. A result that fits carries no notice, so holding
# this back up front would cut a result that would have fitted whole and then spend more
# than it saved: 200 tokens of room, a 100-token result, and reserving first leaves 72 for
# the body and appends ~70 tokens of notice explaining the 28 that were dropped.
_RESULT_NOTICE_RESERVE = 128


def tool_result_budget(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    buffer: float = _TOOL_RESULT_BUDGET_BUFFER,
) -> int:
    """Tokens a tool result may add without pushing the next prompt past its budget.

    The cap before this was a share of the WINDOW, so it never fell as the conversation
    filled: one result could take half a small context however little was left, and the
    next fit cannot recover because it protects the newest turn, the very result that does
    not fit. This prices the same result against the room actually remaining.

    Against ``prompt_budget`` rather than ``context_length``: a result sized to fill the
    physical window leaves the prompt fitting and nothing to answer in. Room for the reply
    is not room for the result. The figure covers the whole tool message, notice included:
    see `_RESULT_NOTICE_RESERVE`, which the truncation charges against this room when it
    turns out to need one. A caller with several tool calls to serve divides what comes
    back between them.
    """
    target = prompt_budget(context_length, max_tokens)
    return max(0, int(target * buffer) - int(prompt_tokens or 0))


def turn_is_servable(
    context_length: int,
    max_tokens: Optional[int],
    prompt_tokens: int,
    *,
    buffer: float = _TOOL_RESULT_BUDGET_BUFFER,
) -> bool:
    """Whether the next prompt fits once this tool returns, given an EMPTY result.

    `tool_result_budget` answers how much a result may add and clamps at zero, which the
    truncation reads as "cut hard" -- a number, never a refusal. So a turn whose prompt is
    already over budget before the tool has returned anything is indistinguishable from one
    with a little room left, and the loop runs the tool either way. The side effect lands,
    the result is squeezed to its notice, and the request is rejected regardless.

    Zero room is the refusal the budget could not express. Asked with the notice reserve
    charged, because a result cut to nothing still carries the notice saying so, and at
    this end of the scale that IS the whole message.
    """
    if context_length <= 1:
        return True
    return prompt_tokens + _RESULT_NOTICE_RESERVE + _reply_floor(context_length) <= context_length


def _reply_floor(context_length: int) -> int:
    """The least reply room a turn must leave to be worth running.

    Deliberately NOT `prompt_budget`, which sets aside the whole of `max_tokens`. That is
    the right reserve for sizing a result, and much too strict for deciding whether a call
    may run at all: at a 4096 window it leaves 3072 for the prompt, so the gate refused
    turns of 3,504 and 3,740 tokens that llama-server would have served without complaint,
    and the model sat retrying smaller and smaller edits against a bar it could not see.

    llama-server admits a prompt on size alone (`n_tokens() >= n_ctx`), so that is the line
    this has to draw, minus only enough to answer in. `_RESCUE_REPLY_FLOOR_DIVISOR` already
    encodes that judgement for the compaction rescue -- small on purpose, ruling out the
    stub-answer end rather than promising a full reply.
    """
    return max(1, context_length // _RESCUE_REPLY_FLOOR_DIVISOR)


# How much of a completed call's arguments has to be at stake before replacing them with a
# receipt is worth the edit. Below this the placeholder is a wash against what it removes,
# and the model loses sight of its own last action for nothing.
# Fields naming the call's destination rather than its payload.
_PATH_KEYS = frozenset({"path", "file_path", "filePath"})
# Longest path worth repeating inside every elided leaf's receipt.
_RECEIPT_PATH_MAX_CHARS = 120

_ARG_COMPACTION_FLOOR_CHARS = 1024
# Once the arguments AS A WHOLE are worth reclaiming, the bar each string has to clear
# drops to this. Not zero: a receipt is about 100 characters, so eliding anything shorter
# grows the call it is meant to shrink.
_ARG_COMPACTION_AGGREGATE_LEAF_FLOOR = 256
# When the total of every string reaches this, the call is worth compacting even though no
# single string does. Matches the per-leaf floor: the same amount of window either way.
_ARG_COMPACTION_TOTAL_FLOOR_CHARS = 1024


def _largest_leaf(value: Any) -> int:
    """Longest string anywhere in the parsed arguments, at any depth."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return max((_largest_leaf(item) for item in value.values()), default = 0)
    if isinstance(value, list):
        return max((_largest_leaf(item) for item in value), default = 0)
    return 0


def _total_leaves(value: Any) -> int:
    """Every string in the parsed arguments added together, at any depth."""
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
    """A receipt standing in for a completed call's arguments, or None to leave them.

    The arguments of a call that has ALREADY run are the one part of a tool exchange that
    is pure history: the tool received them in full, the side effect landed, and for the
    file tools the content is on disk and re-readable. Replaying them verbatim buys the
    model nothing it cannot recover, which is what makes them the right thing to spend
    when a turn spills -- unlike the result, which is the only record of what happened.

    Structured rather than dropped, and naming the path, so the model can still see which
    file it just wrote and go and read it. A bare "[omitted]" reads as a failed call and
    is answered with a retry of the same oversized write.
    """
    # Resolved here, not as a default: the constant is defined below this function, and a
    # literal default silently kept the OLD wording on this path while the executed and
    # refused paths moved to the new one -- the same receipt the model misread as tool
    # output, still being emitted by the most common caller.
    phrase = phrase or _completed_phrase_for(name, reply)
    if not isinstance(arguments, str):
        return None
    # The general floor exists so a receipt is not bigger than what it replaces, and for
    # ordinary history compaction a call under it is not worth the churn. A REFUSED call
    # is the opposite case: the refusal message itself is about to be added to a prompt
    # that already does not fit, so any reduction at all is the difference between the
    # user reading the refusal and reading llama-server's context error. The size check
    # at the end still guarantees the receipt never grows the prompt.
    refused = phrase == _REFUSED_PHRASE
    if not refused and len(arguments) < _ARG_COMPACTION_TOTAL_FLOOR_CHARS:
        return None
    try:
        parsed = json.loads(arguments)
    except Exception:
        # Unparseable arguments still cost the window, and a call that has already run
        # cannot be re-issued from them, so the size alone is an honest receipt.
        #
        # Worded from `phrase` like every other receipt. Hardcoding "after the call ran"
        # meant a REFUSED call with 1024+ characters of malformed JSON was replayed as
        # having run, next to the tool message saying nothing was written: two
        # contradictory accounts of the same call, one of which invites the model to
        # reason from a side effect that never happened.
        _unparseable = json.dumps(
            {"_unsloth_compacted": f"{len(arguments)} chars {phrase.format(where = '')}"},
            ensure_ascii = False,
        )
        # Checked here as well as at the end: without the general floor in front of it,
        # a short refused call can have a receipt longer than the arguments it replaces.
        return _unparseable if len(_unparseable) < len(arguments) else None
    if not isinstance(parsed, dict):
        return None
    path = parsed.get("path") or parsed.get("file_path") or parsed.get("filePath")
    elided = 0

    # Per-leaf or aggregate, whichever lets this call be reclaimed. A batched refactor of
    # fifty 800-character edits is forty thousand characters of window with no single
    # string over the floor, so a per-leaf test compacted NOTHING and the call sat in the
    # prompt permanently -- the pre-execution gate then had nothing to reclaim and refused
    # a turn it could have served. The floor exists so a receipt is not bigger than what
    # it replaces, and the final size check below still enforces that.
    # Chosen from the TOTAL, not the largest leaf. Keying on the largest meant one
    # 1100-character edit beside fifty 800-character ones stayed in per-leaf mode and
    # compacted only the first, leaving about 40 KB replayed -- the mixed payload is
    # exactly the shape a batched refactor produces.
    _leaf_floor = (
        # A refused call takes whatever it can get, floored only at the point where a
        # leaf is longer than the receipt describing it, so eliding cannot lose ground.
        _REFUSED_LEAF_FLOOR
        if refused
        else _ARG_COMPACTION_AGGREGATE_LEAF_FLOOR
        if _total_leaves(parsed) >= _ARG_COMPACTION_TOTAL_FLOOR_CHARS
        else _ARG_COMPACTION_FLOOR_CHARS
    )

    def _shrink(value: Any, key: str = "") -> Any:
        """Elide every large string, at whatever depth it sits.

        Depth is not optional: `edit_file` takes an `edits` ARRAY so several changes cost
        one call instead of one each, which puts the file content at
        `edits[i].new_string`. A top-level-only pass sees nothing there and quietly
        compacts nothing -- the batching and the compaction were written a day apart and
        only the tests noticed they had stopped meeting.
        """
        nonlocal elided
        # The destination is never expendable: it is the one field that says WHICH file
        # the call touched, and the receipt promises the content can be found there. A
        # deeply nested path over the aggregate floor was being elided like content,
        # leaving later turns unable to name the file they had just changed.
        if key in _PATH_KEYS:
            return value
        if isinstance(value, str) and len(value) >= _leaf_floor:
            elided += len(value)
            # Named in the receipt only when repeating it is cheaper than the field it
            # points at. Every elided leaf embeds this, so a 500-character path across
            # four leaves costs more than the content removed and the whole compaction
            # is rejected by the size check below. The `path` field is preserved
            # verbatim either way, so nothing is lost by leaving it out here.
            where = (
                f" to {path}"
                if path and key not in _PATH_KEYS and len(str(path)) <= _RECEIPT_PATH_MAX_CHARS
                else ""
            )
            # `old_string` names text the edit REMOVED. The completed phrase says the
            # content is already written and the file on disk holds it, which of the two
            # halves of a replacement is true only of `new_string`; said of the old text
            # it points the model at content the edit has just taken out.
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
    # Never grow the prompt to describe it: a call whose bulk is spread across many small
    # fields leaves nothing to elide and the receipts cost more than the fields did.
    return compacted if len(compacted) < len(arguments) else None


# A leaf shorter than its own receipt costs room to elide, so this is the break-even
# point for the refusal wording rather than a judgement about what is worth compacting.
_REFUSED_LEAF_FLOOR = 110
_REFUSED_PHRASE = (
    "of arguments you sent, elided; this call was refused before it ran and nothing was written"
)
# Worded so the model cannot mistake it for the tool's OUTPUT. The first version read
# "<2581 chars written; re-read the file to see it>" and was quoted straight back in
# the model's reasoning as "the tool result says ... the output was omitted" -- it
# concluded the sandbox had mangled its file and abandoned a working approach. Says
# "you sent" so the owner of the text is unambiguous, and says what it is not.
# The tail once read "read the file back if you need the content", which contradicted
# every other line this PR added: `edit_file` now says not to read back what you just
# wrote, and the starved and repeated-result notices both say reading again will not help.
# Three messages discouraging a re-read and one inviting it is worse than either rule
# alone, and the re-read is the loop that cost eighteen calls in one turn.
_COMPLETED_PHRASE = "of arguments you sent, already written{where}; elided to save room. Not tool output; the file on disk holds it."
# The same receipt for a tool that writes no file. `python`, `terminal`, the search
# tools and every MCP tool are selected by size and by having been answered, exactly
# like `edit_file`, so a 2000-character `code` argument was handed a receipt saying it
# was "already written" and that "the file on disk holds it" -- a file the model can
# then set out to read, or reason from as persisted, when nothing was persisted at all.
# Neither claim about the filesystem, for every case where the reply does not settle it.
# Says only what is certainly true: these are the model's own arguments, the call ran, and
# this is not the tool's output.
_COMPLETED_NEUTRAL_PHRASE = (
    "of arguments you sent, elided to save room; the call already ran. Not tool output"
)
# Tools whose completed call really does leave the content in a file.
_FILE_WRITING_TOOLS = frozenset({"edit_file"})

# A reply that opens like this reports a call that ran and did NOT do what was asked, so
# the file wording would describe a write that never landed.
_FAILED_REPLY_MARKERS = ("error", "failed", "not found", "no such file", "traceback")

# A reply the WINDOW replaced, not one the tool wrote. Under a near-zero result budget
# `_fit_result_to_room` swaps the real answer -- including an `Error: ...` -- for a stub
# saying there was no room for it, and that stub carries none of the markers above. Read
# as proof of a write, it tells the model an edit landed when the edit may have failed,
# under exactly the tight context that makes compaction run in the first place. Absence
# of evidence, so the neutral wording applies.
_INCONCLUSIVE_REPLY_MARKERS = ("no context room left", "chars for the model;")


def _reply_proves_a_write(name: str, content: object) -> bool:
    """Whether this reply settles that the call left its content in a file.

    The tool NAME alone is wrong in both directions, and under the tight context that
    triggers compaction the model acts on the answer. An `edit_file` that ran and returned
    "old_string not found" is not a write, and telling it the content is already on disk
    invites it to skip the retry the error was asking for. A `python` or `terminal` call
    whose code created files is not a non-write either, and telling it nothing was written
    invites it to do the same work twice.

    So the file wording is earned, not assumed: a file tool AND a reply that does not read
    as a failure. Everything else gets a receipt that claims nothing about the disk.
    """
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
    """Replace ONE just-run call's arguments with a receipt, whatever else is protected.

    `compact_completed_tool_arguments` deliberately holds the newest exchange back, which
    is right while a call is still in flight and wrong the instant it returns. Running a
    tool needs no context at all -- only the NEXT prompt does, and by then the arguments
    describe something already on disk.

    That distinction is what lets an oversized call be run instead of refused: refusing
    left the arguments in the transcript anyway, the model retried with a fresh oversized
    call, and each round added a receipt and a refusal message while reclaiming less.
    Measured over three rounds of one thread: 50%, then 34%, then 15% of the conversation
    recovered, ending in a one-character reply. Running the call and compacting it costs
    the same tokens once and leaves the file written.
    """
    # None, not the constant: the receipt is chosen per tool, so a completed `python`
    # call is not told its arguments are on disk. `edit_file` still gets the wording
    # this path was built and proven against.
    return _compact_one_call(messages, call_id, None)


def compact_refused_tool_arguments(messages: list[dict], call_id: str) -> list[dict]:
    """Drop the arguments of one call that was declined, naming it as never sent.

    Refusing a call does not on its own make the turn servable, and on the template that
    made the refusal necessary it actively does not: an assistant turn's `tool_calls` are
    rendered only once a `tool` message answers them, and the refusal IS such a message.
    So declining costs the prompt the very arguments it was declining to afford, and the
    generation that follows is rejected with nothing written -- an accurate refusal the
    user never gets to act on.

    These arguments are the one case with no replay value whatsoever: the call did not
    run, so nothing on disk reflects them and there is nothing to re-read. Their own
    receipt has to say so, because the wording used for a completed call would tell the
    model to go and read a file that was never written.
    """
    return _compact_one_call(messages, call_id, _REFUSED_PHRASE)


def _last_index_with_call(messages: list[dict], call_id: str) -> int:
    """Where the call this result answers actually is.

    Tool-call IDs are NOT unique across a conversation. The textual parsers number from
    `call_0` with an offset that starts at zero on every turn, and the structured
    fallback does the same when the server omits an ID, so a five-round turn holds
    several `call_0`s. Rewriting every match would relabel an earlier successful call as
    refused, or an earlier refused one as already written -- a receipt describing a
    different call's fate, which is worse than not compacting at all.

    Both callers act on the call just decided, so the last match is the right one.
    """
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
    """The last tool reply answering this id, or None when it has not been answered."""
    for message in reversed(messages):
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") == str(call_id):
            return message.get("content")
    return None


def _compact_one_call(
    messages: list[dict],
    call_id: str,
    phrase: Optional[str] = None,
) -> list[dict]:
    """Rewrite exactly one call's arguments to a receipt worded by `phrase`.

    `None` leaves the wording to the tool, which is what a COMPLETED call wants: only
    the file tools may say the content is on disk.
    """
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


# A `role=tool` reply proves an ANSWER, not an execution. The approval gate answers a
# declined call with exactly such a message, and an unreadable call is answered without
# running either. Compacting those to the completed receipt tells the model a file was
# written that the user refused, which it may then report or reason from.
_DID_NOT_RUN_MARKERS = (
    "the user declined to run this tool call",
    "could not be read",
    "nothing ran",
    "nothing was run",
    "nothing was written",
)


def _reply_shows_execution(content: object) -> bool:
    """Whether this tool reply is the result of a call that actually ran."""
    if not isinstance(content, str):
        return True
    head = content[:200].strip().lower()
    return not any(marker in head for marker in _DID_NOT_RUN_MARKERS)


def _executed_call_sites(messages: list[dict]) -> "dict[tuple[int, str], object]":
    """`(message index, call id)` for each call a reply shows actually ran.

    Keyed on the SITE, not the id. Generated ids restart at `call_0` on every turn, so a
    conversation-wide set of "answered" ids says an earlier successful `call_0` vouches
    for a later one. Under a tight context that is how a call the user DECLINED came to be
    replayed with the completed receipt: the reply marker correctly skipped the denial,
    and the older success had already put the id in the set. The model is then told a file
    was written that it refused, which it may report or reason from.

    Paired the way the transcript reads: a reply answers the most recent announcement of
    that id still waiting for one.
    """
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
        # NEWEST pending announcement, not the oldest. Textual parsers number from
        # `call_0` every turn, so an interrupted call that never got a result leaves a
        # stale site under the same id; pairing this reply with THAT one marks the stale
        # arguments executed and leaves the call that actually ran uncompactable.
        site = sites.pop()
        if _reply_shows_execution(message.get("content")):
            executed[(site, str(call_id))] = message.get("content")
    return executed


def compact_completed_tool_arguments(
    messages: list[dict], *, protect_last: int = 0
) -> tuple[list[dict], int]:
    """Replace oversized arguments of already-executed calls with receipts.

    Returns the new message list and the number of calls compacted. The input is never
    mutated: this rewrites what is REPLAYED to the model, exactly as
    `strip_result_for_model` already does for results, while the stored thread and the
    arguments the tool actually received stay byte-identical.

    Only calls with a `role=tool` reply present are touched. A call still awaiting its
    result is the turn in flight, and rewriting its arguments would tell the model it
    wrote something different from what the tool is at that moment being handed.

    Oldest first, so the freshest exchange -- the one the model is mid-way through
    reasoning about -- is the last thing spent. ``protect_last`` holds that many trailing
    messages clear of the pass entirely.
    """
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
    """The advice key for a turn: its role, except a call the model made itself.

    An assistant turn carrying `tool_calls` is reported as `assistant_tool_call` rather
    than `assistant`, because the two have opposite levers. "The reply being continued is
    too long, start a new reply" is the right answer for a resumed generation and the
    wrong one for a turn whose bulk is an 8 KB file the model passed to `edit_file`:
    starting a new reply re-runs the same write. Kept as a role key so the split costs
    the advice table one row and `_blame_latest_turn` nothing.
    """
    role = str(message.get("role") or "")
    if role == "assistant" and message.get("tool_calls"):
        # Split again by whether a FILE is involved. The file wording reached an
        # oversized `python`, `terminal`, web or MCP call and told the user to ask for a
        # smaller file when the payload was a program, a command or a query -- naming the
        # wrong cause and offering an action that cannot shrink it.
        # From the call that accounts for the turn's SIZE, not from whichever call
        # happens to be a file tool. A parallel batch holding a small `edit_file` beside
        # an oversized `python` or MCP payload was diagnosed as a file that is too large,
        # so the advice was to ask for a smaller file -- an action that cannot shrink the
        # payload that actually caused the refusal.
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
    """Who to blame for the newest turn, which is not always the newest MESSAGE.

    On the strict templates an assistant `tool_calls` block renders only once a `tool`
    message answers it, so the marginal cost of that tool message is the reply PLUS the
    call's arguments, which were invisible until now. Classifying the reply alone reports
    an overflow caused by a large `python`, MCP or `edit_file` payload as a large tool
    result, and the advice -- ask for a smaller slice of the file or page -- cannot shrink
    the thing that actually overflowed.
    """
    if not messages:
        return ""
    latest = messages[-1]
    if str(latest.get("role") or "") != "tool":
        return _blamed_role(latest)
    call_id = latest.get("tool_call_id")
    for message in reversed(messages[:-1]):
        role = str(message.get("role") or "")
        if role != "assistant":
            # Only the call immediately preceding this reply is paired with it; anything
            # else between them means this reply does not belong to a call at all.
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
        # Only when the CALL is the bigger half. A dominant tool result still gets the
        # tool advice, which is right and is what the existing cases assert: the reply is
        # the thing the user can ask for less of. Blaming the call unconditionally
        # reversed that and told them to shrink a payload that was not the problem.
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
    """`(tokens, exact)` for the newest message, estimated if the template refuses it.

    A tool loop can reach the does-not-fit diagnosis with a tool result last, which strict
    templates reject on its own; letting that raise would abort the fit and tell the user
    nothing. An approximate number beats a diagnosis that never arrives -- but the caller
    has to know which it got, because only the counted one carries the prompt's floor.
    """
    if not messages:
        return 0, False
    try:
        return int(count_tokens(messages[-1:])), True
    except Exception:
        return int(estimate_messages_tokens(messages[-1:])), False


def _shared_prompt_tokens(count_tokens: Callable[[list[dict]], int]) -> int:
    """What a rendered prompt costs before any message at all.

    The counter prices a whole PROMPT, not a bag of messages: the template's own wrapper
    and, on a request that advertises tools, the entire tool catalogue rendered into the
    system turn. So that constant sits inside every count the fit takes, including the
    one-message slice above -- on the GGUF tool loop the counters pass `safe_tools` to
    `count_chat_tokens`, which puts the schemas in the prompt it measures, and a couple
    of MCP servers is thousands of tokens of them.

    Measured on the empty prompt rather than estimated: this number is subtracted from
    the two counts that decide which part of the prompt gets named, and a subtraction
    that is only roughly right moves the blame instead of removing it. One extra count,
    taken only on the branch that has already decided to refuse the request.
    """
    try:
        return max(0, int(count_tokens([])))
    except Exception:
        # No measurable floor; zero leaves the diagnosis as it was before this existed.
        return 0


def _marginal_turn_count(
    fitted: list[dict], count_tokens: Callable[[list[dict]], int], irreducible_tokens: int
) -> Optional[int]:
    """What the newest turn ADDED to the prompt that was actually measured.

    The one-message slice is unusable when the template renders the newest message as
    nothing on its own, but the difference against the same prompt WITHOUT it is a real
    tokenizer count of exactly the turn's contribution, with the floor cancelled on both
    sides of the subtraction. So the diagnosis keeps a counted turn where it used to fall
    back to a guess.

    `irreducible_tokens` must be the count of `fitted` itself, which is what both callers
    pass. One extra count, on a branch that has already decided to refuse the request,
    next to the one `_shared_prompt_tokens` takes there. None when the counter cannot
    price the prefix, which is the caller's cue to fall back to the estimate.
    """
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
    """The fields that say WHICH part of a refused prompt is which.

    `shared_prompt_tokens` is the floor both `latest_turn_tokens` and
    `irreducible_tokens` carry, so a consumer comparing the two can take it off both
    sides and compare what the turn actually contributed against what the rest of the
    conversation did. Zero when the turn was estimated rather than counted: that estimate
    prices the message's own JSON and no catalogue, so it has no floor to remove.

    `latest_turn_exact` says which of the two `latest_turn_tokens` is. It is False only
    on the fallback below, where nothing could be counted at all, and a consumer must
    then not compare it against `irreducible_tokens`: that number is a tokenizer count of
    the rendered prompt while the estimate is four characters to a token over the
    message's JSON, and the two do not share units. Measured on the bundled gemma-4
    template with a real Gemma tokenizer, 16,400 characters of newlines estimate 8,207
    tokens against 557 rendered, 14.8x, which is enough for the estimate alone to clear
    the dominance ratio against a prompt the turn is 6% of.

    `fitted` is the message list `irreducible_tokens` was counted over. Given it, a turn
    the template refuses to render alone is still priced by difference rather than
    guessed.
    """
    if not messages:
        # Vacuously exact: nothing was estimated, and a zero count is ignored anyway.
        return {
            "latest_turn_tokens": 0,
            "latest_turn_role": "",
            "shared_prompt_tokens": 0,
            "latest_turn_exact": True,
        }
    latest, exact = _latest_turn_count(messages, count_tokens)
    shared = _shared_prompt_tokens(count_tokens) if exact else 0
    if exact and latest <= shared:
        # Counted, and yet no bigger than the empty prompt: the template rendered the turn
        # as nothing. Both bundled Gemma-4 templates do exactly that to a `role: tool`
        # message on its own -- `{%- if message['role'] != 'tool' -%}` skips it, and the
        # result is emitted only while scanning forward from the assistant tool call that
        # asked for it, which a one-message slice does not contain. So the number is the
        # floor and nothing else, and subtracting the floor from it would leave the turn
        # contributing ~0 and blame the conversation.
        #
        # Price it by DIFFERENCE against the prompt that was measured instead: the slice
        # is unrenderable, the contribution is not. Reported floor-inclusive, so the
        # consumer's existing subtraction of `shared` leaves exactly the marginal and the
        # ratio still compares two tokenizer counts.
        marginal = _marginal_turn_count(
            fitted if fitted is not None else messages, count_tokens, irreducible_tokens
        )
        if marginal is not None and marginal > 0:
            latest = marginal + shared
        else:
            # Nothing countable left: same remedy as a template that refuses the slice
            # outright -- price the message's own JSON, and record no floor, because that
            # estimate does not carry one.
            latest = int(estimate_messages_tokens(messages[-1:]))
            shared = 0
            # What is REPORTED is now the estimate, and that is what the flag describes.
            # Leaving it True would be the worse half of the bug.
            exact = False
    # Never all of either side: a floor at or above them would leave no ratio to compare.
    shared = max(0, min(shared, latest - 1, int(irreducible_tokens) - 1))
    return {
        "latest_turn_tokens": latest,
        # Whose message it is: often a tool result the user cannot shorten.
        "latest_turn_role": _blamed_role_for_turn(messages),
        "shared_prompt_tokens": shared,
        # Whether the number above is a token count or a four-characters-a-token guess.
        "latest_turn_exact": bool(exact),
    }


def clamp_compaction_headroom_ratio(value: Any) -> Optional[float]:
    """Return a usable extra-trim ratio, or None when the caller left it unset.

    ``ROLLING_COMPACTION_HEADROOM_RATIO`` already clamps the process default to
    ``[0, 0.9]``. Per-request overrides go through the same gate so a UI slider
    cannot ask the fitter to drop the entire prompt or to grow it.
    """
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
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """Fit a chat into its real context by dropping oldest complete turns.

    The exact tokenizer/template count decides whether trimming is needed; the cheap
    estimator only picks candidate turns. The current turn is never clipped, so an
    irreducibly large request still reaches llama-server's context-length error.

    ``reserve_tokens`` leaves room for what the caller adds back after fitting (recalled
    turns). It deliberately does not affect whether to trim at all, so a chat that fits
    today is never evicted just because the reserve would not fit alongside it.

    ``sticky_dropped`` is the boundary this thread last compacted to, in messages,
    re-applied before anything else and moved only if what is left still does not fit.
    Without it the fit is stateless (the client re-sends the whole transcript each turn)
    so the boundary slides every reply; with it plus ``_COMPACTION_HEADROOM_RATIO`` of
    slack, compaction is an occasional event the prefix cache can survive.

    Acceptance is still checked against the untightened ``prompt_target``: falling short
    of the headroom is not a failure to fit.
    """
    if context_length <= 1:
        return messages, None

    prompt_target = prompt_budget(context_length, max_tokens)
    fitted = list(messages)
    initial_tokens = count_tokens(fitted)
    current_tokens = initial_tokens
    dropped_total = 0

    # Phase one: put the boundary back where this thread already had it, so a compacted
    # thread stops compacting further every turn. Gated on the prompt not already
    # fitting: a saved boundary describes the branch it was measured on, and after a
    # rollback it would evict most of a chat that comfortably fits.
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

    # Phase two, only if what is left still does not fit: move the boundary, taking a
    # chunk out rather than skimming to the brim so it can stay put for a while.
    trim_target = prompt_target
    # The 5% floor on each eviction bite, given up only by a caller that asked for no
    # extra trim. Keyed on the ratio and not on `headroom`, which `keeps_boundary = False`
    # zeroes for every threadless and incognito request, none of which chose anything.
    min_bite = True
    if current_tokens > prompt_target:
        # Summed, not max()'d: the reserve is spent immediately on recalled passages, so
        # counting it as headroom would hand back room that is already taken.
        #
        # And only for a caller that can put the boundary back next request. The headroom
        # buys quiet turns between compactions by cutting deeper than needed, which is a
        # bargain only if the deeper cut is remembered. An incognito chat, an API request
        # with no persisted thread, or a request whose turns are not saved gets neither
        # the boundary nor a recall of what went, so there it is simply 25% less history
        # than plain eviction would have kept.
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
        # Missing the prompt target only loses reserved reply room, so keep the original
        # if it fits the physical window, else an eviction that does. Strictly under on
        # both sides: llama-server refuses at `n_ctx` exactly. But not under by one token,
        # which would lose the history AND answer in one token, worse than the refusal it
        # replaces. Capped by the reserve, so the floor demands no room the target did not.
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
            # Floor for the conversation, and how much of it is the message just sent:
            # together they say whether the chat or the single message is the problem.
            "irreducible_tokens": current_tokens,
            # `fitted` is what `current_tokens` prices, so the turn can be counted by
            # difference against it rather than estimated.
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
