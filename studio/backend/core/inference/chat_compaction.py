# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rolling-context compaction, archive and recall for local chat.

Shared by the GGUF backend, the safetensors orchestrator and the passthrough routes: fitting a
conversation into the context window, the sticky compaction boundary, and archiving evicted turns
for recall.
"""

import bisect
from typing import Callable, NamedTuple, Optional

from core.inference.context_window import (
    _COMPACTION_HEADROOM_RATIO,
    clamp_compaction_headroom_ratio,
    evicted_messages,
    fit_oversized_attachments,
    fit_rolling_context,
    prompt_budget,
)
from loggers import get_logger

logger = get_logger(__name__)


def _memory_tool_withheld(thread_id, tools) -> bool:
    """Gated on an existing archive: before the first compaction, the tool's absence says nothing."""
    if not thread_id:
        return False
    try:
        from core.rag import conversation_archive
        if not conversation_archive.has_archive(thread_id):
            return False
    except Exception:  # noqa: BLE001 -- unknown archive state is "do not refuse"
        return False
    names = set()
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = (function or {}).get("name") if isinstance(function, dict) else None
        if name:
            names.add(name)
    return "search_conversation" not in names


def _can_reset_epoch(
    thread_id,
    supports_tools: bool,
    *,
    tools_withheld: bool = False,
) -> bool:
    """Refuses unless the turns are archived and the model can receive search_conversation."""
    if not thread_id or not supports_tools:
        return False
    if tools_withheld:
        # The request withdrew the tool loop; a reset would strand the epoch behind a missing tool.
        return False
    try:
        from state.tool_policy import conversation_recall_allowed

        # Without recall a reset strands the epoch behind a missing tool; rolling needs no tool.
        if not conversation_recall_allowed():
            return False
    except Exception:  # noqa: BLE001 -- an unreadable policy is "no", never an error
        return False
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.enabled() and conversation_archive.can_archive(thread_id))
    except Exception:  # noqa: BLE001 -- an unavailable archive is a "no", never an error
        return False


def _archive_is_degraded() -> bool:
    """Probes reachable() now; degraded() is the last write's verdict, and this write has not run yet."""
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.degraded()) or not conversation_archive.reachable()
    except Exception:  # noqa: BLE001 -- an unreadable flag is "healthy", never an error
        return False


def _archive_write_failed() -> bool:
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.degraded())
    except Exception:  # noqa: BLE001 -- an unreadable flag is "healthy", never an error
        return False


def _compaction_fit_kwargs(
    context_policy: Optional[str] = None, compaction_headroom_ratio: Optional[float] = None
) -> dict:
    """Unset fields fall back to UNSLOTH_CONTEXT_POLICY and ROLLING_COMPACTION_HEADROOM_RATIO, so older clients see no change."""
    extra: dict = {}
    if context_policy in ("checkpoint", "rolling"):
        extra["context_policy"] = context_policy
    if compaction_headroom_ratio is not None:
        extra["headroom_ratio"] = compaction_headroom_ratio
    return extra


def _request_uses_checkpoint(requested_policy: Optional[str] = None) -> bool:
    """Process default, unless this request named checkpoint or rolling."""
    if requested_policy == "checkpoint":
        return True
    if requested_policy == "rolling":
        return False
    from core.inference import checkpoint

    return checkpoint.enabled()


def _fit_context(messages, **kwargs):
    """Requests that may not reset silently keep rolling; UNSLOTH_CONTEXT_POLICY=rolling is the escape."""
    can_reset = bool(kwargs.pop("can_reset", False))
    # Whether sticky_dropped is a checkpoint RESET depth; defaults True for callers with no thread state.
    sticky_is_checkpoint = bool(kwargs.pop("sticky_is_checkpoint", True))
    # Naming a tool the request lacks makes the model print the call as text.
    recall_offered = bool(kwargs.pop("recall_offered", True))
    requested_policy = kwargs.pop("context_policy", None)
    if requested_policy not in ("checkpoint", "rolling"):
        requested_policy = None
    # /compact: only a checkpoint reset can honour it, since only its turns are archived.
    force_reset = bool(kwargs.pop("force_reset", False))
    sticky_manual = bool(kwargs.pop("sticky_manual", False))
    replay_only = bool(kwargs.pop("replay_only", False))
    if replay_only and int(kwargs.get("sticky_dropped") or 0) <= 0:
        return messages, None
    if kwargs.get("count_tokens") is not None:
        try:
            messages = fit_oversized_attachments(
                messages,
                context_length = kwargs.get("context_length") or 0,
                max_tokens = kwargs.get("max_tokens"),
                count_tokens = kwargs["count_tokens"],
            )
        except Exception:  # noqa: BLE001 -- the fit below still refuses what cannot fit
            logger.debug("attachment fit failed", exc_info = True)
    if replay_only:
        if sticky_is_checkpoint:
            from core.inference import checkpoint

            # Generation's wording: unsearchable without a reset, else the last write's verdict
            # (not the live probe, which runs an embedding forward per count).
            return checkpoint.fit_checkpoint_context(
                messages,
                can_reset = False,
                searchable = (lambda: recall_offered and not _archive_write_failed())
                if can_reset
                else False,
                sticky_manual = sticky_manual,
                replay_only = True,
                **kwargs,
            )
        return fit_rolling_context(messages, replay_only = True, **kwargs)
    try:
        from core.inference import checkpoint

        use_checkpoint = _request_uses_checkpoint(requested_policy)
        if (
            not can_reset
            and use_checkpoint
            and sticky_is_checkpoint
            and int(kwargs.get("sticky_dropped") or 0) > 0
        ):
            # No reset allowed: replay the epoch and keep the block (rolling would drop the instruction).
            # Falls through to rolling below if the replay cannot fit.
            fitted, truncation = checkpoint.fit_checkpoint_context(
                messages,
                can_reset = False,
                searchable = False,
                sticky_manual = sticky_manual,
                **kwargs,
            )
            if truncation is not None and truncation.get("fits"):
                return fitted, truncation
            # Rolling cannot rebuild a checkpoint boundary, so drop it and recompute; skipped when sticky_is_checkpoint is false, which keeps the count.
            kwargs.pop("sticky_dropped", None)
        if can_reset and use_checkpoint:
            # A degraded archive downgrades reset to replay and refuses only new epochs; searchable must be cleared so unindexed turns are not claimed.
            _degraded: dict = {}

            def _is_degraded() -> bool:
                if "value" not in _degraded:
                    _degraded["value"] = _archive_is_degraded()
                return bool(_degraded["value"])

            fitted, truncation = checkpoint.fit_checkpoint_context(
                messages,
                can_reset = lambda: not _is_degraded(),
                searchable = lambda: recall_offered and not _is_degraded(),
                force_reset = force_reset,
                sticky_manual = sticky_manual,
                **kwargs,
            )
            # can_reset=False has no phase two; a refusal falls through to rolling.
            if truncation is None or truncation.get("fits"):
                return fitted, truncation
    except Exception:  # noqa: BLE001 -- a policy failure must never break a chat
        logger.warning("Checkpoint fit failed; falling back to the rolling window", exc_info = True)
    return _fit_rolling_remembered(messages, kwargs)


def _fit_rolling_remembered(messages, kwargs):
    """`fit_rolling_context`, remembering the cut in memory when the caller has no thread to save it on.

    A one-off request cuts as before; a returning one replays its cut and takes the usual headroom bite.
    """
    if kwargs.get("keeps_boundary") or int(kwargs.get("sticky_dropped") or 0) > 0:
        return fit_rolling_context(messages, **kwargs)
    from core.inference import prefix_boundary

    scope = ("rolling", kwargs.get("context_length"))
    remembered = prefix_boundary.recall(messages, scope)
    fitted, truncation = fit_rolling_context(
        messages,
        **{**kwargs, "keeps_boundary": remembered > 0, "sticky_dropped": remembered},
    )
    if truncation and truncation.get("fits") and int(truncation.get("dropped_messages") or 0) > 0:
        prefix_boundary.remember(messages, fitted, int(truncation["dropped_messages"]), scope)
    return fitted, truncation


def _fit_with_instruction_pins(
    messages,
    *,
    anchor_ids = None,
    **kwargs,
):
    """Pins are recomputed each fit (inline recall swaps the user dict), dropped if they block a fit, and must never mutate the anchor set the caller passed."""
    anchors = set(anchor_ids or ())
    try:
        from core.inference import instruction_pin
        pins = instruction_pin.pinned_instruction_ids(
            messages,
            prompt_target = prompt_budget(
                kwargs.get("context_length") or 0, kwargs.get("max_tokens") or 0
            ),
        )
    except Exception:  # noqa: BLE001 -- a protection heuristic must never break a chat
        pins = set()
    fitted, truncation = _fit_context(
        messages, protected_message_ids = (anchors | pins) or None, **kwargs
    )
    # Only a refusal returning the ORIGINAL messages was already failing; a rescue is servable.
    if (
        not kwargs.get("replay_only")
        and pins
        and truncation
        and not truncation.get("fits")
        and not int(truncation.get("dropped_messages") or 0)
    ):
        return _fit_context(messages, protected_message_ids = anchors or None, **kwargs)
    return fitted, truncation


def _conversation_recall_reserve(thread_id: Optional[str]) -> int:
    """Room to hold back during a fit for the turns recalled straight after it."""
    if not thread_id:
        return 0
    try:
        from core.rag import config as rag_config
        from core.rag import conversation_archive

        # Temporary chats are never archived, so do not reserve room for recall.
        if not conversation_archive.can_archive(thread_id):
            return 0
        # sqlite-vec may exist while the embedder fails; reserving then only loses history.
        if conversation_archive.degraded():
            return 0
        return max(0, int(rag_config.CONVERSATION_RECALL_RESERVE_TOKENS))
    except Exception:
        return 0


def _keeps_compaction_boundary(thread_id: Optional[str]) -> bool:
    """Only a persisting thread can restore a boundary: it rides on saved assistant metadata."""
    if not thread_id:
        return False
    try:
        from storage import studio_db
        return bool(studio_db.chat_thread_has_messages(str(thread_id)))
    except Exception:
        return False


def _archive_branch_transcript(
    branch_messages: Optional[list[dict]], roles: Optional[tuple[str, ...]] = None
) -> Optional[list[str]]:
    """The active branch, one normalised string per message, or None if there is nothing."""
    if not branch_messages:
        return None
    try:
        from core.rag import conversation_archive
        return conversation_archive.branch_message_texts(branch_messages, roles)
    except Exception:
        return None


def _branch_boundary(conversation: list[dict], branch: Optional[list[dict]]) -> int:
    """Count every evicted message by identity, not dropped_messages (tool refits inflate it); stopping at the first survivor un-compacts the epoch."""
    messages = list(branch or ())
    cutoff = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default = max(0, len(messages) - 1),
    )
    messages = messages[:cutoff]
    live = {id(message) for message in conversation}
    count = 0
    for message in messages:
        if message.get("role") in ("system", "developer"):
            continue
        if id(message) not in live:
            count += 1
    return count


def _records_boundary(truncation: dict) -> bool:
    """Not the fits flag: a rescued fit still evicts, and skipping the save compacts silently. Saving is not replaying: sticky logic declines fits-false records."""
    return bool(truncation.get("fits")) or bool(truncation.get("dropped_messages"))


def _effective_headroom_ratio(compaction_headroom_ratio: Optional[float] = None) -> float:
    """The extra-trim share this request's rolling fit actually used."""
    ratio = clamp_compaction_headroom_ratio(compaction_headroom_ratio)
    return _COMPACTION_HEADROOM_RATIO if ratio is None else ratio


def _boundary_metadata(
    fitted: list[dict],
    before: Optional[list[dict]],
    compaction_headroom_ratio: Optional[float] = None,
) -> dict:
    """Saved with the depth: a replayed depth is wrong if the headroom setting has since moved."""
    return {
        "boundary_messages": _branch_boundary(fitted, before),
        "boundary_anchor": _branch_boundary_anchor(fitted, before),
        "boundary_headroom_ratio": _effective_headroom_ratio(compaction_headroom_ratio),
    }


def _branch_non_system(branch: Optional[list[dict]]) -> list[dict]:
    """The branch as ``_branch_boundary`` counts it: system and developer turns removed."""
    return [
        message for message in (branch or ()) if message.get("role") not in ("system", "developer")
    ]


def _branch_boundary_anchor(conversation: list[dict], branch: Optional[list[dict]]) -> str:
    """Text, not id or index, taken after the last eviction (first survivor may be a pin). Only moves the boundary shallower; a stale anchor never evicts live turns."""
    messages = _branch_non_system(branch)
    live = {id(message) for message in conversation}
    last_dropped = max(
        (index for index, message in enumerate(messages) if id(message) not in live),
        default = -1,
    )
    for message in messages[last_dropped + 1 :]:
        if id(message) in live:
            return _anchor_text(message)
    return ""


# Bounded so a long tool argument cannot bloat per-turn UI metadata; the name and args head suffice.
_ANCHOR_ARGS_CHARS = 200


# Anchor rides in every event and persisted turn, so keep only a head; sharing a head costs an extra compaction, never a live turn.
_ANCHOR_TEXT_CHARS = 200


def _anchor_text(message: dict) -> str:
    """Tool-call messages have empty content, so fall back to tool_calls; otherwise the rebase silently fails and evicts a live message per deleted one."""
    text = _archive_message_text(message.get("content"))
    if text:
        return text[:_ANCHOR_TEXT_CHARS]
    parts = []
    for call in message.get("tool_calls") or ():
        function = (call or {}).get("function") or {}
        name = str(function.get("name") or "tool")
        arguments = str(function.get("arguments") or "").strip()[:_ANCHOR_ARGS_CHARS]
        parts.append(f"{name}: {arguments}" if arguments else name)
    return "\n".join(parts)[:_ANCHOR_TEXT_CHARS]


def _archive_message_text(content) -> str:
    """One stored message flattened the way the branch check flattens it, or ""."""
    try:
        from core.rag import conversation_archive
        return conversation_archive.message_text(content)
    except Exception:
        return ""


def _archive_content_on_branch(content, transcript: Optional[list[str]]) -> bool:
    if transcript is None:
        return True
    try:
        from core.rag import conversation_archive
        return conversation_archive.content_on_branch(content, transcript)
    except Exception:
        return True


def _row_replayed(message: dict) -> bool:
    """Status is not the test: a stopped reply with text, or reasoning alone, is kept."""
    if message.get("role") != "assistant":
        return True
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    reasoning = False
    for part in content or ():
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and str(part.get("text") or "").strip():
            return True
        if part.get("type") in ("image", "image-url", "file"):
            return True
        reasoning = reasoning or part.get("type") == "reasoning"
    if reasoning and not _row_ended_early(message):
        return True
    return bool(message.get("attachments"))


def _row_ended_early(message: dict) -> bool:
    """`assistantTurnEndedEarly`: the persisted marker, since status is session state."""
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    custom = metadata.get("custom")
    custom = custom if isinstance(custom, dict) else {}
    return bool(metadata.get("incomplete") or custom.get("incomplete"))


def _archive_branch_chain(
    messages: list[dict], branch_messages: Optional[list[dict]]
) -> Optional[list[dict]]:
    """The stored parent chain selected by this wire branch, when ancestry can prove one."""
    if not branch_messages:
        return None
    try:
        from core.rag import conversation_archive

        def _chain(probe):
            return conversation_archive._active_chain(
                messages,
                probe,
                fallback = False,
                require_unique = True,
            )

        chain = _chain(branch_messages)
        if not chain:
            return None
        # The unstored newest turn only matches on a SIBLING, so settled turns pick the branch.
        settled = list(branch_messages)
        while settled and settled[-1].get("role") == "user":
            settled.pop()
        if settled:
            proof = _chain(settled)
            if not proof:
                return None
            tip = proof[-1].get("id")
            if not any(row.get("id") == tip for row in chain):
                return None

            # Rows past the tip match only the text of unstored turns, keyed by role; matching the whole request let an abandoned row in on repeated text.
            def _keys(rows):
                return [
                    (wire.get("role"), _archive_message_text(wire.get("content")))
                    for wire in conversation_archive._as_wire(list(rows))
                ]

            carried = set(_keys(branch_messages[len(settled) :]))
            past_tip = False
            for row in chain:
                if past_tip:
                    checked = False
                    for key in _keys([row]):
                        if not key[1]:
                            continue
                        if key not in carried:
                            return None
                        checked = True
                    # Only a reply the client drops whole may ride unchecked; refuse re-sent ones.
                    if not checked and _row_replayed(row):
                        return None
                past_tip = past_tip or row.get("id") == tip
        return chain
    except Exception:
        return None


def _archive_as_wire(messages: Optional[list[dict]]) -> list[dict]:
    """Not sanitised: the boundary anchor is written from the raw request, so lookups must match it."""
    try:
        from core.rag import conversation_archive
        return conversation_archive._as_wire(list(messages or ()), sanitise_assistant = False)
    except Exception:
        return list(messages or ())


class _CompactionBranchState(NamedTuple):
    message: dict
    truncation: Optional[dict]
    recorded: int


def _reply_text(content) -> str:
    """A stored reply's visible text, which a Max Tokens continuation re-sends verbatim."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _row_truncation(message: dict) -> Optional[dict]:
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    custom = metadata.get("custom")
    custom = custom if isinstance(custom, dict) else {}
    truncation = metadata.get("contextTruncation") or custom.get("contextTruncation")
    return truncation if isinstance(truncation, dict) else None


# The auto-continue limit is 3 rounds; a manual Continue chain longer than this keeps its refusal.
_RESUME_HOPS = 8


_RESUME_SIBLINGS = 16


def _max_tokens_cut(message: dict) -> bool:
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    custom = metadata.get("custom")
    custom = custom if isinstance(custom, dict) else {}
    incomplete = metadata.get("incomplete") or custom.get("incomplete")
    reason = incomplete.get("reason") if isinstance(incomplete, dict) else incomplete
    return reason == "length"


def _row_key(row: dict):
    return row.get("id") or id(row)


def _resumable_cuts(stored: list[dict]) -> dict:
    """Max Tokens cuts per parent: ([positions], [rows]) in creation order."""
    cuts: dict = {}
    for position, row in enumerate(stored):
        if row.get("role") == "assistant" and _max_tokens_cut(row):
            at, rows = cuts.setdefault(row.get("parentId", row.get("parent_id")), ([], []))
            at.append(position)
            rows.append(row)
    return cuts


def _resumed_reply(message: dict, cuts: dict, positions: dict) -> dict:
    """The reply an unfitted Max Tokens continuation resumed: its refusal records no epoch,
    but the epoch in force is still the resumed reply's."""
    for _ in range(_RESUME_HOPS):
        truncation = _row_truncation(message)
        if (
            truncation is None
            or truncation.get("fits")
            or truncation.get("latest_turn_role") != "assistant"
        ):
            return message
        position = positions.get(_row_key(message))
        if position is None:
            return message
        text = _reply_text(message.get("content"))
        at, rows = cuts.get(message.get("parentId", message.get("parent_id")), ((), ()))
        end = bisect.bisect_left(at, position)
        earlier = rows[max(0, end - _RESUME_SIBLINGS) : end]
        # Newest earlier cut wins; equal text counts, since a refusal adds nothing.
        source = next(
            (
                row
                for row in reversed(earlier)
                if (resumed := _reply_text(row.get("content")).rstrip())
                and text.startswith(resumed)
            ),
            None,
        )
        if source is None:
            return message
        message = source
    return message


def _compaction_branch_states(
    stored: list[dict], branch_messages: Optional[list[dict]] = None
) -> list[_CompactionBranchState]:
    """Several rows only for textually identical Retry siblings; consumers take the conservative one."""
    # Assistant rows only: an abandoned "Done" rides in on a live "not done yet".
    branch = _archive_branch_transcript(branch_messages, ("assistant",))
    if branch_messages and not branch:
        return []

    chain = _archive_branch_chain(stored, branch_messages)
    candidates = [
        message
        for message in reversed(chain if chain is not None else stored)
        if message.get("role") == "assistant"
        and (chain is not None or _archive_content_on_branch(message.get("content"), branch))
    ]
    if not candidates:
        return []

    # Exact first: the branch check is a substring test, so "Done" matches "Not done yet".
    if chain is None:
        live = set(branch or ())
        exact = [
            message
            for message in candidates
            if _archive_message_text(message.get("content")) in live
        ]
        if exact:
            candidates = exact

    # A COMPLETED row with no truncation ends the epoch; only active/aborted are placeholders.
    cuts = _resumable_cuts(stored)
    positions = {_row_key(row): position for position, row in enumerate(stored)}
    states = []
    for message in candidates:
        metadata = _resumed_reply(message, cuts, positions).get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        custom = metadata.get("custom")
        custom = custom if isinstance(custom, dict) else {}
        truncation = metadata.get("contextTruncation") or custom.get("contextTruncation")

        recorded = None
        if isinstance(truncation, dict):
            raw_recorded = truncation.get("boundary_messages")
            if raw_recorded is None:
                raw_recorded = truncation.get("dropped_messages")
            if raw_recorded is not None:
                try:
                    recorded = max(0, int(raw_recorded or 0))
                except (TypeError, ValueError):
                    pass
        if recorded is None and chain is not None:
            # Skipping defers to the previous epoch, so it needs a proved branch; on the text path the rows are twins. A research row reports its run, not whether this fit.
            research_row = (
                (metadata.get("serverManaged") or custom.get("serverManaged"))
                and (metadata.get("researchRunId") or custom.get("researchRunId"))
                and (metadata.get("researchStatus") or custom.get("researchStatus"))
            )
            if research_row:
                continue
            status = metadata.get("generationStatus") or custom.get("generationStatus")
            incomplete = metadata.get("incomplete") or custom.get("incomplete")
            reason = incomplete.get("reason") if isinstance(incomplete, dict) else incomplete
            active = status in {"queued", "running", "cancelling"}
            aborted = status in {"cancelled", "failed"} or reason in {"cancelled", "interrupted"}
            if active or (status != "completed" and aborted):
                continue
        if recorded is None:
            states.append(_CompactionBranchState(message, None, 0))
            continue
        states.append(_CompactionBranchState(message, truncation, recorded))

    if not states:
        return []
    if chain is not None:
        # Parent ancestry orders repeated text on one linear history unambiguously.
        return states[:1]
    newest = _archive_message_text(states[0].message.get("content"))
    return [
        state for state in states if _archive_message_text(state.message.get("content")) == newest
    ]


def _sticky_compaction_state(
    thread_id: Optional[str],
    branch_messages: Optional[list[dict]] = None,
    context_policy: Optional[str] = None,
    can_reset: bool = True,
    compaction_headroom_ratio: Optional[float] = None,
) -> tuple[int, bool]:
    """`_sticky_compaction_details` without the manual flag."""
    return _sticky_compaction_details(
        thread_id,
        branch_messages,
        context_policy = context_policy,
        can_reset = can_reset,
        compaction_headroom_ratio = compaction_headroom_ratio,
    )[:2]


def _sticky_compaction_details(
    thread_id: Optional[str],
    branch_messages: Optional[list[dict]] = None,
    context_policy: Optional[str] = None,
    can_reset: bool = True,
    compaction_headroom_ratio: Optional[float] = None,
) -> tuple[int, bool, bool]:
    """How many leading messages this thread last compacted away, who compacted them, and
    whether the user asked for it (/compact), which replays even while the prompt fits.

    The provenance decides which fitter may consume the count. A rolling boundary handed
    to `fit_checkpoint_context` reads as an epoch already in force, so the fit returns
    `checkpoint_started` false and `_archive_and_recall` skips the recall the rolling
    fallback would have run. Mixed provenance across surviving rows counts as rolling:
    `min(boundaries)` below may pick either row, so the safe reading is the one that still
    recalls.

    The fit is otherwise stateless: the client re-sends the whole transcript each request,
    so "keep the newest N tokens" slides forward every turn, which tells the user nothing
    useful and throws away llama-server's prefix cache each time the head moves. Reading
    the boundary back makes compaction an occasional event.

    Read from the thread's newest assistant turn, which already persists this number for
    the UI, so nothing new is stored and it survives a restart. Never raises: no history,
    a non-persisting API client, or a storage error all mean "no boundary".
    """
    if not thread_id:
        return 0, False, False
    requested_ratio = _effective_headroom_ratio(compaction_headroom_ratio)
    try:
        from storage import studio_db

        stored = list(studio_db.list_chat_messages(thread_id) or [])
        states = _compaction_branch_states(stored, branch_messages)
        if not states:
            return 0, False, False

        boundaries = []
        origins = []
        manual = []
        branch_wire = _archive_as_wire(branch_messages)
        for state in states:
            truncation = state.truncation
            recorded = state.recorded
            if truncation is None:
                return 0, False, False
            # Only a SUCCEEDED fit describes a boundary, but an explicit failure still rules.
            if not truncation.get("fits"):
                return 0, False, False
            # A provider-generated summary boundary is display metadata, not a local-fit epoch. On a provider/model
            # switch the opaque summary is deliberately withheld, so replaying its boundary here would discard the
            # original turns without retaining their summary.
            if truncation.get("summarized") is True:
                return 0, False, False
            # A boundary is valid only under the fit that will consume it, and the two
            # directions fail differently.
            #
            # A checkpoint boundary is the depth of a RESET, affordable only because the
            # block is rebuilt on every replay. Under rolling nothing rebuilds it, so the
            # depth replays with nothing handed back (18 evicted where rolling picks 6).
            # Refused HERE, not at the fit: `boundary_messages` is re-recorded every turn,
            # so a rolling turn that inherited 18 would persist it with no `checkpoint` key
            # and make the reset-sized window permanent.
            #
            # The other way, a rolling boundary reaching `fit_checkpoint_context` reads as
            # an epoch already in force, so the reset returns `checkpoint_started` false
            # and the automatic recall never runs. That only happens where the request may
            # actually reset; `_can_reset_epoch` false means the fit stays rolling whatever
            # the process policy says, and there the rolling boundary is the right one to
            # replay. Rejecting it there instead let the boundary slide every turn for any
            # model whose template cannot render tools.
            recorded_checkpoint = bool(truncation.get("checkpoint"))
            effective_checkpoint = _request_uses_checkpoint(context_policy)
            if recorded_checkpoint and not effective_checkpoint:
                return 0, False, False
            if not recorded_checkpoint and effective_checkpoint and can_reset:
                return 0, False, False
            # Changing the extra trim has to change something. Phase one of
            # `fit_rolling_context` re-applies this count before anything else and stops
            # once the result fits, so a boundary cut under a DIFFERENT ratio never reaches
            # the phase that reads the new one, in either direction: measured on a
            # 121-message transcript at context_length 1600, a stored boundary of 86 gave
            # `dropped 86` whether the request asked for 0.05 or 0.25. Refusing it makes
            # the fit recompute, which is the only place the new ratio is read. Absent on
            # rows saved before the ratio was recorded, which keep replaying as they did.
            if not recorded_checkpoint:
                recorded_ratio = truncation.get("boundary_headroom_ratio")
                try:
                    if (
                        recorded_ratio is not None
                        and abs(float(recorded_ratio) - requested_ratio) > 1e-9
                    ):
                        return 0, False, False
                except (TypeError, ValueError):
                    return 0, False, False
            # A count is only valid against the transcript it was counted on. Deleting an
            # already-evicted turn shortens the front, and replaying the count then evicts
            # that many LIVE messages instead. Re-derive it from the anchor's position on
            # this request's own branch. Only ever downward: an anchor that has moved back
            # (a repeated text, an edited turn) must not deepen the cut.
            anchor = truncation.get("boundary_anchor")
            if isinstance(anchor, str) and anchor:
                for index, branch_message in enumerate(_branch_non_system(branch_wire)):
                    if _anchor_text(branch_message) == anchor[:_ANCHOR_TEXT_CHARS]:
                        recorded = min(recorded, index)
                        break
            boundaries.append(recorded)
            origins.append(recorded_checkpoint)
            manual.append(recorded_checkpoint and truncation.get("manual") is True)
        if not boundaries:
            return 0, False, False
        return min(boundaries), all(origins), all(manual)
    except Exception:
        return 0, False, False


def _sticky_compaction_manual(
    thread_id: Optional[str],
    branch_messages: Optional[list[dict]] = None,
    context_policy: Optional[str] = None,
    can_reset: bool = True,
    compaction_headroom_ratio: Optional[float] = None,
) -> bool:
    """Whether the epoch `_sticky_compaction_state` replays was started by /compact."""
    return _sticky_compaction_details(
        thread_id,
        branch_messages,
        context_policy = context_policy,
        can_reset = can_reset,
        compaction_headroom_ratio = compaction_headroom_ratio,
    )[2]


def _sticky_compaction_boundary(
    thread_id: Optional[str],
    branch_messages: Optional[list[dict]] = None,
    context_policy: Optional[str] = None,
    can_reset: bool = True,
    compaction_headroom_ratio: Optional[float] = None,
) -> int:
    """`_sticky_compaction_state` without the provenance, for callers that only fit."""
    return _sticky_compaction_state(
        thread_id,
        branch_messages,
        context_policy = context_policy,
        can_reset = can_reset,
        compaction_headroom_ratio = compaction_headroom_ratio,
    )[0]


def _prefix_user_text(message: dict, prefix: str) -> dict:
    """Copy of ``message`` with ``prefix`` in front of its text. Never mutates."""
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return {**message, "content": prefix + (content or "")}
    if isinstance(content, list):
        parts = list(content)
        for index, part in enumerate(parts):
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts[index] = {**part, "text": prefix + str(part.get("text") or "")}
                return {**message, "content": parts}
        return {**message, "content": [{"type": "text", "text": prefix}] + parts}
    return message


def _recall_top_k(budget_tokens: int) -> int:
    """Sizes the recall from the room actually left, not the reserve, which a fit may not reach."""
    try:
        from core.rag import config as rag_config
        chunk = max(1, int(rag_config.CHUNK_TOKENS))
        cap = max(0, int(rag_config.CONVERSATION_ARCHIVE_TOP_K))
    except Exception:
        return 0
    fits = int(budget_tokens) // chunk
    if fits <= 0 and int(budget_tokens) > 0:
        # CHUNK_TOKENS is a ceiling; the exact recount below decides.
        fits = 1
    return max(0, min(cap, fits))


def _archive_and_recall(
    conversation: list[dict],
    before: list[dict],
    *,
    thread_id: Optional[str],
    style: str,
    recall_done: bool,
    force_recall: bool = True,
    recall_budget_tokens: int = 0,
    count_tokens: Optional[Callable[[list[dict]], int]] = None,
    branch_messages: Optional[list[dict]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    background: bool = False,
) -> dict:
    """Never raises: any failure falls back to the plain fitted messages. Recall runs at most once per request or refits stack recall blocks."""
    result = {
        "conversation": conversation,
        "events": [],
        "counts": {},
        "recalled": False,
        "anchored": [],
    }
    if not thread_id:
        return result
    try:
        from core.inference.tools import build_conversation_recall
        from core.rag import conversation_archive
        from core.rag.retrieval import query_vector_memo

        if not conversation_archive.enabled():
            return result

        gone = evicted_messages(before, conversation)
        if background:
            # llama.cpp only: its streams gate the idle-gap worker (`inference_running`).
            conversation_archive.note_branch(thread_id, before, cancel_check)
        # `conversation` is the fitted prompt, so it says which repeats of a turn the model
        # can still read. Without it a twice-said turn is archived twice and spends two
        # recall slots on text already in the prompt.
        archived = (
            conversation_archive.archive_turns(
                thread_id,
                gone,
                live = conversation,
                branch = branch_messages or conversation,
                cancel_check = cancel_check,
            )
            if gone
            else 0
        )
        counts = {"archived_messages": archived}

        if recall_done or (cancel_check is not None and cancel_check()):
            result["counts"] = counts
            return result
        if not force_recall:
            # Checkpoint mode past the first turn: archiving still runs (the epoch stops being searchable otherwise); auto lookup only on the reset turn.
            result["counts"] = counts
            return result

        top_k = _recall_top_k(recall_budget_tokens)
        if top_k <= 0:
            # No room, but archiving still runs so turns stay searchable and the next fit can recall them.
            result["counts"] = counts
            return result

        # Exact recount decides; on overflow halve the turn count (4 -> 2 -> 1) instead of giving up.
        attempts = []
        attempt_k = top_k
        while True:
            attempts.append(attempt_k)
            if attempt_k <= 1:
                break
            attempt_k = max(1, attempt_k // 2)

        # Retries re-ask the same queries: encode each once and reuse lexical passes and branch checks.
        with query_vector_memo(), conversation_archive.recall_session():
            for index, k in enumerate(attempts):
                recall = build_conversation_recall(
                    conversation,
                    thread_id,
                    style = style,
                    top_k = k,
                    branch_messages = branch_messages,
                )
                if not recall:
                    result["counts"] = counts
                    return result

                candidate = _inject_recall(conversation, recall, style)
                if candidate is None:
                    result["counts"] = counts
                    return result

                grew = None
                if count_tokens is not None and recall_budget_tokens > 0:
                    try:
                        grew = count_tokens(candidate["conversation"]) - count_tokens(conversation)
                    except Exception:
                        grew = None
                if grew is not None and grew > recall_budget_tokens:
                    logger.info(
                        "conversation_recall.over_budget grew=%s budget=%s top_k=%s",
                        grew,
                        recall_budget_tokens,
                        k,
                    )
                    if index + 1 < len(attempts):
                        continue
                    logger.info(
                        "conversation_recall.dropped_over_budget budget=%s", recall_budget_tokens
                    )
                    result["counts"] = counts
                    return result

                result["conversation"] = candidate["conversation"]
                result["events"] = candidate["events"]
                result["anchored"] = candidate["anchored"]
                break

        counts["recalled_chunks"] = int(recall.get("sources") or 0)
        result["counts"] = counts
        result["recalled"] = True
    except Exception as exc:
        logger.warning("Could not archive or recall compacted turns: %s", exc)
    return result


def _inject_recall(conversation: list[dict], recall: dict, style: str) -> Optional[dict]:
    """Only the injected messages are anchored; anything else would pin a real turn the fit may evict."""
    if style == "inline":
        prefix = recall.get("prefix") or ""
        updated = list(conversation)
        for index in range(len(updated) - 1, -1, -1):
            if updated[index].get("role") == "user":
                updated[index] = _prefix_user_text(updated[index], prefix)
                break
        else:
            return None
        return {"conversation": updated, "events": [], "anchored": [updated[index]]}
    added = list(recall["messages"])
    return {
        "conversation": list(conversation) + added,
        "events": list(recall["events"]),
        "anchored": added,
    }
