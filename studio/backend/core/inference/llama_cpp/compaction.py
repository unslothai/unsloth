# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Context fitting, rolling compaction, conversation archive and recall."""

import bisect

from typing import (
    Callable,
    NamedTuple,
    Optional,
)
from core.inference.context_window import (
    _COMPACTION_HEADROOM_RATIO,
    clamp_compaction_headroom_ratio,
    evicted_messages,
    fit_oversized_attachments,
    fit_rolling_context,
    prompt_budget,
    retrieval_budget,
)

from .log import logger


def _backend_supports_tools(backend) -> bool:
    """`backend.supports_tools`, or False if asking is not safe.

    The property reads load-time state a backend mid-construction or teardown may not have
    yet (measured: AttributeError without the diffusion flag), and a context-policy probe
    has no business raising on the chat path.
    """
    try:
        return bool(backend.supports_tools)
    except Exception:  # noqa: BLE001
        return False


def _memory_tool_withheld(thread_id, tools) -> bool:
    """Whether this request carries no `search_conversation` although its thread has one.

    The tool-loop paths know their own catalogue, so they can answer what the process
    policy cannot: will THIS request reach the archive. A request that NAMED its tools is
    the case that matters on either surface, since both honour the caller's list verbatim;
    resetting an epoch behind a tool that is then absent every turn is the one outcome
    checkpoint compaction must never produce.

    Gated on the thread ALREADY having an archive, which is load-bearing, not an
    optimisation: the archive is written during the first compaction, so on the turn that
    first resets the tool legitimately does not exist yet and its absence says nothing.
    Refusing there would mean no thread could ever start an epoch.
    """
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
    """Whether this request may compact by RESETTING rather than trimming.

    Two refusals, both about not lying to the user:

    * the dropped turns must reach the archive, or the reset makes them unreachable while
      the notice says they are searchable. Incognito, API-only and threadless requests
      archive nothing, so they keep the rolling window.
    * the model must be able to receive `search_conversation` at all, so a template that
      cannot render tools, or a process with the tool loop off, is refused.
    """
    if not thread_id or not supports_tools:
        return False
    if tools_withheld:
        # The REQUEST withdrew the tool loop, which the process policy below cannot see
        # (`tool_choice: "none"` also skips the checkpoint repair that would re-admit
        # `search_conversation` alone). A caller that sets it once usually sets it every
        # turn, so resetting would put the epoch behind a tool that never arrives while the
        # carried-forward header tells the model to search. Same refusal, one scope down.
        return False
    try:
        from state.tool_policy import conversation_recall_allowed

        # `supports_tools` is the TEMPLATE's capability; where recall is refused a reset would
        # strand the epoch behind a tool that never arrives, and rolling needs no tool.
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
    """Whether an archive write for THIS request would fail.

    Separate from `_can_reset_epoch`: those gates decide whether an epoch may EXIST at all,
    this only whether a NEW one may start right now, leaving one in force untouched.

    Both halves are needed. `degraded()` is the verdict on the LAST write, the wrong tense:
    this request's write runs afterwards and swallows its failure, so the first request
    after the store dies would claim searchable turns that were never indexed.
    `reachable()` probes the store and embedder now, closing that turn.
    """
    try:
        from core.rag import conversation_archive
        return bool(conversation_archive.degraded()) or not conversation_archive.reachable()
    except Exception:  # noqa: BLE001 -- an unreadable flag is "healthy", never an error
        return False


def _compaction_fit_kwargs(
    context_policy: Optional[str] = None, compaction_headroom_ratio: Optional[float] = None
) -> dict:
    """Per-request overrides the four rolling-fit call sites share.

    Unset values keep the process defaults (UNSLOTH_CONTEXT_POLICY and
    ROLLING_COMPACTION_HEADROOM_RATIO) so an older client is unaffected.
    """
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
    """The context policy in one place, so the five call sites do not each pick one.

    Checkpoint mode resets the epoch; rolling mode is the previous behaviour, still
    reachable via `UNSLOTH_CONTEXT_POLICY=rolling` as the A/B arm and the escape hatch for
    a misbehaving template family. A request that may not reset (see `_can_reset_epoch`)
    silently keeps rolling, which is why the two fits share a signature.

    ``context_policy`` on the request overrides the process default so Studio can offer
    a sliding window without restarting the server.
    """
    can_reset = bool(kwargs.pop("can_reset", False))
    # Whether `sticky_dropped` is the depth of a checkpoint RESET. Defaults to True, which
    # is what a caller with no thread state to consult has always assumed.
    sticky_is_checkpoint = bool(kwargs.pop("sticky_is_checkpoint", True))
    # Naming a tool the request lacks makes the model print the call as text.
    recall_offered = bool(kwargs.pop("recall_offered", True))
    requested_policy = kwargs.pop("context_policy", None)
    if requested_policy not in ("checkpoint", "rolling"):
        requested_policy = None
    # /compact: only a checkpoint reset can honour it, since only its turns are archived.
    force_reset = bool(kwargs.pop("force_reset", False))
    sticky_manual = bool(kwargs.pop("sticky_manual", False))
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
    try:
        from core.inference import checkpoint

        use_checkpoint = _request_uses_checkpoint(requested_policy)
        if (
            not can_reset
            and use_checkpoint
            and sticky_is_checkpoint
            and int(kwargs.get("sticky_dropped") or 0) > 0
        ):
            # An epoch is in force but THIS request may not reset. Falling straight through
            # to rolling was the worst of both: it replays the checkpoint-sized (near-total)
            # eviction WITHOUT rebuilding the block that made it survivable. Measured on a
            # 24-message thread: 22 dropped either way, but rolling left the standing
            # instruction gone entirely, one turn after the user was told the conversation
            # was compacted and searchable. So replay the epoch and keep the block, with the
            # header's tool sentence swapped for one that promises no lookup this request
            # cannot perform. If the replay cannot fit, fall through to rolling BELOW.
            fitted, truncation = checkpoint.fit_checkpoint_context(
                messages,
                can_reset = False,
                searchable = False,
                sticky_manual = sticky_manual,
                **kwargs,
            )
            if truncation is not None and truncation.get("fits"):
                return fitted, truncation
            # A checkpoint boundary means nothing to rolling, which cannot rebuild what it
            # assumed. Recomputing loses the epoch: an un-compacted window tells no lie.
            #
            # `sticky_is_checkpoint` false skips this branch entirely and keeps the count:
            # rolling recorded it, rolling replays it, and the rolling fit reports no
            # `checkpoint_started`, so `_archive_and_recall` still runs the inline recall.
            kwargs.pop("sticky_dropped", None)
        if can_reset and use_checkpoint:
            # A degraded archive downgrades reset to REPLAY rather than closing the door.
            # Refusing outright sent the request to rolling, which replays the same boundary
            # WITHOUT rebuilding the block, so a thread with an epoch silently lost its
            # standing instructions (and `degraded()` does not self-clear on a broken
            # embedder, so "transient" is not safe to assume). Only starting a NEW epoch is
            # refused, the half that would promise a history nobody is writing.
            #
            # `searchable` goes with it: the archive write runs AFTER this fit and swallows
            # its failure, so without it the block would claim the dropped turns are
            # retrievable while nothing was indexed, and repeat that claim every later turn.
            # Cleared, it says stored but not retrievable now -- true either way -- and the
            # tool is still offered so a recovered archive is not walled off.
            # Resolved lazily, and once. Establishing this probes the store and runs a
            # real embedding forward, and it was being paid on EVERY persisted tool-capable
            # request using truncate_oldest -- including short conversations that never
            # overflow and never render a block. The fit asks only where the answer
            # changes what it does: before starting a new epoch, and before claiming a
            # block is searchable.
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
            # `can_reset = False` has no phase two, so it refuses once the replayed
            # boundary is no longer enough (measured: a threadless request went from
            # `dropped 4, fits True` to `fits False`). Rolling still serves those, so a
            # refusal here falls through rather than reaching the caller.
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
    """`fit_rolling_context`, plus the user's standing instructions held back from eviction.

    Applied through the existing `protected_message_ids` seam rather than by changing the
    evictor, so the rolling-window layer is untouched and this ships as an addition to it.

    Three things this must never do, each of which the code below is shaped by:

    * mutate the caller's anchor set. It is also the recall-anchor set and it is carried
      across tool-loop iterations, so a pin id leaking into it would outlive the message
      the id belonged to.
    * hold pins computed against an older copy of the conversation. Inline recall REPLACES
      the latest user message with a new dict, so ids go stale; recomputing per fit is the
      only version that stays true.
    * turn a servable request into a refusal. If pinning is what made the prompt not fit,
      the pins are dropped and the fit is redone without them. One extra fit, only on a
      path that was already failing.
    """
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
    # Only a refusal that returned the ORIGINAL messages was "already failing". A rescue
    # reports `fits` false too but is servable, and dropping its pins would trade a working
    # prompt WITH the standing instruction for one without, the very thing this seam
    # prevents. `dropped_messages` separates them.
    if (
        pins
        and truncation
        and not truncation.get("fits")
        and not int(truncation.get("dropped_messages") or 0)
    ):
        return _fit_context(messages, protected_message_ids = anchors or None, **kwargs)
    return fitted, truncation


# Default max_tokens to the effective context when known. The floor is high
# enough for reasoning-heavy GGUFs and max_tokens-omitting API clients.
_DEFAULT_MAX_TOKENS_FLOOR = 32768
_DEFAULT_FIRST_TOKEN_TIMEOUT_S = 1200.0  # 20 min


def _conversation_recall_reserve(thread_id: Optional[str]) -> int:
    """Room to hold back during a fit for the turns recalled straight after it."""
    if not thread_id:
        return 0
    try:
        from core.rag import config as rag_config
        from core.rag import conversation_archive

        # Eligibility, not just availability. A temporary chat is never archived, so the
        # reserve stays empty while still being subtracted from the trim target. Measured
        # on a 4K chat: 15 tokens of conversation survived instead of 1,615.
        if not conversation_archive.can_archive(thread_id):
            return 0
        # And that it works: sqlite-vec can be present while the embedder cannot start,
        # so archiving fails quietly and the reserved room is pure loss on every
        # compaction, forgetting more history than having the feature off.
        if conversation_archive.degraded():
            return 0
        return max(0, int(rag_config.CONVERSATION_RECALL_RESERVE_TOKENS))
    except Exception:
        return 0


def _keeps_compaction_boundary(thread_id: Optional[str]) -> bool:
    """Whether this request's boundary will be there to restore next time.

    The boundary rides on the assistant turn's saved metadata, so a thread with no saved
    messages (incognito, or an API caller that persists nothing) never gets one back. The
    fit uses this to decide whether cutting deeper than needed buys anything.
    """
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
    """How many of the REQUEST's own leading messages a fit has evicted.

    The boundary is persisted and re-applied to the NEXT request's saved transcript, so it
    must be counted in those terms. ``dropped_messages`` is not: the tool loop refits each
    iteration and the client sums the counts, so a long agent run also counts the tool
    exchanges it created, which the next request's transcript does not contain.

    Counted by identity. Instruction messages are skipped rather than treated as the front
    of the branch, or an Unsloth request (always system-prefixed) would report zero on every
    compaction and the boundary would slide every turn again. Everything from the newest
    USER turn on is excluded: it is never evicted, and an inline recall rewrites that turn
    into a new dict, which an identity scan reads as an eviction. Excluding just the last
    message is not the same thing, since a continued assistant message follows it.

    Every evicted message is counted, not just the leading run, because that is what the
    replay consumes: ``truncate_oldest_messages``'s ``min_dropped`` counts dropped messages
    and SKIPS protected groups. Stopping at the first survivor matched that only while
    eviction was a clean prefix, which a protected group (an instruction pin, an anchored
    recall) breaks, leaving live turns on both sides. Under a checkpoint reset that
    understated the boundary enough to un-compact the epoch: the removed turns returned on
    the next request, one turn after the user was told they were compacted away. Where
    eviction IS a prefix, which is every pin-free rolling case, the count is unchanged.
    """
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
    """Whether a fit removed turns from the prompt it sent, so its depth is worth saving.

    Not ``fits``: a rescue reports false yet really did evict, and the client reads this
    depth to place the compaction notice, so recording nothing compacts silently. Saving
    is not replaying -- `_sticky_compaction_boundary` still declines any `fits` false
    record, which is where a missed reserve belongs.
    """
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
    """Where this fit put the boundary, and under how much extra trim.

    The ratio rides with the depth because the depth cannot be read without it: phase one
    of `fit_rolling_context` re-applies the saved count before anything else and stops once
    the result fits, so a chat whose setting has moved either way would replay the old cut
    and never reach the phase that consults the new one.
    """
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
    """The text of the first message a fit kept AFTER everything it dropped, or "".

    ``boundary_messages`` is a count against one particular transcript, and the next
    request's transcript is not guaranteed to be that one plus new turns: deleting an
    already-evicted prompt shortens the front, so replaying the count unchanged lands the
    boundary two messages too deep and evicts history that is still live. The anchor names
    the message the boundary was meant to land ON, which survives a deletion in front of it.

    After the LAST eviction, not simply the first survivor. Eviction is a clean prefix only
    while nothing is protected mid-list; an instruction pin or an anchored recall leaves
    live turns on both sides, and the first survivor is then the pin itself, sitting near
    the front. `_sticky_compaction_boundary` clamps the count down to the anchor's index,
    so anchoring on the pin cut the replayed boundary back to the pin and handed back every
    turn compacted after it -- one turn after the user was told they were compacted away,
    which is the same un-compaction `_branch_boundary`'s own count was fixed to prevent.

    Text, not an id or an index: ids are per-request objects and an index is the very thing
    that goes stale. Read back by ``_sticky_compaction_boundary``, which only ever lets the
    anchor make the boundary SHALLOWER, so a stale or ambiguous anchor costs one extra
    compaction and can never evict a live turn.
    """
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


# Bounded so one long tool argument cannot bloat the metadata the UI stores per turn. The
# name plus the head of the arguments is already enough to tell two calls apart.
_ANCHOR_ARGS_CHARS = 200
# And the anchor as a whole. It rides in every `context_truncated` event and is persisted
# on every assistant turn for as long as the boundary stays sticky, so returning a large
# pasted prompt whole duplicated it across the thread's SSE payloads and history rows. A
# head is enough to name a message, and the read side clamps only ever SHALLOWER, so two
# messages sharing a head cost one extra compaction rather than a live turn.
_ANCHOR_TEXT_CHARS = 200


def _anchor_text(message: dict) -> str:
    """A message as the anchor spells it: its text, or its tool calls when it has none.

    A normal OpenAI assistant tool-call message carries empty ``content`` and puts its
    substance in ``tool_calls``. Reading only ``content`` recorded an empty anchor there,
    which silently disables the rebase: the next request cannot re-derive the boundary
    against a shortened transcript and replays the stale count, evicting one more live
    message per deleted one. Both sides of the comparison use this, so what is written is
    what is looked for.
    """
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
    """Whether the client re-sends this reply, by `isAbandonedAssistantTurn`.

    Status is not the test: a Stop that reached text is kept, and only a reply carrying
    nothing is dropped -- along with the tool results its wire projection would expand to.
    A turn that FINISHED on reasoning alone is a reply too, replayed as `reasoning_content`,
    even though `_as_wire` strips reasoning and leaves it nothing to be matched on.
    """
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

            # Rows past that tip ride on the unstored turns alone, so they may only carry
            # THOSE turns' text: matching the whole request instead let an abandoned row
            # in on a text repeated earlier in it. Keyed by ROLE too, as the branch match
            # is, so a stored reply cannot be justified by a user turn of the same words.
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
                    # Nothing comparable came back. Only a reply the client drops whole may
                    # ride on the unstored turns unchecked; one it re-sends -- an image or
                    # an attachment, or a tool call still awaiting its result -- is not
                    # proof of anything, so refuse rather than take it on trust.
                    if not checked and _row_replayed(row):
                        return None
                past_tip = past_tip or row.get("id") == tip
        return chain
    except Exception:
        return None


def _archive_as_wire(messages: Optional[list[dict]]) -> list[dict]:
    """Project persisted rows and wire messages into boundary-counting wire units.

    Unsanitised: `_branch_boundary_anchor` writes the anchor off the request itself, so a
    projection that rewrote assistant text here would look for something nobody wrote. A
    client sending the tokens raw, which the Studio one never does, then found no anchor
    and replayed the stale count instead of rebasing it.
    """
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
    """Newest authoritative compaction state(s) on one request branch.

    Multiple rows are returned only when Retry siblings are textually indistinguishable;
    consumers must then choose the conservative result across all of them.
    """
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
            # Skipping defers to the previous epoch, so it needs a proved branch: on the text
            # path the rows are twins. A research row reports its run, not whether this fit.
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


# A readable local name for the shared `context_window` policy, not a second implementation.
_retrieval_budget = retrieval_budget


def _recall_top_k(budget_tokens: int) -> int:
    """How many archived chunks actually fit in the room a fit obtained.

    The reserve is what the fit AIMS to hold back, not what it gets: protected messages
    (system, latest turn, anchored recalls) can stop the trim reaching its target while
    still passing the prompt budget. Injecting a full reserve on top then overflows the
    window -- reproduced at ctx 8000: fit accepted at 6900, recall took it to 8948. So
    size the recall from the room actually left.
    """
    try:
        from core.rag import config as rag_config
        chunk = max(1, int(rag_config.CHUNK_TOKENS))
        cap = max(0, int(rag_config.CONVERSATION_ARCHIVE_TOP_K))
    except Exception:
        return 0
    fits = int(budget_tokens) // chunk
    if fits <= 0 and int(budget_tokens) > 0:
        # One attempt whenever there is ANY room. `CHUNK_TOKENS` is a ceiling, not the
        # size of a turn: most archived turns are far smaller, so flooring to zero
        # disabled recall on exactly the tight windows that need it. The exact recount
        # below is what actually decides, and it rejects a chunk that does not fit.
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
) -> dict:
    """Keep the turns this fit evicted, then pull the relevant ones back.

    Returns ``{"conversation", "events", "counts", "recalled"}``. Never raises: every
    failure degrades to the plain fitted messages. Recall fires at most once per request,
    or the tool loop's per-iteration refit would stack a recall block onto each pass.
    """
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

        if not conversation_archive.enabled():
            return result

        gone = evicted_messages(before, conversation)
        # `conversation` is the fitted prompt, so it says which repeats of a turn the model
        # can still read. Without it a twice-said turn is archived twice and spends two
        # recall slots on text already in the prompt.
        archived = (
            conversation_archive.archive_turns(
                thread_id, gone, live = conversation, branch = branch_messages or conversation
            )
            if gone
            else 0
        )
        counts = {"archived_messages": archived}

        if recall_done:
            result["counts"] = counts
            return result
        if not force_recall:
            # Checkpoint mode, past the first turn of the epoch. Archiving still ran above
            # (it must, or the epoch stops being searchable), but the automatic lookup fires
            # only on the turn that reset the conversation; after that the model has
            # `search_conversation` and decides for itself.
            result["counts"] = counts
            return result

        top_k = _recall_top_k(recall_budget_tokens)
        if top_k <= 0:
            # No room obtained. Archiving still happened, so the turns stay searchable
            # and the next fit can recall them.
            result["counts"] = counts
            return result

        # An EXACT recount decides affordability: the chunk arithmetic above is doubly an
        # estimate, since CHUNK_TOKENS is an embedding-token limit rather than the chat
        # template's cost and neither it nor the budget covers the wrapper text.
        # When the recount says no, ask for FEWER turns rather than giving up: with the
        # shipped defaults a full top-K of long turns lands just over the reserve once
        # wrapped, and dropping it would disable recall on exactly the long conversations
        # this exists for. Halving is bounded (4 -> 2 -> 1), one small query per attempt.
        attempts = []
        attempt_k = top_k
        while True:
            attempts.append(attempt_k)
            if attempt_k <= 1:
                break
            attempt_k = max(1, attempt_k // 2)

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
    """The conversation with ``recall`` added, or None if there was nowhere to put it.

    ``anchored`` is only what this injection added: the two synthetic tool messages, or
    the rewritten user message inline. The caller protects those from a later refit, and
    anything else would pin a real turn the fit is entitled to evict.
    """
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
