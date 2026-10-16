# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: tool-calling agentic loop."""

import hashlib
import httpx
import json
import re
import threading
import time

from typing import (
    Any,
    Callable,
    Generator,
    Optional,
)
from core.inference.context_window import (
    _reply_floor,
    compact_completed_tool_arguments,
    compact_executed_call_arguments,
    compact_refused_tool_arguments,
    estimate_message_tokens_without_unpriced_media,
    estimate_messages_tokens_conservative,
    estimate_messages_tokens_dense,
    estimate_messages_tokens_without_unpriced_media,
    messages_without_unpriced_media,
    prompt_budget,
    tool_result_budget,
    turn_is_servable,
)
from core.inference.llama_tool_schema import llama_grammar_tools
from core.inference.mcp_images import (
    append_image_turn as append_mcp_image_turn,
    DETACHED_IMAGE_TURN_TEXT as MCP_DETACHED_IMAGE_TURN_TEXT,
    IMAGE_TURN_TEXT as MCP_IMAGE_TURN_TEXT,
    is_image_tool as mcp_is_image_tool,
    MAX_MODEL_IMAGES as MCP_MAX_MODEL_IMAGES,
    MAX_TOTAL_MODEL_IMAGES as MCP_MAX_TOTAL_MODEL_IMAGES,
    mentions_images as mcp_images_mentioned_in,
)
from core.inference.passthrough_healing import nudge_enabled as _nudge_enabled
from core.inference.repetition_guard import is_repetition_dominated
from core.inference.stream_errors import stream_error_from_chunk
from core.inference.tool_call_parser import (
    _balanced_brace_end,
    _GEMMA_BARE_TC_PREFIX_RE,
    blocked_bare_json_chain_may_continue,
    blocked_gemma_chain_may_continue,
    BUDGET_EXHAUSTED_NUDGE,
    continue_after_length_message as _continue_after_length_message,
    is_reprompt_repeat as _is_reprompt_repeat,
    is_short_intent_without_action as _is_short_intent_without_action,
    leading_bare_gemma_call_is_promotable,
    MAX_ACT_REPROMPTS as _MAX_REPROMPTS,
    NUDGE_TOOL_CALLS_STATUS as _NUDGE_TOOL_CALLS_STATUS,
    RAG_MAX_SEARCHES_PER_TURN,
    RAG_SEARCH_CAP_NUDGE,
    RAG_SEARCH_TOOLS,
    reasoning_cap_spent_message as _reasoning_cap_spent_message,
    repeated_result_message as _repeated_result_message,
    REPROMPT_MAX_CHARS as _REPROMPT_MAX_CHARS,
    reprompt_to_act_message as _reprompt_to_act_message,
    starved_result_message as _starved_result_message,
    StreamingMarkupStripper as _StreamingMarkupStripper,
    strip_leading_bare_json_call,
    strip_llama3_leading_sentinels,
    strip_tool_markup as _shared_strip_tool_markup,
    thinking_exhausted_message as _thinking_exhausted_message,
    TOOL_XML_SIGNALS as _SHARED_TOOL_XML_SIGNALS,
    unfinished_thought_progress as _unfinished_thought_progress,
)
from core.inference.tool_loop_controller import (
    _reruns_after_new_work,
    _WORKSPACE_READ_TOOLS,
    append_deferred_nudges,
    awaiting_approval_status,
    BROWSER_TOOL_NAMES,
    deferred_nudge_text,
    provisional_tool_provenance,
    restore_replayed_tool_results,
    tool_call_limit_nudge,
    ToolLoopController,
    with_appended_text,
)
from state.tool_approvals import (
    abort_tool_decision,
    begin_tool_decision,
    DECISION_EXPIRED,
    decision_reason,
    new_approval_id,
    TOOL_APPROVAL_EXPIRED_MESSAGE,
    TOOL_REJECTED_MESSAGE,
    wait_tool_decision,
)

from .answer_artifacts import (
    _has_answer_artifact,
    _text_outside_think,
)
from .compaction import (
    _DEFAULT_MAX_TOKENS_FLOOR,
    _archive_and_recall,
    _backend_supports_tools,
    _boundary_metadata,
    _can_reset_epoch,
    _compaction_fit_kwargs,
    _conversation_recall_reserve,
    _fit_with_instruction_pins,
    _keeps_compaction_boundary,
    _memory_tool_withheld,
    _records_boundary,
    _retrieval_budget,
    _sticky_compaction_details,
    _sticky_compaction_state,
)
from .core_types import (
    _LlamaStreamCancelled,
    _apply_seeded_llama_request,
)
from .log import logger
from .timings import (
    _backfill_usage_from_timings,
    _context_usage_event,
    _perf_callback_wants_timings,
    _report_live_llama_timings,
)
from .tool_loop_text import (
    _CONTINUE_AFTER_LENGTH_STATUS,
    _CONTINUE_TRUNCATED_ANSWER_STATUS,
    _MAX_IDENTICAL_TOOL_RESULTS,
    _MAX_LENGTH_CONTINUATIONS,
    _MAX_TOOL_CALLS_PER_TURN,
    _MIN_USEFUL_RESULT_TOKENS,
    _PROVISIONAL_ARGS_MIN_CHARS,
    _UNSET,
    _finalize_reasoning_only_cumulative,
    _gguf_active_tool_names,
    _gguf_has_genuine_tool_signal,
    _gguf_rehearsal_signal_pos,
    _held_rehearsal_tail_len,
    _is_rehearsal_prefix,
    _is_window_notice,
    _mcp_image_tokens,
    _note_before_envelopes,
    _quote_cut_event,
    _should_suppress_forced_no_tool_output,
    _sniff_text_tool_name,
)


class _ToolLoopMixin:
    # ── Tool-calling agentic loop ──────────────────────────────

    def generate_chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        replayed_image_parts: "tuple" = (),
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        # False when the caller knows the history held reasoning its conversion dropped.
        replay_preserve_thinking: bool = True,
        continue_final_message: bool = False,
        max_tool_iterations: int = 25,
        auto_heal_tool_calls: bool = True,
        nudge_tool_calls: Optional[bool] = None,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        rag_scope: Optional[dict] = None,
        seed: Optional[int] = None,
        disable_parallel_tool_use: bool = False,
        confirm_tool_calls: bool = False,
        bypass_permissions: bool = False,
        permission_mode: Optional[str] = None,
        promote_reasoning_only: bool = True,
        perf_callback: Optional[Callable[[dict], None]] = None,
        reasoning_provenance: Optional[dict] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        tool_choice: Any = None,
        # Appended, never inserted: no bare `*` here, so every parameter is
        # positional-or-keyword and inserting one rebinds later positional arguments.
        #
        # Called at the top of every round with the conversation as it now stands, so KV
        # admission can charge what this run occupies rather than its opening estimate.
        # MAY BLOCK: recost_waiting waits for cache room. Safe at the top of a round,
        # where the previous round's request has completed.
        on_conversation_grew: Optional[Callable[[list], None]] = None,
        on_decode_slot: Optional[Callable[[str, int], None]] = None,
        thinking_budget_tokens: Optional[int] = None,
        mcp_image = None,
        deduplicate_tool_calls: bool = True,
        instruction_anchor_ids = None,
        sandbox_level: Optional[str] = None,
        compact_context: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        permission_mode: "ask" confirms every call (with confirm_tool_calls),
        "auto" only pauses calls detected as potentially unsafe, "off" never
        pauses (sandbox stays on), "full" is the same as bypass_permissions.
        Unset/unknown behaves as "ask".

        Yields dicts:
          {"type": "status", "text": "Searching: ..."/"Reading: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        if replay_preserve_thinking:
            preserve_thinking = self._replay_preserve_thinking(
                preserve_thinking, messages, enable_thinking, reasoning_effort
            )
        from core.inference.tool_stream_exec import (
            accepts_kwarg,
            accepts_output_callback,
            search_images_kwargs,
            stream_tool_execution,
        )
        from core.inference.tools import (
            build_rag_autoinject,
            execute_tool,
            has_text_only_provisional_card,
            is_high_risk_tool_call,
            mcp_image_share,
            mcp_image_targets,
            never_needs_approval,
        )
        from core.inference.mcp_image import note_attached_image

        # "full" and bypass_permissions are the same switch, whichever arrives
        # first wins. "off" keeps the sandbox but never prompts. Unset defaults to
        # "auto"; unknown falls back to the stricter "ask". An explicit
        # confirm_tool_calls=True with no mode is already resolved to "ask" at the
        # request layer, so it never arrives here as an ambiguous unset.
        from state.tool_policy import (
            account_tool_stream,
            needs_tool_confirmation,
            normalize_sandbox_level,
            normalize_tool_permissions,
            requires_os_isolation,
            runs_without_os_sandbox,
            tool_call_may_prompt,
        )

        permission_mode, bypass_permissions = normalize_tool_permissions(
            permission_mode, bypass_permissions
        )
        sandbox_level = normalize_sandbox_level(sandbox_level)
        stream_tool_execution = account_tool_stream(stream_tool_execution)

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        if mcp_image is not None:
            messages = note_attached_image(
                messages, mcp_image_targets(_gguf_active_tool_names(tools))
            )
        conversation = restore_replayed_tool_results(messages)
        # Seeded with tool results an explicit skill load relies on, so fitting keeps them.
        _rolling_anchor_ids: set[int] = set(instruction_anchor_ids or ())
        # The loop refits on every iteration; recall must fire once per request, or each
        # pass stacks another block of recalled turns onto the prompt.
        _conversation_recall_done = False
        # The sticky boundary describes the ORIGINAL transcript, so apply it at most once
        # per request. Each later refit runs on what the previous fit returned, so
        # re-applying the count evicts another boundary-sized block of live history, and
        # the summed events persist the inflated total into the next request.
        _sticky_boundary_applied = False
        # Exact templated prompt tokens minus the estimator's view of the same messages,
        # measured once by the preflight. The estimator cannot see the tool catalogue or
        # the template framing, thousands of tokens with a large (MCP) catalogue: enough
        # for a conversation search to be told there is room it does not have. Carrying
        # the difference makes the estimate exact where it was measured.
        _prompt_token_offset: Optional[int] = None
        # The branch this request is on, kept aside before anything is evicted from or
        # injected into `conversation`. The archive is keyed by thread and the stored rows
        # are the whole DAG (Retry keeps the replaced response as a sibling), so recall
        # has to be told which branch is live, and the client's messages are the answer.
        # Restored dicts are new; both branch lists match by identity.
        _request_branch: list[dict] = list(conversation)
        # ...and everything this request adds to it. A long agent run can evict its own
        # earlier tool exchange, which is archived; filtered against the messages the
        # client sent, that document looks like an abandoned branch and is refused, so the
        # model cannot get back a tool result it still needs to answer. Accumulated rather
        # than read off `conversation`, which has already lost what earlier fits evicted.
        _live_branch: list[dict] = list(conversation)
        _live_branch_ids: set[int] = {id(message) for message in conversation}

        def _extend_live_branch(current: list[dict]) -> list[dict]:
            for message in current:
                if id(message) not in _live_branch_ids:
                    _live_branch_ids.add(id(message))
                    _live_branch.append(message)
            return _live_branch

        for message in reversed(conversation):
            if message.get("role") == "user":
                _rolling_anchor_ids.add(id(message))
                break

        _tool_card_by_message: dict[int, str] = {}
        # Feedback folded in after tool_end, reported so the client replays it.
        _model_note_events: list[dict] = []

        def _attach_internal_feedback_to_tool_result(feedback: str) -> bool:
            """Keep controller instructions inside the current tool exchange."""
            if not conversation or conversation[-1].get("role") != "tool":
                return False
            content = conversation[-1].get("content")
            if not isinstance(content, str):
                return False
            # Replaced, not mutated: ``conversation`` copies the list, not the dicts, and
            # /v1/messages passes a caller's own history through, which may end on a tool
            # result. An in-place edit would leave the nudge there and re-send it next turn.
            previous = conversation[-1]
            conversation[-1] = {**previous, "content": f"{content}\n\n{feedback}"}
            card_id = _tool_card_by_message.pop(id(previous), None)
            if card_id:
                _tool_card_by_message[id(conversation[-1])] = card_id
                _model_note_events.append(
                    {
                        "type": "tool_model_note",
                        "tool_call_id": card_id,
                        "suffix": f"\n\n{feedback}",
                    }
                )
            return True

        # Forced first-pass RAG so a doc question doesn't lose to web_search. Emits
        # the same tool card + citations a real call would. Skip it only when a
        # retrieval call would actually prompt (ask mode); auto never gates the
        # safe search_knowledge_base tool, so retrieval must still run there.
        # off never prompts either, so it also keeps first-pass retrieval.
        from core.inference.chat_template_helpers import (
            forced_tool_name,
            trailing_assistant_resume_kind,
            trailing_assistant_text,
        )

        initial_forced_name = forced_tool_name(tool_choice)
        if initial_forced_name and initial_forced_name not in _gguf_active_tool_names(tools):
            raise ValueError(
                f"Forced tool '{initial_forced_name}' is not enabled for this request."
            )

        # A resumed turn must keep the partial trailing: autoinject appends a tool call
        # plus its result, moving the boundary so the model opens a fresh answer.
        _skip_autoinject = (
            tool_choice == "none"
            or (
                confirm_tool_calls
                and not bypass_permissions
                and permission_mode not in ("auto", "off")
            )
        ) or bool(
            continue_final_message and trailing_assistant_resume_kind(conversation) is not None
        )
        _auto = None if _skip_autoinject else build_rag_autoinject(conversation, rag_scope)
        if _auto:
            for _ev in _auto["events"]:
                yield _ev
            conversation.extend(_auto["messages"])
        _auto_satisfies_forced_choice = bool(_auto) and (
            tool_choice == "required" or forced_tool_name(tool_choice) == "search_knowledge_base"
        )

        _accumulated_completion_tokens = 0
        _accumulated_predicted_ms = 0.0
        _accumulated_predicted_n = 0
        # GGUF buffers reasoning; emit server-side timing before answer text.
        _reasoning_started_at: Optional[float] = None
        _reasoning_summary_emitted = False

        # Gate telling a genuine NAME[ARGS] rehearsal from inactive-name prose; built from the
        # ORIGINAL tools list so a spent one-shot still reads as a tool name. None = no gate.
        from core.inference.tools import mcp_loadable_specs

        # Plus the MCP tools a search can load mid-turn (#10997).
        _loadable_mcp = mcp_loadable_specs(tools)
        _loadable_mcp_names = set(_gguf_active_tool_names(_loadable_mcp))
        _detect_tools = list(tools or []) + _loadable_mcp
        _enabled_names_gate = set(_gguf_active_tool_names(_detect_tools)) if tools else None
        # Detection must see the same names as the strip gate (ORIGINAL list, incl. a spent
        # one-shot), else its repeat is stripped but never drained and the turn ends blank.

        def _reasoning_summary_event(started_at: float) -> dict:
            return {
                "type": "reasoning_summary",
                "duration_ms": round((time.monotonic() - started_at) * 1000.0),
            }

        # Enabled-name gate for the markerless Gemma strip (disabled/example
        # names stay visible). Set per iteration; None = pre-loop name-agnostic.
        _enabled_tool_names = None

        def _gemma_lead_promotable(text: str) -> bool:
            # Reads _enabled_tool_names at call time: it is rebound each tool iteration.
            return leading_bare_gemma_call_is_promotable(text, _enabled_tool_names)

        def _strip_tool_markup(
            text: str,
            *,
            final: bool = False,
            force: bool = False,
        ) -> str:
            if not (auto_heal_tool_calls or force):
                return text
            # Delegate to the shared parser-side strip so the GGUF cleanup covers every family the
            # parser promotes (Llama <|python_tag|>, Mistral [TOOL_CALLS], bare rehearsal, function
            # XML, Gemma) and stays aligned with detection; tool_healing's strip omits the loop-only
            # forms (python_tag / Mistral name) and would leak them into display.
            return _shared_strip_tool_markup(
                text, final = final, enabled_tool_names = _enabled_names_gate
            )

        # Wraps the parser's ``strip_segment`` so the accumulated response is not rescanned
        # end to end on every content token (that was quadratic in the response length).
        # Think blocks stay verbatim inside it: a rehearsed call in one must not be deleted.
        _streaming_stripper = _StreamingMarkupStripper(_enabled_names_gate)

        # The final-answer loop has its own buffer and strips with ``final = False``, so it
        # needs its own instance rather than sharing the one above.
        _final_answer_stripper = _StreamingMarkupStripper(_enabled_names_gate, seg_final = False)

        def _strip_tool_markup_streaming(text: str, *, force: bool = False) -> str:
            if not (auto_heal_tool_calls or force):
                return text
            return _streaming_stripper.strip(text)

        def _build_metadata_event(usage, timings, finish_reason):
            """Final usage+timings metadata event for the given pass, merging its
            usage/timings with the running cross-iteration accumulators. None when
            there is nothing to report."""
            _fu = _backfill_usage_from_timings(usage, timings) or {}
            _fp = _fu.get("prompt_tokens", 0)
            _tc = _fu.get("completion_tokens", 0) + _accumulated_completion_tokens
            if not (usage or timings or _accumulated_completion_tokens or finish_reason):
                return None
            _mt = dict(timings) if timings else {}
            if _accumulated_predicted_ms or _accumulated_predicted_n:
                _mt["predicted_ms"] = _mt.get("predicted_ms", 0) + _accumulated_predicted_ms
                _mt["predicted_n"] = _mt.get("predicted_n", 0) + _accumulated_predicted_n
                if _mt["predicted_ms"] > 0:
                    _mt["predicted_per_second"] = _mt["predicted_n"] / (
                        _mt["predicted_ms"] / 1000.0
                    )
            _usage = {
                "prompt_tokens": _fp,
                "completion_tokens": _tc,
                "total_tokens": _fp + _tc,
            }
            # Earlier passes' completions are already inside _fp, so the context is _fp + this pass only.
            if _accumulated_completion_tokens:
                _usage["context_tokens"] = _fp + int(_fu.get("completion_tokens") or 0)
            # Preserve KV-cache hit details (cached_tokens) so the tool path
            # reports them like the standard non-tool path does, not always 0.
            if _fu.get("prompt_tokens_details"):
                _usage["prompt_tokens_details"] = _fu["prompt_tokens_details"]
            return {
                "type": "metadata",
                "usage": _usage,
                "timings": _mt,
                "finish_reason": finish_reason,
            }

        def _flush_reasoning_and_buffer():
            """Close a live-streamed <think> block (or emit the buffered reasoning
            as one block if it never streamed), then append the held
            content_buffer to the cumulative display text."""
            nonlocal cumulative_display, in_thinking, _prov_entry, _buffer_in_display
            if in_thinking:
                cumulative_display += "</think>"
                in_thinking = False
                _prov_entry = None
            elif reasoning_accum:
                if (
                    reasoning_provenance is not None
                    and not cumulative_display
                    and not _suppress_visible_output
                ):
                    reasoning_provenance["wrapped"] = reasoning_provenance.get("wrapped", 0) + 1
                    reasoning_provenance.setdefault("wraps", []).append(
                        {"len": len(reasoning_accum)}
                    )
                cumulative_display += "<think>" + reasoning_accum + "</think>"
            cumulative_display += content_buffer
            # Not cleared: the callers measure len(content_buffer) right after to place the
            # live-args window. Recorded instead, so the cancel flush does not re-add it.
            _buffer_in_display = True

        def _cancelled_hold_text() -> str:
            """Display text the guards are still holding, which a cancel would drop.

            Mirrors the safetensors loop: a completed blocked markerless object keeps the
            chain guard true while its suffix is empty, so the whole object sits in
            ``content_buffer`` as ordinary display text the parser will never promote, and
            returning on cancellation lost the reply outright. The final strip removes
            genuinely promotable markup, so an aborted real call contributes only prose."""
            if _suppress_visible_output:
                return ""
            held = cumulative_display + ("" if _buffer_in_display else content_buffer)
            if not held:
                return ""
            cleaned = _strip_tool_markup(held, final = True, force = True)
            return cleaned if len(cleaned) > len(_last_emitted) else ""

        def _close_streamed_think() -> bool:
            """Close a live-streamed <think> before a tool call drains, so
            consumers without a reasoning extractor (Anthropic) get a balanced
            block. Returns True when the caller should yield the result."""
            nonlocal cumulative_display, in_thinking, _last_emitted, _prov_entry
            if not in_thinking:
                return False
            cumulative_display += "</think>"
            in_thinking = False
            _prov_entry = None
            if len(cumulative_display) > len(_last_emitted) and not _suppress_visible_output:
                _last_emitted = cumulative_display
                return True
            return False

        def _looks_like_enabled_bare_json(text: str, enabled_tool_names: set) -> bool:
            """True when ``text`` opens with an ENABLED markerless bare-JSON call; an ordinary JSON answer returns False."""
            probe = strip_llama3_leading_sentinels(text.lstrip())
            if not (probe.startswith("{") and ('"name"' in probe or '"function"' in probe)):
                return False
            return strip_leading_bare_json_call(probe, enabled_tool_names) != probe

        # Sanitized at construction, not at each gate: the controller authorizes execution,
        # and llama-server's structured delta.tool_calls path reaches prepare_call without
        # _enabled_tool_names at all. A tool dropped for unsafe markup is absent from the
        # prompt, so it must leave the controller too or it stays executable by name (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_tool_descriptions as _neutralize_tool_descriptions,
        )

        controller_tools = (
            []
            if tool_choice == "none"
            else _neutralize_tool_descriptions(tools, None, self.markup_profile)
        )
        tool_controller = ToolLoopController(
            tools = controller_tools,
            auto_heal_tool_calls = auto_heal_tool_calls,
            deduplicate_tool_calls = deduplicate_tool_calls,
            session_id = session_id,
            thread_id = thread_id,
        )

        def _tool_succeeded(tool_name: str) -> bool:
            key_prefix = f"{tool_name}:"
            return any(
                record.executed and not record.is_error and record.key.startswith(key_prefix)
                for record in tool_controller.history
            )

        _MAX_BUFFER_CHARS = 32
        # Hold a leading ``{`` well past the 32-char XML cap until it balances (mirrors safetensors).
        _MAX_BARE_JSON_BUFFER = 16384
        _append_budget_exhausted_nudge = True
        # RAG: cap knowledge-base searches per assistant turn. The controller is
        # tool-agnostic, so this gate stays in the loop.
        _kb_search_count = 0

        # ── Re-prompt on plan-without-action ─────────────────
        # Model describes intent without calling a tool: re-prompt once. A
        # direct answer ("4", "Hello!") won't match. Pattern shared with the
        # safetensors loop (tool_call_parser.INTENT_SIGNAL).
        _reprompt_count = 0
        # Budgeted apart from _reprompt_count so a pre-tool nudge can't spend it.
        _post_tool_reprompts = 0
        # A browser task is many steps, each a chance to announce the next one and stop: there the
        # budget is one nudge per tool turn, as in the external loop, not one per request.
        _browser_turns = any(
            ((tool.get("function") or {}).get("name") in BROWSER_TOOL_NAMES)
            for tool in (tools or [])
        )
        # Text that triggered the last nudge; if the retry restates it, stop.
        _last_reprompt_text = ""
        # Gates ``max_tool_iterations`` on real tool turns (not the enlarged range) so reserved
        # re-prompt slots don't extend the budget. Mirrors the safetensors guard.
        _tool_iters_done = 0
        _forced_tool_call_pending = False
        # Turns that ended INSIDE their own thought, having produced nothing to show.
        # `_finalize_reasoning_only_cumulative` deliberately refuses to promote a truncated
        # thought as the answer -- correctly, it is not one -- which left the client with an
        # empty message and the user with no reply at all. Observed on a 4096 window: the
        # model spent 2301 tokens reasoning, exactly the room the prompt left, and stopped.
        # Compaction cannot help, because there is no tool result to compact; the whole
        # window went on thinking. So carry a short note of where the thought reached into a
        # fresh turn with thinking OFF, which is the one thing that guarantees the next turn
        # produces something visible.
        _length_continuations = 0
        _effective_enable_thinking = enable_thinking
        # Dropped alongside it, because for the `reasoning_effort` style an explicit
        # effort WINS over enable_thinking: `_request_reasoning_kwargs` returns the
        # caller's "high" and never reaches the enable_thinking branch, so turning
        # thinking off would re-run the turn that just spent the window thinking.
        # Cleared rather than pinned to a level, so the existing thinking-off
        # convention for this style applies -- "low" for the models that cannot
        # actually disable reasoning.
        _effective_reasoning_effort = reasoning_effort
        # Restored once a turn gets somewhere: it is turned on to stitch ONE answer
        # back together, not for the rest of the request.
        _caller_continue_final_message = continue_final_message
        # Single-element lists so the nested execution scope can rebind them.
        _last_tool_result_key: list = [None]
        _identical_result_runs: list = [0]
        _starved_call: list = [False]

        # Reserve extra iterations for re-prompts so they don't consume the
        # caller's tool-call budget; only when tool iterations are allowed.
        _extra = _MAX_REPROMPTS + 1 if max_tool_iterations > 0 else 0
        # Owned by this request and dropped with it: the sweep below re-reads every earlier
        # turn on each iteration, and the rewrite is a function of the text alone (#7066).
        from core.inference.chat_template_helpers import sweep_cache as _sweep_cache

        _markup_cache = _sweep_cache()

        def _evict_until_it_fits(
            candidate,
            tools,
            reasoning_kw,
            continue_flag = False,
            protect = None,
        ):
            """Drop older turns from a continuation candidate that is one eviction short.

            Only reached once the candidate has already been priced and found too large.
            The single-turn case the callers were written for genuinely has nothing to
            evict, but a multi-turn chat under `truncate_oldest` usually does, and
            refusing there ended the turn before the NEXT iteration's ordinary preflight
            could ever run -- so a recoverable continuation was abandoned.

            Deliberately the plain rolling fit with instruction pins, not the loop's full
            preflight: that one archives, recalls and moves this thread's sticky boundary,
            and none of that should happen a second time in the middle of one turn.
            Returns the evicted candidate, or None when there was nothing to evict.

            ``protect`` keeps the current turn intact while older history is evicted.
            """
            if context_overflow != "truncate_oldest" or not self._effective_context_length:
                return None
            try:
                _fitted, _truncation = _fit_with_instruction_pins(
                    list(candidate),
                    context_length = self._effective_context_length,
                    # The room the CONTINUATION needs, which is not the room the caller's
                    # cap asks for: a small `max_tokens` makes `prompt_budget` larger than
                    # the window minus the reply floor, so fitting to the caller's figure
                    # would evict and still leave the candidate unservable.
                    max_tokens = max(
                        int(max_tokens or 0), _reply_floor(self._effective_context_length)
                    ),
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            messages_without_unpriced_media(fitted),
                            _markup_cache,
                            self.markup_profile,
                        ),
                        None,
                        tools,
                        strict = True,
                        chat_template_kwargs = reasoning_kw,
                        continue_final_message = continue_flag,
                    ),
                    estimate_message = estimate_message_tokens_without_unpriced_media,
                    anchor_ids = (_rolling_anchor_ids | protect) if protect else _rolling_anchor_ids,
                )
            except Exception:
                logger.debug("continuation eviction: fit failed", exc_info = True)
                return None
            if not _truncation or not _truncation.get("dropped_messages"):
                return None
            logger.info(
                "Continuation did not fit; evicted %s older message(s) to try to keep it",
                _truncation.get("dropped_messages"),
            )
            # Whether that was ENOUGH is the caller's own fit check, re-run on this: the
            # rule for what counts as servable stays in one place.
            return _fitted

        def _loop_continuation_fits(
            candidate,
            tools,
            reasoning_kw,
            continue_flag = False,
        ) -> bool:
            """Whether the retry's own prompt still fits once the partial is in it.

            The final pass has the same guard. This path did not, and it is the one a
            request with tools enabled actually takes: a `length` stop that came from
            exhausting the window makes the retry prompt window-sized, and with a single
            user turn there is no older history to evict, so llama-server rejects it or
            leaves room for a stub. A count that cannot be taken is not a refusal.
            """
            if not self._effective_context_length:
                return True
            try:
                _tokens = self.count_chat_tokens(
                    neutralize_control_markup_in_messages(
                        messages_without_unpriced_media(candidate),
                        _markup_cache,
                        self.markup_profile,
                    ),
                    None,
                    tools,
                    strict = True,
                    chat_template_kwargs = reasoning_kw,
                    # An answer continuation goes out with `continue_final_message`, which
                    # renders WITHOUT a generation prompt and with the partial as the turn
                    # being extended. Counting it as an ordinary prompt prices a different
                    # request from the one about to be sent.
                    continue_final_message = continue_flag,
                )
            except Exception:
                logger.debug("in-loop continuation: prompt count failed", exc_info = True)
                return True
            return _tokens + _reply_floor(self._effective_context_length) <= (
                self._effective_context_length
            )

        def _loop_budget_left(spent_this_attempt: int) -> "Optional[int]":
            """What remains of a cap the CALLER set, or None when they set none.

            The in-loop payload rebuilds `max_tokens` from the caller's value on every
            iteration, so without this a request capped at 100 tokens ran two more
            100-token generations after its own cap stopped the first.
            """
            if max_tokens is None:
                return None
            window = self._effective_context_length or 0
            if window and max_tokens >= window:
                return None
            return max(0, max_tokens - _accumulated_completion_tokens - spent_this_attempt)

        # A forced function applies until the model produces it; execution or denial resolves it.
        _forced_choice_resolved = _auto_satisfies_forced_choice
        # A continuation re-enters this loop, so without a credit it is charged to the
        # caller's TOOL budget. At max_tool_iterations=1, three plan-without-action
        # re-prompts and two truncated reasoning turns exhaust the loop before the model
        # ever issues its call, and the tool-free final pass then cannot perform the
        # action that was asked for -- the turn fails having spent no tool iteration at
        # all. Granted as they happen rather than reserved up front, so a request that
        # never stalls keeps exactly the bound it has always had.
        # Set by a continuation to hand the NEXT iteration what is left of the caller's
        # output cap, then cleared as it is spent.
        _continuation_max_tokens: Optional[int] = None
        _continuation_credits = 0
        _MAX_CONTINUATION_CREDITS = _MAX_LENGTH_CONTINUATIONS * max(1, max_tool_iterations)
        # A fixed seed keeps the prompt cache once this run has sent a request (#10698).
        _prompt_sent = False
        # Resent on the final pass with tool_choice "none": tools head the system prompt.
        _last_sent_tools = None
        iteration = -1
        # Outside the loop: the cap is across the whole run, and a per-iteration list
        # would only ever see the current batch, leaving every earlier iteration's
        # images in the conversation untrimmed.
        # Seeded with what promotion already put in the conversation, not empty. The cap
        # is across the CONVERSATION, so a resumed chat whose history already carries the
        # allowance would otherwise get a second one for this run and send both.
        loop_mcp_image_parts: list = list(replayed_image_parts)
        while True:
            iteration += 1
            while _model_note_events:
                yield _model_note_events.pop(0)
            # Here rather than at each append: six sites grow the conversation and all of
            # them pass through this one point before the cache must hold the result.
            if on_conversation_grew is not None:
                try:
                    on_conversation_grew(conversation)
                except Exception:  # accounting must never break a run in progress
                    logger.debug("tool loop recost failed", exc_info = True)
            if iteration >= max_tool_iterations + _extra + _continuation_credits:
                break
            if cancel_event is not None and cancel_event.is_set():
                return
            # Whether this turn ran a tool; a no-op-only turn stays False and doesn't consume budget.
            _turn_executed_real_tool = False
            # Set when a reasoning retry was refused because the CALLER's allowance was
            # spent rather than the window. Per iteration: it describes this turn only.
            _reasoning_cap_spent = False

            active_tools = tool_controller.active_tools()
            if not active_tools:
                _append_budget_exhausted_nudge = False
                break
            from core.inference.chat_template_helpers import (
                neutralize_tool_descriptions,
                reconciled_tool_choice,
            )

            # An MCP server's description and inputSchema are remote text the template renders
            # into the system turn (#7066). Computed above the gate: a tool dropped for unsafe
            # markup is absent from the prompt and must be absent from what we EXECUTE too,
            # else the model names it and the raw gate lets the call through.
            safe_tools = neutralize_tool_descriptions(
                active_tools, _markup_cache, self.markup_profile
            )
            requested_choice = "auto" if tool_choice is None else tool_choice
            if _forced_choice_resolved and requested_choice not in ("auto", "none"):
                requested_choice = "auto"
            requested_choice = reconciled_tool_choice(requested_choice, tools, safe_tools) or "auto"
            forced_name = forced_tool_name(requested_choice)
            if forced_name:
                matching_tools = [
                    tool
                    for tool in safe_tools
                    if (tool.get("function") or {}).get("name") == forced_name
                ]
                if matching_tools:
                    safe_tools = matching_tools
                    requested_choice = "required"
            # Gate the markerless bare-JSON form on enabled names so an ordinary JSON answer isn't misread as a call.
            _enabled_tool_names = {
                (tool.get("function") or {}).get("name")
                for tool in safe_tools
                if (tool.get("function") or {}).get("name")
            } | (set() if forced_name else _loadable_mcp_names)
            # Shared signal tuple so GGUF BUFFERING wakes on every format the parser knows (like safetensors).
            _tool_xml_signals = _SHARED_TOOL_XML_SIGNALS

            # Build payload -- stream: True so we detect tool signals
            # in the first 1-2 chunks without a non-streaming penalty.
            from core.inference.chat_template_helpers import (
                append_assistant_turn,
                neutralize_control_markup,
                neutralize_control_markup_in_messages,
            )

            _reasoning_kw = self._request_reasoning_kwargs(
                _effective_enable_thinking, _effective_reasoning_effort, preserve_thinking
            )
            # What THIS request may generate, which on a continuation is the remainder of
            # the caller's cap rather than the whole of it. Read before the payload
            # consumes it below, because every sizing decision this iteration makes runs
            # first: `prompt_budget` shrinks as `max_tokens` grows, so pricing a
            # 600-token continuation as if it could still emit 1000 evicts history the
            # request never needed to lose.
            _iteration_max_tokens = (
                _continuation_max_tokens if _continuation_max_tokens is not None else max_tokens
            )
            _preflight_context_length = None
            _preflight_succeeded = False
            if context_overflow == "truncate_oldest" and self._effective_context_length:
                _preflight_context_length = self._effective_context_length
                try:
                    _before_fit = conversation
                    _iteration_can_reset = _can_reset_epoch(
                        thread_id,
                        _backend_supports_tools(self),
                        tools_withheld = _memory_tool_withheld(thread_id, tools),
                    )
                    # The saved boundary and /compact both apply once, to the loop's first fit.
                    _iteration_force_reset = compact_context and not _sticky_boundary_applied
                    _iteration_sticky, _iteration_sticky_is_checkpoint, _iteration_sticky_manual = (
                        (0, True, False)
                        if _sticky_boundary_applied
                        else _sticky_compaction_details(
                            thread_id,
                            _request_branch,
                            context_policy = context_policy,
                            can_reset = _iteration_can_reset,
                            compaction_headroom_ratio = compaction_headroom_ratio,
                        )
                    )
                    conversation, truncation = _fit_with_instruction_pins(
                        conversation,
                        context_length = self._effective_context_length,
                        max_tokens = _iteration_max_tokens,
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                messages_without_unpriced_media(fitted),
                                _markup_cache,
                                self.markup_profile,
                            ),
                            None,
                            safe_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = bool(
                                continue_final_message
                                and trailing_assistant_resume_kind(fitted) is not None
                            ),
                            should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                        ),
                        estimate_message = estimate_message_tokens_without_unpriced_media,
                        anchor_ids = _rolling_anchor_ids,
                        keeps_boundary = _keeps_compaction_boundary(thread_id),
                        can_reset = _iteration_can_reset,
                        recall_offered = "search_conversation" in (_enabled_tool_names or ()),
                        reserve_tokens = _conversation_recall_reserve(thread_id),
                        sticky_dropped = _iteration_sticky,
                        sticky_is_checkpoint = _iteration_sticky_is_checkpoint,
                        sticky_manual = _iteration_sticky_manual,
                        force_reset = _iteration_force_reset,
                        **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                    )
                    # Accounted for in this request now, whatever the fit decided.
                    _sticky_boundary_applied = True
                    if truncation and truncation.get("fits"):
                        # Before the recall injection, so both sides describe the same
                        # messages, priced by the fit with the catalogue and template.
                        _prompt_token_offset = int(
                            truncation.get("prompt_tokens_after") or 0
                        ) - estimate_messages_tokens_without_unpriced_media(conversation)
                    if truncation:
                        _recalled = _archive_and_recall(
                            conversation,
                            _before_fit,
                            branch_messages = _extend_live_branch(_before_fit),
                            thread_id = thread_id,
                            # Tool style forges an assistant tool_call plus a tool result
                            # for search_conversation, safe only when the request
                            # advertises it. On the FIRST compaction the archive did not
                            # exist at tool-selection time, so it is absent, and a tool
                            # role with no catalogue entry breaks strict templates.
                            style = (
                                "tool" if "search_conversation" in _enabled_tool_names else "inline"
                            ),
                            # Checkpoint mode recalls once, on the turn that reset; rolling has
                            # no such key and recalls on every turn that evicted anything.
                            force_recall = bool(truncation.get("checkpoint_started", True)),
                            # A rescued refusal may evict messages; archive them without recall.
                            recall_done = _conversation_recall_done or not truncation["fits"],
                            # Leave room for the answer appended on the next iteration.
                            recall_budget_tokens = _retrieval_budget(
                                self._effective_context_length,
                                # As above: the cap this iteration will actually send.
                                _iteration_max_tokens,
                                truncation.get("prompt_tokens_after") or 0,
                                reply_returns = True,
                            ),
                            count_tokens = lambda fitted: self.count_chat_tokens(
                                neutralize_control_markup_in_messages(
                                    messages_without_unpriced_media(fitted),
                                    _markup_cache,
                                    self.markup_profile,
                                ),
                                None,
                                safe_tools,
                                strict = True,
                                chat_template_kwargs = _reasoning_kw,
                            ),
                        )
                        conversation = _recalled["conversation"]
                        truncation = {**truncation, **_recalled["counts"]}
                        if _recalled["recalled"]:
                            _conversation_recall_done = True
                            # Anchor exactly what the injection added, so a later refit
                            # cannot evict what we just paid to bring back. NOT the last
                            # two messages: inline recall appends nothing and rewrites the
                            # latest user turn in place, so that slice also pinned a whole
                            # eviction unit the fit was entitled to drop.
                            for _message in _recalled["anchored"]:
                                _rolling_anchor_ids.add(id(_message))
                            for _ev in _recalled["events"]:
                                yield _ev
                    if truncation and _records_boundary(truncation):
                        truncation = {
                            **truncation,
                            **_boundary_metadata(
                                conversation, _request_branch, compaction_headroom_ratio
                            ),
                        }
                    # `fits` False too: it carries the does-not-fit diagnosis.
                    if truncation:
                        yield {"type": "context_truncated", **truncation}
                    _preflight_succeeded = True
                except Exception as exc:
                    logger.warning("Could not preflight the rolling context window: %s", exc)

            payload = {
                # Re-run every iteration: tool results land in ``conversation`` as the
                # loop goes, and a forged turn in one would render for real (#7066).
                "messages": neutralize_control_markup_in_messages(
                    conversation, _markup_cache, self.markup_profile
                ),
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }

            # Progress events feed the first-token deadline; timings stay opt-in.
            payload["return_progress"] = True
            if on_decode_slot is not None:
                payload["verbose"] = True
                payload["response_fields"] = ["id_slot"]
            if _perf_callback_wants_timings(perf_callback):
                payload["timings_per_token"] = True
            if logit_bias:
                payload["logit_bias"] = logit_bias
            # As in the passthrough builder: if every name carried markup the catalog is
            # now empty, and "tools": [] would still advertise tool use.
            if safe_tools:
                payload["tools"] = llama_grammar_tools(safe_tools)
                payload["tool_choice"] = requested_choice
                _last_sent_tools = safe_tools
            if _reasoning_kw is not None:
                payload["chat_template_kwargs"] = _reasoning_kw
            if thinking_budget_tokens is not None:
                payload["thinking_budget_tokens"] = thinking_budget_tokens
            # Re-checked per iteration: once a tool result is appended the partial is
            # no longer trailing, so later turns are normal.
            _resume_kind = (
                trailing_assistant_resume_kind(conversation) if continue_final_message else None
            )
            if _resume_kind is not None:
                payload["continue_final_message"] = True
                payload["add_generation_prompt"] = False
            _resumed_reasoning = (
                conversation[-1]["reasoning_content"] if _resume_kind == "reasoning_content" else ""
            )
            payload["max_tokens"] = (
                max_tokens
                if max_tokens is not None
                else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
            )
            # A continuation is the SAME reply carried across two requests, so it spends
            # the caller's output budget once, not once per attempt. Rebuilt from the
            # caller's value above, this payload would hand a 100-token request another
            # full 100 tokens on every retry.
            if _continuation_max_tokens is not None:
                payload["max_tokens"] = _continuation_max_tokens
                _continuation_max_tokens = None
            if stop:
                payload["stop"] = stop
            _apply_seeded_llama_request(payload, seed, reuse_prompt_cache = _prompt_sent)
            _prompt_sent = True

            _respawn_truncations: list[dict] = []

            def _refit_iteration_after_respawn() -> None:
                nonlocal conversation
                _before_respawn_fit = conversation
                if (
                    _preflight_context_length is None
                    or not self._effective_context_length
                    or (
                        _preflight_succeeded
                        and _preflight_context_length == self._effective_context_length
                    )
                ):
                    return
                if max_tokens is None:
                    payload["max_tokens"] = self._effective_context_length
                try:
                    conversation, truncation = _fit_with_instruction_pins(
                        conversation,
                        context_length = self._effective_context_length,
                        max_tokens = _iteration_max_tokens,
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                messages_without_unpriced_media(fitted),
                                _markup_cache,
                                self.markup_profile,
                            ),
                            None,
                            safe_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = bool(
                                continue_final_message
                                and trailing_assistant_resume_kind(fitted) is not None
                            ),
                            should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                        ),
                        estimate_message = estimate_message_tokens_without_unpriced_media,
                        anchor_ids = _rolling_anchor_ids,
                        keeps_boundary = _keeps_compaction_boundary(thread_id),
                        can_reset = _can_reset_epoch(
                            thread_id,
                            _backend_supports_tools(self),
                            tools_withheld = _memory_tool_withheld(thread_id, tools),
                        ),
                        recall_offered = "search_conversation" in (_enabled_tool_names or ()),
                        **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                    )
                    # Recorded here, not left to the forwarding below. That list is
                    # drained from INSIDE the reopened stream, so a replacement server
                    # that came back with a smaller window and then refused this prompt
                    # raises at the door and the refusal reaches `_friendly_error` as
                    # nothing at all -- which then advises shortening a conversation
                    # eviction has already emptied. Idempotent, and safe on a fit:
                    # `record_fit` clears whatever the last one left.
                    from core.inference import context_refusal  # noqa: PLC0415

                    context_refusal.record_fit(truncation)
                    if truncation:
                        # Archive only. The replacement fit reserves no recall room, and a
                        # rescued refusal may still have evicted messages.
                        truncation.update(
                            _archive_and_recall(
                                conversation,
                                _before_respawn_fit,
                                thread_id = thread_id,
                                style = "inline",
                                recall_done = True,
                                branch_messages = _extend_live_branch(conversation),
                            )["counts"]
                        )
                    payload["messages"] = neutralize_control_markup_in_messages(
                        conversation, _markup_cache, self.markup_profile
                    )
                    if truncation:
                        if _records_boundary(truncation):
                            truncation.update(
                                _boundary_metadata(
                                    conversation, _request_branch, compaction_headroom_ratio
                                )
                            )
                        # Report every shortened retry, including a rescued refusal.
                        _respawn_truncations.append(truncation)
                except Exception as exc:
                    logger.warning("Could not refit rolling context after respawn: %s", exc)
                    raise

            try:
                # ── Speculative buffer state machine ──────────────────
                # BUFFERING: accumulate content, check for tool signals
                # STREAMING: no tool detected, yield tokens to caller
                # DRAINING:  tool signal found, silently consume rest
                _S_BUFFERING = 0
                _S_STREAMING = 1
                _S_DRAINING = 2

                detect_state = _S_BUFFERING
                content_buffer = ""  # Raw content held during BUFFERING
                # Whether content_buffer has already been added to cumulative_display.
                _buffer_in_display = False
                content_accum = ""  # All content tokens (for tool parsing)
                reasoning_accum = ""
                _prov_entry = None
                # Time each reasoning pass so final answers can replace tool timing.
                _reasoning_started_at = None
                _reasoning_summary_emitted = False
                _deferred_reasoning_summary = None
                cumulative_display = ""  # Cumulative yielded text (with <think>)
                # A new buffer, not a continuation of the previous iteration's.
                _streaming_stripper.reset()
                in_thinking = False
                has_content_tokens = False
                tool_calls_acc = {}  # Structured delta.tool_calls fragments
                # accumulator key each index maps to now, (index, call_id) after a fork
                tool_calls_open_key: dict[Any, Any] = {}
                tool_calls_synthetic_ids: set[Any] = set()  # slots still on a placeholder id
                has_structured_tc = False
                _iter_usage = None
                _iter_timings = None
                _iter_finish_reason = None
                _stream_done = False
                _last_emitted = ""
                # Provisional tool_start cards already shown, keyed by tool_call_id.
                provisional_started_tool_calls: dict[str, str] = {}
                resolved_provisional_tool_call_ids: set[str] = set()
                _suppress_visible_output = _forced_tool_call_pending
                # Cards that already got their first tool_args event; later
                # fragments stream individually so a big payload isn't a dead spinner.
                arg_streamed_tool_call_ids: set[str] = set()
                # TEXT tool-call path: a committed call's raw text streams to a
                # provisional card under the parser's first-call id ("call_0"), so
                # the final tool_start reconciles in place.
                _text_args_call_start = -1
                _text_args_streamed_upto = -1
                _text_args_id = ""
                _text_args_name = ""
                _confirm_gated_iteration = bool(confirm_tool_calls) and not bypass_permissions

                # Reclaim reply room before generating, not only before running a tool.
                #
                # The pre-execution gate compacts only when the next prompt would not FIT.
                # A prompt can fit and still leave nothing to answer in, and that is the
                # more common ending: observed on a 4096 window where every tool call was
                # servable, none was refused, the file was written -- and the turn died on
                # `finish_reason: length` with the model still thinking, because the prompt
                # had eaten the room its answer needed. Compaction is exactly the lever for
                # that and nothing was asking it to run.
                #
                # `prompt_budget` is the line here, not the window: it already encodes how
                # much of the window a reply is owed. Over it means tight, not fatal, so
                # this compacts and never refuses -- a turn that fits is always sent.
                if self._effective_context_length and conversation:
                    # Same remainder the preflight used: this check decides whether to
                    # compact, and pricing a continuation as if it had the whole cap
                    # compacts a turn that had room.
                    _reply_target = prompt_budget(
                        self._effective_context_length, _iteration_max_tokens
                    )
                    if (
                        estimate_messages_tokens_dense(
                            messages_without_unpriced_media(conversation)
                        )
                        > 0.7 * _reply_target
                    ):
                        try:
                            _prompt_now = self.count_chat_tokens(
                                neutralize_control_markup_in_messages(
                                    messages_without_unpriced_media(conversation),
                                    _markup_cache,
                                    self.markup_profile,
                                ),
                                None,
                                safe_tools,
                                strict = True,
                                chat_template_kwargs = _reasoning_kw,
                            )
                        except Exception:
                            logger.debug("reply-room fit: prompt count failed", exc_info = True)
                            _prompt_now = None
                        if _prompt_now is not None and _prompt_now > _reply_target:
                            _roomy, _n_roomy = compact_completed_tool_arguments(
                                conversation, protect_last = 1
                            )
                            if _n_roomy:
                                conversation[:] = _roomy
                                payload["messages"] = neutralize_control_markup_in_messages(
                                    conversation, _markup_cache, self.markup_profile
                                )
                                logger.info(
                                    "Compacted %d completed tool call(s) for reply room: "
                                    "prompt was %d tokens against a %d-token budget",
                                    _n_roomy,
                                    _prompt_now,
                                    _reply_target,
                                )

                with self._open_chat_stream_with_respawn_retry(
                    payload,
                    cancel_event,
                    on_respawn = _refit_iteration_after_respawn,
                ) as (
                    response,
                    first_token_deadline,
                ):
                    for truncation in _respawn_truncations:
                        yield {"type": "context_truncated", **truncation}
                    raw_buf = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response,
                        cancel_event,
                        first_token_deadline = first_token_deadline,
                    ):
                        raw_buf += raw_chunk
                        while "\n" in raw_buf:
                            line, raw_buf = raw_buf.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                # Flush thinking state for STREAMING
                                if detect_state == _S_STREAMING and in_thinking:
                                    if has_content_tokens:
                                        cumulative_display += "</think>"
                                        _prov_entry = None
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": _strip_tool_markup(
                                                    cumulative_display,
                                                    final = True,
                                                ),
                                            }
                                    else:
                                        cumulative_display = _finalize_reasoning_only_cumulative(
                                            cumulative_display,
                                            reasoning_accum,
                                            _iter_finish_reason,
                                            promote_reasoning_only,
                                            _resumed_reasoning,
                                        )
                                        _prov_entry = None
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                if on_decode_slot is not None:
                                    slot = self.decode_slot_from_chunk(chunk_data)
                                    if slot is not None:
                                        on_decode_slot(self.base_url, slot)

                                _report_live_llama_timings(perf_callback, chunk_data)
                                _ct = chunk_data.get("timings")
                                if _ct:
                                    _iter_timings = _ct
                                _cu = chunk_data.get("usage")
                                if _cu:
                                    _iter_usage = _cu

                                # See the note on the first stream loop: an error chunk has
                                # no choices, so `continue` below would drop it silently.
                                _stream_error = stream_error_from_chunk(chunk_data)
                                if _stream_error is not None:
                                    raise _stream_error
                                choices = chunk_data.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _iter_finish_reason = _fr

                                # ── Structured tool_calls ──
                                tc_deltas = delta.get("tool_calls")
                                if tc_deltas:
                                    # Preserve any visible preface before draining
                                    # the structured tool call.
                                    has_structured_tc = True
                                    detect_state = _S_DRAINING
                                    # Close the reasoning prefix before the tool card
                                    # (mirrors the is_match path).
                                    if _close_streamed_think():
                                        yield {"type": "content", "text": cumulative_display}
                                    for tc_d in tc_deltas:
                                        idx = tc_d.get("index", 0)
                                        tc_id = tc_d.get("id")
                                        # a second call at a reused index forks its own slot; bare fragments follow whichever call owns the index now
                                        acc_key = tool_calls_open_key.get(idx, idx)
                                        if isinstance(tc_id, str) and tc_id:
                                            open_id = tool_calls_acc.get(acc_key, {}).get("id")
                                            # a synthetic placeholder is not a rival call, so a first real id still lands on the slot that is open
                                            if (
                                                open_id
                                                and acc_key not in tool_calls_synthetic_ids
                                                and open_id != tc_id
                                            ):
                                                first_id = tool_calls_acc.get(idx, {}).get("id")
                                                acc_key = idx if first_id == tc_id else (idx, tc_id)
                                        tool_calls_open_key[idx] = acc_key
                                        if acc_key not in tool_calls_acc:
                                            tool_calls_acc[acc_key] = {
                                                "id": tc_id or f"call_{idx}",
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                            if not tc_id:
                                                tool_calls_synthetic_ids.add(acc_key)
                                        elif tc_id:
                                            # Update ID if a real one
                                            # arrives on a later delta.
                                            tool_calls_acc[acc_key]["id"] = tc_id
                                            tool_calls_synthetic_ids.discard(acc_key)
                                        func = tc_d.get("function", {})
                                        if func.get("name"):
                                            tool_calls_acc[acc_key]["function"]["name"] += func[
                                                "name"
                                            ]
                                        if func.get("arguments"):
                                            tool_calls_acc[acc_key]["function"]["arguments"] += (
                                                func["arguments"]
                                            )
                                        current_name = tool_calls_acc[acc_key]["function"].get(
                                            "name", ""
                                        )
                                        fallback_id = f"call_{idx}"
                                        current_id = tool_calls_acc[acc_key].get("id", fallback_id)
                                        is_first_accumulated_call = (
                                            next(iter(tool_calls_acc)) == acc_key
                                        )
                                        already_started = (
                                            current_id in provisional_started_tool_calls
                                        )
                                        # Empty/synthetic ids cannot reconcile with real starts.
                                        has_real_id = bool(current_id) and current_id != fallback_id
                                        # Show one early card per eligible streamed tool call.
                                        _is_completed_one_shot = (
                                            current_name == "render_html"
                                            and _tool_succeeded("render_html")
                                        )
                                        # render_html is one-shot.
                                        _one_shot_already_provisional = (
                                            current_name == "render_html"
                                            and "render_html"
                                            in provisional_started_tool_calls.values()
                                        )
                                        # Later parallel cards only reconcile when parallel use is enabled.
                                        # In auto mode an always-safe tool (render_html) never
                                        # prompts, so it must stream its early card too; mirror
                                        # that here instead of gating on the raw confirm flag.
                                        _confirm_gated = (
                                            tool_call_may_prompt(
                                                confirm_tool_calls = bool(confirm_tool_calls),
                                                bypass_permissions = bypass_permissions,
                                                permission_mode = permission_mode,
                                                name = current_name,
                                                sandbox_level = sandbox_level,
                                            )
                                            # A text-preview card still streams while gated;
                                            # hiding it blanks the chat.
                                            and not has_text_only_provisional_card(current_name)
                                        )
                                        # Keep small-argument tools on the normal path.
                                        _args_len = len(
                                            tool_calls_acc[acc_key]["function"].get("arguments", "")
                                        )
                                        _payload_is_large = (
                                            current_name == "render_html"
                                            or _args_len >= _PROVISIONAL_ARGS_MIN_CHARS
                                        )
                                        if (
                                            current_name
                                            and (
                                                is_first_accumulated_call
                                                or not disable_parallel_tool_use
                                            )
                                            and has_real_id
                                            and not already_started
                                            and not _is_completed_one_shot
                                            and not _one_shot_already_provisional
                                            and not _confirm_gated
                                            and _payload_is_large
                                            and any(
                                                (tool.get("function") or {}).get("name")
                                                == current_name
                                                for tool in safe_tools
                                            )
                                        ):
                                            provisional_started_tool_calls[current_id] = (
                                                current_name
                                            )
                                            yield {
                                                "type": "tool_start",
                                                "tool_name": current_name,
                                                "tool_call_id": current_id,
                                                "arguments": {},
                                                "provenance": provisional_tool_provenance(
                                                    current_name
                                                ),
                                            }
                                        # Stream argument text so the UI shows the code being
                                        # written: first event the backlog, later the fragment.
                                        # Display only; accumulator untouched.
                                        if current_id in provisional_started_tool_calls:
                                            if current_id not in arg_streamed_tool_call_ids:
                                                arg_streamed_tool_call_ids.add(current_id)
                                                _args_backlog = tool_calls_acc[acc_key][
                                                    "function"
                                                ].get("arguments", "")
                                                if _args_backlog:
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": current_id,
                                                        "tool_name": current_name,
                                                        "text": _args_backlog,
                                                    }
                                            elif func.get("arguments"):
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": current_id,
                                                    "tool_name": current_name,
                                                    "text": func["arguments"],
                                                }
                                    continue

                                # ── Reasoning tokens ──
                                # Stream live except while DRAINING: reasoning is
                                # orthogonal to tool detection (content_buffer
                                # only), and the route resets prev_text on
                                # tool_start, so the <think> block stays a
                                # monotonic prefix like the no-tool path.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if _reasoning_started_at is None:
                                        _reasoning_started_at = time.monotonic()
                                    reasoning_accum += reasoning
                                    if detect_state != _S_DRAINING:
                                        if not in_thinking:
                                            # Suppressed (forced-retry) iterations
                                            # yield no content events, so recording
                                            # their wraps would desync the ledger
                                            # from the emitted stream.
                                            if (
                                                reasoning_provenance is not None
                                                and not cumulative_display
                                                and not _suppress_visible_output
                                            ):
                                                reasoning_provenance["wrapped"] = (
                                                    reasoning_provenance.get("wrapped", 0) + 1
                                                )
                                                _prov_entry = {"len": 0}
                                                reasoning_provenance.setdefault("wraps", []).append(
                                                    _prov_entry
                                                )
                                            cumulative_display += "<think>"
                                            in_thinking = True
                                        cumulative_display += reasoning
                                        if _prov_entry is not None:
                                            _prov_entry["len"] += len(reasoning)
                                        if not _suppress_visible_output:
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }

                                # ── Content tokens ──
                                token = delta.get("content", "")
                                if token:
                                    # First answer token ends reasoning.
                                    if (
                                        _reasoning_started_at is not None
                                        and not _reasoning_summary_emitted
                                    ):
                                        _reasoning_summary_emitted = True
                                        _summary = _reasoning_summary_event(_reasoning_started_at)
                                        if _suppress_visible_output:
                                            _deferred_reasoning_summary = _summary
                                        else:
                                            yield _summary
                                    has_content_tokens = True
                                    content_accum += token

                                    if detect_state == _S_DRAINING:
                                        # Accumulate silently for parsing, but stream the drained
                                        # TEXT call to a provisional card. Gated on an enabled-name
                                        # sniff + size floor so prose/small calls spawn no pane; id
                                        # matches the first call so the final tool_start reconciles.
                                        if not has_structured_tc and _text_args_call_start >= 0:
                                            if not _text_args_id:
                                                _call_text = content_accum[_text_args_call_start:]
                                                _sniffed = _sniff_text_tool_name(
                                                    _call_text, _enabled_tool_names
                                                )
                                                # Structured-path rule: gated calls
                                                # stream only from a text-preview card.
                                                if (
                                                    _sniffed
                                                    and not (
                                                        _confirm_gated_iteration
                                                        and (
                                                            permission_mode != "off"
                                                            or tool_call_may_prompt(
                                                                confirm_tool_calls = True,
                                                                bypass_permissions = False,
                                                                permission_mode = "off",
                                                                name = _sniffed,
                                                                sandbox_level = sandbox_level,
                                                            )
                                                        )
                                                        and not has_text_only_provisional_card(
                                                            _sniffed
                                                        )
                                                    )
                                                    and (
                                                        _sniffed == "render_html"
                                                        or len(_call_text)
                                                        >= _PROVISIONAL_ARGS_MIN_CHARS
                                                    )
                                                ):
                                                    _text_args_id = "call_0"
                                                    _text_args_name = _sniffed
                                                    if (
                                                        _text_args_id
                                                        not in provisional_started_tool_calls
                                                    ):
                                                        provisional_started_tool_calls[
                                                            _text_args_id
                                                        ] = _sniffed
                                                        yield {
                                                            "type": "tool_start",
                                                            "tool_name": _sniffed,
                                                            "tool_call_id": _text_args_id,
                                                            "arguments": {},
                                                            "provenance": provisional_tool_provenance(
                                                                _sniffed
                                                            ),
                                                        }
                                                    yield {
                                                        "type": "tool_args",
                                                        "tool_call_id": _text_args_id,
                                                        "tool_name": _sniffed,
                                                        "text": _call_text,
                                                    }
                                                    _text_args_streamed_upto = len(content_accum)
                                            elif len(content_accum) > _text_args_streamed_upto:
                                                yield {
                                                    "type": "tool_args",
                                                    "tool_call_id": _text_args_id,
                                                    "tool_name": _text_args_name,
                                                    "text": content_accum[
                                                        _text_args_streamed_upto:
                                                    ],
                                                }
                                                _text_args_streamed_upto = len(content_accum)

                                    elif detect_state == _S_STREAMING:
                                        if in_thinking:
                                            cumulative_display += "</think>"
                                            in_thinking = False
                                            _prov_entry = None
                                        cumulative_display += token
                                        cleaned = _strip_tool_markup_streaming(cumulative_display)
                                        # Hold a trailing bare active-tool-name (split rehearsal)
                                        # until [ARGS] arrives; released by later prose or stream end.
                                        _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                        _emit = (
                                            cleaned[: len(cleaned) - _hold] if _hold else cleaned
                                        )
                                        if len(_emit) > len(_last_emitted):
                                            _last_emitted = _emit
                                            if not _suppress_visible_output:
                                                yield {
                                                    "type": "content",
                                                    "text": _emit,
                                                }

                                    elif detect_state == _S_BUFFERING:
                                        content_buffer += token
                                        stripped_buf = content_buffer.lstrip()
                                        if not stripped_buf:
                                            continue

                                        # Bracket tags arrive mid-buffer, so substring-check too;
                                        # ``[ARGS]`` counts only as a regex-matched NAME[ARGS].
                                        is_prefix = False
                                        is_match = False
                                        for sig in _tool_xml_signals:
                                            if stripped_buf.startswith(sig):
                                                is_match = True
                                                break
                                            if sig.startswith(stripped_buf):
                                                is_prefix = True
                                                break
                                            if sig == "[ARGS]":
                                                # Active NAME[ARGS] only; inactive-name prose
                                                # is gated out, not drained/parsed.
                                                if (
                                                    _gguf_rehearsal_signal_pos(
                                                        stripped_buf, _detect_tools
                                                    )
                                                    >= 0
                                                ):
                                                    is_match = True
                                                    break
                                            elif sig.startswith("[") and sig in stripped_buf:
                                                is_match = True
                                                break

                                        # Split rehearsal: hold the bare name until
                                        # its [ARGS] arrives and matches above.
                                        is_rehearsal_prefix = False
                                        if (
                                            not is_match
                                            and not is_prefix
                                            and _is_rehearsal_prefix(stripped_buf, _detect_tools)
                                        ):
                                            is_prefix = True
                                            is_rehearsal_prefix = True

                                        # Signal-less call shapes (mirror the safetensors
                                        # loop): Llama-3.2 bare {"name":..} and Gemma
                                        # call:NAME{...} would otherwise stream raw.
                                        _hold_buffer = False
                                        # Whole buffer is the call (no visible prefix) -- drain silently.
                                        _drain_silently = False
                                        if not is_match and not is_prefix:
                                            _bare = strip_llama3_leading_sentinels(stripped_buf)
                                            if _bare.startswith("{"):
                                                if _balanced_brace_end(_bare, 0) is None:
                                                    if len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    elif _looks_like_enabled_bare_json(
                                                        _bare, _enabled_tool_names
                                                    ):
                                                        # Oversized still-open enabled call: drain
                                                        # rather than leak; a giant ordinary JSON
                                                        # answer still streams.
                                                        _drain_silently = True
                                                elif self._parse_tool_calls_from_text(
                                                    content_buffer,
                                                    allow_incomplete = auto_heal_tool_calls,
                                                    enabled_tool_names = _enabled_tool_names,
                                                ):
                                                    _drain_silently = True
                                                elif blocked_bare_json_chain_may_continue(
                                                    content_buffer, _enabled_tool_names
                                                ):
                                                    if len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    else:
                                                        # Bounded buffer: fail closed rather
                                                        # than expose text a later peer
                                                        # could make executable.
                                                        _drain_silently = True
                                            elif (
                                                "call:".startswith(stripped_buf)
                                                or _GEMMA_BARE_TC_PREFIX_RE.match(stripped_buf)
                                                is not None
                                                or _gemma_lead_promotable(stripped_buf)
                                                or blocked_gemma_chain_may_continue(
                                                    stripped_buf, _enabled_tool_names
                                                )
                                            ):
                                                # Whitespace-tolerant like the parser, and on
                                                # its gate: a rejected name streams as prose.
                                                if _gemma_lead_promotable(stripped_buf):
                                                    _drain_silently = True
                                                elif blocked_gemma_chain_may_continue(
                                                    stripped_buf, _enabled_tool_names
                                                ):
                                                    # A promotable peer behind a blocked
                                                    # call must not stream first.
                                                    if self._parse_tool_calls_from_text(
                                                        content_buffer,
                                                        allow_incomplete = auto_heal_tool_calls,
                                                        enabled_tool_names = _enabled_tool_names,
                                                    ):
                                                        _drain_silently = True
                                                    elif len(stripped_buf) < _MAX_BARE_JSON_BUFFER:
                                                        _hold_buffer = True
                                                    else:
                                                        _drain_silently = True
                                                elif len(stripped_buf) < _MAX_BUFFER_CHARS:
                                                    _hold_buffer = True

                                        if _drain_silently:
                                            # The buffered content IS the call; drain it
                                            # without yielding. A live <think> prefix is
                                            # separate from it -- close that.
                                            detect_state = _S_DRAINING
                                            # Call text begins at the held buffer
                                            # (live arg display only; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                            if _close_streamed_think():
                                                yield {
                                                    "type": "content",
                                                    "text": cumulative_display,
                                                }
                                        elif is_match:
                                            # Tool signal -- flush any visible
                                            # prefix before DRAINING so the
                                            # route sends it before tool_start.
                                            # Use the final strip (all families incl. Llama
                                            # <|python_tag|> / Mistral name): the buffer holds
                                            # the whole call, so a streaming closed-only strip
                                            # would leak its open-ended markup as display text.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                                final = True,
                                                force = True,
                                            )
                                            if len(cleaned) > len(_last_emitted):
                                                _last_emitted = cleaned
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": cleaned,
                                                    }
                                            detect_state = _S_DRAINING
                                            # Live-arg display starts at the held buffer
                                            # (visible prefix flushed above; UI extracts the code).
                                            _text_args_call_start = len(content_accum) - len(
                                                content_buffer
                                            )
                                        elif _hold_buffer or (
                                            is_prefix
                                            and (
                                                is_rehearsal_prefix
                                                or len(stripped_buf) < _MAX_BUFFER_CHARS
                                            )
                                        ):
                                            # A rehearsal prefix is self-bounded; the buffer
                                            # cap must not cut long MCP names short.
                                            pass  # keep buffering
                                        else:
                                            # Not a tool -- flush buffer
                                            detect_state = _S_STREAMING
                                            # Flush reasoning accumulated
                                            # during BUFFERING.
                                            _flush_reasoning_and_buffer()
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                            )
                                            # Same trailing-name hold as STREAMING for this
                                            # first flush out of BUFFERING.
                                            _hold = _held_rehearsal_tail_len(cleaned, _detect_tools)
                                            _emit = (
                                                cleaned[: len(cleaned) - _hold]
                                                if _hold
                                                else cleaned
                                            )
                                            if len(_emit) > len(_last_emitted):
                                                _last_emitted = _emit
                                                if not _suppress_visible_output:
                                                    yield {
                                                        "type": "content",
                                                        "text": _emit,
                                                    }

                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                        if _stream_done:
                            break  # exit outer for

                # ── Resolve BUFFERING at stream end ──
                if detect_state == _S_BUFFERING:
                    stripped_buf = content_buffer.lstrip()
                    # A held bare-JSON fragment has no XML signal; route it to DRAINING (the signal-only
                    # gate below would flush the raw JSON to the user).
                    _bare_eos = strip_llama3_leading_sentinels(stripped_buf)
                    # Gate on enabled names so an ordinary JSON answer isn't routed to DRAINING and dropped.
                    _is_bare_tc = bool(active_tools) and _looks_like_enabled_bare_json(
                        _bare_eos, _enabled_tool_names
                    )
                    if stripped_buf and _gguf_has_genuine_tool_signal(
                        stripped_buf, _tool_xml_signals, _detect_tools
                    ):
                        detect_state = _S_DRAINING
                    elif _is_bare_tc:
                        detect_state = _S_DRAINING
                    elif content_accum or reasoning_accum:
                        detect_state = _S_STREAMING
                        if content_buffer:
                            # Flush reasoning first.
                            _flush_reasoning_and_buffer()
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    ),
                                }
                        elif reasoning_accum and not has_content_tokens:
                            # Reasoning-only reply: show it as the main response,
                            # not a thinking block (mirrors the no-tool path; the
                            # route's extractor closes the streamed <think>).
                            if _reasoning_started_at is not None and not _reasoning_summary_emitted:
                                _reasoning_summary_emitted = True
                                _summary = _reasoning_summary_event(_reasoning_started_at)
                                if _suppress_visible_output:
                                    _deferred_reasoning_summary = _summary
                                else:
                                    yield _summary
                            cumulative_display = _finalize_reasoning_only_cumulative(
                                cumulative_display,
                                reasoning_accum,
                                _iter_finish_reason,
                                promote_reasoning_only,
                                _resumed_reasoning,
                            )
                            _prov_entry = None
                            if not _suppress_visible_output:
                                yield {
                                    "type": "content",
                                    "text": cumulative_display,
                                }
                    else:
                        # Held buffer was no tool signal and no enabled bare-JSON call: a leading ``{`` is an
                        # ordinary JSON answer and must be shown; any other partial-markup prefix is dropped.
                        _held = strip_llama3_leading_sentinels(content_buffer.lstrip())
                        if _held.startswith("{") and not _suppress_visible_output:
                            yield {"type": "content", "text": _held}
                        return

                # ── STREAMING path: no tool call ──
                if detect_state == _S_STREAMING:
                    # Safety net: re-parse the full content for tool calls. The
                    # route layer resets prev_text on tool_start, so post-tool
                    # synthesis streams correctly even if content was emitted
                    # before the tool XML.
                    # Unconditional (not gated on _tool_xml_signals): bare-JSON and Gemma wrapper-less
                    # calls carry no XML signal, so a signal gate would let them slip past.
                    _safety_tc = self._parse_tool_calls_from_text(
                        content_accum,
                        allow_incomplete = auto_heal_tool_calls,
                        enabled_tool_names = _enabled_tool_names,
                    )
                    if not _safety_tc:
                        # ── Re-prompt on plan-without-action ──
                        # Intent described without a tool call: nudge it to act. Up
                        # to _MAX_REPROMPTS times, only on short responses with intent
                        # signals -- "4" or "Hello!" won't trigger it. Uses content,
                        # else reasoning text (reasoning-only stalls).
                        # Classify only what the user sees: tool-call markup is
                        # scrubbed, and reasoning shows only with no content tokens.
                        _visible_raw = content_accum.strip()
                        _visible = (
                            _strip_tool_markup(content_accum, final = True).strip()
                            if _visible_raw
                            else ""
                        )
                        _reasoning = reasoning_accum.strip()
                        # Bracketed reasoning IS the turn for a Magistral-style model
                        # and the strip takes it whole, so fall back to the raw text.
                        _stripped = _visible or _reasoning or _visible_raw
                        # Thinking rendered as <think> in the CONTENT channel stays
                        # in _visible, and a fence inside one was never shown.
                        _visible_answer = _text_outside_think(_visible).strip()
                        # Reasoning stands in for the answer only when the loop
                        # promotes it to visible content; on the Anthropic path it
                        # stays a thinking block and the user saw nothing.
                        _reasoning_shown = (
                            not has_content_tokens
                            and promote_reasoning_only
                            and _iter_finish_reason != "length"
                        )
                        _artifact_text = (
                            _visible_answer
                            if _visible_answer
                            else (_reasoning if _reasoning_shown else "")
                        )
                        # Same for intent: a plan only thought is not one announced.
                        # With nothing outside the block the turn showed nothing, which
                        # IS the stall. _stripped stays whole; it is what gets replayed
                        # as the assistant turn and compared for a repeat.
                        _intent_text = _visible_answer if _visible_answer else _stripped

                        # ── Continue an answer the window cut in half ──
                        # The sibling case below is a turn that showed NOTHING. This one
                        # showed real work and stopped mid-sentence: observed streaming a
                        # 2401-byte file inline at a 4096 window, cut at `ctx.arc(6, -5, 5,`
                        # with no closing tag. Compaction cannot help, because the room that
                        # ran out is the reply's, not the prompt's; the answer has to be
                        # finished across two turns instead.
                        #
                        # `continue_final_message` is what makes that seamless: the partial
                        # goes back as the assistant turn to be EXTENDED, so the model picks
                        # up mid-token rather than restarting and apologising.
                        _partial_answer = content_accum.strip()
                        if (
                            _iter_finish_reason == "length"
                            and _partial_answer
                            and not tool_calls_acc
                            and _length_continuations < _MAX_LENGTH_CONTINUATIONS
                        ):
                            # Withheld BEFORE the nudge is sent, not regretted after. A model
                            # echoing one fragment would otherwise have the echo stitched into
                            # the answer, which is how hermes-agent got a 60,698-char reply.
                            if is_repetition_dominated(_partial_answer):
                                logger.info(
                                    "Truncated answer is repetition-dominated (%d chars); "
                                    "keeping it as-is rather than continuing an echo",
                                    len(_partial_answer),
                                )
                            else:
                                _length_continuations += 1
                                logger.info(
                                    "Answer cut off at %d chars; continuing %d/%d",
                                    len(_partial_answer),
                                    _length_continuations,
                                    _MAX_LENGTH_CONTINUATIONS,
                                )
                                # Candidate first, as the final pass does: a retry that
                                # cannot be served, or one the caller's own cap leaves no
                                # room for, must not spend the conversation or the usage.
                                _cand_c = list(conversation)
                                append_assistant_turn(
                                    _cand_c,
                                    # Unstripped: `strip()` would make the replayed prefix
                                    # differ from the text already streamed, and the next
                                    # delta is concatenated onto the STREAMED one. Inside a
                                    # code block that eats the newline and indentation the
                                    # continuation is about to build on.
                                    {"role": "assistant", "content": content_accum},
                                    continue_final_message = True,
                                )
                                _fu_c = (
                                    _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                                )
                                # A `length` stop means the attempt produced its whole
                                # allowance, which is what to charge when the server
                                # reported no usage.
                                _spent_c = _fu_c.get("completion_tokens", 0) or int(
                                    payload.get("max_tokens") or 0
                                )
                                _cap_left_c = _loop_budget_left(_spent_c)
                                _fits_c = _cap_left_c != 0 and _loop_continuation_fits(
                                    _cand_c, safe_tools, _reasoning_kw, True
                                )
                                if _cap_left_c != 0 and not _fits_c:
                                    # Same as the thought continuation below: abandoning
                                    # here ends the turn, so the next iteration's ordinary
                                    # preflight never gets to drop an older exchange.
                                    _evicted_c = _evict_until_it_fits(
                                        _cand_c, safe_tools, _reasoning_kw, True
                                    )
                                    if _evicted_c is not None and _loop_continuation_fits(
                                        _evicted_c, safe_tools, _reasoning_kw, True
                                    ):
                                        _cand_c = _evicted_c
                                        _fits_c = True
                                if _cap_left_c == 0 or not _fits_c:
                                    logger.info(
                                        "Not continuing the answer: %s",
                                        "the caller's output cap is spent"
                                        if _cap_left_c == 0
                                        else "the retry prompt would not be served",
                                    )
                                    _length_continuations -= 1
                                else:
                                    conversation[:] = _cand_c
                                    continue_final_message = True
                                    if _cap_left_c is not None:
                                        _continuation_max_tokens = _cap_left_c
                                    _accumulated_completion_tokens += _fu_c.get(
                                        "completion_tokens", 0
                                    )
                                    _it_c = _iter_timings or {}
                                    _accumulated_predicted_ms += _it_c.get("predicted_ms", 0)
                                    _accumulated_predicted_n += _it_c.get("predicted_n", 0)
                                    if _continuation_credits < _MAX_CONTINUATION_CREDITS:
                                        _continuation_credits += 1
                                    yield {"type": "status", "text": ""}
                                    yield {
                                        "type": "status",
                                        "text": _CONTINUE_TRUNCATED_ANSWER_STATUS,
                                    }
                                    continue

                        # ── Continue a turn that ended inside its own thought ──
                        # Ahead of the re-prompt below because that one is gated on a SHORT
                        # response, and a thought that filled the window is the opposite of
                        # short -- the case fell through the gap and returned an empty turn.
                        if (
                            _iter_finish_reason == "length"
                            and not content_accum.strip()
                            and reasoning_accum.strip()
                            and _length_continuations < _MAX_LENGTH_CONTINUATIONS
                        ):
                            _length_continuations += 1
                            # Thinking off for the retry. Leaving it on re-runs the turn that
                            # just failed, and the second failure looks identical to the first.
                            _effective_enable_thinking = False
                            _effective_reasoning_effort = None
                            logger.info(
                                "Turn ended inside its reasoning (%d chars, finish_reason "
                                "length); continuing %d/%d with thinking off",
                                len(reasoning_accum),
                                _length_continuations,
                                _MAX_LENGTH_CONTINUATIONS,
                            )
                            # Candidate first, exactly as the partial-answer branch above
                            # does. This one mutated `conversation` directly and checked
                            # neither the fit nor the caller's cap: a thought that reached
                            # the context wall makes the retry prompt window-sized, and
                            # with no evictable history the recovery request is the one
                            # thing that must not be rejected.
                            _cand_l = list(conversation)
                            append_assistant_turn(
                                _cand_l,
                                {
                                    "role": "assistant",
                                    # The stream holds only the resumed thought's tail.
                                    "content": _unfinished_thought_progress(
                                        _resumed_reasoning + reasoning_accum
                                    ),
                                },
                                continue_final_message = continue_final_message,
                            )
                            if _resumed_reasoning:
                                _cand_l[-1].pop("reasoning_content", None)
                            _cand_l.append(
                                {
                                    "role": "user",
                                    "content": _continue_after_length_message(),
                                }
                            )
                            _fu_l = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _spent_l = _fu_l.get("completion_tokens", 0) or int(
                                payload.get("max_tokens") or 0
                            )
                            _cap_left_l = _loop_budget_left(_spent_l)
                            # `_reasoning_kw` was computed at the top of this iteration,
                            # with thinking ON. The retry goes out with it off, which is a
                            # different rendered prompt on any template that reads
                            # `enable_thinking`, so admitting under the old kwargs prices
                            # a request nobody sends. Same fix the final pass already has.
                            _off_kw_l = self._request_reasoning_kwargs(
                                False, None, preserve_thinking
                            )
                            # Counted ONCE in the ordinary case: the eviction attempt
                            # below only runs on a candidate already found too large.
                            _fits_l = _cap_left_l != 0 and _loop_continuation_fits(
                                _cand_l, safe_tools, _off_kw_l
                            )
                            if _cap_left_l != 0 and not _fits_l:
                                # Possibly one eviction short. Refusing here ends the
                                # turn, so the next iteration's ordinary preflight never
                                # gets its chance to drop an older exchange.
                                _evicted_l = _evict_until_it_fits(_cand_l, safe_tools, _off_kw_l)
                                if _evicted_l is not None and _loop_continuation_fits(
                                    _evicted_l, safe_tools, _off_kw_l
                                ):
                                    _cand_l = _evicted_l
                                    _fits_l = True
                            if _cap_left_l == 0 or not _fits_l:
                                logger.info(
                                    "Not continuing the thought: %s",
                                    "the caller's output cap is spent"
                                    if _cap_left_l == 0
                                    else "the retry prompt would not be served",
                                )
                                # Which wall it was, so the give-up below names the lever
                                # the user actually has. Blaming the context window for a
                                # spent Max Tokens sends them to the one setting that was
                                # never the constraint, and this reaches the client as
                                # ordinary content, so nothing downstream can correct it.
                                _reasoning_cap_spent = _cap_left_l == 0
                                _length_continuations -= 1
                                _effective_enable_thinking = enable_thinking
                                _effective_reasoning_effort = reasoning_effort
                            else:
                                conversation[:] = _cand_l
                                if _cap_left_l is not None:
                                    _continuation_max_tokens = _cap_left_l
                                _accumulated_completion_tokens += _fu_l.get("completion_tokens", 0)
                                _it_l = _iter_timings or {}
                                _accumulated_predicted_ms += _it_l.get("predicted_ms", 0)
                                _accumulated_predicted_n += _it_l.get("predicted_n", 0)
                                # Blank first so the route resets its text cursor, as the
                                # re-prompt below does; else the retry reads as a hang.
                                if _continuation_credits < _MAX_CONTINUATION_CREDITS:
                                    _continuation_credits += 1
                                yield {"type": "status", "text": ""}
                                yield {"type": "status", "text": _CONTINUE_AFTER_LENGTH_STATUS}
                                continue

                        if (
                            _iter_finish_reason == "length"
                            and not content_accum.strip()
                            and reasoning_accum.strip()
                        ):
                            # Allowance spent and still nothing to show. Saying so is the
                            # whole point: returning here silently is the original defect,
                            # a turn that renders as an empty message with no indication
                            # that anything went wrong or what the user might change.
                            logger.info(
                                "Reasoning exhausted the window %d times; giving up with "
                                "an explanation",
                                _length_continuations,
                            )
                            # `_reasoning_cap_spent` is only set when a continuation was
                            # REFUSED. Arriving here by exhausting the retry limit leaves
                            # it at its default, so the cap is re-read: the last permitted
                            # attempt may be the one that finished off Max Tokens.
                            _fu_g = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _spent_g = _fu_g.get("completion_tokens", 0) or int(
                                payload.get("max_tokens") or 0
                            )
                            yield {"type": "status", "text": ""}
                            yield {
                                "type": "content",
                                "text": (
                                    _reasoning_cap_spent_message(max_tokens)
                                    if _reasoning_cap_spent or _loop_budget_left(_spent_g) == 0
                                    else _thinking_exhausted_message(self._effective_context_length)
                                ),
                            }
                            _meta = _build_metadata_event(
                                _iter_usage, _iter_timings, _iter_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                            return

                        _render_html_already_done_intent = _tool_succeeded(
                            "render_html"
                        ) and re.search(
                            r"(?i)\brender[_\s-]?html\b",
                            _stripped,
                        )
                        # A post-tool stall still deserves a nudge, but each retry
                        # re-runs tools, so allow only one. RAG autoinject never lands
                        # in history, so _auto keeps a doc-grounded turn from reading
                        # as pre-tool (mirrors safetensors rag_autoinjected).
                        _already_acted = bool(_auto) or any(
                            record.executed for record in tool_controller.history
                        )
                        if _already_acted:
                            _reprompt_used, _reprompt_cap = _post_tool_reprompts, 1
                        else:
                            _reprompt_used, _reprompt_cap = _reprompt_count, _MAX_REPROMPTS
                        # None follows the shared process default; explicit values win.
                        if (
                            auto_heal_tool_calls
                            and _nudge_enabled(nudge_tool_calls)
                            and active_tools
                            and not _render_html_already_done_intent
                            and _reprompt_used < _reprompt_cap
                            and not _is_reprompt_repeat(_stripped, _last_reprompt_text)
                            # On _stripped as well: it is what gets replayed, and the
                            # intent text can be a short tail of a very long turn.
                            and len(_stripped) < _REPROMPT_MAX_CHARS
                            and _is_short_intent_without_action(_intent_text)
                            and not (_artifact_text and _has_answer_artifact(_artifact_text))
                        ):
                            _reprompt_count += 1
                            if _already_acted:
                                _post_tool_reprompts += 1
                            _last_reprompt_text = _stripped
                            logger.info(
                                f"Re-prompt {_reprompt_used + 1}/{_reprompt_cap}: "
                                f"model responded without calling tools "
                                f"({len(_stripped)} chars)"
                            )
                            # Merges into a resumed partial: the nudge that follows is a
                            # user turn, so a second assistant turn breaks alternation.
                            append_assistant_turn(
                                conversation,
                                {"role": "assistant", "content": _stripped},
                                continue_final_message = continue_final_message,
                            )
                            available_tool_names = [
                                (tool.get("function") or {}).get("name")
                                for tool in active_tools
                                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                            ]
                            available_tool_names = [name for name in available_tool_names if name]
                            tool_hint = " or ".join(available_tool_names) or "an available tool"
                            _forced_tool_call_pending = True
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": _reprompt_to_act_message(tool_hint),
                                }
                            )
                            # Accumulate tokens and timing from this iteration.
                            _fu_r = _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                            _accumulated_completion_tokens += _fu_r.get("completion_tokens", 0)
                            _it_r = _iter_timings or {}
                            _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                            # Blank first (the route resets its text cursor only on an
                            # empty status), then the badge so the retry is not a hang.
                            yield {"type": "status", "text": ""}
                            yield {"type": "status", "text": _NUDGE_TOOL_CALLS_STATUS}
                            continue

                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                            if not _should_suppress_forced_no_tool_output(
                                _stripped,
                                _last_reprompt_text,
                            ):
                                if cumulative_display:
                                    forced_visible_text = _strip_tool_markup(
                                        cumulative_display,
                                        final = True,
                                    )
                                elif content_accum:
                                    forced_visible_text = _strip_tool_markup(
                                        content_accum,
                                        final = True,
                                    )
                                else:
                                    forced_visible_text = reasoning_accum
                                if forced_visible_text:
                                    yield {
                                        "type": "content",
                                        "text": forced_visible_text,
                                    }
                                    if _deferred_reasoning_summary is not None:
                                        yield _deferred_reasoning_summary
                        elif not _suppress_visible_output:
                            # Turn ended as a plain answer (no [ARGS] followed): the held
                            # rehearsal tail is real prose, release it.
                            _final_clean = _strip_tool_markup_streaming(cumulative_display)
                            if len(_final_clean) > len(_last_emitted):
                                yield {"type": "content", "text": _final_clean}

                        # Content was already streamed.  Yield metadata.
                        yield {"type": "status", "text": ""}
                        _cut = _quote_cut_event(
                            reasoning_accum,
                            _visible,
                            _iter_finish_reason,
                            promote_reasoning_only,
                        )
                        if _cut is not None:
                            yield _cut
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                    # Safety net caught tool XML -- treat as tool call.
                    tool_calls = _safety_tc
                    content_text = _strip_tool_markup(
                        content_accum,
                        final = True,
                        force = True,
                    )
                    logger.info(
                        f"Safety net: parsed {len(tool_calls)} tool call(s) from streamed content"
                    )
                else:
                    # ── DRAINING path: assemble tool_calls ──
                    tool_calls = None
                    content_text = content_accum
                    if has_structured_tc:
                        # Drop incomplete fragments (e.g. from max_tokens
                        # truncation or disconnect).
                        # first-seen order: a call forked onto a reused index arrived after every call already open, so it must execute and replay last
                        tool_calls = [
                            call
                            for call in tool_calls_acc.values()
                            if (call.get("function", {}).get("name", "").strip())
                        ] or None
                    if not tool_calls:
                        # Unconditional re-parse: we only reach DRAINING when the buffer looked like a
                        # call, and bare-JSON / Gemma wrapper-less calls carry no XML signal to gate on.
                        tool_calls = self._parse_tool_calls_from_text(
                            content_accum,
                            allow_incomplete = auto_heal_tool_calls,
                            enabled_tool_names = _enabled_tool_names,
                        )
                    if tool_calls and not has_structured_tc:
                        content_text = _strip_tool_markup(
                            content_text,
                            final = True,
                            force = True,
                        )
                        # ``_strip_tool_markup`` only knows XML; also drop a leading bare-JSON call so the
                        # executed call isn't replayed as text or next-turn history.
                        content_text = strip_leading_bare_json_call(
                            content_text, _enabled_tool_names
                        )
                    if tool_calls:
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s) from "
                            f"{'structured delta' if has_structured_tc else 'content text'}"
                        )
                    if not tool_calls:
                        # DRAINING but no tool calls (false positive): close any
                        # provisional cards (a sniff can open one whose call never
                        # parses); without a tool_end the card spins forever.
                        for _pid, _pname in provisional_started_tool_calls.items():
                            if _pid not in resolved_provisional_tool_call_ids:
                                resolved_provisional_tool_call_ids.add(_pid)
                                yield {
                                    "type": "tool_end",
                                    "tool_name": _pname,
                                    "tool_call_id": _pid,
                                    "result": "",
                                    "provenance": provisional_tool_provenance(_pname),
                                }
                        # Merge metrics from prior tool iterations so they aren't dropped.
                        yield {"type": "status", "text": ""}
                        if content_accum:
                            # Strip leaked tool-call XML before yielding.
                            content_accum = _strip_tool_markup(content_accum, final = True)
                        # A truncated bare-JSON call has no XML markup to strip and didn't parse. With
                        # Auto-Heal on, drop a leading ENABLED-tool fragment (ordinary JSON answers untouched);
                        # off keeps it visible per the strict contract.
                        if content_accum and active_tools and auto_heal_tool_calls:
                            content_accum = strip_leading_bare_json_call(
                                content_accum, _enabled_tool_names
                            )
                        if content_accum:
                            yield {"type": "content", "text": content_accum}
                        _meta = _build_metadata_event(
                            _iter_usage, _iter_timings, _iter_finish_reason
                        )
                        if _meta is not None:
                            yield _meta
                        return

                # ── Execute tool calls ──
                # The turn got somewhere, so the earlier stall is spent history. Without
                # this reset the allowance is consumed for the whole request and a later,
                # unrelated stall gets fewer attempts than it should -- the shape of
                # NousResearch/hermes-agent#79100, where the counter survived a good
                # tool round and the give-up fired early. Thinking comes back on for the
                # same reason: it was turned off to break one stall, not for the request.
                _length_continuations = 0
                _effective_enable_thinking = enable_thinking
                _effective_reasoning_effort = reasoning_effort
                # Captured BEFORE the reset below. This turn may have been resumed with
                # `continue_final_message`, leaving the partial as the trailing assistant
                # message, and anything recorded further down has to merge into it.
                # Restoring the caller's value first made `append_assistant_turn` add a
                # SECOND consecutive assistant message, which is the role alternation that
                # helper exists to preserve and which strict templates reject.
                _merge_into_partial = continue_final_message
                continue_final_message = _caller_continue_final_message
                _accumulated_completion_tokens += (
                    _backfill_usage_from_timings(_iter_usage, _iter_timings) or {}
                ).get("completion_tokens", 0)
                _it = _iter_timings or {}
                _accumulated_predicted_ms += _it.get("predicted_ms", 0)
                _accumulated_predicted_n += _it.get("predicted_n", 0)
                # Refreshes the context meter from the usage llama-server already reported (no
                # recount). Sent once the first tool is running, so it never delays a tool.
                _pending_context_usage = _context_usage_event(_iter_usage, _iter_timings)

                # Collapse exact-duplicate calls and cap the count for the TEXTUAL
                # fallback (mirrors the safetensors loop; see _MAX_TOOL_CALLS_PER_TURN).
                _over_cap: list = []
                if tool_calls and not has_structured_tc and len(tool_calls) > 1:
                    _seen_keys: set = set()
                    _last_workspace_key = None
                    # One rerun per piece of new work: `test, edit A, test, edit B, test` keeps
                    # every test, while `read, edit, read, edit` stops replaying and cannot
                    # fill the cap.
                    _novel_kept = 0
                    _novel_at_last_keep: dict = {}
                    _deduped: list = []
                    if not deduplicate_tool_calls:
                        _deduped = tool_calls[:_MAX_TOOL_CALLS_PER_TURN]
                        _over_cap = tool_calls[_MAX_TOOL_CALLS_PER_TURN:]
                    for _tc in tool_calls if deduplicate_tool_calls else ():
                        _fn = _tc.get("function", {}) or {}
                        _key = (_fn.get("name", ""), str(_fn.get("arguments", "")))
                        if _reruns_after_new_work(_fn.get("name")):
                            if _key == _last_workspace_key:
                                continue
                            if _key in _seen_keys:
                                if _novel_kept <= _novel_at_last_keep.get(_key, 0):
                                    continue
                            elif _fn.get("name") not in _WORKSPACE_READ_TOOLS:
                                _novel_kept += 1
                            _novel_at_last_keep[_key] = _novel_kept
                            _last_workspace_key = _key
                        # A browser call acts on the page as it is then: two scrolls are two scrolls.
                        elif _key in _seen_keys and _fn.get("name") not in BROWSER_TOOL_NAMES:
                            continue
                        _seen_keys.add(_key)
                        if len(_deduped) < _MAX_TOOL_CALLS_PER_TURN:
                            _deduped.append(_tc)
                        else:
                            _over_cap.append(_tc)
                    if len(_deduped) + len(_over_cap) != len(tool_calls):
                        logger.info(
                            "GGUF textual fallback: collapsed %d repeated tool call(s) "
                            "in one turn to %d",
                            len(tool_calls),
                            len(_deduped) + len(_over_cap),
                        )
                    if _over_cap:
                        logger.info(
                            "GGUF textual fallback: skipped %d tool call(s) over the "
                            "per-turn limit of %d",
                            len(_over_cap),
                            _MAX_TOOL_CALLS_PER_TURN,
                        )
                    tool_calls = _deduped

                # disable_parallel_tool_use: execute only the first tool call
                # this turn. Truncate before building assistant_msg so the
                # conversation stays consistent and extra calls are never executed.
                if disable_parallel_tool_use and tool_calls and len(tool_calls) > 1:
                    tool_calls = tool_calls[:1]
                    _over_cap = []

                assistant_msg: dict = {"role": "assistant", "content": content_text}
                if reasoning_accum.strip():
                    assistant_msg["reasoning_content"] = reasoning_accum
                assistant_appended = False
                # Collect no-op feedback and flush it after the batch, so a
                # suppressed call never drops a later parallel call.
                deferred_noop_msgs: list = []
                # Which tools those no-ops were about, so the flush below can tell
                # whether the trailing result belongs to the same tool.
                deferred_noop_tools: set = set()
                # Per result, so a parallel batch is not squeezed into one
                # result's worth of images.
                batch_mcp_images: list = []
                # Where this batch's results start, for the image turn\'s wording.
                batch_conversation_start = len(conversation)

                # The text-path provisional card uses the parser's default id ("call_0");
                # a Mistral-style call carries its own id and would open a duplicate. Reuse
                # the card's id for the first matching call (same tool name) to reconcile.
                _text_provisional_id = _text_args_id if not has_structured_tc else ""

                for _call_index, tc in enumerate(tool_calls or []):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    if (
                        _text_provisional_id
                        and _text_provisional_id in provisional_started_tool_calls
                        and _text_provisional_id not in resolved_provisional_tool_call_ids
                        and tc.get("id") not in provisional_started_tool_calls
                        and provisional_started_tool_calls[_text_provisional_id] == tool_name
                    ):
                        tc = {**tc, "id": _text_provisional_id}
                    provisional_match = tc.get("id") in provisional_started_tool_calls
                    decision = tool_controller.prepare_call(
                        tc,
                        forced = _forced_tool_call_pending,
                        provisional = provisional_match,
                        allowed_tool_names = {forced_name} if forced_name else None,
                    )
                    image_share = None
                    if decision.should_execute and mcp_image is not None:
                        image_share = mcp_image_share(
                            decision.tool_name, decision.arguments, mcp_image
                        )
                        if image_share is not None:
                            decision = tool_controller.reprepare_call(
                                decision,
                                forced = _forced_tool_call_pending,
                                provisional = provisional_match,
                                allowed_tool_names = {forced_name} if forced_name else None,
                            )
                    # The frontend groups a parallel batch into one assistant message by this.
                    decision.provenance["round_id"] = iteration

                    if not decision.should_execute:
                        if provisional_match:
                            # A provisional tool card is already on screen for this
                            # id; close it so it never dangles when the controller
                            # turns the call into an internal no-op (duplicate /
                            # disabled / render_html_repeat).
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": "",
                                "provenance": decision.provenance,
                            }
                        completion = tool_controller.record_noop(decision)
                        deferred_noop_msgs.append(completion.model_message())
                        deferred_noop_tools.add(decision.tool_name)
                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                        logger.info(
                            "Suppressed local GGUF tool call as internal no-op: "
                            f"action={decision.action} tool={decision.tool_name}"
                        )
                        continue

                    if not assistant_appended:
                        assistant_msg["tool_calls"] = [decision.as_assistant_tool_call()]
                        # Merges into a resumed partial, so a continued turn that calls
                        # a tool stays one assistant message.
                        append_assistant_turn(
                            conversation,
                            assistant_msg,
                            continue_final_message = _merge_into_partial,
                        )
                        assistant_appended = True
                    else:
                        assistant_msg.setdefault("tool_calls", []).append(
                            decision.as_assistant_tool_call()
                        )

                    # Bypass wins here too, so a direct internal caller with both
                    # flags never prompts. "auto" pauses only high-risk calls; "off"
                    # pauses only a high-risk python/terminal call without OS isolation.
                    needs_confirm = needs_tool_confirmation(
                        confirm_tool_calls = bool(confirm_tool_calls),
                        bypass_permissions = bypass_permissions,
                        permission_mode = permission_mode,
                        name = decision.tool_name,
                        arguments = decision.arguments,
                        is_high_risk = is_high_risk_tool_call,
                        never_needs = never_needs_approval,
                        sandbox_level = sandbox_level,
                    )
                    # Sending the user's image always asks, whatever the permission mode.
                    needs_confirm = needs_confirm or image_share is not None
                    strict_isolation = requires_os_isolation(
                        confirm_tool_calls = bool(confirm_tool_calls),
                        bypass_permissions = bypass_permissions,
                        permission_mode = permission_mode,
                        name = decision.tool_name,
                        arguments = decision.arguments,
                        prompted = needs_confirm,
                        is_high_risk = is_high_risk_tool_call,
                        sandbox_level = sandbox_level,
                    )
                    approval_id = new_approval_id() if needs_confirm else ""
                    decision_slot = (
                        begin_tool_decision(session_id, approval_id) if needs_confirm else None
                    )
                    start_event = decision.tool_start_event()
                    start_event["approval_id"] = approval_id
                    start_event["awaiting_confirmation"] = needs_confirm
                    if image_share is not None:
                        start_event["image_disclosure"] = image_share["disclosure"]

                    try:
                        # Gated calls are not running yet; a "Running ..." badge
                        # counting up while it waits on a human reads as a hang.
                        yield {
                            "type": "status",
                            "text": (
                                awaiting_approval_status(decision.tool_name)
                                if needs_confirm
                                else decision.status_text
                            ),
                        }
                        yield start_event
                        if needs_confirm and _pending_context_usage is not None:
                            # Nothing runs while a human decides, so the meter need not wait for it.
                            yield _pending_context_usage
                            _pending_context_usage = None

                        _decision = (
                            wait_tool_decision(
                                decision_slot,
                                approval_id,
                                cancel_event = cancel_event,
                            )
                            if decision_slot is not None
                            else None
                        )
                        # The slot is where the waiter says WHY: the user's refusal, or an approval
                        # nobody answered. Read before decision_slot is dropped in the deny branch.
                        _decision_reason = decision_reason(decision_slot)
                        if _decision is not None and _decision != "deny":
                            # Approved: now it really is running.
                            yield {"type": "status", "text": decision.status_text}
                        if _decision == "deny":
                            decision_slot = None
                            _forced_choice_resolved = True
                            resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                            # An approval nobody answered is not the user's decision, and this
                            # string is the only account of the call both the model and the
                            # reopened card get: the buttons are gone by the time it lands.
                            _denied_text = (
                                TOOL_APPROVAL_EXPIRED_MESSAGE
                                if _decision_reason == DECISION_EXPIRED
                                else TOOL_REJECTED_MESSAGE
                            )
                            yield {
                                "type": "tool_end",
                                "tool_name": decision.tool_name,
                                "tool_call_id": decision.tool_call_id,
                                "result": _denied_text,
                                "provenance": decision.provenance,
                            }
                            denied_message = {
                                "role": "tool",
                                "name": decision.tool_name,
                                "content": _denied_text,
                            }
                            if decision.tool_call_id:
                                denied_message["tool_call_id"] = decision.tool_call_id
                                _tool_card_by_message[id(denied_message)] = decision.tool_call_id
                            conversation.append(denied_message)
                            # The denial is itself the `tool` message that makes the
                            # template render the call it refused, so declining an
                            # oversized write COSTS the prompt the very arguments it
                            # declined to afford. The bulk compaction deliberately skips
                            # denials now that replies are paired with their call, so
                            # without this the user gets a context error in place of the
                            # refusal they asked for.
                            if decision.tool_call_id:
                                conversation[:] = compact_refused_tool_arguments(
                                    conversation, decision.tool_call_id
                                )
                                # Rebuilt, not mutated, so the handle this batch is still
                                # appending to is now detached. Same defect the executed
                                # path had: the next call in the batch would land on the
                                # stale object while its RESULT lands in `conversation`,
                                # and the model receives a result for a call it cannot see.
                                for _dmsg in reversed(conversation):
                                    if _dmsg.get("role") != "assistant":
                                        continue
                                    if any(
                                        (_tc or {}).get("id") == decision.tool_call_id
                                        for _tc in (_dmsg.get("tool_calls") or [])
                                    ):
                                        assistant_msg = _dmsg
                                        break
                            if _forced_tool_call_pending:
                                _forced_tool_call_pending = False
                            continue
                        decision_slot = None
                    finally:
                        if decision_slot is not None:
                            abort_tool_decision(decision_slot, approval_id)

                    # Can the turn this call is part of still be SERVED once it returns?
                    # Everything below prices what a result may add; nothing asked whether
                    # the prompt fits with no result at all. It can already not fit: the
                    # model's own arguments are in `conversation` by now (appended above),
                    # and a whole-file `edit_file` puts the file itself there. The budget
                    # clamps to zero, the result is cut to its notice, the tool runs, and
                    # llama-server rejects the next request anyway -- with the write on
                    # disk. Nothing downstream recovers it: the fit protects the newest
                    # turn, and ctx_shift is off for any mmproj load, so there is no
                    # sliding window underneath this either.
                    #
                    # Two levers, both only once the turn is actually spilling. History
                    # first, via the arguments of calls that already returned -- pure
                    # replay, since the tool got them in full and the file is on disk --
                    # and only then a refusal, which is worth saying precisely because it
                    # is the one outcome on this path that leaves nothing written.
                    _unservable_text: Optional[str] = None
                    # Set when the gate chose to RUN an oversized call rather than
                    # refuse it, on the strength of what compacting it will reclaim.
                    _compact_after_execution = False
                    # The exact count of the prompt that compaction WILL produce, kept so
                    # the result budget below is priced against it.
                    _compacted_turn_tokens: Optional[int] = None
                    if self._effective_context_length:
                        # This iteration's cap, like every other sizing decision in it.
                        _room_target = prompt_budget(
                            self._effective_context_length, _iteration_max_tokens
                        )
                        # Cheap gate first. The exact count is a template render plus a
                        # tokenizer pass over the whole conversation, and this runs per
                        # call; a thread with room to spare must not pay for it. The
                        # estimate only decides whether to ASK, never whether to refuse.
                        # Conservative, not the flat dense estimate. That one still
                        # charges ASCII four characters a token, and base64 tokenises
                        # near one, so a ~6 KB encoded `edit_file` payload prices at a
                        # quarter of its real cost and slips under this threshold at a
                        # 4096 window -- the exact check is skipped, the tool runs, and
                        # the write lands before the next request is rejected. Erring
                        # high here only costs an extra count; erring low costs the
                        # whole gate.
                        if (
                            estimate_messages_tokens_conservative(
                                messages_without_unpriced_media(conversation)
                            )
                            > 0.7 * _room_target
                        ):
                            # Priced with a STAND-IN tool message for the call about to
                            # run, not on the conversation as it stands. A chat template
                            # renders an assistant turn's `tool_calls` only once a `tool`
                            # message answers them -- until then the arguments cost the
                            # prompt nothing, so counting the conversation as-is misses
                            # exactly the tokens this gate exists to catch. Measured on
                            # Qwen3.8-27B: the same messages price at 1,063 tokens without
                            # the stand-in and 2,172 with it, and the request the loop goes
                            # on to send is the second number.
                            #
                            # Empty content, because that is the best case: a result can be
                            # cut to its notice but the call itself cannot be unsent. If
                            # even this does not fit, nothing the tool returns will.
                            _probe_message: dict = {
                                "role": "tool",
                                "name": decision.tool_name,
                                "content": "",
                            }
                            if decision.tool_call_id:
                                _probe_message["tool_call_id"] = decision.tool_call_id
                            _compacted_calls = 0
                            for _attempt in range(2):
                                try:
                                    _turn_tokens = self.count_chat_tokens(
                                        neutralize_control_markup_in_messages(
                                            messages_without_unpriced_media(
                                                [*conversation, _probe_message]
                                            ),
                                            _markup_cache,
                                            self.markup_profile,
                                        ),
                                        None,
                                        safe_tools,
                                        strict = True,
                                        chat_template_kwargs = _reasoning_kw,
                                    )
                                except Exception:
                                    # No count, no refusal: the old behaviour (run it and
                                    # let llama-server decide) beats declining a call on a
                                    # number that could not be taken.
                                    logger.debug(
                                        "pre-execution fit: prompt count failed",
                                        exc_info = True,
                                    )
                                    break
                                if turn_is_servable(
                                    self._effective_context_length, max_tokens, _turn_tokens
                                ):
                                    break
                                if _attempt == 0:
                                    # Spend history, then re-price against the real
                                    # tokenizer rather than trusting the saving.
                                    _compacted, _n = compact_completed_tool_arguments(
                                        conversation, protect_last = 1
                                    )
                                    if _n:
                                        conversation[:] = _compacted
                                        _compacted_calls = _n
                                        logger.info(
                                            "Compacted arguments of %d completed tool call(s) "
                                            "to fit the context window",
                                            _n,
                                        )
                                        continue
                                # Last resort before refusing: run it anyway, and compact
                                # ITS arguments the moment it returns.
                                #
                                # Running a tool needs no context -- only the next prompt
                                # does, and by then the arguments describe a file on disk.
                                # Refusing does not avoid their cost either, because the
                                # refusal is itself the `tool` message that renders them;
                                # it just spends the same tokens with nothing written, and
                                # the model retries with a fresh oversized call. Measured
                                # over three rounds of one thread, each retry reclaimed
                                # less (50%, 34%, 15%) and the turn ended in a
                                # one-character reply.
                                if decision.tool_call_id:
                                    _as_if_run = compact_executed_call_arguments(
                                        [*conversation, _probe_message],
                                        decision.tool_call_id,
                                    )
                                    try:
                                        _after_tokens = self.count_chat_tokens(
                                            neutralize_control_markup_in_messages(
                                                messages_without_unpriced_media(_as_if_run),
                                                _markup_cache,
                                                self.markup_profile,
                                            ),
                                            None,
                                            safe_tools,
                                            strict = True,
                                            chat_template_kwargs = _reasoning_kw,
                                        )
                                    except Exception:
                                        _after_tokens = None
                                    if _after_tokens is not None and turn_is_servable(
                                        self._effective_context_length,
                                        max_tokens,
                                        _after_tokens,
                                    ):
                                        _compact_after_execution = True
                                        _compacted_turn_tokens = _after_tokens
                                        logger.info(
                                            "Running %s despite a %d-token turn: compacting "
                                            "its own arguments afterwards brings the next "
                                            "prompt to %d, so the file gets written",
                                            decision.tool_name,
                                            _turn_tokens,
                                            _after_tokens,
                                        )
                                        break
                                from core.inference import context_refusal  # noqa: PLC0415

                                _unservable_text = context_refusal.describe_unservable_tool_call(
                                    decision.tool_name,
                                    _turn_tokens,
                                    self._effective_context_length,
                                    compacted_calls = _compacted_calls,
                                )
                                logger.warning(
                                    "Refused %s before execution: prompt would be %d tokens "
                                    "against a %d-token window",
                                    decision.tool_name,
                                    _turn_tokens,
                                    self._effective_context_length,
                                )
                                break

                    if _unservable_text is not None:
                        # Same shape as the user-denied path above, so the frontend needs
                        # no new case and the model gets a tool result it can read rather
                        # than a stream that stops mid-turn.
                        resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                        yield {
                            "type": "tool_end",
                            "tool_name": decision.tool_name,
                            "tool_call_id": decision.tool_call_id,
                            "result": _unservable_text,
                            "provenance": decision.provenance,
                        }
                        _refused_message = {
                            "role": "tool",
                            "name": decision.tool_name,
                            "content": _unservable_text,
                        }
                        if decision.tool_call_id:
                            _refused_message["tool_call_id"] = decision.tool_call_id
                            _tool_card_by_message[id(_refused_message)] = decision.tool_call_id
                        conversation.append(_refused_message)
                        # The refusal is itself the `tool` message that makes this call's
                        # arguments start rendering, so declining without this leaves the
                        # next generation over the window anyway: observed as an accurate
                        # refusal followed four seconds later by the 400 it was issued to
                        # prevent. Nothing ran, so nothing on disk reflects them.
                        _before_refused_compaction = len(json.dumps(conversation, default = str))
                        conversation[:] = compact_refused_tool_arguments(
                            conversation, decision.tool_call_id
                        )
                        _after_refused_compaction = len(json.dumps(conversation, default = str))
                        if _after_refused_compaction < _before_refused_compaction:
                            logger.info(
                                "Compacted refused %s arguments: %d -> %d chars",
                                decision.tool_name,
                                _before_refused_compaction,
                                _after_refused_compaction,
                            )
                        if _forced_tool_call_pending:
                            _forced_tool_call_pending = False
                        # The forced choice is spent by this refusal, as it is by a user
                        # denial. Left unresolved, the next iteration advertises the same
                        # tool as required and the model is pushed into calling it again
                        # for the same refusal, instead of being free to say what went
                        # wrong or finish the turn another way.
                        _forced_choice_resolved = True
                        # This assistant turn was rebuilt by the compaction above, so the
                        # handle the batch is appending to no longer belongs to it.
                        for _rmsg in reversed(conversation):
                            if _rmsg.get("role") != "assistant":
                                continue
                            if any(
                                (_tc or {}).get("id") == decision.tool_call_id
                                for _tc in (_rmsg.get("tool_calls") or [])
                            ):
                                assistant_msg = _rmsg
                                break
                        continue

                    _effective_timeout = None if tool_call_timeout >= 9999 else tool_call_timeout
                    # RAG: cap paraphrased KB re-searches that slip past the dup guard.
                    if (
                        decision.tool_name in RAG_SEARCH_TOOLS
                        and _kb_search_count >= RAG_MAX_SEARCHES_PER_TURN
                    ):
                        result = RAG_SEARCH_CAP_NUDGE
                    else:
                        # Execute in a worker thread so live stdout chunks and heartbeats
                        # stream while the tool blocks (the SSE route turns heartbeats into
                        # keepalives). Result is byte-identical to a direct call.
                        # Filled in by `_invoke_tool` below so the result log can report
                        # the budget that was actually handed over, from a scope that
                        # outlives the closure.
                        _last_result_budget: list = ["<not passed>"]

                        # Only a call the user answered: the executor lets it reach the host paths it names.
                        _host_access_approved = _decision not in (None, "deny")

                        def _invoke_tool(
                            _output_callback,
                            _decision = decision,
                            _approved = _host_access_approved,
                            _strict = strict_isolation,
                        ):
                            # execute_tool is injectable and may be monkey-patched with the
                            # pre-PR signature; forward output_callback only if it's accepted.
                            kwargs = dict(
                                cancel_event = cancel_event,
                                timeout = _effective_timeout,
                                session_id = session_id,
                                thread_id = thread_id,
                                rag_scope = rag_scope,
                                disable_sandbox = bypass_permissions,
                            )
                            # Run unasked only because the OS sandbox was on: refuse if it is not any more.
                            if _strict and accepts_kwarg(execute_tool, "tool_execution_mode"):
                                kwargs["tool_execution_mode"] = "required"
                            elif runs_without_os_sandbox(
                                _decision.tool_name, sandbox_level
                            ) and accepts_kwarg(execute_tool, "tool_execution_mode"):
                                kwargs["tool_execution_mode"] = "software"
                            # Same branch the forced recall is filtered against, so a
                            # model-initiated search cannot reach a sibling response the
                            # forced recall correctly refused.
                            if accepts_kwarg(execute_tool, "conversation_branch"):
                                kwargs["conversation_branch"] = _extend_live_branch(conversation)
                            if _approved and accepts_kwarg(execute_tool, "host_access_approved"):
                                kwargs["host_access_approved"] = True
                            # And the room actually left: the model picks its own top_k and
                            # the result lands in the current tool exchange, which rolling
                            # truncation protects, so a top_k the window cannot hold ends
                            # the turn in an unrecoverable context-length error.
                            if self._effective_context_length:
                                # Priced against the catalogue too: the messages alone
                                # leave the tools array out, and a big catalogue can put
                                # the request near its budget while this still reports
                                # room for several 500-token chunks.
                                # `fit_rolling_context` returns None when it dropped
                                # nothing, so a prompt that simply FITS leaves
                                # `_prompt_token_offset` unset -- after a context-length
                                # increase, or on a shorter branch. The character estimate
                                # that stood in for it there leaves out the template's own
                                # framing, so it can report room that is not there, the
                                # recall appends a passage too large for the real window,
                                # and the next iteration cannot evict it again because the
                                # current tool exchange is protected.
                                #
                                # So price it exactly instead, here. A failure falls back
                                # to the estimate, which is what this did before.
                                #
                                # Once per TOOL CALL, not once per retrieval: every result
                                # is now sized against this figure, not only a retrieval's,
                                # because a `cat` of a large file overflows the window the
                                # same way a too-large recall does and is likewise
                                # protected from eviction once appended. One tokenizer pass
                                # over the conversation next to running the tool itself.
                                #
                                # Recomputed on EVERY search, not cached for the request.
                                # The count is absolute, and the loop appends the assistant
                                # call and the tool result of every intervening tool to
                                # `conversation`, so a cached figure understates the prompt
                                # by exactly those exchanges and hands the search room that
                                # is already spent. The offset on the other leg is a
                                # framing correction rather than a total, so it stays valid
                                # as the conversation grows.
                                # Unconditionally, not only when the fit dropped nothing.
                                # Gated on `_prompt_token_offset is None`, a request that
                                # HAD compacted priced every later search as a character
                                # estimate plus an offset measured before the loop's own
                                # tool exchanges existed, so a dense ASCII result appended
                                # after that fit -- code, minified JSON, command output --
                                # was undercounted and the recall spent room that was
                                # already gone. The exact count covers the same ground as
                                # the offset leg, since it prices the conversation WITH
                                # the tool catalogue, so the offset stays only as the
                                # fallback for a tokenizer call that raises.
                                _exact_prompt_tokens: Optional[int] = None
                                # Counted WITH a stand-in tool message for the call about
                                # to run, exactly as the admission gate prices it. Qwen
                                # and templates like it render an assistant tool call only
                                # once a `tool` reply follows, so counting `conversation`
                                # as it stands omits this call's own arguments -- and the
                                # result is then handed room those arguments already
                                # occupy. Empty content, because a result can be cut to
                                # its notice but the call itself cannot be unsent.
                                _size_probe: dict = {
                                    "role": "tool",
                                    "name": decision.tool_name,
                                    "content": "",
                                }
                                if decision.tool_call_id:
                                    _size_probe["tool_call_id"] = decision.tool_call_id
                                try:
                                    _exact_prompt_tokens = self.count_chat_tokens(
                                        neutralize_control_markup_in_messages(
                                            messages_without_unpriced_media(
                                                [*conversation, _size_probe]
                                            ),
                                            _markup_cache,
                                            self.markup_profile,
                                        ),
                                        None,
                                        safe_tools,
                                        strict = True,
                                        chat_template_kwargs = _reasoning_kw,
                                    )
                                except Exception:
                                    logger.debug(
                                        "recall budget: exact prompt count failed",
                                        exc_info = True,
                                    )
                                # Dense on the fallback leg only: `_prompt_token_offset`
                                # already carries the fit's exact tokenizer count, so it
                                # needs no correction, while the estimate that stands in
                                # for it undercounts CJK and emoji by about half.
                                # The gate decided to run this call BECAUSE compacting
                                # its own arguments makes the next prompt fit, and it
                                # already counted that prompt exactly. Pricing the result
                                # against the uncompacted conversation instead hands a
                                # large `python`, `terminal` or file call a near-zero
                                # budget and throws its output away, in the one case where
                                # the room is about to be there.
                                _spent = (
                                    _compacted_turn_tokens
                                    if _compact_after_execution and _compacted_turn_tokens
                                    else _exact_prompt_tokens
                                    if _exact_prompt_tokens is not None
                                    else estimate_messages_tokens_dense(
                                        messages_without_unpriced_media(conversation)
                                    )
                                    + (
                                        _prompt_token_offset
                                        if _prompt_token_offset is not None
                                        else estimate_messages_tokens_dense(safe_tools or [])
                                    )
                                )
                                # What ANY result may add before the next prompt is over
                                # budget, for every tool and not just the retrieval ones.
                                # The old cap was a share of the WINDOW and never fell as
                                # the thread filled, so the last result before an overflow
                                # claimed as much as the first, and nothing downstream
                                # recovers: the fit protects the newest turn.
                                if accepts_kwarg(execute_tool, "result_budget_tokens"):
                                    # Split across the calls still to run in this batch,
                                    # and after their arguments. One assistant turn can
                                    # call several tools, and the loop appends each call
                                    # only as it runs it, so `_spent` here knows nothing
                                    # about the rest of the batch: sized as if it were
                                    # alone, the first result can take all the room the
                                    # other calls and their results still need, and the
                                    # finished exchange is protected as the newest turn.
                                    _pending = list(tool_calls or [])[_call_index + 1 :]
                                    _pending_msgs = [
                                        {
                                            "role": "assistant",
                                            "content": json.dumps(_call, default = str),
                                        }
                                        for _call in _pending
                                    ]
                                    # reserve the notice before sharing space among retained results.
                                    if _over_cap:
                                        _pending_msgs.append(
                                            tool_call_limit_nudge(
                                                _over_cap, _MAX_TOOL_CALLS_PER_TURN
                                            )
                                        )
                                    # Measured, not estimated: the estimator charges ASCII
                                    # four characters per token, and a pending call can
                                    # carry base64, minified JSON or a block of code, which
                                    # run nearer one or two. Under-priced, the first result
                                    # is handed room the later ARGUMENTS already occupy.
                                    _pending_args = 0
                                    _pending_exact = True
                                    if _pending_msgs:
                                        try:
                                            _pending_args = self.count_chat_tokens(
                                                _pending_msgs, None, None, strict = True
                                            )
                                        except Exception:
                                            logger.debug(
                                                "result budget: pending call count failed",
                                                exc_info = True,
                                            )
                                            _pending_exact = False
                                            _pending_args = 2 * estimate_messages_tokens_dense(
                                                _pending_msgs
                                            )
                                    # This iteration's cap, not the caller's whole one. A
                                    # recovery turn with 100 of 1000 tokens left had the
                                    # result priced as if 1000 were still to come, which
                                    # reserves room away and can starve a read the request
                                    # had space for.
                                    _result_max_tokens = _iteration_max_tokens
                                    # Auto-compaction off: nothing fits the next prompt to
                                    # `prompt_budget`, whose quarter-window reply reserve
                                    # emptied every result from 75% of the window on (#13784).
                                    # Price to the reply floor the admission gate uses instead,
                                    # only on an exact count of a thread with no unpriced media:
                                    # the quarter reserve is what covers either error.
                                    if (
                                        context_overflow != "truncate_oldest"
                                        and (
                                            _exact_prompt_tokens is not None
                                            or (_compact_after_execution and _compacted_turn_tokens)
                                        )
                                        and _pending_exact
                                        and messages_without_unpriced_media(conversation)
                                        is conversation
                                    ):
                                        _ctx = self._effective_context_length
                                        _requested = (
                                            _iteration_max_tokens
                                            if _iteration_max_tokens and _iteration_max_tokens > 0
                                            else _ctx
                                        )
                                        _reserve = min(_requested, _reply_floor(_ctx))
                                        # Images join the thread after the whole batch, unpriced:
                                        # the ones already collected, and this call's own.
                                        _images = sum(map(len, batch_mcp_images))
                                        if (
                                            mcp_is_image_tool(decision.tool_name)
                                            and image_share is None
                                            and self.is_vision
                                        ):
                                            _images += MCP_MAX_MODEL_IMAGES
                                        if _images:
                                            _reserve += min(
                                                MCP_MAX_TOTAL_MODEL_IMAGES, _images
                                            ) * _mcp_image_tokens(self)
                                        _result_max_tokens = min(
                                            _reserve, _requested, max(1, _ctx // 4)
                                        )
                                    _result_budget = tool_result_budget(
                                        self._effective_context_length,
                                        _result_max_tokens,
                                        _spent + _pending_args,
                                    ) // (len(_pending) + 1)
                                    # A budget at or near zero means the call cannot deliver
                                    # anything: the result is cut to a notice saying it was
                                    # cut, which reads to the model as a fresh failure rather
                                    # than a wall, so it tries again. Observed at a 4096
                                    # window, reading a 2401-byte file back: six of the last
                                    # eight calls priced at zero and returned the same
                                    # 109-char notice, and the turn went on 18 calls.
                                    #
                                    # Room is exactly what compaction reclaims, so spend it
                                    # here before the call rather than after the turn is lost.
                                    if (
                                        _result_budget < _MIN_USEFUL_RESULT_TOKENS
                                        and self._effective_context_length
                                    ):
                                        _roomier, _n_roomier = compact_completed_tool_arguments(
                                            conversation, protect_last = 1
                                        )
                                        if _n_roomier:
                                            conversation[:] = _roomier
                                            try:
                                                # With the same stand-in reply the initial
                                                # sizing used. Without it, this call's own
                                                # arguments vanish from the count on the
                                                # templates that render a call only once a
                                                # reply follows, and the room they occupy
                                                # is handed to the result.
                                                _spent_after = self.count_chat_tokens(
                                                    neutralize_control_markup_in_messages(
                                                        messages_without_unpriced_media(
                                                            [*conversation, _size_probe]
                                                        ),
                                                        _markup_cache,
                                                        self.markup_profile,
                                                    ),
                                                    None,
                                                    safe_tools,
                                                    strict = True,
                                                    chat_template_kwargs = _reasoning_kw,
                                                )
                                            except Exception:
                                                logger.debug(
                                                    "result budget rescue: count failed",
                                                    exc_info = True,
                                                )
                                                _spent_after = None
                                            if _spent_after is not None:
                                                _rescued = tool_result_budget(
                                                    self._effective_context_length,
                                                    # The same cap the first pricing used,
                                                    # or the rescue is measured against a
                                                    # different request from the one it is
                                                    # rescuing.
                                                    _result_max_tokens,
                                                    _spent_after + _pending_args,
                                                ) // (len(_pending) + 1)
                                                logger.info(
                                                    "Result budget for %s was %d; compacted "
                                                    "%d completed call(s) and it is now %d",
                                                    decision.tool_name,
                                                    _result_budget,
                                                    _n_roomier,
                                                    _rescued,
                                                )
                                                _result_budget = max(_result_budget, _rescued)
                                    kwargs["result_budget_tokens"] = _result_budget
                                    _last_result_budget[0] = kwargs["result_budget_tokens"]
                                    # Known before the tool runs, so there is no reason to
                                    # spend calls rediscovering it: the repeat guard below
                                    # would reach the same conclusion three calls later.
                                    _starved_call[0] = _result_budget < _MIN_USEFUL_RESULT_TOKENS
                                if accepts_kwarg(execute_tool, "conversation_budget_tokens"):
                                    if accepts_kwarg(execute_tool, "conversation_token_counter"):
                                        # The admission check estimates a result's size by
                                        # characters, which is optimistic for ASCII that
                                        # tokenises densely: code, minified JSON, hashes and
                                        # command output all run nearer two or three
                                        # characters per token than four. This path has a
                                        # tokenizer, so hand it over and let the check spend
                                        # the budget as exactly as it computes it.
                                        kwargs["conversation_token_counter"] = lambda text: (
                                            self.count_chat_tokens(
                                                [{"role": "tool", "content": text}],
                                                None,
                                                None,
                                                strict = False,
                                            )
                                        )
                                    # The result and reply are protected on the next fit, so
                                    # the result cannot spend their entire shared budget.
                                    kwargs["conversation_budget_tokens"] = _retrieval_budget(
                                        self._effective_context_length,
                                        # This iteration's cap, like the result budget
                                        # beside it. Reserving the caller's whole
                                        # allowance on a continuation that has a fraction
                                        # of it left returns a near-zero budget and drops
                                        # recall the request had room for.
                                        _iteration_max_tokens,
                                        _spent,
                                        reply_returns = True,
                                    )
                            if accepts_output_callback(execute_tool):
                                kwargs["output_callback"] = _output_callback
                            kwargs.update(search_images_kwargs(execute_tool, _decision.tool_name))
                            if image_share is not None:
                                kwargs["mcp_image"] = image_share["image"]
                            return execute_tool(
                                _decision.tool_name,
                                _decision.arguments,
                                **kwargs,
                            )

                        # Without this the handler below re-raises and a bad
                        # argument kills the whole answer. A raising tool is the
                        # tool's failure; the other two loops hand it back.
                        _started_events = (
                            (_pending_context_usage,) if _pending_context_usage is not None else ()
                        )
                        _pending_context_usage = None
                        try:
                            result = yield from stream_tool_execution(
                                _invoke_tool,
                                tool_name = decision.tool_name,
                                tool_call_id = decision.tool_call_id,
                                cancel_event = cancel_event,
                                started_events = _started_events,
                            )
                        except _LlamaStreamCancelled:
                            # Subclasses Exception, so the arm below would eat
                            # the cancel signal. CancelledError needs no arm:
                            # a BaseException passes straight through.
                            raise
                        except Exception as _tool_exc:
                            if cancel_event is not None and cancel_event.is_set():
                                raise
                            logger.exception(
                                "Tool %s raised: %s",
                                decision.tool_name,
                                _tool_exc,
                            )
                            result = f"Error: tool raised an exception: {_tool_exc}"
                        if decision.tool_name in RAG_SEARCH_TOOLS:
                            _kb_search_count += 1
                    # What the result actually cost against what it was allowed. The pair
                    # is the only way to tell a budget that was never delivered from one
                    # that was delivered and ignored, and a 400 one second after a tool
                    # returns cannot distinguish them on its own.
                    logger.info(
                        "tool result: name=%s budget_tokens=%s chars=%d",
                        decision.tool_name,
                        _last_result_budget[0],
                        len(result) if isinstance(result, str) else -1,
                    )
                    # The window priced this result at nothing before the call ran, so the
                    # model is told now rather than after three calls prove it.
                    _result_was_starved = False
                    # The budget says what the window ALLOWED, not what the tool returned.
                    # A short result fits a few tokens completely -- "Created a.py", "OK",
                    # the acknowledgement a mutation gives -- and telling the model that
                    # nothing usable came back invites it to discard a successful write or
                    # do it a second time. So the nudge waits for evidence the result was
                    # actually cut: the truncation notice, or nothing at all.
                    if (
                        _starved_call[0]
                        and isinstance(result, str)
                        and (not result.strip() or _is_window_notice(result))
                    ):
                        _starved_call[0] = False
                        _result_was_starved = True
                        logger.info(
                            "%s ran with a result budget below %d; telling the model the "
                            "window cannot carry the result",
                            decision.tool_name,
                            _MIN_USEFUL_RESULT_TOKENS,
                        )
                        result = _note_before_envelopes(
                            result,
                            decision.tool_name,
                            lambda body: _starved_result_message(decision.tool_name, body),
                        )
                    else:
                        # Cleared either way: it describes THIS call's budget, and leaving
                        # it set would hand the notice to the next call instead.
                        _starved_call[0] = False

                    # ── No-progress guard: the same answer, over and over ──
                    # Keyed on the RESULT, not the arguments. In the turn this came from,
                    # the arguments differed every time (different line ranges of one file)
                    # while the answer was the same 109-char truncation notice, so an
                    # argument-keyed guard would have watched all 18 calls go by. OpenClaw
                    # keys on the result for the same reason and never fires while results
                    # are still changing, which is what makes this safe for polling.
                    if isinstance(result, str):
                        # The arguments are part of the key UNLESS the result is one the
                        # window itself produced. A tool that answers every distinct
                        # mutation with `OK` is making progress, and the nudge below --
                        # which says different arguments will not change the answer -- can
                        # talk it out of the operations it has left. A starved result is
                        # the opposite: it is the notice, not the tool's answer, so the
                        # arguments are exactly what must NOT be part of the key. That was
                        # the observed turn, where 18 calls read different line ranges of
                        # one file and got the same 109-char truncation notice each time.
                        _result_key = (
                            decision.tool_name,
                            hashlib.sha1(result.encode("utf-8", "replace")).hexdigest(),
                            None
                            if _result_was_starved or _is_window_notice(result)
                            else hashlib.sha1(
                                json.dumps(decision.arguments, sort_keys = True, default = str).encode(
                                    "utf-8", "replace"
                                )
                            ).hexdigest(),
                        )
                        if _result_key == _last_tool_result_key[0]:
                            _identical_result_runs[0] += 1
                        else:
                            _last_tool_result_key[0] = _result_key
                            _identical_result_runs[0] = 1
                        if _identical_result_runs[0] >= _MAX_IDENTICAL_TOOL_RESULTS:
                            logger.info(
                                "%s returned the same result %d times; saying so instead of "
                                "letting the turn spend itself repeating it",
                                decision.tool_name,
                                _identical_result_runs[0],
                            )
                            # Told, not silently stopped. The model can still finish the
                            # task another way, and hard-stopping a turn that is otherwise
                            # healthy would trade one dead end for a worse one.
                            result = _note_before_envelopes(
                                result,
                                decision.tool_name,
                                lambda body: _repeated_result_message(
                                    decision.tool_name, _identical_result_runs[0], body
                                ),
                            )
                            _identical_result_runs[0] = 0
                            _last_tool_result_key[0] = None
                    completion = tool_controller.record_result(decision, result)
                    resolved_provisional_tool_call_ids.add(decision.tool_call_id)
                    # A real execution opens the post-tool phase; carrying the pre-tool
                    # stall text over would read the same sentence as a repeat and
                    # swallow the one post-tool nudge.
                    _last_reprompt_text = ""
                    # A tool ran this turn, so it counts against the caller's budget. Loading
                    # MCP tools does not, or a budget of one could never call what it found.
                    if decision.tool_name != "mcp_tool_search":
                        _turn_executed_real_tool = True
                    _forced_choice_resolved = True
                    _tool_end = completion.tool_end_event()
                    yield _tool_end
                    conversation.append(completion.tool_message())
                    if _tool_end.get("tool_call_id"):
                        _tool_card_by_message[id(conversation[-1])] = _tool_end["tool_call_id"]
                    # Parsed only for a result that carries them and a model that will
                    # be shown them; the marker test first, so a text result never asks.
                    _completion_images = (
                        completion.mcp_images()
                        if mcp_images_mentioned_in(completion.result or "") and self.is_vision
                        else []
                    )
                    if _completion_images:
                        batch_mcp_images.append(_completion_images)
                    if _compact_after_execution and decision.tool_call_id:
                        # The promise the gate made when it let this run. Applied here
                        # rather than on the next pass because the next pass may not
                        # come: this turn's own generation is the request that would
                        # otherwise be rejected.
                        _before_len = len(json.dumps(conversation, default = str))
                        conversation[:] = compact_executed_call_arguments(
                            conversation, decision.tool_call_id
                        )
                        logger.info(
                            "Compacted %s arguments after running it: %d -> %d chars",
                            decision.tool_name,
                            _before_len,
                            len(json.dumps(conversation, default = str)),
                        )
                        # The compaction rebuilds the messages rather than mutating them,
                        # so the local handle now points at a dict that is no longer in
                        # `conversation`. One assistant turn can carry several calls, and
                        # the next one in the batch appends its tool_call to this handle
                        # while its RESULT goes to `conversation`: the model would receive
                        # a tool result answering a call it cannot see. Rebind to the live
                        # message carrying this call.
                        # Backwards: generated ids restart at `call_0` every turn, so a
                        # forward scan binds to the OLDEST turn carrying the id and the
                        # rest of this batch is appended to history instead of to the live
                        # exchange. The call being compacted is always the current one.
                        for _msg in reversed(conversation):
                            if _msg.get("role") != "assistant":
                                continue
                            if any(
                                (_tc or {}).get("id") == decision.tool_call_id
                                for _tc in (_msg.get("tool_calls") or [])
                            ):
                                assistant_msg = _msg
                                break

                    if _forced_tool_call_pending:
                        _forced_tool_call_pending = False

                # On the turn that ends the loop no tools are offered again, so the notice
                # must not ask for a retry and rides the tool result like the budget nudge.
                _over_cap_final = bool(_over_cap) and (
                    tool_controller.force_final_answer
                    or not tool_controller.active_tools()
                    or (_turn_executed_real_tool and _tool_iters_done + 1 >= max_tool_iterations)
                    # The tool-iteration cap is not the only way out: the loop also stops on
                    # the outer iteration range, which no-op and continuation turns consume
                    # without advancing _tool_iters_done. Missing it asks for a retry and then
                    # tells the model not to call any tools, in adjacent user turns.
                    or iteration + 1 >= max_tool_iterations + _extra + _continuation_credits
                )
                _final_over_cap = _over_cap if _over_cap_final else []
                if _over_cap_final:
                    _over_cap = []
                if _over_cap:
                    deferred_noop_msgs.append(
                        tool_call_limit_nudge(
                            _over_cap,
                            _MAX_TOOL_CALLS_PER_TURN,
                            unavailable_tools = {
                                _limit_decision.tool_name
                                for _call in _over_cap
                                if (_limit_decision := tool_controller.prepare_call(_call)).action
                                in ("disabled", "render_html_repeat")
                            },
                        )
                    )
                # A mixed execute/no-op batch already has a real tool result, so keeping the
                # feedback with that result beats appending a newer user turn, which makes
                # templates hide this turn's structured reasoning. Only when the result is
                # the SAME tool the feedback is about: templates label the whole block with
                # the result's own tool name (gemma-4.jinja resolves tool_call_id -> name and
                # wraps the body), so folding a note about tool A into tool B's result reads
                # as B's own output. Then the user turn is the lesser loss. A retry notice for
                # skipped calls is never folded: read as the tail of a result, the calls are not re-issued.
                _fold_target_matches = (
                    not _over_cap
                    and len(deferred_noop_tools) == 1
                    and bool(conversation)
                    and conversation[-1].get("role") == "tool"
                    and conversation[-1].get("name") in deferred_noop_tools
                )
                if deferred_noop_msgs and assistant_appended and _fold_target_matches:
                    if not _attach_internal_feedback_to_tool_result(
                        deferred_nudge_text(deferred_noop_msgs)
                    ):
                        append_deferred_nudges(conversation, deferred_noop_msgs)
                elif deferred_noop_msgs and assistant_appended:
                    append_deferred_nudges(conversation, deferred_noop_msgs)
                else:
                    # A blank trace is not a turn: reuse the same emptiness test as
                    # the field above so a whitespace-only split never appends an
                    # empty assistant message.
                    if not assistant_appended and (
                        content_text or assistant_msg.get("reasoning_content") or _resumed_reasoning
                    ):
                        if assistant_msg.get("reasoning_content") or _resumed_reasoning:
                            # Whole thought: the nudge below hides the original's field.
                            assistant_msg["content"] = neutralize_control_markup(
                                _resumed_reasoning + reasoning_accum,
                                self.markup_profile,
                            )
                            _continued_partial = (
                                trailing_assistant_text(conversation)
                                if continue_final_message
                                else None
                            )
                            if _continued_partial and not _continued_partial.endswith("\n"):
                                assistant_msg["content"] = f"\n{assistant_msg['content']}"
                            if content_text:
                                assistant_msg["content"] += f"\n{content_text}"
                            assistant_msg.pop("reasoning_content", None)
                        append_assistant_turn(
                            conversation,
                            assistant_msg,
                            continue_final_message = _merge_into_partial,
                        )
                        if _resumed_reasoning:
                            conversation[-1].pop("reasoning_content", None)
                        assistant_appended = True
                    append_deferred_nudges(conversation, deferred_noop_msgs)
                if _final_over_cap:
                    _limit_text = tool_call_limit_nudge(
                        _final_over_cap, _MAX_TOOL_CALLS_PER_TURN, final = True
                    )["content"]
                    # Same rule as _fold_target_matches above: a template labels the folded
                    # block with the result's own tool name, so a note about tool A inside
                    # tool B's result reads as B's output. Fold only when every skipped call
                    # belongs to the result's own tool.
                    _limit_names = {
                        (_tc.get("function") or {}).get("name") for _tc in _final_over_cap
                    }
                    _limit_foldable = (
                        bool(conversation)
                        and conversation[-1].get("role") == "tool"
                        and _limit_names == {conversation[-1].get("name")}
                    )
                    if not (
                        _limit_foldable and _attach_internal_feedback_to_tool_result(_limit_text)
                    ):
                        if deferred_noop_msgs and conversation[-1].get("role") == "user":
                            conversation[-1] = {
                                **conversation[-1],
                                "content": f"{conversation[-1]['content']}\n\n{_limit_text}",
                            }
                        else:
                            conversation.append({"role": "user", "content": _limit_text})

                if batch_mcp_images and self.is_vision:
                    # One block after the whole batch. With a single result "the tool
                    # call above" is exact; with several it names whichever ran last,
                    # which may have returned no picture at all, so the block says so
                    # instead -- the external loop's rule.
                    _batch_results = sum(
                        1
                        for m in conversation[batch_conversation_start:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    )
                    append_mcp_image_turn(
                        conversation,
                        batch_mcp_images,
                        per_result = True,
                        owned = loop_mcp_image_parts,
                        lead = MCP_DETACHED_IMAGE_TURN_TEXT
                        if _batch_results != 1
                        else MCP_IMAGE_TURN_TEXT,
                    )

                # Close provisional cards not resolved by execution/no-op handling.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "",
                            "provenance": provisional_tool_provenance(_pname),
                        }

                # No call in the batch ran (denied, duplicate, capped), so nothing was waiting on it.
                if _pending_context_usage is not None:
                    yield _pending_context_usage
                    _pending_context_usage = None

                # Clear tool status badge before next generation/final pass.
                yield {"type": "status", "text": ""}
                if tool_controller.force_final_answer or not tool_controller.active_tools():
                    _append_budget_exhausted_nudge = False
                    break
                # Count only real tool turns against the cap so reserved re-prompt slots can't become
                # extra tool rounds; a no-op correction turn doesn't consume budget (GGUF parity).
                if _turn_executed_real_tool:
                    _tool_iters_done += 1
                    if _browser_turns:
                        _post_tool_reprompts = 0
                    if _tool_iters_done >= max_tool_iterations:
                        break
                continue

            except _LlamaStreamCancelled:
                _held = _cancelled_hold_text()
                if _held:
                    yield {"type": "content", "text": _held}
                return
            except httpx.ConnectError:
                # Mark unresolved provisional cards as failed before raising.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: lost connection to llama-server before the tool call completed.",
                            "provenance": provisional_tool_provenance(_pname),
                        }
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                # Same cleanup for other mid-iteration failures.
                for _pid, _pname in provisional_started_tool_calls.items():
                    if _pid not in resolved_provisional_tool_call_ids:
                        resolved_provisional_tool_call_ids.add(_pid)
                        yield {
                            "type": "tool_end",
                            "tool_name": _pname,
                            "tool_call_id": _pid,
                            "result": "Error: the tool call was interrupted before it completed.",
                            "provenance": provisional_tool_provenance(_pname),
                        }
                raise

        # ── Tool iteration cap reached -- synthesize final answer ──
        # The model used all iterations without a final text response. Nudge
        # the final streaming pass to produce a useful answer instead of
        # continuing to request tools.
        if max_tool_iterations > 0 and _append_budget_exhausted_nudge:
            if not _attach_internal_feedback_to_tool_result(BUDGET_EXHAUSTED_NUDGE):
                # Ride the trailing user turn (over-cap notice or promoted image): two in a row break Gemma.
                if conversation[-1].get("role") == "user":
                    conversation[-1] = {
                        **conversation[-1],
                        "content": with_appended_text(
                            conversation[-1]["content"], f"\n\n{BUDGET_EXHAUSTED_NUDGE}"
                        ),
                    }
                else:
                    conversation.append({"role": "user", "content": BUDGET_EXHAUSTED_NUDGE})

        while _model_note_events:
            yield _model_note_events.pop(0)
        # Clear status.
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context.
        _final_tools = _last_sent_tools
        from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        _final_max_tokens = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        _final_preflight_context_length = None
        _final_preflight_succeeded = False
        if context_overflow == "truncate_oldest" and self._effective_context_length:
            _final_preflight_context_length = self._effective_context_length
            try:
                _before_final_fit = conversation
                # Withheld, not `tools`: this is the synthesised final answer and
                # `stream_payload` below sends tool_choice "none", so the request's
                # catalogue answers a question it does not pose and would let a NEW epoch
                # start on the one pass that can never call `search_conversation`.
                # Replaying an epoch already in force is unaffected.
                _final_can_reset = _can_reset_epoch(
                    thread_id, _backend_supports_tools(self), tools_withheld = True
                )
                _final_sticky, _final_sticky_is_checkpoint = (
                    (0, True)
                    if _sticky_boundary_applied
                    else _sticky_compaction_state(
                        thread_id,
                        _request_branch,
                        context_policy = context_policy,
                        can_reset = _final_can_reset,
                        compaction_headroom_ratio = compaction_headroom_ratio,
                    )
                )
                conversation, truncation = _fit_with_instruction_pins(
                    conversation,
                    context_length = self._effective_context_length,
                    max_tokens = _final_max_tokens,
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            messages_without_unpriced_media(fitted), None, self.markup_profile
                        ),
                        None,
                        _final_tools,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = estimate_message_tokens_without_unpriced_media,
                    anchor_ids = _rolling_anchor_ids,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _final_can_reset,
                    reserve_tokens = _conversation_recall_reserve(thread_id),
                    sticky_dropped = _final_sticky,
                    sticky_is_checkpoint = _final_sticky_is_checkpoint,
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                _sticky_boundary_applied = True
                if truncation:
                    _recalled = _archive_and_recall(
                        conversation,
                        _before_final_fit,
                        branch_messages = _extend_live_branch(_before_final_fit),
                        thread_id = thread_id,
                        # Inline: under tool_choice "none" a forged tool role breaks strict templates.
                        style = "inline",
                        # Checkpoint mode recalls once, on the turn that reset; rolling has no
                        # such key and keeps recalling on every turn that evicted anything.
                        force_recall = bool(truncation.get("checkpoint_started", True)),
                        # A rescued refusal may evict messages; archive them without recall.
                        recall_done = _conversation_recall_done or not truncation["fits"],
                        # The synthesized final answer never returns to the prompt.
                        recall_budget_tokens = _retrieval_budget(
                            self._effective_context_length,
                            max_tokens,
                            truncation.get("prompt_tokens_after") or 0,
                        ),
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                messages_without_unpriced_media(fitted),
                                None,
                                self.markup_profile,
                            ),
                            None,
                            _final_tools,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                        ),
                    )
                    conversation = _recalled["conversation"]
                    truncation = {**truncation, **_recalled["counts"]}
                    if _recalled["recalled"]:
                        _conversation_recall_done = True
                        for _message in _recalled["anchored"]:
                            _rolling_anchor_ids.add(id(_message))
                        for _ev in _recalled["events"]:
                            yield _ev
                if truncation and _records_boundary(truncation):
                    truncation = {
                        **truncation,
                        **_boundary_metadata(
                            conversation, _request_branch, compaction_headroom_ratio
                        ),
                    }
                # `fits` False too: it carries the diagnosis the client needs to explain
                # WHY. Otherwise the user only sees llama-server's error, which reports
                # the whole conversation's size and advises shortening it even when the
                # single message just sent is the part that does not fit.
                if truncation:
                    yield {"type": "context_truncated", **truncation}
                _final_preflight_succeeded = True
            except Exception as exc:
                logger.warning("Could not preflight the rolling context window: %s", exc)

        # The loop's callback fires at the TOP of a round, so the breaks leading here (the
        # tool-iteration cap, a controller turning tools off) leave the assistant turn,
        # its tool results and any nudge appended after the last re-cost -- making this
        # final pass the largest request of the run and the one the pool never heard
        # about. Here rather than at the breaks: the recall above can rebind
        # `conversation`, and every path reaches this point with the list about to be sent.
        if on_conversation_grew is not None:
            try:
                on_conversation_grew(conversation)
            except Exception:  # accounting must never break a run in progress
                logger.debug("tool loop final recost failed", exc_info = True)

        stream_payload = {
            "messages": neutralize_control_markup_in_messages(
                conversation, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if logit_bias:
            stream_payload["logit_bias"] = logit_bias
        if _final_tools:
            stream_payload["tools"] = llama_grammar_tools(_final_tools)
            stream_payload["tool_choice"] = "none"
        if _reasoning_kw is not None:
            stream_payload["chat_template_kwargs"] = _reasoning_kw
        if thinking_budget_tokens is not None:
            stream_payload["thinking_budget_tokens"] = thinking_budget_tokens
        stream_payload["max_tokens"] = _final_max_tokens
        if stop:
            stream_payload["stop"] = stop
        _apply_seeded_llama_request(stream_payload, seed, reuse_prompt_cache = _prompt_sent)
        stream_payload["stream_options"] = {"include_usage": True}

        # Progress events feed the first-token deadline; timings stay opt-in.
        stream_payload["return_progress"] = True
        if _perf_callback_wants_timings(perf_callback):
            stream_payload["timings_per_token"] = True

        _final_respawn_truncations: list[dict] = []
        # Messages appended to the payload but absent from `conversation`.
        _refit_tail: list = []
        _refit_tail_merged = False

        def _record_refit_tail(committed: list, appended: list, merged: bool) -> None:
            """Track appended messages that a respawn refit must replay."""
            nonlocal _refit_tail, _refit_tail_merged
            _live = {id(_message) for _message in committed}
            _new = [_message for _message in appended if id(_message) in _live]
            if len(_new) != len(appended):
                # An eviction split the tail, so fall back to `conversation` alone.
                _refit_tail = []
                _refit_tail_merged = False
                return
            if not merged:
                _refit_tail = _refit_tail + _new
            elif _refit_tail:
                _refit_tail = _refit_tail[:-1] + _new
            else:
                _refit_tail = _new
                _refit_tail_merged = True

        def _refit_final_after_respawn() -> None:
            nonlocal conversation
            _before_respawn_fit = conversation
            if (
                _final_preflight_context_length is None
                or not self._effective_context_length
                or (
                    _final_preflight_succeeded
                    and _final_preflight_context_length == self._effective_context_length
                )
            ):
                return
            if max_tokens is None:
                stream_payload["max_tokens"] = self._effective_context_length
            try:
                conversation, truncation = _fit_with_instruction_pins(
                    conversation,
                    context_length = self._effective_context_length,
                    max_tokens = max_tokens,
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            messages_without_unpriced_media(fitted), None, self.markup_profile
                        ),
                        None,
                        _final_tools,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = estimate_message_tokens_without_unpriced_media,
                    anchor_ids = _rolling_anchor_ids,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _can_reset_epoch(
                        thread_id,
                        _backend_supports_tools(self),
                        # The final pass again, so again tool_choice is "none".
                        tools_withheld = True,
                    ),
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                if truncation:
                    # Archive only; see the iteration respawn refit above.
                    truncation.update(
                        _archive_and_recall(
                            conversation,
                            _before_respawn_fit,
                            thread_id = thread_id,
                            style = "inline",
                            recall_done = True,
                            branch_messages = _extend_live_branch(conversation),
                        )["counts"]
                    )
                # Recorded rather than only forwarded, for the same reason as the
                # iteration refit above.
                from core.inference import context_refusal  # noqa: PLC0415

                context_refusal.record_fit(truncation)
                # Copy because neutralization may return `conversation` itself.
                _refit_messages = list(
                    neutralize_control_markup_in_messages(conversation, None, self.markup_profile)
                )
                if _refit_tail:
                    # A merged tail replaces the trailing assistant prefill.
                    if _refit_tail_merged and trailing_assistant_text(_refit_messages) is not None:
                        _refit_messages[-1:] = _refit_tail
                    else:
                        _refit_messages.extend(_refit_tail)
                stream_payload["messages"] = _refit_messages
                if _refit_tail:
                    # Recheck the restored tail against the replacement window.
                    _refit_continue = bool(stream_payload.get("continue_final_message"))
                    if not _continuation_would_be_served(
                        stream_payload["messages"], _refit_continue
                    ):
                        # Protect the current user turn and its recovery tail.
                        _refit_protect = {id(_message) for _message in _refit_tail}
                        for _message in reversed(
                            _refit_messages[: len(_refit_messages) - len(_refit_tail)]
                        ):
                            _refit_protect.add(id(_message))
                            if _message.get("role") == "user":
                                break
                        _refit_evicted = _evict_until_it_fits(
                            stream_payload["messages"],
                            _final_tools,
                            stream_payload.get("chat_template_kwargs"),
                            _refit_continue,
                            _refit_protect,
                        )
                        if _refit_evicted is not None:
                            # Nothing else is safe to evict.
                            stream_payload["messages"] = _refit_evicted
                if truncation:
                    if _records_boundary(truncation):
                        truncation.update(
                            _boundary_metadata(
                                conversation, _request_branch, compaction_headroom_ratio
                            )
                        )
                    _final_respawn_truncations.append(truncation)
            except Exception as exc:
                logger.warning("Could not refit rolling context after respawn: %s", exc)
                raise

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""
        # Track content separately to preserve literal `</think>` in the answer.
        answer_text = ""
        _prov_entry = None
        _final_reasoning_started_at: Optional[float] = None
        _final_reasoning_summary_emitted = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None
        _stream_done = False

        # The tool loop's continuation does not reach here: this generation runs AFTER
        # the loop breaks, so a turn that spends its whole tool budget and is then cut
        # off mid-answer had nothing to recover it. Observed live: 25 tool calls, 22387
        # tokens, stopped inside drawBird() with the game left half-written.
        _final_length_continuations = 0
        _continue_final = False
        _final_replayed_chars = 0

        def _remaining_output_budget() -> "Optional[int]":
            """What is left of a cap the CALLER set, or None when they set none.

            `finish_reason: "length"` does not say which wall was hit. A caller asking for
            at most 100 completion tokens gets it at their own cap, and continuing twice
            more returned roughly 300 -- over the limit the API promised, for latency and
            tokens nobody asked for. Only the context wall deserves a continuation, and a
            caller who set a cap gets the remainder of it rather than a fresh one.

            None means no explicit cap: `max_tokens` unset, or set to the whole window,
            which is what the backend substitutes for "Max" and is indistinguishable from
            unset. There the length stop IS the context wall, which is the case this
            continuation exists for.
            """
            if max_tokens is None:
                return None
            window = self._effective_context_length or 0
            if window and max_tokens >= window:
                return None
            # This attempt's own cost. A server that reported usage gives it exactly;
            # without one, a `length` stop still means the attempt produced the whole
            # allowance it was given, so charge that. Guessing LOW here is what lets the
            # caller's cap be overrun, which is the defect this exists to prevent.
            _this_attempt = (
                _backfill_usage_from_timings(_metadata_usage, _metadata_timings) or {}
            ).get("completion_tokens", 0) or int(stream_payload.get("max_tokens") or 0)
            return max(0, max_tokens - _accumulated_completion_tokens - _this_attempt)

        def _continuation_would_be_served(
            candidate_messages,
            continue_flag: bool,
            reasoning_kw = _UNSET,
        ) -> bool:
            """Whether the retry's own prompt still fits, now that the partial is in it.

            The first pass reached `finish_reason: length` by consuming the physical
            context, so appending everything it produced makes the next prompt roughly
            context-sized. Reopening the stream without asking gets it rejected before a
            single extra token arrives, which is worse than stopping: the user keeps the
            partial either way, and one of the two paths also shows an error.

            A count that cannot be taken is not a refusal. Failing open here restores the
            behaviour this continuation had before the check, which is the safe side.
            """
            if not self._effective_context_length:
                return True
            try:
                _tokens = self.count_chat_tokens(
                    messages_without_unpriced_media(candidate_messages),
                    None,
                    _final_tools,
                    strict = True,
                    chat_template_kwargs = (
                        stream_payload.get("chat_template_kwargs")
                        if reasoning_kw is _UNSET
                        else reasoning_kw
                    ),
                    continue_final_message = continue_flag,
                )
            except Exception:
                logger.debug("final continuation: prompt count failed", exc_info = True)
                return True
            # NOT `turn_is_servable`: that charges `_RESULT_NOTICE_RESERVE`, the room a
            # truncated TOOL RESULT needs for its notice, and a continuation has no tool
            # result. Borrowing it here refuses continuations that would have been served
            # and produced text, which is the working behaviour this check exists beside,
            # not instead of. llama-server admits a prompt on size alone, so that is the
            # line, minus only enough room to answer in.
            _floor = _reply_floor(self._effective_context_length)
            if _tokens + _floor <= self._effective_context_length:
                return True
            logger.info(
                "Final continuation would not be served (%d tokens of a %d-token "
                "window); keeping the partial answer instead",
                _tokens,
                self._effective_context_length,
            )
            return False

        # Where the visible answer stood when THIS attempt began. `has_content_tokens`
        # and `_last_emitted` are cumulative across continuations by design, so without a
        # per-attempt mark a continuation that emitted nothing but reasoning still looked
        # like another truncated visible answer: it replayed an empty suffix with thinking
        # left on instead of taking the reasoning-only recovery.
        _attempt_started_at = ""
        while True:
            try:
                with self._open_chat_stream_with_respawn_retry(
                    stream_payload,
                    cancel_event,
                    on_respawn = _refit_final_after_respawn,
                ) as (
                    response,
                    first_token_deadline,
                ):
                    for truncation in _final_respawn_truncations:
                        yield {"type": "context_truncated", **truncation}
                    buffer = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response,
                        cancel_event,
                        first_token_deadline = first_token_deadline,
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if (
                                        _final_reasoning_started_at is not None
                                        and not _final_reasoning_summary_emitted
                                    ):
                                        _final_reasoning_summary_emitted = True
                                        yield _reasoning_summary_event(_final_reasoning_started_at)
                                    if has_content_tokens:
                                        cumulative += "</think>"
                                        _prov_entry = None
                                        yield {
                                            "type": "content",
                                            "text": _strip_tool_markup(cumulative, final = True),
                                        }
                                    else:
                                        cumulative = _finalize_reasoning_only_cumulative(
                                            cumulative,
                                            reasoning_text,
                                            _metadata_finish_reason,
                                            promote_reasoning_only,
                                        )
                                        # The block is closed now, so the flag must say
                                        # so. Left set, the thinking-off retry below
                                        # closes it a SECOND time on its first content
                                        # token: `<think>...</think></think>answer`, and
                                        # the route parses only the leading pair, so the
                                        # stray tag is shown to the user as answer text.
                                        in_thinking = False
                                        _prov_entry = None
                                        yield {"type": "content", "text": cumulative}
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])

                                _report_live_llama_timings(perf_callback, chunk_data)
                                # Capture server timings/usage from final chunks.
                                _chunk_timings = chunk_data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = chunk_data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                # See the note on the first stream loop.
                                _stream_error = stream_error_from_chunk(chunk_data)
                                if _stream_error is not None:
                                    raise _stream_error
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    _fr = choices[0].get("finish_reason")
                                    if _fr:
                                        _metadata_finish_reason = _fr

                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        if _final_reasoning_started_at is None:
                                            _final_reasoning_started_at = time.monotonic()
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            if reasoning_provenance is not None and not cumulative:
                                                reasoning_provenance["wrapped"] = (
                                                    reasoning_provenance.get("wrapped", 0) + 1
                                                )
                                                _prov_entry = {"len": 0}
                                                reasoning_provenance.setdefault("wraps", []).append(
                                                    _prov_entry
                                                )
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        if _prov_entry is not None:
                                            _prov_entry["len"] += len(reasoning)
                                        yield {"type": "content", "text": cumulative}

                                    token = delta.get("content", "")
                                    if token:
                                        if (
                                            _final_reasoning_started_at is not None
                                            and not _final_reasoning_summary_emitted
                                        ):
                                            _final_reasoning_summary_emitted = True
                                            yield _reasoning_summary_event(
                                                _final_reasoning_started_at
                                            )
                                        has_content_tokens = True
                                        answer_text += token
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                            _prov_entry = None
                                        cumulative += token
                                        cleaned = (
                                            _final_answer_stripper.strip(cumulative)
                                            if auto_heal_tool_calls
                                            else cumulative
                                        )
                                        # Emit only when cleaned text grows (monotonic).
                                        if len(cleaned) > len(_last_emitted):
                                            _last_emitted = cleaned
                                            yield {"type": "content", "text": cleaned}
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                        if _stream_done:
                            break  # exit outer for
                    # Cut off mid-answer with real text on screen: resume it rather than
                    # stop here. Same rule as the in-loop path -- the partial goes back to
                    # be EXTENDED, thinking is left alone because it was not the problem,
                    # and an answer that is mostly an echo of itself is left as it is.
                    _shown_this_attempt = _last_emitted[len(_attempt_started_at) :]
                    if (
                        _metadata_finish_reason == "length"
                        and has_content_tokens
                        and _shown_this_attempt.strip()
                        and _final_length_continuations < _MAX_LENGTH_CONTINUATIONS
                        and not is_repetition_dominated(_last_emitted)
                    ):
                        from core.inference.chat_template_helpers import (  # noqa: PLC0415
                            append_assistant_turn as _append_assistant_turn,
                        )

                        _final_length_continuations += 1
                        logger.info(
                            "Final answer cut off at %d chars; continuing %d/%d",
                            len(_last_emitted),
                            _final_length_continuations,
                            _MAX_LENGTH_CONTINUATIONS,
                        )
                        # Built as a CANDIDATE, committed only once another stream is
                        # certain to run. Mutating the payload first and folding the usage
                        # in first meant a refused continuation had already spent both:
                        # `_build_metadata_event` adds the same usage again, so the reply
                        # reported roughly double the tokens it really cost.
                        #
                        # Only what is NEW since the last replay goes back. `_last_emitted`
                        # is cumulative across attempts, and the helper PREPENDS the
                        # trailing assistant text, so handing it the cumulative value a
                        # second time yields "fragment1 + fragment1 + continuation1".
                        _candidate_messages = list(stream_payload["messages"])
                        _merged_f = trailing_assistant_text(_candidate_messages) is not None
                        _append_assistant_turn(
                            _candidate_messages,
                            {
                                "role": "assistant",
                                "content": _last_emitted[_final_replayed_chars:],
                            },
                            continue_final_message = True,
                        )
                        # The replay is text the MODEL just produced, and the first payload
                        # neutralized everything it carried. Sent raw, a template delimiter
                        # inside it -- printed code being the obvious case -- is read back
                        # as chat structure. Neutralized before it is counted as well, so
                        # the admission number describes the payload actually sent.
                        _candidate_messages = neutralize_control_markup_in_messages(
                            _candidate_messages, None, self.markup_profile
                        )
                        _continuation_tail = _candidate_messages[-1:]
                        _next_cap = _remaining_output_budget()
                        _served = _next_cap != 0 and _continuation_would_be_served(
                            _candidate_messages, True
                        )
                        if _next_cap != 0 and not _served:
                            # Same reason as the in-loop path: this is the last decision
                            # of the turn, so a candidate one eviction short of servable
                            # gets that eviction rather than being abandoned.
                            _evicted_f = _evict_until_it_fits(
                                _candidate_messages,
                                _final_tools,
                                stream_payload.get("chat_template_kwargs"),
                                True,
                            )
                            if _evicted_f is not None and _continuation_would_be_served(
                                _evicted_f, True
                            ):
                                _candidate_messages = _evicted_f
                                _served = True
                        if _served:
                            stream_payload["messages"] = _candidate_messages
                            _record_refit_tail(_candidate_messages, _continuation_tail, _merged_f)
                            _final_replayed_chars = len(_last_emitted)
                            stream_payload["continue_final_message"] = True
                            stream_payload["add_generation_prompt"] = False
                            if _next_cap is not None:
                                stream_payload["max_tokens"] = _next_cap
                            # Folded in only now, else the reported usage counts the last
                            # fragment alone.
                            _fu_f = (
                                _backfill_usage_from_timings(_metadata_usage, _metadata_timings)
                                or {}
                            )
                            _accumulated_completion_tokens += _fu_f.get("completion_tokens", 0)
                            _it_f = _metadata_timings or {}
                            _accumulated_predicted_ms += _it_f.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_f.get("predicted_n", 0)
                            _stream_done = False
                            _metadata_finish_reason = None
                            # Per attempt, like the finish reason. This attempt's usage has
                            # just been folded into the accumulator, so leaving it here lets
                            # a next attempt that reports none be charged these numbers a
                            # second time -- and the cap read off that total can then look
                            # spent when it is not.
                            _metadata_usage = None
                            _metadata_timings = None
                            # The next attempt starts from what is on screen now, so a
                            # continuation that shows nothing new is judged on its own.
                            _attempt_started_at = _last_emitted
                            _continue_final = True
                            # NO blank status here, unlike the in-loop re-prompt. An empty
                            # status is the route's iteration boundary and resets its
                            # `prev_text` cursor, but this pass keeps `cumulative` across
                            # attempts by design: the retry's first content event carries
                            # the whole prefix again, and diffed from a cursor that has
                            # just been cleared the client is sent the partial twice.
                            yield {
                                "type": "status",
                                "text": _CONTINUE_TRUNCATED_ANSWER_STATUS,
                            }
                        else:
                            _meta = _build_metadata_event(
                                _metadata_usage, _metadata_timings, _metadata_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                    elif (
                        _metadata_finish_reason == "length"
                        and not _shown_this_attempt.strip()
                        and reasoning_text.strip()
                    ):
                        # The other half of the same failure, and the one the in-loop
                        # continuation cannot reach: this pass runs AFTER the loop has
                        # broken, so a turn that spends its last tool call here and then
                        # thinks until the window is gone renders as an empty message.
                        # `force_final_answer` from a one-shot tool lands here too.
                        if _final_length_continuations < _MAX_LENGTH_CONTINUATIONS:
                            from core.inference.chat_template_helpers import (  # noqa: PLC0415
                                append_assistant_turn as _append_assistant_turn,
                            )

                            _final_length_continuations += 1
                            logger.info(
                                "Final pass ended inside its reasoning (%d chars); "
                                "continuing %d/%d with thinking off",
                                len(reasoning_text),
                                _final_length_continuations,
                                _MAX_LENGTH_CONTINUATIONS,
                            )
                            # Candidate first, committed only if it will actually run.
                            # See the answer continuation above: spending the payload and
                            # the usage before the decision double-counted both.
                            _candidate_r = list(stream_payload["messages"])
                            _merged_r = (
                                bool(stream_payload.get("continue_final_message"))
                                and trailing_assistant_text(_candidate_r) is not None
                            )
                            _append_assistant_turn(
                                _candidate_r,
                                {
                                    "role": "assistant",
                                    "content": _unfinished_thought_progress(reasoning_text),
                                },
                                continue_final_message = bool(
                                    stream_payload.get("continue_final_message")
                                ),
                            )
                            _candidate_r.append(
                                {"role": "user", "content": _continue_after_length_message()}
                            )
                            # Same reason as the answer continuation above: the progress
                            # note quotes the model's own reasoning.
                            _candidate_r = neutralize_control_markup_in_messages(
                                _candidate_r, None, self.markup_profile
                            )
                            # The retry runs with thinking OFF, so it has to be ADMITTED
                            # with thinking off too. Counting the candidate under the
                            # original thinking-on kwargs prices a different rendered
                            # prompt from the one about to be sent, which refuses a retry
                            # that would have fit or admits one llama-server then rejects.
                            _off_kw = self._request_reasoning_kwargs(False, None, preserve_thinking)
                            _next_cap_r = _remaining_output_budget()
                            # Protect the current user turn and recovery tail.
                            _recovery_tail = _candidate_r[-2:]
                            _recovery_protect = {id(_message) for _message in _recovery_tail}
                            for _message in reversed(_candidate_r[:-2]):
                                _recovery_protect.add(id(_message))
                                if _message.get("role") == "user":
                                    break
                            _served_r = _next_cap_r != 0 and _continuation_would_be_served(
                                _candidate_r, False, _off_kw
                            )
                            if _next_cap_r != 0 and not _served_r:
                                _evicted_r = _evict_until_it_fits(
                                    _candidate_r, _final_tools, _off_kw, False, _recovery_protect
                                )
                                if _evicted_r is not None and _continuation_would_be_served(
                                    _evicted_r, False, _off_kw
                                ):
                                    _candidate_r = _evicted_r
                                    _served_r = True
                            if _served_r:
                                stream_payload["messages"] = _candidate_r
                                # Preserve both recovery messages across a respawn.
                                _record_refit_tail(_candidate_r, _recovery_tail, _merged_r)
                                # User-ended recovery invalidates continuation flags.
                                stream_payload.pop("continue_final_message", None)
                                stream_payload.pop("add_generation_prompt", None)
                                if _next_cap_r is not None:
                                    stream_payload["max_tokens"] = _next_cap_r
                                if _off_kw is not None:
                                    stream_payload["chat_template_kwargs"] = _off_kw
                                else:
                                    stream_payload.pop("chat_template_kwargs", None)
                                _fu_r = (
                                    _backfill_usage_from_timings(_metadata_usage, _metadata_timings)
                                    or {}
                                )
                                _accumulated_completion_tokens += _fu_r.get("completion_tokens", 0)
                                _it_r = _metadata_timings or {}
                                _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                                _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                                _stream_done = False
                                _metadata_finish_reason = None
                                # Per attempt, like the finish reason. This attempt's usage has
                                # just been folded into the accumulator, so leaving it here lets
                                # a next attempt that reports none be charged these numbers a
                                # second time -- and the cap read off that total can then look
                                # spent when it is not.
                                _metadata_usage = None
                                _metadata_timings = None
                                # The next attempt starts from what is on screen now, so a
                                # continuation that shows nothing new is judged on its own.
                                _attempt_started_at = _last_emitted
                                _continue_final = True
                                # Same reason as the answer continuation above: clearing
                                # the route's cursor here replays what has already been
                                # streamed, reasoning included.
                                yield {
                                    "type": "status",
                                    "text": _CONTINUE_AFTER_LENGTH_STATUS,
                                }
                            else:
                                yield {"type": "status", "text": ""}
                                # Content events on this stream are CUMULATIVE, so an
                                # explanation sent on its own replaces whatever is on
                                # screen. Once a previous attempt has shown an answer,
                                # keeping it is worth more than the advice: the partial
                                # already offers Continue.
                                if not _last_emitted.strip():
                                    yield {
                                        "type": "content",
                                        # Same distinction the in-loop give-up makes. A
                                        # spent output cap is not the context window, and
                                        # this text reaches the client as ordinary
                                        # content, so naming the wrong lever here is the
                                        # last word the user gets.
                                        "text": (
                                            _reasoning_cap_spent_message(max_tokens)
                                            if _next_cap_r == 0
                                            else _thinking_exhausted_message(
                                                self._effective_context_length
                                            )
                                        ),
                                    }
                                _meta = _build_metadata_event(
                                    _metadata_usage,
                                    _metadata_timings,
                                    _metadata_finish_reason,
                                )
                                if _meta is not None:
                                    yield _meta
                        else:
                            # Allowance spent and still nothing on screen. Saying so is
                            # the point: an empty message gives the user nothing to act on.
                            #
                            # Which allowance, re-read here rather than assumed: the last
                            # permitted attempt may itself have finished off an explicit
                            # Max Tokens, and this branch is reached with no decision
                            # about the cap ever having been made.
                            _cap_left_g = _remaining_output_budget()
                            yield {"type": "status", "text": ""}
                            # As above: cumulative content, so this must not overwrite an
                            # answer an earlier attempt already put on screen.
                            if not _last_emitted.strip():
                                yield {
                                    "type": "content",
                                    "text": (
                                        _reasoning_cap_spent_message(max_tokens)
                                        if _cap_left_g == 0
                                        else _thinking_exhausted_message(
                                            self._effective_context_length
                                        )
                                    ),
                                }
                            _meta = _build_metadata_event(
                                _metadata_usage, _metadata_timings, _metadata_finish_reason
                            )
                            if _meta is not None:
                                yield _meta
                    else:
                        _cut = _quote_cut_event(
                            reasoning_text,
                            answer_text,
                            _metadata_finish_reason,
                            promote_reasoning_only,
                        )
                        if _cut is not None:
                            yield _cut
                        _meta = _build_metadata_event(
                            _metadata_usage, _metadata_timings, _metadata_finish_reason
                        )
                        if _meta is not None:
                            yield _meta

                if _continue_final:
                    _continue_final = False
                    continue
                break

            except _LlamaStreamCancelled:
                # No flush here: the buffers ``_cancelled_hold_text`` reads are bound inside the
                # TOOL LOOP and are never rebound for this pass, which emits incrementally and
                # holds nothing back. Flushing them replayed the previous iteration's planning
                # text as this pass's answer, after the status boundary reset the route cursor.
                return
            except httpx.ConnectError:
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                raise
