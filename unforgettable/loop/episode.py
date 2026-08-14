# Copyright 2026-present the Unforgettable contributors.
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

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from unforgettable.agents.admissions import admit
from unforgettable.agents.extractor import (
    episode_summary,
    from_drift,
    from_episode,
    llm_extract,
)
from unforgettable.agents.retriever import RetrievePolicy, format_inject, retrieve
from unforgettable.eyes.basic import inspect_tool_result
from unforgettable.eyes.gate import review_write
from unforgettable.host import GenerateRequest, GenerateResult, Host
from unforgettable.rims.clone import clone_tree
from unforgettable.store.records import insert_record, insert_rollout
from unforgettable.throne.policy import Action, decide, default_policy
from unforgettable.tools.specs import MEMORY_TOOLS

from .context import EpisodeRequest, EpisodeState, last_user_text
from .runtime import bind_episode, reset_episode, set_contact

_MEMORY_PREAMBLE = (
    "You have durable memory tools: memory_write, memory_search, memory_get, "
    "memory_supersede, memory_deprecate. Facts, corrections, and lessons that "
    "should survive this episode must go through those tools."
)


@dataclass
class EpisodeOutcome:
    text: str
    state: EpisodeState
    error_fix_id: Optional[str] = None
    actions: list[str] = field(default_factory=list)


def _with_system(messages: list[dict[str, Any]], extra: str) -> list[dict[str, Any]]:
    block = extra.strip()
    if not block:
        return list(messages)
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0] = {**out[0], "content": f"{out[0].get('content') or ''}\n\n{block}"}
        return out
    return [{"role": "system", "content": block}, *out]


def _pass_failure(result: GenerateResult) -> Optional[str]:
    last = None
    for trace in result.tool_traces:
        if trace.name.startswith("memory.") or trace.name.startswith("memory_"):
            continue
        fail = inspect_tool_result(trace.name, trace.result, contact=trace.contact)
        if fail:
            last = fail.summary
    return last


async def run(host: Host, request: EpisodeRequest) -> EpisodeOutcome:
    episode_id = str(uuid.uuid4())
    db_path = str(host.memory_db_path())
    world = request.world_session_id or host.world_session_id(request)
    state = EpisodeState(episode_id=episode_id, world_session=world)
    policy = default_policy()
    tokens, _ = bind_episode(
        db_path=db_path, episode_id=episode_id, namespace=request.namespace
    )
    actions: list[str] = []
    text = ""
    try:
        retrieve_policy = RetrievePolicy(high_stakes=request.stakes == "high")
        retrieved = retrieve(
            last_user_text(request.messages), policy=retrieve_policy, db_path=db_path
        )
        inject = "\n\n".join(
            part
            for part in (
                _MEMORY_PREAMBLE,
                format_inject(retrieved, policy=retrieve_policy),
            )
            if part
        )
        messages = _with_system(request.messages, inject)
        while True:
            set_contact(state.contact)
            gen = await host.generate(
                GenerateRequest(
                    messages=messages,
                    session_id=state.active_session,
                    thread_id=request.thread_id,
                    stream=request.stream,
                    extra_tools=list(MEMORY_TOOLS),
                    inner_model=request.inner_model,
                    on_chunk=request.on_chunk,
                )
            )
            text = gen.text or text
            state.traces.extend(gen.tool_traces)
            fail_summary = _pass_failure(gen)
            if fail_summary:
                state.note_failure(fail_summary, state.contact)
                event = "failure"
            elif gen.finished:
                state.note_success(gen.text or "completed", state.contact)
                event = "success"
            else:
                event = "finished"
            action = decide(event, state, policy)
            actions.append(action)
            if action == Action.ENTER_SIM:
                sim_id = host.create_sim_session(episode_id)
                clone_tree(host.sandbox_path(world), host.sandbox_path(sim_id))
                state.enter_sim(sim_id)
                messages = _with_system(
                    request.messages,
                    inject
                    + f"\n\nYou are in a sim clone of the world tree. Previous world failure: {fail_summary}",
                )
                continue
            if action == Action.CONTINUE_SIM:
                state.sim_turns += 1
                continue
            if action == Action.RETRY_WORLD:
                state.enter_world()
                messages = _with_system(
                    request.messages,
                    inject + "\n\nRetry in the world with the repaired plan.",
                )
                continue
            break

        error_fix_id = _extract(
            state,
            db_path,
            last_user=last_user_text(request.messages),
            actions=actions,
        )
        return EpisodeOutcome(
            text=text, state=state, error_fix_id=error_fix_id, actions=actions
        )
    finally:
        try:
            if state.sim_session and not state.keep_sim:
                host.remove_sim_session(state.sim_session)
        finally:
            reset_episode(tokens)


ROLLOUT_PASS = "pass"
ROLLOUT_FAIL = "fail"
ROLLOUT_CONTACT_ORDER = ("world", "sim")


def _write_draft(state: EpisodeState, draft: dict[str, Any], db_path: str) -> dict[str, Any]:
    review_reason = review_write(
        kind=draft["kind"],
        title=draft["title"],
        body=draft["body"],
        provenance=draft["provenance"],
        db_path=db_path,
    )
    decision = admit(
        kind=draft["kind"],
        provenance=draft["provenance"],
        explicit=bool(draft.get("explicit")),
        bookkeeping=bool(draft.get("bookkeeping")),
        force_proposed_reason=review_reason or None,
        db_path=db_path,
    )
    return insert_record(
        kind=draft["kind"],
        title=draft["title"],
        body=draft["body"],
        provenance=draft["provenance"],
        status=decision.status,
        source_episode_id=state.episode_id,
        contact_tag=draft["provenance"],
        db_path=db_path,
    )


def _write_rollouts(state: EpisodeState, *, source_record_id: str, db_path: str) -> None:
    last_by_contact: dict[str, dict[str, Any]] = {}
    for event in state.trace_events:
        contact = event.get("contact")
        if contact in ROLLOUT_CONTACT_ORDER:
            last_by_contact[contact] = event
    for contact in ROLLOUT_CONTACT_ORDER:
        event = last_by_contact.get(contact)
        if event is None:
            continue
        outcome = ROLLOUT_PASS if event.get("kind") == "success" else ROLLOUT_FAIL
        insert_rollout(
            episode_id=state.episode_id,
            contact=contact,
            outcome=outcome,
            summary=event.get("summary") or "",
            source_record_id=source_record_id,
            db_path=db_path,
        )


def _extract(
    state: EpisodeState,
    db_path: str,
    *,
    last_user: str,
    actions: list[str],
) -> Optional[str]:
    drafts = list(from_episode(state))
    drafts.extend(from_drift(state))
    drafts.extend(llm_extract(state))
    written_id = None
    draft_ids: list[str] = []
    for draft in drafts:
        rec = _write_draft(state, draft, db_path)
        draft_ids.append(rec["id"])
        if rec["kind"] == "error_fix" and rec["status"] in {"active", "proposed"}:
            state.keep_sim = True
            written_id = rec["id"]
        elif written_id is None:
            written_id = rec["id"]
    episode_draft = episode_summary(
        state, last_user=last_user, draft_ids=draft_ids, actions=actions
    )
    episode_rec = _write_draft(state, episode_draft, db_path)
    _write_rollouts(state, source_record_id=episode_rec["id"], db_path=db_path)
    if written_id is None:
        written_id = episode_rec["id"]
    return written_id
