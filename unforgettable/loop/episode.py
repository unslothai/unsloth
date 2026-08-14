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

import inspect
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
from unforgettable.eyes.basic import (
    ENTER_SIM_TOOL_NAMES,
    grade_run_action,
    inspect_tool_result,
    user_declares_failure,
)
from unforgettable.eyes.gate import LogGateEyes, review_write
from unforgettable.eyes.probes import MAX_EPISODE_PROBES, run_probes
from unforgettable.eyes.protocols import RecognizedFailure
from unforgettable.host import GenerateRequest, GenerateResult, Host
from unforgettable.rims.clone import clone_tree
from unforgettable.rims.detect import resolve_test_command
from unforgettable.store.compile import list_standing, pack_standing
from unforgettable.store.records import (
    insert_inject_stats,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
    list_records,
)
from unforgettable.throne.policy import Action, decide, policy_from_request
from unforgettable.tools.specs import CONTACT_TOOLS, MEMORY_TOOLS

from .context import EpisodeRequest, EpisodeState, last_user_text
from .runtime import bind_episode, current_traces, reset_episode, set_contact

_MEMORY_PREAMBLE = (
    "You have durable memory tools: memory_write, memory_search, memory_get, "
    "memory_supersede, memory_deprecate. Facts, corrections, and lessons that "
    "should survive this episode must go through those tools. After a recognized "
    "failure, call rims_enter_sim to request a sim clone of the world tree."
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
        if trace.name.replace(".", "_") in ENTER_SIM_TOOL_NAMES:
            return "enter_sim requested"
        fail = inspect_tool_result(trace.name, trace.result, contact=trace.contact)
        if fail:
            last = fail.summary
    return last


async def _maybe_run_sim_tests(
    host: Host, state: EpisodeState, on_chunk
) -> tuple[bool, Optional[RecognizedFailure]]:
    cmd = state.test_command
    run_fn = getattr(host, "run_action", None)
    if not cmd or run_fn is None:
        return False, None
    before = len(current_traces())
    result = await run_fn(
        state.active_session,
        "terminal",
        {"command": cmd},
        on_chunk=on_chunk,
    )
    state.traces.extend(current_traces()[before:])
    return True, grade_run_action("terminal", result, contact="sim")


async def _confirm_retry_world(host, request, state, action, policy, db_path: str) -> str:
    if action != Action.RETRY_WORLD or not policy.require_confirm_retry:
        return action
    fn = getattr(host, "confirm", None)
    allowed = False
    if fn is not None:
        allowed = await fn(
            "Retry the repaired plan in the world?",
            kind="retry_world",
            on_chunk=request.on_chunk,
            session_id=state.world_session,
        )
    if not allowed:
        LogGateEyes().note("retry_world: denied", db_path=db_path)
        return Action.ESCALATE
    return action


def _inject_bundle(
    query: str,
    *,
    stakes: Optional[str],
    skip_standing: bool,
    episode_id: str,
    db_path: str,
) -> str:
    standing_rows = [] if skip_standing else list_standing(db_path)
    standing_text, kept_rows = pack_standing(standing_rows)
    compiled_ids = {row["id"] for row in kept_rows}
    policy = RetrievePolicy(
        high_stakes=stakes == "high",
        exclude_ids=frozenset(compiled_ids),
    )
    retrieved = retrieve(query, policy=policy, db_path=db_path)
    retrieve_text = format_inject(retrieved, policy=policy)
    inject = "\n\n".join(
        part for part in (_MEMORY_PREAMBLE, standing_text, retrieve_text) if part
    )
    use_ids = [] if skip_standing else [row["id"] for row in kept_rows]
    use_ids.extend(row["id"] for row in retrieved)
    for record_id in use_ids:
        insert_retrieve_use(
            episode_id=episode_id,
            record_id=record_id,
            contact="world",
            db_path=db_path,
        )
    insert_inject_stats(
        episode_id=episode_id,
        contact="world",
        standing_chars=len(standing_text),
        retrieve_chars=len(retrieve_text),
        trajectory_chars=0,
        total_chars=len(inject),
        compiled_ids=",".join(row["id"] for row in kept_rows),
        retrieved_ids=",".join(row["id"] for row in retrieved),
        db_path=db_path,
    )
    return inject


async def run(host: Host, request: EpisodeRequest) -> EpisodeOutcome:
    episode_id = str(uuid.uuid4())
    db_path = str(host.memory_db_path())
    world = request.world_session_id or host.world_session_id(request)
    state = EpisodeState(episode_id=episode_id, world_session=world)
    policy = policy_from_request(request)
    tokens, _ = bind_episode(db_path=db_path, episode_id=episode_id, namespace=request.namespace)
    actions: list[str] = []
    text = ""
    try:
        inject = _inject_bundle(
            last_user_text(request.messages),
            stakes=request.stakes,
            skip_standing=request.skip_standing,
            episode_id=episode_id,
            db_path=db_path,
        )
        messages = _with_system(request.messages, inject)
        generated = False
        while True:
            set_contact(state.contact)
            if (
                not generated
                and state.contact == "world"
                and state.clone_count == 0
                and user_declares_failure(last_user_text(request.messages))
            ):
                fail_summary = "user declared failure"
                state.note_failure(fail_summary, "world")
                event = "failure"
            else:
                gen = await host.generate(
                    GenerateRequest(
                        messages=messages,
                        session_id=state.active_session,
                        thread_id=request.thread_id,
                        stream=request.stream,
                        extra_tools=list(MEMORY_TOOLS) + list(CONTACT_TOOLS),
                        inner_model=request.inner_model,
                        on_chunk=request.on_chunk,
                    )
                )
                generated = True
                text = gen.text or text
                state.traces.extend(gen.tool_traces)
                ran, grade = False, None
                if state.contact == "sim":
                    ran, grade = await _maybe_run_sim_tests(
                        host, state, request.on_chunk
                    )
                if ran:
                    if grade is None:
                        state.note_success(f"tests: {state.test_command}", "sim")
                        event = "success"
                    else:
                        fail_summary = f"tests: {state.test_command}: {grade.summary}"
                        state.note_failure(fail_summary, "sim")
                        event = "failure"
                else:
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
            action = await _confirm_retry_world(
                host, request, state, action, policy, db_path
            )
            actions.append(action)
            if action == Action.ENTER_SIM:
                sim_id = host.create_sim_session(episode_id)
                if not sim_id or sim_id == world or sim_id.startswith("project-"):
                    raise ValueError(f"refusing to share world sandbox as sim: {sim_id!r}")
                clone_tree(host.sandbox_path(world), host.sandbox_path(sim_id))
                state.enter_sim(sim_id)
                set_contact("sim")
                state.test_command = resolve_test_command(
                    requested=request.test_command,
                    db_path=db_path,
                    tree=host.sandbox_path(sim_id),
                )
                messages = _with_system(
                    request.messages,
                    inject
                    + f"\n\nYou are in a sim clone of the world tree. Previous world failure: {fail_summary}",
                )
                ran, grade = await _maybe_run_sim_tests(host, state, request.on_chunk)
                if ran:
                    if grade is None:
                        state.note_success(f"tests: {state.test_command}", "sim")
                        action = decide("success", state, policy)
                        action = await _confirm_retry_world(
                            host, request, state, action, policy, db_path
                        )
                        actions.append(action)
                        if action == Action.RETRY_WORLD:
                            state.enter_world()
                            messages = _with_system(
                                request.messages,
                                inject + "\n\nRetry in the world with the repaired plan.",
                            )
                            continue
                        if action == Action.CONTINUE_SIM:
                            state.sim_turns += 1
                            continue
                        if action != Action.ENTER_SIM:
                            break
                    else:
                        state.note_failure(
                            f"tests: {state.test_command}: {grade.summary}", "sim"
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

        error_fix_id = await _extract(
            state,
            db_path,
            last_user=last_user_text(request.messages),
            actions=actions,
            host=host,
        )
        await _run_episode_probes(host, request, state, db_path)
        return EpisodeOutcome(text=text, state=state, error_fix_id=error_fix_id, actions=actions)
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


async def _extract(
    state: EpisodeState,
    db_path: str,
    *,
    last_user: str,
    actions: list[str],
    host: Host,
) -> Optional[str]:
    drafts = list(from_episode(state))
    drafts.extend(from_drift(state))
    complete = getattr(host, "complete", None)
    if complete is not None:
        drafts.extend(await llm_extract(state, host))
    written_id = None
    draft_ids: list[str] = []
    for draft in drafts:
        rec = _write_draft(state, draft, db_path)
        draft_ids.append(rec["id"])
        if rec["kind"] == "error_fix" and rec["status"] in {"active", "proposed"}:
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
    state.keep_sim = False
    for rec in list_records(kinds=["error_fix", "twin_note"], db_path=db_path):
        if rec.get("source_episode_id") != state.episode_id:
            continue
        if rec["kind"] == "twin_note":
            state.keep_sim = True
        elif rec["kind"] == "error_fix" and rec["status"] == "active":
            state.keep_sim = True
    return written_id


async def _run_episode_probes(
    host: Host, request: EpisodeRequest, state: EpisodeState, db_path: str
) -> None:
    if state.sim_session is None:
        return
    if getattr(host, "run_action", None) is None:
        return
    result = run_probes(
        world=host.sandbox_path(state.world_session),
        host=host,
        db_path=db_path,
        limit=MAX_EPISODE_PROBES,
        on_chunk=request.on_chunk,
    )
    if inspect.isawaitable(result):
        await result
