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
from unforgettable.constants import DEFAULT_NAMESPACE_ID
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
from unforgettable.sidecar.adapters import STATUS_DISCARDED, get_adapter
from unforgettable.sidecar.pack import ROLE_TRAIN, list_pack_items
from unforgettable.store.compile import list_standing, maybe_compile, pack_standing
from unforgettable.store.records import (
    insert_inject_stats,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
    list_records,
)
from unforgettable.store.trajectories import format_trajectories, retrieve_trajectories
from unforgettable.throne.policy import Action, decide, policy_from_request
from unforgettable.tools.specs import CONTACT_TOOLS, MEMORY_TOOLS

from .context import EpisodeRequest, EpisodeState, last_user_text
from .runtime import bind_episode, current_traces, reset_episode, set_contact

_MEMORY_PREAMBLE = (
    "You have durable memory tools: memory_write, memory_search, memory_get, "
    "memory_supersede, memory_deprecate, memory_compact, memory_compile. "
    "Facts, corrections, and lessons that should survive this episode must go "
    "through those tools. After a recognized failure, call rims_enter_sim to "
    "request a sim clone of the world tree."
)
REPAIR_TEXT_CHARS = 1200


def _clip_repair(text: str, limit: int = REPAIR_TEXT_CHARS) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip() + "..."


def _repair_context(state: EpisodeState) -> str:
    """Working-memory A for CONTINUE_SIM / RETRY_WORLD. Not durable B."""
    lines = ["Repaired-plan notes (this episode only):"]
    if state.last_fail_summary:
        lines.append(f"- Last failure: {state.last_fail_summary}")
    if state.last_sim_summary:
        lines.append(f"- Sim: {state.last_sim_summary}")
    if state.test_command:
        lines.append(f"- Test command: {state.test_command}")
    clip = _clip_repair(state.last_generate_text)
    if clip:
        lines.append("- Last inner pass:")
        lines.append(clip)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


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


def _resolve_attached_adapter(
    request: EpisodeRequest, db_path: str
) -> tuple[Optional[str], frozenset[str]]:
    adapter = None
    adapter_path = None
    if request.adapter_id:
        adapter = get_adapter(request.adapter_id, db_path=db_path)
        if adapter is None or adapter.get("status") == STATUS_DISCARDED:
            LogGateEyes().note("adapter: missing or discarded", db_path=db_path)
            adapter = None
        else:
            adapter_path = adapter.get("path") or None
    shrink = request.shrink_standing is True or (
        request.shrink_standing is None and adapter is not None
    )
    exclude: frozenset[str] = frozenset()
    if shrink and adapter is not None:
        pack_id = adapter.get("pack_id")
        if pack_id:
            exclude = frozenset(
                item["source_id"]
                for item in list_pack_items(pack_id, db_path=db_path)
                if item.get("source_id") and item.get("role") == ROLE_TRAIN
            )
    return adapter_path, exclude


def _inject_bundle(
    query: str,
    *,
    stakes: Optional[str],
    skip_standing: bool,
    episode_id: str,
    db_path: str,
    contact: str = "world",
    exclude_standing_ids: frozenset[str] = frozenset(),
    namespace: Optional[str] = None,
) -> str:
    standing_rows = [] if skip_standing else list_standing(db_path)
    if exclude_standing_ids:
        standing_rows = [
            row for row in standing_rows if row["id"] not in exclude_standing_ids
        ]
    standing_text, kept_rows = pack_standing(standing_rows)
    compiled_ids = {row["id"] for row in kept_rows}
    retrieve_exclude = compiled_ids | set(exclude_standing_ids)
    high_stakes = stakes == "high" and contact == "world"
    policy = RetrievePolicy(
        high_stakes=high_stakes,
        contact=contact,
        exclude_ids=frozenset(retrieve_exclude),
        max_twin_notes=3 if contact == "sim" else 1,
    )
    retrieved = retrieve(
        query, policy=policy, db_path=db_path, namespace_id=namespace
    )
    retrieve_text = format_inject(retrieved, policy=policy)
    trajectories = retrieve_trajectories(
        query,
        contact=contact,
        high_stakes=high_stakes,
        db_path=db_path,
    )
    traj_text = format_trajectories(trajectories)
    inject = "\n\n".join(
        part
        for part in (_MEMORY_PREAMBLE, standing_text, retrieve_text, traj_text)
        if part
    )
    use_ids = [] if skip_standing else [row["id"] for row in kept_rows]
    use_ids.extend(row["id"] for row in retrieved)
    for record_id in use_ids:
        insert_retrieve_use(
            episode_id=episode_id,
            record_id=record_id,
            contact=contact,
            db_path=db_path,
        )
    insert_inject_stats(
        episode_id=episode_id,
        contact=contact,
        standing_chars=len(standing_text),
        retrieve_chars=len(retrieve_text),
        trajectory_chars=len(traj_text),
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
    adapter_path, exclude_standing_ids = _resolve_attached_adapter(request, db_path)
    try:
        def _rebuild(contact: str, suffix: str) -> list[dict[str, Any]]:
            inject = _inject_bundle(
                last_user_text(request.messages),
                stakes=request.stakes,
                skip_standing=request.skip_standing,
                episode_id=episode_id,
                db_path=db_path,
                contact=contact,
                exclude_standing_ids=exclude_standing_ids,
                namespace=request.namespace,
            )
            notes = _repair_context(state)
            parts = [inject, suffix, notes]
            return _with_system(
                request.messages, "\n\n".join(p for p in parts if p)
            )

        messages = _rebuild("world", "")
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
                        adapter_path=adapter_path,
                    )
                )
                generated = True
                text = gen.text or text
                if gen.text:
                    state.last_generate_text = gen.text
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
                state.track_sim(sim_id)
                try:
                    if not sim_id or sim_id == world or sim_id.startswith("project-"):
                        raise ValueError(
                            f"refusing to share world sandbox as sim: {sim_id!r}"
                        )
                    clone_tree(host.sandbox_path(world), host.sandbox_path(sim_id))
                except Exception:
                    LogGateEyes().note(
                        f"sim: clone failed for {sim_id!r}", db_path=db_path
                    )
                    raise
                state.enter_sim(sim_id)
                set_contact("sim")
                state.test_command = resolve_test_command(
                    requested=request.test_command,
                    db_path=db_path,
                    tree=host.sandbox_path(sim_id),
                )
                messages = _rebuild(
                    "sim",
                    f"You are in a sim clone of the world tree. Previous world failure: {fail_summary}",
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
                            messages = _rebuild(
                                "world",
                                "Retry in the world with the repaired plan.",
                            )
                            continue
                        if action == Action.CONTINUE_SIM:
                            state.sim_turns += 1
                            messages = _rebuild(
                                "sim",
                                f"You are in a sim clone of the world tree. Previous world failure: {fail_summary}",
                            )
                            continue
                        if action != Action.ENTER_SIM:
                            break
                    else:
                        state.note_failure(
                            f"tests: {state.test_command}: {grade.summary}", "sim"
                        )
                        messages = _rebuild(
                            "sim",
                            f"You are in a sim clone of the world tree. Previous world failure: {fail_summary}",
                        )
                continue
            if action == Action.CONTINUE_SIM:
                state.sim_turns += 1
                messages = _rebuild(
                    "sim",
                    f"You are in a sim clone of the world tree. Previous world failure: {state.last_fail_summary}",
                )
                continue
            if action == Action.RETRY_WORLD:
                state.enter_world()
                messages = _rebuild(
                    "world",
                    "Retry in the world with the repaired plan.",
                )
                continue
            break

        error_fix_id = await _extract(
            state,
            db_path,
            last_user=last_user_text(request.messages),
            actions=actions,
            host=host,
            namespace=request.namespace,
        )
        await _run_episode_probes(host, request, state, db_path)
        return EpisodeOutcome(text=text, state=state, error_fix_id=error_fix_id, actions=actions)
    finally:
        try:
            kept = state.sim_session if state.keep_sim else None
            seen: set[str] = set()
            for sid in list(state.created_sims):
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                if sid != kept:
                    host.remove_sim_session(sid)
        finally:
            reset_episode(tokens)


ROLLOUT_PASS = "pass"
ROLLOUT_FAIL = "fail"
ROLLOUT_CONTACT_ORDER = ("world", "sim")


def _write_draft(
    state: EpisodeState,
    draft: dict[str, Any],
    db_path: str,
    *,
    namespace: str = DEFAULT_NAMESPACE_ID,
) -> dict[str, Any]:
    rid = str(uuid.uuid4())
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
        namespace_id=namespace,
        record_id=rid,
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
        namespace_id=namespace,
        source_episode_id=state.episode_id,
        contact_tag=draft["provenance"],
        record_id=rid,
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
    namespace: str = DEFAULT_NAMESPACE_ID,
) -> Optional[str]:
    drafts = list(from_episode(state))
    drafts.extend(from_drift(state))
    complete = getattr(host, "complete", None)
    if complete is not None:
        drafts.extend(await llm_extract(state, host))
    written_id = None
    draft_ids: list[str] = []
    for draft in drafts:
        rec = _write_draft(state, draft, db_path, namespace=namespace)
        draft_ids.append(rec["id"])
        if rec["kind"] == "error_fix" and rec["status"] in {"active", "proposed"}:
            written_id = rec["id"]
        elif written_id is None:
            written_id = rec["id"]
    episode_draft = episode_summary(
        state, last_user=last_user, draft_ids=draft_ids, actions=actions
    )
    episode_rec = _write_draft(state, episode_draft, db_path, namespace=namespace)
    _write_rollouts(state, source_record_id=episode_rec["id"], db_path=db_path)
    maybe_compile(db_path)
    if written_id is None:
        written_id = episode_rec["id"]
    state.keep_sim = False
    for rec in list_records(kinds=["error_fix", "twin_note"], db_path=db_path):
        if rec.get("source_episode_id") != state.episode_id:
            continue
        if rec["kind"] == "twin_note" and rec["status"] == "active":
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
