# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run storage: batched event/result writes, persist ordering, gallery summaries, staying off the event loop,
and the bounds on stored runs and the output cache."""

import asyncio

from core.flows.compiler import parse_document
from storage import flows_db

from ._flows_helpers import BASE_HANDLERS, doc, make_engine, node, run_async, run_doc, wait_until


def _last_used(key: str) -> int:
    conn = flows_db.get_connection()
    try:
        return conn.execute("SELECT last_used_at FROM flow_cache WHERE key = ?", (key,)).fetchone()[
            0
        ]
    finally:
        conn.close()


def _age_cache(key: str, at: int) -> None:
    conn = flows_db.get_connection()
    try:
        conn.execute("UPDATE flow_cache SET last_used_at = ? WHERE key = ?", (at, key))
        conn.commit()
    finally:
        conn.close()


def test_a_batched_cache_hit_marks_its_entry_used():
    flows_db.cache_put("touched-key", "text.template", {"text": "v"})
    _age_cache("touched-key", 1)
    flows_db.create_run("touch-run", "touch-flow", {})
    flows_db.write_batch("touch-run", [], [], [], ("touched-key",))
    assert _last_used("touched-key") > 1


def test_the_cache_evicts_least_recently_used_entries_past_its_caps():
    flows_db.cache_clear()
    for i in range(6):
        flows_db.cache_put(f"lru-{i}", "text.template", {"text": "x" * 100})
        _age_cache(f"lru-{i}", 1000 + i)
    _age_cache("lru-0", 5000)
    assert flows_db.prune_cache(max_rows = 4) == 2
    conn = flows_db.get_connection()
    try:
        kept = {r[0] for r in conn.execute("SELECT key FROM flow_cache")}
    finally:
        conn.close()
    assert kept == {"lru-0", "lru-3", "lru-4", "lru-5"}
    assert flows_db.prune_cache(max_bytes = 250) == 2
    assert flows_db.cache_get("lru-0") is not None and flows_db.cache_get("lru-1") is None


def test_only_the_newest_runs_keep_their_events_and_results(monkeypatch):
    monkeypatch.setattr(flows_db, "HISTORY_KEPT", 2)
    flow = "prune-flow"
    ids = [f"prune-{i:02d}" for i in range(6)]
    for i, run_id in enumerate(ids):
        flows_db.create_run(run_id, flow, {})
        conn = flows_db.get_connection()
        try:
            conn.execute(
                "UPDATE flow_runs SET created_at = ?, status = 'completed' WHERE id = ?",
                (i, run_id),
            )
            conn.commit()
        finally:
            conn.close()
        results = [("every", "done", {"v": i}, None)]
        if i == 0:
            results.append(("once", "done", {"v": "first"}, None))
        flows_db.write_batch(run_id, [(1, {"seq": 1, "type": "run.started"})], results, [])
    flows_db.prune_runs(flow, keep = 2)
    assert [bool(flows_db.events_after(r, 0)) for r in ids] == [False] * 4 + [True] * 2
    assert flows_db.node_results(ids[1]) == {}
    # The only run of "once" is pruned, but its output is still the node's newest.
    assert set(flows_db.node_results(ids[0])) == {"once"}
    latest = flows_db.latest_node_outputs(flow)
    assert latest["once"]["outputs"] == {"v": "first"} and latest["every"]["outputs"] == {"v": 5}
    assert flows_db.get_run(ids[0]) is not None
    assert [h["outputs"]["v"] for h in flows_db.node_output_history(flow, "every")] == [5, 4]


@run_async
async def test_a_finished_run_prunes_older_runs_and_the_cache(monkeypatch):
    monkeypatch.setattr(flows_db, "RUNS_KEPT_PER_FLOW", 1)
    monkeypatch.setattr(flows_db, "CACHE_MAX_ROWS", 0)
    flows_db.cache_put("stale", "text.template", {"text": "v"})
    d = doc([node("a", "input.text", value = "x")], id = "prune-engine")
    flows_db.save_flow(parse_document(d)[0])
    eng = make_engine(dict(BASE_HANDLERS))
    first = await run_doc(eng, d)
    second = await run_doc(eng, d)
    assert flows_db.events_after(first.id, 0) == [] and flows_db.events_after(second.id, 0)
    assert flows_db.cache_get("stale") is None


def test_gallery_summaries_match_the_parsed_documents():
    docs = [
        doc([node("a", "input.text")], id = "plain", name = "Plain"),
        doc(
            [node("a", "input.text"), node("b", "input.text")],
            id = "cover",
            name = "Cover",
            description = "Has a cover",
            meta = {"cover": {"url": "/api/flows/assets/x.png", "w": 1}},
        ),
        doc(
            [node("a", "input.text")],
            id = "texty",
            name = "Texty",
            meta = {"cover": "/c.png"},
            baked = {"a": {"text": "x" * 100_000}},
        ),
    ]
    for d in docs:
        flows_db.save_flow(parse_document(d)[0])
    conn = flows_db.get_connection()
    try:
        slow = {r["id"]: flows_db._summary_from_row(r) for r in conn.execute("SELECT * FROM flows")}
    finally:
        conn.close()
    fast = {s["id"]: s for s in flows_db.list_flows()}
    assert fast == slow
    assert fast["cover"]["cover"] == "/api/flows/assets/x.png" and fast["texty"]["nodeCount"] == 1


@run_async
async def test_a_run_writes_its_events_and_results_in_few_transactions(monkeypatch):
    calls = []
    real = flows_db.write_batch

    def counting(
        run_id,
        events,
        results,
        cache_rows,
        touched = (),
    ):
        calls.append((len(events), len(results)))
        return real(run_id, events, results, cache_rows, touched)

    monkeypatch.setattr(flows_db, "write_batch", counting)
    nodes = [node(f"a{i}", "input.text", value = str(i)) for i in range(40)]
    d = doc(nodes, id = "many")
    flows_db.save_flow(parse_document(d)[0])
    rs = await run_doc(make_engine(dict(BASE_HANDLERS)), d)
    assert sum(e for e, _ in calls) == rs.seq
    assert sum(r for _, r in calls) == 40 and len(calls) < 10
    stored = flows_db.events_after(rs.id, 0)
    assert [e["seq"] for e in stored] == list(range(1, rs.seq + 1))
    assert set(flows_db.node_results(rs.id)) == {f"a{i}" for i in range(40)}


def _queued_on(lock: asyncio.Lock) -> bool:
    """A coroutine is asleep waiting for ``lock``."""
    return bool(lock._waiters)


@run_async
async def test_a_reconnect_waits_for_a_batch_already_being_written(monkeypatch):
    import threading

    gate, writing = threading.Event(), threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        writing.set()
        gate.wait(10)
        return real(*args)

    d = doc([node("a", "input.text")], id = "slowwrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
    monkeypatch.setattr(flows_db, "write_batch", slow)
    rs.emit("node.log", "a", {"message": "late"})
    first = asyncio.create_task(eng.persist(rs))
    await wait_until(writing.is_set, what = "the first batch write")
    second = asyncio.create_task(eng.persist(rs))
    await wait_until(lambda: _queued_on(rs._persist_lock), what = "the second persist to queue")
    assert not second.done()
    gate.set()
    await asyncio.gather(first, second)
    assert flows_db.events_after(rs.id, 0)[-1]["data"] == {"message": "late"}


@run_async
async def test_a_cache_entry_stays_a_hit_while_its_batch_is_being_written(monkeypatch):
    import threading

    gate, writing = threading.Event(), threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        writing.set()
        gate.wait(10)
        return real(*args)

    d = doc([node("a", "input.text")], id = "cachewrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
    monkeypatch.setattr(flows_db, "write_batch", slow)
    rs._unsaved_cache["k1"] = ("text.template", {"text": "v"})
    persist = asyncio.create_task(eng.persist(rs))
    await wait_until(writing.is_set, what = "the batch write")
    assert not persist.done() and rs.pending_cache("k1") == ("text.template", {"text": "v"})
    gate.set()
    await persist
    assert rs.pending_cache("k1") is None and flows_db.cache_get("k1") == {"text": "v"}


@run_async
async def test_a_cancelled_flusher_still_finishes_its_batch_before_the_next_persist(monkeypatch):
    import threading

    gate, writing = threading.Event(), threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        writing.set()
        gate.wait(10)
        return real(*args)

    d = doc([node("a", "input.text")], id = "cancelwrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
    monkeypatch.setattr(flows_db, "write_batch", slow)
    rs.emit("node.log", "a", {"message": "mid-write"})
    flusher = asyncio.create_task(eng.persist(rs))
    await wait_until(writing.is_set, what = "the flusher's batch write")
    flusher.cancel()
    final = asyncio.create_task(eng.persist(rs))
    await wait_until(lambda: _queued_on(rs._persist_lock), what = "the final persist to queue")
    assert not final.done()
    gate.set()
    await asyncio.gather(flusher, final, return_exceptions = True)
    assert flows_db.events_after(rs.id, 0)[-1]["data"] == {"message": "mid-write"}


@run_async
async def test_runs_and_invokes_keep_storage_off_the_event_loop(monkeypatch):
    import threading

    on_loop = []
    loop_thread = threading.get_ident()
    real = flows_db.get_connection

    def watched():
        if threading.get_ident() == loop_thread:
            on_loop.append(1)
        return real()

    async def next_scene(ctx):
        return {"scene": {"ok": True}}

    d = doc([node("a", "input.text", value = "x"), node("n", "story.next_scene")], id = "offloop")
    flows_db.save_flow(parse_document(d)[0])
    monkeypatch.setattr(flows_db, "get_connection", watched)
    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})
    await run_doc(eng, d)
    parsed, _ = parse_document(d)
    await eng.invoke(app = None, flow_id = "offloop", doc = parsed, entry = "n")
    await eng.release_spare_keys()
    assert on_loop == []


def test_pruning_keeps_the_runs_behind_a_nodes_history():
    flow = "prune-history"
    for i in range(8):
        run_id = f"hist-{i}"
        flows_db.create_run(run_id, flow, {})
        conn = flows_db.get_connection()
        try:
            conn.execute(
                "UPDATE flow_runs SET created_at = ?, status = 'completed' WHERE id = ?",
                (i, run_id),
            )
            conn.commit()
        finally:
            conn.close()
        # Five takes of "x", then side-branch runs that never touch it.
        node = (
            ("x", "done", {"text": f"take {i}"}, None)
            if i < 5
            else ("other", "done", {"v": i}, None)
        )
        flows_db.write_batch(run_id, [], [node], [])
    flows_db.prune_runs(flow, keep = 2)
    history = [h["outputs"]["text"] for h in flows_db.node_output_history(flow, "x")]
    assert history == [f"take {i}" for i in range(4, -1, -1)]
