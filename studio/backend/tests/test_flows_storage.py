# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stored runs and the output cache stay bounded, and cache hits keep their entries recent."""

from core.flows.compiler import parse_document
from storage import flows_db

from ._flows_helpers import BASE_HANDLERS, doc, make_engine, node, run_async, run_doc


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


def test_only_the_newest_runs_keep_their_events_and_results():
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
