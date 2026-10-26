# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The faster paths return what the slow ones did: gallery summaries, batched run writes, the pre-minted key."""

import asyncio

from core.flows.compiler import parse_document
from storage import flows_db

from ._flows_helpers import doc, node, run_async
from .test_flows_engine import BASE, _engine, _run


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


def test_example_summaries_are_built_once_and_match():
    from core.flows.examples import build_examples as examples
    from routes import flows as routes

    first = routes.list_examples()["examples"]
    assert first is routes.list_examples()["examples"]
    assert first == [routes._example_summary(examples.example(i)) for i in examples.example_ids()]


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
    rs = await _run(_engine(dict(BASE)), d)
    assert sum(e for e, _ in calls) == rs.seq
    assert sum(r for _, r in calls) == 40 and len(calls) < 10
    stored = flows_db.events_after(rs.id, 0)
    assert [e["seq"] for e in stored] == list(range(1, rs.seq + 1))
    assert set(flows_db.node_results(rs.id)) == {f"a{i}" for i in range(40)}


@run_async
async def test_each_run_takes_a_fresh_key_and_the_next_is_minted_ahead():
    minted, revoked = [], []
    eng = _engine(dict(BASE))
    eng.mint_key = lambda account: (minted.append(len(minted)) or (f"k{len(minted)}", len(minted)))
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    d = doc([node("a", "input.text")], id = "keys")
    flows_db.save_flow(parse_document(d)[0])
    first = await _run(eng, d)
    second = await _run(eng, d)
    await asyncio.sleep(0.05)
    assert first.key_id != second.key_id
    assert {first.key_id, second.key_id} <= set(revoked)
    await eng.release_spare_keys()
    assert sorted(revoked) == sorted(set(revoked)) and set(revoked) == set(
        range(1, len(minted) + 1)
    )
