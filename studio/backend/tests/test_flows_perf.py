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
    eng.key_alive = lambda key_id: True
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


@run_async
async def test_an_invoke_key_is_still_revoked_after_the_answer():
    revoked = []

    async def next_scene(ctx):
        return {"scene": {"ok": True}}

    eng = _engine({**BASE, "story.next_scene": next_scene})
    eng.mint_key = lambda account: ("k", 7)
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: True
    parsed, _ = parse_document(doc([node("n", "story.next_scene")]))
    out = await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n")
    assert out["scene"] == {"ok": True}
    await eng.release_spare_keys()
    assert revoked.count(7) >= 1


@run_async
async def test_a_spare_key_deleted_by_a_credential_reset_is_not_used():
    minted, revoked, deleted = [], [], set()
    eng = _engine(dict(BASE))

    def mint(account):
        minted.append(len(minted) + 1)
        return (f"k{minted[-1]}", minted[-1])

    eng.mint_key = mint
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: key_id not in deleted
    d = doc([node("a", "input.text")], id = "resetkeys")
    flows_db.save_flow(parse_document(d)[0])
    await _run(eng, d)
    await asyncio.sleep(0.05)
    deleted.update(minted)
    second = await _run(eng, d)
    assert second.key_id not in deleted
    await eng.release_spare_keys()


def test_a_run_key_reads_dead_after_revoke_or_account_reset(tmp_path, monkeypatch):
    import secrets

    from auth import policy, storage
    from core.flows import caller
    from utils.account_context import AccountContext

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    try:
        storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
        alice = storage.issue_account_setup_code(username = "alice")["account"]
        account = AccountContext(account_id = alice["account_id"], username = "alice", role = "user")
        _, revoked_id = caller.mint_run_key(account)
        caller.revoke_run_key(revoked_id)
        _, reset_id = caller.mint_run_key(account)
        assert caller.run_key_active(reset_id) and not caller.run_key_active(revoked_id)
        storage.set_account_active(alice["account_id"], False)
        assert not caller.run_key_active(reset_id)
        assert caller.run_key_active(None)
    finally:
        policy.invalidate_account_cache()


@run_async
async def test_a_reconnect_waits_for_a_batch_already_being_written(monkeypatch):
    import threading

    gate = threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        gate.wait(2)
        return real(*args)

    monkeypatch.setattr(flows_db, "write_batch", slow)
    d = doc([node("a", "input.text")], id = "slowwrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = _engine(dict(BASE))
    rs = await _run(eng, d)
    rs.emit("node.log", "a", {"message": "late"})
    first = asyncio.create_task(eng.persist(rs))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(eng.persist(rs))
    await asyncio.sleep(0.05)
    assert not second.done()
    gate.set()
    await asyncio.gather(first, second)
    assert flows_db.events_after(rs.id, 0)[-1]["data"] == {"message": "late"}


@run_async
async def test_overlapping_calls_keep_one_spare_key_per_account():
    revoked, minted = [], []

    async def next_scene(ctx):
        return {"scene": {}}

    eng = _engine({**BASE, "story.next_scene": next_scene})

    def mint(account):
        minted.append(len(minted) + 1)
        return (f"k{minted[-1]}", minted[-1])

    eng.mint_key = mint
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: True
    parsed, _ = parse_document(doc([node("n", "story.next_scene")]))
    await asyncio.gather(
        *(eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n") for _ in range(4))
    )
    assert len(eng._spare_keys) == 1
    await eng.release_spare_keys()
    assert sorted(revoked) == sorted(minted)


@run_async
async def test_a_cancelled_flusher_still_finishes_its_batch_before_the_next_persist(monkeypatch):
    import threading

    gate = threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        gate.wait(2)
        return real(*args)

    d = doc([node("a", "input.text")], id = "cancelwrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = _engine(dict(BASE))
    rs = await _run(eng, d)
    monkeypatch.setattr(flows_db, "write_batch", slow)
    rs.emit("node.log", "a", {"message": "mid-write"})
    flusher = asyncio.create_task(eng.persist(rs))
    await asyncio.sleep(0.05)
    flusher.cancel()
    final = asyncio.create_task(eng.persist(rs))
    await asyncio.sleep(0.05)
    assert not final.done()
    gate.set()
    await asyncio.gather(flusher, final, return_exceptions = True)
    assert flows_db.events_after(rs.id, 0)[-1]["data"] == {"message": "mid-write"}


def test_the_token_cache_keeps_only_short_texts():
    from core.flows import memory

    memory._cached_tokens.cache_clear()
    memory.tokens("a short memory about the lantern")
    memory.tokens("lantern " * 1000)
    assert memory._cached_tokens.cache_info().currsize == 1
    assert memory.tokens("lantern " * 1000) == frozenset({"lantern"})
