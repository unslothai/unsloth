# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The faster paths return what the slow ones did: gallery summaries, batched run writes, the pre-minted key."""

import asyncio
from types import SimpleNamespace

from core.flows.compiler import parse_document
from storage import flows_db

from ._flows_helpers import BASE_HANDLERS, doc, edge, make_engine, node, run_async, run_doc


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
    rs = await run_doc(make_engine(dict(BASE_HANDLERS)), d)
    assert sum(e for e, _ in calls) == rs.seq
    assert sum(r for _, r in calls) == 40 and len(calls) < 10
    stored = flows_db.events_after(rs.id, 0)
    assert [e["seq"] for e in stored] == list(range(1, rs.seq + 1))
    assert set(flows_db.node_results(rs.id)) == {f"a{i}" for i in range(40)}


@run_async
async def test_each_run_takes_a_fresh_key_and_the_next_is_minted_ahead():
    minted, revoked = [], []
    eng = make_engine(dict(BASE_HANDLERS))
    eng.mint_key = lambda account: (minted.append(len(minted)) or (f"k{len(minted)}", len(minted)))
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: True
    d = doc([node("a", "input.text")], id = "keys")
    flows_db.save_flow(parse_document(d)[0])
    first = await run_doc(eng, d)
    second = await run_doc(eng, d)
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

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})
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
    eng = make_engine(dict(BASE_HANDLERS))

    def mint(account):
        minted.append(len(minted) + 1)
        return (f"k{minted[-1]}", minted[-1])

    eng.mint_key = mint
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: key_id not in deleted
    d = doc([node("a", "input.text")], id = "resetkeys")
    flows_db.save_flow(parse_document(d)[0])
    await run_doc(eng, d)
    await asyncio.sleep(0.05)
    deleted.update(minted)
    second = await run_doc(eng, d)
    assert second.key_id not in deleted
    await eng.release_spare_keys()


@run_async
async def test_a_cancelled_key_acquisition_still_revokes_its_key():
    import threading

    from utils.account_context import OWNER

    gate, minted, revoked = threading.Event(), [], []
    eng = make_engine(dict(BASE_HANDLERS))

    def mint(account):
        gate.wait(2)
        minted.append(len(minted) + 1)
        return (f"k{minted[-1]}", minted[-1])

    eng.mint_key = mint
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: True
    # Cancelled while minting a fresh key.
    first = asyncio.create_task(eng.take_key(OWNER))
    await asyncio.sleep(0.05)
    first.cancel()
    await asyncio.gather(first, return_exceptions = True)
    gate.set()
    for _ in range(100):
        if revoked:
            break
        await asyncio.sleep(0.02)
    assert revoked == [1]
    # Cancelled while waiting on the spare minted ahead.
    gate.clear()
    eng._spare_keys[(OWNER.account_id, OWNER.username)] = asyncio.create_task(
        eng._mint_spare(OWNER)
    )
    second = asyncio.create_task(eng.take_key(OWNER))
    await asyncio.sleep(0.05)
    second.cancel()
    await asyncio.gather(second, return_exceptions = True)
    gate.set()
    for _ in range(100):
        if len(revoked) == 2:
            break
        await asyncio.sleep(0.02)
    await asyncio.gather(*eng._revoking, return_exceptions = True)
    assert sorted(revoked) == sorted(minted)
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
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
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
async def test_a_cache_entry_stays_a_hit_while_its_batch_is_being_written(monkeypatch):
    import threading

    gate = threading.Event()
    real = flows_db.write_batch

    def slow(*args):
        gate.wait(2)
        return real(*args)

    d = doc([node("a", "input.text")], id = "cachewrite")
    flows_db.save_flow(parse_document(d)[0])
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
    monkeypatch.setattr(flows_db, "write_batch", slow)
    rs._unsaved_cache["k1"] = ("text.template", {"text": "v"})
    writing = asyncio.create_task(eng.persist(rs))
    await asyncio.sleep(0.05)
    assert not writing.done() and rs.pending_cache("k1") == ("text.template", {"text": "v"})
    gate.set()
    await writing
    assert rs.pending_cache("k1") is None and flows_db.cache_get("k1") == {"text": "v"}


@run_async
async def test_overlapping_calls_keep_one_spare_key_per_account():
    revoked, minted = [], []

    async def next_scene(ctx):
        return {"scene": {}}

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})

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
    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, d)
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


@run_async
async def test_repeated_invokes_of_one_revision_compile_the_flow_once(monkeypatch):
    from core.flows import engine as engine_mod

    compiled = []
    real = engine_mod.compile_flow
    monkeypatch.setattr(engine_mod, "compile_flow", lambda d: compiled.append(1) or real(d))

    async def next_scene(ctx):
        return {"scene": {"ok": True}}

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})
    parsed, _ = parse_document(doc([node("n", "story.next_scene")]))
    for _ in range(5):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n", revision = 1)
    assert len(compiled) == 1
    await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n", revision = 2)
    await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n")
    assert len(compiled) == 3
    await eng.release_spare_keys()


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


def test_upstream_sets_are_worked_out_once_per_node(monkeypatch):
    from core.flows import compiler

    calls = []
    real = compiler.Plan.ancestors
    monkeypatch.setattr(
        compiler.Plan, "ancestors", lambda self, t: calls.append(1) or real(self, t)
    )
    nodes = [node(f"t{i}", "text.template") for i in range(30)]
    edges = [edge(f"t{i}", "text", f"t{i + 1}", "values") for i in range(29)]
    plan = compiler.compile_flow(parse_document(doc(nodes, edges))[0])
    calls.clear()
    for _ in range(20):
        for i in range(30):
            plan.upstream(f"t{i}")
    assert len(calls) == 30
    assert plan.upstream("t3") == {"t0", "t1", "t2"}


def test_the_heap_topological_order_matches_the_sorted_list_one():
    import random

    from core.flows.compiler import _topo

    def reference(nodes, inputs):
        position = {nid: i for i, nid in enumerate(nodes)}
        remaining = {n: {s for ss in inputs.get(n, {}).values() for s, _ in ss} for n in nodes}
        dependents = {n: set() for n in nodes}
        for n, d in remaining.items():
            for s in d:
                dependents[s].add(n)
        ready = sorted((n for n, d in remaining.items() if not d), key = position.__getitem__)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for nxt in dependents[n]:
                remaining[nxt].discard(n)
                if not remaining[nxt] and nxt not in order and nxt not in ready:
                    ready.append(nxt)
            ready.sort(key = position.__getitem__)
        return order if len(order) == len(nodes) else None

    rng = random.Random(0)
    for _ in range(200):
        names = [f"n{i}" for i in range(rng.randint(1, 25))]
        rng.shuffle(names)
        nodes = {n: {} for n in names}
        inputs = {}
        for i, n in enumerate(names):
            pool = names if rng.random() < 0.05 else names[:i] + names[i + 1 :]
            picks = rng.sample(pool, min(len(pool), rng.randint(0, 3)))
            if picks:
                inputs[n] = {"a": [(p, "o") for p in picks]}
        assert _topo(nodes, inputs, {n: None for n in names}, {}) == reference(nodes, inputs)


def test_a_cards_load_settings_are_read_once_per_run(monkeypatch):
    from core.flows import model_settings
    from core.flows.engine import FlowEngine, RunState

    reads = []
    real = model_settings.card_load_kwargs
    monkeypatch.setattr(model_settings, "card_load_kwargs", lambda *a: reads.append(1) or real(*a))
    load = {"max_seq_length": 8192}
    nodes = {f"g{i}": {"params": {"model": {"repo": "x/y-GGUF"}, "load": load}} for i in range(5)}
    plan = SimpleNamespace(specs = {n: {"model_kind": "chat"} for n in nodes}, nodes = nodes)
    rs = RunState(id = "r", flow_id = "f", account = None, doc = {}, plan = plan, options = {}, persist = False)
    eng = FlowEngine()
    for _ in range(10):
        eng._pick_gpu(rs, list(nodes))
    assert len(reads) == 5
