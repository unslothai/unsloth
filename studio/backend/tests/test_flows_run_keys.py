# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run API keys: each run takes a fresh key, the next is minted ahead, and no minted key is left unrevoked."""

import asyncio

from core.flows.compiler import parse_document
from storage import flows_db

from ._flows_helpers import BASE_HANDLERS, doc, make_engine, node, run_async, run_doc, wait_until


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
    await asyncio.gather(*eng._revoking, return_exceptions = True)
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
    await asyncio.gather(*eng._spare_keys.values())  # the spare minted ahead
    deleted.update(minted)
    second = await run_doc(eng, d)
    assert second.key_id not in deleted
    await eng.release_spare_keys()


@run_async
async def test_a_cancelled_key_acquisition_still_revokes_its_key():
    import threading

    from utils.account_context import OWNER

    gate, minting, minted, revoked = threading.Event(), threading.Event(), [], []
    eng = make_engine(dict(BASE_HANDLERS))

    def mint(account):
        minting.set()
        gate.wait(10)
        minted.append(len(minted) + 1)
        return (f"k{minted[-1]}", minted[-1])

    eng.mint_key = mint
    eng.revoke_key = lambda key_id: revoked.append(key_id)
    eng.key_alive = lambda key_id: True
    # Cancelled while minting a fresh key.
    first = asyncio.create_task(eng.take_key(OWNER))
    await wait_until(minting.is_set, what = "the fresh mint to start")
    first.cancel()
    await asyncio.gather(first, return_exceptions = True)
    gate.set()
    await wait_until(lambda: revoked, what = "the cancelled mint's key to be revoked")
    assert revoked == [1]
    # Cancelled while waiting on the spare minted ahead.
    gate.clear()
    minting.clear()
    eng._spare_keys[(OWNER.account_id, OWNER.username)] = asyncio.create_task(
        eng._mint_spare(OWNER)
    )
    second = asyncio.create_task(eng.take_key(OWNER))
    # take_key has taken the spare and waits on it while it is still minting.
    await wait_until(
        lambda: minting.is_set() and not eng._spare_keys, what = "take_key to wait on the spare"
    )
    second.cancel()
    await asyncio.gather(second, return_exceptions = True)
    gate.set()
    await wait_until(lambda: len(revoked) == 2, what = "the spare's key to be revoked")
    await asyncio.gather(*eng._revoking, return_exceptions = True)
    assert sorted(revoked) == sorted(minted)
    await eng.release_spare_keys()


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
