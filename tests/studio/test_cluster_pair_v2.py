# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pairing v2: one code that also carries a tailcat address, approval words, LAN first and old peers.
tailcat is the loopback fake in _fake_tailnet.py; workers run in subprocesses like in the v1 tests."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _fake_tailnet  # noqa: E402
from studio import cluster_link  # noqa: E402
from studio.cluster_link import identity, pairing, remote, tokens  # noqa: E402

WORKER = textwrap.dedent(
    """
    import asyncio, json, os, sys
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {here!r})
    if os.environ.get("FAKE_TAILNET_DIR"):
        import _fake_tailnet
        sys.modules["studio.cluster_link.tailnet"] = _fake_tailnet
    from studio.cluster_link import identity, pairing
    if os.environ.get("OLD") == "1":
        pairing.PAIR_V2 = 99  # an older worker: it never asks for v2
    words = []
    try:
        ident = asyncio.run(pairing.join(sys.argv[1], name = "w-" + sys.argv[2], on_words = words.extend))
        master = identity.Registry().load().master()
        print(json.dumps({{"ok": True, "node_id": ident.node_id, "words": words,
                          "master_tc_pub": master.tc_pub, "master_addrs": master.addrs}}))
    except BaseException as exc:
        print(json.dumps({{"ok": False, "paired": identity.worker_identity() is not None,
                          "words": words, "error": type(exc).__name__ + ": " + str(exc)}}))
    """
).format(repo = str(REPO), here = str(Path(__file__).resolve().parent))


@pytest.fixture
def master_dir(tmp_path, monkeypatch):
    d = tmp_path / "master"
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(d))
    return d


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TAILNET_DIR", str(tmp_path / "tailnet"))
    monkeypatch.setitem(sys.modules, "studio.cluster_link.tailnet", _fake_tailnet)
    # `from . import tailnet` reads the package attribute first once the real module was imported.
    monkeypatch.setattr(cluster_link, "tailnet", _fake_tailnet, raising = False)
    return _fake_tailnet


async def _join(
    state_dir: Path,
    token: str,
    tag: str = "1",
    **env,
):
    env = dict(os.environ, UNSLOTH_CLUSTER_STATE_DIR = str(state_dir), **env)
    pipe = asyncio.subprocess.PIPE
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", WORKER, token, tag, env = env, stdout = pipe, stderr = pipe
    )
    out, err = await asyncio.wait_for(proc.communicate(), 60)
    lines = out.decode().strip().splitlines()
    assert lines, err.decode()[-2000:]
    return json.loads(lines[-1])


async def _open(addrs = ("127.0.0.1",), **kw):
    master = identity.master_identity()
    window = pairing.PairingWindow(master, host = "127.0.0.1", port = 0, addrs = list(addrs), **kw)
    return master, window, await window.start()


def _async(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return run


def _workers():
    return identity.Registry().load().workers()


def test_codes_v1_and_v2_round_trip_and_reject_tampering():
    invite = tokens.Invite("c" * 32, "d" * 32, ["10.0.0.2"], 50071, "f" * 64, b"s" * 32, 2e9)
    v1 = tokens.encode(invite)
    assert v1.startswith(tokens.PREFIX) and tokens.decode(v1).tc is None
    invite.tc = "tcABCDEFGHIJKLMN_-"
    v2 = tokens.encode(invite)
    back = tokens.decode(v2)
    assert v2.startswith(tokens.PREFIX_V2) and back.v == 2 and back.tc == invite.tc
    assert "tcABCDEFGHIJ" not in repr(back)  # the address is a secret: never in a repr or a log
    body = json.loads(tokens._b64d(v2[len(tokens.PREFIX_V2) :]))
    for bad in (
        {**body, "t": "not-a-tailcat-address"},
        {k: v for k, v in body.items() if k != "t"},
    ):
        raw = tokens._b64e(json.dumps(bad).encode())
        with pytest.raises(tokens.TokenError):
            tokens.decode(tokens.PREFIX_V2 + raw)
    # A v1 prefix cannot smuggle a v2 body past an old reader, nor the other way round.
    with pytest.raises(tokens.TokenError, match = "version"):
        tokens.decode(tokens.PREFIX + v2[len(tokens.PREFIX_V2) :])
    with pytest.raises(tokens.TokenError, match = "version"):
        tokens.decode(tokens.PREFIX_V2 + v1[len(tokens.PREFIX) :])


def test_the_v2_transcript_binds_the_tailcat_fields_and_the_master_key():
    secret = b"k" * 32
    base = dict(
        cluster_id = "c" * 32, invite_id = "i" * 32, session = b"s" * 16, worker_nonce = b"w" * 32,
        master_nonce = b"m" * 32, master_fp = "f" * 64, csr_digest = b"d" * 32,
    )  # fmt: skip
    fields = ["tcWORKERADDRESS1", "nodekey:abc", "tailnet", "192.168.1.5"]
    data = pairing.transcript(**base, v2 = fields)
    assert data != pairing.transcript(**base)
    for i in range(len(fields)):
        swapped = list(fields)
        swapped[i] = swapped[i] + "X"
        assert pairing._proof(secret, pairing.transcript(**base, v2 = swapped)) != pairing._proof(
            secret, data
        ), i
    assert pairing._confirm(secret, data, "nodekey:m1") != pairing._confirm(
        secret, data, "nodekey:m2"
    )
    words = pairing.approval_words(secret, data)
    assert len(words) == 3 and all(w in pairing.WORDS for w in words)
    assert len(set(pairing.WORDS)) == 256


@_async
async def test_lan_comes_first_and_both_sides_learn_each_others_tailcat_keys(
    master_dir, tmp_path, fake
):
    asked = []
    master, window, token = await _open(approve = lambda name, words: asked.append(words) or True)
    assert token.startswith(tokens.PREFIX_V2) and window.tc
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    peer = await window.wait(5)
    window.close()
    assert asked == [result["words"]] and len(result["words"]) == 3
    assert peer.addrs == ["127.0.0.1"] and peer.tc_addr.startswith("tc") and peer.tc_pub
    assert result["master_tc_pub"] == remote.own_pub("client")
    assert result["master_addrs"] == ["127.0.0.1"]
    # LAN worked, so the pairing never went through the tunnel.
    assert not [e for e in fake.events() if e["kind"] == "forward"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "needs 127.0.0.2 on loopback")
@_async
async def test_a_master_off_the_lan_is_reached_through_tailcat(master_dir, tmp_path, fake):
    _, window, token = await _open(addrs = ["127.0.0.2"])  # nothing answers there
    port = window.bound_port
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    peer = await window.wait(5)
    window.close()
    assert [e["port"] for e in fake.events() if e["kind"] == "forward"] == [port]
    # The tunnel's local end is never recorded as either side's address.
    assert "127.0.0.1" not in peer.addrs and peer.tc_addr
    assert result["master_addrs"] == ["127.0.0.2"]


@_async
async def test_a_refused_or_stolen_code_enrolls_nothing_and_is_spent(master_dir, tmp_path, fake):
    asked = []
    _, window, token = await _open(approve = lambda name, words: asked.append(name) and False)
    thief = await _join(tmp_path / "thief", token, "thief")
    assert not thief["ok"] and not thief["paired"] and "did not approve" in thief["error"], thief
    assert asked == ["w-thief"] and _workers() == []
    with pytest.raises(pairing.PairingError, match = "not approved"):
        await window.wait(5)
    late = await _join(tmp_path / "owner", token, "owner")
    assert not late["ok"] and not late["paired"]
    assert _workers() == []


@_async
async def test_an_async_approval_that_says_yes_pairs(master_dir, tmp_path, fake):
    async def approve(name, words):
        await asyncio.sleep(0.05)
        return True

    _, window, token = await _open(approve = approve)
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    assert (await window.wait(5)).node_id == result["node_id"]
    window.close()


@_async
async def test_a_new_worker_pairs_with_an_old_master(master_dir, tmp_path, fake, monkeypatch):
    monkeypatch.setattr(pairing, "PAIR_V2", 99)  # this master only speaks v1
    _, window, token = await _open(tailnet = False)
    assert token.startswith(tokens.PREFIX)
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    peer = await window.wait(5)
    assert result["words"] == [] and result["master_tc_pub"] == ""
    assert peer.tc_addr == "" and peer.tc_pub == ""


@_async
async def test_an_old_worker_is_refused_by_an_approving_master_and_pairs_with_yes(
    master_dir, tmp_path, fake
):
    asked = []
    _, window, token = await _open(
        tailnet = False, approve = lambda name, words: asked.append(words) or True
    )
    result = await _join(tmp_path / "w1", token, "1", OLD = "1")
    assert not result["ok"] and not result["paired"], result
    with pytest.raises(pairing.PairingError, match = "older Unsloth"):
        await window.wait(5)
    assert asked == [] and _workers() == []

    _, window, token = await _open(tailnet = False)
    result = await _join(tmp_path / "w2", token, "2", OLD = "1")
    assert result["ok"], result
    peer = await window.wait(5)
    assert result["words"] == [] and peer.tc_addr == ""


@_async
async def test_without_tailcat_the_code_stays_v1_and_lan_only(master_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_TAILNET_DIR", raising = False)
    monkeypatch.setenv("UNSLOTH_CLUSTER_TAILNET", "0")  # also when a real helper is installed
    _, window, token = await _open()
    assert token.startswith(tokens.PREFIX) and window.tc is None
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    assert result["master_tc_pub"] == "" and (await window.wait(5)).tc_addr == ""


def test_the_worker_tunnel_follows_its_masters_and_revokes_removed_ones(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "w"))
    assert remote.own_pub("server")
    masters = ["nodekey:m1"]
    tunnel = remote.WorkerTunnel([50070], lambda: list(masters))
    assert tunnel.start()
    served = [e for e in fake.events() if e["kind"] == "serve"]
    assert served[-1]["allow"] == ["nodekey:m1"] and served[-1]["ports"] == [50070]
    masters[:] = ["nodekey:m2"]
    tunnel.sync()
    kinds = [(e["kind"], e.get("pub")) for e in fake.events() if e["kind"] in ("allow", "revoke")]
    assert kinds == [("allow", "nodekey:m2"), ("revoke", "nodekey:m1")]
    tunnel.close()


def test_a_worker_never_paired_through_tailcat_opens_no_tunnel(tmp_path, monkeypatch, fake):
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "w"))
    assert not remote.WorkerTunnel([50070], lambda: ["nodekey:m1"]).start()
    monkeypatch.setenv("UNSLOTH_CLUSTER_TAILNET", "0")
    assert not remote.enabled() and remote.own_pub("server") is None


@_async
async def test_the_agent_serves_its_port_through_tailcat_and_closes_it_on_stop(master_dir, fake):
    from studio.cluster_link.agent import Agent

    me = identity.master_identity()  # any cluster certificate will do for the listener
    assert remote.own_pub("server")
    agent = Agent(
        me, lambda pid: True, host = "127.0.0.1", port = 0, tunnel_allow = lambda: ["nodekey:m"]
    )
    await agent.start()
    try:
        assert agent.tunnel is not None
        served = [e for e in fake.events() if e["kind"] == "serve"][-1]
        assert served["ports"] == [agent.port] and served["allow"] == ["nodekey:m"]
    finally:
        await agent.stop()
    assert (
        agent.tunnel is None
        and not (Path(os.environ["FAKE_TAILNET_DIR"]) / served["address"]).exists()
    )


@_async
async def test_cancelling_after_approval_enrolls_nothing(master_dir, tmp_path, fake):
    holder = {}

    def approve(name, words):
        asyncio.get_running_loop().call_soon(holder["window"].close)
        return True

    _, window, token = await _open(approve = approve)
    holder["window"] = window
    result = await _join(tmp_path / "w", token)
    assert not result["ok"], result
    assert _workers() == [] and window.paired is None
    with pytest.raises(pairing.PairingError):
        await window.wait(5)


@_async
async def test_closing_after_the_commit_still_reports_the_worker(master_dir, tmp_path, fake):
    _, window, token = await _open()
    result = await _join(tmp_path / "w", token)
    assert result["ok"], result
    window.close()
    assert (await window.wait(5)).node_id == result["node_id"] == window.paired.node_id


class _Net:
    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def _blocking(gate, result):
    def call(*_a, **_k):
        gate.wait(10)
        return result

    return call


@_async
async def test_a_tunnel_arriving_after_the_window_closed_is_closed(master_dir, monkeypatch):
    for how in ("cancel", "close"):
        gate, net = threading.Event(), _Net()
        monkeypatch.setattr(remote, "serve", _blocking(gate, (net, "tcLATE")))
        master = identity.master_identity()
        window = pairing.PairingWindow(
            master, host = "127.0.0.1", port = 0, addrs = ["127.0.0.1"], tailnet = True
        )
        task = asyncio.ensure_future(window.start())
        await asyncio.sleep(0.2)
        if how == "cancel":
            task.cancel()
        else:
            window.close()
        gate.set()
        if how == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
            assert window.tc is None
        assert await asyncio.to_thread(net.closed.wait, 5), how
        window.close()


@_async
async def test_a_dial_arriving_after_the_join_was_cancelled_is_closed(fake, monkeypatch):
    gate, net = threading.Event(), _Net()
    monkeypatch.setattr(remote, "dial", _blocking(gate, (net, 1)))
    invite = type("I", (), dict(addrs = ["127.0.0.1"], port = 1, master_fp = "f", tc = "tcX"))()
    task = asyncio.ensure_future(pairing._connect_any(invite, 5))
    await asyncio.sleep(0.5)
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(net.closed.wait, 5)


@_async
async def test_closing_a_window_does_not_wait_for_its_tunnel_on_the_event_loop(master_dir):
    class Slow(_Net):
        def close(self):
            time.sleep(1)
            super().close()

    master = identity.master_identity()
    window = pairing.PairingWindow(
        master, host = "127.0.0.1", port = 0, addrs = ["127.0.0.1"], tailnet = False
    )
    await window.start()
    window._net = net = Slow()
    t0 = time.monotonic()
    window.close()
    assert time.monotonic() - t0 < 0.5
    assert await asyncio.to_thread(net.closed.wait, 5)


@_async
async def test_a_tunnel_started_while_the_agent_stops_is_closed(master_dir, monkeypatch):
    from studio.cluster_link import agent as agent_mod
    for how in ("cancel", "stop"):
        gate, made = threading.Event(), []

        class Tunnel:
            def __init__(self, ports, allowed):
                self.closed = threading.Event()
                made.append(self)

            def start(self):
                gate.wait(10)
                return True

            def close(self):
                self.closed.set()

        monkeypatch.setattr(remote, "WorkerTunnel", Tunnel)
        me = identity.master_identity()
        agent = agent_mod.Agent(
            me, lambda pid: True, host = "127.0.0.1", port = 0, tunnel_allow = lambda: []
        )
        task = asyncio.ensure_future(agent._open_tunnel())
        await asyncio.sleep(0.2)
        if how == "cancel":
            task.cancel()
        else:
            agent._stopping = True
        gate.set()
        if how == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert await asyncio.to_thread(made[0].closed.wait, 5), how
        assert agent.tunnel is None
