# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Machines routes: a real pairing window, a real worker joining from a subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("cryptography")

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from routes import cluster  # noqa: E402

WORKER = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import pairing
    words = []
    try:
        ident = asyncio.run(pairing.join(sys.stdin.readline().strip(), name = "desk", on_words = words.extend))
        print(json.dumps({{"ok": True, "node_id": ident.node_id, "words": words}}))
    except BaseException as exc:
        print(json.dumps({{"ok": False, "words": words, "error": str(exc)}}))
    """
).format(repo = str(REPO))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "master"))
    monkeypatch.setenv("UNSLOTH_CLUSTER_TAILNET", "0")
    app = FastAPI()
    app.include_router(cluster.router, prefix = "/api/cluster")
    app.dependency_overrides[cluster.get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[cluster.authenticated_via_api_key] = lambda: False
    app.dependency_overrides[cluster.policy.require_owner] = lambda: None
    cluster._PAIRINGS.clear()
    with TestClient(app) as c:
        yield c
    for p in list(cluster._PAIRINGS.values()):
        p.window.close()
    cluster._PAIRINGS.clear()


def _worker(tmp_path, code):
    env = dict(
        os.environ, UNSLOTH_CLUSTER_STATE_DIR = str(tmp_path / "worker"), UNSLOTH_CLUSTER_TAILNET = "0"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKER], env = env, text = True,
        stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
    )  # fmt: skip
    proc.stdin.write(code + "\n")
    proc.stdin.flush()
    return proc


def _wait_state(
    client,
    pid,
    want,
    timeout = 30.0,
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/cluster/pair/{pid}").json()
        if st["state"] in want:
            return st
        time.sleep(0.1)
    raise AssertionError(f"pairing stayed {st['state']}")


def _result(proc):
    out, err = proc.communicate(timeout = 60)
    lines = out.strip().splitlines()
    assert lines, err[-2000:]
    return json.loads(lines[-1])


def test_no_cluster_yet_lists_nothing(client):
    assert client.get("/api/cluster/machines").json() == {"role": "none", "machines": []}


def test_pair_approve_then_list_and_remove(client, tmp_path):
    started = client.post("/api/cluster/pair").json()
    code, pid = started.pop("code"), started["id"]
    assert started["state"] == "waiting" and code.startswith("unsloth-pair-")
    proc = _worker(tmp_path, code)
    asked = _wait_state(client, pid, {"approval"})
    # The code is handed out once, by the POST that made it, and never again.
    assert code not in json.dumps(asked) and asked["name"] == "desk" and len(asked["words"]) == 3
    assert client.post(f"/api/cluster/pair/{pid}/approve").status_code == 200
    joined = _result(proc)
    assert joined["ok"] and joined["words"] == asked["words"]
    done = _wait_state(client, pid, {"paired"})
    assert done["node_id"] == joined["node_id"]
    listed = client.get("/api/cluster/machines", params = {"check": False}).json()
    assert listed["role"] == "master"
    assert [m["node_id"] for m in listed["machines"]] == [joined["node_id"]]
    gone = client.delete(f"/api/cluster/machines/{joined['node_id']}")
    assert gone.status_code == 200 and gone.json()["name"] == "desk"
    assert client.get("/api/cluster/machines", params = {"check": False}).json()["machines"] == []
    assert client.delete(f"/api/cluster/machines/{joined['node_id']}").status_code == 404


def test_refused_worker_is_not_paired(client, tmp_path):
    started = client.post("/api/cluster/pair").json()
    proc = _worker(tmp_path, started["code"])
    _wait_state(client, started["id"], {"approval"})
    assert client.post(f"/api/cluster/pair/{started['id']}/refuse").status_code == 200
    assert not _result(proc)["ok"]
    assert _wait_state(client, started["id"], {"refused"})["node_id"] is None
    assert client.get("/api/cluster/machines", params = {"check": False}).json()["machines"] == []
    assert client.post(f"/api/cluster/pair/{started['id']}/approve").status_code == 409


def test_cancel_spends_the_code(client, tmp_path):
    started = client.post("/api/cluster/pair").json()
    assert client.post(f"/api/cluster/pair/{started['id']}/cancel").json()["state"] == "cancelled"
    assert not _result(_worker(tmp_path, started["code"]))["ok"]
    assert client.get("/api/cluster/pair/nope").status_code == 404


def test_a_new_pairing_cancels_the_open_one(client):
    first = client.post("/api/cluster/pair").json()
    client.post("/api/cluster/pair")
    assert client.get(f"/api/cluster/pair/{first['id']}").status_code == 404


def test_pairing_from_studio_installs_the_helper_like_the_cli(client, monkeypatch):
    from studio.cluster_link import remote

    calls = []
    monkeypatch.setattr(remote, "ensure_helper", lambda *a: calls.append(a) or False)
    assert client.post("/api/cluster/pair").status_code == 200
    assert calls == [()]


def test_checked_list_reports_each_status(client, monkeypatch):
    from studio.cluster_link import identity, status

    identity.master_identity(create = True)
    peers = [
        type("P", (), dict(name = n, node_id = i, removed = False, fingerprint = "f"))()
        for n, i in (("desk", "a1"), ("laptop", "b2"))
    ]
    monkeypatch.setattr(identity.Registry, "workers", lambda self: peers)
    rows = [
        status.WorkerStatus("desk", "a1", "10.0.0.2:50070", "ready", via = "local network"),
        status.WorkerStatus("laptop", "b2", "address unknown", "unreachable", reason = "timed out"),
    ]
    monkeypatch.setattr(status, "check_workers", lambda *a, **k: rows)
    got = client.get("/api/cluster/machines").json()["machines"]
    assert [(m["name"], m["state"], m["via"], m["reason"]) for m in got] == [
        ("desk", "ready", "local network", None),
        ("laptop", "unreachable", None, "timed out"),
    ]


def test_an_api_key_cannot_open_a_pairing(client):
    client.app.dependency_overrides[cluster.authenticated_via_api_key] = lambda: True
    assert client.post("/api/cluster/pair").status_code == 403
    assert client.get("/api/cluster/machines").status_code == 403


def test_removing_a_machine_drops_its_tunnel_forward(client, monkeypatch):
    from studio.cluster_link import identity, remote

    identity.master_identity(create = True)
    peer = type("P", (), dict(name = "desk", node_id = "a1", removed = False))()
    monkeypatch.setattr(identity.Registry, "workers", lambda self: [peer])
    monkeypatch.setattr(identity.Registry, "remove", lambda self, node_id: True)
    dropped = []
    monkeypatch.setattr(remote, "drop", dropped.append)
    assert client.delete("/api/cluster/machines/a1").status_code == 200
    assert dropped == ["a1"]


class _SlowWindow:
    """A pairing window whose start() waits for ``gate``; records every window opened or closed."""

    made: list = []
    gate = None

    def __init__(self, *a, **k):
        self.closed, self.paired, self.tc, self.expires_at = False, None, None, time.time() + 60
        _SlowWindow.made.append(self)

    async def start(self):
        await _SlowWindow.gate.wait()
        return "unsloth-pair-x"

    async def wait(self, timeout = None):
        await asyncio.sleep(3600)

    def close(self):
        self.closed = True


def test_concurrent_pairing_starts_leave_one_open_window(client, monkeypatch):
    from studio.cluster_link import pairing, remote

    monkeypatch.setattr(remote, "ensure_helper", lambda *a: False)
    monkeypatch.setattr(pairing, "PairingWindow", _SlowWindow)
    _SlowWindow.made = []

    async def run():
        _SlowWindow.gate = asyncio.Event()
        first = asyncio.ensure_future(cluster.start_pairing())
        second = asyncio.ensure_future(cluster.start_pairing())
        await asyncio.sleep(0.2)
        _SlowWindow.gate.set()
        got = await asyncio.gather(first, second)
        closed = [w.closed for w in _SlowWindow.made]
        tracked = list(cluster._PAIRINGS)
        for p in list(cluster._PAIRINGS.values()):
            p.task.cancel()
        return got, closed, tracked

    got, closed, tracked = asyncio.run(run())
    assert len(got) == 2 and closed == [True, False] and tracked == [got[1]["id"]]
    cluster._PAIRINGS.clear()


def test_a_cancelled_pairing_start_closes_its_window(client, monkeypatch):
    from studio.cluster_link import pairing, remote

    monkeypatch.setattr(remote, "ensure_helper", lambda *a: False)
    monkeypatch.setattr(pairing, "PairingWindow", _SlowWindow)
    _SlowWindow.made = []

    async def run():
        _SlowWindow.gate = asyncio.Event()
        task = asyncio.ensure_future(cluster.start_pairing())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [w.closed for w in _SlowWindow.made] == [True] and cluster._PAIRINGS == {}


def test_cancelling_after_the_worker_was_recorded_reports_paired():
    peer = type("Peer", (), dict(name = "desk", node_id = "a1"))()
    window = type("W", (), dict(paired = peer, close = lambda self: None))()
    p = cluster._Pairing(id = "x", window = window, expires_at = 0.0, remote = False, state = "approval")
    cluster._end(p, "cancelled")
    assert (p.state, p.node_id, p.name) == ("paired", "a1", "desk")
