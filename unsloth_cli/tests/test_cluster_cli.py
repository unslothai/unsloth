# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth worker` / `unsloth cluster` wiring against stand-in identity, pairing, agent and relay
modules: the real ones have their own tests."""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import studio  # noqa: E402

if str(REPO / "studio") not in list(studio.__path__):
    studio.__path__.append(str(REPO / "studio"))
link = importlib.import_module("studio.cluster_link")

import unsloth_cli  # noqa: E402
from unsloth_cli.commands import cluster as cli  # noqa: E402

TOKEN = "unsloth-pair-1.c2VjcmV0LXRva2Vu"
FP = "ab" * 32


def ident(role = "worker"):
    return SimpleNamespace(
        node_id = f"{role}-0123456789abcdef",
        role = role,
        cluster_id = "c1",
        certfile = "c",
        keyfile = "k",
        rootfile = "r",
        fingerprint = FP,
    )


def worker_peer(
    name = "desk",
    node_id = "node-aaaa1111",
    removed = False,
):
    return SimpleNamespace(
        node_id = node_id,
        name = name,
        role = "worker",
        fingerprint = FP,
        cert_pem = "",
        addrs = ["192.168.1.20"],
        agent_port = 50070,
        removed = removed,
    )


@pytest.fixture
def fakes(monkeypatch):
    state = SimpleNamespace(
        master = ident("master"),
        worker = None,
        peers = [],
        removed = [],
        left = [],
        agent_port = None,
        recorded = [],
        sightings = {},
        refused = set(),
    )

    class Registry:
        @classmethod
        def open(cls):
            return cls()

        @contextlib.contextmanager
        def transaction(self):
            yield self

        def workers(self):
            return list(state.peers)

        def authorize(self, peer, role):
            return None

        def remove(self, node_id):
            state.removed.append(node_id)
            return True

        def record_address(
            self,
            node_id,
            host,
            port = None,
        ):
            state.recorded.append((node_id, host, port))

    identity = types.ModuleType("studio.cluster_link.identity")
    identity.master_identity = lambda create = True: state.master
    identity.worker_identity = lambda: state.worker
    identity.worker_agent_port = lambda: state.agent_port
    identity.Registry = Registry
    identity.forget_identity = lambda: state.left.append(True)
    identity.forget_master = lambda: state.left.append("master")
    identity.identity_lock = contextlib.nullcontext

    pairing = types.ModuleType("studio.cluster_link.pairing")

    class PairingError(RuntimeError):
        pass

    class PairingWindow:
        def __init__(self, master, host, port, ttl, addrs):
            state.window = SimpleNamespace(
                master = master, host = host, port = port, ttl = ttl, addrs = addrs
            )

        async def start(self):
            return TOKEN

        async def wait(self, timeout):
            if getattr(state, "pair_error", None):
                raise PairingError(state.pair_error)
            return worker_peer()

        def close(self):
            state.window.closed = True

    async def join(
        token,
        name = None,
        devices = None,
        timeout = 15.0,
    ):
        state.joined = (token, name)
        if token == "bad":
            raise ValueError(f"proof rejected for {token}")
        return ident("worker")

    pairing.PairingWindow = PairingWindow
    pairing.PairingError = PairingError
    pairing.PairingUnconfirmed = type("PairingUnconfirmed", (RuntimeError,), {})
    pairing.join = join

    agent = types.ModuleType("studio.cluster_link.agent")

    class Agent:
        def __init__(self, identity, authorize, **kw):
            state.agent = kw

        async def start(self):
            pass

        async def serve_forever(self):
            pass

        tail = []

        async def wait_rpc_ready(self, timeout):
            return True

        async def stop(self):
            state.agent_stopped = True

    agent.Agent = Agent

    relay = types.ModuleType("studio.cluster_link.relay")

    class RelayManager:
        def __init__(self, identity, authorize):
            pass

        def ensure(self, node_id, host, port):
            self.at = (host, port)
            return "127.0.0.1:41000"

        def status(
            self,
            node_id,
            timeout = 3.0,
        ):
            if self.at in state.refused:
                return {"type": "error", "error": "refused", "reachable": False}
            return {"busy": False, "rpc_up": True, "devices": ["CUDA0"]}

        def stop(self):
            pass

    relay.RelayManager = RelayManager

    discovery = types.ModuleType("studio.cluster_link.discovery")

    class Listener:
        def __init__(
            self,
            accept = None,
            **kw,
        ):
            self.accept = accept

        def start(self):
            return self

        def stop(self):
            pass

        error = None

        def candidates(self, node_id):
            return [s for s in state.sightings.get(node_id, []) if self.accept(node_id, s.fp)]

        def latest(self):
            return {n: v[0] for n, v in state.sightings.items() if v}

    discovery.Listener = Listener
    for name, module in (
        ("identity", identity),
        ("pairing", pairing),
        ("agent", agent),
        ("relay", relay),
        ("discovery", discovery),
    ):
        monkeypatch.setitem(sys.modules, f"studio.cluster_link.{name}", module)
        monkeypatch.setattr(link, name, module, raising = False)
    monkeypatch.setattr(cli, "DISCOVERY_SETTLE_S", 0)
    return state


def invoke(*args, input = None):
    return CliRunner().invoke(unsloth_cli.app, list(args), input = input)


def test_commands_are_registered():
    out = invoke("cluster", "--help").output
    for name in ("pair", "join", "list", "discover", "remove", "leave"):
        assert name in out
    assert invoke("worker", "--help").exit_code == 0


def test_pair_prints_the_token_but_never_in_a_command_line(fakes):
    result = invoke("cluster", "pair", "--ttl", "90s", "--addr", "192.168.1.10")
    assert result.exit_code == 0, result.output
    # A token in argv is readable by every local user on the worker while join runs.
    assert TOKEN in result.output and f"join {TOKEN}" not in result.output
    assert "unsloth cluster join" in result.output
    assert "Paired desk" in result.output
    assert fakes.window.ttl == 90 and fakes.window.addrs == ["192.168.1.10"] and fakes.window.closed


def test_join_from_stdin_and_errors_never_show_the_token(fakes):
    result = invoke("cluster", "join", "--token-stdin", "--name", "box", input = TOKEN + "\n")
    assert result.exit_code == 0, result.output
    assert fakes.joined == (TOKEN, "box")
    assert TOKEN not in result.output
    hidden = invoke("cluster", "join", "--name", "box2", input = TOKEN + "\n")  # the hidden prompt
    assert hidden.exit_code == 0 and fakes.joined == (TOKEN, "box2") and TOKEN not in hidden.output
    assert invoke("cluster", "join", TOKEN).exit_code != 0  # no positional token at all
    bad = invoke("cluster", "join", input = "bad\n")
    assert (
        bad.exit_code == 1
        and "Pairing failed" in bad.output
        and "proof rejected for <token>" in bad.output
    )


def test_worker_needs_pairing_first(fakes):
    result = invoke("worker")
    assert result.exit_code == 1 and "unsloth cluster join" in result.output


def test_worker_needs_the_rpc_server(fakes, monkeypatch):
    fakes.worker = ident("worker")
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: None)
    result = invoke("worker")
    assert result.exit_code == 1 and "install Studio" in result.output


def test_worker_starts_the_agent_on_loopback_rpc(fakes, monkeypatch):
    fakes.worker = ident("worker")
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: "/opt/llama/ggml-rpc-server")
    result = invoke("worker", "--no-discovery", "--device", "CUDA0", "--rpc-port", "50111")
    assert result.exit_code == 0, result.output
    assert fakes.agent["rpc_binary"] == "/opt/llama/ggml-rpc-server"
    assert fakes.agent["rpc_port"] == 50111 and fakes.agent["rpc_args"] == ("--device", "CUDA0")
    assert fakes.agent_stopped


def test_list_remove_and_leave(fakes):
    fakes.peers = [worker_peer(), worker_peer("laptop", "node-bbbb2222")]
    listed = invoke("cluster", "list")
    assert listed.exit_code == 0 and "desk" in listed.output and "ready" in listed.output
    assert invoke("cluster", "remove", "laptop").exit_code == 0
    assert fakes.removed == ["node-bbbb2222"]
    assert invoke("cluster", "remove", "node-a").exit_code == 0
    assert invoke("cluster", "remove", "nobody").exit_code == 1
    fakes.worker = ident("worker")
    assert invoke("cluster", "leave", "--yes").exit_code == 0 and fakes.left == [True]


@pytest.mark.parametrize("text, seconds", [("90", 90), ("30s", 30), ("10m", 600), ("1h", 3600)])
def test_parse_duration(text, seconds):
    assert cli.parse_duration(text) == seconds


def test_parse_duration_rejects_nonsense():
    import typer
    for bad in ("", "abc", "0", "-5m", "5d"):
        with pytest.raises(typer.BadParameter):
            cli.parse_duration(bad)


def test_pairing_without_cryptography_says_what_to_install(fakes, monkeypatch):
    # Bare `pip install unsloth` has no cryptography: a clear message, not an ImportError traceback.
    monkeypatch.setitem(sys.modules, "cryptography", None)
    for args in (("cluster", "pair"), ("cluster", "join", "--token-stdin"), ("worker",)):
        result = invoke(*args, input = "x\n")
        assert result.exit_code == 1, (args, result.output)
        assert "cryptography" in result.output and "Traceback" not in result.output
    assert invoke("cluster", "discover", "--seconds", "0.1").exit_code == 0


def test_a_pairing_failure_is_a_clean_message(fakes):
    fakes.pair_error = "too many wrong proofs"
    result = invoke("cluster", "pair")
    assert result.exit_code == 1 and "too many wrong proofs" in result.output
    assert "Traceback" not in result.output and fakes.window.closed


def test_pair_on_a_worker_machine_says_so(fakes, monkeypatch):
    def refuse(create = True):
        raise RuntimeError("this machine is paired as a worker; run `unsloth cluster leave` first")

    monkeypatch.setattr(fakes_identity(), "master_identity", refuse)
    result = invoke("cluster", "pair")
    assert result.exit_code == 1 and "cluster leave" in result.output


def test_worker_devices_accept_commas_and_repeats(fakes, monkeypatch):
    fakes.worker = ident("worker")
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: "/bin/true")
    result = invoke("worker", "--no-discovery", "--device", "CUDA0,CUDA1", "--device", "Vulkan0")
    assert result.exit_code == 0, result.output
    assert fakes.agent["rpc_args"] == ("--device", "CUDA0,CUDA1,Vulkan0")


def fakes_identity():
    return sys.modules["studio.cluster_link.identity"]


def test_list_reports_a_refused_worker_as_unreachable_not_rpc_down(fakes):
    fakes.peers = [worker_peer()]
    fakes.refused = {("192.168.1.20", 50070)}
    listed = invoke("cluster", "list")
    assert listed.exit_code == 0 and "unreachable (refused)" in listed.output
    assert "rpc down" not in listed.output


def test_list_finds_a_moved_worker_through_discovery_and_records_it(fakes):
    fakes.peers = [worker_peer()]
    fakes.refused = {("192.168.1.20", 50070)}
    seen = SimpleNamespace(host = "192.168.1.31", agent_port = 50123, fp = FP, seen_at = 0.0)
    spoof = SimpleNamespace(host = "192.168.1.66", agent_port = 50070, fp = "f" * 64, seen_at = 0.0)
    fakes.sightings = {"node-aaaa1111": [spoof, seen]}
    listed = invoke("cluster", "list")
    assert "192.168.1.31:50123" in listed.output and "ready" in listed.output, listed.output
    assert "found by discovery" in listed.output
    assert fakes.recorded == [("node-aaaa1111", "192.168.1.31", 50123)]


def test_join_records_the_worker_port_and_worker_defaults_to_it(fakes, monkeypatch):
    seen = {}

    async def join(
        token,
        name = None,
        **kw,
    ):
        seen.update(kw)
        return ident("worker")

    monkeypatch.setattr(sys.modules["studio.cluster_link.pairing"], "join", join)
    assert (
        invoke("cluster", "join", "--token-stdin", "--port", "50111", input = TOKEN + "\n").exit_code
        == 0
    )
    assert seen == {"agent_port": 50111}
    fakes.worker = ident("worker")
    fakes.agent_port = 50111
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: "/bin/true")
    started = invoke("worker", "--no-discovery")
    assert started.exit_code == 0 and fakes.agent["port"] == 50111, started.output
    moved = invoke("worker", "--no-discovery", "--port", "50222")
    assert moved.exit_code == 0 and fakes.agent["port"] == 50222
    assert "only through LAN discovery" in moved.output


def test_names_from_other_machines_cannot_write_control_sequences(fakes):
    fakes.peers = [worker_peer(name = "evil\x1b]52;c;QUJD\x07")]
    listed = invoke("cluster", "list", "--no-check")
    assert listed.exit_code == 0 and "\x1b" not in listed.output and "\x07" not in listed.output
    assert "evil?]52;c;QUJD?" in listed.output


def test_leave_resets_a_master_nobody_joined_but_not_one_with_workers(fakes):
    # A `cluster pair` that nobody joined still made this machine a master: join refuses on a master,
    # so leave is the way back.
    fakes.peers = [worker_peer()]
    refused = invoke("cluster", "leave", "--yes")
    assert refused.exit_code == 1 and "cluster remove" in refused.output and fakes.left == []
    fakes.peers = []
    reset = invoke("cluster", "leave", "--yes")
    assert reset.exit_code == 0 and fakes.left == ["master"]


def test_discover_calls_a_worker_paired_only_when_id_and_fingerprint_both_match(fakes):
    fakes.peers = [worker_peer()]
    honest = SimpleNamespace(host = "192.168.1.20", agent_port = 50070, fp = FP, seen_at = 0.0)
    copied = SimpleNamespace(host = "192.168.1.66", agent_port = 50070, fp = FP, seen_at = 0.0)
    fakes.sightings = {"node-aaaa1111": [honest], "node-evil0000": [copied]}
    out = invoke("cluster", "discover", "--seconds", "0").output
    lines = {l.split()[0]: l for l in out.splitlines() if l.strip()}
    assert "paired as desk" in lines["node-aaaa111"]
    assert "not paired" in lines["node-evil000"]


def test_worker_start_problems_are_messages_and_always_stop_the_agent(fakes, monkeypatch):
    fakes.worker = ident("worker")
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: "/bin/true")
    agent_cls = sys.modules["studio.cluster_link.agent"].Agent
    for exc, text in (
        (RuntimeError("another `unsloth worker` (pid 7) already runs"), "pid 7"),
        (OSError(98, "Address already in use"), "Could not listen on port"),
    ):

        async def start(self, exc = exc):
            raise exc

        monkeypatch.setattr(agent_cls, "start", start)
        fakes.agent_stopped = False
        result = invoke("worker", "--no-discovery")
        assert result.exit_code == 1 and text in result.output and "Traceback" not in result.output
        assert fakes.agent_stopped  # stop() ran although start() never finished


def test_list_tries_older_recorded_addresses_before_discovery(fakes):
    fakes.peers = [worker_peer()]
    fakes.peers[0].addrs = ["192.168.1.99", "192.168.1.20"]
    fakes.refused = {("192.168.1.99", 50070)}
    listed = invoke("cluster", "list")
    assert "192.168.1.20:50070" in listed.output and "ready" in listed.output, listed.output
    assert "Looking for" not in listed.output  # no discovery wait: the registry already knew
    assert fakes.recorded == [("node-aaaa1111", "192.168.1.20", 50070)]
