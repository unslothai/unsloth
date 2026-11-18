# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth worker` / `unsloth cluster` wiring against stand-in cluster_link modules."""

from __future__ import annotations

import contextlib
import errno
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import studio  # noqa: E402

if str(REPO / "studio") not in list(studio.__path__):
    studio.__path__.append(str(REPO / "studio"))
link = importlib.import_module("studio.cluster_link")

import unsloth_cli  # noqa: E402
from studio.cluster_link.tokens import TokenError  # noqa: E402
from unsloth_cli.commands import cluster as cli  # noqa: E402

TOKEN = "unsloth-pair-1.c2VjcmV0LXRva2Vu"
FP = "ab" * 32
HOME = ("192.168.1.20", 50070)
RPC_BIN = "/opt/llama/ggml-rpc-server"


def ident(role = "worker"):
    return SimpleNamespace(
        node_id = f"{role}-0123456789abcdef", role = role, cluster_id = "c1",
        certfile = "c", keyfile = "k", rootfile = "r", fingerprint = FP,
    )  # fmt: skip


def worker_peer(name = "desk", node_id = "node-aaaa1111"):
    return SimpleNamespace(
        node_id = node_id, name = name, role = "worker", fingerprint = FP, cert_pem = "",
        addrs = [HOME[0]], agent_port = HOME[1], removed = False,
    )  # fmt: skip


def sighting(host, port = 50070):
    return SimpleNamespace(host = host, agent_port = port, fp = FP, seen_at = 0.0)


def mod(name):
    return sys.modules[f"studio.cluster_link.{name}"]


def coro(fn):
    async def run(*args, **kw):
        return fn(*args, **kw)

    return run


@pytest.fixture
def fakes(monkeypatch):
    state = SimpleNamespace(
        master = ident("master"), worker = None, peers = [], removed = [], left = [],
        agent_port = None, recorded = [], sightings = {}, errors = {}, pair_error = None,
    )  # fmt: skip
    PairingError = type("PairingError", (RuntimeError,), {})

    class Registry:
        load = lambda self: self
        transaction = lambda self: contextlib.nullcontext(self)
        workers = lambda self: list(state.peers)
        master = lambda self: None
        authorize = lambda self, peer, role: None
        remove = lambda self, node_id: state.removed.append(node_id) or True
        record_address = lambda self, *args: state.recorded.append(args)

    class PairingWindow:
        __init__ = lambda self, master, host, port, ttl, addrs, approve = None: setattr(
            state, "window", SimpleNamespace(ttl = ttl, addrs = addrs, approve = approve)
        )
        tc = None
        start = coro(lambda self: TOKEN)
        close = lambda self: setattr(state.window, "closed", True)

        async def wait(self, timeout):
            if state.pair_error:
                raise PairingError(state.pair_error)
            return worker_peer()

    class Agent:
        tail = []
        __init__ = lambda self, identity, authorize, **kw: setattr(state, "agent", kw)
        start = serve_forever = wait_rpc_ready = coro(lambda self, *args: True)
        stop = coro(lambda self: setattr(state, "agent_stopped", True))

    class RelayManager:
        __init__ = stop = lambda self, *args: None
        ensure = lambda self, node_id, *at: setattr(self, "at", at) or "127.0.0.1:41000"

        def status(self, node_id, **kw):
            if self.at in state.errors:
                return {"type": "error", "error": state.errors[self.at], "reachable": False}
            return {"busy": False, "rpc_up": True, "devices": ["CUDA0"]}

    class Listener:
        error = None
        start, stop = (lambda self: self), (lambda self: None)
        __init__ = lambda self, accept = None: setattr(self, "accept", accept) or setattr(
            state, "accept", accept
        )
        latest = lambda self: {n: v[0] for n, v in state.sightings.items() if v}

        def candidates(self, node_id):
            return [s for s in state.sightings.get(node_id, []) if self.accept(node_id, s.fp)]

    async def join(token, **kw):
        state.joined = (token, kw.get("name"))
        return ident("worker")

    modules = {
        "identity": dict(
            master_identity = lambda create = True: state.master, worker_identity = lambda: state.worker,
            worker_agent_port = lambda: state.agent_port, Registry = Registry,
            forget_identity = lambda: state.left.append(True),
            forget_master = lambda: state.left.append("master"), identity_lock = contextlib.nullcontext,
        ),
        "pairing": dict(
            PairingWindow = PairingWindow, PairingError = PairingError, join = join,
            PairingUnconfirmed = type("PairingUnconfirmed", (RuntimeError,), {}),
        ),
        "agent": dict(Agent = Agent), "relay": dict(RelayManager = RelayManager),
        "discovery": dict(Listener = Listener),
    }  # fmt: skip
    for name, attrs in modules.items():
        module = types.ModuleType(f"studio.cluster_link.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(link, name, module, raising = False)
    monkeypatch.setattr(cli, "DISCOVERY_SETTLE_S", 0)
    return state


@pytest.fixture
def worker(fakes, monkeypatch):
    fakes.worker = ident("worker")
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: RPC_BIN)
    return fakes


def invoke(*args, input = None):
    return CliRunner().invoke(unsloth_cli.app, list(args), input = input)


def test_commands_are_registered():
    out = invoke("cluster", "--help").output
    assert all(name in out for name in ("pair", "join", "list", "discover", "remove", "leave"))
    assert invoke("worker", "--help").exit_code == 0


def test_pair_prints_the_token_but_never_in_a_command_line(fakes):
    result = invoke("cluster", "pair", "--ttl", "90s", "--addr", "192.168.1.10")
    assert result.exit_code == 0, result.output
    # A token in argv is readable by every local user on the worker while join runs.
    assert TOKEN in result.output and f"join {TOKEN}" not in result.output
    assert "unsloth worker --join" in result.output and "Paired desk" in result.output
    assert fakes.window.ttl == 90 and fakes.window.addrs == ["192.168.1.10"] and fakes.window.closed


def test_pair_failures_are_clean_messages(fakes, monkeypatch):
    fakes.pair_error = "too many wrong proofs"
    result = invoke("cluster", "pair")
    assert result.exit_code == 1 and "too many wrong proofs" in result.output
    assert "Traceback" not in result.output and fakes.window.closed

    def on_a_worker(create = True):
        raise RuntimeError("this machine is paired as a worker; run `unsloth cluster leave` first")

    monkeypatch.setattr(mod("identity"), "master_identity", on_a_worker)
    result = invoke("cluster", "pair")
    assert result.exit_code == 1 and "cluster leave" in result.output


def test_join_reads_the_token_from_stdin_or_a_hidden_prompt_only(fakes):
    result = invoke("cluster", "join", "--token-stdin", "--name", "box", input = TOKEN + "\n")
    assert result.exit_code == 0 and fakes.joined == (TOKEN, "box") and TOKEN not in result.output
    hidden = invoke("cluster", "join", "--name", "box2", input = TOKEN + "\n")
    assert hidden.exit_code == 0 and fakes.joined == (TOKEN, "box2") and TOKEN not in hidden.output
    assert invoke("cluster", "join", TOKEN).exit_code != 0  # no positional token at all


JOIN_ERRORS = {
    "unexpected-error-redacted": (lambda p: ValueError(f"proof rejected for {TOKEN}"), "Pairing failed: ValueError: proof rejected for <token>"),
    "token-error": (lambda p: TokenError("not an unsloth pairing token"), "unsloth-pair-"),
    "master-unreachable": (lambda p: p.PairingError("could not reach the master at 10.0.0.2 port 50071: refused"), "still running there"),
}  # fmt: skip


@pytest.mark.parametrize("case", JOIN_ERRORS, ids = str)
def test_join_errors_read_as_sentences_and_never_show_the_token(fakes, monkeypatch, case):
    make, wanted = JOIN_ERRORS[case]
    pairing = mod("pairing")
    exc = make(pairing)

    async def join(token, **kw):
        raise exc

    monkeypatch.setattr(pairing, "join", join)
    for args in (("--token-stdin",), ()):  # stdin and the hidden prompt
        result = invoke("cluster", "join", *args, input = TOKEN + "\n")
        assert result.exit_code == 1 and wanted in result.output, result.output
        # Only an unexpected error names its type; none ever shows the token.
        assert (type(exc).__name__ in result.output) == (case == "unexpected-error-redacted")
        assert TOKEN not in result.output and "Traceback" not in result.output


def test_join_records_the_worker_port_and_worker_defaults_to_it(worker, monkeypatch):
    seen = {}

    async def join(token, **kw):
        kw.pop("name")
        assert callable(kw.pop("on_words"))
        seen.update(kw)
        return ident("worker")

    monkeypatch.setattr(mod("pairing"), "join", join)
    joined = invoke("cluster", "join", "--token-stdin", "--port", "50111", input = TOKEN + "\n")
    assert joined.exit_code == 0 and seen == {"agent_port": 50111}
    worker.agent_port = 50111
    started = invoke("worker", "--no-discovery")
    assert started.exit_code == 0 and worker.agent["port"] == 50111, started.output
    moved = invoke("worker", "--no-discovery", "--port", "50222")
    assert moved.exit_code == 0 and worker.agent["port"] == 50222
    assert "only through LAN discovery" in moved.output


def test_worker_starts_the_agent_on_loopback_rpc_with_merged_devices(worker):
    args = ("--device", "CUDA0,CUDA1", "--device", "Vulkan0", "--rpc-port", "50111")
    result = invoke("worker", "--no-discovery", *args)
    assert result.exit_code == 0, result.output
    assert worker.agent["rpc_binary"] == RPC_BIN and worker.agent["rpc_port"] == 50111
    assert worker.agent["rpc_args"] == ("--device", "CUDA0,CUDA1,Vulkan0") and worker.agent_stopped


START_ERRORS = {
    "already-running": (RuntimeError("another `unsloth worker` (pid 7) already runs"), 1, ["pid 7"]),
    "port-in-use": (
        OSError(errno.EADDRINUSE, "error while attempting to bind: address already in use"),
        1, ["Could not listen on port", "already in use", "another `unsloth worker`"],
    ),
    "ctrl-c": (KeyboardInterrupt(), 0, ["Worker stopped."]),
}  # fmt: skip


@pytest.mark.parametrize("case", START_ERRORS, ids = str)
def test_worker_start_problems_are_messages_and_always_stop_the_agent(worker, monkeypatch, case):
    exc, code, texts = START_ERRORS[case]

    async def start(self):
        raise exc

    monkeypatch.setattr(mod("agent").Agent, "start", start)
    result = invoke("worker", "--no-discovery")
    assert result.exit_code == code and all(t in result.output for t in texts), result.output
    assert "Traceback" not in result.output and "Errno" not in result.output
    assert worker.agent_stopped  # stop() ran although start() never finished


RANGE = (2, ("65535", "range"))
DIFFER = (1, ["must all differ"])  # native restarts rpc-server on --native-port
REFUSALS = {
    "port-out-of-range": (("worker", "--port", "70000"), *RANGE),
    "rpc-port-zero": (("worker", "--rpc-port", "0"), *RANGE),
    "native-port-out-of-range": (("worker", "--native-port", "65536"), *RANGE),
    "pair-port-out-of-range": (("cluster", "pair", "--port", "70000"), *RANGE),
    "join-port-out-of-range": (("cluster", "join", "--port", "70000", "--token-stdin"), *RANGE),
    "rpc-equals-native": (("worker", "--no-discovery", "--rpc-port", "50111", "--native-port", "50111"), *DIFFER),
    "agent-equals-native": (("worker", "--no-discovery", "--port", "50070", "--native-port", "50070"), *DIFFER),
    "agent-equals-rpc": (("worker", "--no-discovery", "--port", "50111", "--rpc-port", "50111"), *DIFFER),
    "unpaired": (("worker",), 1, ["unsloth worker --join"]),
    "no-rpc-server": (("worker",), 1, ["install Studio"]),
}  # fmt: skip


@pytest.mark.parametrize("case", REFUSALS, ids = str)
def test_bad_ports_and_missing_setup_are_refused_before_the_agent(worker, monkeypatch, case):
    # A range error is a CLI error, not an OverflowError from socket.bind(), and never saved by join.
    args, code, texts = REFUSALS[case]
    if case == "unpaired":
        worker.worker = None
    elif case == "no-rpc-server":
        monkeypatch.setattr(cli, "_rpc_server_binary", lambda: None)
    result = invoke(*args, input = "x\n")
    assert result.exit_code == code and any(t in result.output for t in texts), result.output
    assert getattr(worker, "agent", None) is None


def test_pairing_without_cryptography_says_what_to_install(fakes, monkeypatch):
    # Bare `pip install unsloth` has no cryptography: a clear message, not an ImportError traceback.
    monkeypatch.setitem(sys.modules, "cryptography", None)
    for args in (("cluster", "pair"), ("cluster", "join", "--token-stdin"), ("worker",)):
        result = invoke(*args, input = "x\n")
        assert result.exit_code == 1, (args, result.output)
        assert "cryptography" in result.output and "Traceback" not in result.output
    assert invoke("cluster", "discover", "--seconds", "0.1").exit_code == 0


DURATIONS = {"90": 90, "30s": 30, "10m": 600, "1h": 3600, "": None, "abc": None, "0": None, "-5m": None, "5d": None}  # fmt: skip


@pytest.mark.parametrize("text", DURATIONS)
def test_parse_duration(text):
    if DURATIONS[text] is not None:
        assert cli.parse_duration(text) == DURATIONS[text]
        return
    with pytest.raises(typer.BadParameter):
        cli.parse_duration(text)


def test_cluster_list_reads_host_and_port_pairs():
    peer = SimpleNamespace(addrs = ["10.0.0.5:50099", "10.0.0.5:50070", "10.0.0.9"], agent_port = 50099)
    assert cli._addresses(peer) == [("10.0.0.5", 50099), ("10.0.0.5", 50070), ("10.0.0.9", 50099)]


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


def test_leave_resets_a_master_nobody_joined_but_not_one_with_workers(fakes):
    fakes.peers = [worker_peer()]
    refused = invoke("cluster", "leave", "--yes")
    assert refused.exit_code == 1 and "cluster remove" in refused.output and fakes.left == []
    fakes.peers = []
    reset = invoke("cluster", "leave", "--yes")
    assert reset.exit_code == 0 and fakes.left == ["master"]


@pytest.mark.parametrize("role", ["worker", "master"])
def test_leave_deletes_the_tunnel_keys_under_the_identity_lock(fakes, monkeypatch, role):
    events = []

    @contextlib.contextmanager
    def lock():
        events.append("lock")
        yield
        events.append("unlock")

    monkeypatch.setattr(link.identity, "identity_lock", lock)
    remote = SimpleNamespace(forget_keys = lambda held = False: events.append(("keys", held)))
    monkeypatch.setitem(sys.modules, "studio.cluster_link.remote", remote)
    monkeypatch.setattr(link, "remote", remote, raising = False)
    fakes.peers = []
    fakes.worker = ident("worker") if role == "worker" else None
    assert invoke("cluster", "leave", "--yes").exit_code == 0
    assert events == ["lock", ("keys", True), "unlock"]


SPOOF = SimpleNamespace(host = "192.168.1.66", agent_port = 50070, fp = "f" * 64, seen_at = 0.0)
ERRNO = "cannot reach node-aaaa111 at 192.168.1.20:50070: [Errno 111] Connect call failed"
MOVED = {"node-aaaa1111": [SPOOF, sighting("192.168.1.31", 50123)]}  # the spoof has another key
LIST_CASES = {
    "fresh-machine": (dict(master = None), ["unsloth cluster pair"], [], []),
    "on-a-worker": (dict(master = None, worker = ident()), ["is a worker", "on its master"], [], []),
    "refused-is-not-rpc-down": (dict(errors = {HOME: "refused"}), ["unreachable (refused)"], ["rpc down"], []),
    "errno-in-words": (dict(errors = {HOME: ERRNO}), ["unreachable (not running: start `unsloth worker` there)"], ["Errno"], []),
    "moved-worker-found-by-discovery-not-spoof": (
        dict(errors = {HOME: "refused"}, sightings = MOVED),
        ["192.168.1.31:50123", "ready", "found by discovery"], [], [("node-aaaa1111", "192.168.1.31", 50123)],
    ),
    "older-addresses-before-discovery": (
        dict(addrs = ["192.168.1.99", HOME[0]], errors = {("192.168.1.99", 50070): "refused"}),
        ["192.168.1.20:50070", "ready"], ["Looking for"], [("node-aaaa1111", *HOME)],
    ),
    "names-cannot-write-control-sequences": (dict(name = "evil\x1b]52;c;QUJD\x07"), ["evil?]52;c;QUJD?"], ["\x1b", "\x07"], []),
}  # fmt: skip


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_list_points_a_mac_with_no_route_at_local_network_access(fakes, monkeypatch, platform):
    monkeypatch.setattr(cli.sys, "platform", platform)
    fakes.peers, fakes.errors = [worker_peer()], {HOME: "[Errno 65] No route to host"}
    listed = invoke("cluster", "list")
    assert listed.exit_code == 0, listed.output
    assert "unreachable (no route to it on the network" in listed.output
    hint = "allow Local Network access for the app running Unsloth in System Settings > Privacy & Security > Local Network"
    assert (hint in listed.output) == (platform == "darwin"), listed.output


def test_list_discovery_takes_a_paired_node_id_only_under_its_paired_fingerprint(fakes):
    fakes.peers, fakes.errors = [worker_peer()], {HOME: "refused"}
    assert invoke("cluster", "list").exit_code == 0
    assert fakes.accept("node-aaaa1111", FP)
    assert not fakes.accept("node-aaaa1111", SPOOF.fp) and not fakes.accept("node-other", FP)


@pytest.mark.parametrize("case", LIST_CASES, ids = str)
def test_list_reports_each_worker_in_words(fakes, case):
    setup, has, lacks, recorded = LIST_CASES[case]
    setup = dict(setup)
    peer = worker_peer(setup.pop("name", "desk"))
    peer.addrs = setup.pop("addrs", peer.addrs)
    fakes.peers = [peer]
    fakes.__dict__.update(setup)
    listed = invoke("cluster", "list", *(["--no-check"] if "\x1b" in lacks else []))
    assert listed.exit_code == 0, listed.output
    assert all(t in listed.output for t in has), listed.output
    assert not any(t in listed.output for t in lacks), listed.output
    assert fakes.recorded == recorded


def test_discover_calls_a_worker_paired_only_when_id_and_fingerprint_both_match(fakes):
    fakes.peers = [worker_peer()]
    honest, copied = sighting(HOME[0]), sighting("192.168.1.66")
    fakes.sightings = {"node-aaaa1111": [honest], "node-evil0000": [copied]}
    out = invoke("cluster", "discover", "--seconds", "0")
    assert out.exit_code == 0, out.output
    lines = {l.split()[0]: l for l in out.output.splitlines() if l.strip()}
    assert "paired as desk" in lines["node-aaaa111"] and "not paired" in lines["node-evil000"]


def test_discover_rejects_negative_seconds(fakes):
    bad = invoke("cluster", "discover", "--seconds", "-1")
    assert bad.exit_code == 2 and "Traceback" not in bad.output
    assert not isinstance(bad.exception, ValueError)


def test_list_names_how_each_worker_is_reached(fakes, monkeypatch):
    # desk answers on the LAN; laptop and phone only through tailcat, directly and relayed.
    desk, laptop, phone = (
        worker_peer(),
        worker_peer("laptop", "node-bbbb2222"),
        worker_peer("phone", "node-cccc3333"),
    )
    laptop.addrs, phone.addrs = ["10.9.9.1"], ["10.9.9.2"]
    laptop.tc_addr, phone.tc_addr = "tcLAPTOP0123456", "tcPHONE01234567"
    fakes.peers = [desk, laptop, phone]
    fakes.errors = {("10.9.9.1", 50070): "timed out", ("10.9.9.2", 50070): "timed out"}
    ports = {"node-bbbb2222": 47001, "node-cccc3333": 47002}
    relays_allowed = [False]
    remote = SimpleNamespace(
        enabled = lambda: True,
        ensure_helper = lambda note = None: True,
        relay_allowed = lambda: relays_allowed[0],
        forward = lambda node_id, address, port: ports[node_id],
        path = lambda node_id: SimpleNamespace(direct = node_id == "node-bbbb2222"),
        forwarded = lambda: list(ports),
        drop = lambda node_id: None,
        close = lambda: None,
    )
    monkeypatch.setitem(sys.modules, "studio.cluster_link.remote", remote)
    monkeypatch.setattr(link, "remote", remote, raising = False)
    out = invoke("cluster", "list").output
    lines = {line.split()[0]: line for line in out.splitlines() if line.split()}
    assert "ready (local network)" in lines["desk"], out
    assert "ready (direct internet connection)" in lines["laptop"], out
    assert "relayed only: too slow to serve models" in lines["phone"], out
    assert fakes.recorded == []  # a tunnel's local end is never saved as an address
    relays_allowed[0] = True  # UNSLOTH_CLUSTER_ALLOW_RELAY=1: it serves, slowly
    out = invoke("cluster", "list").output
    lines = {line.split()[0]: line for line in out.splitlines() if line.split()}
    assert "ready (through a relay: slow)" in lines["phone"] and "too slow" not in out, out


def test_worker_join_pairs_with_the_code_then_serves(fakes, monkeypatch):
    monkeypatch.setattr(cli, "_rpc_server_binary", lambda: RPC_BIN)
    seen = {}

    async def join(token, **kw):
        seen.update(token = token, **kw)
        kw["on_words"](["amber", "falcon", "river"])
        fakes.worker = ident("worker")
        return fakes.worker

    monkeypatch.setattr(mod("pairing"), "join", join)
    result = invoke("worker", "--join", "--no-discovery", "--name", "box", input = TOKEN + "\n")
    assert result.exit_code == 0, result.output
    assert seen["token"] == TOKEN and seen["name"] == "box" and TOKEN not in result.output
    assert "amber falcon river" in result.output and fakes.agent["tunnel_allow"]() == []
    assert invoke("worker", "--join", TOKEN).exit_code != 0  # never a code in argv


def test_pair_asks_the_owner_unless_yes(fakes):
    assert invoke("cluster", "pair").exit_code == 0 and callable(fakes.window.approve)
    assert invoke("cluster", "pair", "--yes").exit_code == 0 and fakes.window.approve is None
