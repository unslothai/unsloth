# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth worker` and `unsloth cluster`: pair machines on a home network and serve one model across
them. Every link is mutual TLS from the pairing; ggml-rpc-server never listens beyond loopback."""

from __future__ import annotations

import asyncio
import signal
import re
import socket
import sys
import time
from typing import Any, List, Optional

import typer

cluster_app = typer.Typer(
    help = "Pair machines on this network and serve one model across them.",
    no_args_is_help = True,
)


def _say(message: str = "") -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg = typer.colors.RED, err = True)
    raise typer.Exit(code = 1)


def _link(crypto: bool = False):
    try:
        from unsloth_cli._inference import ensure_studio_backend_path

        # So cluster state resolves through Studio's own studio_root() (UNSLOTH_HOME, venv layout),
        # the same directory the Studio backend reads.
        ensure_studio_backend_path(seed_cache_env = False)
    except Exception:
        pass
    try:
        from studio.cluster_link import identity
    except ImportError as exc:
        _fail(f"This install has no cluster support ({exc}). Reinstall Unsloth Studio.")
    if crypto:
        # Certificates and TLS identities need it; it ships with the Studio extra, not bare unsloth.
        try:
            import cryptography  # noqa: F401
        except ImportError:
            _fail(
                "Cluster pairing needs the `cryptography` package, which comes with Unsloth Studio. "
                "Install Studio (or `pip install cryptography`) and run this again."
            )
    return identity


def _registry(identity: Any) -> Any:
    return identity.Registry.open()


def parse_duration(text: str) -> float:
    """``90``, ``30s``, ``10m``, ``1h`` -> seconds."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text or "")
    if not match:
        raise typer.BadParameter(f"not a duration: {text!r} (use 90, 30s, 10m or 1h)")
    value = float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    if value <= 0:
        raise typer.BadParameter("the duration must be positive")
    return value


def _rpc_server_binary() -> Optional[str]:
    try:
        from unsloth_cli._inference import ensure_studio_backend_path

        ensure_studio_backend_path(seed_cache_env = False)
        from core.inference.spark_serving import rpc_server_binary

        return rpc_server_binary()
    except Exception:
        return None


def _rpc_server_env(binary: str) -> Optional[dict]:
    """Studio's env for its llama.cpp build (library paths for the CUDA / ROCm runtime it ships with)."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._llama_server_env_for_binary(binary)
    except Exception:
        return None


def _find_worker(registry: Any, target: str) -> Any:
    workers = [p for p in registry.workers() if not getattr(p, "removed", False)]
    exact = [p for p in workers if target in (p.node_id, p.name)]
    if len(exact) == 1:
        return exact[0]
    prefix = [p for p in workers if p.node_id.startswith(target)]
    if len(prefix) == 1:
        return prefix[0]
    if not exact and not prefix:
        _fail(f"No paired worker is called {target!r}. See `unsloth cluster list`.")
    _fail(f"{target!r} matches more than one worker; use its id from `unsloth cluster list`.")


def worker(
    port: int = typer.Option(None, "--port", help = "Port the agent listens on (all interfaces)."),
    rpc_port: int = typer.Option(None, "--rpc-port", help = "Loopback port for ggml-rpc-server."),
    device: List[str] = typer.Option(
        [],
        "--device",
        help = "Devices ggml-rpc-server exposes (repeatable or comma-separated), e.g. CUDA0.",
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help = "Where ggml-rpc-server caches tensors (-c)."
    ),
    no_discovery: bool = typer.Option(
        False, "--no-discovery", help = "Do not announce this worker on the LAN."
    ),
    allow_any_network: bool = typer.Option(
        False,
        "--allow-any-network",
        help = "Accept the master from a non-private address (e.g. a VPN range). Still needs pairing.",
    ),
):
    """Serve this machine's GPUs to the master it was paired with."""
    identity = _link(crypto = True)
    from studio.cluster_link import AGENT_PORT_DEFAULT, RPC_PORT_DEFAULT

    me = identity.worker_identity()
    if me is None:
        _fail(
            "This machine is not paired yet. On the master run `unsloth cluster pair`, then run "
            "the `unsloth cluster join ...` command it prints here."
        )
    binary = _rpc_server_binary()
    if not binary:
        _fail(
            "ggml-rpc-server was not found. `unsloth worker` uses Unsloth Studio's llama.cpp: "
            "install Studio on this machine first."
        )
    from studio.cluster_link.agent import Agent

    def authorize(peer: Any) -> bool:
        try:
            return _registry(identity).authorize(peer, "master") is not None
        except Exception:
            return False

    paired_port = identity.worker_agent_port() or AGENT_PORT_DEFAULT
    if port and port != paired_port:
        _say(
            f"Note: this worker paired on port {paired_port}; the master finds port {port} only through "
            "LAN discovery. Re-pair with `unsloth cluster join --port` if discovery is blocked here."
        )
    port = port or paired_port
    rpc_port = rpc_port or RPC_PORT_DEFAULT
    rpc_args: List[str] = []
    names = [n.strip() for value in device for n in value.split(",") if n.strip()]
    if names:
        rpc_args += ["--device", ",".join(names)]
    agent = Agent(
        me,
        authorize,
        host = "0.0.0.0",
        port = port,
        rpc_binary = binary,
        rpc_port = rpc_port,
        rpc_args = tuple(rpc_args),
        cache_dir = cache_dir,
        allow_any_network = allow_any_network,
        rpc_env = _rpc_server_env(binary),
    )
    announcer = None
    if not no_discovery:
        from studio.cluster_link.discovery import Announcer
        announcer = Announcer(me.node_id, port, me.fingerprint).start()

    async def serve() -> None:
        # SIGTERM / SIGHUP end the run like Ctrl-C, so stop() reaps rpc-server instead of orphaning it.
        task = asyncio.current_task()
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
            if sig is not None and task is not None:
                try:
                    asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
                except (NotImplementedError, RuntimeError):
                    pass
        # One guard from the first spawn: a signal during start-up must still reap rpc-server.
        try:
            await agent.start()
            # Say "serving" only once rpc-server accepts: a load that arrives before would be refused.
            if await agent.wait_rpc_ready(60.0):
                _say(
                    f"Worker {me.node_id[:12]} serving on port {port}; ggml-rpc-server stays on "
                    f"127.0.0.1:{rpc_port}. Ctrl-C to stop."
                )
            else:
                _say(
                    "ggml-rpc-server has not come up yet; the master will see this worker as rpc down:"
                )
                for line in list(agent.tail)[-10:]:
                    _say("  " + line)
            await agent.serve_forever()
        finally:
            await agent.stop()

    try:
        asyncio.run(serve())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except RuntimeError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"Could not listen on port {port}: {exc}. Pick another with --port.")
    finally:
        if announcer is not None:
            announcer.stop()


@cluster_app.command("pair")
def pair(
    ttl: str = typer.Option("10m", "--ttl", help = "How long the token works (e.g. 90s, 10m)."),
    addr: List[str] = typer.Option(
        [],
        "--addr",
        help = "Address workers should dial (repeatable); default: this host's LAN addresses.",
    ),
    port: int = typer.Option(None, "--port", help = "Pairing port, open only while this runs."),
):
    """Print a one-time token that adds one worker to this machine's cluster."""
    identity = _link(crypto = True)
    from studio.cluster_link import PAIR_PORT_DEFAULT
    from studio.cluster_link.pairing import PairingWindow

    from studio.cluster_link.pairing import PairingError

    seconds = parse_duration(ttl)
    try:
        master = identity.master_identity(create = True)
    except RuntimeError as exc:
        _fail(str(exc))
    window = PairingWindow(
        master, host = "0.0.0.0", port = port or PAIR_PORT_DEFAULT, ttl = seconds, addrs = addr or None
    )

    async def run() -> Any:
        token = await window.start()
        # The token is never part of a command line: other local users can read process arguments.
        _say("On the worker, run `unsloth cluster join` and paste this token when asked:")
        _say("")
        _say(f"    {token}")
        _say("")
        _say(f"(or pipe it to `unsloth cluster join --token-stdin`). It works once, for {ttl}.")
        try:
            return await window.wait(seconds)
        finally:
            window.close()

    try:
        peer = asyncio.run(run())
    except (asyncio.TimeoutError, TimeoutError):
        _fail("No worker joined before the token expired.")
    except KeyboardInterrupt:
        _fail("Pairing cancelled; the token no longer works.")
    except PairingError as exc:
        _fail(f"Pairing failed: {exc}. Run `unsloth cluster pair` again for a new token.")
    except OSError as exc:
        _fail(f"Could not open the pairing port: {exc}. Pick another with --port.")
    _say(f"Paired {peer.name} ({peer.node_id[:12]}). Start `unsloth worker` on it to serve.")


@cluster_app.command("join")
def join(
    token_stdin: bool = typer.Option(
        False, "--token-stdin", help = "Read the token from stdin instead of a hidden prompt."
    ),
    name: Optional[str] = typer.Option(None, "--name", help = "This machine's name in the cluster."),
    port: int = typer.Option(
        None, "--port", help = "Port `unsloth worker` will serve on here (default 50070)."
    ),
):
    """Pair this machine to a master as a worker."""
    _link(crypto = True)
    from studio.cluster_link.pairing import PairingUnconfirmed, join as join_cluster

    # Only stdin or a hidden prompt: a token in argv is readable by every local user while join runs.
    if token_stdin:
        token = sys.stdin.readline().strip()
    else:
        token = typer.prompt("Token", hide_input = True).strip()
    if not token:
        _fail("No token given.")
    try:
        kwargs = {"agent_port": port} if port else {}
        me = asyncio.run(join_cluster(token, name = name or socket.gethostname(), **kwargs))
    except PairingUnconfirmed as exc:
        _fail(_redact(str(exc), token))
    except Exception as exc:
        # The token must never appear in output: report only the kind of failure.
        _fail(f"Pairing failed: {type(exc).__name__}: {_redact(str(exc), token)}")
    _say(f"Joined as {me.node_id[:12]}. Now run `unsloth worker` here.")


# One announcement round: workers announce every 5 s, jittered up to 1.2x, plus margin.
DISCOVERY_SETTLE_S = 6.5


def _printable(text: str) -> str:
    """Names and ids come from other machines: never let one write control sequences to a terminal."""
    return "".join(c if c.isprintable() else "?" for c in str(text))


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text


@cluster_app.command("list")
def list_workers(
    check: bool = typer.Option(True, "--check/--no-check", help = "Ask each worker for its status."),
):
    """Paired workers, their last address and, when reachable, their status."""
    identity = _link(crypto = True)
    master = identity.master_identity(create = False)
    if not master:
        _fail("This machine is not a master: no worker has been paired to it.")
    registry = _registry(identity)
    workers = [p for p in registry.workers() if not getattr(p, "removed", False)]
    if not workers:
        _say("No workers paired. Add one with `unsloth cluster pair`.")
        return
    if not check:
        for peer in workers:
            where = f"{peer.addrs[0]}:{peer.agent_port}" if peer.addrs else "address unknown"
            _say(_printable(f"{peer.name:<20} {peer.node_id[:12]}  {where}"))
        return
    from studio.cluster_link.relay import RelayManager

    relays = RelayManager(
        master, lambda peer: _registry(identity).authorize(peer, "worker") is not None
    )

    def ask(peer: Any, host: str, port: int) -> Any:
        try:
            relays.ensure(peer.node_id, host, port)
            return relays.status(peer.node_id, timeout = 3.0)
        except Exception as exc:
            return {"type": "error", "error": type(exc).__name__, "reachable": False}

    def answered(st: Any) -> bool:
        # A refused or failed connection comes back as an error dict, not an exception.
        return (
            isinstance(st, dict)
            and st.get("reachable", True) is not False
            and st.get("type") != "error"
        )

    found: dict = {}
    try:
        for peer in workers:
            # Every recorded address, newest first, before discovery (which a network may block).
            for i, addr in enumerate(peer.addrs):
                st = ask(peer, addr, peer.agent_port)
                if i == 0 or answered(st):
                    found[peer.node_id] = (f"{addr}:{peer.agent_port}", st, False)
                if answered(st):
                    if i:
                        with registry.transaction():
                            registry.record_address(peer.node_id, addr, peer.agent_port)
                    break
        missing = [p for p in workers if not answered(found.get(p.node_id, (None, None))[1])]
        if missing:
            # Not where it was last seen (moved, or serving on another port): ask LAN discovery.
            from studio.cluster_link.discovery import Listener

            fps = {p.node_id: p.fingerprint for p in workers}
            _say(f"Looking for {len(missing)} worker(s) on the LAN...")
            listener = Listener(accept = lambda node_id, fp: fps.get(node_id) == fp).start()
            try:
                time.sleep(DISCOVERY_SETTLE_S)
                for peer in missing:
                    for seen in listener.candidates(peer.node_id):
                        if seen.fp != peer.fingerprint:
                            continue
                        st = ask(peer, seen.host, seen.agent_port)
                        if answered(st):
                            found[peer.node_id] = (f"{seen.host}:{seen.agent_port}", st, True)
                            with registry.transaction():
                                registry.record_address(peer.node_id, seen.host, seen.agent_port)
                            break
            finally:
                listener.stop()
        for peer in workers:
            where, st, discovered = found.get(peer.node_id, ("address unknown", None, False))
            line = _printable(f"{peer.name:<20} {peer.node_id[:12]}  {where}")
            if not answered(st):
                why = st.get("error") if isinstance(st, dict) else "not found"
                line += f"  unreachable ({why})"
            else:
                state = "busy" if st.get("busy") else ("ready" if st.get("rpc_up") else "rpc down")
                devices = ", ".join(str(d) for d in st.get("devices") or [])
                line += f"  {state}" + (f"  [{devices}]" if devices else "")
                if discovered:
                    line += "  (found by discovery)"
            _say(line)
    finally:
        relays.stop()


@cluster_app.command("discover")
def discover(seconds: float = typer.Option(6.0, "--seconds", help = "How long to listen.")):
    """Workers announcing themselves on this LAN, paired or not."""
    identity = _link()
    from studio.cluster_link.discovery import Listener

    listener = Listener().start()
    try:
        time.sleep(seconds)
        heard = listener.latest()
    finally:
        listener.stop()
    if listener.error:
        _say(f"Cannot listen for announcements here: {listener.error}")
    paired = {}
    try:
        if identity.master_identity(create = False):
            # Both: a fingerprint is public, so any host can announce one under another node id.
            paired = {(p.node_id, p.fingerprint): p for p in _registry(identity).workers()}
    except Exception:
        pass
    if not heard:
        _say("No worker announced itself. Is `unsloth worker` running on the same network?")
        return
    for node_id, seen in sorted(heard.items()):
        peer = paired.get((node_id, seen.fp))
        tag = f"paired as {peer.name}" if peer is not None and not peer.removed else "not paired"
        _say(_printable(f"{node_id[:12]}  {seen.host}:{seen.agent_port}  {tag}"))


@cluster_app.command("remove")
def remove(target: str = typer.Argument(..., help = "Worker name or id (prefix).")):
    """Revoke a worker: it can no longer serve this master, even if it comes back."""
    identity = _link()
    if not identity.master_identity(create = False):
        _fail("This machine is not a master.")
    registry = _registry(identity)
    peer = _find_worker(registry, target)
    # Mutations persist only inside a transaction (reload, change, save under the lock).
    with registry.transaction():
        removed = registry.remove(peer.node_id)
    if not removed:
        _fail(f"Could not remove {peer.name}.")
    _say(f"Removed {peer.name} ({peer.node_id[:12]}). Pair it again to bring it back.")


@cluster_app.command("leave")
def leave(yes: bool = typer.Option(False, "--yes", "-y", help = "Do not ask for confirmation.")):
    """Forget this machine's cluster identity: a worker leaves its master; a master with no paired
    workers is reset (e.g. after a `cluster pair` nobody joined)."""
    identity = _link()
    if identity.worker_identity() is None:
        if identity.master_identity(create = False) is None:
            _fail("This machine is not part of a cluster.")
        if not yes and not typer.confirm("Reset this machine's cluster master identity?"):
            raise typer.Exit(code = 1)
        # Under the lock a pairing commits with: no worker can be recorded between check and reset.
        with identity.identity_lock():
            live = _registry(identity).workers()
            if live:
                _fail(
                    f"This master still has {len(live)} paired worker(s); `unsloth cluster remove` "
                    "them first, or they lose their master."
                )
            identity.forget_master()
        _say("Master identity removed. This machine can now join another cluster as a worker.")
        return
    if not yes and not typer.confirm("Forget the master and this machine's cluster identity?"):
        raise typer.Exit(code = 1)
    identity.forget_identity()
    _say("Left the cluster. Pair again with `unsloth cluster join`.")
