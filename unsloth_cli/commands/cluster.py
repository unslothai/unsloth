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

    port = port or AGENT_PORT_DEFAULT
    rpc_port = rpc_port or RPC_PORT_DEFAULT
    rpc_args: List[str] = []
    names = [n.strip() for value in device for n in value.split(",") if n.strip()]
    if names:
        # ggml-rpc-server takes one comma-separated --device list.
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
    _say(
        f"Worker {me.node_id[:12]} on port {port}; ggml-rpc-server stays on 127.0.0.1:{rpc_port}. "
        "Ctrl-C to stop."
    )

    async def serve() -> None:
        # SIGTERM / SIGHUP end the run like Ctrl-C, so stop() reaps rpc-server instead of orphaning it.
        task = asyncio.current_task()
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
            if sig is not None and task is not None:
                try:
                    asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
                except (NotImplementedError, RuntimeError):
                    pass
        await agent.start()
        try:
            await agent.serve_forever()
        finally:
            await agent.stop()

    try:
        asyncio.run(serve())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
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
        _say("On the worker, run:")
        _say("")
        _say(f"    unsloth cluster join {token}")
        _say("")
        _say(f"(or `unsloth cluster join --token-stdin`). The token works once, for {ttl}.")
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
    token: Optional[str] = typer.Argument(None, help = "The token `unsloth cluster pair` printed."),
    token_stdin: bool = typer.Option(
        False, "--token-stdin", help = "Read the token from stdin (keeps it out of shell history)."
    ),
    name: Optional[str] = typer.Option(None, "--name", help = "This machine's name in the cluster."),
):
    """Pair this machine to a master as a worker."""
    _link(crypto = True)
    from studio.cluster_link.pairing import join as join_cluster

    if token_stdin:
        token = sys.stdin.readline().strip()
    elif not token:
        token = typer.prompt("Token", hide_input = True).strip()
    if not token:
        _fail("No token given.")
    try:
        me = asyncio.run(join_cluster(token, name = name or socket.gethostname()))
    except Exception as exc:
        # The token must never appear in output: report only the kind of failure.
        _fail(f"Pairing failed: {type(exc).__name__}: {_redact(str(exc), token)}")
    _say(f"Joined as {me.node_id[:12]}. Now run `unsloth worker` here.")


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
    relays = None
    if check:
        from studio.cluster_link.relay import RelayManager
        relays = RelayManager(
            master, lambda peer: _registry(identity).authorize(peer, "worker") is not None
        )
    try:
        for peer in workers:
            where = f"{peer.addrs[0]}:{peer.agent_port}" if peer.addrs else "address unknown"
            line = f"{peer.name:<20} {peer.node_id[:12]}  {where}"
            if relays is not None and peer.addrs:
                try:
                    relays.ensure(peer.node_id, peer.addrs[0], peer.agent_port)
                    st = relays.status(peer.node_id, timeout = 3.0)
                    state = (
                        "busy" if st.get("busy") else ("ready" if st.get("rpc_up") else "rpc down")
                    )
                    devices = ", ".join(str(d) for d in st.get("devices") or [])
                    line += f"  {state}" + (f"  [{devices}]" if devices else "")
                except Exception as exc:
                    line += f"  unreachable ({type(exc).__name__})"
            _say(line)
    finally:
        if relays is not None:
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
            paired = {p.fingerprint: p for p in _registry(identity).workers()}
    except Exception:
        pass
    if not heard:
        _say("No worker announced itself. Is `unsloth worker` running on the same network?")
        return
    for node_id, seen in sorted(heard.items()):
        peer = paired.get(seen.fp)
        tag = f"paired as {peer.name}" if peer is not None and not peer.removed else "not paired"
        _say(f"{node_id[:12]}  {seen.host}:{seen.agent_port}  {tag}")


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
    """On a worker: forget the master and this machine's cluster identity."""
    identity = _link()
    if identity.worker_identity() is None:
        _fail("This machine is not paired as a worker.")
    if not yes and not typer.confirm("Forget the master and this machine's cluster identity?"):
        raise typer.Exit(code = 1)
    identity.forget_identity()
    _say("Left the cluster. Pair again with `unsloth cluster join`.")
