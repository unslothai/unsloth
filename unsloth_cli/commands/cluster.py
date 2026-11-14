# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth worker` and `unsloth cluster`: ggml-rpc-server never listens beyond loopback."""

from __future__ import annotations

import errno
import signal
import re
import socket
import sys
import time
from typing import Any, List, Optional, Tuple

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
        ensure_studio_backend_path(seed_cache_env = False)
    except Exception:
        pass
    try:
        from studio.cluster_link import identity
    except ImportError as exc:
        _fail(f"This install has no cluster support ({exc}). Reinstall Unsloth Studio.")
    if crypto:
        try:
            import cryptography  # noqa: F401
        except ImportError:
            _fail(
                "Cluster pairing needs the `cryptography` package, which comes with Unsloth Studio. "
                "Install Studio (or `pip install cryptography`) and run this again."
            )
    return identity


def _registry(identity: Any) -> Any:
    return identity.Registry().load()


def _addresses(peer: Any) -> List[Tuple[str, int]]:
    from studio.cluster_link.status import addresses
    return addresses(peer)


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
    port: int = typer.Option(
        None, "--port", min = 1, max = 65535, help = "Port the agent listens on (all interfaces)."
    ),
    rpc_port: int = typer.Option(
        None, "--rpc-port", min = 1, max = 65535, help = "Loopback port for ggml-rpc-server."
    ),
    native_port: int = typer.Option(
        None,
        "--native-port",
        min = 1,
        max = 65535,
        help = "LAN port for llama.cpp's own TLS transport, used when both ends' llama.cpp has it.",
    ),
    no_native_tls: bool = typer.Option(
        False,
        "--no-native-tls",
        help = "Always serve through this agent's relay, even when llama.cpp could speak TLS itself.",
    ),
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
    join_first: bool = typer.Option(
        False, "--join", help = "Pair with a master first (asks for its code), then serve it."
    ),
    name: Optional[str] = typer.Option(None, "--name", help = "This machine's name, with --join."),
):
    """Serve this machine's GPUs to the master it was paired with."""
    import asyncio

    identity = _link(crypto = True)
    from studio.cluster_link import AGENT_PORT_DEFAULT, NATIVE_RPC_PORT_DEFAULT, RPC_PORT_DEFAULT

    if join_first:
        _join_with_code(_read_code(False), name, port)
    me = identity.worker_identity()
    if me is None:
        _fail(
            "This machine is not paired yet. On the master run `unsloth cluster pair`, then run "
            "`unsloth worker --join` here and paste the code it prints."
        )
    binary = _rpc_server_binary()
    if not binary:
        _fail(
            "ggml-rpc-server was not found. `unsloth worker` uses Unsloth Studio's llama.cpp: "
            "install Studio on this machine first."
        )
    from studio.cluster_link import remote
    from studio.cluster_link.agent import Agent

    if remote.key_path("server").is_file():
        remote.ensure_helper(_say)

    def authorize(peer: Any) -> bool:
        try:
            return _registry(identity).authorize(peer, "master") is not None
        except Exception:
            return False

    def tunnel_allow() -> List[str]:
        master = _registry(identity).master()
        return [master.tc_pub] if master is not None and master.tc_pub else []

    paired_port = identity.worker_agent_port() or AGENT_PORT_DEFAULT
    if port and port != paired_port:
        _say(
            f"Note: this worker paired on port {paired_port}; the master finds port {port} only through "
            "LAN discovery. Re-pair with `unsloth cluster join --port` if discovery is blocked here."
        )
    port = port or paired_port
    rpc_port = rpc_port or RPC_PORT_DEFAULT
    native_port = native_port or NATIVE_RPC_PORT_DEFAULT
    if len({port, rpc_port, native_port}) < 3:
        _fail(
            f"--port ({port}), --rpc-port ({rpc_port}) and --native-port ({native_port}) must all differ."
        )
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
        native_port = native_port,
        native_tls = False if no_native_tls else None,
        tunnel_allow = tunnel_allow,
    )
    announcer = None
    if not no_discovery:
        from studio.cluster_link.discovery import Announcer
        announcer = Announcer(me.node_id, port, me.fingerprint).start()

    async def serve() -> None:
        task = asyncio.current_task()
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
            if sig is not None and task is not None:
                try:
                    asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
                except (NotImplementedError, RuntimeError):
                    pass
        try:
            await agent.start()
            if await agent.wait_rpc_ready(60.0):
                _say(
                    f"Worker {me.node_id[:12]} serving on port {port}; ggml-rpc-server stays on "
                    f"127.0.0.1:{rpc_port}. Ctrl-C to stop."
                )
                if getattr(agent, "native_available", False):
                    _say(
                        f"  llama.cpp TLS: a master with it is served directly on port {agent.native_port}."
                    )
                if getattr(agent, "tunnel", None) is not None:
                    _say("  Also reachable by its master from other networks, through tailcat.")
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
        _say("Worker stopped.")
    except (KeyboardInterrupt, asyncio.CancelledError):
        _say("Worker stopped.")
    except RuntimeError as exc:
        _fail(_sentence(str(exc)))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            _fail(
                f"Could not listen on port {port}: it is already in use (is another `unsloth worker` "
                "running?). Stop that, or pick another port with --port."
            )
        if exc.errno in (errno.EACCES, errno.EADDRNOTAVAIL):
            _fail(f"Could not listen on port {port}: {exc}. Pick another with --port.")
        _fail(f"Worker stopped: {exc}")
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
    port: int = typer.Option(
        None, "--port", min = 1, max = 65535, help = "Pairing port, open only while this runs."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help = "Approve the worker that uses the code without asking."
    ),
):
    """Print a one-time code that adds one worker to this machine's cluster."""
    import asyncio

    identity = _link(crypto = True)
    from studio.cluster_link import PAIR_PORT_DEFAULT, remote
    from studio.cluster_link.pairing import PairingError, PairingWindow

    remote.ensure_helper(_say)

    seconds = parse_duration(ttl)
    try:
        master = identity.master_identity(create = True)
    except RuntimeError as exc:
        _fail(str(exc))

    def ask(name: str, words: Optional[List[str]]) -> bool:
        seen = " ".join(words or [])
        return typer.confirm(_printable(f"{name} wants to join ({seen}). Allow?"), default = True)

    window = PairingWindow(
        master,
        host = "0.0.0.0",
        port = port or PAIR_PORT_DEFAULT,
        ttl = seconds,
        addrs = addr or None,
        approve = None if yes else (lambda name, words: asyncio.to_thread(ask, name, words)),
    )

    async def run() -> Any:
        token = await window.start()
        # The code is never part of a command line: other local users can read process arguments.
        _say("On the other machine run `unsloth worker --join` and paste this code when asked:")
        _say("")
        _say(f"    {token}")
        _say("")
        where = "on this network or another one" if window.tc else "on this network"
        _say(f"It works once, for {ttl}, {where}. Keep it private: it lets a machine join.")
        _say("Waiting for the worker to join (Ctrl-C to cancel)...")
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
        _fail(
            f"Pairing failed: {_clause(str(exc))}. Run `unsloth cluster pair` again for a new token."
        )
    except OSError as exc:
        _fail(f"Could not open the pairing port: {exc}. Pick another with --port.")
    _say(
        _printable(
            f"Paired {peer.name} ({peer.node_id[:12]}). `unsloth cluster list` shows whether it is "
            "ready (it serves while `unsloth worker` runs there)."
        )
    )


@cluster_app.command("join")
def join(
    token_stdin: bool = typer.Option(
        False, "--token-stdin", help = "Read the token from stdin instead of a hidden prompt."
    ),
    name: Optional[str] = typer.Option(None, "--name", help = "This machine's name in the cluster."),
    port: int = typer.Option(
        None,
        "--port",
        min = 1,
        max = 65535,
        help = "Port `unsloth worker` will serve on here (default 50070).",
    ),
):
    """Pair this machine to a master as a worker (`unsloth worker --join` also starts serving)."""
    _link(crypto = True)
    me = _join_with_code(_read_code(token_stdin), name, port)
    _say(f"Joined as {me.node_id[:12]}. Now run `unsloth worker` here to serve the master.")


def _read_code(from_stdin: bool) -> str:
    # Only stdin or a hidden prompt: a code in argv is readable by every local user while join runs.
    token = (
        sys.stdin.readline().strip()
        if from_stdin
        else typer.prompt("Code", hide_input = True).strip()
    )
    if not token:
        _fail("No code given.")
    return token


def _join_with_code(token: str, name: Optional[str], port: Optional[int]) -> Any:
    import asyncio

    _link(crypto = True)
    from studio.cluster_link import remote
    from studio.cluster_link.pairing import PairingError, PairingUnconfirmed, join as join_cluster
    from studio.cluster_link.tokens import PREFIX_V2, TokenError

    if token.strip().startswith(PREFIX_V2):
        remote.ensure_helper(_say)

    def show(words: List[str]) -> None:
        _say(f"Check that the master shows these words, then allow it there: {' '.join(words)}")

    try:
        kwargs = {"agent_port": port} if port else {}
        return asyncio.run(
            join_cluster(token, name = name or socket.gethostname(), on_words = show, **kwargs)
        )
    except PairingUnconfirmed as exc:
        _fail(_redact(str(exc), token))
    except TokenError as exc:
        _fail(
            f"That is not a usable pairing code ({_redact(str(exc), token)}). Copy the whole code "
            "`unsloth cluster pair` printed on the master; it starts with `unsloth-pair-`."
        )
    except PairingError as exc:
        message = _clause(_redact(str(exc), token))
        if message.startswith("could not reach the master"):
            message += (
                ". Is `unsloth cluster pair` still running there, and can this machine reach it "
                "on the LAN?"
            )
        _fail(f"Pairing failed: {message}.")
    except Exception as exc:
        # The code must never appear in output: report only the kind of failure.
        _fail(f"Pairing failed: {type(exc).__name__}: {_redact(str(exc), token)}")


DISCOVERY_SETTLE_S = 6.5


def _printable(text: str) -> str:
    return "".join(c if c.isprintable() else "?" for c in str(text))


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text


def _clause(text: str) -> str:
    return str(text).strip().rstrip(".")


def _sentence(text: str) -> str:
    text = str(text).strip()
    return text[:1].upper() + text[1:] if text else text


@cluster_app.command("list")
def list_workers(
    check: bool = typer.Option(True, "--check/--no-check", help = "Ask each worker for its status."),
):
    """Paired workers, their last address and, when reachable, their status."""
    identity = _link(crypto = True)
    master = identity.master_identity(create = False)
    if not master:
        me = identity.worker_identity()
        if me is not None:
            _say(
                f"This machine is a worker ({me.node_id[:12]}); run `unsloth cluster list` on its "
                "master to see the cluster."
            )
            return
        _say("No workers paired. Add one with `unsloth cluster pair`.")
        return
    registry = _registry(identity)
    workers = [p for p in registry.workers() if not getattr(p, "removed", False)]
    if not workers:
        _say("No workers paired. Add one with `unsloth cluster pair`.")
        return
    if not check:
        for peer in workers:
            pairs = _addresses(peer)
            where = f"{pairs[0][0]}:{pairs[0][1]}" if pairs else "address unknown"
            _say(_printable(f"{peer.name:<20} {peer.node_id[:12]}  {where}"))
        return
    from studio.cluster_link import status

    try:
        rows = status.check_workers(
            master,
            registry,
            workers,
            authorize = lambda peer: _registry(identity).authorize(peer, "worker") is not None,
            settle_s = DISCOVERY_SETTLE_S,
            on_search = lambda n: _say(f"Looking for {n} worker(s) on the LAN..."),
        )
    finally:
        try:
            from studio.cluster_link import remote
            remote.close()
        except Exception:
            pass
    for row in rows:
        line = _printable(f"{row.name:<20} {row.node_id[:12]}  {row.where}")
        if row.state == "unreachable":
            line += _printable(f"  unreachable ({row.reason})")
        elif row.state == "relayed":
            line += "  relayed only: too slow to serve models"
        else:
            state = f"ready ({row.via})" if row.state == "ready" else row.state
            devices = ", ".join(row.devices)
            line += f"  {state}" + (f"  [{devices}]" if devices else "")
            if row.discovered:
                line += "  (found by discovery)"
        _say(line)


@cluster_app.command("discover")
def discover(seconds: float = typer.Option(6.0, "--seconds", min = 0, help = "How long to listen.")):
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
        _fail("No workers are paired to this machine, so there is nothing to remove.")
    registry = _registry(identity)
    peer = _find_worker(registry, target)
    with registry.transaction():
        removed = registry.remove(peer.node_id)
    if not removed:
        _fail(f"Could not remove {peer.name}.")
    _say(f"Removed {peer.name} ({peer.node_id[:12]}). Pair it again to bring it back.")


@cluster_app.command("leave")
def leave(yes: bool = typer.Option(False, "--yes", "-y", help = "Do not ask for confirmation.")):
    """Forget this machine's cluster identity (a worker leaves; an empty master is reset)."""
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
            _forget_tailnet_keys()
        _say("Master identity removed. This machine can now join another cluster as a worker.")
        return
    if not yes and not typer.confirm("Forget the master and this machine's cluster identity?"):
        raise typer.Exit(code = 1)
    with identity.identity_lock():
        identity.forget_identity()
        _forget_tailnet_keys()
    _say("Left the cluster. Pair again with `unsloth worker --join`.")


def _forget_tailnet_keys() -> None:
    """Called under identity_lock, the lock a pairing makes its keys under."""
    try:
        from studio.cluster_link import remote
        remote.forget_keys(held = True)
    except Exception:
        pass
