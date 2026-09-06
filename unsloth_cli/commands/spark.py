# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth spark` -- own one DGX Spark, or several, without knowing any of this.

The entry point is `unsloth spark up`. It looks at the machine, works out which of
the several possible situations this is (one Spark? a cable but no pairing? a paired
peer that is powered off? a degraded link? missing kernels?) and either does the next
thing or names the ONE command that does. Everything else in this group -- status,
setup, env, serve, train, provision, plan, kernels, estimate, doctor -- stays exactly
as it was, for people who already know which one they want.

Two facts drive most of the output, and both are counter-intuitive enough that the
CLI states them rather than assuming anyone will infer them:

  * The biggest measured win on this hardware needs NO second machine. Choosing the
    NVFP4 kernel by workload is 6.2x on prefill (CUTLASS 309 TF/s vs Marlin 50 TF/s
    at M=4096). Most Spark owners have exactly one Spark, so this is surfaced in
    `status`, in `doctor`, and in `up` whenever the kernel is not installed.
  * A second Spark does not make a model that already fits decode faster. Layer-
    splitting such a model measures 0.85x to 1.01x across 1 to 32 users -- never a
    win; a split is for capacity and for prefill. What a second Spark does buy for a
    model that fits is throughput at load: two replicas measured 1.30x at 8 users,
    1.75x at 16 and 1.91x at 32, and 1.00x to 1.13x below 8. So the planner recommends
    replicas from 8 users up, one Spark below that, and a split only for a model that
    does not fit or for prefill-heavy long-prompt work.

Nothing here imports torch, transformers, vllm or numpy at module scope, and nothing
touches the network unless the machine is a DGX Spark with a cabled peer. On a Mac,
a Windows laptop, an AMD box or any x86 Linux machine, every command prints one line
and exits 0.
"""

from __future__ import annotations

import typer

spark_app = typer.Typer(
    help = "Set up and use one or more NVIDIA DGX Sparks. Start with `unsloth spark up`.",
    no_args_is_help = False,
)


# ── Measured numbers ─────────────────────────────────────────────────────────
# The authoritative copies live in studio.spark_cluster (TP_SPEEDUP_2, PP_SPEEDUP_2,
# TP_TPOT_MS_2, LAYER_SPLIT_FITTING_SPEEDUP...) and every planner number printed by
# `plan` comes from there via expected_gain(), so nothing in this file can drift out
# of step with the planner. What is kept here is only what the module does not model:
# the kernel result, which is a single-node fact and not a cluster topology, and the
# prefill/TTFT split, which is what someone deciding whether to BUY a second Spark
# actually needs to know.
PREFILL_KERNEL_SPEEDUP = 6.2  # CUTLASS 309 TF/s vs Marlin 50 TF/s at M=4096

# Llama-3.3-70B fp8, two Sparks (tensor parallel) versus one. Printed where the
# question is "is a second Spark worth it", because the answer is very unevenly
# distributed across workloads and the average of these two numbers helps nobody.
PREFILL_TOKS = (166, 643)  # tok/s, 3.87x
TTFT_MS = (3085, 797)  # median TTFT, 3.87x
TPOT_ONE_SPARK_MS = 332.7
TPOT_TP2_MS = 162.4  # decode, 2.09x

# A "128 GB" Spark is 128 GiB of which about 6.3 GiB is firmware-reserved. Users read
# 128 on the box, size a model against it, and get an OOM they cannot explain, so the
# real number is printed wherever memory is discussed.
MARKETED_GIB = 128.0
USABLE_GIB_FALLBACK = 121.69

# The kernel that the 6.2x depends on, and the version pin it needs. nvidia-cutlass-dsl
# 4.7.0 fails the b12x family with an internal DSL compiler error, which disables the
# kernel family built for this GPU -- so a wrong version is worth flagging separately
# from a missing one.
CUTLASS_PIN = "4.6.2"
# flashinfer-jit-cache is the half people miss, and it is the expensive half to miss.
# Without it FlashInfer JIT-compiles its kernels on first use: ~430 s of cold start, and on a
# Spark the compile itself OOMs (cicc at 7-9 GiB across 20 cores -> `ninja: exit 137`) unless
# MAX_JOBS is throttled. The prebuilt wheel removes both. It is NOT on PyPI -- that 404s --
# so it needs FlashInfer's own index, which carries a cu130 aarch64 build matching this
# hardware exactly (verified resolvable: flashinfer-jit-cache 0.6.18+cu130).
#
# NEVER add flashinfer-cubin: it is ~6.8 GB of prebuilt cubins this GPU cannot load.
# nvidia-cutlass-dsl is pinned at 4.6.2 because 4.7.0 breaks b12x on sm_121 with a DSL
# compiler ICE; 4.7.1 exists but is unverified on GB10.
KERNEL_INSTALL = (
    'pip install flashinfer-python "nvidia-cutlass-dsl==4.6.2" && '
    "pip install flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu130/"
)


# ── Small output helpers (plain ASCII, no colour, no unicode width) ──────────


def _say(line: str = "") -> None:
    typer.echo(line)


def _heading(text: str) -> None:
    _say("")
    _say(text)
    _say("-" * len(text))


def _field(name: str, value: str) -> None:
    _say(f"  {name:<14} {value}")


# ── Defensive access to studio.spark_cluster ─────────────────────────────────
# Another agent is generalising that module from "two Sparks" to N. Everything below
# reads it through getattr and signature inspection, so this file works against both
# the current and the generalised API, and never imports a symbol that may not exist.


def _cluster():
    from studio import spark_cluster
    return spark_cluster


def _cluster_or_none():
    """The module, or None with a message printed. Never raises."""
    try:
        return _cluster()
    except Exception as exc:  # pragma: no cover - import guard
        _say(f"Could not load the DGX Spark support module: {exc}")
        return None


def _on_spark(sc) -> bool:
    try:
        return bool(sc.is_dgx_spark())
    except Exception:
        return False


def _getnum(sc, name: str, default: float) -> float:
    value = getattr(sc, name, None)
    return float(value) if isinstance(value, (int, float)) else default


def _usable_gib(sc) -> float:
    return _getnum(sc, "SPARK_USABLE_GIB", USABLE_GIB_FALLBACK)


def _serve_budget(sc) -> float:
    return _usable_gib(sc) - _getnum(sc, "SERVE_OVERHEAD_GIB", 8.0)


def _max_nodes(sc) -> int:
    """The largest cluster this build will PLAN ADDRESSING for.

    Not the same question as how many Sparks can be seen. Two Sparks are cabled
    QSFP-to-QSFP and need no switch; three or more cannot be, so the flat one-/24-
    per-rail plan is only correct on a switched fabric and the module refuses to emit
    it otherwise. Read from the module rather than hardcoded here, so a change there
    cannot leave this file quietly lying.
    """
    for name in (
        "MAX_PLANNABLE_NODES",
        "MAX_CLUSTER_NODES",
        "MAX_NODES",
        "SUPPORTED_NODES",
        "CLUSTER_NODES",
    ):
        value = getattr(sc, name, None)
        if isinstance(value, int) and value >= 2:
            return value
    return 2


def _plan_deployment(
    sc,
    size_gib,
    nodes: int,
    intent: str = "throughput",
    concurrency: int = 1,
    model: str = "<model>",
    prompt_tokens: int = 512,
    prefill_heavy: bool = False,
):
    """Call plan_deployment against whichever signature the module currently has.

    The N-node signature takes intent and concurrency; the older one took a bare
    ``two_sparks`` bool. Both are supported, because a CLI that raises TypeError after
    an upgrade is a worse outcome than one that gives a slightly plainer answer.
    """
    fn = getattr(sc, "plan_deployment", None)
    if fn is None:
        return None
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        params = {}
    try:
        if "n_nodes" in params:
            kwargs = {"n_nodes": nodes}
            if "intent" in params:
                kwargs["intent"] = intent
            if "concurrency" in params:
                kwargs["concurrency"] = concurrency
            if "model" in params:
                kwargs["model"] = model
            if "prompt_tokens" in params:
                kwargs["prompt_tokens"] = prompt_tokens
            if "prefill_heavy" in params:
                kwargs["prefill_heavy"] = prefill_heavy
            return fn(size_gib, **kwargs)
        if "nodes" in params:
            return fn(size_gib, nodes = nodes)
        if "world" in params:
            return fn(size_gib, world = nodes)
        return fn(size_gib, two_sparks = nodes >= 2)
    except Exception:
        return None


def _require_spark(sc, what: str) -> None:
    """Exit cleanly off a DGX Spark, before delegating anything that assumes one.

    `serve` and `train` in studio.spark_cluster go straight to rail discovery without a
    detection gate -- harmless on a laptop, where the sysfs walk finds nothing, but this
    layer should not depend on that. A clear sentence and exit 0 is the contract for
    every command in this group on every other machine.
    """
    if not _on_spark(sc):
        _say(f"Not a DGX Spark; {what}")
        raise typer.Exit(0)


# ── Kernel readiness: the 6.2x that needs no second machine ──────────────────


def _kernel_state() -> dict:
    """Is the prefill kernel installed in THIS interpreter?

    Uses importlib.metadata rather than importing anything: asking whether flashinfer
    is installed must not cost what importing it costs, and this runs on the `status`
    path that people run casually.
    """
    from importlib.metadata import version, PackageNotFoundError

    def _v(name: str):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
        except Exception:
            return None

    flashinfer = _v("flashinfer-python") or _v("flashinfer")
    cutlass = _v("nvidia-cutlass-dsl")
    # The jit cache is a SEPARATE package and is the one people miss. flashinfer alone works
    # but compiles its kernels on first use: ~430 s of cold start, and the compile itself OOMs
    # on a Spark unless MAX_JOBS is throttled. Reporting "installed" while it is absent would
    # hide the slowest part of a first run behind a green tick.
    jit_cache = _v("flashinfer-jit-cache")
    state = {
        "flashinfer": flashinfer,
        "cutlass": cutlass,
        "jit_cache": jit_cache,
        "vllm": _v("vllm"),
        "pin_wrong": bool(cutlass) and cutlass != CUTLASS_PIN,
        # Only meaningful once flashinfer itself is present.
        "jit_cache_missing": bool(flashinfer) and not jit_cache,
    }
    state["ok"] = bool(flashinfer) and not state["pin_wrong"] and bool(jit_cache)
    return state


def _kernel_banner(state: dict | None = None) -> bool:
    """Print the 6.2x prefill message if the kernel is missing. True if printed."""
    state = state if state is not None else _kernel_state()
    if state["ok"]:
        return False
    _say("")
    _say("=" * 72)
    if not state["flashinfer"]:
        _say(f"  The biggest measured win on this machine is NOT installed:")
        _say(f"  {PREFILL_KERNEL_SPEEDUP}x faster prefill on NVFP4 models, on ONE Spark,")
        _say("  no second machine involved.")
        _say("")
        _say("    measured, same weights, same shape, kernel varied, M=4096:")
        _say("      flashinfer_cutlass   4727 us   309 TF/s")
        _say("      marlin              29257 us    50 TF/s   <- what you get by default")
        _say("")
        _say("  Install it:")
        _say(f"    {KERNEL_INSTALL}")
        _say("  Then serve with:")
        _say("    --linear-backend flashinfer_cutlass")
    elif state.get("jit_cache_missing"):
        # flashinfer is here but its prebuilt kernels are not, so the kernels still work --
        # they are just compiled on demand. A distinct message, because "install flashinfer"
        # is unhelpful advice to someone who already has it.
        _say("  flashinfer is installed but flashinfer-jit-cache is NOT.")
        _say("  The kernels still work; they are compiled on FIRST USE instead:")
        _say("    ~430 s of cold start, and on a Spark the compile itself OOMs")
        _say(
            "    (cicc at 7-9 GiB across 20 cores -> `ninja: exit 137`) unless MAX_JOBS is capped."
        )
        _say("")
        _say("  A prebuilt aarch64 CUDA 13 wheel exists (it is NOT on PyPI):")
        _say(
            "    pip install flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu130/"
        )
        _say("  Do NOT install flashinfer-cubin -- ~6.8 GB of cubins this GPU cannot load.")
    else:
        _say(f"  nvidia-cutlass-dsl is {state['cutlass']}, not the required {CUTLASS_PIN}.")
        _say("  4.7.0 fails the b12x kernel family with an internal DSL compiler error,")
        _say("  which disables the kernel family built for this GPU. Repin it:")
        _say(f'    pip install "nvidia-cutlass-dsl=={CUTLASS_PIN}"')
    _say("")
    _say("  Full numbers and when each kernel wins:  unsloth spark kernels")
    _say("=" * 72)
    return True


# ── Situation detection ──────────────────────────────────────────────────────


def _peer_reachable(
    host: str,
    port: int = 22,
    timeout: float = 3.0,
) -> bool | None:
    """Can we open a TCP connection to the peer? None when we cannot tell.

    A bounded connect, never a ping and never an ssh: this must not hang, must not
    prompt for a password, and must not depend on a tool Windows does not ship. It is
    only ever called on a DGX Spark that has a configured rail, so no other platform
    makes a network call at all.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout = timeout):
            return True
    except OSError:
        return False
    except Exception:  # pragma: no cover - defensive
        return None


def _peer_has_venv(host: str, timeout: float = 12.0) -> bool | None:
    """Whether the peer already has the Unsloth venv. None when we cannot tell.

    Worth one bounded ssh, because a peer without it does not fail loudly: the head
    rank blocks at the rendezvous for 601 seconds and then reports
    `DistStoreError: 1/2 clients joined`, which names neither the peer nor the venv.
    """
    import os
    import shutil
    import subprocess

    if not shutil.which("ssh"):  # Windows, or a stripped image
        return None
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "nvidia"
    # Resolved, not assumed: UNSLOTH_STUDIO_HOME moves the venv, and testing the default
    # path then reports the peer as unprovisioned when it is fine, or fine when it is not.
    try:
        from studio.spark_cluster import venv_activate
        _venv = venv_activate().rsplit("/bin/activate", 1)[0]
    except Exception:
        _venv = "$HOME/.unsloth/studio/unsloth_studio"
    remote = f'test -d "{_venv}" && echo YES || echo NO'
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=6",
                f"{user}@{host}",
                remote,
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception:
        return None
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return None
    if out[-1] == "YES":
        return True
    if out[-1] == "NO":
        return False
    return None


def _situation(
    sc,
    discover_timeout: float = 3.0,
    probe: bool = True,
) -> dict:
    """Everything `up` needs to decide, gathered once and cheaply."""
    info = {
        "state": "not_spark",
        "cable": False,
        "peer_ip": None,
        "local_ip": None,
        "rails": [],
        "peers": [],
        "nodes": 1,
        "seen": 1,
        "max_nodes": _max_nodes(sc),
    }
    if not _on_spark(sc):
        return info
    try:
        info["state"] = sc.cluster_state()
    except Exception:
        info["state"] = "unconfigured"
    try:
        rails = sc.cabled_rails()
    except Exception:
        rails = []
    info["rails"] = rails
    info["cable"] = bool(rails)
    try:
        info["peer_ip"] = sc.peer_ip_for(rails)
    except Exception:
        info["peer_ip"] = None
    for rail in rails:
        for addr in rail.get("ipv4", []):
            info["local_ip"] = addr
            break
        if info["local_ip"]:
            break

    # Discovery is N-aware where the module is; a build without it simply reports the
    # pair, which is still correct for the pair.
    discovered = {}
    try:
        import inspect

        params = inspect.signature(sc.discover_peers).parameters
        kwargs = {"timeout": discover_timeout}
        if "check_reachable" in params and probe:
            kwargs["check_reachable"] = True
        discovered = sc.discover_peers(**kwargs) or {}
    except Exception:
        discovered = {}
    info["peers"] = list(discovered.get("peers") or discovered.get("mdns_peers") or [])
    # How many Sparks are visible at all, versus how many are actually paired into a
    # cluster. These are different numbers and conflating them is how someone ends up
    # planning for three nodes that share no fabric.
    info["seen"] = int(discovered.get("n_nodes") or (len(info["peers"]) + 1))
    configured = info["state"] == "configured" and bool(info["peer_ip"])
    info["nodes"] = max(2, min(info["seen"], info["max_nodes"])) if configured else 1
    return info


def _report_inventory(sc, sit: dict) -> None:
    _field("machine", "NVIDIA DGX Spark")
    _field(
        "memory",
        f"{_usable_gib(sc):.2f} GiB usable per Spark "
        f"(the '{MARKETED_GIB:.0f} GB' figure is {MARKETED_GIB:.0f} GiB "
        f"with ~{MARKETED_GIB - _usable_gib(sc):.1f} GiB firmware-reserved)",
    )
    if sit["rails"]:
        for rail in sit["rails"]:
            ips = ", ".join(rail.get("ipv4") or []) or "no IPv4 yet"
            _field(
                "rail",
                f"{rail.get('ib_device','?')} / {rail.get('netdev','?')} "
                f"mtu={rail.get('mtu','?')}  {ips}",
            )
    else:
        _field("rail", "no QSFP cable detected")
    for peer in sit["peers"]:
        name = peer.get("short") or peer.get("hostname") or "?"
        addr = peer.get("address", "?")
        reach = {True: "reachable", False: "UNREACHABLE", None: "not probed"}.get(
            peer.get("reachable"), "not probed"
        )
        _field("peer seen", f"{name} at {addr} -- {reach}")
    if sit.get("peer_ip"):
        # Deliberately separate from the discovered address. mDNS answers over whatever
        # interface advertised, which is usually Wi-Fi; using that address for NCCL or
        # rsync would quietly bypass the 200GbE link the pairing exists to provide.
        _field("fast link", f"{sit['peer_ip']} -- use THIS for NCCL, ray and rsync")
    _field(
        "cluster",
        f"{sit['nodes']} Sparks paired into one cluster"
        if sit["nodes"] > 1
        else "1 Spark, not paired with another",
    )
    if sit["seen"] > sit["nodes"]:
        _say("")
        _say(
            f"  NOTE: {sit['seen']} Sparks are visible but only {sit['nodes']} "
            f"{'is' if sit['nodes'] == 1 else 'are'} paired."
        )
        if sit["seen"] > 2:
            _say("  Three or more Sparks cannot be cabled QSFP-to-QSFP the way two are.")
            _say("  If they all sit on a switched RoCE fabric, pair them explicitly:")
            _say(
                f"    unsloth spark setup --nodes {min(sit['seen'], sit['max_nodes'])} "
                f"--switched"
            )
            _say("  If they are daisy-chained instead, each cable needs its own subnet and")
            _say("  Unsloth will not guess your cabling -- a wrong netplan is worse than")
            _say("  none, so it refuses rather than emitting one.")
        _say("  Unpaired Sparks are still useful: run one model each behind")
        _say("  `python -m studio.spark_lb`, which fans out to as many backends as given.")


# ── The guided entry point ───────────────────────────────────────────────────


def _next_steps(sit: dict) -> None:
    _say("")
    _say("What to do next:")
    _say("  unsloth spark plan --model <model> --for latency|throughput|capacity")
    _say("  unsloth spark peers                   every Spark this node can see")
    if sit["nodes"] >= 2:
        _say("  unsloth spark doctor                  measure the link (catches both faults)")
        _say("  unsloth spark train --layer-split <model>   train something too big for one")
    _say("  unsloth spark kernels --workload prefill   the 6.2x kernel choice")


@spark_app.command("up")
def up(
    yes: bool = typer.Option(
        False, "--yes", "-y", help = "Do not ask; perform the fixes this finds."
    ),
    check: bool = typer.Option(False, "--check", help = "Report only. Change nothing, ever."),
) -> None:
    """Get this machine to a working state, whatever state it is in now.

    This is the one command to run. It works out whether you have one Spark, two, a
    cable that was never paired, a peer that is powered off, a degraded link or a
    missing kernel, and then either fixes it or prints the single next command. Every
    other subcommand in this group still exists and still does exactly what it did.
    """
    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)

    if not _on_spark(sc):
        _say("This machine is not an NVIDIA DGX Spark, so there is nothing to set up.")
        _say("Unsloth itself works here as normal -- try `unsloth train --help`.")
        raise typer.Exit(0)

    _heading("Unsloth on DGX Spark")
    sit = _situation(sc)
    _report_inventory(sc, sit)

    kernels = _kernel_state()
    state = sit["state"]

    # ── One Spark, no cable: the common case, and the one with the biggest win ──
    if not sit["cable"]:
        _heading("Status")
        _say("  One Spark, no second one cabled. Nothing to pair, and nothing is wrong.")
        if sit["peers"]:
            _say("")
            _say("  Another Spark is visible on the network but is NOT cabled to this one.")
            _say("  A second Spark only helps over its QSFP link; over Ethernet it does not.")
            _say("")
            _say("  NEXT: connect the QSFP cable between the two, reboot BOTH with the cable")
            _say("        already in place, then run `unsloth spark up` again.")
            _say("        (Hot-plugging the cable can leave the ConnectX-7 throttled, which")
            _say("         looks like a slow link and needs another reboot to clear.)")
        # The buying question a lone owner actually has, answered with the measurement
        # rather than with "it depends". The gain is very unevenly distributed across
        # workloads, and the average of the two numbers helps nobody.
        _say("")
        _say("  Would a second Spark be worth it? It depends on your prompts, more than")
        _say("  most people expect. Measured on Llama-3.3-70B fp8, two Sparks vs one:")
        _say("")
        _say("    prefill throughput   166 -> 643 tok/s     3.87x")
        _say("    median TTFT         3085 -> 797 ms        3.87x")
        _say("    decode (TPOT)      332.7 -> 162.4 ms      2.09x")
        _say("")
        _say("  RAG, summarisation, long documents and code review are prefill-heavy and")
        _say("  gain the 3.87x. Short-prompt chat gains the 2.09x. A model that already")
        _say("  fits on one Spark gains nothing in decode from a layer split (0.85x to")
        _say("  1.01x measured, 1 to 32 users); with 8 or more concurrent users two")
        _say("  replicas of it measured 1.30x to 1.91x instead, and below 8 one Spark")
        _say("  is as good as two.")
        if _kernel_banner(kernels):
            _say("")
            _say(f"NEXT: {KERNEL_INSTALL}")
        else:
            _say("")
            _say("  The FlashInfer CUTLASS prefill kernel is installed. You are set up.")
            _next_steps(sit)
        raise typer.Exit(0)

    # ── Cable present but never paired ──
    if state != "configured":
        _heading("Status")
        _say("  A QSFP cable is connected, but the two Sparks are not paired yet.")
        _say("  Pairing assigns static IPs on both rails, sets MTU 9000, and writes the")
        _say("  GB10-correct NCCL settings. Nothing else has to be configured by hand.")
        _say("")
        _say("  Read this before agreeing: `unsloth spark setup` also copies this")
        _say("  machine's environment to the peer with `rsync -a --delete`. On a peer")
        _say("  that is running someone else's job, that is destructive. Make sure the")
        _say("  peer is idle, or run the pieces yourself.")
        if sit["seen"] > 2:
            _say("")
            _say(f"  You have {sit['seen']} Sparks visible. Three or more cannot be cabled")
            _say("  point-to-point, so pairing them needs a switched RoCE fabric and an")
            _say(f"  explicit `--nodes {min(sit['seen'], sit['max_nodes'])} --switched`.")
        _say("")
        _say("NEXT: unsloth spark setup")
        if check:
            raise typer.Exit(1)
        # Never silently: setup mutates the peer, so it needs a yes that was actually
        # given. --yes is that yes; a prompt is that yes; a pipe is not.
        import sys

        agreed = yes or (
            sys.stdin.isatty()
            and typer.confirm(
                "\n  Pair the two Sparks now (this WILL overwrite the peer's environment)?",
                default = False,
            )
        )
        if not agreed:
            _say("")
            _say("  Nothing changed. Run the command above when the peer is idle.")
            raise typer.Exit(1)
        _say("")
        rc = sc.main(["setup", "--yes"])
        if rc != 0:
            _say("")
            _say("Setup did not finish. Re-run `unsloth spark up` once it is resolved.")
            raise typer.Exit(rc)
        sit = _situation(sc)
        state = sit["state"]
        if state != "configured":
            _say("")
            _say("NEXT: unsloth spark up   (re-run once the rails carry addresses)")
            raise typer.Exit(1)

    # ── Paired. Is the peer actually there? ──
    peer = sit["peer_ip"]
    _heading("Peer")
    reach = _peer_reachable(peer) if peer else None
    if reach is False:
        _field("peer", f"{peer} -- NOT RESPONDING")
        _say("")
        _say("  The pairing is configured but the peer is not answering on the fast link.")
        _say("  Usually it is powered off, or its cable came out. If it IS powered on and")
        _say("  the cable is seated, this is one of the two hardware faults:")
        _say("")
        _say("    - GPU enumerates but cuInit returns 100, dmesg shows 0xbadf5600")
        _say("      -> a PLAIN REBOOT of that node fixes it. A module reload does not.")
        _say("    - NCCL capped near 3 GB/s while raw RDMA reads 24.5 GB/s")
        _say("      -> a FULL POWER CYCLE, both nodes, ~30 s unplugged. A reboot does NOT")
        _say("         fix this one.")
        _say("")
        _say("  Same symptom class, opposite remedies -- `unsloth spark doctor` tells them")
        _say("  apart once the peer is reachable again.")
        _say("")
        _say(f"NEXT: power on the peer, then re-run `unsloth spark up`")
        raise typer.Exit(1)
    if reach is None:
        _field("peer", f"{peer or 'unknown'} -- could not test")
    else:
        _field("peer", f"{peer} -- reachable")

    # ── Peer reachable: is it provisioned? ──
    venv = _peer_has_venv(peer) if (peer and reach) else None
    if venv is False:
        _field("peer env", "MISSING the Unsloth venv")
        _say("")
        _say("  A two-node job against a peer without the venv does not report an error.")
        _say("  The head blocks for 601 seconds and dies with 'DistStoreError: 1/2 clients")
        _say("  joined', which names neither the peer nor the cause.")
        _say("")
        _say("  Copying it over the fast link takes minutes; installing on the peer takes")
        _say("  hours (HuggingFace measures ~20 KB/s from these machines, the link ~444 MB/s)")
        _say("  and can drift the two environments apart.")
        if check:
            _say("")
            _say("NEXT: unsloth spark provision")
            raise typer.Exit(1)
        import sys

        go = yes or (
            sys.stdin.isatty()
            and typer.confirm("\n  Copy this Spark's environment to the peer now?", default = True)
        )
        if not go:
            _say("")
            _say("NEXT: unsloth spark provision")
            raise typer.Exit(1)
        _say("")
        rc = sc.main(["provision"])
        if rc != 0:
            _say("")
            _say("NEXT: unsloth spark provision   (retry; the copy did not complete)")
            raise typer.Exit(rc)
    elif venv is True:
        _field("peer env", "Unsloth venv present")
    else:
        _field("peer env", "could not check (no ssh on this machine)")

    # ── Everything structural is in place ──
    _heading("Ready")
    _say(f"  {sit['nodes']} Sparks, paired, peer reachable.")
    _say("")
    _say("  What the second Spark buys, measured on Llama-3.3-70B fp8:")
    _say("")
    _say("    tensor parallel (TP=2)   2.09x / 2.13x / 2.10x / 1.97x at concurrency 1/2/4/8")
    _say(
        f"                             median TPOT {TPOT_ONE_SPARK_MS:.1f} ms -> "
        f"{TPOT_TP2_MS:.1f} ms"
    )
    _say("    pipeline parallel (PP=2) 1.08x / 1.11x / 1.09x / 1.07x, TPOT FLAT")
    _say("                             i.e. capacity, NOT latency -- do not use it for speed")
    _say(
        f"    prefill / TTFT (TP=2)    3.87x  ({PREFILL_TOKS[0]} -> {PREFILL_TOKS[1]} tok/s, "
        f"TTFT {TTFT_MS[0]} -> {TTFT_MS[1]} ms)"
    )
    _say("                             so RAG and long-prompt work gain far more than chat")
    _say("    two replicas             1.30x / 1.75x / 1.91x aggregate decode at 8 / 16 / 32")
    _say("                             users (1.00x to 1.13x below 8), per-request latency")
    _say("                             unchanged; measured on Qwen3.8-27B Q4_K_XL, llama.cpp")
    _say("    layer-split a model")
    _say("      that FITS on one node  decode 0.85x to 1.01x at 1 to 32 users, never a win;")
    _say("                             prefill 1.7x to 1.85x")
    _say("")
    _say("  So: a second Spark is for models that do not fit, for tensor parallelism, and")
    _say("  for serving 8 or more users at once as two replicas. It will not make a")
    _say("  model that already fits decode faster.")
    _say("")
    _say("  The link is not measured here -- only a real NCCL collective can tell a")
    _say("  throttled link from a healthy one, and it takes ~30 s:")
    _say("    unsloth spark doctor")

    _kernel_banner(kernels)
    _next_steps(sit)
    raise typer.Exit(0)


# ── Existing subcommands, unchanged in behaviour ─────────────────────────────


@spark_app.callback(invoke_without_command = True)
def _default(ctx: typer.Context) -> None:
    """Run the guided setup when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(up, yes = False, check = True)


@spark_app.command("status")
def status(
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help = "Measure the rail with ib_write_bw. Needs perftest on both nodes.",
    ),
) -> None:
    """Show the ConnectX rails, any peer Spark, and link health.

    `--benchmark` exists because the status output tells you to run it: carrier
    counters cannot tell a throttled link from a healthy one, and only a measurement
    can. It drives ib_write_bw on both nodes and takes a few seconds.
    """
    rc = _cluster().main(["status"] + (["--benchmark"] if benchmark else []))
    # Appended, not woven in: `status` keeps printing exactly what it printed before,
    # and this adds the one thing a single-Spark owner most needs to hear. Most Spark
    # owners have exactly one Spark, and the 6.2x prefill kernel is a larger win than
    # a second machine would have been.
    sc = _cluster_or_none()
    if sc is not None and _on_spark(sc):
        _kernel_banner()
    raise typer.Exit(rc)


@spark_app.command("setup")
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help = "Do not prompt."),
    nodes: int = typer.Option(None, "--nodes", "-n", help = "How many Sparks to address. Default 2."),
    switched: bool = typer.Option(
        False, "--switched", help = "All Sparks share a switched RoCE fabric. Required above two."
    ),
) -> None:
    """Plan and configure the link between the Sparks.

    Two Sparks are the easy case: one QSFP cable, one flat /24 per PCIe function, no
    switch. THREE OR MORE cannot be cabled that way, so `--nodes 3` without
    `--switched` is REFUSED rather than answered. That refusal is the correct answer,
    not a missing feature: on daisy-chained Sparks each cable needs its own subnet, and
    a flat plan would put two nodes that share no cable on the same subnet and
    black-hole every route between them. A netplan that looks right and does not work
    costs far more to debug than a refusal costs to read.

    Note that this also rsyncs this machine's environment onto the peer, which is what
    prevents the 601 s `DistStoreError: 1/2 clients joined` from a missing venv. It
    uses `--delete`, so do not run it against a peer that is busy.
    """
    argv = ["setup"]
    if yes:
        argv.append("--yes")
    if nodes is not None:
        argv += ["--nodes", str(int(nodes))]
    if switched:
        argv.append("--switched")
    raise typer.Exit(_cluster().main(argv))


@spark_app.command("peers")
def peers(
    probe: bool = typer.Option(
        True, "--probe/--no-probe", help = "TCP-probe each peer for reachability."
    ),
) -> None:
    """List every Spark this node can see, in the order the planner ranks them.

    The ordering is by address and is identical on every node, which is what makes the
    index usable as a node rank -- an order that depended on who ran the discovery
    would hand two nodes the same rank.

    Seeing a Spark is not the same as being able to cluster with it. mDNS answers over
    whatever interface advertised, usually Wi-Fi, which says nothing about whether the
    fast RoCE link between you exists.
    """
    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)
    if not _on_spark(sc):
        _say("Not a DGX Spark; there are no Spark peers to look for.")
        raise typer.Exit(0)
    argv = ["peers"] + ([] if probe else ["--no-probe"])
    try:
        rc = sc.main(argv)
    except SystemExit:
        raise
    except Exception:
        # An older build has no `peers` command; do it here rather than traceback.
        rc = 0
        try:
            found = sc.discover_peers(check_reachable = probe).get("peers") or []
        except Exception:
            found = []
        if not found:
            _say("  No peer Sparks discovered.")
        for peer in found:
            _say(
                f"    node {peer.get('index', '?')}  "
                f"{peer.get('short', '?'):<16} {peer.get('address', '?')}"
            )
    raise typer.Exit(rc)


@spark_app.command("env")
def env() -> None:
    """Print the GB10-correct NCCL settings as shell exports.

    Handy as `eval "$(unsloth spark env)"` before a torchrun/NCCL job that
    Unsloth is not launching itself.
    """
    raise typer.Exit(_cluster().main(["env"]))


@spark_app.command("serve")
def serve(
    model: str = typer.Option(..., "--model", "-m", help = "Path to a .gguf."),
    port: int = typer.Option(8080, "--port", "-p"),
    ctx: int = typer.Option(8192, "--ctx"),
    engines: int = typer.Option(
        2,
        "--engines",
        help = "Independent engines. 2 measured 1.35x one Spark; a single split "
        "engine never beats one Spark on decode (0.85x to 1.01x measured).",
    ),
    slots: int = typer.Option(16, "--slots", help = "Server slots per engine."),
) -> None:
    """Serve a GGUF split across both Sparks via llama.cpp's RPC backend.

    Defaults to TWO engines, each split across both Sparks, behind a round-robin front
    end -- measured at 1.35x a single Spark, where ONE split engine never beats a single
    Spark on decode (0.85x to 1.01x measured from 1 to 32 users). The difference is that
    two engines give the pair data-independent work; a single autoregressive stream
    cannot be pipelined, which is why vLLM and SGLang also require pp_size independent
    batches in flight. A model that fits on one Spark is served as two independent
    replicas instead, which measured 1.30x to 1.91x at 8 to 32 users.

    Before a split is printed, both nodes' llama.cpp bundles are compared and any RPC
    server already listening is asked its protocol version: a peer on a different
    bundle fails at load with "RPC server version mismatch", and the fix is
    `unsloth spark provision`.
    """
    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)
    _require_spark(sc, "there is no second Spark here to serve across.")
    raise typer.Exit(
        sc.main(
            [
                "serve",
                "--model",
                model,
                "--port",
                str(port),
                "--ctx",
                str(ctx),
                "--engines",
                str(engines),
                "--slots",
                str(slots),
            ]
        )
    )


@spark_app.command("train")
def train(
    script: str = typer.Option(
        "", "--script", "-s", help = "Your training script, replicated per node (DDP)."
    ),
    layer_split: str = typer.Option(
        "", "--layer-split", "-L", help = "Model to split across the Sparks instead."
    ),
    data_parallel: str = typer.Option(
        "",
        "--data-parallel",
        "-D",
        help = "Model to replicate on both Sparks (DDP over the LoRA gradients). "
        "Throughput, not capacity: it must fit on one Spark.",
    ),
    fsdp: bool = typer.Option(
        False,
        "--fsdp",
        help = "With --data-parallel: shard the base weights across the pair "
        "instead of replicating them.",
    ),
    shard_load: bool = typer.Option(
        False,
        "--shard-load",
        help = "Load only each node's own layers. Required for a model larger than one Spark.",
    ),
    microbatches: int = typer.Option(
        32,
        "--microbatches",
        help = "Higher fills the pipeline better. Measured on "
        "two Sparks vs one: M=4 1.13x, M=8 1.56x, "
        "M=16 1.70x, M=32 1.96x. The ceiling is 2M/(M+1), "
        "so M=32 already reaches 99% of it and going "
        "beyond gains almost nothing.",
    ),
    pp_backend: str = typer.Option(
        "torch",
        "--pp-backend",
        help = "torch (default) uses torch.distributed.pipelining. "
        "legacy uses our hand-written schedules, kept only "
        "as a control arm -- it is slower and its 1f1b and "
        "interleaved deadlock.",
    ),
    schedule: str = typer.Option(
        "1f1b",
        "--schedule",
        help = "On --pp-backend torch, measured on two Sparks vs one: "
        "1f1b 1.94x (default), dualpipev 1.96x, zbv 1.94x, "
        "interleaved 1.93x, gpipe 1.86x, zerobubble 1.72x. "
        "1f1b is the default over dualpipev because the 0.7% "
        "gap is within noise while 1f1b's peak memory is lower "
        "(7.34 vs 9.58 GiB). Avoid gpipe: it peaked at 99.92 "
        "GiB for the same work, against a 121.69 GiB node.",
    ),
    steps: int = typer.Option(20, "--steps"),
    batch: int = typer.Option(8, "--batch", help = "Global batch per step."),
    seq: int = typer.Option(512, "--seq"),
    full_finetune: bool = typer.Option(False, "--full-finetune"),
    master_port: int = typer.Option(29500, "--master-port"),
    run: bool = typer.Option(
        False, "--run", help = "Launch it on both Sparks instead of printing commands."
    ),
) -> None:
    """Print the two-node commands for DDP (--script), a layer split (--layer-split) or
    a built-in data-parallel LoRA run (--data-parallel).

    The modes solve different problems. DDP replicates the model for throughput, so it
    must still fit on one Spark; --data-parallel is that, on the same loader, LoRA and
    loss as the layer split, so the two can be compared on one model. A layer split
    divides the decoder stack across both, which is the only way to train something
    larger than one Spark's ~117 GiB -- verified here on Llama-3.3-70B, which is 132 GiB
    and cannot be trained on a single node.
    """
    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)
    _require_spark(
        sc,
        "these are two-Spark launch commands and do not apply here. "
        "Train as usual with `unsloth train`.",
    )
    if not script and not layer_split and not data_parallel:
        typer.echo(
            "give one of --script <train.py>, --layer-split <model> or --data-parallel <model>."
        )
        raise typer.Exit(2)
    if layer_split and data_parallel:
        typer.echo("--layer-split and --data-parallel are different topologies; give one.")
        raise typer.Exit(2)
    if data_parallel:
        layer_split = data_parallel
    if layer_split:
        extra = [
            f"--microbatches {microbatches}",
            f"--schedule {schedule}",
            f"--pp-backend {pp_backend}",
            f"--steps {steps}",
            f"--batch {batch}",
            f"--seq {seq}",
        ]
        if shard_load:
            extra.append("--shard-load")
        if full_finetune:
            extra.append("--full-finetune")
        if data_parallel:
            extra.append("--data-parallel")
            if fsdp:
                extra.append("--fsdp")
        argv = [
            "train",
            "--layer-split",
            layer_split,
            "--master-port",
            str(master_port),
            "--pipeline-args",
            " ".join(extra),
        ]
        if run:
            argv.append("--run")
        raise typer.Exit(sc.main(argv))
    raise typer.Exit(sc.main(["train", "--script", script]))


@spark_app.command("doctor")
def doctor(
    parity_only: bool = typer.Option(
        False, "--parity-only", help = "Only compare the two nodes' capability gates. No GPU work."
    ),
    deep: bool = typer.Option(
        False, "--deep", help = "Also import torch/vllm on both nodes. This initialises CUDA on both."
    ),
    skip_parity: bool = typer.Option(
        False, "--skip-parity", help = "Do not run the cross-node parity check."
    ),
) -> None:
    """Measure the link and diagnose the two DGX Spark hardware faults.

    Worth running whenever two-Spark training feels slow. The fault this catches drops
    NCCL from ~21 GB/s to ~3 GB/s -- a 7x hit on every gradient all-reduce -- while
    leaving raw RDMA (`ib_write_bw`) reading a healthy 24.5 GB/s, so nothing cheaper
    than a real collective will reveal it. It needs a FULL POWER CYCLE; a reboot does
    not clear it, which is why it can persist for days. The other fault looks similar
    and has the opposite remedy: a GPU whose compute engine is dead (cuInit returns
    100, dmesg shows 0xbadf5600) needs a PLAIN REBOOT, and a module reload will not do.

    It also compares the two nodes' capability gates. A gate that differs between ranks
    -- `which(nvcc)` finding CUDA on one node's PATH and not the other's is the case that
    cost this project the most -- changes which collectives each rank executes, and a
    collective entered by only some ranks does not raise. It waits, for 1800 s, and then
    reports a gloo transport error that names nothing related to the cause.
    """
    from unsloth_cli.commands.doctor import _workload_guidance, check_parity

    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)
    if not _on_spark(sc):
        _say("This machine is not a DGX Spark; nothing to check.")
        raise typer.Exit(0)
    try:
        peer_ip = sc.peer_ip_for()
    except Exception:
        peer_ip = None
    parity_rc = 0
    if peer_ip and not skip_parity:
        parity_rc = check_parity(peer_ip, deep = deep)
    if parity_only:
        _workload_guidance()
        _kernel_banner()
        raise typer.Exit(1 if parity_rc else 0)
    rc = sc.main(["doctor"])
    _workload_guidance()
    _kernel_banner()
    raise typer.Exit(rc or (1 if parity_rc else 0))


@spark_app.command("provision")
def provision(
    dry_run: bool = typer.Option(
        False, "--dry-run", help = "Show what would be copied, without copying."
    ),
    no_fast: bool = typer.Option(
        False,
        "--no-fast",
        help = "Copy over ssh only. By default bulk bytes go through an ephemeral rsync "
        "daemon on the peer, unencrypted over the direct rail cable (a point-to-point "
        "link with no other host), locked to this node's rail address and a one-shot "
        "secret, and stopped when the command ends. Same as UNSLOTH_SPARK_PROVISION_FAST=0.",
    ),
) -> None:
    """Copy this Spark's environment and warm caches to the peer over the fast link.

    Run after pairing, and again whenever this node's environment changes. It copies
    rather than installs because HuggingFace measures ~20 KB/s from these machines
    while the ConnectX link does ~1 GB/s -- and because copying cannot produce the
    dependency drift that two separate installs can.

    It also prevents two silent failures. A peer missing the venv makes the head block
    for 601 seconds and die with `DistStoreError: 1/2 clients joined`, never showing
    the worker's real error. A peer missing a warm cache rebuilds it from scratch,
    which looks like a 17-minute hang during CUDA graph capture.

    Bulk bytes travel unencrypted over the direct cable between the two Sparks unless
    --no-fast is given; ssh over the same link is 4x slower than the disk.
    """
    argv = ["provision"]
    if dry_run:
        argv.append("--dry-run")
    if no_fast:
        argv.append("--no-fast")
    raise typer.Exit(_cluster().main(argv))


# ── The planner ──────────────────────────────────────────────────────────────

# The intents the module understands, plus "auto" which this layer resolves for
# someone who has not thought about it yet.
_FALLBACK_INTENTS = ("latency", "throughput", "capacity")


def _intents(sc) -> tuple:
    value = getattr(sc, "INTENTS", None)
    if isinstance(value, (list, tuple)) and value:
        return tuple(value)
    return _FALLBACK_INTENTS


def _expected_line(exp: dict | None) -> list:
    """Format an expected-gain dict without ever dressing an estimate as a measurement."""
    if not exp:
        return []
    lines = []
    speedup = exp.get("speedup")
    if speedup is None:
        lines.append("  expected  : NOT MEASURED at this node count")
    else:
        lines.append(
            f"  expected  : {speedup:.2f}x "
            + ("(measured)" if exp.get("measured") else "(NOT measured here)")
        )
    # Only when the note does not already say it -- the same fact twice reads as a bug.
    if exp.get("aggregate") and "aggregate" not in (exp.get("note") or "").lower():
        lines.append(
            f"              ~{exp['aggregate']:.0f}x aggregate throughput, " f"1.00x per request"
        )
    if exp.get("note"):
        lines.append(f"              {exp['note']}")
    return lines


def _print_wrapped(
    text: str,
    indent: str = "  ",
    width: int = 78,
) -> None:
    """Wrap to a fixed width. Plain ASCII, no colour, no terminal queries."""
    import textwrap
    for line in textwrap.wrap(text, width = width - len(indent)) or [""]:
        _say(indent + line)


@spark_app.command("plan")
def plan(
    model: str = typer.Option(
        ..., "--model", "-m", help = "Path, directory, or HF repo id (must be cached)."
    ),
    intent: str = typer.Option(
        "auto", "--intent", "--for", "-f", help = "latency | throughput | capacity | auto."
    ),
    nodes: int = typer.Option(
        None, "--nodes", "-n", help = "How many Sparks to plan for. Default: discovered."
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", "-c", help = "Requests in flight, for the expected number."
    ),
    prompt_tokens: int = typer.Option(
        512, "--prompt-tokens", help = "Typical prompt length, for the llama.cpp layout."
    ),
    prefill_heavy: bool = typer.Option(
        False,
        "--prefill-heavy",
        help = "The work is prefill-heavy long-prompt work (RAG, documents). The only "
        "case in which a model that fits is layer-split.",
    ),
) -> None:
    """Say exactly how to deploy a model here, and what it will buy you.

    Ask for what you actually want -- `--for latency`, `--for throughput` or
    `--for capacity` -- and this prints the command to run and the measured speedup to
    expect. The intent is what decides the ANSWER, not just the wording: the same model
    on the same two Sparks is tensor-parallel for latency and independent replicas for
    throughput, and those are different commands with different results.

    It will also say plainly when more Sparks buy you nothing. A model that already fits
    on one Spark never decodes faster split across two (0.85x to 1.01x measured from 1 to
    32 users), so that layout is recommended only for a model that does not fit, or with
    `--prefill-heavy` for long-prompt work where the split's 1.7x to 1.85x prefill is the
    point. Two replicas of a model that fits measured 1.30x at 8 users, 1.75x at 16 and
    1.91x at 32, and only 1.00x to 1.13x below 8, so `--concurrency` decides between one
    Spark and two. Anything not measured at your node count is reported as not measured
    rather than extrapolated.
    """
    sc = _cluster_or_none()
    if sc is None:
        raise typer.Exit(1)
    choice = (intent or "auto").strip().lower()
    valid = _intents(sc)
    if choice != "auto" and choice not in valid:
        _say(f"--for must be one of: auto, {', '.join(valid)}")
        raise typer.Exit(2)
    if not _on_spark(sc):
        _say("Not a DGX Spark; these layouts and measurements do not apply here.")
        raise typer.Exit(0)

    sit = _situation(sc, discover_timeout = 0.0)
    if nodes is None:
        nodes = sit["nodes"]
    nodes = max(1, int(nodes))

    try:
        size = sc.model_size_gib(model)
    except Exception:
        size = None

    # "auto" is resolved here rather than in the module: latency is what someone who
    # has not thought about it usually means, and capacity is the only honest answer
    # when the thing does not fit at all.
    budget = _serve_budget(sc)
    resolved = choice
    if resolved == "auto":
        resolved = "latency" if (size is not None and size <= budget) else "capacity"

    result = _plan_deployment(
        sc,
        size,
        nodes,
        intent = resolved,
        concurrency = concurrency,
        model = model,
        prompt_tokens = prompt_tokens,
        prefill_heavy = prefill_heavy,
    )
    if result is None:
        _say("Could not produce a plan (the planner is unavailable in this build).")
        raise typer.Exit(1)

    _heading("Deployment plan")
    _field("model", model)
    _field("size", f"{size:.1f} GiB" if size else "unknown (not cached locally)")
    _field(
        "per-Spark",
        f"{budget:.0f} GiB usable for a served model "
        f"({_usable_gib(sc):.2f} GiB total; the '{MARKETED_GIB:.0f} GB' "
        f"on the box is GiB, ~6.3 of which is firmware-reserved)",
    )
    _field(
        "Sparks",
        str(nodes)
        + ("" if nodes == sit["nodes"] else f"  (asked for; {sit['nodes']} actually paired here)"),
    )
    _field("optimising", resolved + (" (auto)" if choice == "auto" else ""))
    if concurrency != 1:
        _field("concurrency", str(concurrency))
    _field("topology", str(result.get("topology")))
    if result.get("axis"):
        _field("axis", str(result.get("axis")))

    if result.get("topology") == "unknown":
        _heading("No recommendation")
        _print_wrapped(
            "The model's size could not be determined, so there is nothing to "
            "recommend. Guessing would hand you a confidently wrong layout, which is "
            "worse than saying nothing."
        )
        _say("")
        _print_wrapped(
            "Point --model at a local .gguf, a local directory, or a repo id that is "
            "already in ~/.cache/huggingface/hub."
        )
        raise typer.Exit(1)

    _heading("Recommendation")
    # One voice. `summary` is the memory-fit narrative and, for a model that needs both
    # nodes, it names the llama.cpp layer-split; `recommendation` names tensor parallel.
    # Both are true of different engines, but printing them as consecutive paragraphs
    # reads as the tool contradicting itself, so the axis answer wins and the fit
    # narrative is shown only when there is no axis answer (an older build).
    if result.get("recommendation"):
        _print_wrapped(str(result["recommendation"]))
    elif result.get("summary"):
        _print_wrapped(str(result["summary"]))

    for line in _expected_line(result.get("expected")):
        _say(line)

    # The one place the measured number can mislead: every cross-node figure in this
    # project came from Llama-3.3-70B fp8, which does NOT fit on one Spark. Quoting
    # 2.09x for a model that does fit would present a number measured on a different
    # regime as though it applied here.
    if result.get("axis") == "tensor-parallel" and result.get("fits_one_node"):
        _say("")
        _print_wrapped(
            "CAVEAT, and read it before you buy anything: the 2.09x was measured on "
            "Llama-3.3-70B fp8, which does not fit on one Spark. This model does fit. "
            "TP adds an all-reduce over the RoCE link for every layer of every token, "
            "and that cost is fixed while the per-node work shrinks with the model -- "
            "so a smaller model keeps less of the 2.09x, and a small enough one can be "
            "SLOWER across two Sparks than on one. Nobody measured this size here. "
            "Benchmark it against a single Spark before you commit, and if it does not "
            "win, serve it on one node."
        )
    if result.get("axis") in ("layer-split", "pipeline-parallel") and result.get("fits_one_node"):
        _say("")
        _print_wrapped(
            "DO NOT DO THIS FOR SPEED: splitting a model that already fits never decodes "
            "faster than the single Spark you already have (0.85x to 1.01x measured from "
            "1 to 32 users). A split of such a model is only for prefill-heavy long-prompt "
            "work, where prefill measured 1.7x to 1.85x."
        )

    # The llama.cpp specific layout: single, replicas or layer split, from the
    # measured table, keyed by the concurrency and prompt length given.
    serving = result.get("serving")
    if isinstance(serving, dict) and serving.get("topology"):
        _say("")
        _field("llama.cpp", str(serving["topology"]).replace("_", " "))
        if serving.get("reason"):
            _print_wrapped(str(serving["reason"]), indent = "                 ")
        if serving.get("measured_on"):
            _print_wrapped(f"(measured on {serving['measured_on']})", indent = "                 ")

    commands = result.get("commands") or []
    if commands:
        _say("")
        _say("  Run:")
        for line in commands:
            _say(f"    {line}")

    if result.get("fallback_axis"):
        _say("")
        _say(f"  Fallback axis: {result['fallback_axis']} -- only if your engine cannot")
        _say("  tensor-parallel across hosts (llama.cpp RPC, for instance).")
        for line in _expected_line(result.get("fallback_expected")):
            _say(line)

    # The one thing the planner cannot know: whether the kernel that is worth more than
    # any of this is installed.
    if nodes > sit["nodes"]:
        _say("")
        _print_wrapped(
            f"This plan assumes {nodes} Sparks. Only {sit['nodes']} "
            f"{'is' if sit['nodes'] == 1 else 'are'} paired on this machine right now, "
            f"so the commands above will not run as written until the rest are paired "
            f"(`unsloth spark peers` to see what is visible). Above two Sparks that "
            f"needs a switched RoCE fabric and an explicit `--switched`."
        )

    _kernel_banner(_kernel_state())
    raise typer.Exit(0)


@spark_app.command("kernels")
def kernels(
    workload: str = typer.Option("mixed", "--workload", help = "decode | prefill | mixed"),
) -> None:
    """Recommend the NVFP4 kernel for a workload on GB10.

    There is no single best kernel: the fastest decode kernel is the slowest prefill
    kernel by 6.5x, and the crossover sits near 256 tokens per forward pass. Measured
    on identical weights with only the kernel changed, so this is not a checkpoint
    difference. vLLM auto-selects sensibly for decode, which makes prefill-heavy and
    long-prompt workloads the case where an explicit flag is worth multiples.
    """
    rc = _cluster().main(["kernels", "--workload", workload])
    sc = _cluster_or_none()
    if sc is not None and _on_spark(sc):
        state = _kernel_state()
        if not state["ok"]:
            _say("")
            _say("  NOT INSTALLED on this machine. Install it before the flag can help:")
            _say(f"    {KERNEL_INSTALL}")
        else:
            _say("")
            _say(
                f"  Installed here: flashinfer-python {state['flashinfer']}"
                + (f", nvidia-cutlass-dsl {state['cutlass']}" if state["cutlass"] else "")
            )
    raise typer.Exit(rc)


@spark_app.command("merge")
def merge(
    save_dir: str = typer.Argument(..., help = "The --save directory holding stage0/, stage1/, ..."),
    out: str = typer.Option(None, "--out", "-o", help = "Where to write the merged adapter."),
    dry_run: bool = typer.Option(False, "--dry-run", help = "Inspect and report; write nothing."),
) -> None:
    """Merge the per-stage adapters from a layer-split run into one loadable checkpoint.

    `spark train --layer-split` leaves one directory per stage, each holding only the LoRA
    weights for the layers that stage owned -- two half-models and nothing you can load.
    This combines them.

    It is a union, not an average: stages replace foreign layers with Identity rather than
    deleting them, so layer numbering is preserved and the two key sets are disjoint by
    construction. If that is ever not true -- overlapping layers, a gap, a missing stage --
    this refuses instead of guessing, because a partly-populated adapter loads without error
    and quietly produces a worse model.

    Needs no GPU and no second Spark; merging is pure file work.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "studio" / "spark_merge.py"
    if not path.is_file():
        _say(f"  spark_merge.py not found at {path}")
        raise typer.Exit(1)
    spec = importlib.util.spec_from_file_location("spark_merge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        rc = mod._cmd_merge(save_dir, out, dry_run)
    except RuntimeError as e:
        _say(f"  {e}")
        raise typer.Exit(1)
    raise typer.Exit(rc)


@spark_app.command("estimate")
def estimate(
    model: str = typer.Option(..., "--model", "-m"),
    batch: int = typer.Option(8, "--batch"),
    microbatches: int = typer.Option(4, "--microbatches"),
    seq: int = typer.Option(512, "--seq"),
    full_finetune: bool = typer.Option(False, "--full-finetune"),
    grad_checkpoint: bool = typer.Option(False, "--grad-checkpoint"),
) -> None:
    """Check whether a layer-split training run will fit, before starting it.

    Worth running every time, because the failure it prevents is slow and severe: a 70B
    arm in this project loaded 66 GiB of weights, spent an hour materialising, then ran
    out of memory and left the node **unreachable over ssh** -- kernel and NIC healthy
    while userspace could no longer fork, which needs a power cycle to clear.

    The estimate is sized on the LAST stage, which is the one that runs out first: it
    holds the logits, and for a 128k vocabulary that single fp32 tensor plus its gradient
    can outweigh every other activation combined.
    """
    argv = [
        "estimate",
        "--model",
        model,
        "--batch",
        str(batch),
        "--microbatches",
        str(microbatches),
        "--seq",
        str(seq),
    ]
    if full_finetune:
        argv.append("--full-finetune")
    if grad_checkpoint:
        argv.append("--grad-checkpoint")
    raise typer.Exit(_cluster().main(argv))
