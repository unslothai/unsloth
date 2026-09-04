# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth doctor` -- check this machine's setup and report what is wrong.

Top-level on purpose: someone whose training is mysteriously slow will try
`unsloth doctor` long before they think to look for a hardware-specific
subcommand. On a DGX Spark pair this runs the link diagnosis; elsewhere it says
so and exits, costing nothing.

It also runs a CAPABILITY-PARITY check across the two nodes, which catches a class
of bug that nothing else here can. See `parity_probe_source` below for why an
asymmetric capability probe is a guaranteed multi-node hang rather than an error.
"""

from __future__ import annotations

import typer

doctor_app = typer.Typer(
    help = "Diagnose this machine's Unsloth setup.",
    no_args_is_help = False,
)


# ── Cross-node capability parity ─────────────────────────────────────────────
#
# The deepest bug found in this project, and the reason this check exists:
#
#   /usr/local/cuda/bin is on the peer's PATH in a LOGIN shell but not in a
#   non-interactive one (`ssh host cmd`, which is how every launch here works)
#     -> shutil.which("nvcc") is None on rank 1 and a path on rank 0
#       -> vLLM's has_flashinfer() is False on rank 1 and True on rank 0
#         -> rank 1 skips flashinfer_autotune(); rank 0 runs it
#           -> rank 0 enters a collective that rank 1 never enters
#             -> 1800 s hang, surfacing as a gloo TCP transport error
#
# The generalisable rule: ANY capability probe evaluated independently per rank --
# `shutil.which(tool)`, a bare try/except import, a feature flag derived from either --
# is a latent multi-node deadlock. A collective entered by some ranks and not others
# does not raise; it waits. The symptom then appears in the communication layer and
# points nowhere near the cause.
#
# What matters is SYMMETRY, not presence. nvcc missing on both nodes is fine: both
# ranks take the same branch. nvcc on one node only is fatal.
#
# Two rules follow for the check itself:
#   * it must run the probe over NON-INTERACTIVE ssh with the peer's own interpreter,
#     exactly as a launch does. A login shell would load /etc/profile.d, find CUDA, and
#     report parity while the real run still hangs.
#   * if the probe cannot RUN, the answer is "unknown", never "OK". A check that fails
#     open is worse than no check, and that failure mode has already bitten this project.

# Environment that is SUPPOSED to differ between the two hosts, so comparing it would
# produce noise that trains people to ignore real findings.
PARITY_SKIP = (
    "host",
    "hostname_resolves_to",
    "VLLM_HOST_IP",
    "MASTER_ADDR",
    "RANK",
    "NODE_RANK",
    "LOCAL_RANK",
)


def parity_probe_source(deep: bool = False) -> str:
    """The probe, as source, run identically on both nodes.

    Deliberately import-free with ``deep`` off: it answers "would this rank take the
    same branch as the other one" using ``shutil.which``, ``importlib.util.find_spec``
    and package metadata, none of which execute the library or touch the GPU. That is
    enough to catch the whole PATH/which class of divergence, and it means the check is
    safe to run while a GPU is busy.

    ``deep`` additionally imports torch, vllm and the flashinfer gates, which is the
    only way to observe a gate that disagrees for a reason other than PATH -- at the
    cost of initialising CUDA on both nodes.
    """
    lines = [
        "import json, os, shutil, socket, sysconfig",
        "import importlib.util",
        "from importlib.metadata import version, PackageNotFoundError",
        "r = {'host': socket.gethostname()}",
        "for t in ('nvcc', 'gcc', 'g++', 'ninja', 'python3', 'torchrun', 'ssh'):",
        "    r['which_' + t] = shutil.which(t)",
        "r['python_h'] = os.path.exists(",
        "    os.path.join(sysconfig.get_paths()['include'], 'Python.h'))",
        "r['python_version'] = '.'.join(map(str, __import__('sys').version_info[:3]))",
        "r['executable'] = __import__('sys').executable",
        "r['PATH'] = os.environ.get('PATH', '')",
        "for k in ('LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES',",
        "          'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'NCCL_IB_HCA',",
        "          'VLLM_HOST_IP', 'TRITON_CACHE_DIR'):",
        "    r[k] = os.environ.get(k)",
        "cuda_ver = None",
        "for p in ('/usr/local/cuda/version.json', '/usr/local/cuda/version.txt'):",
        "    try:",
        "        with open(p) as fh:",
        "            txt = fh.read()",
        "        cuda_ver = (json.loads(txt).get('cuda', {}).get('version')",
        "                    if p.endswith('.json') else txt.strip())",
        "        break",
        "    except Exception:",
        "        continue",
        "if cuda_ver is None and r['which_nvcc']:",
        "    try:",
        "        import subprocess",
        "        out = subprocess.run([r['which_nvcc'], '--version'], capture_output=True,",
        "                             text=True, timeout=20).stdout",
        "        for ln in out.splitlines():",
        "            if 'release' in ln:",
        "                cuda_ver = ln.split('release')[-1].strip()",
        "    except Exception:",
        "        pass",
        "r['cuda_version'] = cuda_ver",
        "r['cuda_symlink'] = os.path.realpath('/usr/local/cuda') "
        "if os.path.exists('/usr/local/cuda') else None",
        "for pkg in ('torch', 'vllm', 'flashinfer-python', 'nvidia-cutlass-dsl',",
        "            'triton', 'transformers'):",
        "    try:",
        "        r['pkg_' + pkg] = version(pkg)",
        "    except PackageNotFoundError:",
        "        r['pkg_' + pkg] = None",
        "    except Exception:",
        "        r['pkg_' + pkg] = 'ERR'",
        "for m in ('flashinfer', 'flashinfer_cubin', 'flashinfer_jit_cache',",
        "          'deep_gemm', 'triton', 'vllm', 'torch'):",
        "    try:",
        "        r['spec_' + m] = importlib.util.find_spec(m) is not None",
        "    except Exception:",
        "        r['spec_' + m] = 'ERR'",
    ]
    if deep:
        # Only under --deep: these execute the libraries, which initialises CUDA.
        lines += [
            "try:",
            "    import torch",
            "    r['torch_cuda_available'] = bool(torch.cuda.is_available())",
            "    r['torch_device_count'] = int(torch.cuda.device_count())",
            "except Exception as e:",
            "    r['torch_import_error'] = type(e).__name__",
            "try:",
            "    from vllm.utils import flashinfer as F",
            "    for g in ('has_flashinfer', 'has_flashinfer_cubin', 'has_flashinfer_comm',",
            "              'has_flashinfer_cutedsl', 'has_nvidia_artifactory'):",
            "        fn = getattr(F, g, None)",
            "        if fn is not None:",
            "            try:",
            "                r['gate_' + g] = bool(fn())",
            "            except Exception as e:",
            "                r['gate_' + g] = 'ERR ' + type(e).__name__",
            "except Exception as e:",
            "    r['flashinfer_gate_error'] = type(e).__name__",
            "try:",
            "    from vllm.utils.deep_gemm import is_deep_gemm_supported",
            "    r['gate_is_deep_gemm_supported'] = bool(is_deep_gemm_supported())",
            "except Exception as e:",
            "    r['gate_is_deep_gemm_supported'] = 'ERR ' + type(e).__name__",
        ]
    lines.append("print('UNSLOTH_PARITY ' + json.dumps(r, sort_keys=True))")
    return "\n".join(lines) + "\n"


def _activate() -> str:
    """Where the peer's venv activate lives, resolved rather than assumed.

    UNSLOTH_STUDIO_HOME moves the venv, and a hardcoded ~/.unsloth/studio path then
    sources nothing on the peer. The visible symptom is not an error but a false
    negative: the probe runs without the venv, `torchrun` is absent, and doctor reports
    "could not measure NCCL bandwidth" on a perfectly healthy pair. Imported lazily and
    behind a fallback so doctor keeps working if spark_cluster is unavailable.
    """
    try:
        from studio.spark_cluster import venv_activate

        return venv_activate()
    except Exception:
        return "$HOME/.unsloth/studio/unsloth_studio/bin/activate"


def _probe_wrapper(source: str) -> str:
    """The shell one-liner that runs the probe, identical on both nodes.

    Symmetry is the whole point. If the local side ran the probe one way and the peer
    another, the difference between the two wrappers would show up as divergence and
    bury the real finding in noise -- which is exactly the failure mode that makes
    people stop reading a check.
    """
    import base64

    blob = base64.b64encode(source.encode()).decode()
    act = _activate()
    return f"[ -f {act} ] && . {act}; " f"echo {blob} | base64 -d | python3 -"


def _extract(stdout: str, stderr: str):
    import json
    for line in reversed((stdout or "").splitlines()):
        if line.startswith("UNSLOTH_PARITY "):
            try:
                return json.loads(line[len("UNSLOTH_PARITY ") :]), None
            except ValueError as exc:
                return None, f"unparseable probe output ({exc})"
    return None, ((stderr or "").strip().splitlines() or ["no output"])[-1][:200]


def _run_probe_local(source: str, timeout: int = 120):
    import shutil
    import subprocess
    import sys

    shell = shutil.which("bash") or shutil.which("sh")
    try:
        if shell:
            proc = subprocess.run(
                [shell, "-c", _probe_wrapper(source)],
                capture_output = True,
                text = True,
                timeout = timeout,
            )
        else:
            # Windows, or an image without a POSIX shell. Not a DGX Spark, so this is
            # only ever reached by a test; run it directly rather than failing.
            proc = subprocess.run(
                [sys.executable, "-c", source], capture_output = True, text = True, timeout = timeout
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return _extract(proc.stdout, proc.stderr)


def _run_probe_peer(
    peer_ip: str,
    source: str,
    timeout: int = 180,
):
    """Run the probe on the peer over NON-INTERACTIVE ssh, as a launch would.

    Not `ssh -t`, not `bash -lc`: a login shell reads /etc/profile.d, which is exactly
    where the CUDA PATH entry that the real run never sees comes from. Checking under a
    login shell would report parity on a pair that is guaranteed to deadlock.

    The venv is sourced because `studio.spark_cluster` sources it for the real launch;
    matching the launch is the whole point.
    """
    import os
    import shutil
    import subprocess

    if not shutil.which("ssh"):
        return None, "no ssh on this machine"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "nvidia"
    remote = _probe_wrapper(source)
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-n",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=8",
                f"{user}@{peer_ip}",
                remote,
            ],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return _extract(proc.stdout, proc.stderr)


def compare_parity(local: dict, peer: dict) -> list:
    """Keys whose values differ, as (key, local, peer). Order is stable."""
    keys = sorted(set(local) | set(peer))
    return [
        (k, local.get(k), peer.get(k))
        for k in keys
        if k not in PARITY_SKIP and local.get(k) != peer.get(k)
    ]


def _fmt(value) -> str:
    if value is None:
        return "<absent>"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    return text if len(text) <= 68 else text[:65] + "..."


def _report_path_divergence(local_path: str, peer_path: str) -> None:
    """PATH is too long to diff by eye, so diff it here."""
    a = list(dict.fromkeys(p for p in (local_path or "").split(":") if p))
    b = list(dict.fromkeys(p for p in (peer_path or "").split(":") if p))
    only_local = [p for p in a if p not in b]
    only_peer = [p for p in b if p not in a]
    for entry in only_local:
        typer.echo(f"      on this Spark only : {entry}")
    for entry in only_peer:
        typer.echo(f"      on the peer only   : {entry}")


def check_parity(peer_ip: str, deep: bool = False) -> int:
    """Compare capability gates across the two nodes. 0 parity, 1 divergence, 2 unknown."""
    source = parity_probe_source(deep = deep)
    typer.echo("")
    typer.echo("Cross-node capability parity")
    typer.echo("----------------------------")
    typer.echo(
        f"  probing this Spark and {peer_ip} over non-interactive ssh"
        + (" (deep: imports torch/vllm)" if deep else "")
        + " ..."
    )

    local, local_err = _run_probe_local(source)
    peer, peer_err = _run_probe_peer(peer_ip, source)

    if local is None or peer is None:
        typer.echo("")
        typer.echo(
            "  UNKNOWN -- the parity probe could not run on "
            + ("this Spark" if local is None else "the peer")
            + "."
        )
        typer.echo(f"    reason: {local_err if local is None else peer_err}")
        typer.echo("    Treat this as divergent until it can be checked. A capability")
        typer.echo("    check that fails open is worse than no check: it reports a healthy")
        typer.echo("    pair right up until the job hangs for 1800 s in the gloo layer.")
        return 2

    diffs = compare_parity(local, peer)
    # PATH is the usual CAUSE, not itself a capability. Counting it as a finding would
    # flag every pair whose shells merely differ in harmless ways, and a check that
    # cries wolf gets ignored -- so it is reported as supporting evidence when
    # something real diverges, and passed over in silence when nothing does.
    capability_diffs = [d for d in diffs if d[0] != "PATH"]
    path_diff = next((d for d in diffs if d[0] == "PATH"), None)
    diffs = capability_diffs
    if not diffs:
        typer.echo("")
        typer.echo(
            f"  OK -- every capability gate matches between "
            f"{local.get('host', 'this Spark')} and {peer.get('host', peer_ip)}."
        )
        if not deep:
            typer.echo(
                "    (Gates that only a real import can evaluate were not checked. Add --deep"
            )
            typer.echo("     for those; it initialises CUDA on both nodes.)")
        return 0

    typer.echo("")
    typer.echo(f"  {len(diffs)} DIVERGENCE(S) -- a multi-node run can DEADLOCK on this.")
    typer.echo("")
    for key, mine, theirs in diffs:
        typer.echo(f"    {key}")
        typer.echo(f"      this Spark : {_fmt(mine)}")
        typer.echo(f"      the peer   : {_fmt(theirs)}")
    if path_diff is not None:
        typer.echo("")
        typer.echo("  Supporting evidence -- the PATH each node's launch actually sees:")
        _report_path_divergence(path_diff[1], path_diff[2])
    typer.echo("")
    typer.echo("  Why this is not cosmetic: each of these is read independently by each")
    typer.echo("  rank to decide which branch to take. A rank that takes the other branch")
    typer.echo("  skips a collective the other rank enters, and a collective that only")
    typer.echo("  some ranks enter does not raise -- it waits. The 1800 s hang that")
    typer.echo("  follows surfaces as a gloo TCP transport error, which names nothing")
    typer.echo("  related to the cause.")
    typer.echo("")
    typer.echo("  FIX: do not rely on the peer's inherited PATH. A non-interactive ssh")
    typer.echo("  does not read /etc/profile.d, so entries like /usr/local/cuda/bin are")
    typer.echo("  present in a login shell and absent in the launch. Set it explicitly in")
    typer.echo("  the launch wrapper, on BOTH nodes, so the two ranks cannot disagree:")
    typer.echo("")
    typer.echo("    export PATH=/usr/local/cuda/bin:$PATH")
    typer.echo("")
    typer.echo("  Symmetry is what matters, not presence: a tool missing on BOTH nodes is")
    typer.echo("  safe, because both ranks then take the same branch.")
    return 1


def _workload_guidance() -> None:
    """Which workloads a second Spark actually helps, measured.

    Worth printing here because the decision people get wrong is not "how do I configure
    it" but "will it help me at all", and the answer depends on the shape of the traffic
    far more than most people expect.
    """
    typer.echo("")
    typer.echo("What a second Spark is worth, by workload")
    typer.echo("----------------------------------------")
    typer.echo("  Measured on Llama-3.3-70B fp8, two Sparks (tensor parallel) versus one:")
    typer.echo("")
    typer.echo("    prefill throughput   166 -> 643 tok/s      3.87x")
    typer.echo("    median TTFT         3085 -> 797 ms         3.87x")
    typer.echo("    decode (TPOT)      332.7 -> 162.4 ms       2.09x")
    typer.echo("")
    typer.echo("  So a PROMPT-HEAVY workload -- RAG, summarisation, long documents, big")
    typer.echo("  system prompts, code review -- gains considerably more from a second")
    typer.echo("  Spark than a chat workload does. Prefill is compute-bound and splits")
    typer.echo("  almost perfectly; decode is memory-bound and splits less well.")
    typer.echo("")
    typer.echo("  If your traffic is short prompts and long answers, budget for the 2.09x,")
    typer.echo("  not the 3.87x.")


@doctor_app.callback(invoke_without_command = True)
def doctor(
    ctx: typer.Context,
    parity_only: bool = typer.Option(
        False,
        "--parity-only",
        help = "Only compare the two nodes' capability gates. No GPU work, no NCCL run.",
    ),
    deep: bool = typer.Option(
        False,
        "--deep",
        help = "Also import torch/vllm on both nodes to compare their runtime gates. "
        "This initialises CUDA on both.",
    ),
    skip_parity: bool = typer.Option(
        False, "--skip-parity", help = "Do not run the cross-node parity check."
    ),
) -> None:
    """Check the setup and report anything that would slow training down, or hang it.

    Three classes of problem, all of which are silent until they are expensive:

    * a power-delivery fault that drops NCCL from ~21 GB/s to ~3 GB/s -- a 7x hit on
      every gradient all-reduce -- while raw RDMA still reads a healthy 24.5 GB/s, so
      nothing cheaper than a real collective reveals it. It survives a REBOOT and needs
      a full power cycle, so it can persist silently for days;
    * a GPU whose compute engine is dead: nvidia-smi works, cuInit returns 100, dmesg
      shows 0xbadf5600. That one needs a plain reboot and a power cycle is not required
      -- the same symptom class as the above, with the opposite remedy;
    * capability divergence between the two nodes, which does not slow anything down --
      it deadlocks the job for 1800 s in a library that has nothing to do with the cause.
    """
    if ctx.invoked_subcommand is not None:
        return
    from studio import spark_cluster

    if not spark_cluster.is_dgx_spark():
        typer.echo("No machine-specific checks apply here (not a DGX Spark).")
        raise typer.Exit(0)

    try:
        peer_ip = spark_cluster.peer_ip_for()
    except Exception:
        peer_ip = None

    parity_rc = 0
    if peer_ip and not skip_parity:
        parity_rc = check_parity(peer_ip, deep = deep)
    elif not peer_ip and not skip_parity:
        typer.echo("")
        typer.echo("Cross-node capability parity")
        typer.echo("----------------------------")
        typer.echo("  No configured peer, so there are no two nodes to compare.")
        typer.echo("  (Pair one with `unsloth spark up`.)")

    if parity_only:
        _workload_guidance()
        raise typer.Exit(1 if parity_rc else 0)

    link_rc = spark_cluster.main(["doctor"])
    _workload_guidance()
    raise typer.Exit(link_rc or (1 if parity_rc else 0))
