# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth doctor` -- check this machine's setup and report what is wrong.

Top-level on purpose: someone whose training is mysteriously slow reaches for
`unsloth doctor` before a hardware-specific subcommand. It also runs two cross-node
checks: CAPABILITY PARITY, where an asymmetric probe is a guaranteed hang rather than an
error, and FAST-PATH parity, its quieter cousin, which hangs nothing and raises nothing
and simply runs one node at a fraction of the speed.
"""

from __future__ import annotations

import typer

doctor_app = typer.Typer(
    help = "Diagnose this machine's Unsloth setup.",
    no_args_is_help = False,
)


# The deepest bug found here, and why this check exists: /usr/local/cuda/bin is on the peer's
# PATH in a LOGIN shell but not in a non-interactive one (`ssh host cmd`, which is how every
# launch works), so `which("nvcc")` differed by rank, vLLM's has_flashinfer() differed with
# it, rank 0 entered a collective rank 1 never entered, and the job hung for 1800 s surfacing
# as a gloo TCP error. Generally: ANY capability probe evaluated independently per rank is a
# latent multi-node deadlock, because a collective some ranks skip does not raise, it waits.
# SYMMETRY is what matters, not presence -- nvcc missing on BOTH nodes is fine. So the probe
# must run over NON-INTERACTIVE ssh, and if it cannot run the answer is "unknown", never OK.

# Environment that is SUPPOSED to differ between the hosts; comparing it would be noise.
PARITY_SKIP = (
    "host",
    "home",
    "hostname_resolves_to",
    "VLLM_HOST_IP",
    "MASTER_ADDR",
    "RANK",
    "NODE_RANK",
    "LOCAL_RANK",
)


def parity_probe_source(deep: bool = False) -> str:
    """The probe, as source, run identically on both nodes. Import-free with ``deep`` off, so
    it never touches the GPU and is safe while one is busy, and that still catches the whole
    PATH/which class. ``deep`` also imports torch and vllm, initialising CUDA on both."""
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
        # Reported so path-valued probes can be compared relative to each node's own home,
        # which provisioning already allows to differ. Skipped in the comparison itself.
        "r['home'] = os.path.expanduser('~')",
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


def _activate_sh() -> str:
    """Where the peer's venv activate lives, as ONE shell word.

    Resolved rather than assumed, because UNSLOTH_STUDIO_HOME moves it and a hardcoded path
    sources nothing, which reads as "could not measure NCCL bandwidth" on a healthy pair. And
    quoted, because a custom home with a space produced
    `[ -f /path with space/bin/activate ]`, a five-argument test: activation was skipped
    without a word of complaint and the probe ran under whichever python3 was on PATH, so a
    supported custom location reported false package and parity results."""
    try:
        from studio.spark_cluster import venv_activate_sh
        return venv_activate_sh()
    except Exception:
        return '"$HOME/.unsloth/studio/unsloth_studio/bin/activate"'


def _probe_wrapper(source: str) -> str:
    """The shell one-liner that runs the probe, identical on both nodes: a difference between
    the two wrappers would itself show up as divergence and bury the real finding."""
    import base64

    blob = base64.b64encode(source.encode()).decode()
    act = _activate_sh()
    return f"[ -f {act} ] && . {act}; " f"echo {blob} | base64 -d | python3 -"


def _extract(
    stdout: str,
    stderr: str,
    marker: str = "UNSLOTH_PARITY",
):
    import json

    prefix = marker + " "
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix) :]), None
            except ValueError as exc:
                return None, f"unparseable probe output ({exc})"
    return None, ((stderr or "").strip().splitlines() or ["no output"])[-1][:200]


def _run_probe_local(
    source: str,
    timeout: int = 120,
    marker: str = "UNSLOTH_PARITY",
):
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
            # Windows or an image with no POSIX shell, so never a Spark: only tests reach this.
            proc = subprocess.run(
                [sys.executable, "-c", source], capture_output = True, text = True, timeout = timeout
            )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return _extract(proc.stdout, proc.stderr, marker)


def _run_probe_peer(
    peer_ip: str,
    source: str,
    timeout: int = 180,
    marker: str = "UNSLOTH_PARITY",
):
    """Run the probe on the peer over NON-INTERACTIVE ssh, as a launch would. Not `ssh -t`,
    not `bash -lc`: a login shell reads /etc/profile.d, which is exactly where the CUDA PATH
    entry the real run never sees comes from, so it would report parity on a doomed pair."""
    import shutil
    import subprocess

    if not shutil.which("ssh"):
        return None, "no ssh on this machine"
    # The same resolution every other SSH here uses. Reading only USER/USERNAME meant that
    # from a service, a cron job or a container -- where neither is set -- the probes went to
    # the literal account `nvidia` and reported UNKNOWN on a pair whose other SSH works.
    user = _ssh_login()
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
    return _extract(proc.stdout, proc.stderr, marker)


def _relative_to_home(probe: dict) -> dict:
    """Path-valued probes rewritten against that node's OWN home.

    A supported pair may use different usernames, and provisioning explicitly allows it, so
    `/home/alice/.unsloth/.../torchrun` and `/home/bob/.unsloth/.../torchrun` are the same
    capability. Compared raw they read as a divergence, doctor returns failure, and it warns of
    a deadlock while both tools are present and equivalent. Only the home prefix is folded: a
    tool at `/usr/bin` on one node and under the managed environment on the other still differs,
    because that one is real."""
    home = (probe.get("home") or "").rstrip("/")
    if not home:
        return probe
    out = dict(probe)
    for key, value in probe.items():
        if not (key.startswith("which_") or key == "executable"):
            continue
        if isinstance(value, str) and value.startswith(home + "/"):
            out[key] = "~" + value[len(home) :]
    return out


def compare_parity(local: dict, peer: dict) -> list:
    local, peer = _relative_to_home(local), _relative_to_home(peer)
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
    a = list(dict.fromkeys(p for p in (local_path or "").split(":") if p))
    b = list(dict.fromkeys(p for p in (peer_path or "").split(":") if p))
    only_local = [p for p in a if p not in b]
    only_peer = [p for p in b if p not in a]
    for entry in only_local:
        typer.echo(f"      on this Spark only : {entry}")
    for entry in only_peer:
        typer.echo(f"      on the peer only   : {entry}")


def check_parity(peer_ip: str, deep: bool = False) -> int:
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
    # PATH is the usual CAUSE, not itself a capability, so it is supporting evidence when
    # something real diverges and silent otherwise; a check that cries wolf gets ignored.
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


# The second-deepest bug, and why THIS check exists: one node was missing `causal_conv1d` and
# `flash-linear-attention`, so Qwen3.5's linear-attention layers silently fell back to the
# torch path -- 2.19x slower and 39% more memory on a byte-identical cell with identical
# clocks and NCCL. Unlike a capability divergence this deadlocks nothing and raises nothing;
# both nodes finish, and the only sign was one "The fast path is not available" line.
# Two rules: COMPARE, do not demand -- a package missing on BOTH nodes may just be a model
# family nobody runs, and a check listing packages nobody needs gets skipped. And ask the
# RUNTIME, not the metadata: an installed extension that does not import is exactly as slow
# as an absent one while reading as present. If it cannot run, the answer is "unknown".

# Distributions that gate a fast path, with metadata aliases, in report order.
FASTPATH_PACKAGES = (
    ("torch", ("torch",)),
    ("transformers", ("transformers",)),
    ("trl", ("trl",)),
    ("peft", ("peft",)),
    ("accelerate", ("accelerate",)),
    ("bitsandbytes", ("bitsandbytes",)),
    ("triton", ("triton", "pytorch-triton")),
    ("flash-attn", ("flash-attn", "flash_attn")),
    ("causal_conv1d", ("causal_conv1d", "causal-conv1d")),
    ("flash-linear-attention", ("flash-linear-attention", "flash_linear_attention")),
    ("fla-core", ("fla-core", "fla_core")),
    ("xformers", ("xformers",)),
)

# What each one buys, one line, so a finding explains itself without a web search.
FASTPATH_WHY = {
    "causal_conv1d": "the fused causal conv1d used by linear-attention and Mamba blocks",
    "flash-linear-attention": "the fla kernels behind Qwen3.5-style linear attention",
    "fla-core": "the fla kernels behind Qwen3.5-style linear attention",
    "flash-attn": "flash attention 2/3 in place of the math attention path",
    "triton": "every Triton kernel, including Unsloth's own",
    "xformers": "memory-efficient attention on the paths that still use it",
    "torch": "everything; a version split here also splits every kernel below it",
    "transformers": "which fast paths the modelling code is even willing to look for",
    "trl": "the trainer, and which of its fast paths exist",
    "peft": "LoRA layer implementations",
    "accelerate": "dispatch and the memory it reserves",
    "bitsandbytes": "4-bit and 8-bit kernels",
}

# These drag the rest of the stack with them, so the report says so rather than pretending.
FASTPATH_HEAVY = ("torch", "triton")

# Run as a CHILD process on purpose: importing flash-attn or a mismatched causal_conv1d
# extension can abort the interpreter, and that must cost the package inventory nothing.
FASTPATH_RUNTIME_SOURCE = """
import json
g = {}


def gate(name, code):
    try:
        exec(code, {})
        g["gate_" + name] = True
    except BaseException as exc:
        g["gate_" + name] = "no (" + type(exc).__name__ + ")"


# The SAME import statements transformers uses, from the same module paths and in the same
# groupings. Sampling one symbol out of a group reported a matching gate on a node where the
# real import fails: modeling_qwen3_next.py takes `causal_conv1d_fn` and `causal_conv1d_update`
# in one try, so a missing `_update` leaves BOTH None and the block falls back, and it reaches
# the delta rule through `fla.ops.gated_delta_rule` rather than `fla.ops`, which is a different
# module and can exist when the other does not.
gate(
    "causal_conv1d_fn",
    "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update",
)
gate(
    "fla_chunk_gated_delta_rule",
    "from fla.ops.gated_delta_rule import chunk_gated_delta_rule, "
    "fused_recurrent_gated_delta_rule",
)
gate("fla_fused_rms_norm_gated", "from fla.modules import FusedRMSNormGated")
gate("flash_attn_func", "from flash_attn import flash_attn_func")
gate("triton", "import triton")
gate("xformers_memory_efficient_attention", "from xformers.ops import memory_efficient_attention")
try:
    import torch
    g["torch_version"] = torch.__version__
except BaseException as exc:
    g["torch_version"] = "ERR " + type(exc).__name__
try:
    from transformers.utils import import_utils as iu
    for fn in ("is_causal_conv1d_available", "is_flash_linear_attention_available",
               "is_flash_attn_2_available", "is_flash_attn_3_available"):
        f = getattr(iu, fn, None)
        # transformers 4.x does not define all of these. An absent gate is not a
        # finding: the modelling code there never consults it either.
        if f is None:
            continue
        try:
            g["tf_" + fn] = bool(f())
        except BaseException as exc:
            g["tf_" + fn] = "ERR " + type(exc).__name__
except BaseException as exc:
    g["tf_gates"] = "ERR " + type(exc).__name__
print("UNSLOTH_FASTPATH_GATES " + json.dumps(g, sort_keys=True))
"""


def fastpath_probe_source(runtime: bool = True) -> str:
    """The fast-path probe, as source, run identically on both nodes. The inventory half reads
    metadata only, so it is safe while a GPU is busy; the ``runtime`` half shells out to a
    child interpreter, the only way to tell an installed package apart from a usable one."""
    import base64

    lines = [
        "import json, socket",
        "from importlib.metadata import version, PackageNotFoundError",
        "r = {'host': socket.gethostname()}",
        "r['executable'] = __import__('sys').executable",
        f"for name, aliases in {FASTPATH_PACKAGES!r}:",
        "    found = None",
        "    for alias in aliases:",
        "        try:",
        "            found = version(alias)",
        "        except PackageNotFoundError:",
        "            continue",
        "        except Exception:",
        "            found = 'ERR'",
        "        break",
        "    r['pkg_' + name] = found",
    ]
    if runtime:
        blob = base64.b64encode(FASTPATH_RUNTIME_SOURCE.encode()).decode()
        lines += [
            "import base64, subprocess, sys",
            f"inner = base64.b64decode({blob!r}).decode()",
            "try:",
            "    p = subprocess.run([sys.executable, '-c', inner], capture_output=True,",
            "                       text=True, timeout=180)",
            "    tag = 'UNSLOTH_FASTPATH_GATES '",
            "    line = next((l for l in reversed(p.stdout.splitlines())",
            "                 if l.startswith(tag)), None)",
            "    if line is None:",
            "        r['gates_error'] = ((p.stderr or '').strip().splitlines()",
            "                            or ['no output'])[-1][:200]",
            "    else:",
            "        r.update(json.loads(line[len(tag):]))",
            "except Exception as exc:",
            "    r['gates_error'] = type(exc).__name__ + ': ' + str(exc)[:160]",
        ]
    lines.append("print('UNSLOTH_FASTPATH ' + json.dumps(r, sort_keys=True))")
    return "\n".join(lines) + "\n"


def compare_fastpath(local: dict, peer: dict) -> list:
    """Findings comparing two probe results; empty means the nodes agree. Each is
    ``{"kind", "name", "local", "peer", "lagging"}``, where ``lagging`` is None when neither
    side is obviously behind. Never raises: a probe that half-ran must still report."""
    findings = []
    for name, _aliases in FASTPATH_PACKAGES:
        key = "pkg_" + name
        mine, theirs = local.get(key), peer.get(key)
        # Absent on both is not a finding: neither workload asked for it.
        if mine == theirs or (mine is None and theirs is None):
            continue
        lagging = "local" if mine is None else ("peer" if theirs is None else None)
        findings.append(
            {"kind": "package", "name": name, "local": mine, "peer": theirs, "lagging": lagging}
        )

    # A node whose runtime probe did not report has no gates to compare, and calling that a
    # divergence would dress an UNKNOWN up as a finding.
    if local.get("gates_error") or peer.get("gates_error"):
        return findings

    gate_keys = sorted(
        k for k in set(local) | set(peer) if k.startswith("gate_") or k.startswith("tf_")
    )
    for key in gate_keys:
        mine, theirs = local.get(key), peer.get(key)
        if mine == theirs:
            continue
        lagging = "local" if mine is not True else ("peer" if theirs is not True else None)
        findings.append(
            {
                "kind": "gate",
                "name": key.split("_", 1)[1] if key.startswith("gate_") else key[3:],
                "local": mine,
                "peer": theirs,
                "lagging": lagging,
            }
        )
    return findings


# Gates worth mentioning when BOTH nodes lack them, because a pip install fixes them here.
# `is_flash_attn_3_available` is deliberately absent: FA3 cannot simply be installed on a
# Spark, so reporting it would be noise on every healthy pair.
FASTPATH_BOTH_SLOW_GATES = (
    "tf_is_causal_conv1d_available",
    "tf_is_flash_linear_attention_available",
    "tf_is_flash_attn_2_available",
)


def fastpath_both_slow(local: dict, peer: dict) -> list:
    """Fast paths that BOTH nodes lack, where transformers itself wanted one. A note, never a
    failure, since nothing is asymmetric. Only transformers' own gates are used: they are the
    one signal separating "this build looks for that kernel" from "nobody runs that model"."""
    return [
        key[3:]
        for key in FASTPATH_BOTH_SLOW_GATES
        if local.get(key) is False and peer.get(key) is False
    ]


def _ssh_login() -> str:
    try:
        from studio.spark_cluster import _ssh_user
        return _ssh_user()
    except Exception:
        import os
        return os.environ.get("USER") or os.environ.get("USERNAME") or "nvidia"


def _install_lines(node: str, peer_ip: str, spec: str) -> list:
    act = _activate_sh()
    if node == "local":
        return [f". {act}", f'python3 -m pip install "{spec}"']
    # These are pasted by a user. `act` is already one double-quoted word, and it is going
    # inside another double-quoted string, so its quotes are escaped like the spec's.
    inner = f'. {act.replace(chr(34), chr(92) + chr(34))}; python3 -m pip install \\"{spec}\\"'
    return [f'ssh {_ssh_login()}@{peer_ip} "{inner}"']


def check_fastpath(peer_ip: str, runtime: bool = True) -> int:
    source = fastpath_probe_source(runtime = runtime)
    typer.echo("")
    typer.echo("Cross-node fast-path parity")
    typer.echo("---------------------------")
    typer.echo(
        f"  comparing the fast-path stack on this Spark and {peer_ip}"
        + (" (imports the entry points)" if runtime else " (metadata only)")
        + " ..."
    )

    local, local_err = _run_probe_local(source, timeout = 300, marker = "UNSLOTH_FASTPATH")
    peer, peer_err = _run_probe_peer(peer_ip, source, timeout = 360, marker = "UNSLOTH_FASTPATH")

    if local is None or peer is None:
        typer.echo("")
        typer.echo(
            "  UNKNOWN -- the fast-path probe could not run on "
            + ("this Spark" if local is None else "the peer")
            + "."
        )
        typer.echo(f"    reason: {local_err if local is None else peer_err}")
        typer.echo("    Treat the pair as unequal until it can be checked. A missing fast")
        typer.echo("    path costs 2.19x throughput and 39% more memory and raises nothing,")
        typer.echo("    so an unrun check is not evidence that the two nodes match.")
        return 2

    findings = compare_fastpath(local, peer)
    notes = fastpath_both_slow(local, peer)
    unknown = [
        node
        for node, data in (("this Spark", local), ("the peer", peer))
        if data.get("gates_error")
    ]

    if not findings:
        typer.echo("")
        typer.echo(
            f"  OK -- {local.get('host', 'this Spark')} and {peer.get('host', peer_ip)} "
            "run the same fast-path stack."
        )
    else:
        typer.echo("")
        typer.echo(f"  {len(findings)} DIFFERENCE(S) -- the node that is behind will train and")
        typer.echo("  infer slower than the other one, silently, and finish without an error.")
        for f in findings:
            typer.echo("")
            label = f["name"] if f["kind"] == "package" else f"{f['name']} (runtime fast path)"
            typer.echo(f"    {label}")
            typer.echo(f"      this Spark : {_fmt(f['local'])}")
            typer.echo(f"      the peer   : {_fmt(f['peer'])}")
            why = FASTPATH_WHY.get(f["name"])
            if why:
                typer.echo(f"      gates      : {why}")
            if f["kind"] != "package" or f["lagging"] is None:
                continue
            behind, ahead = (
                ("local", f["peer"]) if f["lagging"] == "local" else ("peer", f["local"])
            )
            if ahead in (None, "ERR"):
                continue
            where = "this Spark" if behind == "local" else "the peer"
            typer.echo(f"      fix, on {where}:")
            for line in _install_lines(behind, peer_ip, f"{f['name']}=={ahead}"):
                typer.echo(f"        {line}")
            if f["name"] in FASTPATH_HEAVY:
                typer.echo(
                    "      (this one pulls the rest of the stack with it -- do it deliberately.)"
                )

    for note in notes:
        typer.echo("")
        typer.echo(f"  NOTE -- transformers reports {note} on BOTH nodes.")
        typer.echo("    The nodes agree, so nothing here is asymmetric, but the pair is")
        typer.echo("    taking the slow path together. Installing the package on both is")
        typer.echo("    worth 2x on the model families that use it.")

    if unknown:
        typer.echo("")
        typer.echo(
            f"  UNKNOWN -- the runtime fast-path probe did not report on {', '.join(unknown)}."
        )
        for node, data in (("this Spark", local), ("the peer", peer)):
            if data.get("gates_error"):
                typer.echo(f"    {node}: {_fmt(data['gates_error'])}")
        typer.echo("    Package versions above were still compared; what could not be")
        typer.echo("    checked is whether the kernels actually import.")

    if findings:
        typer.echo("")
        typer.echo("  Why this is not cosmetic: measured on this pair, a node missing")
        typer.echo("  causal_conv1d and flash-linear-attention ran Qwen3.5 LoRA training at")
        typer.echo("  1183 tok/s against the other node's 2593 -- 2.19x -- and used 14.02 GiB")
        typer.echo("  against 10.07 GiB, on a byte-identical cell with identical clocks, bf16")
        typer.echo("  throughput, copy bandwidth and NCCL. The only sign was one line of")
        typer.echo('  warning: "The fast path is not available".')
        typer.echo("")
        typer.echo("  On a multi-node run the pair moves at the slower node's pace, so this")
        typer.echo("  halves the whole job, not just one rank.")
        return 1
    return 2 if unknown else 0


def _workload_guidance() -> None:
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
    skip_fastpath: bool = typer.Option(
        False,
        "--skip-fastpath",
        help = "Do not compare the two nodes' fast-path packages and kernels.",
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
      it deadlocks the job for 1800 s in a library that has nothing to do with the cause;
    * a fast-path package present on one node and missing on the other, which deadlocks
      nothing and raises nothing: the pair simply runs at the slower node's pace. Measured
      here at 2.19x on Qwen3.5 LoRA training, with 39% more peak memory.
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

    # Inert without a peer: this check has nothing to say about one machine's package list.
    fastpath_rc = 0
    if peer_ip and not skip_fastpath:
        # `--parity-only` promises no GPU work, and the runtime half of this probe imports
        # torch and the native kernel packages on both nodes, which is minutes and a CUDA
        # context. The package comparison still runs -- that IS a capability gate -- and
        # `--deep` is the flag that asks for the imports.
        fastpath_rc = check_fastpath(peer_ip, runtime = deep or not parity_only)

    if parity_only:
        _workload_guidance()
        raise typer.Exit(1 if (parity_rc or fastpath_rc) else 0)

    link_rc = spark_cluster.main(["doctor"])
    _workload_guidance()
    raise typer.Exit(link_rc or (1 if (parity_rc or fastpath_rc) else 0))
