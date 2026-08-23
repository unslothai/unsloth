# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Build the Kaggle kernel notebook(s) that run the T4 legs.

Two layers, because a Kaggle GPU session is 2xT4 and each payload is written
for one card:

**Payload notebook** -- one per GPU, one per *leg* (see ``legs.py``).
Materialises its sources from inlined copies, installs that leg's library set,
probes its imports, runs the leg's entry script, prints a machine-readable
result line.

**Driver notebook** -- one per kernel. Carries its payloads inline (gzip +
base64) so the kernel needs no dataset attachment and no network fetch of our
sources, gives each payload its own virtualenv and GPU, and runs them
concurrently under papermill. Nothing is checked out on the Kaggle side.

Cell order is load-bearing: **materialise, install, verify, run**. The control
leg installs from a pin file carried inside the notebook, so materialising last
(as an earlier version did) wrote that file after the install needing it.

Four details that are NOT safe to simplify away:

1. **Per-child virtualenv.** The legs deliberately pip-install DIFFERENT
   torch/transformers stacks. One shared site-packages destroys the
   experiment: control's pins and canary's upgrades land in the same tree and
   the last writer wins.
2. **``uv venv --seed``.** Without it the venv has no pip, so ``!pip install``
   falls through PATH to the system pip while the kernel runs the venv
   interpreter: installs and imports target different site-packages.
3. **``UV_SYSTEM_PYTHON=0``.** The Kaggle image ships ``UV_SYSTEM_PYTHON=1``,
   which BEATS ``VIRTUAL_ENV``: ``uv pip install`` writes to the base image
   while ``--system-site-packages`` lets the kernel import from there anyway,
   so both children silently share one tree.
4. **Runtime paths are built in Python, from ROOT.** A shell-shaped fragment
   spliced into a generated cell is a SyntaxError waiting for a Kaggle session,
   so ``@ROOT/`` arguments expand into ``str(ROOT / ...)`` expressions.
   ``test_generated_cells_compile`` is the cheap way to find that out.

Usage:
    python build_kernel.py --payload-dir tests/kaggle/t4_smoke \\
        --out kernel1.ipynb --legs control,canary
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from legs import (  # noqa: E402
    KERNELS,
    PACKAGE_UNDER_TEST,
    PREFETCH_REPOS,
    Leg,
    expand_install,
    resolve,
)

DRIVER_SENTINEL = "KAGGLE_T4_CI_DRIVER"
PAYLOAD_SENTINEL = "KAGGLE_T4_CI_PAYLOAD"
RESULT_PREFIX = "T4_SMOKE_REPORT "

# Where the payload sources land on the Kaggle side, one directory PER LEG: a
# kernel's payloads run concurrently with byte-identical copies of the same
# files, and `write_bytes` truncates first, so one shared directory lets a
# payload empty a file the other is importing.
KERNEL_ROOT = "/kaggle/working/t4_smoke_src"

# A Kaggle GPU session is 2xT4. This is the width the packing is built for and
# what the kernel stands down against when the allocation is short; it is not
# a count of legs, which is now larger than it on purpose.
SESSION_GPUS = 2


def _kernel_root(leg: Leg) -> str:
    return f"{KERNEL_ROOT}_{leg.name}"


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data)).decode("ascii")


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends = True),
    }


def _shared_args_for(leg: Leg, extra_args: tuple[str, ...]) -> list[str]:
    """``--smoke-args``, minus any option the leg already sets for itself.

    Shared arguments keep the control and canary legs differing in nothing but
    versions, and are appended AFTER each leg's own, so for a leg naming the
    same option they override it (argparse takes the last value): the gpt-oss
    leg asks for 3 steps, a measured fit for a 16GB card, and the workflow's
    ``--max-steps 10`` silently retrained the 20B model for ten.
    """
    own = {a.split("=", 1)[0] for a in leg.args if a.startswith("--")}
    kept: list[str] = []
    drop_value = False
    for token in extra_args:
        if token.startswith("--"):
            if token.split("=", 1)[0] in own:
                # `--opt value` has to lose its value token as well.
                drop_value = "=" not in token
                continue
            drop_value = False
            kept.append(token)
        elif drop_value:
            drop_value = False
        else:
            kept.append(token)
    return kept


def _arg_expression(value: str) -> str:
    """One entry of the child's argv, as a Python expression.

    ``@ROOT/x/y`` becomes ``str(ROOT / "x" / "y")`` so the path is assembled on
    the kernel from its own ROOT; everything else is a string literal.
    Interpolating the path into the generated source instead produced a cell
    reading ``"--label", "gpu0" --reference "{ROOT}/..."`` that died with a
    SyntaxError before a single training step ran.
    """
    if value.startswith("@ROOT/"):
        parts = [p for p in value[len("@ROOT/") :].split("/") if p]
        joined = " / ".join(["ROOT"] + [json.dumps(p) for p in parts])
        return f"str({joined})"
    return json.dumps(value)


def build_payload_notebook(
    payload_dir: Path,
    leg: Leg,
    *,
    unsloth_ref: str,
    zoo_ref: str,
    extra_args: tuple[str, ...] = (),
    reference: str | None = None,
) -> dict:
    """A single-GPU notebook that installs one leg's stack and runs it."""
    root = _kernel_root(leg)
    wanted = list(leg.files)
    if leg.entry not in wanted:
        wanted.append(leg.entry)

    # `reference=None` means "whatever the leg asks for"; an explicit "" turns
    # the band check off, which is how a reference recapture is dispatched.
    ref_name = leg.reference if reference is None else reference
    if ref_name:
        wanted.append(f"references/{ref_name}")

    files = {}
    for name in wanted:
        path = payload_dir / name
        if not path.exists():
            raise FileNotFoundError(f"leg {leg.name!r} needs {name}, which is not in {payload_dir}")
        files[name] = _encode_bytes(path.read_bytes())

    materialise = f"""# Materialise the test sources carried inside this notebook.
#
# FIRST, before any install: the control leg installs from a pin file that
# travels in here, so the files have to be on disk before the first pip call.
import base64, gzip, json, os, pathlib
FILES = {json.dumps(files)}
ROOT = pathlib.Path({json.dumps(root)})
for name, blob in FILES.items():
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.decompress(base64.b64decode(blob)))
print("{PAYLOAD_SENTINEL} sources " + json.dumps(sorted(FILES)), flush=True)
"""

    groups = expand_install(leg, unsloth_ref = unsloth_ref, zoo_ref = zoo_ref, payload_dir = payload_dir)
    install = f"""# Install this leg's library set.
#
# The groups below are generated from legs.py and are the ONLY thing that
# differs between the control leg and the version canary. They are printed
# before they run so the kernel log alone says what was asked for, which is
# what makes a canary failure attributable without downloading anything.
import json, subprocess, sys, time
print("{PAYLOAD_SENTINEL} leg {leg.name}: {leg.summary}", flush=True)
GROUPS = {json.dumps(groups)}
print("{PAYLOAD_SENTINEL} install plan " + json.dumps(GROUPS), flush=True)

def pip(args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("  $ " + " ".join(cmd[3:]), flush=True)
    # github.com occasionally 500s on a git fetch, and PyPI occasionally
    # times out; a single upstream blip must not be reported as a notebook
    # regression. The backoff is what lets the third failure be read as a
    # resolution that is genuinely impossible rather than as one bad minute:
    # three immediate retries all land inside the same outage.
    for attempt in (1, 2, 3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        print(f"  install attempt {{attempt}} failed rc={{proc.returncode}}",
              flush=True)
        print("  " + proc.stderr.strip()[-1500:], flush=True)
        if attempt < 3:
            time.sleep(15 * attempt)
            continue
        print("{PAYLOAD_SENTINEL} INSTALL FAILED " + json.dumps(list(args)),
              flush=True)
        # A VERDICT, not missing evidence, for the same reason the dependency
        # probe below writes one: the launcher classifies a leg that reported
        # NOTHING as `partial` or `infra` and the workflow stays green. So a
        # commit whose distribution cannot be resolved -- conflicting
        # requirements in pyproject.toml, a dropped dependency -- used to pass
        # the one job added to test its packaging metadata, by failing early
        # enough that nothing was ever reported.
        print("{RESULT_PREFIX}" + json.dumps({{
            "label": {json.dumps(leg.name)},
            "model": "install",
            "passed": False,
            "failures": ["pip install failed after 3 attempts: "
                         + " ".join(args) + " -- rc=" + str(proc.returncode)
                         + ": " + proc.stderr.strip()[-500:]],
        }}), flush=True)
        raise SystemExit(f"pip install failed: {{args}}")

for group in GROUPS:
    pip(group)
print("{PAYLOAD_SENTINEL} install done", flush=True)
"""

    verify = f"""# Fail fast, and fail legibly.
#
# Without this, a missing dependency surfaces as a traceback buried in a
# child process's captured stdout, forty minutes and one GPU session later.
# Here it surfaces immediately, named, in the driver log.
import importlib, json, subprocess, sys
# Everything above was pip-installed AFTER this interpreter started, and the
# import system caches the directory listing of each sys.path entry. Without
# this, a just-installed package can be invisible to the very next import.
importlib.invalidate_caches()

# The resolved version of every package this CI watches, read from the
# installed distributions rather than from module attributes. This is the
# line a reader diffs between the control leg and the canary leg to name
# what moved, so it is printed before anything can crash.
sys.path.insert(0, {json.dumps(root)})
# Kept in a name rather than printed and discarded: report.version_table
# builds the per-leg comparison out of the REPORTS, not out of the log, so a
# leg whose report omits this is missing from the one table that says which
# release differed from the healthy control. Wrapped because a half-installed
# distribution can make importlib.metadata raise, and losing the report to a
# diagnostic would put this leg back to reporting nothing at all.
try:
    import versions
    RESOLVED = versions.flatten_versions(versions.resolved_versions())
except Exception as exc:
    RESOLVED = {{"error": f"{{type(exc).__name__}}: {{exc}}"[:200]}}
print("{PAYLOAD_SENTINEL} resolved " + json.dumps(RESOLVED), flush=True)

missing = []
# ORDER IS PART OF THE TEST, not cosmetic. `unsloth` comes before
# `unsloth_zoo`: zoo's __init__ ends with
#     if find_spec("unsloth") is None:
#         raise ImportError("Please install Unsloth via `pip install unsloth`!")
# and on a real T4 that fired on a session where unsloth WAS installed and
# imported cleanly one line later -- because by then `unsloth` was in
# sys.modules and find_spec answered from there. Probing zoo first therefore
# reported a missing dependency that was not missing, and aborted the payload
# before a single training step ran. Importing unsloth first is also the
# order every Unsloth notebook uses.
for mod in {json.dumps(list(leg.imports))}:
    try:
        importlib.import_module(mod)
    # BaseException, for the reason the GPU probe below gives: an import that
    # ends in `sys.exit()` raises SystemExit, which is NOT an Exception, and a
    # package refusing an accelerator or a version at import time is exactly
    # how that happens. Uncaught it aborts this cell before the report below
    # is written, the run cell never runs, and a leg that reported nothing is
    # `partial` or `infra` at the launcher -- both green. KeyboardInterrupt is
    # re-raised: that is the runner cancelling the job, not a bad dependency.
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        missing.append(f"{{mod}}: {{type(exc).__name__}}: {{exc}}")
if missing:
    print("{PAYLOAD_SENTINEL} MISSING " + json.dumps(missing), flush=True)
    # A probe failure is a VERDICT, not missing evidence, and the two exit
    # differently: the launcher extracts reports, and with none of them it
    # calls the whole run `infra` and the workflow stays green. So a commit
    # that breaks `import unsloth` -- a syntax error, a dropped runtime
    # dependency -- used to pass this job by failing early enough. The run
    # cell below is the only other thing that emits a report and it is never
    # reached from here, so the report is written HERE instead.
    print("{RESULT_PREFIX}" + json.dumps({{
        "label": {json.dumps(leg.name)},
        "model": "dependency probe",
        "passed": False,
        "versions_flat": RESOLVED,
        "failures": ["import failed -- " + m for m in missing],
    }}), flush=True)
    raise SystemExit("payload dependencies incomplete: " + "; ".join(missing))

# The GPU check, under the same rule as the import probe above: whatever
# goes wrong in here is a VERDICT and has to leave a report behind, because
# nothing further down this notebook ever runs to write one.
#
# The case that made it necessary: the driver shows the T4s to `nvidia-smi`,
# but dependency resolution picked a CPU-only or CUDA-incompatible torch
# wheel, so `device_count()` is 0. The bare assert aborted the cell, the
# launcher extracted no report for this leg, and no report is `infra` (or
# `partial` next to its partner) -- both of which exit 0. A regression that
# makes CUDA unusable was therefore invisible.
#
# `except BaseException`, and every line of the check inside it: the count
# is only one of the ways this fails. `import torch` can raise on a broken
# wheel, and get_device_properties can raise where device_count did not.
try:
    import torch
    print("{PAYLOAD_SENTINEL} gpu " + json.dumps({{
        "count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "visible": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }}), flush=True)
    # Exactly one GPU must be visible: two would mean the driver failed to pin
    # this payload to its own card, and accelerate would shard across both,
    # which is a different test from the single T4 a Colab user gets. Zero
    # means the payload cannot run at all on the hardware it was billed for.
    assert torch.cuda.device_count() == 1, (
        f"expected exactly 1 visible GPU, got {{torch.cuda.device_count()}}")
except BaseException as exc:
    detail = f"{{type(exc).__name__}}: {{exc}}"
    print("{PAYLOAD_SENTINEL} GPU_UNUSABLE " + json.dumps(detail), flush=True)
    print("{RESULT_PREFIX}" + json.dumps({{
        "label": {json.dumps(leg.name)},
        "model": "gpu probe",
        "passed": False,
        "versions_flat": RESOLVED,
        "failures": ["the payload could not use its GPU -- " + detail],
    }}), flush=True)
    raise

# Is what the distribution under test DECLARES it needs actually satisfied?
#
# The import probe above asks the weaker question: a requirement reached only
# by a delayed code path is absent all through a green run. pyproject.toml is
# in this workflow's trigger paths, so a commit that drops a requirement, adds
# one this image cannot satisfy, or tightens a bound past what is installed
# arrives here to be tested, and `pip install unsloth` is where a user meets
# it. LAST of the three checks because it is the only one that says nothing
# about whether this session can run: a card it cannot use is the more urgent
# verdict, and the GPU probe re-raises before reaching this.
#
# `pip check` is pip's own answer rather than a re-implementation of version
# comparison, and ONLY the lines owned by the distribution under test are read.
# The Kaggle image carries pre-existing conflicts of its own, and the frontier
# leg deliberately installs a transformers that unsloth_zoo's metadata forbids;
# both are other packages' lines and neither is this leg's verdict.
OWNER = {json.dumps(PACKAGE_UNDER_TEST)}
def _owned(line):
    head = line.strip().split(" ")[0].lower().replace("_", "-")
    return head == OWNER.lower().replace("_", "-")
_check = subprocess.run([sys.executable, "-m", "pip", "check"],
                        capture_output=True, text=True)
unsatisfied = [ln.strip() for ln in (_check.stdout + _check.stderr).splitlines()
               if _owned(ln)]
if unsatisfied:
    print("{PAYLOAD_SENTINEL} REQUIREMENTS_UNSATISFIED "
          + json.dumps(unsatisfied), flush=True)
    # A VERDICT, for the third time in this cell and for the same reason: the
    # run cell below is the only other thing that writes a report, it is never
    # reached from here, and a leg that reported nothing leaves the workflow
    # green.
    print("{RESULT_PREFIX}" + json.dumps({{
        "label": {json.dumps(leg.name)},
        "model": "requirements",
        "passed": False,
        "versions_flat": RESOLVED,
        "failures": ["declared requirement unsatisfied -- " + u
                     for u in unsatisfied],
    }}), flush=True)
    raise SystemExit("declared requirements unsatisfied: "
                     + "; ".join(unsatisfied))
"""

    argv = list(leg.args) + _shared_args_for(leg, tuple(extra_args))
    if ref_name:
        argv += ["--reference", f"@ROOT/references/{ref_name}"]
    arg_exprs = ", ".join(_arg_expression(a) for a in argv)

    run = f"""# Run the leg in a child process.
#
# A child, not an import, for two reasons: the determinism setup has to run
# before torch is imported (and papermill has already imported plenty), and
# a hard crash then leaves this cell alive to report it.
import json, os, pathlib, subprocess, sys
ROOT = pathlib.Path({json.dumps(root)})
OUT = pathlib.Path("/kaggle/working/t4_out_{leg.name}")
OUT.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["UNSLOTH_DISABLE_STATISTICS"] = "1"
env.update({json.dumps(leg.env)})

cmd = [sys.executable, str(ROOT / {json.dumps(leg.entry)}),
       "--outdir", str(OUT), "--label", {json.dumps(leg.name)}]
cmd += [{arg_exprs}]
print("{PAYLOAD_SENTINEL} exec " + " ".join(cmd), flush=True)

# errors="replace", because the alternative is losing the verdict to the
# output. text=True decodes strictly, and a payload that dies in native code
# writes whatever bytes the crash handler had; one of them not being UTF-8
# raised UnicodeDecodeError HERE, before the synthetic report below, so
# papermill aborted the cell and the launcher, finding no report for this leg,
# called the run partial or infra. Both are green, on a leg that died.
proc = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
print(proc.stdout[-40000:], flush=True)
if proc.stderr.strip():
    print("----- stderr (tail) -----", flush=True)
    print(proc.stderr[-20000:], flush=True)

print("{PAYLOAD_SENTINEL} returncode " + str(proc.returncode), flush=True)

# Re-emit the report on its own line so the driver log alone is enough to
# judge the run, even if artifact collection fails entirely. Parsed and
# re-serialized COMPACTLY rather than echoed: every payload writes this file
# indented, the launcher scans whole lines for the prefix, and echoing the
# indented text verbatim therefore hands it a lone `{{` to decode. The
# recovery path meant for the case where the payload's own compact line fell
# out of the retained stdout tail was thus the one case it could not recover.
report = None
report_path = OUT / "t4_smoke_report.json"
if report_path.exists():
    try:
        report = json.loads(report_path.read_text())
    except Exception as exc:
        print("{PAYLOAD_SENTINEL} REPORT UNREADABLE " + repr(exc)[:300], flush=True)
        report = None

if isinstance(report, dict):
    print("{RESULT_PREFIX}" + json.dumps(report), flush=True)
else:
    # A VERDICT, not missing evidence, and the distinction decides whether
    # this job can go red at all. The child ran; a nonzero exit with no
    # readable report is a CUDA segfault, a native abort or an OOM kill, and
    # the definitive exit status for it is in hand right here. Printing only
    # "NO REPORT WRITTEN" left the launcher with nothing to extract, and no
    # reports at all is `infra` while a partner report is `partial` -- both
    # green. So the crash is reported as the failure it is.
    print("{PAYLOAD_SENTINEL} NO USABLE REPORT WRITTEN rc=" + str(proc.returncode), flush=True)
    print("{RESULT_PREFIX}" + json.dumps({{
        "label": {json.dumps(leg.name)},
        "model": "payload process",
        "passed": False,
        "returncode": proc.returncode,
        "failures": [
            "the payload process exited " + str(proc.returncode) + " and left no "
            "readable t4_smoke_report.json, so it died before it could judge "
            "itself. Its own report is the only thing that could have made this "
            "leg green.",
        ],
        "stderr_tail": proc.stderr[-2000:],
    }}), flush=True)

print("{PAYLOAD_SENTINEL} complete rc=" + str(proc.returncode), flush=True)
# Deliberately does NOT raise. A failing payload must not abort its partner
# on the other T4 and cost the whole session.
"""

    return {
        "cells": [
            _code_cell(materialise),
            _code_cell(install),
            _code_cell(verify),
            _code_cell(run),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle_t4_ci_leg": leg.name,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_driver(
    payloads: dict[str, dict],
    per_run_timeout: int,
    isolation: dict[str, bool] | None = None,
    expected_gpus: int = SESSION_GPUS,
    cpu_lane: str | None = None,
    after_gpu: str | None = None,
    prefetch_repos: tuple[str, ...] = (),
    vram_source: dict | None = None,
) -> dict:
    """Kernel notebook that runs the payloads across the session's GPUs.

    ``isolation`` maps a payload to whether its virtualenv may see the Kaggle
    image's site-packages. Per payload, not per kernel, because legs sharing a
    kernel do not share an answer: see ``Leg.system_site_packages``.

    ``payloads`` may hold MORE entries than there are cards. They queue: one
    worker per card takes the next leg only when its current one has exited, so
    a card carries exactly one payload at a time. ``payloads`` is ordered, and
    that order is the start order. It comes from ``legs.KERNELS``, which
    documents why gptoss sits third rather than first; do not sort it here.

    ``prefetch_repos`` are warmed, in the order given, on a lane that takes no
    card and no virtualenv. It runs in the driver's own interpreter using the
    Kaggle image's ``huggingface_hub``, so it starts at t=0 rather than after a
    venv build, and it deliberately does NOT set ``HF_HOME``: the legs read the
    image default and the whole point is to land in the cache they read.

    ``expected_gpus`` is how many cards the packing was built for. It is what
    the kernel stands down against, and it is deliberately NOT ``len(payloads)``
    any more: see the guard in the generated cell.
    """
    vram_source = vram_source or {}
    encoded = {name: _encode_bytes(json.dumps(nb).encode("utf-8")) for name, nb in payloads.items()}
    isolation = isolation or {}
    system_site = {name: bool(isolation.get(name, True)) for name in payloads}
    # The CARD queue, which is not every payload. cpu_lane runs beside the
    # cards and after_gpu runs once they are free; leaving either in here would
    # hand it a card and defeat the point of both.
    off_queue = {name for name in (cpu_lane, after_gpu) if name}
    for name in off_queue:
        if name not in payloads:
            raise KeyError(f"{name!r} is scheduled off the card queue but is not a payload")
    order = [name for name in payloads if name not in off_queue]
    if not order:
        raise ValueError("every payload is off the card queue, so nothing would use a GPU")
    # hf_home is left unset ON PURPOSE. See build_driver's docstring: a private
    # root here downloads 12 GB into a cache no leg reads and reports success.
    # Per-payload VRAM for the admission check. Off-queue payloads are absent
    # on purpose: neither takes a card, so neither has a budget to consume.
    vram_gb = {name: float(getattr(leg, "vram_gb", 1.0)) for name, leg in vram_source.items()}
    prefetch_blob = ""
    if prefetch_repos:
        prefetch_blob = _encode_bytes(
            _prefetch_builder().prefetch_cell(list(prefetch_repos)).encode("utf-8")
        )

    setup = f"""import base64, gzip, json, os, pathlib, subprocess, sys, threading, time
print("{DRIVER_SENTINEL} start", flush=True)

WORK = pathlib.Path("/kaggle/working")

# Virtualenvs do NOT live under WORK. /kaggle/working is 19.5 GB and is also
# what Kaggle ships home as the artifact; $HOME and /tmp share a ~1 TB overlay.
# Each leg's venv carries its own torch and its own NVIDIA runtime, and with
# two legs per card there can now be FOUR of them alive at once -- which does
# not fit in 19.5 GB and would surface as an install failing halfway through
# for reasons that look nothing like a disk.
#
# Evidence stays on WORK, because that is the only thing that has to come back.
VENV_ROOT = pathlib.Path("/tmp/t4ci_venvs")
try:
    VENV_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    # A box without a writable /tmp is not one this kernel can fix. Falling
    # back to WORK keeps a one-leg-per-card run working, which is the shape
    # that fitted there before co-scheduling.
    VENV_ROOT = WORK
print("{DRIVER_SENTINEL}_VENV_ROOT " + json.dumps({{"path": str(VENV_ROOT)}}), flush=True)
PAYLOADS = {json.dumps(encoded)}
# Which payloads may see the Kaggle image's site-packages. A leg that
# replaces torch must not: pip would then treat torch's pinned NVIDIA
# runtime packages as satisfied by the image's copies and build an
# environment that installs cleanly and cannot be imported.
# repr, not json.dumps: JSON writes `true` and Python wants `True`, and a
# generated cell with a bare `true` in it parses and then dies with a
# NameError on a Kaggle session. test_no_generated_cell_reads_a_name_nothing
# _defines is what caught that, which is the whole reason it exists.
SYSTEM_SITE = {system_site!r}

# How many GPUs did we actually get? Assert rather than assume: a 1-GPU
# allocation would silently serialise everything onto device 0 and double
# every runtime while looking like a slow but healthy run.
try:
    _smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=60)
    GPUS = [l for l in _smi.stdout.strip().splitlines() if l.strip()]
except Exception:
    GPUS = []
print("{DRIVER_SENTINEL}_GPUS " + json.dumps(GPUS), flush=True)
N_GPU = len(GPUS)
# The order payloads are STARTED in, longest expected leg first. Not
# `sorted(PAYLOADS)`: alphabetical puts the longest leg (gptoss) last, and a
# greedy scheduler that picks up the longest leg last cannot balance around
# it -- it is the leg that sets the makespan. Declaration order comes from
# legs.KERNELS, so the packing decision lives beside the legs it packs.
ORDER = {order!r}

# Payloads that are NOT in the card queue.
#
# CPU_LANE runs from t=0 alongside the training legs and never takes a card.
# AFTER_GPU runs once the queue has drained and wants every card. Both are None
# on an ordinary notebook-only kernel, which leaves the schedule exactly as it
# was.
CPU_LANE = {cpu_lane!r}
AFTER_GPU = {after_gpu!r}

# Measured peak_reserved_gb per payload (legs.Leg.vram_gb), and the budget a
# single card will admit. A T4 reports 14.56 GB usable; the budget is 13.0 so
# that fragmentation and the allocator's own headroom are not the thing that
# discovers the limit.
#
# This is what lets two legs share a card. On run 32611343797 the three Qwen
# legs peaked at 0.70 GB EACH -- 4.8% of the card -- while gptoss peaked at
# 12.78 GB, so "one payload per card" was leaving a card 95% empty for three
# quarters of the run. gptoss still never shares: 12.78 + 0.70 = 13.48 is over
# budget, which is the arithmetic rather than a special case.
#
# MAX_LEGS_PER_CARD caps it at 2 even where VRAM would allow 3, because VRAM is
# not the scarce resource here. A Kaggle session has 4 vCPUs and a leg is ~87%
# install, import and download; the legs contend for CORES long before they
# contend for memory, and a third concurrent install buys nothing the profile
# says is available.
VRAM_GB = {vram_gb!r}
CARD_VRAM_BUDGET_GB = 13.0
MAX_LEGS_PER_CARD = 2

# The model prefetch, gzip+base64 like the payloads so its quoting survives
# being embedded in a generated cell. Empty string means no prefetch, which is
# what every kernel did before this existed.
PREFETCH_BLOB = {prefetch_blob!r}

# A shortfall is INFRASTRUCTURE, and it has to be called that HERE, before a
# thread starts. `max(1, ...)` used to make one card look like enough: both
# payloads were pinned to device 0, each child still saw exactly one GPU and
# passed its own visibility assertion, and the contention came back as an OOM
# that read like a code failure. Standing the kernel down instead produces no
# report, which the launcher already classifies as infra rather than red.
#
# Compared against EXPECTED_GPUS, not against the payload count. There are now
# more payloads than cards ON PURPOSE -- they queue, one card at a time -- so
# `N_GPU < len(PAYLOADS)` would stand down every healthy run. What still has to
# be caught is the allocation that hands us fewer CARDS than the packing was
# built for, which silently serialises the whole kernel onto device 0 and
# doubles its wall clock while looking like a slow but healthy run.
EXPECTED_GPUS = {expected_gpus}
if N_GPU < EXPECTED_GPUS:
    print("{DRIVER_SENTINEL}_INFRA " + json.dumps(
        {{"reason": "gpu_shortfall", "gpus": N_GPU, "payloads": len(PAYLOADS),
          "expected_gpus": EXPECTED_GPUS}}),
        flush=True)
    raise SystemExit(
        f"this kernel packs {{len(PAYLOADS)}} payload(s) across "
        f"{{EXPECTED_GPUS}} T4(s); the allocation exposed {{N_GPU}}")

for name, blob in PAYLOADS.items():
    (WORK / name).write_bytes(gzip.decompress(base64.b64decode(blob)))
print("{DRIVER_SENTINEL}_PAYLOADS " + json.dumps(sorted(PAYLOADS)), flush=True)
"""

    runner = f'''results = {{}}
lock = threading.Lock()

def _make_venv(idx, system_site):
    """Give a child its own site-packages. See this file's module docstring."""
    vdir = VENV_ROOT / f"venv_{{idx}}"
    try:
        uv = subprocess.run(["which", "uv"], capture_output=True,
                            text=True).stdout.strip()
        if not uv:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                           check=True, timeout=900)
            uv = "uv"
        venv_cmd = [uv, "venv", str(vdir), "--seed"]
        if system_site:
            venv_cmd.append("--system-site-packages")
        subprocess.run(venv_cmd, check=True, timeout=900)
        py = vdir / "bin" / "python"
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py),
                        "ipykernel"], check=True, timeout=900)
        kname = f"t4ci{{idx}}"
        subprocess.run([str(py), "-m", "ipykernel", "install", "--user",
                        "--name", kname], check=True, timeout=600)
        print(f"{DRIVER_SENTINEL}_VENV " + json.dumps(
            {{"idx": idx, "kernel": kname, "system_site": system_site}}),
              flush=True)
        return str(py), kname, str(vdir / "bin")
    except Exception as exc:
        print(f"{DRIVER_SENTINEL}_VENV_FAIL " + json.dumps(
            {{"idx": idx, "error": f"{{type(exc).__name__}}: {{exc}}"}}), flush=True)
        return None

def run_one(name, gpu_index, idx):
    src = WORK / name
    out = WORK / (pathlib.Path(name).stem + "_output.ipynb")
    env = dict(os.environ)
    # One GPU per payload. This is what makes the run comparable to the
    # single T4 a Colab user gets; two visible GPUs is a different test.
    #
    # gpu_index None means DO NOT PIN, and it is not the same as pinning to
    # nothing. It is used by the Studio lanes: the install half must still see
    # both cards, because install.sh --local resolves torch and a CPU-only
    # torch resolved by an installer that could not find a device is the exact
    # regression Studio's verify cell exists to catch; and the test half wants
    # both because Studio's own device selection is part of what it tests, so
    # masking one would test a machine nobody has. Blanking the variable would
    # do the opposite of both.
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    # One compile cache per payload. `unsloth_compiled_cache` is a RELATIVE
    # path resolved against the working directory, which both papermill
    # children inherit, so the legs -- which exist to install different
    # transformers/TRL versions and then compile the same modules -- would
    # otherwise overwrite and import each other's generated files. Set here
    # rather than in the payload so it also covers the notebook's own early
    # `import unsloth` in the verify cell.
    env["UNSLOTH_COMPILE_LOCATION"] = str(
        WORK / ("unsloth_compiled_cache_" + pathlib.Path(name).stem))

    made = _make_venv(idx, SYSTEM_SITE.get(name, True))
    if not made:
        # No venv means the SHARED system site-packages, which is the one
        # thing this driver exists to prevent: the legs' deliberately
        # different library sets would land in one tree, the last writer
        # would win, and the resulting import error would be reported as a
        # regression in the code under test. Skip the payload instead. It
        # produces no report, which the launcher classifies as infra.
        with lock:
            results[name] = {{"returncode": None, "gpu": gpu_index, "seconds": 0.0,
                              "error": "virtualenv creation failed; payload skipped "
                                       "rather than run in the shared system kernel",
                              "kernel": None, "output_exists": False}}
        print(f"{DRIVER_SENTINEL}_SKIPPED " + json.dumps({{name: results[name]}}),
              flush=True)
        return
    py, kernel, bindir = made
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(pathlib.Path(bindir).parent)
    env["UV_SYSTEM_PYTHON"] = "0"

    log = WORK / (pathlib.Path(name).stem + "_driver.log")
    started = time.time()
    rc, err = None, ""
    try:
        with open(log, "wb") as fh:
            proc = subprocess.run(
                [sys.executable, "-m", "papermill", str(src), str(out),
                 "-k", kernel, "--log-output", "--no-progress-bar"],
                env=env, stdout=fh, stderr=subprocess.STDOUT,
                timeout={per_run_timeout})
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc, err = -9, "papermill timed out after {per_run_timeout}s"
    except Exception as exc:
        rc, err = -1, f"{{type(exc).__name__}}: {{exc}}"
    with lock:
        results[name] = {{"returncode": rc, "gpu": gpu_index,
                          "seconds": round(time.time() - started, 1),
                          "error": err, "kernel": kernel,
                          "output_exists": out.exists()}}
    # Drop this leg's venv NOW, not in the tail cell. The tail prunes venv_*
    # only after every payload has finished, which was fine when a kernel held
    # one payload per card and they all ran at once. Packed, that would leave
    # every leg's tree on /kaggle/working simultaneously -- each one carries
    # its own torch and its own NVIDIA runtime -- and the disk is not sized for
    # that. Freeing here bounds the live count to one per card, which is what
    # it was before the packing. The papermill output notebook and the child
    # log are the evidence and they are NOT touched.
    try:
        import shutil as _sh
        _sh.rmtree(VENV_ROOT / f"venv_{{idx}}", ignore_errors=True)
    except Exception:
        pass
    print(f"{DRIVER_SENTINEL}_DONE " + json.dumps({{name: results[name]}}),
          flush=True)

# One worker per CARD, each pulling the next leg off a shared queue, rather
# than one thread per PAYLOAD. With more payloads than cards the old
# `gpu_index = i % N_GPU` handed the same card to two payloads and started
# them both at once: each child still passed its own single-visible-GPU
# assertion, and the two of them then fought over 15GB of VRAM. A queue makes
# a card take its next leg only once the previous one has exited, so a card
# holds exactly one payload at any instant no matter how many are packed.
#
# The index passed as `idx` is the payload's position in ORDER, not the
# worker's: it names the venv (venv_{{idx}}) and the ipykernel spec
# (t4ci{{idx}}), which must stay distinct per PAYLOAD or two legs would share
# an interpreter and the differing library sets this whole file exists to keep
# apart would land in one tree.
_queue = list(enumerate(ORDER))
# Each card's FIRST leg is assigned here, by position, and not raced for. A
# plain shared queue looked equivalent and was not: with payloads that finish
# quickly -- stubs, or an install that dies in seconds -- card 0 could claim,
# run and finish a leg before card 1 had cleared its start stagger, and a
# two-leg kernel then ran BOTH legs on device 0. That is the silent
# serialisation the shortfall guard exists to catch, arriving by a route the
# guard cannot see. Seeding makes "one leg per card while there are legs to go
# round" a property of the assignment rather than of how fast the legs happen
# to run.
SEEDS = _queue[:N_GPU]
pending = _queue[N_GPU:]
# Reserved BEFORE any worker thread exists, further down. A seat promised to a
# seed but not taken until that worker wakes is a seat a free worker can see as
# empty, which is exactly how run 32667451396 put gptoss and control on one
# card. See the _admit reservation loop below the definition.
pending_lock = threading.Lock()

# What each card is currently carrying, in GB and in count. Admission is taken
# under the same lock as the queue so two workers cannot both look at a card
# with room and both decide to use it.
card_load = {{i: 0.0 for i in range(N_GPU)}}
card_count = {{i: 0 for i in range(N_GPU)}}


def _admit(gpu_index, name):
    """Reserve room for `name` on `gpu_index`, or return False.

    Called holding `pending_lock`. Returning False leaves the leg on the queue
    for a card that can take it -- or for this one, later, once something
    finishes. It must never mutate on the failing path or a refused leg would
    leak capacity it never used.
    """
    want = VRAM_GB.get(name, 1.0)
    if card_count[gpu_index] >= MAX_LEGS_PER_CARD:
        return False
    if card_count[gpu_index] and card_load[gpu_index] + want > CARD_VRAM_BUDGET_GB:
        return False
    card_load[gpu_index] += want
    card_count[gpu_index] += 1
    return True


def _release(gpu_index, name):
    with pending_lock:
        card_load[gpu_index] -= VRAM_GB.get(name, 1.0)
        card_count[gpu_index] -= 1


def worker(gpu_index, seed):
    if seed is not None:
        # The 5s stagger the per-payload threads used to get from the start
        # loop, kept for the same reason: two `uv venv` creations in the same
        # instant contend on the same package cache. Only the first leg on a
        # card needs it; later ones are already separated by however long the
        # previous leg ran.
        if gpu_index:
            time.sleep(5 * gpu_index)
        idx, name = seed
        # No _admit here. The seat was taken in the main thread before ANY
        # worker started, and taking it here instead is what let gptoss onto
        # a card that a seed had already been promised: the stagger below
        # meant card 1 sat unreserved for 5s, a free worker looked at an
        # apparently empty card, and 12.78 + 0.70 GB then ran on one 13.0 GB
        # budget for 691s on run 32667451396. The old call also discarded the
        # answer, so even once it did refuse, the leg ran anyway.
        try:
            run_one(name, gpu_index, idx)
        finally:
            _release(gpu_index, name)
    while True:
        picked = None
        with pending_lock:
            if not pending:
                # Nothing left to start. A worker whose card still holds a leg
                # simply exits; the OTHER thread on that card is the one still
                # running it, and the joins below wait for that.
                return
            # First leg on the queue this card can actually hold. Not strictly
            # the head: gptoss cannot join an occupied card, and refusing to
            # look past it would idle a card that has room for the small leg
            # behind it.
            for position, (idx, name) in enumerate(pending):
                if _admit(gpu_index, name):
                    picked = pending.pop(position)
                    break
        if picked is None:
            # Every remaining leg is too big for what this card has free right
            # now. Wait for a release rather than spin; something is running or
            # the queue would have been empty.
            time.sleep(1.0)
            continue
        idx, name = picked
        try:
            run_one(name, gpu_index, idx)
        finally:
            _release(gpu_index, name)

# The CPU lane, started BEFORE the cards are handed out and running beside
# them. It carries work that needs no GPU -- the Studio install: a checkout, a
# frontend build, a venv, a llama.cpp download and a Playwright browser -- so
# every second of it that overlaps the training legs is a second nobody waits
# for. It is not given a card and does not queue for one.
# The PREFETCH lane. Started before anything else, because it is the only lane
# whose value is entirely a function of how early it starts: a prefetch pays
# only for what it finishes before the leg that wants the model begins, and
# legs.KERNELS puts gptoss third specifically to give this lane that window.
#
# A driver THREAD rather than a papermill payload, deliberately. It needs the
# image's huggingface_hub and nothing else -- no venv, no install, no card --
# so as a payload it would spend its first ~60s building a virtualenv it has no
# use for, which is 60s taken off the head start that is the whole point.
prefetch_thread = None
if PREFETCH_BLOB:
    def _prefetch_lane():
        src = gzip.decompress(base64.b64decode(PREFETCH_BLOB)).decode("utf-8")
        # The cell never raises by contract, but exec'ing generated source on a
        # lane nobody joins before the cards start is not the place to find out
        # otherwise: an escape here would kill the thread silently and the run
        # would look like a prefetch that simply found nothing to do.
        try:
            exec(compile(src, "<prefetch>", "exec"), {{"__name__": "prefetch"}})
        except BaseException as exc:
            print("{DRIVER_SENTINEL}_PREFETCH_FAILED " + json.dumps(
                {{"error": f"{{type(exc).__name__}}: {{exc}}"}}), flush=True)
    prefetch_thread = threading.Thread(target = _prefetch_lane, daemon = True)
    prefetch_thread.start()
    print("{DRIVER_SENTINEL}_PREFETCH " + json.dumps({{"started": True}}), flush=True)

cpu_thread = None
if CPU_LANE:
    def _cpu_lane():
        run_one(CPU_LANE, None, len(ORDER))
    cpu_thread = threading.Thread(target = _cpu_lane, daemon = False)
    cpu_thread.start()
    print("{DRIVER_SENTINEL}_CPU_LANE " + json.dumps({{"payload": CPU_LANE}}), flush=True)

# MAX_LEGS_PER_CARD workers per card, not one. The extra worker is what lets a
# card pick up a second small leg beside the one it is already running; it
# takes nothing when VRAM says no, so a card carrying gptoss behaves exactly as
# it did when there was one worker each.
# Take every seed's seat before a single worker exists. Each seed is the first
# leg on an empty card, so each of these always succeeds; the point is WHEN it
# happens, not whether. Doing it inside the seed worker left card 1 unreserved
# for the length of its start stagger.
for _g, _seed in enumerate(SEEDS):
    _admit(_g, _seed[1])
print("{DRIVER_SENTINEL}_SEEDS " + json.dumps(
    {{str(_g): _seed[1] for _g, _seed in enumerate(SEEDS)}}), flush=True)

threads = []
for gpu_index in range(N_GPU):
    for slot in range(MAX_LEGS_PER_CARD):
        seed = SEEDS[gpu_index] if (slot == 0 and gpu_index < len(SEEDS)) else None
        t = threading.Thread(target=worker, args=(gpu_index, seed), daemon=False)
        t.start()
        threads.append(t)
for t in threads:
    t.join()

# AFTER_GPU wants every card, so it waits for the queue to drain rather than
# taking one from it. Studio's own driver keeps both T4s visible on purpose --
# "Studio's own device selection is part of what is under test; masking one
# would test a machine nobody has" -- and that stays true here.
if cpu_thread is not None:
    cpu_thread.join()

if AFTER_GPU:
    prior = results.get(CPU_LANE) if CPU_LANE else None
    if CPU_LANE and (prior is None or prior.get("returncode") != 0):
        # Do NOT run it. The install half is what puts the interpreter and the
        # llama.cpp on disk, so without it this half fails on a missing venv
        # and reports that as a Studio regression. Recording the skip keeps the
        # cause attached to the effect.
        results[AFTER_GPU] = {{
            "returncode": None, "gpu": None, "seconds": 0.0,
            "error": "the install lane did not succeed, so there is nothing "
                     "installed to test",
            "kernel": None, "output_exists": False,
        }}
        print("{DRIVER_SENTINEL}_SKIPPED " + json.dumps({{AFTER_GPU: results[AFTER_GPU]}}),
              flush=True)
    else:
        run_one(AFTER_GPU, None, len(ORDER) + 1)

print("{DRIVER_SENTINEL}_RESULTS " + json.dumps(results), flush=True)
'''

    tail = f"""# Surface each child's tail inline so the kernel log alone is diagnosable
# even if artifact collection fails.
for name in sorted(PAYLOADS):
    log = WORK / (pathlib.Path(name).stem + "_driver.log")
    print(f"\\n===== {{name}} (last 120 lines) =====", flush=True)
    if log.exists():
        print("\\n".join(log.read_text(errors="replace").splitlines()[-120:]),
              flush=True)
    else:
        print("NO LOG", flush=True)

# Drop the payload copies so `kernels output` returns only executed notebooks
# and child logs.
for name in sorted(PAYLOADS):
    try:
        (WORK / name).unlink()
    except OSError:
        pass

# Drop everything large AND reconstructible. `kernels output` returns the
# whole of /kaggle/working and we then wait for it over the wire; a previous
# sweep shipped 371MB back, almost all of it venv site-packages, and the
# download was still running eighteen minutes later.
import shutil as _shutil
for _pat in ("venv_*", "unsloth_compiled_cache*", "t4_smoke_src*",
             "huggingface_tokenizers_cache", "*/trainer*", "outputs"):
    for _d in WORK.glob(_pat):
        try:
            _shutil.rmtree(_d, ignore_errors=True)
        except Exception:
            pass
print("{DRIVER_SENTINEL}_PRUNED " + json.dumps(
    sorted(p.name for p in WORK.iterdir())), flush=True)

print("{DRIVER_SENTINEL} complete", flush=True)
"""

    return {
        "cells": [_code_cell(setup), _code_cell(runner), _code_cell(tail)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle_t4_ci": {"payloads": sorted(payloads)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


STUDIO_INSTALL_NOTEBOOK = "studio_install.ipynb"
STUDIO_TEST_NOTEBOOK = "studio_test.ipynb"


def _studio_builder():
    """Load ``kaggle_studio_ci/build_kernel.py`` under a name of its own.

    By PATH, not by import. That directory and this one BOTH contain a module
    called ``build_kernel`` (and both contain a ``report`` too), and the test
    suite puts both on ``sys.path``, so a plain ``import build_kernel`` resolves
    to whichever reached ``sys.modules`` first -- which is decided by test
    order, not by intent. That exact collision has already been paid for once
    here: adding one ``sys.path.insert`` to a test took nine unrelated summary
    tests down with it.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "kaggle_studio_ci" / "build_kernel.py"
    spec = importlib.util.spec_from_file_location("kaggle_studio_ci__build_kernel", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the Studio kernel builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prefetch_builder():
    """Load ``kaggle_prefetch.py`` by PATH, for the reason ``_studio_builder``
    documents: two sibling directories both ship a ``build_kernel`` and a
    ``report``, so plain imports here resolve by test order rather than intent.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "kaggle_prefetch.py"
    spec = importlib.util.spec_from_file_location("kaggle_ci__prefetch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the prefetch builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def studio_payloads(*, unsloth_ref: str, repo_url: str, payload_args: str) -> dict[str, dict]:
    """The Studio payload, split into its GPU-free half and its GPU half."""
    studio = _studio_builder()
    return {
        STUDIO_INSTALL_NOTEBOOK: studio.build_payload_notebook(
            unsloth_ref = unsloth_ref,
            repo_url = repo_url,
            payload_args = payload_args,
            phase = "install",
        ),
        STUDIO_TEST_NOTEBOOK: studio.build_payload_notebook(
            unsloth_ref = unsloth_ref,
            repo_url = repo_url,
            payload_args = payload_args,
            phase = "test",
        ),
    }


def build_kernel(
    payload_dir: Path,
    leg_names,
    *,
    unsloth_ref: str,
    zoo_ref: str,
    extra_args: tuple[str, ...],
    per_run_timeout: int,
    skip_reference: bool = False,
    studio: dict | None = None,
    prefetch_repos: tuple[str, ...] = (),
) -> dict:
    payloads = {}
    isolation = {}
    legs_by_payload = {}
    for leg in resolve(leg_names):
        name = f"t4_{leg.name}.ipynb"
        payloads[name] = build_payload_notebook(
            payload_dir,
            leg,
            unsloth_ref = unsloth_ref,
            zoo_ref = zoo_ref,
            extra_args = extra_args,
            reference = "" if skip_reference else None,
        )
        isolation[name] = leg.system_site_packages
        legs_by_payload[name] = leg
    # The card queue is the LEGS. Studio's two halves ride the same kernel but
    # not the same queue, so expected_gpus is derived before they are added:
    # they are what the cards are freed FOR, not more work to schedule onto
    # them.
    expected_gpus = min(len(payloads), SESSION_GPUS)
    cpu_lane = after_gpu = None
    if studio:
        payloads.update(studio_payloads(**studio))
        cpu_lane = STUDIO_INSTALL_NOTEBOOK
        after_gpu = STUDIO_TEST_NOTEBOOK
        # Both halves see the Kaggle image, as the standalone Studio kernel
        # does: install.sh builds its OWN venv under STUDIO_HOME and that is
        # the interpreter every assertion runs under, so the outer one only has
        # to be able to start papermill.
        isolation[STUDIO_INSTALL_NOTEBOOK] = True
        isolation[STUDIO_TEST_NOTEBOOK] = True
    # min(), so a one-leg kernel (a --legs dispatch, or a debugging run) still
    # stands down only on a genuinely empty allocation rather than demanding a
    # second card it will never use.
    return build_driver(
        payloads,
        per_run_timeout,
        isolation,
        expected_gpus = expected_gpus,
        cpu_lane = cpu_lane,
        after_gpu = after_gpu,
        prefetch_repos = prefetch_repos,
        vram_source = legs_by_payload,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-dir", required = True)
    ap.add_argument(
        "--out",
        required = True,
        help = "output notebook. With --all-kernels this is a "
        "prefix and the files are <prefix>1.ipynb, "
        "<prefix>2.ipynb, ...",
    )
    ap.add_argument(
        "--legs", help = "comma separated leg names; one per T4 of the session. See legs.py"
    )
    ap.add_argument(
        "--all-kernels",
        action = "store_true",
        help = "build every kernel in legs.KERNELS. This is what "
        "the workflow uses, so the leg-to-kernel plan lives "
        "in one place rather than being restated in YAML",
    )
    ap.add_argument("--unsloth-ref", default = "main")
    ap.add_argument("--zoo-ref", default = "main")
    ap.add_argument(
        "--smoke-args",
        default = "",
        help = "extra args appended to EVERY leg's entry script. "
        "Shared on purpose: the control and canary legs "
        "must not differ in anything but versions",
    )
    ap.add_argument(
        "--skip-reference",
        action = "store_true",
        help = "build with no band check at all. Only for the one run that recaptures a reference",
    )
    ap.add_argument("--per-run-timeout", type = int, default = 2400)
    ap.add_argument(
        "--no-prefetch",
        action = "store_true",
        help = "do not warm the HF cache on a background lane. The models are "
        "then downloaded by the legs that want them, as they were before the "
        "lane existed. Note that legs.KERNELS is ORDERED for the prefetch -- "
        "gptoss sits third to give the lane a window -- so this flag is for "
        "isolating the lane when debugging, not a supported way to run",
    )
    ap.add_argument(
        "--with-studio",
        action = "store_true",
        help = "also carry the Studio GPU payload in this kernel, split in two: "
        "its checkout/install/browser half runs on a CPU lane beside the "
        "training legs and never takes a card, and its assertions run once "
        "the legs have freed both. Only valid with --all-kernels",
    )
    ap.add_argument(
        "--studio-repo-url",
        default = "https://github.com/unslothai/unsloth",
        help = "repository the Studio half checks out and installs",
    )
    ap.add_argument(
        "--studio-args",
        default = "",
        help = "extra args for tests/kaggle/studio_gpu/run_studio_gpu.py",
    )
    args = ap.parse_args()

    if args.with_studio and not args.all_kernels:
        # --legs builds ONE kernel of named legs, which is the debugging shape.
        # Attaching Studio to it would put a 10-minute install beside a
        # deliberately narrowed run.
        raise SystemExit("--with-studio requires --all-kernels")

    if args.all_kernels == bool(args.legs):
        raise SystemExit("pass exactly one of --legs and --all-kernels")

    if args.all_kernels:
        plan = [list(kernel) for kernel in KERNELS]
        outputs = [Path(f"{args.out}{i + 1}.ipynb") for i in range(len(plan))]
    else:
        plan = [[n.strip() for n in args.legs.split(",") if n.strip()]]
        outputs = [Path(args.out)]
        if not plan[0]:
            raise SystemExit("--legs named nothing")

    # Studio rides the FIRST kernel only. There is one kernel today, so this is
    # not a choice with consequences yet, but naming it stops a future second
    # kernel quietly installing Studio twice and paying for it twice.
    for index, (names, out) in enumerate(zip(plan, outputs)):
        studio = None
        if args.with_studio and index == 0:
            studio = {
                "unsloth_ref": args.unsloth_ref,
                "repo_url": args.studio_repo_url,
                "payload_args": args.studio_args,
            }
        driver = build_kernel(
            Path(args.payload_dir),
            names,
            unsloth_ref = args.unsloth_ref,
            zoo_ref = args.zoo_ref,
            extra_args = tuple(args.smoke_args.split()),
            per_run_timeout = args.per_run_timeout,
            skip_reference = args.skip_reference,
            studio = studio,
            # Only the kernel that CARRIES gptoss should pay for its 12 GB.
            # There is one kernel today, so this reads as "always", but naming
            # it stops a future second kernel prefetching a model it will never
            # load -- which would be pure network cost, on a lane whose entire
            # justification is that it is free.
            prefetch_repos = () if args.no_prefetch else PREFETCH_REPOS,
        )
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(driver, indent = 1), encoding = "utf-8")
        # Says whether Studio is aboard, because a build that quietly stopped
        # packing it looks exactly like one that never asked for it.
        print(
            f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) packing "
            f"{len(names)} leg(s): {', '.join(names)}"
            + (" + the Studio install and test halves" if studio else "")
        )
    # The launcher needs one --notebook per kernel and the expected payload
    # count; both follow from the plan, so they are emitted here rather than
    # restated in the workflow.
    #
    # Studio counts as ONE payload, not two, and that holds on both paths.
    # Its two notebooks are halves of one experiment: on a healthy run the
    # install half emits no report at all and the test half emits the only
    # one, and on a broken install the install half emits a failure report and
    # the driver then SKIPS the test half. Either way the kernel produces
    # exactly one `studio-gpu` report, so counting the install half would make
    # every healthy run look like it lost one.
    #
    # Getting this wrong in the other direction is worse and is why it is
    # derived rather than typed: a merged kernel that quietly stopped running
    # Studio would still report the four legs and go green.
    legs = sum(len(n) for n in plan)
    expected_payloads = legs + (1 if args.with_studio else 0)
    _github_output("notebooks", " ".join(f"--notebook {o}" for o in outputs))
    _github_output("payloads", str(expected_payloads))
    # The launcher counts every report in the kernel; each REPORTER counts only
    # the labels it owns. So the T4 reporter is told the leg count, not the
    # payload count -- handing it 5 would have it treat a complete four-leg
    # result as short by one and dump the kernel log under a healthy run.
    _github_output("legs", str(legs))
    return 0


def _github_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(f"{key}={value}\n")
    print(f"[build] {key}={value}")


if __name__ == "__main__":
    raise SystemExit(main())
