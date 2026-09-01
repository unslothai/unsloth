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
    """A notebook that installs one leg's stack and runs it.

    One GPU for every leg but the `all_cards` one, which is unpinned so that
    unsloth's DEVICE_COUNT > 1 code path is reachable at all.
    """
    root = _kernel_root(leg)
    # Baked in HERE, at build time, from the leg's own declaration. The check
    # in the verify cell compares against this constant rather than against
    # anything the running kernel can see.
    expected_visible = 2 if leg.all_cards else 1
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
import glob, json, os, re, subprocess, sys, time
print("{PAYLOAD_SENTINEL} leg {leg.name}: {leg.summary}", flush=True)
GROUPS = {json.dumps(groups)}
UNINSTALL = {json.dumps(list(leg.uninstall))}
print("{PAYLOAD_SENTINEL} install plan " + json.dumps(GROUPS), flush=True)

# Wheels the driver built ONCE, before any leg started, from the very SHAs
# these groups name. Empty or missing means the build did not happen or did not
# finish, and every spec below then resolves exactly as it did before -- a
# failed wheel build costs its own time and changes nothing about what is
# tested.
WHEEL_DIR = os.environ.get("UNSLOTH_CI_WHEEL_DIR", "")


# `name @ git+...@sha` -> the prebuilt wheel for `name`, when there is one.
#
# Matched on the DISTRIBUTION NAME, normalised the way a wheel filename is
# (PEP 503: runs of -_. collapse to _), because `unsloth_zoo` on the left of the
# `@` arrives as `unsloth_zoo-2026.8.1-py3-none-any.whl` on disk. Getting that
# wrong is invisible: no wheel matches, every leg falls back to its git spec,
# and the run is merely as slow as it was before.
#
# A comment and not a docstring: this whole cell is generated from an f-string
# delimited by triple quotes, so a docstring here ENDS the template and the
# build dies with a SyntaxError in build_kernel.py rather than in the cell.
def _localise(spec):
    if not WHEEL_DIR or "@ git+" not in spec:
        return spec
    name = spec.split("@ git+", 1)[0].strip()
    key = re.sub(r"[-_.]+", "_", name).lower()
    for cand in sorted(glob.glob(os.path.join(WHEEL_DIR, "*.whl"))):
        stem = os.path.basename(cand).split("-", 1)[0]
        if re.sub(r"[-_.]+", "_", stem).lower() == key:
            return cand
    return spec


def pip(args):
    args = [_localise(a) for a in args]
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
    _g0 = time.time()
    pip(group)
    # Per group, because "install took 191s" was all any artifact ever said and
    # it is not actionable: it hides which of the four groups was the cost, and
    # therefore whether the shared wheels helped at all.
    print("{PAYLOAD_SENTINEL} install group " + json.dumps({{
        "seconds": round(time.time() - _g0, 1),
        "spec": [_localise(a) for a in group],
    }}), flush=True)
# After the install groups, never before: these distributions arrive as
# dependencies of something installed above, so removing them earlier would
# remove nothing and the later install would put them back.
#
# Never fatal. If a name is absent pip says so and exits 0, and even a genuine
# failure here should not cost the session -- the payload then fails on its own
# terms with an error that names what actually went wrong, which is more useful
# than a stand-down during setup.
if UNINSTALL:
    _u0 = time.time()
    _up = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *UNINSTALL],
        capture_output=True, text=True)
    print("{PAYLOAD_SENTINEL} uninstall " + json.dumps({{
        "seconds": round(time.time() - _u0, 1),
        "spec": UNINSTALL,
        "returncode": _up.returncode,
        "stdout": _up.stdout[-2000:],
        "stderr": _up.stderr[-2000:],
    }}), flush=True)
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
    # EXACTLY the count this leg asked for, and the number is baked in at build
    # time rather than read from the environment -- reading it from the same
    # variable that sets it would make the check agree with whatever happened.
    #
    # 1 for every ordinary leg: two would mean the driver failed to pin this
    # payload to its own card, and accelerate would shard across both, which is
    # a different test from the single T4 a Colab user gets. Zero means the
    # payload cannot run at all on the hardware it was billed for.
    #
    # 2 for the `multi_gpu` leg, which is deliberately unpinned so that
    # unsloth's DEVICE_COUNT > 1 branch is reachable. A leg that asks for two
    # and gets one still fails here, which is the case that matters: silently
    # running it on one card would leave the multi-GPU claim asserted and
    # untested, and green.
    assert torch.cuda.device_count() == {expected_visible}, (
        f"expected exactly {expected_visible} visible GPU(s), got "
        f"{{torch.cuda.device_count()}}")
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
    after_gpu_concurrent: bool = False,
    shared_wheel_specs: tuple[str, ...] = (),
    overlays: dict[str, tuple[str, ...]] | None = None,
    all_card: tuple[str, ...] = (),
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
    # Only legs carry overlays; Studio's halves install into their own tree.
    overlay_specs = {
        name: tuple(specs) for name, specs in (overlays or {}).items() if specs and name in payloads
    }
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
    # Studio's halves are not legs and have no Leg record. The test half is the
    # only one that takes a card, and it is priced from what it loads: a 2B
    # chat model at UD-Q4_K_XL plus a 0.5B LoRA. Deliberately generous -- an
    # under-price here is what would let it share with something that does not
    # fit, and the whole admission check is only as honest as its inputs.
    for name in (cpu_lane, after_gpu):
        if name:
            vram_gb.setdefault(name, 2.2)
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
# Defined HERE, not beside the build below. run_one exports it into every
# payload's env, and the Studio CPU lane calls run_one before the build block
# is reached -- so assigning it there is a NameError on the first payload to
# start, which is the one that runs earliest and would take the whole kernel
# down with it.
WHEEL_DIR = VENV_ROOT / "wheels"
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

# Per-leg pure-Python version pins, laid OVER the venv rather than resolved
# into it. See Leg.overlay. Empty for a leg that wants the base as it resolved.
OVERLAYS = {overlay_specs!r}
# Distributions an overlay may never carry. torch, triton and bitsandbytes are
# native and coupled to the driver stack: a shadowed copy on PYTHONPATH either
# fails to import or, worse, imports and silently disagrees with the CUDA
# runtime already loaded. The overlay exists to move transformers/trl/peft, not
# to rebuild the machine. Substring match, lowercased, so `nvidia-cublas-cu12`
# and `torchvision` are caught without listing every wheel NVIDIA ships.
OVERLAY_DENY = ("torch", "triton", "bitsandbytes", "nvidia", "cuda",
                "xformers", "flash")

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

# Requirement specs that EVERY leg installs identically, built once up front.
# Derived from the legs' own install groups rather than written out again here,
# because a second copy of the two SHAs is a second thing to keep in step -- and
# one that drifts silently, since a wheel built from the wrong ref installs
# perfectly.
SHARED_WHEEL_SPECS = {shared_wheel_specs!r}

# Payloads that are NOT in the card queue.
#
# CPU_LANE runs from t=0 alongside the training legs and never takes a card.
# AFTER_GPU runs once the queue has drained and wants every card. Both are None
# on an ordinary notebook-only kernel, which leaves the schedule exactly as it
# was.
CPU_LANE = {cpu_lane!r}
AFTER_GPU = {after_gpu!r}
# ALL_CARD payloads want EVERY card visible rather than one. They are not in
# the card queue either, because the queue hands out one card at a time and
# there is no index to give them.
#
# They are NOT the same as AFTER_GPU. AFTER_GPU is unpinned AND waits for the
# queue to drain; an ALL_CARD leg reserves its (small) share on every card
# through _admit_all and then runs BESIDE the training legs, which is what
# makes covering unsloth's DEVICE_COUNT > 1 branch cost nothing.
ALL_CARD = {all_card!r}
# When true, AFTER_GPU does not wait for the cards to drain. It is admitted to
# a card by the same VRAM check the legs use, so it runs BESIDE a light leg
# rather than behind the last one.
#
# Worth ~211s in simulation, because Studio's install-then-test chain is the
# critical path while the legs have slack. What it costs is real and is why it
# is a flag and not the default: Studio then sees ONE card instead of two, and
# its own device selection is part of what it exists to prove.
AFTER_GPU_CONCURRENT = {after_gpu_concurrent!r}

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
# 1 when the venvs had to fall back to WORK. The fallback above says it "keeps
# a one-leg-per-card run working", and until this line that was a description
# of an intention rather than of anything the code did: MAX_LEGS_PER_CARD was a
# constant, so a box with no writable /tmp would still build four torch venvs,
# two at a time per card, in the 19.5 GB /kaggle/working that the comment at
# the top of this block says they do not fit in. It would have surfaced as an
# install failing halfway through for reasons that look nothing like a disk.
MAX_LEGS_PER_CARD = 2 if VENV_ROOT != WORK else 1

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
        # PIN THE INTERPRETER TO THE DRIVER'S OWN, and it is not cosmetic.
        #
        # uv's default python-preference is `managed`, so once ANY managed
        # CPython exists under ~/.local/share/uv/python -- which something in a
        # longer kernel eventually downloads -- a bare `uv venv` silently builds
        # on THAT interpreter instead of the image's. `--system-site-packages`
        # is still honoured and still inherits nothing, because a 3.13 venv
        # cannot see a 3.12 site-packages.
        #
        # Measured, 6 sessions, deterministic per arm: the Default leg came up
        # on python 3.13.13 / torch 2.12.1+cu130 / datasets 5.0.1 in all three
        # arm-B runs and on 3.12.13 / torch 2.10.0+cu128 / datasets 4.3.0 in
        # all three arm-A runs, from one commit. Every other leg stayed on
        # 3.12. The 3.13 runs then failed 3/3 in `Dataset.from_dict` -- see the
        # overlay comment below for why -- and the whole leg was resolving a
        # stack no Kaggle user has.
        venv_cmd = [uv, "venv", str(vdir), "--seed", "--python", sys.executable]
        if system_site:
            venv_cmd.append("--system-site-packages")
        subprocess.run(venv_cmd, check=True, timeout=900)
        py = vdir / "bin" / "python"
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py),
                        "ipykernel"], check=True, timeout=900)
        kname = f"t4ci{{idx}}"
        subprocess.run([str(py), "-m", "ipykernel", "install", "--user",
                        "--name", kname], check=True, timeout=600)
        # The version is REPORTED, not assumed. A venv on the wrong interpreter
        # is invisible in the driver log otherwise, and it was: three sessions
        # went red before anyone read a python version out of a leg's report.
        _ver = subprocess.run(
            [str(py), "-c", "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True, text=True, timeout=120).stdout.strip()
        print(f"{DRIVER_SENTINEL}_VENV " + json.dumps(
            {{"idx": idx, "kernel": kname, "system_site": system_site,
              "python": _ver,
              "driver_python": "%d.%d.%d" % sys.version_info[:3]}}),
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

    # The OTHER three caches, which a separate venv does not protect.
    #
    # UNSLOTH_COMPILE_LOCATION above covered unsloth's generated modules and
    # was assumed to be the whole story. It is not. torch and triton key their
    # caches off $TMPDIR, not off the interpreter, so on torch 2.9.1 an unset
    # TORCHINDUCTOR_CACHE_DIR resolves to `tempfile.gettempdir()/torchinductor_
    # $USER` (torch/_inductor/runtime/cache_dir_utils.py:22) and the triton
    # cache lands UNDER that same directory. Every leg on the box therefore
    # shared /tmp/torchinductor_root -- four payloads whose entire purpose is
    # to install DIFFERENT transformers/TRL/peft versions and compile the same
    # modules, writing one cache between them.
    #
    # Per-leg directories rather than TORCHINDUCTOR_FORCE_DISABLE_CACHES or
    # TRITON_ALWAYS_COMPILE, which are the blunter knobs and do exist
    # (torch/compiler/config.py:85 and triton/knobs.py:358). Those force every
    # leg to recompile from cold, which spends time on the critical path this
    # kernel is being shortened along, and makes the legs measure a machine no
    # user has. Isolation is what is wanted here, not cache defeat.
    #
    # TMPDIR is set too, and it is the belt to the braces: anything else that
    # resolves a cache through gettempdir() follows it without this file having
    # to know the variable's name.
    _leg_tmp = VENV_ROOT / ("tmp_" + pathlib.Path(name).stem)
    for _sub in ("", "inductor", "triton"):
        try:
            (_leg_tmp / _sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    env["TMPDIR"] = str(_leg_tmp)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(_leg_tmp / "inductor")
    env["TRITON_CACHE_DIR"] = str(_leg_tmp / "triton")
    # Where the once-built wheels are. Read by the install cell, which falls
    # back to its git specs when the directory is empty or absent.
    env["UNSLOTH_CI_WHEEL_DIR"] = str(WHEEL_DIR)

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

    # The per-leg OVERLAY: this leg's version pins laid over the venv rather
    # than resolved into it.
    #
    # Two pip calls, and the first one is the point. `--dry-run --report`
    # resolves the FULL closure against the environment as it actually is, so
    # the second call can be `--no-deps` over a manifest instead of a guess. An
    # overlay assembled by naming packages by hand pairs a new transformers with
    # the base's older tokenizers or huggingface_hub, and fails minutes later
    # inside a payload with an import error that reads like a code bug.
    #
    # Never fatal. A leg whose overlay fails runs on the base versions and says
    # so: that is a leg testing the wrong versions, which the payload's own
    # resolved-version record makes visible, and it is strictly better than
    # standing the leg down and reporting nothing at all.
    _ov_specs = OVERLAYS.get(name) or ()
    if _ov_specs:
        # The trailing `site-packages` is LOAD-BEARING, and it is the one part
        # of this path that looks like decoration.
        #
        # dill decides whether to pickle a module by REFERENCE or BY VALUE with
        # `_is_builtin_module` (dill/_dill.py), which answers yes only if the
        # module's __file__ starts with a sys prefix, ends with an extension
        # suffix, or contains the literal string 'site-packages'. A plain
        # `pip install --target /tmp/t4ci_venvs/overlay_X` satisfies none of
        # those, so every package in the overlay is pickled by value.
        #
        # That matters because `Dataset.from_dict` fingerprints through dill:
        # `datasets/utils/_dill.py:_save_arrowTable` saves `create_arrowTable`,
        # dill walks its globals, reaches the pyarrow MODULE, pickles it by
        # value, and hits pyarrow's Cython `MonthDayNano`, whose __module__ is
        # `builtins`. The result is
        #     PicklingError: Can't pickle <class 'MonthDayNano'>:
        #     it's not found as builtins.MonthDayNano
        # from a two-line `Dataset.from_dict` on a dict of plain strings, with
        # nothing in the traceback that names this driver.
        #
        # Reproduced on CPU in seconds against a byte-identical package tree,
        # with the DIRECTORY NAME as the only variable: the plain --target dir
        # raised, the copy named site-packages returned a fingerprint.
        _ov_dir = VENV_ROOT / ("overlay_" + pathlib.Path(name).stem) / "site-packages"
        _ov_report = _leg_tmp / "overlay_report.json"
        _ov_t0 = time.time()
        _ov_rec = {{"payload": name, "requested": list(_ov_specs)}}
        try:
            _rc = subprocess.run(
                [py, "-m", "pip", "install", "--dry-run", "--quiet",
                 "--report", str(_ov_report)] + list(_ov_specs),
                capture_output=True, text=True, timeout=600)
            _closure = []
            if _ov_report.exists():
                for _item in json.loads(_ov_report.read_text()).get("install", []):
                    _meta = _item.get("metadata", {{}})
                    _closure.append((_meta.get("name", "?"), _meta.get("version", "?")))
            _keep, _dropped = [], []
            for _nm, _ver in _closure:
                if any(_d in _nm.lower() for _d in OVERLAY_DENY):
                    _dropped.append(f"{{_nm}}=={{_ver}}")
                else:
                    _keep.append(f"{{_nm}}=={{_ver}}")
            _ov_rec.update({{"resolved": _keep, "denied": _dropped,
                            "resolve_returncode": _rc.returncode}})
            if _keep:
                _rc2 = subprocess.run(
                    [py, "-m", "pip", "install", "--quiet", "--no-deps",
                     "--target", str(_ov_dir)] + _keep,
                    capture_output=True, text=True, timeout=900)
                _ov_rec["install_returncode"] = _rc2.returncode
                if _rc2.returncode == 0:
                    # PREPENDED, so the overlay wins over the venv. The payload
                    # child reads this at interpreter start, which is the only
                    # moment it can matter: an already-imported transformers
                    # cannot be displaced by any later sys.path edit.
                    env["PYTHONPATH"] = str(_ov_dir) + os.pathsep + env.get("PYTHONPATH", "")
                    _ov_rec["active"] = True
                else:
                    _ov_rec["error"] = _rc2.stderr[-400:]
            else:
                _ov_rec["error"] = "resolved to nothing installable"
        except BaseException as _exc:  # noqa: BLE001
            _ov_rec["error"] = f"{{type(_exc).__name__}}: {{_exc}}"[:300]
        _ov_rec["seconds"] = round(time.time() - _ov_t0, 1)
        _ov_rec.setdefault("active", False)
        print(f"{DRIVER_SENTINEL}_OVERLAY " + json.dumps(_ov_rec), flush=True)

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
# ALL_CARD legs are filtered out here rather than left out of ORDER, so ORDER
# stays the single declaration of what this kernel runs and the report's
# ordering is unchanged. Their INDEX is preserved for run_one's numbering.
_queue = [(i, n) for i, n in enumerate(ORDER) if n not in ALL_CARD]
_all_card_queue = [(i, n) for i, n in enumerate(ORDER) if n in ALL_CARD]
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
# Concurrent payloads that hold no card index. A list rather than an int so the
# nested lane functions can mutate it without a `global` in generated source.
unpinned_count = [0]


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
    # An unpinned payload contends for the same 4 vCPUs without holding a card,
    # so the total bound has to see it here as well. Enforcing it only in
    # _admit_all would let the card queue fill up beside it and put N_GPU *
    # MAX_LEGS_PER_CARD + 1 installs on four cores.
    if sum(card_count.values()) + unpinned_count[0] >= N_GPU * MAX_LEGS_PER_CARD:
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


def _admit_all(name):
    """Reserve room for `name` on EVERY card, or on none of them.

    All or nothing, under one hold of `pending_lock`. A partial reservation is
    the failure worth naming: the leg would hold a seat on card 0 while waiting
    for card 1, which is capacity nothing is using and which can deadlock
    against a big leg waiting for card 0.

    VRAM is charged to EVERY card and the CONTENTION to a global counter, and
    both halves of that are measurements rather than preferences. `card_load` tracks memory, and an all-card leg genuinely holds
    a CUDA context on each card -- measured at 1.2 GB on
    unsloth-probe-multigpu-r2-a280e2, which is why its declaration is 1.2 rather
    than the 0.7 the other Qwen legs carry. `card_count` tracks something else
    entirely: MAX_LEGS_PER_CARD exists as a proxy for 4-vCPU CONTENTION, and its
    own comment says so -- "the legs contend for CORES long before they contend
    for memory". An all-card leg is ONE process. Charging it a slot on both
    cards counts one process twice.

    Charging the count PER CARD is wrong in both available ways, and each was
    caught by a different instrument.

    Charged on EVERY card it takes one of the two slots on each, and the driver
    simulation showed no card ever holding two legs at once -- the co-tenancy
    that makes a fifth and sixth leg affordable, gone.

    Charged on ONE card it decides a placement it has nothing to do with, and
    that cost 188.7s of wall clock on hardware. The A/B of
    unsloth-probe-ab-baseline-5leg-20db9c against
    unsloth-probe-ab-with-multigpu-6169ca, same commit and same day:

      A  1936.4s   gpu0 10.0s idle    gpu1 3.0s idle
      B  2125.1s   gpu0 622.6s idle   gpu1 0.0s idle

    In B, Studio wanted a card at t=653. gpu0 held canary plus this leg's count
    -- two, the cap -- so it was refused and took gpu1, where the 1707s vision
    leg had ~1080s left. gpu1 then carried BOTH of the two longest payloads
    while gpu0 idled for over ten minutes. The leg itself ran 28.2 -> 668.1
    fully overlapped and contributed nothing directly.

    So the contention goes on a GLOBAL counter. MAX_LEGS_PER_CARD is a proxy
    for 4 vCPUs, and its own comment says so -- "the legs contend for CORES long
    before they contend for memory". An unpinned process contends for those
    cores without occupying a card, and the total concurrency bound
    (N_GPU * MAX_LEGS_PER_CARD) is what it should count against. VRAM still goes
    on every card, because the CUDA context really is on every card.
    """
    want = VRAM_GB.get(name, 1.0)
    with pending_lock:
        for g in range(N_GPU):
            if card_count[g] and card_load[g] + want > CARD_VRAM_BUDGET_GB:
                return False
        if sum(card_count.values()) + unpinned_count[0] >= N_GPU * MAX_LEGS_PER_CARD:
            return False
        for g in range(N_GPU):
            card_load[g] += want
        unpinned_count[0] += 1
    return True


def _release_all(name):
    want = VRAM_GB.get(name, 1.0)
    with pending_lock:
        for g in range(N_GPU):
            card_load[g] -= want
        unpinned_count[0] -= 1


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

# Build unsloth and unsloth_zoo ONCE, before any leg starts.
#
# Every leg's first two install groups are byte-identical -- the same two
# pinned SHAs, zoo from main and unsloth from the ref under test -- and pip
# does not cache a VCS build, so run 32679427416 cloned and built both of them
# FOUR times. Install was 149-191s per leg there, the single largest phase of
# every leg and 41% of gpt-oss.
#
# Synchronous, and that is the whole design. A background lane would lose the
# race it exists to win: the legs are admitted to their cards and start
# installing at t=0, so a wheel that is still building when they get there
# saves nothing. Studio's CPU lane is started FIRST so its 345s overlaps this
# rather than queueing behind it.
#
# Building from the same SHAs the groups name is what keeps this honest: the
# wheel IS main-plus-the-PR, not a substitute for it. A leg whose wheel is
# missing falls back to its original git spec, so a failed build costs the time
# it took and changes nothing about what is tested.
if SHARED_WHEEL_SPECS:
    _t0 = time.time()
    try:
        WHEEL_DIR.mkdir(parents=True, exist_ok=True)
        _proc = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q",
             "-w", str(WHEEL_DIR), *SHARED_WHEEL_SPECS],
            capture_output=True, text=True, timeout=900)
        _built = sorted(p.name for p in WHEEL_DIR.glob("*.whl"))
        print("{DRIVER_SENTINEL}_WHEELS " + json.dumps({{
            "seconds": round(time.time() - _t0, 1),
            "returncode": _proc.returncode,
            "built": _built,
            "error": None if _proc.returncode == 0 else _proc.stderr.strip()[-400:],
        }}), flush=True)
    except BaseException as _exc:  # noqa: BLE001
        # Never fatal. The legs still carry their git specs.
        print("{DRIVER_SENTINEL}_WHEELS " + json.dumps({{
            "seconds": round(time.time() - _t0, 1), "returncode": None,
            "built": [], "error": f"{{type(_exc).__name__}}: {{_exc}}",
        }}), flush=True)

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

# The ALL_CARD lane(s). One thread each, waiting for room on every card at
# once, then running UNPINNED so torch reports every device.
#
# Started before the card workers so the leg is queued for the first moment
# both cards have a free slot, rather than after whatever the workers grab in
# the same instant. It still cannot jump the seeds: those seats were taken
# above, before any thread existed.
all_card_threads = []
for _idx, _name in _all_card_queue:
    def _all_card_lane(idx=_idx, name=_name):
        while not _admit_all(name):
            # Everything is full. Something is running or the queue would be
            # empty, so wait for a release rather than spin.
            time.sleep(1.0)
        try:
            print("{DRIVER_SENTINEL}_ALL_CARD " + json.dumps(
                {{"payload": name, "gpus": N_GPU}}), flush=True)
            # gpu_index None means DO NOT PIN, which is the whole point: an
            # index here would set CUDA_VISIBLE_DEVICES and the leg would
            # measure the single-card branch under a multi-GPU name.
            run_one(name, None, idx)
        finally:
            _release_all(name)
    _t = threading.Thread(target = _all_card_lane, daemon = False)
    _t.start()
    all_card_threads.append(_t)

threads = []
for gpu_index in range(N_GPU):
    for slot in range(MAX_LEGS_PER_CARD):
        seed = SEEDS[gpu_index] if (slot == 0 and gpu_index < len(SEEDS)) else None
        t = threading.Thread(target=worker, args=(gpu_index, seed), daemon=False)
        t.start()
        threads.append(t)
def _after_gpu_blocked():
    """The install half must have SUCCEEDED before the test half is worth
    running: it is what puts the interpreter and the llama.cpp on disk, so
    without it this half dies on a missing venv and reports that as a Studio
    regression."""
    prior = results.get(CPU_LANE) if CPU_LANE else None
    return bool(CPU_LANE) and (prior is None or prior.get("returncode") != 0)


def _after_gpu_skip():
    results[AFTER_GPU] = {{
        "returncode": None, "gpu": None, "seconds": 0.0,
        "error": "the install lane did not succeed, so there is nothing "
                 "installed to test",
        "kernel": None, "output_exists": False,
    }}
    print("{DRIVER_SENTINEL}_SKIPPED " + json.dumps({{AFTER_GPU: results[AFTER_GPU]}}),
          flush=True)


# Concurrent placement: AFTER_GPU takes a card as soon as its install lane is
# done and some card has the VRAM, rather than waiting for the whole queue.
after_thread = None
if AFTER_GPU and AFTER_GPU_CONCURRENT:
    def _after_gpu_lane():
        if cpu_thread is not None:
            cpu_thread.join()
        if _after_gpu_blocked():
            _after_gpu_skip()
            return
        chosen = None
        while chosen is None:
            with pending_lock:
                for g in range(N_GPU):
                    if _admit(g, AFTER_GPU):
                        chosen = g
                        break
            if chosen is None:
                # Every card is full. Something is running or the legs would
                # have finished, so wait for a release rather than spin.
                time.sleep(1.0)
        try:
            print("{DRIVER_SENTINEL}_AFTER_GPU_SHARED " + json.dumps(
                {{"payload": AFTER_GPU, "gpu": chosen}}), flush=True)
            run_one(AFTER_GPU, chosen, len(ORDER) + 1)
        finally:
            _release(chosen, AFTER_GPU)
    after_thread = threading.Thread(target = _after_gpu_lane, daemon = False)
    after_thread.start()

for t in threads:
    t.join()
# Joined with the card workers, not after AFTER_GPU: an all-card leg runs
# beside the training legs and must be finished before anything concludes the
# cards are free.
for t in all_card_threads:
    t.join()

# On the NON-concurrent path AFTER_GPU wants every card, so it waits for the
# queue to drain. Studio's driver keeps both T4s visible on purpose -- "Studio's
# own device selection is part of what is under test; masking one would test a
# machine nobody has" -- and that stays true here.
if cpu_thread is not None:
    cpu_thread.join()
if after_thread is not None:
    after_thread.join()

if AFTER_GPU and not AFTER_GPU_CONCURRENT:
    if _after_gpu_blocked():
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
        _after_gpu_skip()
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
    after_gpu_concurrent: bool = False,
    shared_wheels: bool = False,
) -> dict:
    payloads = {}
    isolation = {}
    overlays: dict[str, tuple[str, ...]] = {}
    legs_by_payload = {}
    leg_groups: dict[str, list[list[str]]] = {}
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
        overlays[name] = tuple(leg.overlay)
        legs_by_payload[name] = leg
        leg_groups[name] = expand_install(
            leg,
            unsloth_ref = unsloth_ref,
            zoo_ref = zoo_ref,
            payload_dir = payload_dir,
        )
    # The card queue is the LEGS. Studio's two halves ride the same kernel but
    # not the same queue, so expected_gpus is derived before they are added:
    # they are what the cards are freed FOR, not more work to schedule onto
    # them.
    expected_gpus = min(len(payloads), SESSION_GPUS)
    # An all-card leg needs SESSION_GPUS whatever the payload count is. Without
    # this a single-leg dispatch (`--legs multi_gpu`) derives 1, the shortfall
    # guard is satisfied by a one-card allocation, and the run proceeds to fail
    # inside the payload on `device_count() == 2` -- a capacity fact reported as
    # a payload verdict, which is precisely the confusion the guard's own
    # comment says it exists to prevent.
    if any(leg.all_cards for leg in legs_by_payload.values()):
        expected_gpus = SESSION_GPUS
    # Opt in. On run 32689629906 the wheels helped the leg that runs ALONE and
    # cost the three that run concurrently, and that run moved the caches at the
    # same time, so the effect is not yet attributable to either. Off by default
    # until one variable at a time says otherwise.
    shared_wheel_specs = _shared_vcs_specs(leg_groups) if shared_wheels else ()
    # Derived from the legs themselves rather than named again here: a second
    # list of which legs want every card is a second thing to keep in step, and
    # one that drifts silently -- a leg dropped from it simply gets pinned and
    # its multi-GPU assertions go looking for a second card that is not there.
    all_card = tuple(payload for payload, leg in legs_by_payload.items() if leg.all_cards)
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
        after_gpu_concurrent = after_gpu_concurrent,
        shared_wheel_specs = shared_wheel_specs,
        overlays = overlays,
        all_card = all_card,
    )


def _shared_vcs_specs(leg_groups: dict[str, list[list[str]]]) -> tuple[str, ...]:
    """VCS requirements that EVERY leg installs identically.

    Derived rather than declared. The two SHAs already live in the install
    groups, and a second copy here would be a second thing to keep in step --
    one that drifts silently, because a wheel built from the wrong ref installs
    perfectly and fails nothing.

    "Every leg" is the point of the intersection. A spec only one leg carries is
    part of what that leg is testing and must keep its own resolution; building
    it once and sharing it would quietly make the legs agree about something
    they exist to disagree about. Two legs pinning the SAME package to
    DIFFERENT refs therefore yields nothing shared, which is the safe answer.
    """
    if not leg_groups:
        return ()
    sets = []
    for groups in leg_groups.values():
        specs = set()
        for group in groups:
            for token in group:
                if "@ git+" in token or token.startswith("git+"):
                    specs.add(token)
        sets.append(specs)
    common = set.intersection(*sets) if sets else set()
    # Sorted for a stable kernel: an unordered set would rebuild the notebook
    # differently run to run and make a diff of two kernels unreadable.
    return tuple(sorted(common))


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
        "--shared-wheels",
        action = "store_true",
        help = "build unsloth and unsloth_zoo once up front and install every "
        "leg from the wheel instead of resolving the git spec per leg",
    )
    ap.add_argument(
        "--studio-concurrent",
        action = "store_true",
        help = "let the Studio GPU half share a card with a light training leg "
        "instead of waiting for the queue to drain. Faster, but Studio then "
        "sees one T4 rather than two, which narrows what it proves",
    )
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
            after_gpu_concurrent = args.studio_concurrent,
            shared_wheels = args.shared_wheels,
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
