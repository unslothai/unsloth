# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Build the Kaggle kernel notebook(s) that run the T4 legs.

Two layers, because a Kaggle GPU session is 2xT4 and each payload is written
for one card:

**Payload notebook** -- one per GPU, one per *leg* (see ``legs.py``).
Materialises its sources from inlined copies, installs that leg's library
set, probes that the imports it needs are there, runs the leg's entry
script, prints a machine-readable result line.

**Driver notebook** -- one per kernel. Carries its payloads inline (gzip +
base64) so the kernel needs no dataset attachment and no network fetch of
our sources, gives each payload its own virtualenv and its own GPU, and runs
them concurrently under papermill.

Everything the kernel needs travels inside the notebook. There is no
checkout on the Kaggle side.

Cell order is load-bearing: **materialise, install, verify, run**. The
control leg installs from a pin file that is carried inside the notebook, so
the files have to exist on disk before the first pip call. Materialising
last, as an earlier version did, meant the pin file was written after the
install that needed it.

Four hard-won details are load-bearing and are NOT safe to simplify away:

1. **Per-child virtualenv.** Every payload pip-installs a torch/transformers
   stack, and the legs deliberately install DIFFERENT ones. Sharing one
   site-packages does not merely risk corruption here, it destroys the
   experiment: the control leg's pins and the canary leg's upgrades would
   land in the same tree and the last writer would win.
2. **``uv venv --seed``.** Without it the venv has no pip, so a notebook's
   ``!pip install`` falls through PATH to the system pip while the kernel
   runs the venv interpreter: installs and imports then target different
   site-packages.
3. **``UV_SYSTEM_PYTHON=0``.** The Kaggle image ships ``UV_SYSTEM_PYTHON=1``,
   and it BEATS ``VIRTUAL_ENV``. Left alone, ``uv pip install`` writes to
   the base image while ``--system-site-packages`` lets the kernel import
   from there anyway, so both children silently share one tree and the
   isolation the venv exists to provide is undone by an environment
   variable.
4. **Runtime paths are built in Python, from ROOT.** Anything spliced into a
   generated cell as a shell-shaped fragment is a SyntaxError waiting for a
   Kaggle session; ``@ROOT/`` arguments are expanded into ``str(ROOT / ...)``
   expressions instead. ``test_generated_cells_compile`` is the cheap
   version of finding that out.

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

from legs import KERNELS, Leg, expand_install, resolve  # noqa: E402

DRIVER_SENTINEL = "KAGGLE_T4_CI_DRIVER"
PAYLOAD_SENTINEL = "KAGGLE_T4_CI_PAYLOAD"
RESULT_PREFIX = "T4_SMOKE_REPORT "

# Where the payload sources land on the Kaggle side.
KERNEL_ROOT = "/kaggle/working/t4_smoke_src"


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


def _arg_expression(value: str) -> str:
    """One entry of the child's argv, as a Python expression.

    ``@ROOT/x/y`` becomes ``str(ROOT / "x" / "y")`` so the path is assembled
    on the kernel from the kernel's own ROOT. Everything else is a plain
    string literal. The alternative -- interpolating the path into the
    generated source -- is what produced a cell that read
    ``"--label", "gpu0" --reference "{ROOT}/..."`` and died with a
    SyntaxError before a single training step ran.
    """
    if value.startswith("@ROOT/"):
        parts = [p for p in value[len("@ROOT/"):].split("/") if p]
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
    wanted = list(leg.files)
    if leg.entry not in wanted:
        wanted.append(leg.entry)

    # `reference=None` means "whatever the leg asks for"; an explicit ""
    # means the caller is turning the band check off for this run, which is
    # how a reference recapture is dispatched.
    ref_name = leg.reference if reference is None else reference
    if ref_name:
        wanted.append(f"references/{ref_name}")

    files = {}
    for name in wanted:
        path = payload_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"leg {leg.name!r} needs {name}, which is not in {payload_dir}")
        files[name] = _encode_bytes(path.read_bytes())

    materialise = f"""# Materialise the test sources carried inside this notebook.
#
# FIRST, before any install: the control leg installs from a pin file that
# travels in here, so the files have to be on disk before the first pip call.
import base64, gzip, json, os, pathlib
FILES = {json.dumps(files)}
ROOT = pathlib.Path({json.dumps(KERNEL_ROOT)})
for name, blob in FILES.items():
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.decompress(base64.b64decode(blob)))
print("{PAYLOAD_SENTINEL} sources " + json.dumps(sorted(FILES)), flush=True)
"""

    groups = expand_install(leg, unsloth_ref = unsloth_ref, zoo_ref = zoo_ref,
                            payload_dir = payload_dir)
    install = f"""# Install this leg's library set.
#
# The groups below are generated from legs.py and are the ONLY thing that
# differs between the control leg and the version canary. They are printed
# before they run so the kernel log alone says what was asked for, which is
# what makes a canary failure attributable without downloading anything.
import json, subprocess, sys
print("{PAYLOAD_SENTINEL} leg {leg.name}: {leg.summary}", flush=True)
GROUPS = {json.dumps(groups)}
print("{PAYLOAD_SENTINEL} install plan " + json.dumps(GROUPS), flush=True)

def pip(args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("  $ " + " ".join(cmd[3:]), flush=True)
    # github.com occasionally 500s on a git fetch, and PyPI occasionally
    # times out; a single upstream blip must not be reported as a notebook
    # regression. A resolution that is genuinely impossible fails all three.
    for attempt in (1, 2, 3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        print(f"  install attempt {{attempt}} failed rc={{proc.returncode}}",
              flush=True)
        print("  " + proc.stderr.strip()[-1500:], flush=True)
        if attempt == 3:
            print("{PAYLOAD_SENTINEL} INSTALL FAILED " + json.dumps(list(args)),
                  flush=True)
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
import importlib, json, sys
# Everything above was pip-installed AFTER this interpreter started, and the
# import system caches the directory listing of each sys.path entry. Without
# this, a just-installed package can be invisible to the very next import.
importlib.invalidate_caches()

# The resolved version of every package this CI watches, read from the
# installed distributions rather than from module attributes. This is the
# line a reader diffs between the control leg and the canary leg to name
# what moved, so it is printed before anything can crash.
sys.path.insert(0, {json.dumps(KERNEL_ROOT)})
import versions
print("{PAYLOAD_SENTINEL} resolved " + json.dumps(
    versions.flatten_versions(versions.resolved_versions())), flush=True)

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
    except Exception as exc:
        missing.append(f"{{mod}}: {{type(exc).__name__}}: {{exc}}")
if missing:
    print("{PAYLOAD_SENTINEL} MISSING " + json.dumps(missing), flush=True)
    raise SystemExit("payload dependencies incomplete: " + "; ".join(missing))

import torch
print("{PAYLOAD_SENTINEL} gpu " + json.dumps({{
    "count": torch.cuda.device_count(),
    "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "visible": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
}}), flush=True)
# Exactly one GPU must be visible: two would mean the driver failed to pin
# this payload to its own card, and accelerate would shard across both,
# which is a different test from the single T4 a Colab user gets.
assert torch.cuda.device_count() == 1, (
    f"expected exactly 1 visible GPU, got {{torch.cuda.device_count()}}")
"""

    argv = list(leg.args) + list(extra_args)
    if ref_name:
        argv += ["--reference", f"@ROOT/references/{ref_name}"]
    arg_exprs = ", ".join(_arg_expression(a) for a in argv)

    run = f"""# Run the leg in a child process.
#
# A child, not an import, for two reasons: the determinism setup has to run
# before torch is imported (and papermill has already imported plenty), and
# a hard crash then leaves this cell alive to report it.
import json, os, pathlib, subprocess, sys
ROOT = pathlib.Path({json.dumps(KERNEL_ROOT)})
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

proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
print(proc.stdout[-40000:], flush=True)
if proc.stderr.strip():
    print("----- stderr (tail) -----", flush=True)
    print(proc.stderr[-20000:], flush=True)

print("{PAYLOAD_SENTINEL} returncode " + str(proc.returncode), flush=True)

# Re-emit the report on its own line so the driver log alone is enough to
# judge the run, even if artifact collection fails entirely.
report_path = OUT / "t4_smoke_report.json"
if report_path.exists():
    print("{RESULT_PREFIX}" + report_path.read_text(), flush=True)
else:
    print("{PAYLOAD_SENTINEL} NO REPORT WRITTEN", flush=True)

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
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle_t4_ci_leg": leg.name,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_driver(payloads: dict[str, dict], per_run_timeout: int,
                 isolation: dict[str, bool] | None = None) -> dict:
    """Kernel notebook that fans the payloads out one per GPU.

    ``isolation`` maps a payload to whether its virtualenv may see the
    Kaggle image's site-packages. It is per payload rather than per kernel
    because the legs that share a kernel do not share an answer: see
    ``Leg.system_site_packages``.
    """
    encoded = {name: _encode_bytes(json.dumps(nb).encode("utf-8"))
               for name, nb in payloads.items()}
    isolation = isolation or {}
    system_site = {name: bool(isolation.get(name, True)) for name in payloads}

    setup = f"""import base64, gzip, json, os, pathlib, subprocess, sys, threading, time
print("{DRIVER_SENTINEL} start", flush=True)

WORK = pathlib.Path("/kaggle/working")
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
N_GPU = max(1, len(GPUS))

for name, blob in PAYLOADS.items():
    (WORK / name).write_bytes(gzip.decompress(base64.b64decode(blob)))
print("{DRIVER_SENTINEL}_PAYLOADS " + json.dumps(sorted(PAYLOADS)), flush=True)
"""

    runner = f'''results = {{}}
lock = threading.Lock()

def _make_venv(idx, system_site):
    """Give a child its own site-packages. See this file's module docstring."""
    vdir = WORK / f"venv_{{idx}}"
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
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"

    kernel = "python3"
    made = _make_venv(idx, SYSTEM_SITE.get(name, True))
    if made:
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
    print(f"{DRIVER_SENTINEL}_DONE " + json.dumps({{name: results[name]}}),
          flush=True)

names = sorted(PAYLOADS)
threads = []
for i, name in enumerate(names):
    t = threading.Thread(target=run_one, args=(name, i % N_GPU, i), daemon=False)
    t.start()
    threads.append(t)
    time.sleep(5)
for t in threads:
    t.join()

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
for _pat in ("venv_*", "unsloth_compiled_cache", "t4_smoke_src",
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
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle_t4_ci": {"payloads": sorted(payloads)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel(payload_dir: Path, leg_names, *, unsloth_ref: str,
                 zoo_ref: str, extra_args: tuple[str, ...],
                 per_run_timeout: int, skip_reference: bool = False) -> dict:
    payloads = {}
    isolation = {}
    for leg in resolve(leg_names):
        name = f"t4_{leg.name}.ipynb"
        payloads[name] = build_payload_notebook(
            payload_dir, leg,
            unsloth_ref = unsloth_ref, zoo_ref = zoo_ref,
            extra_args = extra_args,
            reference = "" if skip_reference else None)
        isolation[name] = leg.system_site_packages
    return build_driver(payloads, per_run_timeout, isolation)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-dir", required = True)
    ap.add_argument("--out", required = True,
                    help = "output notebook. With --all-kernels this is a "
                           "prefix and the files are <prefix>1.ipynb, "
                           "<prefix>2.ipynb, ...")
    ap.add_argument("--legs",
                    help = "comma separated leg names; one per T4 of the "
                           "session. See legs.py")
    ap.add_argument("--all-kernels", action = "store_true",
                    help = "build every kernel in legs.KERNELS. This is what "
                           "the workflow uses, so the leg-to-kernel plan lives "
                           "in one place rather than being restated in YAML")
    ap.add_argument("--unsloth-ref", default = "main")
    ap.add_argument("--zoo-ref", default = "main")
    ap.add_argument("--smoke-args", default = "",
                    help = "extra args appended to EVERY leg's entry script. "
                           "Shared on purpose: the control and canary legs "
                           "must not differ in anything but versions")
    ap.add_argument("--skip-reference", action = "store_true",
                    help = "build with no band check at all. Only for the one "
                           "run that recaptures a reference")
    ap.add_argument("--per-run-timeout", type = int, default = 2400)
    args = ap.parse_args()

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

    for names, out in zip(plan, outputs):
        driver = build_kernel(
            Path(args.payload_dir), names,
            unsloth_ref = args.unsloth_ref, zoo_ref = args.zoo_ref,
            extra_args = tuple(args.smoke_args.split()),
            per_run_timeout = args.per_run_timeout,
            skip_reference = args.skip_reference)
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(driver, indent = 1), encoding = "utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) packing "
              f"{len(names)} leg(s): {', '.join(names)}")
    # The launcher needs one --notebook per kernel and the expected payload
    # count; both are consequences of the plan, so they are emitted here
    # rather than restated in the workflow.
    _github_output("notebooks", " ".join(f"--notebook {o}" for o in outputs))
    _github_output("payloads", str(sum(len(n) for n in plan)))
    return 0


def _github_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(f"{key}={value}\n")
    print(f"[build] {key}={value}")


if __name__ == "__main__":
    raise SystemExit(main())
