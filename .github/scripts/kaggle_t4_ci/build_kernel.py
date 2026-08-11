# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Build the Kaggle kernel notebook that runs the T4 smoke test.

Two layers, because a Kaggle GPU session is 2xT4 and the smoke payload is
written for one card:

**Payload notebook** -- one per GPU. Installs Unsloth at the ref under test,
materialises ``run_t4_smoke.py`` + ``determinism.py`` + the canary dataset
from inlined copies, runs the test, prints a machine-readable result line.

**Driver notebook** -- one per kernel. Carries the payloads inline (gzip +
base64) so the kernel needs no dataset attachment and no network fetch of
our sources, gives each payload its own virtualenv and its own GPU, and runs
them concurrently under papermill.

Everything the kernel needs travels inside the notebook. There is no
checkout on the Kaggle side.

Three hard-won details are load-bearing and are NOT safe to simplify away:

1. **Per-child virtualenv.** Both payloads pip-install a torch/transformers
   stack. Sharing one site-packages lets concurrent installs corrupt each
   other and fabricates import failures in payloads that pass when run
   alone.
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

Usage:
    python build_kernel.py --payload-dir tests/kaggle/t4_smoke \\
        --out kernel.ipynb --count 2
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import uuid
from pathlib import Path

DRIVER_SENTINEL = "KAGGLE_T4_CI_DRIVER"
PAYLOAD_SENTINEL = "KAGGLE_T4_CI_PAYLOAD"
RESULT_PREFIX = "T4_SMOKE_REPORT "

# Files copied verbatim from the repo into the kernel.
PAYLOAD_FILES = ("run_t4_smoke.py", "determinism.py", "canary_dataset.jsonl")


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


def build_payload_notebook(
    payload_dir: Path,
    *,
    label: str,
    unsloth_ref: str,
    zoo_ref: str,
    smoke_args: str,
    reference: str,
) -> dict:
    """A single-GPU notebook that installs Unsloth and runs the smoke test."""
    files = {}
    for name in PAYLOAD_FILES:
        path = payload_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        files[name] = _encode_bytes(path.read_bytes())
    if reference:
        ref_path = payload_dir / "references" / reference
        if ref_path.exists():
            files[f"references/{reference}"] = _encode_bytes(ref_path.read_bytes())

    install = f"""# Install the code under test.
#
# unsloth_zoo first and WITH deps, then unsloth --no-deps on top, so the
# editable-style overlay does not fight the dependency set zoo resolved.
# This mirrors what unsloth's own install path does.
import subprocess, sys
print("{PAYLOAD_SENTINEL} install start", flush=True)

def pip(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("  $ " + " ".join(cmd[3:]), flush=True)
    # github.com occasionally 500s on a git fetch; a single upstream blip
    # must not be reported as a notebook regression.
    for attempt in (1, 2, 3):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        print(f"  install attempt {{attempt}} failed rc={{proc.returncode}}",
              flush=True)
        print("  " + proc.stderr.strip()[-1500:], flush=True)
        if attempt == 3:
            raise SystemExit(f"pip install failed: {{args}}")

pip("unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zoo_ref}")
pip("--no-deps", "unsloth @ git+https://github.com/unslothai/unsloth@{unsloth_ref}")

# bitsandbytes explicitly, and this is not belt-and-braces.
#
# `unsloth` goes in with --no-deps (so the overlay does not fight the
# dependency set zoo resolved), and unsloth_zoo's own dependency set does
# not pull bitsandbytes. The Kaggle base image does not carry it either,
# and the child venv is --system-site-packages, so nothing supplied it.
# The result was `ModuleNotFoundError: No module named 'bitsandbytes'` at
# `import unsloth`, on a session that had otherwise set itself up perfectly.
# The 4-bit load this test performs cannot happen without it.
pip("bitsandbytes")
print("{PAYLOAD_SENTINEL} install done", flush=True)
"""

    verify = f"""# Fail fast, and fail legibly.
#
# Without this, a missing dependency surfaces as a traceback buried in a
# child process's captured stdout, forty minutes and one GPU session later.
# Here it surfaces immediately, named, in the driver log.
import importlib, json
# Everything above was pip-installed AFTER this interpreter started, and the
# import system caches the directory listing of each sys.path entry. Without
# this, a just-installed package can be invisible to the very next import.
importlib.invalidate_caches()

missing = []
versions = {{}}
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
for mod in ("torch", "transformers", "trl", "peft", "datasets",
            "bitsandbytes", "unsloth", "unsloth_zoo"):
    try:
        m = importlib.import_module(mod)
        versions[mod] = getattr(m, "__version__", "unknown")
    except Exception as exc:
        missing.append(f"{{mod}}: {{type(exc).__name__}}: {{exc}}")
print("{PAYLOAD_SENTINEL} versions " + json.dumps(versions), flush=True)
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

    materialise = f"""# Materialise the test sources carried inside this notebook.
import base64, gzip, json, os, pathlib
FILES = {json.dumps(files)}
ROOT = pathlib.Path("/kaggle/working/t4_smoke_src")
for name, blob in FILES.items():
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.decompress(base64.b64decode(blob)))
print("{PAYLOAD_SENTINEL} sources " + json.dumps(sorted(FILES)), flush=True)
"""

    # The reference path is built at RUNTIME, from the notebook's own ROOT,
    # and appended as two list elements.
    #
    # It used to be a single f-string ' --reference "{{ROOT}}/..."' spliced
    # into the middle of a Python list literal, which is shell syntax in a
    # place that wanted Python. On a real T4 that cell died with
    #   "--outdir", str(OUT), "--label", "gpu0" --reference "{{ROOT}}/..."]
    #   SyntaxError: invalid syntax. Perhaps you forgot a comma?
    # before a single training step ran, and the doubled braces meant ROOT
    # would not have been substituted even if it had parsed. The workflow
    # always passes --reference, so this was on every path. Cheap and
    # airtight to prevent: test_generated_cells_compile.
    ref_line = (
        f'cmd += ["--reference", str(ROOT / "references" / ' f"{json.dumps(reference)})]\n"
        if reference
        else ""
    )
    run = f"""# Run the smoke test in a child process.
#
# A child, not an import, for two reasons: the determinism setup has to run
# before torch is imported (and papermill has already imported plenty), and
# a hard crash then leaves this cell alive to report it.
import json, os, pathlib, subprocess, sys
ROOT = pathlib.Path("/kaggle/working/t4_smoke_src")
OUT = pathlib.Path("/kaggle/working/t4_smoke_out_{label}")
OUT.mkdir(parents=True, exist_ok=True)

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
env["UNSLOTH_DISABLE_STATISTICS"] = "1"

cmd = [sys.executable, str(ROOT / "run_t4_smoke.py"),
       "--outdir", str(OUT), "--label", "{label}"]
{ref_line}cmd += {json.dumps(smoke_args.split())}
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
            _code_cell(install),
            _code_cell(verify),
            _code_cell(materialise),
            _code_cell(run),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_driver(payloads: dict[str, dict], per_run_timeout: int) -> dict:
    """Kernel notebook that fans the payloads out one per GPU."""
    encoded = {name: _encode_bytes(json.dumps(nb).encode("utf-8")) for name, nb in payloads.items()}

    setup = f"""import base64, gzip, json, os, pathlib, subprocess, sys, threading, time
print("{DRIVER_SENTINEL} start", flush=True)

WORK = pathlib.Path("/kaggle/working")
PAYLOADS = {json.dumps(encoded)}

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

def _make_venv(idx):
    """Give a child its own site-packages. See this file's module docstring."""
    vdir = WORK / f"venv_{{idx}}"
    try:
        uv = subprocess.run(["which", "uv"], capture_output=True,
                            text=True).stdout.strip()
        if not uv:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                           check=True, timeout=900)
            uv = "uv"
        subprocess.run([uv, "venv", str(vdir), "--seed",
                        "--system-site-packages"], check=True, timeout=900)
        py = vdir / "bin" / "python"
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py),
                        "ipykernel"], check=True, timeout=900)
        kname = f"t4ci{{idx}}"
        subprocess.run([str(py), "-m", "ipykernel", "install", "--user",
                        "--name", kname], check=True, timeout=600)
        print(f"{DRIVER_SENTINEL}_VENV " + json.dumps(
            {{"idx": idx, "kernel": kname}}), flush=True)
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
    made = _make_venv(idx)
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
             "huggingface_tokenizers_cache", "*/trainer_run*"):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-dir", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument(
        "--count", type = int, default = 2, help = "payloads per kernel; 2 uses both T4s of a session"
    )
    ap.add_argument("--unsloth-ref", default = "main")
    ap.add_argument("--zoo-ref", default = "main")
    ap.add_argument(
        "--reference", default = "", help = "filename under payload-dir/references to band-check"
    )
    ap.add_argument("--smoke-args", default = "", help = "extra args forwarded to run_t4_smoke.py")
    ap.add_argument("--per-run-timeout", type = int, default = 2400)
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    payload_dir = Path(args.payload_dir)
    payloads = {}
    for i in range(args.count):
        label = f"gpu{i}"
        payloads[f"t4_smoke_{label}.ipynb"] = build_payload_notebook(
            payload_dir,
            label = label,
            unsloth_ref = args.unsloth_ref,
            zoo_ref = args.zoo_ref,
            smoke_args = args.smoke_args,
            reference = args.reference,
        )

    driver = build_driver(payloads, args.per_run_timeout)
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(driver, indent = 1), encoding = "utf-8")
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) packing "
        f"{len(payloads)} payload(s): {', '.join(sorted(payloads))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
