# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Build the Kaggle kernel notebook that runs the Unsloth GPU payload.

Sibling of ``.github/scripts/kaggle_t4_ci/build_kernel.py``, and deliberately
a separate file rather than a flag on it: that one carries a fixed list of
payload sources inline and runs ``run_t4_smoke.py``, and Unsloth needs a real
repository on the Kaggle side, which changes the shape of every cell. The
gate, the launcher and the notebook contract are shared; only the assembly
differs.

The differences worth knowing, all of them forced:

**A checkout, not a pip install.** Unsloth is installed by ``install.sh
--local``, which builds the frontend, creates the ``unsloth_studio`` venv and
fetches or builds llama.cpp. That needs the repository on disk, so the kernel
clones it at the ref under test. A pleasant side effect: the payload the
kernel runs is the payload at that ref, so payload and code under test cannot
drift apart.

**Nothing large under /kaggle/working.** That path is 19.5 GB and is also
what ``kernels output`` ships home. The home directory and ``/tmp`` share a
~1 TB overlay, so the checkout, the venv, torch, the models, llama.cpp and
every export live there, and only the evidence is written where Kaggle will
collect it.

**One payload, not two.** The notebook leg runs one payload per T4 because
its payload is a single-GPU training script and the second card is free.
Unsloth is a server, a browser and a llama.cpp process contending for four
CPU cores; a second copy of all that on the same box measures contention
rather than Unsloth. The second T4 is left idle on purpose.

**No per-child virtualenv.** ``install.sh`` makes its own, at
``$UNSLOTH_STUDIO_HOME/unsloth_studio``, and everything Unsloth-side runs
under that interpreter. The notebook leg's ``uv venv --seed`` dance exists to
keep two concurrent pip installs apart, and there is only one here.

Usage:
    python build_kernel.py --payload-dir tests/kaggle/studio_gpu \\
        --out kernel.ipynb --unsloth-ref <sha>
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import uuid
from pathlib import Path

DRIVER_SENTINEL = "KAGGLE_STUDIO_CI_DRIVER"
PAYLOAD_SENTINEL = "KAGGLE_STUDIO_CI_PAYLOAD"

# Shared with the notebook leg's launcher, which scrapes this prefix out of
# the executed notebook and the kernel log. Keeping it identical is what lets
# .github/scripts/kaggle_t4_ci/launch.py transport this payload's result
# without a line of change.
RESULT_PREFIX = "T4_SMOKE_REPORT "

PAYLOAD_NOTEBOOK = "studio_gpu.ipynb"
OUTPUT_NOTEBOOK = "studio_gpu_output.ipynb"

# Files the payload directory has to contain. Checked at build time so a
# rename that breaks the kernel fails on the runner, in seconds, rather than
# forty minutes into a GPU session.
PAYLOAD_FILES = (
    "run_studio_gpu.py",
    "gpu_assert.py",
    "studio_client.py",
    "train_canary.jsonl",
)


# Runs under the Unsloth venv's interpreter and reports what is actually
# importable there. Kept as a plain constant rather than spliced into a
# generated f-string cell: the notebook leg lost a whole GPU session to a
# cell that had been assembled out of nested quoting and did not parse.
_PROBE_SCRIPT = """
import importlib, json
out = {"versions": {}, "missing": []}
# unsloth before unsloth_zoo. zoo's __init__ refuses to import when
# find_spec("unsloth") is None, and probing zoo first has previously reported
# a dependency as missing on a session where it was installed and imported
# cleanly one line later.
for mod in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes",
            "unsloth", "unsloth_zoo", "fastapi", "uvicorn", "playwright"):
    try:
        m = importlib.import_module(mod)
        out["versions"][mod] = getattr(m, "__version__", "unknown")
    except Exception as exc:
        out["missing"].append(mod + ": " + type(exc).__name__ + ": " + str(exc))
try:
    import torch
    out["cuda"] = {
        "available": torch.cuda.is_available(),
        "count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    out["cuda"] = {"error": type(exc).__name__ + ": " + str(exc)}
print(json.dumps(out))
"""


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends = True),
    }


def build_payload_notebook(*, unsloth_ref: str, repo_url: str, payload_args: str) -> dict:
    """The notebook that installs Unsloth and runs the payload against it."""

    setup = f"""# Where everything lives.
#
# /kaggle/working is 19.5 GB and is the directory Kaggle ships back, so it
# holds evidence and nothing else. $HOME and /tmp share a ~1 TB overlay, and
# that is where the checkout, the venv, torch, the models and llama.cpp go.
# Getting this backwards fills the disk somewhere in the middle of the torch
# install and reports itself as an unrelated failure.
import json, os, pathlib, shutil, subprocess, sys, time

print("{PAYLOAD_SENTINEL} start", flush=True)

EVIDENCE = pathlib.Path("/kaggle/working/studio_gpu_out")
EVIDENCE.mkdir(parents=True, exist_ok=True)

def _pick_work_root():
    for candidate in (pathlib.Path.home() / "unsloth_studio_ci",
                      pathlib.Path("/tmp/unsloth_studio_ci")):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(candidate).free / 1e9
        except OSError:
            continue
        print(f"  candidate {{candidate}}: {{free_gb:.0f}} GB free", flush=True)
        if free_gb >= 60:
            return candidate
    raise SystemExit("no work root with room for an Unsloth install")

WORK = _pick_work_root()
REPO = WORK / "unsloth"
STUDIO_HOME = WORK / "studio_home"
HF_HOME = WORK / "hf"
for path in (STUDIO_HOME, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)

os.environ["UNSLOTH_STUDIO_HOME"] = str(STUDIO_HOME)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TMPDIR"] = str(WORK / "tmp")
pathlib.Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
# The installer otherwise ends by offering to launch Unsloth, which in a batch
# kernel is a prompt nobody answers.
os.environ["UNSLOTH_SKIP_AUTOSTART"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
# T4 is sm_75. If the prebuilt CUDA bundle is unavailable and setup.sh falls
# through to a source build, this stops it compiling every architecture NVIDIA
# has ever shipped inside a 45-minute session.
os.environ["UNSLOTH_LLAMA_CUDA_ARCHS"] = "75"

print("{PAYLOAD_SENTINEL} paths " + json.dumps({{
    "work": str(WORK), "studio_home": str(STUDIO_HOME),
    "free_gb": round(shutil.disk_usage(WORK).free / 1e9, 1),
}}), flush=True)


def fail_report(reason):
    # A failure of the INSTALLATION UNDER TEST is a payload failure, not an
    # infra one. Without a T4_SMOKE_REPORT the shared launcher classifies the
    # run as `infra` and the reporter exits 0, so an install.sh or dependency
    # regression -- the exact thing this workflow's path filter selects for --
    # would pass silently. Emitting the report first is what makes it red.
    # Infra outcomes (no GPU assigned, a clone that would not download) keep
    # the no-report path on purpose.
    print("{RESULT_PREFIX}" + json.dumps({{
        "label": "studio-gpu", "model": None, "passed": False,
        "failures": [reason], "assertions": [],
        "environment": {{}}, "config": {{}},
    }}), flush=True)


def sh(cmd, *, cwd=None, timeout=3600, check=True, label=""):
    print(f"  $ {{' '.join(cmd)}}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=dict(os.environ))
    print(f"  -> rc={{proc.returncode}} in {{time.time() - started:.0f}}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
        if check:
            raise SystemExit(f"{{label or cmd[0]}} failed rc={{proc.returncode}}")
    return proc
"""

    clone = f"""# The ref under test, pinned to a SHA by the workflow so a push landing
# mid-run cannot change what was measured. A blob-filtered clone: the repo's
# history is large and none of it is needed.
REPO_URL = {json.dumps(repo_url)}
REF = {json.dumps(unsloth_ref)}

if not REPO.exists():
    sh(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO)],
       timeout=1800, label="git clone")
sh(["git", "fetch", "--depth", "1", "origin", REF], cwd=str(REPO), timeout=1800,
   label="git fetch")
sh(["git", "checkout", "--force", "FETCH_HEAD"], cwd=str(REPO), timeout=600,
   label="git checkout")
head = sh(["git", "rev-parse", "HEAD"], cwd=str(REPO), timeout=60).stdout.strip()
print("{PAYLOAD_SENTINEL} checkout " + json.dumps({{"ref": REF, "head": head}}), flush=True)
"""

    install = f"""# The supported install. Not `pip install unsloth[studio]`: that extra is the
# server's dependency list and does not build the frontend, create the venv or
# put a llama.cpp on disk, all three of which this payload asserts against.
#
# Torch is NOT skipped here. Every other Unsloth workflow installs with
# --no-torch because its runner has no GPU to use one on; the training and
# export assertions need the real CUDA stack.
_install = sh(["bash", "install.sh", "--local"], cwd=str(REPO), timeout=5400, check=False,
              label="install.sh")
if _install.returncode != 0:
    fail_report(f"install.sh --local exited {{_install.returncode}}: the supported "
                f"installation of the checkout under test failed")
    raise SystemExit("install.sh failed")

VENV_PY = STUDIO_HOME / "unsloth_studio" / "bin" / "python"
if not VENV_PY.is_file():
    fail_report(f"install.sh --local succeeded but left no interpreter at {{VENV_PY}}")
    raise SystemExit(f"install.sh left no interpreter at {{VENV_PY}}")
print("{PAYLOAD_SENTINEL} venv " + str(VENV_PY), flush=True)
"""

    browser = f"""# Same Playwright install the repo's ubuntu UI job uses, into the venv the
# payload will run under. Chromium only: the cross-browser matrix is what
# studio-ui-smoke.yml is for, and firefox and webkit add several minutes here
# for coverage that has nothing to do with CUDA.
sh([str(VENV_PY), "-m", "pip", "install", "-q", "playwright>=1.45"], timeout=1200,
   label="pip install playwright")
sh([str(VENV_PY), "-m", "playwright", "install", "--with-deps", "chromium"],
   timeout=1800, label="playwright install")
"""

    verify = f"""# Fail fast and fail legibly. Without this, a missing piece surfaces as a
# traceback inside a child process forty minutes and one GPU session later.
#
# The probe runs under the STUDIO venv, not this notebook's kernel: the Kaggle
# base image has its own torch and its own everything, and asking it what is
# installed answers a question about the wrong interpreter.
PROBE = {json.dumps(_PROBE_SCRIPT)}

proc = subprocess.run([str(VENV_PY), "-c", PROBE], capture_output=True, text=True,
                      timeout=900, env=dict(os.environ))
print("{PAYLOAD_SENTINEL} probe rc=" + str(proc.returncode), flush=True)
print(proc.stdout[-4000:], flush=True)
if proc.returncode != 0:
    print(proc.stderr[-4000:], flush=True)
    fail_report("the dependency probe could not run under the installed Unsloth venv")
    raise SystemExit("dependency probe failed")

probe = json.loads(proc.stdout.strip().splitlines()[-1])
print("{PAYLOAD_SENTINEL} versions " + json.dumps(probe["versions"]), flush=True)
if probe["missing"]:
    print("{PAYLOAD_SENTINEL} MISSING " + json.dumps(probe["missing"]), flush=True)
    fail_report("the installed Unsloth venv is missing dependencies the payload needs: "
                + "; ".join(probe["missing"]))
    raise SystemExit("payload dependencies incomplete")
if not probe.get("cuda", {{}}).get("available"):
    print("{PAYLOAD_SENTINEL} NO CUDA " + json.dumps(probe.get("cuda")), flush=True)
    # Two different outcomes wear the same face here, and only one of them is
    # infra. No GPU assigned at all is Kaggle's doing and keeps the no-report
    # path, which the launcher files as `infra` and the reporter exits 0 for. A
    # GPU that nvidia-smi can see while the venv install.sh --local just built
    # cannot use it is a failure of the INSTALLATION UNDER TEST -- a CPU-only
    # torch resolved by the installer is how it happens -- and taking the
    # infra path there passes the exact CUDA install regression this workflow's
    # path filter selects for.
    try:
        _smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60)
        _visible = ([l for l in _smi.stdout.splitlines() if l.strip()]
                    if _smi.returncode == 0 else [])
    except Exception:
        _visible = []
    print("{PAYLOAD_SENTINEL} NO CUDA host_gpus " + json.dumps(_visible), flush=True)
    if _visible:
        fail_report("install.sh --local succeeded but the Unsloth venv cannot use CUDA "
                    "(torch.cuda.is_available() is False) on a box where nvidia-smi "
                    "reports " + str(len(_visible)) + " GPU(s): " + json.dumps(probe.get("cuda")))
    raise SystemExit("no CUDA device in the Unsloth venv, so there is nothing to test")

marker = STUDIO_HOME / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json"
info = {{}}
if marker.is_file():
    try:
        info = json.loads(marker.read_text())
    except Exception:
        info = {{"unreadable": True}}
print("{PAYLOAD_SENTINEL} llama_cpp " + json.dumps({{
    "marker": str(marker), "install_kind": info.get("install_kind"),
    "tag": info.get("tag"),
}}), flush=True)
"""

    run = f"""# Run the payload in a child of the Unsloth venv, not by importing it: it
# starts a server, spawns a browser and can be killed by either, and a child
# leaves this cell alive to report that.
cmd = [str(VENV_PY), str(REPO / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"),
       "--outdir", str(EVIDENCE),
       "--repo-root", str(REPO),
       "--studio-home", str(STUDIO_HOME)]
cmd += {json.dumps(payload_args.split())}
print("{PAYLOAD_SENTINEL} exec " + " ".join(cmd), flush=True)

proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
print(proc.stdout, flush=True)
if proc.stderr.strip():
    print("----- stderr (tail) -----", flush=True)
    print(proc.stderr[-20000:], flush=True)
print("{PAYLOAD_SENTINEL} returncode " + str(proc.returncode), flush=True)

# Re-emit the report on its own line so the kernel log alone is enough to
# judge the run even if the executed notebook never comes back.
report_path = EVIDENCE / "studio_gpu_report.json"
if report_path.exists():
    print("{RESULT_PREFIX}" + json.dumps(json.loads(report_path.read_text())), flush=True)
else:
    print("{PAYLOAD_SENTINEL} NO REPORT WRITTEN", flush=True)
print("{PAYLOAD_SENTINEL} complete rc=" + str(proc.returncode), flush=True)
# Deliberately does not raise: the report is the verdict, and papermill
# aborting here would lose the cells below it.
"""

    return {
        "cells": [
            _code_cell(setup),
            _code_cell(clone),
            _code_cell(install),
            _code_cell(browser),
            _code_cell(verify),
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


def build_driver(payload: dict, per_run_timeout: int) -> dict:
    """Kernel notebook that runs the payload notebook under papermill."""
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode("utf-8"))).decode("ascii")

    setup = f"""import base64, gzip, json, os, pathlib, subprocess, sys, time
print("{DRIVER_SENTINEL} start", flush=True)

WORK = pathlib.Path("/kaggle/working")
PAYLOAD = {json.dumps(encoded)}
(WORK / {json.dumps(PAYLOAD_NOTEBOOK)}).write_bytes(gzip.decompress(base64.b64decode(PAYLOAD)))

try:
    _smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=60)
    GPUS = [l for l in _smi.stdout.strip().splitlines() if l.strip()]
except Exception:
    GPUS = []
print("{DRIVER_SENTINEL}_GPUS " + json.dumps(GPUS), flush=True)
if not GPUS:
    print("{DRIVER_SENTINEL}_NO_GPU", flush=True)
"""

    runner = f"""src = WORK / {json.dumps(PAYLOAD_NOTEBOOK)}
out = WORK / {json.dumps(OUTPUT_NOTEBOOK)}
log = WORK / "studio_gpu_driver.log"

env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
# Both T4s stay visible. A Kaggle session has two, an Unsloth user on this
# hardware has two, and Unsloth's own device selection is part of what is
# under test; masking one would test a machine nobody has.

started = time.time()
rc, err = None, ""
try:
    with open(log, "wb") as fh:
        proc = subprocess.run(
            [sys.executable, "-m", "papermill", str(src), str(out),
             "-k", "python3", "--log-output", "--no-progress-bar"],
            env=env, stdout=fh, stderr=subprocess.STDOUT,
            timeout={per_run_timeout})
    rc = proc.returncode
except subprocess.TimeoutExpired:
    rc, err = -9, "papermill timed out after {per_run_timeout}s"
    # Whether this is infra or a code failure turns on ONE question: had the
    # payload itself started? Before that, the time went on the clone, the
    # install and the model downloads, and a slow Kaggle session teaches
    # nothing. After it, something under test hung, and with no report the
    # launcher would file that hang as unavailable infrastructure and exit 0.
    try:
        _tail = log.read_text(errors="replace")
    except OSError:
        _tail = ""
    if "{PAYLOAD_SENTINEL} exec" in _tail:
        print("{RESULT_PREFIX}" + json.dumps({{
            "label": "studio-gpu", "model": None, "passed": False,
            "failures": ["the payload was still running when the "
                         "{per_run_timeout}s driver deadline expired, so an "
                         "assertion hung rather than the session being slow to "
                         "start"],
            "assertions": [], "environment": {{}}, "config": {{}},
        }}), flush=True)
except Exception as exc:
    rc, err = -1, f"{{type(exc).__name__}}: {{exc}}"

print("{DRIVER_SENTINEL}_DONE " + json.dumps({{
    "returncode": rc, "seconds": round(time.time() - started, 1),
    "error": err, "output_exists": out.exists(),
}}), flush=True)
"""

    tail = f"""# Surface the child's tail inline so the kernel log alone is diagnosable if
# the executed notebook does not come back.
print("\\n===== studio_gpu (last 200 lines) =====", flush=True)
if log.exists():
    print("\\n".join(log.read_text(errors="replace").splitlines()[-200:]), flush=True)
else:
    print("NO LOG", flush=True)

# The payload notebook itself is a copy of what is already inside this kernel,
# and `kernels output` returns the whole of /kaggle/working over the wire.
try:
    (WORK / {json.dumps(PAYLOAD_NOTEBOOK)}).unlink()
except OSError:
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
            "kaggle_studio_ci": {"payload": PAYLOAD_NOTEBOOK, "output": OUTPUT_NOTEBOOK},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-dir", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--unsloth-ref", default = "main")
    ap.add_argument("--repo-url", default = "https://github.com/unslothai/unsloth.git")
    ap.add_argument("--payload-args", default = "", help = "extra args for run_studio_gpu.py")
    ap.add_argument("--per-run-timeout", type = int, default = 3900)
    args = ap.parse_args()

    payload_dir = Path(args.payload_dir)
    missing = [name for name in PAYLOAD_FILES if not (payload_dir / name).is_file()]
    if missing:
        raise SystemExit(f"payload dir {payload_dir} is missing: {', '.join(missing)}")

    payload = build_payload_notebook(
        unsloth_ref = args.unsloth_ref,
        repo_url = args.repo_url,
        payload_args = args.payload_args,
    )
    driver = build_driver(payload, args.per_run_timeout)
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(driver, indent = 1), encoding = "utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) for ref {args.unsloth_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
