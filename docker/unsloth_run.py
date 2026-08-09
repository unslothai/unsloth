#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""unsloth-run: execute an unslothai/notebooks notebook unchanged, headless.

The robust driven path for the Docker image: it reads the notebook, figures out
which transformers version it wants (its install-cell pin, else the model-name
tier), launches the kernel with that sidecar on PYTHONPATH so the whole kernel
process uses a coherent transformers, and executes every cell with nbconvert.
The notebook's own install cell still runs through the pip/uv shim, so it is safe
and idempotent (the baked torch/vLLM stack is never clobbered).

Usage:
  unsloth-run <notebook.ipynb | URL> [--out OUT.ipynb] [--timeout SECONDS]
              [--transformers X.Y.Z]   # force a version, skip auto-detect

A raw github URL (raw.githubusercontent.com/.../nb/Foo.ipynb) is fetched first.
"""

import argparse, json, os, re, shutil, subprocess, sys, tempfile, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import unsloth_nb_compat as compat
except Exception:
    compat = None

_PIN_RE = re.compile(r"transformers\s*==\s*([0-9][0-9A-Za-z.\-]*)")
_MODEL_RE = re.compile(r"""from_pretrained\(\s*['"]([^'"]+)['"]""")
_MODEL_NAME_RE = re.compile(r"""model_name\s*=\s*['"]([^'"]+)['"]""")


def _load(path_or_url):
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as r:  # nosec - user-provided nb
            data = r.read().decode()
        return json.loads(data)
    with open(path_or_url) as f:
        return json.load(f)


def _scan(nb):
    """Return (pinned_transformers, first_model_name) from the notebook source."""
    pin = model = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if pin is None:
            m = _PIN_RE.search(src)
            if m:
                pin = m.group(1)
        if model is None:
            m = _MODEL_RE.search(src) or _MODEL_NAME_RE.search(src)
            if m:
                model = m.group(1)
    return pin, model


def main():
    ap = argparse.ArgumentParser(prog = "unsloth-run")
    ap.add_argument("notebook")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument("--transformers", dest = "tf")
    args = ap.parse_args()

    nb = _load(args.notebook)
    pin, model = _scan(nb)
    want = args.tf or pin or (compat.tier_for_model(model) if compat else None)
    sidecar = compat.sidecar_for(want) if (compat and want) else None

    # Materialise the notebook for nbconvert. With --out, stage input + result as
    # temp files next to the destination (same dir => atomic os.replace publish)
    # and publish only on success, so a failed run can't destroy the old output.
    tmp_dir = None
    tmp_files = []
    publish_from = None
    if args.out:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok = True)
        fd, src_path = tempfile.mkstemp(prefix = ".unsloth-run-in-", suffix = ".ipynb", dir = out_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(nb, f)
        tmp_files.append(src_path)
        fd, publish_from = tempfile.mkstemp(
            prefix = ".unsloth-run-out-", suffix = ".ipynb", dir = out_dir
        )
        os.close(fd)
        tmp_files.append(publish_from)
    elif args.notebook.startswith(("http://", "https://")):
        tmp_dir = tempfile.mkdtemp()
        src_path = os.path.join(tmp_dir, os.path.basename(args.notebook.split("?")[0]))
        with open(src_path, "w") as f:
            json.dump(nb, f)
        out_path = src_path
    else:
        src_path = args.notebook
        out_path = src_path

    env = dict(os.environ)
    env["UNSLOTH_NB_SHIM"] = "1"  # enable safe-install for the notebook's cells
    # Per-run marker unless the caller pinned one: the shared default would leak
    # this run's transformers pin into concurrent/later runs. An empty marker
    # reads as "no pin", so pre-creating it is safe.
    marker = env.get("UNSLOTH_NB_TF_MARKER")
    if not marker:
        fd, marker = tempfile.mkstemp(prefix = ".unsloth-run-tfmarker-")
        os.close(fd)
        env["UNSLOTH_NB_TF_MARKER"] = marker
        tmp_files.append(marker)
    # The pip/uv shim writes the marker; pre-seed it too so the kernel agrees.
    if want:
        os.makedirs(os.path.dirname(marker) or ".", exist_ok = True)
        open(marker, "w").write(want)
    if sidecar:
        env["PYTHONPATH"] = sidecar + os.pathsep + env.get("PYTHONPATH", "")
        print(f"[unsloth-run] transformers {want} -> sidecar {sidecar}")
    elif want:
        print(f"[unsloth-run] transformers {want}: no sidecar (using base venv's newest)")
    else:
        print("[unsloth-run] no transformers pin/model tier detected; using base venv")

    nbconvert_out = publish_from if publish_from is not None else out_path
    cmd = [
        "/opt/unsloth-venv/bin/jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        src_path,
        "--output",
        os.path.basename(nbconvert_out),
        "--output-dir",
        os.path.dirname(os.path.abspath(nbconvert_out)) or ".",
    ]
    print(
        "[unsloth-run] executing:",
        os.path.basename(args.notebook.split("?")[0]) if args.out else os.path.basename(src_path),
    )
    try:
        rc = subprocess.call(cmd, env = env)
        if rc == 0 and publish_from is not None:
            os.replace(publish_from, out_path)
    finally:
        # Clean up the temp dir and any staging files (already gone when published).
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors = True)
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
