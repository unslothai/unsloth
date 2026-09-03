#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""unsloth-run: execute an unslothai/notebooks notebook unchanged, headless.

Resolves the transformers version the notebook wants (install-cell pin, else the
model-name tier), launches the kernel with that sidecar on PYTHONPATH so the whole
kernel process is coherent, and executes every cell with nbconvert.

Usage:
  unsloth-run <notebook.ipynb | URL> [--out OUT.ipynb] [--timeout SECONDS]
              [--transformers X.Y.Z]   # force a version, skip auto-detect
"""

import argparse, json, os, re, shutil, stat, subprocess, sys, tempfile, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import unsloth_nb_compat as compat
except Exception:
    compat = None

_PIN_RE = re.compile(r"transformers\s*==\s*([0-9][0-9A-Za-z.\-]*)")
_MODEL_RE = re.compile(r"""from_pretrained\(\s*['"]([^'"]+)['"]""")
_MODEL_NAME_RE = re.compile(r"""model_name\s*=\s*['"]([^'"]+)['"]""")

# Only an actual install invocation may supply the pin: `want = pin or tier` below
# lets the first textual match outrank the model tier, so a commented-out install line
# launches the kernel on the wrong sidecar and nothing runs to correct it.
_INSTALL_RE = re.compile(
    r"""^[ \t]*(?![ \t]*\#)[!%]?[ \t]*
        (?: uv (?:[ \t]+-{1,2}\S+)* [ \t]+ )?
        (?: \S+ [ \t]+ -m [ \t]+ )?
        pip[0-9.]* [ \t]+
        (?: -{1,2}\S+ [ \t]+ )*
        install (?: [ \t] | $ )""",
    re.VERBOSE,
)


def _strip_comment(line):
    """Drop a trailing `# ...` from a shell/magic line. Only a `#` starting a token
    counts, so a `git+https://...#egg=` fragment survives, as does a quoted #."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


# A triple-quoted body is data, not code. Blanked keeping newlines, so the
# continuation logic below is unaffected.
_TRIPLE_RE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')


def _live_source(src):
    """Triple-quoted bodies and comments blanked out. Single-quoted strings stay
    intact: the model name this scans for lives inside one."""
    blanked = _TRIPLE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)
    return "\n".join(_strip_comment(line) for line in blanked.splitlines())


def _install_lines(src):
    kept, cont = [], False
    for line in src.splitlines():
        if cont or _INSTALL_RE.match(line):
            code = _strip_comment(line)
            kept.append(code)
            cont = code.rstrip().endswith("\\")
        else:
            cont = False
    return "\n".join(kept)


def _load(path_or_url):
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as r:  # nosec - user-provided nb
            data = r.read().decode()
        return json.loads(data)
    with open(path_or_url) as f:
        return json.load(f)


def _scan(nb):
    """(pinned_transformers, first_model_name); dead code must not count, as above."""
    pin = model = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = _live_source("".join(cell.get("source", [])))
        if pin is None:
            m = _PIN_RE.search(_install_lines(src))
            if m:
                pin = m.group(1)
        if model is None:
            m = _MODEL_RE.search(src) or _MODEL_NAME_RE.search(src)
            if m:
                model = m.group(1)
    return pin, model


def _makedirs_as_host(path):
    """Create `path` owned by the nearest existing ancestor. mkdir(2) uses the CALLER's
    uid/gid and only setgid carries down, so a new `--out sub/dir/` would be root-owned
    and _stage_metadata would then give the OUTPUT that owner too."""
    path = os.path.abspath(path)
    missing = []
    probe = path
    while not os.path.isdir(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    os.makedirs(path, exist_ok = True)
    if not missing:
        return
    try:
        anchor = os.stat(probe)
    except OSError:
        return
    for created in reversed(missing):
        try:
            os.chown(created, anchor.st_uid, anchor.st_gid)
        except (OSError, AttributeError):
            pass


def _stage_metadata(staged, dest):
    """Give the staged output the metadata the destination must end up with: mkstemp()
    creates 0600, nbconvert truncates that same inode, and os.replace carries it onto
    the destination. Best effort."""
    try:
        st = os.stat(dest)
    except OSError:
        # new output: the umask-derived mode a plain write would have produced
        try:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(staged, 0o666 & ~umask)
        except OSError:
            pass
        try:
            _dir = os.stat(os.path.dirname(os.path.abspath(dest)) or ".")
            os.chown(staged, _dir.st_uid, _dir.st_gid)
        except (OSError, AttributeError):
            pass
        return
    try:
        os.chmod(staged, stat.S_IMODE(st.st_mode))
    except OSError:
        pass
    try:
        os.chown(staged, st.st_uid, st.st_gid)
    except (OSError, AttributeError):
        pass


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

    tmp_dir = None
    tmp_files = []
    publish_from = None
    if args.out:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path) or "."
        _makedirs_as_host(out_dir)
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
    env["UNSLOTH_NB_SHIM"] = "1"
    # per-run marker: the shared default leaks this run's pin into concurrent runs
    marker = env.get("UNSLOTH_NB_TF_MARKER")
    if not marker:
        fd, marker = tempfile.mkstemp(prefix = ".unsloth-run-tfmarker-")
        os.close(fd)
        env["UNSLOTH_NB_TF_MARKER"] = marker
        tmp_files.append(marker)
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
            _stage_metadata(publish_from, out_path)
            try:
                os.replace(publish_from, out_path)
            except OSError:
                # rename(2) onto a bind-mounted OUTPUT FILE returns EBUSY even though
                # the file is writable, and such a mount needs the inode write anyway
                try:
                    with open(publish_from, "rb") as staged, open(out_path, "wb") as live:
                        shutil.copyfileobj(staged, live)
                except OSError:
                    # keep the result rather than delete it: a run can be hours long
                    if publish_from in tmp_files:
                        tmp_files.remove(publish_from)
                    print(
                        f"[unsloth-run] could not publish to {out_path}; "
                        f"the executed notebook is at {publish_from}",
                        file = sys.stderr,
                    )
                    raise
    finally:
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
