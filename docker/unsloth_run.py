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

import argparse, json, os, re, shutil, stat, subprocess, sys, tempfile, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import unsloth_nb_compat as compat
except Exception:
    compat = None

_PIN_RE = re.compile(r"transformers\s*==\s*([0-9][0-9A-Za-z.\-]*)")
_MODEL_RE = re.compile(r"""from_pretrained\(\s*['"]([^'"]+)['"]""")
_MODEL_NAME_RE = re.compile(r"""model_name\s*=\s*['"]([^'"]+)['"]""")

# Only an actual install invocation may supply the pin. A notebook's prose -- a
# commented-out legacy workaround, a docstring, a printed message -- names
# versions it does not install, and `want = pin or tier` below lets the first
# textual match outrank the model-name tier. A stale `# !pip install
# transformers==4.57.6` above a gemma-4-12b cell therefore clamps up to the
# lowest eligible sidecar (5.5.0) instead of the 5.10.2 the model needs, and the
# kernel is launched on it: nothing installs, so the pip shim never gets to
# correct the marker or PYTHONPATH. Commenting out an install line is a routine
# notebook edit, so match the invocation instead of the whole cell source.
#
# Covers `pip`/`pip3`, `!`/`%` magics, bare lines (%%bash cells), `uv pip`,
# `<interpreter> -m pip` (`python3`, `{sys.executable}`, ...), options before
# `install`, and any indent -- notebooks guard installs inside if/else, and
# Unsloth's own install cells do exactly that.
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
    """Drop a trailing `# ...` comment from a shell/magic line.

    Only a `#` at the start of a token counts, so the `#egg=`/`#subdirectory=`
    fragment of a `git+https://...` requirement survives; quoted `#` is left
    alone too."""
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


# A triple-quoted body is data, not code: a cell that builds a setup script or
# documents a workaround can contain a line reading `!pip install transformers==X`
# or a `from_pretrained("...")` that never executes. Blank those regions, keeping
# newlines so line numbers and the continuation logic below are unaffected.
_TRIPLE_RE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')


def _live_source(src):
    """Cell source with triple-quoted bodies and comments blanked out.

    Single-quoted strings are deliberately left intact: the model name this scans
    for lives inside one (`from_pretrained("unsloth/...")`)."""
    blanked = _TRIPLE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)
    return "\n".join(_strip_comment(line) for line in blanked.splitlines())


def _install_lines(src):
    """The install-invocation text of a cell source, backslash continuations kept.

    Real install cells split one command over several lines (`!uv pip install \\`
    + a continuation carrying `"transformers==4.56.2"`), so a continued line stays
    part of the invocation."""
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
    """Return (pinned_transformers, first_model_name) from the notebook source."""
    pin = model = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        # Same reasoning as the pin: a commented-out or triple-quoted model
        # reference names a model the notebook does not load, and the FIRST match
        # wins, so a stale `# from_pretrained("unsloth/qwen3.5-4b")` above the real
        # gemma-4-12b call picks the lower sidecar tier. Swapping models by
        # commenting the old line out is the most routine notebook edit there is.
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
    """Create `path` owned by the nearest existing ancestor's owner.

    mkdir(2) gives the new directory the CALLER's uid/gid; only the setgid bit
    carries anything down from the parent. The container runs as root while
    `-v $PWD:/workspace` is the host user's own tree, so `--out sub/dir/x.ipynb`
    into a directory that does not exist yet leaves them a root-owned directory
    they cannot write. _stage_metadata then derives the OUTPUT's owner from that
    same just-created directory, so the notebook is root-owned too and both ends
    of the path are unusable from the host.
    """
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
    # Outermost first, so a partial failure still fixes what it can.
    for created in reversed(missing):
        try:
            os.chown(created, anchor.st_uid, anchor.st_gid)
        except (OSError, AttributeError):
            pass


def _stage_metadata(staged, dest):
    """Give the staged output the metadata the destination must end up with.

    mkstemp() creates 0600 and nbconvert truncates that same inode, so the mode
    survives os.replace(). Under the documented `-v $PWD:/workspace` layout the
    container is root and the host user is not, so publishing as-is hands them an
    output they cannot read, and overwriting an existing one drops its mode and
    owner. Reuse the destination's metadata, else the umask-derived mode a plain
    write would have produced. Best effort: a filesystem that refuses chmod/chown
    must not cost the user their executed notebook.
    """
    try:
        st = os.stat(dest)
    except OSError:
        # New output: no destination to copy from, so take the umask-derived mode
        # a plain write would have produced and the OWNER of the directory it
        # lands in. Under `-v $PWD:/workspace` that directory belongs to the host
        # user, so without the chown they get a root-owned file they can read but
        # not edit, which is the same complaint as the existing-output case.
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

    # Materialise the notebook for nbconvert. With --out, stage input + result as
    # temp files next to the destination (same dir => atomic os.replace publish)
    # and publish only on success, so a failed run can't destroy the old output.
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
            _stage_metadata(publish_from, out_path)
            try:
                os.replace(publish_from, out_path)
            except OSError:
                # `-v $PWD/out.ipynb:/workspace/out.ipynb` bind-mounts the OUTPUT
                # FILE, which makes the destination a mount point, and rename(2)
                # onto one returns EBUSY even though the file itself is perfectly
                # writable. The executed notebook is already complete on disk at
                # this point, so failing here threw away the entire run (the
                # cleanup below deletes the staging file) for a publish step that
                # can just as well write through the existing inode -- which is
                # exactly what such a mount needs, since the host sees the inode,
                # not the directory entry. Not atomic, unlike the rename, so it is
                # only the fallback.
                try:
                    with open(publish_from, "rb") as staged, open(out_path, "wb") as live:
                        shutil.copyfileobj(staged, live)
                except OSError:
                    # Neither publish worked. Keep the result rather than delete
                    # it, and say where it is; a notebook run can be hours long.
                    if publish_from in tmp_files:
                        tmp_files.remove(publish_from)
                    print(
                        f"[unsloth-run] could not publish to {out_path}; "
                        f"the executed notebook is at {publish_from}",
                        file = sys.stderr,
                    )
                    raise
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
