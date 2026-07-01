#!/opt/unsloth-venv/bin/python
"""pip / uv shim for the Unsloth Docker notebook environment.

Installed earlier on PATH than the real tools so a notebook's `!pip install ...`
or `!uv pip install ...` cell becomes SAFE + idempotent instead of clobbering the
carefully-resolved cu128 torch/vLLM/transformers stack:

  * `transformers==X`  -> NOT installed into the base venv. The version X is
    recorded so the sidecar mechanism (unsloth_nb_compat) activates it for the
    model cells. The base stack stays intact.
  * torch / torchvision / torchaudio / triton / xformers / vllm / bitsandbytes /
    flashinfer / nvidia-* -> SKIPPED (the baked, ABI-matched versions are kept;
    a notebook reinstall here only ever breaks the GPU stack).
  * everything else (omegaconf, snac, causal-conv1d, ...) -> passed through to the
    real tool unchanged, so notebooks that genuinely need extra packages still
    get them.

Real tools are at /opt/unsloth-venv/bin/{pip,uv}; this shim invokes them by
absolute path so there is no recursion. `python -m pip` / `%pip` bypass PATH and
are not intercepted -- the driven `unsloth-run` handles those by parsing the
notebook directly.
"""

import os, re, sys, subprocess, tempfile

REAL = {"pip": "/opt/unsloth-venv/bin/pip", "uv": "/opt/unsloth-venv/bin/uv"}
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")

# Packages whose baked version must never be changed by a notebook install cell.
_KEEP = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "triton-rocm",
    "pytorch-triton",
    "xformers",
    "vllm",
    "bitsandbytes",
    "flashinfer",
    "flashinfer-python",
    "unsloth",
    "unsloth-zoo",
    "unsloth_zoo",
}
_KEEP_PREFIX = ("nvidia-", "nvidia_")
# pip/uv flags that consume the following token as a value (so we don't mistake
# that value for a requirement).
_VALUE_FLAGS = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--target",
    "-t",
    "--python",
    "-p",
    "--prefix",
    "--index-strategy",
    "--upgrade-package",
    "-P",
    "--no-binary",
    "--only-binary",
    "--platform",
    "--python-version",
    "--abi",
    "--implementation",
}
# Of those value-flags, the ones whose VALUE is itself an install target: a
# requirements file pulls real requirements. An index-url / find-links /
# constraint / target value is an option, not something to install.
_REQ_FILE_FLAGS = {"-r", "--requirement"}


def _canon(token):
    """Extract the lowercased distribution name from a requirement token, or None
    if the token is not a plain pkg spec (url / path / vcs / option)."""
    if token.startswith("-"):
        return None
    # PEP 508 direct reference: "name [extras] @ <url>" (e.g.
    # "torch @ https://.../torch.whl", "unsloth @ git+https://..."). The name is
    # at the front, so pull it out BEFORE the url/vcs guard below -- otherwise a
    # protected package pinned through a URL slips past _KEEP and reinstalls into
    # the base venv. A non-protected direct reference still returns its name and
    # is kept by the caller exactly as before (treated as an install target).
    _dref = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*@(?:\s|git\+|hg\+|bzr\+|svn\+|[a-z]+://)",
        token,
    )
    if _dref:
        return _dref.group(1).lower().replace("_", "-") or None
    if re.match(r"^[a-z]+\+", token) or "://" in token or token.startswith((".", "/")):
        return None  # vcs / url / local path -> let it pass through
    # strip extras and any version/marker tail
    name = re.split(r"[<>=!~\[\s;@]", token, 1)[0].strip()
    return name.lower().replace("_", "-") or None


def _version_pin(token):
    """Return the pinned version for a `pkg==X` token, else None."""
    m = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", token)
    return m.group(1) if m else None


def _filter_requirements_file(path):
    """Strip baked/protected packages out of a `-r` requirements file.

    Returns (path_to_use, recorded_transformers_version, dropped_specs). The same
    _KEEP / transformers rules the inline args get are applied to each requirement
    line, so a notebook `pip install -r reqs.txt` cannot overwrite the cu128 torch
    / vLLM / transformers stack with versions pinned inside the file. When nothing
    is protected, or the file cannot be read/written, the original path is returned
    unchanged. Comments, blank lines, option lines and nested `-r`/`-c` includes are
    kept verbatim (nested includes are passed through, i.e. filtered one level).
    """
    try:
        with open(path, encoding = "utf-8") as f:
            lines = f.readlines()
    except OSError:
        return path, None, []  # remote URL / unreadable -> let the real tool handle it
    out, dropped, recorded, changed = [], [], None, False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            out.append(line)  # comment / blank / option / nested include -> keep
            continue
        spec = stripped.split(" #", 1)[0].strip()  # drop any inline comment
        name = _canon(spec)
        if name is None:
            out.append(line)  # url / path / vcs / unparseable -> keep
            continue
        if name == "transformers":
            v = _version_pin(spec)
            if v and not recorded:
                recorded = v
            dropped.append(spec)
            changed = True
            continue
        if name in _KEEP or name.startswith(_KEEP_PREFIX):
            dropped.append(spec)
            changed = True
            continue
        out.append(line)
    if not changed:
        return path, None, []
    try:
        fd, tmp = tempfile.mkstemp(prefix = "unsloth-nb-req-", suffix = ".txt")
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.writelines(out)
    except OSError:
        return path, None, []  # can't write temp -> pass the file through unchanged
    return tmp, recorded, dropped


def main():
    tool = "uv" if os.path.basename(sys.argv[0]).startswith("uv") else "pip"
    argv = sys.argv[1:]

    # Only intercept inside a notebook kernel (UNSLOTH_NB_SHIM is set by the baked
    # IPython startup and by `unsloth-run`). EVERYWHERE else -- install.sh during
    # the image build, internal tooling, an interactive shell -- behave exactly
    # like the real tool, so we never disturb the build or system package mgmt.
    if os.environ.get("UNSLOTH_NB_SHIM") != "1":
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    # Locate the `install` verb (uv: `uv pip install ...`; pip: `pip install ...`).
    try:
        if tool == "uv":
            # skip a leading `pip` subcommand
            i = argv.index("install")
        else:
            i = argv.index("install")
    except ValueError:
        os.execv(REAL[tool], [REAL[tool]] + argv)  # not an install -> passthrough
        return

    head, tail = argv[: i + 1], argv[i + 1 :]
    keep_args, dropped, recorded = [], [], None
    has_target = False
    skip_next = False
    prev_flag = None
    for tok in tail:
        if skip_next:
            # The value of -r/--requirement pulls real requirements (a target); the
            # value of an index-url / find-links / constraint / etc. flag is an
            # option, not something to install.
            if prev_flag in _REQ_FILE_FLAGS:
                # Filter baked/protected packages out of the requirements file so a
                # notebook `pip install -r reqs.txt` cannot clobber the cu128 stack
                # or push transformers into the base venv.
                _req_path, _req_rec, _req_drp = _filter_requirements_file(tok)
                keep_args.append(_req_path)
                has_target = True
                if _req_rec and not recorded:
                    recorded = _req_rec
                dropped.extend(_req_drp)
            else:
                keep_args.append(tok)
            skip_next = False
            prev_flag = None
            continue
        # --flag=value form: pip accepts --requirement=reqs.txt / --index-url=URL
        # as a single token. Without this the token starts with "-", so it is kept
        # as an opaque option and a `-r` file is never filtered -- and worse, it
        # never counts as a target, so a cell whose only target is that file
        # silently no-ops and installs nothing.
        if tok.startswith("--") and "=" in tok:
            _flag, _, _val = tok.partition("=")
            if _flag in _VALUE_FLAGS:
                if _flag in _REQ_FILE_FLAGS:
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(_val)
                    keep_args.append(_flag + "=" + _req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                else:
                    keep_args.append(tok)  # option with inline value, not a target
                continue
        if tok in _VALUE_FLAGS:
            keep_args.append(tok)
            skip_next = True
            prev_flag = tok
            continue
        name = _canon(tok)
        if name is None:
            keep_args.append(tok)  # bare flag, or a positional url / path / vcs
            if not tok.startswith("-"):
                has_target = True  # standalone . / ./pkg / git+... / *.whl
            continue
        if name == "transformers":
            v = _version_pin(tok)
            if v:
                recorded = v
            dropped.append(tok)
            continue
        if name in _KEEP or name.startswith(_KEEP_PREFIX):
            dropped.append(tok)
            continue
        keep_args.append(tok)
        has_target = True  # a kept package spec

    if recorded:
        try:
            os.makedirs(os.path.dirname(MARKER), exist_ok = True)
            with open(MARKER, "w") as f:
                f.write(recorded)
            print(
                f"[unsloth-nb] notebook requested transformers=={recorded}; will "
                f"activate its sidecar for the model cells (base stack kept)."
            )
        except OSError:
            pass
    if dropped:
        print("[unsloth-nb] kept baked versions, skipped: " + " ".join(dropped))

    # Anything left to actually install? `has_target` was set during the scan for
    # a kept package spec, a positional url / path / vcs / editable target, or a
    # -r/--requirement file. A line carrying only baked packages plus option flags
    # (e.g. `--extra-index-url <url> torch`) leaves no target, so no-op instead of
    # exec'ing a bare `pip install --extra-index-url <url>` that would fail.
    if not has_target:
        print("[unsloth-nb] nothing to install after keeping the baked stack; ok.")
        return
    cmd = [REAL[tool]] + head + keep_args
    sys.stdout.flush()
    os.execv(REAL[tool], cmd)


if __name__ == "__main__":
    main()
