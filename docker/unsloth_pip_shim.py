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

import os, re, sys, tempfile

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
    "-e",
    "--editable",
}
# Of those value-flags, the ones whose VALUE is itself an install target: a
# requirements file pulls real requirements. An index-url / find-links /
# constraint / target value is an option, not something to install.
_REQ_FILE_FLAGS = {"-r", "--requirement"}
# Constraint files are not install targets, but pip applies their pins during
# resolution, so a `-c constraints.txt` that pins torch/transformers/etc. can
# still downgrade or reinstall a baked package when another target pulls it in.
# Filter protected packages out of them the same way as requirement files.
_CONSTRAINT_FILE_FLAGS = {"-c", "--constraint"}
# -e/--editable <path|url|vcs> takes the NEXT token as its target (pip:
# `-e, --editable <path/url>`), and that target is a real install target. A
# protected editable (e.g. `-e git+https://.../unsloth.git#egg=unsloth`) must
# drop BOTH the flag and its value; dropping the value alone leaves pip a
# dangling `-e` that swallows the next kept package and fails the whole cell.
_EDITABLE_FLAGS = {"-e", "--editable"}
# -P/--upgrade-package <name> is uv's selective-upgrade flag: naming a baked
# package (e.g. `uv pip install -P torch peft`) lets an ordinary install target
# refresh that package and clobber the pinned stack. Filter its value through
# _KEEP too. Unlike -e it is not itself an install target (no has_target).
_UPGRADE_PKG_FLAGS = {"-P", "--upgrade-package"}
# Short value-flags pip/uv accept in the ATTACHED form, i.e. the 2-char flag
# glued to its value in one token: `-rreqs.txt`, `-cconstraints.txt`, `-epath`,
# `-Pname`. The scanner splits the flag from the value so the value is filtered
# (requirement/constraint file) or classified (-e/-P) instead of falling through
# as an opaque option -- otherwise an attached `-r`-only cell no-ops and an
# attached `-c`/`-e`/`-P` value bypasses _KEEP.
_ATTACHED_SHORT_FLAGS = {"-r", "-c", "-e", "-P"}


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
        # A VCS / URL install can still name a protected package via the legacy
        # `#egg=NAME` (or `&egg=NAME`) fragment, e.g.
        # `git+https://github.com/unslothai/unsloth.git#egg=unsloth`. Pull that
        # name out so _KEEP can drop it; otherwise the shim would exec the URL
        # and reinstall a baked package into the venv. A non-protected egg name
        # is returned too, but the caller keeps it as a normal target either way.
        _egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", token)
        if _egg:
            return _egg.group(1).lower().replace("_", "-") or None
        # A direct wheel URL or local wheel path still names its distribution in
        # the PEP 427 filename ({distribution}-{version}-...-...-....whl), so a
        # bare `pip install https://.../torch-2.11.0+cu128-...whl` would slip a
        # protected package past _KEEP as an opaque positional and reinstall the
        # baked torch. Dashes cannot appear inside the distribution component (a
        # run of -_. normalises to a single -), so the leading dash-split of the
        # basename is the distribution name; pull it so _KEEP can drop it. A
        # non-protected wheel returns its name and the caller keeps the token.
        _whl = re.search(r"([^/\\#?]+)\.whl(?:[#?]|$)", token)
        if _whl:
            dist = _whl.group(1).split("-", 1)[0].strip().lower().replace("_", "-")
            if dist:
                return dist
        return None  # vcs / url / local path -> let it pass through
    # strip extras and any version/marker tail
    name = re.split(r"[<>=!~\[\s;@]", token, 1)[0].strip()
    return name.lower().replace("_", "-") or None


def _version_pin(token):
    """Return the pinned version for a `pkg==X` token, else None."""
    m = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", token)
    return m.group(1) if m else None


def _classify_flag_target(spec):
    """Classify the value that rides on -e/--editable or -P/--upgrade-package.

    Returns ("drop", version_or_None) when the value names a protected package
    (so the flag+value pair must be dropped, closing the same bypass the bare
    positional spec closes) or ("keep", None) when it is safe to forward.
    transformers is reported as "drop" with any pinned version so its sidecar
    marker is still recorded, mirroring the bare-spec handling in main()."""
    name = _canon(spec)
    if name == "transformers":
        return "drop", _version_pin(spec)
    if name is not None and (name in _KEEP or name.startswith(_KEEP_PREFIX)):
        return "drop", None
    return "keep", None


def _parse_include(stripped):
    """If `stripped` is an `-r`/`--requirement`/`-c`/`--constraint` include,
    return (flag, target_path, inline_comment_or_None); else (None, None, None)."""
    body, sep, comment = stripped.partition(" #")
    body = body.rstrip()
    comment = ("#" + comment) if sep else None
    for flag in ("-r", "--requirement", "-c", "--constraint"):
        target = None
        if body == flag or body.startswith(flag + " "):
            target = body[len(flag) :].strip()
        elif body.startswith(flag + "="):
            target = body[len(flag) + 1 :].strip()
        elif not flag.startswith("--") and body.startswith(flag) and len(body) > len(flag):
            target = body[len(flag) :].strip()  # attached short form, e.g. `-rextras.txt`
        else:
            continue
        return flag, (target or None), comment
    return None, None, None


def _parse_editable(stripped):
    """If `stripped` is an `-e`/`--editable` install line, return
    (flag, target, inline_comment_or_None); else (None, None, None).

    Handles the separated (`-e <t>` / `--editable <t>`), attached (`-e<t>`),
    long inline (`--editable=<t>`) and short inline (`-e=<t>`) forms pip accepts
    from a requirement file, so a protected editable there is dropped exactly
    like the command-line -e case."""
    body, sep, comment = stripped.partition(" #")
    body = body.rstrip()
    comment = ("#" + comment) if sep else None
    for flag in ("-e", "--editable"):
        target = None
        if body == flag:
            target = None
        elif body.startswith(flag + " "):
            target = body[len(flag) :].strip()
        elif body.startswith(flag + "="):
            target = body[len(flag) + 1 :].strip()
        elif not flag.startswith("--") and body.startswith(flag) and len(body) > len(flag):
            target = body[len(flag) :].strip()  # attached short form, e.g. `-egit+...`
        else:
            continue
        return flag, (target or None), comment
    return None, None, None


def _rewrite_include(line, stripped, src_dir, depth):
    """Rewrite a nested `-r`/`-c` include so pip still resolves it and its
    protected specs are filtered too.

    pip resolves a nested include against the directory of the file it is
    READING; our filtered copy lives under /tmp, so a relative include would
    look in /tmp and fail. Recursively filter the included file (dropping
    protected packages there too, closing the multi-level bypass) and point the
    parent at that filtered copy. URLs and unreadable/absolute-unfiltered files
    fall back to an absolutised path so they still resolve. Returns
    (new_line, changed, recorded, dropped)."""
    flag, target, comment = _parse_include(stripped)
    if not target:
        return line, False, None, []
    newline_char = "\n" if line.endswith("\n") else ""

    def _emit(new_target):
        rebuilt = flag + " " + new_target
        if comment:
            rebuilt += " " + comment
        return rebuilt + newline_char

    # A URL include cannot be filtered locally; leave it verbatim.
    if "://" in target:
        return line, False, None, []
    abs_target = target if os.path.isabs(target) else os.path.join(src_dir, target)
    # Recursively filter the included file. Guard against cyclic / deep includes.
    if depth < 8:
        f_path, f_rec, f_drp = _filter_requirements_file(abs_target, _depth = depth + 1)
        # A nested -c include is a resolver CONSTRAINT, not an install request, so
        # a transformers pin inside it must NOT be recorded as a request (mirrors
        # the top-level -c path in main(), which ignores _c_rec). Only a nested -r
        # requirement include carries real install requests, so keep its pin.
        if flag in _CONSTRAINT_FILE_FLAGS:
            f_rec = None
        if f_path != abs_target:
            # The include was rewritten (protected specs dropped and/or its own
            # nested includes absolutised); point at the filtered copy.
            return _emit(f_path), True, f_rec, f_drp
    # Nothing to filter inside; just make sure the path still resolves from /tmp.
    if not os.path.isabs(target):
        return _emit(abs_target), True, None, []
    return line, False, None, []


def _filter_requirements_file(path, _depth = 0):
    """Strip baked/protected packages out of a `-r` requirements file.

    Returns (path_to_use, recorded_transformers_version, dropped_specs). The same
    _KEEP / transformers rules the inline args get are applied to each requirement
    line, so a notebook `pip install -r reqs.txt` cannot overwrite the cu128 torch
    / vLLM / transformers stack with versions pinned inside the file. When nothing
    is protected, or the file cannot be read/written, the original path is returned
    unchanged. Comments, blank lines and option lines are kept verbatim; a nested
    `-r`/`-c` include is recursively filtered too (protected specs dropped at every
    level).
    """
    try:
        with open(path, encoding = "utf-8") as f:
            lines = f.readlines()
    except OSError:
        return path, None, []  # remote URL / unreadable -> let the real tool handle it
    src_dir = os.path.dirname(os.path.abspath(path))
    out, dropped, recorded, changed = [], [], None, False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)  # comment / blank -> keep
            continue
        if stripped.startswith("-"):
            # An editable requirement (-e/--editable <target>) inside the file is
            # a real install target, so a protected editable such as
            # `-e git+https://.../unsloth.git#egg=unsloth` would reinstall the
            # baked stack. Classify it through _KEEP exactly like the
            # command-line -e case and drop the whole line (flag + target) when
            # the target is protected; a transformers pin is still recorded.
            e_flag, e_target, _e_comment = _parse_editable(stripped)
            if e_target is not None:
                _action, _ver = _classify_flag_target(e_target)
                if _action == "drop":
                    if _ver and not recorded:
                        recorded = _ver
                    dropped.append(e_flag + " " + e_target)
                    changed = True
                    continue
                out.append(line)  # kept editable -> forward the line verbatim
                continue
            # Option or nested include. Recursively filter a nested `-r`/`-c`
            # include (so protected specs deep in the include tree cannot slip
            # past _KEEP) and repoint it so it still resolves from /tmp.
            new_line, rewrote, inc_rec, inc_drp = _rewrite_include(line, stripped, src_dir, _depth)
            out.append(new_line)
            if rewrote:
                changed = True
            if inc_rec and not recorded:
                recorded = inc_rec
            dropped.extend(inc_drp)
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
            elif prev_flag in _CONSTRAINT_FILE_FLAGS:
                # Strip protected pins from the constraint file so it cannot
                # downgrade the baked stack, but a constraint is not an install
                # target and its transformers pin is not an install request, so
                # do not set has_target / recorded here.
                _c_path, _c_rec, _c_drp = _filter_requirements_file(tok)
                keep_args.append(_c_path)
                dropped.extend(_c_drp)
            elif prev_flag in _EDITABLE_FLAGS or prev_flag in _UPGRADE_PKG_FLAGS:
                # The flag was held back (not appended yet): its value is an
                # install target (-e path/url/vcs) or an upgrade selector
                # (-P name), both filtered through _KEEP. Dropping a protected
                # value drops the flag with it, so pip/uv is never left a
                # dangling `-e`/`-P` that fails the cell or refreshes a baked
                # package. A kept editable target sets has_target; -P does not.
                _action, _ver = _classify_flag_target(tok)
                if _action == "drop":
                    if _ver and not recorded:
                        recorded = _ver
                    dropped.append(prev_flag + " " + tok)
                else:
                    keep_args.append(prev_flag)
                    keep_args.append(tok)
                    if prev_flag in _EDITABLE_FLAGS:
                        has_target = True
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
                elif _flag in _CONSTRAINT_FILE_FLAGS:
                    _c_path, _c_rec, _c_drp = _filter_requirements_file(_val)
                    keep_args.append(_flag + "=" + _c_path)
                    dropped.extend(_c_drp)
                elif _flag in _EDITABLE_FLAGS or _flag in _UPGRADE_PKG_FLAGS:
                    # --editable=<target> / --upgrade-package=<name>: filter the
                    # inline value through _KEEP just like the space-separated
                    # form, dropping the whole token for a protected package.
                    _action, _ver = _classify_flag_target(_val)
                    if _action == "drop":
                        if _ver and not recorded:
                            recorded = _ver
                        dropped.append(tok)
                    else:
                        keep_args.append(tok)
                        if _flag in _EDITABLE_FLAGS:
                            has_target = True
                else:
                    keep_args.append(tok)  # option with inline value, not a target
                continue
        # Attached short value-flag form: pip/uv accept `-rreqs.txt`,
        # `-cconstraints.txt`, `-epath` and `-Pname` as ONE token. Without this
        # the token starts with "-" and falls through as an opaque option, so an
        # `-r`-only cell no-ops (has_target stays False) and an attached
        # `-c`/`-e`/`-P` value bypasses _KEEP. Split the 2-char flag from its
        # value and reuse the separated-form handling.
        if len(tok) > 2 and tok[0] == "-" and tok[1] != "-" and tok[:2] in _ATTACHED_SHORT_FLAGS:
            _sflag, _sval = tok[:2], tok[2:]
            if _sflag in _REQ_FILE_FLAGS:
                _req_path, _req_rec, _req_drp = _filter_requirements_file(_sval)
                keep_args.append(_sflag)
                keep_args.append(_req_path)
                has_target = True
                if _req_rec and not recorded:
                    recorded = _req_rec
                dropped.extend(_req_drp)
            elif _sflag in _CONSTRAINT_FILE_FLAGS:
                _c_path, _c_rec, _c_drp = _filter_requirements_file(_sval)
                keep_args.append(_sflag)
                keep_args.append(_c_path)
                dropped.extend(_c_drp)
            else:  # -e / -P: the attached value is an install target / selector
                _action, _ver = _classify_flag_target(_sval)
                if _action == "drop":
                    if _ver and not recorded:
                        recorded = _ver
                    dropped.append(_sflag + " " + _sval)
                else:
                    keep_args.append(_sflag)
                    keep_args.append(_sval)
                    if _sflag in _EDITABLE_FLAGS:
                        has_target = True
            continue
        if tok in _VALUE_FLAGS:
            # -e/--editable and -P/--upgrade-package carry a value that is a
            # potential install target, so hold the flag back and let the
            # skip_next handler emit or drop the flag+value pair together. Every
            # other value-flag keeps its flag verbatim; only its value (an
            # index-url / find-links / target dir / etc.) is an opaque option.
            if tok not in _EDITABLE_FLAGS and tok not in _UPGRADE_PKG_FLAGS:
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
