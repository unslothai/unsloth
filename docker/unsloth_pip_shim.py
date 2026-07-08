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
    "--requirements",
    "-c",
    "--constraint",
    "--constraints",
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
    "--upgrade-strategy",
    "--upgrade-package",
    "-P",
    "--reinstall-package",
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
# uv spells the long forms in the PLURAL (`--requirements`, `--constraints`);
# include both so a `uv pip install --requirements reqs.txt` is filtered too.
_REQ_FILE_FLAGS = {"-r", "--requirement", "--requirements"}
# Constraint files are not install targets, but pip applies their pins during
# resolution, so a `-c constraints.txt` that pins torch/transformers/etc. can
# still downgrade or reinstall a baked package when another target pulls it in.
# Filter protected packages out of them the same way as requirement files.
# (uv's long form is the plural `--constraints`.)
_CONSTRAINT_FILE_FLAGS = {"-c", "--constraint", "--constraints"}
# -e/--editable <path|url|vcs> takes the NEXT token as its target (pip:
# `-e, --editable <path/url>`), and that target is a real install target. A
# protected editable (e.g. `-e git+https://.../unsloth.git#egg=unsloth`) must
# drop BOTH the flag and its value; dropping the value alone leaves pip a
# dangling `-e` that swallows the next kept package and fails the whole cell.
_EDITABLE_FLAGS = {"-e", "--editable"}
# -P/--upgrade-package <name> is uv's selective-upgrade flag and
# --reinstall-package <name> is uv's selective-reinstall flag: naming a baked
# package (e.g. `uv pip install -P torch peft` or
# `uv pip install --reinstall-package torch peft`) lets an ordinary install
# target refresh/reinstall that package and clobber the pinned stack. Filter the
# value through _KEEP too, dropping the flag+value pair for a protected name so
# no dangling selector is left to swallow the next kept target. Unlike -e none of
# these is itself an install target (no has_target).
_UPGRADE_PKG_FLAGS = {"-P", "--upgrade-package", "--reinstall-package"}
# Short value-flags pip/uv accept in the ATTACHED form, i.e. the 2-char flag
# glued to its value in one token: `-rreqs.txt`, `-cconstraints.txt`, `-epath`,
# `-Pname`. The scanner splits the flag from the value so the value is filtered
# (requirement/constraint file) or classified (-e/-P) instead of falling through
# as an opaque option -- otherwise an attached `-r`-only cell no-ops and an
# attached `-c`/`-e`/`-P` value bypasses _KEEP.
_ATTACHED_SHORT_FLAGS = {"-r", "-c", "-e", "-P"}
# Resolver-wide reinstall / ignore-installed switches (pip --force-reinstall,
# --ignore-installed, -I; uv --reinstall) force the tool to REINSTALL packages
# that are already satisfied -- including the baked torch/transformers pulled in
# as dependencies of a kept target. Drop them in shim mode so a
# `pip install --force-reinstall peft` cannot rebuild the pinned stack under the
# guise of installing an unprotected package. The kept target still installs; its
# already-satisfied protected deps are left untouched. Per-package selectors
# (--reinstall-package / -P) are handled through _UPGRADE_PKG_FLAGS instead.
_REINSTALL_FLAGS = {"--force-reinstall", "--ignore-installed", "-I", "--reinstall"}
# Value-flags whose flag+value pair is dropped outright in shim mode.
# `--upgrade-strategy eager` makes pip upgrade EVERY dependency of a kept target
# regardless of whether the installed version already satisfies it, which would
# refresh the baked torch/transformers under the pinned CUDA stack. Dropping the
# flag falls back to pip's default `only-if-needed`, so a kept target still
# installs but already-satisfied protected deps stay put. (`only-if-needed` is
# the default, so dropping a `--upgrade-strategy only-if-needed` is a no-op.)
_DROP_VALUE_FLAGS = {"--upgrade-strategy"}


# Source-distribution / archive suffixes pip accepts as an install target.
_ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip")


def _sdist_name(basename):
    """Distribution name from a source-archive basename ({name}-{version}.ext),
    or None if it is not a recognised archive. Splits at the first hyphen that
    precedes a digit so legacy hyphenated names (flashinfer-python-1.0,
    pytorch-triton-2.0) resolve correctly, not just PEP 625-normalised ones."""
    low = basename.lower()
    stem = None
    for ext in _ARCHIVE_EXTS:
        if low.endswith(ext):
            stem = basename[: -len(ext)]
            break
    if stem is None:
        return None
    m = re.match(r"^(.+?)-\d", stem)
    name = (m.group(1) if m else stem).strip().lower().replace("_", "-")
    return name or None


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
        # A source archive (sdist / zip) URL or path names its distribution the
        # same way ({name}-{version}.tar.gz etc.), so `pip install
        # https://files.pythonhosted.org/.../unsloth-2026.7.1.tar.gz` or
        # `./torch-2.11.0.tar.gz` must be matched against _KEEP too, not passed
        # through as an opaque positional that reinstalls the baked package.
        _arch = _sdist_name(token.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        if _arch:
            return _arch
        # A VCS URL without an #egg= fragment still installs a named project:
        # pip/uv derive the distribution from the repo, and for the packages we
        # protect the repo basename equals the distribution
        # (huggingface/transformers.git -> transformers,
        # unslothai/unsloth-zoo.git -> unsloth-zoo). Infer it from the last path
        # segment so a bare `pip install git+https://github.com/huggingface/
        # transformers.git` -- an egg-less form this repo itself recommends in
        # unsloth/models/loader.py -- cannot reinstall the baked package past
        # _KEEP. A non-protected repo returns its basename and the caller keeps
        # the token as a normal target either way.
        if re.match(r"^[a-z]+\+", token):
            _seg = token.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
            _seg = _seg.split("@", 1)[0]  # drop a @branch / @tag / @commit ref
            if _seg.endswith(".git"):
                _seg = _seg[:-4]
            _seg = _seg.strip().lower().replace("_", "-")
            if _seg:
                return _seg
        return None  # plain url / local path -> let it pass through
    # A bare wheel filename (no ./ or / prefix and no scheme) is still a valid
    # pip target from the CWD: `pip install torch-2.11.0-cp312-...-linux.whl`.
    # It reaches here because it starts with neither `.`/`/` nor a scheme, so
    # without this it would fall through as the whole filename and miss _KEEP,
    # reinstalling the baked torch. Parse its PEP 427 distribution the same way
    # as the URL/path wheel case above.
    if token.lower().endswith(".whl"):
        dist = token.rsplit("/", 1)[-1][:-4].split("-", 1)[0].strip().lower().replace("_", "-")
        if dist:
            return dist
    # A bare source-archive filename from the CWD (`pip install torch-2.11.0.tar.gz`)
    # is a valid pip target too; parse its distribution the same way.
    _barch = _sdist_name(token.rsplit("/", 1)[-1])
    if _barch:
        return _barch
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

    # A remote (URL) nested include cannot be fetched/filtered here, so its
    # protected pins would reach the real tool untouched. Drop the include line
    # instead of letting pip pull an unfiltered requirements file off the network
    # (mirrors the top-level remote `-r`/`-c` refusal in main). new_line=None
    # tells the caller to remove the line entirely.
    if "://" in target:
        return None, True, None, [flag + " " + target]
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
            if new_line is not None:
                out.append(new_line)  # None -> a remote include was dropped
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
            if prev_flag in _REQ_FILE_FLAGS or prev_flag in _CONSTRAINT_FILE_FLAGS:
                if "://" in tok:
                    # Remote requirement/constraint file: it cannot be inspected
                    # or filtered, so refuse it in shim mode rather than let the
                    # real tool fetch and install protected pins off the network.
                    # The flag was appended when we first saw it; pop it so pip/uv
                    # is not left a dangling -r/-c.
                    if keep_args and keep_args[-1] == prev_flag:
                        keep_args.pop()
                    dropped.append(prev_flag + " " + tok)
                elif prev_flag in _REQ_FILE_FLAGS:
                    # Filter baked/protected packages out of the requirements file
                    # so a notebook `pip install -r reqs.txt` cannot clobber the
                    # cu128 stack or push transformers into the base venv.
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(tok)
                    keep_args.append(_req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                else:
                    # Strip protected pins from the constraint file so it cannot
                    # downgrade the baked stack, but a constraint is not an install
                    # target and its transformers pin is not an install request, so
                    # do not set has_target / recorded here.
                    _c_path, _c_rec, _c_drp = _filter_requirements_file(tok)
                    keep_args.append(_c_path)
                    dropped.extend(_c_drp)
            elif prev_flag in _DROP_VALUE_FLAGS:
                # --upgrade-strategy (eager): the flag was appended when we saw
                # it; pop it and drop the flag+value pair so pip falls back to
                # its safe only-if-needed default.
                if keep_args and keep_args[-1] == prev_flag:
                    keep_args.pop()
                dropped.append(prev_flag + " " + tok)
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
                if (_flag in _REQ_FILE_FLAGS or _flag in _CONSTRAINT_FILE_FLAGS) and "://" in _val:
                    # Remote requirement/constraint file in `--flag=URL` form:
                    # refuse it in shim mode (the flag rides in the same token, so
                    # dropping the token leaves nothing dangling).
                    dropped.append(tok)
                elif _flag in _REQ_FILE_FLAGS:
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(_val)
                    keep_args.append(_flag + "=" + _req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                elif _flag in _DROP_VALUE_FLAGS:
                    dropped.append(tok)  # --upgrade-strategy=eager -> drop the pair
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
            if (_sflag in _REQ_FILE_FLAGS or _sflag in _CONSTRAINT_FILE_FLAGS) and "://" in _sval:
                # Remote requirement/constraint file in attached `-rURL`/`-cURL`
                # form: refuse it in shim mode (nothing was appended yet, so just
                # drop the whole token).
                dropped.append(_sflag + " " + _sval)
            elif _sflag in _REQ_FILE_FLAGS:
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
        if tok in _REINSTALL_FLAGS:
            # Resolver-wide reinstall / ignore-installed switch: drop it so pip/uv
            # cannot rebuild already-satisfied baked deps (torch/transformers
            # pulled in by a kept target). The kept target still installs.
            dropped.append(tok)
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
