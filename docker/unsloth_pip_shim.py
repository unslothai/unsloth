#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

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
# pip/uv flags that consume the next token as a value (not a requirement).
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
    # Every remaining value-taking flag of pip/uv install (from both --help). A
    # missing one makes the scanner misread its VALUE. uv:
    "--allow-insecure-host",
    "--build-constraints",
    "-b",
    "--cache-dir",
    "--color",
    "--config-file",
    "--config-setting",
    "-C",
    "--config-settings-package",
    "--default-index",
    "--directory",
    "--exclude-newer",
    "--exclude-newer-package",
    "--excludes",
    "--extra",
    "--fork-strategy",
    "--group",
    "--index",
    "--keyring-provider",
    "--link-mode",
    "--no-build-isolation-package",
    "--no-sources-package",
    "--overrides",
    "--prerelease",
    "--project",
    "--python-platform",
    "--refresh-package",
    "--resolution",
    "--torch-backend",
    # newer uv (0.10+):
    "--no-editable-package",
    "--upgrade-group",
    # pip:
    "--build-constraint",
    "--cert",
    "--client-cert",
    "--config-settings",
    "--exists-action",
    "--log",
    "--progress-bar",
    "--proxy",
    "--report",
    "--resume-retries",
    "--retries",
    "--root",
    "--root-user-action",
    "--src",
    "--timeout",
    "--trusted-host",
    "--use-deprecated",
    "--use-feature",
    # newer pip (26+):
    "--all-releases",
    "--only-final",
    "--requirements-from-script",
    "--uploaded-prior-to",
}
# Value-flags whose VALUE is itself an install target (a requirements file pulls
# real requirements). uv spells the long forms plural; include both.
_REQ_FILE_FLAGS = {"-r", "--requirement", "--requirements"}
# Constraint files aren't install targets, but pip applies their pins, so a -c
# pinning torch/transformers can downgrade a baked package. Filter like -r files.
_CONSTRAINT_FILE_FLAGS = {"-c", "--constraint", "--constraints"}
# -e/--editable takes the next token as a real install target. A protected
# editable must drop BOTH flag and value, else a dangling -e swallows the next
# kept package and fails the cell.
_EDITABLE_FLAGS = {"-e", "--editable"}
# -P/--upgrade-package/--reinstall-package are uv's selective upgrade flags:
# filter the value through _KEEP, dropping the flag+value pair for a protected
# name. Unlike -e, none is itself an install target.
_UPGRADE_PKG_FLAGS = {"-P", "--upgrade-package", "--reinstall-package"}
# Short value-flags accepted ATTACHED (-rreqs.txt, -cX, -epath, -Pname). Split
# flag from value so it's filtered, else -r no-ops and -c/-e/-P bypass _KEEP.
_ATTACHED_SHORT_FLAGS = {"-r", "-c", "-e", "-P"}
# Resolver-wide reinstall/ignore-installed switches (pip --force-reinstall,
# --ignore-installed, -I; uv --reinstall) rebuild baked deps; drop them (the kept
# target still installs). uv's --exact removes everything outside the closure, so
# drop it too.
_REINSTALL_FLAGS = {"--force-reinstall", "--ignore-installed", "-I", "--reinstall", "--exact"}
# Value-flags dropped outright with their value. --upgrade-strategy eager would
# upgrade every dep of a kept target; dropping it falls back to only-if-needed.
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
    # PEP 508 direct reference: "name [extras] @ <url>". Pull the name out BEFORE
    # the url/vcs guard below, else a protected package pinned via URL slips _KEEP.
    _dref = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*@(?:\s|git\+|hg\+|bzr\+|svn\+|[a-z]+://)",
        token,
    )
    if _dref:
        return _dref.group(1).lower().replace("_", "-") or None
    if re.match(r"^[a-z]+\+", token) or "://" in token or token.startswith((".", "/")):
        # A VCS/URL install can name a protected package via the #egg=NAME
        # fragment; pull it out so _KEEP can drop it.
        _egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", token)
        if _egg:
            return _egg.group(1).lower().replace("_", "-") or None
        # A wheel URL/path names its distribution in the PEP 427 filename (leading
        # dash-split of the basename), so a bare torch-*.whl would slip _KEEP.
        _whl = re.search(r"([^/\\#?]+)\.whl(?:[#?]|$)", token)
        if _whl:
            dist = _whl.group(1).split("-", 1)[0].strip().lower().replace("_", "-")
            if dist:
                return dist
        # A source archive ({name}-{version}.tar.gz) names its distribution too;
        # match it against _KEEP instead of passing it through as opaque.
        _arch = _sdist_name(token.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        if _arch:
            return _arch
        # A VCS URL without #egg= still installs a named project; the repo basename
        # equals the distribution for our protected packages. Infer from the last
        # path segment so an egg-less git+ URL can't reinstall past _KEEP.
        if re.match(r"^[a-z]+\+", token):
            _rest = token.split("#", 1)[0].split("?", 1)[0]
            # Drop the @ref before the basename (a ref may contain a slash). Split
            # path from authority first so an SSH userinfo @ isn't the ref; like
            # pip, the ref is everything after the LAST @.
            if "://" in _rest:
                _authority, _slash, _path = _rest.partition("://")[2].partition("/")
                if "@" in _path:
                    _path = _path.rsplit("@", 1)[0]
                _rest = _path if _slash else _authority
            _seg = _rest.rstrip("/").rsplit("/", 1)[-1]
            _seg = _seg.split("@", 1)[0]  # schemeless fallback: drop a plain @ref
            if _seg.endswith(".git"):
                _seg = _seg[:-4]
            _seg = _seg.strip().lower().replace("_", "-")
            if _seg:
                return _seg
        # A local project DIRECTORY installs the project it contains; resolve its
        # name from metadata so _KEEP applies. Metadata-less dirs pass through.
        _local = _local_project_name(token)
        if _local:
            return _local
        return None  # plain url / metadata-less local path -> let it pass through
    # A local project dir referenced without ./ or / is still a path target when
    # it exists on disk; classify it before the spec parse mangles the separator.
    if "/" in token or os.sep in token:
        _local = _local_project_name(token)
        if _local:
            return _local
    # A bare wheel filename from the CWD is a valid pip target; parse its PEP 427
    # distribution like the URL/path wheel case above, else it misses _KEEP.
    if token.lower().endswith(".whl"):
        dist = token.rsplit("/", 1)[-1][:-4].split("-", 1)[0].strip().lower().replace("_", "-")
        if dist:
            return dist
    # A bare source-archive filename from the CWD is a valid target too; parse it.
    _barch = _sdist_name(token.rsplit("/", 1)[-1])
    if _barch:
        return _barch
    # strip extras and any version/marker tail
    name = re.split(r"[<>=!~\[\s;@]", token, 1)[0].strip()
    return name.lower().replace("_", "-") or None


def _local_project_name(token):
    """Distribution name of a local project directory install target, else None.

    Reads the name pip/uv would build: pyproject.toml [project].name, falling
    back to setup.cfg [metadata] name, falling back to the directory basename
    when a setup.py exists (a bare basename guess is used ONLY when the dir is
    an installable project at all). A directory without any project metadata is
    not a pip target and returns None so ordinary paths pass through untouched.
    Names are exact after normalization: a user's own `my-torch-utils` dir never
    matches the protected `torch`.
    """
    path = token.split("#", 1)[0]
    if not os.path.isdir(path):
        return None
    _pyproject = os.path.join(path, "pyproject.toml")
    if os.path.isfile(_pyproject):
        try:
            import tomllib
            with open(_pyproject, "rb") as f:
                _name = (tomllib.load(f).get("project") or {}).get("name")
            if _name:
                return _name.strip().lower().replace("_", "-") or None
        except Exception:
            pass  # unparseable metadata -> fall through to the other signals
    _setup_cfg = os.path.join(path, "setup.cfg")
    if os.path.isfile(_setup_cfg):
        try:
            import configparser

            _cp = configparser.ConfigParser()
            _cp.read(_setup_cfg)
            _name = _cp.get("metadata", "name", fallback = None)
            if _name:
                return _name.strip().lower().replace("_", "-") or None
        except Exception:
            pass
    if os.path.isfile(os.path.join(path, "setup.py")) or os.path.isfile(_pyproject):
        _base = os.path.basename(os.path.normpath(path))
        return _base.strip().lower().replace("_", "-") or None
    return None


def _version_pin(token):
    """Return the pinned version for a `pkg==X` token, else None."""
    m = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", token)
    return m.group(1) if m else None


# pip expands ${UPPERCASE_NAME} in requirements files, so `${PKG}==...` with
# PKG=torch would slip _KEEP. Expand for CLASSIFICATION only; kept lines verbatim.
_ENV_REF_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_refs(text):
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)


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


def _parse_flag_line(stripped, flags):
    """If `stripped` is a `<flag> <target>` requirements-file line for one of
    `flags`, return (flag, target_or_None, inline_comment_or_None); else
    (None, None, None).

    Shared by the `-r`/`--requirement`/`-c`/`--constraint` include parse and
    the `-e`/`--editable` install-line parse. Handles the separated
    (`-r <t>` / `--editable <t>`), inline (`--editable=<t>` / `-e=<t>`) and
    attached short (`-rextras.txt`, `-egit+...`) forms pip accepts from a
    requirement file, so a protected include or editable there is handled
    exactly like the command-line case."""
    body, sep, comment = stripped.partition(" #")
    body = body.rstrip()
    comment = ("#" + comment) if sep else None
    for flag in flags:
        if body == flag or body.startswith(flag + " "):
            target = body[len(flag) :].strip()
        elif body.startswith(flag + "="):
            target = body[len(flag) + 1 :].strip()
        elif not flag.startswith("--") and body.startswith(flag) and len(body) > len(flag):
            target = body[len(flag) :].strip()  # attached short form
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
    flag, raw_target, comment = _parse_flag_line(
        stripped, ("-r", "--requirement", "-c", "--constraint")
    )
    if not raw_target:
        return line, False, None, []
    # Resolve pip's ${VAR} references so the include we read/filter is the file
    # pip would actually read (a literal `${DIR}/reqs.txt` never resolves here).
    target = _expand_env_refs(raw_target)
    newline_char = "\n" if line.endswith("\n") else ""

    def _emit(new_target):
        rebuilt = flag + " " + new_target
        if comment:
            rebuilt += " " + comment
        return rebuilt + newline_char

    # A remote (URL) nested include can't be filtered here, so drop it rather than
    # let pip pull unfiltered pins off the network (mirrors main's top-level
    # refusal). new_line=None tells the caller to remove the line.
    if "://" in target:
        return None, True, None, [flag + " " + raw_target]
    abs_target = target if os.path.isabs(target) else os.path.join(src_dir, target)
    # Recursively filter the included file. Guard against cyclic / deep includes.
    if depth < 8:
        f_path, f_rec, f_drp = _filter_requirements_file(abs_target, _depth = depth + 1)
        # A nested -c include is a resolver CONSTRAINT, not an install request, so
        # don't record its transformers pin (mirrors main's -c path). Only -r
        # includes carry real requests, so keep their pin.
        if flag in _CONSTRAINT_FILE_FLAGS:
            f_rec = None
        if f_path != abs_target:
            # The include was rewritten; point at the filtered copy.
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
            # An -e/--editable <target> in the file is a real install target, so a
            # protected editable would reinstall the baked stack. Classify through
            # _KEEP like the command-line -e case; drop the whole line when
            # protected (a transformers pin is still recorded).
            e_flag, e_target, _e_comment = _parse_flag_line(stripped, ("-e", "--editable"))
            if e_target is not None:
                _action, _ver = _classify_flag_target(_expand_env_refs(e_target))
                if _action == "drop":
                    if _ver and not recorded:
                        recorded = _ver
                    dropped.append(e_flag + " " + e_target)
                    changed = True
                    continue
                out.append(line)  # kept editable -> forward the line verbatim
                continue
            # Option or nested include. Recursively filter a nested `-r`/`-c`
            # include (protected specs deep in the tree) and repoint it for /tmp.
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
        classified = _expand_env_refs(spec)  # classify what pip will SEE
        name = _canon(classified)
        if name is None:
            out.append(line)  # url / path / vcs / unparseable -> keep
            continue
        if name == "transformers":
            v = _version_pin(classified)
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
    except OSError as exc:
        # Fail CLOSED: protected requirements were detected, so forwarding the
        # original would hand pip the specs we must filter. Abort instead.
        raise SystemExit(
            f"[unsloth-nb] could not write a filtered copy of {path} ({exc}); "
            "refusing to forward a requirements file that pins protected packages."
        )
    return tmp, recorded, dropped


def _protected_constraints_file():
    """Write `name==version` pins for every INSTALLED protected package to a
    temp constraints file and return its path (None when nothing is pinned or
    the file cannot be written).

    Argument filtering alone does not constrain pip/uv's RESOLVER: a kept
    package may declare e.g. `torch==99.0` as a dependency and the tool would
    replace the baked torch to satisfy it. Pinning the protected set on every
    forwarded install makes such an install fail loudly instead. This is
    belt-and-braces on top of the argument filtering, so a failure here keeps
    the install usable rather than aborting it.
    """
    try:
        from importlib.metadata import distributions

        pins = {}
        for dist in distributions():
            raw = (dist.metadata["Name"] or "").strip()
            name = raw.lower().replace("_", "-")
            if not name or name in pins:
                continue
            if name == "transformers" or name in _KEEP or name.startswith(_KEEP_PREFIX):
                pins[name] = f"{raw}=={dist.version}"
        if not pins:
            return None
        fd, tmp = tempfile.mkstemp(prefix = "unsloth-nb-protected-", suffix = ".txt")
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write("\n".join(pins[name] for name in sorted(pins)) + "\n")
        return tmp
    except Exception:
        return None


def _selfcheck_value_flags():
    """Assert every value-taking flag the REAL pip/uv document is classified.

    A value flag missing from _VALUE_FLAGS makes the scanner misread its VALUE
    (see --torch-backend in the header of the added block above). Run at image
    build time against the BAKED tools -- the exact versions the shim fronts --
    so a pip/uv bump that adds a value flag fails the build, not a user's cell.
    Exits 0 when clean, 1 with the missing flags listed.
    """
    import subprocess

    known = _VALUE_FLAGS | _DROP_VALUE_FLAGS
    missing = {}
    for label, cmd in (
        ("pip", [REAL["pip"], "install", "--help"]),
        ("uv", [REAL["uv"], "pip", "install", "--help"]),
    ):
        try:
            out = subprocess.run(cmd, capture_output = True, text = True).stdout
        except OSError:
            continue  # tool absent (e.g. a pip-only environment)
        flags = set()
        for m in re.finditer(r"^\s+(-\w)?,?\s*(--[\w-]+)[= ]<", out, re.M):
            if m.group(1):
                flags.add(m.group(1))
            flags.add(m.group(2))
        for m in re.finditer(r"^\s+(-\w) <", out, re.M):
            flags.add(m.group(1))
        gap = flags - known
        if gap:
            missing[label] = sorted(gap)
    if missing:
        print(f"[unsloth-nb] value flags missing from _VALUE_FLAGS: {missing}", file = sys.stderr)
        sys.exit(1)
    print("[unsloth-nb] value-flag selfcheck OK")
    sys.exit(0)


def main():
    tool = "uv" if os.path.basename(sys.argv[0]).startswith("uv") else "pip"
    argv = sys.argv[1:]

    if argv[:1] == ["--unsloth-selfcheck-value-flags"]:
        _selfcheck_value_flags()

    # Only intercept inside a notebook kernel (UNSLOTH_NB_SHIM); everywhere else
    # behave exactly like the real tool.
    if os.environ.get("UNSLOTH_NB_SHIM") != "1":
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    # Locate the `install` verb (pip: `pip install ...`; uv: `uv pip install ...`
    # -- index() already skips uv's leading `pip` subcommand).
    try:
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
            # -r/--requirement's value pulls real requirements (a target); an
            # index-url / find-links / constraint value is an option, not a target.
            if prev_flag in _REQ_FILE_FLAGS or prev_flag in _CONSTRAINT_FILE_FLAGS:
                if "://" in tok:
                    # Remote requirement/constraint file: can't be filtered, so
                    # refuse it rather than fetch protected pins off the network.
                    # Pop the flag we appended so pip/uv has no dangling -r/-c.
                    if keep_args and keep_args[-1] == prev_flag:
                        keep_args.pop()
                    dropped.append(prev_flag + " " + tok)
                elif prev_flag in _REQ_FILE_FLAGS:
                    # Filter protected packages out of the requirements file so
                    # `pip install -r reqs.txt` can't clobber the cu128 stack.
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(tok)
                    keep_args.append(_req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                else:
                    # Strip protected pins from the constraint file so it can't
                    # downgrade the baked stack; a constraint isn't an install
                    # target, so don't set has_target / recorded here.
                    _c_path, _c_rec, _c_drp = _filter_requirements_file(tok)
                    keep_args.append(_c_path)
                    dropped.extend(_c_drp)
            elif prev_flag in _DROP_VALUE_FLAGS:
                # --upgrade-strategy (eager): drop the pair so pip falls back to
                # only-if-needed.
                if keep_args and keep_args[-1] == prev_flag:
                    keep_args.pop()
                dropped.append(prev_flag + " " + tok)
            elif prev_flag in _EDITABLE_FLAGS or prev_flag in _UPGRADE_PKG_FLAGS:
                # Flag held back: its value is an install target (-e) or upgrade
                # selector (-P), filtered through _KEEP. A protected value drops
                # the flag too. A kept editable sets has_target; -P does not.
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
        # --flag=value form (--requirement=reqs.txt / --index-url=URL as one token).
        # Without this the -r file is never filtered and a file-only cell no-ops.
        if tok.startswith("--") and "=" in tok:
            _flag, _, _val = tok.partition("=")
            if _flag in _VALUE_FLAGS:
                if (_flag in _REQ_FILE_FLAGS or _flag in _CONSTRAINT_FILE_FLAGS) and "://" in _val:
                    # Remote requirement/constraint file in `--flag=URL` form:
                    # refuse it (dropping the token leaves nothing dangling).
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
                    # inline value through _KEEP, dropping the token if protected.
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
        # Attached short value-flag form (-rreqs.txt, -cX, -epath, -Pname as ONE
        # token). Split flag from value and reuse the separated-form handling,
        # else -r no-ops and -c/-e/-P bypass _KEEP.
        if len(tok) > 2 and tok[0] == "-" and tok[1] != "-" and tok[:2] in _ATTACHED_SHORT_FLAGS:
            _sflag, _sval = tok[:2], tok[2:]
            if (_sflag in _REQ_FILE_FLAGS or _sflag in _CONSTRAINT_FILE_FLAGS) and "://" in _sval:
                # Remote requirement/constraint file in attached `-rURL`/`-cURL`
                # form: refuse it (nothing appended yet, drop the whole token).
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
            # can't rebuild satisfied baked deps. The kept target still installs.
            dropped.append(tok)
            continue
        if tok in _VALUE_FLAGS:
            # -e/--editable and -P/--upgrade-package carry a potential install
            # target, so hold the flag back and let skip_next emit or drop the
            # pair together. Every other value-flag keeps its flag verbatim; only
            # its value is an opaque option.
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

    # Anything left to install? A line with only baked packages + option flags
    # leaves no target, so no-op instead of exec'ing a bare install that fails.
    if not has_target:
        print("[unsloth-nb] nothing to install after keeping the baked stack; ok.")
        return
    cmd = [REAL[tool]] + head + keep_args
    # Constrain the resolver too: an allowed target could pull an incompatible
    # torch/transformers in as a dependency and replace the baked wheel.
    constraints = _protected_constraints_file()
    if constraints:
        cmd += ["--constraint", constraints]
    sys.stdout.flush()
    os.execv(REAL[tool], cmd)


if __name__ == "__main__":
    main()
