#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""pip / uv shim for the Unsloth Docker notebook environment.

Sits ahead of the real tools on PATH so a notebook `!pip install` cell cannot clobber
the baked cu128 stack: `transformers==X` is recorded for the unsloth_nb_compat sidecar
instead of installed, other _KEEP packages are skipped WHEN THEY ARE ACTUALLY BAKED
IN, everything else passes through. The real tools are invoked by absolute path, so
there is no recursion.
"""

import os, re, sys, tempfile

REAL = {"pip": "/opt/unsloth-venv/bin/pip", "uv": "/opt/unsloth-venv/bin/uv"}
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")

# Membership criterion: replacing the package invalidates the stack the image was
# built and tested against, either an ABI/CUDA-matched wheel the Dockerfile resolved
# deliberately, or a library unsloth/unsloth_zoo patches by version at import time.
_KEEP = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchao",
    "torchcodec",
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
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "huggingface-hub",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
}
_KEEP_PREFIX = ("nvidia-", "nvidia_")
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
    "--no-editable-package",
    "--upgrade-group",
    "--prerelease-package",
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
    "--all-releases",
    "--only-final",
    "--requirements-from-script",
    "--uploaded-prior-to",
}
_REQ_FILE_FLAGS = {"-r", "--requirement", "--requirements"}
_CONSTRAINT_FILE_FLAGS = {"-c", "--constraint", "--constraints"}
# a protected editable must drop BOTH flag and value, or a dangling -e eats the next
_EDITABLE_FLAGS = {"-e", "--editable"}
_UPGRADE_PKG_FLAGS = {"-P", "--upgrade-package", "--reinstall-package"}
_ATTACHED_SHORT_FLAGS = {"-r", "-c", "-e", "-P"}
# these rebuild baked deps; the kept target still installs once they are dropped
_REINSTALL_FLAGS = {"--force-reinstall", "--ignore-installed", "-I", "--reinstall", "--exact"}
# dropped with their value; eager would upgrade every dep of a kept target
# Hidden ALIASES of flags already listed above. pip's --help prints only the
# canonical spelling, so no amount of help-scraping can find these; the selfcheck now
# introspects pip's own option table instead, which is where they came from.
# --python-preference is a uv GLOBAL that appears in neither `uv --help` nor
# `uv help`, and uv still accepts it.
_ALIAS_VALUE_FLAGS = {
    "--default-timeout",
    "--local-log",
    "--log-file",
    "--pypi-url",
    "--source",
    "--source-dir",
    "--source-directory",
    "--python-preference",
}
_VALUE_FLAGS |= _ALIAS_VALUE_FLAGS

_DROP_VALUE_FLAGS = {"--upgrade-strategy"}

# these ARE the install target, with no package on the command line: without them the
# shim scans to no target and no-ops while printing "ok"
_TARGET_VALUE_FLAGS = {"--group", "--upgrade-group", "--requirements-from-script"}


_ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip")


def _norm_name(name):
    """PEP 503 normalised name; every _KEEP / _KEEP_PREFIX comparison goes through here."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower() or None


def _base_site_packages():
    """site-packages of the venv that owns the REAL pip/uv, or [] if it cannot be
    located.

    Every scan below MUST be scoped to this rather than to sys.path. When a notebook
    activates a transformers sidecar, unsloth_nb_compat and unsloth_run PREPEND that
    sidecar to PYTHONPATH, and the sidecars are built with `uv pip install --target`,
    so each one ships a real transformers-X.dist-info. A bare distributions() walks
    sys.path in order and therefore reports the SIDECAR's transformers first, which
    would pin the sidecar version into the protected constraints and let the resolver
    move the shared base install to it.

    The venv that owns the REAL pip is the authoritative answer. purelib is the
    fallback and is safe for the same reason: it derives from sys.prefix, which
    PYTHONPATH cannot influence, so it can never name a sidecar."""
    import glob

    venv = os.path.dirname(os.path.dirname(REAL["pip"]))
    found = [
        p
        for p in sorted(glob.glob(os.path.join(venv, "lib", "python*", "site-packages")))
        if os.path.isdir(p)
    ]
    if found:
        return found
    import sysconfig

    purelib = sysconfig.get_paths().get("purelib") or ""
    return [purelib] if purelib and os.path.isdir(purelib) else []


_INSTALLED_NAMES = []


def _installed_names():
    """Every distribution name in this venv, PEP 503 normalised, or None when the scan
    fails. Cached because importlib.metadata rescans sys.path on every call."""
    if not _INSTALLED_NAMES:
        try:
            from importlib.metadata import distributions

            scope = _base_site_packages()
            if not scope:
                # unknown base: None makes _is_installed answer "assume baked", which
                # is the stricter of the two answers
                raise RuntimeError("base site-packages not found")
            found = set()
            for dist in distributions(path = scope):
                try:
                    n = _norm_name(dist.metadata["Name"] or "")
                except Exception:
                    n = None
                if n:
                    found.add(n)
            _INSTALLED_NAMES.append(found)
        except Exception:
            _INSTALLED_NAMES.append(None)
    return _INSTALLED_NAMES[0]


def _is_protected(name):
    """True when `name` is a baked package that is REALLY installed. Every drop
    decision goes through here so the three call sites cannot drift apart.

    An absent package is nothing to protect, and dropping it turned a recovery
    install into a silent success: the Dockerfile lets the torchcodec bake and the
    non-amd64 vLLM bake fail on purpose, so a notebook `pip install vllm` printed
    "kept baked versions, skipped: vllm" over an image that had no vLLM at all, and
    the GRPO fast_inference path stayed broken. Forwarding instead either installs it
    under the protected constraints or fails loudly, and both beat a false success.
    An unreadable metadata scan keeps the old, stricter answer."""
    if name is None:
        return False
    if not (name in _KEEP or name.startswith(_KEEP_PREFIX)):
        return False
    return _is_installed(name)


def _is_installed(name):
    """True when `name` is really in this venv, or when the scan failed and the
    stricter "assume baked" answer is the safe one."""
    present = _installed_names()
    return present is None or name in present


def _sdist_name(basename):
    """Name from a `{name}-{version}.ext` basename, split at the first hyphen before a
    digit so legacy hyphenated names resolve too."""
    low = basename.lower()
    stem = None
    for ext in _ARCHIVE_EXTS:
        if low.endswith(ext):
            stem = basename[: -len(ext)]
            break
    if stem is None:
        return None
    m = re.match(r"^(.+?)-\d", stem)
    return _norm_name(m.group(1) if m else stem)


def _canon(token):
    if token.startswith("-"):
        return None
    _dref = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*@(?:\s|git\+|hg\+|bzr\+|svn\+|[a-z]+://)",
        token,
    )
    if _dref:
        return _norm_name(_dref.group(1))
    if re.match(r"^[a-z]+\+", token) or "://" in token or token.startswith((".", "/")):
        _egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", token)
        if _egg:
            return _norm_name(_egg.group(1))
        _whl = re.search(r"([^/\\#?]+)\.whl(?:[#?]|$)", token)
        if _whl:
            dist = _norm_name(_whl.group(1).split("-", 1)[0])
            if dist:
                return dist
        _arch = _sdist_name(token.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        if _arch:
            return _arch
        if re.match(r"^[a-z]+\+", token):
            _rest = token.split("#", 1)[0].split("?", 1)[0]
            # split path from authority first, so an SSH userinfo @ is not the ref
            if "://" in _rest:
                _authority, _slash, _path = _rest.partition("://")[2].partition("/")
                if "@" in _path:
                    _path = _path.rsplit("@", 1)[0]
                _rest = _path if _slash else _authority
            _seg = _rest.rstrip("/").rsplit("/", 1)[-1]
            _seg = _seg.split("@", 1)[0]
            if _seg.endswith(".git"):
                _seg = _seg[:-4]
            _seg = _norm_name(_seg)
            if _seg:
                return _seg
        _local = _local_project_name(token)
        if _local:
            return _local
        return None
    if "/" in token or os.sep in token:
        _local = _local_project_name(token)
        if _local:
            return _local
    if token.lower().endswith(".whl"):
        dist = _norm_name(token.rsplit("/", 1)[-1][:-4].split("-", 1)[0])
        if dist:
            return dist
    _barch = _sdist_name(token.rsplit("/", 1)[-1])
    if _barch:
        return _barch
    name = re.split(r"[<>=!~\[\s;@]", token, maxsplit = 1)[0].strip()
    return _norm_name(name)


def _local_project_name(token):
    """Name pip/uv would build for a local project dir, falling back to the basename
    only when the dir is an installable project at all. None for a metadata-less dir,
    so ordinary paths pass through."""
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
                return _norm_name(_name)
        except Exception:
            pass
    _setup_cfg = os.path.join(path, "setup.cfg")
    if os.path.isfile(_setup_cfg):
        try:
            import configparser

            _cp = configparser.ConfigParser()
            _cp.read(_setup_cfg)
            _name = _cp.get("metadata", "name", fallback = None)
            if _name:
                return _norm_name(_name)
        except Exception:
            pass
    if os.path.isfile(os.path.join(path, "setup.py")) or os.path.isfile(_pyproject):
        _base = os.path.basename(os.path.normpath(path))
        return _norm_name(_base)
    return None


def _version_pin(token):
    m = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", token)
    return m.group(1) if m else None


# pip expands ${UPPERCASE} in requirements files, so `${PKG}==` would slip _KEEP.
# Expanded for CLASSIFICATION only; kept lines stay verbatim.
_ENV_REF_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_refs(text):
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), text)


_EXTRAS_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*\[(?P<extras>[^\]]+)\]\s*(?P<rest>.*)$"
)


def _extras_only_target(token):
    """`name[extras]` when `token` asks for EXTRAS of a baked package, else None.

    pip and uv treat `pkg[extra]` against an already-satisfied `pkg` as "add the
    optional dependencies", not "replace pkg", so dropping the whole token loses
    every package the extra pulls in and still prints ok: `pip install
    "datasets[audio]"` became a no-op and the notebook died later on the missing
    decoder. The version specifier is dropped along with the rest of the
    requirement and _protected_constraints_file() pins the baked version, so what
    is forwarded can only ADD. A direct reference (`pkg[extra] @ url`) or a path
    IS a replacement request, so those keep being dropped."""
    if "://" in token or token.startswith((".", "/", "-")) or os.sep in token:
        return None
    m = _EXTRAS_RE.match(token.strip())
    if not m:
        return None
    rest = m.group("rest").strip()
    if rest.startswith("@"):
        return None
    out = m.group("name") + "[" + m.group("extras").strip() + "]"
    marker = rest.split(";", 1)
    if len(marker) == 2 and marker[1].strip():
        out += " ; " + marker[1].strip()
    return out


def _classify_flag_target(spec):
    name = _canon(spec)
    if name == "transformers":
        return "drop", _version_pin(spec)
    if _is_protected(name):
        return "drop", None
    return "keep", None


def _parse_flag_line(stripped, flags):
    """(flag, target_or_None, inline_comment_or_None) for a `<flag> <target>`
    requirements-file line, in any of the forms pip accepts, else (None, None, None)."""
    body, sep, comment = stripped.partition(" #")
    body = body.rstrip()
    comment = ("#" + comment) if sep else None
    for flag in flags:
        if body == flag or body.startswith(flag + " "):
            target = body[len(flag) :].strip()
        elif body.startswith(flag + "="):
            target = body[len(flag) + 1 :].strip()
        elif not flag.startswith("--") and body.startswith(flag) and len(body) > len(flag):
            target = body[len(flag) :].strip()
        else:
            continue
        return flag, (target or None), comment
    return None, None, None


def _rewrite_include(line, stripped, src_dir, depth):
    """Rewrite a nested `-r`/`-c` include -> (new_line, changed, recorded, dropped).
    pip resolves it against the directory of the file it is READING, and our filtered
    copy lives under /tmp, so a relative include must be absolutised."""
    flag, raw_target, comment = _parse_flag_line(
        stripped, ("-r", "--requirement", "-c", "--constraint")
    )
    if not raw_target:
        return line, False, None, []
    target = _expand_env_refs(raw_target)
    newline_char = "\n" if line.endswith("\n") else ""

    def _emit(new_target):
        rebuilt = flag + " " + new_target
        if comment:
            rebuilt += " " + comment
        return rebuilt + newline_char

    if "://" in target:
        return None, True, None, [flag + " " + raw_target]
    abs_target = target if os.path.isabs(target) else os.path.join(src_dir, target)
    if depth < 8:
        f_path, f_rec, f_drp = _filter_requirements_file(abs_target, _depth = depth + 1)
        # a -c include is a constraint, not an install request: no marker
        if flag in _CONSTRAINT_FILE_FLAGS:
            f_rec = None
        if f_path != abs_target:
            return _emit(f_path), True, f_rec, f_drp
    if not os.path.isabs(target):
        return _emit(abs_target), True, None, []
    return line, False, None, []


def _logical_lines(lines):
    """Group physical requirements-file lines into the LOGICAL lines pip parses.

    A trailing backslash continues onto the next line, which is exactly how
    `pip-compile --generate-hashes` / `uv pip compile --generate-hashes` write a
    pinned requirement: `torch==2.11.0 \\` then indented `--hash=...` rows.
    Filtering physical lines dropped only the first row and published the orphaned
    `--hash` rows, which uv rejects outright ("Unexpected '-', expected ... the
    start of a requirement"), killing the whole install."""
    group = []
    for line in lines:
        group.append(line)
        body = line.strip()
        if body.endswith("\\") and not body.startswith("#"):
            continue
        yield group
        group = []
    if group:
        yield group


# Where the real tool will resolve a relative `-r`/`-c` path from. None means our
# own cwd, which is what pip always uses. main() sets it for uv.
_WORKING_DIR = None


def _uv_working_dir(tool, argv):
    """The directory uv will change to before resolving a relative requirements
    path, or None. `--directory` is a uv GLOBAL, so it is accepted in every
    position: before `pip`, between `pip` and `install`, and after `install`. uv
    rejects it more than once ("cannot be used multiple times"), so the first hit
    is the only hit, and it overrides UV_WORKING_DIR. pip has no equivalent."""
    if tool != "uv":
        return None
    for n, tok in enumerate(argv):
        if tok == "--directory":
            return argv[n + 1] if n + 1 < len(argv) else None
        if tok.startswith("--directory="):
            return tok[len("--directory=") :] or None
    return os.environ.get("UV_WORKING_DIR") or None


def _resolve_read_path(path):
    """Read a relative requirements path from where the real tool will read it.
    Reading it from our cwd instead makes open() miss, and a miss is silent: the
    original relative path is forwarded unfiltered and uv, now chdir'd, installs a
    file we never inspected and never recorded a transformers pin from."""
    if not _WORKING_DIR or os.path.isabs(path):
        return path
    return os.path.join(_WORKING_DIR, path)


def _filter_requirements_file(path, _depth = 0):
    """Strip protected packages out of a `-r` file, recursing into nested includes.
    Returns (path_to_use, recorded_transformers_version, dropped_specs)."""
    read_path = _resolve_read_path(path)
    try:
        # utf-8-sig, not utf-8: a BOM would otherwise leave "\ufefftransformers==X" on
        # line 1, matching no handler, so a file whose ONLY protected pin is first
        # forwards unchanged. Both real tools strip it and honour the line.
        with open(read_path, encoding = "utf-8-sig") as f:
            lines = f.readlines()
    except OSError:
        # forward what the caller passed, not the resolved path: uv resolves it
        # itself, and pip never had a working directory to resolve against
        return path, None, []
    src_dir = os.path.dirname(os.path.abspath(read_path))
    out, dropped, recorded, changed = [], [], None, False
    for group in _logical_lines(lines):
        line = group[0]
        stripped = " ".join(part.strip().rstrip("\\").strip() for part in group).strip()
        if not stripped or stripped.startswith("#"):
            out.extend(group)
            continue
        if stripped.startswith("-"):
            e_flag, e_target, _e_comment = _parse_flag_line(stripped, ("-e", "--editable"))
            if e_target is not None:
                _action, _ver = _classify_flag_target(_expand_env_refs(e_target))
                if _action == "drop":
                    if _ver and not recorded:
                        recorded = _ver
                    dropped.append(e_flag + " " + e_target)
                    changed = True
                    continue
                out.extend(group)
                continue
            if len(group) > 1:
                out.extend(group)
                continue
            new_line, rewrote, inc_rec, inc_drp = _rewrite_include(line, stripped, src_dir, _depth)
            if new_line is not None:
                out.append(new_line)
            if rewrote:
                changed = True
            if inc_rec and not recorded:
                recorded = inc_rec
            dropped.extend(inc_drp)
            continue
        spec = stripped.split(" #", 1)[0].strip()
        # the per-requirement options a joined `--hash` block carries are noise here
        report = spec.split(" --", 1)[0].strip()
        classified = _expand_env_refs(spec)
        name = _canon(classified)
        if name is None:
            out.extend(group)
            continue
        if name == "transformers":
            v = _version_pin(classified)
            if v and not recorded:
                recorded = v
            # extras of the BAKED transformers are additive exactly as they are for
            # every _KEEP package below, and the sidecar only replaces the version:
            # dropping the whole token loses deepspeed/sentencepiece/... and still
            # reports ok. The pin is stripped, so this can only ADD.
            extras = _extras_only_target(spec) if _is_installed("transformers") else None
            if extras is not None:
                out.append(extras + ("\n" if group[-1].endswith("\n") else ""))
                changed = True
                continue
            dropped.append(report)
            changed = True
            continue
        if _is_protected(name):
            extras = _extras_only_target(spec)
            if extras is not None:
                out.append(extras + ("\n" if group[-1].endswith("\n") else ""))
                changed = True
                continue
            dropped.append(report)
            changed = True
            continue
        out.extend(group)
    if not changed:
        return path, None, []
    try:
        fd, tmp = tempfile.mkstemp(prefix = "unsloth-nb-req-", suffix = ".txt")
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.writelines(out)
    except OSError as exc:
        # fail CLOSED: forwarding the original hands pip the specs we must filter
        raise SystemExit(
            f"[unsloth-nb] could not write a filtered copy of {path} ({exc}); "
            "refusing to forward a requirements file that pins protected packages."
        )
    return tmp, recorded, dropped


def _protected_constraints_file():
    """Temp constraints file pinning every INSTALLED protected package, or None when
    the environment has none to pin. Argument filtering does not constrain the
    RESOLVER: a kept package declaring `torch==99.0` would replace the baked torch,
    and pinning makes that fail loudly. `_extras_only_target` leans on the same pin --
    it forwards `torch[opt]` with the version specifier stripped precisely because the
    pin holds the baked version, so an unconstrained forward can REPLACE rather than
    only ADD.

    So a failure to build the file fails CLOSED, exactly as _filter_requirements_file
    already does for the same mkstemp error: swallowing it returned None, which the
    caller cannot tell apart from "nothing to protect", and the install then ran with
    the whole protection silently off. Only a wholesale failure aborts; one dist whose
    metadata cannot be read is skipped, since a half-removed `.dist-info` left by an
    interrupted install is ordinary and must not block every later install."""
    try:
        from importlib.metadata import distributions

        scope = _base_site_packages()
        if not scope:
            raise RuntimeError(f"could not locate the site-packages of {REAL['pip']}")
        # scoped, NOT a bare distributions(): see _base_site_packages. An activated
        # sidecar sits ahead of the venv on PYTHONPATH and would otherwise be pinned
        # here in place of the baked version.
        dists = list(distributions(path = scope))
    except Exception as exc:
        raise SystemExit(
            f"[unsloth-nb] could not enumerate installed packages ({exc}); refusing to "
            "install without the constraints that hold the baked stack in place."
        )
    pins = {}
    for dist in dists:
        try:
            raw = (dist.metadata["Name"] or "").strip()
            version = dist.version
        except Exception:
            continue
        name = _norm_name(raw)
        if not name or not version or name in pins:
            continue
        if name == "transformers" or name in _KEEP or name.startswith(_KEEP_PREFIX):
            pins[name] = f"{raw}=={version}"
    if not pins:
        return None
    try:
        fd, tmp = tempfile.mkstemp(prefix = "unsloth-nb-protected-", suffix = ".txt")
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write("\n".join(pins[name] for name in sorted(pins)) + "\n")
    except OSError as exc:
        raise SystemExit(
            f"[unsloth-nb] could not write the protected constraints file ({exc}); "
            "refusing to install without the pins that hold the baked stack in place."
        )
    return tmp


def _selfcheck_value_flags():
    """Assert every value-taking flag the REAL pip/uv accept is classified, else the
    scanner misreads a flag's VALUE as an install target and forwards the install
    unfiltered. Run at image build time.

    pip is INTROSPECTED, not scraped. Its --help prints one spelling per option, so
    seven hidden aliases (--default-timeout among them) were invisible to the old
    scrape while the check still reported OK -- a false success guarding the guard.
    optparse knows all of them.

    uv is a binary with no such table, so it is scraped at all three levels rather
    than just `uv pip install`, and the globals it accepts but documents nowhere are
    probed directly. A level we cannot inspect is a FAILURE, never a pass.
    """
    import subprocess

    known = _VALUE_FLAGS | _DROP_VALUE_FLAGS
    missing = {}

    probe = (
        "from pip._internal.commands import create_command\n"
        "v=set()\n"
        "for o in create_command('install').parser.option_list_all:\n"
        "    if o.takes_value(): v.update(o._long_opts); v.update(o._short_opts)\n"
        "print('\\n'.join(sorted(v)))\n"
    )
    try:
        out = subprocess.run(
            [os.path.join(os.path.dirname(REAL["pip"]), "python"), "-c", probe],
            capture_output = True,
            text = True,
        )
        if out.returncode != 0 or not out.stdout.strip():
            raise OSError(out.stderr.strip()[:200] or "no output")
        gap = {f for f in out.stdout.split() if f.startswith("-")} - known
        if gap:
            missing["pip"] = sorted(gap)
    except OSError as exc:
        print(
            f"[unsloth-nb] could not introspect pip's option table ({exc}); refusing to "
            "certify the value-flag list from --help alone, which cannot see aliases.",
            file = sys.stderr,
        )
        sys.exit(1)

    seen_uv = False
    for cmd in (
        [REAL["uv"], "--help"],
        [REAL["uv"], "pip", "--help"],
        [REAL["uv"], "pip", "install", "--help"],
    ):
        try:
            out = subprocess.run(cmd, capture_output = True, text = True).stdout
        except OSError:
            continue
        seen_uv = True
        flags = set()
        for m in re.finditer(r"^\s+(-\w)?,?\s*(--[\w-]+)[= ]<", out, re.M):
            if m.group(1):
                flags.add(m.group(1))
            flags.add(m.group(2))
        for m in re.finditer(r"^\s+(-\w) <", out, re.M):
            flags.add(m.group(1))
        gap = flags - known
        if gap:
            missing.setdefault("uv", []).extend(sorted(gap))
    if not seen_uv:
        print("[unsloth-nb] could not run uv to check its flags", file = sys.stderr)
        sys.exit(1)

    # uv globals that are accepted but documented nowhere: verify each still takes a
    # value, so the list rots loudly instead of silently.
    for flag in sorted(_ALIAS_VALUE_FLAGS):
        if flag == "--python-preference":
            r = subprocess.run(
                [REAL["uv"], flag, "system", "pip", "install", "--help"],
                capture_output = True,
                text = True,
            )
            if r.returncode != 0:
                print(
                    f"[unsloth-nb] uv no longer accepts `{flag} <value>`; the entry in "
                    "_ALIAS_VALUE_FLAGS is stale and may now swallow an install target.",
                    file = sys.stderr,
                )
                sys.exit(1)

    if missing:
        print(f"[unsloth-nb] value flags missing from _VALUE_FLAGS: {missing}", file = sys.stderr)
        sys.exit(1)
    print("[unsloth-nb] value-flag selfcheck OK")
    sys.exit(0)


def _expand_short_clusters(argv):
    """Split `-qr reqs.txt` into `-q -r reqs.txt`, which is what both real tools see.

    A cluster reached no handler at all: only `tok[:2]` is tested against the
    attached-value flags, the exact-token comparisons never match, and the fallback
    keeps any unrecognised `-...` verbatim. So `pip install -qr reqs.txt` forwarded the
    ORIGINAL requirements file, unfiltered and with nothing recorded, and `-I` hid
    inside a cluster the same way and escaped the reinstall handling.

    A value-taking short flag consumes the REST of the cluster as its attached value,
    or the next argv token when it ends the cluster, exactly as getopt does.
    """
    shorts = {f for f in _VALUE_FLAGS if len(f) == 2 and f[0] == "-" and f[1] != "-"}
    shorts |= _ATTACHED_SHORT_FLAGS
    out = []
    for tok in argv:
        if len(tok) <= 2 or tok[0] != "-" or tok[1] == "-" or tok in _VALUE_FLAGS:
            out.append(tok)
            continue
        i = 1
        while i < len(tok):
            flag = "-" + tok[i]
            rest = tok[i + 1 :]
            if flag in shorts:
                out.append(flag + rest)  # attached value, or bare when rest is empty
                break
            out.append(flag)
            i += 1
        else:
            continue
    return out


def _install_index(tool, argv):
    """Index of the install SUBCOMMAND, or None when this is not a package install.

    pip's command path is `pip [opts] install`, but uv's is `uv [opts] pip [opts]
    install`, and uv has other subcommands ending in `install`. A bare
    argv.index("install") matched those too, which was not merely a wasted rewrite:
    `uv python install 3.13` and `uv tool install ruff` got the protected
    `--constraint` appended, which neither accepts, and `uv tool install transformers`
    was filtered down to nothing and reported ok, installing NO tool at all.

    So match the command path POSITIONALLY: `install` for pip, `pip install` for uv.
    Finding it needs option VALUES skipped, not just options -- `uv pip --directory
    /tmp install torch` is valid (uv lists --directory under Global options), and
    treating `/tmp` as the subcommand loses the command entirely, which is the worst
    outcome available here: the install runs unfiltered and unconstrained, free to
    replace the baked torch/CUDA stack. _VALUE_FLAGS is the same set the tail scanner
    uses and _selfcheck_value_flags() holds it to the real CLIs at build time."""
    expect = ["install"] if tool == "pip" else ["pip", "install"]
    got = _positionals(argv)
    if len(got) < len(expect):
        return None
    for (idx, tok), want in zip(got, expect):
        if tok != want:
            return None
    return got[len(expect) - 1][0]


def _positionals(argv):
    """(index, token) for every positional, skipping options AND their values.

    `--flag=value` carries its own value; a bare value flag eats the next token.
    _VALUE_FLAGS is the same set the tail scanner uses and _selfcheck_value_flags()
    holds it to the real CLIs at build time."""
    out = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok.startswith("-"):
            if "=" not in tok and tok in _VALUE_FLAGS:
                skip = True
            continue
        out.append((i, tok))
    return out


def _is_uv_pip_sync(argv):
    """`uv pip sync` is an install path we cannot make safe by filtering.

    It UNINSTALLS every package absent from the requirements file, so unlike install
    there is nothing to strip: keeping a protected package out of the file is exactly
    what deletes it, and --constraint only bounds versions, never prevents a removal.
    """
    return [tok for _, tok in _positionals(argv)][:2] == ["pip", "sync"]


def main():
    tool = "uv" if os.path.basename(sys.argv[0]).startswith("uv") else "pip"
    argv = sys.argv[1:]

    if argv[:1] == ["--unsloth-selfcheck-value-flags"]:
        _selfcheck_value_flags()

    if os.environ.get("UNSLOTH_NB_SHIM") != "1":
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    if tool == "uv" and _is_uv_pip_sync(argv):
        raise SystemExit(
            "[unsloth-nb] refusing `uv pip sync`: it UNINSTALLS every package missing "
            "from the requirements file, which would strip the baked torch/CUDA stack "
            "this image is built on, and no --constraint can prevent a removal. Use "
            "`uv pip install -r <file>`, which adds without removing."
        )

    global _WORKING_DIR
    _wd = _uv_working_dir(tool, argv)
    _WORKING_DIR = os.path.abspath(_wd) if _wd else None

    i = _install_index(tool, argv)
    if i is None:
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    head, tail = argv[: i + 1], _expand_short_clusters(argv[i + 1 :])
    keep_args, dropped, recorded = [], [], None
    extras_only = []
    has_target = False
    skip_next = False
    prev_flag = None
    for tok in tail:
        if skip_next:
            if prev_flag in _REQ_FILE_FLAGS or prev_flag in _CONSTRAINT_FILE_FLAGS:
                if "://" in tok:
                    if keep_args and keep_args[-1] == prev_flag:
                        keep_args.pop()
                    dropped.append(prev_flag + " " + tok)
                elif prev_flag in _REQ_FILE_FLAGS:
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(tok)
                    keep_args.append(_req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                else:
                    _c_path, _c_rec, _c_drp = _filter_requirements_file(tok)
                    keep_args.append(_c_path)
                    dropped.extend(_c_drp)
            elif prev_flag in _DROP_VALUE_FLAGS:
                if keep_args and keep_args[-1] == prev_flag:
                    keep_args.pop()
                dropped.append(prev_flag + " " + tok)
            elif prev_flag in _EDITABLE_FLAGS or prev_flag in _UPGRADE_PKG_FLAGS:
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
                if prev_flag in _TARGET_VALUE_FLAGS:
                    has_target = True
            skip_next = False
            prev_flag = None
            continue
        if tok.startswith("--") and "=" in tok:
            _flag, _, _val = tok.partition("=")
            if _flag in _VALUE_FLAGS:
                if (_flag in _REQ_FILE_FLAGS or _flag in _CONSTRAINT_FILE_FLAGS) and "://" in _val:
                    dropped.append(tok)
                elif _flag in _REQ_FILE_FLAGS:
                    _req_path, _req_rec, _req_drp = _filter_requirements_file(_val)
                    keep_args.append(_flag + "=" + _req_path)
                    has_target = True
                    if _req_rec and not recorded:
                        recorded = _req_rec
                    dropped.extend(_req_drp)
                elif _flag in _DROP_VALUE_FLAGS:
                    dropped.append(tok)
                elif _flag in _CONSTRAINT_FILE_FLAGS:
                    _c_path, _c_rec, _c_drp = _filter_requirements_file(_val)
                    keep_args.append(_flag + "=" + _c_path)
                    dropped.extend(_c_drp)
                elif _flag in _EDITABLE_FLAGS or _flag in _UPGRADE_PKG_FLAGS:
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
                    keep_args.append(tok)
                    if _flag in _TARGET_VALUE_FLAGS:
                        has_target = True
                continue
        if len(tok) > 2 and tok[0] == "-" and tok[1] != "-" and tok[:2] in _ATTACHED_SHORT_FLAGS:
            _sflag, _sval = tok[:2], tok[2:]
            if (_sflag in _REQ_FILE_FLAGS or _sflag in _CONSTRAINT_FILE_FLAGS) and "://" in _sval:
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
            else:  # -e / -P
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
            dropped.append(tok)
            continue
        if tok in _VALUE_FLAGS:
            # hold -e/-P back so skip_next emits or drops the pair together
            if tok not in _EDITABLE_FLAGS and tok not in _UPGRADE_PKG_FLAGS:
                keep_args.append(tok)
            skip_next = True
            prev_flag = tok
            continue
        name = _canon(tok)
        if name is None:
            keep_args.append(tok)
            if not tok.startswith("-"):
                has_target = True
            continue
        if name == "transformers":
            v = _version_pin(tok)
            if v:
                recorded = v
            # same additive-extras rule as the protected branch below: the version is
            # what the sidecar replaces, so only the pin is suppressed here
            extras = _extras_only_target(tok) if _is_installed("transformers") else None
            if extras is not None:
                keep_args.append(extras)
                extras_only.append(extras)
                has_target = True
                continue
            dropped.append(tok)
            continue
        if _is_protected(name):
            extras = _extras_only_target(tok)
            if extras is not None:
                keep_args.append(extras)
                extras_only.append(extras)
                has_target = True
                continue
            dropped.append(tok)
            continue
        keep_args.append(tok)
        has_target = True

    if extras_only:
        print("[unsloth-nb] kept baked versions, adding extras only: " + " ".join(extras_only))
    if recorded:
        try:
            parent = os.path.dirname(MARKER)
            if parent:  # "" for a bare relative MARKER, and makedirs("") raises
                os.makedirs(parent, exist_ok = True)
            with open(MARKER, "w") as f:
                f.write(recorded)
        except OSError as exc:
            # do NOT abort: nothing has been installed or mis-forwarded, and the baked
            # transformers still works, so hard-failing a notebook cell over an
            # unwritable path the user cannot act on is the worse trade. But do not
            # report success either: transformers is already out of the real arguments,
            # so staying silent leaves the cell claiming the pin was honoured while the
            # model cells import the baked version.
            print(
                f"[unsloth-nb] WARNING: could not record the requested "
                f"transformers=={recorded} at {MARKER} ({exc}); the sidecar will NOT "
                f"activate and the model cells will use the baked transformers.",
                file = sys.stderr,
            )
        else:
            print(
                f"[unsloth-nb] notebook requested transformers=={recorded}; will "
                f"activate its sidecar for the model cells (base stack kept)."
            )
    if dropped:
        print("[unsloth-nb] kept baked versions, skipped: " + " ".join(dropped))

    # only baked packages left: no-op rather than exec a bare install that fails
    if not has_target:
        print("[unsloth-nb] nothing to install after keeping the baked stack; ok.")
        return
    constraints = _protected_constraints_file()
    if constraints:
        # `--` ends option parsing for both real tools, so the pair must go first
        try:
            _eoo = keep_args.index("--")
        except ValueError:
            keep_args += ["--constraint", constraints]
        else:
            keep_args[_eoo:_eoo] = ["--constraint", constraints]
    cmd = [REAL[tool]] + head + keep_args
    sys.stdout.flush()
    os.execv(REAL[tool], cmd)


if __name__ == "__main__":
    main()
