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
_DROP_VALUE_FLAGS = {"--upgrade-strategy"}

# these ARE the install target, with no package on the command line: without them the
# shim scans to no target and no-ops while printing "ok"
_TARGET_VALUE_FLAGS = {"--group", "--upgrade-group", "--requirements-from-script"}


_ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip")


def _norm_name(name):
    """PEP 503 normalised name; every _KEEP / _KEEP_PREFIX comparison goes through here."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower() or None


_INSTALLED_NAMES = []


def _installed_names():
    """Every distribution name in this venv, PEP 503 normalised, or None when the scan
    fails. Cached because importlib.metadata rescans sys.path on every call."""
    if not _INSTALLED_NAMES:
        try:
            from importlib.metadata import distributions

            found = set()
            for dist in distributions():
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
    name = re.split(r"[<>=!~\[\s;@]", token, 1)[0].strip()
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


def _filter_requirements_file(path, _depth = 0):
    """Strip protected packages out of a `-r` file, recursing into nested includes.
    Returns (path_to_use, recorded_transformers_version, dropped_specs)."""
    try:
        with open(path, encoding = "utf-8") as f:
            lines = f.readlines()
    except OSError:
        return path, None, []
    src_dir = os.path.dirname(os.path.abspath(path))
    out, dropped, recorded, changed = [], [], None, False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
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
                out.append(line)
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
        classified = _expand_env_refs(spec)
        name = _canon(classified)
        if name is None:
            out.append(line)
            continue
        if name == "transformers":
            v = _version_pin(classified)
            if v and not recorded:
                recorded = v
            dropped.append(spec)
            changed = True
            continue
        if _is_protected(name):
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
        # fail CLOSED: forwarding the original hands pip the specs we must filter
        raise SystemExit(
            f"[unsloth-nb] could not write a filtered copy of {path} ({exc}); "
            "refusing to forward a requirements file that pins protected packages."
        )
    return tmp, recorded, dropped


def _protected_constraints_file():
    """Temp constraints file pinning every INSTALLED protected package, else None.
    Argument filtering does not constrain the RESOLVER: a kept package declaring
    `torch==99.0` would replace the baked torch, and pinning makes that fail loudly."""
    try:
        from importlib.metadata import distributions

        pins = {}
        for dist in distributions():
            raw = (dist.metadata["Name"] or "").strip()
            name = _norm_name(raw)
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
    """Assert every value-taking flag the REAL pip/uv document is classified, else the
    scanner misreads its VALUE as an install target. Run at image build time."""
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
            continue
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

    if os.environ.get("UNSLOTH_NB_SHIM") != "1":
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    try:
        i = argv.index("install")
    except ValueError:
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    head, tail = argv[: i + 1], argv[i + 1 :]
    keep_args, dropped, recorded = [], [], None
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
            dropped.append(tok)
            continue
        if _is_protected(name):
            dropped.append(tok)
            continue
        keep_args.append(tok)
        has_target = True

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
