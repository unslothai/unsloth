#!/usr/bin/env python3
# coding: utf-8
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""
Static + lightweight-dynamic validator for unslothai/notebooks.

Built to catch the bug classes that landed in (at minimum):
- unslothai/notebooks#258  (Colab torchao 0.10 vs peft 0.19 floor)
- unslothai/notebooks#260  (DONT_UPDATE_EXCEPTIONS coverage drift)
- unslothai/notebooks#261  (torch/torchcodec ABI; --no-deps tokenizers)
- unslothai/notebooks#264  (transformers/tokenizers window with --no-deps)
- unslothai/notebooks#221  (removed unsloth APIs in user cells, git+ install)
- unslothai/notebooks  commit 51b1462 (template/notebook drift)

CPU-only by design: never imports torch / unsloth at module load. The
api subcommand introspects unsloth under the existing
tests/_zoo_aggressive_cuda_spoof.py harness (PR #5312) so it works on
ubuntu-latest without a GPU.

Usage:
  python scripts/notebook_validator.py drift       --notebooks-dir <dir>
  python scripts/notebook_validator.py convert     --notebooks-dir <dir> --out _converted
  python scripts/notebook_validator.py lint        --notebooks-dir <dir> [--colab-pin <file>]
  python scripts/notebook_validator.py exceptions  --notebooks-dir <dir>
  python scripts/notebook_validator.py api         --converted-dir _converted --surface _api_surface.json
  python scripts/notebook_validator.py all         --notebooks-dir <dir>
  python scripts/notebook_validator.py refresh-colab --out scripts/data/colab_pip_freeze.gpu.txt
  python scripts/notebook_validator.py refresh-colab --all --snapshot-dir scripts/data
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import json
import os
import pathlib
import shutil
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Iterator


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    """Atomic write (see scripts/scan_packages.py::update_req_file). A crash between mkstemp and os.replace leaves the prior file intact, so a half-downloaded cache file cannot poison later runs."""
    path.parent.mkdir(parents = True, exist_ok = True)
    dirpath = str(path.parent) or "."
    fd, tmp_path = tempfile.mkstemp(prefix = ".nb_val.", dir = dirpath)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PYPI_CACHE_DIR = DATA_DIR / "pypi_cache"

COLAB_PIP_FREEZE_URL = (
    "https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.gpu.txt"
)
COLAB_FALLBACK_FILE = DATA_DIR / "colab_pip_freeze.gpu.txt"

# The image's Python, from the os-info oracle; an unreadable snapshot replays every requirement.
_COLAB_OS_INFO_FILE = DATA_DIR / "colab_os_info.gpu.txt"
# Keep the prerelease suffix: pip skips `python_full_version >= "3.13.0"` on 3.13.0rc1.
_COLAB_PYTHON_RE = re.compile(
    r"^Python\s+(\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)",
    re.MULTILINE,
)
_PYTHON_RELEASE_RE = re.compile(r"\d+(?:\.\d+)*")


# Follows `lint --colab-pin` so Python version and package snapshot come from the same capture.
_COLAB_ORACLE_DIR: pathlib.Path = DATA_DIR


def _set_colab_oracle_dir(directory: pathlib.Path) -> None:
    global _COLAB_ORACLE_DIR
    _COLAB_ORACLE_DIR = directory
    _colab_python_version.cache_clear()


@functools.lru_cache(maxsize = 1)
def _colab_python_version() -> str | None:
    try:
        text = (_COLAB_ORACLE_DIR / _COLAB_OS_INFO_FILE.name).read_text(encoding = "utf-8")
    except OSError:
        return None
    match = _COLAB_PYTHON_RE.search(text)
    return match.group(1) if match else None


def _marker_environment(colab: dict[str, str]) -> dict[str, str] | None:
    """The environment PEP 508 markers are evaluated against, or None to skip them: only the Colab image, the one environment this can name, since anything else replays every requirement."""
    if not colab:
        return None
    full = _colab_python_version()
    if not full:
        return None
    release = _PYTHON_RELEASE_RE.match(full).group(0)
    suffix = full[len(release) :]
    parts = release.split(".")
    return {
        "python_version": ".".join(parts[:2]),
        "python_full_version": (release if len(parts) > 2 else f"{release}.0") + suffix,
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
        # A top-level requirement selects no extra, so `extra == "..."` markers are false.
        "extra": "",
    }


# Marker.evaluate fills omitted fields from the running process, so pin them all here.
_MARKER_VARIABLES = frozenset(
    {
        "os_name",
        "sys_platform",
        "platform_machine",
        "platform_python_implementation",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_version",
        "python_full_version",
        "implementation_name",
        "implementation_version",
        "extra",
    }
)


def _requirement_applies(raw: str, environment: dict[str, str] | None) -> bool:
    """False only when the requirement carries a marker that is false for `environment`. pip skips such a requirement, so replaying its bounds moves a version the cell never touches; anything unjudgeable (unparseable marker, no `packaging`, no environment) is replayed."""
    if environment is None or ";" not in raw:
        return True
    marker_text = raw.split(";", 1)[1].strip()
    if not marker_text:
        return True
    try:
        return _marker_truth(marker_text, environment) is not False
    except Exception:
        return True


def _marker_variables(text: str) -> set[str]:
    """The marker fields a term references, string literals excluded: `sys_platform ==
    'platform_release'` references one variable, not two."""
    bare = re.sub(r"\"[^\"]*\"|'[^']*'", " ", text)
    return set(re.findall(r"[A-Za-z_]\w*", bare)) & _MARKER_VARIABLES


def _split_marker(text: str) -> tuple[list[str], list[str]]:
    """A marker's top-level terms and the `and`/`or` between them, quotes and parens intact."""
    terms: list[str] = []
    operators: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        joiner = re.match(r"(and|or)\b", text[i:], re.IGNORECASE) if not depth else None
        if joiner is not None and (i == 0 or text[i - 1].isspace()):
            terms.append("".join(buf))
            buf = []
            operators.append(joiner.group(1).lower())
            i += joiner.end()
            continue
        buf.append(ch)
        i += 1
    terms.append("".join(buf))
    return terms, operators


def _marker_truth(text: str, environment: dict[str, str]) -> bool | None:
    """Three-valued marker evaluation: True, False, or unknown. An unanswerable field makes its own TERM unknown, not the whole marker: a decisive `python_version < '3.0' and implementation_name == 'cpython'` stays false on a 3.13 image."""
    text = text.strip()
    if not text:
        return None
    terms, operators = _split_marker(text)
    if len(terms) > 1:
        values = [_marker_truth(term, environment) for term in terms]
        groups: list[list[bool | None]] = [[values[0]]]
        for operator, value in zip(operators, values[1:]):
            if operator == "and":
                groups[-1].append(value)
            else:
                groups.append([value])
        folded = [
            False
            if any(v is False for v in group)
            else (True if all(v is True for v in group) else None)
            for group in groups
        ]
        if any(v is True for v in folded):
            return True
        return False if all(v is False for v in folded) else None
    term = terms[0].strip()
    if term.startswith("(") and term.endswith(")"):
        return _marker_truth(term[1:-1], environment)
    named = _marker_variables(term)
    if not named or named - environment.keys():
        return None
    from packaging.markers import Marker

    return bool(Marker(term).evaluate(environment))


COLAB_ORACLE_FILES: dict[str, str] = {
    "pip-freeze.gpu.txt": "colab_pip_freeze.gpu.txt",
    "apt-list-gpu.txt": "colab_apt_list.gpu.txt",
    "os-info-gpu.txt": "colab_os_info.gpu.txt",
}
# The pip oracle and the os-info Python line are strict; other oracle drift is advisory.
COLAB_STRICT_ORACLE = "pip-freeze.gpu.txt"
COLAB_STRICT_ORACLE_KEYS: dict[str, frozenset[str]] = {
    "os-info-gpu.txt": frozenset({"python"}),
}
COLAB_ORACLE_BASE_URL = "https://raw.githubusercontent.com/googlecolab/backend-info/main/"

# Lockstep rows only: torchcodec 0.12+ is ABI-stable against torch >=2.11 and is handled by
# the short-circuit in rule_inst_004_torchcodec_torch rather than by a row here.
TORCHCODEC_ABI_STABLE_TORCH = "2.11"
TORCHCODEC_ABI_STABLE_CODEC = "0.12"

# torch.minor -> compatible torchcodec minors, from the torchcodec README matrix.
# Mirrors import_fixes._TORCH_TORCHCODEC_MINORS (test_torchcodec_torch_compat asserts equality).
TORCH_TORCHCODEC: dict[str, set[str]] = {
    "2.11": {"0.11"},
    "2.10": {"0.10"},
    "2.9": {"0.8", "0.9"},
    "2.8": {"0.6", "0.7"},
    "2.7": {"0.3", "0.4", "0.5"},
    "2.6": {"0.2"},
    "2.5": {"0.1"},
}

PEFT_TORCHAO_FLOOR: list[dict[str, str]] = [
    {"trigger_peft": "0.19", "torchao_floor": "0.16.0"},
]

GIT_PLUS_ALLOWLIST = (
    "github.com/SparkAudio/Spark-TTS",
    "github.com/state-spaces/mamba",
    "github.com/Dao-AILab/causal-conv1d",
    "github.com/unslothai/unsloth-zoo",
    "github.com/unslothai/unsloth",
)


@dataclasses.dataclass
class Finding:
    rule: str
    file: str
    cell: int | None = None
    line: int | None = None
    severity: str = "error"
    message: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def iter_notebooks(
    notebooks_dir: pathlib.Path, include_templates: bool = False
) -> Iterator[pathlib.Path]:
    """Yield user-facing .ipynb files under nb/ and kaggle/. include_templates=True also walks original_template/ (for convert)."""
    subs = ("nb", "kaggle")
    if include_templates:
        subs = ("nb", "kaggle", "original_template")
    candidates = []
    for sub in subs:
        d = notebooks_dir / sub
        if d.is_dir():
            for p in sorted(d.glob("*.ipynb")):
                candidates.append(p)
    seen = set()
    for p in candidates:
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        yield p


def load_notebook(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding = "utf-8"))


def cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def code_cells(nb: dict[str, Any]) -> list[tuple[int, str]]:
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code":
            out.append((i, cell_source(c)))
    return out


# Anchored on `!` so a pip string in Python is not a cell; `-mpip` counts (no \b before p).
_PIP_CELL_RE = re.compile(
    r"^[ \t]*!.*(?:\b(?:uv\s+)?pip|-m(?:uv\s+)?pip)\s+(?:install|uninstall)\b",
    re.MULTILINE,
)


def install_cells(nb: dict[str, Any]) -> list[tuple[int, str]]:
    """Heuristic: any code cell that contains a `pip install`, `pip uninstall` or `uv pip install` shell command, or a top-line `%%capture` magic."""
    out = []
    for i, src in code_cells(nb):
        first = src.lstrip().splitlines()[:1]
        if first and first[0].strip().startswith("%%capture"):
            out.append((i, src))
            continue
        # Glued, since a `\` continuation can split the `!` from the pip call.
        if any(_PIP_CELL_RE.search(line) for _, line in _glue_line_continuations(src)):
            out.append((i, src))
    return out


# The Colab oracle applies only to notebooks that run on Colab.
def target_environment(notebook_name: str) -> str:
    parts = pathlib.PurePath(notebook_name).parts
    base = parts[-1] if parts else notebook_name
    parent = parts[-2] if len(parts) >= 2 else ""
    if parent == "kaggle" or base.startswith("Kaggle-"):
        return "kaggle"
    if base.startswith("AMD-") or "_AMD_" in base:
        return "amd"
    if base.startswith("HuggingFace Course-") or base.startswith("HuggingFace_Course-"):
        return "colab"
    if "DGX_Spark" in base:
        return "dgx_spark"
    return "colab"


PINNED_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)")


def parse_pip_freeze(path: pathlib.Path) -> dict[str, str]:
    """Return {name_lower: version_str_with_local_version}."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding = "utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = PINNED_RE.match(line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def normalise_version(v: str) -> str:
    """Strip +cu128 / +cpu / -dev local-version metadata."""
    return re.split(r"[+\-]", v, maxsplit = 1)[0]


def version_minor(v: str) -> str:
    parts = normalise_version(v).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


# PEP 440 orders every one of these below the plain release.
_PRERELEASE_RE = re.compile(r"(?:a|b|c|rc|alpha|beta|pre|preview|dev)\d*$", re.IGNORECASE)


def _split_prerelease(version: str) -> tuple[str, bool]:
    """`(release core, is a prerelease)`, hyphenated PEP 440 spellings included.

    PEP 440 spells one prerelease `2.11.0rc1`, `2.11.0-rc1` and `2.11.0.rc1`, and cutting at the
    hyphen read the last two as a plain `2.11.0` above the ABI floor they sit below."""
    text = str(version).split("+", 1)[0].strip().lower()
    text = re.sub(r"[-_.]?(a|b|c|rc|alpha|beta|pre|preview|dev)[-_.]?(\d*)$", r"\1\2", text)
    match = _PRERELEASE_RE.search(text)
    if match is None:
        return text, False
    return text[: match.start()].rstrip("-_."), True


def _is_prerelease(version: str) -> bool:
    """Does this version sort below the release with the same numbers?"""
    return _split_prerelease(version)[1]


def at_least(version: str, floor: str) -> bool:
    """`version >= floor` with PEP 440's prerelease ordering.

    Dotted digits alone read `2.11.0rc1` as 2.11.0.1, above `2.11`, which approved a pairing
    outside the ABI-stable contract."""
    # Split the suffix first, or 2.11.0rc1 compares as 2.11.0.1.
    core, prerelease = _split_prerelease(version)
    order = cmp_versions(core, floor)
    if order != 0:
        return order > 0
    return not prerelease


_PRERELEASE_ORDER = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "c": 3,
    "rc": 3,
    "pre": 3,
    "preview": 3,
}


def _prerelease_key(version: str) -> tuple[int, int]:
    """How a version's prerelease suffix sorts: the release itself is above every one.

    Reading the suffix's digits as another component put `0.12.0rc1` above `0.12.0`, so a floor the
    cell upgrades past looked already met."""
    core, is_pre = _split_prerelease(version)
    if not is_pre:
        return (len(_PRERELEASE_ORDER) + 1, 0)
    match = _PRERELEASE_RE.search(str(version).split("+", 1)[0].strip().lower())
    if match is None:
        return (len(_PRERELEASE_ORDER) + 1, 0)
    text = match.group(0)
    digits = re.search(r"\d+$", text)
    phase = text[: digits.start()] if digits else text
    return (_PRERELEASE_ORDER.get(phase, 0), int(digits.group(0)) if digits else 0)


def cmp_versions(a: str, b: str) -> int:
    """Return -1/0/+1, PEP 440 order over the release core and its prerelease suffix."""

    def to_tuple(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", normalise_version(_split_prerelease(v)[0])))

    ta, tb = to_tuple(a), to_tuple(b)
    # PEP 440 zero-pads the shorter release, so 0.11 == 0.11.0.
    width = max(len(ta), len(tb))
    ta = ta + (0,) * (width - len(ta))
    tb = tb + (0,) * (width - len(tb))
    # The suffix only breaks a tie on the release core.
    ka, kb = ta + _prerelease_key(a), tb + _prerelease_key(b)
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


@dataclasses.dataclass
class PipInvocation:
    tool: str
    flags: set[str]
    packages: list[str]
    raw: str
    line_no: int = 0
    action: str = "install"
    conditional: bool = False  # the fallback side of an `||`: runs only if the left failed


# `python -m pip` parses as bare pip; in step with unsloth_nb_pip_magic.py::_PY_M_PIP.
_INTERPRETER_RE = r"""(?:
        (?:python[0-9.]*|py)
      | ["']?\{\s*sys\.executable\s*\}["']?
      | "(?:[^"]*[/\\])python[0-9.]*(?:\.exe)?"
      | '(?:[^']*[/\\])python[0-9.]*(?:\.exe)?'
      | \S*[/\\]python[0-9.]*(?:\.exe)?
    )"""
# `-m uv pip` too: unsloth_nb_pip_magic rewrites `(pip|uv)` after the module flag.
PIP_LINE_RE = re.compile(
    # Interpreter options may precede -m. `!\s*!` covers bash negation and IPython `!!`.
    r"^\s*!(?:\s*!)*\s*(?P<tool>(?:uv\s+)?pip|"
    + _INTERPRETER_RE
    # -W/-X take an operand; -h, -V, -? and -c end option parsing, so nothing after is an option.
    + r"(?:\s+-[WX]\s*\S+|\s+--check-hash-based-pycs\s+\S+|\s+-(?![hVc?])[A-Za-z]\w*)*"
    # `-m` may be attached: `python -mpip`.
    + r"\s+-m\s*(?:uv\s+)?pip)\s+"
    r"(?P<action>install|uninstall)\b(?P<rest>.*)$",
    re.IGNORECASE | re.VERBOSE,
)
NON_PKG_FLAG_TAKES_VAL = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "--find-links",
    "-e",
    "--editable",
    "--target",
    "--prefix",
}


def parse_pip_line(line: str, line_no: int = 0) -> PipInvocation | None:
    m = PIP_LINE_RE.match(line)
    if not m:
        return None
    # Do not read a plain `python3 -m pip` as uv because "uv" appears in the interpreter path.
    tool = "uv-pip" if re.search(r"(?:^|\s)uv\s+pip\b", m.group("tool"), re.IGNORECASE) else "pip"
    rest = m.group("rest")
    rest = re.split(r"(?<!\S)#", rest, maxsplit = 1)[0]
    try:
        tokens = shlex.split(rest, posix = True)
    except ValueError:
        rest_safe = re.sub(r"\{[^}]+\}", "PLACEHOLDER", rest)
        try:
            tokens = shlex.split(rest_safe, posix = True)
        except ValueError:
            return None
    flags: set[str] = set()
    packages: list[str] = []
    skip_next = False
    for t in tokens:
        if skip_next:
            skip_next = False
            continue
        if t in NON_PKG_FLAG_TAKES_VAL:
            flags.add(t)
            skip_next = True
            continue
        if t.startswith("-"):
            flags.add(t)
            continue
        if t in ("install", "uninstall"):
            continue
        packages.append(t)
    return PipInvocation(
        tool = tool,
        flags = flags,
        packages = packages,
        raw = line,
        line_no = line_no,
        action = m.group("action").lower(),
    )


def _glue_line_continuations(text: str) -> list[tuple[int, str]]:
    """Return (logical_line_no, joined_text) for each logical line, treating a trailing backslash as a continuation. Logical line numbers point at the first physical line of each logical line."""
    out: list[tuple[int, str]] = []
    buf = ""
    start = 0
    for i, raw in enumerate(text.splitlines(), start = 1):
        if buf == "":
            start = i
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
        else:
            buf += raw
            out.append((start, buf))
            buf = ""
    if buf:
        out.append((start, buf))
    return out


# Exec prefixes run the command after them (`env FOO=1 pip ...`). Bare `time` is bash's reserved
# word; only an explicit path reaches GNU time.
_GNU_TIME = "/usr/bin/time"
_SHELL_EXEC_PREFIXES = frozenset({"command", "env", "exec", "nohup", "time", "sudo", _GNU_TIME})
# Resolved in-process by bash; after any other prefix, `exec` is just an argument.
_SHELL_RESOLVED_PREFIXES = frozenset({"command", "exec", "time"})
# External programs, so an absolute path names the same one.
_PATH_QUALIFIED_PREFIXES = frozenset({"env", "nohup", "sudo"})
_PREFIX_TERMINAL_FLAGS = frozenset({"--help", "--version"})
# Options that turn a prefix into a lookup rather than an execution (validate, list).
_PREFIX_LOOKUP_FLAGS: dict[str, frozenset[str]] = {
    "command": frozenset({"-v", "-V"}),
    "sudo": frozenset({"-v", "--validate", "-l", "--list", "-V", "-h"}),
}
# `PATH+=:/opt/bin cmd` is an assignment prefix too.
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_]\w*\+?=")
# Options taking a separate operand; env's split-string operand is the command it runs.
_ENV_SPLIT_STRING_FLAGS = frozenset({"-S", "--split-string"})


def _env_split_string(raw: str) -> str:
    """One shell word off an `env -S` operand, with `\\_` restored to the separator it is.

    GNU env splits the operand on whitespace and documents `\\_` as a space; verified with
    coreutils 9.4, where `env -S 'printf [%s][%s] a\\_b'` prints `[a][b]` exactly as a plain space
    does. bash keeps that backslash inside double quotes, so unescaping the operand as an ordinary
    shell word rebuilt `pip install_git+...` and no invocation was seen at all."""
    return _split_first_word(raw.replace("\\_", " "))[0]


_PREFIX_OPERAND_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "sudo": frozenset(
        {
            "-u",
            "--user",
            "-g",
            "--group",
            "-p",
            "--prompt",
            "-C",
            "--close-from",
            "-r",
            "--role",
            "-t",
            "--type",
            "-U",
            "--other-user",
            "-h",
            "--host",
            "-D",
            "--chdir",
            "-R",
            "--chroot",
            "-T",
            "--command-timeout",
        }
    ),
    "exec": frozenset({"-a"}),
    # Bash's reserved `time` takes none of GNU time's options.
    "time": frozenset(),
    _GNU_TIME: frozenset({"-f", "--format", "-o", "--output"}),
    "command": frozenset(),
    "nohup": frozenset(),
    "builtin": frozenset(),
}


def _split_first_word(text: str) -> tuple[str, str]:
    """One shell word off the front, plus the RAW remainder.

    A shell word may contain whitespace, so `str.split` cut `env TOKEN="a b" pip install ...` into
    `env` / `TOKEN="a` / `b" pip ...` and read the fragment as the executable. The word comes back
    unquoted, for comparing against prefix names; the remainder verbatim, since everything
    downstream re-parses the original text."""
    index, length = 0, len(text)
    while index < length and text[index].isspace():
        index += 1
    word: list[str] = []
    quote = ""
    depth = 0
    # Track `${ }` too: bash keeps `TOKEN=${TOKEN:-a b}` as one word.
    brace = 0
    # Inside an open case an unbalanced `)` ends the arm pattern, not the substitution.
    case_depth = 0
    backtick = False
    while index < length:
        ch = text[index]
        if quote:
            if ch == "\\" and quote == '"' and index + 1 < length:
                index += 1
                word.append(text[index])
            elif ch == quote:
                quote = ""
            else:
                word.append(ch)
        elif ch == "\\" and index + 1 < length:
            index += 1
            word.append(text[index])
        elif backtick:
            if ch == "`":
                backtick = False
            word.append(ch)
        elif brace:
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
            word.append(ch)
        elif depth:
            # A substitution's whitespace and quotes belong to the word.
            if ch in "'\"":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                if not (case_depth and depth == 1):
                    depth -= 1
            elif ch.isalpha() and not text[index - 1 : index].isalnum():
                keyword = _LEADING_WORD_RE.match(text, index)
                if keyword is not None:
                    if keyword.group(0) == "case":
                        case_depth += 1
                    elif keyword.group(0) == "esac" and case_depth:
                        case_depth -= 1
            word.append(ch)
        elif ch in "'\"":
            quote = ch
        elif ch == "`":
            backtick = True
            word.append(ch)
        elif ch == "$" and text[index + 1 : index + 2] == "(":
            depth += 1
            word.append(ch)
            index += 1
            word.append(text[index])
        elif ch == "$" and text[index + 1 : index + 2] == "{":
            brace += 1
            word.append(ch)
            index += 1
            word.append(text[index])
        elif ch.isspace():
            break
        else:
            word.append(ch)
        index += 1
    return "".join(word), text[index:].strip()


def _strip_exec_prefixes(text: str, seen: list[str] | None = None) -> tuple[str, bool]:
    """Drop `env -u VAR`, `sudo -u root`, `nohup`, `A=1` ... and return the command they run.

    Positional, not pattern-matched: each prefix's options and operands are consumed in order, so
    the executable is whatever word is left, even when an operand is spelled `pip`. `seen` collects
    the prefix names in order, for the caller that has to know which ran: `command exec pip ...`
    hands the shell over exactly as `exec pip` does."""
    prefixed = False
    while True:
        word, rest = _split_first_word(text)
        if not word:
            break
        if _ENV_ASSIGNMENT_RE.match(word):
            prefixed = True
            text = rest
            continue
        if _REDIRECTION_RE.match(word):
            # A redirection may precede the command name.
            prefixed = True
            text = _split_first_word(rest)[1] if _REDIRECTION_RE.fullmatch(word) else rest
            continue
        name = word.lower()
        if name.endswith("/time"):
            name = _GNU_TIME  # an explicit path runs the BINARY, which takes GNU's options
        elif "/" in name and name.rsplit("/", 1)[1] in _PATH_QUALIFIED_PREFIXES:
            # Path-qualified external prefixes only; command and exec are builtins.
            name = name.rsplit("/", 1)[1]
        if name not in _SHELL_EXEC_PREFIXES:
            break
        if seen is not None:
            seen.append(name)
        prefixed = True
        operand_flags = _PREFIX_OPERAND_FLAGS.get(name, frozenset())
        while rest:
            token, tail = _split_first_word(rest)
            if token == "--":
                rest = tail
                break
            if token == "-" or not token.startswith("-"):
                break
            if name == "env" and token.startswith("-S") and len(token) > 2:
                # env -S takes a mandatory operand, so an attached `-S'pip install'` is valid.
                raw = rest[: len(rest) - len(tail)].strip()
                rest = f"{_env_split_string(raw[2:])} {tail}".strip()
                break
            if token.startswith("--split-string=") and name == "env":
                raw = rest[: len(rest) - len(tail)].strip()
                rest = f"{_env_split_string(raw.partition('=')[2])} {tail}".strip()
                break
            if token in _PREFIX_TERMINAL_FLAGS or token in _PREFIX_LOOKUP_FLAGS.get(
                name, frozenset()
            ):
                # `env --help` and `command -v` do not run their operands.
                return text, prefixed
            if "=" in token and token.startswith("--"):
                rest = tail
                continue
            if name == "env" and token in _ENV_SPLIT_STRING_FLAGS:
                # env -S's operand is the command; following ARGs are appended to it.
                trailing = _split_first_word(tail)[1]
                raw = tail[: len(tail) - len(trailing)]
                rest = f"{_env_split_string(raw)} {trailing}".strip()
                break
            if token in operand_flags:
                _, rest = _split_first_word(tail)
            else:
                rest = tail
        text = rest
    return text, prefixed


_LEADING_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_SHELL_TEST_KEYWORDS = frozenset({"if", "while", "until", "for", "case"})
_SHELL_BODY_KEYWORDS = frozenset({"then", "elif", "else", "do"})
_SHELL_KEYWORDS = _SHELL_TEST_KEYWORDS | _SHELL_BODY_KEYWORDS | {"fi", "done", "esac"}


def _unquoted_arm_close(text: str) -> int | None:
    """Index of the `)` that closes a case-arm pattern, or None when there is none.

    Shell quoting decides: `"x")` is a pattern, while the `)` in `pip install "a)b"` and in a `$(
    )` substitution belongs to the command."""
    quote = ""
    depth = 0
    opened = False
    index = -1
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote != "'":
            escaped = True  # `x\\)y)` matches a literal `)`, so only the second one closes
            continue
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
            opened = True
        elif ch == ")":
            if depth:
                depth -= 1
            elif opened:
                # A `(` closed before this bracket means a substitution, not an arm label.
                return None
            elif index:
                return index
            else:
                return None
    return None


def _final_bracket_closes_substitution(text: str) -> bool:
    """True when the last character closes a `$( )`, `<( )` or `>( )` opened in `text`.

    That `)` is part of the command and must survive the grouping-bracket strip; a `)` or `}` that
    closes a plain group, or one left over from a group that spanned a separator, is not."""
    if not text.endswith(")"):
        return False
    quote = ""
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if text.startswith("$(", i) or (ch in "<>" and text[i + 1 : i + 2] == "("):
            depth += 1
            i += 2
            continue
        if ch == "(" and depth:
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0 and i == len(text) - 1:
                return True
        i += 1
    return False


# Function headers need empty parens; a body runs only when called, so it is conditional.
_FUNCTION_NAME_RE = re.compile(r"(?:function\s+)?[A-Za-z_]\w*")
_FUNCTION_DEF_RE = re.compile(
    r"(?:function\s+[A-Za-z_]\w*\s*(?:\(\s*\))?|[A-Za-z_]\w*\s*\(\s*\))\s*"
)


def _unwrap_shell_group(command: str) -> tuple[str, bool]:
    """`( pip install x )` -> `("pip install x", False)`, `then pip install x` -> `(..., True)`.

    A grouped or compound command still runs, so leaving the bracket or keyword on hides it from
    PIP_LINE_RE. The flag says the keyword made it conditional, which only a body word does: `if
    pip install ...` is the test and is reached whenever the line is."""
    stripped = command.strip()
    bang = stripped.startswith("!")
    # `! false` negates while `!false` names a command.
    spaced = bang and stripped[1:2].isspace()
    if bang:
        stripped = stripped[1:].lstrip()
    # Keep the `)` that closes a `$( )`; `{` opens a group only as its own token ({sys.executable}
    # is one word). Strip a function header so the body is read.
    definition = _FUNCTION_DEF_RE.match(stripped)
    if definition is not None:
        stripped = stripped[definition.end() :].lstrip()

    def _open_groups(text: str) -> str:
        while text:
            if text[0] == "(":
                text = text[1:].lstrip()
            elif text[0] == "{" and (len(text) == 1 or text[1].isspace()):
                text = text[1:].lstrip()
            else:
                break
        return text.strip()

    stripped = _open_groups(stripped)
    while stripped[-1:] in (")", "}") and not _final_bracket_closes_substitution(stripped):
        stripped = stripped[:-1].rstrip()
    conditional = definition is not None
    while True:
        # Any whitespace: `then\tpip` is the same command to the shell.
        parts = stripped.split(maxsplit = 1)
        if not parts or parts[0].lower() not in _SHELL_KEYWORDS:
            break
        conditional = conditional or parts[0].lower() in _SHELL_BODY_KEYWORDS
        # A keyword can precede a group: `if (pip install ...); then`.
        stripped = _open_groups(parts[1].strip()) if len(parts) > 1 else ""
        # Or a definition: `then f(){ pip install ...; }`.
        behind = _FUNCTION_DEF_RE.match(stripped)
        if behind is not None:
            definition = behind
            conditional = True
            stripped = _open_groups(stripped[behind.end() :].lstrip())
    # A case arm label: only the matching arm runs, so the command is conditional.
    close = _unquoted_arm_close(stripped)
    if close is not None:
        stripped = stripped[close + 1 :].strip()
        conditional = True
    stripped, _prefixed = _strip_exec_prefixes(stripped)
    if not (bang and stripped):
        return stripped, conditional
    return (f"! {stripped}" if spaced else f"!{stripped}"), conditional


# `${name:-word}` expands word only on one branch, so substitutions in it are conditional.
_CONDITIONAL_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_]\w*(?:\[[^\]]*\])?:?[-+=?]")


def _conditional_expansion_spans(command: str) -> list[tuple[int, int]]:
    """Half-open ranges covering the word of every branching `${ }` expansion."""
    spans: list[tuple[int, int]] = []
    for match in _CONDITIONAL_EXPANSION_RE.finditer(command):
        depth, j = 1, match.end()
        quote = ""
        while j < len(command) and depth:
            ch = command[j]
            # `\}` and `'}'` are literal text in the default word, not the closer.
            if ch == "\\" and quote != "'" and j + 1 < len(command):
                j += 2
                continue
            if quote:
                if ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        spans.append((match.end(), j))
    return spans


def _substitution_bodies(command: str, conditional: bool = False) -> list[str]:
    """The insides of every substitution in `command`, in the order the shell runs them.

    `$( )`, backticks and `<( )` / `>( )` all run when the command runs, so a pip call in one is an
    install like any other; single quotes and an escaped `$` make the text literal. `conditional`
    selects only the bodies bash may SKIP, inside a branching `${name:-word}`, so the two calls
    together cover every body exactly once."""
    spans = _conditional_expansion_spans(command)

    def in_branch(index: int) -> bool:
        return any(start <= index < end for start, end in spans)

    bodies: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if ch == "\\" and quote != "'":
            i += 2
            continue
        if quote == "'":
            if ch == "'":
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = "" if ch == quote else (quote or ch)
            i += 1
            continue
        # `$( )` and backticks expand inside double quotes; `<( )` and `>( )` do not.
        opens = command.startswith("$(", i) or (
            not quote and ch in "<>" and command[i + 1 : i + 2] == "("
        )
        if opens:
            depth, j = 1, i + 2
            inner_quote = ""
            # Inside an open case the `)` delimits the arm rather than closing the substitution.
            case_depth = 0
            while j < len(command) and depth:
                inner = command[j]
                if inner == "\\" and inner_quote != "'":
                    j += 2
                    continue
                if inner_quote:
                    if inner == inner_quote:
                        inner_quote = ""
                elif inner in "\"'":
                    inner_quote = inner
                elif inner == "(":
                    depth += 1
                elif inner == ")":
                    if not (case_depth and depth == 1):
                        depth -= 1
                elif inner.isalpha() and not command[j - 1 : j].isalnum():
                    word = _LEADING_WORD_RE.match(command, j)
                    if word is not None:
                        if word.group(0) == "case":
                            case_depth += 1
                        elif word.group(0) == "esac" and case_depth:
                            case_depth -= 1
                j += 1
            if in_branch(i) == conditional:
                bodies.append(command[i + 2 : j - 1 if depth == 0 else j])
            i = j
        elif ch == "`":
            # The first unescaped backtick closes it; nested ones are written escaped.
            j = i + 1
            while j < len(command):
                if command[j] == "\\":
                    j += 2
                    continue
                if command[j] == "`":
                    break
                j += 1
            if j >= len(command):
                break
            # Unescape one level, as the shell does inside backticks.
            if in_branch(i) == conditional:
                bodies.append(command[i + 1 : j].replace("\\`", "`").replace("\\\\", "\\"))
            i = j + 1
        else:
            i += 1
    return [body.strip() for body in bodies if body.strip()]


def _piece_is_pip(piece: str) -> bool:
    """Is this chunk of a chained line a pip command? `!` only ever leads the first piece,
    and the splitter re-adds it to the rest, so it is normalised before asking."""
    # Strip the bang first so `!env X=1 pip ...` matches the env prefix.
    stripped = _strip_exec_prefixes(piece.strip().lstrip("!").strip())[0].strip()
    return bool(stripped) and bool(PIP_LINE_RE.match("!" + stripped))


_REDIRECTION_RE = re.compile(r"^\d*(?:>>|>&|&>|<<<|<<|<>|>|<)")
_EXEC_LONE_FLAGS = frozenset({"-c", "-l"})


def _command_execs(command: str) -> bool:
    """Does this command hand the shell over to `exec`?

    `exec NAME` replaces the shell, so no later command in the list can run; treating it as an
    ordinary prefix replayed both installs in `exec pip install a; pip install b`. With NO utility,
    `exec >/tmp/log` only makes the redirections permanent and hands nothing over."""
    seen: list[str] = []
    rest = _strip_exec_prefixes(command.lstrip("!").strip(), seen)[0]
    if "exec" not in seen:
        return False
    # `env exec` looks for a program called exec, so only in-process prefixes keep the builtin.
    if any(name not in _SHELL_RESOLVED_PREFIXES for name in seen[: seen.index("exec")]):
        return False
    while rest:
        word, tail = _split_first_word(rest)
        if not word:
            break
        if word in _EXEC_LONE_FLAGS:
            rest = tail
            continue
        if word == "-a":
            rest = _split_first_word(tail)[1]
            continue
        if _REDIRECTION_RE.match(word):
            rest = _split_first_word(tail)[1] if _REDIRECTION_RE.fullmatch(word) else tail
            continue
        return True
    return False


def _command_ends_shell(command: str) -> bool:
    """Does this command end the shell, so that nothing after it in the list can run?

    `exec NAME` replaces it and `exit` terminates it; recognising only the first reported `exit 0;
    pip install git+...` as a reachable install."""
    # `{ exit; }` runs in the same shell and ends the line; `( exit )` is a subshell and does not.
    opened = command.lstrip("!").strip()
    while True:
        if opened.startswith("{") and (len(opened) == 1 or opened[1].isspace()):
            opened = opened[1:].lstrip()
            continue
        word, rest = _split_first_word(opened)
        if word.lower() in _SHELL_BODY_KEYWORDS:
            opened = rest.lstrip()
            continue
        break
    if _command_execs(opened):
        return True
    seen: list[str] = []
    rest = _strip_exec_prefixes(opened, seen)[0]
    # Same rule as exec: `env exit 0` looks for a program called exit.
    if any(name not in _SHELL_RESOLVED_PREFIXES for name in seen):
        return False
    return _split_first_word(rest)[0] == "exit"


# `true` and `:` always succeed, so an `&&` after one is always reached.
_ALWAYS_SUCCEEDS = frozenset({"true", ":"})


def _piece_always_succeeds(piece: str) -> bool:
    """Is this piece a command whose exit status is documented as always zero?"""
    return _piece_success_model(piece) is True


def _piece_success_model(
    piece: str,
    functions: "dict[str, bool | None] | None" = None,
    notebook_bang: bool = True,
) -> bool | None:
    """True when the piece certainly succeeds, False when it certainly fails, else None.

    `!` inverts the pipeline's status, so `! false` succeeds and the `&&` behind it always runs,
    while `! pip install x` fails under the replay's model of pip succeeding."""
    text = _unwrap_shell_group(piece)[0]
    # Only the first command carries the notebook bang; elsewhere `!` is bash negation.
    if notebook_bang and text.startswith("!"):
        text = text[1:].lstrip()
    text, negations = _strip_negations(text)
    stripped = _strip_exec_prefixes(text)[0].strip()
    word = _split_first_word(stripped)[0] if stripped else ""
    if word in _ALWAYS_SUCCEEDS or _piece_is_pip(stripped):
        model: bool | None = True
    elif word == "false":
        model = False
    elif word in ("return", "exit") and _split_first_word(stripped)[1].strip().isdigit():
        # `return 0` makes the call succeed; a bare return carries an unknown previous status.
        model = _split_first_word(stripped)[1].strip() == "0"
    elif functions is not None and word in functions:
        # A call exits with its body's status.
        model = functions[word]
    else:
        model = None
    if model is None or not negations % 2:
        return model
    return not model


def _strip_negations(text: str) -> tuple[str, int]:
    """The command behind bash's `!` reserved words, and how many there were."""
    negations = 0
    while True:
        word, rest = _split_first_word(text)
        if word != "!":
            break
        negations += 1
        text = rest.strip()
    return text, negations


def _pipeline_negations(piece: str, notebook_bang: bool = True) -> int:
    """How many `!` lead this pipeline. Bash negates the WHOLE pipeline's status.

    `! true | false` succeeds: the pipeline exits with `false` and the `!` turns that around.
    Reading the negation as the first command's alone discarded it at the pipe."""
    text = _unwrap_shell_group(piece)[0]
    if notebook_bang and text.startswith("!"):
        text = text[1:].lstrip()
    return _strip_negations(text)[1]


def _piece_assumes_pip(piece: str) -> bool:
    """Is this piece's success the REPLAY's assumption about pip, not a documented outcome?

    `true` cannot fail; a pip install can. Modelling both as certain success removed the reachable
    `else` of `if pip install x; then :; else pip install git+...; fi`."""
    text = _unwrap_shell_group(piece)[0]
    if text.startswith("!"):
        text = text[1:].lstrip()
    while True:
        word, rest = _split_first_word(text)
        if word != "!":
            break
        text = rest.strip()
    stripped = _strip_exec_prefixes(text)[0].strip()
    return _piece_is_pip(stripped) and _split_first_word(stripped)[0] not in _ALWAYS_SUCCEEDS


def _close_group(
    assured: list[bool],
    prev_ops: list[str],
    last_ok: list[bool | None],
    models: list[bool | None],
    pending: str,
    notebook_bang: bool = True,
) -> None:
    """Fold a closing group's success into the list that contains it.

    A group exits with its LAST command's status, so `(pip install x) && pip install y` reaches y
    as the ungrouped form does; discarding the inner state marked y conditional. The pending text
    is the command still in hand, which no separator has flushed."""
    if _unwrap_shell_group(pending)[0].strip():
        # Fold through the group's own list: `(false && pip install x)` short-circuits.
        last_ok[-1] = _left_hand_status(models, prev_ops, pending, None, notebook_bang)
    inner_model = last_ok.pop()
    inner = inner_model is True
    assured.pop()
    models.pop()
    prev_ops.pop()
    if prev_ops[-1] == "&&":
        assured[-1] = assured[-1] and inner
    else:
        assured[-1] = assured[-1] or inner
    # Three-valued: `{ false; } || pip install x` always reaches the install.
    models[-1] = (
        inner_model if prev_ops[-1] == "" else _fold_status(models[-1], prev_ops[-1], inner_model)
    )


def _fold_pending(
    assured: list[bool],
    prev_ops: list[str],
    pending: str,
    notebook_bang: bool = True,
    spoken_for: bool = False,
    negations: int = 0,
) -> None:
    """Fold the command in hand into the list, unless a group already spoke for it.

    After `(pip install x)` closes, the level ALREADY carries the group's status and the text in
    hand is the bare bracket, which folded as an unknown command. `spoken_for` is the same case
    with the brackets still around a body: `_close_group` has folded `(false && true)` as the
    failure it is, and reprocessing read its last lexical `true`."""
    if spoken_for or not _unwrap_shell_group(pending)[0].strip():
        return
    # `!` negates the whole pipeline's status.
    piece = _negated(_piece_success_model(pending, None, notebook_bang), negations)
    _fold_and_or(assured, prev_ops, piece is True)


def _negated(status: bool | None, negations: int) -> bool | None:
    """Turn a status around once per `!`. An unknown one stays unknown."""
    if status is None or not negations % 2:
        return status
    return not status


def _fold_and_or(assured: list[bool], prev_ops: list[str], piece_is_pip: bool) -> None:
    """Fold the piece just read into "is this and-or list assumed to have succeeded?".

    `A || B` succeeds when EITHER side did, so a pip install on the left carries the list. `A && B`
    needs both, so an intervening command not modelled as succeeding breaks the chain: a failing
    probe in `pip install torch && probe && pip install torchcodec` leaves the last install
    unreachable."""
    if prev_ops[-1] == "||":
        assured[-1] = assured[-1] or piece_is_pip
    elif prev_ops[-1] == "&&":
        assured[-1] = assured[-1] and piece_is_pip
    else:
        assured[-1] = piece_is_pip


def _fold_status(left: bool | None, op: str, right: bool | None) -> bool | None:
    """The exit status of `left OP right`, or None when it cannot be known.

    Left-associative: the left operand of `a || b && c` is the whole `(a || b)` list, and its
    folded status, not the piece nearest the operator, decides whether `c` runs."""
    if op == "&&":
        if left is None:
            return False if right is False else None
        return right if left else False
    if left is None:
        return True if right is True else None
    return True if left else right


def _left_hand_status(
    models: list[bool | None],
    prev_ops: list[str],
    pending: str,
    functions: "dict[str, bool | None] | None" = None,
    notebook_bang: bool = True,
    spoken_for: bool = False,
) -> bool | None:
    """Fold the piece in hand into its level's running status and return the result.

    Called at each `&&`/`||` so the operator sees the status of everything to its left, not just
    the piece beside it."""
    if spoken_for or not _unwrap_shell_group(pending)[0].strip():
        # A group just closed; the level already carries its status.
        return models[-1]
    piece = _piece_success_model(pending, functions, notebook_bang)
    models[-1] = piece if prev_ops[-1] == "" else _fold_status(models[-1], prev_ops[-1], piece)
    return models[-1]


def _function_name(header: str) -> str:
    """`setup() {` / `function setup {` -> `setup`."""
    words = header.replace("(", " ").replace(")", " ").split()
    return words[1] if words[:1] == ["function"] else words[0]


def _for_list_is_nonempty(text: str) -> bool:
    """Does `for NAME in WORDS` iterate at least once, readably?

    Only a LITERAL list answers: `$LIST` and a glob may both expand to nothing, while `for x in a
    b` runs, so its body is reached as surely as a bare command."""
    # Shell whitespace: `for x\tin\ta` is the same loop.
    match = re.search(r"\sin\s", text)
    if match is None:
        return False
    words = text[match.end() :].split()
    if not words or not any(words):
        return False
    # Quoting decides meaning: '*' is one literal, a bare * is a glob that may match nothing.
    return not any(_word_may_vanish(word) for word in words)


def _word_may_vanish(word: str) -> bool:
    """Could this loop word expand to something other than itself, nothing included?

    Single quotes make every character literal; double quotes still expand `$` and a backquote but
    never a glob. Only what is left unquoted can be a glob."""
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in "\"'":
            close = word.find(ch, i + 1)
            if close == -1:
                close = len(word)
            segment = word[i + 1 : close]
            if ch == '"' and any(c in segment for c in ("$", "`")):
                return True
            i = close + 1
            continue
        if ch in "$`*?[":
            return True
        i += 1
    return False


def _invoked_name(piece: str) -> str:
    """The word this piece runs, read WITHOUT stripping execution prefixes.

    `env f`, `nohup f` and `command f` look for an executable named f, so none reaches a shell
    function and stripping them made every wrapper look like a call. Brackets, a function header
    and the body keywords do come off: they precede the command rather than replace it."""
    text = piece.lstrip("!").strip()
    while text[:1] in ("(", "{"):
        text = text[1:].lstrip()
    header = _FUNCTION_DEF_RE.match(text)
    if header is not None:
        text = text[header.end() :].lstrip().lstrip("({").lstrip()
    while True:
        word, rest = _split_first_word(text)
        # Assignments, redirections and reserved `time` still call the function; exec wrappers do not.
        if (
            word.lower() in _SHELL_BODY_KEYWORDS
            or word == "time"
            or _ENV_ASSIGNMENT_RE.match(word)
            or _REDIRECTION_RE.match(word)
        ):
            text = rest.lstrip()
            continue
        return word


def _behind_keywords(text: str) -> str:
    """`then f(` -> `f(`. The words a compound statement opens with are not the command.

    A definition can sit behind them, and `then f` matched no header, so the body was tracked as a
    plain group."""
    text = text.lstrip("!").strip().lstrip("({").lstrip()
    while True:
        parts = text.split(maxsplit = 1)
        if not parts or parts[0].lower() not in _SHELL_KEYWORDS:
            return text
        text = parts[1].strip() if len(parts) > 1 else ""


def _leading_shell_keywords(piece: str) -> list[str]:
    """The compound-statement words this piece opens with, in order.

    `_unwrap_shell_group` strips them, so their state is read off the RAW piece first, or a body
    spanning a separator keeps only its first command."""
    text = piece.strip()
    if text.startswith("!"):
        text = text[1:].lstrip()
    text = text.lstrip("({").lstrip()
    # A compound may open inside a definition header line.
    definition = _FUNCTION_DEF_RE.match(text)
    if definition is not None:
        text = text[definition.end() :].lstrip().lstrip("({").lstrip()
    words: list[str] = []
    while True:
        parts = text.split(maxsplit = 1)
        if not parts or parts[0].lower() not in _SHELL_KEYWORDS:
            return words
        words.append(parts[0].lower())
        text = parts[1].strip() if len(parts) > 1 else ""


def _split_chained(line: str) -> list[tuple[str, bool]]:
    """One shell line -> `(command, conditional)` per command. Only the first keeps the `!`.

    `pip uninstall -y x && pip install x==1` is two commands with two actions; read as one, the
    reinstall lands in the uninstall's package list. Scanned rather than split on a pattern, since
    a PEP 508 marker puts a quoted `;` inside one argument and a backslash escapes the next
    character outside single quotes.

    A `||` fallback is flagged conditional rather than dropped: it can still run, and the rules
    that must see every install path have to keep seeing it. The tail ends at an `&&` or a `;`, the
    lists being left-associative, and each group keeps its own, so a command is conditional when
    any level above it is in one. A single `&` or `|` runs both sides and opens no tail, while `>&`
    and `&>` are redirections. An unquoted `#` starting a word ends the scan."""
    out: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote = ""
    # One flag per open group; an inner list cannot clear an outer fallback tail.
    tails = [False]
    # Per open group: does it hold a function body? Survives separators inside the body.
    def_levels = [False]
    # A body is conditional until its function is called later in the line, so keep ownership.
    def_names: list[str | None] = [None]
    owners: list[str | None] = []
    nodef: list[bool] = []
    assumed: list[bool] = []
    # Bash requires definition before call, so one left-to-right pass has the status in hand.
    func_status: dict[str, bool | None] = {}
    instances: dict[str, list[str]] = {}
    definitions = 0
    # Per level: is the last flushed command modelled as succeeding (the group's exit status).
    last_ok: list[bool | None] = [None]
    # Per level: has this and-or list already run pip (`A && B` is unconditional only then).
    list_has_pip = [False]
    # Per level: the operator joining the piece in hand to the list before it.
    prev_ops = [""]
    # Per level: folded status of the list to the left; three-valued, only certain failure counts.
    list_models: list[bool | None] = [None]
    buf_conditional = False
    # True when a `(`/`{` opened a grouping; a `$( )` close is inside a word.
    groupings: list[bool] = []
    # Per level: open case statements, whose arm patterns end in an unbalanced `)`.
    case_depths: list[int] = [0]
    grouping_closed = False
    # A group closed with nothing flushed since: do not fold the text in hand again.
    closed_pending = False
    # A leading `!` belongs to the whole pipeline, so it outlives the first pipe.
    pipe_negations = 0
    in_pipeline = False
    func_parens = False
    # Per level: tail made unconditional only by the pip-succeeds assumption (report, never cut).
    assumed_tail = [False]
    # Operators inside a legacy backtick substitution belong to the inner command.
    in_backtick = False
    i = 0

    def in_sub() -> bool:
        return in_backtick or not all(groupings)

    # `exec` under `|` or `&` runs in a subshell, so the list continues.
    seps: list[str] = []

    def flush(separator: str = "") -> None:
        nonlocal buf, closed_pending
        text = "".join(buf)
        last_ok[-1] = _piece_success_model(text, func_status, not out)
        out.append((text, buf_conditional))
        seps.append(separator)
        owners.append(next((name for name in reversed(def_names) if name), None))
        # Without the definition: calling a function makes its body reachable, not unguarded.
        nodef.append(any(tails))
        assumed.append(any(assumed_tail))
        buf = []
        closed_pending = False

    while i < len(line):
        ch = line[i]
        in_substitution = in_sub()
        if not quote and ch.isalpha() and not (buf and buf[-1].isalnum()):
            # Tracked before dispatch: the substitution branch swallows characters whole.
            keyword = _LEADING_WORD_RE.match(line, i)
            if keyword is not None:
                if keyword.group(0) == "case":
                    case_depths[-1] += 1
                elif keyword.group(0) == "esac" and case_depths[-1]:
                    case_depths[-1] -= 1
        if ch == "\\" and quote != "'" and i + 1 < len(line):
            buf.append(ch)
            buf.append(line[i + 1])
            i += 2
        elif ch == "`" and quote != "'":
            # Backticks expand inside double quotes too.
            in_backtick = not in_backtick
            buf.append(ch)
            i += 1
        elif quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
        elif ch in ")}" and in_substitution and not (ch == ")" and case_depths[-1]):
            grouping_closed = groupings.pop() if groupings else True
            if len(case_depths) > 1:
                case_depths.pop()
            if len(tails) > 1:
                tails.pop()
                if len(assumed_tail) > 1:
                    assumed_tail.pop()
                if len(def_levels) > 1:
                    def_levels.pop()
                    closing = def_names.pop()
                    if closing:
                        # Record the group's status before the pop loses it.
                        func_status[closing.split("#")[0]] = last_ok[-1]
                _close_group(list_has_pip, prev_ops, last_ok, list_models, "".join(buf), not out)
                closed_pending = True
            buf.append(ch)
            i += 1
        elif ch == "#" and (
            i == 0
            or line[i - 1].isspace()
            or line[i - 1] in ";&|"
            or (line[i - 1] in ")}" and grouping_closed)
        ):
            break
        elif in_substitution:
            buf.append(ch)
            i += 1
        elif line.startswith("||", i):
            # Only an unknown left side opens a `||` tail, and the left side is the whole list.
            left_model = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            left_model = _negated(left_model, pipe_negations)
            if pipe_negations % 2:
                list_models[-1] = left_model
            _fold_pending(
                list_has_pip, prev_ops, "".join(buf), not out, closed_pending, pipe_negations
            )
            pipe_negations, in_pipeline = 0, False
            prev_ops[-1] = "||"
            flush("||")
            tails[-1] = left_model is not False
            buf_conditional = any(tails) or any(def_levels)
            i += 2
        elif line.startswith("&&", i):
            # `A && B` runs B only if A succeeded; B is unconditional only if a pip command is to its left
            # (modelled as succeeding), so `nvidia-smi && pip install ...` stays conditional.
            left_and = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            left_and = _negated(left_and, pipe_negations)
            if pipe_negations % 2:
                list_models[-1] = left_and
            _fold_pending(
                list_has_pip, prev_ops, "".join(buf), not out, closed_pending, pipe_negations
            )
            pipe_negations, in_pipeline = 0, False
            prev_ops[-1] = "&&"
            # Relies on pip-succeeds: fine for reporting an install, never for making anything unreachable.
            assumed_tail[-1] = assumed_tail[-1] or _piece_assumes_pip("".join(buf))
            flush("&&")
            # A certain-success left side reaches the tail too (`true && ...`).
            tails[-1] = not (list_has_pip[-1] or left_and is True)
            buf_conditional = any(tails) or any(def_levels)
            i += 2
        elif (
            ch == ";"
            or (
                ch in "&|"
                # `>&`, `<&`, `&>` and `>|` are redirections, not separators.
                and not (
                    ch == "&" and (line[i - 1 : i] in ("<", ">") or line[i + 1 : i + 2] == ">")
                )
                and not (ch == "|" and line[i - 1 : i] == ">")
            )
        ):
            # A group exits with its list's folded status, not its last lexical command's.
            folded = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            if ch == "|":
                if not in_pipeline:
                    # `a | ! b` is a syntax error, so only the head carries a negation.
                    pipe_negations = _pipeline_negations("".join(buf), not out)
                    in_pipeline = True
            else:
                folded = _negated(folded, pipe_negations)
                list_models[-1] = folded
                pipe_negations, in_pipeline = 0, False
            flush(ch if ch in "&|" else ";")
            last_ok[-1] = folded
            tails[-1] = False
            list_has_pip[-1] = False
            list_models[-1] = None
            assumed_tail[-1] = False
            prev_ops[-1] = ""
            buf_conditional = any(tails) or any(def_levels)
            i += 1
        else:
            # `f()` is a function header, not a group.
            if (
                ch == "("
                and _FUNCTION_NAME_RE.fullmatch(_behind_keywords("".join(buf)))
                and line[i + 1 :].lstrip().startswith(")")
            ):
                func_parens = True
            if ch == ")" and func_parens:
                func_parens = False
            elif ch in "({":
                # `$(`, `<(`, `>(` run commands; `${ }` expands a word and runs nothing.
                groupings.append(
                    not (
                        (ch == "(" and buf and buf[-1] in "$<>")
                        or (ch == "{" and buf and buf[-1] == "$")
                    )
                )
                # An uncalled function body stays conditional through every later command in it.
                tails.append(False)
                assumed_tail.append(False)
                header = _FUNCTION_DEF_RE.fullmatch(_behind_keywords("".join(buf)))
                def_levels.append(ch == "{" and header is not None)
                if ch == "{" and header:
                    # One key per definition: `f(){ a; }; f; f(){ b; }` calls the first body.
                    name = _function_name(header.group(0))
                    definitions += 1
                    key = f"{name}#{definitions}"
                    instances.setdefault(name, []).append(key)
                    def_names.append(key)
                else:
                    def_names.append(None)
                list_has_pip.append(False)
                list_models.append(None)
                prev_ops.append("")
                last_ok.append(None)
                case_depths.append(0)
                if not "".join(buf).strip():
                    buf_conditional = any(tails) or any(def_levels)
            elif ch in ")}" and not (ch == ")" and case_depths[-1]):
                grouping_closed = groupings.pop() if groupings else True
                if len(case_depths) > 1:
                    case_depths.pop()
                if len(tails) > 1:
                    # The command in hand belongs to the closing level; the pop affects only what follows.
                    tails.pop()
                    if len(assumed_tail) > 1:
                        assumed_tail.pop()
                    if len(def_levels) > 1:
                        def_levels.pop()
                        closing = def_names.pop()
                        if closing:
                            func_status[closing.split("#")[0]] = last_ok[-1]
                    _close_group(
                        list_has_pip, prev_ops, last_ok, list_models, "".join(buf), not out
                    )
                    closed_pending = True
            if ch not in ")}":
                grouping_closed = False
            buf.append(ch)
            i += 1
    flush()
    (head, head_conditional), *rest = out
    head_text, head_keyword = _unwrap_shell_group(head)
    # Keep empty pieces until after the zip, or later pairs slide by one.
    commands = [(head_text, head_conditional or head_keyword)]
    # The keyword flag alone; the piece's own flag would double-count a called definition.
    kw_flags = [head_keyword]
    for piece, flag in rest:
        text, keyword = _unwrap_shell_group(piece.strip())
        # Keep the space so `! false` is not read as a command `!false`.
        commands.append(
            (f"!{' ' if text.startswith('!') else ''}{text}" if text else "", flag or keyword)
        )
        kw_flags.append(keyword)
    # `echo $(pip install x)` runs the install, so inner substitutions are commands too.
    ordered: list[tuple[str, bool]] = []
    # An unconditional exec or exit makes every later outer command unreachable.
    handed_over = False
    seps = seps + [""] * (len(out) - len(seps))
    # One flag per open compound: True once its body has started.
    body_levels: list[bool] = []
    # Per open compound: its test's modelled result; a never-true test makes the body unreachable.
    test_models: list[bool | None] = []
    # Per open compound: did every arm so far certainly fail, and was each known (for elif/else).
    arms_failed: list[bool] = []
    arms_known: list[bool] = []
    # Per open compound: opener word, whether the arm is reached, and whether pip-success was assumed.
    openers: list[str] = []
    arm_reached: list[bool] = []
    cond_assumed: list[bool] = []
    # The condition folded with pip's status unknown, to tell whether the assumption decided it.
    cond_models: list[bool | None] = []
    body_entries: dict[str, list[tuple[int, bool]]] = {}
    # Per body, names it invokes unconditionally; reached only once the body is, so transitive.
    body_invokes: dict[str, set[tuple[str, int]]] = {}
    called: set[tuple[str, int]] = set()
    maybe_called: set[tuple[str, int]] = set()
    # Open-compound depth at an unconditional break/continue; loop-local, unlike exit.
    broke_at: int | None = None
    # `return` ends the body, not the shell; a call resolves against the definition in force then.
    returned: set[str] = set()
    def_last_index: dict[str, int] = {}
    # Functions whose body ends the shell; applied when the call is resolved.
    ends_shell: set[str] = set()
    call_at: dict[str, int] = {}
    for index, ((piece, flag), (text, command_flag), separator) in enumerate(
        zip(out, commands, seps)
    ):
        if handed_over:
            break
        keywords = _leading_shell_keywords(piece)
        # A case selector runs before any arm, so a substitution in it is unconditional.
        opens_case = "case" in keywords
        for keyword in keywords:
            if keyword == "case":
                # Everything until esac sits in some arm, so the level is active from the word itself.
                body_levels.append(True)
                test_models.append(None)
                arms_failed.append(False)
                arms_known.append(False)
                openers.append("case")
                arm_reached.append(False)
                cond_assumed.append(False)
                cond_models.append(None)
            elif keyword in _SHELL_TEST_KEYWORDS:
                body_levels.append(False)
                test_models.append(None)
                arms_failed.append(True)
                arms_known.append(True)
                openers.append(keyword)
                arm_reached.append(True)
                cond_assumed.append(False)
                cond_models.append(None)
            elif keyword in _SHELL_BODY_KEYWORDS:
                if body_levels:
                    body_levels[-1] = True
                    if keyword in ("then", "do"):
                        # Condition complete: fold it, invert an until, and record the result.
                        model = test_models[-1]
                        if openers[-1] == "until" and model is not None:
                            model = not model
                        if not arm_reached[-1]:
                            model = None
                        elif model is False and cond_assumed[-1]:
                            # False only via the pip-succeeds assumption; `if ! pip install x` can run its body.
                            model = None
                        if openers[-1] in ("if", "until", "while"):
                            arms_known[-1] = (
                                arms_known[-1] and model is not None and not cond_assumed[-1]
                            )
                            arms_failed[-1] = arms_failed[-1] and model is False
                        test_models[-1] = model
                        cond_models[-1] = model
                    elif keyword == "else":
                        # `else` runs exactly when every arm failed.
                        test_models[-1] = (
                            True if arms_failed[-1] else (False if arms_known[-1] else None)
                        )
                        cond_models[-1] = test_models[-1]
                    elif keyword == "elif":
                        # An elif test is reached only when every earlier arm failed.
                        arm_reached[-1] = arms_failed[-1]
                        body_levels[-1] = False
                        openers[-1] = "if"
                        cond_assumed[-1] = False
                        test_models[-1] = True if arms_failed[-1] else None
                        cond_models[-1] = test_models[-1]
                else:
                    # Keep stacks in step, or the matching fi pops an unpushed level (IndexError).
                    body_levels.append(True)
                    test_models.append(None)
                    arms_failed.append(False)
                    arms_known.append(False)
                    openers.append("")
                    arm_reached.append(False)
                    cond_assumed.append(False)
                    cond_models.append(None)
            elif body_levels:
                body_levels.pop()
                test_models.pop()
                arms_failed.pop()
                arms_known.pop()
                openers.pop()
                arm_reached.pop()
                cond_assumed.pop()
                cond_models.pop()
        # A level speaks only once its body has started: `if true` runs, `if false` never does,
        # anything else may run. A false branch is inverted for else.
        active = [model for level, model in zip(body_levels, test_models) if level]
        if any(model is False for model in active):
            continue
        if broke_at is not None:
            if len(body_levels) < broke_at:
                broke_at = None
            else:
                continue
        # A then/else/arm flag means conditional only if the branch is not known to be taken.
        # An elif after certain failures is a test, which runs whenever the statement does.
        reached_test = bool(body_levels) and not body_levels[-1] and arm_reached[-1]
        certain_branch = reached_test or (bool(active) and all(model is True for model in active))
        piece_conditional = (
            flag
            or (command_flag and not certain_branch)
            or any(model is not True for model in active)
        )
        # The case selector shares a piece with the first arm.
        selector = (
            zip(body_levels[:-1], test_models[:-1]) if opens_case else zip(body_levels, test_models)
        )
        sub_conditional = (
            flag or command_flag or any(model is not True for level, model in selector if level)
        )
        for inner in _substitution_bodies(piece):
            for inner_text, inner_flag in _split_chained(f"!{inner}"):
                # A substitution inherits the parent's functions; record the call for the reachability walk.
                if sub_conditional or inner_flag:
                    maybe_called.add((_invoked_name(inner_text), index))
                else:
                    called.add((_invoked_name(inner_text), index))
                ordered.append((inner_text, sub_conditional or inner_flag))
        # `${READY:-$(pip install ...)}` may run, never certainly.
        for inner in _substitution_bodies(piece, conditional = True):
            for inner_text, _ in _split_chained(f"!{inner}"):
                maybe_called.add((_invoked_name(inner_text), index))
                ordered.append((inner_text, True))
        if text:
            # A never-true test makes the body unreachable, not conditional. Fold condition pieces until
            # then/do closes it.
            if body_levels and not body_levels[-1]:
                model = (
                    _for_list_is_nonempty(text) or None
                    if openers[-1] == "for"
                    else _piece_success_model(text)
                )
                opens_here = bool(keywords) and keywords[0] in _SHELL_TEST_KEYWORDS | {"elif"}
                joiner = seps[index - 1] if index and not opens_here else ""
                test_models[-1] = (
                    model
                    if joiner not in ("&&", "||")
                    else _fold_status(test_models[-1], joiner, model)
                )
                # Only while the pip-success assumption still decides the condition.
                unassumed = None if _piece_assumes_pip(text) else model
                cond_models[-1] = (
                    unassumed
                    if joiner not in ("&&", "||")
                    else _fold_status(cond_models[-1], joiner, unassumed)
                )
                cond_assumed[-1] = cond_models[-1] is not test_models[-1]
            # Only the raw first word calls a function; env/nohup/command look for an executable.
            invoked = _invoked_name(piece)
            # The flag with the definition entered; on a header piece command_flag is the definition.
            header_match = _FUNCTION_DEF_RE.match(piece.lstrip("!").strip())
            header_piece = header_match is not None
            header_span = (
                len(piece) - len(piece.lstrip("!").strip()) + header_match.end()
                if header_match
                else 0
            )
            entered = bool(
                nodef[index]
                or (kw_flags[index] and not certain_branch and not header_piece)
                or any(model is not True for model in active)
            )
            owner = owners[index]
            if owner is not None:
                # Keyed by definition, not name: a call uses the definition in force at that point.
                def_last_index[owner] = index
                if owner in returned:
                    continue
                body_entries.setdefault(owner, []).append((len(ordered), entered))
                if not entered:
                    body_invokes.setdefault(owner, set()).add((invoked, index))
                    if invoked == "return":
                        returned.add(owner)
                    elif (
                        not assumed[index]
                        and separator not in ("|", "&")
                        and _command_ends_shell(
                            # Strip the header to see the terminator; raw body, since unwrapping strips exec.
                            piece[header_span:] if header_piece else piece
                        )
                    ):
                        ends_shell.add(owner)
            elif not piece_conditional:
                called.add((invoked, index))
                if separator not in ("|", "&"):
                    # `f | cat` runs f in a subshell, so its terminator does not end the parent.
                    call_at.setdefault(invoked, len(ordered))
            ordered.append((text, piece_conditional))
            # Use the raw piece: unwrapping strips exec. An unreached exec hands nothing over.
            handed_over = (
                not piece_conditional
                and not assumed[index]
                and separator not in ("|", "&")
                and _command_ends_shell(piece)
            )
            if (
                not piece_conditional
                and body_levels
                and _split_first_word(_strip_exec_prefixes(text.lstrip("!").strip())[0].strip())[0]
                in ("break", "continue")
            ):
                # break leaves the innermost loop, not the enclosing if; outside a loop bash ignores it.
                loop = [n for n, word in enumerate(openers) if word in ("while", "until", "for")]
                if loop:
                    # `break n` leaves n enclosing loops; a count past the nesting leaves all.
                    _, _, level = (
                        _strip_exec_prefixes(text.lstrip("!").strip())[0].strip().partition(" ")
                    )
                    depth = int(level.strip()) if level.strip().isdigit() else 1
                    broke_at = loop[max(len(loop) - depth, 0)] + 1

    # A body is conditional until called; walk the call graph to a fixed point. A call reaches
    # only a definition that already exists.
    def _definition_in_force(name: str, at: int) -> str | None:
        """The definition of `name` complete before position `at`, or None."""
        best = None
        for key in instances.get(name, ()):
            end = def_last_index.get(key)
            if end is not None and end < at and (best is None or end > def_last_index[best]):
                best = key
        return best

    reached: set[str] = set()
    pending_calls = [
        key for name, at in called if (key := _definition_in_force(name, at)) is not None
    ]
    reached.update(pending_calls)
    while pending_calls:
        for callee, at in body_invokes.get(pending_calls.pop(), ()):
            key = _definition_in_force(callee, at)
            if key is not None and key not in reached:
                reached.add(key)
                pending_calls.append(key)
    # A maybe-call (inside ${X:-$(f)}) reaches the body without making it certain.
    soft = {
        key
        for name, at in maybe_called
        if (key := _definition_in_force(name, at)) is not None and key not in reached
    }
    soft_pending = list(soft)
    reached |= soft
    while soft_pending:
        for callee, at in body_invokes.get(soft_pending.pop(), ()):
            key = _definition_in_force(callee, at)
            if key is not None and key not in reached:
                reached.add(key)
                soft.add(key)
                soft_pending.append(key)
    # An uncalled body is unreachable, not conditional.
    unreached: set[int] = set()
    for name, entries in body_entries.items():
        for position, entered in entries:
            if name in soft:
                ordered[position] = (ordered[position][0], True)
            elif name in reached:
                ordered[position] = (ordered[position][0], entered)
            else:
                unreached.add(position)
    cut = min(
        (
            call_at[key.split("#")[0]]
            for key in reached & ends_shell
            if key.split("#")[0] in call_at
        ),
        default = None,
    )
    if cut is not None:
        del ordered[cut + 1 :]
        unreached = {position for position in unreached if position <= cut}
    if unreached:
        return [entry for n, entry in enumerate(ordered) if n not in unreached]
    return ordered


def unconditional_pip_invocations(install_cell: str) -> Iterator[PipInvocation]:
    """The commands that certainly run.

    Anything asking what the cell leaves installed wants this one. `iter_pip_invocations` yields
    the `||` fallbacks too, for the rules that must see every path a notebook could take."""
    for inv in iter_pip_invocations(install_cell):
        if not inv.conditional:
            yield inv


def iter_pip_invocations(install_cell: str) -> Iterator[PipInvocation]:
    for line_no, line in _glue_line_continuations(install_cell):
        for command, conditional in _split_chained(line):
            inv = parse_pip_line(command, line_no)
            if inv is not None:
                inv.conditional = conditional
                yield inv


SPEC_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?(?P<rest>.*)$")
OP_VERSION_RE = re.compile(r"(==|>=|<=|!=|~=|>|<)\s*([0-9][^,;\s]*)")


@dataclasses.dataclass
class SpecParts:
    name: str
    pins: list[tuple[str, str]]
    raw: str


def parse_spec(spec: str) -> SpecParts | None:
    spec = spec.strip().strip('"').strip("'")
    if not spec or spec.startswith("-") or "://" in spec:
        return None
    m = SPEC_RE.match(spec)
    if not m:
        return None
    name = m.group("name").lower()
    rest = m.group("rest")
    pins = OP_VERSION_RE.findall(rest)
    return SpecParts(name = name, pins = pins, raw = spec)


def _canonical_project(name: str) -> str:
    """PEP 503 name normalization: any run of `-`, `_` or `.` is one `-`, lowercased.

    `huggingface.hub`, `huggingface_hub` and `huggingface-hub` are one project to pip, and the
    snapshot is keyed the last way, so folding only `_` judged a version the cell had removed."""
    return re.sub(r"[-_.]+", "-", name).lower()


def explicit_pin(spec: SpecParts) -> str | None:
    for op, ver in spec.pins:
        if op == "==":
            return ver
    return None


def pypi_metadata(name: str, version: str) -> dict[str, Any] | None:
    PYPI_CACHE_DIR.mkdir(parents = True, exist_ok = True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{name.lower()}__{version}")
    path = PYPI_CACHE_DIR / f"{safe}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout = 10) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    _atomic_write_bytes(path, json.dumps(data).encode("utf-8"))
    return data


def transitive_constraint(name: str, version: str, target: str) -> tuple[str | None, list[str]]:
    """Return (raw_specifier_string_or_None, list_of_(op,version) tuples) for the constraint that `name==version` places on `target`."""
    md = pypi_metadata(name, version)
    if not md:
        return None, []
    info = md.get("info", {}) or {}
    requires = info.get("requires_dist") or []
    target_l = target.lower()
    for req in requires:
        head = req.split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)\s*\(?([^)]*)?\)?\s*$", head)
        if not m:
            continue
        if m.group(1).lower() != target_l:
            continue
        spec = (m.group(2) or "").strip()
        return spec, OP_VERSION_RE.findall(spec)
    return None, []


def constraint_satisfied(version: str, ops: list[tuple[str, str]]) -> bool:
    if not ops:
        return True
    for op, v in ops:
        c = cmp_versions(version, v)
        if op == "==":
            if c != 0:
                return False
        elif op == ">=":
            if c < 0:
                return False
        elif op == "<=":
            if c > 0:
                return False
        elif op == ">":
            if c <= 0:
                return False
        elif op == "<":
            if c >= 0:
                return False
        elif op == "!=":
            if c == 0:
                return False
    return True


def resolved_set(install_cell: str, colab: dict[str, str]) -> dict[str, str]:
    """Merge install-cell constraints with Colab pip-freeze (cell wins). Resolution order per package: (1) exact `==V` pin, (2) upper-bound `<=V` (pip picks the highest allowed = V), (3) Colab fallback. Lower-bound `>=V` is intentionally NOT reflected (it does not lower an already-higher Colab version); R-INST-003 models that via `_install_cell_lower_bound`."""
    out = dict(colab)
    pinned: set[str] = set()
    upper_bounds: dict[str, str] = {}
    environment = _marker_environment(colab)
    for inv in unconditional_pip_invocations(install_cell):
        if _is_dry_run(inv):
            continue
        if inv.action == "uninstall":
            # Uninstall removes the package and its accumulated bound, or a reinstall inherits it.
            for raw in inv.packages:
                sp = parse_spec(raw)
                if sp is None:
                    continue
                # PEP 503: pop both the written and canonical spellings.
                for key in {sp.name, _canonical_project(sp.name)}:
                    out.pop(key, None)
                    pinned.discard(key)
                    upper_bounds.pop(key, None)
            continue
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or not _requirement_applies(raw, environment):
                continue
            for op, ver in sp.pins:
                if op == "==":
                    out[sp.name] = ver
                    pinned.add(sp.name)
                elif op == "<=" and sp.name not in pinned:
                    if sp.name not in upper_bounds or cmp_versions(ver, upper_bounds[sp.name]) < 0:
                        upper_bounds[sp.name] = ver
    for name, ub in upper_bounds.items():
        if name in pinned:
            continue
        existing = out.get(name)
        if existing is None or cmp_versions(existing, ub) > 0:
            out[name] = ub
    return out


# Case-insensitive: pip normalises `Git+https://`.
_GIT_SOURCE_RE = re.compile(r"""git\+[^\s'"]+""", re.IGNORECASE)


def _git_source_repository(source: str) -> str:
    """`git+https://user@github.com/Org/Repo.git@ref` -> `github.com/org/repo`.

    Matched as a path, not a substring: an arbitrary repository can carry
    `github.com/unslothai/unsloth` inside its own path, which a substring test reads as permission."""
    # Quotes and brackets are shell syntax, never part of a repository path.
    source = source.strip().rstrip(")}`\"'")
    remainder = source.split("+", 1)[1] if "+" in source else source
    remainder = remainder.split("://", 1)[-1]
    host, _, path = remainder.partition("/")
    host = host.rsplit("@", 1)[-1]
    path = path.split("#", 1)[0].split("?", 1)[0]
    # The last `@` is the revision delimiter, so `repo@fake/../../x@main` cannot pass.
    path = path.rsplit("@", 1)[0].rstrip("/")
    # Case-insensitive `.git` strip, since host and path are lowered below.
    if path.lower().endswith(".git"):
        path = path[: -len(".git")]
    # Resolve `.` and `..` as a URL client does, or traversal reads as an allowlisted prefix.
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    return "/".join([host.lower(), *(segment.lower() for segment in segments)])


def _git_source_is_allowed(source: str) -> bool:
    """Exact repository match. Every allowlist entry is one `host/org/repo`, and pip puts a
    subdirectory in the URL fragment rather than on the path, so nothing needs a prefix."""
    repository = _git_source_repository(source)
    return any(repository == allowed.lower() for allowed in GIT_PLUS_ALLOWLIST)


def rule_inst_001_git_plus(install_cell: str, file: str, cell_idx: int) -> list[Finding]:
    """Every pip command on the line, conditional ones included.

    The question is whether the cell can reach a `git+` source at all, so a fallback, a `(...)`
    group and an `if ...; then` body all count, which `unconditional_pip_invocations` would drop.
    The command still has to be pip: a `git+` in an `echo` installs nothing.

    Each source is read twice, from the command text and from the arguments shlex made of it:
    `"git+"https://...` is one argument to pip and two words to a text scan."""
    findings: list[Finding] = []
    for line_no, line in _glue_line_continuations(install_cell):
        sources: list[str] = []
        for command, _ in _split_chained(line):
            inv = parse_pip_line(command, line_no)
            if inv is None:
                continue
            sources += _GIT_SOURCE_RE.findall(command)
            sources += [arg for arg in inv.packages if arg.lower().startswith("git+")]
        # Per source, not per line: one allowlisted repo must not clear a prohibited one.
        if not sources or all(_git_source_is_allowed(source) for source in sources):
            continue
        findings.append(
            Finding(
                rule = "R-INST-001",
                file = file,
                cell = cell_idx,
                line = line_no,
                severity = "error",
                message = "install line uses `git+` (volatile, not pinned to a release)",
                hint = f"replace with a `pip install foo==X.Y.Z` from PyPI; allow-list is {GIT_PLUS_ALLOWLIST}",
            )
        )
    return findings


def _removed_by_cell(
    install_cell: str,
    name: str,
    environment: dict[str, str] | None = None,
) -> bool:
    """Did this cell uninstall `name`, rather than simply never mention it?

    `resolved_set` drops an uninstalled package and the rules read that as "no resolution data, say
    nothing", but removing what a `--no-deps` install needs is the state they exist to catch."""
    wanted = _canonical_project(name)
    removed = False
    for inv in unconditional_pip_invocations(install_cell):
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or _canonical_project(sp.name) != wanted:
                continue
            if _is_dry_run(inv):
                continue
            if inv.action == "install" and not _requirement_applies(raw, environment):
                continue
            # Replayed in order: `pip uninstall x; pip install x` leaves x installed.
            removed = inv.action == "uninstall"
    return removed


def rule_inst_002_no_deps_transitive(
    install_cell: str, colab: dict[str, str], file: str, cell_idx: int
) -> list[Finding]:
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    environment = _marker_environment(colab)
    for inv in unconditional_pip_invocations(install_cell):
        if "--no-deps" not in inv.flags:
            continue
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or not _requirement_applies(raw, environment):
                continue
            v = explicit_pin(sp)
            if v is None:
                continue
            for target in (
                "tokenizers",
                "torchao",
                "accelerate",
                "datasets",
                "huggingface-hub",
                "huggingface_hub",
            ):
                spec_str, ops = transitive_constraint(sp.name, v, target)
                if not ops:
                    continue
                resolved_target = res.get(target.replace("_", "-"), res.get(target))
                if resolved_target is None:
                    if not _removed_by_cell(install_cell, target, environment):
                        continue
                    findings.append(
                        Finding(
                            rule = "R-INST-002",
                            file = file,
                            cell = cell_idx,
                            line = inv.line_no,
                            severity = "error",
                            message = f"`--no-deps {sp.name}=={v}` requires `{target}` {spec_str}, and this cell uninstalls it",
                            hint = f"drop the `pip uninstall {target}` or reinstall it inside {sp.name}'s window",
                        )
                    )
                    continue
                if not constraint_satisfied(resolved_target, ops):
                    findings.append(
                        Finding(
                            rule = "R-INST-002",
                            file = file,
                            cell = cell_idx,
                            line = inv.line_no,
                            severity = "error",
                            message = f"`--no-deps {sp.name}=={v}` leaves transitive `{target}` unpinned: resolved {resolved_target} violates {sp.name}'s requirement {spec_str!r}",
                            hint = f'add `"{target}>={ops[0][1]},<={ops[-1][1]}"` (or the exact window from the metadata) to the same install line',
                        )
                    )
    return findings


def _install_cell_lower_bound(
    install_cell: str,
    target: str,
    environment: dict[str, str] | None = None,
) -> str | None:
    """Return the highest lower bound any install line places on `target` (treating `==V` as both bounds), or None. Used by R-INST-003 so a `torchao>=0.16.0` line satisfies the floor without a `==` pin."""
    best: str | None = None
    for inv in unconditional_pip_invocations(install_cell):
        if _is_dry_run(inv):
            continue
        if inv.action == "uninstall":
            # The cell removed it, so no earlier line still floors it.
            if any(
                (sp := parse_spec(raw)) is not None and sp.name == target for raw in inv.packages
            ):
                best = None
            continue
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or sp.name != target:
                continue
            if not _requirement_applies(raw, environment):
                continue
            for op, ver in sp.pins:
                if op in ("==", ">="):
                    if best is None or cmp_versions(ver, best) > 0:
                        best = ver
    return best


def _compatible_release_ceiling(version: str) -> str | None:
    """The exclusive ceiling `~=version` implies: `~=2.10.0` allows `<2.11`, `~=2.10` `<3`.

    PEP 440 drops the last component and increments what is then last."""
    parts = normalise_version(version).split(".")
    if len(parts) < 2:
        return None
    head = parts[:-1]
    try:
        head[-1] = str(int(head[-1]) + 1)
    except ValueError:
        return None
    return ".".join(head)


# pip accepts archive URLs/paths; PEP 427 puts the version in the filename's second field.
_ARCHIVE_RE = re.compile(
    r"(?P<name>[A-Za-z0-9._-]+?)-(?P<version>\d[^-]*?)(?:-.*)?\.(?:whl|tar\.gz|zip)$",
    re.IGNORECASE,
)


def _archive_requirement(argument: str) -> tuple[str, str | None] | None:
    """`(project, version)` for a direct archive install, or None when it is not one.

    The version is None when the target is named but its archive does not encode one, as in
    `torchcodec @ https://.../v0.13.0.zip`: the package is replaced, by something this cannot name."""
    named, sep, reference = argument.partition("@")
    if sep and "://" in reference:
        # Strip a PEP 508 marker from a delimited URL; `;` is legal in a bare path.
        reference = reference.split(";", 1)[0]
        argument = reference.strip()
        named = named.strip().split("[", 1)[0].replace("_", "-").lower()
    else:
        named = ""
    lowered = argument.lower().split("#", 1)[0].split("?", 1)[0]
    if "://" not in argument and not lowered.endswith((".whl", ".tar.gz", ".zip")):
        return None
    leaf = argument.split("#", 1)[0].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    leaf = urllib.parse.unquote(leaf)  # a URL spells the local tag `%2Bcu130`
    match = _ARCHIVE_RE.match(leaf)
    if match is None:
        return (named, None) if named else None
    project = match.group("name").replace("_", "-").lower()
    return (named or project), match.group("version")


def cmp_releases(a: str, b: str) -> int:
    """`cmp_versions` with the release segments padded, as PEP 440 compares them.

    Stopping at the shorter tuple reads `0.11.0` as above `0.11`, which is harmless for ordering
    and wrong wherever the question is whether two spellings name the same release."""
    left = [int(part) for part in re.findall(r"\d+", normalise_version(a))]
    right = [int(part) for part in re.findall(r"\d+", normalise_version(b))]
    width = max(len(left), len(right))
    left += [0] * (width - len(left))
    right += [0] * (width - len(right))
    return (left > right) - (left < right)


def _exclusion_covers_minor(version: str, exclusion: str) -> bool:
    """True when `!=exclusion` rules out every release in `version`'s minor.

    Only a wildcard can: `!=0.11.*` takes the whole 0.11 line, while `!=0.11` and `!=0.11.1.*` each
    remove one release or one patch line and leave the minor reachable."""
    wanted = normalise_version(exclusion).split(".")
    if not wanted or wanted[-1] != "*":
        return False
    wanted = wanted[:-1]
    return len(wanted) <= 2 and normalise_version(version).split(".")[: len(wanted)] == wanted


def _version_is_excluded(version: str, exclusion: str) -> bool:
    """True when `!=exclusion` rules `version` out. A trailing `.*` is a prefix match."""
    wanted = normalise_version(exclusion).split(".")
    if wanted and wanted[-1] == "*":
        wanted = wanted[:-1]
        return normalise_version(version).split(".")[: len(wanted)] == wanted
    return cmp_releases(version, exclusion) == 0


def _window_names_one_minor(
    floor: str | None,
    ceiling: str | None,
    cap: str | None = None,
) -> bool:
    """True when the window above `floor` cannot leave the minor `floor` is in.

    A window lands on the newest release it admits, which it names only when there is one minor to
    land in: `>=0.10,<0.11` qualifies, `>=0.10,<0.12` does not."""
    if floor is None:
        return False
    if cap is not None and version_minor(cap) == version_minor(floor):
        return True
    if ceiling is None:
        return False
    next_minor = _compatible_release_ceiling(f"{version_minor(floor)}.0")
    return next_minor is not None and cmp_releases(ceiling, next_minor) <= 0


def _spec_window(
    pins: list[tuple[str, str]],
) -> tuple[str | None, str | None, str | None, str | None, list[str], bool]:
    """`(exact, floor, cap, ceiling, exclusions, floor_excludes_itself)` for one requirement.

    `cap` is an inclusive `<=`, which names the version pip lands on; `ceiling` is an exclusive `<`
    or the one `~=` implies, which does not. A `>` floor comes back with the flag set, since the
    endpoint it names is the one version pip will not install."""
    exact = floor = cap = ceiling = None
    floor_excludes_itself = False
    exclusions: list[str] = []
    for op, ver in pins:
        if op == "==":
            exact = ver
        elif op == "!=":
            exclusions.append(ver)
        elif op in (">=", ">", "~="):
            if floor is None or cmp_releases(ver, floor) > 0:
                floor = ver
                floor_excludes_itself = op == ">"
            elif cmp_releases(ver, floor) == 0 and op == ">":
                floor_excludes_itself = True
        elif op == "<=":
            if cap is None or cmp_versions(ver, cap) < 0:
                cap = ver
        elif op == "<":
            if ceiling is None or cmp_versions(ver, ceiling) < 0:
                ceiling = ver
        if op == "~=":
            implied = _compatible_release_ceiling(ver)
            if implied is not None and (ceiling is None or cmp_versions(implied, ceiling) < 0):
                ceiling = implied
    return exact, floor, cap, ceiling, exclusions, floor_excludes_itself


# Flags that make pip resolve from the index instead of keeping what is installed.
_RESOLVE_ANYWAY_LONG = frozenset({"--upgrade", "--force-reinstall", "--ignore-installed"})
_RESOLVE_ANYWAY_SHORT = frozenset({"U", "I"})


def _is_dry_run(inv: "PipInvocation") -> bool:
    """`--dry-run` means pip changes nothing: "Don't actually install anything, just print what would
    be" (https://pip.pypa.io/en/stable/cli/pip_install/).

    Both readers have to honour it: `_effective_version` alone was not enough, since `resolved_set`
    had already seeded the version from the same command's pins."""
    return "--dry-run" in inv.flags


def _forces_resolution(flags: set[str]) -> bool:
    """True when any flag makes pip re-resolve rather than keep what is installed.

    Short options bundle: pip takes `-Uq` and parse_pip_line keeps it as one token, so the letters
    are compared rather than the token."""
    if flags & _RESOLVE_ANYWAY_LONG:
        return True
    return any(
        not flag.startswith("--") and flag.startswith("-") and set(flag[1:]) & _RESOLVE_ANYWAY_SHORT
        for flag in flags
    )


def _highest_minor_below(ceiling: str) -> str:
    """The newest minor an exclusive `<ceiling` can still land on: `<0.11` -> `0.10`.

    pip resolves a bounded window to the newest candidate it admits; which patch is not derivable
    offline, and the rules only compare minors. Only a ceiling ON a minor boundary excludes that
    whole minor, so `<0.10.5` still lands on 0.10. The major is carried rather than assumed to be
    0, or torch's `2.N` windows read as `0.N`."""
    parts = [p for p in re.split(r"[.]", ceiling.strip()) if p.isdigit()]
    if len(parts) < 2:
        return ""
    major, minor = int(parts[0]), int(parts[1])
    if any(int(p) for p in parts[2:]):
        return f"{major}.{minor}"
    if minor >= 1:
        return f"{major}.{minor - 1}"
    return ""


def _effective_version(
    install_cell: str,
    target: str,
    resolved: str | None,
    environment: dict[str, str] | None = None,
) -> tuple[str | None, bool]:
    """`resolved` walked forward through the cell's own requirements, in invocation order.

    resolved_set() keeps only `==` and `<=` and applies them at once, but order decides between
    them: without this, R-INST-004's own `torchcodec>=0.12.0` remedy could not clear the error it
    offers.

    Each requirement is a window. An install moves the version into it when it falls outside and
    leaves it alone when it does not, as pip does. It moves to the window's floor, or to an
    inclusive `<=` when the move is downwards, and moving down names a version only when the window
    holds one minor, the granularity the callers compare on. A `>` floor names the one version pip
    will not install, so it too needs a ceiling pinning the minor. Anything that cannot say where
    the install lands clears the version rather than keeping a stale one, and a bound on an absent
    package leaves it absent unless it carries a floor.

    Returns `(version, exact)`. An open floor moves the version up without naming it, since pip
    takes the newest release above it, so it comes back inexact and may only be used where every
    version at or above it gives the same answer."""
    current = resolved
    exact_known = True
    for inv in unconditional_pip_invocations(install_cell):
        if "--dry-run" in inv.flags:
            # A resolution probe leaves the environment unchanged.
            continue
        # pip intersects repeated arguments into one requirement, so treat them as one window.
        pins: list[tuple[str, str]] = []
        named = False
        replaced_unnamed = False
        for raw in inv.packages:
            if not _requirement_applies(raw, environment):
                continue
            # Before parse_spec, which reads `./x-1.0.whl` as a project called `.`.
            archive = _archive_requirement(raw)
            if archive is not None:
                if archive[0] == target:
                    named = True
                    if archive[1] is None:
                        replaced_unnamed = True
                    else:
                        pins.append(("==", archive[1]))
                continue
            sp = parse_spec(raw)
            if sp is None or sp.name != target:
                continue
            named = True
            pins.extend(sp.pins)
        if not named:
            continue
        if inv.action == "uninstall":
            current = None
            continue
        if not pins and not replaced_unnamed and _forces_resolution(inv.flags):
            current, exact_known = None, True
            continue
        if replaced_unnamed:
            current, exact_known = None, True
            continue
        exact, floor, cap, ceiling, exclusions, exclusive_floor = _spec_window(pins)
        landing = floor if _window_names_one_minor(floor, ceiling, cap) else None
        if landing is not None and _split_prerelease(landing)[1]:
            # `~=0.12.0rc1` admits stable 0.12 too, so name the minor, but only where the window admits it.
            core = _split_prerelease(landing)[0]
            if (cap is None or cmp_versions(core, cap) <= 0) and (
                ceiling is None or cmp_versions(core, ceiling) < 0
            ):
                landing = core
        if landing is None and ceiling is not None:
            below = _highest_minor_below(ceiling)
            if below and (floor is None or cmp_versions(below, floor) >= 0):
                landing = below
        # Every specifier applies, so an inclusive cap the exclusive ceiling excludes is not the landing.
        cap_exact = True
        if cap is not None and ceiling is not None and cmp_versions(cap, ceiling) >= 0:
            cap, cap_exact = landing, landing is not None
        if exact is not None:
            current, exact_known = exact, True
        elif current is None or _forces_resolution(inv.flags):
            forced_off = current is not None
            # `--upgrade` takes the newest admitted version, not the installed one. Absent: `<=V` names it,
            # a floor bounds it, and an exclusive ceiling names nothing.
            if cap is not None:
                current, exact_known = cap, cap_exact
            elif landing is not None and floor is not None:
                # pip takes the newest release a bounded window admits.
                current, exact_known = landing, True
            elif floor is not None:
                # An inexact floor is only read by checks that hold above it, so `>V` may include V.
                current, exact_known = floor, False
            elif (
                forced_off
                and landing is not None
                and cmp_versions(version_minor(current), landing) == 0
            ):
                # An upgrade with a ceiling cannot leave the installed minor.
                current, exact_known = landing, True
            elif forced_off:
                current, exact_known = None, True
        elif floor is not None and (
            cmp_versions(floor, current) > 0
            or (exclusive_floor and cmp_versions(floor, current) == 0)
        ):
            if cap is not None:
                current, exact_known = cap, cap_exact
            elif landing is not None:
                current, exact_known = landing, True
            else:
                current, exact_known = floor, False
        elif cap is not None and cmp_versions(current, cap) > 0:
            current, exact_known = cap, cap_exact
        elif ceiling is not None and cmp_versions(current, ceiling) >= 0:
            current, exact_known = landing, True
        if current is not None and any(_version_is_excluded(current, ver) for ver in exclusions):
            # Only an exclusion covering the whole minor removes the landing; check it against the window.
            if (
                landing is not None
                and (cap is None or cmp_versions(landing, cap) <= 0)
                and (ceiling is None or cmp_versions(landing, ceiling) < 0)
                and (floor is None or cmp_versions(landing, floor) >= 0)
                and not any(_exclusion_covers_minor(landing, ver) for ver in exclusions)
                and not (
                    cap is not None
                    and cmp_versions(landing, cap) == 0
                    and any(_version_is_excluded(cap, ver) for ver in exclusions)
                )
            ):
                current, exact_known = landing, True
            else:
                current, exact_known = None, True
    return current, exact_known if current is not None else True


def rule_inst_003_peft_torchao(
    install_cell: str, colab: dict[str, str], file: str, cell_idx: int
) -> list[Finding]:
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    peft_v = res.get("peft")
    if not peft_v:
        return findings
    torchao_explicit = _install_cell_lower_bound(
        install_cell, "torchao", _marker_environment(colab)
    )
    torchao_resolved = torchao_explicit or res.get("torchao")
    for floor in PEFT_TORCHAO_FLOOR:
        if cmp_versions(peft_v, floor["trigger_peft"]) >= 0:
            if (
                torchao_resolved is None
                or cmp_versions(torchao_resolved, floor["torchao_floor"]) < 0
            ):
                findings.append(
                    Finding(
                        rule = "R-INST-003",
                        file = file,
                        cell = cell_idx,
                        severity = "error",
                        message = f"resolved peft=={peft_v} requires torchao>={floor['torchao_floor']}; install cell asserts torchao={torchao_resolved or '(none)'}",
                        hint = f'add `!pip install --no-deps --upgrade "torchao>={floor["torchao_floor"]}"` to the install cell',
                    )
                )
    return findings


def _codec_works_above(torch_floor: str, codec_minor: str) -> bool:
    """Is there ANY torch minor at or above `torch_floor` this codec minor can pair with?

    A floor is normally too weak to judge, since the row that applies depends on where pip lands,
    but not when every candidate is excluded: `torch>=2.11` with `torchcodec==0.10` fails on 2.11
    and on everything past it, whichever release pip picks."""
    if at_least(codec_minor, TORCHCODEC_ABI_STABLE_CODEC):
        return True
    return any(
        cmp_versions(row, torch_floor) >= 0 and codec_minor in minors
        for row, minors in TORCH_TORCHCODEC.items()
    )


def rule_inst_004_torchcodec_torch(
    install_cell: str, colab: dict[str, str], file: str, cell_idx: int
) -> list[Finding]:
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    environment = _marker_environment(colab)
    torch_v, torch_exact = _effective_version(install_cell, "torch", res.get("torch"), environment)
    codec_v, codec_exact = _effective_version(
        install_cell, "torchcodec", res.get("torchcodec"), environment
    )
    if not torch_v or not codec_v:
        return findings
    # torchcodec 0.12+ is ABI-stable against torch >= 2.11 (TORCH_TARGET_VERSION 2.11).
    # An inexact prerelease floor admits the stable release above it.
    codec_clears_abi = at_least(codec_v, TORCHCODEC_ABI_STABLE_CODEC) or (
        not codec_exact and cmp_versions(version_minor(codec_v), TORCHCODEC_ABI_STABLE_CODEC) >= 0
    )
    if at_least(torch_v, TORCHCODEC_ABI_STABLE_TORCH) and codec_clears_abi:
        return findings
    t_minor = version_minor(torch_v)
    c_minor = version_minor(codec_v)
    allowed = TORCH_TORCHCODEC.get(t_minor)
    if allowed is None:
        if not at_least(torch_v, TORCHCODEC_ABI_STABLE_TORCH):
            return findings
        if not codec_exact and not at_least(c_minor, TORCHCODEC_ABI_STABLE_CODEC):
            return findings
        findings.append(
            Finding(
                rule = "R-INST-004",
                file = file,
                cell = cell_idx,
                severity = "error",
                message = f"torch=={torch_v} (minor {t_minor}) is incompatible with torchcodec=={codec_v} (minor {c_minor}); torchcodec <{TORCHCODEC_ABI_STABLE_CODEC} is built against a single older torch minor",
                hint = f"pin `torchcodec>={TORCHCODEC_ABI_STABLE_CODEC}.0` (the ABI-stable line, which targets torch >={TORCHCODEC_ABI_STABLE_TORCH})",
            )
        )
        return findings
    if not torch_exact:
        # The row depends on which torch the floor resolves to, unless no release above it fits.
        if codec_exact and not _codec_works_above(t_minor, c_minor):
            findings.append(
                Finding(
                    rule = "R-INST-004",
                    file = file,
                    cell = cell_idx,
                    severity = "error",
                    message = f"torch>={torch_v} is incompatible with torchcodec=={codec_v} (minor {c_minor}) at every torch minor the floor admits",
                    hint = f"pin `torchcodec>={TORCHCODEC_ABI_STABLE_CODEC}.0` (the ABI-stable line, which targets torch >={TORCHCODEC_ABI_STABLE_TORCH})",
                )
            )
        return findings
    if not codec_exact and cmp_versions(c_minor, sorted(allowed)[-1]) <= 0:
        return findings
    if c_minor not in allowed:
        findings.append(
            Finding(
                rule = "R-INST-004",
                file = file,
                cell = cell_idx,
                severity = "error",
                message = f"torch=={torch_v} (minor {t_minor}) is incompatible with torchcodec=={codec_v} (minor {c_minor}); compatible minors: {sorted(allowed)}",
                hint = f"pin `torchcodec=={sorted(allowed)[-1]}` (or remove the explicit pin and let pip resolve)",
            )
        )
    return findings


def rule_inst_005_transformers_tokenizers(
    install_cell: str, colab: dict[str, str], file: str, cell_idx: int
) -> list[Finding]:
    """Fires only when transformers is installed with `--no-deps` (otherwise pip resolves tokenizers transitively and flagging would be a false positive). Targets the PR #261b/#264 pattern: `--no-deps transformers==X` next to a Colab `tokenizers` outside transformers's window."""
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    tf = res.get("transformers")
    tok = res.get("tokenizers")
    if not tf:
        return findings
    tokenizers_removed = tok is None and _removed_by_cell(
        install_cell, "tokenizers", _marker_environment(colab)
    )
    if tok is None and not tokenizers_removed:
        return findings
    environment = _marker_environment(colab)
    transformers_line_no_deps = False
    for inv in unconditional_pip_invocations(install_cell):
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or sp.name != "transformers":
                continue
            if explicit_pin(sp) is None or not _requirement_applies(raw, environment):
                continue
            if "--no-deps" in inv.flags:
                transformers_line_no_deps = True
                break
        if transformers_line_no_deps:
            break
    if not transformers_line_no_deps:
        return findings
    spec_str, ops = transitive_constraint("transformers", tf, "tokenizers")
    if not ops:
        return findings
    if tokenizers_removed:
        findings.append(
            Finding(
                rule = "R-INST-005",
                file = file,
                cell = cell_idx,
                severity = "error",
                message = f"`--no-deps transformers=={tf}` requires tokenizers {spec_str}, and this cell uninstalls it",
                hint = "drop the `pip uninstall tokenizers` or reinstall it inside the window",
            )
        )
        return findings
    if not constraint_satisfied(tok, ops):
        findings.append(
            Finding(
                rule = "R-INST-005",
                file = file,
                cell = cell_idx,
                severity = "error",
                message = f"`--no-deps transformers=={tf}` skips pip's transitive resolver; resolved tokenizers={tok} violates {spec_str}",
                hint = f'pin `"tokenizers{spec_str}"` (or the matching window) on the same `--no-deps` line',
            )
        )
    return findings


_RE_DOUBLE_BANG = re.compile(r"^[ \t]*!{2,}\s*pip\b", re.MULTILINE)


def rule_inst_006_double_bang(install_cell: str, file: str, cell_idx: int) -> list[Finding]:
    findings: list[Finding] = []
    for m in _RE_DOUBLE_BANG.finditer(install_cell):
        line_no = install_cell.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(
                rule = "R-INST-006",
                file = file,
                cell = cell_idx,
                line = line_no,
                severity = "warning",
                message = "double-bang `!!pip` runs in a subshell; almost always a typo for `!pip`",
                hint = "use a single `!`",
            )
        )
    return findings


class _APIScanner(ast.NodeVisitor):
    """Scan user-facing code cells for known deprecated patterns. R-API-001 (`for_training`/`for_inference`) is intentionally absent: those helpers are still live as of 2026-05 (PR #221 removed them cosmetically, not as a deprecation). R-API-004 catches actual removals dynamically."""

    def __init__(self, file: str, cell_idx: int):
        self.file = file
        self.cell_idx = cell_idx
        self.findings: list[Finding] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "SFTConfig":
            for kw in node.keywords:
                if (
                    kw.arg == "optim"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "adamw_torch_fused"
                ):
                    self.findings.append(
                        Finding(
                            rule = "R-API-003",
                            file = self.file,
                            cell = self.cell_idx,
                            line = kw.value.lineno,
                            severity = "warning",
                            message = "`optim='adamw_torch_fused'` is suboptimal under Unsloth's memory-efficient training",
                            hint = 'use `optim="adamw_8bit"` (or `"paged_adamw_8bit"` for GRPO)',
                        )
                    )
        self.generic_visit(node)


def scan_user_cells(nb: dict[str, Any], file: str) -> list[Finding]:
    findings: list[Finding] = []
    install_idxs = {i for i, _ in install_cells(nb)}
    for i, src in code_cells(nb):
        if i in install_idxs:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        scanner = _APIScanner(file = file, cell_idx = i)
        scanner.visit(tree)
        findings.extend(scanner.findings)
    return findings


POLICY_CLAUSES_DEFAULT = [
    # (id, regex, applies_to_predicate_on_install_cell_text)
    (
        "torchao-floor",
        re.compile(r"torchao>=0\.16\.0"),
        lambda cell: bool(re.search(r"\bpeft\b", cell)),
    ),
    (
        "tokenizers-window",
        re.compile(r"tokenizers>=0\.22\.0,<=0\.23\.0"),
        lambda cell: bool(re.search(r"--no-deps[^\n]*transformers==", cell)),
    ),
]


def extract_policy_clauses(update_script: pathlib.Path) -> list[tuple[str, re.Pattern[str], Any]]:
    """Best-effort scan of update_all_notebooks.py for canonical phrases; falls back to POLICY_CLAUSES_DEFAULT (which we use directly today). The permissive regexes avoid false positives on template rewords."""
    return list(POLICY_CLAUSES_DEFAULT)


def rule_l12_exceptions_coverage(notebooks_dir: pathlib.Path) -> list[Finding]:
    findings: list[Finding] = []
    update_script = notebooks_dir / "update_all_notebooks.py"
    exceptions = _extract_dont_update_exceptions(update_script)
    clauses = extract_policy_clauses(update_script)
    for name in exceptions:
        path = notebooks_dir / "nb" / name
        if not path.is_file():
            continue
        nb = load_notebook(path)
        for idx, cell in install_cells(nb):
            # install_cells is a text heuristic; `!echo "pip install x"` runs no pip.
            if not any(True for _ in iter_pip_invocations(cell)):
                continue
            for cid, pat, applies in clauses:
                if not applies(cell):
                    continue
                if not pat.search(cell):
                    findings.append(
                        Finding(
                            rule = "R-EXC-001",
                            file = str(path),
                            cell = idx,
                            severity = "error",
                            message = f"DONT_UPDATE_EXCEPTIONS notebook missing policy clause `{cid}` (pattern {pat.pattern!r})",
                            hint = f"add the matching install line; the regenerator can't reach this notebook",
                        )
                    )
    return findings


def _extract_dont_update_exceptions(update_script: pathlib.Path) -> list[str]:
    if not update_script.is_file():
        return []
    src = update_script.read_text(encoding = "utf-8")
    m = re.search(r"DONT_UPDATE_EXCEPTIONS\s*=\s*\[(.*?)\]", src, re.DOTALL)
    if not m:
        return []
    out: list[str] = []
    for line in m.group(1).splitlines():
        m2 = re.match(r'\s*"([^"]+\.ipynb)"', line)
        if m2:
            out.append(m2.group(1))
    return out


def cmd_drift(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    update_script = nbdir / "update_all_notebooks.py"
    if not update_script.is_file():
        print(f"FAIL: {update_script} not found", file = sys.stderr)
        return 2
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = nbdir).decode().strip()
    subprocess.run(
        ["git", "-C", str(nbdir), "stash", "--include-untracked"],
        check = False,
        capture_output = True,
    )
    # try/finally so the stash pop runs even on SystemExit or KeyboardInterrupt.
    findings: list[Finding] = []
    rc: int
    try:
        try:
            proc = subprocess.run(
                [sys.executable, str(update_script)],
                cwd = nbdir,
                capture_output = True,
                text = True,
                timeout = 600,
            )
        except subprocess.TimeoutExpired:
            print(
                "FAIL: update_all_notebooks.py timed out (>600s)",
                file = sys.stderr,
            )
            rc = 2
        else:
            if proc.returncode != 0:
                print(
                    f"FAIL: update_all_notebooks.py exited {proc.returncode}",
                    file = sys.stderr,
                )
                sys.stderr.write(proc.stderr[-2000:])
                rc = 2
            else:
                diff_proc = subprocess.run(
                    ["git", "-C", str(nbdir), "diff", "--stat"],
                    capture_output = True,
                    text = True,
                )
                if diff_proc.stdout.strip():
                    for line in diff_proc.stdout.splitlines():
                        findings.append(
                            Finding(
                                rule = "R-DRIFT-001",
                                file = line.strip(),
                                severity = "error",
                                message = "generator-vs-checked-in drift",
                                hint = "run `python update_all_notebooks.py` and commit the diff",
                            )
                        )
                rc = 0 if not findings else 1
    finally:
        subprocess.run(
            ["git", "-C", str(nbdir), "checkout", "."],
            check = False,
            capture_output = True,
        )
        subprocess.run(
            ["git", "-C", str(nbdir), "stash", "pop"],
            check = False,
            capture_output = True,
        )
    _emit(findings)
    return rc


def cmd_convert(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents = True, exist_ok = True)
    converter = HERE / "notebook_to_python.py"
    if not converter.is_file():
        print(f"FAIL: {converter} not found", file = sys.stderr)
        return 2
    notebooks = list(iter_notebooks(nbdir, include_templates = True))
    failed: list[Finding] = []
    BATCH = 32
    for i in range(0, len(notebooks), BATCH):
        chunk = notebooks[i : i + BATCH]
        proc = subprocess.run(
            [sys.executable, str(converter), "-o", str(out), *map(str, chunk)],
            capture_output = True,
            text = True,
        )
        if proc.returncode != 0:
            for nb in chunk:
                failed.append(
                    Finding(
                        rule = "R-CONV-001",
                        file = str(nb),
                        severity = "error",
                        message = "notebook_to_python.py failed for this notebook",
                        hint = proc.stderr[-200:].strip(),
                    )
                )
    print(f"converted {len(notebooks) - len(failed)}/{len(notebooks)} notebooks to {out}")
    _emit(failed)
    return 0 if not failed else 1


def cmd_lint(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    colab_path = pathlib.Path(args.colab_pin).resolve() if args.colab_pin else COLAB_FALLBACK_FILE
    _set_colab_oracle_dir(colab_path.parent)
    colab = parse_pip_freeze(colab_path)
    if not colab:
        print(
            f"WARN: Colab pip-freeze empty / missing at {colab_path}; using empty oracle",
            file = sys.stderr,
        )

    findings: list[Finding] = []
    notebooks = list(iter_notebooks(nbdir))
    for path in notebooks:
        try:
            nb = load_notebook(path)
        except (json.JSONDecodeError, OSError) as e:
            findings.append(
                Finding(
                    rule = "R-CONV-002",
                    file = str(path),
                    severity = "error",
                    message = f"notebook unreadable: {e}",
                )
            )
            continue
        rel = str(path.relative_to(nbdir))
        env = target_environment(rel)
        oracle = colab if env == "colab" else {}
        cells = install_cells(nb)
        for idx, cell in cells:
            findings += rule_inst_001_git_plus(cell, rel, idx)
            findings += rule_inst_006_double_bang(cell, rel, idx)
        # Install steps may span cells, so merge before resolving.
        merged = "\n".join(c for _, c in cells)
        # Compat rules replay only unconditional invocations; R-INST-001 already saw conditional ones.
        if not any(True for _ in unconditional_pip_invocations(merged)):
            merged = ""
        if env == "colab" and merged:
            first_cell = cells[0][0] if cells else None
            findings += rule_inst_003_peft_torchao(merged, oracle, rel, first_cell)
            findings += rule_inst_004_torchcodec_torch(merged, oracle, rel, first_cell)
            findings += rule_inst_005_transformers_tokenizers(merged, oracle, rel, first_cell)
            if not args.no_pypi:
                findings += rule_inst_002_no_deps_transitive(merged, oracle, rel, first_cell)
        findings += scan_user_cells(nb, rel)
    _emit(findings)
    return 0 if not any(f.severity == "error" for f in findings) else 1


def cmd_exceptions(args: argparse.Namespace) -> int:
    findings = rule_l12_exceptions_coverage(pathlib.Path(args.notebooks_dir).resolve())
    _emit(findings)
    return 0 if not findings else 1


def cmd_api(args: argparse.Namespace) -> int:
    surface_path = pathlib.Path(args.surface).resolve()
    if not surface_path.is_file():
        print(
            f"FAIL: {surface_path} not found; run dump-api-surface first",
            file = sys.stderr,
        )
        return 2
    surface = json.loads(surface_path.read_text())
    converted = pathlib.Path(args.converted_dir).resolve()
    findings: list[Finding] = []
    fast_models = (
        set(surface.get("FastVisionModel", []))
        | set(surface.get("FastLanguageModel", []))
        | set(surface.get("FastModel", []))
    )
    for py in sorted(converted.glob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding = "utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                base = node.func.value
                if isinstance(base, ast.Name) and base.id in (
                    "FastVisionModel",
                    "FastLanguageModel",
                    "FastModel",
                ):
                    surface_set = set(surface.get(base.id, []))
                    if surface_set and node.func.attr not in surface_set:
                        findings.append(
                            Finding(
                                rule = "R-API-004",
                                file = str(py.name),
                                line = node.lineno,
                                severity = "error",
                                message = f"`{base.id}.{node.func.attr}` is not in the live API surface for the pinned unsloth tag",
                                hint = "check the unsloth changelog for a renamed/removed API",
                            )
                        )
    _emit(findings)
    return 0 if not findings else 1


def cmd_all(args: argparse.Namespace) -> int:
    rcs: list[int] = []
    rcs.append(cmd_drift(argparse.Namespace(notebooks_dir = args.notebooks_dir)))
    rcs.append(
        cmd_lint(
            argparse.Namespace(
                notebooks_dir = args.notebooks_dir,
                colab_pin = args.colab_pin,
                no_pypi = args.no_pypi,
            )
        )
    )
    rcs.append(cmd_exceptions(argparse.Namespace(notebooks_dir = args.notebooks_dir)))
    return 0 if all(rc == 0 for rc in rcs) else 1


def _fetch_oracle(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout = 15) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"FAIL: could not fetch {url}: {e}", file = sys.stderr)
        return None


# The R-INST rules seed on these, so a truncated payload is refused.
_COLAB_PIP_REQUIRED = frozenset(
    {"torch", "torchcodec", "peft", "torchao", "transformers", "tokenizers"}
)


def _oracle_payload_is_usable(upstream_name: str, data: bytes) -> bool:
    """Does a freshly fetched oracle still carry what a rule reads out of it?"""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    parsed = _COLAB_ORACLE_PARSERS[upstream_name](text)
    if upstream_name == COLAB_STRICT_ORACLE and not _COLAB_PIP_REQUIRED <= parsed.keys():
        return False
    return all(
        _strict_key_usable(upstream_name, key, parsed)
        for key in COLAB_STRICT_ORACLE_KEYS.get(upstream_name, frozenset())
    )


def cmd_refresh_colab(args: argparse.Namespace) -> int:
    """Pull the latest Colab pip-freeze.gpu.txt and write to disk. --all refreshes every oracle file into --snapshot-dir instead, which is how a colab-diff drift report is acknowledged in one command."""
    if args.all:
        snapshot_dir = pathlib.Path(args.snapshot_dir).resolve()
        # Fetch all before writing, so a transient failure cannot leave mixed generations.
        payloads: dict[str, bytes] = {}
        skipped: list[str] = []
        for upstream_name, snapshot_name in COLAB_ORACLE_FILES.items():
            rule_bearing = (
                upstream_name == COLAB_STRICT_ORACLE or upstream_name in COLAB_STRICT_ORACLE_KEYS
            )
            data = _fetch_oracle(COLAB_ORACLE_BASE_URL + upstream_name)
            reason = None
            if data is None:
                reason = "could not be fetched"
            elif rule_bearing and not _oracle_payload_is_usable(upstream_name, data):
                # A payload the rules cannot read would leave both diff sides equally empty.
                reason = "carries no key the rules can read"
            if reason is None:
                payloads[snapshot_name] = data
                continue
            if rule_bearing:
                print(
                    f"FAIL: refresh-colab --all: {upstream_name} {reason}; "
                    "no snapshot was written",
                    file = sys.stderr,
                )
                return 2
            # Advisory oracle: keep the stale snapshot on a transient upstream failure.
            print(f"::notice::skipping {upstream_name}: {reason}")
            skipped.append(upstream_name)
        snapshot_dir.mkdir(parents = True, exist_ok = True)
        # The set lands together or not at all; copy aside first, since restoring by rewrite needs
        # the space the failure proved missing.
        preserved: dict[str, pathlib.Path] = {}
        for name in payloads:
            live = snapshot_dir / name
            if live.is_file():
                keep = snapshot_dir / f".{name}.rollback"
                shutil.copy2(live, keep)
                preserved[name] = keep
        written: list[str] = []
        try:
            for snapshot_name, data in payloads.items():
                _atomic_write_bytes(snapshot_dir / snapshot_name, data)
                written.append(snapshot_name)
        except OSError as e:
            for snapshot_name in written:
                keep = preserved.get(snapshot_name)
                if keep is not None:
                    os.replace(keep, snapshot_dir / snapshot_name)
                else:
                    (snapshot_dir / snapshot_name).unlink(missing_ok = True)
            print(
                f"FAIL: refresh-colab --all could not write the snapshot set ({e}); "
                "the committed one was restored",
                file = sys.stderr,
            )
            return 2
        finally:
            for keep in preserved.values():
                keep.unlink(missing_ok = True)
        for snapshot_name in written:
            size = len(payloads[snapshot_name])
            print(f"wrote {size} bytes to {snapshot_dir / snapshot_name}")
        if skipped:
            print(f"left as committed: {', '.join(skipped)}")
        return 0
    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents = True, exist_ok = True)
    data = _fetch_oracle(COLAB_PIP_FREEZE_URL)
    if data is None:
        return 2
    _atomic_write_bytes(out, data)
    print(f"wrote {len(data)} bytes to {out}")
    return 0


def _parse_pip_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+?)\s*(;.*)?$", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _parse_apt_lines(text: str) -> dict[str, str]:
    """`pkg/release,now ver arch [installed[,automatic]]` -> {pkg: ver}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "Listing...":
            continue
        m = re.match(r"^([^/\s]+)/\S+\s+(\S+)\s+\S+\s+\[installed", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _parse_os_lines(text: str) -> dict[str, str]:
    """Free-form `<tool> <version>` lines -> {tool_lower: rest}."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0].lower()] = parts[1]
        else:
            out[parts[0].lower()] = ""
    return out


_COLAB_ORACLE_PARSERS = {
    "pip-freeze.gpu.txt": _parse_pip_lines,
    "apt-list-gpu.txt": _parse_apt_lines,
    "os-info-gpu.txt": _parse_os_lines,
}


def _diff_oracle(
    upstream: dict[str, str], snapshot: dict[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Return (new, removed, changed). new/removed are (key, value); changed is (key, old, new)."""
    new = sorted((k, upstream[k]) for k in upstream.keys() - snapshot.keys())
    removed = sorted((k, snapshot[k]) for k in snapshot.keys() - upstream.keys())
    changed = sorted(
        (k, snapshot[k], upstream[k])
        for k in upstream.keys() & snapshot.keys()
        if upstream[k] != snapshot[k]
    )
    return new, removed, changed


# Expected value shape per oracle key, so a reformat cannot quietly disable markers.
_STRICT_KEY_VALUE_RE: dict[tuple[str, str], "re.Pattern[str]"] = {
    # Matches what _COLAB_PYTHON_RE reads, prerelease included.
    ("os-info-gpu.txt", "python"): re.compile(
        r"^\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?(?:\s|$)"
    ),
}


def _strict_key_usable(oracle: str, key: str, parsed: dict[str, str]) -> bool:
    """Is the key present AND holding a value its consumer can actually read?"""
    if key not in parsed:
        return False
    pattern = _STRICT_KEY_VALUE_RE.get((oracle, key))
    return pattern is None or pattern.search(parsed[key]) is not None


def cmd_colab_diff(args: argparse.Namespace) -> int:
    """Diff each Colab oracle file against its committed snapshot and print NEW/REMOVED/CHANGED. Advisory (rc=0) by default; --strict makes drift in the rule-bearing oracle (COLAB_STRICT_ORACLE) rc=1 so the daily cron fails loudly on upstream rotation."""
    snapshot_dir = pathlib.Path(args.snapshot_dir).resolve()
    any_diff = False
    strict_diff = False
    for upstream_name, snapshot_name in COLAB_ORACLE_FILES.items():
        url = COLAB_ORACLE_BASE_URL + upstream_name
        snap_path = snapshot_dir / snapshot_name
        try:
            with urllib.request.urlopen(url, timeout = 15) as r:
                upstream_text = r.read().decode("utf-8", errors = "replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if upstream_name == COLAB_STRICT_ORACLE or upstream_name in COLAB_STRICT_ORACLE_KEYS:
                # Not compared does not mean no drift.
                any_diff = True
                strict_diff = True
                print(f"::error::colab-diff: could not fetch {url}: {e}")
            else:
                print(f"::warning::colab-diff: could not fetch {url}: {e}")
            continue
        if not snap_path.exists():
            # Check strict keys before the continue, or --strict passes on a missing snapshot.
            strict_file = upstream_name == COLAB_STRICT_ORACLE
            strict_keys = COLAB_STRICT_ORACLE_KEYS.get(upstream_name, frozenset())
            if strict_file or strict_keys:
                any_diff = True
                strict_diff = True
                print(f"::error::colab-diff: no committed snapshot at {snap_path}")
                if strict_keys:
                    print(f"  (rule-bearing key(s) unavailable: {', '.join(sorted(strict_keys))})")
            else:
                print(f"::warning::colab-diff: no committed snapshot at {snap_path}; skipping")
            continue
        snapshot_text = snap_path.read_text(encoding = "utf-8", errors = "replace")
        parser = _COLAB_ORACLE_PARSERS[upstream_name]
        upstream = parser(upstream_text)
        snapshot = parser(snapshot_text)
        new, removed, changed = _diff_oracle(upstream, snapshot)
        n = len(new) + len(removed) + len(changed)
        print(
            f"\n=== {upstream_name}: "
            f"upstream={len(upstream)} snapshot={len(snapshot)} "
            f"diff={n} (new={len(new)} removed={len(removed)} changed={len(changed)}) ==="
        )
        strict_keys = COLAB_STRICT_ORACLE_KEYS.get(upstream_name, frozenset())
        # Strict keys must be present in both, not merely equal.
        missing_keys = sorted(
            k
            for k in strict_keys
            if not _strict_key_usable(upstream_name, k, upstream)
            or not _strict_key_usable(upstream_name, k, snapshot)
        )
        if missing_keys:
            any_diff = True
            strict_diff = True
            print(
                f"::error::colab-diff: {upstream_name} has no readable "
                f"{', '.join(missing_keys)} value; its parser needs updating"
            )
        if not n:
            if not missing_keys:
                print("  no drift")
            continue
        any_diff = True
        drifted_strict_keys = sorted(
            strict_keys.intersection(
                [k for k, _ in new] + [k for k, _ in removed] + [k for k, _, _ in changed]
            )
        )
        if upstream_name == COLAB_STRICT_ORACLE:
            strict_diff = True
        elif drifted_strict_keys:
            strict_diff = True
            print(f"  (rule-bearing key drifted: {', '.join(drifted_strict_keys)})")
        # Every pip-oracle package is rule-bearing, so name --full in the elision.
        cap_new = len(new) if args.full else 50
        cap_removed = len(removed) if args.full else 50
        cap_changed = len(changed) if args.full else 80
        for k, v in new[:cap_new]:
            print(f"  NEW      {k}=={v}")
        if len(new) > cap_new:
            print(f"  ...and {len(new) - cap_new} more new entries (--full to list them)")
        for k, v in removed[:cap_removed]:
            print(f"  REMOVED  {k} (was {v})")
        if len(removed) > cap_removed:
            print(
                f"  ...and {len(removed) - cap_removed} more removed entries (--full to list them)"
            )
        for k, old, ver in changed[:cap_changed]:
            print(f"  CHANGED  {k}: {old} -> {ver}")
        if len(changed) > cap_changed:
            print(
                f"  ...and {len(changed) - cap_changed} more changed entries (--full to list them)"
            )
    if strict_diff and args.strict:
        print(
            "\n::error::A rule-bearing Colab oracle drifted from its committed "
            "snapshot; run `notebook_validator.py refresh-colab --all "
            "--snapshot-dir scripts/data` to acknowledge.",
            file = sys.stderr,
        )
        return 1
    if any_diff:
        print(
            "\n::notice::Colab oracle drifted; run `notebook_validator.py "
            "refresh-colab --all --snapshot-dir scripts/data` at your convenience."
        )
    return 0


def _emit(findings: list[Finding]) -> None:
    n_err = sum(1 for f in findings if f.severity == "error")
    n_warn = sum(1 for f in findings if f.severity == "warning")
    for f in findings:
        print(json.dumps(f.to_dict(), separators = (",", ":")))
    print(f"# total: {n_err} errors, {n_warn} warnings", file = sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog = "notebook_validator")
    sub = p.add_subparsers(dest = "cmd", required = True)

    pa = sub.add_parser("drift")
    pa.add_argument("--notebooks-dir", required = True)

    pa = sub.add_parser("convert")
    pa.add_argument("--notebooks-dir", required = True)
    pa.add_argument("--out", required = True)

    pa = sub.add_parser("lint")
    pa.add_argument("--notebooks-dir", required = True)
    pa.add_argument("--colab-pin", default = None)
    pa.add_argument(
        "--no-pypi",
        action = "store_true",
        help = "skip rules that require live PyPI metadata fetches",
    )

    pa = sub.add_parser("exceptions")
    pa.add_argument("--notebooks-dir", required = True)

    pa = sub.add_parser("api")
    pa.add_argument("--converted-dir", required = True)
    pa.add_argument("--surface", required = True)

    pa = sub.add_parser("all")
    pa.add_argument("--notebooks-dir", required = True)
    pa.add_argument("--colab-pin", default = None)
    pa.add_argument("--no-pypi", action = "store_true")

    pa = sub.add_parser("refresh-colab")
    pa.add_argument("--out", default = str(COLAB_FALLBACK_FILE))
    pa.add_argument(
        "--all",
        action = "store_true",
        help = "refresh every oracle file into --snapshot-dir, not just pip-freeze",
    )
    pa.add_argument("--snapshot-dir", default = str(DATA_DIR))

    pa = sub.add_parser("colab-diff")
    pa.add_argument("--snapshot-dir", default = str(DATA_DIR))
    pa.add_argument(
        "--strict",
        action = "store_true",
        help = f"exit 1 on {COLAB_STRICT_ORACLE} drift (default: advisory; exit 0)",
    )
    pa.add_argument(
        "--full",
        action = "store_true",
        help = "print every drifted entry instead of capping each list",
    )

    args = p.parse_args(argv)
    return {
        "drift": cmd_drift,
        "convert": cmd_convert,
        "lint": cmd_lint,
        "exceptions": cmd_exceptions,
        "api": cmd_api,
        "all": cmd_all,
        "refresh-colab": cmd_refresh_colab,
        "colab-diff": cmd_colab_diff,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
