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
    """Atomic write (see scripts/scan_packages.py::update_req_file). A crash
    between mkstemp and os.replace leaves the prior file intact, so a
    half-downloaded cache file can't poison later runs."""
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

# Oracle files snapshotted from googlecolab/backend-info; colab-diff reports NEW/REMOVED/
# CHANGED, so a base image rotation reaches CI within ~24h.
# The image's Python, from the os-info oracle. Markers only, so an unreadable snapshot just
# replays every requirement.
_COLAB_OS_INFO_FILE = DATA_DIR / "colab_os_info.gpu.txt"
# The prerelease suffix is part of the version: pip skips `python_full_version >= "3.13.0"`
# on 3.13.0rc1, and truncating to 3.13.0 replayed requirements the image never installs.
_COLAB_PYTHON_RE = re.compile(
    r"^Python\s+(\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)",
    re.MULTILINE,
)
# The numeric release at the front of one, which is what `python_version` names.
_PYTHON_RELEASE_RE = re.compile(r"\d+(?:\.\d+)*")


# Follows `lint --colab-pin` so the Python version and the package snapshot come from the
# SAME capture; otherwise markers are judged against a different image's interpreter.
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
    """The environment PEP 508 markers are evaluated against, or None to skip them.

    Only the Colab image, the one environment this can name; anything else replays every
    requirement."""
    if not colab:
        return None
    full = _colab_python_version()
    if not full:
        return None
    release = _PYTHON_RELEASE_RE.match(full).group(0)
    suffix = full[len(release) :]  # `rc1`, `.dev3`; `python_version` never carries one
    parts = release.split(".")
    return {
        "python_version": ".".join(parts[:2]),
        "python_full_version": (release if len(parts) > 2 else f"{release}.0") + suffix,
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
        # A top-level requirement selects no extra, so `extra` is empty and any
        # `extra == "..."` marker is false. Omitting it replayed requirements pip ignores.
        "extra": "",
    }


# `Marker.evaluate` fills any omitted field from the RUNNING process, so a marker naming one
# of these would be judged against this machine and the answer would move between runners.
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
    """False only when the requirement carries a marker that is false for `environment`.

    pip skips such a requirement, so replaying its bounds moves a version the cell never touches.
    Anything unjudgeable (unparseable marker, no `packaging`, no environment) is replayed."""
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
    """Three-valued marker evaluation: True, False, or unknown.

    An unanswerable field makes its own TERM unknown, not the whole marker: a decisive
    `python_version < '3.0' and implementation_name == 'cpython'` stays false on a 3.13 image."""
    text = text.strip()
    if not text:
        return None
    terms, operators = _split_marker(text)
    if len(terms) > 1:
        values = [_marker_truth(term, environment) for term in terms]
        # `and` binds tighter than `or`, so the conjunctions fold first.
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
        return None  # nothing to judge on, or a field the oracle cannot answer for
    from packaging.markers import Marker

    return bool(Marker(term).evaluate(environment))


COLAB_ORACLE_FILES: dict[str, str] = {
    "pip-freeze.gpu.txt": "colab_pip_freeze.gpu.txt",
    "apt-list-gpu.txt": "colab_apt_list.gpu.txt",
    "os-info-gpu.txt": "colab_os_info.gpu.txt",
}
# The pip oracle fails --strict, since the rules resolve against it. os-info is rule-bearing
# too (its Python line), so both refresh together and COLAB_STRICT_ORACLE_KEYS makes that one
# line strict. The rest is advisory: an Ubuntu bump nothing consults must not redden CI.
COLAB_STRICT_ORACLE = "pip-freeze.gpu.txt"
# Keys within a non-strict oracle that are rule-bearing anyway. `python` is the one
# _parse_os_lines emits for the "Python 3.13.15" line.
COLAB_STRICT_ORACLE_KEYS: dict[str, frozenset[str]] = {
    "os-info-gpu.txt": frozenset({"python"}),
}
COLAB_ORACLE_BASE_URL = "https://raw.githubusercontent.com/googlecolab/backend-info/main/"

# ----- Compat tables. PRs add rows as new releases land. ----- #

# Lockstep rows only: torchcodec 0.12+ is ABI-stable against torch >=2.11 and is handled by
# the short-circuit in rule_inst_004_torchcodec_torch rather than by a row here.
TORCHCODEC_ABI_STABLE_TORCH = "2.11"
TORCHCODEC_ABI_STABLE_CODEC = "0.12"

# torch.minor -> set of compatible torchcodec.minor strings.
# Source: pytorch/torchcodec compatibility matrix on its README.
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

# When peft >= trigger is on the resolved set, torchao >= floor must also be.
PEFT_TORCHAO_FLOOR: list[dict[str, str]] = [
    {"trigger_peft": "0.19", "torchao_floor": "0.16.0"},
]

# git+ allowlist: install lines that legitimately fetch from GitHub. Anything
# else flags R-INST-001.
GIT_PLUS_ALLOWLIST = (
    "github.com/SparkAudio/Spark-TTS",
    "github.com/state-spaces/mamba",
    "github.com/Dao-AILab/causal-conv1d",
    "github.com/unslothai/unsloth-zoo",
    "github.com/unslothai/unsloth",
)

# ----- Findings ----- #


@dataclasses.dataclass
class Finding:
    rule: str
    file: str
    cell: int | None = None
    line: int | None = None
    severity: str = "error"  # error | warning
    message: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ----- Notebook walking ----- #


def iter_notebooks(
    notebooks_dir: pathlib.Path, include_templates: bool = False
) -> Iterator[pathlib.Path]:
    """Yield user-facing .ipynb files under nb/ and kaggle/.
    include_templates=True also walks original_template/ (for convert)."""
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


# A shell line that runs pip. Anchored on `!` so a `pip install` inside a Python string is not
# a cell, open after it so chained and compound commands are. `-mpip` counts too: `\b` finds no
# boundary between the `m` and the `p`, so those cells went undiscovered.
_PIP_CELL_RE = re.compile(
    r"^[ \t]*!.*(?:\b(?:uv\s+)?pip|-m(?:uv\s+)?pip)\s+(?:install|uninstall)\b",
    re.MULTILINE,
)


def install_cells(nb: dict[str, Any]) -> list[tuple[int, str]]:
    """Heuristic: any code cell that contains a `pip install`, `pip uninstall`
    or `uv pip install` shell command, or a top-line `%%capture` magic."""
    out = []
    for i, src in code_cells(nb):
        first = src.lstrip().splitlines()[:1]
        if first and first[0].strip().startswith("%%capture"):
            out.append((i, src))
            continue
        # Glued, since a `\\` continuation can put the `!` and the pip call on different
        # physical lines.
        if any(_PIP_CELL_RE.search(line) for _, line in _glue_line_continuations(src)):
            out.append((i, src))
    return out


# Colab oracle only applies to notebooks that run on Colab; AMD, Kaggle,
# DGX-Spark have their own preinstalls and the Colab-vs-cell rules don't apply.
def target_environment(notebook_name: str) -> str:
    parts = pathlib.PurePath(notebook_name).parts
    base = parts[-1] if parts else notebook_name
    parent = parts[-2] if len(parts) >= 2 else ""
    if parent == "kaggle" or base.startswith("Kaggle-"):
        return "kaggle"
    if base.startswith("AMD-") or "_AMD_" in base:
        return "amd"
    if base.startswith("HuggingFace Course-") or base.startswith("HuggingFace_Course-"):
        return "colab"  # HF Course notebooks still run on Colab.
    if "DGX_Spark" in base:
        return "dgx_spark"
    return "colab"


# ----- Pip-freeze parsing ----- #

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


# `1.2.3rc1`, `1.2.3.dev0`, `1.2.3a2`, `1.2.3b1`: PEP 440 orders every one of these BELOW the
# plain `1.2.3` they lead up to.
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
    # The suffix goes before the digits are read, or `2.11.0rc1` compares as 2.11.0.1 and
    # sorts ABOVE 2.11 on the strength of its prerelease number.
    core, prerelease = _split_prerelease(version)
    order = cmp_versions(core, floor)
    if order != 0:
        return order > 0
    return not prerelease


# PEP 440 orders `dev` < `a` < `b` < `rc`, all below the release; `c`, `alpha`, `beta` and
# `pre`/`preview` normalize into that same order.
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
    # PEP 440 zero-pads the shorter release, so `0.11` == `0.11.0`. Raw tuples sorted `0.11`
    # BELOW `0.11.0`, discarding a ceiling-derived minor when the floor spelled its patch.
    width = max(len(ta), len(tb))
    ta = ta + (0,) * (width - len(ta))
    tb = tb + (0,) * (width - len(tb))
    # The suffix decides only a tie on the release core: `0.12.0rc1` sits below `0.12.0` and
    # above `0.11.9`, which reading its digits as another component got backwards.
    ka, kb = ta + _prerelease_key(a), tb + _prerelease_key(b)
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


# ----- Install-cell parsing ----- #


@dataclasses.dataclass
class PipInvocation:
    tool: str  # "pip" | "uv-pip"
    flags: set[str]  # {'--no-deps', '--upgrade', '--force-reinstall', ...}
    packages: list[str]  # raw package specifiers (e.g. 'transformers==5.5.0')
    raw: str
    line_no: int = 0
    action: str = "install"  # "install" | "uninstall"
    conditional: bool = False  # the fallback side of an `||`: runs only if the left failed


# `python -m pip` parses as bare `pip`, or a matched cell yields nothing and R-INST-001 misses
# a `git+` install. In step with unsloth_nb_pip_magic.py::_PY_M_PIP; the braced and path forms
# are matched too, since transformers see the raw text before IPython expands it.
_INTERPRETER_RE = r"""(?:
        (?:python[0-9.]*|py)
      | ["']?\{\s*sys\.executable\s*\}["']?
      | "(?:[^"]*[/\\])python[0-9.]*(?:\.exe)?"
      | '(?:[^']*[/\\])python[0-9.]*(?:\.exe)?'
      | \S*[/\\]python[0-9.]*(?:\.exe)?
    )"""
# `-m uv pip` as well as `-m pip`: unsloth_nb_pip_magic rewrites `(pip|uv)` after the
# module flag, and uv's pip-compatible interface really is spelled `uv pip <action>`.
PIP_LINE_RE = re.compile(
    # `python [option] ... [-m mod ...]`: interpreter options may sit before `-m`, and without
    # them `python -I -m pip install git+...` matched nothing. Options only, or a script path
    # would read as the module flag.
    # `!\s*!` for bash's negation and IPython's `!!` capture: `! ! pip install git+...` still
    # runs pip, and requiring exactly one leading bang matched nothing.
    r"^\s*!(?:\s*!)*\s*(?P<tool>(?:uv\s+)?pip|"
    + _INTERPRETER_RE
    # `-W arg` and `-X opt` take an operand, attached or separate, so requiring every intervening
    # word to start with `-` missed `python -W ignore -m pip ...`. The operand form is tried first.
    # `-h`, `-V` and `-?` print and exit, and `-c cmd` "terminates option list", so nothing behind
    # them is an interpreter option: `python -V -m pip install git+...` only reports the version.
    # The long spellings never matched this arm, which requires a letter after the dash.
    + r"(?:\s+-[WX]\s*\S+|\s+--check-hash-based-pycs\s+\S+|\s+-(?![hVc?])[A-Za-z]\w*)*"
    # `-m mod` may be written attached: `python -mpip install ...` runs pip, and requiring a
    # separate word after `-m` missed it in both this pattern and cell discovery.
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
    # `uv pip` and `python -m uv pip` are both uv; a plain `python3 -m pip` is not, and
    # must not be read as uv because "uv" appears somewhere in the interpreter path.
    tool = "uv-pip" if re.search(r"(?:^|\s)uv\s+pip\b", m.group("tool"), re.IGNORECASE) else "pip"
    rest = m.group("rest")
    # Strip trailing comment.
    rest = re.split(r"(?<!\S)#", rest, maxsplit = 1)[0]
    try:
        tokens = shlex.split(rest, posix = True)
    except ValueError:
        # f-string interpolation like {xformers}: replace braces with placeholders.
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
    """Return (logical_line_no, joined_text) for each logical line, treating
    a trailing backslash as a continuation. Logical line numbers point at the
    first physical line of each logical line."""
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


# Words introducing a compound command. A pip call behind one still runs, so it has to parse;
# `if pip install ...` is the test and is reached whenever the line is, while a `then` or `do`
# body runs only if that test said so.
# Words that run the command after them rather than being it, so `env FOO=1 pip install ...`
# installs as the bare form does. No `builtin`: `builtin pip ...` is not a shell builtin and
# runs nothing. `time` is bash's reserved word; only an explicit path reaches the GNU binary.
_GNU_TIME = "/usr/bin/time"
_SHELL_EXEC_PREFIXES = frozenset({"command", "env", "exec", "nohup", "time", "sudo", _GNU_TIME})
# The subset bash resolves in-process. Everything else here is an external program, and an
# `exec` behind one is an argument to it rather than the builtin.
_SHELL_RESOLVED_PREFIXES = frozenset({"command", "exec", "time"})
# External programs, so an absolute path names the same one. Not `time`: the reserved word and
# `/usr/bin/time` take different options, which `_GNU_TIME` already tells apart.
_PATH_QUALIFIED_PREFIXES = frozenset({"env", "nohup", "sudo"})
# Options that make a prefix report something and exit instead of running its operands.
_PREFIX_TERMINAL_FLAGS = frozenset({"--help", "--version"})
# Per prefix, the options that turn it into a lookup rather than an execution. sudo(8) has
# `-v/--validate` "without running a command" and `-l/--list`, which displays a path; unwrapping
# past either fabricated an install from a line that never reaches pip.
_PREFIX_LOOKUP_FLAGS: dict[str, frozenset[str]] = {
    "command": frozenset({"-v", "-V"}),
    "sudo": frozenset({"-v", "--validate", "-l", "--list", "-V", "-h"}),
}
# `PATH+=:/opt/bin cmd` is an assignment prefix too: bash runs the child with the appended
# value, so leaving the `+=` word standing made it the supposed executable.
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_]\w*\+?=")
# Per prefix, the options taking a SEPARATE operand; everything else starting with `-` is a
# lone flag and `--` ends them. In `env -u pip pip install ...` the first `pip` is the variable
# being unset. env's split-string operand is the command it runs, not a value to discard.
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
            # sudo(8): `-D`/`--chdir`, `-R`/`--chroot`, `-T`/`--command-timeout`. Missing them left the
            # operand standing as the supposed executable.
            "-D",
            "--chdir",
            "-R",
            "--chroot",
            "-T",
            "--command-timeout",
        }
    ),
    "exec": frozenset({"-a"}),
    # Bare `time` is bash's reserved word, `time [-p] pipeline`: it takes none of GNU time's
    # options, so `time -f %e pip install ...` runs a command named `-f`. The GNU binary keeps its
    # own entry, reached only through an explicit path.
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
    depth = 0  # open `$(` nesting
    # Open `${ }` expansions. bash keeps `TOKEN=${TOKEN:-a b}` as ONE assignment word, and
    # tracking only `$(` ended the word at that space, leaving `b}` as the executable.
    brace = 0
    # A `case` arm's pattern ends in an UNBALANCED `)`, so inside an open case that `)` delimits
    # the arm rather than closing the substitution. Popping on it truncated the assignment word in
    # `TOKEN=$(case x in x) printf a;; esac) pip install ...` and made the install look conditional.
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
            # Only an unescaped backtick closes it; the escape above already consumed `\``.
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
            # A substitution's own whitespace and quotes belong to the word: ending it at the space in
            # `TOKEN=$(printf '%s' 'a b') pip install ...` left `'%s'` as the supposed executable.
            if ch in "'\"":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                if not (case_depth and depth == 1):
                    depth -= 1  # otherwise it is an arm pattern and the word continues
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
            # A redirection may sit before the command name: `>/tmp/log pip install ...` runs pip, and
            # stopping here left the redirection standing as the executable.
            prefixed = True
            text = _split_first_word(rest)[1] if _REDIRECTION_RE.fullmatch(word) else rest
            continue
        name = word.lower()
        if name.endswith("/time"):
            name = _GNU_TIME  # an explicit path runs the BINARY, which takes GNU's options
        elif "/" in name and name.rsplit("/", 1)[1] in _PATH_QUALIFIED_PREFIXES:
            # `/usr/bin/env pip install ...` runs pip as the bare form does, and stopping at the path left
            # the install invisible. Only the external programs: `command` and `exec` are builtins, so a
            # path spelling of either names some other file.
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
                rest = tail  # end of options; what follows is the command
                break
            if token == "-" or not token.startswith("-"):
                break
            if name == "env" and token.startswith("-S") and len(token) > 2:
                # `-S, --split-string=S` takes a MANDATORY operand, so the attached `env -S'pip install' pkg`
                # is valid and runs pip; exact membership missed it.
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
                # `env --help pip install ...` prints help and exits, and `command -v pip` only reports a
                # path; unwrapping past either fabricated an install.
                return text, prefixed
            if "=" in token and token.startswith("--"):
                rest = tail  # `--unset=NAME` carries its operand inline
                continue
            if name == "env" and token in _ENV_SPLIT_STRING_FLAGS:
                # GNU env's `-S, --split-string=S` operand IS the command, so consuming it the way `-u NAME`
                # is consumed left nothing to parse. Unquoted, since env splits on unquoted whitespace.
                # `env -S'cmd args' [ARG]...` appends the following ARGs to the split string, which is how
                # `#!/usr/bin/env -S perl -w` reaches `perl -w script.pl`; dropping them lost the packages.
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


# A bare shell word, used to spot `case` / `esac` while scanning a substitution body.
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
                # A `(` opened and closed before this bracket, so the text starts with a substitution rather
                # than an arm label, and reading the trailing `)` as an arm close marked the install
                # conditional.
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
            depth += 1  # a nested plain group inside a substitution
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0 and i == len(text) - 1:
                return True
        i += 1
    return False


# `name() {`, `name () {` and `function name {`. The parens must be EMPTY, so `time (pip
# install x)` is not a definition. A body runs only when called, so it is exposed as
# conditional.
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
    # Bash's negation is a reserved WORD, so `! false` inverts the status while `!false` names a
    # command; collapsing the space made the two identical.
    spaced = bang and stripped[1:2].isspace()
    if bang:
        stripped = stripped[1:].lstrip()
    # A grouping bracket is noise, but the `)` closing a `$( )` belongs to the command: a bare
    # rstrip(")}") left `echo $(pip install ...` unreadable. A lone `}` from a group spanning a
    # separator still strips.
    # `{` opens a group only as its OWN token, so `{ pip install x; }` is a group while IPython's
    # `{sys.executable}` is one word and stripping it hid the pip command. `(` needs no such space.
    # `setup() { pip install git+... ; }` keeps its name in front of the body, hiding the install
    # from PIP_LINE_RE. Strip the header and let the group handling below read the body.
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
    conditional = definition is not None  # the body runs only when the function is called
    while True:
        # Any whitespace, not a literal space: `then\tpip install ...` is the same command to
        # the shell, and leaving `then\tpip` as one word hides it from every rule.
        parts = stripped.split(maxsplit = 1)
        if not parts or parts[0].lower() not in _SHELL_KEYWORDS:
            break
        conditional = conditional or parts[0].lower() in _SHELL_BODY_KEYWORDS
        # A keyword can sit in front of a group: `if (pip install ...); then` exposes the `(`
        # only once `if` comes off, and leaving it there hid the install from PIP_LINE_RE.
        stripped = _open_groups(parts[1].strip()) if len(parts) > 1 else ""
        # And in front of a DEFINITION: `then f(){ pip install ...; }` matched no header with `then`
        # still there. A definition's body is conditional however it was reached.
        behind = _FUNCTION_DEF_RE.match(stripped)
        if behind is not None:
            definition = behind
            conditional = True
            stripped = _open_groups(stripped[behind.end() :].lstrip())
    # A `case` arm label, quoted or bare. Only the matching arm runs, so the command is
    # conditional. The label ends at the first unquoted `)` with nothing open before it.
    close = _unquoted_arm_close(stripped)
    if close is not None:
        stripped = stripped[close + 1 :].strip()
        conditional = True
    # `env -u VAR pip install ...`, `nohup pip ...`, `A=1 pip ...`: each prefix's options and
    # operands are consumed positionally, so the executable is whatever word survives.
    stripped, _prefixed = _strip_exec_prefixes(stripped)
    if not (bang and stripped):
        return stripped, conditional
    return (f"! {stripped}" if spaced else f"!{stripped}"), conditional


# `${name:-word}` and friends expand the word only on the branch the parameter's state
# selects, so a substitution inside one is conditional. `${name}` opens no branch.
_CONDITIONAL_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_]\w*(?:\[[^\]]*\])?:?[-+=?]")


def _conditional_expansion_spans(command: str) -> list[tuple[int, int]]:
    """Half-open ranges covering the word of every branching `${ }` expansion."""
    spans: list[tuple[int, int]] = []
    for match in _CONDITIONAL_EXPANSION_RE.finditer(command):
        depth, j = 1, match.end()
        quote = ""
        while j < len(command) and depth:
            ch = command[j]
            # `\}` and `'}'` are literal text in the default word, not the closer; counting them ended the
            # span early and read a later `$( )` as unconditional.
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
            i += 2  # escaped: `\\$(` is a literal dollar
            continue
        if quote == "'":
            if ch == "'":
                quote = ""  # single quotes make the text literal, so nothing runs in there
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
            # Inside an open case the `)` in `$(case x in x) pip install ...;; esac)` delimits the arm
            # rather than closing the substitution, and popping on it ended the body at `x)`.
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
                        depth -= 1  # otherwise it is an arm pattern; the body stays open
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
            # The first UNESCAPED backtick closes it: a legacy nested substitution writes its inner
            # delimiters `\\`` so they do not close the outer one, and `find` stopped at the escape.
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
            # Unescape before recording it: inside backticks the shell strips one level, and without that
            # the nested substitution never opens.
            if in_branch(i) == conditional:
                bodies.append(command[i + 1 : j].replace("\\`", "`").replace("\\\\", "\\"))
            i = j + 1
        else:
            i += 1
    return [body.strip() for body in bodies if body.strip()]


def _piece_is_pip(piece: str) -> bool:
    """Is this chunk of a chained line a pip command? `!` only ever leads the first piece,
    and the splitter re-adds it to the rest, so it is normalised before asking."""
    # The bang first: `_strip_exec_prefixes` reads words, and `!env` glued together matched no
    # prefix name, so `!env X=1 pip install ...` did not read as pip.
    stripped = _strip_exec_prefixes(piece.strip().lstrip("!").strip())[0].strip()
    return bool(stripped) and bool(PIP_LINE_RE.match("!" + stripped))


# A redirection operator, optionally with its fd and its target attached (`2>&1`, `>/x`).
_REDIRECTION_RE = re.compile(r"^\d*(?:>>|>&|&>|<<<|<<|<>|>|<)")
# `exec`'s own options. `-a` names the argv[0] to pass on and takes an operand.
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
    # `env exec true` asks env for a PROGRAM called exec, which does not exist, so the parent
    # shell carries on. Only the prefixes bash resolves in-process keep the builtin's meaning.
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
            # `> /x` carries its target as the next word; `>/x` already has it.
            rest = _split_first_word(tail)[1] if _REDIRECTION_RE.fullmatch(word) else tail
            continue
        return True  # a utility to hand the shell over to
    return False


def _command_ends_shell(command: str) -> bool:
    """Does this command end the shell, so that nothing after it in the list can run?

    `exec NAME` replaces it and `exit` terminates it; recognising only the first reported `exit 0;
    pip install git+...` as a reachable install."""
    # `{ exit; ... }` is a brace group: it runs in the SAME shell, so a terminator inside it
    # ends the line. `( exit )` is a subshell and does not, which is why only `{` is stripped.
    opened = command.lstrip("!").strip()
    while True:
        if opened.startswith("{") and (len(opened) == 1 or opened[1].isspace()):
            opened = opened[1:].lstrip()
            continue
        # `then exit` is still an exit. The caller weighs the branch condition separately, so
        # only a body that is actually taken hands anything over.
        word, rest = _split_first_word(opened)
        if word.lower() in _SHELL_BODY_KEYWORDS:
            opened = rest.lstrip()
            continue
        break
    if _command_execs(opened):
        return True
    seen: list[str] = []
    rest = _strip_exec_prefixes(opened, seen)[0]
    # Same rule as `exec`: `env exit 0` asks env for a PROGRAM called exit, which does not exist.
    # Only the prefixes bash resolves in-process keep the builtin.
    if any(name not in _SHELL_RESOLVED_PREFIXES for name in seen):
        return False
    return _split_first_word(rest)[0] == "exit"


# `true` and `:` are documented as always succeeding, so an `&&` after one is always reached.
# Treating every non-pip command as a possibly-failing probe dropped the install behind them.
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
    text = _unwrap_shell_group(piece)[0]  # `( ... )` exits with its last command's status
    # Only the FIRST command of a cell carries the notebook's bang, `!cmd` or `! cmd`. Elsewhere a
    # leading `!` is bash's negation, which the loop below counts.
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
        # `help return`: "exit with the return value specified by N", so `setup() { return 0; };
        # setup && pip install ...` always installs. A BARE `return` carries the previous command's
        # status, which nothing here names.
        model = _split_first_word(stripped)[1].strip() == "0"
    elif functions is not None and word in functions:
        # A call exits with its body's status, so `f() { pip install x; }; f && ...` reaches
        # the tail under the same pip-succeeds model a bare `pip install x &&` rests on.
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
        # Fold it through the group's own list rather than replacing the status: in
        # `(false && pip install x)` the trailing command was short-circuited.
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
    # Three-valued too, for the `||` reachability fold: `{ false; } || pip install x` always
    # reaches the install, and discarding the group's KNOWN failure marked it conditional.
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
    # `negations` is the `!` in front of the PIPELINE this piece ends, which turns its status
    # around: `! true | true` fails, and folding the raw `true` carried a list bash does not.
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
            return False if right is False else None  # `? && false` fails either way
        return right if left else False  # a failed left skips right and keeps the failure
    if left is None:
        return True if right is True else None  # `? || true` succeeds either way
    return True if left else right  # a succeeded left skips right and keeps the success


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
        # A group just closed and the text in hand is its bare bracket. The level ALREADY
        # carries the group's status; folding the bracket as an unknown command wiped it.
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
    # Shell whitespace, not a literal `" in "`: `for x\tin\ta` is the same loop to bash,
    # and finding no list there marked a body that certainly runs as conditional.
    match = re.search(r"\sin\s", text)
    if match is None:
        return False
    words = text[match.end() :].split()
    if not words or not any(words):
        return False
    # Quoting decides what a metacharacter means: `for x in '*'` iterates over one literal star,
    # while a bare `*` is a glob that may match nothing.
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
        # `A=1 f`, `2>/dev/null f` and the reserved word `time f` all still call the function;
        # only the wrappers that go looking for an EXECUTABLE do not.
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
    # `setup() { if true; then ...` opens a compound INSIDE a definition, and reading the header
    # as the first word hid the `if`, losing the body's known outcome.
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
    # One flag per open group, plus the base list. A command is conditional when any level
    # above it is in a fallback tail, so an inner list cannot clear an outer one.
    tails = [False]
    # Per open `(`/`{`: does it hold a FUNCTION BODY? Unlike `tails` this survives a separator
    # inside the body, since `;` starts a new and-or list but does not leave the definition.
    def_levels = [False]
    # The function each open `{` defines, and per flushed piece the innermost one it sits in. A
    # body is conditional until its function is CALLED, by a later command in the same line, so the
    # ownership has to survive to the second pass.
    def_names: list[str | None] = [None]
    owners: list[str | None] = []
    nodef: list[bool] = []
    assumed: list[bool] = []
    # Each definition's exit status once its closing brace is reached. Bash requires the
    # definition to precede the call, so a single left-to-right pass always has it in hand.
    func_status: dict[str, bool | None] = {}
    # Definition keys per name, in the order they appear.
    instances: dict[str, list[str]] = {}
    definitions = 0
    # Per level: whether the last command flushed there is modelled as succeeding. A group
    # exits with that status, which is what the enclosing `&&` reads.
    last_ok: list[bool | None] = [None]
    # Per level: has this and-or list already run a pip command? `A && B` leaves B
    # unconditional only when something to its left is one.
    list_has_pip = [False]
    # Per level: the operator that joined the piece in hand to the list before it. `||` succeeds
    # when either side did and `&&` only when both, which `list_has_pip` alone cannot recover.
    prev_ops = [""]
    # Per level: the folded exit status of the and-or list to the LEFT of the piece in hand,
    # three-valued because only a CERTAIN failure makes a `||` tail unconditional.
    list_models: list[bool | None] = [None]
    buf_conditional = False
    # One entry per open `(`/`{`: True when it opened a grouping. A `)` closing a `$( )` is
    # inside a word, so a `#` after it is a literal, not a comment.
    groupings: list[bool] = []
    # Per level: how many `case` statements are open. An arm's pattern ends in an UNBALANCED `)`,
    # so while one is open that bracket delimits the arm rather than closing the level, and popping
    # on it read an unconditional pip call as conditional.
    case_depths: list[int] = [0]
    grouping_closed = False
    # A `( )` or `{ }` closed and nothing has been flushed since: the level's status already
    # carries it, so folding the text in hand again wiped what the group contributed.
    closed_pending = False
    # `!` in front of a pipeline belongs to the whole pipeline, so it has to outlive the pipe
    # that ended its first command.
    pipe_negations = 0
    in_pipeline = False
    # Inside the empty parens of a function header, whose brackets open no group.
    func_parens = False
    # Per level: was this tail made unconditional by the pip-succeeds assumption? Reporting an
    # install on that basis is intended; cutting a path on it is not.
    assumed_tail = [False]
    # An open legacy `` `...` `` substitution: its operators belong to the inner command, so
    # without this the `;` inside one split the line into an unreadable fragment.
    in_backtick = False
    i = 0

    def in_sub() -> bool:
        return in_backtick or not all(groupings)

    # The operator that ENDED each piece. `exec` under `|` or `&` runs in a subshell, so the
    # parent shell reaches the next command and the list must not be truncated there.
    seps: list[str] = []

    def flush(separator: str = "") -> None:
        nonlocal buf, closed_pending
        text = "".join(buf)
        last_ok[-1] = _piece_success_model(text, func_status, not out)
        out.append((text, buf_conditional))
        seps.append(separator)
        owners.append(next((name for name in reversed(def_names) if name), None))
        # The same flag WITHOUT the definition. Calling a function makes its body reachable,
        # not unguarded: `setup() { false && pip install x; }; setup` still runs no pip.
        nodef.append(any(tails))
        assumed.append(any(assumed_tail))
        buf = []
        closed_pending = False

    while i < len(line):
        ch = line[i]
        in_substitution = in_sub()
        if not quote and ch.isalpha() and not (buf and buf[-1].isalnum()):
            # Tracked ahead of the dispatch below: the `in_substitution` branch swallows a substitution's
            # characters whole, so a `case` opened in one would never be seen there.
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
            # Backticks expand inside double quotes as well, so this is checked before the
            # quote branch; only single quotes make them literal.
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
            # Close it here, before the guard below would swallow the bracket.
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
                        # A group exits with its last list's status, which `_close_group` is
                        # about to fold; record it here, before the pop loses it.
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
            break  # an operator, or a bracket that closed a grouping, ends a word
        elif in_substitution:
            buf.append(ch)  # its separators are its own; the body is split on its own later
            i += 1
        elif line.startswith("||", i):
            # `false || pip install ...` always reaches the fallback, so only an UNKNOWN left side opens a
            # tail, and the left side is the whole list: `true || false || pip install ...` skips it.
            left_model = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            # The pipeline this operator closes exits with its last command's status, turned
            # around by the `!` in front of the whole thing.
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
            # `A && B` runs B only when A succeeded, so B is conditional unless the list to its left
            # contains a pip command, which the replay models as succeeding. The whole list, not the last
            # piece: the left operand of `A || B && C` is `(A || B)`, which succeeds when either ran.
            #
            # The exception is the point: `pip install a && pip install b` is the ordinary idiom and
            # dropping its second half would cost more coverage than it saves. What this fixes is a probe
            # guard, `nvidia-smi && pip install torch==2.12.0`, which installs nothing on a CPU box.
            left_and = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            # The pipeline this operator closes exits with its last command's status, turned
            # around by the `!` in front of the whole thing.
            left_and = _negated(left_and, pipe_negations)
            if pipe_negations % 2:
                list_models[-1] = left_and
            _fold_pending(
                list_has_pip, prev_ops, "".join(buf), not out, closed_pending, pipe_negations
            )
            pipe_negations, in_pipeline = 0, False
            prev_ops[-1] = "&&"
            # Reaching this tail rests on the replay's own model of pip succeeding, which is
            # fine for reporting an install but must not make anything UNREACHABLE.
            assumed_tail[-1] = assumed_tail[-1] or _piece_assumes_pip("".join(buf))
            flush("&&")
            # A left side modelled as CERTAIN success reaches the tail as surely as the pip
            # idiom does: `f() { pip install x; }; f && ...` and `true && ...` both run it.
            tails[-1] = not (list_has_pip[-1] or left_and is True)
            buf_conditional = any(tails) or any(def_levels)
            i += 2
        elif (
            ch == ";"
            or (
                # `A & B` backgrounds A and runs B, `A | B` runs both: unconditional either way.
                ch in "&|"
                # `>&`, `<&`, `&>` and `>|` are redirections rather than separators: `0<&1 pip install ...` is
                # one command, and splitting on its `&` left `1 pip install ...`, which reads as no pip.
                and not (
                    ch == "&" and (line[i - 1 : i] in ("<", ">") or line[i + 1 : i + 2] == ">")
                )
                and not (ch == "|" and line[i - 1 : i] == ">")
            )
        ):
            # A separator ends the and-or LIST, and a group exits with that list's status rather than its
            # last lexical command's: `{ false && pip install x; }` fails, because the install never ran.
            folded = _left_hand_status(
                list_models, prev_ops, "".join(buf), func_status, not out, closed_pending
            )
            if ch == "|":
                if not in_pipeline:
                    # Only the head carries it: `a | ! b` is a syntax error, so no later
                    # segment can introduce one.
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
            prev_ops[-1] = ""  # a new and-or list starts here
            buf_conditional = any(tails) or any(def_levels)
            i += 1
        else:
            # `f()` is a function header, not a group: pushing a level for its empty parens marked the
            # whole body as "a group just closed" and lost the definition's status. Ordinary characters.
            if (
                ch == "("
                and _FUNCTION_NAME_RE.fullmatch(_behind_keywords("".join(buf)))
                and line[i + 1 :].lstrip().startswith(")")
            ):
                func_parens = True
            if ch == ")" and func_parens:
                func_parens = False  # the header's own bracket, matching the skip above
            elif ch in "({":
                # `$(`, `<(`, `>(` open a substitution running its own commands; a bare `(` groups this
                # line's. `${ }` expands a WORD and runs nothing, so splitting on the `||` in
                # `${X:-a||pip install ...}` invents a command bash never runs.
                groupings.append(
                    not (
                        (ch == "(" and buf and buf[-1] in "$<>")
                        or (ch == "{" and buf and buf[-1] == "$")
                    )
                )
                # `setup() { :; pip install ...; }` defines a function nobody called, and the header sits in
                # the piece that opens the brace, so flagging that piece alone freed every later command.
                tails.append(False)
                assumed_tail.append(False)
                header = _FUNCTION_DEF_RE.fullmatch(_behind_keywords("".join(buf)))
                def_levels.append(ch == "{" and header is not None)
                if ch == "{" and header:
                    # One key per DEFINITION, not per name: `f(){ a; }; f; f(){ b; }` calls the FIRST body, and
                    # keying by name compared the call against the last definition of `f`.
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
                    buf_conditional = any(tails) or any(
                        def_levels
                    )  # the group opens before the command
            elif ch in ")}" and not (ch == ")" and case_depths[-1]):
                grouping_closed = groupings.pop() if groupings else True
                if len(case_depths) > 1:
                    case_depths.pop()
                if len(tails) > 1:
                    # The command in hand belongs to the level being closed, so its flag stays
                    # what it was; the pop only affects what comes after.
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
    # One entry per piece in `out`, empties included: dropping them before the zip slid every later
    # pair by one and left a `case`'s last arms unscanned. Filtered after the pairing.
    commands = [(head_text, head_conditional or head_keyword)]
    # The keyword alone, without the piece's own flag folded in: entering a called function removes
    # the definition from that flag, and the combined one double-counted it.
    kw_flags = [head_keyword]
    for piece, flag in rest:
        text, keyword = _unwrap_shell_group(piece.strip())
        # A space when the command itself starts with bash's negation: glued to the notebook bang,
        # `! false` read as a command named `!false` and the negation was lost.
        commands.append(
            (f"!{' ' if text.startswith('!') else ''}{text}" if text else "", flag or keyword)
        )
        kw_flags.append(keyword)
    # `echo $(pip install x)` runs the install while the outer command is not pip, so the inner one
    # is a command of its own. Read off the raw pieces: the unwrap above strips an assignment
    # prefix like ``X=`pip install y` ``.
    ordered: list[tuple[str, bool]] = []
    # An unconditional `exec` or `exit` ends the shell, so every OUTER command after it is
    # unreachable. Its own substitutions expanded first, and one inside a `$( )` replaces only that
    # subshell, so this applies at this level alone.
    handed_over = False
    seps = seps + [""] * (len(out) - len(seps))
    # One flag per open compound statement: True once its BODY has started. `if false; then echo x;
    # pip install ...; fi` runs neither command, but only the piece carrying the `then` was flagged.
    body_levels: list[bool] = []
    # Per open compound: what its test is modelled as returning, or None when unknown. A body
    # whose test can never succeed is unreachable rather than conditional.
    test_models: list[bool | None] = []
    # Per open compound, over the arms so far: did every one certainly fail, and was every one
    # KNOWN? Together they decide an `elif` test and an `else` branch, which neither the arm in hand
    # nor a plain inversion can tell.
    arms_failed: list[bool] = []
    arms_known: list[bool] = []
    # Per open compound: the word that opened it, whether the arm in hand is even reached, and
    # whether the condition folded so far leans on the pip-succeeds assumption.
    openers: list[str] = []
    arm_reached: list[bool] = []
    cond_assumed: list[bool] = []
    # The same condition folded with pip's status left UNKNOWN. Comparing the two says whether the
    # pip-succeeds assumption decided it, which a sticky flag could not.
    cond_models: list[bool | None] = []
    # Where each function's body landed in `ordered`, and the names invoked unconditionally.
    body_entries: dict[str, list[tuple[int, bool]]] = {}
    # Per function body, the names it invokes unconditionally WITHIN that body. Reached only
    # once the body itself is, which is what makes the call graph transitive.
    body_invokes: dict[str, set[tuple[str, int]]] = {}
    called: set[tuple[str, int]] = set()
    maybe_called: set[tuple[str, int]] = set()
    # Depth of open compounds at an unconditional `break`/`continue`. Bash jumps past `done`, so
    # the rest of that body never runs; loop-local, unlike `exit`, which ends the shell.
    broke_at: int | None = None
    # Functions whose body has hit an unconditional `return`, and the last piece index each
    # definition occupies. `return` ends the BODY, not the shell, and a call resolves against the
    # definition in force at that point: bash fails `f` written before `f()`.
    returned: set[str] = set()
    def_last_index: dict[str, int] = {}
    # Functions whose body ends the SHELL, and where each name was first called. The terminator is
    # conditional while it is only a definition, so the effect applies when the call is resolved.
    ends_shell: set[str] = set()
    call_at: dict[str, int] = {}
    for index, ((piece, flag), (text, command_flag), separator) in enumerate(
        zip(out, commands, seps)
    ):
        if handed_over:
            break
        keywords = _leading_shell_keywords(piece)
        # The SELECTOR of a `case` runs before any arm is chosen, so a substitution in it is
        # unconditional even though the arms it opens are not.
        opens_case = "case" in keywords
        for keyword in keywords:
            if keyword == "case":
                # Every command between here and its `esac` sits in some arm and only the matching arm runs.
                # No body keyword opens one, so the level is active from the word itself.
                body_levels.append(True)
                test_models.append(None)
                arms_failed.append(False)
                arms_known.append(False)
                openers.append("case")
                arm_reached.append(False)
                cond_assumed.append(False)
                cond_models.append(None)
            elif keyword in _SHELL_TEST_KEYWORDS:
                body_levels.append(False)  # the test itself runs whenever the line does
                test_models.append(None)
                arms_failed.append(True)  # no arm has run yet
                arms_known.append(True)
                openers.append(keyword)
                arm_reached.append(True)
                cond_assumed.append(False)
                cond_models.append(None)
            elif keyword in _SHELL_BODY_KEYWORDS:
                if body_levels:
                    body_levels[-1] = True
                    if keyword in ("then", "do"):
                        # The condition is complete: fold it in, invert an `until`, and let
                        # the arm bookkeeping see the RESULT rather than the first piece.
                        model = test_models[-1]
                        if openers[-1] == "until" and model is not None:
                            model = not model
                        if not arm_reached[-1]:
                            model = None
                        elif model is False and cond_assumed[-1]:
                            # The condition is false only because the replay assumes pip succeeds, and
                            # `if ! pip install x; then ...` does run its body when that install fails.
                            model = None
                        if openers[-1] in ("if", "until", "while"):
                            arms_known[-1] = (
                                arms_known[-1] and model is not None and not cond_assumed[-1]
                            )
                            arms_failed[-1] = arms_failed[-1] and model is False
                        test_models[-1] = model
                        cond_models[-1] = model
                    elif keyword == "else":
                        # `else` runs exactly when every arm failed. Inverting the arm in hand
                        # answered that only for a bare `if`/`else`.
                        test_models[-1] = (
                            True if arms_failed[-1] else (False if arms_known[-1] else None)
                        )
                        cond_models[-1] = test_models[-1]
                    elif keyword == "elif":
                        # Its test is reached only when every earlier arm failed, and while it
                        # is being read the level is back in a test region.
                        arm_reached[-1] = arms_failed[-1]
                        body_levels[-1] = False
                        openers[-1] = "if"
                        cond_assumed[-1] = False
                        test_models[-1] = True if arms_failed[-1] else None
                        cond_models[-1] = test_models[-1]
                else:
                    # A body word with no open compound above it: every stack grows together, or the matching
                    # `fi` pops one that was never pushed and the lint run dies on an IndexError.
                    body_levels.append(True)
                    test_models.append(None)
                    arms_failed.append(False)
                    arms_known.append(False)
                    openers.append("")
                    arm_reached.append(False)
                    cond_assumed.append(False)
                    cond_models.append(None)
            elif body_levels:
                body_levels.pop()  # fi / done / esac
                test_models.pop()
                arms_failed.pop()
                arms_known.pop()
                openers.pop()
                arm_reached.pop()
                cond_assumed.pop()
                cond_models.pop()
        # `flag` alone is the separator-level state: a substitution inside a compound body or a case
        # arm is expanded only when that body runs.
        # A level speaks only once its BODY has started, and says the outcome of the branch in hand:
        # `if true` certainly runs, `if false` never does, anything else is a path the notebook may
        # take. A false branch still has to be inverted for `else`, which does run.
        active = [model for level, model in zip(body_levels, test_models) if level]
        if any(model is False for model in active):
            continue  # this branch can never be taken, so nothing in it runs
        if broke_at is not None:
            if len(body_levels) < broke_at:
                broke_at = None  # the loop closed; what follows `done` runs again
            else:
                continue  # still inside the loop the `break` jumped out of
        # `command_flag` is the `then`/`else`/arm-label the piece carries, which means "conditional"
        # only because the branch usually is; when the branch is KNOWN to be taken it says nothing. A
        # case arm always leaves a None in `active`, so this can never clear an arm label.
        # An `elif` whose earlier arms all certainly failed is a TEST, and a test runs whenever the
        # statement does.
        reached_test = bool(body_levels) and not body_levels[-1] and arm_reached[-1]
        certain_branch = reached_test or (bool(active) and all(model is True for model in active))
        piece_conditional = (
            flag
            or (command_flag and not certain_branch)
            or any(model is not True for model in active)
        )
        # A `case` selector sits in the same piece as the first arm, so the arm body keeps the
        # level this piece opened while the substitutions ahead of it do not.
        selector = (
            zip(body_levels[:-1], test_models[:-1]) if opens_case else zip(body_levels, test_models)
        )
        sub_conditional = (
            flag or command_flag or any(model is not True for level, model in selector if level)
        )
        for inner in _substitution_bodies(piece):
            for inner_text, inner_flag in _split_chained(f"!{inner}"):
                # A substitution inherits the parent's functions, so `f(){ pip install ...; }; echo $(f)`
                # calls f. The recursive parse cannot see the definition, so the CALL is recorded here, where
                # the reachability walk resolves it. One the notebook may not expand reaches the body
                # without making anything in it certain.
                if sub_conditional or inner_flag:
                    maybe_called.add((_invoked_name(inner_text), index))
                else:
                    called.add((_invoked_name(inner_text), index))
                ordered.append((inner_text, sub_conditional or inner_flag))
        # `${READY:-$(pip install ...)}` expands its word only when READY is unset, so the
        # install inside it is a path the notebook MAY take, never one it certainly does.
        for inner in _substitution_bodies(piece, conditional = True):
            for inner_text, _ in _split_chained(f"!{inner}"):
                maybe_called.add((_invoked_name(inner_text), index))
                ordered.append((inner_text, True))
        if text:
            # `if false` / `while false` / `until true` never reach their body, so what follows is
            # unreachable rather than conditional and an install from it is not a finding.
            # Still reading a condition: fold this piece into it, or `if false || true; then ...` stores
            # the `false` alone. The inversion and arm bookkeeping happen when `then`/`do` closes it.
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
                # Only while the assumption still DECIDES the condition: `if false && pip install x` fails
                # whatever pip does, and a sticky flag marked an `else` bash always runs conditional. Folding
                # the condition again with pip unknown answers it, the two differing exactly when the
                # assumption is load-bearing, which `if ! pip install x` still is.
                unassumed = None if _piece_assumes_pip(text) else model
                cond_models[-1] = (
                    unassumed
                    if joiner not in ("&&", "||")
                    else _fold_status(cond_models[-1], joiner, unassumed)
                )
                cond_assumed[-1] = cond_models[-1] is not test_models[-1]
            # A bare `setup` invokes it. Only the FIRST word: `setup --dry-run` calls it, `echo setup`
            # does not.
            # The RAW first word: `env f`, `nohup f` and `command f` look for an executable named f, so
            # none reaches a shell function, and entering the body let its `exit` truncate the line.
            invoked = _invoked_name(piece)
            # What this command's flag would be with the definition entered; every other reason it is
            # conditional still stands.
            # `command_flag` on the HEADER piece is the definition itself, which entering the function
            # removes; on any other piece it is a real `then` or arm label.
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
                # Keyed by DEFINITION, not by name: `f(){ ...; }; f; f(){ :; }` calls the first body, which
                # comparing against the final definition left conditional.
                def_last_index[owner] = index
                if owner in returned:
                    continue  # the body already returned; nothing after it in this function runs
                body_entries.setdefault(owner, []).append((len(ordered), entered))
                if not entered:
                    body_invokes.setdefault(owner, set()).add((invoked, index))
                    if invoked == "return":
                        returned.add(owner)
                    elif (
                        not assumed[index]
                        and separator not in ("|", "&")
                        and _command_ends_shell(
                            # The header shares this piece and has to come off before the terminator behind it is
                            # visible. Still the RAW body: the unwrap producing `text` strips `exec` with it.
                            piece[header_span:] if header_piece else piece
                        )
                    ):
                        ends_shell.add(owner)
            elif not piece_conditional:
                called.add((invoked, index))
                if separator not in ("|", "&"):
                    # `f | cat` runs f in a subshell, so a terminator inside it ends only
                    # that subshell and the parent shell reaches the next command.
                    call_at.setdefault(invoked, len(ordered))
            ordered.append((text, piece_conditional))
            # The RAW piece: `_unwrap_shell_group` strips `exec` out of `text` with every other
            # transparent prefix. An `exec` bash may never reach hands nothing over, so the body condition
            # counts here as much as the separator one.
            handed_over = (
                not piece_conditional
                and not assumed[index]  # only certainly-reached terminators cut the list
                and separator not in ("|", "&")  # a subshell; the parent shell carries on
                and _command_ends_shell(piece)
            )
            if (
                not piece_conditional
                and body_levels
                and _split_first_word(_strip_exec_prefixes(text.lstrip("!").strip())[0].strip())[0]
                in ("break", "continue")
            ):
                # It jumps out of the innermost LOOP, not always the compound it sits in: in `while ...; do
                # if ...; then break; fi; ...; done` the `fi` closes long before the body it skipped ends.
                # Only inside a loop: bash reports "break: only meaningful in a `for', `while' or `until'
                # loop" and carries on, so binding the jump to an enclosing `if` dropped commands that run.
                loop = [n for n, word in enumerate(openers) if word in ("while", "until", "for")]
                if loop:
                    # `break n` jumps out of the n-th enclosing loop, so `break 2` in a nested body leaves both
                    # and the outer body stops too. A count past the nesting leaves every loop, as bash does.
                    _, _, level = (
                        _strip_exec_prefixes(text.lstrip("!").strip())[0].strip().partition(" ")
                    )
                    depth = int(level.strip()) if level.strip().isdigit() else 1
                    broke_at = loop[max(len(loop) - depth, 0)] + 1

    # A defined body is conditional until something calls it: `setup() { pip install x; }; setup`
    # definitely installs, and leaving it conditional dropped the pairing from the replay.
    # `outer() { inner; }; outer` reaches `inner` only after `outer` is replayed, so the call graph
    # is walked to a fixed point.
    # A call only reaches a definition that already exists: `f || true; f() { ... }` fails at the
    # call, so the name alone is not enough.
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
    # A call the notebook MAY make, inside `${READY:-$(f)}` say, reaches the body without making
    # anything in it certain: leaving it out pruned a body bash can run, and `called` would have
    # replayed it as unconditional.
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
    # A body nobody calls is UNREACHABLE, not merely conditional: bash defines `f` and stops, and
    # the all-path rules read conditional commands, so leaving it in reported a phantom source.
    unreached: set[int] = set()
    for name, entries in body_entries.items():
        for position, entered in entries:
            if name in soft:
                ordered[position] = (ordered[position][0], True)
            elif name in reached:
                ordered[position] = (ordered[position][0], entered)
            else:
                unreached.add(position)
    # The call itself hands the shell over, so nothing the caller writes after it can run.
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


# Spec parsing: only what we need (no full PEP 440).
SPEC_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]*\])?(?P<rest>.*)$")
OP_VERSION_RE = re.compile(r"(==|>=|<=|!=|~=|>|<)\s*([0-9][^,;\s]*)")


@dataclasses.dataclass
class SpecParts:
    name: str
    pins: list[tuple[str, str]]  # list of (op, version)
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


# ----- PyPI metadata cache ----- #


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
    """Return (raw_specifier_string_or_None, list_of_(op,version) tuples)
    for the constraint that `name==version` places on `target`.
    """
    md = pypi_metadata(name, version)
    if not md:
        return None, []
    info = md.get("info", {}) or {}
    requires = info.get("requires_dist") or []
    target_l = target.lower()
    for req in requires:
        # Examples: 'tokenizers (<=0.23.0,>=0.22.0)', 'tokenizers <=0.23.0,>=0.22.0',
        # 'tokenizers (>=0.22.0,<=0.23.0); python_version >= "3.9"'
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


# ----- Resolved set ----- #


def resolved_set(install_cell: str, colab: dict[str, str]) -> dict[str, str]:
    """Merge install-cell constraints with Colab pip-freeze (cell wins).

    Resolution order per package: (1) exact `==V` pin, (2) upper-bound `<=V`
    (pip picks the highest allowed = V), (3) Colab fallback. Lower-bound `>=V`
    is intentionally NOT reflected (it doesn't lower an already-higher Colab
    version); R-INST-003 models that via `_install_cell_lower_bound`.
    """
    out = dict(colab)
    pinned: set[str] = set()
    upper_bounds: dict[str, str] = {}
    environment = _marker_environment(colab)
    for inv in unconditional_pip_invocations(install_cell):
        if _is_dry_run(inv):
            continue
        if inv.action == "uninstall":
            # The cell removed it, so the environment it leaves behind has no such package and neither
            # should this: ignoring the verb left the rules judging something `pip uninstall` had just
            # deleted. The accumulated bound goes too, or a later reinstall inherits it.
            for raw in inv.packages:
                sp = parse_spec(raw)
                if sp is None:
                    continue
                # PEP 503 makes `huggingface_hub` and `huggingface-hub` one project and the snapshot is keyed
                # the second way, so popping the spelling as written left the removed package in place.
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
    # Apply upper bounds where Colab's preinstall violates them.
    for name, ub in upper_bounds.items():
        if name in pinned:
            continue
        existing = out.get(name)
        if existing is None or cmp_versions(existing, ub) > 0:
            out[name] = ub
    return out


# ----- Rules ----- #


# A `git+` target runs to the next shell or quoting boundary. Case-insensitive: pip
# normalises `Git+https://` to the same link.
_GIT_SOURCE_RE = re.compile(r"""git\+[^\s'"]+""", re.IGNORECASE)


def _git_source_repository(source: str) -> str:
    """`git+https://user@github.com/Org/Repo.git@ref` -> `github.com/org/repo`.

    Matched as a path, not a substring: an arbitrary repository can carry
    `github.com/unslothai/unsloth` inside its own path, which a substring test reads as permission."""
    # The raw scan keeps a substitution's closing bracket, and `unsloth.git)` matched no allowlist
    # entry, so a permitted install was reported. Quotes and brackets are shell syntax, never part
    # of a repository path.
    source = source.strip().rstrip(")}`\"'")
    remainder = source.split("+", 1)[1] if "+" in source else source
    # pip normalises the scheme, so the comparison is on the lowered host and path below.
    remainder = remainder.split("://", 1)[-1]
    host, _, path = remainder.partition("/")
    host = host.rsplit("@", 1)[-1]  # drop any credentials
    path = path.split("#", 1)[0].split("?", 1)[0]
    # The LAST `@` after the repo path is the revision delimiter (pip VCS docs), so splitting at
    # the first read `unslothai/unsloth@fake/../../attacker/repo@main` as the allowlisted repo.
    path = path.rsplit("@", 1)[0].rstrip("/")
    # `unsloth.GIT` is the same repository: the host and path are lowered further down, so a
    # case-sensitive strip left `.GIT` on and the entry then matched nothing.
    if path.lower().endswith(".git"):
        path = path[: -len(".git")]
    # Resolve `.` and `..` as a URL client does, or `unslothai/unsloth/../../attacker/repo`
    # reads as an allowlisted prefix.
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
        # Per source, not per line: one allowlisted repository beside a prohibited one must
        # not clear the whole line.
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
                continue  # `--dry-run` reports what pip WOULD do and changes nothing
            if inv.action == "install" and not _requirement_applies(raw, environment):
                continue  # pip skips a requirement its marker excludes, so nothing is put back
            # Replayed in order: `pip uninstall x; pip install x` leaves x installed, and answering on the
            # first uninstall claimed a removal pip puts straight back.
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
            # Check transitive constraints on a curated short list of pkgs.
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
    """Return the highest lower bound any install line places on `target`
    (treating `==V` as both bounds), or None. Used by R-INST-003 so a
    `torchao>=0.16.0` line satisfies the floor without a `==` pin."""
    best: str | None = None
    for inv in unconditional_pip_invocations(install_cell):
        if _is_dry_run(inv):
            continue  # pip makes no environment changes, so it places no floor on anything
        if inv.action == "uninstall":
            # The cell removed it, so no earlier line still places a floor on it. Keeping the
            # bound let R-INST-003 accept an environment the package is no longer in.
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
                continue  # pip skips it, so it satisfies no floor
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


# pip takes an archive URL or path as an install target while parse_spec skips anything with a
# `://`, so a wheel read as no install. PEP 427 puts the version in the filename's second field.
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
        # A PEP 508 marker rides on the end of a direct reference, and left in place it made the
        # archive regex fail, so the package read as replaced by an unknown version. Only for that
        # branch, where the URL is delimited; a `;` in a bare path is a legal character.
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
    # Padded: `<0.11.0` and `<0.11` name the same boundary.
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
                # Same version, stricter operator: intersecting them keeps the exclusion.
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


# Flags that stop pip treating what is installed as satisfying an unbounded requirement, so
# it resolves from the index instead of leaving the version alone.
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
    # `<2.0` lands somewhere in the 1.x line, and which minor that is only the index knows.
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
            # A resolution probe leaves the environment exactly as it was, so replaying its bounds
            # reported a version the cell never installed.
            continue
        # One command names a project once as far as pip is concerned: it intersects repeated
        # arguments into a single requirement, so they have to be one window here too.
        pins: list[tuple[str, str]] = []
        named = False
        replaced_unnamed = False
        for raw in inv.packages:
            if not _requirement_applies(raw, environment):
                continue  # pip skips it, so its bounds never move anything
            # Before parse_spec, which reads `./torchcodec-0.13.0-...whl` as a project called
            # `.` and hides the archive behind a name that never matches.
            archive = _archive_requirement(raw)
            if archive is not None:
                if archive[0] == target:
                    named = True
                    if archive[1] is None:
                        replaced_unnamed = True  # installed, by something with no version here
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
            current = None  # removed; a later install can put it back
            continue
        if not pins and not replaced_unnamed and _forces_resolution(inv.flags):
            # A bare name with any of these takes whatever the index offers, and nothing
            # here names which release that is.
            current, exact_known = None, True
            continue
        if replaced_unnamed:
            current, exact_known = None, True
            continue
        exact, floor, cap, ceiling, exclusions, exclusive_floor = _spec_window(pins)
        # Where an install lands when it has to move, or None when nothing names it.
        landing = floor if _window_names_one_minor(floor, ceiling, cap) else None
        if landing is not None and _split_prerelease(landing)[1]:
            # `~=0.12.0rc1` admits the stable 0.12 releases too and pip takes the newest candidate, so the
            # window names the MINOR rather than the prerelease, which PEP 440 sorts below the ABI floor.
            # Only where the window admits it: `>=0.12.0a1,<0.12.0rc1` stops below every stable 0.12.
            core = _split_prerelease(landing)[0]
            if (cap is None or cmp_versions(core, cap) <= 0) and (
                ceiling is None or cmp_versions(core, ceiling) < 0
            ):
                landing = core
        if landing is None and ceiling is not None:
            # A wider window still names the MINOR pip moves to, which is what the callers
            # compare; without it `<0.10.5` and `>=0.8,<0.11` came back unknown.
            below = _highest_minor_below(ceiling)
            if below and (floor is None or cmp_versions(below, floor) >= 0):
                landing = below
        # A requirement satisfies EVERY specifier it carries, so an inclusive cap the exclusive ceiling
        # rules out is not where pip lands: `>=0.8,<=0.11,<0.10` takes the newest of 0.8 to 0.9.x.
        cap_exact = True
        if cap is not None and ceiling is not None and cmp_versions(cap, ceiling) >= 0:
            cap, cap_exact = landing, landing is not None
        if exact is not None:
            current, exact_known = exact, True
        elif current is None or _forces_resolution(inv.flags):
            forced_off = current is not None  # got here by --upgrade over an installed one
            # `--upgrade` takes the newest available version, so an installed release that merely SATISFIES
            # the range is not where it lands: `-U "torchcodec>=0.10,<0.12"` on 0.10 moves to 0.11.
            # Absent, so the install puts it there and the only question is where: `<=V` names it exactly,
            # a floor says how low, and an exclusive ceiling names nothing, the release below it being
            # only in the index.
            if cap is not None:
                current, exact_known = cap, cap_exact
            elif landing is not None and floor is not None:
                # pip takes the newest release a BOUNDED window admits, so `>=0.10,<0.12` lands on 0.11, not on
                # its floor. A ceiling with no floor under it stays unknown, as above: nothing bounds the guess.
                current, exact_known = landing, True
            elif floor is not None:
                # `>V` does not admit V itself, but an INEXACT bound is only read by checks that hold for every
                # release at or above it, so one extra version can only under-report. Dropping it entirely left
                # `pip install "torch>2.11"` beside torchcodec 0.10 unreported.
                current, exact_known = floor, False
            elif (
                forced_off
                and landing is not None
                and cmp_versions(version_minor(current), landing) == 0
            ):
                # `--upgrade "x<0.12"` on an installed 0.11 cannot leave the 0.11 line: not above the ceiling,
                # and an upgrade does not go below what is there. The minor is the granularity these rules
                # compare, so dropping it hid a pairing every admitted release breaks.
                current, exact_known = landing, True
            elif forced_off:
                # `--upgrade` moves to the newest available release, so what is installed is not where it
                # lands, and with a ceiling and no floor nothing names the landing either.
                current, exact_known = None, True
        elif floor is not None and (
            cmp_versions(floor, current) > 0
            # `>V` is not satisfied by V itself, so equality still forces a move.
            or (exclusive_floor and cmp_versions(floor, current) == 0)
        ):
            if cap is not None:
                current, exact_known = cap, cap_exact  # `<=V` allows V, so V is what pip picks
            elif landing is not None:
                current, exact_known = landing, True  # the window pins the minor
            else:
                # At least the floor, possibly newer. `>V` excludes V itself, but as above an inexact bound
                # carrying one extra version is sound, and discarding it silenced the rule.
                current, exact_known = floor, False
        elif cap is not None and cmp_versions(current, cap) > 0:
            current, exact_known = cap, cap_exact  # `<=V` allows V, so V is what pip picks
        elif ceiling is not None and cmp_versions(current, ceiling) >= 0:
            current, exact_known = landing, True
        # Whatever is left still has to satisfy the requirement's own exclusions.
        if current is not None and any(_version_is_excluded(current, ver) for ver in exclusions):
            # `>=0.11,<0.12,!=0.11.0` stays in the 0.11 line, so only an exclusion covering the whole minor
            # takes the landing away. The landing comes off the CEILING alone, so it is checked against the
            # rest of the window: `<=0.10.0,!=0.10.0,<0.12` can only resolve below 0.10.
            if (
                landing is not None
                and (cap is None or cmp_versions(landing, cap) <= 0)
                and (ceiling is None or cmp_versions(landing, ceiling) < 0)
                and (floor is None or cmp_versions(landing, floor) >= 0)
                and not any(_exclusion_covers_minor(landing, ver) for ver in exclusions)
                # An inclusive cap pins the landing to that exact release, so excluding it leaves nothing in
                # the minor: `<0.12,<=0.11,!=0.11.0` cannot resolve to any 0.11.
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
    # Past the table, only the ABI rule can apply, and it needs a codec at or above its floor.
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
    # torchcodec 0.12+ is ABI-stable against torch >=2.11 (its build sets TORCH_TARGET_VERSION to
    # 2.11), so that half of the matrix is open-ended rather than a finite set of minors. Without
    # it the 2.11 row would flag torchcodec 0.12 through 0.15, all of which upstream supports.
    # An inexact version is a floor, which is enough for a check that only asks whether both
    # sides clear a floor of their own.
    # An inexact codec is a FLOOR, and a prerelease floor admits the stable release above it, so
    # `torchcodec>=0.12.0rc1` may land on 0.12 itself. Compared as written it stayed below the ABI
    # floor and fired on an upgrade range whose every stable member is fine.
    codec_clears_abi = at_least(codec_v, TORCHCODEC_ABI_STABLE_CODEC) or (
        not codec_exact and cmp_versions(version_minor(codec_v), TORCHCODEC_ABI_STABLE_CODEC) >= 0
    )
    if at_least(torch_v, TORCHCODEC_ABI_STABLE_TORCH) and codec_clears_abi:
        return findings  # ABI-stable pairing, not locked to one torch minor
    t_minor = version_minor(torch_v)
    c_minor = version_minor(codec_v)
    allowed = TORCH_TORCHCODEC.get(t_minor)
    if allowed is None:
        if not at_least(torch_v, TORCHCODEC_ABI_STABLE_TORCH):
            return findings  # torch older than the table — don't flag
        if not codec_exact and not at_least(c_minor, TORCHCODEC_ABI_STABLE_CODEC):
            return findings  # a newer codec above this floor would be ABI-stable and fine
        # Past the ABI floor with a pre-0.12 codec: locked to an older torch minor.
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
        # The row that applies depends on which torch the floor resolves to, unless no release at or
        # above it can take this codec at all, which fails whichever torch pip picks.
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
        # Some release at or above the floor is in the row, so nothing is proven.
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
    """Fires only when transformers is installed with `--no-deps` (otherwise
    pip resolves tokenizers transitively and flagging would be a false
    positive). Targets the PR #261b/#264 pattern: `--no-deps transformers==X`
    next to a Colab `tokenizers` outside transformers's window."""
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
    # Find the transformers pin and check for --no-deps.
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


# ----- AST-level rules over user-facing cells ----- #


class _APIScanner(ast.NodeVisitor):
    """Scan user-facing code cells for known deprecated patterns. R-API-001
    (`for_training`/`for_inference`) is intentionally absent: those helpers are
    still live as of 2026-05 (PR #221 removed them cosmetically, not as a
    deprecation). R-API-004 catches actual removals dynamically."""

    def __init__(self, file: str, cell_idx: int):
        self.file = file
        self.cell_idx = cell_idx
        self.findings: list[Finding] = []

    def visit_Call(self, node: ast.Call) -> None:
        # SFTConfig with suboptimal optim (R-API-003).
        # NOTE: PR #221 also stripped gradient_checkpointing kwargs from some
        # vision notebooks, but they're still accepted by live TRL (trl==0.25.1)
        # so that was cosmetic. We don't flag them; R-API-004 catches real drift.
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


# ----- DONT_UPDATE_EXCEPTIONS coverage ----- #

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
    """Best-effort scan of update_all_notebooks.py for canonical phrases;
    falls back to POLICY_CLAUSES_DEFAULT (which we use directly today). The
    permissive regexes avoid false positives on template rewords."""
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
            # install_cells is a text heuristic, so `!echo "pip install peft"` reaches here running no pip
            # and `applies` then demands a clause the notebook has no install to carry.
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


# ----- Drift ----- #


def cmd_drift(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    update_script = nbdir / "update_all_notebooks.py"
    if not update_script.is_file():
        print(f"FAIL: {update_script} not found", file = sys.stderr)
        return 2
    # Stash any pre-existing dirty state, run the updater, diff, restore.
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = nbdir).decode().strip()
    subprocess.run(
        ["git", "-C", str(nbdir), "stash", "--include-untracked"],
        check = False,
        capture_output = True,
    )
    # The restore MUST run even on SystemExit/KeyboardInterrupt, else the
    # working tree stays rolled back into the stash. A bare try/finally keeps
    # the original exception while still running the cleanup (stash pop).
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
        # Restore the working tree (both commands run regardless of exit path).
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


# ----- Convert ----- #


def cmd_convert(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents = True, exist_ok = True)
    converter = HERE / "notebook_to_python.py"
    if not converter.is_file():
        print(f"FAIL: {converter} not found", file = sys.stderr)
        return 2
    # Convert in batches; the script accepts multiple notebooks at once.
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


# ----- Lint (combined) ----- #


def cmd_lint(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    colab_path = pathlib.Path(args.colab_pin).resolve() if args.colab_pin else COLAB_FALLBACK_FILE
    # Pair the marker oracle with the package snapshot actually being used.
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
        # Colab oracle applies only to Colab notebooks; other targets get the
        # environment-agnostic rules only (their preinstalls aren't tracked).
        oracle = colab if env == "colab" else {}
        cells = install_cells(nb)
        # Per-cell forbid-pattern checks.
        for idx, cell in cells:
            findings += rule_inst_001_git_plus(cell, rel, idx)
            findings += rule_inst_006_double_bang(cell, rel, idx)
        # Whole-notebook rules: install steps may span multiple cells, so merge
        # before resolving compat against Colab.
        merged = "\n".join(c for _, c in cells)
        # A cell can look like an install and resolve nothing: `!echo "pip install foo"` runs no pip,
        # and `!command -v uv || pip install foo` runs it only on the fallback side. The compat rules
        # replay UNCONDITIONAL invocations, so both would compare the oracle against itself and report
        # the base image. R-INST-001 still sees the conditional path: it runs per cell, before this.
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


# ----- Exceptions coverage ----- #


def cmd_exceptions(args: argparse.Namespace) -> int:
    findings = rule_l12_exceptions_coverage(pathlib.Path(args.notebooks_dir).resolve())
    _emit(findings)
    return 0 if not findings else 1


# ----- API surface scan ----- #


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


# ----- Orchestrator ----- #


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


# The packages the R-INST rules seed on. A pin file missing them makes every rule return early,
# so a truncated 200 is refused rather than acknowledged: "parsed" is not "usable".
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
    """Pull the latest Colab pip-freeze.gpu.txt and write to disk. --all
    refreshes every oracle file into --snapshot-dir instead, which is how a
    colab-diff drift report is acknowledged in one command."""
    if args.all:
        snapshot_dir = pathlib.Path(args.snapshot_dir).resolve()
        # Fetch everything before writing anything: writing as we go would let a transient failure
        # leave a mixed-generation directory, and the tripwire would go quiet on a failed refresh.
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
                # Acknowledging a payload the rules cannot read is worse than not acknowledging: colab-diff
                # compares the two parses, so a format change written in leaves both sides equally empty.
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
            # Advisory oracle: nothing resolves a rule against it, so a transient upstream failure leaves
            # the stale snapshot in place rather than reddening the daily cron.
            print(f"::notice::skipping {upstream_name}: {reason}")
            skipped.append(upstream_name)
        snapshot_dir.mkdir(parents = True, exist_ok = True)
        # The set lands together or not at all. Each write is atomic on its own, but a failure part way
        # through left a fresh package list beside a stale Python version, and the workflow's `|| echo`
        # fallback then linted against it while reporting the committed snapshot.
        # Copies first, then writes: restoring by writing the bytes back needs the room the failure
        # just proved is missing. A rename cannot fail that way, the copy being on the same filesystem.
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
    """Return (new, removed, changed). new/removed are (key, value);
    changed is (key, old, new)."""
    new = sorted((k, upstream[k]) for k in upstream.keys() - snapshot.keys())
    removed = sorted((k, snapshot[k]) for k in snapshot.keys() - upstream.keys())
    changed = sorted(
        (k, snapshot[k], upstream[k])
        for k in upstream.keys() & snapshot.keys()
        if upstream[k] != snapshot[k]
    )
    return new, removed, changed


# Per oracle and key, what the consumer needs the VALUE to look like. `_parse_os_lines` emits a
# `python` key for any line starting with `Python` while `_colab_python_version` accepts only
# `Python <digits>`, so a reformat leaves both sides equal and markers quietly disabled.
_STRICT_KEY_VALUE_RE: dict[tuple[str, str], "re.Pattern[str]"] = {
    # Matches what _COLAB_PYTHON_RE reads, prerelease included, so a rotation the parse
    # would truncate cannot be acknowledged as usable.
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
    """Diff each Colab oracle file against its committed snapshot and print
    NEW/REMOVED/CHANGED. Advisory (rc=0) by default; --strict makes drift in
    the rule-bearing oracle (COLAB_STRICT_ORACLE) rc=1 so the daily cron fails
    loudly on upstream rotation."""
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
                # Not compared is not "no drift": passing here reported success for a check that never ran. An
                # advisory file stays a warning, as its drift does.
                any_diff = True
                strict_diff = True
                print(f"::error::colab-diff: could not fetch {url}: {e}")
            else:
                print(f"::warning::colab-diff: could not fetch {url}: {e}")
            continue
        if not snap_path.exists():
            # An absent snapshot is not "nothing to compare": without os-info's Python line markers
            # silently replay every requirement, so the strict-key declaration is consulted HERE, before
            # the continue, or --strict passed on a file that was never committed.
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
        # Present in BOTH, not merely equal: a format change acknowledged into the snapshot leaves the
        # two parses identical and empty of the key, so no-drift passed while markers were disabled.
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
        for k, v in new[:50]:
            print(f"  NEW      {k}=={v}")
        if len(new) > 50:
            print(f"  ...and {len(new) - 50} more new entries")
        for k, v in removed[:50]:
            print(f"  REMOVED  {k} (was {v})")
        if len(removed) > 50:
            print(f"  ...and {len(removed) - 50} more removed entries")
        for k, old, ver in changed[:80]:
            print(f"  CHANGED  {k}: {old} -> {ver}")
        if len(changed) > 80:
            print(f"  ...and {len(changed) - 80} more changed entries")
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


# ----- Helpers ----- #


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
