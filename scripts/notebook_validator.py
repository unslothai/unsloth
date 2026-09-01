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

# Oracle files snapshotted from googlecolab/backend-info. The colab-diff
# subcommand surfaces NEW/REMOVED/CHANGED entries so upstream Colab base
# image rotations land in CI within ~24h, giving R-INST-002/003/004/005
# earlier signal.
# The image's Python, read from the os-info oracle beside the pip freeze ("Python 3.13.15").
# Only used to evaluate PEP 508 markers, so an unreadable or absent snapshot just means no
# marker is evaluated and every requirement is replayed, which is the older behaviour.
_COLAB_OS_INFO_FILE = DATA_DIR / "colab_os_info.gpu.txt"
_COLAB_PYTHON_RE = re.compile(r"^Python\s+(\d+\.\d+(?:\.\d+)?)", re.MULTILINE)


@functools.lru_cache(maxsize = 1)
def _colab_python_version() -> str | None:
    try:
        text = _COLAB_OS_INFO_FILE.read_text(encoding = "utf-8")
    except OSError:
        return None
    match = _COLAB_PYTHON_RE.search(text)
    return match.group(1) if match else None


def _marker_environment(colab: dict[str, str]) -> dict[str, str] | None:
    """The environment PEP 508 markers are evaluated against, or None to skip them.

    Only for notebooks resolved against the Colab image, since that is the one environment
    this can name. Everything else replays every requirement, as before.
    """
    if not colab:
        return None
    full = _colab_python_version()
    if not full:
        return None
    parts = full.split(".")
    return {
        "python_version": ".".join(parts[:2]),
        "python_full_version": full if len(parts) > 2 else f"{full}.0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
    }


# PEP 508 names a fixed set of marker variables. `Marker.evaluate` fills any field the given
# environment omits from the running process, so a marker naming one of these would be judged
# against this machine rather than the image and the answer would move between runners.
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

    pip skips such a requirement outright, so replaying its bounds moves a version the cell
    never touches. An unparseable marker, a missing `packaging`, or no environment to judge
    against all mean the requirement is replayed.
    """
    if environment is None or ";" not in raw:
        return True
    marker_text = raw.split(";", 1)[1].strip()
    if not marker_text:
        return True
    named = set(re.findall(r"[A-Za-z_]\w*", marker_text)) & _MARKER_VARIABLES
    if not named or named - environment.keys():
        return True  # nothing to judge on, or a field the oracle cannot answer for
    try:
        from packaging.markers import Marker
        return bool(Marker(marker_text).evaluate(environment))
    except Exception:
        return True


COLAB_ORACLE_FILES: dict[str, str] = {
    "pip-freeze.gpu.txt": "colab_pip_freeze.gpu.txt",
    "apt-list-gpu.txt": "colab_apt_list.gpu.txt",
    "os-info-gpu.txt": "colab_os_info.gpu.txt",
}
# Only the pip oracle feeds a rule: `lint --colab-pin` reads it, and that is
# what R-INST-002/003/004/005 resolve against. apt-list / os-info are human
# context (what else the image ships), so their drift is reported but never
# fails --strict -- otherwise an Ubuntu security bump nothing can consult
# turns the daily cron red.
COLAB_STRICT_ORACLE = "pip-freeze.gpu.txt"
COLAB_ORACLE_BASE_URL = "https://raw.githubusercontent.com/googlecolab/backend-info/main/"

# ----- Compat tables. PRs add rows as new releases land. ----- #

# torch.minor -> set of compatible torchcodec.minor strings.
# Source: pytorch/torchcodec compatibility matrix on its README.
# Mirrors import_fixes._TORCH_TORCHCODEC_MINORS (test_torchcodec_torch_compat asserts equality).
TORCH_TORCHCODEC: dict[str, set[str]] = {
    "2.11": {"0.11"},
    "2.10": {"0.10"},
    "2.9": {"0.8", "0.9"},
    "2.8": {"0.6", "0.7"},
    "2.7": {"0.3", "0.4", "0.5"},
    "2.6": {"0.2", "0.3"},
    "2.5": {"0.1", "0.2"},
}

# torchcodec 0.12+ is ABI-stable against torch >=2.11, so that half is open-ended.
TORCHCODEC_ABI_STABLE_TORCH = "2.11"
TORCHCODEC_ABI_STABLE_CODEC = "0.12"

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


# A shell line that runs pip somewhere in it. Anchored on the `!` so a `pip install` inside a
# Python string is not a cell, and open after it so a chained or compound command is:
# `!echo start; pip install x` and `!if ...; then pip install x; fi` both install.
_PIP_CELL_RE = re.compile(r"^[ \t]*!.*\b(?:uv\s+)?pip\s+(?:install|uninstall)\b", re.MULTILINE)


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


def cmp_versions(a: str, b: str) -> int:
    """Return -1/0/+1. Compares dotted numeric components only."""

    def to_tuple(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", normalise_version(v)))

    ta, tb = to_tuple(a), to_tuple(b)
    if ta < tb:
        return -1
    if ta > tb:
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


PIP_LINE_RE = re.compile(
    r"^\s*!\s*(?P<tool>(?:uv\s+)?pip)\s+(?P<action>install|uninstall)\b(?P<rest>.*)$",
    re.IGNORECASE,
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
    tool = "uv-pip" if "uv" in m.group("tool") else "pip"
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


# Words that introduce a compound command. A pip call behind one still runs, so it has to
# parse. Whether it is certain depends on which word: `if pip install ...` is the test and is
# reached whenever the line is, while a `then` or `do` body runs only if that test said so.
_SHELL_TEST_KEYWORDS = frozenset({"if", "while", "until", "for", "case"})
_SHELL_BODY_KEYWORDS = frozenset({"then", "elif", "else", "do"})
_SHELL_KEYWORDS = _SHELL_TEST_KEYWORDS | _SHELL_BODY_KEYWORDS | {"fi", "done", "esac"}


def _unquoted_arm_close(text: str) -> int | None:
    """Index of the `)` that closes a case-arm pattern, or None when there is none.

    Shell quoting decides: `"x")` is a pattern, while the `)` in `pip install "a)b"` and in a
    `$( )` substitution belongs to the command.
    """
    quote = ""
    depth = 0
    for index, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            if depth:
                depth -= 1
            elif index:
                return index
            else:
                return None
    return None


def _unwrap_shell_group(command: str) -> tuple[str, bool]:
    """`( pip install x )` -> `("pip install x", False)`, `then pip install x` -> `(..., True)`.

    A grouped or compound command still runs, so leaving the bracket or the keyword on hides
    it from PIP_LINE_RE and with it from every rule, R-INST-001's git+ ban included. The flag
    says the keyword made it conditional, which only a body word does: `if pip install ...` is
    the test and is reached whenever the line is.
    """
    stripped = command.strip()
    bang = stripped.startswith("!")
    if bang:
        stripped = stripped[1:].lstrip()
    stripped = stripped.lstrip("({").strip().rstrip(")}").rstrip()
    conditional = False
    while True:
        # Any whitespace, not a literal space: `then\tpip install ...` is the same command to
        # the shell, and leaving `then\tpip` as one word hides it from every rule.
        parts = stripped.split(maxsplit = 1)
        if not parts or parts[0].lower() not in _SHELL_KEYWORDS:
            break
        conditional = conditional or parts[0].lower() in _SHELL_BODY_KEYWORDS
        stripped = parts[1].strip() if len(parts) > 1 else ""
    # A `case` arm label: `x in x) pip install ...`, a quoted `"x") ...`, and the bare
    # `b) pip install ...` of a later arm. Only the matching arm runs, so the command is
    # conditional. The label ends at the first unquoted `)` with nothing open before it; a `)`
    # inside quotes or inside a `$( )` belongs to the command.
    close = _unquoted_arm_close(stripped)
    if close is not None:
        stripped = stripped[close + 1 :].strip()
        conditional = True
    return (f"!{stripped}" if bang and stripped else stripped), conditional


def _split_chained(line: str) -> list[tuple[str, bool]]:
    """One shell line -> `(command, conditional)` per command. Only the first keeps the `!`.

    `pip uninstall -y x && pip install x==1` is two commands with two actions; read as one,
    the regex hands the whole line to the first and the reinstall lands in the uninstall's
    package list. Scanned rather than split on a pattern because a PEP 508 marker puts a
    quoted `;` inside a single argument (`"torch==2.12.0; python_version >= '3.10'"`), and a
    backslash escapes the next character outside single quotes, as the shlex pass in
    parse_pip_line does.

    A `||` fallback runs only when the command before it failed, so it is flagged conditional
    rather than dropped: it can still run, and the rules that must see every install path
    (R-INST-001 and the git+ ban above all) have to keep seeing it. Only the effective-version
    replay skips them. The tail ends at an `&&` or a `;`, since the lists are left-associative
    and `(A || B) && C` runs C when A succeeded. Each group keeps its own tail, and a command
    is conditional when any level above it is in one: the `&&` in `A || (B && C)` ends the
    group's tail and leaves the outer one, the one in `(A || B && C)` ends the only tail there
    is. A single `&` or `|` separates as well, and both sides run either way, so neither opens
    a tail; `>&` and `&>` are redirections and are left alone. An unquoted `#` that starts a
    word comments out the rest of the line, so scanning stops.
    """
    out: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote = ""
    # One flag per open group, plus the base list. A command is conditional when any level
    # above it is in a fallback tail, so an inner list cannot clear an outer one.
    tails = [False]
    buf_conditional = False
    i = 0

    def flush() -> None:
        nonlocal buf
        out.append(("".join(buf), buf_conditional))
        buf = []

    while i < len(line):
        ch = line[i]
        if ch == "\\" and quote != "'" and i + 1 < len(line):
            buf.append(ch)
            buf.append(line[i + 1])
            i += 2
        elif quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
        elif ch == "#" and (i == 0 or line[i - 1].isspace() or line[i - 1] in ";&|)}"):
            break  # an operator or a closing bracket ends a word, so `;#` and `)#` comment
        elif line.startswith("||", i):
            flush()
            tails[-1] = True
            buf_conditional = any(tails)
            i += 2
        elif line.startswith("&&", i):
            flush()
            # Left-associative: (A || B) && C runs C when A succeeded. Only this list's tail
            # ends here, so an enclosing fallback still covers what follows.
            tails[-1] = False
            buf_conditional = any(tails)
            i += 2
        elif ch == ";" or (
            # `A & B` backgrounds A and runs B, `A | B` runs both: unconditional either way.
            # `>&` and `&>` are redirections, not separators.
            ch in "&|"
            and not (ch == "&" and (line[i - 1 : i] == ">" or line[i + 1 : i + 2] == ">"))
        ):
            flush()
            tails[-1] = False
            buf_conditional = any(tails)
            i += 1
        else:
            if ch in "({":
                tails.append(False)
                if not "".join(buf).strip():
                    buf_conditional = any(tails)  # the group opens before the command
            elif ch in ")}" and len(tails) > 1:
                # The command in hand belongs to the level being closed, so its flag stays
                # what it was; the pop only affects what comes after.
                tails.pop()
            buf.append(ch)
            i += 1
    flush()
    (head, head_conditional), *rest = out
    head_text, head_keyword = _unwrap_shell_group(head)
    commands = [(head_text, head_conditional or head_keyword)]
    for piece, flag in rest:
        text, keyword = _unwrap_shell_group(piece.strip())
        if text:
            commands.append((f"!{text}", flag or keyword))
    return commands


def unconditional_pip_invocations(install_cell: str) -> Iterator[PipInvocation]:
    """The commands that certainly run.

    Anything asking what the cell leaves installed wants this one. `iter_pip_invocations`
    yields the `||` fallbacks too, and only a rule that must see every path a notebook could
    take, R-INST-001's git+ ban above all, should be reading those.
    """
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


# A `git+` target runs to the next shell or quoting boundary.
_GIT_SOURCE_RE = re.compile(r"""git\+[^\s'"]+""")


def _git_source_repository(source: str) -> str:
    """`git+https://user@github.com/Org/Repo.git@ref` -> `github.com/org/repo`.

    Matched against the allowlist as a path rather than a substring: an arbitrary repository
    can carry `github.com/unslothai/unsloth` inside its own path, and a substring test reads
    that as permission.
    """
    remainder = source.split("+", 1)[1] if "+" in source else source
    remainder = remainder.split("://", 1)[-1]
    host, _, path = remainder.partition("/")
    host = host.rsplit("@", 1)[-1]  # drop any credentials
    path = path.split("#", 1)[0].split("?", 1)[0]
    path = path.split("@", 1)[0].rstrip("/")  # drop a trailing @ref
    if path.endswith(".git"):
        path = path[: -len(".git")]
    # Resolve `.` and `..` the way a URL client does, or
    # `github.com/unslothai/unsloth/../../attacker/repo` reads as an allowlisted prefix.
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

    The question is whether the cell can reach a `git+` source at all, so the answer must not
    depend on how the line splits: a fallback, a `(...)` group and an `if ...; then` body are
    all reachable, and `unconditional_pip_invocations` would drop them. It does depend on the
    command being pip: a `git+` in an `echo` beside an install installs nothing, and a `git+`
    in a comment is documentation, which `_split_chained` has already dropped.

    Each source is read twice over, from the command text and from the arguments shlex made
    of it: `"git+"https://...` is one argument to pip and two words to a text scan, and a
    source inside a construct only one of them can read still has to be seen.
    """
    findings: list[Finding] = []
    for line_no, line in _glue_line_continuations(install_cell):
        sources: list[str] = []
        for command, _ in _split_chained(line):
            inv = parse_pip_line(command, line_no)
            if inv is None:
                continue
            sources += _GIT_SOURCE_RE.findall(command)
            sources += [arg for arg in inv.packages if arg.startswith("git+")]
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

    PEP 440 drops the last component and increments what is then last.
    """
    parts = normalise_version(version).split(".")
    if len(parts) < 2:
        return None
    head = parts[:-1]
    try:
        head[-1] = str(int(head[-1]) + 1)
    except ValueError:
        return None
    return ".".join(head)


# pip takes `<archive url/path>` as an install target and parse_spec skips anything with a
# `://`, so a wheel that replaces the package reads as no install at all. The version sits in
# the filename: PEP 427 puts it in the second `-` field.
_ARCHIVE_RE = re.compile(
    r"(?P<name>[A-Za-z0-9._-]+?)-(?P<version>\d[^-]*?)(?:-.*)?\.(?:whl|tar\.gz|zip)$",
    re.IGNORECASE,
)


def _archive_requirement(argument: str) -> tuple[str, str | None] | None:
    """`(project, version)` for a direct archive install, or None when it is not one.

    The version is None when the target is named but its archive does not encode one, as in
    `torchcodec @ https://.../v0.13.0.zip`: the package is replaced, by something this cannot
    name.
    """
    named, sep, reference = argument.partition("@")
    if sep and "://" in reference:
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

    `cmp_versions` stops at the shorter tuple, so it reads `0.11.0` as above `0.11`. That is
    harmless for ordering across different releases and wrong wherever the question is whether
    two spellings name the same one, which is what an exclusion and a minor bound both ask.
    """
    left = [int(part) for part in re.findall(r"\d+", normalise_version(a))]
    right = [int(part) for part in re.findall(r"\d+", normalise_version(b))]
    width = max(len(left), len(right))
    left += [0] * (width - len(left))
    right += [0] * (width - len(right))
    return (left > right) - (left < right)


def _exclusion_covers_minor(version: str, exclusion: str) -> bool:
    """True when `!=exclusion` rules out every release in `version`'s minor.

    Only a wildcard can: `!=0.11.*` takes the whole 0.11 line, while `!=0.11` and `!=0.11.1.*`
    each remove one release or one patch line and leave the minor reachable.
    """
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

    A window lands on the newest release it admits, which only the window itself names when
    there is one minor to land in. `>=0.10,<0.11` and `>=0.10,<=0.10.5` qualify;
    `>=0.10,<0.12` and `>=0.10,<=0.11` do not, and pip would pick 0.11 there.
    """
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

    `cap` is an inclusive `<=`, which names the version pip lands on; `ceiling` is an
    exclusive `<` or the one `~=` implies, which does not. A `>` floor comes back with the
    flag set, since the endpoint it names is the one version pip will not install.
    """
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


def _forces_resolution(flags: set[str]) -> bool:
    """True when any flag makes pip re-resolve rather than keep what is installed.

    Short options bundle: pip takes `-Uq` and parse_pip_line keeps it as one token, so the
    letters are compared rather than the token.
    """
    if flags & _RESOLVE_ANYWAY_LONG:
        return True
    return any(
        not flag.startswith("--") and flag.startswith("-") and set(flag[1:]) & _RESOLVE_ANYWAY_SHORT
        for flag in flags
    )


def _effective_version(
    install_cell: str,
    target: str,
    resolved: str | None,
    environment: dict[str, str] | None = None,
) -> tuple[str | None, bool]:
    """`resolved` walked forward through the cell's own requirements, in invocation order.

    resolved_set() drops every bound but `==` and `<=`, and applies those all at once at the
    end. Order is what decides between them: a later requirement overrides an earlier one
    either way. Without this R-INST-004's own `torchcodec>=0.12.0` remedy could not clear the
    error it offers, and `torchcodec>=0.10,<0.11` could not raise one.

    Each requirement is a window. An install moves the version into that window when it falls
    outside and leaves it alone when it does not, which is what pip does. Where it moves to is
    the window's floor, or an inclusive `<=` when the move is downwards. Moving down only
    names a version when the window holds one minor, which is the granularity the callers
    compare on: `>=0.10,<0.11` and `~=0.10.0` do, `>=0.10,<0.12` and `>=0.9,!=0.11.*` do not.
    A `>` floor names the one version pip will not install, so it too only moves the version
    when a ceiling pins the minor. Anything that cannot say where the install lands clears the
    version rather than keeping a stale one, and a bound on an absent package leaves it
    absent, unless the requirement carries a floor, which at least says it is installed and
    how low it can be.

    Returns `(version, exact)`. An open floor moves the version up but does not name it: pip
    takes the newest release above it, so `>=0.8` can land anywhere from 0.8 upwards. That
    comes back inexact, and the caller may only use it where every version at or above it
    gives the same answer.
    """
    current = resolved
    exact_known = True
    for inv in unconditional_pip_invocations(install_cell):
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
            # A bare name with any of these takes whatever the index offers: none of them let
            # the installed version satisfy the requirement, so it is not what the cell ends
            # on, and nothing here names what does.
            current, exact_known = None, True
            continue
        if replaced_unnamed:
            current, exact_known = None, True
            continue
        exact, floor, cap, ceiling, exclusions, exclusive_floor = _spec_window(pins)
        # Where an install lands when it has to move, or None when nothing names it.
        landing = floor if _window_names_one_minor(floor, ceiling, cap) else None
        if exact is not None:
            current, exact_known = exact, True
        elif current is None:
            # Absent, so the install puts it there. A floor is all that can be said about
            # where; without one there is nothing to say at all.
            if floor is not None and not exclusive_floor:
                current, exact_known = floor, landing is not None
        elif floor is not None and (
            cmp_versions(floor, current) > 0
            # `>V` is not satisfied by V itself, so equality still forces a move.
            or (exclusive_floor and cmp_versions(floor, current) == 0)
        ):
            if cap is not None:
                current, exact_known = cap, True  # `<=V` allows V, so V is what pip picks
            elif landing is not None:
                current, exact_known = landing, True  # the window pins the minor
            elif exclusive_floor:
                current, exact_known = None, True  # nothing names where it went
            else:
                current, exact_known = floor, False  # at least the floor, possibly newer
        elif cap is not None and cmp_versions(current, cap) > 0:
            current, exact_known = cap, True  # `<=V` allows V, so V is what pip picks
        elif ceiling is not None and cmp_versions(current, ceiling) >= 0:
            current, exact_known = landing, True
        # Whatever the requirement leaves in place still has to satisfy its own exclusions,
        # which covers the version that was already installed and the one just landed on.
        if current is not None and any(_version_is_excluded(current, ver) for ver in exclusions):
            # A window that pins one minor still pins it: `>=0.11,<0.12,!=0.11.0` moves off
            # 0.11.0 and stays in the 0.11 line, and the minor is what the callers compare.
            # Only an exclusion covering the whole minor takes that away.
            if landing is not None and not any(
                _exclusion_covers_minor(landing, ver) for ver in exclusions
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
    # An inexact version is a floor: everything at or above it is possible. That is enough for
    # the ABI check, which only asks whether both sides clear a floor of their own.
    if (
        cmp_versions(torch_v, TORCHCODEC_ABI_STABLE_TORCH) >= 0
        and cmp_versions(codec_v, TORCHCODEC_ABI_STABLE_CODEC) >= 0
    ):
        return findings  # ABI-stable pairing, not locked to one torch minor
    t_minor = version_minor(torch_v)
    c_minor = version_minor(codec_v)
    allowed = TORCH_TORCHCODEC.get(t_minor)
    if allowed is None:
        if cmp_versions(torch_v, TORCHCODEC_ABI_STABLE_TORCH) < 0:
            return findings  # torch older than the table — don't flag
        if not codec_exact and cmp_versions(c_minor, TORCHCODEC_ABI_STABLE_CODEC) < 0:
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
        # The row that applies depends on which torch this floor resolves to.
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
    if not tf or tok is None:
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


def cmd_refresh_colab(args: argparse.Namespace) -> int:
    """Pull the latest Colab pip-freeze.gpu.txt and write to disk. --all
    refreshes every oracle file into --snapshot-dir instead, which is how a
    colab-diff drift report is acknowledged in one command."""
    if args.all:
        snapshot_dir = pathlib.Path(args.snapshot_dir).resolve()
        # Fetch everything before writing anything. Writing as we go would let a
        # transient apt/os-info failure leave a mixed-generation directory -- and
        # since pip is fetched first and is the only oracle --strict reads, the
        # tripwire would go quiet on a refresh that actually failed.
        payloads: dict[str, bytes] = {}
        for upstream_name, snapshot_name in COLAB_ORACLE_FILES.items():
            data = _fetch_oracle(COLAB_ORACLE_BASE_URL + upstream_name)
            if data is None:
                print(
                    "FAIL: refresh-colab --all could not fetch every oracle; "
                    "no snapshot was written",
                    file = sys.stderr,
                )
                return 2
            payloads[snapshot_name] = data
        snapshot_dir.mkdir(parents = True, exist_ok = True)
        for snapshot_name, data in payloads.items():
            _atomic_write_bytes(snapshot_dir / snapshot_name, data)
            print(f"wrote {len(data)} bytes to {snapshot_dir / snapshot_name}")
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
            print(f"::warning::colab-diff: could not fetch {url}: {e}")
            continue
        if not snap_path.exists():
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
        if not n:
            print("  no drift")
            continue
        any_diff = True
        strict_diff = strict_diff or upstream_name == COLAB_STRICT_ORACLE
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
            f"\n::error::Colab oracle {COLAB_STRICT_ORACLE} drifted from its "
            "committed snapshot; run `notebook_validator.py refresh-colab --all "
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
