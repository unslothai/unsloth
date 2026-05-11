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
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PYPI_CACHE_DIR = DATA_DIR / "pypi_cache"

COLAB_PIP_FREEZE_URL = (
    "https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.gpu.txt"
)
COLAB_FALLBACK_FILE = DATA_DIR / "colab_pip_freeze.gpu.txt"

# Oracle files we snapshot from googlecolab/backend-info. The diff
# subcommand fetches each, compares against the committed snapshot,
# and surfaces NEW / REMOVED / CHANGED entries so upstream Colab base
# image rotations land in CI within ~24h instead of when a notebook
# breaks. Every rule in this validator that resolves against the
# Colab preinstall (R-INST-002/003/004/005) gets earlier signal.
COLAB_ORACLE_FILES: dict[str, str] = {
    "pip-freeze.gpu.txt": "colab_pip_freeze.gpu.txt",
    "apt-list-gpu.txt": "colab_apt_list.gpu.txt",
    "os-info-gpu.txt": "colab_os_info.gpu.txt",
}
COLAB_ORACLE_BASE_URL = (
    "https://raw.githubusercontent.com/googlecolab/backend-info/main/"
)

# ----- Compat tables. PRs add rows as new releases land. ----- #

# torch.minor -> set of compatible torchcodec.minor strings.
# Source: pytorch/torchcodec compatibility matrix on its README.
TORCH_TORCHCODEC: dict[str, set[str]] = {
    "2.10": {"0.10"},
    "2.9": {"0.7", "0.8", "0.9"},
    "2.8": {"0.6"},
    "2.7": {"0.3", "0.4", "0.5"},
    "2.6": {"0.2", "0.3"},
    "2.5": {"0.1", "0.2"},
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
    """Yield user-facing .ipynb files under nb/ and kaggle/. Pass
    include_templates=True to also walk original_template/ (used by the
    convert subcommand which doesn't lint install cells)."""
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


def install_cells(nb: dict[str, Any]) -> list[tuple[int, str]]:
    """Heuristic: any code cell that contains a `pip install`, `pip uninstall`
    or `uv pip install` shell command, or a top-line `%%capture` magic."""
    out = []
    for i, src in code_cells(nb):
        first = src.lstrip().splitlines()[:1]
        if first and first[0].strip().startswith("%%capture"):
            out.append((i, src))
            continue
        if re.search(
            r"^[ \t]*!\s*(uv\s+)?pip\s+(install|uninstall)\b", src, re.MULTILINE
        ):
            out.append((i, src))
    return out


# Notebook target environment. The Colab oracle (pip-freeze.gpu.txt) only
# applies to notebooks that actually run on Colab; AMD-Dev-Cloud,
# Kaggle, HuggingFace-Course, and DGX-Spark notebooks have their own
# preinstalled environments and the Colab-vs-cell rules are not
# applicable to them.
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


PIP_LINE_RE = re.compile(
    r"^\s*!\s*(?P<tool>(?:uv\s+)?pip)\s+(?:install|uninstall)\b(?P<rest>.*)$",
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
        tool = tool, flags = flags, packages = packages, raw = line, line_no = line_no
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


def iter_pip_invocations(install_cell: str) -> Iterator[PipInvocation]:
    for line_no, line in _glue_line_continuations(install_cell):
        inv = parse_pip_line(line, line_no)
        if inv is not None:
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
    path.write_text(json.dumps(data))
    return data


def transitive_constraint(
    name: str, version: str, target: str
) -> tuple[str | None, list[str]]:
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
    """Merge install-cell explicit constraints with Colab pip-freeze. Cell
    wins.

    Resolution order per package, when more than one form is present:
      1. Exact `==V` pin in any install line  (definitive).
      2. Upper-bound `<=V` constraint         (pip picks the highest
                                                allowed; that's V).
      3. Colab pip-freeze fallback.

    The lower-bound `>=V` is intentionally NOT reflected here — a `>=V`
    by itself doesn't change the resolved version when a higher
    Colab-preinstalled version is already in scope. (R-INST-003 calls
    `_install_cell_lower_bound` separately to model that case.)
    """
    out = dict(colab)
    pinned: set[str] = set()
    upper_bounds: dict[str, str] = {}
    for inv in iter_pip_invocations(install_cell):
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None:
                continue
            for op, ver in sp.pins:
                if op == "==":
                    out[sp.name] = ver
                    pinned.add(sp.name)
                elif op == "<=" and sp.name not in pinned:
                    if (
                        sp.name not in upper_bounds
                        or cmp_versions(ver, upper_bounds[sp.name]) < 0
                    ):
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


def rule_inst_001_git_plus(
    install_cell: str, file: str, cell_idx: int
) -> list[Finding]:
    findings: list[Finding] = []
    for inv in iter_pip_invocations(install_cell):
        if any("git+" in p for p in inv.packages) or "git+" in inv.raw:
            if any(allowed in inv.raw for allowed in GIT_PLUS_ALLOWLIST):
                continue
            findings.append(
                Finding(
                    rule = "R-INST-001",
                    file = file,
                    cell = cell_idx,
                    line = inv.line_no,
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
    for inv in iter_pip_invocations(install_cell):
        if "--no-deps" not in inv.flags:
            continue
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None:
                continue
            v = explicit_pin(sp)
            if v is None:
                continue
            # Check transitive constraints on a curated short list of pkgs we
            # care about (transformers/peft/trl/accelerate/torchao/torchcodec).
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


def _install_cell_lower_bound(install_cell: str, target: str) -> str | None:
    """Return the highest LOWER bound that any install line places on `target`,
    or None if no constraint is present. Treats `==V` as both lower and upper.
    Used by R-INST-003: a `pip install torchao>=0.16.0` line is enough to
    satisfy a `torchao>=0.16.0` floor even though it's not a `==` pin."""
    best: str | None = None
    for inv in iter_pip_invocations(install_cell):
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or sp.name != target:
                continue
            for op, ver in sp.pins:
                if op in ("==", ">="):
                    if best is None or cmp_versions(ver, best) > 0:
                        best = ver
    return best


def rule_inst_003_peft_torchao(
    install_cell: str, colab: dict[str, str], file: str, cell_idx: int
) -> list[Finding]:
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    peft_v = res.get("peft")
    if not peft_v:
        return findings
    torchao_explicit = _install_cell_lower_bound(install_cell, "torchao")
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
    torch_v = res.get("torch")
    codec_v = res.get("torchcodec")
    if not torch_v or not codec_v:
        return findings
    t_minor = version_minor(torch_v)
    c_minor = version_minor(codec_v)
    allowed = TORCH_TORCHCODEC.get(t_minor)
    if allowed is None:
        return findings  # unknown torch minor — don't flag
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
    """Fires only when transformers is installed with `--no-deps`. Without
    `--no-deps`, pip resolves the correct tokenizers transitively, so the
    rule would be a false positive (this is the case for older notebooks
    that pin `transformers==4.51.3` but rely on pip's transitive resolver).
    The rule targets the exact pattern PR #261b / #264 fixed:
    `pip install --no-deps transformers==X` next to a Colab preinstall
    `tokenizers` outside transformers's window."""
    findings: list[Finding] = []
    res = resolved_set(install_cell, colab)
    tf = res.get("transformers")
    tok = res.get("tokenizers")
    if not tf or tok is None:
        return findings
    # Find the install line that pins transformers and check for --no-deps.
    transformers_line_no_deps = False
    for inv in iter_pip_invocations(install_cell):
        for raw in inv.packages:
            sp = parse_spec(raw)
            if sp is None or sp.name != "transformers":
                continue
            if explicit_pin(sp) is None:
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


def rule_inst_006_double_bang(
    install_cell: str, file: str, cell_idx: int
) -> list[Finding]:
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
    (`for_training`/`for_inference`) is intentionally absent: those helpers
    are still part of the live unsloth surface as of 2026-05; PR #221 removed
    the calls cosmetically from Vision notebooks but did not deprecate the
    methods. R-API-004 (live API surface diff) catches actual removals
    dynamically without us hand-coding them."""

    def __init__(self, file: str, cell_idx: int):
        self.file = file
        self.cell_idx = cell_idx
        self.findings: list[Finding] = []

    def visit_Call(self, node: ast.Call) -> None:
        # SFTConfig with suboptimal optim  (R-API-003).
        # NOTE: PR #221 also stripped `gradient_checkpointing` /
        # `gradient_checkpointing_kwargs` from a handful of vision notebooks,
        # but those kwargs are still accepted by live TRL (verified against
        # trl==0.25.1 in the unsloth workspace) so removing them was
        # cosmetic, not a deprecation. We do NOT flag them. R-API-004 (live
        # API surface diff in the api subcommand) is the right way to catch
        # actual TRL signature drift.
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


def extract_policy_clauses(
    update_script: pathlib.Path,
) -> list[tuple[str, re.Pattern[str], Any]]:
    """Best-effort: scan update_all_notebooks.py for canonical phrases used by
    multiple templates. Falls back to POLICY_CLAUSES_DEFAULT.

    Today we use POLICY_CLAUSES_DEFAULT directly; the regex form is
    intentionally permissive so a template-side reword (e.g. comment changes)
    doesn't cause false positives. New clauses become 1-line PRs to this list.
    """
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
    head = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd = nbdir)
        .decode()
        .strip()
    )
    subprocess.run(
        ["git", "-C", str(nbdir), "stash", "--include-untracked"],
        check = False,
        capture_output = True,
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(update_script)],
            cwd = nbdir,
            capture_output = True,
            text = True,
            timeout = 600,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: update_all_notebooks.py timed out (>600s)", file = sys.stderr)
        return 2
    if proc.returncode != 0:
        print(
            f"FAIL: update_all_notebooks.py exited {proc.returncode}", file = sys.stderr
        )
        sys.stderr.write(proc.stderr[-2000:])
        return 2
    diff_proc = subprocess.run(
        ["git", "-C", str(nbdir), "diff", "--stat"], capture_output = True, text = True
    )
    findings: list[Finding] = []
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
    # Restore.
    subprocess.run(
        ["git", "-C", str(nbdir), "checkout", "."], check = False, capture_output = True
    )
    subprocess.run(
        ["git", "-C", str(nbdir), "stash", "pop"], check = False, capture_output = True
    )
    _emit(findings)
    return 0 if not findings else 1


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
    print(
        f"converted {len(notebooks) - len(failed)}/{len(notebooks)} notebooks to {out}"
    )
    _emit(failed)
    return 0 if not failed else 1


# ----- Lint (combined) ----- #


def cmd_lint(args: argparse.Namespace) -> int:
    nbdir = pathlib.Path(args.notebooks_dir).resolve()
    colab_path = (
        pathlib.Path(args.colab_pin).resolve()
        if args.colab_pin
        else COLAB_FALLBACK_FILE
    )
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
        # The Colab oracle is the source of truth ONLY for Colab notebooks.
        # Other targets (amd / kaggle / dgx_spark) have their own runtime
        # preinstall sets that aren't tracked here yet, so we apply the
        # environment-agnostic rules and skip the Colab-specific ones.
        oracle = colab if env == "colab" else {}
        cells = install_cells(nb)
        # Per-cell rules: forbid-pattern checks scoped to a single line.
        for idx, cell in cells:
            findings += rule_inst_001_git_plus(cell, rel, idx)
            findings += rule_inst_006_double_bang(cell, rel, idx)
        # Whole-notebook rules: a notebook's install steps are sometimes split
        # across multiple cells (initial install + post-install bumps). Merge
        # all install cells before resolving compat against Colab.
        merged = "\n".join(c for _, c in cells)
        if env == "colab" and merged:
            first_cell = cells[0][0] if cells else None
            findings += rule_inst_003_peft_torchao(merged, oracle, rel, first_cell)
            findings += rule_inst_004_torchcodec_torch(merged, oracle, rel, first_cell)
            findings += rule_inst_005_transformers_tokenizers(
                merged, oracle, rel, first_cell
            )
            if not args.no_pypi:
                findings += rule_inst_002_no_deps_transitive(
                    merged, oracle, rel, first_cell
                )
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


def cmd_refresh_colab(args: argparse.Namespace) -> int:
    """Pull the latest Colab pip-freeze.gpu.txt and write to disk."""
    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents = True, exist_ok = True)
    try:
        with urllib.request.urlopen(COLAB_PIP_FREEZE_URL, timeout = 15) as r:
            data = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"FAIL: could not fetch {COLAB_PIP_FREEZE_URL}: {e}", file = sys.stderr)
        return 2
    out.write_bytes(data)
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
    """Free-form `<tool> <version>` lines. Skip comments. The key is the
    first token lower-cased; the value is the rest of the line."""
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
    """Fetch every Colab oracle file in COLAB_ORACLE_FILES, diff against
    the committed snapshot, and print NEW / REMOVED / CHANGED. Advisory
    by default (rc=0); --strict promotes any diff to rc=1 so the daily
    cron can fail loudly when upstream rotates."""
    snapshot_dir = pathlib.Path(args.snapshot_dir).resolve()
    any_diff = False
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
            print(
                f"::warning::colab-diff: no committed snapshot at {snap_path}; skipping"
            )
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
    if any_diff and args.strict:
        print(
            "\n::error::Colab oracle drifted from committed snapshot; "
            "refresh scripts/data/colab_*.txt to acknowledge.",
            file = sys.stderr,
        )
        return 1
    if any_diff:
        print(
            "\n::notice::Colab oracle drifted; "
            "refresh scripts/data/colab_*.txt at your convenience."
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

    pa = sub.add_parser("colab-diff")
    pa.add_argument("--snapshot-dir", default = str(DATA_DIR))
    pa.add_argument(
        "--strict",
        action = "store_true",
        help = "exit 1 on any drift (default: advisory; exit 0)",
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
