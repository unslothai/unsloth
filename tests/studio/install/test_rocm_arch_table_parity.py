# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drift guards for the AMD gfx tables that are duplicated across the installers.

The same three tables are hand-copied into up to four files each:

  gfx -> AMD index family   install.sh (_amd_arch_index_family_for_gfx)
                            install.ps1 ($archFamilyMap)
                            studio/setup.ps1 ($archFamilyMap)
                            studio/install_python_stack.py (_GFX_TO_AMD_INDEX_ARCH)

  GPU name -> gfx           install.sh (_infer_amd_gfx_arch_from_gpu_name)
                            studio/setup.sh (case "$_setup_mkt")
                            install.ps1 ($nameArchTable)
                            studio/setup.ps1 ($nameArchTable)

  torch>=2.11 pin allowlist install.sh (case "$_torch_index_leaf")
                            install.ps1 ($_pinGfx211)
                            studio/setup.ps1 (Test-RocmPinLeaf211)

Every copy carries a "kept in sync with" comment and nothing enforced it, which is
how the routing family of bugs kept recurring: #7264 / #7280 (Strix left on the
generic rocm7.2 index), #7293 / #7300 (fixed in one installer at a time) and #7277
(RDNA2 gfx1030-1036 added to install.ps1 / setup.ps1 / install_python_stack.py --
install.sh had to follow separately). Half-applied edits are invisible until an AMD
user on the missed path gets CPU-only PyTorch.

These tests parse each copy out of its source file and compare them, so a table
edited in one place fails CI naming the file that was missed.
"""

import fnmatch
import importlib.util
import re
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_parity", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()


# ── Source extraction helpers ────────────────────────────────────────────────


def _sh_function_body(source: str, name: str) -> str:
    """Return a POSIX-shell function body by brace matching (same idea as
    _extract_sh_function_body in test_rocm_support.py, kept local so this file
    stands alone)."""
    needle = f"{name}() {{"
    start = source.find(needle)
    assert start != -1, f"{name}() not found"
    depth = 0
    i = start + len(needle) - 1
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {name}()")


def _sh_case_block(source: str, subject: str) -> str:
    """Return the body of `case <subject> in ... esac` (first match)."""
    start = source.find(f"case {subject} in")
    assert start != -1, f"case {subject} in ... not found"
    end = source.find("esac", start)
    assert end != -1, f"unterminated case {subject}"
    return source[start:end]


def _ps_block(source: str, header: str, open_ch: str, close_ch: str) -> str:
    """Return the balanced `header <open> ... <close>` block from a PowerShell file."""
    start = source.find(header)
    assert start != -1, f"{header} not found"
    i = source.find(open_ch, start)
    assert i != -1
    depth = 0
    while i < len(source):
        if source[i] == open_ch:
            depth += 1
        elif source[i] == close_ch:
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {header}")


def _strip_sh_comment(line: str) -> str:
    """Drop a trailing `# ...` comment. Safe here: no table line contains a '#'
    inside a pattern."""
    return line.split("#", 1)[0]


# ── Table 1: gfx -> AMD index family ─────────────────────────────────────────


def _gfx_family_map_sh() -> dict[str, str]:
    body = _sh_function_body(_INSTALL_SH.read_text(encoding = "utf-8"), "_amd_arch_index_family_for_gfx")
    out: dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"\s*(gfx[^)]*)\)\s*echo\s+(\S+)\s*;;", _strip_sh_comment(line))
        if not m:
            continue
        for arch in m.group(1).split("|"):
            out[arch.strip()] = m.group(2).strip()
    return out


def _gfx_family_map_ps(path: Path) -> dict[str, str]:
    block = _ps_block(path.read_text(encoding = "utf-8"), "$archFamilyMap = @{", "{", "}")
    out: dict[str, str] = {}
    for line in block.splitlines():
        for m in re.finditer(r'"(gfx[0-9a-z]+)"\s*=\s*"([A-Za-z0-9-]+)"', _strip_sh_comment(line)):
            out[m.group(1)] = m.group(2)
    return out


def _gfx_family_maps() -> dict[str, dict[str, str]]:
    return {
        "studio/install_python_stack.py": dict(stack_mod._GFX_TO_AMD_INDEX_ARCH),
        "install.sh": _gfx_family_map_sh(),
        "install.ps1": _gfx_family_map_ps(_INSTALL_PS1),
        "studio/setup.ps1": _gfx_family_map_ps(_SETUP_PS1),
    }


class TestGfxIndexFamilyParity:
    """All four gfx -> AMD index family maps must agree, entry for entry."""

    def test_every_copy_is_non_empty(self):
        for where, table in _gfx_family_maps().items():
            assert table, f"{where}: parsed an empty gfx -> index family map (table moved or renamed?)"

    def test_all_copies_identical(self):
        maps = _gfx_family_maps()
        reference_name = "studio/install_python_stack.py"
        reference = maps[reference_name]
        for where, table in maps.items():
            if where == reference_name:
                continue
            missing = {k: v for k, v in reference.items() if k not in table}
            extra = {k: v for k, v in table.items() if k not in reference}
            wrong = {k: (v, reference[k]) for k, v in table.items() if k in reference and v != reference[k]}
            assert not missing, f"{where} is missing {sorted(missing)} (present in {reference_name})"
            assert not extra, f"{where} has {sorted(extra)} that {reference_name} does not"
            assert not wrong, f"{where} maps {wrong} (value, expected)"

    def test_rdna2_family_present_everywhere(self):
        """#7277 added gfx1030-1036 to three files; install.sh followed later.
        Pin the whole RDNA2 range so the next family lands everywhere at once."""
        for where, table in _gfx_family_maps().items():
            for arch in ("gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"):
                assert table.get(arch) == "gfx103X-all", f"{where}: {arch} -> {table.get(arch)!r}"


class TestSupportedWheelArchList:
    """setup.ps1's $_rocmWheelArches decides whether a detected arch gets ROCm torch
    at all. An arch present in the family map but absent here silently installs
    CPU-only PyTorch (the 'not in supported arch list' report from r/unsloth)."""

    def test_wheel_arch_list_covers_every_mapped_arch(self):
        block = _ps_block(_SETUP_PS1.read_text(encoding = "utf-8"), "$_rocmWheelArches = @(", "(", ")")
        listed = set(re.findall(r'"(gfx[0-9a-z]+)"', block))
        assert listed, "could not parse $_rocmWheelArches"
        mapped = set(stack_mod._GFX_TO_AMD_INDEX_ARCH)
        assert mapped - listed == set(), (
            f"studio/setup.ps1 $_rocmWheelArches is missing {sorted(mapped - listed)}: "
            "those arches map to an AMD index but would still fall back to CPU torch"
        )


# ── Table 2: GPU marketing name -> gfx ───────────────────────────────────────
#
# Each copy is an ordered, first-match-wins table. The shell copies use case
# globs (case-sensitive); the PowerShell copies use -match regexes
# (case-insensitive, and the only place a negative lookahead is available).
# Rather than diff the patterns -- which legitimately differ in syntax -- run
# every copy against the same real GPU names and require the same answer.


def _name_table_sh_function(source: str, name: str) -> list[tuple[list[str], str]]:
    body = _sh_function_body(source, name)
    rows: list[tuple[list[str], str]] = []
    for line in body.splitlines():
        m = re.match(r"\s*(\*.*?)\)\s*echo\s+(gfx[0-9a-z]+)\s*;;", _strip_sh_comment(line))
        if m:
            rows.append(([p.strip() for p in m.group(1).split("|")], m.group(2)))
    return rows


def _name_table_setup_sh() -> list[tuple[list[str], str]]:
    block = _sh_case_block(_SETUP_SH.read_text(encoding = "utf-8"), '"$_setup_mkt"')
    rows: list[tuple[list[str], str]] = []
    for line in block.splitlines():
        m = re.match(r'\s*(\*.*?)\)\s*_setup_gfx="(gfx[0-9a-z]+)"\s*;;', _strip_sh_comment(line))
        if m:
            rows.append(([p.strip() for p in m.group(1).split("|")], m.group(2)))
    return rows


def _name_table_ps(path: Path) -> list[tuple[str, str]]:
    block = _ps_block(path.read_text(encoding = "utf-8"), "$nameArchTable = @(", "(", ")")
    return re.findall(r'@\{\s*P\s*=\s*"([^"]+)"\s*;\s*A\s*=\s*"(gfx[0-9a-z]+)"\s*\}', block)


def _match_sh(rows: list[tuple[list[str], str]], gpu_name: str) -> str | None:
    """Evaluate a shell `case` table: first arm whose glob matches wins."""
    for patterns, arch in rows:
        for pattern in patterns:
            # Shell case globs quote literal segments: *"RX 7900"* -> *RX 7900*
            if fnmatch.fnmatchcase(gpu_name, pattern.replace('"', "")):
                return arch
    return None


def _match_ps(rows: list[tuple[str, str]], gpu_name: str) -> str | None:
    """Evaluate a PowerShell -match table: first arm whose regex matches wins.
    -match is case-insensitive; .NET and Python agree on these patterns
    (alternation plus one negative lookahead)."""
    for pattern, arch in rows:
        if re.search(pattern, gpu_name, re.IGNORECASE):
            return arch
    return None


# Real strings as amd-smi / rocm-smi / WMI report them, one per arch the tables
# claim to cover, including the two ordering traps: "RX 9070 XT" must beat the
# bare "9070" arm, and "RX 7700S" must beat the "RX 7700" arm.
_GPU_NAME_CASES = [
    ("AMD Radeon RX 9070 XT", "gfx1201"),
    ("AMD Radeon RX 9070", "gfx1200"),
    ("AMD Radeon RX 9060 XT", "gfx1200"),
    ("AMD Radeon 8060S Graphics", "gfx1151"),
    ("AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics", "gfx1151"),
    ("AMD Radeon 890M Graphics", "gfx1150"),
    ("AMD Radeon 880M Graphics", "gfx1150"),
    ("AMD Radeon RX 7900 XTX", "gfx1100"),
    ("AMD Radeon RX 7800 XT", "gfx1100"),
    ("AMD Radeon PRO W7900", "gfx1100"),
    ("AMD Radeon RX 7700S", "gfx1102"),
    ("AMD Radeon RX 7600 XT", "gfx1102"),
    ("AMD Radeon 780M Graphics", "gfx1103"),
    ("AMD Radeon RX 6900 XT", "gfx1030"),
    ("AMD Radeon RX 6700 XT", "gfx1030"),
    ("AMD Radeon RX 6600 XT", "gfx1032"),
    ("AMD Radeon RX 6500 XT", "gfx1034"),
]


def _name_tables() -> dict[str, object]:
    install_sh = _INSTALL_SH.read_text(encoding = "utf-8")
    return {
        "install.sh": _name_table_sh_function(install_sh, "_infer_amd_gfx_arch_from_gpu_name"),
        "studio/setup.sh": _name_table_setup_sh(),
        "install.ps1": _name_table_ps(_INSTALL_PS1),
        "studio/setup.ps1": _name_table_ps(_SETUP_PS1),
    }


class TestGpuNameArchParity:
    """All four name -> gfx tables must resolve the same GPU to the same arch."""

    def test_every_copy_is_non_empty(self):
        for where, rows in _name_tables().items():
            assert rows, f"{where}: parsed an empty name -> gfx table (table moved or renamed?)"

    @pytest.mark.parametrize("gpu_name,expected", _GPU_NAME_CASES)
    def test_all_copies_agree(self, gpu_name, expected):
        for where, rows in _name_tables().items():
            got = _match_sh(rows, gpu_name) if where.endswith(".sh") else _match_ps(rows, gpu_name)
            assert got == expected, f"{where}: {gpu_name!r} -> {got!r}, expected {expected!r}"

    def test_unknown_name_matches_nothing_anywhere(self):
        """An unrecognised card must fall through to the CPU path in every copy,
        never onto a neighbouring arm."""
        for where, rows in _name_tables().items():
            got = _match_sh(rows, "NVIDIA GeForce RTX 4090") if where.endswith(".sh") else _match_ps(rows, "NVIDIA GeForce RTX 4090")
            assert got is None, f"{where}: RTX 4090 matched {got!r}"

    def test_inferred_arch_always_has_an_index_family(self):
        """Every arch a name table can produce must be routable to an AMD wheel
        index, else detection succeeds and the install still lands on CPU torch."""
        families = stack_mod._GFX_TO_AMD_INDEX_ARCH
        for where, rows in _name_tables().items():
            arches = {arch for _, arch in rows} if where.endswith(".ps1") else {arch for _, arch in rows}
            for arch in arches:
                assert arch in families, f"{where}: {arch} has no entry in _GFX_TO_AMD_INDEX_ARCH"


# ── Table 3: the torch>=2.11 pin allowlist ───────────────────────────────────


class TestTorch211PinAllowlistParity:
    """gfx120X-all / gfx1151 / gfx1150 (and rocm7.2) ship the null _grouped_mm
    kernel below torch 2.11, so all three installers must raise the same floor.
    A leaf missing from one copy reintroduces the crash on that path."""

    _EXPECTED = {"gfx120x-all", "gfx1151", "gfx1150"}

    def test_install_sh_pins_the_same_leaves(self):
        source = _INSTALL_SH.read_text(encoding = "utf-8")
        idx = source.find('case "$_torch_index_leaf" in')
        assert idx != -1
        arm = re.search(r"\n\s*(rocm7\.2\|[^)]*)\)", source[idx:])
        assert arm, "torch 2.11 pin arm not found in install.sh"
        leaves = {leaf.strip() for leaf in arm.group(1).split("|")}
        assert self._EXPECTED <= leaves, f"install.sh pin arm missing {sorted(self._EXPECTED - leaves)}"
        assert "rocm7.2" in leaves

    def test_install_ps1_pins_the_same_leaves(self):
        source = _INSTALL_PS1.read_text(encoding = "utf-8")
        m = re.search(r"\$_pinGfx211\s*=\s*@\(([^)]*)\)", source)
        assert m, "$_pinGfx211 not found in install.ps1"
        leaves = set(re.findall(r"'([^']+)'", m.group(1)))
        assert leaves == self._EXPECTED, f"install.ps1 pins {sorted(leaves)}"

    def test_setup_ps1_pins_the_same_leaves(self):
        source = _SETUP_PS1.read_text(encoding = "utf-8")
        m = re.search(r"return\s+@\(([^)]*)\)\s*-contains\s*\$Leaf", source)
        assert m, "the 2.11 pin allowlist helper was not found in studio/setup.ps1"
        leaves = set(re.findall(r"'([^']+)'", m.group(1)))
        assert leaves == self._EXPECTED, f"studio/setup.ps1 pins {sorted(leaves)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
