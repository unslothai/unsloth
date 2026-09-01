# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drift guards for the AMD gfx tables that are duplicated across the installers.

The same three tables are hand-copied into up to seven places each:

  gfx -> AMD index family   install.sh (_amd_arch_index_family_for_gfx)
                            install.ps1 ($archFamilyMap)
                            studio/setup.ps1 ($archFamilyMap)
                            studio/install_python_stack.py (_GFX_TO_AMD_INDEX_ARCH)

  GPU name -> gfx           install.sh (_infer_amd_gfx_arch_from_gpu_name)
                            install.sh (case "$_gpu_disp_mkt", detection banner + env tip)
                            studio/setup.sh (_setup_supported_gfx_from_name)
                            install.ps1 ($nameArchTable)
                            studio/setup.ps1 ($nameArchTable)
                            studio/install_python_stack.py (_WIN_GPU_NAME_ARCH_TABLE)
                            tests/_zoo_rocm_spoof.py (_PROFILES, inverted gfx -> name)

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

Counting the copies by hand is itself unreliable -- the in-code "kept in sync
with" comments claimed four when there were seven -- so TestNoUnregisteredArchTable
below rediscovers them by scanning the repo instead of trusting this list.
"""

import ast
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
_PREBUILT_PY = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
_SPOOF_PY = PACKAGE_ROOT / "tests" / "_zoo_rocm_spoof.py"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location("studio_install_python_stack_parity", _STACK_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()


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


def _gfx_family_map_sh() -> dict[str, str]:
    body = _sh_function_body(
        _INSTALL_SH.read_text(encoding = "utf-8"), "_amd_arch_index_family_for_gfx"
    )
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
            assert (
                table
            ), f"{where}: parsed an empty gfx -> index family map (table moved or renamed?)"

    def test_all_copies_identical(self):
        maps = _gfx_family_maps()
        reference_name = "studio/install_python_stack.py"
        reference = maps[reference_name]
        for where, table in maps.items():
            if where == reference_name:
                continue
            missing = {k: v for k, v in reference.items() if k not in table}
            extra = {k: v for k, v in table.items() if k not in reference}
            wrong = {
                k: (v, reference[k])
                for k, v in table.items()
                if k in reference and v != reference[k]
            }
            assert (
                not missing
            ), f"{where} is missing {sorted(missing)} (present in {reference_name})"
            assert not extra, f"{where} has {sorted(extra)} that {reference_name} does not"
            assert not wrong, f"{where} maps {wrong} (value, expected)"

    def test_rdna2_family_present_everywhere(self):
        """#7277 added gfx1030-1036 to three files; install.sh followed later.
        Pin the whole RDNA2 range so the next family lands everywhere at once."""
        for where, table in _gfx_family_maps().items():
            for arch in (
                "gfx1030",
                "gfx1031",
                "gfx1032",
                "gfx1033",
                "gfx1034",
                "gfx1035",
                "gfx1036",
            ):
                assert table.get(arch) == "gfx103X-all", f"{where}: {arch} -> {table.get(arch)!r}"


class TestSupportedWheelArchList:
    """setup.ps1's $_rocmWheelArches decides whether a detected arch gets ROCm torch
    at all. An arch present in the family map but absent here silently installs
    CPU-only PyTorch (the 'not in supported arch list' report from r/unsloth)."""

    def test_wheel_arch_list_covers_every_mapped_arch(self):
        block = _ps_block(
            _SETUP_PS1.read_text(encoding = "utf-8"), "$_rocmWheelArches = @(", "(", ")"
        )
        listed = set(re.findall(r'"(gfx[0-9a-z]+)"', block))
        assert listed, "could not parse $_rocmWheelArches"
        mapped = set(stack_mod._GFX_TO_AMD_INDEX_ARCH)
        assert mapped - listed == set(), (
            f"studio/setup.ps1 $_rocmWheelArches is missing {sorted(mapped - listed)}: "
            "those arches map to an AMD index but would still fall back to CPU torch"
        )


# Table 2: GPU marketing name -> gfx ─────────────────────────────────────── Each copy is an ordered, first-match-wins
def _name_table_sh_function(source: str, name: str) -> list[tuple[list[str], str]]:
    body = _sh_function_body(source, name)
    rows: list[tuple[list[str], str]] = []
    for line in body.splitlines():
        m = re.match(r"\s*(\*.*?)\)\s*echo\s+(gfx[0-9a-z]+)\s*;;", _strip_sh_comment(line))
        if m:
            rows.append(([p.strip() for p in m.group(1).split("|")], m.group(2)))
    return rows


def _name_table_sh_case(source: str, subject: str, var: str) -> list[tuple[list[str], str]]:
    """A bare `case ... in` table that assigns to a variable rather than echoing."""
    block = _sh_case_block(source, subject)
    rows: list[tuple[list[str], str]] = []
    for line in block.splitlines():
        m = re.match(
            rf'\s*(\*.*?)\)\s*{re.escape(var)}="(gfx[0-9a-z]+)"\s*;;', _strip_sh_comment(line)
        )
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


# Real strings as amd-smi / rocm-smi / WMI report them, including the two ordering traps: "RX 9070 XT" must beat the
# bare "9070" arm, and "RX 7700S" must beat the "RX 7700" arm.
_GPU_NAME_LEAF_CASES = [
    ("AMD Radeon RX 9070 XT", "gfx120X-all"),
    ("AMD Radeon RX 9070", "gfx120X-all"),
    # Workstation Navi 48, gfx1201 per rocminfo in #7624 / #7307.
    # Its name holds neither "9070" nor "9080", so every table returned None and a host without the HIP SDK, where name
    # inference is the only path left, got CPU torch ("not detected", PR #8398).
    ("AMD Radeon AI PRO R9700", "gfx120X-all"),
    ("AMD Radeon RX 9060 XT", "gfx120X-all"),
    ("AMD Radeon 8060S Graphics", "gfx1151"),
    ("AMD Ryzen AI Max+ 395 w/ Radeon 8060S Graphics", "gfx1151"),
    ("AMD Radeon 890M Graphics", "gfx1150"),
    ("AMD Radeon 880M Graphics", "gfx1150"),
    ("AMD Radeon 860M Graphics", "gfx1152"),
    ("AMD Radeon 840M Graphics", "gfx1152"),
    ("AMD Ryzen AI 7 350 w/ Radeon 860M", "gfx1152"),
    ("AMD Radeon RX 7900 XTX", "gfx110X-all"),
    ("AMD Radeon RX 7800 XT", "gfx110X-all"),
    ("AMD Radeon PRO W7900", "gfx110X-all"),
    ("AMD Radeon RX 7700S", "gfx110X-all"),
    ("AMD Radeon RX 7600 XT", "gfx110X-all"),
    ("AMD Radeon 780M Graphics", "gfx110X-all"),
    ("AMD Radeon RX 6900 XT", "gfx103X-all"),
    ("AMD Radeon RX 6700 XT", "gfx103X-all"),
    ("AMD Radeon RX 6600 XT", "gfx103X-all"),
    ("AMD Radeon RX 6500 XT", "gfx103X-all"),
]

# Exact gfx ids, transcribed from AMD's ROCm compatibility matrix (the "Radeon GPU" list at
# rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html), NOT from the installer tables.
# Three of these were wrong until the commit that added this table: RX 9070 (non-XT) said gfx1200, RX 7800 XT / 7700 XT
# / PRO W7700 said gfx1100, and PRO V710 said gfx1102.
# The APU rows were added after that: Krackan Point (860M / 840M) said gfx1150 but is gfx1152, and unlike the three
# above that one DID change the wheel, since gfx1150 and gfx1152 are separate index leaves on repo.amd.com.
_AMD_DOCUMENTED_ARCH = {
    # RDNA 4 -- Navi 48 is gfx1201, Navi 44 is gfx1200.
    "AMD Radeon RX 9070 XT": "gfx1201",
    "AMD Radeon RX 9070 GRE": "gfx1201",
    "AMD Radeon RX 9070": "gfx1201",
    "AMD Radeon RX 9060 XT": "gfx1200",
    "AMD Radeon RX 9060": "gfx1200",
    # Navi 48 again, as the R9000 series workstation card.
    # own rocminfo output (#7624, #7307), not from these tables.
    "AMD Radeon AI PRO R9700": "gfx1201",
    "AMD Radeon RX 7900 XTX": "gfx1100",
    "AMD Radeon PRO W7900": "gfx1100",
    "AMD Radeon PRO W7800": "gfx1100",
    "AMD Radeon RX 7800 XT": "gfx1101",
    "AMD Radeon RX 7700 XT": "gfx1101",
    "AMD Radeon PRO W7700": "gfx1101",
    "AMD Radeon PRO V710": "gfx1101",
    "AMD Radeon RX 7600 XT": "gfx1102",
    "AMD Radeon RX 7700S": "gfx1102",
    "AMD Radeon PRO W7600": "gfx1102",
    # RDNA 3.5 APUs
    # Strix Point is gfx1150, Krackan Point (860M/840M) is gfx1152, per AMD's own lemonade GPU table
    # (src/cpp/server/system_info.cpp).
    "AMD Radeon 8060S Graphics": "gfx1151",
    "AMD Radeon 890M Graphics": "gfx1150",
    "AMD Radeon 880M Graphics": "gfx1150",
    "AMD Radeon 860M Graphics": "gfx1152",
    "AMD Radeon 840M Graphics": "gfx1152",
}


def _name_tables() -> dict[str, object]:
    install_sh = _INSTALL_SH.read_text(encoding = "utf-8")
    return {
        "install.sh:_infer_amd_gfx_arch_from_gpu_name": _name_table_sh_function(
            install_sh, "_infer_amd_gfx_arch_from_gpu_name"
        ),
        # install.sh carries the table TWICE.
        # The second copy drives the detection banner and, more importantly, the "Tip: set UNSLOTH_ROCM_GFX_ARCH=<arch>"
        # line, so a wrong id there gets pasted into a user's environment where it becomes authoritative.
        "install.sh:_gpu_disp_gfx": _name_table_sh_case(
            install_sh, '"$_gpu_disp_mkt"', "_gpu_disp_gfx"
        ),
        "studio/setup.sh": _name_table_sh_case(
            _SETUP_SH.read_text(encoding = "utf-8"), '"$_sup_gfx_in"', "_sup_gfx_out"
        ),
        "install.ps1": _name_table_ps(_INSTALL_PS1),
        "studio/setup.ps1": _name_table_ps(_SETUP_PS1),
        "studio/install_python_stack.py": list(stack_mod._WIN_GPU_NAME_ARCH_TABLE),
        # The backend carries the seventh copy: it decides whether a Windows adapter the
        # DirectX registry did not give an AdapterFamily is one a repair could help, and
        # answering that from a stale table would offer the repair to a card no wheel
        # index covers (or withhold it from one that is covered).
        "studio/backend/utils/hardware/hardware.py": _name_table_py_literal(
            PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py",
            "_GPU_NAME_GFX_TABLE",
        ),
    }


def _name_table_py_literal(path: Path, name: str) -> list:
    """A module-level list-of-pairs literal, read without importing the module."""
    import ast

    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in tree.body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else node.targets
            if isinstance(node, ast.Assign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return [tuple(pair) for pair in ast.literal_eval(node.value)]
    raise AssertionError(f"{name} not found in {path}")


def _spoof_profiles() -> dict[str, str]:
    """gfx -> marketing name out of tests/_zoo_rocm_spoof.py::_PROFILES.

    Parsed with ast rather than imported: that module spoofs torch.cuda and the
    AMD identity as an import side effect, which would poison every test sharing
    the process."""
    tree = ast.parse(_SPOOF_PY.read_text(encoding = "utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if target is not None and getattr(target, "id", "") == "_PROFILES":
            return {gfx: value[0] for gfx, value in ast.literal_eval(node.value).items()}
    raise AssertionError("_PROFILES not found in tests/_zoo_rocm_spoof.py")


# The spoof fixture states the mapping backwards (gfx -> the name torch should report), so it is the one copy written
# from the hardware's point of view instead of the installer's.
# That makes it a useful independent witness: it had gfx1101 -> "RX 7800 XT" and gfx1201 -> "RX 9070 XT" correct while
# all six installer copies were wrong, and nothing compared the two.
# RX 6700 XT is a known, deliberate divergence rather than drift.
# AMD's compatibility matrix documents no consumer RX 6000 card and no gfx1031 at all (only "AMD Radeon PRO W6800
# (gfx1030)"), the installer arm is commented "gfx103X family", and no code consumes the exact id
_SPOOF_DIVERGENCES = {
    "gfx1031": "installers group Navi 22 into the gfx1030 arm; see comment above",
}


def _resolve(where: str, rows, gpu_name: str) -> str | None:
    """Shell copies are case globs; the PowerShell and Python copies are both
    ordered first-match regex tables evaluated case-insensitively, so _match_ps
    models either one. `where` may be "<file>:<symbol>" for the files that carry
    the table more than once."""
    return (
        _match_sh(rows, gpu_name)
        if where.split(":")[0].endswith(".sh")
        else _match_ps(rows, gpu_name)
    )


class TestGpuNameArchParity:
    """All four name -> gfx tables must resolve the same GPU the same way."""

    def test_every_copy_is_non_empty(self):
        for where, rows in _name_tables().items():
            assert rows, f"{where}: parsed an empty name -> gfx table (table moved or renamed?)"

    @pytest.mark.parametrize("gpu_name", [name for name, _ in _GPU_NAME_LEAF_CASES])
    def test_all_copies_return_the_same_arch(self, gpu_name):
        """The drift guard proper: no expected value, just agreement. This is what
        catches a table edited in one installer and not the other three, and it
        stays honest even where the shipped gfx id is itself wrong."""
        answers = {where: _resolve(where, rows, gpu_name) for where, rows in _name_tables().items()}
        distinct = set(answers.values())
        assert len(distinct) == 1, f"{gpu_name!r} resolves inconsistently: {answers}"
        assert distinct != {None}, f"{gpu_name!r} is not matched by any copy of the table"

    @pytest.mark.parametrize("gpu_name,expected_leaf", _GPU_NAME_LEAF_CASES)
    def test_every_copy_routes_to_the_right_wheel_index(self, gpu_name, expected_leaf):
        """What the tables are for. A wrong leaf is the user-visible failure:
        CPU-only torch, or a wheel built for the wrong ISA."""
        families = stack_mod._GFX_TO_AMD_INDEX_ARCH
        for where, rows in _name_tables().items():
            arch = _resolve(where, rows, gpu_name)
            assert arch is not None, f"{where}: {gpu_name!r} matched nothing"
            assert (
                families.get(arch) == expected_leaf
            ), f"{where}: {gpu_name!r} -> {arch} -> {families.get(arch)!r}, expected {expected_leaf!r}"

    @pytest.mark.parametrize("gpu_name,expected_arch", sorted(_AMD_DOCUMENTED_ARCH.items()))
    def test_every_copy_matches_amds_documented_arch(self, gpu_name, expected_arch):
        """The gfx id itself, against AMD's matrix rather than against a sibling
        copy of the same table. Agreement between five copies proves nothing if
        all five were transcribed from the same mistake."""
        for where, rows in _name_tables().items():
            arch = _resolve(where, rows, gpu_name)
            assert (
                arch == expected_arch
            ), f"{where}: {gpu_name!r} -> {arch!r}, AMD documents {expected_arch!r}"

    def test_unknown_name_matches_nothing_anywhere(self):
        """An unrecognised card must fall through to the CPU path in every copy,
        never onto a neighbouring arm."""
        for where, rows in _name_tables().items():
            got = _resolve(where, rows, "NVIDIA GeForce RTX 4090")
            assert got is None, f"{where}: RTX 4090 matched {got!r}"

    @pytest.mark.parametrize(
        "gpu_name",
        [
            "ATI Radeon 9700 PRO",
            "ATI Radeon 9800 PRO",
            "AMD Radeon R9 Fury X",
            "AMD Radeon Pro WX 9100",
        ],
    )
    def test_the_r9700_arm_does_not_swallow_older_cards(self, gpu_name):
        """The arm is spelled "R9700", not a bare "9700": ATI shipped a Radeon 9700 PRO in
        2002 and the loose token would hand that card RDNA 4 wheels. None of these pre-RDNA
        names may resolve to anything."""
        for where, rows in _name_tables().items():
            got = _resolve(where, rows, gpu_name)
            assert got is None, f"{where}: {gpu_name!r} matched {got!r}"

    def test_inferred_arch_always_has_an_index_family(self):
        """Every arch a name table can produce must be routable to an AMD wheel
        index, else detection succeeds and the install still lands on CPU torch."""
        families = stack_mod._GFX_TO_AMD_INDEX_ARCH
        for where, rows in _name_tables().items():
            for arch in {arch for _, arch in rows}:
                assert arch in families, f"{where}: {arch} has no entry in _GFX_TO_AMD_INDEX_ARCH"

    def test_every_documented_gpu_resolves_somewhere(self):
        """The reverse of the AMD check above. That one asks "do the tables get
        the documented cards right"; this asks "is a documented card missing
        entirely", which is a silent CPU fallback rather than a wrong id.

        This cannot notice a GPU AMD shipped that nobody transcribed into
        _AMD_DOCUMENTED_ARCH -- doing that honestly would mean fetching AMD's
        matrix at test time, which makes the suite non-hermetic and offline
        runners fail. It does catch a card added to the ground-truth list, or to
        one installer, without the tables being completed."""
        for gpu_name in sorted(_AMD_DOCUMENTED_ARCH):
            for where, rows in _name_tables().items():
                assert (
                    _resolve(where, rows, gpu_name) is not None
                ), f"{where}: {gpu_name!r} matches no arm, so this card gets CPU-only torch"


class TestSpoofFixtureParity:
    """tests/_zoo_rocm_spoof.py is the seventh copy of the name/gfx mapping and
    was outside every drift guard. It is the fixture other ROCm tests build their
    fake AMD host from, so if it and the installers disagree, those tests exercise
    a machine that cannot exist."""

    def test_spoof_profiles_parse(self):
        profiles = _spoof_profiles()
        assert profiles, "parsed an empty _PROFILES (renamed or restructured?)"
        assert all(gfx.startswith("gfx") for gfx in profiles), profiles

    def test_spoof_names_resolve_back_to_their_own_arch(self):
        """Round-trip: feed each spoofed marketing name through the installer
        tables and the answer must be the gfx the spoof claims to be emulating."""
        tables = _name_tables()
        for gfx, gpu_name in sorted(_spoof_profiles().items()):
            if gfx in _SPOOF_DIVERGENCES:
                continue
            for where, rows in tables.items():
                got = _resolve(where, rows, gpu_name)
                assert (
                    got == gfx
                ), f"{where}: spoof says {gfx} is {gpu_name!r}, installer says {got!r}"

    def test_divergences_are_real_and_still_diverging(self):
        """Keeps the exception list from going stale: if the installers are
        corrected later, this fails and the entry has to be removed rather than
        quietly suppressing a check that now passes."""
        tables = _name_tables()
        profiles = _spoof_profiles()
        for gfx in _SPOOF_DIVERGENCES:
            assert gfx in profiles, f"{gfx} is exempted but no longer in the spoof"
            answers = {_resolve(w, r, profiles[gfx]) for w, r in tables.items()}
            assert answers != {gfx}, f"{gfx} now agrees everywhere; drop it from _SPOOF_DIVERGENCES"


# A table line names a card and gives its arch.
_MKT_NAME = re.compile(r"(RX\s*\d{4}|PRO\s*[WV]\d{3,4}|\b90[5-8]0\b)", re.IGNORECASE)
_GFX_ID = re.compile(r"gfx1[0-2][0-9a-z]{1,2}")

# Skip dirs of third-party or generated code;
_SCAN_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "build", "dist", "__pycache__"}

# Every file allowed to carry a name/arch table, as a repo-relative posix path.
_REGISTERED_TABLE_FILES = {
    "install.sh",
    "install.ps1",
    "studio/setup.sh",
    "studio/setup.ps1",
    "studio/install_python_stack.py",
    "studio/backend/utils/hardware/hardware.py",
    "tests/_zoo_rocm_spoof.py",
}

# Three or more such lines means a table.
# One or two means prose: the two known single-line hits are comments ("Verified on gfx1151 (Radeon 8060S)" in
# scripts/install_rocm_wsl_strixhalo.sh, and a parenthetical in studio/install_llama_prebuilt.py).
_TABLE_LINE_THRESHOLD = 3


def _under_cargo_output(path: Path, root: Path) -> bool:
    """Whether `path` sits inside a Cargo `target/` directory.

    Not in _SCAN_SKIP_DIRS because "target" is too generic to skip by name alone, so
    the pairing with a sibling Cargo.toml is what identifies build output. tauri copies
    install.sh into studio/src-tauri/target/debug/, so without this the guard fails for
    anyone who ran `cargo build` before pytest, on their own build output rather than on
    a real copy. CI never saw it because it builds and tests in separate jobs.
    """
    for parent in path.parents:
        if parent == root.parent:
            break
        if parent.name == "target" and (parent.parent / "Cargo.toml").is_file():
            return True
    return False


def _files_carrying_a_name_arch_table(root: Path = PACKAGE_ROOT) -> dict[str, int]:
    found: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.suffix not in {".sh", ".ps1", ".py"} or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in _SCAN_SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if _under_cargo_output(path, root):
            continue
        # Tests that *assert* on the tables quote card names next to gfx ids by nature.
        # Fixtures like _zoo_rocm_spoof.py do not start with test_ and so stay in scope, which is how the seventh copy
        # surfaced.
        if path.name.startswith("test_"):
            continue
        try:
            text = path.read_text(encoding = "utf-8", errors = "ignore")
        except OSError:
            continue
        hits = sum(
            1 for line in text.splitlines() if _MKT_NAME.search(line) and _GFX_ID.search(line)
        )
        if hits >= _TABLE_LINE_THRESHOLD:
            found[rel] = hits
    return found


class TestNoUnregisteredArchTable:
    """The failure this whole file exists for is a copy of the table that nobody
    knew about. Enumerating the copies by hand is the same manual step that let
    them drift, so this rediscovers them from the source tree."""

    def test_scan_still_finds_the_known_copies(self):
        """Guards the guard: if the heuristic stops matching (patterns reformatted
        onto multiple lines, say), it would silently find nothing and pass."""
        found = _files_carrying_a_name_arch_table()
        missing = _REGISTERED_TABLE_FILES - set(found)
        assert not missing, f"scan no longer detects known tables in {sorted(missing)}"

    def test_no_unregistered_copies(self):
        found = _files_carrying_a_name_arch_table()
        extra = {rel: n for rel, n in found.items() if rel not in _REGISTERED_TABLE_FILES}
        assert not extra, (
            f"unregistered GPU-name/arch table(s): {extra}. Wire each into "
            f"_name_tables() (or the spoof check) and add it to "
            f"_REGISTERED_TABLE_FILES, so drift there fails CI too."
        )

    def test_cargo_build_output_is_skipped_but_a_plain_target_dir_is_not(self, tmp_path):
        """The skip is narrow on purpose: `target/` next to a Cargo.toml is build output,
        `target/` anywhere else is source and a copy hiding there still has to fail."""
        table = "\n".join(f"# RX 7{n}00 gfx1100" for n in range(1, 6)) + "\n"
        (tmp_path / "src-tauri" / "target" / "debug").mkdir(parents = True)
        (tmp_path / "src-tauri" / "Cargo.toml").write_text("[package]\n")
        (tmp_path / "src-tauri" / "target" / "debug" / "install.sh").write_text(table)
        (tmp_path / "scripts" / "target").mkdir(parents = True)
        (tmp_path / "scripts" / "target" / "install.sh").write_text(table)

        found = _files_carrying_a_name_arch_table(tmp_path)
        assert "scripts/target/install.sh" in found, (
            "a table under a plain target/ directory was skipped; the guard would miss "
            f"a real copy there: {found}"
        )
        assert (
            "src-tauri/target/debug/install.sh" not in found
        ), f"cargo build output is still scanned: {found}"


class TestTorch211PinAllowlistParity:
    """gfx120X-all / gfx1151 / gfx1150 / gfx1152 (and rocm7.2) ship the null
    _grouped_mm kernel below torch 2.11, so all three installers must raise the
    same floor. A leaf missing from one copy reintroduces the crash there."""

    _EXPECTED = {"gfx120x-all", "gfx1151", "gfx1150", "gfx1152"}

    def test_install_sh_pins_the_same_leaves(self):
        source = _INSTALL_SH.read_text(encoding = "utf-8")
        idx = source.find('case "$_torch_index_leaf" in')
        assert idx != -1
        arm = re.search(r"\n\s*(rocm7\.2\|[^)]*)\)", source[idx:])
        assert arm, "torch 2.11 pin arm not found in install.sh"
        leaves = {leaf.strip() for leaf in arm.group(1).split("|")}
        assert (
            self._EXPECTED <= leaves
        ), f"install.sh pin arm missing {sorted(self._EXPECTED - leaves)}"
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


class TestShadowingIntegratedGfxParity:
    """The shadowing-APU skip (#7776) exists twice: studio/setup.ps1 resolves the
    arch and builds $ROCmIndexUrl before it ever invokes the Python stack
    installer, so both copies of the list have to agree or one entry point keeps
    installing the iGPU's wheel family."""

    _STRIX = {"gfx1150", "gfx1151", "gfx1152"}

    def _setup_ps1_list(self):
        source = _SETUP_PS1.read_text(encoding = "utf-8")
        m = re.search(r"\$script:ShadowingIntegratedGfx\s*=\s*@\(([^)]*)\)", source)
        assert m, "$script:ShadowingIntegratedGfx not found in studio/setup.ps1"
        return set(re.findall(r'"([^"]+)"', m.group(1)))

    def _prebuilt_list(self):
        tree = ast.parse(_PREBUILT_PY.read_text(encoding = "utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SHADOWING_INTEGRATED_GFX" for t in node.targets
            ):
                return set(ast.literal_eval(node.value.args[0]))
        raise AssertionError("SHADOWING_INTEGRATED_GFX not found in install_llama_prebuilt.py")

    def test_setup_ps1_matches_install_python_stack(self):
        assert self._setup_ps1_list() == set(stack_mod._SHADOWING_INTEGRATED_GFX)

    def test_install_llama_prebuilt_matches_install_python_stack(self):
        # _apply_host_overrides() honours setup's repick only for these arches, so drift re-splits torch and llama.cpp
        assert self._prebuilt_list() == set(stack_mod._SHADOWING_INTEGRATED_GFX)

    def test_strix_is_excluded_from_every_copy(self):
        # Supported training targets, not shadowing APUs:
        assert not (self._STRIX & set(stack_mod._SHADOWING_INTEGRATED_GFX))
        assert not (self._STRIX & self._setup_ps1_list())
        assert not (self._STRIX & self._prebuilt_list())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
