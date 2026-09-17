# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM is pinned by splitting rows, so the split must be a true partition.

Every package that needs a different version on win_arm64 is written as two rows:

    X==old ; sys_platform != "win32" or platform_machine != "ARM64"
    X>=new ; sys_platform == "win32" and platform_machine == "ARM64"

The second marker is the exact complement of the first, so in every environment exactly one
row is live. Getting that wrong is silent either way: an OVERLAP makes pip intersect two
specifiers and can render the row unsatisfiable, a GAP drops the package on some platform
nobody tested. The compare is case-sensitive, which is what isolates Windows on ARM (macOS
reports ``arm64`` and Linux ``aarch64``), so that is asserted here too.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FILES = [
    REPO_ROOT / "studio/backend/requirements/extras.txt",
    REPO_ROOT / "studio/backend/requirements/no-torch-runtime.txt",
    REPO_ROOT / "studio/backend/requirements/single-env/constraints.txt",
    REPO_ROOT / "studio/backend/requirements/studio.txt",
]

# (sys_platform, platform_system, platform_machine, os_name)
PLATFORMS = [
    ("linux", "Linux", "x86_64", "posix"),
    ("linux", "Linux", "aarch64", "posix"),
    ("linux", "Linux", "armv7l", "posix"),
    ("linux", "Linux", "ppc64le", "posix"),
    ("linux", "Linux", "s390x", "posix"),
    ("darwin", "Darwin", "arm64", "posix"),
    ("darwin", "Darwin", "x86_64", "posix"),
    ("win32", "Windows", "AMD64", "nt"),
    ("win32", "Windows", "x86", "nt"),
    ("win32", "Windows", "ARM64", "nt"),
]
PYTHONS = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
WOA = ("win32", "Windows", "ARM64", "nt")


def _env(plat, py):
    sys_platform, platform_system, platform_machine, os_name = plat
    return {
        "implementation_name": "cpython",
        "implementation_version": f"{py}.0",
        "os_name": os_name,
        "platform_machine": platform_machine,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": platform_system,
        "platform_version": "",
        "python_full_version": f"{py}.0",
        "python_version": py,
        "sys_platform": sys_platform,
        "extra": "",
    }


#: Every environment the rows below are evaluated in, built once.
ENVS = [(plat, py, _env(plat, py)) for plat, py in itertools.product(PLATFORMS, PYTHONS)]


def _live(rows, env):
    """The rows pip would install in `env`."""
    return [r for r in rows if r.marker is None or r.marker.evaluate(env)]


def _rows(path: Path) -> list[Requirement]:
    out = []
    for raw in path.read_text(encoding = "utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(Requirement(line))
    return out


def _by_name(reqs: list[Requirement]) -> dict[str, list[Requirement]]:
    grouped: dict[str, list[Requirement]] = {}
    for req in reqs:
        grouped.setdefault(req.name.lower().replace("_", "-"), []).append(req)
    return grouped


def _multi_row_groups(reqs):
    """Packages stated more than once, minus the ones that are different targets.

    unsloth[a] and unsloth[b] may legitimately co-exist, so a group whose rows differ in
    their extras is not a platform split at all.
    """
    for name, group in _by_name(reqs).items():
        if len(group) < 2 or len({tuple(sorted(r.extras)) for r in group}) > 1:
            continue
        yield name, group


def _markers(group) -> list[str]:
    return [str(r.marker).replace("'", '"') for r in group if r.marker is not None]


def _pyproject_extras() -> dict[str, list[Requirement]]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - 3.10 runs use tomli
        import tomli as tomllib
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))
    out = {}
    for extra, lines in data["project"]["optional-dependencies"].items():
        out[extra] = [Requirement(l) for l in lines]
    return out


ALL_SOURCES: list[tuple[str, list[Requirement]]] = [(p.name, _rows(p)) for p in REQ_FILES] + [
    (f"pyproject[{k}]", v) for k, v in _pyproject_extras().items()
]
per_source = pytest.mark.parametrize("label,reqs", ALL_SOURCES, ids = [s[0] for s in ALL_SOURCES])


@per_source
def test_split_rows_never_overlap(label, reqs):
    """Two rows for one package must never both be live: pip would intersect them."""
    for name, group in _multi_row_groups(reqs):
        for plat, py, env in ENVS:
            live = _live(group, env)
            assert len(live) <= 1, (
                f"{label}: {name} has {len(live)} live rows on "
                f"{plat[0]}/{plat[2]}/py{py}: {[str(r) for r in live]}"
            )


@per_source
def test_no_package_is_dropped_on_a_non_woa_platform(label, reqs):
    """A split may remove a package on Windows ARM64 only."""
    if "constraints" in label:
        return  # a constraints file may legitimately have no cap in force
    for name, group in _multi_row_groups(reqs):
        # Only the complement-pair shape; triton-windows is two disjoint Windows-only rows.
        markers = _markers(group)
        if not (
            any('platform_machine == "ARM64"' in m for m in markers)
            and any('platform_machine != "ARM64"' in m for m in markers)
        ):
            continue
        for plat, py, env in ENVS:
            if plat == WOA:
                continue
            assert _live(
                group, env
            ), f"{label}: {name} has no live row on {plat[0]}/{plat[2]}/py{py}"


def test_arm64_marker_is_case_sensitive_and_windows_only():
    """``ARM64`` must not match macOS ``arm64`` or Linux ``aarch64``."""
    woa = Requirement('x==1; sys_platform == "win32" and platform_machine == "ARM64"')
    # The complement really is the complement.
    other = Requirement('x==1; sys_platform != "win32" or platform_machine != "ARM64"')
    for plat, py, env in ENVS:
        live = woa.marker.evaluate(env)
        assert live == (plat == WOA), f"win-ARM64 marker fired on {plat[0]}/{plat[2]}"
        assert live != other.marker.evaluate(
            env
        ), f"the two halves are not complementary on {plat[0]}/{plat[2]}"


@per_source
def test_no_row_is_dead_on_arrival(label, reqs):
    """Every row must be live in at least one real environment."""
    for req in reqs:
        if req.marker is None:
            continue
        assert any(req.marker.evaluate(env) for _, _, env in ENVS), (
            f"{label}: `{req}` is live in none of the {len(ENVS)} environments tested, so it "
            f'can never install. A lowercase "arm64" next to sys_platform == "win32" is the '
            f'usual cause: Windows reports "ARM64".'
        )


# Which packages carry a Windows-on-ARM row, PER SOURCE, and in which shape.
#   "split"   -- a positive `platform_machine == "ARM64"` row giving a different version
#   "dropped" -- only the negative row, so the package is absent on Windows on ARM
# Checked per source and by shape: studio.txt and pyproject[studio] mirror each other, so a global
# check stays green when one loses a row.
WOA_ROWS_BY_SOURCE = {
    "extras.txt": {"av": "split", "scikit-learn": "split"},
    "no-torch-runtime.txt": {"pymupdf": "split", "hf-transfer": "dropped", "sqlite-vec": "dropped"},
    "constraints.txt": {
        "av": "split",
        "cryptography": "split",
        "pandas": "split",
        "pyarrow": "split",
    },
    "studio.txt": {"cryptography": "split", "pandas": "split", "pymupdf": "split"},
    "pyproject[studio]": {"cryptography": "split", "pandas": "split", "pymupdf": "split"},
    "pyproject[triton]": {"triton-windows": "split"},
    "pyproject[huggingfacenotorch]": {"hf-transfer": "dropped"},
    "pyproject[windows]": {"xformers": "dropped"},
}


@pytest.mark.parametrize("label,expected", sorted(WOA_ROWS_BY_SOURCE.items()))
def test_the_woa_split_is_used_where_we_claim_it_is(label, expected):
    """Guard against a Windows-on-ARM row silently disappearing in a future edit."""
    groups = _by_name(dict(ALL_SOURCES)[label])
    for name, shape in sorted(expected.items()):
        markers = _markers(groups.get(name, []))
        positive = [m for m in markers if 'platform_machine == "ARM64"' in m]
        negative = [m for m in markers if 'platform_machine != "ARM64"' in m]
        if shape == "split":
            assert positive, f"{label}: {name} lost its Windows-on-ARM row"
        else:
            assert negative and not positive, (
                f"{label}: {name} should be excluded on Windows on ARM by a negative "
                f"marker only; found positive={positive}"
            )


# A package's OWN requires-python floor, for rows pinning into a range that does not exist for
# every interpreter. Only floors above our own 3.9 can make a row unsatisfiable.
PACKAGE_PYTHON_FLOORS = {
    "pandas": [(SpecifierSet(">=3.0"), (3, 11))],
}


def _minor(py: str) -> tuple:
    major, minor = py.split(".")
    return (int(major), int(minor))


def _lowest_allowed(req) -> str:
    """The smallest concrete version the row's specifier admits, for floor comparison."""
    lowers = [s.version for s in req.specifier if s.operator in (">=", "==", "~=", ">")]
    return lowers[0] if lowers else "0"


@per_source
def test_a_selected_row_is_installable_on_the_python_it_was_selected_for(label, reqs):
    """Splitting on platform is not enough on its own: a row can be live for an interpreter
    that no release in its range supports, which is not a resolution failure anyone reads
    as a marker bug -- pip just reports that no version matches.
    """
    for plat, py, env in ENVS:
        for req in _live(reqs, env):
            for spec, floor in PACKAGE_PYTHON_FLOORS.get(req.name.lower(), ()):
                # Does this row admit ONLY versions that need a newer interpreter?
                if not spec.contains(_lowest_allowed(req), prereleases = True):
                    continue
                assert _minor(py) >= floor, (
                    f"{label}: `{req}` is live on Python {py} {plat[2]}, but every "
                    f"version it admits needs Python >= {floor[0]}.{floor[1]}. "
                    "The row is unsatisfiable there; the marker needs a "
                    "python_version bound as well as a platform one."
                )


def test_the_woa_pandas_split_covers_every_supported_python():
    """The complement of the test above: having added a python_version bound, no ARM64
    interpreter may be left with no pandas row at all.
    """
    for label, reqs in ALL_SOURCES:
        rows = [r for r in reqs if r.name.lower() == "pandas"]
        if not rows:
            continue
        for py in PYTHONS:
            live = _live(rows, _env(WOA, py))
            assert len(live) == 1, (
                f"{label}: Windows ARM64 on Python {py} has {len(live)} live pandas "
                f"rows, expected exactly 1: {[str(r) for r in live]}"
            )
            if _minor(py) < (3, 11):
                assert "3.0" not in str(
                    live[0].specifier
                ), f"{label}: Python {py} must not be handed the pandas 3 row"


# install_python_stack.py, loaded so the skip list is read rather than copied here.
_SPEC = importlib.util.spec_from_file_location(
    "_ips_marker_skiplist", REPO_ROOT / "studio" / "install_python_stack.py"
)
IPS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(IPS)
# Every name install_python_stack.py filters out of the requirements files on win_arm64.
WOA_SKIPPED = {IPS._canonical_dist_name(n) for n in IPS.WINDOWS_ARM64_SKIP_PACKAGES}


# Scoped to `studio` deliberately: it is the extra a Windows-on-ARM user installs. The other
# 190-odd are x64 recipes, so an ARM64 marker there would assert what they never promised.
WOA_INSTALLABLE_EXTRAS = ["studio"]
per_extra = pytest.mark.parametrize("extra", WOA_INSTALLABLE_EXTRAS, ids = WOA_INSTALLABLE_EXTRAS)


@per_extra
def test_a_skipped_package_is_not_left_live_in_an_extra(extra):
    """The runtime skip list cannot reach package METADATA, so the extra has to agree.

    `pip install "unsloth[studio]"` never runs the installer's filtering: it resolves
    pyproject's rows directly, so a row left live on win_arm64 for a package with no wheel
    and no buildable sdist there fails the install outright. sqlite-vec was exactly this.
    """
    live = [
        str(req)
        for req in _live(_pyproject_extras()[extra], _env(WOA, "3.13"))
        if IPS._canonical_dist_name(req.name) in WOA_SKIPPED
    ]
    assert not live, (
        f"pyproject[{extra}] leaves these live on Windows ARM64 even though the installer "
        f"treats them as unavailable there, so `pip install unsloth[{extra}]` cannot "
        f"resolve: {live}"
    )


@per_extra
def test_dropping_a_package_on_woa_drops_it_nowhere_else(extra):
    """A negative ARM64 marker is a scalpel: every other platform keeps the row.

    Checked as an outcome rather than a spelling, over the same platform table the rest of
    this file uses, so a marker that reads correctly but excludes (say) Windows x86 as well
    is still caught.
    """
    for req in _pyproject_extras()[extra]:
        if IPS._canonical_dist_name(req.name) not in WOA_SKIPPED or req.marker is None:
            continue
        for plat, py, env in ENVS:
            if plat == WOA:
                continue
            assert req.marker.evaluate(env), (
                f"pyproject[{extra}] {req.name} is dropped on {plat[0]}/{plat[2]}/"
                f"py{py} too, which is not what the ARM64 marker is for"
            )
