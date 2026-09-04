# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM is pinned by splitting rows, so the split must be a true partition.

Every package that needs a different version on win_arm64 is written as two rows:

    X==old ; sys_platform != "win32" or platform_machine != "ARM64"
    X>=new ; sys_platform == "win32" and platform_machine == "ARM64"

The second marker is the exact complement of the first, so in every environment exactly
one row is live. Get that wrong in either direction and it is silent: an OVERLAP makes
pip intersect two specifiers and can render the row unsatisfiable, while a GAP drops the
package on some platform nobody tested.

The compare is case-sensitive, which is what isolates Windows on ARM: macOS reports
``arm64`` and Linux ``aarch64``, so only Windows' ``ARM64`` can match. That is load
bearing rather than incidental, so it is asserted here too.
"""

from __future__ import annotations

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


@pytest.mark.parametrize("label,reqs", ALL_SOURCES, ids = [s[0] for s in ALL_SOURCES])
def test_split_rows_never_overlap(label, reqs):
    """Two rows for one package must never both be live: pip would intersect them."""
    for name, group in _by_name(reqs).items():
        if len(group) < 2:
            continue
        # unsloth[a] and unsloth[b] are different targets and may legitimately co-exist.
        if len({tuple(sorted(r.extras)) for r in group}) > 1:
            continue
        for plat, py in itertools.product(PLATFORMS, PYTHONS):
            env = _env(plat, py)
            live = [r for r in group if r.marker is None or r.marker.evaluate(env)]
            assert len(live) <= 1, (
                f"{label}: {name} has {len(live)} live rows on "
                f"{plat[0]}/{plat[2]}/py{py}: {[str(r) for r in live]}"
            )


@pytest.mark.parametrize("label,reqs", ALL_SOURCES, ids = [s[0] for s in ALL_SOURCES])
def test_no_package_is_dropped_on_a_non_woa_platform(label, reqs):
    """A split may remove a package on Windows ARM64 only.

    hf_transfer and xformers are deliberately dropped there (no win_arm64 wheel, no
    buildable sdist). Anywhere else, a package that has rows must have a live one.

    Scoped to the groups this mechanism actually created -- a group holding a row with
    the ARM64 marker. Plenty of pre-existing rows are legitimately inactive on exotic
    platforms (bitsandbytes in the amd extra has nothing for linux/armv7l), and that is
    not what this test is about.
    """
    for name, group in _by_name(reqs).items():
        if len(group) < 2 or "constraints" in label:
            continue  # a constraints file may legitimately have no cap in force
        if len({tuple(sorted(r.extras)) for r in group}) > 1:
            continue
        # Only the complement-pair shape: one row keyed ON win-ARM64 and one keyed OFF
        # it. triton-windows is two disjoint Windows-only rows instead, and is correctly
        # absent on Linux, so it is not what this invariant is about.
        markers = [str(r.marker).replace("'", '"') for r in group if r.marker is not None]
        if not (
            any('platform_machine == "ARM64"' in m for m in markers)
            and any('platform_machine != "ARM64"' in m for m in markers)
        ):
            continue
        for plat, py in itertools.product(PLATFORMS, PYTHONS):
            if (plat[0], plat[2]) == ("win32", "ARM64"):
                continue
            env = _env(plat, py)
            live = [r for r in group if r.marker is None or r.marker.evaluate(env)]
            assert live, f"{label}: {name} has no live row on {plat[0]}/{plat[2]}/py{py}"


def test_arm64_marker_is_case_sensitive_and_windows_only():
    """``ARM64`` must not match macOS ``arm64`` or Linux ``aarch64``.

    This is the whole isolation mechanism. If the compare were case-insensitive, or if a
    row keyed on platform_machine without sys_platform, every Apple Silicon and Linux
    ARM host would take the Windows-on-ARM pins.
    """
    woa = Requirement('x==1; sys_platform == "win32" and platform_machine == "ARM64"')
    for plat in PLATFORMS:
        env = _env(plat, "3.13")
        live = woa.marker.evaluate(env)
        assert live == (
            (plat[0], plat[2]) == ("win32", "ARM64")
        ), f"win-ARM64 marker fired on {plat[0]}/{plat[2]}"
    # The complement really is the complement.
    other = Requirement('x==1; sys_platform != "win32" or platform_machine != "ARM64"')
    for plat, py in itertools.product(PLATFORMS, PYTHONS):
        env = _env(plat, py)
        assert woa.marker.evaluate(env) != other.marker.evaluate(
            env
        ), f"the two halves are not complementary on {plat[0]}/{plat[2]}"


@pytest.mark.parametrize("label,reqs", ALL_SOURCES, ids = [s[0] for s in ALL_SOURCES])
def test_no_row_is_dead_on_arrival(label, reqs):
    """Every row must be live in at least one real environment.

    This is the one mistake the scope-limited tests above cannot catch by construction:
    write ``arm64`` instead of ``ARM64`` next to ``sys_platform == "win32"`` and the
    conjunction becomes unsatisfiable, the split stops existing, and every check that
    looks for the split simply skips the group and passes. Windows on ARM then quietly
    takes the x64 pins and source-builds.

    Asked semantically rather than by matching marker text: a row like torchcodec's
    pairs ``platform_machine == 'arm64'`` with ``sys_platform == 'darwin'`` in a separate
    disjunct, which is correct, and only a genuinely unsatisfiable row is flagged.
    """
    for req in reqs:
        if req.marker is None:
            continue
        live_on = [
            (plat[0], plat[2], py)
            for plat, py in itertools.product(PLATFORMS, PYTHONS)
            if req.marker.evaluate(_env(plat, py))
        ]
        assert live_on, (
            f"{label}: `{req}` is live in none of the {len(PLATFORMS) * len(PYTHONS)} "
            f'environments tested, so it can never install. A lowercase "arm64" next '
            f'to sys_platform == "win32" is the usual cause: Windows reports "ARM64".'
        )


# Which packages carry a Windows-on-ARM row, PER SOURCE, and in which shape.
#   "split"   -- a positive `platform_machine == "ARM64"` row giving a different version
#   "dropped" -- only the negative row, so the package is absent on Windows on ARM
# Checked per source rather than globally on purpose: studio.txt and pyproject[studio]
# mirror each other, so a global check stays green when only one of them loses a row,
# which is exactly the drift the mirroring exists to prevent. And the shape matters:
# deleting a split's positive row leaves its negative twin behind, so "some row mentions
# ARM64" would still be satisfied while Windows on ARM had quietly lost its pin.
WOA_ROWS_BY_SOURCE = {
    "extras.txt": {"av": "split", "scikit-learn": "split"},
    "no-torch-runtime.txt": {"pymupdf": "split", "hf-transfer": "dropped"},
    "constraints.txt": {
        "av": "split",
        "cryptography": "split",
        "pandas": "split",
        "pyarrow": "split",
    },
    "studio.txt": {"cryptography": "split", "pandas": "split", "pymupdf": "split"},
    "pyproject[studio]": {
        "cryptography": "split",
        "pandas": "split",
        "pymupdf": "split",
    },
    "pyproject[triton]": {"triton-windows": "split"},
    "pyproject[huggingfacenotorch]": {"hf-transfer": "dropped"},
    "pyproject[windows]": {"xformers": "dropped"},
}


@pytest.mark.parametrize("label,expected", sorted(WOA_ROWS_BY_SOURCE.items()))
def test_the_woa_split_is_used_where_we_claim_it_is(label, expected):
    """Guard against a Windows-on-ARM row silently disappearing in a future edit."""
    groups = _by_name(dict(ALL_SOURCES)[label])
    for name, shape in sorted(expected.items()):
        markers = [
            str(r.marker).replace("'", '"') for r in groups.get(name, []) if r.marker is not None
        ]
        positive = [m for m in markers if 'platform_machine == "ARM64"' in m]
        negative = [m for m in markers if 'platform_machine != "ARM64"' in m]
        if shape == "split":
            assert positive, f"{label}: {name} lost its Windows-on-ARM row"
        else:
            assert negative and not positive, (
                f"{label}: {name} should be excluded on Windows on ARM by a negative "
                f"marker only; found positive={positive}"
            )


# A package's OWN requires-python floor, for rows whose specifier pins into a range that
# does not exist for every interpreter this project supports. Only entries where the floor
# is above our own 3.9 are worth listing; the rest cannot make a row unsatisfiable.
#
# Verified against PyPI when added. pandas 3.x is >=3.11 and its win_arm64 wheels start at
# cp311, which is exactly why the Windows-on-ARM row exists.
PACKAGE_PYTHON_FLOORS = {
    "pandas": [(SpecifierSet(">=3.0"), (3, 11))],
}


def _minor(py: str) -> tuple:
    major, minor = py.split(".")
    return (int(major), int(minor))


@pytest.mark.parametrize("label,reqs", ALL_SOURCES, ids = [s[0] for s in ALL_SOURCES])
def test_a_selected_row_is_installable_on_the_python_it_was_selected_for(label, reqs):
    """
    Splitting on platform is not enough on its own: a row can be live for an interpreter
    that no release in its range supports, which is not a resolution failure anyone reads
    as a marker bug -- pip just reports that no version matches.

    This is how `pandas>=3.0,<4` for win32+ARM64 shipped unsatisfiable on 3.9 and 3.10.
    The marker partition was exact, every environment had exactly one live row, and the
    row it had could never install. requires-python is part of whether a split is
    complete, so assert it here rather than trusting the platform axis alone.
    """
    for plat in PLATFORMS:
        for py in PYTHONS:
            env = _env(plat, py)
            for req in reqs:
                if req.marker is not None and not req.marker.evaluate(env):
                    continue
                floors = PACKAGE_PYTHON_FLOORS.get(req.name.lower())
                if not floors:
                    continue
                for spec, floor in floors:
                    # Does this row admit ONLY versions that need a newer interpreter?
                    if not spec.contains(_lowest_allowed(req), prereleases = True):
                        continue
                    assert _minor(py) >= floor, (
                        f"{label}: `{req}` is live on Python {py} {plat[2]}, but every "
                        f"version it admits needs Python >= {floor[0]}.{floor[1]}. "
                        "The row is unsatisfiable there; the marker needs a "
                        "python_version bound as well as a platform one."
                    )


def _lowest_allowed(req) -> str:
    """The smallest concrete version the row's specifier admits, for floor comparison."""
    lowers = [s.version for s in req.specifier if s.operator in (">=", "==", "~=", ">")]
    return lowers[0] if lowers else "0"


def test_the_woa_pandas_split_covers_every_supported_python():
    """
    The complement of the test above: having added a python_version bound, no ARM64
    interpreter may be left with no pandas row at all. 3.9 and 3.10 fall back to the
    2.3.3 row and source-build, which is what they did before win_arm64 wheels existed.
    """
    arm64 = ("win32", "Windows", "ARM64", "nt")
    for label, reqs in ALL_SOURCES:
        rows = [r for r in reqs if r.name.lower() == "pandas"]
        if not rows:
            continue
        for py in PYTHONS:
            env = _env(arm64, py)
            live = [r for r in rows if r.marker is None or r.marker.evaluate(env)]
            assert len(live) == 1, (
                f"{label}: Windows ARM64 on Python {py} has {len(live)} live pandas "
                f"rows, expected exactly 1: {[str(r) for r in live]}"
            )
            if _minor(py) < (3, 11):
                assert "3.0" not in str(live[0].specifier), (
                    f"{label}: Python {py} must not be handed the pandas 3 row"
                )
