# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""transformers and tokenizers have to be pinned together, or Apple Silicon loses Train.

transformers declares a tokenizers window and ENFORCES it at import, so a venv holding
a transformers that rejects its tokenizers is not subtly wrong: every
`import transformers` raises, `import mlx_lm` with it, mlx_repair.mlx_stack_blockers()
reports a blocker, and Studio comes up chat-only with Train and Export disabled.

Two ways that happened, both covered here:

* extras-no-deps.txt pinned `transformers==5.5.0` under --no-deps, which skips
  transformers' own tokenizers requirement, so whatever the earlier with-deps resolve
  had picked stayed. Half a pair is worse than no pin.
* install.sh's core phase runs with UV_OVERRIDE=overrides-darwin-arm64.txt and without
  single-env/constraints.txt. An override REPLACES every requirement on a package, so
  its unbounded `transformers>=5.5.0` also replaced pyproject's `<=5.5.0` cap. From
  transformers 5.16.0 (2026-08-26), the first release wanting `tokenizers>=0.23.1`, that
  resolve began landing tokenizers 0.23.2 on macOS arm64, and step 3b then downgraded
  transformers alone.

No unsloth commit caused the second one: a third-party release walked into an unbounded
override. So the resolver test below checks the pair a fresh macOS arm64 install would
END with, rather than any version number a diff would show.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
EXTRAS_NO_DEPS = REQ_ROOT / "extras-no-deps.txt"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
DARWIN_OVERRIDES = SINGLE_ENV / "overrides-darwin-arm64.txt"

# Files uv never installs FROM: constraints only narrow a resolve, overrides only replace
# requirements. They carry no pairing duty, which is why they are exempt from the "pin the
# pair" rule -- and why the override is the one file that can silently widen it.
NON_INSTALLING = {"constraints.txt", "overrides-darwin-arm64.txt", "overrides.txt"}

# The window each pinned transformers release declares. Keyed by version so that bumping a
# pin fails here until someone confirms the new release's window, rather than silently
# widening the assertion. Both current pins declare the same one.
DECLARED_TOKENIZERS_WINDOW = {
    "5.5.0": ">=0.22.0,<=0.23.0",
    "4.57.6": ">=0.22.0,<=0.23.0",
}

# The first tokenizers release outside that window, i.e. the version a stranded venv ends
# up on today. Named rather than computed so the negative control cannot drift with PyPI.
STRANDED_TOKENIZERS = "0.23.2"


def _requirements(path: pathlib.Path) -> list[Requirement]:
    """Parsed requirement lines; comments, flags and unparseable lines dropped."""
    out: list[Requirement] = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        try:
            out.append(Requirement(text))
        except InvalidRequirement:
            continue
    return out


def _named(reqs: list[Requirement], name: str) -> list[Requirement]:
    """Canonicalized match, so sentence-transformers never counts as transformers."""
    return [req for req in reqs if canonicalize_name(req.name) == canonicalize_name(name)]


def _pinned_versions(reqs: list[Requirement], name: str) -> list[str]:
    return [
        spec.version
        for req in _named(reqs, name)
        for spec in req.specifier
        if spec.operator == "=="
    ]


def _installing_files() -> list[pathlib.Path]:
    return [
        path
        for path in sorted(REQ_ROOT.rglob("*.txt"))
        if path.name not in NON_INSTALLING and not path.name.startswith(".")
    ]


def test_every_installing_file_that_pins_transformers_also_bounds_tokenizers():
    """The pair has to be named in the same file, because the file is the unit that gets
    installed. extras-no-deps.txt is installed with --no-deps, so nothing else in the run
    will supply the other half."""
    offenders = []
    for path in _installing_files():
        reqs = _requirements(path)
        if not _named(reqs, "transformers"):
            continue
        if not _named(reqs, "tokenizers"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"these files pin transformers without bounding tokenizers: {offenders}. "
        f"transformers enforces its tokenizers window at import, so pinning one half "
        f"leaves the venv one `import transformers` away from chat-only."
    )


def test_the_tokenizers_bound_matches_the_pinned_transformers_release():
    for path in _installing_files():
        reqs = _requirements(path)
        pins = _pinned_versions(reqs, "transformers")
        if not pins:
            continue
        bounds = _named(reqs, "tokenizers")
        assert len(bounds) == 1, f"{path.name}: expected one tokenizers line, got {bounds}"
        window = bounds[0].specifier
        for pinned in pins:
            assert pinned in DECLARED_TOKENIZERS_WINDOW, (
                f"{path.name} pins transformers=={pinned}, which this test does not know. "
                f"Add its declared tokenizers window to DECLARED_TOKENIZERS_WINDOW and "
                f"move the pins together."
            )
            declared = SpecifierSet(DECLARED_TOKENIZERS_WINDOW[pinned])
            assert window == declared, (
                f"{path.name}: tokenizers{window} does not match the window "
                f"transformers=={pinned} declares ({declared})"
            )


def test_the_tokenizers_line_covers_both_python_branches():
    """The transformers pins split on python_version; the tokenizers line must not, unless
    it splits the same way. An unconditional line is correct only while both pinned
    releases declare the same window, which the test above is what enforces."""
    for path in _installing_files():
        reqs = _requirements(path)
        if not _pinned_versions(reqs, "transformers"):
            continue
        markers = [req.marker for req in _named(reqs, "tokenizers")]
        assert markers and all(marker is None for marker in markers), (
            f"{path.name}: the tokenizers bound carries a marker ({markers}) while the "
            f"transformers pins split on python_version. Either branch left uncovered "
            f"installs transformers without its tokenizers."
        )


def test_no_requirements_file_admits_a_tokenizers_outside_the_window():
    """Including the non-installing files: a constraint or override that admits 0.23.1+
    lets a later resolve walk tokenizers past what the pinned transformers accepts."""
    offenders = {}
    for path in sorted(REQ_ROOT.rglob("*.txt")):
        if path.name.startswith("."):
            continue
        for req in _named(_requirements(path), "tokenizers"):
            if STRANDED_TOKENIZERS in req.specifier:
                offenders[str(path.relative_to(REPO_ROOT))] = str(req)
    assert not offenders, (
        f"these files admit tokenizers {STRANDED_TOKENIZERS}, which the pinned "
        f"transformers rejects at import: {offenders}"
    )


def test_the_checker_rejects_the_pair_that_broke_apple_silicon():
    """Negative control. Every assertion above is a "nothing found" shape, which is also
    what a checker that has quietly stopped checking reports."""
    window = SpecifierSet(DECLARED_TOKENIZERS_WINDOW["5.5.0"])
    assert STRANDED_TOKENIZERS not in window
    assert "0.22.2" in window


# ---------------------------------------------------------------------------
# The resolver check: what a fresh macOS arm64 install actually ends with.
# ---------------------------------------------------------------------------


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")
    return uv


def _compile(args: list[str], stdin: str | None = None) -> dict[str, str]:
    """Resolved name -> version for one `uv pip compile`, or skip if the index is not
    reachable. A network-flaky run must not read as a dependency regression."""
    proc = subprocess.run(
        [_uv(), "pip", "compile", *args],
        input = stdin,
        capture_output = True,
        text = True,
        timeout = 600,
        cwd = REPO_ROOT,
        # uv resolves aarch64-apple-darwin against macOS 12 by default, and mlx ships
        # macosx_14_0 wheels only, so the resolve would fail for a reason that has nothing
        # to do with the pair under test. Pin the deployment target instead of inheriting
        # whatever the host happens to export.
        env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "15.0"},
    )
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        if any(word in stderr.lower() for word in ("network", "dns", "timed out", "connect")):
            pytest.skip(f"package index unreachable: {stderr.strip()[:200]}")
        pytest.fail(f"uv pip compile failed:\n{stderr[-3000:]}")
    resolved = {}
    for line in proc.stdout.splitlines():
        text = line.split("#", 1)[0].strip()
        if "==" in text:
            name, _, version = text.partition("==")
            resolved[canonicalize_name(name.strip())] = version.strip()
    return resolved


def _declared_window(version: str) -> SpecifierSet:
    """transformers' own tokenizers requirement, read from PyPI rather than this file's
    table, so the check keeps being true after a pin moves."""
    url = f"https://pypi.org/pypi/transformers/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout = 60) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"PyPI unreachable: {exc}")
    for raw in metadata["info"].get("requires_dist") or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if canonicalize_name(req.name) == "tokenizers" and req.marker is None:
            return req.specifier
    pytest.fail(f"transformers {version} declares no unconditional tokenizers requirement")


def test_a_fresh_macos_arm64_install_ends_with_a_consistent_pair():
    """Model the installer IN ORDER. Compiling everything at once would find a consistent
    environment the installer never produces: install.sh's core phase resolves first, with
    the override and without constraints.txt, and step 3b then replaces the packages
    extras-no-deps.txt names, dependencies skipped.

    Fails on the tree that shipped the defect (core phase -> transformers 5.16.1 +
    tokenizers 0.23.2; step 3b -> transformers 5.5.0, tokenizers untouched).
    """
    platform_args = [
        "--python-platform", "aarch64-apple-darwin",
        "--python-version", "3.13",
    ]
    core = _compile(
        ["-", *platform_args, "--override", str(DARWIN_OVERRIDES)],
        stdin = "unsloth\nunsloth-zoo\n",
    )
    overlay = _compile([str(EXTRAS_NO_DEPS), "--no-deps", *platform_args])

    final = dict(core)
    final.update(overlay)  # --no-deps: only the named packages move
    transformers, tokenizers = final.get("transformers"), final.get("tokenizers")
    assert transformers and tokenizers, f"resolve produced neither half of the pair: {final}"

    window = _declared_window(transformers)
    assert tokenizers in window, (
        f"a fresh macOS arm64 install would end with transformers=={transformers} beside "
        f"tokenizers=={tokenizers}, outside the declared window {window}. That venv cannot "
        f"`import transformers`, so Train and Export stay off. Core phase resolved "
        f"tokenizers=={core.get('tokenizers')}; step 3b left it at {tokenizers}."
    )
