# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned Diffusers release has to survive a fresh install.sh, not just an update.

MiniMax-H3 and MiniMax Music 3 need Diffusers 0.40.0 or newer, and Unsloth refuses
to load them otherwise. The pin originally lived in
studio/backend/requirements/base.txt, which did not reach fresh install.sh installs at
the time. base.txt now reaches those installs as an independent shared phase, but it
still runs too early to hold this pin safely.

These tests pin the shape that fixes it: exactly one file names diffusers, and the step
that installs it sits outside every skip.
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import subprocess
import tokenize

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"
PIN_FILE = REQ_ROOT / "diffusers-pin.txt"

# The shape install_python_stack._filter_requirements writes: a dot, the source stem,
# "-filtered-", then tempfile's random suffix. NamedTemporaryFile's suffixes are
# [A-Za-z0-9_]{8}, so this cannot swallow a checked-in file that merely starts with a dot.
_GENERATED_FILTER = re.compile(r"\.[\w.-]+-filtered-\w{8}\.txt")
STACK = REPO_ROOT / "studio" / "install_python_stack.py"
INSTALL_SH = REPO_ROOT / "install.sh"


def _requirements(path: pathlib.Path) -> list[str]:
    """Requirement lines only: comments and flag lines dropped."""
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and not text.startswith("-"):
            out.append(text)
    return out


def _code_only(source: str) -> str:
    """`source` with comment text blanked out, offsets preserved.

    The ordering check scans for requirements filenames and has to read them as installs,
    not prose. Blanking keeps every index truthful; tokenize spares a `#` inside a string."""
    lines = source.splitlines(keepends = True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    out = list(source)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments = [tok for tok in tokens if tok.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):  # pragma: no cover
        return source
    for tok in comments:
        begin = starts[tok.start[0] - 1] + tok.start[1]
        for index in range(begin, begin + len(tok.string)):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def test_the_pin_file_exists_and_names_the_first_supported_release():
    assert PIN_FILE.is_file(), f"{PIN_FILE} is missing"
    lines = _requirements(PIN_FILE)
    modern = [line for line in lines if 'python_version >= "3.10"' in line]
    assert modern == ['diffusers==0.40.0 ; python_version >= "3.10"'], modern
    assert "://" not in modern[0], "the released dependency must not require a source build"
    assert 'python_version >= "3.10"' in modern[0], (
        "diffusers dropped Python 3.9 in 0.38, so the release needs a >= 3.10 marker or "
        "the resolver has no candidate at all on a 3.9 host"
    )


def test_only_the_pin_file_names_diffusers():
    """One source of truth. A second entry anywhere is how a release creeps back in:
    whichever step runs last wins, and the step order is not obvious from any one file."""
    offenders = {}
    for path in sorted(REQ_ROOT.rglob("*.txt")):
        if path == PIN_FILE:
            continue
        # install_python_stack._filter_requirements writes `.{stem}-filtered-XXXX.txt` BESIDE the source on purpose, so
        # relative -r/-c includes still resolve, and it does not delete it.
        # Matched by that exact shape rather than by "starts with a dot": a checked-in hidden file such as
        # .constraints.txt is a real requirements file and a real place the pin could be overridden from, so it stays in
        # the scan.
        if _GENERATED_FILTER.fullmatch(path.name):
            continue
        named = [line for line in _requirements(path) if line.lower().startswith("diffusers")]
        if named:
            offenders[str(path.relative_to(REPO_ROOT))] = named
    assert not offenders, (
        f"diffusers is requirement-listed outside diffusers-pin.txt: {offenders}. "
        f"Move it into the pin file so the dedicated late step remains authoritative."
    )


def test_the_pin_step_is_not_gated_by_skip_base_or_no_torch():
    """The pin must sit at function top level so it reaches every install path."""
    tree = ast.parse(STACK.read_text(encoding = "utf-8"))

    def _installs_pin(node: ast.AST) -> bool:
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", None) != "pip_install":
                continue
            for kw in call.keywords:
                if kw.arg == "req" and "diffusers-pin.txt" in ast.dump(kw.value):
                    return True
        return False

    found = False
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for stmt in func.body:  # top level of the function only, no if/else nesting
            if _installs_pin(stmt):
                found = True
    assert found, (
        "no unconditional pip_install of diffusers-pin.txt found at the top level of any "
        "function in install_python_stack.py. Nested under an `if`, the pin can miss an "
        "install path."
    )


def test_the_pin_step_runs_after_every_other_requirements_install():
    """Ordering matters: a later `uv pip install -r ...` can re-resolve diffusers back to a
    release. Keeping the pin last means nothing is left that could walk it forward."""
    source = _code_only(STACK.read_text(encoding = "utf-8"))
    pin_at = source.index("diffusers-pin.txt")
    later = [
        name
        for name in (
            "extras.txt",
            "extras-no-deps.txt",
            "studio.txt",
            "base.txt",
            "no-torch-runtime.txt",
            "data-designer-deps.txt",
            "data-designer.txt",
        )
        if source.rfind(name) > pin_at
    ]
    assert not later, f"these requirements files are installed after the diffusers pin: {later}"


def test_the_ordering_check_reads_installs_not_prose():
    """The torchcodec comment names extras-no-deps.txt after the pin, so the check must read
    that as prose while a real later install still trips it."""
    pin = 'pip_install("diffusers pin", "-r", "diffusers-pin.txt")\n'

    prose = _code_only(pin + "# cannot live in extras-no-deps.txt because markers\n")
    assert prose.rfind("extras-no-deps.txt") < prose.index(
        "diffusers-pin.txt"
    ), "a commented mention of a requirements file must not count as an install"

    real = _code_only(pin + 'pip_install("extras", "-r", "extras-no-deps.txt")\n')
    assert real.rfind("extras-no-deps.txt") > real.index(
        "diffusers-pin.txt"
    ), "a genuine later install must still be caught"

    # A `#` inside a string literal is not a comment and must survive intact.
    kept = _code_only('marker = "extras-no-deps.txt#egg"\n')
    assert "extras-no-deps.txt#egg" in kept

    source = STACK.read_text(encoding = "utf-8")
    blanked = _code_only(source)
    assert len(blanked) == len(source)
    assert blanked.index("diffusers-pin.txt") == source.index("diffusers-pin.txt")


def test_install_sh_still_delegates_the_core_package_skip():
    """The handoff flag skips core packages while allowing other base entries through."""
    assert 'SKIP_STUDIO_BASE="$_SKIP_BASE"' in INSTALL_SH.read_text(encoding = "utf-8")
    assert "_SKIP_BASE=1" in INSTALL_SH.read_text(encoding = "utf-8")


def test_no_generated_filter_snapshot_is_tracked():
    """They are a copy of a file already in the tree, and one got committed.

    _filter_requirements writes beside the source so relative -r/-c includes resolve.
    pip_install unlinks them in a finally, but a test calling the helper directly, or an
    install killed mid-run, leaves them in the checkout, where `git add -A` picks them up.
    A stale snapshot then reads as a second, silently divergent copy of the pins.
    """
    done = subprocess.run(
        ["git", "ls-files", "-z", "--", "studio/backend/requirements/"],
        cwd = REPO_ROOT,
        capture_output = True,
        text = True,
    )
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [p for p in done.stdout.split("\0") if p]
    offenders = [p for p in tracked if _GENERATED_FILTER.fullmatch(pathlib.Path(p).name)]
    assert not offenders, f"generated filter snapshots are tracked: {offenders}"


def test_gitignore_covers_the_generated_snapshots():
    """So the next `git add -A` cannot put one back."""
    probe = REQ_ROOT / ".studio-filtered-abcd1234.txt"
    assert _GENERATED_FILTER.fullmatch(probe.name), "the probe must match the generated shape"
    done = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", str(probe)],
        cwd = REPO_ROOT,
        capture_output = True,
        text = True,
    )
    if done.returncode == 128:
        pytest.skip("not a git checkout")
    assert done.returncode == 0, f"{probe.name} is not ignored; .gitignore needs the pattern"


# A win_arm64 floor set above the first release that actually publishes one costs the
# resolver every wheel in between, and for scikit-learn it cost the only one that exists on
# a free-threaded 3.13. Each floor below is the earliest release carrying a win_arm64 wheel,
# read off PyPI's own file list, so the pin can be checked against the index by hand.
WIN_ARM64_FLOORS = [
    # scikit-learn 1.9.0 dropped cp313-cp313t; 1.8.0 is the only release with one, so a
    # >=1.9.0 floor leaves a free-threaded 3.13 with no wheel and an sdist to compile.
    ("extras.txt", "scikit-learn", "1.8.0"),
    # av publishes cp311-abi3 plus a cp314t from 17.0.0. No release has a 3.13t wheel.
    ("extras.txt", "av", "17.0.0"),
    ("single-env/constraints.txt", "av", "17.0.0"),
]


@pytest.mark.parametrize(
    "relpath, dist, floor",
    WIN_ARM64_FLOORS,
    ids = [f"{r.split('/')[-1]}:{d}" for r, d, _ in WIN_ARM64_FLOORS],
)
def test_the_win_arm64_floor_is_the_first_release_that_has_a_wheel(relpath, dist, floor):
    text = (REQ_ROOT / relpath).read_text(encoding = "utf-8")
    wanted = f'{dist}>={floor}; sys_platform == "win32" and platform_machine == "ARM64"'
    assert wanted in text, f"{relpath} no longer floors {dist} at {floor}"
    # And nothing else floors the same dist higher on that marker.
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(f"{dist}>=") or 'platform_machine == "ARM64"' not in line:
            continue
        assert line == wanted, f"{relpath}: a second ARM64 floor for {dist}: {line}"
