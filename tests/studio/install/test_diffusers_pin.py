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
# The commit pin, installed by default ON TOP of the release pin. Exempt from the
# one-source-of-truth scan below because it is the one deliberate second namer of diffusers: it is
# installed by the step immediately after the release pin and by nothing else, which the tests here
# pin down rather than assume.
MAIN_FILE = REQ_ROOT / "diffusers-main.txt"

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
        if path in (PIN_FILE, MAIN_FILE):
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
        f"diffusers is requirement-listed outside diffusers-pin.txt and diffusers-main.txt: "
        f"{offenders}. "
        f"Move it into the pin file so the dedicated late step remains authoritative."
    )


def test_the_main_build_pins_a_commit_and_runs_after_the_release():
    """The main-build file exists, names a COMMIT, and runs after the release pin.

    The commit is the load-bearing part and is asserted rather than trusted: a branch ref would
    leave nothing pinning behaviour, because any main build reports 0.41.0.dev0 and
    ``_version_tuple`` truncates it to (0, 41, 0), so no version check can tell two apart. Now that
    this installs by default that is a stronger requirement, not a weaker one, since the whole user
    base would otherwise be on whatever main happened to be that morning.
    """
    assert MAIN_FILE.is_file(), f"{MAIN_FILE} is missing"
    lines = _requirements(MAIN_FILE)
    assert len(lines) == 1, lines
    spec = lines[0]
    assert spec.startswith("diffusers @ git+"), spec
    revision = spec.rpartition("@")[2].strip()
    assert re.fullmatch(r"[0-9a-f]{40}", revision), (
        f"{MAIN_FILE.name} must pin a full 40-character commit, not {revision!r}: a branch ref "
        "moves under an unchanged requirements file and nothing in Unsloth can tell two builds of "
        "main apart"
    )

    source = _code_only(STACK.read_text(encoding = "utf-8"))
    # Installed only through its own step, which reads the opt-out.
    assert "diffusers-main.txt" in source
    assert "_diffusers_main_requested" in source
    assert "UNSLOTH_DIFFUSERS_MAIN" in source
    # And the CALL runs after the release pin install, or the release would overwrite it. Compared
    # on the call site, not on the filename: the helper is DEFINED earlier in the file than either
    # install, so a filename compare answers a different question and passes by accident.
    call = "\n    _diffusers_main_step()\n"
    assert call in source, "the main-build step is never called"
    assert source.index(call) > source.index('req = REQ_ROOT / "diffusers-pin.txt"')


def _probe_module(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, STACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_main_build_is_on_by_default_and_opts_out_on_zero(monkeypatch):
    """Default ON is the whole point, so the unset case is asserted, not assumed.

    The opt-out is deliberately narrow: only an explicit falsy value turns it off, because a typo
    in the variable name silently disabling a model is the worse failure of the two.
    """
    module = _probe_module("install_python_stack_probe")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    assert module._diffusers_main_requested() is True
    for value in ("", "1", "true", "YES", "on", "anything"):
        monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", value)
        assert module._diffusers_main_requested() is True, value
    for value in ("0", "false", "NO", "off", " off "):
        monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", value)
        assert module._diffusers_main_requested() is False, value

    # And opting out really stops the step, rather than only stopping the message.
    monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "0")
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("the step ran after opting out")
    )
    # _progress divides by a total the standalone module never set.
    progressed = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: progressed.append(label))
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    module._diffusers_main_step()
    # It still spends its slot. The total is fixed before the opt-out is known, so returning early
    # without a _progress leaves the bar stuck short of its own total for exactly these users.
    assert len(progressed) == 1, progressed


def test_the_main_build_falls_back_to_the_zip_when_there_is_no_git(monkeypatch):
    """No git is the COMMON case, so it must still get the pinned commit.

    Measured on a host whose ``git`` exits non-zero, which is what the desktop bundle looks like on
    macOS: installing 0.1.811 and then updating to 0.1.812 both left diffusers at 0.40.0, and
    Qwen-Image-2.1 refused to load on an install that had done nothing wrong. Nothing about that
    needed git: the same commit is a zip over plain https. The git clone stays the default because
    it records a ref, and this is the fallback.
    """
    module = _probe_module("install_python_stack_probe2")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_has_working_git", lambda: False)
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: False)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    calls = []

    def _capture(label, *args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(module, "pip_install_try", _capture)
    module._diffusers_main_step()

    assert len(calls) == 1, calls
    args, kwargs = calls[0]
    assert kwargs.get("req") is None, "the git requirement file must not be handed to pip here"
    spec = [arg for arg in args if arg.startswith("diffusers @ ")]
    assert len(spec) == 1, args
    revision = _requirements(MAIN_FILE)[0].rpartition("@")[2].strip().lower()
    assert spec[0] == (
        "diffusers @ https://github.com/huggingface/diffusers/archive/"
        f"{revision}.zip"
    ), spec


def test_the_main_build_keeps_the_release_when_there_is_no_git_and_no_zip(monkeypatch):
    """Diffusers is mandatory, unlike triton_kernels, so a host that can reach neither route must
    be left with the release the previous step installed rather than nothing at all.

    Reached by a pin file the zip route cannot serve: a non-GitHub remote, or a branch ref, which
    an archive cannot pin because it records no ref of its own.
    """
    module = _probe_module("install_python_stack_probe2b")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_has_working_git", lambda: False)
    monkeypatch.setattr(
        module,
        "_direct_reference_in_requirements",
        lambda req: ("https://gitlab.example/huggingface/diffusers", "a" * 40, ""),
    )
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("a git requirement without git")
    )
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    notes = []
    monkeypatch.setattr(module, "_note", lambda msg, *a, **k: notes.append(msg))
    module._diffusers_main_step()
    # And it SAYS so, rather than leaving the install looking like it got the main build.
    assert notes and "no working git" in notes[0].lower()


def test_an_archive_install_reads_back_as_resident(monkeypatch):
    """The half that makes the zip route survive.

    Without it the fallback works exactly once: pip records ``archive_info`` and no ref, the
    residency check only understood ``vcs_info``, so the main build read as absent,
    ``_diffusers_main_supersedes_release`` said the release pin had not been superseded, and the
    NEXT pass reinstalled diffusers 0.40.0 straight over it. Measured, not theorised.
    """
    module = _probe_module("install_python_stack_probe2c")

    revision = _requirements(MAIN_FILE)[0].rpartition("@")[2].strip().lower()
    archive = f"https://github.com/huggingface/diffusers/archive/{revision}.zip"
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: True)

    monkeypatch.setattr(
        module,
        "_recorded_direct_url",
        lambda dist: {"url": archive, "archive_info": {"hash": "sha256=abc"}},
    )
    assert module._direct_reference_is_installed(MAIN_FILE, "diffusers") is True

    # A zip of a DIFFERENT commit is not this one, which is the whole reason the SHA has to be in
    # the URL: an archive records no ref, so the URL is the only provenance there is.
    other = f"https://github.com/huggingface/diffusers/archive/{'b' * 40}.zip"
    monkeypatch.setattr(
        module,
        "_recorded_direct_url",
        lambda dist: {"url": other, "archive_info": {"hash": "sha256=abc"}},
    )
    assert module._direct_reference_is_installed(MAIN_FILE, "diffusers") is False


def test_the_zip_route_refuses_anything_it_cannot_pin():
    """The derivation is narrow on purpose: a wrong URL installs the wrong tree silently."""
    module = _probe_module("install_python_stack_probe2d")
    commit = "0" * 40

    assert module._github_archive_url("https://github.com/a/b.git", commit) == (
        f"https://github.com/a/b/archive/{commit}.zip"
    )
    assert module._github_archive_url("https://www.github.com/a/b/", commit) == (
        f"https://github.com/a/b/archive/{commit}.zip"
    )
    # A short SHA, a branch and a tag all fail: an archive carries no history, so only a full
    # commit in the URL can identify the tree afterwards.
    assert module._github_archive_url("https://github.com/a/b", commit[:12]) is None
    assert module._github_archive_url("https://github.com/a/b", "main") is None
    assert module._github_archive_url("https://github.com/a/b", "v1.2.3") is None
    # Other forges spell archives differently, and ssh carries no https route at all.
    assert module._github_archive_url("https://gitlab.com/a/b", commit) is None
    assert module._github_archive_url("ssh://git@github.com/a/b", commit) is None
    # A subdirectory is part of the package identity and this URL cannot carry it.
    assert module._github_archive_url("https://github.com/a/b", commit, "sub") is None


def test_a_failed_main_build_degrades_instead_of_failing_the_install(monkeypatch):
    """The one that makes default-on safe.

    ``pip_install`` exits the installer. Using it here would make a reachable github.com a hard
    requirement of installing Unsloth: a blocked proxy, an offline mirror or an upstream outage
    would turn a working install into no install. The release pin is already resident, so the
    failure is survivable and must be survived.
    """
    module = _probe_module("install_python_stack_probe3")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: False)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "pip_install", lambda *a, **k: pytest.fail("pip_install exits the installer")
    )
    attempted = []
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: (attempted.append(k.get("req")), False)[1]
    )
    notes = []
    monkeypatch.setattr(module, "_note", lambda msg, *a, **k: notes.append(msg))
    steps = {}
    monkeypatch.setattr(module, "_record_step", lambda name, state: steps.__setitem__(name, state))

    module._diffusers_main_step()

    assert attempted and attempted[0].name == "diffusers-main.txt"
    assert (
        steps["diffusers-main.txt"] == "skipped"
    ), "a failed build recorded as 'ran' would report an install that never happened"
    assert notes and "keeps the pinned" in notes[0]


def test_the_main_build_is_satisfied_without_touching_the_network(monkeypatch):
    """A full SHA is answerable from direct_url.json, and on by default this runs on every pass."""
    module = _probe_module("install_python_stack_probe4")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: True)
    # Provenance AND payload: residency needs both, and a test environment with no installed
    # Diffusers payload would otherwise fall through to the failing stub below and blame the step.
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: True)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("reinstalled an installed pin")
    )
    steps = {}
    monkeypatch.setattr(module, "_record_step", lambda name, state: steps.__setitem__(name, state))
    module._diffusers_main_step()
    assert steps["diffusers-main.txt"] == "skipped"


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
    marker = 'sys_platform == "win32" and platform_machine == "ARM64"'
    # Either operator satisfies what this test is for. The floor exists so the resolver is not
    # pushed above the first release carrying a win_arm64 wheel; an exact pin at that same
    # version is that floor with the ceiling closed too. It is not interchangeable in the other
    # direction: scan_packages_baseline.json keys its reviewed-benign findings by a hash of the
    # scanned file's contents, so a dist recorded there has to be pinned exactly or the security
    # audit reds on whatever unrelated PR is open the day upstream publishes.
    wanted = {
        f"{dist}>={floor}; {marker}",
        f"{dist}=={floor}; {marker}",
    }
    assert wanted & set(
        line.strip() for line in text.splitlines()
    ), f"{relpath} no longer floors {dist} at {floor}"
    # And nothing else floors the same dist higher on that marker.
    for line in text.splitlines():
        line = line.strip()
        if (
            not line.startswith((f"{dist}>=", f"{dist}=="))
            or 'platform_machine == "ARM64"' not in line
        ):
            continue
        assert line in wanted, f"{relpath}: a second ARM64 floor for {dist}: {line}"


def test_the_release_pin_stands_down_once_the_main_build_is_resident(monkeypatch):
    """The one that makes a second update a no-op.

    The release pin is a version pin and the main build reports ``0.41.0.dev0``, so the pin never
    reads as satisfied once the commit is in place. Left alone it reinstalls the release on every
    pass and the step after it reinstalls the same commit on top, which is two Diffusers installs
    per update and an offline update that downgrades a working build and then cannot restore it.
    """
    module = _probe_module("install_python_stack_probe5")

    calls = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: calls.append(label))
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    monkeypatch.setattr(module, "_requirements_satisfied", lambda *a, **k: False)

    # Resident and wanted: the step must not run, and must still spend its slot.
    assert (
        module._skip_step(
            module.REQ_ROOT / "diffusers-pin.txt",
            "diffusers pin",
            no_deps = False,
            superseded = True,
        )
        is True
    )
    assert len(calls) == 1 and "skipped" in calls[0], calls

    # Not superseded: unchanged, so a first install and an opt-out both still get the release.
    calls.clear()
    assert (
        module._skip_step(
            module.REQ_ROOT / "diffusers-pin.txt",
            "diffusers pin",
            no_deps = False,
            superseded = False,
        )
        is False
    )
    assert calls == ["diffusers pin"], calls


def test_opting_out_or_a_missing_main_build_still_reinstates_the_release(monkeypatch):
    """The supersession is narrow on purpose: it is the ONLY thing standing between an opt-out and
    a Studio left on a main build it asked not to have."""
    module = _probe_module("install_python_stack_probe6")
    main_req = module.REQ_ROOT / "diffusers-main.txt"

    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: True)
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    assert module._diffusers_main_requested() and module._direct_reference_is_installed(
        main_req, "diffusers"
    )

    monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "0")
    assert not (
        module._diffusers_main_requested()
        and module._direct_reference_is_installed(main_req, "diffusers")
    ), "opting out must let the release pin run again"

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: False)
    assert not (
        module._diffusers_main_requested()
        and module._direct_reference_is_installed(main_req, "diffusers")
    ), "a first install must lay the release down before the commit goes on top of it"


def test_a_damaged_main_build_is_repaired_rather_than_believed(monkeypatch):
    """Provenance alone is not a build, and here that is worse than it is for triton kernels.

    ``direct_url.json`` survives inside dist-info while the package tree under it is deleted or
    truncated. The release pin reads the same predicate to decide it has been superseded, so a
    provenance-only answer would skip the reinstall AND the repair, on a MANDATORY dependency, on
    every later pass rather than one. Either half failing has to mean "install it".
    """
    module = _probe_module("install_python_stack_probe7")
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)

    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: True)
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: True)
    assert module._diffusers_main_resident() is True

    # The ref still points at the right commit, the files under it are gone.
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: False)
    assert (
        module._diffusers_main_resident() is False
    ), "a damaged payload must not read as installed"
    # And with it False, the release pin is no longer superseded, so the repair really can run.
    assert not (module._diffusers_main_requested() and module._diffusers_main_resident())

    # The step itself reinstalls rather than reporting satisfied.
    attempted = []
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, req = None, **k: (attempted.append(req), True)[1]
    )
    module._diffusers_main_step()
    assert attempted and attempted[0].name == "diffusers-main.txt"


def test_python_39_does_not_clone_a_build_it_can_never_install(monkeypatch):
    """diffusers-pin.txt still names 0.36.0 below 3.10, so 3.9 is a supported install, and main
    declares requires-python >= 3.10. Unmarked, pip clones the repository and only then rejects
    it, and since the build can never become resident that clone repeats on every update."""
    module = _probe_module("install_python_stack_probe8")
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("cloned main on python 3.9")
    )
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)

    class _V(tuple):
        pass

    monkeypatch.setattr(module.sys, "version_info", _V((3, 9, 21)))
    progressed: list = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: progressed.append(label))
    module._diffusers_main_step()
    # The slot is still spent: the denominator is fixed before the interpreter is consulted.
    assert len(progressed) == 1 and "python 3.10" in progressed[0], progressed

    # And the release pin is NOT superseded there, so 3.9 keeps getting its 0.36.0.
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: False)
    assert not (module._diffusers_main_requested() and module._diffusers_main_resident())

    # 3.10 is unaffected.
    attempted: list = []
    monkeypatch.setattr(module.sys, "version_info", _V((3, 10, 0)))
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: False)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, req = None, **k: (attempted.append(req), True)[1]
    )
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    module._diffusers_main_step()
    assert attempted and attempted[0].name == "diffusers-main.txt"


def test_the_full_deps_escape_hatch_reaches_both_diffusers_steps(monkeypatch):
    """UNSLOTH_STUDIO_FULL_DEPS is the documented repair, and this was the one pin-shaped pair it
    could not reach.

    ``_diffusers_main_resident`` is exactly the kind of evidence the hatch exists to override:
    ``_payload_recorded_intact`` compares recorded sizes, so a same-size corruption reads as intact
    and both steps would skip, leaving nothing to repair Diffusers with.
    """
    module = _probe_module("install_python_stack_probe_fulldeps")

    calls: list = []
    installed: list = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: calls.append(label))
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_diffusers_main_resident", lambda req = None: True)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: (installed.append(k.get("req")), True)[1]
    )
    monkeypatch.delenv(module.DIFFUSERS_MAIN_ENV, raising = False)

    # Without the hatch: resident means skip, and the release pin stands down.
    monkeypatch.delenv(module._FULL_DEPS_ENV, raising = False)
    module._diffusers_main_step()
    assert installed == [], installed
    assert len(calls) == 1 and "satisfied, skipped" in calls[0], calls
    assert module._diffusers_main_supersedes_release() is True

    # With it: the build is reinstalled, and the release pin runs first so the order is the one a
    # first install takes.
    calls.clear()
    monkeypatch.setenv(module._FULL_DEPS_ENV, "1")
    module._diffusers_main_step()
    assert len(installed) == 1, installed
    assert calls == ["diffusers main"], calls
    assert module._diffusers_main_supersedes_release() is False

    # Opting out of the main build still wins over the hatch: no source build either way.
    calls.clear()
    installed.clear()
    monkeypatch.setenv(module.DIFFUSERS_MAIN_ENV, "0")
    module._diffusers_main_step()
    assert installed == [], installed
    assert len(calls) == 1 and "opted out" in calls[0], calls
    assert module._diffusers_main_supersedes_release() is False
