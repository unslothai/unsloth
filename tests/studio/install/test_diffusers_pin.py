# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusers is pinned in one file, installed outside every skip, since base.txt runs too early."""

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
MAIN_FILE = REQ_ROOT / "diffusers-main.txt"
# Git-main tests run on a copy with the commit line uncommented (``_active_main``); the shipped state is below.
_ACTIVE_REQ_ROOT: list = []

# NamedTemporaryFile suffixes are [A-Za-z0-9_]{8}, so checked-in dotfiles are not matched.
_GENERATED_FILTER = re.compile(r"\.[\w.-]+-filtered-\w{8}\.txt")
STACK = REPO_ROOT / "studio" / "install_python_stack.py"
INSTALL_SH = REPO_ROOT / "install.sh"


@pytest.fixture(scope = "session", autouse = True)
def _active_req_root(tmp_path_factory):
    import shutil

    root = tmp_path_factory.mktemp("active_main") / "requirements"
    shutil.copytree(REQ_ROOT, root)
    main = root / "diffusers-main.txt"
    text = main.read_text(encoding = "utf-8")
    main.write_text(text.replace("\n# diffusers @ git+", "\ndiffusers @ git+"), encoding = "utf-8")
    _ACTIVE_REQ_ROOT[:] = [root]
    yield root


def _active_main() -> pathlib.Path:
    return _ACTIVE_REQ_ROOT[0] / "diffusers-main.txt"


def _requirements(path: pathlib.Path) -> list[str]:
    """Requirement lines only: comments and flag lines dropped."""
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and not text.startswith("-"):
            out.append(text)
    return out


def _code_only(source: str) -> str:
    """Blanks comment text but keeps offsets, so scans read code only; string literals are left intact."""
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
    assert modern == ['diffusers==0.41.0 ; python_version >= "3.10"'], modern
    legacy = [line for line in lines if 'python_version < "3.10"' in line]
    assert legacy == ['diffusers==0.36.0 ; python_version < "3.10"'], legacy
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
    """The main build must pin a commit, since any main build reports the same version."""
    assert MAIN_FILE.is_file(), f"{MAIN_FILE} is missing"
    lines = _requirements(_active_main())
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
    assert "diffusers-main.txt" in source
    assert "_diffusers_main_requested" in source
    assert "UNSLOTH_DIFFUSERS_MAIN" in source
    # Compare on the call site: the helper is defined before either install.
    call = "\n    _diffusers_main_step()\n"
    assert call in source, "the main-build step is never called"
    assert source.index(call) > source.rindex('req = REQ_ROOT / "diffusers-pin.txt"')


def test_the_shipped_main_build_is_commented_out_but_keeps_its_commit():
    """0.41.0 is on PyPI: nothing installs from git, and re-enabling is one uncomment away."""
    assert _requirements(MAIN_FILE) == []
    text = MAIN_FILE.read_text(encoding = "utf-8")
    assert re.search(r"^# diffusers @ git\+https://\S+@[0-9a-f]{40}$", text, re.M), text
    assert re.search(r"^# archive-sha256: [0-9a-f]{64}$", text, re.M)
    module = _probe_module("install_python_stack_probe_shipped", active = False)
    assert module._diffusers_main_active() is False
    assert module._diffusers_main_resident() is False
    assert module._diffusers_main_supersedes_release() is False


def _probe_module(name: str, active: bool = True):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, STACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if active:
        module.REQ_ROOT = _ACTIVE_REQ_ROOT[0]
    return module


def test_the_main_build_is_on_by_default_and_opts_out_on_zero(monkeypatch):
    """Only an explicit falsy UNSLOTH_DIFFUSERS_MAIN turns the main build off; unset means on."""
    module = _probe_module("install_python_stack_probe")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    assert module._diffusers_main_requested() is True
    for value in ("", "1", "true", "YES", "on", "anything"):
        monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", value)
        assert module._diffusers_main_requested() is True, value
    for value in ("0", "false", "NO", "off", " off "):
        monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", value)
        assert module._diffusers_main_requested() is False, value

    monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "0")
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("the step ran after opting out")
    )
    progressed = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: progressed.append(label))
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    module._diffusers_main_step()
    assert len(progressed) == 1, progressed


def test_the_main_build_falls_back_to_the_zip_when_there_is_no_git(monkeypatch):
    """With no working git, the main build comes from the commit's zip over https."""
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
    revision = _requirements(_active_main())[0].rpartition("@")[2].strip().lower()
    digest = module._archive_sha256_in_requirements(_active_main())
    assert digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest), digest
    # The fragment is what pip and uv verify; the commit in the URL checks no bytes.
    assert spec[0] == (
        "diffusers @ https://github.com/huggingface/diffusers/archive/"
        f"{revision}.zip#sha256={digest}"
    ), spec


def test_the_zip_route_is_skipped_without_exactly_one_pinned_digest(tmp_path, monkeypatch):
    """No single valid digest, no zip route: the host keeps the release."""
    module = _probe_module("install_python_stack_probe2e")
    spec = _requirements(_active_main())[0]
    digest = module._archive_sha256_in_requirements(_active_main())
    pin = tmp_path / "diffusers-main.txt"

    pin.write_text(f"{spec}\n", encoding = "utf-8")
    assert module._diffusers_main_archive(pin) is None
    pin.write_text(
        f"# archive-sha256: {digest}\n# archive-sha256: {'0' * 64}\n{spec}\n", encoding = "utf-8"
    )
    assert module._diffusers_main_archive(pin) is None
    pin.write_text(f"# archive-sha256: {digest[:63]}\n{spec}\n", encoding = "utf-8")
    assert module._diffusers_main_archive(pin) is None
    pin.write_text(f"# archive-sha256: {digest.upper()}\n{spec}\n", encoding = "utf-8")
    assert module._diffusers_main_archive(pin).endswith(f".zip#sha256={digest}")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "REQ_ROOT", tmp_path)
    monkeypatch.setattr(module, "_has_working_git", lambda: False)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(module, "_note", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "pip_install_try", lambda *a, **k: pytest.fail("an unverified archive install")
    )
    pin.write_text(f"{spec}\n", encoding = "utf-8")
    module._diffusers_main_step()


def test_the_main_build_keeps_the_release_when_there_is_no_git_and_no_zip(monkeypatch):
    """Without git or a zip, the release stays: diffusers is mandatory, unlike triton_kernels."""
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
    assert notes and "no working git" in notes[0].lower()


def test_an_archive_install_reads_back_as_resident(monkeypatch):
    """Residency must read archive_info as well as vcs_info, or the next pass reinstalls the release."""
    module = _probe_module("install_python_stack_probe2c")

    revision = _requirements(_active_main())[0].rpartition("@")[2].strip().lower()
    archive = f"https://github.com/huggingface/diffusers/archive/{revision}.zip"
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: True)

    monkeypatch.setattr(
        module,
        "_recorded_direct_url",
        lambda dist: {"url": archive, "archive_info": {"hash": "sha256=abc"}},
    )
    assert module._direct_reference_is_installed(_active_main(), "diffusers") is True

    other = f"https://github.com/huggingface/diffusers/archive/{'b' * 40}.zip"
    monkeypatch.setattr(
        module,
        "_recorded_direct_url",
        lambda dist: {"url": other, "archive_info": {"hash": "sha256=abc"}},
    )
    assert module._direct_reference_is_installed(_active_main(), "diffusers") is False


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
    assert module._github_archive_url("https://github.com/a/b", commit[:12]) is None
    assert module._github_archive_url("https://github.com/a/b", "main") is None
    assert module._github_archive_url("https://github.com/a/b", "v1.2.3") is None
    assert module._github_archive_url("https://gitlab.com/a/b", commit) is None
    assert module._github_archive_url("ssh://git@github.com/a/b", commit) is None
    assert module._github_archive_url("https://github.com/a/b", commit, "sub") is None


def test_a_failed_main_build_degrades_instead_of_failing_the_install(monkeypatch):
    """A failed main build must degrade rather than exit, since the release pin is already resident."""
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
        steps["diffusers-main.txt"] == "failed"
    ), "the setup fast path reads 'failed' to stop forcing a pass that cannot succeed"
    assert notes and "keeps the pinned" in notes[0]


def test_the_main_build_is_satisfied_without_touching_the_network(monkeypatch):
    """A full SHA is answerable from direct_url.json, and on by default this runs on every pass."""
    module = _probe_module("install_python_stack_probe4")

    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_has_working_git", lambda: True)
    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: True)
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
        for stmt in func.body:
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
    # The pass's own install is the last one; the startup repair's comes earlier.
    pin_at = source.rindex('req = REQ_ROOT / "diffusers-pin.txt"')
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
    """Filter snapshots are copies of pins and must not be committed, or a stale one diverges silently."""
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


# Each floor is the earliest release with a win_arm64 wheel, per PyPI's file list.
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
    # An exact pin also works; dists in scan_packages_baseline.json must be pinned exactly.
    wanted = {
        f"{dist}>={floor}; {marker}",
        f"{dist}=={floor}; {marker}",
    }
    assert wanted & set(
        line.strip() for line in text.splitlines()
    ), f"{relpath} no longer floors {dist} at {floor}"
    for line in text.splitlines():
        line = line.strip()
        if (
            not line.startswith((f"{dist}>=", f"{dist}=="))
            or 'platform_machine == "ARM64"' not in line
        ):
            continue
        assert line in wanted, f"{relpath}: a second ARM64 floor for {dist}: {line}"


def test_the_release_pin_stands_down_once_the_main_build_is_resident(monkeypatch):
    """The release pin stands down once the main build is resident, or each update installs twice."""
    module = _probe_module("install_python_stack_probe5")

    calls = []
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: calls.append(label))
    monkeypatch.setattr(module, "_record_step", lambda *a, **k: None)
    monkeypatch.setattr(module, "_requirements_satisfied", lambda *a, **k: False)

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
    """A damaged main build must be reinstalled, since direct_url.json can survive a truncated tree."""
    module = _probe_module("install_python_stack_probe7")
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)

    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: True)
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: True)
    assert module._diffusers_main_resident() is True

    monkeypatch.setattr(module, "_payload_recorded_intact", lambda *a, **k: False)
    assert (
        module._diffusers_main_resident() is False
    ), "a damaged payload must not read as installed"
    assert not (module._diffusers_main_requested() and module._diffusers_main_resident())

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
    assert len(progressed) == 1 and "python 3.10" in progressed[0], progressed

    monkeypatch.setattr(module, "_direct_reference_is_installed", lambda *a, **k: False)
    assert not (module._diffusers_main_requested() and module._diffusers_main_resident())

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
    """The full-deps hatch must reach both Diffusers steps, since size checks miss same-size corruption."""
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

    monkeypatch.delenv(module._FULL_DEPS_ENV, raising = False)
    module._diffusers_main_step()
    assert installed == [], installed
    assert len(calls) == 1 and "satisfied, skipped" in calls[0], calls
    assert module._diffusers_main_supersedes_release() is True

    calls.clear()
    monkeypatch.setenv(module._FULL_DEPS_ENV, "1")
    module._diffusers_main_step()
    assert len(installed) == 1, installed
    assert calls == ["diffusers main"], calls
    assert module._diffusers_main_supersedes_release() is False

    calls.clear()
    installed.clear()
    monkeypatch.setenv(module.DIFFUSERS_MAIN_ENV, "0")
    module._diffusers_main_step()
    assert installed == [], installed
    assert len(calls) == 1 and "opted out" in calls[0], calls
    assert module._diffusers_main_supersedes_release() is False


def _fast_path_probe_module(
    monkeypatch,
    *,
    resident,
    git = True,
    last = None,
    requested = True,
):
    module = _probe_module("install_python_stack_fastpath_probe")
    if requested:
        monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "0")
    monkeypatch.setattr(module, "_has_working_git", lambda: git)
    monkeypatch.setattr(module, "_diffusers_main_resident", lambda *a, **k: resident)
    manifest = {"step_results": {"diffusers-main.txt": last}} if last else {}
    monkeypatch.setattr(module.install_manifest, "read_manifest", lambda *a, **k: manifest)
    return module


def test_the_fast_path_is_forced_when_the_main_build_never_went_in(monkeypatch):
    """An install updated by an installer that predates 11c is current and still on the release."""
    module = _fast_path_probe_module(monkeypatch, resident = False)
    assert module._diffusers_main_needs_dependency_pass() is True


def test_a_resident_main_build_keeps_the_fast_path(monkeypatch):
    module = _fast_path_probe_module(monkeypatch, resident = True)
    assert module._diffusers_main_needs_dependency_pass() is False


def test_a_host_without_git_is_forced_only_when_the_zip_route_exists(monkeypatch):
    """11c installs the pinned commit from a zip without git; with no such route it skips, and
    forcing the pass there would repeat it on every update."""
    module = _fast_path_probe_module(monkeypatch, resident = False, git = False)
    assert module._diffusers_main_archive(module.REQ_ROOT / "diffusers-main.txt") is not None
    assert module._diffusers_main_needs_dependency_pass() is True
    monkeypatch.setattr(module, "_diffusers_main_archive", lambda req: None)
    assert module._diffusers_main_needs_dependency_pass() is False


def test_a_recorded_failed_build_keeps_the_fast_path(monkeypatch):
    """A host that cannot reach github.com would otherwise run the whole pass on every update."""
    module = _fast_path_probe_module(monkeypatch, resident = False, last = "failed")
    assert module._diffusers_main_needs_dependency_pass() is False
    module = _fast_path_probe_module(monkeypatch, resident = False, last = "skipped")
    assert module._diffusers_main_needs_dependency_pass() is True


def test_opting_out_forces_the_pass_only_while_the_main_build_is_resident(monkeypatch):
    """11b reinstates the release, but only if the pass runs."""
    module = _fast_path_probe_module(monkeypatch, resident = True, requested = False)
    assert module._diffusers_main_needs_dependency_pass() is True
    module = _fast_path_probe_module(monkeypatch, resident = False, requested = False)
    assert module._diffusers_main_needs_dependency_pass() is False


@pytest.mark.parametrize("script", ["setup.sh", "setup.ps1"])
def test_both_fast_paths_consult_the_probe(script):
    source = (REPO_ROOT / "studio" / script).read_text(encoding = "utf-8")
    assert "--diffusers-main-needs-dependency-pass" in source


def test_the_probe_flag_answers_without_a_traceback():
    """A crash also exits 1, which setup.sh reads as 'keep the fast path'."""
    import sys

    result = subprocess.run(
        [sys.executable, str(STACK), "--diffusers-main-needs-dependency-pass"],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode in (0, 1), result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def _repair_module(
    monkeypatch,
    *,
    needed,
    resident_after,
    uncontended = (True,),
):
    """``uncontended`` is read once per poll of the pass lock, ``needed`` once the lock is free."""
    import contextlib

    module = _probe_module("install_python_stack_repair_probe")
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    needed, uncontended = iter(needed), iter(uncontended)
    monkeypatch.setattr(module, "_diffusers_main_needs_dependency_pass", lambda: next(needed))
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: True)
    monkeypatch.setattr(module, "_diffusers_main_resident", lambda *a, **k: resident_after)
    monkeypatch.setattr(
        module.install_manifest,
        "pass_lock",
        lambda *a, **k: contextlib.nullcontext(next(uncontended)),
    )
    monkeypatch.setattr(module, "_REPAIR_LOCK_POLL_S", 0)
    recorded = []
    monkeypatch.setattr(
        module.install_manifest, "update_manifest", lambda **extra: recorded.append(extra)
    )
    module.recorded = recorded
    module.lock_polls = lambda: len(polls)
    polls = []
    lock = module.install_manifest.pass_lock
    monkeypatch.setattr(
        module.install_manifest, "pass_lock", lambda *a, **k: polls.append(1) or lock()
    )
    ran = []
    monkeypatch.setattr(module, "_diffusers_main_step", lambda: ran.append(True))
    return module, ran


def test_the_startup_repair_runs_only_the_main_step(monkeypatch):
    """The backend's self-heal: 0 once the build is in, 2 when the step could not install it."""
    module, ran = _repair_module(monkeypatch, needed = [True], resident_after = True)
    assert module._repair_diffusers_main() == 0 and ran == [True]
    assert module.recorded == []
    module, ran = _repair_module(monkeypatch, needed = [True], resident_after = False)
    assert module._repair_diffusers_main() == 2 and ran == [True]
    assert module.recorded == [{module._DIFFUSERS_MAIN_REPAIR_KEY: "failed"}]


def test_a_failed_startup_repair_stops_startup_retries_but_not_an_update(monkeypatch):
    """Every start would otherwise refuse diffusion loads through another fetch this host cannot
    make, while the update the backend's warning points at must still retry."""

    def failed_repair(module):
        monkeypatch.setattr(
            module.install_manifest,
            "read_manifest",
            lambda *a, **k: {module._DIFFUSERS_MAIN_REPAIR_KEY: "failed"},
        )

    module, ran = _repair_module(monkeypatch, needed = [True], resident_after = False)
    failed_repair(module)
    assert module._repair_diffusers_main() == 1 and ran == []
    probe = _fast_path_probe_module(monkeypatch, resident = False)
    failed_repair(probe)
    assert probe._diffusers_main_needs_dependency_pass() is True


def test_the_startup_repair_leaves_a_healthy_install_alone(monkeypatch):
    module, ran = _repair_module(monkeypatch, needed = [False], resident_after = True)
    assert module._repair_diffusers_main() == 1 and ran == []


def test_the_startup_repair_waits_out_a_peer_holding_the_pass(monkeypatch):
    """Returning at once would let the backend that started it import diffusers while a sibling
    backend's repair, or an update, is still replacing it."""
    module, ran = _repair_module(
        monkeypatch, needed = [False], resident_after = True, uncontended = [False, False, True]
    )
    assert module._repair_diffusers_main() == 1 and ran == []
    assert module.lock_polls() == 3
    module, ran = _repair_module(
        monkeypatch, needed = [True], resident_after = True, uncontended = [False, True]
    )
    assert module._repair_diffusers_main() == 0 and ran == [True]


def test_a_recorded_repair_failure_still_waits_for_a_peer(monkeypatch):
    """An update retrying the failed build holds the pass while it rewrites diffusers."""
    module, ran = _repair_module(
        monkeypatch, needed = [True], resident_after = False, uncontended = [False, True]
    )
    monkeypatch.setattr(
        module.install_manifest,
        "read_manifest",
        lambda *a, **k: {module._DIFFUSERS_MAIN_REPAIR_KEY: "failed"},
    )
    assert module._repair_diffusers_main() == 1 and ran == []
    assert module.lock_polls() == 2


@pytest.mark.parametrize("git", [True, False])
def test_the_startup_prefetch_fills_the_cache_and_never_the_environment(monkeypatch, git):
    """The backend stops this at its deadline, which is only safe while it installs nothing here."""
    import types

    module = _probe_module("install_python_stack_prefetch_probe")
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(module, "_diffusers_main_needs_dependency_pass", lambda: True)
    monkeypatch.setattr(module, "_startup_repair_failed", lambda: False)
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: True)
    monkeypatch.setattr(module, "_has_working_git", lambda: git)
    for forbidden in ("_diffusers_main_step", "pip_install_try", "pip_install"):
        monkeypatch.setattr(module, forbidden, lambda *a, **k: pytest.fail("installed"))
    monkeypatch.setattr(
        module.install_manifest, "pass_lock", lambda *a, **k: pytest.fail("took the pass lock")
    )
    runs = []

    def fake_run(cmd, **kwargs):
        runs.append(cmd)
        target = pathlib.Path(cmd[cmd.index("--target") + 1])
        assert target.is_dir()
        return types.SimpleNamespace(returncode = 0 if len(runs) == 1 else 1, stdout = b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._prefetch_diffusers_main() == 0
    cmd = runs[0]
    assert cmd[:3] == ["uv", "pip", "install"] and "--no-deps" in cmd
    target = pathlib.Path(cmd[cmd.index("--target") + 1])
    assert not target.exists(), "the scratch install is thrown away"
    if git:
        assert "-r" in cmd
    else:
        assert any(arg.startswith("diffusers @ https://") for arg in cmd)
    assert module._prefetch_diffusers_main() == 2
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: False)
    assert module._prefetch_diffusers_main() == 1 and len(runs) == 2, "pip has no cache to fill"


def _release_module(
    monkeypatch,
    name,
    installed = "0.41.0",
):
    module = _probe_module(name, active = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(
        module,
        "_installed_distribution_version",
        lambda dist: installed if dist == "diffusers" else None,
    )
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda dist: installed is not None)
    monkeypatch.setattr(module, "_recorded_direct_url", lambda dist: None)
    return module


def test_a_missing_release_still_forces_the_pass(monkeypatch):
    module = _release_module(monkeypatch, "install_python_stack_release_damaged", "0.41.0")
    assert module._diffusers_main_needs_dependency_pass() is False
    # Same-version damage would not converge: 11b's version check skips it.
    monkeypatch.setattr(module, "_payload_recorded_intact", lambda dist: False)
    assert module._diffusers_main_needs_dependency_pass() is False
    module = _release_module(monkeypatch, "install_python_stack_release_missing", None)
    assert module._diffusers_main_needs_dependency_pass() is True


def test_the_shipped_main_step_skips_without_installing(monkeypatch):
    module = _release_module(monkeypatch, "install_python_stack_release_step")
    monkeypatch.setattr(module, "pip_install_try", lambda *a, **k: pytest.fail("installed"))
    monkeypatch.setattr(module, "_has_working_git", lambda: pytest.fail("probed git"))
    progressed, steps = [], {}
    monkeypatch.setattr(module, "_progress", lambda label, *a, **k: progressed.append(label))
    monkeypatch.setattr(module, "_record_step", lambda name, state: steps.__setitem__(name, state))
    module._diffusers_main_step()
    assert len(progressed) == 1 and "skipped" in progressed[0], progressed
    assert steps == {"diffusers-main.txt": "skipped"}


@pytest.mark.parametrize(
    "installed, forced",
    [
        ("0.41.0", False),
        ("0.41.0.dev0", True),
        ("0.41.0rc1", True),
        ("0.42.0", False),
        ("0.40.0", True),
    ],
)
def test_the_fast_path_is_forced_only_by_a_release_behind_the_pin(monkeypatch, installed, forced):
    module = _release_module(monkeypatch, "install_python_stack_release_fast", installed)
    monkeypatch.setattr(module, "_has_working_git", lambda: pytest.fail("probed git"))
    assert module._diffusers_release_target() == "0.41.0"
    assert module._diffusers_main_needs_dependency_pass() is forced


def test_a_git_or_zip_build_of_the_release_keeps_the_fast_path(monkeypatch):
    module = _release_module(monkeypatch, "install_python_stack_release_built", "0.41.0.dev0")
    for direct in ({"vcs_info": {"vcs": "git"}}, {"archive_info": {}}):
        monkeypatch.setattr(module, "_recorded_direct_url", lambda dist, d = direct: dict(d, url = "x"))
        assert module._diffusers_main_needs_dependency_pass() is False


def test_the_release_prefetch_fetches_the_pin_into_scratch_only_when_behind(monkeypatch):
    module = _release_module(monkeypatch, "install_python_stack_release_prefetch", "0.41.0")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: pytest.fail("bootstrapped uv"))
    assert module._prefetch_diffusers_main() == 1

    module = _release_module(monkeypatch, "install_python_stack_release_prefetch2", "0.40.0")
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: True)
    monkeypatch.setattr(module, "_has_working_git", lambda: pytest.fail("probed git"))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return module.subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(module.subprocess, "run", run)
    # Windows ARM64 rewrites the file to a filtered copy, so read what was asked for, not the path.
    asked = []
    real_effective = module._effective_requirements
    monkeypatch.setattr(
        module,
        "_effective_requirements",
        lambda req: asked.append(req.name) or real_effective(req),
    )
    assert module._prefetch_diffusers_main() == 0
    (cmd,) = calls
    assert "--target" in cmd and "--no-deps" not in cmd
    assert asked == ["diffusers-pin.txt"]
    assert "-c" in cmd and str(cmd[cmd.index("-c") + 1]).endswith("constraints.txt"), cmd


def _release_repair(
    monkeypatch,
    name,
    installed,
    manifest = None,
    install_ok = True,
):
    import contextlib

    state = {"version": installed}
    module = _release_module(monkeypatch, name)
    monkeypatch.setattr(
        module,
        "_installed_distribution_version",
        lambda dist: state["version"] if dist == "diffusers" else None,
    )
    monkeypatch.setattr(
        module.install_manifest, "pass_lock", lambda *a, **k: contextlib.nullcontext(True)
    )
    monkeypatch.setattr(module.install_manifest, "read_manifest", lambda *a, **k: manifest or {})
    recorded = []
    monkeypatch.setattr(
        module.install_manifest, "update_manifest", lambda **extra: recorded.append(extra)
    )
    monkeypatch.setattr(module, "_bootstrap_uv", lambda: True)
    monkeypatch.setattr(module, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(module, "_diffusers_main_step", lambda: pytest.fail("ran the git step"))
    installs = []

    def fake_install(
        label,
        *args,
        req = None,
        **kwargs,
    ):
        installs.append(req)
        if install_ok:
            state["version"] = "0.41.0"
        return install_ok

    monkeypatch.setattr(module, "pip_install_try", fake_install)
    return module, installs, recorded


def test_the_startup_repair_installs_the_release_behind_the_pin(monkeypatch):
    # An old git-main failure must not strand the PyPI release.
    old = {"diffusers_main_repair": "failed", "step_results": {"diffusers-main.txt": "failed"}}
    module, installs, recorded = _release_repair(monkeypatch, "rel_repair_ok", "0.40.0", old)
    assert module._repair_diffusers_main() == 0
    assert [req.name for req in installs] == ["diffusers-pin.txt"] and recorded == []


def test_the_startup_repair_leaves_a_current_release_alone(monkeypatch):
    for installed in ("0.41.0", "0.41.0.post1"):
        module, installs, _ = _release_repair(monkeypatch, "rel_repair_noop", installed)
        assert module._repair_diffusers_main() == 1 and installs == []


def test_the_release_repair_leaves_a_user_build_alone(monkeypatch):
    module, installs, _ = _release_repair(monkeypatch, "rel_repair_user", "0.40.0")
    monkeypatch.setattr(
        module, "_recorded_direct_url", lambda dist: {"url": "file:///x", "dir_info": {}}
    )
    assert module._repair_diffusers_main() == 1 and installs == []


def test_a_failed_release_repair_is_recorded_and_not_retried(monkeypatch):
    module, installs, recorded = _release_repair(
        monkeypatch, "rel_repair_fail", "0.40.0", install_ok = False
    )
    assert module._repair_diffusers_main() == 2 and len(installs) == 1
    assert recorded == [{"diffusers_release_repair": "failed"}]
    module, installs, _ = _release_repair(
        monkeypatch, "rel_repair_fail2", "0.40.0", {"diffusers_release_repair": "failed"}
    )
    assert module._repair_diffusers_main() == 1 and installs == []
