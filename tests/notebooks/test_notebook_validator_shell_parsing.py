"""torch / torchcodec ABI guardrails (unslothai/unsloth#7225)."""

from __future__ import annotations
import importlib.util
import re
import sys
import types
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
IMPORT_FIXES_PATH = REPO_ROOT / "unsloth" / "import_fixes.py"
EXTRAS_NO_DEPS_TXT = REPO_ROOT / "studio" / "backend" / "requirements" / "extras-no-deps.txt"
SECURITY_AUDIT_YML = REPO_ROOT / ".github" / "workflows" / "security-audit.yml"
COLAB_TORCH211 = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
TORCHCODEC_WHEEL = (
    "https://download.pytorch.org/whl/torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
)


def _load_notebook_validator_module():
    """Load by path: `from scripts import ...` picks up whatever `scripts`
    package happens to be on sys.path first, which is not always this repo's."""
    spec = importlib.util.spec_from_file_location(
        "unsloth_notebook_validator_under_test",
        REPO_ROOT / "scripts" / "notebook_validator.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules: register before executing.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def test_notebook_validator_rejects_legacy_codec_past_last_lockstep_row():
    """The mirrored notebook rule must flag the same pairing as import_fixes."""
    nv = _load_notebook_validator_module()

    cell = '!pip install --no-deps "torch==2.12.1" "torchcodec==0.11.1"'
    findings = nv.rule_inst_004_torchcodec_torch(cell, {}, "nb.ipynb", 0)
    assert len(findings) == 1
    assert findings[0].rule == "R-INST-004"
    assert findings[0].severity == "error"
    assert "torchcodec>=0.12.0" in findings[0].hint

    old = '!pip install --no-deps "torch==2.4.0" "torchcodec==0.0.3"'
    assert nv.rule_inst_004_torchcodec_torch(old, {}, "nb.ipynb", 0) == []


def test_notebook_validator_allows_abi_stable_pairing():
    """R-INST-004 is an error-severity rule: it must not fire on torch 2.11 + 0.12+."""
    nv = _load_notebook_validator_module()

    cell = '!pip install --no-deps "torch==2.11.0" "torchcodec==0.15.0"'
    assert nv.rule_inst_004_torchcodec_torch(cell, {}, "nb.ipynb", 0) == []

    stale = '!pip install --no-deps "torch==2.11.0" "torchcodec==0.10.0"'
    findings = nv.rule_inst_004_torchcodec_torch(stale, {}, "nb.ipynb", 0)
    assert len(findings) == 1
    assert findings[0].rule == "R-INST-004"


def test_notebook_validator_accepts_its_own_torchcodec_remedy():
    """The hint is a `>=` pin and resolved_set() drops `>=`, so following it changed nothing."""
    nv = _load_notebook_validator_module()

    broken = '!pip install "torch==2.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(broken, COLAB_TORCH211, "nb.ipynb", 0)

    fixed = '!pip install "torch==2.12.0" "torchcodec>=0.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(fixed, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_reads_torch_range_pins():
    """A `torch>=` floor above Colab's torch moves pip while torchcodec stays put."""
    nv = _load_notebook_validator_module()

    ranged = '!pip install "torch>=2.12"'
    findings = nv.rule_inst_004_torchcodec_torch(ranged, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec>=0.12.0" in findings[0].hint


def test_notebook_validator_ignores_lower_bounds_under_the_resolved_version():
    """A floor below the installed version changes nothing, so it must not be flagged."""
    nv = _load_notebook_validator_module()

    for cell in ('!pip install "torchcodec>=0.10"', '!pip install "torch>=2.9"'):
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], cell


def test_notebook_validator_reads_the_bounds_in_invocation_order():
    """Whichever bound runs last is the one pip leaves behind, in both directions."""
    nv = _load_notebook_validator_module()

    downgraded = (
        '!pip install "torch==2.12.0" "torchcodec>=0.12"\n'
        '!pip install --no-deps "torchcodec==0.11.1"'
    )
    findings = nv.rule_inst_004_torchcodec_torch(downgraded, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1, "a later exact pin must win over an earlier floor"
    assert "torchcodec==0.11.1" in findings[0].message

    upgraded = '!pip install "torch==2.12.0" "torchcodec==0.11.1"\n!pip install "torchcodec>=0.12"'
    assert nv.rule_inst_004_torchcodec_torch(upgraded, COLAB_TORCH211, "nb.ipynb", 0) == []

    # The reported torch must be the one the notebook ends on, not the discarded floor.
    retreated = '!pip install "torch>=2.12"\n!pip install "torch==2.10.0"'
    findings = nv.rule_inst_004_torchcodec_torch(retreated, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torch==2.10.0" in findings[0].message

    # `<=` closes the same gap as `==`, and only downwards.
    capped = '!pip install "torch==2.12.0" "torchcodec>=0.12"\n!pip install "torchcodec<=0.11"'
    assert len(nv.rule_inst_004_torchcodec_torch(capped, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    lifted = '!pip install "torch==2.12.0" "torchcodec<=0.11"\n!pip install "torchcodec>=0.12"'
    assert nv.rule_inst_004_torchcodec_torch(lifted, COLAB_TORCH211, "nb.ipynb", 0) == []

    # A one-line range is not a downgrade: the floor still decides.
    ranged = '!pip install "torch==2.12.0" "torchcodec>=0.12,<=0.13"'
    assert nv.rule_inst_004_torchcodec_torch(ranged, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_honours_a_torchcodec_uninstall():
    """Removing the codec is a valid answer to the mismatch, so the new post-2.11 branch
    must not report one. parse_pip_line drops the `install`/`uninstall` word before the
    package list, so without the action an uninstall reads exactly like an install."""
    nv = _load_notebook_validator_module()

    removed = '!pip uninstall -y torchcodec\n!pip install "torch==2.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(removed, COLAB_TORCH211, "nb.ipynb", 0) == []

    # Put back incompatibly and it is a finding again; put back compatibly and it is not.
    back_stale = (
        "!pip uninstall -y torchcodec\n" '!pip install "torch==2.12.0" "torchcodec==0.11.1"'
    )
    assert len(nv.rule_inst_004_torchcodec_torch(back_stale, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    back_ok = "!pip uninstall -y torchcodec\n" '!pip install "torch==2.12.0" "torchcodec==0.13.0"'
    assert nv.rule_inst_004_torchcodec_torch(back_ok, COLAB_TORCH211, "nb.ipynb", 0) == []

    # Uninstalling after a good install still leaves nothing to flag.
    dropped = '!pip install "torch==2.12.0" "torchcodec==0.13.0"\n!pip uninstall -y torchcodec'
    assert nv.rule_inst_004_torchcodec_torch(dropped, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_keeps_an_absent_package_unknown():
    """Only `==` names a version. A floor on a package that is not there resolves to the
    newest release satisfying it, which the bound does not name, so the pairing stays
    unknown rather than being pinned to the floor and reported."""
    nv = _load_notebook_validator_module()

    reinstalled = '!pip uninstall -y torchcodec\n!pip install "torchcodec>=0.10"'
    assert nv.rule_inst_004_torchcodec_torch(reinstalled, COLAB_TORCH211, "nb.ipynb", 0) == []

    # Same when nothing supplies a baseline at all (a non-Colab notebook).
    floor_only = '!pip install "torch==2.11.0" "torchcodec>=0.10"'
    assert nv.rule_inst_004_torchcodec_torch(floor_only, {}, "nb.ipynb", 0) == []

    # An exact pin still names one, with or without a baseline.
    exact = '!pip install --no-deps "torch==2.12.1" "torchcodec==0.11.1"'
    assert len(nv.rule_inst_004_torchcodec_torch(exact, {}, "nb.ipynb", 0)) == 1


def test_notebook_validator_splits_chained_shell_commands():
    """`pip uninstall x && pip install x==V` is two actions on one line. Read as one, the
    reinstall lands in the uninstall's package list and the pairing disappears."""
    nv = _load_notebook_validator_module()

    invocations = list(
        nv.iter_pip_invocations('!pip uninstall -y torchcodec && pip install "torchcodec==0.11.1"')
    )
    assert [inv.action for inv in invocations] == ["uninstall", "install"]
    assert [inv.packages for inv in invocations] == [["torchcodec"], ["torchcodec==0.11.1"]]

    for sep in ("&&", ";"):
        cell = (
            f'!pip uninstall -y torchcodec {sep} pip install "torchcodec==0.11.1"\n'
            '!pip install "torch==2.12.0"'
        )
        assert len(nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)) == 1, sep

    healed = (
        '!pip uninstall -y torchcodec && pip install "torchcodec==0.13.0"\n'
        '!pip install "torch==2.12.0"'
    )
    assert nv.rule_inst_004_torchcodec_torch(healed, COLAB_TORCH211, "nb.ipynb", 0) == []

    # A `;` inside a PEP 508 marker is one argument, not a separator.
    marked = "!pip install \"torch==2.12.0; python_version >= '3.10'\""
    assert nv._split_chained(marked) == [(marked, False)]
    assert len(nv.rule_inst_004_torchcodec_torch(marked, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_skips_the_fallback_side_of_an_or_chain():
    """`A || B` runs B only when A failed, so replaying both reports a codec the notebook
    does not have. The left side still counts."""
    nv = _load_notebook_validator_module()

    fallback = (
        '!pip install "torchcodec==0.13.0" || pip install "torchcodec==0.11.1"\n'
        '!pip install "torch==2.12.0"'
    )
    assert nv.rule_inst_004_torchcodec_torch(fallback, COLAB_TORCH211, "nb.ipynb", 0) == []

    primary = (
        '!pip install "torchcodec==0.11.1" || pip install "torchcodec==0.13.0"\n'
        '!pip install "torch==2.12.0"'
    )
    assert len(nv.rule_inst_004_torchcodec_torch(primary, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_reads_compatible_release_pins():
    """`~=2.12.0` implies `>=2.12.0`, so its floor moves the baseline up."""
    nv = _load_notebook_validator_module()

    upgraded = '!pip install "torch~=2.12.0"'
    findings = nv.rule_inst_004_torchcodec_torch(upgraded, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec>=0.12.0" in findings[0].hint

    remedied = '!pip install "torch==2.12.0" "torchcodec~=0.13.0"'
    assert nv.rule_inst_004_torchcodec_torch(remedied, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_stops_at_a_shell_comment():
    """An unquoted `#` comments out the rest of the line, so a `;` inside it is not a
    separator and the commented-out install must not be replayed."""
    nv = _load_notebook_validator_module()

    cell = '!pip install "torch==2.12.0" # keep codec; pip install "torchcodec==0.13.0"'
    assert nv._split_chained(cell) == [('!pip install "torch==2.12.0"', False)]
    assert len(nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # A control operator ends a word, so `;#` opens a comment with no space in front of it.
    tight = '!pip install "torch==2.12.0";# keep codec; pip install "torchcodec==0.13.0"'
    assert nv._split_chained(tight) == [('!pip install "torch==2.12.0"', False)]
    assert len(nv.rule_inst_004_torchcodec_torch(tight, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # A `#` inside a word is not a comment: it is part of the argument.
    fragment = '!pip install "torchcodec==0.13.0#egg=x"'
    assert nv._split_chained(fragment) == [(fragment, False)]


def test_notebook_validator_resumes_after_an_or_list():
    """`A || B; C` runs C whatever A did. Only the conditional tail is dropped, and a `;`
    ends it."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained(
        "!pip install a && pip install b || pip install c ; pip install d"
    ) == [
        ("!pip install a", False),
        ("!pip install b", False),
        ("!pip install c", True),
        ("!pip install d", False),
    ]

    cell = '!pip install "torchcodec==0.11.1" || echo failed; pip install "torch==2.12.0"'
    assert len(nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_respects_escaped_separators():
    """A backslash-escaped `;` is part of the argument, the way shlex reads it in
    parse_pip_line. Split on it and the fragment ends in a backslash and parses as nothing."""
    nv = _load_notebook_validator_module()

    escaped = "!pip install torch==2.12.0\;\\ python_version\\ \\>\\=\\ \\'3.10\\'"
    assert nv._split_chained(escaped) == [(escaped, False)]
    assert [inv.packages for inv in nv.iter_pip_invocations(escaped)] == [
        ["torch==2.12.0; python_version >= '3.10'"]
    ]
    assert len(nv.rule_inst_004_torchcodec_torch(escaped, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_merges_repeated_requirements_in_one_command():
    """pip intersects a project named twice in one command, so the bounds are one window.
    Applied one argument at a time the floor lands first and the ceiling then clears it."""
    nv = _load_notebook_validator_module()

    split_window = '!pip install "torchcodec>=0.10" "torchcodec<0.11"'
    findings = nv.rule_inst_004_torchcodec_torch(split_window, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec==0.10" in findings[0].message

    # Same answer as the comma spelling, which is the point.
    comma = '!pip install "torchcodec>=0.10,<0.11"'
    assert nv.rule_inst_004_torchcodec_torch(comma, COLAB_TORCH211, "nb.ipynb", 0) == findings

    # A window the baseline already sits in is still a no-op.
    wide = '!pip install "torchcodec>=0.10" "torchcodec<0.12"'
    assert nv.rule_inst_004_torchcodec_torch(wide, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_resumes_after_an_and_following_an_or():
    """And-or lists are left-associative: in `A || B && C`, C runs when A succeeded, so the
    conditional tail ends at the `&&`."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!pip install a || pip install b && pip install c") == [
        ("!pip install a", False),
        ("!pip install b", True),
        ("!pip install c", False),
    ]

    cell = '!pip install "torchcodec==0.11.1" || echo failed && pip install "torch==2.12.0"'
    assert len(nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_will_not_name_a_multi_minor_window():
    """Moving down lands on the newest release the window admits, which the floor only names
    when there is one minor to land in."""
    nv = _load_notebook_validator_module()

    assert nv._window_names_one_minor("0.10", "0.11")
    assert not nv._window_names_one_minor("0.10", "0.12")

    # pip picks 0.11 here, which torch 2.11 is fine with, so nothing is reported.
    spanning = '!pip install "torchcodec==0.15"\n!pip install "torchcodec>=0.10,<0.12"'
    assert nv.rule_inst_004_torchcodec_torch(spanning, COLAB_TORCH211, "nb.ipynb", 0) == []

    # One minor still names its floor.
    single = '!pip install "torchcodec>=0.10,<0.11"'
    assert len(nv.rule_inst_004_torchcodec_torch(single, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_reads_a_direct_wheel_install():
    """pip takes an archive URL as an install target and parse_spec skips it, so a wheel that
    replaces the codec used to read as no install at all."""
    nv = _load_notebook_validator_module()

    assert nv._archive_requirement(TORCHCODEC_WHEEL) == ("torchcodec", "0.13.0")
    assert nv._archive_requirement("torchcodec==0.13.0") is None
    assert nv._archive_requirement("git+https://github.com/meta-pytorch/torchcodec.git") is None
    # A URL spells the local tag percent-encoded.
    assert nv._archive_requirement(
        "https://x/torch-2.12.0%2Bcu130-cp312-cp312-linux_x86_64.whl"
    ) == ("torch", "2.12.0+cu130")

    compatible = f'!pip install "torch==2.12.0" {TORCHCODEC_WHEEL}'
    assert nv.rule_inst_004_torchcodec_torch(compatible, COLAB_TORCH211, "nb.ipynb", 0) == []

    stale = compatible.replace("0.13.0", "0.10.0").replace("torch==2.12.0", "torch==2.11.0")
    assert len(nv.rule_inst_004_torchcodec_torch(stale, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_replays_exclusions():
    """`!=` rules out what is installed, so keeping it reports a version pip cannot leave in
    place. Without a floor to fall back on the pairing is unknown, not stale."""
    nv = _load_notebook_validator_module()

    assert nv._version_is_excluded("0.11.0+cu128", "0.11.*")
    assert nv._version_is_excluded("0.11.0", "0.11.0")
    assert not nv._version_is_excluded("0.11.0", "0.9.*")

    for pin in ("torchcodec!=0.11.*", "torchcodec!=0.11.0"):
        cell = f'!pip install "torch==2.12.0" "{pin}"'
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], pin

    # An exclusion that does not match leaves the baseline, and the pairing, alone.
    untouched = '!pip install "torch==2.12.0" "torchcodec!=0.9.*"'
    assert len(nv.rule_inst_004_torchcodec_torch(untouched, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # What is left after the exclusion spans minors, so the floor does not name it either.
    broad = '!pip install "torch==2.12.0" "torchcodec>=0.9,!=0.11.*"'
    assert nv.rule_inst_004_torchcodec_torch(broad, COLAB_TORCH211, "nb.ipynb", 0) == []

    # One minor left over still names its floor.
    narrow = '!pip install "torch==2.11.0" "torchcodec>=0.10,<0.11,!=0.11.*"'
    assert len(nv.rule_inst_004_torchcodec_torch(narrow, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_keeps_or_fallbacks_visible_to_other_rules():
    """The fallback still runs when the left side fails, so dropping it from
    iter_pip_invocations hid it from R-INST-001's git+ ban. Only the version replay skips it."""
    nv = _load_notebook_validator_module()

    cell = "!pip install foo || pip install git+https://example.com/evil.git"
    assert [inv.conditional for inv in nv.iter_pip_invocations(cell)] == [False, True]
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0))

    # The replay still ignores it, so the fallback codec is not reported as installed.
    fallback = (
        '!pip install "torchcodec==0.13.0" || pip install "torchcodec==0.11.1"\n'
        '!pip install "torch==2.12.0"'
    )
    assert nv.rule_inst_004_torchcodec_torch(fallback, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_will_not_name_an_exclusive_floor():
    """`>V` names the one version pip will not install, so on its own it says the install
    moved but not where. Only a ceiling that pins the minor can answer that."""
    nv = _load_notebook_validator_module()

    for cell in ('!pip install "torch>2.12"', '!pip install "torch>2.11.999"'):
        assert nv._effective_version(cell, "torch", "2.11.0+cu128") == (None, True), cell
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], cell

    # `>=` still names its endpoint, which is what the earlier rounds rest on.
    assert nv._effective_version('!pip install "torch>=2.12"', "torch", "2.11.0+cu128") == (
        "2.12",
        False,
    )
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install "torch>=2.12"', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_notebook_validator_treats_an_open_floor_as_a_floor():
    """pip takes the newest release above an open `>=`, so the floor is a lower bound and not
    the answer. It is enough for the ABI check, where everything above it gives the same
    answer, and not enough for the table, where it does not."""
    nv = _load_notebook_validator_module()

    old_pair = {"torch": "2.9.0+cu128", "torchcodec": "0.7.0+cu128"}

    # 0.8 is in the torch 2.9 row, but `>=0.8` can just as easily land on 0.16, so nothing is
    # proven either way and the rule stays quiet.
    assert (
        nv.rule_inst_004_torchcodec_torch('!pip install "torchcodec>=0.8"', old_pair, "nb.ipynb", 0)
        == []
    )

    # Every release above this floor is outside the torch 2.10 row, so it is provable.
    absent = {"torch": "2.10.0+cu128"}
    findings = nv.rule_inst_004_torchcodec_torch(
        '!pip install "torch==2.10.0" "torchcodec>=0.12.0"', absent, "nb.ipynb", 0
    )
    assert len(findings) == 1

    # An exact pin in the row is still accepted, and one outside it still reported.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec==0.8.0"', old_pair, "nb.ipynb", 0
        )
        == []
    )
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install "torchcodec==0.11.0"', old_pair, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_notebook_validator_ignores_a_conditional_pin_when_seeding():
    """resolved_set seeds the replay, so it has to skip the `||` fallback as well or the
    conditional pin comes back in through the seed."""
    nv = _load_notebook_validator_module()

    cell = '!pip install foo || pip install "torch==2.12.0"'
    assert nv.resolved_set(cell, COLAB_TORCH211)["torch"] == "2.11.0+cu128"
    assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_pads_release_segments_when_excluding():
    """PEP 440 pads the release segment, so `!=0.11` rules out an installed `0.11.0`."""
    nv = _load_notebook_validator_module()

    assert nv._version_is_excluded("0.11.0+cu128", "0.11")
    assert nv._version_is_excluded("0.11", "0.11.0")
    assert not nv._version_is_excluded("0.11.1", "0.11")

    cell = '!pip install "torch==2.12" "torchcodec!=0.11"'
    assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_unwraps_shell_groups():
    """A grouped command still runs, so leaving the bracket on it hides it from PIP_LINE_RE
    and with it from R-INST-001's git+ ban."""
    nv = _load_notebook_validator_module()

    for evil in (
        "!pip install foo || (pip install git+https://example.com/evil.git)",
        "!pip install foo || { pip install git+https://example.com/evil.git; }",
        "!(pip install git+https://example.com/evil.git)",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(evil, "nb.ipynb", 0)
        ), evil

    assert nv._unwrap_shell_group("!( pip install x )") == ("!pip install x", False)
    assert nv._unwrap_shell_group("{ pip install x") == ("pip install x", False)
    assert nv._unwrap_shell_group("}") == ("", False)
    assert nv._unwrap_shell_group("then pip install x") == ("pip install x", True)
    # `if pip install ...` is the test, which runs whenever the line is reached.
    assert nv._unwrap_shell_group("if pip install x") == ("pip install x", False)


def test_notebook_validator_skips_conditional_invocations_in_rule_002(monkeypatch):
    """resolved_set drops a fallback's pins, so a rule reading both has to drop the
    invocation as well or it checks that install against an environment it never made."""
    nv = _load_notebook_validator_module()
    monkeypatch.setattr(
        nv,
        "pypi_metadata",
        lambda name, version: {"info": {"requires_dist": ["tokenizers (>=0.30.0)"]}}
        if name.lower() == "transformers"
        else None,
    )
    colab = {"transformers": "5.0.0", "tokenizers": "0.22.2"}

    # Unconditional, so the mismatch against Colab's tokenizers is real and reported.
    plain = '!pip install --no-deps "transformers==5.5.0"'
    assert [f.rule for f in nv.rule_inst_002_no_deps_transitive(plain, colab, "nb.ipynb", 0)] == [
        "R-INST-002"
    ]

    # Behind an `||`, its pins never reach resolved_set, so checking it compares against an
    # environment this branch did not build.
    fallback = '!pip install foo || pip install --no-deps "transformers==5.5.0"'
    assert [inv.conditional for inv in nv.iter_pip_invocations(fallback)] == [False, True]
    assert nv.rule_inst_002_no_deps_transitive(fallback, colab, "nb.ipynb", 0) == []


def test_notebook_validator_bounds_the_minor_with_an_inclusive_cap():
    """`>=0.10,<=0.10.5` admits only the 0.10 line, so the window names it just as
    `>=0.10,<0.11` does."""
    nv = _load_notebook_validator_module()

    assert nv._window_names_one_minor("0.10", None, "0.10.5")
    assert not nv._window_names_one_minor("0.10", None, "0.11")

    older = {"torch": "2.11.0+cu128", "torchcodec": "0.9.0+cu128"}
    findings = nv.rule_inst_004_torchcodec_torch(
        '!pip install "torchcodec>=0.10,<=0.10.5"', older, "nb.ipynb", 0
    )
    assert len(findings) == 1
    assert "torchcodec==0.10" in findings[0].message

    # A cap that reaches into the next minor still cannot name where it lands.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec>=0.10,<=0.11"', older, "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_applies_exclusions_to_where_it_landed():
    """An exclusion has to hold for the version the requirement leaves in place, not only for
    the one that was there before it ran."""
    nv = _load_notebook_validator_module()

    newer = {"torch": "2.10.0+cu128", "torchcodec": "0.15.0"}
    # An earlier pin puts the codec above the cap, so the cap applies inside the replay
    # rather than in resolved_set. It says 0.11, the exclusion rules the whole 0.11 line out,
    # and pip lands below it: recording the cap reported a 0.11 codec against torch 2.10 when
    # what actually installs there is a 0.10 release the row allows.
    capped = '!pip install "torchcodec==0.15"\n!pip install "torchcodec<=0.11,!=0.11.*"'
    assert nv.rule_inst_004_torchcodec_torch(capped, newer, "nb.ipynb", 0) == []

    # A window that still names a minor after the exclusion keeps naming it.
    findings = nv.rule_inst_004_torchcodec_torch(
        '!pip install "torchcodec>=0.10,<0.11,!=0.11.*"',
        {"torch": "2.11.0+cu128", "torchcodec": "0.15.0"},
        "nb.ipynb",
        0,
    )
    assert len(findings) == 1
    assert "torchcodec==0.10" in findings[0].message


def test_notebook_validator_skips_conditional_invocations_in_rule_003():
    """The torchao floor helper reads the same cell, so a fallback that never runs must not
    satisfy the floor R-INST-003 is checking."""
    nv = _load_notebook_validator_module()

    colab = {"peft": "0.19.1", "torchao": "0.15.0"}
    assert (
        nv._install_cell_lower_bound('!pip install foo || pip install "torchao>=0.16.0"', "torchao")
        is None
    )
    assert [
        f.rule
        for f in nv.rule_inst_003_peft_torchao(
            '!pip install foo || pip install "torchao>=0.16.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-003"]

    # Run unconditionally and it does satisfy the floor.
    assert (
        nv.rule_inst_003_peft_torchao('!pip install "torchao>=0.16.0"', colab, "nb.ipynb", 0) == []
    )


def test_notebook_validator_moves_off_an_exclusive_endpoint():
    """`>V` is not satisfied by V, so an installed version sitting exactly on the endpoint
    still has to be replaced."""
    nv = _load_notebook_validator_module()

    on_the_endpoint = {"torch": "2.11.0+cu128", "torchcodec": "0.8.0"}
    assert nv._effective_version('!pip install "torchcodec>0.8.0"', "torchcodec", "0.8.0") == (
        None,
        True,
    )
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec>0.8.0"', on_the_endpoint, "nb.ipynb", 0
        )
        == []
    )

    # `>=` is satisfied by it, and a lower `>` leaves it alone, so both still report the
    # 0.8 codec against the torch 2.11 row.
    for cell in ('!pip install "torchcodec>=0.8.0"', '!pip install "torchcodec>0.7.0"'):
        assert (
            len(nv.rule_inst_004_torchcodec_torch(cell, on_the_endpoint, "nb.ipynb", 0)) == 1
        ), cell


def test_every_rule_reads_the_filtered_invocations():
    """Every reader asks what the cell leaves installed and takes the filtered iterator.
    R-INST-001 asks what could run at all, and answers it from whole lines instead."""
    nv = _load_notebook_validator_module()

    cell = "!pip install foo || pip install --no-deps git+https://example.com/evil.git"
    assert [inv.packages for inv in nv.unconditional_pip_invocations(cell)] == [["foo"]]
    assert len(list(nv.iter_pip_invocations(cell))) == 2

    # The ban reads whole lines, so no shell construct can put a git+ source out of reach.
    for evil in (
        cell,
        "!pip install foo || (pip install git+https://example.com/evil.git)",
        "!pip install foo || if command -v uv; then pip install git+https://example.com/evil.git; fi",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(evil, "nb.ipynb", 0)
        ), evil

    # Still line-scoped: the allowlist holds, and a line with no pip command is not an install.
    assert (
        nv.rule_inst_001_git_plus(
            "!pip install git+https://github.com/unslothai/unsloth-zoo.git", "nb.ipynb", 0
        )
        == []
    )
    assert nv.rule_inst_001_git_plus('x = "git+https://example.com/evil.git"', "nb.ipynb", 0) == []

    source = (REPO_ROOT / "scripts" / "notebook_validator.py").read_text(encoding = "utf-8")
    assert (
        source.count("in iter_pip_invocations(install_cell)") == 1
    ), "only unconditional_pip_invocations may read the raw iterator; rules take the filtered one"


def test_notebook_validator_keeps_a_group_conditional_throughout():
    """An `&&` or `;` inside a `(` or `{` group belongs to the group, so it does not end the
    fallback: if the left side succeeded, nothing in the group runs."""
    nv = _load_notebook_validator_module()

    for grouped in (
        '!pip install foo || (pip install bar && pip install "torch==2.12.0")',
        '!pip install foo || (pip install bar ; pip install "torch==2.12.0")',
    ):
        assert [flag for _, flag in nv._split_chained(grouped)] == [False, True, True], grouped
        assert (
            nv.rule_inst_004_torchcodec_torch(grouped, COLAB_TORCH211, "nb.ipynb", 0) == []
        ), grouped

    # Outside a group, and after one closes, the operator still ends the tail.
    for ungrouped in (
        '!pip install foo || pip install bar && pip install "torch==2.12.0"',
        '!pip install foo || (pip install bar) && pip install "torch==2.12.0"',
    ):
        assert [flag for _, flag in nv._split_chained(ungrouped)] == [False, True, False], ungrouped
        assert (
            len(nv.rule_inst_004_torchcodec_torch(ungrouped, COLAB_TORCH211, "nb.ipynb", 0)) == 1
        ), ungrouped

    # The git+ ban still sees inside the group, conditional or not.
    evil = "!pip install foo || (pip install bar && pip install git+https://example.com/evil.git)"
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(evil, "nb.ipynb", 0))


def test_notebook_validator_pads_the_minor_boundary():
    """`<0.11.0` and `<0.11` name the same boundary, so both windows hold one minor."""
    nv = _load_notebook_validator_module()

    assert nv.cmp_releases("0.11.0", "0.11") == 0
    assert nv.cmp_releases("0.11.1", "0.11") == 1
    assert nv._window_names_one_minor("0.10", "0.11.0")

    for cell in (
        '!pip install "torchcodec>=0.10,<0.11.0"',
        '!pip install "torchcodec>=0.10,<0.11"',
    ):
        assert (
            len(nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)) == 1
        ), cell

    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec>=0.10,<0.12"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_reads_an_archive_given_as_a_path():
    """`./torchcodec-0.13.0-...whl` parses as a project called `.`, so checking parse_spec
    first hid the wheel behind a name that never matches."""
    nv = _load_notebook_validator_module()

    for path in (
        "./torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "/tmp/torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    ):
        assert nv._archive_requirement(path) == ("torchcodec", "0.13.0"), path
        assert (
            nv.rule_inst_004_torchcodec_torch(
                f'!pip install "torch==2.12.0" {path}', COLAB_TORCH211, "nb.ipynb", 0
            )
            == []
        ), path

    stale = "./torchcodec-0.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                f'!pip install "torch==2.12.0" {stale}', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_git_allowlist_is_scoped_to_each_source():
    """One allowlisted repository on a line must not clear a prohibited one beside it. The
    line-level scan finds every `git+` target; the allowlist then applies to each."""
    nv = _load_notebook_validator_module()

    allowed = "git+https://github.com/unslothai/unsloth-zoo.git"
    evil = "git+https://example.com/evil.git"

    assert nv.rule_inst_001_git_plus(f"!pip install {allowed}", "nb.ipynb", 0) == []
    for cell in (
        f"!pip install {evil}",
        f"!pip install {allowed} ; pip install {evil}",
        f"!pip install {allowed} || pip install {evil}",
        f"!pip install {allowed} {evil}",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell

    two_allowed = "!pip install {} git+https://github.com/state-spaces/mamba.git".format(allowed)
    assert nv.rule_inst_001_git_plus(two_allowed, "nb.ipynb", 0) == []


def test_notebook_validator_keeps_the_stricter_of_two_equal_floors():
    """`>=0.8.0,>0.8.0` intersect to the exclusive one, so the installed 0.8.0 still moves."""
    nv = _load_notebook_validator_module()

    for spelling in ("torchcodec>=0.8.0,>0.8.0", "torchcodec>0.8.0,>=0.8.0"):
        assert nv._spec_window(nv.parse_spec(spelling).pins)[5] is True, spelling
        assert nv._effective_version(f'!pip install "{spelling}"', "torchcodec", "0.8.0") == (
            None,
            True,
        ), spelling

    # Two inclusive floors still name the endpoint.
    assert nv._effective_version(
        '!pip install "torchcodec>=0.8.0,>=0.8.0"', "torchcodec", "0.8.0"
    ) == ("0.8.0", True)


def test_notebook_validator_reads_a_named_direct_reference():
    """`name @ url` replaces the package even when the archive filename does not name it, so
    the old version cannot be reported as if it were still installed."""
    nv = _load_notebook_validator_module()

    tag = "torchcodec @ https://github.com/meta-pytorch/torchcodec/archive/refs/tags/v0.13.0.zip"
    assert nv._archive_requirement(tag) == ("torchcodec", None)
    assert (
        nv.rule_inst_004_torchcodec_torch(
            f'!pip install "torch==2.12.0" "{tag}"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )

    # A named reference whose archive does name a version still yields it, either way.
    wheel = "torchcodec @ https://x/torchcodec-0.10.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    assert nv._archive_requirement(wheel) == ("torchcodec", "0.10.0")
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                f'!pip install "torch==2.11.0" "{wheel}"', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_git_allowlist_matches_the_repository_not_a_substring():
    """An arbitrary repository can carry an allowlisted path inside its own, so the allowlist
    is compared against the normalised host and path."""
    nv = _load_notebook_validator_module()

    assert (
        nv._git_source_repository("git+https://user:pw@github.com/state-spaces/mamba.git@v2.0")
        == "github.com/state-spaces/mamba"
    )
    assert nv._git_source_is_allowed("git+https://github.com/unslothai/unsloth-zoo.git")
    assert not nv._git_source_is_allowed(
        "git+https://evil.example/repo/github.com/unslothai/unsloth.git"
    )

    smuggled = "!pip install git+https://evil.example/repo/github.com/unslothai/unsloth.git"
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(smuggled, "nb.ipynb", 0))

    # Credentials and a trailing ref do not stop an allowlisted repository from matching.
    assert (
        nv.rule_inst_001_git_plus(
            "!pip install git+https://user:pw@github.com/state-spaces/mamba.git@v2.0", "nb.ipynb", 0
        )
        == []
    )


def test_git_ban_reads_commands_not_the_comment():
    """`_split_chained` drops a shell comment, so a comment naming a prohibited source is
    documentation and must not fail the notebook."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!pip install foo # avoid git+https://example.com/evil.git", "nb.ipynb", 0
        )
        == []
    )
    # The executable half of the same line still counts.
    assert any(
        f.rule == "R-INST-001"
        for f in nv.rule_inst_001_git_plus(
            "!pip install git+https://example.com/evil.git # needed", "nb.ipynb", 0
        )
    )


def test_notebook_validator_ends_a_grouped_and_or_list_at_its_own_operator():
    """Which list an operator belongs to is its group depth. `(A || B && C)` is one list, so
    the `&&` ends the tail; `A || (B && C)` is not, so it does not."""
    nv = _load_notebook_validator_module()

    same_list = '!(pip install foo || pip install bar && pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(same_list)] == [False, True, False]
    assert len(nv.rule_inst_004_torchcodec_torch(same_list, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    inner = '!pip install foo || (pip install bar && pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(inner)] == [False, True, True]
    assert nv.rule_inst_004_torchcodec_torch(inner, COLAB_TORCH211, "nb.ipynb", 0) == []


def test_notebook_validator_keeps_a_minor_a_narrow_exclusion_cannot_remove():
    """`>=0.11,<0.12,!=0.11.0` still lands in the 0.11 line, and the minor is what the rule
    compares. Only a wildcard over the whole minor takes it away."""
    nv = _load_notebook_validator_module()

    assert nv._exclusion_covers_minor("0.11", "0.11.*")
    assert not nv._exclusion_covers_minor("0.11", "0.11.0")
    assert not nv._exclusion_covers_minor("0.11", "0.11.1.*")

    older = {"torch": "2.10.0+cu128", "torchcodec": "0.10.0"}
    for cell in (
        '!pip install "torchcodec>=0.11,<0.12,!=0.11.0"',
        '!pip install "torchcodec>=0.11,<=0.11.1,!=0.11"',
    ):
        assert len(nv.rule_inst_004_torchcodec_torch(cell, older, "nb.ipynb", 0)) == 1, cell

    # A wildcard over the minor still clears it.
    newer = {"torch": "2.10.0+cu128", "torchcodec": "0.15.0"}
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec>=0.10,<0.12,!=0.11.*"', newer, "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_keeps_an_outer_fallback_across_a_nested_or():
    """Each group keeps its own tail, so an inner `||` cannot hand the outer one back. A
    command is conditional when any level above it is in a fallback."""
    nv = _load_notebook_validator_module()

    nested = (
        "!pip install foo || (pip install bar || pip install baz && " 'pip install "torch==2.12.0")'
    )
    assert [flag for _, flag in nv._split_chained(nested)] == [False, True, True, True]
    assert nv.rule_inst_004_torchcodec_torch(nested, COLAB_TORCH211, "nb.ipynb", 0) == []

    # The grouped head list still ends its own tail at the `&&`.
    same_list = '!(pip install foo || pip install bar && pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(same_list)] == [False, True, False]
    assert len(nv.rule_inst_004_torchcodec_torch(same_list, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_lands_an_upward_move_on_an_inclusive_cap():
    """`<=V` allows V, so V is what pip picks, whichever side the version moves from."""
    nv = _load_notebook_validator_module()

    # 0.7 upward into a window that spans minors: the cap names where it stops.
    spanning = '!pip install "torchcodec==0.7.0"\n!pip install "torchcodec>=0.8,<=0.10.0"'
    findings = nv.rule_inst_004_torchcodec_torch(spanning, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec==0.10.0" in findings[0].message

    # An open floor has no cap to land on and stays a floor, which the ABI remedy needs.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch==2.12.0" "torchcodec>=0.12.0"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_will_not_keep_a_version_through_an_upgrade():
    """A bare name with `--upgrade` takes the newest release, so the installed version is not
    what the cell ends on. Without the flag pip leaves a satisfied requirement alone."""
    nv = _load_notebook_validator_module()

    # None of these let the installed version satisfy the requirement, so pip resolves from
    # the index and the old version is not what the cell ends on.
    for flag in ("--upgrade", "-U", "--force-reinstall", "--ignore-installed", "-I"):
        cell = f'!pip install {flag} "torch==2.12.0" torchcodec'
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], flag

    # No flag: the requirement is already satisfied, so 0.11 stays and is reported.
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install "torch==2.12.0" torchcodec', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )

    # A bound still bounds it, forced or not.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install --force-reinstall "torch==2.12.0" "torchcodec>=0.12.0"',
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install --upgrade "torch==2.12.0" "torchcodec>=0.12.0"',
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install --upgrade "torch==2.11.0" "torchcodec==0.10.0"',
                COLAB_TORCH211,
                "nb.ipynb",
                0,
            )
        )
        == 1
    )


def test_notebook_validator_keeps_the_flag_of_the_command_in_hand():
    """A group's closing bracket ends the level, not the command being read: the install
    before the `)` still belongs to the fallback the `||` opened."""
    nv = _load_notebook_validator_module()

    closing = '!(pip install foo || pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(closing)] == [False, True]
    assert nv.rule_inst_004_torchcodec_torch(closing, COLAB_TORCH211, "nb.ipynb", 0) == []

    # The same list ending its tail at an `&&` is unchanged.
    ended = '!(pip install foo || pip install bar && pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(ended)] == [False, True, False]
    assert len(nv.rule_inst_004_torchcodec_torch(ended, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_reads_a_compound_only_line():
    """With no standalone pip command on the line, every piece kept a keyword and none
    parsed, so the git+ ban never looked. The body runs, but only if its test did."""
    nv = _load_notebook_validator_module()

    for evil in (
        "!if command -v uv; then pip install git+https://example.com/evil.git; fi",
        "!while true; do pip install git+https://example.com/evil.git; done",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(evil, "nb.ipynb", 0)
        ), evil

    # The `if` test runs; only the `then` body is conditional, and that is the pip call here,
    # so the version replay leaves it alone.
    guarded = '!if command -v uv; then pip install "torch==2.12.0"; fi'
    assert [flag for _, flag in nv._split_chained(guarded)] == [False, True]
    assert nv.rule_inst_004_torchcodec_torch(guarded, COLAB_TORCH211, "nb.ipynb", 0) == []

    # An unguarded install on its own line still counts.
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install "torch==2.12.0"', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_git_ban_reads_the_arguments_shlex_produced():
    """`"git+"https://...` is one argument to pip and two words to a text scan, so the
    source has to be looked for in the parsed packages as well as in the command text."""
    nv = _load_notebook_validator_module()

    concatenated = '!pip install "git+"https://example.com/evil.git'
    assert any(
        f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(concatenated, "nb.ipynb", 0)
    )

    # The allowlist still applies to the joined argument.
    assert (
        nv.rule_inst_001_git_plus(
            '!pip install "git+"https://github.com/unslothai/unsloth-zoo.git', "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_keeps_a_pip_call_used_as_a_test():
    """`if pip install ...` is the condition, reached whenever the line is. Only a `then`,
    `elif`, `else` or `do` body depends on how that condition went."""
    nv = _load_notebook_validator_module()

    older = {"torch": "2.10.0+cu128", "torchcodec": "0.10.0+cu128"}
    for cell in (
        '!pip install foo; if pip install "torch==2.9.0"; then true; fi',
        '!while pip install "torch==2.9.0"; do true; done',
    ):
        assert len(nv.rule_inst_004_torchcodec_torch(cell, older, "nb.ipynb", 0)) == 1, cell

    # The bodies stay conditional.
    for cell in (
        '!if command -v uv; then pip install "torch==2.12.0"; fi',
        '!while true; do pip install "torch==2.12.0"; done',
    ):
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], cell


def test_git_ban_only_reads_pip_commands():
    """A `git+` in an `echo` beside an install installs nothing. The scan is per command, and
    only the ones that parse as pip count."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!echo git+https://example.com/evil.git; pip install numpy", "nb.ipynb", 0
        )
        == []
    )
    assert nv.rule_inst_001_git_plus("!echo git+https://example.com/evil.git", "nb.ipynb", 0) == []

    # The install beside it still counts when it is the one carrying the source.
    assert any(
        f.rule == "R-INST-001"
        for f in nv.rule_inst_001_git_plus(
            "!echo installing; pip install git+https://example.com/evil.git", "nb.ipynb", 0
        )
    )


def test_notebook_validator_evaluates_environment_markers():
    """pip skips a requirement whose marker is false, so replaying its bounds moves a version
    the cell never touches. The environment comes from the os-info oracle beside the pip
    freeze, and without one nothing is evaluated."""
    nv = _load_notebook_validator_module()

    assert nv._colab_python_version() is not None
    environment = nv._marker_environment(COLAB_TORCH211)
    assert environment is not None
    assert not nv._requirement_applies("torch>=2.12; python_version < '3.10'", environment)
    assert nv._requirement_applies("torch==2.12.0; python_version >= '3.10'", environment)
    assert nv._requirement_applies("torch==2.12.0", environment)
    # An unreadable marker is replayed rather than guessed at.
    assert nv._requirement_applies("torch==2.12.0; nonsense !!", environment)

    for skipped in (
        "!pip install \"torch>=2.12; python_version < '3.10'\"",
        "!pip install \"torch==2.12.0; python_version < '3.10'\"",
        "!pip install 'torch==2.12.0; sys_platform == \"win32\"'",
    ):
        assert (
            nv.rule_inst_004_torchcodec_torch(skipped, COLAB_TORCH211, "nb.ipynb", 0) == []
        ), skipped

    # A marker that holds is replayed, and so is one with no environment to judge it against.
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                "!pip install \"torch==2.12.0; python_version >= '3.10'\"",
                COLAB_TORCH211,
                "nb.ipynb",
                0,
            )
        )
        == 1
    )
    assert nv._marker_environment({}) is None
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install --no-deps "torch==2.12.1; python_version < \'3.10\'" "torchcodec==0.11.1"',
                {},
                "nb.ipynb",
                0,
            )
        )
        == 1
    )


def test_notebook_validator_expands_bundled_short_flags():
    """pip takes `-Uq`, and parse_pip_line keeps it as one token, so the letters are what
    gets compared."""
    nv = _load_notebook_validator_module()

    assert nv._forces_resolution({"-Uq"})
    assert nv._forces_resolution({"-qU"})
    assert nv._forces_resolution({"--upgrade"})
    assert not nv._forces_resolution({"-q"})
    assert not nv._forces_resolution({"--quiet"})

    for flag in ("-Uq", "-qU", "-qI"):
        cell = f'!pip install "torch==2.12.0" {flag} torchcodec'
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], flag

    # A quiet flag on its own does not re-resolve anything.
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!pip install "torch==2.12.0" -q torchcodec', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def _one_cell_notebook(source: str) -> dict:
    return {"cells": [{"cell_type": "code", "source": source.splitlines(keepends = True)}]}


def test_install_cell_discovery_finds_compound_commands():
    """The rules only see cells `install_cells` hands them, and it wanted `pip` right after
    the `!`, so every compound form was invisible to the real lint flow no matter what the
    splitter did with it."""
    nv = _load_notebook_validator_module()

    for source in (
        "!pip install torch==2.12.0",
        "!uv pip install torch",
        "!pip uninstall -y torchcodec",
        "!if command -v uv; then pip install git+https://example.com/x.git; fi",
        "!echo start; pip install torch==2.12.0",
        "!pip install foo || (pip install git+https://example.com/x.git)",
    ):
        assert nv.install_cells(_one_cell_notebook(source)), source

    # Still anchored on the `!`, so a pip mention in Python is not an install cell.
    for source in ('cmd = "pip install torch"', "import torch", "# pip install torch"):
        assert nv.install_cells(_one_cell_notebook(source)) == [], source


def test_notebook_validator_splits_on_single_control_operators():
    """`A & B` backgrounds A and runs B, `A | B` runs both. Neither opened a command boundary,
    so a line starting with something other than pip hid the install entirely."""
    nv = _load_notebook_validator_module()

    assert [c for c, _ in nv._split_chained("!sleep 1 & pip install x")] == [
        "!sleep 1",
        "!pip install x",
    ]
    assert [c for c, _ in nv._split_chained("!echo x | pip install y")] == [
        "!echo x",
        "!pip install y",
    ]

    for evil in (
        "!sleep 1 & pip install git+https://example.com/evil.git",
        "!echo x | pip install git+https://example.com/evil.git",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(evil, "nb.ipynb", 0)
        ), evil

    # A redirection is not a separator, and neither is a quoted ampersand.
    assert [c for c, _ in nv._split_chained("!pip install foo > log 2>&1")] == [
        "!pip install foo > log 2>&1"
    ]
    assert nv._split_chained('!pip install "a&b"')[0][0] == '!pip install "a&b"'


def test_torchao_floor_ignores_a_requirement_pip_skips():
    """R-INST-003 reads the floor from the same cell, so a requirement whose marker is false
    must not satisfy it either."""
    nv = _load_notebook_validator_module()

    colab = {"peft": "0.20.0", "torchao": "0.10.0"}
    assert [
        f.rule
        for f in nv.rule_inst_003_peft_torchao(
            "!pip install \"torchao>=0.16.0; python_version < '3.10'\"", colab, "nb.ipynb", 0
        )
    ] == ["R-INST-003"]

    # A marker that holds, and no marker at all, both still clear the floor.
    for cell in (
        "!pip install \"torchao>=0.16.0; python_version >= '3.10'\"",
        '!pip install "torchao>=0.16.0"',
    ):
        assert nv.rule_inst_003_peft_torchao(cell, colab, "nb.ipynb", 0) == [], cell


def test_git_allowlist_resolves_dot_segments_and_matches_exactly():
    """`unslothai/unsloth/../../attacker/repo` reads as an allowlisted prefix and resolves to
    somebody else's repository. Every entry is one `host/org/repo`, and pip puts a
    subdirectory in the fragment, so the match is exact."""
    nv = _load_notebook_validator_module()

    assert (
        nv._git_source_repository(
            "git+https://github.com/unslothai/unsloth/../../attacker/repo.git"
        )
        == "github.com/attacker/repo"
    )
    assert not nv._git_source_is_allowed(
        "git+https://github.com/unslothai/unsloth/../../attacker/repo.git"
    )
    assert not nv._git_source_is_allowed("git+https://github.com/unslothai/unsloth/extra.git")

    # The real forms still match: a ref, credentials, a fragment.
    for allowed in (
        "git+https://github.com/unslothai/unsloth.git",
        "git+https://user:pw@github.com/state-spaces/mamba.git@v2.0",
        "git+https://github.com/unslothai/unsloth-zoo.git#subdirectory=x",
    ):
        assert nv._git_source_is_allowed(allowed), allowed

    smuggled = "!pip install git+https://github.com/unslothai/unsloth/../../attacker/repo.git"
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(smuggled, "nb.ipynb", 0))


def test_install_cell_discovery_glues_continuations():
    """A `\\` continuation can put the `!` and the pip call on different physical lines, and
    discovery reads lines."""
    nv = _load_notebook_validator_module()

    source = "!echo ready && \\\n  pip install git+https://example.com/pkg.git"
    assert nv.install_cells(_one_cell_notebook(source))
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(source, "nb.ipynb", 0))


def test_no_deps_rules_skip_a_requirement_pip_skips(monkeypatch):
    """R-INST-002 and R-INST-005 read the raw pins themselves, so a marker-false `--no-deps`
    requirement must not be treated as installed by either."""
    nv = _load_notebook_validator_module()
    monkeypatch.setattr(
        nv,
        "pypi_metadata",
        lambda name, version: {"info": {"requires_dist": ["tokenizers (>=0.30.0)"]}}
        if name.lower() == "transformers"
        else None,
    )
    colab = {"transformers": "5.0.0", "tokenizers": "0.22.2"}

    skipped = "!pip install --no-deps \"transformers==5.5.0; python_version < '3.10'\""
    assert nv.rule_inst_002_no_deps_transitive(skipped, colab, "nb.ipynb", 0) == []
    assert nv.rule_inst_005_transformers_tokenizers(skipped, colab, "nb.ipynb", 0) == []

    for applied in (
        "!pip install --no-deps \"transformers==5.5.0; python_version >= '3.10'\"",
        '!pip install --no-deps "transformers==5.5.0"',
    ):
        assert nv.rule_inst_002_no_deps_transitive(applied, colab, "nb.ipynb", 0), applied
        assert nv.rule_inst_005_transformers_tokenizers(applied, colab, "nb.ipynb", 0), applied


def test_notebook_validator_splits_keywords_on_any_whitespace():
    """A tab after `then` is the same command to the shell. Splitting on a literal space left
    `then\\tpip` as one word, which parses as nothing and hides the install."""
    nv = _load_notebook_validator_module()

    tabbed = "!if true; then\tpip install git+https://example.com/pkg.git; fi"
    assert [c for c, _ in nv._split_chained(tabbed)][-1] == (
        "!pip install git+https://example.com/pkg.git"
    )
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(tabbed, "nb.ipynb", 0))

    spaced = "!if true; then pip install git+https://example.com/pkg.git; fi"
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(spaced, "nb.ipynb", 0))


def test_notebook_validator_comments_after_a_closing_bracket():
    """A `)` ends a word, so `)#` opens a comment and what follows is documentation."""
    nv = _load_notebook_validator_module()

    cell = "!(pip install unsloth)# alternative: pip install git+https://example.com/pkg.git"
    assert [c for c, _ in nv._split_chained(cell)] == ["!pip install unsloth"]
    assert nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0) == []


def test_notebook_validator_declines_markers_it_cannot_judge():
    """`Marker.evaluate` fills any field the environment omits from the running process, so a
    marker naming one would answer for this machine and move between runners."""
    nv = _load_notebook_validator_module()

    environment = nv._marker_environment(COLAB_TORCH211)
    for unknown in (
        "torch>=2.12; platform_release < '5.0'",
        "torch>=2.12; platform_version == 'x'",
        "torch>=2.12; implementation_version > '3'",
    ):
        assert nv._requirement_applies(unknown, environment), unknown

    # The fields the oracle can answer for are still evaluated.
    assert not nv._requirement_applies("torch>=2.12; python_version < '3.10'", environment)
    assert nv._requirement_applies("torch>=2.12; sys_platform == 'linux'", environment)


def test_notebook_validator_reads_case_arms():
    """Stripping `case` left `x in x) pip install ...`, which parses as nothing. Only the
    matching arm runs, so the command is conditional, and the git+ ban still sees it."""
    nv = _load_notebook_validator_module()

    single = "!case x in x) pip install git+https://example.com/pkg.git ;; esac"
    assert nv._split_chained(single) == [("!pip install git+https://example.com/pkg.git", True)]
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(single, "nb.ipynb", 0))

    # A later arm carries no keyword at all, just its own label.
    multi = (
        "!case x in a) pip install git+https://a.example/a.git ;; "
        "b) pip install git+https://b.example/b.git ;; esac"
    )
    assert [c for c, _ in nv._split_chained(multi)] == [
        "!pip install git+https://a.example/a.git",
        "!pip install git+https://b.example/b.git",
    ]
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(multi, "nb.ipynb", 0))

    # Conditional, so the version replay leaves an arm alone.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!case x in x) pip install "torch==2.12.0" ;; esac', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )

    # Bash accepts a quoted pattern, and the label still ends at its `)`.
    quoted_pattern = '!case x in "x") pip install git+https://example.com/pkg.git ;; esac'
    assert nv._split_chained(quoted_pattern) == [
        ("!pip install git+https://example.com/pkg.git", True)
    ]
    assert any(
        f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(quoted_pattern, "nb.ipynb", 0)
    )

    # A `)` inside a quoted argument, or inside a substitution, belongs to the command.
    assert nv._unquoted_arm_close('pip install "a)b"') is None
    assert nv._unquoted_arm_close("echo $(date) ; pip install a") is None
    quoted = '!pip install "a)b" git+https://example.com/evil.git'
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(quoted, "nb.ipynb", 0))


def test_notebook_validator_reads_pip_in_a_substitution():
    """`echo $(pip install x)` runs the install as surely as the echo, and the outer command
    is not pip, so the inner one has to be a command of its own."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!echo $(pip install git+https://example.com/pkg.git)",
        "!X=`pip install git+https://example.com/pkg.git`",
        "!echo $(echo $(pip install git+https://example.com/pkg.git))",
    ):
        assert "!pip install git+https://example.com/pkg.git" in [
            command for command, _ in nv._split_chained(cell)
        ], cell
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell

    assert nv._substitution_bodies("echo $(pip install x) and `pip install y`") == [
        "pip install x",
        "pip install y",
    ]
    assert nv._substitution_bodies("pip install x") == []


def test_notebook_validator_honours_escaped_case_patterns():
    """A `\\)` in a pattern matches a literal parenthesis, so it is not what closes the arm."""
    nv = _load_notebook_validator_module()

    escaped = "!case 'x)y' in x\\)y) pip install git+https://example.com/pkg.git ;; esac"
    assert nv._split_chained(escaped) == [("!pip install git+https://example.com/pkg.git", True)]
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(escaped, "nb.ipynb", 0))


def test_notebook_validator_ignores_marker_names_in_literals():
    """`sys_platform == 'platform_release'` references one variable, not two, so the marker
    is judged rather than declined."""
    nv = _load_notebook_validator_module()

    environment = nv._marker_environment(COLAB_TORCH211)
    assert not nv._requirement_applies(
        "torch==2.12.0; sys_platform == 'platform_release'", environment
    )
    assert nv._requirement_applies("torch==2.12.0; sys_platform == 'linux'", environment)
    # A real reference to an unknown field is still declined.
    assert nv._requirement_applies("torch>=2.12; platform_release < '5.0'", environment)


def test_notebook_validator_strips_execution_prefixes():
    """`command pip install ...` and `env FOO=1 pip install ...` install exactly as a bare
    `pip install ...` does, so the prefix must not hide them."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!command pip install git+https://example.com/pkg.git",
        "!env FOO=1 pip install git+https://example.com/pkg.git",
        "!FOO=1 pip install git+https://example.com/pkg.git",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell

    # The replay reads them too, so a prefixed install still moves the version.
    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!env FOO=1 pip install "torch==2.12.0"', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_notebook_validator_quotes_inside_a_substitution():
    """A `)` inside a quoted argument closes no substitution, so the body runs past it."""
    nv = _load_notebook_validator_module()

    cell = "!echo \"$(printf 'X)'; pip install git+https://example.com/pkg.git)\""
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0))

    # A backtick body survives the assignment-prefix strip, since the bodies are read off the
    # raw pieces.
    assert any(
        f.rule == "R-INST-001"
        for f in nv.rule_inst_001_git_plus(
            "!X=`pip install git+https://example.com/pkg.git`", "nb.ipynb", 0
        )
    )


def test_notebook_validator_tells_a_grouping_close_from_a_substitution_close():
    """`)` ends a word when it closes a grouping and not when it closes a `$( )` inside one,
    so only the first makes a following `#` a comment."""
    nv = _load_notebook_validator_module()

    # Bash prints `ok#suffix` here and runs the install.
    embedded = "!echo $(printf ok)#suffix; pip install git+https://example.com/pkg.git"
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(embedded, "nb.ipynb", 0))

    # A grouping close still opens a comment.
    grouped = "!(pip install unsloth)# git+https://example.com/pkg.git"
    assert nv.rule_inst_001_git_plus(grouped, "nb.ipynb", 0) == []


def test_notebook_validator_ignores_quoted_substitution_text():
    """Single quotes and an escaped `$` make the text literal, so nothing in it runs and the
    notebook must not be failed for it. Double quotes still expand."""
    nv = _load_notebook_validator_module()

    literal = "!echo '$(pip install git+https://example.com/pkg.git)'; pip install unsloth"
    assert nv._substitution_bodies(literal) == []
    assert nv.rule_inst_001_git_plus(literal, "nb.ipynb", 0) == []

    escaped = (
        '!echo "' + chr(92) + '$(pip install git+https://example.com/pkg.git)"; '
        "pip install unsloth"
    )
    assert nv.rule_inst_001_git_plus(escaped, "nb.ipynb", 0) == []

    # A substitution inside double quotes is expanded, so it still counts.
    expanded = "!echo \"$(printf 'X)'; pip install git+https://example.com/pkg.git)\""
    assert any(f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(expanded, "nb.ipynb", 0))


def test_notebook_validator_reads_process_substitutions():
    """`<( )` and `>( )` run their commands too."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!cat <(pip install git+https://example.com/pkg.git); pip install unsloth",
        "!tee >(pip install git+https://example.com/pkg.git); pip install unsloth",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell


def test_notebook_validator_keeps_substitution_commands_in_order():
    """A substitution runs before its host and its own separators are its own, so its commands
    stay in sequence rather than being split across the line and appended at the end."""
    nv = _load_notebook_validator_module()

    cell = (
        "!echo $(pip install torchcodec==0.12.0; pip install torchcodec==0.11.0); "
        "pip install torch==2.12.0"
    )
    commands = [command for command, _ in nv._split_chained(cell)]
    assert commands[0] == "!pip install torchcodec==0.12.0"
    assert commands[1] == "!pip install torchcodec==0.11.0"
    assert commands[-1] == "!pip install torch==2.12.0"

    # 0.11 is what is left installed, and torch 2.12 does not take it.
    findings = nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec==0.11.0" in findings[0].message


def test_git_sources_are_matched_case_insensitively():
    """pip normalises `Git+https://` to the same link, so the ban has to see it."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!pip install Git+https://example.com/pkg.git",
        "!pip install GIT+HTTPS://example.com/pkg.git",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell

    # The allowlist still clears an allowlisted repository whatever the case.
    assert (
        nv.rule_inst_001_git_plus(
            "!pip install Git+https://github.com/unslothai/unsloth-zoo.git", "nb.ipynb", 0
        )
        == []
    )


def test_notebook_validator_skips_a_prefixs_own_options():
    """`env -u VAR pip install ...` runs pip, and stripping only the word left the options in
    front of it. Rather than an option table per prefix, skip to where pip starts."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!env -u UNUSED pip install git+https://example.com/pkg.git",
        "!sudo -u root pip install git+https://example.com/pkg.git",
    ):
        assert any(
            f.rule == "R-INST-001" for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)
        ), cell

    # Only after a prefix: an ordinary command that merely mentions pip is untouched.
    assert (
        nv.rule_inst_001_git_plus(
            "!echo git+https://example.com/evil.git; pip install numpy", "nb.ipynb", 0
        )
        == []
    )

    assert (
        len(
            nv.rule_inst_004_torchcodec_torch(
                '!env -u X pip install "torch==2.12.0"', COLAB_TORCH211, "nb.ipynb", 0
            )
        )
        == 1
    )


def test_notebook_validator_keeps_redirections_and_quoted_process_forms():
    """`>|` overrides noclobber rather than piping, and `<( )` inside double quotes is text."""
    nv = _load_notebook_validator_module()

    redirected = (
        "!echo harmless >| pip install git+https://example.com/pkg.git; pip install unsloth"
    )
    assert nv.rule_inst_001_git_plus(redirected, "nb.ipynb", 0) == []

    quoted = '!echo "<(pip install git+https://example.com/pkg.git)"; pip install unsloth'
    assert nv.rule_inst_001_git_plus(quoted, "nb.ipynb", 0) == []

    # Unquoted it runs, and a real pipeline still separates.
    assert any(
        f.rule == "R-INST-001"
        for f in nv.rule_inst_001_git_plus(
            "!cat <(pip install git+https://example.com/pkg.git)", "nb.ipynb", 0
        )
    )
    assert any(
        f.rule == "R-INST-001"
        for f in nv.rule_inst_001_git_plus(
            "!echo x | pip install git+https://example.com/evil.git", "nb.ipynb", 0
        )
    )


def test_notebook_validator_reads_a_range_as_one_window():
    """A `>=X,<Y` pair is the same constraint as `~=X`, and the guard's own remedy is
    spelled that way (`pip install 'torchcodec>=0.11,<0.12.0'`), so the rule has to read it
    back. A `<` with nothing under it names no version and leaves the pairing unknown."""
    nv = _load_notebook_validator_module()

    # Colab is on 0.11, which this window excludes, so pip drops into the 0.10 line.
    narrowed = '!pip install "torchcodec>=0.10,<0.11"'
    assert len(nv.rule_inst_004_torchcodec_torch(narrowed, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # The window torch 2.11 actually wants is a no-op on the same baseline.
    matching = '!pip install "torchcodec>=0.11,<0.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(matching, COLAB_TORCH211, "nb.ipynb", 0) == []

    # Open below: the release pip picks is unnamed, so no stale baseline is kept either.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch<2.11"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )

    # An inclusive cap does name one, so it still clamps rather than clearing.
    capped = '!pip install "torch==2.12.0" "torchcodec>=0.12"\n!pip install "torchcodec<=0.11"'
    assert len(nv.rule_inst_004_torchcodec_torch(capped, COLAB_TORCH211, "nb.ipynb", 0)) == 1


def test_notebook_validator_reads_the_compatible_release_ceiling():
    """`~=` pins a window, so it moves the baseline down as well as up. PEP 440 drops the
    last component: `~=2.10.0` allows `<2.11`, `~=2.10` allows `<3`."""
    nv = _load_notebook_validator_module()

    assert nv._compatible_release_ceiling("2.10.0") == "2.11"
    assert nv._compatible_release_ceiling("2.10") == "3"
    assert nv._compatible_release_ceiling("2") is None

    # Colab is on torch 2.11, which `~=2.10.0` excludes, so pip drops into the 2.10 line.
    downgraded = '!pip install "torch~=2.10.0"'
    assert len(nv.rule_inst_004_torchcodec_torch(downgraded, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # `~=2.10` admits 2.11, and `~=2.11.0` is already satisfied. Neither moves anything.
    for cell in ('!pip install "torch~=2.10"', '!pip install "torch~=2.11.0"'):
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], cell


# ----------------------------------------------------------------------------------
# Codex review round 10 (2026-09-02): three P1 defects, each of which let
# R-INST-001 miss a `git+` install that Bash really would run. One test per item,
# each paired with the control that already passed, since what makes these bugs
# rather than gaps is that the neighbouring form was caught all along.
# ----------------------------------------------------------------------------------


def _git_plus_rules(line: str) -> list[str]:
    nv = _load_notebook_validator_module()
    return [f.rule for f in nv.rule_inst_001_git_plus(line, "t.ipynb", 0)]


def test_a_prefix_operand_named_pip_is_not_the_executable():
    """`env -u pip` unsets the variable `pip`; the command is the word after it.

    Selecting the first pip-looking word produced `pip pip install ...`, which PIP_LINE_RE
    rejects outright, so the install was never collected and the git+ source never seen.
    """
    assert _git_plus_rules("!env -u pip pip install git+https://example.com/pkg.git") == [
        "R-INST-001"
    ]
    # The control that always worked: the operand is spelled something else.
    assert _git_plus_rules("!env -u VAR pip install git+https://example.com/pkg.git") == [
        "R-INST-001"
    ]


def test_prefix_operands_are_consumed_for_every_supported_prefix():
    for line in (
        "!sudo -u pip pip install git+https://example.com/pkg.git",
        "!env --unset=pip pip install git+https://example.com/pkg.git",
        "!env -u pip -u also A=1 pip install git+https://example.com/pkg.git",
        "!nohup pip install git+https://example.com/pkg.git",
        "!env -- pip install git+https://example.com/pkg.git",
    ):
        assert _git_plus_rules(line) == ["R-INST-001"], line


def test_a_command_list_inside_backticks_is_not_split_at_its_own_separator():
    """Backticks run their contents like `$( )`, so the `;` inside one is not this line's.

    Splitting there handed `_substitution_bodies` an unmatched fragment, which it cannot
    read, so neither the nested pip command nor its git+ source was ever seen.
    """
    assert _git_plus_rules(
        "!echo `pip install git+https://example.com/pkg.git; echo ok`; pip install unsloth"
    ) == ["R-INST-001"]
    # The control: the same shape written as `$( )` was caught before this fix.
    assert _git_plus_rules(
        "!echo $(pip install git+https://example.com/pkg.git); pip install unsloth"
    ) == ["R-INST-001"]
    # Single quotes make a backtick literal, so nothing runs and nothing is flagged.
    assert _git_plus_rules("!echo 'a `pip install git+https://example.com/p.git` b'") == []


def test_every_case_arm_is_scanned_not_just_the_ones_before_an_empty_piece():
    """`;;` emits an empty piece and `esac` unwraps to "", so `out` outran `commands`.

    `zip` stopped at the shorter list, and the arm holding the substitution fell off the
    end without ever being handed to `_substitution_bodies`.
    """
    assert _git_plus_rules(
        "!case x in y) echo no;; x) echo $(pip install git+https://example.com/pkg.git);; esac"
    ) == ["R-INST-001"]
    # The control: in the first arm, before any empty piece has been dropped.
    assert _git_plus_rules(
        "!case x in x) echo $(pip install git+https://example.com/pkg.git);; y) echo no;; esac"
    ) == ["R-INST-001"]


def test_an_arm_close_paren_is_not_stripped_off_a_substitution():
    """`rstrip(")}")` ate the `)` that closes a `$( )` when no `(` had been stripped."""
    nv = _load_notebook_validator_module()
    text, conditional = nv._unwrap_shell_group(" x) echo $(pip install git+https://e.com/p.git)")
    assert conditional is True
    assert text.endswith(")"), text
    assert nv._substitution_bodies(text) == ["pip install git+https://e.com/p.git"]
    # A real group still loses its brackets.
    assert nv._unwrap_shell_group("( pip install x )")[0] == "pip install x"
