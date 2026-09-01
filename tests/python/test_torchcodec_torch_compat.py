# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

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


def _tomllib():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    return pytest.importorskip("tomli")


def _load_import_fixes_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test",
        IMPORT_FIXES_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pyproject_declares_torch210_audio_extra_with_python_gate():
    text = PYPROJECT.read_text(encoding = "utf-8")
    assert "audio-torch210 = [" in text
    assert "torchcodec>=0.10.0,<0.11.0" in text
    assert "python_version >= '3.10'" in text
    assert "audio-torch290 = [" in text
    assert "audio-torch280 = [" in text
    assert "\naudio = [" not in text


def _stub_torch(monkeypatch, version: str):
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = version
    monkeypatch.setitem(sys.modules, "torch", torch_mod)


def test_torch210_extras_bundle_audio_torch210():
    text = PYPROJECT.read_text(encoding = "utf-8")
    for extra in (
        "cu128-torch2100",
        "cu126-ampere-torch2100",
        "rocm72-torch2100",
    ):
        match = re.search(rf"^{extra} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
        assert match is not None, extra
        assert "unsloth[audio-torch210]" in match.group(1)


def test_torchcodec_matrix_matches_notebook_validator():
    from scripts import notebook_validator as nv
    fixes = _load_import_fixes_module()
    assert fixes._TORCH_TORCHCODEC_MINORS == nv.TORCH_TORCHCODEC


def test_torchcodec_exclusive_upper_bound():
    fixes = _load_import_fixes_module()
    assert fixes._torchcodec_exclusive_upper("0.10") == "<0.11.0"
    assert fixes._torchcodec_exclusive_upper("0.9") == "<0.10.0"


def test_torch290_rejects_torchcodec_07(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.9.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7.0")

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "audio-torch210" not in hint


def test_torch280_accepts_torchcodec_07(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.8.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7.0")

    assert fixes._torchcodec_version_mismatch_hint() is None


def test_torch210_rejects_torchcodec_011(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.11.0",
    )

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "torchcodec 0.11.0" in hint
    assert "audio-torch210" in hint
    assert "<0.11.0" in hint
    assert "<11.0" not in hint


def test_torch210_accepts_torchcodec_010(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.10.0+cu128",
    )

    assert fixes._torchcodec_version_mismatch_hint() is None


def test_import_fixes_loads_on_python39_syntax():
    """Regression: module must import on 3.9 (postponed annotations for str | None)."""
    fixes = _load_import_fixes_module()
    assert callable(fixes._torchcodec_version_mismatch_hint)


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


def _load_install_python_stack():
    studio_dir = REPO_ROOT / "studio"
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    import install_python_stack

    return install_python_stack


def test_torch211_rejects_torchcodec_010(monkeypatch):
    """The guard must not be silent on the torch minor where the mismatch happens."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.11.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.10.0+cu128",
    )

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None, "torch 2.11 + torchcodec 0.10 must not go unreported"
    assert "torchcodec 0.10.0+cu128" in hint
    assert "audio-torch211" in hint
    assert ">=0.11" in hint
    assert "<0.12.0" in hint
    assert "audio-torch210" not in hint


def test_torch211_accepts_torchcodec_011(monkeypatch):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.11.0+cu128")
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.11.1+cu128",
    )

    assert fixes._torchcodec_version_mismatch_hint() is None


def test_torch211_accepts_abi_stable_torchcodec(monkeypatch):
    """torchcodec 0.12+ targets torch >=2.11, so it is not locked to one minor."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    for torch_version in ("2.11.0+cu128", "2.12.0", "2.13.0+cu130"):
        for codec_version in ("0.12.0", "0.15.0+cu130"):
            _stub_torch(monkeypatch, torch_version)
            monkeypatch.setattr(importlib.metadata, "version", lambda _name, _v = codec_version: _v)
            assert (
                fixes._torchcodec_version_mismatch_hint() is None
            ), f"{torch_version} + torchcodec {codec_version} is supported upstream"


def test_torch210_still_rejects_abi_stable_torchcodec(monkeypatch):
    """The ABI-stable floor starts at torch 2.11: 2.10 keeps the exact pairing."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.15.0")

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "audio-torch210" in hint


def test_torch_past_last_lockstep_row_rejects_legacy_torchcodec(monkeypatch):
    """0.11 is pinned to torch 2.11 exactly, so 2.12/2.13 with a pre-0.12 codec still warns."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    for torch_version in ("2.12.1+cu130", "2.13.0"):
        for codec_version in ("0.11.1", "0.10.0"):
            _stub_torch(monkeypatch, torch_version)
            monkeypatch.setattr(importlib.metadata, "version", lambda _name, _v = codec_version: _v)
            hint = fixes._torchcodec_version_mismatch_hint()
            assert hint is not None, f"{torch_version} + torchcodec {codec_version} must warn"
            assert "torchcodec>=0.12.0" in hint
            # No audio-torch2xx extra exists for these minors, so none is offered.
            assert "unsloth[audio-torch" not in hint


def test_torch_below_the_table_stays_silent(monkeypatch):
    """A torch minor older than the matrix keeps the original no-opinion behaviour."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.4.0")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.0.3")
    assert fixes._torchcodec_version_mismatch_hint() is None


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


# Inlined from scripts/data/colab_pip_freeze.gpu.txt: the daily oracle refresh must not
# move these verdicts.
COLAB_TORCH211 = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}


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


def test_pyproject_declares_torch211_audio_extra_with_python_gate():
    text = PYPROJECT.read_text(encoding = "utf-8")
    match = re.search(r"^audio-torch211 = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert match is not None, "pyproject must declare an audio-torch211 extra"
    assert "torchcodec>=0.11.0,<0.12.0" in match.group(1)
    assert "python_version >= '3.10'" in match.group(1)


def test_security_audit_covers_every_installable_torchcodec_line():
    """extras-no-deps.txt used to pin torchcodec flat, so 0.10 was the only line that ever
    installed and the only one audited. `_select_torchcodec_spec` now picks per torch minor,
    and the repairs still resolve torch 2.10, 2.9 and 2.8, so each of those lines needs an
    input of its own: the ranges are disjoint and cannot share a resolve."""
    text = SECURITY_AUDIT_YML.read_text(encoding = "utf-8")
    ips = _load_install_python_stack()
    tomllib = _tomllib()
    extras = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))["project"][
        "optional-dependencies"
    ]

    audited = ["audio-torch211", "audio-torch210", "audio-torch290", "audio-torch280"]
    # Both halves of the workflow build the inputs; one is the advisory audit, one is
    # scan_packages. 211 is folded into unsloth-deps.txt, the rest get a file each.
    assert text.count('optional-dependencies"]["audio-torch211"]') == 2
    assert text.count("for extra in audio-torch210 audio-torch290 audio-torch280; do") == 2
    for extra in audited[1:]:
        assert f"audit-reqs/{extra}.txt" in text, extra
        assert f" {extra} " in text or f" {extra};" in text or f" {extra}'" in text, extra

    # scan_packages.py keeps one requirement per package name, so two disjoint torchcodec
    # ranges in one shard collapse to the first. They have to be scanned apart.
    shards = re.findall(r"files: '([^']+)'", text)
    for shard in shards:
        assert (
            sum(1 for extra in audited if extra in shard.split()) <= 1
        ), f"shard {shard!r} would have its torchcodec lines deduplicated to one"
    scanned = {extra for shard in shards for extra in audited if extra in shard.split()}
    assert scanned == {"audio-torch210", "audio-torch290", "audio-torch280"}

    # Whatever the selector installs on a reachable torch minor has to be in that set.
    for torch_minor in ("2.10", "2.9", "2.8"):
        spec = ips._select_torchcodec_spec(f"{torch_minor}.0")
        assert any(
            spec in dep for extra in audited for dep in extras[extra]
        ), f"torch {torch_minor} installs {spec}, which no audited extra declares"


def test_extras_no_deps_has_no_unconditional_torchcodec_pin():
    """A flat pin cannot serve both torch lines, so the installer picks the spec."""
    lines = [
        line.strip()
        for line in EXTRAS_NO_DEPS_TXT.read_text(encoding = "utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.lower().startswith("torchcodec") for line in lines), (
        "extras-no-deps.txt must not pin torchcodec unconditionally; "
        "install_python_stack._select_torchcodec_spec picks it per torch minor"
    )


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


TORCHCODEC_WHEEL = (
    "https://download.pytorch.org/whl/torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
)


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

    assert nv._unwrap_shell_group("!( pip install x )") == "!pip install x"
    assert nv._unwrap_shell_group("{ pip install x") == "pip install x"
    assert nv._unwrap_shell_group("}") == ""


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
        '!pip install foo || (pip install bar || pip install baz && '
        'pip install "torch==2.12.0")'
    )
    assert [flag for _, flag in nv._split_chained(nested)] == [False, True, True, True]
    assert nv.rule_inst_004_torchcodec_torch(nested, COLAB_TORCH211, "nb.ipynb", 0) == []

    # The grouped head list still ends its own tail at the `&&`.
    same_list = '!(pip install foo || pip install bar && pip install "torch==2.12.0")'
    assert [flag for _, flag in nv._split_chained(same_list)] == [False, True, False]
    assert len(
        nv.rule_inst_004_torchcodec_torch(same_list, COLAB_TORCH211, "nb.ipynb", 0)
    ) == 1


def test_notebook_validator_lands_an_upward_move_on_an_inclusive_cap():
    """`<=V` allows V, so V is what pip picks, whichever side the version moves from."""
    nv = _load_notebook_validator_module()

    # 0.7 upward into a window that spans minors: the cap names where it stops.
    spanning = '!pip install "torchcodec==0.7.0"\n!pip install "torchcodec>=0.8,<=0.10.0"'
    findings = nv.rule_inst_004_torchcodec_torch(spanning, COLAB_TORCH211, "nb.ipynb", 0)
    assert len(findings) == 1
    assert "torchcodec==0.10.0" in findings[0].message

    # An open floor has no cap to land on and stays a floor, which the ABI remedy needs.
    assert nv.rule_inst_004_torchcodec_torch(
        '!pip install "torch==2.12.0" "torchcodec>=0.12.0"', COLAB_TORCH211, "nb.ipynb", 0
    ) == []


def test_notebook_validator_will_not_keep_a_version_through_an_upgrade():
    """A bare name with `--upgrade` takes the newest release, so the installed version is not
    what the cell ends on. Without the flag pip leaves a satisfied requirement alone."""
    nv = _load_notebook_validator_module()

    for flag in ("--upgrade", "-U"):
        cell = f'!pip install {flag} "torch==2.12.0" torchcodec'
        assert nv.rule_inst_004_torchcodec_torch(cell, COLAB_TORCH211, "nb.ipynb", 0) == [], flag

    # No flag: the requirement is already satisfied, so 0.11 stays and is reported.
    assert len(nv.rule_inst_004_torchcodec_torch(
        '!pip install "torch==2.12.0" torchcodec', COLAB_TORCH211, "nb.ipynb", 0
    )) == 1

    # A bound still bounds it, upgrade or not.
    assert nv.rule_inst_004_torchcodec_torch(
        '!pip install --upgrade "torch==2.12.0" "torchcodec>=0.12.0"', COLAB_TORCH211, "nb.ipynb", 0
    ) == []
    assert len(nv.rule_inst_004_torchcodec_torch(
        '!pip install --upgrade "torch==2.11.0" "torchcodec==0.10.0"', COLAB_TORCH211, "nb.ipynb", 0
    )) == 1


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


def test_select_torchcodec_spec_tracks_torch_minor():
    ips = _load_install_python_stack()
    assert ips._select_torchcodec_spec("2.11.0+cu128") == "torchcodec>=0.11.0,<0.12.0"
    assert ips._select_torchcodec_spec("2.10.0+cu130") == "torchcodec>=0.10.0,<0.11.0"
    assert ips._select_torchcodec_spec("2.9.1+cu128") == "torchcodec>=0.8.0,<0.10.0"
    assert ips._select_torchcodec_spec("2.8.0+cu126") == "torchcodec>=0.6.0,<0.8.0"


def test_select_torchcodec_spec_never_caps_newer_torch_to_the_011_line():
    """0.11 is locked to torch 2.11 exactly, so torch >2.11 takes the open floor."""
    ips = _load_install_python_stack()
    for version in ("2.12.0", "2.12.1+cu132", "2.13.0+cu130", "2.99.0"):
        spec = ips._select_torchcodec_spec(version)
        assert spec == ips._TORCHCODEC_ABI_STABLE_SPEC, version
        assert "<" not in spec, version


def test_select_torchcodec_spec_falls_back_on_unknown_torch():
    ips = _load_install_python_stack()
    for value in (None, "", "not-a-version", "3.0.0", "2.rc1"):
        assert ips._select_torchcodec_spec(value) == ips._TORCHCODEC_DEFAULT_SPEC


def test_select_torchcodec_spec_matches_pyproject_audio_extras():
    """The installer's specs and the pip extras must not drift apart."""
    ips = _load_install_python_stack()
    text = PYPROJECT.read_text(encoding = "utf-8")
    for torch_version, extra in (
        ("2.11.0", "audio-torch211"),
        ("2.10.0", "audio-torch210"),
        ("2.9.0", "audio-torch290"),
        ("2.8.0", "audio-torch280"),
    ):
        match = re.search(rf"^{extra} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
        assert match is not None, extra
        assert ips._select_torchcodec_spec(torch_version) in match.group(1), extra


def test_select_torchcodec_spec_matches_compat_matrix():
    """Installer specs must admit exactly the minors the compat matrix allows."""
    from packaging.specifiers import SpecifierSet

    fixes = _load_import_fixes_module()
    ips = _load_install_python_stack()
    probes = [f"0.{n}.0" for n in range(0, 16)]
    for torch_minor, allowed in fixes._TORCH_TORCHCODEC_MINORS.items():
        specifier = SpecifierSet(
            ips._select_torchcodec_spec(f"{torch_minor}.0").split("torchcodec", 1)[1]
        )
        admitted = {p.rsplit(".", 1)[0] for p in probes if specifier.contains(p)}
        assert admitted == allowed, (
            f"torch {torch_minor}: installer admits {sorted(admitted)}, "
            f"matrix allows {sorted(allowed)}"
        )


def test_audio_extras_are_gated_to_platforms_with_a_torchcodec_wheel():
    """torchcodec publishes no sdist and no wheel for Linux aarch64, Windows ARM64 or
    Intel Mac, so an ungated pin makes pip fail the whole install on those hosts instead
    of just skipping audio -- and the cu*/rocm*/intel torch 2.10 extras pull it in.
    The marker must match PLATFORM_LACKS_TORCHCODEC_WHEEL in install_python_stack.py.
    """
    markers = pytest.importorskip("packaging.markers")
    tomllib = _tomllib()
    extras = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))["project"][
        "optional-dependencies"
    ]
    audio = {n: d for n, d in extras.items() if n.startswith("audio-torch")}
    assert audio, "expected audio-torch* extras"

    supported = [
        {"sys_platform": "linux", "platform_machine": "x86_64"},
        {"sys_platform": "win32", "platform_machine": "AMD64"},
        {"sys_platform": "darwin", "platform_machine": "arm64"},
    ]
    unsupported = [
        {"sys_platform": "linux", "platform_machine": "aarch64"},
        {"sys_platform": "win32", "platform_machine": "ARM64"},
        {"sys_platform": "darwin", "platform_machine": "x86_64"},
    ]
    for name, deps in audio.items():
        for dep in deps:
            _, _, marker_text = dep.partition(";")
            assert marker_text.strip(), f"{name}: {dep!r} has no marker"
            marker = markers.Marker(marker_text.strip())
            env = {"python_version": "3.12"}
            for case in supported:
                assert marker.evaluate({**env, **case}), f"{name} must install on {case}"
            for case in unsupported:
                assert not marker.evaluate(
                    {**env, **case}
                ), f"{name} has no wheel for {case} and must not be resolved there"
