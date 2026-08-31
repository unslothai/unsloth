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


def test_security_audit_covers_both_installable_torchcodec_lines():
    """extras-no-deps.txt used to put torchcodec 0.10 in the audit set. It no longer pins
    it, and torch 2.10 is still installable (`_TORCH_FLAVOR_REPAIR_PKG_SPEC` and the default
    ROCm repair both cap below 2.11), so dropping audio-torch210 loses live coverage. The
    two lines cannot share one resolve, hence the separate input."""
    text = SECURITY_AUDIT_YML.read_text(encoding = "utf-8")
    # Both halves of the workflow build the inputs; one is the advisory audit, one is
    # scan_packages.
    assert text.count('optional-dependencies"]["audio-torch211"]') == 2
    assert text.count('optional-dependencies"]["audio-torch210"]') == 2
    assert text.count("audit-reqs/audio-torch210.txt") >= 2  # generated + osv-scanner
    assert "for f in unsloth-deps audio-torch210 " in text  # pip-audit
    assert "files: 'studio overrides extras-no-deps audio-torch210'" in text  # scan_packages


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
