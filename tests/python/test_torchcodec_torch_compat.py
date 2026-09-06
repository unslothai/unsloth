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


def test_the_2_11_row_does_not_flag_an_abi_stable_codec():
    """Adding the 2.11 row without the ABI-stable short-circuit is a false positive.

    torchcodec 0.12+ is built against torch >=2.11 and upstream supports the pairing, but a
    bare `"2.11": {"0.11"}` row makes rule_inst_004 report every 0.12..0.15 against torch
    2.11 -- and that rule is error severity. Before the 2.11 row existed the lookup simply
    missed and stayed silent, so the row and this exemption have to land together.
    """
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


def test_validator_and_runtime_guard_agree_on_the_whole_matrix(monkeypatch):
    """The two checkers must not disagree; half a rule is how they drift.

    The ABI rule has two halves -- exempt 0.12+ above the floor, and reject pre-0.12 past
    it -- and porting only the first left the validator silent on torch 2.12 with
    torchcodec 0.11, which the runtime guard reports. Comparing them pair by pair is what
    stops the next half-port.
    """
    from scripts import notebook_validator as nv

    fixes = _load_import_fixes_module()
    pairs = [
        ("2.13.0", "0.10.0"),
        ("2.12.0", "0.11.1"),
        ("2.12.0", "0.12.0"),
        ("2.11.0", "0.10.0"),
        ("2.11.0", "0.11.0"),
        ("2.11.0", "0.15.0"),
        ("2.10.0", "0.10.0"),
        ("2.10.0", "0.12.0"),
        ("2.9.0", "0.8.0"),
        ("2.4.0", "0.1.0"),
    ]
    for torch_v, codec_v in pairs:
        validator_flags = bool(
            nv.rule_inst_004_torchcodec_torch(
                "", {"torch": torch_v, "torchcodec": codec_v}, "nb.ipynb", 0
            )
        )
        guard_flags = _guard_reports(fixes, monkeypatch, torch_v, codec_v)
        assert validator_flags == guard_flags, (
            f"torch {torch_v} + torchcodec {codec_v}: "
            f"validator={'reports' if validator_flags else 'silent'}, "
            f"guard={'reports' if guard_flags else 'silent'}"
        )


def _guard_reports(fixes, monkeypatch, torch_version: str, codec_version: str) -> bool:
    """Run the runtime guard against one pair, stubbed the way the cases above stub it."""
    import importlib.metadata

    _stub_torch(monkeypatch, torch_version)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: codec_version)
    return fixes._torchcodec_version_mismatch_hint() is not None


def test_the_installer_never_installs_what_the_guard_rejects(monkeypatch):
    """End-to-end invariant across all three checkers, past the table as well as inside it.

    test_select_torchcodec_spec_matches_compat_matrix ties the installer to the matrix, but
    it iterates the rows that EXIST, so a torch minor past the last row is not covered -- and
    that is exactly where the ABI half-port hid. This asks the question that actually matters
    instead: for every torch minor the installer will see, is every codec its own spec admits
    accepted by the runtime guard?
    """
    from packaging.specifiers import SpecifierSet

    fixes = _load_import_fixes_module()
    ips = _load_install_python_stack()
    probes = [f"0.{n}.0" for n in range(0, 20)]

    for minor in range(5, 15):  # torch 2.5 .. 2.14, i.e. past the last lockstep row
        torch_v = f"2.{minor}.0"
        specifier = SpecifierSet(ips._select_torchcodec_spec(torch_v).split("torchcodec", 1)[1])
        admitted = [p for p in probes if specifier.contains(p)]
        assert admitted, f"torch {torch_v}: installer spec admits nothing"
        for codec_v in admitted:
            assert not _guard_reports(fixes, monkeypatch, torch_v, codec_v), (
                f"installer would put torchcodec {codec_v} on torch {torch_v}, "
                f"which the runtime guard then reports as incompatible"
            )
