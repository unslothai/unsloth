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
COLAB_TORCH211 = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
TORCHCODEC_WHEEL = (
    "https://download.pytorch.org/whl/torchcodec-0.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
)


def _tomllib():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    return pytest.importorskip("tomli")


@pytest.fixture(autouse = True)
def _no_inherited_index_config(monkeypatch):
    """Nearly every test here reads the index configuration, so a developer's own
    UNSLOTH_TORCH_INDEX_URL must not be what decides which URL the suite asserts."""
    for name in (
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
        "UNSLOTH_PYTORCH_MIRROR",
    ):
        monkeypatch.delenv(name, raising = False)


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


# One row per (torch, torchcodec) pair the runtime guard has an opinion about: the
# substrings the warning must carry, and the ones it must not. `None` for `contains` means
# the guard has to stay silent -- the pair is supported and a warning would be noise.
_GUARD_CASES = [
    # Inside the lockstep table: each torch minor takes its own codec line and no other.
    ("2.9.0+cu128", "0.7.0", (), ("audio-torch210",)),
    ("2.8.0+cu128", "0.7.0", None, ()),
    # Untagged torch needs no index pin, so the convenient extra stays on offer...
    ("2.10.0", "0.11.0", ("torchcodec 0.11.0", "audio-torch210", "<0.11.0"), ("<11.0",)),
    # ...while a tagged one cannot carry an index in an extra, so it gets the pin alone.
    (
        "2.10.0+cu128",
        "0.11.0",
        ("--index-url https://download.pytorch.org/whl/cu128", "<0.11.0"),
        ("audio-torch210",),
    ),
    ("2.10.0+cu128", "0.10.0+cu128", None, ()),
    # The guard must not be silent on the torch minor where the mismatch happens.
    (
        "2.11.0",
        "0.10.0+cu128",
        ("torchcodec 0.10.0+cu128", "audio-torch211", ">=0.11", "<0.12.0"),
        ("audio-torch210",),
    ),
    ("2.11.0+cu128", "0.11.1+cu128", None, ()),
    # The ABI-stable floor starts at torch 2.11: 2.10 keeps the exact pairing.
    ("2.10.0", "0.15.0", ("audio-torch210",), ()),
    # A torch minor older than the matrix keeps the original no-opinion behaviour.
    ("2.4.0", "0.0.3", None, ()),
]

# torchcodec 0.12+ targets torch >=2.11, so it is not locked to one minor.
_GUARD_CASES += [
    (torch_version, codec_version, None, ())
    for torch_version in ("2.11.0+cu128", "2.12.0", "2.13.0+cu130")
    for codec_version in ("0.12.0", "0.15.0+cu130")
]

# 0.11 is pinned to torch 2.11 exactly, so 2.12/2.13 with a pre-0.12 codec still warns, and
# no audio-torch2xx extra exists for those minors, so none is offered.
_GUARD_CASES += [
    (torch_version, codec_version, ("torchcodec>=0.12.0",), ("unsloth[audio-torch",))
    for torch_version in ("2.12.1+cu130", "2.13.0")
    for codec_version in ("0.11.1", "0.10.0")
]


@pytest.mark.parametrize(
    "torch_version, codec_version, contains, absent",
    _GUARD_CASES,
    ids = [f"torch{t}-codec{c}" for t, c, _, _ in _GUARD_CASES],
)
def test_the_runtime_guard_reports_exactly_the_pairs_the_matrix_forbids(
    monkeypatch, torch_version, codec_version, contains, absent
):
    import importlib.metadata

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, torch_version)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: codec_version)

    hint = fixes._torchcodec_version_mismatch_hint()
    if contains is None:
        assert hint is None, f"{torch_version} + torchcodec {codec_version} is supported upstream"
        return
    assert hint is not None, f"{torch_version} + torchcodec {codec_version} must not go unreported"
    for text in contains:
        assert text in hint, f"{text!r} missing from {hint!r}"
    for text in absent:
        assert text not in hint, f"{text!r} should not appear in {hint!r}"


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


# The published torchcodec compatibility table, transcribed from upstream. Sources agree:
#   https://github.com/meta-pytorch/torchcodec  (README, "older versions" section)
#   https://pypi.org/project/torchcodec/        (same table in the project description)
# Kept as a literal on purpose. Every other check here compares our three tables against each
# OTHER, which passes just as happily when all three are wrong in the same way -- that is how
# `2.6: {0.2, 0.3}` and `2.5: {0.1, 0.2}` survived: upstream pairs 0.3 with torch 2.7 and 0.2
# with torch 2.6, so the installer's window picked a release built against the NEXT torch.
# torch 2.4 -> 0.0.3 is deliberately omitted below: the installer floors at 2.5 and returns
# _TORCHCODEC_DEFAULT_SPEC underneath it.
_UPSTREAM_TORCH_TO_TORCHCODEC_MINORS = {
    "2.11": {"0.11"},
    "2.10": {"0.10"},
    "2.9": {"0.8", "0.9"},
    "2.8": {"0.6", "0.7"},
    "2.7": {"0.3", "0.4", "0.5"},
    "2.6": {"0.2"},
    "2.5": {"0.1"},
}


# What each download.pytorch.org index actually publishes, read off the live listings:
# the torch 2.x minors it serves, and the inclusive range of torchcodec minors.
# Note cu130 starts at codec 0.8, and no index carries 0.1 or 0.2 -- those are PyPI-only.
_INDEX_INVENTORY = {
    "cpu": {"torch": range(5, 15), "codec": (3, 16)},
    "cu118": {"torch": range(5, 8), "codec": (3, 5)},
    "cu126": {"torch": range(6, 15), "codec": (3, 16)},
    "cu128": {"torch": range(7, 12), "codec": (3, 11)},
    "cu130": {"torch": range(9, 15), "codec": (8, 16)},
}


def test_torchcodec_index_follows_the_resident_torch_build():
    """torchcodec ships one wheel per accelerator, so the right version from the wrong index
    is a codec that cannot dlopen. Upstream's install docs say to pass --index-url and match
    it to the torch build; docker/Dockerfile pins cu128 by hand for exactly this reason."""
    ips = _load_install_python_stack()
    base = "https://download.pytorch.org/whl/"
    assert ips._torchcodec_index_url("2.11.0+cu128") == base + "cu128"
    assert ips._torchcodec_index_url("2.11.0+cu126") == base + "cu126"
    assert ips._torchcodec_index_url("2.14.0+cu130") == base + "cu130"
    assert ips._torchcodec_index_url("2.11.0+cpu") == base + "cpu"

    # Untagged is PyPI's own torch, whose counterpart is PyPI's default torchcodec. Pinning
    # cpu here would be wrong: on Linux an untagged torch is a CUDA build.
    assert ips._torchcodec_index_url("2.11.0") is None
    # No torchcodec is published under these, so unpinned beats an index that cannot serve.
    assert ips._torchcodec_index_url("2.9.0+rocm6.4") is None
    assert ips._torchcodec_index_url("2.10.0+xpu") is None
    assert ips._torchcodec_index_url(None) is None
    assert ips._torchcodec_index_url("") is None


def test_pinning_the_index_never_starves_a_reachable_torch():
    """A pin that removed audio from a supported host would trade one bug for another.

    Every torch build that pins must find its selected codec on that same index. This holds
    because torch and torchcodec are cut together: cu128 stops at torch 2.11 and its
    torchcodec stops at 0.11, the exact pair the matrix maps 2.11 to; cu130 starts at torch
    2.9 and its torchcodec starts at 0.8, the pair for 2.9.

    The one gap is deliberate and handled in the helper rather than here: no index carries
    torchcodec 0.1 or 0.2, so torch 2.5 and 2.6 must not pin at all.
    """
    from packaging.specifiers import SpecifierSet

    ips = _load_install_python_stack()
    for tag, inv in _INDEX_INVENTORY.items():
        low, high = inv["codec"]
        for minor in inv["torch"]:
            version = f"2.{minor}.0+{tag}"
            spec = ips._select_torchcodec_spec(version)
            specifier = SpecifierSet(spec.split("torchcodec", 1)[1])
            served = [f"0.{m}.0" for m in range(low, high + 1) if specifier.contains(f"0.{m}.0")]
            index = ips._torchcodec_index_url(version, spec)
            if index is None:
                # Only the PyPI-only rows may decline to pin.
                assert minor in (5, 6), f"torch 2.{minor}+{tag} unexpectedly refused to pin"
                continue
            assert index.endswith("/" + tag)
            assert served, (
                f"torch 2.{minor} pins the {tag} index and selects {spec}, but that index "
                f"publishes only torchcodec 0.{low}-0.{high}"
            )


def test_the_two_pypi_only_rows_stay_unpinned():
    """torchcodec 0.1 and 0.2 were never published to download.pytorch.org, so pinning torch
    2.5 / 2.6 would guarantee a skip on the oldest venvs instead of leaving them as they are."""
    ips = _load_install_python_stack()
    for minor in (5, 6):
        version = f"2.{minor}.0+cu126"
        assert ips._torchcodec_index_url(version, ips._select_torchcodec_spec(version)) is None
    # 2.7 selects >=0.3.0,<0.6.0, which the indexes do carry, so it pins.
    assert (
        ips._torchcodec_index_url("2.7.0+cu118", ips._select_torchcodec_spec("2.7.0")) is not None
    )


# extra -> the first interpreter that must NOT select it, one that must, and whether its
# codec line ships a Linux aarch64 wheel.
#
# torchcodec publishes no sdist, so an extra left open above its last cp tag, or on a host
# with no wheel, makes pip fail the whole install instead of skipping audio -- and the
# cu*/rocm*/intel torch 2.10 extras pull it in. requires-python is open-ended (>=3.9), so a
# newer interpreter reaches these extras too; the marker has to stop it, and it has to match
# install_python_stack.py.
#
# The Python ceilings come from the same upstream table _TORCHCODEC_PYTHON_WINDOWS encodes:
# the 0.6/0.7 line stops at 3.13, everything from 0.9 up runs to 3.14. aarch64 is per-extra
# rather than blanket: torchcodec had no aarch64 wheel until 0.11.0, and every release since
# has kept it, so audio-torch211 must ALLOW aarch64 while the older extras, which top out at
# 0.10, must still exclude it.
_AUDIO_EXTRA_GATES = {
    "audio-torch211": ("3.15", "3.14", True),
    "audio-torch210": ("3.15", "3.14", False),
    "audio-torch290": ("3.15", "3.14", False),
    "audio-torch280": ("3.14", "3.13", False),
}

# Windows ARM64 and Intel Mac have no wheel at any torchcodec version.
_WHEELED_HOSTS = (
    {"sys_platform": "linux", "platform_machine": "x86_64"},
    {"sys_platform": "win32", "platform_machine": "AMD64"},
    {"sys_platform": "darwin", "platform_machine": "arm64"},
)
_WHEELLESS_HOSTS = (
    {"sys_platform": "win32", "platform_machine": "ARM64"},
    {"sys_platform": "darwin", "platform_machine": "x86_64"},
)


def _audio_extras():
    extras = _tomllib().loads(PYPROJECT.read_text(encoding = "utf-8"))["project"][
        "optional-dependencies"
    ]
    return {n: d for n, d in extras.items() if n.startswith("audio-torch")}


def test_every_audio_extra_has_a_gate_of_its_own():
    """A new extra added without a row here would go through the gates below untested."""
    assert set(_audio_extras()) == set(_AUDIO_EXTRA_GATES)


@pytest.mark.parametrize("extra", sorted(_AUDIO_EXTRA_GATES))
def test_an_audio_extra_is_gated_to_the_hosts_and_pythons_its_wheels_cover(extra):
    markers = pytest.importorskip("packaging.markers")
    too_new, supported, allows_aarch64 = _AUDIO_EXTRA_GATES[extra]
    deps = _audio_extras()[extra]
    assert deps, extra

    for dep in deps:
        _, _, marker_text = dep.partition(";")
        assert marker_text.strip(), f"{extra}: {dep!r} has no marker"
        marker = markers.Marker(marker_text.strip())

        base = {"python_version": "3.12", "platform_system": "Linux", "os_name": "posix"}
        for case in _WHEELED_HOSTS:
            assert marker.evaluate({**base, **case}), f"{extra} must install on {case}"
        for case in _WHEELLESS_HOSTS:
            assert not marker.evaluate(
                {**base, **case}
            ), f"{extra} has no wheel for {case} and must not be resolved there"
        aarch64 = marker.evaluate({**base, "sys_platform": "linux", "platform_machine": "aarch64"})
        assert aarch64 == allows_aarch64, (
            f"{extra} {'must not exclude' if allows_aarch64 else 'must not be resolved on'} "
            "Linux aarch64"
        )

        linux = {**_WHEELED_HOSTS[0], "platform_system": "Linux", "os_name": "posix"}
        assert not marker.evaluate(
            {**linux, "python_version": too_new}
        ), f"{extra} still selects torchcodec on Python {too_new}, which has no wheel"
        assert marker.evaluate(
            {**linux, "python_version": supported}
        ), f"{extra} stopped selecting torchcodec on Python {supported}, which does"


def test_compat_matrix_matches_the_published_upstream_table():
    """Pin the runtime guard to upstream, not merely to our own other copies of it."""
    fixes = _load_import_fixes_module()
    assert fixes._TORCH_TORCHCODEC_MINORS == _UPSTREAM_TORCH_TO_TORCHCODEC_MINORS


def test_installer_never_selects_a_torchcodec_built_against_another_torch():
    """The window handed to pip must not contain a release upstream pairs with a different
    torch: pip takes the HIGHEST match, so a window one minor too wide installs the mismatch
    this whole module exists to prevent.

    This also ties the installer to the runtime guard's own matrix, since
    test_compat_matrix_matches_the_published_upstream_table pins that matrix to the literal
    below."""
    from packaging.specifiers import SpecifierSet

    ips = _load_install_python_stack()
    probes = [f"0.{n}.0" for n in range(0, 16)]
    for torch_minor, allowed in _UPSTREAM_TORCH_TO_TORCHCODEC_MINORS.items():
        spec = ips._select_torchcodec_spec(f"{torch_minor}.0")
        specifier = SpecifierSet(spec.split("torchcodec", 1)[1])
        admitted = {p.rsplit(".", 1)[0] for p in probes if specifier.contains(p)}
        assert admitted == allowed, (
            f"torch {torch_minor}: {spec} admits {sorted(admitted)}, "
            f"upstream builds only {sorted(allowed)} against it"
        )
        highest = max(
            (p for p in probes if specifier.contains(p)),
            key = lambda v: tuple(int(x) for x in v.split(".")),
        )
        assert (
            highest.rsplit(".", 1)[0] in allowed
        ), f"torch {torch_minor}: pip would resolve {spec} to {highest}"


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


# ----------------------------------------------------------------------------------
# Wheel availability. The step that installs the selected spec is fatal on failure, so
# "does this spec have a wheel here" decides whether this branch can break an install,
# not merely whether audio works. Verified against the live PyPI index; kept as a table
# so the suite stays deterministic and offline.
# ----------------------------------------------------------------------------------

# torchcodec version -> platforms it publishes. Read off pypi.org/pypi/torchcodec/json.
# The three transitions that matter:
#   win_amd64            absent before 0.7.0
#   manylinux aarch64    absent before 0.11.0
#   macosx arm64         minimum macOS 11.0 through 0.11.1, then 14.0 from 0.12.0
_TORCHCODEC_WHEEL_HISTORY = {
    "0.1.0": {"linux_x86_64", "macos_arm64_11"},
    "0.2.0": {"linux_x86_64", "macos_arm64_11"},
    "0.3.0": {"linux_x86_64", "macos_arm64_11"},
    "0.4.0": {"linux_x86_64", "macos_arm64_11"},
    "0.5": {"linux_x86_64", "macos_arm64_11"},
    "0.6.0": {"linux_x86_64", "macos_arm64_11"},
    "0.7.0": {"linux_x86_64", "macos_arm64_11", "win_amd64"},
    "0.8.0": {"linux_x86_64", "macos_arm64_11", "win_amd64"},
    "0.9.0": {"linux_x86_64", "macos_arm64_11", "win_amd64"},
    "0.10.0": {"linux_x86_64", "macos_arm64_11", "win_amd64"},
    "0.11.0": {"linux_x86_64", "linux_aarch64", "macos_arm64_11", "win_amd64"},
    "0.11.1": {"linux_x86_64", "linux_aarch64", "macos_arm64_11", "win_amd64"},
    "0.12.0": {"linux_x86_64", "linux_aarch64", "macos_arm64_14", "win_amd64"},
    "0.15.0": {"linux_x86_64", "linux_aarch64", "macos_arm64_14", "win_amd64"},
}

# label -> (IS_LINUX, IS_WINDOWS, IS_MACOS, IS_MAC_ARM, IS_MAC_INTEL), machine, macos major
_SIM_HOSTS = {
    "linux-x86_64": ((True, False, False, False, False), "x86_64", None),
    "linux-aarch64": ((True, False, False, False, False), "aarch64", None),
    "linux-ppc64le": ((True, False, False, False, False), "ppc64le", None),
    "windows-amd64": ((False, True, False, False, False), "AMD64", None),
    "windows-arm64": ((False, True, False, False, False), "ARM64", None),
    "macos-arm64-14": ((False, False, True, True, False), "arm64", 14),
    "macos-arm64-13": ((False, False, True, True, False), "arm64", 13),
    "macos-intel": ((False, False, True, False, True), "x86_64", None),
}


def _host_key(label):
    if label.startswith("linux"):
        machine = _SIM_HOSTS[label][1]
        return f"linux_{machine}" if machine in ("x86_64", "aarch64") else None
    if label == "windows-amd64":
        return "win_amd64"
    if label.startswith("macos-arm64"):
        return "macos_arm64"
    return None


def _release_python_window(vt):
    """Upstream's published Python range for a release, independent of architecture."""
    if vt < (0, 2, 0):
        return (3, 9), (3, 12)
    if vt < (0, 8, 0):
        return (3, 9), (3, 13)
    if vt < (0, 9, 0):
        return (3, 10), (3, 13)
    return (3, 10), (3, 14)


def _wheel_exists_in_window(label, floor, ceiling, python):
    """Does any release the window admits publish a wheel for this host AND interpreter?"""
    key = _host_key(label)
    if key is None:
        return False
    macos_major = _SIM_HOSTS[label][2]
    for ver, plats in _TORCHCODEC_WHEEL_HISTORY.items():
        vt = tuple(int(p) for p in ver.split("."))
        vt = vt + (0,) * (3 - len(vt))
        if vt < floor or (ceiling is not None and vt >= ceiling):
            continue
        py_min, py_max = _release_python_window(vt)
        if not py_min <= python <= py_max:
            continue
        if key == "macos_arm64":
            for p in plats:
                if p.startswith("macos_arm64_") and macos_major >= int(p.rsplit("_", 1)[1]):
                    return True
        elif key in plats:
            return True
    return False


def _patch_host(ips, monkeypatch, label):
    flags, machine, macos_major = _SIM_HOSTS[label]
    is_linux, is_windows, is_macos, is_mac_arm, is_mac_intel = flags
    monkeypatch.setattr(ips, "IS_LINUX", is_linux)
    monkeypatch.setattr(ips, "IS_WINDOWS", is_windows)
    monkeypatch.setattr(ips, "IS_MACOS", is_macos)
    monkeypatch.setattr(ips, "IS_MAC_ARM", is_mac_arm)
    monkeypatch.setattr(ips, "IS_MAC_INTEL", is_mac_intel)
    monkeypatch.setattr(ips.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        ips.platform,
        "mac_ver",
        lambda: (f"{macos_major}.0" if macos_major else "", ("", "", ""), ""),
    )


def test_the_installer_never_selects_a_spec_with_no_wheel_here(monkeypatch):
    """The gate must not green-light a window this platform never published into.

    pip_install_try keeps a miss from ending the install, but attempting one is still a
    wasted round trip and, before that call was changed, was fatal. Two cells were real:
    Windows on the cu118 index sits at torch 2.7 and selects `>=0.3.0,<0.6.0`, where no
    release ships win_amd64; and a Mac below 14 selects `>=0.12.0`, which is macosx_14_0
    only.
    """
    ips = _load_install_python_stack()
    for label in _SIM_HOSTS:
        for python in ((3, 9), (3, 10), (3, 12), (3, 13), (3, 14)):
            for minor in range(4, 15):
                _patch_host(ips, monkeypatch, label)
                monkeypatch.setattr(ips.sys, "version_info", python + (0, "final", 0))
                spec = ips._select_torchcodec_spec(f"2.{minor}.0")
                floor, ceiling = ips._torchcodec_spec_bounds(spec)
                gate_says_yes = ips._torchcodec_spec_is_installable(spec)
                really_has = _wheel_exists_in_window(label, floor, ceiling, python)
                assert gate_says_yes == really_has, (
                    f"{label} py{python[0]}.{python[1]} torch 2.{minor}: gate says "
                    f"{'install' if gate_says_yes else 'skip'} for {spec}, but a wheel "
                    f"{'exists' if really_has else 'does not exist'}"
                )


# The cells of the sweep above that were real bugs or are the documented transitions, with
# the answer written by hand rather than read off _TORCHCODEC_WHEEL_HISTORY. The sweep
# checks the gate against that oracle and so passes just as happily when both are wrong;
# these say what the answer has to be.
#   (host, python, torch minor, must the gate install?)
_WHEEL_GATE_ANCHORS = [
    # aarch64 got its first wheel at 0.11.0, the line torch 2.11 selects; 2.10 takes 0.10.
    ("linux-aarch64", (3, 12), 11, True),
    ("linux-aarch64", (3, 12), 10, False),
    # torchcodec 0.12+ is macosx_14_0 only, so an older Mac must not be sent to it.
    ("macos-arm64-13", (3, 12), 12, False),
    ("macos-arm64-13", (3, 12), 11, True),
    ("macos-arm64-14", (3, 12), 12, True),
    # Architecture is not the only wheel axis. torch 2.5 selects the 0.1 line, which stops at
    # Python 3.12, so 3.13 has nothing to install even on plain linux-x86_64. This was masked
    # while the 2.5 window ran to <0.3.0: it reached 0.2, which does ship cp313, so the gate
    # said yes for a release built against torch 2.6.
    ("linux-x86_64", (3, 12), 5, True),
    ("linux-x86_64", (3, 13), 5, False),
    # The floor moves too: 0.8+ dropped 3.9, so torch 2.9 has nothing for a 3.9 interpreter.
    ("linux-x86_64", (3, 9), 9, False),
    ("linux-x86_64", (3, 9), 8, True),
]
# win_amd64 starts at 0.7.0 and torch 2.5-2.7 select lines below it. Reachable rather than
# theoretical: the cu118 index tops out at torch 2.7.
_WHEEL_GATE_ANCHORS += [
    ("windows-amd64", (3, 12), minor, minor >= 8) for minor in (5, 6, 7, 8, 10, 11, 12)
]


@pytest.mark.parametrize(
    "label, python, torch_minor, installable",
    _WHEEL_GATE_ANCHORS,
    ids = [f"{h}-py{p[0]}.{p[1]}-torch2.{m}" for h, p, m, _ in _WHEEL_GATE_ANCHORS],
)
def test_the_wheel_gate_answers_the_transitions_it_was_written_for(
    monkeypatch, label, python, torch_minor, installable
):
    ips = _load_install_python_stack()
    _patch_host(ips, monkeypatch, label)
    monkeypatch.setattr(ips.sys, "version_info", python + (0, "final", 0))
    spec = ips._select_torchcodec_spec(f"2.{torch_minor}.0")
    assert ips._torchcodec_spec_is_installable(spec) == installable, spec


def test_python_windows_match_the_published_upstream_table():
    """Same reason as the compat-matrix pin: transcribed from upstream, not from ourselves."""
    ips = _load_install_python_stack()
    assert ips._TORCHCODEC_PYTHON_WINDOWS == (
        ((0, 1, 0), (3, 9), (3, 12)),
        ((0, 2, 0), (3, 9), (3, 13)),
        ((0, 8, 0), (3, 10), (3, 13)),
        ((0, 9, 0), (3, 10), (3, 14)),
    )


def test_the_torchcodec_step_cannot_end_the_install():
    """Audio is optional; pip_install exits on failure and pip_install_try does not.

    Asserted on the source because the alternative is driving a whole install. The rule
    it encodes is the one the extras-no-deps filter above it already states: the audio
    extras step must not take down the install.
    """
    source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    step = source.split("# 13b. torchcodec", 1)[1].split("# 14.", 1)[0]
    assert "pip_install_try(" in step, "the torchcodec step must use the non-fatal install"
    assert "\n        pip_install(" not in step, "pip_install() exits on failure"


def test_the_runtime_hint_pins_the_index_it_tells_you_to_install_from(monkeypatch):
    """The installer pins the torch index for torchcodec, so a remedy that omits it hands
    back the wrong accelerator build and audio stays broken."""
    import importlib.metadata

    fixes = _load_import_fixes_module()

    _stub_torch(monkeypatch, "2.11.0+cu128")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.10.0")
    hint = fixes._torchcodec_version_mismatch_hint()
    assert "--index-url https://download.pytorch.org/whl/cu128 'torchcodec>=0.11" in hint

    # cpu is an index too, and the ABI-stable branch takes the same treatment.
    _stub_torch(monkeypatch, "2.12.0+cu130")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.1")
    assert (
        "--index-url https://download.pytorch.org/whl/cu130 'torchcodec>=0.12.0'"
        in fixes._torchcodec_version_mismatch_hint()
    )

    # Untagged torch is PyPI's own build, and rocm publishes no torchcodec: no pin either way.
    for version in ("2.11.0", "2.9.0+rocm6.4"):
        _stub_torch(monkeypatch, version)
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7.0")
        assert "--index-url" not in (fixes._torchcodec_version_mismatch_hint() or "")

    # torchcodec 0.1 is PyPI-only, so the 2.5 row must not send anyone to a torch index.
    _stub_torch(monkeypatch, "2.5.0+cu118")
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.5.0")
    assert "--index-url" not in fixes._torchcodec_version_mismatch_hint()


def test_the_remedy_drops_the_extra_when_an_index_pin_is_needed(monkeypatch):
    """An extra cannot carry an index: the marker picks the version, and putting
    --index-url on the whole command would resolve unsloth itself from the torch index. On a
    tagged venv the extra would hand back the same unloadable wheel the warning is about."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.10.0")

    _stub_torch(monkeypatch, "2.11.0+cu128")
    pinned = fixes._torchcodec_version_mismatch_hint()
    assert "--index-url" in pinned
    assert "unsloth[audio-torch211]" not in pinned

    # Untagged torch needs no pin, so the convenient alternative stays on offer.
    _stub_torch(monkeypatch, "2.11.0")
    unpinned = fixes._torchcodec_version_mismatch_hint()
    assert "--index-url" not in unpinned
    assert "unsloth[audio-torch211]" in unpinned


def test_the_codec_index_honours_an_explicitly_pinned_torch_mirror(monkeypatch):
    """UNSLOTH_TORCH_INDEX_URL names the index torch itself came from. Rebuilding a public
    download.pytorch.org URL from the local tag sent an authenticated or air-gapped mirror to
    the internet, and the `--index-url` that follows also drops the inherited index
    configuration, so the codec install fails outright where public PyTorch is unreachable."""
    from studio import install_python_stack as ips

    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://download.pytorch.org/whl/cu128"

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/pytorch/cu128/")
    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://mirror.corp.example/pytorch/cu128"

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu126")
    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://download.pytorch.org/whl/cu126"

    # The override does not make an untagged or rocm torch start pinning.
    assert ips._torchcodec_index_url("2.11.0") is None
    assert ips._torchcodec_index_url("2.11.0+rocm7.0") is None


def test_the_runtime_remedy_honours_a_configured_torch_index(monkeypatch):
    """A warning that tells an air-gapped or authenticated host to install from
    download.pytorch.org either fails or bypasses the artifact source the install was
    configured with. The mirror is where the matching build is."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.0")
    _stub_torch(monkeypatch, "2.10.0+cu128")

    assert "--index-url https://download.pytorch.org/whl/cu128" in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )

    # The variable, not its value: a mirror URL can carry credentials and this string is
    # warned into terminals and CI logs. The shell expands it, so the command still runs.
    monkeypatch.setenv(
        "UNSLOTH_TORCH_INDEX_URL", "https://user:secret@mirror.corp.example/pytorch/cu128/"
    )
    hint = fixes._torchcodec_version_mismatch_hint()
    assert '--index-url "$UNSLOTH_TORCH_INDEX_URL" ' in hint
    assert "secret" not in hint
    assert "mirror.corp.example" not in hint
    assert "download.pytorch.org" not in hint

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu126")
    assert "--index-url https://download.pytorch.org/whl/cu126" in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )


def test_a_mismatched_accelerator_build_is_named_when_the_codec_cannot_load(monkeypatch):
    """A cu128 venv holding PyPI's default torchcodec has the right VERSION and still cannot
    dlopen, so the version hint says nothing and audio used to be disabled in silence. The
    provenance hint only speaks once the load has actually failed, so a working pairing this
    cannot explain never warns."""
    import sys
    import types

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.11.0+cu128")

    codec = types.ModuleType("torchcodec")
    codec.__version__ = "0.11.0"  # untagged: PyPI's default build
    monkeypatch.setitem(sys.modules, "torchcodec", codec)
    hint = fixes._torchcodec_provenance_hint()
    assert hint is not None
    assert "https://download.pytorch.org/whl/cu128" in hint
    assert "audio is disabled" in hint

    # Matching provenance says nothing, and neither does a rocm torch.
    codec.__version__ = "0.11.0+cu128"
    assert fixes._torchcodec_provenance_hint() is None
    codec.__version__ = "0.11.0"
    _stub_torch(monkeypatch, "2.11.0+rocm7.0")
    assert fixes._torchcodec_provenance_hint() is None


def test_the_printed_codec_index_is_redacted(monkeypatch):
    """The install status line goes straight to the terminal and the CI log, not through
    _redact_install_output, which only covers captured pip output. An authenticated mirror
    carries its credentials in the userinfo or a query token, so printing the configured
    index verbatim persists them."""
    from studio import install_python_stack as ips

    monkeypatch.setenv(
        "UNSLOTH_TORCH_INDEX_URL", "https://user:secret@mirror.corp.example/pytorch/cu128/"
    )

    # The installer still receives the exact URL, credentials and all.
    resolved = ips._torchcodec_index_url("2.11.0+cu128")
    assert resolved == "https://user:secret@mirror.corp.example/pytorch/cu128"

    # What gets printed does not.
    shown = ips._strip_index_url_credentials(resolved)
    assert shown == "https://mirror.corp.example/pytorch/cu128"
    assert "secret" not in shown

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/simple?token=abc")
    assert "abc" not in ips._strip_index_url_credentials(ips._torchcodec_index_url("2.11.0+cu128"))

    # The status line itself uses the redacting call, not the raw variable.
    source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    assert 'f" from {_strip_index_url_credentials(_codec_index)}"' in source
    assert 'f" from {_codec_index}"' not in source


def test_the_codec_index_follows_a_configured_pytorch_mirror(monkeypatch):
    """UNSLOTH_PYTORCH_MIRROR replaces the base every other index in install_python_stack is
    built from, so a codec pinned to the public site cannot be fetched on an air-gapped host
    and bypasses the artifact source on a corporate one."""
    import importlib

    monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", "https://mirror.corp.example/whl")
    from studio import install_python_stack as ips

    ips = importlib.reload(ips)  # _PYTORCH_WHL_BASE is read at import time
    try:
        assert ips._torchcodec_index_url("2.11.0+cu128") == "https://mirror.corp.example/whl/cu128"
        # A full URL override still wins over the mirror base.
        monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://other.example/pytorch/cu128")
        assert ips._torchcodec_index_url("2.11.0+cu128") == "https://other.example/pytorch/cu128"
    finally:
        monkeypatch.delenv("UNSLOTH_PYTORCH_MIRROR", raising = False)
        importlib.reload(ips)


def test_the_runtime_remedy_follows_a_configured_pytorch_mirror(monkeypatch):
    """Same for the runtime warning, and by naming the variable rather than its value the
    command still works without disclosing a mirror that may carry credentials."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.0")
    _stub_torch(monkeypatch, "2.10.0+cu128")

    monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", "https://user:secret@mirror.corp.example/whl")
    hint = fixes._torchcodec_version_mismatch_hint()
    assert '--index-url "$UNSLOTH_PYTORCH_MIRROR"/cu128' in hint
    assert "secret" not in hint
    assert "download.pytorch.org" not in hint

    # The family names the leaf under the same mirror.
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu126")
    assert '--index-url "$UNSLOTH_PYTORCH_MIRROR"/cu126' in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )

    # With no mirror configured the public URL comes back, family-aware as before.
    monkeypatch.delenv("UNSLOTH_PYTORCH_MIRROR")
    assert "--index-url https://download.pytorch.org/whl/cu126" in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )


def test_the_provenance_hint_does_not_assert_a_cause_it_has_not_established(monkeypatch):
    """Differing local tags show the two wheels came from different indexes, nothing more.
    torchcodec is published per accelerator on every line, 0.12+ included, so the mismatch
    stays possible there, but the load can equally have failed on a missing libavutil that no
    reinstall repairs. The hint has to name both."""
    import sys
    import types

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.12.0+cu128")

    codec = types.ModuleType("torchcodec")
    codec.__version__ = "0.12.0"
    monkeypatch.setitem(sys.modules, "torchcodec", codec)
    hint = fixes._torchcodec_provenance_hint()
    assert hint is not None  # 0.12+ is ABI-stable against torch, not accelerator-agnostic
    assert "may be built for a different accelerator" in hint
    assert "cannot load" not in hint
    assert "FFmpeg" in hint


def test_the_remedy_uses_the_shell_of_the_host_it_prints_on(monkeypatch):
    """PowerShell is Studio's supported Windows shell and does not expand `$NAME`, so the
    POSIX spelling pasted there produced an empty `--index-url`."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.0")
    _stub_torch(monkeypatch, "2.10.0+cu128")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://user:secret@mirror.corp.example/cu128")

    monkeypatch.setattr(fixes.sys, "platform", "linux")
    assert '--index-url "$UNSLOTH_TORCH_INDEX_URL"' in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )

    monkeypatch.setattr(fixes.sys, "platform", "win32")
    windows = fixes._torchcodec_version_mismatch_hint() or ""
    assert "--index-url $env:UNSLOTH_TORCH_INDEX_URL" in windows
    assert "secret" not in windows

    # The mirror branch follows the same rule.
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")
    monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", "https://mirror.corp.example/whl")
    assert "--index-url $env:UNSLOTH_PYTORCH_MIRROR/cu128" in (
        fixes._torchcodec_version_mismatch_hint() or ""
    )


def test_the_provenance_remedy_pins_the_compatible_window(monkeypatch):
    """The accelerator indexes carry the whole codec line, so a bare `torchcodec` on a torch
    2.9 host installs the newest release there and trades a wrong-accelerator build for a
    wrong-version one, leaving audio just as disabled."""
    import sys
    import types

    fixes = _load_import_fixes_module()

    codec = types.ModuleType("torchcodec")
    codec.__version__ = "0.8.0"
    monkeypatch.setitem(sys.modules, "torchcodec", codec)

    _stub_torch(monkeypatch, "2.9.0+cu128")
    hint = fixes._torchcodec_provenance_hint()
    assert "'torchcodec>=0.9,<0.10.0'" in hint, hint
    assert "--index-url https://download.pytorch.org/whl/cu128 torchcodec`" not in hint

    # Past the table, the ABI-stable floor is the pin instead of a bare name.
    _stub_torch(monkeypatch, "2.12.0+cu128")
    codec.__version__ = "0.11.0"
    assert "'torchcodec>=0.12.0'" in (fixes._torchcodec_provenance_hint() or "")


def test_a_cuda_index_codec_also_installs_npp():
    """torchcodec's CUDA build dlopens libnppicc and libnppc, and NPP is not in torch's own
    dependency set, so a --no-deps install from a cuNNN index reports success and then fails
    to import. docker/Dockerfile installs nvidia-npp-cu12 beside the same wheel."""
    source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    assert 'f"nvidia-npp-cu{_npp_major}"' in source
    assert "Installing torchcodec CUDA runtime (NPP)" in source

    # The major follows the index leaf, and a cpu or rocm index asks for nothing.
    import re

    for url, want in (
        ("https://download.pytorch.org/whl/cu128", "12"),
        ("https://download.pytorch.org/whl/cu130", "13"),
        ("https://mirror.corp.example/whl/cu128/", "12"),
        ("https://download.pytorch.org/whl/cpu", None),
        ("https://download.pytorch.org/whl/rocm6.3", None),
    ):
        match = re.search(r"/cu(\d+)/?$", url)
        assert (match.group(1)[:2] if match else None) == want, url

    # The Dockerfile this mirrors still pairs the two, so the rationale stays checkable.
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding = "utf-8")
    assert "nvidia-npp-cu12" in dockerfile


def test_the_npp_major_comes_from_the_resident_torch_not_the_index_url():
    """Matching `/cuNNN$` against an authenticated mirror ending `/simple?token=...` skipped
    NPP on a `+cu128` host, so the codec then failed to import without a CUDA toolkit."""
    import re as _re

    source = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    namespace: dict = {"re": _re}
    exec(
        _re.search(r"def _cuda_major_for_npp.*?\n(?=\n\ndef )", source, _re.S).group(0),
        namespace,
    )
    npp_major = namespace["_cuda_major_for_npp"]

    # The local tag answers whatever the index URL looks like.
    opaque = "https://mirror.example/simple?token=abc"
    assert npp_major("2.11.0+cu128", opaque) == "12"
    assert npp_major("2.11.0+cu130", opaque) == "13"
    # cpu and rocm ask for nothing, on any URL.
    assert npp_major("2.11.0+cpu", "https://download.pytorch.org/whl/cpu") == ""
    assert npp_major("2.11.0+rocm6.4", "https://download.pytorch.org/whl/rocm6.4") == ""
    # An untagged torch still falls back to the public leaf.
    assert npp_major("2.11.0", "https://download.pytorch.org/whl/cu126") == "12"
    assert npp_major("2.11.0", opaque) == ""

    # And the call site reads the tag rather than re-matching the URL.
    assert "_cuda_major_for_npp(_codec_torch_ver, _codec_index)" in source


def test_the_provenance_hint_reads_a_codec_it_cannot_import(monkeypatch):
    """An unloadable wheel usually raises while `torchcodec/__init__` imports its decoders, and
    Python drops a module whose initialisation raised, so importing it again just repeats the
    exception. Reading the version back that way left the accelerator mismatch undiagnosed in
    exactly the case the hint exists to name; the installer's metadata needs no native library."""
    import importlib.metadata
    import sys

    fixes = _load_import_fixes_module()
    _stub_torch(monkeypatch, "2.11.0+cu128")
    # No importable torchcodec at all, which is what a failed initialisation leaves behind.
    monkeypatch.delitem(sys.modules, "torchcodec", raising = False)

    def _version(name):
        if name == "torchcodec":
            return "0.11.0"  # untagged: PyPI's default build
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _version)
    hint = fixes._torchcodec_provenance_hint()
    assert hint is not None
    assert "torchcodec 0.11.0 came from the default index" in hint
    assert "https://download.pytorch.org/whl/cu128" in hint

    # Nothing installed at all still says nothing.
    def _absent(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _absent)
    assert fixes._torchcodec_provenance_hint() is None
