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
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.11.0",
    )

    # Untagged torch: no index pin is needed, so the extra stays on offer.
    _stub_torch(monkeypatch, "2.10.0")
    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "torchcodec 0.11.0" in hint
    assert "audio-torch210" in hint
    assert "<0.11.0" in hint
    assert "<11.0" not in hint

    # Tagged torch: the extra cannot carry an index, so the pinned command is offered alone.
    _stub_torch(monkeypatch, "2.10.0+cu128")
    tagged = fixes._torchcodec_version_mismatch_hint()
    assert "--index-url https://download.pytorch.org/whl/cu128" in tagged
    assert "audio-torch210" not in tagged
    assert "<0.11.0" in tagged


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
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.10.0+cu128",
    )
    # Untagged, so the extra is offered; the tagged case is covered below.
    _stub_torch(monkeypatch, "2.11.0")

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
    _stub_torch(monkeypatch, "2.10.0")
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


def test_audio_extras_carry_the_python_ceiling_their_codec_line_has():
    """torchcodec publishes no sdist, so an extra left open above its last cp tag makes pip
    fail the whole install instead of skipping audio. requires-python is open-ended (>=3.9),
    so a newer interpreter reaches these extras; the marker has to stop it.

    The ceilings come from the same upstream table _TORCHCODEC_PYTHON_WINDOWS encodes: the
    0.6/0.7 line stops at 3.13, everything from 0.9 up runs to 3.14.
    """
    markers = pytest.importorskip("packaging.markers")
    text = PYPROJECT.read_text(encoding = "utf-8")

    # extra -> the first interpreter that must NOT select it, and one that must.
    expected = {
        "audio-torch211": ("3.15", "3.14"),
        "audio-torch210": ("3.15", "3.14"),
        "audio-torch290": ("3.15", "3.14"),
        "audio-torch280": ("3.14", "3.13"),
    }
    for extra, (too_new, supported) in expected.items():
        match = re.search(rf"^{extra} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
        assert match is not None, extra
        marker_text = match.group(1).split(";", 1)[1].rsplit('"', 1)[0].strip()
        marker = markers.Marker(marker_text)
        env = {
            "sys_platform": "linux",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "os_name": "posix",
        }
        assert not marker.evaluate(
            {**env, "python_version": too_new}
        ), f"{extra} still selects torchcodec on Python {too_new}, which has no wheel"
        assert marker.evaluate(
            {**env, "python_version": supported}
        ), f"{extra} stopped selecting torchcodec on Python {supported}, which does"


def test_compat_matrix_matches_the_published_upstream_table():
    """Pin the runtime guard to upstream, not merely to our own other copies of it."""
    fixes = _load_import_fixes_module()
    assert fixes._TORCH_TORCHCODEC_MINORS == _UPSTREAM_TORCH_TO_TORCHCODEC_MINORS


def test_installer_never_selects_a_torchcodec_built_against_another_torch():
    """The window handed to pip must not contain a release upstream pairs with a different
    torch: pip takes the HIGHEST match, so a window one minor too wide installs the mismatch
    this whole module exists to prevent."""
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


def test_audio_extras_are_gated_to_platforms_with_a_torchcodec_wheel():
    """torchcodec publishes no sdist, so an ungated pin makes pip fail the whole install on
    a host with no wheel instead of just skipping audio -- and the cu*/rocm*/intel torch
    2.10 extras pull it in. The marker must match install_python_stack.py.

    Linux aarch64 is per-extra rather than blanket: torchcodec had no aarch64 wheel when
    this test was written, but 0.11.0 added manylinux_2_28_aarch64 and every release since
    has kept it. So audio-torch211, whose line is >=0.11,<0.12, must ALLOW aarch64, while
    the older extras, which top out at 0.10, must still exclude it. Windows ARM64 and Intel
    Mac have no wheel at any version and stay excluded everywhere.
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
    # No wheel at any torchcodec version, so excluded from every extra.
    never_supported = [
        {"sys_platform": "win32", "platform_machine": "ARM64"},
        {"sys_platform": "darwin", "platform_machine": "x86_64"},
    ]
    linux_aarch64 = {"sys_platform": "linux", "platform_machine": "aarch64"}
    # The extras whose window reaches 0.11.0, where the aarch64 wheel first appears.
    aarch64_capable = {"audio-torch211"}

    for name, deps in audio.items():
        for dep in deps:
            _, _, marker_text = dep.partition(";")
            assert marker_text.strip(), f"{name}: {dep!r} has no marker"
            marker = markers.Marker(marker_text.strip())
            env = {"python_version": "3.12"}
            for case in supported:
                assert marker.evaluate({**env, **case}), f"{name} must install on {case}"
            for case in never_supported:
                assert not marker.evaluate(
                    {**env, **case}
                ), f"{name} has no wheel for {case} and must not be resolved there"
            allows_aarch64 = marker.evaluate({**env, **linux_aarch64})
            if name in aarch64_capable:
                assert allows_aarch64, (
                    f"{name} selects the >=0.11 line, which ships manylinux_2_28_aarch64, "
                    "so it must not exclude Linux aarch64"
                )
            else:
                assert not allows_aarch64, (
                    f"{name} tops out below 0.11, where no aarch64 wheel exists, "
                    "so it must not be resolved there"
                )


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


def test_a_requested_codec_range_beats_the_preinstalled_oracle():
    """resolved_set only overrides the oracle on an exact `==`, so a cell asking for a RANGE
    still read as Colab's preinstalled codec and R-INST-004 (error severity) fired on
    notebooks pip would have resolved correctly.

    Both bounds matter. `torchcodec>=0.12.0,<0.13.0` on torch 2.12 was reported against the
    image's 0.11; `torchcodec>=0.10.0,<0.11.0` on torch 2.10 was reported for the mirror
    reason, the ceiling ruling the oracle out rather than the floor.
    """
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

    # The rule must still fire where pip really does leave a mismatch: an exact wrong pin,
    # a bare torch upgrade that leaves the oracle codec in place, and a floor that is itself
    # incompatible with the requested torch.
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

    Order: pip runs the commands in sequence, so `torchcodec>=0.12.0` then
    `torchcodec<0.12.0` ends pre-0.12. Intersecting the bounds across both invocations
    instead yields a 0.12 that was never installed, and the real mismatch goes unreported.

    Markers: a requirement pip skips must not move anything. This branch has no oracle for
    the image's interpreter, so a marked requirement is left alone and the cell is judged on
    the preinstalled version, exactly as it was before the range reader existed.
    """
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


def test_linux_aarch64_is_served_from_the_011_line_onwards(monkeypatch):
    """aarch64 got its first wheel at 0.11.0, which is the line this branch selects."""
    ips = _load_install_python_stack()
    _patch_host(ips, monkeypatch, "linux-aarch64")
    assert ips._PLATFORM_HAS_TORCHCODEC_WHEEL is not None
    # torch 2.11 -> the 0.11 line, which has aarch64.
    assert ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.11.0"))
    # torch 2.10 -> the 0.10 line, which does not.
    assert not ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.10.0"))


def test_a_mac_below_14_declines_the_abi_stable_line(monkeypatch):
    """torchcodec 0.12+ is macosx_14_0 only, so an older Mac must not be sent to it."""
    ips = _load_install_python_stack()
    _patch_host(ips, monkeypatch, "macos-arm64-13")
    assert not ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.12.0"))
    assert ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.11.0"))
    _patch_host(ips, monkeypatch, "macos-arm64-14")
    assert ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.12.0"))


def test_the_gate_declines_a_line_whose_python_window_excludes_this_interpreter(monkeypatch):
    """Architecture is not the only wheel axis. torch 2.5 selects the 0.1 line, and 0.1 stops
    at Python 3.12 -- on 3.13 there is no wheel to install even on plain linux-x86_64.

    This was masked while the 2.5 window ran to <0.3.0: it reached 0.2, which does ship cp313,
    so the gate said yes for a release built against torch 2.6.
    """
    ips = _load_install_python_stack()
    _patch_host(ips, monkeypatch, "linux-x86_64")
    spec = ips._select_torchcodec_spec("2.5.0")

    monkeypatch.setattr(ips.sys, "version_info", (3, 12, 0, "final", 0))
    assert ips._torchcodec_spec_is_installable(spec)
    monkeypatch.setattr(ips.sys, "version_info", (3, 13, 0, "final", 0))
    assert not ips._torchcodec_spec_is_installable(spec)

    # The floor moves too: 0.8+ dropped 3.9, so torch 2.9 has nothing for a 3.9 interpreter.
    monkeypatch.setattr(ips.sys, "version_info", (3, 9, 0, "final", 0))
    assert not ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.9.0"))
    assert ips._torchcodec_spec_is_installable(ips._select_torchcodec_spec("2.8.0"))


def test_python_windows_match_the_published_upstream_table():
    """Same reason as the compat-matrix pin: transcribed from upstream, not from ourselves."""
    ips = _load_install_python_stack()
    assert ips._TORCHCODEC_PYTHON_WINDOWS == (
        ((0, 1, 0), (3, 9), (3, 12)),
        ((0, 2, 0), (3, 9), (3, 13)),
        ((0, 8, 0), (3, 10), (3, 13)),
        ((0, 9, 0), (3, 10), (3, 14)),
    )


def test_windows_declines_the_pre_070_lines(monkeypatch):
    """win_amd64 starts at 0.7.0; torch 2.5-2.7 select windows below it.

    Reachable rather than theoretical: the cu118 index tops out at torch 2.7.
    """
    ips = _load_install_python_stack()
    _patch_host(ips, monkeypatch, "windows-amd64")
    for minor in (5, 6, 7):
        spec = ips._select_torchcodec_spec(f"2.{minor}.0")
        assert not ips._torchcodec_spec_is_installable(spec), spec
    for minor in (8, 10, 11, 12):
        spec = ips._select_torchcodec_spec(f"2.{minor}.0")
        assert ips._torchcodec_spec_is_installable(spec), spec


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


def test_a_ceiling_only_request_is_unknown_rather_than_the_excluded_oracle():
    """`pip install "torchcodec<0.10.0"` on a 0.11 image downgrades to a version only the
    index names. Returning the excluded oracle reported the exact pairing the cell just
    ruled out, so a clean notebook failed on torch 2.9 + the image's 0.11."""
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


def test_the_codec_reader_matches_pip_on_names_and_uninstalls():
    """Two ways the reader kept judging a codec the cell had already dealt with.

    pip compares distribution names case-insensitively (PEP 503), and parse_spec already
    lowercases for that reason, so `TorchCodec>=0.12.0` is the same requirement and has to
    move the version. And an uninstall leaves nothing installed to judge, but the reader
    still returned the oracle, so the branch this PR added reported a package the cell had
    just deleted. Both were errors on a compatible notebook.
    """
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
    """`~=` is a two-sided bound (PEP 440: `~=0.12.0` is `>=0.12.0,<0.13.0`) and `<=V`
    names its own landing version. Reading neither meant the oracle survived a request that
    had already moved it, and R-INST-004 reported the version pip replaced."""
    from scripts import notebook_validator as nv

    assert nv._compatible_release_ceiling("0.12.0") == "0.13"
    assert nv._compatible_release_ceiling("0.12") == "1"
    assert nv._compatible_release_ceiling("1") == ""  # `~=1` is invalid, so it bounds nothing

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


def test_a_bounded_window_lands_on_its_newest_candidate():
    """pip resolves a window to the newest release it admits, not to the floor
    (https://pip.pypa.io/en/stable/topics/dependency-resolution/).

    `torchcodec>=0.8,<0.11` on torch 2.9 therefore installs the 0.10 line, which 2.9 does
    not support. Modelling it as the floor read 0.8, called that compatible, and the finding
    disappeared. The ceiling names the landing minor without needing an index.
    """
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


def test_a_patch_level_ceiling_keeps_its_own_minor():
    """`<0.10.5` still admits 0.10.0 through 0.10.4, so it lands on 0.10. Decrementing the
    minor regardless modelled it as 0.9 and suppressed a real torch 2.9 mismatch."""
    from scripts import notebook_validator as nv

    assert nv._highest_minor_below("0.10.5") == "0.10"
    assert nv._highest_minor_below("0.10.0") == "0.9"  # on the boundary, so 0.10 is excluded
    assert nv._highest_minor_below("0.11") == "0.10"

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    cell = '!pip install torch==2.9.0 "torchcodec<0.10.5"'
    assert nv._effective_requested_version(cell, "torchcodec", "0.11.0") == ("0.10", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]


def test_a_strict_lower_bound_excludes_the_installed_version():
    """`>0.11.0` rules out the 0.11.0 the image ships, and which release pip picks instead is
    only in the index. Recording nothing kept the excluded 0.11.0 and rejected the notebook."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    cell = '!pip install torch==2.12.0 "torchcodec>0.11.0"'
    assert nv._effective_requested_version(cell, "torchcodec", "0.11.0") == ("", True)
    assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == []

    # A strict bound the installed version already clears leaves it alone.
    satisfied = '!pip install "torchcodec>0.10.0"'
    assert nv._effective_requested_version(satisfied, "torchcodec", "0.11.0") == ("0.11.0", True)


def test_a_later_install_keeps_what_an_earlier_one_landed_on():
    """pip does not reinstall a package that already satisfies the new requirement, so
    `>=0.12.0` followed by the broader `>=0.10.0` ends on the 0.12. Reading only the last
    command re-evaluated the broader range against the image's 0.11 and rejected it."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    widened = '!pip install torch==2.12.0 "torchcodec>=0.12.0"\n!pip install "torchcodec>=0.10.0"'
    assert nv._effective_requested_version(widened, "torchcodec", "0.11.0") == ("0.12.0", False)
    assert nv.rule_inst_004_torchcodec_torch(widened, colab, "nb.ipynb", 0) == []

    # A later command that does NOT admit the earlier landing still moves it back down.
    narrowed = '!pip install "torchcodec>=0.12.0"\n!pip install "torchcodec<0.12.0"'
    assert nv._effective_requested_version(narrowed, "torchcodec", "0.11.0") == ("0.11", True)

    # An exact pin after an uninstall restores a version rather than staying gone.
    restored = '!pip uninstall -y torchcodec\n!pip install "torchcodec==0.11.1"'
    assert nv._effective_requested_version(restored, "torchcodec", "0.11.0") == ("0.11.1", True)


def test_an_exclusion_rules_out_the_installed_version():
    """`!=` inverts `==` (PEP 440 version exclusion), so `torchcodec!=0.11.0` really does
    make pip replace the image's 0.11.0. Ignoring the operator kept the excluded version and
    R-INST-004 rejected a torch 2.12 notebook that pip resolves to an ABI-stable codec."""
    from scripts import notebook_validator as nv

    # An exact clause blocks one release after zero-padding; a `.*` clause blocks the prefix.
    # Local labels are not part of the comparison.
    assert nv._version_is_excluded("0.11.0+cu128", "0.11.0")
    assert nv._version_is_excluded("0.11.0", "0.11")
    assert nv._version_is_excluded("0.11.5", "0.11.*")
    assert not nv._version_is_excluded("0.11.1", "0.11.0")
    assert not nv._version_is_excluded("0.12.0", "0.11.*")

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    for cell in (
        '!pip install "torch==2.12.0" "torchcodec!=0.11.0"',
        '!pip install "torch==2.12.0" "torchcodec!=0.11.*"',
    ):
        assert nv._effective_requested_version(cell, "torchcodec", "0.11.0") == ("", True), cell
        assert nv.rule_inst_004_torchcodec_torch(cell, colab, "nb.ipynb", 0) == [], cell

    # An exclusion that does NOT cover the installed version leaves it alone, so a real
    # mismatch is still reported.
    untouched = '!pip install "torch==2.9.0" "torchcodec!=0.12.0"'
    assert nv._effective_requested_version(untouched, "torchcodec", "0.11.0") == ("0.11.0", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(untouched, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # A window whose own landing the exclusion rules out names nothing.
    self_excluded = '!pip install "torch==2.12.0" "torchcodec>=0.12.0,!=0.12.0"'
    assert nv._effective_requested_version(self_excluded, "torchcodec", "0.11.0") == ("", True)
    # ...but an exclusion elsewhere in the range leaves the landing intact.
    kept = '!pip install "torch==2.12.0" "torchcodec>=0.12.0,!=0.11.0"'
    assert nv._effective_requested_version(kept, "torchcodec", "0.11.0") == ("0.12.0", False)


def test_an_open_floor_is_a_floor_not_a_landing():
    """pip installs the NEWEST release above an open `>=`, so `torchcodec>=0.11.1` on a
    torch 2.12 cell can land on the ABI-stable line. Recording the floor as the answer
    reported a mismatch against a version pip would never have left in place."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    open_floor = '!pip install "torch==2.12.0" "torchcodec>=0.11.1"'
    assert nv._effective_requested_version(open_floor, "torchcodec", "0.11.0") == ("0.11.1", False)
    assert nv.rule_inst_004_torchcodec_torch(open_floor, colab, "nb.ipynb", 0) == []

    # The floor is still usable where every version at or above it gives the same answer:
    # torch 2.10 cannot take any 0.12+ codec, so this one is still reported.
    too_old = '!pip install "torch==2.10.0" "torchcodec>=0.12.0"'
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(too_old, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]


def test_versions_are_compared_with_pep440_zero_padding():
    """PEP 440 pads the shorter release segment, so `0.11` and `0.11.0` are one version.
    Comparing the raw tuples sorted `0.11` below `0.11.0`, which threw away the
    ceiling-derived minor whenever the floor spelled out its patch."""
    from scripts import notebook_validator as nv

    assert nv.cmp_versions("0.11", "0.11.0") == 0
    assert nv.cmp_versions("2.12", "2.12.0.0") == 0
    assert nv.cmp_versions("0.11", "0.11.1") == -1
    assert nv.cmp_versions("0.12", "0.11.9") == 1

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    strict_window = '!pip install "torch==2.12.0" "torchcodec>0.11.0,<0.12.0"'
    assert nv._effective_requested_version(strict_window, "torchcodec", "0.11.0") == ("0.11", True)
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(strict_window, colab, "nb.ipynb", 0)
    ] == ["R-INST-004"]


def test_the_codec_index_honours_an_explicitly_pinned_torch_mirror(monkeypatch):
    """UNSLOTH_TORCH_INDEX_URL names the index torch itself came from. Rebuilding a public
    download.pytorch.org URL from the local tag sent an authenticated or air-gapped mirror
    to the internet, and the `--index-url` that follows also drops the inherited index
    configuration, so the codec install fails outright where public PyTorch is unreachable."""
    from studio import install_python_stack as ips

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://download.pytorch.org/whl/cu128"

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/pytorch/cu128/")
    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://mirror.corp.example/pytorch/cu128"

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu126")
    assert ips._torchcodec_index_url("2.11.0+cu128") == "https://download.pytorch.org/whl/cu126"

    # The override does not make an untagged or rocm torch start pinning.
    assert ips._torchcodec_index_url("2.11.0") is None
    assert ips._torchcodec_index_url("2.11.0+rocm7.0") is None


def test_the_ceiling_landing_respects_an_exclusion_that_covers_it():
    """`>=0.8,<0.11,!=0.10.*` cannot land on 0.10, so pip drops to a lower release the index
    names. Returning the ceiling-derived minor regardless reported a torch 2.9 mismatch
    against a line the cell had already excluded."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    excluded_top = '!pip install torch==2.9.0 "torchcodec>=0.8,<0.11,!=0.10.*"'
    assert nv._effective_requested_version(excluded_top, "torchcodec", "0.11.0") == ("", True)
    assert nv.rule_inst_004_torchcodec_torch(excluded_top, colab, "nb.ipynb", 0) == []

    # An exclusion that misses the landing leaves it alone.
    kept = '!pip install "torchcodec>=0.8,<0.11,!=0.9.*"'
    assert nv._effective_requested_version(kept, "torchcodec", "0.11.0") == ("0.10", True)


def test_an_equal_strict_bound_upgrades_the_floor():
    """`>=V` and `>V` intersect to `>V`. Keeping the inclusive one let `>=0.10,>0.10` read as
    "0.10 is fine" when pip must move above it, suppressing a real R-INST-004."""
    from scripts import notebook_validator as nv

    combined = '!pip install torch==2.10.0 "torchcodec>=0.10,>0.10,<0.12"'
    _, floor, ceiling, _cap, exclusive, _removed, _excl = nv._requested_bounds(
        combined, "torchcodec"
    )[0]
    assert (floor, ceiling, exclusive) == ("0.10", "0.12", True)

    colab = {"torch": "2.10.0+cu128", "torchcodec": "0.10.0+cu128"}
    assert nv._effective_requested_version(combined, "torchcodec", "0.10.0") == ("0.11", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(combined, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # Order does not matter, and a plain `>=` is still inclusive.
    reordered = '!pip install "torchcodec>0.10,>=0.10,<0.12"'
    assert nv._requested_bounds(reordered, "torchcodec")[0][4] is True
    assert (
        nv._requested_bounds('!pip install "torchcodec>=0.10,<0.12"', "torchcodec")[0][4] is False
    )


def test_the_runtime_remedy_honours_a_configured_torch_index(monkeypatch):
    """A warning that tells an air-gapped or authenticated host to install from
    download.pytorch.org either fails or bypasses the artifact source the install was
    configured with. The mirror is where the matching build is."""
    import importlib.metadata

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.11.0")
    _stub_torch(monkeypatch, "2.10.0+cu128")

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
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
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
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
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)

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


def test_a_chained_uninstall_does_not_swallow_the_reinstall():
    """`!pip uninstall -y torchcodec && pip install torchcodec==0.10.0` is one logical line,
    so matching `uninstall` anywhere in it marked the whole thing a removal and the codec the
    cell demonstrably ends with was thrown away. The LAST pip verb decides."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    chained = "!pip uninstall -y torchcodec && pip install torchcodec==0.10.0"
    assert nv._effective_requested_version(chained, "torchcodec", "0.11.0") == ("0.10.0", True)
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(chained, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # A line that really does end on a removal still clears it, in either spelling.
    for removed in (
        "!pip uninstall -y torchcodec",
        "!uv pip uninstall torchcodec",
        "!pip install torchcodec==0.10.0 && pip uninstall -y torchcodec",
    ):
        assert nv._effective_requested_version(removed, "torchcodec", "0.11.0") == (
            "",
            True,
        ), removed
        assert nv.rule_inst_004_torchcodec_torch(removed, colab, "nb.ipynb", 0) == [], removed


def test_the_codec_index_follows_a_configured_pytorch_mirror(monkeypatch):
    """UNSLOTH_PYTORCH_MIRROR replaces the base every other index in install_python_stack is
    built from, so a codec pinned to the public site cannot be fetched on an air-gapped host
    and bypasses the artifact source on a corporate one."""
    import importlib

    monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", "https://mirror.corp.example/whl")
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
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
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)

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


def test_a_torch_range_is_replayed_before_the_pair_is_judged():
    """`pip install "torch>=2.12.0"` does not satisfy the image's 2.11, so pip upgrades torch
    while the codec stays on 0.11. Replaying only the codec left torch at the oracle and the
    rule saw a pairing the cell had already moved away from."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}
    upgraded = '!pip install "torch>=2.12.0"'
    assert [f.rule for f in nv.rule_inst_004_torchcodec_torch(upgraded, colab, "nb.ipynb", 0)] == [
        "R-INST-004"
    ]

    # A floor the image already satisfies moves nothing, so the pair stays as shipped.
    assert (
        nv.rule_inst_004_torchcodec_torch('!pip install "torch>=2.11.0"', colab, "nb.ipynb", 0)
        == []
    )
    # Removing torch leaves nothing to judge.
    assert nv.rule_inst_004_torchcodec_torch("!pip uninstall -y torch", colab, "nb.ipynb", 0) == []
    # An inexact torch cannot pick a row, so the table half stays silent rather than guessing.
    assert (
        nv.rule_inst_004_torchcodec_torch('!pip install "torch>=2.7,<2.8"', colab, "nb.ipynb", 0)
        == []
    )


def test_the_provenance_hint_does_not_assert_a_cause_it_has_not_established(monkeypatch):
    """Differing local tags show the two wheels came from different indexes, nothing more.
    torchcodec is published per accelerator on every line, 0.12+ included, so the mismatch
    stays possible there, but the load can equally have failed on a missing libavutil that no
    reinstall repairs. The hint has to name both."""
    import sys
    import types

    fixes = _load_import_fixes_module()
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.delenv("UNSLOTH_PYTORCH_MIRROR", raising = False)
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


def test_each_pip_command_owns_the_packages_it_names():
    """A chained line can carry several verbs, and this branch has no shell splitter, so the
    whole line arrives as one invocation. A single verb for it is wrong both ways: the first
    verb threw away a reinstall after an uninstall, and the last verb cleared torch and the
    codec when a THIRD package was the one removed."""
    from scripts import notebook_validator as nv

    colab = {"torch": "2.11.0+cu128", "torchcodec": "0.11.0+cu128"}

    # The uninstall names torchaudio, so the incompatible pair the install leaves stands.
    other_package = (
        "!pip install torch==2.10.0 torchcodec==0.12.0 && pip uninstall -y torchaudio"
    )
    assert nv._effective_requested_version(other_package, "torchcodec", "0.11.0") == (
        "0.12.0",
        True,
    )
    assert nv._effective_requested_version(other_package, "torch", "2.11.0") == ("2.10.0", True)
    assert [
        f.rule for f in nv.rule_inst_004_torchcodec_torch(other_package, colab, "nb.ipynb", 0)
    ] == ["R-INST-004"]

    # A reinstall after an uninstall of the SAME package still lands.
    reinstalled = "!pip uninstall -y torchcodec && pip install torchcodec==0.10.0"
    assert nv._effective_requested_version(reinstalled, "torchcodec", "0.11.0") == ("0.10.0", True)

    # ...and an uninstall that really does come last still clears it.
    removed = "!pip install torchcodec==0.10.0 && pip uninstall -y torchcodec"
    assert nv._effective_requested_version(removed, "torchcodec", "0.11.0") == ("", True)
    assert nv.rule_inst_004_torchcodec_torch(removed, colab, "nb.ipynb", 0) == []
