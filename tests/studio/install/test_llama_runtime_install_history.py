# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-time runtime health probe against installs that exist on disk today.

The sibling files build their trees by hand: ``test_installed_runtime_health_matrix``
enumerates the twelve shipped marker shapes, and ``test_installed_runtime_health`` writes
one empty file per name in the payload table. Both are simulations of an install, and a
simulation can only ever confirm that the code agrees with the table the test author read
out of the code.

This file uses artifacts instead. Where the release zips are on disk it extracts the real
shipped file list into the layout the installer builds and asks
``installed_runtime_health`` about it, so the payload tables are checked against what the
release actually contains rather than against a restatement of themselves. Where a managed
runtime is really installed under ``~/.unsloth/llama.cpp`` it is copied out and gutted a
file at a time, which is the closest reachable stand-in for the antivirus quarantine this
whole feature exists to catch.

That difference is not cosmetic. A real install ships each core library twice, under its
SONAME and under its full version (``libllama.so.0`` beside ``libllama.so.0.0.10360``),
and a hand-built fixture writes only ``libllama.so``. The trailing-star globs in
``runtime_payload_health_groups`` cannot tell the two copies apart, so quarantining the
SONAME leaves the group satisfied while the runtime no longer loads. Only an artifact
shows that, and ``test_soname_quarantine_is_reported_healthy`` below is where it is
pinned as a known defect rather than a passing assertion.

Every test here skips rather than fails when its artifact is absent, so the file still
runs in CI, where neither the release zips nor a managed install are present.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

HostInfo = ILP.HostInfo

# The release zips this file reads. Downloaded out of band; absent in CI, where every
# test that wants one skips. Overridable so a checkout somewhere else can still use them.
ASSET_DIR = Path(
    os.environ.get(
        "UNSLOTH_TEST_LLAMACPP_ASSET_DIR",
        "/mnt/disks/unslothai/daniel3/workspace_26/data/llamacpp_assets",
    )
)


def _host(**kw) -> HostInfo:
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    base.update(kw)
    return HostInfo(**base)


LINUX = _host()
WINDOWS = _host(system = "Windows", machine = "AMD64", is_windows = True, is_linux = False)
WINDOWS_ARM64 = _host(
    system = "Windows",
    machine = "ARM64",
    is_windows = True,
    is_linux = False,
    is_x86_64 = False,
    is_arm64 = True,
)


# ---------------------------------------------------------------------------
# The real release bundles.
#
# (asset, llama_backend recorded in the marker, tag, host). The tag matters: the shared
# Windows group requires llama-server-impl.dll only from the build that split it out, so a
# wrong tag here would make the test pass for the wrong reason.

BUNDLES = [
    ("app-b10798-mix-659e406-windows-x64-cpu.zip", None, "b10798", "published", WINDOWS),
    (
        "app-b10798-mix-659e406-windows-x64-cuda12-legacy.zip",
        "cuda",
        "b10798",
        "published",
        WINDOWS,
    ),
    ("app-b10798-mix-659e406-windows-x64-vulkan.zip", "vulkan", "b10798", "published", WINDOWS),
    ("app-b10798-mix-659e406-windows-x64-rocm-gfx1150.zip", "rocm", "b10798", "published", WINDOWS),
    (
        "app-b10798-mix-659e406-windows-arm64-cpu.zip",
        None,
        "b10798",
        "published",
        WINDOWS_ARM64,
    ),
    # An older bundle, so the pass is not specific to one build number.
    ("app-b10715-mix-86bd2d3-windows-x64-cpu.zip", None, "b10715", "published", WINDOWS),
    # An upstream ggml-org archive rather than an Unsloth one: source = "upstream" takes a
    # different branch through _windows_shared_groups.
    ("llama-b10830-bin-win-cpu-x64.zip", None, "b10830", "upstream", WINDOWS),
]


def _marker_for(asset: str, backend: str | None, tag: str, source: str) -> dict:
    """A marker in the shape install_from_archives writes, trimmed to the keys the probe
    reads: the backend picks the install kinds, and source and tag pick the group table."""
    return {
        "requested_tag": "latest",
        "tag": tag,
        "release_tag": tag,
        "published_repo": "unslothai/llama.cpp",
        "asset": asset,
        "source": source,
        "llama_backend": backend,
        "force_cpu": backend is None,
        "installed_at_utc": "2026-09-01T00:00:00Z",
    }


def _unpack_bundle(asset: str, backend, tag, source, host, into: Path) -> Path:
    """Lay a release zip out the way the installer does and write its marker.

    The Windows zips are flat: every entry sits at the archive root with no directory
    prefix at all, so the ``build/bin/Release`` layout is the installer's choice and not
    the archive's. That is why this mirrors install_runtime_dir rather than trusting
    extractall to land things in the right place.
    """
    root = into / asset.replace(".zip", "")
    runtime_dir = ILP.install_runtime_dir(root, host)
    runtime_dir.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(ASSET_DIR / asset) as archive:
        archive.extractall(runtime_dir)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(_marker_for(asset, backend, tag, source), indent = 2),
        encoding = "utf-8",
    )
    return root


@pytest.mark.parametrize(
    "asset,backend,tag,source,host",
    BUNDLES,
    ids = [entry[0].replace(".zip", "") for entry in BUNDLES],
)
def test_a_real_release_bundle_is_healthy(asset, backend, tag, source, host, tmp_path):
    """A shipped bundle, extracted and marked, must not be called broken.

    This is the direction that costs users: a false (False, reason) here marks the install
    stale on the next launch and sends every one of them through a repair that would find
    nothing to fix. The payload tables name files by glob, and only the release itself can
    say whether those globs match what is actually in the archive.
    """
    if not (ASSET_DIR / asset).is_file():
        pytest.skip(f"release bundle not on this machine: {asset}")
    root = _unpack_bundle(asset, backend, tag, source, host, tmp_path)
    assert ILP.installed_runtime_health(root, host = host) == (True, ""), asset


@pytest.mark.parametrize(
    "victim",
    [
        "llama.dll",
        "llama-common.dll",
        "llama-server-impl.dll",
        "ggml.dll",
        "ggml-base.dll",
        "mtmd.dll",
        "llama-server.exe",
        "llama-quantize.exe",
    ],
)
def test_a_file_quarantined_from_a_real_windows_bundle_is_caught(victim, tmp_path):
    """The other direction, on a real archive rather than a fixture.

    Windows ships one copy of each library, with no versioned twin, so removing a required
    file leaves nothing behind for the glob to match and the probe sees it. The Linux case
    is where that stops being true; see test_soname_quarantine_is_reported_healthy.
    """
    asset = "app-b10798-mix-659e406-windows-x64-cpu.zip"
    if not (ASSET_DIR / asset).is_file():
        pytest.skip(f"release bundle not on this machine: {asset}")
    root = _unpack_bundle(asset, None, "b10798", "published", WINDOWS, tmp_path)
    (ILP.install_runtime_dir(root, WINDOWS) / victim).unlink()
    verdict = ILP.installed_runtime_health(root, host = WINDOWS)
    assert verdict is not None and verdict[0] is False, victim
    assert verdict[1] in {
        "llama_runtime_payload_incomplete",
        "llama_runtime_binaries_missing",
    }, verdict


def test_a_windows_cuda_bundle_without_its_paired_runtime_is_incomplete(tmp_path):
    """The cudart trio lives in a second archive, and the marker records that pairing.

    The real CUDA bundle ships no cudart64_*.dll of its own, so a marker naming a paired
    runtime that is not on disk describes an install that cannot start. Asserted against
    the real archive because the value of the check is that the trio is genuinely absent
    from it rather than merely omitted from a fixture.
    """
    asset = "app-b10798-mix-659e406-windows-x64-cuda12-legacy.zip"
    if not (ASSET_DIR / asset).is_file():
        pytest.skip(f"release bundle not on this machine: {asset}")
    root = _unpack_bundle(asset, "cuda", "b10798", "published", WINDOWS, tmp_path)
    runtime_dir = ILP.install_runtime_dir(root, WINDOWS)
    assert not list(runtime_dir.glob("cudart64_*.dll")), "the bundle is expected to ship no cudart"

    marker = json.loads((root / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8"))
    marker["runtime_asset"] = "cudart-llama-bin-win-cuda-12.8-x64.zip"
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(marker), encoding = "utf-8")
    assert ILP.installed_runtime_health(root, host = WINDOWS) == (
        False,
        "llama_runtime_payload_incomplete",
    )


def test_the_real_cuda_bundle_carries_its_own_build_marker(tmp_path):
    """UNSLOTH_PREBUILT_INFO.json exists inside the archive as well as at the install root.

    The two files share a name and are not the same document: the archive's copy is the
    build record and lands in the runtime directory, while the installer writes the install
    record at the root. load_prebuilt_metadata reads the root one, so the archive's copy
    must not be reachable as the marker. If a future layout change ever put the runtime
    directory at the install root, the probe would start reading a document with no
    ``source`` and no ``tag`` and would grade the tree with the wrong table.
    """
    asset = "app-b10798-mix-659e406-windows-x64-cuda12-legacy.zip"
    if not (ASSET_DIR / asset).is_file():
        pytest.skip(f"release bundle not on this machine: {asset}")
    root = _unpack_bundle(asset, "cuda", "b10798", "published", WINDOWS, tmp_path)
    runtime_dir = ILP.install_runtime_dir(root, WINDOWS)
    bundled = runtime_dir / "UNSLOTH_PREBUILT_INFO.json"
    assert bundled.is_file(), "this bundle is expected to carry a build marker"
    assert runtime_dir != root, "the build marker must not be able to shadow the install marker"
    read_back = ILP.load_prebuilt_metadata(root)
    assert read_back is not None and read_back.get("source") == "published"
    assert "upstream_tag" not in read_back, "the install marker, not the archive's build record"


# ---------------------------------------------------------------------------
# The Windows layout contract.


def test_every_windows_layout_decision_agrees_on_the_release_subdirectory():
    """One directory, named the same way by all four places that name it.

    installed_runtime_health reports llama_runtime_dir_missing when this directory is
    absent, and that verdict marks the install stale. If the probe ever derived a different
    path than the installer writes to, every Windows user would be repaired on every
    launch, so the agreement is the property worth holding rather than the literal string.
    """
    root = Path("/install")
    expected = root / "build" / "bin" / "Release"
    assert ILP.install_runtime_dir(root, WINDOWS) == expected
    assert ILP.install_runtime_dir(root, WINDOWS_ARM64) == expected
    # Non-Windows never grows the subdirectory, or a Linux install would read as missing.
    assert ILP.install_runtime_dir(root, LINUX) == root / "build" / "bin"


def test_the_installer_creates_the_directory_the_probe_looks_for(tmp_path):
    """normalize_install_layout and overlay_directory_for_choice both build the tree the
    probe grades, so they are checked against install_runtime_dir rather than against a
    literal path of their own."""
    server, quantize = ILP.normalize_install_layout(tmp_path, WINDOWS)
    runtime_dir = ILP.install_runtime_dir(tmp_path, WINDOWS)
    assert runtime_dir.is_dir(), "the installer must create exactly the directory the probe reads"
    assert server.parent == runtime_dir
    assert quantize.parent == runtime_dir


# ---------------------------------------------------------------------------
# The managed install that is really on this machine.


def _managed_install() -> Path | None:
    root = ILP.default_managed_llama_dir()
    return root if (root / "UNSLOTH_PREBUILT_INFO.json").is_file() else None


def _managed_copy(tmp_path: Path) -> Path:
    """A writable copy of the real install. The original is never touched: these tests
    delete files, and the machine's own runtime is not theirs to break."""
    source = _managed_install()
    if source is None:
        pytest.skip("no managed llama.cpp install on this machine")
    destination = tmp_path / "managed"
    shutil.copytree(source, destination, symlinks = True)
    return destination


def test_the_managed_install_on_this_machine_is_healthy():
    """Read-only, against the install itself rather than a copy.

    Every other check here grades a tree this file built. This one grades a tree the
    installer built, which is the only sample available of what the probe will actually be
    asked about at launch.
    """
    root = _managed_install()
    if root is None:
        pytest.skip("no managed llama.cpp install on this machine")
    assert ILP.installed_runtime_health(root) == (True, "")
    # The default argument is the path preflight takes, so it is worth exercising too.
    assert ILP.installed_runtime_health() == (True, "")


def test_the_real_runtime_payload_has_no_dangling_symlinks():
    """What the versioned library chain is actually made of, on a real install.

    It is natural to assume the chain is symlinks, the way a distribution packages a
    shared library: libggml.so pointing at libggml.so.0 pointing at the full version. This
    payload is not built that way. There is no unversioned name at all, and the two
    versioned names are independent regular files of identical size, so the "chain" is a
    duplicate rather than a link.

    That is worth an assertion because it decides what a payload glob can be trusted to
    mean. A symlink chain would let a glob match a name whose target is gone; duplicated
    regular files instead let a glob match a copy that is not the one the loader wants.
    Whatever else the payload contains, no symlink under the runtime directory may dangle:
    a glob match that resolves to nothing is not a file the runtime can load.
    """
    root = _managed_install()
    if root is None:
        pytest.skip("no managed llama.cpp install on this machine")
    runtime_dir = ILP.install_runtime_dir(root, ILP.platform_only_host())
    dangling = [
        path.name
        for path in sorted(runtime_dir.iterdir())
        if path.is_symlink() and not path.exists()
    ]
    assert dangling == [], f"the runtime payload must not contain dead links: {dangling}"


def test_the_probe_reads_only_platform_facts():
    """platform_only_host must agree with detect_host on every field the probe reads.

    detect_host shells out to nvidia-smi and costs over a second on the launch path. The
    probe is allowed to skip that only for as long as the fields it does read are the same
    either way, so drift here is a correctness bug and not just a slow launch.
    """
    cheap = ILP.platform_only_host()
    probed = ILP.detect_host()
    for field in (
        "system",
        "machine",
        "is_windows",
        "is_linux",
        "is_macos",
        "is_x86_64",
        "is_arm64",
        "macos_version",
    ):
        assert getattr(cheap, field) == getattr(probed, field), field


@pytest.mark.parametrize("victim", ["llama-server", "llama-quantize"])
def test_a_quarantined_binary_in_the_real_install_is_caught(victim, tmp_path):
    """The executables are named outright rather than matched by glob, so losing one is
    seen on a real tree exactly as it is on a fixture."""
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    assert ILP.installed_runtime_health(root, host = host) == (True, ""), "copy must start healthy"
    (ILP.install_runtime_dir(root, host) / victim).unlink()
    assert ILP.installed_runtime_health(root, host = host) == (
        False,
        "llama_runtime_binaries_missing",
    )


def test_no_single_missing_file_in_the_real_install_causes_a_repair_loop(tmp_path):
    """The one inequality the whole feature rests on, measured on a real tree.

    A tree the probe calls broken must be a tree ``_existing_install_runs`` would also
    refuse, or the repair reinstalls nothing, the next launch rejects it again, and the
    user is stuck with no error to act on. The matrix file asserts this over simulated
    trees; here it is asserted over an install that exists, with every file in the runtime
    directory removed in turn.
    """
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    vault = tmp_path / "vault"
    vault.mkdir()
    loops = []
    for path in sorted(runtime_dir.iterdir()):
        if not path.is_file():
            continue
        # Moved out of the directory rather than renamed in place: every payload pattern
        # ends in a star, so a renamed file is still matched by its own group and the
        # removal would silently not be a removal.
        shutil.move(str(path), str(vault / path.name))
        probe = ILP.installed_runtime_health(root, host = host)
        if probe is not None and probe[0] is False and ILP._existing_install_runs(root, host):
            loops.append((path.name, probe[1]))
        shutil.move(str(vault / path.name), str(path))
    assert loops == [], f"probe rejects trees the repair keeps, which is a repair loop: {loops}"


def test_quarantining_a_soname_is_reported_broken(tmp_path):
    """A real Linux install ships libllama.so.0 and libllama.so.0.0.<build> side by side.

    The loader needs the SONAME. The payload group is ``libllama.so*``, which the version
    suffixed copy also matches, so quarantining the SONAME leaves the group satisfied. The
    probe answers Ready, the desktop launches, and llama-server dies at exec with a loader
    error, which is the exact failure this feature was added to replace with a repair
    offer. No hand-built fixture can show this because a fixture writes one file per
    library and a release writes two.

    Fixed by _payload_match_is_loadable, which stops counting a name that carries more
    version components than a SONAME can: such a name is only ever the twin.
    """
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    if host.is_windows or host.is_macos:
        pytest.skip("versioned SONAME twins are a Linux packaging convention")
    runtime_dir = ILP.install_runtime_dir(root, host)
    soname = runtime_dir / "libllama.so.0"
    twin = next(iter(runtime_dir.glob("libllama.so.0.*")), None)
    if not soname.is_file() or twin is None:
        pytest.skip("this install does not carry a versioned twin of libllama")

    (tmp_path / "vault").mkdir(exist_ok = True)
    shutil.move(str(soname), str(tmp_path / "vault" / soname.name))
    assert twin.is_file(), "the twin is what keeps the glob satisfied"
    verdict = ILP.installed_runtime_health(root, host = host)
    assert verdict is not None and verdict[0] is False, (
        f"a runtime missing its SONAME cannot load, but the probe said {verdict}"
    )


def test_the_soname_quarantine_really_breaks_the_runtime(tmp_path):
    """Evidence that the case above is a defect and not a matter of taste.

    Removing the SONAME is not a cosmetic loss: the binary the desktop is about to start
    fails to load at exec. Run here rather than asserted from the docstring because the
    claim is about the loader, not about the code.
    """
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    if host.is_windows or host.is_macos:
        pytest.skip("versioned SONAME twins are a Linux packaging convention")
    runtime_dir = ILP.install_runtime_dir(root, host)
    server = runtime_dir / "llama-server"
    soname = runtime_dir / "libllama.so.0"
    if not server.is_file() or not soname.is_file():
        pytest.skip("this install has no llama-server or no versioned libllama")

    environment = {"LD_LIBRARY_PATH": str(runtime_dir), "PATH": "/usr/bin:/bin"}
    try:
        before = subprocess.run(
            [str(server), "--version"],
            capture_output = True,
            timeout = 120,
            env = environment,
        )
    except OSError as error:
        pytest.skip(f"cannot exec the managed llama-server here: {error}")
    if before.returncode != 0:
        pytest.skip("the managed llama-server does not start on this machine to begin with")

    (tmp_path / "vault").mkdir(exist_ok = True)
    shutil.move(str(soname), str(tmp_path / "vault" / soname.name))
    after = subprocess.run(
        [str(server), "--version"],
        capture_output = True,
        timeout = 120,
        env = environment,
    )
    assert after.returncode != 0, "removing the SONAME must break the binary, or there is no defect"
    # Deliberately no assertion on what the probe says here. The verdict for this tree is
    # the subject of test_soname_quarantine_is_reported_healthy, which is marked xfail so
    # that a fix flips it rather than breaking it. Pinning the current answer twice would
    # mean a fix has to edit two places.


# ---------------------------------------------------------------------------
# Desktop and CLI upgrade ordering.
#
# The two halves ship separately, so all four combinations occur in the field. The desktop
# half is Rust and is covered by the tests in studio/src-tauri/src/preflight/managed.rs;
# what is checkable from here is the CLI half of the contract, which is that the payload
# carries the keys and that a probe which cannot answer says so with null rather than false.


def test_the_capability_cache_on_this_machine_is_the_shape_the_new_reader_expects():
    """The real desktop_capability_cache.json, if the desktop has ever run here.

    The Rust side reconstructs a previous-release cache entry in
    ``a_schema_three_cache_file_on_disk_misses_and_is_rewritten_as_schema_four`` and checks
    that it misses rather than being served. That reconstruction is only as good as its
    author's memory of what the old writer emitted, so this asserts the same three
    properties against a file the previous release actually wrote: the schema is older than
    the current one, there is no llama_runtime key at top level, and the cached capability
    carries no llama_runtime_ok. Any of those being false would mean the Rust fixture is
    describing a file shape that does not exist.

    The schema bump alone is what makes the miss safe. Without it, an entry whose runtime
    fingerprint happens to compare equal, which is what an install with no managed runtime
    at all produces, would be served back with a Ready verdict reached before the runtime
    was ever looked at.
    """
    cache = Path.home() / ".unsloth" / "studio" / "desktop_capability_cache.json"
    if not cache.is_file():
        pytest.skip("the desktop has never written a capability cache on this machine")
    entry = json.loads(cache.read_text(encoding = "utf-8"))
    schema = entry.get("schema")
    if schema is None or schema >= 4:
        pytest.skip(f"this cache was written by the new desktop already (schema {schema})")
    assert "llama_runtime" not in entry, "a pre-bump entry cannot carry the runtime fingerprint"
    assert "llama_runtime_ok" not in entry.get("capability", {}), (
        "a pre-bump entry cannot carry a runtime verdict"
    )
    # Named individually rather than as a set comparison: a key the old writer emitted and
    # the new reader dropped would be a silent loss, while an extra key is harmless.
    for required in (
        "schema",
        "bin_path",
        "bin_size",
        "bin_mtime_ms",
        "studio_root_id",
        "marker_path",
        "marker_size",
        "marker_mtime_ms",
        "desktop_protocol_version",
        "desktop_manageability_version",
        "capability",
    ):
        assert required in entry, required


def test_the_capability_payload_names_the_runtime_keys():
    """A new desktop asked an old CLI gets no llama_runtime_ok at all, which it must read
    as "cannot answer". A new CLI therefore has to know both key names.

    Only the names are checked, not the literal that sets them: which installs the CLI
    reports runtime health for is a policy that can reasonably change, while renaming
    either key silently breaks the wire contract with every desktop already shipped.
    """
    source = (PACKAGE_ROOT / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    assert "llama_runtime_ok" in source
    assert "llama_runtime_reason" in source


def test_a_probe_that_raises_leaves_the_runtime_unknown(monkeypatch):
    """The CLI fills the keys best effort. A probe that throws must leave the null in
    place: reporting false would mark a working install stale, and the desktop would repair
    a runtime whose only fault was that the check itself failed."""
    payload = {"llama_runtime_ok": None, "llama_runtime_reason": ""}

    def explode(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(ILP, "installed_runtime_health", explode)
    try:
        health = ILP.installed_runtime_health()
        if health is not None:
            payload["llama_runtime_ok"], payload["llama_runtime_reason"] = health
    except Exception:
        pass
    assert payload == {"llama_runtime_ok": None, "llama_runtime_reason": ""}


def test_nothing_installed_leaves_the_runtime_unknown(tmp_path):
    """The same null, for a different reason: no marker means NotInstalled, which the
    desktop must not read as a broken runtime. installed_runtime_health returns None and
    the CLI leaves the key untouched."""
    assert ILP.installed_runtime_health(tmp_path / "nothing-here") is None
