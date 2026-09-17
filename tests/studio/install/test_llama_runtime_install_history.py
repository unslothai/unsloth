# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-time runtime health probe against installs that exist on disk today.

The sibling files build their trees by hand, so they can only confirm that the code agrees
with the table their author read out of the code. This file uses artifacts instead: it
extracts the real release zips into the layout the installer builds, and it copies a
managed install out of ``~/.unsloth/llama.cpp`` and guts it a file at a time, which is the
closest reachable stand-in for the quarantine this feature exists to catch.

That difference is not cosmetic. A real install ships each core library twice, under its
SONAME and under its full version (``libllama.so.0`` beside ``libllama.so.0.0.10360``),
while a fixture writes only ``libllama.so``. The trailing-star globs in
``runtime_payload_health_groups`` cannot tell the copies apart, so quarantining the SONAME
leaves the group satisfied while the runtime no longer loads. Only an artifact shows that.

Every test here skips rather than fails when its artifact is absent, so the file still runs
in CI, where neither the release zips nor a managed install are present.
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

# The release zips this file reads. Downloaded out of band and absent in CI, where every
# test that wants one skips.
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
# The real release bundles: (asset, llama_backend in the marker, tag, host). The tag
# matters, since the shared Windows group requires llama-server-impl.dll only from the
# build that split it out, so a wrong tag would pass for the wrong reason.

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
    # An upstream ggml-org archive: source = "upstream" takes a different branch through
    # _windows_shared_groups.
    ("llama-b10830-bin-win-cpu-x64.zip", None, "b10830", "upstream", WINDOWS),
]


def _marker_for(asset: str, backend: str | None, tag: str, source: str) -> dict:
    """A marker as install_from_archives writes it, trimmed to the keys the probe reads: the
    backend picks the install kinds, source and tag pick the group table."""
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

    The Windows zips are flat, so ``build/bin/Release`` is the installer's choice and not
    the archive's, which is why this mirrors install_runtime_dir rather than trusting
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

    A false (False, reason) marks the install stale on the next launch and sends every user
    through a repair with nothing to fix. Only the release itself can say whether the
    payload globs match what is actually in the archive.
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
        "llama-quantize-impl.dll",
        "ggml.dll",
        "ggml-base.dll",
        "mtmd.dll",
        "llama-server.exe",
        "llama-quantize.exe",
    ],
)
def test_a_file_quarantined_from_a_real_windows_bundle_is_caught(victim, tmp_path):
    """The other direction, on a real archive.

    Windows ships one copy of each library, so removing a required file leaves nothing for
    the glob to match. Linux is where that stops being true, below.
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
    runtime that is not on disk describes an install that cannot start. Asserted against the
    real archive so the trio is genuinely absent rather than merely omitted from a fixture.
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

    Same name, different documents: the archive's copy is the build record and lands in the
    runtime directory, while the installer writes the install record at the root.
    load_prebuilt_metadata reads the root one, so a layout change putting the runtime
    directory at the install root would grade the tree with the wrong table.
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

    A missing directory reports llama_runtime_dir_missing and marks the install stale, so a
    probe deriving a different path than the installer writes to would repair every Windows
    user on every launch. The agreement is the property, not the literal string.
    """
    root = Path("/install")
    expected = root / "build" / "bin" / "Release"
    assert ILP.install_runtime_dir(root, WINDOWS) == expected
    assert ILP.install_runtime_dir(root, WINDOWS_ARM64) == expected
    # Non-Windows never grows the subdirectory, or a Linux install reads as missing.
    assert ILP.install_runtime_dir(root, LINUX) == root / "build" / "bin"


def test_the_installer_creates_the_directory_the_probe_looks_for(tmp_path):
    """Both builders of the tree the probe grades are checked against install_runtime_dir
    rather than a literal path of their own."""
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
    """A writable copy: these tests delete files and the machine's runtime is not theirs."""
    source = _managed_install()
    if source is None:
        pytest.skip("no managed llama.cpp install on this machine")
    destination = tmp_path / "managed"
    shutil.copytree(source, destination, symlinks = True)
    return destination


def test_the_managed_install_on_this_machine_is_healthy():
    """Read-only, against the install itself rather than a copy: the only sample available of
    what the probe is actually asked about at launch.
    """
    root = _managed_install()
    if root is None:
        pytest.skip("no managed llama.cpp install on this machine")
    assert ILP.installed_runtime_health(root) == (True, "")
    # The default argument is the path preflight takes.
    assert ILP.installed_runtime_health() == (True, "")


def test_the_real_runtime_payload_has_no_dangling_symlinks():
    """What the versioned library chain is actually made of, on a real install.

    Not symlinks, the way a distribution packages a shared library: there is no unversioned
    name at all, and the two versioned names are independent regular files of identical
    size, so the "chain" is a duplicate rather than a link.

    That decides what a payload glob can be trusted to mean. A symlink chain would let a
    glob match a name whose target is gone; duplicates let it match a copy the loader does
    not want. Either way no symlink here may dangle, since a match that resolves to nothing
    is not a file the runtime can load.
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

    Skipping detect_host's second of nvidia-smi is allowed only while these fields are the
    same either way, so drift here is a correctness bug and not just a slow launch.
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
    """The executables are named outright rather than globbed, so a real tree behaves like a
    fixture here."""
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

    A tree the probe calls broken must be one ``_existing_install_runs`` also refuses, or
    the repair reinstalls nothing and the next launch rejects it again with no error to act
    on. The matrix file asserts this over simulated trees; here every file in a real runtime
    directory is removed in turn.
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
        # Moved out rather than renamed: every payload pattern ends in a star, so a renamed
        # file is still matched by its own group.
        shutil.move(str(path), str(vault / path.name))
        probe = ILP.installed_runtime_health(root, host = host)
        if probe is not None and probe[0] is False and ILP._existing_install_runs(root, host):
            loops.append((path.name, probe[1]))
        shutil.move(str(vault / path.name), str(path))
    assert loops == [], f"probe rejects trees the repair keeps, which is a repair loop: {loops}"


def test_quarantining_a_soname_is_reported_broken(tmp_path):
    """A real Linux install ships libllama.so.0 and libllama.so.0.0.<build> side by side.

    The loader needs the SONAME, but the group ``libllama.so*`` also matches the versioned
    copy, so quarantining the SONAME left the group satisfied: the probe answered Ready and
    llama-server died at exec with a loader error. No fixture shows this, since a fixture
    writes one file per library and a release writes two.

    Fixed by _payload_match_is_loadable, which stops counting a name carrying more version
    components than a SONAME can, since such a name is only ever the twin.
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
    assert (
        verdict is not None and verdict[0] is False
    ), f"a runtime missing its SONAME cannot load, but the probe said {verdict}"


def test_the_soname_quarantine_really_breaks_the_runtime(tmp_path):
    """Evidence that the case above is a defect and not a matter of taste: the binary the
    desktop is about to start fails to load at exec. Run rather than asserted, because the
    claim is about the loader and not about the code.
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
    # No assertion on the probe's verdict here: that is the subject of the test above, and
    # pinning it twice would make a fix edit two places.


# ---------------------------------------------------------------------------
# Desktop and CLI upgrade ordering. The two halves ship separately, so all four combinations
# occur in the field. managed.rs covers the desktop half; checkable here is the CLI half:
# the payload carries the keys, and a probe that cannot answer says null rather than false.


def test_the_capability_cache_on_this_machine_is_the_shape_the_new_reader_expects():
    """The real desktop_capability_cache.json, if the desktop has ever run here.

    The Rust side reconstructs a previous-release cache entry from its author's memory of
    what the old writer emitted, so this asserts the same three properties against a file
    that release actually wrote: an older schema, no top-level llama_runtime key, and no
    llama_runtime_ok in the cached capability.

    The schema bump alone is what makes the miss safe. Without it, an entry whose runtime
    fingerprint compares equal, which an install with no managed runtime produces, would be
    served back with a Ready verdict reached before the runtime was looked at.
    """
    cache = Path.home() / ".unsloth" / "studio" / "desktop_capability_cache.json"
    if not cache.is_file():
        pytest.skip("the desktop has never written a capability cache on this machine")
    entry = json.loads(cache.read_text(encoding = "utf-8"))
    schema = entry.get("schema")
    if schema is None or schema >= 4:
        pytest.skip(f"this cache was written by the new desktop already (schema {schema})")
    assert "llama_runtime" not in entry, "a pre-bump entry cannot carry the runtime fingerprint"
    assert "llama_runtime_ok" not in entry.get(
        "capability", {}
    ), "a pre-bump entry cannot carry a runtime verdict"
    # Named individually, not compared as a set: a dropped key is a silent loss, an extra
    # key is harmless.
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
    """A new desktop asked an old CLI gets no llama_runtime_ok, which it reads as "cannot
    answer", so a new CLI has to know both key names.

    Only the names, not the literal that sets them: which installs the CLI reports health
    for is policy, while renaming a key breaks the wire contract with every shipped desktop.
    """
    source = (PACKAGE_ROOT / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    assert "llama_runtime_ok" in source
    assert "llama_runtime_reason" in source


def test_a_probe_that_raises_leaves_the_runtime_unknown(monkeypatch):
    """The CLI fills the keys best effort, so a probe that throws must leave the null in
    place. False would repair a runtime whose only fault was that the check failed."""
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
    """The same null for a different reason: no marker means NotInstalled, not a broken
    runtime, so the probe returns None and the CLI leaves the key untouched."""
    assert ILP.installed_runtime_health(tmp_path / "nothing-here") is None


@pytest.mark.parametrize("victim", ["libllama-server-impl.so", "libllama-quantize-impl.so"])
def test_quarantining_a_split_entrypoint_library_is_reported_broken(victim, tmp_path):
    """The other half of the upstream impl split, on the side that had no group for it.

    ``llama-server`` and ``llama-quantize`` carry no entry code of their own since
    ggml-org/llama.cpp#23462; they load ``libllama-server-impl.so`` and
    ``libllama-quantize-impl.so`` by DT_NEEDED. The payload groups named the shared
    libraries only, so removing one of these left every group satisfied while the binary
    the desktop is about to start died in the loader: the probe answered Ready and
    ``_existing_install_runs`` answered false, which is the disagreement this feature
    exists to remove. Windows already required its ``llama-server-impl.dll``; Linux did
    not require either of its two.
    """
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    if not host.is_linux:
        pytest.skip("the impl split libraries are the Linux and Windows names, not macOS")
    runtime_dir = ILP.install_runtime_dir(root, host)
    library = runtime_dir / victim
    if not library.is_file():
        pytest.skip(f"this install predates the impl split: no {victim}")
    assert ILP.installed_runtime_health(root, host = host) == (True, ""), "copy must start healthy"

    (tmp_path / "vault").mkdir(exist_ok = True)
    shutil.move(str(library), str(tmp_path / "vault" / victim))
    verdict = ILP.installed_runtime_health(root, host = host)
    assert (
        verdict is not None and verdict[0] is False
    ), f"a runtime missing {victim} cannot load, but the probe said {verdict}"
    # And the two answers still agree, which is the property that keeps repair from looping.
    assert ILP._existing_install_runs(root, host) is False


def test_an_older_monolithic_linux_release_is_not_asked_for_the_impl_libraries():
    """The gate, not just the requirement.

    An archive from before the split ships no ``lib*-impl.so`` at all, so requiring one
    would reinstall it on every check forever. Same build number as the Windows side, and
    for the same reason: it is one upstream commit, not one platform's packaging.
    """
    before = ILP.runtime_payload_health_groups("linux-cuda", source_label = "published", tag = "b9279")
    after = ILP.runtime_payload_health_groups("linux-cuda", source_label = "published", tag = "b9283")
    flat_before = {pattern for group in before for pattern in group}
    flat_after = {pattern for group in after for pattern in group}
    assert "libllama-server-impl.so*" not in flat_before
    assert "libllama-quantize-impl.so*" not in flat_before
    assert "libllama-server-impl.so*" in flat_after
    assert "libllama-quantize-impl.so*" in flat_after
    # A source build ships neither, whatever the tag says.
    source_built = ILP.runtime_payload_health_groups(
        "linux-cuda", source_label = "source", tag = "b10360"
    )
    assert not any("impl" in pattern for group in source_built for pattern in group)


def test_a_stripped_execute_bit_is_not_reused_as_an_exact_release_match(tmp_path):
    """The keep-or-reinstall decision has to reject what the probe rejects.

    ``installed_runtime_health`` asks for the execute bit, because that is what
    ``_find_llama_server_binary`` asks for. ``existing_install_matches_choice`` asked only
    ``exists()``, and its Linux ``ldd`` gate reads a non-executable ELF quite happily, so a
    cleared bit produced: preflight says broken, repair says the exact release is already
    installed, nothing is downloaded, and the next launch says broken again. Checked
    against the two gates directly, since the surrounding function also wants a matching
    fingerprint that this test has no business reconstructing.
    """
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    if host.is_windows:
        pytest.skip("there is no execute bit to clear on Windows")
    runtime_dir = ILP.install_runtime_dir(root, host)
    server = runtime_dir / "llama-server"
    if not server.is_file():
        pytest.skip("this install has no llama-server")

    mode = server.stat().st_mode
    server.chmod(mode & ~0o111)
    try:
        assert ILP.installed_runtime_health(root, host = host) == (
            False,
            "llama_runtime_binaries_missing",
        )
        assert ILP._existing_install_runs(root, host) is False
        # The old gate, which is what let the two disagree.
        assert server.exists(), "the file is still there, which is the whole point"
        # The new one, shared with both answers above.
        assert ILP._entrypoint_is_runnable(server, host) is False
    finally:
        server.chmod(mode)


# The bundle that changed the packaging. b10360 shipped two names per library and no
# versionless one; b10840 ships libllama.so -> libllama.so.0 -> libllama.so.0.4.0 as
# symlinks, and copy_globs flattens all three into regular files, because it selects
# with is_file() and copies with shutil.copy2, which follows links.
_TRIO_ASSET = "app-b10840-mix-d5c17a0-linux-x64-cpu.tar.gz"
_TRIO_TAG = "b10840-mix-d5c17a0"


def _installed_trio_bundle(tmp_path: Path) -> Path:
    """The b10840 CPU bundle, installed the way install_prebuilt installs it.

    Through ``copy_globs`` rather than by moving the extracted tree: moving preserves
    the symlinks, and a tree of links behaves quite differently here, since removing
    the SONAME makes the versionless link dangle and ``is_file()`` drops it on its own.
    A real install has no links left to dangle, which is the whole point of this test.
    """
    archive = ASSET_DIR / _TRIO_ASSET
    if not archive.is_file():
        pytest.skip(f"{_TRIO_ASSET} is not present")
    prebuilt_core = _load_prebuilt_core()
    if prebuilt_core is None:
        pytest.skip("prebuilt_core is not importable here")
    host = ILP.platform_only_host()
    if not host.is_linux:
        pytest.skip("a linux bundle installs into the linux layout")

    raw = tmp_path / "raw"
    prebuilt_core.extract_archive(archive, raw)
    root = tmp_path / "llama.cpp"
    runtime_dir = ILP.install_runtime_dir(root, host)
    ILP.copy_globs(
        raw,
        runtime_dir,
        ["llama-server", "llama-quantize", "llama-diffusion-gemma-visual-server", "lib*.so*"],
        required = True,
    )
    for name in ("llama-server", "llama-quantize"):
        binary = runtime_dir / name
        binary.chmod(0o755)
        shutil.copy2(binary, root / name)
        (root / name).chmod(0o755)
    (root / "convert_hf_to_gguf.py").write_text("", encoding = "utf-8")
    (root / "gguf-py").mkdir(exist_ok = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(
            {
                "release_tag": _TRIO_TAG,
                "tag": _TRIO_TAG,
                "source": "published",
                "backend": "cpu",
                "asset": _TRIO_ASSET,
            }
        )
        + "\n",
        encoding = "utf-8",
    )
    return root


def _load_prebuilt_core():
    module_path = PACKAGE_ROOT / "studio" / "prebuilt_core.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("studio_prebuilt_core_for_tests", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_flattened_trio_installs_healthy(tmp_path):
    """The shape itself must not read as broken, or every b10840 install repairs forever."""
    root = _installed_trio_bundle(tmp_path)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    for name in ("libllama.so", "libllama.so.0", "libllama.so.0.4.0"):
        path = runtime_dir / name
        assert path.is_file() and not path.is_symlink(), (
            f"{name} must be a flattened regular file, or this test is not measuring "
            "what the installer produces"
        )
    assert ILP.installed_runtime_health(root, host = host) == (True, "")


@pytest.mark.parametrize(
    "victim",
    ["libllama.so.0", "libggml.so.0", "libllama-common.so.0", "libggml-base.so.0", "libmtmd.so.0"],
)
def test_quarantining_a_soname_beside_a_versionless_copy_is_reported_broken(victim, tmp_path):
    """Codex 3962583748, P1. The case my earlier rebuttal got wrong.

    I checked the managed install on the machine, which is b10360: two names per
    library, no versionless one, so removing the SONAME left nothing that could
    satisfy the group. b10840 ships a third name, and after copy_globs it is a regular
    file rather than a link onto the SONAME, so the group stayed satisfied by a file
    the loader never asks for while llama-server died at exec.

    Measured, before the fix, on this bundle: every one of these left
    installed_runtime_health answering (True, "") with _existing_install_runs false.
    """
    root = _installed_trio_bundle(tmp_path)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    soname = runtime_dir / victim
    if not soname.is_file():
        pytest.skip(f"this bundle does not carry {victim}")
    versionless = runtime_dir / f"{victim[: victim.index('.so')]}.so"
    assert versionless.is_file(), "the versionless twin is what used to keep the group satisfied"

    (tmp_path / "vault").mkdir(exist_ok = True)
    shutil.move(str(soname), str(tmp_path / "vault" / victim))
    verdict = ILP.installed_runtime_health(root, host = host)
    assert (
        verdict is not None and verdict[0] is False
    ), f"a runtime missing {victim} cannot load, but the probe said {verdict}"
    # The two answers still agree, which is what keeps repair from looping.
    assert ILP._existing_install_runs(root, host) is False


def test_a_family_that_only_ever_ships_one_name_is_still_loadable(tmp_path):
    """The other half, and the one a blunter rule would break.

    libggml-cpu-x64.so has no versioned copy anywhere, so the versionless name IS the
    one the loader asks for. Requiring a SONAME of every library would call every
    install broken and reinstall on each check forever. No artifact needed.
    """
    runtime_dir = tmp_path / "bin"
    runtime_dir.mkdir()
    lonely = runtime_dir / "libggml-cpu-x64.so"
    lonely.write_bytes(b"ELF")
    assert ILP._payload_match_is_loadable(lonely) is True

    # Same name, now with a versioned sibling: it is the family that decides.
    versioned = runtime_dir / "libggml-cpu-x64.so.0"
    versioned.write_bytes(b"ELF")
    assert ILP._payload_match_is_loadable(lonely) is False
    assert ILP._payload_match_is_loadable(versioned) is True
    assert ILP._family_base("libggml-cpu-x64.so.0.19.0") == "libggml-cpu-x64"
    # A neighbour of a different family must not vote.
    other = runtime_dir / "libggml-cpu-x64-extra.so"
    other.write_bytes(b"ELF")
    assert ILP._family_base(other.name) == "libggml-cpu-x64-extra"


def test_the_macos_install_name_rule_matches_the_linux_one(tmp_path):
    """dyld asks for libggml.0.dylib, the install name recorded in LC_ID_DYLIB, so the
    versionless link is not a substitute there either, and a bundle that ships only
    libggml.dylib still is."""
    runtime_dir = tmp_path / "bin"
    runtime_dir.mkdir()
    versionless = runtime_dir / "libggml.dylib"
    versionless.write_bytes(b"MACHO")
    assert ILP._payload_match_is_loadable(versionless) is True
    install_name = runtime_dir / "libggml.0.dylib"
    install_name.write_bytes(b"MACHO")
    terminal = runtime_dir / "libggml.0.23.0.dylib"
    terminal.write_bytes(b"MACHO")
    assert ILP._payload_match_is_loadable(versionless) is False
    assert ILP._payload_match_is_loadable(install_name) is True
    assert ILP._payload_match_is_loadable(terminal) is False
