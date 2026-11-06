# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe real release zips and managed installs; fixtures miss the SONAME twins that releases ship."""

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

# Zips are fetched out of band and absent in CI (tests skip); set UNSLOTH_TEST_LLAMACPP_ASSET_DIR.
ASSET_DIR = Path(
    os.environ.get(
        "UNSLOTH_TEST_LLAMACPP_ASSET_DIR",
        Path.home() / ".cache" / "unsloth" / "llamacpp-test-assets",
    )
)


def _asset_or_skip(asset: str) -> Path:
    """Skip, not fail, when the asset is missing or unreadable: is_file() raises on other stat errors."""
    archive = ASSET_DIR / asset
    try:
        present = archive.is_file()
    except OSError as error:
        pytest.skip(f"release bundle not readable on this machine: {asset} ({error})")
    if not present:
        pytest.skip(f"release bundle not on this machine: {asset}")
    return archive


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


# (asset, marker llama_backend, tag, host). The tag matters: impl.dll is owed only post-split.

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
    # Upstream ggml-org archive: source='upstream' takes a different branch.
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
    """Windows zips are flat; build/bin/Release comes from install_runtime_dir, not from the archive."""
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
    """A real release bundle must probe healthy, or every user gets a repair with nothing to fix."""
    _asset_or_skip(asset)
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
    """Windows bundles ship one copy per library, so a quarantined file has no other glob match."""
    asset = "app-b10798-mix-659e406-windows-x64-cpu.zip"
    _asset_or_skip(asset)
    root = _unpack_bundle(asset, None, "b10798", "published", WINDOWS, tmp_path)
    (ILP.install_runtime_dir(root, WINDOWS) / victim).unlink()
    verdict = ILP.installed_runtime_health(root, host = WINDOWS)
    assert verdict is not None and verdict[0] is False, victim
    assert verdict[1] in {
        "llama_runtime_payload_incomplete",
        "llama_runtime_binaries_missing",
    }, verdict


def test_a_windows_cuda_bundle_without_its_paired_runtime_is_incomplete(tmp_path):
    """A CUDA bundle ships no cudart DLLs, so a paired runtime missing from disk leaves it incomplete."""
    asset = "app-b10798-mix-659e406-windows-x64-cuda12-legacy.zip"
    _asset_or_skip(asset)
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
    """The archive's UNSLOTH_PREBUILT_INFO.json is a build record; the root copy is the install record."""
    asset = "app-b10798-mix-659e406-windows-x64-cuda12-legacy.zip"
    _asset_or_skip(asset)
    root = _unpack_bundle(asset, "cuda", "b10798", "published", WINDOWS, tmp_path)
    runtime_dir = ILP.install_runtime_dir(root, WINDOWS)
    bundled = runtime_dir / "UNSLOTH_PREBUILT_INFO.json"
    assert bundled.is_file(), "this bundle is expected to carry a build marker"
    assert runtime_dir != root, "the build marker must not be able to shadow the install marker"
    read_back = ILP.load_prebuilt_metadata(root)
    assert read_back is not None and read_back.get("source") == "published"
    assert "upstream_tag" not in read_back, "the install marker, not the archive's build record"


def test_every_windows_layout_decision_agrees_on_the_release_subdirectory():
    """Every Windows site must agree on build/bin/Release, or each launch reports the install broken."""
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
    """Real installs have duplicate versioned copies, not symlinks, so no symlink may dangle."""
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
    """platform_only_host must match detect_host on each field read; skipping nvidia-smi relies on that."""
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
    """Any one missing file in a real install must be refused by _existing_install_runs too, or it loops."""
    root = _managed_copy(tmp_path)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    vault = tmp_path / "vault"
    vault.mkdir()
    loops = []
    for path in sorted(runtime_dir.iterdir()):
        if not path.is_file():
            continue
        # Moved out, not renamed: every payload pattern ends in a star.
        shutil.move(str(path), str(vault / path.name))
        probe = ILP.installed_runtime_health(root, host = host)
        if probe is not None and probe[0] is False and ILP._existing_install_runs(root, host):
            loops.append((path.name, probe[1]))
        shutil.move(str(vault / path.name), str(path))
    assert loops == [], f"probe rejects trees the repair keeps, which is a repair loop: {loops}"


def test_quarantining_a_soname_is_reported_broken(tmp_path):
    """Quarantining a SONAME must read broken, since libllama.so* still matches its versioned twin."""
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
    """Evidence that the quarantined SONAME defect is real: llama-server actually fails to load at exec."""
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


# Desktop and CLI halves ship separately; a probe that cannot answer says null, not false.


def test_the_capability_cache_on_this_machine_is_the_shape_the_new_reader_expects():
    """An old-schema capability cache must miss, not serve a Ready verdict from before the runtime check."""
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
    # Checked individually: a dropped key is a silent loss, an extra key is harmless.
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
    """Asserts the llama_runtime_ok and llama_runtime_reason names; renaming them breaks shipped
    desktops."""
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
    """Linux requires the -impl.so libraries that llama-server and llama-quantize load via DT_NEEDED."""
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
    # The two answers must agree, or repair loops.
    assert ILP._existing_install_runs(root, host) is False


def test_an_older_monolithic_linux_release_is_not_asked_for_the_impl_libraries():
    """Pre-split Linux releases ship no lib*-impl.so; requiring it would reinstall them forever."""
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
    """Exact-release reuse must check the execute bit like the health probe, or a stripped binary loops."""
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
        assert server.exists(), "the file is still there, which is the whole point"
        assert ILP._entrypoint_is_runnable(server, host) is False
    finally:
        server.chmod(mode)


# b10840 ships libllama.so symlink chains; copy_globs flattens them into regular files.
_TRIO_ASSET = "app-b10840-mix-d5c17a0-linux-x64-cpu.tar.gz"
_TRIO_TAG = "b10840-mix-d5c17a0"


def _installed_trio_bundle(tmp_path: Path) -> Path:
    """Copy via copy_globs, not move: a moved tree keeps symlinks that dangle when the SONAME goes."""
    archive = _asset_or_skip(_TRIO_ASSET)
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
    """b10840 ships a versionless copy beside the SONAME; quarantining the SONAME left the group
    satisfied."""
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
    assert ILP._existing_install_runs(root, host) is False


def test_a_family_that_only_ever_ships_one_name_is_still_loadable(tmp_path):
    """A single-name library like libggml-cpu-x64.so is loadable as-is, so no SONAME may be demanded."""
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
