# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-time health probe for the managed llama.cpp runtime.

Quarantine takes files out of a tree that is otherwise present and whose marker still
says installed. Preflight decided staleness on the managed Python alone, so the desktop
launched and the missing DLL surfaced at model load as an unrelated-looking error.
``installed_runtime_health`` is what the capability payload answers with instead.

The payload tables are covered by ``test_keep_install_backcompat_9979``; tested here is
the composition around them: nothing installed is not a broken install, a missing runtime
directory differs from a gutted one, and the probe pays for no GPU detection.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)


def _installed(tmp_path: Path, *, binaries: bool = False) -> Path:
    """An install root with a marker and a runtime directory, and nothing else."""
    root = tmp_path / "llama.cpp"
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    runtime_dir.mkdir(parents = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b10830-mix-d5c17a0", "tag": "b10830"}) + "\n",
        encoding = "utf-8",
    )
    if binaries:
        ext = ".exe" if host.is_windows else ""
        for name in ("server", "quantize"):
            binary = runtime_dir / f"llama-{name}{ext}"
            binary.write_text("", encoding = "utf-8")
            os.chmod(binary, 0o755)
    return root


def test_no_marker_is_not_installed_rather_than_broken(tmp_path):
    """None, not (False, ...): a user who never installed a runtime must not be repaired."""
    (tmp_path / "llama.cpp").mkdir()
    assert ILP.installed_runtime_health(tmp_path / "llama.cpp") is None
    assert ILP.installed_runtime_health(tmp_path / "nothing-here") is None


def test_a_marker_without_its_runtime_directory_is_broken(tmp_path):
    """The whole tree can go, not only files inside it, leaving the marker orphaned."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{}\n", encoding = "utf-8")
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_dir_missing")


def test_an_empty_runtime_directory_is_broken_not_healthy(tmp_path):
    """The shape a quarantine leaves behind, which used to pass preflight."""
    root = _installed(tmp_path)
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_payload_incomplete")


def test_the_verdict_is_delegated_to_the_payload_tables(tmp_path, monkeypatch):
    """One payload decider: duplicating the required-file list would let the launch probe
    and the keep-install path drift apart."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    assert ILP.installed_runtime_health(root) == (True, "")


def test_the_default_root_is_the_managed_install_dir(tmp_path, monkeypatch):
    """The CLI calls this with no argument, so the UNSLOTH_LLAMA_CPP_PATH override has to
    keep working through it."""
    root = _installed(tmp_path)
    monkeypatch.setattr(ILP, "default_managed_llama_dir", lambda: root)
    assert ILP.installed_runtime_health() == (False, "llama_runtime_payload_incomplete")


def test_platform_only_host_agrees_with_a_detected_host_on_the_platform_facts():
    """It stands in for detect_host() in the payload checks, which read these booleans and
    nothing else. Disagreement would grade the tree for the wrong operating system."""
    cheap = ILP.platform_only_host()
    detected = ILP.detect_host()
    for field in ("system", "is_windows", "is_linux", "is_macos", "is_x86_64", "is_arm64"):
        assert getattr(cheap, field) == getattr(detected, field), field
    assert cheap.machine == detected.machine.lower()


def test_platform_only_host_does_not_probe_for_gpus(monkeypatch):
    """detect_host() shells out to nvidia-smi and costs over a second; a probe that only
    looks for missing files must not."""

    def refuse(*args, **kwargs):
        raise AssertionError("platform_only_host must not run a subprocess")

    monkeypatch.setattr(ILP.subprocess, "run", refuse)
    monkeypatch.setattr(ILP.shutil, "which", refuse)
    host = ILP.platform_only_host()
    assert host.nvidia_smi is None
    assert host.has_usable_nvidia is False
    assert host.compute_caps == []


@pytest.mark.parametrize("broken", [OSError("denied"), RuntimeError("boom")])
def test_a_probe_that_raises_is_not_swallowed_here(tmp_path, monkeypatch, broken):
    """The best-effort handling lives in the CLI command, which turns a failure into a null
    capability. Swallowing it here too would hide a real payload-table bug."""
    root = _installed(tmp_path)

    def raise_it(*_):
        raise broken

    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", raise_it)
    with pytest.raises(type(broken)):
        ILP.installed_runtime_health(root)


def test_a_quarantined_llama_server_is_caught_even_with_a_complete_payload(tmp_path, monkeypatch):
    """The payload groups name libraries only, so a missing server binary needs its own
    check on Linux and macOS."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    host = ILP.platform_only_host()
    ext = ".exe" if host.is_windows else ""
    (ILP.install_runtime_dir(root, host) / f"llama-server{ext}").unlink()
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_binaries_missing")


def test_llama_quantize_is_required_because_the_setup_scripts_require_it(tmp_path, monkeypatch):
    """_existing_install_runs demands both, so demanding both here keeps this call no
    stricter than the repair that answers it."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    host = ILP.platform_only_host()
    ext = ".exe" if host.is_windows else ""
    (ILP.install_runtime_dir(root, host) / f"llama-quantize{ext}").unlink()
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_binaries_missing")


def test_an_explicit_host_overrides_the_detected_platform(tmp_path):
    """The simulation matrices grade a tree for a platform this machine is not."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin" / "Release").mkdir(parents = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{}\n", encoding = "utf-8")
    windows = ILP.detect_host()
    windows = type(windows)(
        **{
            **windows.__dict__,
            "system": "Windows",
            "is_windows": True,
            "is_linux": False,
            "is_macos": False,
        }
    )
    # Windows looks in build/bin/Release, which exists; a Linux host looks in build/bin.
    assert ILP.installed_runtime_health(root, host = windows)[1] != "llama_runtime_dir_missing"


def test_a_marker_that_exists_but_does_not_parse_is_still_graded(tmp_path):
    """An absent marker is a runtime nobody installed; a present but unreadable one is a real
    tree whose write was interrupted. Short-circuiting the second to None left preflight Ready
    with a library missing, the exact failure this probe catches. The keep path grades such a
    marker as an unknown backend anyway, so grading it here stays no stricter."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text('{"release_tag": "b108', encoding = "utf-8")
    assert ILP.load_prebuilt_metadata(root) is None
    # The payload is empty, so the tree is broken and must be offered for repair.
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_payload_incomplete")


def test_an_unparsable_marker_over_a_complete_tree_is_still_healthy(tmp_path, monkeypatch):
    """The other half, which keeps the loop shut: _existing_install_runs keeps a tree with an
    unreadable marker, so calling it broken would repair, keep, and repair again every launch."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text("not json", encoding = "utf-8")
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    assert ILP.installed_runtime_health(root) == (True, "")


def test_an_absent_marker_is_still_not_installed(tmp_path):
    """The distinction the fix turns on, asserted directly."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").unlink()
    assert ILP.installed_runtime_health(root) is None


def _windows_host():
    """A Windows HostInfo, so these run on the CI machines we actually have."""
    host = ILP.platform_only_host()
    return type(host)(
        **{
            **host.__dict__,
            "system": "Windows",
            "machine": "amd64",
            "is_windows": True,
            "is_linux": False,
            "is_macos": False,
            "is_x86_64": True,
            "is_arm64": False,
        }
    )


"""What a published Windows bundle ships, checked against
app-b10798-mix-659e406-windows-x64-cpu.zip."""
_PUBLISHED_WINDOWS_PAYLOAD = (
    "llama.dll",
    "llama-common.dll",
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama-quantize.exe",
    "llama-quantize-impl.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-haswell.dll",
    "mtmd.dll",
)


def _windows_tree(tmp_path: Path, names, *, marker: str) -> Path:
    root = tmp_path / "llama.cpp"
    host = _windows_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    runtime_dir.mkdir(parents = True)
    for name in names:
        binary = runtime_dir / name
        binary.write_text("", encoding = "utf-8")
        os.chmod(binary, 0o755)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(marker, encoding = "utf-8")
    return root


def test_an_unparseable_marker_over_a_prebuilt_tree_still_owes_the_shared_payload(tmp_path):
    """Codex 3962938529, P2. An unreadable marker names no source, and the source-gated
    groups were dropped with it, so on Windows llama.dll stood in for the whole payload: an
    interrupted marker plus a quarantined ggml-base.dll answered healthy and the runtime
    launched into the loader error this probe exists to pre-empt. The source is read off the
    tree instead, from a name only a published bundle ships."""
    host = _windows_host()
    root = _windows_tree(tmp_path, _PUBLISHED_WINDOWS_PAYLOAD, marker = '{"release_tag": "b108')
    assert ILP.load_prebuilt_metadata(root) is None
    assert ILP.installed_runtime_health(root, host = host) == (True, "")

    quarantined = ILP.install_runtime_dir(root, host) / "ggml-base.dll"
    quarantined.rename(quarantined.with_suffix(".dll.quarantine"))
    assert ILP.installed_runtime_health(root, host = host) == (
        False,
        "llama_runtime_payload_incomplete",
    )


def test_an_unparseable_marker_over_a_source_build_stays_lenient(tmp_path):
    """The other half of the same rule, and the reason it reads the tree rather than assuming
    published: setup.ps1 links statically and ships none of those names, so requiring them
    would fail health on a tree _existing_install_runs keeps, and the repair would run every
    launch with nothing to change."""
    host = _windows_host()
    root = _windows_tree(
        tmp_path,
        ("llama.dll", "llama-server.exe", "llama-quantize.exe"),
        marker = "not json",
    )
    assert ILP.installed_runtime_health(root, host = host) == (True, "")
    assert ILP._tree_looks_prebuilt(root, host) is False


def test_the_windows_quantize_implementation_is_required_alongside_the_server_one(tmp_path):
    """Codex 3962938556, P2. The upstream split gives llama-quantize.exe its own -impl library
    exactly as it gives llama-server.exe one, and a b10798 bundle ships both, so requiring only
    the server's left a quarantined llama-quantize-impl.dll reading as healthy while
    quantization could not start."""
    host = _windows_host()
    root = _windows_tree(
        tmp_path,
        _PUBLISHED_WINDOWS_PAYLOAD,
        marker = json.dumps({"source": "published", "tag": "b10798"}),
    )
    assert ILP.installed_runtime_health(root, host = host) == (True, "")
    (ILP.install_runtime_dir(root, host) / "llama-quantize-impl.dll").unlink()
    assert ILP.installed_runtime_health(root, host = host) == (
        False,
        "llama_runtime_payload_incomplete",
    )


def test_the_hip_backend_module_is_required_by_name_not_by_anything_hip_shaped(tmp_path):
    """Codex 3962938550, P1. A real ROCm bundle carries amdhip64_7.dll, hipblas.dll and
    libhipblaslt.dll beside ggml-hip.dll, all four matching a "*hip*.dll" group, so
    quarantining the one module ggml loads left the group satisfied by three libraries that
    cannot stand in for it. Names read off
    app-b10798-mix-659e406-windows-x64-rocm-gfx1150.zip."""
    groups = ILP.runtime_payload_health_groups(
        "windows-rocm", source_label = "published", tag = "b10798"
    )
    hip_groups = [group for group in groups if any("hip" in name for name in group)]
    assert hip_groups == [["ggml-hip*.dll"]]

    host = _windows_host()
    root = _windows_tree(
        tmp_path,
        _PUBLISHED_WINDOWS_PAYLOAD
        + ("amdhip64_7.dll", "hipblas.dll", "libhipblaslt.dll", "ggml-hip.dll"),
        marker = json.dumps({"source": "published", "tag": "b10798", "install_kind": "windows-rocm"}),
    )
    runtime_dir = ILP.install_runtime_dir(root, host)
    assert ILP._runtime_payload_has(root, host, groups) is True
    (runtime_dir / "ggml-hip.dll").unlink()
    assert (
        ILP._runtime_payload_has(root, host, groups) is False
    ), "the three remaining hip-named libraries must not stand in for the ggml backend"


def test_a_dangling_library_symlink_does_not_count_as_present(tmp_path):
    """The tar payloads ship versioned chains (libggml.so -> libggml.so.0 -> libggml.so.0.9.8)
    and Path.glob does not follow links, so quarantining the versioned target left every
    pattern satisfied by links the loader cannot open. Fixed in _runtime_payload_has, which
    this probe and the keep decision share, so the two tighten together.

    All three rungs, which is what a release ships and what the sentence above already said;
    this used to build only the outer two. The middle rung is the SONAME, the name a
    DT_NEEDED entry records, and _payload_match_is_loadable now needs to see whether the
    family has one before it can say what the versionless name means.
    """
    if os.name == "nt":
        pytest.skip("the shipped Windows payload has no symlink chains")
    root = _installed(tmp_path, binaries = True)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    groups = ILP.runtime_payload_health_groups("linux-cpu")
    real = []
    for group in groups:
        stem = group[0].replace("*", "")
        target = runtime_dir / f"{stem}.0.9.8"
        target.write_text("", encoding = "utf-8")
        soname = runtime_dir / f"{stem}.0"
        os.symlink(target.name, soname)
        os.symlink(soname.name, runtime_dir / stem)
        real.append(target)
    assert ILP._runtime_payload_has(root, host, groups) is True

    # Quarantine takes the versioned target and leaves both links behind.
    real[0].unlink()
    assert (runtime_dir / groups[0][0].replace("*", "")).is_symlink()
    assert ILP._runtime_payload_has(root, host, groups) is False


def test_a_directory_matching_a_payload_pattern_is_not_a_library(tmp_path):
    """The same is_file() guard rejects a directory that happens to match, which a bare glob
    would have accepted."""
    root = _installed(tmp_path, binaries = True)
    host = ILP.platform_only_host()
    groups = [["libllama.so*"]]
    (ILP.install_runtime_dir(root, host) / "libllama.so.0").mkdir()
    assert ILP._runtime_payload_has(root, host, groups) is False


@pytest.mark.parametrize("suffix", [".vir", ".quarantined", "_infected"])
def test_a_library_renamed_in_place_no_longer_satisfies_its_group(tmp_path, suffix):
    """Quarantine that renames rather than deletes. Every Linux group ends in ``.so*``, so
    the renamed victim kept matching its own pattern: on a real b10840 install, renaming
    libggml-base.so.0 to libggml-base.so.0.vir left this answering (True, "") while
    llama-server exited with "cannot open shared object file: libggml-base.so.0" and
    _existing_install_runs answered False. Preflight then reported Ready and the launch
    failed at model load with no repair offered, which is the whole point of the probe."""
    if os.name == "nt":
        pytest.skip("the Windows groups name the extension, so a suffix misses them already")
    root = _installed(tmp_path, binaries = True)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    groups = ILP.runtime_payload_health_groups("linux-cpu")
    for group in groups:
        (runtime_dir / f"{group[0].replace('*', '')}.0").write_text("", encoding = "utf-8")
    assert ILP._runtime_payload_has(root, host, groups) is True

    soname = runtime_dir / f"{groups[0][0].replace('*', '')}.0"
    soname.rename(soname.with_name(soname.name + suffix))
    assert ILP._runtime_payload_has(root, host, groups) is False


def test_a_renamed_library_does_not_stand_in_for_its_own_soname(tmp_path):
    """The end-to-end verdict, not only the group test, and the entrypoints stay untouched so
    the reason has to come from the payload rather than from the binaries check."""
    if os.name == "nt":
        pytest.skip("the Windows groups name the extension, so a suffix misses them already")
    root = _installed(tmp_path, binaries = True)
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    published = ILP.runtime_payload_health_groups(
        "linux-cpu", source_label = "published", tag = "b10830"
    )
    for group in published:
        (runtime_dir / f"{group[0].replace('*', '')}.0").write_text("", encoding = "utf-8")
    assert ILP.installed_runtime_health(root, host = host) == (True, "")

    victim = runtime_dir / "libggml-base.so.0"
    victim.rename(runtime_dir / "libggml-base.so.0.vir")
    assert ILP.installed_runtime_health(root, host = host) == (
        False,
        "llama_runtime_payload_incomplete",
    )


def test_an_entrypoint_is_still_a_file_the_loader_would_start(tmp_path):
    """The rejection is keyed on ``.so`` in the name, so llama-server, llama-server.exe and
    every .dll keep passing: reading them as unparseable library names would fail health on
    a complete tree, which is the repair loop installed_runtime_health forbids."""
    for name in ("llama-server", "llama-server.exe", "ggml-base.dll", "libggml.0.dylib"):
        entry = tmp_path / name
        entry.write_text("", encoding = "utf-8")
        assert ILP._payload_match_is_loadable(entry) is True, name


def _macos_host():
    host = ILP.platform_only_host()
    return type(host)(
        **{
            **host.__dict__,
            "system": "Darwin",
            "is_windows": False,
            "is_linux": False,
            "is_macos": True,
            "machine": "arm64",
            "is_x86_64": False,
            "is_arm64": True,
            "macos_version": (15, 5),
        }
    )


def _macos_payload(runtime_dir: Path) -> None:
    """The dylib set a real macos-arm64 bundle ships, chains and all.

    Taken from llama-b10840-mix-d5c17a0-bin-macos-arm64.tar.gz rather than invented: each
    library is libX.dylib -> libX.0.dylib -> libX.<version>.dylib, and the accelerator and
    transport backends sit beside the core ones matching the same broad prefix.
    """
    for stem, version in (
        ("libllama-common", "0.4.0"),
        ("libllama", "0.4.0"),
        ("libmtmd", "0.4.0"),
        ("libggml", "0.23.0"),
        ("libggml-base", "0.23.0"),
        ("libggml-cpu", "0.23.0"),
        ("libggml-metal", "0.23.0"),
        ("libggml-blas", "0.23.0"),
        ("libggml-rpc", "0.23.0"),
    ):
        (runtime_dir / f"{stem}.{version}.dylib").write_text("", encoding = "utf-8")
        os.symlink(f"{stem}.{version}.dylib", runtime_dir / f"{stem}.0.dylib")
        os.symlink(f"{stem}.0.dylib", runtime_dir / f"{stem}.dylib")


def _macos_tree(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    runtime_dir = root / "build" / "bin"
    runtime_dir.mkdir(parents = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"tag": "b10840", "release_tag": "b10840-mix-d5c17a0", "source": "published"}),
        encoding = "utf-8",
    )
    for name in ("server", "quantize"):
        binary = runtime_dir / f"llama-{name}"
        binary.write_text("", encoding = "utf-8")
        # An installed entrypoint is executable, and the probe now asks for that
        # rather than for mere presence, the way _existing_install_runs does.
        os.chmod(binary, 0o755)
    _macos_payload(runtime_dir)
    return root


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
def test_a_complete_macos_payload_is_healthy(tmp_path):
    assert ILP.installed_runtime_health(_macos_tree(tmp_path), host = _macos_host()) == (True, "")


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
@pytest.mark.parametrize(
    "stem",
    ["libllama-common", "libllama", "libggml", "libggml-base", "libggml-cpu", "libmtmd"],
)
def test_each_essential_macos_dylib_is_required_on_its_own(tmp_path, stem):
    """Codex 3958908349, P1. The groups used to be three broad alternatives, so
    libggml*.dylib stayed satisfied by libggml-base, libggml-blas, libggml-cpu, libggml-metal
    and libggml-rpc after the one the loader needs was quarantined, and the tree reported
    healthy while llama-server died in dyld. The dot in each pattern is what keeps it off its
    siblings."""
    root = _macos_tree(tmp_path)
    runtime_dir = root / "build" / "bin"
    for path in [p for p in runtime_dir.iterdir() if p.name.split(".")[0] == stem]:
        path.unlink()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_payload_incomplete",
    )


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
def test_losing_only_the_macos_chain_target_is_caught(tmp_path):
    """The links survive quarantine of the versioned file they point at, and a name-only
    match would still satisfy the pattern."""
    root = _macos_tree(tmp_path)
    (root / "build" / "bin" / "libggml.0.23.0.dylib").unlink()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_payload_incomplete",
    )


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
def test_the_macos_accelerator_backends_are_not_required(tmp_path):
    """Over-strictness is the repair-loop direction. metal, blas and rpc are the accelerator
    and transport backends, the way libggml-cuda is on Linux, so demanding one a bundle does
    not carry would reinstall every install that lacks it."""
    root = _macos_tree(tmp_path)
    runtime_dir = root / "build" / "bin"
    for path in [
        p
        for p in runtime_dir.iterdir()
        if p.name.startswith(("libggml-metal", "libggml-blas", "libggml-rpc"))
    ]:
        path.unlink()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
def test_losing_the_macos_install_name_link_is_caught(tmp_path):
    """Codex 3959620556, P1. The middle link of the chain is the one dyld asks for, and it
    is the one a name-only match hides: llama-server's LC_LOAD_DYLIB entry is
    ``@rpath/libggml.0.dylib`` (read out of the shipped macos-arm64 bundle, and matching the
    LC_ID_DYLIB recorded in libggml.0.23.0.dylib itself), so with it quarantined the process
    dies in dyld while ``libggml.*.dylib`` is still satisfied by the terminal file. The
    version-depth rule that already covers ``libfoo.so.0`` beside ``libfoo.so.0.0.10360``
    covers this too."""
    root = _macos_tree(tmp_path)
    runtime_dir = root / "build" / "bin"
    (runtime_dir / "libggml.0.dylib").unlink()
    # The terminal file is still there and still a real file, which is what made this
    # read as healthy.
    assert (runtime_dir / "libggml.0.23.0.dylib").is_file()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_payload_incomplete",
    )


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
def test_a_bare_versionless_macos_dylib_still_satisfies_its_group(tmp_path):
    """The other direction, so the rule above cannot become a reinstall loop: a bundle that
    ships libfoo.dylib with no chain at all is a name the loader can resolve and must stay
    healthy."""
    root = _macos_tree(tmp_path)
    runtime_dir = root / "build" / "bin"
    for path in [p for p in runtime_dir.iterdir() if p.name.split(".")[0] == "libggml"]:
        path.unlink()
    (runtime_dir / "libggml.dylib").write_text("", encoding = "utf-8")
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")


@pytest.mark.skipif(os.name == "nt", reason = "POSIX permission bits")
@pytest.mark.parametrize("name", ["server", "quantize"])
def test_a_runtime_binary_stripped_of_its_execute_bit_is_broken(tmp_path, name):
    """Codex 3959620570, P2. Extraction damage or security software can clear the bit without
    deleting the file. ``_find_llama_server_binary`` then classifies it non-executable and
    refuses to fall back to another runtime, and ``_existing_install_runs`` rejects the tree
    on the same ``os.access(X_OK)``, so an ``exists()`` check here was the only thing left
    calling it Ready."""
    root = _macos_tree(tmp_path)
    binary = root / "build" / "bin" / f"llama-{name}"
    os.chmod(binary, 0o755)
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")
    os.chmod(binary, 0o644)
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_binaries_missing",
    )


@pytest.mark.skipif(os.name == "nt", reason = "the fixture needs POSIX symlinks")
@pytest.mark.parametrize("name", ["server", "quantize"])
def test_a_directory_where_a_runtime_entrypoint_belongs_is_broken(tmp_path, name):
    """Codex 3960401513, P2. A directory is searchable, so os.access(X_OK) answers true for
    one and exists() does too, while ``_file_status`` in the finder asks ``is_file()`` and
    rejects the tree. Failed extraction and filesystem corruption both leave exactly that,
    and it read as a healthy install.

    ``_existing_install_runs`` asks the same way now, through the same helper: a tree this
    rejects but that one keeps would be repaired, left unchanged and rejected again next
    launch."""
    root = _macos_tree(tmp_path)
    binary = root / "build" / "bin" / f"llama-{name}"
    binary.unlink()
    binary.mkdir()
    assert os.access(binary, os.X_OK) and binary.exists()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_binaries_missing",
    )
    assert not ILP._entrypoint_is_runnable(binary, _macos_host())


@pytest.mark.skipif(os.name == "nt", reason = "the root wrapper is written on POSIX only")
@pytest.mark.parametrize("name", ["server", "quantize"])
def test_a_rotten_root_entrypoint_is_not_saved_by_a_healthy_build_bin(tmp_path, name):
    """``create_exec_entrypoint`` writes a real wrapper at the install root when it cannot
    make a symlink, and ``_find_llama_server_binary`` reaches that root copy before
    build/bin, so it can rot on its own. ``_existing_install_runs`` probes the root copies
    first for exactly this reason; grading only build/bin here left the tree Ready while the
    backend launched the wrapper the loader refuses."""
    root = _macos_tree(tmp_path)
    wrapper = root / f"llama-{name}"
    wrapper.write_text("#!/bin/sh\n", encoding = "utf-8")
    os.chmod(wrapper, 0o755)
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")

    os.chmod(wrapper, 0o644)
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (
        False,
        "llama_runtime_binaries_missing",
    )


@pytest.mark.skipif(os.name == "nt", reason = "the root wrapper is written on POSIX only")
def test_a_root_entrypoint_that_is_not_there_is_not_a_pin(tmp_path):
    """An absent root copy, and a link whose target went with it, both read as absent to the
    finder, which then falls through to build/bin. Calling either one broken would fail
    health on every install that never got a root entrypoint at all."""
    root = _macos_tree(tmp_path)
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")

    os.symlink("build/bin/llama-server-that-went-away", root / "llama-server")
    assert not (root / "llama-server").exists()
    assert ILP.installed_runtime_health(root, host = _macos_host()) == (True, "")
