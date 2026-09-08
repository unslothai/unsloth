# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launch-time health probe for the managed llama.cpp runtime.

Smart App Control and antivirus quarantine individual files out of a tree that is
otherwise present and whose marker still says installed. Preflight decided staleness on
the managed Python alone, so the desktop launched, and the missing DLL only surfaced at
model load as an unrelated-looking error. ``installed_runtime_health`` is what the
capability payload answers with so that install gets repaired instead.

The payload tables themselves are covered by ``test_keep_install_backcompat_9979``;
what is tested here is the composition around them: nothing installed is not a broken
install, a missing runtime directory is distinguished from a gutted one, and the probe
does not pay for a GPU detection it never reads.
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
            (runtime_dir / f"llama-{name}{ext}").write_text("", encoding = "utf-8")
    return root


def test_no_marker_is_not_installed_rather_than_broken(tmp_path):
    """None, not (False, ...). A user who has never installed a runtime must not be sent
    through repair, and the desktop reads the two answers differently."""
    (tmp_path / "llama.cpp").mkdir()
    assert ILP.installed_runtime_health(tmp_path / "llama.cpp") is None
    assert ILP.installed_runtime_health(tmp_path / "nothing-here") is None


def test_a_marker_without_its_runtime_directory_is_broken(tmp_path):
    """The whole tree can go, not only files inside it: quarantine that takes the last
    file leaves the directory empty, and a manual cleanup leaves the marker orphaned."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text("{}\n", encoding = "utf-8")
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_dir_missing")


def test_an_empty_runtime_directory_is_broken_not_healthy(tmp_path):
    """The marker says installed and the directory exists; every binary is gone. This is
    the shape a quarantine leaves behind, and it used to pass preflight."""
    root = _installed(tmp_path)
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_payload_incomplete")


def test_the_verdict_is_delegated_to_the_payload_tables(tmp_path, monkeypatch):
    """One payload decider, not two. Duplicating the required-file list here is how the
    launch probe and the keep-install path would drift apart."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    assert ILP.installed_runtime_health(root) == (True, "")


def test_the_default_root_is_the_managed_install_dir(tmp_path, monkeypatch):
    """Called with no argument by the CLI, which has no opinion about where the runtime
    lives: the UNSLOTH_LLAMA_CPP_PATH override has to keep working through it."""
    root = _installed(tmp_path)
    monkeypatch.setattr(ILP, "default_managed_llama_dir", lambda: root)
    assert ILP.installed_runtime_health() == (False, "llama_runtime_payload_incomplete")


def test_platform_only_host_agrees_with_a_detected_host_on_the_platform_facts():
    """It stands in for detect_host() in the payload checks, which read these booleans and
    nothing else. If the two ever disagreed, the probe would grade the tree for the wrong
    operating system."""
    cheap = ILP.platform_only_host()
    detected = ILP.detect_host()
    for field in ("system", "is_windows", "is_linux", "is_macos", "is_x86_64", "is_arm64"):
        assert getattr(cheap, field) == getattr(detected, field), field
    assert cheap.machine == detected.machine.lower()


def test_platform_only_host_does_not_probe_for_gpus(monkeypatch):
    """The point of it. detect_host() shells out to nvidia-smi and friends and costs over a
    second; a launch probe that only wants to know whether files are missing must not."""

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
    capability rather than a false stale verdict. Swallowing it twice would hide a real bug
    in the payload tables from the test suite as well."""
    root = _installed(tmp_path)

    def raise_it(*_):
        raise broken

    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", raise_it)
    with pytest.raises(type(broken)):
        ILP.installed_runtime_health(root)


def test_a_quarantined_llama_server_is_caught_even_with_a_complete_payload(tmp_path, monkeypatch):
    """The payload groups are libraries, and on Linux and macOS they name no executable at
    all, so the server binary going missing has to be looked for separately."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    host = ILP.platform_only_host()
    ext = ".exe" if host.is_windows else ""
    (ILP.install_runtime_dir(root, host) / f"llama-server{ext}").unlink()
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_binaries_missing")


def test_llama_quantize_is_required_because_the_setup_scripts_require_it(tmp_path, monkeypatch):
    """Not an arbitrary second file: _existing_install_runs demands both, so demanding
    both here keeps this call no stricter than the repair that answers it."""
    root = _installed(tmp_path, binaries = True)
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    host = ILP.platform_only_host()
    ext = ".exe" if host.is_windows else ""
    (ILP.install_runtime_dir(root, host) / f"llama-quantize{ext}").unlink()
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_binaries_missing")


def test_an_explicit_host_overrides_the_detected_platform(tmp_path):
    """The simulation matrices grade a tree for a platform this machine is not, and the
    desktop capability probe is the only caller that wants the local one."""
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
    """Reversed after review (Codex 3957561256, P2). An absent marker file is a runtime
    nobody installed; a marker that is present and unreadable is a real tree whose write was
    interrupted. Short-circuiting the second to None left preflight Ready when a library was
    missing too, which is the exact failure this probe exists to catch. The keep path reads
    such a marker as an unknown backend and grades the payload anyway, so grading it here
    stays on the safe side of the no-stricter rule."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text('{"release_tag": "b108', encoding = "utf-8")
    assert ILP.load_prebuilt_metadata(root) is None
    # The payload is empty, so the tree is broken and must be offered for repair.
    assert ILP.installed_runtime_health(root) == (False, "llama_runtime_payload_incomplete")


def test_an_unparsable_marker_over_a_complete_tree_is_still_healthy(tmp_path, monkeypatch):
    """The other half, and the one that keeps the loop shut: an unreadable marker on a tree
    that is otherwise intact must not be called broken. _existing_install_runs keeps such a
    tree (confirm_install_tree only checks the marker file exists), so reporting it broken
    would repair, keep, and repair again on every launch for an offline user."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text("not json", encoding = "utf-8")
    monkeypatch.setattr(ILP, "_kept_install_payload_is_healthy", lambda *_: True)
    assert ILP.installed_runtime_health(root) == (True, "")


def test_an_absent_marker_is_still_not_installed(tmp_path):
    """The distinction the fix turns on, asserted directly."""
    root = _installed(tmp_path, binaries = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").unlink()
    assert ILP.installed_runtime_health(root) is None


def test_a_dangling_library_symlink_does_not_count_as_present(tmp_path):
    """Codex 3957928993, P1, reproduced before fixing: the tar payloads ship versioned chains
    (libggml.so -> libggml.so.0 -> libggml.so.0.9.8) and Path.glob lists names without
    following them, so quarantining only the versioned target left every pattern satisfied by
    links the loader cannot open. Fixed in _runtime_payload_has, which both this probe and the
    setup scripts' keep decision share, so the two tighten together."""
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
        os.symlink(target.name, runtime_dir / stem)
        real.append(target)
    assert ILP._runtime_payload_has(root, host, groups) is True

    # Quarantine takes the versioned target and leaves the link behind.
    real[0].unlink()
    assert (runtime_dir / groups[0][0].replace("*", "")).is_symlink()
    assert ILP._runtime_payload_has(root, host, groups) is False


def test_a_directory_matching_a_payload_pattern_is_not_a_library(tmp_path):
    """The same guard, from the other side: is_file() is what rejects a dangling link, and it
    rejects a directory that happens to match too, which a bare glob would have accepted."""
    root = _installed(tmp_path, binaries = True)
    host = ILP.platform_only_host()
    groups = [["libllama.so*"]]
    (ILP.install_runtime_dir(root, host) / "libllama.so.0").mkdir()
    assert ILP._runtime_payload_has(root, host, groups) is False
