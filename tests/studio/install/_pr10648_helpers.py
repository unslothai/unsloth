# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The preamble the six ``test_pr10648_*`` suites genuinely share.

Only what was BYTE-IDENTICAL in more than one suite lives here. Anything describing one
suite's scenario stays in that suite, even where another file uses the same name --
``_install``, ``_host``, ``_fast_path`` and ``_marker`` each mean different things in
different files, and merging them would be a bug rather than a cleanup.

Every suite runs under ``pytest -n 4``, so: nothing here memoises or caches, and
``load_studio_module`` takes the ``sys.modules`` name as an argument rather than picking
one. The suites monkeypatch module globals, so two files sharing a key would see each
other's patches inside an xdist worker; the distinct per-suite names are what keeps them
apart, and they stay spelled out at each call site.

llama, whisper and node each have their OWN ``HostInfo`` dataclass with different fields,
so the factories below stay separate and there is deliberately no fourth that unifies
them. Each takes the dataclass, because every suite loads its own module instance.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    # The installers import each other by module name; spec-based loading needs studio/ reachable.
    sys.path.insert(0, str(STUDIO_DIR))


WINDOWS_HOST = os.name == "nt"
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

# os.access(path, os.X_OK) answers "does this exist" on Windows, and os.chmod cannot clear
# an execute bit Windows does not have, so the mode-bit rows are indistinguishable there.
POSIX_ONLY = pytest.mark.skipif(
    WINDOWS_HOST,
    reason = "mode bits and os.access(X_OK) are POSIX only",
)
# root reads and executes regardless of the bits, so "unreadable" and "not executable"
# stop being the corruptions those tests apply.
NOT_ROOT = pytest.mark.skipif(
    IS_ROOT,
    reason = "root bypasses the permission bits this asserts on",
)
NEEDS_CHOWN = pytest.mark.skipif(
    not hasattr(os, "chown"),
    reason = "os.chown does not exist on this platform",
)


def load_studio_module(module_name: str, filename: str):
    """Load one ``studio/`` installer under *module_name*.

    Every install test loads these modules. ``module_name`` is the caller's to choose and
    must be unique to the calling suite: sharing a ``sys.modules`` key with another test
    file would mean one file's module-level monkeypatching could be observed by another
    under ``pytest -n``.
    """
    spec = importlib.util.spec_from_file_location(module_name, STUDIO_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def llama_host(host_cls, **overrides):
    """A simulated llama.cpp host. Mirrors ``test_selection_logic.make_host``: the platform
    booleans are derived from ``system``/``machine``, so no caller can hand out a host whose
    flags contradict the two strings that name it. Accelerator fields default to a bare CPU
    box, and ``os.name`` is never patched -- that changes ``pathlib`` underneath the trees.
    """
    system = overrides.pop("system", "Linux")
    machine = overrides.pop("machine", "x86_64")
    defaults = dict(
        system = system,
        machine = machine,
        is_linux = system == "Linux",
        is_windows = system == "Windows",
        is_macos = system == "Darwin",
        is_x86_64 = machine.lower() in {"x86_64", "amd64"},
        is_arm64 = machine.lower() in {"arm64", "aarch64"},
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    defaults.update(overrides)
    return host_cls(**defaults)


def whisper_host(host_cls, **overrides):
    """A simulated whisper.cpp host. A different dataclass from the llama one: it carries
    ``whisper_os``/``whisper_arch``/``archive_ext``/``is_apple_silicon`` and no GPU fields."""
    fields = dict(
        system = "Linux",
        machine = "x64",
        whisper_os = "linux",
        whisper_arch = "x64",
        archive_ext = ".tar.gz",
        is_windows = False,
        is_macos = False,
        is_apple_silicon = False,
    )
    fields.update(overrides)
    return host_cls(**fields)


WHISPER_RELEASE_TAG = "v1.9.1-unsloth.1"


def whisper_selection_fields(whisper, **overrides) -> dict:
    """The fields of a plain CPU Linux ``InstallSelection``.

    Returned as a dict rather than an instance: the suites construct it from different
    classes (``prebuilt_core.InstallSelection`` in one, ``install_whisper_prebuilt``'s
    re-export in another) and which instance of ``prebuilt_core`` a suite is exercising is
    part of what that suite is saying.
    """
    fields = dict(
        published_repo = whisper.DEFAULT_PUBLISHED_REPO,
        release_tag = WHISPER_RELEASE_TAG,
        upstream_tag = "v1.9.1",
        source_commit = "0" * 40,
        asset = f"whisper-{WHISPER_RELEASE_TAG}-linux-x64-cpu.tar.gz",
        asset_sha256 = "c" * 64,
        backend = "cpu",
        runtime_line = None,
        coverage = {"min_os": None},
        studio_protocol = "inference/multipart-v1",
        platform_os = "linux",
        platform_arch = "x64",
    )
    fields.update(overrides)
    return fields


def whisper_install_is_intact(
    whisper,
    install_dir: Path,
    host,
    *,
    backend: str = "cpu",
) -> bool:
    """The on-disk half of the whisper precheck: marker identity, the slim pairing, and
    ``installed_tree_is_intact``. No release lookup, so no network."""
    return (
        whisper._existing_install_is_intact(
            install_dir,
            host,
            published_repo = whisper.DEFAULT_PUBLISHED_REPO,
            requested_backend = backend,
        )
        is not None
    )


def git(
    *args: str,
    text: bool = False,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Read-only git against this checkout. Never mutates the working tree."""
    return subprocess.run(
        ["git", "-C", str(PACKAGE_ROOT), *args],
        capture_output = True,
        text = text,
        timeout = timeout,
        check = False,
    )
