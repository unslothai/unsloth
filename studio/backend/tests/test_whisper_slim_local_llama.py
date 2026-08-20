# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Slim whisper pairing against an unmanaged local llama.cpp runtime.

--with-llama-cpp-dir links a user-built llama tree into the canonical install
location with no managed prebuilt marker, which used to make every slim
whisper artifact skip pairing ("no managed llama.cpp prebuilt install") and
the install fail outright. The runtime is now accepted with no release tag,
with the per-file ABI gates (requires_ggml_sonames, backend module glob)
deciding -- these tests pin that contract.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

iwp = importlib.import_module("install_whisper_prebuilt")

if not hasattr(iwp, "_local_unmanaged_llama_runtime"):
    pytest.skip("unmanaged-runtime fallback not present - check branch", allow_module_level = True)


def _host(**overrides) -> iwp.HostInfo:
    info = iwp.HostInfo(
        system = "Linux",
        machine = "x86_64",
        whisper_os = "linux",
        whisper_arch = "x64",
        archive_ext = ".tar.gz",
        is_windows = False,
        is_macos = False,
        is_apple_silicon = False,
    )
    for key, value in overrides.items():
        setattr(info, key, value)
    return info


def _artifact(**overrides) -> dict:
    payload = {
        "asset": "whisper-v1.9.2-unsloth.10-linux-x64-slim.tar.gz",
        "requires_llama_tag": "b10360-mix-87da1a2",
        "requires_ggml_tree": "abc123",
        "requires_ggml_sonames": ["libggml.so", "libggml-base.so", "libggml-cpu.so"],
    }
    payload.update(overrides)
    return payload


def _link_local_tree(monkeypatch, tmp_path, files):
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    if os.name == "nt":
        bin_dir = bin_dir / "Release"
    bin_dir.mkdir(parents = True)
    for name in files:
        (bin_dir / name).write_bytes(b"")
    monkeypatch.setattr(iwp.llama, "default_managed_llama_dir", lambda: tmp_path / "llama.cpp")
    monkeypatch.setattr(iwp, "installed_llama_runtime", lambda *a, **k: None)
    monkeypatch.setattr(iwp, "installed_llama_ggml_tree", lambda *a, **k: None)
    return bin_dir


def test_local_llama_without_marker_pairs_via_abi_gates(monkeypatch, tmp_path):
    bin_dir = _link_local_tree(
        monkeypatch,
        tmp_path,
        ["libggml.so", "libggml-base.so", "libggml-cpu.so", "llama-server"],
    )

    runtime = iwp._local_unmanaged_llama_runtime()
    assert runtime == (bin_dir, "", "")

    paired = iwp.slim_pairing_for_artifact(_artifact(), _host(), "cpu")
    assert paired == (bin_dir, "")
    # The tag heuristic alone would have refused: the required tag pairs with
    # nothing a source build can report.
    assert not iwp.llama_runtime_pairs("", _artifact()["requires_llama_tag"])


def test_local_llama_missing_soname_still_rejects(monkeypatch, tmp_path):
    _link_local_tree(monkeypatch, tmp_path, ["libggml.so", "llama-server"])

    assert iwp.slim_pairing_for_artifact(_artifact(), _host(), "cpu") is None


def test_local_llama_missing_backend_module_still_rejects(monkeypatch, tmp_path):
    # All sonames present, but the cuda module glob finds nothing: a cpu-only
    # source build cannot back the cuda pairing.
    _link_local_tree(
        monkeypatch, tmp_path, ["libggml.so", "libggml-base.so", "libggml-cpu.so"]
    )

    assert iwp.slim_pairing_for_artifact(_artifact(), _host(), "cuda") is None


def test_no_llama_tree_at_all_still_reports_no_managed_install(monkeypatch, tmp_path):
    monkeypatch.setattr(iwp.llama, "default_managed_llama_dir", lambda: tmp_path / "llama.cpp")
    monkeypatch.setattr(iwp, "installed_llama_runtime", lambda *a, **k: None)
    monkeypatch.setattr(iwp, "installed_llama_ggml_tree", lambda *a, **k: None)

    assert iwp._local_unmanaged_llama_runtime() is None
    assert iwp.slim_pairing_for_artifact(_artifact(), _host(), "cpu") is None


def test_managed_runtime_still_wins_over_the_local_tree(monkeypatch, tmp_path):
    bin_dir = _link_local_tree(monkeypatch, tmp_path, ["libggml.so", "llama-server"])
    managed_bin = tmp_path / "managed" / "build" / "bin"
    if os.name == "nt":
        managed_bin = managed_bin / "Release"
    managed_bin.mkdir(parents = True)
    for name in ("libggml.so", "libggml-base.so", "libggml-cpu.so"):
        (managed_bin / name).write_bytes(b"")
    monkeypatch.setattr(
        iwp,
        "installed_llama_runtime",
        lambda *a, **k: (managed_bin, "b10360-mix-87da1a2", "cuda13-newer"),
    )

    paired = iwp.slim_pairing_for_artifact(_artifact(), _host(), "cpu")
    assert paired is not None and paired[0] != bin_dir
