# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows CUDA sysmem-fallback watch honours the same CPU-residency verdict the Model
Memory policy uses: an explicit CPU placement (-ngl 0, --override-tensor .*=CPU, --cpu-moe)
keeps weights in host RAM, so slow generation with VRAM full is legitimate there, not paging."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from test_offload_planner_launch import CARD_MIB, _backend  # noqa: E402


def _watched(tmp_path, monkeypatch, extra_args):
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    seen = []

    def spy(self, **kw):
        seen.append(bool(kw.get("fully_gpu_offloaded")))
        return None

    monkeypatch.setattr(LlamaCppBackend, "_windows_sysmem_fallback_watch", spy)
    monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])
    backend.probe_server_capabilities = lambda _binary = None: {"supports_metrics": True}
    launches: list = []

    class _Process:
        pid = 123
        stdout = ()
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return None

        def kill(self):
            return None

    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return real_popen(cmd, **kwargs)
        launches.append(list(cmd))
        return _Process()

    backend._wait_for_health = lambda timeout, **_kw: True
    with patch.object(subprocess, "Popen", side_effect = popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "t", extra_args = extra_args)
        )
    assert len(launches) == 1, launches
    assert seen and len(seen) == 1, seen
    return launches[0], seen[0]


@pytest.mark.parametrize(
    "extra_args",
    [("-ngl", "0"), ("--override-tensor", ".*=CPU"), ("--cpu-moe",)],
    ids = ["ngl0", "override_tensor_cpu", "cpu_moe"],
)
def test_an_explicit_cpu_placement_is_not_watched_for_sysmem_fallback(
    tmp_path, monkeypatch, extra_args
):
    cmd, watched = _watched(tmp_path, monkeypatch, extra_args)
    assert cmd[cmd.index("--fit") + 1] == "off", cmd
    assert watched is False, (extra_args, cmd)


def test_a_full_gpu_launch_is_still_watched(tmp_path, monkeypatch):
    cmd, watched = _watched(tmp_path, monkeypatch, None)
    assert cmd[cmd.index("--fit") + 1] == "off", cmd
    assert watched is True, cmd


def test_a_fit_on_launch_is_not_watched(tmp_path, monkeypatch):
    cmd, watched = _watched(tmp_path, monkeypatch, ("--fit", "on"))
    assert cmd[-2:] == ["--fit", "on"], cmd
    assert watched is False, cmd
