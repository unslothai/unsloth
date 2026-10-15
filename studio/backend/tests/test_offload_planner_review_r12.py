# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MoE cache's host-RAM admission charges the prompt cache once, and a launch that reaches
the child with the fitter on is not watched as a full offload."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

from core.inference.offload_planner import Plan  # noqa: E402
from test_llama_cpp_placement import (  # noqa: E402
    _cache_launch,
    _discrete_linux_host,  # noqa: F401
    _has_moe_cache,
    _moe_cache_host,  # noqa: F401
)
from test_offload_planner_launch import CARD_MIB, _backend  # noqa: E402


@pytest.mark.parametrize("owns", [False, True], ids = ["flag_off", "flag_on"])
def test_the_moe_cache_admission_charges_the_prompt_cache_once(
    tmp_path, _moe_cache_host, monkeypatch, owns
):
    """15 GiB of experts plus an 8 GiB prompt cache need 23 GiB of the 28 GiB host. The
    planner's own prompt cache term rode into the admission next to the one it adds, so a
    declined plan charged 31 GiB and dropped --moe-cache-mib auto that main emits."""
    from core.inference.llama_cpp import LlamaCppBackend

    if owns:
        monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    else:
        monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)
    monkeypatch.setenv("UNSLOTH_MOE_CACHE", "on")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_planned_tensor_spill",
        lambda self, inputs, **_kw: Plan(reason = "declined for the test"),
    )
    _backend_, cmd = _cache_launch(
        tmp_path, memory = [(0, 20_000, 32_000)], ram_gib = 28, extra_args = ["--cache-ram", "8192"]
    )
    assert cmd[cmd.index("--fit") + 1] == "on", cmd
    assert _has_moe_cache(cmd), cmd


def test_a_launch_that_reaches_the_child_with_the_fitter_on_is_not_watched(tmp_path, monkeypatch):
    """Studio proved a full offload and emitted -ngl -1 --fit off, but the argv the child got
    ends in --fit on (a pass-through here; a --flash-attn off retry that re-enables the fitter
    is the other route): that is the fitter's placement, which may hold layers in host RAM."""
    import subprocess
    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    seen = []

    def spy(self, **kw):
        seen.append(bool(kw.get("fully_gpu_offloaded")))
        return None

    monkeypatch.setattr(LlamaCppBackend, "_windows_sysmem_fallback_watch", spy)
    monkeypatch.delenv("UNSLOTH_SMART_OFFLOAD", raising = False)

    def run(returncodes, extra_args = None):
        backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])
        backend.probe_server_capabilities = lambda _binary = None: {"supports_metrics": True}
        launches: list = []

        class _Process:
            pid = 123
            stdout = ()

            def __init__(self, returncode):
                self.returncode = returncode

            def poll(self):
                return self.returncode

            def terminate(self):
                return None

            def wait(self, timeout = None):
                return self.returncode

            def kill(self):
                return None

        real_popen = subprocess.Popen

        def popen(cmd, **kwargs):
            if not cmd or str(cmd[0]) != "/fake/llama-server":
                return real_popen(cmd, **kwargs)
            launches.append(list(cmd))
            return _Process(returncodes[len(launches) - 1])

        backend._wait_for_health = lambda timeout, **_kw: returncodes[len(launches) - 1] is None
        seen.clear()
        with patch.object(subprocess, "Popen", side_effect = popen):
            assert backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "t", extra_args = extra_args)
            )
        return launches

    clean = run([None])
    assert clean[0][clean[0].index("--fit") + 1] == "off"
    assert seen == [True], seen

    fitted = run([None], extra_args = ("--fit", "on"))
    assert fitted[0][-2:] == ["--fit", "on"], fitted[0]
    assert seen == [False], seen
