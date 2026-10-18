# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""-sm none plans on an unmasked multi-GPU host, and the text-only projector retry after a plan."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_BACKEND_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from core.inference.offload_planner import Plan  # noqa: E402
from test_offload_cost_gate import GIB  # noqa: E402

# nvidia-smi index,gen,width: card 0 on a gen4 x16 slot, card 1 on a gen3 x1 riser.
_LINK_ROWS = "0, 4, 16\n1, 3, 1\n"
_GEN4_X16 = 16 * 1.969 * 0.85 * 1e9 / float(1024**3)


def _sm_none(
    tmp_path,
    monkeypatch,
    plan,
    *,
    order,
    pci = True,
    extra_args = None,
):
    """Two cards (48 and 24 GiB), no mask, no pick, --split-mode none: the planner budgets the first
    row it keeps, llama.cpp keeps only the child's device 0. Returns the argv, the child env, the
    planner inputs and the paging watch's arguments."""
    from core.inference import llama_cpp as llama_mod
    from core.inference import llama_stats
    from test_llama_cpp_placement import _backend, _launch
    from test_offload_planner_launch import DENSE

    cls = llama_mod.LlamaCppBackend
    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(cls, "_host_torch_is_rocm", staticmethod(lambda: False))
    monkeypatch.setattr(cls, "_GPU_IDS_ARE_PCI_INDICES", pci)
    monkeypatch.setattr(cls, "_nvidia_link_query", staticmethod(lambda: _LINK_ROWS))
    monkeypatch.setattr(llama_stats, "maybe_start_stats_logger", lambda *a, **k: None)
    big, small = 48 * 1024, 24 * 1024
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, big, big), (1, small, small)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 90 * GIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
        "supports_metrics": True,
    }
    backend._available_system_memory_mib = lambda: 256 * 1024
    asked = []
    backend._planned_tensor_spill = lambda inputs, **_kw: asked.append(inputs) or plan
    watch = {}

    def fake_watch(**kwargs):
        watch.update(kwargs)
        return None

    backend._windows_sysmem_fallback_watch = fake_watch
    launched = _launch(
        backend,
        gguf,
        speculative_type = "off",
        n_ctx = 8192,
        n_parallel = 1,
        extra_args = list(extra_args or ["--split-mode", "none"]),
    )
    assert asked, "the planner was never consulted"
    return launched["cmd"], launched["env"], asked[0], watch, backend


def _spilling_plan() -> Plan:
    return Plan(changed = True, priced = True, n_ctx = 8192, ot_patterns = ("x",), spilled_blocks = (1,))


def _resident_plan() -> Plan:
    """Moves no weight: every layer stays on the card, the launch the paging watch exists for."""
    return Plan(changed = True, priced = True, n_ctx = 8192, n_parallel = 1)


# --------------------------- an unmasked -sm none plan outside PCI order (r25 finding 5)


@pytest.mark.parametrize(
    "order, pci, keeps_plan",
    [
        # No mask, no pick: the child inherits FASTEST_FIRST, so device 0 is CUDA's fastest card,
        # while the plan priced nvidia-smi's PCI row 0.
        ("FASTEST_FIRST", True, False),
        # PCI order: device 0 is PCI row 0, the card the plan priced.
        ("PCI_BUS_ID", True, True),
        # Torch-fallback rows are CUDA ordinals in the parent's own order, which the unmasked
        # child shares: row 0 is device 0.
        ("FASTEST_FIRST", False, True),
    ],
)
def test_an_unmasked_split_mode_none_plan_declines_outside_pci_order(
    tmp_path, monkeypatch, order, pci, keeps_plan
):
    cmd, env, _inputs, _watch, backend = _sm_none(
        tmp_path, monkeypatch, _spilling_plan(), order = order, pci = pci
    )
    if keeps_plan:
        assert "-ot" in cmd and cmd[cmd.index("--fit") + 1] == "off", cmd
    else:
        assert "-ot" not in cmd and cmd[cmd.index("--fit") + 1] == "on", cmd
        assert not backend._spill_plan_flags


def test_an_unmasked_layer_split_plan_is_not_touched_by_the_split_mode_none_rule(
    tmp_path, monkeypatch
):
    """The control: without -sm none the plan spans both cards and the split pins PCI order."""
    plan = Plan(
        changed = True,
        priced = True,
        n_ctx = 8192,
        ot_patterns = ("x",),
        spilled_blocks = (1,),
        device_layer_counts = (20, 12),
    )
    cmd, env, _inputs, _watch, _backend = _sm_none(
        tmp_path, monkeypatch, plan, order = "PCI_BUS_ID", extra_args = ["--top-k", "5"]
    )
    assert "-ot" in cmd and cmd[cmd.index("--fit") + 1] == "off", cmd


# --------------------------------------- the link probe reads the -sm none card (finding 4)


def test_the_link_probe_reads_only_the_card_an_sm_none_plan_uses(tmp_path, monkeypatch):
    _cmd, _env, inputs, _watch, _backend = _sm_none(
        tmp_path, monkeypatch, _spilling_plan(), order = "PCI_BUS_ID"
    )
    # Card 1's x1 riser (0.78 GiB/s) carries none of this plan's traffic.
    assert inputs["link_gib_s"] == pytest.approx(_GEN4_X16), inputs["link_gib_s"]


def test_the_link_probe_still_reads_every_card_a_layer_split_uses(tmp_path, monkeypatch):
    _cmd, _env, inputs, _watch, _backend = _sm_none(
        tmp_path, monkeypatch, _spilling_plan(), order = "PCI_BUS_ID", extra_args = ["--top-k", "5"]
    )
    assert inputs["link_gib_s"] < 1.0, inputs["link_gib_s"]


# ------------------------------------------- the paging watch reads the -sm none card (finding 7)


def test_the_paging_watch_samples_only_the_card_an_sm_none_plan_uses(tmp_path, monkeypatch):
    cmd, _env, _inputs, watch, _backend = _sm_none(
        tmp_path, monkeypatch, _resident_plan(), order = "PCI_BUS_ID"
    )
    assert cmd[cmd.index("--fit") + 1] == "off" and "-ot" not in cmd, cmd
    assert watch.get("fully_gpu_offloaded") is True, watch
    assert list(watch.get("gpu_indices") or ()) == [0], watch


# ---------------------------- the text-only projector retry after a planned launch (finding 6)


def test_the_text_only_projector_retry_does_not_respawn_the_revoked_plan(tmp_path, monkeypatch):
    """The planned launch fails on its projector without crashing (no in-loop startup-failure
    revocation), so the plan is revoked on `cmd` but `_last_spawn_cmd` still carries it. The
    text-only retry spawns through _start_llama_process directly, bypassing _spawn_and_wait's
    revocation: it must not relaunch --fit off and the plan's -ot."""
    from core.inference.llama_cpp import GgufLoadIntent
    from test_llama_cpp_placement import _REAL_POPEN, _backend, _write_gguf
    from test_offload_planner_launch import CARD_MIB, DENSE, MIB

    monkeypatch.setenv("UNSLOTH_SMART_OFFLOAD", "1")
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, CARD_MIB, CARD_MIB)])

    def read(_path):
        for key, value in DENSE.items():
            setattr(backend, key, value)

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024 * MIB
    del backend._can_estimate_kv
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_kv_unified": True,
        "supports_fit_ctx": True,
        "supports_cache_ram": True,
    }
    backend._available_system_memory_mib = lambda: 64 * 1024
    mmproj = _write_gguf(tmp_path / "mmproj.gguf", architecture = "clip")
    backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
    backend._planned_tensor_spill = lambda inputs, **_kw: _spilling_plan()
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        cmds.append(list(cmd))
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": None,
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        launched = cmds[-1] if cmds else []
        if "--mmproj" in launched:
            backend._stdout_lines = ["clip.cpp:4391: Unknown projector type"]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                is_vision = True,
                speculative_type = "off",
                n_ctx = 8192,
                n_parallel = 1,
            )
        )
    assert "-ot" in cmds[0] and "--mmproj" in cmds[0], cmds[0]
    text_only = cmds[-1]
    assert "--mmproj" not in text_only, text_only
    assert "-ot" not in text_only, text_only
    fits = [text_only[i + 1] for i, tok in enumerate(text_only) if tok == "--fit"]
    assert fits and fits[-1] == "on", text_only
