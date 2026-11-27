# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pass-through --fit-target is honoured on a load that fits (#19240).

llama.cpp reads --fit-target only while its fitter runs (common_fit_apply returns at once
under --fit off), so a fitting load must keep --fit on for a user margin to apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402

GIB = 1024**3
CAPS = {"supports_fit_target": True, "supports_fit_ctx": True, "supports_kv_unified": True}
# GPU 0 drives the monitors, GPU 1 is dedicated.
TWO = [(0, 21000, 24564), (1, 24400, 24564)]


def _launch(
    tmp_path,
    model_gib,
    memory = TWO,
    **kw,
):
    backend, gguf = placement._backend(tmp_path, vulkan = False, memory = list(memory))
    backend.probe_server_capabilities = lambda _binary = None: dict(CAPS)
    backend._get_gguf_size_bytes = lambda _path: int(model_gib * GIB)
    return placement._launch(backend, gguf, **kw)


def _last(cmd, flag):
    hits = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == flag]
    return hits[-1] if hits else None


def test_a_fitting_split_load_keeps_the_fitter_for_a_per_gpu_margin(tmp_path):
    r = _launch(tmp_path, 30, extra_args = ["--fit-target", "3072,256"])
    assert _last(r["cmd"], "--fit") == "on"
    assert _last(r["cmd"], "--fit-target") == "3072,256"
    assert r["env"].get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert "-ngl" not in r["cmd"]


def test_a_list_leaves_a_one_card_pick_alone(tmp_path):
    # The child renumbers a narrowed pick, so the list would land on the dedicated card.
    r = _launch(tmp_path, 10, extra_args = ["--fit-target", "3072,256"])
    assert r["cmd"][r["cmd"].index("--fit") + 1] == "off"
    assert r["env"].get("CUDA_VISIBLE_DEVICES") == "1"


@pytest.mark.parametrize("model_gib, mask", [(30, "0,1"), (10, "1")])
def test_a_scalar_margin_keeps_the_pick(tmp_path, model_gib, mask):
    r = _launch(tmp_path, model_gib, extra_args = ["--fit-target=2048"])
    assert _last(r["cmd"], "--fit") == "on"
    assert r["env"].get("CUDA_VISIBLE_DEVICES") == mask


def test_an_explicit_gpu_pick_stays_pinned(tmp_path):
    r = _launch(tmp_path, 30, extra_args = ["--fit-target", "3072,256"], gpu_ids = [0, 1])
    assert _last(r["cmd"], "--fit") == "on"
    assert r["env"].get("CUDA_VISIBLE_DEVICES") == "0,1"


def test_a_fitting_load_is_not_reported_partial(tmp_path):
    backend, gguf = placement._backend(tmp_path, vulkan = False, memory = TWO)
    backend.probe_server_capabilities = lambda _binary = None: dict(CAPS)
    backend._get_gguf_size_bytes = lambda _path: 30 * GIB
    # refuse_partial_gpu_fit raises on a partial verdict; a user margin on a fitting plan is not one.
    r = placement._launch(
        backend, gguf, extra_args = ["--fit-target", "3072,256"], refuse_partial_gpu_fit = True
    )
    assert _last(r["cmd"], "--fit") == "on"


@pytest.mark.parametrize(
    "extra",
    [
        None,
        ["--top-k", "5"],
        # The user owns the mode or the layer count: their flags decide, as before.
        ["--fit-target", "3072,256", "--fit", "off"],
        ["--fit-target", "3072,256", "-ngl", "99"],
    ],
)
def test_other_fitting_loads_are_unchanged(tmp_path, extra):
    kw = {"extra_args": extra} if extra else {}
    r = _launch(tmp_path, 30, **kw)
    assert r["cmd"][r["cmd"].index("--fit") + 1] == "off"
    assert r["env"].get("CUDA_VISIBLE_DEVICES") == "0,1"


def test_manual_layers_keep_fit_off(tmp_path):
    r = _launch(
        tmp_path,
        30,
        extra_args = ["--fit-target", "3072,256"],
        gpu_memory_mode = "manual",
        gpu_layers = 20,
    )
    assert _last(r["cmd"], "--fit") == "off"


def test_the_helper_reads_argv_only():
    needs = lc.LlamaCppBackend._pass_through_fit_target_needs_fitter
    assert needs(["--fit-target", "1024"])
    assert needs(["-fitt=3072,0"])
    assert not needs(None)
    assert not needs(["--fit-target", "1024", "--fit", "on"])
    assert not needs(["--fit-target", "1024", "--n-gpu-layers", "10"])
    assert not needs(["--fit-target"])
    assert not needs(["--fit-target", "abc"])
    assert not needs(["--fit-target", "3072,256"], spans_every_card = False)
    assert needs(["--fit-target", "2048"], spans_every_card = False)
