# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF decision line reports the --fit state the launch carries (#10821)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_llama_cpp_placement import _backend, _launch  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402

_GPU = [(0, 24 * 1024, 24 * 1024)]


def _decision_line(tmp_path, monkeypatch, **load_kwargs):
    lines = []
    real_info = llama_cpp.logger.info

    def info(msg, *a, **kw):
        lines.append(str(msg))
        return real_info(msg, *a, **kw)

    monkeypatch.setattr(llama_cpp.logger, "info", info)
    backend, gguf = _backend(tmp_path, vulkan = False, memory = _GPU)
    cmd = _launch(backend, gguf, n_ctx = 4096, **load_kwargs)["cmd"]
    (line,) = [l for l in lines if "GPUs free: " in l]
    return line, cmd


@pytest.mark.parametrize("gpu_layers", [0, 42])
def test_manual_layers_log_matches_fit_off_argv(tmp_path, monkeypatch, gpu_layers):
    line, cmd = _decision_line(
        tmp_path, monkeypatch, gpu_memory_mode = "manual", gpu_layers = gpu_layers
    )
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert line.endswith("--fit: off")
    assert "GPUs free: [] (manual mode)," in line


def test_manual_auto_layers_keeps_fit_on(tmp_path, monkeypatch):
    line, cmd = _decision_line(tmp_path, monkeypatch, gpu_memory_mode = "manual", gpu_layers = -1)
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert line.endswith("--fit: on")
    assert "GPUs free: [] (manual mode)," in line


def test_auto_mode_line_unlabelled(tmp_path, monkeypatch):
    line, _ = _decision_line(tmp_path, monkeypatch)
    assert "manual mode" not in line
    assert "GPUs free: [(0, " in line
