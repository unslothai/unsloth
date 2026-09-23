# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-teardown refusal for Prism legacy Q2_0 GGUFs (#11259 / #11264).

The metadata probe is covered in ``test_gguf_metadata``; this file proves ``load_model``
runs it on the reported ``*-dspark-Q4_1.gguf`` path before Phase 1 tears down the resident
chat model.
"""

from __future__ import annotations

import inspect

import pytest

import core.inference.llama_cpp as llama_cpp_module
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from test_gguf_metadata import _write_legacy_q2_offset_mismatch_gguf


def test_reported_dspark_q4_1_refused_before_teardown(monkeypatch, tmp_path):
    gguf = _write_legacy_q2_offset_mismatch_gguf(
        tmp_path / "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
        mismatch_tensor = "dspark.fc.weight",
        architecture = "llama",
    )
    backend = LlamaCppBackend()
    order: list[str] = []
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: False)
    monkeypatch.setattr(backend, "_backend_lacks_gpu_lib", lambda _binary = None: False)
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: False)
    monkeypatch.setattr(backend, "_kill_process", lambda: order.append("kill"))
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_kv_unified": True}),
    )

    with pytest.raises(ValueError) as exc:
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "prism-ml/Ternary-Bonsai-27B-gguf",
            )
        )

    msg = str(exc.value)
    assert "Q2_g64" in msg
    assert "dspark.fc.weight" in msg
    assert "enough memory" not in msg.lower()
    assert order == []


def test_the_legacy_q2_probe_sits_above_the_teardown_in_source():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    probe = src.index("gguf_mainline_q2_offset_mismatch(gguf_path)")
    teardown = src.index("# ── Phase 1: kill old process")
    assert probe < teardown
