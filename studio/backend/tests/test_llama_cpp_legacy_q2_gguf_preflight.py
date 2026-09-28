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
import sys as _sys
from pathlib import Path as _Path

# The fixture writer lives beside this file; tests/ is a package, so put it on the path the
# way the other sibling-helper imports here do.
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

from test_gguf_metadata import _write_legacy_q2_offset_mismatch_gguf  # noqa: E402


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
    teardown = src.index("# ── Phase 1: kill old process")
    assert src.index("_refuse_legacy_q2_gguf_before_teardown(gguf_path)") < teardown
    assert src.index("_refuse_legacy_q2_gguf_before_teardown(_legacy_q2_probe)") < teardown


def _hub_load_backend(monkeypatch, order):
    backend = LlamaCppBackend()
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
    monkeypatch.setattr(
        LlamaCppBackend, "_remote_non_chat_gguf_refusal", classmethod(lambda cls, **_kw: None)
    )
    return backend


def test_a_hub_load_of_a_cached_legacy_file_is_refused_before_teardown(monkeypatch, tmp_path):
    # The Model Hub sends a repo id, never a path: the cached copy is what gets judged.
    gguf = _write_legacy_q2_offset_mismatch_gguf(
        tmp_path / "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
        mismatch_tensor = "dspark.fc.weight",
        architecture = "llama",
    )
    order: list[str] = []
    backend = _hub_load_backend(monkeypatch, order)
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_kw: str(gguf))

    with pytest.raises(ValueError) as exc:
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "prism-ml/Ternary-Bonsai-27B-gguf",
                hf_variant = "Q4_1",
                model_identifier = "prism-ml/Ternary-Bonsai-27B-gguf",
            )
        )
    assert "dspark.fc.weight" in str(exc.value)
    assert order == []


def test_a_hub_load_of_a_cached_mainline_file_reaches_the_teardown(monkeypatch, tmp_path):
    gguf = _write_legacy_q2_offset_mismatch_gguf(
        tmp_path / "Ternary-Bonsai-27B-Q2_g64.gguf", second_offset = 1152
    )
    order: list = []
    backend = _hub_load_backend(monkeypatch, order)
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_kw: str(gguf))

    class _Stop(Exception):
        pass

    def _stop():
        order.append("kill")
        raise _Stop

    monkeypatch.setattr(backend, "_kill_process", _stop)
    with pytest.raises(_Stop):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "prism-ml/Ternary-Bonsai-27B-gguf",
                hf_variant = "Q2_g64",
                model_identifier = "prism-ml/Ternary-Bonsai-27B-gguf",
            )
        )
    assert order == ["kill"]
