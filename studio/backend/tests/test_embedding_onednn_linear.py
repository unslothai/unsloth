# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: the CPU embedder's Linear layers run in oneDNN instead of rocm-openblas.dll (#14313).

The DLL lookup and torch.version.hip are faked; the oneDNN op itself is real, so this needs a torch with mkldnn.
"""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mkldnn.is_available():
    pytest.skip("torch built without oneDNN", allow_module_level = True)

import core.rag.embeddings as embeddings  # noqa: E402


@pytest.fixture
def rocm_windows(monkeypatch):
    monkeypatch.setattr(embeddings, "runtime_capped_openblas_loaded", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(384, 1536), torch.nn.GELU(), torch.nn.Linear(1536, 384, bias = False)
    )


def test_linears_run_in_onednn_with_the_same_vectors(rocm_windows):
    model = _model()
    x = torch.randn(4, 37, 384)
    with torch.inference_mode():
        expected = model(x)

    assert embeddings._route_linears_through_onednn(model) == 2

    assert all(type(m).__name__ == "OneDnnLinear" for m in model if isinstance(m, torch.nn.Linear))
    with torch.inference_mode():
        got = model(x)
    assert torch.allclose(got, expected, rtol = 1e-4, atol = 1e-5)


def test_nothing_changes_without_rocm_openblas(monkeypatch):
    monkeypatch.setattr(embeddings, "runtime_capped_openblas_loaded", lambda: False)
    model = _model()

    assert embeddings._route_linears_through_onednn(model) == 0
    assert type(model[0]) is torch.nn.Linear


def test_a_cuda_wheel_on_windows_keeps_its_blas(monkeypatch):
    monkeypatch.setattr(embeddings, "runtime_capped_openblas_loaded", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    model = _model()

    assert embeddings._route_linears_through_onednn(model) == 0
    assert type(model[0]) is torch.nn.Linear


def test_non_fp32_layers_are_left_alone(rocm_windows):
    model = _model().to(torch.bfloat16)

    assert embeddings._route_linears_through_onednn(model) == 0
    assert type(model[0]) is torch.nn.Linear


def test_a_probe_mismatch_keeps_the_stock_path(rocm_windows, monkeypatch):
    monkeypatch.setattr(torch, "allclose", lambda *a, **k: False)
    model = _model()

    assert embeddings._route_linears_through_onednn(model) == 0
    assert type(model[0]) is torch.nn.Linear


def test_a_missing_onednn_op_keeps_the_stock_path(rocm_windows, monkeypatch):
    class _NoMkldnn:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(torch.ops, "mkldnn", _NoMkldnn(), raising = False)
    model = _model()

    assert embeddings._route_linears_through_onednn(model) == 0
    assert type(model[0]) is torch.nn.Linear


def test_a_model_without_modules_is_ignored(rocm_windows):
    assert embeddings._route_linears_through_onednn(object()) == 0


def test_every_load_routes_the_model():
    source = (Path(__file__).resolve().parent.parent / "core/rag/embeddings.py").read_text(
        encoding = "utf-8"
    )
    get = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_get"
    )
    called = [
        node.func.id
        for node in ast.walk(get)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_route_linears_through_onednn" in called
