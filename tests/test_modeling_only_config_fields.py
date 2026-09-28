# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""K-EXAONE 2.0 fields (`swiglu_limits`, per-layer `sliding_windows`) ignored by transformers without huggingface/transformers#47802."""

import inspect

import pytest

transformers = pytest.importorskip("transformers")
if getattr(transformers, "ExaoneMoeConfig", None) is None:
    pytest.skip("this transformers has no exaone_moe", allow_module_level = True)

from unsloth.models import loader  # noqa: E402

N = 8


def _k2():
    lt = ["full_attention" if i % 4 == 0 else "sliding_attention" for i in range(N)]
    return transformers.ExaoneMoeConfig(
        num_hidden_layers = N,
        layer_types = lt,
        sliding_windows = [
            0 if t == "full_attention" else (4096 if i == 1 else 128) for i, t in enumerate(lt)
        ],
        swiglu_limits = [0.0] * (N - 2) + [7.0, 7.0],
    )


def _k1():
    lt = ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(N)]
    return transformers.ExaoneMoeConfig(
        num_hidden_layers = N,
        layer_types = lt,
        sliding_window = 128,
        sliding_windows = [0 if t == "full_attention" else 128 for t in lt],
    )


def _modeling_supports():
    import transformers.models.exaone_moe.modeling_exaone_moe as m
    src = inspect.getsource(m)
    return "swiglu_limits" in src and "sliding_windows" in src


def test_k_exaone_2_refused_when_transformers_ignores_its_fields(monkeypatch):
    if _modeling_supports():
        pytest.skip("this transformers implements K-EXAONE 2.0")
    monkeypatch.delenv("UNSLOTH_ALLOW_IGNORED_CONFIG", raising = False)
    config = _k2()
    assert config.sliding_window == 4096
    with pytest.raises(RuntimeError, match = "swiglu_limits, sliding_windows"):
        loader._raise_if_modeling_ignores_config(config, ["exaone_moe"])


def test_override_only_warns(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_IGNORED_CONFIG", "1")
    loader._raise_if_modeling_ignores_config(_k2(), ["exaone_moe"])


def test_k_exaone_1_still_loads(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_IGNORED_CONFIG", raising = False)
    loader._raise_if_modeling_ignores_config(_k1(), ["exaone_moe"])


def test_supported_transformers_is_not_refused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_IGNORED_CONFIG", raising = False)
    real = inspect.getsource
    monkeypatch.setattr(
        inspect, "getsource", lambda obj: real(obj) + "\n# swiglu_limits sliding_windows\n"
    )
    loader._raise_if_modeling_ignores_config(_k2(), ["exaone_moe"])


def test_other_model_types_untouched():
    loader._raise_if_modeling_ignores_config(_k2(), ["llama"])
    loader._raise_if_modeling_ignores_config(None, ["exaone_moe"])


def test_adapter_base_config_is_checked():
    import ast
    tree = ast.parse(inspect.getsource(loader))
    checked = set()
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"):
            for node in ast.walk(fn):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "is_peft":
                    if any(
                        isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_raise_if_modeling_ignores_config"
                        for c in ast.walk(node)
                    ):
                        checked.add(cls.name)
    assert {"FastLanguageModel", "FastModel"} <= checked
