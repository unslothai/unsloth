"""`load_in_fp8=True` on a model with no pre-quantized FP8 repo must quantize it: get_model_name
hands back the caller's own name in that case, so the loader's `is None` check never fired and
the model loaded in bf16 with no message. Nothing is downloaded or loaded."""

from __future__ import annotations

import sys
import types

import pytest

loader = pytest.importorskip("unsloth.models.loader")
loader_utils = pytest.importorskip("unsloth.models.loader_utils")


class _Reached(Exception):
    pass


@pytest.mark.parametrize("entry", ["FastLanguageModel", "FastModel"])
def test_an_unmapped_model_reaches_the_fp8_quantizer(monkeypatch, entry):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(loader, "get_model_name", lambda name, **_kwargs: name)

    def check(*_args, **_kwargs):
        raise _Reached()

    monkeypatch.setattr(loader, "_get_fp8_mode_and_check_settings", check)
    with pytest.raises(_Reached):
        getattr(loader, entry).from_pretrained(
            "my-org/Some-Unmapped-8B",
            load_in_4bit = False,
            load_in_fp8 = True,
            fast_inference = False,
        )


def _fake_vllm(monkeypatch, version):
    module = types.ModuleType("vllm")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(
        loader_utils.importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "vllm" else None,
    )


@pytest.mark.parametrize(
    "resolved, fast_inference, vllm, expected",
    [
        ("unsloth/x-fp8", False, None, False),  # a pre-quantized repo was mapped
        ("my-org/x", False, "0.12.0", True),  # vLLM installed but not the one loading
        ("my-org/x", True, "0.12.0", False),  # vLLM quantizes on the fly
        ("my-org/x", True, "0.11.0", True),  # too old to quantize on the fly
    ],
)
def test_who_quantizes(monkeypatch, resolved, fast_inference, vllm, expected):
    if vllm is not None:
        _fake_vllm(monkeypatch, vllm)
    assert (
        loader_utils._fp8_needs_offline_quantize(resolved, "my-org/x", fast_inference) is expected
    )
