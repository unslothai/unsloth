# SPDX-License-Identifier: AGPL-3.0-only
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


@pytest.mark.parametrize("entry", ["FastLanguageModel", "FastModel"])
def test_an_already_quantized_checkpoint_skips_the_fp8_gate(monkeypatch, tmp_path, entry):
    # Base loaded such a checkpoint as it is; the H100 gate must not newly refuse it.
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
            }
        )
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(loader, "get_model_name", lambda name, **_kwargs: name)

    def check(*_args, **_kwargs):
        raise _Reached()

    monkeypatch.setattr(loader, "_get_fp8_mode_and_check_settings", check)
    monkeypatch.setattr(loader, "_offline_quantize_to_fp8", check)
    try:
        getattr(loader, entry).from_pretrained(
            str(tmp_path), load_in_4bit = False, load_in_fp8 = True, fast_inference = False
        )
    except _Reached:
        pytest.fail("an already quantized checkpoint was sent to the fp8 quantizer")
    except Exception:
        pass  # No weights here: the load fails later, past the gate.
