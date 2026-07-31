# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only routing tests for the single-pass GGUF export and parallel quantization.

With convert/quantize monkeypatched, verify save_to_gguf's pass planning:
- a single directly-convertible output type (f32/f16/bf16/q8_0) converts in ONE pass
  with no llama-quantize step and no 16-bit intermediate,
- k-quants and imatrix runs keep the two-pass route,
- multiple quantize passes run through the bounded pool with request order preserved,
- quantize failures still raise the actionable RuntimeError.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

import pytest

import unsloth.save as save_mod


# -- _choose_first_conversion (pure planning logic) ----------------------------------------


@pytest.mark.parametrize(
    "methods, model_dtype, expected",
    [
        (["q8_0"], "f16", "q8_0"),  # default "fast_quantized" path: single pass
        (["q8_0", "q8_0"], "bf16", "q8_0"),  # duplicates collapse to a single pass
        (["f32"], "f16", "f32"),  # 16/32-bit outputs convert directly too
        (["bf16"], "bf16", "bf16"),
        (["q4_k_m"], "f16", "f16"),  # k-quants need a 16-bit base
        (["q4_k_m", "q8_0"], "bf16", "bf16"),  # mixes need the shared base
        (["q8_0", "f16"], "f16", "f16"),
    ],
)
def test_choose_first_conversion(methods, model_dtype, expected):
    assert save_mod._choose_first_conversion(methods, model_dtype) == expected


def test_choose_first_conversion_imatrix_forces_two_pass():
    # Only llama-quantize can apply an imatrix, so q8_0-only must keep the 16-bit base.
    assert save_mod._choose_first_conversion(["q8_0"], "f16", has_imatrix = True) == "f16"


# -- save_to_gguf pass planning (mocked convert/quantize) -----------------------------------


class _Harness:
    """Monkeypatched convert/quantize recording calls and creating real files."""

    def __init__(
        self,
        monkeypatch,
        tmp_path,
        quantize_delays = None,
        quantize_error = None,
    ):
        self.tmp_path = tmp_path
        self.convert_calls = []
        self.quantize_calls = []
        self.active = 0
        self.max_concurrency = 0
        self._lock = threading.Lock()
        self._delays = quantize_delays or {}
        self._error = quantize_error

        monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
        monkeypatch.setattr(
            save_mod,
            "_download_convert_hf_to_gguf",
            lambda: (str(tmp_path / "convert.py"), {"LlamaForCausalLM"}, set()),
        )
        monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
        monkeypatch.setattr(save_mod, "convert_to_gguf", self._convert)
        monkeypatch.setattr(save_mod, "quantize_gguf", self._quantize)

    def _convert(self, **kwargs):
        self.convert_calls.append(kwargs)
        suffix = kwargs["quantization_type"]
        if suffix == "None":
            suffix = kwargs["model_dtype"]
        out = self.tmp_path / f"{kwargs['model_name']}.{suffix.upper()}.gguf"
        out.write_bytes(b"GGUF")
        return [str(out)], False

    def _quantize(
        self,
        input_gguf,
        output_gguf,
        quant_type,
        imatrix = None,
        n_threads = None,
        **kw,
    ):
        with self._lock:
            self.active += 1
            self.max_concurrency = max(self.max_concurrency, self.active)
        try:
            if self._error is not None:
                raise self._error
            time.sleep(self._delays.get(quant_type, 0.02))
            self.quantize_calls.append({"quant_type": quant_type, "n_threads": n_threads})
            with open(output_gguf, "wb") as f:
                f.write(b"GGUF")
            return output_gguf
        finally:
            with self._lock:
                self.active -= 1


def _run(tmp_path, methods, **kwargs):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir(exist_ok = True)
    return save_mod.save_to_gguf(
        model_name = "testmodel",
        model_type = "llama",
        model_dtype = "float16",
        model_directory = str(model_dir),
        quantization_method = methods,
        **kwargs,
    )


def test_q8_0_only_is_single_pass(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    locations, want_full_precision, _ = _run(tmp_path, ["q8_0"])

    assert len(h.convert_calls) == 1
    assert h.convert_calls[0]["quantization_type"] == "q8_0"
    assert h.quantize_calls == [], "single-pass export must not launch llama-quantize"
    assert want_full_precision is True, "the converted file IS the requested output"
    assert len(locations) == 1 and locations[0].endswith("testmodel.Q8_0.gguf")
    assert os.path.exists(locations[0])


def test_fast_quantized_alias_is_single_pass(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    _run(tmp_path, "fast_quantized")  # the default of save_pretrained_gguf
    assert h.convert_calls[0]["quantization_type"] == "q8_0"
    assert h.quantize_calls == []


def test_k_quant_keeps_two_pass(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    locations, want_full_precision, _ = _run(tmp_path, ["q4_k_m"])

    assert h.convert_calls[0]["quantization_type"] == "f16"
    assert [c["quant_type"] for c in h.quantize_calls] == ["q4_k_m"]
    assert want_full_precision is False
    # The 16-bit intermediate must be cleaned up.
    assert len(locations) == 1 and locations[0].endswith("testmodel.Q4_K_M.gguf")


def test_mixed_methods_share_16bit_base(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    _run(tmp_path, ["q4_k_m", "q8_0"])
    assert h.convert_calls[0]["quantization_type"] == "f16"
    assert sorted(c["quant_type"] for c in h.quantize_calls) == ["q4_k_m", "q8_0"]


def test_parallel_quants_preserve_request_order(monkeypatch, tmp_path):
    # First method is the slowest: completion order != request order.
    h = _Harness(
        monkeypatch, tmp_path, quantize_delays = {"q4_k_m": 0.3, "q5_k_m": 0.05, "q6_k": 0.01}
    )
    locations, _, _ = _run(tmp_path, ["q4_k_m", "q5_k_m", "q6_k"])

    assert h.max_concurrency == 2, "quantize passes should overlap, bounded at 2"
    quant_names = [os.path.basename(l) for l in locations if "F16" not in l]
    assert quant_names == [
        "testmodel.Q6_K.gguf",  # list is reversed by the cleanup block, as before
        "testmodel.Q5_K_M.gguf",
        "testmodel.Q4_K_M.gguf",
    ]
    assert all(
        c["n_threads"] is not None for c in h.quantize_calls
    ), "parallel workers must split the thread budget explicitly"


def test_parallel_quants_env_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_PARALLEL_GGUF_QUANTS", "0")
    h = _Harness(monkeypatch, tmp_path, quantize_delays = {"q4_k_m": 0.05, "q5_k_m": 0.05})
    _run(tmp_path, ["q4_k_m", "q5_k_m"])
    assert h.max_concurrency == 1


def test_duplicate_methods_quantize_once(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path)
    _run(tmp_path, ["q4_k_m", "q4_k_m"])
    assert [c["quant_type"] for c in h.quantize_calls] == ["q4_k_m"]


def test_quantize_failure_raises_actionable_error(monkeypatch, tmp_path):
    h = _Harness(monkeypatch, tmp_path, quantize_error = OSError("disk full"))
    with pytest.raises(RuntimeError, match = "Quantization failed"):
        _run(tmp_path, ["q4_k_m", "q5_k_m"])
