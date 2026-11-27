# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import contextlib
import os

import pytest

save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")

PRIOR_BYTES = b"GGUF prior working export"


def _converter(**kwargs):
    name = f"{kwargs['model_name']}.{kwargs['quantization_type'].upper()}.gguf"
    with open(name, "wb") as handle:
        handle.write(b"GGUF intermediate")
    return [name], False


def _wire(monkeypatch, failing):
    monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
    monkeypatch.setattr(
        save_mod,
        "_download_convert_hf_to_gguf",
        lambda: ("convert.py", {"LlamaForCausalLM"}, set()),
    )
    monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
    monkeypatch.setattr(save_mod, "convert_to_gguf", _converter)
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    if hasattr(save_mod, "is_bfloat16_supported"):
        monkeypatch.setattr(save_mod, "is_bfloat16_supported", lambda *a, **k: True)

    def _quantize(input_gguf, output_gguf, quant_type, **kw):
        # llama-quantize truncates its output on open and writes tensors as it goes.
        with open(output_gguf, "wb") as handle:
            handle.write(b"GGUF partial")
            if quant_type in failing:
                raise RuntimeError(f"llama-quantize was killed while writing {quant_type}")
            handle.write(b" " + quant_type.encode())
        return output_gguf

    monkeypatch.setattr(save_mod, "quantize_gguf", _quantize)


def _export(tmp_path, monkeypatch, methods, failing, parallel):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UNSLOTH_PARALLEL_GGUF_QUANTS", "1" if parallel else "0")
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    gguf_directory = tmp_path / "model_gguf"
    gguf_directory.mkdir()
    for method in methods:
        (gguf_directory / f"MyModel.{method.upper()}.gguf").write_bytes(PRIOR_BYTES)
    _wire(monkeypatch, failing)
    return gguf_directory, lambda: save_mod.save_to_gguf(
        model_name = "MyModel",
        model_type = "llama",
        model_dtype = "bfloat16",
        model_directory = str(model_directory),
        quantization_method = methods,
        gguf_directory = str(gguf_directory),
    )


@pytest.mark.parametrize(
    "methods, parallel",
    [(["q4_k_m"], False), (["q4_k_m", "q5_k_m"], False), (["q4_k_m", "q5_k_m"], True)],
    ids = ["single", "sequential", "parallel"],
)
def test_a_failed_requant_keeps_the_prior_gguf(monkeypatch, tmp_path, methods, parallel):
    gguf_directory, export = _export(tmp_path, monkeypatch, methods, {"q4_k_m"}, parallel)
    with pytest.raises(RuntimeError, match = "Quantization failed"):
        export()
    assert (gguf_directory / "MyModel.Q4_K_M.gguf").read_bytes() == PRIOR_BYTES
    for method in methods[1:]:
        assert (gguf_directory / f"MyModel.{method.upper()}.gguf").read_bytes() in (
            PRIOR_BYTES,
            b"GGUF partial " + method.encode(),
        )
    assert not [name for name in os.listdir(gguf_directory) if not name.endswith(".gguf")]


def test_a_successful_requant_replaces_the_prior_gguf(monkeypatch, tmp_path):
    gguf_directory, export = _export(tmp_path, monkeypatch, ["q4_k_m"], set(), False)
    files, _, _ = export()
    output = gguf_directory / "MyModel.Q4_K_M.gguf"
    assert output.read_bytes() == b"GGUF partial q4_k_m"
    assert str(output) in files
    assert sorted(os.listdir(gguf_directory)) == ["MyModel.Q4_K_M.gguf"]


def test_a_near_limit_name_still_requantizes(monkeypatch, tmp_path):
    # 240 + ".Q4_K_M.gguf" is 252 bytes: a ".tmp" suffix would pass NAME_MAX.
    gguf_directory, _ = _export(tmp_path, monkeypatch, [], set(), False)
    name = "M" * 240
    (gguf_directory / f"{name}.Q4_K_M.gguf").write_bytes(PRIOR_BYTES)
    save_mod.save_to_gguf(
        model_name = name,
        model_type = "llama",
        model_dtype = "bfloat16",
        model_directory = str(tmp_path / "model"),
        quantization_method = ["q4_k_m"],
        gguf_directory = str(gguf_directory),
    )
    assert (gguf_directory / f"{name}.Q4_K_M.gguf").read_bytes() == b"GGUF partial q4_k_m"
    assert sorted(os.listdir(gguf_directory)) == [f"{name}.Q4_K_M.gguf"]


@pytest.mark.parametrize("windows", [False, True])
def test_partial_name_is_a_short_sibling_near_path_limits(monkeypatch, windows):
    monkeypatch.setattr(save_mod, "IS_WINDOWS", windows)
    short = os.path.join("out", "MyModel.Q4_K_M.gguf")
    assert save_mod._partial_gguf_path(short) == short + ".tmp"
    finals = [os.path.join("out", "M" * 240 + ".Q4_K_M.gguf")]
    if windows:
        finals.append(os.path.join("D" * 200, "E" * 40, "MyModel.Q4_K_M.gguf"))
    for final in finals:
        partial = save_mod._partial_gguf_path(final)
        assert os.path.dirname(partial) == os.path.dirname(final)
        assert len(os.path.basename(partial)) < 16
