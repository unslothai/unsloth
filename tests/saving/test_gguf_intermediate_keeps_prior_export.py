# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A throwaway GGUF intermediate never replaces a file already in the output directory. A Q4-only
(or F16 + Q4) export converts to BF16 first on bf16 hardware, and F16 on fp16 hardware; placing it
replaced a prior `.BF16.gguf` / `.F16.gguf` export, and the cleanup then deleted it."""

from __future__ import annotations

import contextlib
import os

import pytest

save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")

USER_BYTES = b"GGUF prior export"


class _Converter:
    def __init__(self, shards = 1):
        self.shards = shards

    def __call__(self, **kwargs):
        qtype = kwargs["quantization_type"].upper()
        if self.shards == 1:
            names = [f"{kwargs['model_name']}.{qtype}.gguf"]
        else:
            names = [
                f"{kwargs['model_name']}.{qtype}-{i:05d}-of-{self.shards:05d}.gguf"
                for i in range(1, self.shards + 1)
            ]
        for name in names:
            with open(name, "wb") as handle:
                handle.write(b"GGUF intermediate")
        return names, False


def _wire(monkeypatch, converter, quantized, bf16):
    monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
    monkeypatch.setattr(
        save_mod,
        "_download_convert_hf_to_gguf",
        lambda: ("convert.py", {"LlamaForCausalLM"}, set()),
    )
    monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
    monkeypatch.setattr(save_mod, "convert_to_gguf", converter)
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    if hasattr(save_mod, "is_bfloat16_supported"):
        monkeypatch.setattr(save_mod, "is_bfloat16_supported", lambda *a, **k: bf16)

    def _quantize(input_gguf, output_gguf, quant_type, **kw):
        with open(input_gguf, "rb") as handle:
            assert handle.read() == b"GGUF intermediate", "quantized the wrong input"
        quantized.append((input_gguf, output_gguf))
        with open(output_gguf, "wb") as handle:
            handle.write(b"GGUF " + quant_type.encode())
        return output_gguf

    monkeypatch.setattr(save_mod, "quantize_gguf", _quantize)


def _export(
    tmp_path,
    monkeypatch,
    methods,
    bf16,
    prior,
    shards = 1,
):
    monkeypatch.chdir(tmp_path)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    gguf_directory = tmp_path / "model_gguf"
    gguf_directory.mkdir()
    for name in prior:
        (gguf_directory / name).write_bytes(USER_BYTES)
    quantized = []
    _wire(monkeypatch, _Converter(shards), quantized, bf16)
    files, _, _ = save_mod.save_to_gguf(
        model_name = "MyModel",
        model_type = "llama",
        model_dtype = "bfloat16",
        model_directory = str(model_directory),
        quantization_method = methods,
        gguf_directory = str(gguf_directory),
    )
    return gguf_directory, files, quantized


@pytest.mark.parametrize(
    "bf16, intermediate", [(True, "BF16"), (False, "F16")], ids = ["bf16-host", "fp16-host"]
)
@pytest.mark.parametrize("methods", [["q4_k_m"], ["q4_k_m", "q8_0"]])
def test_a_prior_export_at_the_intermediate_name_survives(
    monkeypatch, tmp_path, methods, bf16, intermediate
):
    prior = f"MyModel.{intermediate}.gguf"
    gguf_directory, files, quantized = _export(tmp_path, monkeypatch, methods, bf16, [prior])
    assert (gguf_directory / prior).read_bytes() == USER_BYTES
    assert sorted(os.path.basename(o) for _, o in quantized) == sorted(
        f"MyModel.{m.upper()}.gguf" for m in methods
    )
    # The intermediate itself is still cleaned up, and nothing but the outputs and the prior file is left.
    assert sorted(os.listdir(gguf_directory)) == sorted(
        [prior] + [f"MyModel.{m.upper()}.gguf" for m in methods]
    )
    assert prior not in {os.path.basename(f) for f in files}


def test_f16_plus_q4_keeps_a_prior_bf16_export(monkeypatch, tmp_path):
    gguf_directory, files, _ = _export(
        tmp_path, monkeypatch, ["f16", "q4_k_m"], True, ["MyModel.BF16.gguf"]
    )
    assert (gguf_directory / "MyModel.BF16.gguf").read_bytes() == USER_BYTES
    assert {"MyModel.F16.gguf", "MyModel.Q4_K_M.gguf"} <= set(os.listdir(gguf_directory))


def test_a_sharded_intermediate_keeps_prior_shards_and_its_split_suffix(monkeypatch, tmp_path):
    prior = ["MyModel.BF16-00001-of-00002.gguf", "MyModel.BF16-00002-of-00002.gguf"]
    gguf_directory, _, quantized = _export(tmp_path, monkeypatch, ["q4_k_m"], True, prior, shards = 2)
    for name in prior:
        assert (gguf_directory / name).read_bytes() == USER_BYTES
    ((source, _),) = quantized
    assert os.path.basename(source).endswith("-00001-of-00002.gguf")
    assert sorted(os.listdir(gguf_directory)) == sorted(prior + ["MyModel.Q4_K_M.gguf"])


def test_a_requested_intermediate_still_replaces_its_own_name(monkeypatch, tmp_path):
    """Asking for bf16 again is a re-export of that file, as before."""
    gguf_directory, files, _ = _export(
        tmp_path, monkeypatch, ["bf16", "q4_k_m"], True, ["MyModel.BF16.gguf"]
    )
    assert (gguf_directory / "MyModel.BF16.gguf").read_bytes() == b"GGUF intermediate"
    assert str(gguf_directory / "MyModel.BF16.gguf") in files


def test_without_a_prior_file_the_names_are_unchanged(monkeypatch, tmp_path):
    _, _, quantized = _export(tmp_path, monkeypatch, ["q4_k_m"], True, [])
    ((source, _),) = quantized
    assert os.path.basename(source) == "MyModel.BF16.gguf"
