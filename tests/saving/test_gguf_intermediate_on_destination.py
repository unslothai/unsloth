# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The intermediate GGUF is written on the destination filesystem, so placing it is a rename."""

from __future__ import annotations

import contextlib
import os

import pytest

save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")


class _Converter:
    """Writes where convert_to_gguf would: a bare name resolves against the CWD."""

    def __init__(self, fail_after_write = False):
        self.fail_after_write = fail_after_write
        self.outfiles = []

    def __call__(self, **kwargs):
        model_name = kwargs["model_name"]
        qtype = kwargs["quantization_type"]
        if qtype == "None":
            qtype = kwargs.get("model_dtype", "bf16")
        out = model_name if model_name.endswith(".gguf") else f"{model_name}.{qtype.upper()}.gguf"
        self.outfiles.append(os.path.abspath(out))
        with open(out, "wb") as handle:
            handle.write(b"GGUF" + b"\0" * 4096)
        if self.fail_after_write:
            raise RuntimeError("converter died mid-write")
        return [out], False


def _wire(monkeypatch, converter, quantized):
    monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
    monkeypatch.setattr(
        save_mod,
        "_download_convert_hf_to_gguf",
        lambda: ("convert.py", {"LlamaForCausalLM"}, set()),
    )
    monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
    monkeypatch.setattr(save_mod, "convert_to_gguf", converter)

    def _quantize(input_gguf, output_gguf, quant_type, **kw):
        quantized.append((input_gguf, output_gguf))
        with open(output_gguf, "wb") as handle:
            handle.write(b"GGUF")
        return output_gguf

    monkeypatch.setattr(save_mod, "quantize_gguf", _quantize)


def _copies(monkeypatch):
    calls = []
    for name in ("copy2", "copyfile", "copy"):
        real = getattr(save_mod.shutil, name)

        def _spy(
            *a,
            _real = real,
            _name = name,
            **kw,
        ):
            calls.append(_name)
            return _real(*a, **kw)

        monkeypatch.setattr(save_mod.shutil, name, _spy)
    return calls


def _layout(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    model_directory = tmp_path / "out" / "model"
    model_directory.mkdir(parents = True)
    return cwd, model_directory, str(model_directory) + "_gguf"


def _save(model_directory, methods):
    return save_mod.save_to_gguf(
        model_name = "MyModel",
        model_type = "llama",
        model_dtype = "bfloat16",
        model_directory = str(model_directory),
        quantization_method = methods,
    )


@pytest.mark.parametrize("methods", [["q4_k_m"], ["q4_k_m", "q8_0"], ["bf16"]])
def test_intermediate_is_written_inside_the_gguf_directory(monkeypatch, tmp_path, methods):
    cwd, model_directory, gguf_directory = _layout(tmp_path, monkeypatch)
    converter = _Converter()
    quantized = []
    _wire(monkeypatch, converter, quantized)
    copies = _copies(monkeypatch)
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    files, _, _ = _save(model_directory, methods)

    # Converter wrote on the destination filesystem, inside the output folder.
    (outfile,) = converter.outfiles
    assert os.path.dirname(os.path.dirname(outfile)) == os.path.abspath(gguf_directory)
    # Final names unchanged, nothing staged left behind, nothing in the CWD, no byte copy.
    assert sorted(os.listdir(gguf_directory)) == sorted(os.path.basename(f) for f in files)
    assert os.path.basename(outfile) == "MyModel.BF16.gguf"
    assert os.listdir(cwd) == []
    assert copies == []
    for source, _ in quantized:
        assert source == os.path.join(gguf_directory, "MyModel.BF16.gguf")


def test_a_failed_conversion_leaves_no_partial_file_anywhere(monkeypatch, tmp_path):
    cwd, model_directory, gguf_directory = _layout(tmp_path, monkeypatch)
    _wire(monkeypatch, _Converter(fail_after_write = True), [])
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    with pytest.raises(RuntimeError, match = "converter died"):
        _save(model_directory, ["q4_k_m"])

    assert os.listdir(cwd) == []
    # The folder did not exist before, so it is not left behind empty either.
    assert not os.path.exists(gguf_directory)


def test_a_failed_conversion_keeps_an_existing_gguf_directory(monkeypatch, tmp_path):
    cwd, model_directory, gguf_directory = _layout(tmp_path, monkeypatch)
    os.makedirs(gguf_directory)
    keep = os.path.join(gguf_directory, "earlier.Q8_0.gguf")
    with open(keep, "wb") as handle:
        handle.write(b"GGUF")
    _wire(monkeypatch, _Converter(fail_after_write = True), [])
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    with pytest.raises(RuntimeError):
        _save(model_directory, ["q4_k_m"])

    assert os.listdir(gguf_directory) == ["earlier.Q8_0.gguf"]
    assert os.listdir(cwd) == []


def test_an_existing_output_of_the_same_name_is_replaced(monkeypatch, tmp_path):
    cwd, model_directory, gguf_directory = _layout(tmp_path, monkeypatch)
    os.makedirs(gguf_directory)
    stale = os.path.join(gguf_directory, "MyModel.BF16.gguf")
    with open(stale, "wb") as handle:
        handle.write(b"stale")
    _wire(monkeypatch, _Converter(), [])
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    _save(model_directory, ["bf16"])

    with open(stale, "rb") as handle:
        assert handle.read().startswith(b"GGUF")
    assert os.listdir(gguf_directory) == ["MyModel.BF16.gguf"]


def test_without_a_staging_folder_the_cwd_placement_is_kept(monkeypatch, tmp_path):
    """An output folder that cannot hold the staging folder falls back to the old CWD write and move."""
    cwd, model_directory, gguf_directory = _layout(tmp_path, monkeypatch)
    converter = _Converter()
    _wire(monkeypatch, converter, [])
    monkeypatch.setattr(save_mod, "_gguf_conversion_staging", lambda directory: (None, False))
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    _save(model_directory, ["bf16"])

    assert converter.outfiles == [os.path.join(str(cwd), "MyModel.BF16.gguf")]
    assert os.listdir(gguf_directory) == ["MyModel.BF16.gguf"]
    assert os.listdir(cwd) == []


def test_place_gguf_output_falls_back_to_move_across_filesystems(monkeypatch, tmp_path):
    src = tmp_path / "a.gguf"
    src.write_bytes(b"GGUF")
    dst = tmp_path / "b.gguf"

    def _exdev(*a, **k):
        raise OSError(18, "Invalid cross-device link")

    moved = []
    monkeypatch.setattr(save_mod.os, "replace", _exdev)
    real_move = save_mod.shutil.move
    monkeypatch.setattr(
        save_mod.shutil, "move", lambda s, d: (moved.append((s, d)), real_move(s, d))[1]
    )
    save_mod._place_gguf_output(str(src), str(dst))
    assert moved == [(str(src), str(dst))]
    assert dst.read_bytes() == b"GGUF" and not src.exists()
