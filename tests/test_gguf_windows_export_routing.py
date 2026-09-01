# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""End-to-end routing proof for #7897, run on Linux.

save.py's module-global ``os`` is swapped for a shim whose pure path functions come
from ``ntpath``, so ``save_to_gguf`` does the exact join arithmetic a Windows host
would. The assertion is on the ``output_gguf`` reaching ``quantize_gguf``, which is
where the GGUF escaped the export directory.
"""

from __future__ import annotations

import contextlib
import ntpath
import posixpath

import pytest

# test_gguf_model_basename.py covers the pure basename contract with no deps;
# this file drives the real save_to_gguf, so it needs the ML stack.
save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")


# OS shim: pure path ops from ntpath/posixpath, filesystem ops stay real.
_PURE = {
    "join",
    "basename",
    "dirname",
    "split",
    "splitdrive",
    "splitext",
    "normpath",
    "normcase",
    "isabs",
    "commonpath",
    "sep",
    "altsep",
}


class _PathShim:
    def __init__(self, flavour):
        self._f = flavour

    def __getattr__(self, name):
        import os as _os
        return getattr(self._f if name in _PURE else _os.path, name)


class _OsShim:
    """`os` with a swapped `.path`; makedirs/listdir/remove stay real."""

    def __init__(self, flavour):
        self.path = _PathShim(flavour)

    def __getattr__(self, name):
        import os as _os
        return getattr(_os, name)


class _Harness:
    """convert/quantize mocked, reproducing unsloth_zoo's real output naming."""

    def __init__(self, monkeypatch, tmp_path, flavour):
        self.tmp_path = tmp_path
        self.flavour = flavour
        self.convert_calls = []
        self.quantize_calls = []
        self.initial_names = []

        monkeypatch.setattr(save_mod, "check_llama_cpp", lambda: ("llama-quantize", "convert.py"))
        monkeypatch.setattr(
            save_mod,
            "_download_convert_hf_to_gguf",
            lambda: (str(tmp_path / "convert.py"), {"LlamaForCausalLM"}, set()),
        )
        monkeypatch.setattr(save_mod, "use_local_gguf", contextlib.nullcontext)
        monkeypatch.setattr(save_mod, "convert_to_gguf", self._convert)
        monkeypatch.setattr(save_mod, "quantize_gguf", self._quantize)
        monkeypatch.setattr(save_mod, "os", _OsShim(flavour))

    def _scratch(self, virtual_path: str):
        """Materialise a virtual (possibly Windows) path as one real POSIX file."""
        safe = virtual_path.replace("\\", "#").replace("/", "#").replace(":", "%")
        real = self.tmp_path / "artifacts" / safe
        real.parent.mkdir(parents = True, exist_ok = True)
        real.write_bytes(b"GGUF")
        return real

    def _convert(self, **kwargs):
        self.convert_calls.append(kwargs)
        model_name = kwargs["model_name"]
        qtype = kwargs["quantization_type"]
        if qtype == "None":
            qtype = kwargs.get("model_dtype", "bf16")
        # unsloth_zoo/llama_cpp.py:2465 -- the name zoo would build.
        final = model_name if model_name.endswith(".gguf") else f"{model_name}.{qtype.upper()}.gguf"
        self.initial_names.append(final)
        # save.py:2037 requires the returned paths to exist.
        # The escape under test is at the *quantize* join (model_name + gguf_directory), not this path.
        return [str(self._scratch(final))], False

    def _quantize(self, input_gguf, output_gguf, quant_type, **kw):
        self.quantize_calls.append(
            {"input": input_gguf, "output": output_gguf, "quant": quant_type}
        )
        self._scratch(output_gguf)
        return output_gguf


_WINDOWS_BASE = r"D:\Models\Merged Models\MyModel"
_EXPORT_DIR = r"C:\Users\u\.unsloth\studio\exports\MyModel\_tmp_model_ab12"


def _run(
    monkeypatch,
    tmp_path,
    model_name,
    model_directory,
    flavour = ntpath,
    methods = None,
):
    harness = _Harness(monkeypatch, tmp_path, flavour)
    monkeypatch.setattr(save_mod, "shutil", _NoMove(), raising = False)
    # gguf_directory is a Windows-style string here;
    # os.makedirs is the *real* one, so chdir into tmp so a literal "C:\..." directory cannot land in the repo.
    monkeypatch.chdir(tmp_path)
    save_mod.save_to_gguf(
        model_name = model_name,
        model_type = "llama",
        model_dtype = "bfloat16",
        model_directory = model_directory,
        quantization_method = methods or ["q5_k_m"],
    )
    return harness


class _NoMove:
    """save_to_gguf relocates initial files with shutil.move, but the sources here are
    virtual Windows paths that do not exist on POSIX. Neutralise the move."""

    def move(self, src, dst):
        return dst

    def rmtree(self, *a, **kw):
        return None

    def copy2(self, *a, **kw):
        return None

    def which(self, *a, **kw):
        return None


def test_derived_stem_keeps_quantized_gguf_inside_gguf_directory(monkeypatch, tmp_path):
    """The fixed derivation: output stays under <model_directory>_gguf."""
    stem = save_mod._model_basename(_WINDOWS_BASE)
    harness = _run(monkeypatch, tmp_path, stem, _EXPORT_DIR)

    assert harness.quantize_calls, (
        "no llama-quantize pass ran, so this parametrization cannot detect the "
        "bug -- a k-quant (not q8_0/f16/bf16) is required"
    )
    expected_dir = _EXPORT_DIR + "_gguf"
    for call in harness.quantize_calls:
        assert ntpath.dirname(call["output"]) == expected_dir, call["output"]
        assert not call["output"].startswith("D:"), call["output"]


def test_legacy_stem_escaped_to_the_base_model_drive(monkeypatch, tmp_path):
    """Pin the #7897 failure mode: the pre-fix stem relocated the output to D:."""
    legacy_stem = _WINDOWS_BASE.split("/")[-1]
    harness = _run(monkeypatch, tmp_path, legacy_stem, _EXPORT_DIR)

    assert harness.quantize_calls
    escaped = harness.quantize_calls[0]["output"]
    assert escaped == r"D:\Models\Merged Models\MyModel.Q5_K_M.gguf"
    assert ntpath.dirname(escaped) != _EXPORT_DIR + "_gguf"


def test_trailing_separator_no_longer_yields_a_hidden_gguf(monkeypatch, tmp_path):
    """OS-agnostic half of the bug: a trailing sep gave an empty stem."""
    assert "".join(_WINDOWS_BASE.rsplit("\\", 1)[1:]) == "MyModel"
    legacy_stem = "/home/u/models/MyModel/".split("/")[-1]
    assert legacy_stem == ""

    stem = save_mod._model_basename("/home/u/models/MyModel/")
    export_dir = str(tmp_path / "exports" / "run")
    harness = _run(monkeypatch, tmp_path, stem, export_dir, flavour = posixpath)

    assert harness.quantize_calls
    out = harness.quantize_calls[0]["output"]
    assert posixpath.basename(out) == "MyModel.Q5_K_M.gguf"
    assert not posixpath.basename(out).startswith(".")
    assert posixpath.dirname(out) == export_dir + "_gguf"


@pytest.mark.parametrize("methods", [["q5_k_m"], ["q4_k_m", "q5_k_m"], ["q4_k_m", "q8_0"]])
def test_multi_quant_all_outputs_stay_inside(monkeypatch, tmp_path, methods):
    stem = save_mod._model_basename(_WINDOWS_BASE)
    harness = _run(monkeypatch, tmp_path, stem, _EXPORT_DIR, methods = methods)
    expected_dir = _EXPORT_DIR + "_gguf"
    assert harness.quantize_calls
    for call in harness.quantize_calls:
        assert ntpath.dirname(call["output"]) == expected_dir, call["output"]


def test_posix_paths_route_identically(monkeypatch, tmp_path):
    """Control arm: the fix must not change POSIX behaviour."""
    stem = save_mod._model_basename("/home/u/models/MyModel")
    assert stem == "MyModel"
    export_dir = str(tmp_path / "exports" / "run")
    harness = _run(monkeypatch, tmp_path, stem, export_dir, flavour = posixpath)
    assert harness.quantize_calls
    for call in harness.quantize_calls:
        assert posixpath.dirname(call["output"]) == export_dir + "_gguf"
