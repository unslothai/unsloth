# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF export must find its output, or fail honestly (#7897).

save_pretrained_gguf already returns the files it wrote, but export_gguf discarded
that and guessed: cwd, new subdirs of the save dir, and a `<checkpoint>_gguf` dir.
A GGUF written anywhere else, as a Windows local base-model path caused, was
invisible, and the export still reported success over an empty directory.

Reuses the harness in test_export_absolute_paths.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


def _backend(
    monkeypatch,
    tmp_path,
    model,
    checkpoint = None,
):
    """An ExportBackend wired to `model`, exporting into tmp_path/'export'."""
    _install_export_backend_stubs(monkeypatch)
    export_mod = _load_module("test_core_export_backend", "core/export/export.py", monkeypatch)

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    save_dir = tmp_path / "export"
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(export_mod, "resolve_export_write_dir", lambda _value: save_dir)

    backend = export_mod.ExportBackend.__new__(export_mod.ExportBackend)
    backend.current_model = model
    backend.current_tokenizer = object()
    backend.current_checkpoint = checkpoint
    return export_mod, backend, save_dir, cwd


def _gguf(path: Path, payload: bytes = b"GGUF") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(payload)
    return path


# _reported_gguf_files: the "is this build telling us anything?" contract.


@pytest.mark.parametrize(
    "result",
    [
        None,  # pre-2025.10 unsloth / non-main process
        "some/path",  # save_method="lora" returns a str
        ("path", True, False),  # hypothetical legacy tuple
        {},  # dict without the key
        {"gguf_files": None},
        {"gguf_files": "not-a-list"},
        {"gguf_files": []},  # empty == "nothing to say"
        {"gguf_files": [123]},  # malformed entry -> distrust all
    ],
    ids = [
        "none",
        "str",
        "tuple",
        "empty_dict",
        "null_files",
        "str_files",
        "empty_list",
        "bad_entry",
    ],
)
def test_reported_files_absent_shapes_fall_back(monkeypatch, tmp_path, result):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    assert export_mod._reported_gguf_files(result) is None


def test_reported_files_filters_missing_and_non_gguf(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    real = _gguf(tmp_path / "a" / "Model.Q4_K_M.gguf")
    out = export_mod._reported_gguf_files(
        {
            "gguf_files": [
                str(real),
                str(tmp_path / "a" / "deleted.gguf"),  # unlinked by cleanup
                str(tmp_path / "a" / "notes.txt"),  # not a gguf
                str(tmp_path / "a"),  # a directory
            ]
        }
    )
    assert out == [str(real)]


def test_reported_files_accepts_future_keys(monkeypatch, tmp_path):
    export_mod, _b, _s, _c = _backend(monkeypatch, tmp_path, object())
    real = _gguf(tmp_path / "a" / "Model.Q4_K_M.gguf")
    out = export_mod._reported_gguf_files(
        {"gguf_files": [str(real)], "some_future_field": 1, "is_vlm": True}
    )
    assert out == [str(real)]


# Table B: where the fake exporter puts its output.


def test_gguf_beside_base_model_is_relocated(monkeypatch, tmp_path):
    """The #7897 shape: output lands outside the save dir, cwd and checkpoint."""
    sibling = tmp_path / "Models" / "Merged Models"

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            stray = _gguf(sibling / "MyModel.Q5_K_M.gguf")
            return {"gguf_files": [str(stray)]}

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, output_path = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is True, message
    assert (save_dir / "MyModel.Q5_K_M.gguf").is_file()
    assert not (sibling / "MyModel.Q5_K_M.gguf").exists()
    assert output_path == str(save_dir.resolve())


def test_zero_files_is_a_failure_not_a_silent_success(monkeypatch, tmp_path):
    """Old unsloth (no manifest) plus a lost output must not report success."""

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            _gguf(tmp_path / "elsewhere" / "MyModel.Q5_K_M.gguf")
            return None  # legacy build

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, output_path = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is False
    assert str(save_dir) in message
    assert "no .gguf" in message
    assert output_path is None
    assert list(save_dir.glob("_tmp_model_*")) == []


def test_nested_gguf_is_rescued_before_rmtree(monkeypatch, tmp_path):
    """The flatten pass rmtree's every new subdir; sharded output sat one deeper."""

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            tmp = Path(model_save_path)
            tmp.mkdir(parents = True)
            gguf_dir = Path(str(tmp) + "_gguf") / "shards"
            for i in (1, 2, 3):
                _gguf(gguf_dir / f"MyModel-{i:05d}-of-00003.gguf")
            return None  # exercise the fallback path

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is True, message
    names = sorted(p.name for p in save_dir.glob("*.gguf"))
    assert names == [
        "MyModel-00001-of-00003.gguf",
        "MyModel-00002-of-00003.gguf",
        "MyModel-00003-of-00003.gguf",
    ]


def test_hidden_gguf_is_reported_not_none(monkeypatch, tmp_path):
    """An empty model stem produced '.Q5_K_M.gguf'; glob.glob could not see it.

    A reporting defect, not file loss: the flatten pass uses Path.glob, which does
    match dot-leading names, so the file was in place while the log said "(none)".
    Still bites, because zero-files is now a failure: reverting the listing to
    glob.glob would fail this export outright.
    """

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            tmp = Path(model_save_path)
            tmp.mkdir(parents = True)
            _gguf(Path(str(tmp) + "_gguf") / ".Q5_K_M.gguf")
            return None

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is True, message
    assert (save_dir / ".Q5_K_M.gguf").is_file()


def test_files_already_in_place_are_not_moved_onto_themselves(monkeypatch, tmp_path):
    """The normal PEFT path already lands inside the save dir."""

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            here = _gguf(tmp_path / "export" / "MyModel.Q5_K_M.gguf")
            return {"gguf_files": [str(here)]}

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is True, message
    assert (save_dir / "MyModel.Q5_K_M.gguf").read_bytes() == b"GGUF"


def test_multi_quant_relocates_every_output(monkeypatch, tmp_path):
    sibling = tmp_path / "beside"

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            files = [
                str(_gguf(sibling / f"MyModel.{q}.gguf")) for q in ("Q4_K_M", "Q5_K_M", "Q8_0")
            ]
            return {"gguf_files": files}

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), ["q4_k_m", "q5_k_m", "q8_0"])

    assert success is True, message
    assert sorted(p.name for p in save_dir.glob("*.gguf")) == [
        "MyModel.Q4_K_M.gguf",
        "MyModel.Q5_K_M.gguf",
        "MyModel.Q8_0.gguf",
    ]
    assert list(sibling.glob("*.gguf")) == []


def test_vlm_mmproj_companion_is_relocated(monkeypatch, tmp_path):
    sibling = tmp_path / "beside"

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            return {
                "gguf_files": [
                    str(_gguf(sibling / "MyVLM.Q4_K_M.gguf")),
                    str(_gguf(sibling / "MyVLM.F16-mmproj.gguf")),
                ]
            }

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), "q4_k_m")

    assert success is True, message
    assert (save_dir / "MyVLM.Q4_K_M.gguf").is_file()
    assert (save_dir / "MyVLM.F16-mmproj.gguf").is_file()


def test_modelfile_is_relocated_not_deleted(monkeypatch, tmp_path):
    """The Modelfile lived in the temp *_gguf dir the flatten pass rmtree's."""

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            tmp = Path(model_save_path)
            tmp.mkdir(parents = True)
            gguf_dir = Path(str(tmp) + "_gguf")
            out = _gguf(gguf_dir / "MyModel.Q5_K_M.gguf")
            mf = gguf_dir / "Modelfile"
            mf.write_text("FROM ./MyModel.Q5_K_M.gguf\n")
            return {"gguf_files": [str(out)], "modelfile_location": str(mf)}

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m")

    assert success is True, message
    assert (save_dir / "Modelfile").is_file()
    assert (save_dir / "MyModel.Q5_K_M.gguf").is_file()


def test_stale_gguf_alone_does_not_fake_a_successful_export(monkeypatch, tmp_path):
    """A leftover destination artifact must not hide a conversion with no output."""

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            return None

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    save_dir.mkdir(parents = True, exist_ok = True)
    _gguf(save_dir / "OldRun.Q4_K_M.gguf")

    success, message, output_path = backend.export_gguf(str(save_dir), "q5_k_m")
    assert success is False
    assert "produced no files" in message
    assert output_path is None


def test_cleanup_failure_does_not_lose_reported_files(monkeypatch, tmp_path):
    """Windows locks make rmtree(ignore_errors=True) a silent no-op."""
    sibling = tmp_path / "beside"

    class _Model:
        def save_pretrained_gguf(self, model_save_path, tokenizer, quantization_method):
            Path(model_save_path).mkdir(parents = True)
            return {"gguf_files": [str(_gguf(sibling / "MyModel.Q5_K_M.gguf"))]}

    export_mod, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())
    monkeypatch.setattr(export_mod.shutil, "rmtree", lambda *a, **k: None)

    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m")
    assert success is True, message
    assert (save_dir / "MyModel.Q5_K_M.gguf").is_file()


def test_materialized_imatrix_is_not_exported_as_a_model(monkeypatch, tmp_path):
    """unsloth copies a *.gguf_file imatrix next to the model as *.gguf; it is not an output."""

    class _Model:
        def save_pretrained_gguf(
            self, model_save_path, tokenizer, quantization_method, imatrix_file = None
        ):
            # _materialize_imatrix copies into the model dir, renaming .gguf_file -> .gguf.
            _gguf(Path(model_save_path) / "imatrix_unsloth.gguf", b"IMATRIX")
            quant = _gguf(Path(f"{model_save_path}_gguf") / "MyModel.Q5_K_M.gguf")
            return {"gguf_files": [str(quant)]}

    _m, backend, save_dir, _cwd = _backend(monkeypatch, tmp_path, _Model())

    success, message, _p = backend.export_gguf(str(save_dir), "q5_k_m", imatrix_file = True)

    assert success is True, message
    assert (save_dir / "MyModel.Q5_K_M.gguf").is_file()
    assert not (save_dir / "imatrix_unsloth.gguf").exists()
