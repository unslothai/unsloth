# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Q4NX folders converted on the Export page show in the NPU picker and load through FastFlowLM."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

from core.export import q4nx
from core.inference import npu_backend as nb
from core.inference import npu_local_models as nlm
from utils.account_context import AccountContext, run_as
from utils.paths.storage_roots import exports_root

ALICE = AccountContext("alice-npu-local", "alice")
BOB = AccountContext("bob-npu-local", "bob")

_QWEN3_06B = {"architecture": "qwen3", "block_count": 28, "embedding_length": 1024}
_BASE_ENTRY = {
    "name": "Qwen3-0.6B-NPU2",
    "url": "https://huggingface.co/FastFlowLM/Qwen3-0.6B-NPU2",
    "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-0.6B-NPU2/tree/main",
    "ms_url": "https://modelscope.cn/models/amd/Qwen3-0.6B-NPU2",
    "size": 600000000,
    "flm_min_version": "1.0.0",
    "default_context_length": 8192,
    "files": ["config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json"],
    "vlm": False,
    "details": {"format": "NPU2", "family": "qwen3", "parameter_size": "0.6B"},
    "label": ["reasoning"],
    "footprint": 0.66,
}


def _converted(
    name = "Qwen3-0.6B-ft.Q4_0-q4nx",
    *,
    source = _QWEN3_06B,
    skip = (),
) -> Path:
    folder = exports_root() / "q4nx" / name
    folder.mkdir(parents = True)
    for file in ("model.q4nx", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if file not in skip:
            (folder / file).write_bytes(b"{}" if file.endswith(".json") else b"weights")
    if source is not None:
        (folder / q4nx.SOURCE_FILE).write_text(json.dumps(source), encoding = "utf-8")
    return folder


def test_converted_folders_are_listed_with_their_kernel_base():
    folder = _converted()
    (exports_root() / "gguf" / "unrelated").mkdir(parents = True)
    (exports_root() / "q4nx" / ".Qwen3-staging").mkdir()
    (exports_root() / "q4nx" / ".Qwen3-staging" / "model.q4nx").write_bytes(b"half")

    [local] = nlm.account_local_models()
    assert local.folder == folder
    assert local.display == "q4nx/Qwen3-0.6B-ft.Q4_0-q4nx"
    assert local.base == "qwen3:0.6b"
    assert local.problem is None
    assert local.id.startswith("local-qwen3-0.6b-ft.q4-0-q4nx-") and local.id.endswith("-q4nx-FLM")
    assert local.checkpoint == f"{local.family}:q4nx"
    assert "chat_template.jinja" in local.files and not local.vision


@pytest.mark.parametrize(
    "skip, source, problem",
    [
        (("tokenizer_config.json",), _QWEN3_06B, "tokenizer_config.json is missing"),
        ((), None, "Convert the GGUF to Q4NX again"),
        ((), [], "conversion record is not readable"),
        (
            (),
            {"architecture": "qwen3", "block_count": 30, "embedding_length": 999},
            "no AMD NPU kernels",
        ),
    ],
)
def test_an_unloadable_folder_is_listed_with_the_reason(skip, source, problem):
    _converted(skip = skip, source = source)
    [local] = nlm.account_local_models()
    assert problem in local.problem


def test_every_shape_the_export_accepts_has_a_base_the_picker_loads():
    # A shape that converts but has no FastFlowLM base would list as unloadable.
    export_shapes = getattr(q4nx, "_NPU_SHAPES", None)
    if export_shapes is None:
        pytest.skip("this tree has no Q4NX export geometry gate")
    for arch, shapes in export_shapes.items():
        assert set(shapes) <= set(nlm._FLM_BASES.get(arch, {})), arch


def test_a_nanbeige_architecture_conversion_loads_on_the_nanbeige_base():
    _converted(source = {"architecture": "nanbeige", "block_count": 32, "embedding_length": 2560})
    [local] = nlm.account_local_models()
    assert local.base == "nanbeige4.1:3b" and local.problem is None


def test_a_conversion_saved_outside_or_deep_in_the_exports_is_listed_once_registered(
    monkeypatch, tmp_path
):
    deep = exports_root() / "a" / "b" / "c" / "d"
    deep.mkdir(parents = True)
    outside = tmp_path / "other-drive"
    outside.mkdir()
    folders = []
    for parent in (deep, outside):
        folder = _converted()
        folders.append(folder.rename(parent / folder.name))
    assert nlm.account_local_models() == []
    assert nlm.hidden_in_exports(folders[0]) and not nlm.hidden_in_exports(folders[1])
    assert not nlm.hidden_in_exports(exports_root() / "q4nx" / "m-q4nx")

    monkeypatch.setattr(nlm, "_registered_folders", lambda: [*folders, tmp_path])
    listed = nlm.account_local_models()
    assert sorted(m.folder for m in listed) == sorted(folders)
    assert all(m.base == "qwen3:0.6b" and m.problem is None for m in listed)
    # Inside exports the row names its exports path; outside it names the full path.
    assert {m.folder: m.display for m in listed} == {
        folders[0]: "a/b/c/d/Qwen3-0.6B-ft.Q4_0-q4nx",
        folders[1]: folders[1].as_posix(),
    }


def test_a_gguf_exports_q4nx_folder_under_a_registered_save_folder_is_listed(monkeypatch, tmp_path):
    # export.py writes the conversion to <save folder>/npu-q4nx and registers the save folder.
    save = tmp_path / "elsewhere" / "Qwen3-0.6B-ft"
    save.mkdir(parents = True)
    folder = _converted().rename(save / nlm.EXPORT_SUBFOLDER)
    monkeypatch.setattr(nlm, "_registered_folders", lambda: [save])
    [local] = nlm.account_local_models()
    assert local.folder == folder and local.display == folder.as_posix()
    assert local.base == "qwen3:0.6b" and local.problem is None


def test_a_registered_folder_inside_the_walk_is_listed_once(monkeypatch):
    folder = _converted()
    monkeypatch.setattr(nlm, "_registered_folders", lambda: [folder])
    assert [m.folder for m in nlm.account_local_models()] == [folder]


def test_another_accounts_registered_conversion_is_not_listed(monkeypatch):
    folder = run_as(BOB, _converted)
    monkeypatch.setattr(nlm, "_registered_folders", lambda: [folder])
    assert run_as(ALICE, nlm.account_local_models) == []


def test_a_junction_is_skipped_on_a_python_without_isjunction(monkeypatch):
    import types

    from utils import cache_inventory

    junction = exports_root() / "q4nx" / "junction"
    junction.mkdir(parents = True)
    _converted().rename(junction / "elsewhere-q4nx")
    kept = _converted()

    def lstat(path):
        tag = 0xA0000003 if Path(path) == junction else 0  # IO_REPARSE_TAG_MOUNT_POINT
        return types.SimpleNamespace(st_reparse_tag = tag)

    # Windows on Python 3.9-3.11: no os.path.isjunction, a mount-point reparse tag.
    pre312 = types.SimpleNamespace(path = types.SimpleNamespace(), name = "nt", lstat = lstat)
    monkeypatch.setattr(cache_inventory, "os", pre312)
    assert [m.folder for m in nlm.account_local_models()] == [kept]


def test_an_account_sees_only_its_own_exports():
    run_as(ALICE, _converted)
    assert [m.folder.name for m in run_as(ALICE, nlm.account_local_models)] == [
        "Qwen3-0.6B-ft.Q4_0-q4nx"
    ]
    assert run_as(BOB, nlm.account_local_models) == []
    assert nlm.account_local_models() == []


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_a_link_out_of_the_exports_is_not_followed(tmp_path):
    outside = tmp_path / "elsewhere" / "model"
    outside.mkdir(parents = True)
    (outside / "model.q4nx").write_bytes(b"weights")
    exports_root().mkdir(parents = True, exist_ok = True)
    (exports_root() / "linked").symlink_to(outside.parent, target_is_directory = True)
    assert nlm.account_local_models() == []


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_a_linked_weights_file_from_another_account_is_neither_listed_nor_staged(tmp_path):
    theirs = run_as(BOB, _converted)
    folder = run_as(ALICE, _converted)
    (folder / "model.q4nx").unlink()
    (folder / "model.q4nx").symlink_to(theirs / "model.q4nx")
    [local] = run_as(ALICE, nlm.account_local_models)
    assert "model.q4nx" not in local.files
    assert "model.q4nx is missing" in local.problem
    listed = run_as(ALICE, lambda: nlm._describe(folder, exports_root()))
    stale = dataclasses.replace(listed, files = ("model.q4nx", *listed.files))
    with pytest.raises(OSError, match = "model.q4nx is a link"):
        run_as(ALICE, lambda: nlm.link_files(stale, tmp_path / "staged"))
    assert not (tmp_path / "staged" / "model.q4nx").exists()


def test_the_registered_entry_runs_on_the_base_kernels_and_cannot_be_pulled():
    folder = _converted()
    [local] = nlm.account_local_models()
    entry = nlm.local_entry(_BASE_ENTRY, local, "1.0.12")
    assert entry["name"] == local.family
    assert entry["xclbin_folder"] == "Qwen3-0.6B-NPU2"
    assert entry["files"] == ["config.json", *local.files]
    assert entry["flm_min_version"] == "1.0.12"
    assert entry["url"].split("/")[-1] == local.family
    assert ".invalid/" in entry["url"] and "ms_url" not in entry
    assert entry["size"] == _BASE_ENTRY["size"]
    assert folder.is_dir()


def test_a_hidden_stamp_lasts_until_its_weights_change():
    folder = _converted()
    stamp = nlm.weights_stamp(folder)
    assert nlm.stamp_is_current(stamp)
    os.utime(folder / "model.q4nx", ns = (1, 1))
    assert not nlm.stamp_is_current(stamp)
    assert not nlm.stamp_is_current("")


def test_a_text_only_conversion_drops_the_vision_tower():
    _converted()
    [local] = nlm.account_local_models()
    config = nlm.local_config(
        {"flm_version": "1.0.12", "vision_model_weight": "vision_weight.q4nx"}, local
    )
    assert "vision_model_weight" not in config


def test_a_file_removed_mid_scan_does_not_fail_the_listing(monkeypatch):
    folder = _converted()
    real_stat = Path.stat

    def _racing_stat(self, *args, **kwargs):
        if self.name == "tokenizer.json" and self.parent == folder:
            raise FileNotFoundError(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(nlm, "_own_file", lambda path: path.name in nlm.REQUIRED_FILES)
    monkeypatch.setattr(Path, "stat", _racing_stat)
    assert [m.folder for m in nlm.find_local_models(folder.parent)] == [folder]


def test_a_text_only_gemma4_conversion_drops_the_audio_tower():
    # FastFlowLM/Gemma4-E2B-IT-NPU2's config.json names both towers; FastFlowLM 1.0.12 opens any it names.
    _converted()
    [local] = nlm.account_local_models()
    config = nlm.local_config(
        {
            "flm_version": "1.0.12",
            "vision_model_weight": "vision_weight.q4nx",
            "audio_model_weight": "audio_weight.q4nx",
        },
        local,
    )
    assert "audio_model_weight" not in config and "vision_model_weight" not in config


def test_the_conversion_records_the_source_geometry(tmp_path, monkeypatch):
    import utils.models.gguf_metadata as gm

    monkeypatch.setattr(
        gm,
        "_parse_gguf_arch_uints",
        lambda path, keys: {"block_count": 28, "embedding_length": 1024},
    )
    q4nx.write_source_record(str(tmp_path / "Qwen3-0.6B.Q4_0.gguf"), tmp_path, "qwen3")
    record = json.loads((tmp_path / q4nx.SOURCE_FILE).read_text(encoding = "utf-8"))
    assert record == {**_QWEN3_06B, "gguf": "Qwen3-0.6B.Q4_0.gguf"}


# Fake lemond is a shell script; the rest runs on POSIX only, like test_npu_backend.py.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason = "shell-script lemond stand-in")


class _Installer:
    def __init__(self, binary: Path) -> None:
        self.binary = binary

    def installed_lemond(self, root):
        return self.binary

    def install(
        self,
        root,
        cancel = None,
    ):
        return self.binary


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    fake = Path(__file__).parent / "fixtures" / "fake_lemond.py"
    binary = tmp_path / "bin" / "lemond"
    binary.parent.mkdir(parents = True)
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n', encoding = "utf-8")
    binary.chmod(0o755)
    installer = _Installer(binary)
    monkeypatch.setattr(nb, "_installer_module", lambda: installer)
    monkeypatch.setattr(
        nb,
        "detect_amd_npu",
        lambda: {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
    )
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    backend.enable()
    flm = backend._flm_binary()
    listing = {"model_path": "models", "models": {"qwen3": {"0.6b": _BASE_ENTRY}}}
    (flm.parent / "model_list.json").write_text(json.dumps(listing), encoding = "utf-8")
    base = backend._flm_model_dir / "models" / "Qwen3-0.6B-NPU2"
    base.mkdir(parents = True)
    (base / "config.json").write_text(
        json.dumps({"flm_version": "1.0.12", "num_hidden_layers": 28}), encoding = "utf-8"
    )
    yield backend
    backend.shutdown()


def test_the_base_config_is_fetched_from_the_manifests_fastflowlm_reads(tmp_path, monkeypatch):
    # Installs that keep model_list.json and model_info.json in share/flm, not beside the binary.
    binary = tmp_path / "bin" / "flm"
    binary.parent.mkdir(parents = True)
    share = tmp_path / "share" / "flm"
    share.mkdir(parents = True)
    listing = {"model_path": "models", "models": {"qwen3": {"0.6b": _BASE_ENTRY}}}
    (share / "model_list.json").write_text(json.dumps(listing), encoding = "utf-8")
    info = {"qwen3:0.6b": [{"path": "config.json", "size": 2, "oid": "x"}]}
    (share / "model_info.json").write_text(json.dumps(info), encoding = "utf-8")
    fetched = []

    def download(model):
        fetched.extend(f.name for f in model.files)
        model.folder.mkdir(parents = True, exist_ok = True)
        (model.folder / "config.json").write_text('{"flm_version": "1.0.12"}', encoding = "utf-8")
        yield from ()

    monkeypatch.setattr(nb, "download_files", download)
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    assert nlm.find_flm_list(binary) == share / "model_list.json"
    config = backend._base_config(binary, share, listing, "qwen3:0.6b", _BASE_ENTRY)
    assert fetched == ["config.json"] and config == {"flm_version": "1.0.12"}


def _requests(backend) -> list[dict]:
    path = backend.root / "cache" / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@posix_only
def test_the_picker_lists_converted_folders_beside_the_catalog(runtime):
    _converted()
    rows = {model.id: model for model in runtime.catalog()}
    [local] = [model for model in rows.values() if model.local]
    assert local.downloaded and local.local["folder"] == "q4nx/Qwen3-0.6B-ft.Q4_0-q4nx"
    assert local.local["base"] == "qwen3:0.6b" and local.local["problem"] is None
    assert local.size_gb == rows["qwen3-0.6b-FLM"].size_gb
    assert local.to_json()["local"] == local.local


@posix_only
def test_a_converted_row_carries_the_memory_fit_of_its_size(runtime, monkeypatch):
    monkeypatch.setattr(nb.npu_memory, "system_ram_gb", lambda: 0.5)
    _converted()
    rows = {model.id: model for model in runtime.catalog()}
    [local] = [model for model in rows.values() if model.local]
    base = rows["qwen3-0.6b-FLM"]
    assert local.fits_memory is False and local.fits_memory == base.fits_memory
    assert local.memory_needed_gb is not None and local.memory_needed_gb == base.memory_needed_gb


@posix_only
def test_load_registers_the_folder_then_loads_it(runtime):
    folder = _converted()
    [local] = nlm.account_local_models()
    runtime.load(local.id)

    assert runtime.loaded_model.id == local.id
    loads = [r["body"]["model_name"] for r in _requests(runtime) if r["path"] == "/v1/load"]
    assert loads == [local.id]
    staged = runtime._flm_model_dir / "models" / local.family
    assert os.path.samefile(staged / "model.q4nx", folder / "model.q4nx")
    assert json.loads((staged / "config.json").read_text())["flm_version"] == "1.0.12"
    overlay = json.loads((runtime.root / "local_q4nx" / "model_list.json").read_text())
    assert overlay["models"]["qwen3"]["0.6b"] == _BASE_ENTRY
    assert overlay["models"][local.family]["q4nx"]["xclbin_folder"] == "Qwen3-0.6B-NPU2"
    assert runtime._server.flm_config_path == runtime.root / "local_q4nx" / "model_list.json"
    assert not (folder / "config.json").exists()


@posix_only
def test_auto_context_sizes_a_conversion_from_its_staged_folder(runtime, monkeypatch):
    planned = []
    real = nb.plan_context

    def plan(model, files, requested):
        planned.append(files)
        return real(model, files, requested)

    monkeypatch.setattr(nb, "plan_context", plan)
    _converted()
    [local] = nlm.account_local_models()
    runtime.load(local.id)
    [files] = planned
    staged = runtime._flm_model_dir / "models" / local.family
    assert files is not None and files.folder == staged
    assert {f.name for f in files.files} >= {"model.q4nx", "config.json"}
    assert json.loads((files.folder / "config.json").read_text())["num_hidden_layers"] == 28


@posix_only
def test_delete_drops_the_registration_and_keeps_the_export(runtime):
    folder = _converted()
    [local] = nlm.account_local_models()
    runtime.load(local.id)
    with pytest.raises(nb.NpuError, match = "Unload"):
        runtime.delete(local.id)
    runtime.unload()
    runtime.delete(local.id)

    assert not (runtime._flm_model_dir / "models" / local.family).exists()
    assert not (runtime.root / "local_q4nx" / "registered.json").exists()
    assert sorted(p.name for p in folder.iterdir()) == sorted([*local.files, q4nx.SOURCE_FILE])
    assert (folder / "model.q4nx").read_bytes() == b"weights"
    assert not any(r["path"] == "/v1/delete" for r in _requests(runtime))
    assert not any(m.local for m in runtime.catalog())
    with pytest.raises(nb.NpuError, match = "not a converted model"):
        runtime.load(local.id)

    (folder / "model.q4nx").write_bytes(b"new weights")
    assert [m.id for m in runtime.catalog() if m.local] == [local.id]


@posix_only
def test_a_staged_copy_whose_export_is_gone_can_still_be_deleted(runtime):
    import shutil

    folder = _converted()
    [local] = nlm.account_local_models()
    runtime.load(local.id)
    runtime.unload()
    staged = runtime._flm_model_dir / "models" / local.family
    assert staged.is_dir()
    shutil.rmtree(folder)

    assert (staged, f"{local.family}:q4nx") in runtime.model_folders()
    assert runtime.delete_checkpoint(f"{local.family}:q4nx") == local.id
    assert not staged.exists()
    assert not (runtime.root / "local_q4nx" / "registered.json").exists()


@posix_only
def test_a_conversion_republished_while_staging_is_refused(runtime, monkeypatch):
    folder = _converted()
    [local] = nlm.account_local_models()
    real = runtime._base_config

    def _republish(*args, **kwargs):
        config = real(*args, **kwargs)
        (folder / "model.q4nx").write_bytes(b"other weights")
        return config

    monkeypatch.setattr(runtime, "_base_config", _republish)
    with pytest.raises(nb.NpuError, match = "changed while loading"):
        runtime.load(local.id)
    assert not (runtime.root / "local_q4nx" / "registered.json").exists()


@posix_only
def test_bulk_delete_clears_an_orphaned_staged_copy(runtime):
    import shutil

    folder = _converted()
    [local] = nlm.account_local_models()
    runtime.load(local.id)
    runtime.unload()
    shutil.rmtree(folder)
    _deleted, errors = runtime.delete_downloaded()
    assert errors == []
    assert not (runtime._flm_model_dir / "models" / local.family).exists()


@posix_only
def test_an_incomplete_folder_is_refused_before_touching_the_npu(runtime):
    _converted(skip = ("tokenizer.json",))
    [local] = nlm.account_local_models()
    with pytest.raises(nb.NpuError, match = "tokenizer.json is missing"):
        runtime.load(local.id)
    with pytest.raises(nb.NpuError, match = "tokenizer.json is missing"):
        runtime.loadable_model(local.id)
    assert not any(r["path"] == "/v1/load" for r in _requests(runtime))
    assert not (runtime.root / "local_q4nx" / "registered.json").exists()


@posix_only
def test_another_accounts_folder_cannot_be_loaded(runtime):
    folder = run_as(ALICE, _converted)
    [local] = run_as(ALICE, nlm.account_local_models)
    with pytest.raises(nb.NpuError, match = "not a converted model"):
        runtime.load(local.id)
    assert not any(m.local for m in runtime.catalog())
    assert folder.is_dir()


@posix_only
def test_a_missing_npu_is_refused_with_the_reason(runtime, monkeypatch):
    _converted()
    [local] = nlm.account_local_models()
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: {"present": False, "supported": False})
    runtime._hardware = None
    with pytest.raises(nb.NpuError, match = "No AMD Ryzen AI NPU"):
        runtime.load(local.id)


@posix_only
def test_a_missing_runtime_is_refused_with_the_reason(runtime, monkeypatch):
    _converted()
    [local] = nlm.account_local_models()
    monkeypatch.setattr(runtime, "_flm_binary", lambda: None)
    with pytest.raises(nb.NpuError, match = "not installed"):
        runtime.load(local.id)


def test_a_converted_model_samples_like_its_base():
    from utils.inference.inference_config import load_inference_config

    converted = nb.NpuModel(
        id = "local-npu-q4nx-1a2b3c4d-q4nx-FLM",
        checkpoint = "local-npu-q4nx-1a2b3c4d:q4nx",
        size_gb = 0.6,
        downloaded = True,
        labels = ("chat",),
        max_context_length = 32768,
        local = {"folder": "q4nx/Qwen3-0.6B", "base": "qwen3:0.6b", "problem": None},
    )
    catalog = dataclasses.replace(converted, id = "qwen3-0.6b-FLM", local = None)
    assert converted.sampling_id == catalog.sampling_id == "qwen3-0.6b-FLM"
    assert load_inference_config(converted.sampling_id) == load_inference_config(catalog.id)
