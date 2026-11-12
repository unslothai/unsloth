# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Q4NX folders converted on the Export page show in the NPU picker and load through FastFlowLM."""

from __future__ import annotations

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

    # Converting into the folder again lists it again.
    (folder / "model.q4nx").write_bytes(b"new weights")
    assert [m.id for m in runtime.catalog() if m.local] == [local.id]


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
