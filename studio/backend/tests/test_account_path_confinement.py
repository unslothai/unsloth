# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's paths, model references and checkpoint loads stay inside what it may read."""

import contextlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from auth import policy  # noqa: E402
from core.training.account_jobs import account_path  # noqa: E402
from hub.services.models import account_access  # noqa: E402
from utils.account_context import OWNER, AccountContext, run_as  # noqa: E402
from utils.paths import storage_roots  # noqa: E402

ALICE = AccountContext("alice-confine", "alice")
BOB = AccountContext("bob-confine", "bob")

symlinks = pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")


@pytest.fixture(autouse = True)
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True, raising = False)
    elsewhere = tmp_path / "server-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return tmp_path / "studio"


def _workspace(account):
    root = run_as(account, storage_roots.workspace_root)
    root.mkdir(parents = True, exist_ok = True)
    return root


def _refused(call, *args, **kwargs):
    with pytest.raises(HTTPException) as exc:
        call(*args, **kwargs)
    return exc.value.status_code


# account_path: the name itself, not only its target, must sit inside the account.


@symlinks
def test_outside_link_into_own_workspace_is_refused(tmp_path):
    ws = _workspace(ALICE)
    (ws / "data.jsonl").write_text("{}")
    link = tmp_path / "outside-link"
    link.symlink_to(ws, target_is_directory = True)
    assert _refused(run_as, ALICE, account_path, str(link / "data.jsonl")) == 403


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason = "needs procfs")
def test_proc_cwd_spelling_is_refused(monkeypatch):
    ws = _workspace(ALICE)
    (ws / "data.jsonl").write_text("{}")
    monkeypatch.chdir(ws)
    assert _refused(run_as, ALICE, account_path, "/proc/self/cwd/data.jsonl") == 403


@symlinks
def test_workspace_paths_through_a_linked_studio_home_still_pass(tmp_path, monkeypatch):
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(linked_home))
    ws = _workspace(ALICE)
    (ws / "data.jsonl").write_text("{}")
    for spelling in (
        ws / "data.jsonl",
        real_home / "accounts" / ALICE.account_id / "data.jsonl",
        linked_home / "accounts" / ALICE.account_id / "data.jsonl",
    ):
        assert run_as(ALICE, account_path, str(spelling)) == str(spelling)


def test_plain_workspace_path_and_owner_unchanged(tmp_path):
    ws = _workspace(ALICE)
    assert run_as(ALICE, account_path, str(ws / "x.jsonl")) == str(ws / "x.jsonl")
    assert run_as(OWNER, account_path, "/proc/self/cwd/x") == "/proc/self/cwd/x"


# account_path reference mode: packaged datasets builders are not Hub ids.


@pytest.mark.parametrize("name", ["pandas", "json", "csv", "text", "parquet", "arrow"])
def test_packaged_builder_names_are_not_hub_references(name):
    assert _refused(run_as, ALICE, account_path, name, reference = True) == 403
    assert run_as(OWNER, account_path, name, reference = True) == name


@pytest.mark.parametrize("name", ["imdb", "org/dataset", "unsloth/Llama-3.2-1B"])
def test_real_hub_ids_still_pass_reference_mode(name):
    assert run_as(ALICE, account_path, name, reference = True) == name


# Backslash spellings: the gate judges what the loader opens after normalizing separators.


def _backslash_escape():
    alice_ws = _workspace(ALICE)
    bob_ws = _workspace(BOB)
    (bob_ws / "m.gguf").write_bytes(b"GGUF")
    return f"{alice_ws}/\\..\\{bob_ws.name}\\m.gguf", alice_ws


@pytest.mark.skipif(os.sep == "\\", reason = "a backslash is a separator on Windows")
def test_backslash_spelling_of_another_workspace_is_refused():
    reference, alice_ws = _backslash_escape()
    assert run_as(ALICE, account_access.model_visible, reference) is False
    assert _refused(run_as, ALICE, account_path, reference) == 403
    assert _refused(run_as, ALICE, account_access.private_directory, reference, "models") == 404
    (alice_ws / "sub").mkdir()
    assert run_as(ALICE, account_access.model_visible, str(alice_ws / "sub" / "x.gguf")) is True


# Weight index entries pointing outside the model folder.


def _write(folder, files):
    folder.mkdir(parents = True, exist_ok = True)
    for name, payload in files.items():
        (folder / name).write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return folder


@pytest.mark.parametrize(
    "files",
    [
        {"model.safetensors.index.json": {"weight_map": {"w": "../victim/model.safetensors"}}},
        {"pytorch_model.bin.index.json": {"weight_map": {"w": "/abs/pytorch_model.bin"}}},
        {"config.json": {"transformers_weights": "../victim/model.safetensors.index.json"}},
    ],
    ids = ["safetensors_dotdot", "bin_absolute", "transformers_weights"],
)
def test_escaping_weight_references_flagged(tmp_path, files):
    from utils.security.file_security import escaping_weight_references
    assert escaping_weight_references(_write(tmp_path / "m", files))


def test_normal_weight_index_is_clean(tmp_path):
    from utils.security.file_security import escaping_weight_references
    folder = _write(
        tmp_path / "m",
        {
            "config.json": {"model_type": "llama"},
            "model.safetensors.index.json": {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "sub/model-00002-of-00002.safetensors",
                    # Walks back up but lands inside the folder, where the loader reads it.
                    "c": "sub/../model-00001-of-00002.safetensors",
                }
            },
        },
    )
    assert escaping_weight_references(folder) == []


# Export: what loads is what was checked.


class _RecordingLoader:
    calls: list = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return types.SimpleNamespace(), types.SimpleNamespace()


def _export_backend(monkeypatch):
    from test_export_absolute_paths import _install_export_backend_stubs, _load_module

    _install_export_backend_stubs(monkeypatch)
    mod = _load_module("test_core_export_backend_confinement", "core/export/export.py", monkeypatch)
    _RecordingLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _RecordingLoader)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    return backend


def test_export_refuses_an_adapter_whose_base_changed_after_the_check(monkeypatch, tmp_path):
    backend = _export_backend(monkeypatch)
    checkpoint = _write(
        tmp_path / "ckpt",
        {"adapter_config.json": {"base_model_name_or_path": "other/private-base"}},
    )
    ok, _ = backend.load_checkpoint(str(checkpoint), base_model = "alice/checked-base")
    assert not ok
    assert _RecordingLoader.calls == []


def test_export_refuses_a_weight_index_outside_the_checkpoint(monkeypatch, tmp_path):
    backend = _export_backend(monkeypatch)
    checkpoint = _write(
        tmp_path / "ckpt",
        {"model.safetensors.index.json": {"weight_map": {"w": "../victim/model.safetensors"}}},
    )
    ok, _ = backend.load_checkpoint(str(checkpoint))
    assert not ok
    assert _RecordingLoader.calls == []


def test_export_still_loads_an_unchanged_adapter(monkeypatch, tmp_path):
    backend = _export_backend(monkeypatch)
    checkpoint = _write(
        tmp_path / "ckpt",
        {"adapter_config.json": {"base_model_name_or_path": "alice/checked-base"}},
    )
    ok, message = backend.load_checkpoint(str(checkpoint), base_model = "alice/checked-base")
    assert ok, message
    assert len(_RecordingLoader.calls) == 1


def test_export_route_authorizes_the_decision_config_base(tmp_path):
    from routes import export as export_routes

    ws = _workspace(ALICE)
    own_base = _write(ws / "base", {"config.json": {"model_type": "llama"}})
    foreign = _write(tmp_path / "foreign-base", {"config.json": {"model_type": "llama"}})
    checkpoint = _write(
        ws / "outputs" / "run",
        {
            "adapter_config.json": {"base_model_name_or_path": str(own_base)},
            "unsloth_decision_config.json": {"base_model": str(foreign)},
        },
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"x")
    assert _refused(run_as, ALICE, export_routes._authorized_adapter_base, str(checkpoint)) == 404
    del_decision = checkpoint / "unsloth_decision_config.json"
    del_decision.unlink()
    assert run_as(ALICE, export_routes._authorized_adapter_base, str(checkpoint)) == str(own_base)


# Training: a folder with full weights and an adapter config still has its base authorized.


def test_training_preflight_authorizes_adapter_base_beside_full_weights(tmp_path, monkeypatch):
    from models.training import TrainingStartRequest
    from routes import training

    ws = _workspace(ALICE)
    foreign = _write(tmp_path / "owner-base", {"config.json": {"model_type": "llama"}})
    (foreign / "model.safetensors").write_bytes(b"x" * 16)
    folder = _write(
        ws / "models" / "mixed",
        {
            "config.json": {"model_type": "llama"},
            "adapter_config.json": {"base_model_name_or_path": str(foreign)},
        },
    )
    (folder / "model.safetensors").write_bytes(b"x" * 16)
    (folder / "adapter_model.safetensors").write_bytes(b"x" * 16)
    monkeypatch.setattr(training, "_has_trainable_local_weights", lambda *a, **k: True)
    request = TrainingStartRequest(
        model_name = str(folder),
        dataset_name = "public/set",
        hf_token = "x",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    assert _refused(run_as, ALICE, training._reject_untrainable_model_request, request) == 403
    result = run_as(OWNER, training._reject_untrainable_model_request, request)
    assert result.adapter_base_pin is None


def test_a_cached_snapshot_adapter_pin_is_rechecked_against_the_snapshot(tmp_path):
    import json

    from core.training import worker

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    config = {
        "model_name": "org/finetune",
        "model_snapshot_path": str(snapshot),
        "adapter_base_pin": "org/base",
    }
    assert worker._adapter_base_changed_error(config) is None
    assert worker._adapter_base_changed_error({**config, "adapter_base_pin": "org/other"})
