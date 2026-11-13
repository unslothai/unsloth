# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MTP sidecar discovery must distinguish an absent file from a failed lookup."""

import ast
import os
from pathlib import Path
import sys
import types

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
import pytest


@pytest.fixture
def reattach():
    # Importing the full loader requires the accelerator stack; execute its actual helper here.
    path = Path(__file__).resolve().parents[1] / "unsloth/models/loader.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_reattach_saved_mtp_head")
    namespace = {"os": os, "__package__": "unsloth.models"}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize(
    "failure", [LocalEntryNotFoundError("not cached"), OSError("connection reset")]
)
def test_uncertain_remote_lookup_warns_without_exposing_error_details(reattach, monkeypatch, capsys, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    reattach(object(), "org/adapter", token = "secret-value", local_files_only = True)
    output = capsys.readouterr().out
    assert "could not check" in output.lower()
    assert "MTP" in output and "stock" in output
    assert "secret-value" not in output and str(failure) not in output


def test_remote_file_absence_is_still_silent(reattach, monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    reattach(object(), "org/adapter")
    assert capsys.readouterr().out == ""


def test_actual_offline_cache_miss_warns(reattach, tmp_path, capsys):
    # This exercises the installed Hub version's real exception hierarchy without network access.
    reattach(object(), "offline/not-cached", cache_dir = str(tmp_path), local_files_only = True)
    output = capsys.readouterr().out
    assert "could not check" in output.lower()
    assert "stock" in output


def test_actual_offline_cached_sidecar_is_reattached(reattach, tmp_path, monkeypatch, capsys):
    repo = tmp_path / "models--org--adapter"
    revision = "a" * 40
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents = True)
    (snapshot / "mtp_head.safetensors").write_bytes(b"head")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision, encoding = "utf-8")

    attached = []
    module = types.ModuleType("unsloth.models.speculative")
    module.attach_saved_mtp_head = lambda model, folder: attached.append(folder)
    monkeypatch.setitem(sys.modules, "unsloth.models.speculative", module)

    reattach(object(), "org/adapter", cache_dir = str(tmp_path), local_files_only = True)
    assert attached == [str(snapshot)]
    assert "Could not check" not in capsys.readouterr().out


def test_existing_local_and_remote_sidecars_still_attach(reattach, tmp_path, monkeypatch):
    weights = tmp_path / "mtp_head.safetensors"
    weights.write_bytes(b"head")
    attached = []
    module = types.ModuleType("unsloth.models.speculative")
    module.attach_saved_mtp_head = lambda model, folder: attached.append(folder)
    monkeypatch.setitem(sys.modules, "unsloth.models.speculative", module)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(weights))
    reattach(object(), str(tmp_path))
    reattach(object(), "org/adapter")
    assert attached == [str(tmp_path), str(tmp_path)]
