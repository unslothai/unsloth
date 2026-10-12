# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio --models-dir` hands its folders to the backend through the env (#8250)."""

from __future__ import annotations

import os

import pytest
import typer

import unsloth_cli.commands.studio as studio_cli

ENV = "UNSLOTH_STUDIO_MODELS_DIRS"


def test_no_flag_leaves_env_alone(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    studio_cli._export_models_dirs(None)
    studio_cli._export_models_dirs([])
    assert ENV not in os.environ


def test_folders_are_resolved_and_exported(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    studio_cli._export_models_dirs([a, b, a])
    assert os.environ[ENV].split(os.pathsep) == [str(a.resolve()), str(b.resolve())]


def test_existing_env_is_kept(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.setenv(ENV, "/already/set")
    studio_cli._export_models_dirs([a])
    assert os.environ[ENV].split(os.pathsep) == ["/already/set", str(a.resolve())]


def test_missing_folder_exits_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    with pytest.raises(typer.Exit) as exc:
        studio_cli._export_models_dirs([tmp_path / "nope"])
    assert exc.value.exit_code == 2
    assert ENV not in os.environ


def test_option_is_registered_on_studio():
    import inspect
    param = inspect.signature(studio_cli.studio_default).parameters["models_dir"]
    assert "--models-dir" in param.default.param_decls
