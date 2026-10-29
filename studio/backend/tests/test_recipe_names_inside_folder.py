# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recipe strings that become path segments stay inside their folder: column names
(`<tmp>/<record id>_<column>.py`, sanitized), processor names (`processors-files/<name>/`) and
GitHub seed repo names (`<base>/<owner>__<name>`)."""

import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from core.data_recipe import service

_BACKEND = Path(__file__).resolve().parents[1]
_PLUGIN_SRC = _BACKEND / "plugins/data-designer-github-repo-seed/src"
_B = chr(92)

_ESCAPING_NAMES = ["x/../../marker", ".." + _B + ".." + _B + "x"]
_SLASH_NAMES = ["pros/cons", "input/output", "Q/A", "a" + _B + "b"]


@pytest.mark.parametrize(
    "name", _SLASH_NAMES + _ESCAPING_NAMES + ["answer", "a.b", "Step 1: answer"]
)
def test_any_column_or_validator_target_name_passes(name):
    column = {
        "name": "lint",
        "column_type": "validation",
        "validator_type": "code",
        "target_columns": ["code", name],
    }
    service.validate_recipe_names(
        {"columns": [{"name": name}, column], "processors": [{"name": "drop"}]}
    )


@pytest.mark.parametrize("name", ["drop", "drop.v2", "pros and cons"])
def test_a_single_folder_processor_name_passes(name):
    service.validate_recipe_names(
        {"processors": [{"name": name, "processor_type": "schema_transform"}]}
    )


@pytest.mark.parametrize(
    "name",
    [
        ".",
        "..",
        "../x",
        "a/../../x",
        ".." + _B + "x",
        "/etc/x",
        _B + "x",
        "C:x",
        "c:" + _B + "x",
        "pros/cons",
        "a/b/c",
        "x/../y",
        "a" + _B + "b",
    ],
)
def test_a_processor_name_that_is_not_one_folder_is_refused(name):
    # Data Designer writes processor batches under processors-files/<name>/ and previews to
    # processors-files/<name>.parquet without creating a nested parent.
    recipe = {"processors": [{"name": name, "processor_type": "schema_transform"}]}
    with pytest.raises(ValueError):
        service.validate_recipe_names(recipe)


@pytest.fixture
def python_validator(monkeypatch):
    # The two methods below are data_designer 0.5.4's PythonValidator, verbatim: the patch must keep
    # writing and reading back the same module name.
    class PythonValidator:
        def _write_code_to_file(self, row, code_column, path):
            with open(
                f"{path}/{self._get_module_name(row[RECORD_ID], code_column)}.py", "w"
            ) as file:
                file.write(row[code_column])

        @staticmethod
        def _get_module_name(record_id, column_name):
            return f"{record_id}_{column_name}"

    RECORD_ID = "internal_code_record_id"
    module = types.ModuleType("data_designer.engine.validators.python")
    module.PythonValidator = PythonValidator
    module.RECORD_ID_COLUMN_NAME = RECORD_ID
    for name in ("data_designer", "data_designer.engine", "data_designer.engine.validators"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    service._apply_data_designer_python_validator_path_patch()
    service._apply_data_designer_python_validator_path_patch()
    return PythonValidator, RECORD_ID


@pytest.mark.parametrize("name", _SLASH_NAMES + _ESCAPING_NAMES + ["code", "Q 1"])
def test_code_validator_file_stays_in_its_temp_folder(tmp_path, python_validator, name):
    validator_cls, record_id = python_validator
    box = tmp_path / "box" / "tmp"
    box.mkdir(parents = True)
    validator_cls()._write_code_to_file({record_id: "rid", name: "x = 1\n"}, name, str(box))

    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert [p.parent for p in written] == [box]
    module = validator_cls._get_module_name("rid", name)
    assert written[0].name == module + ".py"
    assert written[0].read_text() == "x = 1\n"
    assert validator_cls._get_module_name("rid", name) == module
    if name == "code":
        assert module == "rid_code"


def test_different_names_that_sanitize_alike_get_different_files(python_validator):
    validator_cls, _ = python_validator
    assert validator_cls._get_module_name("r", "a/b") != validator_cls._get_module_name("r", "a_b")
    assert validator_cls._get_module_name("r", "a/b") != validator_cls._get_module_name(
        "r", "a" + _B + "b"
    )


def test_job_with_a_slash_column_name_starts(monkeypatch):
    import routes.data_recipe.jobs as jobs_routes
    from models.data_recipe import RecipePayload

    started = []
    monkeypatch.setattr(
        jobs_routes,
        "get_job_manager",
        lambda: type("M", (), {"start": lambda self, **k: started.append(k) or "job"})(),
    )
    recipe = {"columns": [{"name": "pros/cons", "column_type": "expression", "expr": "1"}]}
    jobs_routes.create_job(
        RecipePayload(recipe = recipe), request = None, credential = ("u", None), via_api_key = False
    )
    assert len(started) == 1


def test_job_with_an_escaping_processor_name_is_refused_before_it_starts(monkeypatch):
    import routes.data_recipe.jobs as jobs_routes
    from models.data_recipe import RecipePayload

    started = []
    monkeypatch.setattr(
        jobs_routes,
        "get_job_manager",
        lambda: type("M", (), {"start": lambda self, **k: started.append(k)})(),
    )
    recipe = {
        "columns": [{"name": "a", "column_type": "expression", "expr": "1"}],
        "processors": [{"name": "../../marker", "processor_type": "schema_transform"}],
    }
    with pytest.raises(HTTPException) as exc:
        jobs_routes.create_job(
            RecipePayload(recipe = recipe), request = None, credential = ("u", None), via_api_key = False
        )
    assert exc.value.status_code == 400
    assert started == []


@pytest.fixture
def plugin_config(monkeypatch):
    # The plugin's base class comes from the recipe engine; a plain pydantic model stands in.
    seed_source = types.ModuleType("data_designer.config.seed_source")
    seed_source.SeedSource = type("SeedSource", (BaseModel,), {})
    for name in ("data_designer", "data_designer.config"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "data_designer.config.seed_source", seed_source)
    for name in list(sys.modules):
        if name.startswith("data_designer_github_repo_seed"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(_PLUGIN_SRC))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "data_designer_github_repo_seed_config_under_test",
        _PLUGIN_SRC / "data_designer_github_repo_seed/config.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "repo", [".." + _B + ".." + _B + "OUTSIDE" + _B + "x/y", "owner/..", "owner/" + _B + "x"]
)
def test_plugin_refuses_a_repo_that_is_not_owner_slash_name(plugin_config, repo):
    with pytest.raises(ValueError):
        plugin_config.GitHubRepoSeedSource(repos = [repo], token = "t")


def test_plugin_accepts_a_real_repo(plugin_config):
    source = plugin_config.GitHubRepoSeedSource(
        repos = ["unslothai/unsloth", "my-org/my.repo_2"], token = "t"
    )
    assert source.repos == ["unslothai/unsloth", "my-org/my.repo_2"]
