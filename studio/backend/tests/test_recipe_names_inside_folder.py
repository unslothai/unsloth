# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recipe strings that become path segments must not hold a path separator: column names become
`<tmp>/<record id>_<column>.py` in the code validator, and GitHub seed repo names become
`<base>/<owner>__<name>` in the scraper."""

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


@pytest.mark.parametrize("name", _ESCAPING_NAMES)
def test_a_column_name_that_names_a_path_is_refused(name):
    recipe = {"columns": [{"name": name, "column_type": "expression", "expr": "1"}]}
    # build_config_builder is what validate, preview and the job worker all call.
    with pytest.raises(ValueError):
        service.build_config_builder(recipe)


@pytest.mark.parametrize("name", _ESCAPING_NAMES)
def test_a_validator_target_that_names_a_path_is_refused(name):
    # The target can be a seed column whose name comes from the seed file, not from `columns`.
    column = {
        "name": "lint",
        "column_type": "validation",
        "validator_type": "code",
        "validator_params": {"code_lang": "python"},
        "target_columns": ["code", name],
    }
    with pytest.raises(ValueError):
        service.build_config_builder({"columns": [column]})


@pytest.mark.parametrize("name", [".", ".."])
def test_a_dot_only_processor_name_is_refused(name):
    # Data Designer writes processor batches under processors-files/<name>/.
    recipe = {"processors": [{"name": name, "processor_type": "schema_transform"}]}
    with pytest.raises(ValueError):
        service.validate_recipe_names(recipe)


@pytest.mark.parametrize("name", ["answer", "code_v2", "a.b", "Question 1", "Step 1: answer"])
def test_ordinary_column_names_pass(name):
    service.validate_recipe_names({"columns": [{"name": name}], "processors": [{"name": "drop"}]})


def test_job_with_an_escaping_column_name_is_refused_before_it_starts(monkeypatch):
    import routes.data_recipe.jobs as jobs_routes
    from models.data_recipe import RecipePayload

    started = []
    monkeypatch.setattr(
        jobs_routes,
        "get_job_manager",
        lambda: type("M", (), {"start": lambda self, **k: started.append(k)})(),
    )
    recipe = {"columns": [{"name": "x/../../marker", "column_type": "expression", "expr": "1"}]}
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
