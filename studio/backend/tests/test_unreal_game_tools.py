# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth's Unreal game-template toolset, with a stand-in `unreal` module and an engine Templates folder laid out
like UE 5.8's (TP_VehicleAdvBP + TemplateResources shared packs)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from core.inference import tools
from integrations import unreal as unreal_tools


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return path


def _engine(root: Path) -> Path:
    tp = root / "Templates" / "TP_VehicleAdvBP"
    _write(
        tp / "TP_VehicleAdvBP.uproject",
        json.dumps(
            {
                "Plugins": [
                    {
                        "Name": "GameInputWindows",
                        "Enabled": True,
                        "SupportedTargetPlatforms": ["Win64"],
                    },
                    {
                        "Name": "ModelingToolsEditorMode",
                        "Enabled": True,
                        "TargetAllowList": ["Editor"],
                    },
                    {"Name": "ChaosVehiclesPlugin", "Enabled": True},
                ]
            }
        ),
    )
    _write(
        tp / "Config" / "DefaultEngine.ini",
        "[/Script/EngineSettings.GameMapsSettings]\n"
        "EditorStartupMap=/Game/VehicleTemplate/Maps/Lvl_VehicleBasic.Lvl_VehicleBasic\n"
        "GlobalDefaultGameMode=/Game/VehicleTemplate/Blueprints/BP_VehicleAdvGameMode.BP_VehicleAdvGameMode_C\n",
    )
    _write(
        tp / "Config" / "TemplateDefs.ini",
        "﻿[/Script/GameProjectGeneration.TemplateProjectDefs]\n"
        'SharedContentPacks=(MountName="Vehicles",DetailLevels=("Standard"))\n'
        'SharedContentPacks=(MountName="Input",DetailLevels=("High"))\n'
        'Variants=(Name="Offroad",SharedContentPacks=((DetailLevels=(Standard),MountName="Variant_Offroad")))\n',
    )
    _write(tp / "Content" / "VehicleTemplate" / "Blueprints" / "BP_VehicleAdvGameMode.uasset")
    _write(tp / "Content" / "__ExternalActors__" / "VehicleTemplate" / "Maps" / "A.uasset")
    resources = root / "Templates" / "TemplateResources"
    _write(resources / "Standard" / "Vehicles" / "Content" / "SportsCar" / "SM_SportsCar.uasset")
    _write(resources / "High" / "Input" / "Content" / "IMC_Default.uasset")
    _write(resources / "Standard" / "Variant_Offroad" / "Content" / "Offroad.uasset")
    _write(root / "Templates" / "TP_Blank" / "TP_Blank.uproject", "{}")  # C++ template: not offered
    return root


def _load(
    monkeypatch,
    engine_root: Path,
    project: Path,
    loadable: bool,
    running = None,
):
    """running: plugin names the editor session started with (None = an engine without PluginBlueprintLibrary)."""
    scans = []
    fake = types.SimpleNamespace(
        uclass = lambda: (lambda cls: cls),
        ToolsetDefinition = object,
        Paths = types.SimpleNamespace(
            root_dir = lambda: str(engine_root),
            convert_relative_path_to_full = lambda p: p,
            project_content_dir = lambda: str(project / "Content"),
            get_project_file_path = lambda: str(project / "Game.uproject"),
        ),
        AssetRegistryHelpers = types.SimpleNamespace(
            get_asset_registry = lambda: types.SimpleNamespace(
                scan_paths_synchronous = lambda paths, force: scans.append(list(paths))
            )
        ),
        load_class = lambda outer, path: object() if loadable else None,
        log = lambda _msg: None,
    )
    if running is not None:
        fake.PluginBlueprintLibrary = types.SimpleNamespace(
            get_enabled_plugin_names = lambda: list(running)
        )
    registry = types.ModuleType("toolset_registry")
    registry.tool_call = lambda fn: fn
    registration = types.ModuleType("toolset_registry.registration")
    registration.Registration = lambda classes: types.SimpleNamespace(register = lambda: True)
    monkeypatch.setitem(sys.modules, "unreal", fake)
    monkeypatch.setitem(sys.modules, "toolset_registry", registry)
    monkeypatch.setitem(sys.modules, "toolset_registry.registration", registration)
    spec = importlib.util.spec_from_file_location(
        "unsloth_game_tools_under_test", unreal_tools._GAME_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GameTemplateTools, scans


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "Game"
    _write(
        root / "Game.uproject",
        json.dumps(
            {"FileVersion": 3, "Plugins": [{"Name": "ModelContextProtocol", "Enabled": True}]}
        ),
    )
    return root


def test_templates_are_epics_blueprint_ones(monkeypatch, tmp_path, project):
    tools_, _ = _load(monkeypatch, _engine(tmp_path / "UE"), project, loadable = True)
    listed = json.loads(tools_.list_game_templates())
    assert listed == {
        "vehicleadv": {
            "game_mode": "/Game/VehicleTemplate/Blueprints/BP_VehicleAdvGameMode.BP_VehicleAdvGameMode_C",
            "example_level": "/Game/VehicleTemplate/Maps/Lvl_VehicleBasic",
            "plugins": [
                "ChaosVehiclesPlugin"
            ],  # platform- and target-limited entries are not required
        }
    }


def test_adding_a_template_copies_content_and_packs_and_keeps_project_files(
    monkeypatch, tmp_path, project
):
    engine = _engine(tmp_path / "UE")
    mine = _write(
        project / "Content" / "Vehicles" / "SportsCar" / "SM_SportsCar.uasset",
        "the user's own edit",
    )
    tools_, scans = _load(monkeypatch, engine, project, loadable = True)
    result = json.loads(tools_.add_game_template(" Vehicle_Adv "))
    assert result["game_mode"].endswith("BP_VehicleAdvGameMode_C")
    assert result["example_level"] == "/Game/VehicleTemplate/Maps/Lvl_VehicleBasic"
    assert result["content_folders"] == ["/Game/VehicleTemplate", "/Game/Vehicles", "/Game/Input"]
    assert scans == [result["content_folders"]]  # registered mid-session, no restart
    assert (
        result["files_copied"] == 3
    )  # game mode BP, external actor, input; the existing car mesh is kept
    assert mine.read_text() == "the user's own edit"
    assert (
        project / "Content" / "__ExternalActors__" / "VehicleTemplate" / "Maps" / "A.uasset"
    ).exists()
    assert not (project / "Content" / "Variant_Offroad").exists()  # variants are opt-in
    plugins = json.loads((project / "Game.uproject").read_text())["Plugins"]
    assert {"Name": "ChaosVehiclesPlugin", "Enabled": True} in plugins and len(plugins) == 2
    assert (
        result["plugins_enabled"] == ["ChaosVehiclesPlugin"] and result["restart_required"] is False
    )
    again = json.loads(tools_.add_game_template("vehicleadv"))
    assert again["files_copied"] == 0 and again["plugins_enabled"] == []


@pytest.mark.parametrize("running", [None, ["ModelContextProtocol"]])
def test_a_plugin_that_was_not_loaded_asks_for_a_restart_until_restarted(
    monkeypatch, tmp_path, project, running
):
    tools_, _ = _load(
        monkeypatch, _engine(tmp_path / "UE"), project, loadable = False, running = running
    )
    result = json.loads(tools_.add_game_template("vehicleadv"))
    assert (
        result["plugins_enabled"] == ["ChaosVehiclesPlugin"] and result["restart_required"] is True
    )
    # A second call (or another template sharing the plugin) changes nothing in the .uproject; the plugin is still
    # not running, so the restart is still owed.
    again = json.loads(tools_.add_game_template("vehicleadv"))
    assert again["plugins_enabled"] == [] and again["restart_required"] is True
    assert again["plugins_not_running"] == ["ChaosVehiclesPlugin"]


def test_a_running_plugin_needs_no_restart_even_if_the_uproject_lacked_it(
    monkeypatch, tmp_path, project
):
    running = [
        "ModelContextProtocol",
        "ChaosVehiclesPlugin",
    ]  # enabled by the engine or another plugin
    tools_, _ = _load(
        monkeypatch, _engine(tmp_path / "UE"), project, loadable = True, running = running
    )
    result = json.loads(tools_.add_game_template("vehicleadv"))
    assert result["plugins_enabled"] == ["ChaosVehiclesPlugin"]
    assert result["plugins_not_running"] == [] and result["restart_required"] is False


def test_unknown_template_names_the_real_ones(monkeypatch, tmp_path, project):
    tools_, _ = _load(monkeypatch, _engine(tmp_path / "UE"), project, loadable = True)
    with pytest.raises(RuntimeError, match = r"one of \['vehicleadv'\]"):
        tools_.add_game_template("racing")


def test_the_call_guide_rides_on_unreals_call_tool_only():
    mcp_tools = [
        {"name": "call_tool", "description": "Calls a tool.", "inputSchema": {"type": "object"}}
    ]
    unreal = tools._mcp_specs_for_server(
        {"id": "u", "builtin_id": "unreal", "display_name": "Unreal Engine"}, mcp_tools
    )
    other = tools._mcp_specs_for_server(
        {"id": "o0123456789abcde", "display_name": "Other"}, mcp_tools
    )
    assert "add_game_template" in unreal[0]["function"]["description"]
    assert "save_current_level_as" in unreal[0]["function"]["description"]
    assert other[0]["function"]["description"] == "[Other] Calls a tool."
