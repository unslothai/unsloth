# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth's Unreal level-save toolset: the installer Studio runs, and the module the editor runs."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from integrations import unreal as unreal_tools
from models.mcp_servers import UnrealToolsInstall
from routes import mcp_servers as routes


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "MyGame"
    root.mkdir()
    (root / "MyGame.uproject").write_text("{}", encoding = "utf-8")
    return root


def test_install_writes_the_toolset_and_hooks_it_once(tmp_path):
    root = _project(tmp_path)
    python_dir = root / "Content" / "Python"
    files = unreal_tools.install(str(root / "MyGame.uproject"))
    assert files == [str(python_dir / "unsloth_level_tools.py"), str(python_dir / "init_unreal.py")]
    assert (python_dir / "unsloth_level_tools.py").read_text() == unreal_tools._SOURCE.read_text()
    hook = (python_dir / "init_unreal.py").read_text()
    assert "unsloth_level_tools.register_when_ready()" in hook
    unreal_tools.install(str(root))  # the folder works too, and a repeat adds nothing
    assert (python_dir / "init_unreal.py").read_text() == hook


def test_install_keeps_an_existing_init_unreal(tmp_path):
    root = _project(tmp_path)
    python_dir = root / "Content" / "Python"
    python_dir.mkdir(parents = True)
    (python_dir / "init_unreal.py").write_text(
        "import my_tools\nmy_tools.start()", encoding = "utf-8"
    )
    unreal_tools.install(str(root))
    text = (python_dir / "init_unreal.py").read_text()
    assert text.startswith("import my_tools\nmy_tools.start()\n\n# Unsloth level tools")
    assert text.count("register_when_ready()") == 1


def test_install_accepts_a_path_copied_with_quotes(tmp_path):
    root = _project(tmp_path)
    unreal_tools.install(f'  "{root / "MyGame.uproject"}" ')
    assert (root / "Content" / "Python" / "init_unreal.py").exists()


@pytest.mark.parametrize("plugin_on", [True, False])
def test_hook_never_stops_the_rest_of_init_unreal(tmp_path, monkeypatch, plugin_on):
    """Without the Unreal MCP plugin the toolset cannot import; the hook warns instead of raising."""
    root = _project(tmp_path)
    unreal_tools.install(str(root))
    hook = (root / "Content" / "Python" / "init_unreal.py").read_text() + "\nafter_hook = True\n"
    warnings, registered = [], []
    monkeypatch.setitem(sys.modules, "unreal", types.SimpleNamespace(log_warning = warnings.append))
    if plugin_on:
        toolset = types.SimpleNamespace(register_when_ready = lambda: registered.append(True))
        monkeypatch.setitem(sys.modules, "unsloth_level_tools", toolset)
    else:
        monkeypatch.setitem(sys.modules, "unsloth_level_tools", None)  # import raises ImportError
    scope: dict = {}
    exec(compile(hook, "init_unreal.py", "exec"), scope)
    assert scope["after_hook"] is True
    assert registered == ([True] if plugin_on else [])
    assert len(warnings) == (0 if plugin_on else 1)


@pytest.mark.parametrize("path", ["", "missing", "no_uproject"])
def test_install_needs_a_project_folder(tmp_path, path):
    (tmp_path / "no_uproject").mkdir()
    with pytest.raises(ValueError):
        unreal_tools.install(str(tmp_path / path) if path else "")
    assert not list(tmp_path.rglob("init_unreal.py"))


def test_install_route_follows_the_editor_tools_gate(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: True)
    for auth in ({"via_api_key": True}, {"no_credential": True}):
        with pytest.raises(HTTPException):
            routes.install_unreal_tools(UnrealToolsInstall(project_path = str(root)), **auth)
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: False)  # e.g. a public --secure link
    with pytest.raises(HTTPException):
        routes.install_unreal_tools(UnrealToolsInstall(project_path = str(root)))
    assert not (root / "Content").exists()
    monkeypatch.setattr(routes, "stdio_mcp_enabled", lambda: True)
    with pytest.raises(HTTPException) as bad:
        routes.install_unreal_tools(UnrealToolsInstall(project_path = str(tmp_path / "nope")))
    assert bad.value.status_code == 400
    result = routes.install_unreal_tools(UnrealToolsInstall(project_path = str(root)))
    assert result.restart_required and len(result.files) == 2


class _FakeEditor:
    """Just enough of the `unreal` module for the toolset's logic."""

    def __init__(
        self,
        level = "/Temp/Untitled_1",
        existing = (),
    ):
        self.level, self.existing, self.calls = level, set(existing), []
        self.ticks = []
        self.assets_save = True
        world = types.SimpleNamespace(
            get_outermost = lambda: types.SimpleNamespace(get_name = lambda: self.level)
        )
        editor = self

        class Utils:
            @staticmethod
            def save_map(w, path):
                editor.calls.append(("save_map", w is world, path))
                editor.level = path
                return True

            @staticmethod
            def save_dirty_packages(maps, content):
                editor.calls.append(("save_dirty_packages", maps, content))
                return editor.assets_save

        self.module = types.SimpleNamespace(
            uclass = lambda: (lambda cls: cls),
            ToolsetDefinition = object,
            UnrealEditorSubsystem = object,
            get_editor_subsystem = lambda _cls: types.SimpleNamespace(get_editor_world = lambda: world),
            EditorAssetLibrary = types.SimpleNamespace(does_asset_exist = lambda p: p in self.existing),
            EditorLoadingAndSavingUtils = Utils,
            log = lambda _msg: None,
            register_slate_post_tick_callback = lambda fn: self.ticks.append(fn) or "handle",
            unregister_slate_post_tick_callback = lambda handle: self.ticks.clear(),
        )


def _load_toolset(
    monkeypatch,
    editor: _FakeEditor,
    registry_up = True,
):
    state = {"up": registry_up, "registered": []}

    class Registration:
        def __init__(self, classes):
            self.classes = list(classes)

        def register(self):
            if state["up"]:
                state["registered"].extend(self.classes)
            return state["up"]

    registry = types.ModuleType("toolset_registry")
    registry.tool_call = lambda fn: fn
    registration = types.ModuleType("toolset_registry.registration")
    registration.Registration = Registration
    monkeypatch.setitem(sys.modules, "unreal", editor.module)
    monkeypatch.setitem(sys.modules, "toolset_registry", registry)
    monkeypatch.setitem(sys.modules, "toolset_registry.registration", registration)
    spec = importlib.util.spec_from_file_location(
        "unsloth_level_tools_under_test", unreal_tools._SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, state


def test_toolset_saves_a_new_level_then_the_current_one(monkeypatch):
    editor = _FakeEditor()
    toolset, _ = _load_toolset(monkeypatch, editor)
    tools = toolset.LevelSaveTools
    assert tools.get_current_level_path() == "/Temp/Untitled_1"
    with pytest.raises(RuntimeError, match = "never been saved"):
        tools.save_current_level()
    with pytest.raises(RuntimeError, match = "/Game/"):
        tools.save_current_level_as("/Temp/City")
    editor.existing.add("/Game/Maps/Taken")
    with pytest.raises(RuntimeError, match = "already exists"):
        tools.save_current_level_as("/Game/Maps/Taken")
    assert editor.calls == []
    assert tools.save_current_level_as(" /Game/Maps/City/ ") == "/Game/Maps/City"
    assert editor.calls == [
        ("save_map", True, "/Game/Maps/City"),
        ("save_dirty_packages", True, True),
    ]
    assert tools.save_current_level() == "/Game/Maps/City"


def test_toolset_registers_once_the_registry_is_up(monkeypatch):
    editor = _FakeEditor()
    toolset, state = _load_toolset(monkeypatch, editor, registry_up = False)
    toolset.register_when_ready()
    toolset.register_when_ready()  # a second call does not stack another tick callback
    assert len(editor.ticks) == 1 and state["registered"] == []
    state["up"] = True
    editor.ticks[0](0.016)
    assert state["registered"] == [toolset.LevelSaveTools]
    assert editor.ticks == []


def test_save_as_reports_assets_that_did_not_save(monkeypatch):
    editor = _FakeEditor()
    toolset, _ = _load_toolset(monkeypatch, editor)
    editor.assets_save = False  # e.g. a read-only Blueprint package
    with pytest.raises(RuntimeError, match = "some other assets were not"):
        toolset.LevelSaveTools.save_current_level_as("/Game/Maps/City")
