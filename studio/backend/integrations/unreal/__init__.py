# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installs Unsloth's Unreal Editor toolsets (level saving, game templates) and the Unreal call_tool guide."""

from pathlib import Path

TOOLSET_MODULE = "unsloth_level_tools"
GAME_TOOLSET_MODULE = "unsloth_game_tools"
_SOURCE = Path(__file__).with_name(f"{TOOLSET_MODULE}.py")
_GAME_SOURCE = Path(__file__).with_name(f"{GAME_TOOLSET_MODULE}.py")
# PythonScriptPlugin runs init_unreal.py from every Content/Python folder at editor startup.
_HOOK_MARKER = "# Unsloth level tools (installed by Unsloth Studio)"
_GAME_HOOK_MARKER = "# Unsloth game tools (installed by Unsloth Studio)"


def _hook(marker: str, module: str, label: str) -> str:
    # Guarded: with the Unreal MCP plugin off, toolset_registry is missing and an unguarded import would
    # also skip anything the user adds to init_unreal.py after this block.
    return f"""{marker}
try:
    import {module}
except ImportError as _unsloth_exc:
    import unreal

    unreal.log_warning(f"Unsloth {label} not loaded (enable the Unreal MCP plugin): {{_unsloth_exc}}")
else:
    {module}.register_when_ready()
"""


_HOOK = _hook(_HOOK_MARKER, TOOLSET_MODULE, "level tools")
_GAME_HOOK = _hook(_GAME_HOOK_MARKER, GAME_TOOLSET_MODULE, "game tools")
_TOOLSETS = ((_SOURCE, _HOOK_MARKER, _HOOK), (_GAME_SOURCE, _GAME_HOOK_MARKER, _GAME_HOOK))

# Appended to the Unreal built-in's call_tool description, so every chat with Unreal enabled sees it.
CALL_GUIDE = """
Unreal tips: call tools by their short name with toolset_name (describe_toolset lists them); pass every parameter in
the schema, null for optional objects. For a playable game start from Epic's templates:
unsloth_game_tools.GameTemplateTools add_game_template, then set_level_game_mode and place_player_start; do not
build a player pawn, input or vehicle physics by hand. Save with unsloth_level_tools.LevelSaveTools
(save_current_level_as once, then save_current_level) before StartPIE: Play In Editor on a never-saved level can crash
the editor. After StartPIE, check get_play_state (and set_vehicle_input for a car) before calling the game playable.
In execute_tool_script define run(); only json, math, re, time, copy and datetime import (no random). Never delete
existing /Game folders unless the user asks."""


def project_root(project_path: str) -> Path:
    """The folder holding the .uproject, from that folder or the .uproject file itself."""
    # Windows Explorer's "Copy as path" wraps the path in double quotes.
    raw = project_path.strip().strip('"').strip()
    if not raw:
        raise ValueError("Enter the folder that holds your .uproject file.")
    path = Path(raw).expanduser()
    root = path.parent if path.suffix.lower() == ".uproject" else path
    if not root.is_dir() or not any(root.glob("*.uproject")):
        raise ValueError(f"No .uproject file in {root}. Enter your Unreal project folder.")
    return root.resolve()


def install(project_path: str) -> list[str]:
    """Writes the toolsets into <project>/Content/Python and hooks each into init_unreal.py once; safe to
    repeat. An existing init_unreal.py keeps its contents and gets the missing hooks appended."""
    python_dir = project_root(project_path) / "Content" / "Python"
    python_dir.mkdir(parents = True, exist_ok = True)
    files = []
    hook = python_dir / "init_unreal.py"
    existing = hook.read_text(encoding = "utf-8") if hook.exists() else ""
    text = existing
    for source, marker, block in _TOOLSETS:
        module = python_dir / source.name
        module.write_text(source.read_text(encoding = "utf-8"), encoding = "utf-8")
        files.append(str(module))
        if marker not in text:
            text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block
    if text != existing:
        hook.write_text(text, encoding = "utf-8")
    return files + [str(hook)]
