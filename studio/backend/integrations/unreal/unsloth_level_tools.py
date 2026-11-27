# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs inside the Unreal Editor, not in Studio: Studio copies this file into a project's
Content/Python. Epic's Unreal MCP toolsets can build a whole scene but cannot save the level, so
on a headless editor (no Ctrl+S) everything an agent built is lost when the editor stops. This adds
an MCP toolset, unsloth_level_tools.LevelSaveTools, through Epic's own toolset registry."""

import unreal

import toolset_registry
from toolset_registry.registration import Registration


def _current_level_path() -> str:
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    return world.get_outermost().get_name() if world else ""


@unreal.uclass()
class LevelSaveTools(unreal.ToolsetDefinition):
    """Saves the level open in the editor, including a never-saved (Untitled, /Temp/...) level, and
    every other unsaved asset. Call save_current_level_as once for a new level, then save_current_level.
    Save before starting Play In Editor: PIE on a never-saved World Partition level (the default Open
    World template) can crash the editor (UE 5.8 'Levels mismatch' assertion), losing everything unsaved."""

    @toolset_registry.tool_call
    @staticmethod
    def get_current_level_path() -> str:
        """Returns the package path of the open level; '/Temp/...' means it has never been saved.

        Returns:
            The level package path, e.g. '/Game/Maps/MyLevel' or '/Temp/Untitled_1'.
        """
        return _current_level_path()

    @toolset_registry.tool_call
    @staticmethod
    def save_current_level_as(level_path: str) -> str:
        """Saves the open level, with all its actors, as a new level asset and switches to it.

        Args:
            level_path: Asset path under /Game, e.g. '/Game/Maps/NeonCity'. Must not already exist.

        Returns:
            The saved level path.
        """
        level_path = level_path.strip().rstrip("/")
        if not level_path.startswith("/Game/"):
            raise RuntimeError(f"{level_path!r}: level_path must start with /Game/")
        current = _current_level_path()
        if level_path != current and unreal.EditorAssetLibrary.does_asset_exist(level_path):
            raise RuntimeError(
                f"{level_path!r} already exists; pick a new path or use save_current_level"
            )
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        if not unreal.EditorLoadingAndSavingUtils.save_map(world, level_path):
            raise RuntimeError(f"Saving the level as {level_path!r} failed; see the editor log")
        if _current_level_path() != level_path:
            # A level on disk is copied, not renamed (FEditorFileUtils SaveWorld duplicates it); the
            # editor stays on the original with unsaved changes, so saving dirty packages would
            # overwrite it. Load the copy, which discards them, as the copy already has them.
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(level_path)
        if _current_level_path() != level_path:
            raise RuntimeError(
                f"Saved {level_path!r} but could not open it; {current!r} was left as it is on disk"
            )
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True):
            raise RuntimeError(
                f"The level was saved as {level_path!r}, but some other assets were not; see the editor log"
            )
        return level_path

    @toolset_registry.tool_call
    @staticmethod
    def save_current_level() -> str:
        """Saves the open level and every other unsaved asset. A never-saved level needs
        save_current_level_as first.

        Returns:
            The saved level path.
        """
        path = _current_level_path()
        if path.startswith("/Temp/"):
            raise RuntimeError("This level has never been saved; call save_current_level_as first")
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True):
            raise RuntimeError("Saving failed; see the editor log")
        return path


_registration = Registration([LevelSaveTools])
_tick_handle = None


def register_when_ready() -> None:
    """init_unreal.py order across plugins is not guaranteed, so retry on tick until the registry is up."""
    global _tick_handle
    if _registration.register():
        unreal.log("Unsloth: registered unsloth_level_tools.LevelSaveTools")
        return
    if _tick_handle is not None:
        return

    def _retry(_delta_seconds: float) -> None:
        global _tick_handle
        if _registration.register():
            unreal.log("Unsloth: registered unsloth_level_tools.LevelSaveTools")
            unreal.unregister_slate_post_tick_callback(_tick_handle)
            _tick_handle = None

    _tick_handle = unreal.register_slate_post_tick_callback(_retry)
