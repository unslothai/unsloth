# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs inside the Unreal Editor, not in Studio: Studio copies this file into a project's Content/Python.

Epic's Unreal MCP toolsets can place actors and edit Blueprints, but a model asked for a playable game ends up
hand-writing a player pawn, input mappings and vehicle physics in Blueprint, which rarely works. Epic ships all of
that, tested, as the Blueprint game templates of the New Project dialog. This toolset (unsloth_game_tools.
GameTemplateTools) adds a template's content to the open project mid-session, sets the level's game mode, places the
player start on the ground, and reports what the player actually got in Play In Editor."""

import json
import os
import re
import shutil

import unreal

import toolset_registry
from toolset_registry.registration import Registration

_PACK_RE = re.compile(r'MountName="?([A-Za-z0-9_]+)"?\s*,\s*DetailLevels=\(\s*"?([A-Za-z]+)"?')


def _templates_dir() -> str:
    root = unreal.Paths.convert_relative_path_to_full(unreal.Paths.root_dir())
    return os.path.normpath(os.path.join(root, "Templates"))


def _template_dirs() -> dict:
    """Blueprint templates (TP_<Name>BP with a .uproject), keyed by lowercase name: 'vehicleadv', 'thirdperson'."""
    root = _templates_dir()
    out = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if (
                name.startswith("TP_")
                and name.endswith("BP")
                and os.path.isfile(os.path.join(path, name + ".uproject"))
            ):
                out[name[3:-2].lower()] = path
    return out


def _read_ini_value(path: str, key: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding = "utf-8-sig", errors = "replace") as f:
        for line in f:
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    return ""


def _shared_packs(template_dir: str) -> list:
    """(MountName, DetailLevel) of the shared content packs the template always pulls in (variants are opt-in)."""
    packs = []
    defs = os.path.join(template_dir, "Config", "TemplateDefs.ini")
    if os.path.isfile(defs):
        with open(defs, encoding = "utf-8-sig", errors = "replace") as f:
            for line in f:
                if line.startswith("SharedContentPacks="):
                    match = _PACK_RE.search(line)
                    if match:
                        packs.append((match.group(1), match.group(2)))
    return packs


def _required_plugins(template_dir: str) -> list:
    name = os.path.basename(template_dir)
    with open(os.path.join(template_dir, name + ".uproject"), encoding = "utf-8-sig") as f:
        project = json.load(f)
    return [
        plugin["Name"]
        for plugin in project.get("Plugins", [])
        if plugin.get("Enabled")
        and not plugin.get("SupportedTargetPlatforms")
        and not plugin.get("TargetAllowList")
    ]


def _copy_tree(src: str, dst: str) -> int:
    """Copies files the project does not have yet; never overwrites one."""
    copied = 0
    for base, _dirs, files in os.walk(src):
        rel = os.path.relpath(base, src)
        for fname in files:
            target = os.path.join(dst, rel, fname)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok = True)
                shutil.copy2(os.path.join(base, fname), target)
                copied += 1
    return copied


def _enable_plugins(uproject: str, names: list) -> list:
    """Adds the plugins to the .uproject (the editor reads it at startup); returns the ones it had to add."""
    with open(uproject, encoding = "utf-8-sig") as f:
        project = json.load(f)
    plugins = project.setdefault("Plugins", [])
    added = []
    for name in names:
        entry = next((p for p in plugins if p.get("Name") == name), None)
        if entry is None:
            plugins.append({"Name": name, "Enabled": True})
            added.append(name)
        elif not entry.get("Enabled"):
            entry["Enabled"] = True
            added.append(name)
    if added:
        with open(uproject, "w", encoding = "utf-8") as f:
            json.dump(project, f, indent = "\t")
    return added


def _plugins_not_running(names: list, game_mode_loads: bool) -> list:
    """The required plugins this editor session did not start with (enabling one in the .uproject only takes effect at
    the next start). Asks the plugin manager; without it, an unloadable game mode stands in for all of them."""
    try:
        running = set(unreal.PluginBlueprintLibrary.get_enabled_plugin_names())
    except Exception:
        return [] if game_mode_loads else list(names)
    return [name for name in names if name not in running]


def _keep_always_loaded(actor) -> bool:
    """World Partition streams actors in by distance; keep the player start out of that."""
    try:
        actor.set_editor_property("is_spatially_loaded", False)
        return actor.get_editor_property("is_spatially_loaded") is False
    except Exception:  # not a World Partition level, or an engine without the property
        return False


@unreal.uclass()
class GameTemplateTools(unreal.ToolsetDefinition):
    """Starts a playable game from Epic's own Blueprint templates (the New Project dialog's): a drivable car with
    physics, input and camera ('vehicleadv'), or a walking character ('thirdperson', 'firstperson', 'topdown').
    Use it instead of building a player pawn, input or vehicle physics by hand. Typical order: add_game_template,
    set_level_game_mode, build the level, place_player_start, save, StartPIE, get_play_state."""

    @toolset_registry.tool_call
    @staticmethod
    def list_game_templates() -> str:
        """Lists the game templates this engine ships.

        Returns:
            JSON: {template: {"game_mode": class path, "example_level": level, "plugins": [...]}}.
        """
        out = {}
        for key, path in _template_dirs().items():
            engine_ini = os.path.join(path, "Config", "DefaultEngine.ini")
            out[key] = {
                "game_mode": _read_ini_value(engine_ini, "GlobalDefaultGameMode"),
                "example_level": _read_ini_value(engine_ini, "EditorStartupMap").split(".")[0],
                "plugins": _required_plugins(path),
            }
        return json.dumps(out)

    @toolset_registry.tool_call
    @staticmethod
    def add_game_template(template: str) -> str:
        """Adds a template's content (Blueprints, meshes, input, its example level) to this project; files already
        in the project are kept. Usable at once, no editor restart, unless restart_required says otherwise.

        Args:
            template: A name from list_game_templates, e.g. 'vehicleadv' (car) or 'thirdperson'.

        Returns:
            JSON: game_mode (pass it to set_level_game_mode), example_level, files_copied, plugins_enabled (added to
            the .uproject by this call), plugins_not_running and restart_required (true while a plugin the template
            needs is not running in this editor and its game mode does not load: restart the editor first).
        """
        templates = _template_dirs()
        key = template.strip().lower().replace("_", "").replace(" ", "").replace("-", "")
        if key not in templates:
            raise RuntimeError(f"Unknown template {template!r}; one of {sorted(templates)}")
        src = templates[key]
        content = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir())
        copied, mounts = 0, []
        for entry in sorted(os.listdir(os.path.join(src, "Content"))):
            copied += _copy_tree(os.path.join(src, "Content", entry), os.path.join(content, entry))
            if not entry.startswith(
                "__"
            ):  # __ExternalActors__ / __ExternalObjects__ belong to the levels
                mounts.append("/Game/" + entry)
        resources = os.path.join(_templates_dir(), "TemplateResources")
        for mount, level in _shared_packs(src):
            pack = os.path.join(resources, level, mount, "Content")
            if not os.path.isdir(pack):
                raise RuntimeError(f"Template pack {level}/{mount} is missing from {resources}")
            copied += _copy_tree(pack, os.path.join(content, mount))
            mounts.append("/Game/" + mount)
        unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous(mounts, True)
        plugins = _required_plugins(src)
        enabled = _enable_plugins(unreal.Paths.get_project_file_path(), plugins)
        engine_ini = os.path.join(src, "Config", "DefaultEngine.ini")
        game_mode = _read_ini_value(engine_ini, "GlobalDefaultGameMode")
        loads = bool(game_mode) and unreal.load_class(None, game_mode) is not None
        not_running = _plugins_not_running(plugins, loads)
        return json.dumps(
            {
                "game_mode": game_mode,
                "example_level": _read_ini_value(engine_ini, "EditorStartupMap").split(".")[0],
                "files_copied": copied,
                "content_folders": mounts,
                "plugins_enabled": enabled,
                "plugins_not_running": not_running,
                "restart_required": bool(not_running) and not loads,
            }
        )

    @toolset_registry.tool_call
    @staticmethod
    def set_level_game_mode(game_mode_class: str) -> str:
        """Sets the open level's GameMode override (World Settings), which decides the pawn the player gets.

        Args:
            game_mode_class: Class path from add_game_template, e.g.
                '/Game/VehicleTemplate/Blueprints/BP_VehicleAdvGameMode.BP_VehicleAdvGameMode_C'.

        Returns:
            The game mode class path now set on the level.
        """
        cls = unreal.load_class(None, game_mode_class.strip())
        if cls is None:
            raise RuntimeError(
                f"{game_mode_class!r} is not a loadable class; call add_game_template first"
            )
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        settings = world.get_world_settings()
        settings.modify()
        settings.set_editor_property("default_game_mode", cls)
        return cls.get_path_name()

    @toolset_registry.tool_call
    @staticmethod
    def place_player_start(x: float, y: float, yaw: float) -> str:
        """Puts the player start on the ground at (x, y), facing yaw degrees (0 = +X), replacing any other.
        Traces down to the real ground (roads, floors, landscape) so the player never spawns inside geometry.

        Args:
            x: World X in cm.
            y: World Y in cm.
            yaw: Facing direction in degrees.

        Returns:
            JSON with the start's location, the actor it stands on, and how many old starts it replaced.
        """
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        old_starts = [a for a in actors.get_all_level_actors() if isinstance(a, unreal.PlayerStart)]
        hit = unreal.SystemLibrary.line_trace_single(
            world,
            unreal.Vector(x, y, 100000.0),
            unreal.Vector(x, y, -100000.0),
            unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
            False,
            old_starts,
            unreal.DrawDebugTrace.NONE,
            True,
        )
        if not hit:
            raise RuntimeError(
                f"Nothing to stand on at ({x}, {y}); add a floor or road there first"
            )
        fields = hit.to_tuple()  # (blocking, initial_overlap, time, distance, location, impact_point, ..., hit_actor, ...)
        ground = fields[5]
        ground_actor = (
            fields[9] if len(fields) > 9 and isinstance(fields[9], unreal.Actor) else None
        )
        z = ground.z + 120.0  # capsule half height (~92 cm) plus clearance
        # A new start rather than the level's existing one: the start of the default Open World level never reaches
        # the Play In Editor world (the player then spawns at the world origin), a fresh one does.
        start = actors.spawn_actor_from_class(
            unreal.PlayerStart, unreal.Vector(x, y, z), unreal.Rotator(0.0, 0.0, yaw)
        )
        if start is None:
            raise RuntimeError("Could not add a PlayerStart to the level")
        _keep_always_loaded(start)
        for old in old_starts:  # one start, so the game mode cannot pick another
            actors.destroy_actor(old)
        return json.dumps(
            {
                "location": {"x": x, "y": y, "z": round(z)},
                "ground": ground_actor.get_actor_label() if ground_actor else None,
                "replaced": len(old_starts),
            }
        )

    @toolset_registry.tool_call
    @staticmethod
    def get_play_state() -> str:
        """Reports what the player got in the running Play In Editor session: game mode, possessed pawn, its location
        and speed, and the player starts the game mode can see. Call it after StartPIE before calling a game playable.

        Returns:
            JSON: {"playing", "game_mode", "controller", "pawn_class", "pawn_location", "pawn_speed",
            "player_starts", "game_mode_picks"}.
        """
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        if world is None:
            return json.dumps({"playing": False})
        game_mode = unreal.GameplayStatics.get_game_mode(world)
        controller = unreal.GameplayStatics.get_player_controller(world, 0)
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0)
        starts = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.PlayerStart)
        out = {
            "playing": True,
            "game_mode": game_mode.get_class().get_path_name() if game_mode else None,
            "controller": controller.get_class().get_path_name() if controller else None,
            "pawn_class": pawn.get_class().get_path_name() if pawn else None,
            "player_starts": [
                {k: round(getattr(s.get_actor_location(), k)) for k in "xyz"} for s in starts
            ],
        }
        if game_mode and controller:
            chosen = game_mode.find_player_start(controller, "")
            out["game_mode_picks"] = chosen.get_name() if chosen else None
        if pawn:
            loc, vel = pawn.get_actor_location(), pawn.get_velocity()
            out["pawn_location"] = {"x": round(loc.x), "y": round(loc.y), "z": round(loc.z)}
            out["pawn_speed"] = round((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5, 1)
        return json.dumps(out)

    @toolset_registry.tool_call
    @staticmethod
    def set_vehicle_input(throttle: float, steering: float, brake: float) -> str:
        """Drives the player's vehicle in a running Play In Editor session, for a test drive: set throttle, wait a
        few seconds, then get_play_state to see it moved. Inputs stay until changed.

        Args:
            throttle: 0..1.
            steering: -1 (left) .. 1 (right).
            brake: 0..1.

        Returns:
            The vehicle movement component that received the input.
        """
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0) if world else None
        if pawn is None:
            raise RuntimeError(
                "No possessed pawn: start Play In Editor first and check get_play_state"
            )
        move = pawn.get_component_by_class(unreal.ChaosWheeledVehicleMovementComponent)
        if move is None:
            raise RuntimeError(f"{pawn.get_class().get_name()} is not a Chaos vehicle")
        move.set_throttle_input(max(0.0, min(1.0, throttle)))
        move.set_steering_input(max(-1.0, min(1.0, steering)))
        move.set_brake_input(max(0.0, min(1.0, brake)))
        move.set_handbrake_input(False)
        return move.get_path_name()


_registration = Registration([GameTemplateTools])
_tick_handle = None


def register_when_ready() -> None:
    """init_unreal.py order across plugins is not guaranteed, so retry on tick until the registry is up."""
    global _tick_handle
    if _registration.register():
        unreal.log("Unsloth: registered unsloth_game_tools.GameTemplateTools")
        return
    if _tick_handle is not None:
        return

    def _retry(_delta_seconds: float) -> None:
        global _tick_handle
        if _registration.register():
            unreal.log("Unsloth: registered unsloth_game_tools.GameTemplateTools")
            unreal.unregister_slate_post_tick_callback(_tick_handle)
            _tick_handle = None

    _tick_handle = unreal.register_slate_post_tick_callback(_retry)
