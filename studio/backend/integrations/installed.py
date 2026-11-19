# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort check whether an editor is installed on the backend machine, for a menu hint only."""

import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

_TTL_S = 60.0
_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def _env_path(name: str, default: str = "") -> Path:
    return Path(os.environ.get(name) or default).expanduser()


def _any(patterns: list[str]) -> bool:
    return any(glob.glob(str(Path(p).expanduser())) for p in patterns)


def _unity_hub_dirs() -> list[str]:
    if sys.platform == "win32":
        roots = [str(_env_path("ProgramFiles", "C:/Program Files") / "Unity/Hub/Editor")]
        config = _env_path("APPDATA") / "UnityHub/secondaryInstallPath.json"
    elif sys.platform == "darwin":
        roots = ["/Applications/Unity/Hub/Editor"]
        config = Path(
            "~/Library/Application Support/UnityHub/secondaryInstallPath.json"
        ).expanduser()
    else:
        roots = ["~/Unity/Hub/Editor"]
        config = Path("~/.config/UnityHub/secondaryInstallPath.json").expanduser()
    try:
        extra = json.loads(config.read_text(encoding = "utf-8"))
        if isinstance(extra, str) and extra:
            roots.append(extra)
    except (OSError, ValueError):
        pass
    return roots


def _epic_launcher_has_unreal() -> bool:
    if sys.platform == "win32":
        manifest = (
            _env_path("ProgramData", "C:/ProgramData")
            / "Epic/UnrealEngineLauncher/LauncherInstalled.dat"
        )
    elif sys.platform == "darwin":
        manifest = Path(
            "~/Library/Application Support/Epic/UnrealEngineLauncher/LauncherInstalled.dat"
        ).expanduser()
    else:
        return False
    try:
        installs = json.loads(manifest.read_text(encoding = "utf-8")).get("InstallationList") or []
    except (OSError, ValueError, AttributeError):
        return False
    return any(
        str(item.get("AppName", "")).startswith("UE_")
        for item in installs
        if isinstance(item, dict)
    )


def _engine_root_exists(root: str) -> bool:
    # Entries outlive deleted engines.
    root = root.strip().strip('"')
    try:
        return bool(root) and (Path(root).expanduser() / "Engine").is_dir()
    except (
        OSError,
        ValueError,
    ):  # unreadable parent or NUL byte: keep looking
        return False


def _unreal_registered_engine() -> bool:
    """Unreal's engine registry, used by source builds: HKCU\\Software\\Epic Games\\Unreal Engine\\Builds on
    Windows, [Installations] in Epic's Install.ini elsewhere."""
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Epic Games\Unreal Engine\Builds"
            ) as key:
                index = 0
                while True:
                    try:
                        _name, value, _kind = winreg.EnumValue(key, index)
                    except OSError:  # no more values
                        return False
                    if isinstance(value, str) and _engine_root_exists(value):
                        return True
                    index += 1
        except (ImportError, OSError):
            return False
    if sys.platform == "darwin":
        ini = Path("~/Library/Application Support/Epic/UnrealEngine/Install.ini").expanduser()
    elif sys.platform.startswith("linux"):
        ini = Path("~/.config/Epic/UnrealEngine/Install.ini").expanduser()
    else:
        return False
    try:
        raw = ini.read_bytes()
    except OSError:
        return False
    # Unreal saves ASCII as UTF-8 and anything else as UTF-16 with a BOM.
    encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    in_section = False
    for line in raw.decode(encoding, errors = "replace").splitlines():
        line = line.strip()
        if line.startswith("["):
            in_section = line.lower() == "[installations]"
        elif in_section and "=" in line and _engine_root_exists(line.split("=", 1)[1]):
            return True
    return False


def _detect(builtin_id: str, configured_path: str) -> bool:
    if configured_path and Path(configured_path).expanduser().exists():
        return True
    if builtin_id == "blender":
        return bool(shutil.which("blender")) or _any(
            [
                "/Applications/Blender.app",
                str(
                    _env_path("ProgramFiles", "C:/Program Files")
                    / "Blender Foundation/Blender*/blender.exe"
                ),
                "/snap/bin/blender",
                "/var/lib/flatpak/app/org.blender.Blender",
                "~/.local/share/flatpak/app/org.blender.Blender",
                "~/.steam/steam/steamapps/common/Blender",
            ]
        )
    if builtin_id == "godot":
        return any(shutil.which(name) for name in ("godot", "godot4")) or _any(
            [
                "/Applications/Godot*.app",
                str(_env_path("ProgramFiles", "C:/Program Files") / "Godot*"),
                str(_env_path("LOCALAPPDATA") / "Programs/Godot*"),
                "/var/lib/flatpak/app/org.godotengine.Godot",
                "~/.local/share/flatpak/app/org.godotengine.Godot",
                "/snap/bin/godot*",
                "~/.steam/steam/steamapps/common/Godot Engine",
            ]
        )
    if builtin_id == "unity":
        return any(glob.glob(str(Path(root).expanduser() / "*")) for root in _unity_hub_dirs())
    if builtin_id == "unreal":
        return (
            _epic_launcher_has_unreal()
            or _unreal_registered_engine()
            or _any(
                [
                    str(_env_path("ProgramFiles", "C:/Program Files") / "Epic Games/UE_*"),
                    "/Users/Shared/Epic Games/UE_*",
                    "~/UnrealEngine*",
                    "~/Epic Games/UE_*",
                    "/opt/UnrealEngine*",
                ]
            )
        )
    if builtin_id == "figma":
        return bool(shutil.which("figma-linux")) or _any(
            [
                "/Applications/Figma.app",
                str(_env_path("LOCALAPPDATA") / "Figma/Figma.exe"),
            ]
        )
    if builtin_id == "roblox":
        return _any(
            [
                "/Applications/RobloxStudio.app",
                str(_env_path("LOCALAPPDATA") / "Roblox/mcp.bat"),
            ]
        )
    return True


def is_installed(builtin_id: str, configured_path: str = "") -> Optional[bool]:
    """True / False from a cheap file-system look; None when the look itself failed, so no hint is shown."""
    key = (builtin_id, configured_path)
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < _TTL_S:
        return cached[1]
    try:
        found = _detect(builtin_id, configured_path)
    except Exception:
        return None
    _cache[key] = (time.monotonic(), found)
    return found
