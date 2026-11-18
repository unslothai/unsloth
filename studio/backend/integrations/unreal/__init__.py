# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installs Unsloth's Unreal Editor toolset (level saving, which Epic's Unreal MCP lacks) into a project."""

from pathlib import Path

TOOLSET_MODULE = "unsloth_level_tools"
_SOURCE = Path(__file__).with_name(f"{TOOLSET_MODULE}.py")
# PythonScriptPlugin runs init_unreal.py from every Content/Python folder at editor startup.
_HOOK_MARKER = "# Unsloth level tools (installed by Unsloth Studio)"
# Guarded: with the Unreal MCP plugin off, toolset_registry is missing and an unguarded import would
# also skip anything the user adds to init_unreal.py after this block.
_HOOK = f"""{_HOOK_MARKER}
try:
    import {TOOLSET_MODULE}
except ImportError as _unsloth_exc:
    import unreal

    unreal.log_warning(f"Unsloth level tools not loaded (enable the Unreal MCP plugin): {{_unsloth_exc}}")
else:
    {TOOLSET_MODULE}.register_when_ready()
"""


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
    """Writes the toolset into <project>/Content/Python and hooks it into init_unreal.py; safe to
    repeat. An existing init_unreal.py keeps its contents and gets the hook appended once."""
    python_dir = project_root(project_path) / "Content" / "Python"
    python_dir.mkdir(parents = True, exist_ok = True)
    module = python_dir / f"{TOOLSET_MODULE}.py"
    module.write_text(_SOURCE.read_text(encoding = "utf-8"), encoding = "utf-8")
    hook = python_dir / "init_unreal.py"
    existing = hook.read_text(encoding = "utf-8") if hook.exists() else ""
    if _HOOK_MARKER not in existing:
        prefix = existing.rstrip("\n") + "\n\n" if existing.strip() else ""
        hook.write_text(prefix + _HOOK, encoding = "utf-8")
    return [str(module), str(hook)]
