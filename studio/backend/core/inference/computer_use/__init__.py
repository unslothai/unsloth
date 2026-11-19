# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Computer use: a model sees this machine's screen and drives its mouse and keyboard.

An optional add-on. Unsloth ships this stub and the image budget only; the code that reads the
screen and sends input is installed from Settings > Computer use, into the Studio home."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

COMPUTER_TOOL_NAME = "computer"
COMPUTER_TOOL_NAMES = frozenset({COMPUTER_TOOL_NAME})

# Equal to the version in studio/addons/computer_use/pyproject.toml. The add-on imports this
# backend's modules, so only the exact version this build names is ever loaded.
ADDON_VERSION = "1.0.0"
ADDON_PACKAGE = "unsloth_computer_use"
NOT_INSTALLED = "Computer use is not installed. Install it in Settings > Computer use."

_OWN_PATH = tuple(__path__)


def is_computer_tool(name: Any) -> bool:
    return isinstance(name, str) and name in COMPUTER_TOOL_NAMES


def host_problem() -> Optional[str]:
    """Why this server may not control its computer, installed or not."""
    from auth.policy import full_access_permitted
    from state.tool_policy import get_tool_policy
    from utils.account_context import via_api_key_context
    from utils.host_policy import serves_this_machine_only

    if not full_access_permitted():
        return "Only the installation owner can use computer use."
    if via_api_key_context():
        return "Computer use works only from the Unsloth app, not through an API key."
    if get_tool_policy() is False:
        return "Tools are disabled on this server, so computer use is off."
    if not serves_this_machine_only():
        return (
            "Computer use is off because this Unsloth server can be reached from other machines "
            "(a network bind, Remote Access, or a hosted notebook). It works only when Unsloth "
            "serves the computer it runs on."
        )
    return None


def addon_root() -> Path:
    from utils.paths import studio_root
    return studio_root() / "addons" / "computer_use"


def addon_dir() -> Path:
    return addon_root() / ADDON_VERSION / ADDON_PACKAGE


def attach(source: Path) -> None:
    """Load the add-on's modules as this package's own. Ours stay first, so it can't shadow them."""
    if str(source) not in __path__:
        __path__.append(str(source))


def detach() -> None:
    del __path__[len(_OWN_PATH) :]
    prefix = __name__ + "."
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue
        origin = getattr(module, "__file__", None) or ""
        if not any(origin.startswith(own) for own in _OWN_PATH):
            sys.modules.pop(name, None)
            if "." not in name[len(prefix) :]:
                globals().pop(name[len(prefix) :], None)


def installed() -> bool:
    """Call before importing any module of the add-on: this is what makes them importable."""
    if len(__path__) > len(_OWN_PATH):
        return True
    try:
        found = addon_dir()
        if not (found / "tool.py").is_file():
            return False
    except Exception:  # noqa: BLE001 - an unreadable Studio home reads as not installed
        return False
    attach(found)
    return True


def computer_tool(*args: Any, **kwargs: Any) -> dict:
    from .tool import computer_tool as build
    return build(*args, **kwargs)


def execute_computer_tool(arguments: Any, **kwargs: Any) -> str:
    if not installed():
        return f"Error: {NOT_INSTALLED}"
    from .tool import execute_computer_tool as run
    return run(arguments, **kwargs)


def status_text(arguments: Any) -> str:
    if not installed():
        return "Using the computer"
    from .tool import status_text as text
    return text(arguments)


installed()

__all__ = [
    "COMPUTER_TOOL_NAME",
    "COMPUTER_TOOL_NAMES",
    "computer_tool",
    "execute_computer_tool",
    "host_problem",
    "installed",
    "is_computer_tool",
    "status_text",
]
