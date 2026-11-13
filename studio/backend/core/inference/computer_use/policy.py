# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who may use computer use, and what it asks about or refuses outright. Controlling the desktop
is outside every sandbox Unsloth has, so the gates are: this machine's own user only, the owner's
switch in Settings, a go-ahead per chat, and a question before anything risky."""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

from .actions import INPUT_OPS
from .backends import Backend, FrontWindow, get_backend
from .keys import Chord, chord_text

ENABLED_SETTING_KEY = "computer_use_enabled"
COORDINATES_SETTING_KEY = "computer_use_coordinates"
COORDINATE_SETTINGS = ("auto", "pixels", "grid")

# Families trained to answer on a 0-1000 grid whatever the picture's size (Qwen3-VL cookbook,
# UI-TARS 1.0 action parser, Gemini computer use docs); the others answer in pixels.
_GRID_MODEL_RE = re.compile(
    r"qwen[-_ ]?3(?:\.\d+)?[-_ ]?vl|qwen3vl|ui[-_ ]?tars(?![-_ ]?1\.5)|gemini", re.IGNORECASE
)

_TERMINALS = (
    "terminal", "iterm", "warp", "kitty", "alacritty", "wezterm", "ghostty", "konsole", "xterm",
    "tilix", "terminator", "powershell", "pwsh", "conhost", "mintty", "tabby",
)  # fmt: skip
_SECRETS = (
    "1password", "bitwarden", "keepass", "keychain", "lastpass", "dashlane", "enpass",
    "passwords", "seahorse", "kwallet", "authenticator", "credential",
)  # fmt: skip
_SYSTEM = (
    "system settings", "system preferences", "systempreferences", "systemsettings", "regedit",
    "control panel", "taskmgr", "task manager", "gnome-control-center", "disk utility",
    "diskutility", "script editor", "scripteditor", "automator", "activity monitor",
    "activitymonitor", "installer",
)  # fmt: skip
# Operating system prompts that grant a password or a permission: only the user answers those.
_PROTECTED = (
    "securityagent", "coreservicesuiagent", "usernotificationcenter", "loginwindow",
    "authorizationhost", "polkit", "pinentry", "askpass", "gcr-prompter", "kdesu", "consent",
    "credentialuibroker",
)  # fmt: skip
# Short names that would match inside other words, so they match whole.
_EXACT = {"cmd": "a terminal", "wt": "a terminal", "hyper": "a terminal", "mmc": "a system tool"}

# Whole names, since "arc" or "edge" inside another app's name is not a browser.
_BROWSERS = frozenset(
    {
        "google chrome", "chrome", "chromium", "firefox", "safari", "microsoft edge", "msedge",
        "brave browser", "brave", "opera", "vivaldi", "arc", "zen", "orion", "librewolf",
        "waterfox", "duckduckgo", "tor browser", "com.google.chrome", "org.mozilla.firefox",
        "com.apple.safari", "com.microsoft.edgemac", "com.brave.browser",
        "company.thebrowser.browser", "com.operasoftware.opera", "com.vivaldi.vivaldi",
    }
)  # fmt: skip
_OWN_TITLE_RE = re.compile(r"(?:^|[-–—|] )Unsloth(?: [-–—|]|$)")
_OWN_APP_IDS = ("ai.unsloth.studio",)

# Lock, log out, force quit, kill the display server, switch consoles: nothing a task needs, and
# each takes the screen away from the user.
_BLOCKED = {
    "darwin": {"ctrl+cmd+q", "shift+cmd+q", "alt+shift+cmd+q", "alt+cmd+escape", "alt+shift+cmd+escape"},
    "win32": {"cmd+l", "ctrl+alt+delete"},
    "linux": {"cmd+l", "ctrl+alt+delete", "ctrl+alt+backspace", "ctrl+alt+l"},
}  # fmt: skip
_RISKY = {
    "darwin": {"cmd+q", "cmd+backspace", "shift+cmd+backspace", "alt+shift+cmd+backspace", "alt+cmd+backspace"},
    "win32": {"alt+f4", "shift+delete", "ctrl+shift+delete", "ctrl+shift+escape", "cmd+r", "cmd+x"},
    "linux": {"alt+f4", "shift+delete", "ctrl+shift+delete", "ctrl+alt+t", "alt+f2"},
}  # fmt: skip


def _platform() -> str:
    return "linux" if sys.platform.startswith("linux") else sys.platform


def settings() -> dict:
    """Off and automatic unless the owner chose otherwise; an unreadable store reads as off."""
    try:
        from storage.studio_db import get_app_settings
        stored = get_app_settings([ENABLED_SETTING_KEY, COORDINATES_SETTING_KEY])
    except Exception:  # noqa: BLE001
        stored = {}
    coordinates = stored.get(COORDINATES_SETTING_KEY)
    return {
        "enabled": stored.get(ENABLED_SETTING_KEY) is True,
        "coordinates": coordinates if coordinates in COORDINATE_SETTINGS else "auto",
    }


def update_settings(*, enabled: Optional[bool] = None, coordinates: Optional[str] = None) -> dict:
    changes: dict = {}
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false.")
        changes[ENABLED_SETTING_KEY] = enabled
    if coordinates is not None:
        if coordinates not in COORDINATE_SETTINGS:
            raise ValueError(f"coordinates must be one of: {', '.join(COORDINATE_SETTINGS)}.")
        changes[COORDINATES_SETTING_KEY] = coordinates
    if changes:
        from storage.studio_db import upsert_app_settings
        upsert_app_settings(changes, read_back = False)
    return settings()


def coordinate_mode(model: Any) -> str:
    chosen = settings()["coordinates"]
    if chosen in ("pixels", "grid"):
        return chosen
    return "grid" if isinstance(model, str) and _GRID_MODEL_RE.search(model) else "pixels"


def host_problem() -> Optional[str]:
    """Why this request may not control the computer at all, whatever the settings say."""
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


def unavailable_reason(backend: Optional[Backend] = None) -> Optional[str]:
    problem = host_problem()
    if problem:
        return problem
    return (backend or get_backend()).unavailable_reason()


def offered() -> bool:
    """Whether a chat request may be given the tool."""
    return settings()["enabled"] and unavailable_reason() is None


def _sensitive_kind(front: FrontWindow) -> Optional[str]:
    names = [front.app.casefold(), front.app_id.casefold()]
    stems = {name.rsplit(".", 1)[0] if name.endswith(".exe") else name for name in names}
    for stem in stems:
        if stem in _EXACT:
            return _EXACT[stem]
    for words, kind in (
        (_TERMINALS, "a terminal"),
        (_SECRETS, "a password manager"),
        (_SYSTEM, "a system tool"),
    ):
        if any(word in name for word in words for name in names if name):
            return kind
    return None


def _sensitive_app_name(name: str) -> Optional[str]:
    return _sensitive_kind(FrontWindow(app = name))


def is_own_window(window: FrontWindow) -> bool:
    """Unsloth's own window: the desktop app, or a browser window on its tab. A browser window
    whose title can't be read counts too, since it can't be told apart (macOS hides titles
    without the Screen Recording grant)."""
    names = (window.app.casefold(), window.app_id.casefold())
    if names[1] in _OWN_APP_IDS or names[0] == "unsloth":
        return True
    if not window.title:
        stems = {name.removesuffix(".exe") for name in names}
        return bool(stems & _BROWSERS) or names[0].endswith(" browser")
    return bool(_OWN_TITLE_RE.search(window.title))


def is_security_prompt(front: FrontWindow) -> bool:
    names = (front.app.casefold(), front.app_id.casefold())
    return any(word in name for word in _PROTECTED for name in names if name)


def blocked_chord(chord: Chord) -> bool:
    return chord_text(chord) in _BLOCKED.get(_platform(), set())


def risks(actions: list[dict], front: FrontWindow) -> list[str]:
    """Why these actions want the user's say-so first, in words for the approval card. Looking
    (screenshot, zoom, wait) never does."""
    found: list[str] = []
    inputs = [action for action in actions if action["op"] in INPUT_OPS and action["op"] != "move"]
    kind = _sensitive_kind(front)
    if inputs and kind:
        found.append(f"{front.app or 'The app in front'} is {kind}")
    risky = _RISKY.get(_platform(), set())
    for action in actions:
        if action["op"] == "key":
            for chord in action["chords"]:
                if chord_text(chord) in risky:
                    found.append(f"{chord_text(chord)} can close or delete things")
        elif action["op"] == "open_app":
            opened = _sensitive_app_name(action["name"])
            if opened:
                found.append(f"{action['name']} is {opened}")
    return list(dict.fromkeys(found))


def call_risks(arguments: Any) -> list[str]:
    """Why a call asks first, for its approval card. Never raises."""
    from .actions import parse_actions

    try:
        actions = parse_actions(arguments)
    except ValueError:
        return []
    try:
        front = get_backend().front_window()
    except Exception:  # noqa: BLE001 - display only
        front = FrontWindow()
    return risks(actions, front)


def needs_confirmation(arguments: Any, session_id: Optional[str]) -> bool:
    """Under "auto" and "off": ask for a chat's first action, then only for risky ones. A call
    that will not parse is refused when it runs, so it need not ask."""
    from state import computer_use as state

    from .actions import parse_actions

    try:
        actions = parse_actions(arguments)
    except ValueError:
        return False
    if not state.is_granted(state.session_key(session_id)):
        return True
    try:
        front = get_backend().front_window()
    except Exception:  # noqa: BLE001 - unknown front app: ask
        return True
    return bool(risks(actions, front))
