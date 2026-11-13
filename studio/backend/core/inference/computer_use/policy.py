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
ALLOWED_APPS_SETTING_KEY = "computer_use_allowed_apps"
BLOCKED_APPS_SETTING_KEY = "computer_use_blocked_apps"
COORDINATE_SETTINGS = ("auto", "pixels", "grid")
MAX_ALLOWED_APPS = 100
_MAX_APP_NAME_CHARS = 80

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
        stored = get_app_settings(
            [
                ENABLED_SETTING_KEY,
                COORDINATES_SETTING_KEY,
                ALLOWED_APPS_SETTING_KEY,
                BLOCKED_APPS_SETTING_KEY,
            ]
        )
    except Exception:  # noqa: BLE001
        stored = {}
    coordinates = stored.get(COORDINATES_SETTING_KEY)
    return {
        "enabled": stored.get(ENABLED_SETTING_KEY) is True,
        "coordinates": coordinates if coordinates in COORDINATE_SETTINGS else "auto",
        "allowed_apps": _app_names(stored.get(ALLOWED_APPS_SETTING_KEY)),
        "blocked_apps": _app_names(stored.get(BLOCKED_APPS_SETTING_KEY)),
    }


def _app_names(value: Any) -> list[str]:
    """App names as stored, without blanks or repeats that differ only in case."""
    names: dict[str, str] = {}
    for item in value if isinstance(value, list) else []:
        name = " ".join(item.split()) if isinstance(item, str) else ""
        if name and len(name) <= _MAX_APP_NAME_CHARS:
            names.setdefault(name.casefold(), name)
    return list(names.values())[:MAX_ALLOWED_APPS]


def _save_apps(**lists: list[str]) -> dict:
    from storage.studio_db import upsert_app_settings

    keys = {"allowed": ALLOWED_APPS_SETTING_KEY, "blocked": BLOCKED_APPS_SETTING_KEY}
    upsert_app_settings(
        {keys[kind]: _app_names(names) for kind, names in lists.items()}, read_back = False
    )
    return settings()


def _without(names: list[str], name: str) -> list[str]:
    folded = name.casefold()
    return [app for app in names if app.casefold() != folded]


def allow_app(name: Any) -> dict:
    """Add an app the model may act in without asking, in every chat. The apps that ask every
    time (terminals, password managers, system tools) and Unsloth itself can't be added."""
    cleaned = _app_names([name])
    if not cleaned:
        raise ValueError("Give the application's name.")
    window = FrontWindow(app = cleaned[0])
    kind = _sensitive_kind(window)
    if kind:
        raise ValueError(f"{cleaned[0]} is {kind}, so it asks every time.")
    if is_own_window(FrontWindow(app = cleaned[0], title = "-")):
        raise ValueError("Unsloth's own window is never controlled.")
    current = settings()
    if is_blocked(cleaned[0], current):
        raise ValueError(f"{cleaned[0]} is blocked. Remove it from the blocked apps first.")
    if len(current["allowed_apps"]) >= MAX_ALLOWED_APPS:
        raise ValueError("The list is full. Remove an app first.")
    return _save_apps(allowed = [*current["allowed_apps"], cleaned[0]])


def remove_allowed_app(name: Any) -> dict:
    name = name if isinstance(name, str) else ""
    return _save_apps(allowed = _without(settings()["allowed_apps"], name))


def block_app(name: Any) -> dict:
    """Add an app no model may act in, in any chat or mode. Blocking wins, so the app also
    leaves the always-allowed list."""
    cleaned = _app_names([name])
    if not cleaned:
        raise ValueError("Give the application's name.")
    current = settings()
    if len(current["blocked_apps"]) >= MAX_ALLOWED_APPS:
        raise ValueError("The list is full. Remove an app first.")
    return _save_apps(
        blocked = [*current["blocked_apps"], cleaned[0]],
        allowed = _without(current["allowed_apps"], cleaned[0]),
    )


def unblock_app(name: Any) -> dict:
    name = name if isinstance(name, str) else ""
    return _save_apps(blocked = _without(settings()["blocked_apps"], name))


def is_blocked(name: str, current: Optional[dict] = None) -> bool:
    """Whether the owner blocked this app. A blocked name matches as whole words, so "Chrome"
    covers "Google Chrome": missing an app here is worse than covering one too many."""
    folded = name.casefold()
    if not folded:
        return False
    return any(
        re.search(rf"(?<!\w){re.escape(app.casefold())}(?!\w)", folded)
        for app in (current or settings())["blocked_apps"]
    )


def blocked_window(window: FrontWindow) -> Optional[str]:
    """The blocked app this window belongs to, as the owner named it, or None."""
    current = settings()
    for name in (window.app, window.app_id):
        if is_blocked(name, current):
            return window.app or name
    return None


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
    stems = [name.removesuffix(".exe") for name in names]
    if names[1] in _OWN_APP_IDS or any(
        stem == "unsloth" or stem.startswith("unsloth ") for stem in stems
    ):
        return True
    if not window.title:
        stems = {name.removesuffix(".exe") for name in names}
        return bool(stems & _BROWSERS) or names[0].endswith(" browser")
    return bool(_OWN_TITLE_RE.search(window.title))


def app_allowed(name: str, session: Any) -> bool:
    """Whether the model may act in this app without asking: allowed in this chat, or always."""
    folded = name.casefold()
    return folded in session.apps or any(
        folded == app.casefold() for app in settings()["allowed_apps"]
    )


def asks_every_time(name: str, app_id: str = "") -> bool:
    return _sensitive_kind(FrontWindow(app = name, app_id = app_id)) is not None


def _window_at(
    action: dict,
    session: Any,
    backend: Backend,
    prefix: str = "",
) -> Optional[FrontWindow]:
    """The window a click at the action's point would land in, or None if unknown."""
    from . import screen

    frame = session.frame
    if frame is None:
        return None
    try:
        point = (action[f"{prefix}x"], action[f"{prefix}y"])
        x, y = screen.image_point(frame, session.coordinates, *point)
        return backend.window_at(*frame.to_desktop(x, y))
    except Exception:  # noqa: BLE001 - the run itself reports a bad point
        return None


def _resolved_app(name: str, backend: Backend) -> str:
    """The app a name would open, so the card and the gate speak of what would really start."""
    try:
        return backend.resolve_app(name) or name
    except Exception:  # noqa: BLE001 - the run itself reports a name that opens nothing
        return name


def _window_under_pointer(backend: Backend) -> Optional[FrontWindow]:
    """Where a scroll without a point lands: the window under the pointer, not the one in front."""
    try:
        return backend.window_at(*backend.cursor())
    except Exception:  # noqa: BLE001 - the app in front answers instead
        return None


def windows_used(
    actions: list[dict], session: Any, backend: Backend, front: FrontWindow
) -> list[FrontWindow]:
    """The windows a call's clicks and keys would land in, in order: where each click points,
    and for keys the app in front after the clicks before them. Whole windows, since an app
    may be known for what it is by its id alone."""
    current = front
    used: list[FrontWindow] = []
    for action in actions:
        op = action["op"]
        if op == "open_app":
            current = FrontWindow(app = _resolved_app(action["name"], backend))
            used.append(current)
        elif op in ("click", "drag"):
            # A click brings its window forward, so the keys after it go there.
            current = _window_at(action, session, backend) or current
            used.append(current)
            if op == "drag":
                # A drop is input to the app it lands in, which then also takes the keys.
                current = _window_at(action, session, backend, "to_") or current
                used.append(current)
        elif op == "scroll":
            at = (
                _window_at(action, session, backend)
                if "x" in action
                else _window_under_pointer(backend)
            )
            used.append(at or current)
        elif op in ("type", "key"):
            used.append(current)
    seen: dict[str, FrontWindow] = {}
    for window in used:
        if window.app:
            seen.setdefault(window.app, window)
    return list(seen.values())


def apps_used(actions: list[dict], session: Any, backend: Backend, front: FrontWindow) -> list[str]:
    return [window.app for window in windows_used(actions, session, backend, front)]


def new_apps(actions: list[dict], session: Any, backend: Backend, front: FrontWindow) -> list[str]:
    """Apps this call would act in that the user has not allowed, here or always. The ones that
    ask every time are `risks`' to report, and Unsloth's own window is refused outright."""
    current = settings()
    return [
        window.app
        for window in windows_used(actions, session, backend, front)
        if not _sensitive_kind(window)
        and not is_own_window(FrontWindow(app = window.app, app_id = window.app_id, title = "-"))
        # A blocked app is refused when the call runs, so there is nothing to ask.
        and not blocked_window(window)
        and not app_allowed(window.app, session)
    ]


def is_security_prompt(front: FrontWindow) -> bool:
    names = (front.app.casefold(), front.app_id.casefold())
    return any(word in name for word in _PROTECTED for name in names if name)


def blocked_chord(chord: Chord) -> bool:
    return chord_text(chord) in _BLOCKED.get(_platform(), set())


def risks(
    actions: list[dict],
    front: FrontWindow,
    used: tuple = (),
) -> list[str]:
    """Why these actions want the user's say-so first, in words for the approval card. Looking
    (screenshot, zoom, wait) never does. `used`: the apps the call would act in, where known,
    so a click into a terminal behind the front window counts too."""
    found: list[str] = []
    for item in used:
        window = item if isinstance(item, FrontWindow) else FrontWindow(app = item)
        kind = _sensitive_kind(window)
        if kind:
            found.append(f"{window.app} is {kind}")
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


def call_asks(arguments: Any, session_id: Optional[str]) -> dict:
    """Why a call asks first, for its approval card: what is risky about it, and the apps it
    would be the first to use in this chat. Never raises."""
    from state import computer_use as state

    from .actions import parse_actions

    try:
        actions = parse_actions(arguments)
    except ValueError:
        return {"risks": [], "apps": []}
    try:
        backend = get_backend()
        front = backend.front_window()
        key = state.session_key(session_id)
        session = state.session(key)
        found = risks(actions, front, tuple(windows_used(actions, session, backend, front)))
        if state.is_stopped(key):
            found.insert(0, "you stopped computer use in this chat")
        return {"risks": found, "apps": new_apps(actions, session, backend, front)}
    except Exception:  # noqa: BLE001 - display only
        return {"risks": [], "apps": []}


def needs_confirmation(arguments: Any, session_id: Optional[str]) -> bool:
    """Under "auto" and "off": ask for a chat's first action, for an app the chat has not used,
    and for anything risky. A call that will not parse is refused when it runs, so it need not
    ask."""
    from state import computer_use as state

    from .actions import parse_actions

    try:
        actions = parse_actions(arguments)
    except ValueError:
        return False
    key = state.session_key(session_id)
    if not state.is_granted(key):
        return True
    try:
        backend = get_backend()
        front = backend.front_window()
    except Exception:  # noqa: BLE001 - unknown front app: ask
        return True
    session = state.session(key)
    used = tuple(windows_used(actions, session, backend, front))
    return bool(risks(actions, front, used) or new_apps(actions, session, backend, front))
