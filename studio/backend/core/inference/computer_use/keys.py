# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Key names as models write them (xdotool, pyautogui, DOM, nut-js spellings) to one canonical set."""

from __future__ import annotations

import re
from typing import Any, Optional

# "cmd" is Command on macOS, the Windows key on Windows and Super on Linux.
MODIFIERS = ("ctrl", "alt", "shift", "cmd")

_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "lctrl": "ctrl",
    "rctrl": "ctrl",
    "control_l": "ctrl",
    "control_r": "ctrl",
    "option": "alt",
    "opt": "alt",
    "alt_l": "alt",
    "alt_r": "alt",
    "lalt": "alt",
    "ralt": "alt",
    "shift_l": "shift",
    "shift_r": "shift",
    "lshift": "shift",
    "rshift": "shift",
    "command": "cmd",
    "super": "cmd",
    "super_l": "cmd",
    "super_r": "cmd",
    "win": "cmd",
    "windows": "cmd",
    "meta": "cmd",
    "os": "cmd",
    "return": "enter",
    "kp_enter": "enter",
    "esc": "escape",
    "back_space": "backspace",
    "del": "delete",
    "forwarddelete": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "page_up": "pageup",
    "prior": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "page_down": "pagedown",
    "next": "pagedown",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "uparrow": "up",
    "downarrow": "down",
    "leftarrow": "left",
    "rightarrow": "right",
    "up_arrow": "up",
    "down_arrow": "down",
    "left_arrow": "left",
    "right_arrow": "right",
    "spacebar": "space",
    "caps_lock": "capslock",
    "minus": "-",
    "hyphen": "-",
    "plus": "+",
    "equal": "=",
    "equals": "=",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "apostrophe": "'",
    "quote": "'",
    "grave": "`",
    "backtick": "`",
    "bracketleft": "[",
    "bracketright": "]",
}

NAMED_KEYS = frozenset(
    {
        "enter",
        "tab",
        "space",
        "backspace",
        "delete",
        "escape",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "insert",
        "capslock",
        *(f"f{n}" for n in range(1, 21)),
    }
)

# What a US keyboard types without Shift; a chord's key has to be one physical key.
_PLAIN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-=[]\\;',./`")
_SHIFTED = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))

MAX_CHORDS = 8
MAX_REPEAT = 50

Chord = tuple[tuple[str, ...], Optional[str]]


def canonical_key(raw: str) -> str:
    text = raw.strip()
    if text == "":
        if raw == " ":
            return "space"
        raise ValueError("a key name is empty.")
    lowered = text.lower()
    name = _ALIASES.get(lowered, lowered)
    if name in MODIFIERS or name in NAMED_KEYS or name in _PLAIN_CHARS:
        return name
    if name in _SHIFTED:
        # "ctrl+?" means the key that carries "?", pressed with Shift.
        return "shift+" + _SHIFTED[name]
    raise ValueError(
        f"'{raw}' is not a key. Use names such as enter, tab, escape, backspace, up, f5, or a "
        "single character, joined with + for a shortcut (ctrl+c)."
    )


def _split_chord(text: str) -> list[str]:
    if text in ("+", "-"):
        return [text]
    parts = text.split("+")
    # "ctrl++" is Ctrl with the plus key.
    if len(parts) >= 2 and parts[-1] == "" and parts[-2] == "":
        parts = parts[:-2] + ["+"]
    if len(parts) == 1 and "-" in text and len(text) > 1:
        dashed = text.split("-")
        head = [_ALIASES.get(p.lower(), p.lower()) for p in dashed[:-1]]
        if all(p in MODIFIERS for p in head) and dashed[-1]:
            parts = dashed
    return parts


def _chord(names: list[str]) -> Chord:
    mods: list[str] = []
    key: Optional[str] = None
    for raw in names:
        name = canonical_key(raw)
        if name.startswith("shift+") and len(name) > 6:
            if "shift" not in mods:
                mods.append("shift")
            name = name[6:]
        if name in MODIFIERS:
            if name not in mods:
                mods.append(name)
        elif key is None:
            key = name
        else:
            raise ValueError(
                f"'{'+'.join(names)}' holds two keys at once; a shortcut is modifiers plus one key."
            )
    return tuple(m for m in MODIFIERS if m in mods), key


def parse_keys(value: Any) -> list[Chord]:
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value if item is not None and str(item) != ""]
        if not names:
            raise ValueError("keys is empty.")
        if any("+" in name and name != "+" for name in names):
            chords = [_chord(_split_chord(name)) for name in names]
        else:
            chords = [_chord(names)]
    elif isinstance(value, str) and value != "":
        if value == " ":
            chords = [((), "space")]
        else:
            chords = [_chord(_split_chord(part)) for part in value.split()]
    else:
        raise ValueError('keys must name a key or shortcut, for example "enter" or "ctrl+c".')
    if not chords:
        raise ValueError("keys is empty.")
    if len(chords) > MAX_CHORDS:
        raise ValueError(f"at most {MAX_CHORDS} shortcuts in one key action.")
    return chords


def chord_text(chord: Chord) -> str:
    mods, key = chord
    return "+".join([*mods, *([key] if key else [])])


_PRINTABLE_RE = re.compile(r"[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*\Z")


def typed_text_problem(text: str) -> Optional[str]:
    if not _PRINTABLE_RE.match(text):
        return "text has control characters; use the key action for keys such as escape."
    return None
