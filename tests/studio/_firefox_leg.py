# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Firefox leg of the long-chat gate: open/close time and longest frame per surface, short vs long chat.

The same leg as _webkit_leg.py (in-page timing from just before the trigger to the first frame with the
surface visible, the longest rAF frame inside that window, median of REPS per page, failing cells
measured again on fresh pages, a crashed worker or a surface that stops running fails), run in
Playwright's Firefox. Firefox lays out and paints differently: opening a dialog or menu used to cost it
about 250 ms more at 150 turns (the modal lock's thread cover re-laid out the whole thread, fixed by a
Firefox-only rule in index.css), a cost the Chromium matrix and the WebKit leg could not see.

Surfaces: the modal ones (Radix dialogs and modal menus), the browser pane and typing (a "settle"
surface: the last content mutation after typing four characters). Limits as in the WebKit leg, set from
main readings with margin (LONG_THREAD_FIREFOX_RATIO / _FLOOR_MS / _FRAME_MS / _FRAME_GROWTH / _KNOWN).
"""

from __future__ import annotations

import os

import _area_matrix as am
import _webkit_leg as wk

SURFACES = [
    "settings",
    "composer_menu",
    "tools_menu",
    "chat_options",
    "command_palette",
    "search_chats",
    "browser_pane",
    "type",
]

# Known Firefox long-chat costs on main, pinned so the gate holds today's level (engine:mode:surface:
# metric, as in the WebKit leg). None today.
KNOWN_DEFAULT: dict = {}

FIREFOX = wk.Leg(
    name = "Firefox",
    browser = "firefox",
    env = "LONG_THREAD_FIREFOX",
    engines = {"firefox": {}},
    surfaces = SURFACES,
    known = wk.leg_known("LONG_THREAD_FIREFOX", KNOWN_DEFAULT),
    who = "Firefox",
    modes = ("default", am.GLASS),
    short = int(os.environ.get("LONG_THREAD_FIREFOX_SHORT", "1")),
    long = int(os.environ.get("LONG_THREAD_FIREFOX_LONG", "150")),
    reps = int(os.environ.get("LONG_THREAD_FIREFOX_REPS", "3")),
    ratio = float(os.environ.get("LONG_THREAD_FIREFOX_RATIO", "2.5")),
    floor_ms = float(os.environ.get("LONG_THREAD_FIREFOX_FLOOR_MS", "200")),
    frame_ms = float(os.environ.get("LONG_THREAD_FIREFOX_FRAME_MS", "250")),
    frame_growth = float(os.environ.get("LONG_THREAD_FIREFOX_FRAME_GROWTH", "2.5")),
)


def judge(result: dict, known: dict | None = None) -> list[dict]:
    return wk.judge(result, known = known, leg = FIREFOX)


def measure(dist, **kwargs) -> dict:
    return wk.measure(dist, leg = FIREFOX, **kwargs)


def run(dist, log = print):
    return wk.run(dist, log = log, leg = FIREFOX)


def main() -> int:
    """Frontend CI's Firefox job: the leg alone, on the long-thread production build."""
    return wk.main(FIREFOX)


if __name__ == "__main__":
    raise SystemExit(main())
