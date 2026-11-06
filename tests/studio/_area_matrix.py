# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Interaction matrix for the long-chat gate: every interactive area x appearance mode x chat length.

The real Studio app (production build of index.html, every /api call answered in-process) opens a
seeded agent-style chat of SHORT and LONG turns. For each chat length and each appearance mode (the
default, then every window material incl. Transparent, every lens mode, glass menus off, dark; read
from the app's registries in chat-background/model.ts) one fresh page runs every AREA in turn:
typing, scrolling, each menu and dialog (Settings and its tabs and selects, command palette, search
chats, model picker, chat options, composer and tools menus, rename, API monitor, shortcuts, ...),
the browser pane, the sidebar, and the message controls (action menu, copy, reasoning and tool
blocks, code copy, source cards).

One Chromium trace covers a whole page; each area is a window between two console.timeStamp marks.
Per window: elements restyled and layout objects (deterministic counts), style / layout / paint ms on
the renderer's main thread, the longest main-thread task, and the CPU time of every browser thread
(main, compositor, raster, GPU/viz), which is where glass (backdrop-filter, blurred shadows) costs.

Two checks:
  (a) length: an area's elements restyled or layout objects at LONG turns over SHORT turns above
      LENGTH_RATIO (healthy: 1x; restyling the thread on open: ~LONG/SHORT x).
  (b) appearance: an area's CPU, or elements restyled, in a mode over the same area in the default
      appearance above MAX_RATIO (settings that exist only on glass compare with plain glass).
A failure is measured again on fresh pages before the verdict, and names the area, the mode, the
counters and the threads; for (b) the worst one is re-run with backdrop-filter, box-shadow and
filter switched off to name the property responsible.

Coverage: every popup trigger on the chat page (aria-haspopup, Radix *-trigger slots) must belong
to an area or be listed in NOT_AREAS with the reason, so a new menu, dialog or picker fails the gate
until it is measured (test_area_matrix.py does the same for keyboard shortcuts and Settings tabs).
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "studio" / "frontend" / "src"
MODEL_TS = SRC / "features" / "chat-background" / "model.ts"
SHORTCUTS_TS = SRC / "features" / "settings" / "lib" / "keyboard-shortcuts.ts"
SETTINGS_STORE_TS = SRC / "features" / "settings" / "stores" / "settings-dialog-store.ts"

SHORT = int(os.environ.get("LONG_THREAD_AREAS_SHORT", "40"))
LONG = int(os.environ.get("LONG_THREAD_AREAS_LONG", "200"))
DEFAULT = {"material": "solid", "lens": "off", "menus": True, "theme": "light"}
GLASS = "liquid-3d"
# Thresholds (env overridable); see the measured table in the PR that added this.
LENGTH_RATIO = float(os.environ.get("LONG_THREAD_AREAS_LENGTH_RATIO", "2.0"))
# Counts below this many elements / layout objects at LONG turns never fail the length check.
COUNT_FLOOR = int(os.environ.get("LONG_THREAD_AREAS_COUNT_FLOOR", "3000"))
MAX_RATIO = float(os.environ.get("LONG_THREAD_AREAS_MAX_RATIO", "4.0"))
# CPU ms a mode must add on top of the ratio before it fails.
CPU_FLOOR_MS = float(os.environ.get("LONG_THREAD_AREAS_CPU_FLOOR_MS", "150"))
# Ratios are taken over at least this many ms of the base's CPU: a 20 ms click is not a fair yardstick.
CPU_BASE_FLOOR_MS = float(os.environ.get("LONG_THREAD_AREAS_CPU_BASE_FLOOR_MS", "50"))
RECHECKS = int(os.environ.get("LONG_THREAD_AREAS_RECHECKS", "2"))
# At most this many mode pages are measured again (each loads only its failing areas).
RECHECK_PAGES = int(os.environ.get("LONG_THREAD_AREAS_RECHECK_PAGES", "10"))
# Costs main already has (origin/main 3738b40e43, measured on a 36-core host and pinned to 4 CPUs),
# as `mode:area=ratio`, `mode:*=ratio`, `length:area=ratio` or `count:area=elements` (elements restyled
# at LONG turns, replacing the length ratio for that area); each ceiling is about 1.2-1.3x the highest
# reading, so these may not get worse while a new mode or area meets MAX_RATIO / LENGTH_RATIO.
# Glass: CPU raster of the backdrop-filter on every repaint (Settings scrolling 13-21x, the rest
# up to 8.7x in single readings). Length: these areas restyle the whole thread when they open (4-7x at 200 vs 40 turns,
# 24k-37k elements). Lower or drop an entry when its cause is fixed; LONG_THREAD_AREAS_KNOWN=""
# holds main to the plain limits.
KNOWN_DEFAULT = ",".join(
    [f"{m}:*=10" for m in ("frosted", "blurred", "liquid")]
    + [f"{GLASS}:*=12"]
    + [f"{m}:settings_scroll=28" for m in ("frosted", "blurred", "liquid", GLASS)]
    # Whole-thread restyles on open, held to an absolute count of elements restyled at LONG turns
    # (about 1.2x today's): non-subject :has() rules (`X:has(...) Y`), which another change is rewriting.
    + [
        f"count:{a}={n}"
        for a, n in (
            ("model_picker", 44000),
            ("chat_options", 17000),
            ("browser_pane", 45000),
            ("account_menu", 30000),
            ("organize_chats", 30000),
            ("chat_row_menu", 32000),
            ("sidebar_sections", 45000),
            ("sidebar_toggle", 30000),
            ("run_settings", 45000),
        )
    ]
)
KNOWN = os.environ.get("LONG_THREAD_AREAS_KNOWN", KNOWN_DEFAULT)

# After an open or close is visible: room for its animation, so its frames are in the window.
SETTLE_MS = 60


# --- registries ---------------------------------------------------------------------------------


def parse_registry(source: str, name: str) -> list[str]:
    """String entries of `export const NAME = [ ... ] as const` in a TS module."""
    m = re.search(rf"export const {name}\s*=\s*\[(.*?)\]\s*as const", source, re.S)
    if not m:
        raise ValueError(f"{name} registry not found")
    return re.findall(r"""["']([^"']+)["']""", m.group(1))


def shortcut_ids(source: str) -> list[str]:
    m = re.search(r"export type ShortcutId\s*=(.*?);", source, re.S)
    if not m:
        raise ValueError("ShortcutId union not found")
    return re.findall(r'"([^"]+)"', m.group(1))


def modes(materials: list[str], lens_modes: list[str]) -> list[dict]:
    """Every non-default appearance mode, each with the mode it is compared with."""
    out = [{"name": "default", "base": None, "set": dict(DEFAULT)}]
    for m in materials:
        if m != DEFAULT["material"]:
            out.append({"name": m, "base": "default", "set": {**DEFAULT, "material": m}})
    on_glass = {**DEFAULT, "material": GLASS}
    for lens in lens_modes:
        if lens != DEFAULT["lens"]:
            out.append(
                {"name": f"{GLASS}+lens-{lens}", "base": GLASS, "set": {**on_glass, "lens": lens}}
            )
    out.append({"name": f"{GLASS}+menus-off", "base": GLASS, "set": {**on_glass, "menus": False}})
    out.append({"name": "solid+dark", "base": "default", "set": {**DEFAULT, "theme": "dark"}})
    out.append({"name": f"{GLASS}+dark", "base": GLASS, "set": {**on_glass, "theme": "dark"}})
    return out


def app_modes() -> list[dict]:
    source = MODEL_TS.read_text(encoding = "utf-8")
    return modes(parse_registry(source, "MATERIALS"), parse_registry(source, "LENS_MODES"))


def known_limits(text: str) -> dict[tuple[str, str], float]:
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition("=")
        mode, _, area = key.partition(":")
        out[(mode.strip(), area.strip())] = float(value)
    return out


# --- areas ---------------------------------------------------------------------------------------
# name: {open, ready, close, gone, setup, teardown, covers}
#   actions: ("key", chord) | ("click", css) | ("rclick", css) | ("hover", css) | ("js", expr) | ("type", text)
#            | ("scroll", where) | ("tabs", None) | None ; chords use Mod for Meta on macOS, Control elsewhere
#   ready / gone: ("overlay",) a new visible dialog/menu/listbox | ("overlay_gone",) | ("sel", css) | ("gone", css)
#                 | ("settle",) wait only
#   setup / teardown run outside the measured window; covers: trigger keys (see trigger_key) it measures.
# "last|css": inside the last assistant message; "js|expr": an expression returning the element.
LAST = "last|"
CODE_COPY = "js|(() => { const a = [...document.querySelectorAll('[data-role=\"assistant\"]')].pop(); const pre = a && a.querySelector('pre'); let n = pre; while (n && n !== a && !n.querySelector('button')) n = n.parentElement; return n && n !== a ? n.querySelector('button') : null; })()"
AREAS: dict[str, dict] = {
    "type": {
        "open": ("type", "abcd"),
        "ready": ("settle",),
        "close": ("key", "Control+a"),
        "gone": ("settle",),
        "setup": ("click", ".aui-composer-input"),
        "teardown": ("key", "Delete"),
        "covers": [],
    },
    "thread_scroll": {
        "open": ("scroll", "thread"),
        "ready": ("settle",),
        "close": None,
        "gone": ("settle",),
        "covers": [],
    },
    "settings": {
        "open": ("key", "Mod+Comma"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["Settings", "Open settings"],
    },
    # These three share one open Settings dialog ("keep": no reset before them); tabs ends on the first tab.
    "settings_tabs": {
        "setup": ("key", "Mod+Comma"),
        "open": ("tabs", None),
        "ready": ("settle",),
        "close": None,
        "gone": ("settle",),
        "covers": [],
    },
    "settings_select": {
        "keep": True,
        "open": ("click", '[role="dialog"] [role="combobox"]'),
        "ready": ("sel", '[role="listbox"]'),
        "close": ("key", "Escape"),
        "gone": ("gone", '[role="listbox"]'),
        "covers": [],
    },
    "settings_scroll": {
        "keep": True,
        "open": ("scroll", "dialog"),
        "ready": ("settle",),
        "close": None,
        "gone": ("settle",),
        "teardown": ("key", "Escape"),
        "covers": [],
    },
    "command_palette": {
        "open": ("key", "Mod+KeyP"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": [],
    },
    "search_chats": {
        "open": ("key", "Mod+KeyK"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["Search"],
    },
    "model_picker": {
        "open": ("click", "button.unsloth-model-selector-trigger"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["popover-trigger:Select model"],
    },
    "composer_menu": {
        "open": ("key", "Mod+Shift+Period"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": [],
    },
    "tools_menu": {
        "open": ("click", 'button[aria-label="Tools and attachments"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["dropdown-menu-trigger:Tools and attachments"],
    },
    "permission_menu": {
        "open": ("click", 'button[aria-label="Permission level for tool calls"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["dropdown-menu-trigger:Permission level for tool calls"],
    },
    "chat_options": {
        "open": ("click", 'button[aria-label="Chat options"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["dropdown-menu-trigger:Chat options"],
    },
    "browser_pane": {
        "open": ("click", 'button[aria-label="New tab"]'),
        "ready": ("sel", '[aria-label="Browser tabs"]'),
        "close": ("click", 'button[aria-label="Close browser"]'),
        "gone": ("gone", '[aria-label="Browser tabs"]'),
        "covers": ["dropdown-menu-trigger:New tab"],
    },
    "account_menu": {
        "open": ("click", 'button[aria-label="Unsloth account menu"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["dropdown-menu-trigger:Unsloth account menu"],
    },
    "organize_chats": {
        "open": ("click", 'button[aria-label="Organize chats"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": [
            "dropdown-menu-trigger:Organize chats",
            "dropdown-menu-trigger:Organize projects",
        ],
    },
    "chat_row_menu": {
        "open": ("rclick", '[data-slot="context-menu-trigger"]'),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["context-menu-trigger:*"],
    },
    "sidebar_sections": {
        "open": ("click", 'button[data-slot="collapsible-trigger"]'),
        "ready": ("settle",),
        "close": ("click", 'button[data-slot="collapsible-trigger"]'),
        "gone": ("settle",),
        "covers": ["collapsible-trigger:*"],
    },
    "sidebar_toggle": {
        "open": ("key", "Mod+KeyB"),
        # the sidebar's own state flips, then flips back
        "ready": ("changed", ".sidebar-peer[data-state]", "data-state"),
        "close": ("key", "Mod+KeyB"),
        "gone": ("restored", ".sidebar-peer[data-state]", "data-state"),
        "covers": [],
    },
    "run_settings": {
        "open": ("click", 'button[aria-label="Open run settings"]'),
        "ready": ("settle",),
        "close": ("click", 'button[aria-label="Close run settings"]'),
        "gone": ("settle",),
        "covers": [],
    },
    "api_monitor": {
        "open": ("key", {"mac": "Control+Shift+KeyU", "other": "Control+Alt+Shift+KeyM"}),
        "ready": ("sel", '[data-testid="api-monitor-panel"]'),
        "close": ("key", {"mac": "Control+Shift+KeyU", "other": "Control+Alt+Shift+KeyM"}),
        "gone": ("gone", '[data-testid="api-monitor-panel"]'),
        "covers": [],
    },
    "rename_chat": {
        "open": ("key", "Mod+Alt+KeyR"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": [],
    },
    "shortcuts_dialog": {
        "open": ("key", "Mod+Slash"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": [],
    },
    "message_menu": {
        "open": ("click", "lastbtn|More"),
        "ready": ("overlay",),
        "close": ("key", "Escape"),
        "gone": ("overlay_gone",),
        "covers": ["dropdown-menu-trigger:More"],
    },
    "message_copy": {
        "open": ("click", "lastbtn|Copy"),
        "ready": ("settle",),
        "close": None,
        "gone": ("settle",),
        "covers": [],
    },
    # A real (Playwright) click: native focus moves into the message, so the action bar's focus reveal
    # runs inside the window; the area fails if no focusin reached the message.
    "message_click": {
        "open": ("pclick", f"{LAST}p"),
        "ready": ("focusin", '[data-role="assistant"]'),
        "close": ("pclick", ".aui-composer-input"),
        "gone": ("settle",),
        "covers": [],
    },
    "reasoning_toggle": {
        "open": ("click", f'{LAST}[data-slot="reasoning-trigger"]'),
        "ready": ("settle",),
        "close": ("click", f'{LAST}[data-slot="reasoning-trigger"]'),
        "gone": ("settle",),
        "covers": ["reasoning-trigger:*"],
    },
    "tool_toggle": {
        "open": ("click", f'{LAST}[data-slot="tool-fallback-trigger"]'),
        "ready": ("settle",),
        "close": ("click", f'{LAST}[data-slot="tool-fallback-trigger"]'),
        "gone": ("settle",),
        "covers": ["tool-fallback-trigger:*"],
    },
    "code_copy": {
        "open": ("click", CODE_COPY),
        "ready": ("settle",),
        "close": None,
        "gone": ("settle",),
        "covers": [],
    },
    "source_card": {
        "open": ("hover", f'{LAST}[data-slot="hover-card-trigger"]'),
        "ready": ("sel", '[data-slot="hover-card-content"]'),
        "close": ("hover", ".aui-composer-input"),
        "gone": ("gone", '[data-slot="hover-card-content"]'),
        "covers": ["hover-card-trigger:*"],
    },
}

# Popup triggers on the chat page that are not measured, with why. A trigger that is neither here
# nor in an area's `covers` fails the coverage check.
NOT_AREAS = {
    "popover-trigger:Request failed": "the gate's API stub has no model runtime; this chip only shows a failed status call",
    "dropdown-menu-trigger:Open preset list": "inside the run settings panel, measured as run_settings",
}

# Keyboard shortcuts (features/settings/lib/keyboard-shortcuts.ts): the area that measures what it
# opens, or why it opens nothing to measure. A new shortcut fails test_area_matrix.py until listed.
SHORTCUT_AREAS = {
    "openSettings": "settings",
    "openCommandPalette": "command_palette",
    "searchChats": "search_chats",
    "openComposerMenu": "composer_menu",
    "openModelPicker": "model_picker",
    "toggleSidebar": "sidebar_toggle",
    "toggleApiMonitor": "api_monitor",
    "renameChat": "rename_chat",
    "openKeyboardShortcuts": "shortcuts_dialog",
    "newBrowserTab": "browser_pane",
}
SHORTCUT_NOT_AREAS = {
    "navigation or a route change, not a surface on the chat page": [
        "newChat",
        "newTemporaryChat",
        "newStandaloneChat",
        "nextRecentlyViewedChat",
        "nextChat",
        "nextChatNeedingAttention",
        "previousRecentlyViewedChat",
        "previousChat",
        "goToRecentChat1",
        "goToRecentChat2",
        "goToRecentChat3",
        "goToRecentChat4",
        "goToRecentChat5",
        "goToRecentChat6",
        "switchToChat",
        "switchToProjects",
        "switchToHub",
        "switchToTrain",
        "switchToRecipes",
        "switchToImages",
        "switchToVideo",
        "switchToAudio",
        "switchToExport",
        "logOut",
    ],
    "a state change on the chat row or thread with no surface of its own": [
        "archiveChat",
        "markChatUnread",
        "togglePinChat",
        "selectAllChats",
        "clearChatSelection",
        "deleteSelectedChats",
        "clearAllUnreads",
        "copyChatAsMarkdown",
        "copySessionId",
        "forkChat",
    ],
    "needs a running model, a tool call or a device the gate does not have": [
        "approveToolRequest",
        "declineToolRequest",
        "sendMessage",
        "queueMessage",
        "steerMessage",
        "stopGeneration",
        "startDictation",
        "cycleReasoningEffort",
        "decreaseReasoningEffort",
        "increaseReasoningEffort",
        "attachFiles",
    ],
    "opens a surface measured elsewhere or one the stub cannot fill": [
        "findInPage",
        "toggleBrowserFullView",
        "openMcpServers",
        "openProjectPicker",
    ],
}

# --- seeded chat and API stub --------------------------------------------------------------------

PARA = (
    "The scheduler drains each buffer before the next commit, so the observer sees a stable layout "
    "and the shard boundary stays aligned with the token lane. Gradient checkpoints are written per epoch."
)
CODE = "\n".join(f"def step_{k}(x):\n    return sum(v * {k} for v in range(x))\n" for k in range(6))
T0 = 1_767_225_600_000


def turn(tid: str, t: int, parent: str | None) -> list[dict]:
    """One agent turn: reasoning, a web search with six sources, a python call, prose, code, a table, math."""
    uid, aid = f"{tid}-u{t}", f"{tid}-a{t}"
    q, c = {"query": f"scheduler buffer reference {t}"}, {"code": f"print(sum(range({t + 10})))"}
    parts = [
        {"type": "reasoning", "text": f"Turn {t}. " + "\n\n".join([PARA] * 6)},
        {
            "type": "tool-call",
            "toolCallId": f"ws_{t}",
            "toolName": "web_search",
            "argsText": json.dumps(q),
            "args": q,
            "state": "result",
            "result": " ".join([PARA] * 3),
        },
        *[
            {
                "type": "source",
                "sourceType": "url",
                "id": f"src_{t}_{s}",
                "url": f"https://example{s}.org/t{t}",
                "title": f"Source {s} for turn {t}",
            }
            for s in range(6)
        ],
        {
            "type": "tool-call",
            "toolCallId": f"py_{t}",
            "toolName": "python",
            "argsText": json.dumps(c),
            "args": c,
            "state": "result",
            "result": " ".join([PARA] * 2),
        },
        {
            "type": "text",
            "text": f"Answer {t}. {PARA}\n\n```python\n# turn {t}\n{CODE}```\n\n| step | loss |\n|---|---|\n"
            f"| {t} | 0.{t % 10}4 |\n\nThe update is $\\theta_{{t+1}} = \\theta_t - \\eta g_{{{t}}}$.\n\n- {PARA[:90]}\n- {PARA[:60]}",
        },
    ]
    return [
        {
            "id": uid,
            "threadId": tid,
            "parentId": parent,
            "role": "user",
            "content": [{"type": "text", "text": f"Question {t}: {PARA}"}],
            "createdAt": T0 + 2 * t,
        },
        {
            "id": aid,
            "threadId": tid,
            "parentId": uid,
            "role": "assistant",
            "content": parts,
            "createdAt": T0 + 2 * t + 1,
        },
    ]


def thread(turns: int) -> tuple[dict, list[dict]]:
    tid = f"gate-{turns}"
    msgs, parent = [], None
    for t in range(turns):
        pair = turn(tid, t, parent)
        msgs += pair
        parent = pair[1]["id"]
    rec = {
        "id": tid,
        "title": f"Gate {turns} turns",
        "modelType": "base",
        "modelId": "unsloth/Qwen3-4B",
        "archived": False,
        "createdAt": T0,
        "updatedAt": T0 + 10_000,
    }
    return rec, msgs


class Api:
    """In-process answers for /api: the seeded thread, empty lists, 404 for the rest."""

    def __init__(self, turns: int):
        self.rec, self.msgs = thread(turns)
        tid = self.rec["id"]
        self.bodies = {
            "/api/auth/status": {
                "initialized": True,
                "requires_password_change": False,
                "login_mode": "single",
                "full_access": True,
            },
            "/api/chat/threads": {"threads": [self.rec], "has_more": False},
            f"/api/chat/threads/{tid}": self.rec,
            f"/api/chat/threads/{tid}/messages": {"messages": self.msgs},
            f"/api/chat/threads/{tid}/forks": {"forks": []},
            "/api/chat/projects": {"projects": []},
            "/api/chat/pins": {"pins": []},
            "/api/models/list": {"models": [], "default_models": []},
            "/api/models/loras": {"loras": [], "outputs_dir": "/out"},
            "/api/hub/local": {"models": []},
            "/api/hub/hidden-models": {"hidden": []},
            "/api/settings/pinned-models": {"pinned": []},
            "/api/health": {"status": "ok"},
        }

    def handle(self, route) -> None:
        req = route.request
        path = urlparse(req.url).path
        if path == "/api/chat/messages:batch":
            body = {"messagesByThreadId": {self.rec["id"]: self.msgs}}
        elif req.method == "GET" and path in self.bodies:
            body = self.bodies[path]
        else:
            return route.fulfill(status = 404, content_type = "application/json", body = "{}")
        route.fulfill(status = 200, content_type = "application/json", body = json.dumps(body))


class _SpaHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: D401
        pass

    def send_head(self):
        path = urlparse(self.path).path
        if path == "/" or not (Path(self.directory) / path.lstrip("/")).is_file():
            self.path = "/index.html"
        return super().send_head()


def serve_app(dist: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_SpaHandler, directory = str(dist))
    )
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# --- page side ------------------------------------------------------------------------------------

# Open surfaces; not the bare popper wrapper, which a tooltip (focus returning to a trigger) also uses.
OVER = '[role=dialog],[role=alertdialog],[role=menu],[role=listbox],[cmdk-root],[data-slot$="-content"][data-state="open"]'
PROBE = (
    """() => {
  const OVER = '%s';
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 1 && r.height > 1; };
  window.__gateOverlays = () => [...document.querySelectorAll(OVER)].filter(vis).length;
  window.__gateVisible = (sel) => [...document.querySelectorAll(sel)].some(vis);
  window.__gateFrames = (n) => new Promise((r) => { let k = 0; const f = () => (++k >= n ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); });
  window.__gateFocusins = [];
  document.addEventListener('focusin', (e) => window.__gateFocusins.push(e.target), true);
  window.__gateAttr = (sel, attr) => { const e = document.querySelector(sel); return e ? e.getAttribute(attr) : null; };
}"""
    % OVER
)
APPLIED = """() => {
  const root = document.documentElement;
  const cs = (sel) => { const e = document.querySelector(sel); if (!e) return null; const s = getComputedStyle(e);
    return [s.backdropFilter || s.getPropertyValue('-webkit-backdrop-filter') || 'none', s.boxShadow].join(' | '); };
  return {glass: root.getAttribute('data-glass'), clear: root.hasAttribute('data-glass-clear'),
    menus: root.hasAttribute('data-glass-menus'), dark: root.classList.contains('dark'),
    lens: document.querySelectorAll('.glass-lens, .glass-lens-copy').length,
    composer: cs('.unsloth-composer-surface'), sidebar: cs('[data-slot="sidebar-inner"]')};
}"""
TRIGGERS = """() => {
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const out = {};
  for (const e of document.querySelectorAll('[aria-haspopup]:not([aria-haspopup="false"]), [data-slot$="-trigger"]')) {
    const slot = e.getAttribute('data-slot') || ('haspopup-' + e.getAttribute('aria-haspopup'));
    if (slot === 'tooltip-trigger' || !vis(e)) continue;
    const label = (e.getAttribute('aria-label') || (e.textContent || '').trim()).slice(0, 60);
    const key = slot + ':' + label;
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}"""
SCROLL = """async ([where, frames, step]) => {
  let el = null;
  if (where === 'thread') el = document.querySelector('.aui-thread-viewport');
  else { const dlg = document.querySelector('[role="dialog"]'); let room = 0;
    for (const e of dlg ? dlg.querySelectorAll('*') : []) { const r = e.scrollHeight - e.clientHeight;
      if (r > room && /(auto|scroll)/.test(getComputedStyle(e).overflowY)) { el = e; room = r; } } }
  if (!el) return false;
  const sign = where === 'thread' ? -1 : 1;
  if (where !== 'thread') el.scrollTop = 0;
  for (let i = 0; i < frames; i++) { el.scrollTop += sign * step; await new Promise((r) => requestAnimationFrame(r)); }
  if (where === 'thread') el.scrollTop = el.scrollHeight;
  return true;
}"""
# Every tab the Settings dialog renders, in order, then back to the first: a new tab is measured
# without a list to update. Panels are lazy and behind useDeferredValue + Suspense, so after each
# click this waits until the resolved panel itself has committed (a hidden data-settings-panel-ready
# marker rendered inside the panel's Suspense boundary in settings-dialog.tsx, so a panel still
# showing its Loading fallback has none) and did not fail; a panel that never gets there within
# TAB_READY_MS fails the area.
TABS = """async (timeoutMs) => {
  const dlg = document.querySelector('[role="dialog"]'); if (!dlg) return {count: 0, missing: [], failed: []};
  const tabs = [...dlg.querySelectorAll('button[data-testid^="settings-tab-"]')];
  const missing = [], failed = [];
  for (const tab of tabs.concat(tabs.slice(0, 1))) {
    const id = tab.getAttribute('data-testid').slice('settings-tab-'.length);
    tab.click();
    const deadline = performance.now() + timeoutMs;
    let ready = false;
    while (performance.now() < deadline) {
      await window.__gateFrames(1);
      if (dlg.querySelector('[data-settings-panel-failed]')) { failed.push(id); break; }
      if (dlg.querySelector(`[data-settings-panel-ready="${id}"]`)) { ready = true; break; }
    }
    if (!ready && !failed.includes(id)) missing.push(id);
    await window.__gateFrames(2);
  }
  return {count: tabs.length, missing, failed};
}"""
FIND = """(spec) => {
  document.querySelectorAll('[data-gate-target]').forEach((e) => e.removeAttribute('data-gate-target'));
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  let el = null;
  if (spec.startsWith('js|')) el = (0, eval)(spec.slice(3));
  else if (spec.startsWith('lastbtn|')) { const a = [...document.querySelectorAll('[data-role="assistant"]')].pop();
    el = a ? [...a.querySelectorAll('button')].filter((b) => vis(b) && ((b.getAttribute('aria-label') || b.textContent || '').trim() === spec.slice(8))).pop() : null; }
  else if (spec.startsWith('last|')) { const a = [...document.querySelectorAll('[data-role="assistant"]')].pop();
    el = a ? [...a.querySelectorAll(spec.slice(5))].filter(vis).pop() : null; }
  else el = [...document.querySelectorAll(spec)].filter(vis).pop();
  if (!el) return false;
  el.setAttribute('data-gate-target', ''); el.scrollIntoView({block: 'nearest'}); return true;
}"""
CLICK = """(right) => {
  const el = document.querySelector('[data-gate-target]'); const r = el.getBoundingClientRect();
  const at = {bubbles: true, cancelable: true, composed: true, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2,
              button: right ? 2 : 0, buttons: right ? 2 : 1, pointerType: 'mouse', isPrimary: true, view: window};
  el.dispatchEvent(new PointerEvent('pointerdown', at)); el.dispatchEvent(new MouseEvent('mousedown', at));
  el.dispatchEvent(new PointerEvent('pointerup', {...at, buttons: 0})); el.dispatchEvent(new MouseEvent('mouseup', {...at, buttons: 0}));
  if (right) el.dispatchEvent(new MouseEvent('contextmenu', at)); else el.dispatchEvent(new MouseEvent('click', {...at, buttons: 0}));
}"""
OVERRIDE = """(css) => {
  let s = document.getElementById('gate-override');
  if (!css) { if (s) s.remove(); return; }
  if (!s) { s = document.createElement('style'); s.id = 'gate-override'; document.head.append(s); }
  s.textContent = css;
}"""
OFF = {
    "backdrop-filter": "*, *::before, *::after { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }",
    "box-shadow": "*, *::before, *::after { box-shadow: none !important; }",
    "filter": "*, *::before, *::after { filter: none !important; }",
}
HINTS = {
    "backdrop-filter": "backdrop-filter re-blurs what is behind the surface whenever anything under or inside it repaints; "
    "shrink the blurred area, keep it off surfaces over scrolling or typed content, or give the repainting children "
    "their own layer (will-change: opacity)",
    "box-shadow": "a large blurred box-shadow is re-rasterised when its surface repaints; smaller blur, a border, or "
    "isolate the repainting children in their own layer",
    "filter": "a CSS filter re-renders its whole subtree on every repaint inside it",
}
TRACE_CATS = ["toplevel", "devtools.timeline", "__metadata"]


# A Settings panel that has not committed this long after its tab was clicked fails the area.
TAB_READY_MS = int(os.environ.get("LONG_THREAD_AREAS_TAB_READY_MS", "8000"))


def settings_tabs() -> list[str]:
    """Every Settings tab the app defines; the owner session the gate uses sees all of them."""
    return parse_registry(SETTINGS_STORE_TS.read_text(encoding = "utf-8"), "SETTINGS_TABS")


def auth_token(claims: dict | None = None) -> str:
    """An unsigned token the app reads as the installation owner (account-session.ts sessionAccount):
    a string `sub` and role owner, so owner-only Settings tabs render. The stub API never checks it."""
    import base64

    body = {"exp": 9_999_999_999, "sub": "unsloth", "role": "owner", **(claims or {})}
    payload = base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    return f"gate.{payload}.gate"


def material_storage(mode: dict) -> str:
    state = {
        "material": mode["material"],
        "style": mode["material"]
        if mode["material"] not in ("solid", "transparent", "macos")
        else GLASS,
        "menus": mode["menus"],
        "lens": mode["lens"],
        "scope": "all",
    }
    return json.dumps({"state": state, "version": 0})


class Page:
    """One app page at one chat length and one appearance mode."""

    def __init__(self, browser, base: str, turns: int, mode: dict, settings_tabs: list[str]):
        self.browser, self.turns, self.mode, self.tabs = browser, turns, mode, settings_tabs
        self.api = Api(turns)
        self.ctx = browser.new_context(viewport = {"width": 1440, "height": 900})
        self.ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin = base)
        init = {
            "unsloth_auth_token": auth_token(),
            "unsloth_auth_refresh_token": "r",
            "unsloth_window_material": material_storage(mode["set"]),
            "theme": mode["set"]["theme"],
        }
        self.ctx.add_init_script(
            "(() => { const v = %s; for (const k in v) localStorage.setItem(k, v[k]); })()"
            % json.dumps(init)
        )
        self.ctx.route(lambda url: urlparse(url).path.startswith("/api/"), self.api.handle)
        self.ctx.route(
            lambda url: not url.startswith(base), lambda route: route.fulfill(status = 204, body = "")
        )
        self.page = page = self.ctx.new_page()
        self.errors: list[str] = []
        self.last_error = ""
        self.attr_before = None
        self.focusins_before = 0
        page.on("pageerror", lambda e: self.errors.append(str(e)[:200]))
        page.goto(f"{base}/chat?thread={self.api.rec['id']}", wait_until = "load", timeout = 120_000)
        page.wait_for_function(
            "(n) => document.querySelectorAll('[data-role]').length >= n",
            arg = 2 * turns,
            timeout = 120_000,
        )
        page.evaluate(PROBE)
        page.wait_for_timeout(600)
        self.mac = page.evaluate(
            "() => /mac|iphone|ipad|ipod/i.test(`${navigator.platform || ''} ${navigator.userAgent || ''}`)"
        )
        self.applied = page.evaluate(APPLIED)
        self.hovered = False
        page.mouse.move(1435, 450)
        self.cdp = browser.new_browser_cdp_session()

    def close(self) -> None:
        self.ctx.close()

    def chord(self, arg) -> str:
        if isinstance(arg, dict):
            return arg["mac" if self.mac else "other"]
        return arg.replace("Mod+", "Meta+" if self.mac else "Control+")

    def act(self, action) -> bool:
        if action is None:
            return True
        kind, arg = action
        page = self.page
        if kind == "key":
            page.keyboard.press(self.chord(arg))
            return True
        if kind == "type":
            for ch in arg:
                page.keyboard.press(ch)
                page.wait_for_timeout(40)
            return True
        if kind == "scroll":
            return bool(page.evaluate(SCROLL, [arg, 20, 60 if arg == "thread" else 40]))
        if kind == "tabs":
            got = page.evaluate(TABS, TAB_READY_MS)
            expected = len(settings_tabs())
            problems = []
            if got["count"] < expected:
                problems.append(
                    f"only {got['count']} of {expected} Settings tabs rendered (owner session lost?)"
                )
            if got["missing"]:
                problems.append(
                    f"panel(s) never committed within {TAB_READY_MS} ms: {', '.join(got['missing'])}"
                )
            if got["failed"]:
                problems.append(
                    f"panel(s) rendered their error boundary: {', '.join(got['failed'])}"
                )
            self.last_error = "; ".join(problems)
            return not problems
        if not page.evaluate(FIND, arg):
            return False
        if kind == "hover":
            page.locator("[data-gate-target]").first.hover(timeout = 3_000, force = True)
            self.hovered = True
            return True
        if kind == "pclick":
            # A trusted click (mouse move, press, native focus), for areas whose cost is the focus change.
            page.locator("[data-gate-target]").first.click(timeout = 3_000, force = True)
            self.hovered = True
            return True
        # Clicks are dispatched in the page: the pointer stays parked, so no tooltip or :hover state
        # lands in a window by chance and the counts stay deterministic.
        page.evaluate(CLICK, kind == "rclick")
        return True

    def wait(self, pred, base_overlays: int) -> bool:
        kind = pred[0]
        js = {
            "overlay": f"() => window.__gateOverlays() > {base_overlays}",
            "overlay_gone": f"() => window.__gateOverlays() <= {max(0, base_overlays - 1)} || window.__gateOverlays() === 0",
        }.get(kind)
        try:
            if js:
                self.page.wait_for_function(js, timeout = 3_000)
            elif kind == "sel":
                # A hover card waits its open delay; a busy runner adds to it.
                self.page.wait_for_function(
                    "(s) => window.__gateVisible(s)", arg = pred[1], timeout = 6_000
                )
            elif kind == "gone":
                self.page.wait_for_function(
                    "(s) => !window.__gateVisible(s)", arg = pred[1], timeout = 3_000
                )
            elif kind == "changed":
                self.page.wait_for_function(
                    "([s, a, v]) => window.__gateAttr(s, a) !== null && window.__gateAttr(s, a) !== v",
                    arg = [pred[1], pred[2], self.attr_before],
                    timeout = 3_000,
                )
            elif kind == "restored":
                self.page.wait_for_function(
                    "([s, a, v]) => window.__gateAttr(s, a) === v",
                    arg = [pred[1], pred[2], self.attr_before],
                    timeout = 3_000,
                )
            elif kind == "focusin":
                self.page.wait_for_function(
                    "([s, n]) => window.__gateFocusins.slice(n).some((t) => t instanceof Element && t.closest(s))",
                    arg = [pred[1], self.focusins_before],
                    timeout = 3_000,
                )
            ok = True
        except Exception:  # noqa: BLE001 - reported as a NOT_RUN area
            ok = False
        self.page.evaluate(
            "(ms) => window.__gateFrames(2).then(() => new Promise((r) => setTimeout(r, ms)))",
            SETTLE_MS,
        )
        return ok

    def reset(self) -> None:
        page = self.page
        for _ in range(3):
            if page.evaluate("() => window.__gateOverlays()") == 0:
                break
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)
        if self.hovered:
            page.mouse.move(1435, 450)
            self.hovered = False

    def sweep(self, areas: list[str]) -> dict[str, dict]:
        """Run the areas inside one trace; a failure part way still ends the browser-wide trace, so
        the next page can start its own."""
        try:
            return self._sweep(areas)
        except BaseException:
            if getattr(self, "_tracing", False):
                try:
                    self.cdp.send("Tracing.end")
                except Exception:  # noqa: BLE001 - the original error is the one to report
                    pass
                self._tracing = False
            raise

    def _sweep(self, areas: list[str]) -> dict[str, dict]:
        """Run the areas inside one trace; returns per-area status and counters."""
        page, cdp = self.page, self.cdp
        events: list = []
        done: list = []
        cdp.on("Tracing.dataCollected", lambda d: events.extend(d["value"]))
        cdp.on("Tracing.tracingComplete", lambda d: done.append(bool(d.get("dataLossOccurred"))))
        t_sweep = time.time()
        cdp.send(
            "Tracing.start",
            {
                "traceConfig": {
                    "includedCategories": TRACE_CATS,
                    # Room for a whole page (a 200-turn sweep is ~200k events); a full buffer drops
                    # events and is reported as data loss, which voids the page (TraceDataLoss).
                    "recordMode": "recordAsMuchAsPossible",
                    "traceBufferSizeInKb": TRACE_BUFFER_KB,
                },
                "transferMode": "ReportEvents",
            },
        )
        self._tracing = True
        status: dict[str, str] = {}
        walls: dict[str, float] = {}
        for name in areas:
            spec = AREAS[name]
            t_area = time.time()
            if not spec.get("keep"):
                self.reset()
            try:
                if spec.get("setup") and not self.act(spec["setup"]):
                    status[name] = "NOT_RUN: setup control not found"
                    continue
                if spec.get("setup"):
                    self.wait(("settle",), 0)
                base = page.evaluate("() => window.__gateOverlays()")
                ready = spec["ready"]
                self.attr_before = (
                    page.evaluate("([s, a]) => window.__gateAttr(s, a)", [ready[1], ready[2]])
                    if ready[0] == "changed"
                    else None
                )
                self.focusins_before = page.evaluate("() => window.__gateFocusins.length")
                page.evaluate("(n) => console.timeStamp('gate:begin:' + n)", name)
                self.last_error = ""
                if not self.act(spec["open"]):
                    status[name] = f"NOT_RUN: {self.last_error or 'control not found'}"
                    page.evaluate("(n) => console.timeStamp('gate:end:' + n)", name)
                    continue
                opened = self.wait(spec["ready"], base)
                closed = True
                if spec.get("close"):
                    self.act(spec["close"])
                    closed = self.wait(spec["gone"], base + 1)
                page.evaluate("(n) => console.timeStamp('gate:end:' + n)", name)
                status[name] = area_status(spec, opened, closed)
                if spec.get("teardown"):
                    self.act(spec["teardown"])
                    self.wait(("settle",), 0)
                walls[name] = time.time() - t_area
            except Exception as exc:  # noqa: BLE001 - an area that breaks is reported, the sweep goes on
                status[name] = f"NOT_RUN: {type(exc).__name__}: {str(exc)[:120]}"
                page.evaluate("(n) => console.timeStamp('gate:end:' + n)", name)
        t_end = time.time()
        cdp.send("Tracing.end")
        self._tracing = False
        while not done:
            page.wait_for_timeout(50)
        t_split = time.time()
        if done[0]:
            raise TraceDataLoss(f"Chromium reported trace data loss ({len(events):,} events kept)")
        counters = split_trace(events)
        self.timing = {
            "areas_s": round(t_end - t_sweep, 1),
            "collect_s": round(t_split - t_end, 1),
            "split_s": round(time.time() - t_split, 1),
            "events": len(events),
        }
        return finish_cells(areas, status, walls, counters)


# --- trace -> per-area counters --------------------------------------------------------------------


class TraceDataLoss(RuntimeError):
    """The trace of a page is incomplete; its counters cannot be trusted."""


TRACE_BUFFER_KB = int(os.environ.get("LONG_THREAD_AREAS_TRACE_BUFFER_KB", str(256 * 1024)))


def finish_cells(areas: list[str], status: dict, walls: dict, counters: dict) -> dict[str, dict]:
    """Per-area result. An area the page ran but whose trace window (both marks) is missing has no
    counters to judge, so it is NOT_RUN, never an OK with zero cost."""
    out = {}
    for name in areas:
        st = status.get(name, "NOT_RUN")
        row = counters.get(name)
        if st == "OK" and not row:
            st = "NOT_RUN: its trace window is incomplete (a begin or end mark is missing)"
        out[name] = {"status": st, "wall_s": round(walls.get(name, 0.0), 2), **(row or {})}
    return out


def area_status(spec: dict, opened: bool, closed: bool) -> str:
    """OK only when the area visibly opened and, if it closes, visibly closed again."""
    if not opened:
        return f"NOT_RUN: never reached {spec['ready'][0]}"
    if not closed:
        return f"NOT_RUN: opened but never reached {spec['gone'][0]} after closing"
    return "OK"


def split_trace(events: list) -> dict[str, dict]:
    names, procs = {}, {}
    windows: dict[str, list] = {}
    for e in events:
        if e.get("ph") == "M":
            if e.get("name") == "thread_name":
                names[(e["pid"], e["tid"])] = (e.get("args") or {}).get("name", "")
            elif e.get("name") == "process_name":
                procs[e["pid"]] = (e.get("args") or {}).get("name", "")
        elif e.get("name") == "TimeStamp":
            msg = ((e.get("args") or {}).get("data") or {}).get("message") or ""
            if msg.startswith("gate:"):
                _, edge, area = msg.split(":", 2)
                windows.setdefault(area, [None, None])[0 if edge == "begin" else 1] = e["ts"]
    spans = sorted(
        (w[0], w[1], a) for a, w in windows.items() if w[0] is not None and w[1] is not None
    )
    out = {
        a: {
            "elements": 0,
            "layout_objects": 0,
            "style_ms": 0.0,
            "layout_ms": 0.0,
            "paint_ms": 0.0,
            "longest_task_ms": 0.0,
            "cpu_ms": 0.0,
            "threads": {},
        }
        for _, _, a in spans
    }

    def area_at(ts):
        for b, en, a in spans:
            if b <= ts <= en:
                return a
        return None

    for e in events:
        if e.get("ph") != "X" or "ts" not in e:
            continue
        a = area_at(e["ts"])
        if a is None:
            continue
        row, name, cat = out[a], e.get("name"), e.get("cat", "")
        thread_name = names.get((e.get("pid"), e.get("tid")), "")
        dur = e.get("dur", 0) / 1000
        if cat == "toplevel" and name in (
            "ThreadControllerImpl::RunTask",
            "RunTask",
            "ThreadPool_RunTask",
        ):
            if procs.get(e.get("pid")) == "Browser":
                continue
            ms = e.get("tdur", e.get("dur", 0)) / 1000
            row["cpu_ms"] += ms
            row["threads"][thread_name] = row["threads"].get(thread_name, 0.0) + ms
            if thread_name == "CrRendererMain":
                row["longest_task_ms"] = max(row["longest_task_ms"], dur)
        elif thread_name == "CrRendererMain":
            args = e.get("args") or {}
            if name == "UpdateLayoutTree":
                row["style_ms"] += dur
                row["elements"] += int(args.get("elementCount") or 0)
            elif name == "Layout":
                row["layout_ms"] += dur
                row["layout_objects"] += int((args.get("beginData") or {}).get("dirtyObjects") or 0)
            elif name in ("Paint", "PrePaint", "Layerize", "PaintImage"):
                row["paint_ms"] += dur
    return out


# --- matrix ---------------------------------------------------------------------------------------


# The full product (every mode x every area x both lengths) is LONG_THREAD_AREAS_FULL=1. By default,
# for CI time: every material x every area at SHORT turns; the default and glass at LONG turns; the
# settings layered on a material (lens modes, glass menus off, dark) x VARIANT_AREAS, the areas
# those settings change (the composer and its lens, menus and dialogs, scrolling, cards).
FULL = os.environ.get("LONG_THREAD_AREAS_FULL") == "1"
LONG_MODES = os.environ.get("LONG_THREAD_AREAS_LONG_MODES", "all" if FULL else f"default,{GLASS}")
VARIANT_AREAS = [
    "type",
    "thread_scroll",
    "settings",
    "settings_scroll",
    "composer_menu",
    "model_picker",
    "message_menu",
    "source_card",
]
# Two browsers: on a 4-CPU runner a third adds contention, not throughput (measured).
WORKERS = int(os.environ.get("LONG_THREAD_AREAS_WORKERS", "2"))


def is_variant(mode: dict) -> bool:
    """A setting layered on a material (lens, glass menus, theme), not a material itself."""
    return "+" in mode["name"]


def plan(
    all_modes: list[dict],
    long_modes: str = LONG_MODES,
    full: bool = FULL,
) -> list[tuple[int, dict, list[str]]]:
    """Pages to load, with the areas each runs: every mode at SHORT turns, LONG_MODES at LONG turns."""
    wanted = {m.strip() for m in long_modes.split(",")}
    longs = [m for m in all_modes if "all" in wanted or m["name"] in wanted]
    every = list(AREAS)

    def areas_for(mode):
        return every if full or not is_variant(mode) else with_deps(VARIANT_AREAS)

    return [(LONG, m, areas_for(m)) for m in longs] + [(SHORT, m, areas_for(m)) for m in all_modes]


def _run_jobs(
    browser,
    jobs,
    base: str,
    out: dict,
    log,
    started,
    page_factory = None,
) -> None:
    """Measure each job's page into `out`; any failure is recorded as that page's error, never dropped."""
    page_factory = page_factory or Page
    for turns, mode, page_areas in jobs:
        key = (turns, mode["name"])
        page = None
        for attempt in range(2):
            try:
                page = page_factory(browser, base, turns, mode, [])
                break
            except Exception as exc:  # noqa: BLE001 - one retry, then the page is reported
                if attempt:
                    out[key] = {"error": f"page did not load: {type(exc).__name__}: {exc}"[:300]}
        if page is None:
            continue
        try:
            try:
                cells = page.sweep(page_areas)
            except TraceDataLoss as exc:
                # once more on a fresh page; a second loss is reported as this page's error
                log(
                    f"  areas: {turns} turns, {mode['name']}: {exc}; measuring the page again",
                    flush = True,
                )
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass
                page = page_factory(browser, base, turns, mode, [])
                cells = page.sweep(page_areas)
            out[key] = {
                "cells": cells,
                "applied": page.applied,
                "errors": page.errors[:3],
                "triggers": page.page.evaluate(TRIGGERS) if key == (SHORT, "default") else None,
            }
        except Exception as exc:  # noqa: BLE001 - a broken page is a finding, not a silent gap
            out[key] = {"error": f"measuring the page failed: {type(exc).__name__}: {exc}"[:300]}
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001 - the error above is the one to report
                pass
        log(f"  areas: {turns} turns, {mode['name']} ({time.time() - started:.0f} s)", flush = True)


def _worker(
    jobs: list[tuple[int, dict, list[str]]],
    base: str,
    launch_args,
    areas: list[str],
    out: dict,
    log,
    started,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = launch_args)
            try:
                _run_jobs(browser, jobs, base, out, log, started)
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - the browser itself failed: every page it owed is reported
        for turns, mode, _ in jobs:
            out.setdefault(
                (turns, mode["name"]),
                {"error": f"worker failed: {type(exc).__name__}: {exc}"[:300]},
            )


def fill_missing(out: dict, jobs) -> None:
    """Every planned page must have a result; one that has none is an error, so the gate cannot pass."""
    for turns, mode, _ in jobs:
        out.setdefault(
            (turns, mode["name"]), {"error": "no result: the worker measuring it stopped"}
        )


def measure(
    dist: Path,
    launch_args,
    areas: list[str] | None = None,
    long_modes: str = LONG_MODES,
    workers: int = WORKERS,
    log = print,
) -> dict:
    """{(turns, mode): {cells, applied, ...}} for the planned pages, in `workers` browsers."""
    areas = areas or list(AREAS)
    all_modes = app_modes()
    jobs = plan(all_modes, long_modes)
    server, base = serve_app(dist)
    started = time.time()
    out: dict = {}
    try:
        # Balanced by expected cost (areas, and a long chat at about 1.5x), heaviest first.
        shards: list[list] = [[] for _ in range(max(1, workers))]
        load = [0.0] * len(shards)
        for job in sorted(jobs, key = lambda j: -len(j[2]) * (1.5 if j[0] == LONG else 1.0)):
            k = load.index(min(load))
            shards[k].append(job)
            load[k] += len(job[2]) * (1.5 if job[0] == LONG else 1.0)
        threads = [
            threading.Thread(target = _worker, args = (sh, base, launch_args, areas, out, log, started))
            for sh in shards
            if sh
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        fill_missing(out, jobs)
        return {
            "planned": [(t, m["name"]) for t, m, _ in jobs],
            "pages": out,
            "modes": {m["name"]: m for m in all_modes},
            "areas": areas,
            "seconds": time.time() - started,
            "dist": dist,
            "launch_args": launch_args,
        }
    finally:
        server.shutdown()


def remeasure(
    result: dict,
    items: list[tuple[int, str, str]],
    log = print,
) -> None:
    """Measure (turns, mode, area) again on fresh pages, alternating with the base; keep every reading."""
    from playwright.sync_api import sync_playwright

    server, base = serve_app(result["dist"])
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = result["launch_args"])
            try:
                by_page: dict[tuple[int, str], set] = {}
                for turns, mode, area in items:
                    parent = result["modes"][mode]["base"] or "default"
                    for name in (parent, mode):
                        by_page.setdefault((turns, name), set()).add(area)
                for _ in range(RECHECKS):
                    for (turns, name), areas in sorted(by_page.items()):
                        page = Page(browser, base, turns, result["modes"][name], [])
                        try:
                            cells = page.sweep(with_deps(areas))
                        finally:
                            page.close()
                        row = result["pages"][(turns, name)]["cells"]
                        for area, cell in cells.items():
                            if area in areas:
                                row[area].setdefault("again", []).append(cell)
                log(f"  areas: rechecked {len(items)} cells", flush = True)
            finally:
                browser.close()
    finally:
        server.shutdown()


def _in_thread(fn, *args):
    """Sync Playwright cannot start inside another sync Playwright's thread (the gate's own)."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    t = threading.Thread(target = target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def with_deps(wanted) -> list[str]:
    """The areas to run, in AREAS order, with the areas a "keep" area relies on (the one that opened its dialog)."""
    names = list(AREAS)
    need = set(wanted)
    for name in wanted:
        k = names.index(name)
        while AREAS[names[k]].get("keep") and k > 0:
            k -= 1
            need.add(names[k])
    return [n for n in names if n in need]


def settled(cell: dict) -> dict:
    """A cell that did not run, but ran when measured again, counts as that later reading."""
    if cell.get("status") == "OK":
        return cell
    for again in cell.get("again", []):
        if again.get("status") == "OK":
            return {**again, "again": [a for a in cell.get("again", []) if a is not again]}
    return cell


def cheapest(cell: dict, key: str) -> float:
    vals = [cell.get(key, 0.0)] + [
        c.get(key, 0.0) for c in cell.get("again", []) if c.get("status") == "OK"
    ]
    return min(vals)


def judge(result: dict, known: str | None = None) -> list[dict]:
    """Verdict rows: kind length / appearance / broken; `fail` set on the failing ones."""
    limits = known_limits(KNOWN if known is None else known)
    pages, out = result["pages"], []
    for (turns, mode), page in sorted(pages.items()):
        if "error" in page:
            out.append(
                {
                    "kind": "broken",
                    "turns": turns,
                    "mode": mode,
                    "area": "*",
                    "fail": True,
                    "why": page["error"],
                }
            )
    for (turns, mode), page in sorted(pages.items()):
        if "cells" not in page:
            continue
        for area in result["areas"]:
            if area not in page["cells"]:
                continue
            cell = settled(page["cells"].get(area, {}))
            ok = cell.get("status") == "OK"
            if mode == "default" and not ok:
                out.append(
                    {
                        "kind": "broken",
                        "turns": turns,
                        "mode": mode,
                        "area": area,
                        "fail": True,
                        "why": f"{cell.get('status')}: the area no longer runs on the default page (renamed control or "
                        "changed predicate? update AREAS in tests/studio/_area_matrix.py)",
                    }
                )
                continue
            parent = result["modes"][mode]["base"]
            if parent is None:
                continue
            bpage = pages.get((turns, parent), {})
            if area not in bpage.get("cells", {}):
                continue
            bcell = settled(bpage.get("cells", {}).get(area, {}))
            if bcell.get("status") != "OK":
                continue
            if not ok:
                out.append(
                    {
                        "kind": "broken",
                        "turns": turns,
                        "mode": mode,
                        "area": area,
                        "fail": True,
                        "why": f"{cell.get('status')} in this mode, OK in {parent}",
                    }
                )
                continue
            limit = limits.get((mode, area), limits.get((mode, "*"), MAX_RATIO))
            mc, bc = cheapest(cell, "cpu_ms"), cheapest(bcell, "cpu_ms")
            ratio = mc / max(bc, CPU_BASE_FLOOR_MS)
            out.append(
                {
                    "kind": "appearance",
                    "metric": "cpu_ms",
                    "turns": turns,
                    "mode": mode,
                    "base": parent,
                    "area": area,
                    "value": mc,
                    "base_value": bc,
                    "ratio": ratio,
                    "limit": limit,
                    "fail": ratio > limit and mc - bc >= CPU_FLOOR_MS,
                    "threads": cell.get("threads", {}),
                }
            )
            me, be = cell.get("elements", 0), bcell.get("elements", 0)
            r_el = me / max(be, 1)
            out.append(
                {
                    "kind": "appearance",
                    "metric": "elements",
                    "turns": turns,
                    "mode": mode,
                    "base": parent,
                    "area": area,
                    "value": me,
                    "base_value": be,
                    "ratio": r_el,
                    "limit": limit,
                    "fail": r_el > limit and me - be >= COUNT_FLOOR,
                }
            )
    for (turns, mode), page in sorted(pages.items()):
        if turns != LONG or "cells" not in page or "cells" not in pages.get((SHORT, mode), {}):
            continue
        short = pages[(SHORT, mode)]["cells"]
        for area in result["areas"]:
            if area not in page["cells"] or area not in short:
                continue
            lc, sc = settled(page["cells"].get(area, {})), settled(short.get(area, {}))
            if lc.get("status") != "OK" or sc.get("status") != "OK":
                continue
            limit = limits.get(("length", area), LENGTH_RATIO)
            cap = limits.get(("count", area))
            for metric in ("elements", "layout_objects"):
                lv, sv = lc.get(metric, 0), sc.get(metric, 0)
                ratio = lv / max(sv, 1)
                if metric == "elements" and cap is not None:
                    out.append(
                        {
                            "kind": "length",
                            "metric": metric,
                            "turns": turns,
                            "mode": mode,
                            "area": area,
                            "value": lv,
                            "base_value": sv,
                            "ratio": ratio,
                            "limit": limit,
                            "cap": cap,
                            "fail": lv > cap,
                        }
                    )
                    continue
                out.append(
                    {
                        "kind": "length",
                        "metric": metric,
                        "turns": turns,
                        "mode": mode,
                        "area": area,
                        "value": lv,
                        "base_value": sv,
                        "ratio": ratio,
                        "limit": limit,
                        "fail": ratio > limit and lv >= COUNT_FLOOR,
                    }
                )
    return out


def trigger_matches(key: str, pattern: str) -> bool:
    import fnmatch
    return (
        fnmatch.fnmatchcase(key, pattern)
        or fnmatch.fnmatchcase(key, f"*:{pattern}")
        or key.startswith(pattern)
    )


def uncovered(triggers: dict[str, int]) -> list[str]:
    patterns = [c for spec in AREAS.values() for c in spec.get("covers", [])] + list(NOT_AREAS)
    return sorted(k for k in triggers if not any(trigger_matches(k, p) for p in patterns))


def culprit(result: dict, turns: int, mode: str, area: str) -> list[str]:
    """Re-run one area with one paint property switched off at a time; name what removes the cost."""
    from playwright.sync_api import sync_playwright

    server, base = serve_app(result["dist"])
    out = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = result["launch_args"])
            try:
                page = Page(browser, base, turns, result["modes"][mode], [])
                try:
                    run = with_deps([area])
                    shipped = min(page.sweep(run)[area]["cpu_ms"] for _ in range(2))
                    out.append(
                        f"    measured, {mode} / {area} at {turns} turns, {shipped:.0f} ms CPU as shipped:"
                    )
                    found = []
                    for prop, css in OFF.items():
                        page.page.evaluate(OVERRIDE, css)
                        page.page.wait_for_timeout(300)
                        ms = min(page.sweep(run)[area]["cpu_ms"] for _ in range(2))
                        page.page.evaluate(OVERRIDE, "")
                        out.append(
                            f"      without {prop:<15} {ms:8.0f} ms ({ms / max(shipped, 1):.2f}x)"
                        )
                        if ms <= 0.8 * shipped:
                            found.append((ms, prop))
                    for _, prop in sorted(found):
                        out.append(f"    likely culprit: {prop}: {HINTS[prop]}")
                    if not found:
                        out.append(
                            "    no single paint property explains it; see the threads above"
                        )
                finally:
                    page.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
    return out


def thread_split(threads: dict[str, float]) -> str:
    top = sorted(threads.items(), key = lambda x: -x[1])[:3]
    return ", ".join(f"{n or '?'} {v:.0f} ms" for n, v in top)


def run(
    dist: Path,
    launch_args,
    log = print,
) -> tuple[list[str], list[str], list[str], dict]:
    """(report lines, failures, details, result)."""
    started = time.time()
    result = measure(dist, launch_args, log = log)
    verdicts = judge(result)
    # One reading per cell, so every CPU failure is measured again (fresh pages of the mode and its
    # base, RECHECKS times, the cheapest reading kept). Grouped by page; at most 4 mode pages.
    again = sorted(
        {
            (v["turns"], v["mode"], v["area"])
            for v in verdicts
            if v["fail"]
            and (
                v["kind"] == "appearance"
                and v["metric"] == "cpu_ms"
                or v["kind"] == "broken"
                and v["area"] != "*"
            )
        }
    )
    # Pages with the fewest failing cells first (a lone outlier is the usual noise); a mode that
    # fails many cells is one page anyway.
    per_page: dict = {}
    for t, m, _ in again:
        per_page[(t, m)] = per_page.get((t, m), 0) + 1
    keep_pages = sorted(per_page, key = lambda k: per_page[k])[:RECHECK_PAGES]
    again = [k for k in again if (k[0], k[1]) in keep_pages]
    if again and RECHECKS:
        _in_thread(remeasure, result, again, log)
        verdicts = judge(result)
    failures, details = [], []
    trig = (result["pages"].get((SHORT, "default")) or {}).get("triggers")
    if trig is None:
        failures.append(
            f"no popup-trigger list: the default page at {SHORT} turns was not measured, so coverage could not be checked"
        )
    missing = uncovered(trig or {})
    if missing:
        failures.append(
            f"{len(missing)} popup trigger(s) on the chat page belong to no measured area: {', '.join(missing[:8])}. "
            "Add an area for each to AREAS in tests/studio/_area_matrix.py (or to NOT_AREAS with the reason)"
        )
    for v in verdicts:
        if not v["fail"]:
            continue
        if v["kind"] == "broken":
            failures.append(f"area {v['area']} ({v['mode']}, {v['turns']} turns): {v['why']}")
        elif v["kind"] == "length":
            bound = (
                f"known ceiling {v['cap']:,.0f} elements, LONG_THREAD_AREAS_KNOWN count:{v['area']}=..."
                if "cap" in v
                else f"limit {v['limit']:.1f}x; LONG_THREAD_AREAS_LENGTH_RATIO / _KNOWN length:{v['area']}=..."
            )
            failures.append(
                f"area {v['area']} ({v['mode']}) costs more as the chat grows: {v['metric'].replace('_', ' ')} "
                f"{v['base_value']:,} at {SHORT} turns -> {v['value']:,} at {LONG} turns = {v['ratio']:.1f}x ({bound}). "
                "Opening it restyles or re-lays out the thread. Usual causes: a non-subject :has() rule "
                "(`X:has(...) Y` or `X:has(...) > Y`, e.g. a Tailwind group-has-* variant or a rule anchored on "
                "[data-wallpaper-workspace]): Chromium and WebKit re-run it for every message when anything under "
                "an ancestor of the thread changes (see the :has() lint at the top of this log); or an attribute or "
                "inherited style written on an ancestor of the thread (body, html, the sidebar provider)"
            )
        else:
            unit = "ms CPU" if v["metric"] == "cpu_ms" else "elements restyled"
            failures.append(
                f"appearance mode {v['mode']} makes area {v['area']} cost {v['value']:,.0f} {unit} against "
                f"{v['base_value']:,.0f} in {'the default appearance' if v['base'] == 'default' else v['base']} at "
                f"{v['turns']} turns = {v['ratio']:.1f}x (limit {v['limit']:.1f}x; LONG_THREAD_AREAS_MAX_RATIO / _KNOWN "
                f"{v['mode']}:{v['area']}=...)"
            )
            if v["metric"] == "cpu_ms":
                details.append(
                    f"  {v['mode']} / {v['area']}: busiest threads {thread_split(v['threads'])}"
                )
    worst = max(
        (
            v
            for v in verdicts
            if v["fail"] and v["kind"] == "appearance" and v["metric"] == "cpu_ms"
        ),
        key = lambda v: v["ratio"] / v["limit"],
        default = None,
    )
    if worst is not None and time.time() - started < 900:
        try:
            details += _in_thread(culprit, result, worst["turns"], worst["mode"], worst["area"])
        except Exception as exc:  # noqa: BLE001 - the diagnosis must never hide the failure
            details.append(f"    (culprit probe failed: {exc!r})")
    lines = table(result, verdicts)
    lines.append(
        f"  areas x appearance x length: {len(result['areas'])} areas, {len(result['pages'])} pages, "
        f"{time.time() - started:.0f} s"
    )
    return lines, failures, details, {"result": result, "verdicts": verdicts}


def table(result: dict, verdicts: list[dict]) -> list[str]:
    by = {
        (v["kind"], v.get("metric"), v["turns"], v["mode"], v["area"]): v
        for v in verdicts
        if v["kind"] != "broken"
    }
    lines = [
        f"  areas at {SHORT} turns: CPU ms (all browser threads) in the default appearance, then x per mode "
        f"(! = over its limit); elements restyled {SHORT} -> {LONG} turns"
    ]
    modes = [m for m in result["modes"] if m != "default" and (SHORT, m) in result["pages"]]
    short = (result["pages"].get((SHORT, "default")) or {}).get("cells", {})
    long_ = (result["pages"].get((LONG, "default")) or {}).get("cells", {})
    lines.append(
        "    "
        + f"{'area':<18}{'default':>9}"
        + "".join(f"{m[:9]:>10}" for m in modes)
        + f"{'restyled':>22}"
    )
    for area in result["areas"]:
        row = f"    {area:<18}{cheapest(short.get(area, {}), 'cpu_ms') if short.get(area) else 0:>9.0f}"
        for m in modes:
            v = by.get(("appearance", "cpu_ms", SHORT, m, area))
            row += f"{v['ratio']:>9.1f}{'!' if v['fail'] else ' '}" if v else f"{'-':>10}"
        lv = by.get(("length", "elements", LONG, "default", area))
        row += (
            f"{short.get(area, {}).get('elements', 0):>11,} -> {long_.get(area, {}).get('elements', 0):>7,}"
            + ("!" if lv and lv["fail"] else "")
        )
        lines.append(row)
    return lines
