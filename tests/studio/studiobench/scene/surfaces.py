# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The registry of Unsloth's UI SURFACES, for the parity sweep.

WHY THIS EXISTS. The film drives eighteen actions against the chat thread plus a handful of
overlays, and the parity digest taken at the close of each one is what licenses the claim "this
change does not alter the UI". That claim is far wider than the evidence: Unsloth has fifteen
routes, a settings dialog with twelve lazily loaded panels, a sidebar with six menus, a model
picker with three section tabs, and the media pages. None of them is on the film's path, so a
change that repainted every one of them would pass eighteen out of eighteen parity checks.

WHAT A SURFACE IS. A named, reachable state of the app, with:

    reach     the steps that get there FROM THE KNOWN STATE, never from wherever the last
              surface happened to leave things
    settle    the observation that says it has actually arrived. Never a sleep: a sleep cannot
              tell a surface that rendered from one that never did, and a digest of a surface that
              never rendered is the exact defect this tool is about
    restore   the steps back to the known state, so surface N+1 starts where surface N started
    root      the element the digest is scoped to. Ordered candidates, first visible one wins

THE TRAP THAT SHAPES THE ROOTS. ChatPage, ImagesPage, VideoPage and AudioPage are mounted
PERSISTENTLY by the root layout so an in-flight generation survives leaving the tab; off-route
they are `hidden` and `inert` but still in the document. A digest taken with parity.js's default
root therefore reads the hidden chat thread on every non-chat route, and forty surfaces report one
identical digest that every one of them passes. `@route` resolves the ACTIVE route container by
the property the layout actually sets -- the off-route siblings carry `inert` -- rather than by a
class name that a refactor would quietly change.

WHAT IS DELIBERATELY NOT HERE is listed in `KNOWN_UNCOVERED` at the bottom, with the mechanism
that puts it out of reach. A surface missing from a coverage report because nobody thought of it
and a surface missing because it cannot be reached without destroying the session look identical
in a manifest that only lists what it swept, so the second kind is enumerated.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

#: The state every surface is reached from and restored to: a fresh, empty chat. Fresh matters.
#: The keep-alive chat container is inside `@route`'s siblings and inside `@shell`, so a sweep run
#: against a loaded thread would carry that thread's DOM into the digest of every other surface,
#: and any thread difference would flip all forty of them at once.
KNOWN_STATE_PATH = "/chat"

#: Steps are (verb, *args). Interpreted by surface_sweep, which owns the browser; keeping them
#: declarative is what lets the unit tests check the registry with no browser present.
#:
#:   goto      <path>              navigate, relative to the Unsloth base url
#:   click     <selector>          a REAL mouse click through the driver. Not element.click():
#:                                 Radix menus open on pointerdown and a synthetic click misses
#:                                 them entirely, which reads as a menu that opened instantly
#:   click_if  <selector>          click only when present. For controls that exist in one of two
#:                                 states, e.g. the sidebar is already collapsed
#:   hover     <selector>          park the pointer over an element. REQUIRED before four of the
#:                                 sidebar's controls: the row actions ship
#:                                 `opacity-0 pointer-events-none` and only take
#:                                 `group-hover:pointer-events-auto` when their row is hovered, so
#:                                 a click on them times out against an element that is present,
#:                                 sized and reported visible by every check except the one that
#:                                 matters
#:   press     <key>               a key on the focused element
#:   fill      <selector> <text>   set an input's value through the driver
#:   wait      <ms>                a bounded pause. Only ever ALONGSIDE a settle condition, never
#:                                 instead of one
Step = tuple


@dataclass(frozen = True)
class Surface:
    """One reachable UI surface."""

    id: str
    group: str
    title: str
    reach: tuple[Step, ...]
    restore: tuple[Step, ...]
    root: tuple[str, ...]
    settle: Optional[dict] = None
    #: Set when the surface's absence is a legitimate property of THIS installation rather than a
    #: broken selector: the Connected tab of the model picker needs an external provider, the hub
    #: catalog needs network. A surface that fails to reach still records a reason either way;
    #: this decides whether the manifest counts it against coverage.
    conditional: Optional[str] = None
    #: Surfaces the film already drives. Swept anyway -- the sweep runs on an EMPTY chat and the
    #: film on a loaded one, so the two digests are of different states -- but flagged so the
    #: coverage figure is not inflated by re-counting what was already covered.
    also_in_film: bool = False
    #: The mechanism by which this surface's digest differs between two runs of the SAME build.
    #: Measured, not assumed: three consecutive sweeps against one Unsloth agreed on 44 of 53
    #: surfaces, and every entry below is the mechanism behind one of the nine that did not.
    #:
    #: This is the same distinction `scripts/sbench_ui_parity.py` already draws for the film's
    #: actions with UNSTABLE_ACTIONS, and for the same reason: a comparison that reports a live
    #: memory gauge as a UI change gets ignored within a day, and then so does the rest of it.
    volatile: Optional[str] = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group": self.group,
            "title": self.title,
            "reach": [list(s) for s in self.reach],
            "restore": [list(s) for s in self.restore],
            "root": list(self.root),
            "settle": self.settle,
            "conditional": self.conditional,
            "also_in_film": self.also_in_film,
            "volatile": self.volatile,
            "notes": self.notes,
        }


# ── shared fragments ────────────────────────────────────────────────

#: Back to the known state the hard way. Used as the restore for anything that navigated, and as
#: the recovery path when a surface's own restore left the app dirty.
HOME: tuple[Step, ...] = (("goto", KNOWN_STATE_PATH), ("wait", 400))

#: Escape twice, not once. A settings tab that has opened a sub-view (Data -> Archived chats,
#: Voice -> Dictionary) eats the first Escape closing the sub-view and stays open on the dialog.
ESCAPE_OUT: tuple[Step, ...] = (
    ("press", "Escape"),
    ("wait", 200),
    ("press", "Escape"),
    ("wait", 300),
)

#: The chat composer. Its presence is what "the chat route has rendered" means, everywhere.
COMPOSER = 'textarea[aria-label="Message input"]'

ROOT_ROUTE = ("@route",)
ROOT_SHELL = ("@shell",)
ROOT_SIDEBAR = ("@sidebar",)


#: The empty chat prints a randomly chosen greeting. `pickRandom` in
#: components/assistant-ui/thread.tsx:2019 selects from a time-of-day list, so two loads of one
#: build render different headings, and every digest whose root contains the welcome screen
#: differs run to run for that reason alone.
GREETING_IS_RANDOM = (
    "the empty chat's greeting is chosen by pickRandom "
    "(components/assistant-ui/thread.tsx:2019), so two loads of the same build "
    "render different headings"
)

#: The hub catalogue is fetched from Hugging Face. Download counts, trending order and the arrival
#: time of the list all move independently of the build under test.
HUB_IS_REMOTE = (
    "the catalogue is fetched from Hugging Face, so its content and its arrival time "
    "both move independently of the build under test"
)

#: MEASURED TO MOVE, MECHANISM NOT ESTABLISHED. Three sweeps of one build disagreed on these, and
#: a back-to-back pair of visits produced byte-identical markup, so it is not a timer and not a
#: render race inside the surface. It is state that something earlier in the sweep, or the install,
#: carries between sweeps -- and which state has not been pinned down.
#:
#: Flagged rather than left comparable, and flagged as UNEXPLAINED rather than given a plausible
#: cause. A wrong mechanism in this field is worse than an admitted gap: it is the sentence a
#: reader uses to dismiss a real difference.
UNEXPLAINED_DRIFT = (
    "measured to differ between sweeps of the SAME build while a back-to-back "
    "pair of visits produced byte-identical markup. The mechanism is NOT "
    "established, so the digest is not treated as a parity signal until it is"
)


def _route(
    surface_id: str,
    title: str,
    path: str,
    settle: dict,
    conditional: Optional[str] = None,
    notes: str = "",
    volatile: Optional[str] = None,
) -> Surface:
    return Surface(
        id = surface_id,
        group = "route",
        title = title,
        reach = (("goto", path), ("wait", 600)),
        settle = settle,
        restore = HOME,
        root = ROOT_ROUTE,
        conditional = conditional,
        notes = notes,
        volatile = volatile,
    )


def _settings_tab(tab_id: str, title: str) -> Surface:
    # `/settings` is not a page. Its route component calls `openDialog()` and immediately
    # redirects to /chat, so the dialog is the surface and the URL never stays on /settings --
    # a settle condition written against the URL would never be satisfied.
    return Surface(
        id = f"settings:{tab_id}",
        group = "settings",
        title = title,
        reach = (
            ("goto", "/settings"),
            ("wait", 500),
            ("click", f'[data-testid="settings-tab-{tab_id}"]'),
            ("wait", 400),
        ),
        # The panel is lazily imported per tab, so "the dialog is open" is not enough: the tab
        # would be recorded as reached while its chunk was still in flight and the digest would
        # be of the Suspense fallback. The tab's own pressed state plus a settled panel body is
        # the observation that the panel arrived.
        settle = {
            "js": f"!!document.querySelector('[data-testid=\"settings-tab-{tab_id}\"]')"
            f' && !!document.querySelector(".settings-surface main")'
            f' && (document.querySelector(".settings-surface main").innerText || "")'
            f".trim().length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = (".settings-surface", '[data-slot="dialog-content"]'),
    )


# ── the registry ────────────────────────────────────────────────────

_SURFACES: list[Surface] = [
    # ── routes ──────────────────────────────────────────────────────
    _route(
        "route:chat",
        "Chat, empty state",
        "/chat",
        {"visible": COMPOSER},
        volatile = GREETING_IS_RANDOM,
    ),
    _route("route:projects", "Projects", "/projects", {"js": 'location.pathname === "/projects"'}),
    _route(
        "route:hub",
        "Hub / model catalogue",
        "/hub",
        {"js": 'location.pathname === "/hub"'},
        conditional = "the catalogue needs network; without it the page renders its own "
        "network error state, which is itself a surface worth digesting",
        volatile = HUB_IS_REMOTE,
    ),
    _route(
        "route:train",
        "Train (Unsloth)",
        "/studio",
        # NOT a URL check alone, and not the nav row either: the nav row is in the sidebar and
        # is there before the page is. StudioPage shows a spinner while the hardware verdict is
        # unmeasured and then redirects to /chat if the host is chat-only, so the wizard's own
        # last section is what says the page finished rendering.
        {
            "js": 'location.pathname === "/studio" && '
            '(document.body.innerText || "").includes("Run preview")'
        },
        conditional = "a chat-only host redirects /studio to /chat by design",
        volatile = UNEXPLAINED_DRIFT,
    ),
    _route("route:export", "Export", "/export", {"js": 'location.pathname === "/export"'}),
    _route("route:images", "Images", "/images", {"js": 'location.pathname === "/images"'}),
    _route(
        "route:video",
        "Video",
        "/video",
        {"js": 'location.pathname === "/video"'},
        conditional = "a host without video support renders the gate instead of the page",
    ),
    _route("route:audio", "Audio", "/audio", {"js": 'location.pathname === "/audio"'}),
    _route(
        "route:data-recipes",
        "Data recipes",
        "/data-recipes",
        {"js": 'location.pathname === "/data-recipes"'},
    ),
    _route(
        "route:api-monitor",
        "API monitor",
        "/api-monitor",
        {"js": 'location.pathname === "/api-monitor"'},
    ),
    # The 404 is a rendered surface with its own mascot, heading and back-link, and it is exactly
    # the kind of page a bundler or asset-path change breaks without anyone noticing.
    _route(
        "route:not-found",
        "Not found",
        "/studiobench-no-such-route",
        {"js": 'location.pathname === "/studiobench-no-such-route"'},
    ),
    # ── the settings dialog ─────────────────────────────────────────
    _settings_tab("general", "Settings: General"),
    _settings_tab("profile", "Settings: Profile"),
    _settings_tab("appearance", "Settings: Appearance"),
    dataclasses.replace(
        _settings_tab("resources", "Settings: Resources"),
        volatile = "the panel shows live CPU, memory and disk gauges. Two visits a minute apart "
        "read 1% vs 2% CPU and 82.2 GiB vs 177 GiB used",
    ),
    _settings_tab("chat", "Settings: Chat"),
    _settings_tab("voice", "Settings: Voice"),
    _settings_tab("connections", "Settings: Connections"),
    _settings_tab("data", "Settings: Data"),
    _settings_tab("api-keys", "Settings: API keys"),
    dataclasses.replace(_settings_tab("agents", "Settings: Agents"), volatile = UNEXPLAINED_DRIFT),
    dataclasses.replace(
        _settings_tab("debugging", "Settings: Debugging"),
        volatile = "the panel renders a live tail of the server log, so it carries request "
        "timestamps. Two visits differed by 402 lines",
    ),
    _settings_tab("about", "Settings: About"),
    # The dialog's own search index is a distinct rendering of every panel's labels, and it is the
    # one place a settings string appears without its panel being mounted.
    Surface(
        id = "settings:search",
        group = "settings",
        title = "Settings: search results",
        # By aria-label, not `input[type="text"]`: the search box ships no `type` attribute at
        # all, so a type selector matches the Embedding model field further down the General
        # panel and fills THAT instead -- a reach that succeeds and lands somewhere else.
        reach = (
            ("goto", "/settings"),
            ("wait", 500),
            ("fill", '.settings-surface input[aria-label^="Search settings"]', "model"),
            ("wait", 500),
        ),
        settle = {
            "js": '(document.querySelector(".settings-surface main").innerText || "")'
            ".trim().length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = (".settings-surface", '[data-slot="dialog-content"]'),
    ),
    # ── the sidebar ─────────────────────────────────────────────────
    Surface(
        id = "sidebar:expanded",
        group = "sidebar",
        title = "Sidebar, expanded",
        reach = (("wait", 200),),
        settle = {"visible": '[data-slot="sidebar-container"]'},
        restore = (),
        root = ROOT_SIDEBAR,
    ),
    Surface(
        id = "sidebar:collapsed",
        group = "sidebar",
        title = "Sidebar, collapsed to the rail",
        reach = (("click", 'button[aria-label="Close sidebar"]'), ("wait", 500)),
        # The rail is the same container in a different data-state, so presence proves
        # nothing; the reopen control appearing is what says the collapse happened.
        settle = {"visible": 'button[aria-label="Open sidebar"]'},
        restore = (("click_if", 'button[aria-label="Open sidebar"]'), ("wait", 400)) + HOME,
        root = ROOT_SIDEBAR,
    ),
    Surface(
        id = "sidebar:account-menu",
        group = "sidebar",
        title = "Account menu",
        reach = (("click", 'button[aria-label^="Unsloth account menu"]'), ("wait", 400)),
        settle = {"visible": ".app-user-menu"},
        restore = ESCAPE_OUT,
        root = (".app-user-menu",),
    ),
    Surface(
        id = "sidebar:thread-menu",
        group = "sidebar",
        title = "Chat row options menu",
        reach = (
            ("hover", '[class*="group/recent-item"]'),
            ("wait", 300),
            ("click", 'button[aria-label="Chat options"]'),
            ("wait", 400),
        ),
        settle = {"count_at_least": ['[role="menuitem"]', 3]},
        restore = ESCAPE_OUT,
        root = ('[role="menu"]',),
        conditional = "needs at least one chat in the sidebar",
    ),
    Surface(
        id = "sidebar:organize-menu",
        group = "sidebar",
        title = "Organize chats menu",
        reach = (
            ("hover", '[data-slot="sidebar-group-label"]:has(button[aria-label="Organize chats"])'),
            ("wait", 300),
            ("click", 'button[aria-label="Organize chats"]'),
            ("wait", 400),
        ),
        settle = {"count_at_least": ['[role="menuitemradio"]', 2]},
        restore = ESCAPE_OUT,
        root = ('[role="menu"]',),
    ),
    Surface(
        id = "sidebar:new-project",
        group = "sidebar",
        title = "New project dialog",
        reach = (
            ("hover", '[class*="group/projects-item"]'),
            ("wait", 300),
            ("click", 'button[aria-label="New project"]'),
            ("wait", 500),
        ),
        settle = {"visible": '[data-slot="dialog-content"]'},
        restore = ESCAPE_OUT + HOME,
        root = ('[data-slot="dialog-content"]',),
    ),
    Surface(
        id = "sidebar:search",
        group = "sidebar",
        title = "Chat search command dialog",
        reach = (("click", 'button[aria-label="Search"]'), ("wait", 500)),
        settle = {"visible": ".chat-search-surface"},
        restore = ESCAPE_OUT,
        root = (".chat-search-surface",),
    ),
    Surface(
        id = "sidebar:workflows",
        group = "sidebar",
        title = "Images workflow sub-rows",
        reach = (
            ("hover", '[class*="group/images-item"]'),
            ("wait", 300),
            ("click", 'button[aria-label="Show workflows"]'),
            ("wait", 400),
        ),
        settle = {"visible": 'button[aria-label="Hide workflows"]'},
        restore = (("click_if", 'button[aria-label="Hide workflows"]'), ("wait", 300)),
        root = ROOT_SIDEBAR,
    ),
    # ── the chat composer's own overlays ────────────────────────────
    Surface(
        id = "chat:model-picker",
        group = "chat",
        title = "Model picker",
        reach = (("click", ".unsloth-model-selector-trigger"), ("wait", 600)),
        # The rows, not the menu. The popover mounts before its list has been fetched, so a
        # settle on the container's presence digests an empty picker roughly half the time --
        # and an empty picker and a broken picker have the same digest.
        settle = {"count_at_least": [".unsloth-model-selector-menu button", 2]},
        restore = ESCAPE_OUT,
        root = (".unsloth-model-selector-menu",),
        also_in_film = True,
    ),
    Surface(
        id = "chat:model-picker-ondevice",
        group = "chat",
        title = "Model picker, On Device tab",
        reach = (
            ("click", ".unsloth-model-selector-trigger"),
            ("wait", 600),
            ("click", '.unsloth-model-selector-menu [role="tab"]:nth-of-type(2)'),
            ("wait", 500),
        ),
        settle = {"visible": ".unsloth-model-selector-menu"},
        restore = ESCAPE_OUT,
        root = (".unsloth-model-selector-menu",),
    ),
    Surface(
        id = "chat:plus-menu",
        group = "chat",
        title = "Tools and attachments menu",
        reach = (("click", 'button[aria-label="Tools and attachments"]'), ("wait", 400)),
        settle = {"count_at_least": ['[role="menuitem"]', 1]},
        restore = ESCAPE_OUT,
        root = ('[role="menu"]',),
        also_in_film = True,
    ),
    Surface(
        id = "chat:run-settings",
        group = "chat",
        title = "Run settings sheet",
        reach = (("click", 'button[aria-label="Open run settings"]'), ("wait", 600)),
        settle = {"visible": 'button[aria-label="Close run settings"]'},
        restore = (("click_if", 'button[aria-label="Close run settings"]'), ("wait", 400)) + HOME,
        # The panel, not a sheet. Despite the file being called chat-settings-sheet it renders
        # as a docked column inside the chat layout, so it is in NO overlay list and a root of
        # `[role="dialog"]` falls through to body -- which digested 301,402 characters of
        # whole page and called it the run settings panel.
        root = ('[data-slot="chat-settings-panel"]',),
    ),
    Surface(
        id = "chat:permission-menu",
        group = "chat",
        title = "Tool permission menu",
        reach = (("click", 'button[aria-label="Permission level for tool calls"]'), ("wait", 400)),
        settle = {"count_at_least": ['[role="menuitem"], [role="option"]', 2]},
        restore = ESCAPE_OUT,
        root = ('[role="menu"]', '[role="listbox"]'),
    ),
    Surface(
        id = "chat:system-prompt",
        group = "chat",
        title = "System prompt editor",
        # Through the run settings panel. The edit control is rendered at x=1458 on a 1440
        # viewport when the panel is closed -- present, sized, and off screen -- so clicking
        # it directly times out on an element every presence check reports as there.
        reach = (
            ("click", 'button[aria-label="Open run settings"]'),
            ("wait", 800),
            ("click", 'button[aria-label="Edit system prompt"]'),
            ("wait", 800),
        ),
        settle = {"visible": '[data-slot="dialog-content"]'},
        restore = ESCAPE_OUT + HOME,
        root = ('[data-slot="dialog-content"]',),
    ),
    Surface(
        id = "chat:temporary",
        group = "chat",
        title = "Temporary chat, on",
        reach = (("click", 'button[aria-label="Turn on temporary chat"]'), ("wait", 500)),
        settle = {"visible": 'button[aria-label="Turn off temporary chat"]'},
        restore = (("click_if", 'button[aria-label="Turn off temporary chat"]'), ("wait", 300))
        + HOME,
        root = ROOT_ROUTE,
    ),
    Surface(
        id = "chat:composer-filled",
        group = "chat",
        title = "Composer holding text",
        reach = (("fill", COMPOSER, "studiobench surface sweep"), ("wait", 400)),
        # The send control is what changes, and it is the same control the film's stop action
        # trips over: with text in the box the Stop button is replaced by Queue.
        # ONE EXPRESSION. The settle evaluator wraps the string in `return (...)`, so a
        # statement -- a `const` declaration, say -- raises SyntaxError at evaluation time,
        # and the surface is recorded as never settling for a reason that is about this file
        # rather than about the app.
        settle = {
            "js": "(document.querySelector("
            "'textarea[aria-label=\"Message input\"]') || {}).value ? true : false"
        },
        restore = (("fill", COMPOSER, ""), ("wait", 200)) + HOME,
        root = ROOT_ROUTE,
        also_in_film = True,
        volatile = GREETING_IS_RANDOM,
    ),
    # ── the hub ─────────────────────────────────────────────────────
    Surface(
        id = "hub:datasets",
        group = "hub",
        title = "Hub, datasets tab",
        reach = (
            ("goto", "/hub"),
            ("wait", 800),
            ("click", 'button[aria-label="Datasets"]'),
            ("wait", 800),
        ),
        # The URL flips the instant the tab is pressed, well before the list arrives. Settling
        # on the URL alone digested an empty table on one visit and a populated one on the
        # next, a 424-line difference that had nothing to do with any build.
        settle = {
            "js": 'location.search.includes("kind=datasets") && '
            '(document.body.innerText || "").includes("Downloads")'
        },
        restore = HOME,
        root = ROOT_ROUTE,
        conditional = "needs the catalogue, which needs network",
        volatile = HUB_IS_REMOTE,
    ),
    # The hub's two filter popovers are LISTBOXES, not menus. `[role="menuitem"]` counted zero on
    # a popover that was open and visible on screen, and zero from a wrong selector is the failure
    # mode this tool exists to refuse -- so both the settle and the root read `option`/`listbox`.
    Surface(
        id = "hub:format-filter",
        group = "hub",
        title = "Hub, format filter menu",
        reach = (
            ("goto", "/hub"),
            ("wait", 800),
            ("click", 'button[aria-label="Format filter"]'),
            ("wait", 400),
        ),
        settle = {"count_at_least": ['[role="option"]', 2]},
        restore = ESCAPE_OUT + HOME,
        root = ('[role="listbox"]',),
        conditional = "needs the catalogue, which needs network",
    ),
    Surface(
        id = "hub:sort-menu",
        group = "hub",
        title = "Hub, sort menu",
        reach = (
            ("goto", "/hub"),
            ("wait", 800),
            ("click", 'button[aria-label="Sort models"]'),
            ("wait", 400),
        ),
        settle = {"count_at_least": ['[role="option"]', 2]},
        restore = ESCAPE_OUT + HOME,
        root = ('[role="listbox"]',),
        conditional = "needs the catalogue, which needs network",
    ),
    Surface(
        id = "hub:hf-token",
        group = "hub",
        title = "Hub, Hugging Face token entry",
        reach = (
            ("goto", "/hub"),
            ("wait", 800),
            ("click", 'button[aria-label="Set Hugging Face token"]'),
            ("wait", 800),
        ),
        settle = {"visible": ".settings-surface"},
        restore = ESCAPE_OUT + HOME,
        root = (".settings-surface", '[data-slot="dialog-content"]'),
        notes = "the hub's token control does not open a popover of its own: it opens the "
        "settings dialog. Registered against what it actually does, so a change that "
        "made it open nothing is a failure here rather than an unnoticed no-op",
    ),
    Surface(
        id = "hub:compact-layout",
        group = "hub",
        title = "Hub, compact layout",
        reach = (
            ("goto", "/hub"),
            ("wait", 800),
            ("click", 'button[aria-label="Compact"]'),
            ("wait", 600),
        ),
        settle = {"js": 'location.pathname === "/hub"'},
        restore = HOME,
        root = ROOT_ROUTE,
        conditional = "needs the catalogue, which needs network",
        volatile = HUB_IS_REMOTE,
    ),
    # ── the media pages ─────────────────────────────────────────────
    Surface(
        id = "images:presets",
        group = "media",
        title = "Images, presets menu",
        reach = (
            ("goto", "/images"),
            ("wait", 800),
            ("click", 'button[aria-label="Manage image generation presets"]'),
            ("wait", 500),
        ),
        settle = {
            "js": 'document.querySelectorAll(\'[role="menu"], '
            "[data-radix-popper-content-wrapper]').length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = ('[role="menu"]', "[data-radix-popper-content-wrapper]"),
    ),
    Surface(
        id = "images:model-picker",
        group = "media",
        title = "Images, model picker",
        reach = (
            ("goto", "/images"),
            ("wait", 800),
            ("click", ".unsloth-model-selector-trigger"),
            ("wait", 600),
        ),
        settle = {"visible": ".unsloth-model-selector-menu"},
        restore = ESCAPE_OUT + HOME,
        root = (".unsloth-model-selector-menu",),
    ),
    Surface(
        id = "video:presets",
        group = "media",
        title = "Video, presets menu",
        reach = (
            ("goto", "/video"),
            ("wait", 800),
            ("click", 'button[aria-label="Manage video generation presets"]'),
            ("wait", 500),
        ),
        settle = {
            "js": 'document.querySelectorAll(\'[role="menu"], '
            "[data-radix-popper-content-wrapper]').length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = ('[role="menu"]', "[data-radix-popper-content-wrapper]"),
        conditional = "a host without video support renders the gate instead of the page",
    ),
    # ── train ───────────────────────────────────────────────────────
    Surface(
        id = "train:model-picker",
        group = "train",
        title = "Train, model picker",
        reach = (
            ("goto", "/studio"),
            ("wait", 1200),
            ("click", 'button[aria-label="Model: Select model"]'),
            ("wait", 600),
        ),
        settle = {
            "js": "document.querySelectorAll('.unsloth-model-selector-menu, "
            '[role="menu"], [data-radix-popper-content-wrapper]\').length > 0'
        },
        restore = ESCAPE_OUT + HOME,
        root = (".unsloth-model-selector-menu", "[data-radix-popper-content-wrapper]"),
        conditional = "a chat-only host redirects /studio to /chat",
    ),
    Surface(
        id = "train:method",
        group = "train",
        title = "Train, method selector",
        reach = (
            ("goto", "/studio"),
            ("wait", 1200),
            ("click", 'button[aria-label^="Method:"]'),
            ("wait", 500),
        ),
        settle = {
            "js": 'document.querySelectorAll(\'[role="menu"], [role="listbox"], '
            "[data-radix-popper-content-wrapper]').length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = ('[role="menu"]', '[role="listbox"]', "[data-radix-popper-content-wrapper]"),
        conditional = "a chat-only host redirects /studio to /chat",
    ),
    Surface(
        id = "train:dataset",
        group = "train",
        title = "Train, dataset picker",
        reach = (
            ("goto", "/studio"),
            ("wait", 1200),
            ("click", 'button[aria-label^="Dataset:"]'),
            ("wait", 600),
        ),
        settle = {
            "js": 'document.querySelectorAll(\'[role="menu"], [role="listbox"], '
            "[data-radix-popper-content-wrapper]').length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = ('[role="menu"]', '[role="listbox"]', "[data-radix-popper-content-wrapper]"),
        conditional = "a chat-only host redirects /studio to /chat",
    ),
    Surface(
        id = "train:image-training",
        group = "train",
        title = "Image LoRA training",
        reach = (
            ("goto", "/studio"),
            ("wait", 1200),
            ("click", 'button[aria-label="Image training"]'),
            ("wait", 1200),
        ),
        # It NAVIGATES. The control sits on the Train page but lands on /images with the
        # training panel selected, so a settle written against /studio waits out its timeout
        # on a page that arrived correctly.
        settle = {
            "js": 'location.pathname === "/images" && '
            '(document.body.innerText || "").includes("Train a LoRA")'
        },
        restore = HOME,
        root = ROOT_ROUTE,
        conditional = "a chat-only host redirects /studio to /chat",
        volatile = UNEXPLAINED_DRIFT,
    ),
    # ── the api monitor ─────────────────────────────────────────────
    Surface(
        id = "api:status-filter",
        group = "api",
        title = "API monitor, status filter",
        reach = (
            ("goto", "/api-monitor"),
            ("wait", 800),
            ("click", 'button[aria-label="Filter by status"]'),
            ("wait", 400),
        ),
        settle = {
            "js": 'document.querySelectorAll(\'[role="menu"], [role="listbox"], '
            "[data-radix-popper-content-wrapper]').length > 0"
        },
        restore = ESCAPE_OUT + HOME,
        root = ('[role="menu"]', '[role="listbox"]', "[data-radix-popper-content-wrapper]"),
    ),
]


#: Surfaces that exist and are NOT swept, each with the mechanism that puts them out of reach.
#:
#: This list is the honest half of the coverage number. A manifest that reports "38 of 38 reached"
#: while eleven surfaces were never registered is a worse artefact than one that reports 38 of 38
#: and names the eleven, because the first invites the reader to believe the app is fully covered.
KNOWN_UNCOVERED: tuple[dict, ...] = (
    {
        "id": "route:login",
        "title": "Login page",
        "reason": "requireGuest() redirects an authenticated session to /chat. Reaching it means "
        "clearing the auth token, which ends the session the sweep and the film share, "
        "so it is out of scope for a sweep that runs inside a measurement run",
    },
    {
        "id": "route:change-password",
        "title": "Change password page",
        "reason": "requirePasswordChangeFlow() redirects unless the account is flagged "
        "must_change_password, and studiobench's own authenticate() clears that flag "
        "before the browser ever starts",
    },
    {
        "id": "route:data-recipe-editor",
        "title": "Data recipe editor (/data-recipes/$recipeId)",
        "reason": "needs a saved recipe. Creating one writes to the installation the run is "
        "measuring, and a sweep that mutates the app under test is not a parity check",
    },
    {
        "id": "dialog:destructive-confirms",
        "title": "Delete / archive / shutdown confirmations",
        "reason": "reaching them means arming a destructive action against the installation under "
        "test. The delete confirmation IS reached by the film's delete_message action, on "
        "a thread the film created for the purpose",
    },
    {
        "id": "dialog:promise-driven",
        "title": "HF token warning, remote code consent, "
        "transformers upgrade, stop running chats",
        "reason": "root-mounted and opened by a promise that only resolves when a download, a model "
        "load or a model swap actually needs consent. There is no trigger to click",
    },
    {
        "id": "picker:connected-tab",
        "title": "Model picker, Connected tab",
        "reason": "rendered only when externalModels.length > 0. The A/B registers a pacer provider, "
        "so it appears in a measurement run and not in a bare sweep; it is left out rather "
        "than made to depend on which of the two is running",
    },
    {
        "id": "picker:model-config-page",
        "title": "Model picker, inference settings page",
        "reason": "entered from a downloaded model's row action, so it needs a model cached on the "
        "machine. A sweep that downloads one is not a sweep",
    },
    {
        "id": "chat:message-surfaces",
        "title": "Message action bar, reasoning pane, tool blocks, "
        "response details sheet, attachment preview",
        "reason": "all need a thread with an assistant reply, which is exactly what the film drives. "
        "Covered there, at eighteen points, rather than duplicated here",
    },
    {
        "id": "recipe-studio:*",
        "title": "Recipe studio block sheet and its four dialogs",
        "reason": "reached from inside a recipe being edited, which needs a saved recipe",
    },
    {
        "id": "tour:guided",
        "title": "Guided tour overlay",
        "reason": "driven by a TOUR_OPEN_EVENT and a multi-step walkthrough that navigates on its "
        "own. Sweeping it would leave the app in a state the next surface cannot start "
        "from, which is the one thing a sweep may not do",
    },
    {
        "id": "monitor:floating",
        "title": "Floating resource monitor and API monitor overlays",
        "reason": "both open themselves on live traffic or a settings toggle and then poll, so their "
        "content changes on a timer. A digest of a surface that repaints on its own "
        "reports a difference between two runs of the same build",
    },
)


# ── accessors ───────────────────────────────────────────────────────


def surfaces() -> list[Surface]:
    return list(_SURFACES)


def surface_ids() -> list[str]:
    return [s.id for s in _SURFACES]


def get_surface(surface_id: str) -> Optional[Surface]:
    for s in _SURFACES:
        if s.id == surface_id:
            return s
    return None


def groups() -> dict[str, list[Surface]]:
    out: dict[str, list[Surface]] = {}
    for s in _SURFACES:
        out.setdefault(s.group, []).append(s)
    return out


class RegistryError(ValueError):
    pass


VERBS = {"goto": 1, "click": 1, "click_if": 1, "hover": 1, "press": 1, "fill": 2, "wait": 1}


def validate_registry(entries: Optional[list[Surface]] = None) -> None:
    """Fail loudly on a registry that cannot be executed.

    Run by the unit tests AND by the sweep before it opens a browser. A malformed entry that is
    only discovered halfway through a sweep costs the surfaces after it, and the run reports them
    as unreached for a reason that has nothing to do with the app.
    """
    entries = surfaces() if entries is None else entries
    seen: set[str] = set()
    for s in entries:
        if s.id in seen:
            raise RegistryError(f"duplicate surface id {s.id!r}")
        seen.add(s.id)
        if not s.id or not s.group or not s.title:
            raise RegistryError(f"surface {s.id!r} is missing an id, group or title")
        if not s.root:
            raise RegistryError(f"surface {s.id!r} has no digest root")
        # `reach` may be empty only for a surface that IS the known state; `restore` may be empty
        # only when reaching it changed nothing. Both are declared, never inferred.
        for name, steps in (("reach", s.reach), ("restore", s.restore)):
            for step in steps:
                if not step:
                    raise RegistryError(f"surface {s.id!r} has an empty {name} step")
                verb = step[0]
                if verb not in VERBS:
                    raise RegistryError(
                        f"surface {s.id!r} {name} step uses unknown verb {verb!r}; "
                        f"known verbs are {sorted(VERBS)}"
                    )
                if len(step) - 1 != VERBS[verb]:
                    raise RegistryError(
                        f"surface {s.id!r} {name} step {verb!r} takes {VERBS[verb]} argument(s), "
                        f"got {len(step) - 1}"
                    )
        if s.settle is not None and not isinstance(s.settle, dict):
            raise RegistryError(f"surface {s.id!r} has a non-dict settle condition")
    uncovered = {u["id"] for u in KNOWN_UNCOVERED}
    clash = uncovered & seen
    if clash:
        raise RegistryError(
            f"{sorted(clash)} are both registered and listed as known-uncovered; a surface "
            "cannot be swept and out of reach at the same time"
        )
    for entry in KNOWN_UNCOVERED:
        for key in ("id", "title", "reason"):
            if not entry.get(key):
                raise RegistryError(f"known-uncovered entry {entry!r} is missing {key!r}")


__all__ = [
    "Surface",
    "surfaces",
    "surface_ids",
    "get_surface",
    "groups",
    "validate_registry",
    "RegistryError",
    "KNOWN_UNCOVERED",
    "KNOWN_STATE_PATH",
    "VERBS",
]
