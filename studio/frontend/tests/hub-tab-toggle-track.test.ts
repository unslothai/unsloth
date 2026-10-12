// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The segmented control is a wash on whatever is behind it, so its distance
// from the selected pill is a property of the surface, not of the control. On
// the page the wash lands 21 levels below --accent; on an elevated panel the
// same wash lands 7 below and the two halves stop reading apart. These pin the
// panel case darkening instead, a page-coloured pane inside a panel getting the
// page track back, a raised card in that pane sinking again, and the pill's
// hover on borrowed-pill buttons.

const HUB_CSS = readSrc("features/hub/hub.css");
const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

const PANELS =
  ':is(.menu-soft-surface, .menu-soft-surface-up, [data-slot="dialog-content"])';
const esc = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const PAGE_TRACK = /html\.dark \{([^}]*--hub-tab-track[^}]*)\}/;
const RAISED = '.bg-background[class*="dark:bg-"]';
const PANEL_TRACK = new RegExp(
  `html\\.dark ${esc(PANELS)},\\s*html\\.dark ${esc(PANELS)} ${esc(RAISED)} \\{([^}]*)\\}`,
);
const PANE_TRACK = new RegExp(
  `html\\.dark ${esc(PANELS)} ${esc('.bg-background:not([class*="dark:bg-"])')} \\{([^}]*)\\}`,
);

test("the track reads its colour from the nearest surface", () => {
  const rule = HUB_CSS.match(/html\.dark \.hub-tab-toggle \{([^}]*)\}/);
  assert.ok(rule);
  assert.match(rule[1] ?? "", /background-color: var\(--hub-tab-track\)/);
});

test("a track on an elevated panel sinks toward the page, not off the panel", () => {
  const rule = HUB_CSS.match(PANEL_TRACK);
  assert.ok(rule, "panel-scoped track is missing");
  // Mixing --foreground here is what put the track 7 levels off the pill.
  assert.match(rule[1] ?? "", /--hub-tab-track: [^;]*var\(--background\)/);
  assert.doesNotMatch(rule[1] ?? "", /var\(--foreground\)/);
});

test("the page track is left on its own wash", () => {
  const base = HUB_CSS.match(PAGE_TRACK);
  assert.ok(base);
  assert.match(base[1] ?? "", /var\(--foreground\) calc\(6\.5%/);
});

test("the panel track stops just under the panel rather than far below it", () => {
  // 27% of #181818 over #272727 is #232323.
  assert.match(
    HUB_CSS.match(PANEL_TRACK)?.[1] ?? "",
    /var\(--background\) 27%/,
  );
});

test("a page-coloured pane inside a panel gets the page track back", () => {
  // The Settings content pane is bg-background inside the dialog. Sunk toward
  // the page there, the track matched the pane and the control lost its track.
  const pane = HUB_CSS.match(PANE_TRACK);
  assert.ok(pane, "page-coloured panes inside panels still sink the track");
  const page = HUB_CSS.match(PAGE_TRACK);
  assert.equal(
    pane[1]?.trim(),
    page?.[1]?.trim(),
    "pane and page tracks differ",
  );
  // Later in the file, so it beats the panel rule on a tie.
  assert.ok(HUB_CSS.search(PANE_TRACK) > HUB_CSS.search(PANEL_TRACK));
  const settings = readSrc("features/settings/settings-dialog.tsx");
  assert.match(settings, /<main className="[^"]*\bbg-background\b/);
});

test("a raised card inside a page-coloured pane keeps the panel track", () => {
  // Profile's StatsCard is bg-background washed 6% white in dark. On the page
  // track its pill sat 1.14:1 above the track, against 1.42:1 sunk.
  assert.ok(HUB_CSS.match(PANEL_TRACK), "raised cards lost the sunk track");
  const card = readSrc("features/profile/components/stats/stat-primitives.tsx");
  assert.match(card, /\bbg-background dark:border-transparent dark:bg-\[/);
});

test("a selected segment shows no hover, being the tab you are already on", () => {
  // Grouped with the resting rule, so the two cannot drift apart.
  assert.match(
    HUB_CSS,
    /html\.dark \.hub-tab-toggle-pill,\s*html\.dark \.hub-tab-toggle-pill:hover \{[^}]*background-color: var\(--accent\)/,
  );
});

test("no tab pins its hover to a colour that is only right in one theme", () => {
  const tabs = readSrc(
    "features/model-picker/components/model-selector/pill-tabs.tsx",
  );
  // hover:!bg-[var(--background)] carries no mode variant, so in dark it
  // painted the page colour over the pill and blacked the tab out.
  assert.doesNotMatch(tabs, /hover:!bg-\[var\(--background\)\]/);
});

test("an option menu rests at its trigger's tone, not the panel's", () => {
  const menu = HUB_CSS.match(/html\.dark \.hub-menu-instant \{([^}]*)\}/);
  assert.ok(menu);
  // The same 60% .field-soft rests at, made opaque for a floating surface.
  assert.match(menu[1] ?? "", /var\(--accent\) 60%, var\(--popover\)/);
});

test("a highlighted row still lifts clear of the lighter menu", () => {
  const row = HUB_CSS.match(
    /html\.dark \.hub-menu-instant \[data-slot="select-item"\]:focus,[\s\S]*?\{([^}]*)\}/,
  );
  assert.ok(row);
  // Bare --accent would sit 8 levels above the menu, too close to pick out.
  assert.match(row[1] ?? "", /var\(--foreground\) 6%, var\(--accent\)/);
});

test("a white pill in light mode stays put, having nowhere lighter to go", () => {
  const rule = HUB_CSS.match(
    /html:not\(\.dark\) \.hub-tab-toggle-pill\[role="tab"\]:hover \{([^}]*)\}/,
  );
  assert.ok(rule, "the light tab hover is missing");
  assert.match(rule[1] ?? "", /background-color:\s*var\(--background\)/);
});

test("a button borrowing the pill look gets the hover the pill pins away", () => {
  const hover = HUB_CSS.match(
    /html\.dark \.hub-tab-toggle-pill\.hub-pill-action:hover \{([^}]*)\}/,
  );
  assert.ok(hover, "hub-pill-action hover rule is missing");
  // Lighter than rest: --accent carrying a little --foreground.
  assert.match(hover[1] ?? "", /var\(--foreground\) 8%, var\(--accent\)/);
});

test("pressing a borrowed pill returns it to the selection colour", () => {
  const hover = HUB_CSS.indexOf(
    "html.dark .hub-tab-toggle-pill.hub-pill-action:hover",
  );
  const active = HUB_CSS.indexOf(
    "html.dark .hub-tab-toggle-pill.hub-pill-action:active",
  );
  assert.ok(active > hover, ":active must follow :hover to win the cascade");
  const rule = HUB_CSS.slice(active).match(/\{([^}]*)\}/);
  assert.match(rule?.[1] ?? "", /background-color:\s*var\(--accent\)/);
});

test("Search Hub is a borrowed pill, so it opts into the hover", () => {
  const button = PICKERS.match(
    /aria-label="Search more models on the Hub"[\s\S]{0,200}?className="([^"]*)"/,
  );
  assert.ok(button, "Search Hub button not found");
  assert.match(button[1] ?? "", /\bhub-tab-toggle-pill\b/);
  assert.match(button[1] ?? "", /\bhub-pill-action\b/);
});
