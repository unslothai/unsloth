// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The segmented control is a wash on whatever is behind it, so its distance
// from the selected pill is a property of the surface, not of the control. On
// the page the wash lands 21 levels below --accent; on an elevated panel the
// same wash lands 7 below and the two halves stop reading apart. These pin the
// panel case darkening instead, and the pill's hover on borrowed-pill buttons.

const HUB_CSS = readSrc("features/hub/hub.css");
const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

const MENU_TRACK =
  /html\.dark \.menu-soft-surface \.hub-tab-toggle,[\s\S]*?\{([^}]*)\}/;

test("a track on an elevated panel sinks toward the page, not off the panel", () => {
  const rule = HUB_CSS.match(MENU_TRACK);
  assert.ok(rule, "menu-scoped .hub-tab-toggle rule is missing");
  // Mixing --foreground here is what put the track 7 levels off the pill.
  assert.match(rule[1] ?? "", /var\(--background\)/);
  assert.doesNotMatch(rule[1] ?? "", /var\(--foreground\)/);
});

test("the panel track overrides the page one rather than racing it", () => {
  const base = HUB_CSS.indexOf("html.dark .hub-tab-toggle {");
  const scoped = HUB_CSS.search(MENU_TRACK);
  assert.ok(base >= 0 && scoped >= 0);
  // Higher specificity already wins; source order keeps it obvious.
  assert.ok(scoped > base, "the scoped rule must follow the page one");
});

test("the page track is left on its own wash", () => {
  const base = HUB_CSS.match(/html\.dark \.hub-tab-toggle \{([^}]*)\}/);
  assert.ok(base);
  assert.match(base[1] ?? "", /var\(--foreground\)/);
});

test("the panel track stops just under the panel rather than far below it", () => {
  const rule = HUB_CSS.match(MENU_TRACK);
  // 27% of #181818 over #272727 is #232323.
  assert.match(rule?.[1] ?? "", /var\(--background\) 27%/);
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
