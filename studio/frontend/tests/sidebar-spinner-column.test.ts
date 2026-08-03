// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// Both spinners are ml-auto, so each one sits at its row's padding-right plus
// its own margin-right. The two rows carry different padding, so the margins
// have to make up the difference or the column visibly steps.
const TAILWIND_UNIT = 4;

function inset(classes: string, prefix: string): number {
  const m = new RegExp(`(?:^| )${prefix}-([0-9.]+)(?: |$)`).exec(classes);
  return m ? Number(m[1]) * TAILWIND_UNIT : 0;
}

function grab(source: string, pattern: RegExp, what: string): string {
  const m = pattern.exec(source);
  assert.ok(m, `could not find ${what} in app-sidebar.tsx`);
  return m[1];
}

test("nav and Recents spinners land on one trailing column", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );

  const navRow = grab(
    source,
    /className="(sidebar-nav-btn h-\[33px\] rounded-full[^"]*)"/,
    "NavItem row",
  );
  const navSpinner = grab(
    source,
    /<Spinner className="(ml-auto[^"]*group-data-\[collapsible=icon\]:hidden)"/,
    "NavItem spinner",
  );
  const chatRow = grab(
    source,
    /"(sidebar-nav-btn h-\[[0-9]+px\] cursor-pointer rounded-full[^"]*)"/,
    "Recents chat row",
  );
  const chatSpinner = grab(
    source,
    /data-testid="chat-row-spinner"[\s\S]{0,200}?className="(ml-auto[^"]*)"/,
    "Recents chat spinner",
  );

  const nav = inset(navRow, "pr") + inset(navSpinner, "mr");
  const chat = inset(chatRow, "pr") + inset(chatSpinner, "mr");

  assert.equal(nav, chat, `nav spinner sits ${nav}px in, chat spinner ${chat}px`);
  assert.equal(nav, 16);
});

// The kebab overlays the row's right edge, so a spinner row must pad past it.
test("a generating Recents row clears the kebab on hover", async () => {
  const [source, css] = await Promise.all([
    readFile(new URL("../src/components/app-sidebar.tsx", import.meta.url), "utf8"),
    readFile(new URL("../src/index.css", import.meta.url), "utf8"),
  ]);

  const kebabInset =
    inset(grab(css, /\.sidebar-row-action \{\s*@apply ([^;]*);/, "row action"), "pr") +
    inset(grab(css, /\.sidebar-row-action-glyph \{\s*@apply ([^;]*);/, "action glyph"), "size");
  assert.equal(kebabInset, 30);

  const generating = grab(
    source,
    // Anchor on the ternary so only the branch under test can match.
    /isGenerating\s*\?[\s\S]{0,300}?"(group-hover\/recent-item:pr-[^"]*)"/,
    "the isGenerating padding branch",
  );
  // hover, menu-open and coarse-pointer all reveal the kebab, so all must clear it
  const pads = [...generating.matchAll(/:pr-([0-9.]+)(?: |$)/g)].map(
    (m) => Number(m[1]) * TAILWIND_UNIT,
  );
  assert.ok(pads.length >= 3, `expected hover, open and coarse paddings, got ${pads.length}`);
  for (const pad of pads) {
    assert.ok(pad >= kebabInset, `${pad}px padding, needs ${kebabInset}px to clear the kebab`);
  }

  // focus-visible reveals the kebab without hover, so every spinner row reserves room there too.
  const focusPads = [...source.matchAll(/:focus-visible\]\/[a-z-]+:pr-([0-9.]+)/g)].map(
    (m) => Number(m[1]) * TAILWIND_UNIT,
  );
  assert.equal(focusPads.length, 3, `expected 3 focus paddings, got ${focusPads.length}`);
  for (const pad of focusPads) {
    assert.ok(pad >= kebabInset, `${pad}px focus padding, needs ${kebabInset}px`);
  }
});
