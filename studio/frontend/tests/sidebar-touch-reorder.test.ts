// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Touch never fires dragstart, so the row menu is the only way to reorder a
// list there. A menu behind a trigger without sidebar-touch-reveal is inert on
// coarse pointers, which silently takes manual ordering away from touch users.

async function sidebarSource(): Promise<string> {
  return readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
}

/** The className string of the button carrying `label`. */
function actionClassFor(source: string, label: string): string {
  const match = new RegExp(
    `aria-label="${label}"[\\s\\S]{0,200}?className="([^"]*)"`,
  ).exec(source);
  assert.ok(match, `no button renders aria-label="${label}"`);
  return match[1];
}

test("only sidebar-touch-reveal actions work on a coarse pointer", async () => {
  // The rule the rest of this file depends on.
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  const coarse = /@media \(pointer: coarse\) \{([\s\S]*?)\n\t\}/.exec(css);
  assert.ok(coarse, "no coarse-pointer block in index.css");
  assert.match(coarse[1], /\.sidebar-row-action\.sidebar-touch-reveal/);
});

test("rows that reorder can open their menu on touch", async () => {
  const source = await sidebarSource();

  // Chat rows, both variants.
  const chatActions = source.match(/"sidebar-row-action[^"]*"/g) ?? [];
  const reorderRowActions = chatActions.filter((cls) =>
    /group-hover\/(recent-item|project-chat-item)/.test(cls),
  );
  assert.ok(reorderRowActions.length > 0, "no chat or project row actions");
  for (const cls of reorderRowActions) {
    assert.match(cls, /sidebar-touch-reveal/);
  }

  // The project folder menu holds the folder reorder controls.
  assert.match(
    actionClassFor(source, "Project options"),
    /sidebar-touch-reveal/,
  );
});

test("a folder row reserves the room its touch actions take", async () => {
  // Revealed without reserved padding, the buttons sit on top of the name.
  const source = await sidebarSource();
  const row = /className="(sidebar-nav-btn h-\[33px\] rounded-full gap-\[8\.5px\][^"]*group-hover\/recent-item:pr-16[^"]*)"/.exec(
    source,
  );
  assert.ok(row, "could not find the project folder row");
  assert.match(row[1], /\[@media\(pointer:coarse\)\]:pr-16/);
});
