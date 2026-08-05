// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The pinned top rows run an action rather than open a page, so neither may
// mark itself active: nav rows paint one pill for both states, and an active
// action row therefore sits there looking permanently hovered.

async function sidebarSource(): Promise<string> {
  return readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
}

/** The props of the NavItem carrying `label`, from its tag to the next one. */
function navItemFor(source: string, label: string): string {
  const rows = source.split("<NavItem").slice(1);
  const row = rows.find((chunk) => chunk.includes(label));
  assert.ok(row, `no NavItem renders ${label}`);
  return row;
}

test("New chat never marks itself active", async () => {
  const row = navItemFor(await sidebarSource(), "shell.navigation.newChat");
  assert.match(row, /active=\{false\}/);
});

test("Search never marks itself active either", async () => {
  const row = navItemFor(
    await sidebarSource(),
    'label={t("shell.navigation.search")}',
  );
  assert.match(row, /active=\{false\}/);
});

test("a nav row paints the same pill when active and when hovered", async () => {
  // The reason the rows above pass false. If active ever gets its own
  // background, that reason is gone and this can be revisited.
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  const rule = /([^}]*)\{\s*background-color: var\(--nav-surface-hover\)/.exec(
    css,
  );
  assert.ok(rule, "no rule paints a nav row with --nav-surface-hover");
  assert.match(rule[1], /\.sidebar-nav-btn:hover/);
  assert.match(rule[1], /\.sidebar-nav-btn\[data-active="true"\]/);
});
