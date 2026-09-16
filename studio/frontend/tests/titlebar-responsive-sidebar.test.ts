// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The mobile sidebar is a Sheet, so it holds no layout width, but the corner was
// still drawn at the desktop width (unslothai/unsloth#8600). The separator under
// it is not sidebar geometry and has to survive.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = () =>
  readFile(
    new URL("../src/components/tauri/window-titlebar.tsx", import.meta.url),
    "utf8",
  );

const DECORATION_SLOT = 'data-slot="window-titlebar-decoration"';

async function decorationBlock(): Promise<{ gate: string; body: string }> {
  const titlebar = await source();
  const slot = titlebar.indexOf(DECORATION_SLOT);
  assert.notEqual(slot, -1, "the decoration slot is gone");
  const gateStart = titlebar.lastIndexOf("{", slot);
  const open = titlebar.indexOf("<div", gateStart);
  const end = titlebar.indexOf("<header", slot);
  assert.notEqual(end, -1);
  return {
    gate: titlebar.slice(gateStart, open),
    body: titlebar.slice(open, end),
  };
}

test("the desktop sidebar surface is the mobile-aware flag", async () => {
  const titlebar = await source();
  assert.match(
    titlebar,
    /const showDesktopSidebarSurface =[^;]*\bisMobile\b[^;]*;/,
    "showDesktopSidebarSurface has to fold isMobile into showSidebarSurface",
  );
});

test("both pinned corners are drawn only on the desktop surface", async () => {
  const { body } = await decorationBlock();
  const corners = [...body.matchAll(/\{([^{}]*?)&&\s*\(\s*<div/g)].map(
    (match) => match[1],
  );
  assert.equal(corners.length, 2, "expected the two pinned corner squares");
  for (const gate of corners) {
    assert.match(gate, /showDesktopSidebarSurface/);
    assert.match(gate, /pinned/);
  }
});

test("the titlebar separator survives at mobile widths", async () => {
  const { gate, body } = await decorationBlock();
  // Gating the wrapper on the desktop surface would take the separator with it.
  assert.doesNotMatch(gate, /showDesktopSidebarSurface/);
  assert.match(gate, /showSidebarSurface/);
  assert.match(body, /h-px bg-sidebar-border/);
});

test("mobile never offsets geometry by the desktop sidebar width", async () => {
  const titlebar = await source();
  for (const pattern of [
    /const sidebarWidth = showDesktopSidebarSurface/,
    /const contentBorderLeft =\s*showDesktopSidebarSurface && pinned/,
  ]) {
    assert.match(titlebar, pattern);
  }
  // 7rem on mobile too: Navbar owns the toggle there and a spacer holds the slot.
  assert.match(
    titlebar,
    /const titlebarNavigationWidth =\s*showSidebarSurface && \(isMobile \|\| !pinned\) \? "7rem" : sidebarWidth;/,
  );
});
