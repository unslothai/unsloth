// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// use-sidebar-pin reads the lg media query when it is imported. #11660 made that read
// unconditional, and every test whose import graph reaches the window titlebar supplies a
// partial `window` with no matchMedia, so the import threw and took the whole file down
// (training-dataset-source-transitions.test.ts, on main).

import assert from "node:assert/strict";
import test from "node:test";

const storage = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};

function stubWindow(matchMedia?: (query: string) => unknown) {
  const listeners: string[] = [];
  const target: Record<string, unknown> = {
    addEventListener: (type: string) => listeners.push(type),
    removeEventListener: () => undefined,
    localStorage: storage,
  };
  if (matchMedia) target.matchMedia = matchMedia;
  Object.assign(globalThis, { window: target });
  return listeners;
}

let loads = 0;
async function load() {
  loads += 1;
  return import(`../src/hooks/use-sidebar-pin.ts?load=${loads}`);
}

test("the pin module imports, holds and releases where there is no matchMedia", async () => {
  stubWindow();
  const pin = await load();
  pin.holdSidebarPinned();
  // Release re-derives the width default, the second place the query is read.
  pin.releaseSidebarPinned();
});

test("with matchMedia the width default still comes from the lg query", async () => {
  const queries: string[] = [];
  stubWindow((query) => {
    queries.push(query);
    return {
      matches: false,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    };
  });
  const pin = await load();
  assert.deepEqual(queries, ["(min-width: 1024px)"]);
  pin.holdSidebarPinned();
  pin.releaseSidebarPinned();
  assert.deepEqual(queries, ["(min-width: 1024px)", "(min-width: 1024px)"]);
});
