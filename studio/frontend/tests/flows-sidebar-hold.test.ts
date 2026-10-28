// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Flows' editor unpins the app sidebar while it is open; the user's saved pin must survive it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("holding the sidebar unpinned and releasing it never writes the saved pin", async () => {
  const writes: string[] = [];
  Object.assign(globalThis, {
    window: {
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      localStorage: {
        getItem: () => "true",
        setItem: (key: string, value: string) => writes.push(`${key}=${value}`),
        removeItem: (key: string) => writes.push(`-${key}`),
      },
    },
  });
  const tag = "flows-hold";
  const pin = await import(`../src/hooks/use-sidebar-pin.ts?${tag}`);
  pin.holdSidebarUnpinned();
  pin.releaseSidebarPinned();
  assert.deepEqual(writes, []);
});

test("the Flows editor holds the sidebar while mounted and releases it on leaving", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/flows-editor-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /useEffect\(\(\) => \{\s*holdSidebarUnpinned\(\);\s*return releaseSidebarPinned;\s*\}, \[\]\);/,
  );
});
