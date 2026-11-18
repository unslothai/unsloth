// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// #18013: a sidebar peeking over a full-view browser page opened behind the native page.

async function source(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

test("a held-out sidebar marks itself as covering the native page", async () => {
  const sidebar = await source("../src/components/ui/sidebar.tsx");
  const container = sidebar.split('data-slot="sidebar-container"')[1]?.split(">")[0] ?? "";
  assert.match(container, /data-native-cover=\{heldOut \? "" : undefined\}/);
});

test("native covers hide the page and are tracked while they slide in", async () => {
  const view = await source("../src/features/browser/native-view.ts");
  for (const name of ["OVERLAY_SELECTOR", "BLOCKING_SELECTOR"]) {
    const selector = view.split(`const ${name} =`)[1]?.split(";")[0] ?? "";
    assert.match(selector, /\[data-native-cover\]/, `${name} must include covers`);
  }
  const moved = view.split("const moved = (event: Event) => {")[1]?.split("};")[0] ?? "";
  assert.match(moved, /\[data-native-cover\]/, "a cover's slide-in must re-sync every frame");
});
