// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// onScroll never fires for a list that is not scrolling, so the bottom fade is
// only re-measured when this effect re-runs. Every input that can add or remove
// a row has to be in its deps, and a missing one is invisible: the fade just
// stays wrong until an unrelated scroll or resize.

test("the bottom fade re-measures on every row-count input", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const effect =
    /Recompute bottom-fade on mount[\s\S]*?\}, \[([\s\S]*?)\n  \]\);/.exec(
      source,
    );
  assert.ok(effect, "could not find the bottom-fade effect");
  const deps = effect[1];

  for (const dep of [
    // Each list that renders rows into the scroller.
    "recentChatItems.length",
    "pinnedChatItems.length",
    "projectChatRowCount",
    "visibleProjectRecords.length",
    "runItems.length",
    // Disclosures, which hide and reveal whole sections.
    "chatOpen",
    "runsOpen",
    "pinnedOpen",
    "projectsOpen",
    // Regrouping, which can empty or fill the Projects section outright.
    "organizeBy",
  ]) {
    assert.ok(
      deps.includes(dep),
      `${dep} is missing from the bottom-fade deps, so the fade goes stale when it changes`,
    );
  }
});
