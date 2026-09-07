// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Overlay UX on top of the resumable Downloads list: keep a Downloads entry
// when the list is empty, start collapsed, and expand when a transfer starts.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(relative: string): string {
  return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");
}

const PANEL = read(
  "../src/features/hub/download-manager/download-manager-panel.tsx",
);

test("Downloads stays mounted even when the list is empty", () => {
  // The overlay used to return null with no jobs, so after a failure lingered
  // away there was no entry left to open. The empty state is the entry.
  assert.doesNotMatch(
    PANEL,
    /if \(!enabled \|\| jobKeys.length === 0\) return null/,
  );
  assert.match(PANEL, /if \(!enabled\) return null/);
  assert.match(PANEL, /No downloads yet/);
  assert.match(PANEL, /Open Model hub/);
});

test("Downloads starts collapsed and expands when a transfer starts", () => {
  assert.match(PANEL, /useState\(true\)/);
  assert.match(PANEL, /previousActiveCount/);
  assert.match(
    PANEL,
    /if \(activeCount > 0 && previousActiveCount\.current === 0\)/,
  );
});
