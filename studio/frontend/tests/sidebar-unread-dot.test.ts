// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dot a finished reply leaves on a sidebar row. Read from the source: the
// node suite has no DOM to mount the sidebar in.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const SIDEBAR = readFileSync(
  new URL("../src/components/app-sidebar.tsx", import.meta.url),
  "utf8",
);

test("the unread dot is grey", () => {
  assert.match(SIDEBAR, /size-2 rounded-full bg-muted-foreground\/60/);
});

// A literal pair misses the contrast-boost theme, which recomputes
// --muted-foreground rather than swapping light for dark.
test("the unread dot carries no hardcoded light/dark pair", () => {
  assert.doesNotMatch(SIDEBAR, /d07a5f|df8a6f/i);
});

// Warm there does mean stopped or errored.
test("training run status dots are untouched", () => {
  assert.match(SIDEBAR, /runStatusDotClass\(run\.status\)/);
});
