// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The node suite has no DOM to mount the sidebar, so read from source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SIDEBAR = readSrc("components/app-sidebar.tsx");

test("the unread dot is grey", () => {
  assert.match(SIDEBAR, /size-2 rounded-full bg-muted-foreground\/60/);
});

// A literal pair misses the contrast-boost theme, which recomputes --muted-foreground.
test("the unread dot carries no hardcoded light/dark pair", () => {
  assert.doesNotMatch(SIDEBAR, /d07a5f|df8a6f/i);
});

test("training run status dots are untouched", () => {
  assert.match(SIDEBAR, /runStatusDotClass\(run\.status\)/);
});
