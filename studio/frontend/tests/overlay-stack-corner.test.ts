// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The corner stack holds the update banners, the download panel and the loaded
// models card. Its bottom inset was briefly computed from the Live monitor's
// box so the card would not land under it, which lifted the whole stack: with
// the monitor in that column the banners floated up the middle of the page
// instead of sitting in the corner. The inset is static again, and these keep
// it that way.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const provider = readFileSync(
  new URL("../src/app/provider.tsx", import.meta.url),
  "utf8",
);

const STACK_CLASSES =
  'className="pointer-events-none fixed bottom-4 right-4 z-[9998] flex max-h-[calc(100dvh_-_2rem)] flex-col items-end gap-2"';

test("both stacks are pinned to the bottom-right corner", () => {
  // One for the browser, one for the desktop update layer.
  const pinned = provider.split(STACK_CLASSES).length - 1;
  assert.equal(pinned, 2, "both corner stacks must carry the static inset");
});

test("nothing moves the stack off the corner at runtime", () => {
  assert.doesNotMatch(provider, /useStackGeometry/);
  assert.doesNotMatch(provider, /style=\{\{ bottom:/);
});
