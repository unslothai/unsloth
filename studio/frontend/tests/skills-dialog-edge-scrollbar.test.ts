// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const DIALOG = await readFile(
  new URL("../src/features/chat/components/chat-skills-dialog.tsx", import.meta.url),
  "utf8",
);
const DIALOG_UI = await readFile(new URL("../src/components/ui/dialog.tsx", import.meta.url), "utf8");

test("the Skills list and editor scroll on the dialog's right edge", () => {
  // The offset only lands on the edge while it matches DialogContent's own padding.
  assert.match(DIALOG_UI, /rounded-4xl px-7 pt-8 pb-7/);
  const scrollers = [...DIALOG.matchAll(/className="hover-scrollbar [^"]*"/g)].map((m) => m[0]);
  assert.equal(scrollers.length, 2);
  for (const scroller of scrollers) {
    assert.match(scroller, / -mr-7 /);
    assert.match(scroller, / pr-7 /);
    assert.doesNotMatch(scroller, / pr-1 /);
  }
});
