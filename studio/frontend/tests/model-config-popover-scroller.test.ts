// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The config page scrolls inside the popover rather than making the popover the scroller, so the
// rounded surface keeps its corners when the OS draws scrollbars always. That splits one box into
// two, and the split only holds while the outer one is a height-capped flex column: measured in a
// browser against the built CSS, a non-flex parent leaves the inner box at its full content height
// with nothing scrollable and Run below the clip. Both halves of the contract are pinned here.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const SELECTOR = read(
  "../src/features/model-picker/components/model-selector.tsx",
);
const POPOVER = read("../src/components/ui/popover.tsx");

test("the popover surface clips and caps, so it never scrolls itself", () => {
  // Scrolling the rounded box put the bar inside it, running through the top and bottom right
  // corners and squaring them off.
  assert.match(
    SELECTOR,
    /max-h-\[var\(--radix-popover-content-available-height\)\][^"]*overflow-hidden p-0/,
    "capped and clipped, with the padding handed to the scroller",
  );
});

test("the config page gets its own scroller inside that surface", () => {
  assert.ok(
    SELECTOR.includes(
      '<div className="min-h-0 w-full overflow-y-auto px-4 pt-4 pb-4">',
    ),
    "the inner box scrolls and carries the padding the surface gave up",
  );
});

test("the surface is a flex column, which is what gives the scroller a height", () => {
  // The inner box sets no height of its own. It gets one by being a shrinkable flex item in a
  // capped column: drop `flex flex-col` here and it stays at full content height, nothing
  // scrolls, and the bottom of the config page including Run is clipped and unreachable.
  const content = POPOVER.slice(POPOVER.indexOf("function PopoverContent"));
  const base = content.slice(0, content.indexOf("{...props}"));
  assert.match(base, /\bflex flex-col\b/, "the base class is the constraint");
  // Overriding it from the call site would break the same way, quietly.
  const configClass = SELECTOR.slice(
    SELECTOR.indexOf("max-h-[var(--radix-popover-content-available-height)]"),
  ).slice(0, 200);
  assert.ok(
    !/\b(block|grid|inline-flex)\b/.test(configClass),
    "and the config-target branch does not replace it",
  );
});
