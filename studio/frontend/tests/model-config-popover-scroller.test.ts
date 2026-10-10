// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The outer box must stay a height-capped flex column or the inner one cannot scroll.

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
  // Without `flex flex-col` the inner box keeps full height and Run is clipped.
  const content = POPOVER.slice(POPOVER.indexOf("function PopoverContent"));
  const base = content.slice(0, content.indexOf("{...props}"));
  assert.match(base, /\bflex flex-col\b/, "the base class is the constraint");
  const configClass = SELECTOR.slice(
    SELECTOR.indexOf("max-h-[var(--radix-popover-content-available-height)]"),
  ).slice(0, 200);
  assert.ok(
    !/\b(block|grid|inline-flex)\b/.test(configClass),
    "and the config-target branch does not replace it",
  );
});
