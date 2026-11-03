// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const src = readFileSync(
  new URL("../src/features/flows/editor/node-card.tsx", import.meta.url),
  "utf8",
);

test("a finished card keeps the plain border; green is hover and selection", () => {
  const table = src.slice(src.indexOf("const STATUS_BORDER"), src.indexOf("const LOUD_STATUS"));
  assert.doesNotMatch(table, /emerald/);
  assert.match(src, /"hover:border-emerald-500\/60 in-\[\.selected\]:border-emerald-500\/80"/);
});

test("failed, waiting and running keep their own border on hover", () => {
  assert.match(src, /failed:\s*"border-destructive\/60 hover:border-destructive\/60/);
  assert.match(src, /waiting: "border-dashed border-amber-400\/70 hover:border-amber-400\/70"/);
  assert.match(src, /LOUD_STATUS = new Set\(\["waiting", "loading_model", "running", "failed"\]\)/);
});
