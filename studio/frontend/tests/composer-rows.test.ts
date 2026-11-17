// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { composerMaxRows } from "../src/hooks/use-composer-rows.ts";
import { readSrc } from "./helpers/kit.ts";

test("the prompt grows to 10 rows, fewer on a short or zoomed window, never under 3", () => {
  assert.equal(composerMaxRows(1080, 24), 10);
  assert.equal(composerMaxRows(800, 24), 10);
  assert.equal(composerMaxRows(640, 24), 8);
  // A 900px window at 150% browser zoom is 600 CSS px tall.
  assert.equal(composerMaxRows(600, 24), 7);
  assert.equal(composerMaxRows(200, 24), 3);
  // A larger interface scale means taller rows, so fewer of them fit.
  assert.equal(composerMaxRows(640, 30), 6);
});

test("both composers take the viewport-aware row cap", () => {
  assert.match(
    readSrc("components/assistant-ui/thread.tsx"),
    /maxRows=\{maxRows\}/,
  );
  const shared = readSrc("features/chat/shared-composer.tsx");
  assert.match(shared, /lineHeight \* maxRowsRef\.current \+ paddingY/);
  assert.doesNotMatch(shared, /lineHeight \* 12/);
});
