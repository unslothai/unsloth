// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// In Chrome select() focuses a blurred input, so an unguarded deferred select steals focus.
// Browser coverage: tests/studio/playwright_image_download_cancel_retry.py.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SOURCE = "features/model-picker/components/numeric-value-input.tsx";

test("the queued select is guarded on the input still having focus", () => {
  const source = readSrc(SOURCE);
  const raf = source.indexOf("requestAnimationFrame(");
  assert.ok(raf > 0, "onFocus should still defer the select by a frame");

  const guard = source.indexOf("document.activeElement === target", raf);
  const select = source.indexOf("target.select()", raf);
  assert.ok(
    guard > 0,
    "the deferred select must check that the input still holds focus, or it steals it back",
  );
  assert.ok(
    guard < select,
    "the focus check has to run before select(), not after it",
  );
});

test("nothing else selects an input without checking focus first", () => {
  const source = readSrc(SOURCE);
  const selects = [...source.matchAll(/\.select\(\)/g)];
  assert.equal(
    selects.length,
    1,
    "more than one select() here; each one needs its own focus guard",
  );
});
