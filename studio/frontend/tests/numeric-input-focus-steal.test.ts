// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// NumericValueInput selects its contents a frame after it takes focus. In Chrome
// HTMLInputElement.select() FOCUSES a blurred input, taking focus off whatever holds it, so
// an unguarded select steals focus back from the field the user moved to. Two of these
// focused in one task -- tab, or a click straight from one field to the next -- then steal
// from each other every frame for as long as the page stays open, because each steal fires
// focus on the other input, whose handler queues the next one.
//
// The behaviour is pinned in the browser by tests/studio/playwright_image_download_cancel_retry.py.
// This is the cheap half: it fails in unit CI, with no browser, if the guard is dropped.

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
  // DRIFT: a second deferred select added later would reintroduce the loop while the guard
  // above still reads as present.
  const source = readSrc(SOURCE);
  const selects = [...source.matchAll(/\.select\(\)/g)];
  assert.equal(
    selects.length,
    1,
    "more than one select() here; each one needs its own focus guard",
  );
});
