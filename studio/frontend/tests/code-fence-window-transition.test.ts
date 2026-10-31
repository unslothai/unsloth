// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const DEFER = readFileSync(
  new URL("../src/components/assistant-ui/code-fence-defer.tsx", import.meta.url),
  "utf8",
);

test("a finished fence moves its line window in a transition, a streaming one does not", () => {
  const measure = DEFER.slice(
    DEFER.indexOf("measure.current = () => {"),
    DEFER.indexOf("useLayoutEffect(() => {", DEFER.indexOf("measure.current = () => {")),
  );
  assert.match(
    measure,
    /const shifting =\s*deferShifts && current\.current !== null && frameRects !== null && !resized;/,
    "only a window that already exists, of a finished fence, moved in a frame where the viewport kept its size",
  );
  assert.match(
    measure,
    /const size = `\$\{bounds \? bounds\.width : window\.innerWidth\}x\$\{viewportHeight\}`;\s*if \(size !== viewportSize\.current\) resizedAt\.current = performance\.now\(\);\s*viewportSize\.current = size;\s*const resized = performance\.now\(\) - resizedAt\.current < RESIZE_SETTLE_MS;/,
    "a resize re-wraps every line, so its steps and the frames right after stay synchronous",
  );
  assert.match(
    measure,
    /if \(shifting\) startTransition\(\(\) => setState\(update\)\);\s*else setState\(update\);/,
    "the first measure, print, find and streaming fences keep the synchronous update",
  );
  assert.equal(
    (measure.match(/startTransition\(/g) ?? []).length,
    1,
    "the window that hides every line for print or find is never deferred",
  );
  const body = DEFER.slice(DEFER.indexOf("export const FenceBody = memo("));
  assert.match(
    body,
    /useLineWindow\(\s*code,\s*surface,\s*wrapper,\s*frame,\s*tokens\?\.length \?\? 0,\s*windowing,\s*!isIncomplete,\s*\)/,
    "a fence still streaming keeps its window in step with the lines being appended",
  );
});
