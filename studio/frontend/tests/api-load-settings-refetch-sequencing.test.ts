// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Overlapping forgets each refetch; responses can land out of order, so the last issued wins.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SOURCE = readSrc(
  "features/api-monitor/components/saved-model-settings.tsx",
);

const LOAD = SOURCE.slice(
  SOURCE.indexOf("const load = useCallback("),
  SOURCE.indexOf("useEffect(() => {"),
);

test("every refetch takes a sequence number", () => {
  assert.match(SOURCE, /const loadSeq = useRef\(0\);/);
  assert.match(LOAD, /const seq = \+\+loadSeq\.current;/);
});

test("a superseded refetch paints no rows", () => {
  const guard = LOAD.slice(0, LOAD.indexOf("setOverrides("));
  assert.match(
    guard,
    /if \(seq !== loadSeq\.current\) \{\s*return;\s*\}/,
    "the check must sit between the await and the row write",
  );
  assert.match(LOAD, /const next = await fetchModelOverrides\(\);/);
});

// The old-backend fallback diffs these keys against the returned map; [] would disable it.
test("a forget hands the panel's listed keys to the fallback", () => {
  const forget = SOURCE.slice(
    SOURCE.indexOf("const forget = useCallback("),
    SOURCE.indexOf("const entries = "),
  );
  assert.match(forget, /listedKeys: Object\.keys\(overrides \?\? \{\}\),/);
  assert.match(forget, /\[load, overrides\],\s*\);\s*$/);
});

test("a superseded refetch does not report its failure either", () => {
  const catchBlock = LOAD.slice(LOAD.indexOf("} catch (err: unknown) {"));
  assert.match(catchBlock, /if \(seq !== loadSeq\.current\) \{\s*return;\s*\}/);
  assert.ok(
    catchBlock.indexOf("loadSeq.current") < catchBlock.indexOf("setError("),
    "the guard must precede the error write",
  );
});
