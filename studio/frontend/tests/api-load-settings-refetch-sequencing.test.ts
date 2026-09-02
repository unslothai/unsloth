// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A row's forget button disables only that row, so two forgets overlap, and each one
// refetches the whole map when it lands. The two GETs read the server at different
// moments and answer in whatever order the network gives: the earlier read still holds
// the row the later forget removed, so if it paints last the panel shows an entry the
// server no longer has, and it stays there until the panel remounts. Last issued wins.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/api-monitor/components/saved-model-settings.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
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
  // Read into a local first, or the guard runs before the response exists.
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
