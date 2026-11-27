// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

test("the GGUF save-dir default refuses cache-shaped destinations", async () => {
  const page = await readSrcAsync("features/export/export-page.tsx");
  // A cache-layout copy parked in a scan folder lists as a custom row whose path is its
  // snapshot directory. The sibling default must refuse any path under
  // `models--*/snapshots/`, or the export lands as a fake newest revision that every
  // cache scan then reads back -- and any path ending at the `models--*` repo dir itself,
  // whose sibling would read as another repo.
  assert.match(page, /const parts = trimmed\.split\(/);
  assert.match(page, /part\.toLowerCase\(\) === "snapshots"/);
  assert.match(
    page,
    /\.toLowerCase\(\)\.startsWith\("models--"\)/,
  );
  assert.match(page, /const atCacheRepoDir =/);
  assert.match(page, /if \(insideCacheSnapshots \|\| atCacheRepoDir\) return null;/);
  // The refused default falls back to the ordinary exports-relative directory.
  assert.match(
    page,
    /return siblingGgufDirectory\(localModel\.path\) \?\? relative;/,
  );
});
