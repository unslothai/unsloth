// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

// A stored run's jobId goes into the request path with the UI session: "../../settings/..." must stay one
// path segment, not reach another API route.
test("every recipe job URL encodes the job id as one path segment", () => {
  const source = readFileSync(
    new URL("../src/features/recipe-studio/api/index.ts", import.meta.url),
    "utf8",
  );
  const urls = source.match(/`[^`]*\/jobs\/\$\{[^}]*\}[^`]*`/g) ?? [];
  assert.ok(urls.length >= 7, `found ${urls.length} job URLs`);
  for (const url of urls) {
    assert.match(url, /\/jobs\/\$\{encodeURIComponent\([^)]*\)\}/, url);
  }
});
