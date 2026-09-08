// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// Node's type stripping cannot compile JSX, so pin the status-to-label mapping
// by reading the component source, following the other .tsx tests in this suite.
const source = readFileSync(
  new URL("../src/components/assistant-ui/tool-fallback.tsx", import.meta.url),
  "utf8",
);

test("tool fallback labels distinguish running and finished calls", () => {
  assert.match(
    source,
    /const label = isCancelled\s*\?\s*"Cancelled tool"\s*:\s*isRunning\s*\?\s*"Using tool"\s*:\s*"Used tool";/,
  );
});

test("the visible and shimmering labels share the status-derived text", () => {
  assert.equal(
    source.match(/\{label\}:\{" "\}/g)?.length,
    2,
    "both trigger label layers must render the same status-derived label",
  );
});
