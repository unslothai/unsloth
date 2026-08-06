// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { shouldCoalesceInFlightInventoryFetch } from "../src/features/hub/inventory/inventory-fetch-policy.ts";

const PICKERS_SOURCE = new URL(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
  import.meta.url,
);

test("forced refresh coalesces with an in-flight cold fetch", () => {
  assert.equal(
    shouldCoalesceInFlightInventoryFetch({
      force: true,
      ready: false,
      inFlight: true,
    }),
    true,
  );
});

test("forced refresh after rows are ready queues a new scan", () => {
  assert.equal(
    shouldCoalesceInFlightInventoryFetch({
      force: true,
      ready: true,
      inFlight: true,
    }),
    false,
  );
});

test("non-forced refresh always coalesces with an in-flight request", () => {
  assert.equal(
    shouldCoalesceInFlightInventoryFetch({
      force: false,
      ready: true,
      inFlight: true,
    }),
    true,
  );
});

test("HubModelPicker does not mount-refresh inventory on open", () => {
  const source = readFileSync(PICKERS_SOURCE, "utf8");
  assert.doesNotMatch(
    source,
    /useEffect\(\(\) => \{\s*void refreshInventory\(\);\s*\}, \[refreshInventory\]\);/,
  );
});
