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
const USE_DEVICE_INVENTORY_SOURCE = new URL(
  "../src/features/hub/inventory/use-device-inventory.ts",
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

test("refresh reuses an in-flight post-cold forced scan", () => {
  const source = readFileSync(USE_DEVICE_INVENTORY_SOURCE, "utf8");
  assert.match(source, /function getOrQueuePostColdForce</);
  assert.match(source, /const existingPostCold = postColdForce\.get\(key\)/);
  assert.match(
    source,
    /if \(existingPostCold\) \{\s*return existingPostCold;\s*\}/,
  );
});

test("forced fetch joins an in-flight post-cold forced scan", () => {
  const source = readFileSync(USE_DEVICE_INVENTORY_SOURCE, "utf8");
  assert.match(
    source,
    /if \(options\.force\) \{\s*const postCold = postColdForce\.get\(key\)/,
  );
  assert.match(source, /if \(postCold\) \{\s*return postCold;\s*\}/);
});

test("HubModelPicker refreshes inventory only on warm opens", () => {
  const source = readFileSync(PICKERS_SOURCE, "utf8");
  assert.match(source, /const wasOpenRef = useRef\(false\);/);
  assert.match(source, /if \(!isOpening \|\| !cachedReady\) return;/);
  assert.doesNotMatch(source, /const warmAtMountRef = useRef\(cachedReady\);/);
  assert.doesNotMatch(
    source,
    /useEffect\(\(\) => \{\s*if \(!cachedReady\) return;\s*void refreshInventory\(\);\s*\}, \[cachedReady, refreshInventory\]\);/,
  );
  assert.doesNotMatch(
    source,
    /useEffect\(\(\) => \{\s*void refreshInventory\(\);\s*\}, \[refreshInventory\]\);/,
  );
});
