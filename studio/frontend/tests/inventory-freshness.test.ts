// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  inventoryRefreshDecision,
  isInventoryStampFresh,
} from "../src/features/hub/inventory/inventory-freshness.ts";

const NOW = 100_000;
const KEY = "localModels:7:local";

test("fresh inventory is reused across rapid picker opens", () => {
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: true,
        loading: false,
        error: null,
        key: KEY,
        refreshedAt: NOW - 1_000,
      },
      KEY,
      NOW,
      30_000,
    ),
    "reuse",
  );
});

test("an in-flight inventory request is joined without queuing another scan", () => {
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: false,
        loading: true,
        error: null,
        key: KEY,
        refreshedAt: null,
      },
      KEY,
      NOW,
      30_000,
    ),
    "join",
  );
});

test("expired inventory is refreshed", () => {
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: true,
        loading: false,
        error: null,
        key: KEY,
        refreshedAt: NOW - 30_000,
      },
      KEY,
      NOW,
      30_000,
    ),
    "refresh",
  );
});

test("an invalidated inventory key is fetched without a forced follow-up", () => {
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: true,
        loading: false,
        error: null,
        key: "localModels:6:local",
        refreshedAt: NOW,
      },
      KEY,
      NOW,
      30_000,
    ),
    "join",
  );
});

test("a failed current key is refreshed even while its prior rows are fresh", () => {
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: true,
        loading: false,
        error: "scan failed",
        key: KEY,
        refreshedAt: NOW - 1_000,
      },
      KEY,
      NOW,
      30_000,
    ),
    "refresh",
  );
});

test("a stamp from the future is refreshed, not reused", () => {
  // `Date.now()` follows the system clock, so a stamp can end up in the future after an
  // NTP correction, a VM resume or a user editing the clock. A negative age must not read
  // as "younger than the window", or the picker stops scanning for the length of the skew.
  assert.equal(
    inventoryRefreshDecision(
      {
        ready: true,
        loading: false,
        error: null,
        key: KEY,
        refreshedAt: NOW + 3_600_000,
      },
      KEY,
      NOW,
      30_000,
    ),
    "refresh",
  );
});

test("a zero max age always refreshes, even after the clock steps backwards", () => {
  // refreshIfOlderThan(0) means "refresh unconditionally"; a future stamp must not turn it
  // into a no-op.
  for (const refreshedAt of [NOW - 1, NOW, NOW + 60_000]) {
    assert.equal(
      inventoryRefreshDecision(
        { ready: true, loading: false, error: null, key: KEY, refreshedAt },
        KEY,
        NOW,
        0,
      ),
      "refresh",
    );
  }
});

test("isInventoryStampFresh rejects a null, an expired and a future stamp", () => {
  assert.equal(isInventoryStampFresh(null, NOW, 30_000), false);
  assert.equal(isInventoryStampFresh(NOW - 30_000, NOW, 30_000), false);
  assert.equal(isInventoryStampFresh(NOW + 1, NOW, 30_000), false);
  assert.equal(isInventoryStampFresh(NOW - 29_999, NOW, 30_000), true);
  assert.equal(isInventoryStampFresh(NOW, NOW, 30_000), true);
});
