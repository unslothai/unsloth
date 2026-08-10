// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { inventoryRefreshDecision } from "../src/features/hub/inventory/inventory-freshness.ts";

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
