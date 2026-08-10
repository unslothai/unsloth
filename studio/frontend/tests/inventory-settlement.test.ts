// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { resolveInventorySettlement } from "../src/features/hub/inventory/inventory-settlement.ts";

const emptyReadyInventory = {
  downloadedReady: true,
  emptyRevalidationSignature: "empty-ready",
  hasActiveEmptyRefresh: false,
  hasInventoryRows: false,
  hasUnreadyInventoryFailure: false,
  inventoryFailed: false,
};

test("keeps an empty ready inventory unsettled until revalidation", () => {
  assert.deepEqual(
    resolveInventorySettlement({
      ...emptyReadyInventory,
      lastEmptyRevalidationSignature: null,
    }),
    {
      emptyRevalidationRequired: true,
      inventorySettled: false,
    },
  );
  assert.deepEqual(
    resolveInventorySettlement({
      ...emptyReadyInventory,
      lastEmptyRevalidationSignature: "empty-ready",
    }),
    {
      emptyRevalidationRequired: false,
      inventorySettled: true,
    },
  );
});

test("settles usable rows and failures with prior ready state", () => {
  for (const state of [
    { hasInventoryRows: true, inventoryFailed: false },
    { hasInventoryRows: false, inventoryFailed: true },
  ]) {
    assert.deepEqual(
      resolveInventorySettlement({
        ...emptyReadyInventory,
        ...state,
        lastEmptyRevalidationSignature: null,
      }),
      {
        emptyRevalidationRequired: false,
        inventorySettled: true,
      },
    );
  }
});

test("keeps an unready failure unsettled for the next enable retry", () => {
  assert.deepEqual(
    resolveInventorySettlement({
      ...emptyReadyInventory,
      hasUnreadyInventoryFailure: true,
      inventoryFailed: true,
      lastEmptyRevalidationSignature: null,
    }),
    {
      emptyRevalidationRequired: false,
      inventorySettled: false,
    },
  );
});

test("settles a partial failure when another source returned rows", () => {
  assert.deepEqual(
    resolveInventorySettlement({
      ...emptyReadyInventory,
      hasInventoryRows: true,
      hasUnreadyInventoryFailure: true,
      inventoryFailed: true,
      lastEmptyRevalidationSignature: null,
    }),
    {
      emptyRevalidationRequired: false,
      inventorySettled: true,
    },
  );
});
