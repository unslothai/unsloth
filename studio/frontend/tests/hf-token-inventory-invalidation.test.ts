// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { useHfTokenStore } from "../src/features/hub/stores/hf-token-store.ts";
import { getInventoryVersion } from "../src/features/hub/stores/inventory-events.ts";

test("token edits do not invalidate token-independent inventory", () => {
  const initialVersion = getInventoryVersion();

  useHfTokenStore.getState().setToken("hf_123");
  useHfTokenStore.getState().setToken("hf_1234");
  useHfTokenStore.getState().setToken("hf_12345");

  assert.equal(useHfTokenStore.getState().token, "hf_12345");
  assert.equal(getInventoryVersion(), initialVersion);
});
