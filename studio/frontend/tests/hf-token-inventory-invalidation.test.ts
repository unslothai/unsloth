// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  hfApiToken,
  useHfTokenStore,
} from "../src/features/hub/stores/hf-token-store.ts";
import { getInventoryVersion } from "../src/features/hub/stores/inventory-events.ts";

test("token edits do not invalidate token-independent inventory", () => {
  const initialVersion = getInventoryVersion();

  useHfTokenStore.getState().setToken("hf_123");
  useHfTokenStore.getState().setToken("hf_1234");
  useHfTokenStore.getState().setToken("hf_12345");

  assert.equal(useHfTokenStore.getState().token, "hf_12345");
  assert.equal(getInventoryVersion(), initialVersion);
});

test("API token usability rejects malformed saved values", () => {
  assert.equal(hfApiToken(null), undefined);
  assert.equal(hfApiToken("   "), undefined);
  assert.equal(hfApiToken("not-a-token"), undefined);
  assert.equal(hfApiToken(" 'hf_legacy' "), "hf_legacy");
  assert.equal(
    hfApiToken("hf_1234567890123456789012345678901234"),
    "hf_1234567890123456789012345678901234",
  );
});
