// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  settledFailureStatus,
  shouldRetrySystemDiscovery,
} from "../src/hooks/system-discovery.ts";

test("retries a cold system cache while an interested subscriber exists", () => {
  assert.equal(shouldRetrySystemDiscovery(true, undefined, 1), true);
  assert.equal(shouldRetrySystemDiscovery(true, undefined, 0), false);
});

test("retries only an unavailable Vulkan inventory after discovery", () => {
  assert.equal(
    shouldRetrySystemDiscovery(
      false,
      { backend: "vulkan", available: false },
      1,
    ),
    true,
  );
  assert.equal(
    shouldRetrySystemDiscovery(
      false,
      { backend: "vulkan", available: true },
      1,
    ),
    false,
  );
  assert.equal(
    shouldRetrySystemDiscovery(false, { backend: "cuda", available: false }, 1),
    false,
  );
  assert.equal(shouldRetrySystemDiscovery(false, undefined, 1), false);
});

test("a settled failure leaves the placeholder, not a live check", () => {
  // Polling off means nothing retries, so "pending" would outlive the check.
  assert.equal(settledFailureStatus("pending"), "unavailable");
  assert.equal(settledFailureStatus("unavailable"), "unavailable");
});

test("a settled failure does not blank a reading already on screen", () => {
  assert.equal(settledFailureStatus("ready"), "ready");
});
