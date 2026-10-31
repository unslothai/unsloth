// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Timeouts reject with TimeoutError on Chromium/Gecko but AbortError on WebKit (the desktop
// engine) and on engines without signal.reason.

import assert from "node:assert/strict";
import test from "node:test";

import { describeVariantListingError } from "../src/features/model-picker/components/model-selector/variant-listing-error.ts";

const TIMED_OUT =
  "Timed out listing quantizations. Check your connection to Hugging Face, then retry.";

test("a timeout is named as one on every engine", () => {
  assert.equal(describeVariantListingError(new DOMException("x", "TimeoutError")), TIMED_OUT);
  assert.equal(describeVariantListingError(new DOMException("x", "AbortError")), TIMED_OUT);
  // Older WebKit: DOMException does not inherit from Error, so instanceof misses it.
  assert.equal(describeVariantListingError({ name: "TimeoutError" }), TIMED_OUT);
  assert.equal(describeVariantListingError({ name: "AbortError", message: "" }), TIMED_OUT);
});

test("a real backend failure keeps its own message", () => {
  assert.equal(
    describeVariantListingError(new Error("Failed to list GGUF variants: boom")),
    "Failed to list GGUF variants: boom",
  );
});

test("anything unrecognisable still reads as a failure, never blank", () => {
  for (const value of [null, undefined, "a string", 42, {}, { name: 42 }, new Error("")]) {
    assert.equal(describeVariantListingError(value), "Failed to load variants");
  }
  assert.equal(describeVariantListingError(Object.create(null)), "Failed to load variants");
});
