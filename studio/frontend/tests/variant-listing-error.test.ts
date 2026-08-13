// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Engines disagree on what a timed-out fetch rejects with. Measured in real engines:
// Chromium and Gecko give TimeoutError, WebKit gives AbortError, and engines without
// signal.reason (Safari < 15.4, older WebKitGTK) give AbortError whatever we abort with.
// WebKit is what the desktop app embeds, so classifying only TimeoutError left exactly
// those users on the generic message.

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
