// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isUnrecoverableSttLoadError,
  sttRequestError,
} from "../src/features/chat/adapters/stt-load-error.ts";

test("the backend's message survives on the error", () => {
  const error = sttRequestError("STT model 'small' is not downloaded.", 409);
  assert.ok(error instanceof Error);
  assert.equal(error.message, "STT model 'small' is not downloaded.");
  assert.equal(error.status, 409);
});

test("a model that is not downloaded ends the recording", () => {
  assert.equal(isUnrecoverableSttLoadError(sttRequestError("nope", 409)), true);
});

test("a backend without STT support ends the recording", () => {
  assert.equal(isUnrecoverableSttLoadError(sttRequestError("nope", 501)), true);
});

test("a load that failed for another reason keeps recording", () => {
  // /transcribe/raw loads the model itself, so these can still transcribe and
  // must not cost the user the audio they already spoke.
  for (const status of [400, 401, 403, 404, 429, 500, 502, 503]) {
    assert.equal(
      isUnrecoverableSttLoadError(sttRequestError("nope", status)),
      false,
      `HTTP ${status} should not end the recording`,
    );
  }
});

test("an error carrying no status keeps recording", () => {
  // Older servers, and any non-HTTP failure such as a dropped connection.
  assert.equal(isUnrecoverableSttLoadError(new Error("boom")), false);
  assert.equal(isUnrecoverableSttLoadError(undefined), false);
  assert.equal(isUnrecoverableSttLoadError(null), false);
  assert.equal(isUnrecoverableSttLoadError("409"), false);
});

test("a status that only looks like one is not trusted", () => {
  assert.equal(isUnrecoverableSttLoadError({ status: "409" }), false);
});
