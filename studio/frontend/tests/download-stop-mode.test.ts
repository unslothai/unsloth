// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  downloadActionAriaLabel,
  downloadStopMode,
} from "../src/features/hub/catalog/use-download-card-state.ts";

test("an HTTP download pauses, because its partial can be continued", () => {
  assert.equal(downloadStopMode("http"), "pause");
});

test("a Xet download cancels, because it has to start over", () => {
  assert.equal(downloadStopMode("xet"), "cancel");
});

test("a fresh HTTP download reads its own transport, not the absent partial", () => {
  assert.equal(downloadStopMode("http", null), "pause");
});

test("a restarted conflict follows the new transport, not the old partial", () => {
  assert.equal(downloadStopMode("xet", "http"), "cancel");
  assert.equal(downloadStopMode("http", "xet"), "pause");
});

test("an adopted job with no known transport falls back to the partial", () => {
  assert.equal(downloadStopMode(null, "http"), "pause");
});

test("an unknown transport cancels, never promising a resume", () => {
  assert.equal(downloadStopMode(null), "cancel");
  assert.equal(downloadStopMode(undefined, undefined), "cancel");
});

test("the accessible label matches what the button does", () => {
  assert.equal(downloadActionAriaLabel(true, false, "pause"), "Pause download");
  assert.equal(
    downloadActionAriaLabel(true, false, "cancel"),
    "Cancel download",
  );
  assert.equal(downloadActionAriaLabel(true, true, "pause"), "Cancelling…");
  assert.equal(downloadActionAriaLabel(false, false, "pause"), undefined);
});

test("a Xet run that fell back to HTTP still cancels, not pauses", () => {
  // The retry reclaims the job as HTTP but keeps the Xet cancel marker, so
  // stopping it leaves a partial that has to start over.
  assert.equal(downloadStopMode("http", null, "xet"), "cancel");
});

test("an HTTP job whose marker is also HTTP still pauses", () => {
  assert.equal(downloadStopMode("http", null, "http"), "pause");
});

test("no marker leaves the live transport in charge", () => {
  assert.equal(downloadStopMode("http", null, null), "pause");
  assert.equal(downloadStopMode("http", null, undefined), "pause");
  assert.equal(downloadStopMode("xet", "http", null), "cancel");
});
