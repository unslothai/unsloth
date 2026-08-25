// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultReferenceVideoTrim,
  referenceVideoTrimError,
  referenceVideoTrimFeedback,
} from "../src/features/video/reference-trim.ts";

test("long reference videos default to the first model-sized interval", () => {
  assert.deepEqual(defaultReferenceVideoTrim(20), { start: 0, end: 15 });
  assert.deepEqual(defaultReferenceVideoTrim(15), { start: null, end: null });
  assert.deepEqual(defaultReferenceVideoTrim(undefined), {
    start: null,
    end: null,
  });
});

test("short reference videos need no explicit trim", () => {
  assert.equal(referenceVideoTrimError("Video 1", null, null, 15), null);
});

test("an overlong reference video requires both endpoints", () => {
  assert.match(
    referenceVideoTrimError("Video 2", null, null, 20) ?? "",
    /2 to 15 second section/,
  );
  assert.match(
    referenceVideoTrimError("Video 2", 3, null, 20) ?? "",
    /both start and end times/,
  );
});

test("exact two and fifteen second intervals are accepted", () => {
  assert.equal(referenceVideoTrimError("Video 1", 3, 5, 20), null);
  assert.equal(referenceVideoTrimError("Video 1", 3, 18, 20), null);
});

test("invalid or out-of-source intervals are refused", () => {
  for (const [start, end] of [
    [-1, 3],
    [3, 3],
    [3, 4.9],
    [0, 15.1],
    [10, 21],
    [Number.NaN, 5],
    [0, Number.POSITIVE_INFINITY],
  ]) {
    assert.match(
      referenceVideoTrimError("Video 3", start, end, 20) ?? "",
      /2 to 15 seconds within/,
    );
  }
});

test("trim feedback distinguishes required, optional and selected states", () => {
  assert.deepEqual(referenceVideoTrimFeedback("Video 1", null, null, 10), {
    message: "10.0s source. Trim is optional.",
    invalid: false,
  });
  assert.deepEqual(referenceVideoTrimFeedback("Video 1", null, null, 20), {
    message: "20.0s source. Select a 2 to 15 second section for Video 1.",
    invalid: true,
  });
  assert.deepEqual(referenceVideoTrimFeedback("Video 1", 0, 15, 20), {
    message:
      "20.0s source. First 15.0s selected automatically. Adjust the times to use another section.",
    invalid: false,
  });
  assert.deepEqual(referenceVideoTrimFeedback("Video 1", 5, 13, 20), {
    message: "20.0s source. Selected 5.0s to 13.0s (8.0s).",
    invalid: false,
  });
});

test("intervals the backend accepts are not refused over floating point", () => {
  // validate_h3_reference_trim allows 1e-6 of slack. The 0.1-step inputs reach differences
  // that are inexact in binary, and comparing exactly here refused them.
  assert.equal(2.3 - 0.3 < 2, true);
  assert.equal(16.1 - 1.1 > 15, true);
  for (const [start, end] of [
    [0.3, 2.3],
    [0.8, 2.8],
    [1.3, 3.3],
    [2.1, 4.1],
    [1.1, 16.1],
    [3.1, 18.1],
  ]) {
    assert.equal(
      referenceVideoTrimError("Video 1", start, end, 30),
      null,
      `${start} to ${end} should be accepted`,
    );
  }
});

test("a source shorter than the model minimum is refused before it is sent", () => {
  // No interval inside a 1.5s clip reaches 2s, so the fields are a dead end and the backend
  // refuses it regardless.
  assert.equal(
    referenceVideoTrimError("Video 1", null, null, 1.5),
    "Video 1 is shorter than the 2 second minimum",
  );
  assert.equal(
    referenceVideoTrimError("Video 1", 0, 1.5, 1.5),
    "Video 1 is shorter than the 2 second minimum",
  );
  // Exactly the minimum is fine, and so is an unknown duration.
  assert.equal(referenceVideoTrimError("Video 1", null, null, 2), null);
  assert.equal(referenceVideoTrimError("Video 1", null, null, undefined), null);
});
