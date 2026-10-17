// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ADMISSION_COMMENT_DONE,
  ADMISSION_COMMENT_PAUSED,
  ADMISSION_COMMENT_RECOMPUTED,
  ADMISSION_COMMENT_RESUMED,
  ADMISSION_COMMENT_WAIT,
  admissionStatusLabel,
  readAdmissionComment,
} from "../src/features/chat/utils/admission-status.ts";

test("the four signals are recognised", () => {
  assert.equal(readAdmissionComment(`: ${ADMISSION_COMMENT_WAIT}`), "waiting");
  assert.equal(readAdmissionComment(`: ${ADMISSION_COMMENT_DONE}`), "admitted");
  assert.equal(readAdmissionComment(`: ${ADMISSION_COMMENT_PAUSED}`), "paused");
  assert.equal(
    readAdmissionComment(`: ${ADMISSION_COMMENT_RESUMED}`),
    "resumed",
  );
});

test("a recompute is read, and says nothing about what the run is doing", () => {
  assert.equal(
    readAdmissionComment(`: ${ADMISSION_COMMENT_RECOMPUTED}`),
    "recomputed",
  );
  assert.equal(admissionStatusLabel("recomputed"), null);
});

test("the space after the colon is optional", () => {
  assert.equal(readAdmissionComment(":admission-wait"), "waiting");
  assert.equal(readAdmissionComment(":  admission-wait  "), "waiting");
});

test("the keep-alive comment is left to its own reader", () => {
  assert.equal(readAdmissionComment(": keep-alive"), null);
  assert.equal(readAdmissionComment(": keep-alive 12345"), null);
});

test("a data line is never an admission signal", () => {
  // Only the leading colon tells a comment from a payload spelling one.
  assert.equal(readAdmissionComment("data: admission-wait"), null);
  assert.equal(readAdmissionComment(""), null);
  assert.equal(readAdmissionComment("admission-wait"), null);
});

test("an unknown comment is ignored rather than guessed at", () => {
  assert.equal(readAdmissionComment(": admission-something-new"), null);
});

test("a run that is not generating gets a line, one that is gets none", () => {
  assert.equal(admissionStatusLabel("waiting"), "Waiting for a free slot");
  assert.equal(
    admissionStatusLabel("paused"),
    "Paused while another chat finishes",
  );
  // Null is what clears the indicator, so these two must not return a string.
  assert.equal(admissionStatusLabel("admitted"), null);
  assert.equal(admissionStatusLabel("resumed"), null);
});

test("queued and paused do not share one message", () => {
  assert.notEqual(
    admissionStatusLabel("waiting"),
    admissionStatusLabel("paused"),
  );
});

test("neither line uses failure vocabulary", () => {
  for (const status of ["waiting", "paused"] as const) {
    const label = admissionStatusLabel(status) ?? "";
    assert.ok(label.length > 0);
    for (const word of ["error", "fail", "problem", "unable", "sorry"]) {
      assert.ok(
        !label.toLowerCase().includes(word),
        `${status} label should not say "${word}": ${label}`,
      );
    }
  }
});
