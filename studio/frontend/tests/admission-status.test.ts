// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ADMISSION_COMMENT_DONE,
  ADMISSION_COMMENT_PAUSED,
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

test("the space after the colon is optional", () => {
  // SSE treats `:x` and `: x` as the same comment, and an intermediary may rewrite it.
  assert.equal(readAdmissionComment(":admission-wait"), "waiting");
  assert.equal(readAdmissionComment(":  admission-wait  "), "waiting");
});

test("the keep-alive comment is left to its own reader", () => {
  assert.equal(readAdmissionComment(": keep-alive"), null);
  assert.equal(readAdmissionComment(": keep-alive 12345"), null);
});

test("a data line is never an admission signal", () => {
  // The two are told apart by the leading colon alone, so a payload that happens to spell
  // one must not be mistaken for the comment.
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
  // Queued has produced nothing; paused has visible text above it. One line for both
  // would put "waiting for a free slot" under a half-written answer.
  assert.notEqual(
    admissionStatusLabel("waiting"),
    admissionStatusLabel("paused"),
  );
});

test("neither line uses failure vocabulary", () => {
  // Neither state is an error, and the whole point of the indicator is to say so.
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
