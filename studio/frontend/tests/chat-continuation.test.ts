// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  incompleteLabel,
  isRestart,
  joinContinuation,
  readContinuationRequest,
  readIncompleteInfo,
  stripContinuationOverlap,
} = await import("../src/features/chat/utils/continuation.ts");

const PARTIAL =
  "There are three steps to proofing dough properly. First, warm the bowl and";

test("a token-exact continuation is appended verbatim", () => {
  // What a local model returns: the prompt ended inside the partial turn, so the
  // continuation opens on the next word with nothing repeated.
  assert.equal(
    joinContinuation(PARTIAL, " cover it with a damp cloth."),
    `${PARTIAL} cover it with a damp cloth.`,
  );
});

test("a repeated tail is dropped instead of stuttering", () => {
  const continuation = "warm the bowl and cover it with a damp cloth.";
  assert.equal(
    joinContinuation(PARTIAL, continuation),
    "There are three steps to proofing dough properly. First, warm the bowl and cover it with a damp cloth.",
  );
});

test("a short coincidental overlap is left alone", () => {
  // "and" is below the minimum overlap: trimming it would eat real output.
  const continuation = "and then wait.";
  assert.equal(
    stripContinuationOverlap("...warm the bowl and", continuation),
    continuation,
  );
});

test("a provider that ignores prefill and restarts replaces the partial", () => {
  const restart = `${PARTIAL} cover it with a damp cloth. Second, wait.`;
  assert.ok(isRestart(PARTIAL, restart));
  // Concatenating would render the opening sentence twice.
  assert.equal(joinContinuation(PARTIAL, restart), restart);
});

test("mid-stream the restart check is deferred", () => {
  const restart = `${PARTIAL} cover it`;
  assert.equal(
    joinContinuation(PARTIAL, restart, { streaming: true }),
    `${PARTIAL}${restart.slice(PARTIAL.length)}`,
  );
});

test("a partial too short to judge is never treated as a restart", () => {
  assert.equal(isRestart("Sure!", "Sure! Here is the answer."), false);
  assert.equal(
    joinContinuation("Sure!", " Here is the answer."),
    "Sure! Here is the answer.",
  );
});

test("an empty partial yields the continuation alone", () => {
  assert.equal(joinContinuation("", "anything"), "anything");
});

test("incomplete metadata round-trips and unknown reasons are ignored", () => {
  assert.deepEqual(
    readIncompleteInfo({ custom: { incomplete: { reason: "length" } } }),
    { reason: "length" },
  );
  assert.equal(
    readIncompleteInfo({ custom: { incomplete: { reason: "banana" } } }),
    null,
  );
  assert.equal(readIncompleteInfo(undefined), null);
  assert.equal(readIncompleteInfo({}), null);
});

test("every stop reason has a label", () => {
  assert.equal(incompleteLabel("length"), "Response hit the Max Tokens limit");
  assert.equal(incompleteLabel("cancelled"), "Response stopped");
  assert.equal(incompleteLabel("interrupted"), "Response interrupted");
});

test("a continuation request is read only when it carries text", () => {
  assert.deepEqual(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "half an answer" } },
    }),
    { partial: "half an answer" },
  );
  assert.equal(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "" } },
    }),
    null,
  );
  assert.equal(readContinuationRequest({}), null);
  assert.equal(readContinuationRequest(undefined), null);
});
