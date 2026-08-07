// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  incompleteLabel,
  isContinuableContent,
  isRestart,
  joinContinuation,
  readContinuationRequest,
  readIncompleteInfo,
  rejectsAssistantPrefill,
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

test("a turn that called a tool cannot be continued", () => {
  // The continuation runs as a sibling, so the call and its result are not in the
  // outbound history and the resumed text would have lost its evidence.
  assert.equal(
    isContinuableContent([
      { type: "text", text: "Looking that up." },
      { type: "tool-call", toolCallId: "c1", toolName: "web_search" },
    ]),
    false,
  );
});

test("text and reasoning parts are continuable, empty text is not", () => {
  assert.equal(
    isContinuableContent([
      { type: "reasoning", text: "hmm" },
      { type: "text", text: "The answer is" },
    ]),
    true,
  );
  // Reasoning alone leaves nothing to resume from: it is never replayed.
  assert.equal(
    isContinuableContent([{ type: "reasoning", text: "hmm" }]),
    false,
  );
  assert.equal(isContinuableContent([{ type: "text", text: "" }]), false);
  assert.equal(isContinuableContent(undefined), false);
});

test("Anthropic is the provider that rejects a trailing assistant turn", () => {
  assert.equal(rejectsAssistantPrefill("anthropic"), true);
  assert.equal(rejectsAssistantPrefill("openai"), false);
  assert.equal(rejectsAssistantPrefill(undefined), false);
});

test("the overlap repair can eat a legitimate repeat, so local output skips it", () => {
  // A local backend resumes at the exact token boundary, so its output is already the
  // rest of the answer. Running the repair over it would delete a phrase the model
  // meant to write, which is why the adapter only repairs external provider output.
  const partial = "Ranking them, the clear winner is the second result";
  const continuation = "the second result held up best under load.";
  // "the second result" is trimmed as a repeat even though the model wrote it.
  assert.equal(
    stripContinuationOverlap(partial, continuation),
    " held up best under load.",
  );
  // Verbatim concatenation is what a token-exact backend needs.
  assert.equal(
    `${partial} ${continuation}`,
    "Ranking them, the clear winner is the second result the second result held up best under load.",
  );
});
