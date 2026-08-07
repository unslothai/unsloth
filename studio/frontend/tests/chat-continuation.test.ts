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
  modeAllowsContinuation,
  readContinuationRequest,
  readIncompleteInfo,
  readTextThoughtSignature,
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

test("providers that reject a trailing assistant turn get the instruction path", () => {
  // Anthropic 400s on a trailing assistant message since Claude 4.6; Gemini requires
  // a multiturn request to end in a user turn or a function response. Mistral answers
  // "Expected last role User or Tool (or Assistant with prefix True)" unless the turn
  // carries its `prefix: true` field, which the outbound message type has no room for.
  assert.equal(rejectsAssistantPrefill("anthropic"), true);
  assert.equal(rejectsAssistantPrefill("gemini"), true);
  assert.equal(rejectsAssistantPrefill("mistral"), true);
  assert.equal(rejectsAssistantPrefill("openai"), false);
  assert.equal(rejectsAssistantPrefill(undefined), false);
});

test("modes that answer from scratch do not offer Continue", () => {
  const plain = {
    fromAudioInput: false,
    audioOutputModel: false,
    deepResearchArmed: false,
  };
  assert.equal(modeAllowsContinuation(plain), true);
  assert.equal(
    modeAllowsContinuation({ ...plain, fromAudioInput: true }),
    false,
  );
  // A stopped TTS turn keeps its "Generating audio..." text, which passes the content
  // gates; the resumed run would regenerate the whole clip instead of continuing.
  assert.equal(
    modeAllowsContinuation({ ...plain, audioOutputModel: true }),
    false,
  );
  // Deep Research armed after the turn was cut: the run researches the user message
  // and its report replaces the partial.
  assert.equal(
    modeAllowsContinuation({ ...plain, deepResearchArmed: true }),
    false,
  );
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

test("a continuation carries the Gemini signature of the turn it resumes", () => {
  // The sibling run drops the original assistant message, so the signature has to
  // travel with the partial or the model history goes back to Gemini unsigned.
  assert.equal(
    readTextThoughtSignature([
      { type: "text", text: "first", _google_thought_signature: "SIG-A" },
      { type: "text", text: "second", _google_thought_signature: "SIG-B" },
    ]),
    "SIG-B",
  );
  assert.equal(
    readTextThoughtSignature([{ type: "reasoning", text: "hmm" }]),
    undefined,
  );
  assert.equal(readTextThoughtSignature([{ type: "text", text: "x" }]), undefined);
  assert.equal(readTextThoughtSignature(undefined), undefined);

  assert.deepEqual(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "half", thoughtSignature: "SIG" } },
    }),
    { partial: "half", thoughtSignature: "SIG" },
  );
  // An unsigned turn stays unsigned rather than gaining an empty key.
  assert.deepEqual(
    readContinuationRequest({
      custom: { unslothContinuation: { partial: "half" } },
    }),
    { partial: "half" },
  );
});
