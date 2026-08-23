// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  AUTO_CONTINUE_LIMIT,
  autoContinueCount,
  budgetImpliesTruncation,
  incompleteLabel,
  isContinuableContent,
  isRestart,
  joinContinuation,
  modeAllowsContinuation,
  readContinuationRequest,
  readIncompleteInfo,
  readTextThoughtSignature,
  claimAutoContinue,
  recordAutoContinue,
  rejectsAssistantPrefill,
  resetAutoContinue,
  wasAutoContinued,
  shouldAutoContinue,
  shouldAutoContinueMessage,
  resumesExactly,
  stripContinuationOverlap,
} = await import("../src/features/chat/utils/continuation.ts");

const PARTIAL =
  "There are three steps to proofing dough properly. First, warm the bowl and";

test("a token-exact continuation is appended verbatim", () => {
  // A local model's prompt ended inside the partial turn, so the continuation opens
  // on the next word with nothing repeated.
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

test("a live yield is repaired like the terminal one, so Stop saves clean text", () => {
  // Stop reaches no terminal yield: assistant-ui drops whatever the generator yields
  // after the abort, so the last LIVE yield is what persists.
  assert.equal(
    joinContinuation(PARTIAL, "warm the bowl and cover it with"),
    `${PARTIAL} cover it with`,
  );

  // The two joins diverge past the overlap scan: only the restart check reaches back
  // that far, so a live yield runs the same join as the terminal one.
  const long = `${PARTIAL} ${Array.from(
    { length: 12 },
    (_, i) => `Step ${i} is to knead it for ${i} minutes without adding flour.`,
  ).join(" ")}`;
  const restart = `${long} Second, shape the loaf.`;
  assert.ok(long.length > 400);
  assert.equal(joinContinuation(long, restart), restart);
  // The streaming option would have kept both copies.
  assert.equal(
    joinContinuation(long, restart, { streaming: true }),
    `${long}${restart}`,
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

test("an exhausted budget only implies truncation on the MLX route", () => {
  const capped = { maxTokens: 256, completionTokens: 256 };
  assert.equal(budgetImpliesTruncation({ isMlx: true, ...capped }), true);
  // Everyone else reports "length" itself: llama-server's tool loop sums
  // completion_tokens over passes max_tokens caps individually, and transformers tells
  // an answer that ended on its stop token at the cap from one that ran out.
  assert.equal(
    budgetImpliesTruncation({
      isMlx: false,
      maxTokens: 256,
      completionTokens: 260,
    }),
    false,
  );
  assert.equal(budgetImpliesTruncation({ isMlx: false, ...capped }), false);
  // A route that sends no usage, and an answer inside the budget, stay complete.
  assert.equal(
    budgetImpliesTruncation({
      isMlx: true,
      maxTokens: 256,
      completionTokens: undefined,
    }),
    false,
  );
  assert.equal(
    budgetImpliesTruncation({
      isMlx: true,
      maxTokens: 256,
      completionTokens: 255,
    }),
    false,
  );
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
  // The continuation runs as a sibling, so the call and its result are absent from
  // the outbound history and the resumed text would have lost its evidence.
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
  // Anthropic 400s on a trailing assistant message since Claude 4.6; Gemini requires a
  // multiturn request to end in a user turn or a function response; Mistral needs the
  // turn to carry `prefix: true`, which the outbound message type has no room for.
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
  // gates, but the resumed run regenerates the whole clip.
  assert.equal(
    modeAllowsContinuation({ ...plain, audioOutputModel: true }),
    false,
  );
  // Research armed after the cut: the run replaces the partial with its report.
  assert.equal(
    modeAllowsContinuation({ ...plain, deepResearchArmed: true }),
    false,
  );
});

test("the overlap repair can eat a legitimate repeat, so local output skips it", () => {
  // A local backend resumes at the exact token boundary, so repairing its output would
  // delete a phrase the model meant to write. Hence external providers only.
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
  // The sibling run drops the original assistant message, so the signature travels
  // with the partial or the history goes back to Gemini unsigned.
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

test("only the servers sent the flags resume exactly", () => {
  // The backend forwards the continuation flags to these two only, so only they skip
  // the lossy overlap repair.
  for (const providerType of ["vllm", "llama_cpp"]) {
    assert.equal(resumesExactly(providerType), true, providerType);
  }
  // Ollama and any user-supplied base_url get no flags, so they may still restart.
  for (const providerType of [
    "ollama",
    "custom",
    "openai",
    "anthropic",
    undefined,
  ]) {
    assert.equal(resumesExactly(providerType), false, String(providerType));
  }
});

test("citations appended for display do not block Continue", () => {
  // Sources are attached at finalization and never replayed, exactly like reasoning.
  assert.equal(
    isContinuableContent([
      { type: "text", text: "The answer begins" },
      { type: "source", sourceType: "url", id: "s1", url: "https://x" },
    ]),
    true,
  );
});

test("a Max Tokens cut resumes on its own", () => {
  resetAutoContinue();
  // Not a decision the user made: the reply ran out of room mid-sentence, and asking
  // whether to finish it is asking a question with one sensible answer.
  assert.equal(shouldAutoContinue("length", "parent-1"), true);
});

test("pressing Stop is never undone by an automatic resume", () => {
  resetAutoContinue();
  // The one case where the user HAS decided. Resuming would restart what they stopped.
  assert.equal(shouldAutoContinue("cancelled", "parent-1"), false);
});

test("a dropped connection still asks, rather than retrying silently", () => {
  resetAutoContinue();
  // A silent retry here hides a broken link behind what looks like a slow answer.
  assert.equal(shouldAutoContinue("interrupted", "parent-1"), false);
});

test("automatic resumes are bounded, then the bar comes back", () => {
  resetAutoContinue();
  // A model that will not stop would otherwise loop forever, and every round grows the
  // transcript and drives compaction harder.
  for (let round = 0; round < AUTO_CONTINUE_LIMIT; round += 1) {
    assert.equal(shouldAutoContinue("length", "parent-1"), true);
    recordAutoContinue("parent-1");
  }
  assert.equal(autoContinueCount("parent-1"), AUTO_CONTINUE_LIMIT);
  assert.equal(shouldAutoContinue("length", "parent-1"), false);
});

test("the budget is per turn, so a later turn is not punished for an earlier one", () => {
  resetAutoContinue();
  for (let round = 0; round < AUTO_CONTINUE_LIMIT; round += 1) {
    recordAutoContinue("parent-1");
  }
  assert.equal(shouldAutoContinue("length", "parent-1"), false);
  assert.equal(shouldAutoContinue("length", "parent-2"), true);
});

test("the count is keyed on the parent, which every round of one turn shares", () => {
  resetAutoContinue();
  // A continuation runs as a SIBLING, so each round has a new message id. Keying on that
  // would reset the counter every round and the limit would never be reached.
  recordAutoContinue("parent-1");
  recordAutoContinue("parent-1");
  assert.equal(autoContinueCount("parent-1"), 2);
});

test("a turn with no parent is never resumed automatically", () => {
  resetAutoContinue();
  // The very first message has nothing to hang a sibling off, so there is no stable key
  // to count against and an unbounded loop is the failure mode.
  assert.equal(shouldAutoContinue("length", null), false);
});

test("a turn whose own fit was refused is never resumed automatically", () => {
  resetAutoContinue();
  // Resuming replays the partial as the final assistant turn, which the fit protects, so
  // the next round sends a partial that is only ever longer. Observed at a 4,864-token
  // context: three automatic rounds, each refused identically.
  assert.equal(
    shouldAutoContinue("length", "parent-1", { fits: false }),
    false,
  );
});

test("a turn whose fit only missed the reply reserve is not resumed either", () => {
  resetAutoContinue();
  // A rescue reports `fits: false` too, and it is just as unresumable. The rescue branch
  // is reached only once the eviction loop has run out of eligible turns, so its prompt
  // is already the floor; the continuation then replays the partial as the final
  // assistant turn, which the fit protects, asking for MORE prompt against the same
  // window. Measured on the 460-of-500-token rescue: a 10-character partial already
  // refuses and an 80-character one overflows the window.
  assert.equal(
    shouldAutoContinue("length", "parent-1", { fits: false }),
    false,
  );
});

test("a partial that already fills the budget is not resumed", () => {
  resetAutoContinue();
  // 3,217 tokens of partial against a 3,648-token target left no room for the system turn
  // and the carried-forward block, so the request was irreducible before it was sent.
  assert.equal(
    shouldAutoContinue("length", "parent-1", {
      partialTokens: 3648,
      promptTarget: 3648,
    }),
    false,
  );
  // Comfortably inside the budget still resumes.
  assert.equal(
    shouldAutoContinue("length", "parent-1", {
      partialTokens: 400,
      promptTarget: 3648,
    }),
    true,
  );
});

test("an unknown fit does not block resuming", () => {
  resetAutoContinue();
  // A turn that never truncated carries no metadata, and that is the ordinary case.
  assert.equal(shouldAutoContinue("length", "parent-1", {}), true);
  assert.equal(
    shouldAutoContinue("length", "parent-1", { fits: true }),
    true,
  );
});

test("a message is claimed for automatic continuation exactly once", () => {
  resetAutoContinue();
  assert.equal(claimAutoContinue("m1"), true);
  assert.equal(claimAutoContinue("m1"), false);
  assert.equal(claimAutoContinue("m1"), false);
});

test("the claim survives a remount, which a component ref did not", () => {
  // Leave the chat with a truncated branch selected and come back: a ref was fresh
  // while the parent still had budget, so the effect fired again and created another
  // sibling and another paid provider request.
  resetAutoContinue();
  assert.equal(claimAutoContinue("m1"), true);
  assert.equal(shouldAutoContinue("length", "parent-1"), true);
  assert.equal(claimAutoContinue("m1"), false);
});

test("claims are tracked per message", () => {
  resetAutoContinue();
  assert.equal(claimAutoContinue("m1"), true);
  assert.equal(claimAutoContinue("m2"), true);
  assert.equal(claimAutoContinue("m1"), false);
});

test("a missing message id is refused rather than claimed", () => {
  resetAutoContinue();
  assert.equal(claimAutoContinue(null), false);
  assert.equal(claimAutoContinue(undefined), false);
  assert.equal(claimAutoContinue(""), false);
});

test("a claim is reported and cleared by a full reset", () => {
  resetAutoContinue();
  assert.equal(wasAutoContinued("m1"), false);
  claimAutoContinue("m1");
  assert.equal(wasAutoContinued("m1"), true);
  resetAutoContinue();
  assert.equal(wasAutoContinued("m1"), false);
  assert.equal(claimAutoContinue("m1"), true);
});

test("a message already claimed stops reporting itself as continuing", () => {
  resetAutoContinue();
  // The turn that fires it: nothing has claimed the message yet.
  assert.equal(shouldAutoContinueMessage("m1", "length", "parent-1"), true);
  claimAutoContinue("m1");
  recordAutoContinue("parent-1");

  // Back on the truncated branch, whether through the branch picker or by returning to
  // the chat: `claimAutoContinue` refuses the run, so the turn's own budget still saying
  // yes would leave a spinner nothing is answering, over a hidden manual Continue button.
  assert.equal(shouldAutoContinue("length", "parent-1"), true);
  assert.equal(shouldAutoContinueMessage("m1", "length", "parent-1"), false);
});

test("a claim on one message does not silence another", () => {
  resetAutoContinue();
  claimAutoContinue("m1");
  // The next round of the same turn is a new message with budget left, and continues.
  recordAutoContinue("parent-1");
  assert.equal(shouldAutoContinueMessage("m2", "length", "parent-1"), true);
});

test("a claimed message still honours the gates the turn itself fails", () => {
  resetAutoContinue();
  // Nothing about the claim resurrects a cut that was never automatic in the first place.
  assert.equal(shouldAutoContinueMessage("m1", "cancelled", "parent-1"), false);
  assert.equal(
    shouldAutoContinueMessage("m2", "length", "parent-1", { fits: false }),
    false,
  );
});
