// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  AUTO_CONTINUE_CONTINUED_TTL_MS,
  AUTO_CONTINUE_LEASE_KEY,
  AUTO_CONTINUE_LEASE_RENEW_MS,
  AUTO_CONTINUE_LEASE_TTL_MS,
  AUTO_CONTINUE_LIMIT,
  autoContinueCount,
  createAutoContinueLeaseKeeper,
  createAutoContinueTab,
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

const { createImageGateRunOwner, isImageGateRunOnly } = await import(
  "../src/features/chat/utils/image-input-support.ts"
);

const { issuedRunFrom } = await import(
  "../src/features/chat/utils/auto-continue-issued-run.ts"
);

const PARTIAL =
  "There are three steps to proofing dough properly. First, warm the bowl and";

/**
 * The runtime a claim belongs to. One Thread is one holder; compare mode has two, which is
 * what the two-pane test below stands up.
 */
const PANE = "pane-1";

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
  // A rescue reports `fits: false` too and is just as unresumable: it is reached only
  // once eviction ran out of eligible turns, so its prompt is already the floor, while
  // the continuation replays the partial as the final assistant turn, which the fit
  // protects. Measured on the 460-of-500-token rescue, a 10-character partial refuses.
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

test("a message is claimed for automatic continuation exactly once", async () => {
  resetAutoContinue();
  assert.equal(await claimAutoContinue("m1", PANE), "started");
  assert.equal(await claimAutoContinue("m1", PANE), "skipped");
  assert.equal(await claimAutoContinue("m1", PANE), "skipped");
});

test("the claim survives a remount, which a component ref did not", async () => {
  // Leave the chat with a truncated branch selected and come back: a ref was fresh
  // while the parent still had budget, so the effect fired again and created another
  // sibling and another paid provider request.
  resetAutoContinue();
  assert.equal(await claimAutoContinue("m1", PANE), "started");
  assert.equal(shouldAutoContinue("length", "parent-1"), true);
  assert.equal(await claimAutoContinue("m1", PANE), "skipped");
});

test("claims are tracked per message", async () => {
  resetAutoContinue();
  assert.equal(await claimAutoContinue("m1", PANE), "started");
  assert.equal(await claimAutoContinue("m2", PANE), "started");
  assert.equal(await claimAutoContinue("m1", PANE), "skipped");
});

test("a missing message id is refused rather than claimed", async () => {
  resetAutoContinue();
  assert.equal(await claimAutoContinue(null, PANE), "skipped");
  assert.equal(await claimAutoContinue(undefined, PANE), "skipped");
  assert.equal(await claimAutoContinue("", PANE), "skipped");
});

test("a claim is reported and cleared by a full reset", async () => {
  resetAutoContinue();
  assert.equal(wasAutoContinued("m1"), false);
  await claimAutoContinue("m1", PANE);
  assert.equal(wasAutoContinued("m1"), true);
  resetAutoContinue();
  assert.equal(wasAutoContinued("m1"), false);
  assert.equal(await claimAutoContinue("m1", PANE), "started");
});

test("a message already claimed stops reporting itself as continuing", async () => {
  resetAutoContinue();
  // The turn that fires it: nothing has claimed the message yet.
  assert.equal(shouldAutoContinueMessage("m1", "length", "parent-1"), true);
  await claimAutoContinue("m1", PANE);
  recordAutoContinue("parent-1");

  // Back on the truncated branch, whether through the branch picker or by returning to
  // the chat: `claimAutoContinue` refuses the run, so the turn's own budget still saying
  // yes would leave a spinner nothing is answering, over a hidden manual Continue button.
  assert.equal(shouldAutoContinue("length", "parent-1"), true);
  assert.equal(shouldAutoContinueMessage("m1", "length", "parent-1"), false);
});

test("a claim on one message does not silence another", async () => {
  resetAutoContinue();
  await claimAutoContinue("m1", PANE);
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

// --- cross-tab claim ------------------------------------------------------------------
// The module claim above is per TAB: each one loads its own copy of the module with its
// own empty set. Open the same saved thread twice with a `length` reply last and both
// tabs claim it, both start a run, and the user pays for two continuations and gets two
// sibling branches. A tab here is a second `createAutoContinueTab` over one shared store
// and one shared lock manager, which is what two browser tabs are.

/**
 * An in-memory `localStorage`.
 *
 * `onSet` runs after a write and `onGet` after a read has taken its value, which is where
 * another tab is interleaved: both hooks land in the middle of a read-modify-write, which
 * is the sequence localStorage does not make atomic.
 */
function storageFake(
  onSet?: (store: Map<string, string>) => void,
  onGet?: (store: Map<string, string>) => void,
) {
  const store = new Map<string, string>();
  return {
    store,
    storage: {
      getItem: (key: string) => {
        const value = store.get(key) ?? null;
        // After the value is taken, so the caller carries on with the snapshot it read
        // and whatever the other tab does next is invisible to it. That staleness is the
        // whole bug.
        onGet?.(store);
        return value;
      },
      setItem: (key: string, value: string) => {
        store.set(key, value);
        onSet?.(store);
      },
      removeItem: (key: string) => {
        store.delete(key);
      },
    },
  };
}

/** A win, in the pre-fix boolean shape as well as the current one. */
function startedARun(outcome: unknown): boolean {
  return outcome === "started" || outcome === true;
}

/**
 * A `navigator.locks` stand-in: one queue per name, exclusive, FIFO.
 *
 * The property the real API gives and a bare read-modify-write does not: a second request
 * for a held name does not run until the first has returned.
 */
function lockManagerFake() {
  const tails = new Map<string, Promise<unknown>>();
  return {
    request<T>(name: string, callback: () => T | Promise<T>): Promise<T> {
      const tail = tails.get(name) ?? Promise.resolve();
      const run = tail.then(() => callback());
      // The queue advances on settle, so one throwing holder cannot wedge the name.
      tails.set(
        name,
        run.then(
          () => undefined,
          () => undefined,
        ),
      );
      return run;
    },
  };
}

test("a second tab with the same thread open does not start its own continuation", async () => {
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const first = createAutoContinueTab({ storage, locks });
  const second = createAutoContinueTab({ storage, locks });
  const now = 1_000;

  assert.equal(await first.claim("m1", { now }), "started");
  // Fresh module scope, empty claim set, same truncated reply on screen. Before the lease
  // this said yes and the second tab fired a second paid request.
  assert.equal(await second.claim("m1", { now }), "held-elsewhere");
  // And it reports the message as being continued, so it shows no spinner of its own.
  assert.equal(second.claimed("m1", { now }), true);
});

test("a lease covers only the message it was taken for", async () => {
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const first = createAutoContinueTab({ storage, locks });
  const second = createAutoContinueTab({ storage, locks });
  const now = 1_000;

  assert.equal(await first.claim("m1", { now }), "started");
  // The next round of the same turn is a different message and is nobody's yet.
  assert.equal(await second.claim("m2", { now }), "started");
});

test("two tabs claiming at once leave exactly one winner", async () => {
  // ITEM A. Write-then-read-back is not a compare-and-swap: both tabs read the slot free,
  // both write, and each verifies the token it just wrote if the two verifications happen
  // before the two writes interleave -- so both start a run. Individual localStorage
  // operations are atomic; a read-modify-write across statements is not, and the storage
  // mutex that once serialized such sequences is no longer in the spec.
  //
  // Started together, with no await between them, which is what two tabs reaching the
  // effect in the same instant are. The lock is what makes the second one wait for the
  // first, see the lease it wrote, and stand down.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const first = createAutoContinueTab({ storage, locks });
  const second = createAutoContinueTab({ storage, locks });
  const now = 1_000;

  const outcomes = await Promise.all([
    first.claim("m1", { now }),
    second.claim("m1", { now }),
  ]);
  assert.equal(outcomes.filter(startedARun).length, 1);
  assert.equal(
    outcomes.filter((outcome) => outcome === "held-elsewhere").length,
    1,
  );
});

test("a tab that reads the slot free does not win it behind another tab's back", async () => {
  // ITEM A, at the exact interleaving that survives a write-then-read-back: the second
  // tab claims in full between the first tab READING the slot free and writing to it. The
  // first tab then writes over a lease it never saw and reads its own token back, so both
  // tabs believe they won and the user pays twice. Nothing about the sequence is atomic;
  // only holding a lock across it is.
  const now = 1_000;
  let intruder: (() => unknown) | null = null;
  let secondOutcome: unknown;
  let reads = 0;
  const { storage } = storageFake(undefined, () => {
    reads += 1;
    // The second read is the one the write is about to be based on: a free check, then
    // the read-modify-write itself. Slipping the other tab in there is what leaves the
    // first tab holding a snapshot that is already out of date.
    if (reads !== 2) {
      return;
    }
    const run = intruder;
    intruder = null;
    if (run) {
      secondOutcome = run();
    }
  });
  const locks = lockManagerFake();
  const first = createAutoContinueTab({ storage, locks });
  const second = createAutoContinueTab({ storage, locks });

  intruder = () => second.claim("m1", { now });
  const outcomes = [await first.claim("m1", { now }), await secondOutcome];
  assert.equal(
    outcomes.filter(startedARun).length,
    1,
    "exactly one tab may start the run",
  );
});

test("a claim that loses says so, rather than just failing", async () => {
  // ITEM B. The two ways of not starting a run need opposite things on screen, so they
  // cannot both be a bare false. `held-elsewhere` is the loser, and is what puts this
  // tab's manual Continue button back in place of a spinner for a run it never owned.
  // `skipped` is this tab's own second call -- a StrictMode replay, or a claim already in
  // flight -- where the run is coming and nothing should move.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const winner = createAutoContinueTab({ storage, locks });
  const loser = createAutoContinueTab({ storage, locks });
  const now = 1_000;

  assert.equal(startedARun(await winner.claim("m1", { now })), true);
  const lost = await loser.claim("m1", { now });
  // The winner's own replay is not a loss and must not repaint anything, so the two
  // cannot be the same answer. One bare false for both is what left the losing tab
  // spinning: its effect returned early, changed no state, and hid its own button.
  const replay = await winner.claim("m1", { now });
  assert.notEqual(lost, replay);
  assert.equal(lost, "held-elsewhere");
  assert.equal(replay, "skipped");
  // Nor is a second call while the first is still inside the lock.
  const inFlight = winner.claim("m2", { now });
  assert.equal(await winner.claim("m2", { now }), "skipped");
  assert.equal(await inFlight, "started");
});

test("two tabs writing in the same tick leave exactly one winner", async () => {
  // The same race with no lock manager to settle it, which is what an older browser or an
  // embedded webview gets. The write-then-read-back is all that is left: a record that
  // appears between this tab's write and its read-back means the tab does not find its
  // own token, and stands down.
  const now = 1_000;
  let interleave = true;
  const { storage } = storageFake((store) => {
    if (!interleave) {
      return;
    }
    interleave = false;
    store.set(
      AUTO_CONTINUE_LEASE_KEY,
      JSON.stringify({
        m1: { token: "other-tab", expires: now + AUTO_CONTINUE_LEASE_TTL_MS },
      }),
    );
  });
  const first = createAutoContinueTab({ storage, locks: null });

  assert.equal(await first.claim("m1", { now }), "held-elsewhere");
  // Nothing recorded locally either, so once that lease lapses this tab can take over.
  assert.equal(
    await first.claim("m1", { now: now + AUTO_CONTINUE_LEASE_TTL_MS + 1 }),
    "started",
  );
});

test("a browser with no lock manager still continues", async () => {
  const { storage } = storageFake();
  const only = createAutoContinueTab({ storage, locks: null });
  assert.equal(await only.claim("m1"), "started");
  assert.equal(await only.claim("m1"), "skipped");
});

test("a lock manager that refuses the request does not block the claim", async () => {
  const { storage } = storageFake();
  const angryLocks = {
    request<T>(): Promise<T> {
      return Promise.reject(new Error("NotSupportedError"));
    },
  };
  const tab = createAutoContinueTab({ storage, locks: angryLocks });
  // Degraded to the read-back, not to a refusal: the user is waiting for this run.
  assert.equal(await tab.claim("m1"), "started");
  assert.equal(
    await createAutoContinueTab({ storage }).claim("m1"),
    "held-elsewhere",
  );
});

test("a tab with no storage keeps the claim it always had", async () => {
  // Private mode, an embedded webview, or the test runner: no seam to share. Falling back
  // to module scope is what shipped before the lease, and it must never refuse a
  // continuation the single tab in front of the user is waiting for.
  const only = createAutoContinueTab({ storage: null });
  assert.equal(await only.claim("m1"), "started");
  assert.equal(await only.claim("m1"), "skipped");
  assert.equal(only.claimed("m1"), true);
});

test("storage that throws is no worse than no storage", async () => {
  // A quota-exceeded write, or a getItem that throws before it returns anything.
  const angry = {
    getItem(): string | null {
      throw new Error("SecurityError");
    },
    setItem(): void {
      throw new Error("QuotaExceededError");
    },
    removeItem(): void {
      throw new Error("SecurityError");
    },
  };
  const tab = createAutoContinueTab({ storage: angry });
  assert.equal(await tab.claim("m1"), "started");
  assert.equal(await tab.claim("m1"), "skipped");
  // A second tab cannot see through a broken seam either, so it behaves as it did before
  // the lease: module-only. Duplicates are not made worse, and nothing crashes.
  assert.equal(
    await createAutoContinueTab({ storage: angry }).claim("m1"),
    "started",
  );
});

test("a lease lapses, so a tab that died mid-run does not wedge the message", async () => {
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const winner = createAutoContinueTab({ storage, locks });
  const later = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(await winner.claim("m1", { now: start }), "started");
  // Still inside the lease: the holder is presumed alive and nobody else touches it.
  assert.equal(
    await later.claim("m1", { now: start + AUTO_CONTINUE_LEASE_TTL_MS - 1 }),
    "held-elsewhere",
  );
  // Past it, with no renewal in between: a permanent flag would have left this message
  // unresumable for the life of the profile, because the tab that owned it is gone and
  // can never clear it.
  assert.equal(
    await later.claim("m1", { now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1 }),
    "started",
  );
});

test("a running continuation keeps its lease past the TTL", async () => {
  // ITEM C. A local model on a large Max Tokens can generate for longer than the TTL, and
  // an unrenewed lease would hand its message to the next tab to open, mid-stream. The
  // holder renews for as long as its run is live, so the TTL only ever measures silence.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const running = createAutoContinueTab({ storage, locks });
  const other = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(
    startedARun(await running.claim("m1", { now: start, holder: PANE })),
    true,
  );
  // Three renewals at the interval the keeper uses, still inside the run.
  for (let tick = 1; tick <= 3; tick += 1) {
    await running.renew("m1", PANE, {
      now: start + tick * AUTO_CONTINUE_LEASE_RENEW_MS,
    });
  }
  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(await other.claim("m1", { now: past }), "held-elsewhere");
});

test("a finished run leaves the message continued, not free again", async () => {
  // Released on any terminal state, so the full TTL is left to mean one thing: a crash.
  // Marked done rather than handed back, because the tab that did not win never learns that
  // the sibling was written -- no chat-history event crosses tabs -- so a record that simply
  // lapsed handed the message to a stale tab and bought the same continuation twice.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const running = createAutoContinueTab({ storage, locks });
  const other = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(
    await running.claim("m1", { now: start, holder: PANE }),
    "started",
  );
  await running.release("m1", PANE, { now: start });
  assert.equal(
    await other.claim("m1", { now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1 }),
    "held-elsewhere",
  );
  assert.equal(
    await other.claim("m1", {
      now: start + AUTO_CONTINUE_CONTINUED_TTL_MS - 1,
    }),
    "held-elsewhere",
  );
  // And a release holds nothing, so a later renewal cannot resurrect it as a live lease.
  await running.renew("m1", PANE, { now: start + AUTO_CONTINUE_LEASE_TTL_MS });
  assert.equal(
    await createAutoContinueTab({ storage, locks }).claim("m1", {
      now: start + AUTO_CONTINUE_LEASE_TTL_MS + 2,
    }),
    "held-elsewhere",
  );
  // Bounded, not permanent: the record is pruned like any other once it is old enough that
  // no tab can still be holding the pre-continuation branch in memory.
  assert.equal(
    await createAutoContinueTab({ storage, locks }).claim("m1", {
      now: start + AUTO_CONTINUE_CONTINUED_TTL_MS + 1,
    }),
    "started",
  );
});

test("a stale tab cannot take a message back once it has been continued", async () => {
  // The case the settle window left open. The second tab's render-time check refuses before
  // its effect ever runs, so it records nothing locally and holds no state saying it lost;
  // its `continued` set stays empty. Remount that bar -- leaving the chat and coming back is
  // enough, which is the same remount the module claim exists for -- and the only thing
  // between it and a second paid request is what storage remembers. Nothing refreshes its
  // in-memory history in the meantime: `notifyChatHistoryUpdated` dispatches a same-window
  // event, no chat key has a `storage` listener, and stored messages are re-read only by
  // `history.load`.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const winner = createAutoContinueTab({ storage, locks });
  const stale = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(await winner.claim("m1", { now: start, holder: PANE }), "started");
  // The stale tab renders the truncated reply while the winner is running: refused, and
  // nothing about that refusal is written down on its side.
  assert.equal(await stale.claim("m1", { now: start + 1 }), "held-elsewhere");
  // The winner finishes and writes the sibling. The stale tab still shows the partial.
  await winner.release("m1", PANE, { now: start + 60_000 });
  // Its bar remounts, long after any settle window would have passed.
  assert.equal(
    stale.claimed("m1", { now: start + 60_000 + AUTO_CONTINUE_LEASE_TTL_MS }),
    true,
  );
  assert.equal(
    await stale.claim("m1", {
      now: start + 60_000 + AUTO_CONTINUE_LEASE_TTL_MS,
    }),
    "held-elsewhere",
  );
});

test("a tab that died mid-run still hands the message back", async () => {
  // The other half of the same rule, and the reason a permanent flag is wrong: a record is
  // only marked done by a run that reached a terminal state. A tab killed without cleanup
  // renews nothing and marks nothing, so its lease lapses and the message is claimable
  // again. (Web Locks are released with the context too, so nothing is wedged there either.)
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const killed = createAutoContinueTab({ storage, locks });
  const survivor = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(await killed.claim("m1", { now: start, holder: PANE }), "started");
  // No release, no renewal: the tab is gone.
  assert.equal(
    await survivor.claim("m1", { now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1 }),
    "started",
  );
});

/**
 * Two rounds of one turn, in the order the app produces them.
 *
 * A round ends on another Max Tokens cut, so the bar under the new reply claims the next
 * message while the thread is still winding the finished run down: React runs child
 * effects before parent ones, so the claim lands before the keeper observes `isRunning`
 * going false. `release` therefore has to name the message whose run ended, and nothing
 * wider: the same holder owns both.
 */
async function sequentialRounds(
  locks: ReturnType<typeof lockManagerFake> | null,
) {
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(
    startedARun(await tab.claim("round-1", { now: start, holder: PANE })),
    true,
  );
  // The next round, claimed before the finished one is given back.
  assert.equal(
    startedARun(await tab.claim("round-2", { now: start, holder: PANE })),
    true,
  );
  await tab.release("round-1", PANE, { now: start });

  // The second round is the live one, and its keeper is still renewing it.
  for (let tick = 1; tick <= 3; tick += 1) {
    await tab.renew("round-2", PANE, {
      now: start + tick * AUTO_CONTINUE_LEASE_RENEW_MS,
    });
  }
  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(
    await otherTab.claim("round-2", { now: past }),
    "held-elsewhere",
    "the round that just started keeps its lease",
  );
  // The round that did finish is marked continued, and stays that way.
  assert.equal(
    await otherTab.claim("round-1", {
      now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1,
    }),
    "held-elsewhere",
  );
}

test("the next round keeps its lease when the last one is released", async () => {
  // No lock manager, where the claim resolves soonest and so lands earliest.
  await sequentialRounds(null);
});

test("the next round keeps its lease with a lock manager in the way", async () => {
  // And with one, because the effect ordering is not guaranteed in our favour there
  // either -- the fix must not depend on which path ran.
  await sequentialRounds(lockManagerFake());
});

/**
 * The per-thread run signal the keeper reads, with no notion of what is on screen.
 *
 * Selection is deliberately absent: it is not an input to a lease's lifetime, and the whole
 * point of the case below is that changing it changes nothing.
 */
function runSignalFake(running: Set<string>) {
  const listeners = new Set<() => void>();
  return {
    signal: {
      isRunning: (threadId: string) => running.has(threadId),
      subscribe: (onChange: () => void) => {
        listeners.add(onChange);
        return () => listeners.delete(onChange);
      },
    },
    /** The store told everyone something changed. */
    change: () => {
      for (const listener of [...listeners]) {
        listener();
      }
    },
  };
}

test("a run on a background thread keeps its lease while the user reads another chat", async () => {
  // Switching chats does not stop a generation: the chat view is keyed by project, so the
  // provider is not remounted and the run keeps streaming on the thread the user left. A
  // lease whose lifetime was read off the SELECTED thread was released the moment they
  // looked away, and one settle window later a second tab could claim a message that was
  // still being written.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];

  // Thread A is generating; thread B, which the user is about to open, is idle.
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId, holder, now) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId, holder, now) => {
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  assert.equal(
    startedARun(await tab.claim("a-m1", { now: start, holder: "thread-A" })),
    true,
  );
  keeper.hold("a-m1", "thread-A");
  // The continuation starts on thread A.
  running.add("thread-A");
  runs.change();

  // The user opens idle thread B. Nothing about thread A changed, so nothing here does.
  for (let tick = 1; tick <= 5; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);

  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(
    await otherTab.claim("a-m1", { now: past }),
    "held-elsewhere",
    "the background run keeps the message it is still writing",
  );

  // Thread A's own run ends. That, and only that, gives the lease back.
  clock = past;
  running.delete("thread-A");
  runs.change();
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("a-m1", {
      now: past + AUTO_CONTINUE_LEASE_TTL_MS + 1,
    }),
    "held-elsewhere",
    "the message was continued, so nobody continues it again",
  );
  assert.equal(keeper.held(), 0);
});

test("a hold waits for its own run, not the one already in flight", async () => {
  // The next round is claimed while the finished one is still winding down, so a hold that
  // armed on "the thread is busy" would arm on its predecessor's run and be released the
  // moment THAT one ended, mid-stream.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const otherTab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];
  const running = new Set<string>(["thread-A"]);
  const runs = runSignalFake(running);
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId, holder, now) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId, holder, now) => {
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  await tab.claim("round-2", { now: start, holder: "thread-A" });
  keeper.hold("round-2", "thread-A");

  // The previous round ends, and the next one starts.
  running.delete("thread-A");
  runs.change();
  running.add("thread-A");
  runs.change();
  for (let tick = 1; tick <= 5; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);
  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(await otherTab.claim("round-2", { now: past }), "held-elsewhere");
});

test("a hold keeps its lease while its run is still in preflight", async () => {
  // The adapter does a lot before the run reaches `runningByThreadId`: it awaits this
  // thread's own settings pairing, which alone waits up to 30 seconds, and then
  // `waitForModelReady`, which polls every 500ms for as long as a model is loading -- so a
  // tab opened on a truncated reply while a large local model loads can sit in preflight for
  // minutes. A hold dropped on a fixed arming timeout stopped renewing in the middle of
  // that, its lease lapsed, and another tab claimed a message that was about to stream.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const otherTab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId, holder, now) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: () => {
      assert.fail("a hold that never armed has nothing to give back");
    },
    now: () => clock,
  });

  await tab.claim("a-m1", { now: start, holder: "thread-A" });
  keeper.hold("a-m1", "thread-A");

  // Four minutes of preflight, renewed at the keeper's own interval throughout.
  for (let tick = 1; tick <= 8; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);
  assert.equal(keeper.held(), 1, "the hold is still expecting its run");
  assert.equal(
    await otherTab.claim("a-m1", { now: clock }),
    "held-elsewhere",
    "nobody else may take a message this tab is still about to continue",
  );

  // The run finally starts, and the hold arms on it rather than on a predecessor.
  running.add("thread-A");
  runs.change();
  clock += AUTO_CONTINUE_LEASE_RENEW_MS;
  keeper.tick();
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("a-m1", {
      now: clock + AUTO_CONTINUE_LEASE_TTL_MS - 1,
    }),
    "held-elsewhere",
  );
});

test("one compare pane finishing does not release the other pane's lease", async () => {
  // Compare mode mounts two Thread runtimes in ONE tab, each with its own thread and its
  // own run, and either can be resuming a truncated reply while the other is idle. A
  // release that restamped every lease the tab owns and then dropped the lot would end the
  // hold on the pane still generating: its renewals would find nothing held, and one settle
  // window later another tab could claim its message and pay for a second run.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const compareTab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const base = "pane-base";
  const lora = "pane-lora";
  const start = 1_000;

  assert.equal(
    startedARun(
      await compareTab.claim("base-m1", { now: start, holder: base }),
    ),
    true,
  );
  assert.equal(
    startedARun(
      await compareTab.claim("lora-m1", { now: start, holder: lora }),
    ),
    true,
  );

  // The base pane finishes first. Only its own lease goes back.
  await compareTab.release("base-m1", base, { now: start });

  // The lora pane is still generating, and its keeper is still renewing.
  for (let tick = 1; tick <= 3; tick += 1) {
    await compareTab.renew("lora-m1", lora, {
      now: start + tick * AUTO_CONTINUE_LEASE_RENEW_MS,
    });
  }
  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(
    await otherTab.claim("lora-m1", { now: past }),
    "held-elsewhere",
    "the still-running pane keeps its message",
  );
  // And the pane that did finish leaves its message continued.
  assert.equal(
    await otherTab.claim("base-m1", {
      now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1,
    }),
    "held-elsewhere",
  );
});

test("a renewal touches this tab's leases and nobody else's", async () => {
  const { storage, store } = storageFake();
  const locks = lockManagerFake();
  const mine = createAutoContinueTab({ storage, locks });
  const theirs = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  await mine.claim("m1", { now: start, holder: PANE });
  await theirs.claim("m2", { now: start, holder: PANE });
  await mine.renew("m1", PANE, { now: start + AUTO_CONTINUE_LEASE_RENEW_MS });

  const leases = JSON.parse(store.get(AUTO_CONTINUE_LEASE_KEY) ?? "{}");
  assert.equal(
    leases.m1.expires,
    start + AUTO_CONTINUE_LEASE_RENEW_MS + AUTO_CONTINUE_LEASE_TTL_MS,
  );
  assert.equal(leases.m2.expires, start + AUTO_CONTINUE_LEASE_TTL_MS);
});

test("lapsed leases are pruned rather than accumulating", async () => {
  const { storage, store } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  await tab.claim("m1", { now: start });
  await createAutoContinueTab({ storage, locks }).claim("m2", {
    now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1,
  });
  const leases = JSON.parse(store.get(AUTO_CONTINUE_LEASE_KEY) ?? "{}");
  assert.deepEqual(Object.keys(leases), ["m2"]);
});

test("reloading one tab does not fire a second continuation", async () => {
  // What stops this today is branch selection: the continuation runs as a sibling and
  // becomes the selected branch, so the truncated reply is no longer last and the effect
  // never asks. The lease is the belt to that pair of braces -- a reload lands on the
  // truncated branch, its module scope is empty, and storage is the only thing that
  // remembers. Nothing here changes what selection already refuses.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const before = createAutoContinueTab({ storage, locks });
  const start = 1_000;
  assert.equal(await before.claim("m1", { now: start }), "started");

  // The reload: same tab, same profile, brand new module scope.
  const after = createAutoContinueTab({ storage, locks });
  assert.equal(
    await after.claim("m1", { now: start + 5_000 }),
    "held-elsewhere",
  );
  assert.equal(after.claimed("m1", { now: start + 5_000 }), true);
});

test("a reset gives back this tab's lease and leaves other tabs alone", async () => {
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const mine = createAutoContinueTab({ storage, locks });
  const theirs = createAutoContinueTab({ storage, locks });
  const now = 1_000;

  assert.equal(await mine.claim("m1", { now }), "started");
  mine.reset();
  // Cleared in both places, so "start from zero" means what it did before the lease.
  assert.equal(await mine.claim("m1", { now }), "started");

  assert.equal(await theirs.claim("m2", { now }), "started");
  mine.reset();
  // Not mine to release: a run another tab is driving is still going.
  assert.equal(
    await createAutoContinueTab({ storage, locks }).claim("m2", { now }),
    "held-elsewhere",
  );
});

test("a stored lease survives a full reset by the module, being another tab's", async () => {
  // `resetAutoContinue()` clears the module claim and the round budget. It cannot know
  // about a lease it never wrote, and must not stamp on one.
  const { storage, store } = storageFake();
  const locks = lockManagerFake();
  await createAutoContinueTab({ storage, locks }).claim("m1", { now: 1_000 });
  resetAutoContinue();
  assert.equal(
    await createAutoContinueTab({ storage, locks }).claim("m1", { now: 1_000 }),
    "held-elsewhere",
  );
  assert.equal(store.has(AUTO_CONTINUE_LEASE_KEY), true);
});

test("a malformed lease record is ignored rather than blocking", async () => {
  const { storage, store } = storageFake();
  store.set(AUTO_CONTINUE_LEASE_KEY, "not json at all");
  assert.equal(await createAutoContinueTab({ storage }).claim("m1"), "started");

  const { storage: other, store: otherStore } = storageFake();
  otherStore.set(AUTO_CONTINUE_LEASE_KEY, JSON.stringify({ m1: { token: 7 } }));
  assert.equal(
    await createAutoContinueTab({ storage: other }).claim("m1"),
    "started",
  );
});

test("a hold whose run never starts holds its lease for the life of the tab", async () => {
  // Not a defect on its own: a hold that has not seen its run yet is deliberately renewed
  // rather than timed out, because preflight has no upper bound and a deadline that fired
  // during one lapsed the lease under a run that had since started streaming.
  //
  // It is the reason the bar must not take a hold for a run it never issued. This is what
  // that mistake costs, so the guard in `ContinueMessageBarForLastMessage` below has
  // something concrete to be measured against.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const running = new Set<string>();
  const pending: Promise<unknown>[] = [];
  const start = 1_000;
  let clock = start;
  const keeper = createAutoContinueLeaseKeeper({
    signal: {
      isRunning: (threadId: string) => running.has(threadId),
      subscribe: () => () => {},
    },
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  assert.equal(
    await tab.claim("m1", { now: start, holder: "thread-A" }),
    "started",
  );
  keeper.hold("m1", "thread-A");
  // `thread-A` never runs: the run this hold is waiting for was never issued.
  for (
    let day = 1;
    day <= (3 * 86_400_000) / AUTO_CONTINUE_LEASE_RENEW_MS;
    day += 1
  ) {
    clock = start + day * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);

  assert.equal(keeper.held(), 1, "nothing drops a pending hold, by design");
  assert.equal(
    await otherTab.claim("m1", { now: clock + AUTO_CONTINUE_LEASE_TTL_MS - 1 }),
    "held-elsewhere",
    "three days on, every other tab is still refused this message",
  );
});

test("a claim whose run was never issued is left to lapse, not held", async () => {
  // The bar claims under a Web Lock, so the answer lands a tick or more after the render
  // that asked for it, and `aui.thread()` follows the SELECTION rather than the thread the
  // bar belongs to (`runningByThreadId` exists precisely because "detection survives
  // navigation" and `aui.thread()` does not). Switch chats or branches inside that window
  // and `startContinuation` searches a different thread's messages, finds nothing, and
  // returns without calling `startRun`.
  //
  // Taking the hold anyway is the case above: renewed forever, and every other tab refused
  // the message until this one closes. So the message has to still be there before anything
  // is held. Pinned at the source, since there is no renderer here -- the same way
  // composer-keystroke-subscription-budget.test.ts pins its seams.
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const claimed = thread.indexOf(
    'claimAutoContinue(messageId, runThreadId ?? "")',
  );
  assert.notEqual(claimed, -1, "the claim moved; this test needs rewriting");
  const branch = thread.slice(
    claimed,
    thread.indexOf("held-elsewhere", claimed),
  );

  const guard = branch.search(/messages\.some\(/);
  const hold = branch.indexOf("holdAutoContinueRun(");
  const record = branch.indexOf("recordAutoContinue(");
  const run = branch.indexOf("startContinuation()");
  assert.notEqual(
    guard,
    -1,
    "nothing checks the message is still there before holding",
  );
  assert.notEqual(hold, -1, "the hold is gone; this test needs rewriting");
  assert.ok(
    guard < hold && guard < record && guard < run,
    "the message has to be confirmed present before the lease is held or the round spent",
  );
  // And the guard must leave the claim alone rather than reach for a timer: the lease
  // lapsing on its own TTL is what a tab that closed mid-claim already produces.
  const between = branch.slice(guard, hold);
  assert.match(
    between,
    /\breturn;/,
    "the guard has to stop before the hold, not fall through",
  );
  assert.doesNotMatch(
    between,
    /setTimeout|setInterval/,
    "a deadline here is the arming timeout coming back, which lapses live continuations",
  );
});

test("a losing claim does not follow the row onto the next branch", () => {
  // The bar's "another tab has this one" answer is state, because the claim resolves after
  // the render that asked for it. A bare boolean is wrong state to keep it in: rows are
  // mounted by INDEX, so selecting a different truncated branch at the same index
  // re-renders this component rather than remounting it, and the flag set for the message
  // that lost carried over onto a message nobody has claimed at all -- no automatic
  // continuation for it, for as long as that row lives.
  const rows = readFileSync(
    new URL(
      "../src/components/assistant-ui/progressive-messages.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    rows,
    /<MessageByIndexProvider key=\{index\}/,
    "rows are no longer keyed by index; this test needs rewriting",
  );

  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const start = thread.indexOf("const ContinueMessageBarForLastMessage");
  assert.notEqual(start, -1, "the bar moved; this test needs rewriting");
  const component = thread.slice(
    start,
    thread.indexOf("const WebSearchToolUIConfirmable", start),
  );
  const state =
    /const \[(\w+), (set\w+)\] = useState<string \| null>\(null\)/.exec(
      component,
    );
  assert.ok(
    state,
    "the losing answer is still a bare flag, so it outlives the message it was decided for",
  );
  assert.match(
    component,
    new RegExp(`claimHeldElsewhere =\\s*${state[1]} === messageId`),
    "the flag has to be re-answered against the message on screen now",
  );
  assert.match(
    component,
    new RegExp(`${state[2]}\\(messageId\\)`),
    "the message that lost is what gets remembered",
  );
});

test("a claim taken for a run that was never issued is given back", async () => {
  // The `stillThere` guard leaves the run unstarted, and the claim it took stays in this
  // tab's continued set: every later claim of that message answers "skipped", so the
  // automatic continuation never runs again for a request nobody ever made.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(
    await tab.claim("m1", { now: start, holder: "thread-A" }),
    "started",
  );
  // The branch changed inside the lock's window, so nothing was issued. Roll it back.
  tab.forget("m1");
  // The storage lease is deliberately kept: this tab may still be deciding, and handing
  // the message over now is how two tabs pay for the same continuation.
  assert.equal(
    await otherTab.claim("m1", { now: start + 1 }),
    "held-elsewhere",
    "the rollback must not hand the message to a second tab",
  );
  // Once that lease lapses -- the same record a tab that closed mid-claim leaves behind --
  // the message is continuable again.
  assert.equal(
    await tab.claim("m1", {
      now: start + AUTO_CONTINUE_LEASE_TTL_MS + 1,
      holder: "thread-A",
    }),
    "started",
    "a claim that started nothing may not wedge the message for the life of the tab",
  );
});

test("the bar rolls its claim back when it issues no run", () => {
  // The behaviour above, pinned where it has to be called from: the early return that
  // decided no run would be issued.
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const claimed = thread.indexOf(
    'claimAutoContinue(messageId, runThreadId ?? "")',
  );
  assert.notEqual(claimed, -1, "the claim moved; this test needs rewriting");
  const branch = thread.slice(
    claimed,
    thread.indexOf("held-elsewhere", claimed),
  );
  const guard = branch.search(/messages\.some\(/);
  const hold = branch.indexOf("holdAutoContinueRun(");
  const between = branch.slice(guard, hold);
  assert.match(
    between,
    /forgetAutoContinue\(messageId\)/,
    "the claim survives a run that was never issued, and skips every later one",
  );
});

test("a hold is settled when its run fails before it ever starts", async () => {
  // A preflight failure -- this chat's settings pairing running out, a model that will not
  // load, a connection the adapter refuses -- throws out of `adapter.run` before anything
  // reaches `setThreadRunning(..., true)`, so no `runningByThreadId` transition can ever
  // identify this run as terminal. Without a signal the hold renews its cross-tab lease
  // every 30 seconds for the life of the tab, one leaked hold per failure.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const pending: Promise<unknown>[] = [];
  const start = 1_000;
  let clock = start;
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: () => {
      assert.fail(
        "a run that never started wrote nothing, so nothing may be marked continued",
      );
    },
    now: () => clock,
  });

  assert.equal(
    await tab.claim("m1", { now: start, holder: "thread-A" }),
    "started",
  );
  keeper.hold("m1", "thread-A");
  clock = start + AUTO_CONTINUE_LEASE_RENEW_MS;
  keeper.tick();
  assert.equal(keeper.held(), 1, "a slow preflight is still a live hold");

  // The adapter threw. This is the run's own thread saying so.
  keeper.failed("thread-A");
  assert.equal(
    keeper.held(),
    0,
    "nothing renews a run that failed on its way out",
  );

  const lapsed = clock + AUTO_CONTINUE_LEASE_TTL_MS + 1;
  clock = lapsed;
  keeper.tick();
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("m1", { now: lapsed }),
    "started",
    "the lease lapses, so another tab can recover a message that was never continued",
  );
});

test("a failure on a thread leaves a hold whose run is already streaming alone", async () => {
  // The same event fires for a failure mid-stream, where the run DID reach
  // `runningByThreadId` and the transition to idle is what settles the hold -- with the
  // `done` marker that says the message has been continued, which a preflight failure has
  // no right to write.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const released: string[] = [];
  const pending: Promise<unknown>[] = [];
  const start = 1_000;
  let clock = start;
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      released.push(messageId);
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  await tab.claim("m1", { now: start, holder: "thread-A" });
  keeper.hold("m1", "thread-A");
  running.add("thread-A");
  runs.change();
  clock = start + AUTO_CONTINUE_LEASE_RENEW_MS;
  keeper.tick();

  // The run failed after it had started streaming. Its hold is armed, and armed holds are
  // settled by their own thread going idle, not here.
  keeper.failed("thread-A");
  assert.equal(keeper.held(), 1, "an armed hold is its run's to give back");
  assert.deepEqual(released, []);
  running.delete("thread-A");
  runs.change();
  await Promise.all(pending);
  assert.deepEqual(
    released,
    ["m1"],
    "and the thread going idle is what gives it back",
  );
  assert.equal(keeper.held(), 0);
});

test("the keeper is wired to the failure the adapter already reports", () => {
  // There is exactly one signal for a run that failed on its way out, and it is not a
  // deadline: the adapter wrapper catches everything `adapter.run` throws and announces it
  // per thread. Pinned at both ends, since neither side is exercised by a unit test.
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const wrapper = adapter.slice(adapter.indexOf("yield* adapter.run(args)"));
  assert.match(
    wrapper,
    /catch \(error\) \{[\s\S]*notifyPromptQueueRunFailed\(/,
    "the adapter no longer reports a failed run per thread; this test needs rewriting",
  );

  const wiring = readFileSync(
    new URL(
      "../src/features/chat/utils/auto-continue-run-keeper.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    wiring,
    /PROMPT_QUEUE_RUN_FAILED_EVENT/,
    "nothing settles a hold whose run failed before it started",
  );
  assert.match(
    wiring,
    /keeper\.failed\(/,
    "the failure has to reach the keeper",
  );
  assert.doesNotMatch(
    wiring,
    /setTimeout\(/,
    "a deadline here is the arming timeout coming back, which lapses live continuations",
  );
});

/**
 * The run a hold was taken for, as the keeper sees it: something that settles, once.
 *
 * `runSignalFake` above is the STREAM, which turns true only when tokens are on their way.
 * This is the RUN, pending for the whole preflight however long that is, and settling when
 * that one run ends however it ends -- finished, failed, or cancelled mid-model-load.
 */
function issuedRunFake() {
  const settlers = new Set<() => void>();
  return {
    issued: {
      whenSettled: (onSettled: () => void) => {
        settlers.add(onSettled);
      },
    },
    /** That run ended. */
    settle: () => {
      for (const onSettled of [...settlers]) {
        onSettled();
      }
    },
  };
}

test("a preflight the user stopped gives up its hold instead of keeping it for the tab", async () => {
  // Stop during preflight. The run is aborted, so the adapter wrapper skips its per-thread
  // failure notice ON PURPOSE -- the abort is what was asked for, not a fault to report -- and
  // `runningByThreadId` never moved in either direction, because no token was ever on its way.
  // The hold therefore had nothing to arm on and nothing to settle on, and renewed its lease
  // for the life of the tab while every other tab read that lease as live and refused the
  // message. The run's own promise is the one thing that knows it is over.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const otherTab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const issued = issuedRunFake();
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: () => {
      assert.fail(
        "a run that streamed nothing has continued nothing to record",
      );
    },
    now: () => clock,
  });

  await tab.claim("m1", { now: start, holder: "thread-A" });
  // Taken on the line BEFORE the run is started, which is why the two are separate calls.
  keeper.hold("m1", "thread-A");
  keeper.settleOn("m1", "thread-A", issued.issued);

  // An ordinary preflight: the promise is pending, so nothing here ends anything.
  for (let tick = 1; tick <= 4; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  assert.equal(keeper.held(), 1, "a preflight in progress keeps its hold");

  // The user hits Stop.
  issued.settle();
  assert.equal(
    keeper.held(),
    0,
    "an aborted preflight strands its hold, which then renews the lease forever",
  );
  await Promise.all(pending);

  const lapsed = clock + AUTO_CONTINUE_LEASE_TTL_MS + 1;
  clock = lapsed;
  keeper.tick();
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("m1", { now: lapsed }),
    "started",
    "nothing renews it, so the lease lapses and the message comes back",
  );
});

test("a hold whose run is merely slow is never settled by the clock", async () => {
  // The preflight has no upper bound: this chat's settings pairing waits up to 30 seconds by
  // itself, and `waitForModelReady` then polls for as long as a large local GGUF takes. The
  // promise stays pending throughout, and a pending promise settles nothing.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const otherTab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const issued = issuedRunFake();
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  await tab.claim("m1", { now: start, holder: "thread-A" });
  keeper.hold("m1", "thread-A");
  keeper.settleOn("m1", "thread-A", issued.issued);

  // Ten minutes of preflight, renewed at the keeper's own interval throughout.
  for (let tick = 1; tick <= 20; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);
  assert.equal(keeper.held(), 1, "the hold is still expecting its run");
  assert.equal(
    await otherTab.claim("m1", { now: clock }),
    "held-elsewhere",
    "nobody else may take a message this tab is still about to continue",
  );

  // It finally streams, and the ordinary path gives the lease back with its done marker.
  running.add("thread-A");
  runs.change();
  running.delete("thread-A");
  runs.change();
  await Promise.all(pending);
  assert.equal(keeper.held(), 0, "its own run ending is what gives it back");
});

test("the run that ends is the one the hold was taken for, not the round before it", async () => {
  // The next round is claimed while the previous one is still winding down: the adapter clears
  // `runningByThreadId` from its own `finally`, strictly before the runtime announces that run
  // ending. Anything per-thread would read the predecessor's ending as the successor's and
  // settle a hold whose preflight has only just begun -- lapsing the lease under a live
  // continuation, which is the failure the missing arming deadline exists to avoid.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const otherTab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  let clock = start;
  const pending: Promise<void>[] = [];
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const predecessor = issuedRunFake();
  const round2 = issuedRunFake();
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  // Round one runs and streams on this thread.
  await tab.claim("round-1", { now: start, holder: "thread-A" });
  keeper.hold("round-1", "thread-A");
  keeper.settleOn("round-1", "thread-A", predecessor.issued);
  running.add("thread-A");
  runs.change();
  // Its stream ends, which is what the adapter clears first.
  running.delete("thread-A");
  runs.change();
  await Promise.all(pending);
  assert.equal(
    keeper.held(),
    0,
    "round one is settled by its own stream ending",
  );

  // Round two is claimed and issued while round one is STILL unwinding.
  await tab.claim("round-2", { now: clock, holder: "thread-A" });
  keeper.hold("round-2", "thread-A");
  keeper.settleOn("round-2", "thread-A", round2.issued);

  // Round one's run finally reports itself over. It is not round two's.
  predecessor.settle();
  assert.equal(
    keeper.held(),
    1,
    "the predecessor ending settled the successor's hold mid-preflight",
  );
  for (let tick = 1; tick <= 4; tick += 1) {
    clock = start + tick * AUTO_CONTINUE_LEASE_RENEW_MS;
    keeper.tick();
  }
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("round-2", { now: clock }),
    "held-elsewhere",
    "and its lease is still renewed while its own run is on its way",
  );

  // Only its own run ending ends it.
  round2.settle();
  assert.equal(keeper.held(), 0);
});

test("a run that ends after its hold is gone reaches nothing", async () => {
  // A promise settles whenever it settles, and by then the hold it was taken for may have been
  // given back by its own stream or claimed again for the next round under the same key. The
  // callback is scoped to one hold, not to the message or the thread.
  const { storage } = storageFake();
  const tab = createAutoContinueTab({ storage, locks: null });
  const start = 1_000;
  const pending: Promise<void>[] = [];
  const running = new Set<string>();
  const runs = runSignalFake(running);
  const first = issuedRunFake();
  const released: string[] = [];
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      released.push(messageId);
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => start,
  });

  await tab.claim("m1", { now: start, holder: "thread-A" });
  keeper.hold("m1", "thread-A");
  keeper.settleOn("m1", "thread-A", first.issued);
  // It streams and is released the ordinary way, with the marker that says it was continued.
  running.add("thread-A");
  runs.change();
  running.delete("thread-A");
  runs.change();
  await Promise.all(pending);
  assert.deepEqual(released, ["m1"]);

  // The same message is claimed again for the next round, and only THEN does the first run's
  // promise settle.
  keeper.hold("m1", "thread-A");
  first.settle();
  assert.equal(
    keeper.held(),
    1,
    "an older round's run settled a hold that is not its own",
  );
});

test("settling a hold that was never taken does nothing", () => {
  // `settleOn` runs a line after `hold`, and the hold may have been refused: a thread with no
  // remote id yet is not safe to watch, so nothing is held for it.
  const runs = runSignalFake(new Set<string>());
  const issued = issuedRunFake();
  let attached = 0;
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: () => {},
    release: () => {
      assert.fail("nothing was ever held");
    },
    now: () => 1_000,
  });
  keeper.settleOn("m1", "thread-A", {
    whenSettled: () => {
      attached += 1;
    },
  });
  assert.equal(attached, 0, "no hold, so the run is not watched at all");
  assert.equal(keeper.held(), 0);

  // And a run handed over as nothing leaves an existing hold exactly as it was.
  keeper.hold("m1", "thread-A");
  keeper.settleOn("m1", "thread-A", undefined);
  issued.settle();
  assert.equal(keeper.held(), 1, "no signal is not the same as an ended run");
});

test("only what the runtime actually hands back is treated as the run", () => {
  // `startRun` is DECLARED to return `void` and in fact returns the roundtrip's promise, so
  // the value is passed through untyped and checked here. A value that is not thenable is
  // assistant-ui no longer handing the run back, and the honest answer is no signal at all --
  // the hold is then kept and renewed, which is the behaviour this fix replaced, never an
  // early release.
  for (const notARun of [undefined, null, 0, "", "pending", true, {}, []]) {
    assert.equal(
      issuedRunFrom(notARun),
      undefined,
      `${JSON.stringify(notARun) ?? "undefined"} is not a run`,
    );
  }

  let settledCount = 0;
  const resolved = issuedRunFrom(Promise.resolve("done"));
  assert.notEqual(resolved, undefined, "a promise is a run");
  resolved?.whenSettled(() => {
    settledCount += 1;
  });

  // A thenable rather than a native promise, which is all the contract asks for. The handler
  // is parked on an object rather than in a local so it survives narrowing to `never`.
  const parked: { settle?: (ok: boolean) => void } = {};
  const thenable = {
    then: (onOk: () => void, onErr: () => void) => {
      parked.settle = (ok: boolean) => (ok ? onOk() : onErr());
    },
  };
  const custom = issuedRunFrom(thenable);
  assert.notEqual(custom, undefined, "a thenable is a run");
  let customSettled = 0;
  custom?.whenSettled(() => {
    customSettled += 1;
  });
  assert.equal(customSettled, 0, "and it is pending until it is not");
  parked.settle?.(false);
  assert.equal(
    customSettled,
    1,
    "a run that ended by throwing has still ended",
  );
});

test("a rejected run settles its hold rather than escaping", async () => {
  // The promise rejects when the roundtrip throws and resolves when it is cancelled, and
  // neither says whether the lease may be given back -- only whether the run is still coming.
  // An unobserved rejection here would also be an unhandled one.
  const settled: string[] = [];
  const rejected = issuedRunFrom(Promise.reject(new Error("model refused")));
  rejected?.whenSettled(() => settled.push("rejected"));
  const fulfilled = issuedRunFrom(Promise.resolve());
  fulfilled?.whenSettled(() => settled.push("resolved"));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(
    settled.sort(),
    ["rejected", "resolved"],
    "both outcomes are an ended run",
  );
});

test("the abort case is wired to the run, not to a clock", () => {
  // The bar is the last place that has the run in hand -- the keeper lives in module scope and
  // the runtime is only reachable through a hook -- so `startRun`'s own return value has to be
  // handed over there. Pinned at both ends, since neither side is exercised by a unit test,
  // and re-pinned against a deadline because a deadline here is the arming timeout coming back.
  const bar = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    bar,
    /return aui\.thread\(\)\.startRun\(/,
    "the continuation bar no longer hands its started run back",
  );
  assert.match(
    bar,
    /watchAutoContinueRun\(\s*messageId,\s*runThreadId,\s*startContinuation\(\),?\s*\)/,
    "the started run no longer reaches the keeper",
  );

  const probe = readFileSync(
    new URL(
      "../src/features/chat/utils/auto-continue-issued-run.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    probe,
    /typeof \(started as \{ then\?: unknown \}\)\.then !== "function"/,
    "the declared type is void, so the shape has to be checked rather than assumed",
  );
  const wiredKeeper = readFileSync(
    new URL(
      "../src/features/chat/utils/auto-continue-run-keeper.ts",
      import.meta.url,
    ),
    "utf8",
  );
  for (const [name, source] of [
    ["auto-continue-issued-run.ts", probe],
    ["auto-continue-run-keeper.ts", wiredKeeper],
  ] as const) {
    assert.doesNotMatch(
      source,
      /setTimeout\(/,
      `${name} decides on facts, not on elapsed time`,
    );
  }
});

/**
 * The same field the keeper reads in the app, owners and all.
 *
 * `runSignalFake` above answers from a bare set of thread ids, which is every case where
 * what holds the flag does not matter. Here it does: the image gate flips the flag on and
 * off around a request it never issued, and the real signal in
 * `auto-continue-run-keeper.ts` is what tells that pair from a run. Built on the shipped
 * predicate rather than a copy of it, so a change to either side fails here.
 */
function ownedRunSignalFake() {
  const running = new Map<string, { owner: () => void }[]>();
  const listeners = new Set<() => void>();
  const announce = () => {
    for (const listener of [...listeners]) {
      listener();
    }
  };
  return {
    signal: {
      isRunning: (threadId: string) => {
        const owners = running.get(threadId);
        return Boolean(owners?.length) && !isImageGateRunOnly(owners);
      },
      subscribe: (onChange: () => void) => {
        listeners.add(onChange);
        return () => listeners.delete(onChange);
      },
    },
    /** `setThreadRunning(threadId, true, { owner })`, and the notify it fires. */
    start: (threadId: string, owner: () => void) => {
      running.set(threadId, [...(running.get(threadId) ?? []), { owner }]);
      announce();
    },
    /** `setThreadRunning(threadId, false, { owner })`, clearing that run only. */
    end: (threadId: string, owner: () => void) => {
      const rest = (running.get(threadId) ?? []).filter(
        (entry) => entry.owner !== owner,
      );
      if (rest.length > 0) {
        running.set(threadId, rest);
      } else {
        running.delete(threadId);
      }
      announce();
    },
  };
}

test("a continuation refused by the image gate is not recorded as continued", async () => {
  // Reopen a chat containing a picture with a text-only model loaded -- or switch Vision
  // off for the one that wrote the truncated reply -- and the automatic continuation is
  // refused before a request is made. The gate pulses the thread's running flag true and
  // then false first, so compare mode's `waitForRunEnd` resolves rather than hanging. Read
  // as a run, that pair arms the hold and settles it one call later with the `done` marker
  // that tells every other tab for a day that the message HAS been continued -- for a
  // provider request that was never sent. The failure that follows cannot take it back: a
  // released hold is already gone.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const runs = ownedRunSignalFake();
  const pending: Promise<unknown>[] = [];
  const start = 1_000;
  let clock = start;
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: () => {
      assert.fail(
        "the gate issued no request, so nothing may be marked continued",
      );
    },
    now: () => clock,
  });

  assert.equal(
    await tab.claim("m1", { now: start, holder: "thread-A" }),
    "started",
  );
  keeper.hold("m1", "thread-A");

  // `chat-adapter.ts`: setThreadRunning(true, { owner: gateOwner }) then false, then throw.
  const gateOwner = createImageGateRunOwner();
  runs.start("thread-A", gateOwner);
  assert.equal(
    keeper.held(),
    1,
    "a pulse that stands for a refusal does not arm the hold",
  );
  runs.end("thread-A", gateOwner);
  assert.equal(keeper.held(), 1, "and so cannot settle it either");

  // The throw reaches the wrapper's catch, which is the signal that ends an unarmed hold.
  keeper.failed("thread-A");
  assert.equal(keeper.held(), 0);

  const lapsed = clock + AUTO_CONTINUE_LEASE_TTL_MS + 1;
  clock = lapsed;
  keeper.tick();
  await Promise.all(pending);
  assert.equal(
    await otherTab.claim("m1", { now: lapsed }),
    "started",
    "the message was never continued, so it comes back rather than reading done for a day",
  );
});

test("a real run on the thread the gate refused still owns its lease", async () => {
  // Compare mode puts two runtimes on one key, and an unresolved thread files its run under
  // the shared "__default". A gate firing beside a run that is genuinely streaming must not
  // make the keeper read the thread as idle: that hold would be released mid-stream, and one
  // settle window later another tab could claim the message being written.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const tab = createAutoContinueTab({ storage, locks });
  const otherTab = createAutoContinueTab({ storage, locks });
  const runs = ownedRunSignalFake();
  const released: string[] = [];
  const pending: Promise<unknown>[] = [];
  const start = 1_000;
  let clock = start;
  const keeper = createAutoContinueLeaseKeeper({
    signal: runs.signal,
    renew: (messageId: string, holder: string, now: number) => {
      pending.push(tab.renew(messageId, holder, { now }));
    },
    release: (messageId: string, holder: string, now: number) => {
      released.push(messageId);
      pending.push(tab.release(messageId, holder, { now }));
    },
    now: () => clock,
  });

  await tab.claim("m1", { now: start, holder: "__default" });
  keeper.hold("m1", "__default");

  const streamingRun = () => {};
  runs.start("__default", streamingRun);
  clock = start + AUTO_CONTINUE_LEASE_RENEW_MS;
  keeper.tick();

  // The sibling's image turn is refused while this one streams.
  const gateOwner = createImageGateRunOwner();
  runs.start("__default", gateOwner);
  runs.end("__default", gateOwner);
  assert.deepEqual(released, [], "the run beside it is still generating");
  assert.equal(keeper.held(), 1);

  // Its own run ending is what gives the lease back, marked continued.
  runs.end("__default", streamingRun);
  await Promise.all(pending);
  assert.deepEqual(released, ["m1"]);
  assert.equal(
    await otherTab.claim("m1", {
      now: clock + AUTO_CONTINUE_LEASE_TTL_MS + 1,
    }),
    "held-elsewhere",
    "this one really was continued",
  );
});

test("only the gate's own tokens are read as a refusal", () => {
  // The predicate is asked about "is this thread generating", so anything it cannot prove
  // is a gate pulse has to answer yes: a run from before per-run tracking carries no owner
  // at all, and a key shared with a real run carries one of each.
  const gateOwner = createImageGateRunOwner();
  const runOwner = () => {};
  assert.equal(isImageGateRunOnly([{ owner: gateOwner }]), true);
  assert.equal(isImageGateRunOnly([{ owner: runOwner }]), false);
  assert.equal(
    isImageGateRunOnly([{ owner: gateOwner }, { owner: runOwner }]),
    false,
  );
  assert.equal(isImageGateRunOnly([]), false, "an ownerless flag is a run");
  assert.equal(isImageGateRunOnly(undefined), false);
  assert.notEqual(
    createImageGateRunOwner(),
    gateOwner,
    "one token per pulse, or two gates on a shared key clear each other",
  );
});

test("the gate's pulse is tagged where it is fired and read where it matters", () => {
  // Neither end is exercised by a unit test: the adapter's gate is deep inside a run, and
  // the keeper's real signal reads a zustand store. Pinned at both ends instead.
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const gate = adapter.slice(adapter.indexOf("const imageGateReason ="));
  assert.match(
    gate,
    /const gateOwner = createImageGateRunOwner\(\)/,
    "an anonymous owner is indistinguishable from a run that really started",
  );
  assert.match(
    gate.slice(gate.indexOf("const gateOwner")),
    /setThreadRunning\([\s\S]*owner: gateOwner[\s\S]*setThreadRunning\([\s\S]*owner: gateOwner/,
    "the pulse compare mode waits on is still fired under that token",
  );

  const wiring = readFileSync(
    new URL(
      "../src/features/chat/utils/auto-continue-run-keeper.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    wiring,
    /isImageGateRunOnly\(state\.runOwnerByThreadId\[threadId\]\)/,
    "the keeper is arming holds on a request the gate refused to send",
  );
  assert.doesNotMatch(
    wiring,
    /setTimeout\(/,
    "a deadline here is the arming timeout coming back, which lapses live continuations",
  );
});

// The live streaming publish path used to run the restart check on every arrival.
//
// `isRestart` cannot fire until the continuation reaches 48 characters, and the moment it
// does it publishes the continuation ALONE. Over a 1602-character partial that took the
// published value from 1649 characters to 48 in a single arrival.
//
// That is not just a visible collapse. Stop persists the last STREAMED yield, because
// assistant-ui drops what a run yields after an abort and the terminal merge sits behind
// `!abortSignal.aborted`. So stopping in that window SAVED the 48 characters and threw away
// the whole partial the user was reading.
//
// `joinContinuation`'s `streaming` option exists precisely to suppress that check, and
// production never passed it. These pin the property that actually matters here, which is not
// monotonicity but TEXT PRESERVATION: no streamed publish may drop the partial, and none may
// withhold what the model has generated.

const { createContinuationMerger } = await import(
  "../src/features/chat/utils/continuation.ts"
);

const REASONING =
  "Okay, so the user is asking about how to structure the migration. " +
  "Let me think through this carefully step by step before answering. " +
  "First I need to consider what the existing schema looks like, and " +
  "whether an online migration is even possible given the constraints. ";

/** Replay a stream one character at a time, collecting what each arrival would publish. */
function replay(partial: string, tail: string): string[] {
  const merge = createContinuationMerger(partial, true);
  const published: string[] = [];
  let cumulative = partial;
  published.push(merge(cumulative));
  for (const character of tail) {
    cumulative += character;
    published.push(merge(cumulative));
  }
  return published;
}

test("a restart mid-stream never discards the partial", () => {
  // The original defect, and the one that loses data: Stop here persisted 48 characters.
  const partial = REASONING.repeat(6);
  const published = replay(partial, `${REASONING}and so on. `.repeat(2));

  for (const value of published) {
    assert.equal(
      value.startsWith(partial),
      true,
      "a streamed publish dropped the partial, so Stop would save only the restart",
    );
  }
});

test("no streamed publish withholds generated text", () => {
  // The failure mode of holding the partial back until the repair settles: Stop inside that
  // window saves the stale partial and every new character is lost.
  const partial = REASONING.repeat(6);
  const tail = "The second consideration is throughput, which matters here. ";
  const published = replay(partial, tail);
  const last = published[published.length - 1];

  assert.equal(last, `${partial}${tail}`, "the new text must be present in the live value");
  assert.equal(
    published[10].length > partial.length,
    true,
    "text generated early must appear early, not wait for a settle point",
  );
});

test("what Stop would save mid-stream carries the overlap repair", () => {
  // The failure mode of repairing only at the end: the duplicated tail reaches storage.
  const partial = REASONING.repeat(6);
  const repeated = partial.slice(-60);
  const published = replay(partial, `${repeated}and then it continues onward. `);
  const saved = published[published.length - 1];

  assert.equal(
    saved.includes(`${repeated}${repeated}`),
    false,
    "the repeated tail survived into the value Stop persists",
  );
  assert.equal(saved.startsWith(partial), true, "the partial itself must be intact");
});

test("the join can shift by the overlap, and never by more", () => {
  // Honest about what is NOT fixed. A longer overlap starting to match rewrites the join, so
  // the published length can dip. It is bounded by MAX_OVERLAP and never loses text, which is
  // why it is not worth holding output back to avoid.
  const partial = REASONING.repeat(6);
  const published = replay(
    partial,
    `${partial.slice(-60)}and then it continues onward from there. `,
  );

  let worst = 0;
  for (let i = 1; i < published.length; i += 1) {
    worst = Math.max(worst, published[i - 1].length - published[i].length);
  }
  assert.equal(worst <= 400, true, `the join shifted by ${worst}, beyond MAX_OVERLAP`);
});

test("the final merge still collapses a genuine restart", () => {
  const partial = REASONING.repeat(6);
  const restart = `${REASONING}and so on. `;
  assert.equal(
    createContinuationMerger(partial, true)(partial + restart, { final: true }),
    restart,
    "a restart is collapsed once the turn is complete and the evidence is in",
  );
});

test("a short partial with a whitespace-led restart is not collapsed mid-stream", () => {
  // `isRestart` calls trimStart(), so a leading newline shifts when it can fire. With the
  // restart check off while streaming, that timing cannot produce a mid-stream collapse.
  const partial = "Sure, here is a plan for the migration you asked me about.";
  const merge = createContinuationMerger(partial, true);
  const restart = `\n${partial}`;

  assert.equal(
    merge(`${partial}${restart}`).startsWith(partial),
    true,
    "the partial must survive a whitespace-led restart while streaming",
  );
});

test("a merger with repair off is the identity, streaming or final", () => {
  // Local backends resume at the exact token boundary, so nothing may be trimmed.
  const partial = REASONING.repeat(2);
  const merge = createContinuationMerger(partial, false);
  const full = `${partial}${REASONING}`;
  assert.equal(merge(full), full);
  assert.equal(merge(full, { final: true }), full);
});
