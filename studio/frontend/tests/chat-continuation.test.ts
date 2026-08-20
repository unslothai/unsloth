// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  AUTO_CONTINUE_LEASE_KEY,
  AUTO_CONTINUE_LEASE_RENEW_MS,
  AUTO_CONTINUE_LEASE_SETTLE_MS,
  AUTO_CONTINUE_LEASE_TTL_MS,
  AUTO_CONTINUE_LIMIT,
  autoContinueCount,
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
    await running.renew(PANE, {
      now: start + tick * AUTO_CONTINUE_LEASE_RENEW_MS,
    });
  }
  const past =
    start + AUTO_CONTINUE_LEASE_TTL_MS + AUTO_CONTINUE_LEASE_RENEW_MS;
  assert.equal(await other.claim("m1", { now: past }), "held-elsewhere");
});

test("a finished run gives its lease back, without handing over a stale branch", async () => {
  // Released on any terminal state, so the full TTL is left to mean one thing: a crash.
  // Cut to the settle window rather than deleted, because a tab still showing the
  // pre-continuation branch has not seen the sibling yet and would start the duplicate.
  const { storage } = storageFake();
  const locks = lockManagerFake();
  const running = createAutoContinueTab({ storage, locks });
  const other = createAutoContinueTab({ storage, locks });
  const start = 1_000;

  assert.equal(
    await running.claim("m1", { now: start, holder: PANE }),
    "started",
  );
  await running.release(PANE, { now: start });
  assert.equal(
    await other.claim("m1", { now: start + AUTO_CONTINUE_LEASE_SETTLE_MS - 1 }),
    "held-elsewhere",
  );
  assert.equal(
    await other.claim("m1", { now: start + AUTO_CONTINUE_LEASE_SETTLE_MS + 1 }),
    "started",
  );
  // And a release holds nothing, so a later renewal cannot resurrect it.
  await running.renew(PANE, { now: start + AUTO_CONTINUE_LEASE_SETTLE_MS + 2 });
  assert.equal(
    await createAutoContinueTab({ storage, locks }).claim("m1", {
      now: start + AUTO_CONTINUE_LEASE_SETTLE_MS + 3,
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
  await compareTab.release(base, { now: start });

  // The lora pane is still generating, and its keeper is still renewing.
  for (let tick = 1; tick <= 3; tick += 1) {
    await compareTab.renew(lora, {
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
  // And the pane that did finish settles on the usual schedule.
  assert.equal(
    await otherTab.claim("base-m1", {
      now: start + AUTO_CONTINUE_LEASE_SETTLE_MS + 1,
    }),
    "started",
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
  await mine.renew(PANE, { now: start + AUTO_CONTINUE_LEASE_RENEW_MS });

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
