// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  type ContextTruncation,
  compactionBoundary,
  mergeContextTruncation,
  promptWasShortened,
} from "../src/features/chat/utils/context-truncation.ts";

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const transport = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);

test("local chat opts into the rolling context policy", () => {
  assert.match(adapter, /isGguf === true/);
  assert.match(adapter, /autoCompactEnabled/);
  assert.match(adapter, /ggufCompactionRequestFields\(/);
  assert.match(adapter, /This conversation was compacted/);
});

test("the transport preserves standard chunks with context metadata", () => {
  assert.doesNotMatch(transport, /parsed\.type === "context_truncated"/);
  assert.match(adapter, /chunk\.context_truncated/);
  assert.match(adapter, /contextTruncation = mergeContextTruncation\(/);
});

test("durable replay persists context-truncation metadata", () => {
  const runtimeProvider = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.match(runtimeProvider, /contextTruncation: mergeContextTruncation\(/);
  assert.match(runtimeProvider, /generationChunkCount/);
  assert.match(adapter, /generationFirstChunkAt/);
  assert.match(adapter, /generationChunkCount \+= 1/);
});

test("the compaction notice follows the boundary, not the accumulated drops", () => {
  // A tool-heavy turn reports 12 drops while the boundary moved to 4. Recording 12 as
  // the high-water mark means the next two real advances, to 8 and to 10, are silent.
  assert.equal(
    compactionBoundary({ dropped_messages: 12, boundary_messages: 4, fits: true }),
    4,
  );
  // Turns saved before the boundary existed still report something.
  assert.equal(compactionBoundary({ dropped_messages: 6, fits: true }), 6);
  // A fit that gave up moved no boundary at all.
  assert.equal(
    compactionBoundary({ dropped_messages: 0, boundary_messages: 0, fits: false }),
    0,
  );
  assert.equal(compactionBoundary(undefined), 0);
});

test("a shortened prompt still counts as a compaction, whatever fits says", () => {
  // A fit that lands under the physical window but misses the reply reserve sends the
  // eviction with fits:false. The turns are gone from the model's view, so the notice
  // and the toast must fire; only a fit that returned the ORIGINAL messages stays quiet.
  assert.equal(promptWasShortened({ dropped_messages: 2, fits: false }), true);
  assert.equal(promptWasShortened({ dropped_messages: 0, fits: false }), false);
  assert.equal(promptWasShortened(undefined), false);
});

test("a shortened refusal records its boundary, so the notice survives a reload", () => {
  // A rescue evicts for real, so it reports its depth like any other compaction and the
  // persisted notice can find it. Saving the depth is not the same as replaying it:
  // `_sticky_compaction_boundary` still declines any record whose `fits` is false.
  const rescued = { fits: false, dropped_messages: 6, boundary_messages: 6 };
  assert.equal(compactionBoundary(rescued), 6);
  assert.equal(promptWasShortened(rescued), true);

  // A boundary is absolute, so refitting three times does not inflate it.
  let refits: ContextTruncation = {
    fits: false,
    dropped_messages: 4,
    boundary_messages: 4,
  };
  for (const chunk of [
    { fits: false, dropped_messages: 6, boundary_messages: 6 },
    { fits: false, dropped_messages: 6, boundary_messages: 6 },
  ]) {
    refits = mergeContextTruncation(refits, chunk);
  }
  assert.equal(refits.dropped_messages, 16);
  assert.equal(compactionBoundary(refits), 6);
});

test("a record with no boundary never guesses one from a summed drop count", () => {
  // The legacy fallback exists for turns saved before boundary_messages was recorded, and
  // those all fit. On anything else the count is a per-refit SUM, not a position, and
  // reading it as one sets a high-water mark `showsNotice` cannot see exceeded again.
  const oneRefit = { dropped_messages: 2, fits: false };
  let toolLoop: ContextTruncation = { fits: false, dropped_messages: 4 };
  for (const chunk of [
    { fits: false, dropped_messages: 6 },
    { fits: false, dropped_messages: 6 },
  ]) {
    toolLoop = mergeContextTruncation(toolLoop, chunk);
  }

  assert.equal(toolLoop.dropped_messages, 16);
  assert.equal(compactionBoundary(oneRefit), 0);
  assert.equal(compactionBoundary(toolLoop), 0);
  // The notice still fires: that reads promptWasShortened, not the boundary.
  assert.equal(promptWasShortened(toolLoop), true);
  // And a fit that SUCCEEDED still gets the legacy fallback, for turns saved before
  // boundary_messages existed.
  assert.equal(compactionBoundary({ dropped_messages: 3, fits: true }), 3);
  assert.equal(
    compactionBoundary({ dropped_messages: 16, fits: false, boundary_messages: 4 }),
    4,
  );
});

test("a rescued turn cannot silence the compactions that follow it", () => {
  // The `showsNotice` scan in thread.tsx, which only announces a boundary that ROSE.
  const boundariesShown = (records: any[]) => {
    let high = 0;
    const shown: number[] = [];
    records.forEach((rec, index) => {
      const b = compactionBoundary(rec);
      if (b > high) {
        shown.push(index);
        high = b;
      }
    });
    return shown;
  };

  assert.deepEqual(
    boundariesShown([
      { fits: true, dropped_messages: 4, boundary_messages: 4 },
      { fits: false, dropped_messages: 16 }, // rescued, three refits
      { fits: true, dropped_messages: 2, boundary_messages: 6 },
      { fits: true, dropped_messages: 2, boundary_messages: 8 },
    ]),
    [0, 2, 3],
  );
});

test("the notice and the toast read the same predicate as the boundary", () => {
  const notice = readFileSync(
    new URL(
      "../src/components/assistant-ui/compaction-notice.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(notice, /promptWasShortened\(truncation\)/);
  assert.doesNotMatch(notice, /truncation\?\.fits/);
  assert.match(adapter, /promptWasShortened\(chunk\.context_truncated\)/);
});

test("the compaction boundary takes the latest value, never the sum", () => {
  // dropped_messages counts what each fit removed in front of it, this turn's tool
  // messages included, so summing it and re-applying the total advances the boundary
  // past the turns actually evicted. The boundary is carried separately and absolutely.
  const combined = mergeContextTruncation(
    mergeContextTruncation(undefined, {
      dropped_messages: 4,
      boundary_messages: 4,
      fits: true,
    }),
    { dropped_messages: 4, boundary_messages: 4, fits: true },
  );

  assert.equal(combined.dropped_messages, 8);
  assert.equal(combined.boundary_messages, 4);
});

test("tool-loop truncation metadata accumulates across stream events", () => {
  const first = mergeContextTruncation(undefined, {
    dropped_messages: 2,
    prompt_tokens_before: 1200,
    prompt_tokens_after: 800,
    context_length: 1600,
    fits: true,
  });
  const combined = mergeContextTruncation(first, {
    dropped_messages: 3,
    prompt_tokens_before: 1000,
    prompt_tokens_after: 700,
    context_length: 1400,
    fits: true,
  });

  assert.deepEqual(combined, {
    dropped_messages: 5,
    prompt_tokens_before: 1200,
    prompt_tokens_after: 700,
    context_length: 1400,
    fits: true,
  });
});


test("compaction counts accumulate and stay absent on a plain rolling window", () => {
  // A plain rolling-window response must keep exactly the shape it had before the
  // conversation archive existed, rather than carrying archive keys set to undefined.
  const plain = mergeContextTruncation(
    { dropped_messages: 1, fits: true },
    { dropped_messages: 2, fits: true },
  );
  assert.ok(!("archived_messages" in plain));
  assert.ok(!("recalled_chunks" in plain));

  const archived = mergeContextTruncation(
    { dropped_messages: 1, fits: true, archived_messages: 2, recalled_chunks: 4 },
    { dropped_messages: 2, fits: true, archived_messages: 3, recalled_chunks: 1 },
  );
  assert.equal(archived.archived_messages, 5);
  assert.equal(archived.recalled_chunks, 5);
});

test("the compaction notice renders from persisted metadata, not from a message", () => {
  const notice = readFileSync(
    new URL("../src/components/assistant-ui/compaction-notice.tsx", import.meta.url),
    "utf8",
  );
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  // Read off metadata.custom so it can never become part of the conversation.
  assert.match(thread, /custom\?\.contextTruncation/);
  assert.match(thread, /<CompactionNotice truncation=\{contextTruncation\}/);
  assert.match(notice, /This conversation got long, so it was compacted/);
});

test("the compaction notice is gated on the eviction boundary MOVING", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  // Every request after the window fills runs the fit, so "this turn compacted" puts a
  // notice on every reply. The trigger is dropped_messages rising above the last turn
  // that reported it: more of the conversation actually leaving the context.
  assert.match(thread, /const showsNotice = useAuiState/);
  assert.match(thread, /contextTruncation && showsNotice && !isEditing/);
  assert.match(thread, /dropped > previousDropped/);
  // Walked in order, not against the preceding message: turns between two moves report
  // the same count and must not reset the baseline.
  assert.match(thread, /for \(const message of thread\.messages\)/);
});

// The gate is a pure function of the thread's persisted truncation counts, so it can be
// evaluated directly on the sequences the server actually produces.
const noticeTurns = (dropped: (number | null)[]): number[] => {
  const shown: number[] = [];
  let previousDropped = 0;
  dropped.forEach((value, index) => {
    const d = value ?? 0;
    if (d > previousDropped) {
      shown.push(index);
      previousDropped = d;
    }
  });
  return shown;
};

test("one notice per compaction, and silence on the turns in between", () => {
  // A compaction, a stretch of turns whose boundary does not move, then another.
  assert.deepStrictEqual(
    noticeTurns([0, 0, 52, 52, 52, 52, 52, 62, 62, 62, 74]),
    [2, 7, 10],
  );
  // The uncompacted case stays silent throughout.
  assert.deepStrictEqual(noticeTurns([0, 0, 0]), []);
  // A single compaction that never moves again is reported exactly once.
  assert.deepStrictEqual(noticeTurns([36, 36, 36]), [0]);
});

test("a boundary that goes BACKWARDS does not re-announce", () => {
  // A rollback leaves a shorter branch needing less eviction. Less is missing than
  // before, so there is nothing to say and the baseline must not be dragged down.
  assert.deepStrictEqual(noticeTurns([52, 20, 20, 20]), [0]);
});

/** The source of one function, by brace matching from its declaration. */
const functionBody = (source: string, name: string): string => {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) return "";
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(start, index + 1);
    }
  }
  return "";
};

test("the notice is a NOTICE, never part of the conversation", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const exporter = readFileSync(
    new URL("../src/features/chat/utils/conversation-markdown-export.ts", import.meta.url),
    "utf8",
  );

  // 1. A sibling of the rendered content parts, not one of them: inside
  //    MessagePrimitive.Parts everything that walks parts would pick it up.
  const noticeAt = thread.indexOf("<CompactionNotice");
  const partsAt = thread.indexOf("<MessagePrimitive.Parts", noticeAt);
  assert.ok(noticeAt > 0 && partsAt > noticeAt);
  assert.ok(
    !/<MessagePrimitive\.Parts[^>]*>[\s\S]*<CompactionNotice/.test(thread),
    "the notice must not be rendered inside the message's content parts",
  );

  // 2. Nothing that builds a request may read the key it renders from. Bounded to the
  //    function bodies: slicing to end-of-file also catches the streaming handler,
  //    which reads contextTruncation legitimately on the way IN.
  for (const name of ["toOpenAIMessages", "serializeAssistantReplayMessages"]) {
    const body = functionBody(adapter, name);
    assert.ok(body.length > 0, `${name} not found`);
    assert.ok(
      !body.includes("contextTruncation"),
      `${name} must never read contextTruncation`,
    );
  }

  // 3. Nor may the user-facing export, which is the other way text leaves a thread.
  assert.ok(!exporter.includes("contextTruncation"));
  assert.ok(!exporter.includes("compacted"));

  // 4. Suppressed while editing, so it cannot be saved back as message text.
  assert.match(thread, /contextTruncation && showsNotice && !isEditing/);
});

test("an irreducible fit reports a diagnosis, and it is dropped once something fits", () => {
  // A fit that gave up carries the numbers that say WHICH part is too long.
  const failed = mergeContextTruncation(undefined, {
    dropped_messages: 0,
    fits: false,
    prompt_tokens_before: 10290,
    prompt_tokens_after: 10290,
    context_length: 4096,
    irreducible_tokens: 5050,
    latest_turn_tokens: 5000,
  });
  assert.equal(failed.fits, false);
  assert.equal(failed.latest_turn_tokens, 5000);

  // The loop refits per iteration, and an iteration that DOES fit must not carry the
  // earlier failure's numbers forward, where they describe nothing.
  const recovered = mergeContextTruncation(failed, {
    dropped_messages: 12,
    fits: true,
    prompt_tokens_after: 3000,
    context_length: 4096,
  });
  assert.equal(recovered.fits, true);
  assert.ok(!("irreducible_tokens" in recovered));
  assert.ok(!("latest_turn_tokens" in recovered));

  // And an ordinary response never grows the keys at all, not even set to undefined.
  const plain = mergeContextTruncation(
    { dropped_messages: 1, fits: true },
    { dropped_messages: 2, fits: true },
  );
  assert.ok(!("irreducible_tokens" in plain));
  assert.ok(!("latest_turn_tokens" in plain));
});

test("the too-long advice depends on WHICH part does not fit", () => {
  const adapterSource = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // Telling someone to shorten the conversation is a dead end when the history has
  // already been evicted and the single message is what overflows.
  assert.match(adapterSource, /contextTruncation\?\.fits === false/);
  assert.match(adapterSource, /shortening the conversation will not help/);
  // Matching the wire field name would pin nothing: after the floor fix its only
  // occurrence in that file is prose in a comment.
  assert.match(adapterSource, /latestTurnOwnTokens\(irreducible\)/);
});

test("a fits:false diagnosis is not a compaction", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // The fitter returned the ORIGINAL messages with dropped_messages 0, so "older turns
  // were removed" is untrue, and toasting it burns the once-per-thread flag. Asserted on
  // the predicate rather than the literal expression, so it survives a rewording.
  assert.match(source, /const reallyCompacted = promptWasShortened\(/);
  assert.equal(promptWasShortened({ dropped_messages: 0, fits: false }), false);
});

test("the advice depends on WHOSE turn does not fit", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // A tool loop refits with the tool result appended, so the offending turn is often
  // output the user never wrote and cannot edit, leaving no remedy.
  assert.match(source, /latest_turn_role/);
  assert.match(source, /const userCanShortenIt =/);
  assert.match(source, /The last tool result is/);
  // The user-authored case keeps its advice, and an older server that sends no role
  // still gets it (the default is "user").
  assert.match(source, /latest_turn_role \?\? "user"/);
  assert.match(source, /Shorten this message/);
});

test("the too-long check uses the prompt budget, not the raw window", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // The fit reserves up to a quarter of the window for the reply, so a 3,500-token
  // message cannot fit a 4,096-token context. The raw window would blame the
  // conversation and send the user to a new chat that fails identically.
  assert.match(source, /irreducible\?\.prompt_target \?\? irreducible\?\.context_length/);
  // Still measured against the budget, but through the helper that takes the prompt's
  // shared floor off the turn first.
  assert.match(source, /latestTurnIsTheProblem\(\s*irreducible,\s*budget,?\s*\)/);
});
