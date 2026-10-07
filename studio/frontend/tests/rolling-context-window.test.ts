// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type ContextTruncation,
  compactionBoundary,
  shouldShowCompactionNotice,
  mergeContextTruncation,
  promptWasShortened,
} from "../src/features/chat/utils/context-truncation.ts";

import { readSrc } from "./helpers/kit.ts";

const COMPACTION_NOTICE = readSrc("components/assistant-ui/compaction-notice.tsx");
const THREAD = readSrc("components/assistant-ui/thread.tsx");
const CHAT_ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

const adapter = readSrc("features/chat/api/chat-adapter.ts");
const transport = readSrc("features/chat/api/chat-api.ts");

test("local chat opts into the rolling context policy", () => {
  assert.match(adapter, /isGgufForCompaction/);
  assert.match(adapter, /runtime\.loadedIsGguf/);
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
  const runtimeProvider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(runtimeProvider, /contextTruncation: mergeContextTruncation\(/);
  assert.match(runtimeProvider, /generationChunkCount/);
  assert.match(adapter, /generationFirstChunkAt/);
  assert.match(adapter, /generationChunkCount \+= 1/);
});

test("the compaction notice follows the boundary, not the accumulated drops", () => {
  // Recording drops (12) instead of the boundary (4) would silence the next real advances.
  assert.equal(
    compactionBoundary({ dropped_messages: 12, boundary_messages: 4, fits: true }),
    4,
  );
  assert.equal(compactionBoundary({ dropped_messages: 6, fits: true }), 6);
  assert.equal(
    compactionBoundary({ dropped_messages: 0, boundary_messages: 0, fits: false }),
    0,
  );
  assert.equal(compactionBoundary(undefined), 0);
});

test("a shortened prompt still counts as a compaction, whatever fits says", () => {
  // A fit with fits:false still evicted turns, so the notice must fire.
  assert.equal(promptWasShortened({ dropped_messages: 2, fits: false }), true);
  assert.equal(promptWasShortened({ dropped_messages: 0, fits: false }), false);
  assert.equal(promptWasShortened(undefined), false);
});

test("a shortened refusal records its boundary, so the notice survives a reload", () => {
  const rescued = { fits: false, dropped_messages: 6, boundary_messages: 6 };
  assert.equal(compactionBoundary(rescued), 6);
  assert.equal(promptWasShortened(rescued), true);

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
  // Outside legacy fits, dropped_messages is a per-refit sum, not a position.
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
  assert.equal(promptWasShortened(toolLoop), true);
  assert.equal(compactionBoundary({ dropped_messages: 3, fits: true }), 3);
  assert.equal(
    compactionBoundary({ dropped_messages: 16, fits: false, boundary_messages: 4 }),
    4,
  );
});

test("a rescued turn cannot silence the compactions that follow it", () => {
  const boundariesShown = (records: ContextTruncation[]) => {
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
      { fits: false, dropped_messages: 16 },
      { fits: true, dropped_messages: 2, boundary_messages: 6 },
      { fits: true, dropped_messages: 2, boundary_messages: 8 },
    ]),
    [0, 2, 3],
  );
});

test("the notice and the toast read the same predicate as the boundary", () => {
  assert.match(COMPACTION_NOTICE, /promptWasShortened\(truncation\)/);
  assert.doesNotMatch(COMPACTION_NOTICE, /truncation\?\.fits/);
  assert.match(adapter, /promptWasShortened\(chunk\.context_truncated\)/);
});

test("the compaction boundary takes the latest value, never the sum", () => {
  // dropped_messages includes tool messages, so summing it overshoots; boundary is absolute.
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
  assert.match(THREAD, /custom\?\.contextTruncation/);
  assert.match(THREAD, /<CompactionNotice truncation=\{contextTruncation\}/);
  assert.match(COMPACTION_NOTICE, /This conversation got long, so it was compacted/);
});

test("the compaction notice uses the shared boundary/checkpoint predicate", () => {
  assert.match(THREAD, /const showsNotice = useAuiState/);
  assert.match(THREAD, /contextTruncation && showsNotice && !isEditing/);
  assert.match(THREAD, /shouldShowCompactionNotice\(value, previousDropped\)/);
  assert.match(THREAD, /for \(const message of thread\.messages\)/);
});

const noticeTurns = (dropped: (number | null)[]): number[] => {
  const shown: number[] = [];
  let previousDropped = 0;
  dropped.forEach((value, index) => {
    const d = value ?? 0;
    if (shouldShowCompactionNotice({ dropped_messages: d, boundary_messages: d, fits: true }, previousDropped)) {
      shown.push(index);
      previousDropped = d;
    }
  });
  return shown;
};

test("one notice per compaction, and silence on the turns in between", () => {
  assert.deepStrictEqual(
    noticeTurns([0, 0, 52, 52, 52, 52, 52, 62, 62, 62, 74]),
    [2, 7, 10],
  );
  assert.deepStrictEqual(noticeTurns([0, 0, 0]), []);
  assert.deepStrictEqual(noticeTurns([36, 36, 36]), [0]);
});

test("a boundary that goes BACKWARDS does not re-announce", () => {
  // After a rollback the baseline must not be dragged down.
  assert.deepStrictEqual(noticeTurns([52, 20, 20, 20]), [0]);
});

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
  const exporter = readSrc("features/chat/utils/conversation-markdown-export.ts");

  // Sibling of the content parts: anything walking parts would otherwise pick it up.
  const noticeAt = THREAD.indexOf("<CompactionNotice");
  const partsAt = THREAD.indexOf("<MessagePrimitive.Parts", noticeAt);
  assert.ok(noticeAt > 0 && partsAt > noticeAt);
  assert.ok(
    !/<MessagePrimitive\.Parts[^>]*>[\s\S]*<CompactionNotice/.test(THREAD),
    "the notice must not be rendered inside the message's content parts",
  );

  // Bounded to function bodies: the streaming handler legitimately reads contextTruncation.
  for (const name of ["toOpenAIMessages", "serializeAssistantReplayMessages"]) {
    const body = functionBody(CHAT_ADAPTER, name);
    assert.ok(body.length > 0, `${name} not found`);
    assert.ok(
      !body.includes("contextTruncation"),
      `${name} must never read contextTruncation`,
    );
  }

  assert.ok(!exporter.includes("contextTruncation"));
  assert.ok(!exporter.includes("compacted"));

  assert.match(THREAD, /contextTruncation && showsNotice && !isEditing/);
});

test("an irreducible fit reports a diagnosis, and it is dropped once something fits", () => {
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

  const recovered = mergeContextTruncation(failed, {
    dropped_messages: 12,
    fits: true,
    prompt_tokens_after: 3000,
    context_length: 4096,
  });
  assert.equal(recovered.fits, true);
  assert.ok(!("irreducible_tokens" in recovered));
  assert.ok(!("latest_turn_tokens" in recovered));

  const plain = mergeContextTruncation(
    { dropped_messages: 1, fits: true },
    { dropped_messages: 2, fits: true },
  );
  assert.ok(!("irreducible_tokens" in plain));
  assert.ok(!("latest_turn_tokens" in plain));
});

test("the too-long advice depends on WHICH part does not fit", () => {
  assert.match(CHAT_ADAPTER, /contextTruncation\?\.fits === false/);
  assert.match(CHAT_ADAPTER, /shortening the conversation will not help/);
  assert.match(CHAT_ADAPTER, /latestTurnOwnTokens\(irreducible\)/);
});

test("a fits:false diagnosis is not a compaction", () => {
  // The fitter returned the original messages, and toasting would burn the once-per-thread flag.
  assert.match(CHAT_ADAPTER, /const reallyCompacted = promptWasShortened\(/);
  assert.equal(promptWasShortened({ dropped_messages: 0, fits: false }), false);
});

test("the advice depends on WHOSE turn does not fit", () => {
  assert.match(CHAT_ADAPTER, /latest_turn_role/);
  assert.match(CHAT_ADAPTER, /const userCanShortenIt =/);
  assert.match(CHAT_ADAPTER, /The last tool result is/);
  assert.match(CHAT_ADAPTER, /latest_turn_role \?\? "user"/);
  assert.match(CHAT_ADAPTER, /Shorten this message/);
});

test("the too-long check uses the prompt budget, not the raw window", () => {
  // The fit reserves up to a quarter of the window for the reply.
  assert.match(CHAT_ADAPTER, /irreducible\?\.prompt_target \?\? irreducible\?\.context_length/);
  assert.match(CHAT_ADAPTER, /latestTurnIsTheProblem\(\s*irreducible,\s*budget,?\s*\)/);
});

test("a checkpoint inside the current tool loop shows a notice without advancing the saved boundary", () => {
  const recorded = {
    dropped_messages: 6,
    boundary_messages: 0,
    checkpoint_started: true,
    fits: true,
  };
  assert.equal(shouldShowCompactionNotice(recorded, 0), true);
  assert.equal(shouldShowCompactionNotice({ ...recorded, boundary_messages: 4 }, 4), true);
  assert.equal(shouldShowCompactionNotice({ ...recorded, checkpoint_started: false }, 0), false);
  assert.equal(shouldShowCompactionNotice({ ...recorded, dropped_messages: 0 }, 0), false);
  assert.equal(shouldShowCompactionNotice(undefined, 0), false);
});

test("later tool-loop fits cannot erase a checkpoint started earlier in the same reply", () => {
  const merged = mergeContextTruncation(
    { dropped_messages: 6, boundary_messages: 0, checkpoint_started: true, fits: true },
    { dropped_messages: 0, boundary_messages: 0, checkpoint_started: false, fits: true },
  );
  assert.equal(merged.checkpoint_started, true);
  assert.equal(shouldShowCompactionNotice(merged, 0), true);
  assert.equal(shouldShowCompactionNotice({ ...merged, checkpoint_started: false }, 0), false);
});
