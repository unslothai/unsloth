// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { mergeContextTruncation } from "../src/features/chat/utils/context-truncation.ts";

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
  assert.match(adapter, /context_overflow:\s*"truncate_oldest"/);
  assert.match(adapter, /This conversation was compacted/);
});

test("the transport preserves standard chunks with context metadata", () => {
  assert.doesNotMatch(transport, /parsed\.type === "context_truncated"/);
  assert.match(adapter, /chunk\.context_truncated/);
  assert.match(adapter, /contextTruncation = mergeContextTruncation\(/);
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
  // that reported it, which is more of the conversation actually leaving the context.
  assert.match(thread, /const showsNotice = useAuiState/);
  assert.match(thread, /contextTruncation && showsNotice && !isEditing/);
  assert.match(thread, /dropped > previousDropped/);
  // Walked in order, rather than compared against the immediately preceding message:
  // turns between two moves report the same count and must not reset the baseline.
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
  // Rolling back to an earlier message leaves a shorter branch, which needs less
  // evicting. Less of the conversation is missing than before, so there is nothing to
  // tell the user, and the baseline must not be dragged down into re-announcing it.
  assert.deepStrictEqual(noticeTurns([52, 20, 20, 20]), [0]);
});
