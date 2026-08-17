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

test("the compaction notice is gated to the FIRST compacted turn", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  // A thread that has outgrown its window compacts on every turn from then on, so an
  // ungated notice is a notice on every reply for the rest of the conversation.
  assert.match(thread, /const isFirstCompaction = useAuiState/);
  assert.match(
    thread,
    /contextTruncation && isFirstCompaction && !isEditing/,
  );
  // The gate walks the thread and stops at the first compacted assistant turn, rather
  // than assuming the compacted ones are contiguous or that this message is the last.
  assert.match(thread, /for \(const message of thread\.messages\)/);
});
