// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  compactionDividerCount,
  compactionDividerRows,
  compactionToolCallDividers,
} from "../src/components/assistant-ui/message-derived.ts";
import { ggufCompactionRequestFields } from "../src/features/chat/utils/auto-compaction.ts";
import {
  appendCompactionMarker,
  type CompactionMarker,
  compactionAnchorToolCallId,
  type ContextTruncation,
} from "../src/features/chat/utils/context-truncation.ts";
import {
  isCompactCommand,
  requestThreadCompaction,
  takeThreadCompactionRequest,
} from "../src/features/chat/stores/compact-request-store.ts";

import { readSrc } from "./helpers/kit.ts";

const fit = (
  boundary: number,
  extra: Partial<ContextTruncation> = {},
): ContextTruncation => ({
  dropped_messages: boundary,
  boundary_messages: boundary,
  fits: true,
  ...extra,
});

test("/compact is the command typed alone, in any case", () => {
  assert.equal(isCompactCommand("/compact"), true);
  assert.equal(isCompactCommand("  /COMPACT \n"), true);
  assert.equal(isCompactCommand("/compact the chat please"), false);
  assert.equal(isCompactCommand("compact"), false);
});

test("a compaction request applies to the thread's next request only", () => {
  requestThreadCompaction("t1");
  requestThreadCompaction("t1");
  assert.equal(takeThreadCompactionRequest("t2"), false);
  assert.equal(takeThreadCompactionRequest(undefined), false);
  assert.equal(takeThreadCompactionRequest("t1"), true);
  assert.equal(takeThreadCompactionRequest("t1"), false);
});

test("compact now asks a GGUF or MLX model to compact, even with auto-compaction off", () => {
  const asked = { context_overflow: "truncate_oldest", compact_context: true };
  for (const autoCompactEnabled of [true, false]) {
    assert.deepEqual(
      ggufCompactionRequestFields({ isGguf: true, autoCompactEnabled, compactNow: true }),
      asked,
    );
    assert.deepEqual(
      ggufCompactionRequestFields({
        isGguf: false,
        isMlx: true,
        autoCompactEnabled,
        compactNow: true,
      }),
      asked,
    );
  }
  assert.deepEqual(
    ggufCompactionRequestFields({ isGguf: false, autoCompactEnabled: true, compactNow: true }),
    {},
  );
});

test("a tool loop's replayed cut is not a new marker, a new reset is", () => {
  let markers: CompactionMarker[] | undefined;
  markers = appendCompactionMarker(markers, fit(0, { dropped_messages: 0 }), null);
  assert.equal(markers, undefined);
  markers = appendCompactionMarker(markers, fit(4, { checkpoint_started: true }), null);
  markers = appendCompactionMarker(markers, fit(4), "call-1");
  markers = appendCompactionMarker(markers, fit(4, { checkpoint_started: true }), "call-2");
  assert.deepEqual(
    markers?.map((marker) => marker.afterToolCallId),
    [null, "call-2"],
  );
});

const toolCall = (toolCallId: string) => ({ type: "tool-call", toolCallId });

test("a compaction inside a tool loop is drawn after the call it followed", () => {
  const midLoop = fit(0, { dropped_messages: 6, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [toolCall("call-1"), toolCall("call-2"), toolCall("call-3")],
      metadata: {
        custom: {
          contextTruncation: midLoop,
          compactionMarkers: [{ afterToolCallId: "call-2", truncation: midLoop }],
        },
      },
    },
  ];

  // Not above the first prompt, where nothing older existed to compact.
  assert.equal(compactionDividerRows(messages).size, 0);
  assert.equal(compactionToolCallDividers(messages).get("call-2"), midLoop);
  assert.equal(compactionDividerCount(messages), 1);
});

test("a request that compacted before replying keeps its divider above the prompt", () => {
  const atStart = fit(4, { checkpoint_started: true, manual: true });
  const inLoop = fit(4, { dropped_messages: 3, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    { id: "a1", role: "assistant" },
    { id: "u2", role: "user" },
    {
      id: "a2",
      role: "assistant",
      parts: [toolCall("call-9")],
      metadata: {
        custom: {
          contextTruncation: inLoop,
          compactionMarkers: [
            { afterToolCallId: null, truncation: atStart },
            { afterToolCallId: "call-9", truncation: inLoop },
          ],
        },
      },
    },
  ];

  assert.deepEqual([...compactionDividerRows(messages)], [["u2", atStart]]);
  assert.deepEqual([...compactionToolCallDividers(messages)], [["call-9", inLoop]]);
  assert.equal(compactionDividerCount(messages), 2);
});

test("a marker naming a call the reply no longer shows falls back to the prompt", () => {
  const midLoop = fit(0, { dropped_messages: 2, checkpoint_started: true });
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [],
      metadata: {
        custom: {
          contextTruncation: midLoop,
          compactionMarkers: [{ afterToolCallId: "gone", truncation: midLoop }],
        },
      },
    },
  ];

  assert.deepEqual([...compactionDividerRows(messages)], [["u1", midLoop]]);
  assert.equal(compactionToolCallDividers(messages).size, 0);
});

test("the composer reads /compact before any send path, and only for GGUF or MLX", async () => {
  const thread = await readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(thread.indexOf("const handleSubmit = useCallback("));
  const command = submit.indexOf("isCompactCommand(composerText)");
  assert.ok(command > 0);
  assert.ok(command < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.match(submit, /requestThreadCompaction\(referenceThreadId\)/);
  assert.match(submit, /resumesThought\(\{/);
  assert.match(submit, /!runtime\.supportsTools \|\|/);
  const adapter = await readSrc("features/chat/api/chat-adapter.ts");
  // Taken whatever model sends next, so a switch to an API model cannot leave it armed for later.
  assert.match(adapter, /takeThreadCompactionRequest\(resolvedThreadId\) &&/);
  assert.match(adapter, /compactNow,\n\s+\}\),/);
});

test("a fit that recalled anchors before the recall it already emitted", () => {
  const recalled = fit(4, { checkpoint_started: true, recalled_chunks: 3 });
  const search = { toolCallId: "recall", toolName: "search_conversation" };
  const terminal = { toolCallId: "call-1", toolName: "terminal" };
  // At the start of a reply the recall is the only call so far: the cut goes above the prompt.
  assert.equal(compactionAnchorToolCallId([search], recalled), null);
  assert.equal(compactionAnchorToolCallId([terminal, search], recalled), "call-1");
  // Without a recall, a search the model ran itself is just the last call.
  assert.equal(compactionAnchorToolCallId([terminal, search], fit(4)), "recall");
  assert.equal(compactionAnchorToolCallId([], fit(4)), null);
});

test("a reply that reset on many tool rounds draws only its last reset", () => {
  const reset = (dropped: number) =>
    fit(0, { dropped_messages: dropped, checkpoint_started: true });
  const first = reset(4);
  const last = reset(2);
  const messages = [
    { id: "u1", role: "user" },
    {
      id: "a1",
      role: "assistant",
      parts: [toolCall("c1"), toolCall("c2"), toolCall("c3")],
      metadata: {
        custom: {
          contextTruncation: last,
          compactionMarkers: [
            { afterToolCallId: "c1", truncation: first },
            { afterToolCallId: "c2", truncation: reset(3) },
            { afterToolCallId: "c3", truncation: last },
          ],
        },
      },
    },
  ];

  assert.deepEqual([...compactionToolCallDividers(messages)], [["c3", last]]);
  assert.equal(compactionDividerCount(messages), 1);
});
