// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  compactionNoticeMessageIds,
  memoOnArray,
  partsHaveNonEmptyText,
  partsHaveRenderableRenderHtmlTool,
  partsPrecedingText,
  partsSearchImagesSignature,
  partsTextKey,
} from "../src/components/assistant-ui/message-derived.ts";
import {
  shouldShowCompactionNotice,
  compactionBoundary,
} from "../src/features/chat/utils/context-truncation.ts";
import {
  precedingTextForMessagePart,
  searchImagesSignature,
} from "../src/features/chat/search-images/search-images.ts";

test("memoOnArray computes once per array and again for a new array", () => {
  let calls = 0;
  const length = memoOnArray((a: readonly number[]) => {
    calls += 1;
    return a.length;
  });
  const a = [1, 2, 3];
  assert.equal(length(a), 3);
  assert.equal(length(a), 3);
  assert.equal(calls, 1);
  assert.equal(length([...a, 4]), 4);
  assert.equal(calls, 2);
});

const parts = [
  { type: "reasoning", text: "thinking" },
  { type: "text", text: "first answer" },
  {
    type: "tool-call",
    toolName: "render_html",
    toolCallId: "c1",
    args: { code: "<p>x</p>" },
    result: "Rendered HTML canvas",
  },
  { type: "text", text: "" },
  { type: "text", text: "second answer" },
];

test("cached part values equal the uncached computations", () => {
  assert.equal(
    partsTextKey(parts),
    JSON.stringify(["first answer", "", "second answer"]),
  );
  assert.equal(partsHaveRenderableRenderHtmlTool(parts), true);
  assert.equal(partsHaveRenderableRenderHtmlTool(parts.slice(0, 2)), false);
  assert.equal(partsHaveNonEmptyText(parts), true);
  assert.equal(partsHaveNonEmptyText([{ type: "text", text: "" }]), false);
  assert.equal(partsSearchImagesSignature(parts), searchImagesSignature(parts));
  for (let i = 0; i < parts.length; i += 1) {
    assert.equal(
      partsPrecedingText(parts, i),
      precedingTextForMessagePart(parts, i),
    );
  }
});

test("preceding text matches the uncached join on random part mixes, and is one string per index", () => {
  let seed = 7;
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const kinds = ["text", "reasoning", "tool-call", "text"];
  for (let round = 0; round < 300; round += 1) {
    const mix = Array.from({ length: Math.floor(random() * 12) }, (_, i) => {
      const type = kinds[Math.floor(random() * kinds.length)];
      return random() < 0.1
        ? { type }
        : { type, text: random() < 0.2 ? "" : `${type} ${round} ${i}` };
    });
    for (let i = 0; i <= mix.length + 1; i += 1) {
      assert.equal(
        partsPrecedingText(mix, i),
        precedingTextForMessagePart(mix, i),
      );
      assert.equal(partsPrecedingText(mix, i), partsPrecedingText(mix, i));
    }
  }
});

test("an appended part is a new array and is not served the old answer", () => {
  const streamed = [{ type: "text", text: "a" }];
  assert.equal(partsTextKey(streamed), '["a"]');
  const next = [{ type: "text", text: "ab" }];
  assert.equal(partsTextKey(next), '["ab"]');
  assert.equal(partsPrecedingText(next, 1), "ab");
});

// The walk AssistantMessage did per message before #12552, kept here as the oracle.
function walk(
  messages: readonly { id: string; role: string; metadata?: unknown }[],
  messageId: string,
): boolean {
  let previousDropped = 0;
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    const value = (
      message.metadata as
        | { custom?: { contextTruncation?: unknown } }
        | undefined
    )?.custom?.contextTruncation as Parameters<typeof compactionBoundary>[0];
    const dropped = compactionBoundary(value);
    if (shouldShowCompactionNotice(value, previousDropped)) {
      if (message.id === messageId) return true;
      previousDropped = Math.max(previousDropped, dropped);
    } else if (message.id === messageId) {
      return false;
    }
  }
  return false;
}

test("compaction notice ids match the per-message walk", () => {
  const truncation = (custom: Record<string, unknown>) => ({
    custom: { contextTruncation: custom },
  });
  const messages = [
    { id: "u0", role: "user" },
    { id: "a0", role: "assistant" },
    {
      id: "a1",
      role: "assistant",
      metadata: truncation({
        dropped_messages: 4,
        boundary_messages: 4,
        fits: true,
      }),
    },
    {
      id: "a2",
      role: "assistant",
      metadata: truncation({
        dropped_messages: 4,
        boundary_messages: 4,
        fits: true,
      }),
    },
    {
      id: "a3",
      role: "assistant",
      metadata: truncation({
        dropped_messages: 2,
        boundary_messages: 3,
        checkpoint_started: true,
      }),
    },
    {
      id: "a4",
      role: "assistant",
      metadata: truncation({
        dropped_messages: 8,
        boundary_messages: 9,
        fits: true,
      }),
    },
    {
      id: "a5",
      role: "assistant",
      metadata: truncation({ dropped_messages: 0 }),
    },
    {
      id: "a6",
      role: "assistant",
      metadata: truncation({ dropped_messages: 5, fits: false }),
    },
    {
      id: "u1",
      role: "user",
      metadata: truncation({ dropped_messages: 50, boundary_messages: 50 }),
    },
  ];
  const ids = compactionNoticeMessageIds(messages);
  for (const message of messages) {
    assert.equal(ids.has(message.id), walk(messages, message.id), message.id);
  }
  assert.deepEqual([...ids], ["a1", "a3", "a4"]);
  assert.equal(ids.has("missing"), false);
  assert.equal(compactionNoticeMessageIds(messages), ids);
  assert.notEqual(compactionNoticeMessageIds([...messages]), ids);
});
