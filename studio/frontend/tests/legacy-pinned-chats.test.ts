// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  readLegacyPinnedChatIds,
  writeLocalPinnedChatIds,
} from "../src/features/chat/utils/legacy-pinned-chats.ts";

function storage(pins: unknown, order?: unknown) {
  const values = new Map<string, string>([
    ["unsloth_pinned_chats", JSON.stringify({ state: { pinnedIds: pins } })],
    [
      "unsloth_sidebar_organization",
      JSON.stringify({ state: { manualOrder: { pinned: order } } }),
    ],
  ]);
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  };
}

test("migration retains comparison IDs and the existing manual pin order", () => {
  const saved = storage(
    ["new", "pair-id", "chat-a", "chat-b"],
    ["chat-b", "chat-a", "pair-id"],
  );
  assert.deepEqual(readLegacyPinnedChatIds(saved), [
    "new",
    "chat-b",
    "chat-a",
    "pair-id",
  ]);
});

test("stale order entries do not resurrect unpinned chats", () => {
  assert.deepEqual(readLegacyPinnedChatIds(storage(["a"], ["deleted", "a"])), [
    "a",
  ]);
});

test("legacy migration accepts only unique bounded string IDs", () => {
  assert.deepEqual(
    readLegacyPinnedChatIds(
      storage(["a", "a", null, 12, "", "x".repeat(513), "b"]),
    ),
    ["a", "b"],
  );
  assert.equal(
    readLegacyPinnedChatIds(
      storage(Array.from({ length: 10_001 }, (_, i) => String(i))),
    ).length,
    10_000,
  );
});

test("bad sidebar JSON does not discard valid pins", () => {
  const saved = storage(["a"]);
  saved.setItem("unsloth_sidebar_organization", "{");
  assert.deepEqual(readLegacyPinnedChatIds(saved), ["a"]);
});

test("missing, malformed, and inaccessible browser storage can be ignored", () => {
  assert.deepEqual(readLegacyPinnedChatIds({ getItem: () => null }), []);
  assert.deepEqual(readLegacyPinnedChatIds({ getItem: () => "{" }), []);
  assert.deepEqual(
    readLegacyPinnedChatIds({
      getItem: () => {
        throw new Error("blocked");
      },
    }),
    [],
  );
});

test("legacy migration filters malformed UTF-16 IDs without losing valid pins", () => {
  assert.deepEqual(
    readLegacyPinnedChatIds(
      storage(["a", "\ud800", "\udfff", "\ud83d\udcac", "b"]),
    ),
    ["a", "\ud83d\udcac", "b"],
  );
});

test("local compatibility writes retain new order without reapplying legacy order", () => {
  const saved = storage(["a", "b"], ["b", "a"]);
  writeLocalPinnedChatIds(saved, ["a", "b"]);
  assert.deepEqual(readLegacyPinnedChatIds(saved), ["a", "b"]);
  writeLocalPinnedChatIds(saved, []);
  assert.deepEqual(readLegacyPinnedChatIds(saved), []);
});
