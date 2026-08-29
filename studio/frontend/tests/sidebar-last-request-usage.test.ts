// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  selectSidebarLastRequestUsage,
} from "../src/features/chat/lib/sidebar-last-request-usage.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";

function assistant(
  contextUsage: unknown,
): MessageRecord {
  return {
    id: crypto.randomUUID(),
    threadId: "thread",
    role: "assistant",
    content: [],
    createdAt: Date.now(),
    metadata: contextUsage === undefined ? undefined : { contextUsage },
  };
}

const validUsage = {
  promptTokens: 120,
  completionTokens: 30,
  totalTokens: 999,
  cachedTokens: 20,
  cacheWriteTokens: 10,
};

test("uses the server total from the newest chronological assistant request", () => {
  const selected = selectSidebarLastRequestUsage([
    assistant({ ...validUsage, totalTokens: 150 }),
    assistant(validUsage),
  ]);

  assert.deepEqual(selected, { totalTokens: 999 });
});

test("does not fall back when the newest assistant request is partial or legacy", () => {
  assert.equal(
    selectSidebarLastRequestUsage([assistant(validUsage), assistant(undefined)]),
    undefined,
  );
  assert.equal(
    selectSidebarLastRequestUsage([
      assistant(validUsage),
      assistant({ promptTokens: 1, totalTokens: 2 }),
    ]),
    undefined,
  );
});

test("rejects invalid required and present cache counters", () => {
  for (const invalid of [
    { ...validUsage, promptTokens: -1 },
    { ...validUsage, completionTokens: Number.NaN },
    { ...validUsage, totalTokens: Number.POSITIVE_INFINITY },
    { ...validUsage, cachedTokens: -1 },
    { ...validUsage, cacheWriteTokens: "1" },
  ]) {
    assert.equal(selectSidebarLastRequestUsage([assistant(invalid)]), undefined);
  }
});

test("keeps valid zeroes and does not require optional cache counters", () => {
  assert.deepEqual(
    selectSidebarLastRequestUsage([
      assistant({ promptTokens: 0, completionTokens: 0, totalTokens: 0 }),
    ]),
    { totalTokens: 0 },
  );
});

test("carries usage only to single sidebar rows, never compare rows", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/hooks/use-chat-sidebar-items.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const compare = source.slice(
    source.indexOf('type: "compare"'),
    source.indexOf("} else if (!t.pairId)"),
  );
  const single = source.slice(source.indexOf('type: "single"'));

  assert.doesNotMatch(compare, /lastRequestUsage/);
  assert.match(single, /lastRequestUsage: t\.sidebarLastRequestUsage/);
});

test("the root-mounted sidebar loads and renders the last-request total", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
  const loader = source.slice(
    source.indexOf("useChatSidebarItems({"),
    source.indexOf("const pinnedIds", source.indexOf("useChatSidebarItems({")),
  );
  const rowStart = source.indexOf("function renderChatSidebarItem(");
  const row = source.slice(
    rowStart,
    source.indexOf("\n  return (\n    <Sidebar", rowStart),
  );

  assert.match(loader, /requireMessages: true/);
  assert.match(row, /data-testid="sidebar-last-request-usage"/);
  assert.match(row, /item\.lastRequestUsage\.totalTokens/);
  assert.match(row, /formatTokenCountFull/);
});
