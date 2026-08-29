// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  applySidebarAssistantUsageUpdate,
  newestSidebarAssistantUsageUpdate,
  selectSidebarLastRequestUsage,
  selectSidebarLastRequestUsageFromMetadata,
} from "../src/features/chat/lib/sidebar-last-request-usage.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";

let nextCreatedAt = 0;

function assistant(
  contextUsage: unknown,
  overrides: Partial<MessageRecord> = {},
): MessageRecord {
  return {
    id: crypto.randomUUID(),
    threadId: "thread",
    role: "assistant",
    content: [],
    createdAt: ++nextCreatedAt,
    metadata: contextUsage === undefined ? undefined : { contextUsage },
    ...overrides,
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
  const older = assistant({ ...validUsage, totalTokens: 150 });
  const newer = assistant(validUsage);
  const selected = selectSidebarLastRequestUsage([newer, older]);

  assert.deepEqual(selected, { totalTokens: 999 });
});

test("breaks equal timestamps by message id", () => {
  assert.deepEqual(
    selectSidebarLastRequestUsage([
      assistant({ ...validUsage, totalTokens: 150 }, { createdAt: 1, id: "a" }),
      assistant(validUsage, { createdAt: 1, id: "b" }),
    ]),
    { totalTokens: 999 },
  );
});

test("does not fall back when the newest assistant request is partial or legacy", () => {
  assert.equal(
    selectSidebarLastRequestUsage([
      assistant(validUsage),
      assistant(undefined),
    ]),
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
    assert.equal(
      selectSidebarLastRequestUsage([assistant(invalid)]),
      undefined,
    );
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

test("validates a sidebar summary with the same rules as a saved message", () => {
  assert.deepEqual(
    selectSidebarLastRequestUsageFromMetadata({ contextUsage: validUsage }),
    { totalTokens: 999 },
  );
  assert.equal(
    selectSidebarLastRequestUsageFromMetadata({
      contextUsage: { promptTokens: 1, totalTokens: 2 },
    }),
    undefined,
  );
});

test("immediately clears and replaces usage only for the changed thread", () => {
  const threads = [
    { id: "a", sidebarLastRequestUsage: { totalTokens: 100 } },
    { id: "b", sidebarLastRequestUsage: { totalTokens: 200 } },
  ];
  const partial = applySidebarAssistantUsageUpdate(
    threads,
    newestSidebarAssistantUsageUpdate("a", [assistant(undefined)]),
  );

  assert.equal(partial[0].sidebarLastRequestUsage, undefined);
  assert.deepEqual(partial[1].sidebarLastRequestUsage, { totalTokens: 200 });

  const completed = applySidebarAssistantUsageUpdate(
    partial,
    newestSidebarAssistantUsageUpdate("a", [assistant(validUsage)]),
  );
  assert.deepEqual(completed[0].sidebarLastRequestUsage, { totalTokens: 999 });
  assert.deepEqual(completed[1].sidebarLastRequestUsage, { totalTokens: 200 });
});

test("ignores a late autosave from an older assistant generation", () => {
  const older = assistant(
    { ...validUsage, totalTokens: 150 },
    { id: "assistant-old", createdAt: 10 },
  );
  const newer = assistant(validUsage, {
    id: "assistant-new",
    createdAt: 20,
  });
  const current = applySidebarAssistantUsageUpdate(
    [
      {
        id: "thread",
        sidebarLastRequestUsage: undefined,
        lastAssistantId: undefined,
        lastAssistantCreatedAt: undefined,
      },
    ],
    newestSidebarAssistantUsageUpdate("thread", [newer]),
  );
  const afterOlderSave = applySidebarAssistantUsageUpdate(
    current,
    newestSidebarAssistantUsageUpdate("thread", [older]),
  );

  assert.deepEqual(afterOlderSave[0].sidebarLastRequestUsage, {
    totalTokens: 999,
  });
  assert.equal(afterOlderSave[0].lastAssistantId, "assistant-new");
  assert.equal(afterOlderSave[0].lastAssistantCreatedAt, 20);
});

test("a synced thread with no assistant clears its saved usage", () => {
  const update = newestSidebarAssistantUsageUpdate("thread", [
    { ...assistant(validUsage), role: "user" },
  ]);

  assert.deepEqual(update, { threadId: "thread", hasAssistant: false });
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

  assert.match(loader, /requireMessages: false/);
  assert.match(loader, /includeLastRequestUsage: true/);
  assert.match(row, /data-testid="sidebar-last-request-usage"/);
  assert.match(row, /item\.lastRequestUsage\.totalTokens/);
  assert.match(row, /formatTokenCountFull/);
});
