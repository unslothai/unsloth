// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { SubagentRecord } from "../src/features/chat/types/subagents.ts";
import {
  SUBAGENT_LIST_POLL_ACTIVE_MS,
  SUBAGENT_LIST_POLL_IDLE_MS,
  SUBAGENT_TOOL_NAME,
  formatSubagentDuration,
  mergeSubagentDetail,
  mergeSubagentList,
  shortSubagentModel,
  shouldAutoOpenSubagentPanel,
  subagentElapsedMs,
  subagentForDescription,
  subagentListPollInterval,
  summarizeSubagentArguments,
} from "../src/features/chat/utils/subagents.ts";
import { readSrc } from "./helpers/kit.ts";

function record(overrides: Partial<SubagentRecord> = {}): SubagentRecord {
  return {
    id: "sa_1",
    parentThreadId: "t1",
    childThreadId: "c1",
    sessionId: "s1",
    description: "Inspect backend",
    prompt: "look around",
    model: "openai/gpt-5.2",
    tools: ["web_search"],
    status: "running",
    error: null,
    resultPreview: null,
    toolCallCount: 0,
    pendingApproval: null,
    startedAt: 1000,
    finishedAt: null,
    ...overrides,
  };
}

test("list polls fast while a subagent runs or the thread generates, slowly otherwise", () => {
  assert.equal(
    subagentListPollInterval({ anyRunning: true, threadGenerating: false }),
    SUBAGENT_LIST_POLL_ACTIVE_MS,
  );
  assert.equal(
    subagentListPollInterval({ anyRunning: false, threadGenerating: true }),
    SUBAGENT_LIST_POLL_ACTIVE_MS,
  );
  assert.equal(
    subagentListPollInterval({ anyRunning: false, threadGenerating: false }),
    SUBAGENT_LIST_POLL_IDLE_MS,
  );
  assert.equal(SUBAGENT_LIST_POLL_ACTIVE_MS, 1500);
  assert.equal(SUBAGENT_LIST_POLL_IDLE_MS, 10_000);
});

test("an unchanged list poll returns the previous array and rows", () => {
  const first = mergeSubagentList(undefined, [
    record({ id: "a", startedAt: 2 }),
    record({ id: "b", startedAt: 1 }),
  ]);
  assert.deepEqual(
    first.map((row) => row.id),
    ["b", "a"],
    "oldest first",
  );
  const again = mergeSubagentList(first, [
    record({ id: "b", startedAt: 1 }),
    record({ id: "a", startedAt: 2 }),
  ]);
  assert.equal(again, first);
});

test("a changed row is replaced while untouched rows keep their identity", () => {
  const first = mergeSubagentList(undefined, [
    record({ id: "a", startedAt: 1 }),
    record({ id: "b", startedAt: 2 }),
  ]);
  const next = mergeSubagentList(first, [
    record({ id: "a", startedAt: 1 }),
    record({
      id: "b",
      startedAt: 2,
      status: "completed",
      finishedAt: 9,
      resultPreview: "done",
    }),
  ]);
  assert.notEqual(next, first);
  assert.equal(next[0], first[0]);
  assert.equal(next[1].status, "completed");
});

test("a row the server dropped leaves the list", () => {
  const first = mergeSubagentList(undefined, [
    record({ id: "a" }),
    record({ id: "b", startedAt: 2 }),
  ]);
  const next = mergeSubagentList(first, [record({ id: "b", startedAt: 2 })]);
  assert.deepEqual(
    next.map((row) => row.id),
    ["b"],
  );
});

test("a pending approval change counts as a change", () => {
  const first = mergeSubagentList(undefined, [record()]);
  const next = mergeSubagentList(first, [
    record({
      pendingApproval: {
        approvalId: "ap1",
        toolName: "terminal",
        toolCallId: "tc1",
        arguments: { command: "ls" },
      },
    }),
  ]);
  assert.notEqual(next, first);
  assert.equal(next[0].pendingApproval?.approvalId, "ap1");
});

test("detail merging keeps the cached record until the transcript or status moves", () => {
  const cached = record({
    transcript: [{ kind: "text", text: "hello" }],
  });
  assert.equal(
    mergeSubagentDetail(
      cached,
      record({ transcript: [{ kind: "text", text: "hello" }] }),
    ),
    cached,
  );
  const grown = record({
    transcript: [
      { kind: "text", text: "hello" },
      { kind: "text", text: "world" },
    ],
  });
  assert.equal(mergeSubagentDetail(cached, grown), grown);
});

test("the panel opens by itself for a thread's first subagent only", () => {
  const running = [{ status: "running" as const }];
  const done = [{ status: "completed" as const }];
  // Polled empty before, now has one: a fresh spawn.
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: 0,
      next: running,
      alreadyAutoOpened: false,
    }),
    true,
  );
  // Never polled and already finished: an old chat, stays closed.
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: undefined,
      next: done,
      alreadyAutoOpened: false,
    }),
    false,
  );
  // Never polled but one is still running: the first poll was late.
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: undefined,
      next: running,
      alreadyAutoOpened: false,
    }),
    true,
  );
  // Later subagents and a closed panel are left alone.
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: 1,
      next: [...running, ...running],
      alreadyAutoOpened: false,
    }),
    false,
  );
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: 0,
      next: running,
      alreadyAutoOpened: true,
    }),
    false,
  );
  assert.equal(
    shouldAutoOpenSubagentPanel({
      prevCount: 0,
      next: [],
      alreadyAutoOpened: false,
    }),
    false,
  );
});

test("elapsed time runs to now while running and freezes when finished", () => {
  assert.equal(subagentElapsedMs(record({ startedAt: 1000 }), 4500), 3500);
  assert.equal(
    subagentElapsedMs(
      record({ status: "completed", startedAt: 1000, finishedAt: 3000 }),
      99999,
    ),
    2000,
  );
  assert.equal(
    subagentElapsedMs(record({ status: "cancelled", startedAt: 1000 }), 99999),
    0,
  );
  assert.equal(formatSubagentDuration(900), "0s");
  assert.equal(formatSubagentDuration(42_000), "42s");
  assert.equal(formatSubagentDuration(125_000), "2m 05s");
  assert.equal(formatSubagentDuration(3_725_000), "1h 02m");
});

test("display helpers", () => {
  assert.equal(shortSubagentModel("openai/gpt-5.2"), "gpt-5.2");
  assert.equal(shortSubagentModel("a/b/c-model"), "c-model");
  assert.equal(shortSubagentModel("local-model"), "local-model");
  assert.equal(shortSubagentModel(null), "");
  assert.equal(
    summarizeSubagentArguments({ command: "ls -la", n: 3, skip: null }),
    "command: ls -la, n: 3",
  );
  assert.equal(
    summarizeSubagentArguments({ q: "x".repeat(50) }, 10).length,
    10,
  );
});

test("a tool card finds its subagent only when the description is unique", () => {
  const rows = [
    record({ id: "a", description: "Read docs" }),
    record({ id: "b", description: "Run tests" }),
    record({ id: "c", description: "Run tests" }),
  ];
  assert.equal(subagentForDescription(rows, "Read docs")?.id, "a");
  assert.equal(subagentForDescription(rows, "Run tests"), null);
  assert.equal(subagentForDescription(rows, "Nope"), null);
  assert.equal(subagentForDescription(undefined, "Read docs"), null);
});

test("the toggle puts spawn_subagent into the external Studio-tools request", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.equal(SUBAGENT_TOOL_NAME, "spawn_subagent");
  assert.match(
    adapter,
    /\.\.\.\(subagentsEnabled \? \[SUBAGENT_TOOL_NAME\] : \[\]\)/,
  );
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    store,
    /subagentsEnabled: loadBool\(CHAT_SUBAGENTS_ENABLED_KEY, false\)/,
  );
});
