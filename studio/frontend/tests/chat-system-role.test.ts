// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat importers keep system records (backup import downgrades them on purpose); every converter must too.

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";
import type { ThreadMessage } from "@assistant-ui/react";

import type { MessageRecord } from "../src/features/chat/types.ts";
import { readSrc } from "./helpers/kit.ts";

const PROVIDER = readSrc("features/chat/runtime-provider.tsx");
const SYNC = readSrc("features/chat/utils/delete-thread-message.ts");
const REFRESH = readSrc("features/chat/utils/refresh-context-usage.ts");
const IMPORT = readSrc("features/chat/utils/chat-import.ts");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

function slice(source: string, startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return source.slice(start, end);
}

function run<T>(source: string, exportLine: string, context: Record<string, unknown>): T {
  const unexported = source.replace(/\bexport\s+(?=(async\s+)?function)/g, "");
  const javascript = ts.transpileModule(`${unexported}\n${exportLine}`, {
    compilerOptions: { module: ts.ModuleKind.None, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  vm.runInNewContext(javascript, context);
  return context.__fn as T;
}

function loadToThreadMessage(): (m: MessageRecord) => ThreadMessage {
  const source = slice(
    PROVIDER,
    "function toThreadMessage(",
    "type GenerationRecovery",
  );
  return run(source, "globalThis.__fn = toThreadMessage;", {
    cloneContent: (content: unknown) => (Array.isArray(content) ? [...content] : []),
    cloneAttachments: () => [],
    restoredAssistantStatus: () => ({ type: "complete", reason: "stop" }),
    generationIsCorroboratedLive: () => false,
    Date,
  });
}

function loadExportedItemToRecord(): (
  threadId: string,
  parentId: string | null,
  message: ThreadMessage,
) => MessageRecord {
  const source = slice(
    SYNC,
    "function snapshotContent(",
    "async function withStoredResearchMessages",
  );
  return run(source, "globalThis.__fn = exportedItemToRecord;", { Date });
}

function loadStoredMessageToRunMessage(): (record: MessageRecord) => ThreadMessage {
  const source = slice(
    REFRESH,
    "function storedMessageToRunMessage(",
    "function foldHash(",
  );
  return run(source, "globalThis.__fn = storedMessageToRunMessage;", {
    structuredClone,
    Date,
  });
}

function loadMarkdownToRecords(): (
  messages: Array<{ role: string; content: string }>,
  threadId: string,
  baseTs: number,
) => MessageRecord[] {
  const source = slice(
    IMPORT,
    "function markdownToRecords(",
    "function csvToRecords(",
  );
  return run(source, "globalThis.__fn = markdownToRecords;", { crypto });
}

const SYSTEM_RECORD: MessageRecord = {
  id: "sys-1",
  threadId: "t1",
  parentId: null,
  role: "system",
  content: [{ type: "text", text: "Be terse." }],
  metadata: {},
  createdAt: 1_700_000_000_000,
};

test("a stored system message loads as a system ThreadMessage, not assistant", () => {
  const toThreadMessage = loadToThreadMessage();
  const message = toThreadMessage(SYSTEM_RECORD);
  assert.equal(message.role, "system");
  const parts = message.content as readonly { type: string; text: string }[];
  assert.equal(parts.length, 1);
  assert.equal(parts[0].text, "Be terse.");
});

test("saving a system ThreadMessage writes the system role back", () => {
  const exportedItemToRecord = loadExportedItemToRecord();
  const record = exportedItemToRecord("t1", null, {
    id: "sys-1",
    role: "system",
    content: [{ type: "text", text: "Be terse." }],
    createdAt: new Date(1_700_000_000_000),
    metadata: { custom: {} },
  });
  assert.equal(record.role, "system");
  assert.equal(record.createdAt, 1_700_000_000_000);
});

test("a system record survives a load-then-autosave round trip unchanged", () => {
  const toThreadMessage = loadToThreadMessage();
  const exportedItemToRecord = loadExportedItemToRecord();
  const roundTripped = exportedItemToRecord(
    "t1",
    null,
    toThreadMessage(SYSTEM_RECORD),
  );
  assert.equal(roundTripped.role, "system");
  assert.equal(roundTripped.createdAt, 1_700_000_000_000);
  const parts = roundTripped.content as unknown as readonly { text: string }[];
  assert.equal(parts[0].text, "Be terse.");
});

test("user and assistant roles still map to themselves", () => {
  const toThreadMessage = loadToThreadMessage();
  const exportedItemToRecord = loadExportedItemToRecord();
  for (const role of ["user", "assistant"] as const) {
    const record = { ...SYSTEM_RECORD, role };
    assert.equal(toThreadMessage(record).role, role);
    assert.equal(
      exportedItemToRecord("t1", null, toThreadMessage(record)).role,
      role,
    );
  }
});

test("the token-count reader keeps a stored system record as system", () => {
  const storedMessageToRunMessage = loadStoredMessageToRunMessage();
  const message = storedMessageToRunMessage(SYSTEM_RECORD);
  assert.equal(message.role, "system");
  const parts = message.content as readonly { type: string; text: string }[];
  assert.equal(parts.length, 1);
  assert.equal(parts[0].text, "Be terse.");
});

test("a markdown Developer heading imports as a system record", () => {
  const markdownToRecords = loadMarkdownToRecords();
  const records = markdownToRecords(
    [
      { role: "Developer", content: "Be terse." },
      { role: "User", content: "Hi" },
    ],
    "t1",
    0,
  );
  assert.equal(records[0].role, "system");
  assert.equal(records[1].role, "user");
});

test("an unrecognized markdown role imports as user, not a folded assistant", () => {
  const markdownToRecords = loadMarkdownToRecords();
  const records = markdownToRecords(
    [{ role: "Helper", content: "Something" }],
    "t1",
    0,
  );
  assert.equal(records[0].role, "user");
});

type Outbound = { role: string; content: unknown };
const SECOND_SYSTEM_UNSHIFT =
  /outboundMessages\.unshift\(\{\s*role: "system",\s*content: combinedSystemPrompt/;

test("the Studio prompt merges into an imported leading system turn instead of adding a second", () => {
  const prepend = run<(messages: Outbound[], text: string) => void>(
    slice(
      ADAPTER,
      "function prependSystemPrompt",
      "function toOpenAIMessages(",
    ),
    "globalThis.__fn = prependSystemPrompt;",
    {},
  );
  const imported: Outbound[] = [
    { role: "system", content: "Imported." },
    { role: "user", content: "Hi" },
  ];
  prepend(imported, "Studio.");
  assert.deepEqual(JSON.parse(JSON.stringify(imported)), [
    { role: "system", content: "Studio.\n\nImported." },
    { role: "user", content: "Hi" },
  ]);
  const plain: Outbound[] = [{ role: "user", content: "Hi" }];
  prepend(plain, "Studio.");
  assert.deepEqual(JSON.parse(JSON.stringify(plain)), [
    { role: "system", content: "Studio." },
    { role: "user", content: "Hi" },
  ]);
  assert.doesNotMatch(ADAPTER, SECOND_SYSTEM_UNSHIFT);
});
