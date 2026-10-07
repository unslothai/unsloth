// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import { unwrapPastedTextContent } from "../src/features/chat/utils/pasted-text.ts";
import { readSrc } from "./helpers/kit.ts";

type StoredMessage = {
  id: string;
  parentId: string | null;
  createdAt: number;
  role: string;
  content: string;
};

type FineTuneExport = {
  lines: string[];
  conversations: number;
  skipped: number;
};

const SOURCE = readSrc(
  "features/chat/prompt-storage/prompt-storage-dialog.tsx",
);

function sliceSource(startMarker: string, endMarker: string): string {
  const start = SOURCE.indexOf(startMarker);
  const end = SOURCE.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return SOURCE.slice(start, end);
}

function storedMessages(
  rows: [id: string, parentId: string | null, role: string, content: string][],
): StoredMessage[] {
  return rows.map(([id, parentId, role, content], index) => ({
    id,
    parentId,
    createdAt: index + 1,
    role,
    content,
  }));
}

function loadBuilder(chats: Record<string, StoredMessage[]>) {
  const javascript = ts.transpileModule(
    [
      sliceSource(
        "async function loadConversationMessages(",
        "function exportTs(",
      ),
      sliceSource(
        "// One JSONL line per conversation",
        "/** Download the fine-tuning JSONL",
      ),
      "globalThis.__build = buildFineTuneJsonl;",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const context = {
    exports: {},
    toast: { info: () => {} },
    listStoredChatThreads: async () =>
      Object.keys(chats).map((id) => ({ id })),
    listStoredChatMessages: async (id: string) => chats[id],
    ...liveThreadHead,
    orderByParentChain,
    unwrapPastedTextContent,
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__build as (format: string) => Promise<FineTuneExport>;
}

async function fineTuneMessages(
  chats: Record<string, StoredMessage[]>,
  liveBranch: string[] | null = null,
  openedEarlier: string[] = [],
) {
  const build = loadBuilder(chats);
  const unregisters = liveBranch
    ? [...openedEarlier, "open"].map((remoteId) =>
        liveThreadHead.registerLiveThreadView({
          threads: () => ({ getState: () => ({ mainThreadId: "open" }) }),
          threadListItem: () => ({ getState: () => ({ id: remoteId, remoteId }) }),
          thread: () => ({
            getState: () => ({ messages: liveBranch.map((id) => ({ id })) }),
          }),
        }),
      )
    : [];
  try {
    const { lines } = await build("openai");
    return [...lines].map((line) => JSON.parse(line).messages);
  } finally {
    for (const unregister of unregisters) unregister();
  }
}

const regenerated = (prefix = "") =>
  storedMessages([
    [`${prefix}u1`, null, "user", "Name one fruit."],
    [`${prefix}a1`, `${prefix}u1`, "assistant", "Apples."],
    [`${prefix}a1-retry`, `${prefix}u1`, "assistant", "Pears."],
  ]);

test("chat fine-tune data uses the reply picked in the branch picker", async () => {
  assert.deepEqual(
    await fineTuneMessages(
      { open: regenerated(), other: regenerated() },
      ["u1", "a1"],
    ),
    [
      [
        { role: "user", content: "Name one fruit." },
        { role: "assistant", content: "Apples." },
      ],
      [
        { role: "user", content: "Name one fruit." },
        { role: "assistant", content: "Pears." },
      ],
    ],
  );
});

test("chat fine-tune data uses the newest reply when no chat is open", async () => {
  assert.deepEqual(await fineTuneMessages({ open: regenerated() }), [
    [
      { role: "user", content: "Name one fruit." },
      { role: "assistant", content: "Pears." },
    ],
  ]);
});

test("chat fine-tune data keeps chats opened earlier on their newest reply", async () => {
  assert.deepEqual(
    await fineTuneMessages(
      { open: regenerated(), other: regenerated("other-") },
      ["u1", "a1"],
      ["other"],
    ),
    [
      [
        { role: "user", content: "Name one fruit." },
        { role: "assistant", content: "Apples." },
      ],
      [
        { role: "user", content: "Name one fruit." },
        { role: "assistant", content: "Pears." },
      ],
    ],
  );
});
