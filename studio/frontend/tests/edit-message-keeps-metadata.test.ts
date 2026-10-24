// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import type { ThreadMessage } from "@assistant-ui/react";
import ts from "@typescript/typescript6";
import type { MessageRecord } from "../src/features/chat/types.ts";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import * as savedUsage from "../src/features/chat/utils/saved-context-usage.ts";
import {
  branchContextUsage,
  contextUsageMarkers,
  revisionTotal,
  threadBranchRecords,
} from "../src/features/chat/utils/saved-context-usage.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Module = {
  extractTaggedText: (content: unknown) => string;
  updateThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
    newText: string;
    isIncognito: boolean;
  }) => Promise<unknown>;
};

const RECORD_MODULE = loadWithStubs<{
  exportedItemToRecord: (
    threadId: string,
    parentId: string | null,
    message: unknown,
  ) => MessageRecord;
}>(
  new URL(
    "../src/features/chat/utils/delete-thread-message.ts",
    import.meta.url,
  ),
  {
    "@assistant-ui/core/internal": { MessageRepository },
    "../api/chat-api": { listChatMessages: async () => [] },
    "./chat-history-storage": {
      ensureStoredChatThread: async () => {},
      syncStoredChatMessages: async (
        _threadId: string,
        records: Record<string, unknown>[],
      ) => records,
    },
    "./research-message-sync": researchSync,
    "./saved-context-usage": savedUsage,
  },
);

const CUSTOM = {
  incomplete: { reason: "length" },
  contextUsage: { promptTokens: 900, contextLength: 4096 },
  timing: { tokensPerSecond: 42.5, durationMs: 1200 },
  contextTruncation: { dropped: 4, fits: true },
};

const provider = ts.createSourceFile(
  "runtime-provider.tsx",
  readSrc("features/chat/runtime-provider.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const hydration = ["cloneContent", "cloneAttachments", "toThreadMessage"].map((name) => {
  const declaration = provider.statements.find(
    (node) => ts.isFunctionDeclaration(node) && node.name?.text === name,
  );
  assert.ok(declaration, `${name} was not found`);
  return declaration.getText(provider);
}).join("\n");
const hydrateUser = new Function(ts.transpileModule(
  `${hydration}\nreturn toThreadMessage;`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } },
).outputText)() as (record: MessageRecord) => ThreadMessage;

function harness(persist?: () => Promise<void>) {
  const saved: Record<string, unknown>[] = [];
  const module = loadWithStubs<Module>(
    new URL(
      "../src/features/chat/utils/update-thread-message.ts",
      import.meta.url,
    ),
    {
      "../api/chat-api": {
        saveChatMessage: async (record: Record<string, unknown>) => {
          saved.push(record);
          await persist?.();
          return record;
        },
      },
      "./delete-thread-message": RECORD_MODULE,
      "./research-message-sync": researchSync,
      "./saved-context-usage": savedUsage,
    },
  );
  return { module, saved };
}

function deferred() {
  let resolve!: () => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<void>((accept, refuse) => {
    resolve = accept;
    reject = refuse;
  });
  return { promise, resolve, reject };
}

function thread(content: unknown[]): Exported {
  return {
    headId: "a0",
    messages: [
      {
        parentId: null,
        message: {
          id: "u0",
          role: "user",
          content: [{ type: "text", text: "hello" }],
          createdAt: new Date(1000),
        },
      },
      {
        parentId: "u0",
        message: {
          id: "a0",
          role: "assistant",
          content,
          createdAt: new Date(2000),
          metadata: { custom: CUSTOM },
        },
      },
    ],
  };
}

async function save(
  newText: string,
  content: unknown[] = [{ type: "text", text: "the original reply" }],
) {
  const h = harness();
  const exported = thread(content);
  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText,
    isIncognito: false,
  });
  assert.equal(h.saved.length, 1, "the edit was never written");
  return h.saved[0];
}

function withoutRevision(metadata: unknown): Record<string, unknown> {
  const { revision, ...kept } = metadata as Record<string, unknown>;
  assert.equal(revision, 1);
  return kept;
}

test("an edited reply keeps the metadata the turn was stored with", async () => {
  const record = await save("an edited reply");

  assert.deepEqual(withoutRevision(record.metadata), CUSTOM);
});

test("saving a reply unchanged does not mark it revised, however the parts were stored", async () => {
  for (const content of [
    [{ type: "text", text: "the original reply" }],
    [{ type: "reasoning", text: "\nlet me think\n" }, { type: "text", text: "the reply" }],
    [{ type: "text", text: "one" }, { type: "text", text: "two" }],
    [{ type: "text", text: "the reply", status: { type: "complete" } }],
  ]) {
    const h = harness();
    const exported = thread(content);
    await h.module.updateThreadMessage({
      thread: { export: () => exported, import: () => {} },
      messageId: "a0",
      remoteId: "remote-1",
      newText: h.module.extractTaggedText(content),
      isIncognito: false,
    });
    assert.equal("revision" in ((h.saved[0].metadata ?? {}) as Record<string, unknown>), false);
  }
});

for (const modelId of ["unsloth/gguf-model", "external::openai::gpt-x"]) {
  for (const editedId of ["u0", "a0"]) {
    test(`an edit after reloading a revised user invalidates saved context: ${modelId}, ${editedId}`, async () => {
      const h = harness();
      const user: MessageRecord = {
        id: "u0",
        threadId: "remote-1",
        role: "user",
        content: [{ type: "text", text: "edited prompt" }],
        createdAt: 1000,
        metadata: { revision: 1, createdAtEstimated: true },
      };
      const assistant: MessageRecord = {
        id: "a0",
        threadId: "remote-1",
        parentId: "u0",
        role: "assistant",
        content: [{ type: "text", text: "the original reply" }],
        createdAt: 2000,
      };
      const usage = {
        promptTokens: 90,
        completionTokens: 10,
        totalTokens: 100,
        cachedTokens: 0,
        modelId,
        ...contextUsageMarkers([user, assistant]),
      };
      let exported = thread(assistant.content as unknown[]);
      exported.messages[0].message = hydrateUser(user) as unknown as Record<string, unknown>;
      exported.messages[1].message.metadata = { custom: { contextUsage: usage } };

      await h.module.updateThreadMessage({
        thread: { export: () => exported, import: (data) => { exported = data; } },
        messageId: editedId,
        remoteId: "remote-1",
        newText: "a much longer edited message ".repeat(10),
        isIncognito: false,
      });

      const messages = exported.messages.map(({ message }) => message) as unknown as ThreadMessage[];
      const window = modelId.startsWith("external::") ? null : 4096;
      const shown = branchContextUsage(threadBranchRecords(messages), modelId, window, window);
      assert.ok(shown);
      assert.equal(shown.estimated, true, "the pre-edit count was accepted as exact");
      assert.equal(revisionTotal(messages), 2);
      assert.equal(h.saved.length, 1);
    });
  }
}

test("reloading and saving a user unchanged preserves its revision and timestamp flag", async () => {
  const h = harness();
  const user: MessageRecord = {
    id: "u0",
    threadId: "remote-1",
    role: "user",
    content: [{ type: "text", text: "edited prompt" }],
    createdAt: 1000,
    metadata: { revision: 3, createdAtEstimated: true },
  };
  const exported = thread([]);
  exported.messages[0].message = hydrateUser(user) as unknown as Record<string, unknown>;
  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "u0",
    remoteId: "remote-1",
    newText: h.module.extractTaggedText(user.content),
    isIncognito: false,
  });
  assert.deepEqual(h.saved[0].metadata, user.metadata);
});

test("editing clears stale Gemini continuation replay and keeps other metadata", async () => {
  const record = await saveOwnedReply({
    ...CUSTOM,
    geminiContinuationReplay: {
      turns: [{ text: "the original reply", thoughtSignature: "SIG-OLD" }],
      visiblePrefix: "the original reply",
      stripVisiblePrefix: true,
    },
  });

  assert.deepEqual(withoutRevision(record.metadata), CUSTOM);
  assert.equal(
    "geminiContinuationReplay" in (record.metadata as Record<string, unknown>),
    false,
  );
});

test("the edit still rewrites the content and its identity", async () => {
  const record = await save("an edited reply");

  assert.equal(record.id, "a0");
  assert.equal(record.threadId, "remote-1");
  assert.equal(record.parentId, "u0");
  assert.equal(record.role, "assistant");
  assert.equal(record.createdAt, 2000);
  assert.deepEqual(record.content, [{ type: "text", text: "an edited reply" }]);
});

test("a reply with no metadata of its own writes only its revision", async () => {
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);
  const target = exported.messages[1].message;
  target.metadata = { custom: {} };

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });

  assert.deepEqual(withoutRevision(h.saved[0].metadata), {});
});

test("an incognito edit is never written at all", async () => {
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: true,
  });

  assert.equal(h.saved.length, 0);
});

test("an edit updates content and token-count revision before the existing save settles", async () => {
  let accept!: () => void;
  const pending = new Promise<void>((resolve) => { accept = resolve; });
  const h = harness(() => pending);
  let exported = thread([{ type: "text", text: "original" }]);
  const published: Exported[] = [];
  const save = h.module.updateThreadMessage({
    thread: {
      export: () => exported,
      import: (next) => { exported = next; published.push(next); },
    },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });

  assert.equal(h.saved.length, 1);
  assert.equal(published.length, 1);
  assert.deepEqual(exported.messages[1].message.content, [{ type: "text", text: "edited" }]);
  assert.equal(revisionTotal(exported.messages.map(({ message }) => message) as unknown as ThreadMessage[]), 1);
  accept();
  await save;
  assert.equal(published.length, 1);
  const messages = exported.messages.map(({ message }) => message) as unknown as ThreadMessage[];
  assert.equal(revisionTotal(messages), 1);
  assert.deepEqual(messages[1].content, [{ type: "text", text: "edited" }]);
});

test("upstream edit rollback restores content, revision and the saved token count", async () => {
  const error = new Error("server-managed generation messages cannot be edited");
  const write = deferred();
  const h = harness(() => write.promise);
  let exported = thread([{ type: "text", text: "original" }]);
  const saved = { modelId: "local-model", promptTokens: 42, completionTokens: 8, totalTokens: 50, cachedTokens: 0 };
  exported.messages[1].message.metadata = { custom: { contextUsage: saved } };
  const shownUsage = () => branchContextUsage(
    threadBranchRecords(exported.messages.map(({ message }) => message) as unknown as ThreadMessage[]),
    "local-model", 4096, 4096,
  );

  const saving = h.module.updateThreadMessage({
    thread: {
      export: () => exported,
      import: (next) => { exported = next; },
    },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });
  assert.notEqual(shownUsage(), saved);
  const rejected = assert.rejects(saving, error);
  write.reject(error);
  await rejected;

  assert.equal(shownUsage(), saved);
  const messages = exported.messages.map(({ message }) => message) as unknown as ThreadMessage[];
  assert.equal(revisionTotal(messages), 0);
  assert.deepEqual(messages[1].content, [{ type: "text", text: "original" }]);
});

test("token-count revisions do not serialize or delay upstream edit requests", async () => {
  const firstWrite = deferred();
  const secondWrite = deferred();
  const writes = [firstWrite, secondWrite];
  let writeIndex = 0;
  const h = harness(() => writes[writeIndex++].promise);
  let exported = thread([{ type: "text", text: "original" }]);
  const edit = (newText: string) => h.module.updateThreadMessage({
    thread: {
      export: () => exported,
      import: (next) => { exported = next; },
    },
    messageId: "a0",
    remoteId: "remote-1",
    newText,
    isIncognito: false,
  });
  const first = edit("edit A");
  const second = edit("edit B");
  assert.equal(h.saved.length, 2);
  assert.deepEqual(exported.messages[1].message.content, [{ type: "text", text: "edit B" }]);

  secondWrite.resolve();
  await second;
  firstWrite.resolve();
  await Promise.all([first, second]);

  assert.deepEqual(h.saved.map((record) => record.content), [
    [{ type: "text", text: "edit A" }],
    [{ type: "text", text: "edit B" }],
  ]);
  assert.deepEqual(h.saved.map((record) => (record.metadata as Record<string, unknown>).revision), [1, 2]);
  assert.deepEqual(exported.messages[1].message.content, [{ type: "text", text: "edit B" }]);
  assert.equal(revisionTotal(exported.messages.map(({ message }) => message) as unknown as ThreadMessage[]), 2);
});

test("submitting the same edit twice does not add a second revision", async () => {
  const firstWrite = deferred();
  const h = harness(() => firstWrite.promise);
  let exported = thread([{ type: "text", text: "original" }]);
  const edit = () => h.module.updateThreadMessage({
    thread: {
      export: () => exported,
      import: (next) => { exported = next; },
    },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edit A",
    isIncognito: false,
  });
  const first = edit();
  const second = edit();
  firstWrite.resolve();
  await Promise.all([first, second]);

  assert.deepEqual(h.saved.map((record) => (record.metadata as Record<string, unknown>).revision), [1, 1]);
});

test("repeating an edit with THINK whitespace keeps the accepted revision", async () => {
  const firstWrite = deferred();
  const h = harness(() => firstWrite.promise);
  let exported = thread([{ type: "text", text: "original" }]);
  const edit = () => h.module.updateThreadMessage({
    thread: { export: () => exported, import: (next) => { exported = next; } },
    messageId: "a0", remoteId: "remote-1", isIncognito: false,
    newText: "<THINK>\n  reasoning  \n</THINK>\n\nreply",
  });
  const first = edit();
  const second = edit();
  firstWrite.resolve();
  await Promise.all([first, second]);
  assert.deepEqual(h.saved.map((record) => (record.metadata as Record<string, unknown>).revision), [1, 1]);
});

test("reordering distinct calls to the same tool increments the revision", async () => {
  const tool = (toolCallId: string) => ({ type: "tool-call", toolCallId, toolName: "search", args: {}, result: toolCallId });
  const record = await save("<TOOL 2: search>\n\n<TOOL 1: search>", [tool("a"), tool("b")]);
  assert.equal((record.metadata as Record<string, unknown>).revision, 1);
  assert.deepEqual(record.content, [tool("b"), tool("a")]);
});

test("the turn keeps its timestamp when the export carries epoch millis", async () => {
  const h = harness();
  const exported = thread([{ type: "text", text: "hi" }]);
  (exported.messages[1].message as Record<string, unknown>).createdAt = 2000;

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "edited",
    isIncognito: false,
  });

  assert.equal(h.saved[0].createdAt, 2000);
});

const GENERATION_OWNERSHIP = {
  serverManaged: true,
  generationRunId: "run-1",
  generationSeq: 3,
  generationStatus: "completed",
  generationSettled: true,
};

async function saveOwnedReply(custom: Record<string, unknown>) {
  const h = harness();
  const exported = thread([{ type: "text", text: "the original reply" }]);
  exported.messages[1].message.metadata = { custom };

  await h.module.updateThreadMessage({
    thread: { export: () => exported, import: () => {} },
    messageId: "a0",
    remoteId: "remote-1",
    newText: "an edited reply",
    isIncognito: false,
  });

  assert.equal(h.saved.length, 1, "the edit was never written");
  return h.saved[0];
}

test("editing a generated reply drops the run's claim on the turn", async () => {
  const record = await saveOwnedReply({ ...CUSTOM, ...GENERATION_OWNERSHIP });

  assert.deepEqual(withoutRevision(record.metadata), CUSTOM);
  assert.deepEqual(
    Object.keys(GENERATION_OWNERSHIP).filter(
      (key) => key in (record.metadata as Record<string, unknown>),
    ),
    [],
  );
  assert.deepEqual(record.content, [{ type: "text", text: "an edited reply" }]);
});

test("the strip covers the backend's whole server-managed key set", async () => {
  const record = await saveOwnedReply({
    ...CUSTOM,
    serverManaged: true,
    researchRunId: "research-1",
    researchStatus: "completed",
    researchPlanRevision: 2,
    researchRun: { id: "research-1" },
  });

  assert.deepEqual(withoutRevision(record.metadata), CUSTOM);
});

test("a reply whose only metadata was the run's claim writes only its revision", async () => {
  const record = await saveOwnedReply({ ...GENERATION_OWNERSHIP });

  assert.deepEqual(withoutRevision(record.metadata), {});
});

const adapters = provider.statements.find(
  (node): node is ts.FunctionDeclaration =>
    ts.isFunctionDeclaration(node) && node.name?.text === "useStudioRuntimeAdapters",
);
assert.ok(adapters?.body);
const deletionEffect = adapters.body.statements.find((node) => {
  let subscribes = false;
  const visit = (child: ts.Node) => {
    if (ts.isCallExpression(child) && child.expression.getText(provider) === "onChatAttachmentDeleted") {
      subscribes = true;
    }
    ts.forEachChild(child, visit);
  };
  visit(node);
  return subscribes;
});
assert.ok(deletionEffect);
const deletionEffectCode = ts.transpileModule(deletionEffect.getText(provider), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;

function attachmentDeletionHarness(exported: Exported, hash: () => Promise<string>) {
  let listener!: (event: { messageId: string; attachmentId: string }) => Promise<void>;
  let imports = 0;
  new Function("useEffect", "aui", "onChatAttachmentDeleted",
    "chatContentPartAttachmentSignature", "chatContentPartAttachmentIdFromSignature",
    ...Object.keys(savedUsage), deletionEffectCode)(
    (effect: () => unknown) => effect(),
    { thread: () => ({
      export: () => exported,
      import: (next: Exported) => { Object.assign(exported, next); imports += 1; },
    }) },
    (callback: typeof listener) => { listener = callback; return () => {}; },
    (part: { type: string }) => part.type === "image" ? "image-signature" : null,
    hash,
    ...Object.values(savedUsage),
  );
  return {
    remove: (attachmentId: string) => listener({ messageId: "u0", attachmentId }),
    imports: () => imports,
  };
}

for (const kind of ["attachment", "content-hash", "legacy-content"] as const) {
  test(`deleting an upload (${kind}) invalidates saved usage in the loaded branch and after hydration`, async () => {
    const exported = thread([{ type: "text", text: "reply" }]);
    const user = exported.messages[0].message;
    user.metadata = { custom: { revision: 3, keep: "metadata" } };
    const attachmentId = kind === "attachment" ? "file-1"
      : kind === "content-hash" ? "content-part-sha256-test" : "content-part-1";
    if (kind === "attachment") {
      user.attachments = [{ id: attachmentId, content: [{ type: "text", text: "large file" }] }];
    } else {
      (user.content as unknown[]).push({ type: "image", image: "data:image/png;base64,AA==" });
    }
    const records = () => exported.messages.map((item) =>
      RECORD_MODULE.exportedItemToRecord("thread-1", item.parentId, item.message),
    );
    const usage = {
      promptTokens: 4000, completionTokens: 10, totalTokens: 4010, cachedTokens: 0,
      modelId: "model", ...contextUsageMarkers(records()),
    };
    exported.messages[1].message.metadata = { custom: { contextUsage: usage } };
    assert.equal(branchContextUsage(records(), "model", 8192, 8192), usage);
    const h = attachmentDeletionHarness(exported, async () => attachmentId);
    await h.remove(attachmentId);
    assert.equal(h.imports(), 1);
    const saved = records();
    assert.deepEqual(saved[0].metadata, { revision: 4, keep: "metadata" });
    assert.equal(branchContextUsage(saved, "model", 8192, 8192)?.estimated, true);
    const reloaded = [hydrateUser(saved[0]), exported.messages[1].message as unknown as ThreadMessage];
    assert.equal(revisionTotal(reloaded), 4);
    assert.equal(branchContextUsage(threadBranchRecords(reloaded), "model", 8192, 8192)?.estimated, true);
    await h.remove(attachmentId);
    assert.equal(h.imports(), 1);
    assert.equal(records()[0].metadata?.revision, 4);
  });
}

test("an attachment removed during hashing does not revise the message again", async () => {
  const exported = thread([{ type: "text", text: "reply" }]);
  const attachmentId = "content-part-sha256-document";
  const user = exported.messages[0].message;
  user.attachments = [{ id: attachmentId }];
  user.content = [{ type: "image", image: "data:image/png;base64,AA==" }];
  user.metadata = { custom: { revision: 3 } };
  const h = attachmentDeletionHarness(exported, async () => {
    exported.messages[0].message = {
      ...user, attachments: [], metadata: { custom: { revision: 4, keep: "new metadata" } },
    };
    return "content-part-sha256-image";
  });
  await h.remove(attachmentId);
  assert.deepEqual(exported.messages[0].message.metadata, {
    custom: { revision: 4, keep: "new metadata" },
  });
});
