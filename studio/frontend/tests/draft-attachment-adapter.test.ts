// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import ts from "@typescript/typescript6";
import type { AttachmentAdapter, PendingAttachment } from "@assistant-ui/react";
import {
  DraftScopedAttachmentAdapter,
  wasAttachmentAddCancelled,
} from "../src/features/chat/draft-attachment-adapter.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function attachment(file: File, running = false): PendingAttachment {
  return {
    id: "test-attachment",
    type: "document",
    name: file.name,
    contentType: file.type,
    file,
    status: running
      ? { type: "running", reason: "uploading", progress: 0 }
      : { type: "requires-action", reason: "composer-send" },
  };
}

function setup(add: AttachmentAdapter["add"]) {
  let generation = 1;
  const removed: string[] = [];
  const delegate: AttachmentAdapter = {
    accept: ".txt",
    add,
    remove: async (value) => { removed.push(value.id); },
    send: async (value) => ({ ...value, status: { type: "complete" }, content: [] }),
  };
  const guarded = new DraftScopedAttachmentAdapter(delegate, () => {
    const captured = generation;
    return () => generation === captured;
  });
  return { guarded, delegate, removed, switchDraft: () => { generation += 1; } };
}

async function collect(values: AsyncGenerator<PendingAttachment, void>) {
  const found: PendingAttachment[] = [];
  for await (const value of values) found.push(value);
  return found;
}

test("a promise attachment finishes normally in its original draft", async () => {
  const file = new File(["content"], "notes.txt");
  const value = attachment(file);
  const h = setup(async () => value);
  assert.deepEqual(await collect(h.guarded.add({ file })), [value]);
  assert.equal(wasAttachmentAddCancelled(file), false);
  assert.deepEqual(h.removed, []);
  assert.equal(h.guarded.accept, ".txt");
});

test("a late promise result is discarded and its resources removed", async () => {
  const file = new File(["content"], "notes.txt");
  const gate = deferred<PendingAttachment>();
  const h = setup(() => gate.promise);
  const pending = collect(h.guarded.add({ file }));
  h.switchDraft();
  gate.resolve(attachment(file));
  assert.deepEqual(await pending, []);
  assert.equal(wasAttachmentAddCancelled(file), true);
  assert.deepEqual(h.removed, ["test-attachment"]);
});

test("ownership is captured before a generator is first advanced", async () => {
  const file = new File(["content"], "notes.txt");
  let calls = 0;
  const h = setup(async () => { calls += 1; return attachment(file); });
  const generator = h.guarded.add({ file });
  h.switchDraft();
  assert.deepEqual(await collect(generator), []);
  assert.equal(calls, 0);
  assert.equal(wasAttachmentAddCancelled(file), true);
});

test("every streamed result is checked before it can restore a removed chip", async () => {
  const file = new File(["content"], "notes.txt");
  const gate = deferred<void>();
  let closed = false;
  const h = setup(async function* () {
    try {
      yield attachment(file, true);
      await gate.promise;
      yield attachment(file);
    } finally { closed = true; }
  });
  const iterator = h.guarded.add({ file });
  assert.equal((await iterator.next()).value?.status.type, "running");
  const pending = iterator.next();
  h.switchDraft();
  gate.resolve();
  assert.equal((await pending).done, true);
  assert.equal(closed, true);
  assert.equal(wasAttachmentAddCancelled(file), true);
  assert.deepEqual(h.removed, ["test-attachment"]);
});

test("late generator errors finish quietly instead of resurrecting an incomplete chip", async () => {
  const file = new File(["content"], "notes.txt");
  const gate = deferred<void>();
  const h = setup(async function* () {
    yield attachment(file, true);
    await gate.promise;
  });
  const iterator = h.guarded.add({ file });
  await iterator.next();
  const pending = iterator.next();
  h.switchDraft();
  gate.reject(new Error("Old upload failed"));
  assert.equal((await pending).done, true);
  assert.equal(wasAttachmentAddCancelled(file), true);
  assert.deepEqual(h.removed, ["test-attachment"]);
});

test("current-draft errors still reach the normal attachment error handling", async () => {
  const file = new File(["content"], "notes.txt");
  const h = setup(async () => { throw new Error("Unreadable"); });
  await assert.rejects(collect(h.guarded.add({ file })), /Unreadable/);
  assert.equal(wasAttachmentAddCancelled(file), false);
});

test("cancellation survives a cleanup error and does not poison a later add", async () => {
  const file = new File(["content"], "notes.txt");
  const gate = deferred<PendingAttachment>();
  const h = setup(() => gate.promise);
  h.delegate.remove = () => { throw new Error("Already removed"); };
  const pending = collect(h.guarded.add({ file }));
  h.switchDraft();
  gate.resolve(attachment(file));
  assert.deepEqual(await pending, []);
  assert.equal(wasAttachmentAddCancelled(file), true);
  assert.equal((await collect(h.guarded.add({ file }))).length, 1);
  assert.equal(wasAttachmentAddCancelled(file), false);
});

test("send and explicit removal preserve the delegate contract", async () => {
  const file = new File(["content"], "notes.txt");
  const value = attachment(file);
  const h = setup(async () => value);
  assert.equal((await h.guarded.send(value)).status.type, "complete");
  await h.guarded.remove(value);
  assert.deepEqual(h.removed, [value.id]);
});


function runtimeScope() {
  const source = ts.createSourceFile(
    "runtime-provider.tsx",
    readFileSync(new URL("../src/features/chat/runtime-provider.tsx", import.meta.url), "utf8"),
    ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX,
  );
  let capture: ts.Expression | undefined;
  let dbFactory: ts.FunctionDeclaration | undefined;
  let draftHelper: ts.FunctionDeclaration | undefined;
  const visit = (node: ts.Node) => {
    if (ts.isNewExpression(node) && node.expression.getText(source) === "DraftScopedAttachmentAdapter") {
      capture = node.arguments?.[1];
    }
    if (ts.isFunctionDeclaration(node) && node.name?.text === "createStudioDbAdapter") {
      dbFactory = node;
    }
    if (ts.isFunctionDeclaration(node) && node.name?.text === "getAttachmentDraft") {
      draftHelper = node;
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  assert.ok(capture);
  assert.ok(dbFactory);
  const helperCode = draftHelper?.getText(source) ?? "";
  const code = ts.transpileModule(`${helperCode}\nreturn (${capture.getText(source)});`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  let current: { id: string; remoteId?: string } = { id: "local" };
  const switchRef: { current: {
    attempt: number;
    activeNonce: string;
    nonceThread: { nonce: string; threadId: string } | null;
  } } = { current: {
    attempt: 1,
    activeNonce: "original-draft",
    nonceThread: { nonce: "original-draft", threadId: "local" },
  } };
  let deleted = false;
  let tombstoned = false;
  let failInitialization = false;
  const aui = { threadListItem: () => ({ getState: () => {
    if (deleted) throw new Error("Thread removed");
    return current;
  } }) };
  const captureCurrent = new Function("aui", "newThreadSwitchStateRef", "isChatThreadDeleted", code)(
    aui, switchRef, () => tombstoned,
  ) as () => () => boolean;
  const factoryCode = ts.transpileModule(`${helperCode}\n${dbFactory.getText(source)}\nreturn createStudioDbAdapter;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const createAdapter = new Function(
    "readThreadCreationClaim", "useChatRuntimeStore", "trackStoredChatThreadRecord", factoryCode,
  )(
    () => undefined,
    { getState: () => ({ params: {}, adoptDefaultThreadRun: () => {
      if (failInitialization) throw new Error("Initialization failed");
    } }) },
    () => {}, // Persistence is separate; exercise the actual synchronous ownership boundary.
  );
  const db = createAdapter("base", undefined, undefined, true, switchRef) as {
    initialize(id: string): Promise<{ remoteId: string }>;
  };
  return {
    captureCurrent,
    switchDraft() {
      switchRef.current.attempt += 1;
      switchRef.current.activeNonce = `draft-${switchRef.current.attempt}`;
      switchRef.current.nonceThread = null;
    },
    async initialize() {
      const result = await db.initialize(current.id);
      current.remoteId = result.remoteId;
    },
    replaceRuntime(id: string, remoteId?: string) { current = { id, remoteId }; },
    deleteRuntime() { deleted = true; },
    tombstone() { tombstoned = true; },
    failInitialization(value: boolean) { failInitialization = value; },
  };
}

test("a draft saved through Sources retains pending attachments after New chat", async () => {
  const h = runtimeScope();
  const valid = h.captureCurrent();
  assert.equal(valid(), true, "same draft and backgrounded Compare retain ownership");
  await h.initialize();
  h.switchDraft();
  assert.equal(valid(), true, "the saved original retains its pending picker read");
});

test("initialization started before navigation retains the original pending add", async () => {
  const h = runtimeScope();
  const valid = h.captureCurrent();
  const saving = h.initialize();
  h.switchDraft();
  await saving;
  assert.equal(valid(), true);
});

test("an add begun while initialization publishes its saved ID retains ownership", async () => {
  const h = runtimeScope();
  const saving = h.initialize();
  const valid = h.captureCurrent();
  h.switchDraft();
  await saving;
  assert.equal(valid(), true);
});

test("a failed initialization cannot authorize an old add when a later draft saves", async () => {
  const h = runtimeScope();
  const original = h.captureCurrent();
  h.failInitialization(true);
  await assert.rejects(h.initialize(), /Initialization failed/);
  h.switchDraft();
  const later = h.captureCurrent();
  h.failInitialization(false);
  await h.initialize();
  assert.equal(original(), false);
  assert.equal(later(), true);
});

test("adding to a saved background chat cannot replace the unsaved foreground owner", async () => {
  const h = runtimeScope();
  const original = h.captureCurrent();
  h.replaceRuntime("saved", "saved");
  h.captureCurrent();
  h.replaceRuntime("local");
  await h.initialize();
  h.switchDraft();
  assert.equal(original(), true);
});

test("later drafts and saves cannot erase an earlier saved attachment owner", async () => {
  const h = runtimeScope();
  const original = h.captureCurrent();
  await h.initialize();
  for (let index = 0; index < 20; index += 1) {
    h.switchDraft();
    h.replaceRuntime(`later-${index}`);
    const later = h.captureCurrent();
    await h.initialize();
    assert.equal(later(), true);
  }
  h.replaceRuntime("local", "local");
  assert.equal(original(), true);
});

test("a reused blank runtime saved by a later draft cannot claim an abandoned file", async () => {
  const h = runtimeScope();
  const original = h.captureCurrent();
  h.switchDraft();
  const later = h.captureCurrent();
  await h.initialize();
  assert.equal(original(), false, "even a mutated original snapshot cannot acquire ownership");
  assert.equal(later(), true);
});

test("initializing a later draft without a new attachment does not adopt the old add", async () => {
  const h = runtimeScope();
  const original = h.captureCurrent();
  h.switchDraft();
  await h.initialize();
  assert.equal(original(), false);
});

test("a saved runtime keeps background attachments but cannot transfer them to another runtime", () => {
  const h = runtimeScope();
  h.replaceRuntime("saved", "saved");
  const valid = h.captureCurrent();
  h.switchDraft();
  assert.equal(valid(), true);
  h.replaceRuntime("other", "other");
  assert.equal(valid(), false);
});

test("removed and retained-but-tombstoned saved runtimes both cancel pending adds", () => {
  for (const deletion of ["deleteRuntime", "tombstone"] as const) {
    const h = runtimeScope();
    h.replaceRuntime("saved", "saved");
    const valid = h.captureCurrent();
    h[deletion]();
    assert.equal(valid(), false, deletion);
  }
});

test("a cancelled overlapping add cannot remove the valid add's resources", async () => {
  const file = new File(["content"], "notes.txt");
  const oldGate = deferred<PendingAttachment>();
  const newGate = deferred<PendingAttachment>();
  let calls = 0;
  const h = setup(() => (++calls === 1 ? oldGate.promise : newGate.promise));
  const oldAdd = collect(h.guarded.add({ file }));
  h.switchDraft();
  const newAdd = collect(h.guarded.add({ file }));
  oldGate.resolve({ ...attachment(file), id: "old" });
  assert.deepEqual(await oldAdd, []);
  newGate.resolve({ ...attachment(file), id: "new" });
  assert.equal((await newAdd)[0].id, "new");
  assert.deepEqual(h.removed, ["old"]);
});
