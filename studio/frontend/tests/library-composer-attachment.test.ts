// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createLibraryComposerAttachment } from "../src/features/library/composer-attachment.ts";
import type { LibraryItem } from "../src/features/library/api.ts";
import * as fileNames from "../src/features/library/file-name.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const item: LibraryItem = {
  id: "upload:report",
  name: "report.txt",
  source: "uploaded",
  contentType: "text/plain",
  sizeBytes: 15,
  createdAt: 1,
  updatedAt: 1,
  fileUrl: "/api/library/uploads/report/file",
  threadId: null,
  threadTitle: null,
  textOnly: false,
  favorite: false,
  folderId: null,
  openedAt: null,
  model: null,
};

const file = () => new File(["Revenue: €42\n"], "report.txt", { type: "text/plain" });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function composer(read: (entry: LibraryItem) => Promise<File> = async () => file()) {
  const state = { target: "chat:existing" as string | null, session: 1, editing: true };
  const attached: { target: string | null; file: File }[] = [];
  const attach = createLibraryComposerAttachment({
    getTarget: () => state.target,
    getSession: () => state.session,
    isEditing: () => state.editing,
    read,
    add: async (selected) => { attached.push({ target: state.target, file: selected }); },
  });
  return { state, attached, attach };
}

test("a saved file keeps its bytes and attaches to the ongoing conversation", async () => {
  const requested: string[] = [];
  const api = loadWithStubs<typeof import("../src/features/library/api.ts")>(
    new URL("../src/features/library/api.ts", import.meta.url), {
      "@/features/auth": {
        authFetch: async (url: string) => {
          requested.push(url);
          return new Response("Revenue: €42\n");
        },
      },
      "@/i18n": {},
      "@/lib/api-base": {},
      "@/lib/format-fastapi-error": {},
      "./file-name": fileNames,
      "./note-text": {},
    },
  );
  const { state, attached, attach } = composer((entry) => api.libraryItemFile(entry, 20));
  assert.equal(await attach(item), true);
  assert.deepEqual(requested, ["/api/library/uploads/report/file"]);
  assert.equal(state.target, "chat:existing");
  assert.equal(attached.length, 1);
  assert.equal(attached[0].target, "chat:existing");
  assert.equal(attached[0].file.name, "report.txt");
  assert.equal(attached[0].file.type, "text/plain");
  assert.equal(await attached[0].file.text(), "Revenue: €42\n");
});

test("the existing Library read limit refuses a grown file before it reaches the composer", async () => {
  const api = loadWithStubs<typeof import("../src/features/library/api.ts")>(
    new URL("../src/features/library/api.ts", import.meta.url), {
      "@/features/auth": { authFetch: async () => new Response("x".repeat(21)) },
      "@/i18n": { translate: (key: string) => key },
      "@/lib/api-base": {},
      "@/lib/format-fastapi-error": {},
      "./file-name": fileNames,
      "./note-text": {},
    },
  );
  const { attached, attach } = composer((entry) => api.libraryItemFile(entry, 20));
  await assert.rejects(attach(item), api.LibraryFileTooLarge);
  assert.deepEqual(attached, []);
});

test("a chat switch while the Library file loads does not attach it to the next chat", async () => {
  const pending = deferred<File>();
  const { state, attached, attach } = composer(() => pending.promise);
  const attaching = attach(item);
  state.target = "chat:next";
  pending.resolve(file());
  assert.equal(await attaching, false);
  assert.deepEqual(attached, []);
});

test("a sign-out while a Library file loads does not pass it to the next account", async () => {
  const pending = deferred<File>();
  const { state, attached, attach } = composer(() => pending.promise);
  const attaching = attach(item);
  state.session += 1;
  pending.resolve(file());
  assert.equal(await attaching, false);
  assert.deepEqual(attached, []);
});

test("sending the draft while a Library file loads does not append it to a locked composer", async () => {
  const pending = deferred<File>();
  const { state, attached, attach } = composer(() => pending.promise);
  const attaching = attach(item);
  state.editing = false;
  pending.resolve(file());
  assert.equal(await attaching, false);
  assert.deepEqual(attached, []);
});

test("double-clicking a Library entry starts one read and adds one attachment", async () => {
  const pending = deferred<File>();
  let reads = 0;
  const { attached, attach } = composer(() => {
    reads += 1;
    return pending.promise;
  });
  const first = attach(item);
  const second = attach(item);
  pending.resolve(file());
  assert.deepEqual(await Promise.all([first, second]), [true, false]);
  assert.equal(reads, 1);
  assert.equal(attached.length, 1);
});

test("a failed read can be retried without losing the existing conversation", async () => {
  let reads = 0;
  const { state, attached, attach } = composer(async () => {
    if (++reads === 1) throw new Error("File no longer available");
    return file();
  });
  await assert.rejects(attach(item), /File no longer available/);
  assert.equal(await attach(item), true);
  assert.equal(state.target, "chat:existing");
  assert.equal(attached.length, 1);
});

test("the existing attachment adapter can refuse a file and allow a later retry", async () => {
  let supported = false;
  let calls = 0;
  const attach = createLibraryComposerAttachment({
    getTarget: () => "chat:existing",
    getSession: () => 1,
    isEditing: () => true,
    read: async () => file(),
    add: async () => {
      calls += 1;
      if (!supported) throw new Error("Load a model first");
    },
  });
  await assert.rejects(attach(item), /Load a model first/);
  supported = true;
  assert.equal(await attach(item), true);
  assert.equal(calls, 2);
});

test("a file removed from the composer may be selected again in the same chat", async () => {
  const { attached, attach } = composer();
  assert.equal(await attach(item), true);
  attached.splice(0);
  assert.equal(await attach(item), true);
  assert.equal(attached.length, 1);
});
