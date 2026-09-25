// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";
import * as fflate from "fflate";
import * as zustand from "zustand";
import * as zustandMiddleware from "zustand/middleware";

import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { fireWindowEvent } = installLocalStorageFake();
const SIGNED_OUT = "unsloth:auth-session-cleared";
const { uniqueFileNames } = await import("../src/features/library/file-name.ts");
const { attachLibraryChatFiles, useLibraryChatHandoffStore } = await import(
  "../src/features/library/chat-handoff-store.ts"
);

const file = (name: string) => new File(["x"], name);
const names = (files: File[] | undefined) => files?.map((f) => f.name);

test("files the composer refuses wait for a model instead of being lost", async () => {
  const handoff = useLibraryChatHandoffStore.getState();
  handoff.offer("single:1", { files: [file("a.png"), file("notes.md")] });
  const added: string[] = [];
  let modelLoaded = false;
  const add = async (f: File) => {
    if (f.name.endsWith(".png") && !modelLoaded) throw new Error("Load a model first");
    added.push(f.name);
  };
  assert.equal(await attachLibraryChatFiles("single:2", add), 0);
  assert.equal(await attachLibraryChatFiles("single:1", add), 1);
  assert.deepEqual(added, ["notes.md"]);
  assert.deepEqual(names(useLibraryChatHandoffStore.getState().held?.files), ["a.png"]);
  assert.equal(await attachLibraryChatFiles("single:2", add, true), 0);
  assert.equal(await attachLibraryChatFiles("single:1", add, true), 1);
  modelLoaded = true;
  assert.equal(await attachLibraryChatFiles("single:1", add, true), 0);
  assert.deepEqual(added, ["notes.md", "a.png"]);
  assert.equal(useLibraryChatHandoffStore.getState().held, null);
});

test("a sign-out drops files handed to a chat that has not taken them", async () => {
  const handoff = useLibraryChatHandoffStore.getState();
  handoff.offer("single:3", { files: [file("a.png")] });
  useLibraryChatHandoffStore.setState({ held: { targetKey: "single:3", files: [file("b.png")] } });
  fireWindowEvent(SIGNED_OUT, {});
  assert.equal(handoff.take("single:3"), null);
  assert.equal(useLibraryChatHandoffStore.getState().held, null);
});

test("a sign-out mid-handoff stops it, and keeps nothing for the next account", async () => {
  useLibraryChatHandoffStore
    .getState()
    .offer("single:4", { files: [file("a.png"), file("b.png"), file("c.png")] });
  const added: string[] = [];
  const add = async (f: File) => {
    added.push(f.name);
    if (f.name === "a.png") fireWindowEvent(SIGNED_OUT, {});
    throw new Error("Load a model first");
  };
  assert.equal(await attachLibraryChatFiles("single:4", add), 0);
  assert.deepEqual(added, ["a.png"]);
  assert.equal(useLibraryChatHandoffStore.getState().held, null);
});

function loadApi(session: { epoch: number }, authFetch: (url: string, init?: RequestInit) => unknown) {
  return loadWithStubs<{
    uploadLibraryFiles: (batch: object, folderId: string | null) => Promise<string[]>;
    updateLibraryItem: (id: string, patch: { name?: string }) => Promise<void>;
    updateLibraryFolder: (id: string, patch: { name?: string }) => Promise<void>;
    fetchLibraryBlob: (item: object, type: string, maxBytes?: number) => Promise<Blob>;
    LibraryFileTooLarge: new () => Error;
  }>(new URL("../src/features/library/api.ts", import.meta.url), {
    "@/features/auth": {
      authFetch,
      getAuthSessionEpoch: () => session.epoch,
      getAuthToken: () => null,
    },
    "@/i18n": { translate: (key: string) => key },
    "@/lib/api-base": {},
    "@/lib/format-fastapi-error": {},
    "./file-name": {},
    "./note-text": {},
  });
}

test("two quick changes to one folder reach the server in the order they were made", async () => {
  const sent: string[] = [];
  let answerFirst = () => {};
  const { updateLibraryFolder } = loadApi({ epoch: 1 }, (_url, init) => {
    const { name } = JSON.parse(String(init?.body)) as { name: string };
    sent.push(name);
    if (name !== "first") return Promise.resolve(new Response("{}"));
    return new Promise((resolve) => (answerFirst = () => resolve(new Response("{}"))));
  });
  const first = updateLibraryFolder("f1", { name: "first" });
  const second = updateLibraryFolder("f1", { name: "second" });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(sent, ["first"]);
  answerFirst();
  await Promise.all([first, second]);
  assert.deepEqual(sent, ["first", "second"]);
});

test("a preview stops reading at its cap, whatever size the listing said", async () => {
  const bytes = (n: number) =>
    new ReadableStream({
      start(controller) {
        for (let i = 0; i < n; i++) controller.enqueue(new Uint8Array(4));
        controller.close();
      },
    });
  const answer = { body: (): BodyInit => bytes(1), headers: {} as Record<string, string> };
  const api = loadApi({ epoch: 1 }, async () => new Response(answer.body(), { headers: answer.headers }));
  const item = { name: "grew.png", fileUrl: "/f", sizeBytes: 4 };
  assert.equal((await api.fetchLibraryBlob(item, "image/png", 10)).size, 4);
  answer.body = () => bytes(4);
  await assert.rejects(api.fetchLibraryBlob(item, "image/png", 10), api.LibraryFileTooLarge);
  answer.body = () => "x".repeat(40);
  answer.headers = { "content-length": "40" };
  await assert.rejects(api.fetchLibraryBlob(item, "image/png", 10), api.LibraryFileTooLarge);
});

test("an upload's grants from before a sign-in are not sent under it", async () => {
  const session = { epoch: 1 };
  let requests = 0;
  const { uploadLibraryFiles } = loadApi(session, async () => {
    requests += 1;
    return new Response(JSON.stringify({ ids: ["upload:x"] }));
  });
  const batch = { nativePathLeases: ["lease"], sessionEpoch: session.epoch };
  session.epoch += 1;
  await assert.rejects(uploadLibraryFiles(batch, null), /signedOutBeforeUpload/);
  assert.equal(requests, 0);
  const now = { ...batch, sessionEpoch: session.epoch };
  assert.deepEqual(await uploadLibraryFiles(now, null), ["upload:x"]);
});

test("the next account's edit does not wait behind one the account that left never finished", { timeout: 2000 }, async () => {
  const session = { epoch: 1 };
  const sent: string[] = [];
  const { updateLibraryItem } = loadApi(session, (_url, init) => {
    const { name } = JSON.parse(String(init?.body)) as { name: string };
    sent.push(name);
    return name === "A's" ? new Promise(() => {}) : Promise.resolve(new Response("{}"));
  });
  void updateLibraryItem("attachment:m:a", { name: "A's" });
  await new Promise((resolve) => setTimeout(resolve, 0));
  session.epoch += 1;
  await updateLibraryItem("attachment:m:a", { name: "B's" });
  assert.deepEqual(sent, ["A's", "B's"]);
});

type Item = { id: string; name: string; favorite: boolean; folderId: string | null };

function loadStore(
  api: Record<string, unknown>,
  emitted: unknown[] = [],
  session: { epoch: number } = { epoch: 0 },
  starsUndone: string[] = [],
) {
  return loadWithStubs<{
    useLibraryStore: zustand.UseBoundStore<
      zustand.StoreApi<{
        items: Item[];
        refresh: () => Promise<void>;
        removeItem: (id: string) => Promise<void>;
        patchItem: (id: string, patch: Partial<Item>) => Promise<void>;
      }>
    >;
  }>(new URL("../src/features/library/store.ts", import.meta.url), {
    zustand,
    "zustand/middleware": zustandMiddleware,
    "@/features/auth": {
      AUTH_SESSION_CLEARED_EVENT: SIGNED_OUT,
      getAuthSessionEpoch: () => session.epoch,
    },
    "@/features/chat": {
      deleteFineTunedModel: async () => {},
      emitChatAttachmentDeleted: (event: unknown) => emitted.push(event),
    },
    "@/i18n": { translate: (key: string) => key },
    "@/lib/gallery-flags": { notifyGalleryChanged: () => {} },
    "./api": { errorMessage: String, ...api },
    "./favorites-store": {
      useLibraryFavoritesStore: {
        getState: () => ({
          begin: () => ({}),
          adopt: (_: unknown, ids: unknown) => ids,
          mark: (id: string) => ({ settle: () => {}, undo: () => starsUndone.push(id) }),
        }),
      },
    },
    "./object-url-cache": { clearCachedObjectUrls: () => {} },
  }).useLibraryStore;
}

const item = (id: string, name = id): Item => ({ id, name, favorite: false, folderId: null });

test("a failed edit is undone locally when the refresh meant to undo it fails too", async () => {
  let online = true;
  const store = loadStore({
    getLibrary: async () => {
      if (!online) throw new Error("offline");
      return { items: [item("upload:a"), item("upload:b")], folders: [] };
    },
    deleteLibraryItem: async () => {
      throw new Error("offline");
    },
    updateLibraryItem: async () => {
      throw new Error("offline");
    },
  });
  await store.getState().refresh();
  online = false;
  await assert.rejects(store.getState().removeItem("upload:a"));
  await Promise.all([
    assert.rejects(store.getState().patchItem("upload:b", { name: "one" })),
    assert.rejects(store.getState().patchItem("upload:b", { name: "two" })),
  ]);
  assert.deepEqual(store.getState().items, [item("upload:a"), item("upload:b")]);
});

test("undoing a failed edit keeps what a later edit to the same item saved", async () => {
  let online = true;
  const store = loadStore({
    getLibrary: async () => {
      if (!online) throw new Error("offline");
      return { items: [item("upload:a")], folders: [] };
    },
    updateLibraryItem: async (_id: string, patch: Partial<Item>) => {
      if (patch.name) throw new Error("offline");
    },
  });
  await store.getState().refresh();
  online = false;
  await Promise.allSettled([
    store.getState().patchItem("upload:a", { name: "renamed" }),
    store.getState().patchItem("upload:a", { favorite: true }),
  ]);
  assert.deepEqual(store.getState().items, [{ ...item("upload:a"), favorite: true }]);
});

test("an edit still pending at sign-out does not hold back the next account's Library", async () => {
  const store = loadStore({
    getLibrary: async () => ({ items: [item("upload:b")], folders: [] }),
    updateLibraryItem: () => new Promise(() => {}),
  });
  void store.getState().patchItem("upload:a", { name: "never answered" });
  fireWindowEvent(SIGNED_OUT, {});
  await store.getState().refresh();
  assert.deepEqual(store.getState().items, [item("upload:b")]);
});

test("a star that fails offline is undone where Images and Video read it too", async () => {
  const starsUndone: string[] = [];
  let online = true;
  const store = loadStore(
    {
      getLibrary: async () => {
        if (!online) throw new Error("offline");
        return { items: [item("image:a")], folders: [] };
      },
      updateLibraryItem: async () => {
        throw new Error("offline");
      },
    },
    [],
    { epoch: 0 },
    starsUndone,
  );
  await store.getState().refresh();
  online = false;
  await assert.rejects(store.getState().patchItem("image:a", { favorite: true }));
  assert.deepEqual(starsUndone, ["image:a"]);
  assert.equal(store.getState().items[0]!.favorite, false);
});

test("undoing a mirrored star leaves a newer toggle of it alone", () => {
  const { useLibraryFavoritesStore } = loadWithStubs<{
    useLibraryFavoritesStore: zustand.StoreApi<{
      ids: ReadonlySet<string>;
      mark: (id: string, favorite: boolean) => { settle: () => void; undo: () => void };
    }>;
  }>(new URL("../src/features/library/favorites-store.ts", import.meta.url), {
    react: { useEffect: () => {} },
    zustand,
    "@/features/auth": { AUTH_SESSION_CLEARED_EVENT: SIGNED_OUT },
    "@/i18n": { translate: (key: string) => key },
    "@/lib/toast": { toast: {} },
    "./api": {},
  });
  const stars = useLibraryFavoritesStore.getState();
  const failed = stars.mark("image:a", true);
  failed.undo();
  assert.equal(useLibraryFavoritesStore.getState().ids.has("image:a"), false);
  const first = stars.mark("image:b", true);
  stars.mark("image:b", false);
  first.undo();
  assert.equal(useLibraryFavoritesStore.getState().ids.has("image:b"), false);
  stars.mark("image:b", true).settle();
  first.undo();
  assert.equal(useLibraryFavoritesStore.getState().ids.has("image:b"), true);
});

test("a sandbox file is deleted as the file it was listed as", async () => {
  const deleted: unknown[] = [];
  const store = loadStore({
    getLibrary: async () => ({
      items: [{ ...item("sandbox:t:a.png"), fingerprint: "7:1.5" }],
      folders: [],
    }),
    deleteLibraryItem: async (...args: unknown[]) => void deleted.push(args),
  });
  await store.getState().refresh();
  await store.getState().removeItem("sandbox:t:a.png");
  assert.deepEqual(deleted, [["sandbox:t:a.png", "7:1.5"]]);
});

test("deleting a chat attachment tells an open chat to drop it", async () => {
  const emitted: unknown[] = [];
  const store = loadStore(
    {
      getLibrary: async () => ({ items: [item("attachment:m%3A1:content-part-x")], folders: [] }),
      deleteLibraryItem: async () => {},
    },
    emitted,
  );
  await store.getState().refresh();
  await store.getState().removeItem("attachment:m%3A1:content-part-x");
  await store.getState().removeItem("upload:a");
  assert.deepEqual(emitted, [{ messageId: "m:1", attachmentId: "content-part-x" }]);
});

function loadDownloads(
  fflate: object,
  saved: (File | Blob)[],
  fileNames: object,
  session: { epoch: number } = { epoch: 0 },
  fetched: (name: string) => void = () => {},
) {
  const toast = Object.assign(() => {}, { loading: () => 0, dismiss: () => {}, error: () => {} });
  return loadWithStubs<{
    downloadLibraryItems: (items: { name: string; sizeBytes: number | null }[]) => Promise<void>;
  }>(new URL("../src/features/library/actions.ts", import.meta.url), {
    fflate,
    "@/features/auth": { getAuthSessionEpoch: () => session.epoch },
    "@/features/chat": {},
    "@/features/model-picker": {},
    "@/i18n": { translate: (key: string) => key },
    "@/lib/audio-utils": {},
    "@/lib/api-base": { isTauri: false },
    "@/lib/native-files": {
      downloadFile: async (f: File | Blob) => saved.push(f),
      isDownloadCancelled: () => false,
    },
    "@/lib/toast": { toast },
    "@/lib/video-utils": {},
    "./api": {
      errorMessage: String,
      libraryItemFile: async (i: { name: string }) => {
        fetched(i.name);
        return file(i.name);
      },
    },
    "./file-kind": {},
    "./file-name": { hasOwnFile: () => true, ...fileNames },
    "./chat-handoff-store": {},
  }).downloadLibraryItems;
}

test("an attachment deleted as another account signs in is not dropped from its chats", async () => {
  const emitted: unknown[] = [];
  const session = { epoch: 1 };
  const store = loadStore(
    {
      getLibrary: async () => ({ items: [item("attachment:m:a")], folders: [] }),
      deleteLibraryItem: async () => {
        session.epoch += 1;
      },
    },
    emitted,
    session,
  );
  await store.getState().refresh();
  await store.getState().removeItem("attachment:m:a");
  assert.deepEqual(emitted, []);
});

test("a browser download with a file of unknown size goes one by one, never as a zip", async () => {
  const saved: File[] = [];
  const zipSync = () => {
    throw new Error("zipped in memory");
  };
  const downloadLibraryItems = loadDownloads({ zipSync }, saved, {
    uniqueFileNames: (n: string[]) => n,
  });
  await downloadLibraryItems([
    { name: "a.txt", sizeBytes: 10 },
    { name: "clip.webm", sizeBytes: null },
  ]);
  assert.deepEqual(names(saved), ["a.txt", "clip.webm"]);
});

test("a download started before a sign-in saves nothing of the next account's", async () => {
  const saved: File[] = [];
  const session = { epoch: 1 };
  const fetched: string[] = [];
  const signInDuring = (name: string) => {
    fetched.push(name);
    if (name === "a.txt") session.epoch += 1;
  };
  const downloadLibraryItems = loadDownloads({}, saved, { uniqueFileNames }, session, signInDuring);
  await downloadLibraryItems([
    { name: "a.txt", sizeBytes: null },
    { name: "b.txt", sizeBytes: null },
  ]);
  assert.deepEqual(names(saved), []);
  assert.deepEqual(fetched, ["a.txt"]);
});

test("a zip whose last file is read as another account signs in is not saved", async () => {
  const saved: Blob[] = [];
  const session = { epoch: 1 };
  const download = loadDownloads(fflate, saved, { uniqueFileNames }, session);
  const read = File.prototype.arrayBuffer;
  File.prototype.arrayBuffer = function (this: File) {
    session.epoch += 1;
    return read.call(this);
  };
  try {
    await download([
      { name: "a.txt", sizeBytes: 1 },
      { name: "b.txt", sizeBytes: 1 },
    ]);
  } finally {
    File.prototype.arrayBuffer = read;
  }
  assert.deepEqual(saved, []);
});

test("a file named __proto__ is kept in a zipped download", async () => {
  const saved: Blob[] = [];
  const downloadLibraryItems = loadDownloads(fflate, saved, { uniqueFileNames });
  await downloadLibraryItems([
    { name: "__proto__", sizeBytes: 1 },
    { name: "a.txt", sizeBytes: 1 },
  ]);
  const archive = fflate.unzipSync(new Uint8Array(await saved[0]!.arrayBuffer()));
  assert.deepEqual(Object.keys(archive), ["__proto__ (2)", "a.txt"]);
});
