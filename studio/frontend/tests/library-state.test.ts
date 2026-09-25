// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Library's state across failures: a chat handoff the composer refuses, a sign-out, an edit
// whose rollback cannot reach the server, a deleted chat attachment, and a download of unknown size.

import assert from "node:assert/strict";
import test from "node:test";
import * as fflate from "fflate";
import * as zustand from "zustand";
import * as zustandMiddleware from "zustand/middleware";

import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// The modules listen for sign-out when they load, so the window has to exist first.
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
  // Another chat's retry leaves them; a failed retry keeps them; a loaded model takes them.
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

test("an upload's grants from before a sign-in are not sent under it", async () => {
  let epoch = 1;
  let requests = 0;
  const { uploadLibraryFiles } = loadWithStubs<{
    uploadLibraryFiles: (batch: object, folderId: string | null) => Promise<string[]>;
  }>(new URL("../src/features/library/api.ts", import.meta.url), {
    "@/features/auth": {
      authFetch: async () => {
        requests += 1;
        return new Response(JSON.stringify({ ids: ["upload:x"] }));
      },
      getAuthSessionEpoch: () => epoch,
      getAuthToken: () => null,
    },
    "@/i18n": { translate: (key: string) => key },
    "@/lib/api-base": {},
    "@/lib/format-fastapi-error": {},
    "./file-name": {},
    "./note-text": {},
  });
  const batch = { nativePathLeases: ["lease"], sessionEpoch: epoch };
  epoch += 1;
  await assert.rejects(uploadLibraryFiles(batch, null), /signedOutBeforeUpload/);
  assert.equal(requests, 0);
  assert.deepEqual(await uploadLibraryFiles({ ...batch, sessionEpoch: epoch }, null), ["upload:x"]);
});

type Item = { id: string; name: string; favorite: boolean; folderId: string | null };

function loadStore(api: Record<string, unknown>, emitted: unknown[] = []) {
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
    "@/features/auth": { AUTH_SESSION_CLEARED_EVENT: SIGNED_OUT, getAuthSessionEpoch: () => 0 },
    "@/features/chat": {
      deleteFineTunedModel: async () => {},
      emitChatAttachmentDeleted: (event: unknown) => emitted.push(event),
    },
    "@/i18n": { translate: (key: string) => key },
    "@/lib/gallery-flags": { notifyGalleryChanged: () => {} },
    "./api": { errorMessage: String, ...api },
    "./favorites-store": {
      useLibraryFavoritesStore: {
        getState: () => ({ begin: () => ({}), adopt: (_: unknown, ids: unknown) => ids, mark: () => () => {} }),
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
  // Renamed twice, both failing: it ends as it was before either.
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
  // The message id comes encoded, as any string can be one.
  await store.getState().removeItem("attachment:m%3A1:content-part-x");
  await store.getState().removeItem("upload:a");
  assert.deepEqual(emitted, [{ messageId: "m:1", attachmentId: "content-part-x" }]);
});

function loadDownloads(fflate: object, saved: (File | Blob)[], fileNames: object) {
  const toast = Object.assign(() => {}, { loading: () => 0, dismiss: () => {}, error: () => {} });
  return loadWithStubs<{
    downloadLibraryItems: (items: { name: string; sizeBytes: number | null }[]) => Promise<void>;
  }>(new URL("../src/features/library/actions.ts", import.meta.url), {
    fflate,
    "@/features/auth": { getAuthSessionEpoch: () => 0 },
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
    "./api": { errorMessage: String, libraryItemFile: async (i: { name: string }) => file(i.name) },
    "./file-kind": {},
    "./file-name": { hasOwnFile: () => true, ...fileNames },
    "./chat-handoff-store": {},
  }).downloadLibraryItems;
}

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
