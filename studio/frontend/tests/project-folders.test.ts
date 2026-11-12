// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ProjectFolder } from "../src/features/chat/stores/project-folders-store.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  chatFolderMap,
  chatsInFolder,
  childFolders,
  collectSubtreeIds,
  folderDepth,
  isDescendant,
  mergePersistedProjectFolders,
  normalizeFolderName,
  resolveChatFolderId,
  useProjectFoldersStore,
  wouldCreateCycle,
} = await import("../src/features/chat/stores/project-folders-store.ts");

const PROJECT = "proj-1";
const OTHER = "proj-2";

function reset(): void {
  useProjectFoldersStore.setState({
    folders: [],
    folderByChatId: chatFolderMap(),
  });
}

function store() {
  return useProjectFoldersStore.getState();
}

/** Asserts a folder was created and returns its id, avoiding a non-null assertion. */
function made(id: string | null): string {
  assert.ok(id, "expected createFolder to return an id");
  return id;
}

function folder(
  id: string,
  parentId: string | null,
  projectId = PROJECT,
): ProjectFolder {
  return { id, projectId, parentId, name: id };
}

test("createFolder adds a top-level folder and returns its id", () => {
  reset();
  const id = store().createFolder(PROJECT, null, "Research");
  assert.ok(id);
  const folders = store().folders;
  assert.equal(folders.length, 1);
  assert.equal(folders[0].name, "Research");
  assert.equal(folders[0].parentId, null);
  assert.equal(folders[0].projectId, PROJECT);
});

test("createFolder nests under an existing parent in the same project", () => {
  reset();
  const parent = store().createFolder(PROJECT, null, "Parent");
  const child = store().createFolder(PROJECT, parent, "Child");
  assert.ok(child);
  assert.equal(store().folders.find((f) => f.id === child)?.parentId, parent);
});

test("createFolder rejects an empty name and a parent from another project", () => {
  reset();
  assert.equal(store().createFolder(PROJECT, null, "   "), null);
  const elsewhere = store().createFolder(OTHER, null, "Elsewhere");
  assert.ok(elsewhere);
  // Nesting a proj-1 folder under a proj-2 parent must be refused.
  assert.equal(store().createFolder(PROJECT, elsewhere, "Bad"), null);
  // A non-existent parent is refused too.
  assert.equal(store().createFolder(PROJECT, "nope", "Bad"), null);
});

test("normalizeFolderName collapses whitespace, trims, and caps length", () => {
  assert.equal(normalizeFolderName("  a   b  "), "a b");
  assert.equal(normalizeFolderName("x".repeat(80)).length, 60);
});

test("renameFolder updates the name", () => {
  reset();
  const id = made(store().createFolder(PROJECT, null, "Old"));
  store().renameFolder(id, "New");
  assert.equal(store().folders[0].name, "New");
});

test("moveFolder reparents, but rejects self, cycles, and cross-project targets", () => {
  reset();
  const a = made(store().createFolder(PROJECT, null, "A"));
  const b = made(store().createFolder(PROJECT, a, "B"));
  const c = made(store().createFolder(PROJECT, b, "C"));
  const other = made(store().createFolder(OTHER, null, "Other"));

  // Valid: move C up to the root.
  store().moveFolder(c, null);
  assert.equal(store().folders.find((f) => f.id === c)?.parentId, null);

  // Self-parent is rejected.
  store().moveFolder(a, a);
  assert.equal(store().folders.find((f) => f.id === a)?.parentId, null);

  // Moving A into its own descendant B would make a cycle; rejected.
  store().moveFolder(a, b);
  assert.equal(store().folders.find((f) => f.id === a)?.parentId, null);

  // Cross-project reparent is rejected.
  store().moveFolder(a, other);
  assert.equal(store().folders.find((f) => f.id === a)?.parentId, null);
});

test("deleteFolder promotes child folders and chats to the deleted folder's parent", () => {
  reset();
  const a = made(store().createFolder(PROJECT, null, "A"));
  const b = made(store().createFolder(PROJECT, a, "B"));
  const c = made(store().createFolder(PROJECT, b, "C"));
  store().setChatFolder("chat-b", b);

  store().deleteFolder(b); // B's parent is A
  assert.equal(
    store().folders.find((f) => f.id === b),
    undefined,
  );
  // C was under B, now under A.
  assert.equal(store().folders.find((f) => f.id === c)?.parentId, a);
  // chat-b was in B, now in A.
  assert.equal(store().folderByChatId["chat-b"], a);
});

test("deleting a top-level folder returns its chats to the project root", () => {
  reset();
  const a = made(store().createFolder(PROJECT, null, "A"));
  store().setChatFolder("chat-a", a);
  store().deleteFolder(a);
  assert.equal(store().folderByChatId["chat-a"], undefined);
  assert.equal(
    resolveChatFolderId(
      store().folders,
      store().folderByChatId,
      "chat-a",
      PROJECT,
    ),
    null,
  );
});

test("setChatFolder files a chat and setChatsFolder moves many at once", () => {
  reset();
  const a = made(store().createFolder(PROJECT, null, "A"));
  store().setChatFolder("c1", a);
  assert.equal(store().folderByChatId.c1, a);
  store().setChatFolder("c1", null);
  assert.equal(store().folderByChatId.c1, undefined);

  store().setChatsFolder(["c1", "c2", "c3"], a);
  assert.deepEqual(
    [
      store().folderByChatId.c1,
      store().folderByChatId.c2,
      store().folderByChatId.c3,
    ],
    [a, a, a],
  );
  // Filing into a folder that does not exist is a no-op.
  store().setChatsFolder(["c1"], "ghost");
  assert.equal(store().folderByChatId.c1, a);
});

test("resolveChatFolderId falls back to root for dangling or cross-project mappings", () => {
  const folders = [folder("f1", null)];
  const map = chatFolderMap({ good: "f1", dangling: "gone", wrong: "f1" });
  assert.equal(resolveChatFolderId(folders, map, "good", PROJECT), "f1");
  assert.equal(resolveChatFolderId(folders, map, "dangling", PROJECT), null);
  // f1 belongs to PROJECT, so viewed from OTHER it reads as root.
  assert.equal(resolveChatFolderId(folders, map, "wrong", OTHER), null);
});

test("chatsInFolder groups chats by folder, root included", () => {
  const folders = [folder("f1", null), folder("f2", "f1")];
  const map = chatFolderMap({ a: "f1", b: "f2", c: "f1" });
  const chatIds = ["a", "b", "c", "d"];
  assert.deepEqual(chatsInFolder(folders, map, chatIds, PROJECT, "f1"), [
    "a",
    "c",
  ]);
  assert.deepEqual(chatsInFolder(folders, map, chatIds, PROJECT, "f2"), ["b"]);
  assert.deepEqual(chatsInFolder(folders, map, chatIds, PROJECT, null), ["d"]);
});

test("childFolders returns a level in stored order", () => {
  const folders = [folder("a", null), folder("b", null), folder("a1", "a")];
  assert.deepEqual(
    childFolders(folders, PROJECT, null).map((f) => f.id),
    ["a", "b"],
  );
  assert.deepEqual(
    childFolders(folders, PROJECT, "a").map((f) => f.id),
    ["a1"],
  );
});

test("isDescendant, wouldCreateCycle, collectSubtreeIds, folderDepth", () => {
  const folders = [folder("a", null), folder("b", "a"), folder("c", "b")];
  assert.equal(isDescendant(folders, "a", "c"), true);
  assert.equal(isDescendant(folders, "b", "a"), false);
  assert.equal(wouldCreateCycle(folders, "a", "c"), true);
  assert.equal(wouldCreateCycle(folders, "a", null), false);
  assert.equal(wouldCreateCycle(folders, "c", "a"), false);
  assert.deepEqual([...collectSubtreeIds(folders, "a")].sort(), [
    "a",
    "b",
    "c",
  ]);
  assert.equal(folderDepth(folders, "c"), 2);
  assert.equal(folderDepth(folders, "a"), 0);
});

test("mergePersistedProjectFolders validates folders and chat mappings", () => {
  const current = useProjectFoldersStore.getState();
  const merged = mergePersistedProjectFolders(
    {
      folders: [
        { id: "f1", projectId: PROJECT, parentId: null, name: "  Keep  " },
        { id: "f2", projectId: PROJECT, parentId: "f1", name: "Child" },
        { id: "f3", projectId: PROJECT, parentId: "ghost", name: "Orphan" }, // dangling parent -> null
        { id: "", projectId: PROJECT, parentId: null, name: "No id" }, // dropped
        { id: "f4", projectId: "", parentId: null, name: "No project" }, // dropped
        { id: "f5", projectId: PROJECT, parentId: null, name: "   " }, // empty name -> dropped
      ],
      folderByChatId: { a: "f1", b: "f2", gone: "nope" },
    },
    current,
  );
  assert.deepEqual(merged.folders.map((f) => f.id).sort(), ["f1", "f2", "f3"]);
  assert.equal(merged.folders.find((f) => f.id === "f1")?.name, "Keep");
  assert.equal(merged.folders.find((f) => f.id === "f3")?.parentId, null);
  assert.equal(merged.folderByChatId.a, "f1");
  assert.equal(merged.folderByChatId.b, "f2");
  assert.equal(merged.folderByChatId.gone, undefined);
});

test("guards against a pre-corrupted cycle in storage without hanging", () => {
  // Two folders that point at each other: isDescendant must terminate.
  const folders = [folder("x", "y"), folder("y", "x")];
  assert.equal(isDescendant(folders, "x", "y"), true);
  assert.equal(folderDepth(folders, "x") >= 0, true);
});
