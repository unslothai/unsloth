// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyManualOrder,
  dropEdgeAt,
  folderDropTarget,
  insertIdAt,
  moveIdBy,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  showsInRecents,
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  useSidebarOrganizationStore,
} from "../src/features/chat/stores/sidebar-organization-store.ts";

import { readSrcAsync } from "./helpers/kit.ts";

const ids = (rows: Array<{ id: string }>) => rows.map((row) => row.id);

test("a dragged row lands on the edge the insertion line was drawn on", () => {
  // Above the target, dragging either way.
  assert.deepEqual(insertIdAt(["a", "b", "c", "d"], "d", "b", "top"), [
    "a",
    "d",
    "b",
    "c",
  ]);
  assert.deepEqual(insertIdAt(["a", "b", "c", "d"], "a", "c", "top"), [
    "b",
    "a",
    "c",
    "d",
  ]);
  // And below it, which the index-derived edge could never express on its own.
  assert.deepEqual(insertIdAt(["a", "b", "c", "d"], "d", "b", "bottom"), [
    "a",
    "b",
    "d",
    "c",
  ]);
  assert.deepEqual(insertIdAt(["a", "b", "c", "d"], "a", "c", "bottom"), [
    "b",
    "c",
    "a",
    "d",
  ]);
});

test("a drop that changes nothing leaves the order alone", () => {
  const ids = ["a", "b", "c"];
  // Same row, rows the list no longer holds, and a landing that is where the row already was:
  // each returns the input array itself, which is what tells the caller not to persist it.
  assert.equal(insertIdAt(ids, "b", "b", "top"), ids);
  assert.equal(insertIdAt(ids, "gone", "b", "top"), ids);
  assert.equal(insertIdAt(ids, "b", "gone", "top"), ids);
  assert.equal(insertIdAt(ids, "b", "a", "bottom"), ids);
  assert.equal(insertIdAt(ids, "b", "c", "top"), ids);
});

test("a project chat shows in Recents only when the folders are off", () => {
  // With folders on it would be listed twice, once in each place.
  assert.equal(showsInRecents("p1", "project"), false);
  assert.equal(showsInRecents("p1", "list"), true);
  // A chat in no project is in Recents either way, or it has nowhere to go.
  for (const mode of ["project", "list"] as const) {
    assert.equal(showsInRecents(null, mode), true);
    assert.equal(showsInRecents(undefined, mode), true);
  }
});

test("the drop indicator names the edge the row actually lands on", () => {
  // Painting the cue on the wrong edge is a lie about where the drop goes, so check every pair
  // against what insertIdAt really does with the edge the pointer picked.
  const rows = ["a", "b", "c", "d"];
  for (const dragged of rows) {
    for (const target of rows) {
      if (dragged === target) continue;
      for (const edge of ["top", "bottom"] as const) {
        const next = insertIdAt(rows, dragged, target, edge);
        if (next === rows) continue; // The row was already on that side.
        const landed = next.indexOf(dragged);
        const targetAt = next.indexOf(target);
        assert.equal(
          landed,
          edge === "bottom" ? targetAt + 1 : targetAt - 1,
          `${dragged} onto ${target}'s ${edge} edge landed at ${landed}`,
        );
      }
    }
  }
});

// Pinned draws each folder followed by the chats inside it, so two folder rows can be a whole
// block apart. With only the folder rows answering, the drag reads as doing nothing: most of the
// distance between them refuses the drop, and the near edge of the other folder is the one that
// moves nothing. The chats answer for the folder they belong to instead.
test("a folder dragged over another folder's chats lands against that folder", () => {
  const folderIds = ["a", "b"];
  // Folder b holds a single chat, so that one row has to answer both ends of the block: the
  // lower half of it puts a after b.
  assert.deepEqual(
    folderDropTarget({
      draggedId: "a",
      folderIds,
      folderId: "b",
      rowIndex: 0,
      rowCount: 1,
      pointerEdge: "bottom",
    }),
    { edge: "bottom", next: ["b", "a"] },
  );
  // Its upper half means above b, which is where a already is.
  assert.equal(
    folderDropTarget({
      draggedId: "a",
      folderIds,
      folderId: "b",
      rowIndex: 0,
      rowCount: 1,
      pointerEdge: "top",
    }),
    null,
  );
  // Folder a holds four. The top of the block puts b above a; the bottom would put it back
  // where it already is, which is not a drop at all.
  assert.deepEqual(
    folderDropTarget({
      draggedId: "b",
      folderIds,
      folderId: "a",
      rowIndex: 0,
      rowCount: 4,
      pointerEdge: "top",
    }),
    { edge: "top", next: ["b", "a"] },
  );
  assert.equal(
    folderDropTarget({
      draggedId: "b",
      folderIds,
      folderId: "a",
      rowIndex: 3,
      rowCount: 4,
      pointerEdge: "bottom",
    }),
    null,
  );
  // A folder's own chats are not a target: it cannot land against itself.
  assert.equal(
    folderDropTarget({
      draggedId: "a",
      folderIds,
      folderId: "a",
      rowIndex: 0,
      rowCount: 4,
      pointerEdge: "top",
    }),
    null,
  );
});

// The edge is the row's place in the block, not the pointer's place in the row: reading it off
// the cursor would flip the line from one end of the folder to the other at every chat.
test("the edge holds for every row in the same half of a block", () => {
  const folderIds = ["a", "b", "c"];
  const edgeAt = (rowIndex: number, pointerEdge: "top" | "bottom" = "top") =>
    folderDropTarget({
      draggedId: "c",
      folderIds,
      folderId: "a",
      rowIndex,
      rowCount: 6,
      pointerEdge,
    })?.edge;
  assert.deepEqual([0, 1, 2].map((at) => edgeAt(at)), ["top", "top", "top"]);
  assert.deepEqual(
    [3, 4, 5].map((at) => edgeAt(at)),
    ["bottom", "bottom", "bottom"],
  );
  // The one row the answer changes inside is the middle of the block, not every row.
  assert.equal(edgeAt(2, "bottom"), "bottom");
  assert.equal(edgeAt(3, "top"), "bottom");
});

// The edge follows the cursor, not the two rows' index order: dragging a row down and letting go
// over the top half of a row has to insert above it, or the line moves and the row does not.
test("the pointer's half of a row is the edge it drops on", () => {
  const rect = { top: 100, height: 30 };
  assert.equal(dropEdgeAt(rect, 101), "top");
  assert.equal(dropEdgeAt(rect, 114), "top");
  assert.equal(dropEdgeAt(rect, 115), "bottom");
  assert.equal(dropEdgeAt(rect, 129), "bottom");
});

// Alt + arrow is what a keyboard has instead of a drag, now that the row menu no longer carries
// Move up and Move down.
test("a row moves one slot at a time and stops at the ends", () => {
  assert.deepEqual(moveIdBy(["a", "b", "c"], "a", 1), ["b", "a", "c"]);
  assert.deepEqual(moveIdBy(["a", "b", "c"], "c", -1), ["a", "c", "b"]);
  // Past either end, or a row the list lost, is a no-op.
  const ids = ["a", "b", "c"];
  assert.equal(moveIdBy(ids, "a", -1), ids);
  assert.equal(moveIdBy(ids, "c", 1), ids);
  assert.equal(moveIdBy(ids, "gone", 1), ids);
});

test("moving by keyboard matches dragging onto the neighbour", () => {
  // The two paths must agree, or the same move gives two different orders.
  const rows = ["a", "b", "c", "d"];
  assert.deepEqual(moveIdBy(rows, "b", 1), insertIdAt(rows, "b", "c", "bottom"));
  assert.deepEqual(moveIdBy(rows, "c", -1), insertIdAt(rows, "c", "b", "top"));
});

test("a saved order applies, and undragged rows stay on top in list order", () => {
  const rows = [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "new" }];
  assert.deepEqual(
    ids(applyManualOrder(rows, ["c", "a", "b"], (row) => row.id)),
    // "new" was never dragged, so it keeps the spot the list gave it.
    ["new", "c", "a", "b"],
  );
  // No saved order returns the input untouched, not a copy.
  assert.equal(applyManualOrder(rows, undefined, (row) => row.id), rows);
  assert.equal(applyManualOrder(rows, [], (row) => row.id), rows);
});

test("project folders order independently of any chat list", () => {
  const store = useSidebarOrganizationStore.getState();
  store.setManualOrder(PROJECT_ORDER_SCOPE, ["p2", "p1"]);
  store.setManualOrder(projectOrderScope("p1"), ["chat-b", "chat-a"]);

  const saved = useSidebarOrganizationStore.getState().manualOrder;
  assert.deepEqual(saved[PROJECT_ORDER_SCOPE], ["p2", "p1"]);
  assert.deepEqual(saved["project:p1"], ["chat-b", "chat-a"]);
});

test("each list keeps its own manual order", () => {
  const store = useSidebarOrganizationStore.getState();
  store.setManualOrder(RECENTS_ORDER_SCOPE, ["a", "b"]);
  store.setManualOrder(projectOrderScope("p1"), ["b", "a"]);

  const saved = useSidebarOrganizationStore.getState().manualOrder;
  assert.deepEqual(saved[RECENTS_ORDER_SCOPE], ["a", "b"]);
  assert.deepEqual(saved["project:p1"], ["b", "a"]);
});

test("the sidebar starts grouped by project, sorted by priority", () => {
  // Defaults are what an install without saved preferences renders, so they are
  // part of the layout, not an implementation detail.
  const fresh = useSidebarOrganizationStore.getInitialState();
  assert.equal(fresh.organizeBy, "project");
  assert.equal(fresh.chatSort, "priority");
  // Pinned defaults to manual because pin order already is one: re-sorting the
  // chat lists must not silently rearrange the rows the user pinned by hand.
  assert.equal(fresh.pinnedSort, "manual");
});

test("Pinned sorts independently of the chat lists", () => {
  const store = useSidebarOrganizationStore.getState();
  store.setChatSort("updated");
  store.setPinnedSort("priority");

  const state = useSidebarOrganizationStore.getState();
  assert.equal(state.chatSort, "updated");
  assert.equal(state.pinnedSort, "priority");
});

test("Projects sort on their own, keeping the drag order by default", () => {
  assert.equal(useSidebarOrganizationStore.getInitialState().projectSort, "manual");
  const store = useSidebarOrganizationStore.getState();
  store.setChatSort("priority");
  store.setProjectSort("name");
  const state = useSidebarOrganizationStore.getState();
  assert.equal(state.projectSort, "name");
  assert.equal(state.chatSort, "priority");
  store.setProjectSort("manual");
});

// Reset-all is the only in-app way back to the shipped sidebar layout, and it
// only removes the keys it lists, so an unlisted one survives the reload.
test("Reset all local preferences clears this key", async () => {
  const source = await readSrcAsync("features/settings/tabs/general-tab.tsx");
  const keys = source.slice(
    source.indexOf("const PREFS_KEYS"),
    source.indexOf("];", source.indexOf("const PREFS_KEYS")),
  );
  assert.ok(
    keys.includes("SIDEBAR_ORGANIZATION_STORAGE_KEY") ||
      keys.includes(`"${SIDEBAR_ORGANIZATION_STORAGE_KEY}"`),
    `${SIDEBAR_ORGANIZATION_STORAGE_KEY} missing from PREFS_KEYS`,
  );
});
