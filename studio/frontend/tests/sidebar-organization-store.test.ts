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
  // Returning the input array itself tells the caller not to persist.
  assert.equal(insertIdAt(ids, "b", "b", "top"), ids);
  assert.equal(insertIdAt(ids, "gone", "b", "top"), ids);
  assert.equal(insertIdAt(ids, "b", "gone", "top"), ids);
  assert.equal(insertIdAt(ids, "b", "a", "bottom"), ids);
  assert.equal(insertIdAt(ids, "b", "c", "top"), ids);
});

test("a project chat shows in Recents only when the folders are off", () => {
  assert.equal(showsInRecents("p1", "project"), false);
  assert.equal(showsInRecents("p1", "list"), true);
  for (const mode of ["project", "list"] as const) {
    assert.equal(showsInRecents(null, mode), true);
    assert.equal(showsInRecents(undefined, mode), true);
  }
});

test("the drop indicator names the edge the row actually lands on", () => {
  const rows = ["a", "b", "c", "d"];
  for (const dragged of rows) {
    for (const target of rows) {
      if (dragged === target) continue;
      for (const edge of ["top", "bottom"] as const) {
        const next = insertIdAt(rows, dragged, target, edge);
        if (next === rows) continue;
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

// Chats answer for their folder so far-apart folder rows still accept the drop.
test("a folder dragged over another folder's chats lands against that folder", () => {
  const folderIds = ["a", "b"];
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

// The edge is the row's place in the block, not the cursor's place in the row.
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
  assert.equal(edgeAt(2, "bottom"), "bottom");
  assert.equal(edgeAt(3, "top"), "bottom");
});

// The edge follows the cursor, not index order.
test("the pointer's half of a row is the edge it drops on", () => {
  const rect = { top: 100, height: 30 };
  assert.equal(dropEdgeAt(rect, 101), "top");
  assert.equal(dropEdgeAt(rect, 114), "top");
  assert.equal(dropEdgeAt(rect, 115), "bottom");
  assert.equal(dropEdgeAt(rect, 129), "bottom");
});

test("a row moves one slot at a time and stops at the ends", () => {
  assert.deepEqual(moveIdBy(["a", "b", "c"], "a", 1), ["b", "a", "c"]);
  assert.deepEqual(moveIdBy(["a", "b", "c"], "c", -1), ["a", "c", "b"]);
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
    ["new", "c", "a", "b"],
  );
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

test("the sidebar starts grouped by project, sorted by last updated", () => {
  const fresh = useSidebarOrganizationStore.getInitialState();
  assert.equal(fresh.organizeBy, "project");
  assert.equal(fresh.chatSort, "updated");
  // Pinned defaults to manual so re-sorting does not rearrange hand-pinned rows.
  assert.equal(fresh.pinnedSort, "manual");
});

test("Pinned sorts independently of the chat lists", () => {
  const store = useSidebarOrganizationStore.getState();
  store.setChatSort("manual");
  store.setPinnedSort("updated");

  const state = useSidebarOrganizationStore.getState();
  assert.equal(state.chatSort, "manual");
  assert.equal(state.pinnedSort, "updated");
});

test("Projects sort on their own, keeping the drag order by default", () => {
  assert.equal(useSidebarOrganizationStore.getInitialState().projectSort, "manual");
  const store = useSidebarOrganizationStore.getState();
  store.setChatSort("updated");
  store.setProjectSort("name");
  const state = useSidebarOrganizationStore.getState();
  assert.equal(state.projectSort, "name");
  assert.equal(state.chatSort, "updated");
  store.setProjectSort("manual");
});

// Reset-all only removes keys it lists.
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
