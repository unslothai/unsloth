// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyManualOrder,
  dropEdgeFor,
  moveIdBy,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  reorderIds,
  useSidebarOrganizationStore,
} from "../src/features/chat/stores/sidebar-organization-store.ts";

const ids = (rows: Array<{ id: string }>) => rows.map((row) => row.id);

test("a dragged chat takes the target's slot and pushes the rest along", () => {
  assert.deepEqual(reorderIds(["a", "b", "c", "d"], "d", "b"), [
    "a",
    "d",
    "b",
    "c",
  ]);
  assert.deepEqual(reorderIds(["a", "b", "c", "d"], "a", "c"), [
    "b",
    "c",
    "a",
    "d",
  ]);
});

test("a drop that cannot be resolved leaves the order alone", () => {
  const ids = ["a", "b", "c"];
  // Same row, and rows the list no longer holds: each returns the input array
  // itself, which is what tells the caller there is nothing to persist.
  assert.equal(reorderIds(ids, "b", "b"), ids);
  assert.equal(reorderIds(ids, "gone", "b"), ids);
  assert.equal(reorderIds(ids, "b", "gone"), ids);
});

test("the drop indicator names the edge the row actually lands on", () => {
  // Painting the cue on the wrong edge is a lie about where the drop goes, so
  // check every pair against what reorderIds really does.
  const rows = ["a", "b", "c", "d"];
  for (const dragged of rows) {
    for (const target of rows) {
      if (dragged === target) continue;
      const next = reorderIds(rows, dragged, target);
      const landed = next.indexOf(dragged);
      const targetAt = next.indexOf(target);
      const edge = dropEdgeFor(rows, dragged, target);
      assert.equal(
        landed,
        edge === "bottom" ? targetAt + 1 : targetAt - 1,
        `${dragged} onto ${target} drew ${edge} but landed at ${landed}`,
      );
    }
  }
});

test("a row moves one slot at a time and stops at the ends", () => {
  assert.deepEqual(moveIdBy(["a", "b", "c"], "a", 1), ["b", "a", "c"]);
  assert.deepEqual(moveIdBy(["a", "b", "c"], "c", -1), ["a", "c", "b"]);
  // Past either end, or a row the list lost, is a no-op.
  const ids = ["a", "b", "c"];
  assert.equal(moveIdBy(ids, "a", -1), ids);
  assert.equal(moveIdBy(ids, "c", 1), ids);
  assert.equal(moveIdBy(ids, "gone", 1), ids);
});

test("moving by menu matches dragging onto the neighbour", () => {
  // The two paths must agree, or the same gesture gives two different orders.
  const rows = ["a", "b", "c", "d"];
  assert.deepEqual(moveIdBy(rows, "b", 1), reorderIds(rows, "b", "c"));
  assert.deepEqual(moveIdBy(rows, "c", -1), reorderIds(rows, "c", "b"));
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
