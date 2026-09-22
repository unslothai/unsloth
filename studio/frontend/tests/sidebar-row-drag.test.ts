// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dragging reorders sidebar rows and moves them between lists. One planner decides every drop,
// so these ask it directly, then check the sidebar wires every row and section to it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  dropEdgeAt,
  folderRingKey,
  planKey,
  planSidebarDrop,
  rowKey,
  sectionRingKey,
  SIDEBAR_TAIL_ID,
  STAY,
  type SidebarDragItem,
  type SidebarDropOutcome,
  type SidebarDropPlan,
  type SidebarDropContext,
  type SidebarDropZone,
} from "../src/features/chat/lib/sidebar-drag.ts";
import {
  PINNED_ORDER_SCOPE,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  useSidebarOrganizationStore,
} from "../src/features/chat/stores/sidebar-organization-store.ts";
import { readSrcAsync } from "./helpers/kit.ts";

/** The outcome as a plan, failing the test when the planner answered STAY or null.
 *
 *  `planSidebarDrop` returns `SidebarDropPlan | STAY | null`, so reading `.action` off the
 *  result is a type error until the other two are ruled out. `assert.ok` alone only rules out
 *  null. Asserting it here rather than casting keeps a STAY where a plan was expected a test
 *  failure that names what came back, instead of a cast that would read `undefined` off a
 *  string and fail somewhere less obvious. */
function plannedDrop(
  outcome: SidebarDropOutcome,
  message?: string,
): SidebarDropPlan {
  assert.ok(
    outcome !== null && outcome !== STAY,
    message ?? `expected a drop plan, got ${String(outcome)}`,
  );
  return outcome;
}

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");
const HOOK = await readSrcAsync("features/chat/hooks/use-sidebar-drag.ts");
const EN = await readSrcAsync("i18n/locales/en.ts");

// Two folders, "work" pinned and "home" not; chats c1 and c2 in work, c3 in home, r1 and r2 in
// Recents, and p1 pinned from Recents. Pinned is one list: the folder, then the chat.
function context(
  overrides: Partial<SidebarDropContext> = {},
): SidebarDropContext {
  return {
    organizeBy: "project",
    chatSort: "priority",
    pinnedSort: "manual",
    pinnedChatIds: new Set(["p1"]),
    pinnedProjectIds: new Set(["work"]),
    orders: {
      pinned: ["work", "p1"],
      projects: ["home", "misc"],
      recents: ["r1", "r2"],
      projectChats: (projectId) =>
        ({ work: ["c1", "c2"], home: ["c3"], misc: [] })[projectId] ?? [],
    },
    ...overrides,
  };
}

const chat = (
  id: string,
  section: SidebarDragItem["section"],
  scope: string,
  projectId: string | null,
): SidebarDragItem => ({ kind: "chat", id, section, scope, projectId });
const folder = (
  id: string,
  section: SidebarDragItem["section"],
  scope: string,
): SidebarDragItem => ({
  kind: "project",
  id,
  section,
  scope,
  projectId: null,
});

const chatRow = (
  section: SidebarDropZone["section"],
  scope: string,
  id: string,
  folderId?: string,
  block?: { index: number; count: number },
): SidebarDropZone => ({
  section,
  row: { id, kind: "chat", scope },
  folderId,
  block,
});
const folderRow = (
  section: SidebarDropZone["section"],
  scope: string,
  id: string,
): SidebarDropZone => ({
  section,
  row: { id, kind: "project", scope },
  folderId: id,
});

test("the edge is which half of the row the pointer is over", () => {
  assert.equal(dropEdgeAt({ top: 100, height: 30 }, 110), "top");
  assert.equal(dropEdgeAt({ top: 100, height: 30 }, 115), "bottom");
  assert.equal(dropEdgeAt({ top: 100, height: 30 }, 129), "bottom");
});

test("a chat reorders within its own list, on the edge under the pointer", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("r2", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "top",
      context({ chatSort: "manual" }),
    ),
  );
  assert.deepEqual(plan.action, { kind: "reorder" });
  assert.deepEqual(plan.cue, {
    line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r1"), edge: "top" },
  });
  assert.deepEqual(plan.effects.orders, [
    { scope: RECENTS_ORDER_SCOPE, ids: ["r2", "r1"] },
  ]);
  assert.equal(plan.effects.switchSort, undefined);
});

// A sorted list would undo the drop, so it switches to Manual.
test("a reorder in a sorted list switches it to Manual order", () => {
  const drag = chat("r2", "recents", RECENTS_ORDER_SCOPE, null);
  const zone = chatRow("recents", RECENTS_ORDER_SCOPE, "r1");
  const switching = plannedDrop(planSidebarDrop(drag, zone, "top", context()));
  assert.equal(switching.effects.switchSort, "chats");
  const pinnedDrag = chat("p1", "pinned", PINNED_ORDER_SCOPE, null);
  const pinnedZone = chatRow("pinned", PINNED_ORDER_SCOPE, "p2");
  const pinnedSwitch = plannedDrop(
    planSidebarDrop(
      pinnedDrag,
      pinnedZone,
      "bottom",
      context({
        pinnedSort: "updated",
        pinnedChatIds: new Set(["p1", "p2"]),
        orders: { ...context().orders, pinned: ["work", "p1", "p2"] },
      }),
    ),
  );
  assert.equal(pinnedSwitch.effects.switchSort, "pinned");
  // And a list already on Manual never asks.
  const manual = plannedDrop(
    planSidebarDrop(drag, zone, "top", context({ chatSort: "manual" })),
  );
  assert.equal(manual.action.kind, "reorder");
  assert.equal(manual.effects.switchSort, undefined);
});

// An edge the row is already on is not a move, and a line there would promise one. The spot
// still answers, with STAY, so the section around it does not offer its last slot instead.
test("an edge that moves nothing stays, and still claims the drag", () => {
  const ctx = context({ chatSort: "manual" });
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r2"),
      "top",
      ctx,
    ),
    STAY,
  );
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "bottom",
      ctx,
    ),
    STAY,
  );
  // A pinned row over its own row, or a folder over its own block, is where it already is.
  assert.equal(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE, null),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      ctx,
    ),
    STAY,
  );
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
      "bottom",
      ctx,
    ),
    STAY,
  );
});

test("a chat dropped on a folder, its chats or its empty line is filed there", () => {
  const drag = chat("r1", "recents", RECENTS_ORDER_SCOPE, null);
  for (const zone of [
    folderRow("projects", PROJECT_ORDER_SCOPE, "home"),
    chatRow("projects", projectOrderScope("home"), "c3", "home", {
      index: 0,
      count: 1,
    }),
    { section: "projects", folderId: "misc" } as SidebarDropZone,
    // A pinned folder files it from its chats, or from the lower half of its row.
    chatRow("pinned", projectOrderScope("work"), "c1", "work", {
      index: 0,
      count: 2,
    }),
    { ...folderRow("pinned", PINNED_ORDER_SCOPE, "work"), edge: "bottom" },
  ] as Array<SidebarDropZone & { edge?: "top" | "bottom" }>) {
    const plan = plannedDrop(
      planSidebarDrop(drag, zone, zone.edge ?? "top", context()),
    );
    assert.deepEqual(plan.action, { kind: "move", projectId: zone.folderId });
    // A row to land against gives a slot, whatever the list is sorted by. Only a folder row or
    // an empty line, with nothing to aim at, lights the folder whole.
    assert.deepEqual(
      plan.cue,
      zone.row?.kind === "chat"
        ? {
            line: {
              rowKey: rowKey(zone.row.scope, zone.row.id),
              edge: zone.edge ?? "top",
            },
          }
        : { ring: folderRingKey(zone.folderId!) },
    );
    assert.deepEqual(plan.effects.moveChat, {
      chatId: "r1",
      projectId: zone.folderId,
    });
    assert.equal(plan.effects.unpinChat, undefined);
  }
  // Dropped on the folder it is already in, it stays: no move, and the folder is not lit.
  assert.equal(
    planSidebarDrop(
      chat("c3", "projects", projectOrderScope("home"), "home"),
      folderRow("projects", PROJECT_ORDER_SCOPE, "home"),
      "top",
      context(),
    ),
    STAY,
  );
});

// A folder on Manual order has real slots, so an arriving chat lands in the one under the pointer.
test("a chat filed into a folder on Manual order lands in the slot it dropped on", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("pinned", projectOrderScope("work"), "c2", "work", {
        index: 1,
        count: 2,
      }),
      "top",
      context({ chatSort: "manual" }),
    ),
  );
  assert.deepEqual(plan.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "top" },
  });
  // The slot travels with the snapshot, so a move that lands late can re-aim it.
  assert.deepEqual(plan.effects.orders, [
    {
      scope: projectOrderScope("work"),
      ids: ["c1", "r1", "c2"],
      place: { id: "r1", targetId: "c2", edge: "top" },
    },
  ]);
});

test("a chat dropped on Recents leaves its folder and its pin", () => {
  // Out of a folder.
  const unfiled = plannedDrop(
    planSidebarDrop(
      chat("c3", "projects", projectOrderScope("home"), "home"),
      { section: "recents" },
      "top",
      context(),
    ),
  );
  assert.deepEqual(unfiled.action, { kind: "move", projectId: null });
  assert.deepEqual(unfiled.effects.moveChat, { chatId: "c3", projectId: null });
  // The section's own space lands last, the slot Pinned already gives it, so the line says where
  // the chat is going instead of the whole section saying only that it is going in there.
  assert.deepEqual(unfiled.cue, {
    line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r2"), edge: "bottom" },
  });
  assert.deepEqual(unfiled.effects.orders, [
    {
      scope: RECENTS_ORDER_SCOPE,
      ids: ["r1", "r2", "c3"],
      place: { id: "c3", targetId: "r2", edge: "bottom" },
    },
  ]);
  // And the slot sticks: a sorted list would put the chat back where the sort wants it.
  assert.equal(unfiled.effects.switchSort, "chats");
  // Empty Recents has no row to land against, so the section is the target.
  assert.deepEqual(
    plannedDrop(
      planSidebarDrop(
        chat("c3", "projects", projectOrderScope("home"), "home"),
        { section: "recents" },
        "top",
        context({ orders: { ...context().orders, recents: [] } }),
      ),
    ).cue,
    { ring: sectionRingKey("recents") },
  );
  // Out of Pinned: the pin goes, and so does the folder that would keep it out of Recents.
  const unpinned = plannedDrop(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE, "home"),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "top",
      context(),
    ),
  );
  assert.equal(unpinned.effects.unpinChat, "p1");
  assert.deepEqual(unpinned.effects.moveChat, {
    chatId: "p1",
    projectId: null,
  });
  // With folders off every chat is already a Recents row, so only the pin goes.
  const listMode = plannedDrop(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE, "home"),
      { section: "recents", header: true },
      "top",
      context({ organizeBy: "list" }),
    ),
  );
  assert.deepEqual(listMode.action, { kind: "unpin" });
  assert.equal(listMode.effects.moveChat, undefined);
  // A Recents row dropped on the section itself has no slot to move to.
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      { section: "recents" },
      "top",
      context(),
    ),
    null,
  );
});

test("a chat dropped into Pinned is pinned where it lands", () => {
  const drag = chat("r1", "recents", RECENTS_ORDER_SCOPE, null);
  const onRow = plannedDrop(
    planSidebarDrop(
      drag,
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      context(),
    ),
  );
  assert.deepEqual(onRow.action, { kind: "pin" });
  assert.equal(onRow.effects.pinChat, "r1");
  assert.deepEqual(onRow.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "top" },
  });
  assert.deepEqual(onRow.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "r1", "p1"] },
  ]);
  // Pinned is one list, so the very top is above its first row whatever kind that is: the
  // header, or the upper half of a folder row, both draw the line there.
  for (const zone of [
    {
      section: "pinned",
      header: true,
      row: { id: "work", kind: "project", scope: PINNED_ORDER_SCOPE },
    },
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
  ] as SidebarDropZone[]) {
    const plan = plannedDrop(planSidebarDrop(drag, zone, "top", context()));
    assert.deepEqual(plan.action, { kind: "pin" });
    assert.deepEqual(plan.cue, {
      line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "work"), edge: "top" },
    });
    assert.deepEqual(plan.effects.orders, [
      { scope: PINNED_ORDER_SCOPE, ids: ["r1", "work", "p1"] },
    ]);
  }
  // The space under the rows lands it last, with a line under the last row, never a ring.
  const under = plannedDrop(
    planSidebarDrop(drag, { section: "pinned" }, "bottom", context()),
  );
  assert.deepEqual(under.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "bottom" },
  });
  assert.deepEqual(under.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "p1", "r1"] },
  ]);
  // Pinned sorted by a rule would move the row again, so the drop takes it to Manual.
  assert.equal(
    plannedDrop(
      planSidebarDrop(
        drag,
        { section: "pinned" },
        "bottom",
        context({ pinnedSort: "updated" }),
      ),
    ).effects.switchSort,
    "pinned",
  );
  // A chat already pinned is not pinned again: it reorders, and the last row stays last.
  assert.equal(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE, null),
      { section: "pinned" },
      "bottom",
      context(),
    ),
    STAY,
  );
});

// Dragging a pinned chat onto a folder under Projects unpins it, and files it when the folder
// is another one. Onto a pinned folder it keeps its pin.
test("a pinned chat dragged onto a folder is unpinned, and filed when the folder is new", () => {
  const ctx = context({
    pinnedChatIds: new Set(["p1", "c3"]),
    orders: { ...context().orders, pinned: ["work", "p1", "c3"] },
  });
  const ownFolder = plannedDrop(
    planSidebarDrop(
      chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
      folderRow("projects", PROJECT_ORDER_SCOPE, "home"),
      "top",
      ctx,
    ),
  );
  assert.deepEqual(ownFolder.action, { kind: "unpin" });
  assert.equal(ownFolder.effects.unpinChat, "c3");
  assert.equal(ownFolder.effects.moveChat, undefined);
  const otherFolder = plannedDrop(
    planSidebarDrop(
      chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
      folderRow("projects", PROJECT_ORDER_SCOPE, "misc"),
      "top",
      ctx,
    ),
  );
  assert.deepEqual(otherFolder.action, { kind: "move", projectId: "misc" });
  assert.equal(otherFolder.effects.unpinChat, "c3");
  // In Pinned the folder row's upper half is a slot, so the file lands on its lower half.
  const pinnedFolder = plannedDrop(
    planSidebarDrop(
      chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
      folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(pinnedFolder.action, { kind: "move", projectId: "work" });
  assert.equal(pinnedFolder.effects.unpinChat, undefined);
});

test("a folder reorders among its own list, and the block under it aims at it", () => {
  const ctx = context({
    pinnedProjectIds: new Set(["work", "play"]),
    orders: { ...context().orders, pinned: ["work", "p1", "play"] },
  });
  // Over the other folder's row.
  const onRow = plannedDrop(
    planSidebarDrop(
      folder("play", "pinned", PINNED_ORDER_SCOPE),
      folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
      "top",
      ctx,
    ),
  );
  assert.deepEqual(onRow.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["play", "work", "p1"] },
  ]);
  // Over the chats under it: the top half of the block aims above the folder, the bottom half
  // below it, so the line flips once, in the middle. Below draws under the block's last row.
  const upper = plannedDrop(
    planSidebarDrop(
      folder("play", "pinned", PINNED_ORDER_SCOPE),
      chatRow("pinned", projectOrderScope("work"), "c1", "work", {
        index: 0,
        count: 2,
      }),
      "top",
      ctx,
    ),
  );
  assert.deepEqual(upper.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "work"), edge: "top" },
  });
  const lower = plannedDrop(
    planSidebarDrop(
      folder("play", "pinned", PINNED_ORDER_SCOPE),
      {
        ...chatRow("pinned", projectOrderScope("work"), "c2", "work", {
          index: 1,
          count: 2,
        }),
        blockEnd: { scope: projectOrderScope("work"), id: "c2" },
      },
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(lower.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "bottom" },
  });
  assert.deepEqual(lower.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "play", "p1"] },
  ]);
  // Over a block's own rows, Show more or an empty folder, it lands below that folder.
  const tail = plannedDrop(
    planSidebarDrop(
      folder("play", "pinned", PINNED_ORDER_SCOPE),
      {
        section: "pinned",
        folderId: "work",
        blockEnd: { scope: projectOrderScope("work"), id: "c2" },
      },
      "top",
      ctx,
    ),
  );
  assert.deepEqual(tail.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "bottom" },
  });
  assert.deepEqual(tail.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "play", "p1"] },
  ]);
  // A folder lands between pinned chats too: Pinned is one list.
  const belowChat = plannedDrop(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(belowChat.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["p1", "work", "play"] },
  ]);
  // Its own block keeps it where it is, and Recents is no target at all.
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      chatRow("pinned", projectOrderScope("work"), "c1", "work", {
        index: 0,
        count: 2,
      }),
      "top",
      ctx,
    ),
    STAY,
  );
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "top",
      ctx,
    ),
    null,
  );
});

test("a folder dragged into Pinned is pinned where it lands, and back out is unpinned", () => {
  const pin = plannedDrop(
    planSidebarDrop(
      folder("home", "projects", PROJECT_ORDER_SCOPE),
      folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
      "top",
      context(),
    ),
  );
  assert.deepEqual(pin.action, { kind: "pin" });
  assert.equal(pin.effects.pinProject, "home");
  assert.deepEqual(pin.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["home", "work", "p1"] },
  ]);
  // Over a pinned chat: the slot the line shows, since Pinned is one list.
  const onChat = plannedDrop(
    planSidebarDrop(
      folder("home", "projects", PROJECT_ORDER_SCOPE),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      context(),
    ),
  );
  assert.deepEqual(onChat.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "top" },
  });
  assert.deepEqual(onChat.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "home", "p1"] },
  ]);
  // Pinned sorted by a rule would move the row again, so the drop takes it to Manual.
  assert.equal(
    plannedDrop(
      planSidebarDrop(
        folder("home", "projects", PROJECT_ORDER_SCOPE),
        chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
        "top",
        context({ pinnedSort: "updated" }),
      ),
    ).effects.switchSort,
    "pinned",
  );
  const unpin = plannedDrop(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      folderRow("projects", PROJECT_ORDER_SCOPE, "misc"),
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(unpin.action, { kind: "unpin" });
  assert.equal(unpin.effects.unpinProject, "work");
  assert.deepEqual(unpin.effects.orders, [
    { scope: PROJECT_ORDER_SCOPE, ids: ["home", "misc", "work"] },
  ]);
  const unpinOnHeader = plannedDrop(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      { section: "projects", header: true },
      "top",
      context(),
    ),
  );
  assert.deepEqual(unpinOnHeader.cue, { ring: sectionRingKey("projects") });
});

// Pinning the last project leaves the section drawn but empty, and that is exactly when a folder
// is dragged back into it.
test("an empty Projects section still shows where a folder would land", () => {
  const ctx = context({
    pinnedProjectIds: new Set(["work", "home", "misc"]),
    orders: { ...context().orders, projects: [] },
  });
  const onBody = plannedDrop(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      { section: "projects" },
      "bottom",
      ctx,
    ),
  );
  // No folder to land against, so the whole section is the target.
  assert.deepEqual(onBody.action, { kind: "unpin" });
  assert.deepEqual(onBody.cue, { ring: sectionRingKey("projects") });
  assert.deepEqual(onBody.effects.orders, [
    { scope: PROJECT_ORDER_SCOPE, ids: ["work"] },
  ]);
  // The ring is painted on the section body, which is only there to be painted and hit if the
  // empty section draws a row. A zero-height box is skipped by elementsFromPoint.
  assert.match(
    APP_SIDEBAR,
    /\{visibleProjectRecords\.length === 0 && \(\n\s*<SidebarMenuItem>\n\s*<p className="[^"]*text-nav-fg-muted">\n\s*\{t\("shell\.navigation\.allProjectsPinned"\)\}/,
  );
  assert.match(EN, /allProjectsPinned: "All projects pinned",/);
});

// The gap between a header and its first row is where "above the first row" is aimed, so the
// header stands for that edge for a drag of the same kind, and for the bare section otherwise.
test("a section header stands for the top of its first row", () => {
  const ctx = context({ chatSort: "manual" });
  const aboveFirstChat = plannedDrop(
    planSidebarDrop(
      chat("r2", "recents", RECENTS_ORDER_SCOPE, null),
      {
        section: "recents",
        header: true,
        row: { id: "r1", kind: "chat", scope: RECENTS_ORDER_SCOPE },
      },
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(aboveFirstChat.cue, {
    line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r1"), edge: "top" },
  });
  assert.deepEqual(aboveFirstChat.effects.orders, [
    { scope: RECENTS_ORDER_SCOPE, ids: ["r2", "r1"] },
  ]);
  const aboveFirstFolder = plannedDrop(
    planSidebarDrop(
      folder("misc", "projects", PROJECT_ORDER_SCOPE),
      {
        section: "projects",
        header: true,
        row: { id: "home", kind: "project", scope: PROJECT_ORDER_SCOPE },
      },
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(aboveFirstFolder.cue, {
    line: { rowKey: rowKey(PROJECT_ORDER_SCOPE, "home"), edge: "top" },
  });
  // A chat over the Projects header does not file into the first folder.
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      {
        section: "projects",
        header: true,
        row: { id: "home", kind: "project", scope: PROJECT_ORDER_SCOPE },
      },
      "top",
      ctx,
    ),
    null,
  );
  // Pinned is not conjured up for a drag: it is on screen only when it has rows.
  assert.ok(!APP_SIDEBAR.includes("pinnedTakesDrag"));
  assert.ok(!APP_SIDEBAR.includes("dropToPin"));
  assert.ok(!EN.includes("dropToPin:"));
});

// Equal plans must not re-render the sidebar.
test("a plan's key changes exactly when what it paints or says does", () => {
  const ctx = context();
  const a = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      ctx,
    ),
  );
  const b = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "bottom",
      ctx,
    ),
  );
  assert.equal(planKey(a), planKey({ ...a! }));
  assert.notEqual(planKey(a), planKey(b));
  assert.equal(planKey(null), "");
});

// Every row and section describes itself to the same planner.
test("every row and section is wired to the planner", () => {
  // Chat rows carry their section and folder, and their place in the folder's block.
  assert.match(
    APP_SIDEBAR,
    /\{\.\.\.dnd\.dropZoneProps\(\{\n\s*section: list\.section,\n\s*row: \{ id: item\.id, kind: "chat", scope: list\.scope \},\n\s*folderId: list\.folderId,/,
  );
  // Folder rows are a zone of their folder, and open under a resting pointer while closed.
  assert.match(
    APP_SIDEBAR,
    /row: \{ id: project\.id, kind: "project", scope: order\.scope \},\n\s*folderId: project\.id,\n\s*blockEnd,\n\s*\},\n\s*\{ closed: !expanded \},/,
  );
  // An empty folder's line and its "Show more" row are the folder's block too.
  assert.equal(
    (
      APP_SIDEBAR.match(
        /\{\.\.\.dnd\.dropZoneProps\(\{ section: order\.section, folderId: project\.id, blockEnd \}\)\}/g,
      ) ?? []
    ).length,
    2,
  );
  // Each section's body and its header, which stands for the top of its first row.
  for (const section of ["pinned", "projects", "recents"]) {
    assert.ok(
      APP_SIDEBAR.includes(`{...dnd.dropZoneProps({ section: "${section}" })}`),
      `${section} body is not a zone`,
    );
    assert.match(
      APP_SIDEBAR,
      new RegExp(`section: "${section}",\\n\\s*header: true,\\n\\s*row:`),
      `${section} header is not a zone`,
    );
  }
  // Each list hands its rows the section they are drawn in.
  for (const wiring of [
    // Pinned is one list: its folders and chats share the order a drag rewrites.
    'scope: PINNED_ORDER_SCOPE,\n                            orderedIds: pinnedRowIds,\n                            selectionIds: pinnedProjectRowIds,\n                            section: "pinned",',
    'scope: PINNED_ORDER_SCOPE,\n                            ids: pinnedChatRowIds,\n                            orderIds: pinnedRowIds,\n                            section: "pinned",',
    'scope: RECENTS_ORDER_SCOPE,\n                        ids: recentRowIds,\n                        section: "recents",',
    'orderedIds: projectRowIds,\n                          section: "projects",',
  ]) {
    assert.ok(APP_SIDEBAR.includes(wiring), `missing ${wiring}`);
  }
});

// A dragover can land in the same frame as its dragstart, before state commits.
test("the dragged row is known before React re-renders", async () => {
  const store = await readSrcAsync(
    "features/chat/stores/sidebar-drag-source.ts",
  );
  assert.match(store, /let source: SidebarDragItem \| null = null;/);
  assert.match(HOOK, /setSidebarDragSource\(item\);\n\s*setDrag\(item\);/);
  // Every handler reads it from there; the state is only what paints.
  assert.equal(
    (HOOK.match(/const dragged = sidebarDragSource\(\);/g) ?? []).length,
    2,
  );
  assert.ok(!/const dragged = drag;/.test(HOOK));
});

// The desktop app's webview answers every OS drag itself and never forwards it to the page, so
// `dragstart`, `dragover` and `drop` never fire there and a row wired to them is dead. The whole
// gesture is pointer events, which both the browser and the desktop app deliver.
test("the gesture is pointer events, not the HTML5 drag API", () => {
  for (const source of [HOOK, APP_SIDEBAR]) {
    assert.ok(!/\bdraggable\b\s*[:=]/.test(source));
    assert.ok(!source.includes("onDragStart"));
    assert.ok(!source.includes("onDragOver"));
    assert.ok(!source.includes("onDragEnd"));
    assert.ok(!source.includes("onDragLeave"));
    assert.ok(!source.includes("dataTransfer"));
  }
  assert.match(HOOK, /onPointerDown: \(event: React\.PointerEvent\) => \{/);
  assert.match(HOOK, /window\.addEventListener\("pointermove", onMove\);/);
  assert.match(HOOK, /window\.addEventListener\("pointerup", onUp\);/);
  // A press is only a drag once it travels, or a click on a row would never open the chat.
  assert.match(HOOK, /export const DRAG_THRESHOLD_PX = \d+;/);
  // Touch scrolls the list; its rows keep Move up and Move down.
  assert.match(HOOK, /event\.pointerType === "touch"\) return;/);
});

// A zone is found by hit-testing the pointer, innermost first, so a row still answers before the
// folder block and the section around it. Nothing under the pointer clears the cue outright,
// which no `dragleave` has to be trusted for.
test("a cue over nothing is cleared without trusting dragleave", () => {
  assert.match(HOOK, /for \(const element of document\.elementsFromPoint\(x, y\)\)/);
  // No answer here lets the zone around it answer instead.
  assert.match(HOOK, /if \(outcome\) return \{ hit, outcome \};\n\s*\}\n\s*return null;/);
  assert.match(
    HOOK,
    /if \(!aimed \|\| aimed\.outcome === STAY\) \{\n\s*\/\/[^\n]*\n\s*cancelSpring\(\);\n\s*showPlan\(null\);\n\s*return;\n\s*\}/,
  );
});

// Over its own row a lifted row is already home: nothing is painted and the drop does nothing,
// so the section body never gets to offer its last slot for it.
test("a row over itself claims the drag and paints nothing", () => {
  assert.match(HOOK, /showPlan\(aimed\.outcome\);/);
  assert.match(
    HOOK,
    /if \(dragged && aimed && aimed\.outcome !== STAY\) \{\n\s*optionsRef\.current\.onDrop\(aimed\.outcome, dragged\);/,
  );
});

// A chat's row stays on screen while its move is written, so it can be dropped again before the
// first move lands. Moves of one chat run in order, and only the latest drop commits its slot and
// pin, so an earlier drop cannot land after a later one or park its order in a folder the chat
// never finally joined.
test("a chat dropped again before its move lands keeps only the latest drop", () => {
  assert.match(
    APP_SIDEBAR,
    /const chatMovesRef = useRef\(\n\s*new Map<string, \{ generation: number; chain: Promise<unknown> \}>\(\),\n\s*\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /const generation = \(previous\?\.generation \?\? 0\) \+ 1;/,
  );
  // This drop's work is queued behind the one before it, never alongside.
  assert.match(
    APP_SIDEBAR,
    /const chain = \(previous\?\.chain \?\? Promise\.resolve\(\)\)\n\s*\.then\(\(\) => moveChatToProject\(item, move\.projectId\)\)/,
  );
  assert.match(APP_SIDEBAR, /moves\.set\(item\.id, \{ generation, chain \}\);/);
});

// The menu keeps a 1px gap between rows. A pointer on it would hit the section, whose answer
// is its last slot, so every draggable row's box reaches over the gap below it.
test("rows cover the gap between them, so the section never answers for it", () => {
  assert.match(APP_SIDEBAR, /const DROP_ROW_HIT = "pb-px -mb-px";/);
  assert.match(
    APP_SIDEBAR,
    /: "group\/recent-item relative",\n\s*DROP_ROW_HIT,\n\s*\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /"group\/recent-item relative",\n\s*DROP_ROW_HIT,\n\s*draggingRow\?\.id === project\.id/,
  );
});

// A section's collapsible clips its overflow, so cues stay inside the row.
test("every drop cue is drawn inside its row", () => {
  for (const cue of [
    "${DROP_CUE_BASE} before:top-0",
    "${DROP_CUE_BASE} before:bottom-0",
    "before:inset-x-1 before:inset-y-0 ",
    // A ring is drawn outside its box unless inset, and the first row's box ends at the clip.
    "before:ring-1 before:ring-inset",
  ]) {
    assert.ok(APP_SIDEBAR.includes(cue), `no cue is drawn at ${cue}`);
  }
  assert.ok(
    !/before:-inset-y-px|before:-top-px|before:-bottom-px/.test(APP_SIDEBAR),
    "a cue still hangs outside its row, where a clipped section cuts it off",
  );
});

// Alt + arrow is the keyboard's reorder.
test("alt and an arrow reorder a row without a pointer", async () => {
  assert.match(
    APP_SIDEBAR,
    /onKeyDown: \(event: React\.KeyboardEvent\) => \{\n\s*if \(\n\s*!event\.altKey \|\|\n\s*\(event\.key !== "ArrowUp" && event\.key !== "ArrowDown"\)\n\s*\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /reorderRowBy\(item, orderedIds, sort, event\.key === "ArrowDown" \? 1 : -1\);/,
  );
  // Same rule as a drop: a sorted list switches to Manual.
  assert.match(APP_SIDEBAR, /if \(resorts\) \{\n\s*sort\.set\("manual"\);/);
  // A touch browser starts no drag, so its row menus keep Move up and Move down.
  assert.match(APP_SIDEBAR, /function renderMoveRowItems\(/);
  // A pinned folder moves under Pinned's sort rule, from the keyboard and the touch menu alike:
  // a sorted list switches to Manual, or refuses, exactly as a drop does.
  assert.match(
    APP_SIDEBAR,
    /orderedIds: order\.orderedIds,\n\s*sort: order\.sort,\n\s*\}\)\}/,
  );
  assert.match(APP_SIDEBAR, /order\.orderedIds,\n\s*order\.sort,\n\s*\)\}/);
  assert.match(
    APP_SIDEBAR,
    /selectionIds: pinnedProjectRowIds,\n\s*section: "pinned",\n\s*sort: \{ value: pinnedSort, set: setPinnedSort \},/,
  );
  assert.match(APP_SIDEBAR, /if \(!coarsePointer\) return null;/);
  assert.match(APP_SIDEBAR, /const coarsePointer = useIsCoarsePointer\(\);/);
  // Any touch pointer counts: a laptop with a touchscreen keeps the items too.
  const MOBILE = await readSrcAsync("hooks/use-mobile.ts");
  assert.match(
    MOBILE,
    /const COARSE_POINTER_QUERY = "\(any-pointer: coarse\)";/,
  );
  for (const key of ["moveUp", "moveDown"]) {
    assert.ok(EN.includes(`${key}:`), `${key} is missing from the en locale`);
  }
  // Both paths share the reorder, so both honour the sort rule.
  assert.equal(
    (APP_SIDEBAR.match(/reorderRowBy\(item, orderedIds, sort, /g) ?? []).length,
    3,
  );
});

// The line on the landing edge and the ring around a folder already say where the row goes, so
// no label rides along with the cursor saying it again.
test("nothing follows the cursor while a row is carried", () => {
  assert.ok(!APP_SIDEBAR.includes('data-testid="sidebar-drop-hint"'));
  assert.ok(!APP_SIDEBAR.includes("createPortal"));
  assert.ok(!APP_SIDEBAR.includes("shell.drag."));
  assert.ok(!EN.includes("drag: {"));
  // And the hook has no element to place: it paints the cue and nothing else.
  assert.ok(!HOOK.includes("hintRef"));
});

// The gesture has one behaviour, not three switches: the hint shows, a reorder in a sorted list
// switches it to Manual, and a closed folder opens under a resting pointer. Nothing to persist.
test("drag-and-drop has no preferences of its own", () => {
  // `in`, not an indexed read through a cast: SidebarOrganizationState has no index
  // signature, so `as Record<string, unknown>` is a conversion tsc rejects outright and
  // the test never compiled. `in` needs no cast and asks the stricter question anyway,
  // since a key that came back holding undefined is still back in the store.
  const state: object = useSidebarOrganizationStore.getState();
  for (const key of ["dragHints", "reorderSwitchesSort", "dragOpensFolders"]) {
    assert.ok(!(key in state), `${key} is back in the store`);
    assert.ok(!EN.includes(`${key}:`), `${key} is back in the en locale`);
  }
  assert.ok(!EN.includes("dragDrop:"));
  assert.ok(!APP_SIDEBAR.includes("DRAG_OPTIONS"));
});

// Resting on a closed folder or section opens it.
test("a closed folder or section opens under a resting pointer", () => {
  assert.match(HOOK, /export const SPRING_OPEN_DELAY_MS = \d+;/);
  assert.match(
    APP_SIDEBAR,
    /onSpringOpen: \(zone\) => \{\n\s*if \(zone\.folderId\) \{\n\s*if \(collapsedProjectIds\.has\(zone\.folderId\)\) \{\n\s*toggleProjectCollapsed\(zone\.folderId\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(zone\.section === "pinned"\) setPinnedOpen\(true\);/,
  );
  // A zone carries `closed` in the DOM, and only a closed one arms the timer.
  assert.match(HOOK, /const \{ zone, closed \} = aimed\.hit;/);
  assert.match(
    HOOK,
    /if \(!closed\) \{\n\s*if \(spring\.current\?\.key !== springKey\) cancelSpring\(\);\n\s*return;\n\s*\}/,
  );
  assert.match(HOOK, /zoneOptions\?\.closed \? \{ zone, closed: true \} : \{ zone \}/);
});

// A chat dropped into a folder or Recents is moved, and the move can fail. Its slot in the new
// list and the pin it sheds both wait for the move, or a failed drop would leave the chat
// unpinned, or parked in the manual order of a list it never reached.
test("a drop that moves writes its slot and takes the pin off after the move", () => {
  assert.match(
    APP_SIDEBAR,
    /if \(effects\.unpinChat && !effects\.moveChat && pinnedIdSet\.has\(effects\.unpinChat\)\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /const move = effects\.moveChat;\n\s*if \(!move\) \{\n(?:\s*\/\/[^\n]*\n)*\s*applyOrders\(\);\n\s*return;\n\s*\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /\.then\(\(\) => moveChatToProject\(item, move\.projectId\)\)\n\s*\.then\(\(moved\) => \{\n\s*if \(!moved \|\| moves\.get\(item\.id\)\?\.generation !== generation\) return;\n(?:\s*\/\/[^\n]*\n)*\s*applyOrders\(ordersBefore, sortPicked\);\n\s*if \(unpinAfter\) usePinnedChatsStore\.getState\(\)\.unpin\(unpinAfter\);/,
  );
  // And nothing else in commitDrop writes an order on its own.
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  assert.equal((commit.match(/setManualOrder\(/g) ?? []).length, 1);
  // A slow move can be overtaken by a reorder of the list it lands in. The slot is re-aimed at
  // the list as it stands then, not written from the snapshot taken at the drop.
  assert.match(
    APP_SIDEBAR,
    /const ordersBefore = useSidebarOrganizationStore\.getState\(\)\.manualOrder;\n\s*const moves = chatMovesRef\.current;/,
  );
  assert.match(
    APP_SIDEBAR,
    /before \? landedOrder\(order, before\[order\.scope\]\) : order\.ids,/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(!order\.place \|\| !current \|\| current === before\) return order\.ids;\n\s*return placeIdAt\(current, order\.place\.id, order\.place\.targetId, order\.place\.edge\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /\): Promise<boolean> \{\n\s*if \(item\.projectId === projectId\) return true;/,
  );
});

// applyOrders runs after the move on that path, and it is a closure: a sort read from the render
// it was made in is the drop-time one for good, so a sort the user picked while the move was in
// flight would be overwritten with Manual.
test("a sort picked while a move is in flight is not overwritten", () => {
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  // The change itself is watched. A value read at the drop and again at the end cannot tell a
  // sort picked and picked back from one never touched, and that is a newer intent either way.
  assert.match(
    commit,
    /const stopWatchingSort = useSidebarOrganizationStore\.subscribe\(\n\s*\(now, before\) => \{\n\s*sortPicked \|\|=\n\s*now\.chatSort !== before\.chatSort \|\|\n\s*now\.pinnedSort !== before\.pinnedSort;/,
  );
  // Only the path that waits. Nothing can come between a drop and a switch applied in the turn.
  assert.match(
    commit,
    /if \(!move\) \{\n(?:\s*\/\/[^\n]*\n)*\s*applyOrders\(\);\n\s*return;\n\s*\}/,
  );
  // Read before applyOrders, whose own setChatSort would otherwise trip the watch it reads.
  assert.match(commit, /applyOrders\(ordersBefore, sortPicked\);/);
  // Released on every ending, including a move that failed or was superseded.
  assert.match(commit, /\.finally\(stopWatchingSort\);/);
  // And the switch no longer second-guesses the plan by re-testing the sort's value.
  assert.ok(
    !/chatSort !== "manual"|pinnedSort !== "manual"|=== sortAtDrop\./.test(commit),
    "the switch still tests a sort value",
  );
});

// A folder last in Pinned runs its block to the bottom of the section, so every pixel below its
// title is inside it and a chat aimed past the folder was filed into it. There was no "after".
test("a chat can be dropped after a folder that ends the Pinned list", () => {
  const workScope = projectOrderScope("work");
  const ctx = context({
    pinnedChatIds: new Set(),
    orders: {
      ...context().orders,
      pinned: ["work"],
      projectChats: () => ["w1", "w2"],
    },
  });
  const drag = chat("r1", "recents", RECENTS_ORDER_SCOPE, null);
  // As the folder's chat rows are drawn: the block's last row names itself as its end.
  const lastInBlock: SidebarDropZone = {
    ...chatRow("pinned", workScope, "w2", "work", { index: 1, count: 2 }),
    blockEnd: { scope: workScope, id: "w2" },
  };
  // The lowest row of the block still means "into the folder, last", which is the other fix.
  const intoFolder = plannedDrop(
    planSidebarDrop(drag, lastInBlock, "bottom", ctx),
  );
  assert.deepEqual(intoFolder.action, { kind: "move", projectId: "work" });
  // The section's tail is a row of its own, so "after the folder" has somewhere to aim.
  const afterFolder = plannedDrop(
    planSidebarDrop(
      drag,
      {
        section: "pinned",
        blockEnd: { scope: PINNED_ORDER_SCOPE, id: SIDEBAR_TAIL_ID },
      },
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(afterFolder.action, { kind: "pin" });
  assert.deepEqual(afterFolder.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "r1"] },
  ]);
  // Its own line, not the one under the folder's last chat: same pixels would be two drops.
  assert.deepEqual(afterFolder.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, SIDEBAR_TAIL_ID), edge: "bottom" },
  });
  assert.notDeepEqual(afterFolder.cue, intoFolder.cue);
  // Drawn only while a row is carried, and only when a folder is what ends the list.
  assert.match(
    APP_SIDEBAR,
    /const pinnedEndsInFolder =\n\s*pinnedRows\[pinnedRows\.length - 1\]\?\.kind === "project";/,
  );
  assert.match(
    APP_SIDEBAR,
    /\{draggingRow && pinnedEndsInFolder && \(\n\s*<SidebarMenuItem\n\s*aria-hidden/,
  );
  assert.match(
    APP_SIDEBAR,
    /dropCueClass\(PINNED_ORDER_SCOPE, SIDEBAR_TAIL_ID\),/,
  );
  // A folder over another folder's block still lands below that block, as it always did.
  assert.deepEqual(
    plannedDrop(
      planSidebarDrop(
        folder("home", "projects", PROJECT_ORDER_SCOPE),
        lastInBlock,
        "bottom",
        ctx,
      ),
    ).cue,
    { line: { rowKey: rowKey(workScope, "w2"), edge: "bottom" } },
  );
});

// A pointer resting on the edge of a long list sends no more moves, so scrolling driven off
// pointermove alone took one step and stalled. The frame loop keeps it going, and re-aims: the
// rows slide under a pointer that has not moved, so the cue would otherwise stay on the row that
// left.
test("the edge keeps scrolling while the pointer rests on it", () => {
  assert.match(HOOK, /const onFrame = \(\) => \{[^]*?frame = requestAnimationFrame\(onFrame\);\n\s*if \(edgeScroll\(at\.y\)\) track\(at\.x, at\.y\);/);
  // Started with the drag and cancelled with it, and it stops itself if the drag is gone.
  assert.match(HOOK, /setDrag\(item\);\n\s*frame = requestAnimationFrame\(onFrame\);/);
  assert.match(HOOK, /if \(frame\) cancelAnimationFrame\(frame\);/);
  assert.match(HOOK, /if \(!sidebarDragSource\(\)\) \{\n\s*frame = 0;\n\s*return;\n\s*\}/);
  // track no longer scrolls: one driver, or a move and a frame would both step the list.
  assert.ok(!/const track = useCallback\(\n\s*\(x: number, y: number\) => \{\n\s*(auto|edge)Scroll/.test(HOOK));
});

// The window hears every pointer, not just the one that pressed the row. Without this a finger
// could drive, and drop, a drag the mouse started, which is exactly what touch is kept out of.
test("only the pointer that started the drag drives it", () => {
  for (const handler of [
    /function onMove\(moved: PointerEvent\) \{\n\s*if \(moved\.pointerId !== pointerId \|\| escaped\) return;/,
    /function onUp\(released: PointerEvent\) \{\n\s*if \(released\.pointerId !== pointerId\) return;/,
    /function onCancel\(aborted: PointerEvent\) \{\n\s*if \(aborted\.pointerId !== pointerId\) return;/,
  ]) {
    assert.match(HOOK, handler);
  }
  assert.match(HOOK, /if \(!event\.isPrimary\) return;/);
  // One gesture at a time, and a stale one is abandoned rather than left to block every later
  // drag: a release the window never saw would strand it otherwise.
  assert.match(HOOK, /press\.current\?\.end\(\);/);
  assert.match(HOOK, /end: \(\) => \{\n\s*detach\(\);\n\s*if \(started\) clear\(\);/);
  assert.match(HOOK, /if \(press\.current === self\) press\.current = null;/);
});

// Escape cancels while the button is still down, and the release comes later. A click guard
// armed at the keypress is torn down on the next tick, long before that release, so the row
// under the pointer would open the chat the drag was cancelled out of.
test("escape keeps the click guard until the button is released", () => {
  assert.match(
    HOOK,
    /if \(pressed\.key !== "Escape" \|\| escaped\) return;\n\s*if \(!started\) \{[^]*?detach\(\);\n\s*return;\n\s*\}/,
  );
  // The gesture keeps its listeners: only the release detaches and swallows.
  const onKey = HOOK.slice(
    HOOK.indexOf("function onKey(pressed: KeyboardEvent)"),
    HOOK.indexOf("window.addEventListener(\"pointermove\", onMove);"),
  );
  assert.ok(
    !/escaped = true;[^]*?detach\(\);/.test(onKey),
    "escape must not detach while the button is still down",
  );
  assert.match(onKey, /escaped = true;\n\s*clear\(\);/);
  assert.match(
    HOOK,
    /swallowClick\(\);\n\s*if \(escaped\) return;/,
  );
  // And a cancelled drag does not come back to life on the next move.
  assert.match(HOOK, /if \(moved\.pointerId !== pointerId \|\| escaped\) return;/);
});

// Two ways the bottom of a pinned project was unreachable.
test("a chat can be dropped at the bottom of a pinned project", () => {
  const workScope = projectOrderScope("work");
  const lastRow = chatRow("pinned", workScope, "p2", "work", {
    index: 1,
    count: 2,
  });
  // The default chat sort is Priority, and a chat arriving from another list used to lose its
  // slot to it: the folder lit whole and the chat landed wherever the sort put it. It keeps the
  // slot now and the list switches to Manual, the same as a reorder within one.
  const arriving = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      lastRow,
      "bottom",
      context({ orders: { ...context().orders, projectChats: () => ["p1", "p2"] } }),
    ),
  );
  assert.deepEqual(arriving.cue, {
    line: { rowKey: rowKey(workScope, "p2"), edge: "bottom" },
  });
  assert.equal(arriving.effects.switchSort, "chats");
  assert.deepEqual(arriving.effects.orders[0]?.ids, ["p1", "p2", "r1"]);

  // A folder of more than PROJECT_CHAT_LIMIT chats draws Show more directly under its last
  // visible one, and that row is a drop zone with no row of its own. A chat already in the
  // folder answered nothing there, so there was no way down past the last chat.
  const tail = plannedDrop(
    planSidebarDrop(
      chat("p1", "pinned", workScope, "work"),
      {
        section: "pinned",
        folderId: "work",
        blockEnd: { scope: workScope, id: "p2" },
      },
      "bottom",
      context({
        chatSort: "manual",
        orders: { ...context().orders, projectChats: () => ["p1", "p2", "p3"] },
      }),
    ),
    "the folder's block tail answered nothing",
  );
  assert.deepEqual(tail.action, { kind: "reorder" });
  assert.deepEqual(tail.cue, {
    line: { rowKey: rowKey(workScope, "p2"), edge: "bottom" },
  });
  assert.deepEqual(tail.effects.orders[0]?.ids, ["p2", "p1", "p3"]);

  // The folder's own row is its head, not its tail: the chat is already there.
  assert.equal(
    planSidebarDrop(
      chat("p1", "pinned", workScope, "work"),
      folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
      "bottom",
      context({ chatSort: "manual" }),
    ),
    STAY,
  );
});
