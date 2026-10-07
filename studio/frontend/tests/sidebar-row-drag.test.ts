// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  dropEdgeAt,
  equivalentDrop,
  folderRingKey,
  litRingKey,
  planKey,
  planSidebarDrop,
  rowKey,
  sectionRingKey,
  SIDEBAR_TAIL_SCOPE,
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

/** Narrows planSidebarDrop's result to a plan, failing with what came back otherwise. */
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
const CSS = await readSrcAsync("index.css");

// Two folders, "work" pinned and "home" not; c1,c2 in work, c3 in home, r1,r2 in Recents,
// p1 pinned from Recents. Pinned is one list: the folder, then the chat.
function context(
  overrides: Partial<SidebarDropContext> = {},
): SidebarDropContext {
  return {
    organizeBy: "project",
    chatSort: "updated",
    pinnedSort: "manual",
    projectSort: "manual",
    pinnedChatIds: new Set(["p1"]),
    pinnedProjectIds: new Set(["work"]),
    sectionByChatId: {},
    sectionByProjectId: {},
    sectionByPageId: {},
    sectionSort: () => "manual",
    orders: {
      pinned: ["work", "p1"],
      projects: ["home", "misc"],
      recents: ["r1", "r2"],
      projectChats: (projectId) =>
        ({ work: ["c1", "c2"], home: ["c3"], misc: [] })[projectId] ?? [],
      sections: () => [],
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
  const manual = plannedDrop(
    planSidebarDrop(drag, zone, "top", context({ chatSort: "manual" })),
  );
  assert.equal(manual.action.kind, "reorder");
  assert.equal(manual.effects.switchSort, undefined);
});

test("a folder reorder or unpin switches a sorted Projects list to Manual", () => {
  assert.ok(APP_SIDEBAR.includes("sort: { value: projectSort, set: setProjectSort }"));
  const home = folder("home", "projects", PROJECT_ORDER_SCOPE);
  const miscRow = folderRow("projects", PROJECT_ORDER_SCOPE, "misc");
  const reorder = plannedDrop(
    planSidebarDrop(home, miscRow, "bottom", context({ projectSort: "name" })),
  );
  assert.equal(reorder.action.kind, "reorder");
  assert.equal(reorder.effects.switchSort, "projects");
  const unpin = plannedDrop(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      miscRow,
      "bottom",
      context({ projectSort: "created" }),
    ),
  );
  assert.equal(unpin.action.kind, "unpin");
  assert.equal(unpin.effects.switchSort, "projects");
  const manual = plannedDrop(planSidebarDrop(home, miscRow, "bottom", context()));
  assert.equal(manual.effects.switchSort, undefined);
});

test("a pinned chat is listed once and drops back into its folder unpinned", () => {
  assert.ok(
    APP_SIDEBAR.includes(
      "(item) => !pinnedIdSet.has(item.id) && !sectionByChatId[item.id],",
    ),
    "a folder still lists its pinned chats",
  );
  const ctx = context({
    pinnedChatIds: new Set(["p1", "c1"]),
    orders: {
      ...context().orders,
      pinned: ["work", "c1", "p1"],
      projectChats: (projectId) =>
        ({ work: ["c2"], home: ["c3"], misc: [] })[projectId] ?? [],
    },
  });
  const drag = chat("c1", "pinned", PINNED_ORDER_SCOPE, "work");
  for (const zone of [
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
    chatRow("pinned", projectOrderScope("work"), "c2", "work"),
  ]) {
    const plan = plannedDrop(planSidebarDrop(drag, zone, "bottom", ctx));
    assert.equal(plan.action.kind, "unpin");
    assert.equal(plan.effects.unpinChat, "c1");
    assert.equal(plan.effects.moveChat, undefined);
  }
});

// An edge the row is already on returns STAY, which still claims the drag.
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
    assert.deepEqual(
      plan.cue,
      zone.row?.kind === "chat"
        ? {
            line: {
              rowKey: rowKey(zone.row.scope, zone.row.id),
              edge: zone.edge ?? "top",
              folderId: zone.folderId,
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
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "top", folderId: "work" },
  });
  assert.deepEqual(plan.effects.orders, [
    {
      scope: projectOrderScope("work"),
      ids: ["c1", "r1", "c2"],
      place: { id: "r1", targetId: "c2", edge: "top" },
    },
  ]);
});

test("a chat dropped on Recents leaves its folder and its pin", () => {
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
  assert.equal(unfiled.effects.switchSort, "chats");
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
  const under = plannedDrop(
    planSidebarDrop(drag, { section: "pinned" }, "bottom", context()),
  );
  assert.deepEqual(under.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "bottom" },
  });
  assert.deepEqual(under.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "p1", "r1"] },
  ]);
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
  assert.deepEqual(onBody.action, { kind: "unpin" });
  assert.deepEqual(onBody.cue, { ring: sectionRingKey("projects") });
  assert.deepEqual(onBody.effects.orders, [
    { scope: PROJECT_ORDER_SCOPE, ids: ["work"] },
  ]);
  // Ring is painted on the body; a zero-height box is skipped by elementsFromPoint.
  assert.match(
    APP_SIDEBAR,
    /\{visibleProjectRecords\.length === 0 && \(\n\s*<SidebarMenuItem>\n\s*<p className="[^"]*text-nav-fg-muted">\n\s*\{t\(\n\s*projects\.length === 0\n\s*\? "shell\.navigation\.noProjects"\n\s*: projects\.some\(\(project\) => sectionByProjectId\[project\.id\]\)\n\s*\? "shell\.navigation\.allProjectsFiled"\n\s*: "shell\.navigation\.allProjectsPinned",/,
  );
  assert.match(EN, /noProjects: "No projects",/);
  assert.match(EN, /allProjectsPinned: "All projects pinned",/);
});

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

test("every row and section is wired to the planner", () => {
  assert.match(
    APP_SIDEBAR,
    /\{\.\.\.dnd\.dropZoneProps\(\{\n\s*section: list\.section,\n\s*row: \{ id: item\.id, kind: "chat", scope: list\.scope \},\n\s*folderId: list\.folderId,/,
  );
  assert.match(
    APP_SIDEBAR,
    /row: \{ id: project\.id, kind: "project", scope: order\.scope \},\n\s*folderId: project\.id,\n\s*blockEnd,\n\s*\},\n\s*\{ closed: !expanded \},/,
  );
  assert.equal(
    (
      APP_SIDEBAR.match(
        /\{\.\.\.dnd\.dropZoneProps\(\{ section: order\.section, folderId: project\.id, blockEnd \}\)\}/g,
      ) ?? []
    ).length,
    2,
  );
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
  for (const wiring of [
    'scope: PINNED_ORDER_SCOPE,\n                            orderedIds: pinnedRowIds,\n                            selectionIds: pinnedProjectRowIds,\n                            section: "pinned",',
    'scope: PINNED_ORDER_SCOPE,\n                            ids: pinnedChatRowIds,\n                            orderIds: pinnedRowIds,\n                            section: "pinned",',
    'scope: RECENTS_ORDER_SCOPE,\n                        ids: recentRowIds,\n                        section: "recents",',
    'orderedIds: projectRowIds,\n                          section: "projects",',
  ]) {
    // Whitespace-free, since the sections are drawn by functions at their own indent.
    const flat = (text: string) => text.replace(/\s+/g, "");
    assert.ok(flat(APP_SIDEBAR).includes(flat(wiring)), `missing ${wiring}`);
  }
});

// A dragover can land in the same frame as its dragstart, before state commits.
test("the dragged row is known before React re-renders", async () => {
  const store = await readSrcAsync(
    "features/chat/stores/sidebar-drag-source.ts",
  );
  assert.match(store, /let source: SidebarDragItem \| null = null;/);
  assert.match(HOOK, /setSidebarDragSource\(item\);\n\s*setDrag\(item\);/);
  assert.equal(
    (HOOK.match(/const dragged = sidebarDragSource\(\);/g) ?? []).length,
    2,
  );
  assert.ok(!/const dragged = drag;/.test(HOOK));
});

// The desktop webview never forwards HTML5 drag events, so use pointer events.
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
  // A press is only a drag once it travels, or a row click would never open the chat.
  assert.match(HOOK, /export const DRAG_THRESHOLD_PX = \d+;/);
  // Touch scrolls the list rather than lifting a row.
  assert.match(HOOK, /event\.pointerType === "touch"\) return;/);
});

// Zones are hit-tested innermost first; no dragleave is trusted.
test("a cue over nothing is cleared without trusting dragleave", () => {
  assert.match(HOOK, /for \(const element of document\.elementsFromPoint\(x, y\)\)/);
  assert.match(HOOK, /if \(!outcome\) continue;/);
  assert.match(HOOK, /return \{ hit, outcome \};\n\s*\}\n\s*return null;/);
  assert.match(
    HOOK,
    /if \(!aimed \|\| aimed\.outcome === STAY\) \{\n\s*\/\/[^\n]*\n\s*cancelSpring\(\);\n\s*showPlan\(null\);\n\s*return;\n\s*\}/,
  );
});

test("a row over itself claims the drag and paints nothing", () => {
  assert.match(HOOK, /showPlan\(aimed\.outcome\);/);
  assert.match(
    HOOK,
    /if \(dragged && aimed && aimed\.outcome !== STAY\) \{\n\s*optionsRef\.current\.onDrop\(aimed\.outcome, dragged\);/,
  );
});

// Moves of one chat run in order and only the latest drop commits its slot and pin.
test("a chat dropped again before its move lands keeps only the latest drop", () => {
  assert.match(
    APP_SIDEBAR,
    /const chatMovesRef = useRef\(\n\s*new Map<string, \{ generation: number; chain: Promise<unknown> \}>\(\),\n\s*\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /const generation = \(previous\?\.generation \?\? 0\) \+ 1;/,
  );
  assert.match(
    APP_SIDEBAR,
    /const chain = \(previous\?\.chain \?\? Promise\.resolve\(\)\)\n\s*\.then\(\(\) => moveChatToProject\(item, move\.projectId\)\)/,
  );
  assert.match(APP_SIDEBAR, /moves\.set\(item\.id, \{ generation, chain \}\);/);
});

// Rows reach over the 1px gap so the section never answers for it.
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

// A section's collapsible clips overflow, so cues stay inside the row.
test("every drop cue is drawn inside its row", () => {
  for (const cue of [
    "${DROP_CUE_BASE} before:top-0",
    // A pixel up: the last row's extra hit pixel is clipped by the list.
    "${DROP_CUE_BASE} before:bottom-px",
    "before:inset-x-1 before:inset-y-0 ",
    // Borders snap to whole device pixels, so every cue keeps one thickness.
    "before:h-0 before:border-t-[1.5px] before:border-primary",
    "before:border-[1.5px] before:border-primary",
  ]) {
    assert.ok(APP_SIDEBAR.includes(cue), `no cue is drawn at ${cue}`);
  }
  assert.ok(
    !/before:-inset-y-px|before:-top-px|before:-bottom-px/.test(APP_SIDEBAR),
    "a cue still hangs outside its row, where a clipped section cuts it off",
  );
});

test("alt and an arrow reorder a row without a pointer", () => {
  assert.match(
    APP_SIDEBAR,
    /onKeyDown: \(event: React\.KeyboardEvent\) => \{\n\s*if \(\n\s*!event\.altKey \|\|\n\s*\(event\.key !== "ArrowUp" && event\.key !== "ArrowDown"\)\n\s*\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /reorderRowBy\(item, orderedIds, sort, event\.key === "ArrowDown" \? 1 : -1\);/,
  );
  assert.match(APP_SIDEBAR, /if \(resorts\) \{\n\s*sort\.set\("manual"\);/);
  assert.ok(!APP_SIDEBAR.includes("renderMoveRowItems"));
  assert.ok(!EN.includes("moveUp:") && !EN.includes("moveDown:"));
  assert.match(
    APP_SIDEBAR,
    /orderedIds: order\.orderedIds,\n\s*sort: order\.sort,\n\s*\}\)\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /selectionIds: pinnedProjectRowIds,\n\s*section: "pinned",\n\s*sort: \{ value: pinnedSort, set: setPinnedSort \},/,
  );
});

test("nothing follows the cursor while a row is carried", () => {
  assert.ok(!APP_SIDEBAR.includes('data-testid="sidebar-drop-hint"'));
  assert.ok(!APP_SIDEBAR.includes("createPortal"));
  assert.ok(!APP_SIDEBAR.includes("shell.drag."));
  assert.ok(!EN.includes("drag: {"));
  assert.ok(!HOOK.includes("hintRef"));
});

test("drag-and-drop has no preferences of its own", () => {
  // `in`, not a cast: the state type has no index signature and the cast fails tsc.
  const state: object = useSidebarOrganizationStore.getState();
  for (const key of ["dragHints", "reorderSwitchesSort", "dragOpensFolders"]) {
    assert.ok(!(key in state), `${key} is back in the store`);
    assert.ok(!EN.includes(`${key}:`), `${key} is back in the en locale`);
  }
  assert.ok(!EN.includes("dragDrop:"));
  assert.ok(!APP_SIDEBAR.includes("DRAG_OPTIONS"));
});

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
  assert.match(HOOK, /const \{ zone, closed \} = aimed\.hit;/);
  assert.match(
    HOOK,
    /if \(!closed\) \{\n\s*if \(spring\.current\?\.key !== springKey\) cancelSpring\(\);\n\s*return;\n\s*\}/,
  );
  assert.match(HOOK, /zoneOptions\?\.closed \? \{ zone, closed: true \} : \{ zone \}/);
  // track runs every frame, so re-arming on each would reset the delay forever.
  assert.match(
    HOOK,
    /if \(spring\.current\?\.key === springKey\) return;\n\s*cancelSpring\(\);\n\s*spring\.current = \{/,
  );
});

// Slot and unpin wait for the move, or a failed drop leaves the chat half-moved.
test("a drop that moves writes its slot and takes the pin off after the move", () => {
  assert.match(
    APP_SIDEBAR,
    /if \(effects\.unpinChat && !effects\.moveChat && pinnedIdSet\.has\(effects\.unpinChat\)\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /const move = effects\.moveChat;\n\s*if \(!move\) \{\n(?:\s*\/\/[^\n]*\n)*\s*applyFiling\(\);\n\s*applyOrders\(\);\n\s*return;\n\s*\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /\.then\(\(\) => moveChatToProject\(item, move\.projectId\)\)\n\s*\.then\(\(moved\) => \{\n\s*if \(!moved \|\| moves\.get\(item\.id\)\?\.generation !== generation\) return;\n(?:\s*\/\/[^\n]*\n)*\s*if \(!filedSince\) applyFiling\(\);\n\s*applyOrders\(ordersBefore, sortPicked\);\n\s*if \(unpinAfter\) usePinnedChatsStore\.getState\(\)\.unpin\(unpinAfter\);/,
  );
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  assert.equal((commit.match(/setManualOrder\(/g) ?? []).length, 1);
  // A slow move re-aims its slot at the list as it stands then, not the drop snapshot.
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

// applyOrders is a closure, so a sort read at render would overwrite one picked mid-move.
test("a sort picked while a move is in flight is not overwritten", () => {
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  // Watch the change itself; a value compare cannot see a sort picked and picked back.
  assert.match(
    commit,
    /const stopWatchingSort = useSidebarOrganizationStore\.subscribe\(\(now, before\) => \{\n\s*if \(switching\) \{\n\s*sortPicked \|\|=\n\s*switchedListSort\(now, switching\) !== switchedListSort\(before, switching\);\n\s*\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /if \(list === "pinned"\) return state\.pinnedSort;\n\s*if \(list === "projects"\) return state\.projectSort;\n\s*const sectionId = customSectionIdOf\(list\);/,
  );
  assert.match(commit, /const switching = effects\.switchSort;/);
  assert.match(
    commit,
    /if \(!move\) \{\n(?:\s*\/\/[^\n]*\n)*\s*applyFiling\(\);\n\s*applyOrders\(\);\n\s*return;\n\s*\}/,
  );
  // Read before applyOrders, whose own setChatSort would otherwise trip the watch.
  assert.match(commit, /applyOrders\(ordersBefore, sortPicked\);/);
  // Released on every ending, including a failed or superseded move.
  assert.match(commit, /\.finally\(stopWatchingSort\);/);
  assert.ok(
    !/chatSort !== "manual"|pinnedSort !== "manual"|=== sortAtDrop\./.test(commit),
    "the switch still tests a sort value",
  );
});

// A last Pinned folder's block reached the section bottom, leaving no "after" target.
test("the Pinned tail strip adds no space under the section", () => {
  assert.match(
    APP_SIDEBAR,
    /<SidebarGroup className="[^"]*data-\[state=open\]:-mb-\[calc\(8px\*var\(--ui-space-scale,1\)\)\]"/,
  );
});

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
  const lastInBlock: SidebarDropZone = {
    ...chatRow("pinned", workScope, "w2", "work", { index: 1, count: 2 }),
    blockEnd: { scope: workScope, id: "w2" },
  };
  const intoFolder = plannedDrop(
    planSidebarDrop(drag, lastInBlock, "bottom", ctx),
  );
  assert.deepEqual(intoFolder.action, { kind: "move", projectId: "work" });
  const afterFolder = plannedDrop(
    planSidebarDrop(
      drag,
      {
        section: "pinned",
        blockEnd: { scope: SIDEBAR_TAIL_SCOPE, id: "pinned" },
      },
      "bottom",
      ctx,
    ),
  );
  assert.deepEqual(afterFolder.action, { kind: "pin" });
  assert.deepEqual(afterFolder.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "r1"] },
  ]);
  assert.deepEqual(afterFolder.cue, {
    line: { rowKey: rowKey(SIDEBAR_TAIL_SCOPE, "pinned"), edge: "bottom" },
  });
  assert.notDeepEqual(afterFolder.cue, intoFolder.cue);
  // Row ids are arbitrary strings; scopes are ours, so no row key can collide with the tail.
  const tailKey = rowKey(SIDEBAR_TAIL_SCOPE, "pinned");
  for (const hostile of [
    "pinned",
    "sidebar-tail",
    SIDEBAR_TAIL_SCOPE,
    `${SIDEBAR_TAIL_SCOPE}:pinned`,
  ]) {
    for (const scope of [
      PINNED_ORDER_SCOPE,
      PROJECT_ORDER_SCOPE,
      RECENTS_ORDER_SCOPE,
      projectOrderScope(hostile),
    ]) {
      assert.notEqual(rowKey(scope, hostile), tailKey);
      assert.notEqual(scope, SIDEBAR_TAIL_SCOPE);
    }
  }
  // Printable: a control character makes Git treat the file as binary.
  assert.ok(
    [...SIDEBAR_TAIL_SCOPE].every((ch) => {
      const code = ch.codePointAt(0) ?? 0;
      return code > 0x1f && code !== 0x7f;
    }),
    `the tail scope holds a control character: ${JSON.stringify(SIDEBAR_TAIL_SCOPE)}`,
  );
  // Part of the layout: a row mounting at drag start shifts sections after sampling.
  const pinnedMenu = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function renderPinnedSection(): ReactNode {"),
    APP_SIDEBAR.indexOf("function renderCustomSection("),
  );
  assert.ok(pinnedMenu.length > 0, "the Pinned section moved");
  assert.match(
    pinnedMenu,
    /<SidebarMenuItem\n\s*aria-hidden\n\s*className=\{cn\(\n\s*\/\/[^\n]*\n\s*"relative z-\[1\] h-\[calc\(8px\*var\(--ui-space-scale,1\)\)\]",\n\s*dropCueClass\(SIDEBAR_TAIL_SCOPE, "pinned"\),/,
  );
  assert.ok(
    !/draggingRow && /.test(pinnedMenu),
    "the tail is conditional on a drag again",
  );
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

// A resting pointer sends no moves, so the frame loop scrolls and re-aims every frame.
test("the edge keeps scrolling while the pointer rests on it", () => {
  assert.match(HOOK, /const onFrame = \(\) => \{[^]*?frame = requestAnimationFrame\(onFrame\);\n\s*edgeScroll\(at\.y\);\n\s*if \(ghost\.current\) placeGhost\(ghost\.current, at\.y\);\n\s*track\(at\.x, at\.y\);/);
  // Unconditionally: re-aiming only on scrolled frames leaves other causes stale.
  assert.ok(!/if \(edgeScroll\(/.test(HOOK));
  assert.match(HOOK, /setDrag\(item\);\n\s*frame = requestAnimationFrame\(onFrame\);/);
  assert.match(HOOK, /if \(frame\) cancelAnimationFrame\(frame\);/);
  assert.match(HOOK, /if \(!sidebarDragSource\(\)\) \{\n\s*frame = 0;\n\s*return;\n\s*\}/);
  // One scroll driver, or a move and a frame would both step the list.
  assert.ok(!/const track = useCallback\(\n\s*\(x: number, y: number\) => \{\n\s*(auto|edge)Scroll/.test(HOOK));
});

// The copy is DOM the hook moves itself: no render per move.
test("a carried row lifts a copy that follows the pointer", () => {
  assert.match(HOOK, /started = true;\n\s*scroller\.current = scrollerOf\(row\);\n\s*ghost\.current = liftRow\(/);
  assert.match(HOOK, /const face = row\.querySelector<HTMLElement>\(ROW_FACE_SELECTOR\);/);
  assert.match(HOOK, /Math\.min\(Math\.max\(y - ghost\.grab, view\.top\), view\.bottom - height\)/);
  // Placed at the row first, so a frame before the first transform is not at the top.
  assert.match(HOOK, /top: `\$\{rect\.top\}px`,/);
  assert.match(HOOK, /translate3d\(0, \$\{Math\.round\(top - ghost\.top\)\}px, 0\)/);
  assert.match(HOOK, /dragLayer\(\)\.append\(element\);/);
  assert.match(HOOK, /dragLayer\(\)\.append\(overlay\);/);
  assert.doesNotMatch(HOOK, /document\.body\.append\((element|overlay)\)/);
  assert.match(CSS, /#pointer-drag-layer \{\n\tposition: fixed;[\s\S]*?z-index: 60;\n\tpointer-events: none;/);
  // A picture only: no zone, key, id or test id, so no hit test or lookup finds it.
  for (const attr of ["DROP_ZONE_ATTR", "ROW_KEY_ATTR", '"id"', '"data-active"', '"data-testid"', '"data-thread-id"']) {
    assert.match(HOOK, new RegExp(`GHOST_DROPPED_ATTRS = \\[[^\\]]*${attr.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
  }
  assert.match(CSS, /\.sidebar-row-ghost \* \{\n\s*pointer-events: none;/);
  // clear() is the one exit for every way a drag ends.
  assert.match(HOOK, /const clear = useCallback\(\(\) => \{[^]*?ghost\.current\?\.element\.remove\(\);\n\s*for \(const overlay of ghost\.current\?\.cues \?\? \[\]\) overlay\.remove\(\);\n\s*ghost\.current = null;/);
  assert.equal(APP_SIDEBAR.match(/draggingRow\?\.id === (item|project)\.id && "opacity-40"/g)?.length, 2);
  assert.match(CSS, /\.pointer-dragging \[data-sidebar="menu-button"\]:active \{\n\s*background-color: transparent;/);
  // Marked on the sidebar, never the body: a body class restyles the whole page.
  assert.match(HOOK, /markDragging\(sidebarOf\(row\), true\);/);
  assert.doesNotMatch(HOOK, /document\.body\.classList/);
  assert.match(CSS, /\.pointer-dragging,\n\.pointer-dragging \* \{/);
});

test("a dropped row slides from where it was let go, unless motion is reduced", () => {
  assert.match(HOOK, /const from = ghost\.current\?\.element\.getBoundingClientRect\(\)\.top \?\? null;\n\s*clear\(\);/);
  assert.match(HOOK, /onDrop\(aimed\.outcome, dragged\);\n\s*if \(from !== null && !optionsRef\.current\.reducedMotion\?\.\(\)\) settleRow\(dragged, from\);/);
  assert.match(APP_SIDEBAR, /onDrop: \(plan, dragged\) => commitDrop\(plan, dragged\),\n\s*reducedMotion: prefersReducedMotion,/);
  assert.match(HOOK, /zone\.folderId === item\.id &&/);
  assert.match(HOOK, /!all\.some\(\(other\) => other !== element && other\.contains\(element\)\)/);
});

// The window hears every pointer, so only the pressing pointer may drive the drag.
test("only the pointer that started the drag drives it", () => {
  for (const handler of [
    /function onMove\(moved: PointerEvent\) \{\n\s*if \(moved\.pointerId !== pointerId \|\| escaped\) return;/,
    /function onUp\(released: PointerEvent\) \{\n\s*if \(released\.pointerId !== pointerId\) return;/,
    /function onCancel\(aborted: PointerEvent\) \{\n\s*if \(aborted\.pointerId !== pointerId\) return;/,
  ]) {
    assert.match(HOOK, handler);
  }
  assert.match(HOOK, /if \(!event\.isPrimary\) return;/);
  // A stale gesture is abandoned, or a release the window never saw would block later drags.
  assert.match(HOOK, /press\.current\?\.end\(\);/);
  assert.match(HOOK, /end: \(\) => \{\n\s*detach\(\);\n\s*if \(started\) clear\(\);/);
  assert.match(HOOK, /if \(press\.current === self\) press\.current = null;/);
});

// The click guard must outlive Escape until release, or the row under the pointer opens.
test("escape keeps the click guard until the button is released", () => {
  assert.match(
    HOOK,
    /if \(pressed\.key !== "Escape" \|\| escaped\) return;\n\s*if \(!started\) \{[^]*?detach\(\);\n\s*return;\n\s*\}/,
  );
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
  assert.match(HOOK, /if \(moved\.pointerId !== pointerId \|\| escaped\) return;/);
});

test("a chat can be dropped at the bottom of a pinned project", () => {
  const workScope = projectOrderScope("work");
  const lastRow = chatRow("pinned", workScope, "p2", "work", {
    index: 1,
    count: 2,
  });
  const arriving = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      lastRow,
      "bottom",
      context({ orders: { ...context().orders, projectChats: () => ["p1", "p2"] } }),
    ),
  );
  assert.deepEqual(arriving.cue, {
    line: { rowKey: rowKey(workScope, "p2"), edge: "bottom", folderId: "work" },
  });
  assert.equal(arriving.effects.switchSort, "chats");
  assert.deepEqual(arriving.effects.orders[0]?.ids, ["p1", "p2", "r1"]);

  // Show more is a drop zone with no row, so it must answer for the folder's tail.
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
    line: { rowKey: rowKey(workScope, "p2"), edge: "bottom", folderId: "work" },
  });
  assert.deepEqual(tail.effects.orders[0]?.ids, ["p2", "p1", "p3"]);

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

test("the two edges of one gap are one drop, and draw one line", () => {
  const ctx = context({
    orders: { ...context().orders, recents: ["r1", "r2", "r3"] },
  });
  const drag = chat("r3", "recents", RECENTS_ORDER_SCOPE, null);
  const underR1 = plannedDrop(
    planSidebarDrop(drag, chatRow("recents", RECENTS_ORDER_SCOPE, "r1"), "bottom", ctx),
  );
  const overR2 = plannedDrop(
    planSidebarDrop(drag, chatRow("recents", RECENTS_ORDER_SCOPE, "r2"), "top", ctx),
  );
  assert.notDeepEqual(underR1.cue, overR2.cue);
  assert.ok(equivalentDrop(underR1, overR2));

  const homeScope = projectOrderScope("home");
  const intoHome = plannedDrop(
    planSidebarDrop(
      drag,
      {
        ...chatRow("projects", homeScope, "c3", "home", { index: 0, count: 1 }),
        blockEnd: { scope: homeScope, id: "c3" },
      },
      "bottom",
      ctx,
    ),
  );
  const overMisc = plannedDrop(
    planSidebarDrop(drag, folderRow("projects", PROJECT_ORDER_SCOPE, "misc"), "top", ctx),
  );
  assert.ok(!equivalentDrop(intoHome, overMisc));

  const pinnedCtx = context({
    pinnedChatIds: new Set(["p1", "p2", "p3"]),
    orders: { ...context().orders, pinned: ["p1", "p2", "p3"] },
  });
  const pinnedDrag = chat("p1", "pinned", PINNED_ORDER_SCOPE, null);
  const tail = plannedDrop(
    planSidebarDrop(
      pinnedDrag,
      { section: "pinned", blockEnd: { scope: SIDEBAR_TAIL_SCOPE, id: "pinned" } },
      "bottom",
      pinnedCtx,
    ),
  );
  const underLast = plannedDrop(
    planSidebarDrop(pinnedDrag, chatRow("pinned", PINNED_ORDER_SCOPE, "p3"), "bottom", pinnedCtx),
  );
  assert.notDeepEqual(tail.cue, underLast.cue);
  assert.ok(equivalentDrop(tail, underLast));

  assert.match(
    HOOK,
    /planSidebarDrop\(dragged, next\.zone, tail \? "bottom" : "top", context\)/,
  );
  assert.match(HOOK, /equivalentDrop\(alt, outcome\)/);
  // Only the cue is swapped: the original effects keep the `place` a slow move re-aims by.
  assert.match(HOOK, /outcome: \{ \.\.\.outcome, cue: alt\.cue \}/);
  // Neighbours are found by key and a point probe, not by scanning every zone per frame.
  assert.match(HOOK, /document\.querySelector\(`\[\$\{ROW_KEY_ATTR\}="\$\{CSS\.escape\(key\)\}"\]`\)/);
  assert.doesNotMatch(HOOK, /querySelectorAll\(`\[\$\{DROP_ZONE_ATTR\}\]`\)/);
});

test("a drop into a folder lights the folder, even while a line shows the slot", () => {
  const ctx = context();
  const drag = chat("r1", "recents", RECENTS_ORDER_SCOPE, null);
  const homeScope = projectOrderScope("home");
  const slotted = plannedDrop(
    planSidebarDrop(drag, chatRow("projects", homeScope, "c3", "home", { index: 0, count: 1 }), "top", ctx),
  );
  assert.deepEqual(slotted.action, { kind: "move", projectId: "home" });
  assert.ok("line" in slotted.cue);
  assert.equal(litRingKey(slotted), folderRingKey("home"));

  const ctxRecents = context({ orders: { ...context().orders, recents: ["r1", "r2", "r3"] } });
  const reorder = plannedDrop(
    planSidebarDrop(chat("r3", "recents", RECENTS_ORDER_SCOPE, null), chatRow("recents", RECENTS_ORDER_SCOPE, "r1"), "top", ctxRecents),
  );
  assert.equal(litRingKey(reorder), null);
  assert.equal(litRingKey(null), null);
  assert.match(HOOK, /\(key: string\): boolean => litRingKey\(plan\) === key/);
});

test("a Recents chat dropped past the last row moves to the end", () => {
  const tail: SidebarDropZone = {
    section: "recents",
    blockEnd: { scope: SIDEBAR_TAIL_SCOPE, id: "recents" },
  };
  const plan = plannedDrop(
    planSidebarDrop(chat("r1", "recents", RECENTS_ORDER_SCOPE, null), tail, "bottom", context()),
  );
  assert.deepEqual(plan.effects.orders, [{ scope: RECENTS_ORDER_SCOPE, ids: ["r2", "r1"] }]);
  assert.deepEqual(plan.cue, { line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r2"), edge: "bottom" } });
  assert.equal(
    planSidebarDrop(chat("r2", "recents", RECENTS_ORDER_SCOPE, null), tail, "bottom", context()),
    STAY,
  );
  const pinned = plannedDrop(
    planSidebarDrop(chat("p1", "pinned", PINNED_ORDER_SCOPE, null), tail, "bottom", context()),
  );
  assert.deepEqual(pinned.effects.orders.at(-1)?.ids.at(-1), "p1");
});

test("Recents draws an end strip, and the empty sidebar below it aims there", () => {
  assert.match(
    APP_SIDEBAR,
    /dropCueClass\(SIDEBAR_TAIL_SCOPE, "recents"\),\n\s*\)\}\n\s*\{\.\.\.dnd\.dropZoneProps\(\{\n\s*section: "recents",\n\s*blockEnd: \{ scope: SIDEBAR_TAIL_SCOPE, id: "recents" \},/,
  );
  assert.match(HOOK, /const past = under\.length === 0 \? zonePastRecents\(x, y, scroller\.current\) : null;/);
  assert.match(HOOK, /y < rect\.bottom \|\| x < rect\.left \|\| x > rect\.right\) return null;/);
  assert.match(HOOK, /if \(list && y > list\.getBoundingClientRect\(\)\.bottom\) return null;/);
});

test("the drop cue is drawn again above the carried row's copy", () => {
  for (const name of ["DROP_CUE_BASE", "DROP_INTO_CUE", "DROP_INTO_ROW_CUE"]) {
    assert.match(APP_SIDEBAR, new RegExp(`const ${name} = \`\\$\\{DROP_CUE_CLASS\\} `), name);
  }
  assert.match(HOOK, /track\(at\.x, at\.y\);\n\s*if \(ghost\.current\) placeCue\(ghost\.current\);/);
  assert.match(HOOK, /for \(const overlay of ghost\.current\?\.cues \?\? \[\]\) overlay\.remove\(\);/);
  assert.match(HOOK, /for \(const cue of document\.querySelectorAll<HTMLElement>\(`\.\$\{DROP_CUE_CLASS\}`\)\)/);
  // Its border only: the tint under the copy would double.
  assert.doesNotMatch(HOOK.slice(HOOK.indexOf("function placeCue"), HOOK.indexOf("function zoneOf")), /background/);
  assert.match(CSS, /\.sidebar-drop-cue-overlay \{\n\tposition: fixed;[\s\S]*?z-index: 61;[\s\S]*?pointer-events: none;/);
});

// The same spot can mean into the folder or below it, so the cue says which.
test("a line under a folder's last chat says whether the row lands in the folder or below it", () => {
  const workScope = projectOrderScope("work");
  const lastChat: SidebarDropZone = {
    ...chatRow("pinned", workScope, "c2", "work", { index: 1, count: 2 }),
    blockEnd: { scope: workScope, id: "c2" },
  };
  const into = plannedDrop(
    planSidebarDrop(chat("r1", "recents", RECENTS_ORDER_SCOPE, null), lastChat, "bottom", context()),
  );
  assert.deepEqual(into.cue, { line: { rowKey: rowKey(workScope, "c2"), edge: "bottom", folderId: "work" } });
  assert.equal(litRingKey(into), folderRingKey("work"));
  const below = plannedDrop(
    planSidebarDrop(folder("home", "projects", PROJECT_ORDER_SCOPE), lastChat, "bottom", context()),
  );
  assert.deepEqual(below.cue, { line: { rowKey: rowKey(workScope, "c2"), edge: "bottom" } });
  assert.equal(litRingKey(below), null);
  const within = plannedDrop(
    planSidebarDrop(
      chat("c1", "projects", projectOrderScope("work"), "work"),
      lastChat,
      "bottom",
      context({ chatSort: "manual" }),
    ),
  );
  assert.equal(litRingKey(within), folderRingKey("work"));
  assert.match(HOOK, /return key === rowKey\(scope, id\) \? \{ edge, inFolder: folderId !== undefined \} : undefined;/);
  assert.match(APP_SIDEBAR, /const DROP_CUE_IN_FOLDER = "before:left-\[calc\(39px\*var\(--ui-space-scale,1\)\)\]";/);
  assert.match(APP_SIDEBAR, /cue\.inFolder && DROP_CUE_IN_FOLDER,/);
  assert.match(APP_SIDEBAR, /variant === "project" \? "pl-\[calc\(39px\*var\(--ui-space-scale,1\)\)\]" : "pl-3"/);
});
