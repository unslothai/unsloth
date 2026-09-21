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
  STAY,
  type SidebarDragItem,
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
    reorderSwitchesSort: true,
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
): SidebarDragItem => ({ kind: "project", id, section, scope, projectId: null });

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
  const plan = planSidebarDrop(
    chat("r2", "recents", RECENTS_ORDER_SCOPE, null),
    chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
    "top",
    context({ chatSort: "manual" }),
  );
  assert.ok(plan);
  assert.deepEqual(plan.action, { kind: "reorder" });
  assert.deepEqual(plan.cue, {
    line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r1"), edge: "top" },
  });
  assert.deepEqual(plan.effects.orders, [
    { scope: RECENTS_ORDER_SCOPE, ids: ["r2", "r1"] },
  ]);
  assert.equal(plan.effects.switchSort, undefined);
});

// A sorted list would undo the drop, so it switches to Manual, unless the user turned that off.
test("a reorder in a sorted list switches it to Manual order, or is refused", () => {
  const drag = chat("r2", "recents", RECENTS_ORDER_SCOPE, null);
  const zone = chatRow("recents", RECENTS_ORDER_SCOPE, "r1");
  const switching = planSidebarDrop(drag, zone, "top", context());
  assert.equal(switching?.effects.switchSort, "chats");
  const pinnedDrag = chat("p1", "pinned", PINNED_ORDER_SCOPE, null);
  const pinnedZone = chatRow("pinned", PINNED_ORDER_SCOPE, "p2");
  const pinnedSwitch = planSidebarDrop(
    pinnedDrag,
    pinnedZone,
    "bottom",
    context({
      pinnedSort: "updated",
      pinnedChatIds: new Set(["p1", "p2"]),
      orders: { ...context().orders, pinned: ["work", "p1", "p2"] },
    }),
  );
  assert.equal(pinnedSwitch?.effects.switchSort, "pinned");
  assert.equal(
    planSidebarDrop(drag, zone, "top", context({ reorderSwitchesSort: false })),
    null,
  );
  // And a list already on Manual never asks.
  assert.equal(
    planSidebarDrop(
      drag,
      zone,
      "top",
      context({ chatSort: "manual", reorderSwitchesSort: false }),
    )?.action.kind,
    "reorder",
  );
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
    chatRow("pinned", projectOrderScope("work"), "c1", "work", { index: 0, count: 2 }),
    { ...folderRow("pinned", PINNED_ORDER_SCOPE, "work"), edge: "bottom" },
  ] as Array<SidebarDropZone & { edge?: "top" | "bottom" }>) {
    const plan = planSidebarDrop(drag, zone, zone.edge ?? "top", context());
    assert.ok(plan, `no plan for ${JSON.stringify(zone)}`);
    assert.deepEqual(plan.action, { kind: "move", projectId: zone.folderId });
    assert.deepEqual(plan.cue, { ring: folderRingKey(zone.folderId!) });
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
  const plan = planSidebarDrop(
    chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
    chatRow("pinned", projectOrderScope("work"), "c2", "work", {
      index: 1,
      count: 2,
    }),
    "top",
    context({ chatSort: "manual" }),
  );
  assert.ok(plan);
  assert.deepEqual(plan.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "top" },
  });
  assert.deepEqual(plan.effects.orders, [
    { scope: projectOrderScope("work"), ids: ["c1", "r1", "c2"] },
  ]);
});

test("a chat dropped on Recents leaves its folder and its pin", () => {
  // Out of a folder.
  const unfiled = planSidebarDrop(
    chat("c3", "projects", projectOrderScope("home"), "home"),
    { section: "recents" },
    "top",
    context(),
  );
  assert.ok(unfiled);
  assert.deepEqual(unfiled.action, { kind: "move", projectId: null });
  assert.deepEqual(unfiled.cue, { ring: sectionRingKey("recents") });
  assert.deepEqual(unfiled.effects.moveChat, { chatId: "c3", projectId: null });
  // Out of Pinned: the pin goes, and so does the folder that would keep it out of Recents.
  const unpinned = planSidebarDrop(
    chat("p1", "pinned", PINNED_ORDER_SCOPE, "home"),
    chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
    "top",
    context(),
  );
  assert.ok(unpinned);
  assert.equal(unpinned.effects.unpinChat, "p1");
  assert.deepEqual(unpinned.effects.moveChat, { chatId: "p1", projectId: null });
  // With folders off every chat is already a Recents row, so only the pin goes.
  const listMode = planSidebarDrop(
    chat("p1", "pinned", PINNED_ORDER_SCOPE, "home"),
    { section: "recents", header: true },
    "top",
    context({ organizeBy: "list" }),
  );
  assert.deepEqual(listMode?.action, { kind: "unpin" });
  assert.equal(listMode?.effects.moveChat, undefined);
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
  const onRow = planSidebarDrop(
    drag,
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "top",
    context(),
  );
  assert.ok(onRow);
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
    { section: "pinned", header: true, row: { id: "work", kind: "project", scope: PINNED_ORDER_SCOPE } },
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
  ] as SidebarDropZone[]) {
    const plan = planSidebarDrop(drag, zone, "top", context());
    assert.deepEqual(plan?.action, { kind: "pin" });
    assert.deepEqual(plan?.cue, {
      line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "work"), edge: "top" },
    });
    assert.deepEqual(plan?.effects.orders, [
      { scope: PINNED_ORDER_SCOPE, ids: ["r1", "work", "p1"] },
    ]);
  }
  // The space under the rows lands it last, with a line under the last row, never a ring.
  const under = planSidebarDrop(drag, { section: "pinned" }, "bottom", context());
  assert.deepEqual(under?.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "bottom" },
  });
  assert.deepEqual(under?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "p1", "r1"] },
  ]);
  // Pinned sorted by a rule would move the row again, so the drop takes it to Manual.
  assert.equal(
    planSidebarDrop(drag, { section: "pinned" }, "bottom", context({ pinnedSort: "updated" }))
      ?.effects.switchSort,
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
  const ownFolder = planSidebarDrop(
    chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
    folderRow("projects", PROJECT_ORDER_SCOPE, "home"),
    "top",
    ctx,
  );
  assert.deepEqual(ownFolder?.action, { kind: "unpin" });
  assert.equal(ownFolder?.effects.unpinChat, "c3");
  assert.equal(ownFolder?.effects.moveChat, undefined);
  const otherFolder = planSidebarDrop(
    chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
    folderRow("projects", PROJECT_ORDER_SCOPE, "misc"),
    "top",
    ctx,
  );
  assert.deepEqual(otherFolder?.action, { kind: "move", projectId: "misc" });
  assert.equal(otherFolder?.effects.unpinChat, "c3");
  // In Pinned the folder row's upper half is a slot, so the file lands on its lower half.
  const pinnedFolder = planSidebarDrop(
    chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
    "bottom",
    ctx,
  );
  assert.deepEqual(pinnedFolder?.action, { kind: "move", projectId: "work" });
  assert.equal(pinnedFolder?.effects.unpinChat, undefined);
});

test("a folder reorders among its own list, and the block under it aims at it", () => {
  const ctx = context({
    pinnedProjectIds: new Set(["work", "play"]),
    orders: { ...context().orders, pinned: ["work", "p1", "play"] },
  });
  // Over the other folder's row.
  const onRow = planSidebarDrop(
    folder("play", "pinned", PINNED_ORDER_SCOPE),
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
    "top",
    ctx,
  );
  assert.deepEqual(onRow?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["play", "work", "p1"] },
  ]);
  // Over the chats under it: the top half of the block aims above the folder, the bottom half
  // below it, so the line flips once, in the middle. Below draws under the block's last row.
  const upper = planSidebarDrop(
    folder("play", "pinned", PINNED_ORDER_SCOPE),
    chatRow("pinned", projectOrderScope("work"), "c1", "work", { index: 0, count: 2 }),
    "top",
    ctx,
  );
  assert.deepEqual(upper?.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "work"), edge: "top" },
  });
  const lower = planSidebarDrop(
    folder("play", "pinned", PINNED_ORDER_SCOPE),
    {
      ...chatRow("pinned", projectOrderScope("work"), "c2", "work", { index: 1, count: 2 }),
      blockEnd: { scope: projectOrderScope("work"), id: "c2" },
    },
    "bottom",
    ctx,
  );
  assert.deepEqual(lower?.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "bottom" },
  });
  assert.deepEqual(lower?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "play", "p1"] },
  ]);
  // Over a block's own rows, Show more or an empty folder, it lands below that folder.
  const tail = planSidebarDrop(
    folder("play", "pinned", PINNED_ORDER_SCOPE),
    {
      section: "pinned",
      folderId: "work",
      blockEnd: { scope: projectOrderScope("work"), id: "c2" },
    },
    "top",
    ctx,
  );
  assert.deepEqual(tail?.cue, {
    line: { rowKey: rowKey(projectOrderScope("work"), "c2"), edge: "bottom" },
  });
  assert.deepEqual(tail?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "play", "p1"] },
  ]);
  // A folder lands between pinned chats too: Pinned is one list.
  const belowChat = planSidebarDrop(
    folder("work", "pinned", PINNED_ORDER_SCOPE),
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "bottom",
    ctx,
  );
  assert.deepEqual(belowChat?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["p1", "work", "play"] },
  ]);
  // Its own block keeps it where it is, and Recents is no target at all.
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_ORDER_SCOPE),
      chatRow("pinned", projectOrderScope("work"), "c1", "work", { index: 0, count: 2 }),
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
  const pin = planSidebarDrop(
    folder("home", "projects", PROJECT_ORDER_SCOPE),
    folderRow("pinned", PINNED_ORDER_SCOPE, "work"),
    "top",
    context(),
  );
  assert.deepEqual(pin?.action, { kind: "pin" });
  assert.equal(pin?.effects.pinProject, "home");
  assert.deepEqual(pin?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["home", "work", "p1"] },
  ]);
  // Over a pinned chat: the slot the line shows, since Pinned is one list.
  const onChat = planSidebarDrop(
    folder("home", "projects", PROJECT_ORDER_SCOPE),
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "top",
    context(),
  );
  assert.deepEqual(onChat?.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "top" },
  });
  assert.deepEqual(onChat?.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["work", "home", "p1"] },
  ]);
  // Pinned sorted by a rule would move the row again, so the drop takes it to Manual.
  assert.equal(
    planSidebarDrop(
      folder("home", "projects", PROJECT_ORDER_SCOPE),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      context({ pinnedSort: "updated" }),
    )?.effects.switchSort,
    "pinned",
  );
  const unpin = planSidebarDrop(
    folder("work", "pinned", PINNED_ORDER_SCOPE),
    folderRow("projects", PROJECT_ORDER_SCOPE, "misc"),
    "bottom",
    context(),
  );
  assert.deepEqual(unpin?.action, { kind: "unpin" });
  assert.equal(unpin?.effects.unpinProject, "work");
  assert.deepEqual(unpin?.effects.orders, [
    { scope: PROJECT_ORDER_SCOPE, ids: ["home", "misc", "work"] },
  ]);
  const unpinOnHeader = planSidebarDrop(
    folder("work", "pinned", PINNED_ORDER_SCOPE),
    { section: "projects", header: true },
    "top",
    context(),
  );
  assert.deepEqual(unpinOnHeader?.cue, { ring: sectionRingKey("projects") });
});

// The gap between a header and its first row is where "above the first row" is aimed, so the
// header stands for that edge for a drag of the same kind, and for the bare section otherwise.
test("a section header stands for the top of its first row", () => {
  const ctx = context({ chatSort: "manual" });
  const aboveFirstChat = planSidebarDrop(
    chat("r2", "recents", RECENTS_ORDER_SCOPE, null),
    { section: "recents", header: true, row: { id: "r1", kind: "chat", scope: RECENTS_ORDER_SCOPE } },
    "bottom",
    ctx,
  );
  assert.deepEqual(aboveFirstChat?.cue, {
    line: { rowKey: rowKey(RECENTS_ORDER_SCOPE, "r1"), edge: "top" },
  });
  assert.deepEqual(aboveFirstChat?.effects.orders, [
    { scope: RECENTS_ORDER_SCOPE, ids: ["r2", "r1"] },
  ]);
  const aboveFirstFolder = planSidebarDrop(
    folder("misc", "projects", PROJECT_ORDER_SCOPE),
    { section: "projects", header: true, row: { id: "home", kind: "project", scope: PROJECT_ORDER_SCOPE } },
    "bottom",
    ctx,
  );
  assert.deepEqual(aboveFirstFolder?.cue, {
    line: { rowKey: rowKey(PROJECT_ORDER_SCOPE, "home"), edge: "top" },
  });
  // A chat over the Projects header does not file into the first folder.
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      { section: "projects", header: true, row: { id: "home", kind: "project", scope: PROJECT_ORDER_SCOPE } },
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
  const a = planSidebarDrop(
    chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "top",
    ctx,
  );
  const b = planSidebarDrop(
    chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "bottom",
    ctx,
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
  const store = await readSrcAsync("features/chat/stores/sidebar-drag-source.ts");
  assert.match(store, /let source: SidebarDragItem \| null = null;/);
  assert.match(HOOK, /setSidebarDragSource\(item\);\n\s*setDrag\(item\);/);
  // Every handler reads it from there; the state is only what paints.
  assert.equal((HOOK.match(/const dragged = sidebarDragSource\(\);/g) ?? []).length, 2);
  assert.ok(!/const dragged = drag;/.test(HOOK));
});

// dragleave fires for every child crossed, so one document listener clears the cue instead.
test("a cue over nothing is cleared without trusting dragleave", () => {
  assert.ok(!HOOK.includes("onDragLeave"));
  assert.ok(!APP_SIDEBAR.includes("onDragLeave"));
  assert.match(HOOK, /lastHandledEvent = event\.nativeEvent;/);
  assert.match(HOOK, /if \(lastHandledEvent !== event\) \{\n\s*cancelSpring\(\);\n\s*showPlan\(null\);/);
  // No answer lets the outer zone answer; an answer marks the event rather than stopping it.
  assert.match(HOOK, /if \(!next\) return;\n\s*event\.preventDefault\(\);\n\s*lastHandledEvent = event\.nativeEvent;/);
  assert.ok(!HOOK.includes("event.stopPropagation();\n          lastHandledEvent"));
  assert.match(HOOK, /if \(!dragged \|\| lastHandledEvent === event\.nativeEvent\) return;/);
});

// Over its own row a lifted row is already home. The row claims the drag and paints nothing,
// so the section body never gets to offer its last slot for it.
test("a row over itself claims the drag and paints nothing", () => {
  assert.match(
    HOOK,
    /if \(next === STAY\) \{\n\s*\/\/[^\n]*\n\s*cancelSpring\(\);\n\s*showPlan\(null\);\n\s*return;\n\s*\}/,
  );
  assert.match(HOOK, /if \(next !== STAY\) optionsRef\.current\.onDrop\(next, dragged\);/);
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
  assert.match(APP_SIDEBAR, /reorderRowBy\(item, orderedIds, sort, event\.key === "ArrowDown" \? 1 : -1\);/);
  // Same rule as a drop.
  assert.match(APP_SIDEBAR, /if \(resorts && !reorderSwitchesSort\) return;/);
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
  assert.match(MOBILE, /const COARSE_POINTER_QUERY = "\(any-pointer: coarse\)";/);
  for (const key of ["moveUp", "moveDown"]) {
    assert.ok(EN.includes(`${key}:`), `${key} is missing from the en locale`);
  }
  // Both paths share the reorder, so both honour the sort rule.
  assert.equal((APP_SIDEBAR.match(/reorderRowBy\(item, orderedIds, sort, /g) ?? []).length, 3);
});

// The hint names every kind of drop.
test("the hint beside the cursor names every kind of drop", () => {
  for (const key of ["reorder", "pin", "unpin", "moveTo", "moveToRecents"]) {
    assert.ok(EN.includes(`${key}:`), `${key} is missing from the en locale`);
  }
  for (const use of [
    't("shell.drag.pin")',
    't("shell.drag.unpin")',
    't("shell.drag.reorder")',
    't("shell.drag.moveToRecents")',
    't("shell.drag.moveTo", { name })',
  ]) {
    assert.ok(APP_SIDEBAR.includes(use), `${use} is never rendered`);
  }
  // In a portal, and only while the user wants it.
  assert.match(APP_SIDEBAR, /draggingRow && dragHints && typeof document !== "undefined"\n\s*\? createPortal\(/);
  assert.ok(APP_SIDEBAR.includes('data-testid="sidebar-drop-hint"'));
});

// The three preferences persist and sit in the Organize menu.
test("drag-and-drop preferences persist and show in the Organize menu", () => {
  const state = useSidebarOrganizationStore.getState();
  assert.equal(state.dragHints, true);
  assert.equal(state.reorderSwitchesSort, true);
  assert.equal(state.dragOpensFolders, true);
  state.setDragHints(false);
  state.setReorderSwitchesSort(false);
  state.setDragOpensFolders(false);
  const next = useSidebarOrganizationStore.getState();
  assert.equal(next.dragHints, false);
  assert.equal(next.reorderSwitchesSort, false);
  assert.equal(next.dragOpensFolders, false);
  for (const key of ["dragDrop", "dragHints", "reorderSwitchesSort", "dragOpensFolders"]) {
    assert.ok(EN.includes(`${key}:`), `${key} is missing from the en locale`);
    assert.ok(APP_SIDEBAR.includes(`shell.organize.${key}`), `${key} is never rendered`);
  }
  assert.match(APP_SIDEBAR, /<DropdownMenuCheckboxItem\n\s*key=\{option\.key\}\n\s*checked=\{option\.get\(\)\}/);
});

// Resting on a closed folder or section opens it.
test("a closed folder or section opens under a resting pointer", () => {
  assert.match(HOOK, /export const SPRING_OPEN_DELAY_MS = \d+;/);
  assert.match(
    APP_SIDEBAR,
    /onSpringOpen: \(zone\) => \{\n\s*if \(zone\.folderId\) \{\n\s*if \(collapsedProjectIds\.has\(zone\.folderId\)\) \{\n\s*toggleProjectCollapsed\(zone\.folderId\);/,
  );
  assert.match(APP_SIDEBAR, /if \(zone\.section === "pinned"\) setPinnedOpen\(true\);/);
  assert.match(APP_SIDEBAR, /springOpen: dragOpensFolders,/);
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
    /const move = effects\.moveChat;\n\s*if \(!move\) \{\n\s*applyOrders\(\);\n\s*return;\n\s*\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /void moveChatToProject\(item, move\.projectId\)\.then\(\(moved\) => \{\n\s*if \(!moved\) return;\n\s*applyOrders\(\);\n\s*if \(unpinAfter\) usePinnedChatsStore\.getState\(\)\.unpin\(unpinAfter\);/,
  );
  // And nothing else in commitDrop writes an order on its own.
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  assert.equal((commit.match(/setManualOrder\(/g) ?? []).length, 1);
  assert.match(APP_SIDEBAR, /\): Promise<boolean> \{\n\s*if \(item\.projectId === projectId\) return true;/);
});
