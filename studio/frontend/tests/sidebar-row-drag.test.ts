// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dragging reorders sidebar rows and moves them between lists. One planner decides every drop,
// so these ask it directly, then check the sidebar wires every row and section to it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  dragCanPin,
  dropEdgeAt,
  folderRingKey,
  planKey,
  planSidebarDrop,
  rowKey,
  sectionRingKey,
  type SidebarDragItem,
  type SidebarDropContext,
  type SidebarDropZone,
} from "../src/features/chat/lib/sidebar-drag.ts";
import {
  PINNED_ORDER_SCOPE,
  PINNED_PROJECT_ORDER_SCOPE,
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
// Recents, and p1 pinned from Recents.
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
      pinnedChats: ["p1"],
      pinnedProjects: ["work"],
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
      orders: { ...context().orders, pinnedChats: ["p1", "p2"] },
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

// An edge the row is already on is not a move, and a line there would promise one.
test("an edge that moves nothing is no drop", () => {
  const ctx = context({ chatSort: "manual" });
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r2"),
      "top",
      ctx,
    ),
    null,
  );
  assert.equal(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE, null),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "bottom",
      ctx,
    ),
    null,
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
    // A pinned folder files it just the same.
    folderRow("pinned", PINNED_PROJECT_ORDER_SCOPE, "work"),
  ]) {
    const plan = planSidebarDrop(drag, zone, "top", context());
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
    null,
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
  assert.deepEqual(onRow.effects.orders, [
    { scope: PINNED_ORDER_SCOPE, ids: ["r1", "p1"] },
  ]);
  assert.deepEqual(onRow.cue, {
    line: { rowKey: rowKey(PINNED_ORDER_SCOPE, "p1"), edge: "top" },
  });
  // The section itself, and its header while it is closed, pin it last.
  for (const zone of [
    { section: "pinned" } as SidebarDropZone,
    { section: "pinned", header: true } as SidebarDropZone,
  ]) {
    const plan = planSidebarDrop(drag, zone, "bottom", context());
    assert.deepEqual(plan?.cue, { ring: sectionRingKey("pinned") });
    assert.deepEqual(plan?.effects.orders, [
      { scope: PINNED_ORDER_SCOPE, ids: ["p1", "r1"] },
    ]);
  }
  // Pinned sorted by a rule would move the row again, so the drop takes it to Manual.
  assert.equal(
    planSidebarDrop(drag, { section: "pinned" }, "bottom", context({ pinnedSort: "updated" }))
      ?.effects.switchSort,
    "pinned",
  );
  // A chat already pinned is not pinned again: it reorders.
  assert.equal(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE, null),
      { section: "pinned" },
      "bottom",
      context(),
    ),
    null,
  );
});

// Dragging a pinned chat onto a folder under Projects unpins it, and files it when the folder
// is another one. Onto a pinned folder it keeps its pin.
test("a pinned chat dragged onto a folder is unpinned, and filed when the folder is new", () => {
  const ctx = context({
    pinnedChatIds: new Set(["p1", "c3"]),
    orders: { ...context().orders, pinnedChats: ["p1", "c3"] },
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
  const pinnedFolder = planSidebarDrop(
    chat("c3", "pinned", PINNED_ORDER_SCOPE, "home"),
    folderRow("pinned", PINNED_PROJECT_ORDER_SCOPE, "work"),
    "top",
    ctx,
  );
  assert.deepEqual(pinnedFolder?.action, { kind: "move", projectId: "work" });
  assert.equal(pinnedFolder?.effects.unpinChat, undefined);
});

test("a folder reorders among its own list, and the block under it aims at it", () => {
  const ctx = context({
    pinnedProjectIds: new Set(["work", "play"]),
    orders: { ...context().orders, pinnedProjects: ["work", "play"] },
  });
  // Over the other folder's row.
  const onRow = planSidebarDrop(
    folder("play", "pinned", PINNED_PROJECT_ORDER_SCOPE),
    folderRow("pinned", PINNED_PROJECT_ORDER_SCOPE, "work"),
    "top",
    ctx,
  );
  assert.deepEqual(onRow?.effects.orders, [
    { scope: PINNED_PROJECT_ORDER_SCOPE, ids: ["play", "work"] },
  ]);
  // Over the chats under it: the top half of the block aims above the folder, the bottom half
  // below it, so the line flips once, in the middle.
  const upper = planSidebarDrop(
    folder("play", "pinned", PINNED_PROJECT_ORDER_SCOPE),
    chatRow("pinned", projectOrderScope("work"), "c1", "work", { index: 0, count: 2 }),
    "top",
    ctx,
  );
  assert.deepEqual(upper?.cue, {
    line: { rowKey: rowKey(PINNED_PROJECT_ORDER_SCOPE, "work"), edge: "top" },
  });
  const lower = planSidebarDrop(
    folder("play", "pinned", PINNED_PROJECT_ORDER_SCOPE),
    chatRow("pinned", projectOrderScope("work"), "c2", "work", { index: 1, count: 2 }),
    "bottom",
    ctx,
  );
  // Already below "work": that edge moves nothing.
  assert.equal(lower, null);
  // Its own block is never a target, and neither is Recents or a chat list's row.
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_PROJECT_ORDER_SCOPE),
      chatRow("pinned", projectOrderScope("work"), "c1", "work", { index: 0, count: 2 }),
      "top",
      ctx,
    ),
    null,
  );
  assert.equal(
    planSidebarDrop(
      folder("work", "pinned", PINNED_PROJECT_ORDER_SCOPE),
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
    folderRow("pinned", PINNED_PROJECT_ORDER_SCOPE, "work"),
    "top",
    context(),
  );
  assert.deepEqual(pin?.action, { kind: "pin" });
  assert.equal(pin?.effects.pinProject, "home");
  assert.deepEqual(pin?.effects.orders, [
    { scope: PINNED_PROJECT_ORDER_SCOPE, ids: ["home", "work"] },
  ]);
  // Over Pinned's chats or the section: last among the pinned folders.
  const onChat = planSidebarDrop(
    folder("home", "projects", PROJECT_ORDER_SCOPE),
    chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
    "top",
    context(),
  );
  assert.deepEqual(onChat?.cue, { ring: sectionRingKey("pinned") });
  assert.deepEqual(onChat?.effects.orders, [
    { scope: PINNED_PROJECT_ORDER_SCOPE, ids: ["work", "home"] },
  ]);
  const unpin = planSidebarDrop(
    folder("work", "pinned", PINNED_PROJECT_ORDER_SCOPE),
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
    folder("work", "pinned", PINNED_PROJECT_ORDER_SCOPE),
    { section: "projects", header: true },
    "top",
    context(),
  );
  assert.deepEqual(unpinOnHeader?.cue, { ring: sectionRingKey("projects") });
});

// A first pin needs Pinned on screen to land in.
test("Pinned offers itself to a drag that could pin", () => {
  const ctx = context();
  assert.equal(dragCanPin(chat("r1", "recents", RECENTS_ORDER_SCOPE, null), ctx), true);
  assert.equal(dragCanPin(chat("p1", "pinned", PINNED_ORDER_SCOPE, null), ctx), false);
  assert.equal(dragCanPin(folder("home", "projects", PROJECT_ORDER_SCOPE), ctx), true);
  assert.equal(dragCanPin(folder("work", "pinned", PINNED_PROJECT_ORDER_SCOPE), ctx), false);
  assert.match(
    APP_SIDEBAR,
    /\(organizeBy === "project" && pinnedProjectRecords\.length > 0\) \|\|\n(?:\s*\/\/.*\n)*\s*dnd\.pinnedTakesDrag\) && \(/,
  );
  assert.ok(APP_SIDEBAR.includes('t("shell.drag.dropToPin")'));
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
    /row: \{ id: project\.id, kind: "project", scope: order\.scope \},\n\s*folderId: project\.id,\n\s*\},\n\s*\{ closed: !expanded \},/,
  );
  // An empty folder's line and its "Show more" row are the folder's block too.
  assert.equal(
    (
      APP_SIDEBAR.match(
        /\{\.\.\.dnd\.dropZoneProps\(\{ section: order\.section, folderId: project\.id \}\)\}/g,
      ) ?? []
    ).length,
    2,
  );
  // Each section's body and its header, which stands in while the section is closed.
  for (const section of ["pinned", "projects", "recents"]) {
    assert.ok(
      APP_SIDEBAR.includes(`{...dnd.dropZoneProps({ section: "${section}" })}`),
      `${section} body is not a zone`,
    );
    assert.ok(
      APP_SIDEBAR.includes(`{ section: "${section}", header: true }`),
      `${section} header is not a zone`,
    );
  }
  // Each list hands its rows the section they are drawn in.
  for (const wiring of [
    'scope: PINNED_ORDER_SCOPE,\n                        ids: pinnedRowIds,\n                        section: "pinned",',
    'scope: RECENTS_ORDER_SCOPE,\n                        ids: recentRowIds,\n                        section: "recents",',
    'orderedIds: pinnedProjectRowIds,\n                          section: "pinned",',
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

// A section's collapsible clips its overflow, so cues stay inside the row.
test("every drop cue is drawn inside its row", () => {
  for (const cue of [
    "${DROP_CUE_BASE} before:top-0",
    "${DROP_CUE_BASE} before:bottom-0",
    "before:inset-x-1 before:inset-y-0 ",
  ]) {
    assert.ok(APP_SIDEBAR.includes(cue), `no cue is drawn at ${cue}`);
  }
  assert.ok(
    !/before:-inset-y-px|before:-top-px|before:-bottom-px/.test(APP_SIDEBAR),
    "a cue still hangs outside its row, where a clipped section cuts it off",
  );
});

// Alt + arrow is the keyboard's reorder.
test("alt and an arrow reorder a row without a pointer", () => {
  assert.match(
    APP_SIDEBAR,
    /onKeyDown: \(event: React\.KeyboardEvent\) => \{\n\s*if \(\n\s*!event\.altKey \|\|\n\s*\(event\.key !== "ArrowUp" && event\.key !== "ArrowDown"\)\n\s*\)/,
  );
  assert.match(APP_SIDEBAR, /moveIdBy\(\n\s*orderedIds,\n\s*item\.id,\n\s*event\.key === "ArrowDown" \? 1 : -1,\n\s*\)/);
  // Same rule as a drop.
  assert.match(APP_SIDEBAR, /if \(resorts && !reorderSwitchesSort\) return;/);
  for (const gone of ["renderMoveRowItems", "shell.organize.moveUp", "shell.organize.moveDown"]) {
    assert.ok(!APP_SIDEBAR.includes(gone), `${gone} survived in the sidebar`);
  }
});

// The hint names every kind of drop.
test("the hint beside the cursor names every kind of drop", () => {
  for (const key of ["reorder", "pin", "unpin", "moveTo", "moveToRecents", "dropToPin"]) {
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
