// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sidebar rows are reordered by dragging them, not by a pair of menu items, and a chat is filed
// into a project by being dropped on it. What that costs is a drop landing in a list that sorts
// itself: these check the row wiring that keeps both honest.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");
const EN = await readSrcAsync("i18n/locales/en.ts");
const DERIVED = await readSrcAsync("lib/hugeicons-derived.ts");

// Dragging is the only reorder gesture now, so the menu items that duplicated it are gone;
// including their strings, which would otherwise sit in twelve locales with nothing to render them.
test("no row menu still offers Move up or Move down", () => {
  for (const gone of [
    "renderMoveRowItems",
    "shell.organize.moveUp",
    "shell.organize.moveDown",
  ]) {
    assert.ok(!APP_SIDEBAR.includes(gone), `${gone} survived in the sidebar`);
  }
  for (const gone of ["moveUp:", "moveDown:"]) {
    assert.ok(!EN.includes(gone), `${gone} survived in the en locale`);
  }
});

// Rows used to be draggable only while their list was already on Manual order, which left the
// gesture invisible on the sort every install starts on.
test("a row drags whatever its list is sorted by", () => {
  for (const gate of ["manualDragEnabled", "pinnedDragEnabled"]) {
    assert.ok(!APP_SIDEBAR.includes(gate), `${gate} still gates dragging`);
  }
  // Every chat list hands its row the sort a drop has to switch, and the folders carry none:
  // they order among themselves whatever the chats are sorted by.
  for (const sort of [
    "sort: { value: chatSort, set: setChatSort }",
    "sort: { value: pinnedSort, set: setPinnedSort }",
  ]) {
    assert.ok(APP_SIDEBAR.includes(sort), `no list passes ${sort}`);
  }
});

// A drop into a list on Priority or Last updated would be undone by the next render, so the drop
// takes the list to Manual order and seeds it with the rows exactly as they stand.
test("a drop switches a re-sorting list to Manual order", () => {
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitRowOrder("),
    APP_SIDEBAR.indexOf("function rowDragProps("),
  );
  assert.ok(commit.length > 0, "commitRowOrder moved");
  // Nothing to persist, nothing to switch: an unchanged order leaves the sort alone.
  assert.match(commit, /if \(next === previous\) return;/);
  assert.match(commit, /setManualOrder\(scope, next\);/);
  assert.match(commit, /if \(sort && sort\.value !== "manual"\) \{\n\s*sort\.set\("manual"\);/);
  assert.match(commit, /toast\.info\(t\("shell\.organize\.switchedToManual"\)\)/);
});

// The folders are the drop targets that move a chat between projects; Recents is the one that
// takes it out of every folder. Pinned offers neither: a chat joins it by being pinned.
test("a chat can be dropped into a folder and back out to Recents", () => {
  // The folder row itself, and every chat row under it, name the same destination, so the drop
  // lands anywhere on the folder rather than on one row of it.
  assert.match(
    APP_SIDEBAR,
    /rowDragProps\(\{\n\s*scope: order\.scope,\n\s*orderedIds: order\.orderedIds,\n\s*rowId: project\.id,\n\s*kind: "project",\n\s*chatDrop: \{ projectId: project\.id \},/,
  );
  assert.match(
    APP_SIDEBAR,
    /scope: projectOrderScope\(project\.id\),\n\s*ids: projectChatIds,\n\s*sort: \{ value: chatSort, set: setChatSort \},\n\s*chatDrop: \{ projectId: project\.id \},/,
  );
  // An empty folder has no row to land on, so its "no chats" line takes the drop.
  assert.match(
    APP_SIDEBAR,
    /<SidebarMenuItem \{\.\.\.sectionChatDropProps\(project\.id\)\}>/,
  );
  // Recents takes the drop as a section, so an empty one is still a target.
  assert.match(APP_SIDEBAR, /\{\.\.\.sectionChatDropProps\(null\)\}/);
  assert.match(
    APP_SIDEBAR,
    /scope: RECENTS_ORDER_SCOPE,\n\s*ids: recentRowIds,\n\s*sort: \{ value: chatSort, set: setChatSort \},\n\s*chatDrop: \{ projectId: null \},/,
  );
  // Pinned reorders only: its rows carry a sort and no destination.
  const pinnedRows = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("{sortedPinnedChatItems.map("),
    APP_SIDEBAR.indexOf("</SidebarMenu>", APP_SIDEBAR.indexOf("{sortedPinnedChatItems.map(")),
  );
  assert.ok(pinnedRows.length > 0, "the Pinned chat rows moved");
  assert.ok(
    !pinnedRows.includes("chatDrop"),
    "Pinned takes a chat dropped in from another list",
  );
});

// Dropping a chat where it already lives is not a move, and a folder must not light up for it.
test("a folder only accepts a chat that would actually move", () => {
  assert.match(
    APP_SIDEBAR,
    /function acceptsDraggedChat\(\n\s*dragged: SidebarDragSource \| null,\n\s*projectId: string \| null,\n\s*\): boolean \{\n\s*return dragged\?\.kind === "chat" && \(dragged\.projectId \?\? null\) !== projectId;/,
  );
  // And the row reports the folder it is in, not the one it might be dropped on.
  assert.match(APP_SIDEBAR, /projectId: item\.projectId \?\? null,/);
});

// Pinned draws a folder's chats under it, so its folder rows are a block apart. The rows between
// them have to answer for the folder they belong to, or the only target is a row the pointer has
// to travel a whole section to reach, which is what made a pinned folder read as undraggable.
test("a folder dragged over another folder's chats lands against that folder", () => {
  assert.match(
    APP_SIDEBAR,
    /const landing = folderDropTarget\(\{\n\s*draggedId: dragged\.id,\n\s*folderIds: folderOrder\.orderedIds,\n\s*folderId: folderOrder\.projectId,/,
  );
  // The folder's chats carry the folder's own order, not just their own.
  assert.match(
    APP_SIDEBAR,
    /folderOrder: \{\n\s*scope: order\.scope,\n\s*orderedIds: order\.orderedIds,\n\s*projectId: project\.id,\n\s*\},/,
  );
  // The line lands on the folder row, and the drop writes the folder list, not the chat list.
  assert.match(APP_SIDEBAR, /setDropTargetRow\(folderTarget\);/);
  assert.match(
    APP_SIDEBAR,
    /commitRowOrder\(\n\s*folderOrder\.scope,\n\s*folderTarget\.next,\n\s*folderOrder\.orderedIds,\n\s*\);/,
  );
});

// An edge the row is already on is not a move. Drawing the line there promised one, and in Pinned
// that edge can be most of a section away from the row being dragged.
test("an edge that moves nothing takes no line and no drop", () => {
  assert.match(
    APP_SIDEBAR,
    /if \(insertIdAt\(orderedIds, dragged\.id, rowId, edge\) === orderedIds\) \{\n\s*setDropTargetRow\(\(prev\) => \(prev\?\.id === rowId \? null : prev\)\);\n\s*return;\n\s*\}/,
  );
  // The edge is read before the drop is accepted, so the refusal reaches the cursor too.
  const over = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("onDragOver: (event: React.DragEvent) => {"),
    APP_SIDEBAR.indexOf("onDragLeave: () => {"),
  );
  assert.ok(
    over.indexOf("const edge = dropEdgeAt(") < over.indexOf("event.preventDefault()"),
    "the drop is accepted before it is known to change anything",
  );
});

// A section's collapsible clips its overflow, so a cue drawn a pixel proud of the row lost that
// edge on the first row, which is exactly where a folder sits in Pinned.
test("every drop cue is drawn inside its row", () => {
  // Written as strings: a regex literal holding a backtick trips the type stripper.
  for (const cue of [
    "${DROP_CUE_BASE} before:top-0",
    "${DROP_CUE_BASE} before:bottom-0",
    '"before:absolute before:inset-x-1 before:inset-y-0 ',
  ]) {
    assert.ok(APP_SIDEBAR.includes(cue), `no cue is drawn at ${cue}`);
  }
  assert.ok(
    !/before:-inset-y-px|before:-top-px|before:-bottom-px/.test(APP_SIDEBAR),
    "a cue still hangs outside its row, where a clipped section cuts it off",
  );
});

// A folder dragged onto a chat list has no slot in it, and a chat has none among the folders.
test("a drag only reorders rows of its own kind and list", () => {
  assert.match(
    APP_SIDEBAR,
    /const reordersWith = \(dragged: typeof draggingRow\) =>\n\s*dragged\?\.kind === kind && dragged\.scope === scope;/,
  );
});

// Removing the menu items took the only reorder a keyboard had, so the row itself grew one.
test("alt and an arrow reorder a row without a pointer", () => {
  assert.match(
    APP_SIDEBAR,
    /onKeyDown: \(event: React\.KeyboardEvent\) => \{\n\s*if \(\n\s*!event\.altKey \|\|\n\s*\(event\.key !== "ArrowUp" && event\.key !== "ArrowDown"\)\n\s*\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /moveIdBy\(orderedIds, rowId, event\.key === "ArrowDown" \? 1 : -1\)/,
  );
});

// The three sorts read as synonyms by name alone, and Manual order is not a rule the list applies
// but one the user writes by dragging. Each says so under its name.
test("the sort menu says what each order does", () => {
  for (const key of [
    "priorityHint",
    "lastUpdatedHint",
    "manualOrderHint",
  ] as const) {
    assert.ok(EN.includes(`${key}:`), `${key} is missing from the en locale`);
    assert.ok(
      APP_SIDEBAR.includes(`shell.organize.${key}`),
      `${key} is never rendered`,
    );
  }
  assert.match(
    APP_SIDEBAR,
    /<span className="text-ui-12 leading-ui-16 text-muted-foreground">\n\s*\{t\(option\.hint\)\}/,
  );
});

// The bubble HugeIcons ships carries three dots inside it, which reads as speckle at row size and
// as "typing" next to a chat that is doing nothing.
test("a chat row's icon is the plain bubble, not the one with dots", () => {
  assert.ok(
    !APP_SIDEBAR.includes("BubbleChatIcon"),
    "the sidebar still renders the dotted bubble",
  );
  assert.equal(
    (APP_SIDEBAR.match(/icon=\{MessageCircleIcon\}/g) ?? []).length,
    // The pinned row's own icon. The read and unread items carry an eye instead.
    1,
    "a chat icon came or went",
  );
  // Derived by dropping that second path, so the icon set stays the source of the shape.
  assert.ok(
    DERIVED.includes("BubbleChatIcon.slice("),
    "the plain bubble is no longer derived from the dotted one",
  );
});

// A dragover can land in the same frame as the dragstart that began it. Reading the dragged row
// from state meant that first frame saw null, refused the drop, and a quick drag did nothing.
test("the dragged row is known before React re-renders", async () => {
  // Held outside React, so a dragover in the dragstart's own frame can still read it.
  const store = await readSrcAsync("features/chat/stores/sidebar-drag-source.ts");
  assert.match(store, /let source: SidebarDragSource \| null = null;/);
  assert.match(store, /export function setSidebarDragSource/);
  assert.match(store, /export function sidebarDragSource/);
  assert.match(
    APP_SIDEBAR,
    /setSidebarDragSource\(dragged\);\n\s*setDraggingRow\(dragged\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /setSidebarDragSource\(null\);\n\s*setDraggingRow\(null\);/,
  );
  // Every handler reads it from there; the state is only what paints.
  assert.ok(
    !/const dragged = draggingRow;/.test(APP_SIDEBAR),
    "a drag handler still reads the dragged row from state",
  );
  assert.equal(
    (APP_SIDEBAR.match(/const dragged = sidebarDragSource\(\);/g) ?? []).length,
    4,
    "a drag handler stopped reading the row it is carrying",
  );
});

// Pinning by dragging is the other half of dragging a chat into a folder.
test("a chat dropped into Pinned is pinned where it lands", () => {
  assert.match(
    APP_SIDEBAR,
    /const takesPin = \(dragged: SidebarDragSource \| null\) =>\n\s*Boolean\(config\.pinDrop\) &&\n\s*dragged\?\.kind === "chat" &&\n\s*!pinnedIdSet\.has\(dragged\.id\);/,
  );
  assert.match(
    APP_SIDEBAR,
    /setManualOrder\(\n\s*PINNED_ORDER_SCOPE,\n\s*placeIdAt\(pinnedRowIds, chatId, targetRowId, edge\),\n\s*\);/,
  );
  // Pinned's rows offer it, and so does the section, for the space under them.
  assert.match(APP_SIDEBAR, /pinDrop: true,/);
  assert.match(APP_SIDEBAR, /\{\.\.\.sectionPinDropProps\(\)\}/);
  // A chat already pinned is not pinned again.
  assert.match(APP_SIDEBAR, /if \(pinnedIdSet\.has\(chatId\)\) return;/);
});
