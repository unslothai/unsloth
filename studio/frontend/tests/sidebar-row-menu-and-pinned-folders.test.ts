// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a sidebar row gets wrong when the same chat or folder is on screen twice, or when its
// menu is open rather than merely hovered.

import assert from "node:assert/strict";
import test from "node:test";
import {
  useChatNavigationStore,
  visibleChatItems,
} from "../src/features/chat/stores/chat-navigation-store.ts";
import { applyManualOrder } from "../src/features/chat/stores/sidebar-organization-store.ts";
import type { SidebarItem } from "../src/features/chat/hooks/use-chat-sidebar-items.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");

// The open-state gutter rule is more specific than the hover one, so it wins when both match:
// a hovered row with its menu open laid out with the narrower gutter while hover had revealed
// every action. Measured in a browser: the old pr-8 left the title 36px past the secondary
// action's left edge, pr-16 leaves it 4px clear.
test("an open row menu reserves the same gutter hover does", () => {
  const pattern =
    /group-hover\/([a-z-]+):pr-(\d+)[^"]*group-has-\[\.sidebar-row-action\[data-state=open\]\]\/([a-z-]+):pr-(\d+)/g;
  const seen: string[] = [];
  for (const [, hoverGroup, hoverPad, openGroup, openPad] of APP_SIDEBAR.matchAll(
    pattern,
  )) {
    assert.equal(
      hoverGroup,
      openGroup,
      "the hover and open-state gutters are written for different rows",
    );
    assert.equal(
      openPad,
      hoverPad,
      `${openGroup} reserves pr-${hoverPad} on hover but pr-${openPad} with its menu open, so the title runs under the actions`,
    );
    seen.push(`${openGroup}:${openPad}`);
  }
  // Project chat rows, every row of Pinned and Recents, and the folder rows. The chat rows'
  // branches collapsed into one when the pin became a fixture of every row rather than of the
  // pinned ones, so this counts kinds of row, not branches.
  assert.ok(seen.length >= 3, `only ${seen.length} rows carry both gutters`);
});

// The kebab reveals itself while its menu is open, but the quick-action beside it was
// hover-only, so it slid in over the title even once the gutter was right.
test("the quick-action beside an open menu is revealed with it", () => {
  for (const group of ["project-chat-item", "recent-item"]) {
    const reveal = new RegExp(
      `group-has-\\[\\.sidebar-row-action\\[data-state=open\\]\\]/${group}:opacity-100 group-has-\\[\\.sidebar-row-action\\[data-state=open\\]\\]/${group}:pointer-events-auto`,
    );
    assert.match(APP_SIDEBAR, reveal, `${group} has no open-menu reveal`);
  }
  // Every secondary action carries it: the pin, the unpin and the folder's New chat.
  const secondaries = APP_SIDEBAR.match(/is-unpin-action/g) ?? [];
  assert.equal(secondaries.length, 3, "a secondary row action came or went");
  assert.equal(
    (APP_SIDEBAR.match(/REVEAL_WITH_OPEN_MENU_(PROJECT_CHAT|RECENT),/g) ?? [])
      .length,
    3,
    "a secondary row action is not revealed with its row's menu",
  );
});

// A pinned project chat is two rows at once. Both matched on the chat id, so both mounted an
// autofocused input: the second stole focus and the first's blur cancelled the rename.
test("the rename pill belongs to the row its menu was opened from", () => {
  assert.match(
    APP_SIDEBAR,
    /renamingTarget\.item\.id === item\.id &&\n(?:\s*\/\/[^\n]*\n)*\s*renamingTarget\.rowScope === list\?\.scope;/,
  );
  assert.match(
    APP_SIDEBAR,
    /function openRenameChat\(item: SidebarItem, inline = true, rowScope\?: string\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /onSelect=\{\(\) => openRenameChat\(item, true, list\?\.scope\)\}/,
  );
  // The chord has no row under the cursor, so it opens the dialog and names no scope.
  assert.match(APP_SIDEBAR, /openRenameChat\(item, false\)/);
});

// The row toggles its chats, so the folder says which way it is.
test("a folder row's icon follows its disclosure", () => {
  assert.match(APP_SIDEBAR, /icon=\{expanded \? Folder02Icon : Folder01Icon\}/);
});

// An open folder with no chats reads as one that failed to expand, so it says so.
test("an empty open folder says it is empty", () => {
  assert.match(
    APP_SIDEBAR,
    /\{expanded && projectChats\.length === 0 && \(\n\s*<SidebarMenuItem\n\s*\{\.\.\.dnd\.dropZoneProps\(\{ section: order\.section, folderId: project\.id, blockEnd \}\)\}\n\s*>\n\s*<p className="[^"]*text-nav-fg-muted">\n\s*\{t\("shell\.navigation\.noChats"\)\}/,
  );
  // And it is a row, so the bottom fade has to count it like the "Show more" one.
  assert.match(APP_SIDEBAR, /if \(chats\.length === 0\) rows \+= 1;/);
});

// The submenu holds actions and destinations. The rule separates them, actions above.
test("the Project submenu puts its actions above the destinations", () => {
  const sub = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("<span>Project</span>"),
    APP_SIDEBAR.indexOf("<span>Export</span>"),
  );
  assert.ok(sub.length > 0, "the Project submenu moved");
  const newProject = sub.indexOf("<span>New project</span>");
  const sources = sub.indexOf("<span>Project sources</span>");
  const rule = sub.indexOf("<DropdownMenuSeparator />");
  const recents = sub.indexOf("<span>Recents</span>");
  for (const [name, at] of Object.entries({
    newProject,
    sources,
    rule,
    recents,
  })) {
    assert.notEqual(at, -1, `${name} is gone from the Project submenu`);
  }
  assert.ok(newProject < sources, "Project sources is not under New project");
  assert.ok(sources < rule, "Project sources fell below the rule");
  assert.ok(rule < recents, "the destinations are not under the rule");
  // The old wording is gone.
  assert.ok(!APP_SIDEBAR.includes("<span>Move to project</span>"));
  assert.ok(!APP_SIDEBAR.includes("<span>Save to project sources</span>"));
});

// The folder menu: pin beside Project home, Edit for the dialog that owns the name, and no
// New chat, which the row's own pencil already does.
test("the folder menu leads with where to go, then what to change", async () => {
  const menu = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("<span>Project home</span>"),
    APP_SIDEBAR.indexOf("<span>Delete project</span>"),
  );
  assert.ok(menu.length > 0, "the folder menu moved");
  const pin = menu.indexOf('{isProjectPinned ? "Unpin" : "Pin"}');
  const edit = menu.indexOf("<span>Edit</span>");
  assert.notEqual(pin, -1, "the pin toggle left the folder menu");
  assert.notEqual(edit, -1, "Edit is not in the folder menu");
  assert.ok(pin < edit, "Pin project is not directly under Project home");
  assert.ok(
    !menu.includes("<span>New chat</span>"),
    "New chat is still in the folder menu",
  );
  assert.ok(!APP_SIDEBAR.includes("<span>Rename project</span>"));
  // Edit opens the dialog, and the dialog is mounted with a delete that reuses the confirmation.
  assert.match(APP_SIDEBAR, /onSelect=\{\(\) => setEditingProject\(project\)\}/);
  assert.match(
    APP_SIDEBAR,
    /<EditProjectDialog\n\s*project=\{editingProject\}/,
  );
  assert.match(
    APP_SIDEBAR,
    /onDelete=\{\(project\) => openDeleteDialog\(\{ kind: "project", project \}\)\}/,
  );
  // And the rename dialog no longer carries a project branch nothing can reach.
  assert.ok(!APP_SIDEBAR.includes('renamingTarget?.kind === "project"'));

  const dialog = await readSrcAsync(
    "features/chat/components/edit-project-dialog.tsx",
  );
  // Name, instructions and the linked folders, each writing through its own call.
  assert.match(dialog, /renameChatProject\(target\.id, trimmedName\)/);
  assert.match(
    dialog,
    /updateChatProjectInstructions\(target\.id, trimmedInstructions\)/,
  );
  assert.match(dialog, /scope=\{\{ type: "project", id: project\.id \}\}/);
});

// Pinning a folder used to sort it to the top of Projects, beside a Pinned section of chats.
test("a pinned folder is a row of Pinned, not of Projects", () => {
  // Pinned renders the folders among its chats, in the Pinned section's own order scope.
  assert.match(
    APP_SIDEBAR,
    /row\.kind === "project"\n\s*\? renderProjectFolderRow\(row\.project, \{\n\s*scope: PINNED_ORDER_SCOPE,\n\s*orderedIds: pinnedRowIds,/,
  );
  // Projects renders what is left, in its own.
  assert.match(
    APP_SIDEBAR,
    /visibleProjectRecords\.map\(\(project\) =>\n\s*renderProjectFolderRow\(project, \{\n\s*scope: PROJECT_ORDER_SCOPE,\n\s*orderedIds: projectRowIds,/,
  );
  // And the Projects list no longer carries them, so a folder is never in both sections.
  const records = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("const sidebarProjectRecords = useMemo("),
    APP_SIDEBAR.indexOf("const visibleProjectRecords"),
  );
  assert.match(records, /\.filter\(\(p\) => !pinnedProjectIdSet\.has\(p\.id\)\)/);
  assert.ok(
    !records.includes("pinnedProjectIds"),
    "the Projects list still folds the pinned folders in",
  );
  // The header owns "New project", so its test counts every project, pinned or not.
  assert.match(
    APP_SIDEBAR,
    /const projectsSectionConfigured =\n\s*organizeBy === "project" && projects\.length > 0;/,
  );
  assert.match(
    APP_SIDEBAR,
    /const projectsSectionRendered =\n[\s\S]{0,200}?projectsSectionConfigured;/,
  );
  assert.match(APP_SIDEBAR, /\{projectsSectionRendered && \(/);
});

// Moving a folder into Pinned must not undo an order the user had already dragged it into: the
// Pinned scope is empty on the first run, while the Projects one still holds that order.
test("a pinned folder keeps the order it was dragged into", () => {
  const projects = [{ id: "a" }, { id: "b" }, { id: "c" }];
  const byId = new Map(projects.map((p) => [p.id, p]));
  // Pinned b first, so pin order alone would draw b above a.
  const pinned = ["b", "a"].map((id) => byId.get(id)!);
  const draggedInProjects = ["c", "a", "b"];
  const order = (pinnedScope: string[] | undefined) =>
    applyManualOrder(
      pinned,
      pinnedScope?.length ? pinnedScope : draggedInProjects,
      (project) => project.id,
    ).map((project) => project.id);
  assert.deepEqual(order(undefined), ["a", "b"]);
  // And a drop in Pinned takes over from then on.
  assert.deepEqual(order(["b", "a"]), ["b", "a"]);
  assert.match(
    APP_SIDEBAR,
    /manualOrder\[PINNED_PROJECT_ORDER_SCOPE\]\?\.length\n\s*\? manualOrder\[PINNED_PROJECT_ORDER_SCOPE\]\n\s*: manualOrder\[PROJECT_ORDER_SCOPE\],/,
  );
});

// Pinned draws its folders above its chats, and the chords walk what the store was handed.
test("the walk reads the rows in the order Pinned draws them", () => {
  const item = (id: string): SidebarItem => ({
    type: "single",
    id,
    title: id,
    createdAt: 0,
    updatedAt: 0,
  });
  const folderChats = [item("folder-1"), item("folder-2")];
  // folder-2 is pinned as well as being in a pinned folder, so it is drawn twice.
  const pinnedChats = [item("pin-1"), item("folder-2")];
  useChatNavigationStore.getState().publishLists({
    pinnedItems: [...folderChats, ...pinnedChats],
    projectItems: [item("proj-1")],
    recentItems: [item("recent-1")],
    attentionItemIds: [],
    activeItemId: null,
  });
  assert.deepEqual(
    visibleChatItems(useChatNavigationStore.getState()).map((i) => i.id),
    ["folder-1", "folder-2", "pin-1", "proj-1", "recent-1"],
  );
  // The sidebar publishes Pinned in its drawn order, folders and chats interleaved with each open
  // folder's chats under its row, and leaves the section's own chats to projectItems.
  assert.match(
    APP_SIDEBAR,
    /const pinnedSectionChatItems = useMemo\(\n\s*\(\) =>\n\s*chatListsOnScreen && pinnedOpen\n\s*\? pinnedRows\.flatMap\(\(row\) =>\n\s*row\.kind === "project"\n\s*\? folderChatItems\(true, \[row\.project\]\)\n\s*: \[row\.item\],/,
  );
  assert.match(
    APP_SIDEBAR,
    /pinnedItems: pinnedSectionChatItems,\n\s*projectItems: sectionProjectChatItems,/,
  );
});
