// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const APP_SIDEBAR = readSrc("components/app-sidebar.tsx");

// The selection list also carries drag-reorder ids, so losing it breaks both.

test("every chat list hands its rows a selection list", async () => {
  const source = APP_SIDEBAR;
  for (const list of [
    /scope: PINNED_ORDER_SCOPE,\s*ids: pinnedChatRowIds,\s*orderIds: pinnedRowIds,/,
    /scope: RECENTS_ORDER_SCOPE,\s*ids: recentRowIds,/,
    /scope: projectOrderScope\(project\.id\),\s*ids: projectChatIds,/,
  ]) {
    assert.match(source, list);
  }
});

test("folder rows select too, and open their own bulk menu", async () => {
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /handleProjectSelectionClick\(\n\s*event,\n\s*project\.id,\n\s*order\.selectionIds \?\? order\.orderedIds,\n\s*\)/,
  );
  assert.match(source, /selectProjectForContextMenu\(project\.id\)/);
  assert.match(source, /\{renderProjectContextMenu\(project\)\}/);
  assert.match(source, /selectedProjectIds\.has\(project\.id\)/);
});

function bodyOf(source: string, name: string): string {
  const found = new RegExp(`function ${name}\\(([\\s\\S]*?)\\n  \\}`).exec(
    source,
  );
  assert.ok(found, `no ${name}`);
  return found[1];
}

test("picking one kind of row drops the other", async () => {
  // Chats and folders share no bulk action, so all four entry points must drop the mix.
  const source = APP_SIDEBAR;
  for (const [name, drop] of [
    ["handleSelectionClick", "dropProjectSelection()"],
    ["selectForContextMenu", "dropProjectSelection()"],
    ["handleProjectSelectionClick", "dropChatSelection()"],
    ["selectProjectForContextMenu", "dropChatSelection()"],
  ]) {
    assert.ok(
      bodyOf(source, name).includes(drop),
      `${name} leaves the other kind of row selected`,
    );
  }
});

test("a right-click drops the other kind even on an already-selected row", async () => {
  // Drop before the already-selected early return, or a mixed selection survives.
  const source = APP_SIDEBAR;
  for (const [name, drop] of [
    ["selectForContextMenu", "dropProjectSelection()"],
    ["selectProjectForContextMenu", "dropChatSelection()"],
  ]) {
    const body = bodyOf(source, name);
    const dropAt = body.indexOf(drop);
    const returnAt = body.search(/if \(selected\w+\.has\([\w.]+\)\) return;/);
    assert.ok(dropAt >= 0, `${name} does not drop the other kind`);
    assert.ok(returnAt >= 0, `${name} lost its early return`);
    assert.ok(
      dropAt < returnAt,
      `${name} drops the other kind only after its early return`,
    );
  }
});

test("dropping a selection clears its anchor too", async () => {
  // A kept anchor lets a cleared list regrow on the next shift-click.
  const source = APP_SIDEBAR;
  assert.match(
    source,
    /const dropChatSelection = useCallback\(\(\) => \{\s*selectionAnchorRef\.current = null;/,
  );
  assert.match(
    source,
    /const dropProjectSelection = useCallback\(\(\) => \{\s*projectAnchorRef\.current = null;/,
  );
});

test("the bulk archive failure reads a translated string", async () => {
  const source = APP_SIDEBAR;
  const archive = /async function archiveChatItems\(([\s\S]*?)\n  \}/.exec(source);
  assert.ok(archive, "no archiveChatItems");
  assert.match(archive[1], /translate\("settings\.data\.failedToArchiveChats"\)/);
});

test("one failed archive does not abandon the rest of the batch", async () => {
  // The selection is cleared up front, so the loop must catch per item.
  const source = APP_SIDEBAR;
  assert.match(source, /async function archiveSelected\(\) \{\s*const items = selectedChatItems;\s*clearSelection\(\);\s*await archiveChatItems\(items\);/);
  const archive = /async function archiveChatItems\(([\s\S]*?)\n  \}/.exec(source);
  assert.ok(archive, "no archiveChatItems");
  const body = archive[1];
  const loopAt = body.indexOf("for (const item of items)");
  const tryAt = body.indexOf("try {");
  assert.ok(loopAt >= 0, "no batch loop");
  assert.ok(tryAt > loopAt, "archiveChatItems catches around the loop, not in it");
  assert.match(body, /archived \+= 1/);
  assert.match(body, /if \(archived > 0\) showArchivedChatsToast\(\)/);
  assert.match(body, /if \(archived < items\.length\)/);
});

test("deleting folders in bulk cleans up like deleting one", async () => {
  const source = APP_SIDEBAR;
  const branch = /if \(target\.kind === "projects"\) \{([\s\S]*?)\n      return;/.exec(
    source,
  );
  assert.ok(branch, "no bulk project delete branch");
  assert.match(branch[1], /notifyChatHistoryUpdated\(\)/);
  assert.match(branch[1], /setActiveProjectId\(null\)/);
  assert.match(branch[1], /navigate\(\{ to: "\/chat"/);

  // The redirect reads what was actually deleted, not what was requested.
  assert.match(branch[1], /deletedIds\.add\(project\.id\)/);
  assert.equal(
    /new Set\(target\.projects\.map/.test(branch[1]),
    false,
    "deletedIds is built from the requested projects, not the deleted ones",
  );
});

test("both sidebar expanders read translated labels", async () => {
  // Comments name the control too, so match the rendered ternary, not the words.
  const source = APP_SIDEBAR;
  assert.equal(
    /\?\s*"Show less"\s*:\s*"Show more"/.test(source),
    false,
    "a sidebar expander still hard-codes its label",
  );
  const uses = source.match(/shell\.navigation\.show(More|Less)/g) ?? [];
  assert.equal(uses.length, 4, "both expanders read both keys");
});


test("a row's right-click menu and its 3-dot menu render the same items", () => {
  for (const [kind, render] of [
    ["chat", "renderChatRowMenuItems"],
    ["project", "renderProjectRowMenuItems"],
  ] as const) {
    const calls = APP_SIDEBAR.match(new RegExp(`${render}\\(`, "g")) ?? [];
    assert.equal(
      calls.length,
      3,
      `${kind} rows should build their menu once and render it in both places`,
    );
    assert.match(
      APP_SIDEBAR,
      new RegExp(`${render}\\([^)]*DROPDOWN_ROW_MENU\\)`),
      `${kind}: the 3-dot menu should render the shared items`,
    );
    assert.match(
      APP_SIDEBAR,
      new RegExp(`${render}\\([^)]*CONTEXT_ROW_MENU\\)`),
      `${kind}: the right-click menu should render the same shared items`,
    );
  }
});

test("the bulk menu is reached only by a selection of more than one", () => {
  for (const [name, count] of [
    ["renderChatContextMenu", "selectionCount"],
    ["renderProjectContextMenu", "projectSelectionCount"],
  ] as const) {
    const body = bodyOf(APP_SIDEBAR, name);
    assert.match(
      body,
      new RegExp(`if \\(${count} <= 1\\) \\{`),
      `${name} should send a single row to the row menu`,
    );
    const guard = body.indexOf(`${count} <= 1`);
    const bulk = body.indexOf("shell.selection.");
    assert.ok(guard >= 0 && bulk > guard, `${name}: the guard should precede the bulk items`);
  }
});

test("the shared menu items are written against the injected family", () => {
  for (const name of ["renderChatRowMenuItems", "renderProjectRowMenuItems"]) {
    const body = bodyOf(APP_SIDEBAR, name);
    assert.doesNotMatch(
      body,
      /<DropdownMenu|<ContextMenu/,
      `${name} should use P.Item and friends, not one family's components`,
    );
    assert.match(body, /<P\.Item/);
  }
  assert.doesNotMatch(APP_SIDEBAR, /OpenChatFolder/);
});


// Scoped to .sidebar-row-menu: .unsloth-plus-menu alone dresses many other surfaces.

test("every row-menu surface takes the compact class", () => {
  const surfaces = APP_SIDEBAR.match(/className="unsloth-plus-menu[^"]*"/g) ?? [];
  const rowMenus = surfaces.filter((c) => c.includes("menu-flat-destructive"));
  assert.ok(rowMenus.length >= 6, "expected both 3-dot menus and both right-click menus");
  for (const cls of rowMenus) {
    assert.match(cls, /sidebar-row-menu/, `a row menu surface missed the class: ${cls}`);
  }
  // Sub-content is portaled out of the menu, so it carries the class itself.
  for (const sub of APP_SIDEBAR.match(/<P\.SubContent[^>]*className="[^"]*"/g) ?? []) {
    assert.match(sub, /sidebar-row-menu/, `a submenu missed the class: ${sub}`);
  }
});

test("the compact class is scoped, and both menu families get the same rules", () => {
  const CSS = readSrc("index.css");
  assert.match(CSS, /\.unsloth-plus-menu\.sidebar-row-menu\[data-slot\] \{/);
  assert.match(CSS, /--icon-size: var\(--ui-icon-size-sm\);/);
  // The base rule pins svg size with !important off --ui-icon-size.
  const svgRules = CSS.split("\n\t.unsloth-plus-menu").filter((r) =>
    r.includes("svg:not(.unsloth-tick)"),
  );
  assert.equal(svgRules.length, 2, "expected a base svg rule and a row-menu override");
  const scoped = svgRules.find((r) => r.startsWith(".sidebar-row-menu"));
  assert.ok(scoped, "the row menu needs its own svg size rule");
  assert.match(scoped, /width: var\(--icon-size\) !important;/);
  assert.match(scoped, /height: var\(--icon-size\) !important;/);
  const plusRules = CSS.split("\n\t.unsloth-plus-menu").slice(1);
  for (const rule of plusRules) {
    const head = rule.slice(0, rule.indexOf("{"));
    if (!head.includes('dropdown-menu-item')) continue;
    assert.match(
      head,
      /context-menu-item/,
      `a plus-menu item rule styles only the dropdown family: ${head.trim()}`,
    );
  }
});


// Inline submenus are clipped by the parent's animated transform, so both portal out.

test("both menu families portal their submenus out of the parent", () => {
  for (const [file, primitive] of [
    ["components/ui/dropdown-menu.tsx", "DropdownMenuPrimitive"],
    ["components/ui/context-menu.tsx", "ContextMenuPrimitive"],
  ] as const) {
    const source = readSrc(file);
    const at = source.indexOf(`<${primitive}.SubContent`);
    assert.ok(at > 0, `${file}: no SubContent element`);
    const before = source.slice(0, at);
    assert.match(
      before.slice(-400),
      new RegExp(`<${primitive}\\.Portal>`),
      `${file}: SubContent should be wrapped in a Portal`,
    );
  }
});
