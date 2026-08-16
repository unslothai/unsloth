// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// A row is only selectable if it is handed the list it belongs to. Dropping
// that argument still compiles, since it is optional, and the row just stops
// responding to cmd and shift click.

async function sidebarSource(): Promise<string> {
  return readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );
}

test("every chat list hands its rows a selection list", async () => {
  const source = await sidebarSource();
  for (const list of [
    /\{ scope: PINNED_ORDER_SCOPE, ids: pinnedRowIds \}/,
    /\{ scope: RECENTS_ORDER_SCOPE, ids: recentRowIds \}/,
    /scope: projectOrderScope\(project\.id\),\s*ids: projectChatIds,/,
  ]) {
    assert.match(source, list);
  }
});

test("folder rows select too, and open their own bulk menu", async () => {
  const source = await sidebarSource();
  assert.match(source, /handleProjectSelectionClick\(event, project\.id\)/);
  assert.match(source, /selectProjectForContextMenu\(project\.id\)/);
  assert.match(source, /\{renderProjectContextMenu\(\)\}/);
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
  // Chats and folders have no shared bulk action, so a mixed selection would
  // leave the menu unable to say what it acts on. All four entry points, since
  // one that skips it is what puts the sidebar in that state.
  const source = await sidebarSource();
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
  // Both menus return early when the row is already selected. Dropping after
  // that return would keep a mixed selection alive for exactly the rows a bulk
  // action is most likely to run on.
  const source = await sidebarSource();
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
  // A kept anchor shift-selects a range from a row that no longer looks
  // selected, which is how a cleared list grows again on the next click.
  const source = await sidebarSource();
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
  // Its wording already exists as a key, so a literal here would be the one
  // English toast in an otherwise translated flow.
  const source = await sidebarSource();
  const archive = /async function archiveSelected\(([\s\S]*?)\n  \}/.exec(source);
  assert.ok(archive, "no archiveSelected");
  assert.match(archive[1], /translate\("settings\.data\.failedToArchiveChats"\)/);
});

test("one failed archive does not abandon the rest of the batch", async () => {
  // The selection is cleared up front, so chats skipped by an early exit are
  // left unarchived with nothing left highlighted to retry from. The other two
  // bulk loops catch per item; this one has to as well.
  const source = await sidebarSource();
  const archive = /async function archiveSelected\(([\s\S]*?)\n  \}/.exec(source);
  assert.ok(archive, "no archiveSelected");
  const body = archive[1];
  const loopAt = body.indexOf("for (const item of items)");
  const tryAt = body.indexOf("try {");
  assert.ok(loopAt >= 0, "no batch loop");
  assert.ok(tryAt > loopAt, "archiveSelected catches around the loop, not in it");
  // Reported on what got through, not on whether the loop threw.
  assert.match(body, /archived \+= 1/);
  assert.match(body, /if \(archived > 0\) showArchivedChatsToast\(\)/);
  assert.match(body, /if \(archived < items\.length\)/);
});

test("deleting folders in bulk cleans up like deleting one", async () => {
  // Both branches end the same way, or a batch leaves stale chat rows behind
  // and strands the user on a page whose project is gone.
  const source = await sidebarSource();
  const branch = /if \(target\.kind === "projects"\) \{([\s\S]*?)\n      return;/.exec(
    source,
  );
  assert.ok(branch, "no bulk project delete branch");
  assert.match(branch[1], /notifyChatHistoryUpdated\(\)/);
  assert.match(branch[1], /setActiveProjectId\(null\)/);
  assert.match(branch[1], /navigate\(\{ to: "\/chat"/);

  // The redirect reads what was actually deleted. Built from the requested
  // list instead, a delete that failed would still throw the user off a
  // project that is still there.
  assert.match(branch[1], /deletedIds\.add\(project\.id\)/);
  assert.equal(
    /new Set\(target\.projects\.map/.test(branch[1]),
    false,
    "deletedIds is built from the requested projects, not the deleted ones",
  );
});

test("both sidebar expanders read translated labels", async () => {
  // The two sit one control apart, so an English literal next to a translated
  // twin is the visible half of the omission.
  // Comments name the control too, so match the rendered ternary, not the words.
  const source = await sidebarSource();
  assert.equal(
    /\?\s*"Show less"\s*:\s*"Show more"/.test(source),
    false,
    "a sidebar expander still hard-codes its label",
  );
  const uses = source.match(/shell\.navigation\.show(More|Less)/g) ?? [];
  assert.equal(uses.length, 4, "both expanders read both keys");
});
