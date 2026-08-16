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

test("picking one kind of row drops the other", async () => {
  // Chats and folders have no shared bulk action, so a mixed selection would
  // leave the menu unable to say what it acts on.
  const source = await sidebarSource();
  const chatClick = /function handleSelectionClick\(([\s\S]*?)\n  \}/.exec(
    source,
  );
  assert.ok(chatClick, "no handleSelectionClick");
  assert.match(chatClick[1], /setSelectedProjectIds/);

  const projectClick =
    /function handleProjectSelectionClick\(([\s\S]*?)\n  \}/.exec(source);
  assert.ok(projectClick, "no handleProjectSelectionClick");
  assert.match(projectClick[1], /setSelectedChatIds/);
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
});
