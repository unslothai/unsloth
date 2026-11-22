// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SIDEBAR = readSrc("components/app-sidebar.tsx");
const menu = SIDEBAR.slice(
  SIDEBAR.indexOf("function renderProjectRowMenuItems("),
  SIDEBAR.indexOf("function renderProjectFolderRow("),
);

test("Archive chats reads every chat in the project, not the sidebar's loaded page", () => {
  assert.match(
    menu,
    /await readAllSidebarRows\(sidebarRowQuery\(\{ projectId: project\.id \}\)\)/,
  );
  assert.match(menu, /await archiveChatItems\(all\);/);
  assert.match(menu, /onSelect=\{\(\) => void archiveProjectChats\(\)\}/);
  // A collapsed folder lists no rows, so gating on them would disable the item for a full project.
  assert.doesNotMatch(menu, /listedChatsByProjectId/);
  assert.doesNotMatch(menu, /disabled=\{projectChats/);
});

test("a failed read says so instead of archiving nothing silently", () => {
  assert.match(
    menu,
    /catch \(err\) \{\s*toast\.error\(translate\("settings\.data\.failedToArchiveChats"\)/,
  );
});
