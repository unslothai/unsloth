// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Desktop build can sit on an older backend that drops the create-time lease without spending it.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const DIALOG = await readSrcAsync(
  "features/chat/components/new-project-dialog.tsx",
);

test("a create that comes back without the folder applies the unspent lease", () => {
  const create = DIALOG.indexOf("await createChatProject(");
  const fallback = DIALOG.indexOf(
    "await setChatProjectWorkingFolder(project.id, workingFolder.token)",
  );
  const links = DIALOG.indexOf("await linkStagedFolders(project.id, folders)");
  assert.ok(create > 0 && fallback > create, "the fallback must follow the create");
  assert.ok(fallback < links, "the folder must be set before sources link");
  assert.match(
    DIALOG.slice(create, fallback),
    /if \(workingFolder && !project\.workingFolder\)/,
  );
});

test("a failed folder save asks for a new folder and re-enables the picker", () => {
  assert.match(
    DIALOG,
    /if \(workingFolder && !project\?\.workingFolder\) setNeedsReselect\(true\);/,
  );
  assert.match(
    DIALOG,
    /disabled=\{busy \|\| \(!!createdProject && !needsReselect\)\}\n\s*needsReselect=\{needsReselect\}/,
  );
});

// authFetch replays a POST after a network error; a lease-carrying create that committed then
// 400s on the spent grant, so the draft keeps one id and adopts what the first attempt saved.
test("a retried create reuses the draft's project id and adopts a committed attempt", () => {
  assert.match(DIALOG, /const projectId = useRef\(crypto\.randomUUID\(\)\);/);
  assert.match(
    DIALOG,
    /createChatProject\(\n\s*trimmed,\n\s*workingFolder\?\.token,\n\s*projectId\.current,\n\s*\)/,
  );
  const catchBlock = DIALOG.slice(DIALOG.indexOf("} catch (err) {"));
  const adopt = catchBlock.indexOf("await getChatProject(projectId.current)");
  const reselect = catchBlock.indexOf("setNeedsReselect(true)");
  assert.ok(adopt > 0 && adopt < reselect, "adopt the committed project before asking again");
  assert.match(DIALOG, /projectId\.current = crypto\.randomUUID\(\);/);
});
