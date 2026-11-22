// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Grants are single-use and Desktop authFetch replays after network errors; behaviour is driven
// end to end by tests/studio/playwright_project_folders.py, these pin the shape.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const DIALOG = await readSrcAsync(
  "features/chat/components/new-project-dialog.tsx",
);
const BODY = DIALOG.slice(DIALOG.indexOf("async function commitCreate()"));
const TRY = BODY.slice(0, BODY.indexOf("} catch (err) {"));
const CATCH = BODY.slice(BODY.indexOf("} catch (err) {"));

test("a retried create reuses the draft's project id", () => {
  assert.match(DIALOG, /const projectId = useRef\(crypto\.randomUUID\(\)\);/);
  assert.match(TRY, /createChatProject\(trimmed, token \?\? undefined, projectId\.current\)/);
  assert.match(DIALOG, /projectId\.current = crypto\.randomUUID\(\);/);
});

test("every submit reconciles the saved folder with the current choice", () => {
  const create = TRY.indexOf("await createChatProject(");
  const put = TRY.indexOf("await setChatProjectWorkingFolder(project.id, token)");
  const clear = TRY.indexOf("await setChatProjectWorkingFolder(project.id, null)");
  const links = TRY.indexOf("await linkStagedFolders(project.id, folders)");
  assert.ok(create > 0 && create < put && put < clear && clear < links);
  // Outside the `if (!project)` create branch, so a reused project is reconciled too.
  assert.match(TRY, /\}\n\s*\/\/[^\n]*\n\s*if \(token && appliedLease\.current !== token\) \{/);
  assert.match(TRY, /\} else if \(!token && project\.workingFolder\) \{/);
  assert.match(TRY, /if \(token && project\.workingFolder\) appliedLease\.current = token;/);
});

test("a failure re-reads the project and trusts its folder only when one grant was ever sent", () => {
  const read = CATCH.indexOf("await getChatProject(projectId.current)");
  const reselect = CATCH.indexOf("setNeedsReselect(true)");
  assert.ok(read > 0 && read < reselect);
  assert.ok(CATCH.indexOf("notifyChatProjectsUpdated()") < reselect, "an adopted project is published");
  assert.match(CATCH, /saved\.workingFolder && sentLeases\.current\.size === 1/);
  assert.match(CATCH, /if \(appliedLease\.current !== token\) setNeedsReselect\(true\);/);
  assert.match(DIALOG, /sentLeases\.current = new Set\(\);\n\s*appliedLease\.current = null;/);
});

test("a failed folder save re-enables the picker", () => {
  assert.match(
    DIALOG,
    /disabled=\{busy \|\| \(!!createdProject && !needsReselect\)\}\n\s*needsReselect=\{needsReselect\}/,
  );
});
