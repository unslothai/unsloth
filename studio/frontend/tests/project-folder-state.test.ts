// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MANAGED_FOLDER_LABEL,
  projectFolderState,
  shortenFolderLabel,
} from "../src/features/chat/utils/project-folder-state.ts";
import { readSrc } from "./helpers/kit.ts";

const MANAGED = "/home/me/Documents/Unsloth Studio/Projects/app-1234/sandbox";

test("a project without a chosen folder names the managed one", () => {
  const state = projectFolderState({ sandboxPath: MANAGED });
  assert.equal(state.chosen, false);
  assert.equal(state.unavailable, false);
  assert.equal(state.label, MANAGED_FOLDER_LABEL);
  assert.equal(state.path, MANAGED);
  assert.equal(state.problem, null);
});

test("a usable chosen folder shows its short path and branch", () => {
  const state = projectFolderState({
    sandboxPath: MANAGED,
    workingFolder: "/home/me/code/app",
    workingFolderStatus: "ok",
    workingFolderDisplay: "~/code/app",
    workingFolderBranch: "main",
  });
  assert.deepEqual(
    [state.chosen, state.unavailable, state.label, state.path, state.branch],
    [true, false, "~/code/app", "/home/me/code/app", "main"],
  );
  assert.equal(state.problem, null);
});

test("a backend that predates the status field is taken at its word", () => {
  const state = projectFolderState({ workingFolder: "/home/me/code/app" });
  assert.equal(state.unavailable, false);
  assert.equal(state.label, "/home/me/code/app");
});

test("a missing or swapped folder is flagged and drops its branch", () => {
  for (const status of ["missing", "replaced"] as const) {
    const state = projectFolderState({
      sandboxPath: MANAGED,
      workingFolder: "/home/me/code/app",
      workingFolderStatus: status,
      workingFolderDisplay: "~/code/app",
      workingFolderBranch: "main",
    });
    assert.equal(state.unavailable, true, status);
    assert.equal(state.branch, null, status);
    assert.match(state.problem ?? "", /Unsloth-managed folder/, status);
  }
});

test("a long path keeps its tail, cut at a separator", () => {
  assert.equal(shortenFolderLabel("~/code/app"), "~/code/app");
  const long =
    "~/work/clients/acme/services/billing/frontend/packages/dashboard";
  const short = shortenFolderLabel(long, 32);
  assert.ok(short.length <= 32, short);
  assert.match(short, /^…\//);
  assert.ok(long.endsWith(short.slice(1)));
});

test("the landing page and the edit dialog both read the live folder state", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /<ProjectFolderChip\n\s*project=\{currentProject\}\n\s*onChange=\{\(\) => setEditingProject\(true\)\}/,
  );
  const dialog = readSrc("features/chat/components/edit-project-dialog.tsx");
  assert.match(dialog, /const folder = projectFolderState\(saved\);/);
  assert.match(dialog, /detail=\{folder\}/);
  assert.match(dialog, /project\.workingFolderStatus \?\? ""/);
});

test("an unsent working-folder grant is dropped when it lapses", () => {
  const dialog = readSrc("features/chat/components/new-project-dialog.tsx");
  assert.match(dialog, /if \(sentLeases\.current\.has\(token\)\) return;/);
  assert.match(dialog, /expiresAt - EXPIRY_GRACE_MS - Date\.now\(\)/);
});

test("a failed project load is not shown as an empty list", () => {
  const hook = readSrc("features/chat/hooks/use-chat-projects.ts");
  assert.match(hook, /\} catch \(error\) \{[\s\S]{0,120}setLoadFailed\(true\)/);
  const switcher = readSrc("features/chat/components/project-switcher.tsx");
  assert.match(
    switcher,
    /loadFailed \? "Couldn't load projects" : "No projects yet"/,
  );
});

test("the web picks a working folder by path, and only when the backend allows it", () => {
  const picker = readSrc("features/chat/components/working-folder-picker.tsx");
  assert.match(picker, /const browse = useServerFolderPick\(!isTauri && isOwner\);/);
  assert.match(picker, /const available = native \|\| browse;/);
  assert.match(picker, /if \(choose && !native && !picked\) \{\n\s*setBrowsing\(true\);/);
  const api = readSrc("features/chat/api/chat-api.ts");
  assert.match(api, /body: JSON\.stringify\(\{ serverPath \}\)/);
  assert.match(api, /serverPath: body\?\.serverPath === true/);
  const create = readSrc("features/chat/components/new-project-dialog.tsx");
  assert.match(
    create,
    /if \(serverFolder\) \{\n\s*(?:try \{\n\s*)?project = await setChatProjectWorkingFolderPath\(/,
  );
});
