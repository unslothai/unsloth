// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (relative: string): string =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

const ACTIONS = read("../src/features/chat/utils/project-workspace-actions.ts");
const PROJECT_PAGE = read("../src/features/chat/chat-page.tsx");
const PROJECTS_LIST = read("../src/features/chat/projects-page.tsx");

test("the project page and the Projects list share one set of working-directory actions", () => {
  for (const [name, source] of [
    ["project page", PROJECT_PAGE],
    ["Projects list", PROJECTS_LIST],
  ] as const) {
    assert.ok(source.includes('from "./utils/project-workspace-actions"'), `${name} imports them`);
    for (const action of [
      "chooseProjectWorkspace",
      "switchToManagedWorkspace",
      "revealProjectWorkspace(",
    ]) {
      assert.ok(source.includes(action), `${name} offers ${action}`);
    }
    assert.ok(source.includes("Open project folder"), `${name} labels the reveal`);
    assert.ok(source.includes("Folder unavailable \u00b7 "), `${name} says when the folder is gone`);
  }
});

test("a project folder opens through its workspace session, never the stored path", () => {
  // The session is what the backend resolves to the folder in use now, and a
  // folder that has gone answers 410 there instead of opening something else.
  assert.ok(ACTIONS.includes("revealSandbox(project.workspaceSessionId)"));
  assert.ok(!/revealSandbox\([^)]*workspacePath/.test(ACTIONS));
});
