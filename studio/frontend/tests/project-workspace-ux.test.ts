// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { projectGoalViewState } from "../src/features/chat/components/project-goal-state.ts";
import {
  nextProjectLandingTab,
  projectFolderUnavailable,
} from "../src/features/chat/components/project-landing-state.ts";
import type { ProjectRecord } from "../src/features/chat/types.ts";

function project(overrides: Partial<ProjectRecord> = {}): ProjectRecord {
  return {
    id: "project-1",
    name: "Repository",
    instructions: "",
    archived: false,
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

test("project section keyboard navigation wraps and supports Home and End", () => {
  assert.equal(nextProjectLandingTab("chats", "ArrowRight"), "sources");
  assert.equal(nextProjectLandingTab("sources", "ArrowRight"), "agent");
  assert.equal(nextProjectLandingTab("agent", "ArrowRight"), "chats");
  assert.equal(nextProjectLandingTab("chats", "ArrowLeft"), "agent");
  assert.equal(nextProjectLandingTab("agent", "Home"), "chats");
  assert.equal(nextProjectLandingTab("chats", "End"), "agent");
  assert.equal(nextProjectLandingTab("sources", "Enter"), "sources");
});

test("only an unavailable local folder blocks new project chats", () => {
  assert.equal(
    projectFolderUnavailable(
      project({ workspaceKind: "folder", workspaceAvailable: false }),
    ),
    true,
  );
  assert.equal(
    projectFolderUnavailable(
      project({ workspaceKind: "folder", workspaceAvailable: true }),
    ),
    false,
  );
  assert.equal(
    projectFolderUnavailable(
      project({ workspaceKind: "managed", workspaceAvailable: false }),
    ),
    false,
  );
});

test("project goals stay discoverable before a goal has been set", () => {
  assert.deepEqual(projectGoalViewState(project()), {
    goal: "",
    status: null,
    label: "No project goal set",
    hint: "Set a durable objective here or from chat with `/goal set <text>`.",
  });
  assert.deepEqual(
    projectGoalViewState(
      project({ goal: "  Ship folder workspaces  ", goalStatus: "paused" }),
    ),
    {
      goal: "Ship folder workspaces",
      status: "paused",
      label: "Ship folder workspaces",
      hint: "Manage it here or from chat with `/goal help`.",
    },
  );
});

test("project landing exposes semantic tabs and path-free reconnect guidance", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  assert.match(page, /role="tablist"/);
  assert.match(page, /aria-selected=\{projectTab === "agent"\}/);
  assert.match(page, />\s*Agent workspace\s*</);
  assert.match(page, /Local project folder unavailable/);
  assert.match(page, /The repository remains untouched on disk/);
  assert.doesNotMatch(page, /workspaceError/);
});
