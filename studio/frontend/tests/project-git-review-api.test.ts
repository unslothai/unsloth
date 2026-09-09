// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  ProjectGitDiffManifest,
  ProjectGitStatus,
} from "../src/features/chat/api/project-git-review-api";

import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();

const gitApi = await import(
  "../src/features/chat/api/project-git-review-api.ts"
);

function statusResponse(
  overrides: Record<string, unknown> = {},
): ProjectGitStatus {
  return {
    version: 1,
    projectId: "project one",
    target: { kind: "primary" },
    workspaceRevision: 8,
    head: "1".repeat(40),
    branch: "main",
    sourceFingerprint: "2".repeat(64),
    fingerprintComplete: false,
    coherent: true,
    blockedReasons: [],
    counts: { staged: 1, unstaged: 2, untracked: 3, conflicted: 0 },
    files: [],
    ...overrides,
  } as ProjectGitStatus;
}

function diffResponse(
  overrides: Record<string, unknown> = {},
): ProjectGitDiffManifest {
  return {
    version: 1,
    projectId: "project one",
    target: { kind: "primary" },
    workspaceRevision: 8,
    mode: "staged",
    head: "1".repeat(40),
    sourceFingerprint: "2".repeat(64),
    fingerprintComplete: true,
    selectable: true,
    blockedReasons: [],
    conflictedPaths: [],
    files: [],
    fileCount: 0,
    hunkCount: 0,
    lineCount: 0,
    truncated: false,
    limits: {
      maxBytes: 512000,
      maxFiles: 5000,
      maxHunks: 20000,
      maxLines: 200000,
      maxLineChars: 64000,
      maxUntrackedFileBytes: 128000,
    },
    ...overrides,
  } as ProjectGitDiffManifest;
}

test.beforeEach(() => setAuthFetchHandler(null));

test("Git review client sends primary workspace reads with exact fencing", async () => {
  const calls: Array<{ input: string; cache: RequestCache | null }> = [];
  setAuthFetchHandler((input, init) => {
    calls.push({ input, cache: init?.cache ?? null });
    const body = input.endsWith("mode=staged")
      ? diffResponse()
      : statusResponse();
    return new Response(JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
    });
  });

  await gitApi.getProjectGitStatus("project one", 8);
  await gitApi.getProjectGitDiff("project one", 8, "staged");
  assert.deepEqual(calls, [
    {
      input: "/api/agent/projects/project%20one/git/status?workspaceRevision=8",
      cache: "no-store",
    },
    {
      input:
        "/api/agent/projects/project%20one/git/diff?workspaceRevision=8&mode=staged",
      cache: "no-store",
    },
  ]);
});

test("request guard rejects stale project, workspace, mode, and unmounted results", () => {
  const guard = new gitApi.ProjectGitReviewRequestGuard();
  guard.activate();
  const stale = guard.begin("project one", 8, "head");
  const current = guard.begin("project one", 8, "staged");
  const status = statusResponse();
  const diff = diffResponse();
  assert.equal(guard.acceptsToken(stale), false);
  assert.equal(guard.accepts(current, status, diff), true);

  for (const [nextStatus, nextDiff] of [
    [statusResponse({ projectId: "other" }), diff],
    [statusResponse({ workspaceRevision: 9 }), diff],
    [status, diffResponse({ projectId: "other" })],
    [status, diffResponse({ workspaceRevision: 9 })],
    [status, diffResponse({ mode: "head" })],
    [statusResponse({ target: { kind: "worktree" } }), diff],
  ] as [ProjectGitStatus, ProjectGitDiffManifest][]) {
    assert.equal(guard.accepts(current, nextStatus, nextDiff), false);
  }

  guard.retire();
  assert.equal(guard.accepts(current, status, diff), false);
  assert.equal(guard.acceptsToken(current), false);
});

test("Git review client maps backend conflicts without a snapshot", async () => {
  setAuthFetchHandler(
    () =>
      new Response(JSON.stringify({ detail: "Project workspace changed." }), {
        status: 409,
        headers: { "Content-Type": "application/json" },
      }),
  );
  await assert.rejects(
    gitApi.getProjectGitStatus("project", 3),
    /Project workspace changed/,
  );
});
