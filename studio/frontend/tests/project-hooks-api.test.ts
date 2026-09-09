// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  ProjectHookEvent,
  ProjectHookGroup,
  ProjectHooks,
  ProjectHooksTrustIdentity,
} from "../src/features/chat/api/project-hooks-api";

import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();

const hooksApi = await import("../src/features/chat/api/project-hooks-api.ts");

function hooksResponse(overrides: Record<string, unknown> = {}): Response {
  return new Response(
    JSON.stringify({
      sourcePath: ".codex/hooks.json",
      workspaceAvailable: true,
      workspaceRevision: 8,
      exists: true,
      contentHash:
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      description: "Project checks",
      groupCount: 0,
      handlerCount: 0,
      trusted: false,
      revision: 4,
      hooks: Object.fromEntries(
        hooksApi.PROJECT_HOOK_EVENTS.map((event) => [event, []]),
      ),
      ...overrides,
    }),
    { headers: { "Content-Type": "application/json" } },
  );
}

test.beforeEach(() => setAuthFetchHandler(null));

test("hooks client uses scoped reads and exact optimistic-concurrency bodies", async () => {
  const calls: Array<{
    input: string;
    method: string;
    cache: RequestCache | null;
    contentType: string | null;
    body: unknown;
  }> = [];
  setAuthFetchHandler((input, init) => {
    calls.push({
      input,
      method: init?.method ?? "GET",
      cache: init?.cache ?? null,
      contentType: new Headers(init?.headers).get("Content-Type"),
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    return hooksResponse();
  });

  await hooksApi.getProjectHooks("project one");
  await hooksApi.getProjectHooksTrustState("project one");
  await hooksApi.trustProjectHooks("project one", {
    workspaceRevision: 8,
    contentHash: "hash-one",
    revision: 3,
  });
  await hooksApi.revokeProjectHooks("project one", { revision: 4 });
  await hooksApi.setProjectHookHandlerEnabled("project one", "PreToolUse:0:0", {
    workspaceRevision: 9,
    contentHash: "hash-two",
    revision: 5,
    enabled: false,
  });

  assert.deepEqual(calls, [
    {
      input: "/api/agent/projects/project%20one/hooks",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input: "/api/agent/projects/project%20one/hooks/trust",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input: "/api/agent/projects/project%20one/hooks/trust",
      method: "POST",
      cache: null,
      contentType: "application/json",
      body: { workspaceRevision: 8, contentHash: "hash-one", revision: 3 },
    },
    {
      input: "/api/agent/projects/project%20one/hooks/revoke",
      method: "POST",
      cache: null,
      contentType: "application/json",
      body: { revision: 4 },
    },
    {
      input:
        "/api/agent/projects/project%20one/hooks/handlers/PreToolUse%3A0%3A0",
      method: "POST",
      cache: null,
      contentType: "application/json",
      body: {
        workspaceRevision: 9,
        contentHash: "hash-two",
        revision: 5,
        enabled: false,
      },
    },
  ]);
});

test("hooks client exposes all Codex hook events", () => {
  assert.deepEqual(hooksApi.PROJECT_HOOK_EVENTS, [
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "SubagentStart",
    "SubagentStop",
    "Stop",
  ]);
});

test("untrusted hook text visibly escapes slashes, newlines, controls, and bidi formatting", () => {
  assert.equal(
    hooksApi.visibleProjectHookText("literal\\n\r\nnext\t\u0001\u202Eend"),
    "literal\\\\n\\r\\n\nnext\\t\\u0001\\u202Eend",
  );
  assert.equal(hooksApi.visibleProjectHookText(null), "");
});

test("trust confirmation identity matches every captured field exactly", () => {
  const contentHash =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  const hooks: ProjectHooks = {
    workspaceRevision: 11,
    sourcePath: ".codex/hooks.json",
    exists: true,
    contentHash,
    description: null,
    groupCount: 0,
    handlerCount: 0,
    trusted: false,
    revision: 7,
    hooks: Object.fromEntries(
      hooksApi.PROJECT_HOOK_EVENTS.map((event) => [event, []]),
    ) as unknown as Record<ProjectHookEvent, ProjectHookGroup[]>,
  };
  const identity: ProjectHooksTrustIdentity = {
    projectId: "project-one",
    workspaceRevision: 11,
    contentHash,
    revision: hooks.revision,
  };

  assert.equal(
    hooksApi.projectHooksTrustIdentityMatches(identity, {
      projectId: "project-one",
      hooks,
    }),
    true,
  );
  for (const current of [
    { projectId: "project-two", hooks },
    {
      projectId: "project-one",
      hooks: { ...hooks, workspaceRevision: 12 },
    },
    {
      projectId: "project-one",
      hooks: { ...hooks, contentHash: "f".repeat(64) },
    },
    {
      projectId: "project-one",
      hooks: { ...hooks, revision: 8 },
    },
    {
      projectId: "project-one",
      hooks: { ...hooks, trusted: true },
    },
  ]) {
    assert.equal(
      hooksApi.projectHooksTrustIdentityMatches(identity, current),
      false,
    );
  }
});

test("request guard rejects stale responses and every response after unmount", () => {
  const guard = new hooksApi.ProjectHooksRequestGuard();
  guard.activate();
  const stale = guard.begin();
  const current = guard.begin();

  assert.equal(guard.accepts(stale), false);
  assert.equal(guard.accepts(current), true);

  guard.retire();
  assert.equal(guard.accepts(current), false);
  assert.equal(guard.accepts(guard.begin()), false);

  guard.activate();
  const remounted = guard.begin();
  assert.equal(guard.accepts(remounted), true);
});

test("hooks client maps backend errors without returning a snapshot", async () => {
  setAuthFetchHandler(
    () =>
      new Response(JSON.stringify({ detail: "Hooks revision changed." }), {
        status: 409,
        headers: { "Content-Type": "application/json" },
      }),
  );

  await assert.rejects(
    hooksApi.revokeProjectHooks("project", { revision: 2 }),
    /Hooks revision changed/,
  );
});
