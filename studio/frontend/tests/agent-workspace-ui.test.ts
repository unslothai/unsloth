// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  agentBackgroundAgentRequest,
  agentWorkspaceJsonRequest,
  agentWorkspaceRequestPath,
  agentWorktreeMergeRequest,
} from "../src/features/chat/api/agent-workspace-requests.ts";
import {
  BACKGROUND_AGENT_FULL_ACCESS_WARNING,
  BACKGROUND_AGENT_PERMISSION_POLICY,
  agentBackgroundActions,
  agentBackgroundSnapshot,
  agentPlanProgress,
  agentRepositoryMapSummary,
  agentStatusLabel,
  agentWorkspaceRequestIsCurrent,
  agentWorkspaceStatus,
  agentWorktreeMergeAction,
  backgroundAgentPermissionNeedsConfirmation,
  backgroundAgentPermissionRunsUnattended,
  latestVerificationSummary,
  reconcileAgentBackgroundMutation,
  safeAgentWorkspaceError,
} from "../src/features/chat/components/agent-workspace-state.ts";

test("agent workspace requests encode project identity and bounded queries", () => {
  assert.equal(
    agentWorkspaceRequestPath(" project/with spaces ", "repository-map", {
      max_paths: 20_000,
      max_total_bytes: 2_097_152,
      ignored: undefined,
    }),
    "/api/agent-workspace/projects/project%2Fwith%20spaces/repository-map?max_paths=20000&max_total_bytes=2097152",
  );
  assert.equal(
    agentWorkspaceRequestPath("project-1", "/git/diff/", {
      staged: false,
    }),
    "/api/agent-workspace/projects/project-1/git/diff?staged=false",
  );
  assert.throws(() => agentWorkspaceRequestPath("  ", "workspace"));
});

test("agent workspace mutations use explicit JSON requests", () => {
  assert.deepEqual(agentWorkspaceJsonRequest("POST"), { method: "POST" });
  assert.deepEqual(
    agentWorkspaceJsonRequest("PATCH", { status: "completed" }),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: '{"status":"completed"}',
    },
  );
  assert.deepEqual(
    agentWorkspaceJsonRequest("PUT", {
      checks: [],
      requireForGoalCompletion: true,
    }),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: '{"checks":[],"requireForGoalCompletion":true}',
    },
  );
});

test("background agent and guarded merge requests match the backend contract", () => {
  const queued = agentBackgroundAgentRequest("project-1", {
    instruction: "  fix the failing test  ",
    runtime: {
      kind: "provider",
      model: "gpt-5.6-codex",
      providerId: "codex-account",
      permissionMode: "full",
      reasoningEffort: " high ",
      maxOutputTokens: 12_000,
    },
    planId: "plan-1",
    planTaskId: "task-1",
    worktreeId: "worktree/one",
    cleanupWorktreeOnCancel: true,
    start: false,
  });
  assert.equal(
    queued.path,
    "/api/agent-workspace/projects/project-1/background/agent",
  );
  assert.deepEqual(queued.init, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      instruction: "fix the failing test",
      runtime: {
        kind: "provider",
        model: "gpt-5.6-codex",
        providerId: "codex-account",
        permissionMode: "full",
        reasoningEffort: "high",
        maxOutputTokens: 12_000,
      },
      planId: "plan-1",
      planTaskId: "task-1",
      worktreeId: "worktree/one",
      cleanupWorktreeOnCancel: true,
      start: false,
    }),
  });

  const head = "a".repeat(40);
  const merge = agentWorktreeMergeRequest(
    "project-1",
    " worktree/one ",
    ` ${head} `,
  );
  assert.equal(
    merge.path,
    "/api/agent-workspace/projects/project-1/worktrees/worktree%2Fone/merge",
  );
  assert.deepEqual(merge.init, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expectedTargetHead: head }),
  });
  assert.throws(
    () =>
      agentBackgroundAgentRequest("project-1", {
        instruction: "   ",
        runtime: {
          kind: "local",
          model: "local-model",
          permissionMode: "off",
          maxOutputTokens: 8_192,
        },
      }),
    /instructions are required/,
  );
  assert.throws(
    () => agentWorktreeMergeRequest("project-1", "worktree-1", "HEAD"),
    /target head is invalid/,
  );
});

test("background runtime selection is credential-free and safe for unattended execution", () => {
  const local = agentBackgroundAgentRequest("project-1", {
    instruction: "Inspect the repository",
    runtime: {
      kind: "local",
      model: "  unsloth/local-coder  ",
      permissionMode: "off",
      maxOutputTokens: 8_192,
    },
    start: true,
  });
  assert.deepEqual(JSON.parse(String(local.init.body)).runtime, {
    kind: "local",
    model: "unsloth/local-coder",
    permissionMode: "off",
    maxOutputTokens: 8_192,
  });
  assert.equal("apiKey" in JSON.parse(String(local.init.body)).runtime, false);

  assert.throws(
    () =>
      agentBackgroundAgentRequest("project-1", {
        instruction: "Wait for approval",
        runtime: {
          kind: "local",
          model: "unsloth/local-coder",
          permissionMode: "ask" as never,
          maxOutputTokens: 8_192,
        },
        start: true,
      }),
    /cannot use Ask or Auto/,
  );
  assert.throws(
    () =>
      agentBackgroundAgentRequest("project-1", {
        instruction: "Use a provider",
        runtime: {
          kind: "provider",
          model: "gpt-5.6-codex",
          permissionMode: "full",
          maxOutputTokens: 8_192,
        },
        start: true,
      }),
    /saved provider connection is required/,
  );
  assert.throws(
    () =>
      agentBackgroundAgentRequest("project-1", {
        instruction: "Start without a runtime",
        runtime: undefined as never,
        start: true,
      }),
    /runtime selection is required/,
  );
});

test("the verification completion policy is wired through the Agent panel", () => {
  const api = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/api/agent-workspace-api.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const panel = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/components/agent-workspace-panel.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const goalBar = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/components/project-goal-bar.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );

  assert.match(
    api,
    /agentWorkspaceJsonRequest\("PUT", \{\s*checks,\s*requireForGoalCompletion,\s*expectedRevision,/,
  );
  assert.match(api, /selectedNames, configRevision/);
  assert.match(panel, /queueAgentVerification\(projectId, config\.revision\)/);
  assert.match(panel, /runAgentVerification\(projectId, config\.revision\)/);
  assert.match(panel, /Require fresh verification before goal completion/);
  assert.match(panel, /checked=\{requireVerificationForGoalCompletion\}/);
  assert.match(panel, /Saved policy revision \{verificationConfigRevision\}/);
  assert.match(goalBar, /Could not complete the project goal/);
});

test("background controls follow durable task state", () => {
  assert.deepEqual(agentBackgroundActions({ status: "queued" }), {
    canStart: true,
    canCancel: true,
    canRetry: false,
  });
  assert.deepEqual(agentBackgroundActions({ status: "running" }), {
    canStart: false,
    canCancel: true,
    canRetry: false,
  });
  assert.deepEqual(agentBackgroundActions({ status: "interrupted" }), {
    canStart: false,
    canCancel: false,
    canRetry: true,
  });
  assert.deepEqual(agentBackgroundActions({ status: "completed" }), {
    canStart: false,
    canCancel: false,
    canRetry: false,
  });
});

test("agent status labels are readable without changing stored values", () => {
  assert.equal(agentStatusLabel("in_progress"), "in progress");
  assert.equal(agentStatusLabel("timed_out"), "timed out");
  assert.equal(agentStatusLabel("completed"), "completed");
});

test("background permission policy excludes interactive modes", () => {
  assert.equal(backgroundAgentPermissionRunsUnattended("off"), true);
  assert.equal(backgroundAgentPermissionRunsUnattended("full"), true);
  assert.equal(backgroundAgentPermissionRunsUnattended("ask"), false);
  assert.equal(backgroundAgentPermissionRunsUnattended("auto"), false);
  assert.equal(backgroundAgentPermissionNeedsConfirmation("off"), false);
  assert.equal(backgroundAgentPermissionNeedsConfirmation("full"), true);
  assert.match(
    BACKGROUND_AGENT_PERMISSION_POLICY,
    /no interactive approval stream/,
  );
  assert.match(BACKGROUND_AGENT_PERMISSION_POLICY, /Ask for approval/);
  assert.match(
    BACKGROUND_AGENT_PERMISSION_POLICY,
    /executes tools without prompts inside the project sandbox/,
  );
  assert.match(
    BACKGROUND_AGENT_PERMISSION_POLICY,
    /relaxes ordinary sandboxing, while the project session remains bound/,
  );
  assert.match(
    BACKGROUND_AGENT_FULL_ACCESS_WARNING,
    /Filesystem access remains bound to the selected project/,
  );
  assert.match(
    BACKGROUND_AGENT_FULL_ACCESS_WARNING,
    /Child commands remain subject to the project execution network policy/,
  );
  assert.match(
    BACKGROUND_AGENT_FULL_ACCESS_WARNING,
    /web search tool can still use its configured search service/,
  );
});

test("background context is copied into an immutable visible snapshot", () => {
  const plan = {
    id: "plan-1",
    title: "Original plan",
    tasks: [{ id: "task-1", title: "Original task" }],
  };
  const task = {
    goalSnapshot: "Original goal",
    goalStatusSnapshot: "active",
    goalUpdatedAt: 123,
    planRevision: 4,
    planTaskId: "task-1",
    planSnapshot: plan,
    worktreeId: "worktree-1",
    appExitPolicy: "interrupt",
    payload: {
      runtime: {
        kind: "provider",
        model: "gpt-5.6-codex",
        providerId: "codex-account",
        providerType: "openai_codex",
        permissionMode: "full",
        reasoningEffort: "high",
        maxOutputTokens: 12_000,
        apiKey: "must-not-enter-visible-state",
        routingDigest: "must-not-enter-visible-state",
      },
    },
  };

  const snapshot = agentBackgroundSnapshot(task as never);
  plan.title = "Changed live plan";
  const [firstPlanTask] = plan.tasks;
  if (firstPlanTask) firstPlanTask.title = "Changed live task";
  task.goalSnapshot = "Changed live goal";

  assert.equal(Object.isFrozen(snapshot), true);
  assert.equal(Object.isFrozen(snapshot.runtime), true);
  assert.deepEqual(snapshot, {
    goal: "Original goal",
    goalStatus: "active",
    goalUpdatedAt: 123,
    planTitle: "Original plan",
    planRevision: 4,
    planTaskTitle: "Original task",
    worktreeId: "worktree-1",
    appExitPolicy: "interrupt",
    runtime: {
      kind: "provider",
      model: "gpt-5.6-codex",
      providerId: "codex-account",
      providerType: "openai_codex",
      permissionMode: "full",
      reasoningEffort: "high",
      maxOutputTokens: 12_000,
    },
  });
});

test("worktree merge stays gated on stopped tasks and a fresh clean head", () => {
  const active = { status: "active" } as never;
  const clean = { head: "a".repeat(40), clean: true };
  assert.deepEqual(
    agentWorktreeMergeAction({
      worktree: active,
      gitStatus: clean,
      linkedTask: { status: "running" },
    }),
    { canMerge: false, reason: "Wait for the linked task to stop" },
  );
  assert.deepEqual(
    agentWorktreeMergeAction({
      worktree: active,
      gitStatus: { ...clean, clean: false },
      linkedTask: { status: "completed" },
    }),
    {
      canMerge: false,
      reason: "Commit or stash primary workspace changes first",
    },
  );
  assert.deepEqual(
    agentWorktreeMergeAction({
      worktree: active,
      gitStatus: clean,
      linkedTask: { status: "completed" },
    }),
    { canMerge: true, reason: null },
  );
});

test("retry keeps immutable history and moves the worktree link to the new attempt", () => {
  const previous = {
    id: "task-1",
    status: "interrupted",
    worktreeId: "worktree-1",
  } as never;
  const retried = {
    id: "task-2",
    status: "queued",
    worktreeId: "worktree-1",
    parentTaskId: "task-1",
  } as never;
  const worktree = {
    id: "worktree-1",
    backgroundTaskId: "task-1",
  };

  const reconciled = reconcileAgentBackgroundMutation({
    tasks: [previous],
    worktrees: [worktree as never],
    previousTaskId: "task-1",
    action: "retry",
    updated: retried,
  });

  assert.deepEqual(
    reconciled.tasks.map((task) => task.id),
    ["task-2", "task-1"],
  );
  assert.equal(reconciled.worktrees[0]?.backgroundTaskId, "task-2");
  assert.equal(worktree.backgroundTaskId, "task-1");
});

test("agent task lifecycle and worktree merge controls are wired into the panel", () => {
  const api = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/api/agent-workspace-api.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  const panel = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/components/agent-workspace-panel.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );

  assert.match(api, /export function queueAgentTask/);
  assert.match(api, /action: "start" \| "cancel" \| "retry"/);
  assert.match(api, /export function mergeAgentWorktree/);
  assert.match(panel, /Queue agent/);
  assert.match(panel, /Start agent/);
  assert.match(panel, /Captured context/);
  assert.match(panel, /immutable/);
  assert.match(panel, /void mutateBackgroundTask\(task, "start"\)/);
  assert.match(panel, /void mutateBackgroundTask\(task, "cancel"\)/);
  assert.match(panel, /void mutateBackgroundTask\(task, "retry"\)/);
  assert.match(panel, /void mergeWorktree\(worktree\)/);
});

test("plan, repository, verification, and workspace summaries are deterministic", () => {
  assert.deepEqual(
    agentPlanProgress({
      tasks: [
        { status: "completed" },
        { status: "running" },
        { status: "completed" },
      ],
    } as never),
    { completed: 2, total: 3, percent: 67 },
  );
  assert.equal(agentRepositoryMapSummary(null), "Repository map not loaded");
  assert.equal(
    agentRepositoryMapSummary({
      fileCount: 12,
      pathsScanned: 20,
      truncated: true,
      truncationReasons: ["path-limit"],
    } as never),
    "12 text files, 20 paths scanned, partial (path-limit)",
  );
  assert.equal(latestVerificationSummary([]), "No verification evidence");
  assert.equal(
    latestVerificationSummary([{ status: "passed", stale: false }] as never),
    "passed, fresh",
  );
  assert.deepEqual(agentWorkspaceStatus(null), {
    label: "Checking workspace",
    tone: "neutral",
  });
  assert.deepEqual(
    agentWorkspaceStatus({ available: true, isGitRepository: true } as never),
    { label: "Repository connected", tone: "success" },
  );
  assert.deepEqual(
    agentWorkspaceStatus({ available: false, isGitRepository: false } as never),
    { label: "Workspace unavailable", tone: "danger" },
  );
});

test("project-switch request guards reject stale loads and background polls", () => {
  assert.equal(
    agentWorkspaceRequestIsCurrent({
      requestProjectId: "project-a",
      activeProjectId: "project-a",
      requestGeneration: 4,
      activeGeneration: 4,
    }),
    true,
  );
  assert.equal(
    agentWorkspaceRequestIsCurrent({
      requestProjectId: "project-a",
      activeProjectId: "project-b",
      requestGeneration: 4,
      activeGeneration: 4,
    }),
    false,
  );
  assert.equal(
    agentWorkspaceRequestIsCurrent({
      requestProjectId: "project-b",
      activeProjectId: "project-b",
      requestGeneration: 3,
      activeGeneration: 4,
    }),
    false,
  );
});

test("an old project response cannot overwrite the project selected after it", async () => {
  let activeProjectId = "project-a";
  let activeGeneration = 1;
  let settleProjectA: ((value: string) => void) | undefined;
  const projectAResponse = new Promise<string>((resolve) => {
    settleProjectA = resolve;
  });
  let rendered = "";

  const oldRequest = projectAResponse.then((value) => {
    if (
      agentWorkspaceRequestIsCurrent({
        requestProjectId: "project-a",
        activeProjectId,
        requestGeneration: 1,
        activeGeneration,
      })
    ) {
      rendered = value;
    }
  });

  activeProjectId = "project-b";
  activeGeneration = 2;
  rendered = "project-b dashboard";
  settleProjectA?.("project-a dashboard");
  await oldRequest;

  assert.equal(rendered, "project-b dashboard");
});

test("workspace errors redact local paths before rendering", () => {
  assert.equal(
    safeAgentWorkspaceError(
      new Error("Could not read /Users/alice/private/repository/AGENTS.md"),
    ),
    "Could not read <local path>",
  );
  assert.equal(
    safeAgentWorkspaceError(
      String.raw`Could not read C:\Users\alice\private\repository`,
    ),
    "Could not read <local path>",
  );
});
