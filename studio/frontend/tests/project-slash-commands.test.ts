// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import type {
  AgentPlan,
  AgentReviewSummary,
  AgentVerificationRun,
} from "../src/features/chat/api/agent-workspace-api.ts";
import type { ProjectRecord } from "../src/features/chat/types.ts";
import {
  PROJECT_SLASH_COMMANDS,
  executeGoalSlashCommand,
  executePlanSlashCommand,
  executeReviewSlashCommand,
  executeStatusSlashCommand,
  executeVerifySlashCommand,
  goalCommandPatch,
  goalCommandResponse,
  interceptCompareProjectSlashCommand,
  parseProjectSlashCommand,
  reviewCommandResponse,
  routeProjectSlashCommand,
  verificationCommandResponse,
} from "../src/features/chat/utils/slash-commands.ts";

function project(overrides: Partial<ProjectRecord> = {}): ProjectRecord {
  return {
    id: "project-1",
    name: "Project",
    instructions: "",
    archived: false,
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

function plan(overrides: Partial<AgentPlan> = {}): AgentPlan {
  return {
    id: "plan-1",
    projectId: "project-1",
    title: "Ship the workspace",
    goalSnapshot: "Build Codex-style projects",
    goalUpdatedAt: 1,
    status: "active",
    revision: 0,
    tasks: [],
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

function verificationRun(
  overrides: Partial<AgentVerificationRun> = {},
): AgentVerificationRun {
  return {
    id: "verify-1",
    projectId: "project-1",
    worktreeId: null,
    status: "passed",
    configRevision: 1,
    sourceFingerprint: "a".repeat(64),
    finalFingerprint: "a".repeat(64),
    results: [
      {
        name: "tests",
        kind: "test",
        command: "npm test",
        required: true,
        timeoutSeconds: 300,
        logLimitBytes: 262_144,
        status: "passed",
        exitCode: 0,
        output: "ok",
        outputBytes: 2,
        outputTruncated: false,
        startedAt: 1,
        completedAt: 11,
        durationMs: 10,
      },
    ],
    startedAt: 1,
    completedAt: 11,
    stale: false,
    ...overrides,
  };
}

function review(
  overrides: Partial<AgentReviewSummary> = {},
): AgentReviewSummary {
  return {
    projectId: "project-1",
    goal: "Ship the workspace",
    goalStatus: "active",
    git: {
      head: "a".repeat(40),
      branch: "feat/workspace",
      detached: false,
      clean: false,
      counts: { staged: 1, unstaged: 2, untracked: 1, conflicted: 0 },
      files: [
        { code: "M ", path: "backend.py" },
        { code: "??", path: "frontend.tsx" },
      ],
      truncated: false,
    },
    gitError: null,
    diff: { staged: false, diff: "diff", truncated: false },
    plans: [plan()],
    verification: [verificationRun()],
    limits: { diffBytes: 524_288, verificationRuns: 10 },
    projectRoot: "<project_root>",
    ...overrides,
  };
}

test("parses the exact goal command lifecycle", () => {
  assert.deepEqual(parseProjectSlashCommand("/goal"), {
    name: "goal",
    action: "show",
  });
  assert.deepEqual(parseProjectSlashCommand(" /GOAL set Ship V1 "), {
    name: "goal",
    action: "set",
    value: "Ship V1",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal ship folder projects"), {
    name: "goal",
    action: "set",
    value: "ship folder projects",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal done"), {
    name: "goal",
    action: "complete",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal pause"), {
    name: "goal",
    action: "pause",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal resume"), {
    name: "goal",
    action: "resume",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal reopen"), {
    name: "goal",
    action: "reopen",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal clear"), {
    name: "goal",
    action: "clear",
  });
  assert.deepEqual(parseProjectSlashCommand("/goal help"), {
    name: "goal",
    action: "help",
  });
});

test("an empty explicit set shows help instead of setting the word set", () => {
  assert.deepEqual(parseProjectSlashCommand("/goal set"), {
    name: "goal",
    action: "help",
  });
  assert.equal(parseProjectSlashCommand("/unknown"), null);
  assert.equal(parseProjectSlashCommand("goal without slash"), null);
});

test("registers and parses the exact verify and plan lifecycles", () => {
  assert.deepEqual(
    PROJECT_SLASH_COMMANDS.map((command) => command.name),
    ["goal", "verify", "plan", "status", "review"],
  );
  assert.ok(PROJECT_SLASH_COMMANDS.every((command) => command.projectOnly));
  assert.equal(
    PROJECT_SLASH_COMMANDS.find((command) => command.name === "goal")?.usage,
    "/goal [set <text>|<text>|pause|resume|done|reopen|clear|help]",
  );
  assert.deepEqual(parseProjectSlashCommand("/verify"), {
    name: "verify",
    action: "run",
  });
  assert.deepEqual(parseProjectSlashCommand("/verify help"), {
    name: "verify",
    action: "help",
  });
  assert.deepEqual(parseProjectSlashCommand("/verify tests"), {
    name: "verify",
    action: "help",
  });
  assert.deepEqual(parseProjectSlashCommand("/plan"), {
    name: "plan",
    action: "show",
  });
  assert.deepEqual(parseProjectSlashCommand("/plan create Release candidate"), {
    name: "plan",
    action: "create",
    value: "Release candidate",
  });
  assert.deepEqual(parseProjectSlashCommand("/plan done"), {
    name: "plan",
    action: "complete",
  });
  assert.deepEqual(parseProjectSlashCommand("/plan create"), {
    name: "plan",
    action: "help",
  });
  assert.deepEqual(parseProjectSlashCommand("/plan unknown"), {
    name: "plan",
    action: "help",
  });
  assert.deepEqual(parseProjectSlashCommand("/status"), {
    name: "status",
    action: "show",
  });
  assert.deepEqual(parseProjectSlashCommand("/status detail"), {
    name: "status",
    action: "help",
  });
  assert.deepEqual(parseProjectSlashCommand("/review"), {
    name: "review",
    action: "run",
  });
  assert.deepEqual(parseProjectSlashCommand("/review unknown"), {
    name: "review",
    action: "help",
  });
});

test("goal patches persist set, pause, resume, complete, reopen, and clear state", () => {
  const set = parseProjectSlashCommand("/goal set make tests green");
  assert.ok(set);
  assert.equal(set.name, "goal");
  if (set.name !== "goal") {
    throw new Error("unexpected command");
  }
  assert.deepEqual(goalCommandPatch(set, project(), 99), {
    goal: "make tests green",
    goalStatus: "active",
    goalUpdatedAt: 99,
  });

  const active = project({ goal: "make tests green", goalStatus: "active" });
  assert.deepEqual(
    goalCommandPatch({ name: "goal", action: "pause" }, active, 100),
    { goalStatus: "paused", goalUpdatedAt: 100 },
  );
  assert.deepEqual(
    goalCommandPatch(
      { name: "goal", action: "resume" },
      { ...active, goalStatus: "paused" },
      101,
    ),
    { goalStatus: "active", goalUpdatedAt: 101 },
  );
  assert.deepEqual(
    goalCommandPatch({ name: "goal", action: "complete" }, active, 102),
    { goalStatus: "completed", goalUpdatedAt: 102 },
  );
  assert.deepEqual(
    goalCommandPatch(
      { name: "goal", action: "reopen" },
      { ...active, goalStatus: "completed" },
      103,
    ),
    { goalStatus: "active", goalUpdatedAt: 103 },
  );
  assert.deepEqual(
    goalCommandPatch({ name: "goal", action: "clear" }, active, 104),
    { goal: null, goalStatus: null, goalUpdatedAt: 104 },
  );
  assert.equal(
    goalCommandPatch({ name: "goal", action: "pause" }, project(), 105),
    null,
  );
  assert.equal(
    goalCommandPatch({ name: "goal", action: "resume" }, project(), 106),
    null,
  );
  assert.equal(
    goalCommandPatch({ name: "goal", action: "complete" }, project(), 107),
    null,
  );
  assert.equal(
    goalCommandPatch({ name: "goal", action: "help" }, active, 108),
    null,
  );
});

test("local goal execution mutates only through project persistence", async () => {
  const writes: Partial<ProjectRecord>[] = [];
  const initial = project();
  const command = parseProjectSlashCommand("/goal set finish the harness");
  assert.ok(command);
  assert.equal(command.name, "goal");
  if (command.name !== "goal") {
    throw new Error("unexpected command");
  }

  const result = await executeGoalSlashCommand(
    command,
    initial,
    async (_projectId, patch) => {
      writes.push(patch);
      return { ...initial, ...patch };
    },
  );

  assert.equal(result.persisted, true);
  assert.deepEqual(writes, [
    {
      goal: "finish the harness",
      goalStatus: "active",
      goalUpdatedAt: writes[0]?.goalUpdatedAt,
    },
  ]);
  assert.equal(result.response, "Project goal set: finish the harness");

  const show = await executeGoalSlashCommand(
    { name: "goal", action: "show" },
    result.project,
    async () => {
      throw new Error("show must not write");
    },
  );
  assert.equal(show.persisted, false);
  assert.equal(show.response, "Project goal (active): finish the harness");
});

test("pause and resume persist goal status without changing goal text", async () => {
  let current = project({ goal: "Finish the harness", goalStatus: "active" });
  const writes: Partial<ProjectRecord>[] = [];
  const persist = async (_projectId: string, patch: Partial<ProjectRecord>) => {
    writes.push(patch);
    current = { ...current, ...patch };
    return current;
  };

  const paused = await executeGoalSlashCommand(
    { name: "goal", action: "pause" },
    current,
    persist,
  );
  assert.equal(paused.persisted, true);
  assert.equal(paused.project.goalStatus, "paused");
  assert.equal(paused.project.goal, "Finish the harness");
  assert.equal(paused.response, "Project goal paused: Finish the harness");

  const resumed = await executeGoalSlashCommand(
    { name: "goal", action: "resume" },
    current,
    persist,
  );
  assert.equal(resumed.persisted, true);
  assert.equal(resumed.project.goalStatus, "active");
  assert.equal(resumed.project.goal, "Finish the harness");
  assert.equal(resumed.response, "Project goal resumed: Finish the harness");
  assert.deepEqual(
    writes.map((patch) => patch.goalStatus),
    ["paused", "active"],
  );
  assert.ok(writes.every((patch) => !("goal" in patch)));
});

test("goal commands, including pause and resume, never enter the inference fallback", async () => {
  let modelCalls = 0;
  const output: string[] = [];
  const active = project({ goal: "Finish the harness", goalStatus: "active" });
  for (const input of ["/goal help", "/goal pause", "/goal resume"]) {
    for await (const chunk of routeProjectSlashCommand(
      input,
      async (command) => {
        assert.equal(command.name, "goal");
        if (command.name !== "goal") throw new Error("unexpected command");
        return goalCommandResponse(command, active);
      },
      async function* () {
        modelCalls += 1;
        yield "model response";
      },
    )) {
      output.push(chunk);
    }
  }
  assert.equal(modelCalls, 0);
  assert.deepEqual(output, [
    "Usage: `/goal`, `/goal set <text>`, `/goal <text>`, `/goal pause`, `/goal resume`, `/goal done`, `/goal reopen`, `/goal clear`, or `/goal help`.",
    "Project goal paused: Finish the harness",
    "Project goal resumed: Finish the harness",
  ]);

  for await (const chunk of routeProjectSlashCommand(
    "Explain this code",
    async () => "unexpected local response",
    async function* () {
      modelCalls += 1;
      yield "model response";
    },
  )) {
    output.push(chunk);
  }
  assert.equal(modelCalls, 1);
  assert.equal(output.at(-1), "model response");
});

test("verify runs configured checks locally and reports fresh or stale evidence", async () => {
  let modelCalls = 0;
  let verificationCalls = 0;
  const output: string[] = [];
  for await (const chunk of routeProjectSlashCommand(
    "/verify",
    async (command) => {
      assert.equal(command.name, "verify");
      if (command.name !== "verify") throw new Error("unexpected command");
      return executeVerifySlashCommand(
        command,
        "project-1",
        async (projectId) => {
          verificationCalls += 1;
          assert.equal(projectId, "project-1");
          return verificationRun();
        },
      );
    },
    async function* () {
      modelCalls += 1;
      yield "model response";
    },
  )) {
    output.push(chunk);
  }

  assert.equal(modelCalls, 0);
  assert.equal(verificationCalls, 1);
  assert.match(output[0] ?? "", /Verification passed\. Evidence is fresh\./);
  assert.match(output[0] ?? "", /tests: passed \(exit 0, 10 ms\)/);
  assert.match(output[0] ?? "", /Evidence: run verify-1/);
  assert.match(
    verificationCommandResponse(
      verificationRun({ status: "failed", stale: true }),
    ),
    /Verification failed\. Evidence is stale because the workspace changed after this run\./,
  );

  const help = parseProjectSlashCommand("/verify help");
  assert.ok(help && help.name === "verify");
  const helpResponse = await executeVerifySlashCommand(
    help,
    "project-1",
    async () => {
      throw new Error("help must not run verification");
    },
  );
  assert.match(helpResponse, /Usage: `\/verify`/);
});

test("plan commands create, inspect, and complete durable plans without inference", async () => {
  let modelCalls = 0;
  let currentPlans: AgentPlan[] = [];
  const calls: string[] = [];
  const api = {
    async listPlans(projectId: string): Promise<AgentPlan[]> {
      calls.push(`list:${projectId}`);
      return currentPlans;
    },
    async createPlan(
      projectId: string,
      payload: { title: string; tasks: [] },
    ): Promise<AgentPlan> {
      calls.push(
        `create:${projectId}:${payload.title}:${payload.tasks.length}`,
      );
      const created = plan({ title: payload.title });
      currentPlans = [created];
      return created;
    },
    async updatePlan(
      projectId: string,
      selected: Pick<AgentPlan, "id" | "revision">,
      status: "completed",
    ): Promise<AgentPlan> {
      calls.push(
        `update:${projectId}:${selected.id}:${selected.revision}:${status}`,
      );
      const updated = plan({
        ...currentPlans[0],
        status,
        revision: selected.revision + 1,
      });
      currentPlans = [updated];
      return updated;
    },
  };

  async function runLocal(input: string): Promise<string> {
    const chunks: string[] = [];
    for await (const chunk of routeProjectSlashCommand(
      input,
      async (command) => {
        assert.equal(command.name, "plan");
        if (command.name !== "plan") throw new Error("unexpected command");
        return executePlanSlashCommand(command, "project-1", api);
      },
      async function* () {
        modelCalls += 1;
        yield "model response";
      },
    )) {
      chunks.push(chunk);
    }
    return chunks.join("");
  }

  assert.match(await runLocal("/plan"), /This project has no plans/);
  assert.equal(
    await runLocal("/plan create Release candidate"),
    "Plan created (revision 0): Release candidate",
  );
  assert.match(
    await runLocal("/plan"),
    /Latest plan \(active, revision 0\): Release candidate/,
  );
  assert.equal(
    await runLocal("/plan done"),
    "Plan completed (revision 1): Release candidate",
  );
  assert.equal(modelCalls, 0);
  assert.deepEqual(calls, [
    "list:project-1",
    "create:project-1:Release candidate:0",
    "list:project-1",
    "list:project-1",
    "update:project-1:plan-1:0:completed",
  ]);
});

test("status and review commands return bounded project evidence", async () => {
  const activeProject = project({
    goal: "Ship the workspace",
    goalStatus: "active",
  });
  const statusCommand = parseProjectSlashCommand("/status");
  assert.ok(statusCommand && statusCommand.name === "status");
  const status = await executeStatusSlashCommand(statusCommand, activeProject, {
    async getWorkspace(projectId) {
      assert.equal(projectId, "project-1");
      return {
        projectId,
        workspaceKind: "folder",
        available: true,
        error: null,
        isGitRepository: true,
        capabilities: {
          instructions: true,
          repositoryMap: true,
          verification: true,
          plans: true,
          background: true,
          git: true,
          worktrees: true,
          review: true,
        },
      };
    },
    async getGitStatus() {
      const value = review().git;
      if (!value) throw new Error("missing git fixture");
      return value;
    },
  });
  assert.match(status, /Workspace: folder, available/);
  assert.match(status, /Goal: active, Ship the workspace/);
  assert.match(status, /Git: feat\/workspace, 1 staged, 2 unstaged/);

  const unavailable = await executeStatusSlashCommand(
    statusCommand,
    activeProject,
    {
      async getWorkspace(projectId) {
        return {
          projectId,
          workspaceKind: "folder",
          available: false,
          error: "Missing /Users/alice/private/repository",
          isGitRepository: false,
          capabilities: {
            instructions: false,
            repositoryMap: false,
            verification: false,
            plans: true,
            background: false,
            git: false,
            worktrees: false,
            review: false,
          },
        };
      },
      async getGitStatus() {
        throw new Error("unavailable workspaces must not read Git state");
      },
    },
  );
  assert.match(unavailable, /reconnect the repository in Unsloth Desktop/);
  assert.doesNotMatch(unavailable, /Users|alice|private\/repository/);

  const reviewCommand = parseProjectSlashCommand("/review");
  assert.ok(reviewCommand && reviewCommand.name === "review");
  let reviewCalls = 0;
  const response = await executeReviewSlashCommand(
    reviewCommand,
    "project-1",
    async (projectId) => {
      reviewCalls += 1;
      assert.equal(projectId, "project-1");
      return review();
    },
  );
  assert.equal(reviewCalls, 1);
  assert.match(response, /Plans: 1 active, 1 total/);
  assert.match(response, /Verification: passed, fresh/);
  assert.match(response, /M {2}backend\.py/);
  assert.match(reviewCommandResponse(review()), /Changed files:/);

  const help = parseProjectSlashCommand("/review help");
  assert.ok(help && help.name === "review");
  assert.match(
    await executeReviewSlashCommand(help, "project-1", async () => {
      throw new Error("help must not load review state");
    }),
    /Usage: `\/review`/,
  );
});

test("goal command responses cover no-goal, pause, resume, completion, reopen, and help", () => {
  assert.equal(
    goalCommandResponse({ name: "goal", action: "show" }, project()),
    "This project has no goal set. Set one with `/goal set <text>` or `/goal <text>`.",
  );
  const active = project({ goal: "Finish the harness", goalStatus: "active" });
  assert.equal(
    goalCommandResponse({ name: "goal", action: "pause" }, active),
    "Project goal paused: Finish the harness",
  );
  assert.equal(
    goalCommandResponse({ name: "goal", action: "resume" }, active),
    "Project goal resumed: Finish the harness",
  );
  assert.equal(
    goalCommandResponse({ name: "goal", action: "complete" }, active),
    "Project goal completed: Finish the harness",
  );
  assert.equal(
    goalCommandResponse({ name: "goal", action: "reopen" }, active),
    "Project goal reopened: Finish the harness",
  );
  assert.match(
    goalCommandResponse({ name: "goal", action: "help" }, active),
    /\/goal set <text>.*\/goal pause.*\/goal resume.*\/goal done.*\/goal reopen.*\/goal help/,
  );
});

test("compare executes goal, verify, and plan once and appends one local result to both panes", async () => {
  type HistoryEntry =
    | { role: "user"; content: { type: "text"; text: string }[] }
    | { role: "assistant"; text: string };

  let modelLoadCalls = 0;
  let inferenceCalls = 0;
  let goalWrites = 0;
  let verificationRuns = 0;
  let planCreates = 0;
  const commandCalls = { goal: 0, verify: 0, plan: 0 };
  let currentProject = project();

  const execute = async (
    command: NonNullable<ReturnType<typeof parseProjectSlashCommand>>,
  ): Promise<string> => {
    switch (command.name) {
      case "goal":
        commandCalls.goal += 1;
        return (
          await executeGoalSlashCommand(
            command,
            currentProject,
            async (_projectId, patch) => {
              goalWrites += 1;
              currentProject = { ...currentProject, ...patch };
              return currentProject;
            },
          )
        ).response;
      case "verify":
        commandCalls.verify += 1;
        return executeVerifySlashCommand(command, "project-1", async () => {
          verificationRuns += 1;
          return verificationRun();
        });
      case "plan":
        commandCalls.plan += 1;
        return executePlanSlashCommand(command, "project-1", {
          async listPlans() {
            return [];
          },
          async createPlan(_projectId, payload) {
            planCreates += 1;
            return plan({ title: payload.title });
          },
          async updatePlan() {
            throw new Error("create must not complete a plan");
          },
        });
      default:
        throw new Error(`unexpected command: ${command.name}`);
    }
  };

  const submit = async (input: string): Promise<HistoryEntry[][]> => {
    const histories: HistoryEntry[][] = [[], []];
    const userContent = [{ type: "text" as const, text: input }];
    const pane = (history: HistoryEntry[]) => ({
      appendUserMessage(content: typeof userContent) {
        history.push({ role: "user", content });
      },
      appendAssistantMessage(text: string) {
        history.push({ role: "assistant", text });
      },
    });
    const handled = await interceptCompareProjectSlashCommand({
      input,
      userContent,
      panes: [pane(histories[0]), pane(histories[1])],
      execute,
    });
    if (!handled) {
      modelLoadCalls += 1;
      inferenceCalls += 1;
    }
    return histories;
  };

  const goalHistories = await submit("/goal set finish compare commands");
  const verifyHistories = await submit("/verify");
  const planHistories = await submit("/plan create Release compare mode");

  assert.deepEqual(commandCalls, { goal: 1, verify: 1, plan: 1 });
  assert.equal(goalWrites, 1);
  assert.equal(verificationRuns, 1);
  assert.equal(planCreates, 1);
  assert.equal(modelLoadCalls, 0);
  assert.equal(inferenceCalls, 0);

  for (const histories of [goalHistories, verifyHistories, planHistories]) {
    assert.equal(histories.length, 2);
    assert.deepEqual(histories[0], histories[1]);
    assert.equal(histories[0]?.length, 2);
    assert.equal(histories[0]?.[0]?.role, "user");
    assert.equal(histories[0]?.[1]?.role, "assistant");
  }
  assert.equal(
    goalHistories[0]?.[1]?.role === "assistant"
      ? goalHistories[0][1].text
      : null,
    "Project goal set: finish compare commands",
  );
  assert.match(
    verifyHistories[0]?.[1]?.role === "assistant"
      ? verifyHistories[0][1].text
      : "",
    /Verification passed\. Evidence is fresh\./,
  );
  assert.equal(
    planHistories[0]?.[1]?.role === "assistant"
      ? planHistories[0][1].text
      : null,
    "Plan created (revision 0): Release compare mode",
  );

  const ordinaryHistories = await submit("Explain this repository");
  assert.deepEqual(ordinaryHistories, [[], []]);
  assert.equal(modelLoadCalls, 1);
  assert.equal(inferenceCalls, 1);
});

test("the chat adapter routes locally before inference and uses persisted history", () => {
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  const runtime = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    ),
    "utf8",
  );
  const sharedComposer = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    ),
    "utf8",
  );

  assert.match(
    adapter,
    /const slashCommand = parseProjectSlashCommand\(commandText\)/,
  );
  assert.match(adapter, /await executeLocalProjectSlashCommand\(slashCommand/);
  assert.match(adapter, /case "verify":[\s\S]*?runAgentVerification/);
  assert.match(adapter, /case "plan":[\s\S]*?listPlans: listAgentPlans/);
  assert.match(
    adapter,
    /case "status":[\s\S]*?getWorkspace: getAgentWorkspace/,
  );
  assert.match(adapter, /case "review":[\s\S]*?getAgentReview/);
  assert.match(adapter, /command is available only inside a project/);
  assert.match(
    runtime,
    /createPersistedRunAdapter\(\s*createOpenAIStreamAdapter\(/,
  );
  assert.match(
    runtime,
    /await saveStoredChatMessage\(\{[\s\S]*?role: message\.role,[\s\S]*?content:/,
  );
  const compareInterceptAt = sharedComposer.indexOf(
    "await interceptCompareProjectSlashCommand({",
  );
  const compareModelLoadAt = sharedComposer.indexOf(
    ".beginModelLoading()",
    compareInterceptAt,
  );
  assert.ok(compareInterceptAt >= 0);
  assert.ok(compareModelLoadAt > compareInterceptAt);
  assert.match(
    sharedComposer,
    /appendAssistantMessage: \(text\)[\s\S]*?role: "assistant"[\s\S]*?startRun: false/,
  );
  const tokenCountBuilder = adapter.slice(
    adapter.indexOf("export async function buildOutboundMessagesForTokenCount"),
    adapter.indexOf("export function buildLocalTokenCountReasoning"),
  );
  assert.match(
    tokenCountBuilder,
    /resolveUserSystemPrompt\(\s*params\.systemPrompt,\s*params\.systemVariables/,
  );
  assert.match(
    adapter,
    /const combinedSystemPrompt = resolveUserSystemPrompt\(\s*params\.systemPrompt,\s*params\.systemVariables/,
  );
  assert.doesNotMatch(adapter, /projectInstructionContext/);
  assert.doesNotMatch(adapter, /composeProjectAwareSystemPrompt/);
  assert.doesNotMatch(adapter, /unsloth_project_context/);
  const recount = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/chat/utils/refresh-context-usage.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(
    recount,
    /const sessionId = await resolveSandboxSessionId\(payloadThreadId\)/,
  );
  assert.match(recount, /session_id: sessionId/);
});
