// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  AgentGitStatus,
  AgentPlan,
  AgentReviewSummary,
  AgentVerificationRun,
  AgentWorkspaceOverview,
} from "../api/agent-workspace-api";
import type { ProjectRecord } from "../types";

export type ProjectGoalStatus = "active" | "paused" | "completed";

export type GoalSlashCommand = {
  name: "goal";
  action:
    | "show"
    | "set"
    | "pause"
    | "resume"
    | "complete"
    | "reopen"
    | "clear"
    | "help";
  value?: string;
};

export type VerifySlashCommand = {
  name: "verify";
  action: "run" | "help";
};

export type PlanSlashCommand = {
  name: "plan";
  action: "show" | "create" | "complete" | "help";
  value?: string;
};

export type StatusSlashCommand = {
  name: "status";
  action: "show" | "help";
};

export type ReviewSlashCommand = {
  name: "review";
  action: "run" | "help";
};

export type ProjectSlashCommand =
  | GoalSlashCommand
  | VerifySlashCommand
  | PlanSlashCommand
  | StatusSlashCommand
  | ReviewSlashCommand;

export type CompareSlashCommandPane<TContent> = {
  appendUserMessage: (content: TContent) => void;
  appendAssistantMessage: (response: string) => void;
};

export type SlashCommandDefinition = {
  name: string;
  usage: string;
  description: string;
  projectOnly: boolean;
};

export const PROJECT_SLASH_COMMANDS: readonly SlashCommandDefinition[] = [
  {
    name: "goal",
    usage: "/goal [set <text>|<text>|pause|resume|done|reopen|clear|help]",
    description: "Set and manage the persistent objective for this project.",
    projectOnly: true,
  },
  {
    name: "verify",
    usage: "/verify [help]",
    description: "Run this project's configured verification checks.",
    projectOnly: true,
  },
  {
    name: "plan",
    usage: "/plan [create <title>|done|help]",
    description: "Inspect and manage this project's latest durable plan.",
    projectOnly: true,
  },
  {
    name: "status",
    usage: "/status [help]",
    description: "Show this project's workspace, goal, and Git state.",
    projectOnly: true,
  },
  {
    name: "review",
    usage: "/review [help]",
    description: "Build a bounded review of this project's current changes.",
    projectOnly: true,
  },
] as const;

const PROJECT_COMMAND_PATTERN = /^\/([a-z][a-z0-9_-]*)(?:\s+([\s\S]*))?$/i;
const GOAL_SET_PATTERN = /^set(?:\s+([\s\S]+))?$/i;
const PLAN_CREATE_PATTERN = /^create(?:\s+([\s\S]+))?$/i;

function normalizedCommand(
  input: string,
): { name: string; tail: string } | null {
  const trimmed = input.trim();
  const match = PROJECT_COMMAND_PATTERN.exec(trimmed);
  if (!match?.[1]) {
    return null;
  }
  return { name: match[1].toLowerCase(), tail: (match[2] ?? "").trim() };
}

function parseGoalSlashCommand(tail: string): GoalSlashCommand {
  if (!tail) {
    return { name: "goal", action: "show" };
  }

  const lower = tail.toLowerCase();
  if (lower === "done") {
    return { name: "goal", action: "complete" };
  }
  if (lower === "pause") {
    return { name: "goal", action: "pause" };
  }
  if (lower === "resume") {
    return { name: "goal", action: "resume" };
  }
  if (lower === "reopen") {
    return { name: "goal", action: "reopen" };
  }
  if (lower === "clear") {
    return { name: "goal", action: "clear" };
  }
  if (lower === "help") {
    return { name: "goal", action: "help" };
  }

  const explicitSet = GOAL_SET_PATTERN.exec(tail);
  if (explicitSet && !explicitSet[1]?.trim()) {
    return { name: "goal", action: "help" };
  }
  const value = (explicitSet?.[1] ?? tail).trim();
  return value
    ? { name: "goal", action: "set", value }
    : { name: "goal", action: "help" };
}

function parseVerifySlashCommand(tail: string): VerifySlashCommand {
  return {
    name: "verify",
    action: tail ? "help" : "run",
  };
}

function parsePlanSlashCommand(tail: string): PlanSlashCommand {
  if (!tail) {
    return { name: "plan", action: "show" };
  }
  const lower = tail.toLowerCase();
  if (lower === "done") {
    return { name: "plan", action: "complete" };
  }
  if (lower === "help") {
    return { name: "plan", action: "help" };
  }

  const create = PLAN_CREATE_PATTERN.exec(tail);
  const value = create?.[1]?.trim();
  return value
    ? { name: "plan", action: "create", value }
    : { name: "plan", action: "help" };
}

function parseStatusSlashCommand(tail: string): StatusSlashCommand {
  return { name: "status", action: tail ? "help" : "show" };
}

function parseReviewSlashCommand(tail: string): ReviewSlashCommand {
  return { name: "review", action: tail ? "help" : "run" };
}

export function parseProjectSlashCommand(
  input: string,
): ProjectSlashCommand | null {
  const command = normalizedCommand(input);
  if (!command) {
    return null;
  }
  switch (command.name) {
    case "goal":
      return parseGoalSlashCommand(command.tail);
    case "verify":
      return parseVerifySlashCommand(command.tail);
    case "plan":
      return parsePlanSlashCommand(command.tail);
    case "status":
      return parseStatusSlashCommand(command.tail);
    case "review":
      return parseReviewSlashCommand(command.tail);
    default:
      return null;
  }
}

/**
 * Intercept one project command at the shared compare boundary. The command is
 * executed once, then the same local result is appended to both pane histories.
 * Returning false leaves ordinary compare generation entirely to the caller.
 */
export async function interceptCompareProjectSlashCommand<TContent>(args: {
  input: string;
  userContent: TContent;
  panes: readonly [
    CompareSlashCommandPane<TContent>,
    CompareSlashCommandPane<TContent>,
  ];
  execute: (command: ProjectSlashCommand) => Promise<string>;
  onIntercept?: () => void;
}): Promise<boolean> {
  const command = parseProjectSlashCommand(args.input);
  if (!command) {
    return false;
  }

  args.onIntercept?.();
  const panes = [...args.panes];
  for (const pane of panes) {
    pane.appendUserMessage(args.userContent);
  }
  const response = await args.execute(command);
  for (const pane of panes) {
    pane.appendAssistantMessage(response);
  }
  return true;
}

export function goalCommandPatch(
  command: GoalSlashCommand,
  project: ProjectRecord,
  now = Date.now(),
): Partial<ProjectRecord> | null {
  switch (command.action) {
    case "show":
      return null;
    case "set":
      return {
        goal: command.value?.trim() || null,
        goalStatus: "active",
        goalUpdatedAt: now,
      };
    case "pause":
      return project.goal ? { goalStatus: "paused", goalUpdatedAt: now } : null;
    case "resume":
    case "reopen":
      return project.goal ? { goalStatus: "active", goalUpdatedAt: now } : null;
    case "complete":
      return project.goal
        ? { goalStatus: "completed", goalUpdatedAt: now }
        : null;
    case "clear":
      return { goal: null, goalStatus: null, goalUpdatedAt: now };
    case "help":
      return null;
    default: {
      const unreachable: never = command.action;
      return unreachable;
    }
  }
}

export function goalCommandResponse(
  command: GoalSlashCommand,
  project: ProjectRecord,
): string {
  const goal = project.goal?.trim();
  const status = project.goalStatus ?? (goal ? "active" : null);

  if (command.action === "help") {
    return "Usage: `/goal`, `/goal set <text>`, `/goal <text>`, `/goal pause`, `/goal resume`, `/goal done`, `/goal reopen`, `/goal clear`, or `/goal help`.";
  }

  if (!goal) {
    return command.action === "clear"
      ? "Project goal cleared."
      : "This project has no goal set. Set one with `/goal set <text>` or `/goal <text>`.";
  }

  switch (command.action) {
    case "set":
      return `Project goal set: ${goal}`;
    case "pause":
      return `Project goal paused: ${goal}`;
    case "resume":
      return `Project goal resumed: ${goal}`;
    case "complete":
      return `Project goal completed: ${goal}`;
    case "reopen":
      return `Project goal reopened: ${goal}`;
    case "clear":
      return "Project goal cleared.";
    case "show":
      return `Project goal (${status ?? "active"}): ${goal}`;
    default: {
      const unreachable: never = command.action;
      return unreachable;
    }
  }
}

export async function executeGoalSlashCommand(
  command: GoalSlashCommand,
  project: ProjectRecord,
  persist: (
    projectId: string,
    patch: Partial<ProjectRecord>,
  ) => Promise<ProjectRecord>,
): Promise<{ project: ProjectRecord; response: string; persisted: boolean }> {
  const patch = goalCommandPatch(command, project);
  const updated = patch ? await persist(project.id, patch) : project;
  return {
    project: updated,
    response: goalCommandResponse(command, updated),
    persisted: patch !== null,
  };
}

const VERIFY_HELP_RESPONSE =
  "Usage: `/verify` to run this project's configured checks, or `/verify help` to show this message.";
const PLAN_HELP_RESPONSE =
  "Usage: `/plan` to show the latest plan, `/plan create <title>` to create one, `/plan done` to complete the latest active plan, or `/plan help` to show this message.";
const STATUS_HELP_RESPONSE =
  "Usage: `/status` to show this project's workspace, goal, and Git state, or `/status help` to show this message.";
const REVIEW_HELP_RESPONSE =
  "Usage: `/review` to build a bounded review of current project changes, or `/review help` to show this message.";

function compactFingerprint(value: string | null | undefined): string {
  if (!value) {
    return "unavailable";
  }
  return value.length > 16 ? `${value.slice(0, 16)}...` : value;
}

export function verificationCommandResponse(run: AgentVerificationRun): string {
  const passed = run.results.filter(
    (result) => result.status === "passed",
  ).length;
  const freshness = run.stale
    ? "stale because the workspace changed after this run"
    : "fresh";
  const lines = run.results.map((result) => {
    const exit =
      result.exitCode === null ? "no exit code" : `exit ${result.exitCode}`;
    return `- ${result.name}: ${result.status} (${exit}, ${result.durationMs} ms)`;
  });
  return [
    `Verification ${run.status}. Evidence is ${freshness}.`,
    `${passed} of ${run.results.length} checks passed.`,
    ...lines,
    `Evidence: run ${run.id}, source ${compactFingerprint(run.sourceFingerprint)}, final ${compactFingerprint(run.finalFingerprint)}.`,
  ].join("\n");
}

export async function executeVerifySlashCommand(
  command: VerifySlashCommand,
  projectId: string,
  runVerification: (projectId: string) => Promise<AgentVerificationRun>,
): Promise<string> {
  if (command.action === "help") {
    return VERIFY_HELP_RESPONSE;
  }
  return verificationCommandResponse(await runVerification(projectId));
}

export function planCommandResponse(plan: AgentPlan | null): string {
  if (!plan) {
    return "This project has no plans. Create one with `/plan create <title>`.";
  }
  const tasks = plan.tasks
    .slice(0, 20)
    .map(
      (task, index) =>
        `${index + 1}. [${task.status}] ${task.title}${task.blocker ? ` (blocked: ${task.blocker})` : ""}`,
    );
  if (plan.tasks.length > tasks.length) {
    tasks.push(`${plan.tasks.length - tasks.length} more tasks omitted.`);
  }
  return [
    `Latest plan (${plan.status}, revision ${plan.revision}): ${plan.title}`,
    plan.goalSnapshot ? `Goal snapshot: ${plan.goalSnapshot}` : "",
    plan.tasks.length === 0 ? "No tasks yet." : `Tasks:\n${tasks.join("\n")}`,
  ]
    .filter(Boolean)
    .join("\n");
}

export async function executePlanSlashCommand(
  command: PlanSlashCommand,
  projectId: string,
  api: {
    listPlans: (projectId: string) => Promise<AgentPlan[]>;
    createPlan: (
      projectId: string,
      payload: { title: string; tasks: [] },
    ) => Promise<AgentPlan>;
    updatePlan: (
      projectId: string,
      plan: Pick<AgentPlan, "id" | "revision">,
      status: "completed",
    ) => Promise<AgentPlan>;
  },
): Promise<string> {
  switch (command.action) {
    case "help":
      return PLAN_HELP_RESPONSE;
    case "create": {
      const created = await api.createPlan(projectId, {
        title: command.value ?? "",
        tasks: [],
      });
      return `Plan created (revision ${created.revision}): ${created.title}`;
    }
    case "show": {
      const plans = await api.listPlans(projectId);
      return planCommandResponse(plans[0] ?? null);
    }
    case "complete": {
      const plans = await api.listPlans(projectId);
      const active = plans.find((plan) => plan.status === "active");
      if (!active) {
        return "This project has no active plan to complete.";
      }
      const updated = await api.updatePlan(projectId, active, "completed");
      return `Plan completed (revision ${updated.revision}): ${updated.title}`;
    }
    default: {
      const unreachable: never = command.action;
      return unreachable;
    }
  }
}

function gitStatusSummary(status: AgentGitStatus): string {
  const branch = status.detached
    ? `detached at ${status.head.slice(0, 12)}`
    : status.branch || "unknown branch";
  if (status.clean) {
    return `${branch}, clean`;
  }
  const counts = status.counts;
  return `${branch}, ${counts.staged} staged, ${counts.unstaged} unstaged, ${counts.untracked} untracked, ${counts.conflicted} conflicted`;
}

export async function executeStatusSlashCommand(
  command: StatusSlashCommand,
  project: ProjectRecord,
  api: {
    getWorkspace: (projectId: string) => Promise<AgentWorkspaceOverview>;
    getGitStatus: (projectId: string) => Promise<AgentGitStatus>;
  },
): Promise<string> {
  if (command.action === "help") {
    return STATUS_HELP_RESPONSE;
  }
  const workspace = await api.getWorkspace(project.id);
  const goal = project.goal?.trim();
  const lines = [
    `Project: ${project.name}`,
    `Workspace: ${workspace.workspaceKind}, ${workspace.available ? "available" : "unavailable"}`,
    goal ? `Goal: ${project.goalStatus ?? "active"}, ${goal}` : "Goal: not set",
  ];
  if (!workspace.available) {
    lines.push(
      "Workspace error: reconnect the repository in Unsloth Desktop, then retry.",
    );
  } else if (workspace.capabilities.git) {
    lines.push(`Git: ${gitStatusSummary(await api.getGitStatus(project.id))}`);
  } else {
    lines.push("Git: not a repository");
  }
  return lines.join("\n");
}

export function reviewCommandResponse(review: AgentReviewSummary): string {
  const files = review.git?.files ?? [];
  const listed = files
    .slice(0, 20)
    .map((file) => `- ${file.code} ${file.path}`);
  if (files.length > listed.length) {
    listed.push(`- ${files.length - listed.length} more files omitted`);
  }
  const activePlans = review.plans.filter(
    (plan) => plan.status === "active",
  ).length;
  const latestRun = review.verification[0];
  return [
    review.git
      ? `Git: ${gitStatusSummary(review.git)}`
      : `Git: unavailable${review.gitError ? ` (${review.gitError})` : ""}`,
    `Goal: ${review.goal ? `${review.goalStatus ?? "active"}, ${review.goal}` : "not set"}`,
    `Plans: ${activePlans} active, ${review.plans.length} total`,
    latestRun
      ? `Verification: ${latestRun.status}, ${latestRun.stale ? "stale" : "fresh"}`
      : "Verification: no evidence",
    files.length > 0
      ? `Changed files:\n${listed.join("\n")}`
      : "Changed files: none",
    review.diff?.truncated
      ? "Diff: truncated to the review limit"
      : "Diff: bounded review captured",
  ].join("\n");
}

export async function executeReviewSlashCommand(
  command: ReviewSlashCommand,
  projectId: string,
  getReview: (projectId: string) => Promise<AgentReviewSummary>,
): Promise<string> {
  if (command.action === "help") {
    return REVIEW_HELP_RESPONSE;
  }
  return reviewCommandResponse(await getReview(projectId));
}

export async function* routeProjectSlashCommand<T>(
  input: string,
  handleCommand: (command: ProjectSlashCommand) => T | Promise<T>,
  runModel: () => AsyncIterable<T>,
): AsyncGenerator<T, void, unknown> {
  const command = parseProjectSlashCommand(input);
  if (command) {
    yield await handleCommand(command);
    return;
  }
  yield* runModel();
}
