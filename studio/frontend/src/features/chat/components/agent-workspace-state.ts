// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  AgentBackgroundRuntimeSnapshot,
  AgentBackgroundTask,
  AgentGitStatus,
  AgentPlan,
  AgentPreparedCommit,
  AgentPullRequestHandoffPreview,
  AgentRepositoryMap,
  AgentVerificationRun,
  AgentWorkspaceOverview,
  AgentWorktree,
} from "../api/agent-workspace-api";

export const BACKGROUND_AGENT_PERMISSION_POLICY =
  "Background agents run unattended. Ask for approval and Approve for me are unavailable because there is no interactive approval stream. Run automatically executes tools without prompts inside the project sandbox. Full access also skips prompts and relaxes ordinary sandboxing, while the project session remains bound to this workspace.";

export const BACKGROUND_AGENT_FULL_ACCESS_WARNING =
  "Full access lets this background agent run project tools without approval prompts and relaxes ordinary sandbox checks. Filesystem access remains bound to the selected project. Child commands remain subject to the project execution network policy, while the separate web search tool can still use its configured search service. Enable Full only if you trust this task.";

export function backgroundAgentPermissionRunsUnattended(mode: string): boolean {
  return mode === "off" || mode === "full";
}

export function backgroundAgentPermissionNeedsConfirmation(
  mode: string,
): boolean {
  return mode === "full";
}

export interface AgentBackgroundActions {
  canStart: boolean;
  canCancel: boolean;
  canRetry: boolean;
}

export interface AgentBackgroundSnapshot {
  readonly goal: string | null;
  readonly goalStatus: string | null;
  readonly goalUpdatedAt: number | null;
  readonly planTitle: string | null;
  readonly planRevision: number | null;
  readonly planTaskTitle: string | null;
  readonly worktreeId: string | null;
  readonly appExitPolicy: string;
  readonly runtime: Readonly<AgentBackgroundRuntimeSnapshot> | null;
}

export function agentBackgroundSnapshot(
  task: Pick<
    AgentBackgroundTask,
    | "goalSnapshot"
    | "goalStatusSnapshot"
    | "goalUpdatedAt"
    | "planRevision"
    | "planTaskId"
    | "planSnapshot"
    | "worktreeId"
    | "appExitPolicy"
    | "payload"
  >,
): Readonly<AgentBackgroundSnapshot> {
  const planTask = task.planSnapshot?.tasks.find(
    (candidate) => candidate.id === task.planTaskId,
  );
  const selectedRuntime = task.payload.runtime;
  const runtime = selectedRuntime
    ? Object.freeze({
        kind: selectedRuntime.kind,
        model: selectedRuntime.model,
        providerId: selectedRuntime.providerId ?? null,
        providerType: selectedRuntime.providerType ?? null,
        permissionMode: selectedRuntime.permissionMode,
        reasoningEffort: selectedRuntime.reasoningEffort ?? null,
        maxOutputTokens: selectedRuntime.maxOutputTokens,
      })
    : null;
  return Object.freeze({
    goal: task.goalSnapshot ?? null,
    goalStatus: task.goalStatusSnapshot ?? null,
    goalUpdatedAt: task.goalUpdatedAt ?? null,
    planTitle: task.planSnapshot?.title ?? null,
    planRevision: task.planRevision ?? null,
    planTaskTitle: planTask?.title ?? null,
    worktreeId: task.worktreeId ?? null,
    appExitPolicy: task.appExitPolicy || "interrupt",
    runtime,
  });
}

export function agentBackgroundActions(
  task: Pick<AgentBackgroundTask, "status">,
): AgentBackgroundActions {
  return {
    canStart: task.status === "queued",
    canCancel:
      task.status === "queued" ||
      task.status === "running" ||
      task.status === "cancelling",
    canRetry:
      task.status === "failed" ||
      task.status === "cancelled" ||
      task.status === "interrupted",
  };
}

export function agentWorktreeMergeAction(args: {
  worktree: Pick<AgentWorktree, "status">;
  gitStatus: Pick<AgentGitStatus, "head" | "clean"> | null;
  linkedTask: Pick<AgentBackgroundTask, "status"> | null;
}): { canMerge: boolean; reason: string | null } {
  if (args.worktree.status !== "active") {
    return { canMerge: false, reason: "Worktree is not active" };
  }
  if (
    args.linkedTask &&
    ["queued", "running", "cancelling"].includes(args.linkedTask.status)
  ) {
    return { canMerge: false, reason: "Wait for the linked task to stop" };
  }
  if (!args.gitStatus?.head) {
    return { canMerge: false, reason: "Refresh Git status before merging" };
  }
  if (!args.gitStatus.clean) {
    return {
      canMerge: false,
      reason: "Commit or stash primary workspace changes first",
    };
  }
  return { canMerge: true, reason: null };
}

export function reconcileAgentBackgroundMutation(args: {
  tasks: AgentBackgroundTask[];
  worktrees: AgentWorktree[];
  previousTaskId?: string;
  action: "enqueue" | "start" | "cancel" | "retry";
  updated: AgentBackgroundTask;
}): { tasks: AgentBackgroundTask[]; worktrees: AgentWorktree[] } {
  const tasks =
    args.action === "enqueue" || args.action === "retry"
      ? [args.updated, ...args.tasks]
      : args.tasks.map((task) =>
          task.id === (args.previousTaskId ?? args.updated.id)
            ? args.updated
            : task,
        );
  const relinksWorktree =
    (args.action === "enqueue" || args.action === "retry") &&
    Boolean(args.updated.worktreeId);
  const worktrees = relinksWorktree
    ? args.worktrees.map((worktree) =>
        worktree.id === args.updated.worktreeId
          ? { ...worktree, backgroundTaskId: args.updated.id }
          : worktree,
      )
    : args.worktrees;
  return { tasks, worktrees };
}

export function preparedCommitConfirmation(
  preview: AgentPreparedCommit,
): Readonly<{ preparationId: string; confirmationToken: string }> {
  if (!preview.confirmationToken) {
    throw new Error("Prepared commit confirmation is unavailable.");
  }
  return Object.freeze({
    preparationId: preview.id,
    confirmationToken: preview.confirmationToken,
  });
}

export function pullRequestHandoffConfirmation(
  preview: AgentPullRequestHandoffPreview,
): Readonly<{
  handoffId: string;
  serverId: string;
  confirmationToken: string;
  expectedRequestDigest: string;
}> {
  return Object.freeze({
    handoffId: preview.id,
    serverId: preview.connector.id,
    confirmationToken: preview.confirmationToken,
    expectedRequestDigest: preview.requestDigest,
  });
}

export interface AgentPullRequestSubmissionDisplay {
  readonly status: "submitting" | "submitted" | "unknown";
  readonly connectorName: string;
  readonly repository: string;
  readonly detail: string;
}

export function pullRequestSubmissionDisplay(
  preview: AgentPullRequestHandoffPreview,
  status: AgentPullRequestSubmissionDisplay["status"],
): Readonly<AgentPullRequestSubmissionDisplay> {
  const detail =
    status === "submitting"
      ? "Submission is in progress. Do not submit again."
      : status === "submitted"
        ? "GitHub confirmed the pull request submission."
        : "Submission outcome is unknown. Check GitHub before creating another handoff.";
  return Object.freeze({
    status,
    connectorName: preview.connector.displayName,
    repository: `${preview.request.owner}/${preview.request.repo}`,
    detail,
  });
}

export function pullRequestHandoffCanSubmit(
  preview: AgentPullRequestHandoffPreview | null,
  submission: AgentPullRequestSubmissionDisplay | null,
): boolean {
  return preview !== null && submission === null;
}

export function agentPlanProgress(plan: Pick<AgentPlan, "tasks">): {
  completed: number;
  total: number;
  percent: number;
} {
  const total = plan.tasks.length;
  const completed = plan.tasks.filter(
    (task) => task.status === "completed",
  ).length;
  return {
    completed,
    total,
    percent: total === 0 ? 0 : Math.round((completed / total) * 100),
  };
}

export function agentStatusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

export function agentRepositoryMapSummary(
  repositoryMap: AgentRepositoryMap | null,
): string {
  if (!repositoryMap) {
    return "Repository map not loaded";
  }
  const base = `${repositoryMap.fileCount.toLocaleString()} text files, ${repositoryMap.pathsScanned.toLocaleString()} paths scanned`;
  if (!repositoryMap.truncated) {
    return base;
  }
  const reasons = repositoryMap.truncationReasons.join(", ") || "bounded";
  return `${base}, partial (${reasons})`;
}

export function latestVerificationSummary(
  runs: AgentVerificationRun[],
): string {
  const latest = runs[0];
  if (!latest) {
    return "No verification evidence";
  }
  return `${agentStatusLabel(latest.status)}${latest.stale ? ", stale" : ", fresh"}`;
}

export function agentWorkspaceStatus(
  workspace: AgentWorkspaceOverview | null,
): { label: string; tone: "neutral" | "success" | "danger" } {
  if (!workspace) {
    return { label: "Checking workspace", tone: "neutral" };
  }
  if (!workspace.available) {
    return { label: "Workspace unavailable", tone: "danger" };
  }
  return {
    label: workspace.isGitRepository
      ? "Repository connected"
      : "Folder connected",
    tone: "success",
  };
}

export function agentWorkspaceRequestIsCurrent(args: {
  requestProjectId: string;
  activeProjectId: string;
  requestGeneration: number;
  activeGeneration: number;
}): boolean {
  return (
    args.requestProjectId === args.activeProjectId &&
    args.requestGeneration === args.activeGeneration
  );
}

export function safeAgentWorkspaceError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message
    .replace(/\b[A-Za-z]:[\\/][^\r\n,;)}\]]+/g, "<local path>")
    .replace(
      /(^|[\s("'`])\/(?:Users|home|private|var|tmp|Volumes|mnt|opt|srv|workspace)(?:\/[^\s"'`,;)}\]]+)+/g,
      "$1<local path>",
    );
}
